//! Atomic-write primitive (contract C4 §4): `tmp/<uuid>` -> fsync ->
//! optimistic concurrency check -> rename -> fsync parent (POSIX). Every
//! write this crate performs under `<data>` goes through
//! [`write_atomic`] — `session.json`, `data.parquet`, `derived/*.parquet`,
//! `tracks/*.idl0t`, and the catalog rebuild's file swap. Callers that need
//! C4 §4 step 4's bounded-3-attempts retry of the optimistic-concurrency
//! race (rather than surfacing the first conflict) use
//! [`write_atomic_with_retry`] instead.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Discriminant for [`AtomicWriteError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicWriteErrorKind {
    /// A filesystem operation failed (create/write/fsync/rename after
    /// retries exhausted).
    Io,
    /// `target` exists and its current hash does not match `based_on_hash`
    /// (C4 §4's race: an external write landed between the caller's read
    /// and this write). The caller must re-read `target`, re-derive its
    /// intended bytes against the current content, and retry — this
    /// primitive does not do so itself (see [`write_atomic`]'s doc).
    RenameConflict,
}

/// Error from [`write_atomic`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicWriteError {
    pub kind: AtomicWriteErrorKind,
    pub message: String,
}

impl AtomicWriteError {
    fn new(kind: AtomicWriteErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for AtomicWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AtomicWriteError {}

/// SHA-256 of `bytes`, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Atomically writes `bytes` to `target` (an absolute path somewhere under
/// `data_root`) per contract C4 §4's primitive. Returns the new content's
/// sha256 hex on success.
///
/// `based_on_hash`: the sha256 hex of `target`'s content when the caller
/// last read it, or `None` when the caller believes `target` does not yet
/// exist. If `target` exists at rename time and its current hash differs
/// from `based_on_hash` (`None` counts as "differs from anything"), this
/// returns `AtomicWriteErrorKind::RenameConflict` — the C4 §4 race between
/// read and write. This primitive performs **no retry of its own** for that
/// case (it has no domain knowledge of how to re-derive `bytes`) — it is the
/// single-attempt building block. [`write_atomic_with_retry`] is the C4
/// §4-conformant entry point for callers that need the bounded-3-attempts
/// retry loop; call this function directly only when a single attempt with
/// no retry is actually the desired behavior (e.g. tests, or a caller that
/// implements its own bespoke retry policy).
///
/// The rename step itself (distinct from the conflict check) is retried up
/// to 5 times with a ~50 ms backoff on I/O failure — covers a transient
/// Windows sharing violation (`target` momentarily held open without
/// `FILE_SHARE_DELETE`), per C4 §4.
pub fn write_atomic(
    data_root: &Path,
    target: &Path,
    bytes: &[u8],
    based_on_hash: Option<&str>,
) -> Result<String, AtomicWriteError> {
    let tmp_dir = data_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("create tmp dir: {e}")))?;

    let tmp_path: PathBuf = tmp_dir.join(Uuid::new_v4().to_string());
    write_and_fsync(&tmp_path, bytes)?;

    let new_hash = sha256_hex(bytes);

    // Optimistic concurrency check — read target's *current* bytes, not a
    // cached value, so this always reflects the instant right before rename.
    if target.exists() {
        let current = std::fs::read(target)
            .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("read {}: {e}", target.display())))?;
        let current_hash = sha256_hex(&current);
        if Some(current_hash.as_str()) != based_on_hash {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(AtomicWriteError::new(
                AtomicWriteErrorKind::RenameConflict,
                format!("{} changed since it was read (expected {based_on_hash:?}, found {current_hash})", target.display()),
            ));
        }
    } else if based_on_hash.is_some() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(AtomicWriteError::new(
            AtomicWriteErrorKind::RenameConflict,
            format!("{} does not exist but based_on_hash was Some(..)", target.display()),
        ));
    }

    rename_with_retry(&tmp_path, target)?;
    fsync_parent_dir_posix(target);

    Ok(new_hash)
}

/// Bound (at 3 attempts) number of times [`write_atomic`] is retried through
/// C4 §4 step 4's optimistic-concurrency race before surfacing an error.
const MAX_CONCURRENCY_RETRY_ATTEMPTS: u32 = 3;

/// [`write_atomic`], wrapped in C4 §4 step 4's mandated retry loop: "Abort
/// the rename, discard `tmp/<uuid>`, and re-run the write from the current
/// on-disk state ... Bound retries at 3 attempts before surfacing an error
/// rather than looping forever."
///
/// `bytes`/`based_on_hash` are the first attempt, exactly as for
/// [`write_atomic`]. On a `RenameConflict`, this re-reads `target`'s current
/// on-disk content and calls `rederive(current_bytes)` to get the next
/// attempt's bytes — `rederive` is where a caller supplies the file-class-
/// specific "re-run" logic C4 §4 describes: workbooks re-run the per-cell
/// merge (design §7) against `current_bytes` as the peer edit; every other
/// file class (`session.json`, `data.parquet`, `derived/*.parquet`,
/// `tracks/*.idl0t`, the catalog rebuild swap) re-applies the caller's
/// intended change on top of `current_bytes` instead of the stale base. This
/// primitive is itself file-class-agnostic; it only owns the attempt count,
/// the re-read, and the bounded loop.
///
/// After 3 attempts still conflict, returns the last attempt's
/// `AtomicWriteErrorKind::RenameConflict` rather than retrying forever. A
/// non-conflict error (e.g. `Io`) from any attempt is returned immediately,
/// without consuming a retry.
pub fn write_atomic_with_retry(
    data_root: &Path,
    target: &Path,
    bytes: &[u8],
    based_on_hash: Option<&str>,
    mut rederive: impl FnMut(&[u8]) -> Vec<u8>,
) -> Result<String, AtomicWriteError> {
    let mut attempt_bytes = bytes.to_vec();
    let mut attempt_based_on = based_on_hash.map(str::to_string);

    for attempt in 1..=MAX_CONCURRENCY_RETRY_ATTEMPTS {
        match write_atomic(data_root, target, &attempt_bytes, attempt_based_on.as_deref()) {
            Ok(hash) => return Ok(hash),
            Err(e) if e.kind == AtomicWriteErrorKind::RenameConflict && attempt < MAX_CONCURRENCY_RETRY_ATTEMPTS => {
                let current = std::fs::read(target).map_err(|io_e| {
                    AtomicWriteError::new(
                        AtomicWriteErrorKind::Io,
                        format!("re-read {} for concurrency retry: {io_e}", target.display()),
                    )
                })?;
                attempt_based_on = Some(sha256_hex(&current));
                attempt_bytes = rederive(&current);
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("every loop iteration returns: attempt < MAX_CONCURRENCY_RETRY_ATTEMPTS is always false on the final attempt, so the last iteration always hits the `Err(e) => return Err(e)` arm")
}

fn write_and_fsync(path: &Path, bytes: &[u8]) -> Result<(), AtomicWriteError> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("create {}: {e}", path.display())))?;
    f.write_all(bytes)
        .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("write {}: {e}", path.display())))?;
    f.sync_all()
        .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("fsync {}: {e}", path.display())))?;
    Ok(())
}

fn rename_with_retry(from: &Path, to: &Path) -> Result<(), AtomicWriteError> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AtomicWriteError::new(AtomicWriteErrorKind::Io, format!("create {}: {e}", parent.display())))?;
    }
    let mut last_err = None;
    for attempt in 0..5 {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = Some(e);
                if attempt < 4 {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
    Err(AtomicWriteError::new(
        AtomicWriteErrorKind::Io,
        format!("rename {} -> {} failed after 5 attempts: {}", from.display(), to.display(), last_err.unwrap()),
    ))
}

#[cfg(unix)]
fn fsync_parent_dir_posix(target: &Path) {
    if let Some(parent) = target.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

#[cfg(not(unix))]
fn fsync_parent_dir_posix(_target: &Path) {
    // NTFS commits the rename as a single MFT transaction; no directory
    // fsync exists or is needed on Windows (C4 §4).
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_atomic_creates_file_and_returns_its_hash() {
        // Arrange
        let root = temp_root();
        let target = root.join("sessions").join("s1").join("session.json");

        // Act
        let hash = write_atomic(&root, &target, b"{}", None).unwrap();

        // Assert
        assert_eq!(std::fs::read(&target).unwrap(), b"{}");
        assert_eq!(hash, sha256_hex(b"{}"));
        // tmp/ has no leftover staging file after a successful write.
        let leftovers: Vec<_> = std::fs::read_dir(root.join("tmp")).unwrap().collect();
        assert!(leftovers.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_atomic_second_write_with_correct_based_on_hash_succeeds() {
        // Arrange
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();

        // Act
        let h2 = write_atomic(&root, &target, b"v2", Some(&h1)).unwrap();

        // Assert
        assert_eq!(std::fs::read(&target).unwrap(), b"v2");
        assert_ne!(h1, h2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_atomic_stale_based_on_hash_is_a_rename_conflict_and_leaves_target_untouched() {
        // Arrange — target changes underneath the caller between read and write.
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();
        write_atomic(&root, &target, b"external-write", Some(&h1)).unwrap();

        // Act — caller still thinks the hash is h1.
        let err = write_atomic(&root, &target, b"my-write", Some(&h1)).unwrap_err();

        // Assert
        assert_eq!(err.kind, AtomicWriteErrorKind::RenameConflict);
        assert_eq!(std::fs::read(&target).unwrap(), b"external-write");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_atomic_none_based_on_hash_against_an_existing_file_is_a_conflict() {
        // Arrange
        let root = temp_root();
        let target = root.join("session.json");
        write_atomic(&root, &target, b"v1", None).unwrap();

        // Act — caller believes it's creating a new file, but one exists.
        let err = write_atomic(&root, &target, b"v2", None).unwrap_err();

        // Assert
        assert_eq!(err.kind, AtomicWriteErrorKind::RenameConflict);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_atomic_with_retry_rederives_once_and_succeeds_when_the_stale_read_conflicts_only_on_the_first_attempt() {
        // Arrange — caller's `based_on_hash` (h1) is already stale: an
        // external writer landed "external" on target before the caller's
        // first attempt even runs. `rederive` simulates re-applying the
        // caller's intended change on top of whatever it finds on disk.
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();
        write_atomic(&root, &target, b"external", Some(&h1)).unwrap();

        let rederive_calls = std::cell::Cell::new(0);

        // Act
        let hash = write_atomic_with_retry(&root, &target, b"my-write-on-v1", Some(&h1), |current| {
            rederive_calls.set(rederive_calls.get() + 1);
            let mut b = current.to_vec();
            b.extend_from_slice(b"+mine");
            b
        })
        .unwrap();

        // Assert — exactly one conflict, one rederive, then success.
        assert_eq!(rederive_calls.get(), 1);
        assert_eq!(std::fs::read(&target).unwrap(), b"external+mine");
        assert_eq!(hash, sha256_hex(b"external+mine"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_atomic_with_retry_exhausts_at_3_attempts_and_surfaces_rename_conflict_when_the_race_never_stops() {
        // Arrange — `rederive` itself keeps mutating target on every call, so
        // every retry's optimistic check is stale again by the time it runs
        // (a persistent, never-settling external writer). This is
        // deterministic (no real threads/timing), so not flaky.
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();
        // Stale from the very first attempt, same as the single-conflict test.
        write_atomic(&root, &target, b"external-0", Some(&h1)).unwrap();

        let rederive_calls = std::cell::Cell::new(0);

        // Act
        let err = write_atomic_with_retry(&root, &target, b"my-write", Some(&h1), |current| {
            let n = rederive_calls.get() + 1;
            rederive_calls.set(n);
            // Keep moving the target so the next attempt's based_on_hash is
            // stale again by the time write_atomic re-checks it.
            std::fs::write(&target, format!("external-{n}")).unwrap();
            let mut b = current.to_vec();
            b.extend_from_slice(b"+mine");
            b
        })
        .unwrap_err();

        // Assert — 3 attempts total means exactly 2 rederive calls (the
        // conflicting 1st and 2nd attempts each trigger one; the 3rd
        // attempt's failure is surfaced without a further rederive).
        assert_eq!(rederive_calls.get(), 2);
        assert_eq!(err.kind, AtomicWriteErrorKind::RenameConflict);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn write_atomic_retries_rename_through_a_transient_sharing_violation_and_succeeds_once_the_lock_releases() {
        use std::os::windows::fs::OpenOptionsExt;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // FILE_SHARE_READ only (no FILE_SHARE_DELETE) — this is exactly the
        // "editor without shared-delete semantics" case C4 §4 step 5 names:
        // MoveFileExW (Rust's rename, replacing an existing target) needs
        // delete-sharing on the destination, so this handle forces a real
        // ERROR_SHARING_VIOLATION rather than a simulated one.
        const FILE_SHARE_READ: u32 = 0x0000_0001;

        // Arrange
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();

        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&target)
            .expect("open target with a delete-excluding share mode");

        let released = Arc::new(AtomicBool::new(false));
        let released_writer = released.clone();
        // Well inside rename_with_retry's 5x50ms=250ms window, so the
        // in-flight rename retries must still be running when this drops
        // the lock — the retry, not luck, is what makes the write succeed.
        let holder = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            drop(lock);
            released_writer.store(true, Ordering::SeqCst);
        });

        // Act
        let result = write_atomic(&root, &target, b"v2", Some(&h1));

        // Assert
        holder.join().unwrap();
        assert!(released.load(Ordering::SeqCst), "lock must have been released before write_atomic returned");
        result.expect("rename must succeed once retried past the transient sharing violation");
        assert_eq!(std::fs::read(&target).unwrap(), b"v2");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn write_atomic_exhausts_rename_retries_and_surfaces_io_error_when_the_sharing_violation_outlasts_the_retry_window() {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_SHARE_READ: u32 = 0x0000_0001;

        // Arrange — hold the lock for the whole retry window (5 attempts x
        // ~50ms = ~250ms) plus margin, so every rename attempt fails.
        let root = temp_root();
        let target = root.join("session.json");
        let h1 = write_atomic(&root, &target, b"v1", None).unwrap();

        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&target)
            .expect("open target with a delete-excluding share mode");
        let holder = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            drop(lock);
        });

        // Act
        let err = write_atomic(&root, &target, b"v2", Some(&h1)).unwrap_err();

        // Assert — retries were exhausted, target left exactly as it was.
        assert_eq!(err.kind, AtomicWriteErrorKind::Io);
        assert_eq!(std::fs::read(&target).unwrap(), b"v1");

        holder.join().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }
}
