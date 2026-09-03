//! Atomic-write primitive (contract C4 §4): `tmp/<uuid>` -> fsync ->
//! optimistic concurrency check -> rename -> fsync parent (POSIX). Every
//! write this crate performs under `<data>` goes through
//! [`write_atomic`] — `session.json`, `data.parquet`, `derived/*.parquet`,
//! `tracks/*.idl0t`, and the catalog rebuild's file swap.

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
/// case (it has no domain knowledge of how to re-derive `bytes`); the
/// caller's own retry loop (bounded at 3 attempts, C4 §4) re-reads the
/// current file, re-derives its intended bytes, and calls this function
/// again with the new `based_on_hash`.
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
}
