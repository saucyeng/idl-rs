//! Quarantine store for corrupt content-addressed files pulled out of
//! `<data>` by `verify`'s repair pass (C4 §7's repair actions, ruling R86).
//!
//! A quarantined file lives under `tmp/quarantine/` as two files: the
//! payload `tmp/quarantine/<entry_id>-<original file name>` and a sidecar
//! `tmp/quarantine/<entry_id>.json` (C4 §2's 2026-09-06 post-sign
//! amendment) holding everything the payload's own bytes cannot carry —
//! where it came from and why it was pulled. Nothing under `tmp/` is ever
//! read as catalog truth (C4 §2), so no function here touches the catalog.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::store::atomic::write_atomic;

/// `entry_id`s are C3 §3.2 uuids in their canonical 36-char
/// lowercase-with-dashes form (`<uuid>-<original file name>`,
/// `<uuid>.json`) — fixed length, not split on the first `-`, because a
/// uuid itself contains dashes.
const ENTRY_ID_LEN: usize = 36;

/// Discriminant for [`QuarantineError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineErrorKind {
    /// A filesystem operation failed, or an `entry_id` failed the
    /// path-safety check (a separator or `..` could escape
    /// `tmp/quarantine/` when joined into a path).
    Io,
    /// The named entry does not exist (unknown `entry_id`, or nowhere to
    /// restore an entry with no recorded `original_path`).
    NotFound,
    /// A `Restore`'s destination already exists — never overwritten, so
    /// both copies of the file survive.
    Occupied,
    /// The sidecar JSON could not be encoded or decoded.
    Encode,
}

/// Error from the quarantine store. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineError {
    pub kind: QuarantineErrorKind,
    pub message: String,
}

impl QuarantineError {
    fn new(kind: QuarantineErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for QuarantineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for QuarantineError {}

/// One quarantined file: the payload plus its sidecar (C4 §2, §7).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QuarantineEntry {
    /// The uuid in both filenames — C3 §3.2's `entry_id`.
    pub entry_id: String,
    /// Absolute path of the payload under `<data>/tmp/quarantine/`.
    pub path: String,
    /// Where it was pulled from, absolute. `""` when unknown (no sidecar).
    pub original_path: String,
    /// The C4 §7 finding text that caused the move.
    pub reason: String,
    /// Milliseconds since the Unix epoch, supplied by the caller.
    pub quarantined_at_ms: i64,
}

/// What [`resolve_quarantine`] does with a quarantined entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveAction {
    /// Move the payload back to `original_path`.
    Restore,
    /// Delete the payload.
    Discard,
}

/// The sidecar's on-disk shape (C4 §2's 2026-09-06 amendment). `path` is
/// never stored in it — it is a filesystem fact ([`quarantine_dir`] plus
/// the payload's own filename), not duplicated data that could drift.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SidecarDoc {
    entry_id: String,
    original_path: String,
    reason: String,
    quarantined_at_ms: i64,
}

fn quarantine_dir(data_root: &Path) -> PathBuf {
    data_root.join("tmp").join("quarantine")
}

fn sidecar_path(data_root: &Path, entry_id: &str) -> PathBuf {
    quarantine_dir(data_root).join(format!("{entry_id}.json"))
}

/// Rejects an `entry_id` that could escape `tmp/quarantine/` when joined
/// into a path: empty, containing a path separator, or containing `..`.
fn validate_entry_id(entry_id: &str) -> Result<(), QuarantineError> {
    if entry_id.is_empty() || entry_id.contains('/') || entry_id.contains('\\') || entry_id.contains("..") {
        return Err(QuarantineError::new(
            QuarantineErrorKind::Io,
            format!("entry_id {entry_id:?} is not a safe path component"),
        ));
    }
    Ok(())
}

/// Moves `path` to `tmp/quarantine/<entry_id>-<file name>` and writes the
/// sidecar `tmp/quarantine/<entry_id>.json`. `entry_id` and
/// `quarantined_at_ms` are supplied — this function has no clock and no
/// randomness, so it is deterministic under test (PLAN §3).
///
/// The sidecar is written *before* the payload is moved, through the
/// landed atomic-write primitive, so a crash never leaves a half-written
/// sidecar. If the process dies between the sidecar write and the move,
/// the tree is left with a sidecar and no matching payload —
/// [`list_quarantine`] treats that shape as a half-finished resolve and
/// skips it, and the original file is still sitting untouched at `path`
/// (the move never ran), so nothing is lost: the next `verify_and_repair`
/// simply quarantines it again under a fresh `entry_id`, leaving the
/// stale orphan sidecar to be cleaned up the same way a half-finished
/// resolve is. Ordering it the other way (move first, sidecar second)
/// would instead risk a bare payload silently losing its `original_path`
/// and `reason` forever if the crash landed between the two writes —
/// worse, because a plain "unknown (no sidecar)" listing is the best this
/// module could ever recover, whereas the sidecar-first order can always
/// retry the whole thing from scratch.
pub fn quarantine_file(
    data_root: &Path,
    path: &Path,
    reason: &str,
    entry_id: &str,
    quarantined_at_ms: i64,
) -> Result<QuarantineEntry, QuarantineError> {
    validate_entry_id(entry_id)?;

    let dir = quarantine_dir(data_root);
    std::fs::create_dir_all(&dir)
        .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("create {}: {e}", dir.display())))?;

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| QuarantineError::new(QuarantineErrorKind::Io, format!("{} has no file name", path.display())))?;
    let payload_path = dir.join(format!("{entry_id}-{file_name}"));

    let original_path = path.to_string_lossy().into_owned();
    let sidecar = SidecarDoc {
        entry_id: entry_id.to_string(),
        original_path: original_path.clone(),
        reason: reason.to_string(),
        quarantined_at_ms,
    };
    let bytes =
        serde_json::to_vec(&sidecar).map_err(|e| QuarantineError::new(QuarantineErrorKind::Encode, e.to_string()))?;
    write_atomic(data_root, &sidecar_path(data_root, entry_id), &bytes, None)
        .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, e.to_string()))?;

    move_file(path, &payload_path)?;

    Ok(QuarantineEntry {
        entry_id: entry_id.to_string(),
        path: payload_path.to_string_lossy().into_owned(),
        original_path,
        reason: reason.to_string(),
        quarantined_at_ms,
    })
}

/// Every entry currently in `tmp/quarantine/`, sorted by
/// `quarantined_at_ms` descending then `entry_id`.
///
/// Listing is payload-driven: every file matching `<uuid>-<rest>` is an
/// entry (its sidecar `<uuid>.json` is read if present); a `.json` with no
/// matching payload is a half-finished resolve, not reported here.
pub fn list_quarantine(data_root: &Path) -> Result<Vec<QuarantineEntry>, QuarantineError> {
    let dir = quarantine_dir(data_root);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(QuarantineError::new(QuarantineErrorKind::Io, format!("read {}: {e}", dir.display()))),
    };

    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, e.to_string()))?;
        let file_name = entry.file_name().to_string_lossy().into_owned();
        if file_name.len() <= ENTRY_ID_LEN || file_name.as_bytes()[ENTRY_ID_LEN] != b'-' {
            continue; // not a `<uuid>-<rest>` payload — sidecars and stray files alike
        }
        let entry_id = &file_name[..ENTRY_ID_LEN];
        let payload_path = entry.path();

        let sc_path = sidecar_path(data_root, entry_id);
        let (original_path, reason, quarantined_at_ms) = match std::fs::read(&sc_path) {
            Ok(bytes) => {
                let doc: SidecarDoc = serde_json::from_slice(&bytes)
                    .map_err(|e| QuarantineError::new(QuarantineErrorKind::Encode, e.to_string()))?;
                (doc.original_path, doc.reason, doc.quarantined_at_ms)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                (String::new(), "unknown (no sidecar)".to_string(), payload_mtime_ms(&payload_path)?)
            }
            Err(e) => {
                return Err(QuarantineError::new(QuarantineErrorKind::Io, format!("read {}: {e}", sc_path.display())))
            }
        };

        out.push(QuarantineEntry {
            entry_id: entry_id.to_string(),
            path: payload_path.to_string_lossy().into_owned(),
            original_path,
            reason,
            quarantined_at_ms,
        });
    }

    out.sort_by(|a, b| b.quarantined_at_ms.cmp(&a.quarantined_at_ms).then_with(|| a.entry_id.cmp(&b.entry_id)));
    Ok(out)
}

fn payload_mtime_ms(path: &Path) -> Result<i64, QuarantineError> {
    let meta = std::fs::metadata(path)
        .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("stat {}: {e}", path.display())))?;
    let modified = meta
        .modified()
        .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("mtime {}: {e}", path.display())))?;
    // A pre-1970 mtime is not a real case on any platform this app targets.
    Ok(modified.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0))
}

/// `Restore` moves the payload back to `original_path`; `Discard` deletes
/// it. Both then delete the sidecar. Nothing under `tmp/` is catalog truth
/// (C4 §2), so neither touches the catalog.
pub fn resolve_quarantine(data_root: &Path, entry_id: &str, action: ResolveAction) -> Result<(), QuarantineError> {
    validate_entry_id(entry_id)?;

    let entry = list_quarantine(data_root)?
        .into_iter()
        .find(|e| e.entry_id == entry_id)
        .ok_or_else(|| QuarantineError::new(QuarantineErrorKind::NotFound, format!("no quarantine entry {entry_id}")))?;
    let payload_path = PathBuf::from(&entry.path);

    match action {
        ResolveAction::Restore => {
            if entry.original_path.is_empty() {
                return Err(QuarantineError::new(
                    QuarantineErrorKind::NotFound,
                    "no original_path recorded — nowhere to restore to",
                ));
            }
            let dest = PathBuf::from(&entry.original_path);
            if dest.exists() {
                return Err(QuarantineError::new(QuarantineErrorKind::Occupied, format!("{} already exists", dest.display())));
            }
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("create {}: {e}", parent.display())))?;
            }
            move_file(&payload_path, &dest)?;
        }
        ResolveAction::Discard => {
            std::fs::remove_file(&payload_path)
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("remove {}: {e}", payload_path.display())))?;
        }
    }

    // An entry with no sidecar (the "unknown (no sidecar)" shape) has
    // nothing to remove here — that is not an error.
    let _ = std::fs::remove_file(sidecar_path(data_root, entry_id));

    Ok(())
}

/// Moves `from` to `to` via `rename` when possible; falls back to a copy +
/// fsync + remove when `rename` fails because the two paths are on
/// different filesystem devices (`<data>` mounted or bind-mounted such
/// that `tmp/` and the source aren't on the same volume) — `rename` cannot
/// do that atomically, so this is the best available substitute, not an
/// atomic guarantee, and is expected to be rare (C4 §4's atomic-write
/// primitive relies on the same same-volume assumption for `tmp/<uuid>`).
fn move_file(from: &Path, to: &Path) -> Result<(), QuarantineError> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::CrossesDevices => {
            let bytes = std::fs::read(from)
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("read {}: {e}", from.display())))?;
            let mut f = std::fs::File::create(to)
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("create {}: {e}", to.display())))?;
            f.write_all(&bytes)
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("write {}: {e}", to.display())))?;
            f.sync_all()
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("fsync {}: {e}", to.display())))?;
            std::fs::remove_file(from)
                .map_err(|e| QuarantineError::new(QuarantineErrorKind::Io, format!("remove {}: {e}", from.display())))?;
            Ok(())
        }
        Err(e) => Err(QuarantineError::new(
            QuarantineErrorKind::Io,
            format!("move {} -> {}: {e}", from.display(), to.display()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn uuid_string() -> String {
        Uuid::new_v4().to_string()
    }

    #[test]
    fn quarantine_file_a_corrupt_blob_the_payload_moves_the_sidecar_holds_the_reason_and_original_path_and_the_source_is_gone(
    ) {
        // Arrange
        let root = temp_root();
        let blob_path = root.join("blobs").join("sha256").join("ab").join("c".repeat(62));
        std::fs::create_dir_all(blob_path.parent().unwrap()).unwrap();
        std::fs::write(&blob_path, b"corrupted bytes").unwrap();
        let entry_id = uuid_string();

        // Act
        let entry = quarantine_file(&root, &blob_path, "hash mismatch", &entry_id, 1_000).unwrap();

        // Assert
        assert!(!blob_path.exists());
        assert!(PathBuf::from(&entry.path).exists());
        assert_eq!(std::fs::read(&entry.path).unwrap(), b"corrupted bytes");
        assert_eq!(entry.original_path, blob_path.to_string_lossy());
        assert_eq!(entry.reason, "hash mismatch");
        assert_eq!(entry.quarantined_at_ms, 1_000);

        let sidecar_bytes = std::fs::read(sidecar_path(&root, &entry_id)).unwrap();
        let sidecar: SidecarDoc = serde_json::from_slice(&sidecar_bytes).unwrap();
        assert_eq!(sidecar.original_path, blob_path.to_string_lossy());
        assert_eq!(sidecar.reason, "hash mismatch");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn quarantine_file_quarantine_dir_absent_it_is_created() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("d".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"x").unwrap();
        assert!(!quarantine_dir(&root).exists());

        // Act
        quarantine_file(&root, &source, "hash mismatch", &uuid_string(), 1).unwrap();

        // Assert
        assert!(quarantine_dir(&root).is_dir());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_quarantine_an_empty_or_absent_directory_an_empty_list_not_an_error() {
        // Arrange
        let root = temp_root();

        // Act
        let absent = list_quarantine(&root).unwrap();
        std::fs::create_dir_all(quarantine_dir(&root)).unwrap();
        let empty = list_quarantine(&root).unwrap();

        // Assert
        assert!(absent.is_empty());
        assert!(empty.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_quarantine_a_payload_with_no_sidecar_listed_with_the_unknown_reason_and_no_original_path() {
        // Arrange
        let root = temp_root();
        let entry_id = uuid_string();
        let dir = quarantine_dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{entry_id}-orphan.bin")), b"x").unwrap();

        // Act
        let entries = list_quarantine(&root).unwrap();

        // Assert
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_id, entry_id);
        assert_eq!(entries[0].original_path, "");
        assert_eq!(entries[0].reason, "unknown (no sidecar)");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_quarantine_a_sidecar_with_no_payload_skipped() {
        // Arrange
        let root = temp_root();
        let entry_id = uuid_string();
        let dir = quarantine_dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(sidecar_path(&root, &entry_id), b"{}").unwrap();

        // Act
        let entries = list_quarantine(&root).unwrap();

        // Assert
        assert!(entries.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_quarantine_restore_the_file_is_back_at_original_path_and_both_quarantine_files_are_gone() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("e".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"payload bytes").unwrap();
        let entry_id = uuid_string();
        quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();

        // Act
        resolve_quarantine(&root, &entry_id, ResolveAction::Restore).unwrap();

        // Assert
        assert_eq!(std::fs::read(&source).unwrap(), b"payload bytes");
        assert!(!sidecar_path(&root, &entry_id).exists());
        assert!(list_quarantine(&root).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_quarantine_restore_onto_an_occupied_path_occupied_and_both_copies_still_exist() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("f".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"original").unwrap();
        let entry_id = uuid_string();
        let entry = quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();
        // Something now occupies the original location (e.g. a re-import).
        std::fs::write(&source, b"a new file already lives here").unwrap();

        // Act
        let err = resolve_quarantine(&root, &entry_id, ResolveAction::Restore).unwrap_err();

        // Assert
        assert_eq!(err.kind, QuarantineErrorKind::Occupied);
        assert_eq!(std::fs::read(&source).unwrap(), b"a new file already lives here");
        assert_eq!(std::fs::read(&entry.path).unwrap(), b"original");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_quarantine_discard_payload_and_sidecar_are_gone() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("1".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"discard me").unwrap();
        let entry_id = uuid_string();
        let entry = quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();

        // Act
        resolve_quarantine(&root, &entry_id, ResolveAction::Discard).unwrap();

        // Assert
        assert!(!PathBuf::from(&entry.path).exists());
        assert!(!sidecar_path(&root, &entry_id).exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_quarantine_unknown_entry_id_not_found() {
        // Arrange
        let root = temp_root();
        std::fs::create_dir_all(quarantine_dir(&root)).unwrap();

        // Act
        let err = resolve_quarantine(&root, &uuid_string(), ResolveAction::Discard).unwrap_err();

        // Assert
        assert_eq!(err.kind, QuarantineErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_quarantine_entry_id_containing_a_path_separator_not_found_or_io_and_a_decoy_survives() {
        // Arrange — a file outside tmp/ that a path-traversal entry_id
        // could reach if the guard were missing.
        let root = temp_root();
        let decoy = root.join("blobs").join("sha256").join("ab").join("2".repeat(62));
        std::fs::create_dir_all(decoy.parent().unwrap()).unwrap();
        std::fs::write(&decoy, b"do not touch me").unwrap();
        let malicious = "../blobs/sha256/ab/22222222222222222222222222222222222222222222222222222222222222";

        // Act
        let err = resolve_quarantine(&root, malicious, ResolveAction::Discard).unwrap_err();

        // Assert
        assert!(matches!(err.kind, QuarantineErrorKind::NotFound | QuarantineErrorKind::Io));
        assert_eq!(std::fs::read(&decoy).unwrap(), b"do not touch me");

        let _ = std::fs::remove_dir_all(&root);
    }
}
