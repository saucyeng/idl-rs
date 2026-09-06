//! The `.sync-base` cache (C2 §7's `base` parameter to
//! [`crate::workbook::merge::merge`]): the last workbook document both this
//! device and a peer successfully synced, cached at
//! `<data>/workbooks/.sync-base/<workbook_id>.idl1wb` so the next sync has a
//! three-way merge base instead of only two documents to compare. Excluded
//! from the manifest walk and from `verify` (C4 §6, ruling R88) — this
//! module is the only reader/writer of that path.

use std::path::{Path, PathBuf};

use crate::store::atomic::{sha256_hex, write_atomic};
use crate::store::sync::manifest::{SyncError, SyncErrorKind};
use crate::workbook::v3::{parse_workbook, WorkbookDoc};

/// `<data_root>/workbooks/.sync-base/<workbook_id>.idl1wb` (C2 §7). Does not
/// itself validate `workbook_id` — callers that write through this path
/// (currently only [`write_base`]) are responsible for rejecting an id that
/// would escape `.sync-base/` once joined.
pub fn base_cache_path(data_root: &Path, workbook_id: &str) -> PathBuf {
    data_root.join("workbooks").join(".sync-base").join(format!("{workbook_id}.idl1wb"))
}

/// Rejects a `workbook_id` that would escape `.sync-base/` once joined with
/// `.idl1wb` — a path separator (`/` or `\`) or a `..` segment. The same
/// guard [`crate::track_artifact::write::write_track`] uses for `track_id`.
fn validate_workbook_id(workbook_id: &str) -> Result<(), SyncError> {
    if workbook_id.contains('/') || workbook_id.contains('\\') || workbook_id.contains("..") {
        return Err(SyncError {
            kind: SyncErrorKind::Io,
            message: format!("workbook_id {workbook_id:?} is not a valid file name component"),
        });
    }
    Ok(())
}

/// Reads and parses the cached base document for `workbook_id`. `Ok(None)`
/// both when no cache file exists yet (first-ever sync — the caller then
/// merges against the empty document, C2 §7) and when the cache file exists
/// but is not valid UTF-8 or fails [`parse_workbook`]'s front-matter-fatal
/// checks: a corrupt cache is never allowed to block a sync, so it is
/// treated exactly like an absent one rather than surfacing an error. A
/// document that parses with non-fatal [`crate::workbook::v3::WorkbookError`]s
/// (e.g. a duplicate cell id) is still `Some` — those are collected, not
/// fatal (see `parse_workbook`'s own doc comment).
pub fn read_base(data_root: &Path, workbook_id: &str) -> Result<Option<WorkbookDoc>, SyncError> {
    let path = base_cache_path(data_root, workbook_id);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(SyncError { kind: SyncErrorKind::Io, message: format!("read {}: {e}", path.display()) })
        }
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return Ok(None); // corrupt cache — treated as absent, never a hard failure
    };
    match parse_workbook(&text) {
        Ok((doc, _non_fatal_errors)) => Ok(Some(doc)),
        Err(_fatal_errors) => Ok(None), // corrupt cache — treated as absent
    }
}

/// Overwrites the cached base for `workbook_id` with `bytes` (the `.idl1wb`
/// source text that both sides just successfully synced), through
/// [`write_atomic`] (C4 §4) — creates `.sync-base/` if absent (the atomic
/// primitive's rename step creates its target's parent directory). Rejects
/// a `workbook_id` containing a path separator or a `..` segment before
/// writing anything.
pub fn write_base(data_root: &Path, workbook_id: &str, bytes: &[u8]) -> Result<(), SyncError> {
    validate_workbook_id(workbook_id)?;
    let path = base_cache_path(data_root, workbook_id);
    let based_on_hash = std::fs::read(&path).ok().map(|current| sha256_hex(&current));
    write_atomic(data_root, &path, bytes, based_on_hash.as_deref())
        .map_err(|e| SyncError { kind: SyncErrorKind::Io, message: e.to_string() })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_source() -> String {
        format!("---\nid: {ID}\nname: Fork tuning\n---\n```math id=aaaaaaaa\nx = 1\n```\n")
    }

    #[test]
    fn base_cache_path_joins_workbooks_dot_sync_dash_base_and_the_extension() {
        // Arrange
        let root = PathBuf::from("/data");

        // Act
        let path = base_cache_path(&root, "wb-1");

        // Assert
        assert_eq!(path, PathBuf::from("/data/workbooks/.sync-base/wb-1.idl1wb"));
    }

    #[test]
    fn read_base_no_cache_ok_none() {
        // Arrange
        let root = temp_root();

        // Act
        let result = read_base(&root, "wb-1").unwrap();

        // Assert
        assert!(result.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_base_a_corrupt_cache_ok_none_not_an_error() {
        // Arrange — no front-matter at all, so `parse_workbook` fails fatally.
        let root = temp_root();
        let path = base_cache_path(&root, "wb-1");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a workbook at all").unwrap();

        // Act
        let result = read_base(&root, "wb-1").unwrap();

        // Assert
        assert!(result.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_base_then_read_base_the_same_document() {
        // Arrange
        let root = temp_root();
        let source = sample_source();

        // Act
        write_base(&root, "wb-1", source.as_bytes()).unwrap();
        let back = read_base(&root, "wb-1").unwrap().unwrap();

        // Assert
        assert_eq!(back.id, ID);
        assert_eq!(back.cells.len(), 1);
        assert_eq!(back.cells[0].id, "aaaaaaaa");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_base_creates_sync_base_dir_when_absent() {
        // Arrange
        let root = temp_root();
        assert!(!root.join("workbooks").join(".sync-base").exists());

        // Act
        write_base(&root, "wb-1", sample_source().as_bytes()).unwrap();

        // Assert
        assert!(root.join("workbooks").join(".sync-base").join("wb-1.idl1wb").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_base_an_id_containing_a_path_separator_refused_nothing_written() {
        // Arrange
        let root = temp_root();

        // Act
        let result = write_base(&root, "../escape", sample_source().as_bytes());

        // Assert
        assert!(result.is_err());
        assert!(!root.join("workbooks").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
