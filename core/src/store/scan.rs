//! Folder scan for the bulk-import picker (contract C3 §3.3 `scan_folder`,
//! ruling R191). Lists the importable files sitting directly inside one
//! folder — non-recursive, no import run, no catalog write — so the app can
//! show a preview table and then enqueue `import_file` per chosen row.
//!
//! "Cheap" here means cheap in *decisions*, not in I/O: answering
//! `already_imported` hashes each file's bytes (R191 — a folder of large
//! files is allowed to be slow, because this runs once per picker use and
//! never on a timer).

use std::fmt;
use std::path::{Path, PathBuf};

use crate::import::importer_for_extension;
use crate::parse::peek_session_start_ms;
use crate::store::atomic::sha256_hex;
use crate::store::blob::blob_exists;

/// Discriminant for [`ScanError`] — the C3 §3.3 error kinds `scan_folder`
/// may return (`internal` is L5's, never raised here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanErrorKind {
    /// The folder does not exist, or exists but is not a directory.
    NotFound,
    /// Reading the directory failed.
    Io,
}

/// Error from [`scan_folder`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanError {
    pub kind: ScanErrorKind,
    pub message: String,
}

impl ScanError {
    fn new(kind: ScanErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ScanError {}

/// One file found directly inside the scanned folder (C3 §3.3 `ScanEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanEntry {
    /// Absolute path of the file, the value the app hands back to
    /// `import_file`.
    pub path: PathBuf,
    /// The file's own name, including extension.
    pub file_name: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Importer id for this file's extension (`"idl0"`/`"fit"`/`"gpx"`/
    /// `"csv"`), or `None` when no importer covers it — such a row is listed
    /// but not importable.
    pub importer_id: Option<String>,
    /// `true` when `sha256(file bytes)` is already a blob under
    /// `<data>/blobs` (C4 §3), i.e. importing this file again would be a
    /// no-op. `false` for a file that could not be read.
    pub already_imported: bool,
    /// Session start from a header peek, UTC milliseconds
    /// ([`peek_session_start_ms`]; `0` = the header has no clock value).
    /// `None` when the format offers no peek — every importer except
    /// `.idl0`, and any `.idl0` whose header could not be read.
    pub session_start_utc_ms: Option<i64>,
}

/// C3 §3.3 `scan_folder(path)` — every file directly inside `folder`
/// (non-recursive; sub-directories are skipped, not descended), ordered by
/// `file_name` for a deterministic preview. `data_root` is the C4 data
/// directory whose `blobs/` decides [`ScanEntry::already_imported`].
///
/// Files this build has no importer for are still listed (with
/// `importer_id: None`) so the user can see why the folder's other contents
/// were not offered. A file that cannot be read at all (a permissions
/// failure, or a file deleted between the listing and the hash) is listed
/// with `already_imported: false` and no header peek rather than failing the
/// whole scan.
pub fn scan_folder(data_root: &Path, folder: &Path) -> Result<Vec<ScanEntry>, ScanError> {
    if !folder.is_dir() {
        return Err(ScanError::new(
            ScanErrorKind::NotFound,
            format!("{} is not a directory", folder.display()),
        ));
    }

    let dir = std::fs::read_dir(folder)
        .map_err(|e| ScanError::new(ScanErrorKind::Io, format!("reading {}: {e}", folder.display())))?;

    let mut entries: Vec<ScanEntry> = Vec::new();
    for item in dir {
        let item =
            item.map_err(|e| ScanError::new(ScanErrorKind::Io, format!("reading {}: {e}", folder.display())))?;
        let path = item.path();
        if !path.is_file() {
            continue;
        }
        let file_name = item.file_name().to_string_lossy().into_owned();
        let size_bytes = item.metadata().map(|m| m.len()).unwrap_or(0);
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let importer_id = importer_id_for_extension(&ext);

        // One read of the whole file: it answers `already_imported` (the
        // blob digest is over the file's full bytes, C4 §3) and, for
        // `.idl0`, the header peek as well.
        let bytes = std::fs::read(&path).ok();
        let already_imported = bytes.as_ref().is_some_and(|b| blob_exists(data_root, &sha256_hex(b)));
        let session_start_utc_ms = match (importer_id.as_deref(), bytes.as_ref()) {
            (Some("idl0"), Some(b)) => peek_session_start_ms(b),
            _ => None,
        };

        entries.push(ScanEntry { path, file_name, size_bytes, importer_id, already_imported, session_start_utc_ms });
    }

    entries.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    Ok(entries)
}

/// Extension (lowercase, no dot) → importer id, unifying `.idl0`'s own entry
/// point with the importer table's extensions exactly as `list_importers`
/// does (`.idl0` is deliberately outside that table —
/// [`crate::import::ImporterInfo`]).
fn importer_id_for_extension(ext: &str) -> Option<String> {
    if ext == "idl0" {
        return Some("idl0".to_string());
    }
    importer_for_extension(ext).map(|i| i.source_format().as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::test_buffers::*;
    use uuid::Uuid;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A minimal valid `.idl0` v3 buffer whose header carries
    /// `session_start_ms`.
    fn idl0_bytes(session_start_ms: i64) -> Vec<u8> {
        let accel: f32 = 32.0 / 32768.0;
        cat(&[
            Header { schema_version: 3, session_start_ms, imu_mask: 0x01, ..Default::default() }
                .build(&[v3_registry_entry(0, 4, 800, accel, 0.0, "IMU0_AccelX", "g")]),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ])
    }

    #[test]
    fn scan_folder_a_folder_of_mixed_files_maps_extensions_to_importer_ids_and_skips_sub_directories() {
        // Arrange
        let data_root = temp_dir();
        let folder = temp_dir();
        std::fs::write(folder.join("a.idl0"), idl0_bytes(0)).unwrap();
        std::fs::write(folder.join("b.gpx"), b"<gpx/>").unwrap();
        std::fs::write(folder.join("c.txt"), b"notes").unwrap();
        std::fs::create_dir_all(folder.join("d_subdir")).unwrap();
        std::fs::write(folder.join("d_subdir").join("e.idl0"), idl0_bytes(0)).unwrap();

        // Act
        let entries = scan_folder(&data_root, &folder).unwrap();

        // Assert — three files, alphabetical, the sub-directory's contents
        // never descended into.
        let seen: Vec<(&str, Option<&str>)> =
            entries.iter().map(|e| (e.file_name.as_str(), e.importer_id.as_deref())).collect();
        assert_eq!(seen, vec![("a.idl0", Some("idl0")), ("b.gpx", Some("gpx")), ("c.txt", None)]);
        assert_eq!(entries[1].size_bytes, 6);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_an_idl0_file_reports_its_header_start_and_a_gpx_file_reports_none() {
        // Arrange
        let data_root = temp_dir();
        let folder = temp_dir();
        std::fs::write(folder.join("a.idl0"), idl0_bytes(1_700_000_000_000)).unwrap();
        std::fs::write(folder.join("b.gpx"), b"<gpx/>").unwrap();

        // Act
        let entries = scan_folder(&data_root, &folder).unwrap();

        // Assert
        assert_eq!(entries[0].session_start_utc_ms, Some(1_700_000_000_000));
        assert_eq!(entries[1].session_start_utc_ms, None);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_a_file_whose_bytes_are_already_a_blob_is_marked_already_imported() {
        // Arrange — import the same buffer first, so its blob is in the CAS.
        let data_root = temp_dir();
        let folder = temp_dir();
        let bytes = idl0_bytes(0);
        crate::store::import::import_idl0(&data_root, &bytes).unwrap();
        std::fs::write(folder.join("a.idl0"), &bytes).unwrap();
        std::fs::write(folder.join("b.idl0"), idl0_bytes(1_700_000_000_000)).unwrap();

        // Act
        let entries = scan_folder(&data_root, &folder).unwrap();

        // Assert — same bytes → already imported; different bytes → not.
        assert!(entries[0].already_imported);
        assert!(!entries[1].already_imported);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_a_path_that_is_not_a_directory_is_not_found() {
        // Arrange
        let data_root = temp_dir();

        // Act
        let result = scan_folder(&data_root, &data_root.join("nope"));

        // Assert
        assert!(matches!(result, Err(ScanError { kind: ScanErrorKind::NotFound, .. })));

        let _ = std::fs::remove_dir_all(&data_root);
    }
}
