//! Folder scan for the bulk-import picker (contract C3 §3.3 `scan_folder`,
//! ruling R191). Lists the importable files sitting directly inside one
//! folder — non-recursive, no import run, no catalog write — so the app can
//! show a preview table and then enqueue `import_file` per chosen row.
//!
//! Two entry points, differing only in whether they answer
//! `already_imported` (ruling R201 item 2):
//!
//! - [`scan_folder`] never reads a file body — one `read_dir`, one `stat`
//!   per entry and, for `.idl0`, a [`HEADER_PEEK_BYTES`]-byte head read. It
//!   leaves `already_imported: None`, because deciding it means hashing
//!   whole files: on the 6.9 GB folder that produced R201 that was minutes
//!   of a frozen picker, and import de-duplicates by content hash anyway.
//!   This is what C3 §3.3's `scan_folder` command calls.
//! - [`scan_folder_with_blob_check`] is the old behaviour, hashing each
//!   file's bytes against `<data>/blobs`, for callers that are allowed to
//!   be slow and want the answer up front (the `idl-rs` CLI's `library
//!   scan`, which has no UI to freeze).

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
    /// `Some(true)` when `sha256(file bytes)` is already a blob under
    /// `<data>/blobs` (C4 §3), i.e. importing this file again would be a
    /// no-op; `Some(false)` when it is not, or when the file could not be
    /// read. Always `None` from [`scan_folder`], which never hashes —
    /// "not checked", not "not imported" (ruling R201 item 2).
    pub already_imported: Option<bool>,
    /// Session start from a header peek, UTC milliseconds
    /// ([`peek_session_start_ms`]; `0` = the header has no clock value).
    /// `None` when the format offers no peek — every importer except
    /// `.idl0`, and any `.idl0` whose header could not be read.
    pub session_start_utc_ms: Option<i64>,
}

/// Bytes read from the head of an `.idl0` file for the header peek. Far
/// more than [`peek_session_start_ms`] needs (35 bytes) and still one
/// sequential read, so the scan's cost per file is a `stat` plus one short
/// read regardless of how large the file is.
pub const HEADER_PEEK_BYTES: usize = 4096;

/// C3 §3.3 `scan_folder(path)` — every file directly inside `folder`
/// (non-recursive; sub-directories are skipped, not descended), ordered by
/// `file_name` for a deterministic preview. Returns from directory metadata
/// alone: no file body is read, nothing is hashed, and
/// [`ScanEntry::already_imported`] is always `None` (ruling R201 item 2 —
/// import de-duplicates by content hash regardless, so the picker does not
/// need the answer to be correct up front). The only read is the
/// [`HEADER_PEEK_BYTES`]-byte head of an `.idl0` file, for its start time.
///
/// Files this build has no importer for are still listed (with
/// `importer_id: None`) so the user can see why the folder's other contents
/// were not offered. A file whose head cannot be read (a permissions
/// failure, or a file deleted between the listing and the read) is listed
/// with no header peek rather than failing the whole scan.
pub fn scan_folder(folder: &Path) -> Result<Vec<ScanEntry>, ScanError> {
    scan_folder_inner(None, folder)
}

/// [`scan_folder`] plus the `already_imported` answer: every file's bytes
/// are read and sha256'd against `data_root`'s `blobs/` (C4 §3), so every
/// entry carries `Some(_)`. Costs one full read per file — minutes on a
/// folder of large logs — and so is for callers with no UI to freeze (the
/// CLI's `library scan`). The C3 §3.3 command calls [`scan_folder`]
/// instead (ruling R201 item 2).
pub fn scan_folder_with_blob_check(data_root: &Path, folder: &Path) -> Result<Vec<ScanEntry>, ScanError> {
    scan_folder_inner(Some(data_root), folder)
}

/// The shared body of [`scan_folder`] and [`scan_folder_with_blob_check`]:
/// `data_root` is `Some` exactly when the caller wants `already_imported`
/// decided, which is also what decides whether whole files are read.
fn scan_folder_inner(data_root: Option<&Path>, folder: &Path) -> Result<Vec<ScanEntry>, ScanError> {
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

        // With a `data_root` the whole file is read once and serves both
        // `already_imported` (the blob digest is over the file's full
        // bytes, C4 §3) and the header peek; without one, only the head is
        // read and `already_imported` stays unanswered.
        let (already_imported, head) = match data_root {
            Some(root) => {
                let bytes = std::fs::read(&path).ok();
                let already = bytes.as_ref().is_some_and(|b| blob_exists(root, &sha256_hex(b)));
                (Some(already), bytes)
            }
            None if importer_id.as_deref() == Some("idl0") => (None, read_head(&path)),
            None => (None, None),
        };
        let session_start_utc_ms = match (importer_id.as_deref(), head.as_ref()) {
            (Some("idl0"), Some(b)) => peek_session_start_ms(b),
            _ => None,
        };

        entries.push(ScanEntry { path, file_name, size_bytes, importer_id, already_imported, session_start_utc_ms });
    }

    entries.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    Ok(entries)
}

/// The first [`HEADER_PEEK_BYTES`] bytes of `path` (fewer for a shorter
/// file), or `None` when the file cannot be opened or read — a scan lists
/// such a file without a header peek rather than failing. Loops over
/// `read` because one call may return a short count on any platform.
fn read_head(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; HEADER_PEEK_BYTES];
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    buf.truncate(filled);
    Some(buf)
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
        let entries = scan_folder(&folder).unwrap();

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
        let entries = scan_folder(&folder).unwrap();

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
        let entries = scan_folder_with_blob_check(&data_root, &folder).unwrap();

        // Assert — same bytes → already imported; different bytes → not.
        assert_eq!(entries[0].already_imported, Some(true));
        assert_eq!(entries[1].already_imported, Some(false));

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_a_file_whose_bytes_are_already_a_blob_is_left_unchecked() {
        // Arrange — the same buffer imported first, so the hashing scan
        // would answer `Some(true)` for it.
        let data_root = temp_dir();
        let folder = temp_dir();
        let bytes = idl0_bytes(0);
        crate::store::import::import_idl0(&data_root, &bytes).unwrap();
        std::fs::write(folder.join("a.idl0"), &bytes).unwrap();

        // Act
        let entries = scan_folder(&folder).unwrap();

        // Assert — R201 item 2: never hashed, so never answered.
        assert_eq!(entries[0].already_imported, None);
        assert_eq!(entries[0].session_start_utc_ms, Some(0));

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_an_idl0_file_larger_than_the_peek_still_reports_its_header_start() {
        // Arrange — a file whose body runs well past HEADER_PEEK_BYTES, so
        // a passing peek proves the head read is enough on its own.
        let data_root = temp_dir();
        let folder = temp_dir();
        let mut bytes = idl0_bytes(1_700_000_000_000);
        bytes.extend(std::iter::repeat(0u8).take(HEADER_PEEK_BYTES * 4));
        std::fs::write(folder.join("a.idl0"), &bytes).unwrap();

        // Act
        let entries = scan_folder(&folder).unwrap();

        // Assert
        assert_eq!(entries[0].session_start_utc_ms, Some(1_700_000_000_000));
        assert_eq!(entries[0].size_bytes, bytes.len() as u64);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_a_path_that_is_not_a_directory_is_not_found() {
        // Arrange
        let data_root = temp_dir();

        // Act
        let result = scan_folder(&data_root.join("nope"));

        // Assert
        assert!(matches!(result, Err(ScanError { kind: ScanErrorKind::NotFound, .. })));

        let _ = std::fs::remove_dir_all(&data_root);
    }
}
