//! Library commands (C3 §3.3, added 2026-09-10, ruling R191):
//! `set_session_start`, `scan_folder`, `list_stale_sessions`,
//! `reimport_sessions`, `inbox_status`. None runs on the interaction path.
//!
//! Same `_via`-function idiom as every other module in this directory: each
//! `#[tauri::command]` is a thin wrapper over a plain function this module's
//! own tests exercise directly (`tauri::State`/`tauri::ipc::Channel` cannot
//! be constructed outside a running app).

use std::path::Path;

use crate::commands::catalog::SessionDetail;
use crate::commands::device::Progress;
use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// C3 §3.3 `ScanEntry` — one file `scan_folder` found. Mirrors
/// `idl_rs::store::scan::ScanEntry` with `path` as a wire string.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanEntry {
    /// Absolute path, the value the app hands back to `import_file`.
    pub path: String,
    /// The file's own name, including extension.
    pub file_name: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Importer id by extension (`list_importers`' vocabulary), `null` when
    /// no importer covers this file.
    pub importer_id: Option<String>,
    /// `true` when this file's sha256 is already a blob under
    /// `<data>/blobs` (C4 §3) — re-importing it would be a no-op.
    pub already_imported: bool,
    /// Session start from a header peek, UTC milliseconds (`0` = the header
    /// has no clock value); `null` when the format offers no peek.
    pub session_start_utc_ms: Option<i64>,
}

impl From<idl_rs::store::scan::ScanEntry> for ScanEntry {
    fn from(e: idl_rs::store::scan::ScanEntry) -> Self {
        Self {
            path: e.path.to_string_lossy().into_owned(),
            file_name: e.file_name,
            size_bytes: e.size_bytes,
            importer_id: e.importer_id,
            already_imported: e.already_imported,
            session_start_utc_ms: e.session_start_utc_ms,
        }
    }
}

/// C3 §3.3 `StaleSession` — one session whose `data.parquet` was written by
/// a different importer version than this build runs. Versions are SemVer
/// strings (ruling R194 item 3).
#[derive(Debug, Clone, serde::Serialize)]
pub struct StaleSession {
    pub session_id: String,
    pub importer_id: String,
    /// The version stamped on the session's `data.parquet`, e.g. `"0.1.0"`.
    pub stored_version: String,
    /// The running build's constant for `importer_id`.
    pub current_version: String,
}

impl From<idl_rs::store::catalog_read::StaleSession> for StaleSession {
    fn from(s: idl_rs::store::catalog_read::StaleSession) -> Self {
        Self {
            session_id: s.session_id,
            importer_id: s.importer_id,
            stored_version: s.stored_version,
            current_version: s.current_version,
        }
    }
}

/// One session `reimport_sessions` could not rebuild (C3 §3.3).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReimportFailure {
    pub session_id: String,
    pub error: IpcError,
}

/// C3 §3.3 `ReimportReport`. A session that fails stays on its old
/// `data.parquet` (atomic replace, C4 §4) — the whole command still
/// succeeds, with the per-session error in `failed`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReimportReport {
    pub rebuilt: Vec<String>,
    pub failed: Vec<ReimportFailure>,
}

/// Transport-agnostic core of `set_session_start` (C3 §3.3). Writes the
/// user-supplied start into `session.json` with `timestamp_source: "user"`
/// (C1 §3.1/§6, ruling R194) via core, then re-indexes this session's
/// catalog row so listing and sorting agree with the file (R194 item 3 —
/// the catalog's populate path already prefers `session.json` when the
/// source is `"user"`), and returns the re-read `SessionDetail`, the shape
/// `save_session_metadata` returns for the same reason.
///
/// The catalog re-index is best-effort in exactly one way: a `<data>` with
/// no `catalog.sqlite` yet is left alone rather than having one created
/// here (the catalog is "deletable, rebuildable, never synced" — creating
/// it is `rebuild_catalog`'s job). A catalog that exists but rejects the
/// update is a real failure and surfaces.
fn set_session_start_via(data_dir: &Path, session_id: &str, timestamp_utc_ms: i64) -> Result<SessionDetail, IpcError> {
    let session_dir = data_dir.join("sessions").join(session_id);
    if !session_dir.is_dir() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session {session_id} not found")));
    }

    idl_rs::store::session_json::set_session_start(data_dir, session_id, timestamp_utc_ms)
        .map_err(map_session_json_error)?;

    let catalog_path = data_dir.join("catalog.sqlite");
    if catalog_path.is_file() {
        let conn = idl_rs::store::catalog::open_catalog(&catalog_path)?;
        idl_rs::store::catalog::index_session(&conn, data_dir, session_id)?;
    }

    Ok(idl_rs::store::catalog_read::get_session(data_dir, session_id)?.into())
}

/// `SessionJsonError` → C3 §2, matching `commands::catalog`'s own mapping:
/// `InvalidArgument` (R194's `timestamp_utc_ms <= 0`) and `Io` pass through,
/// a malformed/unsupported document is `Internal` (the file is ours, so a
/// parse failure is a bug or corruption, not a caller mistake), and
/// `write_session_json`'s optimistic-concurrency failure surfaces as
/// `Conflict` — unlike `save_session_metadata` (last-write-wins, R59 Q1(a)),
/// this command's own error list in C3 §3.3 includes `conflict`.
fn map_session_json_error(e: idl_rs::store::session_json::SessionJsonError) -> IpcError {
    use idl_rs::store::session_json::SessionJsonErrorKind;
    match e.kind {
        SessionJsonErrorKind::InvalidArgument => IpcError::new(IpcErrorKind::InvalidArgument, e.message),
        SessionJsonErrorKind::Io if e.message.contains("changed since it was read") => {
            IpcError::new(IpcErrorKind::Conflict, e.message)
        }
        SessionJsonErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        SessionJsonErrorKind::Parse | SessionJsonErrorKind::UnsupportedVersion => {
            IpcError::new(IpcErrorKind::Internal, e.message)
        }
    }
}

/// Transport-agnostic core of `scan_folder` (C3 §3.3): core's own
/// non-recursive folder scan, with `data_dir`'s `blobs/` answering
/// `already_imported`, converted to the wire shape.
fn scan_folder_via(data_dir: &Path, path: &str) -> Result<Vec<ScanEntry>, IpcError> {
    let entries = idl_rs::store::scan::scan_folder(data_dir, Path::new(path))?;
    Ok(entries.into_iter().map(ScanEntry::from).collect())
}

/// Transport-agnostic core of `list_stale_sessions` (C3 §3.3): one catalog
/// query, no file reads. A `<data>` with no catalog yet has nothing
/// catalogued and so nothing stale — an empty list, not an error.
fn list_stale_sessions_via(data_dir: &Path) -> Result<Vec<StaleSession>, IpcError> {
    let catalog_path = data_dir.join("catalog.sqlite");
    if !catalog_path.is_file() {
        return Ok(Vec::new());
    }
    let conn = idl_rs::store::catalog::open_catalog(&catalog_path)?;
    let rows = idl_rs::store::catalog_read::list_stale_sessions(&conn)?;
    Ok(rows.into_iter().map(StaleSession::from).collect())
}

/// Transport-agnostic core of `reimport_sessions` (C3 §3.3): rebuilds each
/// session's `data.parquet` from its blob with the current importer, keeping
/// every human-owned `session.json` field and dropping `derived/` (core's
/// `reimport_session` does all of that). Per-session failures land in the
/// report rather than aborting the run; `on_progress` is called once per
/// session, after it finishes, with `done` counting sessions.
fn reimport_sessions_via(
    data_dir: &Path,
    session_ids: &[String],
    mut on_progress: impl FnMut(u64, u64),
) -> ReimportReport {
    let total = session_ids.len() as u64;
    let mut report = ReimportReport { rebuilt: Vec::new(), failed: Vec::new() };
    for (i, session_id) in session_ids.iter().enumerate() {
        match idl_rs::store::import::reimport_session(data_dir, session_id) {
            Ok(_) => report.rebuilt.push(session_id.clone()),
            Err(e) => report.failed.push(ReimportFailure { session_id: session_id.clone(), error: e.into() }),
        }
        on_progress(i as u64 + 1, total);
    }
    report
}

/// C3 §3.3 `set_session_start(session_id, timestamp_utc_ms)`.
#[tauri::command]
pub fn set_session_start(
    session_id: String,
    timestamp_utc_ms: i64,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<SessionDetail, IpcError> {
    set_session_start_via(&data_dir.0, &session_id, timestamp_utc_ms)
}

/// C3 §3.3 `scan_folder(path)`.
#[tauri::command]
pub fn scan_folder(path: String, data_dir: tauri::State<'_, DataDir>) -> Result<Vec<ScanEntry>, IpcError> {
    scan_folder_via(&data_dir.0, &path)
}

/// C3 §3.3 `list_stale_sessions()`.
#[tauri::command]
pub fn list_stale_sessions(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<StaleSession>, IpcError> {
    list_stale_sessions_via(&data_dir.0)
}

/// C3 §3.3 `reimport_sessions(session_ids, progress)`.
#[tauri::command]
pub fn reimport_sessions(
    session_ids: Vec<String>,
    progress: tauri::ipc::Channel<Progress>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<ReimportReport, IpcError> {
    Ok(reimport_sessions_via(&data_dir.0, &session_ids, |done, total| {
        let _ = progress.send(Progress { done, total: Some(total), phase: "sessions".to_string() });
    }))
}

/// C3 §3.3 `inbox_status()` — desktop only; on mobile the inbox does not
/// exist at all (ruling R191), so this is `unsupported_platform` rather than
/// an empty status, which would claim a working, empty inbox.
#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
pub fn inbox_status(inbox: tauri::State<'_, crate::inbox::InboxState>) -> Result<crate::inbox::InboxStatus, IpcError> {
    Ok(inbox.status())
}

/// C3 §3.3 `inbox_status()` on mobile — see the desktop version's doc
/// comment.
#[cfg(any(target_os = "android", target_os = "ios"))]
#[tauri::command]
pub fn inbox_status() -> Result<serde_json::Value, IpcError> {
    Err(IpcError::with_detail(
        IpcErrorKind::UnsupportedPlatform,
        "the inbox folder is desktop only",
        serde_json::json!({ "platform": std::env::consts::OS }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::parse::test_buffers::{cat, frame, imu_payload, session_end, v3_registry_entry, Header};
    use std::path::PathBuf;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-library-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn idl0_bytes() -> Vec<u8> {
        let accel: f32 = 32.0 / 32768.0;
        cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }
                .build(&[v3_registry_entry(0, 4, 800, accel, 0.0, "IMU0_AccelX", "g")]),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ])
    }

    /// Imports one synthetic `.idl0` into a fresh `<data>` and builds the
    /// catalog. Returns `(data_dir, session_id)`.
    fn imported_root() -> (PathBuf, String) {
        let root = temp_root();
        let report = idl_rs::store::import::import_idl0(&root, &idl0_bytes()).unwrap();
        idl_rs::store::catalog_read::rebuild_catalog_report(&root).unwrap();
        (root, report.session_id)
    }

    #[test]
    fn set_session_start_via_writes_the_user_start_and_the_returned_detail_shows_it() {
        // Arrange
        let (root, session_id) = imported_root();

        // Act
        let detail = set_session_start_via(&root, &session_id, 1_700_000_000_000).unwrap();

        // Assert — the returned detail, the file, and the catalog row all
        // agree on the user-supplied start.
        assert_eq!(detail.timestamp_utc_ms, 1_700_000_000_000);
        let doc =
            idl_rs::store::session_json::read_session_json(&root.join("sessions").join(&session_id).join("session.json"))
                .unwrap();
        assert_eq!(doc.timestamp_utc_ms, Some(1_700_000_000_000));
        assert_eq!(doc.timestamp_source, Some(idl_rs::session::TimestampSource::User));
        let listed = idl_rs::store::catalog_read::list_sessions(&root).unwrap();
        assert_eq!(listed[0].timestamp_utc_ms, 1_700_000_000_000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_session_start_via_a_non_positive_timestamp_is_invalid_argument() {
        // Arrange
        let (root, session_id) = imported_root();

        // Act
        let err = set_session_start_via(&root, &session_id, 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_session_start_via_an_unknown_session_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = set_session_start_via(&root, "nope", 1_700_000_000_000).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scan_folder_via_reports_importer_ids_and_already_imported_against_this_data_dir() {
        // Arrange — the same bytes are both imported and left in the folder.
        let (root, _session_id) = imported_root();
        let folder = temp_root();
        std::fs::write(folder.join("a.idl0"), idl0_bytes()).unwrap();
        std::fs::write(folder.join("b.txt"), b"notes").unwrap();

        // Act
        let entries = scan_folder_via(&root, folder.to_str().unwrap()).unwrap();

        // Assert
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].importer_id.as_deref(), Some("idl0"));
        assert!(entries[0].already_imported);
        // The header peek reports the fixture header's own start, with no
        // record decoding at all.
        assert_eq!(entries[0].session_start_utc_ms, Some(idl_rs::parse::test_buffers::RMC_UTC_MS));
        assert_eq!(entries[1].importer_id, None);
        assert!(!entries[1].already_imported);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn scan_folder_via_a_missing_folder_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = scan_folder_via(&root, root.join("nope").to_str().unwrap()).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_stale_sessions_via_lists_a_row_stamped_with_an_older_importer_version() {
        // Arrange — age the catalogued version, as a previous build's import
        // would have left it.
        let (root, session_id) = imported_root();
        let conn = idl_rs::store::catalog::open_catalog(&root.join("catalog.sqlite")).unwrap();
        conn.execute("UPDATE sessions SET importer_version = '0.0.1'", []).unwrap();
        drop(conn);

        // Act
        let stale = list_stale_sessions_via(&root).unwrap();

        // Assert
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].session_id, session_id);
        assert_eq!(stale[0].importer_id, "idl0");
        assert_eq!(stale[0].stored_version, "0.0.1");
        assert_eq!(stale[0].current_version, idl_rs::parse::IDL0_IMPORTER_VERSION);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_stale_sessions_via_a_data_dir_with_no_catalog_is_empty_not_an_error() {
        // Arrange
        let root = temp_root();

        // Act
        let stale = list_stale_sessions_via(&root).unwrap();

        // Assert
        assert!(stale.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn reimport_sessions_via_reports_rebuilt_and_failed_separately_and_streams_progress() {
        // Arrange — one real session and one id that does not exist.
        let (root, session_id) = imported_root();
        let ids = vec![session_id.clone(), "no-such-session".to_string()];
        let mut progress: Vec<(u64, u64)> = Vec::new();

        // Act
        let report = reimport_sessions_via(&root, &ids, |done, total| progress.push((done, total)));

        // Assert — one rebuilt, one failed, and progress counted sessions.
        assert_eq!(report.rebuilt, vec![session_id]);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].session_id, "no-such-session");
        assert_eq!(report.failed[0].error.kind, IpcErrorKind::Io);
        assert_eq!(progress, vec![(1, 2), (2, 2)]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn reimport_sessions_via_keeps_a_user_supplied_start_across_the_rebuild() {
        // Arrange
        let (root, session_id) = imported_root();
        set_session_start_via(&root, &session_id, 1_700_000_000_000).unwrap();

        // Act
        let report = reimport_sessions_via(&root, &[session_id.clone()], |_, _| {});

        // Assert — the rebuild is a function of (blob, importer version)
        // only; the human's value survives it (C1 §3.1).
        assert!(report.failed.is_empty());
        let doc =
            idl_rs::store::session_json::read_session_json(&root.join("sessions").join(&session_id).join("session.json"))
                .unwrap();
        assert_eq!(doc.timestamp_utc_ms, Some(1_700_000_000_000));

        let _ = std::fs::remove_dir_all(&root);
    }
}
