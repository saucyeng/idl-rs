//! The library summary a device serves to its paired peers (library tiers
//! Phase 1, spec 4a; `GET /idl1/v1/library`): one row per catalogued session
//! and per workbook, enough to browse the other device's library and pick
//! what to pull, without moving a byte of it.
//!
//! Built read-only from the serving library's `catalog.sqlite` — the catalog
//! is an index, so a library whose catalog is missing, mid-rebuild or of
//! another schema version answers [`LibrarySummaryErrorKind::CatalogUnavailable`]
//! rather than being repaired here. Nothing under `<data>` is created or
//! written by this module.

use std::fmt;
use std::path::Path;

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

use crate::store::catalog::CATALOG_SCHEMA_VERSION;

/// `schema_version` of the summary document.
pub const LIBRARY_SUMMARY_SCHEMA_VERSION: u32 = 1;

/// One session in a [`LibrarySummary`]. Text fields the user may leave blank
/// are `None`, never `""`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibrarySession {
    /// The session's id (lowercase hex).
    pub session_id: String,
    /// Recording start, Unix epoch milliseconds; `0` = unknown.
    pub timestamp_utc_ms: i64,
    /// Recording length, milliseconds; `None` when the catalog has no duration.
    pub duration_ms: Option<i64>,
    pub venue_name: Option<String>,
    pub event_name: Option<String>,
    pub rider: Option<String>,
    pub bike: Option<String>,
    pub tag: Option<String>,
    /// `"idl0"`, `"fit"`, `"gpx"` or `"csv"`.
    pub source_format: String,
    /// Laps indexed for the session; `None` until laps are indexed.
    pub lap_count: Option<u32>,
    /// What pulling the whole session moves, bytes: its source blob plus
    /// `data.parquet` plus every `derived/*.parquet`.
    pub size_bytes: u64,
}

/// One workbook in a [`LibrarySummary`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryWorkbook {
    /// Stable id from the workbook's front matter.
    pub workbook_id: String,
    /// The file's stem (without `.idl1wb`).
    pub file_name: String,
    /// File size, bytes.
    pub size_bytes: u64,
    /// File modification time, Unix epoch milliseconds.
    pub updated_at_ms: i64,
}

/// The document `GET /idl1/v1/library` serves (spec 4a).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibrarySummary {
    /// [`LIBRARY_SUMMARY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Most recent recording first.
    pub sessions: Vec<LibrarySession>,
    /// By display name.
    pub workbooks: Vec<LibraryWorkbook>,
}

/// Why a summary could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibrarySummaryErrorKind {
    /// `catalog.sqlite` is absent, locked, of another schema version or
    /// unreadable: the library is still indexing (or needs a rebuild).
    CatalogUnavailable,
}

/// Error from [`library_summary`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibrarySummaryError {
    pub kind: LibrarySummaryErrorKind,
    pub message: String,
}

impl fmt::Display for LibrarySummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for LibrarySummaryError {}

fn unavailable(message: impl fmt::Display) -> LibrarySummaryError {
    LibrarySummaryError { kind: LibrarySummaryErrorKind::CatalogUnavailable, message: message.to_string() }
}

/// `None` for an empty string: the catalog stores "not set" as `""`.
fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Bytes of a file, `0` when it is absent or unreadable (a session mid-import
/// has no `data.parquet` yet).
fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Total bytes of every `*.parquet` file directly under `dir`; `0` for a
/// missing directory.
fn parquet_dir_len(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("parquet"))
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// Builds the library summary for the library rooted at `data_root` from its
/// `catalog.sqlite`, opened read-only: the file is never created, and no
/// schema is bootstrapped. Sizes of `data.parquet` and `derived/` are read
/// from disk; the blob's from the catalog.
///
/// Returns [`LibrarySummaryErrorKind::CatalogUnavailable`] when the catalog
/// file is absent, cannot be opened or queried, or is not at
/// [`CATALOG_SCHEMA_VERSION`].
pub fn library_summary(data_root: &Path) -> Result<LibrarySummary, LibrarySummaryError> {
    let catalog_path = data_root.join("catalog.sqlite");
    if !catalog_path.is_file() {
        return Err(unavailable("catalog.sqlite does not exist"));
    }
    let conn = Connection::open_with_flags(&catalog_path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .map_err(unavailable)?;
    conn.busy_timeout(std::time::Duration::from_millis(500)).map_err(unavailable)?;
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(unavailable)?;
    if version != CATALOG_SCHEMA_VERSION {
        return Err(unavailable(format!("catalog schema version {version}, expected {CATALOG_SCHEMA_VERSION}")));
    }
    let sessions = read_sessions(&conn, data_root).map_err(unavailable)?;
    let workbooks = read_workbooks(&conn).map_err(unavailable)?;
    Ok(LibrarySummary { schema_version: LIBRARY_SUMMARY_SCHEMA_VERSION, sessions, workbooks })
}

fn read_sessions(conn: &Connection, data_root: &Path) -> Result<Vec<LibrarySession>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT s.session_id, s.timestamp_utc_ms, s.duration_ms, s.venue_name, s.event_name, s.rider, s.bike, s.tag, \
         s.source_format, s.lap_count, b.size_bytes \
         FROM sessions s JOIN blobs b ON b.sha256 = s.blob_sha256 ORDER BY s.timestamp_utc_ms DESC, s.session_id ASC",
    )?;
    let rows = stmt.query_map([], |r| {
        let session_id: String = r.get(0)?;
        let blob_bytes = r.get::<_, i64>(10)?.max(0) as u64;
        let dir = data_root.join("sessions").join(&session_id);
        let size_bytes = blob_bytes + file_len(&dir.join("data.parquet")) + parquet_dir_len(&dir.join("derived"));
        Ok(LibrarySession {
            session_id,
            timestamp_utc_ms: r.get(1)?,
            duration_ms: r.get(2)?,
            venue_name: non_empty(r.get(3)?),
            event_name: non_empty(r.get(4)?),
            rider: non_empty(r.get(5)?),
            bike: non_empty(r.get(6)?),
            tag: non_empty(r.get(7)?),
            source_format: r.get(8)?,
            lap_count: r.get::<_, Option<i64>>(9)?.map(|n| n as u32),
            size_bytes,
        })
    })?;
    rows.collect()
}

fn read_workbooks(conn: &Connection) -> Result<Vec<LibraryWorkbook>, rusqlite::Error> {
    let mut stmt =
        conn.prepare("SELECT workbook_id, file_name, size_bytes, updated_at_ms FROM workbooks ORDER BY name ASC, workbook_id ASC")?;
    let rows = stmt.query_map([], |r| {
        Ok(LibraryWorkbook {
            workbook_id: r.get(0)?,
            file_name: r.get(1)?,
            size_bytes: r.get::<_, i64>(2)?.max(0) as u64,
            updated_at_ms: r.get(3)?,
        })
    })?;
    rows.collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use crate::store::blob::write_blob;
    use crate::store::catalog::{open_catalog, rebuild_catalog, upsert_workbook};
    use crate::store::catalog_read::WorkbookSummary;
    use crate::store::derived::{write_derived_parquet, DerivedOutput};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{empty_session_json, write_session_json};

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-libsum-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes one session (blob, `session.json`, `data.parquet`) and returns the blob's byte length.
    fn write_session(root: &Path, session_id: &str, timestamp_utc_ms: i64, rider: &str) -> u64 {
        let raw = format!("raw bytes for {session_id}");
        let blob_sha256 = write_blob(root, raw.as_bytes()).unwrap();
        let mut doc = empty_session_json(session_id);
        doc.rider = rider.to_string();
        write_session_json(root, session_id, &doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: Some("d1".to_string()),
            timestamp_utc_ms,
            timestamp_source: TimestampSource::Header,
            config_checksum: Some("cafef00d".to_string()),
            source_format: SourceFormat::Idl0,
            blob_sha256,
            channels: vec![Channel {
                channel_id: "IMU0_AccelX".to_string(),
                t_us: vec![0, 500_000, 1_000_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0, 3.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, crate::parse::IDL0_IMPORTER_VERSION).unwrap();
        raw.len() as u64
    }

    #[test]
    fn library_summary_a_session_blank_text_fields_are_none_and_size_is_blob_plus_parquet_plus_derived() {
        // Arrange
        let root = temp_root();
        let session_id = "00112233445566778899aabbccddeeff";
        let blob_len = write_session(&root, session_id, 1_700_000_000_000, "");
        let outputs = vec![DerivedOutput { channel_id: "out".to_string(), t_us: vec![0, 1_000_000], values: vec![1.0, 2.0], nominal_rate_hz: 1.0, unit: "deg".to_string() }];
        write_derived_parquet(&root, session_id, "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();
        rebuild_catalog(&root).unwrap();
        let dir = root.join("sessions").join(session_id);
        let parquet_len = std::fs::metadata(dir.join("data.parquet")).unwrap().len();
        let derived_len: u64 = std::fs::read_dir(dir.join("derived")).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum();

        // Act
        let summary = library_summary(&root).unwrap();

        // Assert
        assert_eq!(summary.schema_version, 1);
        assert_eq!(summary.sessions.len(), 1);
        let s = &summary.sessions[0];
        assert_eq!(s.session_id, session_id);
        assert_eq!(s.timestamp_utc_ms, 1_700_000_000_000);
        assert_eq!(s.source_format, "idl0");
        assert_eq!(s.rider, None);
        assert_eq!(s.venue_name, None);
        assert_eq!(s.tag, None);
        assert!(derived_len > 0);
        assert_eq!(s.size_bytes, blob_len + parquet_len + derived_len);
        let json = serde_json::to_value(&summary).unwrap();
        assert!(json["sessions"][0]["rider"].is_null());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn library_summary_sessions_are_most_recent_first_and_a_set_rider_is_carried() {
        // Arrange
        let root = temp_root();
        write_session(&root, "aa112233445566778899aabbccddeeff", 1_000, "Isaac");
        write_session(&root, "bb112233445566778899aabbccddeeff", 2_000, "");
        rebuild_catalog(&root).unwrap();

        // Act
        let summary = library_summary(&root).unwrap();

        // Assert
        let ids: Vec<&str> = summary.sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(ids, vec!["bb112233445566778899aabbccddeeff", "aa112233445566778899aabbccddeeff"]);
        assert_eq!(summary.sessions[1].rider.as_deref(), Some("Isaac"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn library_summary_workbooks_carry_id_file_name_size_and_mtime() {
        // Arrange
        let root = temp_root();
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let row = WorkbookSummary {
            workbook_id: "11111111-1111-4111-8111-111111111111".to_string(),
            file_name: "Fork study".to_string(),
            name: "Fork study".to_string(),
            updated_at_ms: 1_700_000_000_123,
            size_bytes: 4_096,
        };
        upsert_workbook(&conn, &row).unwrap();
        drop(conn);

        // Act
        let summary = library_summary(&root).unwrap();

        // Assert
        assert_eq!(
            summary.workbooks,
            vec![LibraryWorkbook {
                workbook_id: row.workbook_id.clone(),
                file_name: "Fork study".to_string(),
                size_bytes: 4_096,
                updated_at_ms: 1_700_000_000_123,
            }]
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn library_summary_no_catalog_file_is_catalog_unavailable_and_nothing_is_created() {
        // Arrange
        let root = temp_root();

        // Act
        let err = library_summary(&root).unwrap_err();

        // Assert
        assert_eq!(err.kind, LibrarySummaryErrorKind::CatalogUnavailable);
        assert!(!root.join("catalog.sqlite").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn library_summary_a_catalog_of_another_schema_version_is_catalog_unavailable() {
        // Arrange
        let root = temp_root();
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        conn.pragma_update(None, "user_version", CATALOG_SCHEMA_VERSION + 1).unwrap();
        drop(conn);

        // Act
        let err = library_summary(&root).unwrap_err();

        // Assert
        assert_eq!(err.kind, LibrarySummaryErrorKind::CatalogUnavailable);

        let _ = std::fs::remove_dir_all(&root);
    }
}
