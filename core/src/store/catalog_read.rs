//! Catalog read API (contract C3 §3.2) — the seven read queries the
//! `idl-rs-tauri` catalog commands wrap. `store::catalog` (L1) is
//! write/rebuild only; this is a new file so none of it is edited (ledger
//! R40). `get_session`/`get_track` read canonical files directly, never the
//! catalog (C4 §5: "nothing reads the catalog for truth"); `list_laps`
//! reads the catalog's cached `laps`/`lap_summary` tables, as C3 §3.2
//! states explicitly for that one command.
//!
//! **Not-found encoding (ruling R46):** this module raises
//! [`CatalogErrorKind::NotFound`] explicitly wherever an id/path has no
//! backing row/file; a genuine `rusqlite::Error` (e.g. a corrupt
//! `catalog.sqlite`) keeps `CatalogErrorKind::Sql` via `catalog.rs`'s own
//! `From<rusqlite::Error>`, and any other failure (filesystem, parse) folds
//! to `Io`. `idl-rs-tauri`'s `From<CatalogError> for IpcError` maps
//! `NotFound -> not_found`, `Sql -> internal`, `Io -> io` — matching C3
//! §3.2's per-command error sets exactly. (R46 replaced an earlier revision
//! of this module, which reused `Sql` for both meanings and so surfaced a
//! corrupt catalog as `not_found` instead of `internal`.)

use std::path::Path;

use crate::store::catalog::{open_catalog, rebuild_catalog, CatalogError, CatalogErrorKind, RebuildReport};
use crate::store::parquet::{read_session_metadata, read_session_parquet};
use crate::store::session_json::{read_session_json, LapJson, OverlayLapKeyJson, TrackVisitJson};
use crate::track_artifact::read::read_track;

/// One row of the catalog `sessions` table (C4 §5), mirrored field for
/// field per C3 §3.2's `SessionSummary`.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    pub session_id: String,
    /// Hex-encoded, lowercase, 64 chars (C4 §5 `sessions.blob_sha256`).
    pub blob_sha256: String,
    /// One of `"idl0"`/`"fit"`/`"gpx"`/`"csv"` — file-level, which importer
    /// produced this session (C1 §2 `Session.source_format`).
    pub source_format: String,
    /// `None` for FIT/GPX/CSV sources — no device (C1 §2).
    pub device_id: Option<String>,
    /// `None` for FIT/GPX/CSV sources (C1 §2).
    pub config_checksum: Option<String>,
    /// SemVer 2.0.0 (C1 §4.3).
    pub importer_version: String,
    /// e.g. `"v1"` (C1 §4.3).
    pub seam_correction_version: String,
    /// SemVer 2.0.0, `idl-rs` core `CARGO_PKG_VERSION` (C1 §4.3).
    pub engine_version: String,
    /// Recording start, Unix epoch milliseconds; `0` = unknown (C1 §3.1).
    pub timestamp_utc_ms: i64,
    /// Catalog row insert time (import time), Unix epoch milliseconds.
    pub created_at_ms: i64,
    /// `""` = not set.
    pub rider: String,
    pub bike: String,
    pub venue_name: String,
    pub event_name: String,
    pub event_session: String,
    pub short_comment: String,
    pub tag: String,
    /// `None` until laps are indexed for this session; counts the same rows
    /// [`list_laps`] returns.
    pub lap_count: Option<u32>,
    /// Milliseconds. `None` when the catalog row has no duration (see
    /// `catalog.rs`'s `RebuildReport` — sessions with fewer than two `t`
    /// samples).
    pub duration_ms: Option<i64>,
}

/// C1 `Session` metadata plus `session.json` content (C3 §3.2). Read from
/// canonical files, never from the catalog — a `SessionSummary` row is a
/// possibly-stale cache of a subset of these same values.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionDetail {
    pub session_id: String,
    pub device_id: Option<String>,
    pub timestamp_utc_ms: i64,
    pub config_checksum: Option<String>,
    /// One of `"idl0"`/`"fit"`/`"gpx"`/`"csv"`, lowercase (C1 `SourceFormat`).
    pub source_format: String,
    pub blob_sha256: String,
    pub channels: Vec<ChannelSummary>,

    pub rider: String,
    pub bike: String,
    pub bike_comment: String,
    pub venue_name: String,
    pub event_name: String,
    pub event_session: String,
    pub short_comment: String,
    pub long_comment: String,
    pub tag: String,
    /// Verbatim `BikeProfile.config` at recording time (C1 §6).
    pub bike_profile_snapshot: Option<serde_json::Value>,

    /// `session.json`'s own `laps[]` — the file-native lap shape, distinct
    /// from [`LapSummary`] (the catalog's cached shape, [`list_laps`]-only).
    pub laps: Vec<LapDetail>,
    pub track_visits: Vec<TrackVisitSummary>,
    /// `None` = "use fastest lap" (C1 §6).
    pub reference_lap_number: Option<u32>,
    /// Sorted ascending (C1 §6).
    pub ignored_lap_numbers: Vec<u32>,
    pub main_lap_number: Option<u32>,
    pub overlay_lap_key: Option<OverlayLapKeyJson>,
    pub starred_lap_number: Option<u32>,
    /// Opaque, do not parse (C1 §6).
    pub track_visits_library_hash: Option<String>,
}

/// Mirrors C1 §6 `session.json`'s `laps[]` object field for field — the
/// file-native lap shape, distinct from [`LapSummary`]'s catalog shape.
#[derive(Debug, Clone, PartialEq)]
pub struct LapDetail {
    /// 1-based (C1 §6 `laps[].lap_number`).
    pub lap_number: u32,
    /// Unix epoch milliseconds.
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
    /// Milliseconds — `end_timestamp_ms - start_timestamp_ms`.
    pub raw_elapsed_ms: i64,
    /// Milliseconds — `raw_elapsed_ms` minus neutral-zone time.
    pub lap_time_ms: i64,
    /// Seconds, recording-time (t=0-anchored).
    pub start_time_secs: f64,
    pub end_time_secs: f64,
    /// C1 §6 does not fix the element shape beyond "array" (C3 §3.2 open
    /// question 11) — opaque on the Rust side, verbatim from `session.json`.
    pub sectors: serde_json::Value,
    /// As [`LapDetail::sectors`], opaque.
    pub neutral_zone_visits: serde_json::Value,
}

/// One visit to a track-library entry, with its own cached laps (C1 §6).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackVisitSummary {
    /// UUID (C1 §6 `track_visits[].visit_id`).
    pub visit_id: String,
    /// UUID.
    pub track_id: String,
    pub start_timestamp_ms: i64,
    /// Always `>= start_timestamp_ms`.
    pub end_timestamp_ms: i64,
    /// Same shape as [`SessionDetail::laps`]; `[]` when `session.json`
    /// omits the key.
    pub laps: Vec<LapDetail>,
}

/// One channel's summary within [`SessionDetail`] (C3 §3.2).
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSummary {
    pub channel_id: String,
    /// Hz, metadata only — never used to synthesize time (C1 §3.5).
    pub nominal_rate_hz: f64,
    /// C1 §4.1's per-channel unit.
    pub unit: String,
    /// Channel-level: which sensor this channel came from (C1 §4.2), e.g.
    /// `"imu0"`, `"gps"`, `"fit"`, `"gpx"`. Distinct from the file-level
    /// `source_format` on [`SessionSummary`]/[`SessionDetail`].
    pub source_kind: String,
    /// `"event"` iff `nominal_rate_hz == 0.0`, `"fixed-rate"` otherwise
    /// (C1 §4.2, C3 §3.2's own rule).
    pub channel_kind: String,
    pub sample_count: u64,
}

/// One row of the catalog `laps` table plus its `lap_summary` rows (C3
/// §3.2, [`list_laps`]-only shape).
#[derive(Debug, Clone, PartialEq)]
pub struct LapSummary {
    /// 1-based (C4 §5 `laps.lap_number`, C1 §6 `laps[].lap_number`).
    pub lap_number: i32,
    /// Milliseconds (C4 §5 `laps.lap_time_ms`).
    pub lap_time_ms: i64,
    /// `None` if this lap isn't attributed to a track (C4 §5 `laps.track_id`).
    pub track_id: Option<String>,
    /// `lap_summary` rows for this `(session, lap)`, `channel_id` ascending.
    pub channel_stats: Vec<LapChannelStat>,
}

/// One `lap_summary` row (C4 §5).
#[derive(Debug, Clone, PartialEq)]
pub struct LapChannelStat {
    /// Materialised channel name (C4 §5 `lap_summary.channel_id`).
    pub channel_id: String,
    /// 64-hex — which `derived/<hash>.parquet` this was computed from.
    pub derived_hash: String,
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
}

/// One row of the catalog `workbooks` table (C4 §5), mirrored field for
/// field per C3 §3.2's `WorkbookSummary`.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkbookSummary {
    /// Stable id from front matter (C2).
    pub workbook_id: String,
    /// The bare, filesystem-sanitised display name used as
    /// `workbooks/<file_name>.idl1wb` (C4 §2).
    pub file_name: String,
    /// Display name from front matter (C2).
    pub name: String,
    /// File mtime, Unix epoch milliseconds.
    pub updated_at_ms: i64,
    pub size_bytes: u64,
}

/// One row of the catalog `tracks` table (C4 §5), excluding `full_json` —
/// mirrored field for field per C3 §3.2's `TrackSummary`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackSummary {
    pub track_id: String,
    pub name: String,
    pub venue_name: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// The full `.idl0t` artifact content (C3 §3.2). The four fields C3 leaves
/// `unknown` are opaque `serde_json::Value`, lifted verbatim from the
/// artifact's own JSON — no contract has fixed their element shape yet
/// (open question 10).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackDetail {
    pub track_id: String,
    pub name: String,
    pub venue_name: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    /// `Value::Null` when the artifact has no `lap_timing` key — sealed
    /// union (`Circuit` | `PointToPoint`, IDL0_SPEC §16.2a).
    pub lap_timing: serde_json::Value,
    /// `Value::Array(vec![])` when the artifact has no `neutral_zones` key.
    pub neutral_zones: serde_json::Value,
    /// `Value::Array(vec![])` when the artifact has no `sector_gates` key.
    pub sector_gates: serde_json::Value,
    /// `Value::Array(vec![])` when the artifact has no `reference_polyline` key.
    pub reference_polyline: serde_json::Value,
}

/// A not-found-shaped [`CatalogError`] (ruling R46).
fn not_found(message: impl Into<String>) -> CatalogError {
    CatalogError { kind: CatalogErrorKind::NotFound, message: message.into() }
}

/// Folds any I/O/parse/schema failure into `CatalogErrorKind::Io` — this
/// module's default bucket for "not the not-found case" (see the module doc
/// comment). Takes anything `Display` so it covers `ParquetStoreError`,
/// `SessionJsonError`, `ConfigError` and `std::io::Error` alike without a
/// `From` impl per source type.
fn io_like(message: impl std::fmt::Display) -> CatalogError {
    CatalogError { kind: CatalogErrorKind::Io, message: message.to_string() }
}

/// C3 §3.2 `list_sessions()` — every `sessions` row, most recent first.
pub fn list_sessions(data_root: &Path) -> Result<Vec<SessionSummary>, CatalogError> {
    let conn = open_catalog(&data_root.join("catalog.sqlite"))?;
    let mut stmt = conn.prepare(
        "SELECT session_id, blob_sha256, source_format, device_id, config_checksum, importer_version, \
         seam_correction_version, engine_version, timestamp_utc_ms, created_at_ms, rider, bike, venue_name, \
         event_name, event_session, short_comment, tag, lap_count, duration_ms \
         FROM sessions ORDER BY timestamp_utc_ms DESC",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(SessionSummary {
            session_id: r.get(0)?,
            blob_sha256: r.get(1)?,
            source_format: r.get(2)?,
            device_id: r.get(3)?,
            config_checksum: r.get(4)?,
            importer_version: r.get(5)?,
            seam_correction_version: r.get(6)?,
            engine_version: r.get(7)?,
            timestamp_utc_ms: r.get(8)?,
            created_at_ms: r.get(9)?,
            rider: r.get(10)?,
            bike: r.get(11)?,
            venue_name: r.get(12)?,
            event_name: r.get(13)?,
            event_session: r.get(14)?,
            short_comment: r.get(15)?,
            tag: r.get(16)?,
            lap_count: r.get::<_, Option<i64>>(17)?.map(|n| n as u32),
            duration_ms: r.get(18)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(CatalogError::from)
}

/// C3 §3.2 `get_session(session_id)` — C1 `Session` metadata plus
/// `session.json` content, read from canonical files only, never the
/// catalog (C4 §5).
pub fn get_session(data_root: &Path, session_id: &str) -> Result<SessionDetail, CatalogError> {
    let session_dir = data_root.join("sessions").join(session_id);
    if !session_dir.is_dir() {
        return Err(not_found(format!("session {session_id} not found")));
    }

    let dp_path = session_dir.join("data.parquet");
    let sj_path = session_dir.join("session.json");
    let metadata = read_session_metadata(&dp_path).map_err(io_like)?;
    let doc = read_session_json(&sj_path).map_err(io_like)?;
    let session = read_session_parquet(&dp_path).map_err(io_like)?;

    let channels = session
        .channels
        .iter()
        .map(|c| ChannelSummary {
            channel_id: c.channel_id.clone(),
            nominal_rate_hz: c.nominal_rate_hz,
            unit: c.unit.clone(),
            source_kind: c.source_kind.clone(),
            channel_kind: if c.nominal_rate_hz == 0.0 { "event".to_string() } else { "fixed-rate".to_string() },
            sample_count: c.len() as u64,
        })
        .collect();

    Ok(SessionDetail {
        session_id: session_id.to_string(),
        device_id: metadata.device_id,
        timestamp_utc_ms: metadata.timestamp_utc_ms,
        config_checksum: metadata.config_checksum,
        source_format: session.source_format.as_str().to_string(),
        blob_sha256: metadata.blob_sha256,
        channels,
        rider: doc.rider,
        bike: doc.bike,
        bike_comment: doc.bike_comment,
        venue_name: doc.venue_name,
        event_name: doc.event_name,
        event_session: doc.event_session,
        short_comment: doc.short_comment,
        long_comment: doc.long_comment,
        tag: doc.tag,
        bike_profile_snapshot: doc.bike_profile_snapshot,
        laps: doc.laps.iter().map(lap_json_to_detail).collect(),
        track_visits: doc.track_visits.iter().map(track_visit_json_to_summary).collect(),
        reference_lap_number: doc.reference_lap_number,
        ignored_lap_numbers: doc.ignored_lap_numbers,
        main_lap_number: doc.main_lap_number,
        overlay_lap_key: doc.overlay_lap_key,
        starred_lap_number: doc.starred_lap_number,
        track_visits_library_hash: doc.track_visits_library_hash,
    })
}

/// `session.json`'s `LapJson` -> this module's opaque-`sectors`/
/// `neutral_zone_visits` [`LapDetail`] (C3 §3.2 open question 11: the
/// element shape isn't fixed by any contract, so it crosses verbatim as
/// JSON rather than a typed Rust shape).
fn lap_json_to_detail(lap: &LapJson) -> LapDetail {
    LapDetail {
        lap_number: lap.lap_number,
        start_timestamp_ms: lap.start_timestamp_ms,
        end_timestamp_ms: lap.end_timestamp_ms,
        raw_elapsed_ms: lap.raw_elapsed_ms,
        lap_time_ms: lap.lap_time_ms,
        start_time_secs: lap.start_time_secs,
        end_time_secs: lap.end_time_secs,
        sectors: serde_json::to_value(&lap.sectors).unwrap_or_else(|_| serde_json::Value::Array(Vec::new())),
        neutral_zone_visits: serde_json::to_value(&lap.neutral_zone_visits)
            .unwrap_or_else(|_| serde_json::Value::Array(Vec::new())),
    }
}

fn track_visit_json_to_summary(visit: &TrackVisitJson) -> TrackVisitSummary {
    TrackVisitSummary {
        visit_id: visit.visit_id.clone(),
        track_id: visit.track_id.clone(),
        start_timestamp_ms: visit.start_timestamp_ms,
        end_timestamp_ms: visit.end_timestamp_ms,
        laps: visit.laps.iter().map(lap_json_to_detail).collect(),
    }
}

/// C3 §3.2 `list_laps(session_id)` — catalog-backed: `laps` joined with
/// `lap_summary`, `channel_id` ascending within each lap.
/// `session_id` with no `sessions` row is not-found; a session with zero
/// laps is `Ok(vec![])`.
pub fn list_laps(data_root: &Path, session_id: &str) -> Result<Vec<LapSummary>, CatalogError> {
    let conn = open_catalog(&data_root.join("catalog.sqlite"))?;
    let exists: bool = conn
        .query_row("SELECT 1 FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |_| Ok(()))
        .is_ok();
    if !exists {
        return Err(not_found(format!("session {session_id} not found")));
    }

    let mut lap_stmt =
        conn.prepare("SELECT lap_number, lap_time_ms, track_id FROM laps WHERE session_id = ?1 ORDER BY lap_number ASC")?;
    let laps = lap_stmt
        .query_map(rusqlite::params![session_id], |r| {
            Ok((r.get::<_, i32>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<String>>(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(lap_stmt);

    let mut stat_stmt = conn.prepare(
        "SELECT channel_id, derived_hash, min_value, max_value, mean_value FROM lap_summary \
         WHERE session_id = ?1 AND lap_number = ?2 ORDER BY channel_id ASC",
    )?;
    let mut out = Vec::with_capacity(laps.len());
    for (lap_number, lap_time_ms, track_id) in laps {
        let channel_stats = stat_stmt
            .query_map(rusqlite::params![session_id, lap_number], |r| {
                Ok(LapChannelStat {
                    channel_id: r.get(0)?,
                    derived_hash: r.get(1)?,
                    min_value: r.get(2)?,
                    max_value: r.get(3)?,
                    mean_value: r.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        out.push(LapSummary { lap_number, lap_time_ms, track_id, channel_stats });
    }
    Ok(out)
}

/// C3 §3.2 `rebuild_catalog()` — returns `store::catalog::RebuildReport`
/// unchanged; the C3 four-field mapping and timing happen at the command
/// boundary (`idl-rs-tauri`), not here.
pub fn rebuild_catalog_report(data_root: &Path) -> Result<RebuildReport, CatalogError> {
    rebuild_catalog(data_root)
}

/// C3 §3.2 `list_workbooks()` — every `workbooks` row.
pub fn list_workbooks(data_root: &Path) -> Result<Vec<WorkbookSummary>, CatalogError> {
    let conn = open_catalog(&data_root.join("catalog.sqlite"))?;
    let mut stmt = conn.prepare("SELECT workbook_id, file_name, name, updated_at_ms, size_bytes FROM workbooks ORDER BY name ASC")?;
    let rows = stmt.query_map([], |r| {
        Ok(WorkbookSummary {
            workbook_id: r.get(0)?,
            file_name: r.get(1)?,
            name: r.get(2)?,
            updated_at_ms: r.get(3)?,
            size_bytes: r.get::<_, i64>(4)? as u64,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(CatalogError::from)
}

/// C3 §3.2 `list_tracks()` — every `tracks` row, excluding `full_json`.
pub fn list_tracks(data_root: &Path) -> Result<Vec<TrackSummary>, CatalogError> {
    let conn = open_catalog(&data_root.join("catalog.sqlite"))?;
    let mut stmt =
        conn.prepare("SELECT track_id, name, venue_name, created_at_ms, updated_at_ms FROM tracks ORDER BY name ASC")?;
    let rows = stmt.query_map([], |r| {
        Ok(TrackSummary { track_id: r.get(0)?, name: r.get(1)?, venue_name: r.get(2)?, created_at_ms: r.get(3)?, updated_at_ms: r.get(4)? })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(CatalogError::from)
}

/// C3 §3.2 `get_track(track_id)` — the full `<data>/tracks/<track_id>.idl0t`
/// artifact, read from disk (never on a hot path, C3 §4). Scalars come from
/// the parsed domain [`Track`](crate::track_artifact::model::Track); the
/// four opaque fields are lifted verbatim from the same bytes as JSON
/// (`serde_json::Value`), since no contract fixes their element shape yet.
pub fn get_track(data_root: &Path, track_id: &str) -> Result<TrackDetail, CatalogError> {
    let path = data_root.join("tracks").join(format!("{track_id}.idl0t"));
    if !path.is_file() {
        return Err(not_found(format!("track {track_id} not found")));
    }

    let bytes = std::fs::read(&path).map_err(io_like)?;
    let track = read_track(&path).map_err(io_like)?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(io_like)?;
    let track_obj = json.get("track");
    let opt_or_null = |key: &str| track_obj.and_then(|t| t.get(key)).cloned().unwrap_or(serde_json::Value::Null);
    let opt_or_empty_array =
        |key: &str| track_obj.and_then(|t| t.get(key)).cloned().unwrap_or(serde_json::Value::Array(Vec::new()));

    Ok(TrackDetail {
        track_id: track.id,
        name: track.name,
        venue_name: track.venue,
        created_at_ms: track.created_at_ms,
        updated_at_ms: track.updated_at_ms,
        lap_timing: opt_or_null("lap_timing"),
        neutral_zones: opt_or_empty_array("neutral_zones"),
        sector_gates: opt_or_empty_array("sector_gates"),
        reference_polyline: opt_or_empty_array("reference_polyline"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    use crate::session::{Channel, RawColumn, Session, SourceFormat};
    use crate::store::blob::write_blob;
    use crate::store::derived::{write_derived_parquet, DerivedOutput};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{empty_session_json, write_session_json};

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a session's blob, `session.json` and `data.parquet` (one
    /// fixed-rate channel), then rebuilds the catalog — the same tree shape
    /// `catalog.rs`'s own tests build (`write_full_session`), reimplemented
    /// locally per the brief (that helper is `#[cfg(test)]`-private there).
    fn write_full_session(root: &Path, session_id: &str, timestamp_utc_ms: i64, doc: &crate::store::session_json::SessionJson) {
        let blob_sha256 = write_blob(root, format!("raw bytes for {session_id}").as_bytes()).unwrap();
        write_session_json(root, session_id, doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: Some("d1".to_string()),
            timestamp_utc_ms,
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
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn list_sessions_empty_catalog_returns_an_empty_vec() {
        // Arrange
        let root = temp_root();
        rebuild_catalog(&root).unwrap();

        // Act
        let out = list_sessions(&root).unwrap();

        // Assert
        assert!(out.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_sessions_corrupt_catalog_file_is_a_sql_error_not_a_not_found_error() {
        // Arrange — R46: a genuine SQLite failure must stay `Sql` (folds to
        // `internal` at the IPC boundary), never `NotFound` (which would
        // point the caller at re-importing rather than `rebuild_catalog`).
        let root = temp_root();
        std::fs::write(root.join("catalog.sqlite"), b"not a sqlite file").unwrap();

        // Act
        let err = list_sessions(&root).unwrap_err();

        // Assert
        assert_eq!(err.kind, CatalogErrorKind::Sql);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_sessions_one_rebuilt_session_every_c3_field_matches_what_was_seeded() {
        // Arrange
        let root = temp_root();
        let mut doc = empty_session_json("s1");
        doc.rider = "Isaac".to_string();
        doc.venue_name = "Whistler".to_string();
        write_full_session(&root, "s1", 1_700_000_000_000, &doc);
        rebuild_catalog(&root).unwrap();

        // Act
        let out = list_sessions(&root).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        let s = &out[0];
        assert_eq!(s.session_id, "s1");
        assert_eq!(s.source_format, "idl0");
        assert_eq!(s.device_id.as_deref(), Some("d1"));
        assert_eq!(s.config_checksum.as_deref(), Some("cafef00d"));
        assert_eq!(s.timestamp_utc_ms, 1_700_000_000_000);
        assert_eq!(s.rider, "Isaac");
        assert_eq!(s.venue_name, "Whistler");
        assert_eq!(s.lap_count, Some(0));
        assert_eq!(s.duration_ms, Some(1_000));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_session_unknown_id_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = get_session(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, CatalogErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_session_seeded_session_channels_carry_unit_source_kind_and_channel_kind() {
        // Arrange
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 0, &doc);

        // Act
        let detail = get_session(&root, "s1").unwrap();

        // Assert
        assert_eq!(detail.channels.len(), 1);
        let ch = &detail.channels[0];
        assert_eq!(ch.unit, "g");
        assert_eq!(ch.source_kind, "imu0");
        assert_eq!(ch.channel_kind, "fixed-rate");
        assert_eq!(ch.sample_count, 3);
        assert_eq!(detail.source_format, "idl0");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_laps_session_with_two_laps_and_lap_summary_rows_stats_grouped_per_lap() {
        // Arrange
        let root = temp_root();
        let session_id = "sess-1";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![
            LapJson {
                lap_number: 1,
                start_timestamp_ms: 0,
                end_timestamp_ms: 500,
                raw_elapsed_ms: 500,
                lap_time_ms: 500,
                start_time_secs: 0.0,
                end_time_secs: 0.5,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
            LapJson {
                lap_number: 2,
                start_timestamp_ms: 500,
                end_timestamp_ms: 1_000,
                raw_elapsed_ms: 500,
                lap_time_ms: 500,
                start_time_secs: 0.5,
                end_time_secs: 1.0,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
        ];
        write_full_session(&root, session_id, 0, &doc);
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 500_000, 1_000_000],
            values: vec![1.0, 2.0, 3.0],
            nominal_rate_hz: 2.0,
            unit: "deg".to_string(),
        }];
        write_derived_parquet(&root, session_id, "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();
        rebuild_catalog(&root).unwrap();

        // Act
        let laps = list_laps(&root, session_id).unwrap();

        // Assert
        assert_eq!(laps.len(), 2);
        assert_eq!(laps[0].lap_number, 1);
        assert_eq!(laps[0].channel_stats.len(), 1);
        assert_eq!(laps[0].channel_stats[0].channel_id, "Roll (deg)");
        assert_eq!(laps[1].lap_number, 2);
        assert_eq!(laps[1].channel_stats.len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_laps_unknown_session_is_not_found() {
        // Arrange
        let root = temp_root();
        rebuild_catalog(&root).unwrap();

        // Act
        let err = list_laps(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, CatalogErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_workbooks_one_row_field_for_field() {
        // Arrange — `catalog.rs` step 6 leaves `workbooks` empty until L3's
        // `.idl1wb` front-matter parsing lands, so this test seeds the row
        // directly through the same catalog connection `list_workbooks` reads.
        let root = temp_root();
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        conn.execute(
            "INSERT INTO workbooks (workbook_id, file_name, name, updated_at_ms, size_bytes) VALUES (?1,?2,?3,?4,?5)",
            rusqlite::params!["wb-1", "session-review", "Session Review", 1234, 5678],
        )
        .unwrap();
        drop(conn);

        // Act
        let out = list_workbooks(&root).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].workbook_id, "wb-1");
        assert_eq!(out[0].file_name, "session-review");
        assert_eq!(out[0].name, "Session Review");
        assert_eq!(out[0].updated_at_ms, 1234);
        assert_eq!(out[0].size_bytes, 5678);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_tracks_one_row_field_for_field() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line",
            "venue_name":"Whistler","created_at_ms":1111,"updated_at_ms":2222}}"#;
        std::fs::write(tracks_dir.join("t-1.idl0t"), json).unwrap();
        rebuild_catalog(&root).unwrap();

        // Act
        let out = list_tracks(&root).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, "t-1");
        assert_eq!(out[0].name, "A-Line");
        assert_eq!(out[0].venue_name, "Whistler");
        assert_eq!(out[0].created_at_ms, 1111);
        assert_eq!(out[0].updated_at_ms, 2222);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_track_unknown_id_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = get_track(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, CatalogErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_track_seeded_artifact_scalars_and_opaque_fields() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line","venue_name":"Whistler",
            "lap_timing":{"kind":"circuit","name":"S/F","start_finish":{"lat1_deg":1,"lon1_deg":2,"lat2_deg":3,"lon2_deg":4,"name":""}},
            "sector_gates":[{"name":"S1","gate":{"lat1_deg":1,"lon1_deg":2,"lat2_deg":3,"lon2_deg":4,"name":""}}],
            "neutral_zones":[],"reference_polyline":[{"timestamp_ms":0,"latitude_deg":1,"longitude_deg":2}],
            "created_at_ms":1111,"updated_at_ms":2222}}"#;
        std::fs::write(tracks_dir.join("t-1.idl0t"), json).unwrap();

        // Act
        let detail = get_track(&root, "t-1").unwrap();

        // Assert
        assert_eq!(detail.track_id, "t-1");
        assert_eq!(detail.name, "A-Line");
        assert_eq!(detail.venue_name, "Whistler");
        assert_eq!(detail.created_at_ms, 1111);
        assert_eq!(detail.updated_at_ms, 2222);
        assert!(detail.lap_timing.is_object());
        assert_eq!(detail.sector_gates.as_array().unwrap().len(), 1);
        assert!(detail.neutral_zones.as_array().unwrap().is_empty());
        assert_eq!(detail.reference_polyline.as_array().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }
}
