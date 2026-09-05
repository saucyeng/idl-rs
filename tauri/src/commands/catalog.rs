//! Catalog commands (C3 §3.2): `list_sessions`, `get_session`, `list_laps`,
//! `rebuild_catalog`, `list_workbooks`, `list_tracks`, `get_track` — thin
//! wrappers over the new core read layer `idl_rs::store::catalog_read`
//! (ledger R40; L1 landed the catalog writer/rebuilder only).
//!
//! Same idiom as `device.rs`: each `#[tauri::command]` is a one-line
//! wrapper over a `_via`-suffixed plain function taking `data_dir: &Path`
//! (this module's own tests exercise the `_via` functions — `tauri::State`
//! cannot be constructed outside a running app). Response structs mirror
//! C3 §3.2 field for field with `From<idl_rs::store::catalog_read::X>`
//! impls; core's types are not `Serialize`, so they never cross the IPC
//! boundary directly.

use std::path::Path;

use idl_rs::store::catalog_read;

use crate::error::IpcError;
use crate::state::DataDir;

/// C3 §3.2 `SessionSummary` — mirrors the catalog `sessions` table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub blob_sha256: String,
    pub source_format: String,
    pub device_id: Option<String>,
    pub config_checksum: Option<String>,
    pub importer_version: String,
    pub seam_correction_version: String,
    pub engine_version: String,
    pub timestamp_utc_ms: i64,
    pub created_at_ms: i64,
    pub rider: String,
    pub bike: String,
    pub venue_name: String,
    pub event_name: String,
    pub event_session: String,
    pub short_comment: String,
    pub tag: String,
    pub lap_count: Option<u32>,
    pub duration_ms: Option<i64>,
}

impl From<catalog_read::SessionSummary> for SessionSummary {
    fn from(s: catalog_read::SessionSummary) -> Self {
        Self {
            session_id: s.session_id,
            blob_sha256: s.blob_sha256,
            source_format: s.source_format,
            device_id: s.device_id,
            config_checksum: s.config_checksum,
            importer_version: s.importer_version,
            seam_correction_version: s.seam_correction_version,
            engine_version: s.engine_version,
            timestamp_utc_ms: s.timestamp_utc_ms,
            created_at_ms: s.created_at_ms,
            rider: s.rider,
            bike: s.bike,
            venue_name: s.venue_name,
            event_name: s.event_name,
            event_session: s.event_session,
            short_comment: s.short_comment,
            tag: s.tag,
            lap_count: s.lap_count,
            duration_ms: s.duration_ms,
        }
    }
}

/// C3 §3.2 `ChannelSummary`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChannelSummary {
    pub channel_id: String,
    pub nominal_rate_hz: f64,
    pub unit: String,
    pub source_kind: String,
    pub channel_kind: String,
    pub sample_count: u64,
}

impl From<catalog_read::ChannelSummary> for ChannelSummary {
    fn from(c: catalog_read::ChannelSummary) -> Self {
        Self {
            channel_id: c.channel_id,
            nominal_rate_hz: c.nominal_rate_hz,
            unit: c.unit,
            source_kind: c.source_kind,
            channel_kind: c.channel_kind,
            sample_count: c.sample_count,
        }
    }
}

/// C3 §3.2 `LapDetail` (`session.json`'s own lap shape, distinct from
/// `LapSummary`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LapDetail {
    pub lap_number: u32,
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
    pub raw_elapsed_ms: i64,
    pub lap_time_ms: i64,
    pub start_time_secs: f64,
    pub end_time_secs: f64,
    pub sectors: serde_json::Value,
    pub neutral_zone_visits: serde_json::Value,
}

impl From<catalog_read::LapDetail> for LapDetail {
    fn from(l: catalog_read::LapDetail) -> Self {
        Self {
            lap_number: l.lap_number,
            start_timestamp_ms: l.start_timestamp_ms,
            end_timestamp_ms: l.end_timestamp_ms,
            raw_elapsed_ms: l.raw_elapsed_ms,
            lap_time_ms: l.lap_time_ms,
            start_time_secs: l.start_time_secs,
            end_time_secs: l.end_time_secs,
            sectors: l.sectors,
            neutral_zone_visits: l.neutral_zone_visits,
        }
    }
}

/// C3 §3.2 `TrackVisitSummary`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackVisitSummary {
    pub visit_id: String,
    pub track_id: String,
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
    pub laps: Vec<LapDetail>,
}

impl From<catalog_read::TrackVisitSummary> for TrackVisitSummary {
    fn from(v: catalog_read::TrackVisitSummary) -> Self {
        Self {
            visit_id: v.visit_id,
            track_id: v.track_id,
            start_timestamp_ms: v.start_timestamp_ms,
            end_timestamp_ms: v.end_timestamp_ms,
            laps: v.laps.into_iter().map(LapDetail::from).collect(),
        }
    }
}

/// C3 §3.2's `{ session_id, lap_number }` cross-session overlay reference.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OverlayLapKey {
    pub session_id: String,
    pub lap_number: u32,
}

impl From<idl_rs::store::session_json::OverlayLapKeyJson> for OverlayLapKey {
    fn from(k: idl_rs::store::session_json::OverlayLapKeyJson) -> Self {
        Self { session_id: k.session_id, lap_number: k.lap_number }
    }
}

/// C3 §3.2 `SessionDetail` — `get_session`'s return.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionDetail {
    pub session_id: String,
    pub device_id: Option<String>,
    pub timestamp_utc_ms: i64,
    pub config_checksum: Option<String>,
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
    pub bike_profile_snapshot: Option<serde_json::Value>,
    pub laps: Vec<LapDetail>,
    pub track_visits: Vec<TrackVisitSummary>,
    pub reference_lap_number: Option<u32>,
    pub ignored_lap_numbers: Vec<u32>,
    pub main_lap_number: Option<u32>,
    pub overlay_lap_key: Option<OverlayLapKey>,
    pub starred_lap_number: Option<u32>,
    pub track_visits_library_hash: Option<String>,
}

impl From<catalog_read::SessionDetail> for SessionDetail {
    fn from(d: catalog_read::SessionDetail) -> Self {
        Self {
            session_id: d.session_id,
            device_id: d.device_id,
            timestamp_utc_ms: d.timestamp_utc_ms,
            config_checksum: d.config_checksum,
            source_format: d.source_format,
            blob_sha256: d.blob_sha256,
            channels: d.channels.into_iter().map(ChannelSummary::from).collect(),
            rider: d.rider,
            bike: d.bike,
            bike_comment: d.bike_comment,
            venue_name: d.venue_name,
            event_name: d.event_name,
            event_session: d.event_session,
            short_comment: d.short_comment,
            long_comment: d.long_comment,
            tag: d.tag,
            bike_profile_snapshot: d.bike_profile_snapshot,
            laps: d.laps.into_iter().map(LapDetail::from).collect(),
            track_visits: d.track_visits.into_iter().map(TrackVisitSummary::from).collect(),
            reference_lap_number: d.reference_lap_number,
            ignored_lap_numbers: d.ignored_lap_numbers,
            main_lap_number: d.main_lap_number,
            overlay_lap_key: d.overlay_lap_key.map(OverlayLapKey::from),
            starred_lap_number: d.starred_lap_number,
            track_visits_library_hash: d.track_visits_library_hash,
        }
    }
}

/// C3 §3.2 `LapChannelStat`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LapChannelStat {
    pub channel_id: String,
    pub derived_hash: String,
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
}

impl From<catalog_read::LapChannelStat> for LapChannelStat {
    fn from(s: catalog_read::LapChannelStat) -> Self {
        Self { channel_id: s.channel_id, derived_hash: s.derived_hash, min_value: s.min_value, max_value: s.max_value, mean_value: s.mean_value }
    }
}

/// C3 §3.2 `LapSummary` — `list_laps`'s return (the catalog's cached
/// per-lap shape, distinct from `SessionDetail.laps`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LapSummary {
    pub lap_number: i32,
    pub lap_time_ms: i64,
    pub track_id: Option<String>,
    pub channel_stats: Vec<LapChannelStat>,
}

impl From<catalog_read::LapSummary> for LapSummary {
    fn from(s: catalog_read::LapSummary) -> Self {
        Self {
            lap_number: s.lap_number,
            lap_time_ms: s.lap_time_ms,
            track_id: s.track_id,
            channel_stats: s.channel_stats.into_iter().map(LapChannelStat::from).collect(),
        }
    }
}

/// C3 §3.2 `RebuildReport` — four fields only. Core's own `RebuildReport`
/// (`idl_rs::store::catalog::RebuildReport`) also carries `blobs_indexed`,
/// `laps_indexed`, `lap_summary_indexed` and `skipped`; those are dropped
/// at this boundary, not forwarded, since C3 §3.2 fixes this shape.
// TODO(idl0): core's `skipped` (non-fatal per-entity rebuild problems) has
// nowhere to go in C3 §3.2's `RebuildReport` shape — surfacing it needs a
// contract change (e.g. a `warnings: string[]` field), not fixed here.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RebuildReport {
    pub sessions_indexed: u32,
    pub workbooks_indexed: u32,
    pub tracks_indexed: u32,
    pub duration_ms: u64,
}

/// C3 §3.2 `WorkbookSummary`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkbookSummary {
    pub workbook_id: String,
    pub file_name: String,
    pub name: String,
    pub updated_at_ms: i64,
    pub size_bytes: u64,
}

impl From<catalog_read::WorkbookSummary> for WorkbookSummary {
    fn from(w: catalog_read::WorkbookSummary) -> Self {
        Self { workbook_id: w.workbook_id, file_name: w.file_name, name: w.name, updated_at_ms: w.updated_at_ms, size_bytes: w.size_bytes }
    }
}

/// C3 §3.2 `TrackSummary`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackSummary {
    pub track_id: String,
    pub name: String,
    pub venue_name: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

impl From<catalog_read::TrackSummary> for TrackSummary {
    fn from(t: catalog_read::TrackSummary) -> Self {
        Self { track_id: t.track_id, name: t.name, venue_name: t.venue_name, created_at_ms: t.created_at_ms, updated_at_ms: t.updated_at_ms }
    }
}

/// C3 §3.2 `TrackDetail` — `get_track`'s return. The four fields C3 leaves
/// `unknown` cross as raw JSON, exactly as `idl_rs::store::catalog_read`
/// already lifted them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackDetail {
    pub track_id: String,
    pub name: String,
    pub venue_name: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub lap_timing: serde_json::Value,
    pub neutral_zones: serde_json::Value,
    pub sector_gates: serde_json::Value,
    pub reference_polyline: serde_json::Value,
}

impl From<catalog_read::TrackDetail> for TrackDetail {
    fn from(t: catalog_read::TrackDetail) -> Self {
        Self {
            track_id: t.track_id,
            name: t.name,
            venue_name: t.venue_name,
            created_at_ms: t.created_at_ms,
            updated_at_ms: t.updated_at_ms,
            lap_timing: t.lap_timing,
            neutral_zones: t.neutral_zones,
            sector_gates: t.sector_gates,
            reference_polyline: t.reference_polyline,
        }
    }
}

/// Transport-agnostic core of `list_sessions`.
fn list_sessions_via(data_dir: &Path) -> Result<Vec<SessionSummary>, IpcError> {
    Ok(catalog_read::list_sessions(data_dir)?.into_iter().map(SessionSummary::from).collect())
}

/// Transport-agnostic core of `get_session`.
fn get_session_via(data_dir: &Path, session_id: &str) -> Result<SessionDetail, IpcError> {
    Ok(catalog_read::get_session(data_dir, session_id)?.into())
}

/// Transport-agnostic core of `list_laps`.
fn list_laps_via(data_dir: &Path, session_id: &str) -> Result<Vec<LapSummary>, IpcError> {
    Ok(catalog_read::list_laps(data_dir, session_id)?.into_iter().map(LapSummary::from).collect())
}

/// Transport-agnostic core of `rebuild_catalog`: times the core call itself
/// (C4 §5's rebuild has no `duration_ms` of its own) and maps to C3's
/// four-field shape.
fn rebuild_catalog_via(data_dir: &Path) -> Result<RebuildReport, IpcError> {
    let start = std::time::Instant::now();
    let report = catalog_read::rebuild_catalog_report(data_dir)?;
    let duration_ms = start.elapsed().as_millis() as u64;
    Ok(RebuildReport {
        sessions_indexed: report.sessions_indexed as u32,
        workbooks_indexed: report.workbooks_indexed as u32,
        tracks_indexed: report.tracks_indexed as u32,
        duration_ms,
    })
}

/// Transport-agnostic core of `list_workbooks`.
fn list_workbooks_via(data_dir: &Path) -> Result<Vec<WorkbookSummary>, IpcError> {
    Ok(catalog_read::list_workbooks(data_dir)?.into_iter().map(WorkbookSummary::from).collect())
}

/// Transport-agnostic core of `list_tracks`.
fn list_tracks_via(data_dir: &Path) -> Result<Vec<TrackSummary>, IpcError> {
    Ok(catalog_read::list_tracks(data_dir)?.into_iter().map(TrackSummary::from).collect())
}

/// Transport-agnostic core of `get_track`.
fn get_track_via(data_dir: &Path, track_id: &str) -> Result<TrackDetail, IpcError> {
    Ok(catalog_read::get_track(data_dir, track_id)?.into())
}

/// C3 §3.2 `list_sessions()`.
#[tauri::command]
pub fn list_sessions(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<SessionSummary>, IpcError> {
    list_sessions_via(&data_dir.0)
}

/// C3 §3.2 `get_session(session_id)`.
#[tauri::command]
pub fn get_session(session_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<SessionDetail, IpcError> {
    get_session_via(&data_dir.0, &session_id)
}

/// C3 §3.2 `list_laps(session_id)`.
#[tauri::command]
pub fn list_laps(session_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<Vec<LapSummary>, IpcError> {
    list_laps_via(&data_dir.0, &session_id)
}

/// C3 §3.2 `rebuild_catalog()`.
#[tauri::command]
pub fn rebuild_catalog(data_dir: tauri::State<'_, DataDir>) -> Result<RebuildReport, IpcError> {
    rebuild_catalog_via(&data_dir.0)
}

/// C3 §3.2 `list_workbooks()`.
#[tauri::command]
pub fn list_workbooks(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<WorkbookSummary>, IpcError> {
    list_workbooks_via(&data_dir.0)
}

/// C3 §3.2 `list_tracks()`.
#[tauri::command]
pub fn list_tracks(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<TrackSummary>, IpcError> {
    list_tracks_via(&data_dir.0)
}

/// C3 §3.2 `get_track(track_id)`.
#[tauri::command]
pub fn get_track(track_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<TrackDetail, IpcError> {
    get_track_via(&data_dir.0, &track_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat};
    use idl_rs::store::blob::write_blob;
    use idl_rs::store::catalog::rebuild_catalog as core_rebuild_catalog;
    use idl_rs::store::parquet::write_session_parquet;
    use idl_rs::store::session_json::{
        empty_session_json, write_session_json, LapJson, NeutralZoneVisitJson, OverlayLapKeyJson, SectorJson,
        TrackVisitJson,
    };
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_full_session(root: &Path, session_id: &str) {
        let blob_sha256 = write_blob(root, format!("raw bytes for {session_id}").as_bytes()).unwrap();
        let doc = empty_session_json(session_id);
        write_session_json(root, session_id, &doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256,
            channels: vec![Channel {
                channel_id: "IMU0_AccelX".to_string(),
                t_us: vec![0, 500_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn list_sessions_via_happy_path_returns_the_seeded_session() {
        // Arrange
        let root = temp_root();
        write_full_session(&root, "s1");
        core_rebuild_catalog(&root).unwrap();

        // Act
        let out = list_sessions_via(&root).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].session_id, "s1");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_session_via_unknown_id_ipc_error_kind_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = get_session_via(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_session_via_happy_path_every_field_of_the_24_field_conversion_is_distinguishable() {
        // Arrange — every field below holds a value no other field shares
        // (within its own type), so a transposition or dropped field in
        // `SessionDetail::from(catalog_read::SessionDetail)` (the largest
        // `From` impl in this file) fails a specific assertion rather than
        // slipping past a spot-check.
        let root = temp_root();
        let session_id = "s1";

        let mut doc = empty_session_json(session_id);
        doc.rider = "rider-A".to_string();
        doc.bike = "bike-B".to_string();
        doc.bike_comment = "bike-comment-C".to_string();
        doc.venue_name = "venue-D".to_string();
        doc.event_name = "event-E".to_string();
        doc.event_session = "event-session-F".to_string();
        doc.short_comment = "short-comment-G".to_string();
        doc.long_comment = "long-comment-H".to_string();
        doc.tag = "tag-I".to_string();
        doc.bike_profile_snapshot = Some(serde_json::json!({ "snapshot_marker": "snapshot-J" }));
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 1_000,
            end_timestamp_ms: 2_000,
            raw_elapsed_ms: 1_001,
            lap_time_ms: 1_002,
            start_time_secs: 1.1,
            end_time_secs: 2.2,
            sectors: vec![SectorJson {
                name: "sector-K".to_string(),
                start_ms: 1_100,
                end_ms: 1_200,
                start_time_secs: 1.3,
                end_time_secs: 1.4,
            }],
            neutral_zone_visits: vec![NeutralZoneVisitJson { name: "nz-L".to_string(), enter_ms: 1_300, exit_ms: 1_400 }],
        }];
        doc.track_visits = vec![TrackVisitJson {
            visit_id: "visit-M".to_string(),
            track_id: "track-N".to_string(),
            start_timestamp_ms: 3_000,
            end_timestamp_ms: 4_000,
            laps: vec![LapJson {
                lap_number: 20,
                start_timestamp_ms: 3_100,
                end_timestamp_ms: 3_200,
                raw_elapsed_ms: 3_101,
                lap_time_ms: 3_102,
                start_time_secs: 3.3,
                end_time_secs: 3.4,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            }],
        }];
        doc.reference_lap_number = Some(11);
        doc.ignored_lap_numbers = vec![12, 13];
        doc.main_lap_number = Some(14);
        doc.overlay_lap_key = Some(OverlayLapKeyJson { session_id: "overlay-session-O".to_string(), lap_number: 15 });
        doc.starred_lap_number = Some(16);
        doc.track_visits_library_hash = Some("hash-P".to_string());
        write_session_json(&root, session_id, &doc, None).unwrap();

        let blob_sha256 = write_blob(&root, b"raw bytes for s1 (distinguishable test)").unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: Some("device-Q".to_string()),
            timestamp_utc_ms: 5_000,
            config_checksum: Some("checksum-R".to_string()),
            source_format: SourceFormat::Idl0,
            blob_sha256: blob_sha256.clone(),
            channels: vec![Channel {
                channel_id: "chan-S".to_string(),
                t_us: vec![0, 500_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let detail = get_session_via(&root, session_id).unwrap();

        // Assert
        assert_eq!(detail.session_id, "s1");
        assert_eq!(detail.device_id.as_deref(), Some("device-Q"));
        assert_eq!(detail.timestamp_utc_ms, 5_000);
        assert_eq!(detail.config_checksum.as_deref(), Some("checksum-R"));
        assert_eq!(detail.source_format, "idl0");
        assert_eq!(detail.blob_sha256, blob_sha256);

        assert_eq!(detail.channels.len(), 1);
        let ch = &detail.channels[0];
        assert_eq!(ch.channel_id, "chan-S");
        assert_eq!(ch.nominal_rate_hz, 2.0);
        assert_eq!(ch.unit, "g");
        assert_eq!(ch.source_kind, "imu0");
        assert_eq!(ch.channel_kind, "fixed-rate");
        assert_eq!(ch.sample_count, 2);

        assert_eq!(detail.rider, "rider-A");
        assert_eq!(detail.bike, "bike-B");
        assert_eq!(detail.bike_comment, "bike-comment-C");
        assert_eq!(detail.venue_name, "venue-D");
        assert_eq!(detail.event_name, "event-E");
        assert_eq!(detail.event_session, "event-session-F");
        assert_eq!(detail.short_comment, "short-comment-G");
        assert_eq!(detail.long_comment, "long-comment-H");
        assert_eq!(detail.tag, "tag-I");
        assert_eq!(detail.bike_profile_snapshot, Some(serde_json::json!({ "snapshot_marker": "snapshot-J" })));

        assert_eq!(detail.laps.len(), 1);
        let lap = &detail.laps[0];
        assert_eq!(lap.lap_number, 1);
        assert_eq!(lap.start_timestamp_ms, 1_000);
        assert_eq!(lap.end_timestamp_ms, 2_000);
        assert_eq!(lap.raw_elapsed_ms, 1_001);
        assert_eq!(lap.lap_time_ms, 1_002);
        assert_eq!(lap.start_time_secs, 1.1);
        assert_eq!(lap.end_time_secs, 2.2);
        assert_eq!(
            lap.sectors,
            serde_json::json!([{ "name": "sector-K", "start_ms": 1_100, "end_ms": 1_200, "start_time_secs": 1.3, "end_time_secs": 1.4 }])
        );
        assert_eq!(lap.neutral_zone_visits, serde_json::json!([{ "name": "nz-L", "enter_ms": 1_300, "exit_ms": 1_400 }]));

        assert_eq!(detail.track_visits.len(), 1);
        let visit = &detail.track_visits[0];
        assert_eq!(visit.visit_id, "visit-M");
        assert_eq!(visit.track_id, "track-N");
        assert_eq!(visit.start_timestamp_ms, 3_000);
        assert_eq!(visit.end_timestamp_ms, 4_000);
        assert_eq!(visit.laps.len(), 1);
        let nested_lap = &visit.laps[0];
        assert_eq!(nested_lap.lap_number, 20);
        assert_eq!(nested_lap.start_timestamp_ms, 3_100);
        assert_eq!(nested_lap.end_timestamp_ms, 3_200);
        assert_eq!(nested_lap.raw_elapsed_ms, 3_101);
        assert_eq!(nested_lap.lap_time_ms, 3_102);
        assert_eq!(nested_lap.start_time_secs, 3.3);
        assert_eq!(nested_lap.end_time_secs, 3.4);

        assert_eq!(detail.reference_lap_number, Some(11));
        assert_eq!(detail.ignored_lap_numbers, vec![12, 13]);
        assert_eq!(detail.main_lap_number, Some(14));
        assert_eq!(detail.overlay_lap_key.as_ref().map(|k| k.session_id.as_str()), Some("overlay-session-O"));
        assert_eq!(detail.overlay_lap_key.as_ref().map(|k| k.lap_number), Some(15));
        assert_eq!(detail.starred_lap_number, Some(16));
        assert_eq!(detail.track_visits_library_hash.as_deref(), Some("hash-P"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_laps_via_happy_path_returns_an_empty_vec_for_a_lapless_session() {
        // Arrange
        let root = temp_root();
        write_full_session(&root, "s1");
        core_rebuild_catalog(&root).unwrap();

        // Act
        let out = list_laps_via(&root, "s1").unwrap();

        // Assert
        assert!(out.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_via_happy_path_reports_one_session_and_a_duration() {
        // Arrange
        let root = temp_root();
        write_full_session(&root, "s1");

        // Act
        let report = rebuild_catalog_via(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 1);
        assert_eq!(report.workbooks_indexed, 0);
        assert_eq!(report.tracks_indexed, 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_workbooks_via_happy_path_returns_an_empty_vec_on_a_fresh_catalog() {
        // Arrange
        let root = temp_root();
        core_rebuild_catalog(&root).unwrap();

        // Act
        let out = list_workbooks_via(&root).unwrap();

        // Assert
        assert!(out.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_tracks_via_happy_path_returns_the_seeded_track() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        std::fs::write(
            tracks_dir.join("t-1.idl0t"),
            r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line","created_at_ms":1,"updated_at_ms":2}}"#,
        )
        .unwrap();
        core_rebuild_catalog(&root).unwrap();

        // Act
        let out = list_tracks_via(&root).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].track_id, "t-1");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_track_via_happy_path_returns_the_seeded_artifact() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        std::fs::write(
            tracks_dir.join("t-1.idl0t"),
            r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line","created_at_ms":1,"updated_at_ms":2}}"#,
        )
        .unwrap();

        // Act
        let detail = get_track_via(&root, "t-1").unwrap();

        // Assert
        assert_eq!(detail.track_id, "t-1");
        assert_eq!(detail.name, "A-Line");

        let _ = std::fs::remove_dir_all(&root);
    }
}
