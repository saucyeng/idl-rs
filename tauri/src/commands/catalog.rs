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

use std::path::{Path, PathBuf};

use idl_rs::store::catalog_read;

use crate::error::{IpcError, IpcErrorKind};
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

/// C3 §3.2 `LapDetail.sectors` element (C1 §6 `laps[].sectors`, C3 §6 item
/// 11, closed 2026-09-06). Mirrors core's `session_json::SectorJson`
/// field for field rather than deriving `Serialize` on the core type
/// directly, matching this module's own idiom (core types never cross the
/// IPC boundary directly).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LapSector {
    pub name: String,
    /// Unix epoch milliseconds.
    pub start_ms: i64,
    pub end_ms: i64,
    /// Seconds, recording-time (t=0-anchored).
    pub start_time_secs: f64,
    pub end_time_secs: f64,
}

impl From<idl_rs::store::session_json::SectorJson> for LapSector {
    fn from(s: idl_rs::store::session_json::SectorJson) -> Self {
        Self {
            name: s.name,
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            start_time_secs: s.start_time_secs,
            end_time_secs: s.end_time_secs,
        }
    }
}

/// C3 §3.2 `LapDetail.neutral_zone_visits` element (C1 §6
/// `laps[].neutral_zone_visits`, C3 §6 item 11, closed 2026-09-06).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LapNeutralZoneVisit {
    pub name: String,
    /// Unix epoch milliseconds.
    pub enter_ms: i64,
    pub exit_ms: i64,
}

impl From<idl_rs::store::session_json::NeutralZoneVisitJson> for LapNeutralZoneVisit {
    fn from(v: idl_rs::store::session_json::NeutralZoneVisitJson) -> Self {
        Self { name: v.name, enter_ms: v.enter_ms, exit_ms: v.exit_ms }
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
    pub sectors: Vec<LapSector>,
    pub neutral_zone_visits: Vec<LapNeutralZoneVisit>,
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
            sectors: l.sectors.into_iter().map(LapSector::from).collect(),
            neutral_zone_visits: l.neutral_zone_visits.into_iter().map(LapNeutralZoneVisit::from).collect(),
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

/// C3 §3.2 `Gate` — decimal degrees. The `.idl0t` file stores degrees x1e7
/// (SPEC §17b.1); that scaling is a wire detail of the file, never of the
/// IPC surface — core's `laps::model::Gate` is already decimal degrees
/// (ruling R27). `Deserialize` (added Task 4) lets this double as
/// `TrackDraft`'s wire mirror — `save_track`'s input needs the same shape
/// `get_track` returns, just read instead of written.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateWire {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
}

impl From<&idl_rs::laps::model::Gate> for GateWire {
    fn from(g: &idl_rs::laps::model::Gate) -> Self {
        Self { lat1: g.lat1, lon1: g.lon1, lat2: g.lat2, lon2: g.lon2 }
    }
}

/// Inverse of `From<&Gate> for GateWire` — the wire is already decimal
/// degrees (no x1e7 rescale; that only happens at the `.idl0t` file
/// boundary in `track_artifact::model`), so this is a plain field copy.
impl From<&GateWire> for idl_rs::laps::model::Gate {
    fn from(g: &GateWire) -> Self {
        Self { lat1: g.lat1, lon1: g.lon1, lat2: g.lat2, lon2: g.lon2 }
    }
}

/// C3 §3.2 `SectorGate`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SectorGateWire {
    pub name: String,
    pub gate: GateWire,
}

impl From<&idl_rs::laps::model::SectorGate> for SectorGateWire {
    fn from(s: &idl_rs::laps::model::SectorGate) -> Self {
        Self { name: s.name.clone(), gate: (&s.gate).into() }
    }
}

impl From<&SectorGateWire> for idl_rs::laps::model::SectorGate {
    fn from(s: &SectorGateWire) -> Self {
        Self { name: s.name.clone(), gate: (&s.gate).into() }
    }
}

/// C3 §3.2 `NeutralZone`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeutralZoneWire {
    pub name: String,
    pub enter: GateWire,
    pub exit: GateWire,
}

impl From<&idl_rs::laps::model::NeutralZone> for NeutralZoneWire {
    fn from(z: &idl_rs::laps::model::NeutralZone) -> Self {
        Self { name: z.name.clone(), enter: (&z.enter).into(), exit: (&z.exit).into() }
    }
}

impl From<&NeutralZoneWire> for idl_rs::laps::model::NeutralZone {
    fn from(z: &NeutralZoneWire) -> Self {
        Self { name: z.name.clone(), enter: (&z.enter).into(), exit: (&z.exit).into() }
    }
}

/// C3 §3.2 `GpsFix`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpsFixWire {
    pub timestamp_ms: i64,
    pub lat: f64,
    pub lon: f64,
}

impl From<&idl_rs::gps::GpsFix> for GpsFixWire {
    fn from(f: &idl_rs::gps::GpsFix) -> Self {
        Self { timestamp_ms: f.timestamp_ms, lat: f.lat, lon: f.lon }
    }
}

impl From<&GpsFixWire> for idl_rs::gps::GpsFix {
    fn from(f: &GpsFixWire) -> Self {
        Self { timestamp_ms: f.timestamp_ms, lat: f.lat, lon: f.lon }
    }
}

/// C3 §3.2 `LapTiming` — a sealed union, `kind: "circuit" | "point_to_point"`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LapTimingWire {
    Circuit { start_finish: GateWire },
    PointToPoint { start: GateWire, finish: GateWire },
}

impl From<&idl_rs::laps::model::LapTiming> for LapTimingWire {
    fn from(t: &idl_rs::laps::model::LapTiming) -> Self {
        match t {
            idl_rs::laps::model::LapTiming::Circuit { start_finish } => {
                LapTimingWire::Circuit { start_finish: start_finish.into() }
            }
            idl_rs::laps::model::LapTiming::PointToPoint { start, finish } => {
                LapTimingWire::PointToPoint { start: start.into(), finish: finish.into() }
            }
        }
    }
}

impl From<&LapTimingWire> for idl_rs::laps::model::LapTiming {
    fn from(t: &LapTimingWire) -> Self {
        match t {
            LapTimingWire::Circuit { start_finish } => Self::Circuit { start_finish: start_finish.into() },
            LapTimingWire::PointToPoint { start, finish } => Self::PointToPoint { start: start.into(), finish: finish.into() },
        }
    }
}

/// C3 §3.2 `TrackDetail` — `get_track`'s return. **REVISED (2026-09-06, L8x,
/// ruling R86)** — the four fields C3 §6 open question 10 left `unknown` are
/// now typed, decimal degrees on the wire.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackDetail {
    pub track_id: String,
    pub name: String,
    pub venue_name: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub lap_timing: Option<LapTimingWire>,
    pub neutral_zones: Vec<NeutralZoneWire>,
    pub sector_gates: Vec<SectorGateWire>,
    pub reference_polyline: Vec<GpsFixWire>,
}

impl From<catalog_read::TrackDetail> for TrackDetail {
    fn from(t: catalog_read::TrackDetail) -> Self {
        Self {
            track_id: t.track_id,
            name: t.name,
            venue_name: t.venue_name,
            created_at_ms: t.created_at_ms,
            updated_at_ms: t.updated_at_ms,
            lap_timing: t.lap_timing.as_ref().map(Into::into),
            neutral_zones: t.neutral_zones.iter().map(Into::into).collect(),
            sector_gates: t.sector_gates.iter().map(Into::into).collect(),
            reference_polyline: t.reference_polyline.iter().map(Into::into).collect(),
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

/// C3 §3.2 `TrackDraft` — `save_track`'s argument (ruling R86). `track_id:
/// None` creates (the command mints a UUID v4 and both timestamps); `Some`
/// edits an existing track, preserving `created_at_ms` and bumping
/// `updated_at_ms` to now.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrackDraft {
    pub track_id: Option<String>,
    pub name: String,
    pub venue_name: String,
    pub lap_timing: Option<LapTimingWire>,
    pub neutral_zones: Vec<NeutralZoneWire>,
    pub sector_gates: Vec<SectorGateWire>,
    pub reference_polyline: Vec<GpsFixWire>,
}

/// C3 §3.2 `SaveTrackResult` — `save_track`'s return.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SaveTrackResult {
    pub track: TrackDetail,
    /// Sessions whose `track_visits_library_hash` no longer matches the
    /// library after this write. The UI offers `rescan_tracks` per id.
    pub stale_session_ids: Vec<String>,
    pub warnings: Vec<String>,
}

/// Maps a [`idl_rs::track_artifact::write::TrackWriteError`] to `IpcError`.
/// `Io` (a filesystem failure, including exhausting C4 §4's atomic-write
/// retry) maps straight through; `Encode` (the track failed to serialise to
/// JSON — a programmer-error condition, never a caller mistake) folds into
/// the cross-cutting `Internal`, matching `CatalogErrorKind::Sql`'s own
/// precedent (ruling R46) for "a genuine failure, not the caller's fault".
/// No new `IpcErrorKind` (this task's `Do not`).
fn map_track_write_error(e: idl_rs::track_artifact::write::TrackWriteError) -> IpcError {
    use idl_rs::track_artifact::write::TrackWriteErrorKind;
    match e.kind {
        TrackWriteErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        TrackWriteErrorKind::Encode => IpcError::new(IpcErrorKind::Internal, e.message),
    }
}

/// Converts a [`TrackDraft`] into the core domain `Track`, given the id and
/// both timestamps the caller (`save_track_via`) has already resolved. The
/// wire mirrors are already decimal degrees (no x1e7 rescale — that only
/// happens at the `.idl0t` file boundary in `track_artifact::model`), so
/// every field conversion here is a plain, unscaled copy via the `From<&*
/// Wire>` impls above.
fn draft_into_track(draft: &TrackDraft, id: String, created_at_ms: i64, updated_at_ms: i64) -> idl_rs::track_artifact::Track {
    idl_rs::track_artifact::Track {
        id,
        name: draft.name.clone(),
        venue: draft.venue_name.clone(),
        timing: draft.lap_timing.as_ref().map(Into::into),
        sector_gates: draft.sector_gates.iter().map(Into::into).collect(),
        neutral_zones: draft.neutral_zones.iter().map(Into::into).collect(),
        reference_polyline: draft.reference_polyline.iter().map(Into::into).collect(),
        created_at_ms,
        updated_at_ms,
    }
}

/// Every `sessions/<id>/session.json` whose `track_visits_library_hash`
/// stamp no longer matches `current_hash` (C3 §3.2 `SaveTrackResult.
/// stale_session_ids`). A session with no stamp yet (`None` — never
/// rescanned against any track library) is not counted stale: there is no
/// prior lap/visit computation for this write to have invalidated. A
/// `session.json` that fails to read is folded into `warnings`, not a
/// failure of the whole call (mirrors `rescan_tracks_via`'s own catalog-
/// failure handling).
fn stale_session_ids(data_dir: &Path, current_hash: &str, warnings: &mut Vec<String>) -> Vec<String> {
    let sessions_dir = data_dir.join("sessions");
    let mut stale = Vec::new();
    for dir in std::fs::read_dir(&sessions_dir).into_iter().flatten().filter_map(|e| e.ok()).map(|e| e.path()) {
        if !dir.is_dir() {
            continue;
        }
        let session_id = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        let sj_path = dir.join("session.json");
        match idl_rs::store::session_json::read_session_json(&sj_path) {
            Ok(doc) => {
                if let Some(stamp) = &doc.track_visits_library_hash {
                    if stamp != current_hash {
                        stale.push(session_id);
                    }
                }
            }
            Err(e) => warnings.push(format!("{}: {e}", sj_path.display())),
        }
    }
    stale
}

/// Transport-agnostic core of `save_track` (C3 §3.2, ruling R86). One
/// command for create and edit (PLAN Q4): `draft.track_id: None` mints `id`
/// from `new_id` (a UUID v4 in canonical form, minted by the
/// `#[tauri::command]` wrapper) and stamps both timestamps from `now_ms`;
/// `Some(id)` requires the artifact to already exist (`not_found` otherwise),
/// keeps its `created_at_ms` verbatim, and bumps only `updated_at_ms`. Never
/// lets the caller set either timestamp directly. Validates before any
/// filesystem write (`invalid_argument`, carrying the failed field in
/// `detail`), then writes through the already-landed `write_track` (C4 §4
/// atomic write, last-write-wins — no `conflict` kind, consistent with
/// ruling R59 Q1(a)). When `catalog.sqlite` already exists, upserts the one
/// `tracks` row afterward (`store::catalog::upsert_track`); a catalog
/// failure is folded into `warnings`, never fails the call. Does not call
/// `rescan_tracks` (PLAN §4) — instead recomputes `track_library_hash` over
/// the post-write library and returns every stale session id for the UI to
/// offer a rescan.
fn save_track_via(data_dir: &Path, draft: TrackDraft, now_ms: i64, new_id: &str) -> Result<SaveTrackResult, IpcError> {
    let (id, created_at_ms) = match &draft.track_id {
        None => (new_id.to_string(), now_ms),
        Some(existing_id) => {
            let existing = catalog_read::get_track(data_dir, existing_id)?;
            (existing_id.clone(), existing.created_at_ms)
        }
    };

    let track = draft_into_track(&draft, id.clone(), created_at_ms, now_ms);

    idl_rs::track_artifact::validate_track(&track).map_err(|e| {
        IpcError::with_detail(IpcErrorKind::InvalidArgument, e.message, serde_json::json!({ "field": e.field }))
    })?;

    let written_path = idl_rs::track_artifact::write_track(data_dir, &track).map_err(map_track_write_error)?;

    let mut warnings = Vec::new();
    let catalog_path = data_dir.join("catalog.sqlite");
    if catalog_path.is_file() {
        let upserted = std::fs::read_to_string(&written_path)
            .map_err(|e| e.to_string())
            .and_then(|full_json| {
                idl_rs::store::catalog::open_catalog(&catalog_path)
                    .map_err(|e| e.to_string())
                    .and_then(|conn| idl_rs::store::catalog::upsert_track(&conn, &track, &full_json).map_err(|e| e.to_string()))
            });
        if let Err(e) = upserted {
            warnings.push(e);
        }
    }

    let (library, _library_warnings) = idl_rs::store::lap_index::load_track_library(data_dir)?;
    let current_hash = idl_rs::store::lap_index::track_library_hash(&library);
    let stale = stale_session_ids(data_dir, &current_hash, &mut warnings);

    Ok(SaveTrackResult { track: catalog_read::get_track(data_dir, &id)?.into(), stale_session_ids: stale, warnings })
}

/// C3 §3.2 `save_track(track)`. Mints the UUID v4 and `now_ms` here (the
/// only impure inputs `save_track_via` needs), keeping every core/command
/// function underneath deterministic and testable without a clock.
#[tauri::command]
pub fn save_track(track: TrackDraft, data_dir: tauri::State<'_, DataDir>) -> Result<SaveTrackResult, IpcError> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let new_id = uuid::Uuid::new_v4().to_string();
    save_track_via(&data_dir.0, track, now_ms, &new_id)
}

/// C3 §3.2 `DeleteTrackReport` — `delete_track`'s return.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeleteTrackReport {
    pub track_id: String,
    /// Sessions whose `track_visits_library_hash` no longer matches the
    /// library after the delete — their cached visits may name this track.
    /// The UI offers `rescan_tracks` per id; nothing is rewritten here.
    pub stale_session_ids: Vec<String>,
    pub warnings: Vec<String>,
}

/// Transport-agnostic core of `delete_track` (C3 §3.2, ruling R86).
/// Deliberately does **not** rewrite any `session.json`: idl0 left stale
/// `TrackVisit` references behind a delete too
/// (`track_provider.dart`'s `deleteTrack` note, SPEC §12.3), the hierarchy
/// view already skips visits whose `track_id` no longer resolves, and
/// "Rescan tracks" is the user-driven repair. Rewriting every session
/// inside a delete would be unbounded work behind one button; this command
/// instead returns `stale_session_ids` for the UI to offer that rescan.
///
/// An absent `tracks/<id>.idl0t` is `not_found`, checked **first** — before
/// any catalog work — matching `delete_session_via`. `store::track_artifact::
/// write::delete_track` removes the artifact; then, only when
/// `catalog.sqlite` is a file, the one `tracks` row is deleted
/// (`store::catalog::delete_track`). `laps.track_id` is already `REFERENCES
/// tracks(track_id) ON DELETE SET NULL` (C4 §5), so lap rows survive
/// unattributed — this function deletes no `laps` rows itself. A catalog
/// failure is folded into `warnings`, never fails the call, mirroring
/// `save_track_via`.
fn delete_track_via(data_dir: &Path, track_id: &str) -> Result<DeleteTrackReport, IpcError> {
    let removed = idl_rs::track_artifact::delete_track(data_dir, track_id).map_err(map_track_write_error)?;
    if !removed {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("track {track_id} not found")));
    }

    let mut warnings = Vec::new();
    let catalog_path = data_dir.join("catalog.sqlite");
    if catalog_path.is_file() {
        let deleted = idl_rs::store::catalog::open_catalog(&catalog_path)
            .map_err(|e| e.to_string())
            .and_then(|conn| idl_rs::store::catalog::delete_track(&conn, track_id).map_err(|e| e.to_string()));
        if let Err(e) = deleted {
            warnings.push(e);
        }
    }

    let (library, _library_warnings) = idl_rs::store::lap_index::load_track_library(data_dir)?;
    let current_hash = idl_rs::store::lap_index::track_library_hash(&library);
    let stale = stale_session_ids(data_dir, &current_hash, &mut warnings);

    Ok(DeleteTrackReport { track_id: track_id.to_string(), stale_session_ids: stale, warnings })
}

/// C3 §3.2 `delete_track(track_id)`.
#[tauri::command]
pub fn delete_track(track_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<DeleteTrackReport, IpcError> {
    delete_track_via(&data_dir.0, &track_id)
}

/// C3 §3.2 `rescan_tracks`'s return (IDL0_SPEC §17.4 "Rescan Tracks").
#[derive(Debug, Clone, serde::Serialize)]
pub struct RescanReport {
    pub session_id: String,
    pub visits_indexed: u32,
    pub laps_indexed: u32,
    /// Lap-flag fields cleared because their lap number no longer exists
    /// after renumbering (PLAN Q3) — the UI warns the rider that a starred
    /// or ignored lap was dropped.
    pub flags_cleared: Vec<String>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u32,
}

/// Transport-agnostic core of `rescan_tracks`: re-runs visit/lap detection
/// for one session against the current track library
/// (`idl_rs::store::lap_index::reindex_laps`, which always recomputes) and,
/// when `catalog.sqlite` already exists, re-indexes that session's catalog
/// rows (`idl_rs::store::catalog::index_session`) — mirroring
/// `store::import::finish_import`'s own catalog rule (Task 4): a bare data
/// root that has never had `rebuild_catalog` run against it stays
/// catalog-less, and this call is not the thing that creates one. A
/// catalog-indexing failure is folded into `warnings` rather than failing
/// the call, for the same reason: the lap rescan itself already succeeded
/// and `rebuild_catalog` remains the recovery path. An unknown `session_id`
/// (no `sessions/<id>/` directory) is `IpcErrorKind::NotFound`, matching
/// `get_session`/`list_laps`, checked before `reindex_laps` runs so a
/// missing session never surfaces as the less specific `io` a missing
/// `data.parquet` would otherwise produce.
fn rescan_tracks_via(data_dir: &Path, session_id: &str) -> Result<RescanReport, IpcError> {
    let start = std::time::Instant::now();

    let session_dir = data_dir.join("sessions").join(session_id);
    if !session_dir.is_dir() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session {session_id} not found")));
    }

    let report = idl_rs::store::lap_index::reindex_laps(data_dir, session_id)?;

    let mut warnings = report.warnings;
    let catalog_path = data_dir.join("catalog.sqlite");
    if catalog_path.is_file() {
        let indexed = idl_rs::store::catalog::open_catalog(&catalog_path)
            .and_then(|conn| idl_rs::store::catalog::index_session(&conn, data_dir, session_id));
        if let Err(e) = indexed {
            warnings.push(e.to_string());
        }
    }

    Ok(RescanReport {
        session_id: session_id.to_string(),
        visits_indexed: report.visits_indexed as u32,
        laps_indexed: report.laps_indexed as u32,
        flags_cleared: report.flags_cleared,
        warnings,
        elapsed_ms: start.elapsed().as_millis() as u32,
    })
}

/// C3 §3.2 `rescan_tracks(session_id)` — IDL0_SPEC §17.4's "Rescan Tracks".
/// Re-runs visit and lap detection for one session against the current
/// track library, rewrites its `session.json`, and re-indexes its catalog
/// rows when a catalog exists.
#[tauri::command]
pub fn rescan_tracks(session_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<RescanReport, IpcError> {
    rescan_tracks_via(&data_dir.0, &session_id)
}

/// C3 §3.2 `SessionMetadataPatch` — `save_session_metadata`'s argument. A
/// typed struct, not a JSON bag — an unknown key or a non-string value for
/// a known key is a Tauri-level argument-deserialisation rejection, never
/// reaches this command's body. C3's "unknown keys ignored" and
/// "invalid_argument for a non-string field" wording describes the
/// JSON-bag case this typed argument makes structurally moot.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SessionMetadataPatch {
    pub rider: String,
    pub bike: String,
    pub bike_comment: String,
    pub venue_name: String,
    pub event_name: String,
    pub event_session: String,
    pub short_comment: String,
    pub long_comment: String,
    pub tag: String,
}

/// Maps a `session.json` read/write failure to `IpcError` for both of this
/// task's commands. `Io` (a filesystem failure, including
/// `write_session_json`'s `RenameConflict` — folded to `Io` upstream by
/// `SessionJsonError`'s own `From<AtomicWriteError>` impl — R59 Q1(a) is
/// explicit that this pair of commands never raises `Conflict`) maps
/// straight through. `Parse`/`UnsupportedVersion` fold to `Internal`
/// (deliberate — lead ruling 2026-09-05: C3 §3.2 lists only `not_found`/
/// `io`/`internal` for these two commands, and a malformed `session.json`
/// on an existing session directory is a data-integrity condition, not a
/// caller argument problem; `path` is appended to the message here because
/// neither `parse_config`'s nor `SessionJsonError`'s own message carries it
/// for the `Parse`/`UnsupportedVersion` cases, and the lead ruling requires
/// the parse reason and the path both be diagnosable from `IpcError.message`).
/// `InvalidArgument` (added by R194's `set_session_start`, whose IPC glue is
/// not this task's — that command is C3 §3.3's own `invalid_argument`) maps
/// straight through for exhaustiveness; neither of this fn's two callers can
/// produce it today.
fn map_session_json_error(e: idl_rs::store::session_json::SessionJsonError, path: &Path) -> IpcError {
    use idl_rs::store::session_json::SessionJsonErrorKind;
    match e.kind {
        SessionJsonErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        SessionJsonErrorKind::Parse | SessionJsonErrorKind::UnsupportedVersion => {
            IpcError::new(IpcErrorKind::Internal, format!("{}: {}", path.display(), e.message))
        }
        SessionJsonErrorKind::InvalidArgument => IpcError::new(IpcErrorKind::InvalidArgument, e.message),
    }
}

/// Transport-agnostic core of `save_session_metadata` (C3 §3.2). Reads
/// `session.json`, replaces exactly the nine editable fields named in
/// `SessionMetadataPatch`, and writes it back through
/// `store::session_json::write_session_json` (C4 §4 atomic write) with a
/// `based_on_hash` this function computes itself by hashing the bytes it
/// just read — the read-hash-write optimistic-concurrency check lives
/// entirely inside this command, last-write-wins, never raising `Conflict`
/// (ruling R59 Q1(a)). Every other `session.json` key (`laps`,
/// `track_visits`, the lap-flag fields, `bike_profile_snapshot`,
/// `schema_version`) is left untouched. Re-reads and returns
/// `catalog_read::get_session`'s canonical `SessionDetail` afterward, so
/// the caller redraws from what was actually written rather than an echo
/// of the argument. The catalog's `sessions` row is **not** re-indexed by
/// this command; `rebuild_catalog` reconciles it later (C4 §5).
///
/// The not-found check (`session_dir.is_dir()`) duplicates
/// `catalog_read::get_session`'s own identical check by design: this
/// function needs the directory to exist before it can call
/// `read_session_json`/`write_session_json`, and calling `get_session`
/// first only to discard its result and re-read `session.json` a second
/// time would be strictly more I/O for no benefit. Both checks test the
/// exact same condition (`sessions/<id>/` is a directory), so this is a
/// deliberate, matching duplication, not a drift risk.
fn save_session_metadata_via(
    data_dir: &Path,
    session_id: &str,
    metadata: SessionMetadataPatch,
) -> Result<SessionDetail, IpcError> {
    let session_dir = data_dir.join("sessions").join(session_id);
    if !session_dir.is_dir() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session {session_id} not found")));
    }
    let sj_path = session_dir.join("session.json");

    let mut doc =
        idl_rs::store::session_json::read_session_json(&sj_path).map_err(|e| map_session_json_error(e, &sj_path))?;
    let current_bytes = std::fs::read(&sj_path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("read {}: {e}", sj_path.display())))?;
    let current_hash = idl_rs::store::atomic::sha256_hex(&current_bytes);

    doc.rider = metadata.rider;
    doc.bike = metadata.bike;
    doc.bike_comment = metadata.bike_comment;
    doc.venue_name = metadata.venue_name;
    doc.event_name = metadata.event_name;
    doc.event_session = metadata.event_session;
    doc.short_comment = metadata.short_comment;
    doc.long_comment = metadata.long_comment;
    doc.tag = metadata.tag;

    idl_rs::store::session_json::write_session_json(data_dir, session_id, &doc, Some(&current_hash))
        .map_err(|e| map_session_json_error(e, &sj_path))?;

    Ok(catalog_read::get_session(data_dir, session_id)?.into())
}

/// Lists `<data_dir>/sessions/*` directory entries other than
/// `exclude_session_id`, skipping anything that is not a directory. Used by
/// `delete_session_via` to check whether another session's `data.parquet`
/// still names the blob about to be removed.
fn other_session_dirs(data_dir: &Path, exclude_session_id: &str) -> impl Iterator<Item = PathBuf> {
    let exclude = exclude_session_id.to_string();
    std::fs::read_dir(data_dir.join("sessions"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(move |path| path.is_dir() && path.file_name().and_then(|n| n.to_str()) != Some(exclude.as_str()))
}

/// Transport-agnostic core of `delete_session` (C3 §3.2). Removes
/// `<data>/sessions/<session_id>/` recursively, then the catalog's rows for
/// this session via `idl_rs::store::catalog::delete_session` — the raw SQL
/// and the `laps`/`lap_summary` `ON DELETE CASCADE` reasoning live in
/// `core` (ruling R68: bytes-on-disk stays in `core`, this crate stays
/// thin), not a full `rebuild_catalog` (needlessly expensive per delete; an
/// accepted, bounded divergence risk `rebuild_catalog` remains available to
/// reconcile).
///
/// `delete_blob: true` additionally removes the blob at
/// `blobs/sha256/<2>/<62>` named by the session's `blob_sha256`, but only
/// when no *other* session's `data.parquet` still names the same digest —
/// blobs are content-addressed and shared by construction (C4 §3), so a
/// shared blob is never removed even when `delete_blob: true`.
/// `delete_blob: false` keeps the blob unconditionally (idl0's "Forget
/// session"). The blob's presence is checked by reading every other
/// session's `data.parquet` metadata (`read_session_metadata`, cheap — no
/// full parquet row load) rather than querying the catalog's own
/// `sessions.blob_sha256` column, because this session's catalog row is
/// about to be deleted in the same call and querying it mid-delete is an
/// ordering hazard the file-based check avoids entirely.
fn delete_session_via(data_dir: &Path, session_id: &str, delete_blob: bool) -> Result<(), IpcError> {
    let session_dir = data_dir.join("sessions").join(session_id);
    if !session_dir.is_dir() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session {session_id} not found")));
    }
    let blob_sha256 = catalog_read::get_session(data_dir, session_id)?.blob_sha256;

    std::fs::remove_dir_all(&session_dir)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("remove {}: {e}", session_dir.display())))?;

    if delete_blob {
        let still_referenced = other_session_dirs(data_dir, session_id)
            .filter_map(|dir| idl_rs::store::parquet::read_session_metadata(&dir.join("data.parquet")).ok())
            .any(|m| m.blob_sha256 == blob_sha256);
        if !still_referenced {
            let blob_path = idl_rs::store::blob::blob_path(data_dir, &blob_sha256);
            if blob_path.is_file() {
                std::fs::remove_file(&blob_path)
                    .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("remove {}: {e}", blob_path.display())))?;
            }
        }
    }

    let conn = idl_rs::store::catalog::open_catalog(&data_dir.join("catalog.sqlite"))?;
    idl_rs::store::catalog::delete_session(&conn, session_id)?;

    Ok(())
}

/// C3 §3.2 `save_session_metadata(session_id, metadata)`.
#[tauri::command]
pub fn save_session_metadata(
    session_id: String,
    metadata: SessionMetadataPatch,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<SessionDetail, IpcError> {
    save_session_metadata_via(&data_dir.0, &session_id, metadata)
}

/// C3 §3.2 `delete_session(session_id, delete_blob)`.
#[tauri::command]
pub fn delete_session(
    session_id: String,
    delete_blob: bool,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, crate::session_cache::SessionCache>,
) -> Result<(), IpcError> {
    delete_session_via(&data_dir.0, &session_id, delete_blob)?;
    // The decoded channels came from a `data.parquet` that no longer
    // exists (ruling R203.2's invalidation rule). Invalidated after the
    // delete succeeds: a refused delete leaves the file, and its decodes,
    // valid.
    cache.invalidate_session(&session_id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
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
            timestamp_source: TimestampSource::Header,
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

    /// Like `write_full_session`, but the caller supplies both the raw
    /// blob bytes (so two sessions can be made to share one blob by
    /// passing identical bytes — `write_blob` is a verified no-op on a
    /// repeat digest) and the `session.json` document (so a test can seed
    /// non-default `laps`/`bike_profile_snapshot` before calling
    /// `save_session_metadata_via`). Also runs `core_rebuild_catalog` so
    /// this session's `sessions`/`laps`/`lap_summary` rows exist for
    /// Task 5's delete tests to remove. Returns the blob's sha256 hex.
    fn write_full_session_seeded(root: &Path, session_id: &str, raw_bytes: &[u8], doc: &idl_rs::store::session_json::SessionJson) -> String {
        let blob_sha256 = write_blob(root, raw_bytes).unwrap();
        write_session_json(root, session_id, doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: blob_sha256.clone(),
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
        core_rebuild_catalog(root).unwrap();
        blob_sha256
    }

    fn metadata_patch(suffix: &str) -> SessionMetadataPatch {
        SessionMetadataPatch {
            rider: format!("rider-{suffix}"),
            bike: format!("bike-{suffix}"),
            bike_comment: format!("bike-comment-{suffix}"),
            venue_name: format!("venue-{suffix}"),
            event_name: format!("event-{suffix}"),
            event_session: format!("event-session-{suffix}"),
            short_comment: format!("short-comment-{suffix}"),
            long_comment: format!("long-comment-{suffix}"),
            tag: format!("tag-{suffix}"),
        }
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
                sectors: vec![SectorJson {
                    name: "sector-T".to_string(),
                    start_ms: 3_110,
                    end_ms: 3_120,
                    start_time_secs: 3.31,
                    end_time_secs: 3.32,
                }],
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
            timestamp_source: TimestampSource::Header,
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
        assert_eq!(lap.sectors.len(), 1);
        assert_eq!(lap.sectors[0].name, "sector-K");
        assert_eq!(lap.sectors[0].start_ms, 1_100);
        assert_eq!(lap.sectors[0].end_ms, 1_200);
        assert_eq!(lap.sectors[0].start_time_secs, 1.3);
        assert_eq!(lap.sectors[0].end_time_secs, 1.4);
        assert_eq!(lap.neutral_zone_visits.len(), 1);
        assert_eq!(lap.neutral_zone_visits[0].name, "nz-L");
        assert_eq!(lap.neutral_zone_visits[0].enter_ms, 1_300);
        assert_eq!(lap.neutral_zone_visits[0].exit_ms, 1_400);

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
        assert_eq!(nested_lap.sectors.len(), 1);
        assert_eq!(nested_lap.sectors[0].name, "sector-T");

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
        assert!(detail.lap_timing.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_track_via_serialised_track_detail_field_names_match_c3() {
        // Arrange — a circuit-timed track with one sector gate, one neutral
        // zone, and one reference-polyline fix, wire degrees x1e7.
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line","venue_name":"Whistler",
            "lap_timing":{"kind":"circuit","name":"S/F","start_finish":{"lat1_deg":501163000,"lon1_deg":-1229574000,"lat2_deg":10,"lon2_deg":20,"name":""}},
            "sector_gates":[{"name":"S1","gate":{"lat1_deg":10,"lon1_deg":20,"lat2_deg":30,"lon2_deg":40,"name":""}}],
            "neutral_zones":[{"name":"Pit","enter":{"lat1_deg":1,"lon1_deg":2,"lat2_deg":3,"lon2_deg":4,"name":""},"exit":{"lat1_deg":5,"lon1_deg":6,"lat2_deg":7,"lon2_deg":8,"name":""}}],
            "reference_polyline":[{"timestamp_ms":9000,"latitude_deg":501163000,"longitude_deg":-1229574000}],
            "created_at_ms":1111,"updated_at_ms":2222}}"#;
        std::fs::write(tracks_dir.join("t-1.idl0t"), json).unwrap();

        // Act
        let detail = get_track_via(&root, "t-1").unwrap();
        let value = serde_json::to_value(&detail).unwrap();

        // Assert — field names and the sealed-union tag match C3 §3.2.
        assert_eq!(value["lap_timing"]["kind"], serde_json::json!("circuit"));
        assert_eq!(value["lap_timing"]["start_finish"]["lat1"], serde_json::json!(50.1163));
        assert_eq!(value["lap_timing"]["start_finish"]["lon1"], serde_json::json!(-122.9574));
        assert_eq!(value["sector_gates"][0]["name"], serde_json::json!("S1"));
        assert_eq!(value["sector_gates"][0]["gate"]["lat1"], serde_json::json!(0.000001));
        assert_eq!(value["neutral_zones"][0]["name"], serde_json::json!("Pit"));
        assert_eq!(value["neutral_zones"][0]["enter"]["lat1"], serde_json::json!(0.0000001));
        assert_eq!(value["neutral_zones"][0]["exit"]["lat1"], serde_json::json!(0.0000005));
        assert_eq!(value["reference_polyline"][0]["timestamp_ms"], serde_json::json!(9000));
        assert_eq!(value["reference_polyline"][0]["lat"], serde_json::json!(50.1163));
        assert_eq!(value["reference_polyline"][0]["lon"], serde_json::json!(-122.9574));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_session_metadata_via_replaces_exactly_the_nine_fields_and_preserves_the_rest() {
        // Arrange — seed non-default values for fields this command must
        // leave untouched, plus the nine it must replace.
        let root = temp_root();
        let session_id = "s1";
        let mut doc = empty_session_json(session_id);
        doc.rider = "old-rider".to_string();
        doc.schema_version = idl_rs::store::session_json::SESSION_JSON_SCHEMA_VERSION;
        doc.bike_profile_snapshot = Some(serde_json::json!({ "marker": "untouched-snapshot" }));
        doc.laps = vec![LapJson {
            lap_number: 7,
            start_timestamp_ms: 100,
            end_timestamp_ms: 200,
            raw_elapsed_ms: 100,
            lap_time_ms: 100,
            start_time_secs: 0.1,
            end_time_secs: 0.2,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        doc.reference_lap_number = Some(7);
        write_full_session_seeded(&root, session_id, b"raw bytes for s1", &doc);

        // Act
        let patch = metadata_patch("new");
        let detail = save_session_metadata_via(&root, session_id, patch).unwrap();

        // Assert — the nine replaced fields hold the patch's values.
        assert_eq!(detail.rider, "rider-new");
        assert_eq!(detail.bike, "bike-new");
        assert_eq!(detail.bike_comment, "bike-comment-new");
        assert_eq!(detail.venue_name, "venue-new");
        assert_eq!(detail.event_name, "event-new");
        assert_eq!(detail.event_session, "event-session-new");
        assert_eq!(detail.short_comment, "short-comment-new");
        assert_eq!(detail.long_comment, "long-comment-new");
        assert_eq!(detail.tag, "tag-new");

        // Assert — everything else is byte-for-byte unchanged.
        assert_eq!(detail.bike_profile_snapshot, Some(serde_json::json!({ "marker": "untouched-snapshot" })));
        assert_eq!(detail.laps.len(), 1);
        assert_eq!(detail.laps[0].lap_number, 7);
        assert_eq!(detail.reference_lap_number, Some(7));

        let on_disk = idl_rs::store::session_json::read_session_json(&root.join("sessions").join(session_id).join("session.json")).unwrap();
        assert_eq!(on_disk.bike_profile_snapshot, Some(serde_json::json!({ "marker": "untouched-snapshot" })));
        assert_eq!(on_disk.laps.len(), 1);
        assert_eq!(on_disk.reference_lap_number, Some(7));
        assert_eq!(on_disk.schema_version, idl_rs::store::session_json::SESSION_JSON_SCHEMA_VERSION);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_session_metadata_via_unknown_session_id_ipc_error_kind_is_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = save_session_metadata_via(&root, "nope", metadata_patch("x")).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_session_metadata_via_return_value_matches_a_fresh_get_session_call() {
        // Arrange — proves the "re-read, canonical truth" behaviour rather
        // than an echo of the argument: the returned `SessionDetail` must
        // equal an independent `catalog_read::get_session` call made after
        // the write, not just reflect the patch fields back.
        let root = temp_root();
        let session_id = "s1";
        write_full_session_seeded(&root, session_id, b"raw bytes for s1", &empty_session_json(session_id));

        // Act
        let returned = save_session_metadata_via(&root, session_id, metadata_patch("fresh")).unwrap();
        let reread: SessionDetail = catalog_read::get_session(&root, session_id).unwrap().into();

        // Assert
        assert_eq!(returned.rider, reread.rider);
        assert_eq!(returned.bike, reread.bike);
        assert_eq!(returned.tag, reread.tag);
        assert_eq!(returned.session_id, reread.session_id);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_via_delete_blob_false_removes_directory_and_catalog_rows_but_keeps_the_blob() {
        // Arrange
        let root = temp_root();
        let session_id = "s1";
        let blob_sha256 = write_full_session_seeded(&root, session_id, b"raw bytes for s1", &empty_session_json(session_id));

        // Act
        delete_session_via(&root, session_id, false).unwrap();

        // Assert
        assert!(!root.join("sessions").join(session_id).is_dir());
        assert!(list_sessions_via(&root).unwrap().is_empty());
        assert!(idl_rs::store::blob::blob_path(&root, &blob_sha256).is_file());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_via_delete_blob_true_with_no_other_reference_removes_the_blob_file() {
        // Arrange
        let root = temp_root();
        let session_id = "s1";
        let blob_sha256 = write_full_session_seeded(&root, session_id, b"raw bytes for s1", &empty_session_json(session_id));

        // Act
        delete_session_via(&root, session_id, true).unwrap();

        // Assert
        assert!(!root.join("sessions").join(session_id).is_dir());
        assert!(!idl_rs::store::blob::blob_path(&root, &blob_sha256).is_file());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_via_delete_blob_true_with_a_second_session_sharing_the_blob_keeps_the_blob_file() {
        // Arrange — two sessions built from identical raw bytes hash to the
        // same blob (`write_blob` is a verified no-op on a repeat digest).
        let root = temp_root();
        let shared_bytes: &[u8] = b"shared raw bytes";
        let blob_sha256 = write_full_session_seeded(&root, "s1", shared_bytes, &empty_session_json("s1"));
        write_full_session_seeded(&root, "s2", shared_bytes, &empty_session_json("s2"));

        // Act — deleting s1 must not remove the blob s2 still names.
        delete_session_via(&root, "s1", true).unwrap();

        // Assert
        assert!(!root.join("sessions").join("s1").is_dir());
        assert!(root.join("sessions").join("s2").is_dir());
        assert!(idl_rs::store::blob::blob_path(&root, &blob_sha256).is_file());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_via_unknown_session_id_not_found_and_touches_nothing() {
        // Arrange — a real, untouched session must survive the failed call.
        let root = temp_root();
        let blob_sha256 = write_full_session_seeded(&root, "s1", b"raw bytes for s1", &empty_session_json("s1"));

        // Act
        let err = delete_session_via(&root, "nope", true).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);
        assert!(root.join("sessions").join("s1").is_dir());
        assert!(idl_rs::store::blob::blob_path(&root, &blob_sha256).is_file());
        assert_eq!(list_sessions_via(&root).unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lap_detail_serialises_sectors_and_neutral_zone_visits_byte_identical_to_the_old_value_path() {
        // Arrange — the expected JSON is built literally with `json!`, not
        // by calling any of this module's own code, so this test would
        // catch a field-name or shape drift the refactor from
        // `serde_json::Value` introduced.
        let detail = LapDetail {
            lap_number: 1,
            start_timestamp_ms: 1_000,
            end_timestamp_ms: 2_000,
            raw_elapsed_ms: 1_000,
            lap_time_ms: 900,
            start_time_secs: 1.0,
            end_time_secs: 2.0,
            sectors: vec![
                LapSector { name: "S1".to_string(), start_ms: 1_000, end_ms: 1_500, start_time_secs: 1.0, end_time_secs: 1.5 },
                LapSector { name: "S2".to_string(), start_ms: 1_500, end_ms: 2_000, start_time_secs: 1.5, end_time_secs: 2.0 },
            ],
            neutral_zone_visits: vec![LapNeutralZoneVisit { name: "NZ1".to_string(), enter_ms: 1_100, exit_ms: 1_200 }],
        };

        // Act
        let value = serde_json::to_value(&detail).unwrap();

        // Assert
        assert_eq!(
            value,
            serde_json::json!({
                "lap_number": 1,
                "start_timestamp_ms": 1_000,
                "end_timestamp_ms": 2_000,
                "raw_elapsed_ms": 1_000,
                "lap_time_ms": 900,
                "start_time_secs": 1.0,
                "end_time_secs": 2.0,
                "sectors": [
                    { "name": "S1", "start_ms": 1_000, "end_ms": 1_500, "start_time_secs": 1.0, "end_time_secs": 1.5 },
                    { "name": "S2", "start_ms": 1_500, "end_ms": 2_000, "start_time_secs": 1.5, "end_time_secs": 2.0 }
                ],
                "neutral_zone_visits": [
                    { "name": "NZ1", "enter_ms": 1_100, "exit_ms": 1_200 }
                ]
            })
        );
    }

    #[test]
    fn lap_detail_with_no_sectors_or_neutral_zone_visits_serialises_empty_arrays_not_null() {
        // Arrange
        let detail = LapDetail {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            raw_elapsed_ms: 0,
            lap_time_ms: 0,
            start_time_secs: 0.0,
            end_time_secs: 0.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        };

        // Act
        let value = serde_json::to_value(&detail).unwrap();

        // Assert
        assert_eq!(value["sectors"], serde_json::json!([]));
        assert_eq!(value["neutral_zone_visits"], serde_json::json!([]));
    }

    // ---- save_track (Task 4, L8x) -----------------------------------------

    mod save_track {
        use super::*;
        use idl_rs::track_artifact::read_track;

        fn empty_draft(name: &str) -> TrackDraft {
            TrackDraft {
                track_id: None,
                name: name.to_string(),
                venue_name: "Whistler".to_string(),
                lap_timing: None,
                neutral_zones: Vec::new(),
                sector_gates: Vec::new(),
                reference_polyline: Vec::new(),
            }
        }

        #[test]
        fn save_track_via_no_track_id_mints_the_id_both_timestamps_writes_the_artifact_returns_it() {
            // Arrange
            let root = temp_root();

            // Act
            let result = save_track_via(&root, empty_draft("A-Line"), 1_000, "new-id-1").unwrap();

            // Assert
            assert_eq!(result.track.track_id, "new-id-1");
            assert_eq!(result.track.name, "A-Line");
            assert_eq!(result.track.created_at_ms, 1_000);
            assert_eq!(result.track.updated_at_ms, 1_000);
            let on_disk = read_track(&root.join("tracks").join("new-id-1.idl0t")).unwrap();
            assert_eq!(on_disk.created_at_ms, 1_000);
            assert_eq!(on_disk.updated_at_ms, 1_000);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_an_existing_id_preserves_created_at_ms_and_bumps_updated_at_ms() {
            // Arrange
            let root = temp_root();
            save_track_via(&root, empty_draft("A-Line"), 1_000, "t-1").unwrap();
            let mut edit = empty_draft("A-Line-Renamed");
            edit.track_id = Some("t-1".to_string());

            // Act
            let result = save_track_via(&root, edit, 2_000, "unused-id").unwrap();

            // Assert
            assert_eq!(result.track.track_id, "t-1");
            assert_eq!(result.track.name, "A-Line-Renamed");
            assert_eq!(result.track.created_at_ms, 1_000);
            assert_eq!(result.track.updated_at_ms, 2_000);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_an_unknown_id_is_not_found_and_nothing_is_written() {
            // Arrange
            let root = temp_root();
            let mut draft = empty_draft("A-Line");
            draft.track_id = Some("no-such-track".to_string());

            // Act
            let err = save_track_via(&root, draft, 1_000, "unused").unwrap_err();

            // Assert
            assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);
            assert!(!root.join("tracks").exists());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_an_empty_name_is_invalid_argument_and_nothing_is_written() {
            // Arrange
            let root = temp_root();
            let draft = empty_draft("   ");

            // Act
            let err = save_track_via(&root, draft, 1_000, "new-id-1").unwrap_err();

            // Assert
            assert_eq!(err.kind, crate::error::IpcErrorKind::InvalidArgument);
            assert!(!root.join("tracks").exists());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_no_catalog_sqlite_is_ok_and_creates_none() {
            // Arrange
            let root = temp_root();

            // Act
            let result = save_track_via(&root, empty_draft("A-Line"), 1_000, "new-id-1").unwrap();

            // Assert
            assert!(result.warnings.is_empty());
            assert!(!root.join("catalog.sqlite").is_file());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_with_a_catalog_the_tracks_row_matches_and_a_second_save_updates_it() {
            // Arrange
            let root = temp_root();
            core_rebuild_catalog(&root).unwrap();
            save_track_via(&root, empty_draft("A-Line"), 1_000, "t-1").unwrap();
            {
                let conn = idl_rs::store::catalog::open_catalog(&root.join("catalog.sqlite")).unwrap();
                let count: i64 = conn.query_row("SELECT COUNT(*) FROM tracks", [], |r| r.get(0)).unwrap();
                assert_eq!(count, 1);
                let name: String =
                    conn.query_row("SELECT name FROM tracks WHERE track_id = 't-1'", [], |r| r.get(0)).unwrap();
                assert_eq!(name, "A-Line");
            }
            let mut edit = empty_draft("A-Line-Renamed");
            edit.track_id = Some("t-1".to_string());

            // Act — a second save of the same id.
            save_track_via(&root, edit, 2_000, "unused").unwrap();

            // Assert — updated in place, not duplicated.
            let conn = idl_rs::store::catalog::open_catalog(&root.join("catalog.sqlite")).unwrap();
            let count: i64 = conn.query_row("SELECT COUNT(*) FROM tracks", [], |r| r.get(0)).unwrap();
            assert_eq!(count, 1);
            let name: String = conn.query_row("SELECT name FROM tracks WHERE track_id = 't-1'", [], |r| r.get(0)).unwrap();
            assert_eq!(name, "A-Line-Renamed");

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_a_session_stamped_with_the_old_library_hash_is_in_stale_session_ids() {
            // Arrange — a session already stamped against the *empty*
            // library, before this track existed.
            let root = temp_root();
            let empty_hash = idl_rs::store::lap_index::track_library_hash(&[]);
            let mut doc = empty_session_json("s-old");
            doc.track_visits_library_hash = Some(empty_hash);
            write_session_json(&root, "s-old", &doc, None).unwrap();

            // Act
            let result = save_track_via(&root, empty_draft("A-Line"), 1_000, "t-1").unwrap();

            // Assert
            assert_eq!(result.stale_session_ids, vec!["s-old".to_string()]);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_a_session_stamped_with_the_post_write_hash_is_not_stale() {
            // Arrange — the stamp this session already carries is exactly
            // the hash the library will have *after* this save (computed
            // independently here, not by calling `save_track_via` first).
            let root = temp_root();
            let track = idl_rs::track_artifact::Track {
                id: "t-1".to_string(),
                name: "A-Line".to_string(),
                venue: "Whistler".to_string(),
                timing: None,
                sector_gates: Vec::new(),
                neutral_zones: Vec::new(),
                reference_polyline: Vec::new(),
                created_at_ms: 1_000,
                updated_at_ms: 1_000,
            };
            let post_write_hash = idl_rs::store::lap_index::track_library_hash(std::slice::from_ref(&track));
            let mut doc = empty_session_json("s-fresh");
            doc.track_visits_library_hash = Some(post_write_hash);
            write_session_json(&root, "s-fresh", &doc, None).unwrap();

            // Act
            let result = save_track_via(&root, empty_draft("A-Line"), 1_000, "t-1").unwrap();

            // Assert
            assert!(result.stale_session_ids.is_empty());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn save_track_via_a_caller_supplied_created_at_ms_field_in_the_raw_json_is_ignored() {
            // Arrange — `TrackDraft` is a typed struct, not a JSON bag
            // (matching `SessionMetadataPatch`'s own idiom); C3's
            // `TrackDraft` has no `created_at_ms` field at all, so one
            // present in raw JSON is silently dropped by serde's default
            // unknown-field handling and never reaches this function's body.
            let json = r#"{"track_id":null,"name":"A-Line","venue_name":"Whistler","lap_timing":null,
                "neutral_zones":[],"sector_gates":[],"reference_polyline":[],"created_at_ms":999999}"#;
            let draft: TrackDraft = serde_json::from_str(json).unwrap();
            let root = temp_root();

            // Act
            let result = save_track_via(&root, draft, 1_000, "new-id-1").unwrap();

            // Assert
            assert_eq!(result.track.created_at_ms, 1_000);

            let _ = std::fs::remove_dir_all(&root);
        }
    }

    // ---- delete_track (Task 5, L8x) ---------------------------------------

    mod delete_track {
        use super::*;
        use idl_rs::store::session_json::TrackVisitJson;
        use idl_rs::track_artifact::{write_track, Track};

        fn minimal_track(id: &str, name: &str) -> Track {
            Track {
                id: id.to_string(),
                name: name.to_string(),
                venue: "Whistler".to_string(),
                timing: None,
                sector_gates: Vec::new(),
                neutral_zones: Vec::new(),
                reference_polyline: Vec::new(),
                created_at_ms: 0,
                updated_at_ms: 1,
            }
        }

        #[test]
        fn delete_track_via_an_existing_track_the_artifact_is_gone_and_ok() {
            // Arrange
            let root = temp_root();
            write_track(&root, &minimal_track("t-1", "A-Line")).unwrap();

            // Act
            let result = delete_track_via(&root, "t-1").unwrap();

            // Assert
            assert_eq!(result.track_id, "t-1");
            assert!(!root.join("tracks").join("t-1.idl0t").exists());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn delete_track_via_an_unknown_id_not_found_and_nothing_is_removed() {
            // Arrange
            let root = temp_root();
            write_track(&root, &minimal_track("t-1", "A-Line")).unwrap();

            // Act
            let err = delete_track_via(&root, "no-such-track").unwrap_err();

            // Assert
            assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);
            assert!(root.join("tracks").join("t-1.idl0t").exists());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn delete_track_via_with_a_catalog_the_tracks_row_is_gone_and_a_lap_row_that_referenced_it_survives_with_a_null_track_id() {
            // Arrange
            let root = temp_root();
            write_track(&root, &minimal_track("t-1", "A-Line")).unwrap();
            let session_id = "sess-visit";
            let mut doc = empty_session_json(session_id);
            doc.laps = vec![LapJson {
                lap_number: 1,
                start_timestamp_ms: 1_000,
                end_timestamp_ms: 1_500,
                raw_elapsed_ms: 500,
                lap_time_ms: 500,
                start_time_secs: 0.0,
                end_time_secs: 0.5,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            }];
            doc.track_visits = vec![TrackVisitJson {
                visit_id: "v-1".to_string(),
                track_id: "t-1".to_string(),
                start_timestamp_ms: 500,
                end_timestamp_ms: 2_000,
                laps: Vec::new(),
            }];
            write_full_session_seeded(&root, session_id, b"raw bytes for sess-visit", &doc);
            {
                let conn = idl_rs::store::catalog::open_catalog(&root.join("catalog.sqlite")).unwrap();
                let track_id: Option<String> = conn
                    .query_row("SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1", [session_id], |r| {
                        r.get(0)
                    })
                    .unwrap();
                assert_eq!(track_id.as_deref(), Some("t-1"), "fixture setup: lap must start out naming the track");
            }

            // Act
            delete_track_via(&root, "t-1").unwrap();

            // Assert — the `tracks` row is gone…
            let conn = idl_rs::store::catalog::open_catalog(&root.join("catalog.sqlite")).unwrap();
            let track_count: i64 = conn.query_row("SELECT COUNT(*) FROM tracks WHERE track_id = 't-1'", [], |r| r.get(0)).unwrap();
            assert_eq!(track_count, 0);
            // …but the lap row survives, with `track_id` nulled by the FK cascade.
            let track_id: Option<String> = conn
                .query_row("SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1", [session_id], |r| {
                    r.get(0)
                })
                .unwrap();
            assert_eq!(track_id, None, "laps.track_id must be NULL after the ON DELETE SET NULL cascade");

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn delete_track_via_no_catalog_sqlite_ok_and_no_catalog_is_created() {
            // Arrange
            let root = temp_root();
            write_track(&root, &minimal_track("t-1", "A-Line")).unwrap();

            // Act
            let result = delete_track_via(&root, "t-1").unwrap();

            // Assert
            assert!(result.warnings.is_empty());
            assert!(!root.join("catalog.sqlite").is_file());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn delete_track_via_a_session_whose_stamp_names_the_deleted_library_is_stale_and_its_session_json_is_untouched() {
            // Arrange — a session stamped against the library as it stood
            // *with* t-1 present; deleting t-1 changes the library hash, so
            // this session must come back stale, and its `session.json`
            // bytes must be byte-identical afterwards (delete_track never
            // rewrites it).
            let root = temp_root();
            let track = minimal_track("t-1", "A-Line");
            write_track(&root, &track).unwrap();
            let with_track_hash = idl_rs::store::lap_index::track_library_hash(std::slice::from_ref(&track));
            let mut doc = empty_session_json("s-stamped");
            doc.track_visits_library_hash = Some(with_track_hash);
            write_session_json(&root, "s-stamped", &doc, None).unwrap();
            let sj_path = root.join("sessions").join("s-stamped").join("session.json");
            let before = std::fs::read(&sj_path).unwrap();

            // Act
            let result = delete_track_via(&root, "t-1").unwrap();

            // Assert
            assert_eq!(result.stale_session_ids, vec!["s-stamped".to_string()]);
            let after = std::fs::read(&sj_path).unwrap();
            assert_eq!(before, after, "delete_track must never rewrite session.json");

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn delete_track_via_an_id_containing_a_path_separator_is_not_found_or_io_and_a_decoy_file_outside_tracks_survives() {
            // Arrange — a decoy file that a naive path join could reach via
            // `../decoy`, sitting just outside `tracks/`.
            let root = temp_root();
            std::fs::create_dir_all(root.join("tracks")).unwrap();
            std::fs::write(root.join("decoy"), b"do not touch").unwrap();

            // Act
            let err = delete_track_via(&root, "../decoy").unwrap_err();

            // Assert
            assert!(
                err.kind == crate::error::IpcErrorKind::NotFound || err.kind == crate::error::IpcErrorKind::Io,
                "expected not_found or io, got {:?}",
                err.kind
            );
            assert!(root.join("decoy").exists());

            let _ = std::fs::remove_dir_all(&root);
        }
    }

    // ---- rescan_tracks (Task 8) ------------------------------------------

    mod rescan_tracks {
        use super::*;
        use idl_rs::gps::GpsFix;
        use idl_rs::laps::model::{Gate, LapTiming};
        use idl_rs::track_artifact::{write_track, Track};

        /// A there-and-back-and-there GPS track: 3 legs of 100 one-second
        /// fixes, lat sweeping 0.000→0.099→0.000→0.099 at a fixed longitude,
        /// crossing a gate at lat 0.05 three times. Mirrors
        /// `store::lap_index`'s own `three_lap_fixes` fixture (not reachable
        /// from here — that module's test fixtures are private).
        fn three_lap_fixes() -> Vec<GpsFix> {
            let mut fixes = Vec::new();
            let leg = |t0: i64, up: bool| -> Vec<GpsFix> {
                (0..100)
                    .map(|i| {
                        let lat = if up { i as f64 * 0.001 } else { 0.099 - i as f64 * 0.001 };
                        GpsFix { timestamp_ms: t0 + i * 1000, lat, lon: 0.0005 }
                    })
                    .collect()
            };
            fixes.extend(leg(0, true));
            fixes.extend(leg(100_000, false));
            fixes.extend(leg(200_000, true));
            fixes
        }

        /// A track whose reference polyline covers the same lat range/
        /// longitude as [`three_lap_fixes`], with a circuit start/finish
        /// gate at lat 0.05.
        fn circuit_track(id: &str) -> Track {
            let polyline: Vec<GpsFix> =
                (0..=100).map(|i| GpsFix { timestamp_ms: i * 1000, lat: i as f64 * 0.001, lon: 0.0005 }).collect();
            Track {
                id: id.to_string(),
                name: "Loop".to_string(),
                venue: String::new(),
                timing: Some(LapTiming::Circuit { start_finish: Gate { lat1: 0.05, lon1: -0.001, lat2: 0.05, lon2: 0.001 } }),
                sector_gates: Vec::new(),
                neutral_zones: Vec::new(),
                reference_polyline: polyline,
                created_at_ms: 0,
                updated_at_ms: 1,
            }
        }

        /// Writes a GPS-only session's `data.parquet` (a real blob, not an
        /// empty `blob_sha256`, which `store::blob::blob_path` cannot turn
        /// into a path, so `core_rebuild_catalog`/`index_session` can run
        /// against this fixture) — no `session.json`, left to the caller so
        /// a test can seed one with non-default fields first.
        fn write_gps_parquet(root: &Path, session_id: &str, fixes: &[GpsFix]) {
            let lat: Vec<f64> = fixes.iter().map(|f| f.lat).collect();
            let lon: Vec<f64> = fixes.iter().map(|f| f.lon).collect();
            let epoch: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
            let ch = |id: &str, s: Vec<f64>| {
                let t_us: Vec<i64> = (0..s.len() as i64).map(|i| i * 1_000_000).collect();
                Channel {
                    channel_id: id.to_string(),
                    t_us,
                    t_recorded_us: None,
                    nominal_rate_hz: 1.0,
                    column: RawColumn::F64(s),
                    source_kind: id.to_lowercase(),
                    unit: String::new(),
                    gaps: Vec::new(),
                }
            };
            let blob_sha256 = idl_rs::store::blob::write_blob(root, format!("raw bytes for {session_id}").as_bytes()).unwrap();
            let session = Session {
                session_id: session_id.to_string(),
                device_id: None,
                timestamp_utc_ms: 0,
                timestamp_source: TimestampSource::SourceFile,
                config_checksum: None,
                source_format: SourceFormat::Gpx,
                blob_sha256,
                channels: vec![ch("GPS_Latitude", lat), ch("GPS_Longitude", lon), ch("GPS_EpochMs", epoch)],
            };
            write_session_parquet(root, &session, "test-importer").unwrap();
        }

        /// [`write_gps_parquet`] plus a fresh, empty `session.json` — the
        /// no-track-library-yet, first-ever-rescan scenario Q8 exists for.
        fn write_gps_session(root: &Path, session_id: &str, fixes: &[GpsFix]) {
            write_gps_parquet(root, session_id, fixes);
            write_session_json(root, session_id, &empty_session_json(session_id), None).unwrap();
        }

        #[test]
        fn rescan_tracks_via_track_added_after_import_indexes_laps() {
            // Arrange -- session imported with no track library at all, and
            // a catalog already built (so `rescan_tracks` re-indexes it and
            // `list_laps` can see the result).
            let root = temp_root();
            let session_id = "s1";
            write_gps_session(&root, session_id, &three_lap_fixes());
            core_rebuild_catalog(&root).unwrap();

            // A track shows up only now, after import.
            write_track(&root, &circuit_track("loop-1")).unwrap();

            // Act
            let report = rescan_tracks_via(&root, session_id).unwrap();

            // Assert
            assert_eq!(report.session_id, session_id);
            assert_eq!(report.visits_indexed, 1);
            assert_eq!(report.laps_indexed, 3);
            assert!(report.flags_cleared.is_empty());
            assert!(report.warnings.is_empty());

            let laps = list_laps_via(&root, session_id).unwrap();
            assert_eq!(laps.len(), 3);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn rescan_tracks_via_twice_is_idempotent_with_no_duplicate_catalog_rows() {
            // Arrange
            let root = temp_root();
            let session_id = "s1";
            write_gps_session(&root, session_id, &three_lap_fixes());
            write_track(&root, &circuit_track("loop-1")).unwrap();
            core_rebuild_catalog(&root).unwrap();

            // Act -- rescan twice.
            let first = rescan_tracks_via(&root, session_id).unwrap();
            let second = rescan_tracks_via(&root, session_id).unwrap();

            // Assert -- same counts both times, and only one row per lap.
            assert_eq!(first.laps_indexed, 3);
            assert_eq!(second.laps_indexed, 3);
            let laps = list_laps_via(&root, session_id).unwrap();
            assert_eq!(laps.len(), 3);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn rescan_tracks_via_unknown_session_id_is_not_found() {
            // Arrange
            let root = temp_root();

            // Act
            let err = rescan_tracks_via(&root, "nope").unwrap_err();

            // Assert
            assert_eq!(err.kind, crate::error::IpcErrorKind::NotFound);

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn rescan_tracks_via_no_catalog_sqlite_is_ok_and_creates_none() {
            // Arrange -- no `catalog.sqlite` anywhere under root.
            let root = temp_root();
            let session_id = "s1";
            write_gps_session(&root, session_id, &three_lap_fixes());
            write_track(&root, &circuit_track("loop-1")).unwrap();

            // Act
            let report = rescan_tracks_via(&root, session_id).unwrap();

            // Assert
            assert_eq!(report.laps_indexed, 3);
            assert!(!root.join("catalog.sqlite").is_file());

            let _ = std::fs::remove_dir_all(&root);
        }

        #[test]
        fn rescan_tracks_via_clears_a_now_invalid_main_lap_number() {
            // Arrange -- session.json already carries a starred lap number
            // that renumbering (track added late) will not reproduce.
            let root = temp_root();
            let session_id = "s1";
            write_gps_parquet(&root, session_id, &three_lap_fixes());
            let mut doc = empty_session_json(session_id);
            doc.main_lap_number = Some(99);
            write_session_json(&root, session_id, &doc, None).unwrap();
            write_track(&root, &circuit_track("loop-1")).unwrap();

            // Act
            let report = rescan_tracks_via(&root, session_id).unwrap();

            // Assert
            assert_eq!(report.flags_cleared, vec!["main_lap_number".to_string()]);
            let after = get_session_via(&root, session_id).unwrap();
            assert_eq!(after.main_lap_number, None);

            let _ = std::fs::remove_dir_all(&root);
        }
    }
}
