//! The track editor's commands (track editor spec §3): the session's GPS line
//! in degrees for the satellite map, what a set of gates catches on the
//! session before the track is saved, and the map's tiles. Saving is the
//! existing `save_track` (`commands::catalog`).
//!
//! The `_via` idiom of `commands/gps.rs`: the tests call the `_via`
//! functions, because `tauri::State` and `tauri::ipc::Response` cannot be
//! constructed outside a running app.

use std::path::Path;

use idl_rs::gps_projection::{decimate_path, EnuFrame};
use idl_rs::gps_wire::encode_gps_trace_idll;
use idl_rs::laps::{preview_runs, LapTiming, SectorGate};
use idl_rs::map_tiles::{read_cached, write_cached, MapTileError, TileId, TileSource};

use crate::commands::catalog::{GpsFixWire, LapTimingWire, SectorGateWire};
use crate::commands::gps::{window_fixes, MAX_GPS_TRACE_POINTS};
use crate::error::{IpcError, IpcErrorKind};
use crate::session_cache::SessionCache;
use crate::session_source::{load_lazy_session_handle, resolve_window, WindowDto};
use crate::state::DataDir;

/// One run the gates catch (track editor spec §3).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RunDto {
    /// Session-relative seconds, the clock every chart uses.
    pub start_secs: f64,
    pub end_secs: f64,
    /// The run's time, seconds.
    pub duration_secs: f64,
}

/// `preview_track_runs`' return (track editor spec §3).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrackRunsPreview {
    pub runs: Vec<RunDto>,
    /// The reference run's fixes, decimal degrees, every fix; empty when the
    /// gates catch no run. What `save_track` takes for a track made from this
    /// session.
    pub reference_polyline: Vec<GpsFixWire>,
}

/// Transport-agnostic core of `fetch_session_fixes`: `window`'s GPS fixes as
/// `IDLL` v1 bytes, longitude and latitude in degrees. Decimated exactly as
/// `fetch_gps_trace_v2` is (perpendicular distance in a local metre frame),
/// so the two maps keep the same points; only the units sent differ.
///
/// `not_found`: unknown session, or a `colour_by` naming no channel.
/// `invalid_argument`: `budget` outside `1..=MAX_GPS_TRACE_POINTS`, or an
/// unresolvable window.
pub fn fetch_session_fixes_via(
    cache: &SessionCache,
    data_dir: &Path,
    window: &WindowDto,
    colour_by: Option<&str>,
    budget: u32,
) -> Result<Vec<u8>, IpcError> {
    if budget == 0 || budget > MAX_GPS_TRACE_POINTS {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("budget must be in 1..={MAX_GPS_TRACE_POINTS}, got {budget}"),
            serde_json::json!({ "budget": budget, "max_points": MAX_GPS_TRACE_POINTS }),
        ));
    }
    let (handle, source) = load_lazy_session_handle(data_dir, &window.session_id, cache)?;
    if let Some(id) = colour_by {
        if handle.channel_meta(id).is_none() {
            return Err(IpcError::new(
                IpcErrorKind::NotFound,
                format!("channel '{id}' not found in session '{}'", window.session_id),
            ));
        }
    }
    let (t0, t1) = resolve_window(data_dir, window)?;
    let (fixes, times, indices) = window_fixes(&handle, t0, t1);
    let colours: Option<Vec<f64>> = colour_by.map(|id| {
        let all = handle.gps_channel_values(id);
        indices.iter().map(|&i| all.get(i).copied().unwrap_or(f64::NAN)).collect()
    });
    if let Some(err) = source.take_first_error() {
        return Err(err);
    }

    let empty: [f64; 0] = [];
    let Some(frame) = EnuFrame::from_fixes(&fixes) else {
        return Ok(encode_gps_trace_idll(&empty, &empty, &empty, colour_by.map(|_| &empty[..])));
    };
    let (xs, ys): (Vec<f64>, Vec<f64>) = fixes.iter().map(|f| frame.project(f.lat, f.lon)).unzip();
    let keep = decimate_path(&xs, &ys, budget as usize);

    let lons: Vec<f64> = keep.iter().map(|&i| fixes[i].lon).collect();
    let lats: Vec<f64> = keep.iter().map(|&i| fixes[i].lat).collect();
    let ts: Vec<f64> = keep.iter().map(|&i| times[i]).collect();
    let cs: Option<Vec<f64>> = colours.as_deref().map(|c| keep.iter().map(|&i| c.get(i).copied().unwrap_or(f64::NAN)).collect());
    Ok(encode_gps_trace_idll(&lons, &lats, &ts, cs.as_deref()))
}

/// Transport-agnostic core of `preview_track_runs`: the runs `lap_timing`
/// and `sector_gates` catch on the whole session, and the reference run's
/// fixes (`idl_rs::laps::preview_runs`). `lap_timing: None` catches nothing.
///
/// `not_found`: unknown session.
pub fn preview_track_runs_via(
    cache: &SessionCache,
    data_dir: &Path,
    session_id: &str,
    lap_timing: Option<&LapTimingWire>,
    sector_gates: &[SectorGateWire],
) -> Result<TrackRunsPreview, IpcError> {
    let (handle, source) = load_lazy_session_handle(data_dir, session_id, cache)?;
    let Some(timing) = lap_timing else {
        return Ok(TrackRunsPreview { runs: Vec::new(), reference_polyline: Vec::new() });
    };
    let timing: LapTiming = timing.into();
    let sectors: Vec<SectorGate> = sector_gates.iter().map(SectorGate::from).collect();
    let preview = preview_runs(&handle, &timing, &sectors);
    if let Some(err) = source.take_first_error() {
        return Err(err);
    }
    Ok(TrackRunsPreview {
        runs: preview
            .runs
            .iter()
            .map(|r| RunDto {
                start_secs: r.start_time_secs,
                end_secs: r.end_time_secs,
                duration_secs: r.lap_time_ms as f64 / 1000.0,
            })
            .collect(),
        reference_polyline: preview.reference_polyline.iter().map(GpsFixWire::from).collect(),
    })
}

/// Checks a tile request (track editor spec §3): a known source and a tile
/// that exists at its zoom, else `invalid_argument`.
pub fn parse_tile_request(source: &str, z: u32, x: u32, y: u32) -> Result<(TileSource, TileId), IpcError> {
    let src = TileSource::parse(source)
        .ok_or_else(|| IpcError::new(IpcErrorKind::InvalidArgument, MapTileError::UnknownSource(source.to_string()).to_string()))?;
    let tile = TileId::new(z, x, y).map_err(|e| IpcError::new(IpcErrorKind::InvalidArgument, e.to_string()))?;
    Ok((src, tile))
}

/// The session's GPS fixes for the track editor's satellite map, `IDLL` v1
/// bytes (track editor spec §3). Settle-bound, never on a pan or zoom.
#[tauri::command(async)]
pub fn fetch_session_fixes(
    window: WindowDto,
    colour_by: Option<String>,
    budget: u32,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<tauri::ipc::Response, IpcError> {
    let bytes = fetch_session_fixes_via(&cache, &data_dir.0, &window, colour_by.as_deref(), budget)?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// The runs a set of gates catches on one session, and the line a track
/// made from it would take (track editor spec §3).
#[tauri::command(async)]
pub fn preview_track_runs(
    session_id: String,
    lap_timing: Option<LapTimingWire>,
    sector_gates: Vec<SectorGateWire>,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<TrackRunsPreview, IpcError> {
    preview_track_runs_via(&cache, &data_dir.0, &session_id, lap_timing.as_ref(), &sector_gates)
}

/// One satellite map tile (track editor spec §3–4): from the OS cache
/// directory when it is there, else fetched and cached. `not_found` when it
/// is neither (offline, or the server has no such tile): the map draws it
/// blank. `invalid_argument`: an unknown source or a tile that does not
/// exist.
#[tauri::command]
pub async fn fetch_map_tile<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    source: String,
    z: u32,
    x: u32,
    y: u32,
) -> Result<tauri::ipc::Response, IpcError> {
    use tauri::Manager;
    let (src, tile) = parse_tile_request(&source, z, x, y)?;
    let cache_root = app.path().app_cache_dir().map_err(|e| IpcError::new(IpcErrorKind::Internal, e.to_string()))?;
    if let Some(bytes) = read_cached(&cache_root, src, tile) {
        return Ok(tauri::ipc::Response::new(bytes));
    }
    let bytes = idl_transport::map_tiles::fetch_tile(src, tile)
        .await
        .map_err(|e| IpcError::new(IpcErrorKind::NotFound, e.message))?;
    // A tile that fetched but would not cache is still a tile.
    let _ = write_cached(&cache_root, src, tile, &bytes);
    Ok(tauri::ipc::Response::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::commands::catalog::GateWire;
    use crate::session_source::SpanDto;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-track-editor-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Session `"s1"`: 100 fixes at 1 Hz running due north from lat 0.000
    /// to 0.099, lon 0.0005.
    fn seed_northbound(root: &Path) {
        let n = 100usize;
        let gps = |id: &str, samples: Vec<f64>| Channel {
            channel_id: id.to_string(),
            t_us: (0..n as i64).map(|i| i * 1_000_000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1.0,
            column: RawColumn::F64(samples),
            source_kind: "gps".to_string(),
            unit: String::new(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 1_000,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: "0".repeat(64),
            channels: vec![
                gps("GPS_Latitude", (0..n).map(|i| i as f64 * 0.001).collect()),
                gps("GPS_Longitude", vec![0.0005; n]),
                gps("GPS_EpochMs", (0..n).map(|i| 1_000.0 + i as f64 * 1_000.0).collect()),
            ],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn gate_at(lat: f64) -> GateWire {
        GateWire { lat1: lat, lon1: -0.001, lat2: lat, lon2: 0.001 }
    }

    #[test]
    fn fetch_session_fixes_via_northbound_session_sends_degrees_under_idll() {
        // Arrange
        let root = temp_root();
        seed_northbound(&root);
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() };

        // Act
        let bytes = fetch_session_fixes_via(&SessionCache::new(), &root, &window, None, 1024).unwrap();

        // Assert — a straight line decimates to its two ends, in degrees.
        assert_eq!(&bytes[..4], b"IDLL");
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let f = |k: usize| f64::from_le_bytes(bytes[16 + 8 * k..24 + 8 * k].try_into().unwrap());
        assert!(n >= 2);
        assert_eq!(f(0), 0.0005); // first longitude
        assert_eq!(f(n), 0.0); // first latitude
        assert!((f(2 * n - 1) - 0.099).abs() < 1e-12); // last latitude

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preview_track_runs_via_point_to_point_gates_one_run_and_its_line() {
        // Arrange
        let root = temp_root();
        seed_northbound(&root);
        let timing = LapTimingWire::PointToPoint { start: gate_at(0.02), finish: gate_at(0.08) };

        // Act
        let preview = preview_track_runs_via(&SessionCache::new(), &root, "s1", Some(&timing), &[]).unwrap();

        // Assert
        assert_eq!(preview.runs.len(), 1);
        assert_eq!(preview.runs[0].duration_secs, 60.0);
        assert_eq!(preview.reference_polyline.len(), 61);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preview_track_runs_via_no_timing_catches_nothing() {
        // Arrange
        let root = temp_root();
        seed_northbound(&root);

        // Act
        let preview = preview_track_runs_via(&SessionCache::new(), &root, "s1", None, &[]).unwrap();

        // Assert
        assert!(preview.runs.is_empty());
        assert!(preview.reference_polyline.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preview_track_runs_via_unknown_session_not_found() {
        // Act
        let err = preview_track_runs_via(&SessionCache::new(), &temp_root(), "nope", None, &[]).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
    }

    #[test]
    fn parse_tile_request_unknown_source_and_impossible_tile_are_invalid_argument() {
        // Act
        let ok = parse_tile_request("esri_world_imagery", 15, 9000, 12000);
        let bad_source = parse_tile_request("google", 1, 0, 0).unwrap_err();
        let bad_tile = parse_tile_request("esri_world_imagery", 2, 4, 0).unwrap_err();

        // Assert
        assert!(ok.is_ok());
        assert_eq!(bad_source.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(bad_tile.kind, IpcErrorKind::InvalidArgument);
    }
}
