//! GPS trace commands for the map cell (C3 §3.5, ruling R217 item 1):
//! `fetch_gps_trace_v2` over L3's projection and path decimation, encoded by
//! [`idl_rs::gps_wire::encode_gps_trace_idlg`], and its JSON sibling
//! `fetch_gps_trace_meta` carrying the track underlay and axis domains.
//!
//! **Binary and JSON, split the way `fetch_raster`/`fetch_raster_meta` split**
//! — a trace is thousands of `f64` pairs and belongs on C3 §1's binary side; a
//! track outline and two domains are a handful of JSON numbers a chart needs
//! before it can draw an axis.
//!
//! **Latitude never reaches the app.** Both commands return metres in one
//! local ENU frame ([`idl_rs::gps_projection::EnuFrame`]), and both anchor on
//! the **session's** own fixes, so the trace and the underlay are in the same
//! frame and origin (C2 §5.3). The sandbox draws a plane and computes nothing
//! (CLAUDE.md §2).
//!
//! Same `_via`-suffixed idiom as `commands/scatter.rs`: this module's tests
//! exercise the `_via` functions directly, because `tauri::State` and
//! `tauri::ipc::Response` cannot be constructed outside a running app.

use std::path::Path;

use idl_rs::gps::{build_gps_track, GpsFix};
use idl_rs::gps_projection::{decimate_path, EnuFrame};
use idl_rs::gps_wire::encode_gps_trace_idlg;

use crate::error::{IpcError, IpcErrorKind};
use crate::session_cache::SessionCache;
use crate::session_source::{load_lazy_session_handle, resolve_window, WindowDto};
use crate::state::DataDir;

/// Largest `budget` a caller may request (C3 §3.5). The same cap
/// `fetch_scatter` and `fetch_host_channel_v2` carry: past a few tens of
/// thousands of points the renderer, not the transport, is the limit. C2
/// §5.3's own sizing rule — 4 × the map's CSS pixel width, clamped to
/// `[1024, 8000]` — sits well inside it; this is the outer bound on what the
/// command will accept at all, not the sizing rule itself.
pub const MAX_GPS_TRACE_POINTS: u32 = 65_536;

/// C3 §3.5 `GpsTraceMeta` — `fetch_gps_trace_meta`'s return. Every coordinate
/// is metres in the frame `origin` anchors, never degrees.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GpsTraceMeta {
    /// The ENU frame's origin, decimal degrees — the one place a latitude
    /// appears, so a caller can report where it is looking.
    pub origin: OriginDto,
    /// Metres east, over the union of the trace and the polyline.
    pub x_domain: (f64, f64),
    /// Metres north, over the same union.
    pub y_domain: (f64, f64),
    /// The track's reference polyline, projected into the same frame. Empty
    /// when `track_id` is `null` or the track carries no polyline.
    pub polyline: Vec<PointDto>,
    /// The track's gates, each projected into the same frame as a segment.
    pub gates: Vec<GateSegmentDto>,
}

/// The ENU frame's anchor, decimal degrees (C3 §3.5).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct OriginDto {
    pub lat: f64,
    pub lon: f64,
}

/// One projected point, metres east/north of the frame origin (C3 §3.5).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PointDto {
    pub x: f64,
    pub y: f64,
}

/// One projected gate, metres — a segment rather than a point, because a gate
/// is a line the rider crosses (C3 §3.5).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GateSegmentDto {
    pub name: String,
    /// `"start_finish"`, `"start"`, `"finish"`, `"sector"` or
    /// `"neutral_zone"` — which role this gate plays on the track.
    pub kind: String,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// The fixes of `window`, paired with their session-relative times in seconds.
///
/// Fix times come from `GPS_EpochMs` mapped onto the session's own recording
/// clock ([`idl_rs::session::handle::SessionHandle::epoch_ms_to_time_secs`]),
/// which is the axis every other channel is indexed on — a window resolves to
/// a span on that clock, so slicing anywhere else would drift against the rest
/// of the session.
fn window_fixes(handle: &idl_rs::session::handle::SessionHandle, t0: f64, t1: f64) -> (Vec<GpsFix>, Vec<f64>, Vec<usize>) {
    let fixes = build_gps_track(handle);
    if fixes.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let epochs: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
    let times = handle.epoch_ms_to_time_secs(&epochs);

    let mut kept_fixes = Vec::new();
    let mut kept_times = Vec::new();
    let mut kept_indices = Vec::new();
    for (i, fix) in fixes.iter().enumerate() {
        let t = times.get(i).copied().unwrap_or(f64::NAN);
        if t.is_finite() && t >= t0 && t <= t1 {
            kept_fixes.push(*fix);
            kept_times.push(t);
            kept_indices.push(i);
        }
    }
    (kept_fixes, kept_times, kept_indices)
}

/// Transport-agnostic core of `fetch_gps_trace_v2` (C3 §3.5, ruling R217
/// item 1).
///
/// Resolution order matches `fetch_scatter`/`fetch_fft_v2` (rulings R85,
/// R123): `budget` is validated, then the window resolves — an unknown lap,
/// or a range failing R119/R120, is `invalid_argument` before any sample is
/// read — then the fixes are sliced to it, projected, and decimated.
///
/// **A session with no GPS fixes returns an empty trace, not an error.** A
/// bike ridden indoors has no trace; that is a true answer.
///
/// `colour_by` resolves against the session's own channels. *Lane-local
/// decision, 2026-09-11:* a name that is a workbook **definition** rather than
/// a session channel is `not_found` in this revision — resolving one would
/// mean evaluating the whole workbook here, duplicating
/// `fetch_host_channel_v2`. `workbook_id` is carried (and validated as
/// present) so the signature does not have to move when that lands.
///
/// `not_found`: unknown `session_id`, or a `colour_by` naming no channel.
/// `invalid_argument`: an unresolvable window span, or a `budget` outside
/// `1..=MAX_GPS_TRACE_POINTS`.
pub fn fetch_gps_trace_v2_via(
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

    let (t0_secs, t1_secs) = resolve_window(data_dir, window)?;
    let (fixes, times, indices) = window_fixes(&handle, t0_secs, t1_secs);

    // Resampled onto the *whole* session's fix list, then narrowed to this
    // window's own indices — `gps_channel_values` owns the resampling rule
    // (one value per fix, NaN where the channel has none), and re-deriving it
    // over a sliced fix list here would be a second copy of it.
    let colours: Option<Vec<f64>> = colour_by.map(|id| {
        let all = handle.gps_channel_values(id);
        indices.iter().map(|&i| all.get(i).copied().unwrap_or(f64::NAN)).collect()
    });

    if let Some(err) = source.take_first_error() {
        return Err(err);
    }

    let empty: [f64; 0] = [];
    let Some(frame) = EnuFrame::from_fixes(&fixes) else {
        // No fixes in this window: an empty trace, still flagged the way the
        // request asked, so a reader's `has_c` branch does not change shape
        // between a full window and an empty one.
        return Ok(encode_gps_trace_idlg(&empty, &empty, &empty, colour_by.map(|_| &empty[..])));
    };

    let mut xs = Vec::with_capacity(fixes.len());
    let mut ys = Vec::with_capacity(fixes.len());
    for fix in &fixes {
        let (x, y) = frame.project(fix.lat, fix.lon);
        xs.push(x);
        ys.push(y);
    }

    let keep = decimate_path(&xs, &ys, budget as usize);
    let pick = |src: &[f64]| -> Vec<f64> { keep.iter().map(|&i| src.get(i).copied().unwrap_or(f64::NAN)).collect() };
    let (dx, dy, dt) = (pick(&xs), pick(&ys), pick(&times));
    let dc = colours.as_deref().map(pick);

    Ok(encode_gps_trace_idlg(&dx, &dy, &dt, dc.as_deref()))
}

/// Transport-agnostic core of `fetch_gps_trace_meta` (C3 §3.5, ruling R217
/// item 1).
///
/// The frame is the **session's**, not the track's, and the track is projected
/// into it: a track outlives any one session and a session may run only part
/// of it, so anchoring on the session is what keeps the trace centred and the
/// underlay wherever it falls.
///
/// `track_id: None` is the no-track case — empty `polyline` and `gates`, with
/// domains taken from the trace alone. A map is still a map without a track.
///
/// `not_found`: unknown `session_id` or `track_id`.
pub fn fetch_gps_trace_meta_via(
    cache: &SessionCache,
    data_dir: &Path,
    session_id: &str,
    track_id: Option<&str>,
) -> Result<GpsTraceMeta, IpcError> {
    let (handle, _source) = load_lazy_session_handle(data_dir, session_id, cache)?;
    let fixes = build_gps_track(&handle);
    let frame = EnuFrame::from_fixes(&fixes).unwrap_or_else(|| EnuFrame::new(0.0, 0.0));

    let mut polyline = Vec::new();
    let mut gates = Vec::new();
    if let Some(id) = track_id {
        let track = idl_rs::store::catalog_read::get_track(data_dir, id)?;
        polyline = track
            .reference_polyline
            .iter()
            .map(|f| {
                let (x, y) = frame.project(f.lat, f.lon);
                PointDto { x, y }
            })
            .collect();

        let mut push_gate = |name: String, kind: &str, gate: &idl_rs::laps::model::Gate| {
            let (x1, y1) = frame.project(gate.lat1, gate.lon1);
            let (x2, y2) = frame.project(gate.lat2, gate.lon2);
            gates.push(GateSegmentDto { name, kind: kind.to_string(), x1, y1, x2, y2 });
        };
        match track.lap_timing.as_ref() {
            Some(idl_rs::laps::model::LapTiming::Circuit { start_finish }) => {
                push_gate("Start/finish".to_string(), "start_finish", start_finish);
            }
            Some(idl_rs::laps::model::LapTiming::PointToPoint { start, finish }) => {
                push_gate("Start".to_string(), "start", start);
                push_gate("Finish".to_string(), "finish", finish);
            }
            None => {}
        }
        for sector in &track.sector_gates {
            push_gate(sector.name.clone(), "sector", &sector.gate);
        }
        for zone in &track.neutral_zones {
            push_gate(format!("{} (enter)", zone.name), "neutral_zone", &zone.enter);
            push_gate(format!("{} (exit)", zone.name), "neutral_zone", &zone.exit);
        }
    }

    // The domains cover both layers, so a track the session only partly ran
    // is still wholly visible rather than clipped to the trace's own extent.
    let mut xs: Vec<f64> = fixes.iter().map(|f| frame.project(f.lat, f.lon).0).collect();
    let mut ys: Vec<f64> = fixes.iter().map(|f| frame.project(f.lat, f.lon).1).collect();
    xs.extend(polyline.iter().map(|p| p.x));
    ys.extend(polyline.iter().map(|p| p.y));

    Ok(GpsTraceMeta {
        origin: OriginDto { lat: frame.origin_lat, lon: frame.origin_lon },
        x_domain: finite_domain(&xs),
        y_domain: finite_domain(&ys),
        polyline,
        gates,
    })
}

/// The `[min, max]` of the finite values in `values`, or `(0.0, 0.0)` when
/// there are none — a degenerate domain for an empty map, rather than
/// `(inf, -inf)`, which no scale can consume.
fn finite_domain(values: &[f64]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if lo > hi {
        (0.0, 0.0)
    } else {
        (lo, hi)
    }
}

/// One window's GPS trace, projected into a local ENU frame and decimated by
/// perpendicular distance, as `IDLG` v1 bytes (C3 §3.5, ruling R217 item 1).
/// Settle-bound only (C3 §4): never a hover/pan/zoom handler.
#[tauri::command(async)]
pub fn fetch_gps_trace_v2(
    workbook_id: String,
    window: WindowDto,
    colour_by: Option<String>,
    budget: u32,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<tauri::ipc::Response, IpcError> {
    let _ = workbook_id; // see `fetch_gps_trace_v2_via`: carried, not yet resolved against
    let bytes = fetch_gps_trace_v2_via(&cache, &data_dir.0, &window, colour_by.as_deref(), budget)?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// The map cell's underlay and axis domains (C3 §3.5, ruling R217 item 1) —
/// the track's reference polyline and gates projected into the same frame and
/// origin as every `fetch_gps_trace_v2` payload for the same session.
#[tauri::command(async)]
pub fn fetch_gps_trace_meta(
    session_id: String,
    track_id: Option<String>,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<GpsTraceMeta, IpcError> {
    fetch_gps_trace_meta_via(&cache, &data_dir.0, &session_id, track_id.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::session_source::SpanDto;
    use idl_rs::gps_wire::{FLAG_HAS_C, HEADER_LEN};
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-gps-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds session `"s1"` with 1 Hz GPS running due east from
    /// `(50.0, -120.0)`, plus a `"Speed"` channel on the same clock. Sample
    /// `i` is at `i` seconds and epoch `i * 1000` ms.
    fn seed_session(root: &Path, n: usize) {
        let mk = |id: &str, f: &dyn Fn(usize) -> f64| Channel {
            channel_id: id.to_string(),
            t_us: (0..n).map(|i| i as i64 * 1_000_000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1.0,
            column: RawColumn::F64((0..n).map(f).collect()),
            source_kind: "gps".to_string(),
            unit: String::new(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![
                mk("GPS_Latitude", &|_| 50.0),
                mk("GPS_Longitude", &|i| -120.0 + i as f64 * 1e-4),
                mk("GPS_EpochMs", &|i| i as f64 * 1000.0),
                mk("Speed", &|i| i as f64),
            ],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn session_window() -> WindowDto {
        WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() }
    }

    fn point_count(bytes: &[u8]) -> usize {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    }

    fn flags(bytes: &[u8]) -> u16 {
        u16::from_le_bytes(bytes[6..8].try_into().unwrap())
    }

    fn read_f64(bytes: &[u8], offset: usize) -> f64 {
        f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn fetch_gps_trace_v2_via_projects_to_metres_so_no_latitude_reaches_the_caller() {
        // Arrange — a due-east run at 50 N; every y must be 0 in the frame.
        let root = temp_root();
        seed_session(&root, 200);

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, 65536).unwrap();

        // Assert
        assert_eq!(&bytes[0..4], b"IDLG");
        let n = point_count(&bytes);
        assert!(n >= 2, "got {n} points");
        for i in 0..n {
            let x = read_f64(&bytes, HEADER_LEN + i * 8);
            let y = read_f64(&bytes, HEADER_LEN + (n + i) * 8);
            assert!(x.abs() < 5000.0, "x {x} is metres, not a longitude");
            assert!(y.abs() < 1e-3, "a due-east run has no northing: {y}");
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_a_straight_run_decimates_to_its_endpoints_not_to_the_budget() {
        // Arrange — 500 collinear fixes: perpendicular-distance decimation
        // keeps two, where a uniform stride would have kept `budget` of them.
        let root = temp_root();
        seed_session(&root, 500);

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, 100).unwrap();

        // Assert
        assert_eq!(point_count(&bytes), 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_a_colour_channel_sets_has_c_and_appends_its_own_block() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 50);

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), Some("Speed"), 65536).unwrap();

        // Assert
        assert_eq!(flags(&bytes), FLAG_HAS_C);
        let n = point_count(&bytes);
        assert_eq!(bytes.len(), HEADER_LEN + n * 32);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_no_colour_channel_leaves_has_c_clear() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 50);

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, 65536).unwrap();

        // Assert
        assert_eq!(flags(&bytes), 0);
        assert_eq!(bytes.len(), HEADER_LEN + point_count(&bytes) * 24);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_a_range_window_carries_only_that_window_s_fixes() {
        // Arrange — the first 10 s of a 1 Hz fix stream is ~11 fixes, and
        // their times must all fall inside the window.
        let root = temp_root();
        seed_session(&root, 200);
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 0, t1_us: 10_000_000 },
            colour: "--chart-1".to_string(),
        };

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &window, None, 65536).unwrap();

        // Assert
        let n = point_count(&bytes);
        assert!(n >= 2);
        for i in 0..n {
            let t = read_f64(&bytes, HEADER_LEN + (2 * n + i) * 8);
            assert!((0.0..=10.0).contains(&t), "t {t} outside the requested window");
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_a_session_with_no_gps_is_an_empty_trace_not_an_error() {
        // Arrange — one non-GPS channel only.
        let root = temp_root();
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Fork".to_string(),
                t_us: (0..10).map(|i| i as i64 * 1000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1000.0,
                column: RawColumn::F64(vec![0.0; 10]),
                source_kind: "suspension".to_string(),
                unit: "mm".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let bytes = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, 4096).unwrap();

        // Assert
        assert_eq!(point_count(&bytes), 0);
        assert_eq!(bytes.len(), HEADER_LEN);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_unknown_colour_channel_not_found_rather_than_an_uncoloured_trace() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 20);

        // Act
        let err = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), Some("Nope"), 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
        assert!(err.message.contains("Nope"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();
        let window = WindowDto { session_id: "nope".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() };

        // Act
        let err = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &window, None, 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_zero_budget_invalid_argument_before_any_read() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 20);

        // Act
        let err = fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["budget"], 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_v2_via_over_cap_budget_invalid_argument_naming_the_cap() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 20);

        // Act
        let err =
            fetch_gps_trace_v2_via(&SessionCache::new(), &root, &session_window(), None, MAX_GPS_TRACE_POINTS + 1).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["max_points"], MAX_GPS_TRACE_POINTS);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_meta_via_without_a_track_is_an_empty_underlay_over_the_trace_s_own_domain() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 200);

        // Act
        let meta = fetch_gps_trace_meta_via(&SessionCache::new(), &root, "s1", None).unwrap();

        // Assert
        assert!(meta.polyline.is_empty());
        assert!(meta.gates.is_empty());
        assert!((meta.origin.lat - 50.0).abs() < 1e-9, "origin is the session's own mean latitude");
        assert!(meta.x_domain.1 > meta.x_domain.0, "a due-east run has a non-degenerate x domain");
        assert_eq!(meta.y_domain, (0.0, 0.0), "and no northing at all");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_meta_via_a_session_with_no_gps_still_answers_with_a_degenerate_domain() {
        // Arrange
        let root = temp_root();
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Fork".to_string(),
                t_us: (0..10).map(|i| i as i64 * 1000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1000.0,
                column: RawColumn::F64(vec![0.0; 10]),
                source_kind: "suspension".to_string(),
                unit: "mm".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let meta = fetch_gps_trace_meta_via(&SessionCache::new(), &root, "s1", None).unwrap();

        // Assert — a degenerate domain, never (inf, -inf), which no scale can consume.
        assert_eq!(meta.x_domain, (0.0, 0.0));
        assert_eq!(meta.y_domain, (0.0, 0.0));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_meta_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = fetch_gps_trace_meta_via(&SessionCache::new(), &root, "nope", None).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_gps_trace_meta_via_unknown_track_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root, 20);

        // Act
        let err = fetch_gps_trace_meta_via(&SessionCache::new(), &root, "s1", Some("no-such-track")).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }
}
