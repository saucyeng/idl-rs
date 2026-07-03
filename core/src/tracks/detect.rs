//! Multi-track visit detection. Reads session GPS from the handle, assigns each
//! fix to its nearest in-range track in a flat-earth metric frame, and coalesces
//! contiguous on-track runs into visit windows. Verbatim port of Dart
//! `TrackMatcher.findVisits`. Identity (`visit_id`) is assigned by the caller —
//! this function is pure and deterministic.

use crate::gps::{build_gps_track, GpsFix};
use crate::session::handle::SessionHandle;
use crate::tracks::geometry::{closest_point_on_polyline, Bbox};

/// Tuning parameters for visit detection. Domain defaults live here — the one
/// source of truth, not duplicated in the app or CLI.
#[derive(Debug, Clone, Copy)]
pub struct VisitParams {
    /// A fix counts as "on track" within this distance, metres.
    pub threshold_m: f64,
    /// Off-track gap tolerated inside one visit, seconds.
    pub gap_tolerance_s: f64,
    /// Windows shorter than this are discarded, seconds.
    pub min_visit_s: f64,
}

impl Default for VisitParams {
    fn default() -> Self {
        Self { threshold_m: 30.0, gap_tolerance_s: 5.0, min_visit_s: 30.0 }
    }
}

/// One reference track to match against. `polyline` is the track's stored
/// reference polyline (lat/lon at the channel-sample scale, degrees × 1e7).
#[derive(Debug, Clone)]
pub struct TrackRef {
    pub track_id: String,
    pub polyline: Vec<GpsFix>,
}

/// A contiguous on-track window. `visit_id` is assigned by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct VisitWindow {
    pub track_id: String,
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
}

/// Pre-projected polyline for one candidate track (metres frame).
struct TrackProj {
    track_id: String,
    ref_lat_m: Vec<f64>,
    ref_lon_m: Vec<f64>,
}

/// Detect ordered track visits within the session. Returns `[]` when the
/// session has < 2 fixes or `tracks` is empty.
pub fn detect_visits(handle: &SessionHandle, tracks: &[TrackRef], params: VisitParams) -> Vec<VisitWindow> {
    let session_gps = build_gps_track(handle);
    if session_gps.len() < 2 || tracks.is_empty() {
        return Vec::new();
    }

    // Local-metre conversion. Coords are deg × 1e7, so the divide-by-1e7 cancels
    // into the metres-per-deg constant. Latitude scale is fixed; longitude
    // shrinks by cos(lat) at the session centroid.
    let sess_box = Bbox::of(&session_gps);
    let centroid_lat_deg = (sess_box.min_lat + sess_box.max_lat) / 2.0 / 1e7;
    const M_PER_DEG_UNITS: f64 = 111_320.0 / 1e7;
    let lon_scale = M_PER_DEG_UNITS * (centroid_lat_deg * std::f64::consts::PI / 180.0).cos().abs();

    // Step 1: bounding-box pre-filter + pre-projection.
    let mut candidates: Vec<TrackProj> = Vec::new();
    for t in tracks {
        if t.polyline.len() < 2 {
            continue;
        }
        let t_box = Bbox::of(&t.polyline);
        if !sess_box.overlaps(&t_box) {
            continue;
        }
        candidates.push(TrackProj {
            track_id: t.track_id.clone(),
            ref_lat_m: t.polyline.iter().map(|f| f.lat * M_PER_DEG_UNITS).collect(),
            ref_lon_m: t.polyline.iter().map(|f| f.lon * lon_scale).collect(),
        });
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    let threshold_sq = params.threshold_m * params.threshold_m;

    // Step 2: per-sample nearest-track assignment.
    let mut assignment: Vec<Option<&str>> = Vec::with_capacity(session_gps.len());
    for fix in &session_gps {
        let px_m = fix.lat * M_PER_DEG_UNITS;
        let py_m = fix.lon * lon_scale;
        let mut best_id: Option<&str> = None;
        let mut best_dist_sq = threshold_sq;
        for c in &candidates {
            let r = closest_point_on_polyline(px_m, py_m, &c.ref_lat_m, &c.ref_lon_m);
            if r.dist_sq < best_dist_sq {
                best_dist_sq = r.dist_sq;
                best_id = Some(&c.track_id);
            }
        }
        assignment.push(best_id);
    }

    // Step 3: coalesce + Step 4: discard short visits.
    let gap_tolerance_ms = (params.gap_tolerance_s * 1000.0).round() as i64;
    let min_visit_ms = (params.min_visit_s * 1000.0).round() as i64;
    let mut visits: Vec<VisitWindow> = Vec::new();

    // (track_id, start_ms, last_on_track_ms)
    let mut open: Option<(String, i64, i64)> = None;

    fn close_open(open: &mut Option<(String, i64, i64)>, min_visit_ms: i64, visits: &mut Vec<VisitWindow>) {
        if let Some((id, start, last)) = open.take() {
            if last - start >= min_visit_ms {
                visits.push(VisitWindow { track_id: id, start_timestamp_ms: start, end_timestamp_ms: last });
            }
        }
    }

    for (i, fix) in session_gps.iter().enumerate() {
        match assignment[i] {
            None => {
                // Off-track. Tolerate small gaps; close on overrun.
                if let Some((_, _, last)) = &open {
                    if fix.timestamp_ms - *last > gap_tolerance_ms {
                        close_open(&mut open, min_visit_ms, &mut visits);
                    }
                }
            }
            Some(assigned) => match &mut open {
                None => open = Some((assigned.to_string(), fix.timestamp_ms, fix.timestamp_ms)),
                Some((id, _, last)) if id == assigned => *last = fix.timestamp_ms,
                Some(_) => {
                    close_open(&mut open, min_visit_ms, &mut visits);
                    open = Some((assigned.to_string(), fix.timestamp_ms, fix.timestamp_ms));
                }
            },
        }
    }
    close_open(&mut open, min_visit_ms, &mut visits);

    visits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    // Build a handle whose GPS channels replay `fixes` verbatim.
    fn handle_from_fixes(fixes: &[GpsFix]) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let lat: Vec<f64> = fixes.iter().map(|f| f.lat).collect();
        let lon: Vec<f64> = fixes.iter().map(|f| f.lon).collect();
        let epoch: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
        let ch = |id: &str, s: Vec<f64>| ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: 1.0,
            samples: s,
            sample_times_secs: None,
        };
        SessionHandle::from_channels(
            meta,
            vec![ch("GPS_Latitude", lat), ch("GPS_Longitude", lon), ch("GPS_EpochMs", epoch)],
        )
    }

    // A straight polyline of `n` fixes from (lat0,lon0) stepping by (dlat,dlon),
    // 1 second apart starting at t0. lon avoids exactly 0.0 to dodge the (0,0)
    // sentinel drop in build_gps_track.
    fn line(t0: i64, n: usize, lat0: f64, lon0: f64, dlat: f64, dlon: f64) -> Vec<GpsFix> {
        (0..n)
            .map(|i| GpsFix {
                timestamp_ms: t0 + i as i64 * 1000,
                lat: lat0 + i as f64 * dlat,
                lon: lon0 + i as f64 * dlon,
            })
            .collect()
    }

    // A reference track polyline (lat/lon only matter); coords at deg × 1e7.
    fn track(id: &str, fixes: Vec<GpsFix>) -> TrackRef {
        TrackRef { track_id: id.to_string(), polyline: fixes }
    }

    #[test]
    fn detect_visits_single_track_one_window() {
        // Arrange — 40 fixes (≈40 s) hugging a track polyline within threshold.
        let poly = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let session = handle_from_fixes(&line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0));
        let tracks = vec![track("A", poly)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — one visit spanning the run.
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].track_id, "A");
        assert_eq!(v[0].start_timestamp_ms, 0);
        assert_eq!(v[0].end_timestamp_ms, 39_000);
    }

    #[test]
    fn detect_visits_two_tracks_in_sequence_yield_two_ordered_windows() {
        // Arrange — 40 s on track A, then 40 s on a far-away track B.
        let a_poly = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let b_poly = line(0, 40, 9_000_000.0, 9_000_000.0, 100.0, 0.0);
        let mut fixes = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        fixes.extend(line(40_000, 40, 9_000_000.0, 9_000_000.0, 100.0, 0.0));
        let session = handle_from_fixes(&fixes);
        let tracks = vec![track("A", a_poly), track("B", b_poly)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — A then B, in time order.
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].track_id, "A");
        assert_eq!(v[1].track_id, "B");
        assert!(v[0].end_timestamp_ms < v[1].start_timestamp_ms);
    }

    #[test]
    fn detect_visits_short_window_discarded() {
        // Arrange — only 10 s on track (< 30 s min_visit).
        let poly = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let session = handle_from_fixes(&line(0, 10, 1_000_000.0, 1_000_000.0, 100.0, 0.0));
        let tracks = vec![track("A", poly)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — discarded.
        assert!(v.is_empty());
    }

    #[test]
    fn detect_visits_gap_within_tolerance_keeps_one_window() {
        // Arrange — 20 s on track, a single 3 s off-track blip (within 5 s
        // tolerance), then 20 s back on track. lon jumps far away during the blip.
        let poly = line(0, 60, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let mut fixes = line(0, 20, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        // One off-track fix 3 s after the last on-track fix (gap = 3 s ≤ 5 s).
        fixes.push(GpsFix { timestamp_ms: 22_000, lat: 5_000_000.0, lon: 5_000_000.0 });
        // Resume on track; continue the polyline coordinates from index 20.
        fixes.extend(
            (20..40).map(|i| GpsFix {
                timestamp_ms: 23_000 + (i - 20) as i64 * 1000,
                lat: 1_000_000.0 + i as f64 * 100.0,
                lon: 1_000_000.0,
            }),
        );
        let session = handle_from_fixes(&fixes);
        let tracks = vec![track("A", poly)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — the blip is tolerated; a single visit spans the whole run.
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].track_id, "A");
    }

    #[test]
    fn detect_visits_gap_beyond_tolerance_splits_into_two() {
        // Arrange — 40 s on track, a 10 s off-track gap (> 5 s tolerance), then
        // 40 s back on track. Each on-track run is long enough to survive
        // min_visit, so the over-tolerance gap yields two windows.
        let poly = line(0, 200, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let mut fixes = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        // 10 s off-track gap: jump far away for one fix at t = 50 s.
        fixes.push(GpsFix { timestamp_ms: 50_000, lat: 5_000_000.0, lon: 5_000_000.0 });
        // Resume on the polyline from t = 60 s for another 40 s.
        fixes.extend(
            (60..100).map(|i| GpsFix {
                timestamp_ms: 60_000 + (i - 60) as i64 * 1000,
                lat: 1_000_000.0 + i as f64 * 100.0,
                lon: 1_000_000.0,
            }),
        );
        let session = handle_from_fixes(&fixes);
        let tracks = vec![track("A", poly)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — two windows for the same track.
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].track_id, "A");
        assert_eq!(v[1].track_id, "A");
        assert!(v[0].end_timestamp_ms < v[1].start_timestamp_ms);
    }

    #[test]
    fn detect_visits_tight_threshold_rejects_a_near_but_offset_run() {
        // Arrange — session runs parallel to the track, offset ~sub-metre
        // perpendicular to a (1,1) diagonal (within the default 30 m, but
        // outside a 0.1 m threshold). The bboxes overlap, so this exercises the
        // per-sample distance gate, not the bbox pre-filter.
        let poly = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 100.0);
        let session = handle_from_fixes(&line(0, 40, 1_000_050.0, 999_950.0, 100.0, 100.0));
        let tracks = vec![track("A", poly)];

        // Act — default threshold matches; a 0.1 m threshold rejects every sample.
        let at_default = detect_visits(&session, &tracks, VisitParams::default());
        let tight =
            detect_visits(&session, &tracks, VisitParams { threshold_m: 0.1, ..VisitParams::default() });

        // Assert
        assert_eq!(at_default.len(), 1);
        assert!(tight.is_empty());
    }

    #[test]
    fn detect_visits_bbox_non_overlap_rejects_track() {
        // Arrange — session near origin; only candidate is far away.
        let far = line(0, 40, 9_000_000.0, 9_000_000.0, 100.0, 0.0);
        let session = handle_from_fixes(&line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0));
        let tracks = vec![track("Far", far)];

        // Act
        let v = detect_visits(&session, &tracks, VisitParams::default());

        // Assert — no candidates survive the bbox pre-filter.
        assert!(v.is_empty());
    }

    #[test]
    fn detect_visits_empty_when_too_few_fixes_or_no_tracks() {
        // Arrange
        let poly = line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0);
        let one_fix = handle_from_fixes(&line(0, 1, 1_000_000.0, 1_000_000.0, 100.0, 0.0));
        let many = handle_from_fixes(&line(0, 40, 1_000_000.0, 1_000_000.0, 100.0, 0.0));

        // Act + Assert — < 2 fixes → []; empty track list → [].
        assert!(detect_visits(&one_fix, &[track("A", poly.clone())], VisitParams::default()).is_empty());
        assert!(detect_visits(&many, &[], VisitParams::default()).is_empty());
    }
}
