//! Multi-track visit detection: the gates decide, the polyline confirms.
//! Candidate runs come from each track's gate crossings (`laps::detect_runs`);
//! a run counts for its track when enough of its time lies near the track's
//! polyline; where confirmed runs of two tracks overlap, the closer fit keeps
//! the overlap; consecutive surviving runs of one track make a visit. A moment
//! no surviving run covers belongs to no track. Identity (`visit_id`) is
//! assigned by the caller — this function is pure and deterministic.

use crate::gps::{build_gps_track, GpsFix};
use crate::laps::detect::{detect_runs, Run};
use crate::laps::model::LapTiming;
use crate::session::handle::SessionHandle;
use crate::tracks::geometry::{closest_point_on_polyline, Bbox};

/// Tuning parameters for visit detection. Domain defaults live here — the one
/// source of truth, not duplicated in the app or CLI.
#[derive(Debug, Clone, Copy)]
pub struct VisitParams {
    /// A fix counts as "on track" within this distance of the polyline, metres.
    pub threshold_m: f64,
    /// A run is confirmed when at least this fraction of its time (0..=1,
    /// time-weighted) is on track.
    pub min_on_track_fraction: f64,
}

impl Default for VisitParams {
    fn default() -> Self {
        Self { threshold_m: 30.0, min_on_track_fraction: 0.8 }
    }
}

/// One reference track to match against. `polyline` is the track's stored
/// reference polyline (lat/lon at the channel-sample scale — physical
/// decimal degrees, ruling R27); `timing` is its gates. A track with no
/// timing has no gates, so it never yields a visit.
#[derive(Debug, Clone)]
pub struct TrackRef {
    /// The track's id (`.idl1t` `track_id`).
    pub track_id: String,
    /// Reference polyline, decimal degrees.
    pub polyline: Vec<GpsFix>,
    /// Lap timing gates; `None` = none configured.
    pub timing: Option<LapTiming>,
}

/// A visit window: from its first run's start crossing to its last run's
/// end crossing, milliseconds since the Unix epoch. `visit_id` is assigned
/// by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct VisitWindow {
    /// The visited track's id.
    pub track_id: String,
    /// Visit start, milliseconds since the Unix epoch.
    pub start_timestamp_ms: i64,
    /// Visit end, milliseconds since the Unix epoch.
    pub end_timestamp_ms: i64,
}

/// One candidate track's runs and every fix's squared distance to its
/// polyline (metres²).
struct Candidate<'a> {
    track_id: &'a str,
    dist_sq: Vec<f64>,
    runs: Vec<Run>,
    confirmed: Vec<bool>,
}

/// Detect ordered track visits within the session. Returns `[]` when the
/// session has < 2 fixes or `tracks` is empty.
pub fn detect_visits(handle: &SessionHandle, tracks: &[TrackRef], params: VisitParams) -> Vec<VisitWindow> {
    let gps = build_gps_track(handle);
    if gps.len() < 2 || tracks.is_empty() {
        return Vec::new();
    }

    // Local-metre conversion. Latitude scale is fixed; longitude shrinks by
    // cos(lat) at the session centroid.
    let sess_box = Bbox::of(&gps);
    let centroid_lat_deg = (sess_box.min_lat + sess_box.max_lat) / 2.0;
    const M_PER_DEG: f64 = 111_320.0;
    let lon_scale = M_PER_DEG * (centroid_lat_deg * std::f64::consts::PI / 180.0).cos().abs();
    let threshold_sq = params.threshold_m * params.threshold_m;

    // 1. Candidate runs from each track's gates (bounding-box pre-filter
    //    first), each confirmed or not by its polyline.
    let mut candidates: Vec<Candidate> = Vec::new();
    for t in tracks {
        let Some(timing) = &t.timing else { continue };
        if t.polyline.len() < 2 || !sess_box.overlaps(&Bbox::of(&t.polyline)) {
            continue;
        }
        let runs = detect_runs(&gps, timing);
        if runs.is_empty() {
            continue;
        }
        let ref_lat_m: Vec<f64> = t.polyline.iter().map(|f| f.lat * M_PER_DEG).collect();
        let ref_lon_m: Vec<f64> = t.polyline.iter().map(|f| f.lon * lon_scale).collect();
        let dist_sq: Vec<f64> = gps
            .iter()
            .map(|f| closest_point_on_polyline(f.lat * M_PER_DEG, f.lon * lon_scale, &ref_lat_m, &ref_lon_m).dist_sq)
            .collect();
        let on_track: Vec<f64> = dist_sq.iter().map(|&d| if d <= threshold_sq { 1.0 } else { 0.0 }).collect();
        let confirmed = runs
            .iter()
            .map(|r| time_mean(&gps, &on_track, r.start_ms, r.end_ms).is_some_and(|f| f >= params.min_on_track_fraction))
            .collect();
        candidates.push(Candidate { track_id: &t.track_id, dist_sq, runs, confirmed });
    }

    // 2. Overlaps go to the closer track: a confirmed run is dropped when an
    //    overlapping confirmed run of another track has the smaller mean
    //    squared distance over the overlap (ties to the smaller track id).
    //    3. The survivors, as (start, candidate, run index, end), in time order.
    let mut kept: Vec<(i64, usize, usize, i64)> = Vec::new();
    for (ci, c) in candidates.iter().enumerate() {
        for (ri, r) in c.runs.iter().enumerate().filter(|&(ri, _)| c.confirmed[ri]) {
            let loses = candidates.iter().enumerate().filter(|&(oi, _)| oi != ci).any(|(_, o)| {
                o.runs.iter().zip(&o.confirmed).any(|(q, &ok)| {
                    let (s, e) = (r.start_ms.max(q.start_ms), r.end_ms.min(q.end_ms));
                    if !ok || s >= e {
                        return false;
                    }
                    let mine = time_mean(&gps, &c.dist_sq, s, e).unwrap_or(f64::INFINITY);
                    let theirs = time_mean(&gps, &o.dist_sq, s, e).unwrap_or(f64::INFINITY);
                    theirs < mine || (theirs == mine && o.track_id < c.track_id)
                })
            });
            if !loses {
                kept.push((r.start_ms, ci, ri, r.end_ms));
            }
        }
    }
    kept.sort();

    // 4. A visit is consecutive surviving runs of one track: another track's
    //    run, or a dropped run of its own, ends it.
    let mut visits: Vec<(usize, usize, VisitWindow)> = Vec::new(); // (candidate, last run index, window)
    for (start, ci, ri, end) in kept {
        if visits.last().is_some_and(|&(vc, last, _)| vc == ci && last + 1 == ri) {
            let v = visits.len() - 1;
            visits[v].1 = ri;
            visits[v].2.end_timestamp_ms = end;
        } else {
            let track_id = candidates[ci].track_id.to_string();
            visits.push((ci, ri, VisitWindow { track_id, start_timestamp_ms: start, end_timestamp_ms: end }));
        }
    }
    visits.into_iter().map(|(_, _, w)| w).collect()
}

/// Time-weighted mean over `[start_ms, end_ms]` of a per-fix value `q`
/// (linear between fixes, trapezoidal), in `q`'s units. `None` when no fix
/// interval overlaps the span.
fn time_mean(gps: &[GpsFix], q: &[f64], start_ms: i64, end_ms: i64) -> Option<f64> {
    let first = gps.partition_point(|f| f.timestamp_ms <= start_ms).saturating_sub(1);
    let (mut area, mut duration) = (0.0, 0.0);
    for i in first..gps.len().saturating_sub(1) {
        let (t0, t1) = (gps[i].timestamp_ms, gps[i + 1].timestamp_ms);
        if t0 >= end_ms {
            break;
        }
        if t1 <= start_ms || t1 <= t0 {
            continue;
        }
        let (a, b) = (t0.max(start_ms), t1.min(end_ms));
        let at = |t: i64| q[i] + (q[i + 1] - q[i]) * (t - t0) as f64 / (t1 - t0) as f64;
        area += 0.5 * (at(a) + at(b)) * (b - a) as f64;
        duration += (b - a) as f64;
    }
    (duration > 0.0).then(|| area / duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::synthetic::{gate, handle, then_walk, walk};
    use crate::laps::model::Gate;

    /// A reference polyline through `waypoints` (metres), a point every 5 m.
    fn polyline(waypoints: &[(f64, f64)]) -> Vec<GpsFix> {
        walk(0, 1000, 5.0, waypoints)
    }

    fn point_to_point(id: &str, waypoints: &[(f64, f64)], start: Gate, finish: Gate) -> TrackRef {
        TrackRef {
            track_id: id.to_string(),
            polyline: polyline(waypoints),
            timing: Some(LapTiming::PointToPoint { start, finish }),
        }
    }

    /// An east-west gate across north `n` metres, 40 m wide about east `e`.
    fn across(n: f64, e: f64) -> Gate {
        gate((n, e - 20.0), (n, e + 20.0))
    }

    /// A ride through `waypoints` (metres) at 10 m/s, 1 Hz.
    fn ride(waypoints: &[(f64, f64)]) -> SessionHandle {
        handle(&walk(0, 1000, 10.0, waypoints), 1.0)
    }

    // Two trails sharing 95 % of their ground, north up east 0 to north
    // 1000. Track A comes straight up from the south (its polyline 3 m west
    // of where these rides go); track B comes in from the east along north
    // 50 and turns north at east 0. A's gates are on the shared ground at
    // north 100 and 900; B's start is on its own spur, its finish at 950.

    fn track_a() -> TrackRef {
        point_to_point("A", &[(-100.0, -3.0), (1000.0, -3.0)], across(100.0, 0.0), across(900.0, 0.0))
    }

    fn track_b() -> TrackRef {
        let start = gate((40.0, 30.0), (60.0, 30.0));
        point_to_point("B", &[(50.0, 60.0), (50.0, 0.0), (1000.0, 0.0)], start, across(950.0, 0.0))
    }

    const A_ROUTE: [(f64, f64); 2] = [(-50.0, -3.0), (1000.0, -3.0)];
    const B_ROUTE: [(f64, f64); 3] = [(50.0, 100.0), (50.0, 0.0), (1000.0, 0.0)];

    #[test]
    fn detect_visits_two_tracks_sharing_their_ground_each_run_goes_to_the_track_whose_gates_it_crossed() {
        // Arrange — the A route never crosses B's start gate; the B route
        // crosses both tracks' gates, and B's polyline fits it closer (0 m
        // against A's 3 m) over the stretch where both runs overlap.
        let tracks = vec![track_a(), track_b()];

        // Act
        let on_a = detect_visits(&ride(&A_ROUTE), &tracks, VisitParams::default());
        let on_b = detect_visits(&ride(&B_ROUTE), &tracks, VisitParams::default());

        // Assert
        assert_eq!(on_a.iter().map(|v| v.track_id.as_str()).collect::<Vec<_>>(), vec!["A"]);
        assert_eq!(on_b.iter().map(|v| v.track_id.as_str()).collect::<Vec<_>>(), vec!["B"]);
    }

    #[test]
    fn detect_visits_a_run_crossing_both_gates_on_a_different_route_is_rejected() {
        // Arrange — through A's start gate, a long loop 200 m east, back
        // through A's finish gate: well under 80 % of the run near A's line.
        let route = [(-50.0, -3.0), (150.0, -3.0), (150.0, 200.0), (850.0, 200.0), (850.0, -3.0), (1000.0, -3.0)];
        let session = ride(&route);
        let tracks = vec![track_a()];

        // Act
        let strict = detect_visits(&session, &tracks, VisitParams::default());
        let any = detect_visits(&session, &tracks, VisitParams { min_on_track_fraction: 0.0, ..VisitParams::default() });

        // Assert — the gates were crossed (no fraction floor finds the run);
        // the polyline rejects it.
        assert!(strict.is_empty());
        assert_eq!(any.len(), 1);
    }

    #[test]
    fn detect_visits_a_multipath_spike_of_80_m_for_1_s_changes_nothing() {
        // Arrange — the A route with one fix, mid-run, thrown 80 m east.
        let clean = walk(0, 1000, 10.0, &A_ROUTE);
        let mut spiked = clean.clone();
        let (lat, lon) = crate::gps::synthetic::at((500.0, 77.0));
        let k = spiked.iter().position(|f| f.timestamp_ms == 55_000).unwrap();
        spiked[k].lat = lat;
        spiked[k].lon = lon;
        let tracks = vec![track_a()];

        // Act
        let before = detect_visits(&handle(&clean, 1.0), &tracks, VisitParams::default());
        let after = detect_visits(&handle(&spiked, 1.0), &tracks, VisitParams::default());

        // Assert
        assert_eq!(before.len(), 1);
        assert_eq!(after, before);
    }

    #[test]
    fn detect_visits_the_same_ride_at_1_hz_and_10_hz_gives_the_same_visits() {
        // Arrange — the B route at 10 m/s, sampled once and ten times a second.
        let tracks = vec![track_a(), track_b()];
        let one_hz = handle(&walk(0, 1000, 10.0, &B_ROUTE), 1.0);
        let ten_hz = handle(&walk(0, 100, 1.0, &B_ROUTE), 10.0);

        // Act
        let slow = detect_visits(&one_hz, &tracks, VisitParams::default());
        let fast = detect_visits(&ten_hz, &tracks, VisitParams::default());

        // Assert — same tracks, same bounds to the millisecond of rounding.
        assert_eq!(slow.len(), 1);
        assert_eq!(fast.len(), slow.len());
        for (s, f) in slow.iter().zip(&fast) {
            assert_eq!(s.track_id, f.track_id);
            assert!((s.start_timestamp_ms - f.start_timestamp_ms).abs() <= 1, "{s:?} vs {f:?}");
            assert!((s.end_timestamp_ms - f.end_timestamp_ms).abs() <= 1, "{s:?} vs {f:?}");
        }
    }

    #[test]
    fn detect_visits_two_point_to_point_trails_sharing_a_stretch_ridden_once_each_give_one_visit_each() {
        // Arrange — X runs north 0→600, Y north 300→900, sharing 300→600.
        // Ride X, loop round well clear of both, ride Y.
        let x = point_to_point("X", &[(0.0, 0.0), (600.0, 0.0)], across(100.0, 0.0), across(550.0, 0.0));
        let y = point_to_point("Y", &[(300.0, 0.0), (900.0, 0.0)], across(350.0, 0.0), across(850.0, 0.0));
        let mut fixes = walk(0, 1000, 10.0, &[(0.0, 0.0), (600.0, 0.0)]);
        then_walk(&mut fixes, 1000, 10.0, &[(600.0, 0.0), (600.0, 500.0), (300.0, 500.0), (300.0, 0.0), (900.0, 0.0)]);

        // Act
        let v = detect_visits(&handle(&fixes, 1.0), &[x, y], VisitParams::default());

        // Assert
        assert_eq!(v.iter().map(|v| v.track_id.as_str()).collect::<Vec<_>>(), vec!["X", "Y"]);
        assert!(v[0].end_timestamp_ms < v[1].start_timestamp_ms);
    }

    #[test]
    fn detect_visits_consecutive_circuit_laps_make_one_visit_from_the_first_crossing_to_the_last() {
        // Arrange — three times round a 1 km rectangle, start/finish across
        // the bottom straight.
        let lap = [(0.0, 300.0), (200.0, 300.0), (200.0, 0.0), (0.0, 0.0)];
        let mut route = vec![(0.0, 0.0)];
        for _ in 0..3 {
            route.extend(lap);
        }
        route.push((0.0, 200.0));
        let fixes = walk(0, 1000, 10.0, &route);
        let sf = gate((-20.0, 150.0), (20.0, 150.0));
        let track = TrackRef {
            track_id: "loop".to_string(),
            polyline: polyline(&[(0.0, 0.0), (0.0, 300.0), (200.0, 300.0), (200.0, 0.0), (0.0, 0.0)]),
            timing: Some(LapTiming::Circuit { start_finish: sf, out_lap_gate: None, in_lap_gate: None }),
        };

        // Act
        let v = detect_visits(&handle(&fixes, 1.0), &[track], VisitParams::default());

        // Assert — four crossings, three laps, one visit spanning them.
        let crossings = crate::laps::find_crossings(&fixes, &sf);
        assert_eq!(crossings.len(), 4);
        assert_eq!(v.len(), 1);
        assert_eq!((v[0].start_timestamp_ms, v[0].end_timestamp_ms), (crossings[0], crossings[3]));
    }

    #[test]
    fn detect_visits_track_with_no_timing_or_disjoint_bbox_yields_nothing() {
        // Arrange — A's polyline with no gates; A's gates on a polyline far away.
        let session = ride(&A_ROUTE);
        let mut no_gates = track_a();
        no_gates.timing = None;
        let mut far = track_a();
        far.polyline = far.polyline.iter().map(|f| GpsFix { lat: f.lat + 1.0, ..*f }).collect();

        // Act + Assert
        assert!(detect_visits(&session, &[no_gates], VisitParams::default()).is_empty());
        assert!(detect_visits(&session, &[far], VisitParams::default()).is_empty());
    }

    #[test]
    fn detect_visits_empty_when_too_few_fixes_or_no_tracks() {
        // Arrange
        let one_fix = handle(&walk(0, 1000, 10.0, &A_ROUTE)[..1], 1.0);
        let many = ride(&A_ROUTE);

        // Act + Assert — < 2 fixes → []; empty track list → [].
        assert!(detect_visits(&one_fix, &[track_a()], VisitParams::default()).is_empty());
        assert!(detect_visits(&many, &[], VisitParams::default()).is_empty());
    }

    #[test]
    fn detect_visits_walden_moto_1_with_the_pit_exit_as_out_lap_gate_is_one_visit_led_by_an_out_lap() {
        // Arrange — moto 1 at Walden (1 Hz GPS), and Walden '26 given the
        // Walden Out Lap '26 start gate (pit exit) as its out-lap gate: the
        // migration the track-matching spec describes.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
        let handle = SessionHandle::from_path(&format!("{dir}walden-moto1-gps.idl0")).expect("fixture parses");
        let read = |name: &str| crate::track_artifact::read_track(std::path::Path::new(&format!("{dir}{name}")));
        let mut walden = read("walden-26.idl1t").expect("Walden '26 reads");
        let Some(LapTiming::PointToPoint { start: pit_exit, .. }) = read("walden-outlap-26.idl1t").expect("out lap reads").timing
        else {
            panic!("Walden Out Lap '26 is point-to-point");
        };
        if let Some(LapTiming::Circuit { out_lap_gate, .. }) = &mut walden.timing {
            *out_lap_gate = Some(pit_exit);
        }
        let timing = walden.timing.clone().expect("Walden '26 has timing");

        // Act
        let visits = detect_visits(&handle, &[walden.track_ref()], VisitParams::default());
        let window = (visits[0].start_timestamp_ms, visits[0].end_timestamp_ms);
        let laps = crate::laps::detect_laps(&handle, &timing, &walden.sector_gates, &walden.neutral_zones, Some(window));

        // Assert — one visit: the out lap from pit exit, then the moto's
        // three start/finish-to-start/finish laps.
        use crate::laps::LapKind::{Normal, Out};
        assert_eq!(visits.len(), 1);
        assert_eq!(laps.iter().map(|l| l.kind).collect::<Vec<_>>(), vec![Out, Normal, Normal, Normal]);
        assert_eq!(laps.iter().map(|l| l.lap_time_ms).collect::<Vec<_>>(), vec![177_667, 166_131, 155_250, 152_674]);
        assert_eq!(window, (laps[0].start_ms, laps[3].end_ms));
    }
}
