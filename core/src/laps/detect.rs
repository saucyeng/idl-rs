//! Lap detection over a GPS-fix list. Laps are gate-bounded runs: on a
//! circuit, start/finish crossing to the next (plus out-lap and in-lap runs
//! when the track has those gates); on a point-to-point track, a start
//! crossing to the next finish crossing (a second start discards the run in
//! progress). Then sector splits and neutral-zone subtraction. `detect_laps`
//! reads GPS from the handle and keeps the runs inside an optional visit
//! window.

use crate::gps::{build_gps_track, GpsFix};
use crate::laps::geometry::find_crossings;
use crate::laps::model::{Gate, Lap, LapKind, LapTiming, NeutralZone, NeutralZoneVisit, Sector, SectorGate};
use crate::session::handle::SessionHandle;

/// One gate-bounded candidate lap, before sectors and neutral zones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// Start crossing, milliseconds since the Unix epoch.
    pub start_ms: i64,
    /// End crossing, milliseconds since the Unix epoch.
    pub end_ms: i64,
    /// Normal, out or in lap.
    pub kind: LapKind,
}

/// Every gate-bounded run along `gps` under `timing`, in time order. The one
/// definition of where a lap starts and ends: lap detection and visit
/// detection (`crate::tracks::detect_visits`) both read it. On a circuit the
/// stretch before the first start/finish (or out-lap gate) crossing is not a
/// run: it has no start crossing, so it can't be timed.
pub fn detect_runs(gps: &[GpsFix], timing: &LapTiming) -> Vec<Run> {
    #[derive(Clone, Copy)]
    enum Event {
        StartFinish,
        Finish,
        Out,
        In,
    }
    let mut events: Vec<(i64, Event)> = Vec::new();
    let mut add = |gate: &Gate, event: Event| events.extend(find_crossings(gps, gate).into_iter().map(|t| (t, event)));
    match timing {
        LapTiming::Circuit { start_finish, out_lap_gate, in_lap_gate } => {
            add(start_finish, Event::StartFinish);
            if let Some(g) = out_lap_gate {
                add(g, Event::Out);
            }
            if let Some(g) = in_lap_gate {
                add(g, Event::In);
            }
        }
        LapTiming::PointToPoint { start, finish } => {
            // A point-to-point start opens a run exactly as a circuit's
            // start/finish does; only its finish is a separate gate.
            add(start, Event::StartFinish);
            add(finish, Event::Finish);
        }
    }
    // Stable, so a start pushed before a finish at the same millisecond stays first.
    events.sort_by_key(|e| e.0);

    let point_to_point = matches!(timing, LapTiming::PointToPoint { .. });
    let mut runs = Vec::new();
    let mut open: Option<(i64, LapKind)> = None; // (start_ms, kind) of the run in progress
    for (t, event) in events {
        match event {
            Event::StartFinish => {
                if let Some((s, kind)) = open.filter(|&(s, _)| !point_to_point && t > s) {
                    runs.push(Run { start_ms: s, end_ms: t, kind });
                }
                open = Some((t, LapKind::Normal));
            }
            Event::Finish => {
                if let Some((s, kind)) = open.filter(|&(s, _)| t > s) {
                    runs.push(Run { start_ms: s, end_ms: t, kind });
                    open = None;
                }
            }
            Event::Out => open = Some((t, LapKind::Out)),
            Event::In => {
                // Only a lap begun at start/finish ends at pit entry; an out
                // lap that turns straight back into the pits is no lap.
                if let Some((s, LapKind::Normal)) = open.filter(|&(s, _)| t > s) {
                    runs.push(Run { start_ms: s, end_ms: t, kind: LapKind::In });
                }
                open = None;
            }
        }
    }
    runs
}

/// Detect laps for `handle` using `timing` and the supplied gates. Runs are
/// found over the whole recording ([`detect_runs`]); `window`, when
/// `Some((start_ms, end_ms))`, keeps only the runs lying wholly inside that
/// inclusive window (a visit from `crate::tracks::detect_visits`). Laps are
/// numbered 1.. in the order kept. Empty when there are < 2 fixes or no run.
pub fn detect_laps(
    handle: &SessionHandle,
    timing: &LapTiming,
    sector_gates: &[SectorGate],
    neutral_zones: &[NeutralZone],
    window: Option<(i64, i64)>,
) -> Vec<Lap> {
    let gps = build_gps_track(handle);
    if gps.len() < 2 {
        return Vec::new();
    }
    let laps = detect_runs(&gps, timing)
        .into_iter()
        .filter(|r| window.map_or(true, |(start, end)| r.start_ms >= start && r.end_ms <= end))
        .enumerate()
        .map(|(i, r)| {
            let raw = r.end_ms - r.start_ms;
            Lap {
                lap_number: (i + 1) as u32,
                kind: r.kind,
                start_ms: r.start_ms,
                end_ms: r.end_ms,
                start_time_secs: 0.0,
                end_time_secs: 0.0,
                raw_elapsed_ms: raw,
                lap_time_ms: raw,
                sectors: compute_sectors(&gps, sector_gates, r.start_ms, r.end_ms),
                neutral_zone_visits: Vec::new(),
            }
        });
    let laps: Vec<Lap> = if neutral_zones.is_empty() {
        laps.collect()
    } else {
        laps.map(|l| apply_neutral_zones(l, &gps, neutral_zones)).collect()
    };
    fill_time_secs(handle, laps)
}

/// Stamp recording-time seconds onto every lap and sector boundary in one
/// `epoch_ms_to_time_secs` pass. Boundary epochs are gathered in a fixed order
/// (per lap: start, end, then each sector's start, end) and scattered back in
/// the same order.
fn fill_time_secs(handle: &SessionHandle, mut laps: Vec<Lap>) -> Vec<Lap> {
    let mut epochs: Vec<f64> = Vec::new();
    for lap in &laps {
        epochs.push(lap.start_ms as f64);
        epochs.push(lap.end_ms as f64);
        for s in &lap.sectors {
            epochs.push(s.start_ms as f64);
            epochs.push(s.end_ms as f64);
        }
    }
    let secs = handle.epoch_ms_to_time_secs(&epochs);
    let mut k = 0usize;
    for lap in &mut laps {
        lap.start_time_secs = secs[k];
        lap.end_time_secs = secs[k + 1];
        k += 2;
        for s in &mut lap.sectors {
            s.start_time_secs = secs[k];
            s.end_time_secs = secs[k + 1];
            k += 2;
        }
    }
    laps
}

fn compute_sectors(
    track: &[GpsFix],
    sector_gates: &[SectorGate],
    lap_start: i64,
    lap_end: i64,
) -> Vec<Sector> {
    if sector_gates.is_empty() {
        return Vec::new();
    }
    let lap_track: Vec<GpsFix> = track
        .iter()
        .copied()
        .filter(|f| f.timestamp_ms >= lap_start && f.timestamp_ms <= lap_end)
        .collect();
    let mut boundaries = vec![lap_start];
    for sg in sector_gates {
        let crossings = find_crossings(&lap_track, &sg.gate);
        if let Some(&first) = crossings.first() {
            boundaries.push(first);
        }
    }
    boundaries.push(lap_end);
    let mut sectors = Vec::with_capacity(boundaries.len() - 1);
    for i in 0..boundaries.len() - 1 {
        let name = sector_gates
            .get(i)
            .map(|sg| sg.name.clone())
            .unwrap_or_else(|| format!("S{}", i + 1));
        sectors.push(Sector {
            name,
            start_ms: boundaries[i],
            end_ms: boundaries[i + 1],
            start_time_secs: 0.0,
            end_time_secs: 0.0,
        });
    }
    sectors
}

fn apply_neutral_zones(lap: Lap, gps: &[GpsFix], neutral_zones: &[NeutralZone]) -> Lap {
    let mut visits: Vec<NeutralZoneVisit> = Vec::new();
    for zone in neutral_zones {
        let in_window = |t: &i64| *t >= lap.start_ms && *t <= lap.end_ms;
        let enters: Vec<i64> = find_crossings(gps, &zone.enter).into_iter().filter(in_window).collect();
        let exits: Vec<i64> = find_crossings(gps, &zone.exit).into_iter().filter(in_window).collect();
        let (mut i, mut j) = (0usize, 0usize);
        while i < enters.len() && j < exits.len() {
            let enter_ms = enters[i];
            while j < exits.len() && exits[j] <= enter_ms {
                j += 1; // skip exits before/at this enter (unpaired)
            }
            if j >= exits.len() {
                break;
            }
            visits.push(NeutralZoneVisit { name: zone.name.clone(), enter_ms, exit_ms: exits[j] });
            i += 1;
            j += 1;
        }
    }
    visits.sort_by_key(|v| v.enter_ms);
    let subtracted: i64 = visits.iter().map(|v| v.exit_ms - v.enter_ms).sum();
    Lap {
        lap_time_ms: lap.raw_elapsed_ms - subtracted,
        neutral_zone_visits: visits,
        ..lap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::synthetic::{gate, handle, then_stay, then_walk};
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    /// Builds a handle whose GPS channels encode `fixes` (timestamp_ms, lat, lon).
    fn handle_for(fixes: &[(i64, f64, f64)]) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
        };
        let lat: Vec<f64> = fixes.iter().map(|f| f.1).collect();
        let lon: Vec<f64> = fixes.iter().map(|f| f.2).collect();
        let epoch: Vec<f64> = fixes.iter().map(|f| f.0 as f64).collect();
        let mk = |id: &str, s: Vec<f64>| {
            let t_us = (0..s.len() as i64).map(|i| i * 1_000_000).collect();
            ChannelInput {
                channel_id: id.to_string(),
                sample_rate_hz: 1.0,
                samples: s,
                t_us,
                source_kind: id.to_lowercase(),
            }
        };
        SessionHandle::from_channels(
            meta,
            vec![mk("GPS_Latitude", lat), mk("GPS_Longitude", lon), mk("GPS_EpochMs", epoch)],
        )
    }

    // Northbound then southbound past the same east-west gate at lat 0.05.
    fn there_and_back() -> SessionHandle {
        let mut fixes = Vec::new();
        for i in 0..100 {
            fixes.push((1000 + i * 1000, i as f64 * 0.001, 0.0005));
        }
        for i in 0..100 {
            fixes.push((1000 + (100 + i) * 1000, 0.099 - i as f64 * 0.001, 0.0005));
        }
        handle_for(&fixes)
    }

    fn gate_at(lat: f64) -> Gate {
        Gate { lat1: lat, lon1: -0.001, lat2: lat, lon2: 0.001 }
    }

    /// A north-south gate at a given longitude (a longitude threshold); spans
    /// lat -10..10 so any track at a small latitude crosses it.
    fn lon_gate(lon: f64) -> Gate {
        Gate { lat1: -10.0, lon1: lon, lat2: 10.0, lon2: lon }
    }

    /// Circuit timing with no out-lap or in-lap gate.
    fn circuit(start_finish: Gate) -> LapTiming {
        LapTiming::Circuit { start_finish, out_lap_gate: None, in_lap_gate: None }
    }

    #[test]
    fn circuit_two_crossings_one_lap_between_them() {
        // Arrange
        let h = there_and_back();

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], None);

        // Assert — crossed northbound (~fix 50) and southbound (~fix 149):
        // one lap between, starting at the first crossing, not the first fix.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].lap_number, 1);
        assert_eq!(laps[0].kind, LapKind::Normal);
        assert!(laps[0].start_ms > 40_000);
        assert!(laps[0].lap_time_ms > 0);
    }

    #[test]
    fn circuit_no_crossing_zero_laps() {
        // Arrange — gate far north of the track.
        let h = there_and_back();

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(9.0)), &[], &[], None);

        // Assert
        assert!(laps.is_empty());
    }

    #[test]
    fn circuit_stretch_before_the_first_crossing_is_no_lap() {
        // Arrange — single northbound run 0.000→0.099; gate at 0.05 (fix 50).
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], None);

        // Assert — the one crossing has no start crossing before it, so no lap.
        assert!(laps.is_empty());
    }

    #[test]
    fn point_to_point_start_then_finish_one_lap() {
        // Arrange — northbound; start gate at 0.02, finish at 0.08.
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::PointToPoint { start: gate_at(0.02), finish: gate_at(0.08) };

        // Act
        let laps = detect_laps(&h, &timing, &[], &[], None);

        // Assert — one lap from the 0.02 crossing to the 0.08 crossing.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].start_ms, 1000 + 20 * 1000);
        assert_eq!(laps[0].end_ms, 1000 + 80 * 1000);
    }

    #[test]
    fn point_to_point_finish_before_start_zero_laps() {
        // Arrange — northbound; finish gate (0.02) is reached before start (0.08).
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::PointToPoint { start: gate_at(0.08), finish: gate_at(0.02) };

        // Act — finish at fix 20 ignored (no open lap); start at fix 80 opens, never finishes.
        let laps = detect_laps(&h, &timing, &[], &[], None);

        // Assert
        assert!(laps.is_empty());
    }

    #[test]
    fn sector_times_sum_to_lap_time() {
        // Arrange — there and back; start/finish at 0.02 (fixes 20 and 179),
        // one sector gate at 0.05 inside the lap.
        let h = there_and_back();
        let sectors = vec![SectorGate { name: "S1".to_string(), gate: gate_at(0.05) }];

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.02)), &sectors, &[], None);

        // Assert — two sectors; their spans sum to the lap span.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert_eq!(lap.sectors.len(), 2);
        let sum: i64 = lap.sectors.iter().map(|s| s.end_ms - s.start_ms).sum();
        assert_eq!(sum, lap.end_ms - lap.start_ms);
    }

    #[test]
    fn neutral_zone_subtracts_from_lap_time() {
        // Arrange — there and back, start/finish at 0.02; a neutral zone
        // entered at 0.03 and left at 0.06 on the way north. Southbound the
        // enter gate comes after every exit, so it pairs with nothing.
        let h = there_and_back();
        let nz = vec![NeutralZone {
            name: "Pit".to_string(),
            enter: gate_at(0.03),
            exit: gate_at(0.06),
        }];

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.02)), &[], &nz, None);

        // Assert — 30 s neutral (fix 30→60) subtracted; one recorded visit.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert_eq!(lap.neutral_zone_visits.len(), 1);
        assert_eq!(lap.lap_time_ms, lap.raw_elapsed_ms - 30 * 1000);
    }

    #[test]
    fn window_keeps_only_the_runs_inside_it() {
        // Arrange — there and back past a gate at 0.05: one lap.
        let h = there_and_back();
        let all = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], None);
        let (start, end) = (all[0].start_ms, all[0].end_ms);

        // Act — a window exactly the lap keeps it; one a millisecond short drops it.
        let inside = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], Some((start, end)));
        let short = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], Some((start + 1, end)));

        // Assert — the kept lap is the same lap, even though the fix before
        // its start crossing lies outside the window.
        assert_eq!(inside, all);
        assert!(short.is_empty());
    }

    #[test]
    fn point_to_point_second_start_before_finish_discards_in_progress() {
        // Arrange — fixes move in longitude (lat held at 1.0 to avoid (0,0)
        // sentinels). The rider crosses the start gate (lon 2) three times
        // before crossing finish (lon 8): the latest start wins.
        let lons = [0.0, 1.0, 3.0, 4.0, 1.5, 3.0, 5.0, 7.0, 9.0];
        let fixes: Vec<(i64, f64, f64)> =
            lons.iter().enumerate().map(|(i, &lon)| (i as i64 * 100, 1.0, lon)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::PointToPoint { start: lon_gate(2.0), finish: lon_gate(8.0) };

        // Act
        let laps = detect_laps(&h, &timing, &[], &[], None);

        // Assert — exactly one lap, starting after the first (discarded) start
        // crossing (~t=150); the kept lap starts at the last re-crossing.
        assert_eq!(laps.len(), 1);
        assert!(laps[0].start_ms > 200);
    }

    #[test]
    fn neutral_zone_unpaired_enter_does_not_subtract() {
        // Arrange — out to lon -1 and back across a start/finish at lon 0
        // (one lap, t=500→1500); the enter gate (lon -0.6) is crossed inside
        // the lap but the exit (lon 99) never is, so nothing is subtracted.
        let h = handle_for(&[(0, 1.0, 1.0), (1000, 1.0, -1.0), (2000, 1.0, 1.0)]);
        let nz = vec![NeutralZone {
            name: "Pit".to_string(),
            enter: lon_gate(-0.6),
            exit: lon_gate(99.0),
        }];

        // Act
        let laps = detect_laps(&h, &circuit(lon_gate(0.0)), &[], &nz, None);

        // Assert — lap_time unchanged; no recorded visit.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].lap_time_ms, laps[0].raw_elapsed_ms);
        assert!(laps[0].neutral_zone_visits.is_empty());
    }

    #[test]
    fn detect_laps_populates_recording_seconds() {
        // Arrange — there and back, start/finish at 0.02 (fixes 20 and 179),
        // one sector gate at 0.05 (fix 50). GPS is 1 Hz from epoch 1000ms, so
        // fix k is at recording-second k.
        let h = there_and_back();
        let sectors = vec![SectorGate { name: "S1".to_string(), gate: gate_at(0.05) }];

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.02)), &sectors, &[], None);

        // Assert — lap 1: fix 20 → 20.0 s, fix 179 → 179.0 s. Sector split
        // at fix 50 → 50.0 s.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert!((lap.start_time_secs - 20.0).abs() < 1e-6);
        assert!((lap.end_time_secs - 179.0).abs() < 1e-6);
        assert_eq!(lap.sectors.len(), 2);
        assert!((lap.sectors[0].start_time_secs - 20.0).abs() < 1e-6);
        assert!((lap.sectors[0].end_time_secs - 50.0).abs() < 1e-6);
        assert!((lap.sectors[1].start_time_secs - 50.0).abs() < 1e-6);
        assert!((lap.sectors[1].end_time_secs - 179.0).abs() < 1e-6);
    }

    #[test]
    fn fewer_than_two_fixes_is_empty() {
        // Arrange — a single non-sentinel fix.
        let h = handle_for(&[(1000, 1.0, 1.0)]);

        // Act
        let laps = detect_laps(&h, &circuit(gate_at(0.05)), &[], &[], None);

        // Assert
        assert!(laps.is_empty());
    }

    // ---- out-lap and in-lap gates (metres, see `crate::gps::synthetic`) ----
    //
    // A 1 km rectangular circuit, (north, east) metres: east along north 0,
    // up the east side, back along north 200, down the west side. Start/
    // finish crosses the bottom straight at east 150. The pit box sits south
    // of the track at (-40, 100); the exit lane runs east and joins the track
    // at (0, 230), past start/finish; the entry lane leaves it at (0, 60),
    // before.

    const LOOP: [(f64, f64); 4] = [(0.0, 300.0), (200.0, 300.0), (200.0, 0.0), (0.0, 0.0)];
    const PIT_BOX: (f64, f64) = (-40.0, 100.0);

    fn start_finish() -> Gate {
        gate((-20.0, 150.0), (20.0, 150.0))
    }

    /// Across the pit exit lane at east 180.
    fn pit_exit() -> Gate {
        gate((-55.0, 180.0), (-25.0, 180.0))
    }

    /// Across the pit entry lane at north -20.
    fn pit_entry() -> Gate {
        gate((-20.0, 50.0), (-20.0, 80.0))
    }

    /// One stint's waypoints, pit box to pit box: out of the pits, round to
    /// start/finish, `crossings` crossings of it in all, back in.
    fn stint(crossings: usize) -> Vec<(f64, f64)> {
        let mut w = vec![PIT_BOX, (-40.0, 200.0), (0.0, 230.0), (0.0, 300.0), (200.0, 300.0), (200.0, 0.0), (0.0, 0.0)];
        for _ in 0..crossings {
            w.extend(LOOP);
        }
        w.extend([(0.0, 60.0), (-40.0, 70.0), PIT_BOX]);
        w
    }

    /// `stints` stints of three start/finish crossings each, 60 s in the pit
    /// box before each, 10 m/s at 1 Hz.
    fn ride(stints: usize) -> Vec<GpsFix> {
        let mut fixes = Vec::new();
        for _ in 0..stints {
            then_stay(&mut fixes, 1000, 60, PIT_BOX);
            then_walk(&mut fixes, 1000, 10.0, &stint(3));
        }
        fixes
    }

    fn with_gates(out_lap_gate: Option<Gate>, in_lap_gate: Option<Gate>) -> LapTiming {
        LapTiming::Circuit { start_finish: start_finish(), out_lap_gate, in_lap_gate }
    }

    #[test]
    fn out_lap_gate_times_the_out_lap_from_the_gate_to_start_finish_and_marks_it() {
        // Arrange
        let fixes = ride(1);
        let timing = with_gates(Some(pit_exit()), None);

        // Act
        let laps = detect_laps(&handle(&fixes, 1.0), &timing, &[], &[], None);

        // Assert — out lap (pit exit → first S/F), then two ordinary laps;
        // the run open after the last S/F never closes.
        let sf = find_crossings(&fixes, &start_finish());
        assert_eq!(laps.len(), 3);
        assert_eq!(laps[0].kind, LapKind::Out);
        assert_eq!(laps[0].lap_number, 1);
        assert_eq!(laps[0].start_ms, find_crossings(&fixes, &pit_exit())[0]);
        assert_eq!(laps[0].end_ms, sf[0]);
        assert_eq!(laps[0].lap_time_ms, sf[0] - laps[0].start_ms);
        assert_eq!(laps[1].kind, LapKind::Normal);
        assert_eq!((laps[2].start_ms, laps[2].end_ms), (sf[1], sf[2]));
    }

    #[test]
    fn out_lap_gate_two_stints_give_two_out_laps_numbered_through() {
        // Arrange
        let fixes = ride(2);
        let timing = with_gates(Some(pit_exit()), Some(pit_entry()));

        // Act
        let laps = detect_laps(&handle(&fixes, 1.0), &timing, &[], &[], None);

        // Assert — out, two laps, in, per stint; numbers run through.
        use LapKind::{In, Normal, Out};
        let kinds: Vec<LapKind> = laps.iter().map(|l| l.kind).collect();
        assert_eq!(kinds, vec![Out, Normal, Normal, In, Out, Normal, Normal, In]);
        let numbers: Vec<u32> = laps.iter().map(|l| l.lap_number).collect();
        assert_eq!(numbers, (1..=8).collect::<Vec<u32>>());
    }

    #[test]
    fn no_out_lap_gate_gives_no_out_lap_and_laps_from_the_first_start_finish_crossing() {
        // Arrange
        let fixes = ride(1);

        // Act
        let laps = detect_laps(&handle(&fixes, 1.0), &with_gates(None, None), &[], &[], None);

        // Assert
        let sf = find_crossings(&fixes, &start_finish());
        assert_eq!(laps.len(), 2);
        assert!(laps.iter().all(|l| l.kind == LapKind::Normal));
        assert_eq!(laps[0].start_ms, sf[0]);
    }

    #[test]
    fn in_lap_gate_marks_the_last_lap_ending_at_pit_entry_in() {
        // Arrange
        let fixes = ride(1);

        // Act
        let laps = detect_laps(&handle(&fixes, 1.0), &with_gates(None, Some(pit_entry())), &[], &[], None);

        // Assert — two ordinary laps, then last S/F → pit entry as `in`.
        let sf = find_crossings(&fixes, &start_finish());
        assert_eq!(laps.len(), 3);
        let last = &laps[2];
        assert_eq!(last.kind, LapKind::In);
        assert_eq!(last.start_ms, sf[2]);
        assert_eq!(last.end_ms, find_crossings(&fixes, &pit_entry())[0]);
    }

    #[test]
    fn no_in_lap_gate_leaves_the_last_lap_normal() {
        // Arrange
        let fixes = ride(1);

        // Act
        let laps = detect_laps(&handle(&fixes, 1.0), &with_gates(None, None), &[], &[], None);

        // Assert — the last lap is the last S/F → S/F, ordinary.
        let sf = find_crossings(&fixes, &start_finish());
        let last = laps.last().unwrap();
        assert_eq!(last.kind, LapKind::Normal);
        assert_eq!(last.end_ms, sf[2]);
    }
}
