//! Lap detection over a GPS-fix list. Verbatim port of the Dart `LapDetector`:
//! circuit (consecutive crossings of one gate) and point-to-point (start→finish
//! state machine with discard-in-progress), sector splits, and neutral-zone
//! subtraction. `detect_laps` reads GPS from the handle and applies an optional
//! visit window before detecting.

use crate::gps::{build_gps_track, GpsFix};
use crate::laps::geometry::find_crossings;
use crate::laps::model::{Gate, Lap, LapTiming, NeutralZone, NeutralZoneVisit, Sector, SectorGate};
use crate::session::handle::SessionHandle;

/// Detect laps for `handle` using `timing` and the supplied gates. `window`,
/// when `Some((start_ms, end_ms))`, restricts detection to fixes inside that
/// inclusive window (the §17 multi-track visit case) — lap 1 then starts at the
/// window's first fix. Empty when there are < 2 fixes or no start-gate crossing.
pub fn detect_laps(
    handle: &SessionHandle,
    timing: &LapTiming,
    sector_gates: &[SectorGate],
    neutral_zones: &[NeutralZone],
    window: Option<(i64, i64)>,
) -> Vec<Lap> {
    let mut gps = build_gps_track(handle);
    if let Some((start, end)) = window {
        gps.retain(|f| f.timestamp_ms >= start && f.timestamp_ms <= end);
    }
    if gps.len() < 2 {
        return Vec::new();
    }
    let laps = match timing {
        LapTiming::Circuit { start_finish } => detect_circuit(&gps, start_finish, sector_gates),
        LapTiming::PointToPoint { start, finish } => {
            detect_point_to_point(&gps, start, finish, sector_gates)
        }
    };
    let laps = if neutral_zones.is_empty() {
        laps
    } else {
        laps.into_iter().map(|l| apply_neutral_zones(l, &gps, neutral_zones)).collect()
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

fn detect_circuit(track: &[GpsFix], gate: &Gate, sector_gates: &[SectorGate]) -> Vec<Lap> {
    let crossings = find_crossings(track, gate);
    if crossings.is_empty() {
        return Vec::new();
    }
    let mut laps = Vec::with_capacity(crossings.len());
    let mut lap_start = track[0].timestamp_ms;
    for (i, &lap_end) in crossings.iter().enumerate() {
        let raw = lap_end - lap_start;
        let sectors = compute_sectors(track, sector_gates, lap_start, lap_end);
        laps.push(Lap {
            lap_number: (i + 1) as u32,
            start_ms: lap_start,
            end_ms: lap_end,
            start_time_secs: 0.0,
            end_time_secs: 0.0,
            raw_elapsed_ms: raw,
            lap_time_ms: raw,
            sectors,
            neutral_zone_visits: Vec::new(),
        });
        lap_start = lap_end;
    }
    laps
}

fn detect_point_to_point(
    track: &[GpsFix],
    start_gate: &Gate,
    finish_gate: &Gate,
    sector_gates: &[SectorGate],
) -> Vec<Lap> {
    let mut laps = Vec::new();
    let mut lap_number: u32 = 1;
    let mut lap_start: Option<i64> = None; // None = waiting for start
    for w in track.windows(2) {
        let seg = [w[0], w[1]];
        // Chronologically-ordered (timestamp, is_start) events on this segment.
        let mut events: Vec<(i64, bool)> = Vec::new();
        for t in find_crossings(&seg, start_gate) {
            events.push((t, true));
        }
        for t in find_crossings(&seg, finish_gate) {
            events.push((t, false));
        }
        events.sort_by_key(|e| e.0);
        for (t, is_start) in events {
            if is_start {
                lap_start = Some(t); // reset whether waiting or in-lap
            } else if let Some(ls) = lap_start {
                if t > ls {
                    let raw = t - ls;
                    let sectors = compute_sectors(track, sector_gates, ls, t);
                    laps.push(Lap {
                        lap_number,
                        start_ms: ls,
                        end_ms: t,
                        start_time_secs: 0.0,
                        end_time_secs: 0.0,
                        raw_elapsed_ms: raw,
                        lap_time_ms: raw,
                        sectors,
                        neutral_zone_visits: Vec::new(),
                    });
                    lap_number += 1;
                    lap_start = None;
                }
            }
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
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    /// Builds a handle whose GPS channels encode `fixes` (timestamp_ms, lat, lon).
    fn handle_for(fixes: &[(i64, f64, f64)]) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let lat: Vec<f64> = fixes.iter().map(|f| f.1).collect();
        let lon: Vec<f64> = fixes.iter().map(|f| f.2).collect();
        let epoch: Vec<f64> = fixes.iter().map(|f| f.0 as f64).collect();
        let mk = |id: &str, s: Vec<f64>| ChannelInput {
            channel_id: id.to_string(), sample_rate_hz: 1.0, samples: s, sample_times_secs: None,
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

    #[test]
    fn circuit_two_crossings_two_laps() {
        // Arrange
        let h = there_and_back();
        let timing = LapTiming::Circuit { start_finish: gate_at(0.05) };

        // Act
        let laps = detect_laps(&h, &timing, &[], &[], None);

        // Assert — crossed northbound (~fix 50) and southbound (~fix 150).
        assert_eq!(laps.len(), 2);
        assert_eq!(laps[0].lap_number, 1);
        assert_eq!(laps[0].start_ms, 1000); // first fix
        assert_eq!(laps[1].start_ms, laps[0].end_ms);
        assert!(laps[0].lap_time_ms > 0 && laps[1].lap_time_ms > 0);
    }

    #[test]
    fn circuit_no_crossing_zero_laps() {
        // Arrange — gate far north of the track.
        let h = there_and_back();
        let laps = detect_laps(&h, &LapTiming::Circuit { start_finish: gate_at(9.0) }, &[], &[], None);

        // Assert
        assert!(laps.is_empty());
    }

    #[test]
    fn circuit_retroactively_scores_lap_one() {
        // Arrange — single northbound run 0.000→0.099; gate at 0.05 (fix 50).
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);

        // Act
        let laps = detect_laps(&h, &LapTiming::Circuit { start_finish: gate_at(0.05) }, &[], &[], None);

        // Assert — one lap, session start → exact fix-50 timestamp.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].start_ms, 1000);
        assert_eq!(laps[0].end_ms, 1000 + 50 * 1000);
        assert_eq!(laps[0].lap_time_ms, 50 * 1000);
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
        // Arrange — single run; start/finish at 0.09, one sector gate at 0.05.
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::Circuit { start_finish: gate_at(0.09) };
        let sectors = vec![SectorGate { name: "S1".to_string(), gate: gate_at(0.05) }];

        // Act
        let laps = detect_laps(&h, &timing, &sectors, &[], None);

        // Assert — two sectors; their spans sum to the lap span.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert_eq!(lap.sectors.len(), 2);
        let sum: i64 = lap.sectors.iter().map(|s| s.end_ms - s.start_ms).sum();
        assert_eq!(sum, lap.end_ms - lap.start_ms);
    }

    #[test]
    fn neutral_zone_subtracts_from_lap_time() {
        // Arrange — circuit lap 0→0.09; a neutral zone enter at 0.03, exit at 0.06
        // (each a gate the northbound track crosses once within the lap).
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::Circuit { start_finish: gate_at(0.09) };
        let nz = vec![NeutralZone {
            name: "Pit".to_string(),
            enter: gate_at(0.03),
            exit: gate_at(0.06),
        }];

        // Act
        let laps = detect_laps(&h, &timing, &[], &nz, None);

        // Assert — 30 s neutral (fix 30→60) subtracted; one recorded visit.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert_eq!(lap.neutral_zone_visits.len(), 1);
        assert_eq!(lap.lap_time_ms, lap.raw_elapsed_ms - 30 * 1000);
    }

    #[test]
    fn window_restricts_detection_and_anchors_lap_one() {
        // Arrange — there-and-back; window covers only the southbound leg, so only
        // the southbound crossing scores, and lap 1 starts at the window's first
        // fix, not the session start.
        let h = there_and_back();
        let timing = LapTiming::Circuit { start_finish: gate_at(0.05) };
        let win_start = 1000 + 110 * 1000; // into the southbound leg

        // Act
        let laps = detect_laps(&h, &timing, &[], &[], Some((win_start, 1000 + 199 * 1000)));

        // Assert — one lap, anchored at the window's first fix.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].start_ms, win_start);
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
        // Arrange — circuit gate at lon 0; enter (lon 0.6) is crossed but exit
        // (lon 99) never is, so the enter is unpaired → no subtraction.
        let h = handle_for(&[(0, 1.0, 1.0), (1000, 1.0, -1.0)]);
        let timing = LapTiming::Circuit { start_finish: lon_gate(0.0) };
        let nz = vec![NeutralZone {
            name: "Pit".to_string(),
            enter: lon_gate(0.6),
            exit: lon_gate(99.0),
        }];

        // Act
        let laps = detect_laps(&h, &timing, &[], &nz, None);

        // Assert — lap_time unchanged; no recorded visit.
        assert_eq!(laps.len(), 1);
        assert_eq!(laps[0].lap_time_ms, laps[0].raw_elapsed_ms);
        assert!(laps[0].neutral_zone_visits.is_empty());
    }

    #[test]
    fn detect_laps_populates_recording_seconds() {
        // Arrange — single northbound run, start/finish at 0.09 (fix 90), one
        // sector gate at 0.05 (fix 50). GPS is 1 Hz from epoch 1000ms, so fix k
        // is at recording-second k.
        let fixes: Vec<(i64, f64, f64)> =
            (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        let h = handle_for(&fixes);
        let timing = LapTiming::Circuit { start_finish: gate_at(0.09) };
        let sectors = vec![SectorGate { name: "S1".to_string(), gate: gate_at(0.05) }];

        // Act
        let laps = detect_laps(&h, &timing, &sectors, &[], None);

        // Assert — lap 1: start fix 0 → 0.0 s, end fix 90 → 90.0 s. Sector split
        // at fix 50 → 50.0 s.
        assert_eq!(laps.len(), 1);
        let lap = &laps[0];
        assert!((lap.start_time_secs - 0.0).abs() < 1e-6);
        assert!((lap.end_time_secs - 90.0).abs() < 1e-6);
        assert_eq!(lap.sectors.len(), 2);
        assert!((lap.sectors[0].start_time_secs - 0.0).abs() < 1e-6);
        assert!((lap.sectors[0].end_time_secs - 50.0).abs() < 1e-6);
        assert!((lap.sectors[1].start_time_secs - 50.0).abs() < 1e-6);
        assert!((lap.sectors[1].end_time_secs - 90.0).abs() < 1e-6);
    }

    #[test]
    fn fewer_than_two_fixes_is_empty() {
        // Arrange — a single non-sentinel fix.
        let h = handle_for(&[(1000, 1.0, 1.0)]);
        let laps = detect_laps(&h, &LapTiming::Circuit { start_finish: gate_at(0.05) }, &[], &[], None);

        // Assert
        assert!(laps.is_empty());
    }
}
