//! What a track's gates catch on one session, before the track is saved
//! (track editor spec §3, `preview_track_runs`): the runs, and the GPS line a
//! new track takes as its reference polyline.

use crate::gps::{build_gps_track, GpsFix};
use crate::laps::detect::detect_laps;
use crate::laps::model::{Lap, LapTiming, SectorGate};
use crate::session::handle::SessionHandle;

/// [`preview_runs`]' answer.
#[derive(Debug, Clone, PartialEq)]
pub struct RunPreview {
    /// Every run the gates catch on the session, as lap detection numbers
    /// them.
    pub runs: Vec<Lap>,
    /// The session's GPS fixes over the reference run (every fix, not
    /// decimated); empty when there is no run.
    pub reference_polyline: Vec<GpsFix>,
}

/// The runs `timing` and `sector_gates` catch on `handle`, and the fixes of
/// its reference run. Point-to-point: the first run, start gate to finish
/// gate. Circuit: the first lap that starts on a crossing (lap 2 — lap 1 runs
/// from the session's first fix), falling back to lap 1 when there is only
/// one. Neutral zones play no part: they shorten a lap's time, not its line.
pub fn preview_runs(handle: &SessionHandle, timing: &LapTiming, sector_gates: &[SectorGate]) -> RunPreview {
    let runs = detect_laps(handle, timing, sector_gates, &[], None);
    let reference = match timing {
        LapTiming::PointToPoint { .. } => runs.first(),
        LapTiming::Circuit { .. } => runs.get(1).or(runs.first()),
    };
    let reference_polyline = match reference {
        Some(run) => build_gps_track(handle)
            .into_iter()
            .filter(|f| f.timestamp_ms >= run.start_ms && f.timestamp_ms <= run.end_ms)
            .collect(),
        None => Vec::new(),
    };
    RunPreview { runs, reference_polyline }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laps::model::Gate;
    use crate::session::handle::{ChannelInput, SessionMetaInput};

    /// A handle whose GPS encodes `fixes` (timestamp_ms, lat, lon), 1 Hz.
    fn handle_for(fixes: &[(i64, f64, f64)]) -> SessionHandle {
        let meta = SessionMetaInput { session_id: String::new(), device_id: None, timestamp_utc_ms: 0, config_checksum: None };
        let mk = |id: &str, s: Vec<f64>| ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: 1.0,
            t_us: (0..s.len() as i64).map(|i| i * 1_000_000).collect(),
            samples: s,
            source_kind: "gps".to_string(),
        };
        SessionHandle::from_channels(
            meta,
            vec![
                mk("GPS_Latitude", fixes.iter().map(|f| f.1).collect()),
                mk("GPS_Longitude", fixes.iter().map(|f| f.2).collect()),
                mk("GPS_EpochMs", fixes.iter().map(|f| f.0 as f64).collect()),
            ],
        )
    }

    /// Northbound, lat 0.000 → 0.099 at one fix per second from t = 1 s.
    fn northbound() -> SessionHandle {
        let fixes: Vec<(i64, f64, f64)> = (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        handle_for(&fixes)
    }

    fn gate_at(lat: f64) -> Gate {
        Gate { lat1: lat, lon1: -0.001, lat2: lat, lon2: 0.001 }
    }

    #[test]
    fn preview_runs_point_to_point_reference_line_is_the_fixes_between_the_gates() {
        // Arrange
        let h = northbound();
        let timing = LapTiming::PointToPoint { start: gate_at(0.02), finish: gate_at(0.08) };

        // Act
        let preview = preview_runs(&h, &timing, &[]);

        // Assert — one run; the line is fixes 20..=80, the gates sitting on fixes.
        assert_eq!(preview.runs.len(), 1);
        let run = &preview.runs[0];
        assert!(preview.reference_polyline.iter().all(|f| f.timestamp_ms >= run.start_ms && f.timestamp_ms <= run.end_ms));
        assert_eq!(preview.reference_polyline.first().unwrap().timestamp_ms, run.start_ms);
        assert_eq!(preview.reference_polyline.last().unwrap().timestamp_ms, run.end_ms);
        assert_eq!(preview.reference_polyline.len(), 61);
    }

    #[test]
    fn preview_runs_gates_the_ride_never_crosses_no_runs_and_no_line() {
        // Arrange
        let h = northbound();
        let timing = LapTiming::PointToPoint { start: gate_at(5.0), finish: gate_at(6.0) };

        // Act
        let preview = preview_runs(&h, &timing, &[]);

        // Assert
        assert!(preview.runs.is_empty());
        assert!(preview.reference_polyline.is_empty());
    }

    #[test]
    fn preview_runs_circuit_reference_line_is_the_first_lap_between_two_crossings() {
        // Arrange — out and back past one gate: two crossings, so lap 2 runs crossing to crossing.
        let mut fixes: Vec<(i64, f64, f64)> = (0..100).map(|i| (1000 + i * 1000, i as f64 * 0.001, 0.0005)).collect();
        fixes.extend((0..100).map(|i| (1000 + (100 + i) * 1000, 0.099 - i as f64 * 0.001, 0.0005)));
        let h = handle_for(&fixes);
        let timing = LapTiming::Circuit { start_finish: gate_at(0.05) };

        // Act
        let preview = preview_runs(&h, &timing, &[]);

        // Assert
        assert_eq!(preview.runs.len(), 2);
        assert_eq!(preview.reference_polyline.first().unwrap().timestamp_ms, preview.runs[1].start_ms);
    }
}
