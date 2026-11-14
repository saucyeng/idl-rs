//! Lap-detection types. Inputs (gates / timing config) mirror the Dart Track
//! config; outputs (laps / sectors / neutral-zone visits) mirror the Dart
//! result models. Coordinates are the GPS-channel scale — physical decimal
//! degrees (ruling R27) — the crossing geometry (see `geometry`) is
//! scale-invariant, so nothing is rescaled.

use serde::{Deserialize, Serialize};

/// A gate line segment, two GPS posts (physical decimal degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
}

/// How a lap is bounded.
#[derive(Debug, Clone)]
pub enum LapTiming {
    /// One gate; lap n runs between consecutive crossings. `out_lap_gate`
    /// (pit exit) starts a timed [`LapKind::Out`] lap that ends at the next
    /// start/finish crossing; `in_lap_gate` (pit entry) ends a lap begun at a
    /// start/finish crossing as [`LapKind::In`]. `None` = no such gate.
    Circuit { start_finish: Gate, out_lap_gate: Option<Gate>, in_lap_gate: Option<Gate> },
    /// Two gates; lap = next start crossing → next finish crossing.
    PointToPoint { start: Gate, finish: Gate },
}

/// A named sector boundary gate.
#[derive(Debug, Clone)]
pub struct SectorGate {
    pub name: String,
    pub gate: Gate,
}

/// A region whose duration is excluded from lap time (enter→exit pair).
#[derive(Debug, Clone)]
pub struct NeutralZone {
    pub name: String,
    pub enter: Gate,
    pub exit: Gate,
}

/// One sector split within a lap.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sector {
    pub name: String,
    pub start_ms: i64,
    pub end_ms: i64,
    /// Recording-time seconds of `start_ms` (engine-computed; see
    /// `SessionHandle::epoch_ms_to_time_secs`).
    pub start_time_secs: f64,
    /// Recording-time seconds of `end_ms`.
    pub end_time_secs: f64,
}

/// One detected enter→exit pair within a lap.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeutralZoneVisit {
    pub name: String,
    pub enter_ms: i64,
    pub exit_ms: i64,
}

/// What kind of lap a lap is. Out, in and long laps are listed but skipped
/// by "fastest" and by averages, like an ignored lap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LapKind {
    /// An ordinary timed lap, start/finish to start/finish (or start to
    /// finish on a point-to-point track).
    #[default]
    Normal,
    /// Out-lap gate (pit exit) to the next start/finish crossing.
    Out,
    /// Start/finish crossing to the in-lap gate (pit entry).
    In,
    /// A circuit lap more than [`LONG_LAP_FACTOR`] times the median of the
    /// session's other normal laps: most likely two laps merged by a missed
    /// start/finish crossing (a GPS dropout at the line), or a stop.
    Long,
}

/// How many times the median normal lap a lap must take to be
/// [`LapKind::Long`] (a merged double lap is ~2x; a slow lap is rarely 1.6x).
pub const LONG_LAP_FACTOR: f64 = 1.6;

/// Marks every normal lap longer than [`LONG_LAP_FACTOR`] times the median
/// normal `lap_time_ms` as [`LapKind::Long`]. Needs at least three normal
/// laps for the median to mean anything; with fewer, nothing changes.
pub fn mark_long_laps(laps: &mut [Lap]) {
    let mut times: Vec<i64> = laps.iter().filter(|l| l.kind.is_normal()).map(|l| l.lap_time_ms).collect();
    if times.len() < 3 {
        return;
    }
    times.sort_unstable();
    let median = times[times.len() / 2] as f64;
    for lap in laps.iter_mut().filter(|l| l.kind.is_normal()) {
        if lap.lap_time_ms as f64 > LONG_LAP_FACTOR * median {
            lap.kind = LapKind::Long;
        }
    }
}

impl LapKind {
    /// `true` for [`LapKind::Normal`] (`session.json` omits the key then).
    pub fn is_normal(&self) -> bool {
        *self == LapKind::Normal
    }
}

/// One complete lap.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Lap {
    pub lap_number: u32,
    /// Normal, out, in or long lap.
    pub kind: LapKind,
    pub start_ms: i64,
    pub end_ms: i64,
    /// Recording-time seconds of `start_ms` (engine-computed).
    pub start_time_secs: f64,
    /// Recording-time seconds of `end_ms`.
    pub end_time_secs: f64,
    pub raw_elapsed_ms: i64,
    pub lap_time_ms: i64,
    pub sectors: Vec<Sector>,
    pub neutral_zone_visits: Vec<NeutralZoneVisit>,
}

#[cfg(test)]
mod long_lap_tests {
    use super::*;

    fn lap(kind: LapKind, secs: i64) -> Lap {
        Lap {
            lap_number: 0,
            kind,
            start_ms: 0,
            end_ms: secs * 1000,
            start_time_secs: 0.0,
            end_time_secs: 0.0,
            raw_elapsed_ms: secs * 1000,
            lap_time_ms: secs * 1000,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }
    }

    #[test]
    fn mark_long_laps_a_merged_double_lap_among_normal_laps_is_long_and_the_rest_are_untouched() {
        // Arrange — the 14:25 moto: ~138 s laps and one 273 s lap.
        let mut laps = vec![lap(LapKind::Out, 300), lap(LapKind::Normal, 138), lap(LapKind::Normal, 273), lap(LapKind::Normal, 140), lap(LapKind::Normal, 136)];

        // Act
        mark_long_laps(&mut laps);

        // Assert
        let kinds: Vec<LapKind> = laps.iter().map(|l| l.kind).collect();
        assert_eq!(kinds, [LapKind::Out, LapKind::Normal, LapKind::Long, LapKind::Normal, LapKind::Normal]);
    }

    #[test]
    fn mark_long_laps_fewer_than_three_normal_laps_changes_nothing() {
        // Arrange
        let mut laps = vec![lap(LapKind::Normal, 138), lap(LapKind::Normal, 400)];

        // Act
        mark_long_laps(&mut laps);

        // Assert
        assert!(laps.iter().all(|l| l.kind == LapKind::Normal));
    }

    #[test]
    fn long_lap_kind_serialises_as_long() {
        // Act
        let text = serde_json::to_string(&LapKind::Long).unwrap();

        // Assert
        assert_eq!(text, "\"long\"");
    }
}
