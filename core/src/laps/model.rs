//! Lap-detection types. Inputs (gates / timing config) mirror the Dart Track
//! config; outputs (laps / sectors / neutral-zone visits) mirror the Dart
//! result models. Coordinates are the raw GPS-channel scale (degrees × 1e7) —
//! the crossing geometry (see `geometry`) is scale-invariant, so nothing is
//! rescaled.

use serde::Serialize;

/// A gate line segment, two GPS posts (raw degrees × 1e7).
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
    /// One gate; lap n runs between consecutive crossings.
    Circuit { start_finish: Gate },
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

/// One complete lap.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Lap {
    pub lap_number: u32,
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
