//! Lap detection — gate-crossing geometry, circuit / point-to-point detection,
//! sectors, and neutral-zone subtraction. Pure: reads GPS from the session
//! handle (via `crate::gps`), takes the Track config as input, returns laps.
//! Ported from the Dart `LapDetector`.

pub mod detect;
pub mod distance;
pub mod gate_synthesis;
pub mod geometry;
pub mod model;
pub mod renumber;

pub use detect::detect_laps;
pub use distance::{GateCrossing, LapDistanceAccumulator, LapDistanceError, LapDistanceErrorKind};
pub use gate_synthesis::{
    endpoint_gates, endpoint_gates_default, perpendicular_gate_at, snap_to_nearest_fix, GateSynthesisError,
    GateSynthesisErrorKind,
};
pub use geometry::find_crossings;
pub use model::{Gate, Lap, LapTiming, NeutralZone, NeutralZoneVisit, Sector, SectorGate};
pub use renumber::{renumber_session_laps, RenumberedLap};
