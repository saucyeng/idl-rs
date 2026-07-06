//! Lap detection — gate-crossing geometry, circuit / point-to-point detection,
//! sectors, and neutral-zone subtraction. Pure: reads GPS from the session
//! handle (via `crate::gps`), takes the Track config as input, returns laps.
//! Ported from the Dart `LapDetector`.

pub mod detect;
pub mod geometry;
pub mod model;

pub use detect::detect_laps;
pub use geometry::find_crossings;
pub use model::{Gate, Lap, LapTiming, NeutralZone, NeutralZoneVisit, Sector, SectorGate};
