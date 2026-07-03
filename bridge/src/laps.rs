//! FRB wrapper for `idl_rs::laps`. The Track config crosses in as freezed-free
//! args (a `kind` enum + gate fields); the lap table crosses back as mirrored
//! structs. GPS never crosses — the engine reads it from the handle.

use flutter_rust_bridge::frb;

use crate::frb_generated::RustOpaque;
use idl_rs::laps::detect_laps as core_detect_laps;
use idl_rs::laps::model::{Gate, LapTiming, NeutralZone, SectorGate};
pub use idl_rs::laps::{Lap, NeutralZoneVisit, Sector};
pub use idl_rs::session::handle::SessionHandle;

/// FFI gate (four raw-scale coordinates, degrees × 1e7).
pub struct GateArg {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
}

impl GateArg {
    fn into_core(self) -> Gate {
        Gate { lat1: self.lat1, lon1: self.lon1, lat2: self.lat2, lon2: self.lon2 }
    }
}

/// Discriminant for [`LapTimingArg`].
pub enum LapTimingKind {
    Circuit,
    PointToPoint,
}

/// Freezed-free lap-timing config. `Circuit` reads `start` only (`finish`
/// mirrors it); `PointToPoint` reads both.
pub struct LapTimingArg {
    pub kind: LapTimingKind,
    pub start: GateArg,
    pub finish: GateArg,
}

impl LapTimingArg {
    fn into_core(self) -> LapTiming {
        match self.kind {
            LapTimingKind::Circuit => LapTiming::Circuit { start_finish: self.start.into_core() },
            LapTimingKind::PointToPoint => {
                LapTiming::PointToPoint { start: self.start.into_core(), finish: self.finish.into_core() }
            }
        }
    }
}

pub struct SectorGateArg {
    pub name: String,
    pub gate: GateArg,
}

pub struct NeutralZoneArg {
    pub name: String,
    pub enter: GateArg,
    pub exit: GateArg,
}

// ---- Mirrored outputs ----

#[frb(mirror(Sector))]
pub struct _Sector {
    pub name: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub start_time_secs: f64,
    pub end_time_secs: f64,
}

#[frb(mirror(NeutralZoneVisit))]
pub struct _NeutralZoneVisit {
    pub name: String,
    pub enter_ms: i64,
    pub exit_ms: i64,
}

#[frb(mirror(Lap))]
pub struct _Lap {
    pub lap_number: u32,
    pub start_ms: i64,
    pub end_ms: i64,
    pub start_time_secs: f64,
    pub end_time_secs: f64,
    pub raw_elapsed_ms: i64,
    pub lap_time_ms: i64,
    pub sectors: Vec<Sector>,
    pub neutral_zone_visits: Vec<NeutralZoneVisit>,
}

/// Detect laps for the retained session handle. `window_start_ms` +
/// `window_end_ms` form an inclusive visit window when both are `Some`
/// (otherwise the whole session is used).
#[frb]
pub fn detect_laps(
    handle: RustOpaque<SessionHandle>,
    timing: LapTimingArg,
    sector_gates: Vec<SectorGateArg>,
    neutral_zones: Vec<NeutralZoneArg>,
    window_start_ms: Option<i64>,
    window_end_ms: Option<i64>,
) -> Vec<Lap> {
    let timing = timing.into_core();
    let sectors: Vec<SectorGate> = sector_gates
        .into_iter()
        .map(|s| SectorGate { name: s.name, gate: s.gate.into_core() })
        .collect();
    let zones: Vec<NeutralZone> = neutral_zones
        .into_iter()
        .map(|z| NeutralZone { name: z.name, enter: z.enter.into_core(), exit: z.exit.into_core() })
        .collect();
    let window = match (window_start_ms, window_end_ms) {
        (Some(s), Some(e)) => Some((s, e)),
        _ => None,
    };
    core_detect_laps(&handle, &timing, &sectors, &zones, window)
}
