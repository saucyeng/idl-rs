//! Portable Track artifact (`.idl0t`) model. Private serde DTOs mirror the Dart
//! `Track.toJson` wire format; they convert once at the read boundary into the
//! public domain [`Track`], which holds the Phase-4 analysis types directly.

use serde::Deserialize;

use crate::config::VersionedConfig;
use crate::gps::GpsFix;
use crate::laps::model::{Gate, LapTiming, NeutralZone, SectorGate};
use crate::tracks::TrackRef;

/// Highest `.idl0t` schema version this build understands.
pub const SUPPORTED_TRACK_ARTIFACT_VERSION: u32 = 1;

// ---- public domain type ----

/// A loaded portable track: identity + the Phase-4 analysis config.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub venue: String,
    pub timing: Option<LapTiming>,
    pub sector_gates: Vec<SectorGate>,
    pub neutral_zones: Vec<NeutralZone>,
    pub reference_polyline: Vec<GpsFix>,
}

impl Track {
    /// Matcher input for [`crate::tracks::detect_visits`] (clones the polyline —
    /// one-shot CLI use; the matcher takes `&[TrackRef]`).
    pub fn track_ref(&self) -> TrackRef {
        TrackRef { track_id: self.id.clone(), polyline: self.reference_polyline.clone() }
    }
}

// ---- private wire DTOs (the `.idl0t` JSON shape == Dart Track.toJson) ----

#[derive(Deserialize)]
pub(crate) struct TrackArtifact {
    track_artifact_version: u32,
    track: TrackDto,
}

impl VersionedConfig for TrackArtifact {
    const SUPPORTED_VERSION: u32 = SUPPORTED_TRACK_ARTIFACT_VERSION;
    const LABEL: &'static str = "track artifact";
    fn version(&self) -> u32 {
        self.track_artifact_version
    }
}

#[derive(Deserialize)]
struct TrackDto {
    #[serde(default)]
    track_id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    venue_name: String,
    lap_timing: Option<LapTimingDto>,
    #[serde(default)]
    sector_gates: Vec<SectorGateDto>,
    #[serde(default)]
    neutral_zones: Vec<NeutralZoneDto>,
    #[serde(default)]
    reference_polyline: Vec<GpsFixDto>,
}

#[derive(Deserialize)]
struct LapGateDto {
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    // Present in the wire format; the engine `Gate` has no name, so it is dropped.
    #[serde(default)]
    #[allow(dead_code)]
    name: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LapTimingDto {
    Circuit {
        #[serde(default)]
        #[allow(dead_code)]
        name: String,
        start_finish: LapGateDto,
    },
    PointToPoint {
        start: LapGateDto,
        finish: LapGateDto,
    },
}

#[derive(Deserialize)]
struct SectorGateDto {
    name: String,
    gate: LapGateDto,
}

#[derive(Deserialize)]
struct NeutralZoneDto {
    #[serde(default)]
    name: String,
    enter: LapGateDto,
    exit: LapGateDto,
}

#[derive(Deserialize)]
struct GpsFixDto {
    #[serde(default)]
    timestamp_ms: i64,
    latitude_deg: f64,
    longitude_deg: f64,
}

// ---- wire → domain conversions (private) ----

impl LapGateDto {
    fn into_gate(self) -> Gate {
        Gate { lat1: self.lat1_deg, lon1: self.lon1_deg, lat2: self.lat2_deg, lon2: self.lon2_deg }
    }
}
impl LapTimingDto {
    fn into_timing(self) -> LapTiming {
        match self {
            LapTimingDto::Circuit { start_finish, .. } => {
                LapTiming::Circuit { start_finish: start_finish.into_gate() }
            }
            LapTimingDto::PointToPoint { start, finish } => {
                LapTiming::PointToPoint { start: start.into_gate(), finish: finish.into_gate() }
            }
        }
    }
}
impl SectorGateDto {
    fn into_core(self) -> SectorGate {
        SectorGate { name: self.name, gate: self.gate.into_gate() }
    }
}
impl NeutralZoneDto {
    fn into_core(self) -> NeutralZone {
        NeutralZone { name: self.name, enter: self.enter.into_gate(), exit: self.exit.into_gate() }
    }
}
impl GpsFixDto {
    fn into_core(self) -> GpsFix {
        GpsFix { timestamp_ms: self.timestamp_ms, lat: self.latitude_deg, lon: self.longitude_deg }
    }
}

impl From<TrackArtifact> for Track {
    fn from(a: TrackArtifact) -> Self {
        let t = a.track;
        Track {
            id: t.track_id,
            name: t.name,
            venue: t.venue_name,
            timing: t.lap_timing.map(LapTimingDto::into_timing),
            sector_gates: t.sector_gates.into_iter().map(SectorGateDto::into_core).collect(),
            neutral_zones: t.neutral_zones.into_iter().map(NeutralZoneDto::into_core).collect(),
            reference_polyline: t.reference_polyline.into_iter().map(GpsFixDto::into_core).collect(),
        }
    }
}
