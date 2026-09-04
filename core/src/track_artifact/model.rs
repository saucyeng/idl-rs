//! Portable Track artifact (`.idl0t`) model. Private serde DTOs mirror the Dart
//! `Track.toJson` wire format; they convert once at the read boundary into the
//! public domain [`Track`], and back again (`track_artifact::write`) via the
//! same DTOs — this module is the single authority on the wire shape in
//! both directions.

use serde::{Deserialize, Serialize};

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
    /// Track creation time, milliseconds since the Unix epoch (wire field,
    /// C4 §5's `tracks.created_at_ms`).
    pub created_at_ms: i64,
    /// Track last-update time, milliseconds since the Unix epoch (wire
    /// field, C4 §5's `tracks.updated_at_ms`).
    pub updated_at_ms: i64,
}

impl Track {
    /// Matcher input for [`crate::tracks::detect_visits`] (clones the polyline —
    /// one-shot CLI use; the matcher takes `&[TrackRef]`).
    pub fn track_ref(&self) -> TrackRef {
        TrackRef { track_id: self.id.clone(), polyline: self.reference_polyline.clone() }
    }
}

// ---- private wire DTOs (the `.idl0t` JSON shape == Dart Track.toJson) ----

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
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
    #[serde(default)]
    created_at_ms: i64,
    #[serde(default)]
    updated_at_ms: i64,
}

#[derive(Serialize, Deserialize)]
struct LapGateDto {
    // Despite the `_deg` name, these carry degrees x1e7 (matching `Gate`'s
    // own scale), unchanged from idl0 (SPEC §16.3) — see this module's own
    // Gate/GpsFix conversions, which copy verbatim without rescaling.
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    // Present in the wire format; the engine `Gate` has no name, so it is
    // dropped on read (`into_gate`) and written back as `""` on encode
    // (`From<&Gate>`, below) — the engine never round-trips a gate name.
    #[serde(default)]
    name: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LapTimingDto {
    Circuit {
        #[serde(default)]
        name: String,
        start_finish: LapGateDto,
    },
    PointToPoint {
        start: LapGateDto,
        finish: LapGateDto,
    },
}

#[derive(Serialize, Deserialize)]
struct SectorGateDto {
    name: String,
    gate: LapGateDto,
}

#[derive(Serialize, Deserialize)]
struct NeutralZoneDto {
    #[serde(default)]
    name: String,
    enter: LapGateDto,
    exit: LapGateDto,
}

#[derive(Serialize, Deserialize)]
struct GpsFixDto {
    #[serde(default)]
    timestamp_ms: i64,
    // Despite the `_deg` name, these carry degrees x1e7 (matching
    // `GpsFix`'s own scale), unchanged from idl0 (SPEC §16.3).
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
            created_at_ms: t.created_at_ms,
            updated_at_ms: t.updated_at_ms,
        }
    }
}

// ---- domain → wire conversions (private; used by `track_artifact::write`) ----

impl From<&Gate> for LapGateDto {
    fn from(g: &Gate) -> Self {
        LapGateDto { lat1_deg: g.lat1, lon1_deg: g.lon1, lat2_deg: g.lat2, lon2_deg: g.lon2, name: String::new() }
    }
}
impl From<&LapTiming> for LapTimingDto {
    fn from(t: &LapTiming) -> Self {
        match t {
            LapTiming::Circuit { start_finish } => {
                LapTimingDto::Circuit { name: String::new(), start_finish: start_finish.into() }
            }
            LapTiming::PointToPoint { start, finish } => {
                LapTimingDto::PointToPoint { start: start.into(), finish: finish.into() }
            }
        }
    }
}
impl From<&SectorGate> for SectorGateDto {
    fn from(s: &SectorGate) -> Self {
        SectorGateDto { name: s.name.clone(), gate: (&s.gate).into() }
    }
}
impl From<&NeutralZone> for NeutralZoneDto {
    fn from(z: &NeutralZone) -> Self {
        NeutralZoneDto { name: z.name.clone(), enter: (&z.enter).into(), exit: (&z.exit).into() }
    }
}
impl From<&GpsFix> for GpsFixDto {
    fn from(f: &GpsFix) -> Self {
        GpsFixDto { timestamp_ms: f.timestamp_ms, latitude_deg: f.lat, longitude_deg: f.lon }
    }
}
impl From<&Track> for TrackArtifact {
    fn from(t: &Track) -> Self {
        TrackArtifact {
            track_artifact_version: SUPPORTED_TRACK_ARTIFACT_VERSION,
            track: TrackDto {
                track_id: t.id.clone(),
                name: t.name.clone(),
                venue_name: t.venue.clone(),
                lap_timing: t.timing.as_ref().map(Into::into),
                sector_gates: t.sector_gates.iter().map(Into::into).collect(),
                neutral_zones: t.neutral_zones.iter().map(Into::into).collect(),
                reference_polyline: t.reference_polyline.iter().map(Into::into).collect(),
                created_at_ms: t.created_at_ms,
                updated_at_ms: t.updated_at_ms,
            },
        }
    }
}
