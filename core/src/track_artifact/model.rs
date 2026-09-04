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
    // Despite the `_deg` name, these carry degrees x1e7, unchanged from idl0
    // (SPEC §17b.1) — an external file-format contract independent of the
    // engine's internal `Gate` scale. Ruling R27 moved `Gate` to physical
    // decimal degrees, so this module's Gate/GpsFix conversions now rescale
    // explicitly at this wire boundary (they used to copy verbatim, back
    // when both sides were x1e7).
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
    // Despite the `_deg` name, these carry degrees x1e7, unchanged from idl0
    // (SPEC §17b.1) — see `LapGateDto`'s doc comment.
    latitude_deg: f64,
    longitude_deg: f64,
}

// ---- wire → domain conversions (private) ----

impl LapGateDto {
    fn into_gate(self) -> Gate {
        // `/ 1e7`, not `* 1e-7`: 1e7 is exactly representable in binary
        // floating point and division is correctly rounded, so this exactly
        // undoes `From<&Gate>`'s `(deg * 1e7).round()` below for any value
        // that started as a real decimal-degree measurement — multiplying
        // by the inexact constant `1e-7` would not round-trip bit-exact.
        Gate {
            lat1: self.lat1_deg / 1e7,
            lon1: self.lon1_deg / 1e7,
            lat2: self.lat2_deg / 1e7,
            lon2: self.lon2_deg / 1e7,
        }
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
        // See `LapGateDto::into_gate` on why `/ 1e7`, not `* 1e-7`.
        GpsFix { timestamp_ms: self.timestamp_ms, lat: self.latitude_deg / 1e7, lon: self.longitude_deg / 1e7 }
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
        LapGateDto {
            lat1_deg: (g.lat1 * 1e7).round(),
            lon1_deg: (g.lon1 * 1e7).round(),
            lat2_deg: (g.lat2 * 1e7).round(),
            lon2_deg: (g.lon2 * 1e7).round(),
            name: String::new(),
        }
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
        GpsFixDto {
            timestamp_ms: f.timestamp_ms,
            latitude_deg: (f.lat * 1e7).round(),
            longitude_deg: (f.lon * 1e7).round(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of physical decimal-degree values chosen to catch a
    /// regression from `/ 1e7` to `* 1e-7` on the `.idl0t` wire boundary
    /// (`LapGateDto`/`GpsFixDto`'s doc comments explain why the two are not
    /// interchangeable). At least `50.1163`, `-122.9574`, `89.9999999`, and
    /// `-89.9999999` are known — verified independently, not just asserted
    /// here — to land on a different `f64` than they started from if `*
    /// 1e-7` is used instead of `/ 1e7`; the rest (zero, unit values, exact
    /// ±90/±180 boundaries, an arbitrary 7-decimal-digit mid-range value)
    /// round-trip under either operator and are included for coverage, not
    /// as regression bait.
    const CASES: &[f64] = &[
        0.0,
        0.0000001,
        -0.0000001,
        1.0,
        -1.0,
        50.1163,       // fails under `* 1e-7`
        -122.9574,     // fails under `* 1e-7`
        89.9999999,    // fails under `* 1e-7`
        -89.9999999,   // fails under `* 1e-7`
        90.0,
        -90.0,
        179.9999999,
        -179.9999999,
        180.0,
        -180.0,
        45.1234567,
        -45.1234567,
    ];

    #[test]
    fn gate_wire_round_trip_is_bit_exact_across_a_spread_of_coordinates() {
        for &deg in CASES {
            // Arrange — the wire i32 grid point this degree value encodes to.
            let gate = Gate { lat1: deg, lon1: -deg, lat2: deg, lon2: -deg };
            let expected_lat_raw = (deg * 1e7).round() as i32;
            let expected_lon_raw = (-deg * 1e7).round() as i32;

            // Act — write, then read back.
            let wire1 = LapGateDto::from(&gate);
            let (wire1_lat, wire1_lon) = (wire1.lat1_deg, wire1.lon1_deg);
            let gate2 = wire1.into_gate();

            // Assert (1) — the wire value itself is the exact i32 grid point.
            assert_eq!(wire1_lat as i32, expected_lat_raw, "deg={deg}: encode");
            assert_eq!(wire1_lon as i32, expected_lon_raw, "deg={deg}: encode");

            // Assert (2) — the decoded *domain* value is bit-exact against
            // ground truth (`raw / 1e7`, computed independently here, not
            // via `into_gate`). This is the assertion that actually
            // distinguishes `/ 1e7` from `* 1e-7`: a regression to `*
            // 1e-7` decodes to a different `f64` for `deg` values like
            // `50.1163`/`-122.9574`/`89.9999999` above, by up to a few
            // ULPs — an error too small for a *second* `.round()` on
            // re-encoding to ever catch (verified: re-encoding either
            // decoded value recovers the same wire i32 either way, which is
            // why a write→read→write test that only compares the
            // *re-encoded* wire value cannot catch this regression; the
            // domain value itself must be checked).
            assert_eq!(gate2.lat1, wire1_lat / 1e7, "deg={deg}: decode not bit-exact");
            assert_eq!(gate2.lon1, wire1_lon / 1e7, "deg={deg}: decode not bit-exact");
        }
    }

    #[test]
    fn gps_fix_wire_round_trip_is_bit_exact_across_a_spread_of_coordinates() {
        for &deg in CASES {
            // Arrange
            let fix = GpsFix { timestamp_ms: 0, lat: deg, lon: -deg };
            let expected_lat_raw = (deg * 1e7).round() as i32;
            let expected_lon_raw = (-deg * 1e7).round() as i32;

            // Act — write, then read back.
            let wire1 = GpsFixDto::from(&fix);
            let (wire1_lat, wire1_lon) = (wire1.latitude_deg, wire1.longitude_deg);
            let fix2 = wire1.into_core();

            // Assert (1) — wire value is the exact i32 grid point.
            assert_eq!(wire1_lat as i32, expected_lat_raw, "deg={deg}: encode");
            assert_eq!(wire1_lon as i32, expected_lon_raw, "deg={deg}: encode");

            // Assert (2) — decoded domain value is bit-exact against ground
            // truth. See `gate_wire_round_trip_...`'s comment: this is the
            // assertion that actually catches a `/ 1e7` → `* 1e-7`
            // regression; comparing a re-encoded wire value would not.
            assert_eq!(fix2.lat, wire1_lat / 1e7, "deg={deg}: decode not bit-exact");
            assert_eq!(fix2.lon, wire1_lon / 1e7, "deg={deg}: decode not bit-exact");
        }
    }
}
