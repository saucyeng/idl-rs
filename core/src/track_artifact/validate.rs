//! Pre-write validation for a [`Track`] (C3 §3.2 `save_track` ⇒
//! `invalid_argument`). Pure: no clock, no I/O, no randomness — the command
//! layer decides how a failure surfaces over IPC.

use crate::laps::model::{Gate, LapTiming};
use crate::track_artifact::model::Track;

/// Why a `Track` cannot be saved (C3 §3.2 `save_track` ⇒ `invalid_argument`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackValidationErrorKind {
    /// `name` is empty after trimming.
    EmptyName,
    /// A sector gate or neutral zone has an empty name after trimming.
    EmptyChildName,
    /// A coordinate is non-finite, or outside ±90 (lat) / ±180 (lon) degrees.
    CoordinateOutOfRange,
    /// A gate's two endpoints are identical — it can never be crossed.
    DegenerateGate,
}

/// Error from [`validate_track`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackValidationError {
    pub kind: TrackValidationErrorKind,
    /// Which part of the track failed, e.g. `"sector_gates[1].gate"` —
    /// the UI puts the message beside the right field.
    pub field: String,
    pub message: String,
}

fn err(kind: TrackValidationErrorKind, field: impl Into<String>, message: impl Into<String>) -> TrackValidationError {
    TrackValidationError { kind, field: field.into(), message: message.into() }
}

/// True when `deg` is finite and within `[-limit, limit]`.
fn in_range(deg: f64, limit: f64) -> bool {
    deg.is_finite() && deg.abs() <= limit
}

/// Checks one gate's four coordinates are finite and in range, then that its
/// two endpoints are not identical (a zero-length gate can never be
/// crossed). `field` names the gate itself, e.g. `"sector_gates[0].gate"`.
fn validate_gate(gate: &Gate, field: &str) -> Result<(), TrackValidationError> {
    if !in_range(gate.lat1, 90.0) || !in_range(gate.lat2, 90.0) {
        return Err(err(
            TrackValidationErrorKind::CoordinateOutOfRange,
            field,
            format!("{field}: latitude out of range or non-finite"),
        ));
    }
    if !in_range(gate.lon1, 180.0) || !in_range(gate.lon2, 180.0) {
        return Err(err(
            TrackValidationErrorKind::CoordinateOutOfRange,
            field,
            format!("{field}: longitude out of range or non-finite"),
        ));
    }
    if gate.lat1 == gate.lat2 && gate.lon1 == gate.lon2 {
        return Err(err(
            TrackValidationErrorKind::DegenerateGate,
            field,
            format!("{field}: endpoints are identical, the gate can never be crossed"),
        ));
    }
    Ok(())
}

/// Checks one reference-polyline fix's coordinates are finite and in range.
/// `field` names the fix, e.g. `"reference_polyline[0]"`.
fn validate_fix_coords(lat: f64, lon: f64, field: &str) -> Result<(), TrackValidationError> {
    if !in_range(lat, 90.0) || !in_range(lon, 180.0) {
        return Err(err(
            TrackValidationErrorKind::CoordinateOutOfRange,
            field,
            format!("{field}: coordinate out of range or non-finite"),
        ));
    }
    Ok(())
}

/// Validates a draft track before [`write_track`](crate::track_artifact::write::write_track).
/// Decimal degrees throughout (R27) — this runs on the domain type, never
/// the wire DTOs. First failure wins; validation does not collect every
/// error in one pass.
pub fn validate_track(track: &Track) -> Result<(), TrackValidationError> {
    if track.name.trim().is_empty() {
        return Err(err(TrackValidationErrorKind::EmptyName, "name", "name is empty"));
    }

    if let Some(timing) = &track.timing {
        match timing {
            LapTiming::Circuit { start_finish } => {
                validate_gate(start_finish, "lap_timing.start_finish")?;
            }
            LapTiming::PointToPoint { start, finish } => {
                validate_gate(start, "lap_timing.start")?;
                validate_gate(finish, "lap_timing.finish")?;
            }
        }
    }

    for (i, sector) in track.sector_gates.iter().enumerate() {
        if sector.name.trim().is_empty() {
            return Err(err(
                TrackValidationErrorKind::EmptyChildName,
                format!("sector_gates[{i}].name"),
                "sector gate name is empty",
            ));
        }
        validate_gate(&sector.gate, &format!("sector_gates[{i}].gate"))?;
    }

    for (i, zone) in track.neutral_zones.iter().enumerate() {
        if zone.name.trim().is_empty() {
            return Err(err(
                TrackValidationErrorKind::EmptyChildName,
                format!("neutral_zones[{i}].name"),
                "neutral zone name is empty",
            ));
        }
        validate_gate(&zone.enter, &format!("neutral_zones[{i}].enter"))?;
        validate_gate(&zone.exit, &format!("neutral_zones[{i}].exit"))?;
    }

    for (i, fix) in track.reference_polyline.iter().enumerate() {
        validate_fix_coords(fix.lat, fix.lon, &format!("reference_polyline[{i}]"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::GpsFix;
    use crate::laps::model::{NeutralZone, SectorGate};

    fn valid_gate() -> Gate {
        Gate { lat1: 1.0, lon1: 2.0, lat2: 3.0, lon2: 4.0 }
    }

    fn minimal_track() -> Track {
        Track {
            id: "t-1".to_string(),
            name: "A-Line".to_string(),
            venue: "Whistler".to_string(),
            timing: None,
            sector_gates: vec![],
            neutral_zones: vec![],
            reference_polyline: vec![],
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }

    #[test]
    fn validate_track_empty_and_whitespace_only_name_is_empty_name() {
        // Arrange
        let mut track = minimal_track();
        track.name = "   ".to_string();

        // Act
        let result = validate_track(&track);

        // Assert
        assert_eq!(
            result.unwrap_err(),
            TrackValidationError {
                kind: TrackValidationErrorKind::EmptyName,
                field: "name".to_string(),
                message: "name is empty".to_string(),
            }
        );
    }

    #[test]
    fn validate_track_a_sector_gate_named_empty_string_is_empty_child_name_naming_the_index() {
        // Arrange
        let mut track = minimal_track();
        track.sector_gates = vec![SectorGate { name: "".to_string(), gate: valid_gate() }];

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::EmptyChildName);
        assert_eq!(e.field, "sector_gates[0].name");
    }

    #[test]
    fn validate_track_latitude_91_is_coordinate_out_of_range() {
        // Arrange
        let mut track = minimal_track();
        track.sector_gates =
            vec![SectorGate { name: "S1".to_string(), gate: Gate { lat1: 91.0, lon1: 0.0, lat2: 3.0, lon2: 4.0 } }];

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::CoordinateOutOfRange);
        assert_eq!(e.field, "sector_gates[0].gate");
    }

    #[test]
    fn validate_track_longitude_negative_181_is_coordinate_out_of_range() {
        // Arrange
        let mut track = minimal_track();
        track.sector_gates = vec![SectorGate {
            name: "S1".to_string(),
            gate: Gate { lat1: 1.0, lon1: -181.0, lat2: 3.0, lon2: 4.0 },
        }];

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::CoordinateOutOfRange);
        assert_eq!(e.field, "sector_gates[0].gate");
    }

    #[test]
    fn validate_track_nan_coordinate_is_coordinate_out_of_range() {
        // Arrange
        let mut track = minimal_track();
        track.sector_gates = vec![SectorGate {
            name: "S1".to_string(),
            gate: Gate { lat1: f64::NAN, lon1: 0.0, lat2: 3.0, lon2: 4.0 },
        }];

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::CoordinateOutOfRange);
        assert_eq!(e.field, "sector_gates[0].gate");
    }

    #[test]
    fn validate_track_a_circuit_gate_whose_endpoints_are_equal_is_degenerate_gate() {
        // Arrange
        let mut track = minimal_track();
        track.timing = Some(LapTiming::Circuit {
            start_finish: Gate { lat1: 5.0, lon1: 6.0, lat2: 5.0, lon2: 6.0 },
        });

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::DegenerateGate);
        assert_eq!(e.field, "lap_timing.start_finish");
    }

    #[test]
    fn validate_track_a_track_with_no_timing_and_an_empty_polyline_is_ok() {
        // Arrange
        let track = minimal_track();

        // Act
        let result = validate_track(&track);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn validate_track_two_sectors_sharing_a_name_is_ok() {
        // Arrange — locks PLAN Q6 in: duplicate names are display labels, not keys.
        let mut track = minimal_track();
        track.sector_gates = vec![
            SectorGate { name: "S1".to_string(), gate: Gate { lat1: 1.0, lon1: 2.0, lat2: 3.0, lon2: 4.0 } },
            SectorGate { name: "S1".to_string(), gate: Gate { lat1: 5.0, lon1: 6.0, lat2: 7.0, lon2: 8.0 } },
        ];

        // Act
        let result = validate_track(&track);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn validate_track_a_neutral_zone_with_valid_gates_is_ok() {
        // Arrange
        let mut track = minimal_track();
        track.neutral_zones = vec![NeutralZone {
            name: "Pit".to_string(),
            enter: Gate { lat1: 9.0, lon1: 10.0, lat2: 11.0, lon2: 12.0 },
            exit: Gate { lat1: 13.0, lon1: 14.0, lat2: 15.0, lon2: 16.0 },
        }];

        // Act
        let result = validate_track(&track);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn validate_track_a_reference_polyline_fix_out_of_range_is_coordinate_out_of_range() {
        // Arrange — an out-of-range fix would corrupt visit detection, so it
        // is checked even though an empty polyline is legal.
        let mut track = minimal_track();
        track.reference_polyline = vec![GpsFix { timestamp_ms: 0, lat: 91.0, lon: 0.0 }];

        // Act
        let result = validate_track(&track);

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, TrackValidationErrorKind::CoordinateOutOfRange);
        assert_eq!(e.field, "reference_polyline[0]");
    }
}
