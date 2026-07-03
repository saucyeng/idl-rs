//! Read a `.idl0t` track artifact (via the shared `config` reader) into the
//! domain [`Track`].

use std::path::Path;

use crate::config::{self, ConfigError};
use crate::track_artifact::model::{Track, TrackArtifact};

/// Parse a `.idl0t` artifact from JSON bytes into the domain [`Track`].
pub fn parse_track(bytes: &[u8]) -> Result<Track, ConfigError> {
    Ok(config::parse_config::<TrackArtifact>(bytes)?.into())
}

/// Read a `.idl0t` artifact from disk into the domain [`Track`].
pub fn read_track(path: &Path) -> Result<Track, ConfigError> {
    Ok(config::read_config::<TrackArtifact>(path)?.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigErrorKind;
    use crate::laps::model::LapTiming;

    const CIRCUIT: &str = r#"{
        "track_artifact_version": 1,
        "track": {
            "track_id": "t-1", "name": "A-Line", "venue_name": "Whistler",
            "lap_timing": { "kind": "circuit", "name": "S/F",
                "start_finish": {"lat1_deg":501163000,"lon1_deg":-1229574000,"lat2_deg":501163500,"lon2_deg":-1229573000,"name":""} },
            "sector_gates": [ {"name":"S1","gate":{"lat1_deg":1,"lon1_deg":2,"lat2_deg":3,"lon2_deg":4,"name":""}} ],
            "neutral_zones": [],
            "reference_polyline": [ {"timestamp_ms":0,"latitude_deg":501163000,"longitude_deg":-1229574000} ],
            "created_at_ms": 0, "updated_at_ms": 0
        }
    }"#;

    #[test]
    fn parses_circuit_artifact_into_domain_track() {
        // Act
        let t = parse_track(CIRCUIT.as_bytes()).unwrap();

        // Assert — identity + circuit timing + one sector + one reference point.
        assert_eq!(t.id, "t-1");
        assert_eq!(t.name, "A-Line");
        assert!(matches!(t.timing, Some(LapTiming::Circuit { .. })));
        assert_eq!(t.sector_gates.len(), 1);
        assert_eq!(t.sector_gates[0].name, "S1");
        assert_eq!(t.reference_polyline.len(), 1);
        assert_eq!(t.reference_polyline[0].lat, 501163000.0);
        // track_ref carries id + polyline.
        assert_eq!(t.track_ref().track_id, "t-1");
        assert_eq!(t.track_ref().polyline.len(), 1);
    }

    #[test]
    fn parses_point_to_point_timing() {
        // Arrange
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t","name":"n",
            "lap_timing":{"kind":"point_to_point",
                "start":{"lat1_deg":0,"lon1_deg":0,"lat2_deg":1,"lon2_deg":1,"name":""},
                "finish":{"lat1_deg":2,"lon1_deg":2,"lat2_deg":3,"lon2_deg":3,"name":""}},
            "created_at_ms":0,"updated_at_ms":0}}"#;

        // Act
        let t = parse_track(json.as_bytes()).unwrap();

        // Assert
        assert!(matches!(t.timing, Some(LapTiming::PointToPoint { .. })));
    }

    #[test]
    fn missing_lap_timing_is_none() {
        // Arrange — no lap_timing field.
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t","name":"n","created_at_ms":0,"updated_at_ms":0}}"#;

        // Act
        let t = parse_track(json.as_bytes()).unwrap();

        // Assert
        assert!(t.timing.is_none());
        assert!(t.reference_polyline.is_empty());
    }

    #[test]
    fn too_new_version_is_unsupported_error() {
        // Act
        let err = parse_track(
            br#"{"track_artifact_version":999,"track":{"track_id":"t","name":"n","created_at_ms":0,"updated_at_ms":0}}"#,
        )
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::UnsupportedVersion);
    }

    #[test]
    fn malformed_json_is_parse_error() {
        // Act
        let err = parse_track(b"not json").unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::Parse);
    }
}
