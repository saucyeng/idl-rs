//! Read and validate `.idl0wb` workbooks via the shared `config` reader.
//! `parse_workbook` works on in-memory bytes (source-agnostic — roadmap §11
//! seam); `read_workbook` adds filesystem IO.

use std::path::Path;

use crate::config::{self, ConfigError};
use crate::workbook::model::Workbook;

/// Parse a workbook from JSON bytes, rejecting a too-new schema version.
pub fn parse_workbook(bytes: &[u8]) -> Result<Workbook, ConfigError> {
    config::parse_config(bytes)
}

/// Read a workbook from disk, then [`parse_workbook`].
pub fn read_workbook(path: &Path) -> Result<Workbook, ConfigError> {
    config::read_config(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigErrorKind;

    const VALID: &str = r#"{
        "workbook_id": "wb-1",
        "name": "Suspension Tuning",
        "worksheets": [{"ignored": true}],
        "math_channels": [
            {"name": "ForkVelocity", "expression": "differentiate([ForkTravel])", "units": "m/s"},
            {"name": "Power", "expression": "[Force] * [Velocity]"}
        ],
        "created_at_ms": 1,
        "updated_at_ms": 2,
        "workbook_version": 1
    }"#;

    #[test]
    fn parses_valid_workbook_with_math_channels() {
        // Act
        let wb = parse_workbook(VALID.as_bytes()).unwrap();

        // Assert — worksheets ignored; both math channels modeled by name+expr.
        assert_eq!(wb.workbook_id, "wb-1");
        assert_eq!(wb.math_channels.len(), 2);
        assert_eq!(wb.math_channels[0].name, "ForkVelocity");
        assert_eq!(
            wb.math_channels[0].expression,
            "differentiate([ForkTravel])"
        );
    }

    #[test]
    fn too_new_version_is_unsupported_error() {
        // Arrange
        let json = r#"{"workbook_id":"x","name":"x","workbook_version":999}"#;

        // Act
        let err = parse_workbook(json.as_bytes()).unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::UnsupportedVersion);
    }

    #[test]
    fn malformed_json_is_parse_error() {
        // Act
        let err = parse_workbook(b"not json").unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::Parse);
    }

    #[test]
    fn missing_math_channels_defaults_to_empty() {
        // Arrange — no math_channels field.
        let json = r#"{"workbook_id":"x","name":"x","workbook_version":1}"#;

        // Act
        let wb = parse_workbook(json.as_bytes()).unwrap();

        // Assert
        assert!(wb.math_channels.is_empty());
    }
}
