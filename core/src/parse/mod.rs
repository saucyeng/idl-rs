//! `.idl0` binary log parser.
//!
//! Decodes schema-3 `IDL0` logs. Records share the
//! `[type:u8][payload_len:u16 LE][payload]` framing; unknown record types are
//! skipped via `payload_len` for forward compatibility.
//!
//! The public entry point [`parse`] validates the magic bytes and schema byte
//! and dispatches to [`v3::parse_v3`], returning a
//! [`ParseResult`](crate::session::ParseResult) holding the parsed session plus
//! an optional truncation warning. The retired v1 (`ESPL`) and v2 (`IDL0`
//! schema 2) formats are no longer supported.

pub mod reader;
pub mod records;
pub mod v3;

#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_buffers;

use crate::session::{ParseError, ParseResult};

/// Validates the magic bytes (and schema byte) and dispatches to the v3 parser.
///
/// - `IDL0` + schema 3 → [`v3::parse_v3`]
///
/// Returns [`ParseError::InvalidMagicBytes`] for any other magic (including the
/// retired `ESPL` v1 format), [`ParseError::UnsupportedSchemaVersion`] for an
/// `IDL0` file whose schema byte is not 3 (including the retired schema-2 v2
/// format), and [`ParseError::TruncatedRecord`] when the buffer is too short to
/// read the magic / schema byte.
pub fn parse(bytes: &[u8]) -> Result<ParseResult, ParseError> {
    if bytes.len() < 4 {
        return Err(ParseError::TruncatedRecord(
            "File too short to read magic bytes (need 4)".to_string(),
        ));
    }
    let magic = String::from_utf8_lossy(&bytes[0..4]).into_owned();
    if magic != "IDL0" {
        return Err(ParseError::InvalidMagicBytes(format!(
            "Not a valid IDL0 log — expected IDL0, got: {magic}"
        )));
    }
    if bytes.len() < 5 {
        return Err(ParseError::TruncatedRecord(
            "File too short to read schema byte (need 5)".to_string(),
        ));
    }
    match bytes[4] {
        3 => v3::parse_v3(bytes),
        schema => Err(ParseError::UnsupportedSchemaVersion(format!(
            "Update the app to open this file (schema v{schema}, only v3 supported)"
        ))),
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::parse;
    use crate::parse::test_buffers::*;
    use crate::session::ParseError;

    #[test]
    fn auto_detects_v3_and_applies_scaling() {
        let accel: f32 = 32.0 / 32768.0;
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }
                .build(&[v3_registry_entry(0, 4, 800, accel, 0.0, "IMU0_AccelX", "g")]),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ]);
        let r = parse(&buf).unwrap();
        let ch = r.session.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert!((ch.materialize()[0] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn espl_magic_now_returns_invalid_magic_bytes() {
        // v1 (`ESPL`) is retired — its magic is rejected, not parsed.
        let mut buf = vec![0u8; 128];
        buf[0..4].copy_from_slice(b"ESPL");
        assert!(matches!(parse(&buf), Err(ParseError::InvalidMagicBytes(_))));
    }

    #[test]
    fn idl0_schema_2_now_returns_unsupported_schema_version() {
        // v2 (`IDL0` schema 2) is retired — schema byte 2 is rejected.
        let buf = cat(&[Header { schema_version: 2, ..Default::default() }.build(&[]), session_end()]);
        assert!(matches!(parse(&buf), Err(ParseError::UnsupportedSchemaVersion(_))));
    }

    #[test]
    fn unknown_magic_returns_invalid_magic_bytes() {
        let buf = vec![0xDE, 0xAD, 0xBE, 0xEF];
        assert!(matches!(parse(&buf), Err(ParseError::InvalidMagicBytes(_))));
    }

    #[test]
    fn schema_4_returns_unsupported_schema_version() {
        let buf = cat(&[Header { schema_version: 4, ..Default::default() }.build(&[]), session_end()]);
        assert!(matches!(parse(&buf), Err(ParseError::UnsupportedSchemaVersion(_))));
    }

    #[test]
    fn too_short_for_magic_returns_truncated() {
        let buf = vec![0x49, 0x44];
        assert!(matches!(parse(&buf), Err(ParseError::TruncatedRecord(_))));
    }
}
