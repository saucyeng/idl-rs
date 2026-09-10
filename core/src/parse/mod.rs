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
pub mod registry_preview;
pub mod v3;

#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_buffers;

use crate::session::{ParseError, ParseResult};

/// This module's own version (C1 §4.3), SemVer 2.0.0 — bumped whenever the
/// `.idl0` parser's output or its timing-derivation logic changes. Stamped
/// into every `data.parquet` this crate writes for an `.idl0` source
/// ([`crate::store::parquet::write_session_parquet`]'s `importer_version`
/// argument); a mismatch against the currently-running build's value is one
/// of the two triggers for C1 §4.3's regeneration rule (the other being
/// [`crate::session::seam_correction::SEAM_CORRECTION_VERSION`]).
pub const IDL0_IMPORTER_VERSION: &str = "0.1.0";

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

/// Byte offset of the v3 header's `session start UTC ms` field: magic (4) +
/// schema version (1) + session UUID (16) + device id (6)
/// ([`v3::parse_v3`]'s own read order).
const V3_SESSION_START_MS_OFFSET: usize = 4 + 1 + 16 + 6;

/// Reads the v3 header's `session start UTC ms` (UTC milliseconds since the
/// Unix epoch, `0` = the firmware never had a clock) out of the *first bytes*
/// of an `.idl0` file, without decoding a single record — the header peek C3
/// §3.3's `scan_folder` shows in its preview. `None` when `head` is not an
/// `IDL0` schema-3 file or is shorter than the fixed header prefix; a caller
/// with a real file need only pass the first
/// [`V3_SESSION_START_MS_OFFSET`] + 8 bytes.
///
/// Deliberately *not* the back-filled `effective_start_ms` [`v3::parse_v3`]
/// computes (C1 §3.1 `gps_backfill`): recovering that requires decoding GPS
/// records, which a folder scan must not do.
pub fn peek_session_start_ms(head: &[u8]) -> Option<i64> {
    if head.len() < V3_SESSION_START_MS_OFFSET + 8 {
        return None;
    }
    if &head[0..4] != b"IDL0" || head[4] != 3 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&head[V3_SESSION_START_MS_OFFSET..V3_SESSION_START_MS_OFFSET + 8]);
    Some(i64::from_le_bytes(buf))
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

    #[test]
    fn peek_session_start_ms_a_v3_header_returns_the_headers_own_value() {
        // Arrange
        let buf = cat(&[
            Header { schema_version: 3, session_start_ms: 1_700_000_000_000, ..Default::default() }.build(&[]),
            session_end(),
        ]);

        // Act
        let start = super::peek_session_start_ms(&buf);

        // Assert
        assert_eq!(start, Some(1_700_000_000_000));
    }

    #[test]
    fn peek_session_start_ms_a_header_with_no_clock_returns_zero_not_none() {
        // Arrange — `0` means "the firmware had no clock", which is a peek
        // result, not a failure to peek.
        let buf =
            cat(&[Header { schema_version: 3, session_start_ms: 0, ..Default::default() }.build(&[]), session_end()]);

        // Act
        let start = super::peek_session_start_ms(&buf);

        // Assert
        assert_eq!(start, Some(0));
    }

    #[test]
    fn peek_session_start_ms_a_non_idl0_or_short_buffer_is_none() {
        // Arrange
        let not_idl0 = vec![0xDE; 64];
        let too_short = vec![0x49, 0x44, 0x4C, 0x30, 3, 0, 0];

        // Act / Assert
        assert_eq!(super::peek_session_start_ms(&not_idl0), None);
        assert_eq!(super::peek_session_start_ms(&too_short), None);
    }
}
