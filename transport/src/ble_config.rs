//! SPEC §7.2 config push/read-back byte framing. This module treats
//! `idl0_config.json` as an opaque byte string throughout — SPEC schema
//! validation is `idl-rs` core's `parse_config`/`ConfigErrorKind` (C3 §3.8),
//! never this crate's job (CLAUDE.md §2 layer rule: transport is I/O only).

use crate::{TransportError, TransportErrorKind};

/// SPEC §7.2: "opening an 8 KB reassembly buffer on the device."
pub const CONFIG_BUFFER_MAX_BYTES: usize = 8 * 1024;

/// SPEC §7.2: Config TX (FF06) "each read returns the next ≤200-byte chunk."
pub const CONFIG_READ_CHUNK_MAX_BYTES: usize = 200;

/// Client-side pre-check before starting a BEGIN/chunk/COMMIT sequence — SPEC
/// §7.2 says the device itself rejects an over-size buffer at COMMIT
/// (`0x80`/`0x81`), so this is a chosen-not-spec'd optimisation: refuse
/// locally rather than spend a full BLE round trip on a doomed push. See
/// Open question 5 — drop this check if a reviewer prefers matching the
/// device's own rejection point exactly.
pub fn validate_config_size(json: &[u8]) -> Result<(), TransportError> {
    if json.len() > CONFIG_BUFFER_MAX_BYTES {
        return Err(TransportError::new(
            TransportErrorKind::Config,
            format!(
                "config is {} bytes, exceeds the device's {}-byte reassembly buffer",
                json.len(),
                CONFIG_BUFFER_MAX_BYTES
            ),
        ));
    }
    Ok(())
}

/// Splits `json` into `chunk_size`-byte pieces (the last shorter) for
/// sequential Write-with-Response calls to Config RX (FF05). `chunk_size`
/// comes from the connection's negotiated MTU minus ATT overhead (Task 5 —
/// see Open question 6 on why the exact value isn't fixed here).
pub fn chunk_config(json: &[u8], chunk_size: usize) -> Vec<&[u8]> {
    if chunk_size == 0 || json.is_empty() {
        return Vec::new();
    }
    json.chunks(chunk_size).collect()
}

/// Reassembles the Config TX (FF06) read loop: calls `read_chunk` repeatedly
/// and concatenates until it returns an empty `Vec` (SPEC §7.2 "until an
/// empty read signals EOF"). `read_chunk` is injected so this logic is
/// testable without a real GATT read.
pub fn reassemble_config_reads(
    mut read_chunk: impl FnMut() -> Result<Vec<u8>, TransportError>,
) -> Result<Vec<u8>, TransportError> {
    let mut out = Vec::new();
    loop {
        let chunk = read_chunk()?;
        if chunk.is_empty() {
            return Ok(out);
        }
        out.extend_from_slice(&chunk);
    }
}

/// SPEC §7.2's push-verification check: "compares it (compact JSON) to what
/// it sent." Byte equality — normalising to compact form (if the two ever
/// differ in whitespace) is core's job upstream of this call, not this
/// crate's (see the layer-boundary note above).
pub fn configs_match(sent: &[u8], read_back: &[u8]) -> bool {
    sent == read_back
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_config_size_under_limit_returns_ok() {
        // Arrange
        let json = vec![0u8; CONFIG_BUFFER_MAX_BYTES];

        // Act
        let result = validate_config_size(&json);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn validate_config_size_over_limit_returns_config_error() {
        // Arrange
        let json = vec![0u8; CONFIG_BUFFER_MAX_BYTES + 1];

        // Act
        let result = validate_config_size(&json);

        // Assert
        let err = result.unwrap_err();
        assert_eq!(err.kind, TransportErrorKind::Config);
    }

    #[test]
    fn chunk_config_ten_bytes_chunk_size_four_yields_three_chunks_last_shorter() {
        // Arrange
        let json = b"0123456789";

        // Act
        let chunks = chunk_config(json, 4);

        // Assert
        assert_eq!(chunks, vec![&b"0123"[..], &b"4567"[..], &b"89"[..]]);
    }

    #[test]
    fn chunk_config_empty_input_yields_no_chunks() {
        // Arrange
        let json: &[u8] = b"";

        // Act
        let chunks = chunk_config(json, 20);

        // Assert
        assert!(chunks.is_empty());
    }

    #[test]
    fn reassemble_config_reads_three_chunks_then_empty_concatenates_in_order() {
        // Arrange
        let mut reads = vec![b"abc".to_vec(), b"def".to_vec(), Vec::new()].into_iter();
        let read_chunk = move || Ok(reads.next().unwrap());

        // Act
        let result = reassemble_config_reads(read_chunk).unwrap();

        // Assert
        assert_eq!(result, b"abcdef");
    }

    #[test]
    fn reassemble_config_reads_immediate_empty_returns_empty_vec() {
        // Arrange
        let read_chunk = || Ok(Vec::new());

        // Act
        let result = reassemble_config_reads(read_chunk).unwrap();

        // Assert
        assert!(result.is_empty());
    }

    #[test]
    fn reassemble_config_reads_propagates_read_error() {
        // Arrange
        let read_chunk = || {
            Err(TransportError::new(TransportErrorKind::Ble, "disconnected"))
        };

        // Act
        let result = reassemble_config_reads(read_chunk);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn configs_match_identical_bytes_true_differing_bytes_false() {
        // Arrange
        let sent = b"{\"a\":1}";
        let same = b"{\"a\":1}";
        let different = b"{\"a\":2}";

        // Act / Assert
        assert!(configs_match(sent, same));
        assert!(!configs_match(sent, different));
    }
}
