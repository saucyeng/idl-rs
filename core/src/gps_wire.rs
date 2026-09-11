//! Binary encoder for the `fetch_gps_trace_v2` IPC endpoint (C3 §3.5, ruling
//! R217 item 1): a 16-byte header, then the projected path's `x`, `y` and `t`
//! arrays back to back, then the colour-by channel's `c` array when there is
//! one.
//!
//! The map sibling of `scatter_wire.rs`'s `IDLS` — the same "fixed header plus
//! raw arrays" idiom, a different magic and field set because a trace is three
//! or four equal-length arrays rather than two plus four bounds.
//!
//! **`f64`, not `f32`.** These are metres from an origin that may be a
//! kilometre away; `f32` gives roughly a decimetre of resolution there, which
//! is the same order as the difference between two racing lines through one
//! corner — exactly the comparison a map cell exists to show. The cost is
//! bounded by the point budget, which the caller sets.
//!
//! **`c` is a separate optional block, not a run of `NaN`s.** An uncoloured
//! trace (`gps(null)`) has no colour-by channel at all, and a reader can tell
//! that from the flag without scanning a whole array to discover every value
//! is absent.

const MAGIC: &[u8; 4] = b"IDLG";
const VERSION: u16 = 1;

/// `flags` bit 0: a `c` (colour-by) array follows the `t` array.
pub const FLAG_HAS_C: u16 = 1;

/// Header length, bytes. The `u32` at offset 12 is reserved padding, present
/// so the `f64` block at offset 16 starts 8-byte aligned — a decoder may then
/// take `Float64Array` views directly over the buffer rather than copying
/// value by value.
pub const HEADER_LEN: usize = 16;

/// `IDLG` v1 byte encoder (C3 §3.5).
///
/// Layout, little-endian throughout: `magic` (`"IDLG"`, offset 0), `version`
/// (`u16`, offset 4, always `1`), `flags` (`u16`, offset 6, bit 0 = `has_c`),
/// `point_count` (`u32`, offset 8), 4 reserved zero bytes (offset 12), then
/// `point_count` × `f64` for `x` (metres east of the ENU origin), then the
/// same for `y` (metres north), then for `t` (seconds, session-relative), then
/// for `c` when `c` is `Some`. Total length
/// `16 + point_count * (24 or 32)`.
///
/// `point_count` is the shortest of `xs`, `ys` and `ts` (and of `c` when
/// present), so a caller cannot produce a header that over-states the arrays
/// that follow it.
pub fn encode_gps_trace_idlg(xs: &[f64], ys: &[f64], ts: &[f64], c: Option<&[f64]>) -> Vec<u8> {
    let mut n = xs.len().min(ys.len()).min(ts.len());
    if let Some(cs) = c {
        n = n.min(cs.len());
    }
    let has_c = c.is_some();
    let per_point = if has_c { 32 } else { 24 };

    let mut out = Vec::with_capacity(HEADER_LEN + n * per_point);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(if has_c { FLAG_HAS_C } else { 0u16 }).to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // reserved, 8-byte alignment for the f64 block

    for block in [&xs[..n], &ys[..n], &ts[..n]] {
        for &value in block {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    if let Some(cs) = c {
        for &value in &cs[..n] {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f64(bytes: &[u8], offset: usize) -> f64 {
        f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    fn point_count(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap())
    }

    #[test]
    fn encode_gps_trace_idlg_without_a_colour_channel_writes_three_blocks_and_a_clear_flag() {
        // Arrange
        let xs = vec![0.0_f64, 10.0, 20.0];
        let ys = vec![0.0_f64, -5.0, 5.0];
        let ts = vec![0.0_f64, 0.1, 0.2];

        // Act
        let bytes = encode_gps_trace_idlg(&xs, &ys, &ts, None);

        // Assert
        assert_eq!(&bytes[0..4], b"IDLG");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 0, "has_c must be clear");
        assert_eq!(point_count(&bytes), 3);
        assert_eq!(&bytes[12..16], &[0u8; 4]);
        assert_eq!(bytes.len(), HEADER_LEN + 3 * 24);
        assert_eq!(read_f64(&bytes, HEADER_LEN), 0.0);
        assert_eq!(read_f64(&bytes, HEADER_LEN + 8), 10.0);
        assert_eq!(read_f64(&bytes, HEADER_LEN + 3 * 8 + 8), -5.0, "y block follows x");
        assert_eq!(read_f64(&bytes, HEADER_LEN + 6 * 8 + 16), 0.2, "t block follows y");
    }

    #[test]
    fn encode_gps_trace_idlg_with_a_colour_channel_sets_the_flag_and_appends_a_fourth_block() {
        // Arrange
        let xs = vec![1.0_f64, 2.0];
        let ys = vec![3.0_f64, 4.0];
        let ts = vec![5.0_f64, 6.0];
        let cs = vec![7.0_f64, f64::NAN];

        // Act
        let bytes = encode_gps_trace_idlg(&xs, &ys, &ts, Some(&cs));

        // Assert
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), FLAG_HAS_C);
        assert_eq!(bytes.len(), HEADER_LEN + 2 * 32);
        assert_eq!(read_f64(&bytes, HEADER_LEN + 6 * 8), 7.0);
        assert!(read_f64(&bytes, HEADER_LEN + 7 * 8).is_nan(), "an absent colour sample stays NaN on the wire");
    }

    #[test]
    fn encode_gps_trace_idlg_an_empty_trace_is_header_only() {
        // Arrange / Act
        let bytes = encode_gps_trace_idlg(&[], &[], &[], None);

        // Assert — a session with no GPS fixes is a true answer, not a failure.
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(point_count(&bytes), 0);
    }

    #[test]
    fn encode_gps_trace_idlg_mismatched_lengths_encode_the_shortest_so_the_header_never_overstates() {
        // Arrange
        let xs = vec![1.0_f64, 2.0, 3.0];
        let ys = vec![1.0_f64, 2.0];
        let ts = vec![1.0_f64, 2.0, 3.0, 4.0];

        // Act
        let bytes = encode_gps_trace_idlg(&xs, &ys, &ts, None);

        // Assert
        assert_eq!(point_count(&bytes), 2);
        assert_eq!(bytes.len(), HEADER_LEN + 2 * 24);
    }
}
