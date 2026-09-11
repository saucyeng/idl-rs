//! Binary encoder for the `fetch_scatter` IPC endpoint (C3 §3.5, ruling R215
//! item 3): a 48-byte header carrying the cloud's pre-decimation extent, then
//! the decimated `x` array and the `y` array back to back.
//!
//! The scatter sibling of `fft_wire.rs`'s `IDLF` and `tile.rs`'s `IDLT` — same
//! "fixed header + raw arrays" idiom, a different magic and field set because
//! a cloud is two equal-length arrays plus four bounds rather than one array
//! plus a rate.
//!
//! **`f64`, not `f32`.** `IDLF` drops to `f32` on the wire because a spectrum
//! magnitude is a displayed quantity with no downstream arithmetic. A scatter
//! cloud's axes are the channels' own values (g, mm, m/s), and the G-G
//! diagram's whole point is an *equal-aspect* comparison against a reference
//! circle — a wire-precision drop would show up as a visibly non-circular
//! friction circle at small radii. The cost is bounded by the point budget,
//! which the caller sets.
//!
//! **Two separate arrays, not interleaved pairs.** The frontend feeds Plot a
//! record array either way, but separate arrays mean each can be wrapped as
//! one `Float64Array` view with no stride arithmetic, matching how every other
//! array payload in this app crosses into the sandbox.

const MAGIC: &[u8; 4] = b"IDLS";
const VERSION: u16 = 1;

/// Header length, bytes. The `u32` at offset 12 is reserved padding, present
/// so the `f64` bounds block at offset 16 (and therefore both sample arrays)
/// starts 8-byte aligned — a decoder may then take a `Float64Array` view
/// directly over the buffer rather than copying value by value.
pub const HEADER_LEN: usize = 48;

/// `IDLS` v1 byte encoder (C3 §3.5).
///
/// Layout, little-endian throughout: `magic` (`"IDLS"`, offset 0), `version`
/// (`u16`, offset 4, always `1`), `reserved` (2 zero bytes, offset 6),
/// `point_count` (`u32`, offset 8), `reserved` (4 zero bytes, offset 12),
/// then the four `f64` bounds in the order `x_min`, `x_max`, `y_min`, `y_max`
/// (offsets 16, 24, 32, 40). `point_count` × `f64` x values follow from
/// offset 48, then `point_count` × `f64` y values. Total length
/// `48 + point_count * 16`.
///
/// The bounds are the **pre-decimation** extent of the finite cloud over the
/// window ([`crate::scatter::ScatterPoints`]'s own contract), so an
/// equal-aspect caller squares its axes against the true data extent even
/// though the cloud it draws was thinned.
///
/// `xs` and `ys` must be the same length; the shorter is used if they are
/// not, so a caller cannot produce a header that over-states the arrays that
/// follow it.
pub fn encode_scatter_idls(
    xs: &[f64],
    ys: &[f64],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<u8> {
    let n = xs.len().min(ys.len());
    let mut out = Vec::with_capacity(HEADER_LEN + n * 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]); // reserved
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // reserved, 8-byte alignment for the f64 block
    out.extend_from_slice(&x_min.to_le_bytes());
    out.extend_from_slice(&x_max.to_le_bytes());
    out.extend_from_slice(&y_min.to_le_bytes());
    out.extend_from_slice(&y_max.to_le_bytes());
    for &x in &xs[..n] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for &y in &ys[..n] {
        out.extend_from_slice(&y.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_scatter_idls_round_trips_header_bounds_and_both_arrays() {
        // Arrange
        let xs = vec![-1.0_f64, 0.0, 2.5];
        let ys = vec![3.0_f64, -0.5, 1.25];

        // Act
        let bytes = encode_scatter_idls(&xs, &ys, -1.0, 2.5, -0.5, 3.0);

        // Assert — manual byte-offset decode matches the input exactly.
        assert_eq!(bytes.len(), HEADER_LEN + 3 * 16);
        assert_eq!(&bytes[0..4], b"IDLS");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 1);
        assert_eq!(&bytes[6..8], &[0u8, 0u8]);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
        assert_eq!(&bytes[12..16], &[0u8; 4]);
        assert_eq!(f64::from_le_bytes(bytes[16..24].try_into().unwrap()), -1.0);
        assert_eq!(f64::from_le_bytes(bytes[24..32].try_into().unwrap()), 2.5);
        assert_eq!(f64::from_le_bytes(bytes[32..40].try_into().unwrap()), -0.5);
        assert_eq!(f64::from_le_bytes(bytes[40..48].try_into().unwrap()), 3.0);
        for (i, &x) in xs.iter().enumerate() {
            let off = HEADER_LEN + i * 8;
            assert_eq!(f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()), x);
        }
        for (i, &y) in ys.iter().enumerate() {
            let off = HEADER_LEN + xs.len() * 8 + i * 8;
            assert_eq!(f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()), y);
        }
    }

    #[test]
    fn encode_scatter_idls_empty_cloud_is_header_only() {
        // Arrange / Act
        let bytes = encode_scatter_idls(&[], &[], 0.0, 0.0, 0.0, 0.0);

        // Assert
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
    }

    #[test]
    fn encode_scatter_idls_mismatched_lengths_encode_the_shorter_so_the_header_never_overstates() {
        // Arrange
        let xs = vec![1.0_f64, 2.0, 3.0];
        let ys = vec![4.0_f64];

        // Act
        let bytes = encode_scatter_idls(&xs, &ys, 1.0, 3.0, 4.0, 4.0);

        // Assert
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(bytes.len(), HEADER_LEN + 16);
    }

    #[test]
    fn encode_scatter_idls_f64_precision_survives_the_wire() {
        // Arrange — a value f32 could not represent exactly; the G-G
        // diagram's equal-aspect circle depends on this not being rounded.
        let xs = vec![0.100_000_000_000_000_01_f64];

        // Act
        let bytes = encode_scatter_idls(&xs, &[0.0], 0.0, 1.0, 0.0, 1.0);

        // Assert
        let got = f64::from_le_bytes(bytes[HEADER_LEN..HEADER_LEN + 8].try_into().unwrap());
        assert_eq!(got, xs[0]);
    }
}
