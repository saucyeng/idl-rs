//! Binary encoder for the `fetch_fft` IPC endpoint (C3 §3.6): a 16-byte
//! header, then a flat `f32` magnitude array — the FFT-specific sibling of
//! `raster.rs`'s `IDLR` encoder (same "fixed header + raw array" idiom, a
//! different magic/field set since a spectrum is one 1-D array, not a
//! width×height pixel grid).

const MAGIC: &[u8; 4] = b"IDLF";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 16;

/// `IDLF` v1 byte encoder (C3 §3.6). `values` are cast `f64 -> f32` — a
/// deliberate wire-precision drop per C3, `welch`'s own output stays `f64`.
///
/// Layout, little-endian throughout: `magic` (`"IDLF"`, offset 0), `version`
/// (`u16`, offset 4, always `1`), `reserved` (2 zero bytes, offset 6),
/// `bin_count` (`u32`, offset 8, `values.len()`), `sample_rate_hz` (`f32`,
/// offset 12). Header ends at byte 16 (already 4-byte aligned, no padding
/// needed); `bin_count` × `f32` magnitudes follow from offset 16. Total
/// length `16 + values.len() * 4`. `freqs_hz` is never encoded — the
/// frontend derives bin `k`'s frequency from `sample_rate_hz` and
/// `bin_count` per C3's own text.
pub fn encode_fft_idlf(values: &[f64], sample_rate_hz: f64) -> Vec<u8> {
    let bin_count = values.len() as u32;
    let mut out = Vec::with_capacity(HEADER_LEN + values.len() * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]); // reserved
    out.extend_from_slice(&bin_count.to_le_bytes());
    out.extend_from_slice(&(sample_rate_hz as f32).to_le_bytes());
    for &v in values {
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{welch, Averaging, Detrend, FftWindow, Scaling};
    use approx::assert_relative_eq;

    #[test]
    fn encode_fft_idlf_round_trips_header_and_magnitudes() {
        // Arrange
        let values = vec![1.0_f64, 2.5, -3.25, 4.0];
        let sample_rate_hz = 1000.0_f64;

        // Act
        let bytes = encode_fft_idlf(&values, sample_rate_hz);

        // Assert — manual byte-offset decode matches the input exactly
        assert_eq!(bytes.len(), 16 + values.len() * 4);
        assert_eq!(&bytes[0..4], b"IDLF");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 1);
        assert_eq!(&bytes[6..8], &[0u8, 0u8]);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), values.len() as u32);
        assert_relative_eq!(
            f32::from_le_bytes(bytes[12..16].try_into().unwrap()) as f64,
            sample_rate_hz,
            epsilon = 1e-3,
        );
        for (i, &v) in values.iter().enumerate() {
            let off = 16 + i * 4;
            let got = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            assert_relative_eq!(got as f64, v, epsilon = 1e-6);
        }
    }

    #[test]
    fn encode_fft_idlf_sinusoid_peak_bin_survives_the_encoder() {
        // Arrange — 128-sample 10 Hz tone at 128 Hz, same as fft.rs's own
        // `fft_sinusoid_at_known_frequency_peaks_at_correct_bin` pattern,
        // composed through welch() -> encode_fft_idlf().
        let n = 128_usize;
        let sample_rate_hz = 128.0_f64;
        let signal_hz = 10.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * signal_hz * i as f64 / sample_rate_hz).sin())
            .collect();
        let result = welch(
            data, sample_rate_hz, FftWindow::Rectangular, 0, 0,
            Detrend::None, Averaging::Mean, Scaling::Magnitude,
        );

        // Act
        let bytes = encode_fft_idlf(&result.values, sample_rate_hz);

        // Assert — decode every magnitude, find its peak bin, expect bin 10
        let bin_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut peak_bin = 0usize;
        let mut peak_val = f32::MIN;
        for k in 0..bin_count {
            let off = 16 + k * 4;
            let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            if v > peak_val {
                peak_val = v;
                peak_bin = k;
            }
        }
        assert_eq!(peak_bin, 10, "expected peak at bin 10 (10 Hz), got bin {peak_bin}");
    }
}
