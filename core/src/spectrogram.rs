//! Short-Time Fourier spectrogram — a time×frequency power matrix for the
//! Analyze tab's spectrogram chart. Shares `fft::stft()` with `welch()`, so a
//! spectral peak reads identically in both. Pure: samples in, matrix out.

use crate::fft::{stft, Detrend, FftWindow, Scaling};

/// A time×frequency power/magnitude matrix.
///
/// `power` is flat **row-major** `n_times × n_freqs`: frame `t`, bin `f` is at
/// `power[t * n_freqs + f]`. Units follow `Scaling` exactly as in [`crate::fft::welch`]
/// (Magnitude = `|X|`, the bin's complex modulus (amplitude) in input units;
/// Density = PSD in input-units²/Hz).
/// 2-D arrays do not cross FRB, hence the flat buffer + explicit dims.
pub struct SpectrogramResult {
    /// Bin-centre frequencies in Hz (Y axis), length `n_freqs`.
    pub freqs_hz: Vec<f64>,
    /// Frame-centre times in seconds relative to `data[0]` (X axis), length `n_times`.
    pub times_secs: Vec<f64>,
    /// Flat row-major `n_times × n_freqs` spectral values.
    pub power: Vec<f64>,
    /// Number of time frames (columns of the heatmap's X axis).
    pub n_times: u32,
    /// Number of frequency bins (`nperseg/2 + 1`).
    pub n_freqs: u32,
}

impl SpectrogramResult {
    /// The degenerate result for an empty input or empty window.
    pub fn empty() -> Self {
        SpectrogramResult { freqs_hz: Vec::new(), times_secs: Vec::new(), power: Vec::new(), n_times: 0, n_freqs: 0 }
    }
}

/// Spectrogram via the shared STFT: split into overlapping segments and scale
/// each frame independently (no cross-segment averaging — that omission is what
/// makes this a spectrogram rather than a Welch spectrum). `scaling` selects the
/// per-frame units exactly as [`crate::fft::welch`]: Magnitude is `|X|`, the
/// bin's complex modulus (amplitude) in input units; Density is one-sided PSD
/// (window-power normalised, interior bins ×2). Composed on `fft::stft()`
/// (realfft). Empty result for empty input.
pub fn spectrogram(
    data: Vec<f64>,
    sample_rate_hz: f64,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    scaling: Scaling,
) -> SpectrogramResult {
    if data.is_empty() {
        return SpectrogramResult::empty();
    }
    // Window power for the Density normalisation — recompute from the resolved
    // segment length so it matches stft()'s internal segmentation.
    let n = data.len();
    let seg = crate::fft::resolve_seg(nperseg, n);
    let win_power: f64 = crate::fft::window_weights_for(&window, seg).iter().map(|w| w * w).sum();

    let s = stft(data, sample_rate_hz, window, nperseg, noverlap, detrend);
    let n_freqs = s.freqs_hz.len();
    let n_times = s.frames.len();
    let mut power = Vec::with_capacity(n_times * n_freqs);
    let density_norm = sample_rate_hz * win_power;
    for frame in &s.frames {
        for (k, c) in frame.iter().enumerate() {
            let p = c.norm_sqr();
            let v = match scaling {
                Scaling::Magnitude => p.sqrt(),
                Scaling::Density => {
                    let mut psd = p / density_norm;
                    let is_nyquist = seg % 2 == 0 && k == seg / 2;
                    if k != 0 && !is_nyquist {
                        psd *= 2.0;
                    }
                    psd
                }
            };
            power.push(v);
        }
    }
    SpectrogramResult {
        freqs_hz: s.freqs_hz,
        times_secs: s.times_secs,
        power,
        n_times: n_times as u32,
        n_freqs: n_freqs as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spectrogram_steady_tone_concentrates_energy_in_one_frequency_row() {
        // Arrange — a steady 8 Hz tone at 64 Hz, seg 64, 50% overlap.
        let n = 256usize;
        let fs = 64.0;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin()).collect();

        // Act
        let s = spectrogram(data, fs, FftWindow::Hann, 64, 32, Detrend::None, Scaling::Magnitude);

        // Assert — shape consistent; every frame peaks at the 8 Hz bin (index 8).
        assert_eq!(s.power.len() as u32, s.n_times * s.n_freqs);
        assert_eq!(s.freqs_hz.len() as u32, s.n_freqs);
        assert_eq!(s.times_secs.len() as u32, s.n_times);
        for t in 0..s.n_times as usize {
            let row = &s.power[t * s.n_freqs as usize..(t + 1) * s.n_freqs as usize];
            let peak = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            assert_eq!(peak, 8, "frame {t} should peak at 8 Hz");
        }
        assert_relative_eq!(s.freqs_hz[8], 8.0, epsilon = 1e-9);
    }

    #[test]
    fn spectrogram_density_scaling_tone_peaks_at_its_frequency_bin() {
        // Arrange — a steady 8 Hz tone at 64 Hz, single full-window segment (nperseg=0).
        let n = 64usize;
        let fs = 64.0;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin())
            .collect();

        // Act
        let s = spectrogram(data, fs, FftWindow::Hann, 0, 0, Detrend::None, Scaling::Density);

        // Assert — exactly one frame; all values are finite and non-negative;
        // the 8 Hz bin (index 8) is the peak in that frame.
        assert_eq!(s.n_times, 1, "single full-segment should produce exactly one frame");
        for &v in &s.power {
            assert!(v.is_finite() && v >= 0.0, "density value not finite/non-negative: {v}");
        }
        let peak = s.power.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak, 8, "density frame should peak at 8 Hz bin (index 8), got {peak}");
    }

    #[test]
    fn spectrogram_empty_input_is_empty() {
        let s = spectrogram(Vec::new(), 100.0, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);
        assert_eq!(s.n_times, 0);
        assert_eq!(s.n_freqs, 0);
        assert!(s.power.is_empty());
    }
}
