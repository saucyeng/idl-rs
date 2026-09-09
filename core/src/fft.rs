//! FFT with windowing — wraps realfft (real-to-complex, ~2× faster than full
//! complex FFT) with Hann, Hamming, and rectangular windows.
//!
//! Returns one-sided magnitude spectrum for real-valued input signals.
//! Equivalent to abs(numpy.fft.rfft(window * data)).
//!
//! See docs/signal_pipeline.md and IDL0_SPEC.md §10.

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

/// Window function applied to the signal before computing the FFT.
///
/// Reduces spectral leakage from signal discontinuities at frame boundaries.
/// Rectangular applies no weighting (equivalent to no window).
pub enum FftWindow {
    /// No weighting — maximum frequency resolution, highest leakage.
    Rectangular,
    /// Hann window — good general-purpose choice, low leakage.
    Hann,
    /// Hamming window — slightly higher sidelobe than Hann, common for speech.
    Hamming,
}

/// Computes the one-sided magnitude spectrum of a real-valued signal.
///
/// Applies `window` to reduce spectral leakage, then computes FFT via realfft.
/// Returns `floor(n/2) + 1` magnitude bins covering 0 Hz to `sample_rate_hz / 2`.
///
/// `data`: input signal samples — any units
/// `window`: window function applied before FFT (see [FftWindow])
/// `sample_rate_hz`: sample rate in Hz — used by caller to convert bin index
///                   to frequency: `freq_hz = bin * sample_rate_hz / data.len()`
///
/// Returns magnitude (not power) in same units as `data`, length `n/2 + 1`.
///
/// realfft: RealFftPlanner::plan_fft_forward() computes only the non-redundant
/// `n/2 + 1` bins of a real signal directly — ~2× faster than a full complex
/// FFT whose negative-frequency half we would discard.
pub fn fft(data: &[f64], window: FftWindow) -> Vec<f64> {
    let n = data.len();
    let weights = window_weights(&window, n);
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n);
    let mut input = r2c.make_input_vec();
    for (i, (&x, &w)) in data.iter().zip(weights.iter()).enumerate() {
        input[i] = x * w;
    }
    let mut spectrum = r2c.make_output_vec(); // length n/2 + 1
    r2c.process(&mut input, &mut spectrum).expect("realfft length invariant");
    spectrum.iter().map(|c| c.norm()).collect()
}

/// Crate-visible alias of [`window_weights`] for `spectrogram()`'s Density
/// normalisation (it needs the same window-power sum welch uses).
pub(crate) fn window_weights_for(window: &FftWindow, n: usize) -> Vec<f64> {
    window_weights(window, n)
}

// Computes per-sample window weights for a signal of length n.
fn window_weights(window: &FftWindow, n: usize) -> Vec<f64> {
    match window {
        FftWindow::Rectangular => vec![1.0; n],
        FftWindow::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()))
            .collect(),
        FftWindow::Hamming => (0..n)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
            })
            .collect(),
    }
}

/// How to remove a trend from each segment before the FFT. Suppresses the DC
/// spike (and, for `Linear`, slow drift) that would otherwise dominate bin 0.
pub enum Detrend {
    /// Leave the segment unchanged — bin 0 reflects the segment mean.
    None,
    /// Subtract the segment mean (constant detrend).
    Mean,
    /// Subtract a least-squares straight-line fit (removes mean + linear drift).
    Linear,
}

/// How per-segment power estimates are combined across segments.
pub enum Averaging {
    /// Arithmetic mean — standard Welch, lowest variance for clean data.
    Mean,
    /// Per-bin median — robust to transient spikes (impacts, chain slap).
    Median,
    /// No cross-segment fold — takes the first segment's own per-bin power
    /// verbatim. With `nperseg: 0` (one full-record segment, [`resolve_seg`]),
    /// this is the only segment there is, so `welch()` reproduces a single
    /// periodogram exactly (ruling R63 (3)).
    None,
    /// Per-bin maximum across segments — surfaces the loudest transient at
    /// each frequency rather than smoothing it away (ruling R63 (3)).
    Max,
}

/// Failures raised validating a [`welch`] request before the transform runs
/// (ruling R76). Neither variant is raised by `welch()` itself — `welch()`
/// has no way to signal failure and stays infallible for its existing
/// callers (`core/src/session/handle.rs`, `core/src/fft_wire.rs`'s tests);
/// callers that accept `averaging`/`sample_rate_hz` from an untrusted wire
/// argument (`tauri/src/commands/rasters.rs::fetch_fft_via`) must call
/// [`check_none_averaging_segments`] and [`effective_rate_hz_from_t_us`]
/// first and reject on `Err` before calling `welch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FftError {
    /// `averaging: Averaging::None` was requested but the segmentation
    /// (`nperseg`/`noverlap` against the record length) produces more than
    /// one segment; `segments` carries the count that would have been
    /// produced. `None` takes only the first segment's power (see
    /// [`Averaging::None`]'s doc comment) — silently accepting more than one
    /// segment would discard the rest of the record.
    NoneRequiresOneSegment { segments: usize },
    /// The channel's effective sample rate could not be derived: fewer than
    /// two samples (no gap to measure), or every consecutive-sample gap
    /// non-positive (duplicate or out-of-order timestamps), which would
    /// otherwise divide by zero in `welch()`'s `Scaling::Density` branch.
    InvalidSampleRate,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FftError::NoneRequiresOneSegment { segments } => {
                write!(f, "averaging \"none\" requires exactly one segment, got {segments}")
            }
            FftError::InvalidSampleRate => {
                write!(f, "could not derive a sample rate: fewer than two samples or duplicate timestamps")
            }
        }
    }
}

impl std::error::Error for FftError {}

/// Number of segments [`stft`]/[`welch`] will produce for a record of length
/// `n` samples, given `nperseg`/`noverlap` — the same clamping and stepping
/// [`stft`]'s own loop uses, computed without running the transform. `n == 0`
/// returns `0` (matching `welch()`'s own empty-input short-circuit).
pub fn segment_count(nperseg: usize, noverlap: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let seg = resolve_seg(nperseg, n);
    let overlap = if noverlap >= seg { seg - 1 } else { noverlap };
    let step = seg - overlap;
    let mut count = 0;
    let mut start = 0;
    while start + seg <= n {
        count += 1;
        start += step;
    }
    count
}

/// Validates `averaging: Averaging::None` against the segmentation that
/// `nperseg`/`noverlap`/`n` would produce (ruling R76): more than one
/// segment is [`FftError::NoneRequiresOneSegment`]; any other `averaging`,
/// or `None` with zero or one segment, is `Ok`. Call before [`welch`] — see
/// [`FftError`]'s doc comment.
pub fn check_none_averaging_segments(
    averaging: &Averaging,
    nperseg: usize,
    noverlap: usize,
    n: usize,
) -> Result<(), FftError> {
    if matches!(averaging, Averaging::None) {
        let segments = segment_count(nperseg, noverlap, n);
        if segments > 1 {
            return Err(FftError::NoneRequiresOneSegment { segments });
        }
    }
    Ok(())
}

/// Derives a channel's effective sample rate in Hz from its recorded `t_us`
/// (microsecond) timestamp axis: `1e6 / median(consecutive-sample gaps in
/// microseconds)` — never a nominal/configured rate (C1 §3.5: metadata only,
/// never used to synthesize time). Lives in `core` rather than the Tauri
/// wrapper that consumes it (ruling R76): it is arithmetic on sample
/// timestamps, and CLAUDE.md §2's decision rule assigns that to `core`.
///
/// Returns [`FftError::InvalidSampleRate`] when there are fewer than two
/// samples (no gap to measure), or when the derived rate is not finite and
/// positive (every gap non-positive — duplicate or out-of-order timestamps).
/// Call before [`welch`] — see [`FftError`]'s doc comment.
pub fn effective_rate_hz_from_t_us(t_us: &[i64]) -> Result<f64, FftError> {
    if t_us.len() < 2 {
        return Err(FftError::InvalidSampleRate);
    }
    let mut gaps: Vec<i64> = t_us.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_unstable();
    let n = gaps.len();
    let median_us = if n % 2 == 1 {
        gaps[n / 2] as f64
    } else {
        0.5 * (gaps[n / 2 - 1] + gaps[n / 2]) as f64
    };
    let rate_hz = 1e6 / median_us;
    if rate_hz.is_finite() && rate_hz > 0.0 {
        Ok(rate_hz)
    } else {
        Err(FftError::InvalidSampleRate)
    }
}

/// Output units of the spectrum.
pub enum Scaling {
    /// RMS magnitude in input units (sqrt of mean power). Single-segment,
    /// rectangular window, no detrend reproduces the legacy `fft()` bin-for-
    /// bin. **Not** one of scipy's two `scaling` values (`"density"` /
    /// `"spectrum"`) — the math language exposes this one only under the
    /// deliberately-different name `"raw_magnitude"` (R151 item 1,
    /// `runs/2026-09-08/scipy-alignment-plan.md` §5 Q1): the migration
    /// pins every existing `fft()` call to it so no workbook's numbers move
    /// under cover of the rename, and no *new* call defaults to it.
    Magnitude,
    /// Power spectral density in input-units squared per Hz — scipy's
    /// `scaling="density"` (both `periodogram` and `welch`'s default):
    /// `|X|² / (fs · Σ(window²))`, one-sided ×2 on interior bins.
    /// Comparable across segment lengths.
    Density,
    /// Power spectrum in input-units squared — scipy's `scaling="spectrum"`:
    /// `|X|² / (Σwindow)²`, one-sided ×2 on interior bins. Comparable
    /// across window functions at a fixed segment length (unlike `Density`,
    /// this is *not* normalised by bandwidth, so it does not compare across
    /// segment lengths).
    Spectrum,
}

/// Frequencies and matching spectral values returned by [welch].
pub struct WelchResult {
    /// Bin centre frequencies in Hz, length `nperseg_used / 2 + 1`.
    pub freqs_hz: Vec<f64>,
    /// Spectral values in units set by [Scaling], same length as `freqs_hz`.
    pub values: Vec<f64>,
}

// Removes a per-segment trend in place. `Mean` subtracts the average; `Linear`
// subtracts a least-squares straight-line fit (mean + drift). Used before
// windowing in stft() to suppress the DC / low-frequency content that would
// otherwise dominate bin 0. Input/output: same units as the segment.
fn detrend_segment(x: &mut [f64], mode: &Detrend) {
    let n = x.len();
    if n == 0 {
        return;
    }
    match mode {
        Detrend::None => {}
        Detrend::Mean => {
            let mean = x.iter().sum::<f64>() / n as f64;
            for v in x.iter_mut() {
                *v -= mean;
            }
        }
        Detrend::Linear => {
            let nn = n as f64;
            let sx: f64 = (0..n).map(|i| i as f64).sum();
            let sy: f64 = x.iter().sum();
            let sxx: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();
            let sxy: f64 = x.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
            let denom = nn * sxx - sx * sx;
            if denom.abs() < f64::EPSILON {
                // Degenerate (n == 1): fall back to mean removal.
                let mean = sy / nn;
                for v in x.iter_mut() {
                    *v -= mean;
                }
                return;
            }
            let slope = (nn * sxy - sx * sy) / denom;
            let intercept = (sy - slope * sx) / nn;
            for (i, v) in x.iter_mut().enumerate() {
                *v -= slope * i as f64 + intercept;
            }
        }
    }
}

// Resolves the effective segment length: `nperseg`, or one full-record
// segment (`n`) when `nperseg` is 0 or at least the record length. Shared by
// stft() and welch() so the window-power normalisation and the transform can
// never disagree on the segment size.
pub(crate) fn resolve_seg(nperseg: usize, n: usize) -> usize {
    if nperseg == 0 || nperseg >= n { n } else { nperseg }
}

// Median of an already-sorted slice. Odd length -> middle element; even length
// -> mean of the two central elements. Used by welch() for per-bin median
// averaging across segments (robust to transient spikes).
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// One STFT frame matrix: the per-segment complex spectra plus their axes.
/// `frames[i]` is segment *i*'s one-sided spectrum (`seg/2 + 1` complex bins).
/// Crate-internal — `welch()` reduces it to an averaged spectrum and
/// `spectrogram()` keeps it as a time×frequency matrix, so the two views come
/// from the same segmentation and stay physically consistent.
pub(crate) struct Stft {
    /// Bin-centre frequencies in Hz, length `seg/2 + 1`.
    pub freqs_hz: Vec<f64>,
    /// Frame-centre times in seconds, relative to `data[0]`, length `n_frames`.
    pub times_secs: Vec<f64>,
    /// `n_frames × (seg/2 + 1)` one-sided complex spectra (kept complex so phase
    /// is available to future transfer-function / coherence / Hilbert work).
    pub frames: Vec<Vec<Complex<f64>>>,
}

/// Short-Time Fourier Transform: split `data` into overlapping segments and
/// take the one-sided spectrum of each. Per segment: detrend → apply `window`
/// weights → real-to-complex forward FFT. Composed on `realfft` (which wraps
/// `rustfft`); `realfft` computes only the non-redundant `seg/2 + 1` bins of a
/// real signal directly — ~2× faster and ~half the memory vs a full complex
/// FFT whose negative-frequency half we would discard. sci-rs exposes no
/// spectral API, so this is the shared primitive `welch()` and `spectrogram()`
/// both build on. Input units pass through unchanged.
///
/// `data`           input signal, any units (one channel's samples)
/// `sample_rate_hz` sample rate; sets the frequency axis and frame-centre times
/// `window`         per-segment weighting (see [FftWindow])
/// `nperseg`        samples per segment; `0` or `>= data.len()` => one full-record segment
/// `noverlap`       overlap in samples; clamped to `[0, nperseg-1]`
/// `detrend`        per-segment trend removal (see [Detrend])
pub(crate) fn stft(
    data: Vec<f64>,
    sample_rate_hz: f64,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
) -> Stft {
    let n = data.len();
    let empty = Stft { freqs_hz: Vec::new(), times_secs: Vec::new(), frames: Vec::new() };
    if n == 0 {
        return empty;
    }
    let seg = resolve_seg(nperseg, n);
    let overlap = if noverlap >= seg { seg - 1 } else { noverlap };
    let step = seg - overlap;
    let weights = window_weights(&window, seg);
    let n_bins = seg / 2 + 1;

    // realfft: a real-to-complex planner produces the one-sided spectrum directly.
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(seg);
    let mut scratch_in = r2c.make_input_vec(); // length seg (real)
    let mut spectrum = r2c.make_output_vec(); // length seg/2 + 1 (complex)

    let mut frames: Vec<Vec<Complex<f64>>> = Vec::new();
    let mut times_secs: Vec<f64> = Vec::new();
    let mut start = 0;
    while start + seg <= n {
        scratch_in.copy_from_slice(&data[start..start + seg]);
        detrend_segment(&mut scratch_in, &detrend);
        for (i, w) in weights.iter().enumerate() {
            scratch_in[i] *= w;
        }
        // process() requires input length == seg; output length == seg/2 + 1.
        r2c.process(&mut scratch_in, &mut spectrum).expect("realfft length invariant");
        frames.push(spectrum.clone());
        times_secs.push((start as f64 + seg as f64 / 2.0) / sample_rate_hz);
        start += step;
    }

    let freqs_hz: Vec<f64> = (0..n_bins).map(|k| k as f64 * sample_rate_hz / seg as f64).collect();
    Stft { freqs_hz, times_secs, frames }
}

/// One-sided averaged spectrum via Welch's method, computed on `realfft` via
/// the shared `stft()` primitive.
///
/// Splits `data` into overlapping segments of `nperseg` samples; per segment:
/// detrend -> apply `window` weights -> forward FFT -> one-sided power `|X|^2`.
/// Power is combined across segments by `averaging`, then converted to output
/// units by `scaling`. Reuses `window_weights` so window definitions match the
/// legacy `fft()` path. sci-rs exposes no spectral API, so this is composed on
/// `realfft` directly — see docs/signal_pipeline.md for the equations.
///
/// `data`           input signal, any units (one channel's samples)
/// `sample_rate_hz` sample rate; sets the frequency axis and density normalisation
/// `window`         per-segment weighting (see [FftWindow])
/// `nperseg`        samples per segment; `0` or `>= data.len()` => one full-record
///                  segment (legacy single-periodogram behaviour)
/// `noverlap`       overlap in samples; clamped to `[0, nperseg-1]`
/// `detrend`        per-segment trend removal (see [Detrend])
/// `averaging`      cross-segment combiner (see [Averaging])
/// `scaling`        output units (see [Scaling])
///
/// Returns `(freqs_hz, values)`, both length `nperseg_used / 2 + 1`.
pub fn welch(
    data: Vec<f64>,
    sample_rate_hz: f64,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    averaging: Averaging,
    scaling: Scaling,
) -> WelchResult {
    let n = data.len();
    if n == 0 {
        return WelchResult { freqs_hz: Vec::new(), values: Vec::new() };
    }
    let seg = resolve_seg(nperseg, n);
    let weights = window_weights(&window, seg);
    let win_power: f64 = weights.iter().map(|w| w * w).sum();
    let n_bins = seg / 2 + 1;

    // Same segmentation as spectrogram() — reduce the complex frames to power.
    let s = stft(data, sample_rate_hz, window, nperseg, noverlap, detrend);
    let seg_powers: Vec<Vec<f64>> =
        s.frames.iter().map(|f| f.iter().map(|c| c.norm_sqr()).collect()).collect();

    // Combine across segments, per bin.
    let n_segs = seg_powers.len();
    let mut avg_power = vec![0.0_f64; n_bins];
    for (k, slot) in avg_power.iter_mut().enumerate() {
        match averaging {
            Averaging::Mean => {
                let sum: f64 = seg_powers.iter().map(|p| p[k]).sum();
                *slot = sum / n_segs as f64;
            }
            Averaging::Median => {
                let mut col: Vec<f64> = seg_powers.iter().map(|p| p[k]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                *slot = median_sorted(&col);
            }
            Averaging::None => {
                *slot = seg_powers[0][k];
            }
            Averaging::Max => {
                *slot = seg_powers.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
            }
        }
    }

    let win_sum: f64 = weights.iter().sum();
    let values: Vec<f64> = match scaling {
        // RMS magnitude: single-segment rect no-detrend gives sqrt(|X|^2) = |X|.
        Scaling::Magnitude => avg_power.iter().map(|p| p.sqrt()).collect(),
        // scipy scaling="density": normalise by fs * sum(w^2).
        Scaling::Density => one_sided_scale(&avg_power, sample_rate_hz * win_power, seg),
        // scipy scaling="spectrum": normalise by sum(w)^2 (no `fs` — not
        // bandwidth-normalised, so it doesn't compare across segment
        // lengths the way `Density` does).
        Scaling::Spectrum => one_sided_scale(&avg_power, win_sum * win_sum, seg),
    };
    WelchResult { freqs_hz: s.freqs_hz, values }
}

/// Normalises `power` by `norm` and doubles every one-sided interior bin
/// (every bin except DC and, for an even segment length, the Nyquist bin) —
/// the one-sided-spectrum step shared by scipy's `"density"` and
/// `"spectrum"` scalings, differing only in what `norm` is.
fn one_sided_scale(power: &[f64], norm: f64, seg: usize) -> Vec<f64> {
    power
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let mut v = p / norm;
            let is_nyquist = seg % 2 == 0 && k == seg / 2;
            if k != 0 && !is_nyquist {
                v *= 2.0;
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stft_segments_and_returns_complex_frames_with_centre_times() {
        // Arrange — 256 samples at 64 Hz, seg 64, 50% overlap (step 32).
        // Frames start at 0,32,64,...,192 → 7 frames; each has 64/2+1 = 33 bins.
        let n = 256usize;
        let fs = 64.0;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin()).collect();

        // Act
        let s = stft(data, fs, FftWindow::Hann, 64, 32, Detrend::None);

        // Assert — shape + the 8 Hz tone lands on bin 8 (8 Hz / (64/64 Hz) = 8) in frame 0.
        assert_eq!(s.frames.len(), 7);
        assert_eq!(s.frames[0].len(), 33);
        assert_eq!(s.freqs_hz.len(), 33);
        assert_eq!(s.times_secs.len(), 7);
        assert_relative_eq!(s.times_secs[0], 32.0 / fs, epsilon = 1e-9); // centre of first 64-sample frame
        let peak = s.frames[0].iter().enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap()).map(|(i, _)| i).unwrap();
        assert_eq!(peak, 8);
    }

    #[test]
    fn fft_sinusoid_at_known_frequency_peaks_at_correct_bin() {
        // Arrange — 128-sample sine at 10 Hz, sample rate 128 Hz
        // Frequency resolution = 128/128 = 1 Hz/bin → 10 Hz lands exactly on bin 10
        let n = 128_usize;
        let sample_rate_hz = 128.0_f64;
        let signal_hz = 10.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * signal_hz * i as f64 / sample_rate_hz).sin())
            .collect();

        // Act
        let spectrum = fft(&data, FftWindow::Rectangular);

        // Assert — bin 10 must be the maximum in the one-sided spectrum
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            peak_bin, 10,
            "expected peak at bin 10 (10 Hz), got bin {peak_bin}",
        );
    }

    #[test]
    fn detrend_mean_removes_constant_offset() {
        // Arrange — constant-offset ramp; mean detrend should zero the average
        let mut x = vec![10.0, 11.0, 12.0, 13.0];

        // Act
        detrend_segment(&mut x, &Detrend::Mean);

        // Assert — mean is now ~0, shape preserved (symmetric about 0)
        let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-9);
        assert_relative_eq!(x[0], -1.5, epsilon = 1e-9);
    }

    #[test]
    fn detrend_linear_removes_straight_line() {
        // Arrange — pure linear ramp; linear detrend drives every sample to 0
        let mut x: Vec<f64> = (0..16).map(|i| 3.0 * i as f64 + 5.0).collect();

        // Act
        detrend_segment(&mut x, &Detrend::Linear);

        // Assert — residual is ~0 everywhere
        for v in &x {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn detrend_none_leaves_signal_unchanged() {
        // Arrange
        let original = vec![1.0, -2.0, 3.0];
        let mut x = original.clone();

        // Act
        detrend_segment(&mut x, &Detrend::None);

        // Assert
        assert_eq!(x, original);
    }

    #[test]
    fn median_sorted_odd_length_returns_middle() {
        // Arrange
        let sorted = vec![1.0, 2.0, 9.0];

        // Act / Assert — middle element, robust to the high outlier
        assert_relative_eq!(median_sorted(&sorted), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn median_sorted_even_length_returns_mean_of_middle_pair() {
        // Arrange
        let sorted = vec![1.0, 2.0, 4.0, 100.0];

        // Act / Assert — mean of the two middle values
        assert_relative_eq!(median_sorted(&sorted), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn welch_single_segment_rect_no_detrend_equals_fft() {
        // Arrange — same sinusoid as the fft() test, fs = 128 Hz, 10 Hz tone
        let n = 128_usize;
        let fs = 128.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let expected = fft(&data, FftWindow::Rectangular);

        // Act — nperseg=0 => single full-record segment; Magnitude scaling
        let result = welch(
            data, fs, FftWindow::Rectangular, 0, 0,
            Detrend::None, Averaging::Mean, Scaling::Magnitude,
        );

        // Assert — welch reproduces fft() bin-for-bin, plus a correct freq axis
        assert_eq!(result.values.len(), expected.len());
        for (got, want) in result.values.iter().zip(expected.iter()) {
            assert_relative_eq!(*got, *want, epsilon = 1e-9);
        }
        assert_relative_eq!(result.freqs_hz[10], 10.0, epsilon = 1e-9);
    }

    #[test]
    fn welch_detrend_mean_suppresses_dc_bin() {
        // Arrange — 10 Hz tone on a large constant offset
        let n = 256_usize;
        let fs = 256.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| 5000.0 + (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        // Act
        let result = welch(
            data, fs, FftWindow::Hann, 0, 0,
            Detrend::Mean, Averaging::Mean, Scaling::Magnitude,
        );

        // Assert — DC bin is near zero, tone near 10 Hz survives
        assert!(result.values[0] < 1.0, "DC bin not suppressed: {}", result.values[0]);
        let peak_bin = result.values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert_relative_eq!(result.freqs_hz[peak_bin], 10.0, epsilon = 1.0);
    }

    #[test]
    fn welch_averaging_reduces_variance_vs_single_periodogram() {
        // Arrange — deterministic mixed-frequency signal (no RNG)
        let n = 4096_usize;
        let fs = 1000.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * std::f64::consts::PI * 137.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 211.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 311.0 * t).sin()
            })
            .collect();

        // Act — single full-length vs 256-sample Welch segments, 50% overlap
        let single = welch(
            data.clone(), fs, FftWindow::Hann, 0, 0,
            Detrend::Mean, Averaging::Mean, Scaling::Density,
        );
        let averaged = welch(
            data, fs, FftWindow::Hann, 256, 128,
            Detrend::Mean, Averaging::Mean, Scaling::Density,
        );

        // Assert — averaged spectrum is smoother: smaller mean abs first-difference
        let roughness = |v: &[f64]| -> f64 {
            let d: f64 = v.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
            d / (v.len() as f64)
        };
        assert!(
            roughness(&averaged.values) < roughness(&single.values),
            "averaged not smoother: {} vs {}",
            roughness(&averaged.values), roughness(&single.values),
        );
    }

    #[test]
    fn welch_median_robust_to_spiked_segment() {
        // Arrange — two clean segments + one spiked, 3 segments no overlap
        let seg = 64_usize;
        let fs = 64.0_f64;
        let mut data: Vec<f64> = Vec::new();
        for s in 0..3 {
            for i in 0..seg {
                let mut v = (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin();
                if s == 1 && i == 0 {
                    v += 1000.0; // transient spike in the middle segment only
                }
                data.push(v);
            }
        }

        // Act
        let mean = welch(
            data.clone(), fs, FftWindow::Rectangular, seg, 0,
            Detrend::None, Averaging::Mean, Scaling::Magnitude,
        );
        let median = welch(
            data, fs, FftWindow::Rectangular, seg, 0,
            Detrend::None, Averaging::Median, Scaling::Magnitude,
        );

        // Assert — the spike inflates the mean DC bin far more than the median
        assert!(
            median.values[0] < mean.values[0],
            "median {} not below mean {}", median.values[0], mean.values[0],
        );
    }

    #[test]
    fn welch_averaging_none_single_segment_matches_averaging_mean_single_segment() {
        // Arrange — 256-sample tone; nperseg: 0 forces one full-record
        // segment for both calls, so None's "take the only segment" and
        // Mean's "average of one value" must produce identical output.
        let n = 256_usize;
        let fs = 256.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        // Act
        let none = welch(
            data.clone(), fs, FftWindow::Hann, 0, 0,
            Detrend::Mean, Averaging::None, Scaling::Density,
        );
        let mean = welch(
            data, fs, FftWindow::Hann, 0, 0,
            Detrend::Mean, Averaging::Mean, Scaling::Density,
        );

        // Assert — parity, bin for bin
        assert_eq!(none.values.len(), mean.values.len());
        for (got, want) in none.values.iter().zip(mean.values.iter()) {
            assert_relative_eq!(*got, *want, epsilon = 1e-12);
        }
    }

    #[test]
    fn welch_max_picks_spiked_segment_over_others() {
        // Arrange — same three-segment layout as the median test: two clean
        // segments plus one spiked in the middle, no overlap.
        let seg = 64_usize;
        let fs = 64.0_f64;
        let mut data: Vec<f64> = Vec::new();
        let mut spiked_segment: Vec<f64> = Vec::new();
        for s in 0..3 {
            for i in 0..seg {
                let mut v = (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin();
                if s == 1 && i == 0 {
                    v += 1000.0; // transient spike in the middle segment only
                }
                data.push(v);
                if s == 1 {
                    spiked_segment.push(v);
                }
            }
        }

        // Act — the spiked segment's own single-periodogram value (Magnitude
        // scaling, one full-record segment: `Averaging::None`'s reduction is
        // "the only segment there is") is the independently-derived expected
        // value for what `Max` should pick at bin 0.
        let spiked_alone = welch(
            spiked_segment, fs, FftWindow::Rectangular, 0, 0,
            Detrend::None, Averaging::None, Scaling::Magnitude,
        );
        let max = welch(
            data.clone(), fs, FftWindow::Rectangular, seg, 0,
            Detrend::None, Averaging::Max, Scaling::Magnitude,
        );
        let mean = welch(
            data, fs, FftWindow::Rectangular, seg, 0,
            Detrend::None, Averaging::Mean, Scaling::Magnitude,
        );

        // Assert — Max diverges from Mean at the spiked bin and matches the
        // spiked segment's own value exactly.
        assert!(
            max.values[0] != mean.values[0],
            "expected Max to diverge from Mean at the spiked bin, both were {}",
            max.values[0],
        );
        assert_relative_eq!(max.values[0], spiked_alone.values[0], epsilon = 1e-9);
    }

    #[test]
    fn welch_clamps_oversized_segment_and_overlap() {
        // Arrange — nperseg and noverlap both larger than the record
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Act — nperseg > n => single segment of length 4 => 3 one-sided bins
        let result = welch(
            data, 4.0, FftWindow::Rectangular, 999, 999,
            Detrend::None, Averaging::Mean, Scaling::Magnitude,
        );

        // Assert — does not panic; length is n/2 + 1
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.freqs_hz.len(), 3);
    }

    #[test]
    fn check_none_averaging_segments_none_with_two_segments_is_error_with_segment_count() {
        // Arrange — 128 samples, nperseg 32, noverlap 16 => 7 segments (see
        // segment_count's own doc comment for the stepping).
        let averaging = Averaging::None;

        // Act
        let result = check_none_averaging_segments(&averaging, 32, 16, 128);

        // Assert
        assert_eq!(result, Err(FftError::NoneRequiresOneSegment { segments: 7 }));
    }

    #[test]
    fn check_none_averaging_segments_none_with_one_segment_is_ok() {
        // Arrange — nperseg 0 => one full-record segment (resolve_seg)
        let averaging = Averaging::None;

        // Act / Assert
        assert_eq!(check_none_averaging_segments(&averaging, 0, 0, 128), Ok(()));
    }

    #[test]
    fn check_none_averaging_segments_mean_with_many_segments_is_ok() {
        // Arrange — same multi-segment layout as the None/error case, but
        // Mean has no single-segment requirement.
        let averaging = Averaging::Mean;

        // Act / Assert
        assert_eq!(check_none_averaging_segments(&averaging, 32, 16, 128), Ok(()));
    }

    #[test]
    fn effective_rate_hz_from_t_us_uniform_spacing_matches_expected_rate() {
        // Arrange — 64 Hz spacing, i.e. 15625 us gaps
        let t_us: Vec<i64> = (0..8).map(|i| i * 15_625).collect();

        // Act
        let hz = effective_rate_hz_from_t_us(&t_us).unwrap();

        // Assert
        assert!((hz - 64.0).abs() < 1e-6, "expected ~64 Hz, got {hz}");
    }

    #[test]
    fn effective_rate_hz_from_t_us_even_sample_count_uses_mean_of_middle_gaps() {
        // Arrange — 4 samples (3 gaps, odd) would leave no ambiguity, so use
        // 5 samples (4 gaps, even): gaps 1000, 1000, 3000, 1000 us sorted as
        // 1000,1000,1000,3000 -> median = mean(1000, 1000) = 1000 us -> 1 kHz.
        let t_us: Vec<i64> = vec![0, 1_000, 2_000, 5_000, 6_000];

        // Act
        let hz = effective_rate_hz_from_t_us(&t_us).unwrap();

        // Assert
        assert!((hz - 1000.0).abs() < 1e-6, "expected ~1000 Hz, got {hz}");
    }

    #[test]
    fn effective_rate_hz_from_t_us_fewer_than_two_samples_is_invalid_sample_rate_error() {
        // Arrange / Act / Assert
        assert_eq!(effective_rate_hz_from_t_us(&[]), Err(FftError::InvalidSampleRate));
        assert_eq!(effective_rate_hz_from_t_us(&[100]), Err(FftError::InvalidSampleRate));
    }

    #[test]
    fn effective_rate_hz_from_t_us_duplicate_timestamps_is_invalid_sample_rate_error() {
        // Arrange — every timestamp identical => every gap is 0 => median 0
        let t_us = vec![500_i64; 6];

        // Act / Assert
        assert_eq!(effective_rate_hz_from_t_us(&t_us), Err(FftError::InvalidSampleRate));
    }

    #[test]
    fn welch_empty_input_returns_empty() {
        // Arrange / Act
        let result = welch(
            Vec::new(), 100.0, FftWindow::Hann, 0, 0,
            Detrend::Mean, Averaging::Mean, Scaling::Magnitude,
        );

        // Assert
        assert!(result.values.is_empty());
        assert!(result.freqs_hz.is_empty());
    }
}
