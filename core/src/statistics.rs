//! Time-domain transforms over channel samples — numerical differentiation,
//! rolling RMS / mean / standard deviation, and whole-series detrending.
//!
//! The rolling helpers are ported verbatim from the Dart evaluator
//! (`app/lib/data/math_channel_evaluator.dart` `_differentiate`,
//! `_rollingRms`, `_rollingMean`, `_rollingStd`). Their windows are causal and
//! shortened at the left edge (for `i < w` the window is `0..=i`); standard
//! deviation is the population σ. `detrend` is a *global* (non-causal)
//! least-squares operation over the whole series. Pure: data in, data out.

/// Backward finite-difference derivative.
///
/// `result[0] = 0.0` (no prior sample); `result[i] = (data[i] - data[i-1]) * sample_rate_hz`.
/// Output units = input units × Hz (e.g. m → m/s).
pub fn differentiate(data: &[f64], sample_rate_hz: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 1..n {
        result[i] = (data[i] - data[i - 1]) * sample_rate_hz;
    }
    result
}

/// Rolling RMS over a causal window of `w` samples (shortened at the left edge).
pub fn rolling_rms(data: &[f64], w: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w.saturating_sub(1));
        let mut sum_sq = 0.0;
        for v in &data[start..=i] {
            sum_sq += v * v;
        }
        let count = (i - start + 1) as f64;
        result[i] = (sum_sq / count).sqrt();
    }
    result
}

/// Rolling mean over a causal window of `w` samples (shortened at the left edge).
pub fn rolling_mean(data: &[f64], w: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w.saturating_sub(1));
        let sum: f64 = data[start..=i].iter().sum();
        let count = (i - start + 1) as f64;
        result[i] = sum / count;
    }
    result
}

/// Rolling population standard deviation over a causal window of `w` samples
/// (shortened at the left edge).
pub fn rolling_std(data: &[f64], w: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w.saturating_sub(1));
        let window = &data[start..=i];
        let count = window.len() as f64;
        let mean = window.iter().sum::<f64>() / count;
        let var_sum: f64 = window.iter().map(|v| (v - mean).powi(2)).sum();
        result[i] = (var_sum / count).sqrt();
    }
    result
}

/// Which trend [`detrend`] removes from a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetrendMode {
    /// Leave the series unchanged.
    None,
    /// Subtract the mean — constant-offset removal.
    Constant,
    /// Subtract a least-squares straight-line fit (constant offset + linear
    /// drift). The default, and the only trend order we model.
    Linear,
}

/// Removes a global trend from `data`, fit against the **sample index**
/// `i = 0..N-1` (NOT against `[Time]` — this matches `scipy.signal.detrend`
/// and so carries no time dependency). Output inherits the input's length; the
/// caller carries units / quantity / sample rate over unchanged.
///
/// `Linear` subtracts the least-squares straight line `a + b*i`; `Constant`
/// subtracts the mean; `None` returns the input unchanged. The fit is computed
/// **centered** about `ibar`/`ybar` for numerical stability on long runs — this
/// avoids the catastrophic cancellation of the raw `N*Sxx - (Sx)^2`
/// normal-equation form when `i` reaches the hundreds of thousands.
///
/// NaN-aware: the fit is accumulated over the **finite** samples only and NaN
/// samples are passed through untouched. This INTENTIONALLY DIFFERS from scipy
/// (which propagates a single NaN across the whole output) because telemetry
/// has dropouts — one gap must not blank the entire channel. Do not "fix" this
/// back to scipy semantics.
///
/// Edge cases: empty input → empty; zero finite samples → passthrough; fewer
/// than 2 finite samples with `Linear` → fall back to mean removal (the index
/// variance, hence the linear denominator `sum (i-ibar)^2`, is otherwise never
/// zero, so that is the only case where the slope is undefined).
pub fn detrend(data: &[f64], mode: DetrendMode) -> Vec<f64> {
    let n = data.len();
    if n == 0 || matches!(mode, DetrendMode::None) {
        return data.to_vec();
    }

    // Finite-only accumulation of the index/value sums (NaN dropouts excluded).
    let (mut cnt, mut si, mut sy) = (0usize, 0.0_f64, 0.0_f64);
    for (i, &v) in data.iter().enumerate() {
        if v.is_finite() {
            cnt += 1;
            si += i as f64;
            sy += v;
        }
    }
    if cnt == 0 {
        return data.to_vec();
    }
    let (ibar, ybar) = (si / cnt as f64, sy / cnt as f64);

    // Constant mode, or Linear with too few finite samples to define a slope:
    // remove the mean from finite samples and leave NaN in place.
    if cnt < 2 || matches!(mode, DetrendMode::Constant) {
        return data.iter().map(|&v| if v.is_finite() { v - ybar } else { v }).collect();
    }

    // Centered least-squares slope b = Sxy/Sxx and intercept a = ybar - b*ibar,
    // accumulated over the finite samples only.
    let (mut sxy, mut sxx) = (0.0_f64, 0.0_f64);
    for (i, &v) in data.iter().enumerate() {
        if v.is_finite() {
            let xc = i as f64 - ibar;
            sxy += xc * (v - ybar);
            sxx += xc * xc;
        }
    }
    let b = sxy / sxx;
    let a = ybar - b * ibar;
    data.iter()
        .enumerate()
        .map(|(i, &v)| if v.is_finite() { v - (a + b * i as f64) } else { v })
        .collect()
}

/// Zero-velocity (ZUPT) stationary detector — a per-sample boolean flag marking
/// where the device is (quasi-)stationary, for gating the ZUPT/ZARU/gravity
/// pseudo-measurements. Dual indicator over a causal window of `window` samples:
/// a sample is stationary when the **rolling σ of accel magnitude** is below
/// `accel_std_thresh` (no vibration/motion) AND the **rolling mean of gyro
/// magnitude** is below `gyro_thresh` (not rotating).
///
/// `accel_mag`: per-sample accelerometer magnitude (m/s²). `gyro_mag`: per-sample
/// gyro magnitude (rad/s). Output length is `min(accel_mag.len(), gyro_mag.len())`.
/// Reuses [`rolling_std`] / [`rolling_mean`] (causal, left-edge-shortened windows).
pub fn zupt_flags(
    accel_mag: &[f64],
    gyro_mag: &[f64],
    window: usize,
    accel_std_thresh: f64,
    gyro_thresh: f64,
) -> Vec<bool> {
    let n = accel_mag.len().min(gyro_mag.len());
    let accel_std = rolling_std(&accel_mag[..n], window);
    let gyro_avg = rolling_mean(&gyro_mag[..n], window);
    (0..n)
        .map(|i| accel_std[i] < accel_std_thresh && gyro_avg[i] < gyro_thresh)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn differentiate_linear_ramp_yields_constant_slope() {
        // Arrange — ramp x[i] = i at 10 Hz; derivative = 10.0 (units/s), x[0]=0.
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();

        // Act
        let d = differentiate(&data, 10.0);

        // Assert — d[0]=0 (no prior sample); d[i>=1] = (1) * 10 = 10.
        assert_eq!(d.len(), 5);
        assert_relative_eq!(d[0], 0.0, epsilon = 1e-12);
        for v in &d[1..] {
            assert_relative_eq!(*v, 10.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rolling_mean_of_constant_is_constant() {
        // Arrange
        let data = vec![3.0; 6];

        // Act — window 3
        let m = rolling_mean(&data, 3);

        // Assert — mean of constant is the constant, even at the short left edge.
        for v in &m {
            assert_relative_eq!(*v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rolling_rms_first_sample_equals_abs_value() {
        // Arrange — window shortens to 1 at i=0, so rms[0] = |x[0]|.
        let data = vec![4.0, 0.0, 0.0];

        // Act
        let r = rolling_rms(&data, 3);

        // Assert
        assert_relative_eq!(r[0], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn rolling_std_of_constant_is_zero() {
        // Arrange
        let data = vec![7.0; 5];

        // Act
        let s = rolling_std(&data, 4);

        // Assert — population σ of a constant window is 0.
        for v in &s {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    // ---- detrend (global least-squares trend removal) ----

    #[test]
    fn detrend_linear_drives_pure_ramp_to_zero() {
        // Arrange — a pure ramp is entirely trend.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Act
        let out = detrend(&data, DetrendMode::Linear);

        // Assert — the fit removes the whole signal.
        for v in &out {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn detrend_linear_removes_trend_but_preserves_transient() {
        // Arrange — ramp 2..6 with a +8 spike at index 2 riding on top.
        // The critical property: detrend must NOT attenuate the transient.
        let data = vec![2.0, 3.0, 14.0, 5.0, 6.0];

        // Act
        let out = detrend(&data, DetrendMode::Linear);

        // Assert — line removed, spike survives at full height (8).
        let expected = [-2.0, -2.0, 8.0, -2.0, -2.0];
        for (got, want) in out.iter().zip(expected) {
            assert_relative_eq!(*got, want, epsilon = 1e-12);
        }
    }

    #[test]
    fn detrend_constant_removes_only_the_mean() {
        // Arrange — same data, mean = 6.
        let data = vec![2.0, 3.0, 14.0, 5.0, 6.0];

        // Act
        let out = detrend(&data, DetrendMode::Constant);

        // Assert — y - 6, no slope removed.
        let expected = [-4.0, -3.0, 8.0, -1.0, 0.0];
        for (got, want) in out.iter().zip(expected) {
            assert_relative_eq!(*got, want, epsilon = 1e-12);
        }
    }

    #[test]
    fn detrend_linear_fits_over_finite_samples_and_keeps_nan() {
        // Arrange — a ramp with a dropout at index 2.
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

        // Act
        let out = detrend(&data, DetrendMode::Linear);

        // Assert — fit over the finite samples zeroes them; NaN stays NaN.
        assert_relative_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 0.0, epsilon = 1e-12);
        assert!(out[2].is_nan());
        assert_relative_eq!(out[3], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[4], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn detrend_empty_input_returns_empty() {
        // Arrange / Act
        let out = detrend(&[], DetrendMode::Linear);

        // Assert
        assert!(out.is_empty());
    }

    #[test]
    fn detrend_none_passes_through_unchanged() {
        // Arrange
        let data = vec![2.0, 3.0, 14.0, 5.0, 6.0];

        // Act
        let out = detrend(&data, DetrendMode::None);

        // Assert
        assert_eq!(out, data);
    }

    #[test]
    fn detrend_all_nan_passes_through() {
        // Arrange — zero finite samples: nothing to fit.
        let data = vec![f64::NAN, f64::NAN];

        // Act
        let out = detrend(&data, DetrendMode::Linear);

        // Assert — NaNs preserved (passthrough).
        assert_eq!(out.len(), 2);
        assert!(out[0].is_nan() && out[1].is_nan());
    }

    #[test]
    fn detrend_linear_with_single_finite_sample_falls_back_to_mean() {
        // Arrange — only index 1 is finite; a slope is undefined, so Linear
        // must fall back to constant (mean) removal.
        let data = vec![f64::NAN, 5.0, f64::NAN];

        // Act
        let out = detrend(&data, DetrendMode::Linear);

        // Assert — the lone finite sample loses its mean (→ 0); NaNs stay.
        assert!(out[0].is_nan());
        assert_relative_eq!(out[1], 0.0, epsilon = 1e-12);
        assert!(out[2].is_nan());
    }

    #[test]
    fn zupt_flags_true_when_accel_steady_and_gyro_near_zero() {
        // Arrange — accel magnitude steady at g, gyro magnitude ~0 (stationary).
        let accel_mag = vec![9.81; 6];
        let gyro_mag = vec![0.001, 0.0, 0.002, 0.001, 0.0, 0.001];

        // Act — window 3; thresholds 0.05 m/s² (accel σ), 0.05 rad/s (gyro mean).
        let flags = zupt_flags(&accel_mag, &gyro_mag, 3, 0.05, 0.05);

        // Assert — steady accel + tiny gyro ⇒ every sample flagged stationary.
        assert_eq!(flags.len(), 6);
        assert!(flags.iter().all(|&f| f));
    }

    #[test]
    fn zupt_flags_false_when_gyro_exceeds_threshold() {
        // Arrange — accel steady but the device is rotating.
        let accel_mag = vec![9.81; 5];
        let gyro_mag = vec![1.0; 5];

        // Act
        let flags = zupt_flags(&accel_mag, &gyro_mag, 3, 0.05, 0.05);

        // Assert — rotating ⇒ not stationary anywhere.
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn zupt_flags_false_when_accel_varies() {
        // Arrange — gyro quiet, but accel magnitude swings (vibration / impacts).
        let accel_mag = vec![9.81, 12.0, 7.0, 11.0, 8.0, 10.5];
        let gyro_mag = vec![0.0; 6];

        // Act
        let flags = zupt_flags(&accel_mag, &gyro_mag, 3, 0.05, 0.05);

        // Assert — clearly-moving samples are not flagged stationary.
        assert!(!flags[3]);
        assert!(!flags[5]);
    }
}
