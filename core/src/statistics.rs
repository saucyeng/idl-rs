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
///
/// **Not `numpy.gradient`** (R143/R151 item 3): this is a one-sided, causal
/// difference with a zero-filled leading sample — deliberately kept under
/// its own name rather than renamed, because changing the *implementation*
/// to a central difference under cover of a "naming" change would silently
/// move every existing suspension-velocity number a workbook already
/// carries. [`gradient`] is numpy's actual central-difference formula,
/// added alongside as a distinct, honestly-named function.
pub fn differentiate(data: &[f64], sample_rate_hz: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 1..n {
        result[i] = (data[i] - data[i - 1]) * sample_rate_hz;
    }
    result
}

/// Central finite-difference derivative — `numpy.gradient`'s own formula for
/// uniformly-spaced samples, `edge_order=1` (numpy's default): interior
/// samples use the central difference `(data[i+1] - data[i-1]) / (2·dt)`;
/// the two edges fall back to the one-sided forward/backward difference
/// `(data[1]-data[0])/dt` and `(data[n-1]-data[n-2])/dt`. `dt = 1 /
/// sample_rate_hz`. Output units = input units × Hz, same as
/// [`differentiate`]. Fewer than 2 samples has no interior or edge to
/// difference against, so every entry is `0.0` (an empty/single-sample
/// channel has no derivative — matching `differentiate`'s own `result[0] =
/// 0.0` "no prior sample" convention rather than erroring).
pub fn gradient(data: &[f64], sample_rate_hz: f64) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    if n < 2 {
        return result;
    }
    result[0] = (data[1] - data[0]) * sample_rate_hz;
    for i in 1..n - 1 {
        result[i] = (data[i + 1] - data[i - 1]) * 0.5 * sample_rate_hz;
    }
    result[n - 1] = (data[n - 1] - data[n - 2]) * sample_rate_hz;
    result
}

/// Neumaier (improved Kahan) compensated running sum. Adding then subtracting
/// the same term leaves the total accurate to a few ulp of the *result*, not of
/// the largest term that ever passed through it — what a sliding window needs.
#[derive(Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    comp: f64,
}

impl CompensatedSum {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// How many NaN, `+inf` and `-inf` samples a window holds. The running sums
/// only ever see finite samples; a window with any of these is answered from
/// the counts, in O(1), exactly as summing it would have been.
#[derive(Clone, Copy, Default)]
struct NonFinite {
    nan: usize,
    pos_inf: usize,
    neg_inf: usize,
}

impl NonFinite {
    /// The counter `v` belongs to, or `None` for a finite sample.
    fn slot(&mut self, v: f64) -> Option<&mut usize> {
        if v.is_nan() {
            Some(&mut self.nan)
        } else if v == f64::INFINITY {
            Some(&mut self.pos_inf)
        } else if v == f64::NEG_INFINITY {
            Some(&mut self.neg_inf)
        } else {
            None
        }
    }

    /// Whether the window holds any NaN or infinity.
    fn any(&self) -> bool {
        self.nan + self.pos_inf + self.neg_inf > 0
    }
}

/// A window's mean given its non-finite samples (at least one): NaN if it has
/// a NaN or both infinities, else the infinity it has.
fn mean_non_finite(nf: &NonFinite) -> f64 {
    if nf.nan > 0 || (nf.pos_inf > 0 && nf.neg_inf > 0) {
        f64::NAN
    } else if nf.pos_inf > 0 {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

/// A window's RMS given its non-finite samples (at least one): NaN if it has
/// a NaN, else `+inf` (an infinity squares to `+inf`).
fn rms_non_finite(nf: &NonFinite) -> f64 {
    if nf.nan > 0 {
        f64::NAN
    } else {
        f64::INFINITY
    }
}

/// Samples between from-scratch rebuilds of a running window sum (floored at
/// the window width, so the rebuild stays O(1) amortised per sample).
const ROLLING_REBUILD_EVERY: usize = 4096;

/// How far (as a ratio) the window's total magnitude may fall below the
/// largest it has been since the last rebuild before the running sums are
/// rebuilt (2^30).
///
/// A compensated sum is accurate to a few ulp of its *largest* intermediate
/// value, squared: while a huge sample (a corrupt `f32` squared, say) sits in
/// the window, its rounding error is folded into the compensation term and
/// every ordinary sample added beside it is lost. When the huge sample leaves
/// the sum comes back, but the lost samples do not, and the result stays wrong
/// until the next periodic rebuild. The error is about `eps^2 * peak`, so
/// bounding `peak / magnitude` by 2^30 keeps it below `2.4e-7 * eps` of the
/// true magnitude -- far below the 1e-12 relative agreement the rolling
/// windows promise.
const ROLLING_MAX_DECAY: f64 = 1_073_741_824.0;

#[cfg(test)]
thread_local! {
    /// Samples visited by whole-window scans on this thread -- the test seam
    /// behind "the rolling windows are O(N), also around NaN".
    static WINDOW_SCANNED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts `n` samples of a whole-window scan for the test seam.
fn note_scan(n: usize) {
    #[cfg(test)]
    WINDOW_SCANNED.with(|c| c.set(c.get() + n));
    #[cfg(not(test))]
    let _ = n;
}

/// Start index of the causal window ending at `i` for a `w`-sample window
/// (`w = 0` behaves as `w = 1`).
fn window_start(i: usize, w: usize) -> usize {
    i.saturating_sub(w.saturating_sub(1))
}

fn window_mean(window: &[f64]) -> f64 {
    note_scan(window.len());
    window.iter().sum::<f64>() / window.len() as f64
}

fn window_rms(window: &[f64]) -> f64 {
    note_scan(window.len());
    let mut sum_sq = 0.0;
    for v in window {
        sum_sq += v * v;
    }
    (sum_sq / window.len() as f64).sqrt()
}

fn window_std(window: &[f64]) -> f64 {
    note_scan(window.len());
    let count = window.len() as f64;
    let mean = window.iter().sum::<f64>() / count;
    let var_sum: f64 = window.iter().map(|v| (v - mean).powi(2)).sum();
    (var_sum / count).sqrt()
}

/// Shared O(N) driver for [`rolling_mean`] / [`rolling_rms`]: a compensated
/// running sum of `term(v)` over the window's finite samples, finished by
/// `finish(sum, count)`. NaN / inf samples are counted, not summed, and a
/// window holding any is answered by `non_finite` from the counts.
///
/// A window whose magnitude has fallen by more than [`ROLLING_MAX_DECAY`] from
/// its recent peak is rebuilt (a huge sample just left; see the constant).
///
/// A sum that overflows poisons the running state; until a periodic rebuild
/// finds it finite again, windows are computed directly by `direct` (the
/// per-window definition), at most the cost the old code paid on every
/// window.
fn rolling_running_sum(
    data: &[f64],
    w: usize,
    term: fn(f64) -> f64,
    finish: fn(f64, f64) -> f64,
    direct: fn(&[f64]) -> f64,
    non_finite: fn(&NonFinite) -> f64,
) -> Vec<f64> {
    let n = data.len();
    let width = w.max(1);
    let rebuild_every = ROLLING_REBUILD_EVERY.max(width);
    // The window's compensated sum and its total magnitude, from scratch.
    let rebuild = |window: &[f64]| {
        note_scan(window.len());
        let mut acc = CompensatedSum::default();
        let mut magnitude = 0.0;
        for &v in window {
            if v.is_finite() {
                let t = term(v);
                acc.add(t);
                magnitude += t.abs();
            }
        }
        (acc, magnitude)
    };
    let mut result = vec![0.0; n];
    let mut acc = CompensatedSum::default();
    // Plain sum of |term| over the window's finite samples, and the largest
    // it has been since the last rebuild: the dynamic-range guard's inputs.
    let (mut magnitude, mut peak) = (0.0_f64, 0.0_f64);
    let mut counts = NonFinite::default();
    let mut since_rebuild = 0usize;
    let mut poisoned = false;
    for i in 0..n {
        match counts.slot(data[i]) {
            Some(c) => *c += 1,
            None => {
                let t = term(data[i]);
                acc.add(t);
                magnitude += t.abs();
            }
        }
        peak = peak.max(magnitude);
        if i >= width {
            let old = data[i - width];
            match counts.slot(old) {
                Some(c) => *c -= 1,
                None => {
                    let t = term(old);
                    acc.add(-t);
                    magnitude -= t.abs();
                }
            }
        }
        if !poisoned && peak > ROLLING_MAX_DECAY * magnitude {
            (acc, magnitude) = rebuild(&data[window_start(i, w)..=i]);
            peak = magnitude;
            since_rebuild = 0;
        }
        if counts.any() {
            result[i] = non_finite(&counts);
            continue;
        }
        let window = &data[window_start(i, w)..=i];
        since_rebuild += 1;
        if poisoned {
            result[i] = direct(window);
            if since_rebuild >= width {
                (acc, magnitude) = rebuild(window);
                peak = magnitude;
                poisoned = !acc.value().is_finite();
                since_rebuild = 0;
            }
            continue;
        }
        if since_rebuild >= rebuild_every || !acc.value().is_finite() {
            (acc, magnitude) = rebuild(window);
            peak = magnitude;
            since_rebuild = 0;
            if !acc.value().is_finite() {
                poisoned = true;
                result[i] = direct(window);
                continue;
            }
        }
        result[i] = finish(acc.value(), window.len() as f64);
    }
    result
}

/// Rolling RMS over a causal window of `w` samples (shortened at the left edge).
///
/// O(N): a compensated running sum of squares (rebuilt from scratch every few
/// thousand samples). NaN / inf samples are counted, so a window holding one
/// costs no more than any other. Equal to the per-window definition to a few
/// ulp, not bit for bit.
pub fn rolling_rms(data: &[f64], w: usize) -> Vec<f64> {
    rolling_running_sum(data, w, |v| v * v, |sum, count| (sum.max(0.0) / count).sqrt(), window_rms, rms_non_finite)
}

/// Rolling mean over a causal window of `w` samples (shortened at the left edge).
///
/// O(N): a compensated running sum (rebuilt from scratch every few thousand
/// samples). NaN / inf samples are counted, so a window holding one costs no
/// more than any other. Equal to the per-window definition to a few ulp, not
/// bit for bit.
pub fn rolling_mean(data: &[f64], w: usize) -> Vec<f64> {
    rolling_running_sum(data, w, |v| v, |sum, count| sum / count, window_mean, mean_non_finite)
}

/// Mean and population variance of a window from its sums about the shift:
/// `(s1 / count, s2 / count - (s1 / count)²)`.
fn shifted_moments(s1: &CompensatedSum, s2: &CompensatedSum, count: f64) -> (f64, f64) {
    let m = s1.value() / count;
    (m, s2.value() / count - m * m)
}

/// Rolling population standard deviation over a causal window of `w` samples
/// (shortened at the left edge).
///
/// O(N): compensated running sums of `v − K` and `(v − K)²` about a shift `K`
/// near the window mean, so large offsets (1e6 ± noise) do not cancel. `K` is
/// re-centred (an O(w) rebuild) whenever the window mean drifts more than 8σ
/// from it, and every few thousand samples. Never negative or NaN from
/// cancellation (variance is clamped at zero). A window holding NaN or inf is
/// NaN, from counts; one whose sums overflow is computed directly until a
/// re-centre finds them finite. Equal to the per-window definition to a few
/// ulp, not bit for bit.
pub fn rolling_std(data: &[f64], w: usize) -> Vec<f64> {
    let n = data.len();
    let width = w.max(1);
    let rebuild_every = ROLLING_REBUILD_EVERY.max(width);
    // Re-centres on the mean of the window's finite samples and recomputes
    // both sums, and `s2`'s plain magnitude, from scratch.
    let recentre = |window: &[f64], old_k: f64| -> (f64, CompensatedSum, CompensatedSum, f64) {
        note_scan(window.len());
        let (sum, finite) = window.iter().filter(|v| v.is_finite()).fold((0.0_f64, 0usize), |(s, c), v| (s + *v, c + 1));
        let k = if finite > 0 { sum / finite as f64 } else { old_k };
        let (mut s1, mut s2) = (CompensatedSum::default(), CompensatedSum::default());
        let mut magnitude = 0.0;
        for &v in window.iter().filter(|v| v.is_finite()) {
            let d = v - k;
            s1.add(d);
            s2.add(d * d);
            magnitude += d * d;
        }
        (k, s1, s2, magnitude)
    };
    let mut result = vec![0.0; n];
    let mut k = match data.first() {
        Some(v) if v.is_finite() => *v,
        _ => 0.0,
    };
    let (mut s1, mut s2) = (CompensatedSum::default(), CompensatedSum::default());
    // Plain sum of the window's (v - K)^2, and the largest it has been since
    // the last re-centre: the dynamic-range guard's inputs (s2 is >= 0, so it
    // is its own magnitude).
    let (mut magnitude, mut peak) = (0.0_f64, 0.0_f64);
    let mut counts = NonFinite::default();
    let mut since_rebuild = 0usize;
    let mut poisoned = false;
    for i in 0..n {
        match counts.slot(data[i]) {
            Some(c) => *c += 1,
            None => {
                let d = data[i] - k;
                s1.add(d);
                s2.add(d * d);
                magnitude += d * d;
            }
        }
        peak = peak.max(magnitude);
        if i >= width {
            let old = data[i - width];
            match counts.slot(old) {
                Some(c) => *c -= 1,
                None => {
                    let d = old - k;
                    s1.add(-d);
                    s2.add(-(d * d));
                    magnitude -= d * d;
                }
            }
        }
        if !poisoned && peak > ROLLING_MAX_DECAY * magnitude {
            (k, s1, s2, magnitude) = recentre(&data[window_start(i, w)..=i], k);
            peak = magnitude;
            since_rebuild = 0;
        }
        if counts.any() {
            result[i] = f64::NAN;
            continue;
        }
        let window = &data[window_start(i, w)..=i];
        let count = window.len() as f64;
        since_rebuild += 1;
        if poisoned {
            result[i] = window_std(window);
            if since_rebuild >= width {
                (k, s1, s2, magnitude) = recentre(window, k);
                peak = magnitude;
                poisoned = !shifted_moments(&s1, &s2, count).1.is_finite();
                since_rebuild = 0;
            }
            continue;
        }
        let (m, mut var) = shifted_moments(&s1, &s2, count);
        let stale = m * m > 64.0 * var.max(0.0) && m * m > 1e-20 * k * k + f64::MIN_POSITIVE;
        if since_rebuild >= rebuild_every || stale || !var.is_finite() {
            (k, s1, s2, magnitude) = recentre(window, k);
            peak = magnitude;
            since_rebuild = 0;
            (_, var) = shifted_moments(&s1, &s2, count);
            if !var.is_finite() {
                poisoned = true;
                result[i] = window_std(window);
                continue;
            }
        }
        result[i] = var.max(0.0).sqrt();
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
    fn gradient_linear_ramp_yields_constant_slope_including_at_the_edges() {
        // Arrange — same ramp as `differentiate_linear_ramp_...`, but a
        // central difference gets the slope right at the edges too (no
        // `differentiate`-style zero-filled first sample).
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();

        // Act
        let g = gradient(&data, 10.0);

        // Assert
        assert_eq!(g.len(), 5);
        for v in &g {
            assert_relative_eq!(*v, 10.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn gradient_matches_numpys_own_worked_example() {
        // Arrange — numpy.gradient([1, 2, 4, 7, 11, 16]) (unit spacing) is
        // documented as [1. , 1.5, 2.5, 3.5, 4.5, 5. ]; `sample_rate_hz: 1.0`
        // makes `dt = 1` so this reproduces that example exactly.
        let data = vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0];

        // Act
        let g = gradient(&data, 1.0);

        // Assert
        let expected = [1.0, 1.5, 2.5, 3.5, 4.5, 5.0];
        for (got, want) in g.iter().zip(expected.iter()) {
            assert_relative_eq!(*got, *want, epsilon = 1e-12);
        }
    }

    #[test]
    fn gradient_differs_from_differentiate_at_interior_samples() {
        // Arrange — the whole reason `gradient` is a separate function
        // (R151 item 3): a central difference is not the same computation
        // as the causal backward difference, even on the same input.
        let data = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // x^2 samples
        let sample_rate_hz = 1.0;

        // Act
        let backward = differentiate(&data, sample_rate_hz);
        let central = gradient(&data, sample_rate_hz);

        // Assert — differ at every interior index (2nd derivative is
        // nonzero, so backward and central differences disagree).
        assert_ne!(backward[2], central[2]);
    }

    #[test]
    fn gradient_of_fewer_than_two_samples_is_all_zero_not_a_panic() {
        // Arrange / Act / Assert
        assert_eq!(gradient(&[], 10.0), Vec::<f64>::new());
        assert_eq!(gradient(&[5.0], 10.0), vec![0.0]);
    }

    #[test]
    fn gradient_of_exactly_two_samples_uses_the_one_sided_edge_formula_for_both() {
        // Arrange — n=2 has no interior sample; both entries are the same
        // one-sided (forward = backward here) difference.
        let data = vec![3.0, 5.0];

        // Act
        let g = gradient(&data, 2.0);

        // Assert — (5-3)*2 = 4 for both.
        assert_eq!(g, vec![4.0, 4.0]);
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

    // ---- rolling windows: running-sum versions vs. the per-window definition ----

    /// The pre-O(N) implementations, verbatim: the reference every running-sum
    /// version must agree with.
    fn reference_rolling(data: &[f64], w: usize, f: fn(&[f64]) -> f64) -> Vec<f64> {
        (0..data.len()).map(|i| f(&data[i.saturating_sub(w.saturating_sub(1))..=i])).collect()
    }

    /// Deterministic noise in [-1, 1) (xorshift64), so the tests need no RNG crate.
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    fn assert_agrees(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, r)) in got.iter().zip(want).enumerate() {
            if r.is_nan() {
                assert!(g.is_nan(), "{what}[{i}]: want NaN, got {g}");
            } else if r.is_infinite() {
                assert_eq!(g, r, "{what}[{i}]");
            } else {
                assert!((g - r).abs() <= 1e-12 * r.abs().max(1.0), "{what}[{i}]: got {g}, want {r}");
            }
        }
    }

    fn assert_all_three_agree(data: &[f64], w: usize, what: &str) {
        assert_agrees(&rolling_mean(data, w), &reference_rolling(data, w, |s| s.iter().sum::<f64>() / s.len() as f64), &format!("{what} mean w={w}"));
        assert_agrees(&rolling_rms(data, w), &reference_rolling(data, w, |s| (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt()), &format!("{what} rms w={w}"));
        assert_agrees(&rolling_std(data, w), &reference_rolling(data, w, window_std), &format!("{what} std w={w}"));
    }

    #[test]
    fn rolling_running_sums_long_noise_agree_with_the_per_window_definition() {
        // Arrange
        let data = noise(50_000, 0x9E3779B97F4A7C15);

        // Act / Assert — widths: 1, short, medium, longer than the rebuild stride.
        for w in [0, 1, 7, 300, 2000] {
            assert_all_three_agree(&data, w, "noise");
        }
    }

    #[test]
    fn rolling_running_sums_large_offset_agree_with_the_per_window_definition() {
        // Arrange — 1e6 + noise: a naive sum-of-squares std cancels to garbage here.
        let data: Vec<f64> = noise(50_000, 12345).iter().map(|v| 1e6 + v).collect();

        // Act / Assert
        for w in [7, 300, 2000] {
            assert_all_three_agree(&data, w, "offset");
        }
    }

    #[test]
    fn rolling_running_sums_spikes_and_drift_agree_with_the_per_window_definition() {
        // Arrange — a ramp (window mean drifts far from any fixed shift) with
        // isolated 1e6 spikes that later leave the window.
        let mut data: Vec<f64> = noise(40_000, 777).iter().enumerate().map(|(i, v)| v + i as f64 * 0.5).collect();
        for i in (1000..40_000).step_by(2503) {
            data[i] += 1e6;
        }

        // Act / Assert
        for w in [5, 300, 2000] {
            assert_all_three_agree(&data, w, "spikes");
        }
    }

    #[test]
    fn rolling_running_sums_nan_windows_are_nan_and_recover_exactly_when_the_nan_leaves() {
        // Arrange — NaN and inf dropouts, one right at the start.
        let mut data = noise(20_000, 4242);
        for i in [0, 1, 500, 501, 9_999, 15_000] {
            data[i] = f64::NAN;
        }
        data[7_000] = f64::INFINITY;
        data[12_000] = f64::NEG_INFINITY;

        // Act / Assert
        for w in [1, 7, 300] {
            assert_all_three_agree(&data, w, "nan");
        }
    }

    #[test]
    fn rolling_std_constant_large_offset_is_zero_to_rounding_never_negative_or_nan() {
        // Arrange
        let data = vec![1.0e6 + 0.1; 10_000];

        // Act
        let s = rolling_std(&data, 50);

        // Assert
        assert!(s.iter().all(|v| *v >= 0.0 && v.is_finite() && *v < 1e-6));
    }

    #[test]
    fn rolling_running_sums_near_zero_mean_of_huge_terms_stays_absolutely_accurate() {
        // Arrange -- alternating +-1e6 around a 1e-3 offset: every full window of
        // even width has mean exactly 1e-3 and std exactly 1e6, though each
        // partial sum is ~1e6. A plain running sum loses ~1e-10 here.
        let data: Vec<f64> = (0..20_000).map(|i| if i % 2 == 0 { 1e6 } else { -1e6 } + 1e-3).collect();
        let w = 100;
        // 1e6 + 1e-3 and -1e6 + 1e-3 each round at ulp(1e6), so the stored
        // pair's mean is not 1e-3; the pair sums exactly, so this is the true
        // mean of every full window.
        let true_mean = (data[0] + data[1]) / 2.0;

        // Act
        let mean = rolling_mean(&data, w);
        let std = rolling_std(&data, w);

        // Assert
        for i in w - 1..data.len() {
            assert!((mean[i] - true_mean).abs() <= 1e-12, "mean[{i}] = {}", mean[i]);
            assert!((std[i] - 1e6).abs() <= 1e-12 * 1e6, "std[{i}] = {}", std[i]);
        }
    }

    #[test]
    fn rolling_running_sums_a_nan_in_every_window_is_still_linear_and_equals_the_definition() {
        // Arrange -- a NaN dropout every 997 samples with a 2000-sample window
        // puts a NaN in every window; plus an infinity pair to cover the signs.
        let mut data = noise(10_000, 99);
        for i in (0..data.len()).step_by(997) {
            data[i] = f64::NAN;
        }
        data[3_001] = f64::INFINITY;
        data[3_002] = f64::NEG_INFINITY;
        data[6_345] = f64::INFINITY;
        let w = 2_000;
        let clean = noise(10_000, 99);

        // Act
        let scanned_before = WINDOW_SCANNED.with(|c| c.get());
        let _ = (rolling_mean(&data, w), rolling_rms(&data, w), rolling_std(&data, w));
        let scanned = WINDOW_SCANNED.with(|c| c.get()) - scanned_before;

        // Assert -- whole-window scans (rebuilds) touch O(N) samples in all,
        // not O(N * w); and the answers are the definition's.
        assert!(scanned <= 3 * 4 * data.len(), "scanned {scanned} samples");
        assert_all_three_agree(&data, w, "nan every window");
        assert_all_three_agree(&clean, w, "clean");
    }

    #[test]
    fn rolling_running_sums_overflowing_sums_match_the_definition_and_recover() {
        // Arrange -- finite values whose squares and sums overflow, in the middle of a calm signal.
        let mut data = noise(12_000, 5);
        for v in &mut data[4_000..4_010] {
            *v = 1e200;
        }
        data[7_000] = -1e308;
        data[7_001] = -1e308;
        data[7_002] = -1e308;

        // Act / Assert
        for w in [3, 50, 600] {
            assert_all_three_agree(&data, w, "overflow");
        }
    }

    /// Scans made by one `rolling_mean` + `rolling_rms` + `rolling_std` run.
    fn scans_of_all_three(data: &[f64], w: usize) -> usize {
        let before = WINDOW_SCANNED.with(|c| c.get());
        let _ = (rolling_mean(data, w), rolling_rms(data, w), rolling_std(data, w));
        WINDOW_SCANNED.with(|c| c.get()) - before
    }

    #[test]
    fn rolling_running_sums_the_window_after_a_huge_sample_leaves_equals_the_definition() {
        // Arrange -- one corrupt f32-range sample (3e38; 9e76 squared) in calm data.
        let mut data = noise(3_000, 31);
        data[1_000] = 3.0e38;

        // Act / Assert -- including the first window the spike is absent from.
        for w in [2, 50, 700] {
            assert_all_three_agree(&data, w, "spike left");
        }
    }

    #[test]
    fn rolling_running_sums_a_nan_in_the_window_when_a_huge_sample_leaves_still_recovers() {
        // Arrange -- a NaN enters 10 samples after the spike and outlives it
        // (w = 50), so the spike leaves a window that holds a NaN.
        let mut data = noise(3_000, 32);
        data[1_000] = 3.0e38;
        data[1_010] = f64::NAN;

        // Act / Assert -- windows after the NaN leaves too (1_060..).
        for w in [50, 200] {
            assert_all_three_agree(&data, w, "spike left under nan");
        }
    }

    #[test]
    fn rolling_running_sums_a_huge_sample_every_window_and_a_half_is_linear_and_equals_the_definition() {
        // Arrange
        let w = 100;
        let mut data = noise(20_000, 33);
        for i in (50..data.len()).step_by(w + w / 2) {
            data[i] = 3.0e38;
        }

        // Act
        let scanned = scans_of_all_three(&data, w);

        // Assert -- about one rebuild per spike, not one per sample.
        assert!(scanned <= 3 * 3 * data.len(), "scanned {scanned} samples");
        assert_all_three_agree(&data, w, "spike per window");
    }

    #[test]
    fn rolling_running_sums_steady_alternating_huge_terms_trigger_no_range_rebuilds() {
        // Arrange -- near-zero mean, huge terms: the case a naive "window sum
        // fell" test would trip on every sample.
        let w = 100;
        let data: Vec<f64> = (0..40_000).map(|i| if i % 2 == 0 { 1e6 } else { -1e6 } + 1e-3).collect();

        // Act
        let scanned = scans_of_all_three(&data, w);

        // Assert -- only the periodic rebuilds (every 4096 samples, three
        // functions, w samples each) plus a handful of start-up ones.
        let periodic = 3 * (data.len() / ROLLING_REBUILD_EVERY + 1) * w;
        assert!(scanned <= 4 * periodic, "scanned {scanned} samples, periodic budget {periodic}");
    }

    #[test]
    fn rolling_running_sums_empty_input_is_empty() {
        // Arrange / Act / Assert
        assert!(rolling_mean(&[], 5).is_empty());
        assert!(rolling_rms(&[], 5).is_empty());
        assert!(rolling_std(&[], 5).is_empty());
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
