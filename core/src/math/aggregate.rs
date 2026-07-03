//! Channel → scalar reducers for table-cell aggregates (`mean([Fork])`, …).
//! Also usable in channel math via scalar broadcast (e.g. de-meaning
//! `[Fork] - mean([Fork])`). Non-finite samples are skipped; an all-skipped or
//! empty input yields `NaN`. These are elementary folds with no scipy.signal
//! equivalent (numpy-domain, not signal-domain), so nothing here re-implements a
//! sci-rs/nalgebra primitive.
//!
//! In the function dispatch these share names with the existing windowed
//! statistics (`mean`/`std`/`rms`) and elementwise `min`/`max`; arity selects:
//! one channel argument → the scalar aggregate here, two arguments → the
//! existing rolling/elementwise form. See `eval::call_function`.

fn finite(data: &[f64]) -> impl Iterator<Item = f64> + '_ {
    data.iter().copied().filter(|v| v.is_finite())
}

/// Count of finite samples.
pub fn count(data: &[f64]) -> f64 {
    finite(data).count() as f64
}

/// Sum of finite samples (`0.0` for an empty/all-non-finite input).
pub fn sum(data: &[f64]) -> f64 {
    finite(data).sum()
}

/// Arithmetic mean of finite samples; `NaN` when none are finite.
pub fn mean(data: &[f64]) -> f64 {
    let (mut s, mut n) = (0.0, 0u64);
    for v in finite(data) {
        s += v;
        n += 1;
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

/// Maximum finite sample; `NaN` when none are finite.
pub fn max(data: &[f64]) -> f64 {
    finite(data).fold(f64::NEG_INFINITY, f64::max).pipe_nan_if_empty(data)
}

/// Minimum finite sample; `NaN` when none are finite.
pub fn min(data: &[f64]) -> f64 {
    finite(data).fold(f64::INFINITY, f64::min).pipe_nan_if_empty(data)
}

/// Root-mean-square of finite samples; `NaN` when none are finite.
pub fn rms(data: &[f64]) -> f64 {
    let (mut s, mut n) = (0.0, 0u64);
    for v in finite(data) {
        s += v * v;
        n += 1;
    }
    if n == 0 {
        f64::NAN
    } else {
        (s / n as f64).sqrt()
    }
}

/// Population standard deviation of finite samples; `NaN` when none are finite.
pub fn std_pop(data: &[f64]) -> f64 {
    let m = mean(data);
    if m.is_nan() {
        return f64::NAN;
    }
    let (mut s, mut n) = (0.0, 0u64);
    for v in finite(data) {
        s += (v - m) * (v - m);
        n += 1;
    }
    (s / n as f64).sqrt()
}

/// Median (50th percentile) of finite samples.
pub fn median(data: &[f64]) -> f64 {
    percentile(data, 50.0)
}

/// Linear-interpolated percentile (`q` in 0..=100) over finite samples; `NaN`
/// when none are finite.
pub fn percentile(data: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = finite(data).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (q / 100.0) * (v.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (rank - lo as f64) * (v[hi] - v[lo])
    }
}

/// First finite sample; `NaN` when none are finite.
pub fn first(data: &[f64]) -> f64 {
    finite(data).next().unwrap_or(f64::NAN)
}

/// Last finite sample; `NaN` when none are finite.
pub fn last(data: &[f64]) -> f64 {
    finite(data).last().unwrap_or(f64::NAN)
}

// Small helper so max/min return NaN (not ±inf) when no finite sample exists.
trait NanIfEmpty {
    fn pipe_nan_if_empty(self, data: &[f64]) -> f64;
}
impl NanIfEmpty for f64 {
    fn pipe_nan_if_empty(self, data: &[f64]) -> f64 {
        if finite(data).next().is_none() {
            f64::NAN
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reducers_on_known_input() {
        let d = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(mean(&d), 2.5);
        assert_relative_eq!(max(&d), 4.0);
        assert_relative_eq!(min(&d), 1.0);
        assert_relative_eq!(sum(&d), 10.0);
        assert_relative_eq!(rms(&d), (30.0_f64 / 4.0).sqrt());
        assert_relative_eq!(median(&d), 2.5); // even count → mean of middle two
        assert_relative_eq!(std_pop(&d), 1.118033988749895, epsilon = 1e-9);
        assert_relative_eq!(percentile(&d, 50.0), 2.5);
        assert_eq!(count(&d), 4.0);
        assert_eq!(first(&d), 1.0);
        assert_eq!(last(&d), 4.0);
    }

    #[test]
    fn reducers_skip_non_finite() {
        let d = [1.0, f64::NAN, 3.0, f64::INFINITY];
        assert_relative_eq!(mean(&d), 2.0); // only 1.0 and 3.0 counted
        assert_eq!(count(&d), 2.0);
        assert_eq!(last(&d), 3.0); // INFINITY skipped
    }

    #[test]
    fn empty_is_nan() {
        assert!(mean(&[]).is_nan());
        assert!(max(&[]).is_nan());
        assert!(min(&[]).is_nan());
        assert!(median(&[]).is_nan());
    }
}
