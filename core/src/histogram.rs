//! Value-distribution histogram over channel samples — equal-width binning for
//! the Analyze tab's histogram chart (suspension velocity / travel
//! distributions). Pure: samples in, bin edges + counts out.
//!
//! Not a sci-rs call: histogram binning has no scipy.signal equivalent (it is a
//! numpy primitive, outside sci-rs's port surface). The work is an O(n) pass of
//! integer bin increments plus one min/max fold — memory-bound, not
//! float-arithmetic-bound, and auto-vectorised by the compiler at the crate's
//! release opt-level. There is nothing to route through sci-rs/nalgebra here.

/// Equal-width histogram of a channel's samples.
///
/// `counts` are `u32` (not `u64`/`usize`) so the FRB mirror maps to a Dart
/// `int` rather than `BigInt`; a per-session bin count cannot approach
/// `u32::MAX` (≈4.3 G) for any realistic log.
pub struct HistogramResult {
    /// Bin boundaries, ascending, length `bins + 1`. Bin `i` spans
    /// `bin_edges[i]..bin_edges[i + 1]`; the last bin is closed on the right so
    /// the maximum sample lands in it. Empty for a degenerate result.
    pub bin_edges: Vec<f64>,
    /// Finite-sample count per bin, length `bins`. Sums to `total`.
    pub counts: Vec<u32>,
    /// Total finite samples binned (non-finite values are skipped). The chart
    /// normalises each bar to a percentage via `count / total`.
    pub total: u32,
}

impl HistogramResult {
    /// The degenerate result: no bins, no samples. Returned for an absent or
    /// empty channel, `bins == 0`, an all-non-finite channel, or a zero-width
    /// range (e.g. a constant channel) — the chart shows an empty state.
    pub fn empty() -> Self {
        HistogramResult { bin_edges: Vec::new(), counts: Vec::new(), total: 0 }
    }
}

/// Bins finite `samples` into `bins` equal-width buckets.
///
/// Range is `[lo, hi]` when `range` is `Some` (the future manual-range path);
/// otherwise the finite data min/max. When `symmetric` (and no explicit
/// `range`), the auto range is widened to `[-m, m]` with `m = max(|min|, |max|)`
/// so zero sits on a bin boundary — the natural frame for a signed
/// suspension-velocity distribution (compression vs rebound). An explicit
/// `range` takes precedence over `symmetric`.
///
/// Non-finite samples (NaN, ±∞) are skipped and excluded from `total`. Samples
/// outside an explicit `range` are skipped. Degenerate inputs (`bins == 0`, no
/// finite sample, zero-width range) yield [`HistogramResult::empty`].
pub fn histogram(
    samples: &[f64],
    bins: usize,
    symmetric: bool,
    range: Option<(f64, f64)>,
) -> HistogramResult {
    if bins == 0 {
        return HistogramResult::empty();
    }

    // Resolve the binning range. The min/max fold is a single auto-vectorised
    // pass; it runs only when the caller did not supply an explicit range.
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
            for &v in samples {
                if v.is_finite() {
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                }
            }
            if mn > mx {
                return HistogramResult::empty(); // no finite sample
            }
            if symmetric {
                let m = mn.abs().max(mx.abs());
                (-m, m)
            } else {
                (mn, mx)
            }
        }
    };

    let width = hi - lo;
    if !(width > 0.0) {
        return HistogramResult::empty(); // constant channel / inverted range
    }

    let mut counts = vec![0u32; bins];
    let mut total: u32 = 0;
    let bins_f = bins as f64;
    for &v in samples {
        if !v.is_finite() || v < lo || v > hi {
            continue;
        }
        // Fractional position in [0, bins]; v == hi maps to `bins` → clamp into
        // the last bin so the closed-right boundary is honoured.
        let mut idx = ((v - lo) / width * bins_f) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
        total += 1;
    }

    let bin_edges = (0..=bins).map(|i| lo + i as f64 / bins_f * width).collect();
    HistogramResult { bin_edges, counts, total }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn histogram_uniform_ramp_bins_evenly_and_conserves_count() {
        // Arrange — 100 samples 0..99, auto range, 10 bins.
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Act
        let h = histogram(&data, 10, false, None);

        // Assert — 11 edges, 10 counts summing to every sample, no sample lost.
        assert_eq!(h.bin_edges.len(), 11);
        assert_eq!(h.counts.len(), 10);
        assert_eq!(h.total, 100);
        assert_eq!(h.counts.iter().sum::<u32>(), 100);
        assert_relative_eq!(h.bin_edges[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(*h.bin_edges.last().unwrap(), 99.0, epsilon = 1e-12);
    }

    #[test]
    fn histogram_symmetric_puts_zero_on_a_bin_edge() {
        // Arrange — asymmetric signed data; symmetric range should be [-3, 3].
        let data = vec![-1.0, 0.0, 3.0];

        // Act — 6 bins over [-3, 3] → width 1, so 0.0 is an interior edge.
        let h = histogram(&data, 6, true, None);

        // Assert
        assert_relative_eq!(h.bin_edges[0], -3.0, epsilon = 1e-12);
        assert_relative_eq!(*h.bin_edges.last().unwrap(), 3.0, epsilon = 1e-12);
        assert!(h.bin_edges.iter().any(|&e| e.abs() < 1e-12));
        assert_eq!(h.total, 3);
    }

    #[test]
    fn histogram_skips_non_finite_samples() {
        // Arrange — NaN and +∞ must not be counted.
        let data = vec![0.0, f64::NAN, 1.0, f64::INFINITY, 2.0];

        // Act — explicit range [0, 2] so the finite values all bin.
        let h = histogram(&data, 2, false, Some((0.0, 2.0)));

        // Assert — only 0, 1, 2 counted.
        assert_eq!(h.total, 3);
        assert_eq!(h.counts.iter().sum::<u32>(), 3);
    }

    #[test]
    fn histogram_max_sample_lands_in_last_bin() {
        // Arrange — the right edge value must fall in the final bin, not overflow.
        let data = vec![0.0, 10.0];

        // Act
        let h = histogram(&data, 5, false, None);

        // Assert — one sample in the first bin, one in the last.
        assert_eq!(h.counts.first(), Some(&1));
        assert_eq!(h.counts.last(), Some(&1));
        assert_eq!(h.total, 2);
    }

    #[test]
    fn histogram_degenerate_inputs_are_empty() {
        // Zero bins, no finite sample, and a constant channel all yield empty.
        assert!(histogram(&[1.0, 2.0], 0, false, None).counts.is_empty());
        assert!(histogram(&[f64::NAN], 4, false, None).counts.is_empty());
        assert!(histogram(&[5.0, 5.0, 5.0], 4, false, None).counts.is_empty());
    }
}
