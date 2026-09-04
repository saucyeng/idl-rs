//! Two-dimensional value-distribution histogram over a paired `(x, y)` cloud —
//! the binning core for the Analyze tab's 2-D histogram/raster chart (C3
//! §3.6). Pure: `xs`/`ys` in, a binned grid out; unlike `scatter::scatter_density`
//! (session-coupled, windows a live channel), this is the plain grid-out
//! function the raster builder composes on top of.
//!
//! Row convention mirrors `scatter::scatter_density`'s own `counts` layout
//! (row = y bin), **not** `spectrogram::SpectrogramResult`'s row-major
//! layout — that matrix's row axis is time, a different convention entirely.

use crate::scatter::pair_finite;

/// A 2-D equal-width histogram of a paired `(x, y)` cloud.
pub struct Histogram2dResult {
    /// X bin boundaries, ascending, length `nx + 1`.
    pub bin_edges_x: Vec<f64>,
    /// Y bin boundaries, ascending, length `ny + 1`.
    pub bin_edges_y: Vec<f64>,
    /// Finite-pair count per cell, row-major `ny × nx` (row = y bin, matching
    /// `scatter_density`'s own convention). Cell `(row, col)` is at
    /// `counts[row * nx + col]`.
    pub counts: Vec<u32>,
    /// Number of X bins.
    pub nx: usize,
    /// Number of Y bins.
    pub ny: usize,
}

impl Histogram2dResult {
    /// The truly-empty result: no bins, no grid. Reserved for `nx == 0 ||
    /// ny == 0` only — a degenerate (e.g. zero-width) *range* with both bin
    /// counts positive still returns a full-size grid (see [`histogram2d`]).
    pub fn empty() -> Self {
        Histogram2dResult { bin_edges_x: Vec::new(), bin_edges_y: Vec::new(), counts: Vec::new(), nx: 0, ny: 0 }
    }
}

/// Bins finite `(x, y)` pairs into an `nx × ny` equal-width grid.
///
/// `xs`/`ys` are paired by index and filtered to finite pairs via
/// `scatter::pair_finite`. Each axis's range is `range_x`/`range_y` when
/// `Some`, else the finite data's own min/max. A pair outside its resolved
/// `[lo, hi]` range on either axis is skipped (mirrors `histogram.rs`'s
/// explicit-range behaviour) — for an auto-derived range this is a no-op,
/// since the range is exactly the data's own extent.
///
/// Degenerate rule (L3-R31, deliberately **not** `histogram.rs`'s "zero-width
/// → empty"): whenever `nx > 0 && ny > 0`, this always returns a full-size
/// `counts` grid (zeros where nothing lands) with resolved bin edges. A
/// zero-width axis (`hi <= lo`, e.g. a constant channel) collapses every
/// in-range value on that axis onto bin 0, rather than emptying the result —
/// this is what lets the raster builder treat "degenerate input" and "normal
/// input" identically, no special case. Only `nx == 0 || ny == 0` yields
/// [`Histogram2dResult::empty`].
pub fn histogram2d(
    xs: &[f64],
    ys: &[f64],
    nx: usize,
    ny: usize,
    range_x: Option<(f64, f64)>,
    range_y: Option<(f64, f64)>,
) -> Histogram2dResult {
    if nx == 0 || ny == 0 {
        return Histogram2dResult::empty();
    }

    let (xs, ys, _) = pair_finite(xs, ys, None);

    let (x0, x1) = resolve_axis_range(&xs, range_x);
    let (y0, y1) = resolve_axis_range(&ys, range_y);

    let mut counts = vec![0u32; nx * ny];
    for i in 0..xs.len() {
        let (x, y) = (xs[i], ys[i]);
        if x < x0 || x > x1 || y < y0 || y > y1 {
            continue;
        }
        let col = bin_index(x, x0, x1, nx);
        let row = bin_index(y, y0, y1, ny);
        counts[row * nx + col] += 1;
    }

    Histogram2dResult { bin_edges_x: bin_edges(x0, x1, nx), bin_edges_y: bin_edges(y0, y1, ny), counts, nx, ny }
}

// Resolves one axis's binning range: the caller's explicit range, or the
// finite data's own min/max (0.0..0.0 when there is no finite data at all).
fn resolve_axis_range(vs: &[f64], range: Option<(f64, f64)>) -> (f64, f64) {
    if let Some(r) = range {
        return r;
    }
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in vs {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    if mn > mx {
        (0.0, 0.0)
    } else {
        (mn, mx)
    }
}

// Maps `v` in `[lo, hi]` to a bin index in `0..n`. `hi <= lo` (zero-width
// axis) collapses every value onto bin 0 — mirrors `scatter.rs`'s
// `scatter_density::bin_index`.
fn bin_index(v: f64, lo: f64, hi: f64, n: usize) -> usize {
    if hi <= lo {
        return 0;
    }
    let t = (v - lo) / (hi - lo);
    ((t * n as f64).floor() as isize).clamp(0, n as isize - 1) as usize
}

// Ascending bin boundaries over `[lo, hi]`, length `n + 1`.
fn bin_edges(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let width = hi - lo;
    (0..=n).map(|i| lo + i as f64 / n as f64 * width).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn histogram2d_all_finite_uniform_grid_counts_sum_to_total_pairs() {
        // Arrange — 100 points on a 10x10 grid, x/y both 0..9.
        let xs: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let ys: Vec<f64> = (0..100).map(|i| (i / 10) as f64).collect();

        // Act
        let h = histogram2d(&xs, &ys, 10, 10, None, None);

        // Assert — every pair lands somewhere, none lost.
        assert_eq!(h.nx, 10);
        assert_eq!(h.ny, 10);
        assert_eq!(h.counts.len(), 100);
        assert_eq!(h.counts.iter().sum::<u32>(), 100);
    }

    #[test]
    fn histogram2d_nx_or_ny_zero_is_empty() {
        // Arrange/Act/Assert — either axis at zero bins yields the empty sentinel.
        assert_eq!(histogram2d(&[1.0], &[1.0], 0, 4, None, None).counts.len(), 0);
        assert_eq!(histogram2d(&[1.0], &[1.0], 4, 0, None, None).counts.len(), 0);
    }

    #[test]
    fn histogram2d_non_finite_pairs_skipped() {
        // Arrange — one NaN x, one infinite y, one clean pair.
        let xs = vec![f64::NAN, 1.0, 2.0];
        let ys = vec![0.0, f64::INFINITY, 2.0];

        // Act
        let h = histogram2d(&xs, &ys, 4, 4, None, None);

        // Assert — only the (2.0, 2.0) pair counted.
        assert_eq!(h.counts.iter().sum::<u32>(), 1);
    }

    #[test]
    fn histogram2d_explicit_range_narrower_than_data_skips_out_of_range_pairs() {
        // Arrange — data spans 0..10, range restricted to 0..5.
        let xs = vec![1.0, 3.0, 8.0];
        let ys = vec![1.0, 3.0, 8.0];

        // Act
        let h = histogram2d(&xs, &ys, 5, 5, Some((0.0, 5.0)), Some((0.0, 5.0)));

        // Assert — the (8, 8) pair falls outside the range and is skipped.
        assert_eq!(h.counts.iter().sum::<u32>(), 2);
        assert_relative_eq!(*h.bin_edges_x.last().unwrap(), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn histogram2d_degenerate_zero_width_x_range_is_full_size_not_empty() {
        // Arrange — every x value identical (zero-width x range), y varies.
        let xs = vec![3.0, 3.0, 3.0];
        let ys = vec![0.0, 1.0, 2.0];

        // Act
        let h = histogram2d(&xs, &ys, 4, 3, None, None);

        // Assert — full-size grid (per L3-R31), not `Histogram2dResult::empty()`;
        // every value collapses onto x bin 0.
        assert_eq!(h.nx, 4);
        assert_eq!(h.ny, 3);
        assert_eq!(h.counts.len(), 12);
        assert_eq!(h.counts.iter().sum::<u32>(), 3);
        for row in 0..3 {
            for col in 1..4 {
                assert_eq!(h.counts[row * 4 + col], 0);
            }
        }
    }
}
