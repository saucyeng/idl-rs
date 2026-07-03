//! Compact, typed per-channel sample storage with lazy f64 materialization.
//!
//! A channel's resident form is a [`RawColumn`]: integer/f32 wire values plus a
//! `(scale, offset)` pair, or verbatim f64. Physical samples are widened on
//! demand — `physical = (raw as f64) * scale + offset` — reproducing exactly the
//! arithmetic the parser used to apply eagerly (v3 `raw × scale + offset`), so
//! output is byte-identical. The display path never materializes: it reads
//! decimated min/max tiles, which fold over the raw values and scale the
//! resulting pair (see [`RawColumn::min_max`]).
//!
//! See `docs/superpowers/specs/2026-06-03-compact-raw-storage-design.md`.

/// Per-channel sample storage.
///
/// `I16`/`I32`/`F32` are compact (2/4/4 bytes/sample) and carry the registry
/// `scale`/`offset`; `physical = (raw as f64) * scale + offset`. `F64` is
/// verbatim — the value is returned exactly as stored (no `× 1.0 + 0.0`, which
/// would flip `-0.0`), used for GPS, math results, GPX import, and any wire
/// type without a compact variant. `Ramp` (synthesized `Time`) and `Interp`
/// (synthesized `Distance`) compute values on demand — zero / GPS-rate
/// resident bytes respectively.
#[derive(Debug, Clone, PartialEq)]
pub enum RawColumn {
    /// 16-bit signed raw (IMU axes, i16 registry channels).
    I16 { data: Vec<i16>, scale: f64, offset: f64 },
    /// 32-bit signed raw (i32 registry channels).
    I32 { data: Vec<i32>, scale: f64, offset: f64 },
    /// 32-bit float raw (f32 registry channels).
    F32 { data: Vec<f32>, scale: f64, offset: f64 },
    /// Verbatim f64 — stored exactly as the parser computed it.
    F64(Vec<f64>),
    /// Synthesized time ramp: `value(i) = i as f64 / rate`, zero per-sample
    /// storage. The engine's `Time` channel — len/rate fully determine it, so
    /// storing 8 B/sample was pure overhead at season scale.
    Ramp { len: usize, rate: f64 },
    /// Samples held at a coarser base rate (seconds grid `i / base_rate`),
    /// linear-interpolated onto a denser output grid on demand. The `Distance`
    /// channel: base = cumulative metres at GPS rate; output = the Time grid.
    /// `value(i)` reproduces the former eager synthesis loop bit-for-bit
    /// (clamp at both ends; lerp between base samples). `base` is non-empty by
    /// construction (synthesis skips empty speed channels).
    Interp { base: Vec<f64>, base_rate: f64, out_rate: f64, len: usize },
}

impl RawColumn {
    /// Number of samples.
    pub fn len(&self) -> usize {
        match self {
            RawColumn::I16 { data, .. } => data.len(),
            RawColumn::I32 { data, .. } => data.len(),
            RawColumn::F32 { data, .. } => data.len(),
            RawColumn::F64(data) => data.len(),
            RawColumn::Ramp { len, .. } => *len,
            RawColumn::Interp { len, .. } => *len,
        }
    }

    /// `true` when the column holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Widen the whole column to physical f64. Transient — the f64 form is never
    /// resident. For `F64` this clones verbatim.
    pub fn materialize(&self) -> Vec<f64> {
        match self {
            RawColumn::I16 { data, scale, offset } => {
                data.iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::I32 { data, scale, offset } => {
                data.iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::F32 { data, scale, offset } => {
                data.iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::F64(data) => data.clone(),
            RawColumn::Ramp { len, rate } => (0..*len).map(|i| ramp_value(i, *rate)).collect(),
            RawColumn::Interp { base, base_rate, out_rate, len } => {
                (0..*len).map(|i| interp_value(base, *base_rate, *out_rate, i)).collect()
            }
        }
    }

    /// Widen the half-open index window `[start, end)` to physical f64, clamped
    /// to the column length. Empty when `start >= end` or `start >= len`.
    pub fn materialize_range(&self, start: usize, end: usize) -> Vec<f64> {
        let len = self.len();
        let lo = start.min(len);
        let hi = end.min(len);
        if lo >= hi {
            return Vec::new();
        }
        match self {
            RawColumn::I16 { data, scale, offset } => {
                data[lo..hi].iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::I32 { data, scale, offset } => {
                data[lo..hi].iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::F32 { data, scale, offset } => {
                data[lo..hi].iter().map(|&r| r as f64 * scale + offset).collect()
            }
            RawColumn::F64(data) => data[lo..hi].to_vec(),
            RawColumn::Ramp { rate, .. } => (lo..hi).map(|i| ramp_value(i, *rate)).collect(),
            RawColumn::Interp { base, base_rate, out_rate, .. } => {
                (lo..hi).map(|i| interp_value(base, *base_rate, *out_rate, i)).collect()
            }
        }
    }

    /// Physical value at index `i`, or `None` if out of range.
    pub fn value_at(&self, i: usize) -> Option<f64> {
        match self {
            RawColumn::I16 { data, scale, offset } => data.get(i).map(|&r| r as f64 * scale + offset),
            RawColumn::I32 { data, scale, offset } => data.get(i).map(|&r| r as f64 * scale + offset),
            RawColumn::F32 { data, scale, offset } => data.get(i).map(|&r| r as f64 * scale + offset),
            RawColumn::F64(data) => data.get(i).copied(),
            RawColumn::Ramp { len, rate } => (i < *len).then(|| ramp_value(i, *rate)),
            RawColumn::Interp { base, base_rate, out_rate, len } => {
                (i < *len).then(|| interp_value(base, *base_rate, *out_rate, i))
            }
        }
    }

    /// Finite (min, max) of the physical samples, or `None` when empty or all
    /// non-finite. For compact columns the fold runs over the **raw** values and
    /// the resulting raw min/max pair is scaled — identical to scaling every
    /// sample, but allocation-free. With `scale < 0` the affine map is
    /// decreasing, so the raw-min and raw-max swap roles; the result is
    /// re-ordered into `(min, max)`. NaN/±∞ are ignored (matching the decimation
    /// fold). For `F64`, folds the physical values directly.
    pub fn min_max(&self) -> Option<(f64, f64)> {
        match self {
            RawColumn::I16 { data, scale, offset } => int_min_max(data.iter().map(|&r| r as f64), *scale, *offset),
            RawColumn::I32 { data, scale, offset } => int_min_max(data.iter().map(|&r| r as f64), *scale, *offset),
            RawColumn::F32 { data, scale, offset } => int_min_max(data.iter().map(|&r| r as f64), *scale, *offset),
            RawColumn::F64(data) => finite_min_max(data.iter().copied()),
            // Ramp is monotonically increasing: extrema are the endpoints.
            RawColumn::Ramp { len, rate } => (*len > 0).then(|| (0.0, ramp_value(*len - 1, *rate))),
            RawColumn::Interp { base, base_rate, out_rate, len } => {
                finite_min_max((0..*len).map(|i| interp_value(base, *base_rate, *out_rate, i)))
            }
        }
    }

    /// Finite (min, max) of the physical samples in the half-open window
    /// `[lo, hi)`, clamped to the column length; `None` when the clamped window
    /// is empty or all-non-finite. Same extremum-pair scaling as
    /// [`RawColumn::min_max`] — the decimation fold reads raw values without
    /// materializing the window.
    pub fn min_max_range(&self, lo: usize, hi: usize) -> Option<(f64, f64)> {
        let len = self.len();
        let lo = lo.min(len);
        let hi = hi.min(len);
        if lo >= hi {
            return None;
        }
        match self {
            RawColumn::I16 { data, scale, offset } => {
                int_min_max(data[lo..hi].iter().map(|&r| r as f64), *scale, *offset)
            }
            RawColumn::I32 { data, scale, offset } => {
                int_min_max(data[lo..hi].iter().map(|&r| r as f64), *scale, *offset)
            }
            RawColumn::F32 { data, scale, offset } => {
                int_min_max(data[lo..hi].iter().map(|&r| r as f64), *scale, *offset)
            }
            RawColumn::F64(data) => finite_min_max(data[lo..hi].iter().copied()),
            // Ramp is monotonically increasing; lo < hi is guaranteed here.
            RawColumn::Ramp { rate, .. } => Some((ramp_value(lo, *rate), ramp_value(hi - 1, *rate))),
            RawColumn::Interp { base, base_rate, out_rate, .. } => {
                finite_min_max((lo..hi).map(|i| interp_value(base, *base_rate, *out_rate, i)))
            }
        }
    }
}

/// `Ramp` physical value at index `i` — same IEEE ops as the eager builder
/// (`i as f64 / rate`), so output is bit-identical to the former resident
/// `Time` vector.
fn ramp_value(i: usize, rate: f64) -> f64 {
    i as f64 / rate
}

/// `Interp` physical value at output index `i` — the former eager Distance
/// synthesis loop verbatim: `t = i/out_rate; f = t*base_rate;` clamp-lerp on
/// `base`. `base` is non-empty by construction.
fn interp_value(base: &[f64], base_rate: f64, out_rate: f64, i: usize) -> f64 {
    let last_idx = base.len() - 1;
    let t = i as f64 / out_rate;
    let f = t * base_rate;
    if f <= 0.0 {
        0.0
    } else if f >= last_idx as f64 {
        base[last_idx]
    } else {
        let lo = f.floor() as usize;
        let frac = f - lo as f64;
        base[lo] + frac * (base[lo + 1] - base[lo])
    }
}

/// Finite min/max over raw values, then apply `(raw * scale + offset)` to the
/// extremum pair and order the result. Reproduces the per-sample physical value
/// at the min/max index exactly (same IEEE ops), so it matches `materialize`.
fn int_min_max(raws: impl Iterator<Item = f64>, scale: f64, offset: f64) -> Option<(f64, f64)> {
    let (rmin, rmax) = finite_min_max(raws)?;
    let a = rmin * scale + offset;
    let b = rmax * scale + offset;
    Some(if a <= b { (a, b) } else { (b, a) })
}

/// Finite min/max over an f64 iterator. `None` when empty or all non-finite.
fn finite_min_max(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut any = false;
    for v in values {
        if v.is_finite() {
            any = true;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }
    if any {
        Some((min, max))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_materialize_applies_scale_and_offset_like_parse() {
        // Arrange — accel scale 32/32768, offset 0; raw 16384 → 16.0 (same as v3 parse).
        let scale = 32.0 / 32768.0;
        let col = RawColumn::I16 { data: vec![16384, -8192, 0], scale, offset: 0.0 };

        // Act
        let out = col.materialize();

        // Assert — bit-identical to (raw as f64) * scale + offset.
        assert_eq!(out[0], 16384.0 * scale);
        assert_eq!(out[1], -8192.0 * scale);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn i16_with_offset_matches_formula() {
        // Arrange — scale 0.5, offset 1.0 (the v3 Brake test).
        let col = RawColumn::I16 { data: vec![100, 200, 300], scale: 0.5, offset: 1.0 };

        // Act + Assert
        assert_eq!(col.materialize(), vec![100.0 * 0.5 + 1.0, 200.0 * 0.5 + 1.0, 300.0 * 0.5 + 1.0]);
    }

    #[test]
    fn f64_materialize_is_verbatim_including_negative_zero() {
        // Arrange — F64 must NOT apply *1.0+0.0 (which flips -0.0 to +0.0).
        let col = RawColumn::F64(vec![-0.0, 1.5, f64::NAN]);

        // Act
        let out = col.materialize();

        // Assert — -0.0 preserved (bit pattern), NaN preserved.
        assert!(out[0].is_sign_negative() && out[0] == 0.0);
        assert_eq!(out[1], 1.5);
        assert!(out[2].is_nan());
    }

    #[test]
    fn i32_and_f32_materialize_apply_formula() {
        // Arrange
        let i32col = RawColumn::I32 { data: vec![1_000_000, -2_000_000], scale: 2.0, offset: 0.5 };
        let f32col = RawColumn::F32 { data: vec![1.5_f32, -3.25_f32], scale: 1.0, offset: 0.0 };

        // Act + Assert
        assert_eq!(i32col.materialize(), vec![1_000_000.0 * 2.0 + 0.5, -2_000_000.0 * 2.0 + 0.5]);
        assert_eq!(f32col.materialize(), vec![1.5_f32 as f64, -3.25_f32 as f64]);
    }

    #[test]
    fn materialize_range_clamps_half_open() {
        // Arrange
        let col = RawColumn::I16 { data: vec![0, 1, 2, 3, 4], scale: 1.0, offset: 0.0 };

        // Act + Assert
        assert_eq!(col.materialize_range(1, 4), vec![1.0, 2.0, 3.0]);
        assert_eq!(col.materialize_range(3, 100), vec![3.0, 4.0]);
        assert!(col.materialize_range(4, 4).is_empty());
        assert!(col.materialize_range(10, 20).is_empty());
    }

    #[test]
    fn len_and_is_empty() {
        assert_eq!(RawColumn::I16 { data: vec![1, 2], scale: 1.0, offset: 0.0 }.len(), 2);
        assert!(RawColumn::F64(Vec::new()).is_empty());
        assert!(!RawColumn::F64(vec![0.0]).is_empty());
    }

    #[test]
    fn value_at_returns_physical_or_none() {
        let col = RawColumn::I16 { data: vec![10, 20], scale: 0.5, offset: 1.0 };
        assert_eq!(col.value_at(1), Some(20.0 * 0.5 + 1.0));
        assert_eq!(col.value_at(2), None);
    }

    #[test]
    fn min_max_positive_scale_matches_materialized_bounds() {
        // Arrange — scale > 0: raw min/max map straight through.
        let col = RawColumn::I16 { data: vec![3, 1, 4, 1, 5], scale: 2.0, offset: 1.0 };

        // Act
        let (min, max) = col.min_max().unwrap();

        // Assert — equals min/max of materialized values.
        let m = col.materialize();
        assert_eq!(min, m.iter().cloned().fold(f64::INFINITY, f64::min));
        assert_eq!(max, m.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        assert_eq!(min, 1.0 * 2.0 + 1.0);
        assert_eq!(max, 5.0 * 2.0 + 1.0);
    }

    #[test]
    fn min_max_negative_scale_swaps_pair() {
        // Arrange — scale < 0: the affine map is decreasing, so raw-max → physical-min.
        let col = RawColumn::I16 { data: vec![1, 5], scale: -2.0, offset: 0.0 };

        // Act
        let (min, max) = col.min_max().unwrap();

        // Assert — physical values are {-2, -10}; min=-10, max=-2.
        assert_eq!(min, -10.0);
        assert_eq!(max, -2.0);
    }

    #[test]
    fn min_max_f64_ignores_non_finite() {
        let col = RawColumn::F64(vec![f64::NAN, 2.0, f64::INFINITY, -3.0, f64::NEG_INFINITY]);
        assert_eq!(col.min_max(), Some((-3.0, 2.0)));
    }

    #[test]
    fn ramp_materialize_matches_eager_index_over_rate() {
        // Arrange — Time semantics: value(i) = i / rate.
        let col = RawColumn::Ramp { len: 5, rate: 200.0 };

        // Act + Assert — bit-identical to the eager (0..len).map(|i| i / rate).
        let want: Vec<f64> = (0..5).map(|i| i as f64 / 200.0).collect();
        assert_eq!(col.materialize(), want);
        assert_eq!(col.materialize_range(2, 4), &want[2..4]);
        assert_eq!(col.value_at(4), Some(4.0 / 200.0));
        assert_eq!(col.value_at(5), None);
        assert_eq!(col.len(), 5);
        assert_eq!(col.min_max(), Some((0.0, 4.0 / 200.0)));
        assert_eq!(col.min_max_range(1, 3), Some((1.0 / 200.0, 2.0 / 200.0)));
    }

    #[test]
    fn ramp_empty_is_none_min_max() {
        let col = RawColumn::Ramp { len: 0, rate: 100.0 };
        assert_eq!(col.min_max(), None);
        assert!(col.is_empty());
        assert!(col.materialize().is_empty());
    }

    #[test]
    fn interp_materialize_matches_eager_clamped_lerp() {
        // Arrange — Distance semantics: base at 1 Hz lerped onto a 2 Hz grid.
        let base = vec![0.0, 1.0, 2.0];
        let col =
            RawColumn::Interp { base: base.clone(), base_rate: 1.0, out_rate: 2.0, len: 4 };

        // Act + Assert — reproduces the eager synthesis loop exactly:
        // t = i/2; f = t*1; clamp-lerp → 0.0, 0.5, 1.0, 1.5.
        assert_eq!(col.materialize(), vec![0.0, 0.5, 1.0, 1.5]);
        assert_eq!(col.materialize_range(1, 3), vec![0.5, 1.0]);
        assert_eq!(col.value_at(3), Some(1.5));
        assert_eq!(col.value_at(4), None);
        assert_eq!(col.len(), 4);
        assert_eq!(col.min_max(), Some((0.0, 1.5)));
        assert_eq!(col.min_max_range(2, 4), Some((1.0, 1.5)));
    }

    #[test]
    fn interp_clamps_past_base_end_to_last_base_value() {
        // Arrange — output grid extends past the base channel's span.
        let col =
            RawColumn::Interp { base: vec![0.0, 10.0], base_rate: 1.0, out_rate: 1.0, len: 5 };

        // Act + Assert — f >= last_idx clamps to base[last].
        assert_eq!(col.materialize(), vec![0.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn min_max_range_windows_and_clamps_like_materialized_fold() {
        // Arrange — negative scale: the affine map is decreasing.
        let col = RawColumn::I16 { data: vec![3, 1, 4, 1, 5, 9, 2], scale: -2.0, offset: 1.0 };

        // Act + Assert — window [2,5): raws {4,1,5} → physical {-7,-1,-9}.
        assert_eq!(col.min_max_range(2, 5), Some((-9.0, -1.0)));
        assert_eq!(col.min_max_range(5, 100), col.min_max_range(5, 7));
        assert_eq!(col.min_max_range(7, 9), None);
        assert_eq!(col.min_max_range(3, 3), None);
    }

    #[test]
    fn min_max_range_f64_ignores_non_finite_in_window() {
        // Arrange
        let col = RawColumn::F64(vec![f64::NAN, 2.0, f64::INFINITY, -3.0, 7.0]);

        // Act + Assert — window [0,4): finite values {2.0, -3.0}.
        assert_eq!(col.min_max_range(0, 4), Some((-3.0, 2.0)));
        // All-NaN window → None.
        assert_eq!(col.min_max_range(0, 1), None);
    }

    #[test]
    fn min_max_empty_or_all_nan_is_none() {
        assert_eq!(RawColumn::F64(Vec::new()).min_max(), None);
        assert_eq!(RawColumn::F64(vec![f64::NAN]).min_max(), None);
        assert_eq!(RawColumn::I16 { data: Vec::new(), scale: 1.0, offset: 0.0 }.min_max(), None);
    }
}
