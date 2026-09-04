//! Tile-based min/max decimation for time-series chart rendering.
//!
//! Reduces N raw samples to 2 floats per bucket (min, max) so fl_chart
//! can render an envelope that preserves spike fidelity at any zoom.

/// Bucket size at tier k is `TIER_BASE.pow(k)` raw samples.
pub const TIER_BASE: u32 = 8;

/// Number of buckets per tile, at every tier.
pub const TILE_SIZE_BUCKETS: u32 = 1024;

/// Largest tier `k` for which `TIER_BASE.pow(k)` fits `u32` — the engine's
/// configured tier range (C3 §3.5, ledger R25). `fetch_tile`'s L5 wrapper
/// rejects `tier > MAX_TIER` with `invalid_argument` before any bytes are
/// produced; [`decimate_channel`] and `SessionHandle::decimate_tile` still
/// degrade a too-large tier to an all-NaN tile via `checked_pow`, as defense
/// in depth rather than a panic.
pub const MAX_TIER: u32 = 10;

/// Decimates `samples[start..start+span]` into `2 * TILE_SIZE_BUCKETS` floats,
/// interleaved `[min, max, min, max, ...]` per bucket.
///
/// `bucket_size`: raw samples per bucket (= `TIER_BASE.pow(tier)`).
/// `start`: first raw sample index this tile covers.
///
/// NaN handling:
/// - bucket whose samples are all NaN → emits `[NaN, NaN]`
/// - mixed bucket → min/max computed over finite samples only
/// - bucket beyond `samples.len()` → emits `[NaN, NaN]` (right-edge padding)
pub fn decimate_tile_pure(
    samples: &[f64],
    bucket_size: u32,
    start: u32,
) -> Vec<f64> {
    let n_buckets = TILE_SIZE_BUCKETS as usize;
    let mut out = Vec::with_capacity(n_buckets * 2);
    let bs = bucket_size as usize;
    let start = start as usize;

    for b in 0..n_buckets {
        let lo = start + b * bs;
        let hi = (lo + bs).min(samples.len());
        if lo >= samples.len() {
            out.push(f64::NAN);
            out.push(f64::NAN);
            continue;
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut any_finite = false;
        for &v in &samples[lo..hi] {
            if v.is_finite() {
                any_finite = true;
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
        }
        if any_finite {
            out.push(mn);
            out.push(mx);
        } else {
            out.push(f64::NAN);
            out.push(f64::NAN);
        }
    }
    out
}

/// Decimate the tile at (`tier`, `tile_index`) from a channel's raw samples.
/// `tier` 0 = raw (bucket size 1), k = `TIER_BASE.pow(k)`. Returns
/// `2 * TILE_SIZE_BUCKETS` interleaved `[min, max, …]` floats; past-end buckets
/// are NaN-padded. Wraps the tier→bucket-size and tile→start arithmetic around
/// [`decimate_tile_pure`].
pub fn decimate_channel(samples: &[f64], tier: u32, tile_index: u32) -> Vec<f64> {
    let bucket_size = TIER_BASE.checked_pow(tier).unwrap_or(u32::MAX);
    let start = tile_index
        .saturating_mul(TILE_SIZE_BUCKETS)
        .saturating_mul(bucket_size);
    decimate_tile_pure(samples, bucket_size, start)
}

/// An all-NaN tile (every bucket empty) — returned for an absent channel.
pub fn empty_tile() -> Vec<f64> {
    vec![f64::NAN; (TILE_SIZE_BUCKETS as usize) * 2]
}

/// Column `j`'s raw-sample bucket range (half-open, raw-sample index space)
/// within a tile's own sample span — `tile_index * TILE_SIZE_BUCKETS *
/// bucket_size(tier) .. + TILE_SIZE_BUCKETS * bucket_size(tier)`, divided
/// into `column_count` equal-width slices (C3 §3.5, ledger R30). Shared by
/// [`column_stats`] and [`column_times_us`] so their bucket boundaries can
/// never diverge. `u64` throughout — `bucket_size` alone can reach
/// `u32::MAX` under [`MAX_TIER`], so `tile_span`/`tile_start` overflow `u32`.
fn column_sample_range(tier: u32, tile_index: u32, column_count: u32, j: u32) -> (u64, u64) {
    let bucket_size = TIER_BASE.checked_pow(tier).unwrap_or(u32::MAX) as u64;
    let tile_span = TILE_SIZE_BUCKETS as u64 * bucket_size;
    let tile_start = tile_index as u64 * tile_span;
    let cc = column_count as u64;
    let lo = tile_start + (j as u64 * tile_span) / cc;
    let hi = tile_start + ((j as u64 + 1) * tile_span) / cc;
    (lo, hi)
}

/// Computes `(min, max, mean)` per pixel column across a tile's raw-sample
/// span — the coarser per-pixel-column stats shipped alongside the bucket
/// region so hover reads never need IPC (design §6). NaN handling matches
/// [`decimate_tile_pure`]: an all-NaN or past-end column emits
/// `(NaN, NaN, NaN)`; a mixed column computes stats over finite samples
/// only, `mean = sum / count` (never `(min + max) / 2`). `column_count == 0`
/// is legal and returns an empty `Vec`.
pub fn column_stats(
    samples: &[f64],
    tier: u32,
    tile_index: u32,
    column_count: u32,
) -> Vec<(f32, f32, f32)> {
    let mut out = Vec::with_capacity(column_count as usize);
    for j in 0..column_count {
        let (lo, hi) = column_sample_range(tier, tile_index, column_count, j);
        let lo = (lo.min(samples.len() as u64)) as usize;
        let hi = (hi.min(samples.len() as u64)) as usize;
        if lo >= hi {
            out.push((f32::NAN, f32::NAN, f32::NAN));
            continue;
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for &v in &samples[lo..hi] {
            if v.is_finite() {
                if v < mn { mn = v; }
                if v > mx { mx = v; }
                sum += v;
                count += 1;
            }
        }
        if count > 0 {
            out.push((mn as f32, mx as f32, (sum / count as f64) as f32));
        } else {
            out.push((f32::NAN, f32::NAN, f32::NAN));
        }
    }
    out
}

/// Returns the recorded `t_us` (µs) of the **first sample** in each pixel
/// column's bucket range — mirrors [`column_stats`]'s bucket-range math via
/// [`column_sample_range`], but has no NaN concept: `t_us` is exact, never
/// interpolated or synthesized from `nominal_rate_hz` (C1 §2/§3.1, "time is
/// recorded, not assumed"). A column whose bucket range contains **no**
/// sample (past the end of `t_us`, or an empty range) gets the sentinel
/// `i64::MIN` — that sentinel means "no sample here", never "sample present
/// but NaN": a NaN-valued sample at the start of a bucket range still yields
/// its own real `t_us` (C3 §3.5). `column_count == 0` is legal and returns
/// an empty `Vec`.
pub fn column_times_us(t_us: &[i64], tier: u32, tile_index: u32, column_count: u32) -> Vec<i64> {
    let mut out = Vec::with_capacity(column_count as usize);
    for j in 0..column_count {
        let (lo, hi) = column_sample_range(tier, tile_index, column_count, j);
        let lo = (lo.min(t_us.len() as u64)) as usize;
        let hi = (hi.min(t_us.len() as u64)) as usize;
        if lo >= hi {
            out.push(i64::MIN);
        } else {
            out.push(t_us[lo]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_tile_pure_all_finite_bucket_returns_min_and_max() {
        // Arrange
        let samples: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let bucket_size = 8_u32;

        // Act
        let out = decimate_tile_pure(&samples, bucket_size, 0);

        // Assert — bucket 0 covers samples 0..8 (min=0, max=7).
        // bucket 1 covers samples 8..16 (min=8, max=15).
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 7.0);
        assert_eq!(out[2], 8.0);
        assert_eq!(out[3], 15.0);
        // remaining buckets beyond samples.len() are padded NaN
        assert!(out[4].is_nan());
        assert!(out[5].is_nan());
    }

    #[test]
    fn decimate_tile_pure_mixed_nan_bucket_returns_min_max_of_finite_only() {
        // Arrange
        let samples = vec![1.0, f64::NAN, 3.0, f64::NAN, 2.0, f64::NAN, 4.0, f64::NAN];

        // Act
        let out = decimate_tile_pure(&samples, 8, 0);

        // Assert
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 4.0);
    }

    #[test]
    fn decimate_tile_pure_all_nan_bucket_returns_nan_pair() {
        // Arrange
        let samples = vec![f64::NAN; 8];

        // Act
        let out = decimate_tile_pure(&samples, 8, 0);

        // Assert
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
    }

    #[test]
    fn decimate_tile_pure_single_sample_spike_preserved_as_max() {
        // Arrange — eight samples, all 0.1 except a single 99.0 spike
        let mut samples = vec![0.1_f64; 8];
        samples[5] = 99.0;

        // Act
        let out = decimate_tile_pure(&samples, 8, 0);

        // Assert
        assert_eq!(out[0], 0.1);
        assert_eq!(out[1], 99.0);
    }

    #[test]
    fn decimate_tile_pure_partial_last_bucket_no_corruption_earlier_buckets() {
        // Arrange — 10 samples, bucket_size 8 → bucket 0 full, bucket 1 partial (2 samples)
        let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();

        // Act
        let out = decimate_tile_pure(&samples, 8, 0);

        // Assert
        assert_eq!(out[0], 0.0);   // bucket 0 min
        assert_eq!(out[1], 7.0);   // bucket 0 max — unchanged by partial bucket 1
        assert_eq!(out[2], 8.0);   // bucket 1 min over the 2 finite samples
        assert_eq!(out[3], 9.0);   // bucket 1 max
        // bucket 2+ beyond samples.len() → NaN
        assert!(out[4].is_nan());
        assert!(out[5].is_nan());
    }

    #[test]
    fn decimate_tile_pure_start_offset_skips_earlier_samples() {
        // Arrange — 24 samples; ask for tile starting at sample 16
        let samples: Vec<f64> = (0..24).map(|i| i as f64).collect();

        // Act
        let out = decimate_tile_pure(&samples, 8, 16);

        // Assert — bucket 0 of this tile covers samples 16..24
        assert_eq!(out[0], 16.0);
        assert_eq!(out[1], 23.0);
        // bucket 1+ beyond samples.len() → NaN
        assert!(out[2].is_nan());
        assert!(out[3].is_nan());
    }

    #[test]
    fn decimate_channel_tier0_returns_raw_pairs() {
        // Arrange — tier 0 = bucket size 1; each bucket is one sample → [s, s].
        let samples: Vec<f64> = (0..16).map(|i| i as f64).collect();

        // Act
        let out = decimate_channel(&samples, 0, 0);

        // Assert
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn decimate_channel_tier1_buckets_eight_samples() {
        // Arrange — tier 1 = bucket size 8.
        let samples: Vec<f64> = (0..16).map(|i| i as f64).collect();

        // Act
        let out = decimate_channel(&samples, 1, 0);

        // Assert — bucket 0 = samples 0..8 (min 0, max 7); bucket 1 = 8..16.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 7.0);
        assert_eq!(out[2], 8.0);
        assert_eq!(out[3], 15.0);
    }

    #[test]
    fn decimate_channel_nonzero_tile_index_starts_past_small_array() {
        // Arrange — tier 0, tile 1 starts at 1 * 1024 * 1 = 1024, past a tiny array.
        let samples = vec![1.0, 2.0, 3.0];

        // Act
        let out = decimate_channel(&samples, 0, 1);

        // Assert — entirely past-end → all NaN.
        assert_eq!(out.len(), (TILE_SIZE_BUCKETS as usize) * 2);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn empty_tile_is_all_nan_full_length() {
        // Act
        let out = empty_tile();

        // Assert
        assert_eq!(out.len(), (TILE_SIZE_BUCKETS as usize) * 2);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn tile_size_buckets_and_tier_base_are_canonical_values() {
        // Arrange / Act / Assert — locks public constants to spec values.
        assert_eq!(TILE_SIZE_BUCKETS, 1024);
        assert_eq!(TIER_BASE, 8);
    }

    #[test]
    fn decimate_channel_tier_above_max_tier_returns_all_nan_tile_no_panic() {
        // Arrange — MAX_TIER + 1 overflows TIER_BASE.pow in u32; tile_index 1
        // (not 0) so the saturating start offset lands past samples.len(),
        // making every bucket NaN rather than just the ones past bucket 0.
        let samples: Vec<f64> = (0..16).map(|i| i as f64).collect();

        // Act
        let out = decimate_channel(&samples, MAX_TIER + 1, 1);

        // Assert — checked_pow degrades to an all-NaN tile, not a panic.
        assert_eq!(out.len(), (TILE_SIZE_BUCKETS as usize) * 2);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn max_tier_is_the_largest_tier_tier_base_pow_does_not_overflow() {
        // Act / Assert
        assert!(TIER_BASE.checked_pow(MAX_TIER).is_some());
        assert!(TIER_BASE.checked_pow(MAX_TIER + 1).is_none());
    }

    #[test]
    fn column_stats_all_finite_column_returns_min_max_mean() {
        // Arrange — tier 0, tile 0, 4 columns over a 1024-sample span;
        // column 0 covers samples [0, 256).
        let samples: Vec<f64> = (0..1024).map(|i| i as f64).collect();

        // Act
        let out = column_stats(&samples, 0, 0, 4);

        // Assert
        assert_eq!(out[0], (0.0, 255.0, 127.5));
    }

    #[test]
    fn column_stats_mixed_nan_column_mean_over_finite_only() {
        // Arrange — column 0 covers samples [0, 256); half NaN.
        let mut samples = vec![f64::NAN; 1024];
        samples[0] = 1.0;
        samples[1] = 3.0;

        // Act
        let out = column_stats(&samples, 0, 0, 4);

        // Assert — mean over the two finite samples only, not (min+max)/2
        // over the whole (mostly-NaN) column.
        assert_eq!(out[0], (1.0, 3.0, 2.0));
    }

    #[test]
    fn column_stats_all_nan_column_returns_nan_triple() {
        // Arrange
        let samples = vec![f64::NAN; 1024];

        // Act
        let out = column_stats(&samples, 0, 0, 4);

        // Assert
        assert!(out[0].0.is_nan() && out[0].1.is_nan() && out[0].2.is_nan());
    }

    #[test]
    fn column_stats_past_end_column_returns_nan_triple() {
        // Arrange — tile 1 starts at sample 1024, past a tiny array.
        let samples = vec![1.0, 2.0, 3.0];

        // Act
        let out = column_stats(&samples, 0, 1, 4);

        // Assert
        for c in &out {
            assert!(c.0.is_nan() && c.1.is_nan() && c.2.is_nan());
        }
    }

    #[test]
    fn column_stats_single_spike_min_max_mean_all_correct() {
        // Arrange — column 0's 256 samples are all 0.1 except one 99.0 spike.
        let mut samples = vec![0.1_f64; 1024];
        samples[10] = 99.0;

        // Act
        let out = column_stats(&samples, 0, 0, 4);

        // Assert — mean is sum/count, not (min+max)/2.
        let expected_mean = (255.0 * 0.1 + 99.0) / 256.0;
        assert_eq!(out[0].0, 0.1);
        assert_eq!(out[0].1, 99.0);
        assert!((out[0].2 as f64 - expected_mean).abs() < 1e-4);
    }

    #[test]
    fn column_stats_column_count_zero_returns_empty_vec() {
        // Arrange
        let samples: Vec<f64> = (0..1024).map(|i| i as f64).collect();

        // Act
        let out = column_stats(&samples, 0, 0, 0);

        // Assert
        assert!(out.is_empty());
    }

    #[test]
    fn column_times_us_column_fully_within_data_returns_first_sample_t_us() {
        // Arrange — column 0 covers samples [0, 256).
        let t_us: Vec<i64> = (0..1024).map(|i| i * 1000).collect();

        // Act
        let out = column_times_us(&t_us, 0, 0, 4);

        // Assert
        assert_eq!(out[0], 0);
    }

    #[test]
    fn column_times_us_past_end_column_returns_sentinel() {
        // Arrange — tile 1 starts at sample 1024, past a tiny array.
        let t_us = vec![0i64, 1000, 2000];

        // Act
        let out = column_times_us(&t_us, 0, 1, 4);

        // Assert
        for t in &out {
            assert_eq!(*t, i64::MIN);
        }
    }

    #[test]
    fn column_times_us_column_count_zero_returns_empty_vec() {
        // Arrange
        let t_us: Vec<i64> = (0..1024).map(|i| i * 1000).collect();

        // Act
        let out = column_times_us(&t_us, 0, 0, 0);

        // Assert
        assert!(out.is_empty());
    }

    #[test]
    fn column_times_us_nan_valued_sample_at_bucket_start_still_returns_real_t_us() {
        // Arrange — column_times_us takes only t_us, never the samples, so a
        // NaN-valued sample at index 0 carries its own real t_us: the
        // sentinel is for "no sample", never "sample present but NaN".
        let mut t_us: Vec<i64> = (0..1024).map(|i| i * 1000).collect();
        t_us[0] = 12345; // the "NaN sample"'s real recorded time
        let samples_would_be_nan_at_0 = f64::NAN;
        let _ = samples_would_be_nan_at_0; // documents the scenario; unused by column_times_us

        // Act
        let out = column_times_us(&t_us, 0, 0, 4);

        // Assert
        assert_eq!(out[0], 12345);
    }
}
