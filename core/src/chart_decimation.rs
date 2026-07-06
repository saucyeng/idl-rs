//! Tile-based min/max decimation for time-series chart rendering.
//!
//! Reduces N raw samples to 2 floats per bucket (min, max) so fl_chart
//! can render an envelope that preserves spike fidelity at any zoom.

/// Bucket size at tier k is `TIER_BASE.pow(k)` raw samples.
pub const TIER_BASE: u32 = 8;

/// Number of buckets per tile, at every tier.
pub const TILE_SIZE_BUCKETS: u32 = 1024;

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
    let bucket_size = TIER_BASE.pow(tier);
    let start = tile_index
        .saturating_mul(TILE_SIZE_BUCKETS)
        .saturating_mul(bucket_size);
    decimate_tile_pure(samples, bucket_size, start)
}

/// An all-NaN tile (every bucket empty) — returned for an absent channel.
pub fn empty_tile() -> Vec<f64> {
    vec![f64::NAN; (TILE_SIZE_BUCKETS as usize) * 2]
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

}
