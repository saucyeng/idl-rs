//! Binary tile encoder for the chart-tile IPC endpoint (C3 §3.5, v2 layout,
//! ledger R25). Three little-endian regions after a fixed 32-byte header: a
//! **sample region** (`decimate_channel`'s bucket min/max pairs), a
//! **column region** (coarser per-pixel-column min/max/mean, so hover reads
//! never need IPC — design §6), and a **column time region** (each column's
//! first-sample `t_us`, exact — never synthesized from `nominal_rate_hz`;
//! C1 §2/§3.1, "time is recorded, not assumed").

use crate::chart_decimation::{column_stats, column_times_us, decimate_channel};

/// Encodes one chart tile as C3 §3.5's v2 binary layout.
///
/// Takes an already-materialized `(samples, t_us)` window — this is the
/// **reference encoder**, used by tests and any caller that already holds a
/// materialized slice. It does **not** replace
/// `SessionHandle::decimate_tile`, which remains the non-materializing
/// production path for a channel's sample region (master design's "no f64
/// window is ever materialized" principle, `handle.rs`). Extending that
/// non-materializing approach to the column and column-time regions is a
/// recorded deferral, not built this task.
///
/// `samples.len() == t_us.len()` is the caller's invariant — every sample
/// carries a recorded time (C1 §2).
///
/// `tier` is narrowed to the header's `u16` field via
/// `tier.min(u16::MAX as u32) as u16` — safe under
/// `chart_decimation::MAX_TIER = 10`, which fits a `u16` with room to spare.
pub fn build_tile_bytes(
    samples: &[f64],
    t_us: &[i64],
    tier: u32,
    tile_index: u32,
    column_count: u32,
) -> Vec<u8> {
    let sample_pairs = decimate_channel(samples, tier, tile_index);
    let sample_count = (sample_pairs.len() / 2) as u32;
    let columns = column_stats(samples, tier, tile_index, column_count);
    let times = column_times_us(t_us, tier, tile_index, column_count);

    let total = 32
        + sample_count as usize * 8
        + column_count as usize * 12
        + column_count as usize * 8;
    let mut out = Vec::with_capacity(total);

    // Header — 32 bytes, C3 §3.5's field table.
    out.extend_from_slice(b"IDLT");
    out.extend_from_slice(&2u16.to_le_bytes()); // version
    out.extend_from_slice(&(tier.min(u16::MAX as u32) as u16).to_le_bytes());
    out.extend_from_slice(&tile_index.to_le_bytes());
    out.extend_from_slice(&sample_count.to_le_bytes());
    out.extend_from_slice(&column_count.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags — reserved, 0 in this contract
    out.extend_from_slice(&[0u8; 8]); // reserved

    // Sample region — sample_count * 8 bytes (interleaved f32 min/max).
    for v in &sample_pairs {
        out.extend_from_slice(&(*v as f32).to_le_bytes());
    }

    // Column region — column_count * 12 bytes (f32 min, max, mean).
    for &(mn, mx, mean) in &columns {
        out.extend_from_slice(&mn.to_le_bytes());
        out.extend_from_slice(&mx.to_le_bytes());
        out.extend_from_slice(&mean.to_le_bytes());
    }

    // Column time region — column_count * 8 bytes (i64 t_us, µs).
    for &t in &times {
        out.extend_from_slice(&t.to_le_bytes());
    }

    debug_assert_eq!(out.len(), total);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart_decimation::TILE_SIZE_BUCKETS;

    /// Reads the header and all three regions of `bytes` back into their
    /// component values — the in-test decoder mirroring what L5/TS will do.
    fn decode(bytes: &[u8]) -> (u16, u16, u32, u32, u32, Vec<(f32, f32)>, Vec<(f32, f32, f32)>, Vec<i64>) {
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let tier = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let tile_index = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let sample_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let column_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());

        let sample_off = 32usize;
        let mut samples = Vec::with_capacity(sample_count as usize);
        for i in 0..sample_count as usize {
            let base = sample_off + i * 8;
            let mn = f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let mx = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            samples.push((mn, mx));
        }

        let column_off = sample_off + sample_count as usize * 8;
        let mut columns = Vec::with_capacity(column_count as usize);
        for j in 0..column_count as usize {
            let base = column_off + j * 12;
            let mn = f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let mx = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            let mean = f32::from_le_bytes(bytes[base + 8..base + 12].try_into().unwrap());
            columns.push((mn, mx, mean));
        }

        let time_off = column_off + column_count as usize * 12;
        let mut times = Vec::with_capacity(column_count as usize);
        for j in 0..column_count as usize {
            let base = time_off + j * 8;
            times.push(i64::from_le_bytes(bytes[base..base + 8].try_into().unwrap()));
        }

        (version, tier, tile_index, sample_count, column_count, samples, columns, times)
    }

    #[test]
    fn build_tile_bytes_header_matches_c3_3_5_field_table_version_is_2() {
        // Arrange — samples.len() == t_us.len() == 0 keeps this a header-only
        // check (decimate_channel still fills TILE_SIZE_BUCKETS regardless).
        let samples: Vec<f64> = Vec::new();
        let t_us: Vec<i64> = Vec::new();

        // Act
        let bytes = build_tile_bytes(&samples, &t_us, 3, 5, 7);

        // Assert — byte-sliced, not just total length.
        assert_eq!(&bytes[0..4], b"IDLT");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), TILE_SIZE_BUCKETS);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0);
        assert_eq!(&bytes[24..32], &[0u8; 8]);
    }

    #[test]
    fn offset_formula_matches_c3_3_5_worked_example_pure_arithmetic() {
        // Arrange — L3-R30: the worked example's "512 samples" pins the
        // offset arithmetic, not a real `sample_count` this plan ever emits.
        let sample_count: u64 = 512;
        let column_count: u64 = 256;

        // Act
        let sample_region_offset = 32u64;
        let column_region_offset = sample_region_offset + sample_count * 8;
        let column_time_region_offset = column_region_offset + column_count * 12;
        let total = column_time_region_offset + column_count * 8;

        // Assert — C3 §3.5's worked example, verbatim.
        assert_eq!(column_region_offset, 4128);
        assert_eq!(column_time_region_offset, 7200);
        assert_eq!(total, 9248);
    }

    #[test]
    fn build_tile_bytes_real_output_length_matches_fixed_1024_formula() {
        // Arrange — decimate_channel always fills TILE_SIZE_BUCKETS = 1024
        // today (right-edge NaN-padded), so sample_count is always 1024.
        let samples: Vec<f64> = (0..2000).map(|i| i as f64).collect();
        let t_us: Vec<i64> = (0..2000).map(|i| i * 1000).collect();
        let column_count = 10u32;

        // Act
        let bytes = build_tile_bytes(&samples, &t_us, 0, 0, column_count);

        // Assert
        let expected = 32 + (TILE_SIZE_BUCKETS as usize) * 8 + column_count as usize * 12 + column_count as usize * 8;
        assert_eq!(bytes.len(), expected);
    }

    #[test]
    fn build_tile_bytes_total_length_matches_formula_for_several_combinations() {
        // Arrange
        let samples: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let t_us: Vec<i64> = (0..5000).map(|i| i * 1000).collect();

        for &(tier, tile_index, column_count) in &[(0u32, 0u32, 0u32), (1, 0, 16), (2, 1, 200), (5, 3, 1)] {
            // Act
            let bytes = build_tile_bytes(&samples, &t_us, tier, tile_index, column_count);

            // Assert
            let expected = 32
                + (TILE_SIZE_BUCKETS as usize) * 8
                + column_count as usize * 12
                + column_count as usize * 8;
            assert_eq!(bytes.len(), expected, "tier {tier} tile {tile_index} columns {column_count}");
        }
    }

    #[test]
    fn build_tile_bytes_round_trip_decode_matches_independent_computation() {
        // Arrange
        let samples: Vec<f64> = (0..3000).map(|i| (i as f64 * 0.37).sin() * 100.0).collect();
        let t_us: Vec<i64> = (0..3000).map(|i| i * 2500).collect();
        let tier = 2u32;
        let tile_index = 0u32;
        let column_count = 32u32;

        // Act
        let bytes = build_tile_bytes(&samples, &t_us, tier, tile_index, column_count);
        let (version, decoded_tier, decoded_tile, sample_count, decoded_columns, decoded_samples, decoded_stats, decoded_times) =
            decode(&bytes);

        // Assert — every value matches the independent reference computation.
        assert_eq!(version, 2);
        assert_eq!(decoded_tier, tier as u16);
        assert_eq!(decoded_tile, tile_index);
        assert_eq!(sample_count, TILE_SIZE_BUCKETS);
        assert_eq!(decoded_columns, column_count);

        let want_samples = decimate_channel(&samples, tier, tile_index);
        for (i, (mn, mx)) in decoded_samples.iter().enumerate() {
            let (wmn, wmx) = (want_samples[i * 2] as f32, want_samples[i * 2 + 1] as f32);
            assert!((mn.is_nan() && wmn.is_nan()) || *mn == wmn, "sample {i} min");
            assert!((mx.is_nan() && wmx.is_nan()) || *mx == wmx, "sample {i} max");
        }

        let want_stats = column_stats(&samples, tier, tile_index, column_count);
        for (i, got) in decoded_stats.iter().enumerate() {
            let want = want_stats[i];
            assert!((got.0.is_nan() && want.0.is_nan()) || got.0 == want.0, "column {i} min");
            assert!((got.1.is_nan() && want.1.is_nan()) || got.1 == want.1, "column {i} max");
            assert!((got.2.is_nan() && want.2.is_nan()) || got.2 == want.2, "column {i} mean");
        }

        let want_times = column_times_us(&t_us, tier, tile_index, column_count);
        assert_eq!(decoded_times, want_times);
    }

    #[test]
    fn build_tile_bytes_short_channel_degenerate_columns_still_full_length() {
        // Arrange — a 3-sample channel at tier 0; with column_count = 20 over
        // a 1024-sample tile span, most columns are past-end.
        let samples = vec![1.0, 2.0, 3.0];
        let t_us = vec![0i64, 1000, 2000];
        let column_count = 20u32;

        // Act
        let bytes = build_tile_bytes(&samples, &t_us, 0, 0, column_count);
        let (_, _, _, _, _, _, decoded_stats, decoded_times) = decode(&bytes);

        // Assert — column region is mostly (NaN, NaN, NaN), column-time
        // region is mostly i64::MIN, and the tile length is still exactly
        // the formula's value (no truncation).
        let nan_count = decoded_stats.iter().filter(|c| c.0.is_nan() && c.1.is_nan() && c.2.is_nan()).count();
        assert!(nan_count > column_count as usize / 2, "expected most columns past-end");
        let sentinel_count = decoded_times.iter().filter(|&&t| t == i64::MIN).count();
        assert!(sentinel_count > column_count as usize / 2, "expected most column-times sentinel");

        let expected = 32 + (TILE_SIZE_BUCKETS as usize) * 8 + column_count as usize * 12 + column_count as usize * 8;
        assert_eq!(bytes.len(), expected);
    }
}
