//! Non-device (FIT/GPX) time mapping — contract C1 §3.4. FIT/GPX sources
//! have no burst structure: a recorded timestamp maps to `t` directly, no
//! correction. Exposed here so L2's importers share one implementation
//! rather than each hand-rolling the rounding/dedup rules.

use crate::session::ParseError;

/// Maps a UTC-millisecond timestamp to session-relative microseconds:
/// `round((utc_ms - first_utc_ms) * 1000)`. `first_utc_ms` is the session's
/// earliest converted UTC millisecond value across all channels (C1 §3.4) —
/// the caller computes it once, up front, from every channel's timestamps.
pub fn utc_ms_to_t_us(utc_ms: i64, first_utc_ms: i64) -> i64 {
    (utc_ms - first_utc_ms) * 1000
}

/// FIT `timestamp` (u32 seconds since the FIT epoch, 1989-12-31T00:00:00Z
/// UTC) to Unix-epoch UTC milliseconds (C1 §3.4: `(fit_timestamp_s +
/// 631065600) * 1000`).
pub fn fit_timestamp_s_to_utc_ms(fit_timestamp_s: u32) -> i64 {
    (fit_timestamp_s as i64 + 631_065_600) * 1000
}

/// Drops a later duplicate/non-monotonic sample from `(t_us, value)` pairs
/// already computed by [`utc_ms_to_t_us`] (or any other source), enforcing
/// C1 §3.5 invariant 1 (strictly increasing `t_us`) the way C1 §3.4 requires
/// for FIT/GPX: **drop the later duplicate**, never emit a non-increasing
/// `t_us`. Returns the filtered `(t_us, value)` pairs plus a warning when
/// anything was dropped (CLAUDE.md §5 — never silently drop without saying
/// so).
pub fn drop_non_monotonic(mut pairs: Vec<(i64, f64)>) -> (Vec<(i64, f64)>, Option<ParseError>) {
    let mut dropped = 0usize;
    let mut out: Vec<(i64, f64)> = Vec::with_capacity(pairs.len());
    pairs.sort_by_key(|&(t, _)| t); // input order is already chronological in practice; sort defends against any caller that isn't
    for pair in pairs {
        match out.last() {
            Some(&(last_t, _)) if pair.0 <= last_t => dropped += 1,
            _ => out.push(pair),
        }
    }
    let warning = (dropped > 0).then(|| {
        ParseError::TruncatedRecord(format!(
            "{dropped} non-monotonic timestamp(s) dropped during FIT/GPX import (C1 §3.4)"
        ))
    });
    (out, warning)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_ms_to_t_us_scales_and_offsets() {
        // Arrange / Act / Assert
        assert_eq!(utc_ms_to_t_us(1_000, 1_000), 0);
        assert_eq!(utc_ms_to_t_us(1_500, 1_000), 500_000);
    }

    #[test]
    fn fit_epoch_conversion_matches_the_631065600_offset() {
        // Arrange / Act / Assert — FIT epoch 0 == 1989-12-31T00:00:00Z ==
        // Unix ms 631065600000.
        assert_eq!(fit_timestamp_s_to_utc_ms(0), 631_065_600_000);
    }

    #[test]
    fn drop_non_monotonic_keeps_first_of_a_duplicate_pair_and_warns() {
        // Arrange
        let pairs = vec![(0, 1.0), (1000, 2.0), (1000, 3.0), (2000, 4.0)];

        // Act
        let (out, warning) = drop_non_monotonic(pairs);

        // Assert
        assert_eq!(out, vec![(0, 1.0), (1000, 2.0), (2000, 4.0)]);
        assert!(warning.is_some());
    }

    #[test]
    fn drop_non_monotonic_strictly_increasing_input_is_untouched_and_silent() {
        // Arrange
        let pairs = vec![(0, 1.0), (1000, 2.0), (2000, 3.0)];

        // Act
        let (out, warning) = drop_non_monotonic(pairs.clone());

        // Assert
        assert_eq!(out, pairs);
        assert!(warning.is_none());
    }
}
