//! Cursor readout (C3 §3.7, ledger R31): the value of a set of channels at
//! one instant, for the hover/scrub crosshair. JSON, not a heavy-array
//! binary path (design §3's listing is superseded by the signed C3 §3.7 —
//! one line stating this so it is not re-opened at review, G12.6).

use crate::session::handle::nearest_at_t_us;

/// Reads `t_us` (µs) against each `(channel_id, t_us, samples)` triple and
/// returns one nearest-sample value per channel, **in request order** — a
/// duplicate channel id in the input yields two entries in the output; that
/// is the caller's problem, not this function's. Folding the result to C3
/// §3.7's `Record<string, number | null>`, supplying the `t_us` echo, and
/// existence-checking an unknown channel id (C3's `invalid_argument` +
/// `detail.channel`) are all L5's job — this function never sees an unknown
/// channel and returns no error of its own.
///
/// Per channel: `None` when the channel has no samples or no time axis
/// (`t_us.len().min(samples.len()) == 0` — an axis-less result, e.g. a
/// scalar definition, has `t_us` empty and `samples` non-empty, L3-R12) or
/// when `t_us` falls outside the channel's own recorded span, i.e.
/// `t_us < first || t_us > last` (R31 — a channel that ends early must read
/// `null` past its last sample, not freeze at that value). Inside the span,
/// the nearest recorded sample is returned exactly, ties resolving to the
/// earlier sample (delegated to [`nearest_at_t_us`] — one nearest-sample
/// rule in the crate, not re-derived here).
pub fn cursor_readout(
    channels: &[(&str, &[i64], &[f64])],
    t_us: i64,
) -> Vec<(String, Option<f64>)> {
    channels
        .iter()
        .map(|&(id, ch_t_us, samples)| {
            let n = samples.len().min(ch_t_us.len());
            let value = if n == 0 {
                None
            } else if t_us < ch_t_us[0] || t_us > ch_t_us[n - 1] {
                None
            } else {
                nearest_at_t_us(samples, ch_t_us, t_us)
            };
            (id.to_string(), value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_readout_t_us_matches_a_recorded_sample_exactly() {
        // Arrange
        let t_us = [0_i64, 1_000_000, 2_000_000];
        let samples = [1.0, 2.0, 3.0];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 1_000_000);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), Some(2.0))]);
    }

    #[test]
    fn cursor_readout_t_us_between_samples_ties_to_earlier() {
        // Arrange
        let t_us = [0_i64, 1_000_000];
        let samples = [1.0, 2.0];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 500_000);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), Some(1.0))]);
    }

    #[test]
    fn cursor_readout_t_us_before_first_sample_is_none() {
        // Arrange — R31: outside the recorded span is null, not clamped.
        let t_us = [1_000_000_i64, 2_000_000];
        let samples = [1.0, 2.0];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 0);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), None)]);
    }

    #[test]
    fn cursor_readout_t_us_after_last_sample_is_none() {
        // Arrange — R31: a channel that ends early reads null past its end,
        // not a frozen last value.
        let t_us = [0_i64, 1_000_000];
        let samples = [1.0, 2.0];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 5_000_000);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), None)]);
    }

    #[test]
    fn cursor_readout_empty_channel_is_none() {
        // Arrange — no samples, no time axis.
        let t_us: [i64; 0] = [];
        let samples: [f64; 0] = [];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 0);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), None)]);
    }

    #[test]
    fn cursor_readout_multiple_channels_one_empty_others_populated() {
        // Arrange
        let t_a: [i64; 0] = [];
        let s_a: [f64; 0] = [];
        let t_b = [0_i64, 1_000_000];
        let s_b = [10.0, 20.0];
        let channels = [("a", &t_a[..], &s_a[..]), ("b", &t_b[..], &s_b[..])];

        // Act
        let out = cursor_readout(&channels, 1_000_000);

        // Assert — request order preserved, one None and one populated.
        assert_eq!(
            out,
            vec![("a".to_string(), None), ("b".to_string(), Some(20.0))]
        );
    }

    #[test]
    fn cursor_readout_axis_less_channel_is_none() {
        // Arrange — L3-R12: a scalar/rate-0 result has an empty t_us with
        // non-empty values; emptiness is decided by min(t_us.len(),
        // v.len()) == 0, never by is_nan() (G12.3).
        let t_us: [i64; 0] = [];
        let samples = [1.0, 2.0, 3.0];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 0);

        // Assert
        assert_eq!(out, vec![("ch".to_string(), None)]);
    }

    #[test]
    fn cursor_readout_nan_sample_at_nearest_index_is_some_nan() {
        // Arrange — a NaN sample is a real value, not absence (G12.2).
        let t_us = [0_i64, 1_000_000];
        let samples = [1.0, f64::NAN];
        let channels = [("ch", &t_us[..], &samples[..])];

        // Act
        let out = cursor_readout(&channels, 1_000_000);

        // Assert
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "ch");
        assert!(out[0].1.is_some_and(|v| v.is_nan()));
    }
}
