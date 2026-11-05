//! Cleaning a wheel-pulse event channel: dropping the extra edges a marginal
//! magnetic signal adds, without ever dropping a real pulse.
//!
//! Found on the Beacon ride of 2026-11-04: 1.5 % of front-wheel pulses at
//! every speed, on road and trail alike, are a second edge a few
//! milliseconds after a real one (about 0.16 of the way to the next pickup).
//! It is tied to wheel position, not time, so a firmware debounce in
//! microseconds does not catch it, and a lockout as a fraction of the
//! *previous* interval would drop the first real pulses after a locked wheel
//! spins up again (the previous interval is then the lock). The rule here
//! compares a gap with the gaps on both sides instead. Pure: times in, kept
//! indices out.

/// Default for [`deglitch`]'s `max_fraction`. The extra edges seen sit at
/// 0.10–0.17 of their neighbouring gaps; real gaps on a wheel whose pickups
/// alternate in spacing sit at 0.7–1.4 of their neighbours.
pub const DEFAULT_MAX_FRACTION: f64 = 0.35;

/// The indices of `t_us` (a pulse channel's recorded times, µs, ascending)
/// to keep. Pulse `i` is dropped when the gap ending at it is under
/// `max_fraction` of **both** the last kept gap before it and the gap after
/// it, so it reads as an extra edge inside one ordinary gap.
///
/// Both sides, because one side alone fails at the edges of motion:
///
/// - a wheel spinning up after a lock-up has a huge gap before its first
///   real pulse, but the gaps after it only shrink, so that pulse is never
///   short against the gap after it;
/// - under hard braking the gaps grow by a few tens of percent per pulse,
///   nowhere near `max_fraction`.
///
/// The first and last pulses are always kept.
pub fn deglitch(t_us: &[i64], max_fraction: f64) -> Vec<usize> {
    let n = t_us.len();
    if n < 3 {
        return (0..n).collect();
    }
    let gap = |a: usize, b: usize| (t_us[b] - t_us[a]) as f64;

    let mut kept = vec![0usize];
    for i in 1..n - 1 {
        let prev = *kept.last().expect("the first pulse is always kept");
        let this = gap(prev, i);
        let after = gap(i, i + 1);
        let before = kept.len().checked_sub(2).map(|k| gap(kept[k], prev));
        let extra = before.is_some_and(|b| this < max_fraction * b) && this < max_fraction * after;
        if !extra {
            kept.push(i);
        }
    }
    kept.push(n - 1);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pulses every `step_us` from 0, `n` of them.
    fn regular(step_us: i64, n: i64) -> Vec<i64> {
        (0..n).map(|k| k * step_us).collect()
    }

    #[test]
    fn deglitch_an_extra_edge_4_ms_after_a_pulse_is_dropped() {
        // Arrange — the Beacon road cruise: ~29 ms gaps, an edge 4 ms after one.
        let t = vec![0, 39_300, 69_900, 108_400, 141_100, 187_900, 217_000, 221_000, 261_100, 292_800, 337_800];

        // Act
        let kept = deglitch(&t, DEFAULT_MAX_FRACTION);

        // Assert
        assert_eq!(kept, vec![0, 1, 2, 3, 4, 5, 6, 8, 9, 10]);
    }

    #[test]
    fn deglitch_pickups_alternating_short_and_long_keeps_every_pulse() {
        // Arrange — gaps 18, 26, 18, 26 ... ms.
        let mut t = vec![0i64];
        for k in 0..20 {
            t.push(t.last().unwrap() + if k % 2 == 0 { 18_000 } else { 26_000 });
        }

        // Act
        let kept = deglitch(&t, DEFAULT_MAX_FRACTION);

        // Assert
        assert_eq!(kept.len(), t.len());
    }

    #[test]
    fn deglitch_a_wheel_spinning_up_after_a_lock_keeps_every_pulse() {
        // Arrange — locked for 2 s, then gaps shrinking 300, 200, 150, 120, 100 ms.
        let t = vec![0, 100_000, 2_100_000, 2_400_000, 2_600_000, 2_750_000, 2_870_000, 2_970_000, 3_070_000];

        // Act
        let kept = deglitch(&t, DEFAULT_MAX_FRACTION);

        // Assert
        assert_eq!(kept, (0..t.len()).collect::<Vec<_>>());
    }

    #[test]
    fn deglitch_steady_braking_keeps_every_pulse() {
        // Arrange — gaps growing 10 % per pulse, as under hard braking.
        let mut t = vec![0i64];
        let mut g = 20_000.0;
        for _ in 0..30 {
            t.push(t.last().unwrap() + g as i64);
            g *= 1.1;
        }

        // Act
        let kept = deglitch(&t, DEFAULT_MAX_FRACTION);

        // Assert
        assert_eq!(kept.len(), t.len());
    }

    #[test]
    fn deglitch_regular_pulses_all_kept_and_fewer_than_three_untouched() {
        // Act
        let regular_kept = deglitch(&regular(100_000, 20), DEFAULT_MAX_FRACTION);
        let tiny = deglitch(&[0, 5], DEFAULT_MAX_FRACTION);

        // Assert
        assert_eq!(regular_kept.len(), 20);
        assert_eq!(tiny, vec![0, 1]);
    }
}
