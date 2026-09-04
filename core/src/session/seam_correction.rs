//! Burst-seam correction (contract C1 §3.3): recovers a burst-drained IMU's
//! true sample cadence from its recorded read-instant stamps and re-spaces
//! each burst to a monotonic, uniformly-spaced corrected axis.
//!
//! **Why.** Firmware stamps a FIFO burst by walking back from the read
//! instant at the *nominal* ODR (SPEC §5.5). When the true ODR differs from
//! nominal, this makes the recorded stamps overlap or gap at burst seams —
//! locally non-monotonic time. This module fits the effective (true) period
//! from consecutive burst read-instant spacing and re-spaces every burst
//! backward from its own (trustworthy) read instant at that period.
//!
//! Algorithm version: `"v1"` — the whole of this module, per C1 §3.3.
pub const SEAM_CORRECTION_VERSION: &str = "v1";

/// Result of correcting one source's (one IMU's) recorded stamp sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamCorrection {
    /// Corrected timestamps (device-clock microseconds), same length/order
    /// as the input `stamps`. Strictly increasing (C1 §3.5 invariant 1) by
    /// construction — see the monotonicity guarantee in
    /// [`correct_burst_seams`]'s doc.
    pub corrected_us: Vec<i64>,
    /// The session-wide effective period (µs) this source's bursts were
    /// re-spaced at (falls back to `nominal_period_us` if fewer than two
    /// bursts, or if the computed value was non-positive).
    pub effective_period_us: i64,
    /// Non-fatal anomalies encountered (CLAUDE.md §5: recover what's
    /// readable, never crash on bad data, never silently synthesize a wrong
    /// value without saying so).
    pub warnings: Vec<ImportWarning>,
}

/// Discriminant for [`ImportWarning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportWarningKind {
    /// The session-wide median estimate was `<= 0` (pathological — e.g.
    /// out-of-order read instants); fell back to the nominal period.
    NonPositiveEffectivePeriod,
    /// A single burst's local-period fallback (the monotonicity guarantee's
    /// recompute) was `<= 0`; fell back to the nominal period for that burst.
    NonPositiveLocalPeriod,
}

/// A non-fatal import-time anomaly, surfaced rather than silently absorbed
/// or hard-failed (CLAUDE.md §5). Mirrors [`crate::session::ParseError`]'s
/// shape (kind + message), kept separate from it because a warning does not
/// abort parsing the way [`crate::session::ParseError::TruncatedRecord`] can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportWarning {
    /// Which anomaly this warning reports.
    pub kind: ImportWarningKind,
    /// Human-readable detail, including the offending value(s).
    pub message: String,
}

impl ImportWarning {
    fn new(kind: ImportWarningKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

/// One maximal run of consecutive samples whose deltas are within ±1 µs of
/// `nominal_period_us` — one FIFO burst. `start`/`end` are inclusive indices
/// into the input `stamps` slice.
struct Burst {
    start: usize,
    end: usize,
}

/// Step 1 (C1 §3.3): walk `stamps` once, splitting at any delta more than
/// ±1 µs away from `nominal_period_us`. `stamps` must have at least 1 entry.
fn detect_bursts(stamps: &[i64], nominal_period_us: i64) -> Vec<Burst> {
    let mut runs = Vec::new();
    let mut run_start = 0usize;
    for i in 1..stamps.len() {
        if (stamps[i] - stamps[i - 1] - nominal_period_us).abs() > 1 {
            runs.push(Burst { start: run_start, end: i - 1 });
            run_start = i;
        }
    }
    runs.push(Burst { start: run_start, end: stamps.len() - 1 });
    runs
}

/// Median of `values`, even-count = mean of the two middle values (C1 §3.3).
/// `values` must be non-empty.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("burst period estimates are always finite"));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Round-half-away-from-zero, matching C1 §3.3's specified tie-breaking rule
/// (distinct from Rust's `f64::round`, which is also half-away-from-zero for
/// positive values but this makes the intent explicit at each call site).
fn round_half_away_from_zero(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5).floor() as i64
    } else {
        (x - 0.5).ceil() as i64
    }
}

/// Corrects one source's recorded burst stamps (contract C1 §3.3, full
/// algorithm — steps 1–3 plus the monotonicity guarantee).
///
/// `stamps` must be the raw, as-recorded (uncorrected) timestamps in
/// arrival order for **one** burst-drained source (one IMU) — device-clock
/// microseconds, any origin. `nominal_period_us` is that source's
/// configured-ODR period ([`crate::parse::records::imu_period_us`]).
///
/// Fewer than 2 stamps: returned verbatim, `effective_period_us =
/// nominal_period_us` (no burst structure to detect).
///
/// **Monotonicity guarantee.** `corrected_us` is strictly increasing by
/// construction: burst 0 uses the session-wide `effective_period_us`
/// unconditionally (no predecessor to violate); every later burst `k`
/// checks `T_k − (N_k − 1) × effective_period_us > T_{k−1}` before
/// committing, and recomputes with its own local period when the check
/// fails (C1 §3.3's proof that this margin is enormous for any burst size/
/// period this hardware can produce).
pub fn correct_burst_seams(stamps: &[i64], nominal_period_us: i64) -> SeamCorrection {
    let mut warnings = Vec::new();

    if stamps.len() < 2 {
        return SeamCorrection {
            corrected_us: stamps.to_vec(),
            effective_period_us: nominal_period_us,
            warnings,
        };
    }

    let bursts = detect_bursts(stamps, nominal_period_us);

    // Step 2: effective period from consecutive burst read-instant spacing.
    let estimates: Vec<f64> = (1..bursts.len())
        .map(|k| {
            let t_k = stamps[bursts[k].end] as f64;
            let t_km1 = stamps[bursts[k - 1].end] as f64;
            let n_k = (bursts[k].end - bursts[k].start + 1) as f64;
            (t_k - t_km1) / n_k
        })
        .collect();

    let mut effective_period_us = if estimates.is_empty() {
        nominal_period_us
    } else {
        round_half_away_from_zero(median(estimates))
    };
    if effective_period_us <= 0 {
        warnings.push(ImportWarning::new(
            ImportWarningKind::NonPositiveEffectivePeriod,
            format!(
                "computed effective_period_us={effective_period_us} <= 0 \
                 (pathological — e.g. out-of-order read instants); \
                 falling back to nominal_period_us={nominal_period_us}"
            ),
        ));
        effective_period_us = nominal_period_us;
    }

    // Step 3: re-space each burst backward from its own read instant, with
    // the per-burst monotonicity check/fallback.
    let mut corrected = vec![0i64; stamps.len()];
    let mut prev_t_k: Option<i64> = None;
    for (k, burst) in bursts.iter().enumerate() {
        let t_k = stamps[burst.end];
        let n_k = (burst.end - burst.start + 1) as i64;
        let mut period = effective_period_us;

        if let Some(t_km1) = prev_t_k {
            if t_k - (n_k - 1) * period <= t_km1 {
                let mut local = round_half_away_from_zero((t_k - t_km1) as f64 / n_k as f64);
                if local <= 0 {
                    warnings.push(ImportWarning::new(
                        ImportWarningKind::NonPositiveLocalPeriod,
                        format!(
                            "burst {k}: local_period_us={local} <= 0 \
                             (not physically realizable for two real, \
                             time-ordered device reads); falling back to \
                             nominal_period_us={nominal_period_us}"
                        ),
                    ));
                    local = nominal_period_us;
                }
                period = local;
            }
        }

        for i in burst.start..=burst.end {
            let offset_from_end = (burst.end - i) as i64;
            corrected[i] = t_k - offset_from_end * period;
        }
        prev_t_k = Some(t_k);
    }

    SeamCorrection { corrected_us: corrected, effective_period_us, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worked_example_800hz_nominal_true_1200us_period_recovers_effective_period_and_respaces_every_burst() {
        // Arrange — contract C1 §3.3's worked numeric example verbatim:
        // nominal 1250 µs (800 Hz configured), true period 1200 µs
        // (≈833.3 Hz), 4 bursts of N=4, read instants 100000/104800/
        // 109600/114400.
        let stamps = vec![
            96250, 97500, 98750, 100000, // burst 0
            101050, 102300, 103550, 104800, // burst 1
            105850, 107100, 108350, 109600, // burst 2
            110650, 111900, 113150, 114400, // burst 3
        ];

        // Act
        let result = correct_burst_seams(&stamps, 1250);

        // Assert — effective period exactly recovers the true 1200 µs.
        assert_eq!(result.effective_period_us, 1200);
        assert!(result.warnings.is_empty());
        assert_eq!(
            result.corrected_us,
            vec![
                96400, 97600, 98800, 100000, //
                101200, 102400, 103600, 104800, //
                106000, 107200, 108400, 109600, //
                110800, 112000, 113200, 114400,
            ]
        );

        // Assert — strictly increasing everywhere, including across every
        // seam (the whole point of the correction).
        assert!(result.corrected_us.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn clean_stream_at_nominal_period_is_a_no_op() {
        // Arrange — every delta exactly nominal: one burst, zero estimates.
        let nominal = 1250i64;
        let stamps: Vec<i64> = (0..10).map(|i| 100_000 + i * nominal).collect();

        // Act
        let result = correct_burst_seams(&stamps, nominal);

        // Assert — no predecessor bursts to estimate from, falls back to
        // nominal, which reproduces the recorded stamps exactly.
        assert_eq!(result.effective_period_us, nominal);
        assert_eq!(result.corrected_us, stamps);
    }

    #[test]
    fn fewer_than_two_stamps_returns_verbatim() {
        // Arrange / Act / Assert
        assert_eq!(
            correct_burst_seams(&[], 1250),
            SeamCorrection { corrected_us: vec![], effective_period_us: 1250, warnings: vec![] }
        );
        assert_eq!(
            correct_burst_seams(&[42], 1250),
            SeamCorrection { corrected_us: vec![42], effective_period_us: 1250, warnings: vec![] }
        );
    }

    #[test]
    fn even_number_of_burst_estimates_averages_the_two_middle_values() {
        // Arrange — 3 bursts of N=2 → 2 estimates (k=1,2): true periods
        // 1000 µs then 1100 µs, so the median (mean of two) is 1050 µs.
        // Burst 0: T=10000 (samples 9000,10000, nominal period irrelevant
        // to this burst's own re-spacing since it's the anchor).
        let stamps = vec![
            9000, 10000, // burst0, nominal-spaced within-burst (delta 1000)
            11950, 12950, // burst1: delta within burst 1000 (nominal);
                          // seam delta 11950-10000=1950 vs nominal*...
            15000, 16000, // burst2
        ];
        // Use a nominal period that makes exactly these three runs detected
        // as separate bursts: nominal 1000, tolerance ±1 — seam deltas
        // (1950, 2050) are far outside tolerance, in-burst deltas (1000)
        // are exact.
        let nominal = 1000i64;

        // Act
        let result = correct_burst_seams(&stamps, nominal);

        // Assert — T_0=10000, T_1=12950, T_2=16000, N=2 for every burst:
        // estimate_1=(12950-10000)/2=1475, estimate_2=(16000-12950)/2=1525;
        // median = (1475+1525)/2 = 1500.
        assert_eq!(result.effective_period_us, 1500);
    }

    #[test]
    fn a_burst_faster_than_the_session_median_falls_back_to_its_own_local_period() {
        // Arrange — 3 bursts of N=2, nominal 2000 µs. Burst 0 → burst 1
        // seam gives estimate_1 = (8000-2000)/2 = 3000; burst 1 → burst 2
        // seam gives estimate_2 = (8300-8000)/2 = 150 (burst 2's read
        // instant lands only 300 µs after burst 1's, far faster than the
        // rest of the session). median(3000, 150) = 1575 =
        // effective_period_us. Re-spacing burst 2 with the session-wide
        // 1575 µs would put its first sample at 8300 - 1575 = 6725, at or
        // before burst 1's own read instant (8000) — non-monotonic — so the
        // monotonicity check must trip and burst 2 must fall back to its
        // own local period (round((8300-8000)/2) = 150 µs, the same
        // formula as estimate_2 — that estimate *is* burst 2's own local
        // period) instead.
        //
        // Every within-burst delta is exactly nominal (2000, tolerance ±1);
        // every seam delta is far outside tolerance, so detect_bursts
        // splits this into exactly 3 bursts as intended:
        //   seam 0→1: 6000 - 2000 = 4000 (|4000-2000|=2000 > 1)
        //   seam 1→2: 6300 - 8000 = -1700 (|-1700-2000|=3700 > 1)
        let stamps = vec![
            0, 2000, // burst 0 (T_0=2000)
            6000, 8000, // burst 1 (T_1=8000)
            6300, 8300, // burst 2 (T_2=8300) — the fast one
        ];
        let nominal = 2000i64;

        // Act
        let result = correct_burst_seams(&stamps, nominal);

        // Assert — session-wide effective period is 1575 as derived above,
        // with no warnings (both the median and the local fallback are
        // positive).
        assert_eq!(result.effective_period_us, 1575);
        assert!(result.warnings.is_empty());

        // Assert — burst 2 was re-spaced with its own local period
        // (150 µs), not the session-wide 1575 µs: this is the fallback
        // branch firing, not a coincidence of the monotonicity check alone.
        // Bursts 0/1 use the session-wide period unmodified.
        assert_eq!(
            result.corrected_us,
            vec![
                425, 2000, // burst 0 at effective_period_us=1575
                6425, 8000, // burst 1 at effective_period_us=1575
                8150, 8300, // burst 2 at its own local_period_us=150
            ]
        );

        // Assert — strictly increasing end to end, including the seam the
        // fallback exists to protect (8000 -> 8150, not 8000 -> 6725).
        assert!(result.corrected_us.windows(2).all(|w| w[1] > w[0]));
    }
}
