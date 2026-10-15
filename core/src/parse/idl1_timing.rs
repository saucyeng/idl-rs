//! IMU timing for `.idl1` files (DRAFT spec 2026-10-15
//! `docs/superpowers/specs/2026-10-15-idl1-file-format-DRAFT.md` §3).
//!
//! Pure functions, no I/O. Each LSM6DSV320X has one oscillator that drives
//! its sample rate and its timestamp counter, so:
//!
//! 1. **Sample chip times** come from the FIFO timestamp words: samples
//!    between two words are evenly spaced in chip ticks. A window that
//!    received fewer samples than its tick span implies lost samples; those
//!    are counted exactly and returned as a gap.
//! 2. **The chip → ESP32 clock map** is a robust linear fit over the
//!    `(read_chip_ts, read_us)` pairs the firmware records once per FIFO
//!    drain, per window, joined into one continuous piecewise-linear map so
//!    oscillator drift is followed without steps.
//!
//! Chip timestamps are `u32` ticks on the wire; [`unwrap_u32`] makes them a
//! monotonic `i64` first.

/// One `(chip time, ESP32 time)` pair from a drain: the chip's TIMESTAMP
/// register (unwrapped ticks) and `esp_timer` (µs) read together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockPair {
    /// Chip timestamp counter, unwrapped ticks.
    pub chip: i64,
    /// `esp_timer` at the same drain, µs since boot.
    pub esp_us: i64,
}

/// A FIFO timestamp word placed in a stream: it stamps the sample at
/// `sample_index` (0-based across the whole session, received samples only)
/// with chip time `chip` (unwrapped ticks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsMark {
    /// Index of the received sample this word stamps.
    pub sample_index: usize,
    /// Chip timestamp counter at that sample, unwrapped ticks.
    pub chip: i64,
}

/// Unwraps a sequence of `u32` tick counts into a monotonic `i64`, adding
/// 2³² at every wrap. A step backwards of less than half the range is taken
/// as a wrap only when it is larger than half the range in the other
/// direction; anything else (a small backstep) is kept as read.
pub fn unwrap_u32(values: &[u32]) -> Vec<i64> {
    let mut out = Vec::with_capacity(values.len());
    let mut offset: i64 = 0;
    let mut prev: Option<u32> = None;
    for &v in values {
        if let Some(p) = prev {
            if v < p && (p - v) > u32::MAX / 2 {
                offset += 1i64 << 32;
            }
        }
        out.push(offset + v as i64);
        prev = Some(v);
    }
    out
}

/// The median of `values` (mean of the two middle values for an even
/// count). `None` when empty. Sorts a copy; NaNs sort last.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// A robust line `esp = slope × chip + intercept` through `pairs`: the slope
/// is the median of slopes between pairs half the window apart (a linear-time
/// Theil–Sen variant), the intercept the median residual. `None` for fewer
/// than two pairs or no usable slope.
fn robust_line(pairs: &[ClockPair]) -> Option<(f64, f64)> {
    let n = pairs.len();
    if n < 2 {
        return None;
    }
    let h = (n / 2).max(1);
    let slopes: Vec<f64> = (0..n - h)
        .filter_map(|i| {
            let (a, b) = (pairs[i], pairs[i + h]);
            let dc = (b.chip - a.chip) as f64;
            (dc > 0.0).then(|| (b.esp_us - a.esp_us) as f64 / dc)
        })
        .collect();
    let slope = median(&slopes)?;
    let residuals: Vec<f64> = pairs.iter().map(|p| p.esp_us as f64 - slope * p.chip as f64).collect();
    Some((slope, median(&residuals)?))
}

/// A continuous chip → ESP32 time map: anchor points at each fit window's
/// median chip time, linear between them, extrapolated past the ends with the
/// first and last window's own slope.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockMap {
    /// `(chip ticks, esp µs)` anchors, strictly increasing in chip time.
    anchors: Vec<(f64, f64)>,
    /// Slope (µs per tick) below the first anchor.
    slope_first: f64,
    /// Slope (µs per tick) above the last anchor.
    slope_last: f64,
}

impl ClockMap {
    /// The ESP32 time (µs) at chip time `chip` (ticks).
    pub fn map(&self, chip: f64) -> f64 {
        let a = &self.anchors;
        if chip <= a[0].0 {
            return a[0].1 + (chip - a[0].0) * self.slope_first;
        }
        let last = a[a.len() - 1];
        if chip >= last.0 {
            return last.1 + (chip - last.0) * self.slope_last;
        }
        let i = a.partition_point(|p| p.0 <= chip);
        let (lo, hi) = (a[i - 1], a[i]);
        lo.1 + (chip - lo.0) * (hi.1 - lo.1) / (hi.0 - lo.0)
    }

    /// The map's mean slope, µs per chip tick: the tick length measured on
    /// the ESP32's clock.
    pub fn mean_slope(&self) -> f64 {
        let a = &self.anchors;
        if a.len() < 2 {
            return self.slope_first;
        }
        let (f, l) = (a[0], a[a.len() - 1]);
        (l.1 - f.1) / (l.0 - f.0)
    }
}

/// Fits a [`ClockMap`] to `pairs` (any order) in windows of `window_us` of
/// ESP32 time. A window needs at least four pairs to get its own anchor;
/// with too few pairs overall for any window, one line is fitted to them
/// all. `None` with fewer than two usable pairs.
pub fn fit_clock_map(pairs: &[ClockPair], window_us: i64) -> Option<ClockMap> {
    let mut sorted: Vec<ClockPair> = pairs.to_vec();
    sorted.sort_by_key(|p| p.esp_us);
    sorted.dedup_by_key(|p| p.chip);
    let (first, last) = (sorted.first()?, sorted.last()?);
    let span = last.esp_us - first.esp_us;
    let windows = if window_us <= 0 { 1 } else { ((span / window_us) + 1).max(1) as usize };

    let mut fits: Vec<(f64, f64, f64)> = Vec::new(); // (anchor chip, anchor esp, slope)
    for w in 0..windows {
        let lo = first.esp_us + w as i64 * window_us;
        let hi = if w + 1 == windows { i64::MAX } else { lo + window_us };
        let start = sorted.partition_point(|p| p.esp_us < lo);
        let end = sorted.partition_point(|p| p.esp_us < hi);
        let chunk = &sorted[start..end];
        if chunk.len() < 4 {
            continue;
        }
        if let Some((slope, intercept)) = robust_line(chunk) {
            let chips: Vec<f64> = chunk.iter().map(|p| p.chip as f64).collect();
            let c = median(&chips)?;
            fits.push((c, slope * c + intercept, slope));
        }
    }
    if fits.is_empty() {
        let (slope, intercept) = robust_line(&sorted)?;
        let chips: Vec<f64> = sorted.iter().map(|p| p.chip as f64).collect();
        let c = median(&chips)?;
        fits.push((c, slope * c + intercept, slope));
    }
    fits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    fits.dedup_by(|a, b| a.0 <= b.0);
    Some(ClockMap {
        anchors: fits.iter().map(|f| (f.0, f.1)).collect(),
        slope_first: fits[0].2,
        slope_last: fits[fits.len() - 1].2,
    })
}

/// Samples lost inside one window between two timestamp words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LostRun {
    /// Index (into the output with lost samples filled in) of the first
    /// filled slot.
    pub start: usize,
    /// How many samples were lost.
    pub len: usize,
}

/// Every sample's chip time, with lost samples given filled slots.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleTimes {
    /// Chip time (ticks) of each output slot, received and filled.
    pub chip: Vec<f64>,
    /// For each output slot, the received sample it holds, or `None` for a
    /// filled (lost) slot.
    pub source: Vec<Option<usize>>,
    /// The filled runs, in output-slot coordinates.
    pub lost: Vec<LostRun>,
    /// The sample period in chip ticks (median over the timestamp windows).
    pub period_ticks: f64,
}

/// Places `n` received samples in chip time from the timestamp `marks`
/// (sorted or not; marks past `n` are ignored). Between two marks the
/// samples are evenly spaced. The period is the median over the windows of
/// (tick span ÷ samples received), so a rare lossy window can't bias it. A
/// window whose tick span holds more samples than arrived lost the
/// difference; the lost slots go at the start of the window (an overrun
/// discards the oldest samples). Samples before the first and after the last
/// mark are placed at the period from the nearest mark.
///
/// `None` with fewer than two usable marks: there is no chip-time spacing to
/// place samples by.
pub fn place_samples(n: usize, marks: &[TsMark]) -> Option<SampleTimes> {
    let mut m: Vec<TsMark> = marks.iter().copied().filter(|t| t.sample_index < n).collect();
    m.sort_by_key(|t| t.sample_index);
    m.dedup_by_key(|t| t.sample_index);
    if m.len() < 2 {
        return None;
    }
    let periods: Vec<f64> = m
        .windows(2)
        .filter_map(|w| {
            let (ds, dc) = (w[1].sample_index - w[0].sample_index, w[1].chip - w[0].chip);
            (ds > 0 && dc > 0).then(|| dc as f64 / ds as f64)
        })
        .collect();
    let mut period = median(&periods)?;
    if periods.len() >= 3 {
        // Lossy windows read long; re-take the median over windows that
        // look clean against the first estimate.
        let clean: Vec<f64> = periods.iter().copied().filter(|p| *p < period * 1.5).collect();
        period = median(&clean).unwrap_or(period);
    }

    let mut chip = Vec::with_capacity(n);
    let mut source = Vec::with_capacity(n);
    let mut lost = Vec::new();

    // Before the first mark.
    let first = m[0];
    for k in 0..first.sample_index {
        chip.push(first.chip as f64 - (first.sample_index - k) as f64 * period);
        source.push(Some(k));
    }
    // Each window [mark i, mark i+1).
    for w in m.windows(2) {
        let (a, b) = (w[0], w[1]);
        let received = b.sample_index - a.sample_index;
        let span = (b.chip - a.chip) as f64;
        let expected = ((span / period).round() as usize).max(received);
        let missing = expected - received;
        let step = span / expected as f64;
        // Slot 0 is mark `a`'s own sample.
        chip.push(a.chip as f64);
        source.push(Some(a.sample_index));
        if missing > 0 {
            lost.push(LostRun { start: chip.len(), len: missing });
            for j in 1..=missing {
                chip.push(a.chip as f64 + j as f64 * step);
                source.push(None);
            }
        }
        for k in 1..received {
            chip.push(a.chip as f64 + (missing + k) as f64 * step);
            source.push(Some(a.sample_index + k));
        }
    }
    // The last mark and after.
    let last = m[m.len() - 1];
    for k in last.sample_index..n {
        chip.push(last.chip as f64 + (k - last.sample_index) as f64 * period);
        source.push(Some(k));
    }
    Some(SampleTimes { chip, source, lost, period_ticks: period })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_u32_wrap_adds_two_to_the_32() {
        // Arrange
        let ticks = [u32::MAX - 10, u32::MAX - 1, 5, 20];

        // Act
        let out = unwrap_u32(&ticks);

        // Assert
        assert_eq!(out, vec![u32::MAX as i64 - 10, u32::MAX as i64 - 1, (1i64 << 32) + 5, (1i64 << 32) + 20]);
    }

    #[test]
    fn unwrap_u32_small_backstep_is_not_a_wrap() {
        // Arrange
        let ticks = [1000, 990, 1010];

        // Act
        let out = unwrap_u32(&ticks);

        // Assert
        assert_eq!(out, vec![1000, 990, 1010]);
    }

    /// SplitMix64: a deterministic pseudo-random stream for test noise
    /// (no periodic structure for the lag-based slope estimate to alias on).
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Pairs from a chip whose tick is `tick_us` on the ESP32's clock, offset
    /// `offset_us`, every 20 ms for `seconds`, with uniform ±`jitter_us`
    /// bus-read noise. Chip times are whole ticks, as on the wire.
    fn pairs(tick_us: f64, offset_us: f64, seconds: i64, jitter_us: i64) -> Vec<ClockPair> {
        let mut rng = 42u64;
        (0..seconds * 50)
            .map(|i| {
                let esp = i * 20_000;
                let noise = (splitmix(&mut rng) % (2 * jitter_us as u64 + 1)) as i64 - jitter_us;
                let chip = ((esp as f64 - offset_us) / tick_us).round() as i64;
                ClockPair { chip, esp_us: esp + noise }
            })
            .collect()
    }

    #[test]
    fn fit_clock_map_noisy_pairs_recover_the_tick_and_offset() {
        // Arrange — tick 21.75 µs, 50 µs read jitter.
        let p = pairs(21.75, 3_000.0, 60, 50);

        // Act
        let map = fit_clock_map(&p, 10_000_000).unwrap();

        // Assert — within 15 µs anywhere in the session: chip times are whole
        // 21.75 µs ticks, so ±11 µs is the floor from quantization alone.
        for chip in [0.0, 100_000.0, 1_000_000.0, 2_700_000.0] {
            let truth = chip * 21.75 + 3_000.0;
            assert!((map.map(chip) - truth).abs() < 15.0, "chip {chip}: {} vs {truth}", map.map(chip));
        }
        assert!((map.mean_slope() - 21.75).abs() < 1e-4);
    }

    #[test]
    fn fit_clock_map_follows_a_tick_that_drifts_through_the_session() {
        // Arrange — the tick lengthens 30 ppm over two minutes (a warming
        // chip). One straight line would be off by ~450 µs mid-session.
        let mut p = Vec::new();
        let mut chip = 0.0_f64;
        for i in 0..6_000i64 {
            let tick = 21.75 * (1.0 + 30e-6 * i as f64 / 6_000.0);
            chip += 20_000.0 / tick;
            p.push(ClockPair { chip: chip.round() as i64, esp_us: (i + 1) * 20_000 });
        }

        // Act
        let map = fit_clock_map(&p, 10_000_000).unwrap();

        // Assert — the piecewise map stays within 15 µs of truth throughout
        // (whole-tick quantization is ±11 µs of that).
        for q in p.iter().step_by(250) {
            let err = map.map(q.chip as f64) - q.esp_us as f64;
            assert!(err.abs() < 15.0, "at {}: error {err} µs", q.esp_us);
        }
    }

    #[test]
    fn fit_clock_map_fewer_than_two_pairs_is_none() {
        // Act / Assert
        assert!(fit_clock_map(&[ClockPair { chip: 1, esp_us: 1 }], 10_000_000).is_none());
        assert!(fit_clock_map(&[], 10_000_000).is_none());
    }

    /// Marks every 32 received samples, at `period` ticks per sample, with
    /// `lost` samples missing just before the mark at index `lost_at`.
    fn marks(n: usize, period: i64, lost_at: Option<(usize, i64)>) -> Vec<TsMark> {
        (0..n)
            .step_by(32)
            .map(|s| {
                let extra = match lost_at {
                    Some((at, lost)) if s >= at => lost * period,
                    _ => 0,
                };
                TsMark { sample_index: s, chip: 1_000 + s as i64 * period + extra }
            })
            .collect()
    }

    #[test]
    fn place_samples_clean_stream_is_evenly_spaced_with_nothing_lost() {
        // Arrange
        let m = marks(160, 96, None);

        // Act
        let t = place_samples(150, &m).unwrap();

        // Assert
        assert_eq!(t.chip.len(), 150);
        assert!(t.lost.is_empty());
        assert_eq!(t.period_ticks, 96.0);
        for (i, c) in t.chip.iter().enumerate() {
            assert!((c - (1_000.0 + i as f64 * 96.0)).abs() < 1e-9, "slot {i}");
        }
        assert!(t.source.iter().enumerate().all(|(i, s)| *s == Some(i)));
    }

    #[test]
    fn place_samples_lossy_window_counts_the_lost_samples_exactly() {
        // Arrange — 3 samples lost in the window ending at received sample 64.
        let m = marks(128, 96, Some((64, 3)));

        // Act
        let t = place_samples(128, &m).unwrap();

        // Assert
        assert_eq!(t.lost, vec![LostRun { start: 33, len: 3 }]);
        assert_eq!(t.chip.len(), 131);
        assert_eq!(t.source.iter().filter(|s| s.is_none()).count(), 3);
        assert_eq!(t.period_ticks, 96.0);
        // The window's last received sample sits just before the next mark.
        let next_mark = t.source.iter().position(|s| *s == Some(64)).unwrap();
        assert!((t.chip[next_mark] - (1_000.0 + 67.0 * 96.0)).abs() < 1e-9);
    }

    #[test]
    fn place_samples_fewer_than_two_marks_is_none() {
        // Act / Assert
        assert!(place_samples(10, &[TsMark { sample_index: 0, chip: 5 }]).is_none());
    }
}
