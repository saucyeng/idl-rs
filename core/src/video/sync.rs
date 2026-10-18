//! Video↔session sync-offset estimation. See IDL0_SPEC §33.3 and video spec
//! §4.3; ported from idl0 unchanged in behaviour (Isaac, 2026-10-16: "just
//! use the architecture from idl0 for now").
//!
//! Contract (C1 §6): `session_time_s = video_time_s + offset_s`, where
//! `session_time_s` is the session's `*_time_secs` clock (t = 0 at its first
//! sample). Estimation prefers the GPMF UTC anchor, then the container
//! creation time; with neither it is idl0's `Parse` error, which the
//! linking code (`store::video`) turns into a manual offset of 0. A video
//! whose span misses the session is refused.
//!
//! The session's clock is [`SessionClock`]: `GPS_EpochMs` samples paired
//! with their own recorded times, extrapolated beyond the GPS span rather
//! than clamped. idl0 used `SessionHandle::epoch_ms_to_time_secs_extrapolated`,
//! which maps through sample index / nominal rate (idl0's uniform grid) and,
//! on the lazy handle idl1 opens sessions with, silently falls back to the
//! session origin. idl1 keeps every sample's recorded time (design §3), so
//! the clock is read from that instead.

use std::path::Path;

use crate::store::parquet::{read_channel, read_time_axis_bounds, ParquetStoreErrorKind};
use crate::video::gpmf::VideoTelemetry;
use crate::video::mp4box::Mp4Info;
use crate::video::{OverlapSpans, SyncMethod, VideoError};

/// One session's mapping from wall-clock UTC to its `*_time_secs` clock.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionClock {
    /// `(UTC ms, session s)` pairs from `GPS_EpochMs`, strictly increasing
    /// in UTC. Empty when the session has no usable GPS time.
    gps: Vec<(f64, f64)>,
    /// The session's effective start, UTC ms (C1 §6 `effective_start_ms`).
    start_utc_ms: i64,
    /// The session's length on its own clock, s.
    duration_s: f64,
}

/// How far a GPS fix's offset (`epoch_ms − recorded_ms`) may sit from the
/// session's median offset and still count, ms.
const GLITCH_TOLERANCE_MS: f64 = 2_000.0;

/// Seconds of session per ms of UTC when no GPS interval gives a slope:
/// wall-clock 1:1, which is what `GPS_EpochMs` means anyway.
const WALL_CLOCK_SLOPE_S_PER_MS: f64 = 0.001;

impl SessionClock {
    /// A clock from `GPS_EpochMs` values (`gps_epoch_ms`, UTC ms) and their
    /// recorded times (`gps_t_s`, session s), index-aligned. `start_utc_ms`
    /// is the session's effective start (UTC ms); `duration_s` its length
    /// (s).
    ///
    /// Glitch filter (lead ruling, 2026-10-18): zero (pre-lock) and
    /// non-finite epochs are dropped; then a pair is kept only if its
    /// offset `epoch_ms − recorded_ms` is within [`GLITCH_TOLERANCE_MS`] of
    /// the median offset, so one wild epoch removes only itself rather than
    /// every fix after it; of exact-duplicate epochs the first is kept.
    pub fn new(start_utc_ms: i64, duration_s: f64, gps_epoch_ms: &[f64], gps_t_s: &[f64]) -> Self {
        let candidates: Vec<(f64, f64)> = gps_epoch_ms
            .iter()
            .zip(gps_t_s)
            .map(|(&e, &t)| (e, t))
            .filter(|&(e, t)| e.is_finite() && e > 0.0 && t.is_finite())
            .collect();
        let mut offsets: Vec<f64> = candidates.iter().map(|&(e, t)| e - t * 1000.0).collect();
        offsets.sort_by(f64::total_cmp);
        let mut gps: Vec<(f64, f64)> = Vec::with_capacity(candidates.len());
        if let Some(&median) = offsets.get(offsets.len() / 2) {
            for (e, t) in candidates {
                if (e - t * 1000.0 - median).abs() > GLITCH_TOLERANCE_MS {
                    continue;
                }
                if gps.last().is_some_and(|&(last_e, _)| e == last_e) {
                    continue;
                }
                gps.push((e, t));
            }
        }
        debug_assert!(gps.windows(2).all(|w| w[0].0 < w[1].0), "kept GPS epochs must increase");
        SessionClock { gps, start_utc_ms, duration_s }
    }

    /// Reads a session's clock from `<session_dir>/data.parquet`: its
    /// `GPS_EpochMs` channel (absent is not an error) and its `t` axis
    /// bounds. `start_utc_ms` is the session's effective start, UTC ms.
    pub fn read(session_dir: &Path, start_utc_ms: i64) -> Result<Self, VideoError> {
        let io = |e: crate::store::parquet::ParquetStoreError| VideoError::io(format!("reading {}: {}", session_dir.display(), e.message));
        let duration_s = match read_time_axis_bounds(session_dir).map_err(io)? {
            Some((_, hi_us)) => hi_us as f64 / 1e6,
            None => 0.0,
        };
        let (epochs, times) = match read_channel(session_dir, "GPS_EpochMs") {
            Ok(c) => {
                let times: Vec<f64> = c.t_us.iter().map(|&t| t as f64 / 1e6).collect();
                (c.materialize(), times)
            }
            Err(e) if e.kind == ParquetStoreErrorKind::NotFound => (Vec::new(), Vec::new()),
            Err(e) => return Err(io(e)),
        };
        Ok(SessionClock::new(start_utc_ms, duration_s, &epochs, &times))
    }

    /// The session's effective start, UTC ms.
    pub fn start_utc_ms(&self) -> i64 {
        self.start_utc_ms
    }

    /// The session's length on its own clock, s.
    pub fn duration_s(&self) -> f64 {
        self.duration_s
    }

    /// [`Self::epoch_ms_to_session_s`] over many instants, input order
    /// kept — the shape of idl0's `epoch_ms_to_time_secs_extrapolated`.
    pub fn epochs_ms_to_session_s(&self, epochs_ms: &[f64]) -> Vec<f64> {
        epochs_ms.iter().map(|&e| self.epoch_ms_to_session_s(e)).collect()
    }

    /// Maps a UTC instant (ms) to session seconds. Inside the GPS span it
    /// interpolates between the bracketing fixes; outside it extrapolates
    /// with the nearest interval's slope, so an instant before the session
    /// is negative and one after it is past the end. Without GPS it is
    /// `(epoch_ms - start_utc_ms) / 1000`.
    pub fn epoch_ms_to_session_s(&self, epoch_ms: f64) -> f64 {
        let gps = &self.gps;
        match gps.len() {
            0 => (epoch_ms - self.start_utc_ms as f64) * WALL_CLOCK_SLOPE_S_PER_MS,
            1 => gps[0].1 + (epoch_ms - gps[0].0) * WALL_CLOCK_SLOPE_S_PER_MS,
            n => {
                let slope = |a: (f64, f64), b: (f64, f64)| {
                    let s = (b.1 - a.1) / (b.0 - a.0);
                    if s.is_finite() && s > 0.0 {
                        s
                    } else {
                        WALL_CLOCK_SLOPE_S_PER_MS
                    }
                };
                if epoch_ms <= gps[0].0 {
                    return gps[0].1 + (epoch_ms - gps[0].0) * slope(gps[0], gps[1]);
                }
                if epoch_ms >= gps[n - 1].0 {
                    return gps[n - 1].1 + (epoch_ms - gps[n - 1].0) * slope(gps[n - 2], gps[n - 1]);
                }
                // First fix strictly after `epoch_ms`; its predecessor is at
                // or before it.
                let hi = gps.partition_point(|&(e, _)| e <= epoch_ms);
                let (a, b) = (gps[hi - 1], gps[hi]);
                a.1 + (epoch_ms - a.0) * (b.1 - a.1) / (b.0 - a.0)
            }
        }
    }
}

/// An estimated sync offset.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncEstimate {
    /// `session_time_s = video_time_s + offset_s`, in seconds.
    pub offset_s: f64,
    /// 0.9 for `gpmf`, 0.3 for `creation_time`, 0.0 for a manual fallback.
    pub confidence: f64,
    /// Which anchor the offset came from.
    pub method: SyncMethod,
}

/// Confidence reported for a GPMF-anchored estimate.
const CONFIDENCE_GPMF: f64 = 0.9;
/// Confidence reported for a creation-time estimate.
const CONFIDENCE_CREATION_TIME: f64 = 0.3;

/// The video's own time anchor: `(video time s, UTC ms)` from GPMF, else
/// `(0, creation time)` from the container, with the method it came from.
/// `None` when the file has neither.
fn anchor(telemetry: Option<&VideoTelemetry>, info: &Mp4Info) -> Option<(f64, i64, SyncMethod)> {
    if let Some((t_video_s, epoch_ms)) = telemetry.and_then(|t| t.utc_anchor) {
        return Some((t_video_s, epoch_ms, SyncMethod::Gpmf));
    }
    info.creation_time_utc_ms.map(|ms| (0.0, ms, SyncMethod::CreationTime))
}

/// The video's span in UTC ms, `(first frame, end)`, from the same anchor
/// [`estimate_sync`] would use. `None` when the file has no time anchor.
/// Used to match a file to sessions before any one session is chosen.
pub fn video_span_utc_ms(telemetry: Option<&VideoTelemetry>, info: &Mp4Info) -> Option<(i64, i64)> {
    let (t_video_s, epoch_ms, _) = anchor(telemetry, info)?;
    let start = epoch_ms - (t_video_s * 1000.0).round() as i64;
    Some((start, start + (info.duration_s * 1000.0).round() as i64))
}

/// Estimate the video↔session offset. Prefers the GPMF UTC anchor
/// (`telemetry`), falling back to the container `creation_time` in `info`.
///
/// Errors: `Parse` when neither anchor exists (idl0's behaviour; the caller
/// links such a file as manual at 0); [`crate::video::VideoErrorKind::NoOverlap`]
/// (detail: both spans, UTC ms) when the mapped video span misses the
/// session. The anchor is
/// mapped through [`SessionClock::epoch_ms_to_session_s`], which
/// extrapolates: an anchor from outside the session must stay outside it,
/// or footage from a different run saturates to the session edge and passes
/// the overlap check with a fabricated offset.
pub fn estimate_sync(telemetry: Option<&VideoTelemetry>, info: &Mp4Info, clock: &SessionClock) -> Result<SyncEstimate, VideoError> {
    let Some((t_video_s, epoch_ms, method)) = anchor(telemetry, info) else {
        return Err(VideoError::parse(
            "no sync anchor: video has neither GPMF UTC nor a container creation time; \
             pass a manual --offset",
        ));
    };
    let session_s = clock.epoch_ms_to_session_s(epoch_ms as f64);
    let offset_s = session_s - t_video_s;
    let confidence = match method {
        SyncMethod::Gpmf => CONFIDENCE_GPMF,
        _ => CONFIDENCE_CREATION_TIME,
    };

    // Overlap check: video span mapped onto session time vs session span.
    let session_len_s = clock.duration_s();
    let video_start_s = offset_s;
    let video_end_s = offset_s + info.duration_s;
    if video_end_s < 0.0 || video_start_s > session_len_s {
        // An anchored video always has a span (the anchor exists).
        let (video_start_utc_ms, video_end_utc_ms) = video_span_utc_ms(telemetry, info).unwrap_or((epoch_ms, epoch_ms));
        let session_start_utc_ms = clock.start_utc_ms();
        return Err(VideoError::no_overlap(
            format!(
                "video maps to session seconds [{video_start_s:.1}, {video_end_s:.1}] but the \
                 session spans [0.0, {session_len_s:.1}]: no overlap; check the camera's clock or \
                 set the offset by hand"
            ),
            OverlapSpans {
                video_start_utc_ms,
                video_end_utc_ms,
                session_start_utc_ms: Some(session_start_utc_ms),
                session_end_utc_ms: Some(session_start_utc_ms + (session_len_s * 1000.0).round() as i64),
            },
        ));
    }

    Ok(SyncEstimate { offset_s, confidence, method })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::gpmf::VideoTelemetry;
    use crate::video::{VideoErrorDetail, VideoErrorKind};

    /// A session starting at `start_utc_ms`, `len_s` long, with no GPS time:
    /// the clock is the session origin, which is exact for these tests.
    fn clock_with_origin(start_utc_ms: i64, len_s: f64) -> SessionClock {
        SessionClock::new(start_utc_ms, len_s, &[], &[])
    }

    /// A session whose clock comes from a `GPS_EpochMs` channel: one fix per
    /// second starting at `first_epoch_ms`, recorded at t = 0, 1, 2… s.
    fn clock_with_gps(first_epoch_ms: i64, len_s: usize) -> SessionClock {
        let epochs: Vec<f64> = (0..=len_s).map(|i| (first_epoch_ms + i as i64 * 1000) as f64).collect();
        let times: Vec<f64> = (0..=len_s).map(|i| i as f64).collect();
        SessionClock::new(first_epoch_ms, len_s as f64, &epochs, &times)
    }

    fn info(duration_s: f64, creation_ms: Option<i64>) -> Mp4Info {
        Mp4Info {
            width: 1920,
            height: 1080,
            fps: 30.0,
            duration_s,
            creation_time_utc_ms: creation_ms,
            has_gpmd: false,
            video_codec: "h264".to_string(),
        }
    }

    #[test]
    fn estimate_sync_gpmf_utc_anchor_maps_offset_through_session_clock() {
        // Arrange — session t=0 ↔ 1_717_770_600_000 ms; camera says video
        // t=2.0 s ↔ 25 s into the session → offset 23.0.
        let clock = clock_with_origin(1_717_770_600_000, 120.0);
        let telem = VideoTelemetry {
            utc_anchor: Some((2.0, 1_717_770_625_000)),
            fixes: vec![],
        };

        // Act
        let est = estimate_sync(Some(&telem), &info(60.0, None), &clock).unwrap();

        // Assert
        assert!((est.offset_s - 23.0).abs() < 1e-6);
        assert_eq!(est.method, SyncMethod::Gpmf);
        assert!((est.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn estimate_sync_creation_time_fallback_reports_low_confidence() {
        // Arrange — creation 30 s after session start, no telemetry.
        let clock = clock_with_origin(1_000_000, 120.0);
        let inf = info(60.0, Some(1_000_000 + 30_000));

        // Act
        let est = estimate_sync(None, &inf, &clock).unwrap();

        // Assert
        assert!((est.offset_s - 30.0).abs() < 1e-6);
        assert_eq!(est.method, SyncMethod::CreationTime);
        assert!((est.confidence - 0.3).abs() < 1e-9);
    }

    #[test]
    fn estimate_sync_telemetry_without_anchor_falls_back_to_creation_time() {
        // Arrange — GPMF present but GPS-empty (indoor start).
        let clock = clock_with_origin(1_000_000, 120.0);
        let telem = VideoTelemetry::default();

        // Act
        let est = estimate_sync(Some(&telem), &info(60.0, Some(1_010_000)), &clock).unwrap();

        // Assert
        assert_eq!(est.method, SyncMethod::CreationTime);
        assert!((est.offset_s - 10.0).abs() < 1e-6);
    }

    #[test]
    fn estimate_sync_video_before_session_is_no_overlap_error_listing_ranges() {
        // Arrange — video ends 340 s before the session starts.
        let clock = clock_with_origin(1_000_000, 121.0);
        let inf = info(60.0, Some(1_000_000 - 400_000));

        // Act
        let err = estimate_sync(None, &inf, &clock).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NoOverlap);
        assert!(err.message.contains("[-400.0, -340.0]"), "video range: {}", err.message);
        assert!(err.message.contains("[0.0, 121.0]"), "session range: {}", err.message);
        assert_eq!(
            err.detail,
            VideoErrorDetail::Spans(OverlapSpans {
                video_start_utc_ms: 600_000,
                video_end_utc_ms: 660_000,
                session_start_utc_ms: Some(1_000_000),
                session_end_utc_ms: Some(1_121_000),
            })
        );
    }

    #[test]
    fn estimate_sync_gps_clock_video_starting_before_session_yields_negative_offset() {
        // Arrange — GPS-clocked session; the camera rolled 5 s before the
        // logger did, so the true offset is -5.0 (the first 5 s of video
        // precede the session). Real-footage regression from idl0: the GPS
        // mapping used to clamp below-range anchors to 0.0 and lose the
        // lead-in.
        let clock = clock_with_gps(1_784_128_799_000, 113);
        let inf = info(115.0, Some(1_784_128_794_000));

        // Act
        let est = estimate_sync(None, &inf, &clock).unwrap();

        // Assert
        assert!((est.offset_s - -5.0).abs() < 1e-6, "offset was {}", est.offset_s);
    }

    #[test]
    fn estimate_sync_gps_clock_video_hours_after_session_is_no_overlap_error() {
        // Arrange — 129 s session; video shot ~2.9 h later (a different run
        // from the same day). Real-footage regression from idl0: the
        // above-range clamp saturated the anchor to the session end, so this
        // silently passed the overlap check with a fabricated offset.
        let clock = clock_with_gps(1_784_118_096_000, 129);
        let inf = info(115.0, Some(1_784_128_794_000));

        // Act
        let err = estimate_sync(None, &inf, &clock).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NoOverlap);
    }

    #[test]
    fn estimate_sync_gps_clock_anchor_inside_span_is_unchanged_by_extrapolation() {
        // Arrange — anchor 40 s into the GPS span stays exactly 40 s.
        let clock = clock_with_gps(1_000_000_000_000, 120);
        let inf = info(30.0, Some(1_000_000_040_000));

        // Act
        let est = estimate_sync(None, &inf, &clock).unwrap();

        // Assert
        assert!((est.offset_s - 40.0).abs() < 1e-6, "offset {}", est.offset_s);
    }

    #[test]
    fn estimate_sync_no_anchor_and_no_creation_time_is_parse_error() {
        // Arrange
        let clock = clock_with_origin(1_000_000, 120.0);

        // Act
        let err = estimate_sync(None, &info(60.0, None), &clock).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::Parse);
        assert!(err.message.contains("--offset"));
    }

    // ---- idl0 handle.rs `epoch_ms_to_time_secs_extrapolated` tests, ported
    // to SessionClock (idl-rs tag `idl0-video-export`, handle.rs:1467–1559).

    /// `n` GPS fixes `step_ms` apart from `first_ms`, recorded at the
    /// uniform times idl0's index / rate gave them (`i / rate_hz`).
    fn clock_from_uniform_gps(start_utc_ms: i64, first_ms: f64, step_ms: f64, n: usize, rate_hz: f64) -> SessionClock {
        let epochs: Vec<f64> = (0..n).map(|i| first_ms + i as f64 * step_ms).collect();
        let times: Vec<f64> = (0..n).map(|i| i as f64 / rate_hz).collect();
        SessionClock::new(start_utc_ms, (n - 1) as f64 / rate_hz, &epochs, &times)
    }

    #[test]
    fn epoch_ms_to_session_s_maps_below_span_to_negative_seconds() {
        // Arrange — 1 Hz GPS starting at epoch 10_000 ms.
        let clock = clock_from_uniform_gps(10_000, 10_000.0, 1000.0, 5, 1.0);

        // Act — 3 s before the first fix.
        let out = clock.epochs_ms_to_session_s(&[7_000.0]);

        // Assert — clamping would give 0.0; the true mapping is -3 s.
        assert!((out[0] - -3.0).abs() < 1e-9, "got {}", out[0]);
    }

    #[test]
    fn epoch_ms_to_session_s_maps_above_span_past_the_end() {
        // Arrange — 1 Hz GPS, 5 fixes → span ends at t = 4 s.
        let clock = clock_from_uniform_gps(10_000, 10_000.0, 1000.0, 5, 1.0);

        // Act — 6 s past the last fix (last fix epoch 14_000 → t = 4).
        let out = clock.epochs_ms_to_session_s(&[20_000.0]);

        // Assert — clamping would give 4.0; the true mapping is 10 s.
        assert!((out[0] - 10.0).abs() < 1e-9, "got {}", out[0]);
    }

    #[test]
    fn epoch_ms_to_session_s_inside_the_span_interpolates_like_the_clamped_mapping() {
        // Arrange — 10 Hz GPS from epoch 1000 ms. idl0 compared against its
        // clamped twin, which idl1's SessionClock has no need of; the values
        // that twin gives are written out instead.
        let clock = clock_from_uniform_gps(1000, 1000.0, 100.0, 10, 10.0);

        // Act
        let out = clock.epochs_ms_to_session_s(&[1000.0, 1250.0, 1900.0]);

        // Assert
        for (got, want) in out.iter().zip([0.0, 0.25, 0.9]) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn epoch_ms_to_session_s_falls_back_to_the_session_origin_without_gps() {
        // Arrange — no GPS_EpochMs; origin = timestamp_utc_ms = 2000.
        let clock = SessionClock::new(2000, 1.0, &[], &[]);

        // Act — (5000 - 2000)/1000 = 3.0 s.
        let out = clock.epochs_ms_to_session_s(&[5000.0, 2000.0]);

        // Assert
        assert!((out[0] - 3.0).abs() < 1e-9);
        assert!((out[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn epoch_ms_to_session_s_empty_input_is_empty() {
        // Arrange
        let clock = SessionClock::new(0, 0.0, &[], &[]);

        // Act + Assert
        assert!(clock.epochs_ms_to_session_s(&[]).is_empty());
    }

    #[test]
    fn session_clock_uses_recorded_times_not_sample_index() {
        // Arrange — GPS fixes at 1 Hz in UTC, but the logger recorded them
        // at t = 0, 2, 4 s (a clock running at half speed): the recorded
        // time is the session's clock, not the fix index.
        let clock = SessionClock::new(0, 4.0, &[1000.0, 2000.0, 3000.0], &[0.0, 2.0, 4.0]);

        // Act
        let mid = clock.epoch_ms_to_session_s(1500.0);
        let after = clock.epoch_ms_to_session_s(4000.0);
        let before = clock.epoch_ms_to_session_s(500.0);

        // Assert
        assert!((mid - 1.0).abs() < 1e-9);
        assert!((after - 6.0).abs() < 1e-9, "extrapolated with the last slope: {after}");
        assert!((before - -1.0).abs() < 1e-9, "extrapolated with the first slope: {before}");
    }

    #[test]
    fn session_clock_drops_pre_lock_zeros_and_a_repeated_epoch() {
        // Arrange — the logger writes 0 before GPS lock; a repeated epoch
        // must not divide by zero.
        let clock = SessionClock::new(0, 10.0, &[0.0, 0.0, 5000.0, 5000.0, 6000.0], &[0.0, 1.0, 2.0, 3.0, 4.0]);

        // Act
        let t = clock.epoch_ms_to_session_s(5500.0);

        // Assert
        assert!((t - 3.0).abs() < 1e-9, "interpolated between (5000, 2) and (6000, 4): {t}");
    }

    /// 1 Hz fixes from epoch 100_000 ms, recorded at t = 0, 1, 2… s, with
    /// `glitch` overriding some epochs.
    fn ride(n: usize, glitch: &[(usize, f64)]) -> (Vec<f64>, Vec<f64>) {
        let mut epochs: Vec<f64> = (0..n).map(|i| 100_000.0 + i as f64 * 1000.0).collect();
        for &(i, e) in glitch {
            epochs[i] = e;
        }
        (epochs, (0..n).map(|i| i as f64).collect())
    }

    #[test]
    fn session_clock_a_far_future_spike_mid_ride_removes_only_that_fix() {
        // Arrange — fix 5 claims to be a year ahead.
        let (epochs, times) = ride(10, &[(5, 100_000.0 + 3.15e10)]);

        // Act
        let clock = SessionClock::new(0, 9.0, &epochs, &times);

        // Assert — nine fixes left, and the ones after the spike still map.
        assert_eq!(clock.gps.len(), 9);
        assert!((clock.epoch_ms_to_session_s(107_500.0) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn session_clock_a_single_backwards_glitch_removes_only_that_fix() {
        // Arrange — fix 4 jumps an hour back.
        let (epochs, times) = ride(10, &[(4, 104_000.0 - 3_600_000.0)]);

        // Act
        let clock = SessionClock::new(0, 9.0, &epochs, &times);

        // Assert
        assert_eq!(clock.gps.len(), 9);
        assert!((clock.epoch_ms_to_session_s(104_000.0) - 4.0).abs() < 1e-9, "interpolated across the gap");
    }

    #[test]
    fn session_clock_pre_lock_zeros_are_dropped() {
        // Arrange
        let (epochs, times) = ride(6, &[(0, 0.0), (1, 0.0)]);

        // Act
        let clock = SessionClock::new(0, 5.0, &epochs, &times);

        // Assert
        assert_eq!(clock.gps.len(), 4);
        assert_eq!(clock.gps[0], (102_000.0, 2.0));
    }

    #[test]
    fn session_clock_repeated_epochs_keep_the_first() {
        // Arrange — the receiver repeats fix 3's epoch at t = 4 (1 s late,
        // inside the tolerance).
        let (epochs, times) = ride(6, &[(4, 103_000.0)]);

        // Act
        let clock = SessionClock::new(0, 5.0, &epochs, &times);

        // Assert
        assert_eq!(clock.gps.iter().filter(|&&(e, _)| e == 103_000.0).collect::<Vec<_>>(), vec![&(103_000.0, 3.0)]);
        assert_eq!(clock.gps.len(), 5);
    }

    #[test]
    fn video_span_utc_ms_gpmf_anchor_backs_out_the_video_time() {
        // Arrange — GPS time 10_000 ms at video t = 2 s; 30 s long.
        let telem = VideoTelemetry { utc_anchor: Some((2.0, 10_000)), fixes: vec![] };

        // Act
        let span = video_span_utc_ms(Some(&telem), &info(30.0, Some(1)));

        // Assert
        assert_eq!(span, Some((8_000, 38_000)));
    }
}
