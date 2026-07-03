//! Variance functions — variance_time and variance_dist.
//! See lap-delta-rewrite spec §4.2.

use crate::math::eval::ChannelLookup;
use crate::session::handle::SessionHandle;
use crate::track_projection::Projector;

/// Compares each main sample against an overlay lap's channel value at the
/// matching position on the overlay polyline.
///
/// For each main sample at position (E, N, heading), projects onto the
/// overlay reference polyline to recover `t_ref` (overlay-lap-relative time
/// in seconds), then linearly interpolates the overlay channel value at
/// the corresponding sample index. The returned vector is
/// `main_samples[i] - overlay_value_at(t_ref)`. NaN where projection fails
/// (heading mismatch, out of range), where the resolved sample index falls
/// outside the overlay channel, OR where the main sample time falls outside
/// the main lap window (`main_lap_start_sec`..`main_lap_end_sec`). Spec
/// §4.2: out-of-main-lap samples are NaN — the projector's running state
/// would otherwise lock onto wrong segments during paddock idle and inflate
/// the result to lap-time magnitudes.
///
/// `main_sample_rate_hz`: rate of `main_samples`, used to derive each
///   sample's session-relative time (`i / rate`) for the lap-window check.
/// `main_lap_start_sec` / `main_lap_end_sec`: main lap window in
///   session-relative seconds. When `start >= end` (sentinel — no main lap
///   designated) the gating is disabled and every sample is processed.
/// `main_positions`: (E_metres, N_metres, heading_rad) per main sample.
/// `overlay_reference`: (E_metres, N_metres, t_lap_seconds) per overlay
///   sample — `t_lap` is the time elapsed since the overlay lap started.
/// `overlay_sample_rate_hz`: rate of `overlay_channel` (Hz).
/// `overlay_lap_start_t_relative`: session-relative time of the overlay
///   lap's start, in seconds. Combined with `t_ref` it yields the
///   session-relative time used to index `overlay_channel`.
#[allow(clippy::too_many_arguments)]
pub fn variance_time(
    main_samples: &[f64],
    main_sample_rate_hz: f64,
    main_lap_start_sec: f64,
    main_lap_end_sec: f64,
    main_positions: &[(f64, f64, f64)],
    overlay_reference: Vec<(f64, f64, f64)>,
    overlay_channel: &[f64],
    overlay_sample_rate_hz: f64,
    overlay_lap_start_t_relative: f64,
) -> Vec<f64> {
    let mut projector = Projector::new(overlay_reference);
    let mut out = Vec::with_capacity(main_samples.len());
    let gate = main_lap_start_sec < main_lap_end_sec;
    for (i, &main_val) in main_samples.iter().enumerate() {
        if gate {
            let t = (i as f64) / main_sample_rate_hz;
            if t < main_lap_start_sec || t >= main_lap_end_sec {
                out.push(f64::NAN);
                continue;
            }
        }
        let (e, n, h) = main_positions[i];
        let proj = projector.project(e, n, h);
        match proj {
            None => out.push(f64::NAN),
            Some(r) => {
                // overlay channel sample index from t_ref + lap start
                let overlay_session_t = overlay_lap_start_t_relative + r.t_ref;
                let f = overlay_session_t * overlay_sample_rate_hz;
                let lo = f.floor() as isize;
                let frac = f - (lo as f64);
                if lo < 0 || (lo as usize) >= overlay_channel.len() {
                    out.push(f64::NAN);
                } else if frac == 0.0 {
                    out.push(main_val - overlay_channel[lo as usize]);
                } else {
                    let hi = lo as usize + 1;
                    if hi >= overlay_channel.len() {
                        out.push(f64::NAN);
                    } else {
                        let v = overlay_channel[lo as usize] * (1.0 - frac)
                            + overlay_channel[hi] * frac;
                        out.push(main_val - v);
                    }
                }
            }
        }
    }
    out
}

/// Compares each main sample against an overlay lap's channel value at the
/// matching arc length along the lap.
///
/// Linearly interpolates `overlay_samples` at the arc length
/// `main_arc_lengths[i]` and returns `main_samples[i] - overlay_value`.
/// NaN where the main arc length falls outside the overlay's range — laps
/// of different length truncate cleanly. NaN also where the main sample
/// time falls outside the main lap window (spec §4.2); when
/// `main_lap_start_sec >= main_lap_end_sec` the gating is disabled.
///
/// `main_sample_rate_hz`, `main_lap_start_sec`, `main_lap_end_sec`: see
/// `variance_time`.
/// `main_arc_lengths`, `overlay_arc_lengths`: cumulative distance from
/// each lap's start, in metres.
pub fn variance_dist(
    main_samples: &[f64],
    main_sample_rate_hz: f64,
    main_lap_start_sec: f64,
    main_lap_end_sec: f64,
    main_arc_lengths: &[f64],
    overlay_samples: &[f64],
    overlay_arc_lengths: &[f64],
) -> Vec<f64> {
    let n = main_samples.len();
    let mut out = Vec::with_capacity(n);
    if overlay_arc_lengths.len() < 2 || overlay_samples.len() < 2 {
        return vec![f64::NAN; n];
    }
    let first = overlay_arc_lengths[0];
    let last = overlay_arc_lengths[overlay_arc_lengths.len() - 1];
    let gate = main_lap_start_sec < main_lap_end_sec;
    let mut hint = 0;
    for i in 0..n {
        if gate {
            let t = (i as f64) / main_sample_rate_hz;
            if t < main_lap_start_sec || t >= main_lap_end_sec {
                out.push(f64::NAN);
                continue;
            }
        }
        let s = main_arc_lengths.get(i).copied().unwrap_or(f64::NAN);
        if s.is_nan() || s < first || s > last {
            out.push(f64::NAN);
            continue;
        }
        while hint < overlay_arc_lengths.len() - 2
            && overlay_arc_lengths[hint + 1] < s
        {
            hint += 1;
        }
        let d_lo = overlay_arc_lengths[hint];
        let d_hi = overlay_arc_lengths[hint + 1];
        if d_hi <= d_lo {
            out.push(f64::NAN);
            continue;
        }
        let frac = (s - d_lo) / (d_hi - d_lo);
        let v = overlay_samples[hint] * (1.0 - frac)
            + overlay_samples[hint + 1] * frac;
        out.push(main_samples[i] - v);
    }
    out
}

/// Returns the 1-based lap number containing the session-relative time `t`,
/// or `0` when `t` falls outside every lap window.
///
/// `laps`: `(start_s, end_s)` per lap, both expressed in session-relative
/// seconds. The end boundary is exclusive (`t < end`) so a sample landing
/// exactly on a lap boundary belongs to the next lap.
pub fn current_lap_at(laps: &[(f64, f64)], t: f64) -> u32 {
    for (i, (start, end)) in laps.iter().enumerate() {
        if t >= *start && t < *end {
            return (i + 1) as u32;
        }
    }
    0
}

/// Returns the session-relative start time (seconds) of lap `n` (1-based),
/// or `f64::NAN` for `n == 0` or out-of-range `n`.
pub fn lap_start_time(laps: &[(f64, f64)], n: u32) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let idx = (n as usize).saturating_sub(1);
    laps.get(idx).map(|(s, _)| *s).unwrap_or(f64::NAN)
}

/// Returns the 0-based sector index containing the session-relative time
/// `t`, or `u32::MAX` as an "outside any sector" sentinel. The Dart caller
/// maps the sentinel to NaN when wrapping the result for a math channel.
///
/// `sectors`: `(start_s, end_s)` per sector, session-relative seconds. End
/// boundary is exclusive (`t < end`).
pub fn sector_number_at(sectors: &[(f64, f64)], t: f64) -> u32 {
    for (i, (start, end)) in sectors.iter().enumerate() {
        if t >= *start && t < *end {
            return i as u32;
        }
    }
    u32::MAX
}

// ============================================================================
// N-lap variance batch (one reference vs many targets)
// ============================================================================

/// Alignment for [`variance_traces`].
pub enum VarianceMode {
    /// Lap-relative time (no Track required).
    Time,
    /// Track-distance (position projected onto the reference polyline).
    Distance,
}

/// One lap's session + window for [`variance_traces`]. `lap_*_ms` select the
/// lap's GPS by epoch; `lap_start_uniform_sec` is the lap start in uniform-time;
/// `window_*_sec` gate the target channel (set `start >= end` to disable gating,
/// the same sentinel used by [`variance_time`] / [`variance_dist`]).
pub struct LapRef<'a> {
    pub handle: &'a SessionHandle,
    pub lap_start_ms: f64,
    pub lap_end_ms: f64,
    pub lap_start_uniform_sec: f64,
    pub window_start_sec: f64,
    pub window_end_sec: f64,
}

/// One delta series per target: `target[channel] − reference[channel]` at the
/// matching position. The reference geometry (tangent-plane frame, arc lengths,
/// subsampled channel) is built ONCE and reused across every target. Series
/// order matches `targets`; each series is the target channel's length (NaN
/// outside its lap window or where the match fails, NaN-filled when a target
/// lacks GPS). Reuses the `variance_geom` helpers — no second variance
/// implementation. `reference` is the app's Main lap; each `target` is an
/// overlay lap (the engine/app "main"/"overlay" naming is inverted — see the
/// N-lap variance design §2).
pub fn variance_traces(
    reference: &LapRef,
    targets: &[LapRef],
    channel_id: &str,
    mode: VarianceMode,
) -> Vec<Vec<f64>> {
    use crate::math::variance_geom as g;

    // A target's NaN fallback is its channel length (so the series still aligns
    // to the chart's x-axis), or empty when the channel is absent.
    let nan_for = |t: &LapRef| -> Vec<f64> {
        t.handle.lookup(channel_id).map(|c| vec![f64::NAN; c.samples.len()]).unwrap_or_default()
    };

    // Build the reference frame + channel once.
    let Some(r) = g::build_overlay_reference(
        reference.handle,
        reference.lap_start_ms,
        reference.lap_end_ms,
        reference.lap_start_uniform_sec,
    ) else {
        return targets.iter().map(nan_for).collect();
    };
    let Some(ref_ch) = reference.handle.lookup(channel_id) else {
        return targets.iter().map(nan_for).collect();
    };

    // Distance mode reuses one reference arc + subsampled channel for all targets.
    let overlay_arc = g::cumulative_arc(&r.e, &r.n);
    let overlay_samples = g::subsample_to_arc(&ref_ch.samples, &overlay_arc);

    targets
        .iter()
        .map(|t| {
            let Some(tc) = t.handle.lookup(channel_id) else { return Vec::new() };
            let Some(main_pos) =
                g::build_main_positions(t.handle, r.lat0, r.lon0, r.lat_scale, r.lon_scale)
            else {
                return vec![f64::NAN; tc.samples.len()];
            };
            let win = (t.window_start_sec, t.window_end_sec);
            match mode {
                VarianceMode::Time => g::variance_time_against(
                    &r,
                    &ref_ch,
                    &main_pos,
                    &tc.samples,
                    tc.sample_rate_hz,
                    win,
                ),
                VarianceMode::Distance => g::variance_dist_against(
                    &overlay_arc,
                    &overlay_samples,
                    &main_pos,
                    &tc.samples,
                    tc.sample_rate_hz,
                    win,
                ),
            }
        })
        .collect()
}

// ============================================================================
// FFI-friendly call adapters
// ============================================================================
//
// The pure Rust functions above take `&[(f64, f64, f64)]` tuple slices, which
// the FFI bridge cannot pass tuple slices to Dart. The wrappers below accept
// parallel `Vec<f64>` arrays (one vector per tuple component), rebuild the
// tuple vectors, then delegate to the pure implementations.
//
// Dart callers (math_channel_evaluator) build the parallel arrays directly
// from `ChannelData` samples before invoking these adapters.

/// FFI adapter for `variance_time`.
///
/// Takes parallel arrays for the main-sample positions and the overlay
/// reference points; reassembles them into `(f64, f64, f64)` tuples and
/// delegates to `variance_time`. See the wrapped function for semantics
/// and return shape (length matches `main_samples`, NaN where projection
/// or interpolation fails).
///
/// All three position vectors (`main_positions_e/n/heading`) must share
/// the same length as `main_samples`. All three reference vectors
/// (`overlay_ref_e/n/t_lap`) must share the same length.
///
/// `main_positions_e`, `main_positions_n`: easting / northing in metres,
///   local tangent plane.
/// `main_positions_heading`: heading in radians (atan2(dN, dE)).
/// `overlay_ref_e`, `overlay_ref_n`: easting / northing in metres for
///   each overlay reference point.
/// `overlay_ref_t_lap`: overlay-lap-relative time in seconds for each
///   overlay reference point.
/// `overlay_channel`: overlay channel samples to interpolate against.
/// `overlay_sample_rate_hz`: rate of `overlay_channel` in Hz.
/// `overlay_lap_start_t_relative`: session-relative time of the overlay
///   lap's start, in seconds.
#[allow(clippy::too_many_arguments)]
pub fn variance_time_call(
    main_samples: Vec<f64>,
    main_sample_rate_hz: f64,
    main_lap_start_sec: f64,
    main_lap_end_sec: f64,
    main_positions_e: Vec<f64>,
    main_positions_n: Vec<f64>,
    main_positions_heading: Vec<f64>,
    overlay_ref_e: Vec<f64>,
    overlay_ref_n: Vec<f64>,
    overlay_ref_t_lap: Vec<f64>,
    overlay_channel: Vec<f64>,
    overlay_sample_rate_hz: f64,
    overlay_lap_start_t_relative: f64,
) -> Vec<f64> {
    let positions: Vec<(f64, f64, f64)> = main_positions_e
        .iter()
        .zip(main_positions_n.iter())
        .zip(main_positions_heading.iter())
        .map(|((e, n), h)| (*e, *n, *h))
        .collect();
    let reference: Vec<(f64, f64, f64)> = overlay_ref_e
        .iter()
        .zip(overlay_ref_n.iter())
        .zip(overlay_ref_t_lap.iter())
        .map(|((e, n), t)| (*e, *n, *t))
        .collect();
    variance_time(
        &main_samples,
        main_sample_rate_hz,
        main_lap_start_sec,
        main_lap_end_sec,
        &positions,
        reference,
        &overlay_channel,
        overlay_sample_rate_hz,
        overlay_lap_start_t_relative,
    )
}

/// FFI adapter for `variance_dist`.
///
/// `main_samples` and `main_arc_lengths` must share the same length;
/// `overlay_samples` and `overlay_arc_lengths` must share the same
/// length. Arc lengths are cumulative metres from each lap's start.
#[allow(clippy::too_many_arguments)]
pub fn variance_dist_call(
    main_samples: Vec<f64>,
    main_sample_rate_hz: f64,
    main_lap_start_sec: f64,
    main_lap_end_sec: f64,
    main_arc_lengths: Vec<f64>,
    overlay_samples: Vec<f64>,
    overlay_arc_lengths: Vec<f64>,
) -> Vec<f64> {
    variance_dist(
        &main_samples,
        main_sample_rate_hz,
        main_lap_start_sec,
        main_lap_end_sec,
        &main_arc_lengths,
        &overlay_samples,
        &overlay_arc_lengths,
    )
}

/// FFI adapter for `current_lap_at`.
///
/// Takes parallel `(start, end)` arrays for the lap windows (frb cannot
/// auto-bridge `&[(f64, f64)]`) and zips them into tuples before
/// delegating. `laps_starts` and `laps_ends` must share the same length;
/// both are expressed in session-relative seconds.
///
/// Returns the 1-based lap number containing `t`, or `0` when `t` falls
/// outside every window.
pub fn current_lap_at_call(
    laps_starts: Vec<f64>,
    laps_ends: Vec<f64>,
    t: f64,
) -> u32 {
    let laps: Vec<(f64, f64)> = laps_starts
        .iter()
        .zip(laps_ends.iter())
        .map(|(s, e)| (*s, *e))
        .collect();
    current_lap_at(&laps, t)
}

/// FFI adapter for `lap_start_time`.
///
/// `laps_starts` and `laps_ends` define the lap windows (session-relative
/// seconds) as parallel arrays. Returns the start of lap `n` (1-based)
/// in session-relative seconds, or `f64::NAN` if `n == 0` or out of range.
pub fn lap_start_time_call(
    laps_starts: Vec<f64>,
    laps_ends: Vec<f64>,
    n: u32,
) -> f64 {
    let laps: Vec<(f64, f64)> = laps_starts
        .iter()
        .zip(laps_ends.iter())
        .map(|(s, e)| (*s, *e))
        .collect();
    lap_start_time(&laps, n)
}

/// FFI adapter for `sector_number_at`.
///
/// `sectors_starts` and `sectors_ends` define the sector windows
/// (session-relative seconds) as parallel arrays. Returns the 0-based
/// sector index containing `t`, or `u32::MAX` when `t` falls outside
/// every sector — the Dart caller maps the sentinel to NaN.
pub fn sector_number_at_call(
    sectors_starts: Vec<f64>,
    sectors_ends: Vec<f64>,
    t: f64,
) -> u32 {
    let sectors: Vec<(f64, f64)> = sectors_starts
        .iter()
        .zip(sectors_ends.iter())
        .map(|(s, e)| (*s, *e))
        .collect();
    sector_number_at(&sectors, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_time_identity() {
        // Arrange — both laps identical: straight east, 100 m, 10 samples,
        // channel value = sample index.
        let main_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let main_positions: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 10.0, 0.0, 0.0))
            .collect();
        let overlay_ref: Vec<(f64, f64, f64)> = (0..=10)
            .map(|i| (i as f64 * 10.0, 0.0, i as f64))
            .collect();
        let overlay_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();

        // Act — disable lap-window gating (start >= end sentinel) so every
        // sample is processed.
        let out = variance_time(
            &main_samples,
            1.0,    // main rate
            0.0,    // main lap start
            0.0,    // main lap end (sentinel: gating off)
            &main_positions,
            overlay_ref,
            &overlay_samples,
            1.0,    // overlay rate
            0.0,    // overlay lap start = t=0 session-relative
        );

        // Assert — main and overlay match exactly → diff ≈ 0.
        for v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn variance_time_nan_outside_main_lap() {
        // Arrange — same identity setup as above, but gate samples to the
        // window [3, 7) at 1 Hz. Samples 0..2 and 7..9 must be NaN; 3..6 ≈ 0.
        let main_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let main_positions: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 10.0, 0.0, 0.0))
            .collect();
        let overlay_ref: Vec<(f64, f64, f64)> = (0..=10)
            .map(|i| (i as f64 * 10.0, 0.0, i as f64))
            .collect();
        let overlay_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();

        // Act
        let out = variance_time(
            &main_samples,
            1.0, 3.0, 7.0,
            &main_positions,
            overlay_ref,
            &overlay_samples,
            1.0,
            0.0,
        );

        // Assert
        for (i, v) in out.iter().enumerate() {
            if i < 3 || i >= 7 {
                assert!(v.is_nan(), "expected NaN at i={}, got {}", i, v);
            } else {
                assert!(v.abs() < 1e-6, "expected ~0 at i={}, got {}", i, v);
            }
        }
    }

    #[test]
    fn variance_dist_arc_length_mismatch() {
        // Arrange — main 100 m, overlay 80 m. Main samples 0..10 in 10 m
        // increments; overlay samples 0..8 in 10 m increments.
        let main_samples = vec![1.0; 11];
        let main_arc: Vec<f64> = (0..=10).map(|i| i as f64 * 10.0).collect();
        let overlay_samples = vec![1.0; 9];
        let overlay_arc: Vec<f64> = (0..=8).map(|i| i as f64 * 10.0).collect();

        // Act — gating disabled (start >= end sentinel).
        let out = variance_dist(
            &main_samples,
            1.0,    // main rate
            0.0,    // main lap start
            0.0,    // main lap end (sentinel: gating off)
            &main_arc,
            &overlay_samples,
            &overlay_arc,
        );

        // Assert — main samples at arc 0..80 match (diff = 0); 90, 100 are NaN.
        assert_eq!(out.len(), 11);
        for i in 0..=8 {
            assert!(out[i].abs() < 1e-6);
        }
        assert!(out[9].is_nan());
        assert!(out[10].is_nan());
    }

    #[test]
    fn current_lap_in_window() {
        // Arrange — two laps with exclusive end boundaries.
        let laps = vec![
            (0.0, 10.0),  // lap 1
            (10.0, 20.0), // lap 2
        ];

        // Act + Assert
        assert_eq!(current_lap_at(&laps, 5.0), 1);
        assert_eq!(current_lap_at(&laps, 15.0), 2);
        assert_eq!(current_lap_at(&laps, 25.0), 0); // outside
    }

    #[test]
    fn variance_traces_zero_for_identical_target_offset_for_shifted() {
        use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};
        // A straight east lap at 1 Hz GPS, channel = sample index (+bias). Two
        // targets: one identical to the reference, one with channel + 5.
        fn session(id: &str, bias: f64) -> SessionHandle {
            let meta = SessionMetaInput {
                session_id: id.into(),
                device_id: "d".into(),
                timestamp_utc_ms: 0,
                config_checksum: String::new(),
            };
            let lat = vec![0.0; 11];
            let lon: Vec<f64> = (0..11).map(|i| i as f64 * 0.0001).collect(); // marches east
            let epoch: Vec<f64> = (0..11).map(|i| i as f64 * 1000.0).collect(); // ms, 1 Hz
            let fork: Vec<f64> = (0..11).map(|i| i as f64 + bias).collect();
            SessionHandle::from_channels(
                meta,
                vec![
                    ChannelInput { channel_id: "GPS_Latitude".into(), sample_rate_hz: 1.0, samples: lat, sample_times_secs: None },
                    ChannelInput { channel_id: "GPS_Longitude".into(), sample_rate_hz: 1.0, samples: lon, sample_times_secs: None },
                    ChannelInput { channel_id: "GPS_EpochMs".into(), sample_rate_hz: 1.0, samples: epoch, sample_times_secs: None },
                    ChannelInput { channel_id: "Fork".into(), sample_rate_hz: 1.0, samples: fork, sample_times_secs: None },
                ],
            )
        }
        let reference_s = session("R", 0.0);
        let same_s = session("S", 0.0);
        let shifted_s = session("T", 5.0);
        fn lr(h: &SessionHandle) -> LapRef<'_> {
            LapRef {
                handle: h,
                lap_start_ms: 0.0,
                lap_end_ms: 10_000.0,
                lap_start_uniform_sec: 0.0,
                window_start_sec: 0.0,
                window_end_sec: 0.0, // gating off (start >= end sentinel)
            }
        }

        // Act — Distance mode: each target vs the reference at matching position.
        let reference = lr(&reference_s);
        let targets = [lr(&same_s), lr(&shifted_s)];
        let out = variance_traces(&reference, &targets, "Fork", VarianceMode::Distance);

        // Assert — identical target ≈ 0; shifted target ≈ +5 (ignoring any NaN tail).
        assert_eq!(out.len(), 2);
        assert!(
            out[0].iter().filter(|v| !v.is_nan()).all(|v| v.abs() < 1e-6),
            "identical target should be ~0, got {:?}",
            out[0]
        );
        assert!(
            out[1].iter().filter(|v| !v.is_nan()).any(|_| true),
            "shifted target should have at least one non-NaN sample, got {:?}",
            out[1]
        );
        assert!(
            out[1].iter().filter(|v| !v.is_nan()).all(|v| (v - 5.0).abs() < 1e-6),
            "shifted target should be ~+5, got {:?}",
            out[1]
        );
    }
}
