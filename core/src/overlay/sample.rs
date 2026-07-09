//! Frame sampling for overlay rendering: prepare once, sample per frame.
//! All `t` are session recording-time seconds (the synthesized `Time`
//! channel's domain). See docs/IDL0_SPEC.md §33.2.

use std::collections::HashMap;

use crate::gps::build_gps_track;
use crate::laps::model::Lap;
use crate::overlay::model::{OverlayElement, OverlayLayout};
use crate::session::handle::SessionHandle;

/// Max points kept per trace-strip window (stride-decimated above this).
const MAX_TRACE_POINTS: usize = 256;
/// Max points kept in the prepared track polyline.
const MAX_POLYLINE_POINTS: usize = 1024;
/// A map position older than this many seconds renders no dot.
const MAP_POS_STALE_S: f64 = 5.0;

/// One prepared channel: samples plus its time base.
struct PreparedChannel {
    samples: Vec<f64>,
    /// `None` → rate-based (`rate_hz` applies); `Some` → event-driven
    /// per-sample times in seconds.
    times_s: Option<Vec<f64>>,
    /// Nominal rate in Hz (rate-based channels only; 0 = event-driven).
    rate_hz: f64,
    /// Session-wide (min, max) for stable trace axes; `None` when empty.
    min_max: Option<(f64, f64)>,
}

impl PreparedChannel {
    /// Value at `t` seconds: linear interpolation for rate-based channels,
    /// carry-forward for event-driven. `None` outside the channel's span.
    fn value_at(&self, t: f64) -> Option<f64> {
        if self.samples.is_empty() || !t.is_finite() {
            return None;
        }
        match &self.times_s {
            Some(times) => {
                // Event-driven: last sample at-or-before t.
                let n = times.partition_point(|&ts| ts <= t);
                if n == 0 {
                    None
                } else {
                    self.samples.get(n - 1).copied()
                }
            }
            None => {
                if self.rate_hz <= 0.0 {
                    return None;
                }
                let idx = t * self.rate_hz;
                if idx < 0.0 || idx > (self.samples.len() - 1) as f64 {
                    return None;
                }
                let i0 = idx.floor() as usize;
                let i1 = (i0 + 1).min(self.samples.len() - 1);
                let frac = idx - i0 as f64;
                Some(self.samples[i0] * (1.0 - frac) + self.samples[i1] * frac)
            }
        }
    }

    /// (t, value) pairs inside `[t0, t1]`, stride-decimated to
    /// `MAX_TRACE_POINTS`.
    fn window(&self, t0: f64, t1: f64) -> Vec<(f64, f64)> {
        if self.samples.is_empty() || t1 <= t0 {
            return Vec::new();
        }
        let pairs: Vec<(f64, f64)> = match &self.times_s {
            Some(times) => {
                let lo = times.partition_point(|&ts| ts < t0);
                let hi = times.partition_point(|&ts| ts <= t1);
                (lo..hi).map(|i| (times[i], self.samples[i])).collect()
            }
            None => {
                if self.rate_hz <= 0.0 {
                    return Vec::new();
                }
                let lo = (t0 * self.rate_hz).ceil().max(0.0) as usize;
                let hi = ((t1 * self.rate_hz).floor() as usize).min(self.samples.len() - 1);
                if lo > hi {
                    return Vec::new();
                }
                (lo..=hi)
                    .map(|i| (i as f64 / self.rate_hz, self.samples[i]))
                    .collect()
            }
        };
        if pairs.len() <= MAX_TRACE_POINTS {
            return pairs;
        }
        let stride = pairs.len().div_ceil(MAX_TRACE_POINTS);
        pairs.into_iter().step_by(stride).collect()
    }
}

/// Lap-panel state at one instant. All times from the engine lap model
/// (`lap_time_ms` = effective lap time, neutral zones removed).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LapState {
    /// 1-based lap number containing `t`, when inside a lap.
    pub current_lap: Option<u32>,
    /// Seconds elapsed into the current lap (0 when `current_lap` is None).
    pub lap_elapsed_s: f64,
    /// Most recently completed lap's time in ms.
    pub last_lap_ms: Option<i64>,
    /// Best completed lap time in ms so far.
    pub best_lap_ms: Option<i64>,
}

/// One element's sample at an instant; variants parallel
/// [`crate::overlay::model::OverlayElement`].
#[derive(Debug, Clone, PartialEq)]
pub enum ElementSample {
    /// Gauge/Attitude value; `None` = no data at `t`.
    Value(Option<f64>),
    /// TraceStrip: per channel, points normalized to the element —
    /// x = position in window [0, 1] ("now" at 1), y = value in the
    /// channel's session min/max [0, 1]. Empty inner vec = no data.
    Trace(Vec<Vec<(f32, f32)>>),
    /// TrackMap: current position normalized to the track bbox (y up).
    MapPos(Option<(f32, f32)>),
    /// LapPanel state.
    Laps(LapState),
}

/// Every element's sample at `t_secs`, parallel to `layout.elements`.
#[derive(Debug, Clone)]
pub struct FrameSample {
    pub elements: Vec<ElementSample>,
    /// The session recording-time this sample was taken at, in seconds.
    pub t_secs: f64,
}

/// Prepared per-render state: channels materialized once, GPS polyline
/// normalized once, laps sorted once. `sample()` per frame is cheap.
pub struct SampleContext {
    elements: Vec<OverlayElement>,
    channels: HashMap<String, PreparedChannel>,
    /// Normalized (x, y-up) session polyline, bbox-fitted, ≤ 1024 points.
    polyline: Vec<(f32, f32)>,
    /// (t_secs, normalized x, normalized y) per GPS fix, time-ascending.
    gps_norm: Vec<(f64, f32, f32)>,
    laps: Vec<Lap>,
}

impl SampleContext {
    /// Materialize everything `layout` references from `handle`. Channels the
    /// session lacks are simply absent (their elements sample as no-data).
    /// `laps` come from the caller (CLI: `detect_laps`; app: cached visits) —
    /// empty means lap elements render no-data.
    pub fn prepare(
        handle: &SessionHandle,
        layout: &OverlayLayout,
        laps: Vec<Lap>,
    ) -> SampleContext {
        let metas: HashMap<String, (f64, bool)> = handle
            .channels()
            .into_iter()
            .map(|m| (m.channel_id.clone(), (m.sample_rate_hz, m.is_event_driven)))
            .collect();

        let mut channels = HashMap::new();
        for name in layout.referenced_channels() {
            let Some(&(rate_hz, _event)) = metas.get(&name) else {
                continue;
            };
            let samples = handle.channel_samples(&name);
            if samples.is_empty() {
                continue;
            }
            let times_s = handle.channel_sample_times(&name);
            let min_max = handle.channel_min_max(&name);
            channels.insert(
                name,
                PreparedChannel {
                    samples,
                    times_s,
                    rate_hz,
                    min_max,
                },
            );
        }

        // GPS: normalize the whole-session track to its bbox, y up. Raw
        // channel scale (degrees × 1e7) — bbox normalization is scale-free.
        let fixes = build_gps_track(handle);
        let mut polyline = Vec::new();
        let mut gps_norm = Vec::new();
        if !fixes.is_empty() {
            let (mut lat_min, mut lat_max) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut lon_min, mut lon_max) = (f64::INFINITY, f64::NEG_INFINITY);
            for f in &fixes {
                lat_min = lat_min.min(f.lat);
                lat_max = lat_max.max(f.lat);
                lon_min = lon_min.min(f.lon);
                lon_max = lon_max.max(f.lon);
            }
            let lat_span = lat_max - lat_min;
            let lon_span = lon_max - lon_min;
            let norm = |lat: f64, lon: f64| -> (f32, f32) {
                let x = if lon_span > 0.0 {
                    (lon - lon_min) / lon_span
                } else {
                    0.5
                };
                let y = if lat_span > 0.0 {
                    (lat - lat_min) / lat_span
                } else {
                    0.5
                };
                (x as f32, y as f32)
            };
            let epochs: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
            let secs = handle.epoch_ms_to_time_secs(&epochs);
            for (f, &t) in fixes.iter().zip(secs.iter()) {
                let (x, y) = norm(f.lat, f.lon);
                gps_norm.push((t, x, y));
            }
            let stride = fixes.len().div_ceil(MAX_POLYLINE_POINTS).max(1);
            polyline = gps_norm
                .iter()
                .step_by(stride)
                .map(|&(_, x, y)| (x, y))
                .collect();
        }

        let mut laps = laps;
        laps.sort_by(|a, b| a.start_time_secs.total_cmp(&b.start_time_secs));

        SampleContext {
            elements: layout.elements.clone(),
            channels,
            polyline,
            gps_norm,
            laps,
        }
    }

    /// Normalized session GPS polyline for the track map (empty = no GPS).
    pub fn track_polyline(&self) -> &[(f32, f32)] {
        &self.polyline
    }

    /// Sample every element at session-time `t_secs`.
    pub fn sample(&self, t_secs: f64) -> FrameSample {
        let elements = self
            .elements
            .iter()
            .map(|e| match e {
                OverlayElement::Gauge { channel, .. }
                | OverlayElement::Attitude { channel, .. } => ElementSample::Value(
                    self.channels.get(channel).and_then(|c| c.value_at(t_secs)),
                ),
                OverlayElement::TraceStrip {
                    channels, window_s, ..
                } => {
                    let t0 = t_secs - window_s;
                    let series = channels
                        .iter()
                        .map(|name| {
                            let Some(ch) = self.channels.get(name) else {
                                return Vec::new();
                            };
                            let (min, max) = match ch.min_max {
                                Some((min, max)) if max > min => (min, max),
                                // Flat or unknown range: pin to mid-height.
                                _ => {
                                    return ch
                                        .window(t0, t_secs)
                                        .into_iter()
                                        .map(|(ts, _)| (((ts - t0) / window_s) as f32, 0.5))
                                        .collect();
                                }
                            };
                            ch.window(t0, t_secs)
                                .into_iter()
                                .map(|(ts, v)| {
                                    (
                                        (((ts - t0) / window_s) as f32).clamp(0.0, 1.0),
                                        (((v - min) / (max - min)) as f32).clamp(0.0, 1.0),
                                    )
                                })
                                .collect()
                        })
                        .collect();
                    ElementSample::Trace(series)
                }
                OverlayElement::TrackMap { .. } => {
                    let n = self.gps_norm.partition_point(|&(t, _, _)| t <= t_secs);
                    let pos = if n == 0 {
                        None
                    } else {
                        let (t, x, y) = self.gps_norm[n - 1];
                        (t_secs - t <= MAP_POS_STALE_S).then_some((x, y))
                    };
                    ElementSample::MapPos(pos)
                }
                OverlayElement::LapPanel { .. } => {
                    let mut state = LapState::default();
                    for lap in &self.laps {
                        if lap.start_time_secs <= t_secs && t_secs < lap.end_time_secs {
                            state.current_lap = Some(lap.lap_number);
                            state.lap_elapsed_s = t_secs - lap.start_time_secs;
                        }
                        if lap.end_time_secs <= t_secs {
                            state.last_lap_ms = Some(lap.lap_time_ms);
                            state.best_lap_ms = Some(match state.best_lap_ms {
                                Some(best) => best.min(lap.lap_time_ms),
                                None => lap.lap_time_ms,
                            });
                        }
                    }
                    ElementSample::Laps(state)
                }
            })
            .collect();
        FrameSample { elements, t_secs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::model::{AttitudeStyle, GaugeStyle, OverlayElement, OverlayLayout, Rect};
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn meta() -> SessionMetaInput {
        SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        }
    }

    /// 10 Hz ramp 0..=100 over 10 s.
    fn ramp_handle() -> SessionHandle {
        let samples: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        SessionHandle::from_channels(
            meta(),
            vec![ChannelInput {
                channel_id: "Speed".into(),
                sample_rate_hz: 10.0,
                samples,
                sample_times_secs: None,
            }],
        )
    }

    fn rect() -> Rect {
        [0.0, 0.0, 0.1, 0.1].into()
    }

    fn layout_of(elements: Vec<OverlayElement>) -> OverlayLayout {
        OverlayLayout {
            id: "L".into(),
            name: "L".into(),
            canvas: "1920x1080".into(),
            elements,
        }
    }

    fn gauge_layout(channel: &str) -> OverlayLayout {
        layout_of(vec![OverlayElement::Gauge {
            rect: rect(),
            channel: channel.into(),
            style: GaugeStyle::Numeric,
            label: String::new(),
            min: 0.0,
            max: 100.0,
        }])
    }

    fn lap(n: u32, s: f64, e: f64, ms: i64) -> crate::laps::model::Lap {
        crate::laps::model::Lap {
            lap_number: n,
            start_ms: (s * 1000.0) as i64,
            end_ms: (e * 1000.0) as i64,
            start_time_secs: s,
            end_time_secs: e,
            raw_elapsed_ms: ms,
            lap_time_ms: ms,
            sectors: vec![],
            neutral_zone_visits: vec![],
        }
    }

    #[test]
    fn sample_rate_based_between_samples_interpolates_linearly() {
        // Arrange
        let ctx = SampleContext::prepare(&ramp_handle(), &gauge_layout("Speed"), vec![]);

        // Act — 10 Hz ramp: t=1.25 s → index 12.5 → value 12.5
        let s = ctx.sample(1.25);

        // Assert
        match &s.elements[0] {
            ElementSample::Value(Some(v)) => assert!((v - 12.5).abs() < 1e-9),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sample_t_outside_channel_span_returns_no_data() {
        // Arrange
        let ctx = SampleContext::prepare(&ramp_handle(), &gauge_layout("Speed"), vec![]);

        // Act + Assert
        assert!(matches!(
            &ctx.sample(999.0).elements[0],
            ElementSample::Value(None)
        ));
        assert!(matches!(
            &ctx.sample(-1.0).elements[0],
            ElementSample::Value(None)
        ));
    }

    #[test]
    fn sample_missing_channel_returns_no_data_not_error() {
        // Arrange
        let ctx = SampleContext::prepare(&ramp_handle(), &gauge_layout("Nope"), vec![]);

        // Act
        let s = ctx.sample(1.0);

        // Assert
        assert!(matches!(&s.elements[0], ElementSample::Value(None)));
    }

    #[test]
    fn sample_event_driven_channel_carries_forward() {
        // Arrange — beats at t = 1, 2, 4 s with values 60, 62, 64.
        let h = SessionHandle::from_channels(
            meta(),
            vec![ChannelInput {
                channel_id: "HR_BPM".into(),
                sample_rate_hz: 0.0,
                samples: vec![60.0, 62.0, 64.0],
                sample_times_secs: Some(vec![1.0, 2.0, 4.0]),
            }],
        );
        let ctx = SampleContext::prepare(&h, &gauge_layout("HR_BPM"), vec![]);

        // Act + Assert — t=3 between beats → carry 62; t=0.5 predates → None.
        match &ctx.sample(3.0).elements[0] {
            ElementSample::Value(Some(v)) => assert_eq!(*v, 62.0),
            o => panic!("{o:?}"),
        }
        assert!(matches!(
            &ctx.sample(0.5).elements[0],
            ElementSample::Value(None)
        ));
    }

    #[test]
    fn sample_attitude_element_reads_channel_like_gauge() {
        // Arrange
        let layout = layout_of(vec![OverlayElement::Attitude {
            rect: rect(),
            channel: "Speed".into(),
            style: AttitudeStyle::Roll,
            range_deg: 60.0,
        }]);
        let ctx = SampleContext::prepare(&ramp_handle(), &layout, vec![]);

        // Act + Assert
        match &ctx.sample(2.0).elements[0] {
            ElementSample::Value(Some(v)) => assert!((v - 20.0).abs() < 1e-9),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn sample_trace_strip_normalizes_window_points() {
        // Arrange
        let layout = layout_of(vec![OverlayElement::TraceStrip {
            rect: rect(),
            channels: vec!["Speed".into()],
            window_s: 2.0,
        }]);
        let ctx = SampleContext::prepare(&ramp_handle(), &layout, vec![]);

        // Act — window [3, 5] of the 0..100 ramp (values 30..50).
        let s = ctx.sample(5.0);

        // Assert
        match &s.elements[0] {
            ElementSample::Trace(series) => {
                let pts = &series[0];
                assert!(!pts.is_empty());
                let (x0, y0) = pts[0];
                let (xn, yn) = *pts.last().unwrap();
                assert!(
                    (0.0..0.05).contains(&x0),
                    "window start at left edge, got {x0}"
                );
                assert!((xn - 1.0).abs() < 0.05, "now at right edge, got {xn}");
                assert!(
                    (y0 - 0.30).abs() < 0.02,
                    "y0 normalized to session 0..100, got {y0}"
                );
                assert!(
                    (yn - 0.50).abs() < 0.02,
                    "yn normalized to session 0..100, got {yn}"
                );
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn sample_lap_state_mid_third_lap_reports_current_last_best() {
        // Arrange — laps: 40 s (35 s effective), 45 s, then t inside lap 3.
        let laps = vec![
            lap(1, 0.0, 40.0, 35_000),
            lap(2, 40.0, 85.0, 45_000),
            lap(3, 85.0, 130.0, 45_000),
        ];
        let layout = layout_of(vec![OverlayElement::LapPanel { rect: rect() }]);
        let ctx = SampleContext::prepare(&ramp_handle(), &layout, laps);

        // Act
        let s = ctx.sample(95.0);

        // Assert
        match &s.elements[0] {
            ElementSample::Laps(ls) => {
                assert_eq!(ls.current_lap, Some(3));
                assert!((ls.lap_elapsed_s - 10.0).abs() < 1e-9);
                assert_eq!(ls.last_lap_ms, Some(45_000));
                assert_eq!(ls.best_lap_ms, Some(35_000));
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn sample_lap_state_before_any_lap_is_default() {
        // Arrange
        let laps = vec![lap(1, 10.0, 40.0, 30_000)];
        let layout = layout_of(vec![OverlayElement::LapPanel { rect: rect() }]);
        let ctx = SampleContext::prepare(&ramp_handle(), &layout, laps);

        // Act
        let s = ctx.sample(5.0);

        // Assert
        assert!(matches!(&s.elements[0], ElementSample::Laps(ls) if *ls == LapState::default()));
    }

    #[test]
    fn sample_track_map_without_gps_channels_has_no_position() {
        // Arrange
        let layout = layout_of(vec![OverlayElement::TrackMap { rect: rect() }]);
        let ctx = SampleContext::prepare(&ramp_handle(), &layout, vec![]);

        // Act + Assert
        assert!(ctx.track_polyline().is_empty());
        assert!(matches!(
            &ctx.sample(1.0).elements[0],
            ElementSample::MapPos(None)
        ));
    }
}
