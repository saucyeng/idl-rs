//! Channel-vs-channel scatter for the Analyze tab's G-G diagram and any XY
//! cloud. Pure: pairs two channels over a time window, decimates the cloud, and
//! bins the 2D density — data in, reduced result out. No DSP libraries needed;
//! this is finite-pairing, uniform-stride decimation, and a 2D histogram. The
//! engine owns every bound it returns so no axis math happens Dart-side.

use crate::session::handle::SessionHandle;

/// A decimated `(x, y)` cloud over a `[t0, t1]` window, with an optional
/// per-point colour value and the **exact** pre-decimation data extent.
///
/// `xs`/`ys` are index-aligned and equal length; `colors` (when present) matches
/// that length. `x_min`…`y_max` are the finite data extent over the window
/// *before* decimation, so an equal-aspect caller squares its axes from this one
/// result — no second bounds call crosses the FRB boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterPoints {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub colors: Option<Vec<f64>>,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Index-aligns `xs`/`ys` (and optional `cs`) over their common length, keeping
/// only triples whose `x` and `y` are both finite. A non-finite colour is kept
/// (the renderer maps it to a transparent dot). This is the same-rate G-G
/// pairing — both channels come off the central IMU at one rate, so equal length
/// means exact index alignment. Cross-rate resampling is deferred (design §12);
/// mismatched-rate channels pair by sample index over the shorter length.
pub(crate) fn pair_finite(
    xs: &[f64],
    ys: &[f64],
    cs: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let n = xs.len().min(ys.len());
    let mut out_x = Vec::with_capacity(n);
    let mut out_y = Vec::with_capacity(n);
    let mut out_c = cs.map(|_| Vec::with_capacity(n));
    for i in 0..n {
        let (x, y) = (xs[i], ys[i]);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        out_x.push(x);
        out_y.push(y);
        if let (Some(c), Some(out)) = (cs, out_c.as_mut()) {
            out.push(c.get(i).copied().unwrap_or(f64::NAN));
        }
    }
    (out_x, out_y, out_c)
}

/// Paired, finite, decimated `(x, y)` samples over `[t0_secs, t1_secs]`, capped
/// at `max_points` by uniform stride (honest down-sampling that preserves the
/// envelope; density mode is the answer for time-at-state). `color_channel`, when
/// given, attaches one colour value per kept point. The returned extent is the
/// finite data min/max over the window, computed before decimation.
///
/// Input/output units are whatever the channels carry (e.g. g for a G-G plot).
pub fn scatter_points(
    handle: &SessionHandle,
    x_channel: &str,
    y_channel: &str,
    color_channel: Option<&str>,
    t0_secs: f64,
    t1_secs: f64,
    max_points: u32,
) -> ScatterPoints {
    let xs_raw = handle.slice_by_time(x_channel, t0_secs, t1_secs);
    let ys_raw = handle.slice_by_time(y_channel, t0_secs, t1_secs);
    let cs_raw = color_channel.map(|c| handle.slice_by_time(c, t0_secs, t1_secs));
    let (xs, ys, cs) = pair_finite(&xs_raw, &ys_raw, cs_raw.as_deref());

    // Extent over the full finite cloud (pre-decimation), so equal-aspect
    // squaring uses the true bounds even though the cloud is thinned below.
    let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..xs.len() {
        x_min = x_min.min(xs[i]);
        x_max = x_max.max(xs[i]);
        y_min = y_min.min(ys[i]);
        y_max = y_max.max(ys[i]);
    }
    if xs.is_empty() {
        x_min = 0.0;
        x_max = 0.0;
        y_min = 0.0;
        y_max = 0.0;
    }

    // Uniform-stride decimation to <= max_points.
    let cap = max_points.max(1) as usize;
    let (dx, dy, dc) = if xs.len() > cap {
        let stride = (xs.len() + cap - 1) / cap; // ceil(len / cap)
        let dx: Vec<f64> = xs.iter().step_by(stride).copied().collect();
        let dy: Vec<f64> = ys.iter().step_by(stride).copied().collect();
        let dc = cs
            .as_ref()
            .map(|c| c.iter().step_by(stride).copied().collect());
        (dx, dy, dc)
    } else {
        (xs, ys, cs)
    };

    ScatterPoints { xs: dx, ys: dy, colors: dc, x_min, x_max, y_min, y_max }
}

/// A 2D histogram of the `(x, y)` cloud over `[t0, t1]` into a `bins × bins`
/// grid. `counts[r * bins + c]` is the sample count in row `r` (y), column `c`
/// (x). The engine derives and **returns** the bounds it binned into, so the
/// caller renders axes from them and never computes bounds to pass back.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterDensity {
    pub bins: u32,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub counts: Vec<u32>,
}

/// 2D count histogram of the paired cloud. When `equal_aspect`, both axes share
/// one square range covering their combined extent — symmetric about 0 when the
/// data straddles zero (the G-G case, so the friction circle stays centred),
/// else a square fit of the union extent. Otherwise each axis spans its own
/// data min/max. An empty cloud yields an all-zero grid with finite bounds.
pub fn scatter_density(
    handle: &SessionHandle,
    x_channel: &str,
    y_channel: &str,
    t0_secs: f64,
    t1_secs: f64,
    bins: u32,
    equal_aspect: bool,
) -> ScatterDensity {
    let xs_raw = handle.slice_by_time(x_channel, t0_secs, t1_secs);
    let ys_raw = handle.slice_by_time(y_channel, t0_secs, t1_secs);
    let (xs, ys, _) = pair_finite(&xs_raw, &ys_raw, None);
    let bins = bins.max(1);

    // Per-axis data extent.
    let (mut xmn, mut xmx) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymn, mut ymx) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..xs.len() {
        xmn = xmn.min(xs[i]);
        xmx = xmx.max(xs[i]);
        ymn = ymn.min(ys[i]);
        ymx = ymx.max(ys[i]);
    }
    if xs.is_empty() {
        xmn = 0.0;
        xmx = 0.0;
        ymn = 0.0;
        ymx = 0.0;
    }

    // Resolve the binning bounds.
    let (x0, x1, y0, y1) = if equal_aspect {
        let lo = xmn.min(ymn);
        let hi = xmx.max(ymx);
        if lo < 0.0 && hi > 0.0 {
            let a = (-lo).max(hi); // symmetric about 0
            (-a, a, -a, a)
        } else {
            (lo, hi, lo, hi)
        }
    } else {
        (xmn, xmx, ymn, ymx)
    };

    let mut counts = vec![0u32; (bins * bins) as usize];
    let bin_index = |v: f64, lo: f64, hi: f64| -> usize {
        if hi <= lo {
            return 0; // zero-width axis → single column
        }
        let t = (v - lo) / (hi - lo);
        ((t * bins as f64).floor() as isize).clamp(0, bins as isize - 1) as usize
    };
    for i in 0..xs.len() {
        let c = bin_index(xs[i], x0, x1);
        let r = bin_index(ys[i], y0, y1);
        counts[r * bins as usize + c] += 1;
    }

    ScatterDensity { bins, x_min: x0, x_max: x1, y_min: y0, y_max: y1, counts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn meta() -> SessionMetaInput {
        SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        }
    }

    fn ch(id: &str, rate: f64, samples: Vec<f64>) -> ChannelInput {
        ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: rate,
            samples,
            sample_times_secs: None,
        }
    }

    #[test]
    fn scatter_points_pairs_same_rate_channels_and_reports_extent() {
        // Arrange — two 10 Hz channels, 10 samples each over [0, 0.9] s.
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, 0.0]),
                ch("Y", 10.0, vec![0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0]),
            ],
        );

        // Act
        let r = scatter_points(&h, "X", "Y", None, 0.0, 0.9, 1000);

        // Assert — index-aligned, all finite, extent is the data min/max per axis.
        assert_eq!(r.xs.len(), 10);
        assert_eq!(r.ys.len(), 10);
        assert!(r.colors.is_none());
        assert_eq!(r.xs[3], 1.0);
        assert_eq!(r.ys[3], 1.0);
        assert_eq!(r.x_min, -2.0);
        assert_eq!(r.x_max, 2.0);
        assert_eq!(r.y_min, -2.0);
        assert_eq!(r.y_max, 2.0);
    }

    #[test]
    fn scatter_points_drops_non_finite_pairs() {
        // Arrange — a NaN in X and a NaN in Y each kill their pair.
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![1.0, f64::NAN, 3.0, 4.0]),
                ch("Y", 10.0, vec![1.0, 2.0, f64::NAN, 4.0]),
            ],
        );

        // Act
        let r = scatter_points(&h, "X", "Y", None, 0.0, 0.3, 1000);

        // Assert — only indices 0 and 3 survive.
        assert_eq!(r.xs, vec![1.0, 4.0]);
        assert_eq!(r.ys, vec![1.0, 4.0]);
    }

    #[test]
    fn scatter_points_attaches_colors_of_matching_length() {
        // Arrange
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![0.0, 1.0, 2.0]),
                ch("Y", 10.0, vec![0.0, 1.0, 2.0]),
                ch("C", 10.0, vec![10.0, 20.0, 30.0]),
            ],
        );

        // Act
        let r = scatter_points(&h, "X", "Y", Some("C"), 0.0, 0.2, 1000);

        // Assert
        assert_eq!(r.colors, Some(vec![10.0, 20.0, 30.0]));
        assert_eq!(r.colors.as_ref().unwrap().len(), r.xs.len());
    }

    #[test]
    fn scatter_points_decimates_to_max_points() {
        // Arrange — 100 samples, cap at 10.
        let xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let h = SessionHandle::from_channels(
            meta(),
            vec![ch("X", 100.0, xs.clone()), ch("Y", 100.0, xs)],
        );

        // Act
        let r = scatter_points(&h, "X", "Y", None, 0.0, 0.99, 10);

        // Assert — stride-decimated to <= max_points, extent preserved from full data.
        assert!(r.xs.len() <= 10);
        assert_eq!(r.x_min, 0.0);
        assert_eq!(r.x_max, 99.0);
    }

    #[test]
    fn scatter_density_bins_a_known_cloud_and_totals_to_pair_count() {
        // Arrange — 4 points, one in each quadrant of a 2x2 grid over [-1, 1]^2.
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![-0.5, 0.5, -0.5, 0.5]),
                ch("Y", 10.0, vec![-0.5, -0.5, 0.5, 0.5]),
            ],
        );

        // Act — equal-aspect square range, 2 bins per axis.
        let d = scatter_density(&h, "X", "Y", 0.0, 0.3, 2, true);

        // Assert — symmetric square bounds, one sample per cell, total == 4.
        assert_eq!(d.bins, 2);
        assert_eq!(d.x_min, -0.5);
        assert_eq!(d.x_max, 0.5);
        assert_eq!(d.y_min, -0.5);
        assert_eq!(d.y_max, 0.5);
        assert_eq!(d.counts.iter().sum::<u32>(), 4);
        assert!(d.counts.iter().all(|&c| c == 1));
    }

    #[test]
    fn scatter_density_equal_aspect_makes_a_symmetric_square_range() {
        // Arrange — X spans [-2, 2], Y spans [-1, 1]: equal-aspect must square to
        // the larger half-extent on BOTH axes, symmetric about 0.
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![-2.0, 2.0]),
                ch("Y", 10.0, vec![-1.0, 1.0]),
            ],
        );

        // Act
        let d = scatter_density(&h, "X", "Y", 0.0, 0.1, 4, true);

        // Assert
        assert_eq!(d.x_min, -2.0);
        assert_eq!(d.x_max, 2.0);
        assert_eq!(d.y_min, -2.0);
        assert_eq!(d.y_max, 2.0);
    }

    #[test]
    fn scatter_density_without_equal_aspect_uses_per_axis_bounds() {
        // Arrange
        let h = SessionHandle::from_channels(
            meta(),
            vec![
                ch("X", 10.0, vec![0.0, 10.0]),
                ch("Y", 10.0, vec![0.0, 100.0]),
            ],
        );

        // Act
        let d = scatter_density(&h, "X", "Y", 0.0, 0.1, 4, false);

        // Assert — each axis keeps its own extent.
        assert_eq!((d.x_min, d.x_max), (0.0, 10.0));
        assert_eq!((d.y_min, d.y_max), (0.0, 100.0));
    }

    #[test]
    fn scatter_density_empty_cloud_returns_zero_counts_no_nan() {
        // Arrange — window covers no sample.
        let h = SessionHandle::from_channels(meta(), vec![ch("X", 10.0, vec![1.0]), ch("Y", 10.0, vec![1.0])]);

        // Act — t-window past the single sample.
        let d = scatter_density(&h, "X", "Y", 5.0, 6.0, 3, true);

        // Assert — empty grid, finite bounds, no panic.
        assert_eq!(d.counts.len(), 9);
        assert_eq!(d.counts.iter().sum::<u32>(), 0);
        assert!(d.x_min.is_finite() && d.x_max.is_finite());
    }
}
