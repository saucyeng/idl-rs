//! Clipping reconstruction — rebuilds IMU acceleration peaks that saturated
//! at the sensor rail (±32 g on the LSM6DSO32TR, IDL0_SPEC §3.2).
//!
//! A clipped segment is a run of consecutive samples pinned within `eps` of
//! the rail. The true peak is reconstructed by fitting a smooth asymmetric
//! analytic pulse to the segment's unclipped shoulders, so acceleration AND
//! its derivatives (jerk = a′, jounce = a″) are physically close to truth.
//!
//! Pure functions — data in, data out. No Flutter, no I/O.
//! See docs/superpowers/specs/2026-05-29-declip-imu-reconstruction-design.md.

use nalgebra::{Matrix2, Vector2};

/// Finds runs of consecutive samples pinned within `eps` of the ±`rail` value.
///
/// `accel`: input samples in channel units (g for acceleration channels).
/// `rail`: saturation magnitude, same units as `accel` (e.g. 32.0 g).
/// `eps`: tolerance band; a sample counts as pinned when
///        `accel[i].abs() >= rail - eps`.
///
/// Returns inclusive index ranges `(start, end)` of each clipped segment, in
/// ascending order. Empty when nothing is clipped.
///
/// Internal to this module — not part of the public API.
fn find_clipped_segments(accel: &[f64], rail: f64, eps: f64) -> Vec<(usize, usize)> {
    let threshold = rail - eps;
    let mut segments = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &v) in accel.iter().enumerate() {
        let pinned = v.abs() >= threshold;
        match (pinned, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                segments.push((s, i - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        segments.push((s, accel.len() - 1));
    }
    segments
}

/// Smooth unit-peak pulse shapes used as the reconstruction template.
///
/// All variants peak at 1.0 for `u = 0` and decay symmetrically in `u`.
/// `Sech2` and `Lorentzian` are smooth (C²) with a sharper apex than a
/// Gaussian — a better match for the sharp suspension-impact peaks observed
/// in real data. `GenGaussian` exposes a peakedness exponent for tuning.
///
/// The shipped reconstructor only constructs `Sech2` (the tuned default);
/// `GenGaussian` and `Lorentzian` are exercised by the `#[cfg(test)]` tuner as
/// alternative shapes, hence `allow(dead_code)` for the non-test build.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum KernelKind {
    /// exp(-(|u|)^p). p = 2 is Gaussian. Apex stays C² only for p >= 2.
    GenGaussian,
    /// sech²(u) — soliton-like, sharper apex than Gaussian, C² everywhere.
    Sech2,
    /// 1 / (1 + u²) — sharp apex, heavy tails, C² everywhere.
    Lorentzian,
}

/// Evaluates the unit-peak kernel at normalized coordinate `u`.
///
/// `sharpness` is the `GenGaussian` exponent `p` (default 2.0 = Gaussian);
/// it is ignored by `Sech2` and `Lorentzian`, whose shapes are fixed.
/// Returns a value in (0, 1], equal to 1.0 at `u = 0`.
fn kernel(kind: KernelKind, u: f64, sharpness: f64) -> f64 {
    match kind {
        KernelKind::GenGaussian => (-(u.abs()).powf(sharpness)).exp(),
        KernelKind::Sech2 => {
            let c = u.cosh();
            1.0 / (c * c)
        }
        KernelKind::Lorentzian => 1.0 / (1.0 + u * u),
    }
}

/// Fitted parameters for one clipped segment's reconstruction pulse.
#[derive(Clone, Copy, Debug)]
struct PulseFit {
    /// Peak height above the local baseline, channel units.
    amplitude: f64,
    /// Peak location in fractional sample index (absolute, not segment-relative).
    t0: f64,
    /// Rising-edge half-width in samples (t < t0).
    w_rise: f64,
    /// Falling-edge half-width in samples (t >= t0).
    w_fall: f64,
    /// Baseline slope (per sample) and intercept at index `origin` of the fit
    /// window (`x = index − origin`).
    base_slope: f64,
    base_intercept: f64,
}

/// Indices (absolute) used to fit one segment: the unclipped shoulder samples
/// on each side of the clipped span `[clip_start, clip_end]`.
struct FitWindow {
    /// Absolute indices of the shoulder samples used in the fit. Left-shoulder
    /// indices come first (ascending), then right-shoulder indices (ascending).
    idx: Vec<usize>,
    /// Number of left-shoulder samples — `idx[..split]` is the left shoulder,
    /// `idx[split..]` the right.
    split: usize,
    /// Index, within the buffer, that the baseline regression treats as x = 0.
    origin: usize,
}

/// Builds the shoulder fit window: up to `shoulder_n` unclipped samples on each
/// side of `[clip_start, clip_end]`, clamped to the buffer. Returns `None` when
/// neither side has at least 2 samples (cannot constrain a pulse).
fn build_fit_window(
    len: usize,
    clip_start: usize,
    clip_end: usize,
    shoulder_n: usize,
) -> Option<FitWindow> {
    let left_lo = clip_start.saturating_sub(shoulder_n);
    let left_hi = clip_start; // exclusive
    let right_lo = clip_end + 1;
    let right_hi = (clip_end + 1 + shoulder_n).min(len);
    let left_count = left_hi.saturating_sub(left_lo);
    let right_count = right_hi.saturating_sub(right_lo);
    if left_count + right_count < 2 {
        return None;
    }
    let mut idx = Vec::with_capacity(left_count + right_count);
    idx.extend(left_lo..left_hi);
    idx.extend(right_lo..right_hi);
    Some(FitWindow {
        idx,
        split: left_count,
        origin: left_lo,
    })
}

/// Least-squares baseline line `y = slope·x + intercept`, `x = (index −
/// window.origin)`, fit ONLY over the outermost samples of each shoulder.
///
/// The samples nearest the clip are the pulse's own rising/falling flanks, not
/// baseline — including them would let the "baseline" absorb part of the pulse
/// and corrupt the amplitude fit. Using up to [BASELINE_ANCHOR] samples from the
/// far end of each shoulder anchors the line on the local trend the lobe sits
/// on. Solved via the 2×2 normal equations with nalgebra (`Matrix2::try_inverse`);
/// falls back to a flat line at the anchor mean when the system is singular
/// (anchors at a single x, e.g. a one-sided window).
fn fit_baseline(accel: &[f64], win: &FitWindow) -> (f64, f64) {
    let left = &win.idx[..win.split];
    let right = &win.idx[win.split..];
    let left_take = left.len().min(BASELINE_ANCHOR);
    let right_take = right.len().min(BASELINE_ANCHOR);

    let mut anchors: Vec<usize> = Vec::with_capacity(left_take + right_take);
    anchors.extend(&left[..left_take]); // outermost-left = smallest indices
    anchors.extend(&right[right.len() - right_take..]); // outermost-right = largest

    let mut sxx = 0.0;
    let mut sx = 0.0;
    let mut sxy = 0.0;
    let mut sy = 0.0;
    let nrows = anchors.len() as f64;
    for &i in &anchors {
        let x = i as f64 - win.origin as f64;
        let y = accel[i];
        sxx += x * x;
        sx += x;
        sxy += x * y;
        sy += y;
    }
    let a = Matrix2::new(sxx, sx, sx, nrows);
    let b = Vector2::new(sxy, sy);
    match a.try_inverse() {
        Some(inv) => {
            let sol = inv * b;
            (sol[0], sol[1]) // (slope, intercept)
        }
        None => (0.0, sy / nrows),
    }
}

/// Samples taken from the far end of each shoulder to anchor the baseline.
const BASELINE_ANCHOR: usize = 4;

/// Inverse kernel: returns the non-negative `|u|` at which `kernel(kind, u,
/// sharpness) == y`, for `y` in (0, 1). Used to convert a rail-crossing level
/// into a normalized width so the clip span constrains the pulse. Returns
/// `None` when `y` is out of (0, 1) (no finite crossing).
fn kernel_inverse(kind: KernelKind, y: f64, sharpness: f64) -> Option<f64> {
    if !(y > 0.0 && y < 1.0) {
        return None;
    }
    let x = match kind {
        // sech²(u) = y  →  cosh(u) = 1/√y  →  u = acosh(1/√y)
        KernelKind::Sech2 => (1.0 / y.sqrt()).acosh(),
        // exp(-|u|^p) = y  →  |u| = (-ln y)^(1/p)
        KernelKind::GenGaussian => (-y.ln()).powf(1.0 / sharpness),
        // 1/(1+u²) = y  →  |u| = √(1/y − 1)
        KernelKind::Lorentzian => (1.0 / y - 1.0).sqrt(),
    };
    if x.is_finite() && x > 0.0 {
        Some(x)
    } else {
        None
    }
}

/// Fits the asymmetric pulse to the shoulder samples in `win`, with the kernel
/// shape (`kind`, `sharpness`) fixed. `clip_start`/`clip_end` bound the clipped
/// gap; `rail` is the saturation level.
///
/// The shoulder-only fit is ill-posed on its own — a tall narrow pulse and a
/// short wide one fit the shoulders equally well but extrapolate to wildly
/// different peaks. To constrain it, the clip span is treated as ground truth:
/// for a candidate peak position `t0` and amplitude `A`, the pulse must cross
/// the rail at `clip_start − 0.5` (rising) and `clip_end + 0.5` (falling), which
/// pins `w_rise` and `w_fall` via [kernel_inverse]. The free search is then just
/// 2-D over `(t0, A)`; rise/fall asymmetry falls out of `cs`/`ce` sitting at
/// unequal distances from `t0`. The objective is plain SSE on the shoulders.
/// Returns the lowest-SSE `PulseFit`.
fn fit_pulse(
    accel: &[f64],
    win: &FitWindow,
    clip_start: usize,
    clip_end: usize,
    rail: f64,
    kind: KernelKind,
    sharpness: f64,
) -> PulseFit {
    let (base_slope, base_intercept) = fit_baseline(accel, win);
    let base_at = |t: f64| base_slope * (t - win.origin as f64) + base_intercept;

    let shoulders: Vec<(f64, f64)> = win.idx.iter().map(|&i| (i as f64, accel[i])).collect();

    // Estimate where the true signal crossed the rail, by extrapolating the
    // local slope of the two unclipped samples just outside the clip. Far more
    // accurate than assuming the crossing sits at the half-sample midpoint,
    // which biases the derived widths (and hence amplitude). Falls back to the
    // midpoint when there is no room for a slope estimate at the buffer edge.
    let left_cross = {
        let fallback = clip_start as f64 - 0.5;
        if clip_start >= 2 {
            let y1 = accel[clip_start - 1];
            let y0 = accel[clip_start - 2];
            let slope = y1 - y0;
            if slope > 1e-6 {
                let tc = (clip_start - 1) as f64 + (rail - y1) / slope;
                tc.clamp((clip_start - 1) as f64, clip_start as f64)
            } else {
                fallback
            }
        } else {
            fallback
        }
    };
    let right_cross = {
        let fallback = clip_end as f64 + 0.5;
        if clip_end + 2 < accel.len() {
            let y1 = accel[clip_end + 1];
            let y2 = accel[clip_end + 2];
            let slope = y2 - y1;
            if slope < -1e-6 {
                let tc = (clip_end + 1) as f64 + (rail - y1) / slope;
                tc.clamp(clip_end as f64, (clip_end + 1) as f64)
            } else {
                fallback
            }
        } else {
            fallback
        }
    };

    // For (t0, amp): derive widths from the rail crossings, then score the
    // shoulder SSE. Returns (sse, w_rise, w_fall) or None for an invalid pulse.
    let eval = |t0: f64, amp: f64| -> Option<(f64, f64, f64)> {
        let base0 = base_at(t0);
        let ratio = (rail - base0) / amp; // crossing level, normalized to amp
        let xc = kernel_inverse(kind, ratio, sharpness)?;
        let w_rise = (t0 - left_cross) / xc;
        let w_fall = (right_cross - t0) / xc;
        if w_rise <= 0.0 || w_fall <= 0.0 {
            return None;
        }
        let mut sse = 0.0;
        for &(t, v) in &shoulders {
            let ws = if t < t0 { w_rise } else { w_fall };
            let model = base_at(t) + amp * kernel(kind, (t - t0) / ws, sharpness);
            sse += (model - v).powi(2);
        }
        Some((sse, w_rise, w_fall))
    };

    let gap_mid = (clip_start as f64 + clip_end as f64) / 2.0;
    let base_mid = base_at(gap_mid);
    // amp must exceed (rail − base) for a crossing to exist.
    let amp_lo = ((rail - base_mid).max(1e-3)) * 1.01;

    // Bound the peak above the rail by how LONG the signal stayed clipped, not
    // just how steeply it approached. A clip that is above the rail for only one
    // sample carries almost no evidence of a tall peak — a brief, steep graze of
    // the rail must not be reconstructed as a spike. So the overshoot grows
    // QUADRATICALLY with the clip width:
    //   overshoot = clip_w² · (s_l + s_r) / (8 · (clip_w + 1)).
    // This is the height of a parabola that is above the rail for `clip_w`
    // samples with shoulder slopes s_l/s_r, written so that for a 1-sample clip
    // the bound is ≈ (s_l+s_r)/16 (a few g even when the shoulders are steep),
    // while for a genuinely wide clip it approaches the linear clip_w·s/4 — a
    // wide clip really can hide a tall peak. Slopes are the per-sample rise of
    // the unclipped samples just outside the clip (same data the crossing
    // estimate uses); a steep spike on a single sample no longer escapes.
    let s_l = if clip_start >= 2 {
        (accel[clip_start - 1] - accel[clip_start - 2]).max(0.0)
    } else {
        0.0
    };
    let s_r = if clip_end + 2 < accel.len() {
        (accel[clip_end + 1] - accel[clip_end + 2]).max(0.0)
    } else {
        0.0
    };
    let clip_w = (clip_end - clip_start) as f64 + 1.0;
    let s_sum = s_l + s_r;
    let peak_cap = if s_sum > 1e-6 {
        rail + clip_w * clip_w * s_sum / (8.0 * (clip_w + 1.0))
    } else {
        // No usable slope (flat approach or buffer edge): the signal barely
        // exceeded the rail — allow only the minimal peak just above it.
        base_mid + amp_lo
    };
    // Cap the searchable amplitude; never below amp_lo (must still reach rail).
    let amp_cap = (peak_cap - base_mid).max(amp_lo);
    let amp_hi = (amp_lo * 6.0).min(amp_cap).max(amp_lo);

    let t0_steps = 24;
    let amp_steps = 48;
    let mut best = (f64::INFINITY, gap_mid, amp_lo, 1.0, 1.0);
    for ti in 0..=t0_steps {
        let t0 = left_cross + (right_cross - left_cross) * ti as f64 / t0_steps as f64;
        for ai in 0..=amp_steps {
            let amp = amp_lo + (amp_hi - amp_lo) * ai as f64 / amp_steps as f64;
            if let Some((sse, wr, wf)) = eval(t0, amp) {
                if sse < best.0 {
                    best = (sse, t0, amp, wr, wf);
                }
            }
        }
    }
    // Local refine around the best (t0, amp).
    let dt = (right_cross - left_cross) / t0_steps as f64;
    let da = (amp_hi - amp_lo) / amp_steps as f64;
    for ki in -3..=3 {
        let t0 = best.1 + dt * 0.25 * ki as f64;
        for kj in -3..=3 {
            let amp = best.2 + da * 0.25 * kj as f64;
            if let Some((sse, wr, wf)) = eval(t0, amp) {
                if sse < best.0 {
                    best = (sse, t0, amp, wr, wf);
                }
            }
        }
    }

    PulseFit {
        amplitude: best.2,
        t0: best.1,
        w_rise: best.3.max(1e-3),
        w_fall: best.4.max(1e-3),
        base_slope,
        base_intercept,
    }
}

/// Tunable reconstruction configuration. The shipped defaults come from
/// [default_params], chosen by the dev tuning harness (see the `harness_tests`
/// module) against real sub-limit events. The ±32 g rail is the LSM6DSO32TR
/// limit (IDL0_SPEC §3.2); revisit if the hardware moves to a ±64 g part.
#[derive(Clone, Copy, Debug)]
struct ReconstructParams {
    /// Saturation magnitude in channel units (g). Default 32.0.
    rail: f64,
    /// Pinned-detection tolerance in channel units. Default 0.05 g.
    eps: f64,
    /// Shoulder samples used per side for the fit. Default 24.
    shoulder_n: usize,
    /// Kernel family. Default Sech2 (sharp, smooth apex).
    kind: KernelKind,
    /// Kernel sharpness (GenGaussian exponent; ignored by Sech2/Lorentzian).
    sharpness: f64,
}

/// The shipped reconstruction defaults, baked in from the dev tuning harness.
///
/// A plain free function rather than a `Default` impl on purpose: flutter_rust_
/// bridge special-cases `Default` and would expose `ReconstructParams::default`
/// (dragging the struct and `KernelKind` across the FFI boundary) even though
/// the type is private. Keeping this a private `fn` ensures `declip` is the only
/// symbol that crosses to Dart.
fn default_params() -> ReconstructParams {
    ReconstructParams {
        rail: 32.0,
        eps: 0.05,
        shoulder_n: 24,
        kind: KernelKind::Sech2,
        sharpness: 2.0,
    }
}

/// Reconstructs every clipped segment in `accel` on a copy and returns the
/// result. Only pinned samples are overwritten; all other samples are
/// byte-identical to the input. Segments with too few shoulder samples to fit a
/// pulse are left pinned (see [build_fit_window]).
///
/// `sample_rate_hz` is currently unused by the fit (the kernel works in sample
/// space) but is part of the contract for future time-domain tuning and to
/// mirror the other DSP signatures.
///
/// Internal to this module — `declip` is the public entry point.
fn reconstruct_clipped(
    accel: &[f64],
    _sample_rate_hz: f64,
    params: &ReconstructParams,
) -> Vec<f64> {
    let mut out = accel.to_vec();
    let segments = find_clipped_segments(accel, params.rail, params.eps);
    for (cs, ce) in segments {
        let win = match build_fit_window(accel.len(), cs, ce, params.shoulder_n) {
            Some(w) => w,
            None => continue, // too few shoulders — leave pinned
        };

        // Reconstruct in a sign-normalised domain so the positive-pulse fit
        // machinery handles both rails. A segment pinned at −rail is negated to
        // a positive clip, reconstructed, then negated back. Without this a
        // negative clip is rebuilt as a large UPWARD spike (the rail/amplitude
        // search assumes a +rail peak). The sign is taken from the first pinned
        // sample, which sits at ±rail.
        let sign = if accel[cs] < 0.0 { -1.0 } else { 1.0 };
        let neg_buf: Vec<f64> = if sign < 0.0 {
            accel.iter().map(|v| -v).collect()
        } else {
            Vec::new()
        };
        let work: &[f64] = if sign < 0.0 { &neg_buf } else { accel };

        let fit = fit_pulse(work, &win, cs, ce, params.rail, params.kind, params.sharpness);

        // Evaluate the fitted pulse (baseline + scaled kernel) at sample `t`,
        // in the sign-normalised (work) domain.
        let value_at = |t: f64| -> f64 {
            let x = t - win.origin as f64;
            let base = fit.base_slope * x + fit.base_intercept;
            let w = if t < fit.t0 { fit.w_rise } else { fit.w_fall };
            base + fit.amplitude * kernel(params.kind, (t - fit.t0) / w, params.sharpness)
        };

        // Seam blend: linearly taper the value mismatch at each boundary so the
        // reconstructed span joins the real shoulders continuously. delta_left
        // is (real sample just left of clip − fitted value there), in work units.
        let span = (ce - cs) as f64 + 1.0;
        let delta_left = if cs > 0 {
            work[cs - 1] - value_at((cs - 1) as f64)
        } else {
            0.0
        };
        let delta_right = if ce + 1 < work.len() {
            work[ce + 1] - value_at((ce + 1) as f64)
        } else {
            0.0
        };

        for i in cs..=ce {
            let frac = if span > 1.0 {
                (i - cs) as f64 / (span - 1.0)
            } else {
                0.5
            };
            let blend = delta_left * (1.0 - frac) + delta_right * frac;
            // Undo the sign normalisation when writing back.
            out[i] = sign * (value_at(i as f64) + blend);
        }
    }
    out
}

/// Reconstructs IMU acceleration peaks clipped at the ±32 g rail. FRB entry
/// point for the `declip(ch)` math-channel function.
///
/// `accel`: acceleration samples in g (a saturating LSM6DSO32TR channel).
/// `sample_rate_hz`: channel sample rate in Hz.
///
/// Returns a same-length signal with each segment clipped at ±32 g replaced by
/// a fitted smooth asymmetric pulse, so jerk and jounce derived downstream
/// (`differentiate`) are physical. Non-clipped input is returned unchanged.
///
/// Tuned shape constants live in [default_params]. See
/// docs/superpowers/specs/2026-05-29-declip-imu-reconstruction-design.md.
pub fn declip(accel: &[f64], sample_rate_hz: f64) -> Vec<f64> {
    reconstruct_clipped(accel, sample_rate_hz, &default_params())
}

/// First difference (jerk proxy) of a sample series, same length, `[0] = 0`.
#[cfg(test)]
fn diff(series: &[f64]) -> Vec<f64> {
    let mut d = vec![0.0; series.len()];
    for i in 1..series.len() {
        d[i] = series[i] - series[i - 1];
    }
    d
}

/// Derivative-weighted reconstruction error over the index range `[lo, hi]`.
///
/// `E = w0·RMSE(a) + w1·RMSE(a′) + w2·RMSE(a″)`, each RMSE normalized by the
/// truth term's own RMS so the weights are unitless. `a′`/`a″` are first/second
/// finite differences. Returns 0.0 for an empty range.
#[cfg(test)]
fn weighted_error(
    truth: &[f64],
    recon: &[f64],
    lo: usize,
    hi: usize,
    weights: (f64, f64, f64),
) -> f64 {
    if hi < lo || hi >= truth.len() {
        return 0.0;
    }
    let d1_t = diff(truth);
    let d1_r = diff(recon);
    let d2_t = diff(&d1_t);
    let d2_r = diff(&d1_r);

    let norm_rmse = |t: &[f64], r: &[f64]| -> f64 {
        let mut se = 0.0;
        let mut ts = 0.0;
        for i in lo..=hi {
            se += (t[i] - r[i]).powi(2);
            ts += t[i].powi(2);
        }
        let n = (hi - lo + 1) as f64;
        let rmse = (se / n).sqrt();
        let scale = (ts / n).sqrt().max(1e-9);
        rmse / scale
    };

    let (w0, w1, w2) = weights;
    w0 * norm_rmse(truth, recon)
        + w1 * norm_rmse(&d1_t, &d1_r)
        + w2 * norm_rmse(&d2_t, &d2_r)
}

/// One candidate shape evaluated by the tuner.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct ShapeCandidate {
    kind: KernelKind,
    sharpness: f64,
}

/// Result of tuning: the winning candidate and its mean error, plus the full
/// scoreboard for inspection.
#[cfg(test)]
#[derive(Clone, Debug)]
struct TuneReport {
    best: ShapeCandidate,
    best_mean_error: f64,
    scoreboard: Vec<(ShapeCandidate, f64)>,
}

/// Tunes the reconstruction shape against known full (unclipped) `events`.
///
/// For each candidate shape: synthetically clip every event at `rail`,
/// reconstruct, and score with [weighted_error] over the synthetic clip span;
/// the candidate's score is the mean across events. Returns the lowest-mean
/// candidate. `weights` is the (a, a′, a″) objective weighting. Events with no
/// sample above `rail` contribute 0 (nothing to reconstruct).
///
/// Dev-only: invoked by the `harness_tests::declip_tune_report` runner to pick
/// the shipped [ReconstructParams::default]. Compiled only under `cfg(test)`,
/// so it never reaches the shipped build or the FRB boundary.
#[cfg(test)]
fn tune_shape(
    events: &[Vec<f64>],
    sample_rate_hz: f64,
    rail: f64,
    weights: (f64, f64, f64),
    candidates: &[ShapeCandidate],
) -> TuneReport {
    let mut scoreboard = Vec::new();
    for &cand in candidates {
        let mut total = 0.0;
        let mut counted = 0usize;
        for ev in events {
            let clipped: Vec<f64> = ev.iter().map(|&v| v.clamp(-rail, rail)).collect();
            let segs = find_clipped_segments(&clipped, rail, 0.05);
            if segs.is_empty() {
                continue;
            }
            let params = ReconstructParams {
                rail,
                eps: 0.05,
                shoulder_n: 24,
                kind: cand.kind,
                sharpness: cand.sharpness,
            };
            let recon = reconstruct_clipped(&clipped, sample_rate_hz, &params);
            // Score across the union of clip spans (min start .. max end).
            let lo = segs.iter().map(|s| s.0).min().unwrap();
            let hi = segs.iter().map(|s| s.1).max().unwrap();
            total += weighted_error(ev, &recon, lo, hi, weights);
            counted += 1;
        }
        let mean = if counted > 0 {
            total / counted as f64
        } else {
            f64::INFINITY
        };
        scoreboard.push((cand, mean));
    }
    scoreboard.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (best, best_mean_error) = scoreboard[0];
    TuneReport {
        best,
        best_mean_error,
        scoreboard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_clipped_segments_detects_single_positive_run() {
        // Arrange — rail 32, a 3-sample plateau at the rail in the middle
        let accel = vec![10.0, 31.99, 32.0, 32.0, 32.0, 20.0, 5.0];
        let rail = 32.0;
        let eps = 0.05;

        // Act
        let segments = find_clipped_segments(&accel, rail, eps);

        // Assert — indices 1..=4 are within eps of the rail (31.99 >= 31.95)
        assert_eq!(segments, vec![(1, 4)]);
    }

    #[test]
    fn find_clipped_segments_handles_no_clipping() {
        // Arrange
        let accel = vec![1.0, 2.0, -3.0, 4.0];

        // Act
        let segments = find_clipped_segments(&accel, 32.0, 0.05);

        // Assert
        assert!(segments.is_empty());
    }

    #[test]
    fn find_clipped_segments_handles_negative_rail_and_trailing_run() {
        // Arrange — negative plateau that runs to the end of the buffer
        let accel = vec![0.0, -32.0, -32.0, -31.96];
        let rail = 32.0;
        let eps = 0.05;

        // Act
        let segments = find_clipped_segments(&accel, rail, eps);

        // Assert — abs() handles the negative rail; run extends to last index
        assert_eq!(segments, vec![(1, 3)]);
    }

    #[test]
    fn find_clipped_segments_separates_two_runs() {
        // Arrange
        let accel = vec![32.0, 0.0, 0.0, 32.0, 32.0];

        // Act
        let segments = find_clipped_segments(&accel, 32.0, 0.05);

        // Assert
        assert_eq!(segments, vec![(0, 0), (3, 4)]);
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    #[test]
    fn kernel_peaks_at_unity_and_is_symmetric() {
        // Arrange
        let kinds = [
            KernelKind::GenGaussian,
            KernelKind::Sech2,
            KernelKind::Lorentzian,
        ];

        for kind in kinds {
            // Act
            let peak = kernel(kind, 0.0, 2.0);
            let left = kernel(kind, -0.7, 2.0);
            let right = kernel(kind, 0.7, 2.0);

            // Assert — unit peak at origin, even symmetry, decays off-center
            assert!((peak - 1.0).abs() < 1e-12, "{kind:?} peak = {peak}");
            assert!((left - right).abs() < 1e-12, "{kind:?} not symmetric");
            assert!(right < peak, "{kind:?} did not decay off-center");
        }
    }

    #[test]
    fn kernel_second_difference_is_bounded_near_apex() {
        // Arrange — sample sech² densely across the apex
        let kind = KernelKind::Sech2;
        let h = 1e-3;
        let mut max_abs_d2 = 0.0_f64;

        // Act — central second difference over [-0.5, 0.5]
        let mut u = -0.5;
        while u <= 0.5 {
            let d2 = (kernel(kind, u - h, 2.0) - 2.0 * kernel(kind, u, 2.0)
                + kernel(kind, u + h, 2.0))
                / (h * h);
            max_abs_d2 = max_abs_d2.max(d2.abs());
            u += h;
        }

        // Assert — sech'' = -2 at apex; bounded well under 10 (a cusp would blow up)
        assert!(
            max_abs_d2 < 10.0,
            "second difference unbounded near apex: {max_abs_d2}",
        );
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    // Builds a clean asymmetric sech² pulse on a flat baseline, sampled at
    // integer indices, for use as fit/reconstruct ground truth.
    pub(super) fn synth_pulse(
        n: usize,
        amp: f64,
        t0: f64,
        w_rise: f64,
        w_fall: f64,
        base: f64,
    ) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                let w = if t < t0 { w_rise } else { w_fall };
                base + amp * kernel(KernelKind::Sech2, (t - t0) / w, 2.0)
            })
            .collect()
    }

    #[test]
    fn fit_pulse_recovers_known_amplitude_and_peak() {
        // Arrange — a known pulse peaking at index 25, amp 50 on baseline 2
        let truth = synth_pulse(50, 50.0, 25.0, 4.0, 6.0, 2.0);
        // Clip it at rail 32 so the apex region is removed from the fit window
        let rail = 32.0;
        let clipped: Vec<f64> = truth.iter().map(|&v| v.clamp(-rail, rail)).collect();
        let segs = find_clipped_segments(&clipped, rail, 0.05);
        let (cs, ce) = segs[0];
        let win = build_fit_window(clipped.len(), cs, ce, 24).unwrap();

        // Act
        let fit = fit_pulse(&clipped, &win, cs, ce, rail, KernelKind::Sech2, 2.0);

        // Assert — conservative recovery policy: the fit recovers most of the
        // lost peak (well above the rail) and locates it correctly, but does NOT
        // overshoot past the true peak. Exact recovery of the true 52 is not
        // required — the width-scaled cap deliberately under-recovers rather than
        // risk a spike (the clip here is only ~7 samples wide). See fit_pulse.
        let recovered_peak = fit.base_intercept
            + fit.base_slope * (fit.t0 - win.origin as f64)
            + fit.amplitude;
        assert!(
            recovered_peak > rail + 8.0,
            "peak under-recovered (should clear rail+8): {recovered_peak}",
        );
        assert!(
            recovered_peak <= 52.0 + 2.0,
            "peak overshot past truth 52: {recovered_peak}",
        );
        assert!((fit.t0 - 25.0).abs() < 2.0, "t0 {} expected ~25", fit.t0);
    }
}

#[cfg(test)]
mod reconstruct_tests {
    use super::*;
    use super::fit_tests::synth_pulse;

    #[test]
    fn reconstruct_recovers_clipped_peak_within_tolerance() {
        // Arrange — known pulse peaking ~49 g, clipped at 32 g
        let truth = synth_pulse(60, 48.0, 30.0, 4.0, 6.0, 1.0);
        let rail = 32.0;
        let clipped: Vec<f64> = truth.iter().map(|&v| v.clamp(-rail, rail)).collect();
        let params = default_params();

        // Act
        let out = reconstruct_clipped(&clipped, 1000.0, &params);

        // Assert — reconstructed peak recovers most of the lost overshoot
        let truth_peak = truth.iter().cloned().fold(f64::MIN, f64::max);
        let out_peak = out.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            out_peak > rail + 5.0,
            "peak not reconstructed above rail: {out_peak}",
        );
        assert!(
            (out_peak - truth_peak).abs() < 8.0,
            "peak {out_peak} far from truth {truth_peak}",
        );
    }

    #[test]
    fn reconstruct_leaves_unclipped_samples_unchanged() {
        // Arrange — nothing reaches the rail
        let accel = vec![1.0, -2.0, 3.0, -4.0, 5.0];

        // Act
        let out = reconstruct_clipped(&accel, 1000.0, &default_params());

        // Assert
        assert_eq!(out, accel);
    }

    #[test]
    fn reconstruct_skips_segment_at_buffer_edge() {
        // Arrange — clip touches index 0; only a right shoulder exists
        let accel = vec![32.0, 32.0, 20.0, 10.0, 5.0];

        // Act
        let out = reconstruct_clipped(&accel, 1000.0, &default_params());

        // Assert — does not panic and preserves length
        assert_eq!(out.len(), accel.len());
    }

    #[test]
    fn reconstruct_negative_clip_rebuilds_downward_not_a_spike() {
        // Arrange — downward pulse, true trough −48, clipped at −32.
        let truth = synth_pulse(60, -48.0, 30.0, 4.0, 6.0, 0.0);
        let rail = 32.0;
        let clipped: Vec<f64> = truth.iter().map(|&v| v.clamp(-rail, rail)).collect();

        // Act
        let out = reconstruct_clipped(&clipped, 1000.0, &default_params());

        // Assert — reconstructed BELOW the negative rail, near truth, and no
        // spurious large positive excursion (the old sign bug produced +175).
        let out_min = out.iter().cloned().fold(f64::MAX, f64::min);
        let out_max = out.iter().cloned().fold(f64::MIN, f64::max);
        assert!(out_min < -rail - 5.0, "trough not rebuilt: out_min={out_min}");
        assert!((out_min - (-48.0)).abs() < 8.0, "trough far from truth: {out_min}");
        assert!(out_max < rail + 5.0, "spurious positive spike: out_max={out_max}");
    }

    #[test]
    fn reconstruct_broad_flat_top_does_not_overshoot() {
        // Arrange — a wide, flat-topped pulse (true peak 36) clipped at 32. The
        // pointy kernel previously overshot to ~66; the slope cap must keep the
        // reconstructed peak physically bounded.
        let truth: Vec<f64> = (0..60)
            .map(|i| {
                let u = (i as f64 - 30.0) / 8.0;
                36.0 * (-(u * u).powf(2.0)).exp()
            })
            .collect();
        let rail = 32.0;
        let clipped: Vec<f64> = truth.iter().map(|&v| v.clamp(-rail, rail)).collect();

        // Act
        let out = reconstruct_clipped(&clipped, 1000.0, &default_params());

        // Assert — peak stays well under the old 66 overshoot (within ~1.4× truth).
        let out_peak = out.iter().cloned().fold(f64::MIN, f64::max);
        let truth_peak = truth.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            out_peak < truth_peak * 1.4,
            "overshoot not contained: out_peak={out_peak}, truth_peak={truth_peak}",
        );
        assert!(out_peak >= rail, "reconstruction dropped below the rail: {out_peak}");
    }

    #[test]
    fn reconstruct_steep_single_point_does_not_spike() {
        // A single sample clips at the top of a sharp spike with steep shoulders
        // — the real-world failure mode. Because the signal is above the rail for
        // only one sample, the reconstruction must add only a small overshoot,
        // not extrapolate a tall spike from the steep edge slope.
        let rail = 32.0;
        // Steep symmetric approach: 0 → 5 → 31 → [clip] → 31 → 5 → 0; true apex 34.
        let v: Vec<f64> = vec![0.0, 0.0, 5.0, 31.0, 34.0, 31.0, 5.0, 0.0, 0.0];
        let clipped: Vec<f64> = v.iter().map(|&x| x.clamp(-rail, rail)).collect();

        // Act
        let out = reconstruct_clipped(&clipped, 866.0, &default_params());
        let out_peak = out.iter().cloned().fold(f64::MIN, f64::max);

        // Assert — clears the rail (something was reconstructed) but no spike:
        // well under 1.5× the rail (the old linear cap reached ~46–48 here).
        assert!(out_peak >= rail, "did not reconstruct above rail: {out_peak}");
        assert!(
            out_peak < rail + 8.0,
            "single-point clip spiked: out_peak={out_peak}",
        );
    }
}

#[cfg(test)]
mod harness_tests {
    use super::*;
    use super::fit_tests::synth_pulse;

    #[test]
    fn weighted_error_is_zero_for_identical_series() {
        // Arrange
        let s = synth_pulse(40, 30.0, 20.0, 4.0, 5.0, 0.0);

        // Act
        let e = weighted_error(&s, &s, 5, 35, (1.0, 1.0, 1.0));

        // Assert
        assert!(e.abs() < 1e-9, "expected 0 error, got {e}");
    }

    #[test]
    fn weighted_error_grows_with_peak_distortion() {
        // Arrange — truth vs a flattened (clipped-then-not-reconstructed) copy
        let truth = synth_pulse(40, 48.0, 20.0, 4.0, 5.0, 0.0);
        let flat: Vec<f64> = truth.iter().map(|&v| v.min(32.0)).collect();

        // Act
        let e_good = weighted_error(&truth, &truth, 5, 35, (1.0, 1.0, 1.0));
        let e_bad = weighted_error(&truth, &flat, 5, 35, (1.0, 1.0, 1.0));

        // Assert
        assert!(e_bad > e_good, "distortion did not raise error");
    }

    #[test]
    fn tune_shape_prefers_a_shape_that_reconstructs_the_known_event() {
        // Arrange — events that are genuinely sech²-shaped above the rail
        let events: Vec<Vec<f64>> = (0..3)
            .map(|k| synth_pulse(60, 44.0 + k as f64 * 2.0, 30.0, 4.0, 6.0, 0.0))
            .collect();
        let candidates = [
            ShapeCandidate {
                kind: KernelKind::Sech2,
                sharpness: 2.0,
            },
            ShapeCandidate {
                kind: KernelKind::GenGaussian,
                sharpness: 2.0,
            },
            ShapeCandidate {
                kind: KernelKind::Lorentzian,
                sharpness: 2.0,
            },
        ];

        // Act
        let report = tune_shape(&events, 1000.0, 32.0, (1.0, 0.5, 0.25), &candidates);

        // Assert — a finite winner is chosen, no worse than the worst candidate
        assert!(report.best_mean_error.is_finite());
        let worst = report.scoreboard.last().unwrap().1;
        assert!(report.best_mean_error <= worst);
    }

    /// Dev tuning loop. Not run in normal CI (ignored). Run with:
    ///   IDL0_DECLIP_EVENTS=path/to/events.csv \
    ///   cargo test --lib clip_reconstruct::harness_tests::declip_tune_report \
    ///     -- --ignored --nocapture
    /// CSV: one event per line, comma-separated g samples. Without the env var
    /// it falls back to synthetic events so the test is always runnable.
    #[test]
    #[ignore]
    fn declip_tune_report() {
        // Arrange
        let events: Vec<Vec<f64>> = match std::env::var("IDL0_DECLIP_EVENTS") {
            Ok(path) => {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
                text.lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| {
                        l.split(',')
                            .filter_map(|s| s.trim().parse::<f64>().ok())
                            .collect()
                    })
                    .collect()
            }
            Err(_) => (0..5)
                .map(|k| synth_pulse(80, 42.0 + k as f64 * 3.0, 40.0, 4.0 + k as f64, 6.0, 0.0))
                .collect(),
        };
        let candidates = [
            ShapeCandidate {
                kind: KernelKind::Sech2,
                sharpness: 2.0,
            },
            ShapeCandidate {
                kind: KernelKind::Lorentzian,
                sharpness: 2.0,
            },
            ShapeCandidate {
                kind: KernelKind::GenGaussian,
                sharpness: 2.0,
            },
            ShapeCandidate {
                kind: KernelKind::GenGaussian,
                sharpness: 2.5,
            },
            ShapeCandidate {
                kind: KernelKind::GenGaussian,
                sharpness: 3.0,
            },
        ];
        let weights = (1.0, 0.5, 0.25);

        // Act
        let report = tune_shape(&events, 1000.0, 32.0, weights, &candidates);

        // Assert / report
        println!("\n=== declip tune report ({} events) ===", events.len());
        for (cand, err) in &report.scoreboard {
            println!(
                "  {:?} sharpness={:.2} -> mean E {:.4}",
                cand.kind, cand.sharpness, err
            );
        }
        println!(
            "BEST: {:?} sharpness={:.2} (E {:.4})",
            report.best.kind, report.best.sharpness, report.best_mean_error,
        );
        assert!(report.best_mean_error.is_finite());
    }
}
