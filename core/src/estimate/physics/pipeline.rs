//! A session into the physics estimator: the three IMUs on one uniform grid
//! (the IEKF's [`EstimatorInput`], each IMU resampled by its own recorded
//! clock), native-rate clip masks, airborne detection, at-rest biases from a
//! still recording, and the choice of windows. Port of `ml/idlml/data/`.

use super::objective::{Window, NG};
use crate::estimate::run::EstimatorInput;
use crate::math::eval::ChannelLookup;
use crate::session::column::RawColumn;
use crate::session::handle::SessionHandle;

/// Standard gravity, m/s^2.
const G: f64 = 9.80665;
/// A raw int16 sample at or beyond this fraction of full scale (32767) is clipped.
pub const CLIP_FRAC: f64 = 0.98;
/// A grid sample within this of a clipped native sample is masked, us: the
/// resampling and anti-alias filtering smear a clipped plateau into its neighbours.
pub const CLIP_SMEAR_US: i64 = 10_000;

/// A ride ready for the estimator.
pub struct Ride {
    /// The IMUs on IMU0's uniform grid, SI units, each in its own axes.
    pub input: EstimatorInput,
    /// Grid times, us since the session's first sample.
    pub grid_us: Vec<i64>,
    /// Per IMU, per grid sample: clipped (or near a clipped native sample).
    pub clip: [Vec<bool>; 3],
}

impl Ride {
    /// Samples on the grid.
    pub fn len(&self) -> usize {
        self.grid_us.len()
    }
    /// Whether the ride has no samples.
    pub fn is_empty(&self) -> bool {
        self.grid_us.is_empty()
    }
    /// Sample rate, Hz.
    pub fn rate(&self) -> f64 {
        1.0 / self.input.dt
    }
    /// Grid index of session time `t_s` (s since the first sample), clamped.
    pub fn index_at(&self, t_s: f64) -> usize {
        self.grid_us.partition_point(|&t| (t as f64) < t_s * 1e6).min(self.len())
    }
}

/// Why a session cannot drive the estimator.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// No IMU0, or its times span nothing.
    NoFrameImu,
    /// The fork pod (IMU1) or rear pod (IMU2) was not recorded.
    MissingPod {
        /// The IMU index missing.
        imu_index: u8,
    },
}

/// The session's IMUs on IMU0's grid, with clip masks taken at each IMU's
/// native rate on its raw int16 counts.
pub fn ride(h: &SessionHandle) -> Result<Ride, PipelineError> {
    ride_between(h, None)
}

/// As [`ride`], over session times `span` (us since the first sample) only:
/// every channel is cut to the span before it is resampled, so the grid, the
/// windows and everything after hold only that stretch.
pub fn ride_between(h: &SessionHandle, span: Option<(i64, i64)>) -> Result<Ride, PipelineError> {
    let input = match span {
        Some((a, b)) => EstimatorInput::from_lookup(&Cropped { inner: h, from_us: a, to_us: b }),
        None => EstimatorInput::from_lookup(h),
    }
    .ok_or(PipelineError::NoFrameImu)?;
    let mut input = input;
    antialias(&mut input);
    for (k, s) in [(1u8, &input.imu1), (2, &input.imu2)] {
        if s.is_none() {
            return Err(PipelineError::MissingPod { imu_index: k });
        }
    }
    let n = input.imu0.gyro.len();
    let grid_us: Vec<i64> = (0..n).map(|i| ((input.t0_s + i as f64 * input.dt) * 1e6).round() as i64).collect();
    let clip = std::array::from_fn(|k| clip_mask(h, k, &grid_us));
    Ok(Ride { input, grid_us, clip })
}

/// Zero-phase 4th-order Butterworth low-pass of every IMU axis at 0.4 x the
/// grid rate, as the PyTorch reference's cache does: the IEKF's resampler
/// filters only streams faster than the grid, so the pods (about the grid's
/// rate) would otherwise bring their 60-230 Hz vibration, which the model does
/// not describe, into the fit.
fn antialias(input: &mut EstimatorInput) {
    let rate = 1.0 / input.dt;
    let mut series = vec![&mut input.imu0];
    series.extend(input.imu1.as_mut());
    series.extend(input.imu2.as_mut());
    for s in series {
        for v in [&mut s.gyro, &mut s.accel] {
            for a in 0..3 {
                let x: Vec<f64> = v.iter().map(|p| p[a]).collect();
                if let Ok(y) = crate::filters::lowpass(&x, 4, 0.4 * rate, rate) {
                    for (p, y) in v.iter_mut().zip(y) {
                        p[a] = y;
                    }
                }
            }
        }
    }
}

/// A lookup that serves its inner lookup's channels cut to `from_us..to_us`.
pub struct Cropped<'a> {
    /// The session.
    pub inner: &'a SessionHandle,
    /// First time kept, us since the session's first sample.
    pub from_us: i64,
    /// Last time kept, us.
    pub to_us: i64,
}

impl ChannelLookup for Cropped<'_> {
    fn lookup(&self, name: &str) -> Option<crate::math::eval::LookupChannel> {
        let c = self.inner.lookup(name)?;
        if c.t_us.is_empty() {
            return Some(c);
        }
        let a = c.t_us.partition_point(|&t| t < self.from_us);
        let b = c.t_us.partition_point(|&t| t <= self.to_us);
        Some(crate::math::eval::LookupChannel { samples: c.samples[a..b].into(), sample_rate_hz: c.sample_rate_hz, t_us: c.t_us[a..b].into() })
    }
}

/// IMU `k`'s clip mask on `grid_us`: any int16 axis at full scale within
/// [`CLIP_SMEAR_US`] of the grid time. Float channels (e.g. IMU0's high-g
/// accel) have no clip level and never mask.
fn clip_mask(h: &SessionHandle, k: usize, grid_us: &[i64]) -> Vec<bool> {
    let mut clipped: Vec<i64> = Vec::new();
    for q in ["Accel", "Gyro"] {
        for a in ["X", "Y", "Z"] {
            let name = format!("IMU{k}_{q}{a}");
            h.with_channel(&name, |c| {
                if let RawColumn::I16 { data, .. } = &c.column {
                    let lim = (CLIP_FRAC * 32767.0) as i32;
                    clipped.extend(data.iter().zip(&c.t_us).filter(|(&v, _)| (v as i32).abs() >= lim).map(|(_, &t)| t));
                }
            });
        }
    }
    clipped.sort_unstable();
    grid_us
        .iter()
        .map(|&t| {
            let i = clipped.partition_point(|&c| c < t - CLIP_SMEAR_US);
            i < clipped.len() && clipped[i] <= t + CLIP_SMEAR_US
        })
        .collect()
}

fn v3(v: &nalgebra::Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// The estimator's window over grid samples `i0..i1`.
pub fn window(r: &Ride, i0: usize, i1: usize, air: bool) -> Window {
    let inp = &r.input;
    let (p1, p2) = (inp.imu1.as_ref().unwrap(), inp.imu2.as_ref().unwrap());
    let rows = |s: &[nalgebra::Vector3<f64>]| s[i0..i1].iter().map(v3).collect::<Vec<_>>();
    let frame_ok: Vec<bool> = (i0..i1).map(|i| !r.clip[0][i]).collect();
    let mask = |k: usize| (i0..i1).zip(&frame_ok).map(|(i, &ok)| ok && !r.clip[k][i]).collect();
    Window {
        rate: r.rate(),
        frame_gyro: rows(&inp.imu0.gyro),
        frame_accel: rows(&inp.imu0.accel),
        fork_gyro: rows(&p1.gyro),
        fork_accel: rows(&p1.accel),
        rear_gyro: rows(&p2.gyro),
        rear_accel: rows(&p2.accel),
        mask_fork: mask(1),
        mask_rear: mask(2),
        air: if air { Some(airborne(r, i0, i1).0) } else { None },
        nav: None,
    }
}

/// Airborne detection over grid samples `i0..i1` (port of `real.airborne`):
/// the frame's specific force (10 Hz low-pass) under 0.6 g AND both unsprung
/// pods quiet (30-150 Hz accel RMS envelope under 1.5 m/s^2: no tire contact
/// noise) for at least 0.12 s; each event's first 80 ms (the fork may still be
/// rebounding) and last 30 ms (filters smear the landing) are not anchored.
/// The 30-150 Hz band is a 4th-order high-pass then a 4th-order low-pass (the
/// reference uses a 4th-order band-pass design; the thresholds are coarse).
/// Returns (anchor mask, events as (start, end) indices within the window).
pub fn airborne(r: &Ride, i0: usize, i1: usize) -> (Vec<bool>, Vec<(usize, usize)>) {
    let rate = r.rate();
    let inp = &r.input;
    let n = i1 - i0;
    let lp = |x: &[f64], order: usize, fc: f64| crate::filters::lowpass(x, order, fc, rate).unwrap_or_else(|_| x.to_vec());
    let f: Vec<f64> = inp.imu0.accel[i0..i1].iter().map(|v| v.norm()).collect();
    let mut ok: Vec<bool> = lp(&f, 2, 10.0).iter().map(|&x| x < 0.6 * G).collect();
    let hi = (0.45 * rate).min(150.0);
    for pod in [inp.imu1.as_ref().unwrap(), inp.imu2.as_ref().unwrap()] {
        let mut energy = vec![0.0; n];
        for a in 0..3 {
            let x: Vec<f64> = pod.accel[i0..i1].iter().map(|v| v[a]).collect();
            let hp = crate::filters::highpass(&x, 4, 30.0, rate).unwrap_or_else(|_| x.clone());
            let band = lp(&hp, 4, hi);
            for (e, b) in energy.iter_mut().zip(&band) {
                *e += b * b;
            }
        }
        for (o, e) in ok.iter_mut().zip(lp(&energy, 2, 20.0)) {
            *o &= e.max(0.0).sqrt() < 1.5;
        }
    }
    let (min_n, settle, trim) = ((0.12 * rate) as usize, (0.08 * rate) as usize, (0.03 * rate) as usize);
    let mut mask = vec![false; n];
    let mut events = Vec::new();
    let mut i = 0;
    while i < n {
        if !ok[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < n && ok[i] {
            i += 1;
        }
        if i - s >= min_n {
            events.push((s, i));
            for m in mask.iter_mut().take(i.saturating_sub(trim)).skip(s + settle) {
                *m = true;
            }
        }
    }
    (mask, events)
}

/// At-rest gyro biases from a still recording (the session the calibration's
/// still window came from), as the bias globals' prior centres: each IMU's
/// mean gyro after its first second; accel biases centred on 0.
pub fn still_bias_centre(still: &dyn ChannelLookup) -> Option<[f64; NG]> {
    let inp = EstimatorInput::from_lookup(still)?;
    let skip = (1.0 / inp.dt) as usize;
    let mut c = [0.0; NG];
    for (k, s) in [Some(&inp.imu0), inp.imu1.as_ref(), inp.imu2.as_ref()].into_iter().enumerate() {
        let s = s?;
        let g = &s.gyro[skip.min(s.gyro.len().saturating_sub(1))..];
        let off = [0, 3, 9][k];
        for a in 0..3 {
            c[off + a] = g.iter().map(|v| v[a]).sum::<f64>() / g.len() as f64;
        }
    }
    Some(c)
}

/// Greedy: the `count` non-overlapping `n`-sample windows with the highest
/// mean `score`, none touching an `exclude` range (or another chosen window).
pub fn pick_windows(score: &[f64], n: usize, count: usize, exclude: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if score.len() < n || n == 0 {
        return Vec::new();
    }
    let mut cs = vec![0.0; score.len() + 1];
    for (i, s) in score.iter().enumerate() {
        cs[i + 1] = cs[i] + s;
    }
    let mean: Vec<f64> = (0..=score.len() - n).map(|i| (cs[i + n] - cs[i]) / n as f64).collect();
    let mut order: Vec<usize> = (0..mean.len()).collect();
    order.sort_by(|&a, &b| mean[b].partial_cmp(&mean[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut taken: Vec<(usize, usize)> = exclude.to_vec();
    let mut out = Vec::new();
    for i in order {
        if out.len() == count || mean[i] <= 0.0 {
            break;
        }
        let (a, b) = (i, i + n);
        if taken.iter().any(|&(x, y)| a < y && x < b) {
            continue;
        }
        out.push((a, b));
        taken.push((a.saturating_sub(n), b + n));
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_windows_takes_the_best_non_overlapping_windows_outside_the_exclusions() {
        // Arrange: activity peaks at 100..200 and 600..700; 100..200 is excluded
        let mut score = vec![0.0; 1000];
        score[100..200].iter_mut().for_each(|s| *s = 5.0);
        score[600..700].iter_mut().for_each(|s| *s = 3.0);
        score[300..350].iter_mut().for_each(|s| *s = 1.0);

        // Act
        let w = pick_windows(&score, 100, 2, &[(90, 210)]);

        // Assert: the excluded peak is skipped; the second window does not overlap the first
        assert_eq!(w[0].1 - w[0].0, 100);
        assert!(w.contains(&(600, 700)));
        assert!(w.iter().all(|&(a, b)| b <= 90 || a >= 210));
        assert_eq!(w.len(), 2);
    }
}

/// Per-sample frame rotation rate, rad/s: what makes a window informative for geometry.
pub fn rotation_score(r: &Ride) -> Vec<f64> {
    r.input.imu0.gyro.iter().map(|v| v.norm()).collect()
}

/// Per-sample airborne flag over the whole ride (in chunks), as 0/1.
pub fn airtime_score(r: &Ride) -> Vec<f64> {
    let chunk = (600.0 * r.rate()) as usize;
    let pad = (2.0 * r.rate()) as usize;
    let n = r.len();
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let (a, b) = (i.saturating_sub(pad), (i + chunk + pad).min(n));
        let (m, _) = airborne(r, a, b);
        for j in i..(i + chunk).min(n) {
            out[j] = m[j - a] as u8 as f64;
        }
        i += chunk;
    }
    out
}

/// Samples where the bike is being ridden: GPS speed over 3 m/s, held for
/// 2 s either side (no GPS: everything). The estimator solves only these.
pub fn moving(r: &Ride) -> Vec<bool> {
    let n = r.len();
    let gps = &r.input.gps;
    if gps.is_empty() {
        return vec![true; n];
    }
    let mut out = vec![false; n];
    let hold = (2.0 * r.rate()) as usize;
    for g in gps.iter().filter(|g| g.velocity.norm() > 3.0) {
        let i = r.index_at(g.time_s); // GPS fix times share the grid's timeline
        for m in out.iter_mut().take((i + hold).min(n)).skip(i.saturating_sub(hold)) {
            *m = true;
        }
    }
    out
}
