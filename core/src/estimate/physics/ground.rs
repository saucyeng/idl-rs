//! One ground under both tyres (force-model spec `2026-11-25-idl1-force-model-DRAFT.md`, F4).
//!
//! Both tyres roll over the same surface, so a contact patch's height in the world, at its
//! distance along the path, is the same for the rear as it was for the front a wheelbase
//! earlier. The ground is a profile of its own, h(s), a uniform cubic B-spline in distance;
//! both tyres are tied to it alike (Huber, so a line change or a rock that moves costs a capped
//! amount). The profile is not a Gauss-Newton unknown: it would join the two ends of the
//! delay in the solver's band. It is refitted to both tyres' patch heights before each
//! iteration and held fixed through the step ([`fit_profile`]).
//!
//! The heights and distance come from a navigation solve run first in the window
//! ([`GroundWindow`], built by [`ground_window`]), then held fixed in the force solve.

use super::la::{M3, V3};

/// The ground tie's settings (Isaac, 2026-11-27: 5 mm, a cross-slope of 0.2).
#[derive(Clone, Debug)]
pub struct GroundWeights {
    /// The tie's sigma where both lines can be the same, m.
    pub sigma_m: f64,
    /// Height change per metre of lateral offset between the wheels' lines, m/m: what a line
    /// change costs in height on a rough trail.
    pub cross_slope: f64,
    /// Navigation's vertical velocity uncertainty, m/s: its height drift over the delay.
    pub nav_sigma_mps: f64,
    /// Below this speed the tie is off, m/s (the delay gets long, the position drifts).
    pub min_speed_mps: f64,
    /// A tyre with less normal force than this at the last iterate is off the ground, N.
    pub min_normal_n: f64,
    /// The profile's knot spacing, m (a 29er's patch is about 10 cm long).
    pub knot_m: f64,
    /// The profile's second-difference prior per knot, m.
    pub curvature_m: f64,
}

impl Default for GroundWeights {
    fn default() -> Self {
        GroundWeights { sigma_m: 0.005, cross_slope: 0.2, nav_sigma_mps: 0.05, min_speed_mps: 1.5, min_normal_n: 20.0, knot_m: 0.05, curvature_m: 0.02 }
    }
}

/// A window's ground-tie inputs, per sample, from its navigation solve.
#[derive(Clone, Debug)]
pub struct GroundWindow {
    /// The world's up in IMU0 axes (unit).
    pub up: Vec<[f64; 3]>,
    /// The frame pod's height, window-local, m.
    pub z_m: Vec<f64>,
    /// The frame pod's distance along its path, window-local, m.
    pub s_m: Vec<f64>,
    /// The tie's sigma per tyre (front, rear), m; infinite: off.
    pub sigma_m: Vec<[f64; 2]>,
    /// Each patch's distance ahead of the frame pod along the path (front, rear), m.
    pub offset_m: [f64; 2],
}

/// The ground profile at an iterate, and which tyres were on it.
#[derive(Clone, Debug)]
pub struct GroundFit {
    /// Distance of coefficient 0's knot, m.
    pub s0_m: f64,
    /// Knot spacing, m.
    pub knot_m: f64,
    /// Coefficients, m.
    pub coef: Vec<f64>,
    /// Per sample, whether each tyre (front, rear) was on the ground.
    pub on: Vec<[bool; 2]>,
}

/// Uniform cubic B-spline weights at fraction `u` of a span.
fn weights(u: f64) -> [f64; 4] {
    let (u2, u3) = (u * u, u * u * u);
    [(1.0 - u).powi(3) / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0, (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0]
}

impl GroundFit {
    /// The first coefficient and the four weights at distance `s` (m), clamped to the profile.
    pub fn span(&self, s: f64) -> (usize, [f64; 4]) {
        let x = ((s - self.s0_m) / self.knot_m).clamp(0.0, (self.coef.len() - 3) as f64 - 1e-9);
        let k = x.floor() as usize;
        (k, weights(x - k as f64))
    }

    /// The ground's height at distance `s` (m), m.
    pub fn height(&self, s: f64) -> f64 {
        let (k, w) = self.span(s);
        (0..4).map(|i| w[i] * self.coef[k + i]).sum()
    }
}

/// The profile fit's faint pull of every coefficient to the patches' mean height, 1/m^2.
pub const RIDGE: f64 = 1e-6;

/// One patch height for the profile's fit: distance (m), height (m), sigma (m).
#[derive(Clone, Copy, Debug)]
pub struct Patch {
    /// Distance along the path, m.
    pub s_m: f64,
    /// Height, m.
    pub z_m: f64,
    /// The tie's sigma, m.
    pub sigma_m: f64,
}

/// The profile that best fits `patches` (both tyres' heights against distance), weighted by
/// their sigmas, Huber at `huber` sigmas by reweighting, with a second-difference prior of
/// `curvature_m` per knot. None with no patch. A banded solve (half-width 3).
pub fn fit_profile(patches: &[Patch], knot_m: f64, curvature_m: f64, huber: f64) -> Option<(f64, Vec<f64>)> {
    if patches.is_empty() {
        return None;
    }
    let (lo, hi) = patches.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.s_m), a.1.max(p.s_m)));
    let s0 = lo - knot_m;
    let k = ((hi - s0) / knot_m).ceil() as usize + 3;
    let mean = patches.iter().map(|p| p.z_m).sum::<f64>() / patches.len() as f64;
    let mut fit = GroundFit { s0_m: s0, knot_m, coef: vec![mean; k], on: Vec::new() };
    let wd = 4;
    for pass in 0..4 {
        let mut a = vec![0.0; k * wd]; // a[i * wd + d]: row i, column i - d (lower band)
        let mut b = vec![0.0; k];
        for p in patches {
            let (j, w) = fit.span(p.s_m);
            let mut wt = 1.0 / (p.sigma_m * p.sigma_m);
            if pass > 0 && huber > 0.0 {
                let r = ((fit.height(p.s_m) - p.z_m) / p.sigma_m).abs();
                if r > huber {
                    wt *= huber / r;
                }
            }
            for x in 0..4 {
                b[j + x] += wt * w[x] * p.z_m;
                for y in 0..=x {
                    a[(j + x) * wd + (x - y)] += wt * w[x] * w[y];
                }
            }
        }
        let wc = 1.0 / (curvature_m * curvature_m);
        for i in 0..k - 2 {
            let w = [1.0, -2.0, 1.0];
            for x in 0..3 {
                for y in 0..=x {
                    a[(i + x) * wd + (x - y)] += wc * w[x] * w[y];
                }
            }
        }
        // a faint pull to the mean height: knots no patch reaches stay defined
        for i in 0..k {
            a[i * wd] += RIDGE;
            b[i] += RIDGE * mean;
        }
        let ch = BandChol::new(a, k, wd - 1)?;
        ch.solve(&mut b);
        fit.coef = b;
    }
    Some((s0, fit.coef))
}

/// A symmetric positive-definite banded matrix's Cholesky factor (lower band `a[i * (bw + 1) +
/// d]` = L[i][i - d]), for solving it again and again.
pub struct BandChol {
    l: Vec<f64>,
    n: usize,
    bw: usize,
}

impl BandChol {
    /// Factors the lower band `a` (`a[i * (bw + 1) + d]` = A[i][i - d]); None if it is not
    /// positive definite.
    pub fn new(mut a: Vec<f64>, n: usize, bw: usize) -> Option<BandChol> {
        band_factor(&mut a, n, bw)?;
        Some(BandChol { l: a, n, bw })
    }

    /// The matrix's size.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Solves A x = b in place.
    pub fn solve(&self, b: &mut [f64]) {
        band_subst(&self.l, b, self.n, self.bw);
    }
}

fn band_factor(a: &mut [f64], n: usize, bw: usize) -> Option<()> {
    let wd = bw + 1;
    for i in 0..n {
        for d in (0..=bw.min(i)).rev() {
            let j = i - d;
            let mut s = a[i * wd + d];
            for e in 1..=(bw - d).min(j) {
                s -= a[i * wd + d + e] * a[j * wd + e];
            }
            if d == 0 {
                if s <= 0.0 {
                    return None;
                }
                a[i * wd] = s.sqrt();
            } else {
                a[i * wd + d] = s / a[j * wd];
            }
        }
    }
    Some(())
}

fn band_subst(a: &[f64], b: &mut [f64], n: usize, bw: usize) {
    let wd = bw + 1;
    for i in 0..n {
        let mut s = b[i];
        for d in 1..=bw.min(i) {
            s -= a[i * wd + d] * b[i - d];
        }
        b[i] = s / a[i * wd];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for d in 1..=bw.min(n - 1 - i) {
            s -= a[(i + d) * wd + d] * b[i + d];
        }
        b[i] = s / a[i * wd];
    }
}

/// One sample of a navigation solve, as the ground tie reads it.
#[derive(Clone, Copy, Debug)]
pub struct NavPoint {
    /// IMU0 -> nav (X north, Y west, Z up).
    pub r: M3<f64>,
    /// Velocity, nav, m/s.
    pub v: [f64; 3],
    /// The frame's angular rate, IMU0 axes, rad/s.
    pub w0: V3<f64>,
}

/// The tie's inputs from a window's navigation solve: height and distance integrated
/// (trapezoid, window-local), the world's up in IMU0 axes, and the tie's sigma per tyre
/// (spec F4: the base sigma, the lateral offset between the wheels' lines in a turn and in a
/// slide times the cross-slope, navigation's drift over the delay, root-sum-square; infinite
/// below the minimum speed). `fwd`: the bike's forward axis; `offset_m`: each patch's distance
/// ahead of the frame pod (front, rear).
pub fn ground_window(nav: &[NavPoint], dt: f64, fwd: V3<f64>, offset_m: [f64; 2], w: &GroundWeights) -> GroundWindow {
    let n = nav.len();
    let wheelbase = offset_m[0] - offset_m[1];
    let mut out = GroundWindow { up: Vec::with_capacity(n), z_m: Vec::with_capacity(n), s_m: Vec::with_capacity(n), sigma_m: Vec::with_capacity(n), offset_m };
    let speed = |q: &NavPoint| V3(q.v).norm();
    // the slip between the bike's heading and its velocity, less the window's median: a steady
    // offset is navigation's heading (or the mount's), not the bike sliding (Beacon and 11/21: 4-14
    // degrees on average, which no rider holds)
    let slip = |p: &NavPoint| {
        let f = p.r * fwd;
        (f.0[0] * p.v[1] - f.0[1] * p.v[0]).atan2(f.0[0] * p.v[0] + f.0[1] * p.v[1])
    };
    let mut moving: Vec<f64> = nav.iter().filter(|p| speed(p) >= w.min_speed_mps).map(slip).collect();
    moving.sort_by(f64::total_cmp);
    let bias = moving.get(moving.len() / 2).copied().unwrap_or(0.0);
    let (mut z, mut s) = (0.0, 0.0);
    for (t, p) in nav.iter().enumerate() {
        if t > 0 {
            z += 0.5 * (nav[t - 1].v[2] + p.v[2]) * dt;
            s += 0.5 * (speed(&nav[t - 1]) + speed(p)) * dt;
        }
        let up = [p.r.0[2][0], p.r.0[2][1], p.r.0[2][2]];
        let v = speed(p);
        let sigma = if v < w.min_speed_mps {
            f64::INFINITY
        } else {
            let yaw = p.w0.dot(V3(up));
            let beta = (slip(p) - bias).abs();
            let lateral = wheelbase * wheelbase * yaw.abs() / (2.0 * v) + wheelbase * beta.min(std::f64::consts::FRAC_PI_2);
            let drift = w.nav_sigma_mps * wheelbase / v;
            (w.sigma_m.powi(2) + (w.cross_slope * lateral).powi(2) + drift * drift).sqrt()
        };
        out.up.push(up);
        out.z_m.push(z);
        out.s_m.push(s);
        out.sigma_m.push([sigma; 2]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_profile_two_tyres_on_one_bumpy_ground_recovers_it_and_caps_an_outlier() {
        // Arrange: a ground of two bumps, both tyres' patches on it every 1 cm, a 6 cm rock that
        // moved under one patch
        let h = |s: f64| 0.03 * (s * 4.0).sin() + 0.01 * (s * 11.0).cos() - 0.1 * s;
        let mut patches: Vec<Patch> = (0..600).flat_map(|i| {
            let s = i as f64 * 0.01;
            [Patch { s_m: s, z_m: h(s), sigma_m: 0.005 }, Patch { s_m: s + 0.003, z_m: h(s + 0.003), sigma_m: 0.005 }]
        }).collect();
        patches[300].z_m += 0.06;

        // Act
        let fit = |k: f64| {
            let (s0, coef) = fit_profile(&patches, 0.05, 0.02, k).unwrap();
            GroundFit { s0_m: s0, knot_m: 0.05, coef, on: vec![] }
        };
        let (huber, plain) = (fit(3.0), fit(0.0));

        // Assert: within 1 mm away from the rock (at 1.5 m), and Huber holding the rock's pull
        // under half of what least squares lets it pull
        let worst = |f: &GroundFit, near: bool| (10..590).map(|i| i as f64 * 0.01).filter(|s| near || (s - 1.5).abs() > 0.3).map(|s| (f.height(s) - h(s)).abs()).fold(0.0, f64::max);
        assert!(worst(&huber, false) < 0.001, "{} m", worst(&huber, false));
        assert!(worst(&huber, true) < 0.5 * worst(&plain, true), "{} vs {} m", worst(&huber, true), worst(&plain, true));
    }

    #[test]
    fn ground_window_level_straight_run_integrates_distance_and_keeps_the_base_sigma() {
        // Arrange: level, heading north at 5 m/s, climbing 0.2 m/s, no turn
        let p = NavPoint { r: M3::eye(), v: [5.0, 0.0, 0.2], w0: V3::zero() };
        let nav = vec![p; 401];

        // Act
        let g = ground_window(&nav, 0.0025, V3([1.0, 0.0, 0.0]), [0.75, -0.45], &GroundWeights::default());

        // Assert
        let v = (25.0f64 + 0.04).sqrt();
        assert!((g.s_m[400] - v).abs() < 1e-9 && (g.z_m[400] - 0.2).abs() < 1e-9);
        assert_eq!(g.up[0], [0.0, 0.0, 1.0]);
        let expect = (0.005f64.powi(2) + (0.05 * 1.2 / v).powi(2)).sqrt();
        assert!((g.sigma_m[0][0] - expect).abs() < 1e-12, "{:?}", g.sigma_m[0]);
    }

    #[test]
    fn ground_window_turning_widens_the_sigma_and_slow_turns_it_off() {
        // Arrange: a 10 m radius turn at 5 m/s, then the same at 1 m/s
        let turn = NavPoint { r: M3::eye(), v: [5.0, 0.0, 0.0], w0: V3([0.0, 0.0, 0.5]) };
        let slow = NavPoint { v: [1.0, 0.0, 0.0], ..turn };

        // Act
        let g = ground_window(&[turn, slow], 0.0025, V3([1.0, 0.0, 0.0]), [0.75, -0.45], &GroundWeights::default());

        // Assert: the rear 1.2^2 / 20 = 72 mm inside the front's line, 14 mm in height at 0.2
        let expect = (0.005f64.powi(2) + (0.2 * 0.072f64).powi(2) + (0.05 * 1.2 / 5.0f64).powi(2)).sqrt();
        assert!((g.sigma_m[0][0] - expect).abs() < 1e-12, "{} vs {expect}", g.sigma_m[0][0]);
        assert!(g.sigma_m[1][0].is_infinite());
    }
}
