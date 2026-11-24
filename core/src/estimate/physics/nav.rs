//! The navigation rows of the objective (spec `2026-11-24-idl1-one-model-estimator-DRAFT.md`
//! §2-3): the chassis attitude (a cumulative SO(3) spline, [`super::so3spline`])
//! and the nav-frame velocity (a cubic B-spline) explain the frame pod's
//! specific force, agree with the frame-rate latent the frame gyro measures,
//! and meet the GPS velocity fixes and the window's starting state.
//!
//! The nav frame is the GPS's: X north, Y west, Z up (`run::GpsSample`). The
//! attitude maps IMU0 axes to nav. The frame pod's flex against the chassis
//! (well under a milliradian and a millimetre) is left out of these rows: the
//! frame pod's specific force stands for the chassis's.
//!
//! Per sample, the rows ([`NNAV`]):
//!
//! | rows | term |
//! |---|---|
//! | 0..3 | specific force: R^T (v' - g) vs the frame pod's accelerometer |
//! | 3..6 | rate: the attitude spline's body rate vs the frame-rate latent w0 |
//! | 6..9 | GPS velocity north, west (and up when the fix has it), at the fix's sample |
//! | 9..12 | start attitude prior (the window's first sample only) |
//! | 12..15 | start velocity prior (first sample only) |
//!
//! In the air the specific force is near zero, which says nothing about R and
//! pins v' to gravity: free fall needs no switch.

use num_dual::Dual64;

use super::la::{c, exp, M3, Real, V3};
use super::so3spline::{log, rotation_rate_accel};

/// Gravity in the nav frame (Z up), m/s^2.
pub const G_NAV: [f64; 3] = [0.0, 0.0, -9.80665];
/// Navigation rows per sample.
pub const NNAV: usize = 15;
/// Attitude perturbation columns per sample: 4 control rotations x 3 axes.
pub const NATT: usize = 12;

/// Where the navigation latents' knots go.
#[derive(Clone, Debug)]
pub struct NavSpec {
    /// Attitude knot spacing, samples.
    pub m_att: usize,
    /// Velocity knot spacing, samples.
    pub m_vel: usize,
}

impl Default for NavSpec {
    fn default() -> Self {
        // 5 ms and 10 ms at 400 Hz: the velocity at the steer's spacing keeps the band as it is
        NavSpec { m_att: 2, m_vel: 4 }
    }
}

/// A GPS velocity fix, assigned to the grid sample nearest its (latency-corrected) time.
#[derive(Clone, Copy, Debug)]
pub struct GpsFix {
    /// Nav-frame velocity, m/s (north, west, up).
    pub v: [f64; 3],
    /// Whether `v[2]` (vertical, from the altitude) is measured.
    pub up: bool,
}

/// The window's starting state and how sure of it we are: the previous
/// window's solution in the overlap, or for a stretch's first window tilt
/// from gravity and heading from the GPS course.
#[derive(Clone, Debug)]
pub struct NavStart {
    /// Attitude at the first sample, IMU0 -> nav.
    pub att: [[f64; 3]; 3],
    /// Prior sigma of each rotation-vector component (nav-error in IMU0 axes), rad.
    pub att_sigma: [f64; 3],
    /// Velocity at the first sample, nav, m/s.
    pub vel: [f64; 3],
    /// Prior sigma of each velocity component, m/s.
    pub vel_sigma: f64,
}

/// A window's navigation measurements.
#[derive(Clone, Debug)]
pub struct NavWindow {
    /// Per sample: the GPS fix assigned to it, if any.
    pub gps: Vec<Option<GpsFix>>,
    /// The starting state.
    pub start: NavStart,
}

/// Noise levels for the navigation rows.
#[derive(Clone, Debug)]
pub struct NavWeights {
    /// Specific-force residual sigma, m/s^2: wider than the pods' because the
    /// velocity spline cannot follow the frame's vibration.
    pub accel_sigma: f64,
    /// Attitude rate vs the frame-rate latent, rad/s.
    pub rate_sigma: f64,
    /// GPS horizontal velocity sigma, m/s.
    pub gps_sigma: f64,
    /// GPS vertical velocity sigma, m/s.
    pub gps_up_sigma: f64,
}

impl Default for NavWeights {
    fn default() -> Self {
        NavWeights { accel_sigma: 3.0, rate_sigma: 0.02, gps_sigma: 0.3, gps_up_sigma: 0.5 }
    }
}

/// One sample's navigation inputs.
#[derive(Clone, Debug)]
pub struct NavSample<'a> {
    /// The four attitude controls touching this sample.
    pub ctrl: [M3<f64>; 4],
    /// Fraction of the attitude knot span.
    pub u: f64,
    /// Attitude knot span, s.
    pub span_s: f64,
    /// Velocity, nav, m/s.
    pub v: [f64; 3],
    /// Its time derivative, m/s^2.
    pub vd: [f64; 3],
    /// The frame-rate latent, rad/s, IMU0 axes.
    pub w0: [f64; 3],
    /// The frame pod's specific force, m/s^2, IMU0 axes.
    pub f_meas: [f64; 3],
    /// The GPS fix at this sample.
    pub gps: Option<GpsFix>,
    /// The start priors (first sample of the window only).
    pub start: Option<&'a NavStart>,
}

/// Rows and their Jacobians.
pub struct NavJac {
    /// Whitened residuals.
    pub r: [f64; NNAV],
    /// d r / d attitude perturbation (control i, axis a at column 3 i + a; R_i <- R_i Exp(d)).
    pub j_att: [[f64; NATT]; NNAV],
    /// d r / d (v, v').
    pub j_v: [[f64; 6]; NNAV],
    /// d r / d w0.
    pub j_w0: [[f64; 3]; NNAV],
}

/// The rows as a function of the attitude controls (v, v', w0 enter as constants).
fn rows<S: Real>(ctrl: &[M3<S>; 4], s: &NavSample, w: &NavWeights) -> [S; NNAV] {
    let (r, rate, _) = rotation_rate_accel(ctrl, c::<S>(s.u), s.span_s);
    let mut out = [c::<S>(0.0); NNAV];
    let a = V3::<S>::from_f64(std::array::from_fn(|i| s.vd[i] - G_NAV[i]));
    let f = r.t() * a;
    for i in 0..3 {
        out[i] = (f.0[i] - s.f_meas[i]) * (1.0 / w.accel_sigma);
        out[3 + i] = (rate.0[i] - s.w0[i]) * (1.0 / w.rate_sigma);
    }
    if let Some(g) = s.gps {
        for i in 0..2 {
            out[6 + i] = c::<S>((s.v[i] - g.v[i]) / w.gps_sigma);
        }
        if g.up {
            out[8] = c::<S>((s.v[2] - g.v[2]) / w.gps_up_sigma);
        }
    }
    if let Some(st) = s.start {
        let e = log(M3::<S>::from_f64(st.att).t() * r);
        for i in 0..3 {
            out[9 + i] = e.0[i] * (1.0 / st.att_sigma[i]);
            out[12 + i] = c::<S>((s.v[i] - st.vel[i]) / st.vel_sigma);
        }
    }
    out
}

/// The rows at one sample and, if `jac`, their Jacobians.
pub fn nav_rows(s: &NavSample, w: &NavWeights, jac: bool) -> NavJac {
    let r = rows(&s.ctrl, s, w);
    let mut out = NavJac { r, j_att: [[0.0; NATT]; NNAV], j_v: [[0.0; 6]; NNAV], j_w0: [[0.0; 3]; NNAV] };
    if !jac {
        return out;
    }
    let lift: [M3<Dual64>; 4] = s.ctrl.map(|m| m.map(Dual64::from));
    for i in 0..4 {
        for a in 0..3 {
            let mut d = V3::<Dual64>::zero();
            d.0[a] = Dual64::new(0.0, 1.0);
            let mut ctrl = lift;
            ctrl[i] = ctrl[i] * exp(d);
            let y = rows(&ctrl, s, w);
            for row in 0..NNAV {
                out.j_att[row][3 * i + a] = y[row].eps;
            }
        }
    }
    let (rot, _, _) = rotation_rate_accel(&s.ctrl, s.u, s.span_s);
    for i in 0..3 {
        for k in 0..3 {
            out.j_v[i][3 + k] = rot.0[k][i] / w.accel_sigma; // R^T
        }
        out.j_w0[3 + i][i] = -1.0 / w.rate_sigma;
    }
    if let Some(g) = s.gps {
        out.j_v[6][0] = 1.0 / w.gps_sigma;
        out.j_v[7][1] = 1.0 / w.gps_sigma;
        if g.up {
            out.j_v[8][2] = 1.0 / w.gps_up_sigma;
        }
    }
    if let Some(st) = s.start {
        for i in 0..3 {
            out.j_v[12 + i][i] = 1.0 / st.vel_sigma;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(start: &NavStart) -> NavSample<'_> {
        let mut r = exp(V3::from_f64([0.2, -0.3, 1.0]));
        let ctrl = std::array::from_fn(|i| {
            r = r * exp(V3::from_f64([0.01 * i as f64, -0.02, 0.015]));
            r
        });
        NavSample {
            ctrl,
            u: 0.37,
            span_s: 0.005,
            v: [7.0, -1.0, 0.3],
            vd: [0.5, 2.0, -1.0],
            w0: [0.3, -0.1, 0.8],
            f_meas: [1.0, -2.0, 9.0],
            gps: Some(GpsFix { v: [6.5, -0.8, 0.0], up: true }),
            start: Some(start),
        }
    }

    fn start() -> NavStart {
        NavStart { att: exp(V3::from_f64([0.25, -0.3, 0.95])).0, att_sigma: [0.1, 0.1, 0.5], vel: [7.2, -1.1, 0.0], vel_sigma: 1.0 }
    }

    #[test]
    fn attitude_jacobian_matches_finite_differences_of_the_perturbation() {
        // Arrange
        let st = start();
        let s = sample(&st);
        let w = NavWeights::default();
        let h = 1e-7;

        // Act
        let j = nav_rows(&s, &w, true);

        // Assert
        for i in 0..4 {
            for a in 0..3 {
                let mut d = [0.0; 3];
                d[a] = h;
                let (mut sp, mut sm) = (s.clone(), s.clone());
                sp.ctrl[i] = s.ctrl[i] * exp(V3::from_f64(d));
                d[a] = -h;
                sm.ctrl[i] = s.ctrl[i] * exp(V3::from_f64(d));
                let (rp, rm) = (nav_rows(&sp, &w, false).r, nav_rows(&sm, &w, false).r);
                for row in 0..NNAV {
                    let fd = (rp[row] - rm[row]) / (2.0 * h);
                    let tol = 1e-5 * (1.0 + fd.abs());
                    assert!((j.j_att[row][3 * i + a] - fd).abs() < tol, "row {row}, control {i}, axis {a}: {} vs {fd}", j.j_att[row][3 * i + a]);
                }
            }
        }
    }

    #[test]
    fn velocity_and_rate_jacobians_match_finite_differences() {
        // Arrange
        let st = start();
        let s = sample(&st);
        let w = NavWeights::default();
        let h = 1e-6;

        // Act
        let j = nav_rows(&s, &w, true);

        // Assert
        for k in 0..9 {
            let bump = |x: f64| {
                let mut t = s.clone();
                match k {
                    0..=2 => t.v[k] += x,
                    3..=5 => t.vd[k - 3] += x,
                    _ => t.w0[k - 6] += x,
                }
                nav_rows(&t, &w, false).r
            };
            let (rp, rm) = (bump(h), bump(-h));
            for row in 0..NNAV {
                let fd = (rp[row] - rm[row]) / (2.0 * h);
                let got = if k < 6 { j.j_v[row][k] } else { j.j_w0[row][k - 6] };
                assert!((got - fd).abs() < 1e-6 * (1.0 + fd.abs()), "row {row}, input {k}: {got} vs {fd}");
            }
        }
    }

    #[test]
    fn level_and_still_reads_one_g_up_with_zero_residual() {
        // Arrange: identity attitude, no motion, the accelerometer reading +g on Z
        let st = NavStart { att: M3::<f64>::eye().0, att_sigma: [0.1; 3], vel: [0.0; 3], vel_sigma: 1.0 };
        let s = NavSample { ctrl: [M3::eye(); 4], u: 0.5, span_s: 0.005, v: [0.0; 3], vd: [0.0; 3], w0: [0.0; 3], f_meas: [0.0, 0.0, 9.80665], gps: Some(GpsFix { v: [0.0; 3], up: true }), start: Some(&st) };

        // Act
        let j = nav_rows(&s, &NavWeights::default(), false);

        // Assert
        assert!(j.r.iter().all(|x| x.abs() < 1e-12), "{:?}", j.r);
    }
}
