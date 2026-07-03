//! The hand-rolled **iterated extended Kalman filter** (design §2). A concrete
//! struct over the shared [`ProcessModel`]/[`MeasurementModel`] traits — no
//! `Estimator` trait until ≥2 impls exist (the batch smoother is M5).
//!
//! The measurement update is written in **Gauss-Newton / information form**, the
//! same MAP step a batch solve stacks (Bell 1994): for a measurement with residual
//! `r = z ⊟ h(x)` and analytic `H = ∂r/∂δ`, anchored at the predicted mean `x̄`
//! with covariance `P`, each iteration `i` (linearizing at `x_i`, `δ_i = x_i ⊟ x̄`)
//! computes
//!
//! ```text
//!   S = H P Hᵀ + R           K = P Hᵀ S⁻¹
//!   δ = K (H δ_i − r_i)       x_{i+1} = x̄ ⊞ δ
//! ```
//!
//! and on convergence updates `P⁺ = (I − K H) P` (Joseph form for stability). One
//! iteration is the plain EKF; iterating matters for the steering re-linearization
//! through `ψ̂` (M2b). Frozen states carry a tiny prior variance, so the gain barely
//! moves them.

use crate::estimate::model::{ErrorState, ImuInput, MeasurementModel, ProcessModel, SampleContext};
use crate::estimate::process::MtbProcess;
use crate::estimate::schema::StateSchema;
use crate::estimate::state::MtbState;
use nalgebra::DMatrix;

/// Variance a frozen (un-estimated) state component carries — small but nonzero so
/// the covariance stays positive-definite without the gain disturbing it.
pub const FROZEN_VARIANCE: f64 = 1.0e-12;

/// Initial 1-σ priors for the active state components (refined in M3).
#[derive(Debug, Clone, Copy)]
pub struct InitStd {
    /// Attitude tilt prior, rad (per axis).
    pub attitude: f64,
    /// Velocity prior, m/s.
    pub velocity: f64,
    /// Gyro-bias prior, rad/s.
    pub gyro_bias: f64,
    /// Accel-bias prior, m/s².
    pub accel_bias: f64,
    /// Wheel-travel prior, m.
    pub wheel_travel: f64,
    /// Wheel-velocity prior, m/s.
    pub wheel_velocity: f64,
    /// Steer-angle prior, rad.
    pub steer_angle: f64,
    /// Steer-rate prior, rad/s.
    pub steer_rate: f64,
}

impl Default for InitStd {
    fn default() -> Self {
        InitStd {
            attitude: 5_f64.to_radians(),
            velocity: 1.0,
            gyro_bias: 0.05,
            accel_bias: 0.5,
            wheel_travel: 0.05,
            wheel_velocity: 1.0,
            steer_angle: 0.2,
            steer_rate: 1.0,
        }
    }
}

/// Filter belief at one instant: state mean + error-state covariance.
#[derive(Debug, Clone)]
pub struct FilterState {
    /// State mean.
    pub x: MtbState,
    /// Error-state covariance (24×24), in the pinned right-perturbation coordinates.
    pub p: DMatrix<f64>,
}

impl FilterState {
    /// Builds the initial belief: mean `x0` with a diagonal covariance whose active
    /// blocks use `std` and whose frozen blocks use [`FROZEN_VARIANCE`].
    pub fn initial(x0: MtbState, schema: &StateSchema, std: &InitStd) -> FilterState {
        let n = MtbState::DOF;
        let mut p = DMatrix::<f64>::zeros(n, n);
        for c in &schema.components {
            let var = if c.active {
                match c.symbol {
                    "R_chassis" => std.attitude.powi(2),
                    "v_chassis" => std.velocity.powi(2),
                    "b_g0" | "b_g1" | "b_g2" => std.gyro_bias.powi(2),
                    "b_a0" => std.accel_bias.powi(2),
                    "w_f" | "w_r" => std.wheel_travel.powi(2),
                    "dw_f" | "dw_r" => std.wheel_velocity.powi(2),
                    "psi" => std.steer_angle.powi(2),
                    "dpsi" => std.steer_rate.powi(2),
                    _ => FROZEN_VARIANCE,
                }
            } else {
                FROZEN_VARIANCE
            };
            for k in 0..c.dim {
                p[(c.error_index + k, c.error_index + k)] = var;
            }
        }
        FilterState { x: x0, p }
    }
}

/// The iterated EKF over [`MtbProcess`] + the measurement factors.
#[derive(Debug, Clone)]
pub struct Iekf {
    /// Kinematic process model (carries the schema + noise).
    pub process: MtbProcess,
    /// Maximum measurement-update iterations (1 = plain EKF).
    pub max_iters: usize,
    /// Mean tangent-step convergence threshold for the iteration.
    pub tol: f64,
}

impl Iekf {
    /// A plain (single-iteration) EKF over the given process model.
    pub fn new(process: MtbProcess) -> Self {
        Iekf { process, max_iters: 1, tol: 1e-9 }
    }

    /// Time update: propagate the mean and `P ← F P Fᵀ + Q`.
    pub fn predict(&self, fs: &FilterState, u: &ImuInput, dt: f64) -> FilterState {
        let (f, q) = self.process.jacobian_noise(&fs.x, u, dt);
        let x = self.process.predict(&fs.x, u, dt);
        let p = &f * &fs.p * f.transpose() + q;
        FilterState { x, p }
    }

    /// Measurement update for one factor. Inactive factors (per the validity gate)
    /// pass through unchanged. Returns `fs` unchanged if the innovation covariance
    /// is singular (a degenerate factor is skipped, never a hard crash).
    pub fn update(
        &self,
        fs: &FilterState,
        m: &dyn MeasurementModel<MtbState>,
        ctx: &SampleContext,
    ) -> FilterState {
        if !m.active(&fs.x, ctx) {
            return fs.clone();
        }
        let n = MtbState::DOF;
        let x_bar = fs.x.clone();
        let mut x = x_bar.clone();
        let mut last_h = m.jacobian(&x);
        let mut last_k: Option<DMatrix<f64>> = None;

        for _ in 0..self.max_iters.max(1) {
            let h = m.jacobian(&x);
            let r = m.residual(&x);
            let rr = m.noise(&x);
            let delta_i = x.ominus(&x_bar); // x_i ⊟ x̄

            let s = &h * &fs.p * h.transpose() + rr;
            let s_inv = match s.try_inverse() {
                Some(inv) => inv,
                None => return fs.clone(),
            };
            let k = &fs.p * h.transpose() * s_inv; // n × dim
            let delta = &k * (&h * &delta_i - r); // n
            // A non-finite innovation must not poison the run: a NaN Jacobian/residual
            // (e.g. a degenerate factor) yields a NaN `S`, and `try_inverse` returns
            // `Some(NaN)` — so the singular-`S` guard above does NOT catch it. Reject
            // the whole update and skip the factor instead (see F1/F2).
            if !delta.iter().all(|v| v.is_finite()) {
                return fs.clone();
            }
            x = x_bar.oplus(&delta);

            last_h = h;
            last_k = Some(k);
            let step = (&delta - &delta_i).norm();
            if step < self.tol {
                break;
            }
        }

        // Covariance update at the final linearization (Joseph form).
        let p = match last_k {
            Some(k) => {
                let ikh = DMatrix::<f64>::identity(n, n) - &k * &last_h;
                let rr = m.noise(&x);
                &ikh * &fs.p * ikh.transpose() + &k * rr * k.transpose()
            }
            None => fs.p.clone(),
        };
        FilterState { x, p }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::geometry::BikeGeometry;
    use crate::estimate::measurements::gravity::GravityLeveling;
    use crate::estimate::measurements::gps::GpsVelocity;
    use crate::estimate::measurements::prior::{SagPrior, Wheel};
    use crate::estimate::measurements::zupt::{GyroBias, ZeroAngularRate, ZeroVelocity};
    use crate::estimate::process::{MtbProcess, GRAVITY};
    use crate::estimate::schema::StateSchema;
    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, Vector3};

    fn filter() -> Iekf {
        let schema = StateSchema::from_geometry(&BikeGeometry::reference_bike(), false);
        Iekf::new(MtbProcess::new(schema))
    }

    fn schema() -> StateSchema {
        StateSchema::from_geometry(&BikeGeometry::reference_bike(), false)
    }

    fn rest_state() -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::identity(),
            v_chassis: Vector3::zeros(),
            b_g0: Vector3::zeros(),
            b_a0: Vector3::zeros(),
            b_g1: Vector3::zeros(),
            b_g2: Vector3::zeros(),
            d_f: 0.0,
            dd_f: 0.0,
            s_r: 0.0,
            ds_r: 0.0,
            psi: 0.0,
            dpsi: 0.0,
        }
    }

    fn rest_input() -> ImuInput {
        ImuInput {
            gyro0: Vector3::zeros(),
            accel0: Vector3::new(0.0, 0.0, GRAVITY),
            wheel_accel_front: 0.0,
            wheel_accel_rear: 0.0,
        }
    }

    #[test]
    fn static_first_light_velocity_and_attitude_stay_put() {
        // Arrange — 5 s of true rest at 100 Hz with ZUPT + ZARU + gravity active.
        let f = filter();
        let mut fs = FilterState::initial(rest_state(), &schema(), &InitStd::default());
        let dt = 0.01;
        let stationary = SampleContext { stationary: true };

        // Act
        for _ in 0..500 {
            fs = f.predict(&fs, &rest_input(), dt);
            fs = f.update(&fs, &ZeroVelocity { sigma: 0.01 }, &stationary);
            fs = f.update(
                &fs,
                &ZeroAngularRate { target: GyroBias::Imu0, measured: Vector3::zeros(), sigma: 1e-3 },
                &stationary,
            );
            fs = f.update(
                &fs,
                &GravityLeveling {
                    accel0: Vector3::new(0.0, 0.0, GRAVITY),
                    a_kin_nav: Vector3::zeros(),
                    sigma: 0.02,
                },
                &stationary,
            );
        }

        // Assert — nothing drifts: velocity ≈ 0, attitude ≈ level.
        assert!(fs.x.v_chassis.norm() < 1e-3, "velocity drifted: {}", fs.x.v_chassis.norm());
        assert!(fs.x.r_chassis.angle() < 1e-3, "attitude drifted: {}", fs.x.r_chassis.angle());
    }

    #[test]
    fn zaru_recovers_constant_gyro_bias_and_keeps_attitude_stable() {
        // Arrange — truly stationary, but the gyro reads a constant 0.02 rad/s about
        // Z (an un-removed bias). ZARU must learn it so attitude does NOT drift.
        let f = filter();
        let mut fs = FilterState::initial(rest_state(), &schema(), &InitStd::default());
        let dt = 0.01;
        let bias = Vector3::new(0.0, 0.0, 0.02);
        let stationary = SampleContext { stationary: true };
        let u = ImuInput { gyro0: bias, accel0: Vector3::new(0.0, 0.0, GRAVITY), wheel_accel_front: 0.0, wheel_accel_rear: 0.0 };

        // Act
        for _ in 0..1000 {
            fs = f.predict(&fs, &u, dt);
            fs = f.update(
                &fs,
                &ZeroAngularRate { target: GyroBias::Imu0, measured: bias, sigma: 1e-3 },
                &stationary,
            );
            fs = f.update(
                &fs,
                &GravityLeveling { accel0: Vector3::new(0.0, 0.0, GRAVITY), a_kin_nav: Vector3::zeros(), sigma: 0.02 },
                &stationary,
            );
        }

        // Assert — bias learned, attitude held level despite the raw gyro offset.
        assert_relative_eq!(fs.x.b_g0.z, 0.02, epsilon = 1e-3);
        assert!(fs.x.r_chassis.angle() < 5e-3, "attitude drifted under biased gyro");
    }

    #[test]
    fn gps_anchors_velocity_against_a_constant_accel_offset() {
        // Arrange — a persistent 0.3 m/s² forward specific-force error would ramp
        // velocity unboundedly; GPS velocity (truth ≈ 4 m/s) must anchor it.
        let f = filter();
        let mut x0 = rest_state();
        x0.v_chassis = Vector3::new(4.0, 0.0, 0.0);
        let mut fs = FilterState::initial(x0, &schema(), &InitStd::default());
        let dt = 0.01;
        let u = ImuInput {
            gyro0: Vector3::zeros(),
            accel0: Vector3::new(0.3, 0.0, GRAVITY),
            wheel_accel_front: 0.0,
            wheel_accel_rear: 0.0,
        };
        let moving = SampleContext { stationary: false };

        // Act — 10 s; GPS arrives every 0.5 s (every 50 samples).
        for i in 0..1000 {
            fs = f.predict(&fs, &u, dt);
            if i % 50 == 0 {
                fs = f.update(&fs, &GpsVelocity { measured: Vector3::new(4.0, 0.0, 0.0), sigma: 0.1 }, &moving);
            }
        }

        // Assert — velocity stays near the GPS anchor, not ramping toward 4 + 0.3·10.
        assert!((fs.x.v_chassis.x - 4.0).abs() < 0.5, "vx ran away: {}", fs.x.v_chassis.x);
    }

    #[test]
    fn recovers_known_oscillating_front_travel_and_velocity() {
        // Arrange — a known travel profile w(t) = sag + A·sin(ωt) about sag, with the
        // forward model's exact drive control ẅ(t) = −A·ω²·sin(ωt) (design §11: the
        // process model run forward IS the synthetic generator). Init at truth; a
        // weak sag prior keeps the run a real filter pass without attenuating motion.
        let f = filter();
        let sag = 0.046;
        let amp = 0.03;
        let omega = std::f64::consts::PI; // 0.5 Hz
        let dt = 0.005;
        let mut x0 = rest_state();
        x0.d_f = sag; // w(0) = sag + A·sin(0)
        x0.dd_f = amp * omega; // ẇ(0) = A·ω·cos(0)
        let mut fs = FilterState::initial(x0, &schema(), &InitStd::default());
        let always = SampleContext { stationary: false };

        // Act — integrate to a quarter period (t = 0.5 s): the non-trivial peak.
        let steps = 100; // 100 · 0.005 = 0.5 s
        for i in 0..steps {
            let t = i as f64 * dt;
            let u = ImuInput {
                gyro0: Vector3::zeros(),
                accel0: Vector3::new(0.0, 0.0, GRAVITY),
                wheel_accel_front: -amp * omega * omega * (omega * t).sin(),
                wheel_accel_rear: 0.0,
            };
            fs = f.predict(&fs, &u, dt);
            // Negligible sag prior: keeps this a real filter pass while isolating
            // forward-model tracking. (An ordinary-strength sag prior applied every
            // sample would fight the motion — the unanchored wheel double-integrator's
            // covariance inflates fast, which is exactly why sag/ZUPT anchoring is
            // coasting-weighted in a real run. The sag DC-anchoring strength itself is
            // validated in `sag_prior_pulls_front_travel_toward_sag`.)
            fs = f.update(&fs, &SagPrior { wheel: Wheel::Front, sag, sigma: 1.0e4 }, &always);
        }

        // Assert — at the quarter period travel is at its peak (sag + A) and velocity
        // has returned to ~0, recovered within explicit-Euler tolerance.
        assert_relative_eq!(fs.x.d_f, sag + amp, epsilon = 3e-3);
        assert!(fs.x.dd_f.abs() < 0.03, "velocity should be ~0 at the peak: {}", fs.x.dd_f);
    }

    #[test]
    fn sag_prior_pulls_front_travel_toward_sag() {
        // Arrange — front travel starts at 0; sag prior target 46 mm.
        let f = filter();
        let mut fs = FilterState::initial(rest_state(), &schema(), &InitStd::default());
        let dt = 0.01;
        let always = SampleContext { stationary: false };

        // Act — no wheel drive; only the sag prior acts on travel.
        for _ in 0..500 {
            fs = f.predict(&fs, &rest_input(), dt);
            fs = f.update(&fs, &SagPrior { wheel: Wheel::Front, sag: 0.046, sigma: 0.01 }, &always);
        }

        // Assert — travel converges toward sag.
        assert_relative_eq!(fs.x.d_f, 0.046, epsilon = 2e-3);
    }
}
