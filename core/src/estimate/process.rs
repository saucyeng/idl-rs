//! IMU0-driven kinematic process model (design §5, "IMU0 strapdown" + "diff-accel
//! travel input") implementing [`ProcessModel`] for [`MtbState`], with the analytic
//! error-state transition Jacobian `F` and discrete process-noise covariance `Q`.
//!
//! Propagation (right-perturbation error-state, the pinned convention):
//! - **Attitude**: `R⁺ = R·Exp((ω₀−b_g0)·dt)` — bias-corrected IMU0 rate.
//! - **Velocity**: `v⁺ = v + (R·(a₀−b_a0) + g_nav)·dt`, `g_nav = (0,0,−g)`.
//! - **Biases**: random walk (mean unchanged; driven by `Q`).
//! - **Wheels**: double integrators `{w, ẇ}` driven by the precomputed diff-accel
//!   controls in [`ImuInput`] (`ẇ⁺ = ẇ + ẅ_ctrl·dt`, `w⁺ = w + ẇ·dt`).
//! - **Steering**: kinematic `{ψ, ψ̇}` (frozen in M2a via zero `Q` + tight prior).
//!
//! The error-state index layout matches [`MtbState`]'s 24-DOF ordering exactly.

use crate::estimate::model::{ImuInput, ProcessModel};
use crate::estimate::noise::{process_noise_over_dt, ImuNoise};
use crate::estimate::schema::StateSchema;
use crate::estimate::state::MtbState;
use crate::rotation::{exp_so3, lever_arm_accel, right_jacobian_so3, skew};
use nalgebra::{DMatrix, Vector3};

/// Standard gravity magnitude, m/s² (matches the math evaluator's `g`).
pub const GRAVITY: f64 = 9.81;

/// Process-noise PSDs (random-walk coefficients `N`, variance over `dt` = `N²·dt`).
/// Tuned in M3; [`ProcessNoiseConfig::reference_default`] gives MEMS-plausible
/// starting values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessNoiseConfig {
    /// IMU0 white/bias noise (attitude, velocity, b_g0, b_a0).
    pub imu0: ImuNoise,
    /// Front-unsprung (IMU1) gyro bias random-walk, (rad/s)/√s.
    pub gyro1_bias_rw: f64,
    /// Rear-unsprung (IMU2) gyro bias random-walk, (rad/s)/√s.
    pub gyro2_bias_rw: f64,
    /// Wheel-velocity driving noise (unmodeled jerk / diff-accel noise), (m/s²)/√s.
    pub wheel_vel_rw: f64,
    /// Wheel-travel regularizing noise, (m)/√s.
    pub wheel_pos_rw: f64,
    /// Steer-rate driving noise, (rad/s²)/√s (frozen → contributes 0 in M2a).
    pub steer_rate_rw: f64,
    /// Steer-angle random walk, rad/√s: how fast the angle integrated from the
    /// two gyros is allowed to wander before the accelerometers pull it back.
    /// The gyro difference along the steering axis is not only steer rate — a
    /// few degrees of error in the axis or the fork pod's mount lets the
    /// frame's own rotation leak in — and that error is not white, so it is
    /// carried here and not in the rate's noise. 0.03 gives the accelerometer
    /// reference about a 4 s hold.
    pub steer_angle_rw: f64,
}

impl ProcessNoiseConfig {
    /// MEMS-plausible starting values (refined in M3 against real logs).
    pub fn reference_default() -> Self {
        ProcessNoiseConfig {
            imu0: ImuNoise {
                gyro_arw: 0.003,       // rad/√s (~0.17°/√s)
                accel_vrw: 0.05,       // (m/s)/√s
                gyro_bias_rw: 1.0e-4,  // (rad/s)/√s
                accel_bias_rw: 1.0e-3, // (m/s²)/√s
            },
            gyro1_bias_rw: 1.0e-4,
            gyro2_bias_rw: 1.0e-4,
            wheel_vel_rw: 5.0,  // suspension acceleration is energetic
            wheel_pos_rw: 1.0e-3,
            steer_rate_rw: 1.0,
            steer_angle_rw: 0.03,
        }
    }
}

/// The IMU0-driven kinematic process model. Holds the active-state schema (to zero
/// process noise on frozen components) and the noise config; `gravity` is the
/// nav-frame gravity magnitude.
#[derive(Debug, Clone)]
pub struct MtbProcess {
    /// Active/frozen schema (frozen components get zero process noise).
    pub schema: StateSchema,
    /// Process-noise PSDs.
    pub noise: ProcessNoiseConfig,
    /// Gravity magnitude, m/s².
    pub gravity: f64,
}

impl MtbProcess {
    /// Builds a process model with the reference-default noise and `GRAVITY`.
    pub fn new(schema: StateSchema) -> Self {
        MtbProcess { schema, noise: ProcessNoiseConfig::reference_default(), gravity: GRAVITY }
    }
}

/// The projected differential specific force along `axis` — the wheel-drive control
/// `ẅ = axis·(a_unsprung − a₀ − [ω̇×L + ω×(ω×L)])` (design §5). All vectors are in
/// the chassis frame; `a_unsprung`/`a0` are specific forces (gravity cancels in the
/// difference). The Coriolis term `2ω×(ẇ·axis)` drops out of the projection
/// (`axis·(ω×axis) ≡ 0`), so it is correctly absent. Returns m/s².
pub fn wheel_drive_accel(
    axis: Vector3<f64>,
    accel_unsprung: Vector3<f64>,
    accel0: Vector3<f64>,
    omega0: Vector3<f64>,
    omega0_dot: Vector3<f64>,
    lever: Vector3<f64>,
) -> f64 {
    // Rigid-body rotational contribution at the unsprung lever (ω̇×L + ω×(ω×L)).
    let lever_term = lever_arm_accel(Vector3::zeros(), omega0, omega0_dot, lever);
    axis.dot(&(accel_unsprung - accel0 - lever_term))
}

impl ProcessModel<MtbState> for MtbProcess {
    fn predict(&self, x: &MtbState, u: &ImuInput, dt: f64) -> MtbState {
        // Attitude: bias-corrected IMU0 rate, right-multiplied increment.
        let omega0 = u.gyro0 - x.b_g0;
        let r_chassis = x.r_chassis * exp_so3(omega0 * dt);
        // Velocity: nav-rotated specific force + gravity.
        let f_nav = x.r_chassis * (u.accel0 - x.b_a0);
        let a_nav = f_nav + Vector3::new(0.0, 0.0, -self.gravity);
        let v_chassis = x.v_chassis + a_nav * dt;
        MtbState {
            r_chassis,
            v_chassis,
            // Biases random-walk: mean unchanged.
            b_g0: x.b_g0,
            b_a0: x.b_a0,
            b_g1: x.b_g1,
            b_g2: x.b_g2,
            // Wheel double integrators (diff-accel control).
            d_f: x.d_f + x.dd_f * dt,
            dd_f: x.dd_f + u.wheel_accel_front * dt,
            s_r: x.s_r + x.ds_r * dt,
            ds_r: x.ds_r + u.wheel_accel_rear * dt,
            // Steering kinematics (frozen in M2a via Q).
            psi: x.psi + x.dpsi * dt,
            dpsi: x.dpsi,
        }
    }

    fn jacobian_noise(&self, x: &MtbState, u: &ImuInput, dt: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = MtbState::DOF;
        let omega0 = u.gyro0 - x.b_g0;
        let phi = omega0 * dt;
        let r_mat = x.r_chassis.to_rotation_matrix().into_inner();

        // --- Transition Jacobian F (∂δ⁺/∂δ at δ=0) ---
        let mut f = DMatrix::<f64>::identity(n, n);
        // Attitude: ∂δθ⁺/∂δθ = Exp(−φ); ∂δθ⁺/∂δb_g0 = −J_r(φ)·dt.
        let dr_t = exp_so3(-phi).to_rotation_matrix().into_inner();
        let jr_dt = right_jacobian_so3(phi) * dt;
        for i in 0..3 {
            for k in 0..3 {
                f[(i, k)] = dr_t[(i, k)]; // δθ ← δθ
                f[(i, 6 + k)] = -jr_dt[(i, k)]; // δθ ← δb_g0
            }
        }
        // Velocity: ∂δv⁺/∂δθ = −R·[a₀−b_a0]×·dt; ∂δv⁺/∂δb_a0 = −R·dt.
        let a_skew = skew(u.accel0 - x.b_a0);
        let dv_dtheta = -(r_mat * a_skew) * dt;
        let dv_dba = -r_mat * dt;
        for i in 0..3 {
            for k in 0..3 {
                f[(3 + i, k)] = dv_dtheta[(i, k)]; // δv ← δθ
                f[(3 + i, 9 + k)] = dv_dba[(i, k)]; // δv ← δb_a0
            }
        }
        // Wheel / steer double integrators: position ← velocity coupling = dt.
        f[(18, 19)] = dt; // w_f ← ẇ_f
        f[(20, 21)] = dt; // w_r ← ẇ_r
        f[(22, 23)] = dt; // ψ   ← ψ̇

        // --- Process-noise covariance Q (diagonal; frozen components → 0) ---
        let qn = process_noise_over_dt(&self.noise.imu0, dt);
        let mut q = DMatrix::<f64>::zeros(n, n);
        let set = |q: &mut DMatrix<f64>, lo: usize, hi: usize, v: f64| {
            for i in lo..hi {
                q[(i, i)] = v;
            }
        };
        set(&mut q, 0, 3, qn.attitude_var); // δθ
        set(&mut q, 3, 6, qn.velocity_var); // δv
        set(&mut q, 6, 9, qn.gyro_bias_var); // b_g0
        set(&mut q, 9, 12, qn.accel_bias_var); // b_a0
        set(&mut q, 12, 15, self.noise.gyro1_bias_rw.powi(2) * dt); // b_g1 (front always present)
        let gate = |sym: &str, v: f64| if self.schema.is_active(sym) { v } else { 0.0 };
        set(&mut q, 15, 18, gate("b_g2", self.noise.gyro2_bias_rw.powi(2) * dt));
        q[(18, 18)] = gate("w_f", self.noise.wheel_pos_rw.powi(2) * dt);
        q[(19, 19)] = gate("dw_f", self.noise.wheel_vel_rw.powi(2) * dt);
        q[(20, 20)] = gate("w_r", self.noise.wheel_pos_rw.powi(2) * dt);
        q[(21, 21)] = gate("dw_r", self.noise.wheel_vel_rw.powi(2) * dt);
        q[(22, 22)] = gate("psi", self.noise.steer_angle_rw.powi(2) * dt);
        q[(23, 23)] = gate("dpsi", self.noise.steer_rate_rw.powi(2) * dt);

        (f, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::geometry::BikeGeometry;
    use crate::estimate::model::ErrorState;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, UnitQuaternion};

    fn full_sus_process() -> MtbProcess {
        let schema = StateSchema::from_geometry(&BikeGeometry::reference_bike(), false);
        MtbProcess::new(schema)
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
            accel0: Vector3::new(0.0, 0.0, GRAVITY), // reads +g up at rest
            wheel_accel_front: 0.0,
            wheel_accel_rear: 0.0,
        }
    }

    #[test]
    fn at_rest_attitude_level_and_velocity_zero() {
        // Arrange
        let p = full_sus_process();

        // Act
        let x1 = p.predict(&rest_state(), &rest_input(), 0.01);

        // Assert — gravity exactly cancels: no attitude drift, no velocity growth.
        assert_relative_eq!(x1.r_chassis.angle(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(x1.v_chassis, Vector3::zeros(), epsilon = 1e-12);
    }

    #[test]
    fn horizontal_specific_force_integrates_into_velocity() {
        // Arrange — 1 m/s² forward specific force, level attitude, no bias.
        let p = full_sus_process();
        let u = ImuInput {
            gyro0: Vector3::zeros(),
            accel0: Vector3::new(1.0, 0.0, GRAVITY),
            wheel_accel_front: 0.0,
            wheel_accel_rear: 0.0,
        };

        // Act
        let x1 = p.predict(&rest_state(), &u, 0.5);

        // Assert — a_nav = (1,0,0); v⁺ = (0.5, 0, 0).
        assert_relative_eq!(x1.v_chassis, Vector3::new(0.5, 0.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn gyro_rate_rotates_attitude() {
        // Arrange — 1 rad/s about +Z for 0.5 s ⇒ 0.5 rad yaw.
        let p = full_sus_process();
        let u = ImuInput {
            gyro0: Vector3::new(0.0, 0.0, 1.0),
            accel0: Vector3::new(0.0, 0.0, GRAVITY),
            wheel_accel_front: 0.0,
            wheel_accel_rear: 0.0,
        };

        // Act
        let x1 = p.predict(&rest_state(), &u, 0.5);

        // Assert
        assert_relative_eq!(x1.r_chassis.angle(), 0.5, epsilon = 1e-12);
        assert_relative_eq!(x1.r_chassis.axis().unwrap().into_inner(), Vector3::z(), epsilon = 1e-12);
    }

    #[test]
    fn wheel_double_integrator_advances_travel_then_velocity() {
        // Arrange — front wheel moving at 1 m/s with a +2 m/s² drive control.
        let p = full_sus_process();
        let mut x = rest_state();
        x.dd_f = 1.0;
        let u = ImuInput {
            gyro0: Vector3::zeros(),
            accel0: Vector3::new(0.0, 0.0, GRAVITY),
            wheel_accel_front: 2.0,
            wheel_accel_rear: 0.0,
        };

        // Act
        let x1 = p.predict(&x, &u, 0.5);

        // Assert — w⁺ = 0 + 1·0.5 = 0.5; ẇ⁺ = 1 + 2·0.5 = 2.0.
        assert_relative_eq!(x1.d_f, 0.5, epsilon = 1e-12);
        assert_relative_eq!(x1.dd_f, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn wheel_drive_pure_rotation_no_travel_is_zero() {
        // Arrange — the lever-arm omission guard (design testing §): a rigidly
        // rotating unsprung mass with no real travel must project to ẅ ≈ 0.
        // Steady spin ω=3 about +Z, lever [0.5,0,0] ⇒ centripetal (−4.5,0,0).
        let omega = Vector3::new(0.0, 0.0, 3.0);
        let lever = Vector3::new(0.5, 0.0, 0.0);
        let accel0 = Vector3::new(0.0, 0.0, GRAVITY);
        let accel_unsprung = accel0 + Vector3::new(-4.5, 0.0, 0.0); // rigid transfer
        let axis = Vector3::z();

        // Act
        let a = wheel_drive_accel(axis, accel_unsprung, accel0, omega, Vector3::zeros(), lever);

        // Assert
        assert_relative_eq!(a, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn wheel_drive_pure_translation_recovers_axis_component() {
        // Arrange — no rotation; unsprung sees an extra 3 m/s² along the axis.
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let accel0 = Vector3::new(0.0, 0.0, GRAVITY);
        let accel_unsprung = Vector3::new(0.0, 0.0, GRAVITY + 3.0);

        // Act
        let a = wheel_drive_accel(
            axis,
            accel_unsprung,
            accel0,
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::new(0.4, 0.0, -0.2),
        );

        // Assert
        assert_relative_eq!(a, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn analytic_transition_jacobian_matches_finite_difference() {
        // Arrange — a non-trivial state and input so every coupling block is exercised.
        let p = full_sus_process();
        let x = MtbState {
            r_chassis: exp_so3(Vector3::new(0.1, -0.2, 0.3)),
            v_chassis: Vector3::new(1.5, -0.5, 0.2),
            b_g0: Vector3::new(0.01, -0.02, 0.015),
            b_a0: Vector3::new(0.05, -0.03, 0.04),
            b_g1: Vector3::new(0.002, 0.001, -0.003),
            b_g2: Vector3::new(-0.001, 0.002, 0.0015),
            d_f: 0.03,
            dd_f: -0.2,
            s_r: 0.02,
            ds_r: 0.1,
            psi: 0.15,
            dpsi: -0.3,
        };
        let u = ImuInput {
            gyro0: Vector3::new(0.4, -0.3, 0.6),
            accel0: Vector3::new(0.7, -1.2, GRAVITY + 0.5),
            wheel_accel_front: 1.3,
            wheel_accel_rear: -0.8,
        };
        let dt = 0.01;
        let (f, _q) = p.jacobian_noise(&x, &u, dt);
        let x_next = p.predict(&x, &u, dt);

        // Act + Assert — central-difference each error-state column and compare to F.
        let n = MtbState::DOF;
        let eps = 1e-6;
        for j in 0..n {
            let mut plus = DVector::zeros(n);
            plus[j] = eps;
            let mut minus = DVector::zeros(n);
            minus[j] = -eps;
            let col = (p.predict(&x.oplus(&plus), &u, dt).ominus(&x_next)
                - p.predict(&x.oplus(&minus), &u, dt).ominus(&x_next))
                / (2.0 * eps);
            for i in 0..n {
                assert_relative_eq!(f[(i, j)], col[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn process_noise_is_positive_on_active_and_zero_on_frozen_steering() {
        // Arrange — M2a freezes steering (ψ, ψ̇ at indices 22, 23).
        let p = full_sus_process();

        // Act
        let (_f, q) = p.jacobian_noise(&rest_state(), &rest_input(), 0.01);

        // Assert — active blocks carry process noise; frozen steering carries none.
        assert!(q[(0, 0)] > 0.0); // attitude
        assert!(q[(3, 3)] > 0.0); // velocity
        assert!(q[(19, 19)] > 0.0); // front wheel velocity (active)
        assert_eq!(q[(22, 22)], 0.0); // ψ frozen
        assert_eq!(q[(23, 23)], 0.0); // ψ̇ frozen
    }
}
