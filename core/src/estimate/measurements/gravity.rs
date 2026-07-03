//! Acceleration-compensated gravity-leveling (design §5). A genuine **2-DOF**
//! residual on the gravity *direction* (the S² tangent at nav-up), never a 3-vector
//! SO3 log with a dropped yaw row: the residual is the horizontal (x, y) components
//! of the measured "up" direction expressed in the nav frame — zero when level, and
//! intrinsically yaw-null (rotating about nav-Z leaves a vertical vector unchanged).
//!
//! The estimated kinematic acceleration `a_kin_nav` (from `v̇_chassis`, runner-fed)
//! is subtracted so the accelerometer isolates gravity even under
//! braking/cornering/pumping; `a_kin_nav = 0` (static / unknown) reduces to plain
//! gravity-leveling. Pins tilt + the tilt-projected accel bias `b_a0`.

use crate::estimate::measurements::{I_BA0, I_THETA};
use crate::estimate::model::{MeasurementModel, SampleContext};
use crate::estimate::state::MtbState;
use crate::rotation::skew;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Gravity-leveling factor for one sample. `accel0` is the IMU0 specific force
/// (chassis frame); `a_kin_nav` is the compensating chassis kinematic acceleration
/// in the nav frame (0 when static/unknown).
#[derive(Debug, Clone, Copy)]
pub struct GravityLeveling {
    /// IMU0 specific force, chassis frame, m/s².
    pub accel0: Vector3<f64>,
    /// Compensating kinematic acceleration, nav frame, m/s².
    pub a_kin_nav: Vector3<f64>,
    /// Per-axis tilt residual std-dev (dimensionless direction components).
    pub sigma: f64,
}

impl GravityLeveling {
    /// Measured body-frame "up" (unnormalized) and its norm: `m = a₀ − b_a0 − Rᵀ·a_kin`.
    fn up_body_raw(&self, x: &MtbState) -> Vector3<f64> {
        self.accel0 - x.b_a0 - x.r_chassis.inverse() * self.a_kin_nav
    }
}

impl MeasurementModel<MtbState> for GravityLeveling {
    fn dim(&self) -> usize {
        2
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        // Measured up in body, rotated to nav; horizontal components are the tilt error.
        // In (near) free fall the specific force → 0, so there is no measurable up
        // direction: return a zero residual (no information) rather than normalizing a
        // ~0 vector into NaN. `run()` also gates this factor off when airborne; this is
        // the per-sample numerical backstop (a degenerate factor must never NaN-poison
        // the run). Zero residual pairs with the zero Jacobian below ⇒ K=0 ⇒ no-op.
        match self.up_body_raw(x).try_normalize(1e-9) {
            Some(u) => {
                let up_nav = x.r_chassis * u;
                DVector::from_row_slice(&[up_nav.x, up_nav.y])
            }
            None => DVector::zeros(2),
        }
    }

    fn jacobian(&self, x: &MtbState) -> DMatrix<f64> {
        // p = R·û, û = m/‖m‖, m = a₀ − b_a0 − Rᵀ·a_kin. With ∂û/∂m = P_û/‖m‖
        // (P_û = I − ûûᵀ), ∂m/∂δθ = −[Rᵀ·a_kin]×, ∂m/∂δb_a0 = −I, and the explicit
        // rotation term ∂(R û)/∂δθ = −R·[û]×:
        //   ∂p/∂δθ   = R·(P_û/‖m‖)·(−[Rᵀ·a_kin]×) − R·[û]×
        //   ∂p/∂δb_a0 = −R·(P_û/‖m‖)
        // The residual is rows {x, y} of p.
        let r_mat = x.r_chassis.to_rotation_matrix().into_inner();
        let m = self.up_body_raw(x);
        let norm = m.norm();
        // No usable gravity direction (free fall) → zero sensitivity (⇒ K=0, a no-op),
        // matching the zero residual above and guarding the 1/‖m‖ terms from NaN/inf.
        if !(norm > 1e-9) {
            return DMatrix::zeros(2, MtbState::DOF);
        }
        let u = m / norm;
        let p_u = (Matrix3::identity() - u * u.transpose()) / norm;
        let rt_akin = x.r_chassis.inverse() * self.a_kin_nav;
        let dp_dtheta = r_mat * p_u * (-skew(rt_akin)) - r_mat * skew(u);
        let dp_dba = -(r_mat * p_u);

        let mut h = DMatrix::zeros(2, MtbState::DOF);
        for row in 0..2 {
            for k in 0..3 {
                h[(row, I_THETA + k)] = dp_dtheta[(row, k)];
                h[(row, I_BA0 + k)] = dp_dba[(row, k)];
            }
        }
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(2, 2) * self.sigma.powi(2)
    }

    fn active(&self, _x: &MtbState, _ctx: &SampleContext) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::measurements::finite_difference_jacobian;
    use crate::estimate::process::GRAVITY;
    use crate::rotation::exp_so3;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    fn level_state() -> MtbState {
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

    #[test]
    fn residual_is_zero_when_level_and_aligned() {
        // Arrange — level chassis reads +g up; no kinematic accel.
        let f = GravityLeveling {
            accel0: Vector3::new(0.0, 0.0, GRAVITY),
            a_kin_nav: Vector3::zeros(),
            sigma: 0.02,
        };

        // Act
        let r = f.residual(&level_state());

        // Assert
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn residual_is_nonzero_when_tilted() {
        // Arrange — true chassis pitched, but estimate still thinks it is level, so
        // the measured up has a horizontal component.
        let mut x = level_state();
        x.r_chassis = UnitQuaternion::identity();
        // Accelerometer reads gravity rotated into a pitched body (5° about Y).
        let tilt = exp_so3(Vector3::new(0.0, 5_f64.to_radians(), 0.0));
        let f = GravityLeveling {
            accel0: tilt.inverse() * Vector3::new(0.0, 0.0, GRAVITY),
            a_kin_nav: Vector3::zeros(),
            sigma: 0.02,
        };

        // Act
        let r = f.residual(&x);

        // Assert — a pitch error shows up as a nonzero x-component of nav up.
        assert!(r[0].abs() > 0.05);
    }

    #[test]
    fn residual_and_jacobian_are_finite_in_free_fall() {
        // Arrange — free fall: the measured specific force is ~0, so there is no
        // gravity direction to level against. Normalizing it would be NaN.
        let f = GravityLeveling { accel0: Vector3::zeros(), a_kin_nav: Vector3::zeros(), sigma: 0.02 };

        // Act
        let r = f.residual(&level_state());
        let h = f.jacobian(&level_state());

        // Assert — finite, and a no-information factor (zeros) ⇒ K=0 ⇒ no-op, so it
        // can never poison the filter rather than emitting NaN.
        assert!(r.iter().all(|v| v.is_finite()), "residual is non-finite in free fall");
        assert!(h.iter().all(|v| v.is_finite()), "jacobian is non-finite in free fall");
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn jacobian_matches_finite_difference_static() {
        // Arrange — tilted, biased estimate; no kinematic compensation.
        let mut x = level_state();
        x.r_chassis = UnitQuaternion::from_euler_angles(0.08, -0.12, 0.5);
        x.b_a0 = Vector3::new(0.1, -0.05, 0.07);
        let f = GravityLeveling {
            accel0: Vector3::new(0.6, -0.4, GRAVITY - 0.2),
            a_kin_nav: Vector3::zeros(),
            sigma: 0.02,
        };

        // Act + Assert
        assert_relative_eq!(f.jacobian(&x), finite_difference_jacobian(&f, &x), epsilon = 1e-6);
    }

    #[test]
    fn jacobian_matches_finite_difference_with_acceleration_compensation() {
        // Arrange — nonzero a_kin_nav exercises the Rᵀ·a_kin coupling into δθ.
        let mut x = level_state();
        x.r_chassis = UnitQuaternion::from_euler_angles(0.05, 0.2, -0.3);
        x.b_a0 = Vector3::new(-0.08, 0.04, 0.02);
        let f = GravityLeveling {
            accel0: Vector3::new(1.5, -0.9, GRAVITY + 0.3),
            a_kin_nav: Vector3::new(1.2, -0.7, 0.1),
            sigma: 0.02,
        };

        // Act + Assert
        assert_relative_eq!(f.jacobian(&x), finite_difference_jacobian(&f, &x), epsilon = 1e-6);
    }
}
