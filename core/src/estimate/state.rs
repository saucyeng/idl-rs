//! The concrete MTB error-state vector and its boxplus/boxminus retraction on the
//! product manifold SO(3) ⊕ ℝⁿ. Full-suspension composition (24 DOF); the
//! geometry-derived schema that drops rear/steer states on other topologies is an
//! M2 concern.

use crate::estimate::model::ErrorState;
use crate::rotation::{exp_so3, log_so3};
use nalgebra::{DVector, UnitQuaternion, Vector3};

/// The full-suspension MTB estimator state. Error/perturbation ordering (length
/// [`MtbState::DOF`]): `[δθ(3), δv(3), δb_g0(3), δb_a0(3), δb_g1(3), δb_g2(3),
/// δd_f, δḋ_f, δs_r, δṡ_r, δψ, δψ̇]`.
#[derive(Debug, Clone)]
pub struct MtbState {
    /// Chassis (IMU0) attitude, sensor/vehicle → nav. SO(3).
    pub r_chassis: UnitQuaternion<f64>,
    /// Chassis velocity in the nav frame, m/s.
    pub v_chassis: Vector3<f64>,
    /// IMU0 gyro bias, rad/s.
    pub b_g0: Vector3<f64>,
    /// IMU0 accel bias, m/s².
    pub b_a0: Vector3<f64>,
    /// Front-unsprung (IMU1) gyro bias, rad/s.
    pub b_g1: Vector3<f64>,
    /// Rear-unsprung (IMU2) gyro bias, rad/s.
    pub b_g2: Vector3<f64>,
    /// Front wheel travel along the axle-path tangent, m.
    pub d_f: f64,
    /// Front wheel velocity, m/s.
    pub dd_f: f64,
    /// Rear wheel travel along the axle-path tangent, m.
    pub s_r: f64,
    /// Rear wheel velocity, m/s.
    pub ds_r: f64,
    /// Steer angle about the known steer axis, rad.
    pub psi: f64,
    /// Steer rate, rad/s.
    pub dpsi: f64,
}

impl MtbState {
    /// Error-state (tangent) dimension of the full-suspension composition.
    pub const DOF: usize = 24;
}

impl ErrorState for MtbState {
    fn dof(&self) -> usize {
        Self::DOF
    }

    fn oplus(&self, d: &DVector<f64>) -> Self {
        // SO(3) block: right perturbation R·exp([δθ]_×); all others add.
        MtbState {
            r_chassis: self.r_chassis * exp_so3(Vector3::new(d[0], d[1], d[2])),
            v_chassis: self.v_chassis + Vector3::new(d[3], d[4], d[5]),
            b_g0: self.b_g0 + Vector3::new(d[6], d[7], d[8]),
            b_a0: self.b_a0 + Vector3::new(d[9], d[10], d[11]),
            b_g1: self.b_g1 + Vector3::new(d[12], d[13], d[14]),
            b_g2: self.b_g2 + Vector3::new(d[15], d[16], d[17]),
            d_f: self.d_f + d[18],
            dd_f: self.dd_f + d[19],
            s_r: self.s_r + d[20],
            ds_r: self.ds_r + d[21],
            psi: self.psi + d[22],
            dpsi: self.dpsi + d[23],
        }
    }

    fn ominus(&self, o: &Self) -> DVector<f64> {
        // SO(3) block under the right-perturbation convention: log(R_o⁻¹ · R_self).
        let dtheta = log_so3(&(o.r_chassis.inverse() * self.r_chassis));
        let dv = self.v_chassis - o.v_chassis;
        let dbg0 = self.b_g0 - o.b_g0;
        let dba0 = self.b_a0 - o.b_a0;
        let dbg1 = self.b_g1 - o.b_g1;
        let dbg2 = self.b_g2 - o.b_g2;
        DVector::from_vec(vec![
            dtheta[0], dtheta[1], dtheta[2],
            dv[0], dv[1], dv[2],
            dbg0[0], dbg0[1], dbg0[2],
            dba0[0], dba0[1], dba0[2],
            dbg1[0], dbg1[1], dbg1[2],
            dbg2[0], dbg2[1], dbg2[2],
            self.d_f - o.d_f,
            self.dd_f - o.dd_f,
            self.s_r - o.s_r,
            self.ds_r - o.ds_r,
            self.psi - o.psi,
            self.dpsi - o.dpsi,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::model::ErrorState;
    use crate::rotation::exp_so3;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Vector3};

    fn sample_state() -> MtbState {
        MtbState {
            r_chassis: exp_so3(Vector3::new(0.1, -0.2, 0.3)),
            v_chassis: Vector3::new(1.0, 2.0, 3.0),
            b_g0: Vector3::new(0.01, 0.02, 0.03),
            b_a0: Vector3::new(0.1, 0.2, 0.3),
            b_g1: Vector3::new(0.001, 0.002, 0.003),
            b_g2: Vector3::new(0.004, 0.005, 0.006),
            d_f: 0.05,
            dd_f: -0.1,
            s_r: 0.04,
            ds_r: 0.2,
            psi: 0.3,
            dpsi: -0.5,
        }
    }

    #[test]
    fn dof_is_24() {
        // 3 (SO3) + 3 (v) + 4×3 (biases) + 3×2 (suspension/steer pairs) = 24.
        assert_eq!(sample_state().dof(), 24);
    }

    #[test]
    fn oplus_zero_is_identity() {
        // Arrange
        let x = sample_state();

        // Act — x ⊞ 0 must equal x (checked via (x⊞0) ⊟ x ≈ 0).
        let x2 = x.oplus(&DVector::zeros(x.dof()));

        // Assert
        assert!(x2.ominus(&x).norm() < 1e-12);
    }

    #[test]
    fn ominus_self_is_zero() {
        // Arrange
        let x = sample_state();

        // Act + Assert
        assert!(x.ominus(&x).norm() < 1e-12);
    }

    #[test]
    fn oplus_then_ominus_round_trips() {
        // Arrange — a generic tangent perturbation of length dof().
        let x = sample_state();
        let delta = DVector::from_iterator(
            24,
            (0..24).map(|i| 0.01 * (i as f64 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 }),
        );

        // Act — (x ⊞ δ) ⊟ x must recover δ (the SO(3) block round-trips under the
        // pinned right-perturbation convention).
        let rt = x.oplus(&delta).ominus(&x);

        // Assert
        assert_relative_eq!(rt, delta, epsilon = 1e-9);
    }
}
