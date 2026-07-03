//! Horizontal GPS velocity anchor (speed-over-ground + course, no vertical channel).
//!
//! The logger's GPS delivers speed-over-ground (m/s) and course (degrees clockwise from
//! north) — there is no vertical velocity measurement. The residual is therefore 2-DOF,
//! anchoring only the horizontal (`x`, `y`) components of `v_chassis`. This avoids
//! pinning the vertical velocity with a fabricated z = 0 reading.
//!
//! **Nav-frame convention:** X = north, Y = west, Z = up. The runner converts
//! GPS course θ (degrees CW from north) to a nav-frame velocity vector
//! `v · (cos θ, −sin θ, 0)` before constructing [`GpsVelocity`].
//!
//! **GPS internal Kalman filter caveat:** The GPS module runs its own internal Kalman
//! filter, so its velocity output is smoothed, delayed (~0.1–0.3 s latency), and
//! error-correlated over seconds. Consequently:
//! - The runner latency-corrects the measurement to the epoch it describes (offline ⇒
//!   no causality constraint).
//! - The runner gates updates by a minimum-speed threshold to suppress heading noise
//!   when the bike is stationary.
//! - `sigma` models the error of the *smoothed* GPS output — not raw pseudorange noise;
//!   it is larger than raw Doppler noise but smaller than position-differencing noise.
//!
//! The factor anchors low-frequency horizontal velocity and observes heading while moving,
//! complementing the IMU integration which drifts at higher frequencies.

use crate::estimate::measurements::I_V;
use crate::estimate::model::{MeasurementModel, SampleContext};
use crate::estimate::state::MtbState;
use nalgebra::{DMatrix, DVector, Vector3};

/// Horizontal GPS velocity anchor in the nav frame.
///
/// `measured` holds the full nav-frame vector (X = north, Y = west, Z = up), but only
/// the horizontal components (x, y) enter the residual — the logger provides no vertical
/// velocity. Callers set z = 0 by convention. `sigma` is the per-axis std-dev of the
/// *smoothed* GPS output (m/s); the noise model is `σ²·I₂`.
#[derive(Debug, Clone, Copy)]
pub struct GpsVelocity {
    /// Measured nav-frame velocity (X = north, Y = west, Z = up), m/s.
    /// Only x and y are used in the residual; z should be 0.0.
    pub measured: Vector3<f64>,
    /// Per-axis measurement std-dev for the smoothed GPS output, m/s.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for GpsVelocity {
    /// Returns 2: horizontal components only (x = north, y = west).
    fn dim(&self) -> usize {
        2
    }

    /// Residual `[measured.x − v_chassis.x, measured.y − v_chassis.y]` (m/s).
    ///
    /// Vertical velocity is unconstrained — the z row is absent.
    fn residual(&self, x: &MtbState) -> DVector<f64> {
        let r = self.measured - x.v_chassis;
        DVector::from_row_slice(&[r.x, r.y])
    }

    /// 2 × DOF Jacobian; `∂r_i/∂v_chassis_i = −1` for i in {north, west}.
    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(2, MtbState::DOF);
        for i in 0..2 {
            h[(i, I_V + i)] = -1.0;
        }
        h
    }

    /// 2 × 2 diagonal noise matrix `σ²·I₂`.
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
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    fn moving_state() -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::from_euler_angles(0.05, -0.1, 0.4),
            v_chassis: Vector3::new(5.0, 0.3, -0.1),
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
    fn residual_is_measured_minus_estimated_horizontal_velocity() {
        // Arrange — measured (5.2, 0.0, 0.0); state v = (5.0, 0.3, −0.1)
        let g = GpsVelocity { measured: Vector3::new(5.2, 0.0, 0.0), sigma: 0.2 };
        let x = moving_state();

        // Act
        let r = g.residual(&x);

        // Assert — 2 rows: x error and y error; no z row
        assert_eq!(r.len(), 2);
        assert_relative_eq!(r[0], 0.2, epsilon = 1e-12);   // 5.2 − 5.0
        assert_relative_eq!(r[1], -0.3, epsilon = 1e-12);  // 0.0 − 0.3
    }

    #[test]
    fn residual_vertical_velocity_error_only_is_zero() {
        // Arrange — measured horizontal matches state; only vertical differs
        // measured = (5.0, 0.0, 0.0), state v = (5.0, 0.0, −0.4)
        let g = GpsVelocity { measured: Vector3::new(5.0, 0.0, 0.0), sigma: 0.2 };
        let x = MtbState {
            v_chassis: Vector3::new(5.0, 0.0, -0.4),
            ..moving_state()
        };

        // Act
        let r = g.residual(&x);

        // Assert — vertical is unconstrained; both horizontal residual rows ≈ 0
        assert_eq!(r.len(), 2);
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        // Arrange
        let g = GpsVelocity { measured: Vector3::new(5.2, 0.0, 0.0), sigma: 0.2 };
        let x = moving_state();

        // Act + Assert — finite_difference_jacobian adapts to dim() = 2 automatically
        assert_relative_eq!(g.jacobian(&x), finite_difference_jacobian(&g, &x), epsilon = 1e-7);
    }
}
