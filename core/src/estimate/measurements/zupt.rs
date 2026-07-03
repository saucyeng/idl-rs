//! Zero-velocity (ZUPT) and zero-angular-rate (ZARU) pseudo-measurements — the
//! strongest bias/velocity pins, active only on stationary-flagged samples
//! (design §5). ZUPT pins `v_chassis = 0`; ZARU pins a gyro-bias block to the
//! measured rate (true rate ≈ 0 when stationary ⇒ reading = bias).

use crate::estimate::measurements::prior::Wheel;
use crate::estimate::measurements::{I_BG0, I_BG1, I_BG2, I_DWF, I_DWR, I_V};
use crate::estimate::model::{MeasurementModel, SampleContext};
use crate::estimate::state::MtbState;
use nalgebra::{DMatrix, DVector, Vector3};

/// ZUPT: `v_chassis = 0` on stationary samples. Residual `r = 0 ⊟ v = −v`.
#[derive(Debug, Clone, Copy)]
pub struct ZeroVelocity {
    /// Per-axis velocity pseudo-measurement std-dev, m/s.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for ZeroVelocity {
    fn dim(&self) -> usize {
        3
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        DVector::from_row_slice(&[-x.v_chassis.x, -x.v_chassis.y, -x.v_chassis.z])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(3, MtbState::DOF);
        for i in 0..3 {
            h[(i, I_V + i)] = -1.0;
        }
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(3, 3) * self.sigma.powi(2)
    }

    fn active(&self, _x: &MtbState, ctx: &SampleContext) -> bool {
        ctx.stationary
    }
}

/// Which gyro-bias block ZARU pins (selects the error-state column block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GyroBias {
    /// IMU0 (chassis) gyro bias, columns `I_BG0..`.
    Imu0,
    /// IMU1 (front-unsprung) gyro bias, columns `I_BG1..`.
    Imu1,
    /// IMU2 (rear-unsprung) gyro bias, columns `I_BG2..`.
    Imu2,
}

impl GyroBias {
    fn col(self) -> usize {
        match self {
            GyroBias::Imu0 => I_BG0,
            GyroBias::Imu1 => I_BG1,
            GyroBias::Imu2 => I_BG2,
        }
    }

    fn bias_of(self, x: &MtbState) -> Vector3<f64> {
        match self {
            GyroBias::Imu0 => x.b_g0,
            GyroBias::Imu1 => x.b_g1,
            GyroBias::Imu2 => x.b_g2,
        }
    }
}

/// ZARU: pins a gyro-bias block to the measured angular rate on stationary samples.
/// Residual `r = ω_measured ⊟ b = ω_measured − b`.
#[derive(Debug, Clone, Copy)]
pub struct ZeroAngularRate {
    /// Which bias block to pin.
    pub target: GyroBias,
    /// Measured angular rate for this IMU at this sample (chassis/sensor frame), rad/s.
    pub measured: Vector3<f64>,
    /// Per-axis pseudo-measurement std-dev, rad/s.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for ZeroAngularRate {
    fn dim(&self) -> usize {
        3
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        let r = self.measured - self.target.bias_of(x);
        DVector::from_row_slice(&[r.x, r.y, r.z])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(3, MtbState::DOF);
        let c = self.target.col();
        for i in 0..3 {
            h[(i, c + i)] = -1.0;
        }
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(3, 3) * self.sigma.powi(2)
    }

    fn active(&self, _x: &MtbState, ctx: &SampleContext) -> bool {
        ctx.stationary
    }
}

/// Wheel-velocity zero pin: pins a wheel's travel-rate `ẇ = 0`. The travel rate is
/// zero in two regimes — a motionless (stationary) bike, and a wheel **topped out in
/// free fall** (airborne): the suspension is held against its top stop, so it isn't
/// articulating. Nothing else pins `ẇ` directly (the diff-accel only drives it), so
/// without this it random-walks and its uncertainty inflates travel. Residual
/// `r = 0 − ẇ = −ẇ`. **Runner-gated** (like the topout/sag/barrier travel factors):
/// `active` is always true and the runner applies it on stationary and airborne
/// samples.
#[derive(Debug, Clone, Copy)]
pub struct ZeroWheelVelocity {
    /// Target wheel.
    pub wheel: Wheel,
    /// Pseudo-measurement std, m/s.
    pub sigma: f64,
}

impl ZeroWheelVelocity {
    fn col(self) -> usize {
        match self.wheel {
            Wheel::Front => I_DWF,
            Wheel::Rear => I_DWR,
        }
    }

    fn velocity_of(self, x: &MtbState) -> f64 {
        match self.wheel {
            Wheel::Front => x.dd_f,
            Wheel::Rear => x.ds_r,
        }
    }
}

impl MeasurementModel<MtbState> for ZeroWheelVelocity {
    fn dim(&self) -> usize {
        1
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        DVector::from_row_slice(&[-self.velocity_of(x)])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(1, MtbState::DOF);
        h[(0, self.col())] = -1.0;
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.sigma.powi(2))
    }

    // Runner-gated: the runner decides when ẇ = 0 holds (stationary or airborne), so
    // this stays always-active, matching the topout/sag/barrier travel factors.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::measurements::finite_difference_jacobian;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    fn state_with(v: Vector3<f64>, bg0: Vector3<f64>) -> MtbState {
        let mut x = zero_state();
        x.v_chassis = v;
        x.b_g0 = bg0;
        x
    }

    fn zero_state() -> MtbState {
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
    fn zupt_residual_is_negative_velocity() {
        // Arrange
        let z = ZeroVelocity { sigma: 0.01 };
        let x = state_with(Vector3::new(1.0, -2.0, 0.5), Vector3::zeros());

        // Act
        let r = z.residual(&x);

        // Assert — r = 0 − v.
        assert_relative_eq!(r[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(r[2], -0.5, epsilon = 1e-12);
    }

    #[test]
    fn zupt_jacobian_matches_finite_difference() {
        // Arrange — a rotated, biased state so any spurious coupling would show.
        let z = ZeroVelocity { sigma: 0.01 };
        let mut x = state_with(Vector3::new(0.3, 0.7, -0.2), Vector3::new(0.01, 0.0, 0.0));
        x.r_chassis = UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3);

        // Act + Assert
        assert_relative_eq!(z.jacobian(&x), finite_difference_jacobian(&z, &x), epsilon = 1e-7);
    }

    #[test]
    fn zupt_active_only_when_stationary() {
        // Arrange
        let z = ZeroVelocity { sigma: 0.01 };
        let x = zero_state();

        // Act + Assert
        assert!(z.active(&x, &SampleContext { stationary: true }));
        assert!(!z.active(&x, &SampleContext { stationary: false }));
    }

    #[test]
    fn zaru_residual_is_measured_minus_bias() {
        // Arrange — measured rate 0.01/0.02/0.03, current b_g0 estimate offset.
        let z = ZeroAngularRate {
            target: GyroBias::Imu0,
            measured: Vector3::new(0.01, 0.02, 0.03),
            sigma: 1e-3,
        };
        let x = state_with(Vector3::zeros(), Vector3::new(0.005, 0.0, 0.0));

        // Act
        let r = z.residual(&x);

        // Assert
        assert_relative_eq!(r[0], 0.005, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.02, epsilon = 1e-12);
        assert_relative_eq!(r[2], 0.03, epsilon = 1e-12);
    }

    #[test]
    fn wheel_zupt_pins_velocity_and_matches_finite_difference() {
        // Arrange — front wheel velocity 0.3 m/s.
        let z = ZeroWheelVelocity { wheel: Wheel::Front, sigma: 0.02 };
        let mut x = zero_state();
        x.dd_f = 0.3;

        // Act + Assert — residual = −ẇ; Jacobian matches FD. The wheel-velocity pin is
        // **runner-gated** (fires on stationary AND airborne/topped-out samples — both
        // have ẇ = 0), like the topout/sag/barrier travel factors, so `active` always
        // returns true and the runner decides when to apply it.
        assert_relative_eq!(z.residual(&x)[0], -0.3, epsilon = 1e-12);
        assert_relative_eq!(z.jacobian(&x), finite_difference_jacobian(&z, &x), epsilon = 1e-7);
        assert!(z.active(&x, &SampleContext { stationary: true }));
        assert!(z.active(&x, &SampleContext { stationary: false }));
    }

    #[test]
    fn zaru_jacobian_matches_finite_difference_for_each_target() {
        // Arrange
        let mut x = zero_state();
        x.b_g1 = Vector3::new(0.001, -0.002, 0.0);
        x.b_g2 = Vector3::new(0.0, 0.003, -0.001);

        // Act + Assert — all three IMU bias targets land in the right column block.
        for target in [GyroBias::Imu0, GyroBias::Imu1, GyroBias::Imu2] {
            let z = ZeroAngularRate { target, measured: Vector3::new(0.01, 0.02, 0.03), sigma: 1e-3 };
            assert_relative_eq!(z.jacobian(&x), finite_difference_jacobian(&z, &x), epsilon = 1e-7);
        }
    }
}
