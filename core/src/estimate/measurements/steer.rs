//! Steering factors. The fork turns relative to the frame about one axis, so the
//! steer angle `ψ` (rad, positive = the fork turned anticlockwise seen from
//! above, i.e. bars to the left) is one number, seen two ways:
//!
//! - [`SteerRate`]: the fork gyro less the frame gyro, along the steering axis,
//!   is `ψ̇`. A rotation about the axis leaves a vector's component along the
//!   axis alone, so the fork gyro needs no knowledge of `ψ` to be compared.
//! - [`SteerLevel`]: over a second or so the frame and the fork feel the same
//!   specific force. The fork's reading, already turned into the chassis frame
//!   as it sits with the bars straight, has to be turned by `ψ` about the axis
//!   to match the frame's. This pins the drift of the first, and works only
//!   because the axis is not along that force (it leans back by the head
//!   angle's complement).
//!
//! Neither ties the steer angle to how the bike is turning: the gap between the
//! two is the bike's understeer or oversteer, and is left to be read.

use crate::estimate::measurements::{I_BG1, I_DPSI, I_PSI};
use crate::estimate::model::MeasurementModel;
use crate::estimate::state::MtbState;
use nalgebra::{DMatrix, DVector, Vector3};

/// Steer rate from the two gyros.
#[derive(Debug, Clone, Copy)]
pub struct SteerRate {
    /// Frame (IMU0) angular rate in the chassis frame, as measured, rad/s.
    pub frame_rate: Vector3<f64>,
    /// Fork (IMU1) angular rate turned into the chassis frame by its
    /// bars-straight mount, as measured, rad/s.
    pub fork_rate: Vector3<f64>,
    /// Steering axis, chassis frame, unit, pointing up.
    pub axis: Vector3<f64>,
    /// Std-dev of the rate difference along the axis, rad/s.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for SteerRate {
    fn dim(&self) -> usize {
        1
    }

    /// `((ω₁ − b_g1) − (ω₀ − b_g0))·s − ψ̇`, rad/s.
    fn residual(&self, x: &MtbState) -> DVector<f64> {
        let rate = ((self.fork_rate - x.b_g1) - (self.frame_rate - x.b_g0)).dot(&self.axis);
        DVector::from_row_slice(&[rate - x.dpsi])
    }

    /// The frame gyro's offset enters the residual but is given no column: it
    /// is taken as known here. With one, anything this factor cannot explain as
    /// steering (the frame's rotation leaking in through a few degrees of mount
    /// error) is shared out to the frame's bias, and the steering then moves the
    /// frame's heading — 2° on ride `173461a2`. Steering reads the frame; it
    /// does not get to correct it.
    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(1, MtbState::DOF);
        for k in 0..3 {
            h[(0, I_BG1 + k)] = -self.axis[k];
        }
        h[(0, I_DPSI)] = -1.0;
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(1, 1) * self.sigma.powi(2)
    }
}

/// The angle about `axis` that turns `fork_force` onto `frame_force`, rad, and
/// the length of the shorter of the two once the part along the axis is
/// removed (the lever the angle is read with), in the forces' own unit.
pub fn steer_level_angle(frame_force: &Vector3<f64>, fork_force: &Vector3<f64>, axis: &Vector3<f64>) -> (f64, f64) {
    let across = |v: &Vector3<f64>| v - axis * v.dot(axis);
    let (frame, fork) = (across(frame_force), across(fork_force));
    (fork.cross(&frame).dot(axis).atan2(fork.dot(&frame)), frame.norm().min(fork.norm()))
}

/// Steer angle from the two accelerometers agreeing on the slow specific force.
#[derive(Debug, Clone, Copy)]
pub struct SteerLevel {
    /// The steer angle the two forces imply ([`steer_level_angle`]), rad.
    pub angle: f64,
    /// Its std-dev, rad.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for SteerLevel {
    fn dim(&self) -> usize {
        1
    }

    /// `angle − ψ`, wrapped to `(−π, π]`.
    fn residual(&self, x: &MtbState) -> DVector<f64> {
        let d = (self.angle - x.psi + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        DVector::from_row_slice(&[d])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(1, MtbState::DOF);
        h[(0, I_PSI)] = -1.0;
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(1, 1) * self.sigma.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::measurements::finite_difference_jacobian;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    fn state() -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::from_euler_angles(0.05, -0.1, 0.4),
            v_chassis: Vector3::new(5.0, 0.3, -0.1),
            b_g0: Vector3::new(0.01, -0.02, 0.005),
            b_a0: Vector3::zeros(),
            b_g1: Vector3::new(-0.004, 0.003, 0.02),
            b_g2: Vector3::zeros(),
            d_f: 0.0,
            dd_f: 0.0,
            s_r: 0.0,
            ds_r: 0.0,
            psi: 0.15,
            dpsi: -0.3,
        }
    }

    /// A steering axis leaning back 26°.
    fn axis() -> Vector3<f64> {
        Vector3::new(-26_f64.to_radians().sin(), 0.0, 26_f64.to_radians().cos())
    }

    #[test]
    fn steer_rate_residual_is_zero_when_the_gyros_differ_by_the_steer_rate_and_their_offsets() {
        // Arrange — the frame pitches and yaws; the fork does the same plus
        // −0.3 rad/s about the axis; each gyro carries its own offset.
        let x = state();
        let frame_true = Vector3::new(0.2, -0.5, 0.7);
        let m = SteerRate {
            frame_rate: frame_true + x.b_g0,
            fork_rate: frame_true + axis() * x.dpsi + x.b_g1,
            axis: axis(),
            sigma: 0.02,
        };

        // Act
        let r = m.residual(&x);

        // Assert
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn steer_rate_jacobian_matches_finite_difference_except_the_frame_bias_it_leaves_alone() {
        // Arrange
        let m = SteerRate { frame_rate: Vector3::new(0.2, -0.5, 0.7), fork_rate: Vector3::new(0.1, -0.4, 0.2), axis: axis(), sigma: 0.02 };
        let x = state();

        // Act
        let (analytic, mut numeric) = (m.jacobian(&x), finite_difference_jacobian(&m, &x));

        // Assert — the residual does depend on the frame's gyro offset (the axis,
        // by the numbers); the factor deliberately gives it no say.
        for k in 0..3 {
            assert_relative_eq!(numeric[(0, crate::estimate::measurements::I_BG0 + k)], axis()[k], epsilon = 1e-6);
            assert_eq!(analytic[(0, crate::estimate::measurements::I_BG0 + k)], 0.0);
            numeric[(0, crate::estimate::measurements::I_BG0 + k)] = 0.0;
        }
        assert_relative_eq!(analytic, numeric, epsilon = 1e-6);
    }

    #[test]
    fn steer_level_angle_recovers_a_turn_of_the_fork_about_the_axis() {
        // Arrange — the bike upright and accelerating; the fork turned 0.3 rad to
        // the left reads the same force in axes turned with it.
        let s = axis();
        let frame = Vector3::new(1.5, 0.4, 9.7);
        let turn = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(s), 0.3);
        let fork = turn.inverse() * frame;

        // Act
        let (angle, lever) = steer_level_angle(&frame, &fork, &s);

        // Assert — and the lever is the force's part across the axis.
        assert_relative_eq!(angle, 0.3, epsilon = 1e-12);
        assert_relative_eq!(lever, (frame - s * frame.dot(&s)).norm(), epsilon = 1e-12);
    }

    #[test]
    fn steer_level_angle_force_along_the_axis_has_no_lever() {
        // Arrange — nothing across the axis to read an angle from.
        let s = axis();

        // Act
        let (_, lever) = steer_level_angle(&(s * 9.8), &(s * 9.8), &s);

        // Assert
        assert!(lever < 1e-9);
    }

    #[test]
    fn steer_level_residual_is_the_shortest_way_round() {
        // Arrange — measured just past −π, state just short of +π.
        let mut x = state();
        x.psi = 3.1;
        let m = SteerLevel { angle: -3.1, sigma: 0.1 };

        // Act
        let r = m.residual(&x);

        // Assert — 0.083 rad apart, not 6.2.
        assert_relative_eq!(r[0], std::f64::consts::TAU - 6.2, epsilon = 1e-12);
    }

    #[test]
    fn steer_level_jacobian_matches_finite_difference() {
        // Arrange
        let m = SteerLevel { angle: 0.2, sigma: 0.1 };
        let x = state();

        // Act
        let (analytic, numeric) = (m.jacobian(&x), finite_difference_jacobian(&m, &x));

        // Assert
        assert_relative_eq!(analytic, numeric, epsilon = 1e-6);
    }
}
