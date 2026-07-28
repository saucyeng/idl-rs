//! Attitude and gravity-removed acceleration, read out of the estimator's
//! chassis rotation. Pure — no filter state, no I/O — so the sign conventions
//! are testable against known orientations without running a filter.
//!
//! Frame is ISO 8855 chassis: X forward, Y left, Z up (SPEC §9). `r_chassis`
//! maps chassis → nav. See
//! `docs/superpowers/specs/2026-07-11-attitude-and-overlay-visualization-design.md` §5.1.

use nalgebra::{UnitQuaternion, Vector3};

/// Standard gravity, m/s².
pub const G_MPS2: f64 = 9.80665;

/// Roll and pitch in **degrees** from the chassis attitude.
///
/// - `roll` > 0 ⇒ leaning **right** (right side down)
/// - `pitch` > 0 ⇒ **nose up** (climbing)
///
/// nalgebra's `euler_angles()` decomposes the rotation as roll about X, pitch
/// about Y, yaw about Z. In ISO 8855 (Y left, Z up) a positive rotation about
/// X drops the right side — so roll passes through — while a positive rotation
/// about Y tips the nose *down*, so pitch is negated to report climbing as
/// positive. Yaw is deliberately not returned: it is a free gauge at rest and
/// only weakly observed via GPS course.
pub fn roll_pitch_deg(r_chassis: &UnitQuaternion<f64>) -> (f64, f64) {
    let (roll, pitch, _yaw) = r_chassis.euler_angles();
    (roll.to_degrees(), -pitch.to_degrees())
}

/// Gravity-removed acceleration in the **chassis body frame**, in g.
///
/// Returns `(longitudinal, lateral)`:
/// - `longitudinal` > 0 ⇒ accelerating **forward**
/// - `lateral` > 0 ⇒ accelerating **right**
///
/// `f_body` is the measured specific force in the chassis frame, m/s² (level
/// and stationary it reads `+g` on Z). Since `f = a − g`, the true
/// acceleration is `a_body = f_body + Rᵀ·g_nav` with `g_nav = (0, 0, −G)`;
/// `UnitQuaternion::inverse_transform_vector` applies `Rᵀ`.
///
/// Resolving in the **body** frame is deliberate: it needs only the tilt part
/// of the attitude, which is well observed, and never touches yaw, which is
/// not. Y is left in ISO 8855, so the lateral component is negated to report
/// right-positive.
pub fn body_accel_g(f_body: &Vector3<f64>, r_chassis: &UnitQuaternion<f64>) -> (f64, f64) {
    let g_nav = Vector3::new(0.0, 0.0, -G_MPS2);
    let a_body = f_body + r_chassis.inverse_transform_vector(&g_nav);
    (a_body.x / G_MPS2, -a_body.y / G_MPS2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    /// Chassis-frame specific force for a level, stationary bike: the
    /// accelerometer reads +1 g on Z (SPEC §9).
    fn upright_specific_force() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, G_MPS2)
    }

    #[test]
    fn roll_pitch_of_level_attitude_is_zero() {
        // Arrange
        let level = UnitQuaternion::identity();

        // Act
        let (roll, pitch) = roll_pitch_deg(&level);

        // Assert
        assert!(roll.abs() < 1e-9, "roll {roll}");
        assert!(pitch.abs() < 1e-9, "pitch {pitch}");
    }

    #[test]
    fn positive_rotation_about_forward_axis_reads_as_leaning_right() {
        // Arrange — X is forward; by the right-hand rule a positive rotation
        // about it lifts +Y (left) toward +Z (up), i.e. the right side drops.
        let leaned = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 30f64.to_radians());

        // Act
        let (roll, pitch) = roll_pitch_deg(&leaned);

        // Assert — right side down is reported positive.
        assert!((roll - 30.0).abs() < 1e-6, "roll {roll}");
        assert!(pitch.abs() < 1e-6, "pitch {pitch}");
    }

    #[test]
    fn positive_rotation_about_left_axis_reads_as_nose_down() {
        // Arrange — Y is left; a positive rotation about it tips +X (forward)
        // toward -Z (down). Reported pitch is nose-up-positive, so this is
        // negative.
        let nose_down = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 10f64.to_radians());

        // Act
        let (_roll, pitch) = roll_pitch_deg(&nose_down);

        // Assert
        assert!((pitch - -10.0).abs() < 1e-6, "pitch {pitch}");
    }

    #[test]
    fn climbing_reads_as_positive_pitch() {
        // Arrange — the opposite sense: nose up 10°.
        let nose_up = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -10f64.to_radians());

        // Act
        let (_roll, pitch) = roll_pitch_deg(&nose_up);

        // Assert
        assert!((pitch - 10.0).abs() < 1e-6, "pitch {pitch}");
    }

    #[test]
    fn body_accel_of_level_stationary_bike_is_zero() {
        // Arrange
        let level = UnitQuaternion::identity();

        // Act
        let (long, lat) = body_accel_g(&upright_specific_force(), &level);

        // Assert — gravity fully removed.
        assert!(long.abs() < 1e-12, "long {long}");
        assert!(lat.abs() < 1e-12, "lat {lat}");
    }

    #[test]
    fn body_accel_removes_gravity_even_when_leaned() {
        // Arrange — leaned right 30° and stationary. The accelerometer now
        // reads gravity spread across Y and Z, but true acceleration is zero.
        // This is the berm blind spot the whole design exists to fix: a naive
        // atan2(Ay, Az) would report a lean here while reporting none mid-corner.
        let lean = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 30f64.to_radians());
        let f_body = lean.inverse_transform_vector(&Vector3::new(0.0, 0.0, G_MPS2));

        // Act
        let (long, lat) = body_accel_g(&f_body, &lean);

        // Assert
        assert!(long.abs() < 1e-9, "long {long}");
        assert!(lat.abs() < 1e-9, "lat {lat}");
    }

    #[test]
    fn forward_acceleration_reads_positive_longitudinal() {
        // Arrange — level, accelerating forward at 0.5 g. Specific force is
        // f = a - g, so it carries both the forward term and the +1 g on Z.
        let level = UnitQuaternion::identity();
        let f_body = Vector3::new(0.5 * G_MPS2, 0.0, G_MPS2);

        // Act
        let (long, lat) = body_accel_g(&f_body, &level);

        // Assert
        assert!((long - 0.5).abs() < 1e-12, "long {long}");
        assert!(lat.abs() < 1e-12, "lat {lat}");
    }

    #[test]
    fn rightward_acceleration_reads_positive_lateral() {
        // Arrange — level, accelerating to the RIGHT at 0.3 g. Y is left, so
        // rightward acceleration is -Y in the body frame.
        let level = UnitQuaternion::identity();
        let f_body = Vector3::new(0.0, -0.3 * G_MPS2, G_MPS2);

        // Act
        let (_long, lat) = body_accel_g(&f_body, &level);

        // Assert — reported right-positive.
        assert!((lat - 0.3).abs() < 1e-12, "lat {lat}");
    }
}