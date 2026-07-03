//! Orientation + bias pre-step (design §7). Static gravity gives **tilt** (roll +
//! pitch), not yaw: over a near-stationary window the mean specific force is the
//! "up" direction, so the initial chassis attitude is the rotation taking measured
//! up onto nav +Z, and the mean angular rate is the per-IMU gyro bias (true rate ≈ 0
//! when stationary). Heading/yaw needs motion (GPS course) and is left at the datum.
//!
//! The coarse "which way is it mounted" pick is carried by `BikeGeometry`'s per-IMU
//! mount; this step refines tilt + biases on top, from the log itself. Inputs here
//! are already in the chassis frame (mount applied by the caller).

use crate::rotation::rotation_between;
use nalgebra::{UnitQuaternion, Vector3};

/// Initial attitude + per-IMU gyro biases resolved from a stationary window.
#[derive(Debug, Clone)]
pub struct OrientationFit {
    /// Initial chassis attitude (tilt; yaw at the datum).
    pub r_chassis0: UnitQuaternion<f64>,
    /// IMU0 gyro bias, rad/s.
    pub b_g0: Vector3<f64>,
    /// IMU1 gyro bias, rad/s.
    pub b_g1: Vector3<f64>,
    /// IMU2 gyro bias, rad/s (zero if no rear IMU).
    pub b_g2: Vector3<f64>,
}

/// Mean of a vector slice (zero for an empty slice).
fn mean(v: &[Vector3<f64>]) -> Vector3<f64> {
    if v.is_empty() {
        return Vector3::zeros();
    }
    v.iter().sum::<Vector3<f64>>() / v.len() as f64
}

/// Initial chassis attitude from the mean specific force over a stationary window:
/// the rotation mapping measured "up" (the specific-force direction, chassis frame)
/// onto nav +Z. Level → identity.
pub fn initial_attitude(mean_accel0: Vector3<f64>) -> UnitQuaternion<f64> {
    rotation_between(mean_accel0, Vector3::z())
}

/// Refines a **coarse** sensor→chassis mount (§7 coarse pick) so the measured
/// static gravity — the mean specific force in the sensor frame over a stationary
/// window — maps onto `up_ref` in the chassis frame. This corrects the mount's
/// **tilt** (roll/pitch) to the data while preserving the coarse pick's discrete
/// orientation + in-plane (yaw) datum (the part gravity can't observe). For an
/// unsprung IMU on a rigid link, the fitted tilt is the link's static tilt.
///
/// `coarse`: rough sensor→chassis rotation.
/// `mean_accel_sensor`: mean accel over a stationary window, sensor frame (≈ +g).
/// `up_ref`: the chassis-frame direction the sensor's static gravity must map onto.
///   Pass the **mounted IMU0's mean specific force over the same stationary window**
///   (i.e. `imu0_mount * mean(imu0_accel[ws..we])`), not `Vector3::z()`. Using +Z
///   silently assumes the bike was parked perfectly upright: any kickstand lean or
///   bar-flop is absorbed as a permanent mount-tilt error and later leaks as a DC
///   gravity component into the differential-acceleration wheel-drive signal.
///
/// `up_ref` need not be normalised — [`rotation_between`] operates on directions
/// only (it normalises both arguments internally).
///
/// Returns the refined mount.
pub fn refine_mount_tilt(
    coarse: UnitQuaternion<f64>,
    mean_accel_sensor: Vector3<f64>,
    up_ref: Vector3<f64>,
) -> UnitQuaternion<f64> {
    let up_chassis = coarse * mean_accel_sensor;
    rotation_between(up_chassis, up_ref) * coarse
}

/// Refines an unsprung IMU's **coarse** mount against the mean specific force over
/// the stationary window `[start, end)` of its *sensor-frame* accel samples — the
/// per-session "auto-fit" layered on the coarse pick (§7). Gravity over the window
/// fixes the link's static tilt; the coarse pick supplies the discrete orientation
/// gravity can't observe. Returns `coarse` unchanged for an empty slice/window
/// (nothing to fit against). `accel_sensor` is in the sensor frame (mount not yet
/// applied), m/s².
///
/// `up_ref`: chassis-frame reference direction the measured gravity must align to —
/// pass the mounted IMU0's mean specific force over the **same** window, not +Z.
/// See [`refine_mount_tilt`] for why +Z is incorrect when the bike is parked at an
/// angle.
pub fn refine_mount_from_window(
    coarse: UnitQuaternion<f64>,
    accel_sensor: &[Vector3<f64>],
    start: usize,
    end: usize,
    up_ref: Vector3<f64>,
) -> UnitQuaternion<f64> {
    if accel_sensor.is_empty() {
        return coarse;
    }
    let lo = start.min(accel_sensor.len());
    let hi = end.min(accel_sensor.len());
    if lo >= hi {
        return coarse;
    }
    refine_mount_tilt(coarse, mean(&accel_sensor[lo..hi]), up_ref)
}

/// Fits initial attitude + gyro biases from the stationary sample window
/// `[start, end)`. IMU1/IMU2 gyro slices may be empty (bias → 0).
pub fn fit_from_window(
    imu0_accel: &[Vector3<f64>],
    imu0_gyro: &[Vector3<f64>],
    imu1_gyro: &[Vector3<f64>],
    imu2_gyro: &[Vector3<f64>],
    start: usize,
    end: usize,
) -> OrientationFit {
    let win = |v: &[Vector3<f64>]| -> Vec<Vector3<f64>> {
        if v.is_empty() {
            vec![]
        } else {
            v[start.min(v.len())..end.min(v.len())].to_vec()
        }
    };
    OrientationFit {
        r_chassis0: initial_attitude(mean(&win(imu0_accel))),
        b_g0: mean(&win(imu0_gyro)),
        b_g1: mean(&win(imu1_gyro)),
        b_g2: mean(&win(imu2_gyro)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;


    #[test]
    fn level_window_yields_identity_attitude_and_zero_bias() {
        // Arrange — 20 samples of level gravity, zero rate.
        let accel = vec![Vector3::new(0.0, 0.0, 9.81); 20];
        let gyro = vec![Vector3::zeros(); 20];

        // Act
        let fit = fit_from_window(&accel, &gyro, &[], &[], 0, 20);

        // Assert
        assert_relative_eq!(fit.r_chassis0.angle(), 0.0, epsilon = 1e-9);
        assert_relative_eq!(fit.b_g0, Vector3::zeros(), epsilon = 1e-12);
    }

    #[test]
    fn tilted_window_levels_the_measured_gravity() {
        // Arrange — the chassis is pitched, so the accelerometer's "up" is tilted;
        // the fitted attitude must rotate that measured up back onto nav +Z.
        let tilt = UnitQuaternion::from_euler_angles(0.0, 8_f64.to_radians(), 0.0);
        let up_meas = tilt * Vector3::new(0.0, 0.0, 9.81);
        let accel = vec![up_meas; 10];
        let gyro = vec![Vector3::zeros(); 10];

        // Act
        let fit = fit_from_window(&accel, &gyro, &[], &[], 0, 10);

        // Assert — applying the fit levels the measurement.
        let leveled = fit.r_chassis0 * up_meas;
        assert_relative_eq!(leveled.normalize(), Vector3::z(), epsilon = 1e-9);
    }

    #[test]
    fn refine_mount_tilt_aligns_measured_gravity_to_chassis_up() {
        // Arrange — a true mount that is the coarse pick off by a 6° tilt error. The
        // sensor sees gravity as `true_mount⁻¹·(g·ẑ)`; refining the coarse mount must
        // map that measured gravity back onto chassis +Z.
        let coarse = UnitQuaternion::from_euler_angles(0.0, 0.0, 90_f64.to_radians()); // Z↔ axes swap
        let tilt_err = UnitQuaternion::from_euler_angles(6_f64.to_radians(), 0.0, 0.0);
        let true_mount = tilt_err * coarse;
        let mean_accel_sensor = true_mount.inverse() * (9.81 * Vector3::z());

        // Act
        let refined = refine_mount_tilt(coarse, mean_accel_sensor, Vector3::z());

        // Assert — refined maps the measured gravity onto +Z.
        let up = (refined * mean_accel_sensor).normalize();
        assert_relative_eq!(up, Vector3::z(), epsilon = 1e-9);
    }

    #[test]
    fn refine_mount_tilt_is_identity_when_coarse_is_exact() {
        // Arrange — coarse already exact: measured gravity maps to +Z under it.
        let coarse = UnitQuaternion::from_euler_angles(0.2, -0.3, 1.0);
        let mean_accel_sensor = coarse.inverse() * (9.81 * Vector3::z());

        // Act
        let refined = refine_mount_tilt(coarse, mean_accel_sensor, Vector3::z());

        // Assert — no tilt correction needed; gravity still maps to +Z.
        let up = (refined * mean_accel_sensor).normalize();
        assert_relative_eq!(up, Vector3::z(), epsilon = 1e-9);
    }

    #[test]
    fn refine_mount_from_window_recovers_a_tilted_true_mount() {
        // Arrange — true mount = a coarse axis-swap plus a 6° pitch error; over a
        // stationary window the sensor reads true⁻¹·g. Refining the coarse mount
        // against that window must recover the true mount's up (measured g → +Z).
        let coarse = UnitQuaternion::from_euler_angles(0.0, 0.0, 90_f64.to_radians());
        let tilt_err = UnitQuaternion::from_euler_angles(0.0, 6_f64.to_radians(), 0.0);
        let true_mount = tilt_err * coarse;
        let g = 9.81 * Vector3::z();
        let sensor = true_mount.inverse() * g;
        let window = vec![sensor; 30];

        // Act
        let refined = refine_mount_from_window(coarse, &window, 0, 30, Vector3::z());

        // Assert — refined maps the measured gravity onto chassis +Z.
        let up = (refined * sensor).normalize();
        assert_relative_eq!(up, Vector3::z(), epsilon = 1e-9);
    }

    #[test]
    fn refine_mount_from_window_is_coarse_on_empty_window() {
        // Arrange — no samples to fit against.
        let coarse = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);

        // Act + Assert — the coarse pick is returned unchanged.
        let refined = refine_mount_from_window(coarse, &[], 0, 0, Vector3::z());
        assert_relative_eq!(refined.angle_to(&coarse), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn refine_mount_tilt_parked_lean_with_imu0_mean_up_reference_aligns_sensor_gravity_to_the_reference_not_vertical() {
        // Arrange — true mount = a 90° yaw coarse pick (zero tilt error against that
        // coarse).  The bike is parked pitched 8° (kickstand lean), so the chassis-
        // frame gravity direction (= up_ref, what IMU0 measured during the same
        // window) is NOT +Z:
        let lean = UnitQuaternion::from_euler_angles(0.0, 8_f64.to_radians(), 0.0);
        let g = 9.81 * Vector3::z();
        let up_ref = lean * g; // chassis-frame gravity as IMU0 saw it

        // The true mount for IMU1 has zero tilt error relative to the coarse pick; the
        // sensor reads whatever the true mount maps up_ref onto:
        let true_mount = UnitQuaternion::from_euler_angles(0.0, 0.0, 90_f64.to_radians());
        let sensor = true_mount.inverse() * up_ref;

        // Act — refine with the correct up_ref (IMU0 window-mean).
        let refined = refine_mount_tilt(true_mount, sensor, up_ref);

        // Assert — the refined mount maps sensor gravity onto up_ref (not +Z), and
        // it is indistinguishable from the true mount (no overcorrection).
        let aligned = (refined * sensor).normalize();
        assert_relative_eq!(aligned, up_ref.normalize(), epsilon = 1e-9);
        assert_relative_eq!(refined.angle_to(&true_mount), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn refine_mount_tilt_parked_lean_against_vertical_produces_mount_error_equal_to_lean_angle() {
        // Arrange — same parked-8°-lean scenario as the previous test; this time we
        // refine against +Z instead of the correct up_ref, documenting the failure
        // mode the fix removes.
        let lean = UnitQuaternion::from_euler_angles(0.0, 8_f64.to_radians(), 0.0);
        let g = 9.81 * Vector3::z();
        let up_ref = lean * g; // chassis-frame gravity (bike is leaning)

        let true_mount = UnitQuaternion::from_euler_angles(0.0, 0.0, 90_f64.to_radians());
        let sensor = true_mount.inverse() * up_ref;

        // Act — refine against +Z (the old behaviour / placeholder).
        let refined = refine_mount_tilt(true_mount, sensor, Vector3::z());

        // Assert — the mount error equals the lean angle (8°) to within 0.2°,
        // confirming that the parked lean was absorbed as a permanent tilt error.
        let error_rad = refined.angle_to(&true_mount);
        assert!((error_rad - 8_f64.to_radians()).abs() < 0.2_f64.to_radians(),
            "expected ~8° mount error, got {:.4}°", error_rad.to_degrees());
    }

    #[test]
    fn gyro_bias_is_the_stationary_mean_rate_per_imu() {
        // Arrange — a constant offset on each IMU's gyro while truly stationary.
        let accel = vec![Vector3::new(0.0, 0.0, 9.81); 10];
        let g0 = vec![Vector3::new(0.01, -0.02, 0.0); 10];
        let g1 = vec![Vector3::new(0.0, 0.0, 0.005); 10];
        let g2 = vec![Vector3::new(-0.003, 0.0, 0.0); 10];

        // Act
        let fit = fit_from_window(&accel, &g0, &g1, &g2, 0, 10);

        // Assert
        assert_relative_eq!(fit.b_g0, Vector3::new(0.01, -0.02, 0.0), epsilon = 1e-12);
        assert_relative_eq!(fit.b_g1, Vector3::new(0.0, 0.0, 0.005), epsilon = 1e-12);
        assert_relative_eq!(fit.b_g2, Vector3::new(-0.003, 0.0, 0.0), epsilon = 1e-12);
    }
}
