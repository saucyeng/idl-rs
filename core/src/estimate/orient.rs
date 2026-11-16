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
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

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

/// Why [`fit_imu0_mount_from_riding`] did or did not replace the reference mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountFitReason {
    /// The riding mount passed every gate and is in use.
    Fitted,
    /// Under [`RIDING_MIN_STRAIGHT_S`] of straight-line riding above
    /// [`RIDING_MIN_SPEED_MPS`] (or no GPS speed at all).
    TooLittleStraightRiding,
    /// The forward axis did not correlate with the GPS longitudinal acceleration
    /// (|correlation| under [`RIDING_MIN_CORR`]), so its sign is not trusted.
    WeakForwardCorrelation,
    /// The riding "up" is [`RIDING_MAX_UP_ANGLE_DEG`] or more from the reference
    /// mount's +Z: the coarse pick is for a different installation.
    UpFarFromReference,
}

/// IMU0's sensor→chassis mount fitted from riding data, with the evidence for it.
#[derive(Debug, Clone, Copy)]
pub struct MountFit {
    /// The mount to use: the fitted one when `used`, else the coarse one passed in.
    pub mount: UnitQuaternion<f64>,
    /// Straight-line riding the fit had to work with, s.
    pub straight_s: f64,
    /// Correlation between the forward-axis acceleration and the GPS longitudinal
    /// acceleration over the straight-line samples (−1..1; its sign set forward).
    pub corr: f64,
    /// Whether `mount` is the fitted mount.
    pub used: bool,
    /// Why, either way.
    pub reason: MountFitReason,
}

/// Riding mask: GPS speed above this, m/s (15 km/h).
pub const RIDING_MIN_SPEED_MPS: f64 = 15.0 / 3.6;
/// Straight-line mask: lateral acceleration `|v · (ω·u)|` below this fraction of g.
pub const RIDING_STRAIGHT_LAT_G: f64 = 0.1;
/// Minimum straight-line riding for the fit to be used, s.
pub const RIDING_MIN_STRAIGHT_S: f64 = 30.0;
/// Minimum |correlation| for the forward sign to be trusted.
pub const RIDING_MIN_CORR: f64 = 0.1;
/// The riding "up" must be within this of the coarse mount's +Z, degrees.
pub const RIDING_MAX_UP_ANGLE_DEG: f64 = 80.0;

/// Fits IMU0's sensor→chassis mount from riding data
/// (`2026-11-08-idl1-imu0-riding-mount-DRAFT.md` §2, M-1 … M-5), so the chassis
/// frame is the bike's and not the sensor's: +Z is the mean specific force while
/// riding straight, +X is the coarse mount's forward made perpendicular to it,
/// signed by agreement with the GPS longitudinal acceleration.
///
/// `accel` (m/s²) and `gyro` (rad/s) are IMU0 in its **sensor frame** on a
/// uniform grid of period `dt` (s); `speed` is the GPS speed on that grid, m/s,
/// NaN where there is none; `path_pitch` is the slope of the path there, rad,
/// positive climbing, NaN where unknown; `coarse` is the reference mount. Mount
/// yaw about "up" is not fitted (it stays the coarse mount's).
///
/// **Slope.** The mean specific force is gravity, and gravity is only the
/// chassis +Z on level ground. A trail ride is fast mostly downhill (measured on
/// one: the riding "up" sat 9° behind the true chassis +Z), so the fitted axes
/// are pitched back by the mean `path_pitch` over the straight-line samples.
/// With no known `path_pitch` the correction is zero (a closed MX lap needs none).
///
/// Falls back to `coarse` (`used: false`) with the reason when a gate fails.
pub fn fit_imu0_mount_from_riding(
    accel: &[Vector3<f64>],
    gyro: &[Vector3<f64>],
    speed: &[f64],
    path_pitch: &[f64],
    dt: f64,
    coarse: UnitQuaternion<f64>,
) -> MountFit {
    let fallback = |straight_s: f64, corr: f64, reason: MountFitReason| MountFit {
        mount: coarse,
        straight_s,
        corr,
        used: false,
        reason,
    };
    let n = accel.len().min(gyro.len()).min(speed.len());
    let unit_mean = |mask: &dyn Fn(usize) -> bool| -> Option<(Vector3<f64>, usize)> {
        let (mut sum, mut count) = (Vector3::zeros(), 0usize);
        for i in (0..n).filter(|&i| mask(i)) {
            sum += accel[i];
            count += 1;
        }
        sum.try_normalize(1e-9).map(|u| (u, count))
    };
    // (M-1) riding, (M-2) up: all riding first, then straight-line riding only so
    // a track with more left than right turns does not tilt it.
    let riding = |i: usize| speed[i] > RIDING_MIN_SPEED_MPS;
    let Some((u_all, _)) = unit_mean(&riding) else {
        return fallback(0.0, 0.0, MountFitReason::TooLittleStraightRiding);
    };
    const G: f64 = 9.80665;
    let straight: Vec<bool> = (0..n)
        .map(|i| riding(i) && (speed[i] * gyro[i].dot(&u_all)).abs() / G < RIDING_STRAIGHT_LAT_G)
        .collect();
    let Some((u, count)) = unit_mean(&|i| straight[i]) else {
        return fallback(0.0, 0.0, MountFitReason::TooLittleStraightRiding);
    };
    let straight_s = count as f64 * dt;
    if straight_s < RIDING_MIN_STRAIGHT_S {
        return fallback(straight_s, 0.0, MountFitReason::TooLittleStraightRiding);
    }
    let coarse_up = coarse.inverse() * Vector3::z();
    if u.angle(&coarse_up).to_degrees() >= RIDING_MAX_UP_ANGLE_DEG {
        return fallback(straight_s, 0.0, MountFitReason::UpFarFromReference);
    }
    // (M-3) forward: the coarse forward with its `u` part removed.
    let f0 = coarse.inverse() * Vector3::x();
    let Some(c) = (f0 - u * f0.dot(&u)).try_normalize(1e-6) else {
        return fallback(straight_s, 0.0, MountFitReason::UpFarFromReference);
    };
    let rate = 1.0 / dt;
    let lp = |v: Vec<f64>, cutoff: f64| crate::filters::lowpass(&v, 2, cutoff, rate).unwrap_or(v);
    let a_fwd = lp(accel[..n].iter().map(|a| a.dot(&c)).collect(), 1.5);
    let v_lp = lp(speed[..n].iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect(), 0.4);
    let a_gps = crate::statistics::gradient(&v_lp, rate);
    let idx: Vec<usize> = (0..n).filter(|&i| straight[i]).collect();
    let m = idx.len() as f64;
    let (ma, mg) = (idx.iter().map(|&i| a_fwd[i]).sum::<f64>() / m, idx.iter().map(|&i| a_gps[i]).sum::<f64>() / m);
    let (mut sag, mut saa, mut sgg) = (0.0, 0.0, 0.0);
    for &i in &idx {
        let (da, dg) = (a_fwd[i] - ma, a_gps[i] - mg);
        sag += da * dg;
        saa += da * da;
        sgg += dg * dg;
    }
    let corr = if saa > 0.0 && sgg > 0.0 { sag / (saa * sgg).sqrt() } else { 0.0 };
    if corr.abs() < RIDING_MIN_CORR {
        return fallback(straight_s, corr, MountFitReason::WeakForwardCorrelation);
    }
    // Gravity seen from a chassis pitched up by p is (sin p, 0, cos p), so with the
    // mean path pitch p̄ the level-ground axes are the measured pair turned by −p̄.
    let level = c * corr.signum();
    let known: Vec<f64> = idx.iter().filter_map(|&i| path_pitch.get(i).copied().filter(|p| p.is_finite())).collect();
    let mean_pitch = if known.is_empty() { 0.0 } else { known.iter().sum::<f64>() / known.len() as f64 };
    let (sp, cp) = mean_pitch.sin_cos();
    let (f, u) = (level * cp + u * sp, u * cp - level * sp);
    // (M-4) rows (forward, left, up) take a sensor vector to the chassis frame.
    let left = u.cross(&f);
    let rows = Matrix3::from_rows(&[f.transpose(), left.transpose(), u.transpose()]);
    let mount = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rows));
    MountFit { mount, straight_s, corr, used: true, reason: MountFitReason::Fitted }
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

    /// IMU0 as a sensor mounted by `true_mount` would record `seconds` of straight
    /// riding at 100 Hz on a path pitched `pitch` rad (positive climbing): speed
    /// 8 ± 2 m/s at 0.08 Hz, so there is longitudinal acceleration to sign forward
    /// with. Returns (accel, gyro, speed, dt).
    fn straight_ride(
        true_mount: UnitQuaternion<f64>,
        seconds: f64,
        pitch: f64,
    ) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>, Vec<f64>, f64) {
        let dt = 0.01;
        let n = (seconds / dt) as usize;
        let (mut accel, mut speed) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let t = i as f64 * dt;
            speed.push(8.0 + 2.0 * (0.5 * t).sin());
            let a_long = 2.0 * 0.5 * (0.5 * t).cos();
            // Gravity seen from a chassis pitched up by p is g·(sin p, 0, cos p).
            let chassis = Vector3::new(a_long + 9.81 * pitch.sin(), 0.0, 9.81 * pitch.cos());
            accel.push(true_mount.inverse() * chassis);
        }
        (accel, vec![Vector3::zeros(); n], speed, dt)
    }

    /// The reference bike's IMU0 pick: X rearward, Y right, Z up.
    fn coarse_mount() -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI)
    }

    #[test]
    fn riding_mount_synthetic_imu_tilted_57_degrees_about_y_recovers_the_tilt_and_forward_sign() {
        // Arrange — the sensor is the coarse pick tilted 57° about the chassis Y.
        let true_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 57_f64.to_radians()) * coarse_mount();
        let (accel, gyro, speed, dt) = straight_ride(true_mount, 60.0, 0.0);
        let level = vec![0.0; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &speed, &level, dt, coarse_mount());

        // Assert
        assert!(fit.used, "fit rejected: {:?}", fit.reason);
        assert_eq!(fit.reason, MountFitReason::Fitted);
        assert!(fit.corr > 0.5, "forward correlation {}", fit.corr);
        assert!(fit.mount.angle_to(&true_mount).to_degrees() < 0.5, "off by {}°", fit.mount.angle_to(&true_mount).to_degrees());
    }

    #[test]
    fn riding_mount_coarse_pick_facing_backwards_the_forward_sign_flips() {
        // Arrange — the coarse pick has forward and rearward swapped.
        let true_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 20_f64.to_radians()) * coarse_mount();
        let backwards = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI) * coarse_mount();
        let (accel, gyro, speed, dt) = straight_ride(true_mount, 60.0, 0.0);
        let level = vec![0.0; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &speed, &level, dt, backwards);

        // Assert — the correlation is negative and the fitted forward is the true one.
        assert!(fit.used);
        assert!(fit.corr < -0.5, "forward correlation {}", fit.corr);
        assert!(fit.mount.angle_to(&true_mount).to_degrees() < 0.5);
    }

    #[test]
    fn riding_mount_under_30_s_of_straight_riding_falls_back_to_the_reference_mount() {
        // Arrange
        let true_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 57_f64.to_radians()) * coarse_mount();
        let (accel, gyro, speed, dt) = straight_ride(true_mount, 20.0, 0.0);
        let level = vec![0.0; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &speed, &level, dt, coarse_mount());

        // Assert
        assert!(!fit.used);
        assert_eq!(fit.reason, MountFitReason::TooLittleStraightRiding);
        assert_relative_eq!(fit.mount.angle_to(&coarse_mount()), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn riding_mount_no_gps_speed_falls_back_to_the_reference_mount() {
        // Arrange — a ride with no GPS: speed unknown everywhere.
        let (accel, gyro, _, dt) = straight_ride(coarse_mount(), 60.0, 0.0);
        let unknown = vec![f64::NAN; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &unknown, &unknown, dt, coarse_mount());

        // Assert
        assert!(!fit.used);
        assert_eq!(fit.reason, MountFitReason::TooLittleStraightRiding);
    }

    #[test]
    fn riding_mount_all_fast_riding_downhill_known_slope_is_taken_out_of_up() {
        // Arrange — every fast sample is on an 8° descent, as on a trail ride.
        let true_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 12_f64.to_radians()) * coarse_mount();
        let slope = -8_f64.to_radians();
        let (accel, gyro, speed, dt) = straight_ride(true_mount, 60.0, slope);
        let path_pitch = vec![slope; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &speed, &path_pitch, dt, coarse_mount());

        // Assert
        assert!(fit.used);
        assert!(fit.mount.angle_to(&true_mount).to_degrees() < 0.5, "off by {}°", fit.mount.angle_to(&true_mount).to_degrees());
    }

    #[test]
    fn riding_mount_all_fast_riding_downhill_unknown_slope_tilts_up_by_the_slope() {
        // Arrange — the same descent with no altitude to say so.
        let true_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 12_f64.to_radians()) * coarse_mount();
        let (accel, gyro, speed, dt) = straight_ride(true_mount, 60.0, -8_f64.to_radians());
        let unknown = vec![f64::NAN; accel.len()];

        // Act
        let fit = fit_imu0_mount_from_riding(&accel, &gyro, &speed, &unknown, dt, coarse_mount());

        // Assert — the limit of the method without a slope: the frame is 8° out.
        assert!(fit.used);
        assert_relative_eq!(fit.mount.angle_to(&true_mount).to_degrees(), 8.0, epsilon = 0.5);
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
