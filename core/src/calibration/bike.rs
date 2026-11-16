//! A bike's calibration: where its IMUs sit and which way the bike points.
//!
//! This is what lives in the bike's file and what the estimator reads in place
//! of [`BikeGeometry::reference_bike`]'s authored mounts. It is fitted from the
//! calibration recordings, with every window named by the caller — nothing
//! here averages over a ride, where the bike spends time on a kickstand or
//! lying on its side:
//!
//! - **up** per IMU, from a window with the bike still and level;
//! - each unsprung IMU's **rotation and lever arm** relative to IMU0, from a
//!   window where the bike was turned about by hand as one rigid body
//!   ([`super::rigid::calibrate`], the fork pod treated as rigid with the frame
//!   over a window where the bars did not turn);
//! - the **steering axis**, from a window with the frame still and the bars
//!   turning ([`steer_axis`]);
//! - **forward**, from the steering axis (it lies in the bike's centre plane
//!   and leans back) or, without one, from the line to the fork pod.
//!
//! Everything is stored in IMU0's own sensor axes, so the record does not
//! depend on any choice of bike frame; [`BikeCalibration::apply_to`] turns it
//! into the estimator's chassis frame (X forward, Y left, Z up).

use nalgebra::{Matrix3, Quaternion, Rotation3, SymmetricEigen, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use super::rigid::{CalibrationInput, GRAVITY_M_S2};
use crate::estimate::geometry::{BikeGeometry, ImuPose};

/// Version of this fit. Bump when the same recordings would give a different
/// calibration.
pub const BIKE_CALIBRATION_VERSION: u32 = 1;

/// Largest gyro scatter (RMS about its mean, rad/s) a "still" window may show.
/// The Beacon recordings of 2026-11-04 sit at 0.002 rad/s; a hand on the bars
/// is ten times that.
pub const STILL_MAX_GYRO_RMS_RAD_S: f64 = 0.01;
/// Largest departure of the still window's mean specific force from 1 g
/// (fraction). Beyond the sensors' offset spread the bike was moving.
pub const STILL_MAX_G_ERROR: f64 = 0.06;
/// Smallest RMS bar rate (rad/s) over the steer window for an axis to be fitted.
pub const STEER_MIN_RATE_RMS_RAD_S: f64 = 0.1;
/// Smallest ratio of the bar rotation's energy along its main axis to that
/// along the next one. Below it the frame was moving too, or the pod's
/// rotation into IMU0's axes is wrong.
pub const STEER_MIN_AXIS_DOMINANCE: f64 = 25.0;
/// Tumble-rate gate handed to [`super::rigid::calibrate`], rad/s. A bike turned
/// about by hand manages about 1 rad/s (0.96–1.18 on the Beacon recording), not
/// the 2 rad/s the spec's gate asks of a bare logger; lever richness is what
/// decides whether the lever arms are usable.
pub const HAND_TUMBLE_MIN_RATE_RAD_S: f64 = 0.5;

/// A stretch of one recording, seconds from the first sample every IMU has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceWindow {
    /// The session the window is in.
    pub session_id: String,
    /// Start, s.
    pub from_s: f64,
    /// End, s. Absent = the end of the recording.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_s: Option<f64>,
}

/// How [`CalibrationFit::forward`] was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardFrom {
    /// Opposite the steering axis's lean (the axis leans back, in the centre plane).
    SteerAxis,
    /// The level direction from IMU0 to the fork pod; off by the pod's sideways
    /// offset over its distance.
    ForkPod,
    /// One of the above, then turned so that straight-line riding in one
    /// session reads zero slip ([`CalibrationFit::with_forward_turned`]).
    Riding,
}

/// One unsprung IMU relative to IMU0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImuFit {
    /// `imu_index` as recorded (1 = fork pod, 2 = swingarm pod).
    pub imu_index: u8,
    /// Rotation from this sensor's axes into IMU0's, `(w, x, y, z)`, with the
    /// bars straight and the suspension as it sat in the recording.
    pub mount: [f64; 4],
    /// From IMU0 to this sensor, in IMU0's axes, metres.
    pub lever_m: [f64; 3],
    /// RMS of what the rigid fit left of this sensor's angular rate, rad/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_residual_rad_s: Option<f64>,
    /// Angle between this sensor's own "up" in the still window, turned into
    /// IMU0's axes, and IMU0's, degrees. Accelerometer offsets account for
    /// about a degree; more means the bars were turned or the suspension sat
    /// differently in the still window than in the rigid one.
    pub up_disagreement_deg: f64,
}

/// The fitted numbers, all in IMU0's sensor axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationFit {
    /// [`BIKE_CALIBRATION_VERSION`] at the time of the fit.
    pub version: u32,
    /// Unit vector pointing up with the bike level.
    pub up: [f64; 3],
    /// Unit vector pointing forward, level. Absent when only IMU0 was recorded:
    /// one still sensor cannot tell which way the bike points.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<[f64; 3]>,
    /// How `forward` was found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_from: Option<ForwardFrom>,
    /// How far `forward` has been turned from the recordings' own answer to
    /// match the direction the bike travels, degrees anticlockwise seen from
    /// above (positive = forward moved to the left). Absent = not turned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_trim_deg: Option<f64>,
    /// Steering axis, unit vector pointing up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer_axis: Option<[f64; 3]>,
    /// IMU0's gyro offset in the still window, rad/s.
    pub imu0_gyro_bias_rad_s: [f64; 3],
    /// The unsprung IMUs that were fitted, ascending by index.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imus: Vec<ImuFit>,
}

/// One calibration as the bike's file holds it: the fit, where it came from,
/// and what a merge of two devices' copies needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BikeCalibration {
    /// Stable id.
    pub calibration_id: String,
    /// When the recordings were made, UTC ms. A session uses the newest
    /// calibration captured at or before its own start.
    pub captured_at_ms: i64,
    /// When this record last changed, UTC ms; the newer copy wins a merge.
    pub updated_at_ms: i64,
    /// Removed. Kept as a tombstone so a merge does not bring it back.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deleted: bool,
    /// Free text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// The still, level window `up` came from.
    pub still: SourceWindow,
    /// The window the swingarm pod was fitted over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid: Option<SourceWindow>,
    /// The window (bars not turning) the fork pod was fitted over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_rigid: Option<SourceWindow>,
    /// The window (frame still, bars turning) the steering axis came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer: Option<SourceWindow>,
    /// The ride `forward` was turned to match ([`ForwardFrom::Riding`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_ride: Option<String>,
    /// The fitted numbers.
    #[serde(flatten)]
    pub fit: CalibrationFit,
}

/// Why a calibration could not be fitted or used.
#[derive(Debug, Clone, PartialEq)]
pub enum BikeCalibrationError {
    /// The window holds no IMU0 samples.
    NoImu0,
    /// The "still" window was not still.
    NotStill {
        /// The sensor that moved.
        imu_index: u8,
        /// Its gyro scatter, rad/s.
        gyro_rms_rad_s: f64,
    },
    /// The still window's mean specific force is not 1 g.
    NotOneG {
        /// The sensor.
        imu_index: u8,
        /// What it read, in g.
        measured_g: f64,
    },
    /// The rigid fit of this IMU gave no rotation or no lever arm.
    RigidFitFailed {
        /// The sensor.
        imu_index: u8,
        /// What the fit reported.
        reason: String,
    },
    /// A stored vector is not usable (zero length, or forward along up).
    Degenerate(&'static str),
}

impl std::fmt::Display for BikeCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BikeCalibrationError::NoImu0 => write!(f, "the window holds no IMU0 samples"),
            BikeCalibrationError::NotStill { imu_index, gyro_rms_rad_s } => write!(
                f,
                "IMU{imu_index} was moving in the still window (gyro scatter {gyro_rms_rad_s:.4} rad/s, {STILL_MAX_GYRO_RMS_RAD_S} or less needed)"
            ),
            BikeCalibrationError::NotOneG { imu_index, measured_g } => write!(
                f,
                "IMU{imu_index} read {measured_g:.3} g in the still window; a still sensor reads 1 g within {STILL_MAX_G_ERROR}"
            ),
            BikeCalibrationError::RigidFitFailed { imu_index, reason } => {
                write!(f, "IMU{imu_index} could not be fitted over the rigid window: {reason}")
            }
            BikeCalibrationError::Degenerate(what) => write!(f, "the calibration's {what} is not usable"),
        }
    }
}

impl std::error::Error for BikeCalibrationError {}

/// One sensor over the still window.
#[derive(Debug, Clone, PartialEq)]
pub struct StillSensor {
    /// `imu_index` as recorded.
    pub imu_index: u8,
    /// Unit vector along the mean specific force (up), in the sensor's own axes.
    pub up: Vector3<f64>,
    /// Mean angular rate (the gyro offset), rad/s, sensor axes.
    pub gyro_bias: Vector3<f64>,
}

/// Each sensor's "up" and gyro offset over a still window.
///
/// # Errors
///
/// [`BikeCalibrationError::NotStill`] or [`BikeCalibrationError::NotOneG`] when
/// a sensor was moving, [`BikeCalibrationError::NoImu0`] with no samples.
pub fn still_sensors(still: &CalibrationInput) -> Result<Vec<StillSensor>, BikeCalibrationError> {
    let mut out = Vec::with_capacity(still.sensors.len());
    for s in &still.sensors {
        let n = s.gyro_rad_s.len().min(s.accel_m_s2.len());
        if n == 0 {
            return Err(BikeCalibrationError::NoImu0);
        }
        let gyro_bias = s.gyro_rad_s[..n].iter().sum::<Vector3<f64>>() / n as f64;
        let accel = s.accel_m_s2[..n].iter().sum::<Vector3<f64>>() / n as f64;
        let gyro_rms =
            (s.gyro_rad_s[..n].iter().map(|g| (g - gyro_bias).norm_squared()).sum::<f64>() / n as f64).sqrt();
        if gyro_rms > STILL_MAX_GYRO_RMS_RAD_S {
            return Err(BikeCalibrationError::NotStill { imu_index: s.imu_index, gyro_rms_rad_s: gyro_rms });
        }
        let measured_g = accel.norm() / GRAVITY_M_S2;
        if (measured_g - 1.0).abs() > STILL_MAX_G_ERROR {
            return Err(BikeCalibrationError::NotOneG { imu_index: s.imu_index, measured_g });
        }
        out.push(StillSensor { imu_index: s.imu_index, up: accel.normalize(), gyro_bias });
    }
    if !out.iter().any(|s| s.imu_index == 0) {
        return Err(BikeCalibrationError::NoImu0);
    }
    out.sort_by_key(|s| s.imu_index);
    Ok(out)
}

/// The steering axis in IMU0's axes, pointing along `up`: the one direction the
/// fork pod turns about relative to the frame while the bars are swung.
///
/// `frame_rate` and `fork_rate` are the two gyros over the steer window with
/// their offsets removed, each in its own sensor axes and sample for sample;
/// `fork_to_frame` turns the fork pod's axes into IMU0's. Returns None when the
/// bars barely moved ([`STEER_MIN_RATE_RMS_RAD_S`]) or the relative rotation is
/// not about one axis ([`STEER_MIN_AXIS_DOMINANCE`]).
pub fn steer_axis(
    frame_rate: &[Vector3<f64>],
    fork_rate: &[Vector3<f64>],
    fork_to_frame: &UnitQuaternion<f64>,
    up: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let n = frame_rate.len().min(fork_rate.len());
    if n < 3 {
        return None;
    }
    let mut energy = Matrix3::zeros();
    for k in 0..n {
        let relative = fork_to_frame * fork_rate[k] - frame_rate[k];
        energy += relative * relative.transpose();
    }
    energy /= n as f64;
    let eig = SymmetricEigen::new(energy);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let (main, next) = (eig.eigenvalues[order[0]], eig.eigenvalues[order[1]]);
    if main.sqrt() < STEER_MIN_RATE_RMS_RAD_S || main < STEER_MIN_AXIS_DOMINANCE * next.max(0.0) {
        return None;
    }
    let axis: Vector3<f64> = eig.eigenvectors.column(order[0]).into_owned();
    Some(if axis.dot(up) < 0.0 { -axis } else { axis })
}

/// The level forward direction in IMU0's axes: opposite the steering axis's
/// lean when there is one, else toward the fork pod. None when neither gives a
/// level direction (no fork pod; or an axis that is vertical).
pub fn forward_direction(
    up: &Vector3<f64>,
    steer_axis: Option<&Vector3<f64>>,
    fork_lever: Option<&Vector3<f64>>,
) -> Option<(Vector3<f64>, ForwardFrom)> {
    let level = |v: &Vector3<f64>| v - up * v.dot(up);
    if let Some(axis) = steer_axis {
        let back = level(axis);
        // A head angle of 85° or steeper leaves too little lean to read.
        if back.norm() > 5_f64.to_radians().sin() {
            let mut forward = -back.normalize();
            // The fork pod is ahead of IMU0; an axis fitted upside down is not.
            if let Some(lever) = fork_lever {
                if level(lever).dot(&forward) < 0.0 {
                    forward = -forward;
                }
            }
            return Some((forward, ForwardFrom::SteerAxis));
        }
    }
    let toward_fork = level(fork_lever?);
    (toward_fork.norm() > 0.05).then(|| (toward_fork.normalize(), ForwardFrom::ForkPod))
}

fn unit(v: &[f64; 3], what: &'static str) -> Result<Vector3<f64>, BikeCalibrationError> {
    let v = Vector3::new(v[0], v[1], v[2]);
    if v.norm() < 1e-6 || !v.iter().all(|x| x.is_finite()) {
        return Err(BikeCalibrationError::Degenerate(what));
    }
    Ok(v.normalize())
}

fn quat(q: &[f64; 4]) -> UnitQuaternion<f64> {
    UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]))
}

impl CalibrationFit {
    /// Rotation from IMU0's sensor axes into the chassis frame (X forward, Y
    /// left, Z up). With no `forward`, the level direction nearest
    /// `fallback_forward` (a chassis-frame-to-be direction in IMU0's axes, e.g.
    /// the reference mount's) is used; the tilt is the calibration's either way.
    ///
    /// # Errors
    ///
    /// [`BikeCalibrationError::Degenerate`] when `up` is unusable or forward
    /// lies along it.
    pub fn imu0_mount(&self, fallback_forward: &Vector3<f64>) -> Result<UnitQuaternion<f64>, BikeCalibrationError> {
        let z = unit(&self.up, "up")?;
        let wanted = match &self.forward {
            Some(f) => unit(f, "forward")?,
            None => *fallback_forward,
        };
        let level = wanted - z * wanted.dot(&z);
        if level.norm() < 1e-3 {
            return Err(BikeCalibrationError::Degenerate("forward"));
        }
        let x = level.normalize();
        let y = z.cross(&x);
        let rows = Matrix3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
        Ok(UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rows)))
    }

    /// `base` with this calibration's IMU mounts, lever arms and steering axis
    /// in place of its own. What a calibration does not measure (travel, sag,
    /// the rear axle path) stays `base`'s; an IMU the calibration has no fit
    /// for keeps `base`'s pose, turned with IMU0's so the three stay
    /// consistent with each other.
    ///
    /// # Errors
    ///
    /// As [`Self::imu0_mount`].
    pub fn apply_to(&self, base: &BikeGeometry) -> Result<BikeGeometry, BikeCalibrationError> {
        let base_forward = base.imu0.mount.inverse() * Vector3::x();
        let m0 = self.imu0_mount(&base_forward)?;
        let shift = m0 * base.imu0.mount.inverse();
        let pose = |index: u8, fallback: &ImuPose| match self.imus.iter().find(|i| i.imu_index == index) {
            Some(fit) => ImuPose {
                mount: m0 * quat(&fit.mount),
                lever: m0 * Vector3::new(fit.lever_m[0], fit.lever_m[1], fit.lever_m[2]),
            },
            None => ImuPose { mount: shift * fallback.mount, lever: fallback.lever },
        };
        let mut out = base.clone();
        out.imu0 = ImuPose { mount: m0, lever: base.imu0.lever };
        out.imu1 = pose(1, &base.imu1);
        out.imu2 = base.imu2.as_ref().map(|p| pose(2, p));
        if let Some(axis) = &self.steer_axis {
            let axis = m0 * unit(axis, "steer_axis")?;
            out.steer_axis = axis;
            // A telescoping fork slides along the axis it steers about.
            out.front_axis = axis;
        }
        Ok(out)
    }

    /// This calibration with `forward` turned about `up` by `trim_deg` more
    /// (anticlockwise seen from above), so the bike's forward is the direction
    /// it travels and not only what the recordings said. The turn is added to
    /// any already made.
    ///
    /// # Errors
    ///
    /// [`BikeCalibrationError::Degenerate`] when there is no usable `forward`
    /// to turn.
    pub fn with_forward_turned(&self, trim_deg: f64) -> Result<CalibrationFit, BikeCalibrationError> {
        let up = unit(&self.up, "up")?;
        let forward = unit(self.forward.as_ref().ok_or(BikeCalibrationError::Degenerate("forward"))?, "forward")?;
        let (sin, cos) = trim_deg.to_radians().sin_cos();
        let turned = forward * cos + up.cross(&forward) * sin;
        Ok(CalibrationFit {
            forward: Some([turned.x, turned.y, turned.z]),
            forward_from: Some(ForwardFrom::Riding),
            forward_trim_deg: Some(self.forward_trim_deg.unwrap_or(0.0) + trim_deg),
            ..self.clone()
        })
    }

    /// Whether the calibration fixes which way the bike points (it has a
    /// `forward`); without one only the tilt is the calibration's.
    pub fn has_forward(&self) -> bool {
        self.forward.is_some()
    }
}

/// The calibration a session starting at `session_start_ms` uses: the newest
/// one captured at or before it, else the oldest captured after it (a
/// calibration recorded at the end of the day still describes that day's
/// bike). Deleted ones are skipped. The flag is true for the second case.
pub fn calibration_for(calibrations: &[BikeCalibration], session_start_ms: i64) -> Option<(&BikeCalibration, bool)> {
    let live = || calibrations.iter().filter(|c| !c.deleted);
    live()
        .filter(|c| c.captured_at_ms <= session_start_ms)
        .max_by_key(|c| (c.captured_at_ms, c.updated_at_ms))
        .map(|c| (c, false))
        .or_else(|| live().min_by_key(|c| (c.captured_at_ms, -c.updated_at_ms)).map(|c| (c, true)))
}

/// Two devices' lists of one bike's calibrations as one: every calibration
/// either has, and for one both have, the copy changed last (a tie keeps
/// `ours`). Ordered by capture time, then id.
pub fn merge_calibrations(ours: &[BikeCalibration], theirs: &[BikeCalibration]) -> Vec<BikeCalibration> {
    let mut out: Vec<BikeCalibration> = ours.to_vec();
    for c in theirs {
        match out.iter_mut().find(|o| o.calibration_id == c.calibration_id) {
            Some(o) => {
                if c.updated_at_ms > o.updated_at_ms {
                    *o = c.clone();
                }
            }
            None => out.push(c.clone()),
        }
    }
    out.sort_by(|a, b| (a.captured_at_ms, &a.calibration_id).cmp(&(b.captured_at_ms, &b.calibration_id)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::rigid::{Body, SensorStream};
    use crate::estimate::noise::ImuNoise;
    use approx::assert_relative_eq;

    fn input(sensors: Vec<SensorStream>) -> CalibrationInput {
        CalibrationInput {
            sample_rate_hz: 400.0,
            noise: ImuNoise { gyro_arw: 1e-3, accel_vrw: 1e-2, gyro_bias_rw: 0.0, accel_bias_rw: 0.0 },
            sensors,
            captured_utc: None,
            min_median_rate_rad_s: None,
        }
    }

    fn still_stream(imu_index: u8, up: Vector3<f64>, bias: Vector3<f64>, wobble: f64) -> SensorStream {
        let n = 800;
        SensorStream {
            imu_index,
            body: Body::of_imu(imu_index),
            gyro_rad_s: (0..n).map(|k| bias + Vector3::x() * wobble * (k as f64 * 0.3).sin()).collect(),
            accel_m_s2: vec![up * GRAVITY_M_S2; n],
        }
    }

    fn fit(up: Vector3<f64>, forward: Option<Vector3<f64>>) -> CalibrationFit {
        CalibrationFit {
            version: BIKE_CALIBRATION_VERSION,
            up: [up.x, up.y, up.z],
            forward: forward.map(|f| [f.x, f.y, f.z]),
            forward_from: forward.map(|_| ForwardFrom::SteerAxis),
            forward_trim_deg: None,
            steer_axis: None,
            imu0_gyro_bias_rad_s: [0.0; 3],
            imus: Vec::new(),
        }
    }

    fn calibration(id: &str, captured_at_ms: i64, updated_at_ms: i64) -> BikeCalibration {
        BikeCalibration {
            calibration_id: id.into(),
            captured_at_ms,
            updated_at_ms,
            deleted: false,
            note: String::new(),
            still: SourceWindow { session_id: "s".into(), from_s: 1.0, to_s: Some(7.0) },
            rigid: None,
            fork_rigid: None,
            steer: None,
            forward_ride: None,
            fit: fit(Vector3::z(), Some(Vector3::x())),
        }
    }

    #[test]
    fn still_sensors_reads_up_and_the_gyro_offset_of_each_sensor() {
        // Arrange — IMU0 tilted, IMU1 with a gyro offset.
        let up0 = Vector3::new(0.2, 0.0, 0.98).normalize();
        let bias1 = Vector3::new(0.01, -0.02, 0.003);
        let still = input(vec![
            still_stream(1, Vector3::y(), bias1, 0.0),
            still_stream(0, up0, Vector3::zeros(), 0.0),
        ]);

        // Act
        let sensors = still_sensors(&still).unwrap();

        // Assert — ascending by index, whatever order they came in.
        assert_eq!(sensors[0].imu_index, 0);
        assert_relative_eq!(sensors[0].up, up0, epsilon = 1e-12);
        assert_relative_eq!(sensors[1].gyro_bias, bias1, epsilon = 1e-12);
    }

    #[test]
    fn still_sensors_refuses_a_window_where_a_sensor_was_moving() {
        // Arrange — the fork pod swings at 0.1 rad/s.
        let still = input(vec![
            still_stream(0, Vector3::z(), Vector3::zeros(), 0.0),
            still_stream(1, Vector3::z(), Vector3::zeros(), 0.1),
        ]);

        // Act
        let err = still_sensors(&still).unwrap_err();

        // Assert
        assert!(matches!(err, BikeCalibrationError::NotStill { imu_index: 1, .. }), "{err}");
    }

    #[test]
    fn still_sensors_refuses_a_window_that_does_not_read_one_g() {
        // Arrange — 1.2 g: the bike was being moved, however smoothly.
        let mut s = still_stream(0, Vector3::z(), Vector3::zeros(), 0.0);
        s.accel_m_s2.iter_mut().for_each(|a| *a *= 1.2);

        // Act
        let err = still_sensors(&input(vec![s])).unwrap_err();

        // Assert
        assert!(matches!(err, BikeCalibrationError::NotOneG { imu_index: 0, .. }), "{err}");
    }

    #[test]
    fn steer_axis_is_the_axis_the_fork_turns_about_in_frame_axes_pointing_up() {
        // Arrange — the fork pod is mounted turned 90° about its own X; it swings
        // about an axis leaning back 26° while the frame stays still.
        let true_axis = Vector3::new(-26_f64.to_radians().sin(), 0.0, 26_f64.to_radians().cos());
        let fork_to_frame = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 1.2);
        let n = 600;
        let rate = |k: usize| 1.5 * (k as f64 * 0.02).sin();
        let fork: Vec<Vector3<f64>> = (0..n).map(|k| fork_to_frame.inverse() * (-true_axis * rate(k))).collect();
        let frame = vec![Vector3::zeros(); n];

        // Act
        let axis = steer_axis(&frame, &fork, &fork_to_frame, &Vector3::z()).unwrap();

        // Assert — the fork swung about −axis; the result still points up.
        assert_relative_eq!(axis, true_axis, epsilon = 1e-9);
    }

    #[test]
    fn steer_axis_is_absent_when_the_bars_did_not_move() {
        // Arrange
        let n = 600;
        let quiet = vec![Vector3::new(1e-3, 0.0, 0.0); n];

        // Act
        let axis = steer_axis(&quiet, &quiet, &UnitQuaternion::identity(), &Vector3::z());

        // Assert
        assert!(axis.is_none());
    }

    #[test]
    fn steer_axis_is_absent_when_the_relative_rotation_is_about_two_axes() {
        // Arrange — the fork turns about Z while the frame is rocked about X.
        let n = 600;
        let fork: Vec<Vector3<f64>> = (0..n).map(|k| Vector3::z() * (k as f64 * 0.02).sin()).collect();
        let frame: Vec<Vector3<f64>> = (0..n).map(|k| Vector3::x() * 0.6 * (k as f64 * 0.031).cos()).collect();

        // Act
        let axis = steer_axis(&frame, &fork, &UnitQuaternion::identity(), &Vector3::z());

        // Assert
        assert!(axis.is_none());
    }

    #[test]
    fn forward_direction_is_opposite_the_steering_axis_lean() {
        // Arrange — axis leaning back toward −X; fork pod ahead and off to one side.
        let axis = Vector3::new(-0.44, 0.0, 0.9).normalize();
        let lever = Vector3::new(0.8, 0.1, -0.2);

        // Act
        let (forward, from) = forward_direction(&Vector3::z(), Some(&axis), Some(&lever)).unwrap();

        // Assert — the pod's sideways offset does not enter.
        assert_eq!(from, ForwardFrom::SteerAxis);
        assert_relative_eq!(forward, Vector3::x(), epsilon = 1e-12);
    }

    #[test]
    fn forward_direction_without_a_steering_axis_points_at_the_fork_pod() {
        // Arrange
        let lever = Vector3::new(0.8, 0.0, -0.3);

        // Act
        let (forward, from) = forward_direction(&Vector3::z(), None, Some(&lever)).unwrap();

        // Assert
        assert_eq!(from, ForwardFrom::ForkPod);
        assert_relative_eq!(forward, Vector3::x(), epsilon = 1e-12);
    }

    #[test]
    fn forward_direction_is_absent_with_one_sensor() {
        assert!(forward_direction(&Vector3::z(), None, None).is_none());
    }

    #[test]
    fn imu0_mount_puts_up_on_chassis_z_and_forward_on_chassis_x() {
        // Arrange — a sensor with its X pointing back and tilted 12°.
        let up = Vector3::new(0.2, 0.0, 0.98).normalize();
        let forward = Vector3::new(-0.98, 0.0, 0.2).normalize();

        // Act
        let m0 = fit(up, Some(forward)).imu0_mount(&Vector3::x()).unwrap();

        // Assert
        assert_relative_eq!(m0 * up, Vector3::z(), epsilon = 1e-12);
        assert_relative_eq!(m0 * forward, Vector3::x(), epsilon = 1e-12);
    }

    #[test]
    fn imu0_mount_without_forward_keeps_the_calibrated_tilt_and_the_fallback_heading() {
        // Arrange — the fallback forward is not level in this sensor's axes.
        let up = Vector3::new(-0.83, 0.0, 0.56).normalize();
        let fallback = Vector3::new(0.5, 0.1, 0.86).normalize();

        // Act
        let m0 = fit(up, None).imu0_mount(&fallback).unwrap();

        // Assert — up exactly; forward is the fallback levelled.
        assert_relative_eq!(m0 * up, Vector3::z(), epsilon = 1e-12);
        let f = m0 * fallback;
        assert_relative_eq!(f.y, 0.0, epsilon = 1e-12);
        assert!(f.x > 0.0);
    }

    #[test]
    fn apply_to_expresses_mounts_levers_and_the_steering_axis_in_the_chassis_frame() {
        // Arrange — sensor X points back: forward is −X. One fork pod fitted.
        let pod_mount = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.4);
        let mut f = fit(Vector3::z(), Some(-Vector3::x()));
        f.steer_axis = Some([0.44, 0.0, 0.9]);
        f.imus = vec![ImuFit {
            imu_index: 1,
            mount: [pod_mount.w, pod_mount.i, pod_mount.j, pod_mount.k],
            lever_m: [-0.8, -0.05, -0.25],
            rate_residual_rad_s: None,
            up_disagreement_deg: 0.0,
        }];
        let base = BikeGeometry::reference_bike();

        // Act
        let g = f.apply_to(&base).unwrap();

        // Assert — 0.8 m behind in sensor axes is 0.8 m ahead on the bike; 5 cm
        // to sensor −Y is 5 cm to the bike's left.
        assert_relative_eq!(g.imu1.lever, Vector3::new(0.8, 0.05, -0.25), epsilon = 1e-12);
        assert_relative_eq!(g.imu1.mount.angle_to(&(g.imu0.mount * pod_mount)), 0.0, epsilon = 1e-12);
        assert!(g.steer_axis.x < 0.0 && g.steer_axis.z > 0.0, "leans back: {:?}", g.steer_axis);
        assert_relative_eq!(g.front_axis, g.steer_axis, epsilon = 1e-12);
        // Travel and sag are not a calibration's to change.
        assert_eq!(g.front_travel_max, base.front_travel_max);
    }

    #[test]
    fn with_forward_turned_moves_forward_to_the_left_about_up_and_adds_to_an_earlier_turn() {
        // Arrange — forward is chassis X in a sensor whose axes are the chassis's.
        let f = fit(Vector3::z(), Some(Vector3::x()));

        // Act
        let once = f.with_forward_turned(5.0).unwrap();
        let twice = once.with_forward_turned(-2.0).unwrap();

        // Assert — 5° anticlockwise seen from above is toward +Y (left).
        let forward = once.forward.unwrap();
        assert_relative_eq!(forward[0], 5_f64.to_radians().cos(), epsilon = 1e-12);
        assert_relative_eq!(forward[1], 5_f64.to_radians().sin(), epsilon = 1e-12);
        assert_eq!(once.forward_from, Some(ForwardFrom::Riding));
        assert_relative_eq!(twice.forward_trim_deg.unwrap(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn with_forward_turned_nothing_to_turn_is_refused() {
        assert!(fit(Vector3::z(), None).with_forward_turned(5.0).is_err());
    }

    #[test]
    fn apply_to_turns_an_unfitted_pod_with_imu0() {
        // Arrange — no pod fits; IMU0 is 10° off the base's mount.
        let base = BikeGeometry::reference_bike();
        let base_up = base.imu0.mount.inverse() * Vector3::z();
        let tilt = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 10_f64.to_radians());
        let up = tilt * base_up;

        // Act
        let g = fit(up, None).apply_to(&base).unwrap();

        // Assert — the pod still sits where it sat relative to IMU0.
        let before = base.imu0.mount.inverse() * base.imu1.mount;
        let after = g.imu0.mount.inverse() * g.imu1.mount;
        assert_relative_eq!(before.angle_to(&after), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn calibration_for_takes_the_newest_one_captured_before_the_session() {
        // Arrange
        let all = [calibration("a", 100, 1), calibration("b", 200, 1), calibration("c", 300, 1)];

        // Act
        let (chosen, after) = calibration_for(&all, 250).unwrap();

        // Assert
        assert_eq!(chosen.calibration_id, "b");
        assert!(!after);
    }

    #[test]
    fn calibration_for_falls_forward_to_the_oldest_later_one_and_says_so() {
        // Arrange — the ride came before any calibration was recorded.
        let all = [calibration("b", 200, 1), calibration("c", 300, 1)];

        // Act
        let (chosen, after) = calibration_for(&all, 150).unwrap();

        // Assert
        assert_eq!(chosen.calibration_id, "b");
        assert!(after);
    }

    #[test]
    fn calibration_for_skips_deleted_calibrations() {
        // Arrange
        let mut gone = calibration("b", 200, 1);
        gone.deleted = true;
        let all = [calibration("a", 100, 1), gone];

        // Act
        let chosen = calibration_for(&all, 250).map(|(c, _)| c.calibration_id.clone());

        // Assert
        assert_eq!(chosen.as_deref(), Some("a"));
    }

    #[test]
    fn merge_calibrations_keeps_both_sides_and_the_newer_copy_of_a_shared_one() {
        // Arrange — both have "a"; theirs deleted it later.
        let mut theirs_a = calibration("a", 100, 9);
        theirs_a.deleted = true;
        let ours = [calibration("a", 100, 5), calibration("b", 200, 1)];
        let theirs = [theirs_a, calibration("c", 50, 1)];

        // Act
        let merged = merge_calibrations(&ours, &theirs);
        let back = merge_calibrations(&theirs, &ours);

        // Assert — same answer either way round, oldest capture first.
        assert_eq!(merged, back);
        assert_eq!(merged.iter().map(|c| c.calibration_id.as_str()).collect::<Vec<_>>(), ["c", "a", "b"]);
        assert!(merged[1].deleted);
    }

    #[test]
    fn a_calibration_round_trips_through_json_with_absent_fields_left_out() {
        // Arrange
        let c = calibration("a", 100, 5);

        // Act
        let text = serde_json::to_string(&c).unwrap();
        let back: BikeCalibration = serde_json::from_str(&text).unwrap();

        // Assert
        assert_eq!(back, c);
        assert!(!text.contains("\"steer_axis\":") && !text.contains("deleted"), "{text}");
    }
}
