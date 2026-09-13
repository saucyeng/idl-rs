//! What `session calibrate` needs that is not the fit itself: turning a parsed
//! session into the model's input, and scoring a fitted record against a
//! `session synth` truth file.
//!
//! Both live here rather than in the CLI so the command stays a wrapper
//! (R230: no CLI-only logic) and so the CLI needs no `nalgebra` of its own.

use nalgebra::{Matrix3, Vector3};

use crate::calibration::rigid::{
    Body, CalibrationInput, CalibrationRecord, MountOrigin, SensorStream, GRAVITY_M_S2,
};
use crate::estimate::noise::ImuNoise;
use crate::session::Session;
use crate::synth::{Truth, ACCEL_SIGMA_M_S2, GYRO_SIGMA_RAD_S};

/// Most IMU indices `IDL0_SPEC.md` §3.2 could ever name. Three are defined
/// today; scanning a few more costs nothing and means a fourth sensor needs no
/// change here.
const MAX_IMU_INDEX: u8 = 8;

/// Why a session cannot be fed to the calibration model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationInputError {
    /// The session carries no complete `IMU{n}_*` sextet.
    NoImuChannels,
    /// The channel registry declares no nominal IMU rate, which the model —
    /// written on a uniform grid — needs.
    NoSampleRate,
}

impl std::fmt::Display for CalibrationInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationInputError::NoImuChannels => write!(
                f,
                "no IMU channels in this session; a calibration needs at least IMU0's six axes"
            ),
            CalibrationInputError::NoSampleRate => write!(
                f,
                "the channel registry declares no IMU sample rate, which the model needs"
            ),
        }
    }
}

impl std::error::Error for CalibrationInputError {}

/// Builds the calibration model's input from a parsed session: every complete
/// `IMU{n}_*` sextet, converted from the wire's `g` and `dps` (SPEC §5.3) into
/// the m/s² and rad/s the model works in.
///
/// The sample rate comes from the channel registry's nominal rate rather than
/// from the timestamps: the model is written on a uniform grid, and the
/// registry records what the device was configured at.
///
/// `noise` defaults to the generator's reference σ, which is also the spec's
/// working default (§1.3).
///
/// # Errors
///
/// [`CalibrationInputError`] when the session has no IMU sextet or no declared
/// rate.
pub fn calibration_input_from_session(
    session: &Session,
) -> Result<CalibrationInput, CalibrationInputError> {
    const AXES: [&str; 6] = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"];
    let dps_to_rad = std::f64::consts::PI / 180.0;

    let mut sensors = Vec::new();
    let mut sample_rate_hz = 0.0_f64;

    for index in 0..MAX_IMU_INDEX {
        let mut columns = Vec::with_capacity(AXES.len());
        for axis in AXES {
            let name = format!("IMU{index}_{axis}");
            let Some(channel) = session.channels.iter().find(|c| c.channel_id == name) else {
                break;
            };
            if channel.nominal_rate_hz > 0.0 {
                sample_rate_hz = channel.nominal_rate_hz;
            }
            columns.push(channel.materialize());
        }
        if columns.len() != AXES.len() {
            continue;
        }

        // A partial last sample on one axis would make the streams ragged,
        // which the model refuses; truncating to the shortest is the only
        // reading that keeps every axis on the same grid slot.
        let n = columns.iter().map(Vec::len).min().unwrap_or(0);
        sensors.push(SensorStream {
            imu_index: index,
            body: Body::of_imu(index),
            gyro_rad_s: (0..n)
                .map(|k| Vector3::new(columns[3][k], columns[4][k], columns[5][k]) * dps_to_rad)
                .collect(),
            accel_m_s2: (0..n)
                .map(|k| Vector3::new(columns[0][k], columns[1][k], columns[2][k]) * GRAVITY_M_S2)
                .collect(),
        });
    }

    if sensors.is_empty() {
        return Err(CalibrationInputError::NoImuChannels);
    }
    if sample_rate_hz <= 0.0 {
        return Err(CalibrationInputError::NoSampleRate);
    }

    Ok(CalibrationInput {
        sample_rate_hz,
        noise: ImuNoise {
            gyro_arw: GYRO_SIGMA_RAD_S,
            accel_vrw: ACCEL_SIGMA_M_S2,
            gyro_bias_rw: 0.0,
            accel_bias_rw: 0.0,
        },
        captured_utc: None,
        sensors,
    })
}

/// The worst error in each fitted quantity against a `session synth` truth
/// file. Every field is a magnitude, so zero is perfect and larger is worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthErrors {
    /// Worst geodesic mount error over the **measured** rotations, degrees.
    pub worst_mount_deg: f64,
    /// Worst per-axis lever-arm error, millimetres. `None` when no lever arm
    /// was fitted.
    pub worst_lever_mm: Option<f64>,
    /// Steering-axis error, degrees. `None` when the hinge was not fitted, or
    /// the truth file has no hinge block.
    pub steer_axis_deg: Option<f64>,
    /// Worst per-axis gyro-bias error, rad/s. `None` with no rest hold.
    pub worst_gyro_bias_rad_s: Option<f64>,
}

/// Scores a fitted record against the truth the generator recorded.
///
/// The truth stores **body→sensor** rotations while the record reports
/// **sensor→body**, so the comparison transposes. A rotation marked
/// [`MountOrigin::Gauge`] is skipped: it is the spec §1.2a convention, not a
/// measurement, and scoring a convention against a truth scores nothing.
pub fn truth_errors(record: &CalibrationRecord, truth: &Truth) -> TruthErrors {
    let mut worst_mount_deg: f64 = 0.0;
    let mut worst_lever_mm: Option<f64> = None;
    let mut worst_gyro_bias_rad_s: Option<f64> = None;

    for sensor in &record.sensors {
        let Some(entry) = truth.sensors.iter().find(|s| s.index == sensor.imu_index) else {
            continue;
        };
        let m = entry.rotation_body_to_sensor;
        let expected = Matrix3::new(
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        )
        .transpose();

        if sensor.mount_origin == MountOrigin::Measured {
            let fitted = sensor.mount.to_rotation_matrix();
            let trace = (fitted.matrix().transpose() * expected).trace();
            let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
            worst_mount_deg = worst_mount_deg.max(angle);
        }
        if let Some(lever) = sensor.lever_m {
            for axis in 0..3 {
                let error = (lever[axis] - entry.lever_arm_m[axis]).abs() * 1000.0;
                worst_lever_mm = Some(worst_lever_mm.unwrap_or(0.0).max(error));
            }
        }
        if let Some(bias) = sensor.gyro_bias_rad_s {
            for axis in 0..3 {
                let error = (bias[axis] - entry.gyro_bias_rad_s[axis]).abs();
                worst_gyro_bias_rad_s = Some(worst_gyro_bias_rad_s.unwrap_or(0.0).max(error));
            }
        }
    }

    let steer_axis_deg = match (&record.steer_axis, &truth.hinge) {
        (Some(fitted), Some(hinge)) => {
            let expected =
                Vector3::new(hinge.steer_axis_r[0], hinge.steer_axis_r[1], hinge.steer_axis_r[2]);
            Some(fitted.dot(&expected).clamp(-1.0, 1.0).acos().to_degrees())
        }
        _ => None,
    };

    TruthErrors { worst_mount_deg, worst_lever_mm, steer_axis_deg, worst_gyro_bias_rad_s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::rigid;
    use crate::parse;
    use crate::synth::{self, Protocol, SynthConfig};

    fn generated() -> (Vec<u8>, Truth) {
        let config = SynthConfig {
            protocol: Protocol::Calibration,
            imu_rate_hz: 833,
            // The generator's σ is per-sample; the spec's budget is an
            // angle-random-walk coefficient, so √ODR puts the two on the same
            // footing. See `calibration::rigid`'s tests.
            noise_scale: 28.861739379323623,
            ..SynthConfig::default()
        };
        let out = synth::generate(&config).expect("generated");
        (out.log, out.truth)
    }

    #[test]
    fn a_parsed_session_becomes_three_si_sensor_streams_on_the_registry_rate() {
        // Arrange
        let (log, _) = generated();
        let session = parse::parse(&log).expect("parses").session;

        // Act
        let input = calibration_input_from_session(&session).expect("builds an input");

        // Assert — wire `dps` and `g` are gone; the model sees rad/s and m/s².
        // The rate is the registry's, which the parser snaps to the LSM6DSO32's
        // real 833⅓ Hz ODR rather than the integer the header carries.
        assert!((input.sample_rate_hz - 833.3333333).abs() < 1e-4, "{}", input.sample_rate_hz);
        assert_eq!(input.sensors.len(), 3);
        assert_eq!(input.sensors[1].body, Body::Front);
        assert_eq!(input.sensors[0].body, Body::Rear);
        let resting = input.sensors[0].accel_m_s2[0].norm();
        assert!((resting - GRAVITY_M_S2).abs() < 2.0, "resting specific force {resting} m/s²");
    }

    #[test]
    fn a_session_with_no_imu_channels_is_a_typed_error() {
        // Arrange
        let session = Session {
            session_id: "x".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: crate::session::TimestampSource::Header,
            config_checksum: None,
            source_format: crate::session::SourceFormat::Idl0,
            blob_sha256: "y".to_string(),
            channels: Vec::new(),
        };

        // Act
        let error = calibration_input_from_session(&session).expect_err("refused");

        // Assert
        assert_eq!(error, CalibrationInputError::NoImuChannels);
    }

    #[test]
    fn scoring_a_fit_against_its_own_truth_lands_inside_every_spec_threshold() {
        // Arrange
        let (log, truth) = generated();
        let session = parse::parse(&log).expect("parses").session;
        let input = calibration_input_from_session(&session).expect("input");
        let record = rigid::calibrate(&input).expect("fits");

        // Act
        let errors = truth_errors(&record, &truth);

        // Assert — spec §5's acceptance table, end to end through the parser.
        assert!(errors.worst_mount_deg < 0.5, "{:?}", errors);
        assert!(errors.worst_lever_mm.unwrap() < 10.0, "{:?}", errors);
        assert!(errors.steer_axis_deg.unwrap() < 1.0, "{:?}", errors);
        assert!(errors.worst_gyro_bias_rad_s.unwrap() < 0.002, "{:?}", errors);
    }

    #[test]
    fn the_gauge_rotation_is_left_out_of_the_score() {
        // Arrange — IMU0's mount is the §1.2a gauge, so it is a convention and
        // scoring it would report an error that means nothing. The generator
        // happens to make it the identity too, so the only way to see the
        // exclusion is that the worst error is the fitted sensors'.
        let (log, truth) = generated();
        let session = parse::parse(&log).expect("parses").session;
        let record = rigid::calibrate(&calibration_input_from_session(&session).unwrap()).unwrap();

        // Act
        let errors = truth_errors(&record, &truth);

        // Assert
        assert!(errors.worst_mount_deg > 0.0, "at least one measured mount was scored");
        assert_eq!(
            record.sensors.iter().filter(|s| s.mount_origin == MountOrigin::Gauge).count(),
            1
        );
    }
}
