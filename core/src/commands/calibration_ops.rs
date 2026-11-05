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

/// Which sensors and which stretch of a session feed the fit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalibrationSelection {
    /// IMU indices to include; `None` = every complete sextet.
    pub imus: Option<Vec<u8>>,
    /// IMU indices to treat as riding the **rear (chassis) body** although
    /// their default is the front one: a fork pod over a window where the
    /// bars did not turn is rigid with the frame, so its lever arm from
    /// `IMU0` is fitted the same way as the swingarm pod's.
    pub rigid_with_chassis: Vec<u8>,
    /// Keep only `[t0, t1)`, seconds from the first grid sample.
    pub window_s: Option<(f64, f64)>,
    /// Passed through to [`CalibrationInput::min_median_rate_rad_s`].
    pub min_median_rate_rad_s: Option<f64>,
}

/// One axis as recorded: its own hardware times and its values.
struct Axis {
    t_us: Vec<i64>,
    values: Vec<f64>,
}

/// `axis` at time `t` (µs) by linear interpolation; held at either end.
fn axis_at(axis: &Axis, t: f64) -> f64 {
    let n = axis.t_us.len().min(axis.values.len());
    let i = axis.t_us[..n].partition_point(|&x| (x as f64) < t);
    if i == 0 {
        return axis.values[0];
    }
    if i >= n {
        return axis.values[n - 1];
    }
    let (t0, t1) = (axis.t_us[i - 1] as f64, axis.t_us[i] as f64);
    let w = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    axis.values[i - 1] + w * (axis.values[i] - axis.values[i - 1])
}

/// Builds the calibration model's input from a parsed session: every complete
/// `IMU{n}_*` sextet, converted from the wire's `g` and `dps` (SPEC §5.3) into
/// the m/s² and rad/s the model works in.
///
/// **Time is recorded, not assumed.** The model is written on one uniform
/// grid, but a real logger samples each IMU on its own clock (the Beacon
/// calibration of 2026-11-04: 21 935 and 22 026 samples, and more, over the
/// same 47 s), so pairing samples by index would pair different instants.
/// Every axis is therefore read at the grid times by linear interpolation of
/// its own `t_us`. The grid runs at `IMU0`'s gyro rate over the span every
/// stream covers. A session whose streams already share one time base (the
/// synthetic generator's) is taken as is.
///
/// The accelerometer is the low-g stream (`IMU{n}_LgAccel*`) where the
/// importer split it from a high-g one: it shares the gyro's clock, and the
/// merged `IMU{n}_Accel*` runs on the high-g grid.
///
/// `noise` defaults to the generator's reference σ, which is also the spec's
/// working default (§1.3).
///
/// # Errors
///
/// [`CalibrationInputError`] when the session has no IMU sextet or no declared
/// rate.
pub fn calibration_input_from_session(session: &Session) -> Result<CalibrationInput, CalibrationInputError> {
    calibration_input_from_session_with(session, &CalibrationSelection::default())
}

/// [`calibration_input_from_session`] narrowed by `selection`.
///
/// # Errors
///
/// As [`calibration_input_from_session`]; also
/// [`CalibrationInputError::NoImuChannels`] when the window leaves fewer than
/// two samples.
pub fn calibration_input_from_session_with(
    session: &Session,
    selection: &CalibrationSelection,
) -> Result<CalibrationInput, CalibrationInputError> {
    const ACCEL: [&str; 3] = ["AccelX", "AccelY", "AccelZ"];
    const GYRO: [&str; 3] = ["GyroX", "GyroY", "GyroZ"];
    let dps_to_rad = std::f64::consts::PI / 180.0;
    let find = |name: &str| session.channels.iter().find(|c| c.channel_id == name);

    // (imu index, [accel x, y, z, gyro x, y, z])
    let mut streams: Vec<(u8, Vec<Axis>)> = Vec::new();
    let mut sample_rate_hz = 0.0_f64;
    for index in 0..MAX_IMU_INDEX {
        if let Some(wanted) = &selection.imus {
            if !wanted.contains(&index) {
                continue;
            }
        }
        let mut axes = Vec::with_capacity(6);
        for axis in ACCEL {
            let Some(channel) = find(&format!("IMU{index}_Lg{axis}")).or_else(|| find(&format!("IMU{index}_{axis}"))) else {
                break;
            };
            axes.push(Axis { t_us: channel.t_us.clone(), values: channel.materialize() });
        }
        for axis in GYRO {
            let Some(channel) = find(&format!("IMU{index}_{axis}")) else { break };
            if sample_rate_hz <= 0.0 && channel.nominal_rate_hz > 0.0 {
                sample_rate_hz = channel.nominal_rate_hz;
            }
            axes.push(Axis { t_us: channel.t_us.clone(), values: channel.materialize() });
        }
        if axes.len() == 6 && axes.iter().all(|a| !a.values.is_empty() && !a.t_us.is_empty()) {
            streams.push((index, axes));
        }
    }

    if streams.is_empty() {
        return Err(CalibrationInputError::NoImuChannels);
    }
    if sample_rate_hz <= 0.0 {
        return Err(CalibrationInputError::NoSampleRate);
    }

    // The grid: the shared times when every axis already has them, else
    // IMU0's gyro rate over the overlap of every stream.
    let first = streams[0].1[3].t_us.clone();
    let shared = streams.iter().all(|(_, axes)| axes.iter().all(|a| a.t_us == first && a.values.len() >= first.len()));
    let full_grid_us: Vec<f64> = if shared {
        first.iter().map(|&t| t as f64).collect()
    } else {
        let start = streams.iter().flat_map(|(_, axes)| axes.iter().map(|a| a.t_us[0])).max().unwrap_or(0) as f64;
        let end = streams
            .iter()
            .flat_map(|(_, axes)| axes.iter().map(|a| *a.t_us.last().unwrap_or(&0)))
            .min()
            .unwrap_or(0) as f64;
        let step = 1e6 / sample_rate_hz;
        let n = if end > start { ((end - start) / step).floor() as usize + 1 } else { 0 };
        (0..n).map(|k| start + k as f64 * step).collect()
    };
    let origin = full_grid_us.first().copied().unwrap_or(0.0);
    let kept: Vec<usize> = (0..full_grid_us.len())
        .filter(|&k| match selection.window_s {
            Some((t0, t1)) => {
                let t = (full_grid_us[k] - origin) / 1e6;
                t >= t0 && t < t1
            }
            None => true,
        })
        .collect();
    if kept.len() < 2 {
        return Err(CalibrationInputError::NoImuChannels);
    }

    let sensors = streams
        .iter()
        .map(|(index, axes)| {
            let read = |a: usize, k: usize| if shared { axes[a].values[k] } else { axis_at(&axes[a], full_grid_us[k]) };
            let body = if selection.rigid_with_chassis.contains(index) { Body::Rear } else { Body::of_imu(*index) };
            SensorStream {
                imu_index: *index,
                body,
                gyro_rad_s: kept.iter().map(|&k| Vector3::new(read(3, k), read(4, k), read(5, k)) * dps_to_rad).collect(),
                accel_m_s2: kept.iter().map(|&k| Vector3::new(read(0, k), read(1, k), read(2, k)) * GRAVITY_M_S2).collect(),
            }
        })
        .collect();

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
        min_median_rate_rad_s: selection.min_median_rate_rad_s,
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

    /// One IMU's six axes on its own clock: sample `k` at `t0_us + k·step_us`,
    /// every axis `value(t)` (dps for the gyro, g for the accel), the accel
    /// named with `accel_prefix` (`"Accel"` or `"LgAccel"`).
    fn imu(index: u8, t0_us: i64, step_us: i64, n: usize, accel_prefix: &str, value: impl Fn(f64) -> f64) -> Vec<crate::session::Channel> {
        let t_us: Vec<i64> = (0..n as i64).map(|k| t0_us + k * step_us).collect();
        let values: Vec<f64> = t_us.iter().map(|&t| value(t as f64)).collect();
        let axes = [
            format!("{accel_prefix}X"),
            format!("{accel_prefix}Y"),
            format!("{accel_prefix}Z"),
            "GyroX".to_string(),
            "GyroY".to_string(),
            "GyroZ".to_string(),
        ];
        axes.iter()
            .map(|axis| crate::session::Channel {
                channel_id: format!("IMU{index}_{axis}"),
                t_us: t_us.clone(),
                t_recorded_us: None,
                nominal_rate_hz: 1e6 / step_us as f64,
                column: crate::session::RawColumn::F64(values.clone()),
                source_kind: format!("imu{index}"),
                unit: if axis.starts_with("Gyro") { "dps".to_string() } else { "g".to_string() },
                gaps: Vec::new(),
            })
            .collect()
    }

    fn session_of(channels: Vec<crate::session::Channel>) -> Session {
        Session {
            session_id: "s".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: crate::session::TimestampSource::Header,
            config_checksum: None,
            source_format: crate::session::SourceFormat::Idl0,
            blob_sha256: String::new(),
            channels,
        }
    }

    #[test]
    fn calibration_input_two_imus_on_different_clocks_each_read_at_the_shared_grid_times() {
        // Arrange — IMU0 every 10 ms from 0; IMU1 every 9 ms from 4 ms; both a
        // ramp in time, which linear interpolation reproduces exactly.
        let ramp = |t_us: f64| t_us / 1000.0;
        let mut channels = imu(0, 0, 10_000, 101, "Accel", ramp);
        channels.extend(imu(1, 4_000, 9_000, 112, "Accel", ramp));
        let session = session_of(channels);

        // Act
        let input = calibration_input_from_session(&session).unwrap();

        // Assert — the grid starts where both streams have begun (4 ms) and
        // steps at IMU0's 100 Hz; each IMU reads the ramp at those times.
        let dps = std::f64::consts::PI / 180.0;
        assert_eq!(input.sample_rate_hz, 100.0);
        for k in [0usize, 7, 50] {
            let t_ms = 4.0 + 10.0 * k as f64;
            assert!((input.sensors[0].gyro_rad_s[k].x - t_ms * dps).abs() < 1e-9, "IMU0 sample {k}");
            assert!((input.sensors[1].gyro_rad_s[k].x - t_ms * dps).abs() < 1e-9, "IMU1 sample {k}");
        }
        assert_eq!(input.sensors[0].gyro_rad_s.len(), input.sensors[1].gyro_rad_s.len());
    }

    #[test]
    fn calibration_input_low_g_stream_present_preferred_over_the_merged_accel() {
        // Arrange — the merged `Accel` reads 5 g, the low-g stream 1 g.
        let mut channels = imu(0, 0, 10_000, 50, "LgAccel", |_| 1.0);
        channels.extend(imu(0, 0, 10_000, 50, "Accel", |_| 5.0).into_iter().filter(|c| c.channel_id.contains("_Accel")));
        let session = session_of(channels);

        // Act
        let input = calibration_input_from_session(&session).unwrap();

        // Assert
        assert!((input.sensors[0].accel_m_s2[3].x - GRAVITY_M_S2).abs() < 1e-9);
    }

    #[test]
    fn calibration_input_with_a_window_an_imu_subset_and_a_rigid_fork_pod_narrowed_accordingly() {
        // Arrange
        let mut channels = imu(0, 0, 10_000, 101, "Accel", |_| 0.0);
        channels.extend(imu(1, 0, 10_000, 101, "Accel", |_| 0.0));
        channels.extend(imu(2, 0, 10_000, 101, "Accel", |_| 0.0));
        let session = session_of(channels);
        let selection = CalibrationSelection {
            imus: Some(vec![0, 1]),
            rigid_with_chassis: vec![1],
            window_s: Some((0.2, 0.5)),
            min_median_rate_rad_s: Some(0.5),
        };

        // Act
        let input = calibration_input_from_session_with(&session, &selection).unwrap();

        // Assert
        assert_eq!(input.sensors.iter().map(|s| s.imu_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(input.sensors[1].body, Body::Rear);
        assert_eq!(input.sensors[0].gyro_rad_s.len(), 30);
        assert_eq!(input.min_median_rate_rad_s, Some(0.5));
    }
}
