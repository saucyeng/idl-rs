//! A stretch of a library session as the drivetrain sim's excitation
//! (`idl_rs::sim::replay`, sim spec §5): IMU0 from the session, turned into
//! the bike's side plane by the calibration the session's bike had then, the
//! rear travel the physics estimator solves over the stretch, and GPS speed.
//!
//! The estimator runs through [`estimate_physics`], so its channels are kept
//! as the session's derived file too. A bike whose calibration has no current
//! physics fit is fitted on the whole session first (the fit wants a minute
//! of riding to pick its windows from); only the stretch is solved.

use std::path::Path;

use idl_rs::estimate::physics::pipeline::Cropped;
use idl_rs::estimate::run::EstimatorInput;
use idl_rs::sim::excite::Excitation;
use idl_rs::sim::replay::{excitation, Mount, Recording};
use idl_rs::store::parquet::open_session_lazy;
use idl_rs::store::session_json::{effective_start_ms, read_session_json};

use crate::bikelog::calibration_for_session;
use crate::physics::{estimate_physics, PhysicsError, PhysicsErrorKind, PhysicsOptions, PhysicsReport};

/// Longest stretch a replay takes, s.
pub const MAX_REPLAY_S: f64 = 120.0;

/// A replay's excitation and how it was made.
#[derive(Debug, Clone)]
pub struct Replay {
    /// The inputs, time 0 at the stretch's start.
    pub excitation: Excitation,
    /// The estimator run that gave the rear travel.
    pub physics: PhysicsReport,
    /// Mean ground speed over the stretch, m/s.
    pub mean_speed_mps: f64,
}

fn fail(kind: PhysicsErrorKind, message: impl Into<String>) -> PhysicsError {
    PhysicsError { kind, message: message.into() }
}

/// Session `session_id` from `from_s` to `to_s` (s from its first sample) as
/// an excitation, IMU0 low-passed at `cutoff_hz`. `phase` hears what the run
/// is doing ("estimating travel", "reading IMU0"). Errors: `NotFound` (the
/// session), `NotCalibrated` (no calibration, or one without a forward
/// direction), `Unusable` (a bad stretch, no solved travel or no GPS in it),
/// and the estimator's own.
pub fn replay(data_root: &Path, session_id: &str, from_s: f64, to_s: f64, cutoff_hz: f64, phase: &mut dyn FnMut(&str)) -> Result<Replay, PhysicsError> {
    if !(from_s >= 0.0 && to_s > from_s + 0.1 && to_s - from_s <= MAX_REPLAY_S) {
        return Err(fail(PhysicsErrorKind::Unusable, format!("a replay is 0.1 to {MAX_REPLAY_S} s of a session")));
    }
    let dir = data_root.join("sessions").join(session_id);
    let doc = read_session_json(&dir.join("session.json")).map_err(|e| fail(PhysicsErrorKind::NotFound, format!("{session_id}: {e:?}")))?;
    let handle = open_session_lazy(&dir).map_err(|e| fail(PhysicsErrorKind::Io, e.to_string()))?;
    let start_ms = effective_start_ms(&doc, handle.metadata().timestamp_utc_ms);
    let (cal, _) = calibration_for_session(data_root, &doc, start_ms).ok_or_else(|| fail(PhysicsErrorKind::NotCalibrated, format!("{} has no calibration", doc.bike_id)))?;
    let forward = cal.fit.forward.ok_or_else(|| fail(PhysicsErrorKind::NotCalibrated, "the calibration has no forward direction (only IMU0 was recorded)"))?;
    let mount = Mount { up: cal.fit.up, forward, gyro_bias_rad_s: cal.fit.imu0_gyro_bias_rad_s };

    // rear travel: fit on the whole session first if the calibration has no current fit
    phase("estimating travel");
    let span_ms = Some(((from_s * 1000.0).floor() as i64, (to_s * 1000.0).ceil() as i64));
    let physics = estimate_physics(data_root, session_id, &PhysicsOptions { span_ms, fit_whole_session: true, ..PhysicsOptions::default() })?;
    let travel: Vec<(f64, f64)> = physics.grid_us.iter().zip(&physics.rear_travel_m).map(|(&t, &s)| (t as f64 / 1e6, s)).collect();

    phase("reading IMU0");
    let lookup = Cropped { inner: &handle, from_us: (from_s * 1e6) as i64, to_us: (to_s * 1e6) as i64 };
    let input = EstimatorInput::from_lookup(&lookup).ok_or_else(|| fail(PhysicsErrorKind::Unusable, format!("{session_id} has no IMU0 from {from_s:.1} to {to_s:.1} s")))?;
    let accel: Vec<[f64; 3]> = input.imu0.accel.iter().map(|a| [a.x, a.y, a.z]).collect();
    let gyro: Vec<[f64; 3]> = input.imu0.gyro.iter().map(|g| [g.x, g.y, g.z]).collect();
    let speed: Vec<(f64, f64)> = input.gps.iter().map(|g| (g.time_s, g.velocity.x.hypot(g.velocity.y))).collect();
    let rec = Recording { t0_s: input.t0_s, dt_s: input.dt, accel: &accel, gyro: &gyro, travel: &travel, speed: &speed };
    let excitation = excitation(&rec, &mount, cutoff_hz).map_err(|e| fail(PhysicsErrorKind::Unusable, e.to_string()))?;
    let mean_speed_mps = excitation.speed_mps.iter().sum::<f64>() / excitation.speed_mps.len() as f64;
    Ok(Replay { excitation, physics, mean_speed_mps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_a_stretch_out_of_range_or_too_long_is_unusable() {
        // Arrange
        let root = std::env::temp_dir().join("idl-replay-no-such-root");

        // Act
        let backwards = replay(&root, "x", 5.0, 4.0, 100.0, &mut |_| {});
        let long = replay(&root, "x", 0.0, MAX_REPLAY_S + 1.0, 100.0, &mut |_| {});

        // Assert
        assert_eq!(backwards.unwrap_err().kind, PhysicsErrorKind::Unusable);
        assert_eq!(long.unwrap_err().kind, PhysicsErrorKind::Unusable);
    }

    #[test]
    fn replay_a_missing_session_is_not_found() {
        // Arrange
        let root = std::env::temp_dir().join("idl-replay-no-such-root");

        // Act
        let r = replay(&root, "0123456789abcdef0123456789abcdef", 0.0, 10.0, 100.0, &mut |_| {});

        // Assert
        assert_eq!(r.unwrap_err().kind, PhysicsErrorKind::NotFound);
    }
}
