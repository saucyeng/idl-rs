//! Per-sample validity gates the measurement factors consult (design §5). For
//! M2a this is the **(quasi-)stationary** detector that gates ZUPT / ZARU; it
//! composes [`crate::statistics::zupt_flags`] over the IMU0 vector streams. The
//! topout/bottomout event detectors (tight travel-DC references) are deferred to
//! M3 (real-log tuning) — they are not in the M2a factor set.

use crate::estimate::model::SampleContext;
use crate::statistics::zupt_flags;
use nalgebra::Vector3;

/// Builds the per-sample [`SampleContext`] stream by flagging (quasi-)stationary
/// samples from the IMU0 specific-force and angular-rate vectors.
///
/// A sample is stationary when, over a centered `window`, the accel magnitude
/// **standard deviation** is below `accel_std_thresh` (m/s²) — steady ≈ gravity,
/// no dynamics — and the mean gyro magnitude is below `gyro_thresh` (rad/s). The
/// accel *level* is intentionally not tested (it sits at ~g when still), only its
/// steadiness, so the gate is independent of accel-bias offset.
pub fn stationary_context(
    accel0: &[Vector3<f64>],
    gyro0: &[Vector3<f64>],
    window: usize,
    accel_std_thresh: f64,
    gyro_thresh: f64,
) -> Vec<SampleContext> {
    let accel_mag: Vec<f64> = accel0.iter().map(|a| a.norm()).collect();
    let gyro_mag: Vec<f64> = gyro0.iter().map(|w| w.norm()).collect();
    zupt_flags(&accel_mag, &gyro_mag, window, accel_std_thresh, gyro_thresh)
        .into_iter()
        .map(|stationary| SampleContext { stationary })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_gravity_zero_rate_stream_is_all_stationary() {
        // Arrange — 30 samples of constant gravity, zero angular rate.
        let accel = vec![Vector3::new(0.0, 0.0, 9.81); 30];
        let gyro = vec![Vector3::zeros(); 30];

        // Act
        let ctx = stationary_context(&accel, &gyro, 5, 0.05, 0.02);

        // Assert
        assert!(ctx.iter().all(|c| c.stationary));
    }

    #[test]
    fn angular_rate_burst_marks_those_samples_moving() {
        // Arrange — steady except a sustained yaw rate over samples 12..18.
        let mut gyro = vec![Vector3::zeros(); 30];
        for w in gyro.iter_mut().take(18).skip(12) {
            *w = Vector3::new(0.0, 0.0, 1.0); // 1 rad/s ≫ threshold
        }
        let accel = vec![Vector3::new(0.0, 0.0, 9.81); 30];

        // Act
        let ctx = stationary_context(&accel, &gyro, 3, 0.05, 0.02);

        // Assert — the rotating samples are not stationary; a quiet early sample is.
        assert!(!ctx[14].stationary);
        assert!(ctx[2].stationary);
    }
}
