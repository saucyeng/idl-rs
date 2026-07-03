//! IMU calibration — bias capture and rotation matrix from gravity vector.
//!
//! Computes per-IMU calibration from a static sample set (bike stationary,
//! upright, rider off). Output written to `imu.bias` and `imu.orientation`
//! in idl0_config.json.
//!
//! See docs/calibration.md and IDL0_SPEC.md §11, §18.

use nalgebra::Vector3;

/// Computes per-axis bias by averaging N static samples.
///
/// Each sample is a 6-element vector [ax, ay, az, gx, gy, gz] in raw LSB counts.
/// The mean of all samples is the zero-point bias: the value the sensor outputs
/// when it should read zero (zero-g for accel axes, zero-rate for gyro axes).
///
/// `samples`: N static samples, each with 6 elements [ax, ay, az, gx, gy, gz]
///            in raw LSB counts — must all have the same length
///
/// Returns 6-element bias vector [ax_bias, ay_bias, az_bias, gx_bias, gy_bias, gz_bias]
/// in raw LSB counts.
pub fn compute_bias(samples: Vec<Vec<f64>>) -> Vec<f64> {
    let n = samples.len() as f64;
    let channels = samples[0].len();
    let mut sums = vec![0.0_f64; channels];
    for sample in &samples {
        for (i, &v) in sample.iter().enumerate() {
            sums[i] += v;
        }
    }
    sums.iter().map(|s| s / n).collect()
}

/// Computes the 3×3 rotation matrix mapping sensor body frame → vehicle frame (ISO 8855).
///
/// When the bike is stationary and upright, the accelerometer reads the gravity
/// reaction force. In vehicle frame (ISO 8855: Z=up), that vector is [0, 0, g].
/// The rotation matrix is the transform that maps the observed gravity direction
/// in sensor frame to [0, 0, 1] in vehicle frame.
///
/// `gravity_sensor`: mean accelerometer reading during static calibration
///                   [ax, ay, az] in sensor body frame, any units — only
///                   direction matters (magnitude is ignored)
///
/// Returns the rotation matrix as a flat 9-element row-major Vec<f64>:
///   [R00, R01, R02, R10, R11, R12, R20, R21, R22]
///
/// Degenerate case (sensor mounted exactly upside-down, gravity antiparallel to
/// vehicle Z): [`crate::rotation::rotation_between`] supplies a 180° flip about a
/// perpendicular axis (the in-plane choice is unobservable from gravity alone).
///
/// Delegates the rotation to [`crate::rotation::rotation_between`]; the resulting
/// matrix is stored row-major for FRB serialization.
pub fn rotation_from_gravity(gravity_sensor: Vec<f64>) -> Vec<f64> {
    let g = Vector3::new(gravity_sensor[0], gravity_sensor[1], gravity_sensor[2]);
    let vehicle_z = Vector3::new(0.0_f64, 0.0, 1.0);

    // Minimal rotation mapping observed gravity direction → vehicle Z-up.
    let rot = crate::rotation::rotation_between(g, vehicle_z).to_rotation_matrix();
    let m = rot.matrix();

    // Row-major flat layout: R[row][col] for row in 0..3, col in 0..3.
    (0..3)
        .flat_map(|row| (0..3).map(move |col| m[(row, col)]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn compute_bias_known_offset_returns_correct_mean() {
        // Arrange — 5 identical samples: accel [1000, -500, 16384], gyro [100, -200, 50] LSB
        let sample = vec![1000.0, -500.0, 16384.0, 100.0, -200.0, 50.0];
        let samples = vec![sample; 5];

        // Act
        let bias = compute_bias(samples);

        // Assert — mean of identical samples equals the sample value
        assert_eq!(bias.len(), 6);
        assert_relative_eq!(bias[0], 1000.0, epsilon = 1e-9);
        assert_relative_eq!(bias[1], -500.0, epsilon = 1e-9);
        assert_relative_eq!(bias[2], 16384.0, epsilon = 1e-9);
        assert_relative_eq!(bias[3], 100.0, epsilon = 1e-9);
        assert_relative_eq!(bias[4], -200.0, epsilon = 1e-9);
        assert_relative_eq!(bias[5], 50.0, epsilon = 1e-9);
    }

    #[test]
    fn rotation_from_gravity_identity_when_sensor_z_aligned_with_vehicle_z() {
        // Arrange — gravity reads along sensor +Z: sensor is already vehicle-aligned
        let gravity = vec![0.0, 0.0, 9.81];

        // Act
        let mat = rotation_from_gravity(gravity);

        // Assert — rotation matrix is identity (no correction needed)
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, (&actual, &exp)) in mat.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(actual, exp, epsilon = 1e-9, max_relative = 1e-9);
            assert!((actual - exp).abs() < 1e-9, "element {i}: got {actual}, expected {exp}");
        }
    }

    #[test]
    fn rotation_from_gravity_90_degree_rotation_when_sensor_x_points_up() {
        // Arrange — gravity reads along sensor +X: sensor X axis is vehicle Z axis
        let gravity = vec![9.81, 0.0, 0.0];

        // Act
        let mat = rotation_from_gravity(gravity);

        // Assert — applying this rotation to [1,0,0] must yield [0,0,1]
        // i.e. the first column of R^T (= first row of R) applied to [1,0,0]
        let rotated_x = [
            mat[0] * 1.0 + mat[1] * 0.0 + mat[2] * 0.0, // R row 0 · [1,0,0]
            mat[3] * 1.0 + mat[4] * 0.0 + mat[5] * 0.0, // R row 1 · [1,0,0]
            mat[6] * 1.0 + mat[7] * 0.0 + mat[8] * 0.0, // R row 2 · [1,0,0]
        ];
        assert_relative_eq!(rotated_x[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(rotated_x[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(rotated_x[2], 1.0, epsilon = 1e-9);
    }
}
