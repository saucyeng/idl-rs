//! Process-noise model: maps Allan-variance IMU noise parameters to the discrete
//! process-noise variances the filter/smoother accumulate over a timestep.

/// Per-axis (isotropic) IMU noise parameters from an Allan-variance
/// characterization. Each is a random-walk coefficient `N` defined so that the
/// variance accumulated over time `t` is `N²·t`.
///
/// - `gyro_arw`: angle random walk (gyro white noise), rad/√s — drives attitude.
/// - `accel_vrw`: velocity random walk (accel white noise), (m/s)/√s — drives velocity.
/// - `gyro_bias_rw`: gyro bias random-walk rate, (rad/s)/√s.
/// - `accel_bias_rw`: accel bias random-walk rate, (m/s²)/√s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoise {
    pub gyro_arw: f64,
    pub accel_vrw: f64,
    pub gyro_bias_rw: f64,
    pub accel_bias_rw: f64,
}

/// Discrete per-axis process-noise variances accumulated over one step `dt`.
/// Diagonal building blocks for the error-state process-noise covariance `Q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessNoise {
    /// Attitude-increment variance per axis (rad²).
    pub attitude_var: f64,
    /// Velocity-increment variance per axis ((m/s)²).
    pub velocity_var: f64,
    /// Gyro-bias-increment variance per axis ((rad/s)²).
    pub gyro_bias_var: f64,
    /// Accel-bias-increment variance per axis ((m/s²)²).
    pub accel_bias_var: f64,
}

/// Discretizes [`ImuNoise`] over a step `dt` (seconds): for a random-walk
/// coefficient `N` (units/√s), the variance accumulated over `dt` is `N²·dt` (the
/// standard random-walk discretization, the Allan-variance +1/2-slope region).
/// Returns the per-axis diagonal variances for the attitude, velocity, and bias
/// process-noise blocks.
pub fn process_noise_over_dt(noise: &ImuNoise, dt: f64) -> ProcessNoise {
    ProcessNoise {
        attitude_var: noise.gyro_arw.powi(2) * dt,
        velocity_var: noise.accel_vrw.powi(2) * dt,
        gyro_bias_var: noise.gyro_bias_rw.powi(2) * dt,
        accel_bias_var: noise.accel_bias_rw.powi(2) * dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn process_noise_is_coefficient_squared_times_dt() {
        // Arrange — round coefficients so the expected values are independent literals.
        let noise = ImuNoise {
            gyro_arw: 0.1,       // rad/√s
            accel_vrw: 0.2,      // (m/s)/√s
            gyro_bias_rw: 0.01,  // (rad/s)/√s
            accel_bias_rw: 0.05, // (m/s²)/√s
        };

        // Act — accumulated over dt = 0.5 s.
        let q = process_noise_over_dt(&noise, 0.5);

        // Assert — variance = coefficient² · dt.
        assert_relative_eq!(q.attitude_var, 0.005, epsilon = 1e-15); // 0.1²·0.5
        assert_relative_eq!(q.velocity_var, 0.02, epsilon = 1e-15); // 0.2²·0.5
        assert_relative_eq!(q.gyro_bias_var, 5e-5, epsilon = 1e-18); // 0.01²·0.5
        assert_relative_eq!(q.accel_bias_var, 1.25e-3, epsilon = 1e-15); // 0.05²·0.5
    }

    #[test]
    fn process_noise_zero_dt_is_zero() {
        // Arrange
        let noise = ImuNoise {
            gyro_arw: 0.1,
            accel_vrw: 0.2,
            gyro_bias_rw: 0.01,
            accel_bias_rw: 0.05,
        };

        // Act
        let q = process_noise_over_dt(&noise, 0.0);

        // Assert — no elapsed time ⇒ no accumulated process noise.
        assert_eq!(q.attitude_var, 0.0);
        assert_eq!(q.velocity_var, 0.0);
        assert_eq!(q.gyro_bias_var, 0.0);
        assert_eq!(q.accel_bias_var, 0.0);
    }
}
