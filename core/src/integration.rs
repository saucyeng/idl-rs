//! Trapezoidal integration — cumulative sum of trapezoidal areas between samples.
//!
//! Equivalent to scipy.signal.cumtrapz(y, dx=1/fs, initial=0).
//! sci-rs 0.4 does not include cumtrapz; the algorithm is implemented directly.
//!
//! See docs/signal_pipeline.md and IDL0_SPEC.md §10.

/// Integrates a signal using the trapezoidal rule, returning a cumulative
/// integral of the same length as the input.
///
/// Equivalent to `scipy.integrate.cumtrapz(data, dx=1/sample_rate_hz, initial=0)`.
/// Output[0] is always 0.0 (zero initial condition). Output[i] accumulates the
/// area of trapezoids between adjacent samples.
///
/// `data`: input signal samples — units determine output units
///         (e.g., m/s² → m/s after integrating, m/s → m after a second pass)
/// `sample_rate_hz`: sample rate of `data` in Hz; determines time step dt = 1/fs
///
/// Returns cumulative integral in units of (input units × seconds), same length
/// as `data`.
///
/// Algorithm: result[0] = 0; result[i] = result[i-1] + (data[i-1] + data[i]) / 2 * dt
pub fn integrate(data: &[f64], sample_rate_hz: f64) -> Vec<f64> {
    let dt = 1.0 / sample_rate_hz;
    let mut result = Vec::with_capacity(data.len());
    result.push(0.0_f64);
    for i in 1..data.len() {
        let prev = result[i - 1];
        let area = (data[i - 1] + data[i]) * 0.5 * dt;
        result.push(prev + area);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrate_constant_input_produces_linear_output_with_correct_slope() {
        // Arrange — constant 2.0 m/s² at 100 Hz for 1 second
        let sample_rate_hz = 100.0_f64;
        let value = 2.0_f64;
        let data: Vec<f64> = vec![value; 100];
        let dt = 1.0 / sample_rate_hz;

        // Act
        let output = integrate(&data, sample_rate_hz);

        // Assert — output[i] = value * i * dt (linear ramp, slope = value m/s² → m/s)
        for i in 0..output.len() {
            let expected = value * i as f64 * dt;
            assert!(
                (output[i] - expected).abs() < 1e-10,
                "output[{i}] = {}, expected {expected}",
                output[i],
            );
        }
    }
}
