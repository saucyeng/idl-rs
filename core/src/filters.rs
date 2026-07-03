//! Butterworth filters — wraps sci-rs butter_dyn + sosfiltfilt_dyn.
//!
//! Provides zero-phase high-pass and low-pass filtering for IMU signal
//! conditioning per IDL0_SPEC.md §10. scipy-equivalent function names.
//!
//! See docs/signal_pipeline.md and IDL0_SPEC.md §10.

use sci_rs::signal::filter::{
    design::{butter_dyn, DigitalFilter, FilterBandType, FilterOutputType, Sos},
    sosfiltfilt_dyn,
};

/// Applies a zero-phase Butterworth high-pass filter to a signal.
///
/// Suppresses DC offset and low-frequency drift before integration.
/// Uses sosfiltfilt (zero-phase, forward-backward pass) to avoid phase
/// distortion that would corrupt velocity and position output.
///
/// `data`: input signal samples — any units (raw LSB counts, g, dps, etc.)
/// `order`: filter order; IDL0_SPEC §10 default is 2
/// `cutoff_hz`: high-pass cutoff in Hz; IDL0_SPEC §10 default range 0.15–0.3 Hz
/// `sample_rate_hz`: sample rate of `data` in Hz
///
/// Returns filtered signal in same units as `data`.
///
/// sci-rs: butter_dyn() designs SOS coefficients, sosfiltfilt_dyn() applies
/// zero-phase forward-backward pass. SOS form chosen over BA to avoid
/// numerical instability at high filter orders.
pub fn highpass(
    data: &[f64],
    order: usize,
    cutoff_hz: f64,
    sample_rate_hz: f64,
) -> Vec<f64> {
    let sos = design_sos(order, cutoff_hz, FilterBandType::Highpass, sample_rate_hz);
    sosfiltfilt_dyn(data.iter(), &sos)
}

/// Applies a zero-phase Butterworth low-pass filter to a signal.
///
/// Removes high-frequency noise while preserving low-frequency content.
/// Uses sosfiltfilt (zero-phase, forward-backward pass) to avoid phase
/// distortion.
///
/// `data`: input signal samples — any units
/// `order`: filter order; IDL0_SPEC §10 default is 2
/// `cutoff_hz`: low-pass cutoff in Hz
/// `sample_rate_hz`: sample rate of `data` in Hz
///
/// Returns filtered signal in same units as `data`.
///
/// sci-rs: butter_dyn() designs SOS coefficients, sosfiltfilt_dyn() applies
/// zero-phase forward-backward pass. SOS form chosen over BA to avoid
/// numerical instability at high filter orders.
pub fn lowpass(
    data: &[f64],
    order: usize,
    cutoff_hz: f64,
    sample_rate_hz: f64,
) -> Vec<f64> {
    let sos = design_sos(order, cutoff_hz, FilterBandType::Lowpass, sample_rate_hz);
    sosfiltfilt_dyn(data.iter(), &sos)
}

// Designs Butterworth SOS coefficients for the given band type.
// SOS (second-order sections) avoids numerical issues of direct-form BA at
// high orders. Required format for sosfiltfilt_dyn.
fn design_sos(
    order: usize,
    cutoff_hz: f64,
    btype: FilterBandType,
    sample_rate_hz: f64,
) -> Vec<Sos<f64>> {
    match butter_dyn::<f64>(
        order,
        vec![cutoff_hz],
        Some(btype),
        Some(false),
        Some(FilterOutputType::Sos),
        Some(sample_rate_hz),
    ) {
        DigitalFilter::Sos(f) => f.sos,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: f64 = 200.0;
    const ORDER: usize = 2;
    const CUTOFF_HZ: f64 = 0.3;

    fn rms(data: &[f64]) -> f64 {
        let sum_sq: f64 = data.iter().map(|x| x * x).sum();
        (sum_sq / data.len() as f64).sqrt()
    }

    #[test]
    fn highpass_dc_input_converges_to_zero() {
        // Arrange — 5 s of constant 1.0 (pure DC) at 200 Hz
        let data: Vec<f64> = vec![1.0; 1000];

        // Act
        let output = highpass(&data, ORDER, CUTOFF_HZ, SAMPLE_RATE_HZ);

        // Assert — high-pass blocks DC; output RMS must be negligible
        let output_rms = rms(&output);
        assert!(
            output_rms < 1e-4,
            "expected DC suppressed to near zero, got RMS = {output_rms:.2e}",
        );
    }

    #[test]
    fn highpass_sinusoid_above_cutoff_passes_with_less_than_5_percent_attenuation() {
        // Arrange — 10 Hz sine at 200 Hz sample rate, amplitude 1.0
        // 10 Hz is ~33× the 0.3 Hz cutoff so the filter is flat in this band
        let signal_hz = 10.0_f64;
        let n = 1000_usize;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * signal_hz * i as f64 / SAMPLE_RATE_HZ).sin()
            })
            .collect();
        let input_rms = rms(&data);

        // Act
        let output = highpass(&data, ORDER, CUTOFF_HZ, SAMPLE_RATE_HZ);

        // Assert — amplitude within 5% of input; pass-band should be flat at 10 Hz
        let output_rms = rms(&output);
        let attenuation = (input_rms - output_rms).abs() / input_rms;
        assert!(
            attenuation < 0.05,
            "expected <5% attenuation at {signal_hz} Hz (cutoff {CUTOFF_HZ} Hz), \
             got {:.1}%",
            attenuation * 100.0,
        );
    }
}
