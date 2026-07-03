//! Parity gate: cases ported from the Dart evaluator suite
//! (`app/test/data/math_channel_evaluator_test.dart`). Value-domain cases pin
//! the exact output vectors the Dart suite asserts; DSP-backed cases assert
//! *delegation parity* — the evaluator's output equals a direct call to the
//! same core DSP function (the Dart suite used a passthrough adapter, so its
//! DSP cases only proved dispatch + arg wiring, which this reproduces against
//! the real engine). Variance cases are covered separately in Phase B.

use std::sync::Arc;

use crate::math::eval::{evaluate, ChannelLookup, LookupChannel, MathLapContext, MathOverlay};
use crate::math::MathEvalErrorKind;

const EPS: f64 = 1e-9;

struct Ctx(Vec<(String, Vec<f64>, f64)>);
impl ChannelLookup for Ctx {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        self.0
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, s, r)| LookupChannel { samples: s.clone().into(), sample_rate_hz: *r })
    }
    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        self.0
            .iter()
            .filter(|(_, _, r)| *r > 0.0)
            .map(|(_, s, r)| (s.len(), *r))
            .fold(None, |best, (len, rate)| match best {
                Some((bl, _)) if bl >= len => best,
                _ => Some((len, rate)),
            })
    }
}

fn ctx(pairs: &[(&str, Vec<f64>, f64)]) -> Ctx {
    Ctx(pairs.iter().cloned().map(|(n, s, r)| (n.to_string(), s, r)).collect())
}

fn no_laps() -> MathLapContext {
    MathLapContext::default()
}

fn assert_samples(expr: &str, c: &Ctx, expected: &[f64], expected_rate: f64) {
    let out = evaluate(expr, c, &no_laps()).unwrap();
    assert_eq!(out.samples.len(), expected.len(), "length mismatch for `{expr}`");
    for (i, (g, e)) in out.samples.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(g.is_nan(), "`{expr}` sample {i}: expected NaN, got {g}");
        } else {
            assert!((g - e).abs() <= EPS, "`{expr}` sample {i}: expected {e}, got {g}");
        }
    }
    assert!((out.sample_rate_hz - expected_rate).abs() <= EPS, "rate mismatch for `{expr}`");
}

fn eval_samples(expr: &str, c: &Ctx) -> (Vec<f64>, f64) {
    let out = evaluate(expr, c, &no_laps()).unwrap();
    (out.samples, out.sample_rate_hz)
}

fn assert_close(expr: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "length mismatch for `{expr}`");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() <= 1e-9, "`{expr}` sample {i}: expected {e}, got {g}");
    }
}

// A 64-sample sine, long enough for real sosfiltfilt / fft / declip.
fn sine(n: usize, fs: f64, hz: f64) -> Vec<f64> {
    (0..n).map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin()).collect()
}

// ---- Value-domain cases (literal vectors from the Dart suite) ----

#[test]
fn parity_numeric_literal_is_one_sample_rate_zero() {
    // Dart test 1: '3.14' → [3.14] @ rate 0.
    let c = ctx(&[]);
    assert_samples("3.14", &c, &[3.14], 0.0);
}

#[test]
fn parity_channel_reference_returns_samples_and_rate() {
    // Dart test 2.
    let c = ctx(&[("Speed", vec![1.0, 2.0, 3.0], 100.0)]);
    assert_samples("[Speed]", &c, &[1.0, 2.0, 3.0], 100.0);
}

#[test]
fn parity_channel_plus_scalar() {
    // Dart test 3.
    let c = ctx(&[("A", vec![1.0, 2.0, 3.0], 100.0)]);
    assert_samples("[A] + 10.0", &c, &[11.0, 12.0, 13.0], 100.0);
}

#[test]
fn parity_channel_plus_channel() {
    // Dart test 4.
    let c = ctx(&[("A", vec![1.0, 2.0, 3.0], 100.0), ("B", vec![4.0, 5.0, 6.0], 100.0)]);
    assert_samples("[A] + [B]", &c, &[5.0, 7.0, 9.0], 100.0);
}

#[test]
fn parity_mismatched_rates_errors() {
    // Dart test 5: throws MathChannelEvaluationException → Runtime kind.
    let c = ctx(&[("A", vec![1.0, 2.0], 100.0), ("B", vec![3.0, 4.0], 200.0)]);
    let err = evaluate("[A] + [B]", &c, &no_laps()).unwrap_err();
    assert_eq!(err.kind, MathEvalErrorKind::Runtime);
}

#[test]
fn parity_differentiate_finite_differences() {
    // Dart test 10: [0,1,3,6] @1Hz → [0,1,2,3].
    let c = ctx(&[("Pos", vec![0.0, 1.0, 3.0, 6.0], 1.0)]);
    assert_samples("differentiate([Pos])", &c, &[0.0, 1.0, 2.0, 3.0], 1.0);
}

#[test]
fn parity_unknown_channel_errors_with_name() {
    // Dart test 12.
    let c = ctx(&[]);
    let err = evaluate("[NoSuchChannel]", &c, &no_laps()).unwrap_err();
    assert_eq!(err.kind, MathEvalErrorKind::UnknownChannel);
    assert!(err.message.contains("[NoSuchChannel]"));
}

#[test]
fn parity_unknown_function_errors() {
    // Dart test 13: message contains 'unknown function "badFunc"'.
    let c = ctx(&[]);
    let err = evaluate("badFunc(1.0)", &c, &no_laps()).unwrap_err();
    assert_eq!(err.kind, MathEvalErrorKind::UnknownFunction);
    assert!(err.message.contains("unknown function \"badFunc\""));
}

#[test]
fn parity_if_branch_selection() {
    // Dart test 14: cond=0 picks F, cond=1 picks T.
    let c = ctx(&[
        ("Cond", vec![0.0, 1.0, 0.0, 1.0], 100.0),
        ("T", vec![10.0, 20.0, 30.0, 40.0], 100.0),
        ("F", vec![1.0, 2.0, 3.0, 4.0], 100.0),
    ]);
    assert_samples("if([Cond], [T], [F])", &c, &[1.0, 20.0, 3.0, 40.0], 100.0);
}

#[test]
fn parity_comparison_greater_than() {
    // Dart test 15.
    let c = ctx(&[("X", vec![1.0, 5.0, 3.0, 8.0], 100.0)]);
    assert_samples("[X] > 4.0", &c, &[0.0, 1.0, 0.0, 1.0], 100.0);
}

#[test]
fn parity_channel_name_with_space_resolves_verbatim() {
    // Dart test: '[New channel]'.
    let c = ctx(&[("New channel", vec![1.0, 2.0, 3.0], 100.0)]);
    assert_samples("[New channel]", &c, &[1.0, 2.0, 3.0], 100.0);
}

#[test]
fn parity_subtract_channels_name_with_space_and_digit_segment() {
    // Dart test: '[Declipped 1_AccelX]-[IMU1_AccelX]' → [9,18,27].
    let c = ctx(&[
        ("Declipped 1_AccelX", vec![10.0, 20.0, 30.0], 100.0),
        ("IMU1_AccelX", vec![1.0, 2.0, 3.0], 100.0),
    ]);
    assert_samples("[Declipped 1_AccelX]-[IMU1_AccelX]", &c, &[9.0, 18.0, 27.0], 100.0);
}

// ---- DSP delegation parity (eval output == direct core call) ----

#[test]
fn parity_integrate_delegates_to_core() {
    // Dart test 6 (delegation). Real integration over a short ramp.
    let samples = vec![0.5, 1.0, 1.5, 2.0, 2.5];
    let c = ctx(&[("Accel", samples.clone(), 800.0)]);
    let (got, rate) = eval_samples("integrate([Accel])", &c);
    let expected = crate::integration::integrate(&samples, 800.0);
    assert_eq!(rate, 800.0);
    assert_close("integrate", &got, &expected);
}

#[test]
fn parity_butter_high_delegates_to_highpass() {
    // Dart test 7 (delegation). 64-sample sine, order 2, cutoff 0.3, high.
    let samples = sine(64, 400.0, 10.0);
    let c = ctx(&[("Accel", samples.clone(), 400.0)]);
    let (got, rate) = eval_samples("butter(2, 0.3, \"high\", [Accel])", &c);
    let expected = crate::filters::highpass(&samples, 2, 0.3, 400.0);
    assert_eq!(rate, 400.0);
    assert_close("butter high", &got, &expected);
}

#[test]
fn parity_butter_low_delegates_to_lowpass() {
    // Dart test 8 (delegation). 64-sample sine, order 4, cutoff 10, low.
    let samples = sine(64, 400.0, 10.0);
    let c = ctx(&[("Accel", samples.clone(), 400.0)]);
    let (got, _rate) = eval_samples("butter(4, 10.0, \"low\", [Accel])", &c);
    let expected = crate::filters::lowpass(&samples, 4, 10.0, 400.0);
    assert_close("butter low", &got, &expected);
}

#[test]
fn parity_fft_hann_delegates_to_core() {
    // Dart test 9 (delegation). Hann window FFT.
    let samples = sine(64, 64.0, 8.0);
    let c = ctx(&[("Sig", samples.clone(), 64.0)]);
    let (got, _rate) = eval_samples("fft([Sig], \"hann\")", &c);
    let expected = crate::fft::fft(&samples, crate::fft::FftWindow::Hann);
    assert_close("fft hann", &got, &expected);
}

#[test]
fn parity_declip_delegates_to_core() {
    // Dart declip test (delegation). A clipped plateau in a 64-sample window.
    let mut samples = sine(64, 1000.0, 5.0).into_iter().map(|x| x * 20.0).collect::<Vec<_>>();
    for s in samples.iter_mut().skip(20).take(6) {
        *s = 32.0; // rail-pinned plateau
    }
    let c = ctx(&[("IMU1_AccelZ", samples.clone(), 1000.0)]);
    let (got, rate) = eval_samples("declip([IMU1_AccelZ])", &c);
    let expected = crate::clip_reconstruct::declip(&samples, 1000.0);
    assert_eq!(rate, 1000.0);
    assert_close("declip", &got, &expected);
}

#[test]
fn parity_nested_integrate_of_highpass_applies_inner_first() {
    // Dart test 11: integrate(butter high) — highpass first, then integrate.
    let samples = sine(64, 200.0, 10.0);
    let c = ctx(&[("Accel", samples.clone(), 200.0)]);
    let (got, _rate) = eval_samples("integrate(butter(2, 0.2, \"high\", [Accel]))", &c);
    let hp = crate::filters::highpass(&samples, 2, 0.2, 200.0);
    let expected = crate::integration::integrate(&hp, 200.0);
    assert_close("integrate(highpass)", &got, &expected);
}

// ---- Variance (domain-derived: the Dart evaluator suite has no variance
// vectors; the kernel is proven in variance.rs, this proves the assembly). ----

// A straight-east lap at 1 Hz, channel value = sample index. Used for both main
// and overlay so the identity case must produce ~0 inside the window.
fn straight_east_lap() -> Ctx {
    ctx(&[
        ("GPS_Latitude", vec![0.0; 10], 1.0),
        ("GPS_Longitude", (0..10).map(|i| i as f64 * 0.001).collect(), 1.0),
        ("GPS_EpochMs", (0..10).map(|i| (i * 1000) as f64).collect(), 1.0),
        ("LapTime", (0..10).map(|i| i as f64).collect(), 1.0),
    ])
}

fn identity_overlay_ctx(main_lap_number: Option<u32>, window: (f64, f64)) -> MathLapContext {
    MathLapContext {
        main_lap_bounds: vec![window],
        main_sectors: Vec::new(),
        main_lap_number,
        overlay: Some(MathOverlay {
            lookup: Arc::new(straight_east_lap()),
            lap_start_ms: 0.0,
            lap_end_ms: 9000.0,
            lap_start_uniform_sec: 0.0,
        }),
        baseline_row: None,
    }
}

#[test]
fn parity_variance_time_identity_is_zero_in_window() {
    // Arrange — main == overlay, full window so every sample is in-lap.
    let main = straight_east_lap();
    let ctx = identity_overlay_ctx(Some(1), (0.0, 9.0));

    // Act
    let out = evaluate("variance_time([LapTime])", &main, &ctx).unwrap();

    // Assert — diff ≈ 0 on the in-window (non-NaN) samples.
    let inside: Vec<f64> = out.samples.iter().copied().filter(|x| !x.is_nan()).collect();
    assert!(!inside.is_empty());
    for x in inside {
        assert!(x.abs() < 1e-3, "expected ~0, got {x}");
    }
}

#[test]
fn parity_variance_time_nan_outside_main_lap_window() {
    // Arrange — gate to [3,7); samples outside must be NaN.
    let main = straight_east_lap();
    let ctx = identity_overlay_ctx(Some(1), (3.0, 7.0));

    // Act
    let out = evaluate("variance_time([LapTime])", &main, &ctx).unwrap();

    // Assert — indices 0..2 and 7..9 are NaN (end exclusive).
    for (i, v) in out.samples.iter().enumerate() {
        if i < 3 || i >= 7 {
            assert!(v.is_nan(), "expected NaN at i={i}, got {v}");
        } else {
            assert!(v.abs() < 1e-3, "expected ~0 at i={i}, got {v}");
        }
    }
}

#[test]
fn parity_variance_dist_identity_is_zero_in_window() {
    // Arrange
    let main = straight_east_lap();
    let ctx = identity_overlay_ctx(Some(1), (0.0, 9.0));

    // Act
    let out = evaluate("variance_dist([LapTime])", &main, &ctx).unwrap();

    // Assert — identical arc-length polylines → diff ≈ 0 in-window.
    let inside: Vec<f64> = out.samples.iter().copied().filter(|x| !x.is_nan()).collect();
    assert!(!inside.is_empty());
    for x in inside {
        assert!(x.abs() < 1e-3, "expected ~0, got {x}");
    }
}
