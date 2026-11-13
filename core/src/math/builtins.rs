//! idl-rs's builtins converted to the [`Registry`] (prototype of spec
//! `2026-11-13-idl-rs-builtin-registry-v2.md`): each one a typed function and
//! its hand-written [`BuiltinMeta`]. The bodies are the old `call_function`
//! arms minus their argument unpacking.

use std::sync::Arc;

use crate::math::eval::{channel, describe_channel, err, estimator_channel_id, require_channel, require_scalar, resample_onto, timed, uniform_rate};
use crate::math::registry::{BuiltinMeta, Ctx, Registry};
use crate::math::value::{ChannelValue, Spacing, Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// Registers the converted builtins on `r`.
pub(super) fn register(r: &mut Registry) {
    r.add(BuiltinMeta { name: "butter", category: "Filter", unit_rule: "SameAsArg(3)", shape: "[t]", kwargs: &[],
        signature: "butter(order, cutoff_hz, \"low\"|\"lowpass\"|\"high\"|\"highpass\", ch)",
        description: "Zero-phase Butterworth low- or high-pass filter; designs and applies the filter in one call. Needs a fixed-rate channel: put an event channel on a grid first, `resample(x, [Time])`.",
        example: "butter(2, 5, \"low\", [Fork])" }, butter);
    r.add(BuiltinMeta { name: "cumulative_trapezoid", category: "Time-domain", unit_rule: "Product(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "cumulative_trapezoid(ch)",
        description: "Cumulative trapezoidal integral of a channel with respect to time. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.",
        example: "cumulative_trapezoid([AccelZ])" }, cumulative_trapezoid);
    r.add(BuiltinMeta { name: "cumtrapz", category: "Time-domain", unit_rule: "Product(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "cumtrapz(ch)",
        description: "A permanent second spelling of `cumulative_trapezoid`, as scipy itself carries both.",
        example: "cumtrapz([AccelZ])" }, cumulative_trapezoid);
    r.add(BuiltinMeta { name: "resample", category: "Resampling", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "resample(x, onto)",
        description: "Linear interpolation of `x` onto the per-sample recorded times of `onto`; the result carries `onto`'s time axis, rate and length, so the two can then be combined. Times outside `x`'s span are `NaN` -- never extrapolated. The explicit way to combine two channels recorded on different clocks; nothing resamples implicitly.",
        example: "resample([IMU2_AccelZ], [IMU1_AccelZ])" }, resample);
    r.add(BuiltinMeta { name: "deglitch_pulses", category: "Resampling", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &["max_fraction"],
        signature: "deglitch_pulses(pulses, max_fraction=0.35)",
        description: "A wheel-pulse event channel without the extra edges a marginal sensor adds: a pulse is dropped when the gap ending at it is under `max_fraction` of both the gap before and the gap after, so it reads as an extra edge inside one ordinary gap. Comparing both sides means the first pulses after a locked wheel spins up, and the growing gaps of hard braking, are never dropped. The result is renumbered 1, 2, 3, ... from the first kept pulse, at the kept pulses' recorded times.",
        example: "deglitch_pulses([WheelFront])" }, deglitch_pulses);
    r.add(BuiltinMeta { name: "attitude", category: "Estimator (diagnostic)", unit_rule: "Fixed(deg)", shape: "[t]", kwargs: &[],
        signature: "attitude(\"roll\"|\"pitch\")",
        description: "Bike attitude estimated by the AHRS filter, in degrees.",
        example: "attitude(\"roll\")" }, estimator);
    r.add(BuiltinMeta { name: "body_accel", category: "Estimator (diagnostic)", unit_rule: "Fixed(g)", shape: "[t]", kwargs: &[],
        signature: "body_accel(\"long\"|\"lat\")",
        description: "Gravity-compensated body-frame acceleration, in g.",
        example: "body_accel(\"long\")" }, estimator);
    r.add(BuiltinMeta { name: "wheel_travel", category: "Estimator", unit_rule: "Fixed(mm)", shape: "[t]", kwargs: &[],
        signature: "wheel_travel(\"front\"|\"rear\")",
        description: "Suspension travel at the named wheel, in mm.",
        example: "wheel_travel(\"front\")" }, estimator);
    r.add(BuiltinMeta { name: "wheel_velocity", category: "Estimator", unit_rule: "Fixed(mm/s)", shape: "[t]", kwargs: &[],
        signature: "wheel_velocity(\"front\"|\"rear\")",
        description: "Suspension velocity at the named wheel, in mm/s.",
        example: "wheel_velocity(\"front\")" }, estimator);
}

/// Zero-phase Butterworth filter of `ch`: `order` (rounded to an integer),
/// `cutoff_hz` in Hz, `kind` low/lowpass or high/highpass.
fn butter(order: f64, cutoff: f64, ftype: String, ch: ChannelValue) -> Result<Value, MathEvalError> {
    let order = order.round() as usize;
    match ftype.as_str() {
        "high" | "highpass" => {
            let out = crate::filters::highpass(&ch.samples, order, cutoff, uniform_rate(&ch, "butter")?)
                .map_err(|e| err(MathEvalErrorKind::Runtime, e.to_string()))?;
            Ok(channel(out, ch.sample_rate_hz, ch.t_us))
        }
        "low" | "lowpass" => {
            let out = crate::filters::lowpass(&ch.samples, order, cutoff, uniform_rate(&ch, "butter")?)
                .map_err(|e| err(MathEvalErrorKind::Runtime, e.to_string()))?;
            Ok(channel(out, ch.sample_rate_hz, ch.t_us))
        }
        "band" => Err(err(
            MathEvalErrorKind::Runtime,
            "butter: band-pass filter not yet implemented",
        )),
        other => Err(err(
            MathEvalErrorKind::Runtime,
            format!("butter: unknown type \"{other}\"; expected \"low\" or \"high\""),
        )),
    }
}

/// `cumulative_trapezoid` and its permanent second spelling `cumtrapz` (scipy
/// carries both, R151 item 6): the running integral of `ch` over time, in the
/// channel's unit times seconds.
fn cumulative_trapezoid(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
    let integrated = match timed(&ch, ctx.name())? {
        Spacing::Uniform(rate) => crate::integration::integrate(&ch.samples, rate),
        Spacing::Recorded(t_us) => crate::integration::integrate_at_times(&ch.samples, t_us),
        Spacing::Untimed => unreachable!("timed() refuses an untimed channel"),
    };
    Ok(channel(
        integrated,
        ch.sample_rate_hz,
        ch.t_us,
    ))
}

/// The estimator-backed virtual sensors (`wheel_travel`, `wheel_velocity`,
/// `attitude`, `body_accel`): store reads of the offline geometry-constrained
/// estimator, which the lookup runs once per session and caches.
fn estimator(ctx: &Ctx, arg: String) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    let channel_id = estimator_channel_id(name, &arg)?;
    let ch = ctx.lookup().estimator_channel(channel_id).ok_or_else(|| {
        err(
            MathEvalErrorKind::Runtime,
            format!(
                "{name}(\"{arg}\"): the suspension/attitude estimator could not run for \
                 this session — it needs IMU0 accel and gyro channels"
            ),
        )
    })?;
    Ok(channel(ch.samples.to_vec(), ch.sample_rate_hz, ch.t_us))
}

/// `x` linearly interpolated onto `onto`'s recorded times (R242): the
/// explicit, and only, way two channels recorded on different clocks come
/// together (C2 §3.6.2 rule 4, "never an implicit resample").
fn resample(x: Value, onto: Value) -> Result<Value, MathEvalError> {
    if let Value::Scalar(n) = &onto {
        // `resample(ch, num)` — the scipy-style target-sample-count
        // form C2 §3.3 reserved until R242. Named explicitly so an
        // author who wrote it gets the reason, not "expected channel".
        return Err(err(
            MathEvalErrorKind::NotImplemented,
            format!(
                "resample(x, onto): the second argument is the channel whose time axis \
                 to resample onto, not a sample count (got {n}). The count form \
                 (`scipy.signal.resample`) is not implemented."
            ),
        ));
    }
    let x = require_channel(&x, "resample(x, onto) — x")?;
    let onto = require_channel(&onto, "resample(x, onto) — onto")?;
    resample_onto(&x, &onto)
}

/// A wheel-pulse event channel with split gaps merged back (2026-11-05,
/// `crate::pulse`): the kept pulses at their own recorded times, the running
/// count renumbered so `differentiate` gives 1 / gap. `max_fraction=`
/// (dimensionless) defaults to [`crate::pulse::DEFAULT_MAX_FRACTION`].
fn deglitch_pulses(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    let Spacing::Recorded(t_us) = ch.spacing() else {
        return Err(err(
            MathEvalErrorKind::Runtime,
            format!("{name} takes a pulse event channel (samples at recorded times), and {} is not one", describe_channel(&ch)),
        ));
    };
    let max_fraction = match ctx.kwarg("max_fraction") {
        Some(v) => require_scalar(v, name)?,
        None => crate::pulse::DEFAULT_MAX_FRACTION,
    };
    let kept = crate::pulse::deglitch(t_us, max_fraction);
    let first = ch.samples.first().copied().unwrap_or(0.0);
    let samples: Vec<f64> = (0..kept.len()).map(|k| first + k as f64).collect();
    let times: Arc<[i64]> = kept.iter().map(|&i| t_us[i]).collect();
    Ok(channel(samples, 0.0, times))
}
