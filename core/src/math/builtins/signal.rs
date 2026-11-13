//! Filters, spectra, time-domain operators, resampling and the estimator
//! family: the builtins that step through a channel's time axis.

use std::sync::Arc;

use crate::math::eval::{channel, channel_window, describe_channel, err, estimator_channel_id, kwarg_string, parse_fft_averaging, parse_fft_detrend, parse_fft_scaling, parse_fft_window, require_channel, require_scalar, require_string, resample_onto, timed, uniform_rate};
use crate::math::registry::Ctx;
use crate::math::value::{ChannelValue, Spacing, Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// `periodogram(ch, window=, detrend=, scaling=)`: one spectrum of the
/// selected window's samples, scipy-named and scipy-scaled. `fft` is retired
/// (R146/R151 item 1): it was one un-normalised windowed magnitude spectrum,
/// a false friend beside the charts' `welch()`; a `version: 3` call migrates
/// in memory to `periodogram(ch, window=.., scaling="raw_magnitude")` (plan
/// §3.3), so no existing workbook's numbers move.
pub(super) fn periodogram(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
    let (name, kwargs) = (ctx.name(), ctx.kwargs());
    let window = parse_fft_window(kwarg_string(name, kwargs, "window", "boxcar")?, name)?;
    let detrend = parse_fft_detrend(kwarg_string(name, kwargs, "detrend", "constant")?, name)?;
    let scaling = parse_fft_scaling(kwarg_string(name, kwargs, "scaling", "density")?, name)?;
    // Single-spectrum reduction (R123/R124): the selected window's
    // samples only, so a lap's spectrum is the lap's spectrum, not
    // the session's. `nperseg: 0` / `noverlap: 0` resolve to one
    // full-record segment (`resolve_seg`), reproducing scipy's
    // `periodogram` exactly; `averaging` is irrelevant with exactly
    // one segment.
    let (start, end) = channel_window(ctx.lap(), &ch);
    let result = crate::fft::welch(
        ch.samples[start..end].to_vec(),
        uniform_rate(&ch, name)?,
        window,
        0,
        0,
        detrend,
        crate::fft::Averaging::Mean,
        scaling,
    );
    // Output is n/2+1 bins; preserve the original rate so the caller
    // can compute freq[k] = k * sample_rate_hz / n. Bins are not
    // per-sample time — no t_us axis applies (empty, not ch's, per
    // L3-R12).
    Ok(channel(result.values, ch.sample_rate_hz, Arc::from(&[] as &[i64])))
}

/// `welch(ch, window=, nperseg=, noverlap=, detrend=, average=, scaling=)`:
/// the segmented, averaged spectrum the charts compute, scipy's defaults.
pub(super) fn welch(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
    let (name, kwargs) = (ctx.name(), ctx.kwargs());
    let (start, end) = channel_window(ctx.lap(), &ch);
    let n = end - start;
    let window = parse_fft_window(kwarg_string(name, kwargs, "window", "hann")?, name)?;
    // scipy's own `welch` defaults: `nperseg` is 256 samples (or the
    // whole record if shorter), `noverlap` is half of `nperseg`.
    let nperseg = match kwargs.get("nperseg") {
        Some(v) => require_scalar(v, name)?.round().max(0.0) as usize,
        None => n.min(256),
    };
    let noverlap = match kwargs.get("noverlap") {
        Some(v) => require_scalar(v, name)?.round().max(0.0) as usize,
        None => nperseg / 2,
    };
    let detrend = parse_fft_detrend(kwarg_string(name, kwargs, "detrend", "constant")?, name)?;
    let average = parse_fft_averaging(kwarg_string(name, kwargs, "average", "mean")?, name)?;
    let scaling = parse_fft_scaling(kwarg_string(name, kwargs, "scaling", "density")?, name)?;
    let result = crate::fft::welch(
        ch.samples[start..end].to_vec(),
        uniform_rate(&ch, name)?,
        window,
        nperseg,
        noverlap,
        detrend,
        average,
        scaling,
    );
    Ok(channel(result.values, ch.sample_rate_hz, Arc::from(&[] as &[i64])))
}

/// `declip(ch)`: samples lost to clipping reconstructed, in the channel's unit.
pub(super) fn declip(ch: ChannelValue) -> Result<Value, MathEvalError> {
    Ok(channel(
        crate::clip_reconstruct::declip(&ch.samples, uniform_rate(&ch, "declip")?),
        ch.sample_rate_hz,
        ch.t_us,
    ))
}

/// `differentiate(ch)`: the causal-difference time derivative, in the
/// channel's unit per second.
pub(super) fn differentiate(ch: ChannelValue) -> Result<Value, MathEvalError> {
    let derivative = match timed(&ch, "differentiate")? {
        Spacing::Uniform(rate) => crate::statistics::differentiate(&ch.samples, rate),
        Spacing::Recorded(t_us) => crate::statistics::differentiate_at_times(&ch.samples, t_us),
        Spacing::Untimed => unreachable!("timed() refuses an untimed channel"),
    };
    Ok(channel(
        derivative,
        ch.sample_rate_hz,
        ch.t_us,
    ))
}

/// `gradient(ch)`: numpy.gradient's own central-difference formula (R151
/// item 3), in the channel's unit per second — `differentiate` keeps its
/// distinct, causal-difference meaning rather than being renamed to this.
pub(super) fn gradient(ch: ChannelValue) -> Result<Value, MathEvalError> {
    let derivative = match timed(&ch, "gradient")? {
        Spacing::Uniform(rate) => crate::statistics::gradient(&ch.samples, rate),
        Spacing::Recorded(t_us) => crate::statistics::gradient_at_times(&ch.samples, t_us),
        Spacing::Untimed => unreachable!("timed() refuses an untimed channel"),
    };
    Ok(channel(
        derivative,
        ch.sample_rate_hz,
        ch.t_us,
    ))
}

/// `detrend(ch)` / `detrend(ch, mode)`: global least-squares trend removal
/// over the sample index (NOT [Time]); linear by default. The output
/// inherits the input rate and length.
pub(super) fn detrend(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    if args.is_empty() || args.len() > 2 {
        return Err(err(
            MathEvalErrorKind::ArgCount,
            format!("{name}: expected detrend(ch) or detrend(ch, mode), got {} argument(s)", args.len()),
        ));
    }
    let ch = require_channel(&args[0], name)?;
    let mode = if args.len() == 2 {
        match require_string(&args[1], name)? {
            "linear" => crate::statistics::DetrendMode::Linear,
            // "mean" is the spectral-detrend vocab for constant removal.
            "constant" | "mean" => crate::statistics::DetrendMode::Constant,
            "none" => crate::statistics::DetrendMode::None,
            other => {
                return Err(err(
                    MathEvalErrorKind::Runtime,
                    format!(
                        "detrend: unknown mode \"{other}\"; expected \"linear\", \"constant\", or \"none\""
                    ),
                ))
            }
        }
    } else {
        crate::statistics::DetrendMode::Linear
    };
    Ok(channel(crate::statistics::detrend(&ch.samples, mode), ch.sample_rate_hz, ch.t_us))
}

/// Zero-phase Butterworth filter of `ch`: `order` (rounded to an integer),
/// `cutoff_hz` in Hz, `kind` low/lowpass or high/highpass.
pub(super) fn butter(order: f64, cutoff: f64, ftype: String, ch: ChannelValue) -> Result<Value, MathEvalError> {
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
pub(super) fn cumulative_trapezoid(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
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
pub(super) fn estimator(ctx: &Ctx, arg: String) -> Result<Value, MathEvalError> {
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
pub(super) fn resample(x: Value, onto: Value) -> Result<Value, MathEvalError> {
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
pub(super) fn deglitch_pulses(ctx: &Ctx, ch: ChannelValue) -> Result<Value, MathEvalError> {
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
