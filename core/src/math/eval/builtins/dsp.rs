//! Channel DSP builtins: integration, differentiation, filters, spectra,
//! detrend, declip, resampling, pulse deglitching, and the committed-but-deferred
//! stubs. Arm bodies moved verbatim from `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

// `cumtrapz` is a permanent second spelling, not a deprecated one —
// scipy itself carries both (`scipy.integrate.cumulative_trapezoid`
// and the older `cumtrapz`), so matching that costs nothing (R151
// item 6). Retired from `integrate` (R143): gratuitous rename, no
// behaviour change.
/// The `cumulative_trapezoid` / `cumtrapz` builtin: one evaluator for every spelling, which reads `call.name` for its messages.
pub(crate) fn cumulative_trapezoid(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    let ch = require_channel(&args[0], name)?;
    let integrated = match timed(&ch, name)? {
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

/// The `butter` builtin.
pub(crate) fn butter(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 4)?;
    let order = require_scalar(&args[0], name)?;
    let cutoff = require_scalar(&args[1], name)?;
    let ftype = require_string(&args[2], name)?;
    let ch = require_channel(&args[3], name)?;
    let order = order.round() as usize;
    match ftype {
        "high" | "highpass" => {
            let out = crate::filters::highpass(&ch.samples, order, cutoff, uniform_rate(&ch, name)?)
                .map_err(|e| err(MathEvalErrorKind::Runtime, e.to_string()))?;
            Ok(channel(out, ch.sample_rate_hz, ch.t_us))
        }
        "low" | "lowpass" => {
            let out = crate::filters::lowpass(&ch.samples, order, cutoff, uniform_rate(&ch, name)?)
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

// `fft` is retired (R146/R151 item 1, plan §1 row 3): it was never a
// true FFT surface, it was one un-normalised windowed magnitude
// spectrum with no segmentation or averaging — a false friend on
// top of a real inconsistency with the charts' `welch()`. Split
// into `periodogram` (this function's single-segment shape,
// scipy-named and scipy-scaled) and `welch` (segmented/averaged,
// what the charts already compute). A `version: 3` call migrates in
// memory to `periodogram(ch, window=.., scaling="raw_magnitude")`
// before it ever reaches here (plan §3.3) — no existing workbook's
// numbers move. `fft` itself is reserved for a future true complex
// DFT and dispatches nowhere below, so it now falls to the
// catch-all's [`unknown_function_error`], which names its
// replacement via [`crate::math::math_name_migrations`].
/// The `periodogram` builtin.
pub(crate) fn periodogram(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, kwargs, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    reject_unknown_kwargs(name, kwargs, &["window", "detrend", "scaling"])?;
    let ch = require_channel(&args[0], name)?;
    let window = parse_fft_window(kwarg_string(name, kwargs, "window", "boxcar")?, name)?;
    let detrend = parse_fft_detrend(kwarg_string(name, kwargs, "detrend", "constant")?, name)?;
    let scaling = parse_fft_scaling(kwarg_string(name, kwargs, "scaling", "density")?, name)?;
    // Single-spectrum reduction (R123/R124): the selected window's
    // samples only, so a lap's spectrum is the lap's spectrum, not
    // the session's. `nperseg: 0` / `noverlap: 0` resolve to one
    // full-record segment (`resolve_seg`), reproducing scipy's
    // `periodogram` exactly; `averaging` is irrelevant with exactly
    // one segment.
    let (start, end) = channel_window(lap_ctx, &ch);
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

/// The `welch` builtin.
pub(crate) fn welch(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, kwargs, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    reject_unknown_kwargs(name, kwargs, &["window", "nperseg", "noverlap", "detrend", "average", "scaling"])?;
    let ch = require_channel(&args[0], name)?;
    let (start, end) = channel_window(lap_ctx, &ch);
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

/// The `declip` builtin.
pub(crate) fn declip(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    let ch = require_channel(&args[0], name)?;
    Ok(channel(
        crate::clip_reconstruct::declip(&ch.samples, uniform_rate(&ch, name)?),
        ch.sample_rate_hz,
        ch.t_us,
    ))
}

/// The `differentiate` builtin.
pub(crate) fn differentiate(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    let ch = require_channel(&args[0], name)?;
    let derivative = match timed(&ch, name)? {
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

// numpy.gradient's own central-difference formula (R151 item 3) —
// `differentiate` keeps its distinct, causal-difference meaning
// rather than being renamed to this name (see `differentiate`'s own
// doc comment for why changing the implementation under the name
// would have been the wrong fix).
/// The `gradient` builtin.
pub(crate) fn gradient(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    let ch = require_channel(&args[0], name)?;
    let derivative = match timed(&ch, name)? {
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

/// The `detrend` builtin.
pub(crate) fn detrend(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    // Global least-squares trend removal over the sample index (NOT
    // [Time]). 1-arg → linear (default); 2-arg → explicit mode string.
    // The mode string is validated here, mirroring butter's rejection
    // of a bad direction arg. Output inherits the input rate + length.
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

// R242: the explicit, and only, way two channels recorded on different
// clocks come together. Never implicit — C2 §3.6.2 rule 4 ("never an
// implicit resample") is the contract this satisfies rather than
// weakens.
/// The `resample` builtin.
pub(crate) fn resample(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    if let Value::Scalar(n) = &args[1] {
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
    let x = require_channel(&args[0], "resample(x, onto) — x")?;
    let onto = require_channel(&args[1], "resample(x, onto) — onto")?;
    resample_onto(&x, &onto)
}

// A wheel-pulse event channel with split gaps merged back (2026-11-05,
// `crate::pulse`): the kept pulses at their own recorded times, the
// running count renumbered so `differentiate` gives 1 / gap.
/// The `deglitch_pulses` builtin.
pub(crate) fn deglitch_pulses(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, kwargs, .. } = call;
    require_arg_count(name, &args, 1)?;
    reject_unknown_kwargs(name, kwargs, &["max_fraction"])?;
    let ch = require_channel(&args[0], name)?;
    let Spacing::Recorded(t_us) = ch.spacing() else {
        return Err(err(
            MathEvalErrorKind::Runtime,
            format!("{name} takes a pulse event channel (samples at recorded times), and {} is not one", describe_channel(&ch)),
        ));
    };
    let max_fraction = match kwargs.get("max_fraction") {
        Some(v) => require_scalar(v, name)?,
        None => crate::pulse::DEFAULT_MAX_FRACTION,
    };
    let kept = crate::pulse::deglitch(t_us, max_fraction);
    let first = ch.samples.first().copied().unwrap_or(0.0);
    let samples: Vec<f64> = (0..kept.len()).map(|k| first + k as f64).collect();
    let times: Arc<[i64]> = kept.iter().map(|&i| t_us[i]).collect();
    Ok(channel(samples, 0.0, times))
}

/// The `spectrogram` / `envelope` / `correlate` / `convolve` / `sosfilt` builtin: one evaluator for every spelling, which reads `call.name` for its messages.
pub(crate) fn not_implemented(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, .. } = call;
    Err(err(MathEvalErrorKind::NotImplemented, format!("not yet implemented: {name}")))
}
