//! Windowed scalar aggregates and their rolling / elementwise second forms
//! (`rms`, `mean`, `std`, `min`, `max`, `sum`, `count`, `first`, `last`, `median`,
//! `percentile`) and the table-cell-only `main`. Arm bodies moved verbatim from
//! `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

/// The `rms` builtin.
pub(crate) fn rms(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate (windowed, R123/R124); 2-arg → rolling
    // RMS over a sample-count window (unaffected — a different
    // "window").
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(&args, "rms", lap_ctx)?;
        Ok(Value::Scalar(crate::math::aggregate::rms(&samples[s..e])))
    } else {
        require_arg_count(name, &args, 2)?;
        let ch = require_channel(&args[0], name)?;
        let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
        Ok(channel(crate::statistics::rolling_rms(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
    }
}

/// The `mean` builtin.
pub(crate) fn mean(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, kwargs, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate (windowed, R123/R124); 2-arg → rolling
    // mean over a sample-count window (unaffected). `mean(x,
    // window=w)` (C2 §3.2, R143 item 1) is a non-overloaded spelling
    // of the same rolling form, additive to the positional one.
    reject_unknown_kwargs(name, kwargs, &["window"])?;
    let window_kw = take_kwarg(name, "window", args.len() >= 2, kwargs)?;
    match (args.len(), window_kw) {
        (1, None) => {
            let (samples, s, e) = one_channel_windowed(&args, "mean", lap_ctx)?;
            Ok(Value::Scalar(crate::math::aggregate::mean(&samples[s..e])))
        }
        (1, Some(w_val)) => {
            let ch = require_channel(&args[0], name)?;
            let w = require_scalar(w_val, name)?.round().max(0.0) as usize;
            Ok(channel(crate::statistics::rolling_mean(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
        }
        _ => {
            require_arg_count(name, &args, 2)?;
            let ch = require_channel(&args[0], name)?;
            let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
            Ok(channel(crate::statistics::rolling_mean(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
        }
    }
}

/// The `std` builtin.
pub(crate) fn std_dev(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate (population σ, windowed, R123/R124);
    // 2-arg → rolling std over a sample-count window (unaffected).
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(&args, "std", lap_ctx)?;
        Ok(Value::Scalar(crate::math::aggregate::std_pop(&samples[s..e])))
    } else {
        require_arg_count(name, &args, 2)?;
        let ch = require_channel(&args[0], name)?;
        let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
        Ok(channel(crate::statistics::rolling_std(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
    }
}

/// The `min` builtin.
pub(crate) fn min(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate (column/series minimum, windowed,
    // R123/R124); 2-arg → elementwise minimum of two operands.
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(&args, "min", lap_ctx)?;
        Ok(Value::Scalar(crate::math::aggregate::min(&samples[s..e])))
    } else {
        require_arg_count(name, &args, 2)?;
        let mut it = args.into_iter();
        elemwise(it.next().unwrap(), it.next().unwrap(), "min", |a, b| Ok(a.min(b)))
    }
}

/// The `max` builtin.
pub(crate) fn max(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate (column/series maximum, windowed,
    // R123/R124); 2-arg → elementwise maximum of two operands.
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(&args, "max", lap_ctx)?;
        Ok(Value::Scalar(crate::math::aggregate::max(&samples[s..e])))
    } else {
        require_arg_count(name, &args, 2)?;
        let mut it = args.into_iter();
        elemwise(it.next().unwrap(), it.next().unwrap(), "max", |a, b| Ok(a.max(b)))
    }
}

/// The `main` builtin.
pub(crate) fn main(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    // Table-cell reference: the Main lap's value in a `{col[]}` column.
    // Reuses the whole-column array form; returns the element at the
    // table's Main row, or NaN when no Main row is set (channel math).
    let arr = one_channel(&args, "main")?;
    let v = lap_ctx
        .baseline_row
        .and_then(|r| arr.get(r).copied())
        .unwrap_or(f64::NAN);
    Ok(Value::Scalar(v))
}

/// The `sum` builtin.
pub(crate) fn sum(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    let (samples, s, e) = one_channel_windowed(&args, "sum", lap_ctx)?;
    Ok(Value::Scalar(crate::math::aggregate::sum(&samples[s..e])))
}

/// The `count` builtin.
pub(crate) fn count(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    let (samples, s, e) = one_channel_windowed(&args, "count", lap_ctx)?;
    Ok(Value::Scalar(crate::math::aggregate::count(&samples[s..e])))
}

/// The `first` builtin.
pub(crate) fn first(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    let (samples, s, e) = one_channel_windowed(&args, "first", lap_ctx)?;
    Ok(Value::Scalar(crate::math::aggregate::first(&samples[s..e])))
}

/// The `last` builtin.
pub(crate) fn last(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    let (samples, s, e) = one_channel_windowed(&args, "last", lap_ctx)?;
    Ok(Value::Scalar(crate::math::aggregate::last(&samples[s..e])))
}

/// The `median` builtin.
pub(crate) fn median(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    // 1-arg → scalar aggregate. A 2-arg rolling median remains deferred
    // (preserve the exact deferred-stub message for parity).
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(&args, "median", lap_ctx)?;
        Ok(Value::Scalar(crate::math::aggregate::median(&samples[s..e])))
    } else {
        Err(err(MathEvalErrorKind::NotImplemented, "not yet implemented: median".to_string()))
    }
}

// Retired from `p` (R143): gratuitous rename — matches
// `numpy.percentile`'s name, not just its behaviour.
/// The `percentile` builtin.
pub(crate) fn percentile(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { args, lap_ctx, .. } = call;
    // percentile(channel, quantile) → linear-interpolated percentile
    // (quantile in 0..=100), windowed per R123/R124.
    if args.len() != 2 {
        return Err(err(MathEvalErrorKind::ArgCount, "percentile(channel, quantile) takes 2 args"));
    }
    let ch = require_channel(&args[0], "percentile")?;
    let q = require_scalar(&args[1], "percentile quantile")?;
    let (start, end) = channel_window(lap_ctx, &ch);
    Ok(Value::Scalar(crate::math::aggregate::percentile(&ch.samples[start..end], q)))
}
