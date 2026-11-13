//! Scalar aggregates and their colliding rolling/elementwise forms: one
//! argument reduces the selected window to a scalar (R123/R124), two take
//! the other call form. Their bodies count the arguments, because the arity
//! errors name the call forms as they always have.

use crate::math::eval::{channel, channel_window, elemwise, err, one_channel, one_channel_windowed, require_arg_count, require_channel, require_scalar, take_kwarg};
use crate::math::registry::Ctx;
use crate::math::value::Value;
use crate::math::{MathEvalError, MathEvalErrorKind};

/// A one-channel scalar reduction over the selected window (`sum`, `count`,
/// `first`, `last`), in the unit `f` gives.
pub(super) fn reduce(f: fn(&[f64]) -> f64) -> impl Fn(&Ctx<'_>, &[Value]) -> Result<Value, MathEvalError> + Send + Sync + 'static {
    move |ctx: &Ctx<'_>, args: &[Value]| {
        let (samples, s, e) = one_channel_windowed(args, ctx.name(), ctx.lap())?;
        Ok(Value::Scalar(f(&samples[s..e])))
    }
}

/// The shared shape of `rms`/`std`: one argument → `aggregate` over the
/// window; two → `rolling` over a `w`-sample window (unaffected by the lap
/// window — a different "window").
fn aggregate_or_rolling(
    ctx: &Ctx,
    args: &[Value],
    aggregate: fn(&[f64]) -> f64,
    rolling: fn(&[f64], usize) -> Vec<f64>,
) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(args, name, ctx.lap())?;
        Ok(Value::Scalar(aggregate(&samples[s..e])))
    } else {
        require_arg_count(name, args, 2)?;
        let ch = require_channel(&args[0], name)?;
        let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
        Ok(channel(rolling(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
    }
}

/// `rms(ch)` / `rms(ch, w)`, in the channel's unit.
pub(super) fn rms(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    aggregate_or_rolling(ctx, args, crate::math::aggregate::rms, crate::statistics::rolling_rms)
}

/// `std(ch)` / `std(ch, w)`: population σ, in the channel's unit.
pub(super) fn std(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    aggregate_or_rolling(ctx, args, crate::math::aggregate::std_pop, crate::statistics::rolling_std)
}

/// `mean(ch)` / `mean(ch, w)` / `mean(ch, window=w)`: the keyword (C2 §3.2,
/// R143 item 1) is a non-overloaded spelling of the rolling form, additive
/// to the positional one.
pub(super) fn mean(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    let window_kw = take_kwarg(name, "window", args.len() >= 2, ctx.kwargs())?;
    match (args.len(), window_kw) {
        (1, None) => {
            let (samples, s, e) = one_channel_windowed(args, "mean", ctx.lap())?;
            Ok(Value::Scalar(crate::math::aggregate::mean(&samples[s..e])))
        }
        (1, Some(w_val)) => {
            let ch = require_channel(&args[0], name)?;
            let w = require_scalar(w_val, name)?.round().max(0.0) as usize;
            Ok(channel(crate::statistics::rolling_mean(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
        }
        _ => {
            require_arg_count(name, args, 2)?;
            let ch = require_channel(&args[0], name)?;
            let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
            Ok(channel(crate::statistics::rolling_mean(&ch.samples, w), ch.sample_rate_hz, ch.t_us))
        }
    }
}

/// The shared shape of `min`/`max`: one argument → the window's extreme;
/// two → the elementwise extreme of two operands.
fn aggregate_or_elementwise(
    ctx: &Ctx,
    args: &[Value],
    aggregate: fn(&[f64]) -> f64,
    pair: fn(f64, f64) -> f64,
) -> Result<Value, MathEvalError> {
    let name = ctx.name();
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(args, name, ctx.lap())?;
        Ok(Value::Scalar(aggregate(&samples[s..e])))
    } else {
        require_arg_count(name, args, 2)?;
        elemwise(args[0].clone(), args[1].clone(), name, |a, b| Ok(pair(a, b)))
    }
}

/// `min(ch)` / `min(a, b)`, in the operands' unit.
pub(super) fn min(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    aggregate_or_elementwise(ctx, args, crate::math::aggregate::min, f64::min)
}

/// `max(ch)` / `max(a, b)`, in the operands' unit.
pub(super) fn max(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    aggregate_or_elementwise(ctx, args, crate::math::aggregate::max, f64::max)
}

/// `median(ch)`; the 2-argument rolling median stays deferred, with the
/// exact deferred-stub message kept for parity.
pub(super) fn median(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    if args.len() == 1 {
        let (samples, s, e) = one_channel_windowed(args, "median", ctx.lap())?;
        Ok(Value::Scalar(crate::math::aggregate::median(&samples[s..e])))
    } else {
        Err(err(MathEvalErrorKind::NotImplemented, "not yet implemented: median".to_string()))
    }
}

/// `percentile(ch, q)`: the linear-interpolated percentile of the window,
/// `q` in 0..=100 (`numpy.percentile`'s name, retired from `p`, R143).
pub(super) fn percentile(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    if args.len() != 2 {
        return Err(err(MathEvalErrorKind::ArgCount, "percentile(channel, quantile) takes 2 args"));
    }
    let ch = require_channel(&args[0], "percentile")?;
    let q = require_scalar(&args[1], "percentile quantile")?;
    let (start, end) = channel_window(ctx.lap(), &ch);
    Ok(Value::Scalar(crate::math::aggregate::percentile(&ch.samples[start..end], q)))
}

/// `main({col[]})`: the Main lap's row of a whole-column reference, or NaN
/// when no Main row is set (channel math).
pub(super) fn main(ctx: &Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    let arr = one_channel(args, "main")?;
    let v = ctx.lap()
        .baseline_row
        .and_then(|r| arr.get(r).copied())
        .unwrap_or(f64::NAN);
    Ok(Value::Scalar(v))
}
