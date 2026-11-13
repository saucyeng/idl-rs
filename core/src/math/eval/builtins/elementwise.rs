//! Elementwise maths, trig and per-sample logic (`clip`, `where`, `keep`).
//! Arm bodies moved verbatim from `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

/// One-argument elementwise builtin: checks the single argument, then maps
/// `$f` over it (a scalar stays a scalar, a channel keeps its axis).
macro_rules! unary {
    ($fn_name:ident, $f:expr) => {
        #[doc = concat!("The `", stringify!($fn_name), "` builtin: `", stringify!($f), "` over every sample (or the scalar).")]
        pub(crate) fn $fn_name(call: Call<'_>) -> Result<Value, MathEvalError> {
            let Call { name, args, .. } = call;
            require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), $f)
        }
    };
}

unary!(abs, f64::abs);
unary!(sqrt, f64::sqrt);
unary!(sign, dart_sign);
unary!(floor, f64::floor);
unary!(ceil, f64::ceil);
unary!(round, f64::round);
unary!(sin, f64::sin);
unary!(cos, f64::cos);
unary!(tan, f64::tan);
unary!(asin, f64::asin);
unary!(acos, f64::acos);
unary!(atan, f64::atan);
unary!(sinh, f64::sinh);
unary!(cosh, f64::cosh);
unary!(tanh, f64::tanh);

/// The `deg2rad` builtin.
pub(crate) fn deg2rad(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    map_value(args.into_iter().next().unwrap(), |x| x * std::f64::consts::PI / 180.0)
}

/// The `rad2deg` builtin.
pub(crate) fn rad2deg(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    map_value(args.into_iter().next().unwrap(), |x| x * 180.0 / std::f64::consts::PI)
}

/// The `pow` builtin.
pub(crate) fn pow(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    let mut it = args.into_iter();
    elemwise(it.next().unwrap(), it.next().unwrap(), "pow", |a, b| Ok(a.powf(b)))
}

/// The `atan2` builtin.
pub(crate) fn atan2(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    let mut it = args.into_iter();
    elemwise(it.next().unwrap(), it.next().unwrap(), "atan2", |a, b| Ok(a.atan2(b)))
}

// Retired from `clamp` (R143): gratuitous rename — matches
// `numpy.clip`'s name, not just its behaviour.
/// The `clip` builtin.
pub(crate) fn clip(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 3)?;
    let ch = require_channel(&args[0], name)?;
    let lo = require_scalar(&args[1], name)?;
    let hi = require_scalar(&args[2], name)?;
    // std's f64::clamp panics if lo/hi is NaN or lo > hi — both
    // reachable from a user-typed scalar (an infinite lo or hi is a
    // legitimate one-sided clamp and stays allowed). Written as
    // `lo <= hi` (not `!(lo > hi)`) so a NaN lo/hi, which makes every
    // comparison false, is caught by this same check rather than
    // slipping through to the panic.
    if !(lo <= hi) {
        return Err(err(
            MathEvalErrorKind::Runtime,
            format!("clip: lo ({lo}) and hi ({hi}) must not be NaN, and lo must be <= hi"),
        ));
    }
    let out = ch.samples.iter().map(|&x| x.clamp(lo, hi)).collect();
    Ok(channel(out, ch.sample_rate_hz, ch.t_us))
}

// Retired from `if` (R143): gratuitous rename — matches
// `numpy.where`'s name. Also widens `cond` to accept a scalar
// (additive, plan §1 row 7): `numpy.where` never requires its
// condition to be array-shaped, and a scalar `cond` selecting a
// whole branch (rather than per-sample) is a strict superset of the
// old `if(cond,t,f)` behaviour — every existing per-sample call
// still takes the channel-cond path below, unchanged.
/// The `where` builtin.
pub(crate) fn where_(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 3)?;
    if let Value::Scalar(c) = &args[0] {
        let cond_true = *c != 0.0;
        let mut it = args.into_iter();
        let (_cond, t, f) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        return Ok(if cond_true { t } else { f });
    }
    let cond = require_channel(&args[0], "where(cond,t,f) — cond")?;
    let n = cond.samples.len();
    // A channel branch is paired with `cond` sample for sample, so it
    // must agree with it in shape, rate and length exactly as an
    // arithmetic operand would — otherwise a 1 Hz branch under an
    // 800 Hz condition read past its own end (P0-3). A scalar branch
    // broadcasts and is not checked.
    for branch in [&args[1], &args[2]] {
        if let Value::Channel(b) = branch {
            require_same_shape("where(cond,t,f)", &cond, b)?;
        }
    }
    // L3-R33: cond's t_us no longer wins by default — it is folded via
    // combine_t_us against the t/f operands' own axes too (a scalar
    // operand contributes no axis, i.e. empty). Equal-or-empty passes
    // through; a genuine mismatch is the same typed Runtime error
    // combine_t_us already gives elemwise(), naming both spans.
    let t_us = combine_t_us("where(cond,t,f)", &cond.t_us, &value_t_us(&args[1]))?;
    let t_us = combine_t_us("where(cond,t,f)", &t_us, &value_t_us(&args[2]))?;
    let out = collect_fallible(n, |i| {
        if cond.samples[i] != 0.0 {
            value_at(&args[1], i)
        } else {
            value_at(&args[2], i)
        }
    })?;
    Ok(channel_shared(out, cond.sample_rate_hz, t_us))
}

/// The `keep` builtin.
pub(crate) fn keep(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    let x = require_channel(&args[0], name)?;
    keep_where(&x, &args[1])
}
