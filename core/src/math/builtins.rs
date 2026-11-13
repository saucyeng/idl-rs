//! idl-rs's own builtins (spec `2026-11-13-idl-rs-builtin-registry-v2.md`):
//! each one a typed function and its hand-written [`BuiltinMeta`], registered
//! in C2 §3.3's table order, which [`crate::math::math_builtin_catalog`] and
//! the generated references keep. One-line bodies sit in the list as closures;
//! longer ones live in the family modules below.

mod aggregate;
mod laps;
mod signal;

use crate::math::eval::{collect_fallible, combine_t_us, dart_sign, elemwise, keep_where, map_value, require_arg_count, require_channel, require_same_shape, require_scalar, value_at, value_t_us, channel, channel_shared, err};
use crate::math::registry::{BuiltinMeta, Registry, RegistryError};
use crate::math::value::{ChannelValue, Value};
use crate::math::vector;
use crate::math::{MathEvalError, MathEvalErrorKind};

/// Registers every idl-rs builtin on `r`, in catalog order.
///
/// # Errors
/// [`RegistryError::DuplicateName`] when `r` already holds one of the names.
pub(super) fn register(r: &mut Registry) -> Result<(), RegistryError> {
    r.add(BuiltinMeta { name: "butter", category: "Filter", unit_rule: "SameAsArg(3)", shape: "[t]", kwargs: &[],
        signature: "butter(order, cutoff_hz, \"low\"|\"lowpass\"|\"high\"|\"highpass\", ch)",
        description: "Zero-phase Butterworth low- or high-pass filter; designs and applies the filter in one call. Needs a fixed-rate channel: put an event channel on a grid first, `resample(x, [Time])`.",
        example: "butter(2, 5, \"low\", [Fork])" }, signal::butter)?;
    r.add_not_implemented(BuiltinMeta { name: "sosfilt", category: "Filter", unit_rule: "SameAsArg(1)", shape: "[t]", kwargs: &[],
        signature: "sosfilt(sos, ch)",
        description: "Applies a second-order-section filter to a channel. Not implemented.",
        example: "sosfilt(sos, [Fork])" }, &[2])?;
    r.add(BuiltinMeta { name: "declip", category: "Reconstruction", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "declip(ch)",
        description: "Reconstructs samples lost to clipping; designed for accelerometer data clipped at +/-32 g. Needs a fixed-rate channel.",
        example: "declip([AccelZ])" }, signal::declip)?;
    r.add(BuiltinMeta { name: "cumulative_trapezoid", category: "Time-domain", unit_rule: "Product(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "cumulative_trapezoid(ch)",
        description: "Cumulative trapezoidal integral of a channel with respect to time. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.",
        example: "cumulative_trapezoid([AccelZ])" }, signal::cumulative_trapezoid)?;
    r.add(BuiltinMeta { name: "cumtrapz", category: "Time-domain", unit_rule: "Product(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "cumtrapz(ch)",
        description: "A permanent second spelling of `cumulative_trapezoid`, as scipy itself carries both.",
        example: "cumtrapz([AccelZ])" }, signal::cumulative_trapezoid)?;
    r.add(BuiltinMeta { name: "differentiate", category: "Time-domain", unit_rule: "Quotient(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "differentiate(ch)",
        description: "Backward difference with respect to time, with `result[0] = 0`. Deliberately not `gradient`'s central difference. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.",
        example: "differentiate([Fork])" }, signal::differentiate)?;
    r.add(BuiltinMeta { name: "gradient", category: "Time-domain", unit_rule: "Quotient(SameAsArg(0), Fixed(s))", shape: "[t]", kwargs: &[],
        signature: "gradient(ch)",
        description: "Central difference with respect to time, `numpy.gradient`'s own formula. A fixed-rate channel steps by its rate; an event channel by its recorded sample times.",
        example: "gradient([Fork])" }, signal::gradient)?;
    r.add_raw(BuiltinMeta { name: "detrend", category: "Time-domain", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "detrend(ch) | detrend(ch, \"linear\"|\"constant\"|\"mean\"|\"none\")",
        description: "Removes a constant or linear trend from a channel.",
        example: "detrend([Fork], \"linear\")" }, &[1, 2], signal::detrend)?;
    r.add_raw(BuiltinMeta { name: "rms", category: "Time-domain / aggregate", unit_rule: "SameAsArg(0)", shape: "scalar | [t]", kwargs: &[],
        signature: "rms(ch) -> scalar | rms(ch, w) -> rolling channel, `w` a window in samples",
        description: "Root mean square over the whole window, or rolling over `w` samples.",
        example: "rms([Fork], 64)" }, &[1, 2], aggregate::rms)?;
    r.add_raw(BuiltinMeta { name: "mean", category: "Time-domain / aggregate", unit_rule: "SameAsArg(0)", shape: "scalar | [t]", kwargs: &["window"],
        signature: "mean(ch) -> scalar | mean(ch, w) -> rolling channel",
        description: "Arithmetic mean over the whole window, or rolling over `w` samples.",
        example: "mean([Fork], 64)" }, &[1, 2], aggregate::mean)?;
    r.add_raw(BuiltinMeta { name: "std", category: "Time-domain / aggregate", unit_rule: "SameAsArg(0)", shape: "scalar | [t]", kwargs: &[],
        signature: "std(ch) -> scalar (population sigma) | std(ch, w) -> rolling channel",
        description: "Population standard deviation over the whole window, or rolling over `w` samples.",
        example: "std([Fork])" }, &[1, 2], aggregate::std)?;
    r.add_raw(BuiltinMeta { name: "median", category: "Aggregate", unit_rule: "SameAsArg(0)", shape: "scalar", kwargs: &[],
        signature: "median(ch) -> scalar",
        description: "Median of every sample in the window. The two-argument rolling form is not implemented.",
        example: "median([Fork])" }, &[1], aggregate::median)?;
    r.add_raw(BuiltinMeta { name: "sum", category: "Aggregate", unit_rule: "SameAsArg(0)", shape: "scalar", kwargs: &[],
        signature: "sum(ch) -> scalar",
        description: "Raw sum of every sample in the window, not time-normalised.",
        example: "sum([Fork])" }, &[1], aggregate::reduce(crate::math::aggregate::sum))?;
    r.add_raw(BuiltinMeta { name: "count", category: "Aggregate", unit_rule: "Dimensionless", shape: "scalar", kwargs: &[],
        signature: "count(ch) -> scalar",
        description: "Number of samples in the window.",
        example: "count([Fork])" }, &[1], aggregate::reduce(crate::math::aggregate::count))?;
    r.add_raw(BuiltinMeta { name: "first", category: "Aggregate", unit_rule: "SameAsArg(0)", shape: "scalar", kwargs: &[],
        signature: "first(ch) -> scalar",
        description: "First sample in the window.",
        example: "first([Speed])" }, &[1], aggregate::reduce(crate::math::aggregate::first))?;
    r.add_raw(BuiltinMeta { name: "last", category: "Aggregate", unit_rule: "SameAsArg(0)", shape: "scalar", kwargs: &[],
        signature: "last(ch) -> scalar",
        description: "Last sample in the window.",
        example: "last([Speed])" }, &[1], aggregate::reduce(crate::math::aggregate::last))?;
    r.add_raw(BuiltinMeta { name: "percentile", category: "Aggregate", unit_rule: "SameAsArg(0)", shape: "scalar", kwargs: &[],
        signature: "percentile(ch, quantile) -> scalar, `quantile` in [0, 100]",
        description: "Value below which `quantile` percent of the samples fall. Retired name: `p`.",
        example: "percentile([Fork], 95)" }, &[2], aggregate::percentile)?;
    r.add(BuiltinMeta { name: "abs", category: "Elementwise", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "abs(x)",
        description: "Absolute value, sample by sample.",
        example: "abs([AccelY])" }, unary(f64::abs))?;
    r.add(BuiltinMeta { name: "sqrt", category: "Elementwise", unit_rule: "PowN(0, 1/2)", shape: "[t]", kwargs: &[],
        signature: "sqrt(x)",
        description: "Square root, sample by sample; every unit exponent is halved.",
        example: "sqrt([Power])" }, unary(f64::sqrt))?;
    r.add(BuiltinMeta { name: "sign", category: "Elementwise", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "sign(x)",
        description: "-1, 0 or 1 by sign; NaN stays NaN.",
        example: "sign(differentiate([Fork]))" }, unary(dart_sign))?;
    r.add(BuiltinMeta { name: "floor", category: "Elementwise", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "floor(x)",
        description: "Rounds each sample down to the nearest integer.",
        example: "floor([Speed])" }, unary(f64::floor))?;
    r.add(BuiltinMeta { name: "ceil", category: "Elementwise", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "ceil(x)",
        description: "Rounds each sample up to the nearest integer.",
        example: "ceil([Speed])" }, unary(f64::ceil))?;
    r.add(BuiltinMeta { name: "round", category: "Elementwise", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "round(x)",
        description: "Rounds half away from zero (`2.5` gives `3`), not `numpy.round`'s banker's rounding.",
        example: "round([Speed])" }, unary(f64::round))?;
    r.add(BuiltinMeta { name: "pow", category: "Elementwise", unit_rule: "PowN(0, y) for a literal `y`; otherwise Dimensionless or Unknown", shape: "[t]", kwargs: &[],
        signature: "pow(x, y)",
        description: "Raises each sample to the power `y`.",
        example: "pow([Speed], 2)" }, binary("pow", f64::powf))?;
    r.add_raw(BuiltinMeta { name: "min", category: "Aggregate / elementwise", unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument", shape: "scalar | [t]", kwargs: &[],
        signature: "min(ch) -> scalar | min(a, b) -> elementwise",
        description: "Smallest sample in the window, or the per-sample smaller of two operands.",
        example: "min([Fork], [Shock])" }, &[1, 2], aggregate::min)?;
    r.add_raw(BuiltinMeta { name: "max", category: "Aggregate / elementwise", unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument", shape: "scalar | [t]", kwargs: &[],
        signature: "max(ch) -> scalar | max(a, b) -> elementwise",
        description: "Largest sample in the window, or the per-sample larger of two operands.",
        example: "max([Fork])" }, &[1, 2], aggregate::max)?;
    r.add(BuiltinMeta { name: "clip", category: "Elementwise", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "clip(ch, lo, hi)",
        description: "Limits each sample to `lo`..`hi`, both given in the channel's own units. Retired name: `clamp`.",
        example: "clip([Fork], 0, 160)" }, clip)?;
    r.add(BuiltinMeta { name: "sin", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "sin(x)",
        description: "Sine of an angle in radians.",
        example: "sin(deg2rad([Roll]))" }, unary(f64::sin))?;
    r.add(BuiltinMeta { name: "cos", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "cos(x)",
        description: "Cosine of an angle in radians.",
        example: "cos(deg2rad([Roll]))" }, unary(f64::cos))?;
    r.add(BuiltinMeta { name: "tan", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "tan(x)",
        description: "Tangent of an angle in radians.",
        example: "tan(deg2rad([Roll]))" }, unary(f64::tan))?;
    r.add(BuiltinMeta { name: "asin", category: "Trig", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "asin(x)",
        description: "Arcsine, in radians; the argument must be dimensionless.",
        example: "asin([Ratio])" }, unary(f64::asin))?;
    r.add(BuiltinMeta { name: "acos", category: "Trig", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "acos(x)",
        description: "Arccosine, in radians; the argument must be dimensionless.",
        example: "acos([Ratio])" }, unary(f64::acos))?;
    r.add(BuiltinMeta { name: "atan", category: "Trig", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "atan(x)",
        description: "Arctangent, in radians; the argument must be dimensionless.",
        example: "atan([Ratio])" }, unary(f64::atan))?;
    r.add(BuiltinMeta { name: "atan2", category: "Trig", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "atan2(y, x)",
        description: "Quadrant-correct two-argument arctangent, in radians.",
        example: "atan2([AccelY], [AccelZ])" }, binary("atan2", f64::atan2))?;
    r.add(BuiltinMeta { name: "sinh", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "sinh(x)",
        description: "Hyperbolic sine.",
        example: "sinh([Ratio])" }, unary(f64::sinh))?;
    r.add(BuiltinMeta { name: "cosh", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "cosh(x)",
        description: "Hyperbolic cosine.",
        example: "cosh([Ratio])" }, unary(f64::cosh))?;
    r.add(BuiltinMeta { name: "tanh", category: "Trig", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "tanh(x)",
        description: "Hyperbolic tangent.",
        example: "tanh([Ratio])" }, unary(f64::tanh))?;
    r.add(BuiltinMeta { name: "deg2rad", category: "Trig conversion", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "deg2rad(x)",
        description: "Converts degrees to radians.",
        example: "deg2rad([Roll])" }, unary(|x| x * std::f64::consts::PI / 180.0))?;
    r.add(BuiltinMeta { name: "rad2deg", category: "Trig conversion", unit_rule: "Fixed(deg)", shape: "[t]", kwargs: &[],
        signature: "rad2deg(x)",
        description: "Converts radians to degrees.",
        example: "rad2deg(atan2([AccelY], [AccelZ]))" }, unary(|x| x * 180.0 / std::f64::consts::PI))?;
    r.add(BuiltinMeta { name: "periodogram", category: "Frequency", unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")", shape: "[f]", kwargs: &["window", "detrend", "scaling"],
        signature: "periodogram(ch, window=\"boxcar\", detrend=\"constant\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
        description: "Single-segment power spectrum, scipy-named and scipy-scaled. Needs a fixed-rate channel. Retired name: `fft`.",
        example: "periodogram([Fork])" }, signal::periodogram)?;
    r.add(BuiltinMeta { name: "welch", category: "Frequency", unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")", shape: "[f]", kwargs: &["window", "nperseg", "noverlap", "detrend", "average", "scaling"],
        signature: "welch(ch, window=\"hann\", nperseg=n, noverlap=n, detrend=\"constant\", average=\"mean\"|\"median\"|\"max\"|\"none\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
        description: "Segmented, averaged power spectrum -- the spectrum the FFT charts compute. Needs a fixed-rate channel.",
        example: "welch([Fork])" }, signal::welch)?;
    r.add_not_implemented(BuiltinMeta { name: "spectrogram", category: "Frequency", unit_rule: "SelectByLiteral(scaling, ...) as `periodogram`, over a `[t,f]` value", shape: "[t,f]", kwargs: &[],
        signature: "spectrogram(ch, window_size, hop_size, window, detrend, scaling)",
        description: "Time-varying spectrum. Not implemented; the signature and shape are fixed by C2 3.6.3.",
        example: "spectrogram([Fork])" }, &[1])?;
    r.add_not_implemented(BuiltinMeta { name: "envelope", category: "Frequency", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "envelope(ch)",
        description: "Amplitude envelope -- a magnitude, not scipy's complex analytic signal. Not implemented. Retired name: `hilbert`.",
        example: "envelope([Fork])" }, &[1])?;
    r.add_not_implemented(BuiltinMeta { name: "correlate", category: "Correlation", unit_rule: "Product(0, 1)", shape: "[t]", kwargs: &[],
        signature: "correlate(a, b)",
        description: "Cross-correlation of two channels. Not implemented.",
        example: "correlate([Fork], [Shock])" }, &[2])?;
    r.add_not_implemented(BuiltinMeta { name: "convolve", category: "Correlation", unit_rule: "Product(0, 1)", shape: "[t]", kwargs: &[],
        signature: "convolve(ch, kernel)",
        description: "Convolution of a channel with a kernel. Not implemented.",
        example: "convolve([Fork], kernel)" }, &[2])?;
    r.add(BuiltinMeta { name: "resample", category: "Resampling", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "resample(x, onto)",
        description: "Linear interpolation of `x` onto the per-sample recorded times of `onto`; the result carries `onto`'s time axis, rate and length, so the two can then be combined. Times outside `x`'s span are `NaN` -- never extrapolated. The explicit way to combine two channels recorded on different clocks; nothing resamples implicitly.",
        example: "resample([IMU2_AccelZ], [IMU1_AccelZ])" }, signal::resample)?;
    r.add(BuiltinMeta { name: "deglitch_pulses", category: "Resampling", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &["max_fraction"],
        signature: "deglitch_pulses(pulses, max_fraction=0.35)",
        description: "A wheel-pulse event channel without the extra edges a marginal sensor adds: a pulse is dropped when the gap ending at it is under `max_fraction` of both the gap before and the gap after, so it reads as an extra edge inside one ordinary gap. Comparing both sides means the first pulses after a locked wheel spins up, and the growing gaps of hard braking, are never dropped. The result is renumbered 1, 2, 3, ... from the first kept pulse, at the kept pulses' recorded times.",
        example: "deglitch_pulses([WheelFront])" }, signal::deglitch_pulses)?;
    r.add(BuiltinMeta { name: "keep", category: "Logic", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "keep(x, cond)",
        description: "Only the samples of `x` where `cond` is true (non-zero), each at its own recorded time: an event list, such as every pulse gap over a threshold, comes out as a short channel instead of a full-length one with zeros. `cond` pairs with `x` sample for sample (put it on `x`'s axis with `resample` first if it is not); a number keeps all or nothing. Nothing kept is an empty channel. Over a `[lap]` value it keeps laps.",
        example: "keep([gap_ratio], [gap_ratio] > 1.8)" }, |x: ChannelValue, cond: Value| keep_where(&x, &cond))?;
    r.add(BuiltinMeta { name: "where", category: "Logic", unit_rule: "AllMatch(1, 2)", shape: "[t]", kwargs: &[],
        signature: "where(cond, t, f)",
        description: "Per-sample choice between two branches; `cond` may also be a scalar, selecting a whole branch. Retired name: `if`.",
        example: "where([Speed] > 10, [Fork], 0)" }, where_)?;
    r.add(BuiltinMeta { name: "current_lap", category: "Lap", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "current_lap()",
        description: "1-based lap number at each sample, `0` outside any lap.",
        example: "current_lap()" }, laps::current_lap)?;
    r.add(BuiltinMeta { name: "lap_start_time", category: "Lap", unit_rule: "Fixed(s)", shape: "scalar", kwargs: &[],
        signature: "lap_start_time(n)",
        description: "Session time at which lap `n` starts; NaN when `n` is out of range.",
        example: "lap_start_time(2)" }, laps::lap_start_time)?;
    r.add(BuiltinMeta { name: "lap_start_distance", category: "Lap", unit_rule: "Fixed(m)", shape: "scalar", kwargs: &[],
        signature: "lap_start_distance(n)",
        description: "Distance at which lap `n` starts; NaN when `n` is out of range or the session has no `[Distance]`.",
        example: "lap_start_distance(2)" }, laps::lap_start_distance)?;
    r.add(BuiltinMeta { name: "sector_number", category: "Lap", unit_rule: "Dimensionless", shape: "[t]", kwargs: &[],
        signature: "sector_number()",
        description: "0-based sector index at each sample, NaN outside any sector.",
        example: "sector_number()" }, laps::sector_number)?;
    r.add(BuiltinMeta { name: "lap_number", category: "Lap", unit_rule: "Dimensionless", shape: "scalar | [lap]", kwargs: &[],
        signature: "lap_number()",
        description: "The 1-based lap number: this row's lap in a table cell, every lap in a math cell.",
        example: "lap_number()" }, laps::lap_number)?;
    r.add(BuiltinMeta { name: "lap_time", category: "Lap", unit_rule: "Fixed(s)", shape: "scalar | [lap]", kwargs: &[],
        signature: "lap_time()",
        description: "Lap duration in seconds: this row's lap in a table cell, every lap in a math cell.",
        example: "lap_time()" }, laps::lap_time)?;
    r.add(BuiltinMeta { name: "sector_time", category: "Lap", unit_rule: "Fixed(s)", shape: "scalar | [lap]", kwargs: &[],
        signature: "sector_time(i)",
        description: "Duration of the lap's `i`-th sector (0-based, as `sector_number()` counts) in seconds; NaN when the lap has no such sector.",
        example: "sector_time(1)" }, laps::sector_time)?;
    r.add(BuiltinMeta { name: "lap_delta_time", category: "Lap delta", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "lap_delta_time(ch)",
        description: "Main lap minus overlay lap, time-matched; the mean across every overlay when more than one is selected. Retired name: `variance_time`.",
        example: "lap_delta_time([Speed])" }, laps::lap_delta_time)?;
    r.add(BuiltinMeta { name: "lap_delta_dist", category: "Lap delta", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "lap_delta_dist(ch)",
        description: "Main lap minus overlay lap, arc-length-matched. Retired name: `variance_dist`.",
        example: "lap_delta_dist([Speed])" }, laps::lap_delta_dist)?;
    r.add(BuiltinMeta { name: "attitude", category: "Estimator (diagnostic)", unit_rule: "Fixed(deg)", shape: "[t]", kwargs: &[],
        signature: "attitude(\"roll\"|\"pitch\")",
        description: "Bike attitude estimated by the AHRS filter, in degrees.",
        example: "attitude(\"roll\")" }, signal::estimator)?;
    r.add(BuiltinMeta { name: "body_accel", category: "Estimator (diagnostic)", unit_rule: "Fixed(g)", shape: "[t]", kwargs: &[],
        signature: "body_accel(\"long\"|\"lat\")",
        description: "Gravity-compensated body-frame acceleration, in g.",
        example: "body_accel(\"long\")" }, signal::estimator)?;
    r.add(BuiltinMeta { name: "wheel_travel", category: "Estimator", unit_rule: "Fixed(mm)", shape: "[t]", kwargs: &[],
        signature: "wheel_travel(\"front\"|\"rear\")",
        description: "Suspension travel at the named wheel, in mm.",
        example: "wheel_travel(\"front\")" }, signal::estimator)?;
    r.add(BuiltinMeta { name: "wheel_velocity", category: "Estimator", unit_rule: "Fixed(mm/s)", shape: "[t]", kwargs: &[],
        signature: "wheel_velocity(\"front\"|\"rear\")",
        description: "Suspension velocity at the named wheel, in mm/s.",
        example: "wheel_velocity(\"front\")" }, signal::estimator)?;
    r.add(BuiltinMeta { name: "vec", category: "Vector", unit_rule: "AllMatch(0, 1, 2)", shape: "Vec3", kwargs: &[],
        signature: "vec(x, y, z)",
        description: "Builds a Vec3 from three components, which must share one unit.",
        example: "vec([AccelX], [AccelY], [AccelZ])" }, |x: Value, y: Value, z: Value| vector::make_vec3(&x, &y, &z, "vec"))?;
    r.add(BuiltinMeta { name: "vx", category: "Vector", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "vx(v)",
        description: "The x component of a Vec3.",
        example: "vx(accel)" }, |v: Value| vector::component(&v, 0, "vx"))?;
    r.add(BuiltinMeta { name: "vy", category: "Vector", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "vy(v)",
        description: "The y component of a Vec3.",
        example: "vy(accel)" }, |v: Value| vector::component(&v, 1, "vy"))?;
    r.add(BuiltinMeta { name: "vz", category: "Vector", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "vz(v)",
        description: "The z component of a Vec3.",
        example: "vz(accel)" }, |v: Value| vector::component(&v, 2, "vz"))?;
    r.add(BuiltinMeta { name: "vadd", category: "Vector", unit_rule: "AllMatch(0, 1)", shape: "Vec3", kwargs: &[],
        signature: "vadd(a, b)",
        description: "Component-wise sum of two Vec3s.",
        example: "vadd(a, b)" }, |a: Value, b: Value| vector::vadd(&a, &b, "vadd"))?;
    r.add(BuiltinMeta { name: "vsub", category: "Vector", unit_rule: "AllMatch(0, 1)", shape: "Vec3", kwargs: &[],
        signature: "vsub(a, b)",
        description: "Component-wise difference of two Vec3s.",
        example: "vsub(a, b)" }, |a: Value, b: Value| vector::vsub(&a, &b, "vsub"))?;
    r.add(BuiltinMeta { name: "vscale", category: "Vector", unit_rule: "Product(0, 1)", shape: "Vec3", kwargs: &[],
        signature: "vscale(v, s)",
        description: "Scales a Vec3 by a scalar.",
        example: "vscale(accel, 9.81)" }, |v: Value, s: Value| vector::vscale(&v, &s, "vscale"))?;
    r.add(BuiltinMeta { name: "cross", category: "Vector", unit_rule: "Product(0, 1)", shape: "Vec3", kwargs: &[],
        signature: "cross(a, b)",
        description: "Cross product of two Vec3s.",
        example: "cross(a, b)" }, |a: Value, b: Value| vector::cross(&a, &b, "cross"))?;
    r.add(BuiltinMeta { name: "dot", category: "Vector", unit_rule: "Product(0, 1)", shape: "[t]", kwargs: &[],
        signature: "dot(a, b)",
        description: "Dot product of two Vec3s.",
        example: "dot(a, b)" }, |a: Value, b: Value| vector::dot(&a, &b, "dot"))?;
    r.add(BuiltinMeta { name: "norm", category: "Vector", unit_rule: "SameAsArg(0)", shape: "[t]", kwargs: &[],
        signature: "norm(v)",
        description: "Euclidean length of a Vec3.",
        example: "norm(accel)" }, |v: Value| vector::norm(&v, "norm"))?;
    r.add(BuiltinMeta { name: "normalize", category: "Vector", unit_rule: "Dimensionless", shape: "Vec3", kwargs: &[],
        signature: "normalize(v)",
        description: "Unit vector in the same direction as `v`.",
        example: "normalize(accel)" }, |v: Value| vector::normalize(&v, "normalize"))?;
    r.add(BuiltinMeta { name: "angle_between", category: "Vector", unit_rule: "Fixed(rad)", shape: "[t]", kwargs: &[],
        signature: "angle_between(a, b)",
        description: "Angle between two Vec3s, in radians, in [0, pi]. Retired name: `angle`.",
        example: "angle_between(a, b)" }, |a: Value, b: Value| vector::angle(&a, &b, "angle_between"))?;
    r.add_raw(BuiltinMeta { name: "rotate_mat", category: "Rotation", unit_rule: "SameAsArg(0)", shape: "Vec3", kwargs: &[],
        signature: "rotate_mat(v, m00..m22) (row-major, scalar entries)",
        description: "Rotates a Vec3 by a row-major 3x3 matrix.",
        example: "rotate_mat(a, 1, 0, 0, 0, 1, 0, 0, 0, 1)" }, &[10], rotate_mat)?;
    r.add(BuiltinMeta { name: "rotate_axis", category: "Rotation", unit_rule: "SameAsArg(0)", shape: "Vec3", kwargs: &[],
        signature: "rotate_axis(v, ax, ay, az, angle) (scalars; `angle` in radians)",
        description: "Rotates a Vec3 about an axis by an angle in radians.",
        example: "rotate_axis(a, 0, 0, 1, 1.5708)" }, |v: Value, ax: f64, ay: f64, az: f64, angle: f64| vector::rotate_axis(&v, ax, ay, az, angle, "rotate_axis"))?;
    r.add(BuiltinMeta { name: "rotate_euler", category: "Rotation", unit_rule: "SameAsArg(0)", shape: "Vec3", kwargs: &[],
        signature: "rotate_euler(v, roll, pitch, yaw) (radians; the angles may be channels)",
        description: "Rotates a Vec3 by roll/pitch/yaw in radians; the angles may be channels, giving a per-sample rotation.",
        example: "rotate_euler(a, [Roll], [Pitch], [Yaw])" }, |v: Value, roll: Value, pitch: Value, yaw: Value| vector::rotate_euler(&v, &roll, &pitch, &yaw, "rotate_euler"))?;
    // `main({col[]})`: the Main lap's row of a table column — a table-cell
    // form, left out of the catalog (C2 §4, not §3.3).
    r.add_undocumented("main", &[1], aggregate::main)?;
    Ok(())
}

/// An elementwise `f64 -> f64` map over a scalar or channel argument.
fn unary(f: fn(f64) -> f64) -> impl Fn(Value) -> Result<Value, MathEvalError> + Send + Sync + 'static {
    move |x: Value| map_value(x, f)
}

/// An elementwise binary operation, broadcasting a scalar against a channel.
fn binary(name: &'static str, f: fn(f64, f64) -> f64) -> impl Fn(Value, Value) -> Result<Value, MathEvalError> + Send + Sync + 'static {
    move |a: Value, b: Value| elemwise(a, b, name, |x, y| Ok(f(x, y)))
}

/// `clip(ch, lo, hi)`: every sample limited to `[lo, hi]`, in the channel's unit.
fn clip(ch: ChannelValue, lo: f64, hi: f64) -> Result<Value, MathEvalError> {
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

/// `where(cond, t, f)`, `numpy.where`'s name (retired from `if`, R143): `t`
/// where `cond` is non-zero, else `f`. A scalar `cond` selects a whole
/// branch (additive, plan §1 row 7) — every per-sample call still takes the
/// channel-cond path below, unchanged.
fn where_(cond: Value, t: Value, f: Value) -> Result<Value, MathEvalError> {
    if let Value::Scalar(c) = &cond {
        return Ok(if *c != 0.0 { t } else { f });
    }
    let cond = require_channel(&cond, "where(cond,t,f) — cond")?;
    let n = cond.samples.len();
    // A channel branch is paired with `cond` sample for sample, so it
    // must agree with it in shape, rate and length exactly as an
    // arithmetic operand would — otherwise a 1 Hz branch under an
    // 800 Hz condition read past its own end (P0-3). A scalar branch
    // broadcasts and is not checked.
    for branch in [&t, &f] {
        if let Value::Channel(b) = branch {
            require_same_shape("where(cond,t,f)", &cond, b)?;
        }
    }
    // L3-R33: cond's t_us no longer wins by default — it is folded via
    // combine_t_us against the t/f operands' own axes too (a scalar
    // operand contributes no axis, i.e. empty). Equal-or-empty passes
    // through; a genuine mismatch is the same typed Runtime error
    // combine_t_us already gives elemwise(), naming both spans.
    let t_us = combine_t_us("where(cond,t,f)", &cond.t_us, &value_t_us(&t))?;
    let t_us = combine_t_us("where(cond,t,f)", &t_us, &value_t_us(&f))?;
    let out = collect_fallible(n, |i| {
        if cond.samples[i] != 0.0 {
            value_at(&t, i)
        } else {
            value_at(&f, i)
        }
    })?;
    Ok(channel_shared(out, cond.sample_rate_hz, t_us))
}

/// `rotate_mat(v, m00 … m22)`: `v` times a constant row-major 3×3 matrix.
/// Ten arguments, past [`crate::math::registry::IntoBuiltin`]'s six, so the
/// body counts them.
fn rotate_mat(_: &crate::math::registry::Ctx, args: &[Value]) -> Result<Value, MathEvalError> {
    require_arg_count("rotate_mat", args, 10)?;
    let mut m = [0.0_f64; 9];
    for (slot, arg) in m.iter_mut().zip(&args[1..]) {
        *slot = require_scalar(arg, "rotate_mat")?;
    }
    vector::rotate_mat(&args[0], &m, "rotate_mat")
}
