# Workbook reference

Everything a `.idl1wb` workbook can say: every math builtin, the
annotations a definition accepts, the variables a `js` cell is given,
and the chart grammar the Properties panel reads and writes.

**This file is generated in part.** The builtin catalog and the
retired-name table below are rendered from the engine itself by
`idl-rs docs workbook --out docs/WORKBOOK-REFERENCE.md`; CI regenerates
them and fails if the committed file differs. Edit the engine, or the
curated sources under `docs/reference-src/`, never this file.

## Math builtins

77 functions a `math` cell's expression can call, grouped by category
(C2 §3.3). A function marked **not implemented** parses and validates —
it is part of the committed language surface — but evaluating it is an
error today.

- **Filter** — [`butter`](#butter), [`sosfilt`](#sosfilt)
- **Reconstruction** — [`declip`](#declip)
- **Time-domain** — [`cumulative_trapezoid`](#cumulative_trapezoid), [`cumtrapz`](#cumtrapz), [`differentiate`](#differentiate), [`gradient`](#gradient), [`detrend`](#detrend)
- **Time-domain / aggregate** — [`rms`](#rms), [`mean`](#mean), [`std`](#std)
- **Aggregate** — [`median`](#median), [`sum`](#sum), [`count`](#count), [`first`](#first), [`last`](#last), [`percentile`](#percentile)
- **Elementwise** — [`abs`](#abs), [`sqrt`](#sqrt), [`sign`](#sign), [`floor`](#floor), [`ceil`](#ceil), [`round`](#round), [`pow`](#pow), [`clip`](#clip)
- **Aggregate / elementwise** — [`min`](#min), [`max`](#max)
- **Trig** — [`sin`](#sin), [`cos`](#cos), [`tan`](#tan), [`asin`](#asin), [`acos`](#acos), [`atan`](#atan), [`atan2`](#atan2), [`sinh`](#sinh), [`cosh`](#cosh), [`tanh`](#tanh)
- **Trig conversion** — [`deg2rad`](#deg2rad), [`rad2deg`](#rad2deg)
- **Frequency** — [`periodogram`](#periodogram), [`welch`](#welch), [`spectrogram`](#spectrogram), [`envelope`](#envelope)
- **Correlation** — [`correlate`](#correlate), [`convolve`](#convolve)
- **Resampling** — [`resample`](#resample), [`deglitch_pulses`](#deglitch_pulses)
- **Logic** — [`keep`](#keep), [`where`](#where)
- **Lap** — [`current_lap`](#current_lap), [`lap_start_time`](#lap_start_time), [`lap_start_distance`](#lap_start_distance), [`sector_number`](#sector_number), [`lap_number`](#lap_number), [`lap_time`](#lap_time), [`sector_time`](#sector_time)
- **Lap delta** — [`lap_delta_time`](#lap_delta_time), [`lap_delta_dist`](#lap_delta_dist)
- **Estimator (diagnostic)** — [`attitude`](#attitude), [`body_accel`](#body_accel)
- **Estimator** — [`wheel_travel`](#wheel_travel), [`wheel_velocity`](#wheel_velocity)
- **Vector** — [`vec`](#vec), [`vx`](#vx), [`vy`](#vy), [`vz`](#vz), [`vadd`](#vadd), [`vsub`](#vsub), [`vscale`](#vscale), [`cross`](#cross), [`dot`](#dot), [`norm`](#norm), [`normalize`](#normalize), [`angle_between`](#angle_between)
- **Rotation** — [`rotate_mat`](#rotate_mat), [`rotate_axis`](#rotate_axis), [`rotate_euler`](#rotate_euler)

### Filter

#### butter

```
butter(order, cutoff_hz, "low"|"lowpass"|"high"|"highpass", ch)
```

Zero-phase Butterworth low- or high-pass filter; designs and applies the filter in one call. Needs a fixed-rate channel: put an event channel on a grid first, `resample(x, [Time])`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(3)` |
| Status | implemented |
| Arguments | 4 |

Example:

```
result = butter(2, 5, "low", [Fork])
```

#### sosfilt

```
sosfilt(sos, ch)
```

Applies a second-order-section filter to a channel. Not implemented.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(1)` |
| Status | **not implemented** |
| Arguments | 2 |

Example:

```
result = sosfilt(sos, [Fork])
```

### Reconstruction

#### declip

```
declip(ch)
```

Reconstructs samples lost to clipping; designed for accelerometer data clipped at +/-32 g. Needs a fixed-rate channel.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = declip([AccelZ])
```

### Time-domain

#### cumulative_trapezoid

```
cumulative_trapezoid(ch)
```

Cumulative trapezoidal integral of a channel with respect to time. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Product(SameAsArg(0), Fixed(s))` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = cumulative_trapezoid([AccelZ])
```

#### cumtrapz

```
cumtrapz(ch)
```

A permanent second spelling of `cumulative_trapezoid`, as scipy itself carries both.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Product(SameAsArg(0), Fixed(s))` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = cumtrapz([AccelZ])
```

#### differentiate

```
differentiate(ch)
```

Backward difference with respect to time, with `result[0] = 0`. Deliberately not `gradient`'s central difference. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Quotient(SameAsArg(0), Fixed(s))` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = differentiate([Fork])
```

#### gradient

```
gradient(ch)
```

Central difference with respect to time, `numpy.gradient`'s own formula. A fixed-rate channel steps by its rate; an event channel by its recorded sample times.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Quotient(SameAsArg(0), Fixed(s))` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = gradient([Fork])
```

#### detrend

```
detrend(ch) | detrend(ch, "linear"|"constant"|"mean"|"none")
```

Removes a constant or linear trend from a channel.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = detrend([Fork], "linear")
```

### Time-domain / aggregate

#### rms

```
rms(ch) -> scalar | rms(ch, w) -> rolling channel, `w` a window in samples
```

Root mean square over the whole window, or rolling over `w` samples.

| | |
|---|---|
| Shape | `scalar | [t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = rms([Fork], 64)
```

#### mean

```
mean(ch) -> scalar | mean(ch, w) -> rolling channel
```

Arithmetic mean over the whole window, or rolling over `w` samples.

| | |
|---|---|
| Shape | `scalar | [t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = mean([Fork], 64)
```

#### std

```
std(ch) -> scalar (population sigma) | std(ch, w) -> rolling channel
```

Population standard deviation over the whole window, or rolling over `w` samples.

| | |
|---|---|
| Shape | `scalar | [t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = std([Fork])
```

### Aggregate

#### median

```
median(ch) -> scalar
```

Median of every sample in the window. The two-argument rolling form is not implemented.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = median([Fork])
```

#### sum

```
sum(ch) -> scalar
```

Raw sum of every sample in the window, not time-normalised.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sum([Fork])
```

#### count

```
count(ch) -> scalar
```

Number of samples in the window.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = count([Fork])
```

#### first

```
first(ch) -> scalar
```

First sample in the window.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = first([Speed])
```

#### last

```
last(ch) -> scalar
```

Last sample in the window.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = last([Speed])
```

#### percentile

```
percentile(ch, quantile) -> scalar, `quantile` in [0, 100]
```

Value below which `quantile` percent of the samples fall. Retired name: `p`.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = percentile([Fork], 95)
```

### Elementwise

#### abs

```
abs(x)
```

Absolute value, sample by sample.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = abs([AccelY])
```

#### sqrt

```
sqrt(x)
```

Square root, sample by sample; every unit exponent is halved.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `PowN(0, 1/2)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sqrt([Power])
```

#### sign

```
sign(x)
```

-1, 0 or 1 by sign; NaN stays NaN.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sign(differentiate([Fork]))
```

#### floor

```
floor(x)
```

Rounds each sample down to the nearest integer.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = floor([Speed])
```

#### ceil

```
ceil(x)
```

Rounds each sample up to the nearest integer.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = ceil([Speed])
```

#### round

```
round(x)
```

Rounds half away from zero (`2.5` gives `3`), not `numpy.round`'s banker's rounding.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = round([Speed])
```

#### pow

```
pow(x, y)
```

Raises each sample to the power `y`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `PowN(0, y) for a literal `y`; otherwise Dimensionless or Unknown` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = pow([Speed], 2)
```

#### clip

```
clip(ch, lo, hi)
```

Limits each sample to `lo`..`hi`, both given in the channel's own units. Retired name: `clamp`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 3 |

Example:

```
result = clip([Fork], 0, 160)
```

### Aggregate / elementwise

#### min

```
min(ch) -> scalar | min(a, b) -> elementwise
```

Smallest sample in the window, or the per-sample smaller of two operands.

| | |
|---|---|
| Shape | `scalar | [t]` |
| Unit rule | `SameAsArg(0) one-argument; AllMatch(0, 1) two-argument` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = min([Fork], [Shock])
```

#### max

```
max(ch) -> scalar | max(a, b) -> elementwise
```

Largest sample in the window, or the per-sample larger of two operands.

| | |
|---|---|
| Shape | `scalar | [t]` |
| Unit rule | `SameAsArg(0) one-argument; AllMatch(0, 1) two-argument` |
| Status | implemented |
| Arguments | 1 or 2 |

Example:

```
result = max([Fork])
```

### Trig

#### sin

```
sin(x)
```

Sine of an angle in radians.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sin(deg2rad([Roll]))
```

#### cos

```
cos(x)
```

Cosine of an angle in radians.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = cos(deg2rad([Roll]))
```

#### tan

```
tan(x)
```

Tangent of an angle in radians.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = tan(deg2rad([Roll]))
```

#### asin

```
asin(x)
```

Arcsine, in radians; the argument must be dimensionless.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = asin([Ratio])
```

#### acos

```
acos(x)
```

Arccosine, in radians; the argument must be dimensionless.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = acos([Ratio])
```

#### atan

```
atan(x)
```

Arctangent, in radians; the argument must be dimensionless.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = atan([Ratio])
```

#### atan2

```
atan2(y, x)
```

Quadrant-correct two-argument arctangent, in radians.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = atan2([AccelY], [AccelZ])
```

#### sinh

```
sinh(x)
```

Hyperbolic sine.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sinh([Ratio])
```

#### cosh

```
cosh(x)
```

Hyperbolic cosine.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = cosh([Ratio])
```

#### tanh

```
tanh(x)
```

Hyperbolic tangent.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = tanh([Ratio])
```

### Trig conversion

#### deg2rad

```
deg2rad(x)
```

Converts degrees to radians.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = deg2rad([Roll])
```

#### rad2deg

```
rad2deg(x)
```

Converts radians to degrees.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(deg)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = rad2deg(atan2([AccelY], [AccelZ]))
```

### Frequency

#### periodogram

```
periodogram(ch, window="boxcar", detrend="constant", scaling="density"|"spectrum"|"raw_magnitude")
```

Single-segment power spectrum, scipy-named and scipy-scaled. Needs a fixed-rate channel. Retired name: `fft`.

| | |
|---|---|
| Shape | `[f]` |
| Unit rule | `SelectByLiteral(scaling, { "density" -> Quotient(PowN(0, 2), Fixed(Hz)), "spectrum" -> PowN(0, 2), "raw_magnitude" -> SameAsArg(0) }, default "density")` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = periodogram([Fork])
```

#### welch

```
welch(ch, window="hann", nperseg=n, noverlap=n, detrend="constant", average="mean"|"median"|"max"|"none", scaling="density"|"spectrum"|"raw_magnitude")
```

Segmented, averaged power spectrum -- the spectrum the FFT charts compute. Needs a fixed-rate channel.

| | |
|---|---|
| Shape | `[f]` |
| Unit rule | `SelectByLiteral(scaling, { "density" -> Quotient(PowN(0, 2), Fixed(Hz)), "spectrum" -> PowN(0, 2), "raw_magnitude" -> SameAsArg(0) }, default "density")` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = welch([Fork])
```

#### spectrogram

```
spectrogram(ch, window_size, hop_size, window, detrend, scaling)
```

Time-varying spectrum. Not implemented; the signature and shape are fixed by C2 3.6.3.

| | |
|---|---|
| Shape | `[t,f]` |
| Unit rule | `SelectByLiteral(scaling, ...) as `periodogram`, over a `[t,f]` value` |
| Status | **not implemented** |
| Arguments | 1 |

Example:

```
result = spectrogram([Fork])
```

#### envelope

```
envelope(ch)
```

Amplitude envelope -- a magnitude, not scipy's complex analytic signal. Not implemented. Retired name: `hilbert`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | **not implemented** |
| Arguments | 1 |

Example:

```
result = envelope([Fork])
```

### Correlation

#### correlate

```
correlate(a, b)
```

Cross-correlation of two channels. Not implemented.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Product(0, 1)` |
| Status | **not implemented** |
| Arguments | 2 |

Example:

```
result = correlate([Fork], [Shock])
```

#### convolve

```
convolve(ch, kernel)
```

Convolution of a channel with a kernel. Not implemented.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Product(0, 1)` |
| Status | **not implemented** |
| Arguments | 2 |

Example:

```
result = convolve([Fork], kernel)
```

### Resampling

#### resample

```
resample(x, onto)
```

Linear interpolation of `x` onto the per-sample recorded times of `onto`; the result carries `onto`'s time axis, rate and length, so the two can then be combined. Times outside `x`'s span are `NaN` -- never extrapolated. The explicit way to combine two channels recorded on different clocks; nothing resamples implicitly.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = resample([IMU2_AccelZ], [IMU1_AccelZ])
```

#### deglitch_pulses

```
deglitch_pulses(pulses, max_fraction=0.35)
```

A wheel-pulse event channel without the extra edges a marginal sensor adds: a pulse is dropped when the gap ending at it is under `max_fraction` of both the gap before and the gap after, so it reads as an extra edge inside one ordinary gap. Comparing both sides means the first pulses after a locked wheel spins up, and the growing gaps of hard braking, are never dropped. The result is renumbered 1, 2, 3, ... from the first kept pulse, at the kept pulses' recorded times.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = deglitch_pulses([WheelFront])
```

### Logic

#### keep

```
keep(x, cond)
```

Only the samples of `x` where `cond` is true (non-zero), each at its own recorded time: an event list, such as every pulse gap over a threshold, comes out as a short channel instead of a full-length one with zeros. `cond` pairs with `x` sample for sample (put it on `x`'s axis with `resample` first if it is not); a number keeps all or nothing. Nothing kept is an empty channel. Over a `[lap]` value it keeps laps.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = keep([gap_ratio], [gap_ratio] > 1.8)
```

#### where

```
where(cond, t, f)
```

Per-sample choice between two branches; `cond` may also be a scalar, selecting a whole branch. Retired name: `if`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `AllMatch(1, 2)` |
| Status | implemented |
| Arguments | 3 |

Example:

```
result = where([Speed] > 10, [Fork], 0)
```

### Lap

#### current_lap

```
current_lap()
```

1-based lap number at each sample, `0` outside any lap.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 0 |

Example:

```
result = current_lap()
```

#### lap_start_time

```
lap_start_time(n)
```

Session time at which lap `n` starts; NaN when `n` is out of range.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `Fixed(s)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = lap_start_time(2)
```

#### lap_start_distance

```
lap_start_distance(n)
```

Distance at which lap `n` starts; NaN when `n` is out of range or the session has no `[Distance]`.

| | |
|---|---|
| Shape | `scalar` |
| Unit rule | `Fixed(m)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = lap_start_distance(2)
```

#### sector_number

```
sector_number()
```

0-based sector index at each sample, NaN outside any sector.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 0 |

Example:

```
result = sector_number()
```

#### lap_number

```
lap_number()
```

The 1-based lap number: this row's lap in a table cell, every lap in a math cell.

| | |
|---|---|
| Shape | `scalar | [lap]` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 0 |

Example:

```
result = lap_number()
```

#### lap_time

```
lap_time()
```

Lap duration in seconds: this row's lap in a table cell, every lap in a math cell.

| | |
|---|---|
| Shape | `scalar | [lap]` |
| Unit rule | `Fixed(s)` |
| Status | implemented |
| Arguments | 0 |

Example:

```
result = lap_time()
```

#### sector_time

```
sector_time(i)
```

Duration of the lap's `i`-th sector (0-based, as `sector_number()` counts) in seconds; NaN when the lap has no such sector.

| | |
|---|---|
| Shape | `scalar | [lap]` |
| Unit rule | `Fixed(s)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = sector_time(1)
```

### Lap delta

#### lap_delta_time

```
lap_delta_time(ch)
```

Main lap minus overlay lap, time-matched; the mean across every overlay when more than one is selected. Retired name: `variance_time`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = lap_delta_time([Speed])
```

#### lap_delta_dist

```
lap_delta_dist(ch)
```

Main lap minus overlay lap, arc-length-matched. Retired name: `variance_dist`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = lap_delta_dist([Speed])
```

### Estimator (diagnostic)

#### attitude

```
attitude("roll"|"pitch")
```

Bike attitude estimated by the AHRS filter, in degrees.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(deg)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = attitude("roll")
```

#### body_accel

```
body_accel("long"|"lat")
```

Gravity-compensated body-frame acceleration, in g.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(g)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = body_accel("long")
```

### Estimator

#### wheel_travel

```
wheel_travel("front"|"rear")
```

Suspension travel at the named wheel, in mm.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(mm)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = wheel_travel("front")
```

#### wheel_velocity

```
wheel_velocity("front"|"rear")
```

Suspension velocity at the named wheel, in mm/s.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(mm/s)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = wheel_velocity("front")
```

### Vector

#### vec

```
vec(x, y, z)
```

Builds a Vec3 from three components, which must share one unit.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `AllMatch(0, 1, 2)` |
| Status | implemented |
| Arguments | 3 |

Example:

```
result = vec([AccelX], [AccelY], [AccelZ])
```

#### vx

```
vx(v)
```

The x component of a Vec3.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = vx(accel)
```

#### vy

```
vy(v)
```

The y component of a Vec3.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = vy(accel)
```

#### vz

```
vz(v)
```

The z component of a Vec3.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = vz(accel)
```

#### vadd

```
vadd(a, b)
```

Component-wise sum of two Vec3s.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `AllMatch(0, 1)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = vadd(a, b)
```

#### vsub

```
vsub(a, b)
```

Component-wise difference of two Vec3s.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `AllMatch(0, 1)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = vsub(a, b)
```

#### vscale

```
vscale(v, s)
```

Scales a Vec3 by a scalar.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `Product(0, 1)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = vscale(accel, 9.81)
```

#### cross

```
cross(a, b)
```

Cross product of two Vec3s.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `Product(0, 1)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = cross(a, b)
```

#### dot

```
dot(a, b)
```

Dot product of two Vec3s.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Product(0, 1)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = dot(a, b)
```

#### norm

```
norm(v)
```

Euclidean length of a Vec3.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = norm(accel)
```

#### normalize

```
normalize(v)
```

Unit vector in the same direction as `v`.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `Dimensionless` |
| Status | implemented |
| Arguments | 1 |

Example:

```
result = normalize(accel)
```

#### angle_between

```
angle_between(a, b)
```

Angle between two Vec3s, in radians, in [0, pi]. Retired name: `angle`.

| | |
|---|---|
| Shape | `[t]` |
| Unit rule | `Fixed(rad)` |
| Status | implemented |
| Arguments | 2 |

Example:

```
result = angle_between(a, b)
```

### Rotation

#### rotate_mat

```
rotate_mat(v, m00..m22) (row-major, scalar entries)
```

Rotates a Vec3 by a row-major 3x3 matrix.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 10 |

Example:

```
result = rotate_mat(a, 1, 0, 0, 0, 1, 0, 0, 0, 1)
```

#### rotate_axis

```
rotate_axis(v, ax, ay, az, angle) (scalars; `angle` in radians)
```

Rotates a Vec3 about an axis by an angle in radians.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 5 |

Example:

```
result = rotate_axis(a, 0, 0, 1, 1.5708)
```

#### rotate_euler

```
rotate_euler(v, roll, pitch, yaw) (radians; the angles may be channels)
```

Rotates a Vec3 by roll/pitch/yaw in radians; the angles may be channels, giving a per-sample rotation.

| | |
|---|---|
| Shape | `Vec3` |
| Unit rule | `SameAsArg(0)` |
| Status | implemented |
| Arguments | 4 |

Example:

```
result = rotate_euler(a, [Roll], [Pitch], [Yaw])
```

## Retired names

Names the language no longer uses, and what replaced them. A `version: 3`
workbook is migrated on read and rewritten on save; a `version: 4`
workbook using one is a typed error naming its replacement (C2 §3.8).

| Retired | Replacement |
|---|---|
| `variance_time` | `lap_delta_time` |
| `variance_dist` | `lap_delta_dist` |
| `fft` | `periodogram` |
| `angle` | `angle_between` |
| `p` | `percentile` |
| `clamp` | `clip` |
| `if` | `where` |
| `integrate` | `cumulative_trapezoid` |
| `hilbert` | `envelope` |

## Definitions and annotations

A `math` cell's body is a sequence of lines. Blank lines and `#` comments are
ignored; everything else is either a constant or a definition.

```
const sag_target = 0.3          # a workbook-wide constant
fork_velocity = differentiate([Fork])   # a definition
```

A definition's name on the left of `=` must be a valid identifier —
`[A-Za-z_][A-Za-z0-9_]*` — because every definition is also bound as a
JavaScript variable of that name inside `js` cells. `const` is reserved and
cannot be a definition name.

A `[Channel Name]` reference on the right of `=` is captured verbatim, spaces
included, so a raw session channel whose name has a space is still reachable.
A `[Name]` naming another definition must match that definition's identifier
exactly.

### The three annotations

A trailing `#` comment is an ordinary comment with three exceptions. Each is
the literal word, a colon, then free text to the end of the line.

| Annotation | What it does |
|---|---|
| `# label: <text>` | The display name. What the channel list, chart legends and Properties controls show instead of the identifier. Without one, the UI falls back to the identifier itself. |
| `# unit: <text>` | Overrides the inferred unit for this definition. Use it when the engine's inference cannot reach a unit it should have — never to relabel a value whose real unit is different, because a wrong unit is worse than none. |
| `# shape: <text>` | Declares the value's shape when the expression's own shape is ambiguous. |

```
fork_velocity = differentiate([Fork])   # label: Fork velocity
compression = -fork_velocity            # label: Compression  # unit: mm/s
```

### Constants

Constants live in one flat, workbook-wide namespace fed from two places: the
front matter's `constants:` map, and `const name = value` lines in any `math`
cell. A name declared in both is a per-cell error, not a silent win for either.

A `const` line's name must be an identifier. A front-matter constant's name
need not be — it is only ever read from JavaScript as `constants["rider mass"]`,
never as a bare variable — so a spaced or free-text name is expressible in YAML
only.

A front-matter constant may carry a unit as a string: `rider_mass_kg: "82 kg"`
evaluates to `82`, and the `kg` is display metadata. It is never dimensionally
checked and never converted.

### Units, and the three states

Every value the engine produces carries one of three unit states, and the
distinction is deliberate: a value that *has* no unit and a value whose unit
could not be worked out are different claims.

| State | What it means |
|---|---|
| `known` | The unit was inferred (or declared with `# unit:`) and is shown. |
| `dimensionless` | The value genuinely has no unit — a count, a ratio, a sign. |
| `unknown` | The unit could not be inferred. A CSV channel with no declared unit, or `pow(x, [n])` with a non-literal exponent. |

Nothing is ever labelled with a guess. A prose `${…}` span does **not**
auto-append a unit either, because the span is an arbitrary expression whose
value may not be the quantity the unit describes — write
`${peak_travel} ${channel("fork").unit}` when you want it.

## Windows and laps

Everything a workbook evaluates is evaluated over a **selection**, and a
selection is a list of *windows*. A window is one span of one session. Two
windows over the same channel is the ordinary case, not the exotic one: it is
how you compare one lap against another.

### What a window changes

An aggregate (`mean`, `rms`, `percentile`, `count`) is computed over the
selected span, not the whole session. A per-sample function is computed over
the samples inside it. Nothing about a definition's text changes when the
selection changes — a host variable is keyed by the definition alone, never by
which windows are selected, so a cell's code never depends on what is currently
clicked.

### Several windows in one payload

When more than one window is selected, `channel(name)` returns *every* selected
window's samples in one payload rather than one payload per window:

- the records are each window's own samples, window after window, in window
  order.
- `w[i]` is the index into `windows` naming which window produced sample `i`.
- `windows[j]` is that window's `{ sessionId, span, colour, label }`. The
  `colour` is a `--chart-1`…`--chart-8` token, which is how a per-window colour
  reaches a mark.

Exactly one `NaN` record (`t = v = w = NaN`) is inserted between each adjacent
pair of windows. Observable Plot breaks a line mark at a `NaN`, so a cell that
destructures only `{t, v}` and knows nothing about `w` still draws *n* separate
segments instead of one line vaulting from one window's last sample to the
next window's first.

A single selected window is byte-identical to having no window concept at all:
`w` is all zeros and `windows` has one entry.

### Laps

`laps` is the active session's lap table — `{ number, startT, endT }[]`, with
`number` 1-based.

The lap functions divide into three groups:

- **Where am I?** `current_lap()` gives the 1-based lap number at each sample
  and `0` outside any lap; `sector_number()` gives the 0-based sector index and
  NaN outside any sector. Both are per-sample channels at the session's base
  rate, so both are usable as a `where(...)` condition over a channel at that
  same rate. A slower channel — GPS at 1 Hz, say — has to be brought onto the
  base rate first, because nothing resamples implicitly:

  ```
  where(sector_number() == 3, resample([GPS_SpeedKmh], [Time]), 0)
  ```

  `[Time]` is the synthesized base-rate channel, so the resampled speed lands
  on the axis `sector_number()` counts along. Written without the `resample`,
  the same expression is a typed error naming the two rates — never a wrong
  answer.
- **Where does a lap start?** `lap_start_time(n)` and `lap_start_distance(n)`
  are scalars, NaN when `n` is out of range (and `lap_start_distance` is also
  NaN when the session has no `[Distance]` channel).
- **How does this lap differ?** `lap_delta_time(ch)` and `lap_delta_dist(ch)`
  subtract an overlay lap from the main lap — time-matched and
  arc-length-matched respectively — and take the mean across every overlay when
  more than one is selected. Both carry `ch`'s own units: they are a difference
  of two `[ch]` series, not a time or a distance.

`channel(name, { lap: n })` fixes a lookup to one lap, and
`channel(name, { from: a, to: b })` to a stretch of session seconds; see
[The data a chart gets](#the-data-a-chart-gets). A different session is
reached by selecting its window, not from cell code.

### The lap-relative time axis

A time chart's x binding is either `"t"` (session time) or `"tr"` (time
relative to the start of each window). Use `"tr"` when comparing laps: session
time puts two laps a minute apart on the x axis, which is never what a lap
comparison wants. The binding is stored per mark and edited per plot.

## JavaScript host variables

A `js` cell is an Observable Runtime cell. The host injects the variables
below into the module scope before any cell runs. None of them crosses IPC
from inside cell code — the host resolves every one ahead of time and feeds
the results in, which is why a cell never awaits the engine.

| Variable | Shape |
|---|---|
| one per `math` definition, **by name** | For a time-axis definition, the same records `channel(name)` returns, over the whole selected window and decimated to a few thousand points. A rank-0 value binds a bare number; a rank-1 value on some other axis keys its records by that axis (`lap`, `f`) instead of `t`; a rank ≥ 2 value binds `{ shape, axes, v }`. |
| `channel(name, opts?)` | An array of `{ t, v, tr, w }` records, one per sample, plus a `windows` descriptor. `opts` is a scope: `{ lap: n }` or `{ from: a, to: b }` (session seconds, number literals). See [The data a chart gets](#the-data-a-chart-gets). |
| `spectrum(name, params)` | The FFT payload for a channel, keyed by `spectrumKey(channelId, fftParams)`. |
| `histogram(name, params)` | The binned payload for a channel. |
| `scatter(xChannel, yChannel, params)` | The point cloud for a channel pair. |
| `laps` | `{ number, startT, endT }[]` for the active session. |
| `session` | `{ id, name?, timestampUtcMs }`. |
| `constants` | `{ [name]: number }` — front matter and every `const` line, flattened. Keys may contain spaces. (Before 2026-11-05 this was always empty.) |
| `Plot` | `@observablehq/plot` 0.6.17, bundled. |
| `d3` | `d3` 7.9.0, bundled. |
| `Inputs` | `@observablehq/inputs` 0.12.0, bundled. |
| `html` | The standard tagged-template helper. |

Every library is bundled with the app. There is no CDN, ever, so a cell that
reaches for a URL will not load.

### The data a chart gets

`channel(name)` returns an array of records, one per sample:

| Field | Meaning |
|---|---|
| `t` | Seconds since the session's first sample. |
| `v` | The value, in its own unit. |
| `tr` | Seconds since this sample's own window began. |
| `w` | Index into `windows`: which selected window the sample came from. |

A mark addresses the fields by name:

```js
Plot.plot({
  marks: [Plot.lineY(channel("front_travel"), { x: "t", y: "v" })]
})
```

**It is decimated for the chart.** A chart never gets every sample of a long
window: the data is planned for the chart's span and width, a min/max envelope
of about two points per pixel column, so every spike survives but a 0.1 s event
in a whole ride is a few pixels wide. Zooming the chart (or narrowing the
selected window on the timeline) fetches the finer detail for the new span.

**A scope fixes a call's span.** `channel(name, { from: a, to: b })` covers
session seconds `a` to `b`, and `channel(name, { lap: n })` covers lap `n`.
That data is planned for that span at the chart's width, so a short range
arrives at full detail, and it stays put when the chart is panned or zoomed.
The numbers must be literals: the host reads them before the cell runs. Two
calls over different ranges of one channel are two separate pieces of data.

```js
const zone = channel("front_v", { from: 5827.3, to: 5829.8 });
return Plot.plot({ marks: [Plot.lineY(zone, { x: "t", y: "v" })] });
```

A definition is still **evaluated** over the whole selected window; a scope
only chooses which part of the result is sent. A definition that reads data
outside the range (a calibration stretch elsewhere in the ride) is unaffected.

**Event lists come from the maths, not the chart.** A list of moments (every
pulse gap over a threshold, every stop) is a number result, so compute it with
`keep`, which returns only the matching samples as a short channel; it arrives
exact, whatever the decimation:

```
breaks = keep([gap_ratio], [gap_ratio] > 1.8)
```

The records are built from the bytes the engine transferred; treat them as
read-only.

### Units on a channel

A `channel(...)` result carries its unit as two **non-enumerable** properties,
so neither appears in a `for...of` over the samples nor in Plot's column
inference:

| Property | Meaning |
|---|---|
| `.unit` | The unit as text, ready to splice into prose. Empty when the value is dimensionless. |
| `.unitState` | `"known"`, `"dimensionless"` or `"unknown"`. |

The unit comes from whichever source resolved the value — a definition's own
result, or a raw channel's recorded unit. It never comes from the transferred
sample bytes, which carry samples and framing only.

### Colouring by window

`w` is an index into `windows`, so a multi-window chart colours itself by
stroking on `"w"` and building the range from each window's own colour token:

```js
Plot.plot({
  color: { legend: true },
  marks: [Plot.lineY(channel("front_travel"), { x: "tr", y: "v", stroke: "w" })]
})
```

## The chart grammar (`plotForm`)

The Properties panel does not own a chart's configuration — the `js` cell's own
code does. The panel reads the code, and writing a control writes the code
back. That round-trip works over one narrow subset of `Plot.plot(...)`, and a
cell outside the subset is shown as **Custom**: it still renders, it simply
stops being editable by control.

Staying inside the subset is worth it while a chart is one of the four kinds
below; step outside it deliberately, for something the panel could never offer.

### The shape of an editable cell

```js
Plot.plot({
  title: "Fork travel",
  x: { label: "Time (s)", domain: [0, 60], type: "linear" },
  y: { label: "Travel (mm)", domain: [0, 160], type: "linear" },
  color: { legend: true },
  marks: [ /* one of the four mark sets below */ ]
})
```

Every option is optional, but each one that is present must take exactly one of
the forms listed. `x.type` is `"linear"` or `"log"`. `y.type` adds `"sqrt"` and
`"pow"`; `"pow"` alone may carry an `exponent`.

### The four mark sets

A cell's `marks` array is one of these four, never a mixture.

**Time.** One or more marks over `channel(...)`, optionally led by a zero rule.

```js
marks: [
  Plot.ruleY([0]),
  Plot.lineY(channel("front_travel"), { x: "tr", y: "v", stroke: "#e07a3f", strokeWidth: 1.5 })
]
```

Mark names: `lineY`, `dot`, `areaY`, `rectY`, `ruleY`. The x binding is `"t"`
(session time) or `"tr"` (relative to each window's start); `y` is always
`"v"`. A `channel(...)` call may carry `{ lap: n }`.

There is no distance binding, by design.

**Spectrum.** Exactly one mark over `spectrum(...)`, with all six parameters
present in this order:

```js
marks: [
  Plot.lineY(spectrum("front_travel", {
    windowSize: 1024, hopSize: 512, window: "hann",
    detrend: "mean", scaling: "density", averaging: "mean"
  }), { x: "f", y: "m" })
]
```

`windowSize` and `hopSize` are integers or `"all"`. `window` is
`"rectangular" | "hann" | "hamming"`. `detrend` is `"none" | "mean" | "linear"`.
`scaling` is `"density" | "spectrum" | "raw_magnitude"` — the maths language's
own three names. `averaging` is `"none" | "mean" | "median" | "max"`.

The older spelling `"magnitude"` is still read, so a cell written before the
rename parses and round-trips; it is never offered by a picker and never newly
written.

**Histogram.** Exactly one `Plot.rectY` over `histogram(...)`, four parameters,
fixed order:

```js
marks: [
  Plot.rectY(histogram("fork_velocity", {
    binMode: "width", binValue: 25, symmetric: true, normalise: "fraction"
  }), { x1: "v0", x2: "v1", y: "n" })
]
```

`binMode` is `"count"` (a bin count) or `"width"` (a bin width in the channel's
units). `normalise` is `"counts"` or `"fraction"`.

**Scatter.** Exactly one `Plot.dot` over `scatter(...)`, two parameters:

```js
marks: [
  Plot.dot(scatter("accel_lat", "accel_long", {
    pointBudget: 20000, equalAspect: true
  }), { x: "x", y: "y" })
]
```

`equalAspect` squares the domain, which is what a G-G plot wants. When several
windows are shown, they share one squared domain rather than each computing its
own.

### What makes a cell Custom

Anything the grammar above does not list: a mark name outside the set, a mixed
`marks` array, an option the subset has no production for, a computed value
where a literal is required, or any code around the `Plot.plot(...)` call. The
round-trip is byte-identical for code the panel itself wrote, so a cell that
goes Custom did so because of an edit, not because of a reformat.

## Lap tables and progression

Two pictures answer "how did each lap compare?": a **lap table**, one row per
lap and one column per measurement, and a **lap progression**, the same
measurement drawn against lap number. Both are built on one idea — a value
with one entry per lap rather than one per sample.

### Per-lap values

A definition's value carries a shape. Most are a series over time, written
`[t]` — one number per sample, each keeping its own recorded timestamp. A
per-lap value is written `[lap]`: one number per lap, and its coordinate is
the 1-based lap number, not a time.

Three builtins read the lap a value belongs to:

| Function | In a table row | In a `math` cell |
|---|---|---|
| `lap_time()` | that row's lap time, seconds | a `[lap]` value — every lap's time |
| `sector_time(i)` | that row's time in sector `i`, **0-based** | a `[lap]` value — every lap's sector `i` |
| `lap_number()` | that row's 1-based lap number | a `[lap]` value of the lap numbers |

`lap_time()` reads the recorded lap time, so a neutralised zone that was
subtracted when the session was recorded stays subtracted here.

Reducing a `[t]` series per lap, `mean([peak_freq], "t:lap")`, is planned but
not supported yet: the engine rejects it with `expected scalar argument, got
string`. Today a `[lap]` value comes from the three builtins above:

```
lap_time_s = lap_time()                         # shape: [lap]
```

For one lap's reduction headless, window the evaluation with `idl-rs workbook
data --track <track> --main-lap <n>`.

Combining a `[lap]` value with a `[t]` one is an error naming both shapes —
lap 3 and second 3 are not the same coordinate, and the alternative to the
error is a plausible-looking wrong answer.

### Drawing a lap progression

A lap chart is its own cell kind. Its single mark takes a **bare identifier**
— a `[lap]` definition — never a `channel(...)` call, because a per-lap value
is always something a definition computed:

```js
Plot.plot({
  x: { label: "Lap" },
  y: { label: "Lap time (s)" },
  marks: [ Plot.lineY(lap_time_s, { x: "lap", y: "v", z: "w" }) ]
})
```

`x` binds `"lap"`, never `"t"`: a `[lap]` value reaches JavaScript keyed by
its own axis, as `{ lap, v, w }`. A cell that asked for `"t"` would find
nothing rather than plot lap 3 at three seconds.

`z: "w"` separates the series by window, so selecting three laps of two
sessions draws one line per window in that window's own colour. Lap numbers
repeating across sessions is correct — x is an ordinal, and `w` keeps the
lines apart. There is no decimation and no cap: laps per session are tens.

Naming a definition that is not `[lap]`-shaped is reported on the cell, with
both shapes named and the reduction to write instead.

### The lap table

A lap table is a `table` cell, not a chart. Its rows follow the selection:

```json
{
  "rowSource": "windowLaps",
  "mainRowId": "fastest",
  "columns": [
    { "id": "c0", "name": "Lap time", "template": "lap_time()" },
    { "id": "c1", "name": "Sector 1", "template": "sector_time(0)" },
    { "id": "c2", "name": "Fork max", "template": "max([Fork travel])" }
  ],
  "rows": [],
  "cells": []
}
```

`rowSource: "windowLaps"` derives one row per lap of each selected window, in
window order then lap order, and each row evaluates **in its own lap** — so
`max([Fork travel])` in row 3 is lap 3's maximum, not the session's. Authored
`rows` are ignored while it is set, and every cell of a derived row evaluates
its column's `template`; a column with no template renders empty.

`mainRowId` names the row `main({col[]})` compares against. The literal
`"fastest"` is reserved for derived rows: the row with the smallest recorded
`lap_time()`, skipping any lap with no recorded time, ties going to the first.
It is highlighted in the grid. Under `rowSource: "authored"` the word is a
validation error rather than a silently unset baseline — an authored table's
rows are named, so a magic id there would shadow a real row id.

Column headers show the column's name and, where it can be stated, its unit:
`lap_time()` and `sector_time(i)` are seconds. A column whose template is an
arbitrary expression shows no unit rather than a guessed one.

A cell that fails shows its own error and nothing else in the grid changes —
one lap with no samples never blanks the other laps' numbers.

### What is not here yet

Cross-session comparison is a selection capability, not a grammar one:
nothing in either form above changes when it lands. Rank-2 values (a
spectrogram matrix) have no lap form — reduce them to a series or a per-lap
value first.

## Naming policy

Where a function is semantically equivalent to a scipy or numpy one, it takes
that name. Where it is **not** equivalent, it takes a deliberately different
name. A familiar name with unfamiliar behaviour is worse than an invented one,
because an invented name makes a reader check and a false friend does not.

That is why the retired-name table above exists, and why two known false
friends were left in place rather than renamed:

- **`butter`** designs *and* applies its filter, zero-phase, in one call.
  scipy's `butter` only designs one; applying it is a separate call. Splitting
  the two here is a real behaviour change with its own migration, not a naming
  fix, so it is recorded rather than done.
- **`round`** rounds half away from zero (`2.5` gives `3`), where `numpy.round`
  rounds half to even (`2.5` gives `2`). Changing it would silently move every
  existing value sitting exactly on a halfway point.

Neither is a promise that either name changes soon. Both are documented in
their own entries above.
