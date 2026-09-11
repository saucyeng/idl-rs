//! Static catalog of every math-expression builtin function, for
//! `list_math_builtins` (C3 §3.4, lead ruling R64.2,
//! `runs/2026-09-03/decisions.md`). Hand transcribed from C2 §3.3's
//! "Builtin catalog" table
//! (`docs/superpowers/specs/2026-09-03-idl1-c2-workbook-v3.md`) — the same
//! source L6's `functionCatalog.ts` was transcribed from — because
//! `eval::call_function` is a dispatch `match` with no attached metadata to
//! extract mechanically; there is nothing else to transcribe from.
//!
//! `arity` is transcribed from C2 §3.3's Signature column: the set of valid
//! argument counts for that name (more than one entry when a name has more
//! than one call form, e.g. `rms(ch)` / `rms(ch, w)` → `&[1, 2]`). `status`
//! mirrors C2 §3.3's Status column collapsed to R64.2's two-value wire
//! enum: a function whose *only* unimplemented form is a secondary arity
//! (`median`'s 2-arg rolling form) is still [`MathBuiltinStatus::Implemented`]
//! here, because its 1-arg form dispatches to real logic.
//!
//! Excludes `main(col[])` (table-cell only, documented in C2 §4, not §3.3)
//! and the grammar keywords `and`/`or`/`not` (parsed as operators in
//! `math::parse`, never reach `call_function`'s dispatch at all) — the same
//! three exclusions L6's `functionCatalog.ts` transcription already made,
//! per that file's own doc comment. C2 §3.3 originally stated 69 named
//! functions total (63 `Implemented` + 6 `NotImplemented`); the
//! scipy-alignment lane (`runs/2026-09-08/scipy-alignment-plan.md`) adds
//! three entries without removing any — `fft` split into `periodogram` +
//! `welch` (task 6), `cumtrapz` added as `cumulative_trapezoid`'s permanent
//! second spelling (task 10, R151 item 6), and `gradient` added alongside
//! `differentiate` (task 11, R151 item 3) — so this catalog now has 72
//! entries; C2 §3.3 itself needs the matching row-count update
//! (spec-during, not done in this lane's Rust-only scope).
//!
//! R64.2 requires this catalog's *name* set be checked against
//! `eval::call_function`'s real dispatch table, "never a second hand
//! copy." `call_function` itself is private to `eval.rs` and its match
//! arms aren't enumerable from outside that file without literally
//! retyping every arm's pattern — which would be exactly the "second hand
//! copy" the ruling forbids. This module's own `#[cfg(test)]` below
//! resolves that by calling [`crate::math::evaluate`] (the crate's public
//! parse+eval entry point, which reaches `call_function`) with `"{name}()"`
//! for every catalog entry, and asserting the failure is never
//! [`MathEvalErrorKind::UnknownFunction`]. That proves every catalog name
//! really dispatches; it can't prove the converse (that `call_function`
//! dispatches no *other* name) without the forbidden second copy — flagged
//! in L8w Task 12b's report as a judgment call for the lead to confirm.

/// Whether a catalog entry's function is implemented in
/// `math::eval::call_function` today. Mirrors C2 §3.3's Status column,
/// collapsed to the two-value split that column's own text totals (63
/// Implemented / 6 NotImplemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBuiltinStatus {
    /// The function's documented call form(s) all dispatch to real logic.
    Implemented,
    /// The function parses and validates but its match arm returns
    /// `MathEvalErrorKind::NotImplemented` (C2 §3.3's committed-but-deferred
    /// surface, e.g. `spectrogram`, `envelope`).
    NotImplemented,
}

/// One math builtin's catalog metadata: its name, the set of argument
/// counts C2 §3.3's signature documents as valid, whether it is
/// implemented, and (added 2026-09-11, ruling R222 item 1) the
/// documentation columns the generated workbook reference and the
/// notebook editor's hover card are both rendered from. See the module
/// doc for provenance and the R64.2 ruling this exists for.
///
/// The six documentation fields are transcribed from the same C2 §3.3
/// table row as `arity`/`status` — `signature` and `category` verbatim
/// from those columns, `unit_rule` from the "Unit rule (§3.3.1)" column
/// with its explanatory tail dropped, `shape` from the signature's own
/// result annotation cross-checked against §3.6, and `description` and
/// `example` written here (no column carries either). One source, two
/// consumers: `idl-rs docs workbook` renders them into
/// `docs/WORKBOOK-REFERENCE.md` and `list_math_builtins` puts the same
/// strings on the wire for the editor's hover, so the reference and the
/// tooltip can never disagree.
#[derive(Debug, Clone, Copy)]
pub struct MathBuiltin {
    /// The function name as written in a `math` cell expression.
    pub name: &'static str,
    /// Valid argument counts (more than one entry when C2 §3.3's signature
    /// documents multiple call forms).
    pub arity: &'static [u32],
    /// Whether the function is implemented (see [`MathBuiltinStatus`]).
    pub status: MathBuiltinStatus,
    /// C2 §3.3's Category column, verbatim — the grouping heading the
    /// generated reference sorts entries under.
    pub category: &'static str,
    /// C2 §3.3's Signature column, verbatim: the call form(s) as a human
    /// reads them, `|`-separated when there is more than one.
    pub signature: &'static str,
    /// C2 §3.3's "Unit rule (§3.3.1)" column, the rule expression only
    /// (the column's prose explanation is not carried).
    pub unit_rule: &'static str,
    /// The result's value shape in §3.6's vocabulary: `scalar`, `[t]`
    /// (a time-axis channel), `[f]`, `[t,f]`, `Vec3`, or a `|`-separated
    /// pair when the shape depends on the call form.
    pub shape: &'static str,
    /// One line saying what the function computes. Written for this
    /// catalog; no C2 column carries it.
    pub description: &'static str,
    /// One runnable example call, as it would be typed on the right of a
    /// `math` definition's `=`. Written for this catalog.
    pub example: &'static str,
}

/// Returns the full 72-entry math builtin catalog (C2 §3.3, minus
/// `main(col[])` and the `and`/`or`/`not` grammar keywords — see module
/// doc). Order matches C2 §3.3's table row order.
pub fn math_builtin_catalog() -> &'static [MathBuiltin] {
    use MathBuiltinStatus::{Implemented as I, NotImplemented as N};
    &[
        MathBuiltin {
            name: "butter",
            arity: &[4],
            status: I,
            category: "Filter",
            signature: "butter(order, cutoff_hz, \"low\"|\"lowpass\"|\"high\"|\"highpass\", ch)",
            unit_rule: "SameAsArg(3)",
            shape: "[t]",
            description: "Zero-phase Butterworth low- or high-pass filter; designs and applies the filter in one call.",
            example: "butter(2, 5, \"low\", [Fork])",
        },
        MathBuiltin {
            name: "sosfilt",
            arity: &[2],
            status: N,
            category: "Filter",
            signature: "sosfilt(sos, ch)",
            unit_rule: "SameAsArg(1)",
            shape: "[t]",
            description: "Applies a second-order-section filter to a channel. Not implemented.",
            example: "sosfilt(sos, [Fork])",
        },
        MathBuiltin {
            name: "declip",
            arity: &[1],
            status: I,
            category: "Reconstruction",
            signature: "declip(ch)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Reconstructs samples lost to clipping; designed for accelerometer data clipped at +/-32 g.",
            example: "declip([AccelZ])",
        },
        MathBuiltin {
            name: "cumulative_trapezoid",
            arity: &[1],
            status: I,
            category: "Time-domain",
            signature: "cumulative_trapezoid(ch)",
            unit_rule: "Product(SameAsArg(0), Fixed(s))",
            shape: "[t]",
            description: "Cumulative trapezoidal integral of a channel with respect to time.",
            example: "cumulative_trapezoid([AccelZ])",
        },
        MathBuiltin {
            name: "cumtrapz",
            arity: &[1],
            status: I,
            category: "Time-domain",
            signature: "cumtrapz(ch)",
            unit_rule: "Product(SameAsArg(0), Fixed(s))",
            shape: "[t]",
            description: "A permanent second spelling of `cumulative_trapezoid`, as scipy itself carries both.",
            example: "cumtrapz([AccelZ])",
        },
        MathBuiltin {
            name: "differentiate",
            arity: &[1],
            status: I,
            category: "Time-domain",
            signature: "differentiate(ch)",
            unit_rule: "Quotient(SameAsArg(0), Fixed(s))",
            shape: "[t]",
            description: "Backward difference with respect to time, with `result[0] = 0`. Deliberately not `gradient`'s central difference.",
            example: "differentiate([Fork])",
        },
        MathBuiltin {
            name: "gradient",
            arity: &[1],
            status: I,
            category: "Time-domain",
            signature: "gradient(ch)",
            unit_rule: "Quotient(SameAsArg(0), Fixed(s))",
            shape: "[t]",
            description: "Central difference with respect to time, `numpy.gradient`'s own formula.",
            example: "gradient([Fork])",
        },
        MathBuiltin {
            name: "detrend",
            arity: &[1, 2],
            status: I,
            category: "Time-domain",
            signature: "detrend(ch) | detrend(ch, \"linear\"|\"constant\"|\"mean\"|\"none\")",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Removes a constant or linear trend from a channel.",
            example: "detrend([Fork], \"linear\")",
        },
        MathBuiltin {
            name: "rms",
            arity: &[1, 2],
            status: I,
            category: "Time-domain / aggregate",
            signature: "rms(ch) -> scalar | rms(ch, w) -> rolling channel, `w` a window in samples",
            unit_rule: "SameAsArg(0)",
            shape: "scalar | [t]",
            description: "Root mean square over the whole window, or rolling over `w` samples.",
            example: "rms([Fork], 64)",
        },
        MathBuiltin {
            name: "mean",
            arity: &[1, 2],
            status: I,
            category: "Time-domain / aggregate",
            signature: "mean(ch) -> scalar | mean(ch, w) -> rolling channel",
            unit_rule: "SameAsArg(0)",
            shape: "scalar | [t]",
            description: "Arithmetic mean over the whole window, or rolling over `w` samples.",
            example: "mean([Fork], 64)",
        },
        MathBuiltin {
            name: "std",
            arity: &[1, 2],
            status: I,
            category: "Time-domain / aggregate",
            signature: "std(ch) -> scalar (population sigma) | std(ch, w) -> rolling channel",
            unit_rule: "SameAsArg(0)",
            shape: "scalar | [t]",
            description: "Population standard deviation over the whole window, or rolling over `w` samples.",
            example: "std([Fork])",
        },
        MathBuiltin {
            name: "median",
            arity: &[1],
            status: I,
            category: "Aggregate",
            signature: "median(ch) -> scalar",
            unit_rule: "SameAsArg(0)",
            shape: "scalar",
            description: "Median of every sample in the window. The two-argument rolling form is not implemented.",
            example: "median([Fork])",
        },
        MathBuiltin {
            name: "sum",
            arity: &[1],
            status: I,
            category: "Aggregate",
            signature: "sum(ch) -> scalar",
            unit_rule: "SameAsArg(0)",
            shape: "scalar",
            description: "Raw sum of every sample in the window, not time-normalised.",
            example: "sum([Fork])",
        },
        MathBuiltin {
            name: "count",
            arity: &[1],
            status: I,
            category: "Aggregate",
            signature: "count(ch) -> scalar",
            unit_rule: "Dimensionless",
            shape: "scalar",
            description: "Number of samples in the window.",
            example: "count([Fork])",
        },
        MathBuiltin {
            name: "first",
            arity: &[1],
            status: I,
            category: "Aggregate",
            signature: "first(ch) -> scalar",
            unit_rule: "SameAsArg(0)",
            shape: "scalar",
            description: "First sample in the window.",
            example: "first([Speed])",
        },
        MathBuiltin {
            name: "last",
            arity: &[1],
            status: I,
            category: "Aggregate",
            signature: "last(ch) -> scalar",
            unit_rule: "SameAsArg(0)",
            shape: "scalar",
            description: "Last sample in the window.",
            example: "last([Speed])",
        },
        MathBuiltin {
            name: "percentile",
            arity: &[2],
            status: I,
            category: "Aggregate",
            signature: "percentile(ch, quantile) -> scalar, `quantile` in [0, 100]",
            unit_rule: "SameAsArg(0)",
            shape: "scalar",
            description: "Value below which `quantile` percent of the samples fall. Retired name: `p`.",
            example: "percentile([Fork], 95)",
        },
        MathBuiltin {
            name: "abs",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "abs(x)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Absolute value, sample by sample.",
            example: "abs([AccelY])",
        },
        MathBuiltin {
            name: "sqrt",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "sqrt(x)",
            unit_rule: "PowN(0, 1/2)",
            shape: "[t]",
            description: "Square root, sample by sample; every unit exponent is halved.",
            example: "sqrt([Power])",
        },
        MathBuiltin {
            name: "sign",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "sign(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "-1, 0 or 1 by sign; NaN stays NaN.",
            example: "sign(differentiate([Fork]))",
        },
        MathBuiltin {
            name: "floor",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "floor(x)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Rounds each sample down to the nearest integer.",
            example: "floor([Speed])",
        },
        MathBuiltin {
            name: "ceil",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "ceil(x)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Rounds each sample up to the nearest integer.",
            example: "ceil([Speed])",
        },
        MathBuiltin {
            name: "round",
            arity: &[1],
            status: I,
            category: "Elementwise",
            signature: "round(x)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Rounds half away from zero (`2.5` gives `3`), not `numpy.round`'s banker's rounding.",
            example: "round([Speed])",
        },
        MathBuiltin {
            name: "pow",
            arity: &[2],
            status: I,
            category: "Elementwise",
            signature: "pow(x, y)",
            unit_rule: "PowN(0, y) for a literal `y`; otherwise Dimensionless or Unknown",
            shape: "[t]",
            description: "Raises each sample to the power `y`.",
            example: "pow([Speed], 2)",
        },
        MathBuiltin {
            name: "min",
            arity: &[1, 2],
            status: I,
            category: "Aggregate / elementwise",
            signature: "min(ch) -> scalar | min(a, b) -> elementwise",
            unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument",
            shape: "scalar | [t]",
            description: "Smallest sample in the window, or the per-sample smaller of two operands.",
            example: "min([Fork], [Shock])",
        },
        MathBuiltin {
            name: "max",
            arity: &[1, 2],
            status: I,
            category: "Aggregate / elementwise",
            signature: "max(ch) -> scalar | max(a, b) -> elementwise",
            unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument",
            shape: "scalar | [t]",
            description: "Largest sample in the window, or the per-sample larger of two operands.",
            example: "max([Fork])",
        },
        MathBuiltin {
            name: "clip",
            arity: &[3],
            status: I,
            category: "Elementwise",
            signature: "clip(ch, lo, hi)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Limits each sample to `lo`..`hi`, both given in the channel's own units. Retired name: `clamp`.",
            example: "clip([Fork], 0, 160)",
        },
        MathBuiltin {
            name: "sin",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "sin(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Sine of an angle in radians.",
            example: "sin(deg2rad([Roll]))",
        },
        MathBuiltin {
            name: "cos",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "cos(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Cosine of an angle in radians.",
            example: "cos(deg2rad([Roll]))",
        },
        MathBuiltin {
            name: "tan",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "tan(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Tangent of an angle in radians.",
            example: "tan(deg2rad([Roll]))",
        },
        MathBuiltin {
            name: "asin",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "asin(x)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Arcsine, in radians; the argument must be dimensionless.",
            example: "asin([Ratio])",
        },
        MathBuiltin {
            name: "acos",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "acos(x)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Arccosine, in radians; the argument must be dimensionless.",
            example: "acos([Ratio])",
        },
        MathBuiltin {
            name: "atan",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "atan(x)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Arctangent, in radians; the argument must be dimensionless.",
            example: "atan([Ratio])",
        },
        MathBuiltin {
            name: "atan2",
            arity: &[2],
            status: I,
            category: "Trig",
            signature: "atan2(y, x)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Quadrant-correct two-argument arctangent, in radians.",
            example: "atan2([AccelY], [AccelZ])",
        },
        MathBuiltin {
            name: "sinh",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "sinh(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Hyperbolic sine.",
            example: "sinh([Ratio])",
        },
        MathBuiltin {
            name: "cosh",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "cosh(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Hyperbolic cosine.",
            example: "cosh([Ratio])",
        },
        MathBuiltin {
            name: "tanh",
            arity: &[1],
            status: I,
            category: "Trig",
            signature: "tanh(x)",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "Hyperbolic tangent.",
            example: "tanh([Ratio])",
        },
        MathBuiltin {
            name: "deg2rad",
            arity: &[1],
            status: I,
            category: "Trig conversion",
            signature: "deg2rad(x)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Converts degrees to radians.",
            example: "deg2rad([Roll])",
        },
        MathBuiltin {
            name: "rad2deg",
            arity: &[1],
            status: I,
            category: "Trig conversion",
            signature: "rad2deg(x)",
            unit_rule: "Fixed(deg)",
            shape: "[t]",
            description: "Converts radians to degrees.",
            example: "rad2deg(atan2([AccelY], [AccelZ]))",
        },
        MathBuiltin {
            name: "periodogram",
            arity: &[1],
            status: I,
            category: "Frequency",
            signature: "periodogram(ch, window=\"boxcar\", detrend=\"constant\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
            unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")",
            shape: "[f]",
            description: "Single-segment power spectrum, scipy-named and scipy-scaled. Retired name: `fft`.",
            example: "periodogram([Fork])",
        },
        MathBuiltin {
            name: "welch",
            arity: &[1],
            status: I,
            category: "Frequency",
            signature: "welch(ch, window=\"hann\", nperseg=n, noverlap=n, detrend=\"constant\", average=\"mean\"|\"median\"|\"max\"|\"none\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
            unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")",
            shape: "[f]",
            description: "Segmented, averaged power spectrum -- the spectrum the FFT charts compute.",
            example: "welch([Fork])",
        },
        MathBuiltin {
            name: "spectrogram",
            arity: &[1],
            status: N,
            category: "Frequency",
            signature: "spectrogram(ch, window_size, hop_size, window, detrend, scaling)",
            unit_rule: "SelectByLiteral(scaling, ...) as `periodogram`, over a `[t,f]` value",
            shape: "[t,f]",
            description: "Time-varying spectrum. Not implemented; the signature and shape are fixed by C2 3.6.3.",
            example: "spectrogram([Fork])",
        },
        MathBuiltin {
            name: "envelope",
            arity: &[1],
            status: N,
            category: "Frequency",
            signature: "envelope(ch)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Amplitude envelope -- a magnitude, not scipy's complex analytic signal. Not implemented. Retired name: `hilbert`.",
            example: "envelope([Fork])",
        },
        MathBuiltin {
            name: "correlate",
            arity: &[2],
            status: N,
            category: "Correlation",
            signature: "correlate(a, b)",
            unit_rule: "Product(0, 1)",
            shape: "[t]",
            description: "Cross-correlation of two channels. Not implemented.",
            example: "correlate([Fork], [Shock])",
        },
        MathBuiltin {
            name: "convolve",
            arity: &[2],
            status: N,
            category: "Correlation",
            signature: "convolve(ch, kernel)",
            unit_rule: "Product(0, 1)",
            shape: "[t]",
            description: "Convolution of a channel with a kernel. Not implemented.",
            example: "convolve([Fork], kernel)",
        },
        MathBuiltin {
            name: "resample",
            arity: &[2],
            status: N,
            category: "Resampling",
            signature: "resample(ch, num)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Resamples a channel to `num` total samples, as `scipy.signal.resample`. `num` is a count, not a rate. Not implemented.",
            example: "resample([Fork], 4096)",
        },
        MathBuiltin {
            name: "where",
            arity: &[3],
            status: I,
            category: "Logic",
            signature: "where(cond, t, f)",
            unit_rule: "AllMatch(1, 2)",
            shape: "[t]",
            description: "Per-sample choice between two branches; `cond` may also be a scalar, selecting a whole branch. Retired name: `if`.",
            example: "where([Speed] > 10, [Fork], 0)",
        },
        MathBuiltin {
            name: "current_lap",
            arity: &[0],
            status: I,
            category: "Lap",
            signature: "current_lap()",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "1-based lap number at each sample, `0` outside any lap.",
            example: "current_lap()",
        },
        MathBuiltin {
            name: "lap_start_time",
            arity: &[1],
            status: I,
            category: "Lap",
            signature: "lap_start_time(n)",
            unit_rule: "Fixed(s)",
            shape: "scalar",
            description: "Session time at which lap `n` starts; NaN when `n` is out of range.",
            example: "lap_start_time(2)",
        },
        MathBuiltin {
            name: "lap_start_distance",
            arity: &[1],
            status: I,
            category: "Lap",
            signature: "lap_start_distance(n)",
            unit_rule: "Fixed(m)",
            shape: "scalar",
            description: "Distance at which lap `n` starts; NaN when `n` is out of range or the session has no `[Distance]`.",
            example: "lap_start_distance(2)",
        },
        MathBuiltin {
            name: "sector_number",
            arity: &[0],
            status: I,
            category: "Lap",
            signature: "sector_number()",
            unit_rule: "Dimensionless",
            shape: "[t]",
            description: "0-based sector index at each sample, NaN outside any sector.",
            example: "sector_number()",
        },
        MathBuiltin {
            name: "lap_delta_time",
            arity: &[1],
            status: I,
            category: "Lap delta",
            signature: "lap_delta_time(ch)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Main lap minus overlay lap, time-matched; the mean across every overlay when more than one is selected. Retired name: `variance_time`.",
            example: "lap_delta_time([Speed])",
        },
        MathBuiltin {
            name: "lap_delta_dist",
            arity: &[1],
            status: I,
            category: "Lap delta",
            signature: "lap_delta_dist(ch)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Main lap minus overlay lap, arc-length-matched. Retired name: `variance_dist`.",
            example: "lap_delta_dist([Speed])",
        },
        MathBuiltin {
            name: "attitude",
            arity: &[1],
            status: I,
            category: "Estimator (diagnostic)",
            signature: "attitude(\"roll\"|\"pitch\")",
            unit_rule: "Fixed(deg)",
            shape: "[t]",
            description: "Bike attitude estimated by the AHRS filter, in degrees.",
            example: "attitude(\"roll\")",
        },
        MathBuiltin {
            name: "body_accel",
            arity: &[1],
            status: I,
            category: "Estimator (diagnostic)",
            signature: "body_accel(\"long\"|\"lat\")",
            unit_rule: "Fixed(g)",
            shape: "[t]",
            description: "Gravity-compensated body-frame acceleration, in g.",
            example: "body_accel(\"long\")",
        },
        MathBuiltin {
            name: "wheel_travel",
            arity: &[1],
            status: I,
            category: "Estimator",
            signature: "wheel_travel(\"front\"|\"rear\")",
            unit_rule: "Fixed(mm)",
            shape: "[t]",
            description: "Suspension travel at the named wheel, in mm.",
            example: "wheel_travel(\"front\")",
        },
        MathBuiltin {
            name: "wheel_velocity",
            arity: &[1],
            status: I,
            category: "Estimator",
            signature: "wheel_velocity(\"front\"|\"rear\")",
            unit_rule: "Fixed(mm/s)",
            shape: "[t]",
            description: "Suspension velocity at the named wheel, in mm/s.",
            example: "wheel_velocity(\"front\")",
        },
        MathBuiltin {
            name: "vec",
            arity: &[3],
            status: I,
            category: "Vector",
            signature: "vec(x, y, z)",
            unit_rule: "AllMatch(0, 1, 2)",
            shape: "Vec3",
            description: "Builds a Vec3 from three components, which must share one unit.",
            example: "vec([AccelX], [AccelY], [AccelZ])",
        },
        MathBuiltin {
            name: "vx",
            arity: &[1],
            status: I,
            category: "Vector",
            signature: "vx(v)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "The x component of a Vec3.",
            example: "vx(accel)",
        },
        MathBuiltin {
            name: "vy",
            arity: &[1],
            status: I,
            category: "Vector",
            signature: "vy(v)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "The y component of a Vec3.",
            example: "vy(accel)",
        },
        MathBuiltin {
            name: "vz",
            arity: &[1],
            status: I,
            category: "Vector",
            signature: "vz(v)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "The z component of a Vec3.",
            example: "vz(accel)",
        },
        MathBuiltin {
            name: "vadd",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "vadd(a, b)",
            unit_rule: "AllMatch(0, 1)",
            shape: "Vec3",
            description: "Component-wise sum of two Vec3s.",
            example: "vadd(a, b)",
        },
        MathBuiltin {
            name: "vsub",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "vsub(a, b)",
            unit_rule: "AllMatch(0, 1)",
            shape: "Vec3",
            description: "Component-wise difference of two Vec3s.",
            example: "vsub(a, b)",
        },
        MathBuiltin {
            name: "vscale",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "vscale(v, s)",
            unit_rule: "Product(0, 1)",
            shape: "Vec3",
            description: "Scales a Vec3 by a scalar.",
            example: "vscale(accel, 9.81)",
        },
        MathBuiltin {
            name: "cross",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "cross(a, b)",
            unit_rule: "Product(0, 1)",
            shape: "Vec3",
            description: "Cross product of two Vec3s.",
            example: "cross(a, b)",
        },
        MathBuiltin {
            name: "dot",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "dot(a, b)",
            unit_rule: "Product(0, 1)",
            shape: "[t]",
            description: "Dot product of two Vec3s.",
            example: "dot(a, b)",
        },
        MathBuiltin {
            name: "norm",
            arity: &[1],
            status: I,
            category: "Vector",
            signature: "norm(v)",
            unit_rule: "SameAsArg(0)",
            shape: "[t]",
            description: "Euclidean length of a Vec3.",
            example: "norm(accel)",
        },
        MathBuiltin {
            name: "normalize",
            arity: &[1],
            status: I,
            category: "Vector",
            signature: "normalize(v)",
            unit_rule: "Dimensionless",
            shape: "Vec3",
            description: "Unit vector in the same direction as `v`.",
            example: "normalize(accel)",
        },
        MathBuiltin {
            name: "angle_between",
            arity: &[2],
            status: I,
            category: "Vector",
            signature: "angle_between(a, b)",
            unit_rule: "Fixed(rad)",
            shape: "[t]",
            description: "Angle between two Vec3s, in radians, in [0, pi]. Retired name: `angle`.",
            example: "angle_between(a, b)",
        },
        MathBuiltin {
            name: "rotate_mat",
            arity: &[10],
            status: I,
            category: "Rotation",
            signature: "rotate_mat(v, m00..m22) (row-major, scalar entries)",
            unit_rule: "SameAsArg(0)",
            shape: "Vec3",
            description: "Rotates a Vec3 by a row-major 3x3 matrix.",
            example: "rotate_mat(a, 1, 0, 0, 0, 1, 0, 0, 0, 1)",
        },
        MathBuiltin {
            name: "rotate_axis",
            arity: &[5],
            status: I,
            category: "Rotation",
            signature: "rotate_axis(v, ax, ay, az, angle) (scalars; `angle` in radians)",
            unit_rule: "SameAsArg(0)",
            shape: "Vec3",
            description: "Rotates a Vec3 about an axis by an angle in radians.",
            example: "rotate_axis(a, 0, 0, 1, 1.5708)",
        },
        MathBuiltin {
            name: "rotate_euler",
            arity: &[4],
            status: I,
            category: "Rotation",
            signature: "rotate_euler(v, roll, pitch, yaw) (radians; the angles may be channels)",
            unit_rule: "SameAsArg(0)",
            shape: "Vec3",
            description: "Rotates a Vec3 by roll/pitch/yaw in radians; the angles may be channels, giving a per-sample rotation.",
            example: "rotate_euler(a, [Roll], [Pitch], [Yaw])",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{evaluate, ChannelLookup, LookupChannel, MathEvalErrorKind, MathLapContext};

    /// A lookup with no channels — every catalog entry's dispatch check
    /// calls its function with zero arguments, so no channel reference is
    /// ever reached.
    struct EmptyLookup;
    impl ChannelLookup for EmptyLookup {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
    }

    #[test]
    fn math_builtin_catalog_len_is_72_after_the_scipy_alignment_lanes_splits() {
        // Arrange / Act
        let n = math_builtin_catalog().len();

        // Assert — was 69 (C2 §3.3's original stated total: 63 Implemented +
        // 6 NotImplemented); the scipy-alignment lane adds three entries
        // without removing any (`runs/2026-09-08/scipy-alignment-plan.md`):
        // `fft` split into `periodogram` + `welch` (net +1, task 6),
        // `cumtrapz` added as `cumulative_trapezoid`'s permanent second
        // spelling (net +1, task 10, R151 item 6), and `gradient` added
        // alongside `differentiate` (net +1, task 11, R151 item 3) —
        // 69 + 3 = 72.
        assert_eq!(n, 72);
    }

    #[test]
    fn implemented_and_not_implemented_counts_split_66_and_6() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let not_implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::NotImplemented).count();
        let implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::Implemented).count();

        // Assert — was 63/6; the scipy-alignment lane's three net-new
        // entries (see the length test above) are all Implemented, so only
        // that side of the split moves.
        assert_eq!(not_implemented, 6);
        assert_eq!(implemented, 66);
    }

    #[test]
    fn multi_arity_entries_have_more_than_one_valid_arity() {
        // Arrange
        let catalog = math_builtin_catalog();
        let arity_of = |name: &str| catalog.iter().find(|b| b.name == name).unwrap().arity;

        // Act / Assert
        for name in ["rms", "mean", "std", "min", "max"] {
            assert!(arity_of(name).len() > 1, "{name} should have more than one valid arity");
        }
    }

    #[test]
    fn butter_has_a_single_fixed_arity_of_four() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let butter = catalog.iter().find(|b| b.name == "butter").unwrap();

        // Assert
        assert_eq!(butter.arity, &[4]);
    }

    #[test]
    fn no_duplicate_names_in_the_catalog() {
        // Arrange
        let mut names: Vec<&str> = math_builtin_catalog().iter().map(|b| b.name).collect();
        let before = names.len();

        // Act
        names.sort_unstable();
        names.dedup();

        // Assert
        assert_eq!(names.len(), before, "duplicate name in math_builtin_catalog");
    }

    #[test]
    fn every_catalog_name_dispatches_in_call_functions_real_match() {
        // Arrange — see module doc for why this is the check R64.2 asks
        // for rather than a second hand-typed name list.
        let lookup = EmptyLookup;
        let lap_ctx = MathLapContext::empty();

        for entry in math_builtin_catalog() {
            // Act — a zero-arg call is always syntactically valid; every
            // match arm in `call_function` validates argument count/type
            // before indexing `args`, so this never panics regardless of
            // the function's real arity.
            let result = evaluate(&format!("{}()", entry.name), &lookup, &lap_ctx);

            // Assert — any error other than UnknownFunction proves the
            // name reached the dispatch match (ArgCount/Type/NotImplemented
            // etc. are expected and fine here).
            if let Err(e) = result {
                assert_ne!(
                    e.kind,
                    MathEvalErrorKind::UnknownFunction,
                    "{}: not found in eval::call_function's dispatch table",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn every_entry_carries_all_six_documentation_columns() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act / Assert — a blank column would render as an empty cell in
        // `docs::render_workbook_reference`'s table and as an empty hover
        // card in the editor, neither of which fails loudly on its own.
        for entry in catalog {
            assert!(!entry.category.is_empty(), "{}: empty category", entry.name);
            assert!(!entry.signature.is_empty(), "{}: empty signature", entry.name);
            assert!(!entry.unit_rule.is_empty(), "{}: empty unit_rule", entry.name);
            assert!(!entry.shape.is_empty(), "{}: empty shape", entry.name);
            assert!(!entry.description.is_empty(), "{}: empty description", entry.name);
            assert!(!entry.example.is_empty(), "{}: empty example", entry.name);
        }
    }

    #[test]
    fn every_signature_and_example_opens_with_the_entrys_own_name() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act / Assert — the cheapest guard against a copy-paste row, which
        // is the realistic transcription error in a 72-row hand-typed table.
        for entry in catalog {
            assert!(
                entry.signature.starts_with(&format!("{}(", entry.name)),
                "{}: signature does not open with the function's own name",
                entry.name
            );
            assert!(
                entry.example.starts_with(&format!("{}(", entry.name)),
                "{}: example does not call the function it documents",
                entry.name
            );
        }
    }

    #[test]
    fn not_implemented_entries_say_so_in_their_description() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let silent: Vec<&str> = catalog
            .iter()
            .filter(|b| b.status == MathBuiltinStatus::NotImplemented)
            .filter(|b| !b.description.contains("Not implemented"))
            .map(|b| b.name)
            .collect();

        // Assert — the generated reference prints the description under the
        // heading; a deferred function whose prose reads like a working one
        // is exactly the false friend C2 §3.8's naming policy exists to stop.
        assert_eq!(silent, Vec::<&str>::new());
    }
}
