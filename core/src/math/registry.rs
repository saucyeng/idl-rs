//! The math builtin registry: one static, ordered table of every builtin
//! spelling, carrying both the catalog metadata `list_math_builtins` and
//! `idl-rs docs workbook` render (C3 §3.4, lead ruling R64.2,
//! `runs/2026-09-03/decisions.md`; R222 item 1) and the evaluator
//! `eval::call_function` dispatches to
//! (`docs/superpowers/specs/2026-11-12-idl-rs-builtin-registry.md`, phase 1).
//! Before this table the same facts lived in three hand-synced places: the
//! dispatch `match`, the hand-transcribed `catalog.rs`, and a
//! `KWARG_AWARE_FUNCTIONS` name list.
//!
//! The metadata columns were transcribed from C2 §3.3's "Builtin catalog"
//! table (`docs/superpowers/specs/2026-09-03-idl1-c2-workbook-v3.md`) — the
//! same source L6's `functionCatalog.ts` was transcribed from.
//!
//! `arity` is transcribed from C2 §3.3's Signature column: the set of valid
//! argument counts for that name (more than one entry when a name has more
//! than one call form, e.g. `rms(ch)` / `rms(ch, w)` → `&[1, 2]`). It is
//! documentation only: phase 1 adds no central arity check, each evaluator
//! keeps its own. `status` mirrors C2 §3.3's Status column collapsed to
//! R64.2's two-value wire enum: a function whose *only* unimplemented form is
//! a secondary arity (`median`'s 2-arg rolling form) is still
//! [`MathBuiltinStatus::Implemented`] here, because its 1-arg form dispatches
//! to real logic.
//!
//! One entry per spelling, in C2 §3.3's row order (the generated Markdown
//! reference depends on it; the JSON catalog is name-sorted). Spellings that
//! share an evaluator (`cumulative_trapezoid` / `cumtrapz`, the four
//! estimator names, the five not-implemented names) point at the same
//! [`EvalFn`]. `main(col[])` (table-cell only, documented in C2 §4, not §3.3)
//! is the one `documented: false` entry and sits last, so
//! [`math_builtin_catalog`] is the documented prefix — the 77 entries the
//! catalog has always had. The grammar keywords `and`/`or`/`not` are parsed as
//! operators in `math::parse` and never reach dispatch at all.

use crate::math::eval::builtins;
use crate::math::eval::{ChannelLookup, Kwargs, MathLapContext};
use crate::math::{MathEvalError, Value};

use MathBuiltinStatus::{Implemented as I, NotImplemented as N};

/// What a builtin's evaluator receives: the call as `eval::call_function`
/// got it, after the entry's keyword-argument gate has passed.
pub(crate) struct Call<'a> {
    /// The spelling the expression used (`cumtrapz` vs `cumulative_trapezoid`);
    /// evaluators put it in their error messages.
    pub name: &'a str,
    /// Positional arguments, already evaluated, owned (evaluators consume them).
    pub args: Vec<Value>,
    /// Keyword arguments, already evaluated (C2 §3.2); empty unless the
    /// entry's [`Builtin::kwargs`] is non-empty.
    pub kwargs: &'a Kwargs,
    /// The channel source the expression is evaluated against.
    pub lookup: &'a dyn ChannelLookup,
    /// The injected lap / window context (R123/R124).
    pub lap_ctx: &'a MathLapContext,
}

/// A builtin's evaluator: one call in, one value or typed error out.
pub(crate) type EvalFn = fn(Call<'_>) -> Result<Value, MathEvalError>;

/// Whether a catalog entry's function is implemented today. Mirrors C2
/// §3.3's Status column, collapsed to the two-value split that column's own
/// text totals (63 Implemented / 6 NotImplemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBuiltinStatus {
    /// The function's documented call form(s) all dispatch to real logic.
    Implemented,
    /// The function parses and validates but its evaluator returns
    /// `MathEvalErrorKind::NotImplemented` (C2 §3.3's committed-but-deferred
    /// surface, e.g. `spectrogram`, `envelope`).
    NotImplemented,
}

/// One builtin spelling: its name, the set of argument counts C2 §3.3's
/// signature documents as valid, whether it is implemented, the
/// documentation columns the generated workbook reference and the notebook
/// editor's hover card are both rendered from (added 2026-09-11, ruling R222
/// item 1), the keyword arguments it accepts, and its evaluator. See the
/// module doc for provenance.
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
#[derive(Clone, Copy)]
pub struct Builtin {
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
    /// The keyword arguments the call accepts (C2 §3.2). Empty means any
    /// keyword is rejected at `call_function`'s entry gate before the
    /// evaluator runs; non-empty lets keywords through to the evaluator,
    /// which checks each name against this same list.
    pub kwargs: &'static [&'static str],
    /// Whether the entry appears in [`math_builtin_catalog`] (the generated
    /// reference and the editor's catalog). False only for `main`.
    pub documented: bool,
    /// The evaluator `call_function` dispatches this spelling to.
    pub(crate) eval: EvalFn,
}

impl std::fmt::Debug for Builtin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Builtin")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .field("status", &self.status)
            .field("category", &self.category)
            .field("signature", &self.signature)
            .field("unit_rule", &self.unit_rule)
            .field("shape", &self.shape)
            .field("description", &self.description)
            .field("example", &self.example)
            .field("kwargs", &self.kwargs)
            .field("documented", &self.documented)
            .finish_non_exhaustive()
    }
}

/// The catalog's name for a [`Builtin`], kept for its consumers (`docs.rs`,
/// the bridge's `list_math_builtins`).
pub type MathBuiltin = Builtin;

/// The registry entry for `name`, if any spelling matches (a linear scan of
/// 78 entries per call, once per call site per evaluation — never per sample).
pub(crate) fn builtin(name: &str) -> Option<&'static Builtin> {
    BUILTINS.iter().find(|b| b.name == name)
}

/// Returns the 77-entry documented math builtin catalog (C2 §3.3, minus
/// `main(col[])` and the `and`/`or`/`not` grammar keywords — see module
/// doc). Order matches C2 §3.3's table row order.
pub fn math_builtin_catalog() -> &'static [MathBuiltin] {
    let documented = BUILTINS.iter().take_while(|b| b.documented).count();
    &BUILTINS[..documented]
}

/// Every builtin spelling, documented entries first in C2 §3.3's row order,
/// then the undocumented `main`.
pub(crate) static BUILTINS: &[Builtin] = &[
    Builtin {
        name: "butter",
        arity: &[4],
        status: I,
        category: "Filter",
        signature: "butter(order, cutoff_hz, \"low\"|\"lowpass\"|\"high\"|\"highpass\", ch)",
        unit_rule: "SameAsArg(3)",
        shape: "[t]",
        description: "Zero-phase Butterworth low- or high-pass filter; designs and applies the filter in one call. Needs a fixed-rate channel: put an event channel on a grid first, `resample(x, [Time])`.",
        example: "butter(2, 5, \"low\", [Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::butter,
    },
    Builtin {
        name: "sosfilt",
        arity: &[2],
        status: N,
        category: "Filter",
        signature: "sosfilt(sos, ch)",
        unit_rule: "SameAsArg(1)",
        shape: "[t]",
        description: "Applies a second-order-section filter to a channel. Not implemented.",
        example: "sosfilt(sos, [Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::not_implemented,
    },
    Builtin {
        name: "declip",
        arity: &[1],
        status: I,
        category: "Reconstruction",
        signature: "declip(ch)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Reconstructs samples lost to clipping; designed for accelerometer data clipped at +/-32 g. Needs a fixed-rate channel.",
        example: "declip([AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::declip,
    },
    Builtin {
        name: "cumulative_trapezoid",
        arity: &[1],
        status: I,
        category: "Time-domain",
        signature: "cumulative_trapezoid(ch)",
        unit_rule: "Product(SameAsArg(0), Fixed(s))",
        shape: "[t]",
        description: "Cumulative trapezoidal integral of a channel with respect to time. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.",
        example: "cumulative_trapezoid([AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::cumulative_trapezoid,
    },
    Builtin {
        name: "cumtrapz",
        arity: &[1],
        status: I,
        category: "Time-domain",
        signature: "cumtrapz(ch)",
        unit_rule: "Product(SameAsArg(0), Fixed(s))",
        shape: "[t]",
        description: "A permanent second spelling of `cumulative_trapezoid`, as scipy itself carries both.",
        example: "cumtrapz([AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::cumulative_trapezoid,
    },
    Builtin {
        name: "differentiate",
        arity: &[1],
        status: I,
        category: "Time-domain",
        signature: "differentiate(ch)",
        unit_rule: "Quotient(SameAsArg(0), Fixed(s))",
        shape: "[t]",
        description: "Backward difference with respect to time, with `result[0] = 0`. Deliberately not `gradient`'s central difference. A fixed-rate channel steps by its rate; an event channel (GPS, wheel pulses) by its recorded sample times.",
        example: "differentiate([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::differentiate,
    },
    Builtin {
        name: "gradient",
        arity: &[1],
        status: I,
        category: "Time-domain",
        signature: "gradient(ch)",
        unit_rule: "Quotient(SameAsArg(0), Fixed(s))",
        shape: "[t]",
        description: "Central difference with respect to time, `numpy.gradient`'s own formula. A fixed-rate channel steps by its rate; an event channel by its recorded sample times.",
        example: "gradient([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::gradient,
    },
    Builtin {
        name: "detrend",
        arity: &[1, 2],
        status: I,
        category: "Time-domain",
        signature: "detrend(ch) | detrend(ch, \"linear\"|\"constant\"|\"mean\"|\"none\")",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Removes a constant or linear trend from a channel.",
        example: "detrend([Fork], \"linear\")",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::detrend,
    },
    Builtin {
        name: "rms",
        arity: &[1, 2],
        status: I,
        category: "Time-domain / aggregate",
        signature: "rms(ch) -> scalar | rms(ch, w) -> rolling channel, `w` a window in samples",
        unit_rule: "SameAsArg(0)",
        shape: "scalar | [t]",
        description: "Root mean square over the whole window, or rolling over `w` samples.",
        example: "rms([Fork], 64)",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::rms,
    },
    Builtin {
        name: "mean",
        arity: &[1, 2],
        status: I,
        category: "Time-domain / aggregate",
        signature: "mean(ch) -> scalar | mean(ch, w) -> rolling channel",
        unit_rule: "SameAsArg(0)",
        shape: "scalar | [t]",
        description: "Arithmetic mean over the whole window, or rolling over `w` samples.",
        example: "mean([Fork], 64)",
        kwargs: &["window"],
        documented: true,
        eval: builtins::windowed::mean,
    },
    Builtin {
        name: "std",
        arity: &[1, 2],
        status: I,
        category: "Time-domain / aggregate",
        signature: "std(ch) -> scalar (population sigma) | std(ch, w) -> rolling channel",
        unit_rule: "SameAsArg(0)",
        shape: "scalar | [t]",
        description: "Population standard deviation over the whole window, or rolling over `w` samples.",
        example: "std([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::std_dev,
    },
    Builtin {
        name: "median",
        arity: &[1],
        status: I,
        category: "Aggregate",
        signature: "median(ch) -> scalar",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "Median of every sample in the window. The two-argument rolling form is not implemented.",
        example: "median([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::median,
    },
    Builtin {
        name: "sum",
        arity: &[1],
        status: I,
        category: "Aggregate",
        signature: "sum(ch) -> scalar",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "Raw sum of every sample in the window, not time-normalised.",
        example: "sum([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::sum,
    },
    Builtin {
        name: "count",
        arity: &[1],
        status: I,
        category: "Aggregate",
        signature: "count(ch) -> scalar",
        unit_rule: "Dimensionless",
        shape: "scalar",
        description: "Number of samples in the window.",
        example: "count([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::count,
    },
    Builtin {
        name: "first",
        arity: &[1],
        status: I,
        category: "Aggregate",
        signature: "first(ch) -> scalar",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "First sample in the window.",
        example: "first([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::first,
    },
    Builtin {
        name: "last",
        arity: &[1],
        status: I,
        category: "Aggregate",
        signature: "last(ch) -> scalar",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "Last sample in the window.",
        example: "last([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::last,
    },
    Builtin {
        name: "percentile",
        arity: &[2],
        status: I,
        category: "Aggregate",
        signature: "percentile(ch, quantile) -> scalar, `quantile` in [0, 100]",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "Value below which `quantile` percent of the samples fall. Retired name: `p`.",
        example: "percentile([Fork], 95)",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::percentile,
    },
    Builtin {
        name: "abs",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "abs(x)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Absolute value, sample by sample.",
        example: "abs([AccelY])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::abs,
    },
    Builtin {
        name: "sqrt",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "sqrt(x)",
        unit_rule: "PowN(0, 1/2)",
        shape: "[t]",
        description: "Square root, sample by sample; every unit exponent is halved.",
        example: "sqrt([Power])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::sqrt,
    },
    Builtin {
        name: "sign",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "sign(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "-1, 0 or 1 by sign; NaN stays NaN.",
        example: "sign(differentiate([Fork]))",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::sign,
    },
    Builtin {
        name: "floor",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "floor(x)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Rounds each sample down to the nearest integer.",
        example: "floor([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::floor,
    },
    Builtin {
        name: "ceil",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "ceil(x)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Rounds each sample up to the nearest integer.",
        example: "ceil([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::ceil,
    },
    Builtin {
        name: "round",
        arity: &[1],
        status: I,
        category: "Elementwise",
        signature: "round(x)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Rounds half away from zero (`2.5` gives `3`), not `numpy.round`'s banker's rounding.",
        example: "round([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::round,
    },
    Builtin {
        name: "pow",
        arity: &[2],
        status: I,
        category: "Elementwise",
        signature: "pow(x, y)",
        unit_rule: "PowN(0, y) for a literal `y`; otherwise Dimensionless or Unknown",
        shape: "[t]",
        description: "Raises each sample to the power `y`.",
        example: "pow([Speed], 2)",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::pow,
    },
    Builtin {
        name: "min",
        arity: &[1, 2],
        status: I,
        category: "Aggregate / elementwise",
        signature: "min(ch) -> scalar | min(a, b) -> elementwise",
        unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument",
        shape: "scalar | [t]",
        description: "Smallest sample in the window, or the per-sample smaller of two operands.",
        example: "min([Fork], [Shock])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::min,
    },
    Builtin {
        name: "max",
        arity: &[1, 2],
        status: I,
        category: "Aggregate / elementwise",
        signature: "max(ch) -> scalar | max(a, b) -> elementwise",
        unit_rule: "SameAsArg(0) one-argument; AllMatch(0, 1) two-argument",
        shape: "scalar | [t]",
        description: "Largest sample in the window, or the per-sample larger of two operands.",
        example: "max([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::windowed::max,
    },
    Builtin {
        name: "clip",
        arity: &[3],
        status: I,
        category: "Elementwise",
        signature: "clip(ch, lo, hi)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Limits each sample to `lo`..`hi`, both given in the channel's own units. Retired name: `clamp`.",
        example: "clip([Fork], 0, 160)",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::clip,
    },
    Builtin {
        name: "sin",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "sin(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Sine of an angle in radians.",
        example: "sin(deg2rad([Roll]))",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::sin,
    },
    Builtin {
        name: "cos",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "cos(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Cosine of an angle in radians.",
        example: "cos(deg2rad([Roll]))",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::cos,
    },
    Builtin {
        name: "tan",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "tan(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Tangent of an angle in radians.",
        example: "tan(deg2rad([Roll]))",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::tan,
    },
    Builtin {
        name: "asin",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "asin(x)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Arcsine, in radians; the argument must be dimensionless.",
        example: "asin([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::asin,
    },
    Builtin {
        name: "acos",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "acos(x)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Arccosine, in radians; the argument must be dimensionless.",
        example: "acos([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::acos,
    },
    Builtin {
        name: "atan",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "atan(x)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Arctangent, in radians; the argument must be dimensionless.",
        example: "atan([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::atan,
    },
    Builtin {
        name: "atan2",
        arity: &[2],
        status: I,
        category: "Trig",
        signature: "atan2(y, x)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Quadrant-correct two-argument arctangent, in radians.",
        example: "atan2([AccelY], [AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::atan2,
    },
    Builtin {
        name: "sinh",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "sinh(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Hyperbolic sine.",
        example: "sinh([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::sinh,
    },
    Builtin {
        name: "cosh",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "cosh(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Hyperbolic cosine.",
        example: "cosh([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::cosh,
    },
    Builtin {
        name: "tanh",
        arity: &[1],
        status: I,
        category: "Trig",
        signature: "tanh(x)",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "Hyperbolic tangent.",
        example: "tanh([Ratio])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::tanh,
    },
    Builtin {
        name: "deg2rad",
        arity: &[1],
        status: I,
        category: "Trig conversion",
        signature: "deg2rad(x)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Converts degrees to radians.",
        example: "deg2rad([Roll])",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::deg2rad,
    },
    Builtin {
        name: "rad2deg",
        arity: &[1],
        status: I,
        category: "Trig conversion",
        signature: "rad2deg(x)",
        unit_rule: "Fixed(deg)",
        shape: "[t]",
        description: "Converts radians to degrees.",
        example: "rad2deg(atan2([AccelY], [AccelZ]))",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::rad2deg,
    },
    Builtin {
        name: "periodogram",
        arity: &[1],
        status: I,
        category: "Frequency",
        signature: "periodogram(ch, window=\"boxcar\", detrend=\"constant\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
        unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")",
        shape: "[f]",
        description: "Single-segment power spectrum, scipy-named and scipy-scaled. Needs a fixed-rate channel. Retired name: `fft`.",
        example: "periodogram([Fork])",
        kwargs: &["window", "detrend", "scaling"],
        documented: true,
        eval: builtins::dsp::periodogram,
    },
    Builtin {
        name: "welch",
        arity: &[1],
        status: I,
        category: "Frequency",
        signature: "welch(ch, window=\"hann\", nperseg=n, noverlap=n, detrend=\"constant\", average=\"mean\"|\"median\"|\"max\"|\"none\", scaling=\"density\"|\"spectrum\"|\"raw_magnitude\")",
        unit_rule: "SelectByLiteral(scaling, { \"density\" -> Quotient(PowN(0, 2), Fixed(Hz)), \"spectrum\" -> PowN(0, 2), \"raw_magnitude\" -> SameAsArg(0) }, default \"density\")",
        shape: "[f]",
        description: "Segmented, averaged power spectrum -- the spectrum the FFT charts compute. Needs a fixed-rate channel.",
        example: "welch([Fork])",
        kwargs: &["window", "nperseg", "noverlap", "detrend", "average", "scaling"],
        documented: true,
        eval: builtins::dsp::welch,
    },
    Builtin {
        name: "spectrogram",
        arity: &[1],
        status: N,
        category: "Frequency",
        signature: "spectrogram(ch, window_size, hop_size, window, detrend, scaling)",
        unit_rule: "SelectByLiteral(scaling, ...) as `periodogram`, over a `[t,f]` value",
        shape: "[t,f]",
        description: "Time-varying spectrum. Not implemented; the signature and shape are fixed by C2 3.6.3.",
        example: "spectrogram([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::not_implemented,
    },
    Builtin {
        name: "envelope",
        arity: &[1],
        status: N,
        category: "Frequency",
        signature: "envelope(ch)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Amplitude envelope -- a magnitude, not scipy's complex analytic signal. Not implemented. Retired name: `hilbert`.",
        example: "envelope([Fork])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::not_implemented,
    },
    Builtin {
        name: "correlate",
        arity: &[2],
        status: N,
        category: "Correlation",
        signature: "correlate(a, b)",
        unit_rule: "Product(0, 1)",
        shape: "[t]",
        description: "Cross-correlation of two channels. Not implemented.",
        example: "correlate([Fork], [Shock])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::not_implemented,
    },
    Builtin {
        name: "convolve",
        arity: &[2],
        status: N,
        category: "Correlation",
        signature: "convolve(ch, kernel)",
        unit_rule: "Product(0, 1)",
        shape: "[t]",
        description: "Convolution of a channel with a kernel. Not implemented.",
        example: "convolve([Fork], kernel)",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::not_implemented,
    },
    Builtin {
        name: "resample",
        arity: &[2],
        status: I,
        category: "Resampling",
        signature: "resample(x, onto)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Linear interpolation of `x` onto the per-sample recorded times of `onto`; the result carries `onto`'s time axis, rate and length, so the two can then be combined. Times outside `x`'s span are `NaN` -- never extrapolated. The explicit way to combine two channels recorded on different clocks; nothing resamples implicitly.",
        example: "resample([IMU2_AccelZ], [IMU1_AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::resample,
    },
    // `deglitch_pulses`'s evaluator checks a `max_fraction=` keyword, but
    // the entry gate in front of it has always rejected every keyword
    // for it (it was never in the old `KWARG_AWARE_FUNCTIONS`), so
    // `kwargs` stays empty: phase 1 changes no behaviour.
    Builtin {
        name: "deglitch_pulses",
        arity: &[1],
        status: I,
        category: "Resampling",
        signature: "deglitch_pulses(pulses, max_fraction=0.35)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "A wheel-pulse event channel without the extra edges a marginal sensor adds: a pulse is dropped when the gap ending at it is under `max_fraction` of both the gap before and the gap after, so it reads as an extra edge inside one ordinary gap. Comparing both sides means the first pulses after a locked wheel spins up, and the growing gaps of hard braking, are never dropped. The result is renumbered 1, 2, 3, ... from the first kept pulse, at the kept pulses' recorded times.",
        example: "deglitch_pulses([WheelFront])",
        kwargs: &[],
        documented: true,
        eval: builtins::dsp::deglitch_pulses,
    },
    Builtin {
        name: "keep",
        arity: &[2],
        status: I,
        category: "Logic",
        signature: "keep(x, cond)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Only the samples of `x` where `cond` is true (non-zero), each at its own recorded time: an event list, such as every pulse gap over a threshold, comes out as a short channel instead of a full-length one with zeros. `cond` pairs with `x` sample for sample (put it on `x`'s axis with `resample` first if it is not); a number keeps all or nothing. Nothing kept is an empty channel. Over a `[lap]` value it keeps laps.",
        example: "keep([gap_ratio], [gap_ratio] > 1.8)",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::keep,
    },
    Builtin {
        name: "where",
        arity: &[3],
        status: I,
        category: "Logic",
        signature: "where(cond, t, f)",
        unit_rule: "AllMatch(1, 2)",
        shape: "[t]",
        description: "Per-sample choice between two branches; `cond` may also be a scalar, selecting a whole branch. Retired name: `if`.",
        example: "where([Speed] > 10, [Fork], 0)",
        kwargs: &[],
        documented: true,
        eval: builtins::elementwise::where_,
    },
    Builtin {
        name: "current_lap",
        arity: &[0],
        status: I,
        category: "Lap",
        signature: "current_lap()",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "1-based lap number at each sample, `0` outside any lap.",
        example: "current_lap()",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::current_lap,
    },
    Builtin {
        name: "lap_start_time",
        arity: &[1],
        status: I,
        category: "Lap",
        signature: "lap_start_time(n)",
        unit_rule: "Fixed(s)",
        shape: "scalar",
        description: "Session time at which lap `n` starts; NaN when `n` is out of range.",
        example: "lap_start_time(2)",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_start_time,
    },
    Builtin {
        name: "lap_start_distance",
        arity: &[1],
        status: I,
        category: "Lap",
        signature: "lap_start_distance(n)",
        unit_rule: "Fixed(m)",
        shape: "scalar",
        description: "Distance at which lap `n` starts; NaN when `n` is out of range or the session has no `[Distance]`.",
        example: "lap_start_distance(2)",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_start_distance,
    },
    Builtin {
        name: "sector_number",
        arity: &[0],
        status: I,
        category: "Lap",
        signature: "sector_number()",
        unit_rule: "Dimensionless",
        shape: "[t]",
        description: "0-based sector index at each sample, NaN outside any sector.",
        example: "sector_number()",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::sector_number,
    },
    // The lap-shaped trio C2 §3.3 has listed since the spec was written
    // and `math_builtin_catalog()` lacked until 2026-09-13 (the drift the
    // docs lane filed, closed by ruling R233). `shape` is the pair the
    // signature's own result annotation gives: a table row binds one lap,
    // so a cell gets a scalar; a `math` cell gets one entry per lap.
    Builtin {
        name: "lap_number",
        arity: &[0],
        status: I,
        category: "Lap",
        signature: "lap_number()",
        unit_rule: "Dimensionless",
        shape: "scalar | [lap]",
        description: "The 1-based lap number: this row's lap in a table cell, every lap in a math cell.",
        example: "lap_number()",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_number,
    },
    Builtin {
        name: "lap_time",
        arity: &[0],
        status: I,
        category: "Lap",
        signature: "lap_time()",
        unit_rule: "Fixed(s)",
        shape: "scalar | [lap]",
        description: "Lap duration in seconds: this row's lap in a table cell, every lap in a math cell.",
        example: "lap_time()",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_time,
    },
    Builtin {
        name: "sector_time",
        arity: &[1],
        status: I,
        category: "Lap",
        signature: "sector_time(i)",
        unit_rule: "Fixed(s)",
        shape: "scalar | [lap]",
        description: "Duration of the lap's `i`-th sector (0-based, as `sector_number()` counts) in seconds; NaN when the lap has no such sector.",
        example: "sector_time(1)",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::sector_time,
    },
    Builtin {
        name: "lap_delta_time",
        arity: &[1],
        status: I,
        category: "Lap delta",
        signature: "lap_delta_time(ch)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Main lap minus overlay lap, time-matched; the mean across every overlay when more than one is selected. Retired name: `variance_time`.",
        example: "lap_delta_time([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_delta_time,
    },
    Builtin {
        name: "lap_delta_dist",
        arity: &[1],
        status: I,
        category: "Lap delta",
        signature: "lap_delta_dist(ch)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Main lap minus overlay lap, arc-length-matched. Retired name: `variance_dist`.",
        example: "lap_delta_dist([Speed])",
        kwargs: &[],
        documented: true,
        eval: builtins::laps::lap_delta_dist,
    },
    Builtin {
        name: "attitude",
        arity: &[1],
        status: I,
        category: "Estimator (diagnostic)",
        signature: "attitude(\"roll\"|\"pitch\")",
        unit_rule: "Fixed(deg)",
        shape: "[t]",
        description: "Bike attitude estimated by the AHRS filter, in degrees.",
        example: "attitude(\"roll\")",
        kwargs: &[],
        documented: true,
        eval: builtins::estimator::virtual_sensor,
    },
    Builtin {
        name: "body_accel",
        arity: &[1],
        status: I,
        category: "Estimator (diagnostic)",
        signature: "body_accel(\"long\"|\"lat\")",
        unit_rule: "Fixed(g)",
        shape: "[t]",
        description: "Gravity-compensated body-frame acceleration, in g.",
        example: "body_accel(\"long\")",
        kwargs: &[],
        documented: true,
        eval: builtins::estimator::virtual_sensor,
    },
    Builtin {
        name: "wheel_travel",
        arity: &[1],
        status: I,
        category: "Estimator",
        signature: "wheel_travel(\"front\"|\"rear\")",
        unit_rule: "Fixed(mm)",
        shape: "[t]",
        description: "Suspension travel at the named wheel, in mm.",
        example: "wheel_travel(\"front\")",
        kwargs: &[],
        documented: true,
        eval: builtins::estimator::virtual_sensor,
    },
    Builtin {
        name: "wheel_velocity",
        arity: &[1],
        status: I,
        category: "Estimator",
        signature: "wheel_velocity(\"front\"|\"rear\")",
        unit_rule: "Fixed(mm/s)",
        shape: "[t]",
        description: "Suspension velocity at the named wheel, in mm/s.",
        example: "wheel_velocity(\"front\")",
        kwargs: &[],
        documented: true,
        eval: builtins::estimator::virtual_sensor,
    },
    Builtin {
        name: "vec",
        arity: &[3],
        status: I,
        category: "Vector",
        signature: "vec(x, y, z)",
        unit_rule: "AllMatch(0, 1, 2)",
        shape: "Vec3",
        description: "Builds a Vec3 from three components, which must share one unit.",
        example: "vec([AccelX], [AccelY], [AccelZ])",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vec,
    },
    Builtin {
        name: "vx",
        arity: &[1],
        status: I,
        category: "Vector",
        signature: "vx(v)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "The x component of a Vec3.",
        example: "vx(accel)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vx,
    },
    Builtin {
        name: "vy",
        arity: &[1],
        status: I,
        category: "Vector",
        signature: "vy(v)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "The y component of a Vec3.",
        example: "vy(accel)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vy,
    },
    Builtin {
        name: "vz",
        arity: &[1],
        status: I,
        category: "Vector",
        signature: "vz(v)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "The z component of a Vec3.",
        example: "vz(accel)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vz,
    },
    Builtin {
        name: "vadd",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "vadd(a, b)",
        unit_rule: "AllMatch(0, 1)",
        shape: "Vec3",
        description: "Component-wise sum of two Vec3s.",
        example: "vadd(a, b)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vadd,
    },
    Builtin {
        name: "vsub",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "vsub(a, b)",
        unit_rule: "AllMatch(0, 1)",
        shape: "Vec3",
        description: "Component-wise difference of two Vec3s.",
        example: "vsub(a, b)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vsub,
    },
    Builtin {
        name: "vscale",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "vscale(v, s)",
        unit_rule: "Product(0, 1)",
        shape: "Vec3",
        description: "Scales a Vec3 by a scalar.",
        example: "vscale(accel, 9.81)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::vscale,
    },
    Builtin {
        name: "cross",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "cross(a, b)",
        unit_rule: "Product(0, 1)",
        shape: "Vec3",
        description: "Cross product of two Vec3s.",
        example: "cross(a, b)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::cross,
    },
    Builtin {
        name: "dot",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "dot(a, b)",
        unit_rule: "Product(0, 1)",
        shape: "[t]",
        description: "Dot product of two Vec3s.",
        example: "dot(a, b)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::dot,
    },
    Builtin {
        name: "norm",
        arity: &[1],
        status: I,
        category: "Vector",
        signature: "norm(v)",
        unit_rule: "SameAsArg(0)",
        shape: "[t]",
        description: "Euclidean length of a Vec3.",
        example: "norm(accel)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::norm,
    },
    Builtin {
        name: "normalize",
        arity: &[1],
        status: I,
        category: "Vector",
        signature: "normalize(v)",
        unit_rule: "Dimensionless",
        shape: "Vec3",
        description: "Unit vector in the same direction as `v`.",
        example: "normalize(accel)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::normalize,
    },
    Builtin {
        name: "angle_between",
        arity: &[2],
        status: I,
        category: "Vector",
        signature: "angle_between(a, b)",
        unit_rule: "Fixed(rad)",
        shape: "[t]",
        description: "Angle between two Vec3s, in radians, in [0, pi]. Retired name: `angle`.",
        example: "angle_between(a, b)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::angle_between,
    },
    Builtin {
        name: "rotate_mat",
        arity: &[10],
        status: I,
        category: "Rotation",
        signature: "rotate_mat(v, m00..m22) (row-major, scalar entries)",
        unit_rule: "SameAsArg(0)",
        shape: "Vec3",
        description: "Rotates a Vec3 by a row-major 3x3 matrix.",
        example: "rotate_mat(a, 1, 0, 0, 0, 1, 0, 0, 0, 1)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::rotate_mat,
    },
    Builtin {
        name: "rotate_axis",
        arity: &[5],
        status: I,
        category: "Rotation",
        signature: "rotate_axis(v, ax, ay, az, angle) (scalars; `angle` in radians)",
        unit_rule: "SameAsArg(0)",
        shape: "Vec3",
        description: "Rotates a Vec3 about an axis by an angle in radians.",
        example: "rotate_axis(a, 0, 0, 1, 1.5708)",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::rotate_axis,
    },
    Builtin {
        name: "rotate_euler",
        arity: &[4],
        status: I,
        category: "Rotation",
        signature: "rotate_euler(v, roll, pitch, yaw) (radians; the angles may be channels)",
        unit_rule: "SameAsArg(0)",
        shape: "Vec3",
        description: "Rotates a Vec3 by roll/pitch/yaw in radians; the angles may be channels, giving a per-sample rotation.",
        example: "rotate_euler(a, [Roll], [Pitch], [Yaw])",
        kwargs: &[],
        documented: true,
        eval: builtins::vector::rotate_euler,
    },
    // Table-cell only (C2 §4, not §3.3): a `{col[]}` reference to the
    // Main lap's row. Dispatches like any builtin but is not in the
    // generated reference or the editor's catalog, so it stays last and
    // `documented: false`; `math_builtin_catalog` is the prefix before it.
    Builtin {
        name: "main",
        arity: &[1],
        status: I,
        category: "Table",
        signature: "main(col[])",
        unit_rule: "SameAsArg(0)",
        shape: "scalar",
        description: "The Main lap's value in a table column; NaN when no Main row is set.",
        example: "main({col[]})",
        kwargs: &[],
        documented: false,
        eval: builtins::windowed::main,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{evaluate, LookupChannel, MathEvalErrorKind};

    /// A lookup with no channels — every dispatch check calls its function
    /// with literal arguments only, so no channel reference is ever reached.
    struct EmptyLookup;
    impl ChannelLookup for EmptyLookup {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
    }

    /// The documented catalog's names in the order `catalog.rs` listed them
    /// before the registry replaced it — the order the generated Markdown
    /// reference is rendered in.
    const CATALOG_ORDER: &[&str] = &[
        "butter",
        "sosfilt",
        "declip",
        "cumulative_trapezoid",
        "cumtrapz",
        "differentiate",
        "gradient",
        "detrend",
        "rms",
        "mean",
        "std",
        "median",
        "sum",
        "count",
        "first",
        "last",
        "percentile",
        "abs",
        "sqrt",
        "sign",
        "floor",
        "ceil",
        "round",
        "pow",
        "min",
        "max",
        "clip",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "deg2rad",
        "rad2deg",
        "periodogram",
        "welch",
        "spectrogram",
        "envelope",
        "correlate",
        "convolve",
        "resample",
        "deglitch_pulses",
        "keep",
        "where",
        "current_lap",
        "lap_start_time",
        "lap_start_distance",
        "sector_number",
        "lap_number",
        "lap_time",
        "sector_time",
        "lap_delta_time",
        "lap_delta_dist",
        "attitude",
        "body_accel",
        "wheel_travel",
        "wheel_velocity",
        "vec",
        "vx",
        "vy",
        "vz",
        "vadd",
        "vsub",
        "vscale",
        "cross",
        "dot",
        "norm",
        "normalize",
        "angle_between",
        "rotate_mat",
        "rotate_axis",
        "rotate_euler",
    ];

    #[test]
    fn math_builtin_catalog_len_is_77_after_the_scipy_splits_the_lap_shaped_trio_deglitch_pulses_and_keep() {
        // Arrange / Act
        let n = math_builtin_catalog().len();

        // Assert — was 69 (C2 §3.3's original stated total: 63 Implemented +
        // 6 NotImplemented); the scipy-alignment lane adds three entries
        // without removing any (`runs/2026-09-08/scipy-alignment-plan.md`):
        // `fft` split into `periodogram` + `welch` (net +1, task 6),
        // `cumtrapz` added as `cumulative_trapezoid`'s permanent second
        // spelling (net +1, task 10, R151 item 6), and `gradient` added
        // alongside `differentiate` (net +1, task 11, R151 item 3) —
        // 69 + 3 = 72; then `lap_number`/`lap_time`/`sector_time`, which C2
        // §3.3 has listed all along and this catalog lacked (R233), 72 + 3 = 75;
        // then `deglitch_pulses` for wheel-pulse channels (2026-11-05), 76;
        // then `keep` for event lists (2026-11-05), 77.
        assert_eq!(n, 77);
    }

    #[test]
    fn implemented_and_not_implemented_counts_split_71_and_5() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let not_implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::NotImplemented).count();
        let implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::Implemented).count();

        // Assert — was 63/6; the scipy-alignment lane's three net-new entries
        // and the lap-shaped trio (see the length test above) are all
        // Implemented, so only that side of the split moved (69/6). R242 then
        // implements `resample`, moving one entry across: 70/5. `deglitch_pulses`
        // arrives implemented: 71/5; `keep` too: 72/5.
        assert_eq!(not_implemented, 5);
        assert_eq!(implemented, 72);
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
    fn registry_every_spelling_including_undocumented_main_is_unique() {
        // Arrange
        let mut names: Vec<&str> = BUILTINS.iter().map(|b| b.name).collect();
        let before = names.len();

        // Act
        names.sort_unstable();
        names.dedup();

        // Assert — `builtin()` returns the first match, so a duplicate
        // spelling would silently shadow the second entry's evaluator.
        assert_eq!(names.len(), before, "duplicate name in BUILTINS");
    }

    #[test]
    fn registry_documented_entries_in_the_previous_catalog_order_then_only_main_undocumented() {
        // Arrange
        let catalog: Vec<&str> = math_builtin_catalog().iter().map(|b| b.name).collect();
        let undocumented: Vec<&str> = BUILTINS.iter().filter(|b| !b.documented).map(|b| b.name).collect();

        // Act
        let tail: Vec<&str> = BUILTINS[catalog.len()..].iter().map(|b| b.name).collect();

        // Assert — the Markdown reference renders in this order, so a moved
        // entry is a docs diff; and `math_builtin_catalog` is the documented
        // prefix, so nothing documented may sit after `main`.
        assert_eq!(catalog, CATALOG_ORDER);
        assert_eq!(undocumented, vec!["main"]);
        assert_eq!(tail, vec!["main"]);
    }

    #[test]
    fn registry_every_spelling_dispatches_through_call_function() {
        // Arrange — R64.2: the catalog's names are checked against the real
        // dispatch, never a second hand copy. Dispatch *is* the table now, so
        // this pins the path `evaluate` → `call_function` → `builtin()`.
        let lookup = EmptyLookup;
        let lap_ctx = MathLapContext::empty();

        for entry in BUILTINS {
            // Act — a zero-arg call is always syntactically valid; every
            // evaluator validates argument count/type before indexing
            // `args`, so this never panics regardless of the function's
            // real arity.
            let result = evaluate(&format!("{}()", entry.name), &lookup, &lap_ctx);

            // Assert — any error other than UnknownFunction proves the
            // name reached its evaluator (ArgCount/Type/NotImplemented etc.
            // are expected and fine here).
            if let Err(e) = result {
                assert_ne!(
                    e.kind,
                    MathEvalErrorKind::UnknownFunction,
                    "{}: does not dispatch through eval::call_function",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn registry_kwargs_lists_are_exactly_what_each_evaluator_accepts() {
        // Arrange — one positional scalar (enough to pass `periodogram` /
        // `welch`'s arg-count check, which runs before their keyword check)
        // and one keyword no builtin accepts.
        let lookup = EmptyLookup;
        let lap_ctx = MathLapContext::empty();

        for entry in BUILTINS {
            // Act
            let e = evaluate(&format!("{}(1, zz_unknown=1)", entry.name), &lookup, &lap_ctx).unwrap_err();

            // Assert — an empty list is refused at the entry gate; a
            // non-empty list reaches the evaluator, whose own error names
            // the keywords it accepts, which must be this list exactly.
            assert_eq!(e.kind, MathEvalErrorKind::Runtime, "{}: {}", entry.name, e.message);
            if entry.kwargs.is_empty() {
                assert!(e.message.contains("takes no keyword arguments"), "{}: {}", entry.name, e.message);
            } else {
                let accepts = format!("(accepts: {})", entry.kwargs.join(", "));
                assert!(e.message.contains(&accepts), "{}: {} (want {accepts})", entry.name, e.message);
            }
        }
        let aware: Vec<&str> = BUILTINS.iter().filter(|b| !b.kwargs.is_empty()).map(|b| b.name).collect();
        assert_eq!(aware, vec!["mean", "periodogram", "welch"]);
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
        // is the realistic transcription error in a 75-row hand-typed table.
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
