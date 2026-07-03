//! Evaluator: walks an [`Ast`] against a [`ChannelLookup`] and a
//! [`MathLapContext`], producing a [`Value`]. Operator and function semantics
//! are ported from the Dart `_Evaluator` (`math_channel_evaluator.dart`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::math::parse::{parse, Ast, BinOp, UnOp};
use crate::math::value::{ChannelValue, Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// A channel resolved by [`ChannelLookup`]. Samples are an `Arc<[f64]>`: the
/// handle's math store sits behind a `RwLock` that cannot lend a slice past the
/// read guard, so the buffer is widened once and shared. A reference clones the
/// `Arc` (a counter bump, no data copy); the per-pass [`MemoLookup`] ensures a
/// channel referenced N times in one expression is widened exactly once.
pub struct LookupChannel {
    pub samples: Arc<[f64]>,
    pub sample_rate_hz: f64,
}

/// Resolves `[Name]` channel references to sample data. Implemented by
/// `SessionHandle` (base + synthesized + math-store channels) and by test
/// doubles. A future Rust dependency resolver is just another impl (spec §5).
pub trait ChannelLookup {
    fn lookup(&self, name: &str) -> Option<LookupChannel>;

    /// `(length, sample_rate_hz)` of the highest-rate non-event channel with
    /// the most samples — used to synthesize a time base when no explicit
    /// `Time` channel is in scope. Default `None` (callers fall back to an
    /// empty 10 Hz base). `SessionHandle` overrides this (Task A11).
    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        None
    }

    /// `(length, sample_rate_hz)` of `name` WITHOUT materializing its samples.
    /// Default `None`; `SessionHandle` overrides it. The closed-form time base
    /// reads this so the zero-storage `Time` ramp is never widened.
    fn channel_dims(&self, _name: &str) -> Option<(usize, f64)> {
        None
    }

    /// Resolve a `{cell}` reference to a scalar. Default `None` — channel-math
    /// lookups never provide cells, so `{cell}` is structurally unavailable
    /// there (the firewall). `CellLookup` (tables) overrides this.
    fn lookup_cell(&self, _name: &str) -> Option<f64> {
        None
    }

    /// Resolve a `{colname[]}` whole-column reference to its values. Default
    /// `None`. Used by aggregates over a table column.
    fn lookup_cell_column(&self, _name: &str) -> Option<Vec<f64>> {
        None
    }

    /// Event-driven per-sample times for `name`, seconds on the recording
    /// timeline (the same clock as fixed-rate channels' `index / rate`).
    /// `None` for fixed-rate or absent channels. Default `None`;
    /// `SessionHandle` overrides it (GPS channels carry per-fix event times).
    fn sample_times(&self, _name: &str) -> Option<Vec<f64>> {
        None
    }
}

/// Per-evaluation-pass memoizer over a [`ChannelLookup`]. Caches each `lookup`
/// result so a channel referenced N times in one expression is widened once;
/// repeats clone the cached `Arc<[f64]>` (a counter bump, no data copy). The
/// evaluator is single-threaded per pass, so `RefCell` is sound. Lives for one
/// [`evaluate`] call. `best_time_base_dims` / `lookup_cell` / `lookup_cell_column`
/// delegate to `inner` (cell/dims results are cheap and not worth caching).
struct MemoLookup<'a> {
    inner: &'a dyn ChannelLookup,
    cache: RefCell<HashMap<String, LookupChannel>>,
}

impl<'a> MemoLookup<'a> {
    fn new(inner: &'a dyn ChannelLookup) -> Self {
        MemoLookup { inner, cache: RefCell::new(HashMap::new()) }
    }
}

impl ChannelLookup for MemoLookup<'_> {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        if let Some(hit) = self.cache.borrow().get(name) {
            // Cache hit: clone the shared Arc buffer (no data copy).
            return Some(LookupChannel {
                samples: hit.samples.clone(),
                sample_rate_hz: hit.sample_rate_hz,
            });
        }
        // Miss: resolve once, cache an Arc-sharing copy, return the original.
        // (A miss is not cached, so an absent channel is re-queried cheaply.)
        let resolved = self.inner.lookup(name)?;
        self.cache.borrow_mut().insert(
            name.to_string(),
            LookupChannel { samples: resolved.samples.clone(), sample_rate_hz: resolved.sample_rate_hz },
        );
        Some(resolved)
    }

    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        self.inner.best_time_base_dims()
    }

    fn lookup_cell(&self, name: &str) -> Option<f64> {
        self.inner.lookup_cell(name)
    }

    fn lookup_cell_column(&self, name: &str) -> Option<Vec<f64>> {
        self.inner.lookup_cell_column(name)
    }

    fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
        self.inner.channel_dims(name)
    }
}

/// Overlay (reference) session data for `variance_*`. Crosses as a second
/// handle/lookup — overlay channels are read Rust-side, never marshalled
/// (spec §6). Constructed by the bridge from the second `RustOpaque<SessionHandle>`.
#[derive(Clone)]
pub struct MathOverlay {
    /// Lookup over the overlay session (its channels, read Rust-side).
    pub lookup: Arc<dyn ChannelLookup + Send + Sync>,
    /// Overlay lap window in raw epoch ms (selects overlay GPS samples).
    pub lap_start_ms: f64,
    pub lap_end_ms: f64,
    /// Overlay lap start in uniform-time seconds (pre-converted Dart-side).
    pub lap_start_uniform_sec: f64,
}

impl std::fmt::Debug for MathOverlay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MathOverlay")
            .field("lap_start_ms", &self.lap_start_ms)
            .field("lap_end_ms", &self.lap_end_ms)
            .field("lap_start_uniform_sec", &self.lap_start_uniform_sec)
            .finish_non_exhaustive()
    }
}

/// Injected lap/overlay state for the lap-aware and variance functions.
/// Bounds and sectors are session-relative seconds (the Dart side converts
/// epoch-ms → uniform-time before constructing this). `overlay` carries the
/// second-handle data used by `variance_*`.
#[derive(Debug, Clone, Default)]
pub struct MathLapContext {
    /// Per-lap `(start_s, end_s)` in session-relative seconds, in lap order.
    pub main_lap_bounds: Vec<(f64, f64)>,
    /// Sectors flattened across laps in arrival order, `(start_s, end_s)`.
    pub main_sectors: Vec<(f64, f64)>,
    /// 1-based designated main lap, or `None`.
    pub main_lap_number: Option<u32>,
    /// Overlay session for `variance_*`. `None` when no overlay is designated.
    pub overlay: Option<MathOverlay>,
    /// Row index the table's `main({col[]})` function reads from (the Main
    /// lap's row). `None` outside the table-cell path — `main()` then yields
    /// `NaN`, so it never crosses into channel math (the firewall).
    pub baseline_row: Option<usize>,
}

impl MathLapContext {
    /// A context with no laps, no sectors, and no overlay. Used by headless
    /// callers (the CLI) where lap detection has not run — lap-aware functions
    /// then return `MathEvalErrorKind::NoLapContext`.
    pub fn empty() -> Self {
        Self {
            main_lap_bounds: Vec::new(),
            main_sectors: Vec::new(),
            main_lap_number: None,
            overlay: None,
            baseline_row: None,
        }
    }
}

/// Output of a top-level evaluation: a sample buffer + rate. A scalar result
/// is a one-sample, rate-0 channel (matching the Dart public API).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalOutput {
    pub samples: Vec<f64>,
    pub sample_rate_hz: f64,
}

fn err(kind: MathEvalErrorKind, msg: impl Into<String>) -> MathEvalError {
    MathEvalError::new(kind, msg)
}

/// Parses and evaluates `expression`, returning the output buffer + rate.
pub fn evaluate(
    expression: &str,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> Result<EvalOutput, MathEvalError> {
    let ast = parse(expression)?;
    // Memoize lookups for this pass so a channel referenced N times is widened
    // once and shared (Arc) across references. See MemoLookup.
    let memo = MemoLookup::new(lookup);
    let value = eval(&ast, &memo, lap_ctx)?;
    match value {
        Value::Channel(c) => Ok(EvalOutput { samples: c.samples.to_vec(), sample_rate_hz: c.sample_rate_hz }),
        Value::Scalar(v) => Ok(EvalOutput { samples: vec![v], sample_rate_hz: 0.0 }),
        Value::Str(_) => Err(err(
            MathEvalErrorKind::Type,
            "Expression evaluated to a string, not a channel or scalar",
        )),
        Value::Vec3(_) => Err(err(
            MathEvalErrorKind::Type,
            "Expression evaluated to a 3-vector. Extract a component with vx()/vy()/vz() \
             or reduce it with norm() to plot a scalar channel.",
        )),
    }
}

/// Evaluate `expression` and require a single scalar result — the table-cell
/// entry point. Reuses `evaluate`; a scalar surfaces as a rate-0 one-sample
/// `EvalOutput` (eval.rs already does this), so this is a thin output adapter,
/// not a second evaluator. A multi-sample (channel) result is rejected: a cell
/// must reduce to one value (use an aggregate like `mean([Fork])`).
pub fn evaluate_scalar(
    expression: &str,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> Result<f64, MathEvalError> {
    let out = evaluate(expression, lookup, lap_ctx)?;
    if out.sample_rate_hz == 0.0 && out.samples.len() == 1 {
        Ok(out.samples[0])
    } else {
        Err(err(
            MathEvalErrorKind::Runtime,
            "Cell formula must reduce to a single value — wrap a channel in an \
             aggregate, e.g. mean([Fork]).",
        ))
    }
}

/// Evaluates a single AST node.
pub fn eval(
    ast: &Ast,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> Result<Value, MathEvalError> {
    match ast {
        Ast::Number(v) => Ok(Value::Scalar(*v)),
        Ast::Str(s) => Ok(Value::Str(s.clone())),
        Ast::ChannelRef(name) => {
            let ch = lookup.lookup(name).ok_or_else(|| {
                err(
                    MathEvalErrorKind::UnknownChannel,
                    format!("Channel '[{name}]' not in this session"),
                )
            })?;
            Ok(Value::Channel(ChannelValue {
                samples: ch.samples,
                sample_rate_hz: ch.sample_rate_hz,
                channel_id: Some(name.clone()),
            }))
        }
        Ast::CellRef(name) => {
            // `name[]` → whole column (array as a rate-0 channel); else scalar.
            if let Some(col) = name.strip_suffix("[]") {
                let vals = lookup.lookup_cell_column(col).ok_or_else(|| {
                    err(
                        MathEvalErrorKind::UnknownChannel,
                        format!("Cell column '{{{col}[]}}' not found"),
                    )
                })?;
                Ok(Value::Channel(ChannelValue {
                    samples: Arc::from(vals),
                    sample_rate_hz: 0.0,
                    channel_id: None,
                }))
            } else {
                let v = lookup.lookup_cell(name).ok_or_else(|| {
                    err(
                        MathEvalErrorKind::UnknownChannel,
                        format!("Cell '{{{name}}}' not found"),
                    )
                })?;
                Ok(Value::Scalar(v))
            }
        }
        Ast::Unary { op, expr } => {
            let v = eval(expr, lookup, lap_ctx)?;
            match op {
                UnOp::Neg => map_value(v, |x| -x),
                UnOp::Not => map_value(v, |x| if x != 0.0 { 0.0 } else { 1.0 }),
            }
        }
        Ast::Binary { op, left, right } => {
            let l = eval(left, lookup, lap_ctx)?;
            let r = eval(right, lookup, lap_ctx)?;
            apply_binary(*op, l, r)
        }
        Ast::Call { name, args } => {
            let argv: Vec<Value> = args
                .iter()
                .map(|a| eval(a, lookup, lap_ctx))
                .collect::<Result<_, _>>()?;
            call_function(name, argv, lookup, lap_ctx)
        }
    }
}

fn apply_binary(op: BinOp, left: Value, right: Value) -> Result<Value, MathEvalError> {
    let name = format!("{op:?}");
    match op {
        BinOp::Add => elemwise(left, right, &name, |a, b| Ok(a + b)),
        BinOp::Sub => elemwise(left, right, &name, |a, b| Ok(a - b)),
        BinOp::Mul => elemwise(left, right, &name, |a, b| Ok(a * b)),
        BinOp::Div => elemwise(left, right, &name, |a, b| {
            if b == 0.0 {
                Err(err(
                    MathEvalErrorKind::DivisionByZero,
                    "Division by zero in math channel expression",
                ))
            } else {
                Ok(a / b)
            }
        }),
        BinOp::Lt => elemwise(left, right, &name, |a, b| Ok(if a < b { 1.0 } else { 0.0 })),
        BinOp::Gt => elemwise(left, right, &name, |a, b| Ok(if a > b { 1.0 } else { 0.0 })),
        BinOp::LtEq => elemwise(left, right, &name, |a, b| Ok(if a <= b { 1.0 } else { 0.0 })),
        BinOp::GtEq => elemwise(left, right, &name, |a, b| Ok(if a >= b { 1.0 } else { 0.0 })),
        BinOp::EqEq => elemwise(left, right, &name, |a, b| Ok(if a == b { 1.0 } else { 0.0 })),
        BinOp::BangEq => elemwise(left, right, &name, |a, b| Ok(if a != b { 1.0 } else { 0.0 })),
        BinOp::And => {
            elemwise(left, right, &name, |a, b| Ok(if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }))
        }
        BinOp::Or => {
            elemwise(left, right, &name, |a, b| Ok(if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }))
        }
    }
}

/// Element-wise binary op over channel×channel / channel×scalar /
/// scalar×channel / scalar×scalar. Mirrors Dart `_elemwiseOp`: channel×channel
/// requires equal rate AND length. `op` may itself fail (division by zero).
//
// TODO(idl0): perf — maps a fallible `Fn(f64,f64) -> Result` per element and
// allocates a fresh output Vec per node. The per-element `Result` (potential
// early-Err) can defeat LLVM auto-vectorization, and the per-node allocation is
// memory-bandwidth-bound at season scale. When this shows up in a profile:
// (1) split the infallible ops (Add/Sub/Mul/comparisons) into a plain
// `f64 -> f64` slice map so they auto-vectorize, keeping the fallible path only
// for Div; (2) fuse to drop intermediate allocations. Prefer this over a SIMD
// crate (`wide`) — vector width is a target-feature/portability decision, not a
// dependency choice (a baseline build lowers `f64x4` to 2×SSE2 anyway).
pub(crate) fn elemwise(
    left: Value,
    right: Value,
    op_name: &str,
    op: impl Fn(f64, f64) -> Result<f64, MathEvalError>,
) -> Result<Value, MathEvalError> {
    match (left, right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(op(a, b)?)),
        (Value::Channel(a), Value::Channel(b)) => {
            if a.sample_rate_hz != b.sample_rate_hz {
                return Err(err(
                    MathEvalErrorKind::Runtime,
                    format!(
                        "Cannot \"{op_name}\" channels with different sample rates ({} Hz vs {} Hz). \
                         Use resample() to match rates first.",
                        a.sample_rate_hz, b.sample_rate_hz
                    ),
                ));
            }
            if a.samples.len() != b.samples.len() {
                return Err(err(
                    MathEvalErrorKind::Runtime,
                    format!(
                        "\"{op_name}\": channel lengths differ ({} vs {})",
                        a.samples.len(),
                        b.samples.len()
                    ),
                ));
            }
            let mut out = Vec::with_capacity(a.samples.len());
            for i in 0..a.samples.len() {
                out.push(op(a.samples[i], b.samples[i])?);
            }
            Ok(Value::Channel(ChannelValue {
                samples: Arc::from(out),
                sample_rate_hz: a.sample_rate_hz,
                channel_id: None,
            }))
        }
        (Value::Channel(a), Value::Scalar(b)) => {
            let out = a.samples.iter().map(|&x| op(x, b)).collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Channel(ChannelValue {
                samples: Arc::from(out),
                sample_rate_hz: a.sample_rate_hz,
                channel_id: None,
            }))
        }
        (Value::Scalar(a), Value::Channel(b)) => {
            let out = b.samples.iter().map(|&x| op(a, x)).collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Channel(ChannelValue {
                samples: Arc::from(out),
                sample_rate_hz: b.sample_rate_hz,
                channel_id: None,
            }))
        }
        (l, r) => Err(err(
            MathEvalErrorKind::Type,
            format!("\"{op_name}\": unexpected value types ({}, {})", type_name(&l), type_name(&r)),
        )),
    }
}

/// Applies `f` element-wise to a scalar or channel. Mirrors Dart `_mapValue`.
pub(crate) fn map_value(v: Value, f: impl Fn(f64) -> f64) -> Result<Value, MathEvalError> {
    match v {
        Value::Scalar(x) => Ok(Value::Scalar(f(x))),
        Value::Channel(c) => Ok(Value::Channel(ChannelValue {
            samples: c.samples.iter().map(|&x| f(x)).collect(),
            sample_rate_hz: c.sample_rate_hz,
            channel_id: None,
        })),
        Value::Str(_) => {
            Err(err(MathEvalErrorKind::Type, "Cannot apply numeric operation to a string"))
        }
        Value::Vec3(_) => Err(err(
            MathEvalErrorKind::Type,
            "Cannot apply a scalar operation to a 3-vector. Use the vector functions \
             (vadd/vsub/vscale/cross/dot/...) or extract a component first.",
        )),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Channel(_) => "channel",
        Value::Scalar(_) => "scalar",
        Value::Str(_) => "string",
        Value::Vec3(_) => "vec3",
    }
}

// ---- Argument helpers ----

fn require_arg_count(name: &str, args: &[Value], expected: usize) -> Result<(), MathEvalError> {
    if args.len() != expected {
        return Err(err(
            MathEvalErrorKind::ArgCount,
            format!("{name}: expected {expected} argument(s), got {}", args.len()),
        ));
    }
    Ok(())
}

fn require_channel(v: &Value, ctx: &str) -> Result<ChannelValue, MathEvalError> {
    match v {
        Value::Channel(c) => Ok(c.clone()),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected channel argument, got {}", type_name(other)),
        )),
    }
}

fn require_scalar(v: &Value, ctx: &str) -> Result<f64, MathEvalError> {
    match v {
        Value::Scalar(x) => Ok(*x),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected scalar argument, got {}", type_name(other)),
        )),
    }
}

/// Extract a single channel argument's samples for a scalar aggregate `name`
/// (`mean`/`max`/`min`/…, one-argument form). The argument is either a real
/// channel or a `{col[]}` whole-column reference (a rate-0 channel).
fn one_channel(args: &[Value], name: &str) -> Result<Arc<[f64]>, MathEvalError> {
    if args.len() != 1 {
        return Err(err(
            MathEvalErrorKind::ArgCount,
            format!("{name}() takes one channel argument"),
        ));
    }
    Ok(require_channel(&args[0], name)?.samples)
}

fn require_string<'a>(v: &'a Value, ctx: &str) -> Result<&'a str, MathEvalError> {
    match v {
        Value::Str(s) => Ok(s),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected string argument, got {}", type_name(other)),
        )),
    }
}

fn channel(samples: Vec<f64>, sample_rate_hz: f64) -> Value {
    Value::Channel(ChannelValue { samples: Arc::from(samples), sample_rate_hz, channel_id: None })
}

// Like require_channel but also demands a direct-reference channel_id (variance
// needs it to find the same-named overlay channel). Returns (samples, rate, id).
// Mirrors the Dart "argument must be a direct channel reference" guard.
fn require_ref_channel(v: &Value, ctx: &str) -> Result<(Arc<[f64]>, f64, String), MathEvalError> {
    match v {
        Value::Channel(c) => match &c.channel_id {
            Some(id) => Ok((c.samples.clone(), c.sample_rate_hz, id.clone())),
            None => Err(err(
                MathEvalErrorKind::Runtime,
                format!(
                    "{ctx}(): argument must be a direct channel reference (e.g. [LapTime]); \
                     derived expressions are not yet supported."
                ),
            )),
        },
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected channel argument, got {}", type_name(other)),
        )),
    }
}

// The designated main lap's `(start_sec, end_sec)` from the already-uniform
// `main_lap_bounds`, or the `(0.0, 0.0)` gating-off sentinel. Mirrors the
// uniform-time `_mainLapWindow`.
fn main_lap_window(lap_ctx: &MathLapContext) -> (f64, f64) {
    match lap_ctx.main_lap_number {
        Some(n) => {
            lap_ctx.main_lap_bounds.get((n as usize).saturating_sub(1)).copied().unwrap_or((0.0, 0.0))
        }
        None => (0.0, 0.0),
    }
}

// Resolves the per-sample time base for lap-aware functions as a closed-form
// `(len, rate)`: consumers compute `i as f64 / rate` at the point of use, so the
// zero-storage `Time` ramp is never materialized. Prefers an explicit `Time`
// channel's dims; else the highest-rate channel's dims; else an empty 10 Hz base.
// Time is the synthesized uniform ramp (value i/rate), so the closed form is
// exact. Mirrors Dart `_resolveTimeBase`.
fn resolve_time_base(lookup: &dyn ChannelLookup) -> (usize, f64) {
    if let Some((len, rate)) = lookup.channel_dims("Time") {
        if len > 0 && rate > 0.0 {
            return (len, rate);
        }
    }
    if let Some((len, rate)) = lookup.best_time_base_dims() {
        if len > 0 && rate > 0.0 {
            return (len, rate);
        }
    }
    (0, 10.0)
}

// Scalar value at index `i`: the i-th sample of a channel or the scalar value.
// Mirrors Dart `_valueAt`. Out-of-bounds on a channel is a Runtime error.
fn value_at(v: &Value, i: usize) -> Result<f64, MathEvalError> {
    match v {
        Value::Scalar(x) => Ok(*x),
        Value::Channel(c) => c.samples.get(i).copied().ok_or_else(|| {
            err(
                MathEvalErrorKind::Runtime,
                format!("Sample index {i} out of bounds (length {})", c.samples.len()),
            )
        }),
        Value::Str(_) => Err(err(MathEvalErrorKind::Type, "Expected numeric value, got string")),
        Value::Vec3(_) => {
            Err(err(MathEvalErrorKind::Type, "Expected numeric value, got 3-vector"))
        }
    }
}

// Mirrors Dart `double.sign`: NaN→NaN, +/-0→0, else ±1.
fn dart_sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Function-call dispatch. Arms are grouped by task (A6 DSP; A7 stats; A8
/// elementwise/trig; A9 clamp/if/stubs; A10 lap-aware).
fn call_function(
    name: &str,
    args: Vec<Value>,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> Result<Value, MathEvalError> {
    match name {
        // ---- A6: DSP-backed ----
        "integrate" => {
            require_arg_count(name, &args, 1)?;
            let ch = require_channel(&args[0], name)?;
            Ok(channel(crate::integration::integrate(&ch.samples, ch.sample_rate_hz), ch.sample_rate_hz))
        }
        "butter" => {
            require_arg_count(name, &args, 4)?;
            let order = require_scalar(&args[0], name)?;
            let cutoff = require_scalar(&args[1], name)?;
            let ftype = require_string(&args[2], name)?;
            let ch = require_channel(&args[3], name)?;
            let order = order.round() as usize;
            match ftype {
                "high" | "highpass" => Ok(channel(
                    crate::filters::highpass(&ch.samples, order, cutoff, ch.sample_rate_hz),
                    ch.sample_rate_hz,
                )),
                "low" | "lowpass" => Ok(channel(
                    crate::filters::lowpass(&ch.samples, order, cutoff, ch.sample_rate_hz),
                    ch.sample_rate_hz,
                )),
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
        "fft" => {
            require_arg_count(name, &args, 2)?;
            let ch = require_channel(&args[0], name)?;
            let window_str = require_string(&args[1], name)?;
            let window = match window_str {
                "hann" => crate::fft::FftWindow::Hann,
                "hamming" => crate::fft::FftWindow::Hamming,
                "rect" | "rectangular" => crate::fft::FftWindow::Rectangular,
                other => {
                    return Err(err(
                        MathEvalErrorKind::Runtime,
                        format!(
                            "fft: unknown window \"{other}\"; expected \"hann\", \"hamming\", or \"rect\""
                        ),
                    ))
                }
            };
            // Output is n/2+1 bins; preserve the original rate so the caller can
            // compute freq[k] = k * sample_rate_hz / n.
            Ok(channel(crate::fft::fft(&ch.samples, window), ch.sample_rate_hz))
        }
        "declip" => {
            require_arg_count(name, &args, 1)?;
            let ch = require_channel(&args[0], name)?;
            Ok(channel(crate::clip_reconstruct::declip(&ch.samples, ch.sample_rate_hz), ch.sample_rate_hz))
        }

        // ---- A7: time-domain statistics ----
        "differentiate" => {
            require_arg_count(name, &args, 1)?;
            let ch = require_channel(&args[0], name)?;
            Ok(channel(crate::statistics::differentiate(&ch.samples, ch.sample_rate_hz), ch.sample_rate_hz))
        }
        "detrend" => {
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
            Ok(channel(crate::statistics::detrend(&ch.samples, mode), ch.sample_rate_hz))
        }
        "rms" => {
            // 1-arg → scalar aggregate; 2-arg → rolling RMS over a window.
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::rms(&one_channel(&args, "rms")?)))
            } else {
                require_arg_count(name, &args, 2)?;
                let ch = require_channel(&args[0], name)?;
                let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
                Ok(channel(crate::statistics::rolling_rms(&ch.samples, w), ch.sample_rate_hz))
            }
        }
        "mean" => {
            // 1-arg → scalar aggregate; 2-arg → rolling mean over a window.
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::mean(&one_channel(&args, "mean")?)))
            } else {
                require_arg_count(name, &args, 2)?;
                let ch = require_channel(&args[0], name)?;
                let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
                Ok(channel(crate::statistics::rolling_mean(&ch.samples, w), ch.sample_rate_hz))
            }
        }
        "std" => {
            // 1-arg → scalar aggregate (population σ); 2-arg → rolling std.
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::std_pop(&one_channel(&args, "std")?)))
            } else {
                require_arg_count(name, &args, 2)?;
                let ch = require_channel(&args[0], name)?;
                let w = require_scalar(&args[1], name)?.round().max(0.0) as usize;
                Ok(channel(crate::statistics::rolling_std(&ch.samples, w), ch.sample_rate_hz))
            }
        }

        // ---- A8: elementwise math + trig ----
        "abs" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::abs) }
        "sqrt" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::sqrt) }
        "sign" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), dart_sign) }
        "floor" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::floor) }
        "ceil" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::ceil) }
        "round" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::round) }
        "sin" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::sin) }
        "cos" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::cos) }
        "tan" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::tan) }
        "asin" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::asin) }
        "acos" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::acos) }
        "atan" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::atan) }
        "sinh" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::sinh) }
        "cosh" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::cosh) }
        "tanh" => { require_arg_count(name, &args, 1)?; map_value(args.into_iter().next().unwrap(), f64::tanh) }
        "deg2rad" => {
            require_arg_count(name, &args, 1)?;
            map_value(args.into_iter().next().unwrap(), |x| x * std::f64::consts::PI / 180.0)
        }
        "rad2deg" => {
            require_arg_count(name, &args, 1)?;
            map_value(args.into_iter().next().unwrap(), |x| x * 180.0 / std::f64::consts::PI)
        }
        "pow" => {
            require_arg_count(name, &args, 2)?;
            let mut it = args.into_iter();
            elemwise(it.next().unwrap(), it.next().unwrap(), "pow", |a, b| Ok(a.powf(b)))
        }
        "min" => {
            // 1-arg → scalar aggregate (column/series minimum); 2-arg →
            // elementwise minimum of two operands.
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::min(&one_channel(&args, "min")?)))
            } else {
                require_arg_count(name, &args, 2)?;
                let mut it = args.into_iter();
                elemwise(it.next().unwrap(), it.next().unwrap(), "min", |a, b| Ok(a.min(b)))
            }
        }
        "max" => {
            // 1-arg → scalar aggregate (column/series maximum); 2-arg →
            // elementwise maximum of two operands.
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::max(&one_channel(&args, "max")?)))
            } else {
                require_arg_count(name, &args, 2)?;
                let mut it = args.into_iter();
                elemwise(it.next().unwrap(), it.next().unwrap(), "max", |a, b| Ok(a.max(b)))
            }
        }
        "main" => {
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

        // ---- Scalar aggregates (channel/column → scalar). The colliding names
        // (mean/std/rms/min/max/median) are arity-dispatched in their arms
        // above; these have no windowed/elementwise counterpart. ----
        "sum" => Ok(Value::Scalar(crate::math::aggregate::sum(&one_channel(&args, "sum")?))),
        "count" => Ok(Value::Scalar(crate::math::aggregate::count(&one_channel(&args, "count")?))),
        "first" => Ok(Value::Scalar(crate::math::aggregate::first(&one_channel(&args, "first")?))),
        "last" => Ok(Value::Scalar(crate::math::aggregate::last(&one_channel(&args, "last")?))),
        "median" => {
            // 1-arg → scalar aggregate. A 2-arg rolling median remains deferred
            // (preserve the exact deferred-stub message for parity).
            if args.len() == 1 {
                Ok(Value::Scalar(crate::math::aggregate::median(&one_channel(&args, "median")?)))
            } else {
                Err(err(MathEvalErrorKind::NotImplemented, "not yet implemented: median".to_string()))
            }
        }
        "p" => {
            // p(channel, quantile) → linear-interpolated percentile (quantile in 0..=100).
            if args.len() != 2 {
                return Err(err(MathEvalErrorKind::ArgCount, "p(channel, quantile) takes 2 args"));
            }
            let ch = require_channel(&args[0], "p")?;
            let q = require_scalar(&args[1], "p quantile")?;
            Ok(Value::Scalar(crate::math::aggregate::percentile(&ch.samples, q)))
        }
        "atan2" => {
            require_arg_count(name, &args, 2)?;
            let mut it = args.into_iter();
            elemwise(it.next().unwrap(), it.next().unwrap(), "atan2", |a, b| Ok(a.atan2(b)))
        }

        // ---- A9: clamp, if, deferred stubs ----
        "clamp" => {
            require_arg_count(name, &args, 3)?;
            let ch = require_channel(&args[0], name)?;
            let lo = require_scalar(&args[1], name)?;
            let hi = require_scalar(&args[2], name)?;
            Ok(channel(ch.samples.iter().map(|&x| x.clamp(lo, hi)).collect(), ch.sample_rate_hz))
        }
        "if" => {
            require_arg_count(name, &args, 3)?;
            let cond = require_channel(&args[0], "if(cond,t,f) — cond")?;
            let n = cond.samples.len();
            let mut out = vec![0.0; n];
            for i in 0..n {
                out[i] = if cond.samples[i] != 0.0 {
                    value_at(&args[1], i)?
                } else {
                    value_at(&args[2], i)?
                };
            }
            Ok(channel(out, cond.sample_rate_hz))
        }
        "spectrogram" | "hilbert" | "correlate" | "convolve" | "resample" | "sosfilt" => {
            Err(err(MathEvalErrorKind::NotImplemented, format!("not yet implemented: {name}")))
        }

        // ---- A10: lap-aware (consume injected MathLapContext) ----
        "current_lap" => {
            require_arg_count(name, &args, 0)?;
            let (len, rate) = resolve_time_base(lookup);
            let out = (0..len)
                .map(|i| {
                    crate::variance::current_lap_at(&lap_ctx.main_lap_bounds, i as f64 / rate) as f64
                })
                .collect();
            Ok(channel(out, rate))
        }
        "lap_start_time" => {
            require_arg_count(name, &args, 1)?;
            match &args[0] {
                Value::Scalar(n) => Ok(Value::Scalar(crate::variance::lap_start_time(
                    &lap_ctx.main_lap_bounds,
                    n.round() as u32,
                ))),
                Value::Channel(c) => {
                    let out = c
                        .samples
                        .iter()
                        .map(|&n| {
                            if n.is_nan() || n <= 0.0 || !n.is_finite() {
                                f64::NAN
                            } else {
                                crate::variance::lap_start_time(&lap_ctx.main_lap_bounds, n.round() as u32)
                            }
                        })
                        .collect();
                    Ok(channel(out, c.sample_rate_hz))
                }
                other => Err(err(
                    MathEvalErrorKind::Type,
                    format!("lap_start_time(): expected scalar or channel, got {}", type_name(other)),
                )),
            }
        }
        "lap_start_distance" => {
            require_arg_count(name, &args, 1)?;
            let distance = lookup.lookup("Distance").ok_or_else(|| {
                err(
                    MathEvalErrorKind::Runtime,
                    "lap_start_distance(): no [Distance] channel in this session (GPS_SpeedKmh missing).",
                )
            })?;
            if distance.samples.is_empty() {
                return Err(err(
                    MathEvalErrorKind::Runtime,
                    "lap_start_distance(): no [Distance] channel in this session (GPS_SpeedKmh missing).",
                ));
            }
            let bounds = &lap_ctx.main_lap_bounds;
            let dist_rate = distance.sample_rate_hz;
            let dist_samples = distance.samples; // owned Vec<f64>, moved in
            let distance_at_lap = |n: i64| -> f64 {
                if n <= 0 || n as usize > bounds.len() {
                    return f64::NAN;
                }
                let t_sec = bounds[(n - 1) as usize].0;
                let idx = (t_sec * dist_rate).round() as i64;
                if idx < 0 || idx as usize >= dist_samples.len() {
                    return f64::NAN;
                }
                dist_samples[idx as usize]
            };
            match &args[0] {
                Value::Scalar(n) => Ok(Value::Scalar(distance_at_lap(n.round() as i64))),
                Value::Channel(c) => {
                    let out = c
                        .samples
                        .iter()
                        .map(|&n| {
                            if n.is_nan() || n <= 0.0 || !n.is_finite() {
                                f64::NAN
                            } else {
                                distance_at_lap(n.round() as i64)
                            }
                        })
                        .collect();
                    Ok(channel(out, c.sample_rate_hz))
                }
                other => Err(err(
                    MathEvalErrorKind::Type,
                    format!("lap_start_distance(): expected scalar or channel, got {}", type_name(other)),
                )),
            }
        }
        "sector_number" => {
            require_arg_count(name, &args, 0)?;
            let (len, rate) = resolve_time_base(lookup);
            if lap_ctx.main_sectors.is_empty() {
                return Ok(channel(vec![f64::NAN; len], rate));
            }
            let out = (0..len)
                .map(|i| {
                    let idx =
                        crate::variance::sector_number_at(&lap_ctx.main_sectors, i as f64 / rate);
                    if idx == u32::MAX {
                        f64::NAN
                    } else {
                        idx as f64
                    }
                })
                .collect();
            Ok(channel(out, rate))
        }

        // ---- B2: variance (overlay second handle) ----
        "variance_time" => {
            require_arg_count(name, &args, 1)?;
            let overlay = lap_ctx
                .overlay
                .as_ref()
                .filter(|_| lap_ctx.main_lap_number.is_some())
                .ok_or_else(|| {
                    err(
                        MathEvalErrorKind::NoLapContext,
                        "variance_time(): requires a main lap AND an overlay lap to be designated. \
                         Pick both in the Analyze lap table.",
                    )
                })?;
            let (main_samples, main_rate, channel_id) = require_ref_channel(&args[0], "variance_time")?;
            crate::math::variance_geom::eval_variance_time(
                &main_samples,
                main_rate,
                &channel_id,
                lookup,
                overlay,
                main_lap_window(lap_ctx),
            )
        }
        "variance_dist" => {
            require_arg_count(name, &args, 1)?;
            let overlay = lap_ctx
                .overlay
                .as_ref()
                .filter(|_| lap_ctx.main_lap_number.is_some())
                .ok_or_else(|| {
                    err(
                        MathEvalErrorKind::NoLapContext,
                        "variance_dist(): requires a main lap AND an overlay lap to be designated. \
                         Pick both in the Analyze lap table.",
                    )
                })?;
            let (main_samples, main_rate, channel_id) = require_ref_channel(&args[0], "variance_dist")?;
            crate::math::variance_geom::eval_variance_dist(
                &main_samples,
                main_rate,
                &channel_id,
                lookup,
                overlay,
                main_lap_window(lap_ctx),
            )
        }

        // ---- Vector & rotation primitives (math::vector, SPEC §19) ----
        "vec" => {
            require_arg_count(name, &args, 3)?;
            crate::math::vector::make_vec3(&args[0], &args[1], &args[2], name)
        }
        "vx" => {
            require_arg_count(name, &args, 1)?;
            crate::math::vector::component(&args[0], 0, name)
        }
        "vy" => {
            require_arg_count(name, &args, 1)?;
            crate::math::vector::component(&args[0], 1, name)
        }
        "vz" => {
            require_arg_count(name, &args, 1)?;
            crate::math::vector::component(&args[0], 2, name)
        }
        "vadd" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::vadd(&args[0], &args[1], name)
        }
        "vsub" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::vsub(&args[0], &args[1], name)
        }
        "vscale" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::vscale(&args[0], &args[1], name)
        }
        "cross" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::cross(&args[0], &args[1], name)
        }
        "dot" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::dot(&args[0], &args[1], name)
        }
        "norm" => {
            require_arg_count(name, &args, 1)?;
            crate::math::vector::norm(&args[0], name)
        }
        "normalize" => {
            require_arg_count(name, &args, 1)?;
            crate::math::vector::normalize(&args[0], name)
        }
        "angle" => {
            require_arg_count(name, &args, 2)?;
            crate::math::vector::angle(&args[0], &args[1], name)
        }
        "rotate_mat" => {
            // v + 9 row-major matrix entries (constant scalars).
            require_arg_count(name, &args, 10)?;
            let mut m = [0.0_f64; 9];
            for (slot, arg) in m.iter_mut().zip(&args[1..]) {
                *slot = require_scalar(arg, name)?;
            }
            crate::math::vector::rotate_mat(&args[0], &m, name)
        }
        "rotate_axis" => {
            // v, ax, ay, az, angle — constant scalars.
            require_arg_count(name, &args, 5)?;
            let ax = require_scalar(&args[1], name)?;
            let ay = require_scalar(&args[2], name)?;
            let az = require_scalar(&args[3], name)?;
            let ang = require_scalar(&args[4], name)?;
            crate::math::vector::rotate_axis(&args[0], ax, ay, az, ang, name)
        }
        "rotate_euler" => {
            // v, roll, pitch, yaw — angle args may be channels (per-sample).
            require_arg_count(name, &args, 4)?;
            crate::math::vector::rotate_euler(&args[0], &args[1], &args[2], &args[3], name)
        }

        _ => Err(err(MathEvalErrorKind::UnknownFunction, format!("unknown function \"{name}\""))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // A minimal in-memory lookup for tests.
    struct MapLookup(std::collections::HashMap<String, (Vec<f64>, f64)>);
    impl ChannelLookup for MapLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            self.0.get(name).map(|(s, r)| LookupChannel { samples: s.clone().into(), sample_rate_hz: *r })
        }
        fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
            // Mirror SessionHandle: (len, rate) without materializing, so the
            // closed-form time base resolves a `Time` channel here too.
            self.0.get(name).map(|(s, r)| (s.len(), *r))
        }
    }
    fn lookup(pairs: &[(&str, Vec<f64>, f64)]) -> MapLookup {
        MapLookup(pairs.iter().cloned().map(|(n, s, r)| (n.to_string(), (s, r))).collect())
    }
    fn no_laps() -> MathLapContext {
        MathLapContext::default()
    }

    fn eval_expr(src: &str, lk: &dyn ChannelLookup) -> Result<Value, crate::math::MathEvalError> {
        let ast = crate::math::parse::parse(src).unwrap();
        eval(&ast, lk, &no_laps())
    }

    #[test]
    fn empty_lap_context_has_no_laps_or_overlay() {
        // Act
        let ctx = MathLapContext::empty();

        // Assert
        assert!(ctx.main_lap_bounds.is_empty());
        assert!(ctx.main_sectors.is_empty());
        assert_eq!(ctx.main_lap_number, None);
        assert!(ctx.overlay.is_none());
    }

    #[test]
    fn channel_plus_scalar_adds_elementwise() {
        // Arrange
        let lk = lookup(&[("x", vec![1.0, 2.0, 3.0], 10.0)]);

        // Act
        let v = eval_expr("[x] + 10", &lk).unwrap();

        // Assert
        match v {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![11.0, 12.0, 13.0].into());
                assert_eq!(c.sample_rate_hz, 10.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn multi_reference_widening_preserves_values() {
        // Arrange — `a` referenced three times. The per-reference Arc share
        // (and the per-pass memo) must preserve values exactly, not corrupt
        // them. [a] + [a] - [a] == [a].
        let lk = lookup(&[("a", vec![1.0, 2.0, 3.0], 10.0)]);

        // Act
        let v = eval_expr("[a] + [a] - [a]", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![1.0, 2.0, 3.0].into()));
    }

    #[test]
    fn repeated_reference_widens_source_once() {
        use std::cell::RefCell;
        use std::collections::HashMap;

        // A lookup double that counts how many times each channel is widened.
        struct CountingLookup {
            data: HashMap<String, (Vec<f64>, f64)>,
            counts: RefCell<HashMap<String, usize>>,
        }
        impl ChannelLookup for CountingLookup {
            fn lookup(&self, name: &str) -> Option<LookupChannel> {
                *self.counts.borrow_mut().entry(name.to_string()).or_insert(0) += 1;
                self.data
                    .get(name)
                    .map(|(s, r)| LookupChannel { samples: s.clone().into(), sample_rate_hz: *r })
            }
        }

        // Arrange — `A` referenced four times in one expression.
        let lk = CountingLookup {
            data: HashMap::from([("A".to_string(), (vec![1.0, 2.0, 3.0], 10.0))]),
            counts: RefCell::new(HashMap::new()),
        };

        // Act
        let out = evaluate("[A] + [A] + [A] + [A]", &lk, &MathLapContext::empty()).unwrap();

        // Assert — value correct AND the source was widened exactly once (memo).
        assert_eq!(out.samples, vec![4.0, 8.0, 12.0]);
        assert_eq!(*lk.counts.borrow().get("A").unwrap(), 1);
    }

    #[test]
    fn time_base_is_closed_form_and_never_widens_the_ramp() {
        use std::cell::RefCell;

        // Records every full widen via `lookup`; serves dims cheaply.
        struct TimeProbe {
            len: usize,
            rate: f64,
            widen_calls: RefCell<usize>,
        }
        impl ChannelLookup for TimeProbe {
            fn lookup(&self, name: &str) -> Option<LookupChannel> {
                *self.widen_calls.borrow_mut() += 1;
                (name == "Time").then(|| {
                    let s: Vec<f64> = (0..self.len).map(|i| i as f64 / self.rate).collect();
                    LookupChannel { samples: s.into(), sample_rate_hz: self.rate }
                })
            }
            fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
                (name == "Time").then_some((self.len, self.rate))
            }
        }

        // Arrange
        let probe = TimeProbe { len: 50_000, rate: 833.0, widen_calls: RefCell::new(0) };

        // Act — the closed-form time base.
        let base = resolve_time_base(&probe);

        // Assert — dims returned, and `Time` was NEVER widened via lookup().
        assert_eq!(base, (50_000, 833.0));
        assert_eq!(*probe.widen_calls.borrow(), 0);
    }

    #[test]
    fn channel_times_channel_requires_matching_rate() {
        // Arrange — different rates must error (Runtime kind).
        let lk = lookup(&[("a", vec![1.0, 2.0], 10.0), ("b", vec![1.0, 2.0], 20.0)]);

        // Act
        let err = eval_expr("[a] * [b]", &lk).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::Runtime);
        assert!(err.message.contains("sample rates"));
    }

    #[test]
    fn division_by_zero_is_typed_error() {
        // Arrange
        let lk = lookup(&[("x", vec![1.0], 1.0)]);

        // Act
        let err = eval_expr("[x] / 0", &lk).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::DivisionByZero);
    }

    #[test]
    fn unary_minus_negates_channel() {
        // Arrange
        let lk = lookup(&[("x", vec![1.0, -2.0], 1.0)]);

        // Act
        let v = eval_expr("-[x]", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![-1.0, 2.0].into()));
    }

    #[test]
    fn comparison_yields_one_or_zero() {
        // Arrange
        let lk = lookup(&[("x", vec![1.0, 5.0, 9.0], 1.0)]);

        // Act
        let v = eval_expr("[x] > 4", &lk).unwrap();

        // Assert — 1>4=0, 5>4=1, 9>4=1.
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![0.0, 1.0, 1.0].into()));
    }

    #[test]
    fn unknown_channel_is_typed_error() {
        // Arrange
        let lk = lookup(&[]);

        // Act
        let err = eval_expr("[nope]", &lk).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn cellref_resolves_scalar_when_lookup_provides_it() {
        // Arrange — a lookup that knows cell `x = 2.5`.
        struct CellAware;
        impl ChannelLookup for CellAware {
            fn lookup(&self, _: &str) -> Option<LookupChannel> {
                None
            }
            fn lookup_cell(&self, name: &str) -> Option<f64> {
                if name == "x" {
                    Some(2.5)
                } else {
                    None
                }
            }
        }
        // Act
        let out = evaluate("{x} * 2", &CellAware, &MathLapContext::empty()).unwrap();
        // Assert — scalar result: rate 0, one sample.
        assert_eq!(out.sample_rate_hz, 0.0);
        assert_eq!(out.samples, vec![5.0]);
    }

    #[test]
    fn main_fn_returns_baseline_row_element_else_nan() {
        // Arrange — a lookup whose column "v" is [10, 20, 30].
        struct ColAware;
        impl ChannelLookup for ColAware {
            fn lookup(&self, _: &str) -> Option<LookupChannel> { None }
            fn lookup_cell_column(&self, name: &str) -> Option<Vec<f64>> {
                (name == "v").then(|| vec![10.0, 20.0, 30.0])
            }
        }
        let with_row = MathLapContext { baseline_row: Some(1), ..MathLapContext::empty() };
        let no_row = MathLapContext::empty();

        // Act
        let hit = evaluate("main({v[]})", &ColAware, &with_row).unwrap();
        let miss = evaluate("main({v[]})", &ColAware, &no_row).unwrap();

        // Assert — baseline row 1 → 20.0; no baseline → NaN.
        assert_eq!(hit.samples, vec![20.0]);
        assert!(miss.samples[0].is_nan());
    }

    #[test]
    fn cellref_errors_under_channel_only_lookup() {
        // The Maths-editor lookup never provides cells → {x} is unknown.
        struct ChannelOnly;
        impl ChannelLookup for ChannelOnly {
            fn lookup(&self, _: &str) -> Option<LookupChannel> {
                None
            }
        }
        let err = evaluate("{x}", &ChannelOnly, &MathLapContext::empty()).unwrap_err();
        assert_eq!(err.kind, MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn evaluate_scalar_returns_single_value() {
        struct Empty;
        impl ChannelLookup for Empty {
            fn lookup(&self, _: &str) -> Option<LookupChannel> {
                None
            }
        }
        assert_eq!(evaluate_scalar("1 + 2 * 3", &Empty, &MathLapContext::empty()).unwrap(), 7.0);
    }

    #[test]
    fn evaluate_scalar_rejects_a_channel_result() {
        // A lookup returning a real (multi-sample) channel → not a scalar → error.
        struct OneChannel;
        impl ChannelLookup for OneChannel {
            fn lookup(&self, name: &str) -> Option<LookupChannel> {
                (name == "Fork").then(|| LookupChannel { samples: vec![1.0, 2.0].into(), sample_rate_hz: 10.0 })
            }
        }
        assert!(evaluate_scalar("[Fork]", &OneChannel, &MathLapContext::empty()).is_err());
    }

    #[test]
    fn scalar_arithmetic_stays_scalar() {
        // Arrange
        let lk = lookup(&[]);

        // Act
        let v = eval_expr("2 + 3 * 4", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Scalar(x) if x == 14.0));
    }

    #[test]
    fn top_level_evaluate_wraps_scalar_as_one_sample_rate_zero() {
        // Arrange
        let lk = lookup(&[]);

        // Act
        let out = evaluate("42", &lk, &no_laps()).unwrap();

        // Assert — matches Dart: scalar → (samples:[v], sampleRateHz:0).
        assert_eq!(out.samples, vec![42.0]);
        assert_relative_eq!(out.sample_rate_hz, 0.0, epsilon = 1e-12);
    }

    // ---- A6: DSP-backed functions ----

    #[test]
    fn integrate_constant_channel_ramps_linearly() {
        // Arrange — constant 2.0 at 10 Hz; cumulative trapezoid → i * 2 * 0.1.
        let lk = lookup(&[("a", vec![2.0; 5], 10.0)]);

        // Act
        let v = eval_expr("integrate([a])", &lk).unwrap();

        // Assert
        match v {
            Value::Channel(c) => {
                assert_eq!(c.sample_rate_hz, 10.0);
                assert_relative_eq!(c.samples[0], 0.0, epsilon = 1e-12);
                assert_relative_eq!(c.samples[4], 2.0 * 4.0 * 0.1, epsilon = 1e-9);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn butter_unknown_type_errors() {
        // Arrange
        let lk = lookup(&[("a", vec![0.0; 8], 100.0)]);

        // Act
        let err = eval_expr("butter(2, 0.3, \"weird\", [a])", &lk).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::Runtime);
        assert!(err.message.contains("unknown type"));
    }

    #[test]
    fn butter_band_is_not_implemented_error_message() {
        // Arrange
        let lk = lookup(&[("a", vec![0.0; 8], 100.0)]);

        // Act
        let err = eval_expr("butter(2, 0.3, \"band\", [a])", &lk).unwrap_err();

        // Assert — Dart surfaces "band-pass filter not yet implemented".
        assert!(err.message.contains("band-pass"));
    }

    #[test]
    fn fft_unknown_window_errors() {
        // Arrange
        let lk = lookup(&[("a", vec![0.0; 8], 100.0)]);

        // Act
        let err = eval_expr("fft([a], \"blackman\")", &lk).unwrap_err();

        // Assert
        assert!(err.message.contains("unknown window"));
    }

    #[test]
    fn integrate_wrong_arg_count_errors() {
        // Arrange
        let lk = lookup(&[("a", vec![1.0], 1.0)]);

        // Act
        let err = eval_expr("integrate([a], 2)", &lk).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::ArgCount);
    }

    // ---- A7: time-domain statistics ----

    #[test]
    fn differentiate_ramp_channel_yields_constant() {
        // Arrange — ramp i at 10 Hz; backward diff → 10.0 after the first sample.
        let lk = lookup(&[("a", (0..5).map(|i| i as f64).collect(), 10.0)]);

        // Act
        let v = eval_expr("differentiate([a])", &lk).unwrap();

        // Assert
        match v {
            Value::Channel(c) => {
                assert_relative_eq!(c.samples[0], 0.0, epsilon = 1e-12);
                assert_relative_eq!(c.samples[3], 10.0, epsilon = 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rolling_mean_function_smooths_to_constant() {
        // Arrange
        let lk = lookup(&[("a", vec![5.0; 6], 1.0)]);

        // Act
        let v = eval_expr("mean([a], 3)", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples.iter().all(|&x| (x - 5.0).abs() < 1e-12)));
    }

    #[test]
    fn rms_one_arg_is_aggregate_two_arg_is_rolling() {
        // Arrange — rms([3,4]) aggregate = sqrt((9+16)/2) = sqrt(12.5).
        let lk = lookup(&[("a", vec![3.0, 4.0], 1.0)]);

        // Act / Assert — 1-arg is the scalar aggregate; 2-arg is rolling (a
        // channel); 3 args is an arity error.
        assert!(
            matches!(eval_expr("rms([a])", &lk).unwrap(), Value::Scalar(x) if (x - 12.5_f64.sqrt()).abs() < 1e-12)
        );
        assert!(matches!(eval_expr("rms([a], 2)", &lk).unwrap(), Value::Channel(_)));
        assert_eq!(
            eval_expr("rms([a], 2, 3)", &lk).unwrap_err().kind,
            crate::math::MathEvalErrorKind::ArgCount
        );
    }

    #[test]
    fn aggregate_reduces_channel_to_scalar() {
        struct Ch;
        impl ChannelLookup for Ch {
            fn lookup(&self, name: &str) -> Option<LookupChannel> {
                (name == "F").then(|| LookupChannel { samples: vec![1.0, 2.0, 3.0].into(), sample_rate_hz: 10.0 })
            }
        }
        let out = evaluate_scalar("max([F]) - min([F])", &Ch, &MathLapContext::empty()).unwrap();
        assert_eq!(out, 2.0);
    }

    #[test]
    fn detrend_default_mode_is_linear_and_preserves_rate_and_length() {
        // Arrange — a pure ramp at 50 Hz; default (no mode arg) is linear.
        let lk = lookup(&[("a", vec![1.0, 2.0, 3.0, 4.0, 5.0], 50.0)]);

        // Act
        let v = eval_expr("detrend([a])", &lk).unwrap();

        // Assert — ramp removed; rate and length inherited from the input.
        match v {
            Value::Channel(c) => {
                assert_eq!(c.sample_rate_hz, 50.0);
                assert_eq!(c.samples.len(), 5);
                for s in c.samples.iter() {
                    assert_relative_eq!(*s, 0.0, epsilon = 1e-12);
                }
            }
            _ => panic!("expected a channel"),
        }
    }

    #[test]
    fn detrend_explicit_linear_mode_matches_default() {
        // Arrange — passing "linear" explicitly is identical to the default.
        let lk = lookup(&[("a", vec![2.0, 3.0, 14.0, 5.0, 6.0], 10.0)]);

        // Act
        let v = eval_expr("detrend([a], \"linear\")", &lk).unwrap();

        // Assert — same trend-removed-with-spike result as default linear.
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![-2.0, -2.0, 8.0, -2.0, -2.0].into()));
    }

    #[test]
    fn detrend_constant_mode_removes_only_the_mean() {
        // Arrange — mean is 6; constant mode removes it without touching slope.
        let lk = lookup(&[("a", vec![2.0, 3.0, 14.0, 5.0, 6.0], 10.0)]);

        // Act
        let v = eval_expr("detrend([a], \"constant\")", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![-4.0, -3.0, 8.0, -1.0, 0.0].into()));
    }

    #[test]
    fn detrend_mean_is_an_alias_for_constant() {
        // Arrange — the spectral vocab calls constant detrend "mean".
        let lk = lookup(&[("a", vec![2.0, 3.0, 14.0, 5.0, 6.0], 10.0)]);

        // Act
        let v = eval_expr("detrend([a], \"mean\")", &lk).unwrap();

        // Assert — identical to "constant".
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![-4.0, -3.0, 8.0, -1.0, 0.0].into()));
    }

    #[test]
    fn detrend_none_mode_passes_through() {
        // Arrange
        let lk = lookup(&[("a", vec![2.0, 3.0, 14.0, 5.0, 6.0], 10.0)]);

        // Act
        let v = eval_expr("detrend([a], \"none\")", &lk).unwrap();

        // Assert — input unchanged.
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![2.0, 3.0, 14.0, 5.0, 6.0].into()));
    }

    #[test]
    fn detrend_unknown_mode_is_rejected() {
        // Arrange — invalid mode, mirroring butter's rejected direction arg.
        let lk = lookup(&[("a", vec![1.0, 2.0, 3.0], 10.0)]);

        // Act
        let err = eval_expr("detrend([a], \"quadratic\")", &lk).unwrap_err();

        // Assert — same error path/kind as butter's bad string arg.
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::Runtime);
        assert!(err.message.contains("unknown mode"));
    }

    #[test]
    fn detrend_wrong_arg_count_errors() {
        // Arrange
        let lk = lookup(&[("a", vec![1.0, 2.0, 3.0], 10.0)]);

        // Act / Assert — zero args and three args are both arity errors.
        assert_eq!(
            eval_expr("detrend()", &lk).unwrap_err().kind,
            crate::math::MathEvalErrorKind::ArgCount
        );
        assert_eq!(
            eval_expr("detrend([a], \"linear\", 2)", &lk).unwrap_err().kind,
            crate::math::MathEvalErrorKind::ArgCount
        );
    }

    // ---- A8: elementwise math + trig ----

    #[test]
    fn abs_and_sqrt_map_over_channel() {
        // Arrange
        let lk = lookup(&[("a", vec![-4.0, 9.0], 1.0)]);

        // Act
        let v = eval_expr("sqrt(abs([a]))", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![2.0, 3.0].into()));
    }

    #[test]
    fn pow_is_two_arg_elementwise() {
        // Arrange
        let lk = lookup(&[("a", vec![2.0, 3.0], 1.0)]);

        // Act
        let v = eval_expr("pow([a], 2)", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![4.0, 9.0].into()));
    }

    #[test]
    fn sign_matches_dart_semantics_for_zero() {
        // Arrange
        let lk = lookup(&[("a", vec![-2.0, 0.0, 5.0], 1.0)]);

        // Act
        let v = eval_expr("sign([a])", &lk).unwrap();

        // Assert — 0 stays 0 (not +1 as f64::signum would give).
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![-1.0, 0.0, 1.0].into()));
    }

    #[test]
    fn deg2rad_then_sin_of_90_is_one() {
        // Arrange
        let lk = lookup(&[]);

        // Act
        let v = eval_expr("sin(deg2rad(90))", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Scalar(x) if (x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn min_max_two_arg_elementwise() {
        // Arrange
        let lk = lookup(&[("a", vec![1.0, 8.0], 1.0)]);

        // Act / Assert
        assert!(matches!(eval_expr("min([a], 5)", &lk).unwrap(), Value::Channel(c) if c.samples == vec![1.0, 5.0].into()));
        assert!(matches!(eval_expr("max([a], 5)", &lk).unwrap(), Value::Channel(c) if c.samples == vec![5.0, 8.0].into()));
    }

    // ---- A9: clamp, if, deferred stubs ----

    #[test]
    fn clamp_limits_channel_to_range() {
        // Arrange
        let lk = lookup(&[("a", vec![-5.0, 0.5, 9.0], 1.0)]);

        // Act
        let v = eval_expr("clamp([a], 0, 1)", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![0.0, 0.5, 1.0].into()));
    }

    #[test]
    fn if_selects_per_sample_from_condition() {
        // Arrange — cond = [a] > 0; pick 100 when true, -1 when false.
        let lk = lookup(&[("a", vec![-2.0, 3.0, -1.0], 5.0)]);

        // Act
        let v = eval_expr("if([a] > 0, 100, -1)", &lk).unwrap();

        // Assert — length and rate follow the condition channel.
        match v {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![-1.0, 100.0, -1.0].into());
                assert_eq!(c.sample_rate_hz, 5.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn deferred_stub_reports_not_implemented() {
        // Arrange
        let lk = lookup(&[("a", vec![1.0], 1.0)]);

        // Act
        let err = eval_expr("median([a], 3)", &lk).unwrap_err();

        // Assert — exact Dart message + NotImplemented kind.
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::NotImplemented);
        assert_eq!(err.message, "not yet implemented: median");
    }

    // ---- A10: lap-aware ----

    fn laps_ctx(bounds: Vec<(f64, f64)>) -> MathLapContext {
        MathLapContext {
            main_lap_bounds: bounds,
            main_sectors: Vec::new(),
            main_lap_number: None,
            overlay: None,
            baseline_row: None,
        }
    }
    fn eval_with_laps(src: &str, lk: &dyn ChannelLookup, ctx: &MathLapContext) -> Value {
        eval(&crate::math::parse::parse(src).unwrap(), lk, ctx).unwrap()
    }

    #[test]
    fn current_lap_labels_each_time_sample() {
        // Arrange — Time = 0,1,2,3,4 s at 1 Hz; lap 1 = [0,2), lap 2 = [2,4).
        let lk = lookup(&[("Time", vec![0.0, 1.0, 2.0, 3.0, 4.0], 1.0)]);
        let ctx = laps_ctx(vec![(0.0, 2.0), (2.0, 4.0)]);

        // Act
        let v = eval_with_laps("current_lap()", &lk, &ctx);

        // Assert — 1,1,2,2,0 (4.0 is outside both windows; end is exclusive).
        assert!(matches!(v, Value::Channel(c) if c.samples == vec![1.0, 1.0, 2.0, 2.0, 0.0].into()));
    }

    #[test]
    fn lap_start_time_scalar_returns_window_start() {
        // Arrange
        let lk = lookup(&[]);
        let ctx = laps_ctx(vec![(0.0, 2.0), (2.0, 4.0)]);

        // Act
        let v = eval_with_laps("lap_start_time(2)", &lk, &ctx);

        // Assert
        assert!(matches!(v, Value::Scalar(x) if x == 2.0));
    }

    #[test]
    fn lap_time_expression_subtracts_lap_start() {
        // Arrange — the tutorial LapTime: Time - lap_start_time(current_lap()).
        let lk = lookup(&[("Time", vec![0.0, 1.0, 2.0, 3.0], 1.0)]);
        let ctx = laps_ctx(vec![(0.0, 2.0), (2.0, 4.0)]);

        // Act
        let v = eval_with_laps("[Time] - lap_start_time(current_lap())", &lk, &ctx);

        // Assert — within lap1: 0,1; within lap2: 0,1.
        match v {
            Value::Channel(c) => assert_eq!(c.samples, vec![0.0, 1.0, 0.0, 1.0].into()),
            _ => panic!(),
        }
    }

    #[test]
    fn sector_number_all_nan_when_no_sectors() {
        // Arrange
        let lk = lookup(&[("Time", vec![0.0, 1.0], 1.0)]);
        let ctx = laps_ctx(vec![(0.0, 2.0)]);

        // Act
        let v = eval_with_laps("sector_number()", &lk, &ctx);

        // Assert
        assert!(matches!(v, Value::Channel(c) if c.samples.iter().all(|x| x.is_nan())));
    }

    #[test]
    fn lap_start_distance_indexes_distance_channel() {
        // Arrange — Distance ramps 0,10,20,30 at 1 Hz; lap 2 starts at t=2 → idx 2 → 20 m.
        let lk = lookup(&[("Distance", vec![0.0, 10.0, 20.0, 30.0], 1.0)]);
        let ctx = laps_ctx(vec![(0.0, 2.0), (2.0, 4.0)]);

        // Act
        let v = eval_with_laps("lap_start_distance(2)", &lk, &ctx);

        // Assert
        assert!(matches!(v, Value::Scalar(x) if x == 20.0));
    }

    // ---- B2: variance (overlay second handle) ----

    #[test]
    fn variance_time_identity_overlay_is_near_zero() {
        // Arrange — main == overlay: straight-east lap, channel = sample index.
        // GPS at 1 Hz: lat const, lon increasing; EpochMs 0..9000 ms.
        let lon: Vec<f64> = (0..10).map(|i| i as f64 * 0.001).collect();
        let lat = vec![0.0; 10];
        let epoch: Vec<f64> = (0..10).map(|i| (i * 1000) as f64).collect();
        let chan: Vec<f64> = (0..10).map(|i| i as f64).collect();

        let main = lookup(&[
            ("GPS_Latitude", lat.clone(), 1.0),
            ("GPS_Longitude", lon.clone(), 1.0),
            ("GPS_EpochMs", epoch.clone(), 1.0),
            ("LapTime", chan.clone(), 1.0),
        ]);
        let overlay = lookup(&[
            ("GPS_Latitude", lat, 1.0),
            ("GPS_Longitude", lon, 1.0),
            ("GPS_EpochMs", epoch, 1.0),
            ("LapTime", chan, 1.0),
        ]);
        let ctx = MathLapContext {
            main_lap_bounds: vec![(0.0, 9.0)],
            main_sectors: Vec::new(),
            main_lap_number: Some(1),
            overlay: Some(MathOverlay {
                lookup: std::sync::Arc::new(overlay),
                lap_start_ms: 0.0,
                lap_end_ms: 9000.0,
                lap_start_uniform_sec: 0.0,
            }),
            baseline_row: None,
        };

        // Act
        let v = eval(&crate::math::parse::parse("variance_time([LapTime])").unwrap(), &main, &ctx)
            .unwrap();

        // Assert — identity main==overlay → diff ≈ 0 inside the lap window.
        match v {
            Value::Channel(c) => {
                let inside: Vec<f64> = c.samples.iter().copied().filter(|x| !x.is_nan()).collect();
                assert!(!inside.is_empty(), "expected some in-window samples");
                for x in inside {
                    assert!(x.abs() < 1e-3, "expected ~0, got {x}");
                }
            }
            _ => panic!(),
        }
    }

    // ---- Vector & rotation primitives (parser-level integration) ----

    #[test]
    fn vec_then_component_round_trips_through_parser() {
        // Arrange
        let lk = lookup(&[]);

        // Act — extract the y component of an assembled vector.
        let v = eval_expr("vy(vec(1, 2, 3))", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Scalar(x) if x == 2.0));
    }

    #[test]
    fn top_level_vector_is_rejected_with_extract_hint() {
        // Arrange
        let lk = lookup(&[]);

        // Act — a bare vector cannot be plotted.
        let e = evaluate("vec(1, 2, 3)", &lk, &no_laps()).unwrap_err();

        // Assert
        assert_eq!(e.kind, crate::math::MathEvalErrorKind::Type);
        assert!(e.message.contains("vx()"));
    }

    #[test]
    fn cross_norm_of_unit_axes_is_one_through_parser() {
        // Arrange
        let lk = lookup(&[]);

        // Act — |x̂ × ŷ| = |ẑ| = 1.
        let v = eval_expr("norm(cross(vec(1,0,0), vec(0,1,0)))", &lk).unwrap();

        // Assert
        assert!(matches!(v, Value::Scalar(x) if (x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn rotate_axis_through_parser_maps_x_to_y() {
        // Arrange — π/2 as a literal (independent of the universal-constant lane).
        let lk = lookup(&[]);

        // Act — rotate x̂ by 90° about Z → ŷ; extract y.
        let v = eval_expr("vy(rotate_axis(vec(1,0,0), 0, 0, 1, 1.5707963267948966))", &lk)
            .unwrap();

        // Assert
        assert!(matches!(v, Value::Scalar(x) if (x - 1.0).abs() < 1e-9));
    }

    #[test]
    fn vector_op_over_channels_preserves_rate_and_length() {
        // Arrange — vadd of two channel-x vectors → channel x.
        let lk = lookup(&[("a", vec![1.0, 2.0, 3.0], 100.0), ("b", vec![10.0, 20.0, 30.0], 100.0)]);

        // Act
        let v = eval_expr("vx(vadd(vec([a],0,0), vec([b],0,0)))", &lk).unwrap();

        // Assert
        match v {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![11.0, 22.0, 33.0].into());
                assert_eq!(c.sample_rate_hz, 100.0);
            }
            _ => panic!("expected channel"),
        }
    }

    #[test]
    fn frame_at_axle_centripetal_vertical_matches_hand_calc() {
        // Arrange — the worked example (design §"Worked example: frame-at-axle"),
        // reduced to a constant pitch-rate case so the result is hand-checkable.
        // IMU0 frame, +X rear, +Y right, +Z up; lever r = (-0.835, 0, -0.46) m.
        // Constant GyroY = 100 dps → ωy = 100·0.0174533 rad/s, GyroX=GyroZ=0.
        // alpha = differentiate(ω) ≈ 0 at interior samples, so the vertical
        // lever term is purely centripetal: vz(ω × (ω × r)).
        // ω = (0, ωy, 0); ω×r = (-0.46·ωy, 0, 0.835·ωy);
        // ω×(ω×r) = (0.835·ωy², 0, 0.46·ωy²) ⇒ vz = 0.46·ωy².
        // front_axle_vert = AccelZ_g + 0.46·ωy²/9.81.
        let n = 20;
        let gyro_y = vec![100.0; n]; // dps, constant
        let gyro_zero = vec![0.0; n];
        let accel_z = vec![1.0; n]; // g
        let lk = lookup(&[
            ("IMU0_GyroX", gyro_zero.clone(), 100.0),
            ("IMU0_GyroY", gyro_y, 100.0),
            ("IMU0_GyroZ", gyro_zero, 100.0),
            ("IMU0_AccelZ", accel_z, 100.0),
        ]);
        let expr = "[IMU0_AccelZ] + ( \
            vz(cross( \
                vec(differentiate([IMU0_GyroX]*0.0174533), \
                    differentiate([IMU0_GyroY]*0.0174533), \
                    differentiate([IMU0_GyroZ]*0.0174533)), \
                vec(-0.835, 0.0, -0.46))) \
            + vz(cross( \
                vec([IMU0_GyroX]*0.0174533, [IMU0_GyroY]*0.0174533, [IMU0_GyroZ]*0.0174533), \
                cross( \
                    vec([IMU0_GyroX]*0.0174533, [IMU0_GyroY]*0.0174533, [IMU0_GyroZ]*0.0174533), \
                    vec(-0.835, 0.0, -0.46)))) \
            ) / 9.81";

        // Act
        let v = eval_expr(expr, &lk).unwrap();

        // Assert — at an interior sample (alpha term = 0).
        let omega_y = 100.0 * 0.0174533_f64;
        let expected = 1.0 + (0.46 * omega_y * omega_y) / 9.81;
        match v {
            Value::Channel(c) => {
                assert_eq!(c.sample_rate_hz, 100.0);
                assert_relative_eq!(c.samples[10], expected, epsilon = 1e-6);
            }
            _ => panic!("expected channel"),
        }
    }

    #[test]
    fn cross_of_non_vector_arg_is_type_error() {
        // Arrange
        let lk = lookup(&[("a", vec![1.0], 1.0)]);

        // Act — a scalar where a vector is required.
        let e = eval_expr("cross([a], vec(0,0,1))", &lk).unwrap_err();

        // Assert
        assert_eq!(e.kind, crate::math::MathEvalErrorKind::Type);
    }

    #[test]
    fn variance_time_without_overlay_errors() {
        // Arrange — no overlay designated.
        let lk = lookup(&[("LapTime", vec![1.0, 2.0], 1.0)]);
        let ctx = laps_ctx(vec![(0.0, 2.0)]);

        // Act
        let err = eval(&crate::math::parse::parse("variance_time([LapTime])").unwrap(), &lk, &ctx)
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::math::MathEvalErrorKind::NoLapContext);
    }
}
