//! FRB wrappers for `idl_rs::math` — the math-channel evaluator. Dart passes a
//! retained `RustOpaque<SessionHandle>`, the expression, and an FFI-friendly
//! lap context; Rust evaluates and returns an output buffer + rate (or a typed
//! failure). The overlay session crosses as a second handle, never as samples.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use flutter_rust_bridge::frb;

use crate::frb_generated::RustOpaque;
use idl_rs::math::channel_def::MathChannelDef;
use idl_rs::math::eval::{ChannelLookup, LookupChannel, MathLapContext, MathOverlay};
use idl_rs::math::resolve::resolve_dependencies;
use idl_rs::math::{evaluate, MathEvalError, MathEvalErrorKind};
pub use idl_rs::session::handle::SessionHandle;

// ---- Error crossing (freezed-free: unit-enum kind + message) ----

/// Discriminant for [`MathEvalFailure`]. Unit enum → plain Dart enum.
pub enum MathEvalFailureKind {
    Parse,
    UnknownFunction,
    UnknownChannel,
    ArgCount,
    Type,
    DivisionByZero,
    NoLapContext,
    NotImplemented,
    Runtime,
}

/// Error returned by [`eval_math`].
pub struct MathEvalFailure {
    pub kind: MathEvalFailureKind,
    pub message: String,
}

impl From<MathEvalError> for MathEvalFailure {
    fn from(e: MathEvalError) -> Self {
        let kind = match e.kind {
            MathEvalErrorKind::Parse => MathEvalFailureKind::Parse,
            MathEvalErrorKind::UnknownFunction => MathEvalFailureKind::UnknownFunction,
            MathEvalErrorKind::UnknownChannel => MathEvalFailureKind::UnknownChannel,
            MathEvalErrorKind::ArgCount => MathEvalFailureKind::ArgCount,
            MathEvalErrorKind::Type => MathEvalFailureKind::Type,
            MathEvalErrorKind::DivisionByZero => MathEvalFailureKind::DivisionByZero,
            MathEvalErrorKind::NoLapContext => MathEvalFailureKind::NoLapContext,
            MathEvalErrorKind::NotImplemented => MathEvalFailureKind::NotImplemented,
            MathEvalErrorKind::Runtime => MathEvalFailureKind::Runtime,
        };
        MathEvalFailure { kind, message: e.message }
    }
}

// ---- Eval output (metadata only) ----

/// Metadata of a stored eval result — the samples stay in the handle's math
/// store; Dart reads them only as decimated tiles / bounded views.
pub struct EvalStoredMeta {
    /// Number of samples in the stored result.
    pub length: u32,
    /// Output sample rate in Hz (0.0 = scalar-as-channel).
    pub sample_rate_hz: f64,
}

// ---- FFI-friendly lap context argument ----

/// Overlay designation for `variance_*`. The overlay session crosses as a
/// second handle; its lap window crosses as scalars.
pub struct MathOverlayArg {
    pub handle: RustOpaque<SessionHandle>,
    pub lap_start_ms: f64,
    pub lap_end_ms: f64,
    pub lap_start_uniform_sec: f64,
}

/// FFI-friendly mirror of the core `MathLapContext`. Lap/sector bounds cross as
/// parallel arrays (FRB cannot bridge `Vec<(f64,f64)>`).
pub struct MathLapCtxArg {
    pub main_lap_starts: Vec<f64>,
    pub main_lap_ends: Vec<f64>,
    pub main_sector_starts: Vec<f64>,
    pub main_sector_ends: Vec<f64>,
    pub main_lap_number: Option<u32>,
    pub overlay: Option<MathOverlayArg>,
}

fn zip_bounds(starts: Vec<f64>, ends: Vec<f64>) -> Vec<(f64, f64)> {
    starts.into_iter().zip(ends).collect()
}

/// Shares the retained overlay handle with the evaluator without deep-cloning
/// the session — cloning a `RustOpaque` is an Arc refcount bump. Replaces the
/// correctness-first `(*handle).clone()`, which copied every column plus the
/// math store on every eval/resolve call (twice per overlay evaluation).
struct SharedHandleLookup(RustOpaque<SessionHandle>);

impl ChannelLookup for SharedHandleLookup {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        self.0.lookup(name)
    }
    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        self.0.best_time_base_dims()
    }
}

impl MathLapCtxArg {
    fn into_core(self) -> MathLapContext {
        let overlay = self.overlay.map(|o| MathOverlay {
            lookup: Arc::new(SharedHandleLookup(o.handle)) as Arc<dyn ChannelLookup + Send + Sync>,
            lap_start_ms: o.lap_start_ms,
            lap_end_ms: o.lap_end_ms,
            lap_start_uniform_sec: o.lap_start_uniform_sec,
        });
        MathLapContext {
            main_lap_bounds: zip_bounds(self.main_lap_starts, self.main_lap_ends),
            main_sectors: zip_bounds(self.main_sector_starts, self.main_sector_ends),
            main_lap_number: self.main_lap_number,
            overlay,
            // Channel-math context has no table Main row; main({col[]}) is a
            // table-cell-only reference (see idl-rs table::evaluate_table_multi).
            baseline_row: None,
        }
    }
}

// ---- Entry points ----

/// Evaluate `expression` against the retained session handle + lap context and
/// upsert the result into the handle's math store under `store_as`. Only
/// metadata crosses FFI — at 100M samples the old full-vector return cost
/// ~800 MB serialized each way per eval (spec §15 seam).
#[frb]
pub fn eval_math_into_store(
    handle: RustOpaque<SessionHandle>,
    expression: String,
    store_as: String,
    lap_ctx: MathLapCtxArg,
) -> Result<EvalStoredMeta, MathEvalFailure> {
    let core_ctx = lap_ctx.into_core();
    let out = evaluate(&expression, &*handle, &core_ctx).map_err(MathEvalFailure::from)?;
    let meta =
        EvalStoredMeta { length: out.samples.len() as u32, sample_rate_hz: out.sample_rate_hz };
    handle.store_math(&store_as, out.sample_rate_hz, out.samples);
    Ok(meta)
}
/// FFI-friendly math-channel definition — the engine needs only name +
/// expression (the output rate is derived at eval time). See spec §8.1.
pub struct MathChannelDefArg {
    pub name: String,
    pub expression: String,
}

/// Resolve the transitive math-channel dependencies of `target_expression` into
/// `handle`'s math store, so the caller's subsequent `eval_math` reads them
/// Rust-side. `defs` is the full math-channel set; `target_name` seeds the cycle
/// guard. Replaces the former Dart `_resolveDependenciesIntoHandle`.
#[frb]
pub fn resolve_math_dependencies(
    handle: RustOpaque<SessionHandle>,
    target_name: String,
    target_expression: String,
    defs: Vec<MathChannelDefArg>,
    lap_ctx: MathLapCtxArg,
) {
    let core_ctx = lap_ctx.into_core();
    let defs_map: HashMap<String, MathChannelDef> = defs
        .into_iter()
        .map(|d| (d.name.clone(), MathChannelDef { name: d.name, expression: d.expression }))
        .collect();
    let mut visited = HashSet::from([target_name]);
    resolve_dependencies(&handle, &target_expression, &defs_map, &core_ctx, &mut visited);
}
