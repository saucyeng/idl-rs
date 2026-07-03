//! idl-rs-bridge — flutter_rust_bridge shim over the pure `idl-rs` engine.
//!
//! Thin `#[frb]` wrappers delegate to `idl_rs::…`; the app sees only this
//! crate. The sole place flutter_rust_bridge is a dependency.

mod frb_generated;

// Crate-root re-export: frb_generated imports `crate::*` and resolves the
// opaque handle type through it. The per-module `pub use`s remain the
// canonical FRB surface (see the opaque-type-duplication note in session.rs).
pub use idl_rs::session::handle::SessionHandle;

// DSP wrappers (filters / fft / integration / clip_reconstruct / variance /
// calibration / rotation sync fns) were pruned 2026-06-11: the app consumes
// that math through the engine (eval_math_into_store, welch_channel), never
// via direct bridge calls — and a sync full-vector call would block the UI
// thread. `fft` survives for the mirrored Welch types only.
pub mod chart_decimation;
pub mod fft;
pub mod histogram;
pub mod laps;
pub mod math;
pub mod scatter;
pub mod session;
pub mod spectrogram;
pub mod table;
pub mod tracks;
