//! FRB external-type mirror for `idl_rs::histogram`.
//!
//! The `channel_histogram` wrapper itself lives in `session.rs` (beside the
//! other `SessionHandle` accessors) so it shares the canonical opaque-handle
//! type — see the opaque-type-duplication note in `session.rs`.

pub use idl_rs::histogram::HistogramResult;

/// Mirror of [`idl_rs::histogram::HistogramResult`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(HistogramResult))]
pub struct _HistogramResult {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<u32>,
    pub total: u32,
}
