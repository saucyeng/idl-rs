//! FRB wrappers over `idl_rs::scatter` — the Analyze scatter / G-G chart. Each
//! call is one `handle in → reduced result out` crossing: a decimated cloud or
//! a binned grid, never the raw per-sample arrays. The engine owns every bound
//! it returns, so no axis math round-trips across the boundary.

use flutter_rust_bridge::frb;

use crate::frb_generated::RustOpaque;

// Re-export the result types so the `mirror` attributes resolve and
// `frb_generated.rs` can refer to them as `crate::scatter::Scatter*`. The handle
// re-export matches the chart_decimation / tracks wrappers (a free-fn wrapper
// taking `RustOpaque<SessionHandle>` in its own module, per the opaque-type note).
pub use idl_rs::scatter::{ScatterDensity, ScatterPoints};
pub use idl_rs::session::handle::SessionHandle;

#[frb(mirror(ScatterPoints))]
pub struct _ScatterPoints {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub colors: Option<Vec<f64>>,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

#[frb(mirror(ScatterDensity))]
pub struct _ScatterDensity {
    pub bins: u32,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub counts: Vec<u32>,
}

/// Paired, finite, decimated `(x, y)` cloud over `[t0, t1]` capped at
/// `max_points`, with optional per-point colour and the pre-decimation extent.
#[frb]
pub fn scatter_points(
    handle: RustOpaque<SessionHandle>,
    x_channel: String,
    y_channel: String,
    color_channel: Option<String>,
    t0_secs: f64,
    t1_secs: f64,
    max_points: u32,
) -> ScatterPoints {
    idl_rs::scatter::scatter_points(
        &handle,
        &x_channel,
        &y_channel,
        color_channel.as_deref(),
        t0_secs,
        t1_secs,
        max_points,
    )
}

/// 2D count histogram of the `(x, y)` cloud over `[t0, t1]`; `equal_aspect`
/// squares the range for the G-G friction circle. Bounds come back in the result.
#[frb]
pub fn scatter_density(
    handle: RustOpaque<SessionHandle>,
    x_channel: String,
    y_channel: String,
    t0_secs: f64,
    t1_secs: f64,
    bins: u32,
    equal_aspect: bool,
) -> ScatterDensity {
    idl_rs::scatter::scatter_density(
        &handle,
        &x_channel,
        &y_channel,
        t0_secs,
        t1_secs,
        bins,
        equal_aspect,
    )
}
