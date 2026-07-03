//! FRB wrapper for `idl_rs` chart decimation. Tiles are decimated straight from
//! the retained `SessionHandle` (the engine owns the samples post-parse), so no
//! ingest/registry handoff exists.

use crate::frb_generated::RustOpaque;
pub use idl_rs::session::handle::SessionHandle;

/// Decimate one chart tile from the retained session handle. `tier` 0 = raw;
/// `tile_index` selects the 1024-bucket window. All-NaN for an absent channel.
#[flutter_rust_bridge::frb]
pub fn decimate_tile(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    tier: u32,
    tile_index: u32,
) -> Vec<f64> {
    handle.decimate_tile(&channel_id, tier, tile_index)
}
