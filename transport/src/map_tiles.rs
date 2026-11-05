//! Fetching one satellite map tile over HTTPS (track editor spec §4). The
//! cache, the tile arithmetic and the source URLs are `idl_rs::map_tiles`;
//! this module is only the bytes on the wire.

use std::time::Duration;

use idl_rs::map_tiles::{TileId, TileSource};

use crate::error::{TransportError, TransportErrorKind};
use crate::firmware_catalog::https_client;

/// How long one tile may take. A map asks for a dozen at once, and a tile
/// that has not arrived by then is drawn blank rather than holding the rest.
const TILE_TIMEOUT: Duration = Duration::from_secs(15);

/// The largest tile accepted, bytes. Real satellite tiles run 10–60 kB.
const MAX_TILE_BYTES: usize = 2 * 1024 * 1024;

/// Downloads tile `t` from `source`. Any failure — offline, a timeout, a
/// non-success status, an empty or oversized body — is a
/// [`TransportErrorKind::Wifi`] error (the HTTP-transfer kind, as the
/// firmware catalog uses), and the caller draws the tile blank.
pub async fn fetch_tile(source: TileSource, t: TileId) -> Result<Vec<u8>, TransportError> {
    let url = source.url(t);
    let response = https_client()?
        .get(&url)
        .header(reqwest::header::USER_AGENT, concat!("idl1/", env!("CARGO_PKG_VERSION")))
        .timeout(TILE_TIMEOUT)
        .send()
        .await
        .map_err(|e| tile_error(t, format!("could not reach the tile server: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(tile_error(t, format!("the tile server returned status {}", status.as_u16())));
    }
    let bytes = response.bytes().await.map_err(|e| tile_error(t, format!("could not read the tile: {e}")))?;
    if bytes.is_empty() || bytes.len() > MAX_TILE_BYTES {
        return Err(tile_error(t, format!("the tile server sent {} bytes", bytes.len())));
    }
    Ok(bytes.to_vec())
}

fn tile_error(t: TileId, message: String) -> TransportError {
    TransportError::new(TransportErrorKind::Wifi, format!("map tile {}/{}/{}: {message}", t.z, t.x, t.y))
}
