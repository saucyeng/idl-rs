//! The map's satellite tiles on disk (track editor spec §4): which tiles
//! exist, where a cached one lives, and reading and writing it. The fetch
//! itself is `idl1-transport`'s; this module never touches the network.
//!
//! The cache lives under the OS cache directory the caller passes in, never
//! in the library: tiles are not the rider's data, are never synced, and can
//! be deleted at any time.

use std::fmt;
use std::path::{Path, PathBuf};

/// Deepest zoom a source serves (Esri World Imagery: 19).
pub const MAX_ZOOM: u8 = 19;

/// A basemap the map can draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSource {
    /// Esri World Imagery, satellite (as idl0 had).
    EsriWorldImagery,
}

impl TileSource {
    /// The source named on the wire (`"esri_world_imagery"`), or `None`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "esri_world_imagery" => Some(Self::EsriWorldImagery),
            _ => None,
        }
    }

    /// The wire name, also the cache's directory name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EsriWorldImagery => "esri_world_imagery",
        }
    }

    /// Where tile `t` is fetched from.
    pub fn url(self, t: TileId) -> String {
        match self {
            Self::EsriWorldImagery => format!(
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{}/{}/{}",
                t.z, t.y, t.x
            ),
        }
    }
}

/// One Web Mercator tile: zoom `z`, column `x`, row `y` (row 0 at the top).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// Why a tile request was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapTileError {
    /// Not a source this build knows.
    UnknownSource(String),
    /// `z` above [`MAX_ZOOM`], or `x`/`y` outside `0..2^z`.
    OutOfRange { z: u32, x: u32, y: u32 },
}

impl fmt::Display for MapTileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSource(name) => write!(f, "unknown map tile source {name:?}"),
            Self::OutOfRange { z, x, y } => write!(f, "tile {z}/{x}/{y} does not exist (zoom 0..={MAX_ZOOM}, x and y in 0..2^zoom)"),
        }
    }
}

impl std::error::Error for MapTileError {}

impl TileId {
    /// A tile that exists at its zoom, or [`MapTileError::OutOfRange`].
    pub fn new(z: u32, x: u32, y: u32) -> Result<Self, MapTileError> {
        if z > MAX_ZOOM as u32 {
            return Err(MapTileError::OutOfRange { z, x, y });
        }
        let side = 1u64 << z;
        if x as u64 >= side || y as u64 >= side {
            return Err(MapTileError::OutOfRange { z, x, y });
        }
        Ok(Self { z: z as u8, x, y })
    }
}

/// `<cache_root>/map-tiles/<source>/<z>/<x>/<y>` — the cached tile's path.
/// No extension: the bytes are whatever image the source served.
pub fn tile_path(cache_root: &Path, source: TileSource, t: TileId) -> PathBuf {
    cache_root.join("map-tiles").join(source.as_str()).join(t.z.to_string()).join(t.x.to_string()).join(t.y.to_string())
}

/// The cached bytes of `t`, or `None` when it is not cached (or unreadable,
/// which the caller treats the same way: fetch it again).
pub fn read_cached(cache_root: &Path, source: TileSource, t: TileId) -> Option<Vec<u8>> {
    std::fs::read(tile_path(cache_root, source, t)).ok().filter(|b| !b.is_empty())
}

/// Stores `bytes` as `t`, written beside its final name and renamed into
/// place so a reader never sees half a tile. A failure is returned for the
/// caller to ignore: the tile was fetched, and only the cache misses out.
pub fn write_cached(cache_root: &Path, source: TileSource, t: TileId, bytes: &[u8]) -> std::io::Result<()> {
    let path = tile_path(cache_root, source, t);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("part-{}", uuid::Uuid::new_v4()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_id_new_coordinates_outside_the_zoom_level_are_refused() {
        // Act
        let ok = TileId::new(3, 7, 7);
        let x_out = TileId::new(3, 8, 0);
        let z_out = TileId::new(20, 0, 0);

        // Assert
        assert_eq!(ok, Ok(TileId { z: 3, x: 7, y: 7 }));
        assert!(matches!(x_out, Err(MapTileError::OutOfRange { .. })));
        assert!(matches!(z_out, Err(MapTileError::OutOfRange { .. })));
    }

    #[test]
    fn tile_source_parse_only_the_known_name() {
        // Act
        let esri = TileSource::parse("esri_world_imagery");
        let other = TileSource::parse("google");

        // Assert
        assert_eq!(esri, Some(TileSource::EsriWorldImagery));
        assert_eq!(other, None);
    }

    #[test]
    fn tile_source_url_esri_orders_zoom_row_column() {
        // Act
        let url = TileSource::EsriWorldImagery.url(TileId { z: 15, x: 9000, y: 12000 });

        // Assert
        assert!(url.ends_with("/World_Imagery/MapServer/tile/15/12000/9000"));
    }

    #[test]
    fn write_cached_then_read_cached_round_trips_and_a_missing_tile_is_none() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-rs-tiles-{}", uuid::Uuid::new_v4()));
        let t = TileId { z: 2, x: 1, y: 3 };

        // Act
        let before = read_cached(&root, TileSource::EsriWorldImagery, t);
        write_cached(&root, TileSource::EsriWorldImagery, t, b"jpeg").unwrap();
        let after = read_cached(&root, TileSource::EsriWorldImagery, t);

        // Assert
        assert_eq!(before, None);
        assert_eq!(after.as_deref(), Some(&b"jpeg"[..]));
        assert_eq!(std::fs::read_dir(tile_path(&root, TileSource::EsriWorldImagery, t).parent().unwrap()).unwrap().count(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }
}
