//! Chart tile command (C3 §3.5): `fetch_tile` over L3's
//! `idl_rs::tile::build_tile_bytes` v2 encoder (ledger R25, R43).
//!
//! Same idiom as `commands/rasters.rs`: the `#[tauri::command]` is a
//! one-line wrapper over a `_via`-suffixed plain function taking
//! `data_dir: &Path` — this module's own tests exercise `fetch_tile_via`
//! directly (`tauri::State`/`tauri::ipc::Response` cannot be constructed
//! outside a running app).

use std::path::Path;

use idl_rs::chart_decimation::MAX_TIER;
use idl_rs::tile::build_tile_bytes;

use crate::error::{IpcError, IpcErrorKind};
use crate::session_cache::SessionCache;
use crate::session_source::session_dir;
use crate::state::DataDir;

/// Largest `column_count` a caller may request (C3 §3.5, ruling R43).
const MAX_COLUMN_COUNT: u32 = 4096;

/// Transport-agnostic core of `fetch_tile`. Every validation happens
/// before any bytes are built (C3 §1's binary-transport rule), in the
/// order C3 §3.5/ruling R43 specify: `tier > MAX_TIER` → `invalid_argument`
/// (ledger's L3 Task 10 tracked note — this check is load-bearing, not
/// defensive: without it a bad tier reaches the encoder and yields a
/// plausible-looking all-NaN tile rather than an error); `column_count`
/// outside `1..=4096` → `invalid_argument`; unknown `session_id`/`channel`
/// → `not_found`. Only then does it call
/// [`idl_rs::tile::build_tile_bytes`], which owns decimation, column
/// stats and the header — this function re-implements none of it.
pub fn fetch_tile_via(
    cache: &SessionCache,
    data_dir: &Path,
    session_id: &str,
    channel: &str,
    tier: u32,
    tile_index: u32,
    column_count: u32,
) -> Result<Vec<u8>, IpcError> {
    if tier > MAX_TIER {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("tier {tier} exceeds the engine's configured range (max_tier = {MAX_TIER})"),
            serde_json::json!({ "tier": tier, "max_tier": MAX_TIER }),
        ));
    }
    if column_count == 0 || column_count > MAX_COLUMN_COUNT {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("column_count must be in 1..={MAX_COLUMN_COUNT}, got {column_count}"),
            serde_json::json!({ "column_count": column_count }),
        ));
    }

    // One channel's column, not the whole `data.parquet` (ruling R203.1),
    // and decoded at most once while the cache holds it (R203.2) — this is
    // the pan/zoom path, called dozens of times a minute.
    let ch = cache.channel(&session_dir(data_dir, session_id), session_id, channel)?;

    let samples = ch.materialize();
    Ok(build_tile_bytes(&samples, &ch.t_us, tier, tile_index, column_count))
}

/// Fetches and encodes one chart tile (C3 §3.5, v2 layout). `column_count`
/// is the caller's own chart width in pixel columns (ruling R43).
#[tauri::command(async)]
pub fn fetch_tile(
    session_id: String,
    channel: String,
    tier: u32,
    tile_index: u32,
    column_count: u32,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<tauri::ipc::Response, IpcError> {
    let bytes = fetch_tile_via(&cache, &data_dir.0, &session_id, &channel, tier, tile_index, column_count)?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-tiles-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds a session `"s1"` with one fixed-rate channel `"Speed"` of 2000
    /// samples at 1 kHz (`t_us` known: sample `i` at `i * 1000` µs).
    fn seed_session(root: &Path) {
        let n = 2000usize;
        let samples: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Speed".to_string(),
                t_us: (0..n).map(|i| i as i64 * 1000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1000.0,
                column: RawColumn::F64(samples),
                source_kind: "wheel".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn fetch_tile_tier0_tile0_256_columns_header_fields_and_total_length_match_c3_3_5_v2_formula() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_tile_via(&SessionCache::new(), &root, "s1", "Speed", 0, 0, 256).unwrap();

        // Assert — C3 §3.5 v2: 32 + sample_count*8 + column_count*12 + column_count*8.
        assert_eq!(&bytes[0..4], b"IDLT");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
        let sample_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(sample_count, 1024);
        let column_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(column_count, 256);
        assert_eq!(bytes.len(), 32 + 1024 * 8 + 256 * 12 + 256 * 8);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_tile_column_time_region_column_0_carries_first_samples_recorded_t_us() {
        // Arrange — C3 §3.5 spec 625-632: column 0's t_us is the recorded
        // time of the first sample in its bucket range, exact, no
        // interpolation. Tile 0's first bucket starts at sample index 0,
        // whose t_us is 0 by seed_session's construction.
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_tile_via(&SessionCache::new(), &root, "s1", "Speed", 0, 0, 256).unwrap();

        // Assert
        let sample_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let column_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let time_off = 32 + sample_count * 8 + column_count * 12;
        let column0_t_us = i64::from_le_bytes(bytes[time_off..time_off + 8].try_into().unwrap());
        assert_eq!(column0_t_us, 0);
        assert_ne!(column0_t_us, i64::MIN, "column 0 has real samples in its range, not the empty sentinel");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_tile_tier_above_max_tier_invalid_argument_before_any_bytes() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_tile_via(&SessionCache::new(), &root, "s1", "Speed", MAX_TIER + 1, 0, 256).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        let detail = err.detail.unwrap();
        assert_eq!(detail["tier"], MAX_TIER + 1);
        assert_eq!(detail["max_tier"], MAX_TIER);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_tile_unknown_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_tile_via(&SessionCache::new(), &root, "s1", "NopeChannel", 0, 0, 256).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_tile_unknown_session_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = fetch_tile_via(&SessionCache::new(), &root, "nope", "Speed", 0, 0, 256).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_tile_column_count_zero_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_tile_via(&SessionCache::new(), &root, "s1", "Speed", 0, 0, 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        let detail = err.detail.unwrap();
        assert_eq!(detail["column_count"], 0);

        let _ = std::fs::remove_dir_all(&root);
    }
}
