//! Channel-vs-channel scatter command (C3 §3.5, ruling R215 item 3):
//! `fetch_scatter` over L3's [`idl_rs::scatter::scatter_points`] pairing and
//! decimation, encoded by [`idl_rs::scatter_wire::encode_scatter_idls`].
//!
//! **Binary, unlike `fetch_histogram`.** A G-G cloud is up to `point_budget`
//! points — thousands, each two `f64`s — so this belongs with `fetch_tile`
//! and `fetch_fft` on C3 §1's binary side, not with the histogram's few
//! hundred JSON numbers. The engine owns the pairing, the finite filtering,
//! the extent and the decimation; the sandbox draws and nothing else (design
//! §4, CLAUDE.md §2).
//!
//! Same `_via`-suffixed idiom as `commands/tiles.rs`/`commands/rasters.rs`:
//! this module's tests exercise `fetch_scatter_via` directly, because
//! `tauri::State`/`tauri::ipc::Response` cannot be constructed outside a
//! running app.

use std::path::Path;

use idl_rs::scatter::scatter_points;
use idl_rs::scatter_wire::encode_scatter_idls;

use crate::error::{IpcError, IpcErrorKind};
use crate::session_cache::SessionCache;
use crate::session_source::{load_lazy_session_handle, resolve_window, WindowDto};
use crate::state::DataDir;

/// Largest `point_budget` a caller may request (C3 §3.5, ruling R215 item 3).
/// A decimated cloud is drawn one SVG circle per point; past a few tens of
/// thousands the renderer, not the transport, is the limit — 65 536 also
/// bounds one response at 1 MiB, the same order as a wide `fetch_tile`.
pub const MAX_SCATTER_POINTS: u32 = 65_536;

/// Transport-agnostic core of `fetch_scatter` (C3 §3.5, ruling R215 item 3).
///
/// Resolution order matches `fetch_histogram`/`fetch_fft_v2` (rulings R85,
/// R123): the window resolves first — an unknown `lap`, or a `range` failing
/// R119/R120, is `invalid_argument` before any sample is read — then both
/// channels are sliced to it and paired. A scatter consumes the time axis (it
/// plots one channel against another, not against `t`), so the window is
/// correctly this function's slicing domain.
///
/// Both channels are read through the byte-budgeted [`SessionCache`], one
/// column at a time (rulings R203.1, R211), via the lazy session handle
/// `scatter_points` pairs from.
///
/// **Both channel ids are validated before any pairing.** `scatter_points`
/// slices by name and an absent channel silently yields an empty vector,
/// which would pair to an empty cloud — indistinguishable from "these two
/// channels genuinely never overlap". A typo must say so.
///
/// `not_found`: unknown `session_id`, `x_channel` or `y_channel`.
/// `invalid_argument`: an unresolvable window span, or a `point_budget`
/// outside `1..=MAX_SCATTER_POINTS`.
pub fn fetch_scatter_via(
    cache: &SessionCache,
    data_dir: &Path,
    window: &WindowDto,
    x_channel: &str,
    y_channel: &str,
    point_budget: u32,
) -> Result<Vec<u8>, IpcError> {
    if point_budget == 0 || point_budget > MAX_SCATTER_POINTS {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("point_budget must be in 1..={MAX_SCATTER_POINTS}, got {point_budget}"),
            serde_json::json!({ "point_budget": point_budget, "max_points": MAX_SCATTER_POINTS }),
        ));
    }

    let (handle, source) = load_lazy_session_handle(data_dir, &window.session_id, cache)?;
    for id in [x_channel, y_channel] {
        if handle.channel_meta(id).is_none() {
            return Err(IpcError::new(
                IpcErrorKind::NotFound,
                format!("channel '{id}' not found in session '{}'", window.session_id),
            ));
        }
    }

    let (t0_secs, t1_secs) = resolve_window(data_dir, window)?;
    let points = scatter_points(&handle, x_channel, y_channel, None, t0_secs, t1_secs, point_budget);

    // A memory refusal while decoding a column reads as "no such channel" to
    // `ChannelSource` (it has nowhere to put an error), so an empty cloud
    // could otherwise be reported as a legitimate result. Surfacing the
    // stashed error turns that back into `resource_exhausted`.
    if let Some(err) = source.take_first_error() {
        return Err(err);
    }

    Ok(encode_scatter_idls(
        &points.xs,
        &points.ys,
        points.x_min,
        points.x_max,
        points.y_min,
        points.y_max,
    ))
}

/// Pairs two channels over `window` into a decimated XY cloud as `IDLS` v1
/// bytes (C3 §3.5, ruling R215 item 3). `point_budget` is the caller's own
/// cap on how many points it will draw. Settle-bound only (C3 §4): never a
/// hover/pan/zoom handler.
#[tauri::command(async)]
pub fn fetch_scatter(
    window: WindowDto,
    x_channel: String,
    y_channel: String,
    point_budget: u32,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<tauri::ipc::Response, IpcError> {
    let bytes = fetch_scatter_via(&cache, &data_dir.0, &window, &x_channel, &y_channel, point_budget)?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::session_source::SpanDto;
    use idl_rs::scatter_wire::HEADER_LEN;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-scatter-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds session `"s1"` with two 1 kHz channels of 1000 samples each:
    /// `"AccelX"` = `i`, `"AccelY"` = `-i`. Sample `i` is at `i * 1000` µs.
    fn seed_session(root: &Path) {
        let n = 1000usize;
        let mk = |id: &str, f: fn(usize) -> f64| Channel {
            channel_id: id.to_string(),
            t_us: (0..n).map(|i| i as i64 * 1000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1000.0,
            column: RawColumn::F64((0..n).map(f).collect()),
            source_kind: "imu".to_string(),
            unit: "g".to_string(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![mk("AccelX", |i| i as f64), mk("AccelY", |i| -(i as f64))],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn session_window() -> WindowDto {
        WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() }
    }

    fn point_count(bytes: &[u8]) -> usize {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    }

    fn bounds(bytes: &[u8]) -> (f64, f64, f64, f64) {
        (
            f64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            f64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            f64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            f64::from_le_bytes(bytes[40..48].try_into().unwrap()),
        )
    }

    #[test]
    fn fetch_scatter_via_whole_session_under_budget_returns_every_paired_point() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", 4096).unwrap();

        // Assert — 1000 pairs, header + 1000 * 16 bytes.
        assert_eq!(&bytes[0..4], b"IDLS");
        assert_eq!(point_count(&bytes), 1000);
        assert_eq!(bytes.len(), HEADER_LEN + 1000 * 16);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_over_budget_decimates_in_the_engine_not_the_caller() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", 100).unwrap();

        // Assert
        assert!(point_count(&bytes) <= 100, "got {} points for a budget of 100", point_count(&bytes));
        assert!(point_count(&bytes) > 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_bounds_are_the_pre_decimation_extent_not_the_thinned_cloud() {
        // Arrange — decimation drops the last samples under most strides,
        // so a post-decimation extent would be visibly narrower.
        let root = temp_root();
        seed_session(&root);

        // Act
        let full = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", 4096).unwrap();
        let thin = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", 7).unwrap();

        // Assert
        assert_eq!(bounds(&full), bounds(&thin));
        assert_eq!(bounds(&full), (0.0, 999.0, -999.0, 0.0));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_range_window_pairs_only_that_window() {
        // Arrange — the first 100 ms of a 1 kHz pair is 100 samples.
        let root = temp_root();
        seed_session(&root);
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 0, t1_us: 100_000 },
            colour: "--chart-1".to_string(),
        };

        // Act
        let bytes = fetch_scatter_via(&SessionCache::new(), &root, &window, "AccelX", "AccelY", 4096).unwrap();

        // Assert
        assert!(point_count(&bytes) < 1000);
        let (x_min, x_max, _, _) = bounds(&bytes);
        assert_eq!(x_min, 0.0);
        assert!(x_max < 999.0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_same_channel_on_both_axes_is_the_identity_diagonal() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelX", 4096).unwrap();

        // Assert — every point on y = x, so both extents match.
        let (x_min, x_max, y_min, y_max) = bounds(&bytes);
        assert_eq!((x_min, x_max), (y_min, y_max));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_unknown_x_channel_not_found_rather_than_an_empty_cloud() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "Nope", "AccelY", 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
        assert!(err.message.contains("Nope"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_unknown_y_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "Nope", 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();
        let window = WindowDto { session_id: "nope".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() };

        // Act
        let err = fetch_scatter_via(&SessionCache::new(), &root, &window, "AccelX", "AccelY", 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_zero_budget_invalid_argument_before_any_read() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["point_budget"], 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_over_cap_budget_invalid_argument_naming_the_cap() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err =
            fetch_scatter_via(&SessionCache::new(), &root, &session_window(), "AccelX", "AccelY", MAX_SCATTER_POINTS + 1).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["max_points"], MAX_SCATTER_POINTS);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_scatter_via_range_wholly_outside_the_session_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 10_000_000, t1_us: 11_000_000 },
            colour: "--chart-1".to_string(),
        };

        // Act
        let err = fetch_scatter_via(&SessionCache::new(), &root, &window, "AccelX", "AccelY", 4096).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }
}
