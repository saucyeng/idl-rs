//! `fetch_seams` (C3 §3.5, ruling R237): burst-seam boundary spans for one
//! channel, so the chart can hatch a seam the way it already hatches a
//! [`idl_rs::session::GapSpan`] gap — a second tone, since a seam is
//! corrected data, not missing data.
//!
//! **A small JSON sibling command, not a `fetch_tile` field (ruling R237's
//! "choose the cheaper one" — chosen here).** A channel's seam spans are the
//! same set regardless of tier or tile index — they are a property of the
//! whole channel's burst structure, C1 §3.3, not of any one tile's window —
//! so folding them into every `fetch_tile` response would resend the same
//! handful of `(i64, i64)` pairs on every tier/tile/pan/zoom request instead
//! of once per channel. `fetch_raster_meta`/`fetch_gps_trace_meta` already
//! establish this split (a binary per-window/per-tile payload beside a small
//! JSON one the app fetches once and caches) — this is that same idiom.
//!
//! Same `_via`-suffixed idiom as `commands/tiles.rs`: this module's tests
//! exercise `fetch_seams_via` directly.

use std::path::Path;

use idl_rs::parse::records::imu_period_us;
use idl_rs::session::seam_correction::seam_spans;

use crate::error::IpcError;
use crate::session_cache::SessionCache;
use crate::session_source::session_dir;
use crate::state::DataDir;

/// `fetch_seams`'s return: one `(start_us, end_us)` pair per burst-seam
/// boundary, on the channel's own corrected time axis (same axis
/// `fetch_tile`'s column time region reports). Empty for a channel with no
/// burst correction applied (C1 §3.3: every non-burst source, or a burst
/// source whose bursts were already exactly nominal-spaced).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SeamsResponse {
    /// `t0_us`/`t1_us` pairs, in ascending order, one per seam.
    pub spans: Vec<(i64, i64)>,
}

/// Transport-agnostic core of `fetch_seams`. `not_found` for an unknown
/// `session_id`/`channel` (same as `fetch_tile_via`); a channel that was
/// never burst-corrected (`t_recorded_us: None`, C1 §3.2) returns an empty
/// `spans` list rather than an error — "no seams" is a true answer, not a
/// failure.
pub fn fetch_seams_via(
    cache: &SessionCache,
    data_dir: &Path,
    session_id: &str,
    channel: &str,
) -> Result<SeamsResponse, IpcError> {
    let ch = cache.channel(&session_dir(data_dir, session_id), session_id, channel)?;

    let Some(recorded_us) = ch.t_recorded_us.as_ref() else {
        return Ok(SeamsResponse { spans: Vec::new() });
    };

    let nominal_period_us = imu_period_us(ch.nominal_rate_hz.round() as u16);
    let spans = seam_spans(recorded_us, &ch.t_us, nominal_period_us);
    Ok(SeamsResponse { spans })
}

/// Fetches one channel's burst-seam boundary spans (C3 §3.5, ruling R237).
/// Cheap JSON, served from the session cache like `fetch_gps_trace_meta` —
/// the app fetches this once per channel and caches it, never per tile.
#[tauri::command(async)]
pub fn fetch_seams(
    session_id: String,
    channel: String,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<SeamsResponse, IpcError> {
    fetch_seams_via(&cache, &data_dir.0, &session_id, &channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::IpcErrorKind;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-seams-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds session `"s1"` with one IMU channel carrying the C1 §3.3
    /// worked example's recorded/corrected stamps (4 bursts of N=4, one
    /// seam every 4 samples), and one wheel-speed channel with no
    /// `t_recorded_us` at all (never burst-corrected).
    fn seed_session(root: &Path) {
        let recorded = vec![
            96250, 97500, 98750, 100000, //
            101050, 102300, 103550, 104800, //
            105850, 107100, 108350, 109600, //
            110650, 111900, 113150, 114400,
        ];
        let corrected =
            idl_rs::session::seam_correction::correct_burst_seams(&recorded, 1250).corrected_us;

        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: "a".repeat(64),
            channels: vec![
                Channel {
                    channel_id: "IMU0_AccelZ".to_string(),
                    t_us: corrected.clone(),
                    t_recorded_us: Some(recorded),
                    nominal_rate_hz: 800.0,
                    column: RawColumn::F64(vec![0.0; corrected.len()]),
                    source_kind: "imu0".to_string(),
                    unit: "g".to_string(),
                    gaps: Vec::new(),
                },
                Channel {
                    channel_id: "WheelFront".to_string(),
                    t_us: (0..10).map(|i| i as i64 * 1000).collect(),
                    t_recorded_us: None,
                    nominal_rate_hz: 1000.0,
                    column: RawColumn::F64(vec![0.0; 10]),
                    source_kind: "wheel".to_string(),
                    unit: "km/h".to_string(),
                    gaps: Vec::new(),
                },
            ],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn fetch_seams_via_a_burst_corrected_channel_reports_one_span_per_seam() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let result = fetch_seams_via(&SessionCache::new(), &root, "s1", "IMU0_AccelZ").unwrap();

        // Assert — matches the C1 §3.3 worked example's three seams.
        assert_eq!(result.spans, vec![(100000, 101200), (104800, 106000), (109600, 110800)]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_seams_via_a_never_corrected_channel_returns_empty_spans_not_an_error() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let result = fetch_seams_via(&SessionCache::new(), &root, "s1", "WheelFront").unwrap();

        // Assert
        assert!(result.spans.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_seams_via_unknown_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_seams_via(&SessionCache::new(), &root, "s1", "NopeChannel").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_seams_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = fetch_seams_via(&SessionCache::new(), &root, "nope", "IMU0_AccelZ").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }
}
