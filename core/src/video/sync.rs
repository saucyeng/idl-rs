//! Video↔session sync-offset estimation. See docs/IDL0_SPEC.md §33.3.
//!
//! Contract: `session_time_s = video_time_s + offset_s`. Auto-estimation is
//! coarse by design (GPMF UTC anchor, else container creation time);
//! sub-second precision is the user's manual nudge (design doc §5) and a
//! manual offset always wins. Rendering never re-estimates.

use serde::Serialize;

use crate::session::handle::SessionHandle;
use crate::video::gpmf::VideoTelemetry;
use crate::video::mp4box::Mp4Info;
use crate::video::{VideoError, VideoErrorKind};

/// How an offset estimate was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMethod {
    /// GPMF UTC anchor vs the session's GPS-anchored clock.
    Gpmf,
    /// Container `creation_time` vs the session start — coarse, camera
    /// clocks are routinely wrong; confidence is reported low.
    CreationTime,
}

/// An estimated sync offset.
#[derive(Debug, Clone, Serialize)]
pub struct SyncEstimate {
    /// `session_time_s = video_time_s + offset_s`, in seconds.
    pub offset_s: f64,
    /// 0.9 for `gpmf`, 0.3 for `creation_time`.
    pub confidence: f64,
    pub method: SyncMethod,
}

/// Confidence reported for a GPMF-anchored estimate.
const CONFIDENCE_GPMF: f64 = 0.9;
/// Confidence reported for a creation-time estimate.
const CONFIDENCE_CREATION_TIME: f64 = 0.3;

/// Estimate the video↔session offset. Prefers the GPMF UTC anchor
/// (`telemetry`), falling back to the container `creation_time` in `info`.
/// Errors: `Parse` when neither anchor exists; `NoOverlap` (message lists
/// both ranges in seconds) when the mapped video span misses the session.
pub fn estimate_sync(
    telemetry: Option<&VideoTelemetry>,
    info: &Mp4Info,
    handle: &SessionHandle,
) -> Result<SyncEstimate, VideoError> {
    let (offset_s, method, confidence) =
        match telemetry.and_then(|t| t.utc_anchor) {
            Some((t_video_s, epoch_ms)) => {
                let session_s = handle.epoch_ms_to_time_secs(&[epoch_ms as f64])[0];
                (session_s - t_video_s, SyncMethod::Gpmf, CONFIDENCE_GPMF)
            }
            None => match info.creation_time_utc_ms {
                Some(creation_ms) => {
                    let session_s = handle.epoch_ms_to_time_secs(&[creation_ms as f64])[0];
                    (
                        session_s,
                        SyncMethod::CreationTime,
                        CONFIDENCE_CREATION_TIME,
                    )
                }
                None => return Err(VideoError::parse(
                    "no sync anchor: video has neither GPMF UTC nor a container creation time; \
                     pass a manual --offset",
                )),
            },
        };

    // Overlap check: video span mapped onto session time vs session span.
    let session_len_s = handle.metadata().duration_ms as f64 / 1000.0;
    let video_start_s = offset_s;
    let video_end_s = offset_s + info.duration_s;
    if video_end_s < 0.0 || video_start_s > session_len_s {
        return Err(VideoError::new(
            VideoErrorKind::NoOverlap,
            format!(
                "video maps to session seconds [{video_start_s:.1}, {video_end_s:.1}] but the \
                 session spans [0.0, {session_len_s:.1}] — no overlap; check the sync source or \
                 pass a manual --offset"
            ),
        ));
    }

    Ok(SyncEstimate {
        offset_s,
        confidence,
        method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};
    use crate::video::gpmf::VideoTelemetry;

    /// Session starting at `start_utc_ms`, `len_s` seconds of 1 Hz data. No
    /// `GPS_EpochMs` channel → `epoch_ms_to_time_secs` uses the session
    /// origin fallback, which is exact for these tests.
    fn handle_with_origin(start_utc_ms: i64, len_s: usize) -> SessionHandle {
        SessionHandle::from_channels(
            SessionMetaInput {
                session_id: String::new(),
                device_id: String::new(),
                timestamp_utc_ms: start_utc_ms,
                config_checksum: String::new(),
            },
            vec![ChannelInput {
                channel_id: "Speed".into(),
                sample_rate_hz: 1.0,
                samples: vec![0.0; len_s + 1],
                sample_times_secs: None,
            }],
        )
    }

    fn info(duration_s: f64, creation_ms: Option<i64>) -> Mp4Info {
        Mp4Info {
            width: 1920,
            height: 1080,
            fps: 30.0,
            duration_s,
            creation_time_utc_ms: creation_ms,
            has_gpmd: false,
        }
    }

    #[test]
    fn estimate_sync_gpmf_utc_anchor_maps_offset_through_session_clock() {
        // Arrange — session t=0 ↔ 1_717_770_600_000 ms; camera says video
        // t=2.0 s ↔ 25 s into the session → offset 23.0.
        let h = handle_with_origin(1_717_770_600_000, 120);
        let telem = VideoTelemetry {
            utc_anchor: Some((2.0, 1_717_770_625_000)),
            fixes: vec![],
        };

        // Act
        let est = estimate_sync(Some(&telem), &info(60.0, None), &h).unwrap();

        // Assert
        assert!((est.offset_s - 23.0).abs() < 1e-6);
        assert_eq!(est.method, SyncMethod::Gpmf);
        assert!((est.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn estimate_sync_creation_time_fallback_reports_low_confidence() {
        // Arrange — creation 30 s after session start, no telemetry.
        let h = handle_with_origin(1_000_000, 120);
        let inf = info(60.0, Some(1_000_000 + 30_000));

        // Act
        let est = estimate_sync(None, &inf, &h).unwrap();

        // Assert
        assert!((est.offset_s - 30.0).abs() < 1e-6);
        assert_eq!(est.method, SyncMethod::CreationTime);
        assert!((est.confidence - 0.3).abs() < 1e-9);
    }

    #[test]
    fn estimate_sync_telemetry_without_anchor_falls_back_to_creation_time() {
        // Arrange — GPMF present but GPS-empty (indoor start).
        let h = handle_with_origin(1_000_000, 120);
        let telem = VideoTelemetry::default();

        // Act
        let est = estimate_sync(Some(&telem), &info(60.0, Some(1_010_000)), &h).unwrap();

        // Assert
        assert_eq!(est.method, SyncMethod::CreationTime);
        assert!((est.offset_s - 10.0).abs() < 1e-6);
    }

    #[test]
    fn estimate_sync_video_before_session_is_no_overlap_error_listing_ranges() {
        // Arrange — video ends 100 s before the session starts.
        let h = handle_with_origin(1_000_000, 120);
        let inf = info(60.0, Some(1_000_000 - 400_000));

        // Act
        let err = estimate_sync(None, &inf, &h).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NoOverlap);
        assert!(
            err.message.contains("[-400.0, -340.0]"),
            "video range: {}",
            err.message
        );
        assert!(
            err.message.contains("[0.0, 121.0]"),
            "session range: {}",
            err.message
        );
    }

    #[test]
    fn estimate_sync_no_anchor_and_no_creation_time_is_parse_error() {
        // Arrange
        let h = handle_with_origin(1_000_000, 120);

        // Act
        let err = estimate_sync(None, &info(60.0, None), &h).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::Parse);
        assert!(err.message.contains("--offset"));
    }
}
