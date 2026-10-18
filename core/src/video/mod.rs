//! Linked video (video spec `2026-10-16-idl1-video-DRAFT.md`, step 1):
//! ISO-BMFF walking, GPMF telemetry parsing and sync-offset estimation,
//! recovered from idl0 (idl-rs tag `idl0-video-export`, IDL0_SPEC §33.3)
//! without its painters (design D9 still holds for those). Pure: `std::fs`
//! and `std::io` only. The `session.json` side of linking lives in
//! [`crate::store::video`].
//!
//! [`VideoLink`] is one C1 §6 `videos` entry, serialised as is across IPC
//! (C3 §3.12).

use std::fmt;

use serde::{Deserialize, Serialize};

pub mod gpmf;
pub mod mp4box;
pub mod sync;

/// How a [`VideoLink`]'s `offset_s` was found (C1 §6 `sync_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMethod {
    /// The GoPro's GPMF GPS time (`GPSU`) against the session's GPS clock.
    Gpmf,
    /// The container's `mvhd` creation time against the session's clock.
    /// Coarse: camera clocks are often local time, and drift.
    CreationTime,
    /// Set by hand, or linked with no time anchor at all. Never replaced
    /// automatically (video spec §4.3).
    Manual,
}

/// One video linked to a session: a C1 §6 `videos` entry. The entry's
/// identity is `blob_sha256`; sync merges entries by it (C4 §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoLink {
    /// The MP4's blob (C4 §2), 64 lowercase hex characters.
    pub blob_sha256: String,
    /// The MP4's size, bytes.
    pub size_bytes: u64,
    /// The name the file was linked from. Display only.
    pub file_name: String,
    /// `"h264"`, `"hevc"`, or the video track's `stsd` fourcc as text.
    pub video_codec: String,
    /// Coded width, px.
    pub width_px: u32,
    /// Coded height, px.
    pub height_px: u32,
    /// Nominal frame rate, frames/s.
    pub fps: f64,
    /// Presentation duration, s.
    pub duration_s: f64,
    /// Seconds: `session_time_s = video_time_s + offset_s`, on the session's
    /// `*_time_secs` clock (t = 0 at the session's first sample).
    pub offset_s: f64,
    /// How `offset_s` was found.
    pub sync_method: SyncMethod,
    /// A tombstone (C4 §6): the entry was unlinked. Omitted when `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub unlinked: bool,
    /// When this entry last changed, UTC ms since the Unix epoch. The sync
    /// merge's tiebreak (C4 §6).
    pub updated_at_ms: i64,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// A live [`VideoLink`] as `get_session` reports it (C3 §3.12): the entry
/// plus whether its blob is on this device.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionVideo {
    /// The `session.json` entry, its fields inlined.
    #[serde(flatten)]
    pub link: VideoLink,
    /// `true` when `blobs/sha256/…` holds the MP4 here. A missing blob is a
    /// normal state (C1 §6): deleted on this device, or never pulled.
    pub on_device: bool,
}

/// What went wrong in the video subsystem. C3 §3.12 maps these to
/// `video_parse`, `video_no_overlap`, `video_ambiguous`, `not_found` and
/// `io`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoErrorKind {
    /// A filesystem read or write failed.
    Io,
    /// The container or its telemetry did not parse.
    Parse,
    /// The container has no GPMF (`gpmd`) track. Internal to estimation,
    /// which falls back to the creation time.
    NoGpmf,
    /// The video's time span misses the session's.
    NoOverlap,
    /// More than one session overlaps the video (inbox only).
    Ambiguous,
    /// The session, or the `videos` entry, does not exist (or the entry's
    /// file is not on this device, for re-estimation).
    NotFound,
    /// A caller-supplied argument is invalid (a non-finite offset).
    InvalidArgument,
    /// The caller's cancel flag was set during a copy. Nothing was linked.
    Cancelled,
}

/// The two time spans a [`VideoErrorKind::NoOverlap`] names, UTC ms since
/// the Unix epoch. The video's side is `None` when the file carries no time
/// anchor at all; the session's side is `None` when no one session is
/// meant (a file matched against the whole library).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlapSpans {
    /// The video's first frame, UTC ms.
    pub video_start_utc_ms: Option<i64>,
    /// The video's end, UTC ms.
    pub video_end_utc_ms: Option<i64>,
    /// The session's start, UTC ms.
    pub session_start_utc_ms: Option<i64>,
    /// The session's end, UTC ms.
    pub session_end_utc_ms: Option<i64>,
}

/// Structured detail beside a [`VideoError`]'s message.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum VideoErrorDetail {
    /// Nothing structured to add.
    #[default]
    None,
    /// [`VideoErrorKind::NoOverlap`]: the two spans.
    Spans(OverlapSpans),
    /// [`VideoErrorKind::Ambiguous`]: the candidate session ids.
    SessionIds(Vec<String>),
}

/// Error for the video subsystem (CLAUDE.md §5: typed, never
/// `Err(String)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoError {
    /// The failure class.
    pub kind: VideoErrorKind,
    /// Human-readable text.
    pub message: String,
    /// Structured detail for the kinds that carry one.
    pub detail: VideoErrorDetail,
}

impl VideoError {
    /// An error with no structured detail.
    pub fn new(kind: VideoErrorKind, message: impl Into<String>) -> Self {
        VideoError { kind, message: message.into(), detail: VideoErrorDetail::None }
    }

    /// Shorthand for a [`VideoErrorKind::Parse`] error.
    pub fn parse(message: impl Into<String>) -> Self {
        VideoError::new(VideoErrorKind::Parse, message)
    }

    /// Shorthand for a [`VideoErrorKind::Io`] error.
    pub fn io(message: impl Into<String>) -> Self {
        VideoError::new(VideoErrorKind::Io, message)
    }

    /// A [`VideoErrorKind::NoOverlap`] error naming both spans.
    pub fn no_overlap(message: impl Into<String>, spans: OverlapSpans) -> Self {
        VideoError { kind: VideoErrorKind::NoOverlap, message: message.into(), detail: VideoErrorDetail::Spans(spans) }
    }

    /// A [`VideoErrorKind::Ambiguous`] error naming the candidates.
    pub fn ambiguous(message: impl Into<String>, session_ids: Vec<String>) -> Self {
        VideoError { kind: VideoErrorKind::Ambiguous, message: message.into(), detail: VideoErrorDetail::SessionIds(session_ids) }
    }
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for VideoError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> VideoLink {
        VideoLink {
            blob_sha256: "a".repeat(64),
            size_bytes: 10,
            file_name: "GX010123.MP4".to_string(),
            video_codec: "h264".to_string(),
            width_px: 1920,
            height_px: 1080,
            fps: 59.94,
            duration_s: 12.5,
            offset_s: -3.25,
            sync_method: SyncMethod::CreationTime,
            unlinked: false,
            updated_at_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn video_link_live_entry_round_trips_and_omits_unlinked() {
        // Arrange
        let entry = link();

        // Act
        let text = serde_json::to_string(&entry).unwrap();
        let back: VideoLink = serde_json::from_str(&text).unwrap();

        // Assert
        assert_eq!(back, entry);
        assert!(!text.contains("unlinked"));
        assert!(text.contains("\"sync_method\":\"creation_time\""));
    }

    #[test]
    fn video_link_tombstone_round_trips_with_unlinked_true() {
        // Arrange
        let entry = VideoLink { unlinked: true, ..link() };

        // Act
        let text = serde_json::to_string(&entry).unwrap();
        let back: VideoLink = serde_json::from_str(&text).unwrap();

        // Assert
        assert!(text.contains("\"unlinked\":true"));
        assert_eq!(back, entry);
    }

    #[test]
    fn session_video_serialises_flat_with_on_device() {
        // Arrange
        let v = SessionVideo { link: link(), on_device: true };

        // Act
        let value = serde_json::to_value(&v).unwrap();

        // Assert
        assert_eq!(value["on_device"], true);
        assert_eq!(value["blob_sha256"], "a".repeat(64));
    }
}
