//! FRB wrappers for the video subsystem (SPEC §33.3): container probing and
//! sync-offset estimation at link time. The export driver is NOT bridged in
//! phase 2 — desktop export UI is phase 3.

use idl_rs::session::handle::SessionHandle;
use idl_rs::video::gpmf::parse_gpmf;
use idl_rs::video::mp4box::{read_gpmd_samples_path, read_info_path};
use idl_rs::video::sync::{estimate_sync, SyncMethod};
use idl_rs::video::{VideoError, VideoErrorKind};

/// Discriminant for [`VideoFailure`] — freezed-free error crossing (the
/// `ParseFailure` precedent). Mirrors `idl_rs::video::VideoErrorKind`.
pub enum VideoFailureKind {
    Io,
    Parse,
    NoGpmf,
    NoOverlap,
    Export,
}

/// Error returned by the video bridge entry points.
pub struct VideoFailure {
    pub kind: VideoFailureKind,
    pub message: String,
}

impl From<VideoError> for VideoFailure {
    fn from(e: VideoError) -> Self {
        let kind = match e.kind {
            VideoErrorKind::Io => VideoFailureKind::Io,
            VideoErrorKind::Parse => VideoFailureKind::Parse,
            VideoErrorKind::NoGpmf => VideoFailureKind::NoGpmf,
            VideoErrorKind::NoOverlap => VideoFailureKind::NoOverlap,
            VideoErrorKind::Export => VideoFailureKind::Export,
        };
        VideoFailure { kind, message: e.message }
    }
}

/// Container facts for a video file (pure-Rust ISO-BMFF walk; no ffmpeg).
/// `fps` in frames/second, `duration_s` seconds, creation time in UTC ms.
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_s: f64,
    pub creation_time_utc_ms: Option<i64>,
    pub has_gpmd: bool,
}

/// How a sync offset was estimated. Plain unit enum → plain Dart enum.
pub enum VideoSyncMethod {
    Gpmf,
    CreationTime,
}

/// An estimated video↔session sync: `session_time_s = video_time_s +
/// offset_s`; confidence 0.9 (gpmf) / 0.3 (creation_time).
pub struct VideoSyncOutcome {
    pub offset_s: f64,
    pub confidence: f64,
    pub method: VideoSyncMethod,
}

/// Probe an `.mp4`/`.mov` container (SPEC §33.6 `video probe`, in-process).
pub fn video_probe(path: String) -> Result<VideoInfo, VideoFailure> {
    let info = read_info_path(&path)?;
    Ok(VideoInfo {
        width: info.width,
        height: info.height,
        fps: info.fps,
        duration_s: info.duration_s,
        creation_time_utc_ms: info.creation_time_utc_ms,
        has_gpmd: info.has_gpmd,
    })
}

/// Estimate the sync offset for `video_path` against the session: GPMF UTC
/// anchor when present, else container creation time. GPMF *absence* falls
/// through silently (normal); other errors surface. SPEC §33.3.
pub fn estimate_video_sync(
    handle: &SessionHandle,
    video_path: String,
) -> Result<VideoSyncOutcome, VideoFailure> {
    let info = read_info_path(&video_path)?;
    let telemetry = match read_gpmd_samples_path(&video_path) {
        Ok(samples) => Some(parse_gpmf(&samples)?),
        Err(e) if e.kind == VideoErrorKind::NoGpmf => None,
        Err(e) => return Err(e.into()),
    };
    let est = estimate_sync(telemetry.as_ref(), &info, handle)?;
    Ok(VideoSyncOutcome {
        offset_s: est.offset_s,
        confidence: est.confidence,
        method: match est.method {
            SyncMethod::Gpmf => VideoSyncMethod::Gpmf,
            SyncMethod::CreationTime => VideoSyncMethod::CreationTime,
        },
    })
}
