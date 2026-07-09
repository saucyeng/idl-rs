//! Video subsystem (phase 1): ISO-BMFF walking, GPMF telemetry parsing,
//! sync-offset estimation, and overlay-frame rasterization. Pure — the
//! sidecar-ffmpeg export driver lives in the `idl-rs-video-export` crate.
//! See docs/IDL0_SPEC.md §33.

pub mod gpmf;
pub mod mp4box;
pub mod render;
pub mod sync;

/// What went wrong in the video subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoErrorKind {
    /// Filesystem read failed.
    Io,
    /// Container/telemetry bytes did not parse.
    Parse,
    /// The container has no GPMF (`gpmd`) track.
    NoGpmf,
    /// The video and session time ranges do not overlap.
    NoOverlap,
    /// Export-side failure surfaced through the engine boundary.
    Export,
}

/// Error for the video subsystem: unit-enum `kind` + human-readable
/// `message` (mirrors `ConfigError`, SPEC §14 exception philosophy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoError {
    pub kind: VideoErrorKind,
    pub message: String,
}

impl VideoError {
    pub fn new(kind: VideoErrorKind, message: impl Into<String>) -> Self {
        VideoError {
            kind,
            message: message.into(),
        }
    }

    /// Shorthand for a `Parse`-kind error.
    pub fn parse(message: impl Into<String>) -> Self {
        VideoError::new(VideoErrorKind::Parse, message)
    }

    /// Shorthand for an `Io`-kind error.
    pub fn io(message: impl Into<String>) -> Self {
        VideoError::new(VideoErrorKind::Io, message)
    }
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for VideoError {}
