//! Video commands (C3 §3.12, video spec `2026-10-16-idl1-video-DRAFT.md`
//! step 1): thin glue over `idl_rs::store::video`. Every command is `async`
//! (R201): linking streams gigabytes, and even the small ones read every
//! session's `session.json` to decide what a blob is still linked by.
//!
//! Playback bytes are not a command: see [`crate::video::video_protocol`].

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use idl_rs::store::video as store;
use idl_rs::video::VideoLink;

use crate::commands::device::Progress;
use crate::error::IpcError;
use crate::state::DataDir;

/// UTC milliseconds now.
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// One file's answer from `match_videos` (C3 §3.12).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VideoMatch {
    /// The file, as given.
    pub path: String,
    /// Sessions whose time span the video's overlaps, ascending.
    pub session_ids: Vec<String>,
    /// Why the file could not be matched, when it could not.
    pub error: Option<IpcError>,
}

/// Transport-agnostic core of `link_video`. `cancel` is core's copy flag;
/// C3 defines no cancel command, so the command passes one that is never
/// set.
pub(crate) fn link_video_via(
    data_dir: &Path,
    session_id: &str,
    path: &Path,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<VideoLink, IpcError> {
    Ok(store::link_video(data_dir, session_id, path, now_ms(), cancel, &mut |p| {
        on_progress(Progress { done: p.done, total: p.total, phase: p.phase.to_string() })
    })?)
}

/// Transport-agnostic core of `match_videos`.
pub(crate) fn match_videos_via(data_dir: &Path, paths: &[String]) -> Vec<VideoMatch> {
    let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    store::match_videos(data_dir, &paths)
        .into_iter()
        .map(|m| VideoMatch {
            path: m.path.to_string_lossy().into_owned(),
            session_ids: m.session_ids,
            error: m.error.map(IpcError::from),
        })
        .collect()
}

/// C3 §3.12 `link_video(session_id, path, on_progress)`: copies the file
/// into `blobs/` (phase `"copying"`, bytes) after reading its container
/// (phase `"probing"`), estimates the offset and appends the entry.
#[tauri::command(async)]
pub fn link_video(
    session_id: String,
    path: String,
    on_progress: tauri::ipc::Channel<Progress>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<VideoLink, IpcError> {
    let never = AtomicBool::new(false);
    link_video_via(&data_dir.0, &session_id, Path::new(&path), &never, &mut |p| {
        let _ = on_progress.send(p);
    })
}

/// C3 §3.12 `match_videos(paths)`: the sessions each file overlaps.
#[tauri::command(async)]
pub fn match_videos(paths: Vec<String>, data_dir: tauri::State<'_, DataDir>) -> Vec<VideoMatch> {
    match_videos_via(&data_dir.0, &paths)
}

/// C3 §3.12 `set_video_offset(session_id, blob_sha256, offset_s)`: a hand
/// alignment, `sync_method: "manual"`. `offset_s` in seconds.
#[tauri::command(async)]
pub fn set_video_offset(
    session_id: String,
    blob_sha256: String,
    offset_s: f64,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<VideoLink, IpcError> {
    Ok(store::set_video_offset(&data_dir.0, &session_id, &blob_sha256, offset_s, now_ms())?)
}

/// C3 §3.12 `reestimate_video_offset(session_id, blob_sha256)`: the only
/// way a manual offset is replaced.
#[tauri::command(async)]
pub fn reestimate_video_offset(session_id: String, blob_sha256: String, data_dir: tauri::State<'_, DataDir>) -> Result<VideoLink, IpcError> {
    Ok(store::reestimate_video_offset(&data_dir.0, &session_id, &blob_sha256, now_ms())?)
}

/// C3 §3.12 `unlink_video(session_id, blob_sha256, delete_file)`: leaves a
/// tombstone; `delete_file` removes this device's copy only when no other
/// session's live entry references it.
#[tauri::command(async)]
pub fn unlink_video(
    session_id: String,
    blob_sha256: String,
    delete_file: bool,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<VideoLink, IpcError> {
    Ok(store::unlink_video(&data_dir.0, &session_id, &blob_sha256, delete_file, now_ms())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IpcErrorKind;

    #[test]
    fn match_videos_via_an_unreadable_file_carries_a_typed_video_parse_error() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("not-a-video.mp4");
        std::fs::write(&file, b"nope").unwrap();

        // Act
        let out = match_videos_via(root.path(), &[file.to_string_lossy().into_owned()]);

        // Assert
        assert_eq!(out.len(), 1);
        assert!(out[0].session_ids.is_empty());
        assert_eq!(out[0].error.as_ref().unwrap().kind, IpcErrorKind::VideoParse);
    }

    #[test]
    fn link_video_via_an_unknown_session_is_not_found() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("a.mp4");
        std::fs::write(&file, b"nope").unwrap();

        // Act
        let err = link_video_via(root.path(), "00112233445566778899aabbccddeeff", &file, &AtomicBool::new(false), &mut |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
    }
}
