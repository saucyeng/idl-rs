//! Linking videos to sessions over `<data>` (video spec §3–§4, C1 §6
//! `videos`, C3 §3.12, C4 §2/§6). `std::fs` only.
//!
//! An MP4 is copied into `blobs/sha256/…` unchanged, streamed and hashed in
//! one pass, and the session's `session.json` gains a [`VideoLink`] entry.
//! Unlinking leaves a tombstone (`unlinked: true`) so sync cannot bring the
//! entry back (C4 §6). No function here has a clock: `now_ms` (UTC ms) is
//! injected by the caller and stamps `updated_at_ms`.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry};
use crate::store::blob::{blob_exists, blob_path};
use crate::store::parquet::{read_session_metadata, read_time_axis_bounds};
use crate::store::sync::ids::{is_valid_id, IdClass};
use crate::store::session_json::{effective_start_ms, parse_session_json, read_session_json, SessionJson};
use crate::video::gpmf::{parse_gpmf, VideoTelemetry};
use crate::video::mp4box::{probe_path, Mp4Info};
use crate::video::sync::{estimate_sync, video_span_utc_ms, SessionClock, SyncEstimate};
use crate::video::{SessionVideo, SyncMethod, VideoError, VideoErrorKind, VideoLink};

/// Bytes read and written per step of the streaming copy.
const COPY_CHUNK_BYTES: usize = 1024 * 1024;

/// Bytes copied between two `"copying"` progress reports.
const PROGRESS_EVERY_BYTES: u64 = 16 * 1024 * 1024;

/// One progress report from [`link_video`] (C3 §3.12's `Progress`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkProgress {
    /// `"probing"` (reading the container) or `"copying"` (into `blobs/`).
    pub phase: &'static str,
    /// Bytes copied so far (`"copying"`); `0` while probing.
    pub done: u64,
    /// The file's size in bytes (`"copying"`); `None` while probing.
    pub total: Option<u64>,
}

/// The sessions one file's time span overlaps (C3 §3.12 `match_videos`).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoMatch {
    /// The file, as given.
    pub path: PathBuf,
    /// Overlapping session ids, ascending.
    pub session_ids: Vec<String>,
    /// The video's span, UTC ms `(first frame, end)`; `None` when the file
    /// could not be read or carries no time.
    pub video_span_utc_ms: Option<(i64, i64)>,
    /// Why the file could not be matched, when it could not.
    pub error: Option<VideoError>,
}

fn not_found(message: impl Into<String>) -> VideoError {
    VideoError::new(VideoErrorKind::NotFound, message)
}

fn session_dir(data_root: &Path, session_id: &str) -> Result<PathBuf, VideoError> {
    // A session id is a directory name; anything not shaped like one (and
    // so anything that could leave `sessions/`) is simply not a session.
    if !is_valid_id(session_id, IdClass::Session) {
        return Err(not_found(format!("session {session_id:?} not found")));
    }
    let dir = data_root.join("sessions").join(session_id);
    if !dir.join("session.json").is_file() {
        return Err(not_found(format!("session {session_id} not found")));
    }
    Ok(dir)
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Reads, changes and atomically rewrites one session's `session.json`
/// (C4 §4). `apply` makes the change and returns the entry to report; on an
/// optimistic-concurrency conflict it is re-applied to the file as it now
/// stands, so a concurrent edit to another field is kept.
fn modify_session_json(
    data_root: &Path,
    session_id: &str,
    mut apply: impl FnMut(&mut SessionJson) -> Result<VideoLink, VideoError>,
) -> Result<VideoLink, VideoError> {
    let target = session_dir(data_root, session_id)?.join("session.json");
    let raw = std::fs::read(&target).map_err(|e| VideoError::io(format!("read {}: {e}", target.display())))?;
    let based_on = sha256_hex(&raw);
    let mut doc = parse_session_json(&raw).map_err(|e| VideoError::io(format!("{}: {}", target.display(), e.message)))?;
    let mut link = apply(&mut doc)?;
    let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| VideoError::io(format!("serialising session.json: {e}")))?;

    let mut retry_err: Option<VideoError> = None;
    write_atomic_with_retry(data_root, &target, &bytes, Some(&based_on), |current| {
        let Ok(mut current_doc) = parse_session_json(current) else { return current.to_vec() };
        match apply(&mut current_doc) {
            Ok(l) => {
                link = l;
                serde_json::to_vec_pretty(&current_doc).unwrap_or_else(|_| current.to_vec())
            }
            Err(e) => {
                retry_err = Some(e);
                current.to_vec()
            }
        }
    })
    .map_err(|e| VideoError::io(format!("write {}: {e}", target.display())))?;
    match retry_err {
        Some(e) => Err(e),
        None => Ok(link),
    }
}

/// The session's clock (effective start, GPS time, length) for estimation.
fn session_clock(data_root: &Path, session_id: &str) -> Result<SessionClock, VideoError> {
    let dir = session_dir(data_root, session_id)?;
    let meta = read_session_metadata(&dir.join("data.parquet"))
        .map_err(|e| VideoError::io(format!("reading {}: {}", dir.display(), e.message)))?;
    let doc = read_session_json(&dir.join("session.json"))
        .map_err(|e| VideoError::io(format!("reading {}: {}", dir.display(), e.message)))?;
    SessionClock::read(&dir, effective_start_ms(&doc, meta.timestamp_utc_ms))
}

/// Probes a file: container facts plus GPMF telemetry. A GPMF track that
/// does not parse is treated as absent, so estimation falls back to the
/// creation time rather than refusing the file.
fn probe_file(path: &Path) -> Result<(Mp4Info, Option<VideoTelemetry>), VideoError> {
    let (info, gpmd) = probe_path(path)?;
    let telemetry = gpmd.and_then(|samples| parse_gpmf(&samples).ok());
    Ok((info, telemetry))
}

/// Streams `source` into `blobs/sha256/…`, hashing as it goes, through a
/// `tmp/` file renamed into place (C4 §4). Returns `(sha256 hex, size in
/// bytes, whether this call wrote the blob)`: when the blob already exists
/// the temporary copy is discarded and nothing is written. `cancel` is
/// polled between chunks; a cancelled copy leaves nothing behind.
fn copy_into_blobs(
    data_root: &Path,
    source: &Path,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(LinkProgress),
) -> Result<(String, u64, bool), VideoError> {
    let io = |what: &str, e: std::io::Error| VideoError::io(format!("{what}: {e}"));
    let mut input = std::fs::File::open(source).map_err(|e| io(&format!("open {}", source.display()), e))?;
    let total = input.metadata().map_err(|e| io(&format!("stat {}", source.display()), e))?.len();

    let tmp_dir = data_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| io(&format!("create {}", tmp_dir.display()), e))?;
    let tmp_path = tmp_dir.join(format!("video-{}.part", uuid::Uuid::new_v4()));

    let result = (|| -> Result<(String, u64), VideoError> {
        let mut output = std::fs::File::create(&tmp_path).map_err(|e| io(&format!("create {}", tmp_path.display()), e))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; COPY_CHUNK_BYTES];
        let mut done = 0u64;
        let mut reported = 0u64;
        on_progress(LinkProgress { phase: "copying", done: 0, total: Some(total) });
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(VideoError::new(VideoErrorKind::Cancelled, "linking was cancelled"));
            }
            let n = input.read(&mut buf).map_err(|e| io(&format!("read {}", source.display()), e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            output.write_all(&buf[..n]).map_err(|e| io(&format!("write {}", tmp_path.display()), e))?;
            done += n as u64;
            if done - reported >= PROGRESS_EVERY_BYTES {
                reported = done;
                on_progress(LinkProgress { phase: "copying", done, total: Some(total) });
            }
        }
        output.sync_all().map_err(|e| io(&format!("fsync {}", tmp_path.display()), e))?;
        on_progress(LinkProgress { phase: "copying", done, total: Some(total) });
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Ok((digest, done))
    })();

    let (digest, size) = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
    };

    let dest = blob_path(data_root, &digest);
    if dest.is_file() {
        let _ = std::fs::remove_file(&tmp_path);
        return Ok((digest, size, false));
    }
    if let Some(parent) = dest.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(io(&format!("create {}", parent.display()), e));
        }
    }
    if let Err(e) = std::fs::rename(&tmp_path, &dest) {
        let _ = std::fs::remove_file(&tmp_path);
        // Another link of the same file may have won the race.
        if dest.is_file() {
            return Ok((digest, size, false));
        }
        return Err(io(&format!("rename into {}", dest.display()), e));
    }
    Ok((digest, size, true))
}

/// Links the MP4 at `source` to `session_id` (C3 §3.12 `link_video`).
///
/// In order: reads the container (`"probing"`) and estimates the offset
/// against the session's clock (GPMF GPS time, else creation time, else
/// [`SyncMethod::Manual`] at 0), then streams the file into `blobs/`
/// (`"copying"`; the blob is not rewritten when it already exists), then
/// appends the entry to `session.json`. Probing first means a file that is
/// not an MP4, or that misses the session, is refused before gigabytes are
/// copied. The source file is only read, never moved.
///
/// A file whose blob already has an entry in this session, live or
/// tombstoned, is re-linked: the entry comes back live with its offset and
/// `sync_method` kept, even when a fresh estimate would miss the session.
///
/// Errors: `NotFound` (no such session), `Parse` (not an MP4 we can read),
/// `NoOverlap` (an anchored video misses the session), `Cancelled`
/// (`cancel` was set during the copy), `Io`.
pub fn link_video(
    data_root: &Path,
    session_id: &str,
    source: &Path,
    now_ms: i64,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(LinkProgress),
) -> Result<VideoLink, VideoError> {
    let dir = session_dir(data_root, session_id)?;
    on_progress(LinkProgress { phase: "probing", done: 0, total: None });
    let (info, telemetry) = probe_file(source)?;
    let estimate: Result<SyncEstimate, VideoError> =
        session_clock(data_root, session_id).and_then(|clock| estimate_sync(telemetry.as_ref(), &info, &clock));
    if let Err(e) = &estimate {
        if e.kind != VideoErrorKind::NoOverlap {
            return Err(e.clone());
        }
        // Only an entry already in this session can rescue a miss (it keeps
        // its offset). Matching the size first spares a pointless copy.
        let size = std::fs::metadata(source).map(|m| m.len()).unwrap_or(u64::MAX);
        let doc = read_session_json(&dir.join("session.json")).map_err(|e| VideoError::io(e.message))?;
        if !doc.videos.iter().any(|v| v.size_bytes == size) {
            return Err(e.clone());
        }
    }

    let (sha, size_bytes, wrote_blob) = copy_into_blobs(data_root, source, cancel, on_progress)?;
    let file_name = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();

    let result = modify_session_json(data_root, session_id, |doc| {
        if let Some(existing) = doc.videos.iter_mut().find(|v| v.blob_sha256 == sha) {
            existing.unlinked = false;
            existing.updated_at_ms = now_ms;
            return Ok(existing.clone());
        }
        let est = estimate.clone()?;
        let link = VideoLink {
            blob_sha256: sha.clone(),
            size_bytes,
            file_name: file_name.clone(),
            video_codec: info.video_codec.clone(),
            width_px: info.width,
            height_px: info.height,
            fps: info.fps,
            duration_s: info.duration_s,
            offset_s: est.offset_s,
            sync_method: est.method,
            unlinked: false,
            updated_at_ms: now_ms,
        };
        doc.videos.push(link.clone());
        Ok(link)
    });

    if result.is_err() && wrote_blob && !live_video_blobs(data_root).contains(&sha) {
        // Nothing links the copy this call made; don't leave it behind.
        let _ = std::fs::remove_file(blob_path(data_root, &sha));
    }
    result
}

/// Sets an entry's offset by hand (C3 §3.12 `set_video_offset`):
/// `sync_method` becomes [`SyncMethod::Manual`]. `offset_s` (s) must be
/// finite. A tombstoned entry is `NotFound`.
pub fn set_video_offset(data_root: &Path, session_id: &str, blob_sha256: &str, offset_s: f64, now_ms: i64) -> Result<VideoLink, VideoError> {
    if !offset_s.is_finite() {
        return Err(VideoError::new(VideoErrorKind::InvalidArgument, format!("offset_s must be finite, got {offset_s}")));
    }
    modify_session_json(data_root, session_id, |doc| {
        let entry = live_entry_mut(doc, blob_sha256)?;
        entry.offset_s = offset_s;
        entry.sync_method = SyncMethod::Manual;
        entry.updated_at_ms = now_ms;
        Ok(entry.clone())
    })
}

/// Re-runs the estimator for an entry (C3 §3.12 `reestimate_video_offset`)
/// — the only way a manual offset is replaced. Needs the blob on this
/// device (`NotFound` otherwise). An estimate that misses the session is
/// `NoOverlap` and leaves the entry unchanged.
pub fn reestimate_video_offset(data_root: &Path, session_id: &str, blob_sha256: &str, now_ms: i64) -> Result<VideoLink, VideoError> {
    let doc = read_session_json(&session_dir(data_root, session_id)?.join("session.json")).map_err(|e| VideoError::io(e.message))?;
    if !doc.videos.iter().any(|v| v.blob_sha256 == blob_sha256 && !v.unlinked) {
        return Err(not_found(format!("session {session_id} has no linked video {blob_sha256}")));
    }
    if !is_sha256_hex(blob_sha256) || !blob_exists(data_root, blob_sha256) {
        return Err(not_found(format!("video {blob_sha256} is not on this device")));
    }
    let (info, telemetry) = probe_file(&blob_path(data_root, blob_sha256))?;
    let est = estimate_sync(telemetry.as_ref(), &info, &session_clock(data_root, session_id)?)?;
    modify_session_json(data_root, session_id, |doc| {
        let entry = live_entry_mut(doc, blob_sha256)?;
        entry.offset_s = est.offset_s;
        entry.sync_method = est.method;
        entry.updated_at_ms = now_ms;
        Ok(entry.clone())
    })
}

/// Unlinks an entry (C3 §3.12 `unlink_video`), leaving a tombstone. With
/// `delete_file`, the blob is removed from this device too, but only when
/// no other session's live entry references it; otherwise the request is
/// ignored and the blob kept. Unlinking a tombstone again returns it
/// unchanged (and still honours `delete_file`).
pub fn unlink_video(data_root: &Path, session_id: &str, blob_sha256: &str, delete_file: bool, now_ms: i64) -> Result<VideoLink, VideoError> {
    let tombstone = modify_session_json(data_root, session_id, |doc| {
        let entry = doc
            .videos
            .iter_mut()
            .find(|v| v.blob_sha256 == blob_sha256)
            .ok_or_else(|| not_found(format!("session {session_id} has no video {blob_sha256}")))?;
        if !entry.unlinked {
            entry.unlinked = true;
            entry.updated_at_ms = now_ms;
        }
        Ok(entry.clone())
    })?;
    if delete_file && is_sha256_hex(blob_sha256) && !live_video_blobs(data_root).contains(blob_sha256) {
        let path = blob_path(data_root, blob_sha256);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(VideoError::io(format!("remove {}: {e}", path.display()))),
        }
    }
    Ok(tombstone)
}

fn live_entry_mut<'a>(doc: &'a mut SessionJson, blob_sha256: &str) -> Result<&'a mut VideoLink, VideoError> {
    let session_id = doc.session_id.clone();
    doc.videos
        .iter_mut()
        .find(|v| v.blob_sha256 == blob_sha256 && !v.unlinked)
        .ok_or_else(|| not_found(format!("session {session_id} has no linked video {blob_sha256}")))
}

/// Every session's parsed `session.json`, in directory order. A session
/// whose file is missing or does not parse is skipped.
fn all_session_docs(data_root: &Path) -> Vec<SessionJson> {
    let Ok(entries) = std::fs::read_dir(data_root.join("sessions")) else { return Vec::new() };
    entries
        .flatten()
        .filter_map(|e| read_session_json(&e.path().join("session.json")).ok())
        .collect()
}

/// Blobs referenced by at least one live (not unlinked) `videos` entry, in
/// any session. What the `idl1-video` scheme may serve, what sync pulls
/// (C4 §6 "not pulled when unlinked"), and what `unlink_video` must not
/// delete.
pub fn live_video_blobs(data_root: &Path) -> BTreeSet<String> {
    all_session_docs(data_root)
        .into_iter()
        .flat_map(|d| d.videos.into_iter().filter(|v| !v.unlinked).map(|v| v.blob_sha256))
        .collect()
}

/// Blobs referenced by any `videos` entry, live or tombstoned, in any
/// session: the blobs the sync manifest marks `kind: "video"` (C4 §6).
pub fn video_blobs(data_root: &Path) -> BTreeSet<String> {
    all_session_docs(data_root)
        .into_iter()
        .flat_map(|d| d.videos.into_iter().map(|v| v.blob_sha256))
        .collect()
}

/// A session's live entries, in `session.json` order, each with whether its
/// blob is on this device and how many other sessions link it live (C3
/// §3.12, `get_session`'s `videos`). `other_links` comes from a scan of
/// every session's `session.json`, the canonical files, not the catalog.
pub fn session_videos(data_root: &Path, doc: &SessionJson) -> Vec<SessionVideo> {
    let live: Vec<&VideoLink> = doc.videos.iter().filter(|v| !v.unlinked).collect();
    if live.is_empty() {
        return Vec::new();
    }
    let others: Vec<SessionJson> = all_session_docs(data_root).into_iter().filter(|d| d.session_id != doc.session_id).collect();
    live.into_iter()
        .map(|v| SessionVideo {
            link: v.clone(),
            on_device: is_sha256_hex(&v.blob_sha256) && blob_exists(data_root, &v.blob_sha256),
            other_links: others
                .iter()
                .filter(|d| d.videos.iter().any(|o| !o.unlinked && o.blob_sha256 == v.blob_sha256))
                .count() as u32,
        })
        .collect()
}

/// One session's span in UTC ms, `(start, end)`, from its effective start
/// and its `t` axis. `None` when the session cannot be read.
fn session_span_utc_ms(session_dir: &Path) -> Option<(i64, i64)> {
    let meta = read_session_metadata(&session_dir.join("data.parquet")).ok()?;
    let doc = read_session_json(&session_dir.join("session.json")).ok()?;
    let start = effective_start_ms(&doc, meta.timestamp_utc_ms);
    let hi_us = read_time_axis_bounds(session_dir).ok()?.map(|(_, hi)| hi).unwrap_or(0);
    Some((start, start + hi_us / 1000))
}

/// For each file, the sessions whose UTC span overlaps the video's (C3
/// §3.12 `match_videos`). Reads only each file's container header and each
/// session's footer. A file that is not a readable MP4 carries `Parse`; one
/// with no time anchor matches nothing and carries `NoTime`.
pub fn match_videos(data_root: &Path, paths: &[PathBuf]) -> Vec<VideoMatch> {
    let mut sessions: Vec<(String, (i64, i64))> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data_root.join("sessions")) {
        for entry in entries.flatten() {
            if let Some(span) = session_span_utc_ms(&entry.path()) {
                sessions.push((entry.file_name().to_string_lossy().into_owned(), span));
            }
        }
    }
    sessions.sort_by(|a, b| a.0.cmp(&b.0));

    paths
        .iter()
        .map(|path| {
            let span = probe_file(path).map(|(info, telemetry)| video_span_utc_ms(telemetry.as_ref(), &info));
            match span {
                Err(e) => VideoMatch { path: path.clone(), session_ids: Vec::new(), video_span_utc_ms: None, error: Some(e) },
                Ok(None) => VideoMatch {
                    path: path.clone(),
                    session_ids: Vec::new(),
                    video_span_utc_ms: None,
                    error: Some(VideoError::new(VideoErrorKind::NoTime, crate::video::NO_TIME_MESSAGE)),
                },
                Ok(Some((v0, v1))) => VideoMatch {
                    path: path.clone(),
                    session_ids: sessions.iter().filter(|(_, (s0, s1))| v0 <= *s1 && v1 >= *s0).map(|(id, _)| id.clone()).collect(),
                    video_span_utc_ms: Some((v0, v1)),
                    error: None,
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{empty_session_json, write_session_json};
    use crate::video::mp4box::fixture::synthetic_mp4;
    use crate::video::VideoErrorDetail;
    use uuid::Uuid;

    /// Seconds between the MP4 epoch (1904) and the Unix epoch (1970).
    const MP4_EPOCH_TO_UNIX_S: u64 = 2_082_844_800;
    /// The fixture sessions' start, UTC ms.
    const START_MS: i64 = 1_700_000_000_000;
    const S1: &str = "00112233445566778899aabbccddeeff";
    const S2: &str = "ffeeddccbbaa99887766554433221100";

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A session starting at `start_ms` with `len_s` seconds of 1 Hz
    /// `GPS_EpochMs`, recorded at t = 0, 1, 2… s.
    fn write_session(root: &Path, session_id: &str, start_ms: i64, len_s: i64) {
        write_session_json(root, session_id, &empty_session_json(session_id), None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: Some("d1".to_string()),
            timestamp_utc_ms: start_ms,
            timestamp_source: TimestampSource::Header,
            config_checksum: Some("cafef00d".to_string()),
            source_format: SourceFormat::Idl0,
            blob_sha256: "0".repeat(64),
            channels: vec![Channel {
                channel_id: "GPS_EpochMs".to_string(),
                t_us: (0..=len_s).map(|i| i * 1_000_000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1.0,
                column: RawColumn::F64((0..=len_s).map(|i| (start_ms + i * 1000) as f64).collect()),
                source_kind: "gps".to_string(),
                unit: "ms".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, crate::parse::IDL0_IMPORTER_VERSION).unwrap();
    }

    /// A 10 s synthetic MP4 created at `created_utc_ms` (0 = no creation
    /// time), written outside `<data>`. `tag` makes the bytes distinct.
    fn write_video(dir: &Path, name: &str, created_utc_ms: i64, tag: &[u8]) -> PathBuf {
        let creation_1904_s = if created_utc_ms == 0 { 0 } else { (created_utc_ms / 1000) as u64 + MP4_EPOCH_TO_UNIX_S };
        let path = dir.join(name);
        std::fs::write(&path, synthetic_mp4(creation_1904_s, &[tag])).unwrap();
        path
    }

    fn doc(root: &Path, session_id: &str) -> SessionJson {
        read_session_json(&root.join("sessions").join(session_id).join("session.json")).unwrap()
    }

    fn link(root: &Path, session_id: &str, source: &Path, now_ms: i64) -> Result<VideoLink, VideoError> {
        link_video(root, session_id, source, now_ms, &AtomicBool::new(false), &mut |_| {})
    }

    #[test]
    fn link_video_creation_time_inside_the_session_links_with_the_estimated_offset() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "GX010123.MP4", START_MS + 30_000, b"a");
        let mut phases: Vec<LinkProgress> = Vec::new();

        // Act
        let entry = link_video(&root, S1, &source, 42, &AtomicBool::new(false), &mut |p| phases.push(p)).unwrap();

        // Assert
        let bytes = std::fs::read(&source).unwrap();
        assert_eq!(entry.blob_sha256, sha256_hex(&bytes));
        assert_eq!(entry.size_bytes, bytes.len() as u64);
        assert_eq!(entry.file_name, "GX010123.MP4");
        assert_eq!(entry.video_codec, "h264");
        assert_eq!((entry.width_px, entry.height_px), (1920, 1080));
        assert!((entry.offset_s - 30.0).abs() < 1e-6, "offset {}", entry.offset_s);
        assert_eq!(entry.sync_method, SyncMethod::CreationTime);
        assert_eq!(entry.updated_at_ms, 42);
        assert_eq!(std::fs::read(blob_path(&root, &entry.blob_sha256)).unwrap(), bytes);
        assert!(source.is_file(), "the source is copied, never moved");
        assert_eq!(doc(&root, S1).videos, vec![entry]);
        assert_eq!(phases.first().unwrap().phase, "probing");
        assert_eq!(phases.last().unwrap(), &LinkProgress { phase: "copying", done: bytes.len() as u64, total: Some(bytes.len() as u64) });
        let leftovers = std::fs::read_dir(root.join("tmp")).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("video-")).count();
        assert_eq!(leftovers, 0, "no .part left in tmp/");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_a_file_with_no_time_links_as_manual_at_zero() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "quik.mp4", 0, b"a");

        // Act
        let entry = link(&root, S1, &source, 1).unwrap();

        // Assert
        assert_eq!(entry.sync_method, SyncMethod::Manual);
        assert_eq!(entry.offset_s, 0.0);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_a_video_hours_away_is_no_overlap_and_nothing_is_written() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "later.mp4", START_MS + 3 * 3_600_000, b"a");

        // Act
        let err = link(&root, S1, &source, 1).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NoOverlap);
        assert!(matches!(err.detail, VideoErrorDetail::Spans(s) if s.session_start_utc_ms == Some(START_MS)));
        assert!(!root.join("blobs").exists());
        assert!(doc(&root, S1).videos.is_empty());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_cancelled_leaves_no_blob_no_part_and_no_entry() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS, b"a");

        // Act
        let err = link_video(&root, S1, &source, 1, &AtomicBool::new(true), &mut |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::Cancelled);
        assert!(!root.join("blobs").exists());
        let leftovers = std::fs::read_dir(root.join("tmp")).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("video-")).count();
        assert_eq!(leftovers, 0);
        assert!(doc(&root, S1).videos.is_empty());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_not_an_mp4_is_parse_and_an_unknown_session_is_not_found() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let garbage = outside.join("notes.mp4");
        std::fs::write(&garbage, b"definitely not a video").unwrap();
        let source = write_video(&outside, "a.mp4", START_MS, b"a");

        // Act
        let parse = link(&root, S1, &garbage, 1).unwrap_err();
        let missing = link(&root, S2, &source, 1).unwrap_err();
        let traversal = link(&root, "../../etc", &source, 1).unwrap_err();

        // Assert
        assert_eq!(parse.kind, VideoErrorKind::Parse);
        assert_eq!(missing.kind, VideoErrorKind::NotFound);
        assert_eq!(traversal.kind, VideoErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_relinking_an_unlinked_file_brings_the_entry_back_with_its_manual_offset() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS + 30_000, b"a");
        let first = link(&root, S1, &source, 1).unwrap();
        set_video_offset(&root, S1, &first.blob_sha256, 12.5, 2).unwrap();
        unlink_video(&root, S1, &first.blob_sha256, false, 3).unwrap();

        // Act
        let again = link(&root, S1, &source, 4).unwrap();

        // Assert
        assert!(!again.unlinked);
        assert_eq!(again.offset_s, 12.5);
        assert_eq!(again.sync_method, SyncMethod::Manual);
        assert_eq!(again.updated_at_ms, 4);
        assert_eq!(doc(&root, S1).videos.len(), 1, "one entry per blob");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn link_video_relinking_a_manually_aligned_video_the_estimator_would_refuse_still_relinks() {
        // Arrange — the camera clock is hours out, so the estimate misses
        // the session, but the entry already exists with a hand-set offset.
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS + 3 * 3_600_000, b"a");
        let sha = sha256_hex(&std::fs::read(&source).unwrap());
        let mut d = doc(&root, S1);
        d.videos.push(VideoLink {
            blob_sha256: sha.clone(),
            size_bytes: std::fs::metadata(&source).unwrap().len(),
            file_name: "a.mp4".to_string(),
            video_codec: "h264".to_string(),
            width_px: 1920,
            height_px: 1080,
            fps: 10.0,
            duration_s: 10.0,
            offset_s: 7.0,
            sync_method: SyncMethod::Manual,
            unlinked: true,
            updated_at_ms: 1,
        });
        let raw = std::fs::read(root.join("sessions").join(S1).join("session.json")).unwrap();
        write_session_json(&root, S1, &d, Some(&sha256_hex(&raw))).unwrap();

        // Act
        let again = link(&root, S1, &source, 5).unwrap();

        // Assert
        assert!(!again.unlinked);
        assert_eq!(again.offset_s, 7.0);
        assert!(blob_exists(&root, &sha));

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn set_video_offset_sets_manual_and_refuses_a_tombstone_or_a_non_finite_offset() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS + 30_000, b"a");
        let sha = link(&root, S1, &source, 1).unwrap().blob_sha256;

        // Act
        let set = set_video_offset(&root, S1, &sha, -2.0, 9).unwrap();
        let nan = set_video_offset(&root, S1, &sha, f64::NAN, 10).unwrap_err();
        unlink_video(&root, S1, &sha, false, 11).unwrap();
        let gone = set_video_offset(&root, S1, &sha, 1.0, 12).unwrap_err();

        // Assert
        assert_eq!((set.offset_s, set.sync_method, set.updated_at_ms), (-2.0, SyncMethod::Manual, 9));
        assert_eq!(nan.kind, VideoErrorKind::InvalidArgument);
        assert_eq!(gone.kind, VideoErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn reestimate_video_offset_replaces_a_manual_offset_and_needs_the_file_here() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS + 30_000, b"a");
        let sha = link(&root, S1, &source, 1).unwrap().blob_sha256;
        set_video_offset(&root, S1, &sha, 99.0, 2).unwrap();

        // Act
        let back = reestimate_video_offset(&root, S1, &sha, 3).unwrap();
        std::fs::remove_file(blob_path(&root, &sha)).unwrap();
        let missing = reestimate_video_offset(&root, S1, &sha, 4).unwrap_err();

        // Assert
        assert_eq!(back.sync_method, SyncMethod::CreationTime);
        assert!((back.offset_s - 30.0).abs() < 1e-6);
        assert_eq!(missing.kind, VideoErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn unlink_video_delete_file_keeps_a_blob_another_session_still_links() {
        // Arrange — the same file linked to two overlapping sessions.
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        write_session(&root, S2, START_MS, 120);
        let source = write_video(&outside, "a.mp4", START_MS + 30_000, b"a");
        let sha = link(&root, S1, &source, 1).unwrap().blob_sha256;
        link(&root, S2, &source, 1).unwrap();

        // Act
        let shared = session_videos(&root, &doc(&root, S1));
        let t1 = unlink_video(&root, S1, &sha, true, 5).unwrap();
        let after = session_videos(&root, &doc(&root, S2));
        let kept = blob_exists(&root, &sha);
        unlink_video(&root, S2, &sha, true, 6).unwrap();

        // Assert
        assert_eq!(shared[0].other_links, 1, "S2 links it too");
        assert_eq!(after[0].other_links, 0, "S1's entry is a tombstone now");
        assert!(t1.unlinked);
        assert_eq!(t1.updated_at_ms, 5);
        assert!(kept, "S2 still linked it");
        assert!(!blob_exists(&root, &sha), "nothing links it any more");
        assert!(doc(&root, S1).videos[0].unlinked, "the tombstone stays");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn unlink_video_an_unknown_entry_is_not_found() {
        // Arrange
        let root = temp_root();
        write_session(&root, S1, START_MS, 120);

        // Act
        let err = unlink_video(&root, S1, &"a".repeat(64), true, 1).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn session_videos_reports_live_entries_with_on_device_and_the_blob_sets_split_live_from_all() {
        // Arrange
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        let a = link(&root, S1, &write_video(&outside, "a.mp4", START_MS, b"a"), 1).unwrap().blob_sha256;
        let b = link(&root, S1, &write_video(&outside, "b.mp4", START_MS, b"b"), 1).unwrap().blob_sha256;
        let c = link(&root, S1, &write_video(&outside, "c.mp4", START_MS, b"c"), 1).unwrap().blob_sha256;
        unlink_video(&root, S1, &b, false, 2).unwrap();
        std::fs::remove_file(blob_path(&root, &c)).unwrap();

        // Act
        let videos = session_videos(&root, &doc(&root, S1));

        // Assert
        let seen: Vec<(String, bool)> = videos.iter().map(|v| (v.link.blob_sha256.clone(), v.on_device)).collect();
        assert_eq!(seen, vec![(a.clone(), true), (c.clone(), false)]);
        assert_eq!(live_video_blobs(&root), [a.clone(), c.clone()].into_iter().collect());
        assert_eq!(video_blobs(&root), [a, b, c].into_iter().collect());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn match_videos_names_the_overlapping_session_and_types_each_failure() {
        // Arrange — S1 spans [START, +120 s]; S2 starts a day later.
        let root = temp_root();
        let outside = temp_root();
        write_session(&root, S1, START_MS, 120);
        write_session(&root, S2, START_MS + 86_400_000, 120);
        let inside = write_video(&outside, "in.mp4", START_MS + 60_000, b"a");
        let nowhere = write_video(&outside, "nowhere.mp4", START_MS + 3 * 3_600_000, b"b");
        let untimed = write_video(&outside, "untimed.mp4", 0, b"c");
        let garbage = outside.join("garbage.mp4");
        std::fs::write(&garbage, b"nope").unwrap();

        // Act
        let out = match_videos(&root, &[inside.clone(), nowhere, untimed, garbage]);

        // Assert
        assert_eq!(out[0].path, inside);
        assert_eq!(out[0].session_ids, vec![S1.to_string()]);
        assert!(out[0].error.is_none());
        assert!(out[1].session_ids.is_empty());
        assert!(out[1].error.is_none());
        assert_eq!(out[2].error.as_ref().unwrap().kind, VideoErrorKind::NoTime);
        assert_eq!(out[3].error.as_ref().unwrap().kind, VideoErrorKind::Parse);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
