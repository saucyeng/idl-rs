//! A linked video's own motion: sync by it, and keep it as channels of the
//! ride (camera IMU spec, `docs/superpowers/specs/2026-11-23-idl1-camera-imu-DRAFT.md`).
//!
//! The camera's `ACCL`, `GYRO` and `GRAV` (GPMF, [`parse_camera_imu`]) are
//! written to `sessions/<id>/derived/<hash>.parquet` with `derived_kind`
//! `"camera"`, one file per stream, content-addressed by the video's sha256,
//! its offset and the parser version (C1 §5). They depend on the video and
//! its offset, not on the logger's blob, so they cannot live in
//! `data.parquet`. [`crate::estimated`] reads them beside the physics
//! solver's outputs, so `[Cam_AccelX]` is a channel like any other.
//!
//! [`refine_offset`] sharpens a GPS or creation-time offset by
//! cross-correlating the camera's accelerometer with the pods' (spec §2).
//! `std::fs` only.

use std::path::{Path, PathBuf};

use parquet::file::reader::{FileReader, SerializedFileReader};

use idl_rs::store::parquet::{read_channel, ParquetStoreErrorKind};
use idl_rs::store::session_json::read_session_json;
use idl_rs::video::gpmf::{parse_camera_imu, CameraImu, ImuStream};
use idl_rs::video::mp4box::probe_path;
use idl_rs::video::sync::{refine_offset_by_imu, SyncEstimate};
use idl_rs::video::SyncMethod;

use crate::blob::{blob_exists, blob_path};
use crate::derived::{write_derived_parquet, DerivedOutput};

/// The `derived_kind` camera channels are written under.
pub const CAMERA_DERIVED_KIND: &str = "camera";

/// Bumped when [`parse_camera_imu`]'s output changes, so files written by an
/// older parser are rewritten (it is part of each file's address).
const CAMERA_PARSER_VERSION: u32 = 1;

/// How far past the clip's ends the pods are read for the motion sync, s
/// (the sync's ±3 s search plus margin).
const POD_MARGIN_S: f64 = 4.0;

/// The camera's motion from the video file at `path`, or `None` when it has
/// no GPMF track or no `ACCL` in it.
pub fn read_camera_imu(path: &Path) -> Option<CameraImu> {
    let (_, gpmd) = probe_path(path).ok()?;
    let imu = parse_camera_imu(&gpmd?).ok()?;
    (!imu.accel_g.xyz.is_empty()).then_some(imu)
}

/// Each pod's accelerometer magnitude, g, over session seconds
/// `[from_s, to_s]`: `(session s, |accel|)` per pod that has
/// `IMU<n>_AccelX/Y/Z` on one time axis. A pod that cannot be read is left
/// out (the sync is best effort).
pub fn pod_accel_magnitudes(session_dir: &Path, from_s: f64, to_s: f64) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut pods = Vec::new();
    for n in 0..4 {
        let mut axes = Vec::with_capacity(3);
        for axis in ["X", "Y", "Z"] {
            match read_channel(session_dir, &format!("IMU{n}_Accel{axis}")) {
                Ok(c) => {
                    let t: Vec<f64> = c.t_us.iter().map(|&t| t as f64 / 1e6).collect();
                    let (a, b) = (t.partition_point(|&s| s < from_s), t.partition_point(|&s| s <= to_s));
                    let v = c.materialize();
                    axes.push((t[a..b].to_vec(), v[a..b].to_vec()));
                }
                Err(e) if e.kind == ParquetStoreErrorKind::NotFound => break,
                Err(_) => break,
            }
        }
        if axes.len() != 3 || axes[0].0.len() != axes[1].0.len() || axes[0].0.len() != axes[2].0.len() || axes[0].0.len() < 2 {
            continue;
        }
        let t = axes[0].0.clone();
        let m: Vec<f64> = (0..t.len()).map(|i| (axes[0].1[i].powi(2) + axes[1].1[i].powi(2) + axes[2].1[i].powi(2)).sqrt()).collect();
        let keep: Vec<usize> = (0..t.len()).filter(|&i| m[i].is_finite()).collect();
        pods.push((keep.iter().map(|&i| t[i]).collect(), keep.iter().map(|&i| m[i]).collect()));
    }
    pods
}

/// Sharpens `estimate` by the camera's motion against the pods (spec §2).
/// A manual estimate, a camera with no accelerometer, or a sync the pods do
/// not agree on returns `estimate` unchanged; an accepted one becomes
/// [`SyncMethod::Imu`] with the pods' median correlation as its confidence.
pub fn refine_offset(session_dir: &Path, imu: Option<&CameraImu>, duration_s: f64, estimate: SyncEstimate) -> SyncEstimate {
    let Some(imu) = imu else { return estimate };
    if estimate.method == SyncMethod::Manual || imu.accel_g.xyz.len() < 2 {
        return estimate;
    }
    let pods = pod_accel_magnitudes(session_dir, estimate.offset_s - POD_MARGIN_S, estimate.offset_s + duration_s + POD_MARGIN_S);
    let cam_m: Vec<f64> = imu.accel_g.xyz.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()).collect();
    let pod_refs: Vec<(&[f64], &[f64])> = pods.iter().map(|(t, m)| (t.as_slice(), m.as_slice())).collect();
    match refine_offset_by_imu((&imu.accel_g.t_video_s, &cam_m), &pod_refs, estimate.offset_s) {
        Some(found) => SyncEstimate { offset_s: found.offset_s, confidence: found.confidence, method: SyncMethod::Imu },
        None => estimate,
    }
}

/// `(stored stream key, channel prefix, unit)` for each stream written.
const STREAMS: [(&str, &str, &str); 3] = [("ACCL", "Cam_Accel", "g"), ("GYRO", "Cam_Gyro", "deg/s"), ("GRAV", "Cam_Grav", "")];

/// The derived outputs of one stream on the ride's clock (`offset_s`, s),
/// samples before the ride's first dropped. Empty when the stream is.
fn stream_outputs(stream: &ImuStream, prefix: &str, unit: &str, offset_s: f64) -> Vec<DerivedOutput> {
    let keep: Vec<usize> = (0..stream.t_video_s.len()).filter(|&i| stream.t_video_s[i] + offset_s >= 0.0).collect();
    if keep.len() < 2 {
        return Vec::new();
    }
    let t_us: Vec<i64> = keep.iter().map(|&i| ((stream.t_video_s[i] + offset_s) * 1e6).round() as i64).collect();
    let span_s = (t_us[t_us.len() - 1] - t_us[0]) as f64 / 1e6;
    let rate_hz = if span_s > 0.0 { (t_us.len() - 1) as f64 / span_s } else { 0.0 };
    ["X", "Y", "Z"]
        .iter()
        .enumerate()
        .map(|(axis, name)| DerivedOutput {
            channel_id: format!("{prefix}{name}"),
            t_us: t_us.clone(),
            values: keep.iter().map(|&i| stream.xyz[i][axis]).collect(),
            nominal_rate_hz: rate_hz,
            unit: unit.to_string(),
        })
        .collect()
}

/// A derived file's `(derived_kind, config_json)` from its footer.
fn derived_meta(path: &Path) -> Option<(String, serde_json::Value)> {
    let reader = SerializedFileReader::new(std::fs::File::open(path).ok()?).ok()?;
    let kv = reader.metadata().file_metadata().key_value_metadata()?.clone();
    let get = |k: &str| kv.iter().find(|e| e.key == k).and_then(|e| e.value.clone());
    Some((get("derived_kind")?, serde_json::from_str(&get("config_json")?).ok()?))
}

/// Every `"camera"` derived file of `session_id` with its config.
fn camera_files(data_root: &Path, session_id: &str) -> Vec<(PathBuf, serde_json::Value)> {
    let dir = data_root.join("sessions").join(session_id).join("derived");
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .filter_map(|p| derived_meta(&p).filter(|(kind, _)| kind == CAMERA_DERIVED_KIND).map(|(_, cfg)| (p, cfg)))
        .collect()
}

/// Whether `cfg` is a current file for `video_sha256` at `offset_s`.
fn is_current(cfg: &serde_json::Value, video_sha256: &str, offset_s: f64) -> bool {
    cfg["video_sha256"].as_str() == Some(video_sha256)
        && cfg["offset_s"].as_f64() == Some(offset_s)
        && cfg["parser"].as_u64() == Some(CAMERA_PARSER_VERSION as u64)
}

/// Writes the camera channels of `video_sha256` at `offset_s` and deletes
/// that video's other `"camera"` files (an older offset or parser). Returns
/// the files written; nothing is written for a camera with no IMU streams.
pub fn write_camera_channels(data_root: &Path, session_id: &str, video_sha256: &str, offset_s: f64, imu: &CameraImu, now_ms: i64) -> Result<Vec<PathBuf>, String> {
    let mut sha = [0u8; 32];
    for (i, b) in sha.iter_mut().enumerate() {
        *b = u8::from_str_radix(video_sha256.get(i * 2..i * 2 + 2).ok_or("short sha256")?, 16).map_err(|e| e.to_string())?;
    }
    let inputs = vec![(format!("video:{video_sha256}"), sha)];
    let mut written = Vec::new();
    for ((key, prefix, unit), stream) in STREAMS.iter().zip([&imu.accel_g, &imu.gyro_dps, &imu.gravity]) {
        let outputs = stream_outputs(stream, prefix, unit, offset_s);
        if outputs.is_empty() {
            continue;
        }
        let config = serde_json::json!({ "video_sha256": video_sha256, "offset_s": offset_s, "stream": key, "parser": CAMERA_PARSER_VERSION });
        written.push(write_derived_parquet(data_root, session_id, CAMERA_DERIVED_KIND, &inputs, &config, &outputs, now_ms).map_err(|e| e.to_string())?);
    }
    for (path, cfg) in camera_files(data_root, session_id) {
        if cfg["video_sha256"].as_str() == Some(video_sha256) && !written.contains(&path) {
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(written)
}

/// Deletes every `"camera"` file of `video_sha256` in `session_id` (on unlink).
pub fn remove_camera_channels(data_root: &Path, session_id: &str, video_sha256: &str) {
    for (path, cfg) in camera_files(data_root, session_id) {
        if cfg["video_sha256"].as_str() == Some(video_sha256) {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Writes camera channels for every live linked video of `session_id` whose
/// blob is on this device and has no current file (linked before camera
/// channels existed, or synced from another device). Best effort: a video
/// that cannot be read is skipped, and the ride opens regardless.
pub fn ensure_camera_channels(data_root: &Path, session_id: &str, now_ms: i64) {
    let dir = data_root.join("sessions").join(session_id);
    let Ok(doc) = read_session_json(&dir.join("session.json")) else { return };
    if doc.videos.iter().all(|v| v.unlinked) {
        return;
    }
    let files = camera_files(data_root, session_id);
    for v in doc.videos.iter().filter(|v| !v.unlinked) {
        if files.iter().any(|(_, cfg)| is_current(cfg, &v.blob_sha256, v.offset_s)) || !blob_exists(data_root, &v.blob_sha256) {
            continue;
        }
        if let Some(imu) = read_camera_imu(&blob_path(data_root, &v.blob_sha256)) {
            let _ = write_camera_channels(data_root, session_id, &v.blob_sha256, v.offset_s, &imu, now_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(n: usize, dt: f64, t0: f64) -> ImuStream {
        ImuStream { t_video_s: (0..n).map(|i| t0 + i as f64 * dt).collect(), xyz: (0..n).map(|i| [i as f64, 0.0, 1.0]).collect() }
    }

    #[test]
    fn stream_outputs_mapped_by_offset_drops_samples_before_the_ride() {
        // Arrange — 5 samples at 0.0..0.4 s of video, offset −0.15 s.
        let s = stream(5, 0.1, 0.0);

        // Act
        let out = stream_outputs(&s, "Cam_Accel", "g", -0.15);

        // Assert — the first two (−0.15, −0.05 s) are before the ride.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].channel_id, "Cam_AccelX");
        assert_eq!(out[0].t_us, vec![50_000, 150_000, 250_000]);
        assert_eq!(out[0].values, vec![2.0, 3.0, 4.0]);
        assert!((out[0].nominal_rate_hz - 10.0).abs() < 1e-9);
    }

    #[test]
    fn write_camera_channels_new_offset_replaces_the_old_files() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl1-camera-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("sessions").join("s1")).unwrap();
        let imu = CameraImu { accel_g: stream(10, 0.005, 0.0), gyro_dps: stream(10, 0.005, 0.0), gravity: ImuStream::default() };
        let sha = "ab".repeat(32);

        // Act
        let first = write_camera_channels(&root, "s1", &sha, 10.0, &imu, 1).unwrap();
        let second = write_camera_channels(&root, "s1", &sha, 11.0, &imu, 2).unwrap();

        // Assert
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|p| !p.exists()), "old offset's files removed");
        assert!(second.iter().all(|p| p.exists()));
        assert_eq!(camera_files(&root, "s1").len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
