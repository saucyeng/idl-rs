//! Source-video probing via sidecar `ffprobe` (JSON output). The parse is a
//! pure function over the JSON text; only [`probe`] spawns a process.

use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::export::{ExportError, ExportErrorKind};

/// Facts about a source video the export plan needs.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoProbe {
    /// Coded width in pixels (pre-rotation).
    pub width: u32,
    /// Coded height in pixels (pre-rotation).
    pub height: u32,
    /// Frame rate in frames/second (from `r_frame_rate`, falling back to
    /// `avg_frame_rate`).
    pub fps: f64,
    /// Container duration in seconds.
    pub duration_s: f64,
    /// Display rotation in degrees from stream side data (0 when absent).
    /// ±90/±270 swap the display width/height.
    pub rotation_deg: i32,
    /// True when the container carries at least one audio stream.
    pub has_audio: bool,
}

#[derive(Deserialize)]
struct FfprobeDoc {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
    #[serde(default)]
    format: FfprobeFormat,
}

#[derive(Deserialize, Default)]
struct FfprobeFormat {
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Deserialize)]
struct FfprobeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    #[serde(default)]
    side_data_list: Vec<FfprobeSideData>,
}

#[derive(Deserialize)]
struct FfprobeSideData {
    rotation: Option<f64>,
}

fn err_probe(msg: impl Into<String>) -> ExportError {
    ExportError::new(ExportErrorKind::Probe, msg)
}

/// Parse an ffprobe rational like `"60000/1001"` → fps.
fn parse_rate(s: &str) -> Option<f64> {
    let mut it = s.split('/');
    let num: f64 = it.next()?.trim().parse().ok()?;
    match it.next() {
        Some(den) => {
            let den: f64 = den.trim().parse().ok()?;
            (den != 0.0).then(|| num / den)
        }
        None => Some(num),
    }
}

/// Parse `ffprobe -print_format json -show_streams -show_format` output.
pub fn parse_ffprobe_json(json: &str) -> Result<VideoProbe, ExportError> {
    let doc: FfprobeDoc =
        serde_json::from_str(json).map_err(|e| err_probe(format!("ffprobe JSON: {e}")))?;
    let video = doc
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| err_probe("ffprobe reported no video stream"))?;
    let fps = video
        .r_frame_rate
        .as_deref()
        .and_then(parse_rate)
        .filter(|&f| f > 0.0)
        .or_else(|| video.avg_frame_rate.as_deref().and_then(parse_rate))
        .filter(|&f| f > 0.0)
        .ok_or_else(|| err_probe("ffprobe reported no usable frame rate"))?;
    let rotation_deg = video
        .side_data_list
        .iter()
        .find_map(|sd| sd.rotation)
        .map(|r| r as i32)
        .unwrap_or(0);
    let duration_s = doc
        .format
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .ok_or_else(|| err_probe("ffprobe reported no container duration"))?;
    Ok(VideoProbe {
        width: video
            .width
            .ok_or_else(|| err_probe("video stream missing width"))?,
        height: video
            .height
            .ok_or_else(|| err_probe("video stream missing height"))?,
        fps,
        duration_s,
        rotation_deg,
        has_audio: doc
            .streams
            .iter()
            .any(|s| s.codec_type.as_deref() == Some("audio")),
    })
}

/// Spawn `ffprobe` at `ffprobe_path` against `video` and parse its JSON.
pub fn probe(video: &Path, ffprobe_path: &str) -> Result<VideoProbe, ExportError> {
    let output = Command::new(ffprobe_path)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(video)
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ExportError::new(
                    ExportErrorKind::FfmpegMissing,
                    format!(
                        "ffprobe not found at '{ffprobe_path}' — install ffmpeg or pass --ffmpeg"
                    ),
                )
            } else {
                err_probe(format!("spawn ffprobe: {e}"))
            }
        })?;
    if !output.status.success() {
        return Err(err_probe(format!(
            "ffprobe failed on {}: {}",
            video.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_ffprobe_json(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROBE_JSON: &str = r#"{ "streams": [
        { "codec_type": "video", "width": 1920, "height": 1080,
          "r_frame_rate": "60000/1001", "avg_frame_rate": "60000/1001",
          "side_data_list": [ { "rotation": -90 } ] },
        { "codec_type": "audio" } ],
      "format": { "duration": "63.400000" } }"#;

    #[test]
    fn parse_ffprobe_json_video_and_audio_streams_extracts_fields() {
        // Arrange: PROBE_JSON above

        // Act
        let p = parse_ffprobe_json(PROBE_JSON).unwrap();

        // Assert
        assert_eq!((p.width, p.height), (1920, 1080));
        assert!((p.fps - 59.94).abs() < 0.01);
        assert_eq!(p.rotation_deg, -90);
        assert!(p.has_audio);
        assert!((p.duration_s - 63.4).abs() < 1e-6);
    }

    #[test]
    fn parse_ffprobe_json_no_audio_no_rotation_defaults() {
        // Arrange
        let json = r#"{ "streams": [
            { "codec_type": "video", "width": 1280, "height": 720,
              "r_frame_rate": "30/1" } ],
          "format": { "duration": "5.0" } }"#;

        // Act
        let p = parse_ffprobe_json(json).unwrap();

        // Assert
        assert_eq!(p.rotation_deg, 0);
        assert!(!p.has_audio);
        assert!((p.fps - 30.0).abs() < 1e-9);
    }

    #[test]
    fn parse_ffprobe_json_without_video_stream_is_probe_error() {
        // Arrange
        let json = r#"{ "streams": [ { "codec_type": "audio" } ], "format": { "duration": "1" } }"#;

        // Act
        let err = parse_ffprobe_json(json).unwrap_err();

        // Assert
        assert_eq!(err.kind, ExportErrorKind::Probe);
    }
}
