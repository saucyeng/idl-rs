//! Export planning: output geometry, frame count, and the exact ffmpeg argv.
//! Everything here is pure and unit-tested; the process runs in `export`.

use std::path::{Path, PathBuf};

use crate::probe::VideoProbe;

/// One planned overlay export.
#[derive(Debug, Clone)]
pub struct ExportPlan {
    /// Source video path.
    pub video: PathBuf,
    /// Final output path (the driver writes `<output>.part` then renames).
    pub output: PathBuf,
    /// Source probe.
    pub probe: VideoProbe,
    /// Clip start in video seconds (`None` = from the beginning).
    pub start_s: Option<f64>,
    /// Clip duration in seconds (`None` = to the end).
    pub duration_s: Option<f64>,
    /// ffmpeg video encoder (default `libx264`).
    pub encoder: String,
    /// ffmpeg binary path or name.
    pub ffmpeg_path: String,
}

impl ExportPlan {
    /// Effective clip duration in seconds after `start`/`duration` clamping.
    pub fn effective_duration_s(&self) -> f64 {
        let start = self.start_s.unwrap_or(0.0).max(0.0);
        let remaining = (self.probe.duration_s - start).max(0.0);
        match self.duration_s {
            Some(d) => d.max(0.0).min(remaining),
            None => remaining,
        }
    }

    /// Output frame count: ceil(fps × effective duration).
    pub fn total_frames(&self) -> u64 {
        (self.probe.fps * self.effective_duration_s()).ceil() as u64
    }

    /// Overlay frame dimensions: display size after rotation (±90/±270 swap
    /// coded width/height).
    pub fn frame_dims(&self) -> (u32, u32) {
        match self.probe.rotation_deg.rem_euclid(180) {
            90 => (self.probe.height, self.probe.width),
            _ => (self.probe.width, self.probe.height),
        }
    }

    /// The ffmpeg argv (without the program name). Input 0 = source video
    /// (clipped via `-ss`/`-t`), input 1 = rawvideo RGBA overlay on stdin;
    /// overlay composited, audio stream-copied when present, CFR output.
    pub fn ffmpeg_args(&self, part_path: &Path) -> Vec<String> {
        let (w, h) = self.frame_dims();
        let fps = format!("{:.6}", self.probe.fps);
        let mut args: Vec<String> = vec!["-hide_banner".into(), "-y".into()];
        if let Some(start) = self.start_s {
            args.extend(["-ss".into(), format!("{start:.3}")]);
        }
        args.extend(["-i".into(), self.video.display().to_string()]);
        args.extend([
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgba".into(),
            "-s".into(),
            format!("{w}x{h}"),
            "-r".into(),
            fps.clone(),
            "-i".into(),
            "pipe:0".into(),
            "-filter_complex".into(),
            "[0:v][1:v]overlay=format=auto[out]".into(),
            "-map".into(),
            "[out]".into(),
        ]);
        if self.probe.has_audio {
            args.extend(["-map".into(), "0:a".into(), "-c:a".into(), "copy".into()]);
        }
        if let Some(dur) = self.duration_s {
            args.extend(["-t".into(), format!("{dur:.3}")]);
        }
        args.extend([
            "-r".into(),
            fps,
            "-vsync".into(),
            "cfr".into(),
            "-c:v".into(),
            self.encoder.clone(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-movflags".into(),
            "+faststart".into(),
            // The `.part` suffix defeats extension-based muxer inference.
            "-f".into(),
            "mp4".into(),
            part_path.display().to_string(),
        ]);
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(fps: f64, has_audio: bool, rotation_deg: i32) -> VideoProbe {
        VideoProbe {
            width: 1280,
            height: 720,
            fps,
            duration_s: 10.0,
            rotation_deg,
            has_audio,
        }
    }

    fn plan(start: Option<f64>, dur: Option<f64>, has_audio: bool) -> ExportPlan {
        ExportPlan {
            video: PathBuf::from("in.mp4"),
            output: PathBuf::from("out.mp4"),
            probe: probe(30.0, has_audio, 0),
            start_s: start,
            duration_s: dur,
            encoder: "libx264".into(),
            ffmpeg_path: "ffmpeg".into(),
        }
    }

    #[test]
    fn frame_dims_rotation_minus_90_swaps_width_height() {
        // Arrange
        let mut p = plan(None, None, false);
        p.probe = probe(30.0, false, -90);

        // Act + Assert
        assert_eq!(p.frame_dims(), (720, 1280));
        p.probe.rotation_deg = 180;
        assert_eq!(p.frame_dims(), (1280, 720));
    }

    #[test]
    fn total_frames_2997_fps_10s_is_300() {
        // Arrange
        let mut p = plan(None, None, false);
        p.probe = probe(29.97, false, 0);

        // Act + Assert — ceil(10 × 29.97) = 300.
        assert_eq!(p.total_frames(), 300);
    }

    #[test]
    fn effective_duration_clamps_to_remaining_source() {
        // Arrange — 10 s source, start at 8, ask for 5.
        let p = plan(Some(8.0), Some(5.0), false);

        // Act + Assert
        assert!((p.effective_duration_s() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ffmpeg_args_no_audio_no_clip_omits_ss_t_and_audio_map() {
        // Arrange
        let p = plan(None, None, false);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        assert!(!args.contains(&"-ss".to_string()));
        assert!(!args.contains(&"-t".to_string()));
        assert!(!args.contains(&"0:a".to_string()));
        assert_eq!(args.last().unwrap(), "out.mp4.part");
    }

    #[test]
    fn ffmpeg_args_clip_and_audio_produces_exact_argv() {
        // Arrange
        let p = plan(Some(10.0), Some(5.0), true);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — byte-exact so drift is caught.
        let expected: Vec<String> = [
            "-hide_banner",
            "-y",
            "-ss",
            "10.000",
            "-i",
            "in.mp4",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            "1280x720",
            "-r",
            "30.000000",
            "-i",
            "pipe:0",
            "-filter_complex",
            "[0:v][1:v]overlay=format=auto[out]",
            "-map",
            "[out]",
            "-map",
            "0:a",
            "-c:a",
            "copy",
            "-t",
            "5.000",
            "-r",
            "30.000000",
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            "out.mp4.part",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(args, expected);
    }
}
