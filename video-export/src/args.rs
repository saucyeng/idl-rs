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
    /// Extra counter-clockwise rotation applied to the video **on top of**
    /// any container rotation metadata, in degrees (0 / 90 / 180 / 270).
    ///
    /// For footage shot with the camera deliberately mounted rotated, where
    /// the container carries no rotation metadata to correct it. The overlay
    /// is composited *after* the rotation, so its text stays upright.
    pub rotate_ccw_deg: i32,
    /// ffmpeg hardware decoder to use for the source video (`cuda`, `qsv`,
    /// `d3d11va`, …), or `None` for software decode.
    ///
    /// Decoded frames are downloaded to system memory, so the overlay
    /// composite and rotation stay on the portable CPU filter path — this
    /// buys back the decode cost without a GPU-specific filter graph.
    /// Pair with a hardware `encoder` (e.g. `h264_nvenc`) to move both ends
    /// off the CPU.
    pub hwaccel: Option<String>,
    /// Constant-quality target (CRF for x264/x265, CQ for NVENC/QSV/AMF).
    /// `None` leaves the encoder's own default. Lower is better quality.
    pub quality: Option<u8>,
    /// Cap on CPU parallelism: bounds both the overlay render pool and
    /// ffmpeg's encoder threads. `None` uses every core.
    ///
    /// A full-resolution export otherwise saturates the machine — both halves
    /// of the pipeline are embarrassingly parallel, so leaving a core or two
    /// free is the difference between a usable desktop and a frozen one.
    pub threads: Option<usize>,
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
    /// coded width/height). Container rotation (applied by ffmpeg on decode)
    /// and [`Self::rotate_ccw_deg`] (applied by our filter) compose, so this
    /// is the size of the *final* frame the overlay is drawn on.
    pub fn frame_dims(&self) -> (u32, u32) {
        let (w, h) = match self.probe.rotation_deg.rem_euclid(180) {
            90 => (self.probe.height, self.probe.width),
            _ => (self.probe.width, self.probe.height),
        };
        match self.rotate_ccw_deg.rem_euclid(180) {
            90 => (h, w),
            _ => (w, h),
        }
    }

    /// ffmpeg filter chain implementing [`Self::rotate_ccw_deg`], or `None`
    /// when no extra rotation is requested. `transpose=2` is 90°
    /// counter-clockwise, `transpose=1` is 90° clockwise (= 270° CCW).
    fn rotation_filter(&self) -> Option<&'static str> {
        match self.rotate_ccw_deg.rem_euclid(360) {
            90 => Some("transpose=2"),
            180 => Some("hflip,vflip"),
            270 => Some("transpose=1"),
            _ => None,
        }
    }

    /// The ffmpeg argv (without the program name). Input 0 = source video
    /// (clipped via `-ss`/`-t`), input 1 = rawvideo RGBA overlay on stdin;
    /// overlay composited, audio stream-copied when present, CFR output.
    pub fn ffmpeg_args(&self, part_path: &Path) -> Vec<String> {
        let (w, h) = self.frame_dims();
        let fps = format!("{:.6}", self.probe.fps);
        let mut args: Vec<String> = vec!["-hide_banner".into(), "-y".into()];
        // Input options, all before `-i`: hwaccel selects the decoder, `-ss`
        // seeks. Only input 0 (the source video) is hardware-decoded; the
        // rawvideo overlay pipe is already uncompressed.
        if let Some(hw) = &self.hwaccel {
            args.extend(["-hwaccel".into(), hw.clone()]);
        }
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
            match self.rotation_filter() {
                Some(f) => format!("[0:v]{f}[vr];[vr][1:v]overlay=format=auto[out]"),
                None => "[0:v][1:v]overlay=format=auto[out]".into(),
            },
            "-map".into(),
            "[out]".into(),
        ]);
        if self.probe.has_audio {
            args.extend(["-map".into(), "0:a".into(), "-c:a".into(), "copy".into()]);
        }
        if let Some(dur) = self.duration_s {
            args.extend(["-t".into(), format!("{dur:.3}")]);
        }
        if let Some(n) = self.threads {
            args.extend(["-threads".into(), n.to_string()]);
        }
        args.extend([
            "-r".into(),
            fps,
            "-vsync".into(),
            "cfr".into(),
            "-c:v".into(),
            self.encoder.clone(),
        ]);
        args.extend(self.quality_args());
        args.extend([
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

    /// Constant-quality flags for the configured encoder.
    ///
    /// Without these each encoder falls back to its own default, and those
    /// defaults are wildly different: libx264 targets CRF 23 (visually fine),
    /// while NVENC targets a fixed ~2 Mbps bitrate — at 2028p that is heavy
    /// mush, and it made the GPU encoder look "2.6× faster" when it was really
    /// just producing a 16× smaller file. Pinning quality on both sides makes
    /// the encoders comparable, so choosing one is a speed decision rather
    /// than a silent quality cut.
    ///
    /// x264/x265 take `-crf`; NVENC needs `-rc vbr` + `-cq` with `-b:v 0` to
    /// leave bitrate uncapped (`-cq` alone is ignored under the default rate
    /// control). The two scales are both 0–51 and roughly comparable for H.264.
    /// Unrecognised encoders get nothing — their own defaults are less wrong
    /// than a flag they may reject.
    fn quality_args(&self) -> Vec<String> {
        let Some(q) = self.quality else {
            return Vec::new();
        };
        let e = self.encoder.as_str();
        if e.contains("nvenc") || e.contains("qsv") || e.contains("amf") {
            vec![
                "-rc".into(),
                "vbr".into(),
                "-cq".into(),
                q.to_string(),
                "-b:v".into(),
                "0".into(),
            ]
        } else if e.starts_with("libx26") || e.contains("libsvtav1") || e.contains("librav1e") {
            vec!["-crf".into(), q.to_string()]
        } else {
            Vec::new()
        }
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
            quality: None,
            ffmpeg_path: "ffmpeg".into(),
            rotate_ccw_deg: 0,
            hwaccel: None,
            threads: None,
        }
    }

    #[test]
    fn ffmpeg_args_hwaccel_precedes_the_source_input_only() {
        // Arrange — clipped, so `-ss` is present too.
        let mut p = plan(Some(4.0), None, false);
        p.hwaccel = Some("cuda".into());

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — `-hwaccel cuda` is an input option: before `-ss` and
        // before the source `-i`, and applied once (not to the pipe input).
        let hw = args.iter().position(|a| a == "-hwaccel").unwrap();
        let first_i = args.iter().position(|a| a == "-i").unwrap();
        assert_eq!(args[hw + 1], "cuda");
        assert!(hw < args.iter().position(|a| a == "-ss").unwrap());
        assert!(hw < first_i);
        assert_eq!(args.iter().filter(|a| *a == "-hwaccel").count(), 1);
    }

    #[test]
    fn ffmpeg_args_no_hwaccel_omits_the_flag() {
        // Arrange
        let p = plan(None, None, false);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        assert!(!args.contains(&"-hwaccel".to_string()));
    }

    #[test]
    fn frame_dims_user_rotation_90_swaps_the_display_dims() {
        // Arrange — 1280x720 source, no container rotation.
        let mut p = plan(None, None, false);
        p.rotate_ccw_deg = 90;

        // Act + Assert — the overlay canvas is the *final* portrait frame.
        assert_eq!(p.frame_dims(), (720, 1280));
    }

    #[test]
    fn frame_dims_user_rotation_composes_with_container_rotation() {
        // Arrange — container already reports 90 (ffmpeg auto-rotates on
        // decode → 720x1280), then the user rotates another 90 → back to
        // landscape.
        let mut p = plan(None, None, false);
        p.probe = probe(30.0, false, 90);
        p.rotate_ccw_deg = 90;

        // Act + Assert
        assert_eq!(p.frame_dims(), (1280, 720));
    }

    #[test]
    fn frame_dims_user_rotation_180_keeps_orientation() {
        // Arrange
        let mut p = plan(None, None, false);
        p.rotate_ccw_deg = 180;

        // Act + Assert
        assert_eq!(p.frame_dims(), (1280, 720));
    }

    #[test]
    fn ffmpeg_args_rotate_90_ccw_transposes_before_the_overlay() {
        // Arrange
        let mut p = plan(None, None, false);
        p.rotate_ccw_deg = 90;

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — transpose=2 is ffmpeg's 90° counter-clockwise; the
        // overlay composites onto the rotated frame so its text stays
        // upright, and the rawvideo input is sized to the rotated frame.
        let filter = &args[args.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        assert_eq!(filter, "[0:v]transpose=2[vr];[vr][1:v]overlay=format=auto[out]");
        let size = &args[args.iter().position(|a| a == "-s").unwrap() + 1];
        assert_eq!(size, "720x1280");
    }

    #[test]
    fn ffmpeg_args_rotate_270_ccw_uses_clockwise_transpose() {
        // Arrange — 270 CCW is the same rotation as 90 CW (transpose=1).
        let mut p = plan(None, None, false);
        p.rotate_ccw_deg = 270;

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        let filter = &args[args.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        assert_eq!(filter, "[0:v]transpose=1[vr];[vr][1:v]overlay=format=auto[out]");
    }

    #[test]
    fn ffmpeg_args_rotate_180_flips_both_axes() {
        // Arrange
        let mut p = plan(None, None, false);
        p.rotate_ccw_deg = 180;

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        let filter = &args[args.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        assert_eq!(filter, "[0:v]hflip,vflip[vr];[vr][1:v]overlay=format=auto[out]");
    }

    #[test]
    fn ffmpeg_args_rotate_zero_omits_the_transpose_filter() {
        // Arrange
        let p = plan(None, None, false);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — unrotated exports keep the original single-filter graph.
        let filter = &args[args.iter().position(|a| a == "-filter_complex").unwrap() + 1];
        assert_eq!(filter, "[0:v][1:v]overlay=format=auto[out]");
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

    #[test]
    fn quality_maps_to_crf_for_x264_and_cq_for_nvenc() {
        // Arrange — same quality target, two encoder families.
        let mut x264 = plan(None, None, false);
        x264.quality = Some(20);
        let mut nvenc = plan(None, None, false);
        nvenc.encoder = "h264_nvenc".into();
        nvenc.quality = Some(20);

        // Act
        let a = x264.ffmpeg_args(Path::new("out.mp4.part"));
        let b = nvenc.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — x264 takes -crf; NVENC needs -rc vbr and -b:v 0 alongside
        // -cq, or the default rate control ignores the quality target and
        // silently caps the bitrate instead.
        assert!(a.windows(2).any(|w| w == ["-crf", "20"]), "{a:?}");
        assert!(!a.iter().any(|x| x == "-cq"), "x264 must not get -cq: {a:?}");
        assert!(b.windows(2).any(|w| w == ["-cq", "20"]), "{b:?}");
        assert!(b.windows(2).any(|w| w == ["-rc", "vbr"]), "{b:?}");
        assert!(b.windows(2).any(|w| w == ["-b:v", "0"]), "{b:?}");
    }

    #[test]
    fn quality_none_emits_no_rate_flags() {
        // Arrange — quality unset leaves the encoder's own default.
        let p = plan(None, None, false);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        assert!(!args.iter().any(|a| a == "-crf" || a == "-cq" || a == "-b:v"), "{args:?}");
    }

    #[test]
    fn quality_is_skipped_for_an_unrecognised_encoder() {
        // Arrange — a codec whose rate-control flag we do not know.
        let mut p = plan(None, None, false);
        p.encoder = "mpeg4".into();
        p.quality = Some(20);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — emit nothing rather than a flag the encoder may reject.
        assert!(!args.iter().any(|a| a == "-crf" || a == "-cq"), "{args:?}");
    }

    #[test]
    fn ffmpeg_args_threads_caps_the_encoder_and_precedes_the_output() {
        // Arrange
        let mut p = plan(None, None, false);
        p.threads = Some(5);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert — an output option, so after the inputs and before the file.
        let t = args.iter().position(|a| a == "-threads").unwrap();
        assert_eq!(args[t + 1], "5");
        assert!(t > args.iter().position(|a| a == "-filter_complex").unwrap());
        assert!(t < args.len() - 1);
    }

    #[test]
    fn ffmpeg_args_no_thread_cap_omits_the_flag() {
        // Arrange
        let p = plan(None, None, false);

        // Act
        let args = p.ffmpeg_args(Path::new("out.mp4.part"));

        // Assert
        assert!(!args.contains(&"-threads".to_string()));
    }
}
