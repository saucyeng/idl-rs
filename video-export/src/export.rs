//! The export driver: spawn ffmpeg, pump rendered frames into its stdin in
//! order, rename `.part` → final on success. Frames render on a rayon pool
//! in ordered chunks (chunked `collect` preserves order and gives encode
//! backpressure). See docs/IDL0_SPEC.md §33.5.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::args::ExportPlan;

/// Frames rendered per parallel chunk before writing to the pipe. Bounds
/// memory (chunk × frame bytes) and keeps render ahead of encode without
/// running away.
const CHUNK_FRAMES: usize = 32;
/// Bytes of ffmpeg stderr tail kept for error reporting.
const STDERR_TAIL_BYTES: usize = 4096;

/// What went wrong during an export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportErrorKind {
    /// ffmpeg/ffprobe binary not found.
    FfmpegMissing,
    /// Source probing failed.
    Probe,
    /// Writing frames to ffmpeg's stdin failed.
    Pipe,
    /// ffmpeg exited non-zero (message carries the stderr tail).
    FfmpegFailed,
    /// The caller cancelled; partial output was removed.
    Cancelled,
    /// Filesystem operation failed.
    Io,
}

/// Export-driver error: unit-enum `kind` + human-readable `message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportError {
    pub kind: ExportErrorKind,
    pub message: String,
}

impl ExportError {
    pub fn new(kind: ExportErrorKind, message: impl Into<String>) -> Self {
        ExportError {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ExportError {}

/// Export progress: frames fed to the encoder vs total planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub frames_done: u64,
    pub frames_total: u64,
}

/// Run the export. `render(i)` must return the straight-RGBA bytes of output
/// frame `i` at `plan.frame_dims()` size; it runs on a rayon pool (must be
/// `Sync`). `progress` fires after each written chunk. Setting `cancel` kills
/// ffmpeg, removes the partial file, and returns `Cancelled`.
pub fn run_export<F>(
    plan: &ExportPlan,
    render: F,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<(), ExportError>
where
    F: Fn(u64) -> Vec<u8> + Sync,
{
    let part = plan.output.with_extension("mp4.part");
    let total = plan.total_frames();
    let (w, h) = plan.frame_dims();
    let frame_bytes = (w * h * 4) as usize;

    let mut child = Command::new(&plan.ffmpeg_path)
        .args(plan.ffmpeg_args(&part))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ExportError::new(
                    ExportErrorKind::FfmpegMissing,
                    format!(
                        "ffmpeg not found at '{}' — install ffmpeg or pass --ffmpeg",
                        plan.ffmpeg_path
                    ),
                )
            } else {
                ExportError::new(ExportErrorKind::Io, format!("spawn ffmpeg: {e}"))
            }
        })?;

    // Drain stderr on a thread, keeping only the tail for error reports.
    let stderr = child.stderr.take().expect("stderr piped");
    let stderr_tail = std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        let mut reader = std::io::BufReader::new(stderr);
        let _ = reader.read_to_end(&mut buf);
        let start = buf.len().saturating_sub(STDERR_TAIL_BYTES);
        String::from_utf8_lossy(&buf[start..]).into_owned()
    });

    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut done: u64 = 0;
    let mut failed: Option<ExportError> = None;

    'pump: for chunk_start in (0..total).step_by(CHUNK_FRAMES) {
        if cancel.load(Ordering::Relaxed) {
            failed = Some(ExportError::new(
                ExportErrorKind::Cancelled,
                "export cancelled",
            ));
            break 'pump;
        }
        let chunk_end = (chunk_start + CHUNK_FRAMES as u64).min(total);
        // Ordered parallel render: collect preserves input order.
        let frames: Vec<Vec<u8>> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(&render)
            .collect();
        for frame in frames {
            debug_assert_eq!(frame.len(), frame_bytes, "render(i) must match frame_dims");
            if let Err(e) = stdin.write_all(&frame) {
                failed = Some(ExportError::new(
                    ExportErrorKind::Pipe,
                    format!("write frame to ffmpeg: {e}"),
                ));
                break 'pump;
            }
            done += 1;
        }
        progress(Progress {
            frames_done: done,
            frames_total: total,
        });
    }

    if let Some(err) = failed {
        let _ = child.kill();
        let _ = child.wait();
        let _ = stderr_tail.join();
        let _ = std::fs::remove_file(&part);
        return Err(err);
    }

    drop(stdin); // EOF → ffmpeg finalizes the file.
    let status = child
        .wait()
        .map_err(|e| ExportError::new(ExportErrorKind::Io, format!("wait ffmpeg: {e}")))?;
    let tail = stderr_tail.join().unwrap_or_default();
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(ExportError::new(
            ExportErrorKind::FfmpegFailed,
            format!("ffmpeg exited with {status}: {}", tail.trim()),
        ));
    }

    std::fs::rename(&part, &plan.output).map_err(|e| {
        ExportError::new(
            ExportErrorKind::Io,
            format!(
                "rename {} -> {}: {e}",
                part.display(),
                plan.output.display()
            ),
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{probe, VideoProbe};
    use std::path::PathBuf;

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok()
    }

    #[test]
    fn run_export_missing_binary_reports_ffmpeg_missing() {
        // Arrange
        let plan = ExportPlan {
            video: PathBuf::from("in.mp4"),
            output: PathBuf::from("out.mp4"),
            probe: VideoProbe {
                width: 64,
                height: 64,
                fps: 30.0,
                duration_s: 1.0,
                rotation_deg: 0,
                has_audio: false,
            },
            start_s: None,
            duration_s: None,
            encoder: "libx264".into(),
            ffmpeg_path: "definitely-not-a-real-ffmpeg-binary".into(),
        };

        // Act
        let err = run_export(
            &plan,
            |_| vec![0u8; 64 * 64 * 4],
            &mut |_| {},
            &AtomicBool::new(false),
        )
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, ExportErrorKind::FfmpegMissing);
        assert!(err.message.contains("--ffmpeg"));
    }

    /// End-to-end smoke: auto-skips when ffmpeg is absent (no recordings /
    /// tooling on CI yet — design-doc testing note).
    #[test]
    fn run_export_synthetic_input_end_to_end_produces_playable_mp4() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }

        // Arrange — generate a 1 s 64x64 30 fps red test input.
        let dir = std::env::temp_dir().join("idlrs_video_export_e2e");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.mp4");
        let output = dir.join("out.mp4");
        let _ = std::fs::remove_file(&output);
        let gen = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=64x64:rate=30:duration=1",
            ])
            .arg(&input)
            .output()
            .unwrap();
        assert!(gen.status.success(), "test-input generation failed");

        let plan = ExportPlan {
            video: input.clone(),
            output: output.clone(),
            probe: probe(&input, "ffprobe").unwrap(),
            start_s: None,
            duration_s: None,
            encoder: "libx264".into(),
            ffmpeg_path: "ffmpeg".into(),
        };
        let (w, h) = plan.frame_dims();

        // Act — moving white square on a transparent overlay.
        let mut last = Progress {
            frames_done: 0,
            frames_total: 0,
        };
        run_export(
            &plan,
            |i| {
                let mut frame = vec![0u8; (w * h * 4) as usize];
                let x0 = (i as u32 * 2) % (w - 8);
                for y in 8..16u32 {
                    for x in x0..x0 + 8 {
                        let idx = ((y * w + x) * 4) as usize;
                        frame[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
                    }
                }
                frame
            },
            &mut |p| last = p,
            &AtomicBool::new(false),
        )
        .unwrap();

        // Assert
        assert!(output.exists(), "output written");
        assert!(
            !plan.output.with_extension("mp4.part").exists(),
            ".part renamed away"
        );
        assert_eq!(last.frames_done, last.frames_total);
        let re = probe(&output, "ffprobe").unwrap();
        assert_eq!((re.width, re.height), (64, 64));
    }
}
