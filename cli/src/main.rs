//! `idl-rs` — command-line front end for the idl-rs engine.
//!
//! `info` and `channels` inspect a log; `export` writes its channel set to
//! CSV (long/tidy) or JSON. All commands read through `SessionHandle`, so the
//! synthesized `Time`/`Distance` channels are visible everywhere.
//!
//! Every command speaks the JSON envelope (see [`envelope`]): structured
//! commands (`info`, `channels`, `laps`, `visits`) emit a success envelope on
//! stdout under `--format json` (text stays the default) and an error envelope
//! on stdout on failure; bulk commands (`export`, `math`, `fit`, `recover`,
//! `scan`) write their raw artifact on success and an error envelope to stderr
//! on failure. See `docs/IDL0_SPEC.md` (CLI section) for the contract.

mod envelope;
mod recover;
mod table_cmd;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Value};

use idl_rs::export::{self, ExportFormat, ExportOptions, FitLap, FitOptions, FitSport};
use idl_rs::fft::{Averaging, Detrend, FftWindow, Scaling};
use idl_rs::laps::detect_laps;
use idl_rs::laps::model::Lap;
use idl_rs::math::MathLapContext;
use idl_rs::overlay::sample::SampleContext;
use idl_rs::session::handle::{ChannelMeta, SessionHandle, SessionMeta};
use idl_rs::session::Channel;
use idl_rs::track_artifact::{self, Track};
use idl_rs::tracks::{detect_visits, VisitParams, VisitWindow};
use idl_rs::video::render::render_overlay_frame;
use idl_rs::video::{gpmf, mp4box, sync::estimate_sync, VideoError, VideoErrorKind};
use idl_rs::workbook::{self, ApplyReport};

use envelope::{emit_bulk, emit_structured, CliError, ErrorKind, Structured, Warning};
use idl_rs_video_export::{run_export, ExportError, ExportErrorKind, ExportPlan, Progress};

#[derive(Parser)]
#[command(
    name = "idl-rs",
    version,
    about = "Read, inspect, and export IDL0 (.idl0) data-acquisition logs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print session metadata (IDs, start time, checksum, channel count).
    Info {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// List every channel: name, sample rate, sample count, synthesized flag.
    Channels {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Export the channel set to CSV (long/tidy) or JSON.
    Export {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Output file. Format inferred from extension (.csv/.json) unless --format is given.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format; overrides extension inference. Defaults to CSV on stdout.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        /// Channel id to include (repeatable). Default: all channels.
        #[arg(long = "channel")]
        channels: Vec<String>,
    },
    /// Evaluate a workbook's math channels against a session and export the
    /// derived results (CSV/JSON). Lap-aware channels are skipped until lap
    /// detection lands (Phase 4).
    Math {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Path to an `.idl0wb` workbook file.
        #[arg(long)]
        workbook: PathBuf,
        /// Output file. Format inferred from extension (.csv/.json) unless --format is given.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format; overrides extension inference. Defaults to CSV on stdout.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        /// Also export base + synthesized channels (default: derived only).
        #[arg(long)]
        include_base: bool,
        /// Channel id to include (repeatable). Default: all in the result set.
        #[arg(long = "channel")]
        channels: Vec<String>,
    },
    /// Detect laps for a track over a session and print the lap table.
    Laps {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Path to a `.idl0t` track artifact.
        #[arg(long)]
        track: PathBuf,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Inspect a video container or estimate the video-session sync offset.
    #[command(subcommand)]
    Video(VideoCmd),
    /// Render a workbook overlay layout onto a video (sidecar ffmpeg).
    Overlay {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Path to an `.mp4`/`.mov` video file.
        #[arg(long)]
        video: PathBuf,
        /// Path to an `.idl0wb` workbook holding the overlay layout(s).
        #[arg(long)]
        workbook: PathBuf,
        /// Layout name; may be omitted when the workbook has exactly one.
        #[arg(long)]
        layout: Option<String>,
        /// `.idl0t` track for the lap panel (omitted: lap elements render
        /// no-data).
        #[arg(long)]
        track: Option<PathBuf>,
        /// Manual sync offset in seconds (skips auto-sync).
        #[arg(long)]
        offset: Option<f64>,
        /// Clip start in video seconds.
        #[arg(long)]
        start: Option<f64>,
        /// Clip duration in seconds.
        #[arg(long)]
        duration: Option<f64>,
        /// Output path; default: `<video stem>_overlay.mp4` beside the video.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// ffmpeg video encoder.
        #[arg(long, default_value = "libx264")]
        encoder: String,
        /// Path to the ffmpeg binary (ffprobe is resolved beside it).
        #[arg(long, default_value = "ffmpeg")]
        ffmpeg: String,
    },
    /// Compute the Welch spectrum of one channel (optionally windowed by
    /// `--from` / `--to`) and emit frequency+magnitude pairs. Wraps
    /// `welch_channel` / `welch_channel_windowed` in the engine; the FFT input
    /// never crosses FFI — only the `WelchResult` does.
    Fft {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Channel id to transform (e.g. `IMU0_AccelZ`).
        #[arg(long)]
        channel: String,
        /// Window start in session-relative seconds (default: session start).
        #[arg(long)]
        from: Option<f64>,
        /// Window end in session-relative seconds (default: session end).
        #[arg(long)]
        to: Option<f64>,
        /// Window function applied to each segment before the FFT.
        #[arg(long, value_enum, default_value_t = WindowArg::Hann)]
        window: WindowArg,
        /// Welch segment length in samples (0 = one full-record segment).
        #[arg(long, default_value_t = 0)]
        nperseg: usize,
        /// Welch overlap in samples between consecutive segments.
        #[arg(long, default_value_t = 0)]
        noverlap: usize,
        /// Per-segment trend removal before windowing.
        #[arg(long, value_enum, default_value_t = DetrendArg::Mean)]
        detrend: DetrendArg,
        /// Cross-segment power averaging strategy.
        #[arg(long, value_enum, default_value_t = AveragingArg::Mean)]
        averaging: AveragingArg,
        /// Output units of the spectrum.
        #[arg(long, value_enum, default_value_t = ScalingArg::Magnitude)]
        scaling: ScalingArg,
        /// Output format: `text` (default, one `freq_hz\tvalue` line per bin)
        /// or `json` (success envelope with `data.freqs_hz` / `data.values`).
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Compute the time×frequency spectrogram of one channel and emit the
    /// flat power matrix. Slices by `--from` / `--to` (required for a useful
    /// heatmap — defaults to the whole session when omitted). Wraps
    /// `spectrogram_channel` in the engine; samples never cross FFI.
    Spectrogram {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Channel id to transform (e.g. `IMU0_AccelZ`).
        #[arg(long)]
        channel: String,
        /// Window start in session-relative seconds (default: session start).
        #[arg(long)]
        from: Option<f64>,
        /// Window end in session-relative seconds (default: session end).
        #[arg(long)]
        to: Option<f64>,
        /// Window function applied to each segment before the FFT.
        #[arg(long, value_enum, default_value_t = WindowArg::Hann)]
        window: WindowArg,
        /// Welch segment length in samples (0 = one full-record segment).
        #[arg(long, default_value_t = 0)]
        nperseg: usize,
        /// Welch overlap in samples between consecutive segments.
        #[arg(long, default_value_t = 0)]
        noverlap: usize,
        /// Per-segment trend removal before windowing.
        #[arg(long, value_enum, default_value_t = DetrendArg::Mean)]
        detrend: DetrendArg,
        /// Output units of each time×frequency cell.
        #[arg(long, value_enum, default_value_t = ScalingArg::Density)]
        scaling: ScalingArg,
        /// Output format: `text` (default, prints dimensions) or `json`
        /// (success envelope with `data.freqs_hz`, `data.times_secs`,
        /// `data.power`, `data.n_times`, `data.n_freqs`).
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Report which tracks a session visited, and when.
    Visits {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Path to a `.idl0t` track artifact (repeatable).
        #[arg(long = "track", required = true)]
        tracks: Vec<PathBuf>,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Convert a session to a Garmin FIT activity file (GPS + speed + altitude
    /// + heart rate) for Strava / Garmin Connect upload.
    Fit {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Output `.fit` file. Defaults to the input path with a `.fit` extension.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Activity sport classification.
        #[arg(long, value_enum, default_value_t = SportArg::Cycling)]
        sport: SportArg,
        /// Optional `.idl0t` track artifact; when given, lap splits are detected
        /// and written as FIT lap messages.
        #[arg(long)]
        track: Option<PathBuf>,
    },
    /// Recover a session from a raw device or image whose filesystem metadata is
    /// truncated/orphaned (e.g. after a power loss). Reads the source READ-ONLY
    /// and reconstructs the `.idl0` by walking the record stream directly.
    Recover {
        /// Source to scan, READ-ONLY: a raw device (e.g. `\\.\PhysicalDrive1`),
        /// a disk image, or an `.idl0` file.
        device: PathBuf,
        /// Output `.idl0` path. Write this to a DIFFERENT drive than the source.
        #[arg(short, long)]
        output: PathBuf,
        /// Only accept a header whose 32-char hex session UUID matches. Strongly
        /// recommended — it makes a false-positive match effectively impossible.
        #[arg(long)]
        session_id: Option<String>,
        /// Stop scanning for the header after this many bytes (default 16 GiB).
        #[arg(long)]
        scan_limit: Option<u64>,
        /// Maximum bytes to recover from the header onward (default 512 MiB).
        #[arg(long)]
        window: Option<usize>,
        /// Byte offset to start scanning from (skips earlier regions, e.g. a
        /// large FAT). Must be sector-aligned; defaults to 0.
        #[arg(long)]
        from: Option<u64>,
    },
    /// Sweep a whole raw device/image (READ-ONLY) for every `.idl0` session and
    /// list them; with --out-dir, also write each recovered session out. Use
    /// this to salvage all logs from a card after a logging failure.
    Scan {
        /// Source to sweep, READ-ONLY: a raw device (e.g. `\\.\PhysicalDrive1`),
        /// a disk image, or an `.idl0` file.
        device: PathBuf,
        /// Directory to write each recovered `<session>_<offset>.idl0` into.
        /// Must be on a DIFFERENT drive than the source.
        #[arg(short, long)]
        out_dir: Option<PathBuf>,
        /// Stop scanning after this many bytes (default: the whole device).
        #[arg(long)]
        scan_limit: Option<u64>,
    },
    /// Evaluate, list, or validate a workbook's tables (the `table` group).
    Table {
        #[command(subcommand)]
        action: table_cmd::TableAction,
    },
}

/// Output format for the structured inspect commands (`info`, `channels`,
/// `laps`, `visits`). `text` (default) keeps the human table; `json` emits the
/// success envelope.
#[derive(Clone, Copy, ValueEnum)]
enum OutFormat {
    Text,
    Json,
}

/// CLI mirror of [`ExportFormat`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum FormatArg {
    Csv,
    Json,
}

impl From<FormatArg> for ExportFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Csv => ExportFormat::Csv,
            FormatArg::Json => ExportFormat::Json,
        }
    }
}

/// CLI mirror of [`FitSport`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum SportArg {
    Cycling,
    Motorcycling,
    Running,
    Generic,
}

impl From<SportArg> for FitSport {
    fn from(s: SportArg) -> Self {
        match s {
            SportArg::Cycling => FitSport::Cycling,
            SportArg::Motorcycling => FitSport::Motorcycling,
            SportArg::Running => FitSport::Running,
            SportArg::Generic => FitSport::Generic,
        }
    }
}

/// CLI mirror of [`FftWindow`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum WindowArg {
    /// Hann window — good general-purpose choice, low leakage.
    Hann,
    /// Hamming window — slightly higher sidelobe than Hann.
    Hamming,
    /// Rectangular (no weighting) — maximum frequency resolution.
    Rect,
}

impl WindowArg {
    fn to_engine(self) -> FftWindow {
        match self {
            WindowArg::Hann => FftWindow::Hann,
            WindowArg::Hamming => FftWindow::Hamming,
            WindowArg::Rect => FftWindow::Rectangular,
        }
    }
}

/// CLI mirror of [`Detrend`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum DetrendArg {
    /// Leave the segment unchanged — bin 0 reflects the segment mean.
    None,
    /// Subtract the segment mean (constant detrend).
    Mean,
    /// Subtract a least-squares straight-line fit (removes mean + drift).
    Linear,
}

impl DetrendArg {
    fn to_engine(self) -> Detrend {
        match self {
            DetrendArg::None => Detrend::None,
            DetrendArg::Mean => Detrend::Mean,
            DetrendArg::Linear => Detrend::Linear,
        }
    }
}

/// CLI mirror of [`Averaging`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum AveragingArg {
    /// Arithmetic mean — standard Welch, lowest variance for clean data.
    Mean,
    /// Per-bin median — robust to transient spikes (impacts, chain slap).
    Median,
}

impl AveragingArg {
    fn to_engine(self) -> Averaging {
        match self {
            AveragingArg::Mean => Averaging::Mean,
            AveragingArg::Median => Averaging::Median,
        }
    }
}

/// CLI mirror of [`Scaling`] for clap's value parsing.
#[derive(Clone, Copy, ValueEnum)]
enum ScalingArg {
    /// RMS magnitude in input units (sqrt of mean power).
    Magnitude,
    /// Power spectral density in input-units² per Hz (window-normalised).
    Density,
}

impl ScalingArg {
    fn to_engine(self) -> Scaling {
        match self {
            ScalingArg::Magnitude => Scaling::Magnitude,
            ScalingArg::Density => Scaling::Density,
        }
    }
}

/// Dispatch each subcommand through its envelope wrapper. clap handles
/// pre-dispatch usage errors itself (exit 2, native stderr message);
/// everything past parse goes through the JSON envelope.

/// `video` subcommands (SPEC 33.6).
#[derive(Subcommand)]
enum VideoCmd {
    /// Container facts + GPMF presence (pure-Rust walker; no ffprobe needed).
    Probe {
        /// Path to an `.mp4`/`.mov` video file.
        #[arg(long)]
        video: PathBuf,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Estimate the sync offset (GPMF UTC anchor, else creation time).
    Sync {
        /// Path to an `.idl0` log file.
        file: PathBuf,
        /// Path to an `.mp4`/`.mov` video file.
        #[arg(long)]
        video: PathBuf,
        /// Output format: text (default) or json.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { file, format } => emit_structured("info", cmd_info(&file, format)),
        Command::Channels { file, format } => {
            emit_structured("channels", cmd_channels(&file, format))
        }
        Command::Laps {
            file,
            track,
            format,
        } => emit_structured("laps", cmd_laps(&file, &track, format)),
        Command::Video(VideoCmd::Probe { video, format }) => {
            emit_structured("video probe", cmd_video_probe(&video, format))
        }
        Command::Video(VideoCmd::Sync {
            file,
            video,
            format,
        }) => emit_structured("video sync", cmd_video_sync(&file, &video, format)),
        Command::Overlay {
            file,
            video,
            workbook,
            layout,
            track,
            offset,
            start,
            duration,
            output,
            encoder,
            ffmpeg,
        } => emit_bulk(
            "overlay",
            cmd_overlay(
                &file,
                &video,
                &workbook,
                layout.as_deref(),
                track.as_deref(),
                offset,
                start,
                duration,
                output,
                encoder,
                ffmpeg,
            ),
        ),
        Command::Fft {
            file,
            channel,
            from,
            to,
            window,
            nperseg,
            noverlap,
            detrend,
            averaging,
            scaling,
            format,
        } => emit_structured(
            "fft",
            cmd_fft(
                &file, &channel, from, to, window, nperseg, noverlap, detrend, averaging, scaling,
                format,
            ),
        ),
        Command::Spectrogram {
            file,
            channel,
            from,
            to,
            window,
            nperseg,
            noverlap,
            detrend,
            scaling,
            format,
        } => emit_structured(
            "spectrogram",
            cmd_spectrogram(
                &file, &channel, from, to, window, nperseg, noverlap, detrend, scaling, format,
            ),
        ),
        Command::Visits {
            file,
            tracks,
            format,
        } => emit_structured("visits", cmd_visits(&file, &tracks, format)),
        Command::Export {
            file,
            output,
            format,
            channels,
        } => emit_bulk(
            "export",
            cmd_export(&file, output.as_deref(), format, channels),
        ),
        Command::Math {
            file,
            workbook,
            output,
            format,
            include_base,
            channels,
        } => emit_bulk(
            "math",
            cmd_math(
                &file,
                &workbook,
                output.as_deref(),
                format,
                include_base,
                channels,
            ),
        ),
        Command::Fit {
            file,
            output,
            sport,
            track,
        } => emit_bulk("fit", cmd_fit(&file, output, sport, track.as_deref())),
        Command::Recover {
            device,
            output,
            session_id,
            scan_limit,
            window,
            from,
        } => emit_bulk(
            "recover",
            recover::run(
                &device,
                &output,
                session_id.as_deref(),
                scan_limit.unwrap_or(recover::DEFAULT_SCAN_LIMIT),
                window.unwrap_or(recover::DEFAULT_WINDOW),
                from.unwrap_or(0),
            ),
        ),
        Command::Scan {
            device,
            out_dir,
            scan_limit,
        } => emit_bulk(
            "scan",
            recover::scan_all(&device, out_dir.as_deref(), scan_limit.unwrap_or(u64::MAX)),
        ),
        Command::Table { action } => table_cmd::run(action),
    }
}

// ---------------------------------------------------------------------------
// Structured commands — return a `data` payload (JSON mode) or render text.
// ---------------------------------------------------------------------------

/// `info` — session metadata (§12 shape). Truncation surfaces in `warnings`.
fn cmd_info(file: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let meta = handle.metadata();
    match format {
        OutFormat::Text => {
            print_info(&meta);
            warn_if_truncated(&meta);
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data: json!({
                "session_id": meta.session_id,
                "device_id": meta.device_id,
                "timestamp_utc_ms": meta.timestamp_utc_ms,
                "config_checksum": meta.config_checksum,
                "channel_count": meta.channel_count,
                "duration_ms": meta.duration_ms,
            }),
            warnings: truncation_warnings(&meta),
        }),
    }
}

/// `channels` — per-channel metadata (§12 shape).
fn cmd_channels(file: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let channels = handle.channels();
    let meta = handle.metadata();
    match format {
        OutFormat::Text => {
            print_channels(&channels);
            warn_if_truncated(&meta);
            Ok(Structured::Text)
        }
        OutFormat::Json => {
            let rows: Vec<Value> = channels
                .iter()
                .map(|c| {
                    json!({
                        "channel_id": c.channel_id,
                        "sample_rate_hz": c.sample_rate_hz,
                        "length": c.length,
                        "synthesized": c.synthesized,
                    })
                })
                .collect();
            Ok(Structured::Json {
                data: json!({ "channels": rows }),
                warnings: truncation_warnings(&meta),
            })
        }
    }
}

/// `laps` — detected laps for a track (§12 shape, `Lap` serde form).
fn cmd_laps(file: &Path, track: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let t = track_artifact::read_track(track)?;
    let timing = t
        .timing
        .as_ref()
        .ok_or_else(|| no_timing_error(&t.name, track))?;
    let laps = detect_laps(&handle, timing, &t.sector_gates, &t.neutral_zones, None);
    let meta = handle.metadata();
    match format {
        OutFormat::Text => {
            print_laps(&laps);
            warn_if_truncated(&meta);
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data: json!({ "laps": laps }),
            warnings: truncation_warnings(&meta),
        }),
    }
}

/// `visits` — track-visit windows over a session (§12 shape).
fn cmd_visits(file: &Path, tracks: &[PathBuf], format: OutFormat) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let loaded: Vec<Track> = tracks
        .iter()
        .map(|p| track_artifact::read_track(p).map_err(CliError::from))
        .collect::<Result<_, _>>()?;
    let refs: Vec<_> = loaded.iter().map(|t| t.track_ref()).collect();
    let windows = detect_visits(&handle, &refs, VisitParams::default());
    // Join window.track_id → display name (later artifact wins a shared id).
    let name_of = |id: &str| -> String {
        loaded
            .iter()
            .rev()
            .find(|t| t.id == id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let meta = handle.metadata();
    match format {
        OutFormat::Text => {
            print_visits(&windows, &name_of);
            warn_if_truncated(&meta);
            Ok(Structured::Text)
        }
        OutFormat::Json => {
            let rows: Vec<Value> = windows
                .iter()
                .map(|w| {
                    json!({
                        "track_id": w.track_id,
                        "name": name_of(&w.track_id),
                        "start_ms": w.start_timestamp_ms,
                        "end_ms": w.end_timestamp_ms,
                        "duration_ms": w.end_timestamp_ms - w.start_timestamp_ms,
                    })
                })
                .collect();
            Ok(Structured::Json {
                data: json!({ "visits": rows }),
                warnings: truncation_warnings(&meta),
            })
        }
    }
}

/// Pure core for `fft` — resolves the window, calls `welch_channel` or
/// `welch_channel_windowed`, and returns the JSON payload. Kept separate from
/// `cmd_fft` so it can be called directly in tests without file I/O.
///
/// Returns `{ channel, freqs_hz, values }`.
/// `nperseg = 0` → one full-record segment.
///
/// `from`/`to` both `None` → whole-session path (`welch_channel`).
/// Either set → windowed path (`welch_channel_windowed`) with the missing
/// bound filled from `[0, duration_s]`.
#[allow(clippy::too_many_arguments)]
fn fft_json_data(
    handle: &SessionHandle,
    channel: &str,
    from: Option<f64>,
    to: Option<f64>,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    averaging: Averaging,
    scaling: Scaling,
) -> Result<Value, CliError> {
    let available: Vec<String> = handle
        .channels()
        .into_iter()
        .map(|c| c.channel_id)
        .collect();
    if !available.iter().any(|c| c == channel) {
        return Err(CliError::unknown_channel(channel, &available));
    }
    let r = match (from, to) {
        (None, None) => handle.welch_channel(
            channel, window, nperseg, noverlap, detrend, averaging, scaling,
        ),
        (f, t) => {
            let dur = handle.metadata().duration_ms as f64 / 1000.0;
            handle.welch_channel_windowed(
                channel,
                f.unwrap_or(0.0),
                t.unwrap_or(dur),
                window,
                nperseg,
                noverlap,
                detrend,
                averaging,
                scaling,
            )
        }
    };
    Ok(json!({ "channel": channel, "freqs_hz": r.freqs_hz, "values": r.values }))
}

/// `fft` — Welch spectrum of one channel.
///
/// Text mode: one `freq_hz\tvalue` line per bin (Hz and spectral value).
/// JSON mode: success envelope with `data = { channel, freqs_hz, values }`.
#[allow(clippy::too_many_arguments)]
fn cmd_fft(
    file: &Path,
    channel: &str,
    from: Option<f64>,
    to: Option<f64>,
    window: WindowArg,
    nperseg: usize,
    noverlap: usize,
    detrend: DetrendArg,
    averaging: AveragingArg,
    scaling: ScalingArg,
    format: OutFormat,
) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let data = fft_json_data(
        &handle,
        channel,
        from,
        to,
        window.to_engine(),
        nperseg,
        noverlap,
        detrend.to_engine(),
        averaging.to_engine(),
        scaling.to_engine(),
    )?;
    match format {
        OutFormat::Text => {
            // One "freq_hz\tvalue" line per bin.
            let empty: Vec<serde_json::Value> = Vec::new();
            let freqs = data["freqs_hz"].as_array().unwrap_or(&empty);
            let values = data["values"].as_array().unwrap_or(&empty);
            for (f, v) in freqs.iter().zip(values.iter()) {
                println!("{f}\t{v}");
            }
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data,
            warnings: truncation_warnings(&handle.metadata()),
        }),
    }
}

/// Pure core for `spectrogram` — resolves the window, calls
/// `spectrogram_channel`, and returns the JSON payload. Kept separate from
/// `cmd_spectrogram` so it can be called directly in tests without file I/O.
///
/// Returns `{ channel, freqs_hz, times_secs, power, n_times, n_freqs }`.
///
/// When `from`/`to` are both `None`, passes `[0, duration_s]` — there is no
/// whole-session spectrogram accessor, and the windowed one is identical for
/// the full span.
#[allow(clippy::too_many_arguments)]
fn spectrogram_json_data(
    handle: &SessionHandle,
    channel: &str,
    from: Option<f64>,
    to: Option<f64>,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    scaling: Scaling,
) -> Result<Value, CliError> {
    let available: Vec<String> = handle
        .channels()
        .into_iter()
        .map(|c| c.channel_id)
        .collect();
    if !available.iter().any(|c| c == channel) {
        return Err(CliError::unknown_channel(channel, &available));
    }
    let dur = handle.metadata().duration_ms as f64 / 1000.0;
    let t0 = from.unwrap_or(0.0);
    let t1 = to.unwrap_or(dur);
    let s =
        handle.spectrogram_channel(channel, t0, t1, window, nperseg, noverlap, detrend, scaling);
    Ok(json!({
        "channel": channel,
        "freqs_hz": s.freqs_hz,
        "times_secs": s.times_secs,
        "power": s.power,
        "n_times": s.n_times,
        "n_freqs": s.n_freqs,
    }))
}

/// `spectrogram` — time×frequency power matrix of one channel.
///
/// Text mode: prints the dimensions (`n_times`, `n_freqs`). JSON mode: success
/// envelope with `data = { channel, freqs_hz, times_secs, power, n_times, n_freqs }`.
#[allow(clippy::too_many_arguments)]
fn cmd_spectrogram(
    file: &Path,
    channel: &str,
    from: Option<f64>,
    to: Option<f64>,
    window: WindowArg,
    nperseg: usize,
    noverlap: usize,
    detrend: DetrendArg,
    scaling: ScalingArg,
    format: OutFormat,
) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let data = spectrogram_json_data(
        &handle,
        channel,
        from,
        to,
        window.to_engine(),
        nperseg,
        noverlap,
        detrend.to_engine(),
        scaling.to_engine(),
    )?;
    match format {
        OutFormat::Text => {
            let n_times = data["n_times"].as_u64().unwrap_or(0);
            let n_freqs = data["n_freqs"].as_u64().unwrap_or(0);
            println!("n_times\t{n_times}");
            println!("n_freqs\t{n_freqs}");
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data,
            warnings: truncation_warnings(&handle.metadata()),
        }),
    }
}

// ---------------------------------------------------------------------------
// Bulk commands — write the raw artifact, envelope only on failure.
// ---------------------------------------------------------------------------

/// `export` — write the channel set to CSV/JSON (stdout or `-o`).
fn cmd_export(
    file: &Path,
    output: Option<&Path>,
    format: Option<FormatArg>,
    channels: Vec<String>,
) -> Result<(), CliError> {
    let handle = load(file)?;
    let fmt = resolve_format(format, output)?;
    let options = ExportOptions { channels };
    export_to_sink(&handle, output, fmt, &options)?;
    warn_if_truncated(&handle.metadata());
    Ok(())
}

/// `math` — evaluate a workbook's math channels and export the results.
fn cmd_math(
    file: &Path,
    wb_path: &Path,
    output: Option<&Path>,
    format: Option<FormatArg>,
    include_base: bool,
    channels: Vec<String>,
) -> Result<(), CliError> {
    let handle = load(file)?;
    let wb = workbook::read_workbook(wb_path)?;
    let report = workbook::apply_workbook(&handle, &wb, &MathLapContext::empty());
    report_outcomes(&report);
    let ok_count = report.results.iter().filter(|r| r.error.is_none()).count();
    if ok_count == 0 {
        // Every channel failed to evaluate. Per-channel detail stayed on stderr
        // (§6 known limitation: bulk commands have no success envelope).
        return Err(CliError::new(
            ErrorKind::Eval,
            "no math channels evaluated (see per-channel messages above)",
        ));
    }
    let meta = handle.metadata();
    let syn: Vec<String> = if include_base {
        handle.synthesized_channel_ids().to_vec()
    } else {
        Vec::new()
    };
    let export_channels = assemble_export_channels(report, &handle, include_base);
    let fmt = resolve_format(format, output)?;
    let options = ExportOptions { channels };
    write_math_sink(&meta, &syn, &export_channels, output, fmt, &options)?;
    warn_if_truncated(&handle.metadata());
    Ok(())
}

/// `fit` — convert a session to a Garmin FIT activity file.
fn cmd_fit(
    file: &Path,
    output: Option<PathBuf>,
    sport: SportArg,
    track: Option<&Path>,
) -> Result<(), CliError> {
    let handle = load(file)?;
    let laps = match track {
        Some(track_path) => {
            let t = track_artifact::read_track(track_path)?;
            let timing = t
                .timing
                .as_ref()
                .ok_or_else(|| no_timing_error(&t.name, track_path))?;
            let detected = detect_laps(&handle, timing, &t.sector_gates, &t.neutral_zones, None);
            Some(
                detected
                    .iter()
                    .map(|l| FitLap {
                        start_ms: l.start_ms,
                        end_ms: l.end_ms,
                        elapsed_ms: l.lap_time_ms,
                    })
                    .collect(),
            )
        }
        None => None,
    };
    let out_path = output.unwrap_or_else(|| default_fit_output(file));
    let opts = FitOptions {
        sport: sport.into(),
        laps,
    };
    let f = File::create(&out_path)
        .map_err(|e| CliError::io(format!("cannot write {}: {e}", out_path.display())))?;
    let mut wtr = BufWriter::new(f);
    export::write_fit(&handle, &opts, &mut wtr)?;
    wtr.flush()
        .map_err(|e| CliError::io(format!("cannot write {}: {e}", out_path.display())))?;
    eprintln!("wrote {}", out_path.display());
    warn_if_truncated(&handle.metadata());
    Ok(())
}

// ---------------------------------------------------------------------------
// Human renderers (text mode) + shared helpers.
// ---------------------------------------------------------------------------

/// Print a lap table: one row per lap, with sector + neutral-zone detail.
fn print_laps(laps: &[Lap]) {
    if laps.is_empty() {
        println!("no laps detected");
        return;
    }
    println!(
        "{:>4} {:>14} {:>14} {:>14}",
        "LAP", "START(ms)", "LAP(ms)", "RAW(ms)"
    );
    for l in laps {
        println!(
            "{:>4} {:>14} {:>14} {:>14}",
            l.lap_number, l.start_ms, l.lap_time_ms, l.raw_elapsed_ms
        );
        for s in &l.sectors {
            println!("       sector {:<12} {:>14}", s.name, s.end_ms - s.start_ms);
        }
        for z in &l.neutral_zone_visits {
            println!(
                "       neutral {:<12} -{:>13}",
                z.name,
                z.exit_ms - z.enter_ms
            );
        }
    }
}

/// Print track visits in time order. `name_of` maps a track_id to a display name.
fn print_visits(windows: &[VisitWindow], name_of: &dyn Fn(&str) -> String) {
    if windows.is_empty() {
        println!("no track visits detected");
        return;
    }
    println!(
        "{:<24} {:>14} {:>14} {:>14}",
        "TRACK", "START(ms)", "END(ms)", "DURATION(ms)"
    );
    for w in windows {
        println!(
            "{:<24} {:>14} {:>14} {:>14}",
            name_of(&w.track_id),
            w.start_timestamp_ms,
            w.end_timestamp_ms,
            w.end_timestamp_ms - w.start_timestamp_ms
        );
    }
}

/// Assemble the export channel set: derived channels (always), preceded by the
/// handle's base + synthesized channels when `include_base` is set. Owned — the
/// CLI is a one-shot tool, so cloning base channels is acceptable (D8's no-copy
/// rule governs the FFI bridge, not the CLI).
fn assemble_export_channels(
    report: ApplyReport,
    handle: &SessionHandle,
    include_base: bool,
) -> Vec<Channel> {
    let mut channels = Vec::new();
    if include_base {
        channels.extend(handle.channel_data().iter().cloned());
    }
    channels.extend(report.evaluated);
    channels
}

/// Print one stderr line per channel outcome. Data goes to the output sink; this
/// keeps diagnostics off it.
fn report_outcomes(report: &ApplyReport) {
    use idl_rs::math::MathEvalErrorKind;
    for r in &report.results {
        match &r.error {
            None => eprintln!("ok       {}", r.name),
            Some(e) if matches!(e.kind, MathEvalErrorKind::NoLapContext) => {
                eprintln!("skipped  {} (requires lap context — Phase 4)", r.name)
            }
            Some(e) => eprintln!("error    {} ({})", r.name, e.message),
        }
    }
}

/// Write the assembled channels to a file or stdout via `export::write_channels`.
fn write_math_sink(
    meta: &SessionMeta,
    synthesized_ids: &[String],
    channels: &[Channel],
    output: Option<&Path>,
    fmt: ExportFormat,
    options: &ExportOptions,
) -> Result<(), CliError> {
    let result = match output {
        Some(path) => {
            let file = File::create(path)
                .map_err(|e| CliError::io(format!("cannot write {}: {e}", path.display())))?;
            let mut w = BufWriter::new(file);
            let r = export::write_channels(meta, synthesized_ids, channels, &mut w, fmt, options);
            w.flush()
                .map_err(|e| CliError::io(format!("cannot write {}: {e}", path.display())))?;
            r
        }
        None => {
            let stdout = io::stdout();
            let mut w = stdout.lock();
            export::write_channels(meta, synthesized_ids, channels, &mut w, fmt, options)
        }
    };
    result.map_err(|e| match e {
        export::ExportError::UnknownChannel(name) => {
            let available: Vec<String> = channels.iter().map(|c| c.channel_id.clone()).collect();
            CliError::unknown_channel(&name, &available)
        }
        other => CliError::from(other),
    })
}

/// Parse a log file into an owned [`SessionHandle`] (synthesis runs inside).
/// Map an `idl-rs-video-export` error onto the envelope's error kinds.
fn export_err(e: ExportError) -> CliError {
    let kind = match e.kind {
        ExportErrorKind::FfmpegMissing => ErrorKind::Usage,
        ExportErrorKind::Probe => ErrorKind::InvalidInput,
        ExportErrorKind::Pipe | ExportErrorKind::FfmpegFailed | ExportErrorKind::Io => {
            ErrorKind::Io
        }
        ExportErrorKind::Cancelled => ErrorKind::Internal,
    };
    CliError::new(kind, e.message)
}

/// `ffprobe` binary path beside the given `ffmpeg` path (or bare name).
fn ffprobe_beside(ffmpeg: &str) -> String {
    let p = Path::new(ffmpeg);
    let probe_name = match p.file_name().and_then(|n| n.to_str()) {
        Some(name) => name.replacen("ffmpeg", "ffprobe", 1),
        None => "ffprobe".to_string(),
    };
    match p.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(probe_name).display().to_string(),
        _ => probe_name,
    }
}

/// `overlay` — render a workbook overlay layout onto a video (SPEC 33.6).
/// Bulk command: the artifact is the output video; errors envelope to stderr.
#[allow(clippy::too_many_arguments)]
fn cmd_overlay(
    file: &Path,
    video: &Path,
    wb_path: &Path,
    layout_name: Option<&str>,
    track: Option<&Path>,
    offset: Option<f64>,
    start: Option<f64>,
    duration: Option<f64>,
    output: Option<PathBuf>,
    encoder: String,
    ffmpeg: String,
) -> Result<(), CliError> {
    let handle = load(file)?;
    let wb = workbook::read_workbook(wb_path)?;

    // Math channels into the handle's store first — layout channels may be
    // math outputs. Per-channel failures degrade those elements to no-data.
    let report = workbook::apply_workbook(&handle, &wb, &MathLapContext::empty());
    for r in report.results.iter().filter(|r| r.error.is_some()) {
        eprintln!(
            "warning: math channel '{}' skipped: {}",
            r.name,
            r.error.as_ref().unwrap().message
        );
    }

    let layout = wb.overlay_layout(layout_name).map_err(CliError::usage)?;

    // Laps for the lap panel (optional --track; the FIT export precedent).
    let laps = match track {
        Some(tp) => {
            let t = track_artifact::read_track(tp)?;
            match t.timing.as_ref() {
                Some(timing) => {
                    detect_laps(&handle, timing, &t.sector_gates, &t.neutral_zones, None)
                }
                None => {
                    eprintln!(
                        "warning: track '{}' has no timing gates; lap panel renders no-data",
                        t.name
                    );
                    Vec::new()
                }
            }
        }
        None => Vec::new(),
    };

    // Sync offset: manual wins; else GPMF, else creation time.
    let video_path = path_str(video)?;
    let offset_s = match offset {
        Some(o) => o,
        None => {
            let info = mp4box::read_info_path(video_path).map_err(video_err)?;
            let telemetry = match mp4box::read_gpmd_samples_path(video_path) {
                Ok(samples) => Some(gpmf::parse_gpmf(&samples).map_err(video_err)?),
                Err(e) if e.kind == VideoErrorKind::NoGpmf => None,
                Err(e) => return Err(video_err(e)),
            };
            let est = estimate_sync(telemetry.as_ref(), &info, &handle).map_err(video_err)?;
            eprintln!(
                "sync: {:.3} s (method: {:?}, confidence: {:.1})",
                est.offset_s, est.method, est.confidence
            );
            est.offset_s
        }
    };

    // Plan the export.
    let probe = idl_rs_video_export::probe(video, &ffprobe_beside(&ffmpeg)).map_err(export_err)?;
    let default_name = {
        let stem = video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video");
        video.with_file_name(format!("{stem}_overlay.mp4"))
    };
    let plan = ExportPlan {
        video: video.to_path_buf(),
        output: output.unwrap_or(default_name),
        probe,
        start_s: start,
        duration_s: duration,
        encoder,
        ffmpeg_path: ffmpeg,
    };

    // Prepared sampling context; frame i → video clock → session clock.
    let ctx = SampleContext::prepare(&handle, layout, laps);
    let (fw, fh) = plan.frame_dims();
    let clip0 = start.unwrap_or(0.0);
    let fps = plan.probe.fps;
    let layout = layout.clone();
    let render = move |i: u64| {
        let t_video = clip0 + i as f64 / fps;
        let sample = ctx.sample(t_video + offset_s);
        render_overlay_frame(&layout, &sample, ctx.track_polyline(), fw, fh)
    };

    run_export(
        &plan,
        render,
        &mut |p: Progress| {
            eprint!("\r{} / {} frames", p.frames_done, p.frames_total);
        },
        &std::sync::atomic::AtomicBool::new(false),
    )
    .map_err(export_err)?;
    eprintln!("\nwrote {}", plan.output.display());
    Ok(())
}

/// Map an engine `VideoError` onto the envelope's closed error kinds.
fn video_err(e: VideoError) -> CliError {
    let kind = match e.kind {
        VideoErrorKind::Io => ErrorKind::Io,
        VideoErrorKind::Parse | VideoErrorKind::NoOverlap => ErrorKind::InvalidInput,
        VideoErrorKind::NoGpmf => ErrorKind::NotFound,
        VideoErrorKind::Export => ErrorKind::Internal,
    };
    CliError::new(kind, e.message)
}

/// UTF-8 path or a usage error (video files, like session paths).
fn path_str(path: &Path) -> Result<&str, CliError> {
    path.to_str()
        .ok_or_else(|| CliError::usage("path is not valid UTF-8"))
}

/// `video probe` — container facts via the pure-Rust ISO-BMFF walker.
fn cmd_video_probe(video: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let info = mp4box::read_info_path(path_str(video)?).map_err(video_err)?;
    match format {
        OutFormat::Text => {
            println!("width:         {}", info.width);
            println!("height:        {}", info.height);
            println!("fps:           {:.3}", info.fps);
            println!("duration_s:    {:.3}", info.duration_s);
            match info.creation_time_utc_ms {
                Some(ms) => println!("creation_utc:  {ms} ms"),
                None => println!("creation_utc:  (unset)"),
            }
            println!(
                "gpmf:          {}",
                if info.has_gpmd { "present" } else { "absent" }
            );
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data: json!({
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "duration_s": info.duration_s,
                "creation_time_utc_ms": info.creation_time_utc_ms,
                "has_gpmd": info.has_gpmd,
            }),
            warnings: vec![],
        }),
    }
}

/// `video sync` — offset estimate; GPMF absence falls through to
/// creation-time (normal, not an error).
fn cmd_video_sync(file: &Path, video: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let handle = load(file)?;
    let video_path = path_str(video)?;
    let info = mp4box::read_info_path(video_path).map_err(video_err)?;
    let telemetry = match mp4box::read_gpmd_samples_path(video_path) {
        Ok(samples) => Some(gpmf::parse_gpmf(&samples).map_err(video_err)?),
        Err(e) if e.kind == VideoErrorKind::NoGpmf => None,
        Err(e) => return Err(video_err(e)),
    };
    let est = estimate_sync(telemetry.as_ref(), &info, &handle).map_err(video_err)?;
    match format {
        OutFormat::Text => {
            println!(
                "offset: {:.3} s  (method: {}, confidence: {:.1})",
                est.offset_s,
                match est.method {
                    idl_rs::video::sync::SyncMethod::Gpmf => "gpmf",
                    idl_rs::video::sync::SyncMethod::CreationTime => "creation_time",
                },
                est.confidence
            );
            println!("pass --offset to `idl-rs overlay` to override");
            Ok(Structured::Text)
        }
        OutFormat::Json => Ok(Structured::Json {
            data: serde_json::to_value(&est).expect("SyncEstimate serializes"),
            warnings: vec![],
        }),
    }
}

fn load(path: &Path) -> Result<SessionHandle, CliError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| CliError::usage("path is not valid UTF-8"))?;
    Ok(SessionHandle::from_path(path_str)?)
}

/// Resolve the export format: explicit flag wins, else the output extension,
/// else CSV for stdout. Unknown/missing extension with no flag is a `usage`
/// error.
fn resolve_format(
    explicit: Option<FormatArg>,
    output: Option<&Path>,
) -> Result<ExportFormat, CliError> {
    if let Some(f) = explicit {
        return Ok(f.into());
    }
    match output {
        Some(path) => match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => Ok(ExportFormat::Csv),
            Some("json") => Ok(ExportFormat::Json),
            _ => Err(CliError::usage(
                "cannot infer format from output extension; pass --format csv|json",
            )),
        },
        None => Ok(ExportFormat::Csv),
    }
}

/// Default FIT output path: the input path with its extension replaced by
/// `fit` (or `.fit` appended when the input has no extension).
fn default_fit_output(input: &Path) -> PathBuf {
    input.with_extension("fit")
}

/// Write the export to a file (when `output` is set) or to stdout.
fn export_to_sink(
    handle: &SessionHandle,
    output: Option<&Path>,
    fmt: ExportFormat,
    options: &ExportOptions,
) -> Result<(), CliError> {
    let result = match output {
        Some(path) => {
            let file = File::create(path)
                .map_err(|e| CliError::io(format!("cannot write {}: {e}", path.display())))?;
            let mut w = BufWriter::new(file);
            let r = export::write(handle, &mut w, fmt, options);
            w.flush()
                .map_err(|e| CliError::io(format!("cannot write {}: {e}", path.display())))?;
            r
        }
        None => {
            let stdout = io::stdout();
            let mut w = stdout.lock();
            export::write(handle, &mut w, fmt, options)
        }
    };
    result.map_err(|e| export_error_to_cli(e, handle))
}

/// Turn an [`export::ExportError`] into a [`CliError`]; for an unknown channel,
/// attach the session's available channel ids so a consumer can self-correct.
fn export_error_to_cli(e: export::ExportError, handle: &SessionHandle) -> CliError {
    match e {
        export::ExportError::UnknownChannel(name) => {
            let available: Vec<String> = handle
                .channels()
                .into_iter()
                .map(|c| c.channel_id)
                .collect();
            CliError::unknown_channel(&name, &available)
        }
        other => CliError::from(other),
    }
}

/// An `invalid_input` error for a track artifact with no lap timing configured.
fn no_timing_error(name: &str, track: &Path) -> CliError {
    CliError::with_details(
        ErrorKind::InvalidInput,
        format!(
            "track '{}' has no lap timing configured",
            display_or_dash(name)
        ),
        json!({ "track": track.display().to_string() }),
    )
}

/// Build the machine-readable `warnings` array from a session's truncation
/// state — the §6 source of truth carried in the success envelope.
fn truncation_warnings(meta: &SessionMeta) -> Vec<Warning> {
    match &meta.truncation_warning {
        Some(w) => vec![Warning::truncated_log(format!("log incomplete — {w}"))],
        None => Vec::new(),
    }
}

/// Print the human truncation line to stderr (text mode and bulk commands).
fn warn_if_truncated(meta: &SessionMeta) {
    if let Some(w) = &meta.truncation_warning {
        eprintln!("warning: log incomplete — {w}");
    }
}

fn print_info(meta: &SessionMeta) {
    println!("session_id     {}", display_or_dash(&meta.session_id));
    println!("device_id      {}", display_or_dash(&meta.device_id));
    println!("start_utc_ms   {}", meta.timestamp_utc_ms);
    println!("config_crc32   {}", display_or_dash(&meta.config_checksum));
    println!("channels       {}", meta.channel_count);
    println!("duration_ms    {}", meta.duration_ms);
}

fn print_channels(channels: &[ChannelMeta]) {
    println!(
        "{:<20} {:>10} {:>10} {:>7}",
        "CHANNEL", "RATE(Hz)", "SAMPLES", "SYNTH"
    );
    for c in channels {
        let rate = if c.sample_rate_hz == 0.0 {
            "event".to_string()
        } else {
            format!("{:.3}", c.sample_rate_hz)
        };
        println!(
            "{:<20} {:>10} {:>10} {:>7}",
            c.channel_id, rate, c.length, c.synthesized
        );
    }
}

fn display_or_dash(s: &str) -> &str {
    if s.is_empty() {
        "-"
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::session::handle::{ChannelInput, SessionMetaInput};
    use idl_rs::workbook::ChannelApplyResult;
    use std::path::PathBuf;

    fn test_handle() -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "X".to_string(),
                sample_rate_hz: 10.0,
                samples: vec![1.0, 2.0],
                sample_times_secs: None,
            }],
        )
    }

    /// Build a handle with a named channel at `rate_hz` filled with `samples`.
    fn test_handle_with_channel(id: &str, rate_hz: f64, samples: Vec<f64>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: id.to_string(),
                sample_rate_hz: rate_hz,
                samples,
                sample_times_secs: None,
            }],
        )
    }

    // ---- fft_json_data tests --------------------------------------------------

    #[test]
    fn fft_json_data_returns_freqs_and_values_for_a_channel() {
        // Arrange — build a tiny handle in-process (no file IO).
        let handle = test_handle_with_channel(
            "X",
            64.0,
            (0..256).map(|i| (i as f64 * 0.1).sin()).collect(),
        );

        // Act
        let data = fft_json_data(
            &handle,
            "X",
            None,
            None,
            FftWindow::Hann,
            0,
            0,
            Detrend::Mean,
            Averaging::Mean,
            Scaling::Magnitude,
        )
        .unwrap();

        // Assert — freqs_hz and values arrays are present and equal length.
        let freqs = data["freqs_hz"].as_array().unwrap();
        let values = data["values"].as_array().unwrap();
        assert!(!freqs.is_empty());
        assert_eq!(freqs.len(), values.len());
        assert_eq!(data["channel"].as_str().unwrap(), "X");
    }

    #[test]
    fn fft_json_data_unknown_channel_returns_not_found_error() {
        // Arrange
        let handle = test_handle_with_channel("X", 64.0, vec![0.0; 64]);

        // Act
        let result = fft_json_data(
            &handle,
            "NOPE",
            None,
            None,
            FftWindow::Hann,
            0,
            0,
            Detrend::Mean,
            Averaging::Mean,
            Scaling::Magnitude,
        );

        // Assert — unknown channel yields a NotFound CliError.
        let e = result.unwrap_err();
        assert_eq!(e.kind, ErrorKind::NotFound);
    }

    #[test]
    fn fft_json_data_windowed_path_returns_valid_spectrum() {
        // Arrange — 10 Hz, 100 samples (10 s); window [2.0, 8.0] s.
        let handle = test_handle_with_channel(
            "X",
            10.0,
            (0..100).map(|i| (i as f64 * 0.2).sin()).collect(),
        );

        // Act
        let data = fft_json_data(
            &handle,
            "X",
            Some(2.0),
            Some(8.0),
            FftWindow::Hann,
            0,
            0,
            Detrend::Mean,
            Averaging::Mean,
            Scaling::Density,
        )
        .unwrap();

        // Assert — arrays are non-empty and equal length.
        let freqs = data["freqs_hz"].as_array().unwrap();
        let values = data["values"].as_array().unwrap();
        assert!(!freqs.is_empty());
        assert_eq!(freqs.len(), values.len());
    }

    // ---- spectrogram_json_data tests -----------------------------------------

    #[test]
    fn spectrogram_json_data_returns_matrix_dims_and_arrays() {
        // Arrange — 64 Hz tone, 256 samples; expect n_times >= 1, n_freqs >= 1.
        let handle = test_handle_with_channel(
            "X",
            64.0,
            (0..256)
                .map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / 64.0).sin())
                .collect(),
        );

        // Act
        let data = spectrogram_json_data(
            &handle,
            "X",
            None,
            None,
            FftWindow::Hann,
            64,
            32,
            Detrend::None,
            Scaling::Magnitude,
        )
        .unwrap();

        // Assert — shape fields are consistent with the flat power array.
        let n_times = data["n_times"].as_u64().unwrap();
        let n_freqs = data["n_freqs"].as_u64().unwrap();
        let power = data["power"].as_array().unwrap();
        let freqs = data["freqs_hz"].as_array().unwrap();
        let times = data["times_secs"].as_array().unwrap();
        assert!(n_times >= 1);
        assert!(n_freqs >= 1);
        assert_eq!(power.len() as u64, n_times * n_freqs);
        assert_eq!(freqs.len() as u64, n_freqs);
        assert_eq!(times.len() as u64, n_times);
        assert_eq!(data["channel"].as_str().unwrap(), "X");
    }

    #[test]
    fn spectrogram_json_data_unknown_channel_returns_not_found_error() {
        // Arrange
        let handle = test_handle_with_channel("X", 64.0, vec![0.0; 64]);

        // Act
        let result = spectrogram_json_data(
            &handle,
            "NOPE",
            None,
            None,
            FftWindow::Hann,
            0,
            0,
            Detrend::Mean,
            Scaling::Density,
        );

        // Assert
        let e = result.unwrap_err();
        assert_eq!(e.kind, ErrorKind::NotFound);
    }

    // ---- arg enum mapping tests ----------------------------------------------

    #[test]
    fn window_arg_maps_to_engine_variants() {
        // Act + Assert — each CLI arg maps to the expected engine variant.
        assert!(matches!(WindowArg::Hann.to_engine(), FftWindow::Hann));
        assert!(matches!(WindowArg::Hamming.to_engine(), FftWindow::Hamming));
        assert!(matches!(
            WindowArg::Rect.to_engine(),
            FftWindow::Rectangular
        ));
    }

    #[test]
    fn detrend_arg_maps_to_engine_variants() {
        assert!(matches!(DetrendArg::None.to_engine(), Detrend::None));
        assert!(matches!(DetrendArg::Mean.to_engine(), Detrend::Mean));
        assert!(matches!(DetrendArg::Linear.to_engine(), Detrend::Linear));
    }

    #[test]
    fn averaging_arg_maps_to_engine_variants() {
        assert!(matches!(AveragingArg::Mean.to_engine(), Averaging::Mean));
        assert!(matches!(
            AveragingArg::Median.to_engine(),
            Averaging::Median
        ));
    }

    #[test]
    fn scaling_arg_maps_to_engine_variants() {
        assert!(matches!(
            ScalingArg::Magnitude.to_engine(),
            Scaling::Magnitude
        ));
        assert!(matches!(ScalingArg::Density.to_engine(), Scaling::Density));
    }

    fn derived_report() -> ApplyReport {
        ApplyReport {
            results: vec![ChannelApplyResult {
                name: "D".to_string(),
                error: None,
            }],
            evaluated: vec![Channel::from_f64("D", 10.0, vec![2.0, 4.0], None)],
        }
    }

    #[test]
    fn assemble_derived_only_excludes_base_channels() {
        // Arrange
        let h = test_handle();

        // Act
        let chans = assemble_export_channels(derived_report(), &h, false);

        // Assert — only the derived channel "D"; no "X" or synthesized "Time".
        let ids: Vec<&str> = chans.iter().map(|c| c.channel_id.as_str()).collect();
        assert_eq!(ids, vec!["D"]);
    }

    #[test]
    fn assemble_include_base_prepends_base_and_synth() {
        // Arrange
        let h = test_handle();

        // Act
        let chans = assemble_export_channels(derived_report(), &h, true);

        // Assert — base "X" and synthesized "Time" present, plus derived "D" last.
        let ids: Vec<&str> = chans.iter().map(|c| c.channel_id.as_str()).collect();
        assert!(ids.contains(&"X"));
        assert!(ids.contains(&"Time"));
        assert_eq!(ids.last(), Some(&"D"));
    }

    #[test]
    fn resolve_format_prefers_explicit_flag() {
        // Arrange — extension says json, flag says csv.
        let out = PathBuf::from("x.json");

        // Act
        let f = resolve_format(Some(FormatArg::Csv), Some(&out)).unwrap();

        // Assert
        assert_eq!(f, ExportFormat::Csv);
    }

    #[test]
    fn resolve_format_infers_from_extension() {
        // Act + Assert
        assert_eq!(
            resolve_format(None, Some(&PathBuf::from("a.csv"))).unwrap(),
            ExportFormat::Csv
        );
        assert_eq!(
            resolve_format(None, Some(&PathBuf::from("a.json"))).unwrap(),
            ExportFormat::Json
        );
    }

    #[test]
    fn resolve_format_defaults_to_csv_for_stdout() {
        // Act + Assert — no output path, no flag.
        assert_eq!(resolve_format(None, None).unwrap(), ExportFormat::Csv);
    }

    #[test]
    fn resolve_format_unknown_extension_is_usage_error() {
        // Act
        let r = resolve_format(None, Some(&PathBuf::from("a.txt")));

        // Assert — a `usage` error (format cannot be inferred).
        let e = r.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Usage);
    }

    #[test]
    fn default_fit_output_path_replaces_extension() {
        // Act + Assert
        assert_eq!(
            default_fit_output(&PathBuf::from("ride.idl0")),
            PathBuf::from("ride.fit")
        );
        assert_eq!(
            default_fit_output(&PathBuf::from("/data/logs/run.idl0")),
            PathBuf::from("/data/logs/run.fit")
        );
        // No extension → append .fit.
        assert_eq!(
            default_fit_output(&PathBuf::from("noext")),
            PathBuf::from("noext.fit")
        );
    }

    #[test]
    fn no_timing_error_is_invalid_input_with_track_detail() {
        // Act
        let e = no_timing_error("Whistler A-Line", &PathBuf::from("whistler.idl0t"));

        // Assert
        assert_eq!(e.kind, ErrorKind::InvalidInput);
        assert_eq!(e.details.unwrap()["track"], "whistler.idl0t");
    }

    #[test]
    fn sport_arg_maps_to_fit_sport() {
        // Act + Assert
        assert_eq!(FitSport::from(SportArg::Cycling), FitSport::Cycling);
        assert_eq!(
            FitSport::from(SportArg::Motorcycling),
            FitSport::Motorcycling
        );
        assert_eq!(FitSport::from(SportArg::Running), FitSport::Running);
        assert_eq!(FitSport::from(SportArg::Generic), FitSport::Generic);
    }
}
