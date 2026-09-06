//! Raster commands (C3 §3.6): `fetch_raster` (binary pixels) and
//! `fetch_raster_meta` (axis domains + colour scale, JSON) over L3's
//! `idl_rs::raster` spectrogram/histogram2d encoders.
//!
//! Same idiom as `commands/device.rs`/`commands/workbook.rs`: each
//! `#[tauri::command]` is a one-line wrapper over a `_via`-suffixed plain
//! function taking `data_dir: &Path` — this module's own tests exercise the
//! `_via` functions (`tauri::State`/`tauri::ipc::Response` cannot be
//! constructed outside a running app).

use std::path::Path;

use idl_rs::fft::{Averaging, Detrend, FftWindow, Scaling};
use idl_rs::raster::{
    build_histogram2d_raster_bytes, build_spectrogram_raster_bytes, histogram2d_raster_meta,
    spectrogram_raster_meta, RasterMeta as CoreRasterMeta,
};
use idl_rs::session::{Channel, Session};

use crate::error::{IpcError, IpcErrorKind};
use crate::session_source::load_session;
use crate::state::DataDir;

/// `SpectrogramParams.window` (C3 §3.6) — `idl_rs::fft::FftWindow`'s three
/// tokens, snake_case on the wire.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowToken {
    Rectangular,
    Hann,
    Hamming,
}

impl From<WindowToken> for FftWindow {
    fn from(t: WindowToken) -> Self {
        match t {
            WindowToken::Rectangular => FftWindow::Rectangular,
            WindowToken::Hann => FftWindow::Hann,
            WindowToken::Hamming => FftWindow::Hamming,
        }
    }
}

/// `SpectrogramParams.detrend` (C3 §3.6) — `idl_rs::fft::Detrend`'s three
/// tokens, snake_case on the wire.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetrendToken {
    None,
    Mean,
    Linear,
}

impl From<DetrendToken> for Detrend {
    fn from(t: DetrendToken) -> Self {
        match t {
            DetrendToken::None => Detrend::None,
            DetrendToken::Mean => Detrend::Mean,
            DetrendToken::Linear => Detrend::Linear,
        }
    }
}

/// `SpectrogramParams.scaling` (C3 §3.6) — `idl_rs::fft::Scaling`'s two
/// tokens, snake_case on the wire.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalingToken {
    Magnitude,
    Density,
}

impl From<ScalingToken> for Scaling {
    fn from(t: ScalingToken) -> Self {
        match t {
            ScalingToken::Magnitude => Scaling::Magnitude,
            ScalingToken::Density => Scaling::Density,
        }
    }
}

/// `fetch_fft`'s `averaging` argument (C3 §3.6) — `idl_rs::fft::Averaging`'s
/// four tokens, snake_case on the wire. Ruling R63 (3): the enum was
/// extended with `None`/`Max` to close the gap C3's own text used to call
/// out (a two-variant engine enum against a four-token wire union) — every
/// token maps directly now, none rejected.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AveragingToken {
    None,
    Mean,
    Median,
    Max,
}

impl From<AveragingToken> for Averaging {
    fn from(t: AveragingToken) -> Self {
        match t {
            AveragingToken::None => Averaging::None,
            AveragingToken::Mean => Averaging::Mean,
            AveragingToken::Median => Averaging::Median,
            AveragingToken::Max => Averaging::Max,
        }
    }
}

/// `fetch_raster`/`fetch_raster_meta`'s `params` when `kind == "spectrogram"`
/// (C3 §3.6). `window_size`/`hop_size` are in **samples**.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SpectrogramParams {
    /// `nperseg`, samples per FFT segment.
    pub window_size: u32,
    /// Samples between segment starts. `idl_rs::fft`'s own parameter is
    /// `noverlap`; this task's boundary converts `noverlap = window_size -
    /// hop_size` (`core/src/raster.rs:64-66`).
    pub hop_size: u32,
    pub window: WindowToken,
    pub detrend: DetrendToken,
    pub scaling: ScalingToken,
}

/// `fetch_raster`/`fetch_raster_meta`'s `params` when `kind == "histogram2d"`
/// (C3 §3.6, amended by ruling R42). `x_bins`/`y_bins` must equal the
/// command's own `width`/`height` in wave 1 — see [`fetch_raster_via`]'s doc
/// comment for why the fields are kept anyway.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Histogram2dParams {
    /// The second channel; the command's own `channel` argument is the X channel.
    pub y_channel: String,
    pub x_bins: u32,
    pub y_bins: u32,
}

/// Converts `SpectrogramParams`' `window`/`detrend`/`scaling` tokens and
/// `window_size`/`hop_size` into `idl_rs::fft`'s own engine types plus a
/// `noverlap` sample count, rejecting `window_size == 0`, `hop_size == 0`,
/// and `hop_size > window_size` before any engine call (a channel with too
/// few samples to honour the request is the engine's own business — this
/// only rejects requests the engine cannot even be asked to run).
fn resolve_spectrogram_params(p: &SpectrogramParams) -> Result<(FftWindow, Detrend, Scaling, usize, usize), IpcError> {
    if p.window_size == 0 {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, "window_size must be nonzero"));
    }
    if p.hop_size == 0 {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, "hop_size must be nonzero"));
    }
    if p.hop_size > p.window_size {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("hop_size ({}) must not exceed window_size ({})", p.hop_size, p.window_size),
            serde_json::json!({ "hop_size": p.hop_size, "window_size": p.window_size }),
        ));
    }
    let noverlap = (p.window_size - p.hop_size) as usize;
    Ok((p.window.into(), p.detrend.into(), p.scaling.into(), p.window_size as usize, noverlap))
}

/// Deserialises `params` into [`SpectrogramParams`], mapping a bad shape to
/// `invalid_argument` with serde's own message in `detail` — deliberately
/// not an `#[serde(untagged)]` enum over both param shapes (an untagged
/// enum would turn a typo in one field into a silent match against the
/// *other* shape).
fn parse_spectrogram_params(params: &serde_json::Value) -> Result<SpectrogramParams, IpcError> {
    serde_json::from_value(params.clone()).map_err(|e| {
        IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("invalid spectrogram params: {e}"),
            serde_json::json!({ "serde_error": e.to_string() }),
        )
    })
}

/// Deserialises `params` into [`Histogram2dParams`] — see
/// [`parse_spectrogram_params`] for why this is explicit dispatch, not an
/// untagged enum.
fn parse_histogram2d_params(params: &serde_json::Value) -> Result<Histogram2dParams, IpcError> {
    serde_json::from_value(params.clone()).map_err(|e| {
        IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("invalid histogram2d params: {e}"),
            serde_json::json!({ "serde_error": e.to_string() }),
        )
    })
}

/// Finds `channel_id` in `session.channels`, or `not_found` naming it.
fn find_channel<'a>(session: &'a Session, channel_id: &str) -> Result<&'a Channel, IpcError> {
    session
        .channels
        .iter()
        .find(|c| c.channel_id == channel_id)
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, format!("channel '{channel_id}' not found")))
}

/// Rejects `x_bins`/`y_bins` that disagree with the command's own
/// `width`/`height` (ruling R42) — see [`fetch_raster_via`]'s doc comment.
fn check_bins_match_pixels(p: &Histogram2dParams, width: u16, height: u16) -> Result<(), IpcError> {
    if p.x_bins != width as u32 || p.y_bins != height as u32 {
        return Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            "x_bins/y_bins must equal width/height: the landed histogram2d encoder builds one bin per pixel and does not rebin, so the two argument pairs must describe one grid",
            serde_json::json!({ "x_bins": p.x_bins, "y_bins": p.y_bins, "width": width, "height": height }),
        ));
    }
    Ok(())
}

/// Spectrogram axis labels (C3 §3.6's `fetch_raster_meta`) — fixed strings,
/// not derived from any channel: a spectrogram's axes are always time and
/// frequency regardless of the source channel's own unit (the engine's own
/// axes, `SpectrogramResult.times_secs`/`freqs_hz`, `core/src/spectrogram.rs`).
const SPECTROGRAM_X_LABEL: &str = "time (s)";
const SPECTROGRAM_Y_LABEL: &str = "frequency (Hz)";

/// Builds a histogram2d axis label as `"<channel> (<unit>)"`, omitting the
/// parenthesised unit when `unit` is empty.
fn histogram_axis_label(channel_id: &str, unit: &str) -> String {
    if unit.is_empty() {
        channel_id.to_string()
    } else {
        format!("{channel_id} ({unit})")
    }
}

/// C3 §3.6's `RasterMeta.scale` — `vmin`/`vmax` copied verbatim from core's
/// flat `RasterMeta` (ledger R38: resolution-independent, scanned over the
/// full pre-rebin matrix), `kind` fixed at `"linear"` (wave 1 has no other
/// colour-scale kind).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RasterMetaScale {
    pub vmin: f64,
    pub vmax: f64,
    pub kind: String,
}

/// `fetch_raster_meta`'s return (C3 §3.6). Nests core's flat `vmin`/`vmax`
/// inside `scale`, and fills `x_label`/`y_label` here — core has no
/// session/catalog context to derive them from (`core/src/raster.rs`
/// module doc).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RasterMetaOut {
    pub x_domain: (f64, f64),
    pub y_domain: (f64, f64),
    pub x_label: String,
    pub y_label: String,
    pub scale: RasterMetaScale,
    pub transparent_zero: bool,
}

impl RasterMetaOut {
    fn from_core(meta: CoreRasterMeta, x_label: String, y_label: String) -> Self {
        Self {
            x_domain: meta.x_domain,
            y_domain: meta.y_domain,
            x_label,
            y_label,
            scale: RasterMetaScale { vmin: meta.vmin, vmax: meta.vmax, kind: "linear".to_string() },
            transparent_zero: meta.transparent_zero,
        }
    }
}

/// Transport-agnostic core of `fetch_raster`. Every validation happens
/// before any bytes are built (C3 §1's binary-transport rule): unknown
/// session/channel(s) → `not_found`; `width == 0 || height == 0`, unknown
/// `kind`, bad `params`, an event-driven channel requested for a
/// spectrogram, mismatched `x_bins`/`y_bins`, or mismatched channel lengths
/// for a histogram2d → `invalid_argument`.
///
/// **`x_bins`/`y_bins` vs `width`/`height` (ruling R42):** the landed
/// `build_histogram2d_raster_bytes` builds exactly one bin per pixel and
/// does not rebin (unlike the spectrogram path) — so the two argument pairs
/// describe one grid, and a mismatch is rejected rather than silently
/// preferring one. The fields are kept (not deleted) because bins < pixels
/// — an upsampled display of a coarse histogram — is the intended future
/// extension.
pub fn fetch_raster_via(
    data_dir: &Path,
    session_id: &str,
    channel: &str,
    kind: &str,
    width: u16,
    height: u16,
    params: &serde_json::Value,
) -> Result<Vec<u8>, IpcError> {
    if width == 0 || height == 0 {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("width/height must be nonzero, got {width}x{height}")));
    }
    let session = load_session(data_dir, session_id)?;

    match kind {
        "spectrogram" => {
            let ch = find_channel(&session, channel)?;
            if ch.nominal_rate_hz == 0.0 {
                return Err(IpcError::new(
                    IpcErrorKind::InvalidArgument,
                    format!("channel '{channel}' is event-driven (nominal_rate_hz = 0); a spectrogram needs a fixed-rate channel"),
                ));
            }
            let p = parse_spectrogram_params(params)?;
            let (window, detrend, scaling, window_size, noverlap) = resolve_spectrogram_params(&p)?;
            let samples = ch.materialize();
            Ok(build_spectrogram_raster_bytes(&samples, ch.nominal_rate_hz, width, height, window, window_size, noverlap, detrend, scaling))
        }
        "histogram2d" => {
            let p = parse_histogram2d_params(params)?;
            check_bins_match_pixels(&p, width, height)?;
            let x_ch = find_channel(&session, channel)?;
            let y_ch = find_channel(&session, &p.y_channel)?;
            let xs = x_ch.materialize();
            let ys = y_ch.materialize();
            if xs.len() != ys.len() {
                // TODO(idl0): resampling one channel onto the other's recorded
                // time axis so mismatched-length pairs can still be histogrammed
                // is a wave-2 decision — no contract fixes it yet (C1: every
                // channel keeps its own recorded time axis).
                return Err(IpcError::with_detail(
                    IpcErrorKind::InvalidArgument,
                    format!(
                        "histogram2d requires equal-length channels: '{channel}' has {} samples, '{}' has {}",
                        xs.len(),
                        p.y_channel,
                        ys.len()
                    ),
                    serde_json::json!({
                        "x_channel": channel, "x_len": xs.len(),
                        "y_channel": p.y_channel, "y_len": ys.len(),
                    }),
                ));
            }
            Ok(build_histogram2d_raster_bytes(&xs, &ys, width, height, None, None))
        }
        other => Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("unknown raster kind '{other}'"))),
    }
}

/// Transport-agnostic core of `fetch_raster_meta`. Same argument validation
/// as [`fetch_raster_via`] — see its doc comment. `vmin`/`vmax` come from
/// the meta functions, never from encoded pixel bytes (ledger R38); `width`/
/// `height` are never passed into the meta functions (they take none,
/// deliberately — `core/src/raster.rs:192-206`/`:217-…`).
pub fn fetch_raster_meta_via(
    data_dir: &Path,
    session_id: &str,
    channel: &str,
    kind: &str,
    width: u16,
    height: u16,
    params: &serde_json::Value,
) -> Result<RasterMetaOut, IpcError> {
    if width == 0 || height == 0 {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("width/height must be nonzero, got {width}x{height}")));
    }
    let session = load_session(data_dir, session_id)?;

    match kind {
        "spectrogram" => {
            let ch = find_channel(&session, channel)?;
            if ch.nominal_rate_hz == 0.0 {
                return Err(IpcError::new(
                    IpcErrorKind::InvalidArgument,
                    format!("channel '{channel}' is event-driven (nominal_rate_hz = 0); a spectrogram needs a fixed-rate channel"),
                ));
            }
            let p = parse_spectrogram_params(params)?;
            let (window, detrend, scaling, window_size, noverlap) = resolve_spectrogram_params(&p)?;
            let samples = ch.materialize();
            let meta = spectrogram_raster_meta(&samples, ch.nominal_rate_hz, window, window_size, noverlap, detrend, scaling);
            Ok(RasterMetaOut::from_core(meta, SPECTROGRAM_X_LABEL.to_string(), SPECTROGRAM_Y_LABEL.to_string()))
        }
        "histogram2d" => {
            let p = parse_histogram2d_params(params)?;
            check_bins_match_pixels(&p, width, height)?;
            let x_ch = find_channel(&session, channel)?;
            let y_ch = find_channel(&session, &p.y_channel)?;
            let xs = x_ch.materialize();
            let ys = y_ch.materialize();
            if xs.len() != ys.len() {
                // TODO(idl0): see fetch_raster_via — same deferred resampling decision.
                return Err(IpcError::with_detail(
                    IpcErrorKind::InvalidArgument,
                    format!(
                        "histogram2d requires equal-length channels: '{channel}' has {} samples, '{}' has {}",
                        xs.len(),
                        p.y_channel,
                        ys.len()
                    ),
                    serde_json::json!({
                        "x_channel": channel, "x_len": xs.len(),
                        "y_channel": p.y_channel, "y_len": ys.len(),
                    }),
                ));
            }
            let meta = histogram2d_raster_meta(&xs, &ys, width as usize, height as usize, None, None);
            let x_label = histogram_axis_label(channel, &x_ch.unit);
            let y_label = histogram_axis_label(&p.y_channel, &y_ch.unit);
            Ok(RasterMetaOut::from_core(meta, x_label, y_label))
        }
        other => Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("unknown raster kind '{other}'"))),
    }
}

/// Fetches and encodes one raster (C3 §3.6). `params` is
/// [`SpectrogramParams`] or [`Histogram2dParams`] JSON, matching `kind`.
#[tauri::command]
pub fn fetch_raster(
    session_id: String,
    channel: String,
    kind: String,
    width: u16,
    height: u16,
    params: serde_json::Value,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<tauri::ipc::Response, IpcError> {
    let bytes = fetch_raster_via(&data_dir.0, &session_id, &channel, &kind, width, height, &params)?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// Fetches one raster's axis domains and colour scale without decoding
/// pixel bytes (C3 §3.6, added post-sign, ledger R25). Same arguments as
/// `fetch_raster`.
#[tauri::command]
pub fn fetch_raster_meta(
    session_id: String,
    channel: String,
    kind: String,
    width: u16,
    height: u16,
    params: serde_json::Value,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<RasterMetaOut, IpcError> {
    fetch_raster_meta_via(&data_dir.0, &session_id, &channel, &kind, width, height, &params)
}

/// Derives a channel's effective sample rate in Hz from its recorded `t_us`
/// axis: `1e6 / median(consecutive-sample gaps in microseconds)` — never
/// `nominal_rate_hz` (C1 §3.5: metadata only, never used to synthesize
/// time). No existing "effective rate from t_us" helper was found under
/// `core/src/session/` as of this task's writing (`grep -rn
/// "effective_rate|derive.*rate" core/src/session` — no hits); this is this
/// wrapper's own derivation, not a call into a pre-existing helper. Fewer
/// than two samples has no gap to measure and returns `0.0`.
fn effective_rate_hz_from_t_us(t_us: &[i64]) -> f64 {
    if t_us.len() < 2 {
        return 0.0;
    }
    let mut gaps: Vec<i64> = t_us.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_unstable();
    let n = gaps.len();
    let median_us = if n % 2 == 1 {
        gaps[n / 2] as f64
    } else {
        0.5 * (gaps[n / 2 - 1] + gaps[n / 2]) as f64
    };
    if median_us <= 0.0 {
        0.0
    } else {
        1e6 / median_us
    }
}

/// Builds the [`IpcErrorKind::InvalidArgument`] `fetch_fft` returns for any
/// non-null `lap` (C3 §3.6: "`lap` must be `null` in practice until lap
/// indexing lands"). `fetch_fft`'s `lap` argument is a plain `Option<u32>`,
/// not a `LapContext` (contrast `eval_workbook`'s
/// [`crate::session_source::load_lap_context`]), so this gate is this
/// command's own — not a call into Task 9's helper. Task 9's gate only fires
/// when `session.json` exists with a non-matching `laps[]`; since
/// `session.json` may be entirely absent, that helper alone would silently
/// accept a non-null `lap` here, so this command rejects unconditionally
/// instead (matching C3's own "always" wording). Note for the lead: once lap
/// indexing lands and Task 9's helper resolves real bounds, this duplication
/// should be reconciled — likely by widening `load_lap_context` (or a new
/// sibling) to also serve a single-lap sample-window lookup.
fn reject_non_null_lap(lap: Option<u32>) -> Result<(), IpcError> {
    match lap {
        None => Ok(()),
        Some(n) => Err(IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("lap {n} not supported: lap indexing has not landed"),
            serde_json::json!({ "lap": n }),
        )),
    }
}

/// Transport-agnostic core of `fetch_fft` (C3 §3.6, ruling R63 (3)). Loads
/// the channel's samples via [`load_session`]/[`find_channel`], derives
/// `sample_rate_hz` from the channel's recorded `t_us` axis (see
/// [`effective_rate_hz_from_t_us`]), and calls `idl_rs::fft::welch`.
/// `not_found`: unknown `session_id` or `channel`. `invalid_argument`: a
/// non-null `lap` (see [`reject_non_null_lap`]), or `params` that fail
/// [`resolve_spectrogram_params`]'s validation.
pub fn fetch_fft_via(
    data_dir: &Path,
    session_id: &str,
    channel: &str,
    lap: Option<u32>,
    params: &SpectrogramParams,
    averaging: Averaging,
) -> Result<Vec<u8>, IpcError> {
    reject_non_null_lap(lap)?;
    let session = load_session(data_dir, session_id)?;
    let ch = find_channel(&session, channel)?;
    let (window, detrend, scaling, window_size, noverlap) = resolve_spectrogram_params(params)?;
    let samples = ch.materialize();
    let sample_rate_hz = effective_rate_hz_from_t_us(&ch.t_us);
    let result = idl_rs::fft::welch(samples, sample_rate_hz, window, window_size, noverlap, detrend, averaging, scaling);
    Ok(idl_rs::fft_wire::encode_fft_idlf(&result.values, sample_rate_hz))
}

/// Fetches one channel's FFT spectrum as `IDLF` v1 bytes (C3 §3.6, ruling
/// R63 (3)). `lap` must be `null` today — see [`reject_non_null_lap`].
#[tauri::command]
pub fn fetch_fft(
    session_id: String,
    channel: String,
    lap: Option<u32>,
    params: SpectrogramParams,
    averaging: AveragingToken,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<tauri::ipc::Response, IpcError> {
    let bytes = fetch_fft_via(&data_dir.0, &session_id, &channel, lap, &params, averaging.into())?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{RawColumn, SourceFormat};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-rasters-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds a session with `session_id` = "s1", holding a fixed-rate sine
    /// channel "Speed" and a second fixed-rate channel "Cadence" of equal
    /// length, plus an event-driven ("rate 0") channel "Lap".
    fn seed_session(root: &std::path::Path) {
        let n = 128usize;
        let fs = 64.0;
        let sine: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin()).collect();
        let cosine: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).cos()).collect();
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![
                Channel {
                    channel_id: "Speed".to_string(),
                    t_us: (0..n).map(|i| (i as f64 * 1_000_000.0 / fs) as i64).collect(),
                    t_recorded_us: None,
                    nominal_rate_hz: fs,
                    column: RawColumn::F64(sine),
                    source_kind: "wheel".to_string(),
                    unit: "m/s".to_string(),
                    gaps: Vec::new(),
                },
                Channel {
                    channel_id: "Cadence".to_string(),
                    t_us: (0..n).map(|i| (i as f64 * 1_000_000.0 / fs) as i64).collect(),
                    t_recorded_us: None,
                    nominal_rate_hz: fs,
                    column: RawColumn::F64(cosine),
                    source_kind: "cadence".to_string(),
                    unit: "rpm".to_string(),
                    gaps: Vec::new(),
                },
                Channel {
                    channel_id: "Lap".to_string(),
                    t_us: vec![0, 1_000_000],
                    t_recorded_us: None,
                    nominal_rate_hz: 0.0,
                    column: RawColumn::F64(vec![1.0, 2.0]),
                    source_kind: "event".to_string(),
                    unit: String::new(),
                    gaps: Vec::new(),
                },
            ],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn spectrogram_params_json() -> serde_json::Value {
        serde_json::json!({ "window_size": 32, "hop_size": 16, "window": "hann", "detrend": "mean", "scaling": "density" })
    }

    fn histogram2d_params_json(x_bins: u32, y_bins: u32) -> serde_json::Value {
        serde_json::json!({ "y_channel": "Cadence", "x_bins": x_bins, "y_bins": y_bins })
    }

    #[test]
    fn window_detrend_scaling_tokens_convert_to_engine_types() {
        // Arrange / Act / Assert — one representative each; conversion is total.
        assert!(matches!(FftWindow::from(WindowToken::Hann), FftWindow::Hann));
        assert!(matches!(FftWindow::from(WindowToken::Rectangular), FftWindow::Rectangular));
        assert!(matches!(FftWindow::from(WindowToken::Hamming), FftWindow::Hamming));
        assert!(matches!(Detrend::from(DetrendToken::Linear), Detrend::Linear));
        assert!(matches!(Detrend::from(DetrendToken::None), Detrend::None));
        assert!(matches!(Detrend::from(DetrendToken::Mean), Detrend::Mean));
        assert!(matches!(Scaling::from(ScalingToken::Density), Scaling::Density));
        assert!(matches!(Scaling::from(ScalingToken::Magnitude), Scaling::Magnitude));
    }

    #[test]
    fn resolve_spectrogram_params_computes_noverlap_from_window_size_minus_hop_size() {
        // Arrange
        let p = SpectrogramParams { window_size: 64, hop_size: 16, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let (_, _, _, window_size, noverlap) = resolve_spectrogram_params(&p).unwrap();

        // Assert
        assert_eq!(window_size, 64);
        assert_eq!(noverlap, 48);
    }

    #[test]
    fn resolve_spectrogram_params_hop_size_larger_than_window_size_invalid_argument() {
        // Arrange
        let p = SpectrogramParams { window_size: 16, hop_size: 32, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let result = resolve_spectrogram_params(&p);

        // Assert — engine types (`FftWindow`/`Detrend`/`Scaling`) carry no
        // `Debug` impl, so this matches rather than `unwrap_err()`s.
        match result {
            Err(err) => assert_eq!(err.kind, IpcErrorKind::InvalidArgument),
            Ok(_) => panic!("expected invalid_argument, got Ok"),
        }
    }

    #[test]
    fn resolve_spectrogram_params_zero_hop_size_invalid_argument() {
        // Arrange
        let p = SpectrogramParams { window_size: 16, hop_size: 0, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act / Assert
        match resolve_spectrogram_params(&p) {
            Err(err) => assert_eq!(err.kind, IpcErrorKind::InvalidArgument),
            Ok(_) => panic!("expected invalid_argument, got Ok"),
        }
    }

    #[test]
    fn resolve_spectrogram_params_zero_window_size_invalid_argument() {
        // Arrange
        let p = SpectrogramParams { window_size: 0, hop_size: 0, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act / Assert
        match resolve_spectrogram_params(&p) {
            Err(err) => assert_eq!(err.kind, IpcErrorKind::InvalidArgument),
            Ok(_) => panic!("expected invalid_argument, got Ok"),
        }
    }

    #[test]
    fn fetch_raster_spectrogram_64x32_header_fields_and_total_length_match_c3_formula() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_raster_via(&root, "s1", "Speed", "spectrogram", 64, 32, &spectrogram_params_json()).unwrap();

        // Assert — C3 §3.6: 16 + width*height*4 = 16 + 64*32*4 = 8208.
        assert_eq!(bytes.len(), 8208);
        assert_eq!(&bytes[0..4], b"IDLR");
        assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 64);
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()), 32);
        assert_eq!(u16::from_le_bytes(bytes[10..12].try_into().unwrap()), 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_histogram2d_x_bins_y_bins_equal_width_height_well_formed_bytes() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let bytes = fetch_raster_via(&root, "s1", "Speed", "histogram2d", 16, 8, &histogram2d_params_json(16, 8)).unwrap();

        // Assert
        assert_eq!(bytes.len(), 16 + 16 * 8 * 4);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_histogram2d_x_bins_mismatch_invalid_argument_with_both_in_detail() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_raster_via(&root, "s1", "Speed", "histogram2d", 16, 8, &histogram2d_params_json(4, 8)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        let detail = err.detail.unwrap();
        assert_eq!(detail["x_bins"], 4);
        assert_eq!(detail["y_bins"], 8);
        assert_eq!(detail["width"], 16);
        assert_eq!(detail["height"], 8);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_width_zero_invalid_argument_no_bytes_produced() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_raster_via(&root, "s1", "Speed", "spectrogram", 0, 32, &spectrogram_params_json()).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_unknown_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_raster_via(&root, "s1", "NopeChannel", "spectrogram", 8, 8, &spectrogram_params_json()).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_event_driven_channel_requesting_spectrogram_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_raster_via(&root, "s1", "Lap", "spectrogram", 8, 8, &spectrogram_params_json()).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_hop_size_larger_than_window_size_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let bad_params = serde_json::json!({ "window_size": 16, "hop_size": 32, "window": "hann", "detrend": "mean", "scaling": "density" });

        // Act
        let err = fetch_raster_via(&root, "s1", "Speed", "spectrogram", 8, 8, &bad_params).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_meta_spectrogram_scale_kind_linear_and_bounds_match_direct_call() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let session = load_session(&root, "s1").unwrap();
        let ch = find_channel(&session, "Speed").unwrap();
        let samples = ch.materialize();
        let direct = spectrogram_raster_meta(&samples, ch.nominal_rate_hz, FftWindow::Hann, 32, 16, Detrend::Mean, Scaling::Density);

        // Act
        let meta = fetch_raster_meta_via(&root, "s1", "Speed", "spectrogram", 64, 32, &spectrogram_params_json()).unwrap();

        // Assert — R38's guarantee, tested rather than trusted: same bounds
        // as calling the core meta function directly, regardless of width/height.
        assert_eq!(meta.scale.kind, "linear");
        assert_eq!(meta.scale.vmin, direct.vmin);
        assert_eq!(meta.scale.vmax, direct.vmax);
        assert_eq!(meta.x_domain, direct.x_domain);
        assert_eq!(meta.y_domain, direct.y_domain);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_raster_meta_histogram2d_labels_carry_channel_names_and_units() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let meta = fetch_raster_meta_via(&root, "s1", "Speed", "histogram2d", 16, 8, &histogram2d_params_json(16, 8)).unwrap();

        // Assert
        assert_eq!(meta.x_label, "Speed (m/s)");
        assert_eq!(meta.y_label, "Cadence (rpm)");
        assert!(meta.transparent_zero);

        let _ = std::fs::remove_dir_all(&root);
    }

    fn fft_params() -> SpectrogramParams {
        SpectrogramParams { window_size: 32, hop_size: 16, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density }
    }

    #[test]
    fn effective_rate_hz_from_t_us_uniform_spacing_matches_expected_rate() {
        // Arrange — 64 Hz spacing, i.e. 15625 us gaps
        let t_us: Vec<i64> = (0..8).map(|i| i * 15_625).collect();

        // Act
        let hz = effective_rate_hz_from_t_us(&t_us);

        // Assert
        assert!((hz - 64.0).abs() < 1e-6, "expected ~64 Hz, got {hz}");
    }

    #[test]
    fn effective_rate_hz_from_t_us_fewer_than_two_samples_returns_zero() {
        // Arrange / Act / Assert
        assert_eq!(effective_rate_hz_from_t_us(&[]), 0.0);
        assert_eq!(effective_rate_hz_from_t_us(&[100]), 0.0);
    }

    #[test]
    fn fetch_fft_via_lap_some_invalid_argument_regardless_of_n() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_fft_via(&root, "s1", "Speed", Some(7), &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 7 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = fetch_fft_via(&root, "nope", "Speed", None, &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_unknown_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_fft_via(&root, "s1", "NopeChannel", None, &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_averaging_none_and_max_succeed_and_produce_well_formed_idlf_bytes() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let none_bytes = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::None).unwrap();
        let max_bytes = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::Max).unwrap();

        // Assert — well-formed IDLF header on both; the whole point of R63 (3)
        // is that these no longer reject.
        for bytes in [&none_bytes, &max_bytes] {
            assert_eq!(&bytes[0..4], b"IDLF");
            assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 1);
            let bin_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            assert_eq!(bytes.len(), 16 + bin_count * 4);
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_malformed_params_zero_window_size_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let bad_params = SpectrogramParams { window_size: 0, hop_size: 0, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let err = fetch_fft_via(&root, "s1", "Speed", None, &bad_params, Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }
}
