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
use crate::session_source::{load_session, load_session_handle, resolve_lap_window};
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
    find_channel_in(&session.channels, channel_id)
}

/// [`find_channel`]'s body, over a plain channel slice — `fetch_fft_via`
/// looks a channel up in a [`idl_rs::session::handle::SessionHandle`]'s
/// [`idl_rs::session::handle::SessionHandle::channel_data`] rather than a
/// [`Session`], so this is the one place both paths share the lookup and
/// its `not_found` message.
fn find_channel_in<'a>(channels: &'a [Channel], channel_id: &str) -> Result<&'a Channel, IpcError> {
    channels
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

/// Maps [`idl_rs::fft::FftError`] to the IPC `invalid_argument` shape
/// (ruling R76). Both variants are checked before `welch()` runs, so no
/// partial or `NaN`/`Infinity` bytes are ever produced on this path.
fn map_fft_error(e: idl_rs::fft::FftError) -> IpcError {
    match e {
        idl_rs::fft::FftError::NoneRequiresOneSegment { segments } => IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("averaging \"none\" requires exactly one segment, got {segments}"),
            serde_json::json!({ "segments": segments }),
        ),
        idl_rs::fft::FftError::InvalidSampleRate => IpcError::new(
            IpcErrorKind::InvalidArgument,
            "channel has too few samples or duplicate timestamps to derive a sample rate",
        ),
    }
}

/// Transport-agnostic core of `fetch_fft` (C3 §3.6, ruling R63 (3), R76,
/// R83). Loads the channel via [`load_session_handle`]/[`find_channel_in`].
/// `lap: None` takes the whole channel (`Channel::materialize`); `lap:
/// Some(n)` resolves `n`'s recording-time window via
/// [`crate::session_source::resolve_lap_window`] and takes only
/// `idl_rs::session::handle::SessionHandle::slice_by_time`'s samples in that
/// window — an unknown `n` surfaces `resolve_lap_window`'s own
/// `invalid_argument`/`detail: { "lap": n }` before any FFT runs. Either way,
/// both of R76's core guards run against the *resulting* window, never the
/// whole channel: `averaging: none` is validated against the sliced sample
/// count (see `idl_rs::fft::check_none_averaging_segments`), and
/// `sample_rate_hz` is derived from the sliced window's own `t_us` (see
/// `idl_rs::fft::effective_rate_hz_from_t_us`) using the same
/// inclusive-window boundary as `slice_by_time` itself (ruling R85) — a
/// 1–2-sample lap window (a degenerate lap boundary) fails this guard
/// instead of silently clamping to a one-segment, possibly-`NaN` spectrum.
/// The whole-channel path (`lap: None`) is unaffected: its window *is* the
/// whole channel, so both guards already ran against the same `t_us` before
/// this ruling. Both core validations run before any FFT executes.
/// `not_found`: unknown `session_id` or `channel`. `invalid_argument`: an
/// unknown `lap`, `params` that fail [`resolve_spectrogram_params`]'s
/// validation, `averaging: none` with more than one segment (`detail: {
/// "segments": n }`), or a window with too few samples/duplicate timestamps
/// to derive a sample rate.
pub fn fetch_fft_via(
    data_dir: &Path,
    session_id: &str,
    channel: &str,
    lap: Option<u32>,
    params: &SpectrogramParams,
    averaging: Averaging,
) -> Result<Vec<u8>, IpcError> {
    let handle = load_session_handle(data_dir, session_id)?;
    let ch = find_channel_in(handle.channel_data(), channel)?;
    let (window, detrend, scaling, window_size, noverlap) = resolve_spectrogram_params(params)?;
    let (samples, window_t_us) = match lap {
        None => (ch.materialize(), None),
        Some(n) => {
            let (t0_secs, t1_secs) = resolve_lap_window(data_dir, session_id, n)?;
            let sliced = handle.slice_by_time(channel, t0_secs, t1_secs);
            let t_us = slice_t_us_by_time(&ch.t_us, t0_secs, t1_secs);
            (sliced, Some(t_us))
        }
    };
    idl_rs::fft::check_none_averaging_segments(&averaging, window_size, noverlap, samples.len())
        .map_err(map_fft_error)?;
    let rate_t_us = window_t_us.as_deref().unwrap_or(&ch.t_us);
    let sample_rate_hz = idl_rs::fft::effective_rate_hz_from_t_us(rate_t_us).map_err(map_fft_error)?;
    let result = idl_rs::fft::welch(samples, sample_rate_hz, window, window_size, noverlap, detrend, averaging, scaling);
    Ok(idl_rs::fft_wire::encode_fft_idlf(&result.values, sample_rate_hz))
}

/// Returns the `t_us` entries whose sample time falls in `[t0_secs,
/// t1_secs]` (ruling R85). Mirrors `idl_rs`'s private `slice_channel_by_time`
/// boundary convention exactly — round each end to microseconds, then
/// `partition_point` for `< t0_us` / `<= t1_us` — so `check_none_averaging_
/// segments`/`effective_rate_hz_from_t_us` see the identical window that
/// `SessionHandle::slice_by_time` sliced `samples` from; not a core change,
/// since `t_us` is already `pub` on `Channel` and this performs no DSP, only
/// the same index-window arithmetic `fetch_fft_via` already relies on.
fn slice_t_us_by_time(t_us: &[i64], t0_secs: f64, t1_secs: f64) -> Vec<i64> {
    if t_us.is_empty() || t1_secs < t0_secs {
        return Vec::new();
    }
    let t0_us = (t0_secs * 1e6).round() as i64;
    let t1_us = (t1_secs * 1e6).round() as i64;
    let lo = t_us.partition_point(|&t| t < t0_us);
    let hi = t_us.partition_point(|&t| t <= t1_us);
    if lo >= hi {
        return Vec::new();
    }
    t_us[lo..hi].to_vec()
}

/// Fetches one channel's FFT spectrum as `IDLF` v1 bytes (C3 §3.6, ruling
/// R63 (3), R76, R83). `lap: null` is the whole channel; `lap: n` is that
/// lap's recording-time window, resolved from `session.json`'s `laps[]` (see
/// [`fetch_fft_via`]).
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
    use idl_rs::store::session_json::{empty_session_json, write_session_json, LapJson};
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

    /// Writes `s1`'s `session.json` with three contiguous, non-overlapping
    /// laps over the 128-sample/64 Hz `Speed` channel `seed_session` writes:
    /// lap 1 is samples 0..=63, lap 2 is samples 64..=73 (10 samples —
    /// deliberately fewer than `fft_params()`'s `window_size: 32`, so its
    /// spectrum has a visibly smaller `bin_count`), lap 3 is 74..=127.
    fn seed_laps_for_s1(root: &std::path::Path) {
        let dt = 1.0 / 64.0;
        let lap = |n: u32, start_idx: usize, end_idx: usize| LapJson {
            lap_number: n,
            start_timestamp_ms: (start_idx as f64 * dt * 1000.0).round() as i64,
            end_timestamp_ms: (end_idx as f64 * dt * 1000.0).round() as i64,
            raw_elapsed_ms: ((end_idx - start_idx) as f64 * dt * 1000.0).round() as i64,
            lap_time_ms: ((end_idx - start_idx) as f64 * dt * 1000.0).round() as i64,
            start_time_secs: start_idx as f64 * dt,
            end_time_secs: end_idx as f64 * dt,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        };
        let mut doc = empty_session_json("s1");
        doc.laps = vec![lap(1, 0, 63), lap(2, 64, 73), lap(3, 74, 127)];
        write_session_json(root, "s1", &doc, None).unwrap();
    }

    #[test]
    fn fetch_fft_via_lap_some_known_lap_bin_count_differs_from_whole_channel() {
        // Arrange — averaging: mean sidesteps R76's one-segment rule; lap 2's
        // 10 samples are fewer than fft_params()'s window_size: 32, so
        // idl_rs::fft::resolve_seg clamps its segment to the slice's own
        // length, shrinking bin_count relative to the whole 128-sample
        // channel's.
        let root = temp_root();
        seed_session(&root);
        seed_laps_for_s1(&root);

        // Act
        let whole = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::Mean).unwrap();
        let lap2 = fetch_fft_via(&root, "s1", "Speed", Some(2), &fft_params(), Averaging::Mean).unwrap();

        // Assert
        let bin_count = |bytes: &[u8]| u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(bin_count(&whole), 17); // 32 / 2 + 1
        assert_eq!(bin_count(&lap2), 6); // 10 / 2 + 1 (resolve_seg clamps to the 10-sample slice)
        assert_ne!(bin_count(&whole), bin_count(&lap2));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_lap_unknown_number_invalid_argument_with_detail_lap() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        seed_laps_for_s1(&root);

        // Act
        let err = fetch_fft_via(&root, "s1", "Speed", Some(99), &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 99 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_lap_none_byte_identical_whether_or_not_session_json_has_laps() {
        // Arrange — `lap: None` must ignore `session.json` entirely: the
        // pre-R83 behaviour (no `session.json` at all) and the post-R83
        // behaviour (one with real `laps[]`) must produce identical bytes.
        let root = temp_root();
        seed_session(&root);

        // Act
        let before = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::Mean).unwrap();
        seed_laps_for_s1(&root);
        let after = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::Mean).unwrap();

        // Assert
        assert_eq!(before, after);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_lap_window_multi_segment_under_none_averaging_invalid_argument_with_sliced_segment_count() {
        // Arrange — lap 3 is samples 74..=127 (54 samples). Under
        // `fft_params()`'s window_size: 32/hop_size: 16 (noverlap 16), that
        // slice segments into 2 windows (segment_count(32, 16, 54) == 2) —
        // more than R76 allows for averaging: none. Proves ordering: the
        // check runs against the *lap-sliced* length (2 segments), not the
        // whole 128-sample channel's (7, the pre-existing whole-channel
        // multi-segment test's count).
        let root = temp_root();
        seed_session(&root);
        seed_laps_for_s1(&root);

        // Act
        let err = fetch_fft_via(&root, "s1", "Speed", Some(3), &fft_params(), Averaging::None).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "segments": 2 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Seeds `s1` with an event-driven `Dup` channel whose `t_us` is `[0,
    /// 100_000, 100_000, 200_000, 300_000]` — a duplicate pair at indices
    /// 1–2 — plus one `session.json` lap per degenerate window a lap
    /// boundary could produce (ruling R85): lap 1 is `[0.0, 0.0]` s
    /// (index 0 only, 1 sample); lap 2 is `[0.1, 0.1]` s (indices 1–2, the
    /// duplicate-timestamp pair, 2 samples with a single zero-length gap).
    fn seed_degenerate_lap_windows_for_s1(root: &std::path::Path) {
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "b".repeat(64),
            channels: vec![Channel {
                channel_id: "Dup".to_string(),
                t_us: vec![0, 100_000, 100_000, 200_000, 300_000],
                t_recorded_us: None,
                nominal_rate_hz: 0.0,
                column: RawColumn::F64(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
                source_kind: "wheel".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();

        let lap = |n: u32, t: f64| LapJson {
            lap_number: n,
            start_timestamp_ms: (t * 1000.0).round() as i64,
            end_timestamp_ms: (t * 1000.0).round() as i64,
            raw_elapsed_ms: 0,
            lap_time_ms: 0,
            start_time_secs: t,
            end_time_secs: t,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        };
        let mut doc = empty_session_json("s1");
        doc.laps = vec![lap(1, 0.0), lap(2, 0.1)];
        write_session_json(root, "s1", &doc, None).unwrap();
    }

    #[test]
    fn fetch_fft_via_lap_window_single_sample_invalid_argument() {
        // Arrange — lap 1 slices to exactly one sample (index 0); R85: the
        // rate guard must run on the window's own `t_us`, not the whole
        // (5-sample, `len() >= 2`) channel's, so this must fail instead of
        // reaching `welch()`'s Hann weights (which would divide by
        // `n - 1 == 0`).
        let root = temp_root();
        seed_degenerate_lap_windows_for_s1(&root);
        let params = SpectrogramParams { window_size: 0, hop_size: 0, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let err = fetch_fft_via(&root, "s1", "Dup", Some(1), &params, Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_lap_window_two_duplicate_timestamp_samples_invalid_argument() {
        // Arrange — lap 2 slices to the duplicate-timestamp pair (indices
        // 1–2, both `t_us == 100_000`): the window's single gap is zero, so
        // `effective_rate_hz_from_t_us` on the *window's* `t_us` must reject
        // it, the same way it already rejects a whole duplicate-timestamp
        // channel.
        let root = temp_root();
        seed_degenerate_lap_windows_for_s1(&root);
        let params = SpectrogramParams { window_size: 0, hop_size: 0, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let err = fetch_fft_via(&root, "s1", "Dup", Some(2), &params, Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_lap_window_healthy_multi_sample_output_unchanged_by_r85() {
        // Arrange — lap 2 of `seed_laps_for_s1` (10 samples, 64 Hz uniform
        // spacing): its own `t_us` yields the same 64 Hz rate as the whole
        // channel's, so slicing the rate guard's input (R85) must not move
        // this healthy window's bytes at all. Cross-checked against a
        // manual `welch()` call built directly from the handle's own slice
        // and the window's own derived rate, not just re-run through
        // `fetch_fft_via` (which would not catch a rate regression).
        let root = temp_root();
        seed_session(&root);
        seed_laps_for_s1(&root);

        // Act
        let got = fetch_fft_via(&root, "s1", "Speed", Some(2), &fft_params(), Averaging::Mean).unwrap();

        let handle = load_session_handle(&root, "s1").unwrap();
        let (t0, t1) = resolve_lap_window(&root, "s1", 2).unwrap();
        let window_samples = handle.slice_by_time("Speed", t0, t1);
        let want_rate_hz = 64.0;
        let want = idl_rs::fft::welch(
            window_samples, want_rate_hz, FftWindow::Hann, 32, 16, Detrend::Mean, Averaging::Mean, Scaling::Density,
        );
        let want_bytes = idl_rs::fft_wire::encode_fft_idlf(&want.values, want_rate_hz);

        // Assert
        assert_eq!(got, want_bytes);

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
    fn fetch_fft_via_averaging_none_full_record_window_and_max_succeed_and_produce_well_formed_idlf_bytes() {
        // Arrange — "Speed" has 128 samples; None needs a window covering
        // the whole record (ruling R76) so window_size/hop_size = 128 here,
        // distinct from Max's window_size: 32 (Max has no segment-count limit).
        let root = temp_root();
        seed_session(&root);
        let none_params = SpectrogramParams { window_size: 128, hop_size: 128, window: WindowToken::Hann, detrend: DetrendToken::Mean, scaling: ScalingToken::Density };

        // Act
        let none_bytes = fetch_fft_via(&root, "s1", "Speed", None, &none_params, Averaging::None).unwrap();
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
    fn fetch_fft_via_averaging_none_with_multi_segment_window_invalid_argument_with_segment_count() {
        // Arrange — "Speed" has 128 samples; fft_params()'s window_size: 32,
        // hop_size: 16 (noverlap 16) segments it into 7 windows, so
        // averaging: none must be rejected rather than silently keeping
        // only the first segment's power (ruling R76).
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_fft_via(&root, "s1", "Speed", None, &fft_params(), Averaging::None).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "segments": 7 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_channel_with_one_sample_invalid_argument() {
        // Arrange — a single-sample channel has no gap to derive a rate from
        let root = temp_root();
        let session = Session {
            session_id: "s2".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "b".repeat(64),
            channels: vec![Channel {
                channel_id: "Speed".to_string(),
                t_us: vec![0],
                t_recorded_us: None,
                nominal_rate_hz: 64.0,
                column: RawColumn::F64(vec![1.0]),
                source_kind: "wheel".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let err = fetch_fft_via(&root, "s2", "Speed", None, &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_fft_via_channel_with_duplicate_timestamps_invalid_argument() {
        // Arrange — every sample stamped at the same t_us => every gap is 0
        let root = temp_root();
        let n = 4usize;
        let session = Session {
            session_id: "s3".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "c".repeat(64),
            channels: vec![Channel {
                channel_id: "Speed".to_string(),
                t_us: vec![0; n],
                t_recorded_us: None,
                nominal_rate_hz: 64.0,
                column: RawColumn::F64(vec![1.0; n]),
                source_kind: "wheel".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let err = fetch_fft_via(&root, "s3", "Speed", None, &fft_params(), Averaging::Mean).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

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
