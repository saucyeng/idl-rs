//! FRB wrappers for `idl_rs::session::handle` — the opaque session handle and
//! its output-shaped accessors. Parsing happens in Rust; Dart holds a
//! `RustOpaque<SessionHandle>` and pulls metadata / per-channel samples.

use flutter_rust_bridge::frb;

use crate::frb_generated::RustOpaque;

pub use idl_rs::session::handle::{
    ChannelInput, ChannelMeta, SessionHandle, SessionMeta, SessionMetaInput, SliceRole,
};
// `FitLap`/`FitSport` are re-exported (`pub use`) because the generated
// `frb_generated.rs` refers to the mirrors as `crate::session::FitLap` /
// `crate::session::FitSport`; `write_fit`/`FitOptions` are used only inside
// `export_fit_to_vec`, so they stay private — matching the pub-use-referenced /
// use-internal split on lines 9 and 13.
pub use idl_rs::export::{FitLap, FitSport};
use idl_rs::export::{write_fit, FitOptions};
use idl_rs::estimate::geometry::BikeGeometry;
use idl_rs::estimate::iekf::InitStd;
use idl_rs::estimate::noise::ImuNoise;
use idl_rs::estimate::process::ProcessNoiseConfig;
use idl_rs::estimate::run::{run as run_estimator, EstimatorConfig, EstimatorInput};
use idl_rs::session::ParseError;

// ---- Error crossing ----------------------------------------------------------------
//
// `idl_rs::ParseError` is a *data-carrying* enum (each variant holds a message).
// FRB renders data enums as `freezed` Dart classes, which would pull `freezed` +
// `build_runner` into the app — a build dependency this project deliberately
// avoids. So we cross a freezed-free pair instead: a plain unit enum `kind` +
// a `message` String. The Dart provider maps `kind` onto the typed exceptions
// in `app/lib/data/exceptions.dart`.

/// Discriminant for [`ParseFailure`]. A unit enum → a plain Dart enum (no freezed).
pub enum ParseErrorKind {
    InvalidMagicBytes,
    UnsupportedSchemaVersion,
    TruncatedRecord,
    Io,
}

/// Error returned by the `parse_session*` entry points.
pub struct ParseFailure {
    pub kind: ParseErrorKind,
    pub message: String,
}

impl From<ParseError> for ParseFailure {
    fn from(e: ParseError) -> Self {
        let (kind, message) = match e {
            ParseError::InvalidMagicBytes(m) => (ParseErrorKind::InvalidMagicBytes, m),
            ParseError::UnsupportedSchemaVersion(m) => (ParseErrorKind::UnsupportedSchemaVersion, m),
            ParseError::TruncatedRecord(m) => (ParseErrorKind::TruncatedRecord, m),
            ParseError::Io(m) => (ParseErrorKind::Io, m),
        };
        ParseFailure { kind, message }
    }
}

// ---- Mirrored cross-boundary types -------------------------------------------------

#[frb(mirror(SessionMeta))]
pub struct _SessionMeta {
    pub session_id: String,
    pub device_id: String,
    pub timestamp_utc_ms: i64,
    pub config_checksum: String,
    pub channel_count: u32,
    pub duration_ms: i64,
    pub truncation_warning: Option<String>,
}

#[frb(mirror(ChannelMeta))]
pub struct _ChannelMeta {
    pub channel_id: String,
    pub sample_rate_hz: f64,
    pub length: u32,
    pub is_event_driven: bool,
    pub synthesized: bool,
}

#[frb(mirror(SessionMetaInput))]
pub struct _SessionMetaInput {
    pub session_id: String,
    pub device_id: String,
    pub timestamp_utc_ms: i64,
    pub config_checksum: String,
}

#[frb(mirror(ChannelInput))]
pub struct _ChannelInput {
    pub channel_id: String,
    pub sample_rate_hz: f64,
    pub samples: Vec<f64>,
    pub sample_times_secs: Option<Vec<f64>>,
}

/// FIT sport classification (mirrors `idl_rs::export::FitSport`). Unit enum →
/// plain Dart enum (no freezed).
#[frb(mirror(FitSport))]
pub enum _FitSport {
    Generic,
    Running,
    Cycling,
    Motorcycling,
}

/// One lap's timing for FIT export (mirrors `idl_rs::export::FitLap`). All
/// fields are milliseconds.
#[frb(mirror(FitLap))]
pub struct _FitLap {
    pub start_ms: i64,
    pub end_ms: i64,
    pub elapsed_ms: i64,
}

// ---- Construction ------------------------------------------------------------------

/// Parse `.idl0` bytes into an owned session handle.
#[frb]
pub fn parse_session(bytes: Vec<u8>) -> Result<RustOpaque<SessionHandle>, ParseFailure> {
    SessionHandle::from_bytes(&bytes).map(RustOpaque::new).map_err(ParseFailure::from)
}

/// Read + parse an `.idl0` file path (Rust does the IO; no big buffer crosses FFI).
#[frb]
pub fn parse_session_from_path(path: String) -> Result<RustOpaque<SessionHandle>, ParseFailure> {
    SessionHandle::from_path(&path).map(RustOpaque::new).map_err(ParseFailure::from)
}

/// Build a handle from caller-parsed channels (GPX path).
#[frb]
pub fn session_from_channels(
    meta: SessionMetaInput,
    channels: Vec<ChannelInput>,
) -> RustOpaque<SessionHandle> {
    RustOpaque::new(SessionHandle::from_channels(meta, channels))
}

// ---- Accessors ---------------------------------------------------------------------

#[frb]
pub fn session_metadata(handle: RustOpaque<SessionHandle>) -> SessionMeta {
    handle.metadata()
}

#[frb]
pub fn session_channels(handle: RustOpaque<SessionHandle>) -> Vec<ChannelMeta> {
    handle.channels()
}

/// Resident sample-storage bytes for the handle — the input to the app's
/// byte-budgeted residency policy (§15.3).
#[frb]
pub fn session_resident_bytes(handle: RustOpaque<SessionHandle>) -> u64 {
    handle.resident_bytes()
}

#[frb]
pub fn channel_samples(handle: RustOpaque<SessionHandle>, channel_id: String) -> Vec<f64> {
    handle.channel_samples(&channel_id)
}

#[frb]
pub fn channel_sample_times(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
) -> Option<Vec<f64>> {
    handle.channel_sample_times(&channel_id)
}

/// Encode the session as a Garmin FIT activity (for Strava) and return the
/// bytes. `laps` is empty for a single whole-ride lap, or one entry per detected
/// lap (Strava renders them as splits). Heart rate + GPS are read from the
/// handle. Returns an error string on failure (only `NoGpsData` in practice —
/// the app gates export on GPS presence). Lives here so it shares the canonical
/// opaque `SessionHandle` type.
#[frb]
pub fn export_fit_to_vec(
    handle: RustOpaque<SessionHandle>,
    sport: FitSport,
    laps: Vec<FitLap>,
) -> Result<Vec<u8>, String> {
    let options = FitOptions {
        sport,
        laps: if laps.is_empty() { None } else { Some(laps) },
    };
    let mut buf = Vec::new();
    write_fit(&handle, &options, &mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

// ---- Phase 0 seam (master design §4) -----------------------------------------------
//
// Bounded views onto the handle's samples: a min/max scalar pair, an index-range
// materialization, and a time-windowed slice. These let consumers stop draining
// whole channels (channelDataProvider) — the f64 form crosses FFI only as the
// bounded result. `welch_tile` / `gps_track_tile` are reserved by the design and
// land with the fft/gps consumer rewire (Phase D-drain).

/// Finite (min, max) bounds for a channel — FFI shape for
/// [`SessionHandle::channel_min_max`]. Plain struct (no freezed).
pub struct ChannelBounds {
    pub min: f64,
    pub max: f64,
}

/// Finite Y-axis bounds for `channel_id`; `None` when absent or all-non-finite.
#[frb]
pub fn channel_min_max(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
) -> Option<ChannelBounds> {
    handle.channel_min_max(&channel_id).map(|(min, max)| ChannelBounds { min, max })
}

/// Physical samples for the half-open index window `[start, end)`; clamped to the
/// channel length, empty for an absent channel or empty window.
#[frb]
pub fn materialize_f64(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    start: u32,
    end: u32,
) -> Vec<f64> {
    handle.materialize_f64(&channel_id, start, end)
}

/// Result of [`slice_lap_into_store`]: the opaque storage token the chart
/// decimates by, plus the slice length (0 = no sample in window, nothing stored).
pub struct SliceLapResult {
    pub token: String,
    pub length: u32,
}

/// Slice `channel_id` to `[t0_secs, t1_secs]`, sample-0-rebase it, and upsert it
/// into the handle's derived store under a typed lap-slice key. `overlay` picks
/// the role (overlay vs main lap); `lap` is the lap number. Returns the storage
/// token + length; the slice never crosses FFI (spec §15.3 seam).
#[frb]
pub fn slice_lap_into_store(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    overlay: bool,
    lap: u32,
    t0_secs: f64,
    t1_secs: f64,
) -> SliceLapResult {
    let role = if overlay { SliceRole::Overlay } else { SliceRole::Main };
    let (token, length) = handle.slice_lap_into_store(&channel_id, role, lap, t0_secs, t1_secs);
    SliceLapResult { token, length: length as u32 }
}

/// Reclaim derived entries not backed by reality: keeps math outputs whose name
/// is in `live_sources` and lap slices whose source is live or a base channel,
/// dropping the rest (a deleted/renamed math channel's output + slices). Called
/// on the eval path with the current channel-name list (spec §4).
#[frb]
pub fn retain_derived(handle: RustOpaque<SessionHandle>, live_sources: Vec<String>) {
    handle.retain_derived(&live_sources);
}

/// Welch spectrum for `channel_id`, computed in the engine straight from the
/// retained handle ([`SessionHandle::welch_channel`]) — the channel's samples
/// never cross FFI, only the `WelchResult` does (Phase D-drain). Empty result for
/// an absent channel. Lives here, beside the other handle accessors, so it shares
/// the canonical `SessionHandle` opaque type (the FFT `WelchResult`/window enums
/// are mirrored in `crate::fft`).
#[frb]
#[allow(clippy::too_many_arguments)]
pub fn welch_channel(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    window: crate::fft::FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: crate::fft::Detrend,
    averaging: crate::fft::Averaging,
    scaling: crate::fft::Scaling,
) -> crate::fft::WelchResult {
    handle.welch_channel(&channel_id, window, nperseg, noverlap, detrend, averaging, scaling)
}

/// Windowed Welch spectrum for `channel_id` over `[t0_secs, t1_secs]`, computed
/// in the engine ([`SessionHandle::welch_channel_windowed`]) — only the
/// `WelchResult` crosses FFI. Lives here beside `welch_channel` so it shares the
/// canonical opaque `SessionHandle` type.
#[frb]
#[allow(clippy::too_many_arguments)]
pub fn welch_channel_windowed(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    t0_secs: f64,
    t1_secs: f64,
    window: crate::fft::FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: crate::fft::Detrend,
    averaging: crate::fft::Averaging,
    scaling: crate::fft::Scaling,
) -> crate::fft::WelchResult {
    handle.welch_channel_windowed(&channel_id, t0_secs, t1_secs, window, nperseg, noverlap, detrend, averaging, scaling)
}

/// Spectrogram for `channel_id` over `[t0_secs, t1_secs]`, computed in the engine
/// ([`SessionHandle::spectrogram_channel`]) — only the flat
/// [`SpectrogramResult`](crate::spectrogram::SpectrogramResult) crosses FFI.
/// `times_secs` in the result are absolute session seconds. Lives here beside the
/// other handle accessors so it shares the canonical `SessionHandle` opaque type
/// (the `SpectrogramResult` mirror is in `crate::spectrogram`).
#[frb]
#[allow(clippy::too_many_arguments)]
pub fn spectrogram_channel(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    t0_secs: f64,
    t1_secs: f64,
    window: crate::fft::FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: crate::fft::Detrend,
    scaling: crate::fft::Scaling,
) -> crate::spectrogram::SpectrogramResult {
    handle.spectrogram_channel(&channel_id, t0_secs, t1_secs, window, nperseg, noverlap, detrend, scaling)
}

/// Value-distribution histogram for `channel_id`, computed in the engine from
/// the retained handle ([`SessionHandle::histogram`]) — the channel's samples
/// never cross FFI, only the `HistogramResult` does. `bin_count` equal-width
/// bins; `symmetric` centres the range on zero (signed velocity distributions).
/// Empty result for an absent channel. Lives here beside the other handle
/// accessors so it shares the canonical `SessionHandle` opaque type (the
/// `HistogramResult` mirror is in `crate::histogram`).
#[frb]
pub fn channel_histogram(
    handle: RustOpaque<SessionHandle>,
    channel_id: String,
    bin_count: u32,
    symmetric: bool,
    range_min: Option<f64>,
    range_max: Option<f64>,
) -> crate::histogram::HistogramResult {
    // An explicit `[range_min, range_max]` shares the binning extent across an
    // overlay's series so their bin edges align; otherwise the engine derives
    // the range from the data (auto, or zero-centred when `symmetric`).
    let range = match (range_min, range_max) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    };
    handle.histogram(&channel_id, bin_count as usize, symmetric, range)
}

/// Evaluate a table against the session handle. `row_windows[r]` is row `r`'s
/// `(t0_secs, t1_secs)` and `row_has_window[r]` whether that window applies
/// (FRB has no `Option<(f64,f64)>` across a `Vec`, so the window is split into a
/// value vec + a presence vec). Only the small per-cell results cross FFI. Lives
/// here (not table.rs) so it shares the opaque `SessionHandle` type.
#[frb]
pub fn evaluate_table(
    handle: RustOpaque<SessionHandle>,
    table: crate::table::TableModel,
    row_windows: Vec<(f64, f64)>,
    row_has_window: Vec<bool>,
) -> Vec<Vec<crate::table::CellResult>> {
    let windows: Vec<Option<(f64, f64)>> = row_windows
        .into_iter()
        .zip(row_has_window)
        .map(|(w, has)| if has { Some(w) } else { None })
        .collect();
    idl_rs::table::evaluate_table(&handle, &table, &windows)
}

/// Multi-session table evaluation. `handles` is the distinct session pool;
/// `row_handle_idx[r]` selects row `r`'s handle. `baseline_row` (when present)
/// is the row `main({col[]})` reads from. See `idl_rs::table::evaluate_table_multi`.
#[frb]
pub fn evaluate_table_multi(
    handles: Vec<RustOpaque<SessionHandle>>,
    row_handle_idx: Vec<u32>,
    table: crate::table::TableModel,
    row_windows: Vec<(f64, f64)>,
    row_has_window: Vec<bool>,
    baseline_row: Option<u32>,
) -> Vec<Vec<crate::table::CellResult>> {
    let refs: Vec<&SessionHandle> = handles.iter().map(|h| &**h).collect();
    let row_handles: Vec<usize> = row_handle_idx.iter().map(|&i| i as usize).collect();
    let windows: Vec<Option<(f64, f64)>> = row_windows
        .into_iter()
        .zip(row_has_window)
        .map(|(w, has)| if has { Some(w) } else { None })
        .collect();
    idl_rs::table::evaluate_table_multi(
        &refs,
        &row_handles,
        &table,
        &windows,
        baseline_row.map(|r| r as usize),
    )
}

/// Batch variance: each target lap vs one reference lap, in time (mode 0) or
/// distance (mode 1). Target lap fields are parallel arrays (FRB has no
/// `Vec<struct-with-opaque>`); each target's handle is in `target_handles`.
/// Returns one delta series per target, in order. See
/// `idl_rs::variance::variance_traces`.
#[frb]
#[allow(clippy::too_many_arguments)]
pub fn variance_traces(
    reference: RustOpaque<SessionHandle>,
    reference_lap_start_ms: f64,
    reference_lap_end_ms: f64,
    reference_lap_start_uniform_sec: f64,
    target_handles: Vec<RustOpaque<SessionHandle>>,
    target_lap_start_ms: Vec<f64>,
    target_lap_end_ms: Vec<f64>,
    target_lap_start_uniform_sec: Vec<f64>,
    target_window_start_sec: Vec<f64>,
    target_window_end_sec: Vec<f64>,
    channel_id: String,
    mode: u32,
) -> Vec<Vec<f64>> {
    use idl_rs::variance::{variance_traces as core_traces, LapRef, VarianceMode};
    let reference_ref = LapRef {
        handle: &reference,
        lap_start_ms: reference_lap_start_ms,
        lap_end_ms: reference_lap_end_ms,
        lap_start_uniform_sec: reference_lap_start_uniform_sec,
        window_start_sec: 0.0,
        window_end_sec: 0.0,
    };
    let targets: Vec<LapRef> = (0..target_handles.len())
        .map(|i| LapRef {
            handle: &target_handles[i],
            lap_start_ms: target_lap_start_ms[i],
            lap_end_ms: target_lap_end_ms[i],
            lap_start_uniform_sec: target_lap_start_uniform_sec[i],
            window_start_sec: target_window_start_sec[i],
            window_end_sec: target_window_end_sec[i],
        })
        .collect();
    let mode = if mode == 1 { VarianceMode::Distance } else { VarianceMode::Time };
    core_traces(&reference_ref, &targets, &channel_id, mode)
}

// ---- Suspension estimator (offline geometry-constrained IEKF) -----------------------
//
// Runs the `idl_rs::estimate` engine over the retained handle and stores its
// per-sample outputs into the handle's math store — same metadata-only crossing as
// `eval_math_into_store`, so the chart decimates the results by id without samples
// ever crossing FFI. Geometry is the engine reference bike (coarse mounts
// auto-refined per session); the per-session geometry store is deferred, so the app
// supplies only the live-tunable filter parameters via [`SuspensionConfig`].

/// Flat, FFI-friendly mirror of the engine's `EstimatorConfig` — the full filter
/// tuning surface the app exposes for live (hot-reload) tuning. Start from
/// [`suspension_config_default`] and override fields.
pub struct SuspensionConfig {
    /// Estimate steering states (M2a: false — wheels first).
    pub estimate_steering: bool,
    /// Measurement-update iterations (1 = plain EKF).
    pub iekf_iters: u32,
    /// Stationary-detector window, samples.
    pub zupt_window: u32,
    /// Stationary accel-magnitude std threshold, m/s².
    pub zupt_accel_std: f64,
    /// Stationary mean gyro-magnitude threshold, rad/s.
    pub zupt_gyro_thresh: f64,
    /// ZUPT velocity pseudo-measurement std, m/s.
    pub zupt_sigma: f64,
    /// ZARU bias pseudo-measurement std, rad/s.
    pub zaru_sigma: f64,
    /// Gravity-leveling tilt residual std.
    pub gravity_sigma: f64,
    /// GPS velocity std, m/s.
    pub gps_sigma: f64,
    /// Sag-prior std, m (deliberately loose — a weak anti-drift tether).
    pub sag_sigma: f64,
    /// Travel-barrier std, m (smaller ⇒ stiffer `[0, max]` wall).
    pub barrier_sigma: f64,
    /// IMU0 specific-force magnitude below which a sample is airborne, m/s².
    pub airborne_accel_thresh: f64,
    /// Max lever-compensated diff-accel magnitude (per unsprung IMU) for a sample to
    /// still count as airborne, m/s² — the relative free-fall veto. A grounded but
    /// terrain-driven wheel has a large diff-accel even when the chassis is light, so
    /// requiring it small stops rough ground from snapping travel to topout. Larger ⇒
    /// looser; smaller ⇒ stricter.
    pub airborne_diff_thresh: f64,
    /// Topout-reference std applied on airborne samples (the travel-DC anchor), m.
    pub topout_sigma: f64,
    /// GPS output latency, s — a fix logged at `t` describes the bike at
    /// `t − latency` (the module's internal filter delays its solution). Tune
    /// against hard-braking events.
    pub gps_latency_s: f64,
    /// Minimum GPS horizontal speed for a fix to be used, m/s (course is noise
    /// below walking pace; ZUPT covers stationary).
    pub gps_min_speed_mps: f64,
    /// Re-fit the unsprung mounts' tilt against this session's first stationary
    /// window (default false — the geometry carries flat-ground-calibrated mounts;
    /// a ride's parked lean / flopped bars would be absorbed as mount error).
    pub refine_mounts: bool,
    /// Run the RTS backward pass over the wheel travel/velocity chains (default
    /// true) — distributes each topout/stop anchor over the interval before it
    /// instead of correcting only forward. `false` = raw causal filter.
    pub smooth: bool,
    /// IMU0 gyro angle random walk, rad/√s.
    pub gyro_arw: f64,
    /// IMU0 accel velocity random walk, (m/s)/√s.
    pub accel_vrw: f64,
    /// IMU0 gyro-bias random walk, (rad/s)/√s.
    pub gyro_bias_rw: f64,
    /// IMU0 accel-bias random walk, (m/s²)/√s.
    pub accel_bias_rw: f64,
    /// IMU1 (front-unsprung) gyro-bias random walk, (rad/s)/√s.
    pub gyro1_bias_rw: f64,
    /// IMU2 (rear-unsprung) gyro-bias random walk, (rad/s)/√s.
    pub gyro2_bias_rw: f64,
    /// Wheel-velocity driving noise, (m/s²)/√s.
    pub wheel_vel_rw: f64,
    /// Wheel-travel regularizing noise, (m)/√s.
    pub wheel_pos_rw: f64,
    /// Steer-rate driving noise, (rad/s²)/√s.
    pub steer_rate_rw: f64,
    /// Initial attitude-tilt 1-σ prior, rad.
    pub init_attitude: f64,
    /// Initial velocity 1-σ prior, m/s.
    pub init_velocity: f64,
    /// Initial gyro-bias 1-σ prior, rad/s.
    pub init_gyro_bias: f64,
    /// Initial accel-bias 1-σ prior, m/s².
    pub init_accel_bias: f64,
    /// Initial wheel-travel 1-σ prior, m.
    pub init_wheel_travel: f64,
    /// Initial wheel-velocity 1-σ prior, m/s.
    pub init_wheel_velocity: f64,
    /// Initial steer-angle 1-σ prior, rad.
    pub init_steer_angle: f64,
    /// Initial steer-rate 1-σ prior, rad/s.
    pub init_steer_rate: f64,
}

impl From<SuspensionConfig> for EstimatorConfig {
    fn from(c: SuspensionConfig) -> Self {
        EstimatorConfig {
            estimate_steering: c.estimate_steering,
            process_noise: ProcessNoiseConfig {
                imu0: ImuNoise {
                    gyro_arw: c.gyro_arw,
                    accel_vrw: c.accel_vrw,
                    gyro_bias_rw: c.gyro_bias_rw,
                    accel_bias_rw: c.accel_bias_rw,
                },
                gyro1_bias_rw: c.gyro1_bias_rw,
                gyro2_bias_rw: c.gyro2_bias_rw,
                wheel_vel_rw: c.wheel_vel_rw,
                wheel_pos_rw: c.wheel_pos_rw,
                steer_rate_rw: c.steer_rate_rw,
            },
            init_std: InitStd {
                attitude: c.init_attitude,
                velocity: c.init_velocity,
                gyro_bias: c.init_gyro_bias,
                accel_bias: c.init_accel_bias,
                wheel_travel: c.init_wheel_travel,
                wheel_velocity: c.init_wheel_velocity,
                steer_angle: c.init_steer_angle,
                steer_rate: c.init_steer_rate,
            },
            iekf_iters: c.iekf_iters as usize,
            zupt_window: c.zupt_window as usize,
            zupt_accel_std: c.zupt_accel_std,
            zupt_gyro_thresh: c.zupt_gyro_thresh,
            zupt_sigma: c.zupt_sigma,
            zaru_sigma: c.zaru_sigma,
            gravity_sigma: c.gravity_sigma,
            gps_sigma: c.gps_sigma,
            sag_sigma: c.sag_sigma,
            barrier_sigma: c.barrier_sigma,
            airborne_accel_thresh: c.airborne_accel_thresh,
            airborne_diff_thresh: c.airborne_diff_thresh,
            topout_sigma: c.topout_sigma,
            gps_latency_s: c.gps_latency_s,
            gps_min_speed_mps: c.gps_min_speed_mps,
            refine_mounts: c.refine_mounts,
            smooth: c.smooth,
            // Bounds-only: the app surface deliberately does not expose a sag-prior
            // toggle (the recovered velocity, not absolute travel DC, is the
            // deliverable — see `EstimatorConfig::use_sag_prior`). Hardcoding false here
            // keeps the FFI mirror unchanged (no codegen) while making bounds-only the
            // app's behaviour; flip the engine default to A/B from a test or the CLI.
            use_sag_prior: false,
        }
    }
}

/// Metadata of a stored suspension-estimator run. The per-sample outputs stay in
/// the handle's math store under `channel_ids`; the chart reads them by id.
pub struct SuspensionEstimateMeta {
    /// Ids of the channels stored this run (front/rear travel + velocity).
    pub channel_ids: Vec<String>,
    /// Number of samples per output channel.
    pub length: u32,
    /// Output sample rate, Hz (= IMU0 rate).
    pub sample_rate_hz: f64,
    /// Front dynamic sag — median front travel over riding samples, mm (None if
    /// never riding). An emergent output, not an input (see the engine doc).
    pub front_dynamic_sag_mm: Option<f64>,
    /// Rear dynamic sag, mm (None if never riding or no rear suspension).
    pub rear_dynamic_sag_mm: Option<f64>,
}

/// Failure from [`estimate_suspension_into_store`].
pub struct SuspensionEstimateFailure {
    pub message: String,
}

/// Default filter tuning (the engine's reference defaults) — the app's starting
/// point for live tuning. See [`SuspensionConfig`].
#[frb]
pub fn suspension_config_default() -> SuspensionConfig {
    let c = EstimatorConfig::default();
    let p = c.process_noise;
    let s = c.init_std;
    SuspensionConfig {
        estimate_steering: c.estimate_steering,
        iekf_iters: c.iekf_iters as u32,
        zupt_window: c.zupt_window as u32,
        zupt_accel_std: c.zupt_accel_std,
        zupt_gyro_thresh: c.zupt_gyro_thresh,
        zupt_sigma: c.zupt_sigma,
        zaru_sigma: c.zaru_sigma,
        gravity_sigma: c.gravity_sigma,
        gps_sigma: c.gps_sigma,
        sag_sigma: c.sag_sigma,
        barrier_sigma: c.barrier_sigma,
        airborne_accel_thresh: c.airborne_accel_thresh,
        airborne_diff_thresh: c.airborne_diff_thresh,
        topout_sigma: c.topout_sigma,
        gps_latency_s: c.gps_latency_s,
        gps_min_speed_mps: c.gps_min_speed_mps,
        refine_mounts: c.refine_mounts,
        smooth: c.smooth,
        gyro_arw: p.imu0.gyro_arw,
        accel_vrw: p.imu0.accel_vrw,
        gyro_bias_rw: p.imu0.gyro_bias_rw,
        accel_bias_rw: p.imu0.accel_bias_rw,
        gyro1_bias_rw: p.gyro1_bias_rw,
        gyro2_bias_rw: p.gyro2_bias_rw,
        wheel_vel_rw: p.wheel_vel_rw,
        wheel_pos_rw: p.wheel_pos_rw,
        steer_rate_rw: p.steer_rate_rw,
        init_attitude: s.attitude,
        init_velocity: s.velocity,
        init_gyro_bias: s.gyro_bias,
        init_accel_bias: s.accel_bias,
        init_wheel_travel: s.wheel_travel,
        init_wheel_velocity: s.wheel_velocity,
        init_steer_angle: s.steer_angle,
        init_steer_rate: s.steer_rate,
    }
}

/// Run the offline suspension-kinematics estimator over the retained session handle
/// and store its per-sample outputs into the handle's math store: front (and, with
/// rear suspension, rear) **wheel travel in mm** and **wheel velocity in mm/s**.
/// Only metadata crosses FFI — the chart decimates the stored channels by id (spec
/// §15.3 seam), exactly like `eval_math_into_store`. Geometry is the engine
/// reference bike, carrying flat-ground-calibrated unsprung mounts (per-session
/// tilt refitting is the opt-in `refine_mounts`); `config` carries the live-tunable
/// filter parameters, including GPS-velocity aiding (latency, min-speed gate) and
/// the RTS wheel smoother (`smooth`, default on). Errors only when the session has
/// no IMU0 channels.
#[frb]
pub fn estimate_suspension_into_store(
    handle: RustOpaque<SessionHandle>,
    config: SuspensionConfig,
) -> Result<SuspensionEstimateMeta, SuspensionEstimateFailure> {
    let input = EstimatorInput::from_lookup(&*handle).ok_or_else(|| SuspensionEstimateFailure {
        message: "session has no IMU0 channels to estimate from".to_string(),
    })?;
    let geometry = BikeGeometry::reference_bike();
    let est = run_estimator(&input, &geometry, &config.into());

    let rate = if est.dt > 0.0 { 1.0 / est.dt } else { 0.0 };
    let to_mm = |v: &[f64]| v.iter().map(|x| x * 1000.0).collect::<Vec<f64>>();
    let mut channel_ids = Vec::new();

    handle.store_math("Front travel (mm)", rate, to_mm(&est.front_travel));
    channel_ids.push("Front travel (mm)".to_string());
    handle.store_math("Front velocity (mm/s)", rate, to_mm(&est.front_velocity));
    channel_ids.push("Front velocity (mm/s)".to_string());
    if est.schema.is_active("w_r") {
        handle.store_math("Rear travel (mm)", rate, to_mm(&est.rear_travel));
        channel_ids.push("Rear travel (mm)".to_string());
        handle.store_math("Rear velocity (mm/s)", rate, to_mm(&est.rear_velocity));
        channel_ids.push("Rear velocity (mm/s)".to_string());
    }

    Ok(SuspensionEstimateMeta {
        channel_ids,
        length: est.len() as u32,
        sample_rate_hz: rate,
        front_dynamic_sag_mm: est.front_dynamic_sag().map(|s| s * 1000.0),
        rear_dynamic_sag_mm: est.rear_dynamic_sag().map(|s| s * 1000.0),
    })
}
