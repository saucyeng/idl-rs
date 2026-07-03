//! The estimator entry point: `run(input, geometry, config) -> StateEstimate`
//! (design §1). Pure core — typed sample arrays in, a state trajectory + ledger out;
//! the `ChannelLookup`/FFI adapter that pulls named session channels into
//! [`EstimatorInput`] is a thin bridge-layer concern, kept out of here.
//!
//! Pipeline: stationary detection → orientation/bias pre-step → forward IEKF over
//! the shared factors (diff-accel wheel drive, ZUPT/ZARU, gravity-leveling, sag,
//! barrier, GPS velocity) → observability ledger. The wheel-drive controls are the
//! projected differential specific forces (design §5), built per sample here.

use crate::estimate::detect::stationary_context;
use crate::math::eval::ChannelLookup;
use crate::parse::records::IMU_CHANNEL_NAMES;
use crate::estimate::geometry::BikeGeometry;
use crate::estimate::iekf::{FilterState, Iekf, InitStd};
use crate::estimate::ledger::ObservabilityLedger;
use crate::estimate::measurements::gravity::GravityLeveling;
use crate::estimate::measurements::gps::GpsVelocity;
use crate::estimate::measurements::prior::{SagPrior, TopoutReference, TravelBarrier, Wheel};
use crate::estimate::measurements::zupt::{GyroBias, ZeroAngularRate, ZeroVelocity, ZeroWheelVelocity};
use crate::estimate::model::{ImuInput, SampleContext};
use crate::estimate::orient::{fit_from_window, refine_mount_from_window};
use crate::estimate::process::{wheel_drive_accel, MtbProcess, ProcessNoiseConfig, GRAVITY};
use crate::estimate::schema::StateSchema;
use crate::estimate::smooth;
use crate::estimate::state::MtbState;
use crate::rotation::lever_arm_accel;
use nalgebra::{UnitQuaternion, Vector3};

/// One IMU's synchronized, uniform-rate samples in the **chassis frame** (the
/// caller has applied the per-IMU mount). `gyro` rad/s, `accel` (specific force) m/s².
#[derive(Debug, Clone)]
pub struct ImuSeries {
    pub gyro: Vec<Vector3<f64>>,
    pub accel: Vec<Vector3<f64>>,
}

/// One GPS velocity fix on the recording timeline, built from the log's
/// `GPS_SpeedKmh` + `GPS_Heading` channels (speed over ground + course — the module
/// has no vertical velocity channel). The runner maps `time_s` onto the IMU sample
/// timeline after subtracting the configured GPS output latency
/// ([`EstimatorConfig::gps_latency_s`]).
#[derive(Debug, Clone, Copy)]
pub struct GpsSample {
    /// Fix epoch on the recording timeline (the GPS record's event time), s.
    pub time_s: f64,
    /// Nav-frame velocity, m/s — X = north, Y = west, Z = up; `z` is always 0
    /// (no vertical channel; the GPS factor is horizontal-only).
    pub velocity: Vector3<f64>,
}

/// Degrees-per-second → radians-per-second (gyro channels are stored in `dps`).
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Pulls one IMU's six channels (`IMU{n}_{Accel,Gyro}{X,Y,Z}`) from a lookup and
/// converts to SI chassis-frame vectors: accel `g → m/s²`, gyro `dps → rad/s`.
/// Returns the series + its sample rate (Hz), or None if **no** axis is present.
///
/// **Axis-tolerant:** an absent axis is zero-filled rather than rejecting the whole
/// IMU. This is safe when the missing axis is orthogonal to the suspension-travel
/// plane (e.g. some loggers omit the unsprung IMUs' *lateral* accel, which the
/// sagittal-plane fork-tangent projection discards anyway). If a *travel-relevant*
/// axis were missing the zero-fill would silently lose signal — the correct mount
/// orientation determines which axis is lateral, so this pairs with §7 orientation.
fn imu_series_from_lookup(lookup: &dyn ChannelLookup, imu: usize) -> Option<(ImuSeries, f64)> {
    let names = IMU_CHANNEL_NAMES[imu];
    let ch: Vec<Option<crate::math::eval::LookupChannel>> = names.iter().map(|n| lookup.lookup(n)).collect();
    if ch.iter().all(|c| c.is_none()) {
        return None; // no channels for this IMU at all
    }
    let rate = ch.iter().flatten().next().unwrap().sample_rate_hz;
    let n = ch.iter().flatten().map(|c| c.samples.len()).min().unwrap_or(0);
    // Per-axis samples (g for accel, dps for gyro), zero-filled if absent.
    let axis_g = |idx: usize| -> Vec<f64> {
        ch[idx].as_ref().map(|c| c.samples[..n].to_vec()).unwrap_or_else(|| vec![0.0; n])
    };
    // **Declip** the accel axes: hard landings rail the ±32 g accelerometer, and
    // clipped compression peaks would underestimate integrated travel. The
    // reconstruction fits a smooth pulse over each clipped segment (no-op on clean
    // data). Gyro is not declipped (it does not rail at the accel range).
    let ax = crate::clip_reconstruct::declip(&axis_g(0), rate);
    let ay = crate::clip_reconstruct::declip(&axis_g(1), rate);
    let az = crate::clip_reconstruct::declip(&axis_g(2), rate);
    let (gx, gy, gz) = (axis_g(3), axis_g(4), axis_g(5));
    let accel = (0..n).map(|i| Vector3::new(ax[i], ay[i], az[i]) * GRAVITY).collect();
    let gyro = (0..n).map(|i| Vector3::new(gx[i], gy[i], gz[i]) * DEG2RAD).collect();
    Some((ImuSeries { gyro, accel }, rate))
}

/// GPS course (degrees clockwise from north) → nav-frame unit direction. Nav frame:
/// X = north, Y = west, Z = up (right-handed), so east is −Y and course θ maps to
/// `(cos θ, −sin θ, 0)`.
fn course_to_nav(heading_deg: f64) -> Vector3<f64> {
    let th = heading_deg.to_radians();
    Vector3::new(th.cos(), -th.sin(), 0.0)
}

/// Pulls the GPS velocity fixes from a lookup: `GPS_SpeedKmh` (km/h, physical) ×
/// `GPS_Heading` (raw centidegrees, ÷100 → degrees clockwise from north), stamped
/// with the speed channel's event times (`sample_times`, seconds on the recording
/// timeline; falls back to `index / rate` for a fixed-rate GPS channel). Fixes with
/// `GPS_FixQuality` = 0 (no fix) are dropped when that channel is present. Returns
/// an empty vec when speed/heading/timing are unavailable — GPS aiding is optional.
fn gps_samples_from_lookup(lookup: &dyn ChannelLookup) -> Vec<GpsSample> {
    let (Some(speed), Some(heading)) = (lookup.lookup("GPS_SpeedKmh"), lookup.lookup("GPS_Heading"))
    else {
        return vec![];
    };
    let n = speed.samples.len().min(heading.samples.len());
    let times = match lookup.sample_times("GPS_SpeedKmh") {
        Some(t) => t,
        None if speed.sample_rate_hz > 0.0 => {
            (0..n).map(|i| i as f64 / speed.sample_rate_hz).collect()
        }
        None => return vec![],
    };
    let quality = lookup.lookup("GPS_FixQuality");
    let mut out = Vec::with_capacity(n);
    for i in 0..n.min(times.len()) {
        if let Some(q) = &quality {
            if q.samples.get(i).is_some_and(|&v| v < 1.0) {
                continue; // no fix — the reported speed/course are meaningless
            }
        }
        let speed_mps = speed.samples[i] / 3.6; // km/h → m/s
        let heading_deg = heading.samples[i] / 100.0; // centidegrees → degrees
        out.push(GpsSample { time_s: times[i], velocity: speed_mps * course_to_nav(heading_deg) });
    }
    out
}

/// All synchronized inputs for one estimator run.
#[derive(Debug, Clone)]
pub struct EstimatorInput {
    /// Uniform sample period, s.
    pub dt: f64,
    /// Chassis/sprung IMU (required).
    pub imu0: ImuSeries,
    /// Front-unsprung IMU (drives front wheel travel; None ⇒ front states unobserved).
    pub imu1: Option<ImuSeries>,
    /// Rear-unsprung IMU (None on a hardtail).
    pub imu2: Option<ImuSeries>,
    /// GPS velocity samples (may be empty).
    pub gps: Vec<GpsSample>,
}

impl EstimatorInput {
    /// Builds an input by pulling the standard IMU channels
    /// (`IMU{0,1,2}_{Accel,Gyro}{X,Y,Z}`) and the GPS velocity fixes
    /// (`GPS_SpeedKmh` + `GPS_Heading`, at their event times) from a
    /// [`ChannelLookup`] (e.g. a parsed `SessionHandle`). IMU0 is required;
    /// IMU1/IMU2/GPS are optional. `dt` is taken from IMU0's sample rate. Channels
    /// are assumed uniform-rate and sample-aligned across IMUs (true for a v3
    /// `.idl0`); GPS fixes carry their own event times and are mapped onto the IMU
    /// timeline by the runner (latency-corrected, speed-gated).
    ///
    /// Returns None if IMU0 is absent or its sample rate is non-positive.
    pub fn from_lookup(lookup: &dyn ChannelLookup) -> Option<EstimatorInput> {
        let (imu0, rate) = imu_series_from_lookup(lookup, 0)?;
        if rate <= 0.0 {
            return None;
        }
        Some(EstimatorInput {
            dt: 1.0 / rate,
            imu0,
            imu1: imu_series_from_lookup(lookup, 1).map(|(s, _)| s),
            imu2: imu_series_from_lookup(lookup, 2).map(|(s, _)| s),
            gps: gps_samples_from_lookup(lookup),
        })
    }
}

/// Tuning for one run (refined in M3). [`EstimatorConfig::default`] gives sensible
/// starting values for the reference bike.
#[derive(Debug, Clone, Copy)]
pub struct EstimatorConfig {
    /// Estimate steering states (M2a: false — wheels first).
    pub estimate_steering: bool,
    /// Process-noise PSDs.
    pub process_noise: ProcessNoiseConfig,
    /// Initial-covariance priors.
    pub init_std: InitStd,
    /// Measurement-update iterations (1 = plain EKF).
    pub iekf_iters: usize,
    /// Stationary-detector window (samples).
    pub zupt_window: usize,
    /// Stationary accel-magnitude std threshold, m/s².
    pub zupt_accel_std: f64,
    /// Stationary mean gyro-magnitude threshold, rad/s.
    pub zupt_gyro_thresh: f64,
    /// ZUPT velocity pseudo-measurement std, m/s.
    pub zupt_sigma: f64,
    /// ZARU bias pseudo-measurement std, rad/s.
    pub zaru_sigma: f64,
    /// Gravity-leveling tilt residual std (direction components).
    pub gravity_sigma: f64,
    /// GPS velocity std, m/s.
    pub gps_sigma: f64,
    /// Sag prior std, m.
    pub sag_sigma: f64,
    /// Travel-barrier std, m.
    pub barrier_sigma: f64,
    /// IMU0 specific-force magnitude below which a sample is treated as **airborne**
    /// (free fall ⇒ suspension topped out), m/s².
    pub airborne_accel_thresh: f64,
    /// Max lever-compensated differential specific-force magnitude (per present
    /// unsprung IMU) for a sample to still count as **airborne**, m/s². The absolute
    /// criterion (`airborne_accel_thresh`) alone mislabels rough-but-grounded
    /// light-chassis moments (rebound crests, terrain unweighting) as free fall: a
    /// driven wheel has a large diff-accel even when `|accel0|` momentarily dips low.
    /// In real rigid-body free fall the unsprung mass tracks the chassis ⇒ diff-accel
    /// ≈ 0, so requiring it small vetoes those false topout events. Larger ⇒ looser
    /// (more samples pass as airborne); smaller ⇒ stricter (only a truly quiet,
    /// un-driven wheel counts).
    pub airborne_diff_thresh: f64,
    /// Topout-reference std applied on airborne samples (the travel-DC anchor), m.
    pub topout_sigma: f64,
    /// GPS output latency, s. The module's internal filter delays its velocity
    /// solution, so a fix logged at `t` describes the bike at `t − latency`; the
    /// runner applies each fix at the latency-corrected sample (offline ⇒ no
    /// causality constraint). Consumer modules typically sit at 0.1–0.3 s; tune
    /// against hard braking (GPS speed visibly lags the IMU deceleration).
    pub gps_latency_s: f64,
    /// Minimum GPS horizontal speed for a fix to be used, m/s. Below this the
    /// course (heading) is dominated by the GPS filter's own noise, so the derived
    /// velocity vector is meaningless; ZUPT covers the stationary case anyway.
    pub gps_min_speed_mps: f64,
    /// Refine the unsprung IMUs' mount tilt against this session's first stationary
    /// window. Default **false**: the mounts in [`BikeGeometry`] are calibrated from
    /// a controlled flat-ground recording, and a ride log's parked window (leaning
    /// bike, flopped bars) would be absorbed as mount error — a DC gravity leak into
    /// the wheel drive. Enable only for uncalibrated geometry.
    pub refine_mounts: bool,
    /// Run the fixed-interval RTS smoother over the wheel `{w, ẇ}` chains (default
    /// **true**). The backward pass distributes each topout/ZUPT anchor over the
    /// interval *before* it (two-sided boundary conditions on the travel integral),
    /// removing the forward filter's drift-then-yank artifact. Same drives, same
    /// factors, same tuning — no new physics; `false` gives the raw causal pass.
    pub smooth: bool,
    /// Fire the **sag prior** — a soft continuous pull of riding-window travel toward
    /// static sag. **Default `false` (bounds-only).** With it off, travel DC is anchored
    /// only by the physical `[0, travel_max]` barrier and the airborne topout reference;
    /// nothing pulls travel toward a nominal operating point. This is deliberate: the
    /// sag prior's recapture corner (~1–3 Hz) sits *inside* the suspension band and
    /// attenuates real motion, so it trades a cleaner recovered **velocity** (the
    /// deliverable — FFT / spectrogram / histogram) for a tighter absolute-travel DC.
    /// Enabling it (`true`) restores the continuous position reference at that cost; the
    /// trade-off and the `sag_sigma` tuning notes apply only then. When `false`,
    /// `sag_sigma` is inert. See design §5.
    pub use_sag_prior: bool,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        EstimatorConfig {
            estimate_steering: false,
            process_noise: ProcessNoiseConfig::reference_default(),
            init_std: InitStd::default(),
            iekf_iters: 1,
            zupt_window: 20,
            zupt_accel_std: 0.1,
            zupt_gyro_thresh: 0.05,
            zupt_sigma: 0.02,
            zaru_sigma: 1.0e-3,
            gravity_sigma: 0.05,
            // GPS velocity std models the error of the module's own smoothed/filtered
            // output — correlated over seconds and lagging the bike's dynamics — not
            // raw datasheet velocity accuracy, hence deliberately loose. The anchor
            // only needs to pin the DC/low-frequency band; the IMU carries the rest.
            gps_sigma: 0.5,
            // Sag-prior strength — used ONLY when `use_sag_prior` is true (off by
            // default; see that field). When on, this is a MODERATE continuous anti-
            // drift anchor: travel DC = double-integrated diff-accel, which drifts, and
            // topout events are too sparse to bound it between jumps, so sag is the only
            // continuous position reference this sensor set offers. Tuned to bound the
            // drift while letting real travel motion pass: weaker (larger σ) → more
            // motion through but it rails above ~3–4× this; stronger → pins harder toward
            // sag. Recapture is fast (τ < 0.4 s), so its corner lands inside the
            // suspension band — which is exactly why the default is bounds-only.
            sag_sigma: 0.5,
            // Bounds-only by default: anchor travel DC with the [0, max] barrier and the
            // airborne topout reference, not a sag pull, so the recovered velocity band
            // is undistorted. Flip to true to trade velocity fidelity for travel-DC
            // tightness (see `use_sag_prior`).
            use_sag_prior: false,
            barrier_sigma: 0.005,
            // Only deeper unweighting counts as free-fall; the sustained-free-fall
            // gate (below) rejects instantaneous sub-threshold dips so they don't
            // re-zero travel. A real airborne/topout event is sustained.
            airborne_accel_thresh: 2.5,
            // Relative free-fall veto: a grounded wheel driven by terrain shows a large
            // diff-accel even when the chassis is momentarily light, so a light-but-
            // grounded sample must NOT read as airborne. Seeded mid-range; tune on real
            // logs (lower if rough ground still snaps to topout, higher if real floats
            // get rejected and travel drifts mid-air).
            airborne_diff_thresh: 5.0,
            topout_sigma: 0.01,
            // Seeded mid-range for consumer GPS modules (their internal filter lags
            // ~0.1–0.3 s); tune against hard-braking events on real logs.
            gps_latency_s: 0.2,
            gps_min_speed_mps: 0.5,
            // Mounts come calibrated from the flat-ground recording baked into the
            // geometry; per-session refitting is opt-in (see `refine_mounts`).
            refine_mounts: false,
            smooth: true,
        }
    }
}

/// The result of one run: the wheel-output channels + the observability ledger.
/// With [`EstimatorConfig::smooth`] (the default) the four wheel outputs are the
/// **RTS-smoothed** trajectories — each topout/ZUPT anchor distributed backward over
/// the interval before it; with it off they are the raw causal forward pass.
/// The full per-sample [`MtbState`] trajectory is deliberately **not** retained
/// — only the four wheel outputs (the derived channels the bridge maps to `store_math`),
/// the per-sample stationary flag, and the final state — so a run's memory is O(outputs),
/// not O(samples × full state). A multi-hour log (millions of samples) stays bounded
/// instead of allocating hundreds of MB of state. The full covariance stays lazy too.
#[derive(Debug, Clone)]
pub struct StateEstimate {
    /// The geometry-derived schema this run estimated.
    pub schema: StateSchema,
    /// Uniform sample period, s.
    pub dt: f64,
    /// Front wheel travel per sample, m.
    pub front_travel: Vec<f64>,
    /// Front wheel velocity per sample, m/s.
    pub front_velocity: Vec<f64>,
    /// Rear wheel travel per sample, m (zeros on a hardtail).
    pub rear_travel: Vec<f64>,
    /// Rear wheel velocity per sample, m/s.
    pub rear_velocity: Vec<f64>,
    /// Per-sample (quasi-)stationary flag (true ⇒ not riding). Lets derived ride
    /// statistics (e.g. dynamic sag) exclude parked/stopped portions.
    pub stationary: Vec<bool>,
    /// The final filter mean (the trajectory is not retained; this carries the
    /// end-of-run attitude / velocity / biases for diagnostics and the ledger).
    pub final_state: MtbState,
    /// Per-component observability (confidence + DC-source).
    pub ledger: ObservabilityLedger,
}

impl StateEstimate {
    /// Number of samples in the run.
    pub fn len(&self) -> usize {
        self.front_travel.len()
    }

    /// Whether the run produced no samples.
    pub fn is_empty(&self) -> bool {
        self.front_travel.is_empty()
    }

    /// **Dynamic sag** — the median recovered front travel over *riding* (non-
    /// stationary) samples, m. This is an emergent **measurement that drops out of
    /// the estimate**, not an input: unlike static sag (rider-on, motionless), the
    /// mean/median ride position reflects braking, pumping, terrain, and body
    /// position, so it is only meaningful once the estimate is accurate (the sag
    /// *prior* is a weak fallback that deliberately does not target this). Median
    /// (not mean) for robustness to topout/bottomout spikes. None if never riding.
    pub fn front_dynamic_sag(&self) -> Option<f64> {
        median_where(&self.front_travel, &self.stationary)
    }

    /// Dynamic sag for the rear wheel, m (see [`StateEstimate::front_dynamic_sag`]).
    pub fn rear_dynamic_sag(&self) -> Option<f64> {
        median_where(&self.rear_travel, &self.stationary)
    }
}

/// Median of `values` at indices where `stationary` is false (riding samples).
/// None if there are no such samples.
fn median_where(values: &[f64], stationary: &[bool]) -> Option<f64> {
    let mut riding: Vec<f64> = values
        .iter()
        .zip(stationary.iter())
        .filter(|(_, &st)| !st)
        .map(|(&v, _)| v)
        .collect();
    if riding.is_empty() {
        return None;
    }
    riding.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = riding.len() / 2;
    Some(if riding.len() % 2 == 0 { 0.5 * (riding[mid - 1] + riding[mid]) } else { riding[mid] })
}

/// Central-difference derivative of a vector stream (forward/backward at the ends).
fn central_diff(v: &[Vector3<f64>], dt: f64) -> Vec<Vector3<f64>> {
    let n = v.len();
    let mut d = vec![Vector3::zeros(); n];
    for i in 0..n {
        d[i] = if n < 2 {
            Vector3::zeros()
        } else if i == 0 {
            (v[1] - v[0]) / dt
        } else if i == n - 1 {
            (v[n - 1] - v[n - 2]) / dt
        } else {
            (v[i + 1] - v[i - 1]) / (2.0 * dt)
        };
    }
    d
}

/// Minimum duration a free-fall (airborne) reading must persist before it counts
/// as a topout event, seconds. A real airborne/jump lasts ≳100 ms; this rejects
/// the instantaneous sub-threshold dips (rebound crests, sharp unweights, accel
/// noise toward zero) that would otherwise re-zero travel every few samples via
/// the topout reference. ~50 ms is short enough to still catch a small bunny-hop
/// yet long enough to exclude single-sample dips. (Refined in M3; could become a
/// tuning field.)
const AIRBORNE_MIN_DURATION_S: f64 = 0.05;

/// Maximum interruption, in seconds, that may break a free-fall run without ending
/// the float. A real jump's specific force dips back *above* the free-fall threshold
/// for a few samples mid-air (sensor noise, a clipped landing-edge precursor, a brief
/// tap of a feature) — without bridging, that flicker opens a coasting window in which
/// the (now negligible) sag prior would otherwise re-capture travel before landing.
/// Bridging gaps up to ~100 ms keeps the topout reference firing continuously across
/// the whole float, so travel stays at topout until the (sustained, high-g) landing.
/// Landings are far longer than this, so they are never bridged. (Refined in M3.)
const AIRBORNE_MAX_GAP_S: f64 = 0.10;

/// Fills runs of consecutive `false`s of length ≤ `max_gap` that sit **between** two
/// `true` runs (morphological closing). Leading/trailing `false` runs and gaps longer
/// than `max_gap` are left open. Used to bridge brief mid-float interruptions of the
/// free-fall flag so one jump reads as a single continuous airborne phase.
fn close_short_gaps(flags: &[bool], max_gap: usize) -> Vec<bool> {
    let n = flags.len();
    let mut out = flags.to_vec();
    let mut i = 0;
    while i < n {
        if flags[i] {
            i += 1;
            continue;
        }
        // [i, j) is a maximal run of `false`.
        let start = i;
        while i < n && !flags[i] {
            i += 1;
        }
        let bounded = start > 0 && i < n; // a true run on both sides
        if bounded && (i - start) <= max_gap {
            out[start..i].fill(true);
        }
    }
    out
}

/// Marks each index `true` only if it lies inside a maximal run of consecutive
/// `true`s in `flags` of length ≥ `min_len`; isolated or too-short runs become
/// `false`. Used to require a **sustained** free-fall before the topout reference
/// fires, so an instantaneous low-g dip never re-zeros wheel travel.
fn sustained_runs(flags: &[bool], min_len: usize) -> Vec<bool> {
    let n = flags.len();
    let mut out = vec![false; n];
    let mut i = 0;
    while i < n {
        if !flags[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && flags[i] {
            i += 1;
        }
        if i - start >= min_len {
            out[start..i].fill(true);
        }
    }
    out
}

/// First maximal run of stationary-flagged samples, as `[start, end)`.
fn first_stationary_window(ctx: &[SampleContext]) -> Option<(usize, usize)> {
    let start = ctx.iter().position(|c| c.stationary)?;
    let mut end = start;
    while end < ctx.len() && ctx[end].stationary {
        end += 1;
    }
    Some((start, end))
}

/// Runs the estimator over `input` for the given `geometry` and `config`.
pub fn run(input: &EstimatorInput, geometry: &BikeGeometry, config: &EstimatorConfig) -> StateEstimate {
    run_with_trace(input, geometry, config).0
}

/// The single pass shared by [`run`] and [`run_trace`]: forward IEKF, optional RTS
/// wheel smoothing, and the per-sample trace of exactly what the wheel integrators
/// and the airborne detector saw.
fn run_with_trace(
    input: &EstimatorInput,
    geometry: &BikeGeometry,
    config: &EstimatorConfig,
) -> (StateEstimate, EstimatorTrace) {
    let n = input.imu0.gyro.len();
    let schema = StateSchema::from_geometry(geometry, config.estimate_steering);

    // Apply IMU0's mount ONCE, up front, so every downstream IMU0 consumer — the
    // orientation/bias fit, the lever-term ω̇, and the predict/leveling loop — works in
    // the chassis frame, matching `fit_from_window`'s contract ("inputs already in the
    // chassis frame"). The old code applied the mount only per-sample in the loop and
    // left the fit + ω̇ on raw IMU0 data: harmless under an identity mount, wrong once
    // IMU0 carries a real one (it is mounted X-rear/Y-right — a 180° yaw).
    let m0 = geometry.imu0.mount;
    let imu0_accel: Vec<Vector3<f64>> = input.imu0.accel.iter().map(|a| m0 * a).collect();
    let imu0_gyro: Vec<Vector3<f64>> = input.imu0.gyro.iter().map(|g| m0 * g).collect();

    // Stationary gate + orientation/bias pre-step (first stationary window, else the
    // opening samples).
    let ctx = stationary_context(
        &input.imu0.accel,
        &input.imu0.gyro,
        config.zupt_window,
        config.zupt_accel_std,
        config.zupt_gyro_thresh,
    );
    let (ws, we) = first_stationary_window(&ctx).unwrap_or((0, n.min(50)));
    let imu1_gyro: &[Vector3<f64>] = input.imu1.as_ref().map(|s| s.gyro.as_slice()).unwrap_or(&[]);
    let imu2_gyro: &[Vector3<f64>] = input.imu2.as_ref().map(|s| s.gyro.as_slice()).unwrap_or(&[]);
    let fit = fit_from_window(&imu0_accel, &imu0_gyro, imu1_gyro, imu2_gyro, ws, we);

    // Unsprung mounts: the geometry carries mounts calibrated from a controlled
    // flat-ground recording (see `BikeGeometry`), used as-is by default. Optional
    // per-session tilt refinement (`config.refine_mounts`) re-fits them against this
    // log's first stationary window — against the mounted IMU0's window-mean "up",
    // NOT chassis +Z, so a leaning parked bike doesn't read as mount tilt. (A
    // bar-flopped fork still would — gravity alone can't separate parked steer from
    // mount error — which is why calibrated geometry + refine_mounts=false is the
    // default.) IMU0 defines the chassis frame, so it is never refined.
    let up0_ref = {
        let hi = we.min(imu0_accel.len());
        let lo = ws.min(hi);
        let win = &imu0_accel[lo..hi];
        let mean = if win.is_empty() {
            Vector3::zeros()
        } else {
            win.iter().sum::<Vector3<f64>>() / win.len() as f64
        };
        if mean.norm() > 1e-6 { mean } else { Vector3::z() }
    };
    let imu1_mount = match (&input.imu1, config.refine_mounts) {
        (Some(s), true) => refine_mount_from_window(geometry.imu1.mount, &s.accel, ws, we, up0_ref),
        _ => geometry.imu1.mount,
    };
    let imu2_mount = match (&input.imu2, &geometry.imu2) {
        (Some(s), Some(pose)) if config.refine_mounts => {
            refine_mount_from_window(pose.mount, &s.accel, ws, we, up0_ref)
        }
        (_, Some(pose)) => pose.mount,
        _ => UnitQuaternion::identity(),
    };

    // Initial state: leveled attitude + fitted biases. Travel is seeded at **topout
    // (0)** — the unambiguous physical floor an unweighted/parked bike rests at — not
    // sag: sag only holds with rider weight on. Real travel DC comes from the diff-
    // accel integrating a weighting/compression event and (M3) topout/bottomout
    // references; the sag prior is only a loose coasting-window nudge (below).
    let sag_front = geometry.front_sag * geometry.front_travel_max;
    let sag_rear = geometry.rear_sag * geometry.rear_travel_max;
    let x0 = MtbState {
        r_chassis: fit.r_chassis0,
        v_chassis: Vector3::zeros(),
        b_g0: fit.b_g0,
        b_a0: Vector3::zeros(),
        b_g1: fit.b_g1,
        b_g2: fit.b_g2,
        d_f: 0.0,
        dd_f: 0.0,
        s_r: 0.0,
        ds_r: 0.0,
        psi: 0.0,
        dpsi: 0.0,
    };
    let init_fs = FilterState::initial(x0, &schema, &config.init_std);
    let filter = Iekf {
        process: MtbProcess { schema: schema.clone(), noise: config.process_noise, gravity: GRAVITY },
        max_iters: config.iekf_iters,
        tol: 1e-9,
    };

    // Per-sample IMU0 angular acceleration for the lever term (chassis frame: from the
    // mounted gyro, so ω̇ matches the mounted ω the transport term pairs it with).
    let omega0_dot = central_diff(&imu0_gyro, input.dt);
    // GPS velocity onto the IMU timeline: each fix applies at its latency-corrected
    // epoch (the GPS filter's output lags the bike by `gps_latency_s`), gated by the
    // min-speed threshold (course is noise at walking pace and below).
    let mut gps_at: Vec<Option<Vector3<f64>>> = vec![None; n];
    for s in &input.gps {
        if s.velocity.xy().norm() < config.gps_min_speed_mps {
            continue;
        }
        let t = s.time_s - config.gps_latency_s;
        if t < 0.0 {
            continue;
        }
        let idx = (t / input.dt).round() as usize;
        if idx < n {
            gps_at[idx] = Some(s.velocity);
        }
    }

    let front_axis = geometry.front_tangent();
    let dt = input.dt;
    // Airborne / topout (design §5 topout event). The raw per-sample flag below is
    // gated twice before it can fire the topout reference: a *sustained* run of at
    // least `AIRBORNE_MIN_DURATION_S` (a single low-g instant — a rebound crest, a
    // sharp unweight, accel noise — is not a jump), and the two-criteria test here.
    // Free fall needs BOTH: (1) the chassis specific force collapses toward 0 (the
    // absolute criterion), AND (2) every present unsprung IMU's lever-compensated
    // diff-accel is small (the relative criterion). In real rigid-body free fall the
    // wheel is topped out and tracks the chassis ⇒ diff-accel ≈ 0; a rough-but-
    // grounded light-chassis moment fails (2) because the wheel is still terrain-
    // driven, so it no longer mislabels as airborne and re-zeros travel.
    // Per-sample detector inputs, retained for the [`EstimatorTrace`] (the tuning
    // GUI re-derives the flag from these) and consumed by the flag test below.
    let mut accel0_norm = vec![0.0; n];
    let mut front_diff_mag = vec![0.0; n];
    let mut rear_diff_mag = vec![0.0; n];
    for i in 0..n {
        let accel0 = imu0_accel[i];
        accel0_norm[i] = accel0.norm();
        let omega0 = imu0_gyro[i] - fit.b_g0;
        if let Some(s) = &input.imu1 {
            let lever_term = lever_arm_accel(Vector3::zeros(), omega0, omega0_dot[i], geometry.imu1.lever);
            front_diff_mag[i] = (imu1_mount * s.accel[i] - accel0 - lever_term).norm();
        }
        if let (Some(s), Some(pose)) = (&input.imu2, &geometry.imu2) {
            let lever_term = lever_arm_accel(Vector3::zeros(), omega0, omega0_dot[i], pose.lever);
            rear_diff_mag[i] = (imu2_mount * s.accel[i] - accel0 - lever_term).norm();
        }
    }
    let airborne_raw: Vec<bool> = (0..n)
        .map(|i| {
            if accel0_norm[i] >= config.airborne_accel_thresh {
                return false;
            }
            let front_quiet = input.imu1.is_none() || front_diff_mag[i] < config.airborne_diff_thresh;
            let rear_quiet = !(input.imu2.is_some() && geometry.imu2.is_some())
                || rear_diff_mag[i] < config.airborne_diff_thresh;
            front_quiet && rear_quiet
        })
        .collect();
    // Bridge brief mid-float interruptions of the free-fall flag (sensor noise, a
    // clipped peak) so one jump is a single continuous airborne phase — otherwise the
    // gap opens a coasting window where the sag prior re-captures travel before
    // landing. Then require the bridged run to be sustained (a real jump lasts ≳100 ms)
    // so an instantaneous low-g dip never re-zeros travel via the topout reference.
    let max_gap = ((AIRBORNE_MAX_GAP_S / dt).round() as usize).max(1);
    let min_airborne = ((AIRBORNE_MIN_DURATION_S / dt).round() as usize).max(1);
    let airborne = sustained_runs(&close_short_gaps(&airborne_raw, max_gap), min_airborne);
    let mut fs = init_fs.clone();
    // Stream the four wheel outputs per sample; the full per-sample state is NOT
    // retained (see [`StateEstimate`]), so memory stays O(outputs) on long sessions.
    let mut front_travel = Vec::with_capacity(n);
    let mut front_velocity = Vec::with_capacity(n);
    let mut rear_travel = Vec::with_capacity(n);
    let mut rear_velocity = Vec::with_capacity(n);
    // The exact wheel-drive controls fed to the integrators — consumed by the RTS
    // smoothing pass below and exposed on the trace (8 B/sample each).
    let mut front_drive = vec![0.0; n];
    let mut rear_drive = vec![0.0; n];

    for i in 0..n {
        let accel0 = imu0_accel[i];
        let gyro0 = imu0_gyro[i];
        let omega0 = gyro0 - fit.b_g0; // bias-corrected chassis rate for the lever term

        // Front wheel drive: projected differential specific force along the fork axis.
        let wheel_front = match &input.imu1 {
            Some(s) => wheel_drive_accel(
                front_axis,
                imu1_mount * s.accel[i],
                accel0,
                omega0,
                omega0_dot[i],
                geometry.imu1.lever,
            ),
            None => 0.0,
        };
        // Rear wheel drive: tangent evaluated at the current rear travel estimate.
        let wheel_rear = match (&input.imu2, &geometry.imu2, geometry.rear_path.is_some()) {
            (Some(s), Some(pose), true) => {
                let axis = geometry.rear_tangent(fs.x.s_r.clamp(0.0, geometry.rear_travel_max));
                wheel_drive_accel(axis, imu2_mount * s.accel[i], accel0, omega0, omega0_dot[i], pose.lever)
            }
            _ => 0.0,
        };

        front_drive[i] = wheel_front;
        rear_drive[i] = wheel_rear;
        let u = ImuInput { gyro0, accel0, wheel_accel_front: wheel_front, wheel_accel_rear: wheel_rear };
        fs = filter.predict(&fs, &u, dt);

        let c = ctx[i];
        // ZUPT + per-IMU ZARU on stationary samples.
        if c.stationary {
            fs = filter.update(&fs, &ZeroVelocity { sigma: config.zupt_sigma }, &c);
            fs = filter.update(
                &fs,
                &ZeroAngularRate { target: GyroBias::Imu0, measured: gyro0, sigma: config.zaru_sigma },
                &c,
            );
            if let Some(s) = &input.imu1 {
                fs = filter.update(
                    &fs,
                    &ZeroAngularRate { target: GyroBias::Imu1, measured: imu1_mount * s.gyro[i], sigma: config.zaru_sigma },
                    &c,
                );
            }
            if let (Some(s), Some(_pose)) = (&input.imu2, &geometry.imu2) {
                fs = filter.update(
                    &fs,
                    &ZeroAngularRate { target: GyroBias::Imu2, measured: imu2_mount * s.gyro[i], sigma: config.zaru_sigma },
                    &c,
                );
            }
            // Wheel-velocity ZUPT: a motionless bike has zero travel-rate.
            if input.imu1.is_some() {
                fs = filter.update(&fs, &ZeroWheelVelocity { wheel: Wheel::Front, sigma: config.zupt_sigma }, &c);
            }
            if geometry.rear_path.is_some() {
                fs = filter.update(&fs, &ZeroWheelVelocity { wheel: Wheel::Rear, sigma: config.zupt_sigma }, &c);
            }
        }
        let air = airborne[i];
        // Gravity-leveling (M2a: kinematic compensation off — a_kin from v̇ is M3 tuning).
        // GATED OFF in free fall: airborne ⇒ specific force → 0 ⇒ there is no gravity
        // direction to level against. Running it there normalizes a ~0 vector (NaN, or
        // a ~9× Jacobian blow-up short of NaN) and injects bogus tilt that corrupts
        // attitude → the lever term → travel. Attitude propagates open-loop on the gyro
        // through the (short) float and re-levels on landing. (A brief sub-threshold dip
        // that never sustains into `airborne` still fires, but the factor is NaN-safe.)
        if !air {
            fs = filter.update(
                &fs,
                &GravityLeveling { accel0, a_kin_nav: Vector3::zeros(), sigma: config.gravity_sigma },
                &c,
            );
        }
        // Travel DC factors. The authoritative anchor is the **topout reference**:
        // when airborne (free fall), the wheel is unweighted → topped out → travel 0,
        // which re-zeros the double-integrator. The **barrier** (the physical [0, max]
        // wall) applies always. The **sag prior** is an OPTIONAL coasting-window pull
        // toward static sag, fired only when `use_sag_prior` is set — off by default
        // (bounds-only), because its recapture corner sits inside the suspension band
        // and attenuates the recovered velocity (see `EstimatorConfig::use_sag_prior`).
        // A parked (stationary) bike gets none of these (it sits at topout, but we
        // don't assert it).
        if air {
            fs = filter.update(&fs, &TopoutReference { wheel: Wheel::Front, sigma: config.topout_sigma }, &c);
            // A topped-out wheel in free fall is held against its stop — not moving.
            // Pin ẇ = 0 so its uncertainty (and the travel uncertainty it inflates)
            // stays bounded through the float, and the in-air velocity output reads ~0
            // instead of integrator noise.
            fs = filter.update(&fs, &ZeroWheelVelocity { wheel: Wheel::Front, sigma: config.zupt_sigma }, &c);
        } else if !c.stationary && config.use_sag_prior {
            fs = filter.update(&fs, &SagPrior { wheel: Wheel::Front, sag: sag_front, sigma: config.sag_sigma }, &c);
        }
        fs = filter.update(
            &fs,
            &TravelBarrier { wheel: Wheel::Front, travel_max: geometry.front_travel_max, sigma: config.barrier_sigma },
            &c,
        );
        if geometry.rear_path.is_some() {
            if air {
                fs = filter.update(&fs, &TopoutReference { wheel: Wheel::Rear, sigma: config.topout_sigma }, &c);
                fs = filter.update(&fs, &ZeroWheelVelocity { wheel: Wheel::Rear, sigma: config.zupt_sigma }, &c);
            } else if !c.stationary && config.use_sag_prior {
                fs = filter.update(&fs, &SagPrior { wheel: Wheel::Rear, sag: sag_rear, sigma: config.sag_sigma }, &c);
            }
            fs = filter.update(
                &fs,
                &TravelBarrier { wheel: Wheel::Rear, travel_max: geometry.rear_travel_max, sigma: config.barrier_sigma },
                &c,
            );
        }
        // GPS velocity anchor when present.
        if let Some(vel) = gps_at[i] {
            fs = filter.update(&fs, &GpsVelocity { measured: vel, sigma: config.gps_sigma }, &c);
        }

        front_travel.push(fs.x.d_f);
        front_velocity.push(fs.x.dd_f);
        rear_travel.push(fs.x.s_r);
        rear_velocity.push(fs.x.ds_r);
    }

    let ledger = ObservabilityLedger::build(&init_fs, &fs, &schema, 0.5);
    let stationary: Vec<bool> = ctx.iter().map(|c| c.stationary).collect();

    // RTS smoothing over the wheel chains (design: the offline backward pass). The
    // wheel `{w, ẇ}` block is exactly decoupled from the rest of the 24-DOF filter
    // (block-diagonal F/Q/P₀ and no factor shares its columns), so a standalone
    // 2-state pass over the same drives + factor schedule reproduces the forward
    // marginals, and the backward sweep then distributes each topout/ZUPT anchor
    // over the interval BEFORE it — two-sided boundary conditions instead of the
    // causal drift-then-yank. (Replay exactness holds at `iekf_iters = 1`, the
    // default, and when the wheel's own IMU is present — the shipped configuration.)
    if config.smooth {
        let wheel_params = |sag: f64, travel_max: f64| smooth::WheelParams {
            dt,
            q_pos: config.process_noise.wheel_pos_rw.powi(2) * dt,
            q_vel: config.process_noise.wheel_vel_rw.powi(2) * dt,
            init_travel_var: config.init_std.wheel_travel.powi(2),
            init_vel_var: config.init_std.wheel_velocity.powi(2),
            zupt_sigma: config.zupt_sigma,
            topout_sigma: config.topout_sigma,
            barrier_sigma: config.barrier_sigma,
            sag_sigma: config.sag_sigma,
            sag,
            travel_max,
            use_sag_prior: config.use_sag_prior,
        };
        // Mirror the loop's gating exactly: the stationary wheel-ZUPT fires only
        // when the front unsprung IMU is present.
        let front_stationary: Vec<bool> =
            if input.imu1.is_some() { stationary.clone() } else { vec![false; n] };
        let (tw, tv) = smooth::smooth_wheel(
            &front_drive,
            &front_stationary,
            &airborne,
            &wheel_params(sag_front, geometry.front_travel_max),
        );
        front_travel = tw;
        front_velocity = tv;
        if geometry.rear_path.is_some() {
            let (tw, tv) = smooth::smooth_wheel(
                &rear_drive,
                &stationary,
                &airborne,
                &wheel_params(sag_rear, geometry.rear_travel_max),
            );
            rear_travel = tw;
            rear_velocity = tv;
        }
    }

    let final_state = fs.x;
    let est = StateEstimate {
        schema,
        dt: input.dt,
        front_travel,
        front_velocity,
        rear_travel,
        rear_velocity,
        stationary: stationary.clone(),
        final_state,
        ledger,
    };
    let trace = EstimatorTrace {
        dt,
        front_drive,
        rear_drive,
        accel0_norm,
        front_diff_mag,
        rear_diff_mag,
        stationary,
        airborne,
    };
    (est, trace)
}

/// Per-sample diagnostic trace alongside a run, for the offline tuning GUI
/// (`tools/estimator_sim`). All vectors are length `n` (the sample count).
///
/// `front_drive`/`rear_drive` are the **exact** wheel-drive accelerations the run
/// fed its `{w, ẇ}` integrators — captured from the forward loop itself (the rear
/// tangent evaluated at the loop's own per-sample travel), so a downstream 2-state
/// replay over them reproduces the engine's forward travel exactly. `accel0_norm` +
/// `*_diff_mag` are the airborne-detector inputs (so threshold sliders can re-derive
/// the flag), `airborne` is the final bridged/sustained flag the run actually used,
/// and `stationary` is the ZUPT gate.
#[derive(Debug, Clone)]
pub struct EstimatorTrace {
    /// Sample period, s.
    pub dt: f64,
    /// Front wheel-drive acceleration fed to the integrator, m/s².
    pub front_drive: Vec<f64>,
    /// Rear wheel-drive acceleration fed to the integrator, m/s².
    pub rear_drive: Vec<f64>,
    /// IMU0 specific-force magnitude (the airborne absolute test), m/s².
    pub accel0_norm: Vec<f64>,
    /// Lever-compensated front diff-accel magnitude (the airborne relative test), m/s².
    pub front_diff_mag: Vec<f64>,
    /// Lever-compensated rear diff-accel magnitude, m/s².
    pub rear_diff_mag: Vec<f64>,
    /// Per-sample stationary flag (ZUPT gate).
    pub stationary: Vec<bool>,
    /// Per-sample airborne flag as the run used it (gap-bridged + sustained) —
    /// gates the topout reference and the in-air wheel-velocity pin.
    pub airborne: Vec<bool>,
}

/// Runs the estimator and additionally returns the per-sample [`EstimatorTrace`] of
/// the inputs the wheel integrators and the airborne detector saw — for the offline
/// tuning GUI. The trace is captured from the forward loop itself (same mounts,
/// bias, lever transport, and the loop's own per-sample rear travel for the rear
/// tangent), so it matches the engine exactly by construction.
pub fn run_trace(
    input: &EstimatorInput,
    geometry: &BikeGeometry,
    config: &EstimatorConfig,
) -> (StateEstimate, EstimatorTrace) {
    run_with_trace(input, geometry, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::ledger::DcSource;
    use crate::math::eval::LookupChannel;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    /// A minimal ChannelLookup test double over a name → (samples, rate) map, with
    /// optional per-channel event times (the GPS path).
    struct FakeLookup {
        channels: HashMap<String, (Vec<f64>, f64)>,
        times: HashMap<String, Vec<f64>>,
    }

    impl ChannelLookup for FakeLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            self.channels
                .get(name)
                .map(|(s, r)| LookupChannel { samples: s.clone().into(), sample_rate_hz: *r })
        }

        fn sample_times(&self, name: &str) -> Option<Vec<f64>> {
            self.times.get(name).cloned()
        }
    }

    #[test]
    fn from_lookup_converts_units_and_builds_per_imu_series() {
        // Arrange — IMU0 only: accel reads 1 g up; gyro reads 90 dps about Z, at 200 Hz.
        let mut channels = HashMap::new();
        for (name, val) in [
            ("IMU0_AccelX", 0.0),
            ("IMU0_AccelY", 0.0),
            ("IMU0_AccelZ", 1.0), // 1 g
            ("IMU0_GyroX", 0.0),
            ("IMU0_GyroY", 0.0),
            ("IMU0_GyroZ", 90.0), // 90 dps
        ] {
            channels.insert(name.to_string(), (vec![val; 4], 200.0));
        }
        let lookup = FakeLookup { channels, times: HashMap::new() };

        // Act
        let input = EstimatorInput::from_lookup(&lookup).unwrap();

        // Assert — dt from rate; g → m/s²; dps → rad/s; IMU1/2 absent.
        assert_relative_eq!(input.dt, 1.0 / 200.0, epsilon = 1e-12);
        assert_eq!(input.imu0.accel.len(), 4);
        assert_relative_eq!(input.imu0.accel[0], Vector3::new(0.0, 0.0, GRAVITY), epsilon = 1e-9);
        assert_relative_eq!(input.imu0.gyro[0], Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2), epsilon = 1e-9);
        assert!(input.imu1.is_none());
        assert!(input.imu2.is_none());
    }

    #[test]
    fn dynamic_sag_is_median_travel_over_riding_samples_only() {
        // Arrange — a hand-built estimate: travel rises 0→0.10 m, but the first two
        // samples are parked (stationary). Dynamic sag must ignore the parked ones.
        let geometry = BikeGeometry::reference_bike();
        let schema = StateSchema::from_geometry(&geometry, false);
        let est = StateEstimate {
            schema,
            dt: 0.01,
            front_travel: vec![0.0, 0.0, 0.04, 0.06, 0.10], // last 3 are riding
            front_velocity: vec![0.0; 5],
            rear_travel: vec![0.0; 5],
            rear_velocity: vec![0.0; 5],
            stationary: vec![true, true, false, false, false],
            final_state: rest_state(),
            ledger: crate::estimate::ledger::ObservabilityLedger { components: vec![] },
        };

        // Act + Assert — median of the riding subset {0.04, 0.06, 0.10} is 0.06; the
        // parked-at-topout samples are excluded.
        assert_relative_eq!(est.front_dynamic_sag().unwrap(), 0.06, epsilon = 1e-12);
    }

    #[test]
    fn dynamic_sag_is_none_when_never_riding() {
        // Arrange — a fully parked estimate.
        let geometry = BikeGeometry::reference_bike();
        let est = StateEstimate {
            schema: StateSchema::from_geometry(&geometry, false),
            dt: 0.01,
            front_travel: vec![0.0, 0.0],
            front_velocity: vec![0.0, 0.0],
            rear_travel: vec![0.0, 0.0],
            rear_velocity: vec![0.0, 0.0],
            stationary: vec![true, true],
            final_state: rest_state(),
            ledger: crate::estimate::ledger::ObservabilityLedger { components: vec![] },
        };

        // Act + Assert
        assert!(est.front_dynamic_sag().is_none());
    }

    fn rest_state() -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::identity(),
            v_chassis: Vector3::zeros(),
            b_g0: Vector3::zeros(),
            b_a0: Vector3::zeros(),
            b_g1: Vector3::zeros(),
            b_g2: Vector3::zeros(),
            d_f: 0.0,
            dd_f: 0.0,
            s_r: 0.0,
            ds_r: 0.0,
            psi: 0.0,
            dpsi: 0.0,
        }
    }

    #[test]
    fn from_lookup_zero_fills_a_missing_axis() {
        // Arrange — IMU0 full; IMU1 present but missing AccelZ (the lateral axis on
        // some loggers). The adapter must keep IMU1 with AccelZ zero-filled.
        let mut channels = HashMap::new();
        for n in IMU_CHANNEL_NAMES[0] {
            channels.insert(n.to_string(), (vec![0.0; 3], 100.0));
        }
        for n in IMU_CHANNEL_NAMES[1] {
            if n == "IMU1_AccelZ" {
                continue; // omit the lateral axis
            }
            channels.insert(n.to_string(), (vec![2.0; 3], 100.0));
        }
        let lookup = FakeLookup { channels, times: HashMap::new() };

        // Act
        let input = EstimatorInput::from_lookup(&lookup).unwrap();

        // Assert — IMU1 present; its accel X/Y are 2 g, Z zero-filled.
        let s1 = input.imu1.expect("imu1 present despite missing one axis");
        assert_relative_eq!(s1.accel[0].x, 2.0 * GRAVITY, epsilon = 1e-9);
        assert_relative_eq!(s1.accel[0].z, 0.0, epsilon = 1e-12);
        assert!(input.imu2.is_none());
    }

    #[test]
    fn from_lookup_returns_none_without_imu0() {
        // Arrange — empty lookup.
        let lookup = FakeLookup { channels: HashMap::new(), times: HashMap::new() };

        // Act + Assert
        assert!(EstimatorInput::from_lookup(&lookup).is_none());
    }

    /// A motionless, upright, flat recording: gravity up, zero rate, on every IMU.
    /// The synthetic mirror of the real 10 s static first-light log (design testing §).
    fn static_input(n: usize, dt: f64) -> EstimatorInput {
        let up = Vector3::new(0.0, 0.0, GRAVITY);
        let series = || ImuSeries { gyro: vec![Vector3::zeros(); n], accel: vec![up; n] };
        EstimatorInput { dt, imu0: series(), imu1: Some(series()), imu2: Some(series()), gps: vec![] }
    }

    #[test]
    fn static_first_light_reads_zero_and_stays_level() {
        // Arrange — 10 s at 100 Hz, motionless and flat.
        let input = static_input(1000, 0.01);
        let geometry = BikeGeometry::reference_bike();

        // Act
        let est = run(&input, &geometry, &EstimatorConfig::default());

        // Assert — every output reads its rest value and attitude is level & stable.
        // The bike is motionless and **unweighted**, so travel rests at topout (0),
        // NOT sag (sag is gated to coasting windows — a parked bike isn't at sag).
        assert_eq!(est.len(), 1000);
        let last = &est.final_state;
        assert!(last.v_chassis.norm() < 1e-3, "velocity drifted: {}", last.v_chassis.norm());
        assert!(last.r_chassis.angle() < 1e-3, "attitude drifted: {}", last.r_chassis.angle());
        assert!(last.d_f.abs() < 1e-3, "front travel left topout: {}", last.d_f);
        assert!(last.dd_f.abs() < 1e-2, "front velocity nonzero: {}", last.dd_f);
        assert!(last.ds_r.abs() < 1e-2, "rear velocity nonzero: {}", last.ds_r);
    }

    #[test]
    fn static_run_ledger_pins_velocity_and_flags_rest_yaw() {
        // Arrange + Act
        let input = static_input(800, 0.01);
        let est = run(&input, &BikeGeometry::reference_bike(), &EstimatorConfig::default());

        // Assert — velocity confidently pinned (ZUPT); attitude relative-only at rest
        // (yaw gauge); steering frozen (M2a).
        assert_eq!(est.ledger.get("v_chassis").unwrap().dc_source, DcSource::Pinned);
        assert_eq!(est.ledger.get("R_chassis").unwrap().dc_source, DcSource::RelativeOnly);
        assert_eq!(est.ledger.get("psi").unwrap().dc_source, DcSource::Frozen);
    }

    #[test]
    fn close_short_gaps_bridges_brief_gaps_but_not_long_ones() {
        // Arrange — two airborne runs split by a 2-sample gap, then a 4-sample gap.
        // A real float dips back above the free-fall threshold for a few samples
        // mid-air (sensor noise, a clipped peak), which must NOT end the float.
        let flags = [
            true, true, true, false, false, true, true, true, false, false, false, false, true, true,
        ];

        // Act — bridge gaps of at most 3 samples.
        let out = close_short_gaps(&flags, 3);

        // Assert — the 2-sample gap is filled (one continuous float); the 4-sample
        // gap stays open (a genuine ground contact between two airborne phases).
        assert_eq!(
            out,
            vec![
                true, true, true, true, true, true, true, true, false, false, false, false, true,
                true,
            ]
        );
    }

    #[test]
    fn sustained_runs_drops_short_dips_and_keeps_long_runs() {
        // Arrange — a lone 1-sample dip, a 2-sample dip, then a sustained run of 4.
        let flags = [false, true, false, true, true, false, true, true, true, true, false];

        // Act — require at least 3 consecutive samples to count as sustained.
        let out = sustained_runs(&flags, 3);

        // Assert — the 1- and 2-sample dips drop; only the run of 4 survives.
        assert_eq!(
            out,
            vec![false, false, false, false, false, false, true, true, true, true, false]
        );
    }

    #[test]
    fn run_refines_a_tilted_unsprung_mount_so_no_phantom_travel() {
        // Arrange — a stationary pre-roll then a "riding" phase (IMU0 oscillates so
        // the detector flags non-stationary). The bike is rigid — no real suspension
        // travel — so each unsprung IMU sees exactly the chassis specific force,
        // transformed by its TRUE mount (the reference coarse pick plus a 7° pitch
        // error). Without mount auto-refinement the mis-leveled unsprung gravity
        // projects onto the fork tangent and ramps phantom travel toward the barrier;
        // refining each coarse mount against the static window cancels it.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize;
        let ride = 200usize;
        let n = pre + ride;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let tilt_err = UnitQuaternion::from_euler_angles(0.0, 7_f64.to_radians(), 0.0);
        let imu1_true = tilt_err * geometry.imu1.mount;
        let imu2_true = tilt_err * geometry.imu2.as_ref().unwrap().mount;

        // Chassis specific force per sample: gravity at rest, then gravity + a brisk
        // 5 Hz forward oscillation while riding. A *cosine* (starts at full amplitude)
        // makes the riding phase non-stationary from its very first sample, so the
        // detector's trailing window separates the static pre-roll cleanly at the
        // boundary — the mount fit then sees only true-gravity samples. The steady
        // high variance also keeps every riding window non-stationary (no wheel-ZUPT
        // fires to mask a phantom drive). Zero gyro keeps the lever term zero (no
        // rotational transfer to model). The chassis accel shape is irrelevant to the
        // *correct* answer — with the mount refined the unsprung-minus-chassis
        // difference is zero whatever the motion — so travel must stay at topout.
        let osc = 2.0 * std::f64::consts::PI * 5.0; // 5 Hz
        let chassis_accel: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                if i < pre {
                    g
                } else {
                    let t = (i - pre) as f64 * dt;
                    g + Vector3::new(4.0 * (osc * t).cos(), 0.0, 0.0)
                }
            })
            .collect();
        let zero = vec![Vector3::zeros(); n];
        // Each IMU reads the chassis specific force in ITS OWN sensor frame, i.e.
        // mount⁻¹·chassis — including IMU0, whose mount is no longer identity (X-rear/
        // Y-right). Building IMU0's series as `mount⁻¹·chassis` (not `chassis`) keeps
        // the synthetic bike physically rigid for *any* mount, so `run()` maps every
        // IMU back to the same chassis accel ⇒ zero differential ⇒ no phantom travel.
        let imu0 = ImuSeries {
            gyro: zero.clone(),
            accel: chassis_accel.iter().map(|a| geometry.imu0.mount.inverse() * a).collect(),
        };
        let imu1 = ImuSeries {
            gyro: zero.clone(),
            accel: chassis_accel.iter().map(|a| imu1_true.inverse() * a).collect(),
        };
        let imu2 = ImuSeries {
            gyro: zero,
            accel: chassis_accel.iter().map(|a| imu2_true.inverse() * a).collect(),
        };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
        // Sag prior effectively off: the only thing that could move travel here is a
        // (phantom) wheel drive, isolating the mount-refinement effect. Per-session
        // refinement is opt-in now (calibrated mounts are the default) — this test
        // exercises the refinement path itself, so it opts in.
        let config =
            EstimatorConfig { sag_sigma: 1.0e6, refine_mounts: true, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &geometry, &config);

        // Assert — the rigid bike has no real travel; refinement keeps the front at
        // ~topout through the riding phase (without it the mis-leveled gravity would
        // slam the front toward the travel barrier near 0.17 m).
        let last = &est.final_state;
        assert!(last.d_f.abs() < 0.01, "front phantom travel from tilted mount: {} m", last.d_f);
    }

    #[test]
    fn sag_prior_gate_off_by_default_keeps_rigid_riding_travel_at_topout() {
        // Arrange — a rigid bike on the ground (never airborne) that is *riding*: a
        // stationary pre-roll then a brisk 5 Hz chassis oscillation so the detector
        // flags non-stationary. Mounts are the true reference mounts (no tilt), so the
        // unsprung-minus-chassis differential is ~0 ⇒ no real OR phantom travel drive.
        // The ONLY factor that could lift travel off topout here is the sag prior, and
        // we hand it a TIGHT sigma so, if it fired, it would yank travel hard toward
        // static sag (~46 mm front). Bounds-only (the new default: use_sag_prior =
        // false) must instead leave riding travel resting at topout (~0). Zero gyro
        // keeps the lever term zero, and the |accel0| ≈ 1 g oscillation never trips the
        // airborne gate, so neither topout nor ZUPT fires — the contrast is purely sag.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize;
        let ride = 300usize;
        let n = pre + ride;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let osc = 2.0 * std::f64::consts::PI * 5.0; // 5 Hz, on the ground
        let chassis_accel: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                if i < pre {
                    g
                } else {
                    let t = (i - pre) as f64 * dt;
                    g + Vector3::new(4.0 * (osc * t).cos(), 0.0, 0.0)
                }
            })
            .collect();
        let zero = vec![Vector3::zeros(); n];
        let imu0 = ImuSeries {
            gyro: zero.clone(),
            accel: chassis_accel.iter().map(|a| geometry.imu0.mount.inverse() * a).collect(),
        };
        let imu1 = ImuSeries {
            gyro: zero.clone(),
            accel: chassis_accel.iter().map(|a| geometry.imu1.mount.inverse() * a).collect(),
        };
        let imu2 = ImuSeries {
            gyro: zero,
            accel: chassis_accel
                .iter()
                .map(|a| geometry.imu2.as_ref().unwrap().mount.inverse() * a)
                .collect(),
        };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };

        // Act — bounds-only (the default) with a tight sag sigma that WOULD dominate if
        // the prior were allowed to fire; then the same run with the prior explicitly
        // enabled, to prove the assertion is not vacuous (the prior still pulls to sag).
        let bounds_only = EstimatorConfig { sag_sigma: 0.05, ..EstimatorConfig::default() };
        let est_off = run(&input, &geometry, &bounds_only);
        let with_sag = EstimatorConfig { use_sag_prior: true, ..bounds_only };
        let est_on = run(&input, &geometry, &with_sag);

        // Assert — median riding travel. Bounds-only rests at topout (~0); the sag
        // prior, when on, pulls it up toward static sag (~46 mm).
        let sag_front = geometry.front_sag * geometry.front_travel_max;
        let med_off = est_off.front_dynamic_sag().expect("riding samples present");
        let med_on = est_on.front_dynamic_sag().expect("riding samples present");
        assert!(med_off.abs() < 0.01, "bounds-only travel should rest at topout, got {med_off} m");
        assert!(
            med_on > 0.5 * sag_front,
            "sag prior (on) should pull travel toward sag {sag_front} m, got {med_on} m"
        );
    }

    #[test]
    fn brief_midair_gap_does_not_snap_travel_to_sag_before_landing() {
        // Arrange — a stationary pre-roll, then a sustained free-fall whose IMU0
        // specific force briefly dips back above the airborne threshold mid-air (the
        // real-jump "flicker": sensor noise / a clipped peak). The bike is rigid (the
        // unsprung IMUs mirror the chassis through their mounts ⇒ diff-accel ≈ 0, no
        // real travel), so travel must stay at topout (0) across the WHOLE float. The
        // gap-bridge keeps the topout reference firing through the flicker; without it
        // the gap opens a coasting window in which the sag prior — here deliberately
        // turned ON (tight σ) to expose the failure — would re-capture travel toward
        // sag (~46 mm) before landing. Zero gyro ⇒ the lever term is zero.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize; // stationary pre-roll for the orientation/mount fit
        let fl1 = 20usize; // float, first segment (airborne)
        let gap = 5usize; // brief above-threshold flicker (0.05 s < 0.10 s ⇒ bridged)
        let fl2 = 20usize; // float, second segment (airborne)
        let post = 20usize; // settle
        let n = pre + fl1 + gap + fl2 + post;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let osc = 2.0 * std::f64::consts::PI * 4.0; // 4 Hz

        // Chassis specific force: gravity at rest; a brisk sub-threshold oscillation
        // (|a| ≤ 1.5 < 2.5 m/s² ⇒ airborne, and varying ⇒ non-stationary) during the
        // float; gravity (|a| = g > 2.5 ⇒ NOT airborne) in the mid-air gap and settle.
        let airborne_seg = |k: usize| Vector3::new(1.5 * (osc * (k as f64) * dt).cos(), 0.0, 0.0);
        let chassis: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                if i < pre {
                    g
                } else if i < pre + fl1 {
                    airborne_seg(i - pre)
                } else if i < pre + fl1 + gap {
                    g // flicker: back above the free-fall threshold
                } else if i < pre + fl1 + gap + fl2 {
                    airborne_seg(i - pre)
                } else {
                    g
                }
            })
            .collect();
        let zero = vec![Vector3::zeros(); n];
        let imu1_mount = geometry.imu1.mount;
        let imu2_mount = geometry.imu2.as_ref().unwrap().mount;
        let imu0 = ImuSeries { gyro: zero.clone(), accel: chassis.clone() };
        let imu1 = ImuSeries { gyro: zero.clone(), accel: chassis.iter().map(|a| imu1_mount.inverse() * a).collect() };
        let imu2 = ImuSeries { gyro: zero, accel: chassis.iter().map(|a| imu2_mount.inverse() * a).collect() };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
        // Sag ON (tight) so the failure mode (snap-to-sag in the gap) is exposed; the
        // bridge must keep travel at topout despite it.
        let config = EstimatorConfig { sag_sigma: 0.05, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &geometry, &config);

        // Assert — travel never climbs toward sag anywhere across the float (incl. the
        // bridged gap): it stays pinned at topout. (Without the gap-bridge the gap
        // window snaps it to ~40+ mm before landing.)
        let max_travel = est.front_travel[pre..pre + fl1 + gap + fl2]
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(max_travel < 0.008, "travel snapped toward sag mid-float: {} m", max_travel);
    }

    #[test]
    fn rough_low_g_ground_with_large_diff_accel_is_not_flagged_airborne() {
        // Arrange — a stationary pre-roll, then a "rough trail" window whose chassis
        // specific force oscillates *below* the airborne accel threshold (|a| ≤ 1.5 <
        // 2.5 m/s² ⇒ the absolute free-fall criterion alone would fire) — but the front
        // wheel is being driven hard by terrain, so the lever-compensated differential
        // specific force is large (12 m/s² along the fork axis). A grounded-but-light
        // chassis is NOT free fall: the relative criterion must veto the airborne flag
        // so the topout reference never fires. With only the absolute criterion the
        // sustained sub-threshold window reads as airborne and the strong topout
        // reference pins travel at 0 (the false snap the rider saw at 1:32.4); with the
        // diff-accel veto, no topout fires and the wheel-drive + sag prior carry travel
        // well off topout. Zero gyro ⇒ the lever term is zero.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize; // stationary pre-roll for the orientation/mount fit
        let rough = 60usize; // grounded rough-trail window (light chassis, driven wheel)
        let n = pre + rough;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let osc = 2.0 * std::f64::consts::PI * 4.0; // 4 Hz
        let front_axis = geometry.front_tangent();

        // Chassis: gravity at rest; a brisk sub-threshold oscillation during the rough
        // window (|a| ≤ 1.5 < 2.5 ⇒ passes the absolute airborne criterion).
        let chassis: Vec<Vector3<f64>> = (0..n)
            .map(|i| if i < pre { g } else { Vector3::new(1.5 * (osc * ((i - pre) as f64) * dt).cos(), 0.0, 0.0) })
            .collect();
        // Front unsprung: chassis + a large terrain drive along the fork axis during the
        // rough window ⇒ |diff-accel| = 12 m/s², far above any sane diff threshold. (At
        // rest it mirrors the chassis ⇒ diff ≈ 0.) Stored in the IMU body frame, so the
        // mount maps it back to chassis frame inside the run.
        let imu1_mount = geometry.imu1.mount;
        let imu2_mount = geometry.imu2.as_ref().unwrap().mount;
        let zero = vec![Vector3::zeros(); n];
        let accel1_chassis: Vec<Vector3<f64>> = (0..n)
            .map(|i| if i < pre { chassis[i] } else { chassis[i] + front_axis * 12.0 })
            .collect();
        let imu0 = ImuSeries { gyro: zero.clone(), accel: chassis.clone() };
        let imu1 = ImuSeries { gyro: zero.clone(), accel: accel1_chassis.iter().map(|a| imu1_mount.inverse() * a).collect() };
        // Rear stays rigid with the chassis (diff ≈ 0) — the front veto alone must reject airborne.
        let imu2 = ImuSeries { gyro: zero, accel: chassis.iter().map(|a| imu2_mount.inverse() * a).collect() };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
        let config = EstimatorConfig::default();

        // Act
        let est = run(&input, &geometry, &config);

        // Assert — travel is NOT pinned at topout through the rough window: the
        // wheel-drive + sag prior carry it well off 0. (With only the absolute
        // criterion the window reads as airborne and the topout reference holds travel
        // at ~0.)
        let max_travel = est.front_travel[pre..n].iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_travel > 0.05, "travel was falsely pinned at topout on rough ground: {} m", max_travel);
    }

    #[test]
    fn biased_gyro_recovered_so_attitude_holds() {
        // Arrange — a true-stationary recording but IMU0 reads a constant yaw-rate
        // offset; orientation + ZARU must absorb it so attitude does not wind up.
        let n = 1000;
        let bias = Vector3::new(0.0, 0.0, 0.03);
        let up = Vector3::new(0.0, 0.0, GRAVITY);
        let imu0 = ImuSeries { gyro: vec![bias; n], accel: vec![up; n] };
        let still = || ImuSeries { gyro: vec![Vector3::zeros(); n], accel: vec![up; n] };
        let input = EstimatorInput { dt: 0.01, imu0, imu1: Some(still()), imu2: Some(still()), gps: vec![] };

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &EstimatorConfig::default());

        // Assert — bias learned, attitude held level despite the raw gyro offset.
        let last = &est.final_state;
        assert!((last.b_g0.z - 0.03).abs() < 2e-3, "bias not recovered: {}", last.b_g0.z);
        assert!(last.r_chassis.angle() < 5e-3, "attitude wound up: {}", last.r_chassis.angle());
    }

    #[test]
    fn sustained_free_fall_does_not_nan_or_inject_tilt() {
        // Arrange — a level stationary pre-roll, then a sustained free fall (IMU0
        // specific force collapses toward 0, oscillating through ~0 so it stays
        // non-stationary and reads airborne), then settle. Gyro is zero throughout and
        // the bike is level, so TRUE attitude stays identity the whole time.
        // GravityLeveling must NOT run in free fall: normalizing a ~0 "up" vector both
        // NaN-poisons the run and (short of NaN) inflates the Jacobian ~9x and injects
        // bogus tilt. Gated off when airborne, attitude propagates open-loop on the
        // (zero) gyro and stays level.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize;
        let fall = 40usize; // 0.4 s free fall, well past the airborne min-duration gate
        let post = 40usize;
        let n = pre + fall + post;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let osc = 2.0 * std::f64::consts::PI * 4.0;
        let m0 = geometry.imu0.mount;
        let imu1_mount = geometry.imu1.mount;
        let imu2_mount = geometry.imu2.as_ref().unwrap().mount;
        // Chassis specific force: gravity at rest; a near-zero oscillation through 0
        // (|a| <= 0.3 < 2.5 => airborne, varying magnitude => non-stationary) in free fall.
        let chassis: Vec<Vector3<f64>> = (0..n)
            .map(|i| if i < pre || i >= pre + fall { g } else { Vector3::new(0.3 * (osc * ((i - pre) as f64) * dt).sin(), 0.0, 0.0) })
            .collect();
        let zero = vec![Vector3::zeros(); n];
        // Each IMU reads the chassis specific force in its OWN sensor frame (mount^-1).
        let imu0 = ImuSeries { gyro: zero.clone(), accel: chassis.iter().map(|a| m0.inverse() * a).collect() };
        let imu1 = ImuSeries { gyro: zero.clone(), accel: chassis.iter().map(|a| imu1_mount.inverse() * a).collect() };
        let imu2 = ImuSeries { gyro: zero, accel: chassis.iter().map(|a| imu2_mount.inverse() * a).collect() };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };

        // Act
        let est = run(&input, &geometry, &EstimatorConfig::default());

        // Assert — no non-finite output, and no bogus tilt injected by the float.
        let all_finite = est.front_travel.iter().chain(est.rear_travel.iter())
            .chain(est.front_velocity.iter()).chain(est.rear_velocity.iter())
            .all(|v| v.is_finite());
        assert!(all_finite, "free fall produced non-finite output (NaN-poisoned run)");
        let angle = est.final_state.r_chassis.angle();
        assert!(angle.is_finite() && angle < 0.02, "free fall injected bogus tilt: {} rad", angle);
    }

    #[test]
    fn gps_samples_from_lookup_builds_nav_velocity_at_event_times() {
        // Arrange — two fixes: 3.6 km/h due east (course 90°) at t = 1.25 s, then
        // 7.2 km/h due north (course 0°) at t = 2.5 s. Heading is wire-format
        // centidegrees; both have fix quality 1.
        let mut channels = HashMap::new();
        channels.insert("GPS_SpeedKmh".to_string(), (vec![3.6, 7.2], 0.0));
        channels.insert("GPS_Heading".to_string(), (vec![9000.0, 0.0], 0.0));
        channels.insert("GPS_FixQuality".to_string(), (vec![1.0, 1.0], 0.0));
        let mut times = HashMap::new();
        times.insert("GPS_SpeedKmh".to_string(), vec![1.25, 2.5]);
        let lookup = FakeLookup { channels, times };

        // Act
        let gps = gps_samples_from_lookup(&lookup);

        // Assert — nav frame is X = north, Y = west, Z = up: east = −Y. 3.6 km/h =
        // 1 m/s east ⇒ (0, −1, 0); 7.2 km/h north ⇒ (2, 0, 0). Times pass through.
        assert_eq!(gps.len(), 2);
        assert_relative_eq!(gps[0].time_s, 1.25, epsilon = 1e-12);
        assert_relative_eq!(gps[0].velocity, Vector3::new(0.0, -1.0, 0.0), epsilon = 1e-9);
        assert_relative_eq!(gps[1].time_s, 2.5, epsilon = 1e-12);
        assert_relative_eq!(gps[1].velocity, Vector3::new(2.0, 0.0, 0.0), epsilon = 1e-9);
    }

    #[test]
    fn gps_samples_from_lookup_drops_no_fix_samples() {
        // Arrange — first fix has quality 0 (no fix): its speed/course are garbage
        // and must be dropped; the second (quality 2) survives.
        let mut channels = HashMap::new();
        channels.insert("GPS_SpeedKmh".to_string(), (vec![99.9, 7.2], 0.0));
        channels.insert("GPS_Heading".to_string(), (vec![12345.0, 0.0], 0.0));
        channels.insert("GPS_FixQuality".to_string(), (vec![0.0, 2.0], 0.0));
        let mut times = HashMap::new();
        times.insert("GPS_SpeedKmh".to_string(), vec![0.5, 1.0]);
        let lookup = FakeLookup { channels, times };

        // Act
        let gps = gps_samples_from_lookup(&lookup);

        // Assert
        assert_eq!(gps.len(), 1);
        assert_relative_eq!(gps[0].time_s, 1.0, epsilon = 1e-12);
        assert_relative_eq!(gps[0].velocity, Vector3::new(2.0, 0.0, 0.0), epsilon = 1e-9);
    }

    #[test]
    fn run_gps_velocity_anchors_chassis_velocity_against_accel_bias() {
        // Arrange — a moving log with a persistent +0.3 m/s² forward specific-force
        // error that would ramp velocity unboundedly on the IMU alone. A brisk 5 Hz
        // vertical oscillation at FULL amplitude from the first sample keeps every
        // window non-stationary (a quiet lead-in would fire one ZUPT and crush the
        // velocity variance, starving the GPS gain), so GPS is the only velocity
        // anchor. Fixes arrive at 2 Hz saying 4 m/s north.
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let n = 1000; // 10 s
        let m0 = geometry.imu0.mount;
        let osc = 2.0 * std::f64::consts::PI * 5.0;
        let chassis_accel: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                Vector3::new(0.3, 0.0, GRAVITY + 0.8 * (osc * t).cos())
            })
            .collect();
        let imu0 = ImuSeries {
            gyro: vec![Vector3::zeros(); n],
            accel: chassis_accel.iter().map(|a| m0.inverse() * a).collect(),
        };
        let gps: Vec<GpsSample> = (0..20)
            .map(|k| GpsSample { time_s: 0.5 * k as f64, velocity: Vector3::new(4.0, 0.0, 0.0) })
            .collect();
        let input = EstimatorInput { dt, imu0, imu1: None, imu2: None, gps };
        // Zero latency so fixes land on their exact sample; snug sigma (plumbing test).
        let config =
            EstimatorConfig { gps_latency_s: 0.0, gps_sigma: 0.2, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &geometry, &config);

        // Assert — velocity holds near the GPS anchor instead of ramping toward
        // 0.3·10 = 3 m/s of accumulated bias error.
        let vx = est.final_state.v_chassis.x;
        assert!((vx - 4.0).abs() < 0.5, "GPS failed to anchor velocity: vx = {vx}");
    }

    #[test]
    fn run_smooth_off_forward_wheel_outputs_match_standalone_replay() {
        // Arrange — the decoupling proof behind the RTS pass: the wheel {w, ẇ} block
        // of the 24-DOF filter must be reproducible by the standalone 2-state forward
        // filter over the trace's drives + flags. A stationary pre-roll (wheel ZUPT
        // fires), a riding phase with a real 3 Hz front drive (unsprung sees an extra
        // oscillation along the fork axis), and an airborne window (chassis specific
        // force → small, unsprung rigid with chassis ⇒ topout + in-air ZWV fire)
        // exercise every wheel factor except sag (off by default).
        let geometry = BikeGeometry::reference_bike();
        let dt = 0.01;
        let pre = 120usize;
        let ride = 200usize;
        let air = 30usize;
        let post = 50usize;
        let n = pre + ride + air + post;
        let g = Vector3::new(0.0, 0.0, GRAVITY);
        let osc = 2.0 * std::f64::consts::PI * 3.0;
        let front_axis = geometry.front_tangent();
        let m0 = geometry.imu0.mount;
        let imu1_mount = geometry.imu1.mount;
        let imu2_mount = geometry.imu2.as_ref().unwrap().mount;
        let chassis: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                if i < pre || i >= pre + ride + air {
                    g
                } else if i < pre + ride {
                    let t = (i - pre) as f64 * dt;
                    g + Vector3::new(3.0 * (osc * t).cos(), 0.0, 0.0)
                } else {
                    let t = (i - pre - ride) as f64 * dt;
                    Vector3::new(0.4 * (osc * t).sin(), 0.0, 0.0) // free fall
                }
            })
            .collect();
        // Front unsprung: rigid with the chassis except a 2 m/s² 3 Hz drive along the
        // fork axis during the riding phase (real suspension motion).
        let accel1: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                let extra = if i >= pre && i < pre + ride {
                    let t = (i - pre) as f64 * dt;
                    front_axis * (2.0 * (osc * t).sin())
                } else {
                    Vector3::zeros()
                };
                imu1_mount.inverse() * (chassis[i] + extra)
            })
            .collect();
        let zero = vec![Vector3::zeros(); n];
        let imu0 = ImuSeries {
            gyro: zero.clone(),
            accel: chassis.iter().map(|a| m0.inverse() * a).collect(),
        };
        let imu1 = ImuSeries { gyro: zero.clone(), accel: accel1 };
        let imu2 = ImuSeries {
            gyro: zero,
            accel: chassis.iter().map(|a| imu2_mount.inverse() * a).collect(),
        };
        let input = EstimatorInput { dt, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
        let config = EstimatorConfig { smooth: false, ..EstimatorConfig::default() };

        // Act — the engine's forward wheel outputs, and the standalone replay over
        // the trace's exact drives + flags with the same tuning.
        let (est, trace) = run_trace(&input, &geometry, &config);
        let params = smooth::WheelParams {
            dt,
            q_pos: config.process_noise.wheel_pos_rw.powi(2) * dt,
            q_vel: config.process_noise.wheel_vel_rw.powi(2) * dt,
            init_travel_var: config.init_std.wheel_travel.powi(2),
            init_vel_var: config.init_std.wheel_velocity.powi(2),
            zupt_sigma: config.zupt_sigma,
            topout_sigma: config.topout_sigma,
            barrier_sigma: config.barrier_sigma,
            sag_sigma: config.sag_sigma,
            sag: geometry.front_sag * geometry.front_travel_max,
            travel_max: geometry.front_travel_max,
            use_sag_prior: config.use_sag_prior,
        };
        let replay =
            smooth::forward_filter(&trace.front_drive, &trace.stationary, &trace.airborne, &params);

        // Assert — the 2-state replay reproduces the 24-DOF filter's front wheel
        // marginal essentially exactly (the wheel block is decoupled; differences are
        // pure f64 rounding), and the profile is non-trivial (real travel recovered).
        assert_eq!(replay.len(), n);
        for i in 0..n {
            assert!(
                (replay[i].w - est.front_travel[i]).abs() < 1e-9,
                "travel diverged at {i}: replay {} vs engine {}",
                replay[i].w,
                est.front_travel[i]
            );
            assert!(
                (replay[i].wv - est.front_velocity[i]).abs() < 1e-9,
                "velocity diverged at {i}: replay {} vs engine {}",
                replay[i].wv,
                est.front_velocity[i]
            );
        }
        let max_travel = est.front_travel.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_travel > 0.005, "test profile produced no real travel: {max_travel}");
    }
}
