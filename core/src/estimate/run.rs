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
use crate::estimate::measurements::gps::{GpsVelocity, VerticalVelocity};
use crate::estimate::measurements::{I_THETA, I_V};
use crate::estimate::measurements::prior::{SagPrior, TopoutReference, TravelBarrier, Wheel};
use crate::estimate::measurements::zupt::{GyroBias, ZeroAngularRate, ZeroVelocity, ZeroWheelVelocity};
use crate::estimate::model::{ImuInput, SampleContext};
use crate::estimate::orient::{fit_from_window, fit_imu0_mount_from_riding, refine_mount_from_window, MountFit};
use crate::estimate::process::{wheel_drive_accel, MtbProcess, ProcessNoiseConfig, GRAVITY};
use crate::estimate::schema::StateSchema;
use crate::estimate::smooth;
use crate::estimate::state::MtbState;
use crate::rotation::lever_arm_accel;
use nalgebra::{UnitQuaternion, Vector3};

/// One IMU's uniform-rate samples in its own **sensor frame** (the runner applies
/// the per-IMU mount). `gyro` rad/s, `accel` (specific force) m/s².
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
    /// (the module logs no vertical velocity; the horizontal factor ignores it).
    pub velocity: Vector3<f64>,
    /// Vertical velocity, m/s, positive up, fitted from the GPS altitude over a few
    /// seconds. None when the log has no altitude or too few fixes around this
    /// one. Loose: it stops the vertical velocity state drifting, it does not
    /// track a jump.
    pub up_mps: Option<f64>,
}

/// Degrees-per-second → radians-per-second (gyro channels are stored in `dps`).
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// A channel's own mean sample rate from its recorded times, Hz (0 when it has
/// fewer than two samples or no time span).
fn recorded_rate_hz(t_us: &[i64]) -> f64 {
    match (t_us.first(), t_us.last()) {
        (Some(&a), Some(&b)) if t_us.len() >= 2 && b > a => (t_us.len() - 1) as f64 * 1e6 / (b - a) as f64,
        _ => 0.0,
    }
}

/// The run's uniform time grid, µs on the recording timeline: one point per IMU0
/// gyro sample, evenly spaced over that stream's recorded span (accel times when
/// IMU0 has no gyro axis). Returns the grid and its period in seconds, or None
/// when IMU0 is absent or its recorded times span nothing.
///
/// The three IMUs free-run on their own oscillators (461 / 463 / 468 Hz on one
/// IDL1 ride) and an IDL1 merged accel channel runs at twice its gyro's rate, so
/// a sample index means a different instant on every stream. Every channel is
/// therefore resampled onto this one grid by its recorded time.
fn imu0_grid(lookup: &dyn ChannelLookup) -> Option<(Vec<i64>, f64)> {
    let names = IMU_CHANNEL_NAMES[0];
    let ch = [3, 4, 5, 0, 1, 2].iter().find_map(|&i| lookup.lookup(names[i]))?;
    let n = ch.t_us.len().min(ch.samples.len());
    if n < 2 {
        return None;
    }
    let (first, last) = (ch.t_us[0], ch.t_us[n - 1]);
    if last <= first {
        return None;
    }
    let period_us = (last - first) as f64 / (n - 1) as f64;
    let grid = (0..n).map(|i| first + (i as f64 * period_us).round() as i64).collect();
    Some((grid, period_us / 1e6))
}

/// Resamples `v`, recorded at ascending times `t_us` (µs), onto the ascending
/// `grid_us`. Catmull-Rom cubic through the four surrounding samples where they
/// are evenly spaced (each neighbouring interval within 0.5–1.5× the bracketing
/// one); linear across a recording gap and at the two ends; held at the first /
/// last sample outside the recorded span. An empty channel reads as zeros.
fn resample_onto(t_us: &[i64], v: &[f64], grid_us: &[i64]) -> Vec<f64> {
    let n = t_us.len().min(v.len());
    if n == 0 {
        return vec![0.0; grid_us.len()];
    }
    let mut k = 0usize; // first index with t_us[k] >= the query time
    grid_us
        .iter()
        .map(|&tq| {
            while k < n && t_us[k] < tq {
                k += 1;
            }
            if k == 0 {
                return v[0];
            }
            if k >= n {
                return v[n - 1];
            }
            let (t0, t1) = (t_us[k - 1], t_us[k]);
            let h = (t1 - t0) as f64;
            if h <= 0.0 {
                return v[k];
            }
            let u = (tq - t0) as f64 / h;
            let linear = v[k - 1] + u * (v[k] - v[k - 1]);
            if k < 2 || k + 1 >= n {
                return linear;
            }
            let even = |a: i64| (a as f64) > 0.5 * h && (a as f64) < 1.5 * h;
            if !even(t0 - t_us[k - 2]) || !even(t_us[k + 1] - t1) {
                return linear;
            }
            let (p0, p1, p2, p3) = (v[k - 2], v[k - 1], v[k], v[k + 1]);
            p1 + 0.5 * u * ((p2 - p0) + u * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) + u * (3.0 * (p1 - p2) + p3 - p0)))
        })
        .collect()
}

/// A channel faster than this multiple of the grid rate is low-passed before it
/// is resampled, so the content above the grid's Nyquist does not alias.
const ANTIALIAS_RATE_RATIO: f64 = 1.25;

/// Pulls one IMU's six channels (`IMU{n}_{Accel,Gyro}{X,Y,Z}`) from a lookup onto
/// `grid_us` and converts to SI sensor-frame vectors: accel `g → m/s²`, gyro
/// `dps → rad/s`. Returns None if **no** axis is present.
///
/// Each axis is resampled by its own recorded times ([`resample_onto`]). An axis
/// recorded faster than [`ANTIALIAS_RATE_RATIO`] × the grid rate (the IDL1 merged
/// accel, at twice the gyro rate) is first low-passed zero-phase at 0.4 × the
/// grid rate (4th-order Butterworth, forward-backward).
///
/// **Axis-tolerant:** an absent axis is zero-filled rather than rejecting the whole
/// IMU. This is safe when the missing axis is orthogonal to the suspension-travel
/// plane (e.g. some loggers omit the unsprung IMUs' *lateral* accel, which the
/// sagittal-plane fork-tangent projection discards anyway). If a *travel-relevant*
/// axis were missing the zero-fill would silently lose signal — the correct mount
/// orientation determines which axis is lateral, so this pairs with §7 orientation.
fn imu_series_from_lookup(
    lookup: &dyn ChannelLookup,
    imu: usize,
    grid_us: &[i64],
    grid_dt: f64,
) -> Option<ImuSeries> {
    let names = IMU_CHANNEL_NAMES[imu];
    let ch: Vec<Option<crate::math::eval::LookupChannel>> = names.iter().map(|n| lookup.lookup(n)).collect();
    if ch.iter().all(|c| c.is_none()) {
        return None; // no channels for this IMU at all
    }
    let grid_rate = 1.0 / grid_dt;
    // One axis on the grid (g for accel, dps for gyro), zero-filled if absent.
    let axis = |idx: usize| -> Vec<f64> {
        let Some(c) = ch[idx].as_ref() else {
            return vec![0.0; grid_us.len()];
        };
        let n = c.samples.len().min(c.t_us.len());
        let own_rate = recorded_rate_hz(&c.t_us[..n]);
        let mut v = c.samples[..n].to_vec();
        if idx < 3 && own_rate > 0.0 {
            // **Declip** the accel axes at their own rate: hard landings rail the
            // ±32 g accelerometer, and clipped compression peaks would underestimate
            // integrated travel. The reconstruction fits a smooth pulse over each
            // clipped segment (no-op on clean data, and on a 16 g rail). Gyro is
            // not declipped.
            v = crate::clip_reconstruct::declip(&v, own_rate);
        }
        if own_rate > ANTIALIAS_RATE_RATIO * grid_rate {
            if let Ok(f) = crate::filters::lowpass(&v, 4, 0.4 * grid_rate, own_rate) {
                v = f;
            }
        }
        resample_onto(&c.t_us[..n], &v, grid_us)
    };
    let (ax, ay, az) = (axis(0), axis(1), axis(2));
    let (gx, gy, gz) = (axis(3), axis(4), axis(5));
    let n = grid_us.len();
    let accel = (0..n).map(|i| Vector3::new(ax[i], ay[i], az[i]) * GRAVITY).collect();
    let gyro = (0..n).map(|i| Vector3::new(gx[i], gy[i], gz[i]) * DEG2RAD).collect();
    Some(ImuSeries { gyro, accel })
}

/// GPS course (degrees clockwise from north) → nav-frame unit direction. Nav frame:
/// X = north, Y = west, Z = up (right-handed), so east is −Y and course θ maps to
/// `(cos θ, −sin θ, 0)`.
fn course_to_nav(heading_deg: f64) -> Vector3<f64> {
    let th = heading_deg.to_radians();
    Vector3::new(th.cos(), -th.sin(), 0.0)
}

/// Half-width of the window the GPS vertical velocity is fitted over, s.
const GPS_UP_HALF_WINDOW_S: f64 = 2.5;

/// Vertical velocity at each fix from the GPS altitude: the least-squares slope
/// of altitude (m) against time (s, ascending) over the fixes within
/// [`GPS_UP_HALF_WINDOW_S`] either side, m/s. None where that window holds fewer
/// than three fixes or spans under 2 s.
fn altitude_slopes(times: &[f64], altitude: &[f64]) -> Vec<Option<f64>> {
    let n = times.len().min(altitude.len());
    let (mut lo, mut hi) = (0usize, 0usize);
    (0..n)
        .map(|i| {
            while times[lo] < times[i] - GPS_UP_HALF_WINDOW_S {
                lo += 1;
            }
            while hi < n && times[hi] <= times[i] + GPS_UP_HALF_WINDOW_S {
                hi += 1;
            }
            let m = hi - lo;
            if m < 3 || times[hi - 1] - times[lo] < 2.0 {
                return None;
            }
            let tm = times[lo..hi].iter().sum::<f64>() / m as f64;
            let am = altitude[lo..hi].iter().sum::<f64>() / m as f64;
            let (mut num, mut den) = (0.0, 0.0);
            for k in lo..hi {
                num += (times[k] - tm) * (altitude[k] - am);
                den += (times[k] - tm).powi(2);
            }
            (den > 0.0).then(|| num / den)
        })
        .collect()
}

/// Pulls the GPS velocity fixes from a lookup: `GPS_SpeedKmh` (km/h, physical) ×
/// `GPS_Heading` (physical degrees clockwise from north, ruling R27), stamped
/// with the speed channel's recorded times (seconds on the recording timeline:
/// `sample_times` for an event channel, else the channel's own `t_us`). Each fix
/// also carries a vertical velocity fitted from `GPS_Altitude` when that channel
/// is present ([`altitude_slopes`]). Fixes with `GPS_FixQuality` = 0 (no fix) are
/// dropped when that channel is present. Returns an empty vec when speed/heading
/// are unavailable — GPS aiding is optional.
fn gps_samples_from_lookup(lookup: &dyn ChannelLookup) -> Vec<GpsSample> {
    let (Some(speed), Some(heading)) = (lookup.lookup("GPS_SpeedKmh"), lookup.lookup("GPS_Heading"))
    else {
        return vec![];
    };
    let times: Vec<f64> = match lookup.sample_times("GPS_SpeedKmh") {
        Some(t) => t,
        None => speed.t_us.iter().map(|&t| t as f64 / 1e6).collect(),
    };
    let n = speed.samples.len().min(heading.samples.len()).min(times.len());
    let quality = lookup.lookup("GPS_FixQuality");
    let has_fix = |i: usize| !quality.as_ref().is_some_and(|q| q.samples.get(i).is_some_and(|&v| v < 1.0));
    // Vertical velocity from the altitude of the fixes that have a fix.
    let mut up: Vec<Option<f64>> = vec![None; n];
    if let Some(alt) = lookup.lookup("GPS_Altitude").filter(|a| a.samples.len() >= n) {
        let kept: Vec<usize> = (0..n).filter(|&i| has_fix(i)).collect();
        let t: Vec<f64> = kept.iter().map(|&i| times[i]).collect();
        let a: Vec<f64> = kept.iter().map(|&i| alt.samples[i]).collect();
        for (k, slope) in altitude_slopes(&t, &a).into_iter().enumerate() {
            up[kept[k]] = slope;
        }
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if !has_fix(i) {
            continue; // no fix — the reported speed/course are meaningless
        }
        let speed_mps = speed.samples[i] / 3.6; // km/h → m/s
        let heading_deg = heading.samples[i]; // already physical degrees (ruling R27)
        out.push(GpsSample { time_s: times[i], velocity: speed_mps * course_to_nav(heading_deg), up_mps: up[i] });
    }
    out
}

/// All synchronized inputs for one estimator run. Every IMU series is on one
/// uniform grid of period `dt` starting at `t0_s`, in each IMU's own **sensor
/// frame** (the runner applies the mounts).
#[derive(Debug, Clone)]
pub struct EstimatorInput {
    /// Uniform sample period, s.
    pub dt: f64,
    /// Recording-timeline time of the first sample, s. GPS fix times are on the
    /// same timeline, so a fix at `time_s` belongs to sample `(time_s − t0_s) / dt`.
    pub t0_s: f64,
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
    /// (`GPS_SpeedKmh` + `GPS_Heading`, at their recorded times) from a
    /// [`ChannelLookup`] (e.g. a parsed `SessionHandle`). IMU0 is required;
    /// IMU1/IMU2/GPS are optional.
    ///
    /// The grid is IMU0's gyro stream made uniform ([`imu0_grid`]); every channel
    /// of every IMU is resampled onto it by its own recorded time, so streams at
    /// different rates stay aligned for the whole ride.
    ///
    /// Returns None if IMU0 is absent or its recorded times span nothing.
    pub fn from_lookup(lookup: &dyn ChannelLookup) -> Option<EstimatorInput> {
        let (grid, dt) = imu0_grid(lookup)?;
        let imu0 = imu_series_from_lookup(lookup, 0, &grid, dt)?;
        Some(EstimatorInput {
            dt,
            t0_s: grid[0] as f64 / 1e6,
            imu0,
            imu1: imu_series_from_lookup(lookup, 1, &grid, dt),
            imu2: imu_series_from_lookup(lookup, 2, &grid, dt),
            gps: gps_samples_from_lookup(lookup),
        })
    }
}

/// Where the estimator takes IMU0's mount from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imu0Mount {
    /// Fitted from this session's straight-line riding
    /// ([`fit_imu0_mount_from_riding`]), falling back to the geometry's mount
    /// when its gates fail (too little straight riding, no GPS). For a bike
    /// with no calibration: the fit absorbs how the bike sits with a rider on
    /// it, and is a couple of degrees worse than a calibration.
    Riding,
    /// The geometry's, as given: a calibrated bike. The riding fit is still
    /// run and reported as a check ([`StateEstimate::imu0_riding_check_deg`]).
    Geometry,
    /// The geometry's tilt with the riding fit's forward direction: a bike
    /// whose calibration is a still recording of IMU0 alone, which fixes up
    /// but not which way the bike points.
    GeometryTilt,
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
    /// Gravity-leveling tilt residual std (direction components, ≈ rad) of one
    /// *independent* look at gravity; see `gravity_corr_s` for how often one comes.
    pub gravity_sigma: f64,
    /// Correlation time of the acceleration that disturbs gravity-leveling, s. The
    /// specific force is off gravity for as long as the bike accelerates, which is
    /// seconds, not one sample, so consecutive samples are not independent looks:
    /// the per-sample std is inflated by `√(rate · gravity_corr_s)` to make the
    /// factor worth one look per correlation time. (Treating every sample as
    /// independent made it 20× overconfident at 461 Hz: on one ride ten seconds of
    /// handling the bike drove the accel-bias state to 6 m/s².)
    pub gravity_corr_s: f64,
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
    /// GPS output latency, s: a fix logged at `t` describes the bike at
    /// `t − latency`, and the runner applies each fix at the latency-corrected
    /// sample (offline ⇒ no causality constraint). Used as given when
    /// `fit_gps_latency` is off or the session has too little turning to
    /// measure it ([`fit_gps_latency`]).
    pub gps_latency_s: f64,
    /// Measure the GPS latency from this session ([`fit_gps_latency`]) instead of
    /// assuming `gps_latency_s`. Default **true**: it was 0.40 s on a 3 Hz
    /// receiver and 0.60 s on a 1 Hz one (rides `173461a2`, `4f3f3032`), and at
    /// a moto's turn rates 0.2 s of error is over 10° of direction.
    pub fit_gps_latency: bool,
    /// Turn the chassis frame about its own up so that this ride's straight-line
    /// riding reads zero slip ([`forward_trim`]). Default **true**, for a bike
    /// with no calibration: the riding fit knows forward only to several
    /// degrees. A calibrated bike runs with this off — its forward lives in the
    /// calibration, the same for every ride, so that what differs between two
    /// rides is the riding and not the frame they are read in — and the trim is
    /// still measured and reported ([`StateEstimate::forward_trim_deg`]).
    pub trim_forward: bool,
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
    /// Where IMU0's mount (which way the bike's up and forward lie in the
    /// sensor's axes) comes from. Default [`Imu0Mount::Riding`], for a bike
    /// with no calibration.
    pub imu0_mount: Imu0Mount,
    /// GPS speed at which the heading is first set from the GPS course, m/s. The
    /// initial attitude knows tilt only; until a fix this fast arrives the heading
    /// is unknown and no GPS velocity is fused.
    pub heading_init_min_speed_mps: f64,
    /// Heading std-dev assigned when it is set from the GPS course, rad.
    pub heading_init_sigma: f64,
    /// Std-dev of the GPS-altitude vertical velocity ([`GpsSample::up_mps`]), m/s.
    pub gps_up_sigma: f64,
    /// How long after a fused GPS fix the run still counts as GPS-aided, s. While
    /// GPS-aided and moving, gravity-leveling is off: it assumes the specific
    /// force is gravity, which is false in a corner (it reads a leaned bike as
    /// upright), and the GPS velocity already observes tilt through the strapdown.
    pub gps_hold_s: f64,
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
            gravity_corr_s: 1.0,
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
            fit_gps_latency: true,
            trim_forward: true,
            gps_min_speed_mps: 0.5,
            // Mounts come calibrated from the flat-ground recording baked into the
            // geometry; per-session refitting is opt-in (see `refine_mounts`).
            refine_mounts: false,
            smooth: true,
            imu0_mount: Imu0Mount::Riding,
            heading_init_min_speed_mps: 3.0,
            heading_init_sigma: 10_f64.to_radians(),
            // A slope over ±2.5 s of metre-level altitude noise.
            gps_up_sigma: 1.0,
            gps_hold_s: 2.0,
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
    /// Chassis roll per sample, degrees. Positive ⇒ leaning right.
    pub roll: Vec<f64>,
    /// Chassis pitch per sample, degrees. Positive ⇒ nose up.
    pub pitch: Vec<f64>,
    /// Chassis heading per sample, degrees clockwise from north in `[0, 360)`
    /// (the GPS course convention). `NaN` until the heading has been set from the
    /// GPS course (never, on a run without GPS or one that never moves).
    pub heading: Vec<f64>,
    /// The filter's own 1-sigma uncertainty of `heading` per sample, degrees.
    /// `NaN` wherever `heading` is.
    pub heading_sigma: Vec<f64>,
    /// Direction the bike is travelling per sample, degrees clockwise from north
    /// in `[0, 360)`: the direction of the estimator's own velocity (the
    /// accelerometers carried between GPS fixes and corrected by them), taken at
    /// the rear wheel ([`slip_reference`]). `NaN` wherever `heading` is, and
    /// below [`EstimatorConfig::heading_init_min_speed_mps`], where a velocity
    /// this small has no direction worth the name.
    pub course: Vec<f64>,
    /// `heading − course` per sample, degrees in `(−180, 180]`: how far the frame
    /// points away from where its rear wheel is going. Positive = nose to the
    /// right of travel. `NaN` wherever `course` is.
    pub slip: Vec<f64>,
    /// The filter's own 1-sigma uncertainty of `slip` per sample, degrees: the
    /// heading's and the travel direction's, combined as independent. `NaN`
    /// wherever `slip` is.
    pub slip_sigma: Vec<f64>,
    /// The riding-data fit of IMU0's mount ([`fit_imu0_mount_from_riding`]) and
    /// whether its gates passed. It is the mount in use only under
    /// [`Imu0Mount::Riding`]; otherwise it is a check on the calibration.
    pub imu0_mount: Option<MountFit>,
    /// Angle between the mount in use and the riding fit, degrees, when the
    /// riding fit passed its gates and is not itself the mount in use. A few
    /// degrees is the fit's own error plus how the bike sits with a rider on
    /// it; much more says a sensor has moved since the calibration.
    pub imu0_riding_check_deg: Option<f64>,
    /// The GPS latency the run used, s: measured from the session when
    /// `gps_latency_fitted`, else [`EstimatorConfig::gps_latency_s`].
    pub gps_latency_s: f64,
    /// Whether `gps_latency_s` was measured from this session.
    pub gps_latency_fitted: bool,
    /// How far the chassis frame would have to turn about its up for this ride's
    /// straight-line riding to read zero slip, degrees, positive = forward to
    /// the left (the frame reads to the right of travel). None when the ride has
    /// too little straight riding to say. With `forward_trim_applied` the
    /// outputs are in the frame turned by it; without, it is a check on the
    /// bike's calibration, whose forward was used as given.
    pub forward_trim_deg: Option<f64>,
    /// Whether the outputs (roll, pitch, heading, slip, body accelerations) are
    /// in the frame turned by `forward_trim_deg`
    /// ([`EstimatorConfig::trim_forward`]).
    pub forward_trim_applied: bool,
    /// Gravity-removed longitudinal acceleration per sample, g. Positive ⇒
    /// accelerating forward.
    pub accel_long: Vec<f64>,
    /// Gravity-removed lateral acceleration per sample, g. Positive ⇒
    /// accelerating right.
    pub accel_lat: Vec<f64>,
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

/// Longest gap between two GPS fixes the speed is interpolated across, s.
const GPS_SPEED_MAX_GAP_S: f64 = 2.5;

/// GPS speed (m/s) and path pitch (rad, positive climbing, from the vertical
/// velocity over the speed) on the run's grid of `n` samples: linear between
/// consecutive fixes at their latency-corrected times, `NaN` before the first
/// fix, after the last, across a gap longer than [`GPS_SPEED_MAX_GAP_S`], and
/// (pitch only) where a fix has no vertical velocity.
fn gps_speed_on_grid(gps: &[GpsSample], t0_s: f64, latency_s: f64, dt: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![f64::NAN; n];
    let mut pitch = vec![f64::NAN; n];
    let slope = |s: &GpsSample| s.up_mps.map_or(f64::NAN, |up| up.atan2(s.velocity.xy().norm()));
    for pair in gps.windows(2) {
        let (ta, tb) = (pair[0].time_s - t0_s - latency_s, pair[1].time_s - t0_s - latency_s);
        if tb <= ta || tb - ta > GPS_SPEED_MAX_GAP_S {
            continue;
        }
        let (va, vb) = (pair[0].velocity.xy().norm(), pair[1].velocity.xy().norm());
        let first = (ta / dt).ceil().max(0.0) as usize;
        let last = ((tb / dt).floor() as isize).min(n as isize - 1);
        for i in first as isize..=last {
            let u = (i as f64 * dt - ta) / (tb - ta);
            out[i as usize] = va + u * (vb - va);
            pitch[i as usize] = slope(&pair[0]) + u * (slope(&pair[1]) - slope(&pair[0]));
        }
    }
    (out, pitch)
}

/// Sets the heading from a GPS velocity `vel` (nav frame, m/s): rotates the whole
/// belief about nav-up so the chassis +X points along the GPS course, and gives
/// the heading a std-dev of `sigma` (rad). The attitude error is body-frame
/// (right perturbation) so its covariance is unchanged by a nav-frame rotation;
/// the velocity error is nav-frame and rotates with it. Returns `fs` unchanged
/// when the chassis +X is vertical (no heading to speak of).
fn align_heading(fs: &FilterState, vel: Vector3<f64>, sigma: f64) -> FilterState {
    let fwd = fs.x.r_chassis * Vector3::x();
    if fwd.xy().norm() < 1e-3 {
        return fs.clone();
    }
    let turn = vel.y.atan2(vel.x) - fwd.y.atan2(fwd.x);
    let rz = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), turn);
    let mut x = fs.x.clone();
    x.r_chassis = rz * fs.x.r_chassis;
    x.v_chassis = rz * fs.x.v_chassis;
    let n = MtbState::DOF;
    let mut t = nalgebra::DMatrix::<f64>::identity(n, n);
    let rz_mat = rz.to_rotation_matrix().into_inner();
    for r in 0..3 {
        for c in 0..3 {
            t[(I_V + r, I_V + c)] = rz_mat[(r, c)];
        }
    }
    let mut p = &t * &fs.p * t.transpose();
    // Nav-up in the body frame is the direction a heading error perturbs δθ along.
    let up_body = x.r_chassis.inverse() * Vector3::z();
    for r in 0..3 {
        for c in 0..3 {
            p[(I_THETA + r, I_THETA + c)] += sigma.powi(2) * up_body[r] * up_body[c];
        }
    }
    FilterState { x, p }
}

/// Chassis heading, degrees clockwise from north in `[0, 360)`. Nav frame is
/// X = north, Y = west, so east is −Y.
fn heading_deg(r_chassis: &UnitQuaternion<f64>) -> f64 {
    let fwd = r_chassis * Vector3::x();
    (-fwd.y).atan2(fwd.x).to_degrees().rem_euclid(360.0)
}

/// The filter's 1-sigma heading uncertainty, rad: the attitude-error covariance
/// projected onto nav-up expressed in the body frame (the direction a heading
/// error perturbs the body-frame `δθ` along).
fn heading_sigma_rad(fs: &FilterState) -> f64 {
    let up_body = fs.x.r_chassis.inverse() * Vector3::z();
    let mut var = 0.0;
    for r in 0..3 {
        for c in 0..3 {
            var += up_body[r] * fs.p[(I_THETA + r, I_THETA + c)] * up_body[c];
        }
    }
    var.max(0.0).sqrt()
}

/// Longest GPS latency [`fit_gps_latency`] looks for, s.
const GPS_LATENCY_MAX_S: f64 = 1.5;
/// Fewest pairs of consecutive fixes at riding speed for the latency to be measured.
const GPS_LATENCY_MIN_PAIRS: usize = 100;
/// Smallest correlation between the GPS course changes and the gyro's, at the
/// best latency, for it to be believed.
const GPS_LATENCY_MIN_CORR: f64 = 0.5;

/// The GPS latency, s, measured from the session itself: the shift that best
/// lines the course change between consecutive fixes up with the gyro's turn
/// about the bike's up over the same (shifted) interval. No estimator state is
/// involved, so it can be measured before anything is filtered.
///
/// `gyro` is IMU0's angular rate in its sensor axes on the run's grid, rad/s
/// (its offset drops out: both ends of a pair carry it); `up_sensor` is the
/// bike's up in those axes. Only pairs of consecutive fixes above
/// [`crate::estimate::orient::RIDING_MIN_SPEED_MPS`] are used. None when there
/// are fewer than [`GPS_LATENCY_MIN_PAIRS`] of them, the best shift correlates
/// worse than [`GPS_LATENCY_MIN_CORR`], or it sits at an end of the search
/// (0 to [`GPS_LATENCY_MAX_S`]).
pub fn fit_gps_latency(gps: &[GpsSample], gyro: &[Vector3<f64>], up_sensor: Vector3<f64>, t0_s: f64, dt: f64) -> Option<f64> {
    use crate::estimate::orient::RIDING_MIN_SPEED_MPS;
    let n = gyro.len();
    if n < 2 || gps.len() < 2 {
        return None;
    }
    // Gyro heading, rad clockwise (a compass course): the running sum of the turn rate about up.
    let mut turned = Vec::with_capacity(n);
    let mut acc = 0.0;
    for g in gyro {
        turned.push(acc);
        acc -= g.dot(&up_sensor) * dt;
    }
    let turned_at = |t_s: f64| -> Option<f64> {
        let x = (t_s - t0_s) / dt;
        if !(x >= 0.0) || x > (n - 1) as f64 {
            return None;
        }
        let i = (x.floor() as usize).min(n - 2);
        Some(turned[i] + (turned[i + 1] - turned[i]) * (x - i as f64))
    };
    let course = |s: &GpsSample| (-s.velocity.y).atan2(s.velocity.x);
    let mut gaps: Vec<f64> = gps.windows(2).map(|w| w[1].time_s - w[0].time_s).filter(|g| *g > 0.0).collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(f64::total_cmp);
    let usual_gap = gaps[gaps.len() / 2];
    // (start, end, course change) per usable pair.
    let pairs: Vec<(f64, f64, f64)> = gps
        .windows(2)
        .filter(|w| w.iter().all(|s| s.velocity.xy().norm() >= RIDING_MIN_SPEED_MPS))
        .filter(|w| w[1].time_s > w[0].time_s && w[1].time_s - w[0].time_s <= 1.5 * usual_gap)
        .map(|w| {
            let change = (course(&w[1]) - course(&w[0]) + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            (w[0].time_s, w[1].time_s, change)
        })
        // A course that swings faster than 2 rad/s between fixes is a receiver glitch, not a bike.
        .filter(|(a, b, change)| change.abs() <= 2.0 * (b - a))
        .collect();
    if pairs.len() < GPS_LATENCY_MIN_PAIRS {
        return None;
    }
    // Mean squared mismatch and correlation at one latency.
    let score = |latency_s: f64| -> Option<(f64, f64)> {
        let (mut count, mut sq, mut sg, mut sc, mut sgg, mut scc, mut sgc) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for &(a, b, change) in &pairs {
            let (Some(ta), Some(tb)) = (turned_at(a - latency_s), turned_at(b - latency_s)) else { continue };
            let g = tb - ta;
            count += 1.0;
            sq += (g - change).powi(2);
            sg += g;
            sc += change;
            sgg += g * g;
            scc += change * change;
            sgc += g * change;
        }
        if count < GPS_LATENCY_MIN_PAIRS as f64 {
            return None;
        }
        let cov = sgc / count - sg / count * sc / count;
        let var = (sgg / count - (sg / count).powi(2)) * (scc / count - (sc / count).powi(2));
        Some((sq / count, if var > 0.0 { cov / var.sqrt() } else { 0.0 }))
    };
    let step = 0.02;
    let steps = (GPS_LATENCY_MAX_S / step).round() as usize;
    let scores: Vec<Option<(f64, f64)>> = (0..=steps).map(|k| score(k as f64 * step)).collect();
    let best = (0..=steps).filter(|&k| scores[k].is_some()).min_by(|&a, &b| scores[a].unwrap().0.total_cmp(&scores[b].unwrap().0))?;
    let (mismatch, corr) = scores[best]?;
    if best == 0 || best == steps || corr < GPS_LATENCY_MIN_CORR {
        return None;
    }
    // The minimum of the parabola through the best step and its neighbours.
    let refine = match (scores[best - 1], scores[best + 1]) {
        (Some((lo, _)), Some((hi, _))) if lo + hi - 2.0 * mismatch > 0.0 => 0.5 * (lo - hi) / (lo + hi - 2.0 * mismatch),
        _ => 0.0,
    };
    Some((best as f64 + refine) * step)
}

/// Largest turn rate (rad/s, about the vertical) that counts as riding straight
/// for [`forward_trim`].
const FORWARD_TRIM_MAX_TURN_RATE: f64 = 0.05;
/// Largest trim [`forward_trim`] will hand back, rad. A frame more than 20° off
/// the direction of travel is not a mount error to be trimmed away.
const FORWARD_TRIM_MAX: f64 = 0.35;

/// How far to turn the chassis frame about its own up so that straight-line
/// riding reads zero slip, rad anticlockwise seen from above: the median of
/// `slip` (degrees) over the samples riding straight
/// ([`FORWARD_TRIM_MAX_TURN_RATE`], from `heading` over one second) at or above
/// [`crate::estimate::orient::RIDING_MIN_SPEED_MPS`]. A median, so slides and
/// crosswind crabbing in a few straights do not move it. None with less than
/// [`crate::estimate::orient::RIDING_MIN_STRAIGHT_S`] of such riding, or when
/// the answer is beyond [`FORWARD_TRIM_MAX`].
fn forward_trim(heading: &[f64], slip: &[f64], speed: &[f64], dt: f64) -> Option<f64> {
    use crate::estimate::orient::{RIDING_MIN_SPEED_MPS, RIDING_MIN_STRAIGHT_S};
    let n = heading.len().min(slip.len()).min(speed.len());
    let half = ((0.5 / dt).round() as usize).max(1);
    // A sample every 20 ms is plenty for a median and keeps the sort small.
    let stride = ((0.02 / dt).round() as usize).max(1);
    let mut straight = Vec::new();
    let mut i = half;
    while i + half < n {
        let turned = ((heading[i + half] - heading[i - half] + 180.0).rem_euclid(360.0) - 180.0).to_radians();
        let rate = turned / (2.0 * half as f64 * dt);
        if slip[i].is_finite() && rate.abs() <= FORWARD_TRIM_MAX_TURN_RATE && speed[i] >= RIDING_MIN_SPEED_MPS {
            straight.push(slip[i]);
        }
        i += stride;
    }
    if (straight.len() * stride) as f64 * dt < RIDING_MIN_STRAIGHT_S {
        return None;
    }
    straight.sort_by(f64::total_cmp);
    let trim = straight[straight.len() / 2].to_radians();
    (trim.abs() <= FORWARD_TRIM_MAX).then_some(trim)
}

/// Re-states `est` in the chassis frame turned by `trim` (rad, anticlockwise
/// seen from above) about its own up. Exact, and needs no second run: nothing
/// in the filter depends on which way the chassis X axis points within the
/// bike's level plane, so the turn only relabels the outputs. The heading and
/// slip drop by the trim (exactly so on a level bike), the travel direction is
/// unchanged, and roll, pitch and the body accelerations are re-read about the
/// turned axes.
fn apply_forward_trim(est: &mut StateEstimate, trim: f64) {
    let rz = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), trim);
    let (sin, cos) = trim.sin_cos();
    for i in 0..est.roll.len() {
        // Rebuild the attitude from what was stored; the yaw is the heading where
        // there is one, and where there is none it does not enter roll or pitch.
        let yaw = if est.heading[i].is_finite() { -est.heading[i].to_radians() } else { 0.0 };
        let r = UnitQuaternion::from_euler_angles(est.roll[i].to_radians(), -est.pitch[i].to_radians(), yaw) * rz;
        let (roll, pitch) = crate::estimate::attitude::roll_pitch_deg(&r);
        est.roll[i] = roll;
        est.pitch[i] = pitch;
        if est.heading[i].is_finite() {
            let turned = heading_deg(&r);
            // Slip stays heading minus the (unchanged) travel direction. On a
            // pitched or leaned bike the heading moves by a hair more or less
            // than the trim, which is about the bike's own up, not the vertical.
            if est.slip[i].is_finite() {
                est.slip[i] = (est.slip[i] + turned - est.heading[i] + 180.0).rem_euclid(360.0) - 180.0;
            }
            est.heading[i] = turned;
        }
        // Longitudinal is +X, lateral is −Y (right-positive).
        let (long, lat) = (est.accel_long[i], est.accel_lat[i]);
        est.accel_long[i] = long * cos - lat * sin;
        est.accel_lat[i] = long * sin + lat * cos;
    }
    let x = &mut est.final_state;
    x.r_chassis *= rz;
    x.b_g0 = rz.inverse() * x.b_g0;
    x.b_a0 = rz.inverse() * x.b_a0;
}

/// Where on the bike travel direction is read for [`StateEstimate::slip`], as an
/// offset from IMU0 in the chassis frame, m: the swingarm pod, which sits by the
/// rear axle. A yawing frame sweeps its front sideways, so the angle between
/// where the frame points and where it is going depends on where along the
/// frame it is asked; at the rear wheel it is the rear tyre's slide. The pod's
/// height is dropped (a roll rate times the pod's height above the ground is
/// not the tyre moving sideways). With no swingarm pod it is IMU0 itself.
pub fn slip_reference(geometry: &BikeGeometry) -> Vector3<f64> {
    geometry.imu2.as_ref().map_or(Vector3::zeros(), |pose| Vector3::new(pose.lever.x, pose.lever.y, 0.0))
}

/// Travel direction at `lever` (chassis frame, m, from IMU0): degrees clockwise
/// from north in `[0, 360)`, and its 1-sigma uncertainty in rad from the
/// velocity covariance across the direction of travel. `rate` is the chassis
/// angular rate with the gyro offset removed, rad/s. None below `min_speed` m/s.
fn travel_direction(fs: &FilterState, rate: Vector3<f64>, lever: Vector3<f64>, min_speed: f64) -> Option<(f64, f64)> {
    let v = fs.x.v_chassis + fs.x.r_chassis * rate.cross(&lever);
    let speed = v.xy().norm();
    if !(speed >= min_speed) {
        return None;
    }
    // Unit vector across the direction of travel, level.
    let across = Vector3::new(-v.y, v.x, 0.0) / speed;
    let mut var = 0.0;
    for r in 0..3 {
        for c in 0..3 {
            var += across[r] * fs.p[(I_V + r, I_V + c)] * across[c];
        }
    }
    Some(((-v.y).atan2(v.x).to_degrees().rem_euclid(360.0), var.max(0.0).sqrt() / speed))
}

/// Largest squared Mahalanobis distance (3 degrees of freedom) a zero-rate
/// update's residual may have: the 99.9 % point of χ²₃.
const ZARU_GATE_CHI2: f64 = 16.27;

/// The zero-angular-rate update, applied only when the rate it would write into
/// the gyro bias is one the bias could plausibly have drifted to.
///
/// The stationary test passes anything turning slower than
/// `zupt_gyro_thresh` (2.9°/s), which a bike being walked round or swung on its
/// stand does. Taken as "rate is zero", that turn goes straight into the bias:
/// on ride `173461a2` at 2500 s a 1.2°/s turn while nearly stopped did exactly
/// that, and the heading then ran away at 0.75°/s for the next four minutes. A
/// gyro's offset does not move that far between stops, and the filter's own
/// bias covariance says how far it could have.
fn gated_zaru(filter: &Iekf, fs: &FilterState, zaru: &ZeroAngularRate, ctx: &SampleContext) -> FilterState {
    let col = zaru.target.col();
    let residual = zaru.measured - zaru.target.bias_of(&fs.x);
    let mut s = nalgebra::Matrix3::identity() * zaru.sigma.powi(2);
    for r in 0..3 {
        for c in 0..3 {
            s[(r, c)] += fs.p[(col + r, col + c)];
        }
    }
    match s.try_inverse() {
        Some(inv) if (residual.transpose() * inv * residual)[(0, 0)] <= ZARU_GATE_CHI2 => filter.update(fs, zaru, ctx),
        _ => fs.clone(),
    }
}

/// `tilt`'s up with `heading`'s forward: the mount whose chassis +Z is where
/// `tilt` puts it and whose chassis +X is `heading`'s, levelled. `tilt` itself
/// when `heading`'s forward lies along that up.
fn tilt_with_heading_of(tilt: &UnitQuaternion<f64>, heading: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    let up = tilt.inverse() * Vector3::z();
    let forward = heading.inverse() * Vector3::x();
    let level = forward - up * forward.dot(&up);
    if level.norm() < 1e-3 {
        return *tilt;
    }
    let x = level.normalize();
    let y = up.cross(&x);
    let rows = nalgebra::Matrix3::from_rows(&[x.transpose(), y.transpose(), up.transpose()]);
    UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rows))
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
    //
    // The mount is the geometry's (a calibrated bike), or fitted from this
    // session's riding (`config.imu0_mount`). The unsprung mounts are stated
    // against the geometry's IMU0 frame, so they rotate with any change to it
    // (`mount_shift`), which leaves the IMU0–IMU1–IMU2 relative geometry as it was.
    // The GPS latency is measured against the gyro's turn rate about the bike's
    // up, which needs the mount's up; the riding fit of the mount needs the
    // latency. The geometry's up starts it, and one more round with the
    // fitted mount's up settles it (the moto's IMU0 sits 45° from the
    // reference mount's).
    let measure_latency = |up_sensor: Vector3<f64>| {
        config.fit_gps_latency.then(|| fit_gps_latency(&input.gps, &input.imu0.gyro, up_sensor, input.t0_s, input.dt)).flatten()
    };
    let fit_mount = |latency_s: f64| {
        let (speed, path_pitch) = gps_speed_on_grid(&input.gps, input.t0_s, latency_s, input.dt, n);
        fit_imu0_mount_from_riding(&input.imu0.accel, &input.imu0.gyro, &speed, &path_pitch, input.dt, geometry.imu0.mount)
    };
    let first_latency = measure_latency(geometry.imu0.mount.inverse() * Vector3::z());
    let mut mount_fit = fit_mount(first_latency.unwrap_or(config.gps_latency_s));
    let mut latency = first_latency;
    if config.imu0_mount == Imu0Mount::Riding && mount_fit.used {
        latency = measure_latency(mount_fit.mount.inverse() * Vector3::z()).or(first_latency);
        if latency != first_latency {
            mount_fit = fit_mount(latency.unwrap_or(config.gps_latency_s));
        }
    }
    let (gps_latency_s, gps_latency_fitted) = (latency.unwrap_or(config.gps_latency_s), latency.is_some());
    let (speed, _) = gps_speed_on_grid(&input.gps, input.t0_s, gps_latency_s, input.dt, n);
    let m0 = match config.imu0_mount {
        Imu0Mount::Riding => mount_fit.mount,
        Imu0Mount::Geometry => geometry.imu0.mount,
        Imu0Mount::GeometryTilt if mount_fit.used => tilt_with_heading_of(&geometry.imu0.mount, &mount_fit.mount),
        Imu0Mount::GeometryTilt => geometry.imu0.mount,
    };
    let imu0_riding_check_deg = (mount_fit.used && config.imu0_mount != Imu0Mount::Riding)
        .then(|| m0.angle_to(&mount_fit.mount).to_degrees());
    let mount_fit = Some(mount_fit);
    let mount_shift = m0 * geometry.imu0.mount.inverse();
    let imu0_accel: Vec<Vector3<f64>> = input.imu0.accel.iter().map(|a| m0 * a).collect();
    let imu0_gyro: Vec<Vector3<f64>> = input.imu0.gyro.iter().map(|g| m0 * g).collect();

    // Stationary gate + orientation/bias pre-step (first stationary window, else the
    // opening samples).
    //
    // The IMU test alone (steady specific force, no rotation) is also what smooth
    // riding in a straight line at a steady speed looks like, and a zero-velocity
    // update there fights the GPS. A GPS speed at or above `gps_min_speed_mps`
    // therefore vetoes the flag; with no GPS speed the IMU test stands.
    let mut ctx = stationary_context(
        &input.imu0.accel,
        &input.imu0.gyro,
        config.zupt_window,
        config.zupt_accel_std,
        config.zupt_gyro_thresh,
    );
    for (c, v) in ctx.iter_mut().zip(&speed) {
        if *v >= config.gps_min_speed_mps {
            c.stationary = false;
        }
    }
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
        (Some(s), true) => refine_mount_from_window(mount_shift * geometry.imu1.mount, &s.accel, ws, we, up0_ref),
        _ => mount_shift * geometry.imu1.mount,
    };
    let imu2_mount = match (&input.imu2, &geometry.imu2) {
        (Some(s), Some(pose)) if config.refine_mounts => {
            refine_mount_from_window(mount_shift * pose.mount, &s.accel, ws, we, up0_ref)
        }
        (_, Some(pose)) => mount_shift * pose.mount,
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
    let mut gps_up_at: Vec<Option<f64>> = vec![None; n];
    for s in &input.gps {
        let t = s.time_s - input.t0_s - gps_latency_s;
        if t < 0.0 {
            continue;
        }
        let idx = (t / input.dt).round() as usize;
        if idx >= n {
            continue;
        }
        gps_up_at[idx] = s.up_mps;
        if s.velocity.xy().norm() >= config.gps_min_speed_mps {
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
    // Gravity-leveling std for one sample of specific-force magnitude `f` (m/s²).
    // An acceleration `a` tilts the specific force by up to `a/g`, and `|f/g − 1|`
    // is the part of it the magnitude shows, so it adds to the base std; the whole
    // is inflated for sample-to-sample correlation (`gravity_corr_s`).
    let looks_per_sample = (config.gravity_corr_s / dt).max(1.0).sqrt();
    let leveling_sigma =
        |f: f64| looks_per_sample * config.gravity_sigma.hypot(f / GRAVITY - 1.0);
    let mut fs = init_fs.clone();
    // Stream the wheel and attitude outputs per sample; the full per-sample state
    // is NOT retained (see [`StateEstimate`]), so memory stays O(outputs) on long
    // sessions.
    let mut front_travel = Vec::with_capacity(n);
    let mut front_velocity = Vec::with_capacity(n);
    let mut rear_travel = Vec::with_capacity(n);
    let mut rear_velocity = Vec::with_capacity(n);
    let mut roll = Vec::with_capacity(n);
    let mut pitch = Vec::with_capacity(n);
    let mut heading = Vec::with_capacity(n);
    let mut heading_sigma = Vec::with_capacity(n);
    let mut course = Vec::with_capacity(n);
    let mut slip = Vec::with_capacity(n);
    let mut slip_sigma = Vec::with_capacity(n);
    let slip_lever = slip_reference(geometry);
    // Heading is unknown until the first fix fast enough to give a course; until
    // then no GPS velocity is fused. `last_gps` is the sample of the last fused fix.
    let mut heading_set = false;
    let mut last_gps: Option<usize> = None;
    let gps_hold = (config.gps_hold_s / dt).round() as usize;
    let mut accel_long = Vec::with_capacity(n);
    let mut accel_lat = Vec::with_capacity(n);
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
            fs = gated_zaru(&filter, &fs, &ZeroAngularRate { target: GyroBias::Imu0, measured: gyro0, sigma: config.zaru_sigma }, &c);
            if let Some(s) = &input.imu1 {
                let zaru = ZeroAngularRate { target: GyroBias::Imu1, measured: imu1_mount * s.gyro[i], sigma: config.zaru_sigma };
                fs = gated_zaru(&filter, &fs, &zaru, &c);
            }
            if let (Some(s), Some(_pose)) = (&input.imu2, &geometry.imu2) {
                let zaru = ZeroAngularRate { target: GyroBias::Imu2, measured: imu2_mount * s.gyro[i], sigma: config.zaru_sigma };
                fs = gated_zaru(&filter, &fs, &zaru, &c);
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
        //
        // GATED OFF while GPS-aided and moving: leveling takes the specific force
        // for gravity, and in a corner it is not (a leaned bike in a coordinated
        // turn reads as upright). There the GPS velocity observes tilt through the
        // strapdown instead. Stationary, or with no recent fix, it is the tilt
        // reference as before.
        let gps_aided = last_gps.is_some_and(|j| i - j <= gps_hold);
        if !air && (c.stationary || !gps_aided) {
            fs = filter.update(
                &fs,
                &GravityLeveling { accel0, a_kin_nav: Vector3::zeros(), sigma: leveling_sigma(accel0.norm()) },
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
            // The heading is set from the GPS course once, at the first fix fast
            // enough to have one. After that it is the gyro's, corrected through
            // the GPS velocity like any other state: it is never pulled onto the
            // course, so a slide (heading and course apart) stays visible.
            if !heading_set && vel.xy().norm() >= config.heading_init_min_speed_mps {
                fs = align_heading(&fs, vel, config.heading_init_sigma);
                heading_set = true;
            }
            if heading_set {
                fs = filter.update(&fs, &GpsVelocity { measured: vel, sigma: config.gps_sigma }, &c);
                last_gps = Some(i);
            }
        }
        if let Some(up) = gps_up_at[i] {
            fs = filter.update(&fs, &VerticalVelocity { measured: up, sigma: config.gps_up_sigma }, &c);
        }

        front_travel.push(fs.x.d_f);
        front_velocity.push(fs.x.dd_f);
        rear_travel.push(fs.x.s_r);
        rear_velocity.push(fs.x.ds_r);

        // Attitude and gravity-removed body acceleration, read off the same
        // posterior the wheel outputs come from. `accel0` is the mount-corrected
        // chassis-frame specific force already bound above for `ImuInput`.
        let (roll_deg, pitch_deg) = crate::estimate::attitude::roll_pitch_deg(&fs.x.r_chassis);
        roll.push(roll_deg);
        pitch.push(pitch_deg);
        heading.push(if heading_set { heading_deg(&fs.x.r_chassis) } else { f64::NAN });
        heading_sigma.push(if heading_set { heading_sigma_rad(&fs).to_degrees() } else { f64::NAN });
        let travel = heading_set
            .then(|| travel_direction(&fs, gyro0 - fs.x.b_g0, slip_lever, config.heading_init_min_speed_mps))
            .flatten();
        match travel {
            Some((course_deg, course_sigma_rad)) => {
                let h = heading_deg(&fs.x.r_chassis);
                course.push(course_deg);
                slip.push((h - course_deg + 180.0).rem_euclid(360.0) - 180.0);
                slip_sigma.push(heading_sigma_rad(&fs).hypot(course_sigma_rad).to_degrees());
            }
            None => {
                course.push(f64::NAN);
                slip.push(f64::NAN);
                slip_sigma.push(f64::NAN);
            }
        }
        let (long_g, lat_g) = crate::estimate::attitude::body_accel_g(&accel0, &fs.x.r_chassis);
        accel_long.push(long_g);
        accel_lat.push(lat_g);
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
    let mut est = StateEstimate {
        schema,
        dt: input.dt,
        front_travel,
        front_velocity,
        rear_travel,
        rear_velocity,
        roll,
        pitch,
        heading,
        heading_sigma,
        course,
        slip,
        slip_sigma,
        imu0_mount: mount_fit,
        imu0_riding_check_deg,
        gps_latency_s,
        gps_latency_fitted,
        forward_trim_deg: None,
        forward_trim_applied: false,
        accel_long,
        accel_lat,
        stationary: stationary.clone(),
        final_state,
        ledger,
    };
    if let Some(trim) = forward_trim(&est.heading, &est.slip, &speed, dt) {
        est.forward_trim_deg = Some(trim.to_degrees());
        if config.trim_forward {
            apply_forward_trim(&mut est, trim);
            est.forward_trim_applied = true;
        }
    }
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

    /// Synthetic uniform `t_us` for a test double with no real recorded time
    /// (matches [`crate::session::Channel::from_f64`]'s formula).
    fn synthetic_t_us(len: usize, rate: f64) -> std::sync::Arc<[i64]> {
        if rate > 0.0 {
            (0..len).map(|i| (i as f64 * 1_000_000.0 / rate) as i64).collect()
        } else {
            vec![0i64; len].into()
        }
    }

    /// A minimal ChannelLookup test double over a name → (samples, rate) map, with
    /// optional per-channel event times (the GPS path).
    struct FakeLookup {
        channels: HashMap<String, (Vec<f64>, f64)>,
        times: HashMap<String, Vec<f64>>,
    }

    impl ChannelLookup for FakeLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            self.channels.get(name).map(|(s, r)| {
                // Synthetic uniform t_us consistent with the declared rate —
                // this test double models no real recorded time, so a raw
                // index relabeled as µs would be wrong; derive it the same
                // way Channel::from_f64 does.
                let t_us = synthetic_t_us(s.len(), *r);
                LookupChannel { samples: s.clone().into(), sample_rate_hz: *r, t_us }
            })
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
            roll: vec![0.0; 5],
            pitch: vec![0.0; 5],
            heading: vec![f64::NAN; 5],
            heading_sigma: vec![f64::NAN; 5],
            course: vec![f64::NAN; 5],
            slip: vec![f64::NAN; 5],
            slip_sigma: vec![f64::NAN; 5],
            imu0_mount: None,
            imu0_riding_check_deg: None,
            gps_latency_s: 0.0,
            gps_latency_fitted: false,
            forward_trim_deg: None,
            forward_trim_applied: false,
            accel_long: vec![0.0; 5],
            accel_lat: vec![0.0; 5],
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
            roll: vec![0.0, 0.0],
            pitch: vec![0.0, 0.0],
            heading: vec![f64::NAN; 2],
            heading_sigma: vec![f64::NAN; 2],
            course: vec![f64::NAN; 2],
            slip: vec![f64::NAN; 2],
            slip_sigma: vec![f64::NAN; 2],
            imu0_mount: None,
            imu0_riding_check_deg: None,
            gps_latency_s: 0.0,
            gps_latency_fitted: false,
            forward_trim_deg: None,
            forward_trim_applied: false,
            accel_long: vec![0.0, 0.0],
            accel_lat: vec![0.0, 0.0],
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
    fn resample_onto_a_sine_recorded_at_another_rate_reads_the_sine_at_the_grid_times() {
        // Arrange — 3 Hz sine recorded at 103 Hz; grid at 100 Hz inside its span.
        let f = |t_us: i64| (2.0 * std::f64::consts::PI * 3.0 * t_us as f64 / 1e6).sin();
        let t_us: Vec<i64> = (0..206).map(|i| (i as f64 * 1e6 / 103.0) as i64).collect();
        let v: Vec<f64> = t_us.iter().map(|&t| f(t)).collect();
        let grid: Vec<i64> = (5..190).map(|i| i * 10_000).collect();

        // Act
        let out = resample_onto(&t_us, &v, &grid);

        // Assert
        for (o, &t) in out.iter().zip(&grid) {
            assert!((o - f(t)).abs() < 1e-3, "at {t} µs: {o} vs {}", f(t));
        }
    }

    #[test]
    fn resample_onto_outside_the_recorded_span_holds_the_end_samples() {
        // Arrange
        let (t_us, v) = (vec![1_000, 2_000, 3_000], vec![1.0, 2.0, 3.0]);

        // Act
        let out = resample_onto(&t_us, &v, &[0, 1_500, 9_000]);

        // Assert — held, linear, held.
        assert_relative_eq!(out[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[1], 1.5, epsilon = 1e-12);
        assert_relative_eq!(out[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn resample_onto_across_a_recording_gap_is_linear() {
        // Arrange — even 1 ms samples, then a 10 ms gap; a cubic would overshoot.
        let t_us = vec![0, 1_000, 2_000, 12_000, 13_000, 14_000];
        let v = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];

        // Act
        let out = resample_onto(&t_us, &v, &[7_000]);

        // Assert
        assert_relative_eq!(out[0], 5.0, epsilon = 1e-12);
    }

    /// A lookup whose IMU0 and IMU1 `AccelZ` both record `sin(2π·t)` g, IMU0 at
    /// `rate0` Hz and IMU1 at `rate1` Hz, both for about 4 s, with `IMU0_GyroZ` at
    /// `gyro_rate` Hz.
    fn two_rate_lookup(rate0: f64, rate1: f64, gyro_rate: f64) -> FakeLookup {
        let sine = |rate: f64| -> Vec<f64> {
            (0..(4.0 * rate) as usize + 8)
                .map(|i| (2.0 * std::f64::consts::PI * synthetic_t_us(i + 1, rate)[i] as f64 / 1e6).sin())
                .collect()
        };
        let mut channels = HashMap::new();
        channels.insert("IMU0_AccelZ".to_string(), (sine(rate0), rate0));
        channels.insert("IMU1_AccelZ".to_string(), (sine(rate1), rate1));
        channels.insert("IMU0_GyroZ".to_string(), (vec![0.0; (4.0 * gyro_rate) as usize], gyro_rate));
        FakeLookup { channels, times: HashMap::new() }
    }

    #[test]
    fn from_lookup_imus_recorded_at_different_rates_the_same_index_is_the_same_instant() {
        // Arrange — the fork IMU runs 1.5 % fast, as the pods do on a real ride.
        let lookup = two_rate_lookup(400.0, 406.0, 400.0);

        // Act
        let input = EstimatorInput::from_lookup(&lookup).unwrap();

        // Assert — both IMUs saw the same sine, so on the common grid they agree
        // sample for sample (by index they would be 60 ms apart by the end).
        let imu1 = input.imu1.as_ref().unwrap();
        assert_eq!(imu1.accel.len(), input.imu0.accel.len());
        for i in 0..input.imu0.accel.len() {
            let (a0, a1) = (input.imu0.accel[i].z / GRAVITY, imu1.accel[i].z / GRAVITY);
            assert!((a0 - a1).abs() < 1e-3, "sample {i}: {a0} vs {a1}");
        }
    }

    #[test]
    fn from_lookup_accel_recorded_at_twice_the_gyro_rate_period_is_the_gyro_s_and_accel_stays_on_time() {
        // Arrange — an IDL1 merged accel channel: 800 Hz beside a 400 Hz gyro.
        let lookup = two_rate_lookup(800.0, 400.0, 400.0);

        // Act
        let input = EstimatorInput::from_lookup(&lookup).unwrap();

        // Assert — one sample per gyro sample, and accel sample i is the sine at
        // i·dt (it would be the sine at i·dt/2 if paired by index).
        assert_relative_eq!(input.dt, 1.0 / 400.0, epsilon = 1e-9);
        assert_eq!(input.imu0.accel.len(), input.imu0.gyro.len());
        for i in (40..input.imu0.accel.len() - 40).step_by(7) {
            let expect = (2.0 * std::f64::consts::PI * i as f64 * input.dt).sin();
            assert!((input.imu0.accel[i].z / GRAVITY - expect).abs() < 5e-3, "sample {i}");
        }
    }

    #[test]
    fn altitude_slopes_steady_climb_is_the_climb_rate() {
        // Arrange — 3 Hz fixes climbing at 0.5 m/s.
        let times: Vec<f64> = (0..30).map(|i| i as f64 / 3.0).collect();
        let altitude: Vec<f64> = times.iter().map(|t| 100.0 + 0.5 * t).collect();

        // Act
        let slopes = altitude_slopes(&times, &altitude);

        // Assert
        assert_eq!(slopes.len(), 30);
        for s in slopes {
            assert_relative_eq!(s.unwrap(), 0.5, epsilon = 1e-9);
        }
    }

    #[test]
    fn altitude_slopes_a_lone_fix_has_none() {
        // Arrange — one fix 30 s from its neighbours.
        let (times, altitude) = (vec![0.0, 30.0, 60.0], vec![100.0, 110.0, 120.0]);

        // Act
        let slopes = altitude_slopes(&times, &altitude);

        // Assert
        assert!(slopes.iter().all(|s| s.is_none()));
    }

    #[test]
    fn gps_speed_on_grid_is_linear_between_fixes_and_unknown_across_a_long_gap() {
        // Arrange — fixes at 1 s (2 m/s north) and 2 s (4 m/s north), then 10 s.
        let fix = |time_s: f64, v: f64, up: Option<f64>| GpsSample { time_s, velocity: Vector3::new(v, 0.0, 0.0), up_mps: up };
        let gps = vec![fix(1.0, 2.0, Some(2.0)), fix(2.0, 4.0, Some(4.0)), fix(10.0, 4.0, None)];

        // Act — 10 Hz grid from t = 0, no latency.
        let (speed, pitch) = gps_speed_on_grid(&gps, 0.0, 0.0, 0.1, 120);

        // Assert — before the first fix unknown; halfway 3 m/s on a 45° climb;
        // unknown inside the 8 s gap.
        assert!(speed[5].is_nan());
        assert_relative_eq!(speed[15], 3.0, epsilon = 1e-9);
        assert_relative_eq!(pitch[15], std::f64::consts::FRAC_PI_4, epsilon = 1e-9);
        assert!(speed[60].is_nan() && pitch[60].is_nan());
    }

    #[test]
    fn align_heading_points_the_chassis_along_the_gps_course_and_keeps_the_tilt() {
        // Arrange — leaned 20° and heading north; GPS says travelling west at 5 m/s.
        let lean = UnitQuaternion::from_euler_angles(0.35, 0.0, 0.0);
        let schema = StateSchema::from_geometry(&BikeGeometry::reference_bike(), false);
        let mut x = rest_state();
        x.r_chassis = lean;
        let fs = FilterState::initial(x, &schema, &InitStd::default());

        // Act
        let out = align_heading(&fs, Vector3::new(0.0, 5.0, 0.0), 0.1);

        // Assert — heading 270° (west), roll unchanged, heading variance raised.
        assert_relative_eq!(heading_deg(&out.x.r_chassis), 270.0, epsilon = 1e-6);
        let (roll, _) = crate::estimate::attitude::roll_pitch_deg(&out.x.r_chassis);
        assert_relative_eq!(roll, 0.35_f64.to_degrees(), epsilon = 1e-6);
        assert!(out.p[(2, 2)] > fs.p[(2, 2)]);
    }

    /// A ride with known truth, IMU0 only, 100 Hz, GPS at 5 Hz: parked 2 s, up to
    /// 8 m/s heading north, straight until 12 s, then a steady left turn at
    /// 0.5 rad/s (lean `atan(v·Ω/g)` = 22.2°) to the end. The bike leans into the
    /// turn so the specific force stays in its own vertical plane (a coordinated
    /// turn). Returns the input and the true (roll°, heading°) per sample.
    fn coordinated_turn_ride(seconds: f64) -> (EstimatorInput, Vec<(f64, f64)>) {
        let dt = 0.01;
        let n = (seconds / dt) as usize;
        let ramp = |t: f64, t0: f64, len: f64| -> f64 {
            let u = ((t - t0) / len).clamp(0.0, 1.0);
            u * u * (3.0 - 2.0 * u)
        };
        let speed = |t: f64| 8.0 * ramp(t, 2.0, 4.0);
        let yaw_rate = |t: f64| 0.5 * ramp(t, 12.0, 2.0);
        // Truth at every sample (one extra each side for the differences).
        let mut yaw = 0.0;
        let mut att = Vec::with_capacity(n + 2);
        let mut vel = Vec::with_capacity(n + 2);
        for i in 0..n + 2 {
            let t = i as f64 * dt;
            let lean = (speed(t) * yaw_rate(t) / GRAVITY).atan();
            // Nav X north, Y west: +yaw turns left; a left lean is −roll about X.
            att.push(UnitQuaternion::from_euler_angles(-lean, 0.0, yaw));
            vel.push(speed(t) * Vector3::new(yaw.cos(), yaw.sin(), 0.0));
            yaw += yaw_rate(t) * dt;
        }
        let m0 = BikeGeometry::reference_bike().imu0.mount;
        let (mut gyro, mut accel, mut truth, mut gps) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let rate = crate::rotation::log_so3(&(att[i].inverse() * att[i + 1])) / dt;
            let a_nav = if i == 0 { Vector3::zeros() } else { (vel[i + 1] - vel[i - 1]) / (2.0 * dt) };
            let f_chassis = att[i].inverse() * (a_nav + Vector3::new(0.0, 0.0, GRAVITY));
            gyro.push(m0.inverse() * rate);
            accel.push(m0.inverse() * f_chassis);
            let (roll, _) = crate::estimate::attitude::roll_pitch_deg(&att[i]);
            truth.push((roll, heading_deg(&att[i])));
            if i % 20 == 0 {
                gps.push(GpsSample { time_s: i as f64 * dt, velocity: vel[i], up_mps: Some(0.0) });
            }
        }
        let input = EstimatorInput { dt, t0_s: 0.0, imu0: ImuSeries { gyro, accel }, imu1: None, imu2: None, gps };
        (input, truth)
    }

    #[test]
    fn run_steady_coordinated_turn_with_gps_roll_is_the_lean_not_upright() {
        // Arrange — the specific force in a coordinated turn points along the
        // bike's own vertical, so gravity-leveling alone reads the bike as upright.
        let (input, truth) = coordinated_turn_ride(24.0);
        let config = EstimatorConfig { gps_latency_s: 0.0, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &config);

        // Assert — 8 s into the turn: 22.2° of left lean, within 2°.
        let i = 2200;
        assert_relative_eq!(truth[i].0, -22.2, epsilon = 0.1);
        assert!((est.roll[i] - truth[i].0).abs() < 2.0, "roll {} vs {}", est.roll[i], truth[i].0);
    }

    #[test]
    fn run_heading_is_unknown_until_the_bike_moves_then_follows_the_gps_course() {
        // Arrange
        let (input, truth) = coordinated_turn_ride(24.0);
        let config = EstimatorConfig { gps_latency_s: 0.0, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &config);

        // Assert — parked: no heading. Mid-turn: within 3° of the truth.
        assert!(est.heading[100].is_nan());
        let i = 2200;
        let off = (est.heading[i] - truth[i].1 + 180.0).rem_euclid(360.0) - 180.0;
        assert!(off.abs() < 3.0, "heading {} vs {}", est.heading[i], truth[i].1);
    }

    #[test]
    fn run_handling_the_bike_before_the_ride_does_not_push_tilt_into_the_accel_bias() {
        // Arrange — 2 s parked, then 8 s of the bike being shoved about by hand:
        // a 3 m/s² sideways specific force with no rotation and no GPS. Leveling
        // sees a tilted "gravity" for seconds on end.
        let n = 1000;
        let m0 = BikeGeometry::reference_bike().imu0.mount;
        let accel: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.01;
                let shove = if t < 2.0 { 0.0 } else { 3.0 * (2.0 * std::f64::consts::PI * 0.2 * t).sin() };
                m0.inverse() * Vector3::new(shove, 0.0, GRAVITY)
            })
            .collect();
        let input = EstimatorInput {
            dt: 0.01,
            t0_s: 0.0,
            imu0: ImuSeries { gyro: vec![Vector3::zeros(); n], accel },
            imu1: None,
            imu2: None,
            gps: vec![],
        };

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &EstimatorConfig::default());

        // Assert — the bias stays a plausible sensor offset (it reached 6 m/s² on a
        // real ride when every sample counted as an independent look at gravity).
        assert!(est.final_state.b_a0.norm() < 0.5, "accel bias {}", est.final_state.b_a0.norm());
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
        EstimatorInput { dt, t0_s: 0.0, imu0: series(), imu1: Some(series()), imu2: Some(series()), gps: vec![] }
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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };

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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
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

    /// A bike weaving at 8 m/s for `seconds`: IMU0's gyro (sensor Z is up) at
    /// 100 Hz, and GPS fixes every `fix_s` whose course is the bike's heading
    /// `late_s` before the fix's own time.
    fn weaving_ride(seconds: f64, fix_s: f64, late_s: f64) -> (Vec<GpsSample>, Vec<Vector3<f64>>) {
        let dt = 0.01;
        let n = (seconds / dt) as usize;
        // Yaw (anticlockwise) is a slow weave; a compass course is its negative.
        let yaw = |t: f64| 0.9 * (t * std::f64::consts::TAU / 9.0).sin() + 0.3 * (t * std::f64::consts::TAU / 3.7).sin();
        let gyro = (0..n).map(|i| Vector3::z() * (yaw((i + 1) as f64 * dt) - yaw(i as f64 * dt)) / dt).collect();
        let fixes = (seconds / fix_s) as usize;
        let gps = (1..fixes)
            .map(|k| {
                let t = k as f64 * fix_s;
                let y = yaw(t - late_s);
                GpsSample { time_s: t, velocity: 8.0 * Vector3::new(y.cos(), y.sin(), 0.0), up_mps: None }
            })
            .collect();
        (gps, gyro)
    }

    #[test]
    fn fit_gps_latency_course_350_ms_behind_the_gyro_is_measured_to_20_ms() {
        // Arrange — a 3 Hz receiver reporting 0.35 s late.
        let (gps, gyro) = weaving_ride(120.0, 1.0 / 3.0, 0.35);

        // Act
        let latency = fit_gps_latency(&gps, &gyro, Vector3::z(), 0.0, 0.01).expect("measured");

        // Assert
        assert!((latency - 0.35).abs() < 0.02, "latency {latency} s");
    }

    #[test]
    fn fit_gps_latency_one_hertz_receiver_600_ms_late_is_measured_to_50_ms() {
        // Arrange
        let (gps, gyro) = weaving_ride(400.0, 1.0, 0.6);

        // Act
        let latency = fit_gps_latency(&gps, &gyro, Vector3::z(), 0.0, 0.01).expect("measured");

        // Assert
        assert!((latency - 0.6).abs() < 0.05, "latency {latency} s");
    }

    #[test]
    fn fit_gps_latency_too_few_fixes_is_not_measured() {
        // Arrange — 20 s at 3 Hz is 60 pairs, under the 100 needed.
        let (gps, gyro) = weaving_ride(20.0, 1.0 / 3.0, 0.35);

        // Act
        let latency = fit_gps_latency(&gps, &gyro, Vector3::z(), 0.0, 0.01);

        // Assert
        assert!(latency.is_none());
    }

    #[test]
    fn fit_gps_latency_riding_dead_straight_is_not_measured() {
        // Arrange — nothing turns, so no shift lines up better than another.
        let n = 12_000;
        let gyro = vec![Vector3::zeros(); n];
        let gps: Vec<GpsSample> =
            (1..360).map(|k| GpsSample { time_s: k as f64 / 3.0, velocity: Vector3::new(8.0, 0.0, 0.0), up_mps: None }).collect();

        // Act
        let latency = fit_gps_latency(&gps, &gyro, Vector3::z(), 0.0, 0.01);

        // Assert
        assert!(latency.is_none());
    }

    #[test]
    fn forward_trim_is_the_median_slip_riding_straight_and_ignores_the_corners() {
        // Arrange — 80 s at 100 Hz, 8 m/s. First half straight with the frame
        // reading 5° right of travel; second half turning at 20°/s with 30° of slip.
        let (dt, n) = (0.01, 8000);
        let heading: Vec<f64> = (0..n).map(|i| if i < 4000 { 90.0 } else { (90.0 + (i - 4000) as f64 * 0.2).rem_euclid(360.0) }).collect();
        let slip: Vec<f64> = (0..n).map(|i| if i < 4000 { 5.0 + 0.5 * ((i % 7) as f64 - 3.0) } else { 30.0 }).collect();
        let speed = vec![8.0; n];

        // Act
        let trim = forward_trim(&heading, &slip, &speed, dt).expect("enough straight riding");

        // Assert
        assert!((trim.to_degrees() - 5.0).abs() < 0.6, "trim {} deg", trim.to_degrees());
    }

    #[test]
    fn forward_trim_under_30_s_of_straight_riding_is_absent() {
        // Arrange — 20 s straight.
        let n = 2000;
        let (heading, slip, speed) = (vec![90.0; n], vec![5.0; n], vec![8.0; n]);

        // Act
        let trim = forward_trim(&heading, &slip, &speed, 0.01);

        // Assert
        assert!(trim.is_none());
    }

    #[test]
    fn forward_trim_slow_riding_is_not_counted() {
        // Arrange — a minute straight at walking pace.
        let n = 6000;
        let (heading, slip, speed) = (vec![90.0; n], vec![5.0; n], vec![1.5; n]);

        // Act
        let trim = forward_trim(&heading, &slip, &speed, 0.01);

        // Assert
        assert!(trim.is_none());
    }

    #[test]
    fn apply_forward_trim_turns_heading_and_slip_and_re_reads_tilt_and_acceleration_about_the_new_axes() {
        // Arrange — climbing at 10° heading north, the frame reading 5° right of
        // travel, accelerating forward at 0.2 g. The true forward is 5° to the left.
        let mut est = run(&static_input(50, 0.01), &BikeGeometry::reference_bike(), &EstimatorConfig::default());
        let n = est.len();
        est.roll = vec![0.0; n];
        est.pitch = vec![10.0; n];
        est.heading = vec![0.0; n];
        est.slip = vec![5.0; n];
        est.accel_long = vec![0.2; n];
        est.accel_lat = vec![0.0; n];

        // Act
        apply_forward_trim(&mut est, 5_f64.to_radians());

        // Assert — heading and slip drop by the trim; the 10° of climb is now seen
        // from axes turned 5°, so a little of it reads as roll; the forward
        // acceleration is mostly still forward, with a part to the right.
        assert!((est.heading[10] - 355.0).abs() < 0.1, "heading {}", est.heading[10]);
        assert!(est.slip[10].abs() < 0.1, "slip {}", est.slip[10]);
        assert_relative_eq!(est.slip[10], (est.heading[10] - 355.0 + 180.0).rem_euclid(360.0) - 180.0, epsilon = 1e-9);
        assert!((est.pitch[10] - 9.96).abs() < 0.02, "pitch {}", est.pitch[10]);
        assert!((est.roll[10].abs() - 0.87).abs() < 0.03, "roll {}", est.roll[10]);
        assert_relative_eq!(est.accel_long[10], 0.2 * 5_f64.to_radians().cos(), epsilon = 1e-12);
        assert_relative_eq!(est.accel_lat[10], 0.2 * 5_f64.to_radians().sin(), epsilon = 1e-12);
        assert_eq!(est.forward_trim_deg, None, "the caller records the trim; this only restates the outputs");
    }

    #[test]
    fn travel_direction_at_the_rear_wheel_takes_out_the_sweep_of_a_yawing_frame() {
        // Arrange — heading north at 6 m/s while yawing left at 1 rad/s about a
        // point 0.5 m behind IMU0: IMU0 itself is swept left at 0.5 m/s.
        let (_, mut fs) = state_with_bias_sigma(1e-3);
        fs.x.v_chassis = Vector3::new(6.0, 0.5, 0.0);
        let rate = Vector3::new(0.0, 0.0, 1.0);

        // Act
        let (at_imu0, _) = travel_direction(&fs, rate, Vector3::zeros(), 3.0).unwrap();
        let (at_rear, sigma) = travel_direction(&fs, rate, Vector3::new(-0.5, 0.0, 0.0), 3.0).unwrap();

        // Assert — at IMU0 the travel is 4.8° left of north; at the rear wheel it is north.
        assert!((at_imu0 - (360.0 - 4.76)).abs() < 0.05, "{at_imu0}");
        assert!(at_rear < 1e-6 || at_rear > 360.0 - 1e-6, "{at_rear}");
        assert!(sigma > 0.0 && sigma.is_finite());
    }

    #[test]
    fn travel_direction_below_the_minimum_speed_is_absent() {
        // Arrange
        let (_, mut fs) = state_with_bias_sigma(1e-3);
        fs.x.v_chassis = Vector3::new(1.0, 0.0, 0.0);

        // Act
        let direction = travel_direction(&fs, Vector3::zeros(), Vector3::zeros(), 3.0);

        // Assert
        assert!(direction.is_none());
    }

    #[test]
    fn tilt_with_heading_of_keeps_one_mounts_up_and_takes_the_others_forward() {
        // Arrange — two mounts 20° apart in tilt and 30° apart in heading.
        let tilt = UnitQuaternion::from_euler_angles(0.0, 0.35, 0.0);
        let heading = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.52);

        // Act
        let m = tilt_with_heading_of(&tilt, &heading);

        // Assert — up is `tilt`'s exactly; forward is `heading`'s, levelled to it.
        let up = tilt.inverse() * Vector3::z();
        assert_relative_eq!(m * up, Vector3::z(), epsilon = 1e-12);
        let forward = m * (heading.inverse() * Vector3::x());
        assert_relative_eq!(forward.y, 0.0, epsilon = 1e-12);
        assert!(forward.x > 0.9);
    }

    /// A filter state with every gyro-bias block known to `sigma` rad/s.
    fn state_with_bias_sigma(sigma: f64) -> (Iekf, FilterState) {
        let schema = StateSchema::from_geometry(&BikeGeometry::reference_bike(), false);
        let init = InitStd { gyro_bias: sigma, ..InitStd::default() };
        let x = MtbState {
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
        };
        let fs = FilterState::initial(x, &schema, &init);
        let filter = Iekf {
            process: MtbProcess { schema, noise: ProcessNoiseConfig::reference_default(), gravity: GRAVITY },
            max_iters: 1,
            tol: 1e-9,
        };
        (filter, fs)
    }

    const STILL: SampleContext = SampleContext { stationary: true };

    #[test]
    fn gated_zaru_rate_within_the_bias_uncertainty_moves_the_bias() {
        // Arrange — bias known to 1 mrad/s; the gyro reads 1.5 mrad/s at rest.
        let (filter, fs) = state_with_bias_sigma(1e-3);
        let zaru = ZeroAngularRate { target: GyroBias::Imu0, measured: Vector3::new(0.0, 0.0, 1.5e-3), sigma: 1e-3 };

        // Act
        let out = gated_zaru(&filter, &fs, &zaru, &STILL);

        // Assert
        assert!(out.x.b_g0.z > 5e-4, "bias did not move: {}", out.x.b_g0.z);
    }

    #[test]
    fn gated_zaru_rate_far_outside_the_bias_uncertainty_is_a_slow_turn_and_is_ignored() {
        // Arrange — bias known to 1 mrad/s; 20 mrad/s is the bike being turned.
        let (filter, fs) = state_with_bias_sigma(1e-3);
        let zaru = ZeroAngularRate { target: GyroBias::Imu0, measured: Vector3::new(0.0, 0.0, 0.02), sigma: 1e-3 };

        // Act
        let out = gated_zaru(&filter, &fs, &zaru, &STILL);

        // Assert
        assert_eq!(out.x.b_g0, Vector3::zeros());
    }

    #[test]
    fn gated_zaru_before_the_bias_is_known_accepts_a_large_offset() {
        // Arrange — the first rest of a run: nothing is known about the offset yet.
        let (filter, fs) = state_with_bias_sigma(0.05);
        let zaru = ZeroAngularRate { target: GyroBias::Imu0, measured: Vector3::new(0.0, 0.0, 0.02), sigma: 1e-3 };

        // Act
        let out = gated_zaru(&filter, &fs, &zaru, &STILL);

        // Assert
        assert!((out.x.b_g0.z - 0.02).abs() < 1e-3, "{}", out.x.b_g0.z);
    }

    #[test]
    fn run_bike_swung_slowly_on_its_stand_after_a_rest_turns_the_heading_not_the_gyro_bias() {
        // Arrange — 20 s still, then 20 s turned about the vertical at 0.02 rad/s
        // (1.1°/s): slow and smooth enough that the stationary test still passes.
        let (dt, rest, turn) = (0.01, 2000usize, 2000usize);
        let up = Vector3::new(0.0, 0.0, GRAVITY);
        let rate = Vector3::new(0.0, 0.0, 0.02);
        let series = || ImuSeries {
            gyro: (0..rest + turn).map(|i| if i < rest { Vector3::zeros() } else { rate }).collect(),
            accel: vec![up; rest + turn],
        };
        let input = EstimatorInput { dt, t0_s: 0.0, imu0: series(), imu1: Some(series()), imu2: Some(series()), gps: vec![] };
        let mut geometry = BikeGeometry::reference_bike();
        geometry.imu0.mount = UnitQuaternion::identity();
        geometry.imu1.mount = UnitQuaternion::identity();
        geometry.imu2.as_mut().unwrap().mount = UnitQuaternion::identity();
        let config = EstimatorConfig { imu0_mount: Imu0Mount::Geometry, ..EstimatorConfig::default() };

        // Act
        let est = run(&input, &geometry, &config);

        // Assert — the 0.4 rad of turn is in the attitude, and the bias is not 0.02.
        let last = &est.final_state;
        let (_, _, yaw) = last.r_chassis.euler_angles();
        assert!((yaw - 0.4).abs() < 0.03, "yaw {yaw} rad, 0.4 turned");
        assert!(last.b_g0.z.abs() < 2e-3, "the turn went into the bias: {}", last.b_g0.z);
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
        let input = EstimatorInput { dt: 0.01, t0_s: 0.0, imu0, imu1: Some(still()), imu2: Some(still()), gps: vec![] };

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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };

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
        // 7.2 km/h due north (course 0°) at t = 2.5 s. Heading is already
        // physical degrees (ruling R27); both have fix quality 1.
        let mut channels = HashMap::new();
        channels.insert("GPS_SpeedKmh".to_string(), (vec![3.6, 7.2], 0.0));
        channels.insert("GPS_Heading".to_string(), (vec![90.0, 0.0], 0.0));
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
        channels.insert("GPS_Heading".to_string(), (vec![123.45, 0.0], 0.0));
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
            .map(|k| GpsSample { time_s: 0.5 * k as f64, velocity: Vector3::new(4.0, 0.0, 0.0), up_mps: None })
            .collect();
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: None, imu2: None, gps };
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
        let input = EstimatorInput { dt, t0_s: 0.0, imu0, imu1: Some(imu1), imu2: Some(imu2), gps: vec![] };
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

    /// A level, stationary session: `n` samples of pure +1 g on Z and no
    /// rotation. Physically the bike is parked on flat ground.
    fn level_stationary_input(n: usize) -> EstimatorInput {
        EstimatorInput {
            dt: 1.0 / 800.0,
            t0_s: 0.0,
            imu0: ImuSeries {
                gyro: vec![Vector3::zeros(); n],
                accel: vec![Vector3::new(0.0, 0.0, crate::estimate::attitude::G_MPS2); n],
            },
            imu1: None,
            imu2: None,
            gps: Vec::new(),
        }
    }

    #[test]
    fn run_emits_attitude_and_body_accel_for_every_sample() {
        // Arrange
        let input = level_stationary_input(1600);

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &EstimatorConfig::default());

        // Assert — one value per sample, matching the wheel outputs.
        assert_eq!(est.roll.len(), est.front_travel.len());
        assert_eq!(est.pitch.len(), est.front_travel.len());
        assert_eq!(est.accel_long.len(), est.front_travel.len());
        assert_eq!(est.accel_lat.len(), est.front_travel.len());
    }

    #[test]
    fn run_on_a_level_parked_bike_reports_no_lean_and_no_acceleration() {
        // Arrange
        let input = level_stationary_input(1600);

        // Act
        let est = run(&input, &BikeGeometry::reference_bike(), &EstimatorConfig::default());

        // Assert — sampled at the end, after the filter has settled. Tolerances
        // are loose enough to survive init transients but tight enough that a
        // sign error or a missing gravity subtraction fails.
        let last = est.roll.len() - 1;
        assert!(est.roll[last].abs() < 1.0, "roll {}", est.roll[last]);
        assert!(est.pitch[last].abs() < 1.0, "pitch {}", est.pitch[last]);
        assert!(est.accel_long[last].abs() < 0.05, "long {}", est.accel_long[last]);
        assert!(est.accel_lat[last].abs() < 0.05, "lat {}", est.accel_lat[last]);
    }
}
