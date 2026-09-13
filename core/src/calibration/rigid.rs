//! Rigid-body IMU calibration — two bodies joined by a one-DoF steering hinge.
//!
//! Implements `docs/superpowers/specs/2026-09-10-idl1-rigid-body-calibration.md`.
//! Section references below are to that spec. Pure maths: no I/O, no clock, no
//! network, `nalgebra` only.
//!
//! # What it recovers, and what it cannot
//!
//! From one session of the §5.1 protocol — a stationary hold, a tumble with
//! the bars straight, then a bar turn — the fit returns, per sensor, the
//! rotation from its own frame into its body's frame, its lever arm from the
//! body origin, and its gyro bias; plus the steering axis shared by the two
//! bodies. It does **not** return an individual accelerometer bias, because
//! there is none to return: §8.3 shows the data constrains only the difference
//! `R_i b_a,i − R_j b_a,j` between two sensors on one body, which is what
//! [`CalibrationRecord::accel_bias_differences`] carries. Nor does it return
//! either body frame in absolute terms: both are gauges (§1.2a), reported with
//! [`MountOrigin::Gauge`] so a consumer can tell a fitted rotation from a
//! declared one.
//!
//! # Shape of the solve
//!
//! Closed form first (§2.1–§2.4): gyro bias from the rest hold, rotations by
//! Kabsch over the tumble, the steering axis from the δ-free quadratic form
//! (7), lever arms and the bias difference by linear least squares on (6).
//! Then Levenberg–Marquardt over the joint parameter vector (§2.6), which
//! removes the cross-coupling the staged solve leaves behind. Every excitation
//! gate of §3 is computed and reported whether it passes or not; a gate that
//! fails removes fields from the record rather than failing the call.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};

use crate::estimate::noise::ImuNoise;

/// Version of the model this record was produced by (§6). Bump it whenever a
/// change to the maths would give a different answer for the same input; a
/// consumer compares it to decide whether a stored record must be refitted.
pub const MODEL_VERSION: u32 = 1;

/// Standard gravity, m/s².
pub const GRAVITY_M_S2: f64 = 9.80665;

// ── §3 excitation gates ─────────────────────────────────────────────────

/// Largest tolerated `cond(Σ_t ω_R ω_Rᵀ)` over the tumble (§3).
pub const MAX_ROTATION_CONDITION: f64 = 20.0;
/// Smallest tolerated median `‖ω_R‖` over the tumble, rad/s (§3).
pub const MIN_MEDIAN_RATE_RAD_S: f64 = 2.0;
/// Smallest tolerated `min eig(Σ_t Ω(t)ᵀΩ(t)) / N`, s⁻⁴ (§3).
pub const MIN_LEVER_RICHNESS: f64 = 1.0;
/// Smallest tolerated peak-to-peak steer angle, rad (§3).
pub const MIN_STEER_PEAK_TO_PEAK_RAD: f64 = 0.5;
/// Shortest tolerated stationary hold, seconds (§3).
pub const MIN_REST_S: f64 = 2.0;
/// Shortest tolerated tumble, seconds (§3).
pub const MIN_DATUM_S: f64 = 30.0;

// ── segmentation thresholds (§2.0) ──────────────────────────────────────

/// Smoothed `‖ω‖` below which the machine counts as stationary, rad/s. Well
/// above the rest-hold noise floor (3 axes at σ ≈ 0.087 rad/s give ‖ω‖ ≈ 0.15)
/// and far below the ≥ 2 rad/s the tumble is required to reach.
const REST_RATE_THRESHOLD_RAD_S: f64 = 0.6;
/// Magnitude of the smoothed `‖ω_F‖² − ‖ω_R‖²` above which the bars count as
/// turning, s⁻². Through the datum the signed quantity is zero-mean, so
/// smoothing drives it to the noise floor; through the turn its `δ̇²` term
/// averages to about 2 s⁻², four times this gate.
/// The quantity is `2δ̇ sᵀω_R + δ̇²`, whose second term never cancels, so unlike
/// a plain norm difference it cannot vanish at an unlucky bar angle.
const STEER_DISCRIMINANT_THRESHOLD: f64 = 0.5;
/// Half-width of the zero-phase moving average used for segmentation, seconds.
const SEGMENT_SMOOTH_S: f64 = 0.5;
/// Half-width of the zero-phase moving average applied to `ω` before it is
/// differentiated for `ω̇`, seconds (§2.3: raw differentiation is noise-
/// dominated). Short enough to pass the ~1 Hz content of a hand tumble.
const RATE_SMOOTH_S: f64 = 0.02;

// ── §2.6 Levenberg–Marquardt ────────────────────────────────────────────

/// Most LM iterations before the refinement gives up and keeps what it has.
const LM_MAX_ITERATIONS: u32 = 12;
/// Relative cost improvement below which LM stops.
const LM_COST_TOLERANCE: f64 = 1e-10;
/// Initial LM damping, relative to the mean diagonal of `JᵀJ`.
const LM_INITIAL_LAMBDA: f64 = 1e-3;
/// Step used for the central-difference Jacobian, in the natural units of each
/// parameter block (rad, rad/s, m, m/s²).
const LM_JACOBIAN_STEP: f64 = 1e-5;
/// Most samples per segment the LM stage evaluates. The closed-form stage has
/// already used every sample and sits at the noise floor; LM is there to
/// remove cross-coupling, which a strided subset shows just as well, and the
/// step is accepted only if it lowers the cost measured on **all** samples.
const LM_MAX_SAMPLES_PER_SEGMENT: usize = 6000;

/// Which of the two bodies a sensor is bolted to (§1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Body {
    /// Frame plus swingarm — `IMU0` and `IMU2`.
    Rear,
    /// Fork lowers, steerer and bar — `IMU1`.
    Front,
}

impl Body {
    /// The body `IDL0_SPEC.md` §3.2 puts this IMU index on: `IMU1` is the
    /// front-unsprung (fork) sensor, everything else rides the frame.
    pub fn of_imu(imu_index: u8) -> Body {
        if imu_index == 1 {
            Body::Front
        } else {
            Body::Rear
        }
    }
}

/// Whether a reported rotation was fitted from data or declared by convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountOrigin {
    /// Fitted (§2.2).
    Measured,
    /// A gauge of §1.2a — `R₀ ≡ I` for the reference sensor, `C₀ ≡ I` for the
    /// steer datum. Nothing in the protocol observes it.
    Gauge,
}

/// How much of the model the session supported (§6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationSource {
    /// The full §1 model: at least one excitation gate beyond the rest hold
    /// passed, so something more than gyro bias was fitted.
    RigidBody,
    /// §4: the stationary hold was all the session contained.
    RestOnly,
}

/// One sensor's samples, in **its own frame** and SI units, on the session's
/// sample grid. Wire units (`g`, `dps`) are converted by the caller.
#[derive(Debug, Clone)]
pub struct SensorStream {
    /// `imu_index` as it appears in the file's records (`IDL0_SPEC.md` §3.2).
    pub imu_index: u8,
    /// Which body this sensor is bolted to.
    pub body: Body,
    /// Angular rate in the sensor frame, rad/s.
    pub gyro_rad_s: Vec<Vector3<f64>>,
    /// Specific force in the sensor frame, m/s².
    pub accel_m_s2: Vec<Vector3<f64>>,
}

/// Everything [`calibrate`] reads.
#[derive(Debug, Clone)]
pub struct CalibrationInput {
    /// Sample rate of every stream, Hz. One grid, shared.
    pub sample_rate_hz: f64,
    /// Allan-variance noise parameters, which set the per-sample σ the
    /// residuals are weighted by (§1.3).
    pub noise: ImuNoise,
    /// One entry per instrumented sensor. Order is irrelevant; the reference
    /// is chosen as the lowest `imu_index` on [`Body::Rear`].
    pub sensors: Vec<SensorStream>,
    /// When the session was captured, RFC 3339, copied straight into the
    /// record. Supplied by the caller because `calibrate` reads no clock.
    pub captured_utc: Option<String>,
}

/// One named way the motion fell short of §3. Each carries what was seen and
/// what was needed, so the message can say both — never a bare failure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExcitationShortfall {
    /// `cond(Σ ω ωᵀ)` too large: the tumble turned about too few axes.
    RotationRichness { seen: f64, needed: f64 },
    /// Median `‖ω_R‖` too small: the tumble was too slow.
    RateMagnitude { seen: f64, needed: f64 },
    /// `min eig(Σ ΩᵀΩ)/N` too small: lever arms are not separable.
    LeverArmRichness { seen: f64, needed: f64 },
    /// Peak-to-peak `δ` too small: the bars did not turn far enough.
    SteerRichness { seen: f64, needed: f64 },
    /// The stationary hold was too short, or absent.
    RestHold { seen_s: f64, needed_s: f64 },
    /// The tumble was too short.
    Duration { seen_s: f64, needed_s: f64 },
    /// No sensor was found on the front body, so there is no hinge to fit.
    NoFrontSensor,
    /// Only one sensor was found on the rear body, so no lever arm is
    /// differenceable (§2.3 needs a pair).
    NoRearPair,
}

impl std::fmt::Display for ExcitationShortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExcitationShortfall::RotationRichness { seen, needed } => write!(
                f,
                "the machine was turned about too few axes (rate condition number {seen:.1}, {needed:.1} or less needed); repeat the figure-8s, tumbling through roll, pitch and yaw rather than spinning about one direction"
            ),
            ExcitationShortfall::RateMagnitude { seen, needed } => write!(
                f,
                "the tumble was too slow (median {seen:.2} rad/s, {needed:.2} rad/s needed)"
            ),
            ExcitationShortfall::LeverArmRichness { seen, needed } => write!(
                f,
                "the lever arms are uncertain: the motion was too nearly a single-axis spin ({seen:.2} s⁻⁴, {needed:.2} needed)"
            ),
            ExcitationShortfall::SteerRichness { seen, needed } => write!(
                f,
                "the bars did not turn far enough to find the steering axis ({:.0}° seen, {:.0}° needed), so it is not kept",
                seen.to_degrees(),
                needed.to_degrees()
            ),
            ExcitationShortfall::RestHold { seen_s, needed_s } => write!(
                f,
                "the stationary hold was too short ({seen_s:.1} s, {needed_s:.1} s needed), so gyro bias is not kept"
            ),
            ExcitationShortfall::Duration { seen_s, needed_s } => write!(
                f,
                "the tumble was too short ({seen_s:.1} s, {needed_s:.1} s needed)"
            ),
            ExcitationShortfall::NoFrontSensor => {
                write!(f, "no sensor on the front body, so there is no steering axis to fit")
            }
            ExcitationShortfall::NoRearPair => write!(
                f,
                "only one sensor on the rear body, so no lever arm can be differenced"
            ),
        }
    }
}

/// Why a calibration could not be attempted at all. A motion that is merely
/// poor is **not** an error — it is a thinner record plus
/// [`Quality::shortfalls`] (§3).
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    /// [`CalibrationInput::sensors`] was empty.
    NoSensors,
    /// No sensor was on [`Body::Rear`], so there is no reference frame.
    NoReferenceSensor,
    /// Streams disagreed on length, or a sensor's gyro and accel did.
    RaggedStreams { imu_index: u8, gyro: usize, accel: usize },
    /// Fewer samples than the differentiation and the gates can use.
    TooFewSamples { seen: usize, needed: usize },
    /// `sample_rate_hz` was not a finite positive number.
    BadSampleRate(f64),
    /// A sample was NaN or infinite. Carries where, so the caller can say so.
    NonFiniteSample { imu_index: u8, sample: usize },
    /// Every §3 gate failed, so nothing beyond gyro bias could be fitted and
    /// even that had no rest hold to come from.
    MotionNotRich(Vec<ExcitationShortfall>),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::NoSensors => write!(f, "no sensor streams were supplied"),
            CalibrationError::NoReferenceSensor => {
                write!(f, "no sensor on the rear body, so there is no reference frame")
            }
            CalibrationError::RaggedStreams { imu_index, gyro, accel } => write!(
                f,
                "IMU{imu_index} has {gyro} gyro samples and {accel} accel samples; every stream must share one grid"
            ),
            CalibrationError::TooFewSamples { seen, needed } => {
                write!(f, "{seen} samples is too few; at least {needed} are needed")
            }
            CalibrationError::BadSampleRate(v) => {
                write!(f, "sample rate must be finite and positive, got {v}")
            }
            CalibrationError::NonFiniteSample { imu_index, sample } => {
                write!(f, "IMU{imu_index} sample {sample} is not finite")
            }
            CalibrationError::MotionNotRich(shortfalls) => {
                write!(f, "the motion was not rich enough to calibrate anything:")?;
                for s in shortfalls {
                    write!(f, " {s};")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

/// One sensor's fitted extrinsics (§6). Every optional field is **absent**
/// when its gate failed, never zero (R190).
#[derive(Debug, Clone, PartialEq)]
pub struct SensorCalibration {
    /// `imu_index` as it appears in the file's records.
    pub imu_index: u8,
    /// Which body it rides.
    pub body: Body,
    /// Rotation sensor frame → body frame, dimensionless.
    pub mount: UnitQuaternion<f64>,
    /// Whether [`Self::mount`] was fitted or is the §1.2a gauge.
    pub mount_origin: MountOrigin,
    /// Lever arm from the body origin to the sensor, body frame, metres.
    /// Absent for a body-origin sensor (`r ≡ 0` by §1.2) and whenever the
    /// lever-arm gate failed.
    pub lever_m: Option<Vector3<f64>>,
    /// Gyro bias in the sensor frame, rad/s. Absent with no rest hold.
    pub gyro_bias_rad_s: Option<Vector3<f64>>,
}

/// `β_ij = R_i b_a,i − R_j b_a,j` for two sensors on one body, in that body's
/// frame, m/s² (§2.3). The individual biases are not observable (§8.3).
#[derive(Debug, Clone, PartialEq)]
pub struct AccelBiasDifference {
    /// `i` — the sensor whose bias is rotated in with a `+`.
    pub from_imu: u8,
    /// `j` — the sensor whose bias is rotated in with a `−`.
    pub to_imu: u8,
    /// The difference, body frame, m/s².
    pub value_m_s2: Vector3<f64>,
}

/// The §3 metrics as measured, plus whatever they ruled out.
#[derive(Debug, Clone, PartialEq)]
pub struct Quality {
    /// `cond(Σ_t ω_R ω_Rᵀ)` over the tumble.
    pub rotation_condition: Option<f64>,
    /// Median `‖ω_R‖` over the tumble, rad/s.
    pub median_rate_rad_s: Option<f64>,
    /// `min eig(Σ_t Ω(t)ᵀΩ(t)) / N`, s⁻⁴.
    pub lever_richness: Option<f64>,
    /// Peak-to-peak `δ` over the bar turn, rad.
    pub steer_peak_to_peak_rad: Option<f64>,
    /// Length of the stationary hold, seconds.
    pub rest_duration_s: f64,
    /// Length of the tumble, seconds.
    pub datum_duration_s: f64,
    /// RMS of the rear body's gyro-consistency residual after the fit, rad/s —
    /// §7.6's runtime check that `IMU2` really is rigid with the frame.
    pub rear_body_residual_rad_s: Option<f64>,
    /// LM iterations actually taken (§2.6).
    pub lm_iterations: u32,
    /// Every gate that failed, in §3 order.
    pub shortfalls: Vec<ExcitationShortfall>,
}

/// The calibration record of §6 — the *measured* replacement for the authored
/// constants in `estimate::geometry`.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationRecord {
    /// [`MODEL_VERSION`] at the time of the fit.
    pub model_version: u32,
    /// When the session was captured, RFC 3339. Absent when the caller did not
    /// supply one — [`calibrate`] reads no clock.
    pub captured_utc: Option<String>,
    /// How much of the model the session supported.
    pub source: CalibrationSource,
    /// Per sensor, in ascending `imu_index`.
    pub sensors: Vec<SensorCalibration>,
    /// `β_ij` per rear-body pair (§2.3).
    pub accel_bias_differences: Vec<AccelBiasDifference>,
    /// Steering-axis unit vector in the rear body frame, pointing up. Absent
    /// when the steer gate failed.
    pub steer_axis: Option<Vector3<f64>>,
    /// `C₀`, the front→rear rotation at `δ = 0`. Always the §1.2a identity
    /// gauge; present so a future multi-sensor front body can carry a fitted
    /// value without changing the record's shape.
    pub steer_datum: UnitQuaternion<f64>,
    /// The §3 metrics.
    pub quality: Quality,
}

// ── small linear-algebra helpers ────────────────────────────────────────

/// `[v]ₓ`, the skew-symmetric matrix with `[v]ₓ w = v × w`.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// A centred moving average of half-width `half`, clamped at the ends. Zero
/// phase by construction — the window is symmetric about each sample — so no
/// filter delay leaks into `ω̇` or into a segment boundary.
fn smooth(series: &[f64], half: usize) -> Vec<f64> {
    let n = series.len();
    if n == 0 || half == 0 {
        return series.to_vec();
    }
    // Prefix sums, so the cost is one pass regardless of the window width.
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for &v in series {
        prefix.push(prefix[prefix.len() - 1] + v);
    }
    (0..n)
        .map(|k| {
            let lo = k.saturating_sub(half);
            let hi = (k + half + 1).min(n);
            (prefix[hi] - prefix[lo]) / (hi - lo) as f64
        })
        .collect()
}

/// The same centred moving average, applied to each component of a 3-vector
/// series.
fn smooth_vectors(series: &[Vector3<f64>], half: usize) -> Vec<Vector3<f64>> {
    let x = smooth(&series.iter().map(|v| v.x).collect::<Vec<_>>(), half);
    let y = smooth(&series.iter().map(|v| v.y).collect::<Vec<_>>(), half);
    let z = smooth(&series.iter().map(|v| v.z).collect::<Vec<_>>(), half);
    (0..series.len()).map(|k| Vector3::new(x[k], y[k], z[k])).collect()
}

/// Median of a slice, by sorting a copy. `NaN` is impossible here: the input
/// is validated finite before any of this runs.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    }
}

/// Solves Wahba's problem (§2.2): the rotation `Q` minimising
/// `Σ ‖ reference − Q sensor ‖²`, from the SVD of `M = Σ reference·sensorᵀ`.
/// The `det` correction keeps the answer in `SO(3)` rather than letting a
/// reflection win.
fn kabsch(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let u = svd.u.expect("SVD computed with u");
    let vt = svd.v_t.expect("SVD computed with v_t");
    let d = (u * vt).determinant().signum();
    u * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d) * vt
}

/// Eigenvalues of a symmetric 3×3 matrix, ascending.
fn symmetric_eigenvalues(m: &Matrix3<f64>) -> [f64; 3] {
    let mut e: Vec<f64> = m.symmetric_eigenvalues().iter().copied().collect();
    e.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    [e[0], e[1], e[2]]
}

/// Any unit vector orthogonal to `v`, chosen so the construction never divides
/// by a near-zero: the smallest component of `v` names the axis furthest from
/// parallel to it.
fn any_orthogonal(v: &Vector3<f64>) -> Vector3<f64> {
    let axis = if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        Vector3::x()
    } else if v.y.abs() <= v.z.abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    v.cross(&axis).normalize()
}

/// Centred difference over `±stride` samples: `(x[k+m] − x[k−m]) / (2 m dt)`,
/// clamped at the ends.
///
/// The wide stencil is not an optimisation, it is the point (§2.3). Over a
/// one-sample stencil the gyro's white noise is amplified by `√2/(2dt)` —
/// at 833 Hz and σ ≈ 0.087 rad/s that is 50 rad/s², an order of magnitude
/// above the ω̇ of a hand tumble. Pairing the stencil half-width with the
/// smoothing half-width makes the two averaged windows disjoint, so their
/// noise is independent and the result is `σ/√(2m+1) · √2/(2m·dt)`.
fn differentiate_wide(series: &[Vector3<f64>], dt: f64, stride: usize) -> Vec<Vector3<f64>> {
    let n = series.len();
    let m = stride.max(1);
    (0..n)
        .map(|k| {
            let lo = k.saturating_sub(m);
            let hi = (k + m).min(n - 1);
            if hi == lo {
                Vector3::zeros()
            } else {
                (series[hi] - series[lo]) / ((hi - lo) as f64 * dt)
            }
        })
        .collect()
}

/// Standard deviation left in one element of `ω̇`, rad/s², given the raw
/// per-sample gyro σ, the smoothing half-width and the sample interval.
///
/// The centred average over `2h+1` samples divides the noise by `√(2h+1)`, and
/// the `±h` stencil then differences two windows that share exactly one
/// sample — effectively independent — over `2h·dt`, which contributes `√2`
/// over that span.
fn rate_derivative_sigma(raw_sigma: f64, half: usize, dt: f64) -> f64 {
    let window = (2 * half + 1) as f64;
    let span = 2.0 * half as f64 * dt;
    raw_sigma * std::f64::consts::SQRT_2 / (window.sqrt() * span)
}

/// The three segments of §2.0, as half-open sample ranges over the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segments {
    rest: (usize, usize),
    datum: (usize, usize),
    steer: (usize, usize),
}

impl Segments {
    fn rest_len(&self) -> usize {
        self.rest.1.saturating_sub(self.rest.0)
    }
    fn datum_len(&self) -> usize {
        self.datum.1.saturating_sub(self.datum.0)
    }
    fn steer_len(&self) -> usize {
        self.steer.1.saturating_sub(self.steer.0)
    }
}

/// Finds the §2.0 segments from the data alone — never from an assumed
/// schedule, because an operator's timing is not a contract.
///
/// The rest hold is the leading run whose smoothed `‖ω‖` stays under
/// [`REST_RATE_THRESHOLD_RAD_S`]. The bar turn begins where the smoothed
/// `| ‖ω_F‖² − ‖ω_R‖² |` first clears [`STEER_DISCRIMINANT_THRESHOLD`]; that
/// quantity is `2δ̇ sᵀω_R + δ̇²` (§2.4), and its second term is the reason a
/// plain norm difference will not do — that one vanishes whenever the bar rate
/// happens to sit across the body rate, and the bars would look straight.
///
/// Both boundaries are then pulled back by the smoothing half-width, so the
/// window that bleeds a later segment's energy backwards cannot put one of its
/// samples in an earlier segment.
fn segment(
    rear_rate: &[Vector3<f64>],
    front_rate: Option<&[Vector3<f64>]>,
    dt: f64,
) -> Segments {
    let n = rear_rate.len();
    let half = ((SEGMENT_SMOOTH_S / dt).round() as usize).max(1);

    let rear_norms: Vec<f64> = rear_rate.iter().map(|w| w.norm()).collect();
    let smoothed = smooth(&rear_norms, half);
    let mut rest_end = 0;
    while rest_end < n && smoothed[rest_end] < REST_RATE_THRESHOLD_RAD_S {
        rest_end += 1;
    }
    let rest_end = rest_end.saturating_sub(half);

    let steer_start = match front_rate {
        None => n,
        Some(front) => {
            // Smooth the **signed** discriminant and take the magnitude
            // afterwards. Rectifying first would average |noise| instead of
            // noise: at the spec's per-sample σ the signed quantity swings
            // ±0.7 s⁻² through the datum, whose mean absolute value alone
            // clears any usable threshold, and the bars would look turned
            // from the first sample of the tumble.
            let discriminant: Vec<f64> = (0..n)
                .map(|k| front[k].norm_squared() - rear_rate[k].norm_squared())
                .collect();
            let smoothed: Vec<f64> =
                smooth(&discriminant, half).into_iter().map(f64::abs).collect();
            let mut k = rest_end;
            while k < n && smoothed[k] < STEER_DISCRIMINANT_THRESHOLD {
                k += 1;
            }
            if k >= n {
                n
            } else {
                k.saturating_sub(half).max(rest_end)
            }
        }
    };

    Segments { rest: (0, rest_end), datum: (rest_end, steer_start), steer: (steer_start, n) }
}

/// Everything the residuals read that no parameter changes: the streams, the
/// segments, the weights and the cached `Ω(t)`.
struct Context<'a> {
    sensors: &'a [SensorStream],
    reference: usize,
    front: Option<usize>,
    /// Rear sensors other than the reference, ascending by `imu_index`.
    rear_others: Vec<usize>,
    segments: Segments,
    /// `Ω(t) = [ω̇]ₓ + [ω]ₓ²` for every sample, s⁻², from the reference
    /// sensor's smoothed rate. Held fixed through LM: it depends on the
    /// reference gyro bias only through a constant offset of `ω`, whose effect
    /// on `Ω` is second order, and freezing it keeps the accelerometer
    /// residual linear in the lever arm and the bias difference.
    omega_cap: Vec<Matrix3<f64>>,
    /// 1/σ for a gyro sample, (rad/s)⁻¹.
    gyro_weight: f64,
    /// 1/σ for an accel sample, (m/s²)⁻¹.
    accel_weight: f64,
    /// Standard deviation of the noise left in each element of `ω̇` after the
    /// smoothing and the wide stencil, rad/s². Feeds the errors-in-variables
    /// correction in [`solve_lever_and_bias`].
    rate_derivative_sigma: f64,
    /// Whether each block of residuals is active at all.
    fit_rotations: bool,
    fit_levers: bool,
    fit_steer: bool,
    have_rest: bool,
}

/// The parameters, in the form the residuals consume them.
#[derive(Debug, Clone)]
struct Solution {
    /// Rotation sensor frame → body frame, per sensor. The reference's is the
    /// §1.2a identity gauge and is never perturbed.
    mount: Vec<Matrix3<f64>>,
    /// Gyro bias in the sensor frame, per sensor, rad/s.
    gyro_bias: Vec<Vector3<f64>>,
    /// Steering axis in the rear body frame.
    steer_axis: Vector3<f64>,
    /// Lever arm per entry of [`Context::rear_others`], metres.
    levers: Vec<Vector3<f64>>,
    /// `β_i,ref` per entry of [`Context::rear_others`], m/s².
    bias_differences: Vec<Vector3<f64>>,
}

impl Solution {
    /// Bias-corrected rate of sensor `i` at sample `k`, in the sensor frame.
    fn corrected(&self, ctx: &Context, i: usize, k: usize) -> Vector3<f64> {
        ctx.sensors[i].gyro_rad_s[k] - self.gyro_bias[i]
    }

    /// The rear body's rate at sample `k`, in the rear body frame. The
    /// reference sensor's mount is the identity gauge, so this is just its
    /// bias-corrected reading.
    fn rear_rate(&self, ctx: &Context, k: usize) -> Vector3<f64> {
        self.mount[ctx.reference] * self.corrected(ctx, ctx.reference, k)
    }

    /// The front body's rate at sample `k`, in the **front body** frame.
    fn front_rate(&self, ctx: &Context, k: usize) -> Option<Vector3<f64>> {
        ctx.front.map(|f| self.mount[f] * self.corrected(ctx, f, k))
    }
}

/// `δ̇(t) = sᵀ(ω_F − ω_R)` (§2.4 equation (8)), then `δ(t)` by trapezoidal
/// integration forward from `δ = 0` at the first sample of the bar turn.
///
/// The steering axis has the same coordinates in both body frames — a rotation
/// about `s` leaves `s` alone — so no frame conversion belongs here.
fn steer_angles(sol: &Solution, ctx: &Context, dt: f64) -> (Vec<f64>, Vec<f64>) {
    let (lo, hi) = ctx.segments.steer;
    let n = hi.saturating_sub(lo);
    let mut rates = Vec::with_capacity(n);
    for k in lo..hi {
        let front = sol.front_rate(ctx, k).unwrap_or_else(Vector3::zeros);
        let rear = sol.rear_rate(ctx, k);
        rates.push(sol.steer_axis.dot(&(front - rear)));
    }
    let mut angles = Vec::with_capacity(n);
    let mut running = 0.0;
    for k in 0..n {
        if k > 0 {
            running += 0.5 * (rates[k] + rates[k - 1]) * dt;
        }
        angles.push(running);
    }
    (angles, rates)
}

/// Solves §2.4's equation (7) for the steering axis over the bar-turn segment.
///
/// (7) is linear in `S = s sᵀ`, so the six independent entries of `S` come
/// from a 6×6 normal-equation system, with `tr S = 1` added as a row weighted
/// by the sample count so it carries the same authority as the data. `s` is
/// then the dominant eigenvector of the symmetrised solution, signed so it
/// points up: `(s, δ)` and `(−s, −δ)` describe the same hinge, and up is what
/// `BikeGeometry::steer_axis` uses.
fn solve_steer_axis(sol: &Solution, ctx: &Context) -> Option<Vector3<f64>> {
    let (lo, hi) = ctx.segments.steer;
    let count = hi.saturating_sub(lo);
    if count < 2 {
        return None;
    }

    let mut normal: SMatrix<f64, 6, 6> = SMatrix::zeros();
    let mut rhs: SVector<f64, 6> = SVector::zeros();
    for k in lo..hi {
        let front = sol.front_rate(ctx, k)?;
        let rear = sol.rear_rate(ctx, k);
        let m = rear * rear.transpose() - front * front.transpose();
        let row = SVector::<f64, 6>::from_column_slice(&[
            m[(0, 0)],
            m[(1, 1)],
            m[(2, 2)],
            2.0 * m[(0, 1)],
            2.0 * m[(0, 2)],
            2.0 * m[(1, 2)],
        ]);
        let target = rear.norm_squared() - front.norm_squared();
        normal += row * row.transpose();
        rhs += row * target;
    }

    let trace_row = SVector::<f64, 6>::from_column_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    let trace_weight = count as f64;
    normal += trace_row * trace_row.transpose() * trace_weight;
    rhs += trace_row * trace_weight;

    let x = normal.lu().solve(&rhs)?;
    let s_matrix = Matrix3::new(x[0], x[3], x[4], x[3], x[1], x[5], x[4], x[5], x[2]);
    let eigen = s_matrix.symmetric_eigen();
    let (best, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))?;
    let mut axis = Vector3::new(
        eigen.eigenvectors[(0, best)],
        eigen.eigenvectors[(1, best)],
        eigen.eigenvectors[(2, best)],
    );
    if axis.norm() < 1e-12 {
        return None;
    }
    axis.normalize_mut();
    if axis.z < 0.0 {
        axis = -axis;
    }
    Some(axis)
}

/// Solves §2.3's equation (6) for one rear-body pair: the lever arm relative
/// to the reference sensor and the accel-bias difference, by stacked linear
/// least squares.
///
/// Accumulated as a 6×6 normal-equation system rather than a `3N×6` design
/// matrix: at 833 Hz over 30 s that is 288 bytes instead of 3.6 MB, and this
/// machine is memory-bound (R13). Conditioning is adequate because `Ω` runs
/// `O(1..10) s⁻²` against an identity block of exactly 1 — the two column
/// groups sit within an order of magnitude of each other.
fn solve_lever_and_bias(
    sol: &Solution,
    ctx: &Context,
    sensor: usize,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let (lo, hi) = ctx.segments.datum;
    if hi.saturating_sub(lo) < 6 {
        return None;
    }

    let mut normal: SMatrix<f64, 6, 6> = SMatrix::zeros();
    let mut rhs: SVector<f64, 6> = SVector::zeros();
    let reference = ctx.reference;
    for k in lo..hi {
        let y = sol.mount[sensor] * ctx.sensors[sensor].accel_m_s2[k]
            - sol.mount[reference] * ctx.sensors[reference].accel_m_s2[k];
        let cap = ctx.omega_cap[k];
        // A = [ Ω | I ], one 3×6 block per sample.
        let mut design: SMatrix<f64, 3, 6> = SMatrix::zeros();
        design.fixed_view_mut::<3, 3>(0, 0).copy_from(&cap);
        design.fixed_view_mut::<3, 3>(0, 3).copy_from(&Matrix3::identity());
        normal += design.transpose() * design;
        rhs += design.transpose() * y;
    }

    // Errors-in-variables correction. `Ω` is not data, it is an estimate: its
    // `[ω̇]ₓ` half carries noise of standard deviation σ_d per element, and
    // `E[Ω̂ᵀΩ̂] = ΩᵀΩ + N·E[ΔᵀΔ]` with `E[[d]ₓᵀ[d]ₓ] = 2σ_d² I`. Left in, that
    // term attenuates the lever arm by a fraction of a percent. Subtracting it
    // makes the estimator consistent rather than merely close: on the
    // synthetic body at §5's noise level it is worth about a millimetre on the
    // worst axis, which is small because the lever block is kept out of LM
    // (see `refine`) — left in, the same inconsistency compounded into a 77
    // mm/s² error on the bias difference. Only the `[ω̇]ₓ` term is corrected:
    // the `[ω]ₓ[n]ₓ` cross terms are an order of magnitude smaller and their
    // squares two.
    let count = (hi - lo) as f64;
    let attenuation = 2.0 * ctx.rate_derivative_sigma * ctx.rate_derivative_sigma * count;
    for axis in 0..3 {
        normal[(axis, axis)] -= attenuation;
    }

    let x = normal.lu().solve(&rhs)?;
    Some((Vector3::new(x[0], x[1], x[2]), Vector3::new(x[3], x[4], x[5])))
}

// ── §2.6 Levenberg–Marquardt ────────────────────────────────────────────

/// Where each block of parameters sits in the LM vector.
#[derive(Debug, Clone)]
struct Layout {
    /// Sensors whose rotation is refined (everything but the reference), each
    /// with the offset of its 3-vector tangent increment.
    rotations: Vec<(usize, usize)>,
    /// Every sensor, with the offset of its 3-vector gyro bias.
    biases: Vec<(usize, usize)>,
    /// Offset of the steering axis's 2-vector tangent increment.
    steer: Option<usize>,
    /// Total parameter count.
    len: usize,
}

impl Layout {
    fn build(ctx: &Context) -> Layout {
        let mut len = 0;
        let mut rotations = Vec::new();
        if ctx.fit_rotations {
            for i in 0..ctx.sensors.len() {
                if i != ctx.reference {
                    rotations.push((i, len));
                    len += 3;
                }
            }
        }
        let mut biases = Vec::new();
        if ctx.have_rest {
            for i in 0..ctx.sensors.len() {
                biases.push((i, len));
                len += 3;
            }
        }
        let steer = if ctx.fit_steer {
            len += 2;
            Some(len - 2)
        } else {
            None
        };
        Layout { rotations, biases, steer, len }
    }
}

/// Applies a parameter increment to a solution. Rotations take a tangent
/// 3-vector through the exponential map, re-linearised every iteration, so the
/// increment is unconstrained and no norm constraint enters the normal
/// equations (§2.6). The steering axis takes a 2-vector in the tangent plane
/// of the current axis, then renormalises.
fn apply_delta(base: &Solution, layout: &Layout, delta: &DVector<f64>) -> Solution {
    let mut out = base.clone();

    for &(sensor, offset) in &layout.rotations {
        let tangent = Vector3::new(delta[offset], delta[offset + 1], delta[offset + 2]);
        let increment = if tangent.norm() < 1e-15 {
            Matrix3::identity()
        } else {
            *UnitQuaternion::from_scaled_axis(tangent).to_rotation_matrix().matrix()
        };
        out.mount[sensor] = increment * base.mount[sensor];
    }

    for &(sensor, offset) in &layout.biases {
        out.gyro_bias[sensor] = base.gyro_bias[sensor]
            + Vector3::new(delta[offset], delta[offset + 1], delta[offset + 2]);
    }

    if let Some(offset) = layout.steer {
        let e1 = any_orthogonal(&base.steer_axis);
        let e2 = base.steer_axis.cross(&e1);
        let moved = base.steer_axis + e1 * delta[offset] + e2 * delta[offset + 1];
        if moved.norm() > 1e-12 {
            out.steer_axis = moved.normalize();
        }
    }

    out
}

/// The samples of one segment that an LM sweep visits: every one of them,
/// strided so the segment contributes no more than `limit`.
fn strided(range: (usize, usize), limit: Option<usize>) -> impl Iterator<Item = usize> {
    let (lo, hi) = range;
    let count = hi.saturating_sub(lo);
    let stride = match limit {
        Some(limit) if limit > 0 && count > limit => count.div_ceil(limit),
        _ => 1,
    };
    (lo..hi).step_by(stride.max(1))
}

/// Which residual block a sample belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    Rest,
    Datum,
    Steer,
}

/// The weighted residuals at one sample (§2.6), appended to `out`.
///
/// - `Rest`: `(ω_i − b_i)/σ_g` for every sensor — §2.1, the only place an
///   individual gyro bias is observable.
/// - `Datum`: `(R_i(ω_i − b_i) − ω_R)/σ_g` for every non-reference sensor, the
///   front one included because `δ ≡ 0` there (§2.2); plus
///   `(R_i a_i − a_ref − Ω r_i − β_i)/σ_a` for each rear pair (§2.3).
/// - `Steer`: `(exp([s]δ) ω_F − ω_R − δ̇ s)/σ_g`, equation (4).
fn residuals(
    ctx: &Context,
    sol: &Solution,
    block: Block,
    k: usize,
    angles: &[f64],
    rates: &[f64],
    out: &mut Vec<f64>,
) {
    out.clear();
    match block {
        Block::Rest => {
            if !ctx.have_rest {
                return;
            }
            for i in 0..ctx.sensors.len() {
                let r = sol.corrected(ctx, i, k) * ctx.gyro_weight;
                out.extend_from_slice(&[r.x, r.y, r.z]);
            }
        }
        Block::Datum => {
            let rear = sol.rear_rate(ctx, k);
            if ctx.fit_rotations {
                for i in 0..ctx.sensors.len() {
                    if i == ctx.reference {
                        continue;
                    }
                    let r = (sol.mount[i] * sol.corrected(ctx, i, k) - rear) * ctx.gyro_weight;
                    out.extend_from_slice(&[r.x, r.y, r.z]);
                }
            }
            if ctx.fit_levers {
                let reference_accel =
                    sol.mount[ctx.reference] * ctx.sensors[ctx.reference].accel_m_s2[k];
                for (slot, &i) in ctx.rear_others.iter().enumerate() {
                    let predicted = ctx.omega_cap[k] * sol.levers[slot] + sol.bias_differences[slot];
                    let r = (sol.mount[i] * ctx.sensors[i].accel_m_s2[k]
                        - reference_accel
                        - predicted)
                        * ctx.accel_weight;
                    out.extend_from_slice(&[r.x, r.y, r.z]);
                }
            }
        }
        Block::Steer => {
            if !ctx.fit_steer {
                return;
            }
            let Some(front) = sol.front_rate(ctx, k) else {
                return;
            };
            let index = k - ctx.segments.steer.0;
            let (delta, delta_rate) = match (angles.get(index), rates.get(index)) {
                (Some(&a), Some(&r)) => (a, r),
                _ => return,
            };
            let rotation = UnitQuaternion::from_scaled_axis(sol.steer_axis * delta);
            let rear = sol.rear_rate(ctx, k);
            let r = (rotation * front - rear - sol.steer_axis * delta_rate) * ctx.gyro_weight;
            out.extend_from_slice(&[r.x, r.y, r.z]);
        }
    }
}

/// Weighted sum of squared residuals (§2.6); with `perturbed` and `normal`
/// supplied, the `JᵀJ` and `Jᵀr` of a central-difference Jacobian are
/// accumulated in the same sweep.
///
/// Residuals are streamed sample by sample and never stored: at 833 Hz over a
/// minute, materialising 23 Jacobian columns would cost about 100 MB, and this
/// machine is memory-bound (R13).
fn evaluate(
    ctx: &Context,
    dt: f64,
    solution: &Solution,
    layout: &Layout,
    perturbed: Option<&[(Solution, Solution)]>,
    limit: Option<usize>,
    mut normal: Option<(&mut DMatrix<f64>, &mut DVector<f64>)>,
) -> f64 {
    let mut cost = 0.0;
    let width = layout.len;
    let mut base_residuals: Vec<f64> = Vec::with_capacity(16);
    let mut minus_residuals: Vec<f64> = Vec::with_capacity(16);
    let mut plus_residuals: Vec<f64> = Vec::with_capacity(16);
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(16); width];

    // δ(t) is profiled out of the parameter vector: recomputed from whichever
    // steering axis is in hand rather than carried as an unknown (§2.6). The
    // perturbed solutions get their own, since a perturbed axis implies a
    // perturbed angle.
    let profile = |s: &Solution| {
        if ctx.fit_steer {
            steer_angles(s, ctx, dt)
        } else {
            (Vec::new(), Vec::new())
        }
    };
    let (angles, rates) = profile(solution);
    let perturbed_profiles: Vec<((Vec<f64>, Vec<f64>), (Vec<f64>, Vec<f64>))> = match perturbed {
        Some(pairs) => pairs.iter().map(|(m, p)| (profile(m), profile(p))).collect(),
        None => Vec::new(),
    };

    for block in [Block::Rest, Block::Datum, Block::Steer] {
        let range = match block {
            Block::Rest => ctx.segments.rest,
            Block::Datum => ctx.segments.datum,
            Block::Steer => ctx.segments.steer,
        };
        for k in strided(range, limit) {
            residuals(ctx, solution, block, k, &angles, &rates, &mut base_residuals);
            if base_residuals.is_empty() {
                continue;
            }
            for r in base_residuals.iter() {
                cost += r * r;
            }

            let (Some(pairs), Some((jtj, jtr))) = (perturbed, normal.as_mut()) else {
                continue;
            };
            for (p, (minus, plus)) in pairs.iter().enumerate() {
                let (minus_profile, plus_profile) = &perturbed_profiles[p];
                residuals(ctx, minus, block, k, &minus_profile.0, &minus_profile.1, &mut minus_residuals);
                residuals(ctx, plus, block, k, &plus_profile.0, &plus_profile.1, &mut plus_residuals);
                columns[p].clear();
                for r in 0..base_residuals.len() {
                    let lo = minus_residuals.get(r).copied().unwrap_or(0.0);
                    let hi = plus_residuals.get(r).copied().unwrap_or(0.0);
                    columns[p].push((hi - lo) / (2.0 * LM_JACOBIAN_STEP));
                }
            }
            for p in 0..width {
                for q in p..width {
                    let mut sum = 0.0;
                    for r in 0..base_residuals.len() {
                        sum += columns[p][r] * columns[q][r];
                    }
                    jtj[(p, q)] += sum;
                    if p != q {
                        jtj[(q, p)] += sum;
                    }
                }
                let mut sum = 0.0;
                for r in 0..base_residuals.len() {
                    sum += columns[p][r] * base_residuals[r];
                }
                jtr[p] += sum;
            }
        }
    }

    cost
}

/// Levenberg–Marquardt over the §2.6 parameter vector, with a central-
/// difference Jacobian. Returns the refined solution and the number of steps
/// accepted.
///
/// The vector is the rotations, the gyro biases and the steering axis — **not**
/// the lever arm or the accel-bias difference, which the draft's §2.6 listed
/// and the code rules out. Those two enter the residual only through `Ω(t)`,
/// which is an *estimate* of the body's motion rather than data; least squares
/// against a noisy regressor is inconsistent, so an LM that is free to move
/// them lands on the attenuated answer and undoes the errors-in-variables
/// correction [`solve_lever_and_bias`] applies. On the synthetic body letting
/// LM have them cost a factor of four in the lever arm (2.9 mm → 1.1 mm) and
/// six in the bias difference (77 mm/s² → 13 mm/s², through and under §5's
/// 50 mm/s²). The accelerometer residual stays in the objective, so those two
/// still constrain everything LM does move.
///
/// The sweep is strided (`LM_MAX_SAMPLES_PER_SEGMENT`), which is safe because
/// the caller keeps the refinement only if it also lowers the cost measured on
/// every sample: a step a subset liked but the whole session does not cannot
/// make the answer worse.
fn refine(ctx: &Context, dt: f64, seed: &Solution) -> (Solution, u32) {
    let layout = Layout::build(ctx);
    if layout.len == 0 {
        return (seed.clone(), 0);
    }

    let limit = Some(LM_MAX_SAMPLES_PER_SEGMENT);
    let mut current = seed.clone();
    let mut cost = evaluate(ctx, dt, &current, &layout, None, limit, None);
    let mut lambda = LM_INITIAL_LAMBDA;
    let mut taken = 0;

    for _ in 0..LM_MAX_ITERATIONS {
        // One perturbed pair per parameter, built once and reused across every
        // sample of the sweep.
        let mut pairs = Vec::with_capacity(layout.len);
        for p in 0..layout.len {
            let mut minus = DVector::zeros(layout.len);
            let mut plus = DVector::zeros(layout.len);
            minus[p] = -LM_JACOBIAN_STEP;
            plus[p] = LM_JACOBIAN_STEP;
            pairs.push((
                apply_delta(&current, &layout, &minus),
                apply_delta(&current, &layout, &plus),
            ));
        }

        let mut jtj = DMatrix::zeros(layout.len, layout.len);
        let mut jtr = DVector::zeros(layout.len);
        evaluate(ctx, dt, &current, &layout, Some(&pairs), limit, Some((&mut jtj, &mut jtr)));

        let mean_diagonal = (0..layout.len).map(|p| jtj[(p, p)]).sum::<f64>() / layout.len as f64;
        if !(mean_diagonal.is_finite() && mean_diagonal > 0.0) {
            break;
        }

        let mut improved = false;
        for _ in 0..6 {
            let mut damped = jtj.clone();
            for p in 0..layout.len {
                damped[(p, p)] += lambda * mean_diagonal;
            }
            let Some(step) = damped.clone().lu().solve(&(-&jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let candidate = apply_delta(&current, &layout, &step);
            let candidate_cost = evaluate(ctx, dt, &candidate, &layout, None, limit, None);
            if candidate_cost.is_finite() && candidate_cost < cost {
                let relative = (cost - candidate_cost) / cost.max(f64::MIN_POSITIVE);
                current = candidate;
                cost = candidate_cost;
                lambda = (lambda * 0.3).max(1e-12);
                taken += 1;
                improved = relative > LM_COST_TOLERANCE;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }

    (current, taken)
}

// ── the entry point ─────────────────────────────────────────────────────

/// Rejects an input the model cannot be run against at all. Everything here is
/// a caller mistake or a corrupt stream, never a judgement about the motion —
/// poor motion is a shortfall, not an error (§3).
fn validate(input: &CalibrationInput) -> Result<usize, CalibrationError> {
    if !(input.sample_rate_hz.is_finite() && input.sample_rate_hz > 0.0) {
        return Err(CalibrationError::BadSampleRate(input.sample_rate_hz));
    }
    if input.sensors.is_empty() {
        return Err(CalibrationError::NoSensors);
    }

    let n = input.sensors[0].gyro_rad_s.len();
    for sensor in &input.sensors {
        if sensor.gyro_rad_s.len() != sensor.accel_m_s2.len() || sensor.gyro_rad_s.len() != n {
            return Err(CalibrationError::RaggedStreams {
                imu_index: sensor.imu_index,
                gyro: sensor.gyro_rad_s.len(),
                accel: sensor.accel_m_s2.len(),
            });
        }
        for k in 0..n {
            if !sensor.gyro_rad_s[k].iter().all(|v| v.is_finite())
                || !sensor.accel_m_s2[k].iter().all(|v| v.is_finite())
            {
                return Err(CalibrationError::NonFiniteSample {
                    imu_index: sensor.imu_index,
                    sample: k,
                });
            }
        }
    }

    // Three samples is the floor for a centred difference; below that nothing
    // downstream has anything to work with.
    if n < 3 {
        return Err(CalibrationError::TooFewSamples { seen: n, needed: 3 });
    }
    Ok(n)
}

/// Fits the §1 model to one session of the §5.1 protocol.
///
/// Runs the staged closed form of §2.1–§2.4, then the Levenberg–Marquardt
/// refinement of §2.6, keeping whichever of the two has the lower cost over
/// the whole session. Every §3 gate is measured and reported; a gate that
/// fails drops the fields it governs from the record and adds an
/// [`ExcitationShortfall`], so a thin session yields a thin record rather than
/// a failure.
///
/// # Errors
///
/// [`CalibrationError`] when the streams are unusable — ragged, non-finite,
/// too short, or with no sensor on the rear body — or
/// [`CalibrationError::MotionNotRich`] when not even the rest hold survived,
/// leaving nothing at all to report.
pub fn calibrate(input: &CalibrationInput) -> Result<CalibrationRecord, CalibrationError> {
    let n = validate(input)?;
    let dt = 1.0 / input.sample_rate_hz;
    let sensor_count = input.sensors.len();

    let mut rear: Vec<usize> = (0..sensor_count)
        .filter(|&i| input.sensors[i].body == Body::Rear)
        .collect();
    rear.sort_by_key(|&i| input.sensors[i].imu_index);
    let reference = *rear.first().ok_or(CalibrationError::NoReferenceSensor)?;
    let rear_others: Vec<usize> = rear.iter().copied().filter(|&i| i != reference).collect();
    let front = (0..sensor_count).find(|&i| input.sensors[i].body == Body::Front);

    // ── §2.0 segments, from the raw reference rate ───────────────────────
    // Bias is not known yet; at the rates that matter here a milliradian of
    // bias moves no boundary, so the raw streams segment the session.
    let front_rate = front.map(|f| input.sensors[f].gyro_rad_s.as_slice());
    let segments = segment(&input.sensors[reference].gyro_rad_s, front_rate, dt);

    let rest_s = segments.rest_len() as f64 * dt;
    let datum_s = segments.datum_len() as f64 * dt;
    let have_rest = rest_s >= MIN_REST_S;

    let mut shortfalls = Vec::new();
    if !have_rest {
        shortfalls.push(ExcitationShortfall::RestHold { seen_s: rest_s, needed_s: MIN_REST_S });
    }
    if datum_s < MIN_DATUM_S {
        shortfalls.push(ExcitationShortfall::Duration { seen_s: datum_s, needed_s: MIN_DATUM_S });
    }

    // ── §2.1 gyro bias, from the rest hold ───────────────────────────────
    let mut gyro_bias = vec![Vector3::zeros(); sensor_count];
    if have_rest {
        let (lo, hi) = segments.rest;
        let count = (hi - lo) as f64;
        for i in 0..sensor_count {
            let sum: Vector3<f64> = (lo..hi).map(|k| input.sensors[i].gyro_rad_s[k]).sum();
            gyro_bias[i] = sum / count;
        }
    }

    // ── excitation metrics over the tumble (§3) ──────────────────────────
    let corrected_reference: Vec<Vector3<f64>> = (0..n)
        .map(|k| input.sensors[reference].gyro_rad_s[k] - gyro_bias[reference])
        .collect();

    let (datum_lo, datum_hi) = segments.datum;
    let mut scatter = Matrix3::zeros();
    let mut norms = Vec::with_capacity(segments.datum_len());
    for k in datum_lo..datum_hi {
        let w = corrected_reference[k];
        scatter += w * w.transpose();
        norms.push(w.norm());
    }
    let eigenvalues = symmetric_eigenvalues(&scatter);
    let rotation_condition = if eigenvalues[0] > 0.0 {
        Some(eigenvalues[2] / eigenvalues[0])
    } else {
        None
    };
    let median_rate = if norms.is_empty() { None } else { Some(median(&norms)) };

    // ── §2.3 Ω(t), from the smoothed rate ────────────────────────────────
    let smooth_half = ((RATE_SMOOTH_S / dt).round() as usize).max(1);
    let smoothed_rate = smooth_vectors(&corrected_reference, smooth_half);
    let rate_derivative = differentiate_wide(&smoothed_rate, dt, smooth_half);
    let omega_cap: Vec<Matrix3<f64>> = (0..n)
        .map(|k| {
            let s = skew(&smoothed_rate[k]);
            skew(&rate_derivative[k]) + s * s
        })
        .collect();

    let mut lever_scatter = Matrix3::zeros();
    for k in datum_lo..datum_hi {
        lever_scatter += omega_cap[k].transpose() * omega_cap[k];
    }
    let lever_richness = if segments.datum_len() > 0 {
        Some(symmetric_eigenvalues(&lever_scatter)[0] / segments.datum_len() as f64)
    } else {
        None
    };

    if let Some(condition) = rotation_condition {
        if condition > MAX_ROTATION_CONDITION {
            shortfalls.push(ExcitationShortfall::RotationRichness {
                seen: condition,
                needed: MAX_ROTATION_CONDITION,
            });
        }
    }
    if let Some(rate) = median_rate {
        if rate < MIN_MEDIAN_RATE_RAD_S {
            shortfalls.push(ExcitationShortfall::RateMagnitude {
                seen: rate,
                needed: MIN_MEDIAN_RATE_RAD_S,
            });
        }
    }
    if let Some(richness) = lever_richness {
        if richness < MIN_LEVER_RICHNESS {
            shortfalls.push(ExcitationShortfall::LeverArmRichness {
                seen: richness,
                needed: MIN_LEVER_RICHNESS,
            });
        }
    }
    if front.is_none() {
        shortfalls.push(ExcitationShortfall::NoFrontSensor);
    }
    if rear_others.is_empty() {
        shortfalls.push(ExcitationShortfall::NoRearPair);
    }

    let rotation_ok = rotation_condition.is_some_and(|c| c <= MAX_ROTATION_CONDITION)
        && median_rate.is_some_and(|r| r >= MIN_MEDIAN_RATE_RAD_S);
    let lever_ok = rotation_ok
        && !rear_others.is_empty()
        && lever_richness.is_some_and(|r| r >= MIN_LEVER_RICHNESS);

    // ── §2.2 rotations, by Kabsch over the tumble ────────────────────────
    let mut mount = vec![Matrix3::<f64>::identity(); sensor_count];
    if rotation_ok {
        for i in 0..sensor_count {
            if i == reference {
                continue;
            }
            let mut m = Matrix3::zeros();
            for k in datum_lo..datum_hi {
                let sensor_rate = input.sensors[i].gyro_rad_s[k] - gyro_bias[i];
                m += corrected_reference[k] * sensor_rate.transpose();
            }
            mount[i] = kabsch(&m);
        }
    }

    let mut ctx = Context {
        sensors: &input.sensors,
        reference,
        front,
        rear_others: rear_others.clone(),
        segments,
        omega_cap,
        gyro_weight: 1.0 / (input.noise.gyro_arw * input.sample_rate_hz.sqrt()),
        accel_weight: 1.0 / (input.noise.accel_vrw * input.sample_rate_hz.sqrt()),
        rate_derivative_sigma: rate_derivative_sigma(
            input.noise.gyro_arw * input.sample_rate_hz.sqrt(),
            smooth_half,
            dt,
        ),
        fit_rotations: rotation_ok,
        fit_levers: lever_ok,
        fit_steer: false,
        have_rest,
    };

    let mut solution = Solution {
        mount,
        gyro_bias,
        steer_axis: Vector3::z(),
        levers: vec![Vector3::zeros(); rear_others.len()],
        bias_differences: vec![Vector3::zeros(); rear_others.len()],
    };

    // ── §2.4 the steering axis, then δ(t) ────────────────────────────────
    let mut steer_peak_to_peak = None;
    if rotation_ok && front.is_some() && segments.steer_len() > 2 {
        if let Some(axis) = solve_steer_axis(&solution, &ctx) {
            solution.steer_axis = axis;
            ctx.fit_steer = true;
            let (angles, _) = steer_angles(&solution, &ctx, dt);
            let lo = angles.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = hi - lo;
            steer_peak_to_peak = Some(span);
            if span < MIN_STEER_PEAK_TO_PEAK_RAD {
                shortfalls.push(ExcitationShortfall::SteerRichness {
                    seen: span,
                    needed: MIN_STEER_PEAK_TO_PEAK_RAD,
                });
                ctx.fit_steer = false;
            }
        }
    }

    // ── §2.3 lever arms and the accel-bias difference ────────────────────
    if lever_ok {
        for (slot, &i) in rear_others.iter().enumerate() {
            if let Some((lever, difference)) = solve_lever_and_bias(&solution, &ctx, i) {
                solution.levers[slot] = lever;
                solution.bias_differences[slot] = difference;
            }
        }
    }

    // Nothing at all survived: not even a rest hold to take gyro bias from.
    if !have_rest && !rotation_ok && !ctx.fit_steer {
        return Err(CalibrationError::MotionNotRich(shortfalls));
    }

    // ── §2.6 refinement, kept only if it helps over the whole session ────
    let layout = Layout::build(&ctx);
    let (refined, lm_iterations) = refine(&ctx, dt, &solution);
    let seed_cost = evaluate(&ctx, dt, &solution, &layout, None, None, None);
    let refined_cost = evaluate(&ctx, dt, &refined, &layout, None, None, None);
    let (solution, lm_iterations) = if refined_cost.is_finite() && refined_cost < seed_cost {
        (refined, lm_iterations)
    } else {
        (solution, 0)
    };

    // ── §7.6 the rear body's own consistency, after the fit ──────────────
    let rear_body_residual = if rotation_ok && !rear_others.is_empty() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for k in datum_lo..datum_hi {
            let body_rate = solution.rear_rate(&ctx, k);
            for &i in &rear_others {
                sum += (solution.mount[i] * solution.corrected(&ctx, i, k) - body_rate)
                    .norm_squared();
                count += 3;
            }
        }
        if count == 0 {
            None
        } else {
            Some((sum / count as f64).sqrt())
        }
    } else {
        None
    };

    // ── §6 the record ────────────────────────────────────────────────────
    let mut sensors: Vec<SensorCalibration> = (0..sensor_count)
        .map(|i| {
            let is_reference = i == reference;
            let is_body_origin = is_reference || Some(i) == front;
            let lever_slot = rear_others.iter().position(|&other| other == i);
            SensorCalibration {
                imu_index: input.sensors[i].imu_index,
                body: input.sensors[i].body,
                mount: UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
                    solution.mount[i],
                )),
                mount_origin: if is_reference || !rotation_ok {
                    MountOrigin::Gauge
                } else {
                    MountOrigin::Measured
                },
                // `r ≡ 0` for a body-origin sensor is a convention, not a
                // measurement, so it is reported as absent rather than zero.
                lever_m: match (is_body_origin, lever_ok, lever_slot) {
                    (false, true, Some(slot)) => Some(solution.levers[slot]),
                    _ => None,
                },
                gyro_bias_rad_s: if have_rest { Some(solution.gyro_bias[i]) } else { None },
            }
        })
        .collect();
    sensors.sort_by_key(|s| s.imu_index);

    let accel_bias_differences = if lever_ok {
        rear_others
            .iter()
            .enumerate()
            .map(|(slot, &i)| AccelBiasDifference {
                from_imu: input.sensors[i].imu_index,
                to_imu: input.sensors[reference].imu_index,
                value_m_s2: solution.bias_differences[slot],
            })
            .collect()
    } else {
        Vec::new()
    };

    let fitted_anything = rotation_ok || lever_ok || ctx.fit_steer;
    Ok(CalibrationRecord {
        model_version: MODEL_VERSION,
        captured_utc: input.captured_utc.clone(),
        source: if fitted_anything {
            CalibrationSource::RigidBody
        } else {
            CalibrationSource::RestOnly
        },
        sensors,
        accel_bias_differences,
        steer_axis: if ctx.fit_steer { Some(solution.steer_axis) } else { None },
        steer_datum: UnitQuaternion::identity(),
        quality: Quality {
            rotation_condition,
            median_rate_rad_s: median_rate,
            lever_richness,
            steer_peak_to_peak_rad: steer_peak_to_peak,
            rest_duration_s: rest_s,
            datum_duration_s: datum_s,
            rear_body_residual_rad_s: rear_body_residual,
            lm_iterations,
            shortfalls,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use crate::session::Session;
    use crate::synth::{self, Protocol, SynthConfig, Truth};

    /// The §5 acceptance thresholds, one place.
    const ROTATION_TOLERANCE_DEG: f64 = 0.5;
    const LEVER_TOLERANCE_M: f64 = 0.010;
    const STEER_AXIS_TOLERANCE_DEG: f64 = 1.0;
    const GYRO_BIAS_TOLERANCE_RAD_S: f64 = 0.002;
    const ACCEL_BIAS_DIFFERENCE_TOLERANCE_M_S2: f64 = 0.05;

    /// The generator's reference noise, which is also the estimator's working
    /// default (spec §1.3). `*_bias_rw` play no part in this fit — there is no
    /// time-varying bias in the model — so they are set to zero rather than
    /// invented.
    fn reference_noise() -> ImuNoise {
        ImuNoise {
            gyro_arw: synth::GYRO_SIGMA_RAD_S,
            accel_vrw: synth::ACCEL_SIGMA_M_S2,
            gyro_bias_rw: 0.0,
            accel_bias_rw: 0.0,
        }
    }

    /// Noise scale that puts the generator's output at the noise level §5's
    /// error budget is written against.
    ///
    /// The generator's `GYRO_SIGMA_RAD_S` is a **per-sample** σ in rad/s; the
    /// spec's 0.003 is an **angle-random-walk coefficient** in rad/√s, which
    /// at 833 Hz means a per-sample σ of `0.003·√833 ≈ 0.087` rad/s — the very
    /// figure §5 divides by to justify each threshold. At `--noise 1.0` the
    /// synthetic body is therefore 29× quieter than the acceptance arithmetic
    /// assumes, and a fit validated there would be flattered. Scaling by √ODR
    /// closes the gap, so these tests exercise the thresholds as written.
    const SPEC_NOISE_SCALE: f64 = 28.861739379323623; // √833

    /// A calibration session at the rate the spec's error budget assumes.
    fn calibration_config(imu_count: u8, noise_scale: f64) -> SynthConfig {
        SynthConfig {
            protocol: Protocol::Calibration,
            imu_rate_hz: 833,
            imu_count,
            noise_scale,
            ..SynthConfig::default()
        }
    }

    /// One channel's samples by registry name, in the units the file carries.
    fn channel(session: &Session, name: &str) -> Vec<f64> {
        session
            .channels
            .iter()
            .find(|c| c.channel_id == name)
            .unwrap_or_else(|| panic!("no channel {name}"))
            .materialize()
    }

    /// Builds the solver's input from a generated log, exactly the way the CLI
    /// does: parse the real file, then convert the wire's `g` and `dps` to SI.
    fn input_from(log: &[u8], rate_hz: f64, imu_count: usize) -> CalibrationInput {
        let session = parse::parse(log).expect("generated bytes parse").session;
        let dps_to_rad = std::f64::consts::PI / 180.0;
        let sensors = (0..imu_count)
            .map(|i| {
                let ax = channel(&session, &format!("IMU{i}_AccelX"));
                let ay = channel(&session, &format!("IMU{i}_AccelY"));
                let az = channel(&session, &format!("IMU{i}_AccelZ"));
                let gx = channel(&session, &format!("IMU{i}_GyroX"));
                let gy = channel(&session, &format!("IMU{i}_GyroY"));
                let gz = channel(&session, &format!("IMU{i}_GyroZ"));
                SensorStream {
                    imu_index: i as u8,
                    body: Body::of_imu(i as u8),
                    gyro_rad_s: (0..gx.len())
                        .map(|k| Vector3::new(gx[k], gy[k], gz[k]) * dps_to_rad)
                        .collect(),
                    accel_m_s2: (0..ax.len())
                        .map(|k| Vector3::new(ax[k], ay[k], az[k]) * GRAVITY_M_S2)
                        .collect(),
                }
            })
            .collect();
        CalibrationInput { sample_rate_hz: rate_hz, noise: reference_noise(), sensors, captured_utc: None }
    }

    /// The truth's sensor→body rotation. The generator stores body→sensor, so
    /// the mount the solver reports is its transpose.
    fn truth_mount(truth: &Truth, index: usize) -> Matrix3<f64> {
        let m = truth.sensors[index].rotation_body_to_sensor;
        Matrix3::new(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])
            .transpose()
    }

    /// Geodesic angle between two rotations, degrees.
    fn rotation_error_deg(fitted: &UnitQuaternion<f64>, truth: &Matrix3<f64>) -> f64 {
        let a = fitted.to_rotation_matrix();
        let trace = (a.matrix().transpose() * truth).trace();
        (((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
    }

    fn sensor<'a>(record: &'a CalibrationRecord, imu_index: u8) -> &'a SensorCalibration {
        record.sensors.iter().find(|s| s.imu_index == imu_index).expect("sensor present")
    }

    /// Generates a full three-sensor calibration session with noise on and
    /// fits it. Used by the accuracy tests, which read different fields of the
    /// same fit rather than paying for it five times.
    fn fit_reference_session() -> (CalibrationRecord, Truth) {
        let config = calibration_config(3, SPEC_NOISE_SCALE);
        let out = synth::generate(&config).expect("generated");
        let input = input_from(&out.log, config.imu_rate_hz as f64, config.imu_count as usize);

        let record = calibrate(&input).expect("a full protocol session calibrates");

        (record, out.truth)
    }

    #[test]
    fn full_protocol_session_passes_every_excitation_gate() {
        // Arrange + Act
        let (record, _) = fit_reference_session();

        // Assert — the generated protocol is required to clear §3 outright, so
        // a shortfall here means the generator, not the solver, has drifted.
        assert!(record.quality.shortfalls.is_empty(), "{:?}", record.quality.shortfalls);
        assert_eq!(record.source, CalibrationSource::RigidBody);
        assert!(record.quality.rest_duration_s >= MIN_REST_S);
        assert!(record.quality.datum_duration_s >= MIN_DATUM_S);
    }

    #[test]
    fn fitted_mounts_are_within_half_a_degree_of_the_generated_truth() {
        // Arrange
        let (record, truth) = fit_reference_session();

        // Act — the reference sensor's mount is the §1.2a gauge; the other two
        // are fitted, the fork one through the bars-straight datum (§2.2).
        let errors: Vec<f64> = [1u8, 2]
            .iter()
            .map(|&i| rotation_error_deg(&sensor(&record, i).mount, &truth_mount(&truth, i as usize)))
            .collect();

        // Assert
        assert_eq!(sensor(&record, 0).mount_origin, MountOrigin::Gauge);
        assert_eq!(sensor(&record, 1).mount_origin, MountOrigin::Measured);
        for (i, error) in errors.iter().enumerate() {
            assert!(*error < ROTATION_TOLERANCE_DEG, "IMU{} mount error {error}°", i + 1);
        }
    }

    #[test]
    fn fitted_lever_arm_is_within_ten_millimetres_per_axis() {
        // Arrange
        let (record, truth) = fit_reference_session();
        let expected = truth.sensors[2].lever_arm_m;

        // Act — only IMU2 has an observable lever: IMU0 is the rear body's
        // origin and IMU1 is the front body's (§1.2), and §2.5.3 rules the
        // cross-body arm out entirely.
        let fitted = sensor(&record, 2).lever_m.expect("IMU2 lever fitted");

        // Assert
        for axis in 0..3 {
            let error = (fitted[axis] - expected[axis]).abs();
            assert!(error < LEVER_TOLERANCE_M, "lever axis {axis} error {:.1} mm", error * 1000.0);
        }
        assert!(sensor(&record, 0).lever_m.is_none(), "the body origin has no measured lever");
        assert!(sensor(&record, 1).lever_m.is_none(), "the front body origin has no measured lever");
    }

    #[test]
    fn fitted_steering_axis_is_within_one_degree_of_the_generated_hinge() {
        // Arrange
        let (record, truth) = fit_reference_session();
        let hinge = truth.hinge.expect("calibration truth carries a hinge");
        let expected = Vector3::new(hinge.steer_axis_r[0], hinge.steer_axis_r[1], hinge.steer_axis_r[2]);

        // Act
        let fitted = record.steer_axis.expect("steer axis fitted");

        // Assert
        let error_deg = fitted.dot(&expected).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(error_deg < STEER_AXIS_TOLERANCE_DEG, "steer axis error {error_deg}°");
        assert!(record.quality.steer_peak_to_peak_rad.unwrap() > MIN_STEER_PEAK_TO_PEAK_RAD);
    }

    #[test]
    fn fitted_gyro_biases_are_within_two_milliradians_per_second() {
        // Arrange
        let (record, truth) = fit_reference_session();

        // Act + Assert — one per sensor, all from the rest hold (§2.1).
        for i in 0..3u8 {
            let fitted = sensor(&record, i).gyro_bias_rad_s.expect("bias fitted");
            let expected = truth.sensors[i as usize].gyro_bias_rad_s;
            for axis in 0..3 {
                let error = (fitted[axis] - expected[axis]).abs();
                assert!(
                    error < GYRO_BIAS_TOLERANCE_RAD_S,
                    "IMU{i} gyro bias axis {axis} error {error} rad/s"
                );
            }
        }
    }

    #[test]
    fn fitted_accel_bias_difference_is_within_five_centimetres_per_second_squared() {
        // Arrange — §8.3: the individual bias is not observable, only
        // β = R_2 b_a,2 − b_a,0, so that is what the truth is reduced to.
        let (record, truth) = fit_reference_session();
        let b2 = Vector3::from(truth.sensors[2].accel_bias_m_s2);
        let b0 = Vector3::from(truth.sensors[0].accel_bias_m_s2);
        let expected = truth_mount(&truth, 2) * b2 - b0;

        // Act
        let difference = &record.accel_bias_differences;

        // Assert
        assert_eq!(difference.len(), 1);
        assert_eq!(difference[0].from_imu, 2);
        assert_eq!(difference[0].to_imu, 0);
        for axis in 0..3 {
            let error = (difference[0].value_m_s2[axis] - expected[axis]).abs();
            assert!(
                error < ACCEL_BIAS_DIFFERENCE_TOLERANCE_M_S2,
                "beta axis {axis} error {error} m/s²"
            );
        }
    }

    #[test]
    fn rear_body_gyro_residual_stays_at_the_noise_floor() {
        // Arrange — §7.6's runtime check that IMU2 really is rigid with the
        // frame. On the synthetic body it is, by construction, so the residual
        // must be the per-sample gyro σ and nothing more.
        let (record, _) = fit_reference_session();
        let sigma = synth::GYRO_SIGMA_RAD_S * SPEC_NOISE_SCALE;

        // Act
        let residual = record.quality.rear_body_residual_rad_s.expect("residual reported");

        // Assert — two independent noisy streams differenced, so √2 σ.
        assert!(residual < 1.5 * std::f64::consts::SQRT_2 * sigma, "{residual} rad/s");
    }

    #[test]
    fn noiseless_session_recovers_the_truth_far_inside_every_threshold() {
        // Arrange — with the noise off, only i16 quantisation and the model's
        // own approximations remain, so the errors must collapse.
        let config = calibration_config(3, 0.0);
        let out = synth::generate(&config).expect("generated");
        let input = input_from(&out.log, config.imu_rate_hz as f64, 3);

        // Act
        let record = calibrate(&input).expect("calibrates");

        // Assert — an order of magnitude inside the noisy thresholds.
        let mount_error = rotation_error_deg(&sensor(&record, 1).mount, &truth_mount(&out.truth, 1));
        assert!(mount_error < 0.05, "noiseless mount error {mount_error}°");
        let hinge = out.truth.hinge.expect("hinge");
        let axis_error = record
            .steer_axis
            .unwrap()
            .dot(&Vector3::new(hinge.steer_axis_r[0], hinge.steer_axis_r[1], hinge.steer_axis_r[2]))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(axis_error < 0.1, "noiseless steer axis error {axis_error}°");
    }

    #[test]
    fn hardtail_without_a_rear_pair_fits_rotations_but_reports_no_lever() {
        // Arrange — two sensors: IMU0 on the frame, IMU1 on the fork. §2.3
        // needs a pair on one body, and there is none.
        let config = calibration_config(2, SPEC_NOISE_SCALE);
        let out = synth::generate(&config).expect("generated");
        let input = input_from(&out.log, config.imu_rate_hz as f64, 2);

        // Act
        let record = calibrate(&input).expect("a hardtail still calibrates");

        // Assert
        assert!(record.accel_bias_differences.is_empty());
        assert!(record.sensors.iter().all(|s| s.lever_m.is_none()));
        assert!(record.steer_axis.is_some(), "the hinge is still observable");
        assert!(
            record.quality.shortfalls.contains(&ExcitationShortfall::NoRearPair),
            "{:?}",
            record.quality.shortfalls
        );
    }

    #[test]
    fn a_loop_session_is_refused_with_every_failed_gate_named() {
        // Arrange — the loop protocol is a bike riding a circuit: its body rate
        // never approaches the 2 rad/s a calibration tumble needs, it is never
        // a single-axis-free tumble, and it never holds still. Nothing at all
        // is fittable, which §3 says is the one case that is a typed *error*.
        let config =
            SynthConfig { imu_rate_hz: 100, laps: 1, lap_length_m: 200.0, ..SynthConfig::default() };
        let out = synth::generate(&config).expect("generated");
        let input = input_from(&out.log, config.imu_rate_hz as f64, 3);

        // Act
        let error = calibrate(&input).expect_err("a ride log cannot be a calibration");

        // Assert — the degeneracies are *named*, never silently mis-fitted:
        // in particular the lever-arm gate catches the near-single-axis yaw of
        // a circuit, which is exactly the §2.5.2 degeneracy.
        let CalibrationError::MotionNotRich(shortfalls) = error else {
            panic!("expected MotionNotRich, got {error:?}");
        };
        assert!(
            shortfalls.iter().any(|s| matches!(s, ExcitationShortfall::RateMagnitude { .. })),
            "{shortfalls:?}"
        );
        assert!(
            shortfalls.iter().any(|s| matches!(s, ExcitationShortfall::LeverArmRichness { .. })),
            "{shortfalls:?}"
        );
        assert!(
            shortfalls.iter().any(|s| matches!(s, ExcitationShortfall::RestHold { .. })),
            "{shortfalls:?}"
        );
    }

    #[test]
    fn ragged_streams_are_a_typed_error_not_a_panic() {
        // Arrange
        let mut input = CalibrationInput {
            sample_rate_hz: 833.0,
            noise: reference_noise(),
            captured_utc: None,
            sensors: vec![SensorStream {
                imu_index: 0,
                body: Body::Rear,
                gyro_rad_s: vec![Vector3::zeros(); 10],
                accel_m_s2: vec![Vector3::zeros(); 9],
            }],
        };

        // Act
        let error = calibrate(&input).expect_err("ragged streams are refused");

        // Assert
        assert_eq!(error, CalibrationError::RaggedStreams { imu_index: 0, gyro: 10, accel: 9 });

        // Act — and a non-finite sample names where it is.
        input.sensors[0].accel_m_s2 = vec![Vector3::zeros(); 10];
        input.sensors[0].gyro_rad_s[4] = Vector3::new(0.0, f64::NAN, 0.0);
        let error = calibrate(&input).expect_err("a NaN is refused");

        // Assert
        assert_eq!(error, CalibrationError::NonFiniteSample { imu_index: 0, sample: 4 });
    }

    #[test]
    fn a_session_with_no_rear_sensor_has_no_reference_frame() {
        // Arrange — one fork sensor and nothing on the frame.
        let input = CalibrationInput {
            sample_rate_hz: 833.0,
            noise: reference_noise(),
            captured_utc: None,
            sensors: vec![SensorStream {
                imu_index: 1,
                body: Body::Front,
                gyro_rad_s: vec![Vector3::zeros(); 100],
                accel_m_s2: vec![Vector3::zeros(); 100],
            }],
        };

        // Act
        let error = calibrate(&input).expect_err("no rear sensor is refused");

        // Assert
        assert_eq!(error, CalibrationError::NoReferenceSensor);
    }

    #[test]
    fn every_shortfall_names_what_was_seen_and_what_was_needed() {
        // Arrange — the operator-facing rendering of §3 must never be a bare
        // "calibration failed".
        let shortfall = ExcitationShortfall::SteerRichness { seen: 0.31, needed: 0.5 };

        // Act
        let message = shortfall.to_string();

        // Assert
        assert!(message.contains("18°"), "{message}");
        assert!(message.contains("29°") || message.contains("30°"), "{message}");
    }
}
