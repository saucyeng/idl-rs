//! Synthetic `.idl0` session generator (contract C1 §9, roadmap "Firmware",
//! ruling R187).
//!
//! Simulates a rigid body carrying up to three IMUs around a closed planar
//! loop and emits **a real schema-3 `.idl0` file** — not a `data.parquet`
//! shortcut — so an imported synthetic session exercises the importer, the
//! CAS blob store, the catalog, the parquet cache, lap detection and every
//! chart path exactly as a device recording does. Beside the log it produces
//! a ground-truth JSON: the extrinsics, the lap times, the loop geometry and
//! the noise settings, which is what makes M6.3's rigid-body calibration
//! testable before the hardware exists (the calibration draft's §5 validation
//! plan asks for precisely this) and what lets a lap detector be *scored*
//! rather than smoke-tested.
//!
//! # Pure, and deterministic across platforms
//!
//! [`generate`] takes a [`SynthConfig`] and returns bytes and a [`Truth`]. It
//! reads no clock, no environment variable and no file, and the CLI's
//! `session synth` is the only writer. The same configuration produces
//! byte-identical output on every target and every run, which is what lets
//! consumers commit a generated fixture and diff it: see [`dtrig`] (no libm)
//! and [`rng`] (no system entropy, no logarithm) for how, and C1 §9.7 for
//! why it matters.
//!
//! # What it does not model
//!
//! A rigid body, rigidly attached sensors, a planar loop. No suspension
//! travel, no tyre compliance, no rider, no wheel-speed or pressure channels.
//! Anything that depends on those is not testable against this generator, and
//! a test that pretends otherwise is testing fiction.

pub mod dtrig;
pub mod rng;
pub mod track;
pub mod wire;

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use dtrig::{atan2, round_half_away, sin_cos, PI, TAU};
use rng::{crc32, Pcg32};
use track::Loop;

/// This generator's own version, SemVer 2.0.0. Bump it whenever a change
/// alters the bytes a given [`SynthConfig`] produces; the committed fixture
/// test (§9.8) is what will tell you that you must.
pub const SYNTH_VERSION: &str = "1.0.0";

/// `schema_version` of the truth file (C1 §9.6).
pub const TRUTH_SCHEMA_VERSION: u32 = 1;

/// The header's `Session start UTC` (C1 §9.5): 2026-01-01T00:00:00Z, a fixed
/// literal so the output never depends on when it was generated.
pub const SESSION_START_UTC_MS: i64 = 1_767_225_600_000;

/// The first record's `timestamp_us`. Non-zero on purpose: SPEC §5.6's
/// wall-clock back-fill subtracts the first sample's device timestamp, and a
/// zero would satisfy that arithmetic trivially instead of exercising it.
pub const DEVICE_T0_US: i64 = 4_000_000;

/// Standard gravity, m/s².
pub const GRAVITY_M_S2: f64 = 9.80665;

/// Accelerometer full-scale range, ±g (LSM6DSO32 at its widest setting).
pub const ACCEL_RANGE_G: f64 = 32.0;

/// Gyroscope full-scale range, ±dps.
pub const GYRO_RANGE_DPS: f64 = 2000.0;

/// Gyro noise σ per axis at `--noise 1.0`, rad/s — the calibration draft's
/// `reference_default()` figure, so its acceptance thresholds apply to this
/// generator's output unchanged.
pub const GYRO_SIGMA_RAD_S: f64 = 0.003;

/// Accelerometer noise σ per axis at `--noise 1.0`, m/s². Same provenance as
/// [`GYRO_SIGMA_RAD_S`].
pub const ACCEL_SIGMA_M_S2: f64 = 0.05;

/// Imposed roll amplitude, radians (C1 §9.3).
const ROLL_AMPLITUDE_RAD: f64 = 0.15;
/// Imposed roll frequency, Hz.
const ROLL_FREQ_HZ: f64 = 0.37;
/// Imposed pitch amplitude, radians.
const PITCH_AMPLITUDE_RAD: f64 = 0.08;
/// Imposed pitch frequency, Hz. Deliberately not a rational multiple of
/// [`ROLL_FREQ_HZ`]: equal or harmonically related rates would make the three
/// body-rate components dependent, which is the degeneracy the calibration
/// draft §2.4 says a fit cannot resolve.
const PITCH_FREQ_HZ: f64 = 0.23;
/// Imposed pitch phase offset, radians.
const PITCH_PHASE_RAD: f64 = 1.0;

/// Local projection origin latitude, degrees north (C1 §9.5).
pub const ORIGIN_LAT_DEG: f64 = 51.5;
/// Local projection origin longitude, degrees east.
pub const ORIGIN_LON_DEG: f64 = -1.5;
/// Metres per degree of latitude used by the flat-earth projection.
pub const METRES_PER_DEG_NORTH: f64 = 111_320.0;
/// Constant altitude written into every fix, metres.
const ALTITUDE_M: f64 = 100.0;
/// `fix_quality` written into every fix (1 = GPS, SPEC §5.6).
const FIX_QUALITY: u8 = 1;
/// `satellites` written into every fix.
const SATELLITES: u8 = 12;
/// The speed uncertainty the truth file reports, mm/s — the M10 datasheet's
/// 0.05 m/s velocity accuracy. It is *not* in the log: SPEC §5.6 has no
/// `sAcc` field (C1 §9.5).
const SPEED_ACCURACY_MM_S: f64 = 50.0;

/// Everything a caller can choose. Every field has a default, and the
/// defaults are what `idl-rs session synth --out <file>` alone produces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SynthConfig {
    /// Complete circuits of the loop. Lap 1 begins at `t = 0`.
    pub laps: u32,
    /// Loop perimeter, metres. The loop is scaled to hit it exactly.
    pub lap_length_m: f64,
    /// IMU output data rate, Hz — every IMU, every axis.
    pub imu_rate_hz: u16,
    /// GPS fix rate, Hz.
    pub gps_rate_hz: u8,
    /// PRNG seed. Same seed plus same flags gives byte-identical output.
    pub seed: u64,
    /// Multiplies every noise σ together. `0.0` gives a noiseless recording.
    pub noise_scale: f64,
    /// How many of the three sensors to instrument, in [`SENSORS`] order.
    pub imu_count: u8,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig {
            laps: 3,
            lap_length_m: 400.0,
            imu_rate_hz: 800,
            gps_rate_hz: 5,
            seed: 1,
            noise_scale: 1.0,
            imu_count: 3,
        }
    }
}

impl SynthConfig {
    /// The configuration behind the committed fixture (C1 §9.8): three laps
    /// of a 120 m loop with all three sensors at 25 Hz, which lands under
    /// 200 KB. Small enough to commit, useless for anything spectral,
    /// adequate for lap detection, importer round-trips and catalog work.
    pub fn fixture() -> SynthConfig {
        SynthConfig { laps: 3, lap_length_m: 120.0, imu_rate_hz: 25, ..SynthConfig::default() }
    }

    /// Rejects a configuration this generator cannot honour. Bounds are
    /// generous — they exist to turn a typo into a typed error rather than a
    /// multi-gigabyte allocation or a divide by zero.
    pub fn validate(&self) -> Result<(), SynthError> {
        if self.laps == 0 || self.laps > 1000 {
            return Err(SynthError::LapsOutOfRange(self.laps));
        }
        if !self.lap_length_m.is_finite() || self.lap_length_m < 10.0 || self.lap_length_m > 100_000.0
        {
            return Err(SynthError::LapLengthOutOfRange(self.lap_length_m));
        }
        if self.imu_rate_hz == 0 || self.imu_rate_hz > 6664 {
            return Err(SynthError::ImuRateOutOfRange(self.imu_rate_hz));
        }
        if self.gps_rate_hz == 0 || self.gps_rate_hz > 20 {
            return Err(SynthError::GpsRateOutOfRange(self.gps_rate_hz));
        }
        if self.imu_count == 0 || self.imu_count as usize > SENSORS.len() {
            return Err(SynthError::ImuCountOutOfRange(self.imu_count));
        }
        if !self.noise_scale.is_finite() || self.noise_scale < 0.0 || self.noise_scale > 1000.0 {
            return Err(SynthError::NoiseOutOfRange(self.noise_scale));
        }
        Ok(())
    }
}

/// Why a configuration was refused, or why generation could not proceed.
/// Typed, never a bare string (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq)]
pub enum SynthError {
    /// `laps` outside 1..=1000.
    LapsOutOfRange(u32),
    /// `lap_length_m` outside 10..=100000, or not finite.
    LapLengthOutOfRange(f64),
    /// `imu_rate_hz` outside 1..=6664.
    ImuRateOutOfRange(u16),
    /// `gps_rate_hz` outside 1..=20.
    GpsRateOutOfRange(u8),
    /// `imu_count` outside 1..=3.
    ImuCountOutOfRange(u8),
    /// `noise_scale` outside 0..=1000, or not finite.
    NoiseOutOfRange(f64),
    /// The configuration yields fewer than three IMU samples, which is too
    /// few for the centred difference that produces angular acceleration.
    TooFewSamples(usize),
}

impl fmt::Display for SynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthError::LapsOutOfRange(v) => write!(f, "laps must be 1..=1000, got {v}"),
            SynthError::LapLengthOutOfRange(v) => {
                write!(f, "lap-length-m must be a finite 10..=100000, got {v}")
            }
            SynthError::ImuRateOutOfRange(v) => write!(f, "rate-hz must be 1..=6664, got {v}"),
            SynthError::GpsRateOutOfRange(v) => write!(f, "gps-hz must be 1..=20, got {v}"),
            SynthError::ImuCountOutOfRange(v) => write!(f, "imu-count must be 1..=3, got {v}"),
            SynthError::NoiseOutOfRange(v) => {
                write!(f, "noise must be a finite 0..=1000, got {v}")
            }
            SynthError::TooFewSamples(n) => write!(
                f,
                "configuration yields {n} IMU sample(s); at least 3 are needed for the angular-acceleration difference"
            ),
        }
    }
}

impl std::error::Error for SynthError {}

/// One sensor's fixed extrinsics and biases (C1 §9.4). Sensor 0 *is* the body
/// frame, so a fit that recovers sensors 1 and 2 is recovering a relative
/// rotation whose answer is known.
#[derive(Debug, Clone, Copy)]
pub struct SensorSpec {
    /// `frame`, `fork` or `rear`.
    pub role: &'static str,
    /// Body→sensor rotation as ZYX Euler angles in degrees: yaw, pitch, roll.
    pub euler_zyx_deg: [f64; 3],
    /// Sensor position in the body frame, metres (X forward, Y left, Z up).
    pub lever_arm_m: [f64; 3],
    /// Constant gyro bias in the sensor frame, rad/s.
    pub gyro_bias_rad_s: [f64; 3],
    /// Constant accelerometer bias in the sensor frame, m/s².
    pub accel_bias_m_s2: [f64; 3],
}

/// The three sensors, in the order `--imu-count` takes them (C1 §9.4).
pub const SENSORS: [SensorSpec; 3] = [
    SensorSpec {
        role: "frame",
        euler_zyx_deg: [0.0, 0.0, 0.0],
        lever_arm_m: [0.0, 0.0, 0.0],
        gyro_bias_rad_s: [0.0010, -0.0015, 0.0007],
        accel_bias_m_s2: [0.020, -0.035, 0.015],
    },
    SensorSpec {
        role: "fork",
        euler_zyx_deg: [5.0, -12.0, 3.0],
        lever_arm_m: [0.640, 0.000, -0.180],
        gyro_bias_rad_s: [-0.0022, 0.0009, 0.0014],
        accel_bias_m_s2: [-0.041, 0.012, 0.028],
    },
    SensorSpec {
        role: "rear",
        euler_zyx_deg: [-7.0, 4.0, -9.0],
        lever_arm_m: [-0.430, 0.020, -0.260],
        gyro_bias_rad_s: [0.0005, 0.0021, -0.0011],
        accel_bias_m_s2: [0.033, 0.026, -0.019],
    },
];

// ---------------------------------------------------------------------------
// Small 3-vector / 3×3 helpers.
//
// Plain arrays rather than `nalgebra`: the determinism argument in C1 §9.7 is
// "only IEEE-exact operations, in a fixed order", and that is easiest to hold
// true — and to read — when the operation order is written out here.
// ---------------------------------------------------------------------------

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

fn mat_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `Rx(roll) · Ry(pitch) · Rz(yaw)` — the ZYX Euler rotation that takes a
/// vector from the reference frame into the rotated frame.
pub fn rotation_zyx(yaw: f64, pitch: f64, roll: f64) -> Mat3 {
    let (sy, cy) = sin_cos(yaw);
    let (sp, cp) = sin_cos(pitch);
    let (sr, cr) = sin_cos(roll);
    [
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
    ]
}

/// Body-frame angular velocity from ZYX Euler angles and their rates.
fn body_rates(yaw_rate: f64, pitch: f64, pitch_rate: f64, roll: f64, roll_rate: f64) -> Vec3 {
    let (sp, cp) = sin_cos(pitch);
    let (sr, cr) = sin_cos(roll);
    [
        roll_rate - yaw_rate * sp,
        pitch_rate * cr + yaw_rate * cp * sr,
        -pitch_rate * sr + yaw_rate * cp * cr,
    ]
}

/// Centred difference of a vector series on a uniform grid, one-sided at the
/// two ends. `dt` must be positive.
fn differentiate(series: &[Vec3], dt: f64) -> Vec<Vec3> {
    let n = series.len();
    let mut out = vec![[0.0; 3]; n];
    for k in 0..n {
        let (lo, hi, span) = if k == 0 {
            (0, 1, dt)
        } else if k == n - 1 {
            (n - 2, n - 1, dt)
        } else {
            (k - 1, k + 1, 2.0 * dt)
        };
        for axis in 0..3 {
            out[k][axis] = (series[hi][axis] - series[lo][axis]) / span;
        }
    }
    out
}

/// Centred difference of a scalar series, one-sided at the two ends.
fn differentiate_scalar(series: &[f64], dt: f64) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![0.0; n];
    for k in 0..n {
        let (lo, hi, span) = if k == 0 {
            (0, 1, dt)
        } else if k == n - 1 {
            (n - 2, n - 1, dt)
        } else {
            (k - 1, k + 1, 2.0 * dt)
        };
        out[k] = (series[hi] - series[lo]) / span;
    }
    out
}

/// Rounds to the nearest `i16`, saturating. Returns the value and whether it
/// saturated — a clipped sample is a real outcome the truth file counts, not
/// an error (C1 §9.4).
fn quantise(value: f64, scale: f64) -> (i16, bool) {
    let counts = round_half_away(value / scale);
    if counts > 32767.0 {
        (32767, true)
    } else if counts < -32768.0 {
        (-32768, true)
    } else {
        (counts as i16, false)
    }
}

// ---------------------------------------------------------------------------
// The truth file (C1 §9.6).
// ---------------------------------------------------------------------------

/// Ground truth written beside the log. A test fixture, not a contract any
/// app code reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    /// [`TRUTH_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// [`SYNTH_VERSION`].
    pub generator_version: String,
    /// Every flag's resolved value, defaults included.
    pub config: SynthConfig,
    pub session: TruthSession,
    #[serde(rename = "loop")]
    pub loop_: TruthLoop,
    /// One entry per lap, by construction rather than by detection.
    pub laps: Vec<TruthLap>,
    pub sensors: Vec<TruthSensor>,
    pub noise: TruthNoise,
    pub gps: TruthGps,
}

/// Identity and shape of the generated session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthSession {
    /// 32 lowercase hex characters.
    pub session_id: String,
    /// 12 lowercase hex characters.
    pub device_id: String,
    /// [`SESSION_START_UTC_MS`].
    pub start_utc_ms: i64,
    /// [`DEVICE_T0_US`].
    pub device_t0_us: i64,
    pub imu_rate_hz: u16,
    pub gps_rate_hz: u8,
    /// Samples per IMU, not in total.
    pub imu_sample_count: usize,
    pub gps_fix_count: usize,
    /// Recording length, seconds.
    pub duration_s: f64,
}

/// Loop geometry and the projection that places it on the globe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthLoop {
    pub semi_major_m: f64,
    pub semi_minor_m: f64,
    pub perimeter_m: f64,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub metres_per_deg_north: f64,
    pub metres_per_deg_east: f64,
    pub gate: TruthGate,
}

/// The start/finish gate: a point and the unit normal of the line through it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthGate {
    pub east_m: f64,
    pub north_m: f64,
    pub normal_east: f64,
    pub normal_north: f64,
}

/// One lap, exact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthLap {
    /// 1-based.
    pub index: u32,
    pub start_s: f64,
    pub end_s: f64,
    pub start_utc_ms: i64,
    pub end_utc_ms: i64,
    pub duration_s: f64,
    pub distance_m: f64,
}

/// One sensor's extrinsics as generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthSensor {
    /// Matches the `imu_index` in the file's `IMU_SAMPLE` records.
    pub index: u8,
    pub role: String,
    /// Yaw, pitch, roll in degrees.
    pub euler_zyx_deg: [f64; 3],
    /// The same rotation as a matrix, row-major; rotates a body-frame vector
    /// into the sensor frame.
    pub rotation_body_to_sensor: [[f64; 3]; 3],
    pub lever_arm_m: [f64; 3],
    pub gyro_bias_rad_s: [f64; 3],
    pub accel_bias_m_s2: [f64; 3],
    /// g per LSB, matching the channel registry entry.
    pub accel_scale_g_per_lsb: f64,
    /// dps per LSB, matching the channel registry entry.
    pub gyro_scale_dps_per_lsb: f64,
    /// Axis values that saturated at the `i16` rail.
    pub clipped_sample_count: usize,
}

/// The noise actually applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthNoise {
    pub scale: f64,
    pub gyro_sigma_rad_s: f64,
    pub accel_sigma_m_s2: f64,
    /// Always `irwin-hall-12` — an approximation to a normal, named so no
    /// consumer assumes Gaussian tails (C1 §9.7).
    pub distribution: String,
}

/// What the GPS records do and do not carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthGps {
    /// The per-fix speed uncertainty the M10 would report. Not in the log —
    /// SPEC §5.6 has no field for it (C1 §9.5).
    pub speed_accuracy_mm_s: f64,
    /// Exact distance travelled over the whole recording, metres.
    pub total_distance_m: f64,
    /// The `docs/HARDWARE_M10_SETUP.md` §3 fields SPEC §5.6 cannot carry.
    pub fields_omitted: Vec<String>,
}

/// What [`generate`] produces: the `.idl0` bytes and the ground truth.
#[derive(Debug, Clone)]
pub struct SynthOutput {
    /// The complete `.idl0` file.
    pub log: Vec<u8>,
    /// The ground truth for [`SynthOutput::log`].
    pub truth: Truth,
}

/// Lowercase hex of a byte slice.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The canonical bytes the session identity and the header CRC are taken
/// over: the configuration and the generator version, compact JSON.
///
/// `serde_json`'s map is ordered, so this string is stable for a given
/// configuration; including [`SYNTH_VERSION`] means a generator change gives
/// the session a new identity rather than silently reusing the old one.
///
/// # Panics
///
/// Never in practice. Serialising a [`SynthConfig`] fails only on a non-finite
/// `f64`, and the only two `f64` fields are rejected by
/// [`SynthConfig::validate`], which every caller runs first.
fn identity_bytes(config: &SynthConfig) -> Vec<u8> {
    let value = serde_json::json!({
        "generator": "idl-rs session synth",
        "generator_version": SYNTH_VERSION,
        "config": config,
    });
    serde_json::to_string(&value).expect("SynthConfig serialises").into_bytes()
}

/// Generates one synthetic session.
///
/// Pure: no clock, no filesystem, no environment. The same `config` gives
/// byte-identical `log` bytes on every platform and every run (C1 §9.7).
///
/// # Errors
///
/// [`SynthError`] when the configuration is out of range, or when it would
/// yield fewer than three IMU samples.
pub fn generate(config: &SynthConfig) -> Result<SynthOutput, SynthError> {
    config.validate()?;

    let identity = identity_bytes(config);
    let digest = Sha256::digest(&identity);
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&digest[0..16]);
    let mut device_id = [0u8; 6];
    device_id.copy_from_slice(&digest[16..22]);
    let config_crc = crc32(&identity);

    let lp = Loop::build(config.lap_length_m);
    let duration_s = lp.lap_time_s * config.laps as f64;

    let imu_dt = 1.0 / config.imu_rate_hz as f64;
    let imu_sample_count = (duration_s * config.imu_rate_hz as f64).floor() as usize;
    if imu_sample_count < 3 {
        return Err(SynthError::TooFewSamples(imu_sample_count));
    }
    let gps_fix_count = (duration_s * config.gps_rate_hz as f64).floor() as usize;

    let sensor_count = config.imu_count as usize;
    let accel_scale_g = ACCEL_RANGE_G / 32768.0;
    let gyro_scale_dps = GYRO_RANGE_DPS / 32768.0;

    // ── body kinematics, sample by sample ────────────────────────────────
    let mut speeds = Vec::with_capacity(imu_sample_count);
    let mut headings = Vec::with_capacity(imu_sample_count);
    let mut yaw_rates = Vec::with_capacity(imu_sample_count);
    let mut omegas: Vec<Vec3> = Vec::with_capacity(imu_sample_count);
    let mut attitudes: Vec<(f64, f64)> = Vec::with_capacity(imu_sample_count);

    for k in 0..imu_sample_count {
        let t = k as f64 * imu_dt;
        let state = lp.state_at(lap_local_time(t, lp.lap_time_s, config.laps));

        let (roll, roll_rate) = imposed_angle(ROLL_AMPLITUDE_RAD, ROLL_FREQ_HZ, 0.0, t);
        let (pitch, pitch_rate) =
            imposed_angle(PITCH_AMPLITUDE_RAD, PITCH_FREQ_HZ, PITCH_PHASE_RAD, t);

        speeds.push(state.speed_m_s);
        headings.push(state.heading_rad);
        yaw_rates.push(state.yaw_rate_rad_s);
        attitudes.push((roll, pitch));
        omegas.push(body_rates(state.yaw_rate_rad_s, pitch, pitch_rate, roll, roll_rate));
    }

    let tangential = differentiate_scalar(&speeds, imu_dt);
    let omega_dots = differentiate(&omegas, imu_dt);

    // Specific force at the body origin, in the body frame (C1 §9.4).
    let mut specific_forces: Vec<Vec3> = Vec::with_capacity(imu_sample_count);
    for k in 0..imu_sample_count {
        let (sy, cy) = sin_cos(headings[k]);
        let lateral = speeds[k] * yaw_rates[k];
        let world = [
            tangential[k] * cy - lateral * sy,
            tangential[k] * sy + lateral * cy,
            GRAVITY_M_S2,
        ];
        let (roll, pitch) = attitudes[k];
        let r_wb = rotation_zyx(headings[k], pitch, roll);
        specific_forces.push(mat_vec(&r_wb, &world));
    }

    // ── sensor rotations, resolved once ──────────────────────────────────
    let deg = PI / 180.0;
    let rotations: Vec<Mat3> = SENSORS[..sensor_count]
        .iter()
        .map(|s| rotation_zyx(s.euler_zyx_deg[0] * deg, s.euler_zyx_deg[1] * deg, s.euler_zyx_deg[2] * deg))
        .collect();

    // ── emission ─────────────────────────────────────────────────────────
    let mut registry = Vec::with_capacity(sensor_count * 6);
    for i in 0..sensor_count {
        for (axis, (name, scale, units)) in axis_registry_specs(i, accel_scale_g, gyro_scale_dps)
            .into_iter()
            .enumerate()
        {
            registry.push(wire::registry_entry(
                (i * 6 + axis) as u8,
                wire::DATA_TYPE_I16,
                config.imu_rate_hz,
                scale as f32,
                0.0,
                &name,
                units,
            ));
        }
    }

    let header = wire::Header {
        uuid,
        device_id,
        session_start_ms: SESSION_START_UTC_MS,
        config_crc,
        imu_mask: (1u32 << (sensor_count * 6)) - 1,
        imu_count: config.imu_count,
        imu_sample_rate_hz: config.imu_rate_hz,
        gps_sample_rate_hz: config.gps_rate_hz,
    };
    let mut log = wire::write_header(&header, &registry);

    let mut prng = Pcg32::new(config.seed);
    let mut clipped = vec![0usize; sensor_count];
    let gyro_sigma = GYRO_SIGMA_RAD_S * config.noise_scale;
    let accel_sigma = ACCEL_SIGMA_M_S2 * config.noise_scale;
    let rad_to_dps = 180.0 / PI;

    let mut next_fix = 0usize;
    for k in 0..imu_sample_count {
        let imu_ts_us = DEVICE_T0_US + round_half_away(k as f64 * 1e6 * imu_dt) as i64;

        // Fixes due at or before this sample go first, so the file's
        // timestamps are non-decreasing (C1 §9.5).
        while next_fix < gps_fix_count {
            let fix_t = next_fix as f64 / config.gps_rate_hz as f64;
            let fix_ts_us = DEVICE_T0_US + round_half_away(fix_t * 1e6) as i64;
            if fix_ts_us > imu_ts_us {
                break;
            }
            push_fix(&mut log, &lp, config, fix_t, fix_ts_us);
            next_fix += 1;
        }

        for (i, rotation) in rotations.iter().enumerate() {
            let spec = &SENSORS[i];
            let omega = omegas[k];
            let lever = spec.lever_arm_m;
            let euler = cross(&omega_dots[k], &lever);
            let centripetal = cross(&omega, &cross(&omega, &lever));
            let at_sensor = add(&add(&specific_forces[k], &euler), &centripetal);

            let gyro_body = mat_vec(rotation, &omega);
            let accel_body = mat_vec(rotation, &at_sensor);

            let mut axes = [0i16; 6];
            for axis in 0..3 {
                let value =
                    accel_body[axis] + spec.accel_bias_m_s2[axis] + accel_sigma * prng.next_gaussian();
                let (counts, clip) = quantise(value / GRAVITY_M_S2, accel_scale_g);
                axes[axis] = counts;
                if clip {
                    clipped[i] += 1;
                }
            }
            for axis in 0..3 {
                let value =
                    gyro_body[axis] + spec.gyro_bias_rad_s[axis] + gyro_sigma * prng.next_gaussian();
                let (counts, clip) = quantise(value * rad_to_dps, gyro_scale_dps);
                axes[3 + axis] = counts;
                if clip {
                    clipped[i] += 1;
                }
            }

            wire::push_record(&mut log, wire::IMU_SAMPLE, &wire::imu_payload(i as u8, imu_ts_us, axes));
        }
    }

    while next_fix < gps_fix_count {
        let fix_t = next_fix as f64 / config.gps_rate_hz as f64;
        let fix_ts_us = DEVICE_T0_US + round_half_away(fix_t * 1e6) as i64;
        push_fix(&mut log, &lp, config, fix_t, fix_ts_us);
        next_fix += 1;
    }

    wire::push_record(&mut log, wire::SESSION_END, &[]);

    // ── truth ────────────────────────────────────────────────────────────
    let (gate_e, gate_n, gate_ne, gate_nn) = lp.gate();
    let laps = (1..=config.laps)
        .map(|index| {
            let start_s = (index - 1) as f64 * lp.lap_time_s;
            let end_s = index as f64 * lp.lap_time_s;
            TruthLap {
                index,
                start_s,
                end_s,
                start_utc_ms: SESSION_START_UTC_MS + round_half_away(start_s * 1000.0) as i64,
                end_utc_ms: SESSION_START_UTC_MS + round_half_away(end_s * 1000.0) as i64,
                duration_s: lp.lap_time_s,
                distance_m: lp.perimeter_m,
            }
        })
        .collect();

    let sensors = (0..sensor_count)
        .map(|i| TruthSensor {
            index: i as u8,
            role: SENSORS[i].role.to_string(),
            euler_zyx_deg: SENSORS[i].euler_zyx_deg,
            rotation_body_to_sensor: rotations[i],
            lever_arm_m: SENSORS[i].lever_arm_m,
            gyro_bias_rad_s: SENSORS[i].gyro_bias_rad_s,
            accel_bias_m_s2: SENSORS[i].accel_bias_m_s2,
            accel_scale_g_per_lsb: accel_scale_g,
            gyro_scale_dps_per_lsb: gyro_scale_dps,
            clipped_sample_count: clipped[i],
        })
        .collect();

    let truth = Truth {
        schema_version: TRUTH_SCHEMA_VERSION,
        generator_version: SYNTH_VERSION.to_string(),
        config: *config,
        session: TruthSession {
            session_id: hex(&uuid),
            device_id: hex(&device_id),
            start_utc_ms: SESSION_START_UTC_MS,
            device_t0_us: DEVICE_T0_US,
            imu_rate_hz: config.imu_rate_hz,
            gps_rate_hz: config.gps_rate_hz,
            imu_sample_count,
            gps_fix_count,
            duration_s,
        },
        loop_: TruthLoop {
            semi_major_m: lp.semi_major_m,
            semi_minor_m: lp.semi_minor_m,
            perimeter_m: lp.perimeter_m,
            origin_lat_deg: ORIGIN_LAT_DEG,
            origin_lon_deg: ORIGIN_LON_DEG,
            metres_per_deg_north: METRES_PER_DEG_NORTH,
            metres_per_deg_east: metres_per_deg_east(),
            gate: TruthGate {
                east_m: gate_e,
                north_m: gate_n,
                normal_east: gate_ne,
                normal_north: gate_nn,
            },
        },
        laps,
        sensors,
        noise: TruthNoise {
            scale: config.noise_scale,
            gyro_sigma_rad_s: gyro_sigma,
            accel_sigma_m_s2: accel_sigma,
            distribution: "irwin-hall-12".to_string(),
        },
        gps: TruthGps {
            speed_accuracy_mm_s: SPEED_ACCURACY_MM_S,
            total_distance_m: lp.perimeter_m * config.laps as f64,
            fields_omitted: ["sAcc", "velD", "odo_distance", "odo_distance_std"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        },
    };

    Ok(SynthOutput { log, truth })
}

/// Metres per degree of longitude at the projection origin. Computed through
/// [`dtrig`], not `f64::cos`, so it is the same constant on every platform.
pub fn metres_per_deg_east() -> f64 {
    let (_, c) = sin_cos(ORIGIN_LAT_DEG * PI / 180.0);
    METRES_PER_DEG_NORTH * c
}

/// Elapsed time within the current lap, clamped so floating-point drift at
/// the very end of the recording cannot index a lap that does not exist.
fn lap_local_time(t: f64, lap_time_s: f64, laps: u32) -> f64 {
    let mut lap = (t / lap_time_s).floor();
    if lap > (laps - 1) as f64 {
        lap = (laps - 1) as f64;
    }
    if lap < 0.0 {
        lap = 0.0;
    }
    t - lap * lap_time_s
}

/// An imposed sinusoidal attitude angle and its rate (C1 §9.3).
fn imposed_angle(amplitude: f64, freq_hz: f64, phase: f64, t: f64) -> (f64, f64) {
    let w = TAU * freq_hz;
    let (s, c) = sin_cos(w * t + phase);
    (amplitude * s, amplitude * w * c)
}

/// The six registry names, scales and units for IMU `index`, in SPEC §5.2's
/// canonical axis order.
fn axis_registry_specs(
    index: usize,
    accel_scale_g: f64,
    gyro_scale_dps: f64,
) -> Vec<(String, f64, &'static str)> {
    vec![
        (format!("IMU{index}_AccelX"), accel_scale_g, "g"),
        (format!("IMU{index}_AccelY"), accel_scale_g, "g"),
        (format!("IMU{index}_AccelZ"), accel_scale_g, "g"),
        (format!("IMU{index}_GyroX"), gyro_scale_dps, "dps"),
        (format!("IMU{index}_GyroY"), gyro_scale_dps, "dps"),
        (format!("IMU{index}_GyroZ"), gyro_scale_dps, "dps"),
    ]
}

/// Appends one `GPS_FIX` record for session time `fix_t` (C1 §9.5).
fn push_fix(log: &mut Vec<u8>, lp: &Loop, config: &SynthConfig, fix_t: f64, fix_ts_us: i64) {
    let state = lp.state_at(lap_local_time(fix_t, lp.lap_time_s, config.laps));

    let lat_deg = ORIGIN_LAT_DEG + state.north_m / METRES_PER_DEG_NORTH;
    let lon_deg = ORIGIN_LON_DEG + state.east_m / metres_per_deg_east();

    // Compass heading: degrees clockwise from north, where `heading_rad` is
    // radians anticlockwise from east.
    let mut compass_deg = 90.0 - state.heading_rad * 180.0 / PI;
    while compass_deg < 0.0 {
        compass_deg += 360.0;
    }
    while compass_deg >= 360.0 {
        compass_deg -= 360.0;
    }

    let payload = wire::gps_payload(
        SESSION_START_UTC_MS + round_half_away(fix_t * 1000.0) as i64,
        fix_ts_us,
        round_half_away(lat_deg * 1e7) as i32,
        round_half_away(lon_deg * 1e7) as i32,
        round_half_away(ALTITUDE_M * 10.0) as i16,
        round_half_away(state.speed_m_s * 3.6 * 100.0) as u16,
        round_half_away(compass_deg * 100.0) as u16,
        FIX_QUALITY,
        SATELLITES,
    );
    wire::push_record(log, wire::GPS_FIX, &payload);
}

/// Angle between two rotation matrices, radians — the geodesic distance on
/// SO(3), from `trace(A·Bᵀ) = 1 + 2cos θ`. Exposed because it is what a
/// calibration test compares its fitted rotation against the truth with.
pub fn rotation_angle_between(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut trace = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            trace += a[i][j] * b[i][j];
        }
    }
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    // atan2 of the sine and cosine rather than an arccos: `dtrig` has no
    // `acos`, and this identity needs only what it does have.
    atan2((1.0 - cos_theta * cos_theta).max(0.0).sqrt(), cos_theta)
}

#[cfg(test)]
mod tests;
