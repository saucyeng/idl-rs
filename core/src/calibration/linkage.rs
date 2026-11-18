//! A bike's rear linkage as clicked on a side photo (`ml/tools/linkage.html`),
//! and the physics estimator's fitted geometry for one calibration.
//!
//! The linkage is bike-level data (it does not change with a recalibration);
//! the library keeps it in the bike's file beside the calibrations. The
//! physics fit belongs to a calibration (it is fitted with that calibration's
//! mounts and lever arms), so it is stored inside the calibration entry.

use serde::{Deserialize, Serialize};

/// Points of a planar 4-bar rear end (Horst link) in the bike's side plane:
/// origin at the rear axle, x toward the front axle, y up, metres. Topped out
/// (as photographed, unweighted).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkagePoints {
    /// Front axle.
    pub front_axle: [f64; 2],
    /// Rear axle (the origin, [0, 0]).
    pub rear_axle: [f64; 2],
    /// Chainstay's pivot on the frame.
    pub main_pivot: [f64; 2],
    /// Chainstay-seatstay (Horst) pivot.
    pub horst: [f64; 2],
    /// Seatstay-rocker pivot.
    pub ss_rocker: [f64; 2],
    /// Rocker's pivot on the frame.
    pub rocker_frame: [f64; 2],
    /// Shock's frame eye.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shock_frame: Option<[f64; 2]>,
    /// Shock's rocker eye.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shock_rocker: Option<[f64; 2]>,
    /// The frame pod (IMU0).
    pub imu0: [f64; 2],
    /// The fork pod, as a check on the calibration's lever arm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_pod: Option<[f64; 2]>,
    /// The rear pod, as a check on the calibration's lever arm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rear_pod: Option<[f64; 2]>,
}

/// A bike's rear linkage from the photo tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Linkage {
    /// When this record last changed, UTC ms; the newer copy wins a merge.
    pub updated_at_ms: i64,
    /// The points, metres (scaled from the photo by the wheelbase).
    pub points_m: LinkagePoints,
    /// Wheelbase used to scale the photo, metres.
    pub wheelbase_m: f64,
    /// The rear wheel's travel, metres.
    pub rear_travel_m: f64,
    /// The fork's travel, metres (not a linkage dimension, but the other end
    /// stop the estimator needs). Absent: given on the command line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub front_travel_m: Option<f64>,
    /// The shock's specified eye-to-eye length, metres. With the shock's
    /// drawn length it gives the photo's scale error, the prior on
    /// [`PhysicsFit::link_scale`]. Absent: prior centred on 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shock_eye_to_eye_m: Option<f64>,
    /// The photo's file name, for reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image: Option<String>,
}

impl Linkage {
    /// The shock's drawn eye-to-eye length, metres, when both eyes were clicked.
    pub fn drawn_shock_m(&self) -> Option<f64> {
        let (a, b) = (self.points_m.shock_frame?, self.points_m.shock_rocker?);
        Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
    }

    /// The prior centre for the photo's scale: specified / drawn shock length, or 1.
    pub fn scale_prior(&self) -> f64 {
        match (self.shock_eye_to_eye_m, self.drawn_shock_m()) {
            (Some(spec), Some(drawn)) if drawn > 0.0 => spec / drawn,
            _ => 1.0,
        }
    }
}

/// Version of the physics fit. Bump when the same recordings would give a
/// different fit (a model change in `estimate::physics`).
pub const PHYSICS_FIT_VERSION: u32 = 1;

/// The bike geometry the physics estimator fits from riding, for one
/// calibration (IMU0 axes, SI units).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsFit {
    /// [`PHYSICS_FIT_VERSION`] at the time of the fit.
    pub version: u32,
    /// When it was fitted, UTC ms.
    pub fitted_at_ms: i64,
    /// A point on the steering axis, metres.
    pub steer_point: [f64; 3],
    /// Crown-to-fork-pod length at top-out, metres.
    pub bend_len_m: f64,
    /// Scale of the linkage photo (1 = as drawn). 1 without a linkage.
    pub link_scale: f64,
    /// Session the windows came from.
    pub session_id: String,
    /// Fit windows, session seconds `[start, end]`.
    pub windows_s: Vec<[f64; 2]>,
    /// Gauss-Newton iterations run.
    pub iterations: u32,
    /// Whether the last iteration gained less than the tolerance.
    pub converged: bool,
}
