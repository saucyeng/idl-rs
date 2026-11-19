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

/// Bike side-plane point, metres: x toward the front axle, y up.
pub type P2 = [f64; 2];

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn norm(a: P2) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}
fn rot2(v: P2, t: f64) -> P2 {
    let (s, c) = t.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

/// The intersection of circle (`c0`, `r0`) and (`c1`, `r1`) on the `sign`
/// side (+1 or -1) of the line from `c0` to `c1`. Metres.
pub fn intersect(c0: P2, r0: f64, c1: P2, r1: f64, sign: f64) -> P2 {
    let d = sub(c1, c0);
    let dn = norm(d);
    let a = (r0 * r0 - r1 * r1 + dn * dn) / (2.0 * dn);
    let hh = (r0 * r0 - a * a).max(0.0).sqrt();
    let u = [d[0] / dn, d[1] / dn];
    [c0[0] + a * u[0] - sign * hh * u[1], c0[1] + a * u[1] + sign * hh * u[0]]
}

/// The Horst 4-bar at one chainstay angle: each moving link as a rigid
/// motion of its drawn (topped-out) points. The front triangle does not move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkagePose {
    /// Chainstay rotation about the main pivot from the drawn pose, rad (CCW, x forward y up).
    pub chainstay_rad: f64,
    /// Seatstay rotation from the drawn pose, rad (CCW).
    pub seatstay_rad: f64,
    /// Main pivot, metres (fixed).
    pub main_pivot: P2,
    /// Horst pivot as drawn, metres.
    pub horst_drawn: P2,
    /// Horst pivot at this pose, metres.
    pub horst: P2,
}

impl LinkagePose {
    /// Where a point drawn on the chainstay is at this pose, metres.
    pub fn chainstay_point(&self, p: P2) -> P2 {
        let r = rot2(sub(p, self.main_pivot), self.chainstay_rad);
        [self.main_pivot[0] + r[0], self.main_pivot[1] + r[1]]
    }

    /// Where a point drawn on the seatstay (axle, hanger, derailleur) is at this pose, metres.
    pub fn seatstay_point(&self, p: P2) -> P2 {
        let r = rot2(sub(p, self.horst_drawn), self.seatstay_rad);
        [self.horst[0] + r[0], self.horst[1] + r[1]]
    }
}

/// The 4-bar's pose with the chainstay turned `theta` rad about the main
/// pivot, on the circle-intersection branch `sign` (+1 or -1).
pub fn pose_at_angle(p: &LinkagePoints, theta: f64, sign: f64) -> LinkagePose {
    let (m, h, s, r) = (p.main_pivot, p.horst, p.ss_rocker, p.rocker_frame);
    let hr = rot2(sub(h, m), theta);
    let h2 = [m[0] + hr[0], m[1] + hr[1]];
    let s2 = intersect(h2, norm(sub(s, h)), r, norm(sub(s, r)), sign);
    let v = sub(s2, h2);
    let v0 = sub(s, h);
    let phi = v[1].atan2(v[0]) - v0[1].atan2(v0[0]);
    LinkagePose { chainstay_rad: theta, seatstay_rad: phi, main_pivot: m, horst_drawn: h, horst: h2 }
}

/// The branch of the seatstay-rocker intersection the drawn linkage is on (+1 or -1).
pub fn branch_sign(p: &LinkagePoints) -> f64 {
    let (h, s, r) = (p.horst, p.ss_rocker, p.rocker_frame);
    let cand = [1.0, -1.0].map(|sg| intersect(h, norm(sub(s, h)), r, norm(sub(s, r)), sg));
    if norm(sub(cand[0], s)) < norm(sub(cand[1], s)) { 1.0 } else { -1.0 }
}

/// A photo's scale and axes: the two axle centres clicked on it (pixels,
/// image y down) and the wheelbase between them. Maps pixels to the bike
/// side plane (origin rear axle, x toward the front axle, y up, metres) the
/// way `ml/tools/linkage.html` does.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhotoFrame {
    /// Rear axle centre, pixels.
    pub rear_axle_px: P2,
    /// Front axle centre, pixels.
    pub front_axle_px: P2,
    /// Wheelbase, metres.
    pub wheelbase_m: f64,
}

impl PhotoFrame {
    fn axes(&self) -> (P2, P2, f64) {
        let d = sub(self.front_axle_px, self.rear_axle_px);
        let len = norm(d).max(1e-9);
        let ex = [d[0] / len, d[1] / len];
        // the perpendicular that points up in the image (image y grows down)
        let mut ey = [ex[1], -ex[0]];
        if ey[1] > 0.0 {
            ey = [-ey[0], -ey[1]];
        }
        (ex, ey, len / self.wheelbase_m.max(1e-9))
    }

    /// Pixels per metre.
    pub fn px_per_m(&self) -> f64 {
        self.axes().2
    }

    /// A pixel position as a bike side-plane point, metres.
    pub fn to_bike(&self, px: P2) -> P2 {
        let (ex, ey, ppm) = self.axes();
        let v = sub(px, self.rear_axle_px);
        [(v[0] * ex[0] + v[1] * ex[1]) / ppm, (v[0] * ey[0] + v[1] * ey[1]) / ppm]
    }

    /// A bike side-plane point (metres) as a pixel position.
    pub fn to_px(&self, m: P2) -> P2 {
        let (ex, ey, ppm) = self.axes();
        [self.rear_axle_px[0] + (m[0] * ex[0] + m[1] * ey[0]) * ppm, self.rear_axle_px[1] + (m[0] * ex[1] + m[1] * ey[1]) * ppm]
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
