//! The bike for the physics estimator, from a calibration, the bike's linkage
//! (if it has one) and the calibration's physics fit (if fitted). Mirrors the
//! PyTorch reference's `Bike.__init__` and `gt_force()`.

use super::bike::{Bike, Rear};
use super::fourbar::{fourbar, FourBarError};
use super::la::{quat_to_mat, V3};
use crate::calibration::bike::CalibrationFit;
use crate::calibration::linkage::{Linkage, PhysicsFit};

/// Default distance of the fork pod beside the steering axis before a fit, m (a guess).
pub const STEER_OFFSET_M: f64 = 0.06;
/// Default crown-to-fork-pod length at top-out before a fit, m (a guess).
pub const BEND_LEN_M: f64 = 0.45;
/// Bushing overlap at top-out, m (a guess, not measured; only
/// [`Bike::bend_compliance`] uses it).
pub const BUSH_OVERLAP_M: f64 = 0.10;
/// Prior sigmas of the fitted geometry: steer point (m), bending length (m),
/// linkage scale. From the PyTorch reference's `FITTABLE`.
pub const GEOMETRY_SIGMA: [f64; 5] = [0.05, 0.05, 0.05, 0.15, 0.07];

/// The geometry the estimator fits: steer point (3, m), bending length (m),
/// linkage scale, in that order.
pub type Geometry = [f64; 5];

/// Why a bike could not be set up.
#[derive(Debug, Clone, PartialEq)]
pub enum SetupError {
    /// The calibration has no forward direction (only IMU0 was recorded).
    NoForward,
    /// The calibration has no fork pod (IMU1) or no rear pod (IMU2).
    MissingPod {
        /// The IMU index missing.
        imu_index: u8,
    },
    /// The linkage cannot be used.
    Linkage(FourBarError),
}

fn unit(v: [f64; 3]) -> V3<f64> {
    V3::<f64>::from_f64(v).unit()
}

/// The bike: calibration mounts and lever arms, the linkage's 4-bar (or the
/// generic rear without one), and the fitted geometry (or the defaults).
pub fn bike(cal: &CalibrationFit, linkage: Option<&Linkage>, fit: Option<&PhysicsFit>, rear_travel_m: f64) -> Result<Bike<f64>, SetupError> {
    let up = unit(cal.up);
    let fwd = unit(cal.forward.ok_or(SetupError::NoForward)?);
    let axis = cal.steer_axis.map(unit).unwrap_or(up);
    let pod = |i: u8| cal.imus.iter().find(|p| p.imu_index == i).ok_or(SetupError::MissingPod { imu_index: i });
    let (fork, rear) = (pod(1)?, pod(2)?);
    let r1 = V3::from_f64(fork.lever_m);
    let r2 = V3::from_f64(rear.lever_m);
    let lat = axis.cross(fwd).unit();
    let rear_path = match linkage {
        Some(l) => Rear::FourBar(fourbar(&l.points_m, rear_travel_m).map_err(SetupError::Linkage)?),
        None => Rear::Generic { axis: up.cross(fwd).unit(), pivot: r2.scale(0.6), rot_per_m: 0.25, dir: up },
    };
    let mut b = Bike {
        up,
        fwd,
        steer_axis: axis,
        steer_point: r1 + lat.scale(STEER_OFFSET_M),
        fork_mount: quat_to_mat(fork.mount),
        fork_lever: r1,
        bend_len: BEND_LEN_M,
        bush_overlap: BUSH_OVERLAP_M,
        bend_vs_travel: true,
        rear_mount: quat_to_mat(rear.mount),
        rear_lever: r2,
        rear: rear_path,
        link_scale: linkage.map_or(1.0, Linkage::scale_prior),
        mount_err_fork: V3::zero(),
        mount_err_rear: V3::zero(),
    };
    if let Some(f) = fit {
        b = with_geometry(&b, &[f.steer_point[0], f.steer_point[1], f.steer_point[2], f.bend_len_m, f.link_scale]);
    }
    Ok(b)
}

/// The bike's fitted geometry.
pub fn geometry(b: &Bike<f64>) -> Geometry {
    let p = b.steer_point.0;
    [p[0], p[1], p[2], b.bend_len, b.link_scale]
}

/// `b` with geometry `g`.
pub fn with_geometry(b: &Bike<f64>, g: &Geometry) -> Bike<f64> {
    let mut out = b.clone();
    out.steer_point = V3([g[0], g[1], g[2]]);
    out.bend_len = g[3];
    out.link_scale = g[4];
    out
}
