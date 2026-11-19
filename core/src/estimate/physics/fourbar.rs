//! A planar 4-bar from its points: rear travel -> chainstay angle as a
//! polynomial fitted to the closed-form linkage (exact to micro-radians), so
//! the pose is smooth and its derivatives exact. Port of
//! `ml/idlml/bike/fourbar.py`.

use super::bike::FourBar;
use crate::calibration::linkage::LinkagePoints;

use crate::calibration::linkage::{branch_sign, pose_at_angle, P2};

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

/// Chainstay angle `theta` -> rear axle position (side plane, m).
fn axle(p: &LinkagePoints, theta: f64, sign: f64) -> P2 {
    pose_at_angle(p, theta, sign).seatstay_point(p.rear_axle)
}

/// The chainstay angle (rad) the 4-bar's polynomial gives at rear travel `travel_m` (m).
pub fn chainstay_angle(coef: &[f64], travel_m: f64) -> f64 {
    horner(coef, travel_m)
}

/// Why a linkage cannot be used.
#[derive(Debug, Clone, PartialEq)]
pub enum FourBarError {
    /// The linkage reaches only `reached_m` of travel before it locks.
    Locks {
        /// Travel reached, m.
        reached_m: f64,
        /// Travel asked for, m.
        travel_m: f64,
    },
    /// The travel -> angle polynomial misses the linkage by `err_rad`.
    PoorFit {
        /// Largest error, rad.
        err_rad: f64,
    },
}

/// The 4-bar the estimator uses, from photo points and the rear travel (m).
pub fn fourbar(p: &LinkagePoints, travel_max_m: f64) -> Result<FourBar<f64>, FourBarError> {
    let sign = branch_sign(p);
    let dir = if axle(p, 1e-3, sign)[1] > 0.0 { 1.0 } else { -1.0 };
    let (mut th, mut tr) = (Vec::new(), Vec::new());
    for i in 0..=6000 {
        let t = dir * 0.6 * i as f64 / 6000.0;
        let y = axle(p, t, sign)[1];
        if y <= travel_max_m * 1.3 {
            th.push(t);
            tr.push(y);
        }
    }
    let reached = tr.iter().cloned().fold(f64::MIN, f64::max);
    if reached < travel_max_m {
        return Err(FourBarError::Locks { reached_m: reached, travel_m: travel_max_m });
    }
    let coef = polyfit(&tr, &th, 9);
    let err = tr.iter().zip(&th).map(|(&x, &y)| (horner(&coef, x) - y).abs()).fold(0.0, f64::max);
    if err > 1e-5 {
        return Err(FourBarError::PoorFit { err_rad: err });
    }
    let rel = |q: P2| sub(q, p.imu0);
    Ok(FourBar { main_pivot: rel(p.main_pivot), horst: rel(p.horst), ss_rocker: rel(p.ss_rocker), rocker_frame: rel(p.rocker_frame), coef, sign })
}

fn horner(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

/// Least-squares polynomial (lowest power first), columns scaled for conditioning.
fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let k = degree + 1;
    let xs = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
    let a = nalgebra::DMatrix::from_fn(x.len(), k, |i, j| (x[i] / xs).powi(j as i32));
    let b = nalgebra::DVector::from_column_slice(y);
    let sol = a.svd(true, true).solve(&b, 1e-14).expect("svd solve");
    (0..k).map(|j| sol[j] / xs.powi(j as i32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt_force() -> LinkagePoints {
        // the GT Force's points as clicked (ml/bikes/gt-force-linkage.json)
        LinkagePoints {
            front_axle: [1.28, 0.0],
            rear_axle: [0.0, 0.0],
            main_pivot: [0.43717, 0.08031],
            horst: [0.07805, -0.01296],
            ss_rocker: [0.37094, 0.18068],
            rocker_frame: [0.48793, 0.19457],
            shock_frame: Some([0.52277, 0.04682]),
            shock_rocker: Some([0.55827, 0.25971]),
            imu0: [0.47677, 0.36738],
            fork_pod: None,
            rear_pod: None,
        }
    }

    #[test]
    fn gt_force_linkage_reproduces_the_photo_tools_sweep() {
        // Arrange
        let p = gt_force();

        // Act
        let fb = fourbar(&p, 0.16).unwrap();

        // Assert: the tool's sweep has the seatstay turned -0.193148 rad at 160.11 mm,
        // so the axle at the polynomial's angle for that travel is back at that travel
        let th = horner(&fb.coef, 0.16011);
        assert!((axle(&p, th, fb.sign)[1] - 0.16011).abs() < 1e-6);
        assert!(horner(&fb.coef, 0.0).abs() < 1e-6);
        // and agrees with the PyTorch reference's polynomial over the travel
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/physics_golden.json")).unwrap();
        let py: Vec<f64> = serde_json::from_value(doc["bike"]["linkage"]["coef"].clone()).unwrap();
        for i in 0..=160 {
            let x = i as f64 / 1000.0;
            assert!((horner(&fb.coef, x) - horner(&py, x)).abs() < 2e-6, "at {x} m");
        }
        assert_eq!(fb.sign, doc["bike"]["linkage"]["sign"].as_f64().unwrap());
    }
}
