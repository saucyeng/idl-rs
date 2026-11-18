//! A planar 4-bar from its points: rear travel -> chainstay angle as a
//! polynomial fitted to the closed-form linkage (exact to micro-radians), so
//! the pose is smooth and its derivatives exact. Port of
//! `ml/idlml/bike/fourbar.py`.

use super::bike::FourBar;
use crate::calibration::linkage::LinkagePoints;

type P2 = [f64; 2];

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

/// The intersection of circle (c0, r0) and (c1, r1) on the `sign` side.
fn intersect(c0: P2, r0: f64, c1: P2, r1: f64, sign: f64) -> P2 {
    let d = sub(c1, c0);
    let dn = norm(d);
    let a = (r0 * r0 - r1 * r1 + dn * dn) / (2.0 * dn);
    let hh = (r0 * r0 - a * a).max(0.0).sqrt();
    let u = [d[0] / dn, d[1] / dn];
    [c0[0] + a * u[0] - sign * hh * u[1], c0[1] + a * u[1] + sign * hh * u[0]]
}

/// Chainstay angle `theta` -> rear axle position (side plane, m).
fn axle(p: &LinkagePoints, theta: f64, sign: f64) -> P2 {
    let (m, h, s, r) = (p.main_pivot, p.horst, p.ss_rocker, p.rocker_frame);
    let h2 = [m[0] + rot2(sub(h, m), theta)[0], m[1] + rot2(sub(h, m), theta)[1]];
    let s2 = intersect(h2, norm(sub(s, h)), r, norm(sub(s, r)), sign);
    let v = sub(s2, h2);
    let v0 = sub(s, h);
    let phi = v[1].atan2(v[0]) - v0[1].atan2(v0[0]);
    let back = rot2([-h[0], -h[1]], phi);
    [h2[0] + back[0], h2[1] + back[1]]
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
    let (h, s, r) = (p.horst, p.ss_rocker, p.rocker_frame);
    let cand = [1.0, -1.0].map(|sg| intersect(h, norm(sub(s, h)), r, norm(sub(s, r)), sg));
    let sign = if norm(sub(cand[0], s)) < norm(sub(cand[1], s)) { 1.0 } else { -1.0 };
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
