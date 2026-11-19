//! Chain growth, kickback, cage angle and clearance through travel: the
//! quasi-static answers (spec §2.2). Geometry only, milliseconds to run.

use crate::calibration::linkage::P2;

use super::bike::{SimBike, WheelRole, CHAIN_HALF_ROLLER_M};
use super::path::{chain_path, close_cage, ChainPath};
use super::SimError;

/// One gear's curves through travel. Every vector has one value per travel sample.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GearCurves {
    /// Gear (0 = the smallest cog).
    pub gear: usize,
    /// Cog teeth.
    pub teeth: u32,
    /// Rear travel, mm.
    pub travel_mm: Vec<f64>,
    /// Chain growth, mm: how much longer the chain from a cog tooth to a
    /// chainring tooth along the top span gets, with the wheel held to the
    /// seatstay. Positive pulls the cranks back.
    pub growth_mm: Vec<f64>,
    /// Pedal kickback with the wheel held to the seatstay, degrees of crank.
    pub kickback_deg: Vec<f64>,
    /// The cage angle that closes the loop, forward (slack direction) from
    /// its drawn angle relative to the seatstay, degrees.
    pub cage_deg: Vec<f64>,
    /// How much longer the loop's path gets with the cage held at its
    /// top-out angle, mm: what the cage has to take up (negative: let out).
    pub loop_change_mm: Vec<f64>,
    /// Top span's free length, mm.
    pub top_span_mm: Vec<f64>,
    /// Clearance from the chain to each contact surface, mm (chain roller
    /// edge to surface; negative: the chain path crosses it). One vector per surface.
    pub clearance_mm: Vec<Vec<f64>>,
    /// Whether each roller touches the chain. One vector per roller.
    pub roller_touching: Vec<Vec<bool>>,
}

fn seg_dist(p: P2, a: P2, b: P2) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-18);
    let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0);
    let q = [a[0] + t * d[0] - p[0], a[1] + t * d[1] - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

fn crosses(a: P2, b: P2, c: P2, d: P2) -> bool {
    let o = |p: P2, q: P2, r: P2| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    o(a, b, c) * o(a, b, d) < 0.0 && o(c, d, a) * o(c, d, b) < 0.0
}

/// Distance from the chain's pin line to a polyline, metres (0 when they cross).
pub fn path_distance(path: &ChainPath, poly: &[P2]) -> f64 {
    let pts = path.polyline(0.003);
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        for s in poly.windows(2) {
            if crosses(w[0], w[1], s[0], s[1]) {
                return 0.0;
            }
            best = best.min(seg_dist(w[0], s[0], s[1])).min(seg_dist(s[0], w[0], w[1]));
        }
    }
    best
}

fn wrap_pi(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// The top run's length between body-fixed teeth: a tooth on the cog (it
/// turns with the seatstay, `ss_rad`) at angle `cog_ref + ss_rad` and a
/// fixed tooth on the ring at `ring_ref`.
fn top_run(path: &ChainPath, cog_ref: f64, ss_rad: f64, ring_ref: f64) -> f64 {
    let (ci, cog) = path.arc(WheelRole::Cog).expect("the cog is always in the path");
    let (ri, ring) = path.arc(WheelRole::Ring).expect("the ring is always in the path");
    let mut l = cog.wheel.len_radius * wrap_pi(cog_ref + ss_rad - cog.depart);
    for k in ci..ri {
        l += path.lines[k].len;
        if k > ci {
            l += path.arcs[k].len;
        }
    }
    l + ring.wheel.len_radius * wrap_pi(ring.arrive - ring_ref)
}

/// The curves for gear `gear` at `samples` travels from top-out to full
/// travel, with the rollers `enabled` says are fitted.
pub fn gear_curves(bike: &SimBike, gear: usize, samples: usize, enabled: &dyn Fn(usize) -> bool) -> Result<GearCurves, SimError> {
    let n = samples.max(2);
    let p0 = bike.pose(0.0);
    let (fwd0, path0) = close_cage(bike, &p0, gear, enabled)?;
    let cog_ref = path0.arc(WheelRole::Cog).expect("cog").1.depart - p0.seatstay_rad;
    let ring_ref = path0.arc(WheelRole::Ring).expect("ring").1.arrive;
    let run0 = top_run(&path0, cog_ref, p0.seatstay_rad, ring_ref);
    let fixed0 = path0.length;
    let ring_len_r = path0.arc(WheelRole::Ring).expect("ring").1.wheel.len_radius;
    let surfaces = &bike.drivetrain.contacts;
    let mut out = GearCurves {
        gear,
        teeth: bike.cog_teeth(gear)?,
        travel_mm: Vec::with_capacity(n),
        growth_mm: Vec::with_capacity(n),
        kickback_deg: Vec::with_capacity(n),
        cage_deg: Vec::with_capacity(n),
        loop_change_mm: Vec::with_capacity(n),
        top_span_mm: Vec::with_capacity(n),
        clearance_mm: vec![Vec::with_capacity(n); surfaces.len()],
        roller_touching: vec![Vec::with_capacity(n); bike.drivetrain.rollers.len()],
    };
    for k in 0..n {
        let s = bike.travel_max_m * k as f64 / (n - 1) as f64;
        let pose = bike.pose(s);
        let (fwd, path) = close_cage(bike, &pose, gear, enabled)?;
        let held = chain_path(bike, &pose, gear, bike.cage_abs(&pose, fwd0), enabled)?;
        let g = top_run(&path, cog_ref, pose.seatstay_rad, ring_ref) - run0;
        out.travel_mm.push(s * 1000.0);
        out.growth_mm.push(g * 1000.0);
        out.kickback_deg.push((g / ring_len_r).to_degrees());
        out.cage_deg.push(fwd.to_degrees());
        out.loop_change_mm.push((held.length - fixed0) * 1000.0);
        let (ci, _) = path.arc(WheelRole::Cog).expect("cog");
        out.top_span_mm.push(path.lines[ci].len * 1000.0);
        for (j, surf) in surfaces.iter().enumerate() {
            let poly: Vec<P2> = surf.polyline_m.iter().map(|&p| bike.place(&pose, surf.body, p)).collect();
            out.clearance_mm[j].push((path_distance(&path, &poly) - CHAIN_HALF_ROLLER_M) * 1000.0);
        }
        for i in 0..bike.drivetrain.rollers.len() {
            out.roller_touching[i].push(path.arc(WheelRole::Roller(i)).is_some());
        }
    }
    Ok(out)
}

/// What to draw for the bike at one travel: the chain, its wheels, and the linkage.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Snapshot {
    /// Rear travel, mm.
    pub travel_mm: f64,
    /// Cage angle, forward from drawn, degrees.
    pub cage_deg: f64,
    /// The chain's pin line, closed, metres.
    pub chain: Vec<P2>,
    /// Wheels the chain goes round, as (centre, radius) in metres; rollers
    /// clear of the chain are included.
    pub wheels: Vec<WheelView>,
    /// Linkage points at this travel, by name, metres.
    pub linkage: Vec<(String, P2)>,
    /// Contact surfaces at this travel, metres.
    pub surfaces: Vec<Vec<P2>>,
}

/// A wheel to draw.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WheelView {
    /// "cog", "ring", "upper_jockey", "lower_jockey" or "roller:<index>".
    pub role: String,
    /// Centre, metres.
    pub centre: P2,
    /// Pin-line radius, metres.
    pub radius: f64,
    /// Touching the chain.
    pub touching: bool,
}

/// The role's name on the wire.
pub fn role_name(role: WheelRole) -> String {
    match role {
        WheelRole::Cog => "cog".into(),
        WheelRole::Ring => "ring".into(),
        WheelRole::UpperJockey => "upper_jockey".into(),
        WheelRole::LowerJockey => "lower_jockey".into(),
        WheelRole::Roller(i) => format!("roller:{i}"),
    }
}

/// The linkage's points at `pose`, by name, metres.
pub fn linkage_points(bike: &SimBike, pose: &crate::calibration::linkage::LinkagePose) -> Vec<(String, P2)> {
    let p = &bike.points;
    let mut out = vec![
        ("front_axle".to_string(), p.front_axle),
        ("rear_axle".to_string(), pose.seatstay_point(p.rear_axle)),
        ("main_pivot".to_string(), p.main_pivot),
        ("horst".to_string(), pose.horst),
        ("ss_rocker".to_string(), pose.seatstay_point(p.ss_rocker)),
        ("rocker_frame".to_string(), p.rocker_frame),
        ("bb".to_string(), bike.drivetrain.bb_m),
    ];
    if let Some(s) = p.shock_frame {
        out.push(("shock_frame".into(), s));
    }
    if let Some(s) = p.shock_rocker {
        out.push(("shock_rocker".into(), bike.rocker_point(pose, s)));
    }
    out
}

/// The bike at travel `travel_m` in gear `gear` with the loop closed.
pub fn snapshot(bike: &SimBike, travel_m: f64, gear: usize, enabled: &dyn Fn(usize) -> bool) -> Result<Snapshot, SimError> {
    let pose = bike.pose(travel_m);
    let (fwd, path) = close_cage(bike, &pose, gear, enabled)?;
    let mut wheels: Vec<WheelView> = path.arcs.iter().map(|a| WheelView { role: role_name(a.wheel.role), centre: a.wheel.centre, radius: a.wheel.radius, touching: true }).collect();
    for i in 0..bike.drivetrain.rollers.len() {
        if enabled(i) && path.arc(WheelRole::Roller(i)).is_none() {
            let w = bike.roller(&pose, i);
            wheels.push(WheelView { role: role_name(w.role), centre: w.centre, radius: w.radius, touching: false });
        }
    }
    let mut linkage = linkage_points(bike, &pose);
    linkage.push(("cage_pivot".into(), bike.cage_pivot(&pose, gear)));
    Ok(Snapshot {
        travel_mm: travel_m * 1000.0,
        cage_deg: fwd.to_degrees(),
        chain: path.polyline(0.004),
        wheels,
        linkage,
        surfaces: bike.drivetrain.contacts.iter().map(|s| s.polyline_m.iter().map(|&p| bike.place(&pose, s.body, p)).collect()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::bike::tests::{gt_force_drivetrain, gt_force_linkage};
    use super::super::bike::Wrap;
    use super::super::bike::Wheel;
    use super::super::path::path_through;
    use super::*;
    use crate::calibration::drivetrain::{Body, ChainSide, Roller, Span};
    use crate::calibration::linkage::{Linkage, LinkagePoints};

    #[test]
    fn gear_curves_start_at_zero_growth_and_zero_kickback_at_top_out() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let c = gear_curves(&bike, 5, 9, &|_| true).unwrap();

        // Assert
        assert_eq!(c.travel_mm.len(), 9);
        assert!(c.growth_mm[0].abs() < 1e-9 && c.kickback_deg[0].abs() < 1e-9 && c.loop_change_mm[0].abs() < 1e-9);
        assert!((c.travel_mm[8] - 160.0).abs() < 1e-9);
    }

    #[test]
    fn top_span_length_is_constant_when_the_axle_swings_about_the_bb() {
        // Arrange: the Horst pivot at the axle and the main pivot at the BB,
        // so the axle swings on a circle about the BB
        let points = LinkagePoints {
            front_axle: [1.2, 0.0],
            rear_axle: [0.0, 0.0],
            main_pivot: [0.43, 0.03],
            horst: [0.0, 0.0],
            ss_rocker: [0.3, 0.2],
            rocker_frame: [0.45, 0.25],
            shock_frame: None,
            shock_rocker: None,
            imu0: [0.5, 0.4],
            fork_pod: None,
            rear_pod: None,
        };
        let linkage = Linkage { updated_at_ms: 0, points_m: points, wheelbase_m: 1.2, rear_travel_m: 0.1, front_travel_m: None, shock_eye_to_eye_m: None, source_image: None };
        let mut d = gt_force_drivetrain();
        d.bb_m = [0.43, 0.03];
        d.chainring_teeth = 30;
        d.cassette_teeth = vec![30];
        let bike = SimBike::new(&linkage, &d).unwrap();

        // Act: the top span alone, ring and cog the same size, so the span
        // is parallel to the axle-BB line and its length is that distance
        let p0 = bike.pose(0.0);
        let p1 = bike.pose(0.1);
        let span = |pose: &crate::calibration::linkage::LinkagePose| {
            let [cog, ring, _, _] = bike.fixed_wheels(pose, 0, 0.0).unwrap();
            let lower = Wheel { wrap: Wrap::Cw, ..cog };
            path_through(&[lower, ring]).unwrap().lines[0].len
        };

        // Assert
        assert!((span(&p0) - span(&p1)).abs() < 1e-9);
    }

    #[test]
    fn a_top_roller_pressing_on_the_top_span_changes_growth_and_the_loop_still_closes() {
        // Arrange: a 16T roller on the chainstay holding the top span down
        let mut d = gt_force_drivetrain();
        d.rollers.push(Roller { name: "Top roller".into(), body: Body::Chainstay, centre_m: [0.36, 0.07], teeth: Some(16), radius_m: 0.0, span: Span::Top, side: ChainSide::Under, inertia_kgm2: 2e-6, enabled: true, threaded: false });
        let bike = SimBike::new(&gt_force_linkage(), &d).unwrap();

        // Act
        let with = gear_curves(&bike, 5, 5, &|_| true).unwrap();
        let without = gear_curves(&bike, 5, 5, &|_| false).unwrap();

        // Assert
        assert!(with.roller_touching[0].iter().all(|&t| t));
        assert!((with.growth_mm[4] - without.growth_mm[4]).abs() > 0.1, "{} vs {}", with.growth_mm[4], without.growth_mm[4]);
        assert!(without.roller_touching[0].iter().all(|t| !t));
    }

    #[test]
    fn snapshot_draws_a_closed_chain_and_every_wheel() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let s = snapshot(&bike, 0.05, 3, &|_| true).unwrap();

        // Assert
        let (a, b) = (s.chain[0], *s.chain.last().unwrap());
        assert!(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() < 1e-9);
        assert_eq!(s.wheels.len(), 4);
    }

    #[test]
    fn a_threaded_idler_on_the_main_pivot_is_in_the_path_at_every_travel_and_growth_has_no_jumps() {
        // Arrange: Isaac's idler, 16T on the main pivot, the top span over it
        let mut d = gt_force_drivetrain();
        d.rollers.push(Roller { name: "Top idler".into(), body: Body::Chainstay, centre_m: [0.43717, 0.08031], teeth: Some(16), radius_m: 0.0, span: Span::Top, side: ChainSide::Over, inertia_kgm2: 2e-6, enabled: true, threaded: true });
        let bike = SimBike::new(&gt_force_linkage(), &d).unwrap();
        let fits: Vec<u32> = (90..170).step_by(2).filter(|&l| {
            let mut b = bike.clone();
            b.drivetrain.chain_links = l;
            gear_curves(&b, 5, 9, &|_| true).is_ok()
        }).collect();
        let mut b = bike.clone();
        b.drivetrain.chain_links = fits[fits.len() / 2];

        // Act
        let c = gear_curves(&b, 5, 17, &|_| true).unwrap();

        // Assert: always on the idler; no step in growth bigger than a smooth curve's (1 cm in 1 cm of travel)
        assert!(c.roller_touching[0].iter().all(|&t| t));
        let worst = c.growth_mm.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        assert!(worst < 10.0, "growth jumps {worst:.1} mm between samples: {:?}", c.growth_mm);
    }
}
