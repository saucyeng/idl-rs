//! Chain growth, kickback, cage angle and clearance through travel: the
//! quasi-static answers (spec §2.2). Geometry only, milliseconds to run.

use crate::calibration::drivetrain::CHAIN_PITCH_M;
use crate::calibration::linkage::{LinkagePose, P2};

use super::bike::{SimBike, WheelRole, CHAIN_HALF_ROLLER_M};
use super::path::{chain_path, close_cage, close_cage_near, ChainPath};
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
    /// The squat line's angle above the ground, degrees ([`Squat`]); null
    /// where it is undefined (the links parallel, or the chain along the
    /// axle's line).
    pub squat_angle_deg: Vec<Option<f64>>,
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
        squat_angle_deg: Vec::with_capacity(n),
    };
    let mut near = fwd0;
    for k in 0..n {
        let s = bike.travel_max_m * k as f64 / (n - 1) as f64;
        let pose = bike.pose(s);
        let (fwd, path) = close_cage_near(bike, &pose, gear, near, enabled)?;
        near = fwd;
        let held = chain_path(bike, &pose, gear, bike.cage_abs(&pose, fwd0), enabled)?;
        let g = top_run(&path, cog_ref, pose.seatstay_rad, ring_ref) - run0;
        out.travel_mm.push(s * 1000.0);
        out.growth_mm.push(g * 1000.0);
        out.kickback_deg.push((g / ring_len_r).to_degrees());
        out.cage_deg.push(fwd.to_degrees());
        out.squat_angle_deg.push(squat(bike, &pose, &path).angle_deg);
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
    /// The chain's pins, one per link, a pitch apart along the pin line
    /// from where it arrives on the first wheel, metres.
    pub pins: Vec<P2>,
    /// Wheels the chain goes round, as (centre, radius) in metres; rollers
    /// clear of the chain are included.
    pub wheels: Vec<WheelView>,
    /// Linkage points at this travel, by name, metres.
    pub linkage: Vec<(String, P2)>,
    /// Contact surfaces at this travel, metres.
    pub surfaces: Vec<Vec<P2>>,
    /// The squat construction at this travel.
    pub squat: Squat,
}

/// The squat construction (Cossalter, Motorcycle Dynamics 2nd ed., §6.1.3,
/// PDF pp. 246-248): the squat line runs from the rear tyre's contact
/// point through A, where the chain's top span (from the cog) meets the
/// line from the rear axle through the instant centre of the body carrying
/// it (the seatstay, about the frame: where the chainstay's and the
/// rocker's lines cross). The frame is held level and the ground is a
/// wheel radius below each axle; the rider's weight shift (sag, pitch) is
/// not modelled.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Squat {
    /// Rear axle, metres.
    pub axle: P2,
    /// The seatstay's instant centre about the frame, metres; null when the
    /// chainstay and rocker are parallel.
    pub ic: Option<P2>,
    /// Point A, metres; null when undefined.
    pub point: Option<P2>,
    /// The rear tyre's contact point, metres.
    pub contact: P2,
    /// The front tyre's contact point, metres.
    pub front_contact: P2,
    /// The squat line's angle above the ground line (rear to front contact), degrees.
    pub angle_deg: Option<f64>,
}

/// Where two lines (each through two points) cross; None when parallel.
fn cross_lines(a0: P2, a1: P2, b0: P2, b1: P2) -> Option<P2> {
    let (da, db) = ([a1[0] - a0[0], a1[1] - a0[1]], [b1[0] - b0[0], b1[1] - b0[1]]);
    let den = da[0] * db[1] - da[1] * db[0];
    let scale = (da[0].hypot(da[1]) * db[0].hypot(db[1])).max(1e-12);
    if den.abs() < 1e-9 * scale {
        return None;
    }
    let t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / den;
    Some([a0[0] + t * da[0], a0[1] + t * da[1]])
}

/// The squat construction at `pose` with the chain on `path`.
pub fn squat(bike: &SimBike, pose: &LinkagePose, path: &ChainPath) -> Squat {
    let p = &bike.points;
    let r = bike.drivetrain.wheel_radius_m;
    let axle = pose.seatstay_point(p.rear_axle);
    let ic = cross_lines(p.main_pivot, pose.horst, p.rocker_frame, pose.seatstay_point(p.ss_rocker));
    let (ci, _) = path.arc(WheelRole::Cog).expect("the cog is always in the path");
    let chain = &path.lines[ci];
    let point = ic.and_then(|c| cross_lines(axle, c, chain.from, chain.to));
    let contact = [axle[0], axle[1] - r];
    let front_contact = [p.front_axle[0], p.front_axle[1] - r];
    let ground = (front_contact[1] - contact[1]).atan2(front_contact[0] - contact[0]);
    let angle_deg = point.map(|a| ((a[1] - contact[1]).atan2(a[0] - contact[0]) - ground).to_degrees());
    Squat { axle, ic, point, contact, front_contact, angle_deg }
}

/// Anti-squat, percent: the squat line's slope over the load transfer
/// line's (Cossalter §6.1.3, the squat ratio ℜ = tan τ / tan σ inverted to
/// the bicycle convention, 100 % = no squat), for a centre of mass
/// `cog_height_m` above the ground on a `wheelbase_m` wheelbase.
pub fn anti_squat_pct(squat_angle_deg: f64, cog_height_m: f64, wheelbase_m: f64) -> f64 {
    100.0 * squat_angle_deg.to_radians().tan() * wheelbase_m / cog_height_m.max(1e-6)
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
    snapshot_near(bike, travel_m, gear, 0.0, enabled)
}

/// [`snapshot`] with the cage on the loop-closing angle nearest `near_fwd`
/// (rad forward from drawn): the previous sample's, through travel.
pub fn snapshot_near(bike: &SimBike, travel_m: f64, gear: usize, near_fwd: f64, enabled: &dyn Fn(usize) -> bool) -> Result<Snapshot, SimError> {
    let pose = bike.pose(travel_m);
    let (fwd, path) = close_cage_near(bike, &pose, gear, near_fwd, enabled)?;
    Ok(snapshot_of(bike, &pose, travel_m, gear, fwd, &path, enabled))
}

/// The bike as drawn: topped out, in the photo's gear, with the cage at its
/// drawn angle, so the chain runs round the jockeys where they were clicked
/// whatever the chain's length (the loop needn't close).
pub fn drawn_snapshot(bike: &SimBike, enabled: &dyn Fn(usize) -> bool) -> Result<Snapshot, SimError> {
    let pose = bike.pose(0.0);
    let gear = bike.drivetrain.derailleur.photo_gear;
    let path = chain_path(bike, &pose, gear, bike.cage_abs(&pose, 0.0), enabled)?;
    Ok(snapshot_of(bike, &pose, 0.0, gear, 0.0, &path, enabled))
}

fn snapshot_of(bike: &SimBike, pose: &LinkagePose, travel_m: f64, gear: usize, fwd: f64, path: &ChainPath, enabled: &dyn Fn(usize) -> bool) -> Snapshot {
    let pose = *pose;
    let mut wheels: Vec<WheelView> = path.arcs.iter().map(|a| WheelView { role: role_name(a.wheel.role), centre: a.wheel.centre, radius: a.wheel.radius, touching: true }).collect();
    for i in 0..bike.drivetrain.rollers.len() {
        if enabled(i) && path.arc(WheelRole::Roller(i)).is_none() {
            let w = bike.roller(&pose, i);
            wheels.push(WheelView { role: role_name(w.role), centre: w.centre, radius: w.radius, touching: false });
        }
    }
    let mut linkage = linkage_points(bike, &pose);
    linkage.push(("cage_pivot".into(), bike.cage_pivot(&pose, gear)));
    Snapshot {
        travel_mm: travel_m * 1000.0,
        cage_deg: fwd.to_degrees(),
        chain: path.polyline(0.004),
        pins: (0..(path.length / CHAIN_PITCH_M).floor() as usize).map(|k| path.point_at(k as f64 * CHAIN_PITCH_M)).collect(),
        wheels,
        linkage,
        surfaces: bike.drivetrain.contacts.iter().map(|s| s.polyline_m.iter().map(|&p| bike.place(&pose, s.body, p)).collect()).collect(),
        squat: squat(bike, &pose, path),
    }
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

    #[test]
    fn drawn_snapshot_puts_the_jockeys_where_they_were_clicked_in_the_photos_gear() {
        // Arrange: a photo taken on the fourth cog
        let mut d = gt_force_drivetrain();
        d.derailleur.photo_gear = 3;
        let bike = SimBike::new(&gt_force_linkage(), &d).unwrap();

        // Act
        let s = drawn_snapshot(&bike, &|_| true).unwrap();

        // Assert
        let at = |role: &str| s.wheels.iter().find(|w| w.role == role).unwrap().centre;
        let near = |a: P2, b: P2| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() < 1e-6;
        assert!(near(at("upper_jockey"), d.derailleur.upper_jockey_m) && near(at("lower_jockey"), d.derailleur.lower_jockey_m), "{:?} {:?} vs {:?} {:?}, pose {:?}", at("upper_jockey"), at("lower_jockey"), d.derailleur.upper_jockey_m, d.derailleur.lower_jockey_m, bike.pose(0.0));
        assert_eq!(s.cage_deg, 0.0);
        let cog = s.wheels.iter().find(|w| w.role == "cog").unwrap();
        assert!((cog.radius - crate::calibration::drivetrain::pitch_radius(d.cassette_teeth[3])).abs() < 1e-9, "{}", cog.radius);
    }

    #[test]
    fn gear_curves_cage_angle_moves_smoothly_through_travel_in_every_gear() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act / Assert: no sample jumps to another branch
        for g in 0..bike.drivetrain.cassette_teeth.len() {
            let c = gear_curves(&bike, g, 41, &|_| true).unwrap();
            let worst = c.cage_deg.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
            assert!(worst < 3.0, "gear {g}: cage jumps {worst:.1} deg between samples: {:?}", c.cage_deg);
        }
    }

    #[test]
    fn squat_point_is_on_the_chain_line_and_on_the_axle_to_instant_centre_line() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let pose = bike.pose(0.05);
        let (_, path) = close_cage(&bike, &pose, 5, &|_| true).unwrap();

        // Act
        let q = squat(&bike, &pose, &path);

        // Assert: A is collinear with both lines; the contact is a wheel radius under the axle
        let (a, ic) = (q.point.unwrap(), q.ic.unwrap());
        let (ci, _) = path.arc(WheelRole::Cog).unwrap();
        let ch = &path.lines[ci];
        let off = |p0: P2, p1: P2, x: P2| ((p1[0] - p0[0]) * (x[1] - p0[1]) - (p1[1] - p0[1]) * (x[0] - p0[0])) / (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
        assert!(off(ch.from, ch.to, a).abs() < 1e-9 && off(q.axle, ic, a).abs() < 1e-9);
        assert!((q.axle[1] - q.contact[1] - bike.drivetrain.wheel_radius_m).abs() < 1e-12);
        assert!(q.angle_deg.unwrap().is_finite());
    }

    #[test]
    fn anti_squat_is_100_percent_when_the_squat_line_points_at_the_centre_of_mass_over_the_front_contact() {
        // Arrange: CoM 1.1 m up on a 1.25 m wheelbase
        let tau = (1.1f64 / 1.25).atan().to_degrees();

        // Act / Assert
        assert!((anti_squat_pct(tau, 1.1, 1.25) - 100.0).abs() < 1e-9);
        assert!(anti_squat_pct(0.0, 1.1, 1.25).abs() < 1e-12);
    }
}
