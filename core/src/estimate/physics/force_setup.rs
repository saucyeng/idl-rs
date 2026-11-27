//! The force model of a real bike (spec `2026-11-25-idl1-force-model-DRAFT.md` §3): masses,
//! springs and tyres from the bike's parameters (at the ride's date: its log's running sheet
//! on top), axles and the shock from its linkage, all in IMU0's axes. Masses are measured or
//! estimated, never fitted (whole-bike sim spec D3); every estimate here is marked.

use super::bike::{Bike, NL, S};
use super::forces::{kin, AirChambers, Head, ForceBike, ForceModel, ForceSpec, ForceWeights, NFP, P_FORK_DAMP, P_FRICTION, P_PRESSURE, P_RIDER_C, P_RIDER_K, P_SHOCK_DAMP, P_TYRE_C};
use crate::calibration::drivetrain::Drivetrain;
use crate::calibration::linkage::Linkage;
use crate::calibration::params::{mass_table, number, Params};
use crate::sim::bike::SimBike;
use crate::sim::laws::TyreLaw;
use crate::sim::ride::Damper;
use crate::sim::suspension::{air_spring, shock_stroke, End, PSI_PA};
use crate::sim::tyre::Tyre;

/// Standard gravity, m/s^2.
const G: f64 = 9.80665;

/// Shank and foot, each side, as a share of body mass: de Leva (1996), "Adjustments to
/// Zatsiorsky-Seluyanov's segment inertia parameters", J. Biomech. 29(9), males: shank
/// 4.33 %, foot 1.37 % (not on the shelf; as reproduced in Healthcare 2022 10:2054 table 5
/// and PMC7801088 table 2). Standing, they move with the pedals: the frame carries them.
pub const LOWER_LEGS_SHARE: f64 = 2.0 * (0.0433 + 0.0137);

/// The limbs, both sides, as a share of body mass: legs (thigh 14.16 %, shank 4.33 %, foot
/// 1.37 %: de Leva 1996 males, as reproduced in PMC7801088 table 2) and arms (upper arm 2.71 %,
/// forearm 1.62 %, hand 0.61 %: de Leva 1996 males, not checked against a source: the copies
/// reachable on 2026-11-27 were blocked).
pub const LIMBS_SHARE: f64 = 2.0 * (0.1416 + 0.0433 + 0.0137) + 2.0 * (0.0271 + 0.0162 + 0.0061);
/// The share of the limbs' mass the frame carries (Isaac, 2026-11-27: the limbs sit between,
/// isolated from the body and from the frame by the ankles and wrists). 7 % of the rider on the
/// frame: the fit's best between Beacon's laps (7 %) and 11/21's (3 % or less) in a sweep of 3,
/// 7, 11, 18 and 25 %; the share moves the fitted dampers, so it is to be fitted in the end.
pub const LIMBS_ON_FRAME: f64 = 0.07 / LIMBS_SHARE;

/// The standing rider's centre of mass above the bottom bracket without a height, m (an
/// estimate: attack position, knees and hips bent; the rider pose latent's 0.25 m prior takes
/// the rest).
pub const RIDER_CG_ABOVE_BB_M: f64 = 0.85;
/// ... with `rider.height_m`, as a share of it (an estimate: standing a little upright, as
/// Isaac rides, 2026-11-25; upright standing puts it near 0.55 of height above the feet).
pub const RIDER_CG_HEIGHT_SHARE: f64 = 0.5;
/// ... and ahead of it, m (an estimate: hips over the cranks).
pub const RIDER_CG_AHEAD_OF_BB_M: f64 = 0.05;
/// The sprung bike's centre of mass, as the share of the way from the bottom bracket to the
/// bottom of the head tube (an estimate: frame, fork crown and stanchions, shock, cockpit).
pub const FRAME_CG_SHARE: f64 = 0.4;
/// The sprung bike's inertia about its centre of mass, about (fwd, lat, up), kg m^2 per kg
/// (an estimate: a 1.2 m by 0.6 m frame as a thin plate).
pub const FRAME_GYRATION_M2: [f64; 3] = [0.03, 0.12, 0.09];
/// The rider body's inertia about its centre of mass, about (fwd, lat, up), kg m^2 per kg
/// (an estimate: a crouched trunk with arms and thighs, radii of gyration 0.28 / 0.28 / 0.17 m).
pub const RIDER_GYRATION_M2: [f64; 3] = [0.08, 0.08, 0.03];

/// Why a bike's force model cannot be set up.
#[derive(Debug, Clone, PartialEq)]
pub enum ForceSetupError {
    /// A parameter the model needs is not entered (the key, e.g. `fork.psi`).
    Missing(String),
    /// The linkage has no shock eyes, so there is no shock stroke.
    NoShockEyes,
    /// The linkage cannot be posed through its travel.
    Linkage(String),
}

/// The nominal value of each force parameter (multiplier 1): estimates until a ride
/// identifies them. Fork damping at the axle, mid clicks; the shock at its shaft
/// ([`Damper::default`]); seal friction; tyre radial damping; the rider's legs and arms
/// together (N/m, N m/rad and their dampers).
pub fn nominal() -> [f64; NFP] {
    let mut n = [1.0; NFP];
    n[P_FORK_DAMP..P_FORK_DAMP + 5].copy_from_slice(&[1500.0, 500.0, 3000.0, 1200.0, 0.3]);
    let d = Damper::default();
    n[P_SHOCK_DAMP..P_SHOCK_DAMP + 5].copy_from_slice(&[d.lsc_ns_m, d.hsc_ns_m, d.lsr_ns_m, d.hsr_ns_m, d.knee_mps]);
    n[P_FRICTION..P_FRICTION + 2].copy_from_slice(&[30.0, 40.0]);
    n[P_PRESSURE..P_PRESSURE + 4].copy_from_slice(&[1.0; 4]);
    n[P_TYRE_C..P_TYRE_C + 2].copy_from_slice(&[100.0, 100.0]);
    n[P_RIDER_K..P_RIDER_K + 6].copy_from_slice(&[20_000.0, 15_000.0, 30_000.0, 1500.0, 2000.0, 1000.0]);
    n[P_RIDER_C..P_RIDER_C + 6].copy_from_slice(&[800.0, 600.0, 1200.0, 60.0, 80.0, 40.0]);
    // back 0.15 m, down 0.03 m per g of braking (a guess; fitted)
    n[super::forces::P_BRAKE_SHIFT..super::forces::P_BRAKE_SHIFT + 2].copy_from_slice(&[0.15, 0.03]);
    n
}

/// The force model of the bike `bike` (the estimator's, from its calibration and linkage)
/// with parameters `p` and its linkage; `travel_m`: front and rear travel at the stops.
/// `drivetrain`: the bike's, for the derailleur cage's share of the rear's force in
/// [`DESCENT_COG_TEETH`] (None: left out).
pub fn force_model(bike: &Bike<f64>, p: &Params, linkage: &Linkage, drivetrain: Option<&Drivetrain>, travel_m: [f64; 2]) -> Result<ForceModel, ForceSetupError> {
    let need = |k: &str| number(p, k).filter(|v| *v > 0.0).ok_or_else(|| ForceSetupError::Missing(k.to_string()));
    let pts = &linkage.points_m;
    let up = bike.up.unit();
    let fwd = (bike.fwd - up.scale(up.dot(bike.fwd))).unit();
    // the linkage's side plane (x toward the front axle, y up, origin the rear axle) in IMU0's axes
    let at = |q: [f64; 2]| fwd.scale(q[0] - pts.imu0[0]) + up.scale(q[1] - pts.imu0[1]);
    let chainstay = need("geometry.chainstay_mm")? / 1000.0;
    let drop = number(p, "geometry.bb_drop_mm").unwrap_or(0.0) / 1000.0;
    let bb = [(chainstay * chainstay - drop * drop).sqrt(), -drop];
    let (reach, stack) = (need("geometry.reach_mm")? / 1000.0, need("geometry.stack_mm")? / 1000.0);
    let (ht, ha) = (need("geometry.head_tube_mm")? / 1000.0, need("geometry.head_angle_deg")?.to_radians());
    let ht_bottom = [bb[0] + reach + ht * ha.cos(), bb[1] + stack - ht * ha.sin()];
    let frame_cg = [bb[0] + FRAME_CG_SHARE * (ht_bottom[0] - bb[0]), bb[1] + FRAME_CG_SHARE * (ht_bottom[1] - bb[1])];
    let above_bb = number(p, "rider.height_m").filter(|h| *h > 0.5).map_or(RIDER_CG_ABOVE_BB_M, |h| RIDER_CG_HEIGHT_SHARE * h);
    let rider_cg = [bb[0] + RIDER_CG_AHEAD_OF_BB_M, bb[1] + above_bb];

    let mt = mass_table(p);
    let rider = need("rider.mass_kg")? + number(p, "rider.carried_kg").unwrap_or(0.0);
    let sprung = mt.bike_sprung_kg.ok_or_else(|| ForceSetupError::Missing("mass.bike_kg".into()))?;
    let share = LIMBS_ON_FRAME * LIMBS_SHARE;
    // the head (F2) on the body: its share of the rider plus the helmet and camera
    let m_head = HEAD_SHARE * rider + number(p, "rider.helmet_kg").filter(|v| *v >= 0.0).unwrap_or(HELMET_KG);
    let (m_frame, m_rider) = (sprung + share * rider, (1.0 - share - HEAD_SHARE) * rider);
    let radius = number(p, "wheels.tyre_radius_m").filter(|v| *v > 0.0).unwrap_or(0.37);

    // springs: chambers from the parameters, pressures from the sheet, resting where the
    // static load holds them (the load split from the centres of mass)
    let total = m_frame + m_rider + m_head + mt.front_unsprung_kg + mt.rear_unsprung_kg;
    let cg_x = (m_frame * frame_cg[0] + (m_rider + m_head) * rider_cg[0] + mt.front_unsprung_kg * pts.front_axle[0] + mt.rear_unsprung_kg * pts.rear_axle[0]) / total;
    let wheelbase = pts.front_axle[0] - pts.rear_axle[0];
    let front_load = total * G * (cg_x - pts.rear_axle[0]) / wheelbase;
    let rear_load = total * G - front_load;
    let axis = bike.steer_axis.unit();
    let axis = if axis.dot(up) < 0.0 { -axis } else { axis };
    let fork_spring = air_spring(p, End::Fork, travel_m[0]);
    let fork_gauge = need("fork.psi")? * PSI_PA;
    let fork_rest = fork_spring.stroke_under(fork_gauge, ((front_load - mt.front_unsprung_kg * G) * up.dot(axis)).max(1.0));
    let sim = SimBike::new(linkage, &Drivetrain::typical(bb, 0)).map_err(|e| ForceSetupError::Linkage(format!("{e:?}")))?;
    let stroke_at = |s: f64| shock_stroke(&sim, s).ok_or(ForceSetupError::NoShockEyes);
    let shock_stroke_max = stroke_at(travel_m[1])?;
    let shock_spring = air_spring(p, End::Shock, shock_stroke_max);
    let shock_gauge = need("shock.psi")? * PSI_PA;
    // the shock's stroke against rear travel, a cubic through the linkage
    let xs: Vec<f64> = (0..=32).map(|i| travel_m[1] * i as f64 / 32.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&s| stroke_at(s)).collect::<Result<_, _>>()?;
    let shock_coef = cubic(&xs, &ys)?;
    // the axle's member's angle through travel (the rear caliper's mount)
    let member: Vec<f64> = xs.iter().map(|&s| sim.pose(s).seatstay_rad).collect();
    let member_coef = cubic(&xs, &member)?;
    // the cage's angle through travel in the descending gear, and its moments
    let (cage_coef, cage_moment) = match drivetrain {
        Some(dt) => cage_curve(linkage, dt)?,
        None => (vec![], [0.0; 2]),
    };
    let mut fb = ForceBike {
        up,
        fwd,
        fork_axis: axis,
        front_axle: at(pts.front_axle),
        rear_axle: at(pts.rear_axle),
        frame_cg: at(frame_cg),
        rider_cg: at(rider_cg),
        m_frame,
        m_front: mt.front_unsprung_kg,
        m_rear: mt.rear_unsprung_kg,
        m_rider,
        m_limbs: share * rider,
        head: Some(Head {
            m_kg: m_head,
            gyration_m: HEAD_GYRATION_M,
            neck: fwd.scale(NECK_FROM_RIDER_CG_M[0]) + up.scale(NECK_FROM_RIDER_CG_M[1]),
            cg: fwd.scale(HEAD_CG_FROM_NECK_M[0]) + up.scale(HEAD_CG_FROM_NECK_M[1]),
            cam: fwd.scale(CAMERA_FROM_NECK_M[0]) + up.scale(CAMERA_FROM_NECK_M[1]),
        }),
        i_frame: FRAME_GYRATION_M2.map(|k| k * m_frame),
        i_rider: RIDER_GYRATION_M2.map(|k| k * m_rider),
        wheel_inertia: [mt.front.inertia_kgm2, mt.rear.inertia_kgm2],
        wheel_radius: [radius, radius],
        fork: AirChambers { area_m2: fork_spring.area_m2, positive_m3: fork_spring.positive_m3, negative_m3: fork_spring.negative_m3, gauge_pa: fork_gauge, rest_m: fork_rest },
        shock: AirChambers { area_m2: shock_spring.area_m2, positive_m3: shock_spring.positive_m3, negative_m3: shock_spring.negative_m3, gauge_pa: shock_gauge, rest_m: 0.0 },
        shock_stroke: shock_coef,
        axle_member_angle: member_coef,
        cage_angle: cage_coef,
        cage_moment_nm: cage_moment,
        travel_max: travel_m,
        stop_n_m: 2.0e6,
        tyre: [tyre(p, "front_tyre")?, tyre(p, "rear_tyre")?],
        nominal: nominal(),
        smooth_mps: 0.01,
        smooth_m: 0.001,
    };
    // the shock rests where the rear's static load holds it, through the leverage
    let mut q = [0.0; NL];
    let (mut lo, mut hi) = (0.0, travel_m[1]);
    for _ in 0..60 {
        let s = 0.5 * (lo + hi);
        q[S] = s;
        let k = kin(&fb, bike, &q);
        let isothermal = crate::sim::laws::air_force_n(fb.shock.area_m2, fb.shock.positive_m3, fb.shock.negative_m3, shock_gauge, k.x, 1.0);
        if isothermal * k.dx < (rear_load - fb.m_rear * G) * up.dot(k.dar) { lo = s } else { hi = s }
    }
    let rest_s = 0.5 * (lo + hi);
    fb.shock.rest_m = fb.stroke(rest_s).0;
    let defl = [front_load / fb.tyre[0].stiffness_n_m, rear_load / fb.tyre[1].stiffness_n_m];
    Ok(ForceModel { bike: fb, spec: ForceSpec::standard(defl), w: ForceWeights::default() })
}

/// Head and neck as a share of body mass: de Leva (1996) males, 6.94 % (not checked against a
/// source: the copies reachable on 2026-11-27 were blocked; Isaac, 2026-11-27: about 5.5 kg).
pub const HEAD_SHARE: f64 = 0.0694;
/// Full-face helmet with the camera when the sheet has no `rider.helmet_kg`, kg (Isaac,
/// 2026-11-27: about 1.1 kg), on top of `rider.mass_kg`.
pub const HELMET_KG: f64 = 1.1;
/// The head's radius of gyration about its centre of mass, m (a sphere; an estimate).
pub const HEAD_GYRATION_M: f64 = 0.09;
/// The neck pivot from the rider body's centre of mass, ahead and above, m (an estimate for an
/// attack position).
pub const NECK_FROM_RIDER_CG_M: [f64; 2] = [0.20, 0.35];
/// The head's centre of mass from the neck pivot, ahead and above, m (an estimate).
pub const HEAD_CG_FROM_NECK_M: [f64; 2] = [0.03, 0.12];
/// The helmet camera from the neck pivot, ahead and above, m (Isaac, 2026-11-27: front of the
/// top, above the eyes; the numbers an estimate).
pub const CAMERA_FROM_NECK_M: [f64; 2] = [0.08, 0.22];

/// The cog the force model takes the rider to descend in, teeth (Isaac, 2026-11-26: "say the
/// 14T"); the nearest the cassette has.
pub const DESCENT_COG_TEETH: u32 = 14;

/// A cubic (lowest power first) through `(xs, ys)` by least squares.
fn cubic(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, ForceSetupError> {
    let a = nalgebra::DMatrix::from_fn(xs.len(), 4, |i, j| xs[i].powi(j as i32));
    let coef = a.svd(true, true).solve(&nalgebra::DVector::from_column_slice(ys), 1e-14).map_err(|e| ForceSetupError::Linkage(e.to_string()))?;
    Ok(coef.iter().cloned().collect())
}

/// The derailleur cage's angle (rad, forward) against rear travel in [`DESCENT_COG_TEETH`], as
/// a cubic, and its spring's and clutch's moments (N m), from the sim's chain loop.
fn cage_curve(linkage: &Linkage, dt: &Drivetrain) -> Result<(Vec<f64>, [f64; 2]), ForceSetupError> {
    let sim = SimBike::new(linkage, dt).map_err(|e| ForceSetupError::Linkage(format!("{e:?}")))?;
    let gear = (0..dt.cassette_teeth.len()).min_by_key(|&g| dt.cassette_teeth[g].abs_diff(DESCENT_COG_TEETH)).ok_or_else(|| ForceSetupError::Linkage("no cassette".into()))?;
    let enabled = |i: usize| dt.rollers[i].enabled;
    let gc = crate::sim::kinematics::gear_curves(&sim, gear, 33, &enabled).map_err(|e| ForceSetupError::Linkage(format!("{e:?}")))?;
    let xs: Vec<f64> = gc.travel_mm.iter().map(|t| t * 1e-3).collect();
    let ys: Vec<f64> = gc.cage_deg.iter().map(|d| d.to_radians()).collect();
    let coef = cubic(&xs, &ys)?;
    let mid = gc.cage_deg[gc.cage_deg.len() / 2];
    Ok((coef, [dt.derailleur.spring_torque(mid), dt.derailleur.clutch_torque_nm]))
}

/// The tyre `key` (`front_tyre` or `rear_tyre`): width, pressure, a DH casing when its model
/// names one, an insert when one is entered (its height or its mass).
fn tyre(p: &Params, key: &str) -> Result<TyreLaw<f64>, ForceSetupError> {
    let psi = number(p, &format!("{key}.psi")).filter(|v| *v > 0.0).ok_or_else(|| ForceSetupError::Missing(format!("{key}.psi")))?;
    let width = number(p, &format!("{key}.width_in")).unwrap_or(2.4);
    let dh = p.get(&format!("{key}.model")).and_then(|v| v.as_str()).is_some_and(|m| m.to_ascii_uppercase().contains("DH"));
    let insert = number(p, &format!("{key}.insert_height_mm")).map_or(number(p, &format!("{key}.insert_kg")).is_some_and(|k| k > 0.0), |h| h > 0.0);
    Ok(TyreLaw::from_tyre(&Tyre::mtb_29(width, psi, dh, insert)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_legs_share_is_de_levas_shank_and_foot_both_sides() {
        // Arrange / Act / Assert
        assert!((LOWER_LEGS_SHARE - 0.114).abs() < 1e-9);
    }

    #[test]
    fn tyre_a_dh_casing_and_an_insert_come_from_the_parameters() {
        // Arrange
        let mut p = Params::new();
        p.insert("front_tyre.psi".into(), serde_json::json!(24));
        p.insert("front_tyre.model".into(), serde_json::json!("Maxxis Assegai DH casing"));
        p.insert("front_tyre.insert_kg".into(), serde_json::json!(0.26));

        // Act
        let t = tyre(&p, "front_tyre").unwrap();
        let plain = TyreLaw::<f64>::from_tyre(&Tyre::mtb_29(2.4, 24.0, false, false));

        // Assert
        assert!(t.insert_touch_m.is_some() && t.stiffness_n_m > plain.stiffness_n_m);
        assert_eq!(tyre(&Params::new(), "rear_tyre"), Err(ForceSetupError::Missing("rear_tyre.psi".into())));
    }
}
