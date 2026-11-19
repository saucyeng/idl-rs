//! The linkage and the drivetrain put together: where every wheel the chain
//! goes round is, at a rear travel, gear and cage angle.

use crate::calibration::drivetrain::{pitch_radius, Body, ChainSide, Drivetrain, Span, CHAIN_PITCH_M};
use crate::calibration::linkage::{pose_at_angle, Linkage, LinkagePose, LinkagePoints, P2};
use crate::estimate::physics::fourbar::{chainstay_angle, fourbar};

use super::{SimError, SimErrorKind};

/// Half the height of a chain's roller, metres: a smooth roller or a surface
/// touches the chain this far from its pin line.
pub const CHAIN_HALF_ROLLER_M: f64 = 0.0039;

/// Which way the chain goes round a wheel, seen from the drive side with the
/// chain moving as it does when pedalling forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrap {
    /// Clockwise: the wheel's centre is on the right of the chain's travel.
    Cw,
    /// Counter-clockwise: the centre is on the left.
    Ccw,
}

impl Wrap {
    /// +1 for clockwise, -1 for counter-clockwise (the signed-radius convention of [`super::path`]).
    pub fn sign(self) -> f64 {
        match self {
            Wrap::Cw => 1.0,
            Wrap::Ccw => -1.0,
        }
    }
}

/// What a wheel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WheelRole {
    /// The cog of the gear being ridden (on the seatstay, at the axle).
    Cog,
    /// The chainring (front triangle, at the BB).
    Ring,
    /// The derailleur's upper jockey.
    UpperJockey,
    /// The derailleur's lower jockey.
    LowerJockey,
    /// A roller, by its index in [`Drivetrain::rollers`].
    Roller(usize),
}

/// A wheel the chain can go round, placed at one pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wheel {
    /// What it is.
    pub role: WheelRole,
    /// Centre, metres (front-triangle frame).
    pub centre: P2,
    /// Distance from the centre to the chain's pin line, metres.
    pub radius: f64,
    /// Chain length per radian of wrap, metres per radian (for a sprocket,
    /// teeth × pitch / 2π, a little under [`Self::radius`]).
    pub len_radius: f64,
    /// Which way the chain goes round it.
    pub wrap: Wrap,
    /// Toothed (the chain cannot slip round it).
    pub toothed: bool,
}

/// The linkage and drivetrain of one bike, ready to simulate.
#[derive(Debug, Clone)]
pub struct SimBike {
    /// The 4-bar's drawn points, metres.
    pub points: LinkagePoints,
    /// The drivetrain.
    pub drivetrain: Drivetrain,
    /// Full rear travel, metres.
    pub travel_max_m: f64,
    coef: Vec<f64>,
    sign: f64,
    /// +1 when the cage turning forward (the slack direction) is CCW.
    cage_fwd_sign: f64,
}

fn rot(v: P2, t: f64) -> P2 {
    let (s, c) = t.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

fn sprocket(role: WheelRole, centre: P2, teeth: u32, wrap: Wrap) -> Wheel {
    let len_radius = teeth as f64 * CHAIN_PITCH_M / std::f64::consts::TAU;
    Wheel { role, centre, radius: pitch_radius(teeth), len_radius, wrap, toothed: true }
}

impl SimBike {
    /// The bike from its linkage and drivetrain. Fails when the 4-bar locks
    /// before full travel or the drivetrain has no cassette.
    pub fn new(linkage: &Linkage, drivetrain: &Drivetrain) -> Result<SimBike, SimError> {
        let fb = fourbar(&linkage.points_m, linkage.rear_travel_m).map_err(|e| SimError::new(SimErrorKind::Linkage, format!("the linkage can't be used: {e:?}")))?;
        if drivetrain.cassette_teeth.is_empty() || drivetrain.chainring_teeth < 3 || drivetrain.chain_links < 10 {
            return Err(SimError::new(SimErrorKind::Drivetrain, "the drivetrain needs a chainring, a cassette and a chain"));
        }
        let d = &drivetrain.derailleur;
        let v = [d.lower_jockey_m[0] - d.cage_pivot_m[0], d.lower_jockey_m[1] - d.cage_pivot_m[1]];
        // turning CCW by a little moves the lower jockey by (-v.y, v.x): forward when it hangs below the pivot
        let cage_fwd_sign = if -v[1] >= 0.0 { 1.0 } else { -1.0 };
        Ok(SimBike { points: linkage.points_m.clone(), drivetrain: drivetrain.clone(), travel_max_m: linkage.rear_travel_m, coef: fb.coef, sign: fb.sign, cage_fwd_sign })
    }

    /// The 4-bar's pose at rear travel `travel_m` (metres, clamped to the travel).
    pub fn pose(&self, travel_m: f64) -> LinkagePose {
        let s = travel_m.clamp(0.0, self.travel_max_m);
        pose_at_angle(&self.points, chainstay_angle(&self.coef, s), self.sign)
    }

    /// Where point `p`, drawn on `body`, is at `pose`, metres.
    pub fn place(&self, pose: &LinkagePose, body: Body, p: P2) -> P2 {
        match body {
            Body::FrontTriangle => p,
            Body::Chainstay => pose.chainstay_point(p),
            Body::Seatstay => pose.seatstay_point(p),
        }
    }

    /// +1 when the cage turning in the slack direction is CCW, else -1.
    pub fn cage_fwd_sign(&self) -> f64 {
        self.cage_fwd_sign
    }

    /// The cage's absolute rotation from its drawn angle, rad (CCW), at
    /// `pose` with the cage turned `cage_fwd_rad` forward relative to the seatstay.
    pub fn cage_abs(&self, pose: &LinkagePose, cage_fwd_rad: f64) -> f64 {
        pose.seatstay_rad + self.cage_fwd_sign * cage_fwd_rad
    }

    /// The cage relative to the seatstay, forward from drawn, rad, given its absolute rotation.
    pub fn cage_fwd(&self, pose: &LinkagePose, cage_abs: f64) -> f64 {
        (cage_abs - pose.seatstay_rad) * self.cage_fwd_sign
    }

    /// How far the parallelogram moves the cage in gear `gear` from the
    /// photo's gear, metres, in drawn seatstay coordinates: along the line
    /// from the axle to the upper jockey, by the change in cog radius.
    fn gear_shift(&self, gear: usize) -> P2 {
        let d = &self.drivetrain;
        let teeth = |g: usize| d.cassette_teeth.get(g).or(d.cassette_teeth.last()).copied().unwrap_or(10);
        let dr = pitch_radius(teeth(gear)) - pitch_radius(teeth(d.derailleur.photo_gear));
        let u = [d.derailleur.upper_jockey_m[0] - self.points.rear_axle[0], d.derailleur.upper_jockey_m[1] - self.points.rear_axle[1]];
        let n = (u[0] * u[0] + u[1] * u[1]).sqrt().max(1e-9);
        [dr * u[0] / n, dr * u[1] / n]
    }

    /// The cage pivot at `pose` in gear `gear`, metres.
    pub fn cage_pivot(&self, pose: &LinkagePose, gear: usize) -> P2 {
        let (p, g) = (self.drivetrain.derailleur.cage_pivot_m, self.gear_shift(gear));
        pose.seatstay_point([p[0] + g[0], p[1] + g[1]])
    }

    /// Upper and lower jockey centres at `pose` in gear `gear` with the cage
    /// at absolute rotation `cage_abs`, metres.
    pub fn jockeys(&self, pose: &LinkagePose, gear: usize, cage_abs: f64) -> (P2, P2) {
        let d = &self.drivetrain.derailleur;
        let p = self.cage_pivot(pose, gear);
        let at = |j: P2| {
            let r = rot([j[0] - d.cage_pivot_m[0], j[1] - d.cage_pivot_m[1]], cage_abs);
            [p[0] + r[0], p[1] + r[1]]
        };
        (at(d.upper_jockey_m), at(d.lower_jockey_m))
    }

    /// Teeth on gear `gear` (0 = the smallest cog).
    pub fn cog_teeth(&self, gear: usize) -> Result<u32, SimError> {
        self.drivetrain
            .cassette_teeth
            .get(gear)
            .copied()
            .ok_or_else(|| SimError::new(SimErrorKind::Drivetrain, format!("gear {} is not on a {}-speed cassette", gear + 1, self.drivetrain.cassette_teeth.len())))
    }

    /// Which way the chain goes round roller `i`.
    pub fn roller_wrap(&self, i: usize) -> Wrap {
        let r = &self.drivetrain.rollers[i];
        // the top span runs forward (cog to ring), the lower span rearward
        match (r.span, r.side) {
            (Span::Top, ChainSide::Over) | (Span::Lower, ChainSide::Under) => Wrap::Cw,
            (Span::Top, ChainSide::Under) | (Span::Lower, ChainSide::Over) => Wrap::Ccw,
        }
    }

    /// Roller `i` placed at `pose`.
    pub fn roller(&self, pose: &LinkagePose, i: usize) -> Wheel {
        let r = &self.drivetrain.rollers[i];
        let centre = self.place(pose, r.body, r.centre_m);
        let wrap = self.roller_wrap(i);
        match r.teeth {
            Some(t) => sprocket(WheelRole::Roller(i), centre, t, wrap),
            None => {
                let radius = r.radius_m + CHAIN_HALF_ROLLER_M;
                Wheel { role: WheelRole::Roller(i), centre, radius, len_radius: radius, wrap, toothed: false }
            }
        }
    }

    /// The four wheels the chain always goes round, at `pose`, in gear
    /// `gear`, with the cage at absolute rotation `cage_abs`: cog, ring,
    /// lower jockey, upper jockey.
    pub fn fixed_wheels(&self, pose: &LinkagePose, gear: usize, cage_abs: f64) -> Result<[Wheel; 4], SimError> {
        let d = &self.drivetrain;
        let cog = sprocket(WheelRole::Cog, pose.seatstay_point(self.points.rear_axle), self.cog_teeth(gear)?, Wrap::Cw);
        let ring = sprocket(WheelRole::Ring, d.bb_m, d.chainring_teeth, Wrap::Cw);
        let (uj, lj) = self.jockeys(pose, gear, cage_abs);
        let lower = sprocket(WheelRole::LowerJockey, lj, d.derailleur.jockey_teeth[1], Wrap::Cw);
        let upper = sprocket(WheelRole::UpperJockey, uj, d.derailleur.jockey_teeth[0], Wrap::Ccw);
        Ok([cog, ring, lower, upper])
    }

    /// The fitted rollers on `span`, by index.
    pub fn rollers_on(&self, span: Span, enabled: &dyn Fn(usize) -> bool) -> Vec<usize> {
        (0..self.drivetrain.rollers.len()).filter(|&i| self.drivetrain.rollers[i].span == span && enabled(i)).collect()
    }

    /// The rocker's rotation at `pose` (rad, CCW, about its frame pivot) and
    /// where a point drawn on the rocker is: the rocker turns so its seatstay
    /// pivot follows the seatstay.
    pub fn rocker_point(&self, pose: &LinkagePose, p: P2) -> P2 {
        let rf = self.points.rocker_frame;
        let s0 = self.points.ss_rocker;
        let s1 = pose.seatstay_point(s0);
        let a = (s1[1] - rf[1]).atan2(s1[0] - rf[0]) - (s0[1] - rf[1]).atan2(s0[0] - rf[0]);
        let r = rot([p[0] - rf[0], p[1] - rf[1]], a);
        [rf[0] + r[0], rf[1] + r[1]]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::calibration::drivetrain::Drivetrain;

    /// The GT Force's linkage as clicked (ml/bikes/gt-force-linkage.json).
    pub(crate) fn gt_force_linkage() -> Linkage {
        Linkage {
            updated_at_ms: 0,
            points_m: LinkagePoints {
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
            },
            wheelbase_m: 1.28,
            rear_travel_m: 0.16,
            front_travel_m: None,
            shock_eye_to_eye_m: None,
            source_image: None,
        }
    }

    /// A plausible drivetrain on the GT Force: BB 100 mm below and 12 mm
    /// behind the main pivot, a SRAM-like derailleur hanging off the axle.
    pub(crate) fn gt_force_drivetrain() -> Drivetrain {
        let mut d = Drivetrain::typical([0.425, -0.02], 0);
        // the even link count in the middle of what fits every gear
        let bike = SimBike::new(&gt_force_linkage(), &d).unwrap();
        let (lo, hi) = super::super::path::chain_fit(&bike, &|_| true).expect("the typical drivetrain fits the GT Force");
        d.chain_links = (((lo + hi) / 4.0).round() * 2.0) as u32;
        d.chain_mass_kg = 0.257 * d.chain_links as f64 / 120.0;
        d
    }

    #[test]
    fn sim_bike_at_top_out_puts_the_cog_on_the_axle_and_moves_the_cage_down_for_a_bigger_cog() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let pose = bike.pose(0.0);
        let [cog, _, lower, upper] = bike.fixed_wheels(&pose, 5, bike.cage_abs(&pose, 0.0)).unwrap();

        // Assert
        assert!(cog.centre[0].abs() < 1e-6 && cog.centre[1].abs() < 1e-6);
        assert!(upper.centre[1] < -0.058 - 0.02, "the 21T moves the cage down: {:?}", upper.centre);
        assert!(lower.centre[1] < -0.22);
    }

    #[test]
    fn sim_bike_at_full_travel_lifts_the_axle_by_the_travel() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let pose = bike.pose(0.16);
        let axle = pose.seatstay_point([0.0, 0.0]);

        // Assert
        assert!((axle[1] - 0.16).abs() < 1e-5, "{axle:?}");
    }

    #[test]
    fn cage_turned_forward_moves_the_lower_jockey_forward() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let pose = bike.pose(0.0);

        // Act
        let (_, before) = bike.jockeys(&pose, 0, bike.cage_abs(&pose, 0.0));
        let (_, after) = bike.jockeys(&pose, 0, bike.cage_abs(&pose, 0.2));

        // Assert
        assert!(after[0] > before[0]);
    }
}
