//! A bike's drivetrain for the drivetrain simulator (`sim`): where the
//! chainring, derailleur, rollers and the surfaces the chain can hit sit, and
//! the masses, torques and teeth that drive the chain's dynamics. Spec:
//! idl1-app `docs/superpowers/specs/2026-11-19-idl1-drivetrain-sim-DRAFT.md` §4.
//!
//! Bike-level data like [`super::linkage::Linkage`], kept beside it in the
//! bike's file. Points are in the same frame as
//! [`super::linkage::LinkagePoints`]: the bike side plane seen from the drive
//! side, origin at the rear axle, x toward the front axle, y up, metres,
//! topped out (as photographed). Each point is drawn on one [`Body`] and moves
//! with it through travel.

use serde::{Deserialize, Serialize};

use super::linkage::{PhotoFrame, P2};

/// Roller chain pitch, metres (1/2 inch).
pub const CHAIN_PITCH_M: f64 = 0.0127;

/// The rigid part of the bike a point is drawn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Body {
    /// Front triangle (BB, main pivot, ISCG mounts). Does not move with travel.
    FrontTriangle,
    /// Chainstay: main pivot to Horst pivot. Turns about the main pivot.
    Chainstay,
    /// Seatstay: Horst pivot to rear axle and rocker. Carries the axle and hanger.
    Seatstay,
}

/// Which chain span a roller guides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Span {
    /// The top (drive) span: cog to chainring.
    Top,
    /// The lower (slack) span: chainring to the lower jockey.
    Lower,
}

/// Which side of a roller the chain runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainSide {
    /// The chain runs over the roller (the roller pushes the chain up).
    Over,
    /// The chain runs under the roller (the roller holds the chain down).
    Under,
}

/// Which side of a contact surface the chain stays on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceSide {
    /// The chain stays above the surface (a chainstay protector's top edge).
    Above,
    /// The chain stays below the surface (the stay's underside).
    Below,
}

/// A chain roller or idler. It touches the chain only when the chain would
/// otherwise cross it, so a guide that sits clear of the chain at some
/// travel does nothing there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Roller {
    /// Name shown in the app ("Top roller", "ISCG lower").
    pub name: String,
    /// The body it is mounted on.
    pub body: Body,
    /// Centre, metres, drawn.
    pub centre_m: P2,
    /// Teeth; `None` for a smooth roller (radius from [`Self::radius_m`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teeth: Option<u32>,
    /// A smooth roller's radius where it touches the chain's rollers, metres. Unused when toothed.
    #[serde(default)]
    pub radius_m: f64,
    /// The span it guides.
    pub span: Span,
    /// The side of it the chain runs on.
    pub side: ChainSide,
    /// Its rotating inertia, kg·m².
    #[serde(default = "default_roller_inertia")]
    pub inertia_kgm2: f64,
    /// Whether it is fitted (off: kept in the file for an A/B, but not simulated).
    #[serde(default = "yes")]
    pub enabled: bool,
    /// An idler the chain is threaded round, so it stays on that side of it at
    /// every travel (a high-pivot or main-pivot idler). Otherwise a guide that
    /// acts only when the chain reaches it (an ISCG roller, a chain guide).
    #[serde(default)]
    pub threaded: bool,
}

fn default_roller_inertia() -> f64 {
    2.0e-6
}
fn yes() -> bool {
    true
}

/// A surface the chain can hit, as a polyline on one body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactSurface {
    /// Name ("Chainstay protector").
    pub name: String,
    /// The body it is on.
    pub body: Body,
    /// Points along the surface, metres, drawn.
    pub polyline_m: Vec<P2>,
    /// The side the chain stays on.
    pub side: SurfaceSide,
}

/// The rear derailleur's cage and its spring and clutch. The three points
/// are clicked on the photo in the gear it was taken in; the cage turns about
/// [`Self::cage_pivot_m`] relative to the seatstay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Derailleur {
    /// Cage pivot (the P-knuckle the cage turns about), metres, drawn. On the seatstay.
    pub cage_pivot_m: P2,
    /// Upper jockey centre, metres, drawn.
    pub upper_jockey_m: P2,
    /// Lower jockey centre, metres, drawn.
    pub lower_jockey_m: P2,
    /// Upper and lower jockey teeth.
    pub jockey_teeth: [u32; 2],
    /// The spring's torque on the cage, toward chain tension, N·m, against
    /// the cage's rotation from its drawn angle in the slack direction
    /// (forward), degrees: `[[deg, N·m], …]`, sorted by angle. Linear between
    /// points and beyond the ends.
    pub spring_torque_nm: Vec<[f64; 2]>,
    /// Clutch friction against the cage turning in the slack direction, N·m.
    /// It is one-way: the cage turns back toward tension against the spring
    /// and bearing only.
    pub clutch_torque_nm: f64,
    /// How far the cage turns forward, degrees, after turning back, before
    /// the clutch takes hold (the play in its one-way bearing). 0 = none.
    #[serde(default)]
    pub clutch_free_play_deg: f64,
    /// Bearing friction either way, N·m.
    pub bearing_torque_nm: f64,
    /// Cage inertia about its pivot, jockeys' masses included, kg·m².
    pub cage_inertia_kgm2: f64,
    /// Each jockey's inertia about its axle, kg·m².
    pub jockey_inertia_kgm2: f64,
    /// Cage mass with its jockeys, kg (its weight and the bike's
    /// accelerations swing the cage about its pivot).
    #[serde(default = "default_cage_mass")]
    pub cage_mass_kg: f64,
    /// The gear the photo was taken in (0 = the smallest cog). In other
    /// gears the parallelogram moves the cage pivot along the line from the
    /// axle to the upper jockey by [`Derailleur::slant`] times the change in
    /// cog radius.
    #[serde(default)]
    pub photo_gear: usize,
    /// How much the parallelogram drops the cage toward the axle as it
    /// shifts, as a fraction of the change in cog radius (dimensionless).
    /// 0: a horizontal parallelogram (SRAM Eagle's X-Horizon), where the
    /// pivot only moves sideways and the offset upper jockey tracks the
    /// cassette as the cage turns. 1: a slant parallelogram that keeps the
    /// upper jockey's gap to the cog as drawn with the cage still.
    #[serde(default)]
    pub slant: f64,
}

fn default_cage_mass() -> f64 {
    0.05
}

impl Derailleur {
    /// The cage's lever arm for a force at the lower jockey's centre, m: the
    /// distance from the cage pivot, so a force pulled square to the cage
    /// (along the chain, which leaves the lower jockey about square to it)
    /// times this is the torque on the cage.
    pub fn lower_jockey_arm_m(&self) -> f64 {
        let d = [self.lower_jockey_m[0] - self.cage_pivot_m[0], self.lower_jockey_m[1] - self.cage_pivot_m[1]];
        (d[0] * d[0] + d[1] * d[1]).sqrt()
    }

    /// The spring's torque toward tension, N·m, at the cage turned `deg_forward`
    /// degrees from its drawn angle in the slack direction.
    pub fn spring_torque(&self, deg_forward: f64) -> f64 {
        let t = &self.spring_torque_nm;
        match t.len() {
            0 => 0.0,
            1 => t[0][1],
            n => {
                let i = (0..n - 1).find(|&i| deg_forward <= t[i + 1][0]).unwrap_or(n - 2);
                let (a, b) = (t[i], t[i + 1]);
                let span = (b[0] - a[0]).abs().max(1e-9);
                a[1] + (b[1] - a[1]) * (deg_forward - a[0]) / span
            }
        }
    }
}

/// Where the photo the points were clicked on is, and its frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrivetrainPhoto {
    /// The photo's file name beside the bike file (`bikes/<file>`).
    pub file: String,
    /// The photo's scale and axes.
    pub frame: PhotoFrame,
}

/// A bike's drivetrain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Drivetrain {
    /// When this record last changed, UTC ms; the newer copy wins a merge.
    pub updated_at_ms: i64,
    /// Bottom bracket (chainring centre), metres, on the front triangle.
    pub bb_m: P2,
    /// Chainring teeth.
    pub chainring_teeth: u32,
    /// Cassette teeth, smallest cog first.
    pub cassette_teeth: Vec<u32>,
    /// Chain length, links.
    pub chain_links: u32,
    /// Chain mass, kg.
    pub chain_mass_kg: f64,
    /// The rear derailleur.
    pub derailleur: Derailleur,
    /// Rollers and idlers, in any order.
    #[serde(default)]
    pub rollers: Vec<Roller>,
    /// Surfaces the chain can hit.
    #[serde(default)]
    pub contacts: Vec<ContactSurface>,
    /// Cassette and freehub body inertia about the axle, kg·m².
    pub cassette_inertia_kgm2: f64,
    /// Freehub engagement angle (360° / points of engagement), degrees.
    pub freehub_engagement_deg: f64,
    /// Freehub drag torque while the wheel overruns the cassette, N·m.
    pub freehub_drag_nm: f64,
    /// Rear wheel rolling radius, metres.
    pub wheel_radius_m: f64,
    /// Crank length, metres (pedal force from top-span tension).
    pub crank_length_m: f64,
    /// The photo the points were clicked on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub photo: Option<DrivetrainPhoto>,
}

/// Pitch radius of a sprocket with `teeth` teeth, metres.
pub fn pitch_radius(teeth: u32) -> f64 {
    CHAIN_PITCH_M / (2.0 * (std::f64::consts::PI / teeth.max(3) as f64).sin())
}

impl Drivetrain {
    /// A starting drivetrain for a bike: the derailleur placed below the
    /// axle and the BB where the caller gives it, and parameters typical of a
    /// 12-speed 1× clutch drivetrain. A placeholder until the points are
    /// clicked on the bike's photo; every number is an estimate to be measured.
    pub fn typical(bb_m: P2, now_ms: i64) -> Drivetrain {
        Drivetrain {
            updated_at_ms: now_ms,
            bb_m,
            chainring_teeth: 32,
            cassette_teeth: vec![10, 12, 14, 16, 18, 21, 24, 28, 32, 36, 42, 52],
            chain_links: 120,
            chain_mass_kg: 0.257,
            derailleur: Derailleur {
                cage_pivot_m: [0.0, -0.048],
                upper_jockey_m: [0.008, -0.058],
                lower_jockey_m: [0.0, -0.22],
                jockey_teeth: [12, 12],
                // a long coil: constant moment, 12 N back at the lower jockey
                // (172 mm from the pivot); 27 N to pull it forward, so a 15 N
                // clutch; 1 mm of play there (a GT Force's SRAM, measured 2026-11-20)
                spring_torque_nm: vec![[0.0, 12.0 * 0.172], [60.0, 12.0 * 0.172]],
                clutch_torque_nm: 15.0 * 0.172,
                clutch_free_play_deg: (0.001_f64 / 0.172).to_degrees(),
                bearing_torque_nm: 0.01,
                cage_inertia_kgm2: 1.5e-4,
                jockey_inertia_kgm2: 3.0e-6,
                cage_mass_kg: 0.05,
                photo_gear: 0,
                // these guessed points were laid out for a slant parallelogram
                slant: 1.0,
            },
            rollers: Vec::new(),
            contacts: Vec::new(),
            cassette_inertia_kgm2: 5.0e-4,
            freehub_engagement_deg: 6.7,
            freehub_drag_nm: 0.1,
            wheel_radius_m: 0.37,
            crank_length_m: 0.17,
            photo: None,
        }
    }

    /// Chain length along the pin line, metres.
    pub fn chain_length_m(&self) -> f64 {
        self.chain_links as f64 * CHAIN_PITCH_M
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_radius_of_a_32_tooth_ring_is_64_78_mm() {
        // Arrange
        let teeth = 32;

        // Act
        let r = pitch_radius(teeth);

        // Assert: 12.7 / (2 sin(pi/32))
        assert!((r - 0.064_78).abs() < 1e-5, "{r}");
    }

    #[test]
    fn spring_torque_interpolates_between_points_and_extends_past_the_ends() {
        // Arrange
        let mut d = Drivetrain::typical([0.43, 0.02], 0).derailleur;
        d.spring_torque_nm = vec![[0.0, 0.45], [60.0, 0.9]];

        // Act
        let (at0, mid, past) = (d.spring_torque(0.0), d.spring_torque(30.0), d.spring_torque(120.0));

        // Assert
        assert!((at0 - 0.45).abs() < 1e-12);
        assert!((mid - 0.675).abs() < 1e-12);
        assert!((past - 1.35).abs() < 1e-12);
    }

    #[test]
    fn drivetrain_round_trips_through_json() {
        // Arrange
        let mut d = Drivetrain::typical([0.43, 0.02], 7);
        d.rollers.push(Roller {
            name: "Top roller".into(),
            body: Body::Chainstay,
            centre_m: [0.437, 0.08],
            teeth: Some(16),
            radius_m: 0.0,
            span: Span::Top,
            side: ChainSide::Over,
            inertia_kgm2: 2e-6,
            enabled: true,
            threaded: true,
        });

        // Act
        let back: Drivetrain = serde_json::from_slice(&serde_json::to_vec(&d).unwrap()).unwrap();

        // Assert
        assert_eq!(back, d);
    }
}
