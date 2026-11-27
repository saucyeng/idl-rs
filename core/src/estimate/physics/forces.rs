//! The force model's rows of the objective (spec `2026-11-25-idl1-force-model-DRAFT.md`, phase
//! F1): Newton per body beside the pod rows and the navigation rows.
//!
//! Accelerometers read specific force, `f = a - g`, so with gravity moved to the left every
//! balance is `m f = sum of contact forces` and its moment form: no attitude is needed. The
//! bodies:
//!
//! - the front and rear unsprung masses, whose specific force is what their pods read (each
//!   pod sits on its unsprung body; the pod stands for the body's centre of mass);
//! - the sprung frame, its specific force carried from the frame pod's to its centre of mass;
//! - the rider, one rigid body on a 6-axis spring and damper (the legs and arms together) with
//!   an active wrench (pumping, weight shifts, leaning the bike), its pose a latent.
//!
//! The whole-system balance (bike and rider) has only the two tyre forces outside it, so it
//! needs no joint forces. Per tyre: a normal force from the radial law ([`crate::sim::laws`])
//! at the deflection latent, along the bike's up axis, and tangential and lateral forces as
//! latents. The ground's slope (the tilt of the normal) arrives with the ground in F4.
//!
//! Rows per sample ([`NFR`]):
//!
//! | rows | term |
//! |---|---|
//! | 0 | front unsprung along the fork axis: fork spring, damper, friction, stop; the tyre |
//! | 1 | rear unsprung along the axle path (virtual work: the shock through its stroke rate) |
//! | 2..5 | rider force, bike axes (fwd, lat, up) |
//! | 5..8 | rider moment about its centre of mass |
//! | 8..11 | whole system force |
//! | 11..14 | whole system moment about the frame pod, with the wheels' spin |
//! | 14..18 | tyre tangential and lateral priors |
//! | 18..24 | rider pose priors |
//! | 24..30 | rider active wrench priors |
//! | 30..32 | tyre deflection priors (about the static deflection) |
//! | 32..35 | the head's moment about its centre of mass (F2; 0 without a head) |
//! | 35..41 | the helmet camera's accelerometer and gyro (F2; 0 without a camera sample) |
//! | 41..47 | the head's rotation and active neck torque priors |
//! | 47..49 | the ground tie (F4): each tyre's patch height against the ground profile, front, rear |
//!
//! The parameters ([`NFP`]) are globals, each a log multiplier on a nominal value (positive
//! by construction, one prior width for factors): both dampers' curves, seal friction, the
//! air springs' pressure, the tyres' stiffness and damping, the rider's springs and dampers.
//! Masses, inertias and geometry are fixed (D3).
//!
//! Simplifications of F1, each to replace when the data asks: the frame pod's flex is left
//! out (its specific force stands for the chassis's, as in the navigation rows); the pod
//! biases are left out of these rows (0.5 m/s^2 on a 4 kg body is 2 N against a 20 N
//! sigma); the unsprung bodies' rotation other than the wheels' spin is left out; the
//! sigmas are fixed (not proportional to the force yet).

use num_dual::Dual64;

use super::bike::{Bike, B_FA, B_LAT, D, F_TWIST, NL, PSI, R_ROLL, R_YAW, S};
use super::la::{c, exp, M3, Real, V3};
use super::readings::{Sample, NZ};
use crate::sim::laws::{air_force_about_n, friction_n, relu, step, stop_n, DamperLaw, TyreLaw};

/// Force latents per sample.
pub const NFL: usize = 24;
/// Front tyre radial deflection, m.
pub const DEF_F: usize = 0;
/// Front tyre tangential force along the bike's forward axis, N (+ pushes the bike forward).
pub const TAN_F: usize = 1;
/// Front tyre lateral force along the bike's lateral axis, N.
pub const LAT_F: usize = 2;
/// Rear tyre radial deflection, m.
pub const DEF_R: usize = 3;
/// Rear tyre tangential force, N.
pub const TAN_R: usize = 4;
/// Rear tyre lateral force, N.
pub const LAT_R: usize = 5;
/// Rider position against its neutral place, bike axes (fwd, lat, up), m: latents 6..9.
pub const POS: usize = 6;
/// Rider rotation against neutral, bike axes, rad (small): latents 9..12.
pub const ROT: usize = 9;
/// Rider's active force on himself from the bike, bike axes, N: latents 12..15.
pub const ACT_F: usize = 12;
/// Rider's active moment, bike axes, N m: latents 15..18.
pub const ACT_M: usize = 15;
/// The head's rotation against the rider's body (the neck), bike axes, rad (small): 18..21.
pub const HEAD: usize = 18;
/// The rider's active neck torque on his head, bike axes, N m: latents 21..24.
pub const NECK_ACT: usize = 21;

/// Force parameters (globals), each a log multiplier on [`ForceBike::nominal`].
pub const NFP: usize = 49;
/// Fork damper (LSC, HSC, LSR, HSR, knee): parameters 0..5.
pub const P_FORK_DAMP: usize = 0;
/// Shock damper, at the shaft: parameters 5..10.
pub const P_SHOCK_DAMP: usize = 5;
/// Seal friction, fork then shock: 10..12.
pub const P_FRICTION: usize = 10;
/// Air spring pressure scale, fork then shock: 12..14.
pub const P_PRESSURE: usize = 12;
/// Tyre stiffness scale, front then rear: 14..16.
pub const P_TYRE_K: usize = 14;
/// Tyre radial damping, front then rear: 16..18.
pub const P_TYRE_C: usize = 16;

/// Width the tyre's normal force is smoothed over at zero, N (a softplus: an airborne tyre
/// keeps w ln 2, 0.7 N).
pub const TYRE_SMOOTH_N: f64 = 1.0;
/// Rider springs (x, y, z, roll, pitch, yaw): 18..24.
pub const P_RIDER_K: usize = 18;
/// Rider dampers: 24..30.
pub const P_RIDER_C: usize = 24;
/// Air chamber volumes (fork positive, shock positive, fork negative, shock negative): 30..34,
/// factors on the bike file's (the spring's progression and its top-out, not only its force).
pub const P_VOLUME: usize = 30;
/// The rider's rest pose under braking (back, down), m per g of deceleration: 34..36 (Isaac,
/// 2026-11-27: he shifts back under braking, a little down, much more back than down).
pub const P_BRAKE_SHIFT: usize = 34;
/// The rider's limb mass the frame carries, a factor on [`ForceBike::m_limbs`]: 36 (the share
/// moves the fitted dampers, so it is fitted rather than guessed).
pub const P_LIMBS: usize = 36;
/// The neck's spring per axis (bike axes), a factor on 20 N m/rad: 37..40.
pub const P_NECK_K: usize = 37;
/// The neck's damper per axis, a factor on 1 N m s/rad: 40..43.
pub const P_NECK_C: usize = 40;
/// The helmet camera's mount: a small rotation (rad, additive, not a factor) on the per-clip
/// estimate its readings were turned by: 43..46.
pub const P_MOUNT: usize = 43;
/// The helmet camera's gyro bias, rad/s (additive): 46..49.
pub const P_CAM_BIAS: usize = 46;

/// Whether force parameter `i` is additive (its value used as it is) rather than a log
/// multiplier on a nominal value.
pub fn additive(i: usize) -> bool {
    (P_MOUNT..P_CAM_BIAS + 3).contains(&i)
}

/// The value parameter `i` prints as: a factor on the nominal, or an additive value.
pub fn par_display(i: usize, x: f64) -> f64 {
    if additive(i) { x } else { x.exp() }
}

/// Each force parameter's name, in index order.
pub const PAR_NAMES: [&str; NFP] = [
    "fork LSC", "fork HSC", "fork LSR", "fork HSR", "fork knee",
    "shock LSC", "shock HSC", "shock LSR", "shock HSR", "shock knee",
    "fork friction", "shock friction", "fork pressure", "shock pressure",
    "front tyre stiffness", "rear tyre stiffness", "front tyre damping", "rear tyre damping",
    "rider k fwd", "rider k lat", "rider k up", "rider k roll", "rider k pitch", "rider k yaw",
    "rider c fwd", "rider c lat", "rider c up", "rider c roll", "rider c pitch", "rider c yaw",
    "fork positive volume", "shock positive volume", "fork negative volume", "shock negative volume",
    "rider back under braking", "rider down under braking", "limbs on the frame",
    "neck k roll", "neck k pitch", "neck k yaw", "neck c roll", "neck c pitch", "neck c yaw",
    "camera mount roll", "camera mount pitch", "camera mount yaw", "camera gyro bias x", "camera gyro bias y", "camera gyro bias z",
];

/// Rows per sample (see the module table).
pub const NFR: usize = 49;
/// The balance rows (0..NBAL), the ones Huber weighting applies to.
pub const NBAL: usize = 14;
/// The head's balance and the camera's rows (32..41), Huber-weighted like the balances.
pub const CAM_ROWS: (usize, usize) = (32, 41);
/// The ground tie's rows (F4), front then rear: Huber-weighted, a line change or a rock that
/// moved costs a capped amount (Isaac, 2026-11-27: the tie is soft).
pub const TIE_ROWS: (usize, usize) = (47, 49);

/// Whether force row `r` is Huber-weighted (a balance, a camera or a ground-tie row, not a prior).
pub fn huber_row(r: usize) -> bool {
    r < NBAL || (CAM_ROWS.0..CAM_ROWS.1).contains(&r) || (TIE_ROWS.0..TIE_ROWS.1).contains(&r)
}

/// The rear's anti-rise at rear travel `s_m`, percent: how much of the extension the load
/// transfer of a rear-wheel stop would cause the brake's force (its pull at the contact patch
/// through the axle's path, and its torque on the axle's member) holds back; 100: the rear
/// neither rises nor squats. From the bike's centre of mass (frame and rider) over the rear
/// contact patch and its wheelbase.
pub fn anti_rise_pct(m: &ForceModel, bike: &Bike<f64>, s_m: f64) -> f64 {
    let fb = &m.bike;
    let mut q = [0.0; NL];
    q[S] = s_m;
    let k = kin(fb, bike, &q);
    let (_, dphi) = poly(&fb.axle_member_angle, s_m);
    let r = fb.wheel_radius[1];
    let mass = fb.m_frame + fb.m_rider;
    let cg = (fb.frame_cg.scale(fb.m_frame) + fb.rider_cg.scale(fb.m_rider)).scale(1.0 / mass);
    let h = (cg - fb.rear_axle).dot(fb.up) + r;
    let wb = (fb.front_axle - fb.rear_axle).dot(fb.fwd);
    100.0 * -(k.dar.dot(fb.fwd) + r * dphi) / (h / wb * k.dar.dot(fb.up))
}

/// A polynomial (lowest power first) and its derivative at `s`.
pub fn poly<S: Real>(coef: &[f64], s: S) -> (S, S) {
    let (mut x, mut dx) = (c::<S>(0.0), c::<S>(0.0));
    for (k, &a) in coef.iter().enumerate().rev() {
        x = x * s + a;
        if k > 0 {
            dx = dx * s + a * k as f64;
        }
    }
    (x, dx)
}

/// An air spring's chambers and fill: [`crate::sim::suspension::AirSpring`]'s law, fast
/// (adiabatic) about the stroke it rests at.
#[derive(Clone, Debug)]
pub struct AirChambers {
    /// Air piston area, m^2.
    pub area_m2: f64,
    /// Positive chamber at top-out, spacers out, m^3.
    pub positive_m3: f64,
    /// Negative chamber at top-out, m^3.
    pub negative_m3: f64,
    /// Gauge pressure, Pa.
    pub gauge_pa: f64,
    /// Stroke it rests at (isothermal there: sag), m.
    pub rest_m: f64,
}

impl AirChambers {
    /// The spring's force, N, at stroke `x_m` with the pressure scaled by `scale` and the
    /// chambers' volumes by `vol` (positive, negative), fast about [`AirChambers::rest_at`].
    pub fn force_n<S: Real>(&self, scale: S, vol: [S; 2], x_m: S) -> S {
        let (vp, vn) = (vol[0] * self.positive_m3, vol[1] * self.negative_m3);
        air_force_about_n(self.area_m2, vp, vn, scale * self.gauge_pa, self.rest_at(scale, vol), x_m, 1.4)
    }

    /// The stroke it rests at, m, with the pressure scaled by `scale`: where the slow
    /// (isothermal) spring carries the static load it carries at `rest_m` unscaled (the same
    /// rider and bike on a softer or firmer spring sit deeper or higher; the leverage's change
    /// with the rest is left out). The same with the chambers' volumes scaled by `vol`.
    /// Newton from `rest_m`, so unit factors give `rest_m`.
    pub fn rest_at<S: Real>(&self, scale: S, vol: [S; 2]) -> S {
        let (a, vp0, vn0) = (self.area_m2, self.positive_m3, self.negative_m3);
        let load = (self.gauge_pa + crate::sim::suspension::ATM_PA) * a * (vp0 / (vp0 - a * self.rest_m) - vn0 / (vn0 + a * self.rest_m));
        let (vp, vn) = (vol[0] * vp0, vol[1] * vn0);
        let p0 = scale * self.gauge_pa + crate::sim::suspension::ATM_PA;
        let (lo, hi) = (0.0, 0.95 * vp.re() / a);
        let mut x: S = c(self.rest_m);
        for _ in 0..6 {
            let (pos, neg) = (-(x * a) + vp, x * a + vn);
            let f = p0 * a * (pos.recip() * vp - neg.recip() * vn) - load;
            let df = p0 * a * a * (pos.recip() * pos.recip() * vp + neg.recip() * neg.recip() * vn);
            x = x - f / df;
            if x.re() < lo || x.re() > hi {
                return c(self.rest_m.clamp(lo, hi));
            }
        }
        x
    }
}

/// The rider's head on his body (F2): a ball joint at the neck, the helmet camera on the head.
#[derive(Clone, Debug)]
pub struct Head {
    /// Head, neck share and helmet with camera, kg.
    pub m_kg: f64,
    /// Radius of gyration about its centre of mass (a sphere), m.
    pub gyration_m: f64,
    /// The neck pivot from the rider body's centre of mass at its neutral pose, IMU0 axes, m.
    pub neck: V3<f64>,
    /// The head's centre of mass from the neck pivot, IMU0 axes, m.
    pub cg: V3<f64>,
    /// The helmet camera from the neck pivot, IMU0 axes, m.
    pub cam: V3<f64>,
}

/// The bike and rider as the force rows see them: everything fixed (D3), IMU0 axes, SI.
#[derive(Clone, Debug)]
pub struct ForceBike {
    /// Up (unit).
    pub up: V3<f64>,
    /// Forward (unit, perpendicular to up).
    pub fwd: V3<f64>,
    /// The direction the fork lowers move as the fork compresses (unit; the steering axis,
    /// pointing up the fork).
    pub fork_axis: V3<f64>,
    /// Front axle at zero latents, m.
    pub front_axle: V3<f64>,
    /// Rear axle at zero latents, m.
    pub rear_axle: V3<f64>,
    /// Sprung frame's centre of mass, m.
    pub frame_cg: V3<f64>,
    /// Rider's centre of mass in his neutral place, m.
    pub rider_cg: V3<f64>,
    /// Sprung frame (with the lower legs, which move with the pedals), kg.
    pub m_frame: f64,
    /// Front unsprung, kg.
    pub m_front: f64,
    /// Rear unsprung (reduced to the axle), kg.
    pub m_rear: f64,
    /// Rider's body, kg.
    pub m_rider: f64,
    /// The share of the rider's mass in `m_frame` (limbs moving with the bike), kg: scaled by
    /// [`P_LIMBS`], moved between the frame and the rider's body.
    pub m_limbs: f64,
    /// Frame inertia about its centre of mass, about (fwd, lat, up), kg m^2.
    pub i_frame: [f64; 3],
    /// Rider's inertia, about (fwd, lat, up), kg m^2.
    pub i_rider: [f64; 3],
    /// The head on the rider's body (F2; None: the head is part of the body).
    pub head: Option<Head>,
    /// Wheels' spin inertia, front and rear, kg m^2.
    pub wheel_inertia: [f64; 2],
    /// Wheels' rolling radius, m.
    pub wheel_radius: [f64; 2],
    /// Fork air spring (stroke = front travel).
    pub fork: AirChambers,
    /// Shock air spring (stroke = shock shortening).
    pub shock: AirChambers,
    /// Shock stroke against rear travel, polynomial (m -> m), lowest power first.
    pub shock_stroke: Vec<f64>,
    /// The rear axle's member's angle against rear travel, polynomial (m -> rad, CCW with x
    /// forward and y up), lowest power first. The rear brake's caliper is taken to sit on that
    /// member (the GT Force's on its seatstay, which carries the axle): its torque reacts there.
    pub axle_member_angle: Vec<f64>,
    /// The derailleur cage's angle against rear travel in the descending gear, polynomial
    /// (m -> rad, forward: the slack direction), lowest power first; empty: no drivetrain.
    pub cage_angle: Vec<f64>,
    /// The cage's moments, N m: its spring (always) and its clutch (against turning forward).
    pub cage_moment_nm: [f64; 2],
    /// Front and rear travel at the stops, m.
    pub travel_max: [f64; 2],
    /// The travel stops' stiffness, N/m.
    pub stop_n_m: f64,
    /// Tyres' radial laws (stiffness at scale 1), front and rear.
    pub tyre: [TyreLaw<f64>; 2],
    /// Each parameter's value at multiplier 1 (units as the parameter: N s/m, m/s, N, 1, N/m, N m/rad, N m s/rad).
    pub nominal: [f64; NFP],
    /// Smoothing widths: speeds (dampers, friction), m/s.
    pub smooth_mps: f64,
    /// Smoothing widths: lengths (tyre contact, stops), m.
    pub smooth_m: f64,
}

impl ForceBike {
    /// Lateral axis (up x fwd... as fwd x up: the bike's left with x forward, z up).
    pub fn lat(&self) -> V3<f64> {
        self.up.cross(self.fwd).unit()
    }

    /// The shock's stroke (m) and its rate with rear travel at travel `s` (m).
    pub fn stroke<S: Real>(&self, s: S) -> (S, S) {
        poly(&self.shock_stroke, s)
    }
}

/// The active wrench's smoothness, times the core latents' ([`ForceSpec::smooth`]).
pub const ACTIVE_SMOOTH: f64 = 259.0;

/// How the force latents are laid out: knot spacing (samples), value at zero coefficients
/// (SI) and SI units per coefficient unit, per latent.
#[derive(Clone, Debug)]
pub struct ForceSpec {
    /// Knot spacing, samples.
    pub m: [usize; NFL],
    /// Value at zero coefficients.
    pub init: [f64; NFL],
    /// Units per coefficient unit.
    pub scale: [f64; NFL],
    /// Smoothness weight per latent, times the core latents' (a band limit without coarse
    /// knots: coarse knots touch many samples and widen the solver's band).
    pub smooth: [f64; NFL],
}

impl ForceSpec {
    /// Deflection every sample (the tyre hops at tens of Hz), tyre forces every 4 samples,
    /// the rider's pose every 2, his active wrench every 4 with a smoothness [`ACTIVE_SMOOTH`]
    /// times the core's (a low-pass near 5 Hz, Isaac 2026-11-27: he works the bars through his
    /// wrists up to about that; a fast active part could absorb what the dampers do, and the
    /// rider's stroke, [`ForceWeights::stroke_m`], is what bounds it). `static_deflection_m`: where
    /// each tyre starts.
    pub fn standard(static_deflection_m: [f64; 2]) -> Self {
        let mut m = [2; NFL];
        let mut init = [0.0; NFL];
        let mut scale = [0.01; NFL];
        let mut smooth = [1.0; NFL];
        for (w, d) in [(0, DEF_F), (1, DEF_R)] {
            m[d] = 1;
            init[d] = static_deflection_m[w];
            scale[d] = 0.001;
            m[d + 1] = 4;
            m[d + 2] = 4;
            scale[d + 1] = 100.0;
            scale[d + 2] = 100.0;
        }
        for i in 0..3 {
            m[NECK_ACT + i] = 4;
            smooth[NECK_ACT + i] = ACTIVE_SMOOTH;
            scale[NECK_ACT + i] = 1.0;
            m[ACT_F + i] = 4;
            m[ACT_M + i] = 4;
            smooth[ACT_F + i] = ACTIVE_SMOOTH;
            smooth[ACT_M + i] = ACTIVE_SMOOTH;
            scale[ACT_F + i] = 100.0;
            scale[ACT_M + i] = 10.0;
        }
        ForceSpec { m, init, scale, smooth }
    }
}

/// The force rows' noise levels and prior widths.
#[derive(Clone, Debug)]
pub struct ForceWeights {
    /// Unsprung balance rows, N.
    pub unsprung_n: f64,
    /// Rider force rows, N.
    pub rider_n: f64,
    /// Rider moment rows, N m.
    pub rider_nm: f64,
    /// System force rows, N.
    pub system_n: f64,
    /// System moment rows, N m.
    pub system_nm: f64,
    /// Tyre tangential and lateral prior, N.
    pub tyre_tan_n: f64,
    /// Rider position prior, bike axes, m.
    pub pos_m: [f64; 3],
    /// Rider rotation prior, rad.
    pub rot_rad: f64,
    /// Active force prior, N.
    pub act_n: f64,
    /// Active moment prior, N m.
    pub act_nm: f64,
    /// The head's moment rows, N m.
    pub head_nm: f64,
    /// The helmet camera's accelerometer rows, m/s^2, and gyro rows, rad/s.
    pub cam: [f64; 2],
    /// The head's rotation prior against the body, rad, and the active neck torque's, N m.
    pub head_prior: [f64; 2],
    /// The rider's stroke each way from his neutral pose, m: soft stops past it (Isaac,
    /// 2026-11-27: arms and legs travel about 0.33 m; the body's actuators bounded, not weak).
    pub stroke_m: f64,
    /// Tyre deflection prior about its static value, m: keeps a solve from parking a tyre in
    /// rim strike or far off the ground to soak up a misfit.
    pub defl_m: f64,
    /// Each parameter's prior width, log units (1: a factor of e).
    pub par_sigma: [f64; NFP],
    /// Huber threshold on the balance rows, sigmas (0: plain least squares).
    pub huber: f64,
    /// The ground tie (F4; None: off, the default until navigation's heights carry it: on
    /// Beacon and 11/21 the tie's sigma comes to 30-80 mm and the patches sit 1-3 of it apart).
    pub ground: Option<super::ground::GroundWeights>,
}

impl Default for ForceWeights {
    fn default() -> Self {
        let mut par_sigma = [1.0; NFP];
        par_sigma[P_PRESSURE] = 0.05;
        par_sigma[P_PRESSURE + 1] = 0.05;
        par_sigma[P_TYRE_K] = 0.3;
        par_sigma[P_TYRE_K + 1] = 0.3;
        // chamber volumes: the bike file's are estimates (no sag tests); a third either way
        par_sigma[P_VOLUME..P_VOLUME + 4].copy_from_slice(&[0.3; 4]);
        // the limbs' share: tight until the helmet (F2) sees the rider; free, it runs to 0 and
        // takes the fork's damping with it (2026-11-27: fork LSR 0.00 on 11/21)
        par_sigma[P_LIMBS] = 0.2;
        // the camera's mount correction, rad, and its gyro bias, rad/s (additive)
        par_sigma[P_MOUNT..P_MOUNT + 3].copy_from_slice(&[0.1; 3]);
        par_sigma[P_CAM_BIAS..P_CAM_BIAS + 3].copy_from_slice(&[0.02; 3]);
        ForceWeights {
            unsprung_n: 20.0,
            rider_n: 50.0,
            rider_nm: 20.0,
            system_n: 50.0,
            system_nm: 20.0,
            tyre_tan_n: 500.0,
            // Isaac, 2026-11-25: he shifts fore and aft about as much as up and down
            pos_m: [0.25, 0.2, 0.25],
            rot_rad: 0.3,
            act_n: 300.0,
            stroke_m: 0.165,
            head_nm: 2.0,
            cam: [2.0, 0.1],
            head_prior: [0.3, 5.0],
            act_nm: 60.0,
            defl_m: 0.02,
            par_sigma,
            huber: 3.0,
            ground: None,
        }
    }
}

/// The force model of one bike and rider: what the objective's force rows need.
#[derive(Clone, Debug)]
pub struct ForceModel {
    /// Bike and rider.
    pub bike: ForceBike,
    /// Latent layout.
    pub spec: ForceSpec,
    /// Weights.
    pub w: ForceWeights,
}

/// A window's extra measurements for the force rows.
#[derive(Clone, Debug)]
pub struct ForceWindow {
    /// Ground speed per sample (GPS), m/s: the wheels' spin.
    pub speed_mps: Vec<f64>,
    /// The helmet camera's accelerometer (m/s^2) and gyro (rad/s) per sample, turned by its
    /// clip's mount estimate into the frame pod's axes at the rider's neutral pose (empty, or
    /// None at a sample: no camera there).
    pub cam: Vec<Option<[f64; 6]>>,
    /// The ground tie's inputs from the window's navigation solve (F4; None: no tie).
    pub ground: Option<super::ground::GroundWindow>,
}

/// One sample's measurements the force rows read.
#[derive(Clone, Copy, Debug)]
pub struct ForceMeas {
    /// Frame pod specific force, its axes (as the chassis's), m/s^2.
    pub frame: [f64; 3],
    /// Fork pod specific force, its axes, m/s^2.
    pub fork: [f64; 3],
    /// Rear pod specific force, its axes, m/s^2.
    pub rear: [f64; 3],
    /// Ground speed, m/s.
    pub speed: f64,
    /// Its rate, m/s^2.
    pub speed_rate: f64,
    /// The helmet camera's reading here ([`ForceWindow::cam`]).
    pub cam: Option<[f64; 6]>,
    /// The ground tie's inputs here (F4; None: no tie).
    pub ground: Option<GroundMeas>,
}

/// One sample's ground-tie inputs (F4): from the window's navigation solve, and the ground
/// profile at the current iterate.
#[derive(Clone, Copy, Debug)]
pub struct GroundMeas {
    /// The world's up in IMU0 axes (unit).
    pub up: [f64; 3],
    /// The frame pod's height, window-local, m.
    pub z_m: f64,
    /// The ground profile's height under the front and the rear patch, m.
    pub h_m: [f64; 2],
    /// The tie's sigma per tyre, m (infinite: that tyre's row off).
    pub sigma_m: [f64; 2],
}

/// The kinematics the rows need from the core latents: pod rotations (pod -> IMU0), axle
/// positions, the rear axle's path and the shock's stroke.
#[derive(Clone, Copy, Debug)]
pub struct Kin<S> {
    /// Fork pod rotation.
    pub rf: M3<S>,
    /// Rear pod rotation.
    pub rr: M3<S>,
    /// Front axle, m.
    pub af: V3<S>,
    /// Rear axle, m.
    pub ar: V3<S>,
    /// Rear axle's rate with rear travel (m/m).
    pub dar: V3<S>,
    /// Shock stroke, m.
    pub x: S,
    /// Its rate with rear travel (m/m).
    pub dx: S,
}

impl<S: Real> Kin<S> {
    fn map<T: Real>(&self, f: impl Fn(S) -> T + Copy) -> Kin<T> {
        Kin { rf: self.rf.map(f), rr: self.rr.map(f), af: self.af.map(f), ar: self.ar.map(f), dar: self.dar.map(f), x: f(self.x), dx: f(self.dx) }
    }
}

/// The kinematics at latents `q`.
pub fn kin<S: Real>(fb: &ForceBike, bike: &Bike<S>, q: &[S; NL]) -> Kin<S> {
    let lift = |v: V3<f64>| v.map(c::<S>);
    let rf = bike.front_pose(q).0;
    let af = bike.front_at(q, lift(fb.front_axle)).1;
    let rr = bike.rear_pose(q).0;
    let ar = bike.rear_at(q, lift(fb.rear_axle)).1;
    let h = 1e-5;
    let (mut qp, mut qm) = (*q, *q);
    qp[S] = q[S] + h;
    qm[S] = q[S] - h;
    let dar = (bike.rear_at(&qp, lift(fb.rear_axle)).1 - bike.rear_at(&qm, lift(fb.rear_axle)).1).scale(c(0.5 / h));
    let (x, dx) = fb.stroke(q[S]);
    Kin { rf, rr, af, ar, dar, x, dx }
}

/// The rows' inputs other than the kinematics.
#[derive(Clone, Copy, Debug)]
pub struct ForceIn<S> {
    /// Front travel, m, and its rate, m/s.
    pub d: [S; 2],
    /// Rear travel, m, and its rate, m/s.
    pub s: [S; 2],
    /// Chassis angular rate, rad/s.
    pub w0: V3<S>,
    /// Chassis angular acceleration, rad/s^2.
    pub a0: V3<S>,
    /// Force latents.
    pub f: [S; NFL],
    /// Their rates.
    pub fd: [S; NFL],
    /// Their second derivatives.
    pub fdd: [S; NFL],
    /// Force parameters (log multipliers).
    pub par: [S; NFP],
}

impl<S: Real> ForceIn<S> {
    fn map<T: Real>(&self, f: impl Fn(S) -> T + Copy) -> ForceIn<T> {
        ForceIn {
            d: self.d.map(f),
            s: self.s.map(f),
            w0: self.w0.map(f),
            a0: self.a0.map(f),
            f: self.f.map(f),
            fd: self.fd.map(f),
            fdd: self.fdd.map(f),
            par: self.par.map(f),
        }
    }
}

/// Every value the rows compute on the way, in SI and IMU0 axes (for outputs and tests).
#[derive(Clone, Copy, Debug)]
pub struct Forces<S> {
    /// Front and rear tyre force vectors, N.
    pub tyre: [V3<S>; 2],
    /// Front and rear tyre normal forces, N.
    pub normal: [S; 2],
    /// Fork force along its axis (+ compressing), N.
    pub fork: S,
    /// Shock force at the shaft (+ compressing), N.
    pub shock: S,
    /// Bushing force and moment on the rider from the bike (passive), N, N m.
    pub bush: [V3<S>; 2],
    /// Each patch's height in the world, window-local, front and rear, m (0 without the ground
    /// tie's inputs).
    pub patch_z: [S; 2],
}

/// The rows (whitened) and the forces they used.
pub fn rows<S: Real>(m: &ForceModel, k: &Kin<S>, x: &ForceIn<S>, meas: &ForceMeas) -> ([S; NFR], Forces<S>) {
    let (fb, w) = (&m.bike, &m.w);
    let lift = |v: V3<f64>| v.map(c::<S>);
    let (up, fwd, lat, axis) = (lift(fb.up), lift(fb.fwd), lift(fb.lat()), lift(fb.fork_axis));
    let bv = |a: S, b: S, cc: S| fwd.scale(a) + lat.scale(b) + up.scale(cc);
    let comps = |v: V3<S>| [v.dot(fwd), v.dot(lat), v.dot(up)];
    let inertia = |i: [f64; 3], v: V3<S>| fwd.scale(v.dot(fwd) * i[0]) + lat.scale(v.dot(lat) * i[1]) + up.scale(v.dot(up) * i[2]);
    let par = |i: usize| x.par[i].exp() * fb.nominal[i];
    let damper = |p0: usize| DamperLaw { lsc_ns_m: par(p0), hsc_ns_m: par(p0 + 1), lsr_ns_m: par(p0 + 2), hsr_ns_m: par(p0 + 3), knee_mps: par(p0 + 4) };
    let (wv, wl) = (fb.smooth_mps, fb.smooth_m);
    let f = &x.f;
    let (fd, fdd) = (&x.fd, &x.fdd);
    let (w0, a0) = (x.w0, x.a0);
    let f0 = V3::<S>::from_f64(meas.frame);
    // tyres
    let tyre = |i: usize, d0: usize| {
        let t = &fb.tyre[i];
        let law = TyreLaw { stiffness_n_m: par(P_TYRE_K + i) * t.stiffness_n_m, insert_touch_m: t.insert_touch_m, insert_n_m: c::<S>(t.insert_n_m), insert_solid_m: t.insert_solid_m, rim_m: t.rim_m, rim_n_m: t.rim_n_m };
        // a tyre pushes, never pulls: the spring and damping together kept >= 0 (in fast rebound
        // the damping term alone would take the total below zero)
        let n = relu(law.force_n(f[d0], wl) + par(P_TYRE_C + i) * fd[d0] * step(f[d0], wl), TYRE_SMOOTH_N);
        (n, up.scale(n) + fwd.scale(f[d0 + 1]) + lat.scale(f[d0 + 2]))
    };
    let (nf, tf) = tyre(0, DEF_F);
    let (nr, tr) = tyre(1, DEF_R);
    let mut r = [c::<S>(0.0); NFR];
    // front unsprung along the fork axis
    let fork = fb.fork.force_n(par(P_PRESSURE), [par(P_VOLUME), par(P_VOLUME + 2)], x.d[0]) + damper(P_FORK_DAMP).force_n(x.d[1], wv) + friction_n(par(P_FRICTION), x.d[1], wv) + stop_n(x.d[0], 0.0, fb.travel_max[0], fb.stop_n_m, wl);
    let uf = k.rf * V3::<S>::from_f64(meas.fork);
    r[0] = (uf.dot(axis) * fb.m_front - tf.dot(axis) + fork) * (1.0 / w.unsprung_n);
    // rear unsprung along its path: generalised force on the travel coordinate
    let xd = k.dx * x.s[1];
    let shock = fb.shock.force_n(par(P_PRESSURE + 1), [par(P_VOLUME + 1), par(P_VOLUME + 3)], k.x) + damper(P_SHOCK_DAMP).force_n(xd, wv) + friction_n(par(P_FRICTION + 1), xd, wv);
    // the derailleur's cage takes up the chain's growth: its spring always, its clutch one way
    // (resisting the cage turning forward, which compression does in the descending gear)
    let chain = if fb.cage_angle.is_empty() {
        c::<S>(0.0)
    } else {
        let (_, dcage) = poly(&fb.cage_angle, x.s[0]);
        dcage * (step(dcage * x.s[1], wv) * fb.cage_moment_nm[1] + fb.cage_moment_nm[0])
    };
    let wheel = shock * k.dx + chain + stop_n(x.s[0], 0.0, fb.travel_max[1], fb.stop_n_m, wl);
    // the rear brake's torque on the axle's member: braking pulls back at the contact patch, a
    // wheel-radius below the axle, and the caliper holds that moment against the member (the
    // tyre's force itself acts at the axle); a forward force is the chain's, not the brake's
    let (_, dphi) = poly(&fb.axle_member_angle, x.s[0]);
    let brake = -(relu(-tr.dot(fwd), TYRE_SMOOTH_N) * (-f[DEF_R] + fb.wheel_radius[1])) * dphi;
    let ur = k.rr * V3::<S>::from_f64(meas.rear);
    r[1] = (ur.dot(k.dar) * fb.m_rear - tr.dot(k.dar) - brake + wheel) * (1.0 / w.unsprung_n);
    // rider
    let dp = bv(f[POS], f[POS + 1], f[POS + 2]);
    let dpd = bv(fd[POS], fd[POS + 1], fd[POS + 2]);
    let dpdd = bv(fdd[POS], fdd[POS + 1], fdd[POS + 2]);
    let thd = bv(fd[ROT], fd[ROT + 1], fd[ROT + 2]);
    let thdd = bv(fdd[ROT], fdd[ROT + 1], fdd[ROT + 2]);
    let pb = lift(fb.rider_cg) + dp;
    let f_rider = f0 + a0.cross(pb) + w0.cross(w0.cross(pb)) + w0.cross(dpd).scale(c(2.0)) + dpdd;
    let spring = |i: usize, q: usize| -(par(P_RIDER_K + i) * f[q + i] + par(P_RIDER_C + i) * fd[q + i]);
    let bush_f = bv(spring(0, POS), spring(1, POS), spring(2, POS));
    let bush_m = bv(spring(3, ROT - 3), spring(4, ROT - 3), spring(5, ROT - 3));
    let act_f = bv(f[ACT_F], f[ACT_F + 1], f[ACT_F + 2]);
    let act_m = bv(f[ACT_M], f[ACT_M + 1], f[ACT_M + 2]);
    // the limbs' share between the frame and the body, fitted
    let moved = (par(P_LIMBS) - 1.0) * fb.m_limbs;
    let (m_frame, m_rider) = (moved + fb.m_frame, -moved + fb.m_rider);
    let wb = w0 + thd;
    let ab = a0 + thdd + w0.cross(thd);
    let m_rider_in = inertia(fb.i_rider, ab) + wb.cross(inertia(fb.i_rider, wb));
    // the head (F2): a ball joint at the neck on the body; the neck's force on the head is its
    // mass times its specific force, its torque a spring and damper plus an active part
    let sf_at = |p: V3<S>, v: V3<S>, a: V3<S>| f0 + a0.cross(p) + w0.cross(w0.cross(p)) + w0.cross(v).scale(c(2.0)) + a;
    let mut neck_on_body = (V3::<S>::zero(), V3::<S>::zero());
    let mut head_sys = (V3::<S>::zero(), V3::<S>::zero());
    if let Some(h) = &fb.head {
        let th_b = bv(f[ROT], f[ROT + 1], f[ROT + 2]);
        let (th_h, thd_h, thdd_h) = (bv(f[HEAD], f[HEAD + 1], f[HEAD + 2]), bv(fd[HEAD], fd[HEAD + 1], fd[HEAD + 2]), bv(fdd[HEAD], fdd[HEAD + 1], fdd[HEAD + 2]));
        let (rb, rh) = (exp(th_b), exp(th_h));
        let n0 = rb * lift(h.neck);
        let (pn, vn, an) = (pb + n0, dpd + thd.cross(n0), dpdd + thdd.cross(n0) + thd.cross(thd.cross(n0)));
        let (wr, ar) = (thd + thd_h, thdd + thdd_h);
        let point = |off: V3<f64>| {
            let o = rb * (rh * lift(off));
            (pn + o, vn + wr.cross(o), an + ar.cross(o) + wr.cross(wr.cross(o)))
        };
        let (ph, vh, ahh) = point(h.cg);
        let f_neck = sf_at(ph, vh, ahh).scale(c(h.m_kg));
        let w_h = w0 + wr;
        let a_h = a0 + ar + w0.cross(wr);
        let ih = h.m_kg * h.gyration_m * h.gyration_m;
        let neck = |i: usize| -(par(P_NECK_K + i) * f[HEAD + i] + par(P_NECK_C + i) * fd[HEAD + i]);
        let tau = bv(neck(0), neck(1), neck(2)) + bv(f[NECK_ACT], f[NECK_ACT + 1], f[NECK_ACT + 2]);
        let hm = comps(a_h.scale(c(ih)) - tau - (pn - ph).cross(f_neck));
        for i in 0..3 {
            r[32 + i] = hm[i] * (1.0 / w.head_nm);
        }
        neck_on_body = (f_neck, (pn - pb).cross(f_neck) + tau);
        head_sys = (f_neck, ph.cross(f_neck) + a_h.scale(c(ih)));
        if let Some(cm) = meas.cam {
            let (pc, vc, ac) = point(h.cam);
            let mount = exp(V3([x.par[P_MOUNT], x.par[P_MOUNT + 1], x.par[P_MOUNT + 2]]));
            let rc = (rb * rh * mount).t();
            let acc = rc * sf_at(pc, vc, ac);
            let gyr = rc * w_h + V3([x.par[P_CAM_BIAS], x.par[P_CAM_BIAS + 1], x.par[P_CAM_BIAS + 2]]);
            for i in 0..3 {
                r[35 + i] = (acc.0[i] - cm[i]) * (1.0 / w.cam[0]);
                r[38 + i] = (gyr.0[i] - cm[3 + i]) * (1.0 / w.cam[1]);
            }
        }
    }
    let rf_res = comps(f_rider.scale(m_rider) - bush_f - act_f + neck_on_body.0);
    let rm_res = comps(m_rider_in - bush_m - act_m + neck_on_body.1);
    // whole system
    let ps = lift(fb.frame_cg);
    let f_frame = f0 + a0.cross(ps) + w0.cross(w0.cross(ps));
    let sf = comps(f_frame.scale(m_frame) + uf.scale(c(fb.m_front)) + ur.scale(c(fb.m_rear)) + f_rider.scale(m_rider) + head_sys.0 - tf - tr);
    let spin = up.cross(fwd).unit();
    let wheel_h = |i: usize| {
        let h = spin.scale(c(fb.wheel_inertia[i] * meas.speed / fb.wheel_radius[i]));
        spin.scale(c(fb.wheel_inertia[i] * meas.speed_rate / fb.wheel_radius[i])) + w0.cross(h)
    };
    let pcf = k.af - up.scale(-f[DEF_F] + fb.wheel_radius[0]);
    let pcr = k.ar - up.scale(-f[DEF_R] + fb.wheel_radius[1]);
    let sm = comps(
        ps.cross(f_frame.scale(m_frame)) + k.af.cross(uf.scale(c(fb.m_front))) + k.ar.cross(ur.scale(c(fb.m_rear))) + pb.cross(f_rider.scale(m_rider))
            + inertia(fb.i_frame, a0) + w0.cross(inertia(fb.i_frame, w0)) + m_rider_in + head_sys.1 + wheel_h(0) + wheel_h(1)
            - pcf.cross(tf) - pcr.cross(tr),
    );
    for i in 0..3 {
        r[2 + i] = rf_res[i] * (1.0 / w.rider_n);
        r[5 + i] = rm_res[i] * (1.0 / w.rider_nm);
        r[8 + i] = sf[i] * (1.0 / w.system_n);
        r[11 + i] = sm[i] * (1.0 / w.system_nm);
    }
    // priors
    for (j, lat_) in [TAN_F, LAT_F, TAN_R, LAT_R].into_iter().enumerate() {
        r[14 + j] = f[lat_] * (1.0 / w.tyre_tan_n);
    }
    // the rider's rest pose slides back (and a little down) as the bike slows (GPS speed's rate)
    let decel_g = relu(c::<S>(-meas.speed_rate), 0.1) * (1.0 / 9.80665);
    let rest = [-(par(P_BRAKE_SHIFT) * decel_g), c::<S>(0.0), -(par(P_BRAKE_SHIFT + 1) * decel_g)];
    for i in 0..3 {
        // a soft stop past the stroke: a centimetre beyond it costs as much as a sigma
        let stop = (relu(f[POS + i] - w.stroke_m, 0.002) - relu(-f[POS + i] - w.stroke_m, 0.002)) * (1.0 / 0.01);
        r[18 + i] = (f[POS + i] - rest[i]) * (1.0 / w.pos_m[i]) + stop;
        r[21 + i] = f[ROT + i] * (1.0 / w.rot_rad);
        r[24 + i] = f[ACT_F + i] * (1.0 / w.act_n);
        r[27 + i] = f[ACT_M + i] * (1.0 / w.act_nm);
    }
    r[30] = (f[DEF_F] - m.spec.init[DEF_F]) * (1.0 / w.defl_m);
    r[31] = (f[DEF_R] - m.spec.init[DEF_R]) * (1.0 / w.defl_m);
    for i in 0..3 {
        r[41 + i] = f[HEAD + i] * (1.0 / w.head_prior[0]);
        r[44 + i] = f[NECK_ACT + i] * (1.0 / w.head_prior[1]);
    }
    // the ground tie (F4): each patch's height in the world against the ground profile there
    let mut patch_z = [c::<S>(0.0); 2];
    if let Some(g) = &meas.ground {
        let u = V3::<S>::from_f64(g.up);
        for (i, p) in [pcf, pcr].into_iter().enumerate() {
            patch_z[i] = p.dot(u) + g.z_m;
            if g.sigma_m[i].is_finite() {
                r[TIE_ROWS.0 + i] = (patch_z[i] - g.h_m[i]) * (1.0 / g.sigma_m[i]);
            }
        }
    }
    (r, Forces { tyre: [tf, tr], normal: [nf, nr], fork, shock, bush: [bush_f, bush_m], patch_z })
}

/// The force rows at one sample and their Jacobians.
pub struct ForceJac {
    /// Whitened residuals.
    pub r: [f64; NFR],
    /// d r / d core local inputs (q, q', q'', w0, a0; [`NZ`] columns).
    pub j_core: Box<[[f64; NZ]; NFR]>,
    /// d r / d force latents: value (column i), rate (NFL + i), second derivative (2 NFL + i).
    pub j_frc: Box<[[f64; 3 * NFL]; NFR]>,
    /// d r / d force parameters.
    pub j_par: [[f64; NFP]; NFR],
}

/// The core latents the force rows depend on (through the pod rotations and the axles).
const Q_USED: [usize; 8] = [D, PSI, B_FA, B_LAT, S, R_ROLL, R_YAW, F_TWIST];

/// The rows at one sample (`s`: its core inputs; `f`, `fd`, `fdd`: the force latents and
/// their derivatives; `par`: the force parameters) and, if `jac`, their Jacobians, by one
/// dual pass per input (the kinematics re-evaluated only for the core latents).
#[allow(clippy::too_many_arguments)]
pub fn force_rows(m: &ForceModel, bike: &Bike<f64>, bike_d: &Bike<Dual64>, s: &Sample<f64>, f: &[f64; NFL], fd: &[f64; NFL], fdd: &[f64; NFL], par: &[f64; NFP], meas: &ForceMeas, jac: bool) -> ForceJac {
    let k = kin(&m.bike, bike, &s.q);
    let x = ForceIn { d: [s.q[D], s.qd[D]], s: [s.q[S], s.qd[S]], w0: s.w0, a0: s.a0, f: *f, fd: *fd, fdd: *fdd, par: *par };
    let (r, _) = rows(m, &k, &x, meas);
    let mut out = ForceJac { r, j_core: Box::new([[0.0; NZ]; NFR]), j_frc: Box::new([[0.0; 3 * NFL]; NFR]), j_par: [[0.0; NFP]; NFR] };
    if !jac {
        return out;
    }
    let kc = k.map(Dual64::from);
    let xc = x.map(Dual64::from);
    let run = |kk: &Kin<Dual64>, xx: &ForceIn<Dual64>| rows(m, kk, xx, meas).0.map(|y| y.eps);
    let one = |v: f64| Dual64::new(v, 1.0);
    for j in Q_USED {
        let mut q = s.q.map(Dual64::from);
        q[j] = one(s.q[j]);
        let kj = kin(&m.bike, bike_d, &q);
        let mut xj = xc;
        if j == D {
            xj.d[0] = one(s.q[D]);
        }
        if j == S {
            xj.s[0] = one(s.q[S]);
        }
        let y = run(&kj, &xj);
        for r in 0..NFR {
            out.j_core[r][j] = y[r];
        }
    }
    for (j, which) in [(D, 0), (S, 1)] {
        let mut xj = xc;
        if which == 0 {
            xj.d[1] = one(s.qd[D]);
        } else {
            xj.s[1] = one(s.qd[S]);
        }
        let y = run(&kc, &xj);
        for r in 0..NFR {
            out.j_core[r][NL + j] = y[r];
        }
    }
    for i in 0..3 {
        let mut xj = xc;
        xj.w0.0[i] = one(s.w0.0[i]);
        let y = run(&kc, &xj);
        let mut xa = xc;
        xa.a0.0[i] = one(s.a0.0[i]);
        let ya = run(&kc, &xa);
        for r in 0..NFR {
            out.j_core[r][3 * NL + i] = y[r];
            out.j_core[r][3 * NL + 3 + i] = ya[r];
        }
    }
    for i in 0..NFL {
        for d in 0..3 {
            let mut xj = xc;
            match d {
                0 => xj.f[i] = one(f[i]),
                1 => xj.fd[i] = one(fd[i]),
                _ => xj.fdd[i] = one(fdd[i]),
            }
            let y = run(&kc, &xj);
            for r in 0..NFR {
                out.j_frc[r][d * NFL + i] = y[r];
            }
        }
    }
    for i in 0..NFP {
        let mut xj = xc;
        xj.par[i] = one(par[i]);
        let y = run(&kc, &xj);
        for r in 0..NFR {
            out.j_par[r][i] = y[r];
        }
    }
    out
}


/// The bike at rest: travel where the springs rest, the tyres at their static deflection,
/// the rider on his legs' static sag, every pod reading gravity alone. Returns each balance
/// row's misfit in its own unit (N or N m), which a consistent model holds near 0: a check of
/// a bike's setup (masses, centres, springs, geometry) before any ride.
pub fn static_misfit(m: &ForceModel, bike: &Bike<f64>) -> [f64; NBAL] {
    let fb = &m.bike;
    let g = 9.80665;
    // rear travel whose shock stroke is the shock's rest stroke
    let (mut lo, mut hi) = (0.0, fb.travel_max[1]);
    for _ in 0..60 {
        let s = 0.5 * (lo + hi);
        if fb.stroke(s).0 < fb.shock.rest_m { lo = s } else { hi = s }
    }
    let mut q = [0.0; NL];
    q[D] = fb.fork.rest_m;
    q[S] = 0.5 * (lo + hi);
    let k = kin(fb, bike, &q);
    let up_g = fb.up.scale(g);
    let meas = ForceMeas { frame: up_g.0, fork: (k.rf.t() * up_g).0, rear: (k.rr.t() * up_g).0, speed: 0.0, speed_rate: 0.0, cam: None, ground: None };
    let mut f = [0.0; NFL];
    f[DEF_F] = m.spec.init[DEF_F];
    f[DEF_R] = m.spec.init[DEF_R];
    f[POS + 2] = -(fb.m_rider + fb.head.as_ref().map_or(0.0, |h| h.m_kg)) * g / fb.nominal[P_RIDER_K + 2];
    let x = ForceIn { d: [q[D], 0.0], s: [q[S], 0.0], w0: V3::zero(), a0: V3::zero(), f, fd: [0.0; NFL], fdd: [0.0; NFL], par: [0.0; NFP] };
    let r = rows(m, &k, &x, &meas).0;
    let w = &m.w;
    let sig = [w.unsprung_n, w.unsprung_n, w.rider_n, w.rider_n, w.rider_n, w.rider_nm, w.rider_nm, w.rider_nm, w.system_n, w.system_n, w.system_n, w.system_nm, w.system_nm, w.system_nm];
    std::array::from_fn(|i| r[i] * sig[i])
}

#[cfg(test)]
pub(crate) mod synth {
    //! The exact-model ride: the frame's motion prescribed, the fork, the rear, the tyres'
    //! deflection and the rider integrated under the force model's own laws. At each substep
    //! the balance rows are linear in (fork and rear accelerations, the tyres' deflection
    //! rates, the tyre forces, the rider's accelerations): one 14 x 14 solve gives them.

    use super::*;
    use crate::estimate::physics::fixture;
    use crate::estimate::physics::readings::pod_readings;
    use crate::sim::tyre::Tyre;


    /// The test bike (the golden fixture's pods) and a plausible rider and suspension on it.
    pub fn model() -> (Bike<f64>, ForceModel) {
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/physics_window.json")).unwrap();
        let bike = fixture::bike(&doc["bike"]);
        let up = bike.up.unit();
        let fwd = (bike.fwd - up.scale(up.dot(bike.fwd))).unit();
        let axis = bike.steer_axis.unit();
        assert!(axis.dot(up) > 0.5, "the fixture's steering axis points up the fork");
        let at = |x: f64, z: f64| fwd.scale(x) + up.scale(z);
        let side = up.cross(fwd).unit();
        let plane = |p: V3<f64>| p - side.scale(side.dot(p)); // axles on the centre plane (the pods sit to one side)
        let mut nominal = [0.0; NFP];
        nominal[P_FORK_DAMP..P_FORK_DAMP + 5].copy_from_slice(&[1500.0, 500.0, 3000.0, 1200.0, 0.3]);
        nominal[P_SHOCK_DAMP..P_SHOCK_DAMP + 5].copy_from_slice(&[4000.0, 1200.0, 9000.0, 3500.0, 0.15]);
        nominal[P_FRICTION..P_FRICTION + 2].copy_from_slice(&[30.0, 40.0]);
        nominal[P_PRESSURE..P_PRESSURE + 4].copy_from_slice(&[1.0; 4]);
        nominal[P_TYRE_C..P_TYRE_C + 2].copy_from_slice(&[100.0, 100.0]);
        nominal[P_RIDER_K..P_RIDER_K + 6].copy_from_slice(&[20_000.0, 15_000.0, 30_000.0, 1500.0, 2000.0, 1000.0]);
        nominal[P_RIDER_C..P_RIDER_C + 6].copy_from_slice(&[800.0, 600.0, 1200.0, 60.0, 80.0, 40.0]);
        nominal[P_VOLUME..P_VOLUME + 4].copy_from_slice(&[1.0; 4]);
        nominal[P_BRAKE_SHIFT..P_BRAKE_SHIFT + 2].copy_from_slice(&[0.15, 0.03]);
        nominal[P_LIMBS] = 1.0;
        let tyre = |psi: f64| TyreLaw::<f64>::from_tyre(&Tyre::mtb_29(2.5, psi, true, true));
        let fb = ForceBike {
            up,
            fwd,
            fork_axis: axis,
            front_axle: plane(bike.fork_lever - axis.scale(0.08)),
            rear_axle: plane(bike.rear_lever - fwd.scale(0.05)),
            frame_cg: at(-0.15, -0.05),
            rider_cg: at(-0.25, 0.75),
            m_frame: 12.0,
            m_front: 4.0,
            m_rear: 4.5,
            m_rider: 72.0,
            m_limbs: 0.0,
            i_frame: [0.3, 1.2, 1.0],
            i_rider: [5.0, 6.0, 2.0],
            head: None,
            wheel_inertia: [0.18, 0.2],
            wheel_radius: [0.37, 0.37],
            fork: AirChambers { area_m2: 8.0e-4, positive_m3: 1.6e-4, negative_m3: 8.0e-5, gauge_pa: 0.0, rest_m: 0.05 },
            shock: AirChambers { area_m2: 9.5e-4, positive_m3: 1.6e-4, negative_m3: 9.0e-5, gauge_pa: 0.0, rest_m: 0.045 / 2.6 },
            shock_stroke: vec![0.0, 1.0 / 2.6],
            axle_member_angle: vec![],
            cage_angle: vec![],
            cage_moment_nm: [0.0; 2],
            travel_max: [0.17, 0.16],
            stop_n_m: 2.0e6,
            tyre: [tyre(24.0), tyre(28.0)],
            nominal,
            smooth_mps: 0.01,
            smooth_m: 0.001,
        };
        let mut m = ForceModel { bike: fb, spec: ForceSpec::standard([0.0; 2]), w: ForceWeights::default() };
        balance_statics(&bike, &mut m);
        (bike, m)
    }

    /// [`model`] for the road ([`ride_on_road`]): the rider over the middle of the bike (about
    /// 37 % on the front instead of 17 %: a front that light leaves the ground on every bump),
    /// the tyres' damping at 600 N s/m (the real rides fit 2-14 times the 100 nominal; at 100 its
    /// force is under the rows' sigma), exactly at rest.
    pub fn road_model() -> (Bike<f64>, ForceModel) {
        let (bike, mut m) = model();
        m.bike.rider_cg = m.bike.rider_cg + m.bike.fwd.scale(0.25);
        m.bike.nominal[P_TYRE_C..P_TYRE_C + 2].copy_from_slice(&[600.0, 600.0]);
        m.w.ground = Some(crate::estimate::physics::ground::GroundWeights::default());
        balance_statics(&bike, &mut m);
        exact_statics(&bike, &mut m);
        (bike, m)
    }

    /// Total weight, N.
    fn weight(fb: &ForceBike) -> f64 {
        (fb.m_frame + fb.m_rider + fb.m_front + fb.m_rear) * 9.80665
    }

    /// The gauge pressure (Pa) at which `ch` holds `load_n` at its rest stroke, isothermal.
    fn fill(ch: &AirChambers, load_n: f64) -> f64 {
        let (mut lo, mut hi) = (1e3, 5e6);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let f = crate::sim::laws::air_force_n(ch.area_m2, ch.positive_m3, ch.negative_m3, mid, ch.rest_m, 1.0);
            if f < load_n { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    /// Splits the weight between the wheels so the bike at rest does not pitch, and fills the
    /// springs so they hold it at their rest strokes; the tyres start at their static
    /// deflections.
    fn balance_statics(bike: &Bike<f64>, m: &mut ForceModel) {
        let w = weight(&m.bike);
        let set = |m: &mut ForceModel, beta: f64| {
            let fb = &mut m.bike;
            let up_axis = fb.up.dot(fb.fork_axis);
            fb.fork.gauge_pa = fill(&fb.fork, (beta * w - fb.m_front * 9.80665) * up_axis);
            let mut q = [0.0; NL];
            q[S] = 0.045;
            let k = kin(fb, bike, &q);
            let wheel = (((1.0 - beta) * w - fb.m_rear * 9.80665) * fb.up.dot(k.dar)).max(1.0);
            fb.shock.gauge_pa = fill(&fb.shock, wheel / k.dx);
            let defl = [beta * w / fb.tyre[0].stiffness_n_m, (1.0 - beta) * w / fb.tyre[1].stiffness_n_m];
            m.spec = ForceSpec::standard(defl);
        };
        let bike_h = bike.map(num_dual::HyperDual::from_re);
        let pitch = |m: &ForceModel| {
            let u = solve(bike, m, &bike_h, -10.0, &State::at_rest(m));
            (0..3).map(|i| u[3 + i] * [m.bike.fwd, m.bike.lat(), m.bike.up][i].dot(m.bike.lat())).sum::<f64>()
        };
        let (mut b0, mut b1) = (0.35, 0.45);
        set(m, b0);
        let mut p0 = pitch(m);
        for _ in 0..6 {
            set(m, b1);
            let p1 = pitch(m);
            if (p1 - p0).abs() < 1e-12 {
                break;
            }
            let b2 = b1 - p1 * (b1 - b0) / (p1 - p0);
            (b0, p0, b1) = (b1, p1, b2);
        }
        set(m, b1);
    }

    /// Puts `m` exactly at rest, by Newton on the setup check ([`static_misfit`]): the fork's
    /// and the rear's balance, the system's vertical force and pitch moment, over both springs'
    /// fill and both tyres' deflection. [`balance_statics`] leaves tens of newtons (the fork
    /// 48 N, the rear 231 N on [`model`]), which a ride with prescribed deflections never feels
    /// and one on a road does (the bike settles, pitches and lifts a tyre).
    fn exact_statics(bike: &Bike<f64>, m: &mut ForceModel) {
        let rows = [0, 1, 10, 12];
        let misfit = |m: &ForceModel| {
            let r = static_misfit(m, bike);
            nalgebra::Vector4::from_fn(|i, _| r[rows[i]])
        };
        // fills in bar (additive: a fill can start near zero), deflections in metres
        let nudge = |m: &mut ForceModel, x: &nalgebra::Vector4<f64>| {
            m.bike.fork.gauge_pa = (m.bike.fork.gauge_pa + 1e5 * x[0]).max(0.0);
            m.bike.shock.gauge_pa = (m.bike.shock.gauge_pa + 1e5 * x[1]).max(0.0);
            m.spec.init[DEF_F] += x[2];
            m.spec.init[DEF_R] += x[3];
        };
        let steps = [1e-4, 1e-4, 1e-7, 1e-7];
        for _ in 0..12 {
            let r0 = misfit(m);
            if r0.amax() < 1e-6 {
                break;
            }
            let mut j = nalgebra::Matrix4::zeros();
            for c_ in 0..4 {
                let mut trial = m.clone();
                let mut x = nalgebra::Vector4::zeros();
                x[c_] = steps[c_];
                nudge(&mut trial, &x);
                j.set_column(c_, &((misfit(&trial) - r0) / steps[c_]));
            }
            let x = -(j.lu().solve(&r0).expect("the setup's rows reach their unknowns"));
            nudge(m, &x);
        }
    }

    /// Prescribed inputs, ramped in over the settling run (0 before -1 s, 1 from 0 s).
    pub struct Inputs {
        /// Tyre deflection (m) and its rate, front and rear: the ground (the rear meets the
        /// front's bumps 0.2 s later).
        pub defl: [[f64; 2]; 2],
        /// Tangential and lateral force, front and rear, N (brakes, cornering, zero-mean: with the loads
        /// prescribed there is no load transfer to hold a steady brake's dive).
        pub tan_lat: [[f64; 2]; 2],
        /// The rider's active force and moment, bike axes, N and N m.
        pub act: [f64; 6],
        /// Ground speed (m/s) and its rate.
        pub speed: [f64; 2],
    }

    /// The inputs at time `t` (s).
    pub fn inputs_at(m: &ForceModel, t: f64) -> Inputs {
        let tau = std::f64::consts::TAU;
        let ramp = |t: f64| (0.5 * (1.0 + ((t + 0.5) / 0.15).tanh()), 0.5 / 0.15 / ((t + 0.5) / 0.15).cosh().powi(2));
        // a ramped sum of sines and its rate
        let wave = |t: f64, parts: &[(f64, f64, f64)]| {
            let (r, rd) = ramp(t);
            let (mut v, mut vd) = (0.0, 0.0);
            for &(a, hz, ph) in parts {
                v += a * (tau * hz * t + ph).sin();
                vd += a * tau * hz * (tau * hz * t + ph).cos();
            }
            (r * v, r * vd + rd * v)
        };
        let bumps = [(0.35, 4.0, 0.0), (0.15, 7.3, 1.0), (0.1, 11.0, 0.4)];
        let defl = |i: usize, delay: f64| {
            let d0 = m.spec.init[[DEF_F, DEF_R][i]];
            let (v, vd) = wave(t - delay, &bumps);
            [d0 * (1.0 + v), d0 * vd]
        };
        let a = |parts: &[(f64, f64, f64)]| wave(t, parts).0;
        Inputs {
            defl: [defl(0, 0.0), defl(1, 0.2)],
            tan_lat: [[a(&[(100.0, 0.8, 0.0)]), a(&[(60.0, 0.7, 0.0)])], [a(&[(80.0, 0.6, 0.0)]), a(&[(50.0, 0.7, 0.3)])]],
            act: [a(&[(40.0, 0.9, 0.0)]), a(&[(20.0, 0.6, 0.0)]), a(&[(60.0, 1.3, 0.5)]), a(&[(5.0, 0.7, 0.0)]), a(&[(8.0, 1.1, 0.2)]), a(&[(4.0, 0.5, 0.0)])],
            speed: [6.0 + a(&[(0.5, 0.3, 0.0)]), wave(t, &[(0.5, 0.3, 0.0)]).1],
        }
    }

    /// The truth at each sample.
    pub struct Truth {
        /// Core samples (travel and its derivatives; the rest 0) with the chassis motion.
        pub core: Vec<Sample<f64>>,
        /// Force latents, their rates and second derivatives.
        pub f: Vec<[[f64; NFL]; 3]>,
        /// Tyre normal forces, N.
        pub normal: Vec<[f64; 2]>,
        /// Ground speed and its rate.
        pub speed: Vec<[f64; 2]>,
        /// What the pods read (frame gyro, fork gyro and accel, rear gyro and accel).
        pub readings: Vec<[f64; 15]>,
        /// The world's up in IMU0 axes, the frame pod's height (m) and its distance along the
        /// road (m): what a navigation solve would give the ground tie.
        pub world: Vec<([f64; 3], f64, f64)>,
    }

    /// The road of [`ride_on_road`]: (amplitude m, wavelength m, phase rad), ramped in over its
    /// first metres (at 6 m/s about 4, 7.3 and 11 Hz, the bumps of the prescribed ride).
    pub const ROAD: [(f64, f64, f64); 3] = [(0.005, 1.5, 0.0), (0.002, 0.82, 1.0), (0.001, 0.55, 0.4)];

    /// The road's height (m) and slope at distance `s` (m).
    pub fn road(s: f64) -> (f64, f64) {
        let tau = std::f64::consts::TAU;
        let x = (s - 2.5) / 0.5;
        let (r, dr) = (0.5 * (1.0 + x.tanh()), 1.0 / x.cosh().powi(2));
        let (mut h, mut hs) = (0.0, 0.0);
        for &(a, l, ph) in &ROAD {
            h += a * (tau * s / l + ph).sin();
            hs += a * tau / l * (tau * s / l + ph).cos();
        }
        (r * h, r * hs + dr * h)
    }

    /// Each patch's distance ahead of the frame pod along the road (front, rear), m.
    pub fn patch_offsets(fb: &ForceBike) -> [f64; 2] {
        [fb.front_axle.dot(fb.fwd), fb.rear_axle.dot(fb.fwd)]
    }

    /// The integrated state.
    #[derive(Clone, Copy)]
    pub(crate) struct State {
        /// Front travel (m) and rate.
        d: [f64; 2],
        /// Rear travel (m) and rate.
        s: [f64; 2],
        /// Chassis rate, rad/s.
        w0: V3<f64>,
        /// Rider pose (bike axes: m, rad).
        pose: [f64; 6],
        /// Its rate.
        pose_rate: [f64; 6],
        /// The attitude holder's trims: deflection moved front to rear (m), front and rear
        /// lateral force (N). The ground and the rider hold a real bike's attitude; here
        /// every tyre force is prescribed, so a slow controller does it.
        ctrl: [f64; 3],
        /// Their rates.
        ctrl_rate: [f64; 3],
        /// On the road ([`ride_on_road`]): the tyres' deflections from the patch heights over
        /// it, and the pitch trim off (the road holds the pitch).
        road: bool,
        /// IMU0 -> world (Z up).
        att: M3<f64>,
        /// The frame pod's world velocity, m/s.
        vel: V3<f64>,
        /// Its height, m.
        z: f64,
        /// Its distance along the road, m (the ground speed integrated).
        dist: f64,
    }

    impl State {
        /// At rest: travel at the springs' rest strokes, the rider on his legs' static sag.
        fn at_rest(m: &ForceModel) -> State {
            let fb = &m.bike;
            let mut pose = [0.0; 6];
            pose[2] = -fb.m_rider * 9.80665 / fb.nominal[P_RIDER_K + 2];
            let att = M3([fb.fwd.0, fb.lat().0, fb.up.0]);
            State { d: [fb.fork.rest_m, 0.0], s: [0.045, 0.0], w0: V3::zero(), pose, pose_rate: [0.0; 6], ctrl: [0.0; 3], ctrl_rate: [0.0; 3], road: false, att, vel: V3::zero(), z: 0.0, dist: 0.0 }
        }

        /// At rest on the flat start of the road: pitched and raised so both tyres sit at
        /// their static deflections.
        fn at_rest_on_road(bike: &Bike<f64>, m: &ForceModel) -> State {
            let fb = &m.bike;
            let mut st = State { road: true, ..State::at_rest(m) };
            let mut q = [0.0; NL];
            (q[D], q[S]) = (st.d[0], st.s[0]);
            let k = kin(fb, bike, &q);
            let base = st.att;
            let lift = |i: usize| fb.wheel_radius[i] - m.spec.init[[DEF_F, DEF_R][i]];
            // the patch at the loaded radius along the bike's up, on the road: z + u.a = lift (u.up)
            let miss = |th: f64| {
                let u = (base * exp(fb.lat().scale(th))).t() * V3([0.0, 0.0, 1.0]);
                (lift(0) * u.dot(fb.up) - u.dot(k.af)) - (lift(1) * u.dot(fb.up) - u.dot(k.ar))
            };
            let (mut a, mut b) = (0.0, 0.01);
            for _ in 0..20 {
                let (fa, fb_) = (miss(a), miss(b));
                if (fb_ - fa).abs() < 1e-15 {
                    break;
                }
                (a, b) = (b, b - fb_ * (b - a) / (fb_ - fa));
            }
            st.att = base * exp(fb.lat().scale(b));
            let u = st.att.t() * V3([0.0, 0.0, 1.0]);
            st.z = lift(0) * u.dot(fb.up) - u.dot(k.af);
            st
        }

        /// The tyres' deflections (m) and their rates on the road, front and rear: each patch
        /// (the axle less the loaded radius along the bike's up) on the road's height there.
        fn on_road(&self, bike: &Bike<f64>, m: &ForceModel, speed: f64) -> [[f64; 2]; 2] {
            let fb = &m.bike;
            let mut q = [0.0; NL];
            (q[D], q[S]) = (self.d[0], self.s[0]);
            let k = kin(fb, bike, &q);
            let h = 1e-6;
            let (mut qp, mut qm) = (q, q);
            qp[D] += h;
            qm[D] -= h;
            let daf = (kin(fb, bike, &qp).af - kin(fb, bike, &qm).af).scale(0.5 / h);
            let u = self.att.t() * V3([0.0, 0.0, 1.0]);
            let ud = -self.w0.cross(u);
            let off = patch_offsets(fb);
            let mut out = [[0.0; 2]; 2];
            for (i, a, ad) in [(0, k.af, daf.scale(self.d[1])), (1, k.ar, k.dar.scale(self.s[1]))] {
                let (hh, hs) = road(self.dist + off[i]);
                let above = self.z + u.dot(a) - hh;
                let rate = self.vel.0[2] + ud.dot(a) + u.dot(ad) - hs * speed;
                let (cu, cud) = (u.dot(fb.up), ud.dot(fb.up));
                out[i] = [fb.wheel_radius[i] - above / cu, -(rate * cu - above * cud) / (cu * cu)];
            }
            out
        }
    }

    /// The unknowns of a substep, `u`: the frame pod's specific force (3, bike axes), the
    /// chassis angular acceleration (3, bike axes), d'', s'', the rider's pose'' (6).
    fn sample(bike: &Bike<f64>, m: &ForceModel, t: f64, st: &State, u: &[f64; 14]) -> (Sample<f64>, [[f64; NFL]; 3], ForceMeas) {
        let fb = &m.bike;
        let mut inp = inputs_at(m, t);
        if st.road {
            // a straight road: the bumps, the brakes and the rider's pumping and pitch, in the
            // pitch plane where the ground tie works; no side forces to lay the bike over
            inp.defl = st.on_road(bike, m, inp.speed[0]);
            for i in 0..2 {
                inp.tan_lat[i][1] = 0.0;
            }
            for i in [1, 3, 5] {
                inp.act[i] = 0.0;
            }
        } else {
            inp.defl[0][0] += st.ctrl[0];
            inp.defl[0][1] += st.ctrl_rate[0];
            inp.defl[1][0] -= st.ctrl[0];
            inp.defl[1][1] -= st.ctrl_rate[0];
        }
        inp.tan_lat[0][1] += st.ctrl[1];
        inp.tan_lat[1][1] += st.ctrl[2];
        let bv = |v: &[f64]| fb.fwd.scale(v[0]) + fb.lat().scale(v[1]) + fb.up.scale(v[2]);
        let (mut q, mut qd, mut qdd) = ([0.0; NL], [0.0; NL], [0.0; NL]);
        (q[D], qd[D], qdd[D]) = (st.d[0], st.d[1], u[6]);
        (q[S], qd[S], qdd[S]) = (st.s[0], st.s[1], u[7]);
        let f0 = bv(&u[0..3]);
        let core = Sample { q, qd, qdd, w0: st.w0, a0: bv(&u[3..6]), f_meas: f0 };
        let mut f = [[0.0; NFL]; 3];
        for (i, d0) in [DEF_F, DEF_R].into_iter().enumerate() {
            (f[0][d0], f[1][d0]) = (inp.defl[i][0], inp.defl[i][1]);
            (f[0][d0 + 1], f[0][d0 + 2]) = (inp.tan_lat[i][0], inp.tan_lat[i][1]);
        }
        for i in 0..6 {
            (f[0][POS + i], f[1][POS + i], f[2][POS + i]) = (st.pose[i], st.pose_rate[i], u[8 + i]);
            f[0][ACT_F + i] = inp.act[i];
        }
        (core, f, ForceMeas { frame: f0.0, fork: [0.0; 3], rear: [0.0; 3], speed: inp.speed[0], speed_rate: inp.speed[1], cam: None, ground: None })
    }

    /// One evaluation at `t` with unknowns `u`.
    struct Eval {
        r: [f64; NFR],
        forces: Forces<f64>,
        core: Sample<f64>,
        f: [[f64; NFL]; 3],
        meas: ForceMeas,
        readings: [f64; 15],
    }

    fn eval(bike: &Bike<f64>, m: &ForceModel, bike_h: &Bike<num_dual::HyperDual<f64>>, t: f64, st: &State, u: &[f64; 14]) -> Eval {
        let (core, f, mut meas) = sample(bike, m, t, st, u);
        let p = pod_readings(bike_h, &core);
        meas.fork = [p[6], p[7], p[8]];
        meas.rear = [p[12], p[13], p[14]];
        let k = kin(&m.bike, bike, &core.q);
        let x = ForceIn { d: [core.q[D], core.qd[D]], s: [core.q[S], core.qd[S]], w0: core.w0, a0: core.a0, f: f[0], fd: f[1], fdd: f[2], par: [0.0; NFP] };
        let (r, forces) = rows(m, &k, &x, &meas);
        Eval { r, forces, core, f, meas, readings: p }
    }

    /// The substep's unknowns: the balance rows are linear in them, so one solve.
    fn solve(bike: &Bike<f64>, m: &ForceModel, bike_h: &Bike<num_dual::HyperDual<f64>>, t: f64, st: &State) -> [f64; 14] {
        let u0 = [0.0; 14];
        let r0 = eval(bike, m, bike_h, t, st, &u0).r;
        let mut j = nalgebra::DMatrix::zeros(NBAL, 14);
        for col in 0..14 {
            let mut u = u0;
            u[col] = 1.0;
            let r1 = eval(bike, m, bike_h, t, st, &u).r;
            for row in 0..NBAL {
                j[(row, col)] = r1[row] - r0[row];
            }
        }
        let b = nalgebra::DVector::from_iterator(NBAL, r0[..NBAL].iter().map(|x| -x));
        let x = j.lu().solve(&b).expect("the balance rows are solvable");
        std::array::from_fn(|i| x[i])
    }

    /// Runs the exact-model ride: `n` samples at `dt` after a settling run of `settle_s`.
    pub fn ride(bike: &Bike<f64>, m: &ForceModel, n: usize, dt: f64, settle_s: f64) -> Truth {
        ride_from(bike, m, State::at_rest(m), n, dt, settle_s)
    }

    /// [`ride`] over [`ROAD`]: the tyres' deflections from where the patches sit over the road,
    /// not prescribed; the frame's world attitude, velocity and height integrated with gravity.
    pub fn ride_on_road(bike: &Bike<f64>, m: &ForceModel, n: usize, dt: f64, settle_s: f64) -> Truth {
        ride_from(bike, m, State::at_rest_on_road(bike, m), n, dt, settle_s)
    }

    fn ride_from(bike: &Bike<f64>, m: &ForceModel, start: State, n: usize, dt: f64, settle_s: f64) -> Truth {
        let bike_h = bike.map(num_dual::HyperDual::from_re);
        let mut st = start;
        let sub = 20;
        let h = dt / sub as f64;
        let skip = (settle_s / dt).round() as usize;
        let mut truth = Truth { core: Vec::with_capacity(n), f: Vec::with_capacity(n), normal: Vec::with_capacity(n), speed: Vec::with_capacity(n), readings: Vec::with_capacity(n), world: Vec::with_capacity(n) };
        let fb = &m.bike;
        // the attitude holder: angular acceleration (bike axes) per unit trim, at rest
        let ang = |st: &State| { let u = solve(bike, m, &bike_h, -10.0, st); nalgebra::Vector3::new(u[3], u[4], u[5]) };
        let a_rest = ang(&st);
        let unit = [1e-4, 10.0, 10.0];
        let mut jc = nalgebra::Matrix3::zeros();
        for c_ in 0..3 {
            let mut s2 = st;
            s2.ctrl[c_] = unit[c_];
            jc.set_column(c_, &((ang(&s2) - a_rest) / unit[c_]));
        }
        // on the road the pitch is the road's: the lateral trims hold roll and yaw alone
        let jinv = if st.road {
            let j2 = nalgebra::Matrix2::new(jc[(0, 1)], jc[(0, 2)], jc[(2, 1)], jc[(2, 2)]);
            let i2 = j2.try_inverse().expect("the lateral trims reach roll and yaw");
            nalgebra::Matrix3::new(0.0, 0.0, 0.0, i2[(0, 0)], 0.0, i2[(0, 1)], i2[(1, 0)], 0.0, i2[(1, 1)])
        } else {
            jc.try_inverse().expect("the trims reach every axis")
        };
        let hold_s = 0.2;
        for i in 0..skip + n {
            for k_ in 0..sub {
                let t = (i as f64 - skip as f64) * dt + k_ as f64 * h;
                let wb = nalgebra::Vector3::new(st.w0.dot(fb.fwd), st.w0.dot(fb.lat()), st.w0.dot(fb.up));
                // on the road the roll angle is held too (a rider steers to stay up; here the
                // lateral trims do it), critically damped at 1.5 Hz, or the side forces lay the
                // bike over and lift its tyres
                let target = if st.road {
                    let u = st.att.t() * V3([0.0, 0.0, 1.0]);
                    let (wn, roll) = (std::f64::consts::TAU * 1.5, u.dot(fb.lat()).atan2(u.dot(fb.up)));
                    nalgebra::Vector3::new(2.0 * wn * wb[0] + wn * wn * roll, 0.0, wb[2] / hold_s)
                } else {
                    wb / hold_s
                };
                let c_new = -(jinv * target);
                for c_ in 0..3 {
                    st.ctrl_rate[c_] = (c_new[c_] - st.ctrl[c_]) / h;
                    st.ctrl[c_] = c_new[c_];
                }
                if i == 0 && k_ == 0 { st.ctrl_rate = [0.0; 3]; }
                let u = solve(bike, m, &bike_h, t, &st);
                if k_ == 0 && i >= skip {
                    let e = eval(bike, m, &bike_h, t, &st, &u);
                    truth.core.push(e.core);
                    truth.f.push(e.f);
                    truth.normal.push(e.forces.normal);
                    truth.speed.push([e.meas.speed, e.meas.speed_rate]);
                    truth.readings.push(e.readings);
                    truth.world.push(((st.att.t() * V3([0.0, 0.0, 1.0])).0, st.z, st.dist));
                }
                // semi-implicit Euler; the frame's world motion with gravity
                let f0 = fb.fwd.scale(u[0]) + fb.lat().scale(u[1]) + fb.up.scale(u[2]);
                st.vel = st.vel + (st.att * f0 + V3([0.0, 0.0, -9.80665])).scale(h);
                st.z += st.vel.0[2] * h;
                st.dist += inputs_at(m, t).speed[0] * h;
                st.att = st.att * exp(st.w0.scale(h));
                st.w0 = st.w0 + (fb.fwd.scale(u[3]) + fb.lat().scale(u[4]) + fb.up.scale(u[5])).scale(h);
                st.d[1] += u[6] * h;
                st.d[0] += st.d[1] * h;
                st.s[1] += u[7] * h;
                st.s[0] += st.s[1] * h;
                for p in 0..6 {
                    st.pose_rate[p] += u[8 + p] * h;
                    st.pose[p] += st.pose_rate[p] * h;
                }
            }
        }
        truth
    }

    #[test]
    fn air_spring_rest_softer_spring_sits_deeper_on_the_same_load() {
        // Arrange: a shock-sized spring resting at 15 mm stroke at 190 psi
        let ch = AirChambers { area_m2: 22.0e-4, positive_m3: 3.6e-4, negative_m3: 8.4e-5, gauge_pa: 190.0 * 6894.76, rest_m: 0.015 };
        let iso = |scale: f64, x: f64| crate::sim::laws::air_force_n(ch.area_m2, ch.positive_m3, ch.negative_m3, scale * ch.gauge_pa, x, 1.0);

        // Act
        let same: f64 = ch.rest_at(1.0, [1.0; 2]);
        let soft: f64 = ch.rest_at(0.8, [1.0; 2]);
        let firm: f64 = ch.rest_at(1.25, [1.0; 2]);

        // Assert: the nominal rest unmoved, and each rest carrying the nominal load
        assert!((same - 0.015).abs() < 1e-12, "{same}");
        assert!(soft > 0.015 && firm < 0.015, "{soft} {firm}");
        let load = iso(1.0, 0.015);
        assert!((iso(0.8, soft) - load).abs() < 1e-6 * load, "{} vs {load}", iso(0.8, soft));
        assert!((iso(1.25, firm) - load).abs() < 1e-6 * load, "{} vs {load}", iso(1.25, firm));
    }

    #[test]
    fn exact_ride_satisfies_its_own_balances_and_keeps_both_tyres_loaded() {
        // Arrange
        let (bike, m) = model();

        // Act
        let truth = ride(&bike, &m, 200, 1.0 / 400.0, 1.0);

        // Assert: tyres loaded, travel inside its range, the rider within his prior
        let nmin = truth.normal.iter().fold(f64::MAX, |a, n| a.min(n[0]).min(n[1]));
        let wmax = truth.core.iter().map(|s| s.w0.norm()).fold(0.0, f64::max);
        for (name, ax) in [("roll", m.bike.fwd), ("pitch", m.bike.lat()), ("yaw", m.bike.up)] {
            let (lo, hi) = truth.core.iter().map(|s| s.w0.dot(ax)).fold((f64::MAX, f64::MIN), |a, x| (a.0.min(x), a.1.max(x)));
            eprintln!("  {name} rate {lo:.2}..{hi:.2} rad/s; first {:.2}, last {:.2}", truth.core[0].w0.dot(ax), truth.core.last().unwrap().w0.dot(ax));
        }
        let d = truth.core.iter().map(|s| s.q[D]).fold((f64::MAX, f64::MIN), |a, x| (a.0.min(x), a.1.max(x)));
        let s = truth.core.iter().map(|s| s.q[S]).fold((f64::MAX, f64::MIN), |a, x| (a.0.min(x), a.1.max(x)));
        let pz = truth.f.iter().map(|f| f[0][POS + 2].abs()).fold(0.0, f64::max);
        eprintln!("exact ride: tyre normal min {nmin:.0} N; front travel {:.1}..{:.1} mm, rear {:.1}..{:.1} mm; rider |z| max {:.1} mm; chassis rate max {wmax:.2} rad/s",
                  d.0 * 1e3, d.1 * 1e3, s.0 * 1e3, s.1 * 1e3, pz * 1e3);
        assert!(nmin > 50.0, "a tyre unloaded: {nmin} N");
        assert!(d.0 > 0.01 && d.1 < 0.16 && s.0 > 0.01 && s.1 < 0.15);
        assert!(pz < 0.1);
    }

    /// The window the solver sees from an exact ride.
    pub fn window(truth: &Truth, rate: f64) -> crate::estimate::physics::objective::Window {
        let col = |a: usize| truth.readings.iter().map(|p| [p[a], p[a + 1], p[a + 2]]).collect::<Vec<_>>();
        let n = truth.core.len();
        crate::estimate::physics::objective::Window {
            rate,
            frame_gyro: col(0),
            frame_accel: truth.core.iter().map(|s| s.f_meas.0).collect(),
            fork_gyro: col(3),
            fork_accel: col(6),
            rear_gyro: col(9),
            rear_accel: col(12),
            mask_fork: vec![true; n],
            mask_rear: vec![true; n],
            air: None,
            nav: None,
            forces: Some(ForceWindow { speed_mps: truth.speed.iter().map(|s| s[0]).collect(), cam: vec![], ground: None }),
        }
    }

    /// The tyre normal forces the objective's solution implies at sample `t`.
    pub fn normal_at(obj: &crate::estimate::physics::objective::Objective, v: &crate::estimate::physics::objective::Vars, t: usize) -> [f64; 2] {
        let m = obj.forces.as_ref().unwrap();
        let s = obj.sample(v, t);
        let f = obj.force_inputs(v, t).unwrap();
        let g0 = obj.force_glob0();
        let par: [f64; NFP] = std::array::from_fn(|i| v.glob[g0 + i]);
        let k = kin(&m.bike, &obj.bike, &s.q);
        let x = ForceIn { d: [s.q[D], s.qd[D]], s: [s.q[S], s.qd[S]], w0: s.w0, a0: s.a0, f: f[0], fd: f[1], fdd: f[2], par };
        let meas = ForceMeas { frame: obj.win.frame_accel[t], fork: obj.win.fork_accel[t], rear: obj.win.rear_accel[t], speed: 0.0, speed_rate: 0.0, cam: None, ground: None };
        rows(m, &k, &x, &meas).1.normal
    }

    #[test]
    fn exact_ride_solve_recovers_travel_tyre_loads_and_the_rider() {
        // Arrange: 1 s of the exact ride; the solver starts cold (travel 45 mm, the rider at neutral)
        use crate::estimate::physics::objective::{Objective, Spec, Weights, NG};
        use crate::estimate::physics::solve::{force_latents, gauss_newton, latents};
        let (bike, m) = model();
        let (n, dt) = (400, 1.0 / 400.0);
        let truth = ride(&bike, &m, n, dt, 1.0);
        let obj = Objective::new(bike, window(&truth, 400.0), Spec::standard(n, dt), Weights::default(), [0.0; NG]).with_forces(m);

        // Act
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), 40, 1e-4).unwrap();

        // Assert: away from the window's ends
        let q = latents(&obj, &v);
        let f = force_latents(&obj, &v);
        let inner = 20..n - 20;
        let rms = |e: &dyn Fn(usize) -> f64| (inner.clone().map(|t| e(t).powi(2)).sum::<f64>() / inner.len() as f64).sqrt();
        let d_rms = rms(&|t| q[t][D] - truth.core[t].q[D]);
        let s_rms = rms(&|t| q[t][S] - truth.core[t].q[S]);
        let nf = rms(&|t| normal_at(&obj, &v, t)[0] - truth.normal[t][0]);
        let nr = rms(&|t| normal_at(&obj, &v, t)[1] - truth.normal[t][1]);
        let pose: Vec<f64> = (0..6).map(|i| rms(&|t| f[t][POS + i] - truth.f[t][0][POS + i])).collect();
        let mean_n = inner.clone().map(|t| truth.normal[t][0] + truth.normal[t][1]).sum::<f64>() / inner.len() as f64;
        eprintln!("seconds: normal equations {:.2}, solve {:.2}, loss {:.2}", info.seconds[0], info.seconds[1], info.seconds[2]);
        eprintln!("force solve: {} iterations, loss {:.2} -> {:.2}; travel rms front {:.2} mm, rear {:.2} mm; tyre load rms front {nf:.1} N, rear {nr:.1} N (mean total {mean_n:.0} N); rider pose rms {:?}",
                  info.iterations, info.loss[0], info.loss.last().unwrap(), d_rms * 1e3, s_rms * 1e3, pose.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
        assert!(d_rms < 0.002 && s_rms < 0.002, "travel rms {d_rms} / {s_rms} m");
        assert!(nf < 0.05 * mean_n && nr < 0.05 * mean_n, "tyre loads off by {nf} / {nr} N");
    }

    /// Starts the damper, tyre and rider numbers wrong (and centres their priors there),
    /// solves `secs` of the exact ride, and returns each start and where it ended (log units).
    fn identify(secs: f64) -> Vec<(&'static str, f64, f64)> {
        identify_on(secs, false)
    }

    /// [`identify`], on the road with the ground tie (`road`) or with prescribed deflections.
    fn identify_on(secs: f64, road: bool) -> Vec<(&'static str, f64, f64)> {
        use crate::estimate::physics::objective::{Objective, Spec, Weights, NG};
        use crate::estimate::physics::solve::gauss_newton;
        let (bike, m) = if road { road_model() } else { model() };
        let dt = 1.0 / 400.0;
        let n = (secs / dt).round() as usize;
        let truth = if road { ride_on_road(&bike, &m, n, dt, 1.0) } else { ride(&bike, &m, n, dt, 1.0) };
        let mut win = window(&truth, 400.0);
        if road {
            win.forces.as_mut().unwrap().ground = Some(ground_of(&truth, &m));
        }
        let mut obj = Objective::new(bike, win, Spec::standard(n, dt), Weights::default(), [0.0; NG]).with_forces(m);
        let g0 = obj.force_glob0();
        let wrong = [
            ("fork LSC", P_FORK_DAMP, 0.4),
            ("fork LSR", P_FORK_DAMP + 2, -0.4),
            ("shock LSC", P_SHOCK_DAMP, 0.4),
            ("shock LSR", P_SHOCK_DAMP + 2, -0.3),
            ("rider vertical k", P_RIDER_K + 2, 0.5),
            ("rider vertical c", P_RIDER_C + 2, -0.5),
            ("rider pitch k", P_RIDER_K + 4, 0.4),
            ("front tyre damping", P_TYRE_C, -0.5),
        ];
        for &(_, i, x) in &wrong {
            obj.glob_centre[g0 + i] = x;
        }
        // from the pods-only solve, as a ride's windows start (the ground tie moves both tyres'
        // common height a little per iteration, so a cold start crawls)
        let v0 = if road { crate::estimate::physics::solve::warm_start(&obj, 40, 1e-4).unwrap() } else { obj.initial_vars() };
        let (v, info) = gauss_newton(&obj, v0, 60, 1e-5).unwrap();
        let out: Vec<_> = wrong.iter().map(|&(name, i, x)| (name, x, v.glob[g0 + i])).collect();
        eprintln!("identification, {secs} s ({} iterations): {}", info.iterations, out.iter().map(|(n, a, b)| format!("{n} {a:+.2} -> {b:+.3}")).collect::<Vec<_>>().join(", "));
        out
    }

    /// The ground tie's inputs from an exact ride's world state: a perfect navigation, so a
    /// tie of 0.5 mm (the real rides' 5 mm is for navigation's drift and line changes, and is
    /// looser than this gentle road's whole deflection swing).
    pub fn ground_of(truth: &Truth, m: &ForceModel) -> crate::estimate::physics::ground::GroundWindow {
        let sigma = 0.0005;
        crate::estimate::physics::ground::GroundWindow {
            up: truth.world.iter().map(|w| w.0).collect(),
            z_m: truth.world.iter().map(|w| w.1).collect(),
            s_m: truth.world.iter().map(|w| w.2).collect(),
            sigma_m: vec![[sigma; 2]; truth.world.len()],
            offset_m: patch_offsets(&m.bike),
        }
    }

    #[test]
    fn exact_ride_on_road_sits_both_patches_on_the_road_and_keeps_both_tyres_on_it() {
        // Arrange
        let (bike, m) = road_model();

        // Act
        let truth = ride_on_road(&bike, &m, 400, 1.0 / 400.0, 1.0);

        // Assert: each patch (the axle less the loaded radius along the bike's up) at the road's
        // height, which the tie's rows read
        let fb = &m.bike;
        let off = patch_offsets(fb);
        let mut worst: f64 = 0.0;
        for (t, s) in truth.core.iter().enumerate() {
            let k = kin(fb, &bike, &s.q);
            let (u, z, dist) = truth.world[t];
            for (i, a, d0) in [(0, k.af, DEF_F), (1, k.ar, DEF_R)] {
                let p = a - fb.up.scale(fb.wheel_radius[i] - truth.f[t][0][d0]);
                worst = worst.max((z + p.dot(V3(u)) - road(dist + off[i]).0).abs());
            }
        }
        let nmin = truth.normal.iter().fold(f64::MAX, |a, n| a.min(n[0]).min(n[1]));
        let defl = truth.f.iter().fold((f64::MAX, f64::MIN), |a, f| (a.0.min(f[0][DEF_F]).min(f[0][DEF_R]), a.1.max(f[0][DEF_F]).max(f[0][DEF_R])));
        eprintln!("road ride: patch off the road {:.2e} m; tyre normal min {nmin:.0} N; deflection {:.1}..{:.1} mm", worst, defl.0 * 1e3, defl.1 * 1e3);
        assert!(worst < 1e-9, "{worst} m");
        assert!(nmin > 5.0, "a tyre off the road: {nmin} N");
    }

    #[test]
    fn exact_ride_on_road_solve_with_the_ground_tie_recovers_travel_loads_and_the_road() {
        // Arrange: 1 s on the road, the tie's inputs from the ride's world state; a cold start
        use crate::estimate::physics::objective::{Objective, Spec, Weights, NG};
        use crate::estimate::physics::solve::{gauss_newton, latents};
        let (bike, m) = road_model();
        let (n, dt) = (400, 1.0 / 400.0);
        let truth = ride_on_road(&bike, &m, n, dt, 1.0);
        let mut win = window(&truth, 400.0);
        let gw = ground_of(&truth, &m);
        win.forces.as_mut().unwrap().ground = Some(gw.clone());
        let obj = Objective::new(bike, win, Spec::standard(n, dt), Weights::default(), [0.0; NG]).with_forces(m);

        // Act
        let v0 = crate::estimate::physics::solve::warm_start(&obj, 40, 1e-4).unwrap();
        let (v, info) = gauss_newton(&obj, v0, 150, 1e-6).unwrap();

        // Assert: away from the window's ends
        let q = latents(&obj, &v);
        let inner = 20..n - 20;
        let rms = |e: &dyn Fn(usize) -> f64| (inner.clone().map(|t| e(t).powi(2)).sum::<f64>() / inner.len() as f64).sqrt();
        let d_rms = rms(&|t| q[t][D] - truth.core[t].q[D]);
        let s_rms = rms(&|t| q[t][S] - truth.core[t].q[S]);
        let nf = rms(&|t| normal_at(&obj, &v, t)[0] - truth.normal[t][0]);
        let nr = rms(&|t| normal_at(&obj, &v, t)[1] - truth.normal[t][1]);
        let g = v.ground.as_ref().expect("a ground profile");
        let h_rms = rms(&|t| g.height(gw.s_m[t] + gw.offset_m[0]) - road(gw.s_m[t] + gw.offset_m[0]).0);
        let mean_n = inner.clone().map(|t| truth.normal[t][0] + truth.normal[t][1]).sum::<f64>() / inner.len() as f64;
        let bias = |e: &dyn Fn(usize) -> f64| inner.clone().map(e).sum::<f64>() / inner.len() as f64;
        let g0 = obj.force_glob0();
        eprintln!("road solve bias: travel {:.2} / {:.2} mm, ground {:.2} mm, defl {:.2} / {:.2} mm; pars {:?}", bias(&|t| q[t][D] - truth.core[t].q[D]) * 1e3, bias(&|t| q[t][S] - truth.core[t].q[S]) * 1e3,
                  bias(&|t| g.height(gw.s_m[t] + gw.offset_m[0]) - road(gw.s_m[t] + gw.offset_m[0]).0) * 1e3,
                  bias(&|t| crate::estimate::physics::solve::force_latents(&obj, &v)[t][DEF_F] - truth.f[t][0][DEF_F]) * 1e3, bias(&|t| crate::estimate::physics::solve::force_latents(&obj, &v)[t][DEF_R] - truth.f[t][0][DEF_R]) * 1e3,
                  (0..18).map(|i| (v.glob[g0 + i] * 100.0).round() / 100.0).collect::<Vec<_>>());
        eprintln!("road solve: {} iterations, loss {:.2} -> {:.2}; travel rms {:.2} / {:.2} mm; tyre load rms {nf:.1} / {nr:.1} N; ground rms {:.2} mm",
                  info.iterations, info.loss[0], info.loss.last().unwrap(), d_rms * 1e3, s_rms * 1e3, h_rms * 1e3);
        assert!(d_rms < 0.002 && s_rms < 0.002, "travel rms {d_rms} / {s_rms} m");
        assert!(nf < 0.05 * mean_n && nr < 0.05 * mean_n, "tyre loads off by {nf} / {nr} N");
        assert!(h_rms < 0.002, "ground rms {h_rms} m");
    }

    #[test]
    fn exact_ride_on_road_identification_with_the_ground_tie_recovers_the_tyre_damping() {
        // Arrange / Act: 6 s (over 3 s of this gentle road the tyre's damping keeps half its
        // error: the ground is unknown too, and the prior is centred on the wrong start)
        let out = identify_on(6.0, true);

        // Assert: the dampers and the rider's vertical as without the tie, and now the front
        // tyre's damping, which forces alone could not tell (F1: it does not move)
        for (name, start, end) in out[..6].iter().chain(&out[7..8]) {
            assert!(end.abs() < 0.3 * start.abs(), "{name}: {start} -> {end}");
        }
    }

    #[test]
    fn exact_ride_identification_three_seconds_recovers_the_dampers_and_the_rider_vertical() {
        // Arrange / Act
        let out = identify(3.0);

        // Assert: the four damper numbers and the rider's vertical spring and damper keep under 30 %
        // of their error (1 s keeps 15-70 %: identification needs seconds of riding, and a real fit
        // pools many windows). Not identified from forces alone, as expected: the tyre's damping
        // (needs the shared ground, F4) and the rider's pitch spring (needs the helmet, F2)
        for (name, start, end) in &out[..6] {
            assert!(end.abs() < 0.3 * start.abs(), "{name}: {start} -> {end}");
        }
    }

    /// A head for a test: the setup's geometry on `fb`'s axes.
    fn test_head(fb: &ForceBike) -> Head {
        use super::super::force_setup::{CAMERA_FROM_NECK_M, HEAD_CG_FROM_NECK_M, HEAD_GYRATION_M, NECK_FROM_RIDER_CG_M};
        let at = |v: [f64; 2]| fb.fwd.scale(v[0]) + fb.up.scale(v[1]);
        Head { m_kg: 6.6, gyration_m: HEAD_GYRATION_M, neck: at(NECK_FROM_RIDER_CG_M), cg: at(HEAD_CG_FROM_NECK_M), cam: at(CAMERA_FROM_NECK_M) }
    }

    #[test]
    fn helmet_camera_still_rider_on_a_turning_bike_reads_the_frames_motion() {
        // Arrange: the rider and his head at their neutral pose, the bike turning steadily
        let (bike, mut m) = model();
        m.bike.head = Some(test_head(&m.bike));
        let h = m.bike.head.clone().unwrap();
        let truth = ride(&bike, &m, 40, 1.0 / 400.0, 0.5);
        let mut s = truth.core[30];
        s.w0 = V3([0.3, -0.2, 0.5]);
        s.a0 = V3::zero();
        let f0 = s.f_meas;
        let pc = m.bike.rider_cg + h.neck + h.cam;
        let expect = f0 + s.w0.cross(s.w0.cross(pc));
        let z = [0.0; NFL];
        let meas = ForceMeas { frame: f0.0, fork: [0.0; 3], rear: [0.0; 3], speed: 0.0, speed_rate: 0.0, cam: Some([expect.0[0], expect.0[1], expect.0[2], s.w0.0[0], s.w0.0[1], s.w0.0[2]]), ground: None };
        let bike_d = bike.map(Dual64::from);

        // Act
        let j = force_rows(&m, &bike, &bike_d, &s, &z, &z, &z, &[0.0; NFP], &meas, false);

        // Assert: the camera's six rows read nothing to explain
        for r in 35..41 {
            assert!(j.r[r].abs() < 1e-9, "row {r}: {}", j.r[r]);
        }
    }

    #[test]
    fn force_rows_jacobian_matches_finite_differences() {
        // Arrange: a moving sample of the exact ride, parameters off their nominal; the rear
        // braking hard, with the caliper on a rotating axle member and a derailleur cage taking
        // up the chain (the GT Force's curves), so their terms are in the rows
        let (bike, mut m) = model();
        let truth = ride(&bike, &m, 40, 1.0 / 400.0, 0.5);
        m.bike.axle_member_angle = vec![0.0004, -0.855, -2.83, 4.03];
        m.bike.cage_angle = vec![0.945, 0.870, 2.64, -2.43];
        m.bike.cage_moment_nm = [1.27, 1.58];
        m.bike.m_limbs = 5.0;
        // and the head turned on its neck, the helmet camera reading something
        m.bike.head = Some(test_head(&m.bike));
        let t = 30;
        let s = truth.core[t];
        let [mut f, mut fd, mut fdd] = truth.f[t];
        f[TAN_R] = -300.0;
        for i in 0..3 {
            f[HEAD + i] = 0.05 * (i as f64 + 1.0);
            fd[HEAD + i] = 0.4 - 0.3 * i as f64;
            fdd[HEAD + i] = 3.0 * (i as f64 - 1.0);
            f[NECK_ACT + i] = 1.5 - i as f64;
        }
        let p = truth.readings[t];
        let meas = ForceMeas { frame: s.f_meas.0, fork: [p[6], p[7], p[8]], rear: [p[12], p[13], p[14]], speed: truth.speed[t][0], speed_rate: truth.speed[t][1], cam: Some([0.5, -1.0, 9.0, 0.1, -0.2, 0.3]), ground: Some(GroundMeas { up: [0.1, -0.05, 0.99], z_m: 0.4, h_m: [-0.32, -0.27], sigma_m: [0.005, 0.009] }) };
        let par: [f64; NFP] = std::array::from_fn(|i| 0.1 * ((i % 5) as f64 - 2.0));
        let bike_d = bike.map(Dual64::from);
        let at = |s: &Sample<f64>, f: &[f64; NFL], fd: &[f64; NFL], fdd: &[f64; NFL], par: &[f64; NFP]| force_rows(&m, &bike, &bike_d, s, f, fd, fdd, par, &meas, false).r;

        // Act
        let j = force_rows(&m, &bike, &bike_d, &s, &f, &fd, &fdd, &par, &meas, true);

        // Assert
        let check = |what: &str, got: &dyn Fn(usize) -> f64, bump: &dyn Fn(f64) -> [f64; NFR], h: f64| {
            let (up, dn) = (bump(h), bump(-h));
            for r in 0..NFR {
                let fdv = (up[r] - dn[r]) / (2.0 * h);
                assert!((got(r) - fdv).abs() <= 1e-4 * (1.0 + fdv.abs()), "{what}, row {r}: {} vs {fdv}", got(r));
            }
        };
        for &i in &Q_USED {
            check(&format!("q {i}"), &|r| j.j_core[r][i], &|h| { let mut s2 = s; s2.q[i] += h; at(&s2, &f, &fd, &fdd, &par) }, 1e-6);
        }
        for i in [D, S] {
            check(&format!("q' {i}"), &|r| j.j_core[r][NL + i], &|h| { let mut s2 = s; s2.qd[i] += h; at(&s2, &f, &fd, &fdd, &par) }, 1e-6);
        }
        for i in 0..3 {
            check(&format!("w0 {i}"), &|r| j.j_core[r][3 * NL + i], &|h| { let mut s2 = s; s2.w0.0[i] += h; at(&s2, &f, &fd, &fdd, &par) }, 1e-6);
            check(&format!("a0 {i}"), &|r| j.j_core[r][3 * NL + 3 + i], &|h| { let mut s2 = s; s2.a0.0[i] += h; at(&s2, &f, &fd, &fdd, &par) }, 1e-6);
        }
        for i in 0..NFL {
            let h = if i == DEF_F || i == DEF_R { 1e-7 } else { 1e-4 };
            check(&format!("f {i}"), &|r| j.j_frc[r][i], &|e| { let mut g = f; g[i] += e; at(&s, &g, &fd, &fdd, &par) }, h);
            check(&format!("f' {i}"), &|r| j.j_frc[r][NFL + i], &|e| { let mut g = fd; g[i] += e; at(&s, &f, &g, &fdd, &par) }, 1e-5);
            check(&format!("f'' {i}"), &|r| j.j_frc[r][2 * NFL + i], &|e| { let mut g = fdd; g[i] += e; at(&s, &f, &fd, &g, &par) }, 1e-4);
        }
        for i in 0..NFP {
            check(&format!("parameter {i}"), &|r| j.j_par[r][i], &|e| { let mut g = par; g[i] += e; at(&s, &f, &fd, &fdd, &g) }, 1e-6);
        }
    }

    #[test]
    fn normal_equations_by_sample_chunks_on_threads_equal_the_serial_ones() {
        // Arrange: the exact ride with the force model on, at its cold start
        use crate::estimate::physics::objective::{Objective, Spec, Weights, NG};
        use crate::estimate::physics::solve::{normal_equations, Layout};
        let (bike, m) = model();
        let (n, dt) = (300, 1.0 / 400.0);
        let truth = ride(&bike, &m, n, dt, 0.5);
        let serial = Objective::new(bike, window(&truth, 400.0), Spec::standard(n, dt), Weights::default(), [0.0; NG]).with_forces(m);
        let threaded = Objective { threads: 4, ..Objective { bike: serial.bike.clone(), win: serial.win.clone(), spec: serial.spec.clone(), w: serial.w.clone(), glob_centre: serial.glob_centre.clone(), fit: false, forces: serial.forces.clone(), threads: 1 } };
        let v = serial.initial_vars();
        let lay = Layout::new(&serial);

        // Act
        let (a, b) = (normal_equations(&serial, &lay, &v, true), normal_equations(&threaded, &lay, &v, true));

        // Assert: the same system, to rounding
        let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(p, q)| (p - q).abs() <= 1e-9 * (1.0 + p.abs().max(q.abs())));
        assert!(close(&a.band, &b.band) && close(&a.border, &b.border) && close(&a.c, &b.c));
        assert!(close(&a.g_loc, &b.g_loc) && close(&a.g_glob, &b.g_glob));
        assert!((a.loss - b.loss).abs() <= 1e-9 * a.loss);
    }
}
