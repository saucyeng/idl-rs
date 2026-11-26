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
use super::la::{c, M3, Real, V3};
use super::readings::{Sample, NZ};
use crate::sim::laws::{air_force_about_n, friction_n, step, stop_n, DamperLaw, TyreLaw};

/// Force latents per sample.
pub const NFL: usize = 18;
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

/// Force parameters (globals), each a log multiplier on [`ForceBike::nominal`].
pub const NFP: usize = 30;
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
/// Rider springs (x, y, z, roll, pitch, yaw): 18..24.
pub const P_RIDER_K: usize = 18;
/// Rider dampers: 24..30.
pub const P_RIDER_C: usize = 24;

/// Rows per sample (see the module table).
pub const NFR: usize = 30;
/// The balance rows (0..NBAL), the ones Huber weighting applies to.
pub const NBAL: usize = 14;

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
    /// The spring's force, N, at stroke `x_m` with the pressure scaled by `scale`.
    pub fn force_n<S: Real>(&self, scale: S, x_m: S) -> S {
        air_force_about_n(self.area_m2, self.positive_m3, self.negative_m3, scale * self.gauge_pa, c(self.rest_m), x_m, 1.4)
    }
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
    /// Frame inertia about its centre of mass, about (fwd, lat, up), kg m^2.
    pub i_frame: [f64; 3],
    /// Rider's inertia, about (fwd, lat, up), kg m^2.
    pub i_rider: [f64; 3],
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
        let (mut x, mut dx) = (c::<S>(0.0), c::<S>(0.0));
        for (k, &a) in self.shock_stroke.iter().enumerate().rev() {
            x = x * s + a;
            if k > 0 {
                dx = dx * s + a * k as f64;
            }
        }
        (x, dx)
    }
}

/// The active wrench's smoothness, times the core latents' ([`ForceSpec::smooth`]).
pub const ACTIVE_SMOOTH: f64 = 2000.0;

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
    /// the rider's pose every 2, his active wrench every 4 with a smoothness 2000 times the
    /// core's (a low-pass near 3 Hz: pumping and weight shifts are slow, and a fast active part
    /// would absorb what the dampers and the rider's springs do). `static_deflection_m`: where
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
    /// Each parameter's prior width, log units (1: a factor of e).
    pub par_sigma: [f64; NFP],
    /// Huber threshold on the balance rows, sigmas (0: plain least squares).
    pub huber: f64,
}

impl Default for ForceWeights {
    fn default() -> Self {
        let mut par_sigma = [1.0; NFP];
        par_sigma[P_PRESSURE] = 0.05;
        par_sigma[P_PRESSURE + 1] = 0.05;
        par_sigma[P_TYRE_K] = 0.3;
        par_sigma[P_TYRE_K + 1] = 0.3;
        ForceWeights {
            unsprung_n: 20.0,
            rider_n: 50.0,
            rider_nm: 20.0,
            system_n: 50.0,
            system_nm: 20.0,
            tyre_tan_n: 500.0,
            // Isaac, 2026-11-25: he shifts fore and aft about as much as up and down
            pos_m: [0.25, 0.1, 0.25],
            rot_rad: 0.3,
            act_n: 300.0,
            act_nm: 60.0,
            par_sigma,
            huber: 3.0,
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
        let n = law.force_n(f[d0], wl) + par(P_TYRE_C + i) * fd[d0] * step(f[d0], wl);
        (n, up.scale(n) + fwd.scale(f[d0 + 1]) + lat.scale(f[d0 + 2]))
    };
    let (nf, tf) = tyre(0, DEF_F);
    let (nr, tr) = tyre(1, DEF_R);
    let mut r = [c::<S>(0.0); NFR];
    // front unsprung along the fork axis
    let fork = fb.fork.force_n(par(P_PRESSURE), x.d[0]) + damper(P_FORK_DAMP).force_n(x.d[1], wv) + friction_n(par(P_FRICTION), x.d[1], wv) + stop_n(x.d[0], 0.0, fb.travel_max[0], fb.stop_n_m, wl);
    let uf = k.rf * V3::<S>::from_f64(meas.fork);
    r[0] = (uf.dot(axis) * fb.m_front - tf.dot(axis) + fork) * (1.0 / w.unsprung_n);
    // rear unsprung along its path: generalised force on the travel coordinate
    let xd = k.dx * x.s[1];
    let shock = fb.shock.force_n(par(P_PRESSURE + 1), k.x) + damper(P_SHOCK_DAMP).force_n(xd, wv) + friction_n(par(P_FRICTION + 1), xd, wv);
    let wheel = shock * k.dx + stop_n(x.s[0], 0.0, fb.travel_max[1], fb.stop_n_m, wl);
    let ur = k.rr * V3::<S>::from_f64(meas.rear);
    r[1] = (ur.dot(k.dar) * fb.m_rear - tr.dot(k.dar) + wheel) * (1.0 / w.unsprung_n);
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
    let rf_res = comps(f_rider.scale(c(fb.m_rider)) - bush_f - act_f);
    let wb = w0 + thd;
    let ab = a0 + thdd + w0.cross(thd);
    let m_rider_in = inertia(fb.i_rider, ab) + wb.cross(inertia(fb.i_rider, wb));
    let rm_res = comps(m_rider_in - bush_m - act_m);
    // whole system
    let ps = lift(fb.frame_cg);
    let f_frame = f0 + a0.cross(ps) + w0.cross(w0.cross(ps));
    let sf = comps(f_frame.scale(c(fb.m_frame)) + uf.scale(c(fb.m_front)) + ur.scale(c(fb.m_rear)) + f_rider.scale(c(fb.m_rider)) - tf - tr);
    let spin = up.cross(fwd).unit();
    let wheel_h = |i: usize| {
        let h = spin.scale(c(fb.wheel_inertia[i] * meas.speed / fb.wheel_radius[i]));
        spin.scale(c(fb.wheel_inertia[i] * meas.speed_rate / fb.wheel_radius[i])) + w0.cross(h)
    };
    let pcf = k.af - up.scale(-f[DEF_F] + fb.wheel_radius[0]);
    let pcr = k.ar - up.scale(-f[DEF_R] + fb.wheel_radius[1]);
    let sm = comps(
        ps.cross(f_frame.scale(c(fb.m_frame))) + k.af.cross(uf.scale(c(fb.m_front))) + k.ar.cross(ur.scale(c(fb.m_rear))) + pb.cross(f_rider.scale(c(fb.m_rider)))
            + inertia(fb.i_frame, a0) + w0.cross(inertia(fb.i_frame, w0)) + m_rider_in + wheel_h(0) + wheel_h(1)
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
    for i in 0..3 {
        r[18 + i] = f[POS + i] * (1.0 / w.pos_m[i]);
        r[21 + i] = f[ROT + i] * (1.0 / w.rot_rad);
        r[24 + i] = f[ACT_F + i] * (1.0 / w.act_n);
        r[27 + i] = f[ACT_M + i] * (1.0 / w.act_nm);
    }
    (r, Forces { tyre: [tf, tr], normal: [nf, nr], fork, shock, bush: [bush_f, bush_m] })
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
            i_frame: [0.3, 1.2, 1.0],
            i_rider: [5.0, 6.0, 2.0],
            wheel_inertia: [0.18, 0.2],
            wheel_radius: [0.37, 0.37],
            fork: AirChambers { area_m2: 8.0e-4, positive_m3: 1.6e-4, negative_m3: 8.0e-5, gauge_pa: 0.0, rest_m: 0.05 },
            shock: AirChambers { area_m2: 9.5e-4, positive_m3: 1.6e-4, negative_m3: 9.0e-5, gauge_pa: 0.0, rest_m: 0.045 / 2.6 },
            shock_stroke: vec![0.0, 1.0 / 2.6],
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
    }

    impl State {
        /// At rest: travel at the springs' rest strokes, the rider on his legs' static sag.
        fn at_rest(m: &ForceModel) -> State {
            let fb = &m.bike;
            let mut pose = [0.0; 6];
            pose[2] = -fb.m_rider * 9.80665 / fb.nominal[P_RIDER_K + 2];
            State { d: [fb.fork.rest_m, 0.0], s: [0.045, 0.0], w0: V3::zero(), pose, pose_rate: [0.0; 6], ctrl: [0.0; 3], ctrl_rate: [0.0; 3] }
        }
    }

    /// The unknowns of a substep, `u`: the frame pod's specific force (3, bike axes), the
    /// chassis angular acceleration (3, bike axes), d'', s'', the rider's pose'' (6).
    fn sample(m: &ForceModel, t: f64, st: &State, u: &[f64; 14]) -> (Sample<f64>, [[f64; NFL]; 3], ForceMeas) {
        let fb = &m.bike;
        let mut inp = inputs_at(m, t);
        inp.defl[0][0] += st.ctrl[0];
        inp.defl[0][1] += st.ctrl_rate[0];
        inp.defl[1][0] -= st.ctrl[0];
        inp.defl[1][1] -= st.ctrl_rate[0];
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
        (core, f, ForceMeas { frame: f0.0, fork: [0.0; 3], rear: [0.0; 3], speed: inp.speed[0], speed_rate: inp.speed[1] })
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
        let (core, f, mut meas) = sample(m, t, st, u);
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
        let bike_h = bike.map(num_dual::HyperDual::from_re);
        let mut st = State::at_rest(m);
        let sub = 20;
        let h = dt / sub as f64;
        let skip = (settle_s / dt).round() as usize;
        let mut truth = Truth { core: Vec::with_capacity(n), f: Vec::with_capacity(n), normal: Vec::with_capacity(n), speed: Vec::with_capacity(n), readings: Vec::with_capacity(n) };
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
        let jinv = jc.try_inverse().expect("the trims reach every axis");
        let hold_s = 0.2;
        for i in 0..skip + n {
            for k_ in 0..sub {
                let t = (i as f64 - skip as f64) * dt + k_ as f64 * h;
                let wb = nalgebra::Vector3::new(st.w0.dot(fb.fwd), st.w0.dot(fb.lat()), st.w0.dot(fb.up));
                let c_new = -(jinv * (wb / hold_s));
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
                }
                // semi-implicit Euler
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
            forces: Some(ForceWindow { speed_mps: truth.speed.iter().map(|s| s[0]).collect() }),
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
        let meas = ForceMeas { frame: obj.win.frame_accel[t], fork: obj.win.fork_accel[t], rear: obj.win.rear_accel[t], speed: 0.0, speed_rate: 0.0 };
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
        use crate::estimate::physics::objective::{Objective, Spec, Weights, NG};
        use crate::estimate::physics::solve::gauss_newton;
        let (bike, m) = model();
        let dt = 1.0 / 400.0;
        let n = (secs / dt).round() as usize;
        let truth = ride(&bike, &m, n, dt, 1.0);
        let mut obj = Objective::new(bike, window(&truth, 400.0), Spec::standard(n, dt), Weights::default(), [0.0; NG]).with_forces(m);
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
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), 60, 1e-5).unwrap();
        let out: Vec<_> = wrong.iter().map(|&(name, i, x)| (name, x, v.glob[g0 + i])).collect();
        eprintln!("identification, {secs} s ({} iterations): {}", info.iterations, out.iter().map(|(n, a, b)| format!("{n} {a:+.2} -> {b:+.3}")).collect::<Vec<_>>().join(", "));
        out
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

    #[test]
    fn force_rows_jacobian_matches_finite_differences() {
        // Arrange: a moving sample of the exact ride, parameters off their nominal
        let (bike, m) = model();
        let truth = ride(&bike, &m, 40, 1.0 / 400.0, 0.5);
        let t = 30;
        let s = truth.core[t];
        let [f, fd, fdd] = truth.f[t];
        let p = truth.readings[t];
        let meas = ForceMeas { frame: s.f_meas.0, fork: [p[6], p[7], p[8]], rear: [p[12], p[13], p[14]], speed: truth.speed[t][0], speed_rate: truth.speed[t][1] };
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
}
