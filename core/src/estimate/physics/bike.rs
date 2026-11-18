//! The bike as pose functions: where each pod sits, in IMU0's axes, for given
//! latents. A port of `ml/idlml/bike/model.py` (the reference; the golden test
//! holds the two equal). Everything is generic over [`Real`], so the same code
//! gives values, time derivatives (hyper-dual) and Jacobians (dual).

use super::la::{c, exp, rot_axis, M3, Real, V3};

/// Number of latents per sample (see the index constants).
pub const NL: usize = 13;
/// Front travel along the steering axis, m (fork compression; 0 = top-out).
pub const D: usize = 0;
/// Steer angle about the steering axis, rad.
pub const PSI: usize = 1;
/// Fork fore-aft bend, rad (positive: axle moves rearward, as braking).
pub const B_FA: usize = 2;
/// Fork lateral bend, rad.
pub const B_LAT: usize = 3;
/// Rear wheel travel, m (vertical, at the axle).
pub const S: usize = 4;
/// Rear member roll flex about the fore-aft axis, rad.
pub const R_ROLL: usize = 5;
/// Rear member yaw flex about the up axis, rad.
pub const R_YAW: usize = 6;
/// Frame-pod flex vs the rigid chassis: rotation vector (rad), latents 7..10.
pub const F_ROT: usize = 7;
/// Frame-pod flex vs the rigid chassis: translation (m), latents 10..13.
pub const F_TRANS: usize = 10;

/// A planar 4-bar rear end, in the bike's side plane (x toward the front
/// axle, y up), relative to IMU0, m, at the photo's scale.
#[derive(Clone, Debug)]
pub struct FourBar<S> {
    /// Main pivot, Horst pivot, seatstay-rocker pivot, rocker-frame pivot.
    pub main_pivot: [S; 2],
    /// Horst (chainstay-seatstay) pivot at the drawn pose.
    pub horst: [S; 2],
    /// Seatstay-rocker pivot at the drawn pose.
    pub ss_rocker: [S; 2],
    /// Rocker-frame pivot.
    pub rocker_frame: [S; 2],
    /// Polynomial, rear travel (m, unscaled) -> chainstay angle (rad), lowest power first.
    pub coef: Vec<f64>,
    /// Which branch of the circle intersection the linkage is on (+1 or -1).
    pub sign: f64,
}

/// The rear end's path.
#[derive(Clone, Debug)]
pub enum Rear<S> {
    /// A 4-bar from the linkage photo tool.
    FourBar(FourBar<S>),
    /// Rotation `rot_per_m` (rad per m of travel) about `axis` through `pivot`
    /// plus translation along `dir`, for a bike without a linkage file.
    Generic {
        /// Lateral rotation axis (unit).
        axis: V3<S>,
        /// Point on the axis, m.
        pivot: V3<S>,
        /// Rotation per metre of travel, rad/m.
        rot_per_m: S,
        /// Direction of the travel's translation (unit).
        dir: V3<S>,
    },
}

/// Per-bike geometry, IMU0 axes, SI units.
#[derive(Clone, Debug)]
pub struct Bike<S> {
    /// Up, in IMU0 axes (unit).
    pub up: V3<S>,
    /// Forward (unit).
    pub fwd: V3<S>,
    /// Steering axis direction (normalised where used).
    pub steer_axis: V3<S>,
    /// A point on the steering axis, m.
    pub steer_point: V3<S>,
    /// Fork pod mount: pod axes -> IMU0 axes.
    pub fork_mount: M3<S>,
    /// Fork pod position at zero latents, m.
    pub fork_lever: V3<S>,
    /// Crown-to-fork-pod length at top-out, m.
    pub bend_len: S,
    /// Bushing overlap at top-out, m.
    pub bush_overlap: S,
    /// Whether the bending arm shortens with travel (false: fixed at `bend_len`).
    pub bend_vs_travel: bool,
    /// Rear pod mount: pod axes -> IMU0 axes.
    pub rear_mount: M3<S>,
    /// Rear pod position at zero latents, m.
    pub rear_lever: V3<S>,
    /// The rear end's path.
    pub rear: Rear<S>,
    /// Scale of the 4-bar's photo (1 = as drawn).
    pub link_scale: S,
    /// Fork mount correction, rotation vector in pod axes, rad (its part about the steer axis is unused).
    pub mount_err_fork: V3<S>,
    /// Rear mount correction, rotation vector in pod axes, rad.
    pub mount_err_rear: V3<S>,
}

/// A pose: (rotation pod -> IMU0 axes, position in IMU0 axes, m).
pub type Pose<S> = (M3<S>, V3<S>);

impl<S: Real> Bike<S> {
    /// The same bike in another scalar type (every number through `f`).
    pub fn map<T: Real>(&self, f: impl Fn(S) -> T + Copy) -> Bike<T> {
        let p2 = |p: [S; 2]| [f(p[0]), f(p[1])];
        Bike {
            up: self.up.map(f),
            fwd: self.fwd.map(f),
            steer_axis: self.steer_axis.map(f),
            steer_point: self.steer_point.map(f),
            fork_mount: self.fork_mount.map(f),
            fork_lever: self.fork_lever.map(f),
            bend_len: f(self.bend_len),
            bush_overlap: f(self.bush_overlap),
            bend_vs_travel: self.bend_vs_travel,
            rear_mount: self.rear_mount.map(f),
            rear_lever: self.rear_lever.map(f),
            rear: match &self.rear {
                Rear::FourBar(b) => Rear::FourBar(FourBar {
                    main_pivot: p2(b.main_pivot),
                    horst: p2(b.horst),
                    ss_rocker: p2(b.ss_rocker),
                    rocker_frame: p2(b.rocker_frame),
                    coef: b.coef.clone(),
                    sign: b.sign,
                }),
                Rear::Generic { axis, pivot, rot_per_m, dir } => {
                    Rear::Generic { axis: axis.map(f), pivot: pivot.map(f), rot_per_m: f(*rot_per_m), dir: dir.map(f) }
                }
            },
            link_scale: f(self.link_scale),
            mount_err_fork: self.mount_err_fork.map(f),
            mount_err_rear: self.mount_err_rear.map(f),
        }
    }

    /// Crown-to-pod length at front travel `d` (m), m.
    pub fn bend_arm(&self, d: S) -> S {
        if self.bend_vs_travel {
            self.bend_len - d
        } else {
            self.bend_len
        }
    }

    /// Relative fork bending compliance at travel `d` (1 at top-out): tip
    /// angle under a tip load goes as the arm squared, the bushings' share as
    /// arm / overlap. **Not used by the objective**: a bending prior whose sigma
    /// shrinks with travel makes less travel cheaper and biased front travel
    /// 67 -> 25 mm on the Beacon ride (`ml/README.md`). Kept for analysis.
    pub fn bend_compliance(&self, d: S) -> S {
        if !self.bend_vs_travel {
            return c(1.0);
        }
        let arm = self.bend_arm(d) / self.bend_len;
        let d0 = if d.re() > 0.0 { d } else { c(0.0) };
        arm * arm * self.bush_overlap / (self.bush_overlap + d0)
    }

    /// The fork pod: steer about the steering axis, travel along it, bending.
    pub fn front_pose(&self, q: &[S; NL]) -> Pose<S> {
        let a = self.steer_axis.unit();
        let fa = (self.fwd - a.scale(a.dot(self.fwd))).unit();
        let lat = a.cross(fa);
        let (bfa, blat) = (q[B_FA], q[B_LAT]);
        let kl = self.bend_arm(q[D]) * (2.0 / 3.0); // cantilever tip: deflection = 2/3 L angle
        let x = self.fork_lever + a.scale(q[D]) - lat.cross(a).scale(kl * bfa) - fa.cross(a).scale(kl * blat);
        let r_bend = rot_axis(lat, bfa) * rot_axis(fa, blat);
        let r_steer = rot_axis(a, q[PSI]);
        let p = self.steer_point + r_steer * (x - self.steer_point);
        let ap = self.fork_mount.t() * a; // steer axis in pod axes
        let e = self.mount_err_fork;
        let mount = self.fork_mount * exp(e - ap.scale(ap.dot(e)));
        (r_steer * r_bend * mount, p)
    }

    fn side_to_frame(&self, v: [S; 2]) -> V3<S> {
        self.fwd.scale(v[0]) + self.up.scale(v[1])
    }

    /// The rear pod.
    pub fn rear_pose(&self, q: &[S; NL]) -> Pose<S> {
        let s = q[S];
        let (r, pivot, p) = match &self.rear {
            Rear::FourBar(fb) => {
                let (phi, h2, h0) = fourbar_pose(fb, s, self.link_scale);
                let n = self.fwd.cross(self.up).unit();
                let r = rot_axis(n, phi);
                let h2 = self.side_to_frame(h2);
                let h0 = self.side_to_frame(h0);
                (r, h2, h2 + r * (self.rear_lever - h0))
            }
            Rear::Generic { axis, pivot, rot_per_m, dir } => {
                let r = rot_axis(axis.unit(), *rot_per_m * s);
                (r, *pivot, *pivot + r * (self.rear_lever - *pivot) + dir.unit().scale(s))
            }
        };
        let r_flex = rot_axis(self.fwd, q[R_ROLL]) * rot_axis(self.up, q[R_YAW]);
        let p = pivot + r_flex * (p - pivot);
        (r_flex * r * self.rear_mount * exp(self.mount_err_rear), p)
    }

    /// The frame pod (IMU0) relative to the rigid chassis.
    pub fn frame_pose(&self, q: &[S; NL]) -> Pose<S> {
        (exp(V3([q[F_ROT], q[F_ROT + 1], q[F_ROT + 2]])), V3([q[F_TRANS], q[F_TRANS + 1], q[F_TRANS + 2]]))
    }
}

/// Rear travel `s` (m) -> (seatstay angle phi (rad), Horst pivot, rest Horst
/// pivot), side plane relative to IMU0, scaled by `scale`.
fn fourbar_pose<S: Real>(fb: &FourBar<S>, s: S, scale: S) -> (S, [S; 2], [S; 2]) {
    let sc = |p: [S; 2]| [p[0] * scale, p[1] * scale];
    let (m, h, sr, r) = (sc(fb.main_pivot), sc(fb.horst), sc(fb.ss_rocker), sc(fb.rocker_frame));
    let u = s / scale;
    let mut th = c::<S>(0.0);
    for &k in fb.coef.iter().rev() {
        th = th * u + k; // Horner
    }
    let (st, ct) = th.sin_cos();
    let hm = [h[0] - m[0], h[1] - m[1]];
    let h2 = [m[0] + ct * hm[0] - st * hm[1], m[1] + st * hm[0] + ct * hm[1]];
    let len = |a: [S; 2], b: [S; 2]| ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])).sqrt();
    let (r0, r1) = (len(sr, h), len(sr, r));
    let dv = [r[0] - h2[0], r[1] - h2[1]];
    let dn = (dv[0] * dv[0] + dv[1] * dv[1]).sqrt();
    let a = (r0 * r0 - r1 * r1 + dn * dn) / (dn * 2.0);
    let hh2 = r0 * r0 - a * a;
    let hh = if hh2.re() > 1e-12 { hh2.sqrt() } else { c::<S>(1e-12).sqrt() };
    let un = [dv[0] / dn, dv[1] / dn];
    let s2 = [h2[0] + a * un[0] - hh * un[1] * fb.sign, h2[1] + a * un[1] + hh * un[0] * fb.sign];
    let v = [s2[0] - h2[0], s2[1] - h2[1]];
    let v0 = [sr[0] - h[0], sr[1] - h[1]];
    (v[1].atan2(v[0]) - v0[1].atan2(v0[0]), h2, h)
}
