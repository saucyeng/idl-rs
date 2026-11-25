//! What each pod reads, given the latents and the chassis motion; and the
//! exact Jacobian of those readings, per sample. A port of `pod_readings` in
//! `ml/idlml/bike/model.py`.
//!
//! A pod's velocity and acceleration relative to the frame are the exact time
//! derivatives of its pose along the latent trajectory: evaluating the pose
//! with hyper-dual latents `q + q' e1 + q' e2 + q'' e1e2` gives the pose
//! (real part), its rate (e1) and its second derivative
//! `f''[q', q'] + f' q''` (e1e2) in one pass. Then rigid-body transport:
//!
//! ```text
//! w_pod = R^T (w0 + w_rel)
//! f_pod = R^T (f0 + a0 x p + w0 x (w0 x p) + 2 w0 x p' + p'')
//! ```
//!
//! with w0, a0 the rigid chassis's rate and angular acceleration and f0 its
//! specific force, recovered from the frame pod's measurement through the
//! frame pod's own flex. Gravity is common to all pods, so no attitude is needed.

use num_dual::{Dual64, HyperDual};

use super::bike::{Bike, Pose, B_FA, B_LAT, D, F_TWIST, NL, PSI, R_ROLL, R_YAW, S};
use super::la::{vee, M3, Real, V3};

/// Readings per sample: frame gyro (3), fork gyro (3), fork accel (3), rear
/// gyro (3), rear accel (3); rad/s and m/s^2, each in its own pod's axes.
pub const NR: usize = 15;
/// Local inputs per sample, in the solver's order: q, q', q'' (NL each), w0, a0 (3 each).
pub const NZ: usize = 3 * NL + 6;

/// One sample's inputs.
#[derive(Clone, Copy, Debug)]
pub struct Sample<S> {
    /// Latents.
    pub q: [S; NL],
    /// Their first time derivatives (per second).
    pub qd: [S; NL],
    /// Their second time derivatives (per second squared).
    pub qdd: [S; NL],
    /// Chassis angular rate, rad/s, chassis axes.
    pub w0: V3<S>,
    /// Chassis angular acceleration, rad/s^2.
    pub a0: V3<S>,
    /// The frame pod's measured specific force, m/s^2, its own axes.
    pub f_meas: V3<S>,
}

type Hd<S> = HyperDual<S>;
type PoseFn<S> = fn(&Bike<Hd<S>>, &[Hd<S>; NL]) -> Pose<Hd<S>>;

/// A pose (rotation pod -> IMU0 axes, position, m) and the time derivatives
/// the readings use: R' (1/s), p' (m/s), p'' (m/s^2).
#[derive(Clone, Copy, Debug)]
pub struct Motion<S> {
    /// Rotation, pod -> IMU0 axes.
    pub r: M3<S>,
    /// Position, m.
    pub p: V3<S>,
    /// dR/dt, 1/s.
    pub rd: M3<S>,
    /// dp/dt, m/s.
    pub pd: V3<S>,
    /// d2p/dt2, m/s^2.
    pub pdd: V3<S>,
}

impl<S: Real> Motion<S> {
    fn map<T: Real>(&self, f: impl Fn(S) -> T + Copy) -> Motion<T> {
        Motion { r: self.r.map(f), p: self.p.map(f), rd: self.rd.map(f), pd: self.pd.map(f), pdd: self.pdd.map(f) }
    }
}

/// The three poses in reading order: frame pod (its flex), fork pod, rear pod.
fn poses<S: Real>() -> [PoseFn<S>; 3] {
    [Bike::frame_pose, Bike::front_pose, Bike::rear_pose]
}

/// Which pose (0 frame, 1 fork, 2 rear) latent `i` moves: each latent moves exactly one.
pub const fn owner(i: usize) -> usize {
    match i {
        D | PSI | B_FA | B_LAT | F_TWIST => 1,
        S | R_ROLL | R_YAW => 2,
        _ => 0,
    }
}

/// Hyper-dual latents q + q' e1 + q' e2 + q'' e1e2: a pose evaluated in them
/// carries its first (e1) and second (e1e2) time derivatives along q(t).
fn hyper<S: Real>(s: &Sample<S>) -> [Hd<S>; NL] {
    std::array::from_fn(|i| HyperDual::new(s.q[i], s.qd[i], s.qd[i], s.qdd[i]))
}

fn motion<S: Real>(bike_h: &Bike<Hd<S>>, pose: PoseFn<S>, qh: &[Hd<S>; NL]) -> Motion<S> {
    let (r, p) = pose(bike_h, qh);
    Motion { r: r.map(|x| x.re), p: p.map(|x| x.re), rd: r.map(|x| x.eps1), pd: p.map(|x| x.eps1), pdd: p.map(|x| x.eps1eps2) }
}

/// Rigid-body transport from the chassis to each pod (see the module docs).
fn transport<S: Real>(m: &[Motion<S>; 3], w0: V3<S>, a0: V3<S>, f_meas: V3<S>) -> [S; NR] {
    let two = S::from(2.0);
    let fr = &m[0];
    let w_flex = vee(fr.rd * fr.r.t());
    let frame_gyro = fr.r.t() * (w0 + w_flex);
    let f0 = fr.r * f_meas - (a0.cross(fr.p) + w0.cross(w0.cross(fr.p)) + w0.cross(fr.pd).scale(two) + fr.pdd);
    let mut out = [S::from(0.0); NR];
    out[..3].copy_from_slice(&frame_gyro.0);
    for k in 0..2 {
        let pm = &m[k + 1];
        let w_rel = vee(pm.rd * pm.r.t());
        let f = f0 + a0.cross(pm.p) + w0.cross(w0.cross(pm.p)) + w0.cross(pm.pd).scale(two) + pm.pdd;
        let rt: M3<S> = pm.r.t();
        out[3 + 6 * k..6 + 6 * k].copy_from_slice(&(rt * (w0 + w_rel)).0);
        out[6 + 6 * k..9 + 6 * k].copy_from_slice(&(rt * f).0);
    }
    out
}

/// The pod readings for one sample (see the module docs). `bike_h` is the bike
/// lifted to hyper-dual numbers (`bike.map(HyperDual::from_re)`), passed in so
/// a caller evaluating many samples lifts it once.
pub fn pod_readings<S: Real>(bike_h: &Bike<Hd<S>>, s: &Sample<S>) -> [S; NR] {
    let qh = hyper(s);
    let m = poses::<S>().map(|f| motion(bike_h, f, &qh));
    transport(&m, s.w0, s.a0, s.f_meas)
}

/// `re` with the derivative part taken from `src`'s, times `k`.
fn with_eps_v(re: V3<f64>, src: V3<Dual64>, k: f64) -> V3<Dual64> {
    V3(std::array::from_fn(|i| Dual64::new(re.0[i], k * src.0[i].eps)))
}

fn with_eps_m(re: M3<f64>, src: M3<Dual64>) -> M3<Dual64> {
    M3(std::array::from_fn(|i| std::array::from_fn(|j| Dual64::new(re.0[i][j], src.0[i][j].eps))))
}

/// Readings and their exact Jacobian with respect to the [`NZ`] local inputs
/// (`jac[r][z]`). `bike_h` / `bike_dh`: the bike lifted to hyper-dual and to
/// hyper-dual-over-dual numbers, once per window.
///
/// One dual pass per latent, of the one pose that latent moves; its q' and q''
/// columns follow from the same pass, because p' = J q' and
/// p'' = J q'' + H[q', q']: d p''/d q''_j = dp/dq_j, d p'/d q'_j = dp/dq_j,
/// d p''/d q'_j = 2 d p'/d q_j, d R'/d q'_j = dR/dq_j. The w0 and a0 columns
/// need only the transport. 13 pose passes per sample instead of 45 x 3.
pub fn readings_and_jacobian(bike_h: &Bike<Hd<f64>>, bike_dh: &Bike<Hd<Dual64>>, s: &Sample<f64>) -> ([f64; NR], [[f64; NZ]; NR]) {
    let qh = hyper(s);
    let m = poses::<f64>().map(|f| motion(bike_h, f, &qh));
    let val = transport(&m, s.w0, s.a0, s.f_meas);
    let lift = |x: f64| Dual64::from(x);
    let md: [Motion<Dual64>; 3] = m.map(|x| x.map(lift));
    let (w0, a0, fm) = (s.w0.map(lift), s.a0.map(lift), s.f_meas.map(lift));
    let mut jac = [[0.0; NZ]; NR];
    let mut put = |col: usize, y: [Dual64; NR]| {
        for r in 0..NR {
            jac[r][col] = y[r].eps;
        }
    };
    let qdh: [Hd<Dual64>; NL] = std::array::from_fn(|i| HyperDual::new(lift(s.q[i]), lift(s.qd[i]), lift(s.qd[i]), lift(s.qdd[i])));
    let pose_d = poses::<Dual64>();
    for j in 0..NL {
        let k = owner(j);
        let mut q = qdh;
        q[j].re.eps = 1.0;
        let dm = motion(bike_dh, pose_d[k], &q); // values and d/dq_j of the one pose q_j moves
        let mut mm = md;
        mm[k] = dm;
        put(j, transport(&mm, w0, a0, fm));
        let mut mm = md;
        mm[k].rd = with_eps_m(m[k].rd, dm.r);
        mm[k].pd = with_eps_v(m[k].pd, dm.p, 1.0);
        mm[k].pdd = with_eps_v(m[k].pdd, dm.pd, 2.0);
        put(NL + j, transport(&mm, w0, a0, fm));
        let mut mm = md;
        mm[k].pdd = with_eps_v(m[k].pdd, dm.p, 1.0);
        put(2 * NL + j, transport(&mm, w0, a0, fm));
    }
    for i in 0..3 {
        let (mut w, mut a) = (w0, a0);
        w.0[i].eps = 1.0;
        put(3 * NL + i, transport(&md, w, a0, fm));
        a.0[i].eps = 1.0;
        put(3 * NL + 3 + i, transport(&md, w0, a, fm));
    }
    (val, jac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::physics::la::V3;

    fn golden() -> (Bike<f64>, serde_json::Value) {
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/physics_golden.json")).unwrap();
        (crate::estimate::physics::fixture::bike(&doc["bike"]), doc)
    }

    /// Latents in the PyTorch goldens: the first 13 (no fork torsion, which is 0 there).
    const GL: usize = 13;

    fn sample(j: &serde_json::Value) -> Sample<f64> {
        let z: Vec<f64> = serde_json::from_value(j["z"].clone()).unwrap();
        let f: Vec<f64> = serde_json::from_value(j["f_meas"].clone()).unwrap();
        let pick = |base: usize| -> [f64; NL] { std::array::from_fn(|i| if i < GL { z[base + i] } else { 0.0 }) };
        Sample {
            q: pick(0),
            qd: pick(GL),
            qdd: pick(2 * GL),
            w0: V3([z[3 * GL], z[3 * GL + 1], z[3 * GL + 2]]),
            a0: V3([z[3 * GL + 3], z[3 * GL + 4], z[3 * GL + 5]]),
            f_meas: V3([f[0], f[1], f[2]]),
        }
    }

    /// This port's input column for the goldens' column `c` (their layout has [`GL`] latents).
    fn col(c: usize) -> usize {
        match c {
            c if c < 3 * GL => (c / GL) * NL + c % GL,
            c => 3 * NL + (c - 3 * GL),
        }
    }

    #[test]
    fn pod_readings_and_jacobian_match_the_pytorch_reference() {
        // Arrange: the bike and samples the PyTorch model was evaluated on (float64)
        let (bike, doc) = golden();
        let bike_h = bike.map(HyperDual::from_re);
        let bike_dh = bike.map(|x| HyperDual::from_re(Dual64::from(x)));

        for sj in doc["samples"].as_array().unwrap() {
            // Act
            let (val, jac) = readings_and_jacobian(&bike_h, &bike_dh, &sample(sj));

            // Assert: readings to 1e-9 relative, Jacobian to 1e-7 of its row scale
            let want: Vec<f64> = serde_json::from_value(sj["readings"].clone()).unwrap();
            let wj: Vec<Vec<f64>> = serde_json::from_value(sj["jacobian"].clone()).unwrap();
            for r in 0..NR {
                assert!((val[r] - want[r]).abs() <= 1e-9 * (1.0 + want[r].abs()), "reading {r}: {} vs {}", val[r], want[r]);
                let scale = 1.0 + wj[r].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for z in 0..wj[r].len() {
                    let c = col(z);
                    assert!((jac[r][c] - wj[r][z]).abs() <= 1e-7 * scale, "d reading {r} / d input {z}: {} vs {}", jac[r][c], wj[r][z]);
                }
            }
        }
    }

    #[test]
    fn fork_torsion_columns_of_the_jacobian_match_finite_differences_and_move_only_the_fork_pod() {
        // Arrange: a golden sample with some torsion, its rate and acceleration
        let (bike, doc) = golden();
        let bike_h = bike.map(HyperDual::from_re);
        let bike_dh = bike.map(|x| HyperDual::from_re(Dual64::from(x)));
        let mut s = sample(&doc["samples"][0]);
        (s.q[F_TWIST], s.qd[F_TWIST], s.qdd[F_TWIST]) = (0.004, 0.3, 40.0);
        let h = 1e-7;

        // Act
        let (_, jac) = readings_and_jacobian(&bike_h, &bike_dh, &s);
        let fd = |which: usize| -> [f64; NR] {
            let (mut up, mut dn) = (s.clone(), s.clone());
            let (u, d) = match which {
                0 => (&mut up.q, &mut dn.q),
                1 => (&mut up.qd, &mut dn.qd),
                _ => (&mut up.qdd, &mut dn.qdd),
            };
            u[F_TWIST] += h;
            d[F_TWIST] -= h;
            let (a, b) = (pod_readings(&bike_h, &up), pod_readings(&bike_h, &dn));
            std::array::from_fn(|r| (a[r] - b[r]) / (2.0 * h))
        };

        // Assert: q, q', q'' columns; frame and rear readings (0..3, 9..15) untouched
        for (which, base) in [(0, 0), (1, NL), (2, 2 * NL)] {
            let want = fd(which);
            for r in 0..NR {
                assert!((jac[r][base + F_TWIST] - want[r]).abs() <= 1e-5 * (1.0 + want[r].abs()), "reading {r}, column {which}: {} vs {}", jac[r][base + F_TWIST], want[r]);
                if !(3..9).contains(&r) {
                    assert_eq!(jac[r][base + F_TWIST], 0.0, "reading {r} moved with fork torsion");
                }
            }
        }
        assert!(jac[3..6].iter().any(|row| row[F_TWIST + NL].abs() > 0.1), "fork gyro must see the torsion rate");
    }

    #[test]
    fn plain_and_dual_evaluation_agree() {
        // Arrange
        let (bike, doc) = golden();
        let s = sample(&doc["samples"][0]);

        // Act
        let plain = pod_readings(&bike.map(HyperDual::from_re), &s);
        let (dual, _) = readings_and_jacobian(&bike.map(HyperDual::from_re), &bike.map(|x| HyperDual::from_re(Dual64::from(x))), &s);

        // Assert
        for r in 0..NR {
            assert!((plain[r] - dual[r]).abs() < 1e-12 * (1.0 + plain[r].abs()));
        }
    }
}
