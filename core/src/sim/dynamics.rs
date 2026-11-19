//! The chain loop's dynamics (spec §2.4): one point mass per link joint,
//! rigid links, and every rotating part as a body with its inertia. Links,
//! the stays and the ratchet are constraints, projected each small step
//! (XPBD); the chain's contact with its wheels is a force.
//!
//! **Wheels are circles at the pitch radius, and their teeth are grip.** A
//! link's roller pressing into a wheel feels a stiff, damped normal force,
//! and an elastic stick-slip friction anchored to the wheel, capped at
//! `tooth_grip` × the normal force. Nothing decides where the chain joins or
//! leaves a wheel: a link leaves where the chain stops pressing it on, which
//! for a taut chain is the tangent, and a little sooner or later in motion.
//! The reactions turn the wheel and, through a jockey's axle, the cage.
//!
//! The run is in the front triangle's frame, which accelerates and turns
//! with the bike: each node feels minus the specific force at its position
//! (gravity included, as an accelerometer there would read), plus Coriolis.
//! The rear end moves through the 4-bar at the prescribed travel.
//!
//! - Chainring: locked (the rider holds the cranks while coasting).
//! - Cassette: inertia, freehub drag while the wheel overruns it, and the
//!   ratchet: it can never turn forward past the wheel by more than the
//!   engagement gap, so at low speed chain growth pulls on the cranks.
//! - Jockeys and toothed rollers: free to spin, with inertia.
//! - Cage: inertia, mass (its weight and the bike's accelerations swing it),
//!   the spring toward tension, and the clutch's Coulomb friction against
//!   turning forward (slack).
//! - Contact surfaces: the chain can't pass through them; the speed it
//!   arrives at is recorded.

use crate::calibration::drivetrain::{SurfaceSide, CHAIN_PITCH_M};
use crate::calibration::linkage::{LinkagePose, P2};

use super::bike::{SimBike, Wheel, WheelRole, CHAIN_HALF_ROLLER_M};
use super::excite::{Excitation, Sample};
use super::kinematics::{linkage_points, role_name};
use super::path::close_cage;
use super::{SimError, SimErrorKind};

/// Solver settings.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DynConfig {
    /// Time step, seconds.
    pub dt_s: f64,
    /// Constraint passes per step.
    pub iterations: u32,
    /// Frames recorded per second of simulated time.
    pub frame_rate_hz: f64,
    /// Time the chain is left to settle under the first sample before recording, seconds.
    pub settle_s: f64,
    /// Velocity damping while settling, 1/s.
    pub settle_damping_per_s: f64,
    /// The chain's own damping (joint friction), 1/s.
    pub chain_damping_per_s: f64,
    /// One link's axial stiffness, N/m (a steel roller chain is about 1e7).
    pub link_stiffness_npm: f64,
    /// Crank speed forward, rad/s; 0 holds the cranks (coasting, the default).
    #[serde(default)]
    pub crank_rate_rad_s: f64,
    /// How hard a sprocket's teeth hold the chain from sliding round it, as a
    /// friction coefficient on its pitch circle (teeth are not modelled; their
    /// grip is). By the capstan relation a wrap of θ holds a tension ratio of
    /// e^(μθ), so 1.5 never slips under a real load, while a link the chain
    /// stops pressing on is free to leave.
    #[serde(default = "default_grip")]
    pub tooth_grip: f64,
    /// Contact stiffness between a link's roller and a wheel, N/m.
    #[serde(default = "default_contact_k")]
    pub contact_stiffness_npm: f64,
    /// Contact damping, N·s/m.
    #[serde(default = "default_contact_c")]
    pub contact_damping_nspm: f64,
    /// Friction between the chain and a contact surface (a rubber protector).
    #[serde(default = "default_surface_mu")]
    pub surface_friction: f64,
}

fn default_surface_mu() -> f64 {
    0.5
}

fn default_grip() -> f64 {
    1.5
}
fn default_contact_k() -> f64 {
    2.0e5
}
fn default_contact_c() -> f64 {
    20.0
}

impl Default for DynConfig {
    fn default() -> Self {
        DynConfig { dt_s: 5.0e-5, iterations: 2, frame_rate_hz: 500.0, settle_s: 0.4, settle_damping_per_s: 40.0, chain_damping_per_s: 1.5, link_stiffness_npm: 1.0e7, crank_rate_rad_s: 0.0, tooth_grip: 1.5, contact_stiffness_npm: 2.0e5, contact_damping_nspm: 20.0, surface_friction: 0.5 }
    }
}

/// One per-frame result channel.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Channel {
    /// Name ("Top span tension").
    pub name: String,
    /// Unit ("N").
    pub unit: String,
    /// One value per frame.
    pub values: Vec<f32>,
}

/// Totals over the run.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Summary {
    /// Times a link hit a contact surface faster than 0.05 m/s.
    pub impacts: u32,
    /// Fastest of them, m/s.
    pub max_impact_mps: f64,
    /// Total cage rotation in the slack direction against the clutch, degrees.
    pub clutch_slip_deg: f64,
    /// Highest top-span tension at the chainring, N.
    pub max_top_tension_n: f64,
    /// Highest pedal force from it, N.
    pub max_pedal_force_n: f64,
    /// Largest top-span deflection from straight, mm.
    pub max_top_deflection_mm: f64,
    /// Largest lower-span deflection from straight, mm.
    pub max_lower_deflection_mm: f64,
    /// Largest link stretch the solver left, mm (a check on the solve).
    pub max_link_error_mm: f64,
}

/// A run's result, frame by frame.
#[derive(Debug, Clone, PartialEq)]
pub struct DynResult {
    /// Gear (0 = the smallest cog).
    pub gear: usize,
    /// Chain nodes (links).
    pub nodes: usize,
    /// Frame times, seconds.
    pub frame_t_s: Vec<f64>,
    /// Node positions, metres: `[frame][node][x, y]`.
    pub node_xy: Vec<f32>,
    /// Names of the moving points drawn each frame (linkage points, wheel
    /// centres, surface points as `surface:<i>:<k>`).
    pub point_names: Vec<String>,
    /// Their positions, metres: `[frame][point][x, y]`.
    pub point_xy: Vec<f32>,
    /// Wheels, by role name, with their pin-line radius (m) and teeth (0 smooth).
    pub wheels: Vec<(String, f64, u32)>,
    /// Wheel rotation angles, rad: `[frame][wheel]`. Their centres are the
    /// points named `wheel:<role>`.
    pub wheel_angle: Vec<f32>,
    /// Per-frame channels.
    pub channels: Vec<Channel>,
    /// Totals.
    pub summary: Summary,
}

#[derive(Clone, Copy)]
enum Mount {
    Ring,
    Cog,
    Upper,
    Lower,
    Roller(usize),
}

#[derive(Clone, Copy)]
struct DWheel {
    mount: Mount,
    radius: f64,
    toothed: bool,
    teeth: u32,
    inv_inertia: f64,
    centre: P2,
    psi: f64,
    psi_prev: f64,
    omega: f64,
    /// Centre last step, m (the wheel's own velocity at a contact).
    centre_prev: P2,
}

#[derive(Clone, Copy)]
struct Contact {
    wheel: usize,
    /// Where the grip's spring is anchored: the touch-down point in the wheel's own frame, m.
    anchor: P2,
}

fn perp(v: P2) -> P2 {
    [-v[1], v[0]]
}

#[derive(Clone)]
struct Surf {
    a: P2,
    u: P2,
    n: P2,
    len: f64,
}

/// Runs the chain loop in gear `gear` (with the rollers `enabled` says are
/// fitted) through `exc`. `progress` gets the fraction done now and then and
/// returns false to stop.
pub fn simulate(bike: &SimBike, gear: usize, enabled: &dyn Fn(usize) -> bool, exc: &Excitation, cfg: &DynConfig, progress: &mut dyn FnMut(f64) -> bool) -> Result<DynResult, SimError> {
    exc.validate()?;
    if !(cfg.dt_s > 0.0 && cfg.dt_s < 1e-3 && cfg.frame_rate_hz > 0.0) {
        return Err(SimError::new(SimErrorKind::Excitation, "the time step must be between 0 and 1 ms"));
    }
    let d = &bike.drivetrain;
    let dr = &d.derailleur;
    let (t_start, t_end) = exc.span();
    let s0 = exc.at(t_start);
    let pose0 = bike.pose(s0.travel_m);
    let (fwd0, path0) = close_cage(bike, &pose0, gear, enabled)?;
    let n = d.chain_links as usize;
    let p = CHAIN_PITCH_M;
    let m_node = d.chain_mass_kg / n as f64;
    let w_node = 1.0 / m_node;
    let imu0 = bike.points.imu0;
    let sigma = bike.cage_fwd_sign();
    let dt = cfg.dt_s;

    // nodes along the closed path, one per pitch
    let mut x: Vec<P2> = (0..n).map(|k| path0.point_at(k as f64 * p)).collect();
    let mut xp = x.clone();
    let mut v = vec![[0.0f64; 2]; n];

    // wheels: everything the chain can touch
    let cog_teeth = bike.cog_teeth(gear)?;
    let mut wheels: Vec<DWheel> = Vec::new();
    let mk = |mount: Mount, w: &Wheel, teeth: u32, inertia: f64| DWheel {
        mount,
        radius: w.radius,
        toothed: w.toothed,
        teeth,
        inv_inertia: if inertia > 0.0 { 1.0 / inertia } else { 0.0 },
        centre: w.centre,
        psi: 0.0,
        psi_prev: 0.0,
        omega: 0.0,
        centre_prev: w.centre,
    };
    let mut cage = bike.cage_abs(&pose0, fwd0);
    let [cog_w, ring_w, low_w, up_w] = bike.fixed_wheels(&pose0, gear, cage)?;
    wheels.push(mk(Mount::Ring, &ring_w, d.chainring_teeth, 0.0));
    wheels.push(mk(Mount::Cog, &cog_w, cog_teeth, d.cassette_inertia_kgm2));
    wheels.push(mk(Mount::Upper, &up_w, dr.jockey_teeth[0], dr.jockey_inertia_kgm2));
    wheels.push(mk(Mount::Lower, &low_w, dr.jockey_teeth[1], dr.jockey_inertia_kgm2));
    for i in 0..d.rollers.len() {
        if enabled(i) {
            let w = bike.roller(&pose0, i);
            let r = &d.rollers[i];
            wheels.push(mk(Mount::Roller(i), &w, r.teeth.unwrap_or(0), if r.teeth.is_some() { r.inertia_kgm2 } else { 0.0 }));
        }
    }
    const RING: usize = 0;
    const COG: usize = 1;
    let jockey_inv_cage = 1.0 / dr.cage_inertia_kgm2.max(1e-9);

    let place_wheels = |wheels: &mut [DWheel], pose: &LinkagePose, cage: f64| {
        let (uj, lj) = bike.jockeys(pose, gear, cage);
        for w in wheels.iter_mut() {
            w.centre = match w.mount {
                Mount::Ring => d.bb_m,
                Mount::Cog => pose.seatstay_point(bike.points.rear_axle),
                Mount::Upper => uj,
                Mount::Lower => lj,
                Mount::Roller(i) => bike.place(pose, d.rollers[i].body, d.rollers[i].centre_m),
            };
        }
    };

    let surfaces_at = |pose: &LinkagePose| -> Vec<(usize, Surf)> {
        let mut out = Vec::new();
        for (j, s) in d.contacts.iter().enumerate() {
            let pts: Vec<P2> = s.polyline_m.iter().map(|&q| bike.place(pose, s.body, q)).collect();
            for w in pts.windows(2) {
                let dd = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
                let len = (dd[0] * dd[0] + dd[1] * dd[1]).sqrt();
                if len < 1e-6 {
                    continue;
                }
                let u = [dd[0] / len, dd[1] / len];
                let mut nn = perp(u);
                let up = nn[1] >= 0.0;
                if (s.side == SurfaceSide::Above) != up {
                    nn = [-nn[0], -nn[1]];
                }
                out.push((j, Surf { a: w[0], u, n: nn, len }));
            }
        }
        out
    };

    // which wheel each link's roller is pressing on, with its grip's anchor
    let mut contact: Vec<Option<Contact>> = vec![None; n];
    let (k_n, c_n) = (cfg.contact_stiffness_npm, cfg.contact_damping_nspm);
    let mut f_contact = vec![[0.0f64; 2]; n];

    // the cage
    let mut cage_prev;
    let mut cage_omega = 0.0;
    let fwd_of = |pose: &LinkagePose, cage: f64| (cage - pose.seatstay_rad) * sigma;
    // the freehub
    let engage = d.freehub_engagement_deg.max(0.1).to_radians();
    let mut wheel_fwd = 0.0f64; // the wheel's forward rotation relative to the seatstay, rad
    let mut q_stop = 0.0f64;

    // frame bookkeeping
    let names: Vec<String> = {
        let mut v: Vec<String> = linkage_points(bike, &pose0).into_iter().map(|(k, _)| k).collect();
        v.push("cage_pivot".into());
        for w in &wheels {
            v.push(format!("wheel:{}", role_name(mount_role(w.mount))));
        }
        for (j, s) in d.contacts.iter().enumerate() {
            for k in 0..s.polyline_m.len() {
                v.push(format!("surface:{j}:{k}"));
            }
        }
        v
    };
    let mut out = DynResult {
        gear,
        nodes: n,
        frame_t_s: Vec::new(),
        node_xy: Vec::new(),
        point_names: names,
        point_xy: Vec::new(),
        wheels: wheels.iter().map(|w| (role_name(mount_role(w.mount)), w.radius, if w.toothed { w.teeth } else { 0 })).collect(),
        wheel_angle: Vec::new(),
        channels: [
            ("Rear travel", "mm"),
            ("Cage angle", "deg"),
            ("Clutch slipping", ""),
            ("Clutch slip total", "deg"),
            ("Top span tension", "N"),
            ("Lower span tension", "N"),
            ("Pedal force", "N"),
            ("Top span deflection", "mm"),
            ("Lower span deflection", "mm"),
            ("Links on surfaces", ""),
            ("Impact speed", "m/s"),
            ("Impacts", ""),
            ("Links on chainring", ""),
            ("Freehub engaged", ""),
            ("Link error", "mm"),
        ]
        .iter()
        .map(|(a, b)| Channel { name: a.to_string(), unit: b.to_string(), values: Vec::new() })
        .collect(),
        summary: Summary { impacts: 0, max_impact_mps: 0.0, clutch_slip_deg: 0.0, max_top_tension_n: 0.0, max_pedal_force_n: 0.0, max_top_deflection_mm: 0.0, max_lower_deflection_mm: 0.0, max_link_error_mm: 0.0 },
    };

    let settle_steps = (cfg.settle_s / dt).ceil() as usize;
    let run_steps = ((t_end - t_start) / dt).ceil() as usize;
    let total = settle_steps + run_steps;
    let frame_every = (1.0 / (cfg.frame_rate_hz * dt)).round().max(1.0) as usize;
    let mut surfs_prev: Vec<(usize, Surf)> = Vec::new();
    let mut lam = vec![0.0f64; n];
    let alpha_t = 1.0 / (cfg.link_stiffness_npm.max(1.0) * dt * dt);
    let mut on_surface = vec![false; n];
    let mut v_pre = vec![[0.0f64; 2]; n];
    let mut acc_top = 0.0;
    let mut acc_low = 0.0;
    let mut acc_k = 0usize;
    let mut frame_impact = 0.0f64;
    let mut slipping_frame = false;
    let mut pose;
    let mut ss_prev = pose0.seatstay_rad;
    let mut piv_hist = [bike.cage_pivot(&pose0, gear); 3];
    let ring_len_r = d.chainring_teeth as f64 * p / std::f64::consts::TAU;
    let mut last_report = 0usize;

    for step in 0..total {
        let settling = step < settle_steps;
        let t = if settling { t_start } else { t_start + (step - settle_steps + 1) as f64 * dt };
        let s: Sample = exc.at(t);
        pose = bike.pose(s.travel_m);
        let ss_rate = (pose.seatstay_rad - ss_prev) / dt;
        ss_prev = pose.seatstay_rad;
        piv_hist = [piv_hist[1], piv_hist[2], bike.cage_pivot(&pose, gear)];
        let a_piv = if settling { [0.0, 0.0] } else { [(piv_hist[2][0] - 2.0 * piv_hist[1][0] + piv_hist[0][0]) / (dt * dt), (piv_hist[2][1] - 2.0 * piv_hist[1][1] + piv_hist[0][1]) / (dt * dt)] };
        let spec = |q: P2| -> P2 {
            let r = [q[0] - imu0[0], q[1] - imu0[1]];
            [s.f[0] - s.alpha * r[1] - s.omega * s.omega * r[0], s.f[1] + s.alpha * r[0] - s.omega * s.omega * r[1]]
        };
        let damp = (-(if settling { cfg.settle_damping_per_s } else { cfg.chain_damping_per_s }) * dt).exp();

        // the stays where they are now; their speed from where they were
        let surfs = surfaces_at(&pose);
        if surfs_prev.len() != surfs.len() {
            surfs_prev = surfs.clone();
        }
        // contact forces: each link's roller against the wheel it presses into
        let mut wheel_torque = vec![0.0f64; wheels.len()];
        let rot: Vec<(f64, f64)> = wheels.iter().map(|w| w.psi.sin_cos()).collect();
        let mut cage_torque = 0.0f64;
        let piv = piv_hist[2];
        for i in 0..n {
            f_contact[i] = [0.0, 0.0];
            let mut deepest: Option<(usize, f64)> = None;
            for (k, w) in wheels.iter().enumerate() {
                // most links are nowhere near most wheels: skip them without a square root
                let d2 = (x[i][0] - w.centre[0]).powi(2) + (x[i][1] - w.centre[1]).powi(2);
                if d2 >= w.radius * w.radius || d2 <= 0.25 * w.radius * w.radius {
                    continue;
                }
                let gap = d2.sqrt() - w.radius;
                if deepest.map_or(true, |b| gap < b.1) {
                    deepest = Some((k, gap));
                }
            }
            let Some((k, gap)) = deepest else {
                contact[i] = None;
                continue;
            };
            let w = &wheels[k];
            let rv = [x[i][0] - w.centre[0], x[i][1] - w.centre[1]];
            let dist = (rv[0] * rv[0] + rv[1] * rv[1]).sqrt().max(1e-9);
            let nn = [rv[0] / dist, rv[1] / dist];
            let tt = perp(nn);
            // the wheel's surface velocity there: its centre's, plus its spin
            let vc = [(w.centre[0] - w.centre_prev[0]) / dt, (w.centre[1] - w.centre_prev[1]) / dt];
            let vs = [vc[0] - w.omega * rv[1], vc[1] + w.omega * rv[0]];
            let vr = [v[i][0] - vs[0], v[i][1] - vs[1]];
            let (vn, vt) = (vr[0] * nn[0] + vr[1] * nn[1], vr[0] * tt[0] + vr[1] * tt[1]);
            let f_n = (-k_n * gap - c_n * vn).max(0.0);
            let mu = if w.toothed { cfg.tooth_grip } else { 0.0 };
            let mut f_t = 0.0;
            if mu > 0.0 {
                // the wheel's frame: world = centre + R(psi) local
                let (sp, cp) = rot[k];
                let to_local = |q: P2| [cp * q[0] + sp * q[1], -sp * q[0] + cp * q[1]];
                let on_circle = [w.radius * nn[0], w.radius * nn[1]];
                let ct = match contact[i] {
                    Some(c) if c.wheel == k => c,
                    _ => Contact { wheel: k, anchor: to_local(on_circle) },
                };
                // the grip: a stiff spring to where the link touched down, sliding past mu × N
                let a = ct.anchor;
                let aw = [cp * a[0] - sp * a[1], sp * a[0] + cp * a[1]];
                let slip = (rv[0] - aw[0]) * tt[0] + (rv[1] - aw[1]) * tt[1];
                let trial = -k_n * slip - c_n * vt;
                let cap = mu * f_n;
                let anchor = if trial.abs() > cap {
                    f_t = cap * trial.signum();
                    // re-anchor so the spring holds exactly the sliding force
                    let s_new = -f_t / k_n;
                    to_local([on_circle[0] - s_new * tt[0], on_circle[1] - s_new * tt[1]])
                } else {
                    f_t = trial;
                    a
                };
                contact[i] = Some(Contact { wheel: k, anchor });
            } else {
                contact[i] = Some(Contact { wheel: k, anchor: [0.0, 0.0] });
            }
            let f = [f_n * nn[0] + f_t * tt[0], f_n * nn[1] + f_t * tt[1]];
            f_contact[i] = f;
            // the reaction turns the wheel, and a jockey's axle carries it to the cage
            wheel_torque[k] -= rv[0] * f[1] - rv[1] * f[0];
            if matches!(w.mount, Mount::Upper | Mount::Lower) {
                let arm = [w.centre[0] - piv[0], w.centre[1] - piv[1]];
                cage_torque -= arm[0] * f[1] - arm[1] * f[0];
            }
        }

        // and against the stays: the same stiff, damped contact, with rubber's friction.
        // Only a link on the surface's side, within 6 mm of it, feels it, so a span on
        // the far side of a stay is not pulled through.
        // a box round each surface, 6 mm proud, to skip the links nowhere near it
        let boxes: Vec<[f64; 4]> = surfs
            .iter()
            .map(|(_, sf)| {
                let b = [sf.a[0] + sf.len * sf.u[0], sf.a[1] + sf.len * sf.u[1]];
                let m = 6.0e-3 + CHAIN_HALF_ROLLER_M;
                [sf.a[0].min(b[0]) - m, sf.a[0].max(b[0]) + m, sf.a[1].min(b[1]) - m, sf.a[1].max(b[1]) + m]
            })
            .collect();
        for i in 0..n {
            for (si, (_, sf)) in surfs.iter().enumerate() {
                let bx = &boxes[si];
                if x[i][0] < bx[0] || x[i][0] > bx[1] || x[i][1] < bx[2] || x[i][1] > bx[3] {
                    continue;
                }
                let w = [x[i][0] - sf.a[0], x[i][1] - sf.a[1]];
                let along = w[0] * sf.u[0] + w[1] * sf.u[1];
                if along < 0.0 || along > sf.len {
                    continue;
                }
                let gap = w[0] * sf.n[0] + w[1] * sf.n[1] - CHAIN_HALF_ROLLER_M;
                if gap >= 0.0 || gap < -6.0e-3 {
                    continue;
                }
                let sp = &surfs_prev[si].1;
                let fr = (along / sf.len).clamp(0.0, 1.0);
                let now = [sf.a[0] + fr * sf.len * sf.u[0], sf.a[1] + fr * sf.len * sf.u[1]];
                let was = [sp.a[0] + fr * sp.len * sp.u[0], sp.a[1] + fr * sp.len * sp.u[1]];
                let vr = [v[i][0] - (now[0] - was[0]) / dt, v[i][1] - (now[1] - was[1]) / dt];
                let (vn, vt) = (vr[0] * sf.n[0] + vr[1] * sf.n[1], vr[0] * sf.u[0] + vr[1] * sf.u[1]);
                let f_n = (-k_n * gap - c_n * vn).max(0.0);
                let f_t = -cfg.surface_friction * f_n * (vt / 0.02).tanh();
                f_contact[i][0] += f_n * sf.n[0] + f_t * sf.u[0];
                f_contact[i][1] += f_n * sf.n[1] + f_t * sf.u[1];
            }
        }

        // predict the nodes
        for i in 0..n {
            let f = spec(x[i]);
            let cor = [2.0 * s.omega * v[i][1], -2.0 * s.omega * v[i][0]];
            let fc = f_contact[i];
            v[i][0] = (v[i][0] + (-f[0] + cor[0] + fc[0] * w_node) * dt) * damp;
            v[i][1] = (v[i][1] + (-f[1] + cor[1] + fc[1] * w_node) * dt) * damp;
            v_pre[i] = v[i];
            xp[i] = x[i];
            x[i][0] += v[i][0] * dt;
            x[i][1] += v[i][1] * dt;
        }
        // predict the wheels
        let wheel_rate = if settling { 0.0 } else { s.speed_mps / d.wheel_radius_m.max(0.05) };
        for (k, w) in wheels.iter_mut().enumerate() {
            let mut torque_acc = -s.alpha + wheel_torque[k] * w.inv_inertia;
            if k == COG && w.inv_inertia > 0.0 {
                // forward is CW (negative); drag pulls the cassette toward the wheel's speed
                let cog_fwd = -(w.omega - ss_rate);
                torque_acc += -d.freehub_drag_nm * ((wheel_rate - cog_fwd) / 1.0).tanh() * w.inv_inertia;
            }
            if w.inv_inertia > 0.0 {
                w.omega = (w.omega + torque_acc * dt) * if settling { damp } else { 1.0 };
            } else if matches!(w.mount, Mount::Ring) {
                // the cranks: held, or turned forward (CW) at the set rate
                w.omega = if settling { 0.0 } else { -cfg.crank_rate_rad_s };
            } else {
                w.omega = ss_rate;
            }
            w.psi_prev = w.psi;
            w.psi += w.omega * dt;
        }
        // predict the cage
        {
            let fwd_deg = fwd_of(&pose, cage).to_degrees();
            let spring = -sigma * dr.spring_torque(fwd_deg);
            let (uj, lj) = bike.jockeys(&pose, gear, cage);
            let com = [0.5 * (uj[0] + lj[0]), 0.5 * (uj[1] + lj[1])];
            let piv = piv_hist[2];
            let r = [com[0] - piv[0], com[1] - piv[1]];
            let fc = spec(com);
            let force = [-dr.cage_mass_kg * (fc[0] + a_piv[0]), -dr.cage_mass_kg * (fc[1] + a_piv[1])];
            let weight = r[0] * force[1] - r[1] * force[0];
            let acc = (spring + weight + cage_torque) * jockey_inv_cage - s.alpha;
            cage_omega = (cage_omega + acc * dt) * if settling { damp } else { 1.0 };
            cage_prev = cage;
            cage += cage_omega * dt;
        }
        for w in wheels.iter_mut() {
            w.centre_prev = w.centre;
        }
        place_wheels(&mut wheels, &pose, cage);
        for l in lam.iter_mut() {
            *l = 0.0;
        }

        // solve
        for it in 0..cfg.iterations {
            // links, alternating direction
            for jj in 0..n {
                let i = if it % 2 == 0 { jj } else { n - 1 - jj };
                let j = if i + 1 == n { 0 } else { i + 1 };
                let dd = [x[j][0] - x[i][0], x[j][1] - x[i][1]];
                let dist = (dd[0] * dd[0] + dd[1] * dd[1]).sqrt().max(1e-12);
                let c = dist - p;
                let nn = [dd[0] / dist, dd[1] / dist];
                // XPBD with the link's compliance: lam is its force × dt² so far this step
                let corr = (c - alpha_t * lam[i]) / (2.0 * w_node + alpha_t);
                x[i][0] += w_node * corr * nn[0];
                x[i][1] += w_node * corr * nn[1];
                x[j][0] -= w_node * corr * nn[0];
                x[j][1] -= w_node * corr * nn[1];
                lam[i] += corr;
            }
            // the ratchet: the cassette can't run forward past the wheel
            let q = -(wheels[COG].psi - pose.seatstay_rad) - wheel_fwd - wheel_rate * dt;
            if q > q_stop {
                wheels[COG].psi += q - q_stop;
            }
        }

        // velocities
        for i in 0..n {
            v[i] = [(x[i][0] - xp[i][0]) / dt, (x[i][1] - xp[i][1]) / dt];
        }
        for w in wheels.iter_mut() {
            w.omega = (w.psi - w.psi_prev) / dt;
        }
        cage_omega = (cage - cage_prev) / dt;
        // clutch and bearing friction on the cage, relative to the seatstay
        let rel = cage_omega - ss_rate;
        let fwd_rate = sigma * rel;
        let slip;
        if fwd_rate > 0.0 {
            let stop = (dr.clutch_torque_nm + dr.bearing_torque_nm) * dt * jockey_inv_cage;
            let red = fwd_rate.min(stop);
            cage_omega -= sigma * red;
            slip = fwd_rate - red;
        } else {
            let red = (-fwd_rate).min(dr.bearing_torque_nm * dt * jockey_inv_cage);
            cage_omega += sigma * red;
            slip = 0.0;
        }
        if !settling && slip > 1e-3 {
            slipping_frame = true;
            out.summary.clutch_slip_deg += (slip * dt).to_degrees();
        }
        // the freehub
        wheel_fwd += wheel_rate * dt;
        let q = -(wheels[COG].psi - pose.seatstay_rad) - wheel_fwd;
        while q < q_stop - engage {
            q_stop -= engage;
        }
        let engaged = q >= q_stop - 1e-4;
        // surface impacts
        let mut touching = 0usize;
        for i in 0..n {
            let mut on = false;
            for (si, (_, sf)) in surfs.iter().enumerate() {
                let w = [x[i][0] - sf.a[0], x[i][1] - sf.a[1]];
                let along = w[0] * sf.u[0] + w[1] * sf.u[1];
                if along < 0.0 || along > sf.len {
                    continue;
                }
                let dist = w[0] * sf.n[0] + w[1] * sf.n[1];
                if dist < CHAIN_HALF_ROLLER_M + 2e-4 && dist > CHAIN_HALF_ROLLER_M - 2e-3 {
                    on = true;
                    if !on_surface[i] && !settling {
                        // the surface's own speed at that point, from its last step
                        let sp = &surfs_prev[si].1;
                        let f = (along / sf.len).clamp(0.0, 1.0);
                        let now = [sf.a[0] + f * sf.len * sf.u[0], sf.a[1] + f * sf.len * sf.u[1]];
                        let was = [sp.a[0] + f * sp.len * sp.u[0], sp.a[1] + f * sp.len * sp.u[1]];
                        let vs = [(now[0] - was[0]) / dt, (now[1] - was[1]) / dt];
                        let speed = -((v_pre[i][0] - vs[0]) * sf.n[0] + (v_pre[i][1] - vs[1]) * sf.n[1]);
                        if speed > 0.05 {
                            out.summary.impacts += 1;
                            out.summary.max_impact_mps = out.summary.max_impact_mps.max(speed);
                            frame_impact = frame_impact.max(speed);
                        }
                    }
                }
            }
            on_surface[i] = on;
            touching += on as usize;
        }
        surfs_prev = surfs;
        // span tensions at the chainring
        let on_ring = |i: usize| matches!(contact[i], Some(c) if c.wheel == RING);
        let (k0, k1) = ring_run(n, &on_ring);
        if let (Some(k0), Some(k1)) = (k0, k1) {
            // lam is the link's force × dt² this step
            let tension = |i: usize| lam[i].max(0.0) / (dt * dt);
            acc_top += tension((k0 + n - 1) % n);
            acc_low += tension(k1);
            acc_k += 1;
        }

        if settling {
            continue;
        }
        let k_run = step - settle_steps;
        if k_run % frame_every == 0 || step + 1 == total {
            let tt = if acc_k > 0 { acc_top / acc_k as f64 } else { 0.0 };
            let tl = if acc_k > 0 { acc_low / acc_k as f64 } else { 0.0 };
            acc_top = 0.0;
            acc_low = 0.0;
            acc_k = 0;
            let free = |i: usize| contact[i].is_none();
            let (td, ld) = match (k0, k1) {
                (Some(k0), Some(k1)) => (deflection(&x, n, (k0 + n - 1) % n, false, &free), deflection(&x, n, (k1 + 1) % n, true, &free)),
                _ => (0.0, 0.0),
            };
            let mut link_err: f64 = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                link_err = link_err.max((((x[j][0] - x[i][0]).powi(2) + (x[j][1] - x[i][1]).powi(2)).sqrt() - p).abs());
            }
            let pedal = tt * ring_len_r / d.crank_length_m.max(0.05);
            let sm = &mut out.summary;
            sm.max_top_tension_n = sm.max_top_tension_n.max(tt);
            sm.max_pedal_force_n = sm.max_pedal_force_n.max(pedal);
            sm.max_top_deflection_mm = sm.max_top_deflection_mm.max(td * 1000.0);
            sm.max_lower_deflection_mm = sm.max_lower_deflection_mm.max(ld * 1000.0);
            sm.max_link_error_mm = sm.max_link_error_mm.max(link_err * 1000.0);
            let ring_links = (0..n).filter(|&i| on_ring(i)).count();
            let vals = [
                s.travel_m * 1000.0,
                fwd_of(&pose, cage).to_degrees(),
                slipping_frame as u8 as f64,
                out.summary.clutch_slip_deg,
                tt,
                tl,
                pedal,
                td * 1000.0,
                ld * 1000.0,
                touching as f64,
                frame_impact,
                out.summary.impacts as f64,
                ring_links as f64,
                engaged as u8 as f64,
                link_err * 1000.0,
            ];
            for (c, v) in out.channels.iter_mut().zip(vals) {
                c.values.push(v as f32);
            }
            frame_impact = 0.0;
            slipping_frame = false;
            out.frame_t_s.push(t);
            for q in &x {
                out.node_xy.push(q[0] as f32);
                out.node_xy.push(q[1] as f32);
            }
            for (_, q) in linkage_points(bike, &pose) {
                out.point_xy.push(q[0] as f32);
                out.point_xy.push(q[1] as f32);
            }
            let piv = bike.cage_pivot(&pose, gear);
            out.point_xy.extend([piv[0] as f32, piv[1] as f32]);
            for w in &wheels {
                out.point_xy.extend([w.centre[0] as f32, w.centre[1] as f32]);
                out.wheel_angle.push(w.psi as f32);
            }
            for sf in &d.contacts {
                for &q in &sf.polyline_m {
                    let q = bike.place(&pose, sf.body, q);
                    out.point_xy.extend([q[0] as f32, q[1] as f32]);
                }
            }
        }
        if step - last_report > total / 100 + 1 {
            last_report = step;
            if !progress(step as f64 / total as f64) {
                return Err(SimError::new(SimErrorKind::Excitation, "stopped"));
            }
        }
        if x.iter().any(|q| !q[0].is_finite() || !q[1].is_finite()) {
            return Err(SimError::new(SimErrorKind::NoPath, format!("the solve diverged at {:.3} s; try a smaller time step", t)));
        }
    }
    progress(1.0);
    Ok(out)
}

fn mount_role(m: Mount) -> WheelRole {
    match m {
        Mount::Ring => WheelRole::Ring,
        Mount::Cog => WheelRole::Cog,
        Mount::Upper => WheelRole::UpperJockey,
        Mount::Lower => WheelRole::LowerJockey,
        Mount::Roller(i) => WheelRole::Roller(i),
    }
}

/// The chainring's nodes as one run: its first node (where the top span
/// arrives) and its last (where the lower span leaves), split at the
/// largest gap between ring nodes. A link lifted in the middle of the wrap
/// stays inside the run.
fn ring_run(n: usize, on: &dyn Fn(usize) -> bool) -> (Option<usize>, Option<usize>) {
    let ring: Vec<usize> = (0..n).filter(|&i| on(i)).collect();
    if ring.is_empty() {
        return (None, None);
    }
    let mut best = (0usize, 0usize);
    for (k, &i) in ring.iter().enumerate() {
        let next = ring[(k + 1) % ring.len()];
        let gap = (next + n - i) % n;
        let gap = if gap == 0 { n } else { gap };
        if gap > best.1 {
            best = (k, gap);
        }
    }
    (Some(ring[(best.0 + 1) % ring.len()]), Some(ring[best.0]))
}

/// Largest distance of a free run's nodes from the chord across it, metres.
/// The run starts at `from` and goes backward (`forward` false) or forward
/// while nodes are free of every wheel.
fn deflection(x: &[P2], n: usize, from: usize, forward: bool, free: &dyn Fn(usize) -> bool) -> f64 {
    let step = |i: usize| if forward { (i + 1) % n } else { (i + n - 1) % n };
    let back = |i: usize| if forward { (i + n - 1) % n } else { (i + 1) % n };
    let a = back(from);
    let mut run = Vec::new();
    let mut i = from;
    while free(i) && run.len() < n {
        run.push(i);
        i = step(i);
    }
    if run.len() < 2 {
        return 0.0;
    }
    let b = i;
    let (pa, pb) = (x[a], x[b]);
    let d = [pb[0] - pa[0], pb[1] - pa[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
    run.iter().map(|&k| ((x[k][0] - pa[0]) * d[1] - (x[k][1] - pa[1]) * d[0]).abs() / l).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::super::bike::tests::{gt_force_drivetrain, gt_force_linkage};
    use super::super::excite::{BumpTrain, Excitation, G};
    use super::*;

    fn still(seconds: f64) -> Excitation {
        Excitation { t_s: vec![0.0, seconds], travel_m: vec![0.04; 2], fx: vec![0.0; 2], fy: vec![G; 2], omega: vec![0.0; 2], speed_mps: vec![0.0; 2] }
    }

    #[test]
    fn a_still_bike_keeps_its_chain_on_and_its_links_the_pitch() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let r = simulate(&bike, 5, &|_| true, &still(0.2), &DynConfig::default(), &mut |_| true).unwrap();

        // Assert
        assert!(r.summary.max_link_error_mm < 0.5, "{:?}", r.summary);
        assert!(r.channels[12].values.last().copied().unwrap() > 8.0, "links on the ring");
        assert!(r.frame_t_s.len() >= 100);
        assert_eq!(r.node_xy.len(), r.frame_t_s.len() * r.nodes * 2);
    }

    #[test]
    fn the_still_lower_span_tension_balances_the_spring() {
        // Arrange: the chain round the lower jockey pulls it with about twice
        // the span tension; the spring holds the cage against that
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        // Act
        let r = simulate(&bike, 5, &|_| true, &still(0.3), &DynConfig::default(), &mut |_| true).unwrap();

        // Assert: a 0.45-0.9 N·m spring on a ~90 mm cage gives a few N to a few tens of N
        let t = *r.channels[5].values.last().unwrap();
        assert!(t > 1.0 && t < 60.0, "lower span {t} N");
    }

    #[test]
    fn a_bump_train_moves_the_cage_and_the_chain() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let e = BumpTrain { count: 2, ..BumpTrain::default() }.excitation(1000.0);

        // Act
        let r = simulate(&bike, 5, &|_| true, &e, &DynConfig::default(), &mut |_| true).unwrap();

        // Assert
        let cage = &r.channels[1].values;
        let swing = cage.iter().cloned().fold(f32::MIN, f32::max) - cage.iter().cloned().fold(f32::MAX, f32::min);
        assert!(swing > 0.5, "cage swing {swing} deg");
        assert!(r.summary.max_lower_deflection_mm > 0.5, "{:?}", r.summary);
        assert!(r.summary.max_link_error_mm < 1.0, "{:?}", r.summary);
    }
}

#[cfg(test)]
mod bench {
    use super::super::bike::tests::{gt_force_drivetrain, gt_force_linkage};
    use super::super::excite::Excitation;
    use super::super::kinematics::role_name;
    use super::super::path::chain_path;
    use super::*;

    /// The worst difference, over every frame after `from_s` and every wheel,
    /// between the links on the wheel in the run and on the exact tangent
    /// path at the same travel and cage angle (both counting the end links).
    pub(super) fn worst_wrap_error(bike: &SimBike, r: &DynResult, from_s: f64) -> (f64, String) {
        let ch = |n: &str| r.channels.iter().find(|c| c.name == n).unwrap().values.clone();
        let (trav, cage) = (ch("Rear travel"), ch("Cage angle"));
        let np = r.point_names.len();
        let mut worst = (0.0f64, String::new());
        for f in 0..r.frame_t_s.len() {
            if r.frame_t_s[f] < from_s {
                continue;
            }
            let pose = bike.pose(trav[f] as f64 / 1000.0);
            let kp = chain_path(bike, &pose, r.gear, bike.cage_abs(&pose, (cage[f] as f64).to_radians()), &|_| true).unwrap();
            for (role, radius, _) in &r.wheels {
                let pi = r.point_names.iter().position(|n| n == &format!("wheel:{role}")).unwrap();
                let c = [r.point_xy[(f * np + pi) * 2] as f64, r.point_xy[(f * np + pi) * 2 + 1] as f64];
                let on = (0..r.nodes)
                    .filter(|&i| {
                        let q = [r.node_xy[(f * r.nodes + i) * 2] as f64, r.node_xy[(f * r.nodes + i) * 2 + 1] as f64];
                        (((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).sqrt() - radius).abs() < 5e-4
                    })
                    .count() as f64;
                let want = kp.arcs.iter().find(|a| role_name(a.wheel.role) == *role).map(|a| a.len / CHAIN_PITCH_M + 1.0).unwrap_or(0.0);
                if (on - want).abs() > worst.0 {
                    worst = ((on - want).abs(), format!("{role} at {:.3} s: {on} links, the tangent path holds {want:.1}", r.frame_t_s[f]));
                }
            }
        }
        worst
    }

    #[test]
    fn a_taut_chain_driven_round_its_wheels_leaves_each_at_the_tangent() {
        // Arrange: no gravity or bumps; the cranks turn the chain round, the hub
        // overruns a little so the cassette freewheels with it
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let crank = 3.0;
        let ring_r = bike.drivetrain.chainring_teeth as f64 * CHAIN_PITCH_M / std::f64::consts::TAU;
        let cog_r = bike.cog_teeth(5).unwrap() as f64 * CHAIN_PITCH_M / std::f64::consts::TAU;
        let speed = 1.1 * crank * ring_r / cog_r * bike.drivetrain.wheel_radius_m;
        let e = Excitation { t_s: vec![0.0, 0.6], travel_m: vec![0.04; 2], fx: vec![0.0; 2], fy: vec![0.0; 2], omega: vec![0.0; 2], speed_mps: vec![speed; 2] };
        let cfg = DynConfig { crank_rate_rad_s: crank, ..DynConfig::default() };

        // Act
        let r = simulate(&bike, 5, &|_| true, &e, &cfg, &mut |_| true).unwrap();

        // Assert: the chain moved round, and every wheel holds the tangent path's links within one
        let cog = r.wheels.iter().position(|w| w.0 == "cog").unwrap();
        let turned = (r.wheel_angle[(r.frame_t_s.len() - 1) * r.wheels.len() + cog] - r.wheel_angle[cog]).abs();
        assert!(turned > 1.0, "the cassette turned only {turned} rad");
        let (worst, at) = worst_wrap_error(&bike, &r, 0.2);
        assert!(worst <= 1.0, "{at}");
    }
}
