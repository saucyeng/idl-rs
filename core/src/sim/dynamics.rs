//! The chain loop's dynamics (spec §2.4): one point mass per link joint,
//! rigid links, and every rotating part as a body with its inertia, solved
//! with small-step XPBD (positions projected onto the constraints each step,
//! velocities from the positions).
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
}

impl Default for DynConfig {
    fn default() -> Self {
        DynConfig { dt_s: 2.5e-5, iterations: 2, frame_rate_hz: 500.0, settle_s: 0.4, settle_damping_per_s: 40.0, chain_damping_per_s: 1.5, link_stiffness_npm: 1.0e7 }
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
}

#[derive(Clone, Copy)]
struct Contact {
    wheel: usize,
    a0: f64,
}

fn wrap_pi(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}
fn perp(v: P2) -> P2 {
    [-v[1], v[0]]
}

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

    // contacts at the start
    let mut contact: Vec<Option<Contact>> = vec![None; n];
    let enter_tol = 3.0e-4;
    let leave_tol = 1.0e-3;
    let assign = |x: &[P2], contact: &mut [Option<Contact>], wheels: &[DWheel]| {
        for i in 0..x.len() {
            if contact[i].is_some() {
                continue;
            }
            let mut best: Option<(usize, f64)> = None;
            for (k, w) in wheels.iter().enumerate() {
                let dx = [x[i][0] - w.centre[0], x[i][1] - w.centre[1]];
                let gap = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt() - w.radius;
                if gap < enter_tol && gap > -0.5 * w.radius && best.map_or(true, |b| gap < b.1) {
                    best = Some((k, gap));
                }
            }
            if let Some((k, _)) = best {
                let w = &wheels[k];
                let a = (x[i][1] - w.centre[1]).atan2(x[i][0] - w.centre[0]);
                contact[i] = Some(Contact { wheel: k, a0: a - w.psi });
            }
        }
    };
    assign(&x, &mut contact, &wheels);

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

        // predict the nodes
        for i in 0..n {
            let f = spec(x[i]);
            let cor = [2.0 * s.omega * v[i][1], -2.0 * s.omega * v[i][0]];
            v[i][0] = (v[i][0] + (-f[0] + cor[0]) * dt) * damp;
            v[i][1] = (v[i][1] + (-f[1] + cor[1]) * dt) * damp;
            v_pre[i] = v[i];
            xp[i] = x[i];
            x[i][0] += v[i][0] * dt;
            x[i][1] += v[i][1] * dt;
        }
        // predict the wheels
        let wheel_rate = if settling { 0.0 } else { s.speed_mps / d.wheel_radius_m.max(0.05) };
        for (k, w) in wheels.iter_mut().enumerate() {
            let mut torque_acc = -s.alpha;
            if k == COG && w.inv_inertia > 0.0 {
                // forward is CW (negative); drag pulls the cassette toward the wheel's speed
                let cog_fwd = -(w.omega - ss_rate);
                torque_acc += -d.freehub_drag_nm * ((wheel_rate - cog_fwd) / 1.0).tanh() * w.inv_inertia;
            }
            if w.inv_inertia > 0.0 {
                w.omega = (w.omega + torque_acc * dt) * if settling { damp } else { 1.0 };
            } else {
                w.omega = ss_rate * if matches!(w.mount, Mount::Ring) { 0.0 } else { 1.0 };
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
            let acc = (spring + weight) * jockey_inv_cage - s.alpha;
            cage_omega = (cage_omega + acc * dt) * if settling { damp } else { 1.0 };
            cage_prev = cage;
            cage += cage_omega * dt;
        }
        place_wheels(&mut wheels, &pose, cage);
        let surfs = surfaces_at(&pose);
        if surfs_prev.len() != surfs.len() {
            surfs_prev = surfaces_at(&pose);
        }
        assign(&x, &mut contact, &wheels);
        for l in lam.iter_mut() {
            *l = 0.0;
        }

        // solve
        for it in 0..cfg.iterations {
            // links, alternating direction
            for jj in 0..n {
                let i = if it % 2 == 0 { jj } else { n - 1 - jj };
                let j = (i + 1) % n;
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
            // wheels
            for i in 0..n {
                let Some(ct) = contact[i] else { continue };
                let k = ct.wheel;
                let is_jockey = matches!(wheels[k].mount, Mount::Upper | Mount::Lower);
                let c = wheels[k].centre;
                let dx = [x[i][0] - c[0], x[i][1] - c[1]];
                let dist = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt().max(1e-9);
                let nn = [dx[0] / dist, dx[1] / dist];
                let r = wheels[k].radius;
                let piv = piv_hist[2];
                let dc = perp([c[0] - piv[0], c[1] - piv[1]]);
                // radial: the chain sits on the pitch circle, never inside it
                let cr = dist - r;
                if cr < 0.0 {
                    let g = if is_jockey { -(nn[0] * dc[0] + nn[1] * dc[1]) } else { 0.0 };
                    let wc = if is_jockey { jockey_inv_cage } else { 0.0 };
                    let dl = -cr / (w_node + wc * g * g);
                    x[i][0] += w_node * nn[0] * dl;
                    x[i][1] += w_node * nn[1] * dl;
                    if is_jockey {
                        cage += wc * g * dl;
                    }
                }
                // tangential: a tooth holds the link
                if wheels[k].toothed {
                    let a = dx[1].atan2(dx[0]);
                    let ct_err = r * wrap_pi(a - wheels[k].psi - ct.a0);
                    let t_hat = perp(nn);
                    let gx = r / dist;
                    let gpsi = -r;
                    let gth = if is_jockey { -(r / dist) * (t_hat[0] * dc[0] + t_hat[1] * dc[1]) } else { 0.0 };
                    let wc = if is_jockey { jockey_inv_cage } else { 0.0 };
                    let wsum = w_node * gx * gx + wheels[k].inv_inertia * gpsi * gpsi + wc * gth * gth;
                    if wsum > 0.0 {
                        let dl = -ct_err / wsum;
                        x[i][0] += w_node * gx * t_hat[0] * dl;
                        x[i][1] += w_node * gx * t_hat[1] * dl;
                        wheels[k].psi += wheels[k].inv_inertia * gpsi * dl;
                        if is_jockey {
                            cage += wc * gth * dl;
                        }
                    }
                }
                if is_jockey {
                    let (uj, lj) = bike.jockeys(&pose, gear, cage);
                    for w in wheels.iter_mut() {
                        match w.mount {
                            Mount::Upper => w.centre = uj,
                            Mount::Lower => w.centre = lj,
                            _ => {}
                        }
                    }
                }
            }
            // surfaces: a link that was on the allowed side can't cross
            for i in 0..n {
                for (si, (_, sf)) in surfs.iter().enumerate() {
                    let w = [x[i][0] - sf.a[0], x[i][1] - sf.a[1]];
                    let along = w[0] * sf.u[0] + w[1] * sf.u[1];
                    if along < 0.0 || along > sf.len {
                        continue;
                    }
                    let sp = &surfs_prev[si].1;
                    let before = (xp[i][0] - sp.a[0]) * sp.n[0] + (xp[i][1] - sp.a[1]) * sp.n[1];
                    let dist = w[0] * sf.n[0] + w[1] * sf.n[1];
                    if dist < CHAIN_HALF_ROLLER_M && before > CHAIN_HALF_ROLLER_M - 2e-3 {
                        let c = dist - CHAIN_HALF_ROLLER_M;
                        x[i][0] -= c * sf.n[0];
                        x[i][1] -= c * sf.n[1];
                    }
                }
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
        // leave wheels the chain has pulled off
        for i in 0..n {
            if let Some(ct) = contact[i] {
                let c = wheels[ct.wheel].centre;
                let gap = ((x[i][0] - c[0]).powi(2) + (x[i][1] - c[1]).powi(2)).sqrt() - wheels[ct.wheel].radius;
                if gap > leave_tol {
                    contact[i] = None;
                }
            }
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
