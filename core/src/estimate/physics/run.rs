//! Running the physics estimator over a session: learn the bike's geometry
//! from a few windows (once per calibration), then solve every stretch of
//! riding in overlapping windows with it frozen, and write the result as a
//! derived session file of named channels.

use super::bike::{Bike, B_FA, B_LAT, D, F_ROT, F_TRANS, F_TWIST, NL, PSI, R_ROLL, R_YAW, S};
use super::forces::{ForceModel, ForceWindow, ACT_F, ACT_M, DEF_F, DEF_R, LAT_F, LAT_R, NFP, TAN_F, TAN_R};
use super::ground::{ground_window, NavPoint};
use super::la::{M3, V3};
use super::nav::{GpsFix, NavSpec, NavStart, NavWindow};
use super::objective::{Ctx, Objective, Spec, Vars, Weights, NG};
use super::pipeline::{airtime_score, bike_axes, first_start, gps_latency, gps_on_grid, moving, pick_windows, rotation_score, window, Ride};
use super::so3spline::rotation;
use super::setup::{geometry, with_geometry, Geometry};
use super::solve::{gauss_newton, gauss_newton_multi, latents, warm_start, SolveError, SolveInfo};

/// Version of the estimator's output. Bump when the same inputs give
/// different channels (a model or solver change); stored in the output file
/// so an older file can be shown as stale.
pub const PHYSICS_ESTIMATOR_VERSION: u32 = 4; // 2: IMUs low-passed at 0.4 x the grid rate; 3: Huber (k 3) on the sensor rows; 4: fork torsional vibration

/// How a run is set up.
#[derive(Clone, Debug)]
pub struct Config {
    /// Solve window, s.
    pub window_s: f64,
    /// Overlap between consecutive windows, s (half of it is dropped each side).
    pub overlap_s: f64,
    /// Bike-fit window, s.
    pub fit_window_s: f64,
    /// Bike-fit windows chosen for airtime (anchor the slow travel).
    pub fit_air_windows: usize,
    /// Bike-fit windows chosen for frame rotation (the lever-arm terms).
    pub fit_busy_windows: usize,
    /// Gauss-Newton iteration cap, per solve.
    pub iters: usize,
    /// Stop when an iteration gains less loss than this.
    pub tol: f64,
    /// Anchor travel near 0 in free fall.
    pub air_anchor: bool,
    /// Estimate attitude and velocity too (spec one-model §2-5): each riding
    /// stretch's windows then solve in order, each starting from the last.
    pub nav: bool,
    /// The force model (spec force-model F1; None: off): its parameters are identified on
    /// [`Config::force_id_windows`] windows together, then every window solves with them frozen.
    pub forces: Option<ForceModel>,
    /// Windows the force parameters are identified on (spread through the ride).
    pub force_id_windows: usize,
    /// Length of each identification window, s.
    pub force_id_window_s: f64,
    /// Gauss-Newton iteration cap for the identification (cold latents and shared parameters
    /// take longer than a window solve).
    pub force_id_iters: usize,
    /// Solve only inside these spans, channel time in us (None: all riding). A prototyping
    /// aid: the force model tried on the descents alone (`IDL_PHYSICS_LAPS`).
    pub only_us: Option<Vec<(i64, i64)>>,
    /// The helmet camera on the ride's grid (F2): accelerometer m/s^2 and gyro rad/s per sample,
    /// turned into the frame pod's axes by each clip's mount estimate (None: no camera).
    pub camera: Option<Vec<Option<[f64; 6]>>>,
    /// Worker threads for the window solves (1: none).
    pub threads: usize,
    /// Noise levels and priors.
    pub weights: Weights,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            window_s: 12.0,
            overlap_s: 2.0,
            fit_window_s: 15.0, // 30 s windows held ~170 MB of normal matrix each
            fit_air_windows: 2,
            fit_busy_windows: 2,
            iters: 20,
            tol: 1.0,
            air_anchor: true,
            nav: false,
            forces: None,
            force_id_windows: 6,
            force_id_window_s: 6.0,
            force_id_iters: 60,
            only_us: None,
            camera: None,
            threads: default_threads(),
            weights: Weights::default(),
        }
    }
}

/// Worker threads for the window solves when none are asked for: every core but two, which are
/// left for the rest of the machine (at least one; Isaac, 2026-11-25). Each thread holds one
/// window's normal matrix and its factor: about 140 MB for a 12 s window, about 210 MB with
/// navigation. Phones keep the old cap of 4, set when the dev machine had 16 GB.
pub fn default_threads() -> usize {
    let n = std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    if cfg!(any(target_os = "android", target_os = "ios")) { n.min(4) } else { n }
}

/// What a bike fit found.
#[derive(Clone, Debug)]
pub struct FitReport {
    /// The fitted geometry.
    pub geometry: Geometry,
    /// Fit windows, grid indices.
    pub windows: Vec<(usize, usize)>,
    /// The solve.
    pub info: SolveInfo,
}

/// Learns the bike's geometry (steer point, bending length, linkage scale)
/// from the ride's most informative windows: some with the most airtime,
/// the rest with the most frame rotation, all solved together with shared
/// geometry and biases. `exclude`: grid ranges not to fit on (e.g. held out).
pub fn fit_bike(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config, exclude: &[(usize, usize)]) -> Result<FitReport, SolveError> {
    let n = (cfg.fit_window_s * r.rate()) as usize;
    let mut windows = if cfg.fit_air_windows > 0 { pick_windows(&airtime_score(r), n, cfg.fit_air_windows, exclude) } else { Vec::new() };
    let taken: Vec<(usize, usize)> = exclude.iter().chain(&windows).cloned().collect();
    windows.extend(pick_windows(&rotation_score(r), n, cfg.fit_busy_windows + cfg.fit_air_windows - windows.len(), &taken));
    windows.sort();
    let inner = (cfg.threads / windows.len().max(1)).max(1);
    let objs: Vec<Objective> = windows
        .iter()
        .map(|&(a, b)| Objective { threads: inner, ..Objective::fitting(bike.clone(), window(r, a, b, cfg.air_anchor), Spec::standard(b - a, r.input.dt), cfg.weights.clone(), bias_centre) })
        .collect();
    let vs = objs.iter().map(Objective::initial_vars).collect();
    let (vs, info) = gauss_newton_multi(&objs, vs, cfg.iters, cfg.tol)?;
    let geometry: Geometry = std::array::from_fn(|i| vs[0].glob[NG + i]);
    Ok(FitReport { geometry, windows, info })
}

/// The estimate over a ride: latents per grid sample (NaN where not solved:
/// not riding, or a window that failed), the airtime anchor mask, and how
/// the solves went.
pub struct Solution {
    /// Latents per grid sample, SI units (see `bike`'s index constants).
    pub q: Vec<[f64; NL]>,
    /// Samples anchored as airborne.
    pub air: Vec<bool>,
    /// Windows solved, and how many failed.
    pub windows: (usize, usize),
    /// Gauss-Newton iterations, summed.
    pub iterations: usize,
    /// Attitude and velocity per grid sample (empty unless [`Config::nav`]).
    pub nav: Vec<NavState>,
    /// The force outputs per grid sample, in [`FORCE_CHANNELS`]' units (empty unless
    /// [`Config::forces`]; NaN where not solved).
    pub forces: Vec<[f64; NFO]>,
    /// The identified force parameters, log multipliers on the model's nominal values.
    pub force_par: Option<[f64; NFP]>,
}

/// The bike's attitude and velocity at one grid sample (NaN where not solved).
#[derive(Clone, Copy, Debug)]
pub struct NavState {
    /// Bike axes (X forward, Y left, Z up) -> nav (X north, Y west, Z up).
    pub r: [[f64; 3]; 3],
    /// Velocity, nav, m/s.
    pub v: [f64; 3],
    /// Acceleration, nav, m/s^2.
    pub vd: [f64; 3],
}

impl NavState {
    /// Not solved.
    pub const NAN: NavState = NavState { r: [[f64::NAN; 3]; 3], v: [f64::NAN; 3], vd: [f64::NAN; 3] };
}

/// How sure a window is of the state its predecessor hands it: tilt, tilt, heading (rad); velocity (m/s).
const HANDOFF_SIGMA: ([f64; 3], f64) = ([0.02, 0.02, 0.05], 0.3);

/// The solve windows over the riding stretches: (start, end, keep_start, keep_end).
fn plan(r: &Ride, cfg: &Config) -> Vec<(usize, usize, usize, usize)> {
    let rate = r.rate();
    let mut mov = moving(r);
    if let Some(spans) = &cfg.only_us {
        for (m, t) in mov.iter_mut().zip(&r.grid_us) {
            *m &= spans.iter().any(|&(a, b)| (a..b).contains(t));
        }
    }
    plan_windows(&mov, (cfg.window_s * rate) as usize, (cfg.overlap_s * rate) as usize, (2.0 * rate) as usize)
}

/// Solve windows over the `true` stretches of `mov` (stretches shorter than
/// `min` samples skipped): windows of `w` samples overlapping by `o`, as
/// (start, end, keep_start, keep_end). The kept parts tile every stretch
/// exactly, each window keeping its middle (`o / 2` from an inner edge).
pub fn plan_windows(mov: &[bool], w: usize, o: usize, min: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < mov.len() {
        if !mov[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < mov.len() && mov[i] {
            i += 1;
        }
        let e = i;
        if e - s < min {
            continue;
        }
        // consecutive windows overlap by `o`; each keeps from where the last
        // one's kept part ended to `o / 2` before its own end (the stretch's
        // last window keeps to the end, shifted back to full length if it can)
        let mut start = s;
        let mut keep_from = s;
        loop {
            let a = if e - start < w { e.saturating_sub(w).max(s) } else { start };
            let b = (a + w).min(e);
            let keep_to = if b == e { e } else { b - o / 2 };
            if keep_to > keep_from {
                out.push((a, b, keep_from, keep_to));
                keep_from = keep_to;
            }
            if b == e || b - o <= start {
                break;
            }
            start = b - o;
        }
    }
    out
}

/// Watching a [`solve_ride_with`] run from another thread: told after each
/// window, and able to stop the rest.
pub struct SolveProgress<'a> {
    /// Called after each window with (windows finished, windows planned),
    /// from the solving threads.
    pub on_window: &'a (dyn Fn(usize, usize) + Sync),
    /// Set to stop: windows not yet started end as [`SolveError::Cancelled`].
    pub cancel: &'a std::sync::atomic::AtomicBool,
}

/// Solves every riding stretch of `r` with `bike` frozen, windows in parallel.
pub fn solve_ride(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config) -> Solution {
    solve_ride_with(r, bike, bias_centre, cfg, None)
}

/// [`solve_ride`], reporting each finished window to `progress` and
/// skipping the windows not yet started once its `cancel` is set.
pub fn solve_ride_with(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config, progress: Option<&SolveProgress>) -> Solution {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let plan = plan(r, cfg);
    let finished = AtomicUsize::new(0);
    // the ground tie (F4) needs navigation in every force window, with or without its outputs
    let ground = cfg.forces.as_ref().is_some_and(|m| m.w.ground.is_some());
    let gps_all: Option<Vec<Option<GpsFix>>> = (cfg.nav || ground).then(|| gps_on_grid(r, gps_latency(r, bike)));
    let gps = if cfg.nav { gps_all.clone() } else { None };
    let axes = bike_axes(bike);
    // the force model: its parameters identified first, then frozen for every window
    let speed: Option<Vec<f64>> = cfg.forces.as_ref().map(|_| speed_on_grid(r, gps_latency(r, bike)));
    let force_par = match (&cfg.forces, &speed) {
        (Some(m), Some(sp)) => identify_forces(r, bike, bias_centre, cfg, m, sp, gps_all.as_deref(), &plan)
            .inspect_err(|e| {
                if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
                    eprintln!("force identification failed: {e:?}");
                }
            })
            .ok(),
        _ => None,
    };
    let frozen = cfg.forces.as_ref().map(|m| {
        let mut m = m.clone();
        m.w.par_sigma = [FROZEN_SIGMA; NFP];
        m
    });
    // chains of windows solved in order: with navigation on, one riding stretch
    // (overlapping windows), each window starting from the last; without, every window alone
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for i in 0..plan.len() {
        match chains.last_mut() {
            Some(c) if gps.is_some() && plan[i].0 < plan[*c.last().unwrap()].1 => c.push(i),
            _ => chains.push(vec![i]),
        }
    }
    // threads inside each window: what the chains leave over (a short span has few windows)
    let inner = (cfg.threads.max(1) / chains.len().max(1)).max(1);
    type WinOut = (Vec<[f64; NL]>, Option<Vec<bool>>, usize, Vec<NavState>, Vec<[f64; NFO]>);
    let solve_chain = |chain: &Vec<usize>| -> Vec<(usize, Result<WinOut, SolveError>)> {
        let mut out = Vec::with_capacity(chain.len());
        let mut handoff: Option<NavStart> = None;
        for (ci, &i) in chain.iter().enumerate() {
            let (a, b, _, _) = plan[i];
            if progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
                out.push((i, Err(SolveError::Cancelled)));
                continue;
            }
            let mut win = window(r, a, b, cfg.air_anchor);
            let air = win.air.clone();
            let mut spec = Spec::standard(b - a, r.input.dt);
            if let Some(g) = &gps {
                let start = handoff.take().unwrap_or_else(|| first_start(r, bike, g, a));
                win.nav = Some(NavWindow { gps: g[a..b].to_vec(), start });
                spec.nav = Some(NavSpec::default());
            } else if let (Some(g), true) = (&gps_all, ground) {
                // navigation for the ground tie alone: each window starts from gravity and the GPS course
                win.nav = Some(NavWindow { gps: g[a..b].to_vec(), start: first_start(r, bike, g, a) });
                spec.nav = Some(NavSpec::default());
            }
            if let Some(sp) = &speed {
                win.forces = Some(ForceWindow { speed_mps: sp[a..b].to_vec(), cam: cfg.camera.as_ref().map_or(vec![], |c| c[a..b].to_vec()), ground: None });
            }
            let mut obj = Objective::new(bike.clone(), win, spec, cfg.weights.clone(), bias_centre);
            obj.threads = inner;
            if let Some(m) = &frozen {
                obj = obj.with_forces(m.clone());
                let g0 = obj.force_glob0();
                obj.glob_centre[g0..g0 + NFP].copy_from_slice(&force_par.unwrap_or([0.0; NFP]));
            }
            let solved = ground_start(&obj, cfg.nav, cfg.iters, cfg.tol).and_then(|(o, v0)| {
                obj = o;
                gauss_newton(&obj, v0, cfg.iters, cfg.tol)
            });
            if let Some(p) = progress {
                (p.on_window)(finished.fetch_add(1, Ordering::Relaxed) + 1, plan.len());
            }
            let res = solved.map(|(v, info)| {
                let mut states = Vec::new();
                if obj.nav().is_some() {
                    let next = chain.get(ci + 1).map(|&n| plan[n].0 - a);
                    for t in 0..b - a {
                        let s = obj.sample(&v, t);
                        let ns = obj.nav_sample(&v, &s, t).expect("navigation on");
                        let rot: M3<f64> = rotation(&ns.ctrl, ns.u);
                        states.push(NavState { r: (rot * axes).0, v: ns.v, vd: ns.vd });
                        if next == Some(t) {
                            handoff = Some(NavStart { att: rot.0, att_sigma: HANDOFF_SIGMA.0, vel: ns.v, vel_sigma: HANDOFF_SIGMA.1 });
                        }
                    }
                }
                let fo = if obj.force_on().is_some() {
                    let ctx = obj.ctx(&v.glob, false);
                    (0..b - a).map(|t| force_outputs(&obj, &ctx, &v, t)).collect()
                } else {
                    Vec::new()
                };
                (latents(&obj, &v), air, info.iterations, states, fo)
            });
            out.push((i, res));
        }
        out
    };
    let threads = cfg.threads.max(1);
    let mut results: Vec<Option<Result<WinOut, SolveError>>> = (0..plan.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        let parts: Vec<Vec<usize>> = (0..threads).map(|t| (t..chains.len()).step_by(threads).collect()).collect();
        let handles: Vec<_> = parts
            .into_iter()
            .map(|idx| {
                let chains = &chains;
                let solve_chain = &solve_chain;
                sc.spawn(move || idx.into_iter().flat_map(|c| solve_chain(&chains[c])).collect::<Vec<_>>())
            })
            .collect();
        for h in handles {
            for (i, res) in h.join().expect("solve thread panicked") {
                results[i] = Some(res);
            }
        }
    });
    let mut sol = Solution {
        q: vec![[f64::NAN; NL]; r.len()],
        air: vec![false; r.len()],
        windows: (plan.len(), 0),
        iterations: 0,
        nav: if gps.is_some() { vec![NavState::NAN; r.len()] } else { Vec::new() },
        forces: if cfg.forces.is_some() { vec![[f64::NAN; NFO]; r.len()] } else { Vec::new() },
        force_par,
    };
    for (&(a, _, ks, ke), res) in plan.iter().zip(results) {
        match res.expect("every window solved") {
            Ok((q, air, it, states, fo)) => {
                sol.iterations += it;
                for t in ks..ke {
                    sol.q[t] = q[t - a];
                    sol.air[t] = air.as_ref().is_some_and(|m| m[t - a]);
                    if !states.is_empty() {
                        sol.nav[t] = states[t - a];
                    }
                    if !fo.is_empty() {
                        sol.forces[t] = fo[t - a];
                    }
                }
            }
            Err(e) => {
                if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
                    eprintln!("window at {a} failed: {e:?}");
                }
                sol.windows.1 += 1
            }
        }
    }
    sol
}


/// The force parameters' prior width once identified, log units: frozen for the window solves.
pub const FROZEN_SIGMA: f64 = 1e-3;

/// The ground speed on the grid, m/s: the GPS fixes' horizontal speed (each at its time less
/// `latency_s`), linear between them, held past the ends; 0 without GPS (fixes without a
/// speed are skipped: one NaN in the wheels' spin poisons every moment row of the solve).
pub fn speed_on_grid(r: &Ride, latency_s: f64) -> Vec<f64> {
    let pts: Vec<(f64, f64)> = r.input.gps.iter().map(|g| (g.time_s - latency_s, g.velocity.x.hypot(g.velocity.y))).filter(|p| p.0.is_finite() && p.1.is_finite()).collect();
    r.grid_us
        .iter()
        .map(|&t| {
            let ts = t as f64 / 1e6;
            let i = pts.partition_point(|p| p.0 < ts);
            match (i.checked_sub(1).and_then(|j| pts.get(j)), pts.get(i)) {
                (Some(a), Some(b)) if b.0 > a.0 => a.1 + (b.1 - a.1) * (ts - a.0) / (b.0 - a.0),
                (Some(a), _) => a.1,
                (None, Some(b)) => b.1,
                (None, None) => 0.0,
            }
        })
        .collect()
}

/// A force objective's start (F4): its warm start, which solves navigation too when the window
/// has it; from that navigation the ground tie's inputs (the frame pod's height, its distance
/// along the path, the world's up); then, unless `keep_nav`, the objective and its start
/// without navigation (the force solve stays attitude-free). Without the tie or navigation, the
/// objective as it is and its warm start.
pub fn ground_start(obj: &Objective, keep_nav: bool, iters: usize, tol: f64) -> Result<(Objective, Vars), SolveError> {
    let mut v = warm_start(obj, iters, tol)?;
    let mut out = Objective { bike: obj.bike.clone(), win: obj.win.clone(), spec: obj.spec.clone(), w: obj.w.clone(), glob_centre: obj.glob_centre.clone(), fit: obj.fit, forces: obj.forces.clone(), threads: obj.threads };
    let (Some(m), Some(_)) = (&obj.forces, obj.nav()) else { return Ok((out, v)) };
    let Some(gw) = &m.w.ground else { return Ok((out, v)) };
    let nav: Vec<NavPoint> = (0..obj.spec.n)
        .map(|t| {
            let s = obj.sample(&v, t);
            let ns = obj.nav_sample(&v, &s, t).expect("navigation on");
            NavPoint { r: rotation(&ns.ctrl, ns.u), v: ns.v, w0: s.w0 }
        })
        .collect();
    // IDL_GROUND_ACCEL_TAU (prototyping, s): the frame's velocity for the tie from its own
    // accelerometer (turned to the world by the attitude, gravity removed), pulled to
    // navigation's over this time constant: fast motion measured, slow motion from navigation
    let nav = match std::env::var("IDL_GROUND_ACCEL_TAU").ok().and_then(|x| x.parse::<f64>().ok()).filter(|x| *x > 0.0) {
        Some(tau) => {
            let dt = obj.spec.dt;
            let mut vh = V3(nav[0].v);
            nav.iter()
                .enumerate()
                .map(|(t, p)| {
                    if t > 0 {
                        let f = V3(obj.win.frame_accel[t - 1]);
                        let a = nav[t - 1].r * f + V3(super::nav::G_NAV);
                        vh = vh + (a + (V3(nav[t - 1].v) - vh).scale(1.0 / tau)).scale(dt);
                    }
                    NavPoint { v: vh.0, ..*p }
                })
                .collect()
        }
        None => nav,
    };
    let fb = &m.bike;
    let offset = [fb.front_axle.dot(fb.fwd), fb.rear_axle.dot(fb.fwd)];
    if let Some(fw) = out.win.forces.as_mut() {
        fw.ground = Some(ground_window(&nav, obj.spec.dt, fb.fwd, offset, gw));
    }
    if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
        // how well navigation's heights and the pods-only travel put both patches on one ground
        let mut probe = v.clone();
        probe.att.clear();
        probe.vel = [Vec::new(), Vec::new(), Vec::new()];
        let mut o2 = Objective { bike: out.bike.clone(), win: out.win.clone(), spec: out.spec.clone(), w: out.w.clone(), glob_centre: out.glob_centre.clone(), fit: out.fit, forces: out.forces.clone(), threads: out.threads };
        o2.win.nav = None;
        o2.spec.nav = None;
        o2.refit_ground(&mut probe);
        let ctx = o2.ctx(&probe.glob, false);
        let (mut n, mut ss, mut big, mut sig) = ([0usize; 2], [0.0; 2], [0usize; 2], [0.0; 2]);
        for t in 0..o2.spec.n {
            let s = o2.sample(&probe, t);
            if let Some(fj) = o2.force_residuals(&ctx, &probe, &s, t, false) {
                let gw = o2.win.forces.as_ref().unwrap().ground.as_ref().unwrap();
                for i in 0..2 {
                    let r = fj.r[super::forces::TIE_ROWS.0 + i];
                    if gw.sigma_m[t][i].is_finite() && r != 0.0 {
                        n[i] += 1;
                        ss[i] += r * r;
                        sig[i] += gw.sigma_m[t][i];
                        big[i] += (r.abs() > 3.0) as usize;
                    }
                }
            }
        }
        // the sigma's terms, mean over moving samples, mm: the turn's offset, the slip's, navigation's drift
        let (lb, mut terms, mut cnt, mut beta_deg) = (offset[0] - offset[1], [0.0; 3], 0usize, 0.0);
        let slip_of = |p: &NavPoint| { let f = p.r * fb.fwd; (f.0[0] * p.v[1] - f.0[1] * p.v[0]).atan2(f.0[0] * p.v[0] + f.0[1] * p.v[1]) };
        let mut sl: Vec<f64> = nav.iter().filter(|p| V3(p.v).norm() >= gw.min_speed_mps).map(slip_of).collect();
        sl.sort_by(f64::total_cmp);
        let slip_bias = sl.get(sl.len() / 2).copied().unwrap_or(0.0);
        for p in &nav {
            let sp = V3(p.v).norm();
            if sp < gw.min_speed_mps {
                continue;
            }
            let up = V3([p.r.0[2][0], p.r.0[2][1], p.r.0[2][2]]);
            let beta = (slip_of(p) - slip_bias).abs().min(std::f64::consts::FRAC_PI_2);
            terms[0] += gw.cross_slope * lb * lb * p.w0.dot(up).abs() / (2.0 * sp);
            terms[1] += gw.cross_slope * lb * beta;
            terms[2] += gw.nav_sigma_mps * lb / sp;
            beta_deg += beta.to_degrees();
            cnt += 1;
        }
        // the rear against the front's ground, the rear shifted along the path: does a distance
        // error or a height error make the mismatch?
        {
            let gwin = o2.win.forces.as_ref().unwrap().ground.as_ref().unwrap().clone();
            let mut pz = Vec::with_capacity(o2.spec.n);
            for t in 0..o2.spec.n {
                if let Some((fo, _)) = o2.forces_at(&ctx, &probe, t) {
                    pz.push((t, fo.patch_z));
                }
            }
            let ok = |t: usize| gwin.sigma_m[t][0].is_finite();
            let front: Vec<super::ground::Patch> = pz.iter().filter(|(t, _)| ok(*t)).map(|(t, z)| super::ground::Patch { s_m: gwin.s_m[*t] + gwin.offset_m[0], z_m: z[0], sigma_m: 0.005 }).collect();
            if let Some((s0_m, coef)) = super::ground::fit_profile(&front, gw.knot_m, gw.curvature_m, 3.0) {
                let hf = super::ground::GroundFit { s0_m, knot_m: gw.knot_m, coef, on: Vec::new() };
                let (lo, hi) = (front.first().map_or(0.0, |p| p.s_m), front.last().map_or(0.0, |p| p.s_m));
                let mis = |d: f64, med: bool| {
                    let mut e: Vec<f64> = pz.iter().filter(|(t, _)| ok(*t)).map(|(t, z)| (gwin.s_m[*t] + gwin.offset_m[1] + d, z[1])).filter(|(s, _)| *s > lo && *s < hi).map(|(s, z)| z - hf.height(s)).collect();
                    if e.is_empty() {
                        return f64::NAN;
                    }
                    let m = e.iter().sum::<f64>() / e.len() as f64;
                    if med {
                        e.iter_mut().for_each(|x| *x = (*x - m).abs());
                        e.sort_by(f64::total_cmp);
                        return e[e.len() / 2];
                    }
                    (e.iter().map(|x| (x - m).powi(2)).sum::<f64>() / e.len() as f64).sqrt()
                };
                // the gap by how far apart the turn puts the lines (L^2 |yaw rate| / 2v), and on
                // the straight by the bike's own pitch rate: line difference or something else
                {
                    let lb = offset[0] - offset[1];
                    let mut bins: [Vec<f64>; 4] = Default::default();
                    let mut e_all: Vec<(f64, usize)> = Vec::new();
                    for (t, z) in pz.iter().filter(|(t, _)| ok(*t)) {
                        let s = gwin.s_m[*t] + gwin.offset_m[1];
                        if s <= lo || s >= hi {
                            continue;
                        }
                        let p = &nav[*t];
                        let sp = V3(p.v).norm().max(0.1);
                        let up = V3([p.r.0[2][0], p.r.0[2][1], p.r.0[2][2]]);
                        let lat = lb * lb * p.w0.dot(up).abs() / (2.0 * sp);
                        let b = if lat < 0.02 { 0 } else if lat < 0.05 { 1 } else if lat < 0.10 { 2 } else { 3 };
                        e_all.push((z[1] - hf.height(s), b));
                    }
                    let m = e_all.iter().map(|x| x.0).sum::<f64>() / e_all.len().max(1) as f64;
                    for (e, b) in e_all {
                        bins[b].push((e - m).abs());
                    }
                    let med = |v: &mut Vec<f64>| {
                        v.sort_by(f64::total_cmp);
                        if v.is_empty() { f64::NAN } else { 1e3 * v[v.len() / 2] }
                    };
                    let n: Vec<usize> = bins.iter().map(|b| b.len()).collect();
                    let ms: Vec<f64> = bins.iter_mut().map(med).collect();
                    eprintln!("ground tie gap by the turn's line offset, median mm (samples): <2 cm {:.0} ({}), 2-5 cm {:.0} ({}), 5-10 cm {:.0} ({}), >10 cm {:.0} ({})", ms[0], n[0], ms[1], n[1], ms[2], n[2], ms[3], n[3]);
                }
                // which term the gap follows: the front's travel and pitch and height where the rear
                // is now (fitted against distance), the rear's own; least squares, each column's
                // share of the gap's variance and its coefficient
                {
                    let prof = |vals: &dyn Fn(usize) -> f64| {
                        let pts: Vec<super::ground::Patch> = pz.iter().filter(|(t, _)| ok(*t)).map(|(t, _)| super::ground::Patch { s_m: gwin.s_m[*t] + gwin.offset_m[0], z_m: vals(*t), sigma_m: 0.005 }).collect();
                        super::ground::fit_profile(&pts, gw.knot_m, gw.curvature_m, 0.0).map(|(s0_m, coef)| super::ground::GroundFit { s0_m, knot_m: gw.knot_m, coef, on: Vec::new() })
                    };
                    let q: Vec<[f64; NL]> = (0..o2.spec.n).map(|t| o2.sample(&probe, t).q).collect();
                    let pitch = |t: usize| V3(gwin.up[t]).dot(fb.fwd).asin();
                    let (fd, fp, fz) = (prof(&|t| q[t][D]), prof(&|t| pitch(t)), prof(&|t| gwin.z_m[t]));
                    if let (Some(fd), Some(fp), Some(fz)) = (fd, fp, fz) {
                        let mut rows: Vec<([f64; 5], f64)> = Vec::new();
                        for (t, z) in pz.iter().filter(|(t, _)| ok(*t)) {
                            let s = gwin.s_m[*t] + gwin.offset_m[1];
                            if s <= lo || s >= hi {
                                continue;
                            }
                            rows.push(([fd.height(s), q[*t][S], pitch(*t) - fp.height(s), gwin.z_m[*t] - fz.height(s), 1.0], z[1] - hf.height(s)));
                        }
                        let k = 5;
                        let mut a = nalgebra::DMatrix::<f64>::zeros(k, k);
                        let mut b = nalgebra::DVector::<f64>::zeros(k);
                        for (x, e) in &rows {
                            for i in 0..k {
                                b[i] += x[i] * e;
                                for j in 0..k {
                                    a[(i, j)] += x[i] * x[j];
                                }
                            }
                        }
                        let mean = |c: usize| rows.iter().map(|r| r.0[c]).sum::<f64>() / rows.len().max(1) as f64;
                        let em = rows.iter().map(|r| r.1).sum::<f64>() / rows.len().max(1) as f64;
                        let var = rows.iter().map(|r| (r.1 - em).powi(2)).sum::<f64>() / rows.len().max(1) as f64;
                        if let Some(c) = a.clone().lu().solve(&b) {
                            let names = ["front travel", "rear travel", "pitch change", "height change"];
                            let share: Vec<String> = (0..4).map(|i| {
                                let m = mean(i);
                                let cv = rows.iter().map(|r| (r.0[i] - m) * (r.1 - em)).sum::<f64>() / rows.len().max(1) as f64;
                                let vx = rows.iter().map(|r| (r.0[i] - m).powi(2)).sum::<f64>() / rows.len().max(1) as f64;
                                format!("{} r2 {:.2} coef {:+.3}", names[i], cv * cv / (vx * var).max(f64::MIN_POSITIVE), c[i])
                            }).collect();
                            let res = rows.iter().map(|(x, e)| (e - (0..k).map(|i| c[i] * x[i]).sum::<f64>()).powi(2)).sum::<f64>() / rows.len().max(1) as f64;
                            eprintln!("ground tie gap {:.1} mm rms, {:.1} after the fit: {}", 1e3 * var.sqrt(), 1e3 * res.sqrt(), share.join(", "));
                        }
                    }
                }
                let best = (-40..=40).map(|i| i as f64 * 0.02).map(|d| (d, mis(d, false))).filter(|x| x.1.is_finite()).min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((d, m)) = best {
                    eprintln!("ground tie, rear on the front's ground: rms {:.1} mm (median |e| {:.1}) as placed; best {:.1} mm (median {:.1}) with the rear {:+.2} m along the path", 1e3 * mis(0.0, false), 1e3 * mis(0.0, true), 1e3 * m, 1e3 * mis(d, true), d);
                }
            }
        }
        let c = cnt.max(1) as f64;
        eprintln!("ground tie sigma terms (mean mm): turn {:.1}, slip {:.1} (slip {:.1} deg, less the heading offset {:.1} deg), drift {:.1}", 1e3 * terms[0] / c, 1e3 * terms[1] / c, beta_deg / c, slip_bias.to_degrees(), 1e3 * terms[2] / c);
        let gw = o2.win.forces.as_ref().unwrap().ground.as_ref().unwrap();
        eprintln!("ground tie at the warm start: front rms {:.2} sigma ({} rows, {:.0} % past 3), rear rms {:.2} sigma ({} rows, {:.0} % past 3), mean sigma {:.1} mm; height span {:.2} m, distance {:.0} m",
                  (ss[0] / n[0].max(1) as f64).sqrt(), n[0], 100.0 * big[0] as f64 / n[0].max(1) as f64, (ss[1] / n[1].max(1) as f64).sqrt(), n[1], 100.0 * big[1] as f64 / n[1].max(1) as f64,
                  1e3 * (sig[0] + sig[1]) / (n[0] + n[1]).max(1) as f64, gw.z_m.iter().cloned().fold(f64::MIN, f64::max) - gw.z_m.iter().cloned().fold(f64::MAX, f64::min), gw.s_m.last().copied().unwrap_or(0.0));
    }
    if !keep_nav {
        out.win.nav = None;
        out.spec.nav = None;
        v.att.clear();
        v.vel = [Vec::new(), Vec::new(), Vec::new()];
    }
    Ok((out, v))
}

/// Identifies the force parameters on [`Config::force_id_windows`] windows spread through
/// the ride ([`Config::force_id_window_s`] each, from the start of every plan window picked),
/// solved together with shared parameters (and biases). Returns the log multipliers.
pub fn identify_forces(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config, m: &ForceModel, speed: &[f64], gps: Option<&[Option<GpsFix>]>, plan: &[(usize, usize, usize, usize)]) -> Result<[f64; NFP], SolveError> {
    let len = (cfg.force_id_window_s * r.rate()) as usize;
    let k = cfg.force_id_windows.min(plan.len());
    if k == 0 {
        return Ok([0.0; NFP]);
    }
    let t0 = std::time::Instant::now();
    let objs: Vec<Objective> = (0..k)
        .map(|i| {
            let (a, b0, _, _) = plan[i * plan.len() / k];
            let b = (a + len).min(b0);
            let mut win = window(r, a, b, cfg.air_anchor);
            win.forces = Some(ForceWindow { speed_mps: speed[a..b].to_vec(), cam: cfg.camera.as_ref().map_or(vec![], |c| c[a..b].to_vec()), ground: None });
            let mut spec = Spec::standard(b - a, r.input.dt);
            if let (Some(g), Some(_)) = (gps, &m.w.ground) {
                win.nav = Some(NavWindow { gps: g[a..b].to_vec(), start: first_start(r, bike, g, a) });
                spec.nav = Some(NavSpec::default());
            }
            let o = Objective::new(bike.clone(), win, spec, cfg.weights.clone(), bias_centre).with_forces(m.clone());
            Objective { threads: (cfg.threads / k).max(1), ..o }
        })
        .collect();
    if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
        let lay = super::solve::Layout::new(&objs[0]);
        eprintln!("force identification: window 0 has {} samples, {} local unknowns, band half-width {}, {} globals", objs[0].spec.n, lay.n_loc, lay.bw, objs[0].n_glob());
    }
    let (objs, vs): (Vec<Objective>, Vec<Vars>) = super::solve::par_map_n(&objs, k, |o| ground_start(o, false, cfg.iters, cfg.tol)).into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
    let t_warm = t0.elapsed().as_secs_f64();
    if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
        // the pods' own fit at the warm starts, by pod: frame gyro, fork pod, rear pod
        let mut by = [0.0; 3];
        for (o, v) in objs.iter().zip(&vs) {
            let ctx = o.ctx(&v.glob, false);
            for t in 0..o.spec.n {
                let s = o.sample(v, t);
                let r = o.sample_residuals(&ctx, &s, &v.glob, t, false).r;
                for (i, (a, b)) in [(0, 3), (3, 9), (9, 15)].into_iter().enumerate() {
                    by[i] += 0.5 * r[a..b].iter().map(|&x| super::objective::huber2(x, o.w.huber)).sum::<f64>();
                }
            }
        }
        eprintln!("warm-start pod loss: frame gyro {:.0}, fork pod {:.0}, rear pod {:.0}, total {:.0}", by[0], by[1], by[2], by.iter().sum::<f64>());
    }
    let (vs, info) = gauss_newton_multi(&objs, vs, cfg.force_id_iters, cfg.tol)?;
    if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
        eprintln!("force identification: warm starts {t_warm:.1} s, joint solve {:.1} s (assembly / solve / loss {:.1?} s)", t0.elapsed().as_secs_f64() - t_warm, info.seconds);
        eprintln!("force identification: {} windows, {} iterations, loss {:?}", objs.len(), info.iterations, info.loss);
        for (o, v) in objs.iter().zip(&vs) {
            eprintln!("  window loss by row group: {}", row_group_loss(o, v));
        }
        match super::solve::globals_covariance(&objs, &vs) {
            Ok((cov, ng)) => {
                let (g0, p) = (objs[0].force_glob0(), objs[0].force_par(&vs[0]));
                let sig = |i: usize| cov[(g0 + i) * ng + g0 + i].max(0.0).sqrt();
                let txt: Vec<String> = super::forces::PAR_NAMES.iter().enumerate().map(|(i, n)| if super::forces::additive(i) { format!("{n} {:+.3} (+- {:.3})", p[i], sig(i)) } else { format!("{n} {:.2} (x/ {:.2})", p[i].exp(), sig(i).exp()) }).collect();
                eprintln!("  force parameters, factor on the nominal (x/ one sigma): {}", txt.join(", "));
                // the pairs the data cannot tell apart: the strongest correlations
                let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
                for i in 0..super::forces::NFP {
                    for j in 0..i {
                        pairs.push((cov[(g0 + i) * ng + g0 + j] / (sig(i) * sig(j)).max(f64::MIN_POSITIVE), i, j));
                    }
                }
                pairs.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
                let names = &super::forces::PAR_NAMES;
                eprintln!("  strongest correlations: {}", pairs.iter().take(12).map(|(r, i, j)| format!("{} ~ {} {r:+.2}", names[*i], names[*j])).collect::<Vec<_>>().join(", "));
            }
            Err(e) => eprintln!("  force parameter sigmas failed: {e:?}"),
        }
    }
    if std::env::var_os("IDL_FORCE_OBSMAP").is_some() {
        observability_map(&objs, &vs);
    }
    Ok(objs[0].force_par(&vs[0]))
}

/// Half the sum of squares of a solution's rows, by group (a diagnostic): the pod rows and the
/// core priors, then each force row group, then the smoothness.
pub fn row_group_loss(o: &Objective, v: &Vars) -> String {
    let ctx = o.ctx(&v.glob, false);
    let groups = [("fork", 0, 1), ("rear", 1, 2), ("riderF", 2, 5), ("riderM", 5, 8), ("sysF fwd", 8, 9), ("sysF lat", 9, 10), ("sysF up", 10, 11), ("sysM", 11, 14), ("tyreT", 14, 18), ("pose", 18, 24), ("act", 24, 30), ("defl", 30, 32), ("head", 32, 35), ("camA", 35, 38), ("camG", 38, 41), ("headP", 41, 47), ("ground", 47, 49)];
    let k = o.forces.as_ref().map_or(0.0, |m| m.w.huber);
    let (mut sensor, mut prior) = (0.0, 0.0);
    let mut fg = [0.0; 17];
    for t in 0..o.spec.n {
        let s = o.sample(v, t);
        let r = o.sample_residuals(&ctx, &s, &v.glob, t, false).r;
        sensor += r[..super::objective::SENSOR_ROWS].iter().map(|&x| super::objective::huber2(x, o.w.huber)).sum::<f64>();
        prior += r[super::objective::SENSOR_ROWS..].iter().map(|x| x * x).sum::<f64>();
        if let Some(fj) = o.force_residuals(&ctx, v, &s, t, false) {
            for (g, &(_, a, b)) in groups.iter().enumerate() {
                fg[g] += (a..b).map(|i| if super::forces::huber_row(i) { super::objective::huber2(fj.r[i], k) } else { fj.r[i] * fj.r[i] }).sum::<f64>();
            }
        }
    }
    let mut out = format!("pods {:.0}, core priors {:.0}", 0.5 * sensor, 0.5 * prior);
    for (g, &(name, _, _)) in groups.iter().enumerate() {
        out += &format!(", {name} {:.0}", 0.5 * fg[g]);
    }
    out += &format!("; total {:.0}", o.loss(v));
    out
}

/// The force outputs: count of [`FORCE_CHANNELS`].
pub const NFO: usize = 23;

/// The force output channels (with [`Config::forces`]): id and unit, in [`force_outputs`]
/// order. The rider's force and moment are what the bike puts on him (the frame gets the
/// opposite), bike axes; his pump is the active part of them.
pub const FORCE_CHANNELS: [(&str, &str); NFO] = [
    ("Physics front tyre force (N)", "N"),
    ("Physics rear tyre force (N)", "N"),
    ("Physics front tyre deflection (mm)", "mm"),
    ("Physics rear tyre deflection (mm)", "mm"),
    ("Physics front tyre tangential (N)", "N"),
    ("Physics front tyre lateral (N)", "N"),
    ("Physics rear tyre tangential (N)", "N"),
    ("Physics rear tyre lateral (N)", "N"),
    ("Physics fork force (N)", "N"),
    ("Physics shock force (N)", "N"),
    ("Physics rider force fwd (N)", "N"),
    ("Physics rider force lat (N)", "N"),
    ("Physics rider force up (N)", "N"),
    ("Physics rider moment roll (N m)", "N m"),
    ("Physics rider moment pitch (N m)", "N m"),
    ("Physics rider moment yaw (N m)", "N m"),
    ("Physics rider pump fwd (N)", "N"),
    ("Physics rider pump lat (N)", "N"),
    ("Physics rider pump up (N)", "N"),
    ("Physics rider pump roll (N m)", "N m"),
    ("Physics rider pump pitch (N m)", "N m"),
    ("Physics rider pump yaw (N m)", "N m"),
    ("Physics ground (mm)", "mm"),
];

/// One sample's force outputs, in [`FORCE_CHANNELS`] order (NaN with the force model off).
pub fn force_outputs(obj: &Objective, ctx: &Ctx, v: &Vars, t: usize) -> [f64; NFO] {
    let (Some((fr, f)), Some(m)) = (obj.forces_at(ctx, v, t), obj.forces.as_ref()) else {
        return [f64::NAN; NFO];
    };
    let (fwd, lat, up) = (m.bike.fwd, m.bike.lat(), m.bike.up);
    let comps = |x: V3<f64>| [x.dot(fwd), x.dot(lat), x.dot(up)];
    let mut o = [0.0; NFO];
    o[..10].copy_from_slice(&[fr.normal[0], fr.normal[1], f[0][DEF_F] * 1000.0, f[0][DEF_R] * 1000.0, f[0][TAN_F], f[0][LAT_F], f[0][TAN_R], f[0][LAT_R], fr.fork, fr.shock]);
    let (bf, bm) = (comps(fr.bush[0]), comps(fr.bush[1]));
    for i in 0..3 {
        o[10 + i] = bf[i] + f[0][ACT_F + i];
        o[13 + i] = bm[i] + f[0][ACT_M + i];
        o[16 + i] = f[0][ACT_F + i];
        o[19 + i] = f[0][ACT_M + i];
    }
    // the ground under the front patch (F4), window-local: NaN without the tie
    o[22] = match (obj.win.forces.as_ref().and_then(|w| w.ground.as_ref()), &v.ground) {
        (Some(gw), Some(gf)) => gf.height(gw.s_m[t] + gw.offset_m[0]) * 1000.0,
        _ => f64::NAN,
    };
    o
}

/// The output channels: (id, unit, latent index, scale to the unit).
pub const OUTPUT_CHANNELS: [(&str, &str, usize, f64); 14] = [
    ("Physics front travel (mm)", "mm", D, 1000.0),
    ("Physics rear travel (mm)", "mm", S, 1000.0),
    ("Physics steer (deg)", "deg", PSI, 57.295_779_513_082_32),
    ("Physics fork bend fore-aft (deg)", "deg", B_FA, 57.295_779_513_082_32),
    ("Physics fork bend lateral (deg)", "deg", B_LAT, 57.295_779_513_082_32),
    ("Physics rear flex roll (deg)", "deg", R_ROLL, 57.295_779_513_082_32),
    ("Physics rear flex yaw (deg)", "deg", R_YAW, 57.295_779_513_082_32),
    ("Physics frame flex rot X (deg)", "deg", F_ROT, 57.295_779_513_082_32),
    ("Physics frame flex rot Y (deg)", "deg", F_ROT + 1, 57.295_779_513_082_32),
    ("Physics frame flex rot Z (deg)", "deg", F_ROT + 2, 57.295_779_513_082_32),
    ("Physics frame flex X (mm)", "mm", F_TRANS, 1000.0),
    ("Physics frame flex Y (mm)", "mm", F_TRANS + 1, 1000.0),
    ("Physics frame flex Z (mm)", "mm", F_TRANS + 2, 1000.0),
    // last: readers take the travel channels by position
    ("Physics fork twist (deg)", "deg", F_TWIST, 57.295_779_513_082_32),
];

/// One output channel: id, unit, values per grid sample (NaN where not solved).
pub type Output = (String, String, Vec<f64>);

/// The solution as named channels on the ride's grid ([`OUTPUT_CHANNELS`]),
/// plus `Physics airborne` (1 where the airtime anchor held).
pub fn outputs(sol: &Solution) -> Vec<Output> {
    let mut out: Vec<Output> =
        OUTPUT_CHANNELS.iter().map(|&(id, unit, lat, k)| (id.to_string(), unit.to_string(), sol.q.iter().map(|q| q[lat] * k).collect())).collect();
    out.push(("Physics airborne".to_string(), String::new(), sol.air.iter().map(|&a| a as u8 as f64).collect()));
    if !sol.forces.is_empty() {
        for (k, &(id, unit)) in FORCE_CHANNELS.iter().enumerate() {
            out.push((id.to_string(), unit.to_string(), sol.forces.iter().map(|f| f[k]).collect()));
        }
    }
    if !sol.nav.is_empty() {
        let per: Vec<[f64; 8]> = sol.nav.iter().map(nav_outputs).collect();
        for (k, &(id, unit)) in NAV_CHANNELS.iter().enumerate() {
            out.push((id.to_string(), unit.to_string(), per.iter().map(|p| p[k]).collect()));
        }
    }
    out
}

/// The navigation output channels (with [`Config::nav`]): id and unit, in [`nav_outputs`]' order.
pub const NAV_CHANNELS: [(&str, &str); 8] = [
    ("Physics roll (deg)", "deg"),
    ("Physics pitch (deg)", "deg"),
    ("Physics heading (deg)", "deg"),
    ("Physics course (deg)", "deg"),
    ("Physics slip (deg)", "deg"),
    ("Physics speed (m/s)", "m/s"),
    ("Physics long accel (g)", "g"),
    ("Physics lat accel (g)", "g"),
];

/// Below this ground speed course and slip are blank, m/s.
const COURSE_MIN_SPEED_MPS: f64 = 3.0;

/// One sample's navigation outputs, in [`NAV_CHANNELS`]' order. Conventions as
/// the IEKF's (`estimate/attitude.rs`): roll + right side down, pitch + nose
/// up, heading and course degrees clockwise from north, slip = heading -
/// course (+ nose right of travel), long accel + speeding up, lat + to the right.
pub fn nav_outputs(s: &NavState) -> [f64; 8] {
    if !s.r[0][0].is_finite() {
        return [f64::NAN; 8];
    }
    let m = nalgebra::Matrix3::from_fn(|i, j| s.r[i][j]);
    let q = nalgebra::UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(m));
    let (roll, pitch) = crate::estimate::attitude::roll_pitch_deg(&q);
    let (_, _, yaw) = q.euler_angles();
    let heading = (-yaw).to_degrees().rem_euclid(360.0); // yaw is about up, toward west
    let speed = s.v[0].hypot(s.v[1]);
    let (course, slip) = if speed > COURSE_MIN_SPEED_MPS {
        let c = (-s.v[1]).atan2(s.v[0]).to_degrees().rem_euclid(360.0);
        (c, (heading - c + 540.0).rem_euclid(360.0) - 180.0)
    } else {
        (f64::NAN, f64::NAN)
    };
    let g = crate::estimate::attitude::G_MPS2;
    let body: [f64; 3] = std::array::from_fn(|j| (0..3).map(|i| s.r[i][j] * s.vd[i]).sum::<f64>() / g); // R^T a
    [roll, pitch, heading, course, slip, speed, body[0], -body[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_windows_tiles_each_riding_stretch_exactly_with_overlap() {
        // Arrange: two stretches, 0..1000 and 1500..1530 (too short), and 2000..2350
        let mut mov = vec![false; 3000];
        for (a, b) in [(0, 1000), (1500, 1530), (2000, 2350)] {
            mov[a..b].iter_mut().for_each(|m| *m = true);
        }

        // Act
        let p = plan_windows(&mov, 400, 80, 100);

        // Assert: kept parts are contiguous and cover each stretch, each inside its window
        let mut covered = vec![false; 3000];
        for &(a, b, ks, ke) in &p {
            assert!(a <= ks && ks < ke && ke <= b && b - a <= 400);
            covered[ks..ke].iter_mut().for_each(|c| *c = true);
        }
        assert_eq!(covered[..1000], mov[..1000]);
        assert!(covered[1500..1530].iter().all(|c| !c));
        assert_eq!(covered[2000..2350], mov[2000..2350]);
        let kept: usize = p.iter().map(|&(_, _, ks, ke)| ke - ks).sum();
        assert_eq!(kept, 1000 + 350);
    }
}

/// A short stable hash of a geometry, for the output's staleness stamp.
pub fn geometry_hash(g: &Geometry) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for x in g {
        h.update(x.to_le_bytes());
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// `bike` with the geometry of `report`.
pub fn fitted_bike(bike: &Bike<f64>, report: &FitReport) -> Bike<f64> {
    with_geometry(bike, &report.geometry)
}

/// The geometry `bike` runs with.
pub fn bike_geometry(bike: &Bike<f64>) -> Geometry {
    geometry(bike)
}

/// The observability map on a fit's windows (`IDL_FORCE_OBSMAP`, prototyping): each force
/// parameter's sigma from the data alone (its prior opened to 10 in log; the reduced Hessian at
/// the fit), with the rider and the tyres' deflection free, either taken as measured (their
/// priors closed), and both; printed as factors (x/: 1.05 is 5 %, 22026 not determined). The
/// information a parameter gets depends on the motion it sees, which here is the ride's.
pub fn observability_map(objs: &[Objective], vs: &[Vars]) {
    use super::forces::{NFP, PAR_NAMES, P_BRAKE_SHIFT, P_DAMP_MASS, P_LIMBS, P_NECK_K};
    let cases = [("free", false, false), ("rider known", true, false), ("deflection known", false, true), ("both known", true, true)];
    let mut table: Vec<Vec<f64>> = Vec::new();
    for &(name, rider, defl) in &cases {
        let os: Vec<Objective> = objs
            .iter()
            .map(|o| {
                let mut m = o.forces.clone().expect("a force fit");
                for i in 0..NFP {
                    m.w.par_sigma[i] = if (P_NECK_K..P_DAMP_MASS).contains(&i) || i == P_LIMBS || (P_BRAKE_SHIFT..P_BRAKE_SHIFT + 2).contains(&i) { 1e-3 } else { 10.0 };
                }
                if rider {
                    m.w.pos_m = [1e-5; 3];
                    m.w.rot_rad = 1e-5;
                    m.w.act_n = 1e-3;
                    m.w.act_nm = 1e-4;
                }
                if defl {
                    m.w.defl_m = 1e-7;
                }
                let mut glob_centre = o.glob_centre.clone();
                let g0 = o.force_glob0();
                glob_centre[g0..g0 + NFP].copy_from_slice(&o.force_par(&vs[0]));
                Objective { bike: o.bike.clone(), win: o.win.clone(), spec: o.spec.clone(), w: o.w.clone(), glob_centre, fit: o.fit, forces: Some(m), threads: o.threads }
            })
            .collect();
        match super::solve::globals_covariance(&os, vs) {
            Ok((cov, ng)) => {
                let g0 = os[0].force_glob0();
                table.push((0..NFP).map(|i| cov[(g0 + i) * ng + g0 + i].max(0.0).sqrt().exp()).collect());
            }
            Err(e) => {
                eprintln!("observability map, {name}: {e:?}");
                table.push(vec![f64::NAN; NFP]);
            }
        }
    }
    eprintln!("observability map (x/ one sigma from the data alone):");
    eprintln!("  {:<24}{}", "parameter", cases.iter().map(|c| format!("{:>18}", c.0)).collect::<String>());
    for i in (0..P_NECK_K).chain(P_DAMP_MASS..NFP) {
        if i == P_LIMBS || (P_BRAKE_SHIFT..P_BRAKE_SHIFT + 2).contains(&i) {
            continue;
        }
        eprintln!("  {:<24}{}", PAR_NAMES[i], table.iter().map(|t| format!("{:>18.3}", t[i])).collect::<String>());
    }
}
