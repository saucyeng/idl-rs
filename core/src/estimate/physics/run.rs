//! Running the physics estimator over a session: learn the bike's geometry
//! from a few windows (once per calibration), then solve every stretch of
//! riding in overlapping windows with it frozen, and write the result as a
//! derived session file of named channels.

use super::bike::{Bike, B_FA, B_LAT, D, F_ROT, F_TRANS, F_TWIST, NL, PSI, R_ROLL, R_YAW, S};
use super::forces::{ForceModel, ForceWindow, ACT_F, ACT_M, DEF_F, DEF_R, LAT_F, LAT_R, NFP, TAN_F, TAN_R};
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
    let gps: Option<Vec<Option<GpsFix>>> = cfg.nav.then(|| gps_on_grid(r, gps_latency(r, bike)));
    let axes = bike_axes(bike);
    // the force model: its parameters identified first, then frozen for every window
    let speed: Option<Vec<f64>> = cfg.forces.as_ref().map(|_| speed_on_grid(r, gps_latency(r, bike)));
    let force_par = match (&cfg.forces, &speed) {
        (Some(m), Some(sp)) => identify_forces(r, bike, bias_centre, cfg, m, sp, &plan)
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
            }
            if let Some(sp) = &speed {
                win.forces = Some(ForceWindow { speed_mps: sp[a..b].to_vec() });
            }
            let mut obj = Objective::new(bike.clone(), win, spec, cfg.weights.clone(), bias_centre);
            obj.threads = inner;
            if let Some(m) = &frozen {
                obj = obj.with_forces(m.clone());
                let g0 = obj.force_glob0();
                obj.glob_centre[g0..g0 + NFP].copy_from_slice(&force_par.unwrap_or([0.0; NFP]));
            }
            let solved = warm_start(&obj, cfg.iters, cfg.tol).and_then(|v0| gauss_newton(&obj, v0, cfg.iters, cfg.tol));
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

/// Identifies the force parameters on [`Config::force_id_windows`] windows spread through
/// the ride ([`Config::force_id_window_s`] each, from the start of every plan window picked),
/// solved together with shared parameters (and biases). Returns the log multipliers.
pub fn identify_forces(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config, m: &ForceModel, speed: &[f64], plan: &[(usize, usize, usize, usize)]) -> Result<[f64; NFP], SolveError> {
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
            win.forces = Some(ForceWindow { speed_mps: speed[a..b].to_vec() });
            let o = Objective::new(bike.clone(), win, Spec::standard(b - a, r.input.dt), cfg.weights.clone(), bias_centre).with_forces(m.clone());
            Objective { threads: (cfg.threads / k).max(1), ..o }
        })
        .collect();
    if std::env::var_os("IDL_FORCE_DEBUG").is_some() {
        let lay = super::solve::Layout::new(&objs[0]);
        eprintln!("force identification: window 0 has {} samples, {} local unknowns, band half-width {}, {} globals", objs[0].spec.n, lay.n_loc, lay.bw, objs[0].n_glob());
    }
    let vs = super::solve::par_map_n(&objs, k, |o| warm_start(o, cfg.iters, cfg.tol)).into_iter().collect::<Result<Vec<_>, _>>()?;
    let t_warm = t0.elapsed().as_secs_f64();
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
                let txt: Vec<String> = super::forces::PAR_NAMES.iter().enumerate().map(|(i, n)| format!("{n} {:.2} (x/ {:.2})", p[i].exp(), sig(i).exp())).collect();
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
    Ok(objs[0].force_par(&vs[0]))
}

/// Half the sum of squares of a solution's rows, by group (a diagnostic): the pod rows and the
/// core priors, then each force row group, then the smoothness.
pub fn row_group_loss(o: &Objective, v: &Vars) -> String {
    let ctx = o.ctx(&v.glob, false);
    let groups = [("fork", 0, 1), ("rear", 1, 2), ("riderF", 2, 5), ("riderM", 5, 8), ("sysF", 8, 11), ("sysM", 11, 14), ("tyreT", 14, 18), ("pose", 18, 24), ("act", 24, 30), ("defl", 30, 32)];
    let k = o.forces.as_ref().map_or(0.0, |m| m.w.huber);
    let (mut sensor, mut prior) = (0.0, 0.0);
    let mut fg = [0.0; 10];
    for t in 0..o.spec.n {
        let s = o.sample(v, t);
        let r = o.sample_residuals(&ctx, &s, &v.glob, t, false).r;
        sensor += r[..super::objective::SENSOR_ROWS].iter().map(|&x| super::objective::huber2(x, o.w.huber)).sum::<f64>();
        prior += r[super::objective::SENSOR_ROWS..].iter().map(|x| x * x).sum::<f64>();
        if let Some(fj) = o.force_residuals(&ctx, v, &s, t, false) {
            for (g, &(_, a, b)) in groups.iter().enumerate() {
                fg[g] += fj.r[a..b].iter().map(|&x| if b <= super::forces::NBAL { super::objective::huber2(x, k) } else { x * x }).sum::<f64>();
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
pub const NFO: usize = 22;

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
