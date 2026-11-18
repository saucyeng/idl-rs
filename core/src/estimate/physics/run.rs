//! Running the physics estimator over a session: learn the bike's geometry
//! from a few windows (once per calibration), then solve every stretch of
//! riding in overlapping windows with it frozen, and write the result as a
//! derived session file of named channels.

use super::bike::{Bike, B_FA, B_LAT, D, F_ROT, F_TRANS, NL, PSI, R_ROLL, R_YAW, S};
use super::objective::{Objective, Spec, Weights, NG};
use super::pipeline::{airtime_score, moving, pick_windows, rotation_score, window, Ride};
use super::setup::{geometry, with_geometry, Geometry};
use super::solve::{gauss_newton, gauss_newton_multi, latents, SolveError, SolveInfo};

/// Version of the estimator's output. Bump when the same inputs give
/// different channels (a model or solver change); stored in the output file
/// so an older file can be shown as stale.
pub const PHYSICS_ESTIMATOR_VERSION: u32 = 2; // 2: IMUs low-passed at 0.4 x the grid rate

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
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(4)),
            weights: Weights::default(),
        }
    }
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
    let objs: Vec<Objective> = windows
        .iter()
        .map(|&(a, b)| Objective::fitting(bike.clone(), window(r, a, b, cfg.air_anchor), Spec::standard(b - a, r.input.dt), cfg.weights.clone(), bias_centre))
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
}

/// The solve windows over the riding stretches: (start, end, keep_start, keep_end).
fn plan(r: &Ride, cfg: &Config) -> Vec<(usize, usize, usize, usize)> {
    let rate = r.rate();
    plan_windows(&moving(r), (cfg.window_s * rate) as usize, (cfg.overlap_s * rate) as usize, (2.0 * rate) as usize)
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

/// Solves every riding stretch of `r` with `bike` frozen, windows in parallel.
pub fn solve_ride(r: &Ride, bike: &Bike<f64>, bias_centre: [f64; NG], cfg: &Config) -> Solution {
    let plan = plan(r, cfg);
    let solve_one = |&(a, b, _, _): &(usize, usize, usize, usize)| -> Result<(Vec<[f64; NL]>, Option<Vec<bool>>, usize), SolveError> {
        let win = window(r, a, b, cfg.air_anchor);
        let air = win.air.clone();
        let obj = Objective::new(bike.clone(), win, Spec::standard(b - a, r.input.dt), cfg.weights.clone(), bias_centre);
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), cfg.iters, cfg.tol)?;
        Ok((latents(&obj, &v), air, info.iterations))
    };
    let threads = cfg.threads.max(1);
    let mut results: Vec<Option<Result<_, SolveError>>> = (0..plan.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        let chunks: Vec<Vec<usize>> = (0..threads).map(|t| (t..plan.len()).step_by(threads).collect()).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|idx| {
                let plan = &plan;
                let solve_one = &solve_one;
                sc.spawn(move || idx.into_iter().map(|i| (i, solve_one(&plan[i]))).collect::<Vec<_>>())
            })
            .collect();
        for h in handles {
            for (i, res) in h.join().expect("solve thread panicked") {
                results[i] = Some(res);
            }
        }
    });
    let mut sol = Solution { q: vec![[f64::NAN; NL]; r.len()], air: vec![false; r.len()], windows: (plan.len(), 0), iterations: 0 };
    for (&(a, _, ks, ke), res) in plan.iter().zip(results) {
        match res.expect("every window solved") {
            Ok((q, air, it)) => {
                sol.iterations += it;
                for t in ks..ke {
                    sol.q[t] = q[t - a];
                    sol.air[t] = air.as_ref().is_some_and(|m| m[t - a]);
                }
            }
            Err(_) => sol.windows.1 += 1,
        }
    }
    sol
}

/// The output channels: (id, unit, latent index, scale to the unit).
pub const OUTPUT_CHANNELS: [(&str, &str, usize, f64); 13] = [
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
];

/// One output channel: id, unit, values per grid sample (NaN where not solved).
pub type Output = (String, String, Vec<f64>);

/// The solution as named channels on the ride's grid ([`OUTPUT_CHANNELS`]),
/// plus `Physics airborne` (1 where the airtime anchor held).
pub fn outputs(sol: &Solution) -> Vec<Output> {
    let mut out: Vec<Output> =
        OUTPUT_CHANNELS.iter().map(|&(id, unit, lat, k)| (id.to_string(), unit.to_string(), sol.q.iter().map(|q| q[lat] * k).collect())).collect();
    out.push(("Physics airborne".to_string(), String::new(), sol.air.iter().map(|&a| a as u8 as f64).collect()));
    out
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
