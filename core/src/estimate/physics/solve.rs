//! Sparse Gauss-Newton / Levenberg-Marquardt over one window. A port of
//! `ml/idlml/solve/gn.py` (CPU, float64).
//!
//! Each sample's residuals depend on the 4 coefficients of each spline row
//! that touch that sample, and on the globals. With every coefficient ordered
//! by the time it acts at, the normal matrix is a band (half-width ~ the
//! coefficients any one sample touches) plus a dense border for the globals:
//! a banded Cholesky and a Schur complement solve it in O(n bw^2). Damping
//! starts near zero (Marquardt damping brakes hardest on exactly the slow,
//! nearly flat travel modes; the problem is close to linear).

use super::objective::{basis, Objective, Vars, NROW};
use super::bike::NL;

/// One spline row's place in the unknown vector.
struct Group {
    /// None: the frame-rate spline axis `row`; Some(level): a latent row.
    level: Option<usize>,
    row: usize,
    m: usize,
    scale: f64,
    /// Local-input columns this row's value, rate and acceleration feed.
    z: [Option<usize>; 3],
    /// Position in the unknown vector of each of the row's coefficients.
    pos: Vec<usize>,
}

/// Unknowns ordered by the time they act at.
pub struct Layout {
    groups: Vec<Group>,
    /// Local unknowns (spline coefficients).
    pub n_loc: usize,
    /// Band half-width of the normal matrix.
    pub bw: usize,
}

impl Layout {
    /// The layout for `obj`'s spec (latent levels, then the frame-rate axes).
    pub fn new(obj: &Objective) -> Self {
        let sp = &obj.spec;
        let mut groups = Vec::new();
        for (li, (m, ix)) in sp.levels.iter().enumerate() {
            for (row, &lat) in ix.iter().enumerate() {
                groups.push(Group { level: Some(li), row, m: *m, scale: sp.scale[lat], z: [Some(lat), Some(NL + lat), Some(2 * NL + lat)], pos: vec![] });
            }
        }
        for x in 0..3 {
            groups.push(Group { level: None, row: x, m: 1, scale: 1.0, z: [Some(3 * NL + x), Some(3 * NL + 3 + x), None], pos: vec![] });
        }
        // order every coefficient by when it acts (stable: group order breaks ties)
        let mut keys: Vec<(f64, usize, usize)> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            for k in 0..sp.n_knots(g.m) {
                keys.push(((k as f64 - 1.5) * g.m as f64, gi, k));
            }
        }
        keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for g in groups.iter_mut() {
            g.pos = vec![0; sp.n_knots(g.m)];
        }
        for (p, &(_, gi, k)) in keys.iter().enumerate() {
            groups[gi].pos[k] = p;
        }
        let n_loc = keys.len();
        // band: widest span of the coefficients one sample, or one smoothness prior, touches
        let mut bw = 0;
        for t in 0..sp.n {
            let (mut lo, mut hi) = (usize::MAX, 0);
            for g in &groups {
                let b = t / g.m;
                lo = lo.min(g.pos[b]);
                hi = hi.max(g.pos[b + 3]);
            }
            bw = bw.max(hi - lo);
        }
        for g in &groups {
            for i in 0..g.pos.len() - 2 {
                bw = bw.max(g.pos[i + 2] - g.pos[i]);
            }
        }
        Layout { groups, n_loc, bw }
    }

    fn coef<'a>(&self, g: &Group, v: &'a Vars) -> &'a [f64] {
        match g.level {
            Some(li) => &v.lvl[li][g.row],
            None => &v.frame[g.row],
        }
    }

    /// `v` stepped by `dl` (local, in layout order) and `dg` (globals).
    pub fn apply(&self, v: &Vars, dl: &[f64], dg: &[f64]) -> Vars {
        let mut out = v.clone();
        for g in &self.groups {
            let c = match g.level {
                Some(li) => &mut out.lvl[li][g.row],
                None => &mut out.frame[g.row],
            };
            for (k, &p) in g.pos.iter().enumerate() {
                c[k] += dl[p];
            }
        }
        for (g, d) in out.glob.iter_mut().zip(dg) {
            *g += d;
        }
        out
    }
}

/// The normal equations J^T J x = -J^T r in band + border form.
pub struct Normal {
    /// Number of globals.
    pub ng: usize,
    /// Lower band, row-major: row i holds columns i-bw..=i (`band[i*(bw+1) + (j + bw - i)]`).
    pub band: Vec<f64>,
    /// Border: d^2 / d local d global, row-major [n_loc][ng].
    pub border: Vec<f64>,
    /// Globals block, row-major [ng][ng].
    pub c: Vec<f64>,
    /// Gradient, local.
    pub g_loc: Vec<f64>,
    /// Gradient, globals.
    pub g_glob: Vec<f64>,
    /// Loss at the linearisation point.
    pub loss: f64,
}

/// Assemble the normal equations at `v`; `global_prior`: include the priors
/// on the globals (a multi-window fit counts them once).
pub fn normal_equations(obj: &Objective, lay: &Layout, v: &Vars, global_prior: bool) -> Normal {
    let (n, bw, ng) = (lay.n_loc, lay.bw, obj.n_glob());
    let wd = bw + 1;
    let mut ne = Normal { ng, band: vec![0.0; n * wd], border: vec![0.0; n * ng], c: vec![0.0; ng * ng], g_loc: vec![0.0; n], g_glob: vec![0.0; ng], loss: 0.0 };
    let sp = &obj.spec;
    let ctx = obj.ctx(&v.glob, true);
    let ncol = lay.groups.len() * 4;
    let mut a = vec![0.0; NROW * ncol];
    let mut cols = vec![0usize; ncol];
    for t in 0..sp.n {
        let s = obj.sample(v, t);
        let sj = obj.sample_residuals(&ctx, &s, &v.glob, t, true);
        // chain d r / d z through the spline basis to the coefficients
        a.iter_mut().for_each(|x| *x = 0.0);
        for (gi, g) in lay.groups.iter().enumerate() {
            let (j, base) = (t % g.m, t / g.m);
            let b = basis(j, g.m);
            let h = g.m as f64 * sp.dt;
            let wz = [g.scale, g.scale / h, g.scale / (h * h)];
            for i in 0..4 {
                cols[gi * 4 + i] = g.pos[base + i];
                for (d, z) in g.z.iter().enumerate() {
                    if let Some(z) = *z {
                        let wgt = b[d][i] * wz[d];
                        for r in 0..NROW {
                            a[r * ncol + gi * 4 + i] += sj.jz[r][z] * wgt;
                        }
                    }
                }
            }
        }
        for r in 0..NROW {
            ne.loss += 0.5 * sj.r[r] * sj.r[r];
        }
        for p in 0..ncol {
            let cp = cols[p];
            let mut gp = 0.0;
            for r in 0..NROW {
                gp += a[r * ncol + p] * sj.r[r];
            }
            ne.g_loc[cp] += gp;
            for q in 0..ncol {
                let cq = cols[q];
                if cq > cp {
                    continue;
                }
                let mut h = 0.0;
                for r in 0..NROW {
                    h += a[r * ncol + p] * a[r * ncol + q];
                }
                ne.band[cp * wd + (cq + bw - cp)] += h;
            }
            for gq in 0..ng {
                let mut h = 0.0;
                for r in 0..NROW {
                    h += a[r * ncol + p] * sj.jg[r][gq];
                }
                ne.border[cp * ng + gq] += h;
            }
        }
        for gp in 0..ng {
            for r in 0..NROW {
                ne.g_glob[gp] += sj.jg[r][gp] * sj.r[r];
            }
            for gq in 0..ng {
                let mut h = 0.0;
                for r in 0..NROW {
                    h += sj.jg[r][gp] * sj.jg[r][gq];
                }
                ne.c[gp * ng + gq] += h;
            }
        }
    }
    // smoothness priors: k (c_i - 2 c_i+1 + c_i+2) on each latent row
    let k = obj.smooth_weight();
    let pw = [k, -2.0 * k, k];
    for g in lay.groups.iter().filter(|g| g.level.is_some()) {
        let c = lay.coef(g, v);
        for i in 0..c.len() - 2 {
            let r = k * (c[i] - 2.0 * c[i + 1] + c[i + 2]);
            ne.loss += 0.5 * r * r;
            for a_ in 0..3 {
                let pa = g.pos[i + a_];
                ne.g_loc[pa] += pw[a_] * r;
                for b_ in 0..3 {
                    let pb = g.pos[i + b_];
                    if pb <= pa {
                        ne.band[pa * wd + (pb + bw - pa)] += pw[a_] * pw[b_];
                    }
                }
            }
        }
    }
    if global_prior {
        let sig = obj.glob_sigma();
        for i in 0..ng {
            let gw = 1.0 / sig[i];
            let gp = (v.glob[i] - obj.glob_centre[i]) * gw;
            ne.c[i * ng + i] += gw * gw;
            ne.g_glob[i] += gw * gp;
            ne.loss += 0.5 * gp * gp;
        }
    }
    ne
}

/// In-place Cholesky of the lower band (row-major, see [`Normal::band`]).
fn band_cholesky(band: &mut [f64], n: usize, bw: usize) -> Result<(), SolveError> {
    let wd = bw + 1;
    for i in 0..n {
        let lo_i = i.saturating_sub(bw);
        for j in lo_i..=i {
            let lo = lo_i.max(j.saturating_sub(bw));
            let mut s = band[i * wd + (j + bw - i)];
            let ri = &band[i * wd + (lo + bw - i)..i * wd + (j + bw - i)];
            let rj = &band[j * wd + (lo + bw - j)..j * wd + bw];
            s -= ri.iter().zip(rj).map(|(a, b)| a * b).sum::<f64>();
            if j == i {
                if !(s > 0.0) {
                    return Err(SolveError::NotPositiveDefinite { row: i });
                }
                band[i * wd + bw] = s.sqrt();
            } else {
                band[i * wd + (j + bw - i)] = s / band[j * wd + bw];
            }
        }
    }
    Ok(())
}

/// Solve L L^T x = b in place, L the factored band.
fn band_solve(l: &[f64], n: usize, bw: usize, x: &mut [f64]) {
    let wd = bw + 1;
    for i in 0..n {
        let lo = i.saturating_sub(bw);
        let row = &l[i * wd + (lo + bw - i)..i * wd + bw];
        let s: f64 = row.iter().zip(&x[lo..i]).map(|(a, b)| a * b).sum();
        x[i] = (x[i] - s) / l[i * wd + bw];
    }
    for i in (0..n).rev() {
        x[i] /= l[i * wd + bw];
        let xi = x[i];
        let lo = i.saturating_sub(bw);
        let row = &l[i * wd + (lo + bw - i)..i * wd + bw];
        for (xk, a) in x[lo..i].iter_mut().zip(row) {
            *xk -= a * xi;
        }
    }
}

/// Why a solve failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The damped normal matrix was not positive definite at this row (the
    /// problem is degenerate there, e.g. unobserved latents with no prior).
    NotPositiveDefinite {
        /// Row of the band (or n_loc for the globals' block) where it failed.
        row: usize,
    },
}

/// One window's damped band solved against -g_loc and the border: A^-1 (-g_loc), A^-1 B.
struct Factored {
    ar: Vec<f64>,
    /// A^-1 B, column-major [ng][n_loc].
    ab: Vec<Vec<f64>>,
}

fn factor(ne: &Normal, lay: &Layout, lam: f64) -> Result<Factored, SolveError> {
    let (n, bw) = (lay.n_loc, lay.bw);
    let wd = bw + 1;
    let mut l = ne.band.clone();
    for i in 0..n {
        l[i * wd + bw] = l[i * wd + bw] * (1.0 + lam) + 1e-9;
    }
    band_cholesky(&mut l, n, bw)?;
    let mut ar: Vec<f64> = ne.g_loc.iter().map(|g| -g).collect();
    band_solve(&l, n, bw, &mut ar);
    let ab = (0..ne.ng)
        .map(|j| {
            let mut col: Vec<f64> = (0..n).map(|k| ne.border[k * ne.ng + j]).collect();
            band_solve(&l, n, bw, &mut col);
            col
        })
        .collect();
    Ok(Factored { ar, ab })
}

/// This window's share of the globals' Schur system: (C - B^T A^-1 B, -g - B^T A^-1 (-g_loc)).
fn schur_part(ne: &Normal, f: &Factored, lam: f64) -> (Vec<f64>, Vec<f64>) {
    let (ng, n) = (ne.ng, ne.g_loc.len());
    let mut s = vec![0.0; ng * ng];
    let mut rhs = vec![0.0; ng];
    for i in 0..ng {
        for j in 0..ng {
            let bab: f64 = (0..n).map(|k| ne.border[k * ng + i] * f.ab[j][k]).sum();
            s[i * ng + j] = ne.c[i * ng + j] - bab + if i == j { lam * ne.c[i * ng + i] + 1e-9 } else { 0.0 };
        }
        let bar: f64 = (0..n).map(|k| ne.border[k * ng + i] * f.ar[k]).sum();
        rhs[i] = -ne.g_glob[i] - bar;
    }
    (s, rhs)
}

fn solve_globals(s: &[f64], rhs: &[f64], ng: usize, n_loc: usize) -> Result<Vec<f64>, SolveError> {
    let m = nalgebra::DMatrix::from_row_slice(ng, ng, s);
    let x = m.cholesky().ok_or(SolveError::NotPositiveDefinite { row: n_loc })?.solve(&nalgebra::DVector::from_column_slice(rhs));
    Ok(x.iter().cloned().collect())
}

fn back_substitute(f: &Factored, xg: &[f64]) -> Vec<f64> {
    (0..f.ar.len()).map(|k| f.ar[k] - xg.iter().zip(&f.ab).map(|(x, col)| col[k] * x).sum::<f64>()).collect()
}

/// (H + lam diag(H)) x = -g for one window's band + border system.
pub fn solve_damped(ne: &Normal, lay: &Layout, lam: f64) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
    let f = factor(ne, lay, lam)?;
    let (s, rhs) = schur_part(ne, &f, lam);
    let xg = solve_globals(&s, &rhs, ne.ng, lay.n_loc)?;
    Ok((back_substitute(&f, &xg), xg))
}

/// What a solve did.
#[derive(Debug, Clone)]
pub struct SolveInfo {
    /// Loss after each accepted iteration (first entry: the start).
    pub loss: Vec<f64>,
    /// Iterations run.
    pub iterations: usize,
    /// Whether the last iteration gained less than the tolerance.
    pub converged: bool,
    /// Seconds assembling normal equations, solving, and evaluating the loss.
    pub seconds: [f64; 3],
}

/// Levenberg-Marquardt from `v` until an iteration gains less than `tol` in loss.
pub fn gauss_newton(obj: &Objective, v: Vars, iters: usize, tol: f64) -> Result<(Vars, SolveInfo), SolveError> {
    let (mut vs, info) = gauss_newton_multi(std::slice::from_ref(obj), vec![v], iters, tol)?;
    Ok((vs.remove(0), info))
}

/// Levenberg-Marquardt over several windows that share their globals (sensor
/// biases and, when fitting, the bike's geometry); the latents are per window.
/// Each window's band is factored on its own; the shared unknowns come from
/// the summed Schur complements; the global priors count once. How the bike
/// is learned from several windows before per-window solves with it frozen.
pub fn gauss_newton_multi(objs: &[Objective], mut vs: Vec<Vars>, iters: usize, tol: f64) -> Result<(Vec<Vars>, SolveInfo), SolveError> {
    let lays: Vec<Layout> = objs.iter().map(Layout::new).collect();
    let ng = objs[0].n_glob();
    // every window sees window 0's globals
    let share = |vs: &mut Vec<Vars>| {
        let g = vs[0].glob.clone();
        for v in vs.iter_mut() {
            v.glob = g.clone();
        }
    };
    share(&mut vs);
    let total = |vs: &[Vars]| objs.iter().zip(vs).map(|(o, v)| o.data_loss(v)).sum::<f64>() + objs[0].prior_loss(&vs[0]);
    let mut lam = 1e-10;
    let mut loss = total(&vs);
    let mut info = SolveInfo { loss: vec![loss], iterations: 0, seconds: [0.0; 3], converged: false };
    for it in 0..iters {
        let t0 = std::time::Instant::now();
        let nes: Vec<Normal> = objs.iter().zip(&vs).zip(&lays).enumerate().map(|(w, ((o, v), l))| normal_equations(o, l, v, w == 0)).collect();
        info.seconds[0] += t0.elapsed().as_secs_f64();
        let mut new;
        loop {
            let t1 = std::time::Instant::now();
            let mut fs = Vec::with_capacity(objs.len());
            let (mut s, mut rhs) = (vec![0.0; ng * ng], vec![0.0; ng]);
            for (ne, lay) in nes.iter().zip(&lays) {
                let f = factor(ne, lay, lam)?;
                let (sw, rw) = schur_part(ne, &f, lam);
                s.iter_mut().zip(&sw).for_each(|(a, b)| *a += b);
                rhs.iter_mut().zip(&rw).for_each(|(a, b)| *a += b);
                fs.push(f);
            }
            let xg = solve_globals(&s, &rhs, ng, lays[0].n_loc)?;
            let cand: Vec<Vars> = vs.iter().zip(&lays).zip(&fs).map(|((v, lay), f)| lay.apply(v, &back_substitute(f, &xg), &xg)).collect();
            info.seconds[1] += t1.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            new = total(&cand);
            info.seconds[2] += t2.elapsed().as_secs_f64();
            if new < loss {
                vs = cand;
                lam = (lam / 10.0).max(1e-12);
                break;
            }
            lam *= 10.0;
            if lam > 1e2 {
                break;
            }
        }
        let gain = loss - new;
        loss = loss.min(new);
        info.loss.push(loss);
        info.iterations = it + 1;
        if gain < tol || lam > 1e2 {
            info.converged = true;
            break;
        }
    }
    Ok((vs, info))
}

/// Latent values per sample, [n][NL], SI units.
pub fn latents(obj: &Objective, v: &Vars) -> Vec<[f64; NL]> {
    (0..obj.spec.n).map(|t| obj.sample(v, t).q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::physics::bike::{D, S};
    use crate::estimate::physics::fixture;
    use crate::estimate::physics::objective::{Spec, Weights, Window};

    fn rows(j: &serde_json::Value) -> Vec<[f64; 3]> {
        serde_json::from_value(j.clone()).unwrap()
    }

    #[test]
    fn solve_of_a_real_window_matches_the_pytorch_solver() {
        // Arrange: 2 s of the Beacon ride with a jump, as the PyTorch solver saw it
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/physics_window.json")).unwrap();
        let bike = fixture::bike(&doc["bike"]);
        let win = Window {
            rate: doc["rate"].as_f64().unwrap(),
            frame_gyro: rows(&doc["frame_gyro"]),
            frame_accel: rows(&doc["frame_accel"]),
            fork_gyro: rows(&doc["fork_gyro"]),
            fork_accel: rows(&doc["fork_accel"]),
            rear_gyro: rows(&doc["rear_gyro"]),
            rear_accel: rows(&doc["rear_accel"]),
            mask_fork: serde_json::from_value(doc["mask_fork"].clone()).unwrap(),
            mask_rear: serde_json::from_value(doc["mask_rear"].clone()).unwrap(),
            air: Some(serde_json::from_value(doc["air"].clone()).unwrap()),
        };
        let n = win.frame_gyro.len();
        let mut centre = [0.0; crate::estimate::physics::objective::NG];
        for (i, key) in ["frame_bias", "fork.gyro_bias", "fork.accel_bias", "rear.gyro_bias", "rear.accel_bias"].iter().enumerate() {
            let c = &doc["bias_prior"][key];
            for a in 0..3 {
                centre[3 * i + a] = c.as_f64().unwrap_or_else(|| c[a].as_f64().unwrap());
            }
        }
        let obj = Objective::new(bike, win, Spec::standard(n, 1.0 / 400.0), Weights::default(), centre);

        // Act
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), 30, 1.0).unwrap();
        let q = latents(&obj, &v);

        // Assert: the same minimum (PyTorch runs the physics in float32, so not bit-equal)
        let want: Vec<Vec<f64>> = serde_json::from_value(doc["solution"]["q"].clone()).unwrap();
        let want_loss = doc["solution"]["loss"].as_f64().unwrap();
        let rms = |lat: usize| (q.iter().zip(&want).map(|(a, b)| (a[lat] - b[lat]).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(rms(D) < 5e-4, "front travel differs by {:.2} mm rms", rms(D) * 1e3);
        assert!(rms(S) < 5e-4, "rear travel differs by {:.2} mm rms", rms(S) * 1e3);
        let loss = *info.loss.last().unwrap();
        eprintln!("rust: loss {loss:.2} in {} iterations ({:.2} s); pytorch {want_loss:.2}; travel rms difference front {:.3} mm, rear {:.3} mm",
                  info.iterations, info.seconds.iter().sum::<f64>(), rms(D) * 1e3, rms(S) * 1e3);
        eprintln!("  seconds: normal equations {:.2}, solve {:.2}, loss {:.2}; n_loc {}, band {}", info.seconds[0], info.seconds[1], info.seconds[2], Layout::new(&obj).n_loc, Layout::new(&obj).bw);
        assert!((loss - want_loss).abs() < 0.01 * want_loss, "loss {loss} vs {want_loss}");
    }
}
