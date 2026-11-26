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

use super::objective::{basis, huber2, Objective, Vars, NROW, SENSOR_ROWS};
use super::bike::NL;
use super::la::{exp, V3};
use super::forces::{NBAL, NFL, NFR};
use super::nav::NNAV;

/// What a group of coefficients is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// A latent row of this spline level.
    Latent(usize),
    /// The frame-rate spline, axis `row`.
    Frame,
    /// The attitude spline's perturbation, axis `row` (stepped as R <- R Exp(d)).
    Att,
    /// The velocity spline, nav axis `row`.
    Vel,
    /// A force latent (`row`: its index).
    Force,
}

/// One spline row's place in the unknown vector.
struct Group {
    kind: Kind,
    row: usize,
    m: usize,
    scale: f64,
    /// Smoothness weight, times the objective's.
    smooth: f64,
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
                groups.push(Group { kind: Kind::Latent(li), row, m: *m, scale: sp.scale[lat], smooth: 1.0, z: [Some(lat), Some(NL + lat), Some(2 * NL + lat)], pos: vec![] });
            }
        }
        for x in 0..3 {
            groups.push(Group { kind: Kind::Frame, row: x, m: 1, scale: 1.0, smooth: 1.0, z: [Some(3 * NL + x), Some(3 * NL + 3 + x), None], pos: vec![] });
        }
        if let Some((ns, _)) = obj.nav() {
            for (kind, m) in [(Kind::Att, ns.m_att), (Kind::Vel, ns.m_vel)] {
                for x in 0..3 {
                    groups.push(Group { kind, row: x, m, scale: 1.0, smooth: 1.0, z: [None; 3], pos: vec![] });
                }
            }
        }
        if let Some((m, _)) = obj.force_on() {
            for i in 0..NFL {
                groups.push(Group { kind: Kind::Force, row: i, m: m.spec.m[i], scale: m.spec.scale[i], smooth: m.spec.smooth[i].sqrt(), z: [None; 3], pos: vec![] });
            }
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
        match g.kind {
            Kind::Latent(li) => &v.lvl[li][g.row],
            Kind::Frame => &v.frame[g.row],
            Kind::Vel => &v.vel[g.row],
            Kind::Force => &v.frc[g.row],
            Kind::Att => &[],
        }
    }

    /// `v` stepped by `dl` (local, in layout order) and `dg` (globals).
    pub fn apply(&self, v: &Vars, dl: &[f64], dg: &[f64]) -> Vars {
        let mut out = v.clone();
        let mut d_att = vec![[0.0; 3]; v.att.len()];
        for g in &self.groups {
            let c = match g.kind {
                Kind::Latent(li) => &mut out.lvl[li][g.row],
                Kind::Frame => &mut out.frame[g.row],
                Kind::Vel => &mut out.vel[g.row],
                Kind::Force => &mut out.frc[g.row],
                Kind::Att => {
                    for (k, &p) in g.pos.iter().enumerate() {
                        d_att[k][g.row] = dl[p];
                    }
                    continue;
                }
            };
            for (k, &p) in g.pos.iter().enumerate() {
                c[k] += dl[p];
            }
        }
        for (r, d) in out.att.iter_mut().zip(d_att) {
            *r = *r * exp(V3::from_f64(d));
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
    let nav_rows = if obj.nav().is_some() { NNAV } else { 0 };
    let f0 = NROW + nav_rows; // first force row
    let nt = f0 + if obj.force_on().is_some() { NFR } else { 0 };
    let (fg0, nb) = (obj.force_glob0(), ng.min(super::objective::NGMAX));
    let k_force = obj.forces.as_ref().map_or(0.0, |m| m.w.huber);
    let mut a = vec![0.0; nt * ncol];
    let mut res = vec![0.0; nt];
    let mut jg = vec![0.0; nt * ng];
    let mut cols = vec![0usize; ncol];
    for t in 0..sp.n {
        let s = obj.sample(v, t);
        let sj = obj.sample_residuals(&ctx, &s, &v.glob, t, true);
        let nj = obj.nav_residuals(v, &s, t, true);
        let fj = obj.force_residuals(&ctx, v, &s, t, true);
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
        res[..NROW].copy_from_slice(&sj.r);
        jg.iter_mut().for_each(|x| *x = 0.0);
        for r in 0..NROW {
            jg[r * ng..r * ng + nb].copy_from_slice(&sj.jg[r][..nb]);
        }
        if let Some(nj) = &nj {
            // navigation rows: attitude columns directly, velocity and frame rate through their basis
            res[NROW..f0].copy_from_slice(&nj.r);
            for (gi, g) in lay.groups.iter().enumerate() {
                let b = basis(t % g.m, g.m);
                let h = g.m as f64 * sp.dt;
                for i in 0..4 {
                    for r in 0..NNAV {
                        a[(NROW + r) * ncol + gi * 4 + i] = match g.kind {
                            Kind::Att => nj.j_att[r][3 * i + g.row],
                            Kind::Vel => b[0][i] * nj.j_v[r][g.row] + b[1][i] / h * nj.j_v[r][3 + g.row],
                            Kind::Frame => b[0][i] * nj.j_w0[r][g.row],
                            Kind::Latent(_) | Kind::Force => 0.0,
                        };
                    }
                }
            }
        }
        if let Some(fj) = &fj {
            // force rows: core latents and the frame rate through their basis like the pod rows, the
            // force latents through theirs, the parameters as globals
            res[f0..].copy_from_slice(&fj.r);
            for (gi, g) in lay.groups.iter().enumerate() {
                let (j, base) = (t % g.m, t / g.m);
                let _ = base;
                let b = basis(j, g.m);
                let h = g.m as f64 * sp.dt;
                let wz = [g.scale, g.scale / h, g.scale / (h * h)];
                for i in 0..4 {
                    for d in 0..3 {
                        let wgt = b[d][i] * wz[d];
                        let col = match (g.kind, g.z[d]) {
                            (Kind::Force, _) => Some((true, d * NFL + g.row)),
                            (Kind::Latent(_) | Kind::Frame, Some(z)) => Some((false, z)),
                            _ => None,
                        };
                        if let Some((frc, z)) = col {
                            for r in 0..NFR {
                                let jr = if frc { fj.j_frc[r][z] } else { fj.j_core[r][z] };
                                a[(f0 + r) * ncol + gi * 4 + i] += jr * wgt;
                            }
                        }
                    }
                }
            }
            for r in 0..NFR {
                jg[(f0 + r) * ng + fg0..(f0 + r) * ng + fg0 + super::forces::NFP].copy_from_slice(&fj.j_par[r]);
            }
        }
        // Huber on the sensor rows (iteratively reweighted): a row beyond k sigmas is scaled by
        // sqrt(k/|r|), so its gradient is the Huber loss's and its curvature the IRLS weight's
        let k = obj.w.huber;
        for r in 0..nt {
            let k = if r < SENSOR_ROWS { k } else if r >= f0 && r < f0 + NBAL { k_force } else { 0.0 };
            if k > 0.0 && res[r].abs() > k {
                ne.loss += 0.5 * huber2(res[r], k);
                let s = (k / res[r].abs()).sqrt();
                res[r] *= s;
                a[r * ncol..(r + 1) * ncol].iter_mut().for_each(|x| *x *= s);
                jg[r * ng..(r + 1) * ng].iter_mut().for_each(|x| *x *= s);
            } else {
                ne.loss += 0.5 * res[r] * res[r];
            }
        }
        for p in 0..ncol {
            let cp = cols[p];
            let mut gp = 0.0;
            for r in 0..nt {
                gp += a[r * ncol + p] * res[r];
            }
            ne.g_loc[cp] += gp;
            for q in 0..ncol {
                let cq = cols[q];
                if cq > cp {
                    continue;
                }
                let mut h = 0.0;
                for r in 0..nt {
                    h += a[r * ncol + p] * a[r * ncol + q];
                }
                ne.band[cp * wd + (cq + bw - cp)] += h;
            }
            for gq in 0..ng {
                let mut h = 0.0;
                for r in 0..nt {
                    h += a[r * ncol + p] * jg[r * ng + gq];
                }
                ne.border[cp * ng + gq] += h;
            }
        }
        for gp in 0..ng {
            for r in 0..nt {
                ne.g_glob[gp] += jg[r * ng + gp] * res[r];
            }
            for gq in 0..ng {
                let mut h = 0.0;
                for r in 0..nt {
                    h += jg[r * ng + gp] * jg[r * ng + gq];
                }
                ne.c[gp * ng + gq] += h;
            }
        }
    }
    // smoothness priors: k (c_i - 2 c_i+1 + c_i+2) on each latent row
    let k0 = obj.smooth_weight();
    for g in lay.groups.iter().filter(|g| matches!(g.kind, Kind::Latent(_) | Kind::Force)) {
        let k = k0 * g.smooth;
        let pw = [k, -2.0 * k, k];
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
    /// Stopped before this window started ([`super::run::SolveProgress::cancel`]).
    Cancelled,
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
    let (mut lam, mut nu) = (1e-10, 2.0);
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
            let xls: Vec<Vec<f64>> = fs.iter().map(|f| back_substitute(f, &xg)).collect();
            // the quadratic model's predicted loss drop, about -g.x / 2 for a small damping
            let pred = -0.5
                * (nes.iter().zip(&xls).map(|(ne, xl)| ne.g_loc.iter().zip(xl).map(|(g, x)| g * x).sum::<f64>()).sum::<f64>()
                    + nes.iter().map(|ne| ne.g_glob.iter().zip(&xg).map(|(g, x)| g * x).sum::<f64>()).sum::<f64>());
            let cand: Vec<Vars> = vs.iter().zip(&lays).zip(&xls).map(|((v, lay), xl)| lay.apply(v, xl, &xg)).collect();
            info.seconds[1] += t1.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            new = total(&cand);
            info.seconds[2] += t2.elapsed().as_secs_f64();
            // Nielsen's damping update (as in Madsen, Nielsen & Tingleff, "Methods for non-linear
            // least squares problems", 2004): a ÷10 after every success overshot, and climbing back
            // cost up to five rejected factorisations per iteration
            if new < loss {
                let rho = (loss - new) / pred.max(f64::MIN_POSITIVE);
                vs = cand;
                lam = (lam * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3))).max(1e-12);
                nu = 2.0;
                break;
            }
            // a step that promised less than the tolerance and did not deliver: converged, not
            // worth raising the damping to the cap one factorisation at a time
            if pred < tol {
                lam = f64::INFINITY;
                break;
            }
            lam *= nu;
            nu *= 2.0;
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

/// Force latent values per sample, [n][NFL], SI units (empty with the force model off).
pub fn force_latents(obj: &Objective, v: &Vars) -> Vec<[f64; NFL]> {
    (0..obj.spec.n).filter_map(|t| obj.force_inputs(v, t).map(|f| f[0])).collect()
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
            nav: None,
            forces: None,
        };
        let n = win.frame_gyro.len();
        let mut centre = [0.0; crate::estimate::physics::objective::NG];
        for (i, key) in ["frame_bias", "fork.gyro_bias", "fork.accel_bias", "rear.gyro_bias", "rear.accel_bias"].iter().enumerate() {
            let c = &doc["bias_prior"][key];
            for a in 0..3 {
                centre[3 * i + a] = c.as_f64().unwrap_or_else(|| c[a].as_f64().unwrap());
            }
        }
        // the PyTorch solver's problem: plain least squares, no fork torsion (both this port's additions)
        let w = Weights { huber: 0.0, fork_twist_sigma: 1e-9, ..Weights::default() };
        let obj = Objective::new(bike, win, Spec::standard(n, 1.0 / 400.0), w, centre);

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

    #[test]
    fn solve_with_navigation_recovers_a_known_attitude_from_gravity_and_gps() {
        // Arrange: 2 s of invented frame motion (turning, rolling, speeding up) with every
        // sensor exactly consistent; the pods read what the bike at its initial latents would,
        // GPS gives velocity at 10 Hz, and the start attitude is off by 2 deg of tilt and 9 deg of yaw
        use crate::estimate::physics::la::{exp, V3};
        use crate::estimate::physics::nav::{GpsFix, NavSpec, NavStart, NavWindow, G_NAV};
        use crate::estimate::physics::readings::{pod_readings, Sample};
        use crate::estimate::physics::so3spline::{log, rotation};
        use num_dual::HyperDual;
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../tests/fixtures/physics_window.json")).unwrap();
        let bike = fixture::bike(&doc["bike"]);
        let (n, dt) = (800, 1.0 / 400.0);
        let tau = std::f64::consts::TAU;
        let rate = |t: f64| V3([0.8 * (tau * 0.6 * t).sin(), 0.4 * (tau * 0.9 * t + 0.5).sin(), 0.5 + 0.3 * (tau * 0.4 * t).cos()]);
        let rate_d = |t: f64| V3([0.8 * tau * 0.6 * (tau * 0.6 * t).cos(), 0.4 * tau * 0.9 * (tau * 0.9 * t + 0.5).cos(), -0.3 * tau * 0.4 * (tau * 0.4 * t).sin()]);
        let vel = |t: f64| [6.0 + 2.0 * (tau * 0.5 * t).sin(), 3.0 * (tau * 0.35 * t).cos(), 0.4 * (tau * 1.1 * t).sin()];
        let acc = |t: f64| [2.0 * tau * 0.5 * (tau * 0.5 * t).cos(), -3.0 * tau * 0.35 * (tau * 0.35 * t).sin(), 0.4 * tau * 1.1 * (tau * 1.1 * t).cos()];
        let mut truth = Vec::with_capacity(n);
        let mut r = exp(V3::from_f64([0.1, -0.05, 0.7]));
        for i in 0..n {
            truth.push(r);
            for k in 0..20 {
                let tm = (i as f64 + (k as f64 + 0.5) / 20.0) * dt;
                r = r * exp(rate(tm).scale(dt / 20.0));
            }
        }
        let spec0 = Spec::standard(n, dt);
        let bike_h = bike.map(HyperDual::from_re);
        let mut win = Window {
            rate: 400.0,
            frame_gyro: vec![],
            frame_accel: vec![],
            fork_gyro: vec![],
            fork_accel: vec![],
            rear_gyro: vec![],
            rear_accel: vec![],
            mask_fork: vec![true; n],
            mask_rear: vec![true; n],
            air: None,
            nav: None,
            forces: None,
        };
        let mut gps = vec![None; n];
        for i in 0..n {
            let t = i as f64 * dt;
            let a = acc(t);
            let f = truth[i].t() * V3(std::array::from_fn(|k| a[k] - G_NAV[k]));
            let s = Sample { q: spec0.init, qd: [0.0; NL], qdd: [0.0; NL], w0: rate(t), a0: rate_d(t), f_meas: f };
            let p = pod_readings(&bike_h, &s);
            win.frame_gyro.push(rate(t).0);
            win.frame_accel.push(f.0);
            win.fork_gyro.push([p[3], p[4], p[5]]);
            win.fork_accel.push([p[6], p[7], p[8]]);
            win.rear_gyro.push([p[9], p[10], p[11]]);
            win.rear_accel.push([p[12], p[13], p[14]]);
            if i % 40 == 0 {
                gps[i] = Some(GpsFix { v: vel(t), up: true });
            }
        }
        let v0 = vel(0.0);
        let start = NavStart { att: (truth[0] * exp(V3::from_f64([0.03, -0.02, 0.15]))).0, att_sigma: [0.2, 0.2, 0.5], vel: [v0[0] + 0.3, v0[1] - 0.2, v0[2] + 0.1], vel_sigma: 1.0 };
        win.nav = Some(NavWindow { gps, start });
        let mut spec = spec0;
        spec.nav = Some(NavSpec::default());
        let obj = Objective::new(bike, win, spec, Weights::default(), [0.0; crate::estimate::physics::objective::NG]);

        // Act
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), 30, 1e-6).unwrap();

        // Assert: tilt within 0.1 deg and yaw within 0.5 deg away from the window's ends; travel undisturbed
        let (mut tilt, mut yaw) = (0.0f64, 0.0f64);
        for t in 20..n - 20 {
            let s = obj.sample(&v, t);
            let ns = obj.nav_sample(&v, &s, t).unwrap();
            let e = log(rotation(&ns.ctrl, ns.u) * truth[t].t()); // nav-frame error
            tilt = tilt.max((e.0[0] * e.0[0] + e.0[1] * e.0[1]).sqrt().to_degrees());
            yaw = yaw.max(e.0[2].abs().to_degrees());
        }
        let q = latents(&obj, &v);
        let travel = q.iter().map(|q| (q[D] - 0.045).abs().max((q[S] - 0.045).abs())).fold(0.0, f64::max);
        eprintln!("navigation: {} iterations, loss {:.3e}; max tilt error {tilt:.4} deg, yaw {yaw:.4} deg; travel moved {:.3} mm",
                  info.iterations, info.loss.last().unwrap(), travel * 1e3);
        assert!(tilt < 0.1, "tilt error {tilt} deg");
        assert!(yaw < 0.5, "yaw error {yaw} deg");
        assert!(travel < 1e-3, "travel moved {} mm", travel * 1e3);
    }
}
