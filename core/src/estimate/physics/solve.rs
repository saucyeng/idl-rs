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

use super::objective::{basis, Objective, Vars, NG, NROW};
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
        for i in 0..NG {
            out.glob[i] += dg[i];
        }
        out
    }
}

/// The normal equations J^T J x = -J^T r in band + border form.
pub struct Normal {
    /// Lower band, row-major: row i holds columns i-bw..=i (`band[i*(bw+1) + (j + bw - i)]`).
    pub band: Vec<f64>,
    /// Border: d^2 / d local d global, [n_loc][NG].
    pub border: Vec<[f64; NG]>,
    /// Globals block, [NG][NG].
    pub c: [[f64; NG]; NG],
    /// Gradient, local.
    pub g_loc: Vec<f64>,
    /// Gradient, globals.
    pub g_glob: [f64; NG],
    /// Loss at the linearisation point.
    pub loss: f64,
}

/// Assemble the normal equations at `v`.
pub fn normal_equations(obj: &Objective, lay: &Layout, v: &Vars) -> Normal {
    let (n, bw) = (lay.n_loc, lay.bw);
    let wd = bw + 1;
    let mut ne = Normal { band: vec![0.0; n * wd], border: vec![[0.0; NG]; n], c: [[0.0; NG]; NG], g_loc: vec![0.0; n], g_glob: [0.0; NG], loss: 0.0 };
    let sp = &obj.spec;
    let ncol = lay.groups.len() * 4;
    let mut a = vec![0.0; NROW * ncol];
    let mut cols = vec![0usize; ncol];
    for t in 0..sp.n {
        let s = obj.sample(v, t);
        let sj = obj.sample_residuals(&s, &v.glob, t, true);
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
            for gq in 0..NG {
                let mut h = 0.0;
                for r in 0..NROW {
                    h += a[r * ncol + p] * sj.jg[r][gq];
                }
                ne.border[cp][gq] += h;
            }
        }
        for gp in 0..NG {
            for r in 0..NROW {
                ne.g_glob[gp] += sj.jg[r][gp] * sj.r[r];
            }
            for gq in 0..NG {
                let mut h = 0.0;
                for r in 0..NROW {
                    h += sj.jg[r][gp] * sj.jg[r][gq];
                }
                ne.c[gp][gq] += h;
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
    // global priors
    let sig = obj.glob_sigma();
    for i in 0..NG {
        let gw = 1.0 / sig[i];
        let gp = (v.glob[i] - obj.glob_centre[i]) * gw;
        ne.c[i][i] += gw * gw;
        ne.g_glob[i] += gw * gp;
        ne.loss += 0.5 * gp * gp;
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
        let mut s = x[i];
        for k in lo..i {
            s -= l[i * wd + (k + bw - i)] * x[k];
        }
        x[i] = s / l[i * wd + bw];
    }
    for i in (0..n).rev() {
        x[i] /= l[i * wd + bw];
        let xi = x[i];
        let lo = i.saturating_sub(bw);
        for k in lo..i {
            x[k] -= l[i * wd + (k + bw - i)] * xi;
        }
    }
}

/// Why a solve failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The damped normal matrix was not positive definite at this row (the
    /// problem is degenerate there, e.g. unobserved latents with no prior).
    NotPositiveDefinite {
        /// Row of the band where the factorisation failed.
        row: usize,
    },
}

/// (H + lam diag(H)) x = -g for the band + border system.
pub fn solve_damped(ne: &Normal, lay: &Layout, lam: f64) -> Result<(Vec<f64>, [f64; NG]), SolveError> {
    let (n, bw) = (lay.n_loc, lay.bw);
    let wd = bw + 1;
    let mut l = ne.band.clone();
    for i in 0..n {
        l[i * wd + bw] = l[i * wd + bw] * (1.0 + lam) + 1e-9;
    }
    band_cholesky(&mut l, n, bw)?;
    let mut ar: Vec<f64> = ne.g_loc.iter().map(|g| -g).collect();
    band_solve(&l, n, bw, &mut ar);
    let mut ab: Vec<Vec<f64>> = (0..NG).map(|j| ne.border.iter().map(|r| r[j]).collect()).collect();
    for col in ab.iter_mut() {
        band_solve(&l, n, bw, col);
    }
    // Schur complement for the globals
    let mut sch = nalgebra::DMatrix::<f64>::zeros(NG, NG);
    let mut rhs = nalgebra::DVector::<f64>::zeros(NG);
    for i in 0..NG {
        for j in 0..NG {
            let bab: f64 = (0..n).map(|k| ne.border[k][i] * ab[j][k]).sum();
            sch[(i, j)] = ne.c[i][j] - bab + if i == j { lam * ne.c[i][i] + 1e-9 } else { 0.0 };
        }
        let bar: f64 = (0..n).map(|k| ne.border[k][i] * ar[k]).sum();
        rhs[i] = -ne.g_glob[i] - bar;
    }
    let xg = sch.cholesky().ok_or(SolveError::NotPositiveDefinite { row: n })?.solve(&rhs);
    let xg: [f64; NG] = std::array::from_fn(|i| xg[i]);
    let xl: Vec<f64> = (0..n).map(|k| ar[k] - (0..NG).map(|j| ab[j][k] * xg[j]).sum::<f64>()).collect();
    Ok((xl, xg))
}

/// What a solve did.
#[derive(Debug, Clone)]
pub struct SolveInfo {
    /// Loss after each accepted iteration (first entry: the start).
    pub loss: Vec<f64>,
    /// Iterations run.
    pub iterations: usize,
    /// Seconds assembling normal equations, solving, and evaluating the loss.
    pub seconds: [f64; 3],
}

/// Levenberg-Marquardt from `v` until an iteration gains less than `tol` in loss.
pub fn gauss_newton(obj: &Objective, mut v: Vars, iters: usize, tol: f64) -> Result<(Vars, SolveInfo), SolveError> {
    let lay = Layout::new(obj);
    let mut lam = 1e-10;
    let mut loss = obj.loss(&v);
    let mut info = SolveInfo { loss: vec![loss], iterations: 0, seconds: [0.0; 3] };
    for it in 0..iters {
        let t0 = std::time::Instant::now();
        let ne = normal_equations(obj, &lay, &v);
        info.seconds[0] += t0.elapsed().as_secs_f64();
        let mut new;
        loop {
            let t1 = std::time::Instant::now();
            let (xl, xg) = solve_damped(&ne, &lay, lam)?;
            info.seconds[1] += t1.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            let cand = lay.apply(&v, &xl, &xg);
            new = obj.loss(&cand);
            info.seconds[2] += t2.elapsed().as_secs_f64();
            if new < loss {
                v = cand;
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
            break;
        }
    }
    Ok((v, info))
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
        let mut centre = [0.0; NG];
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
