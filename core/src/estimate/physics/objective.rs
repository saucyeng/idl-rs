//! The objective: whitened residuals whose half sum of squares is a negative
//! log-likelihood. A port of `ml/idlml/factors/objective.py` and the latent
//! layout of `ml/idlml/solve/batch.py`.
//!
//! Latents are uniform cubic B-splines at several knot spacings (`Spec`);
//! the frame's angular rate is a full-rate spline that the frame gyro
//! measures (its derivative is the angular acceleration). Per sample, the
//! residual rows ([`NROW`]) are:
//!
//! | rows | term |
//! |---|---|
//! | 0..3 | frame gyro vs the frame-rate latent (through the frame pod's flex) |
//! | 3..9 | fork pod gyro, accel vs the forward model (clipped samples masked) |
//! | 9..15 | rear pod gyro, accel |
//! | 15..19 | walls: travel below 0 or past its stop, front and rear |
//! | 19..21 | fork bending prior (zero-mean) |
//! | 21..23 | rear-member flex prior |
//! | 23..29 | frame-pod flex prior (rotation, translation) |
//! | 29..31 | airtime anchor: travel near 0 in free fall |
//!
//! plus a second-difference smoothness prior on every latent spline row and
//! Gaussian priors on the [`NG`] global biases.

use super::bike::{Bike, B_FA, B_LAT, D, F_ROT, F_TRANS, NL, R_ROLL, R_YAW, S};
use super::la::V3;
use super::readings::{pod_readings, readings_and_jacobian, Sample, NR, NZ};
use super::setup::{geometry, with_geometry, Geometry, GEOMETRY_SIGMA};
use num_dual::{Dual64, HyperDual};

/// Residual rows per sample (see the module table).
pub const NROW: usize = 31;
/// Bias globals: frame gyro bias, fork gyro bias, fork accel bias, rear gyro
/// bias, rear accel bias (3 each; rad/s and m/s^2), in reading order.
pub const NG: usize = 15;
/// Geometry globals when fitting the bike ([`super::setup::Geometry`]).
pub const NP: usize = 5;
/// Most globals an objective has (biases + geometry).
pub const NGMAX: usize = NG + NP;
/// Central-difference steps for the geometry derivatives: m, m, m, m, scale.
const GEOMETRY_STEP: [f64; NP] = [1e-5, 1e-5, 1e-5, 1e-5, 1e-6];

/// Noise levels and prior widths (`Weights` in the PyTorch reference, with
/// the values `ride_fit.py` uses on real rides).
#[derive(Clone, Debug)]
pub struct Weights {
    /// Gyro residual sigma, rad/s.
    pub gyro_sigma: f64,
    /// Pod accelerometer noise sigma, m/s^2.
    pub accel_sigma: f64,
    /// The frame's specific-force noise, which lands on every pod, m/s^2.
    pub frame_accel_sigma: f64,
    /// Front and rear end stops, m.
    pub travel_max: [f64; 2],
    /// Metres past an end stop that cost one sigma, m.
    pub wall_sigma: f64,
    /// Fork bending prior, rad.
    pub bend_sigma: f64,
    /// Rear-member flex prior, rad.
    pub rear_flex_sigma: f64,
    /// Frame-pod flex prior, rotation, rad.
    pub frame_flex_rot_sigma: f64,
    /// Frame-pod flex prior, translation, m.
    pub frame_flex_trans_sigma: f64,
    /// Smoothness weight on spline second differences (coefficient units).
    pub smooth: f64,
    /// Gyro bias prior sigma, rad/s.
    pub gyro_bias_sigma: f64,
    /// Accelerometer bias prior sigma, m/s^2.
    pub accel_bias_sigma: f64,
    /// Airtime anchor sigma, m.
    pub air_sigma: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            gyro_sigma: 0.05,
            accel_sigma: 1.0,
            frame_accel_sigma: 1.0,
            travel_max: [0.17, 0.16],
            wall_sigma: 0.002,
            bend_sigma: 0.01,
            rear_flex_sigma: 0.01,
            frame_flex_rot_sigma: 0.002,
            frame_flex_trans_sigma: 0.0005,
            smooth: 1e-3,
            gyro_bias_sigma: 0.002,
            accel_bias_sigma: 0.5,
            air_sigma: 0.005,
        }
    }
}

/// How the latents are laid out in time.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Samples in the window.
    pub n: usize,
    /// Sample spacing, s.
    pub dt: f64,
    /// Spline levels: (knot spacing in samples, latent indices).
    pub levels: Vec<(usize, Vec<usize>)>,
    /// Latent value at zero coefficients (SI units).
    pub init: [f64; NL],
    /// Latent units per coefficient unit.
    pub scale: [f64; NL],
}

impl Spec {
    /// The layout the PyTorch reference uses: bending and rear flex every
    /// sample, travel and frame flex every 2, steer every 4.
    pub fn standard(n: usize, dt: f64) -> Self {
        Spec {
            n,
            dt,
            levels: vec![(1, vec![B_FA, B_LAT, R_ROLL, R_YAW]), (2, vec![D, S, 7, 8, 9, 10, 11, 12]), (4, vec![1])],
            init: [0.045, 0.0, 0.0, 0.0, 0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            scale: [0.01, 0.1, 0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
    }

    /// Coefficients per spline row at knot spacing `m`.
    pub fn n_knots(&self, m: usize) -> usize {
        self.n.div_ceil(m) + 3
    }
}

/// Uniform cubic B-spline weights at offset `j` of `m` within a knot span:
/// value, first and second derivative (per knot spacing), for the 4
/// coefficients base..base+4.
pub fn basis(j: usize, m: usize) -> [[f64; 4]; 3] {
    let u = j as f64 / m as f64;
    let s = 1.0 / 6.0;
    [
        [(1.0 - u).powi(3) * s, (3.0 * u * u * u - 6.0 * u * u + 4.0) * s, (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) * s, u * u * u * s],
        [-3.0 * (1.0 - u).powi(2) * s, (9.0 * u * u - 12.0 * u) * s, (-9.0 * u * u + 6.0 * u + 3.0) * s, 3.0 * u * u * s],
        [6.0 * (1.0 - u) * s, (18.0 * u - 12.0) * s, (-18.0 * u + 6.0) * s, 6.0 * u * s],
    ]
}

/// One window's measurements, each in its own sensor's axes.
#[derive(Clone, Debug)]
pub struct Window {
    /// Sample rate, Hz.
    pub rate: f64,
    /// Frame pod gyro, rad/s.
    pub frame_gyro: Vec<[f64; 3]>,
    /// Frame pod specific force, m/s^2.
    pub frame_accel: Vec<[f64; 3]>,
    /// Fork pod gyro, rad/s.
    pub fork_gyro: Vec<[f64; 3]>,
    /// Fork pod specific force, m/s^2.
    pub fork_accel: Vec<[f64; 3]>,
    /// Rear pod gyro, rad/s.
    pub rear_gyro: Vec<[f64; 3]>,
    /// Rear pod specific force, m/s^2.
    pub rear_accel: Vec<[f64; 3]>,
    /// Fork pod sample usable (false: clipped).
    pub mask_fork: Vec<bool>,
    /// Rear pod sample usable.
    pub mask_rear: Vec<bool>,
    /// Airborne samples, where travel is anchored near 0 (None: no anchor).
    pub air: Option<Vec<bool>>,
}

/// The unknowns: spline coefficients (per level, per row) and the globals.
#[derive(Clone, Debug)]
pub struct Vars {
    /// `lvl[i][row][k]`: level i's row (one latent), coefficient k, in coefficient units.
    pub lvl: Vec<Vec<Vec<f64>>>,
    /// The frame-rate spline's coefficients per axis, rad/s (full rate).
    pub frame: [Vec<f64>; 3],
    /// Globals: the [`NG`] biases, then (when fitting the bike) the [`NP`] geometry.
    pub glob: Vec<f64>,
}

/// A window's objective: bike, measurements, layout, weights and priors.
pub struct Objective {
    /// Bike geometry (with `fit`, the starting point and prior centre of the fitted part).
    pub bike: Bike<f64>,
    /// Measurements.
    pub win: Window,
    /// Latent layout.
    pub spec: Spec,
    /// Weights and priors.
    pub w: Weights,
    /// Prior centres of the globals (biases: e.g. still-recording gyro biases).
    pub glob_centre: Vec<f64>,
    /// Whether the bike's geometry is estimated too (globals [`NG`]..[`NGMAX`]).
    pub fit: bool,
}

/// The bike lifted for one linearisation point: values, values + Jacobian,
/// and, when fitting, the geometry nudged each way for central differences.
pub struct Ctx {
    bike_h: Bike<HyperDual<f64>>,
    bike_dh: Bike<HyperDual<Dual64>>,
    fd: Vec<(Bike<HyperDual<f64>>, Bike<HyperDual<f64>>, f64)>,
}

/// Per-sample residuals and their Jacobians.
pub struct SampleJac {
    /// Residuals.
    pub r: [f64; NROW],
    /// d r / d local inputs (q, q', q'', w0, a0).
    pub jz: [[f64; NZ]; NROW],
    /// d r / d globals (first [`Objective::n_glob`] columns used).
    pub jg: [[f64; NGMAX]; NROW],
}

impl Objective {
    /// An objective over `win` with `bike` frozen; `bias_centre`: the biases' prior centres.
    pub fn new(bike: Bike<f64>, win: Window, spec: Spec, w: Weights, bias_centre: [f64; NG]) -> Self {
        Objective { bike, win, spec, w, glob_centre: bias_centre.to_vec(), fit: false }
    }

    /// As [`Self::new`], also estimating the bike's geometry, with a prior
    /// ([`GEOMETRY_SIGMA`]) centred on `bike`'s.
    pub fn fitting(bike: Bike<f64>, win: Window, spec: Spec, w: Weights, bias_centre: [f64; NG]) -> Self {
        let mut c = bias_centre.to_vec();
        c.extend(geometry(&bike));
        Objective { bike, win, spec, w, glob_centre: c, fit: true }
    }

    /// Number of globals: [`NG`], or [`NGMAX`] when fitting.
    pub fn n_glob(&self) -> usize {
        if self.fit {
            NGMAX
        } else {
            NG
        }
    }

    /// Prior sigma of each global.
    pub fn glob_sigma(&self) -> Vec<f64> {
        (0..self.n_glob())
            .map(|i| match i {
                i if i >= NG => GEOMETRY_SIGMA[i - NG],
                6..=8 | 12..=14 => self.w.accel_bias_sigma,
                _ => self.w.gyro_bias_sigma,
            })
            .collect()
    }

    /// The bike at the globals' geometry (the frozen bike when not fitting).
    pub fn bike_at(&self, glob: &[f64]) -> Bike<f64> {
        if self.fit {
            with_geometry(&self.bike, &std::array::from_fn(|i| glob[NG + i]))
        } else {
            self.bike.clone()
        }
    }

    /// The lifted bikes for linearising at `glob` (`jac`: with the geometry nudges).
    pub fn ctx(&self, glob: &[f64], jac: bool) -> Ctx {
        let b = self.bike_at(glob);
        let fd = if self.fit && jac {
            let g: Geometry = geometry(&b);
            (0..NP)
                .map(|p| {
                    let (mut lo, mut hi) = (g, g);
                    lo[p] -= GEOMETRY_STEP[p];
                    hi[p] += GEOMETRY_STEP[p];
                    (with_geometry(&b, &hi).map(HyperDual::from_re), with_geometry(&b, &lo).map(HyperDual::from_re), 2.0 * GEOMETRY_STEP[p])
                })
                .collect()
        } else {
            Vec::new()
        };
        Ctx { bike_h: b.map(HyperDual::from_re), bike_dh: b.map(|x| HyperDual::from_re(Dual64::from(x))), fd }
    }

    /// Starting values: latents at `spec.init`, the frame rate through the
    /// frame gyro, biases at their prior centres.
    pub fn initial_vars(&self) -> Vars {
        let lvl = self.spec.levels.iter().map(|(m, ix)| vec![vec![0.0; self.spec.n_knots(*m)]; ix.len()]).collect();
        let g = &self.win.frame_gyro;
        let n = self.spec.n;
        let frame = std::array::from_fn(|x| {
            let mut c = Vec::with_capacity(n + 3);
            c.push(g[0][x]);
            c.extend(g.iter().map(|v| v[x]));
            c.push(g[n - 1][x]);
            c.push(g[n - 1][x]);
            c
        });
        Vars { lvl, frame, glob: self.glob_centre.clone() }
    }

    /// The local inputs at sample `t`: latents and their derivatives, chassis rate and acceleration.
    pub fn sample(&self, v: &Vars, t: usize) -> Sample<f64> {
        let sp = &self.spec;
        let mut q = sp.init;
        let mut qd = [0.0; NL];
        let mut qdd = [0.0; NL];
        for (li, (m, ix)) in sp.levels.iter().enumerate() {
            let (j, base) = (t % m, t / m);
            let b = basis(j, *m);
            let h = *m as f64 * sp.dt;
            for (row, &lat) in ix.iter().enumerate() {
                let c = &v.lvl[li][row][base..base + 4];
                let dot = |w: &[f64; 4]| w[0] * c[0] + w[1] * c[1] + w[2] * c[2] + w[3] * c[3];
                q[lat] += sp.scale[lat] * dot(&b[0]);
                qd[lat] += sp.scale[lat] * dot(&b[1]) / h;
                qdd[lat] += sp.scale[lat] * dot(&b[2]) / (h * h);
            }
        }
        let b = basis(0, 1);
        let mut w0 = [0.0; 3];
        let mut a0 = [0.0; 3];
        for x in 0..3 {
            let c = &v.frame[x][t..t + 4];
            w0[x] = b[0].iter().zip(c).map(|(a, b)| a * b).sum();
            a0[x] = b[1].iter().zip(c).map(|(a, b)| a * b).sum::<f64>() / sp.dt;
        }
        Sample { q, qd, qdd, w0: V3(w0), a0: V3(a0), f_meas: V3(self.win.frame_accel[t]) }
    }

    /// Residuals at sample `t` and, if `jac`, their Jacobians.
    pub fn sample_residuals(&self, ctx: &Ctx, s: &Sample<f64>, glob: &[f64], t: usize, jac: bool) -> SampleJac {
        let w = &self.w;
        let (val, jr) = if jac { readings_and_jacobian(&ctx.bike_h, &ctx.bike_dh, s) } else { (pod_readings(&ctx.bike_h, s), [[0.0; NZ]; NR]) };
        // geometry: central differences of the readings
        let mut jp = [[0.0; NP]; NR];
        for (p, (hi, lo, h)) in ctx.fd.iter().enumerate() {
            let (a, b) = (pod_readings(hi, s), pod_readings(lo, s));
            for r in 0..NR {
                jp[r][p] = (a[r] - b[r]) / h;
            }
        }
        let mut out = SampleJac { r: [0.0; NROW], jz: [[0.0; NZ]; NROW], jg: [[0.0; NGMAX]; NROW] };
        let acc_sigma = (w.accel_sigma * w.accel_sigma + w.frame_accel_sigma * w.frame_accel_sigma).sqrt();
        let win = &self.win;
        // readings rows: frame gyro (no mask), then each pod's gyro and accel (masked)
        let meas: [([f64; 3], usize, f64, f64); 5] = [
            (win.frame_gyro[t], 0, w.gyro_sigma, 1.0),
            (win.fork_gyro[t], 3, w.gyro_sigma, win.mask_fork[t] as u8 as f64),
            (win.fork_accel[t], 6, acc_sigma, win.mask_fork[t] as u8 as f64),
            (win.rear_gyro[t], 9, w.gyro_sigma, win.mask_rear[t] as u8 as f64),
            (win.rear_accel[t], 12, acc_sigma, win.mask_rear[t] as u8 as f64),
        ];
        for (m, row0, sig, mask) in meas {
            let k = mask / sig;
            for a in 0..3 {
                let r = row0 + a; // reading index == residual row and global index here
                out.r[r] = (val[r] + glob[r] - m[a]) * k;
                out.jg[r][r] = k;
                for p in 0..ctx.fd.len() {
                    out.jg[r][NG + p] = jp[r][p] * k;
                }
                if jac {
                    for z in 0..NZ {
                        out.jz[r][z] = jr[r][z] * k;
                    }
                }
            }
        }
        let q = &s.q;
        // walls
        for (i, (lat, top)) in [(D, w.travel_max[0]), (S, w.travel_max[1])].into_iter().enumerate() {
            let (lo, hi) = (15 + 2 * i, 16 + 2 * i);
            if q[lat] < 0.0 {
                out.r[lo] = -q[lat] / w.wall_sigma;
                out.jz[lo][lat] = -1.0 / w.wall_sigma;
            }
            if q[lat] > top {
                out.r[hi] = (q[lat] - top) / w.wall_sigma;
                out.jz[hi][lat] = 1.0 / w.wall_sigma;
            }
        }
        let mut prior = |row: usize, lat: usize, sig: f64| {
            out.r[row] = q[lat] / sig;
            out.jz[row][lat] = 1.0 / sig;
        };
        prior(19, B_FA, w.bend_sigma);
        prior(20, B_LAT, w.bend_sigma);
        prior(21, R_ROLL, w.rear_flex_sigma);
        prior(22, R_YAW, w.rear_flex_sigma);
        for a in 0..3 {
            prior(23 + a, F_ROT + a, w.frame_flex_rot_sigma);
            prior(26 + a, F_TRANS + a, w.frame_flex_trans_sigma);
        }
        if win.air.as_ref().is_some_and(|air| air[t]) {
            prior(29, D, w.air_sigma);
            prior(30, S, w.air_sigma);
        }
        out
    }

    /// The smoothness weight on coefficient second differences.
    pub fn smooth_weight(&self) -> f64 {
        (self.w.smooth * 2.0).sqrt()
    }

    /// The objective's value: half the sum of squared whitened residuals.
    pub fn loss(&self, v: &Vars) -> f64 {
        self.data_loss(v) + self.prior_loss(v)
    }

    /// Half the sum of squares of the per-sample residuals and the smoothness
    /// priors (everything but the global priors).
    pub fn data_loss(&self, v: &Vars) -> f64 {
        let ctx = self.ctx(&v.glob, false);
        let mut l = 0.0;
        for t in 0..self.spec.n {
            let s = self.sample(v, t);
            l += self.sample_residuals(&ctx, &s, &v.glob, t, false).r.iter().map(|x| x * x).sum::<f64>();
        }
        let k = self.smooth_weight();
        for lvl in &v.lvl {
            for row in lvl {
                for i in 0..row.len() - 2 {
                    let d2 = k * (row[i] - 2.0 * row[i + 1] + row[i + 2]);
                    l += d2 * d2;
                }
            }
        }
        0.5 * l
    }

    /// Half the sum of squares of the global priors.
    pub fn prior_loss(&self, v: &Vars) -> f64 {
        let sig = self.glob_sigma();
        0.5 * (0..self.n_glob()).map(|i| ((v.glob[i] - self.glob_centre[i]) / sig[i]).powi(2)).sum::<f64>()
    }
}
