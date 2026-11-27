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
//! With navigation on (`Spec::nav` and `Window::nav`), the [`super::nav`]
//! rows follow these: attitude and velocity latents that the frame pod's
//! specific force, the frame-rate latent and GPS constrain.
//!
//! plus a second-difference smoothness prior on every latent spline row and
//! Gaussian priors on the [`NG`] global biases.

use super::bike::{Bike, B_FA, B_LAT, D, F_ROT, F_TRANS, F_TWIST, NL, R_ROLL, R_YAW, S};
use super::la::{exp, M3, V3};
use super::forces::{force_rows, kin, rows as force_rows_at, ForceIn, ForceJac, ForceMeas, ForceModel, ForceWindow, Forces, GroundMeas as ForceGround, NBAL, NFL, NFP};
use super::nav::{nav_rows, NavJac, NavSample, NavSpec, NavWeights, NavWindow};
use super::readings::{pod_readings, readings_and_jacobian, Sample, NR, NZ};
use super::setup::{geometry, with_geometry, Geometry, GEOMETRY_SIGMA};
use num_dual::{Dual64, HyperDual};

/// Residual rows per sample (see the module table).
pub const NROW: usize = 32;
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
    /// Fork torsional vibration prior, rad.
    pub fork_twist_sigma: f64,
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
    /// The navigation rows' noise levels.
    pub nav: NavWeights,
    /// Huber threshold on the sensor rows, whitened (sigmas); 0: plain least squares. A reading
    /// further out than this counts linearly, so one pod's unexplained moment (a big hit) cannot
    /// drag the shared frame motion, and through it the other pods' latents.
    pub huber: f64,
}

/// Rows of [`SampleJac::r`] that are sensor readings ([`NR`]): the rows Huber weighting applies to.
pub const SENSOR_ROWS: usize = NR;

/// Twice the Huber loss of a whitened residual `x` at threshold `k` (`x^2` inside `k`,
/// `2k|x| - k^2` beyond), so that half the sum stays the objective; `k <= 0` is plain `x^2`.
pub fn huber2(x: f64, k: f64) -> f64 {
    let a = x.abs();
    if k <= 0.0 || a <= k { x * x } else { 2.0 * k * a - k * k }
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
            fork_twist_sigma: 0.01,
            frame_flex_rot_sigma: 0.002,
            frame_flex_trans_sigma: 0.0005,
            smooth: 1e-3,
            gyro_bias_sigma: 0.002,
            accel_bias_sigma: 0.5,
            air_sigma: 0.005,
            nav: NavWeights::default(),
            // 2026-11-25, 11/21 lap-2 bottom-out: rear travel min -47 -> +3 mm, front peak 185 -> 175 mm;
            // an exact-model bottom-out still solves to 170.0
            huber: 3.0,
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
    /// Attitude and velocity latents (None: the attitude-free model).
    pub nav: Option<NavSpec>,
}

impl Spec {
    /// The layout the PyTorch reference uses: bending and rear flex every
    /// sample, travel and frame flex every 2, steer every 4.
    pub fn standard(n: usize, dt: f64) -> Self {
        Spec {
            n,
            dt,
            levels: vec![(1, vec![B_FA, B_LAT, R_ROLL, R_YAW, F_TWIST]), (2, vec![D, S, 7, 8, 9, 10, 11, 12]), (4, vec![1])],
            init: [0.045, 0.0, 0.0, 0.0, 0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            scale: [0.01, 0.1, 0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            nav: None,
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
    /// GPS fixes and the starting state, for the navigation rows (None: off).
    pub nav: Option<NavWindow>,
    /// The force rows' extra measurements (None: off).
    pub forces: Option<ForceWindow>,
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
    /// Attitude spline controls, IMU0 -> nav (empty with navigation off).
    pub att: Vec<M3<f64>>,
    /// Velocity spline coefficients per nav axis, m/s (empty with navigation off).
    pub vel: [Vec<f64>; 3],
    /// Force latent spline coefficients, one row per force latent (empty with the force model off).
    pub frc: Vec<Vec<f64>>,
    /// The ground profile under both tyres (F4): not a Gauss-Newton unknown, refitted before each
    /// iteration ([`Objective::refit_ground`]); None until the first refit, or with the tie off.
    pub ground: Option<super::ground::GroundFit>,
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
    /// The force model (None: off); its parameters follow the other globals.
    pub forces: Option<ForceModel>,
    /// Threads one solve of this objective may use (normal equations by sample chunks, the
    /// border's solves): 1 when many windows already run side by side.
    pub threads: usize,
}

/// The bike lifted for one linearisation point: values, values + Jacobian,
/// and, when fitting, the geometry nudged each way for central differences.
pub struct Ctx {
    bike_f: Bike<f64>,
    bike_d: Bike<Dual64>,
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
        Objective { bike, win, spec, w, glob_centre: bias_centre.to_vec(), fit: false, forces: None, threads: 1 }
    }

    /// As [`Self::new`], also estimating the bike's geometry, with a prior
    /// ([`GEOMETRY_SIGMA`]) centred on `bike`'s.
    pub fn fitting(bike: Bike<f64>, win: Window, spec: Spec, w: Weights, bias_centre: [f64; NG]) -> Self {
        let mut c = bias_centre.to_vec();
        c.extend(geometry(&bike));
        Objective { bike, win, spec, w, glob_centre: c, fit: true, forces: None, threads: 1 }
    }

    /// This objective with the force model on (its parameters' prior centres at 0: the
    /// model's nominal values). The rows run where the window has [`Window::forces`].
    pub fn with_forces(mut self, m: ForceModel) -> Self {
        self.glob_centre.truncate(self.force_glob0());
        self.glob_centre.extend([0.0; NFP]);
        self.forces = Some(m);
        self
    }

    /// Index of the first force parameter among the globals.
    pub fn force_glob0(&self) -> usize {
        if self.fit { NGMAX } else { NG }
    }

    /// Number of globals: [`NG`], or [`NGMAX`] when fitting, plus [`NFP`] with the force model.
    pub fn n_glob(&self) -> usize {
        self.force_glob0() + if self.forces.is_some() { NFP } else { 0 }
    }

    /// The force model and the window's force measurements, when both are on.
    pub fn force_on(&self) -> Option<(&ForceModel, &ForceWindow)> {
        Some((self.forces.as_ref()?, self.win.forces.as_ref()?))
    }

    /// Prior sigma of each global.
    pub fn glob_sigma(&self) -> Vec<f64> {
        (0..self.n_glob())
            .map(|i| match i {
                i if i >= self.force_glob0() => self.forces.as_ref().map_or(1.0, |m| m.w.par_sigma[i - self.force_glob0()]),
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
        Ctx { bike_d: b.map(Dual64::from), bike_h: b.map(HyperDual::from_re), bike_f: b.clone(), bike_dh: b.map(|x| HyperDual::from_re(Dual64::from(x))), fd }
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
        let (att, vel) = self.initial_nav();
        let frc = match &self.forces {
            Some(m) if self.win.forces.is_some() => (0..NFL).map(|i| vec![0.0; self.spec.n_knots(m.spec.m[i])]).collect(),
            _ => Vec::new(),
        };
        Vars { lvl, frame, glob: self.glob_centre.clone(), att, vel, frc, ground: None }
    }

    /// The navigation layout and measurements, when both are on.
    pub fn nav(&self) -> Option<(&NavSpec, &NavWindow)> {
        Some((self.spec.nav.as_ref()?, self.win.nav.as_ref()?))
    }

    /// Starting attitude controls (the frame gyro, less its bias centre,
    /// integrated from the start attitude, and carried on past both ends) and
    /// velocity coefficients (the latest GPS fix, else the start velocity).
    fn initial_nav(&self) -> (Vec<M3<f64>>, [Vec<f64>; 3]) {
        let Some((ns, nw)) = self.nav() else {
            return (Vec::new(), [Vec::new(), Vec::new(), Vec::new()]);
        };
        let (n, dt) = (self.spec.n, self.spec.dt);
        let g = &self.win.frame_gyro;
        let rate = |t: usize| V3::from_f64(std::array::from_fn(|a| g[t.min(n - 1)][a] - self.glob_centre[a]));
        // attitude at samples -m .. n + 3m (index offset m)
        let m = ns.m_att as isize;
        let (lo, hi) = (-m, n as isize + 3 * m);
        let mut at = vec![M3::eye(); (hi - lo + 1) as usize];
        let r0 = M3::from_f64(nw.start.att);
        at[m as usize] = r0;
        for t in 1..=(hi as usize) {
            at[(t as isize - lo) as usize] = at[(t as isize - 1 - lo) as usize] * exp(rate(t - 1).scale(dt));
        }
        for t in (lo..0).rev() {
            at[(t - lo) as usize] = at[(t + 1 - lo) as usize] * exp(rate(0).scale(-dt));
        }
        let att = (0..self.spec.n_knots(ns.m_att)).map(|k| at[(((k as isize - 1) * m).clamp(lo, hi) - lo) as usize]).collect();
        let mut per = Vec::with_capacity(n);
        let mut last = nw.start.vel;
        for t in 0..n {
            if let Some(f) = nw.gps[t] {
                last = f.v;
                if !f.up {
                    last[2] = 0.0;
                }
            }
            per.push(last);
        }
        let mv = ns.m_vel as isize;
        let vel = std::array::from_fn(|a| (0..self.spec.n_knots(ns.m_vel)).map(|k| per[((k as isize - 1) * mv).clamp(0, n as isize - 1) as usize][a]).collect());
        (att, vel)
    }

    /// The navigation inputs at sample `t` (`s`: that sample's local inputs, for w0).
    pub fn nav_sample<'a>(&'a self, v: &Vars, s: &Sample<f64>, t: usize) -> Option<NavSample<'a>> {
        let (ns, nw) = self.nav()?;
        let base = t / ns.m_att;
        let ctrl = std::array::from_fn(|i| v.att[base + i]);
        let (j, vb) = (t % ns.m_vel, t / ns.m_vel);
        let b = basis(j, ns.m_vel);
        let h = ns.m_vel as f64 * self.spec.dt;
        let mut vel = [0.0; 3];
        let mut vd = [0.0; 3];
        for a in 0..3 {
            let c = &v.vel[a][vb..vb + 4];
            vel[a] = (0..4).map(|i| b[0][i] * c[i]).sum();
            vd[a] = (0..4).map(|i| b[1][i] * c[i]).sum::<f64>() / h;
        }
        Some(NavSample {
            ctrl,
            u: (t % ns.m_att) as f64 / ns.m_att as f64,
            span_s: ns.m_att as f64 * self.spec.dt,
            v: vel,
            vd,
            w0: s.w0.0,
            f_meas: self.win.frame_accel[t],
            gps: nw.gps[t],
            start: if t == 0 { Some(&nw.start) } else { None },
        })
    }

    /// The navigation rows at sample `t` (None with navigation off).
    pub fn nav_residuals(&self, v: &Vars, s: &Sample<f64>, t: usize, jac: bool) -> Option<NavJac> {
        Some(nav_rows(&self.nav_sample(v, s, t)?, &self.w.nav, jac))
    }

    /// The force latents at sample `t`: values, rates, second derivatives (SI).
    pub fn force_inputs(&self, v: &Vars, t: usize) -> Option<[[f64; NFL]; 3]> {
        let (m, _) = self.force_on()?;
        let sp = &m.spec;
        let mut out = [[0.0; NFL]; 3];
        for i in 0..NFL {
            let mi = sp.m[i];
            let (j, base) = (t % mi, t / mi);
            let b = basis(j, mi);
            let h = mi as f64 * self.spec.dt;
            let c = &v.frc[i][base..base + 4];
            let dot = |w: &[f64; 4]| w[0] * c[0] + w[1] * c[1] + w[2] * c[2] + w[3] * c[3];
            out[0][i] = sp.init[i] + sp.scale[i] * dot(&b[0]);
            out[1][i] = sp.scale[i] * dot(&b[1]) / h;
            out[2][i] = sp.scale[i] * dot(&b[2]) / (h * h);
        }
        Some(out)
    }

    /// The force rows' measurements at sample `t` (None with the force model off), with the
    /// ground tie's inputs where the window has them and `v` a ground profile.
    pub fn force_meas(&self, v: &Vars, t: usize) -> Option<ForceMeas> {
        let (_, fw) = self.force_on()?;
        let ground = fw.ground.as_ref().zip(v.ground.as_ref()).map(|(gw, gf)| ForceGround {
            up: gw.up[t],
            z_m: gw.z_m[t],
            h_m: std::array::from_fn(|i| gf.height(gw.s_m[t] + gw.offset_m[i])),
            sigma_m: std::array::from_fn(|i| if gf.on[t][i] { gw.sigma_m[t][i] } else { f64::INFINITY }),
        });
        let n = self.spec.n;
        let sp = &fw.speed_mps;
        let (a, b) = (t.saturating_sub(1), (t + 1).min(n - 1));
        let speed_rate = if b > a { (sp[b] - sp[a]) / ((b - a) as f64 * self.spec.dt) } else { 0.0 };
        let finite = |x: f64| if x.is_finite() { x } else { 0.0 };
        Some(ForceMeas { frame: self.win.frame_accel[t], fork: self.win.fork_accel[t], rear: self.win.rear_accel[t], speed: finite(sp[t]), speed_rate: finite(speed_rate), cam: fw.cam.get(t).copied().flatten(), ground })
    }

    /// The ground profile under each tyre at sample `t` (F4): its first coefficient, the four
    /// weights and the tie's 1 / sigma (1/m); None where that tyre's row is off.
    pub fn tie_basis(&self, v: &Vars, t: usize) -> [Option<(usize, [f64; 4], f64)>; 2] {
        let none = [None, None];
        let (Some((_, fw)), Some(gf)) = (self.force_on(), &v.ground) else { return none };
        let Some(gw) = &fw.ground else { return none };
        std::array::from_fn(|i| {
            let sig = gw.sigma_m[t][i];
            (gf.on[t][i] && sig.is_finite()).then(|| {
                let (k, w) = gf.span(gw.s_m[t] + gw.offset_m[i]);
                (k, w, 1.0 / sig)
            })
        })
    }

    /// The ground profile's curvature prior's weight per second difference (1/m^2; F4), None
    /// without the tie.
    pub fn ground_curvature_weight(&self) -> Option<f64> {
        let (m, _) = self.force_on()?;
        self.has_ground().then(|| 1.0 / m.w.ground.as_ref().map_or(1.0, |g| g.curvature_m).powi(2))
    }

    /// Half the ground profile's curvature prior (F4; 0 without a profile): counted with the
    /// tie, since the refit minimises the two together.
    pub fn ground_prior_loss(&self, v: &Vars) -> f64 {
        let (Some(wc), Some(g)) = (self.ground_curvature_weight(), &v.ground) else { return 0.0 };
        0.5 * wc * g.coef.windows(3).map(|c| (c[0] - 2.0 * c[1] + c[2]).powi(2)).sum::<f64>()
    }

    /// Whether the ground tie (F4) runs in this window: the force model with its settings and
    /// the window with the tie's inputs.
    pub fn has_ground(&self) -> bool {
        self.force_on().is_some_and(|(m, fw)| m.w.ground.is_some() && fw.ground.is_some())
    }

    /// Refits the ground profile in `v` to both tyres' patch heights at `v` (F4): each tyre that
    /// carries at least the minimum normal force, where its tie is on, weighted by the tie's
    /// sigma, Huber like the rows. Nothing without the tie's inputs.
    pub fn refit_ground(&self, v: &mut Vars) {
        let Some((m, fw)) = self.force_on() else { return };
        let (Some(gw), Some(w)) = (&fw.ground, &m.w.ground) else { return };
        let ctx = self.ctx(&v.glob, false);
        let par = self.force_par(v);
        // the patch heights do not depend on the profile: any profile will do for the rows
        let flat = super::ground::GroundFit { s0_m: 0.0, knot_m: 1.0, coef: vec![0.0; 4], on: vec![[true; 2]; self.spec.n] };
        let probe = Vars { ground: Some(flat), ..v.clone() };
        let mut patches = Vec::with_capacity(2 * self.spec.n);
        let mut on = vec![[false; 2]; self.spec.n];
        for t in 0..self.spec.n {
            let s = self.sample(v, t);
            let Some(f) = self.force_inputs(v, t) else { return };
            let Some(meas) = self.force_meas(&probe, t) else { return };
            let k = kin(&m.bike, &ctx.bike_f, &s.q);
            let x = ForceIn { d: [s.q[D], s.qd[D]], s: [s.q[S], s.qd[S]], w0: s.w0, a0: s.a0, f: f[0], fd: f[1], fdd: f[2], par };
            let fo = force_rows_at(m, &k, &x, &meas).1;
            for i in 0..2 {
                on[t][i] = fo.normal[i] >= w.min_normal_n;
                if on[t][i] && gw.sigma_m[t][i].is_finite() {
                    patches.push(super::ground::Patch { s_m: gw.s_m[t] + gw.offset_m[i], z_m: fo.patch_z[i], sigma_m: gw.sigma_m[t][i] });
                }
            }
        }
        v.ground = super::ground::fit_profile(&patches, w.knot_m, w.curvature_m, m.w.huber).map(|(s0_m, coef)| super::ground::GroundFit { s0_m, knot_m: w.knot_m, coef, on });
    }

    /// The force parameters (log multipliers) in `v`.
    pub fn force_par(&self, v: &Vars) -> [f64; NFP] {
        let g0 = self.force_glob0();
        std::array::from_fn(|i| v.glob.get(g0 + i).copied().unwrap_or(0.0))
    }

    /// The forces the solution implies at sample `t`, and the force latents there (None with
    /// the force model off).
    pub fn forces_at(&self, ctx: &Ctx, v: &Vars, t: usize) -> Option<(Forces<f64>, [[f64; NFL]; 3])> {
        let (m, _) = self.force_on()?;
        let s = self.sample(v, t);
        let f = self.force_inputs(v, t)?;
        let k = kin(&m.bike, &ctx.bike_f, &s.q);
        let x = ForceIn { d: [s.q[D], s.qd[D]], s: [s.q[S], s.qd[S]], w0: s.w0, a0: s.a0, f: f[0], fd: f[1], fdd: f[2], par: self.force_par(v) };
        Some((force_rows_at(m, &k, &x, &self.force_meas(v, t)?).1, f))
    }

    /// The force rows at sample `t` (None with the force model off). Where either unsprung
    /// pod is clipped the sample keeps only its priors (its balance rows read 0).
    pub fn force_residuals(&self, ctx: &Ctx, v: &Vars, s: &Sample<f64>, t: usize, jac: bool) -> Option<ForceJac> {
        let (m, _) = self.force_on()?;
        let f = self.force_inputs(v, t)?;
        let meas = self.force_meas(v, t)?;
        let par = self.force_par(v);
        let mut out = force_rows(m, &ctx.bike_f, &ctx.bike_d, s, &f[0], &f[1], &f[2], &par, &meas, jac);
        if !(self.win.mask_fork[t] && self.win.mask_rear[t]) {
            for r in 0..NBAL {
                out.r[r] = 0.0;
                out.j_core[r] = [0.0; NZ];
                out.j_frc[r] = [0.0; 3 * NFL];
                out.j_par[r] = [0.0; NFP];
            }
        }
        Some(out)
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
        prior(31, F_TWIST, w.fork_twist_sigma);
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
            let r = self.sample_residuals(&ctx, &s, &v.glob, t, false).r;
            l += r[..SENSOR_ROWS].iter().map(|&x| huber2(x, self.w.huber)).sum::<f64>();
            l += r[SENSOR_ROWS..].iter().map(|x| x * x).sum::<f64>();
            if let Some(nj) = self.nav_residuals(v, &s, t, false) {
                l += nj.r.iter().map(|x| x * x).sum::<f64>();
            }
            if let Some(fj) = self.force_residuals(&ctx, v, &s, t, false) {
                let k = self.forces.as_ref().map_or(0.0, |m| m.w.huber);
                l += fj.r.iter().enumerate().map(|(i, &x)| if super::forces::huber_row(i) { huber2(x, k) } else { x * x }).sum::<f64>();
            }
        }
        l += 2.0 * self.ground_prior_loss(v);
        let k0 = self.smooth_weight();
        let frc_smooth = |i: usize| self.forces.as_ref().map_or(1.0, |m| m.spec.smooth[i].sqrt());
        let rows = v.lvl.iter().flatten().map(|r| (r, 1.0)).chain(v.frc.iter().enumerate().map(|(i, r)| (r, frc_smooth(i))));
        for (row, ks) in rows {
            let k = k0 * ks;
            for i in 0..row.len() - 2 {
                let d2 = k * (row[i] - 2.0 * row[i + 1] + row[i + 2]);
                l += d2 * d2;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber2_inside_the_threshold_is_the_square_beyond_it_linear_and_both_join_smoothly() {
        // Arrange
        let k = 3.0;
        let h = 1e-6;

        // Act
        let (inside, at, beyond) = (huber2(2.0, k), huber2(k, k), huber2(-10.0, k));
        let slope_below = (huber2(k, k) - huber2(k - h, k)) / h;
        let slope_above = (huber2(k + h, k) - huber2(k, k)) / h;

        // Assert
        assert_eq!(inside, 4.0);
        assert_eq!(at, 9.0);
        assert_eq!(beyond, 2.0 * 3.0 * 10.0 - 9.0);
        assert!((slope_below - slope_above).abs() < 1e-4, "{slope_below} vs {slope_above}");
        assert_eq!(huber2(-10.0, 0.0), 100.0);
    }
}
