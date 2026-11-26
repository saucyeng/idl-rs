//! Does the physics estimator overshoot a bottom-out when its model is exact?
//!
//! The 2 s Beacon golden window's frame motion (gyro smoothed, accel as
//! measured); true front travel resting at 60 mm, compressing at 2 m/s and
//! stopping at 170 mm within a few ms (a smooth max, `tau` s), halfway through;
//! rear travel, steer and the flexes small and smooth. Pod readings from the
//! estimator's own forward model plus sensor noise; then the usual solve.
//! Reports the solved front travel's peak and its error around the stop.
//!
//! Variants: `--tau-ms <x>` (how sharp the stop is, default 2), `--skew-us <x>`
//! (the fork pod's clock this far ahead of the frame's, default 0),
//! `--wall-sigma-mm <x>` (the end stop's softness, default the solver's).
//!
//! Run: `cargo run --release -p idl-rs --example physics_bottom_out -- --skew-us 1000`.

#[path = "common/physics_fixture.rs"]
mod physics_fixture;

use std::f64::consts::TAU;

use idl_rs::estimate::physics::bike::{D, NL, PSI, S};
use idl_rs::estimate::physics::la::V3;
use idl_rs::estimate::physics::objective::{Objective, Spec, Weights, Window, NG};
use idl_rs::estimate::physics::readings::{pod_readings, Sample};
use idl_rs::estimate::physics::solve::{gauss_newton, latents};
use num_dual::HyperDual;

fn rows(j: &serde_json::Value) -> Vec<[f64; 3]> {
    serde_json::from_value(j.clone()).unwrap()
}

fn arg(name: &str, default: f64) -> f64 {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
}

/// Softplus with width `w`: a smooth max(x, 0).
fn sp(x: f64, w: f64) -> f64 {
    let z = x / w;
    if z > 30.0 { x } else { w * z.exp().ln_1p() }
}

fn main() {
    let tau = arg("--tau-ms", 2.0) * 1e-3;
    let skew = arg("--skew-us", 0.0) * 1e-6;
    let doc: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/physics_window.json")).unwrap();
    let bike = physics_fixture::bike_from(&doc["bike"]);
    let rate = doc["rate"].as_f64().unwrap();
    let dt = 1.0 / rate;
    let gyro = rows(&doc["frame_gyro"]);
    let accel = rows(&doc["frame_accel"]);
    let n = gyro.len();
    let mut rng = Lcg(11);

    let half = 5;
    let w0: Vec<[f64; 3]> = (0..n)
        .map(|t| {
            let (a, b) = (t.saturating_sub(half), (t + half).min(n - 1));
            let mut s = [0.0; 3];
            for g in &gyro[a..=b] {
                (0..3).for_each(|i| s[i] += g[i]);
            }
            s.map(|x| x / (b - a + 1) as f64)
        })
        .collect();
    let a0: Vec<[f64; 3]> = (0..n)
        .map(|t| {
            let (a, b) = (t.saturating_sub(1), (t + 1).min(n - 1));
            std::array::from_fn(|i| (w0[b][i] - w0[a][i]) / ((b - a) as f64 * dt))
        })
        .collect();

    // the true front travel: a line at 2 m/s ending at 170 mm with a stop `tau` wide, floored
    // smoothly at 60 mm (so it rests, compresses in ~55 ms, and stops)
    let (dmax, d0, v0, t0) = (0.170, 0.060, 2.0, n as f64 * dt * 0.5);
    let front = |t: f64| -> f64 {
        let line = dmax - v0 * sp(t0 - t, tau);
        d0 + sp(line - d0, 0.004)
    };
    let others = |t: f64| -> [f64; NL] {
        let mut q = [0.0; NL];
        q[S] = 0.05 + 0.01 * (TAU * 0.9 * t).sin();
        q[PSI] = 0.03 * (TAU * 0.7 * t).sin();
        q
    };
    // values and time derivatives by central differences (h small against tau)
    let h = (tau / 20.0).min(1e-4);
    let state = |t: f64| -> ([f64; NL], [f64; NL], [f64; NL]) {
        let at = |t: f64| {
            let mut q = others(t);
            q[D] = front(t);
            q
        };
        let (qm, q, qp) = (at(t - h), at(t), at(t + h));
        (q, std::array::from_fn(|i| (qp[i] - qm[i]) / (2.0 * h)), std::array::from_fn(|i| (qp[i] - 2.0 * q[i] + qm[i]) / (h * h)))
    };

    let bike_h = bike.map(HyperDual::from_re);
    let (gn, an) = (0.01, 0.2);
    let mut meas = vec![[[0.0; 3]; 5]; n];
    let mut truth = vec![[0.0; NL]; n];
    for t in 0..n {
        let tt = t as f64 * dt;
        truth[t] = state(tt).0;
        for (k, shift) in [(0usize, 0.0), (1, skew), (2, skew), (3, 0.0), (4, 0.0)] {
            // the fork pod (readings 1, 2) sampled `skew` later on the true timeline
            let (q, qd, qdd) = state(tt + shift);
            let s = Sample { q, qd, qdd, w0: V3(w0[t]), a0: V3(a0[t]), f_meas: V3(accel[t]) };
            let r = pod_readings(&bike_h, &s);
            let sig = if k == 2 || k == 4 { an } else { gn };
            meas[t][k] = std::array::from_fn(|a| r[3 * k + a] + sig * rng.normal());
        }
    }
    let col = |k: usize| meas.iter().map(|m| m[k]).collect::<Vec<_>>();
    let win = Window {
        rate,
        frame_gyro: col(0),
        frame_accel: accel.clone(),
        fork_gyro: col(1),
        fork_accel: col(2),
        rear_gyro: col(3),
        rear_accel: col(4),
        mask_fork: vec![true; n],
        mask_rear: vec![true; n],
        air: None,
        nav: None,
        forces: None,
    };
    let mut w = Weights::default();
    w.wall_sigma = arg("--wall-sigma-mm", w.wall_sigma * 1e3) * 1e-3;
    let obj = Objective::new(bike.clone(), win, Spec::standard(n, dt), w, [0.0; NG]);
    let (v, info) = gauss_newton(&obj, obj.initial_vars(), 40, 1e-3).expect("solve");
    let q = latents(&obj, &v);

    let peak = q.iter().map(|x| x[D]).fold(f64::MIN, f64::max);
    let ti = (t0 / dt) as usize;
    let around = |lo: f64, hi: f64| -> f64 {
        let (a, b) = (((t0 + lo) / dt) as usize, (((t0 + hi) / dt) as usize).min(n));
        (q[a..b].iter().zip(&truth[a..b]).map(|(x, y)| (x[D] - y[D]).powi(2)).sum::<f64>() / (b - a) as f64).sqrt() * 1e3
    };
    let after = q[ti..(ti + (0.2 / dt) as usize).min(n)].iter().map(|x| x[D]).fold(f64::MIN, f64::max);
    println!(
        "stop {:.1} ms wide, fork clock +{:.0} us, wall sigma {:.1} mm: {} iterations, loss {:.1}",
        tau * 1e3,
        skew * 1e6,
        obj.w.wall_sigma * 1e3,
        info.iterations,
        info.loss.last().unwrap()
    );
    println!(
        "  true peak 170.0 mm; solved peak {:.1} mm ({:+.1}); in the 200 ms after the stop max {:.1} mm",
        peak * 1e3,
        (peak - dmax) * 1e3,
        after * 1e3
    );
    println!(
        "  front travel rms error: 100 ms before the stop {:.2} mm, 20 ms around it {:.2} mm, 100 ms after {:.2} mm",
        around(-0.1, -0.01),
        around(-0.01, 0.01),
        around(0.01, 0.1)
    );
}
