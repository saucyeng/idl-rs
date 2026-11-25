//! Can the physics estimator tell travel from flex when its model is exact?
//!
//! Takes the 2 s Beacon golden window's frame motion (gyro smoothed, accel as
//! measured), invents true latents (travel, steer, fork bend, rear flex,
//! frame-pod flex, sized like `ml/idlml/sim/synth.py`), predicts every pod's
//! readings with the estimator's own forward model, adds sensor noise, and
//! solves. Any error left is the estimator's, not the model's: a latent that
//! drifts, or an answer that depends on the start, is a direction the
//! sensors cannot see. Solves twice: from the usual start, and with the
//! travel start 20 mm high.
//!
//! Run: `cargo run --release -p idl-rs --example physics_identifiability`.

#[path = "common/physics_fixture.rs"]
mod physics_fixture;

use std::f64::consts::TAU;

use idl_rs::estimate::physics::bike::{B_FA, B_LAT, D, F_ROT, F_TRANS, NL, PSI, R_ROLL, R_YAW, S};
use idl_rs::estimate::physics::la::V3;
use idl_rs::estimate::physics::objective::{Objective, Spec, Weights, Window, NG};
use idl_rs::estimate::physics::readings::{pod_readings, Sample};
use idl_rs::estimate::physics::solve::{gauss_newton, latents};
use num_dual::HyperDual;

fn rows(j: &serde_json::Value) -> Vec<[f64; 3]> {
    serde_json::from_value(j.clone()).unwrap()
}

/// A small deterministic generator (no dependency): uniform in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
}

/// One latent as a sum of sines: value, rate, acceleration at `t`.
#[derive(Clone)]
struct Sines(f64, Vec<(f64, f64, f64)>); // offset, (amplitude, Hz, phase)
impl Sines {
    fn at(&self, t: f64) -> (f64, f64, f64) {
        let (mut x, mut xd, mut xdd) = (self.0, 0.0, 0.0);
        for &(a, f, p) in &self.1 {
            let w = TAU * f;
            x += a * (w * t + p).sin();
            xd += a * w * (w * t + p).cos();
            xdd -= a * w * w * (w * t + p).sin();
        }
        (x, xd, xdd)
    }
    /// `n` sines with frequencies spread over `lo..hi` Hz and an rms of `rms`.
    fn band(rng: &mut Lcg, rms: f64, lo: f64, hi: f64, n: usize) -> Sines {
        let a = rms * (2.0 / n as f64).sqrt();
        Sines(0.0, (0..n).map(|k| (a, lo + (hi - lo) * (k as f64 + rng.next()) / n as f64, TAU * rng.next())).collect())
    }
}

fn main() {
    let doc: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/physics_window.json")).unwrap();
    let bike = physics_fixture::bike_from(&doc["bike"]);
    let rate = doc["rate"].as_f64().unwrap();
    let dt = 1.0 / rate;
    let gyro = rows(&doc["frame_gyro"]);
    let accel = rows(&doc["frame_accel"]);
    let n = gyro.len();
    let mut rng = Lcg(7);

    // the chassis rate: the measured frame gyro, smoothed (a 25 ms moving average) so the truth is
    // a motion a spline can follow; its rate of change by central differences
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
    let a0: Vec<[f64; 3]> = (0..n).map(|t| {
        let (a, b) = (t.saturating_sub(1), (t + 1).min(n - 1));
        std::array::from_fn(|i| (w0[b][i] - w0[a][i]) / ((b - a) as f64 * dt))
    }).collect();

    // true latents, sized like synth.py's Beacon-shaped ride
    let mut truth: Vec<Sines> = (0..NL).map(|_| Sines(0.0, Vec::new())).collect();
    truth[D] = Sines(0.06, vec![(0.025, 1.1, 0.3), (0.010, 3.7, 1.9), (0.004, 9.0, 4.0)]);
    truth[S] = Sines(0.05, vec![(0.020, 0.9, 1.0), (0.008, 4.3, 2.2), (0.003, 11.0, 5.1)]);
    truth[PSI] = Sines(0.0, vec![(0.03, 0.7, 0.5)]);
    truth[B_FA] = Sines::band(&mut rng, 0.0015, 12.0, 40.0, 4);
    truth[B_LAT] = Sines::band(&mut rng, 0.0008, 12.0, 40.0, 4);
    truth[R_ROLL] = Sines::band(&mut rng, 0.002, 10.0, 45.0, 4);
    truth[R_YAW] = Sines::band(&mut rng, 0.0015, 10.0, 45.0, 4);
    for i in F_ROT..F_ROT + 3 {
        truth[i] = Sines::band(&mut rng, 0.0005, 10.0, 45.0, 4);
    }
    for i in F_TRANS..F_TRANS + 3 {
        truth[i] = Sines::band(&mut rng, 0.0002, 10.0, 45.0, 4);
    }
    let q_true: Vec<[f64; NL]> = (0..n).map(|t| std::array::from_fn(|i| truth[i].at(t as f64 * dt).0)).collect();

    // the readings the pods would give, plus noise (gyro 0.01 rad/s, accel 0.2 m/s^2)
    let bike_h = bike.map(HyperDual::from_re);
    let (gn, an) = (0.01, 0.2);
    let mut meas = vec![[[0.0; 3]; 5]; n];
    for t in 0..n {
        let tt = t as f64 * dt;
        let s = Sample {
            q: std::array::from_fn(|i| truth[i].at(tt).0),
            qd: std::array::from_fn(|i| truth[i].at(tt).1),
            qdd: std::array::from_fn(|i| truth[i].at(tt).2),
            w0: V3(w0[t]),
            a0: V3(a0[t]),
            f_meas: V3(accel[t]),
        };
        let r = pod_readings(&bike_h, &s);
        for k in 0..5 {
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
    };

    println!("{n} samples ({:.1} s), model exact, noise gyro {gn} rad/s accel {an} m/s^2", n as f64 * dt);
    let mut solved: Vec<Vec<[f64; NL]>> = Vec::new();
    for (label, travel_shift) in [("usual start", 0.0), ("travel start +20 mm", 0.02)] {
        let mut spec = Spec::standard(n, dt);
        spec.init[D] += travel_shift;
        spec.init[S] += travel_shift;
        let obj = Objective::new(bike.clone(), win.clone(), spec, Weights::default(), [0.0; NG]);
        let (v, info) = gauss_newton(&obj, obj.initial_vars(), 30, 1e-3).expect("solve");
        let q = latents(&obj, &v);
        println!("\n{label}: {} iterations, loss {:.1}", info.iterations, info.loss.last().unwrap());
        report(&q, &q_true);
        solved.push(q);
    }
    println!("\nbetween the two starts (rms difference):");
    report(&solved[1], &solved[0]);

    // the real window, as measured: what the model leaves unexplained, per reading, in sensor units
    let real = Window {
        frame_gyro: gyro.clone(),
        fork_gyro: rows(&doc["fork_gyro"]),
        fork_accel: rows(&doc["fork_accel"]),
        rear_gyro: rows(&doc["rear_gyro"]),
        rear_accel: rows(&doc["rear_accel"]),
        mask_fork: serde_json::from_value(doc["mask_fork"].clone()).unwrap(),
        mask_rear: serde_json::from_value(doc["mask_rear"].clone()).unwrap(),
        air: Some(serde_json::from_value(doc["air"].clone()).unwrap()),
        ..win.clone()
    };
    let mut centre = [0.0; NG];
    for (i, key) in ["frame_bias", "fork.gyro_bias", "fork.accel_bias", "rear.gyro_bias", "rear.accel_bias"].iter().enumerate() {
        let c = &doc["bias_prior"][key];
        for a in 0..3 {
            centre[3 * i + a] = c.as_f64().unwrap_or_else(|| c[a].as_f64().unwrap());
        }
    }
    // timing: shift each pod's samples by k (positive: the pod's data later) and re-solve the real
    // window; a minimum away from 0 is a clock offset between the pods
    if std::env::args().any(|a| a == "--skew") {
        let shift = |v: &Vec<[f64; 3]>, k: i64| -> Vec<[f64; 3]> {
            (0..n as i64).map(|t| v[(t - k).clamp(0, n as i64 - 1) as usize]).collect()
        };
        for pod in ["fork", "rear"] {
            let mut line = String::new();
            for k in -4i64..=4 {
                let mut w = real.clone();
                if pod == "fork" {
                    w.fork_gyro = shift(&real.fork_gyro, k);
                    w.fork_accel = shift(&real.fork_accel, k);
                } else {
                    w.rear_gyro = shift(&real.rear_gyro, k);
                    w.rear_accel = shift(&real.rear_accel, k);
                }
                let obj = Objective::new(bike.clone(), w, Spec::standard(n, dt), Weights::default(), centre);
                let (_, info) = gauss_newton(&obj, obj.initial_vars(), 30, 1e-3).expect("solve");
                line += &format!(" {k:+}: {:.0};", info.loss.last().unwrap());
            }
            println!("{pod} pod shifted by k samples ({:.1} ms each), loss:{line}", dt * 1e3);
        }
        return;
    }
    for (label, w) in [("synthetic", win.clone()), ("real Beacon", real)] {
        let obj = Objective::new(bike.clone(), w, Spec::standard(n, dt), Weights::default(), if label == "synthetic" { [0.0; NG] } else { centre });
        let (v, _) = gauss_newton(&obj, obj.initial_vars(), 30, 1e-3).expect("solve");
        let ctx = obj.ctx(&v.glob, false);
        let wt = Weights::default();
        let acc = (wt.accel_sigma.powi(2) + wt.frame_accel_sigma.powi(2)).sqrt();
        let sig = [wt.gyro_sigma, wt.gyro_sigma, acc, wt.gyro_sigma, acc];
        let mut ss = [[0.0; 3]; 5];
        let mut spectrum: Vec<[f64; 15]> = Vec::with_capacity(n);
        for t in 0..n {
            let s = obj.sample(&v, t);
            let r = obj.sample_residuals(&ctx, &s, &v.glob, t, false).r;
            let mut row = [0.0; 15];
            for k in 0..5 {
                for a in 0..3 {
                    let e = r[3 * k + a] * sig[k];
                    ss[k][a] += e * e;
                    row[3 * k + a] = e;
                }
            }
            spectrum.push(row);
        }
        println!("\n{label}: unexplained rms per reading (x, y, z in each pod's axes)");
        for (k, name) in ["frame gyro rad/s", "fork gyro rad/s", "fork accel m/s2", "rear gyro rad/s", "rear accel m/s2"].iter().enumerate() {
            let r: Vec<String> = ss[k].iter().map(|x| format!("{:.3}", (x / n as f64).sqrt())).collect();
            println!("  {name:<17} [{}]", r.join(", "));
        }
        // where in frequency the unexplained part sits: power in octave bands, fork and rear accel
        let bands: [(f64, f64); 6] = [(0.5, 4.0), (4.0, 12.0), (12.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, 200.0)];
        let mut line = String::new();
        for &(lo, hi) in &bands {
            let mut p = 0.0;
            for k in [6usize, 7, 8, 12, 13, 14] {
                let x: Vec<f64> = spectrum.iter().map(|r| r[k]).collect();
                let mut f = 0.0;
                let nf = (rate / (n as f64)).max(1e-9);
                let mut fq = lo.max(nf);
                while fq < hi {
                    let (mut c, mut s_) = (0.0, 0.0);
                    for (t, xv) in x.iter().enumerate() {
                        let ph = TAU * fq * t as f64 * dt;
                        c += xv * ph.cos();
                        s_ += xv * ph.sin();
                    }
                    f += (c * c + s_ * s_) / (n as f64 * n as f64) * 2.0;
                    fq += nf;
                }
                p += f;
            }
            line += &format!(" {lo}-{hi} Hz {:.3};", p.sqrt());
        }
        println!("  pod accel unexplained, rms by band (m/s2):{line}");
        // does the answer depend on the start? the same window from travel 20 mm higher
        let mut spec = Spec::standard(n, dt);
        spec.init[D] += 0.02;
        spec.init[S] += 0.02;
        let obj2 = Objective::new(bike.clone(), obj.win.clone(), spec, Weights::default(), if label == "synthetic" { [0.0; NG] } else { centre });
        let (v2, info2) = gauss_newton(&obj2, obj2.initial_vars(), 30, 1e-3).expect("solve");
        let (qa, qb) = (latents(&obj, &v), latents(&obj2, &v2));
        let d = |i: usize| (qa.iter().zip(&qb).map(|(a, b)| (a[i] - b[i]).powi(2)).sum::<f64>() / n as f64).sqrt() * 1e3;
        let mean = |q: &[[f64; NL]], i: usize| q.iter().map(|x| x[i]).sum::<f64>() / n as f64 * 1e3;
        for (name, i) in [("rear pitch flex", 13usize), ("fork twist", 14), ("steer", 1), ("fork bend fa", 2), ("fork bend lat", 3), ("rear roll", 5), ("rear yaw", 6), ("frame rot x", 7), ("frame rot y", 8), ("frame rot z", 9)] {
            if i < NL {
                let xs: Vec<f64> = qa.iter().map(|x| x[i]).collect();
                let m = xs.iter().sum::<f64>() / n as f64;
                let rms = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64).sqrt();
                let peak = xs.iter().fold(0.0f64, |a, x| a.max(x.abs()));
                println!("  {name:<16} mean {:+.2} deg, rms {:.2} deg, peak {:.2} deg", m.to_degrees(), rms.to_degrees(), peak.to_degrees());
            }
        }
        println!(
            "  start +20 mm: loss {:.1} vs {:.1}; travel differs rms front {:.2} mm, rear {:.2} mm (mean {:.1} vs {:.1}, {:.1} vs {:.1})",
            info2.loss.last().unwrap(), obj.loss(&v), d(D), d(S), mean(&qa, D), mean(&qb, D), mean(&qa, S), mean(&qb, S)
        );
    }
}

/// Mean and rms error of each latent group against `truth`.
fn report(q: &[[f64; NL]], truth: &[[f64; NL]]) {
    let groups: [(&str, Vec<usize>, f64, &str); 7] = [
        ("front travel", vec![D], 1e3, "mm"),
        ("rear travel", vec![S], 1e3, "mm"),
        ("steer", vec![PSI], 57.2958, "deg"),
        ("fork bend", vec![B_FA, B_LAT], 1e3, "mrad"),
        ("rear flex", vec![R_ROLL, R_YAW], 1e3, "mrad"),
        ("frame flex rot", (F_ROT..F_ROT + 3).collect(), 1e3, "mrad"),
        ("frame flex trans", (F_TRANS..F_TRANS + 3).collect(), 1e3, "mm"),
    ];
    // the window's edges are weakly held by a spline; judge the middle 80 %
    let (a, b) = (q.len() / 10, q.len() * 9 / 10);
    for (name, ix, k, unit) in groups {
        let mut mean = vec![0.0; ix.len()];
        let mut ss = 0.0;
        let mut tr = 0.0;
        for t in a..b {
            for (j, &i) in ix.iter().enumerate() {
                let e = q[t][i] - truth[t][i];
                mean[j] += e;
                ss += e * e;
                tr += (truth[t][i] - truth[a..b].iter().map(|x| x[i]).sum::<f64>() / (b - a) as f64).powi(2);
            }
        }
        let m = (b - a) as f64;
        let means: Vec<String> = mean.iter().map(|x| format!("{:+.2}", x / m * k)).collect();
        println!(
            "  {name:<17} rms err {:>7.3} {unit:<4} (truth rms {:>7.3})  mean err [{}]",
            (ss / (m * ix.len() as f64)).sqrt() * k,
            (tr / (m * ix.len() as f64)).sqrt() * k,
            means.join(", ")
        );
    }
}
