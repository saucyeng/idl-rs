//! Speed of the physics estimator's forward model: pod readings and their
//! per-sample Jacobian over one window the size of the PyTorch benchmark
//! (50.1 s at 400 Hz = 20 040 samples), single-threaded.
//!
//! Run: `cargo run -p idl-rs --example physics_bench` (the dev profile builds
//! idl-rs at opt-level 3). For comparison, the PyTorch solver on a GTX 1650
//! spends ~7.7 s per Gauss-Newton iteration building the same Jacobians
//! (normal equations 54 s over 7 iterations, `ml/runs/real`).

use std::time::Instant;

#[path = "common/physics_fixture.rs"]
mod physics_fixture;
use physics_fixture::{bike_from, sample};
use idl_rs::estimate::physics::la::V3;
use idl_rs::estimate::physics::readings::{pod_readings, readings_and_jacobian, Sample, NZ};
use num_dual::{Dual64, DualNum, HyperDual};  // DualNum: HyperDual::from_re

fn main() {
    let doc: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/physics_golden.json")).unwrap();
    let bike = bike_from(&doc["bike"]);
    let base = sample(&doc["samples"][0]);
    let n = 20_040;
    // vary the inputs smoothly so nothing is constant-folded
    let samples: Vec<Sample<f64>> = (0..n)
        .map(|i| {
            let t = i as f64 / 400.0;
            let mut s = base;
            s.q[0] = 0.06 + 0.04 * (3.0 * t).sin();
            s.q[4] = 0.05 + 0.03 * (2.0 * t).cos();
            s.q[1] = 0.1 * (0.7 * t).sin();
            s.w0.0[1] += (5.0 * t).sin();
            s
        })
        .collect();

    let bike_h = bike.map(HyperDual::from_re);
    let t0 = Instant::now();
    let mut sum = 0.0;
    for s in &samples {
        sum += pod_readings(&bike_h, s)[6];
    }
    let values = t0.elapsed().as_secs_f64();

    let bike_dh = bike.map(|x| HyperDual::from_re(Dual64::from(x)));
    let t0 = Instant::now();
    for s in &samples {
        let (v, j) = readings_and_jacobian(&bike_h, &bike_dh, s);
        sum += v[6] + j[6][NZ - 1];
    }
    let jac = t0.elapsed().as_secs_f64();

    println!("{n} samples ({:.1} s of ride at 400 Hz), one thread:", n as f64 / 400.0);
    println!("  readings only:         {:7.3} s  ({:.2} us/sample)", values, values / n as f64 * 1e6);
    println!("  readings + Jacobian:   {:7.3} s  ({:.2} us/sample, {NZ} inputs)", jac, jac / n as f64 * 1e6);
    println!("  (checksum {sum:.3}; PyTorch on the GTX 1650: ~7.7 s per iteration for this step)");
}
