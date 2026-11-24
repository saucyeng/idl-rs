//! Speed of the physics estimator's window solve (Gauss-Newton) on the 2 s
//! golden window of the Beacon ride, with the time split by stage.
//!
//! Run: `cargo run -p idl-rs --example physics_solve_bench`.

#[path = "common/physics_fixture.rs"]
mod physics_fixture;

use idl_rs::estimate::physics::objective::{Objective, Spec, Weights, Window, NG};
use idl_rs::estimate::physics::solve::{gauss_newton, normal_equations, solve_damped, Layout};

fn rows(j: &serde_json::Value) -> Vec<[f64; 3]> {
    serde_json::from_value(j.clone()).unwrap()
}

fn main() {
    let doc: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/physics_window.json")).unwrap();
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
    };
    let n = win.frame_gyro.len();
    let mut centre = [0.0; NG];
    for (i, key) in ["frame_bias", "fork.gyro_bias", "fork.accel_bias", "rear.gyro_bias", "rear.accel_bias"].iter().enumerate() {
        let c = &doc["bias_prior"][key];
        for a in 0..3 {
            centre[3 * i + a] = c.as_f64().unwrap_or_else(|| c[a].as_f64().unwrap());
        }
    }
    let obj = Objective::new(physics_fixture::bike_from(&doc["bike"]), win, Spec::standard(n, 1.0 / 400.0), Weights::default(), centre);
    let lay = Layout::new(&obj);
    let v0 = obj.initial_vars();
    let t = std::time::Instant::now();
    let ne = normal_equations(&obj, &lay, &v0, true);
    let t_ne = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let _ = solve_damped(&ne, &lay, 1e-10).unwrap();
    let t_solve = t.elapsed().as_secs_f64();
    println!("one assembly {:.3} s ({:.0} us/sample), one damped solve {:.3} s", t_ne, t_ne / n as f64 * 1e6, t_solve);
    let (_, info) = gauss_newton(&obj, obj.initial_vars(), 30, 1.0).unwrap();
    let total: f64 = info.seconds.iter().sum();
    println!("{n} samples ({:.1} s of ride), {} unknowns, band half-width {}", n as f64 / 400.0, lay.n_loc, lay.bw);
    println!("  {} iterations, {:.2} s: normal equations {:.2}, solve {:.2}, loss {:.2}", info.iterations, total, info.seconds[0], info.seconds[1], info.seconds[2]);
    println!("  final loss {:.2} (PyTorch {:.2})", info.loss.last().unwrap(), doc["solution"]["loss"].as_f64().unwrap());
    println!("  per second of ride: {:.2} s", total / (n as f64 / 400.0));
}
