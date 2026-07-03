//! Dev diagnostic for Option A: does the sprung-mass (chassis) vertical specific
//! force observe wheel travel via the spring force? Dumps, per sample, the
//! bounds-only estimator travel/velocity and the pitch-corrected chassis vertical
//! specific force at each axle (replicating the workbook "Front/Rear axle vert
//! accel": IMU0 vertical + lever·d/dt(pitch rate)). Downstream Python regresses
//! a_axle against (w, ẇ) to see if a proportional spring coupling exists and
//! whether its slope ≈ g/sag.
//!
//!   cargo run -q --example diag_coupling -- <session.idl0> > coupling.csv

use idl_rs::estimate::geometry::BikeGeometry;
use idl_rs::estimate::run::{run, EstimatorConfig, EstimatorInput};
use idl_rs::session::handle::SessionHandle;
use std::io::Write;

fn central_diff(v: &[f64], dt: f64) -> Vec<f64> {
    let n = v.len();
    let mut d = vec![0.0; n];
    for i in 0..n {
        d[i] = if n < 2 {
            0.0
        } else if i == 0 {
            (v[1] - v[0]) / dt
        } else if i == n - 1 {
            (v[n - 1] - v[n - 2]) / dt
        } else {
            (v[i + 1] - v[i - 1]) / (2.0 * dt)
        };
    }
    d
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: diag_coupling <session.idl0>");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let handle = SessionHandle::from_bytes(&bytes).expect("parse .idl0");
    let input = EstimatorInput::from_lookup(&handle).expect("session has IMU0 channels");
    let geometry = BikeGeometry::reference_bike();
    let est = run(&input, &geometry, &EstimatorConfig::default()); // bounds-only

    let dt = input.dt;
    // Raw IMU0 sensor frame is X-rear, Y-right, Z-up: vertical = accel.z (m/s²),
    // pitch rate = gyro.y (rad/s). Pitch-correct to each axle with the geometry
    // levers (front +0.835, rear −0.445 m), matching the workbook channels.
    let az: Vec<f64> = input.imu0.accel.iter().map(|a| a.z).collect();
    let gy: Vec<f64> = input.imu0.gyro.iter().map(|g| g.y).collect();
    let dgy = central_diff(&gy, dt);
    let n = est.front_travel.len().min(az.len());

    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());
    writeln!(w, "t,ft_m,fv_ms,a_front,rt_m,rv_ms,a_rear").unwrap();
    for i in 0..n {
        let a_front = az[i] + 0.835 * dgy[i];
        let a_rear = az[i] - 0.445 * dgy[i];
        writeln!(
            w,
            "{:.5},{:.6},{:.5},{:.5},{:.6},{:.5},{:.5}",
            i as f64 * dt,
            est.front_travel[i],
            est.front_velocity[i],
            a_front,
            est.rear_travel[i],
            est.rear_velocity[i],
            a_rear
        )
        .unwrap();
    }
    eprintln!("wrote {n} samples dt={dt:.6}s");
}
