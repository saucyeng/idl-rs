//! Exports a per-sample estimator trace from a real `.idl0` session to CSV (stdout),
//! for the offline tuning GUI (`tools/estimator_sim/gui.py`). The drives are the
//! EXACT wheel-drive accelerations the engine fed its integrators, so a downstream
//! 2-state replay reproduces the engine's travel.
//!
//!   cargo run -q --example estimate_trace -- <session.idl0> > trace.csv
//!
//! Columns: t, front_drive, rear_drive, accel0_norm, front_diff_mag, rear_diff_mag,
//! stationary, eng_front_mm, eng_rear_mm  (drives m/s², travel mm).

use idl_rs::estimate::geometry::BikeGeometry;
use idl_rs::estimate::run::{run_trace, EstimatorConfig, EstimatorInput};
use idl_rs::session::handle::SessionHandle;
use std::io::Write;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: estimate_trace <session.idl0>  (writes CSV to stdout)");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let handle = SessionHandle::from_bytes(&bytes).expect("parse .idl0");
    let input = EstimatorInput::from_lookup(&handle).expect("session has IMU0 channels");
    let geometry = BikeGeometry::reference_bike();
    let config = EstimatorConfig::default();

    let (est, tr) = run_trace(&input, &geometry, &config);

    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());
    writeln!(w, "t,front_drive,rear_drive,accel0_norm,front_diff_mag,rear_diff_mag,stationary,eng_front_mm,eng_rear_mm").unwrap();
    for i in 0..tr.front_drive.len() {
        writeln!(
            w,
            "{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{},{:.4},{:.4}",
            i as f64 * tr.dt,
            tr.front_drive[i],
            tr.rear_drive[i],
            tr.accel0_norm[i],
            tr.front_diff_mag[i],
            tr.rear_diff_mag[i],
            tr.stationary[i] as u8,
            est.front_travel[i] * 1000.0,
            est.rear_travel[i] * 1000.0,
        )
        .unwrap();
    }
    eprintln!("wrote {} samples (dt={:.6}s, {:.1}s)", tr.front_drive.len(), tr.dt, tr.front_drive.len() as f64 * tr.dt);
}
