//! Dev A/B harness: runs the suspension estimator on a real `.idl0` session twice —
//! bounds-only (the new default, `use_sag_prior = false`) and with the sag prior on —
//! and prints front/rear travel + velocity distribution stats for each, so the
//! bounds-only change can be judged on real data (does travel rail? does velocity
//! widen toward the DSP-filtered shape?).
//!
//!   cargo run -q --example compare_sag -- <session.idl0>

use idl_rs::estimate::geometry::BikeGeometry;
use idl_rs::estimate::run::{run, EstimatorConfig, EstimatorInput};
use idl_rs::session::handle::SessionHandle;

fn stats(label: &str, v: &[f64]) {
    if v.is_empty() {
        println!("  {label:<22} (empty)");
        return;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    let skew = if sd > 0.0 {
        v.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((p * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)];
    println!(
        "  {label:<22} mean={:+.4} sd={:.4} skew={:+.3} min={:+.4} p1={:+.4} p50={:+.4} p99={:+.4} max={:+.4}",
        mean, sd, skew, s[0], q(0.01), q(0.50), q(0.99), s[s.len() - 1]
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: compare_sag <session.idl0>");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let handle = SessionHandle::from_bytes(&bytes).expect("parse .idl0");
    let input = EstimatorInput::from_lookup(&handle).expect("session has IMU0 channels");
    let geometry = BikeGeometry::reference_bike();

    for (label, cfg) in [
        ("BOUNDS-ONLY (default)", EstimatorConfig::default()),
        ("SAG PRIOR ON", EstimatorConfig { use_sag_prior: true, ..EstimatorConfig::default() }),
    ] {
        let est = run(&input, &geometry, &cfg);
        let ft: Vec<f64> = est.front_travel.iter().map(|x| x * 1000.0).collect(); // mm
        let fv: Vec<f64> = est.front_velocity.iter().map(|x| x * 1000.0).collect(); // mm/s
        let rt: Vec<f64> = est.rear_travel.iter().map(|x| x * 1000.0).collect();
        let rv: Vec<f64> = est.rear_velocity.iter().map(|x| x * 1000.0).collect();
        let riding = est.stationary.iter().filter(|s| !**s).count();
        println!(
            "\n=== {label} ===  ({} samples, {} riding, travel_max front={:.0}mm rear={:.0}mm)",
            est.len(),
            riding,
            geometry.front_travel_max * 1000.0,
            geometry.rear_travel_max * 1000.0
        );
        stats("front travel (mm)", &ft);
        stats("front velocity (mm/s)", &fv);
        stats("rear travel (mm)", &rt);
        stats("rear velocity (mm/s)", &rv);
        if let Some(s) = est.front_dynamic_sag() {
            println!("  front dynamic sag (median riding travel) = {:.1} mm", s * 1000.0);
        }
    }
}
