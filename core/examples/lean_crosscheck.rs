//! Cross-checks estimator roll against an independent physical estimate.
//!
//! In a steady coordinated turn, lean satisfies `tan φ = v·ψ̇ / g` — speed times
//! yaw rate over gravity. That estimate shares none of the IEKF's machinery
//! (no gyro integration, no gravity levelling, no filter), so agreement is real
//! evidence rather than a tautology.
//!
//!   cargo run -q --release --example lean_crosscheck -- <session.idl0>
//!
//! Prints one row per second of steady turning: t, estimator roll (deg),
//! geometric roll (deg), difference (deg).

use idl_rs::estimate::geometry::BikeGeometry;
use idl_rs::math::eval::ChannelLookup;
use idl_rs::session::handle::SessionHandle;
use nalgebra::Vector3;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: lean_crosscheck <session.idl0>");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let handle = SessionHandle::from_bytes(&bytes).expect("parse .idl0");

    let roll = handle
        .estimator_channel("Roll (deg)")
        .expect("estimator could not run (needs IMU0)");
    let roll_hz = roll.sample_rate_hz;

    let speed_kmh = handle.channel_samples("GPS_SpeedKmh");
    // Yaw rate must be in the CHASSIS frame. IMU0 is mounted rotated (the
    // reference bike has it X-rear/Y-right plus a fitted tilt), so raw
    // `IMU0_GyroZ` mixes yaw with roll and pitch rate. Apply the same static
    // mount quaternion the estimator uses — that is calibration, not filter
    // output, so this stays independent of the IEKF.
    let mount = BikeGeometry::reference_bike().imu0.mount;
    let gx = handle.channel_samples("IMU0_GyroX");
    let gy = handle.channel_samples("IMU0_GyroY");
    let gz = handle.channel_samples("IMU0_GyroZ");
    let yaw_hz = handle
        .channel_meta("IMU0_GyroZ")
        .map(|m| m.sample_rate_hz)
        .unwrap_or(0.0);
    if speed_kmh.is_empty() || gz.is_empty() || gx.len() != gz.len() || gy.len() != gz.len() {
        eprintln!("session lacks GPS_SpeedKmh or a complete IMU0 gyro triple");
        return;
    }
    let yaw_dps: Vec<f64> = (0..gz.len())
        .map(|i| (mount * Vector3::new(gx[i], gy[i], gz[i])).z)
        .collect();

    println!("t_s,estimator_roll_deg,geometric_roll_deg,diff_deg");
    // GPS is 1 Hz; step one second at a time and average the faster channels
    // over that second. No `resample` exists in the math language, which is why
    // this lives here rather than in a workbook.
    for (sec, v_kmh) in speed_kmh.iter().enumerate() {
        let v = v_kmh / 3.6;
        if v < 3.0 {
            continue; // too slow for a meaningful coordinated turn
        }
        let mean = |samples: &[f64], hz: f64| -> f64 {
            let a = (sec as f64 * hz) as usize;
            let b = (((sec + 1) as f64) * hz) as usize;
            if hz <= 0.0 || a >= samples.len() {
                return f64::NAN;
            }
            let b = b.min(samples.len());
            samples[a..b].iter().sum::<f64>() / (b - a) as f64
        };
        // Steady-turn gate: the formula only holds through a sustained turn, so
        // require the yaw rate to keep one sign for the whole second. Without
        // this a corner entry and exit average to mush and the comparison is
        // meaningless.
        let a = (sec as f64 * yaw_hz) as usize;
        let b = (((sec + 1) as f64) * yaw_hz) as usize;
        if a >= yaw_dps.len() {
            continue;
        }
        let win = &yaw_dps[a..b.min(yaw_dps.len())];
        if win.is_empty() {
            continue;
        }
        // 90%, not 100%: at 833 Hz a real sustained turn still has individual
        // samples crossing zero from gyro noise, and demanding unanimity throws
        // away almost every genuine corner.
        let pos = win.iter().filter(|w| **w > 0.0).count();
        let dominant = pos.max(win.len() - pos) as f64 / win.len() as f64;
        if dominant < 0.90 {
            continue;
        }
        let yaw_rate = mean(&yaw_dps, yaw_hz).to_radians();
        if yaw_rate.abs() < 0.15 {
            continue; // near-straight: lean is not turn-driven
        }
        // Positive yaw rate is a LEFT turn (nose swings +X toward +Y), and the
        // rider leans into it — left side down — which is negative roll under
        // the right-side-down-positive convention. Hence the leading minus.
        let geometric = -(v * yaw_rate / 9.80665).atan().to_degrees();
        let est = mean(&roll.samples, roll_hz);
        println!("{sec},{est:.2},{geometric:.2},{:.2}", est - geometric);
    }
}
