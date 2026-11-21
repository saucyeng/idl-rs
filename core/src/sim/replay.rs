//! A recorded ride as an [`Excitation`] (spec §5): IMU0's specific force and
//! pitch rate turned into the bike's side plane by the calibration's up and
//! forward directions and low-passed, the rear travel the physics estimator
//! solved, and the GPS ground speed, all on IMU0's grid.

use super::excite::Excitation;
use super::{SimError, SimErrorKind};

/// Default low-pass cut-off for IMU0 in a replay, Hz: IMU0's content up to
/// here is real frame motion (spec §5).
pub const DEFAULT_CUTOFF_HZ: f64 = 100.0;

/// How IMU0 sits in the bike: the calibration's directions, in IMU0's axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mount {
    /// Unit vector pointing up with the bike level.
    pub up: [f64; 3],
    /// Unit vector pointing forward, level.
    pub forward: [f64; 3],
    /// Gyro offset at rest, rad/s.
    pub gyro_bias_rad_s: [f64; 3],
}

/// A stretch of recording, every time on one timeline, s.
#[derive(Debug, Clone, Copy)]
pub struct Recording<'a> {
    /// Time of IMU0's first sample, s.
    pub t0_s: f64,
    /// IMU0's uniform sample period, s.
    pub dt_s: f64,
    /// IMU0's specific force, m/s², its own axes, one per sample.
    pub accel: &'a [[f64; 3]],
    /// IMU0's rotation rate, rad/s, its own axes, one per sample.
    pub gyro: &'a [[f64; 3]],
    /// Rear travel: (time s, travel m), increasing time; NaN where not solved.
    pub travel: &'a [(f64, f64)],
    /// Ground speed: (time s, m/s), increasing time.
    pub speed: &'a [(f64, f64)],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(a, a).sqrt();
    (n > 1e-9 && n.is_finite()).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

/// `series` (time, value) at `t`: linear between the finite samples, held
/// past the ends; None without a finite sample.
fn interp(series: &[(f64, f64)], t: f64) -> Option<f64> {
    let i = series.partition_point(|p| p.0 < t);
    let before = series[..i].iter().rev().find(|p| p.1.is_finite());
    let after = series[i..].iter().find(|p| p.1.is_finite());
    match (before, after) {
        (Some(a), Some(b)) if b.0 > a.0 => Some(a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0)),
        (Some(a), _) => Some(a.1),
        (None, Some(b)) => Some(b.1),
        (None, None) => None,
    }
}

/// The recording as an excitation starting at time 0 (= `rec.t0_s`):
/// specific force along the bike's forward and up directions, the rotation
/// rate about its right-hand side (nose up positive, the sim's CCW), each
/// low-passed at `cutoff_hz` (zero phase; skipped at or above 0.45 × the
/// sample rate), the travel and speed interpolated onto IMU0's grid (travel
/// bridged across unsolved gaps and held at the ends, never below 0).
/// Errors: `Excitation` for fewer than two IMU0 samples, a bad period, a
/// mount without two independent directions, no solved travel or no speed.
pub fn excitation(rec: &Recording, mount: &Mount, cutoff_hz: f64) -> Result<Excitation, SimError> {
    let bad = |m: &str| SimError::new(SimErrorKind::Excitation, m);
    let n = rec.accel.len().min(rec.gyro.len());
    if n < 2 || !(rec.dt_s > 0.0 && rec.dt_s.is_finite()) {
        return Err(bad("the recording needs at least two IMU0 samples on a uniform grid"));
    }
    let up = unit(mount.up).ok_or_else(|| bad("the calibration's up direction is not a direction"))?;
    // forward made square to up, then the right-hand side completes the frame
    let f = mount.forward;
    let k = dot(f, up);
    let fwd = unit([f[0] - k * up[0], f[1] - k * up[1], f[2] - k * up[2]]).ok_or_else(|| bad("the calibration's forward direction lies along up"))?;
    let right = [fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2], fwd[0] * up[1] - fwd[1] * up[0]];
    let b = mount.gyro_bias_rad_s;

    let mut fx: Vec<f64> = rec.accel[..n].iter().map(|a| dot(*a, fwd)).collect();
    let mut fy: Vec<f64> = rec.accel[..n].iter().map(|a| dot(*a, up)).collect();
    let mut omega: Vec<f64> = rec.gyro[..n].iter().map(|g| dot([g[0] - b[0], g[1] - b[1], g[2] - b[2]], right)).collect();
    let rate = 1.0 / rec.dt_s;
    if cutoff_hz > 0.0 && cutoff_hz < 0.45 * rate && n > 30 {
        for v in [&mut fx, &mut fy, &mut omega] {
            if let Ok(y) = crate::filters::lowpass(v, 4, cutoff_hz, rate) {
                *v = y;
            }
        }
    }

    let t_abs = |i: usize| rec.t0_s + i as f64 * rec.dt_s;
    let travel_m = (0..n)
        .map(|i| interp(rec.travel, t_abs(i)).map(|s| s.max(0.0)))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| bad("no solved rear travel in this stretch (the estimator solves only while riding)"))?;
    let speed_mps = (0..n)
        .map(|i| interp(rec.speed, t_abs(i)).map(|v| v.max(0.0)))
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| bad("no GPS speed in this stretch"))?;
    let exc = Excitation { t_s: (0..n).map(|i| i as f64 * rec.dt_s).collect(), travel_m, fx, fy, omega, speed_mps };
    exc.validate()?;
    Ok(exc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::excite::G;

    fn level() -> Mount {
        Mount { up: [0.0, 0.0, 1.0], forward: [1.0, 0.0, 0.0], gyro_bias_rad_s: [0.0; 3] }
    }

    #[test]
    fn excitation_level_mount_at_rest_reads_g_up_and_nothing_forward() {
        // Arrange
        let accel = vec![[0.0, 0.0, G]; 100];
        let gyro = vec![[0.0; 3]; 100];
        let rec = Recording { t0_s: 5.0, dt_s: 0.001, accel: &accel, gyro: &gyro, travel: &[(5.0, 0.04)], speed: &[(5.0, 3.0)] };

        // Act
        let e = excitation(&rec, &level(), DEFAULT_CUTOFF_HZ).unwrap();

        // Assert
        assert!((e.fy[50] - G).abs() < 1e-9 && e.fx[50].abs() < 1e-9);
        assert_eq!(e.t_s[0], 0.0);
        assert!((e.travel_m[99] - 0.04).abs() < 1e-12 && (e.speed_mps[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn excitation_turned_mount_forward_along_y_projects_y_into_fx() {
        // Arrange: IMU0 turned 90° about up, and a forward slightly tilted off level
        let m = Mount { up: [0.0, 0.0, 1.0], forward: [0.0, 1.0, 0.1], gyro_bias_rad_s: [0.0; 3] };
        let accel = vec![[0.0, 2.0, G]; 10];
        let gyro = vec![[0.0; 3]; 10];
        let rec = Recording { t0_s: 0.0, dt_s: 0.01, accel: &accel, gyro: &gyro, travel: &[(0.0, 0.0)], speed: &[(0.0, 1.0)] };

        // Act
        let e = excitation(&rec, &m, 0.0).unwrap();

        // Assert
        assert!((e.fx[3] - 2.0).abs() < 1e-9 && (e.fy[3] - G).abs() < 1e-9, "{} {}", e.fx[3], e.fy[3]);
    }

    #[test]
    fn excitation_pitch_rate_about_the_right_side_less_bias_is_nose_up_positive() {
        // Arrange: x forward, z up, so the right side is -y; nose up turns about -y
        let m = Mount { gyro_bias_rad_s: [0.0, 0.1, 0.0], ..level() };
        let accel = vec![[0.0, 0.0, G]; 10];
        let gyro = vec![[0.0, -0.9, 0.0]; 10];
        let rec = Recording { t0_s: 0.0, dt_s: 0.01, accel: &accel, gyro: &gyro, travel: &[(0.0, 0.0)], speed: &[(0.0, 1.0)] };

        // Act
        let e = excitation(&rec, &m, 0.0).unwrap();

        // Assert
        assert!((e.omega[4] - 1.0).abs() < 1e-12, "{}", e.omega[4]);
    }

    #[test]
    fn excitation_travel_bridges_an_unsolved_gap_and_never_goes_negative() {
        // Arrange
        let accel = vec![[0.0, 0.0, G]; 11];
        let gyro = vec![[0.0; 3]; 11];
        let travel = [(0.0, -0.01), (0.05, f64::NAN), (0.1, 0.07)];
        let rec = Recording { t0_s: 0.0, dt_s: 0.01, accel: &accel, gyro: &gyro, travel: &travel, speed: &[(0.0, 1.0), (0.1, 2.0)] };

        // Act
        let e = excitation(&rec, &level(), 0.0).unwrap();

        // Assert
        assert!((e.travel_m[5] - 0.03).abs() < 1e-9, "{}", e.travel_m[5]);
        assert!((e.speed_mps[5] - 1.5).abs() < 1e-9);
        assert_eq!(e.travel_m[0], 0.0);
    }

    #[test]
    fn excitation_without_solved_travel_or_speed_or_forward_is_an_error() {
        // Arrange
        let accel = vec![[0.0, 0.0, G]; 10];
        let gyro = vec![[0.0; 3]; 10];
        let rec = |travel: &'static [(f64, f64)], speed: &'static [(f64, f64)]| Recording { t0_s: 0.0, dt_s: 0.01, accel: &accel, gyro: &gyro, travel, speed };
        let along_up = Mount { forward: [0.0, 0.0, 2.0], ..level() };

        // Act
        let no_travel = excitation(&rec(&[(0.0, f64::NAN)], &[(0.0, 1.0)]), &level(), 0.0);
        let no_speed = excitation(&rec(&[(0.0, 0.01)], &[]), &level(), 0.0);
        let no_forward = excitation(&rec(&[(0.0, 0.01)], &[(0.0, 1.0)]), &along_up, 0.0);

        // Assert
        assert!(no_travel.is_err() && no_speed.is_err() && no_forward.is_err());
    }

    #[test]
    fn excitation_low_pass_takes_out_vibration_above_the_cut_off() {
        // Arrange: 1 kHz, 300 Hz buzz on top of g
        let accel: Vec<[f64; 3]> = (0..2000).map(|i| [0.0, 0.0, G + 5.0 * (std::f64::consts::TAU * 300.0 * i as f64 / 1000.0).sin()]).collect();
        let gyro = vec![[0.0; 3]; 2000];
        let rec = Recording { t0_s: 0.0, dt_s: 0.001, accel: &accel, gyro: &gyro, travel: &[(0.0, 0.0)], speed: &[(0.0, 1.0)] };

        // Act
        let e = excitation(&rec, &level(), DEFAULT_CUTOFF_HZ).unwrap();

        // Assert
        let worst = e.fy[200..1800].iter().map(|v| (v - G).abs()).fold(0.0, f64::max);
        assert!(worst < 0.1, "{worst}");
    }
}
