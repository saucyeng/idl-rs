//! Runs the estimator over one library session and prints the numbers used to
//! check it for regressions (see `src/estimate/README.md`): the measured GPS
//! lateness and forward trim, roll against the lean GPS implies, heading against
//! GPS course, steering against the bar angle a non-sliding bike needs, and the
//! understeer angle.
//!
//!   cargo run -q --release -p idl-rs --example estimator_regression -- \
//!       <library>/sessions/<session_id> [--bike <library>/bikes/<bike_id>.json]
//!
//! With `--bike`, the session gets that bike's calibration by the same rule the
//! CLI and the app use (`calibration_for`: the newest captured at or before the
//! session, else the oldest after it) and the same geometry and tuning a
//! workbook gets (`session_setup`). Without it the session runs uncalibrated,
//! whatever bike it names.
//!
//! Reads only; the library is not touched. A full Beacon ride takes about
//! 100 s in release, Walden about 9 minutes.
//!
//! How each number is measured. These are the definitions the reference
//! numbers in the README were taken with; change one and old and new numbers
//! stop being comparable.
//!
//! - The outputs are read every 20 ms. Output sample `i` is at
//!   `t0_s + i·dt` on the recording's timeline: the outputs start at IMU0's
//!   first sample, which is not 0 on every ride.
//! - Every output is read at a GPS fix's time **less the fitted lateness**: a
//!   fix describes the bike that long ago.
//! - **Roll vs GPS lean**: at each fix moving faster than [`MIN_SPEED_MPS`],
//!   the course rate is the course change from the fix before to the fix after
//!   over their time apart, and the GPS lean is `atan(v·Ω/g)` (right turn =
//!   lean right = positive). The estimator's is its mean roll over the same
//!   two-fix span. The line is fitted where the GPS lean is over
//!   [`LEAN_FIT_MIN_DEG`]: slope of estimator roll on GPS lean, and correlation.
//! - **Heading − course**: at each fix faster than [`MIN_SPEED_MPS`], the
//!   estimator heading at the first output at or after the fix's time, less
//!   the GPS course, wrapped to ±180°. Median and RMS.
//! - **Steering**: the estimator's steer angle and the turn rate from its
//!   heading, each through a 0.5 s moving average, give the bar angle needed
//!   ([`kinematic_steer`], with the GPS speed and the estimator's roll); taken
//!   where the GPS speed is over [`STEER_MIN_SPEED_MPS`] and the heading and
//!   steer are known, every fifth output (10 Hz). Slope of steer on needed,
//!   correlation, and the RMS of steer about that line ("scatter").
//! - **Understeer**: the median of the estimator's `understeer` output over
//!   the outputs where it is known.

use idl_rs::calibration::bike::{calibration_for, BikeCalibration};
use idl_rs::estimate::process::GRAVITY;
use idl_rs::estimate::run::{kinematic_steer, run, session_setup, EstimatorInput, GpsSample};
use std::path::PathBuf;
use std::process::ExitCode;

/// GPS speed above which a fix is scored for roll and heading, m/s.
const MIN_SPEED_MPS: f64 = 4.0;
/// Smallest GPS lean, either way, the roll line is fitted over, degrees.
const LEAN_FIT_MIN_DEG: f64 = 5.0;
/// GPS speed above which steering is scored, m/s.
const STEER_MIN_SPEED_MPS: f64 = 3.0;
/// Interval the outputs are read at, s.
const READ_EVERY_S: f64 = 0.02;
/// Moving-average length for the steering comparison, s.
const STEER_SMOOTH_S: f64 = 0.5;

fn main() -> ExitCode {
    match main_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("estimator_regression: {message}");
            ExitCode::FAILURE
        }
    }
}

fn main_inner() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut session_dir: Option<PathBuf> = None;
    let mut bike_file: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bike" => bike_file = Some(args.next().ok_or("--bike needs a path")?.into()),
            _ if session_dir.is_none() => session_dir = Some(a.into()),
            _ => return Err(format!("unexpected argument {a:?}")),
        }
    }
    let session_dir = session_dir.ok_or("usage: estimator_regression <session dir> [--bike <bike file>]")?;
    let session_id = session_dir.file_name().map_or(String::new(), |s| s.to_string_lossy().into_owned());

    let handle = idl_rs::store::parquet::open_session_lazy(&session_dir)
        .map_err(|e| format!("reading {}: {}", session_dir.display(), e.message))?;

    let calibration = match &bike_file {
        None => None,
        Some(path) => {
            let json = std::fs::read(session_dir.join("session.json")).map_err(|e| format!("reading session.json: {e}"))?;
            let session = idl_rs::store::session_json::parse_session_json(&json).map_err(|e| format!("parsing session.json: {e:?}"))?;
            let start_ms = idl_rs::store::session_json::effective_start_ms(&session, handle.metadata().timestamp_utc_ms);
            let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
            let file: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("parsing {}: {e}", path.display()))?;
            let all: Vec<BikeCalibration> = serde_json::from_value(file.get("calibrations").cloned().unwrap_or_default())
                .map_err(|e| format!("reading the calibrations in {}: {e}", path.display()))?;
            let (chosen, after) = calibration_for(&all, start_ms).ok_or_else(|| format!("{} holds no calibration", path.display()))?;
            Some((chosen.clone(), after))
        }
    };

    let input = EstimatorInput::from_lookup(&handle).ok_or("the session has no IMU0")?;
    let (geometry, config) = session_setup(calibration.as_ref().map(|(c, _)| &c.fit));
    let started = std::time::Instant::now();
    let est = run(&input, &geometry, &config);
    let took = started.elapsed().as_secs_f64();

    let row = |name: &str, value: String| println!("{name:<26}{value}");
    let opt = |v: Option<f64>, digits: usize| v.map_or("-".to_string(), |x| format!("{x:.digits$}"));
    row("session", session_id);
    row(
        "calibration",
        match &calibration {
            Some((c, after)) => format!("{}{}", c.calibration_id, if *after { " (captured after the session)" } else { "" }),
            None => "none".to_string(),
        },
    );
    row("samples", format!("{} at {:.1} Hz, {:.0} s, first at {:.3} s", est.len(), 1.0 / est.dt, est.len() as f64 * est.dt, input.t0_s));
    row("run time", format!("{took:.1} s"));
    row("gps fixes", input.gps.len().to_string());
    row("gps lateness s", format!("{:.2}{}", est.gps_latency_s, if est.gps_latency_fitted { "" } else { " (assumed, not fitted)" }));
    row(
        "forward trim deg",
        format!("{}{}", opt(est.forward_trim_deg, 1), if est.forward_trim_applied { " (applied)" } else { " (measured, not applied)" }),
    );
    row("imu0 riding check deg", opt(est.imu0_riding_check_deg, 1));
    row("steer zero deg", opt(est.steer_zero_deg, 1));

    // The outputs every READ_EVERY_S, on the recording's timeline.
    let step = ((READ_EVERY_S / est.dt).round() as usize).max(1);
    let pick = |v: &[f64]| v.iter().step_by(step).copied().collect::<Vec<f64>>();
    let t: Vec<f64> = (0..est.len()).step_by(step).map(|i| input.t0_s + i as f64 * est.dt).collect();
    let (roll, heading, steer, understeer) = (pick(&est.roll), pick(&est.heading), pick(&est.steer), pick(&est.understeer));
    // Fixes, at the time they describe.
    let fixes: Vec<Fix> = input.gps.iter().map(|s| Fix::of(s, est.gps_latency_s)).collect();

    let (gps_lean, est_roll) = roll_against_gps_lean(&fixes, &t, &roll);
    match fit_line(&gps_lean, &est_roll) {
        Some(f) => row("roll vs gps lean", format!("slope {:.2}  corr {:.2}  (n {})", f.slope, f.corr, f.n)),
        None => row("roll vs gps lean", "-".to_string()),
    }

    let mut heading_err: Vec<f64> = fixes
        .iter()
        .filter(|f| f.speed > MIN_SPEED_MPS)
        .filter_map(|f| {
            let h = heading[t.partition_point(|&x| x < f.t).min(t.len() - 1)];
            h.is_finite().then(|| (h - f.course_deg + 180.0).rem_euclid(360.0) - 180.0)
        })
        .collect();
    if heading_err.is_empty() {
        row("heading - course deg", "-".to_string());
    } else {
        let rms = (heading_err.iter().map(|e| e * e).sum::<f64>() / heading_err.len() as f64).sqrt();
        let n = heading_err.len();
        row("heading - course deg", format!("median {:.2}  rms {:.2}  (n {n})", median(&mut heading_err), rms));
    }

    match geometry.wheelbase {
        Some(wheelbase) if steer.iter().any(|s| s.is_finite()) => {
            let caster = geometry.steer_axis.z.clamp(-1.0, 1.0).acos();
            let (needed, measured) = steer_against_needed(&fixes, &t, &heading, &steer, &roll, wheelbase, caster);
            match fit_line(&needed, &measured) {
                Some(f) => row("steer vs needed", format!("slope {:.2}  corr {:.2}  scatter {:.1} deg  (n {})", f.slope, f.corr, f.scatter, f.n)),
                None => row("steer vs needed", "-".to_string()),
            }
        }
        _ => row("steer vs needed", "- (no steering, or no wheelbase)".to_string()),
    }
    let mut under: Vec<f64> = understeer.into_iter().filter(|u| u.is_finite()).collect();
    row("understeer median deg", if under.is_empty() { "-".to_string() } else { format!("{:.2}", median(&mut under)) });
    Ok(())
}

/// One GPS fix at the time it describes (logged time less the lateness), s.
struct Fix {
    t: f64,
    speed: f64,
    course_deg: f64,
}

impl Fix {
    fn of(s: &GpsSample, latency_s: f64) -> Fix {
        // Nav frame is X north, Y west; course is clockwise from north.
        Fix { t: s.time_s - latency_s, speed: s.velocity.xy().norm(), course_deg: (-s.velocity.y).atan2(s.velocity.x).to_degrees() }
    }
}

/// `(GPS lean, estimator mean roll)`, degrees, for each fix the roll line is fitted over.
fn roll_against_gps_lean(fixes: &[Fix], t: &[f64], roll: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let course = unwrap_deg(&fixes.iter().map(|f| f.course_deg).collect::<Vec<_>>());
    // Running sum of roll, for the mean over a span.
    let mut sum = vec![0.0];
    for r in roll {
        sum.push(sum.last().unwrap() + r);
    }
    let (first, last) = (t[0], t[t.len() - 1]);
    let (mut lean, mut est) = (Vec::new(), Vec::new());
    for k in 1..fixes.len().saturating_sub(1) {
        let (before, here, after) = (&fixes[k - 1], &fixes[k], &fixes[k + 1]);
        let span = after.t - before.t;
        if !(span > 0.0 && here.speed > MIN_SPEED_MPS && here.t > first && here.t < last) {
            continue;
        }
        let rate = (course[k + 1] - course[k - 1]).to_radians() / span;
        let gps = (here.speed * rate / GRAVITY).atan().to_degrees();
        let (i, j) = (t.partition_point(|&x| x < before.t), t.partition_point(|&x| x < after.t));
        if j <= i + 5 || gps.abs() <= LEAN_FIT_MIN_DEG {
            continue;
        }
        let mean = (sum[j] - sum[i]) / (j - i) as f64;
        if mean.is_finite() {
            lean.push(gps);
            est.push(mean);
        }
    }
    (lean, est)
}

/// `(bar angle needed, measured steer)`, degrees, at 10 Hz over the outputs where steering is scored.
fn steer_against_needed(fixes: &[Fix], t: &[f64], heading: &[f64], steer: &[f64], roll: &[f64], wheelbase: f64, caster: f64) -> (Vec<f64>, Vec<f64>) {
    let n = t.len();
    let dt = READ_EVERY_S;
    let k = ((STEER_SMOOTH_S / dt).round() as usize).max(1);
    // Turn rate from the heading, left positive (heading is clockwise), rad/s.
    let unwrapped: Vec<f64> = unwrap_deg(&heading.iter().map(|h| if h.is_finite() { *h } else { 0.0 }).collect::<Vec<_>>());
    let clockwise: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
            if b > a { (unwrapped[b] - unwrapped[a]).to_radians() / (t[b] - t[a]) } else { 0.0 }
        })
        .collect();
    let left_rate: Vec<f64> = moving_average(&clockwise, k).into_iter().map(|r| -r).collect();
    let steer_avg = moving_average(&steer.iter().map(|s| if s.is_finite() { *s } else { 0.0 }).collect::<Vec<_>>(), k);
    let speed = gps_speed_at(fixes, t);
    let (mut needed, mut measured) = (Vec::new(), Vec::new());
    let scored: Vec<usize> = (0..n).filter(|&i| heading[i].is_finite() && steer[i].is_finite() && speed[i] > STEER_MIN_SPEED_MPS).collect();
    for &i in scored.iter().step_by(5) {
        let (need, _) = kinematic_steer(wheelbase, left_rate[i], speed[i].max(0.1), roll[i].to_radians(), caster, 0.0);
        needed.push(need.to_degrees());
        measured.push(steer_avg[i]);
    }
    (needed, measured)
}

/// GPS speed at each time in `t`, linear between fixes and held beyond the ends, m/s.
fn gps_speed_at(fixes: &[Fix], t: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&x| {
            let k = fixes.partition_point(|f| f.t < x);
            match (k.checked_sub(1).map(|j| &fixes[j]), fixes.get(k)) {
                (Some(a), Some(b)) if b.t > a.t => a.speed + (x - a.t) / (b.t - a.t) * (b.speed - a.speed),
                (Some(a), _) => a.speed,
                (None, Some(b)) => b.speed,
                (None, None) => f64::NAN,
            }
        })
        .collect()
}

/// Centred moving average over `k` samples, the ends averaged over what is there
/// as if zero-padded (numpy's `convolve(..., mode='same')` with a box of 1/k).
fn moving_average(v: &[f64], k: usize) -> Vec<f64> {
    let n = v.len();
    let mut sum = vec![0.0; n + 1];
    for i in 0..n {
        sum[i + 1] = sum[i] + v[i];
    }
    let before = (k - 1) / 2;
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(k - 1 - before);
            let hi = (i + before + 1).min(n);
            (sum[hi] - sum[lo]) / k as f64
        })
        .collect()
}

/// Degrees with each step folded into ±180°, so a series that crosses north is continuous.
fn unwrap_deg(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        out.push(if i == 0 { x } else { out[i - 1] + (x - v[i - 1] + 180.0).rem_euclid(360.0) - 180.0 });
    }
    out
}

/// A least-squares line of `y` on `x`.
struct Line {
    slope: f64,
    corr: f64,
    /// RMS of `y` about the line.
    scatter: f64,
    n: usize,
}

fn fit_line(x: &[f64], y: &[f64]) -> Option<Line> {
    let n = x.len().min(y.len());
    if n < 3 {
        return None;
    }
    let (mx, my) = (x.iter().sum::<f64>() / n as f64, y.iter().sum::<f64>() / n as f64);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let residual = (syy - slope * sxy).max(0.0) / n as f64;
    Some(Line { slope, corr: sxy / (sxx * syy).sqrt(), scatter: residual.sqrt(), n })
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
