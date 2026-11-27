//! Runs the physics estimator (`idl_rs::estimate::physics`) over a library
//! session: the CLI's `session estimate-physics`.
//!
//! The session's bike gives the calibration (the one [`calibration_for`]
//! picks for the session's start), the linkage and the travel. The bike's
//! geometry is fitted from this session the first time (or with `refit`) and
//! saved into the calibration; then every stretch of riding is solved and the
//! channels are written to `sessions/<id>/derived/<hash>.parquet` (contract
//! C1 §5, `derived_kind` `"physics"`), content-addressed by the IMU channels,
//! the configuration and the engine version, so a new estimator version or a
//! new fit gives a new file and an older one is never mistaken for current.
//!
//! [`calibration_for`]: idl_rs::calibration::bike::calibration_for

use std::path::{Path, PathBuf};
use std::time::Instant;

use idl_rs::calibration::linkage::{PhysicsFit, PHYSICS_FIT_VERSION};
use idl_rs::estimate::physics::objective::NG;
use idl_rs::estimate::physics::pipeline::{ride_between, still_bias_centre, PipelineError, Ride};
pub use idl_rs::estimate::physics::run::SolveProgress;
use idl_rs::estimate::physics::run::{bike_geometry, fit_bike, fitted_bike, outputs, solve_ride_with, Config, PHYSICS_ESTIMATOR_VERSION};
use idl_rs::estimate::physics::setup::{bike, SetupError};
use idl_rs::estimate::physics::solve::SolveError;
use idl_rs::math::eval::ChannelLookup;
use idl_rs::store::parquet::open_session_lazy;
use idl_rs::store::session_json::{effective_start_ms, read_session_json};

use crate::bikelog::{calibration_for_session, is_valid_id, read_bike_file, save_calibration};
use crate::derived::{derived_input_hash, write_derived_parquet, DerivedOutput};

/// What a run may do.
#[derive(Debug, Clone, Default)]
pub struct PhysicsOptions {
    /// Fit the bike's geometry again even if the calibration has a current fit.
    pub refit: bool,
    /// Fork travel, m; overrides the bike's linkage record.
    pub front_travel_m: Option<f64>,
    /// Rear travel, m; overrides the bike's linkage record.
    pub rear_travel_m: Option<f64>,
    /// Worker threads (None: up to 4).
    pub threads: Option<usize>,
    /// Report what would be done; write nothing.
    pub dry_run: bool,
    /// Only this stretch of the session, ms from its first sample (both or
    /// neither): less memory, and the bike fit picks its windows inside it.
    pub span_ms: Option<(i64, i64)>,
    /// With `span_ms`, a bike fit (when one is needed) still picks its
    /// windows from the whole session; only the solve is cut to the span.
    pub fit_whole_session: bool,
    /// Estimate attitude and velocity too (the `Physics roll (deg)`... channels;
    /// spec one-model). Off while it is checked against the IEKF.
    pub nav: bool,
    /// Run the force model too (spec force-model F1: tyre loads, suspension forces, the
    /// rider's wrench; the dampers and the rider identified on a few windows first). Needs
    /// the bike's linkage with its shock eyes, masses and pressures.
    pub forces: bool,
    /// Parameters on top of the bike's at the ride's date, for this run only (never saved):
    /// a setup the log does not hold yet, or a what-if.
    pub param_overrides: idl_rs::calibration::params::Params,
}

/// Distribution of one travel channel over the solved samples, mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TravelSummary {
    /// 1st percentile.
    pub p01: f64,
    /// Median.
    pub median: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Median over the airborne (anchored) samples; NaN with none.
    pub airborne_median: f64,
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct PhysicsReport {
    /// The session.
    pub session_id: String,
    /// Its bike.
    pub bike_id: String,
    /// The calibration used.
    pub calibration_id: String,
    /// The geometry fit made by this run (None: the calibration's was used).
    pub new_fit: Option<PhysicsFit>,
    /// The geometry the solve ran with: steer point (m, 3), bending length (m), linkage scale.
    pub geometry: [f64; 5],
    /// Solve windows, and how many failed.
    pub windows: (usize, usize),
    /// Seconds of ride solved.
    pub solved_s: f64,
    /// Front and rear travel over the solved samples.
    pub travel: [TravelSummary; 2],
    /// The derived file written (None on a dry run).
    pub output: Option<PathBuf>,
    /// Grid times, us since the session's first sample.
    pub grid_us: Vec<i64>,
    /// Rear travel per grid sample, m (NaN where not solved).
    pub rear_travel_m: Vec<f64>,
    /// With [`PhysicsOptions::forces`]: the identified force parameters, as
    /// (name, log multiplier on the model's nominal value).
    pub force_par: Option<Vec<(String, f64)>>,
    /// With [`PhysicsOptions::forces`]: each force channel's spread over the solved samples
    /// (in its own unit; the summary's fields read as that unit, not mm).
    pub force_summary: Vec<(String, TravelSummary)>,
    /// Wall time, s.
    pub seconds: f64,
}

/// Why a run failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsErrorKind {
    /// The session, its bike or its bike's file is not there.
    NotFound,
    /// The bike has no calibration, or no travel was given.
    NotCalibrated,
    /// The session or the bike cannot drive the estimator.
    Unusable,
    /// The solver failed.
    Solve,
    /// Reading or writing failed.
    Io,
    /// Stopped through [`SolveProgress::cancel`]; nothing was written.
    Cancelled,
}

/// A run's failure, typed (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsError {
    /// What kind.
    pub kind: PhysicsErrorKind,
    /// Detail for a person.
    pub message: String,
}

impl std::fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for PhysicsError {}

fn err(kind: PhysicsErrorKind, message: impl Into<String>) -> PhysicsError {
    PhysicsError { kind, message: message.into() }
}

/// Wall-clock UTC ms.
pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// The GPS channels navigation reads (`estimate::run::gps_samples_from_lookup`).
const GPS_CHANNELS: [&str; 4] = ["GPS_SpeedKmh", "GPS_Heading", "GPS_FixQuality", "GPS_Altitude"];

const IMU_CHANNELS: [&str; 18] = [
    "IMU0_AccelX", "IMU0_AccelY", "IMU0_AccelZ", "IMU0_GyroX", "IMU0_GyroY", "IMU0_GyroZ",
    "IMU1_AccelX", "IMU1_AccelY", "IMU1_AccelZ", "IMU1_GyroX", "IMU1_GyroY", "IMU1_GyroZ",
    "IMU2_AccelX", "IMU2_AccelY", "IMU2_AccelZ", "IMU2_GyroX", "IMU2_GyroY", "IMU2_GyroZ",
];

fn summary(r: &Ride, values: &[f64], air: &[bool]) -> TravelSummary {
    let mut v: Vec<f64> = values.iter().cloned().filter(|x| x.is_finite()).collect();
    let mut a: Vec<f64> = values.iter().zip(air).filter(|(x, &m)| m && x.is_finite()).map(|(x, _)| *x).collect();
    let pct = |v: &mut Vec<f64>, p: f64| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    };
    let _ = r;
    TravelSummary { p01: pct(&mut v, 0.01), median: pct(&mut v, 0.5), p99: pct(&mut v, 0.99), airborne_median: pct(&mut a, 0.5) }
}

/// Runs the physics estimator over `session_id` (see the module docs).
pub fn estimate_physics(data_root: &Path, session_id: &str, opts: &PhysicsOptions) -> Result<PhysicsReport, PhysicsError> {
    estimate_physics_with(data_root, session_id, opts, None)
}

/// [`estimate_physics`], telling `progress` about each solved window and
/// stopping when its `cancel` is set: then nothing is written and the result
/// is [`PhysicsErrorKind::Cancelled`]. A bike fit, when one is needed, runs
/// before the first window is reported and is not interrupted.
pub fn estimate_physics_with(
    data_root: &Path,
    session_id: &str,
    opts: &PhysicsOptions,
    progress: Option<&SolveProgress>,
) -> Result<PhysicsReport, PhysicsError> {
    let t0 = Instant::now();
    let dir = data_root.join("sessions").join(session_id);
    let doc = read_session_json(&dir.join("session.json")).map_err(|e| err(PhysicsErrorKind::NotFound, format!("{session_id}: {e:?}")))?;
    if !is_valid_id(&doc.bike_id) {
        return Err(err(PhysicsErrorKind::NotCalibrated, format!("{session_id} names no bike")));
    }
    let handle = open_session_lazy(&dir).map_err(|e| err(PhysicsErrorKind::Io, e.to_string()))?;
    let start_ms = effective_start_ms(&doc, handle.metadata().timestamp_utc_ms);
    let (cal, _after) = calibration_for_session(data_root, &doc, start_ms)
        .ok_or_else(|| err(PhysicsErrorKind::NotCalibrated, format!("{} has no calibration", doc.bike_id)))?;
    let file = read_bike_file(data_root, &doc.bike_id).map_err(|e| err(PhysicsErrorKind::NotFound, e.to_string()))?;
    let linkage = file.linkage.as_ref();
    let front = opts.front_travel_m.or(linkage.and_then(|l| l.front_travel_m));
    let rear = opts.rear_travel_m.or(linkage.map(|l| l.rear_travel_m));
    let (Some(front), Some(rear)) = (front, rear) else {
        return Err(err(PhysicsErrorKind::NotCalibrated, format!("{}: fork and rear travel unknown; give --front-travel-mm and --rear-travel-mm or a linkage", doc.bike_id)));
    };

    // a margin each side, so the stretch's own ends are not window edges
    let span = opts.span_ms.map(|(a, b)| ((a - 2000).max(0) * 1000, (b + 2000) * 1000));
    let r = ride_between(&handle, span).map_err(|e| {
        err(
            PhysicsErrorKind::Unusable,
            match e {
                PipelineError::NoFrameImu => format!("{session_id} has no IMU0"),
                PipelineError::MissingPod { imu_index } => format!("{session_id} has no IMU{imu_index}"),
            },
        )
    })?;
    let current = cal.physics.as_ref().filter(|p| p.version == PHYSICS_FIT_VERSION && !opts.refit);
    // a fit from the whole session while the solve takes only the span
    let fit_ride = if current.is_none() && opts.fit_whole_session && span.is_some() {
        Some(ride_between(&handle, None).map_err(|_| err(PhysicsErrorKind::Unusable, format!("{session_id} cannot be fitted: an IMU is missing")))?)
    } else {
        None
    };
    // the inputs' column hashes (C1 §5) now, then drop the session: its decoded
    // channels would otherwise sit in memory through every solve
    let gps_inputs: &[&str] = if opts.nav { &GPS_CHANNELS } else { &[] };
    let inputs: Vec<(String, [u8; 32])> = IMU_CHANNELS
        .iter()
        .chain(gps_inputs)
        .filter_map(|&name| handle.lookup(name).map(|c| (name.to_string(), derived_input_hash(name, &c.t_us, &c.samples))))
        .collect();
    // IDL_PHYSICS_LAPS (prototyping): solve only the session's laps, its descents; a lap number
    // (IDL_PHYSICS_LAPS=3) keeps that lap alone
    let only_us: Option<Vec<(i64, i64)>> = std::env::var("IDL_PHYSICS_LAPS").ok().map(|v| {
        let one: Option<u32> = v.trim().parse().ok();
        let epochs: Vec<f64> = doc.laps.iter().filter(|l| one.is_none_or(|n| l.lap_number == n)).flat_map(|l| [l.start_timestamp_ms as f64, l.end_timestamp_ms as f64]).collect();
        let secs = handle.epoch_ms_to_time_secs(&epochs);
        secs.chunks(2).map(|p| ((p[0] * 1e6) as i64, (p[1] * 1e6) as i64)).collect()
    });
    if let Some(sp) = &only_us {
        eprintln!("IDL_PHYSICS_LAPS: {} laps, {:.0} s", sp.len(), sp.iter().map(|(a, b)| (b - a) as f64 * 1e-6).sum::<f64>());
    }
    drop(handle);
    // at-rest biases from the calibration's still recording, else IMU0's from the calibration
    let mut centre = [0.0; NG];
    centre[..3].copy_from_slice(&cal.fit.imu0_gyro_bias_rad_s);
    if let Ok(still) = open_session_lazy(&data_root.join("sessions").join(&cal.still.session_id)) {
        if let Some(c) = still_bias_centre(&still as &dyn ChannelLookup) {
            centre = c;
        }
    }

    let mut cfg = Config::default();
    cfg.weights.travel_max = [front, rear];
    if let Some(t) = opts.threads {
        cfg.threads = t.max(1);
    }
    cfg.nav = opts.nav;
    cfg.only_us = only_us;
    // IDL_FORCE_ID_ITERS (prototyping): the force identification's iteration cap
    if let Some(n) = std::env::var("IDL_FORCE_ID_ITERS").ok().and_then(|v| v.trim().parse().ok()) {
        cfg.force_id_iters = n;
    }
    let force_setup = |bk: &idl_rs::estimate::physics::bike::Bike<f64>| -> Result<Option<idl_rs::estimate::physics::forces::ForceModel>, PhysicsError> {
        if !opts.forces {
            return Ok(None);
        }
        let l = linkage.ok_or_else(|| err(PhysicsErrorKind::NotCalibrated, format!("{}: the force model needs the bike's linkage", doc.bike_id)))?;
        let params = idl_rs::calibration::params::overlay(&crate::bikelog::effective_parameters(&file, start_ms), &opts.param_overrides);
        idl_rs::estimate::physics::force_setup::force_model(bk, &params, l, file.drivetrain.as_ref(), [front, rear])
            .map(Some)
            .map_err(|e| err(PhysicsErrorKind::NotCalibrated, format!("{}: the force model cannot be set up: {e:?}", doc.bike_id)))
    };
    let base = bike(&cal.fit, linkage, current, rear).map_err(|e| {
        err(
            PhysicsErrorKind::Unusable,
            match e {
                SetupError::NoForward => "the calibration has no forward direction".to_string(),
                SetupError::MissingPod { imu_index } => format!("the calibration has no IMU{imu_index}"),
                SetupError::Linkage(l) => format!("the linkage cannot be used: {l:?}"),
            },
        )
    })?;
    let solve_err = |e: SolveError| err(PhysicsErrorKind::Solve, format!("{e:?}"));
    let (bk, new_fit) = if current.is_some() {
        (base, None)
    } else {
        let fr = fit_ride.as_ref().unwrap_or(&r);
        let rep = fit_bike(fr, &base, centre, &cfg, &[]).map_err(solve_err)?;
        let fit = PhysicsFit {
            version: PHYSICS_FIT_VERSION,
            fitted_at_ms: now_ms(),
            steer_point: [rep.geometry[0], rep.geometry[1], rep.geometry[2]],
            bend_len_m: rep.geometry[3],
            link_scale: rep.geometry[4],
            session_id: session_id.to_string(),
            windows_s: rep.windows.iter().map(|&(a, b)| [fr.grid_us[a] as f64 / 1e6, fr.grid_us[b - 1] as f64 / 1e6]).collect(),
            iterations: rep.info.iterations as u32,
            converged: rep.info.converged,
        };
        if !opts.dry_run {
            let mut saved = cal.clone();
            saved.physics = Some(fit.clone());
            saved.updated_at_ms = now_ms().max(cal.updated_at_ms + 1);
            save_calibration(data_root, &doc.bike_id, &saved).map_err(|e| err(PhysicsErrorKind::Io, e.to_string()))?;
        }
        (fitted_bike(&base, &rep), Some(fit))
    };

    drop(fit_ride);
    cfg.forces = force_setup(&bk)?;
    // IDL_FORCE_ACTIVE_HZ (prototyping): the rider's active wrench's band, Hz, in place of the
    // standard 3 Hz (its smoothness weight scales as the band to the -4th)
    if let (Some(m), Some(hz)) = (cfg.forces.as_mut(), std::env::var("IDL_FORCE_ACTIVE_HZ").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|h| *h > 0.0)) {
        use idl_rs::estimate::physics::forces::{ACTIVE_SMOOTH, ACT_F, ACT_M};
        for i in 0..3 {
            m.spec.smooth[ACT_F + i] = ACTIVE_SMOOTH * (3.0 / hz).powi(4);
            m.spec.smooth[ACT_M + i] = ACTIVE_SMOOTH * (3.0 / hz).powi(4);
        }
        eprintln!("IDL_FORCE_ACTIVE_HZ: the rider's active wrench to {hz} Hz (smoothness {:.0})", ACTIVE_SMOOTH * (3.0 / hz).powi(4));
    }
    // IDL_FORCE_FREEZE (prototyping): force parameters held at their nominal, by name or part of
    // one, comma separated (IDL_FORCE_FREEZE=friction holds both seals' friction)
    if let (Some(m), Ok(v)) = (cfg.forces.as_mut(), std::env::var("IDL_FORCE_FREEZE")) {
        let keys: Vec<&str> = v.split(',').map(str::trim).filter(|k| !k.is_empty()).collect();
        for (i, name) in idl_rs::estimate::physics::forces::PAR_NAMES.iter().enumerate() {
            if keys.iter().any(|k| name.contains(k)) {
                m.w.par_sigma[i] = idl_rs::estimate::physics::run::FROZEN_SIGMA;
                eprintln!("IDL_FORCE_FREEZE: {name} held at its nominal");
            }
        }
    }
    if let (Some(m), Some(_)) = (&cfg.forces, std::env::var_os("IDL_FORCE_DEBUG")) {
        eprintln!("force model static misfit (N, N m): {:?}", idl_rs::estimate::physics::forces::static_misfit(m, &bk).map(|x| x.round()));
        eprintln!("force model: {:?}", m.bike);
        let rest = m.bike.travel_max[1] * 0.3;
        eprintln!("force model: rear anti-rise {:.0} % at top-out, {:.0} % at 30 % travel, {:.0} % at 70 %", idl_rs::estimate::physics::forces::anti_rise_pct(m, &bk, 0.0), idl_rs::estimate::physics::forces::anti_rise_pct(m, &bk, rest), idl_rs::estimate::physics::forces::anti_rise_pct(m, &bk, m.bike.travel_max[1] * 0.7));
    }
    let sol = solve_ride_with(&r, &bk, centre, &cfg, progress);
    if progress.is_some_and(|p| p.cancel.load(std::sync::atomic::Ordering::Relaxed)) {
        return Err(err(PhysicsErrorKind::Cancelled, format!("{session_id}: solve cancelled")));
    }
    let outs = outputs(&sol);
    let rear_travel_m: Vec<f64> = outs[1].2.iter().map(|v| v / 1000.0).collect();
    let solved = sol.q.iter().filter(|q| q[0].is_finite()).count();
    let travel = [summary(&r, &outs[0].2, &sol.air), summary(&r, &outs[1].2, &sol.air)];
    let force_ids: Vec<&str> = idl_rs::estimate::physics::run::FORCE_CHANNELS.iter().map(|c| c.0).collect();
    let force_summary = outs.iter().filter(|o| force_ids.contains(&o.0.as_str())).map(|o| (o.0.clone(), summary(&r, &o.2, &sol.air))).collect();
    let geometry = bike_geometry(&bk);
    let output = if opts.dry_run {
        None
    } else {
        let config = serde_json::json!({
            "estimator_version": PHYSICS_ESTIMATOR_VERSION,
            "calibration_id": cal.calibration_id,
            "geometry": geometry,
            "travel_m": [front, rear],
            "linkage_updated_at_ms": linkage.map(|l| l.updated_at_ms),
            "air_anchor": cfg.air_anchor,
            "nav": cfg.nav,
            "forces": cfg.forces.is_some(),
            "param_overrides": opts.param_overrides,
            "window_s": cfg.window_s,
            "overlap_s": cfg.overlap_s,
            "bias_centre": centre.to_vec(),
            "span_ms": opts.span_ms,
        });
        let derived: Vec<DerivedOutput> = outs
            .into_iter()
            .map(|(id, unit, values)| DerivedOutput { channel_id: id, t_us: r.grid_us.clone(), values, nominal_rate_hz: r.rate(), unit })
            .collect();
        Some(write_derived_parquet(data_root, session_id, "physics", &inputs, &config, &derived, now_ms()).map_err(|e| err(PhysicsErrorKind::Io, e.to_string()))?)
    };
    Ok(PhysicsReport {
        session_id: session_id.to_string(),
        bike_id: doc.bike_id.clone(),
        calibration_id: cal.calibration_id.clone(),
        new_fit,
        geometry,
        windows: sol.windows,
        solved_s: solved as f64 / r.rate(),
        travel,
        output,
        grid_us: r.grid_us.clone(),
        rear_travel_m,
        force_par: sol.force_par.map(|p| idl_rs::estimate::physics::forces::PAR_NAMES.iter().zip(p).map(|(n, x)| (n.to_string(), x)).collect()),
        force_summary,
        seconds: t0.elapsed().as_secs_f64(),
    })
}
