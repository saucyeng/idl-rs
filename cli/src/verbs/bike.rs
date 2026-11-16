//! The `bike` verbs (ruling R251): `bike list`, `bike show`, `bike note`.
//!
//! `bike note` is how the analysis agent keeps a bike's setup sheet current
//! from the rider's notes: each entry goes through `store::bikelog`'s merge,
//! so a note written here and one written on the phone meanwhile both
//! survive.

use idl_rs::time::now_ms;
use std::collections::BTreeMap;

use clap::ArgMatches;
use serde_json::{json, Value};

use idl_library::bikelog::{self, BikeLogError, BikeLogErrorKind, Entry};
use idl_library::catalog_read::{get_session, list_sessions, SessionDetail};
use idl_rs::calibration::bike::{BikeCalibration, ForwardFrom, SourceWindow};
use idl_rs::commands::calibration_ops::{fit_bike_calibration, ride_trims, ProtocolWindows};

use super::{many, opt_integer, opt_text, text, Ctx, VerbOutput};
use crate::envelope::{CliError, ErrorKind};

impl From<BikeLogError> for CliError {
    fn from(e: BikeLogError) -> Self {
        let kind = match e.kind {
            BikeLogErrorKind::Io => ErrorKind::Io,
            BikeLogErrorKind::InvalidId => ErrorKind::NotFound,
            BikeLogErrorKind::Parse => ErrorKind::InvalidInput,
            BikeLogErrorKind::Encode => ErrorKind::Internal,
        };
        CliError::new(kind, e.message)
    }
}

/// One entry as the CLI prints it: the stored entry plus its current text,
/// whether it is deleted and whether two devices edited it unresolved.
fn entry_json(e: &Entry) -> Value {
    json!({
        "entry": e,
        "current": e.current(),
        "deleted": e.is_deleted(),
        "in_conflict": e.in_conflict(),
    })
}

/// `bike list` — the bikes in the library.
pub fn list(ctx: &Ctx, _m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let load = bikelog::load_bikes(root);
    let text = load
        .bikes
        .iter()
        .map(|b| format!("{}  {}", b.bike_id, b.name))
        .chain(std::iter::once(format!("({} bike(s))", load.bikes.len())))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(VerbOutput::new(text, json!({ "bikes": load.bikes })))
}

/// `bike show` — a bike's log and its setup sheet now (or `--at`).
pub fn show(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let bike_id = text(m, "bike_id")?;
    let file = bikelog::read_bike_file(root, &bike_id)?;
    let at_ms = opt_integer(m, "at").unwrap_or_else(now_ms);
    let sheet = bikelog::running_sheet(&file.entries, at_ms);

    let mut lines = vec![format!("{} ({})", file.bike.name, file.bike.bike_id)];
    for e in file.entries.iter().filter(|e| !e.is_deleted()) {
        if let Some(v) = e.current() {
            let kind = match e.kind {
                bikelog::EntryKind::Note => "note",
                bikelog::EntryKind::Sheet => "sheet",
            };
            lines.push(format!("{}  {kind}  {}", v.at_ms, v.text));
        }
    }
    for c in file.calibrations.iter().filter(|c| !c.deleted) {
        lines.push(format!("{}  calibration {}  {}", c.captured_at_ms, c.calibration_id, calibration_line(c)));
    }
    lines.push("setup sheet:".to_string());
    lines.extend(sheet.values.iter().map(|(k, v)| format!("  {k} = {v}")));

    Ok(VerbOutput::new(
        lines.join("\n"),
        json!({
            "bike": file.bike,
            "entries": file.entries.iter().map(entry_json).collect::<Vec<_>>(),
            "calibrations": file.calibrations,
            "at_ms": at_ms,
            "running_sheet": sheet,
        }),
    ))
}

/// `bike note` — add a note to a bike's log. `--change key=value` values
/// update the setup sheet; `--device` says who wrote it (the analysis agent
/// passes `agent`).
pub fn note(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let bike_id = text(m, "bike_id")?;
    let note_text = opt_text(m, "text").unwrap_or_default();
    let mut changes = BTreeMap::new();
    for spec in many(m, "change") {
        let (key, value) = bikelog::parse_change(&spec).ok_or_else(|| CliError::usage(format!("--change {spec:?} is not key=value")))?;
        changes.insert(key, value);
    }
    if note_text.trim().is_empty() && changes.is_empty() {
        return Err(CliError::usage("a note needs --text, --change, or both"));
    }
    // The bike must exist, dry run or not.
    bikelog::read_bike_file(root, &bike_id)?;

    let now = now_ms();
    let device = opt_text(m, "device").unwrap_or_else(|| "cli".to_string());
    let entry = bikelog::new_note(&bike_id, &note_text, changes, opt_integer(m, "at").unwrap_or(now), &device, now);

    if ctx.dry_run {
        return Ok(VerbOutput::new(
            format!("would add a note to {bike_id}: {}", entry.versions[0].text),
            json!({ "entry": entry, "written": false }),
        ));
    }
    let saved = bikelog::save_entry(root, &entry)?;
    Ok(VerbOutput::new(
        format!("added note {} to {bike_id}", saved.entry_id),
        json!({ "entry": saved, "written": true }),
    ))
}

/// One calibration in a line: the tilt of IMU0, what else was fitted.
fn calibration_line(c: &BikeCalibration) -> String {
    let tilt = c.fit.up[2].clamp(-1.0, 1.0).acos().to_degrees();
    let pods = c.fit.imus.iter().map(|i| format!("IMU{}", i.imu_index)).collect::<Vec<_>>().join(", ");
    format!(
        "IMU0 tilted {tilt:.1} deg; pods: {}; steering axis: {}; forward: {}",
        if pods.is_empty() { "none" } else { &pods },
        if c.fit.steer_axis.is_some() { "yes" } else { "no" },
        match c.fit.forward_from {
            Some(ForwardFrom::SteerAxis) => "from the steering axis",
            Some(ForwardFrom::ForkPod) => "toward the fork pod",
            Some(ForwardFrom::Riding) => "matched to a ride",
            None => "from each ride",
        },
    )
}

/// The one catalogued session whose id is, or starts with, `id`.
fn session_by_prefix(root: &std::path::Path, id: &str) -> Result<SessionDetail, CliError> {
    let matches: Vec<String> =
        list_sessions(root)?.into_iter().map(|s| s.session_id).filter(|s| s.starts_with(id)).collect();
    match matches.as_slice() {
        [one] => Ok(get_session(root, one)?),
        [] => Err(CliError::new(ErrorKind::NotFound, format!("no session whose id starts with {id:?}"))),
        many => Err(CliError::usage(format!("{id:?} matches {} sessions; give more of the id", many.len()))),
    }
}

/// The recording a session was imported from, parsed.
fn parsed_recording(root: &std::path::Path, detail: &SessionDetail) -> Result<idl_rs::session::Session, CliError> {
    let bytes = idl_library::blob::read_blob(root, &detail.blob_sha256)
        .map_err(|e| CliError::io(format!("reading the recording of {}: {e}", detail.session_id)))?;
    Ok(idl_rs::parse::parse(&bytes).map_err(|e| CliError::new(ErrorKind::InvalidInput, e.to_string()))?.session)
}

/// `[from, to)` in seconds from a pair of `--…-from-ms` / `--…-to-ms` flags
/// (an absent end is open); None when neither was given.
fn window_s(m: &ArgMatches, stem: &str) -> Option<(f64, Option<f64>)> {
    let (from, to) = (opt_integer(m, &format!("{stem}-from-ms")), opt_integer(m, &format!("{stem}-to-ms")));
    (from.is_some() || to.is_some()).then(|| (from.unwrap_or(0) as f64 / 1000.0, to.map(|t| t as f64 / 1000.0)))
}

/// `bike calibrate` — fits a bike's IMU calibration from its recordings over
/// the windows given and saves it in the bike's file, or removes one with
/// `--delete`. Sessions on the bike use the newest calibration captured at or
/// before them (`idl_rs::calibration::bike::calibration_for`).
pub fn calibrate(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let bike_id = text(m, "bike_id")?;
    let file = bikelog::read_bike_file(root, &bike_id)?;
    let now = now_ms();

    if let Some(id) = opt_text(m, "delete") {
        let Some(found) = file.calibrations.iter().find(|c| c.calibration_id == id && !c.deleted) else {
            return Err(CliError::new(ErrorKind::NotFound, format!("{bike_id} has no calibration {id:?}")));
        };
        let gone = BikeCalibration { deleted: true, updated_at_ms: now.max(found.updated_at_ms + 1), ..found.clone() };
        if ctx.dry_run {
            return Ok(VerbOutput::new(
                format!("would remove calibration {id} from {bike_id}"),
                json!({ "calibration": gone, "written": false }),
            ));
        }
        let saved = bikelog::save_calibration(root, &bike_id, &gone)?;
        return Ok(VerbOutput::new(
            format!("removed calibration {id} from {bike_id}"),
            json!({ "calibration": saved, "written": true }),
        ));
    }

    if let Some(wheelbase_mm) = opt_integer(m, "wheelbase-mm") {
        if !(300..=3000).contains(&wheelbase_mm) {
            return Err(CliError::usage(format!("--wheelbase-mm {wheelbase_mm} is not a bike's wheelbase in millimetres")));
        }
        let Some(current) = file.calibrations.iter().filter(|c| !c.deleted).max_by_key(|c| (c.captured_at_ms, c.updated_at_ms)) else {
            return Err(CliError::new(ErrorKind::NotFound, format!("{bike_id} has no calibration to hold a wheelbase; fit one with --still first")));
        };
        let mut changed = BikeCalibration { updated_at_ms: now.max(current.updated_at_ms + 1), ..current.clone() };
        changed.fit.wheelbase_m = Some(wheelbase_mm as f64 / 1000.0);
        let line = format!("wheelbase of calibration {} set to {wheelbase_mm} mm", changed.calibration_id);
        if ctx.dry_run {
            return Ok(VerbOutput::new(format!("would save: {line}"), json!({ "calibration": changed, "written": false })));
        }
        let saved = bikelog::save_calibration(root, &bike_id, &changed)?;
        return Ok(VerbOutput::new(line, json!({ "calibration": saved, "written": true })));
    }

    if let Some(ride_id) = opt_text(m, "forward-from") {
        let Some(current) = file.calibrations.iter().filter(|c| !c.deleted).max_by_key(|c| (c.captured_at_ms, c.updated_at_ms)) else {
            return Err(CliError::new(ErrorKind::NotFound, format!("{bike_id} has no calibration to turn; fit one with --still first")));
        };
        let ride = session_by_prefix(root, &ride_id)?;
        let dir = root.join("sessions").join(&ride.session_id);
        let handle = idl_rs::store::parquet::open_session_lazy(&dir)
            .map_err(|e| CliError::io(format!("reading {}: {}", dir.join("data.parquet").display(), e.message)))?;
        let too_little = || {
            CliError::new(ErrorKind::InvalidInput, format!("{} has too little straight-line riding with GPS to say which way the bike travels", ride.session_id))
        };
        let trims = ride_trims(&handle, &current.fit).ok_or_else(too_little)?;
        let trim = trims.forward_deg.ok_or_else(too_little)?;
        let mut fit = current.fit.with_forward_turned(trim).map_err(|e| CliError::new(ErrorKind::InvalidInput, e.to_string()))?;
        if let Some(zero) = trims.steer_zero_deg {
            fit = fit.with_steer_zero(zero);
        }
        let turned = BikeCalibration { fit, forward_ride: Some(ride.session_id.clone()), updated_at_ms: now.max(current.updated_at_ms + 1), ..current.clone() };
        let line = format!(
            "calibration {} matched to ride {}: forward turned {:+.1} deg ({}){}",
            turned.calibration_id,
            ride.session_id,
            trim,
            if trim >= 0.0 { "to the left" } else { "to the right" },
            trims.steer_zero_deg.map_or(String::new(), |z| format!("; steering zero moved {z:+.1} deg")),
        );
        if ctx.dry_run {
            return Ok(VerbOutput::new(format!("would save: {line}"), json!({ "calibration": turned, "trim_deg": trim, "steer_zero_deg": trims.steer_zero_deg, "written": false })));
        }
        let saved = bikelog::save_calibration(root, &bike_id, &turned)?;
        return Ok(VerbOutput::new(line, json!({ "calibration": saved, "trim_deg": trim, "steer_zero_deg": trims.steer_zero_deg, "written": true })));
    }

    let still_id = opt_text(m, "still")
        .ok_or_else(|| CliError::usage("a calibration needs --still: a recording with the bike still and level"))?;
    let still_detail = session_by_prefix(root, &still_id)?;
    let still_w = window_s(m, "still").unwrap_or((1.0, None));
    let (rigid, fork, steer) = (window_s(m, "rigid"), window_s(m, "fork"), window_s(m, "steer"));
    let protocol_detail = match opt_text(m, "protocol") {
        Some(id) => Some(session_by_prefix(root, &id)?),
        None if rigid.is_some() || fork.is_some() || steer.is_some() => Some(still_detail.clone()),
        None => None,
    };
    let open = |(from, to): (f64, Option<f64>)| (from, to.unwrap_or(f64::INFINITY));
    let windows = match (&protocol_detail, rigid) {
        (Some(_), Some(r)) => Some(ProtocolWindows { rigid_s: open(r), fork_rigid_s: fork.map(open), steer_s: steer.map(open) }),
        (Some(_), None) => {
            return Err(CliError::usage(
                "the protocol recording needs --rigid-from-ms/--rigid-to-ms: the window where the bike was turned about as one body",
            ))
        }
        (None, _) => None,
    };

    let still_session = parsed_recording(root, &still_detail)?;
    let protocol_session = protocol_detail.as_ref().map(|d| parsed_recording(root, d)).transpose()?;
    let fit = fit_bike_calibration(&still_session, open(still_w), protocol_session.as_ref().zip(windows.as_ref()))
        .map_err(|e| CliError::new(ErrorKind::InvalidInput, e.to_string()))?;

    let source = |detail: &SessionDetail, (from_s, to_s): (f64, Option<f64>)| SourceWindow {
        session_id: detail.session_id.clone(),
        from_s,
        to_s,
    };
    let has = |index: u8| fit.imus.iter().any(|i| i.imu_index == index);
    let protocol_window =
        |on: bool, w: Option<(f64, Option<f64>)>| protocol_detail.as_ref().zip(w).filter(|_| on).map(|(d, w)| source(d, w));
    let calibration = BikeCalibration {
        calibration_id: bikelog::new_id(now),
        captured_at_ms: protocol_detail
            .as_ref()
            .map_or(still_detail.timestamp_utc_ms, |d| d.timestamp_utc_ms.max(still_detail.timestamp_utc_ms)),
        updated_at_ms: now,
        deleted: false,
        note: opt_text(m, "note").unwrap_or_default(),
        still: source(&still_detail, still_w),
        rigid: protocol_window(has(2), rigid),
        fork_rigid: protocol_window(has(1), fork.or(rigid)),
        steer: protocol_window(fit.steer_axis.is_some(), steer),
        forward_ride: None,
        fit,
    };

    let mut lines = vec![calibration_line(&calibration)];
    for i in &calibration.fit.imus {
        lines.push(format!(
            "  IMU{}: {:.3} m from IMU0, lever [{:.3}, {:.3}, {:.3}] m (IMU0 axes); its level reading is {:.1} deg from IMU0's",
            i.imu_index,
            (i.lever_m[0].powi(2) + i.lever_m[1].powi(2) + i.lever_m[2].powi(2)).sqrt(),
            i.lever_m[0],
            i.lever_m[1],
            i.lever_m[2],
            i.up_disagreement_deg,
        ));
    }
    if steer.is_some() && calibration.fit.steer_axis.is_none() {
        lines.push("  no steering axis: the bars barely moved in the steer window, or the frame moved with them".to_string());
    }
    if ctx.dry_run {
        lines.insert(0, format!("would save a calibration for {bike_id}:"));
        return Ok(VerbOutput::new(lines.join("\n"), json!({ "calibration": calibration, "written": false })));
    }
    let saved = bikelog::save_calibration(root, &bike_id, &calibration)?;
    lines.insert(0, format!("saved calibration {} for {bike_id}:", saved.calibration_id));
    Ok(VerbOutput::new(lines.join("\n"), json!({ "calibration": saved, "written": true })))
}
