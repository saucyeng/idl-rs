//! The `bike` verbs (ruling R251): `bike list`, `bike show`, `bike note`.
//!
//! `bike note` is how the analysis agent keeps a bike's setup sheet current
//! from the rider's notes: each entry goes through `store::bikelog`'s merge,
//! so a note written here and one written on the phone meanwhile both
//! survive.

use std::collections::BTreeMap;

use clap::ArgMatches;
use serde_json::{json, Value};

use idl_rs::store::bikelog::{self, BikeLogError, BikeLogErrorKind, Entry};

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

/// UTC milliseconds now.
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
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
    lines.push("setup sheet:".to_string());
    lines.extend(sheet.values.iter().map(|(k, v)| format!("  {k} = {v}")));

    Ok(VerbOutput::new(
        lines.join("\n"),
        json!({
            "bike": file.bike,
            "entries": file.entries.iter().map(entry_json).collect::<Vec<_>>(),
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
