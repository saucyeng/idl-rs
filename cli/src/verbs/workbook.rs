//! The `workbook` verbs (ruling R229): `new`, `check`, `cells`, `eval`,
//! `data`, `export`.
//!
//! # What `eval` covers
//!
//! [`eval_cells`] is the engine's Tauri-free cell evaluator, so `math` and
//! `table` cells evaluate here exactly as they do in the app. `js` cells do
//! not: a `js` cell runs in the app's sandboxed webview, and `core` has no
//! JavaScript engine — by CLAUDE.md §2 it never will. A `js` cell therefore
//! reports `evaluated: false` with the reason, rather than silently
//! producing nothing. `workbook export --format json` says the same.
//!
//! # Sessions (engine-gaps N6, session-sets spec §6.1)
//!
//! `--session` takes an `.idl0` file or a session id inside `--data-dir`; an
//! id opens that session's `data.parquet` lazily, with laps from its
//! `session.json` (`--track` re-detects them instead). A workbook with a
//! `sessions:` set, evaluated with `--data-dir` and no `--session`, evaluates
//! its session tables over the set through `store::set_job` — the runner the
//! app's `eval_session_set` uses. A repeated `--session` is an ad-hoc
//! one-group set.

use std::path::{Path, PathBuf};

use clap::ArgMatches;
use serde_json::{json, Value};

use idl_rs::commands::lap_ops::{self, MainLap};
use idl_rs::commands::workbook_ops::{
    new_workbook_id, new_workbook_source, NewWorkbook, WorkbookTemplate,
};
use idl_rs::laps::detect_laps;
use idl_rs::laps::model::Lap;
use idl_rs::math::eval::MathLapContext;
use idl_rs::session::handle::SessionHandle;
use idl_rs::track_artifact;
use idl_rs::math::units::UnitLabel;
use idl_rs::store::set_job::{engine_identity, run_session_set, SetJobOptions, SetJobReport};
use idl_rs::table::{CellResult, RowSource, TableModel};
use idl_rs::workbook::v3::session_set::{parse_session_set, SessionSet, SetGroup, SetMember};
use idl_rs::workbook::v3::set_eval::{MemberResults, MemberValue, PlanScope};
use idl_rs::workbook::v3::table_value::{definition_values, evaluate_table_in_window};
use idl_rs::workbook::v3::{
    eval_cells, parse_workbook, AxisKind, CellDoc, CellError, CellEvalResult, CellKindToken, WorkbookDoc,
};

use crate::envelope::{CliError, ErrorKind};
use crate::verbs::{opt_integer, opt_path, opt_text, path, text, Ctx, VerbOutput};

/// Why a `js` cell has no value in a headless run.
const JS_NOT_EVALUATED: &str =
    "js cells run in the app's sandbox; the CLI has no JavaScript engine";

/// `workbook new` — write a skeleton, refusing to overwrite.
pub fn new(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let template_name = opt_text(m, "template").unwrap_or_else(|| "blank".to_string());
    let template = WorkbookTemplate::from_str(&template_name)
        .ok_or_else(|| CliError::usage(format!("unknown template `{template_name}`")))?;

    // Never clobber: a workbook is a user's document, and `new` is not the
    // verb that replaces one.
    if file.exists() {
        return Err(CliError::usage(format!("{} already exists", file.display())));
    }

    let name = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string());
    let spec = NewWorkbook {
        id: new_workbook_id(),
        name: name.clone(),
        session_id: opt_text(m, "session"),
    };
    let source = new_workbook_source(template, &spec);

    let data = json!({
        "file": file.display().to_string(),
        "workbook_id": spec.id,
        "name": name,
        "template": template.as_str(),
        "written": !ctx.dry_run,
    });

    if ctx.dry_run {
        return Ok(VerbOutput::new(
            format!("would create {} from the {} template", file.display(), template.as_str()),
            data,
        ));
    }

    write_file(&file, source.as_bytes())?;
    Ok(VerbOutput::new(
        format!("created {} from the {} template", file.display(), template.as_str()),
        data,
    ))
}

/// `workbook check` — parse, and evaluate too when a session is given.
pub fn check(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let (doc, structural) = read_workbook(&file)?;

    // Without a session there is nothing to resolve channels against, so
    // `check` reports structure only — which is the useful thing to be able
    // to do on a workbook whose data is somewhere else.
    let cells = match session_context(ctx, m)? {
        Some((handle, lap_ctx)) => eval_cells(&doc, &structural, &handle, &lap_ctx),
        None => Vec::new(),
    };

    let mut problems: Vec<Value> = structural.iter().map(structural_json).collect();
    for cell in &cells {
        for def in &cell.defs {
            if let Some(error) = &def.error {
                problems.push(json!({
                    "cell_id": cell.cell_id,
                    "definition": def.name,
                    "kind": "eval",
                    "message": error.message,
                }));
            }
        }
        for error in &cell.errors {
            if let CellError::Eval(e) = error {
                problems.push(json!({
                    "cell_id": cell.cell_id,
                    "kind": "eval",
                    "message": e.message,
                }));
            }
        }
    }

    let text = if problems.is_empty() {
        format!("{}: no problems ({} cell(s))", file.display(), doc.cells.len())
    } else {
        problems
            .iter()
            .map(|p| format!("{}: {}", p["cell_id"].as_str().unwrap_or("?"), p["message"]))
            .chain(std::iter::once(format!("({} problem(s))", problems.len())))
            .collect::<Vec<_>>()
            .join("\n")
    };

    Ok(VerbOutput::new(
        text,
        json!({
            "file": file.display().to_string(),
            "cell_count": doc.cells.len(),
            "problems": problems,
        }),
    ))
}

/// `workbook cells` — the cell list, in document order.
pub fn cells(_ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let (doc, _) = read_workbook(&file)?;

    let text = doc
        .cells
        .iter()
        .map(|cell| format!("{}  {}", cell.id, kind_name(cell.kind_token)))
        .chain(std::iter::once(format!("({} cell(s))", doc.cells.len())))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({
            "file": file.display().to_string(),
            "cells": doc.cells.iter().map(cell_summary_json).collect::<Vec<_>>(),
        }),
    ))
}

/// `workbook eval` — every cell's result against a session, or the
/// workbook's session tables over its set (see the module docs).
pub fn eval(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let (doc, structural) = read_workbook(&file)?;
    let values = m.get_flag("values");

    if let Some(set) = set_to_run(ctx, m, &doc)? {
        let report = run_set(ctx, m, &doc, &set, values)?;
        return Ok(set_output(&file, &doc, &set, &report, values));
    }

    let (handle, lap_ctx) = session_context(ctx, m)?.ok_or_else(|| {
        CliError::usage("workbook eval needs --session, or --data-dir with a workbook that declares `sessions`")
    })?;

    let results = eval_cells(&doc, &structural, &handle, &lap_ctx);

    let text = if values {
        value_lines(&results).join("\n")
    } else {
        results
            .iter()
            .map(|cell| {
                let values = cell.defs.iter().filter(|d| d.value.is_some()).count();
                let errors = cell.defs.iter().filter(|d| d.error.is_some()).count() + cell.errors.len();
                format!("{}  {}  {values} value(s), {errors} error(s)", cell.cell_id, kind_name(cell.kind))
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut data = json!({
        "file": file.display().to_string(),
        "session_id": handle.metadata().session_id,
        "cells": results.iter().map(eval_cell_json).collect::<Vec<_>>(),
    });
    if values {
        data["values"] = Value::Array(value_json(&results));
    }
    Ok(VerbOutput::new(text, data))
}

/// `workbook data` — one cell's evaluated series.
pub fn data(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let cell_id = text(m, "cell")?;
    let (doc, structural) = read_workbook(&file)?;

    // A session-row table is a set table: evaluated over the set, printed as
    // its rows (gap N7).
    let set_table = doc.cells.iter().find(|c| c.id == cell_id).and_then(|c| c.table.as_ref())
        .is_some_and(|t| t.row_source == RowSource::Sessions);
    if set_table {
        let set = set_to_run(ctx, m, &doc)?.ok_or_else(|| {
            CliError::usage("this is a session table: pass --data-dir (and --session ids, or a workbook with `sessions`)")
        })?;
        let report = run_set(ctx, m, &doc, &set, false)?;
        let table = report.tables.iter().find(|t| t.cell_id == cell_id).ok_or_else(|| {
            CliError::new(ErrorKind::Internal, format!("session table `{cell_id}` was not evaluated"))
        })?;
        let text = render_set_table(&doc, &set, &report, table);
        return Ok(VerbOutput::new(
            text,
            json!({ "file": file.display().to_string(), "cell_id": cell_id, "table": set_table_json(&report, table) }),
        ));
    }

    let (handle, lap_ctx) = session_context(ctx, m)?.ok_or_else(|| {
        CliError::usage("workbook data needs --session: there is nothing to evaluate against")
    })?;

    let results = eval_cells(&doc, &structural, &handle, &lap_ctx);
    let cell = results.iter().find(|c| c.cell_id == cell_id).ok_or_else(|| {
        CliError::with_details(
            ErrorKind::NotFound,
            format!("no cell `{cell_id}` in {}", file.display()),
            json!({ "available": results.iter().map(|c| &c.cell_id).collect::<Vec<_>>() }),
        )
    })?;

    if cell.kind == CellKindToken::Js {
        return Err(CliError::new(ErrorKind::Unsupported, JS_NOT_EVALUATED));
    }

    // A table prints its rows, `[name]` reading this session's definitions
    // first (C2 §4, ruling Q6).
    if let Some(model) = doc.cells.iter().find(|c| c.id == cell_id).and_then(|c| c.table.as_ref()) {
        let (planned, rows) = evaluate_table_in_window(model, &definition_values(&results), &handle, &lap_ctx);
        let text = render_plain_table(&planned, &rows);
        return Ok(VerbOutput::new(
            text,
            json!({ "file": file.display().to_string(), "cell_id": cell_id, "table": { "model": planned, "results": rows } }),
        ));
    }

    let series: Vec<Value> = cell
        .defs
        .iter()
        .map(|def| match &def.value {
            Some(view) => {
                // This verb prints the whole series, so it is the one place
                // that materialises the seconds array.
                let channel = view.to_host_channel();
                json!({
                    "name": def.name,
                    "label": def.label,
                    "length": channel.length,
                    "sample_rate_hz": def.sample_rate_hz,
                    "t": channel.t,
                    "v": channel.v,
                })
            }
            None => json!({
                "name": def.name,
                "label": def.label,
                "error": def.error.as_ref().map(|e| e.message.clone()),
            }),
        })
        .collect();

    let text = series
        .iter()
        .map(|s| match s.get("length") {
            Some(length) => format!("{}  {length} sample(s)", s["name"].as_str().unwrap_or("?")),
            None => format!("{}  error: {}", s["name"].as_str().unwrap_or("?"), s["error"]),
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({ "file": file.display().to_string(), "cell_id": cell_id, "series": series }),
    ))
}

/// `workbook export` — re-render the source, or write the evaluated cells.
pub fn export(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let format = opt_text(m, "format").unwrap_or_else(|| "md".to_string());
    let out = opt_path(m, "out");

    let body = match format.as_str() {
        "md" => {
            let (doc, _) = read_workbook(&file)?;
            idl_rs::workbook::v3::render_workbook(&doc)
        }
        "json" => {
            let (doc, structural) = read_workbook(&file)?;
            let (handle, lap_ctx) = session_context(ctx, m)?.ok_or_else(|| {
                CliError::usage("workbook export --format json needs --session")
            })?;
            let results = eval_cells(&doc, &structural, &handle, &lap_ctx);
            serde_json::to_string_pretty(&json!({
                "file": file.display().to_string(),
                "session_id": handle.metadata().session_id,
                "cells": results.iter().map(eval_cell_json).collect::<Vec<_>>(),
            }))
            .map_err(|e| CliError::new(ErrorKind::Internal, e.to_string()))?
        }
        other => return Err(CliError::usage(format!("unknown export format `{other}`"))),
    };

    let data = json!({
        "file": file.display().to_string(),
        "format": format,
        "out": out.as_ref().map(|p| p.display().to_string()),
        "bytes": body.len(),
        "written": out.is_some() && !ctx.dry_run,
    });

    match (&out, ctx.dry_run) {
        (_, true) => Ok(VerbOutput::new(
            format!("would write {} byte(s) of {format}", body.len()),
            data,
        )),
        (Some(out), false) => {
            write_file(out, body.as_bytes())?;
            Ok(VerbOutput::new(format!("wrote {}", out.display()), data))
        }
        // No `--out`: the rendered body itself is the text output, and JSON
        // mode reports what would have been written rather than embedding a
        // whole document inside an envelope.
        (None, false) => Ok(VerbOutput::new(body, data)),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// Reads and parses a workbook, mapping a fatal parse failure.
fn read_workbook(file: &Path) -> Result<(WorkbookDoc, Vec<idl_rs::workbook::v3::WorkbookError>), CliError> {
    let source = std::fs::read_to_string(file)
        .map_err(|e| CliError::io(format!("reading {}: {e}", file.display())))?;
    parse_workbook(&source).map_err(|errors| {
        CliError::with_details(
            ErrorKind::InvalidInput,
            errors
                .first()
                .map(|e| e.message.clone())
                .unwrap_or_else(|| format!("{} is not a workbook", file.display())),
            json!({ "problems": errors.iter().map(structural_json).collect::<Vec<_>>() }),
        )
    })
}

/// Loads `--session` (and `--track`, when given) into the pair `eval_cells`
/// needs. `Ok(None)` when no session was given.
///
/// `--session` is an existing `.idl0` file, else a session id inside the data
/// directory (gap N6): that session's `data.parquet`, opened lazily, with its
/// laps from `session.json` over the whole session (the app's whole-session
/// window, `store::session_context`). `--track` re-detects laps instead.
fn session_context(ctx: &Ctx, m: &ArgMatches) -> Result<Option<(SessionHandle, MathLapContext)>, CliError> {
    // A flag that cannot mean anything is a usage error before it is an I/O
    // one: without `--track` there are no laps, so nothing can be numbered.
    // Checked ahead of the session load so the message names the real fault
    // rather than whichever file happened to be missing too.
    if opt_integer(m, "main-lap").is_some() && opt_path(m, "track").is_none() {
        return Err(CliError::usage("--main-lap needs --track: there are no laps to number"));
    }

    let Some(session) = opt_path(m, "session") else {
        return Ok(None);
    };
    let library_dir = ctx.data_dir.as_deref().map(|d| d.join("sessions").join(&session));
    let (handle, stored_laps) = match library_dir.filter(|d| !session.is_file() && d.join("data.parquet").is_file()) {
        Some(dir) => {
            let handle = idl_rs::store::parquet::open_session_lazy(&dir).map_err(|e| {
                CliError::io(format!("reading {}: {}", dir.join("data.parquet").display(), e.message))
            })?;
            let json = std::fs::read(dir.join("session.json"))
                .ok()
                .and_then(|b| idl_rs::store::session_json::parse_session_json(&b).ok());
            let bounds = idl_rs::store::session_context::whole_session_bounds(&dir)
                .map_err(|e| CliError::io(format!("sizing {}: {}", dir.display(), e.message)))?;
            (handle, Some(idl_rs::store::session_context::window_lap_context(json.as_ref(), bounds)))
        }
        None if !session.exists() && ctx.data_dir.is_some() => {
            return Err(CliError::with_details(
                ErrorKind::NotFound,
                format!("no session `{}`: not a file, and not a session in the data directory", session.display()),
                json!({ "session": session.display().to_string() }),
            ));
        }
        None => (crate::load(&session)?, None),
    };

    // Without a track there are no laps for an `.idl0` file, and
    // `MathLapContext::empty()` makes every lap-aware function report
    // `NoLapContext` rather than a wrong number — which is the honest
    // headless answer. A library session has its catalogued laps.
    let Some(track) = opt_path(m, "track") else {
        return Ok(Some((handle, stored_laps.unwrap_or_else(MathLapContext::empty))));
    };
    let lap_ctx = lap_context(&handle, &track, opt_integer(m, "main-lap"))?;
    Ok(Some((handle, lap_ctx)))
}

/// The session set `workbook eval`/`data` should evaluate, if any: a
/// repeated `--session` is an ad-hoc one-group set; no `--session` with a
/// data directory is the workbook's own `sessions` set. `None` means "one
/// session, the ordinary way".
fn set_to_run(ctx: &Ctx, m: &ArgMatches, doc: &WorkbookDoc) -> Result<Option<SessionSet>, CliError> {
    let ids: Vec<String> =
        m.get_many::<PathBuf>("session").map(|v| v.map(|p| p.display().to_string()).collect()).unwrap_or_default();
    if ids.len() > 1 {
        ctx.data_dir()?;
        return Ok(Some(SessionSet {
            groups: vec![SetGroup {
                name: None,
                members: ids.into_iter().map(|id| SetMember { id, label: None }).collect(),
            }],
        }));
    }
    if !ids.is_empty() || ctx.data_dir.is_none() {
        return Ok(None);
    }
    parse_session_set(&doc.front_matter_unknown).map_err(|e| {
        CliError::with_details(ErrorKind::InvalidInput, e.message.clone(), json!({ "kind": format!("{:?}", e.kind) }))
    })
}

/// Runs `set` through `store::set_job`, progress on stderr.
fn run_set(
    ctx: &Ctx,
    m: &ArgMatches,
    doc: &WorkbookDoc,
    set: &SessionSet,
    values: bool,
) -> Result<SetJobReport, CliError> {
    let data_dir = ctx.data_dir()?;
    // The same fixed ceiling the CLI's indexing pool uses (core has no
    // physical-memory query); members are admitted one at a time against it.
    let budget = idl_rs::store::index_job::ByteBudget::new(crate::library::CLI_INDEX_BUDGET_BYTES);
    let options = SetJobOptions {
        use_cache: !m.try_get_one::<bool>("no-cache").ok().flatten().copied().unwrap_or(false),
        engine_identity: engine_identity(),
        scope: if values { PlanScope::AllDefinitions } else { PlanScope::Tables },
        cancel: None,
        budget: Some(&budget),
    };
    Ok(run_session_set(data_dir, doc, set, options, &mut |p| {
        eprintln!("[{}/{}] {} {}", p.done + 1, p.total, &p.session_id, p.phase.as_str());
    }))
}

/// Detects laps against `track` and folds them into a [`MathLapContext`].
///
/// `main_lap` is the 1-based lap a lap-scoped expression means. There is no
/// default: a `.idl0` file carries no designation, and an absent main lap
/// means lap-scoped aggregates read the whole session (`main_lap_window`
/// treats `None` as "no window", R128). That is a wrong answer to a
/// lap-scoped question, so it is said out loud on stderr rather than left
/// for the reader to notice in the numbers.
fn lap_context(
    handle: &SessionHandle,
    track: &Path,
    main_lap: Option<i64>,
) -> Result<MathLapContext, CliError> {
    let artifact = track_artifact::read_track(track).map_err(CliError::from)?;
    let timing = artifact.timing.as_ref().ok_or_else(|| {
        CliError::with_details(
            ErrorKind::InvalidInput,
            format!("track '{}' has no lap timing configured", artifact.name),
            json!({ "track": artifact.name }),
        )
    })?;
    let laps: Vec<Lap> =
        detect_laps(handle, timing, &artifact.sector_gates, &artifact.neutral_zones, None);

    let designated = match main_lap {
        Some(n) => MainLap::Number(u32::try_from(n).map_err(|_| {
            CliError::usage(format!("--main-lap must be a positive lap number, got {n}"))
        })?),
        None => {
            if !laps.is_empty() {
                eprintln!(
                    "note: no --main-lap given, so lap-scoped expressions read the whole session, not one lap ({} lap(s) detected)",
                    laps.len()
                );
            }
            MainLap::None
        }
    };

    lap_ops::lap_context(&laps, designated).map_err(|available| {
        CliError::with_details(
            ErrorKind::NotFound,
            "--main-lap names a lap this session does not have",
            json!({ "available_lap_numbers": available }),
        )
    })
}

/// Writes `bytes`, creating the parent directory.
fn write_file(path: &PathBuf, bytes: &[u8]) -> Result<(), CliError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::io(format!("creating {}: {e}", parent.display())))?;
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| CliError::io(format!("writing {}: {e}", path.display())))
}

/// `math` / `table` / `js` / `video`.
fn kind_name(kind: CellKindToken) -> &'static str {
    match kind {
        CellKindToken::Math => "math",
        CellKindToken::Table => "table",
        CellKindToken::Js => "js",
        CellKindToken::Video => "video",
    }
}

/// One structural error as JSON.
fn structural_json(error: &idl_rs::workbook::v3::WorkbookError) -> Value {
    json!({
        "cell_id": error.cell_id,
        "kind": format!("{:?}", error.kind),
        "message": error.message,
    })
}

/// One cell, without its values — what `workbook cells` lists.
fn cell_summary_json(cell: &CellDoc) -> Value {
    json!({
        "cell_id": cell.id,
        "kind": kind_name(cell.kind_token),
        "table_rows": cell.table.as_ref().map(|t| t.rows.len()),
    })
}

/// One cell's evaluation result, values summarised rather than inlined —
/// `workbook data` is the verb that prints samples.
fn eval_cell_json(cell: &CellEvalResult) -> Value {
    json!({
        "cell_id": cell.cell_id,
        "kind": kind_name(cell.kind),
        "evaluated": cell.kind != CellKindToken::Js,
        "not_evaluated_reason": (cell.kind == CellKindToken::Js).then_some(JS_NOT_EVALUATED),
        "definitions": cell.defs.iter().map(|def| json!({
            "name": def.name,
            "label": def.label,
            "length": def.value.as_ref().map(|c| c.length),
            "sample_rate_hz": def.sample_rate_hz,
            "error": def.error.as_ref().map(|e| e.message.clone()),
        })).collect::<Vec<_>>(),
        "errors": cell.errors.iter().map(|e| match e {
            CellError::Structural(s) => structural_json(s),
            CellError::Eval(m) => json!({ "cell_id": cell.cell_id, "kind": "eval", "message": m.message }),
        }).collect::<Vec<_>>(),
    })
}

/// A value as text: up to four decimals, trailing zeros dropped.
fn fmt_value(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.to_string() }
}

/// A unit label as text; empty when there is none to show.
fn unit_text(unit: &UnitLabel) -> String {
    match unit {
        UnitLabel::Known { text } => text.clone(),
        _ => String::new(),
    }
}

/// `--values` lines for one evaluation: every rank-0 and `[lap]` result.
fn value_lines(cells: &[CellEvalResult]) -> Vec<String> {
    let mut out = Vec::new();
    for def in cells.iter().flat_map(|c| c.defs.iter()) {
        let unit = unit_text(&def.unit);
        match (&def.value, &def.error) {
            (Some(v), _) if v.length == 1 && v.axis == AxisKind::None => {
                out.push(format!("{}  {}  {unit}", def.name, fmt_value(v.v[0])).trim_end().to_string())
            }
            (Some(v), _) if v.axis == AxisKind::Lap => {
                let laps: Vec<String> =
                    v.t_us.iter().zip(v.v.iter()).map(|(n, x)| format!("{n}:{}", fmt_value(*x))).collect();
                out.push(format!("{}  [lap] {}  {unit}", def.name, laps.join(" ")).trim_end().to_string())
            }
            (None, Some(e)) => out.push(format!("{}  error: {}", def.name, e.message)),
            _ => {}
        }
    }
    out
}

/// `--values` as JSON for one evaluation.
fn value_json(cells: &[CellEvalResult]) -> Vec<Value> {
    cells
        .iter()
        .flat_map(|c| c.defs.iter())
        .filter_map(|def| match (&def.value, &def.error) {
            (Some(v), _) if v.length == 1 && v.axis == AxisKind::None => {
                Some(json!({ "name": def.name, "shape": "[]", "value": v.v[0], "unit": unit_text(&def.unit) }))
            }
            (Some(v), _) if v.axis == AxisKind::Lap => Some(json!({
                "name": def.name, "shape": "[lap]", "laps": v.t_us.to_vec(), "values": v.v.to_vec(), "unit": unit_text(&def.unit),
            })),
            (None, Some(e)) => Some(json!({ "name": def.name, "error": e.message })),
            _ => None,
        })
        .collect()
}

/// `--values` lines for one set member's kept results (authored definitions
/// only, in document order).
fn member_value_lines(doc: &WorkbookDoc, results: &MemberResults) -> Vec<String> {
    let mut out = Vec::new();
    for def in &doc.defs {
        let Some(r) = results.0.get(&def.name) else { continue };
        let unit = unit_text(&r.unit);
        match &r.value {
            MemberValue::Scalar(v) => out.push(format!("{}  {}  {unit}", def.name, fmt_value(*v)).trim_end().to_string()),
            MemberValue::Laps { laps, values } => {
                let laps: Vec<String> = laps.iter().zip(values).map(|(n, x)| format!("{n}:{}", fmt_value(*x))).collect();
                out.push(format!("{}  [lap] {}  {unit}", def.name, laps.join(" ")).trim_end().to_string())
            }
            MemberValue::Error(e) => out.push(format!("{}  error: {}", def.name, e.message)),
            MemberValue::FullRate => {}
        }
    }
    out
}

/// One table cell result as text.
fn cell_text(r: &CellResult) -> String {
    match (r.value, &r.error) {
        (Some(v), _) => fmt_value(v),
        (None, Some(e)) => format!("error: {e}"),
        (None, None) => String::new(),
    }
}

/// Lays `rows` out as aligned, two-space-separated columns.
fn align(rows: &[Vec<String>]) -> Vec<String> {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> =
        (0..cols).map(|c| rows.iter().filter_map(|r| r.get(c)).map(|s| s.chars().count()).max().unwrap_or(0)).collect();
    rows.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(c, s)| format!("{s:<w$}", w = widths[c]))
                .collect::<Vec<_>>()
                .join("  ")
                .trim_end()
                .to_string()
        })
        .collect()
}

/// A per-window table as text: a header of column names, one line per row.
fn render_plain_table(model: &TableModel, rows: &[Vec<CellResult>]) -> String {
    let mut grid = vec![std::iter::once("row".to_string())
        .chain(model.columns.iter().map(|c| c.name.clone().unwrap_or_else(|| c.id.clone())))
        .collect::<Vec<_>>()];
    for (r, row) in model.rows.iter().zip(rows) {
        grid.push(std::iter::once(r.id.clone()).chain(row.iter().map(cell_text)).collect());
    }
    align(&grid).join("\n")
}

/// A set table as text: a header with each column's unit, then each group's
/// rows under a `== group ==` line.
fn render_set_table(
    doc: &WorkbookDoc,
    set: &SessionSet,
    report: &SetJobReport,
    table: &idl_rs::workbook::v3::set_eval::SetTable,
) -> String {
    let unit_of = |c: usize| -> String {
        let Some(t) = report.plan.templates.iter().find(|t| t.cell_id == table.cell_id && t.column == c) else {
            return String::new();
        };
        report
            .results
            .iter()
            .flatten()
            .filter_map(|r| r.0.get(&t.def_name).map(|x| unit_text(&x.unit)))
            .find(|u| !u.is_empty())
            .unwrap_or_default()
    };
    let header: Vec<String> = std::iter::once("session".to_string())
        .chain(table.model.columns.iter().enumerate().map(|(c, col)| {
            let name = col.name.clone().unwrap_or_else(|| col.id.clone());
            match unit_of(c) {
                u if u.is_empty() => name,
                u => format!("{name} ({u})"),
            }
        }))
        .collect();
    let mut lines = vec![format!("table {}", table.cell_id)];
    let mut grid = vec![header];
    let mut group_rows: Vec<(usize, Option<usize>)> = Vec::new();
    let mut last_group = None;
    for (r, &i) in table.row_members.iter().enumerate() {
        let member = &report.members[i];
        if last_group != Some(member.group) {
            last_group = Some(member.group);
            group_rows.push((grid.len(), Some(member.group)));
        }
        grid.push(std::iter::once(member.label.clone()).chain(table.results[r].iter().map(cell_text)).collect());
    }
    let aligned = align(&grid);
    for (k, line) in aligned.into_iter().enumerate() {
        if let Some((_, Some(g))) = group_rows.iter().find(|(at, _)| *at == k) {
            if set.groups.len() > 1 || set.groups[*g].name.is_some() {
                lines.push(format!("== {} ==", set.groups[*g].name.as_deref().unwrap_or("(unnamed)")));
            }
        }
        lines.push(line);
    }
    let _ = doc;
    lines.join("\n")
}

/// A set table as JSON (the `SessionSetResult.tables[]` shape).
fn set_table_json(report: &SetJobReport, table: &idl_rs::workbook::v3::set_eval::SetTable) -> Value {
    let _ = report;
    json!({ "cell_id": table.cell_id, "model": table.model, "results": table.results, "row_members": table.row_members })
}

/// `workbook eval`'s output over a set: every session table, then the
/// member summary (and, with `--values`, each member's values).
fn set_output(file: &Path, doc: &WorkbookDoc, set: &SessionSet, report: &SetJobReport, values: bool) -> VerbOutput {
    let mut parts: Vec<String> = report.tables.iter().map(|t| render_set_table(doc, set, report, t)).collect();
    let mut summary = vec!["members".to_string()];
    for m in &report.members {
        let error = m.error.as_ref().map(|e| format!("  {}", e.message)).unwrap_or_default();
        summary.push(format!("  {}  {}  {}{error}", m.label, m.session_id, m.status.as_str()));
    }
    parts.push(summary.join("\n"));
    for w in &report.warnings {
        parts.push(format!("warning: {}", w.message));
    }
    if values {
        for (m, r) in report.members.iter().zip(&report.results) {
            if let Some(r) = r {
                parts.push(format!("values {} ({})\n{}", m.label, m.session_id, member_value_lines(doc, r).join("\n")));
            }
        }
    }
    let data = json!({
        "file": file.display().to_string(),
        "set_key": report.set_key,
        "groups": set.groups.iter().map(|g| json!({ "name": g.name })).collect::<Vec<_>>(),
        "members": report.members.iter().map(|m| json!({
            "session_id": m.session_id,
            "label": m.label,
            "group": m.group,
            "start_utc_ms": m.start_utc_ms,
            "status": m.status.as_str(),
            "error": m.error.as_ref().map(|e| e.message.clone()),
            "decoded_channels": m.decoded_channels,
            "timing_ms": {
                "open": m.timing.open_ms,
                "decode": m.timing.decode_ms,
                "evaluate": m.timing.evaluate_ms,
                "cache_write": m.timing.cache_write_ms,
                "slowest": m.timing.slowest,
            },
        })).collect::<Vec<_>>(),
        "tables": report.tables.iter().map(|t| set_table_json(report, t)).collect::<Vec<_>>(),
        "warnings": report.warnings.iter().map(|w| json!({ "cell_id": w.cell_id, "missing": w.missing, "message": w.message })).collect::<Vec<_>>(),
        "cache_error": report.cache_error,
    });
    VerbOutput::new(parts.join("\n\n"), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cell_kind_has_a_name() {
        // Arrange
        let kinds = [CellKindToken::Math, CellKindToken::Table, CellKindToken::Js];

        // Act
        let names: Vec<&str> = kinds.iter().copied().map(kind_name).collect();

        // Assert
        assert_eq!(names, ["math", "table", "js"]);
    }

    #[test]
    fn a_js_cell_reports_that_it_was_not_evaluated_and_why() {
        // Arrange
        let cell = CellEvalResult {
            cell_id: "aaaaaaaa".to_string(),
            kind: CellKindToken::Js,
            defs: Vec::new(),
            errors: Vec::new(),
        };

        // Act
        let value = eval_cell_json(&cell);

        // Assert — silence would read as "this cell has no content".
        assert_eq!(value["evaluated"], false);
        assert_eq!(value["not_evaluated_reason"], JS_NOT_EVALUATED);
    }

    #[test]
    fn a_math_cell_reports_that_it_was_evaluated_and_gives_no_reason() {
        // Arrange
        let cell = CellEvalResult {
            cell_id: "aaaaaaaa".to_string(),
            kind: CellKindToken::Math,
            defs: Vec::new(),
            errors: Vec::new(),
        };

        // Act
        let value = eval_cell_json(&cell);

        // Assert
        assert_eq!(value["evaluated"], true);
        assert!(value["not_evaluated_reason"].is_null());
    }

    #[test]
    fn reading_a_file_that_is_not_a_workbook_is_an_invalid_input_error() {
        // Arrange
        let dir = std::env::temp_dir().join(format!("idl-rs-wb-verbs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("not-a-workbook.idl0wb");
        std::fs::write(&file, b"no front matter here\n").unwrap();

        // Act
        let err = read_workbook(&file).unwrap_err();

        // Assert
        assert_eq!(err.kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn reading_a_missing_file_is_an_io_error() {
        // Arrange
        let missing = std::env::temp_dir().join("idl-rs-wb-verbs-missing.idl0wb");
        let _ = std::fs::remove_file(&missing);

        // Act
        let err = read_workbook(&missing).unwrap_err();

        // Assert
        assert_eq!(err.kind, ErrorKind::Io);
    }

    /// Parses `argv` against the generated tree and returns the leaf matches
    /// for `workbook eval`.
    fn eval_matches(argv: &[&str]) -> ArgMatches {
        let mut full = vec!["idl-rs", "workbook", "eval"];
        full.extend_from_slice(argv);
        crate::verbs::augment(clap::Command::new("idl-rs"))
            .try_get_matches_from(full)
            .unwrap()
            .remove_subcommand()
            .unwrap()
            .1
            .remove_subcommand()
            .unwrap()
            .1
    }

    #[test]
    fn main_lap_without_track_is_a_usage_error_and_never_reads_the_session() {
        // Arrange — a session path that does not exist, so an I/O error
        // would win if the flags were checked after the load.
        let missing = std::env::temp_dir().join("idl-rs-wb-verbs-no-such-session.idl0");
        let _ = std::fs::remove_file(&missing);
        let m = eval_matches(&[
            "wb.idl0wb",
            "--session",
            &missing.display().to_string(),
            "--main-lap",
            "2",
        ]);

        // Act
        let err = session_context(&Ctx::from_matches(&m), &m).unwrap_err();

        // Assert — the fault is the flag pair, not the file.
        assert_eq!(err.kind, ErrorKind::Usage);
        assert!(err.message.contains("--main-lap needs --track"), "{}", err.message);
    }

    #[test]
    fn no_session_at_all_is_no_context_rather_than_an_error() {
        // Arrange
        let m = eval_matches(&["wb.idl0wb"]);

        // Act
        let ctx = session_context(&Ctx::from_matches(&m), &m).unwrap();

        // Assert — a workbook with no session still parses and reports.
        assert!(ctx.is_none());
    }
    /// A data root with sessions `s1` (Fork 1, 4) and `s2` (Fork 2, 9), each
    /// with one 10 s lap in `session.json`, and a set workbook over both.
    fn library() -> (PathBuf, PathBuf) {
        use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("idl-rs-wb-set-{}-{nanos}", std::process::id()));
        for (id, fork) in [("s1", vec![1.0, 4.0]), ("s2", vec![2.0, 9.0])] {
            let session = Session {
                session_id: id.to_string(),
                device_id: None,
                timestamp_utc_ms: 0,
                timestamp_source: TimestampSource::Header,
                config_checksum: None,
                source_format: SourceFormat::Idl0,
                blob_sha256: format!("blob-{id}"),
                channels: vec![Channel {
                    channel_id: "Fork".to_string(),
                    t_us: vec![0, 1_000_000],
                    t_recorded_us: None,
                    nominal_rate_hz: 1.0,
                    column: RawColumn::F64(fork),
                    source_kind: "test".to_string(),
                    unit: String::new(),
                    gaps: Vec::new(),
                }],
            };
            idl_rs::store::parquet::write_session_parquet(&root, &session, "0.1.0").unwrap();
            let mut json = idl_rs::store::session_json::empty_session_json(id);
            json.laps = vec![idl_rs::store::session_json::LapJson {
                lap_number: 1,
                start_timestamp_ms: 0,
                end_timestamp_ms: 0,
                raw_elapsed_ms: 10_000,
                lap_time_ms: 10_000,
                start_time_secs: 0.0,
                end_time_secs: 10.0,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            }];
            std::fs::write(root.join("sessions").join(id).join("session.json"), serde_json::to_vec(&json).unwrap()).unwrap();
        }
        let wb = root.join("set.idl1wb");
        std::fs::write(
            &wb,
            "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Set\nsessions:\n  - group: Day 1\n    members:\n      - { id: s1, label: D1 S1 }\n      - { id: s2, label: D1 S2 }\n---\n\n```math id=aaaaaaaa\npeak = max([Fork])\nlt = sum(lap_time())\n```\n\n```table id=bbbbbbbb\n{\"rowSource\":\"sessions\",\"columns\":[{\"id\":\"c0\",\"name\":\"Peak\",\"template\":\"[peak]\"}],\"rows\":[],\"cells\":[]}\n```\n",
        )
        .unwrap();
        (root, wb)
    }

    #[test]
    fn workbook_eval_set_workbook_with_data_dir_prints_each_session_table() {
        // Arrange
        let (root, wb) = library();
        let m = eval_matches(&[&wb.display().to_string(), "--data-dir", &root.display().to_string(), "--no-cache"]);

        // Act
        let out = eval(&Ctx::from_matches(&m), &m).unwrap();

        // Assert
        assert!(out.text.contains("== Day 1 =="), "{}", out.text);
        assert!(out.text.lines().any(|l| l.starts_with("D1 S1") && l.trim_end().ends_with('4')), "{}", out.text);
        assert!(out.text.lines().any(|l| l.starts_with("D1 S2") && l.trim_end().ends_with('9')), "{}", out.text);
        assert!(!root.join("cache").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn workbook_eval_session_id_with_data_dir_reads_data_parquet_lazily_and_laps_from_session_json() {
        // Arrange
        let (root, wb) = library();
        let m = eval_matches(&[
            &wb.display().to_string(),
            "--data-dir",
            &root.display().to_string(),
            "--session",
            "s2",
            "--values",
        ]);

        // Act
        let out = eval(&Ctx::from_matches(&m), &m).unwrap();

        // Assert
        assert!(out.text.lines().any(|l| l == "peak  9"), "{}", out.text);
        assert!(out.text.lines().any(|l| l.starts_with("lt  10")), "{}", out.text);
        let _ = std::fs::remove_dir_all(&root);
    }
}
