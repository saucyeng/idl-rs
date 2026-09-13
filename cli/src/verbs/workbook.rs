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
use idl_rs::workbook::v3::{
    eval_cells, parse_workbook, CellDoc, CellError, CellEvalResult, CellKindToken, WorkbookDoc,
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
pub fn check(_ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let (doc, structural) = read_workbook(&file)?;

    // Without a session there is nothing to resolve channels against, so
    // `check` reports structure only — which is the useful thing to be able
    // to do on a workbook whose data is somewhere else.
    let cells = match session_context(m)? {
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

/// `workbook eval` — every cell's result against a session.
pub fn eval(_ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let (doc, structural) = read_workbook(&file)?;
    let (handle, lap_ctx) = session_context(m)?.ok_or_else(|| {
        CliError::usage("workbook eval needs --session: there is nothing to evaluate against")
    })?;

    let results = eval_cells(&doc, &structural, &handle, &lap_ctx);

    let text = results
        .iter()
        .map(|cell| {
            let values = cell.defs.iter().filter(|d| d.value.is_some()).count();
            let errors = cell.defs.iter().filter(|d| d.error.is_some()).count() + cell.errors.len();
            format!("{}  {}  {values} value(s), {errors} error(s)", cell.cell_id, kind_name(cell.kind))
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({
            "file": file.display().to_string(),
            "session_id": handle.metadata().session_id,
            "cells": results.iter().map(eval_cell_json).collect::<Vec<_>>(),
        }),
    ))
}

/// `workbook data` — one cell's evaluated series.
pub fn data(_ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let file = path(m, "file")?;
    let cell_id = text(m, "cell")?;
    let (doc, structural) = read_workbook(&file)?;
    let (handle, lap_ctx) = session_context(m)?.ok_or_else(|| {
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

    let series: Vec<Value> = cell
        .defs
        .iter()
        .map(|def| match &def.value {
            Some(channel) => json!({
                "name": def.name,
                "label": def.label,
                "length": channel.length,
                "sample_rate_hz": def.sample_rate_hz,
                "t": channel.t,
                "v": channel.v,
            }),
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
            let (handle, lap_ctx) = session_context(m)?.ok_or_else(|| {
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
fn session_context(m: &ArgMatches) -> Result<Option<(SessionHandle, MathLapContext)>, CliError> {
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
    let handle = crate::load(&session)?;

    // Without a track there are no laps, and `MathLapContext::empty()` makes
    // every lap-aware function report `NoLapContext` rather than a wrong
    // number — which is the honest headless answer.
    let Some(track) = opt_path(m, "track") else {
        return Ok(Some((handle, MathLapContext::empty())));
    };
    let lap_ctx = lap_context(&handle, &track, opt_integer(m, "main-lap"))?;
    Ok(Some((handle, lap_ctx)))
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

/// `math` / `table` / `js`.
fn kind_name(kind: CellKindToken) -> &'static str {
    match kind {
        CellKindToken::Math => "math",
        CellKindToken::Table => "table",
        CellKindToken::Js => "js",
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
        let err = session_context(&m).unwrap_err();

        // Assert — the fault is the flag pair, not the file.
        assert_eq!(err.kind, ErrorKind::Usage);
        assert!(err.message.contains("--main-lap needs --track"), "{}", err.message);
    }

    #[test]
    fn no_session_at_all_is_no_context_rather_than_an_error() {
        // Arrange
        let m = eval_matches(&["wb.idl0wb"]);

        // Act
        let ctx = session_context(&m).unwrap();

        // Assert — a workbook with no session still parses and reports.
        assert!(ctx.is_none());
    }
}
