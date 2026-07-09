//! The `table` command group: evaluate / list / validate a workbook's tables.
//! Thin formatter over the engine (`Workbook::tables`, `table::lap_windows`,
//! `table::validate`, `evaluate_table`) that adopts the JSON envelope.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Subcommand, ValueEnum};
use serde::Serialize;
use serde_json::json;

use idl_rs::laps::detect_laps;
use idl_rs::laps::model::Lap;
use idl_rs::session::handle::SessionHandle;
use idl_rs::table::{
    evaluate_table, lap_windows, validate, CellResult, Column, RowContext, TableProblem,
};
use idl_rs::track_artifact;
use idl_rs::workbook::{self, WorkbookTable};

use crate::envelope::{emit_structured, CliError, ErrorKind, Structured, Warning};
use crate::{load, OutFormat};

/// The `table` sub-actions. Each takes exactly the arguments it needs.
#[derive(Subcommand)]
pub enum TableAction {
    /// Evaluate a workbook's tables against a session.
    Eval {
        /// The `.idl0` session to evaluate against.
        session: PathBuf,
        /// Path to the `.idl0wb` workbook.
        #[arg(long)]
        workbook: PathBuf,
        /// `.idl0t` track for lap detection (required iff a table has lap-bound rows).
        #[arg(long)]
        track: Option<PathBuf>,
        /// Evaluate only the table with this block id (default: all).
        #[arg(long)]
        table: Option<String>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = TableEvalFormat::Text)]
        format: TableEvalFormat,
    },
    /// List the tables a workbook contains (no session, no evaluation).
    List {
        /// Path to the `.idl0wb` workbook.
        #[arg(long)]
        workbook: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
    /// Validate a workbook's tables and report problems.
    Check {
        /// Optional `.idl0` session; with it, channel/eval errors are also reported.
        session: Option<PathBuf>,
        /// Path to the `.idl0wb` workbook.
        #[arg(long)]
        workbook: PathBuf,
        /// `.idl0t` track for lap detection (optional).
        #[arg(long)]
        track: Option<PathBuf>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutFormat::Text)]
        format: OutFormat,
    },
}

/// Output format for `table eval`: human grid (default), CSV grid, or JSON.
#[derive(Clone, Copy, ValueEnum)]
pub enum TableEvalFormat {
    Text,
    Csv,
    Json,
}

/// Dispatch a `table` sub-action through its envelope wrapper.
pub fn run(action: TableAction) -> ExitCode {
    match action {
        TableAction::Eval {
            session,
            workbook,
            track,
            table,
            format,
        } => emit_structured(
            "table eval",
            cmd_eval(
                &session,
                &workbook,
                track.as_deref(),
                table.as_deref(),
                format,
            ),
        ),
        TableAction::List { workbook, format } => {
            emit_structured("table list", cmd_list(&workbook, format))
        }
        TableAction::Check {
            session,
            workbook,
            track,
            format,
        } => emit_structured(
            "table check",
            cmd_check(session.as_deref(), &workbook, track.as_deref(), format),
        ),
    }
}

/// Read the workbook and surface its tables, mapping the engine error.
fn load_tables(workbook: &Path) -> Result<Vec<WorkbookTable>, CliError> {
    let wb = workbook::read_workbook(workbook).map_err(CliError::from)?;
    Ok(wb.tables())
}

/// One `table list` entry: identity, label, layout metadata, column names, rows.
#[derive(Serialize)]
struct ListEntry {
    block_id: String,
    worksheet: String,
    placement: String,
    overlay_target_id: Option<String>,
    columns: Vec<String>,
    row_count: usize,
}

/// Project the surfaced tables into list entries (column names fall back to id).
fn list_entries(tables: &[WorkbookTable]) -> Vec<ListEntry> {
    tables
        .iter()
        .map(|t| ListEntry {
            block_id: t.block_id.clone(),
            worksheet: t.worksheet.clone(),
            placement: t.placement.clone(),
            overlay_target_id: t.overlay_target_id.clone(),
            columns: t
                .table
                .columns
                .iter()
                .map(|c| c.name.clone().unwrap_or_else(|| c.id.clone()))
                .collect(),
            row_count: t.table.rows.len(),
        })
        .collect()
}

fn cmd_list(workbook: &Path, format: OutFormat) -> Result<Structured, CliError> {
    let tables = load_tables(workbook)?;
    let entries = list_entries(&tables);
    match format {
        OutFormat::Json => Ok(Structured::Json {
            data: json!({ "tables": entries }),
            warnings: vec![],
        }),
        OutFormat::Text => {
            print_list_text(&entries);
            Ok(Structured::Text)
        }
    }
}

fn print_list_text(entries: &[ListEntry]) {
    if entries.is_empty() {
        println!("no tables in workbook");
        return;
    }
    for e in entries {
        println!(
            "{}  [{}]  {} cols, {} rows",
            e.worksheet,
            e.block_id,
            e.columns.len(),
            e.row_count
        );
    }
}

/// A resolved row window, recording-time seconds.
#[derive(Serialize)]
struct Window {
    t0: f64,
    t1: f64,
}

/// One evaluated row: its lap binding, resolved window, and per-cell results.
#[derive(Serialize)]
struct EvalRowPayload {
    context: Option<RowContext>,
    window: Option<Window>,
    cells: Vec<CellResult>,
}

/// One evaluated table: identity, label, layout metadata, columns, rows.
#[derive(Serialize)]
struct EvalTablePayload {
    block_id: String,
    worksheet: String,
    placement: String,
    overlay_target_id: Option<String>,
    columns: Vec<Column>,
    rows: Vec<EvalRowPayload>,
}

/// Select the tables to evaluate: all, or the one matching `id` (else
/// `not_found` with the available block ids for self-correction).
fn select_tables(
    tables: Vec<WorkbookTable>,
    id: Option<&str>,
) -> Result<Vec<WorkbookTable>, CliError> {
    match id {
        None => Ok(tables),
        Some(want) => {
            if tables.iter().any(|t| t.block_id == want) {
                Ok(tables.into_iter().filter(|t| t.block_id == want).collect())
            } else {
                let available: Vec<String> = tables.iter().map(|t| t.block_id.clone()).collect();
                Err(CliError::with_details(
                    ErrorKind::NotFound,
                    format!("no table with block id '{want}'"),
                    json!({ "entity": "table", "name": want, "available": available }),
                ))
            }
        }
    }
}

/// Read a track and detect the session's laps, mapping engine errors.
fn detect_laps_for(handle: &SessionHandle, track: &Path) -> Result<Vec<Lap>, CliError> {
    let t = track_artifact::read_track(track).map_err(CliError::from)?;
    let timing = t.timing.as_ref().ok_or_else(|| {
        CliError::with_details(
            ErrorKind::InvalidInput,
            format!("track '{}' has no lap timing configured", t.name),
            json!({ "track": t.name }),
        )
    })?;
    Ok(detect_laps(
        handle,
        timing,
        &t.sector_gates,
        &t.neutral_zones,
        None,
    ))
}

fn cmd_eval(
    session: &Path,
    workbook: &Path,
    track: Option<&Path>,
    table_id: Option<&str>,
    format: TableEvalFormat,
) -> Result<Structured, CliError> {
    let selected = select_tables(load_tables(workbook)?, table_id)?;
    let handle = load(session)?;

    // Lap detection only when a selected table binds laps; then --track is required.
    let needs_laps = selected
        .iter()
        .any(|wt| wt.table.rows.iter().any(|r| r.context.is_some()));
    let laps: Vec<Lap> = match (needs_laps, track) {
        (true, None) => {
            let wt = selected
                .iter()
                .find(|wt| wt.table.rows.iter().any(|r| r.context.is_some()))
                .expect("needs_laps implies a lap-bound table");
            return Err(CliError::with_details(
                ErrorKind::Usage,
                format!("table '{}' has lap-bound rows; pass --track", wt.block_id),
                json!({ "table": wt.block_id }),
            ));
        }
        (true, Some(tp)) => detect_laps_for(&handle, tp)?,
        (false, _) => Vec::new(),
    };

    let session_id = handle.metadata().session_id;
    let mut warnings: Vec<Warning> = Vec::new();
    let mut payloads: Vec<EvalTablePayload> = Vec::new();

    for wt in &selected {
        let windows = lap_windows(&wt.table, &laps);
        for (r, row) in wt.table.rows.iter().enumerate() {
            if let Some(ctx) = &row.context {
                if !ctx.session_id.is_empty() && ctx.session_id != session_id {
                    warnings.push(Warning {
                        kind: "session_mismatch".into(),
                        message: format!(
                            "table '{}' row {r} binds session {} but evaluating {session_id}",
                            wt.block_id, ctx.session_id
                        ),
                    });
                }
                if (ctx.lap_index as usize) >= laps.len() {
                    warnings.push(Warning {
                        kind: "lap_out_of_range".into(),
                        message: format!(
                            "table '{}' row {r} lap {} is past the {} detected laps",
                            wt.block_id,
                            ctx.lap_index,
                            laps.len()
                        ),
                    });
                }
            }
        }
        let results = evaluate_table(&handle, &wt.table, &windows);
        payloads.push(build_eval_payload(wt, &windows, results));
    }

    match format {
        TableEvalFormat::Json => Ok(Structured::Json {
            data: json!({ "tables": payloads }),
            warnings,
        }),
        TableEvalFormat::Text => {
            for w in &warnings {
                eprintln!("warning: {}", w.message);
            }
            print!("{}", eval_text_string(&payloads));
            Ok(Structured::Text)
        }
        TableEvalFormat::Csv => {
            for w in &warnings {
                eprintln!("warning: {}", w.message);
            }
            print!("{}", eval_csv_string(&payloads));
            Ok(Structured::Text)
        }
    }
}

fn build_eval_payload(
    wt: &WorkbookTable,
    windows: &[Option<(f64, f64)>],
    results: Vec<Vec<CellResult>>,
) -> EvalTablePayload {
    let rows = wt
        .table
        .rows
        .iter()
        .enumerate()
        .map(|(r, row)| EvalRowPayload {
            context: row.context.clone(),
            window: windows[r].map(|(t0, t1)| Window { t0, t1 }),
            cells: results[r].clone(),
        })
        .collect();
    EvalTablePayload {
        block_id: wt.block_id.clone(),
        worksheet: wt.worksheet.clone(),
        placement: wt.placement.clone(),
        overlay_target_id: wt.overlay_target_id.clone(),
        columns: wt.table.columns.clone(),
        rows,
    }
}

/// Format a cell for a text/csv grid: its value, `#ERR`, or empty when blank.
fn cell_field(c: &CellResult) -> String {
    match (&c.value, &c.error) {
        (_, Some(_)) => "#ERR".to_string(),
        (Some(v), None) => format!("{v}"),
        (None, None) => String::new(),
    }
}

fn eval_csv_string(tables: &[EvalTablePayload]) -> String {
    let mut out = String::new();
    for (i, t) in tables.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("# {} ({})\n", t.worksheet, t.block_id));
        let header: Vec<String> = t
            .columns
            .iter()
            .map(|c| c.name.clone().unwrap_or_else(|| c.id.clone()))
            .collect();
        out.push_str(&format!("{}\n", header.join(",")));
        for row in &t.rows {
            let fields: Vec<String> = row.cells.iter().map(cell_field).collect();
            out.push_str(&format!("{}\n", fields.join(",")));
        }
    }
    out
}

fn eval_text_string(tables: &[EvalTablePayload]) -> String {
    if tables.is_empty() {
        return "no tables\n".to_string();
    }
    let mut out = String::new();
    for t in tables {
        out.push_str(&format!("{} [{}]\n", t.worksheet, t.block_id));
        let header: Vec<String> = t
            .columns
            .iter()
            .map(|c| c.name.clone().unwrap_or_else(|| c.id.clone()))
            .collect();
        out.push_str(&format!("{}\n", header.join("  ")));
        for row in &t.rows {
            let fields: Vec<String> = row.cells.iter().map(cell_field).collect();
            out.push_str(&format!("{}\n", fields.join("  ")));
        }
        out.push('\n');
    }
    out
}

/// One checked table: identity, label, and the problems found.
#[derive(Serialize)]
struct CheckTablePayload {
    block_id: String,
    worksheet: String,
    problems: Vec<TableProblem>,
}

fn cmd_check(
    session: Option<&Path>,
    workbook: &Path,
    track: Option<&Path>,
    format: OutFormat,
) -> Result<Structured, CliError> {
    let tables = load_tables(workbook)?;

    // Optional session-aware pass: load the session and (if a track is given and
    // any table binds laps) its laps. --track is never required for check.
    let session_eval: Option<(SessionHandle, Vec<Lap>)> = match session {
        Some(sp) => {
            let handle = load(sp)?;
            let needs_laps = tables
                .iter()
                .any(|wt| wt.table.rows.iter().any(|r| r.context.is_some()));
            let laps = match (needs_laps, track) {
                (true, Some(tp)) => detect_laps_for(&handle, tp)?,
                _ => Vec::new(),
            };
            Some((handle, laps))
        }
        None => None,
    };

    let mut out: Vec<CheckTablePayload> = Vec::new();
    for wt in &tables {
        let mut problems = validate(&wt.table);
        if let Some((handle, laps)) = &session_eval {
            let windows = if laps.is_empty() {
                vec![None; wt.table.rows.len()]
            } else {
                lap_windows(&wt.table, laps)
            };
            let results = evaluate_table(handle, &wt.table, &windows);
            for (r, row) in results.iter().enumerate() {
                for (c, cell) in row.iter().enumerate() {
                    if let Some(msg) = &cell.error {
                        problems.push(TableProblem {
                            row: Some(r),
                            col: Some(c),
                            kind: "eval_error".into(),
                            message: msg.clone(),
                        });
                    }
                }
            }
        }
        out.push(CheckTablePayload {
            block_id: wt.block_id.clone(),
            worksheet: wt.worksheet.clone(),
            problems,
        });
    }

    match format {
        OutFormat::Json => Ok(Structured::Json {
            data: json!({ "tables": out }),
            warnings: vec![],
        }),
        OutFormat::Text => {
            print!("{}", check_text_string(&out));
            Ok(Structured::Text)
        }
    }
}

fn check_text_string(tables: &[CheckTablePayload]) -> String {
    let total: usize = tables.iter().map(|t| t.problems.len()).sum();
    if total == 0 {
        return format!("all tables OK ({} checked)\n", tables.len());
    }
    let mut out = String::new();
    for t in tables {
        if t.problems.is_empty() {
            continue;
        }
        out.push_str(&format!("{} [{}]\n", t.worksheet, t.block_id));
        for p in &t.problems {
            let loc = match (p.row, p.col) {
                (Some(r), Some(c)) => format!("r{r}c{c}"),
                (Some(r), None) => format!("r{r}"),
                _ => "table".to_string(),
            };
            out.push_str(&format!("  {loc} {}: {}\n", p.kind, p.message));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::table::{Column, Row, TableModel};

    fn table_with(name: &str, id: &str, cols: Vec<Column>, rows: usize) -> WorkbookTable {
        WorkbookTable {
            block_id: id.to_string(),
            worksheet: name.to_string(),
            placement: "inFlow".to_string(),
            overlay_target_id: None,
            table: TableModel {
                columns: cols,
                rows: (0..rows)
                    .map(|i| Row {
                        id: format!("r{i}"),
                        context: None,
                    })
                    .collect(),
                cells: (0..rows).map(|_| vec![]).collect(),
            },
        }
    }

    #[test]
    fn list_entries_uses_name_then_id_fallback() {
        // Arrange — a named column and an unnamed one.
        let wt = table_with(
            "WS",
            "blk",
            vec![
                Column {
                    id: "c0".into(),
                    name: Some("lap".into()),
                    template: None,
                },
                Column {
                    id: "c1".into(),
                    name: None,
                    template: None,
                },
            ],
            2,
        );

        // Act
        let e = list_entries(&[wt]);

        // Assert
        assert_eq!(e[0].columns, vec!["lap", "c1"]);
        assert_eq!(e[0].row_count, 2);
        assert_eq!(e[0].placement, "inFlow");
    }

    #[test]
    fn select_tables_unknown_id_is_not_found_with_available() {
        // Arrange
        let tables = vec![
            table_with("WS", "a", vec![], 0),
            table_with("WS", "b", vec![], 0),
        ];

        // Act
        let err = select_tables(tables, Some("zzz")).unwrap_err();

        // Assert — not_found, with the available ids for self-correction.
        assert_eq!(err.kind, ErrorKind::NotFound);
        let d = err.details.unwrap();
        assert_eq!(d["available"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn select_tables_filters_to_one() {
        // Arrange
        let tables = vec![
            table_with("WS", "a", vec![], 0),
            table_with("WS", "b", vec![], 0),
        ];

        // Act
        let got = select_tables(tables, Some("b")).unwrap();

        // Assert
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].block_id, "b");
    }

    #[test]
    fn eval_csv_renders_values_and_err() {
        // Arrange — one table, two columns, one row with a value and an error.
        let payloads = vec![EvalTablePayload {
            block_id: "blk".into(),
            worksheet: "WS".into(),
            placement: "inFlow".into(),
            overlay_target_id: None,
            columns: vec![
                Column {
                    id: "c0".into(),
                    name: Some("a".into()),
                    template: None,
                },
                Column {
                    id: "c1".into(),
                    name: Some("b".into()),
                    template: None,
                },
            ],
            rows: vec![EvalRowPayload {
                context: None,
                window: None,
                cells: vec![
                    CellResult {
                        value: Some(1.5),
                        error: None,
                    },
                    CellResult {
                        value: None,
                        error: Some("boom".into()),
                    },
                ],
            }],
        }];

        // Act
        let csv = eval_csv_string(&payloads);

        // Assert
        assert!(csv.contains("a,b"));
        assert!(csv.contains("1.5,#ERR"));
    }

    #[test]
    fn check_text_summarizes_clean_and_problem_tables() {
        // Arrange — one clean table, one with a static problem.
        let clean = CheckTablePayload {
            block_id: "ok".into(),
            worksheet: "WS".into(),
            problems: vec![],
        };
        let bad = CheckTablePayload {
            block_id: "bad".into(),
            worksheet: "WS".into(),
            problems: vec![TableProblem {
                row: Some(0),
                col: Some(1),
                kind: "parse_error".into(),
                message: "boom".into(),
            }],
        };

        // Act
        let clean_out = check_text_string(&[clean]);
        let bad_out = check_text_string(&[bad]);

        // Assert
        assert!(clean_out.contains("all tables OK"));
        assert!(bad_out.contains("parse_error"));
        assert!(bad_out.contains("r0c1"));
    }
}
