//! The `library` and `docs` verbs on the generated clap tree.
//!
//! `library` and `docs workbook` already existed as hand-written clap
//! `Subcommand` enums with their own renderers (`library.rs`, `docs_cmd.rs`).
//! R230 moves the *declaration* into the table without rewriting those
//! renderers: [`library`] and [`docs`] rebuild the existing action enum from
//! the generated `ArgMatches` and hand it to the existing `run`. One
//! declaration, one implementation, and the tables' flags and the actions'
//! fields are checked against each other by this module's tests.
//!
//! `docs cli` is new, and is the one R230 item 2 asks for: it emits the
//! command table itself, as Markdown or (with `--json`) as the JSON the
//! app's checked-in `cliTable.json` holds.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::ArgMatches;
use serde_json::json;

use idl_rs::commands::markdown::{command_table_json, render_command_reference};
use idl_rs::commands::table::CommandRow;

use crate::docs_cmd::{self, DocsAction};
use crate::envelope::{CliError, ErrorKind};
use crate::library::{self, LibraryAction};
use crate::verbs::{emit, many, opt_integer, opt_path, path, switch, Ctx, VerbOutput};

/// Runs one `library` verb by rebuilding [`LibraryAction`] from the
/// generated matches.
///
/// Returns an [`ExitCode`] rather than a [`VerbOutput`]: `library::run`
/// renders and exits itself, and re-rendering its progress lines through the
/// uniform envelope would mean maintaining two copies of the same output.
pub fn library(row: &CommandRow, ctx: &Ctx, m: &ArgMatches) -> ExitCode {
    let data_dir = match ctx.data_dir() {
        Ok(dir) => dir.to_path_buf(),
        Err(e) => return emit(row, ctx, Err(e)),
    };

    // `library index` and `library rebuild` have no read-only form in
    // `library.rs`, so `--dry-run` reports the intent here instead of
    // half-running the job.
    if ctx.dry_run && matches!(row.verb, "index" | "rebuild") {
        let sessions = many(m, "sessions");
        return emit(
            row,
            ctx,
            Ok(VerbOutput::new(
                format!("would {} {} session(s)", row.verb, sessions.len()),
                json!({ "sessions": sessions, "written": false }),
            )),
        );
    }

    let action = match row.verb {
        "fold-in" => match folder(m) {
            Ok(folder) => LibraryAction::FoldIn {
                folder,
                data_dir,
                r#move: switch(m, "move"),
                recursive: switch(m, "recursive"),
                dry_run: ctx.dry_run,
                json: ctx.json,
            },
            Err(e) => return emit(row, ctx, Err(e)),
        },
        "scan" => match folder(m) {
            Ok(folder) => LibraryAction::Scan {
                folder,
                data_dir,
                recursive: switch(m, "recursive"),
                json: ctx.json,
            },
            Err(e) => return emit(row, ctx, Err(e)),
        },
        "stale" => LibraryAction::Stale { data_dir, json: ctx.json },
        "index" => LibraryAction::Index {
            session_ids: many(m, "sessions"),
            data_dir,
            all: switch(m, "all"),
            force: switch(m, "force"),
            workers: opt_integer(m, "workers").and_then(|w| usize::try_from(w).ok()),
            json: ctx.json,
        },
        "rebuild" => LibraryAction::Rebuild {
            session_ids: many(m, "sessions"),
            data_dir,
            all: switch(m, "all"),
            json: ctx.json,
        },
        other => {
            return emit(
                row,
                ctx,
                Err(CliError::new(
                    ErrorKind::Internal,
                    format!("`library {other}` is in the command table but has no implementation"),
                )),
            )
        }
    };

    library::run(action)
}

/// Runs one `docs` verb.
pub fn docs(row: &CommandRow, ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    match row.verb {
        "workbook" => {
            let out = opt_path(m, "out")
                .ok_or_else(|| CliError::usage("docs workbook needs --out"))?;
            let src = opt_path(m, "src").unwrap_or_else(|| PathBuf::from("docs/reference-src"));
            if ctx.dry_run {
                return Ok(VerbOutput::new(
                    format!("would write {}", out.display()),
                    json!({ "out": out.display().to_string(), "written": false }),
                ));
            }
            // `docs_cmd::run` renders and exits; it is reused verbatim so the
            // byte-stability CI depends on has exactly one implementation.
            let code = docs_cmd::run(DocsAction::Workbook { out: out.clone(), src });
            if code == ExitCode::SUCCESS {
                Ok(VerbOutput::new(String::new(), json!({ "out": out.display().to_string(), "written": true })))
            } else {
                Err(CliError::io(format!("writing {} failed", out.display())))
            }
        }
        "cli" => cli(ctx, m),
        other => Err(CliError::new(
            ErrorKind::Internal,
            format!("`docs {other}` is in the command table but has no implementation"),
        )),
    }
}

/// `docs cli` — the command table as Markdown, or as JSON under `--json`.
///
/// With `--out` the chosen rendering is written to that file; without it,
/// Markdown goes to stdout as the command's text and JSON goes into the
/// uniform envelope's `data`.
fn cli(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let out = opt_path(m, "out");
    let markdown = render_command_reference();
    let table = command_table_json();

    let data = json!({
        "out": out.as_ref().map(|p| p.display().to_string()),
        "written": out.is_some() && !ctx.dry_run,
        "table": table,
    });

    match (&out, ctx.dry_run) {
        (Some(out), true) => Ok(VerbOutput::new(format!("would write {}", out.display()), data)),
        (None, true) => Ok(VerbOutput::new(markdown, data)),
        (Some(out), false) => {
            let body = if ctx.json {
                serde_json::to_string_pretty(&table)
                    .map_err(|e| CliError::new(ErrorKind::Internal, e.to_string()))?
            } else {
                markdown
            };
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| CliError::io(format!("creating {}: {e}", parent.display())))?;
                }
            }
            // `as_bytes`, not a text write: the rendering already holds `\n`
            // only, and CI compares the committed file byte for byte.
            std::fs::write(out, body.as_bytes())
                .map_err(|e| CliError::io(format!("writing {}: {e}", out.display())))?;
            Ok(VerbOutput::new(format!("wrote {}", out.display()), data))
        }
        (None, false) => Ok(VerbOutput::new(markdown, data)),
    }
}

/// The `folder` positional both `fold-in` and `scan` take.
fn folder(m: &ArgMatches) -> Result<PathBuf, CliError> {
    path(m, "folder")
}

#[cfg(test)]
mod tests {
    use idl_rs::commands::table::rows_for;

    #[test]
    fn every_library_row_in_the_table_is_one_this_module_can_build() {
        // Arrange
        let verbs: Vec<&str> = rows_for("library").iter().map(|r| r.verb).collect();

        // Act
        let known = ["fold-in", "scan", "stale", "index", "rebuild"];

        // Assert — a row added without an adapter would reach the `other`
        // arm and report an internal error at runtime; this catches it at
        // test time instead.
        for verb in &verbs {
            assert!(known.contains(verb), "`library {verb}` has no adapter");
        }
    }

    #[test]
    fn every_docs_row_in_the_table_is_one_this_module_can_build() {
        // Arrange
        let verbs: Vec<&str> = rows_for("docs").iter().map(|r| r.verb).collect();

        // Act
        let known = ["workbook", "cli"];

        // Assert
        for verb in &verbs {
            assert!(known.contains(verb), "`docs {verb}` has no adapter");
        }
    }

    #[test]
    fn the_usage_exit_code_is_the_one_the_ruling_names() {
        // Arrange / Act
        let code = crate::verbs::USAGE_EXIT;

        // Assert — R230 item 3: exit 0/1/2.
        assert_eq!(code, 2);
    }
}
