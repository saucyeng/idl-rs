//! The noun-rooted command surface (ruling R230): `idl-rs <noun> <verb>`.
//!
//! Nothing here declares a subcommand by hand. [`augment`] walks
//! [`idl_rs::commands::COMMANDS`] and builds one clap subcommand per row, so
//! the flags a command accepts, the help it prints and the generated
//! reference all come from the same table — a row is the only place a
//! command is declared.
//!
//! [`dispatch`] routes a parsed `(noun, verb)` back to the function that
//! does the work. That function is always a thin wrapper over one `idl_rs`
//! call (R230: no CLI-only logic); the wrapper's job is turning clap's
//! `ArgMatches` into the core call's arguments and the core result into the
//! two renderings [`VerbOutput`] carries.
//!
//! # Uniform behaviour (R230 item 3)
//!
//! - `--json` on every command. Success is one envelope on **stdout**;
//!   failure is a typed error envelope on **stderr**, so a consumer reading
//!   stdout never has to distinguish the two streams' shapes. (The
//!   pre-R230 commands in `envelope.rs` put structured errors on stdout;
//!   that is the older contract and is left alone.)
//! - `--data-dir`, falling back to `IDL1_DATA_DIR`, on every command that
//!   works against a data directory.
//! - `--dry-run` on every writer.
//! - Exit `0` on success, `1` on a failed operation, `2` on a usage error.

mod catalog;
mod session;
mod store;
mod workbook;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Arg as ClapArg, ArgAction, ArgMatches, Command as ClapCommand};
use serde_json::{json, Value};

use idl_rs::commands::table::{
    find, nouns_in_use, rows_for, Arg, CommandRow, Flag, ValueKind, DATA_DIR_ENV,
    JSON_SCHEMA_VERSION,
};

use crate::envelope::{CliError, ErrorKind};

/// The exit code a usage error returns (R230 item 3).
pub const USAGE_EXIT: u8 = 2;

/// One verb's result, in both the renderings the CLI can be asked for.
///
/// Both are built whichever mode is in force. They are small — a summary
/// line and a `Value` over data already in memory — and building both means
/// a text-mode bug cannot hide behind `--json` or the other way round.
pub struct VerbOutput {
    /// What text mode prints to stdout. May be empty for a command whose
    /// only product is a file it wrote.
    pub text: String,
    /// What `--json` puts in the envelope's `data`.
    pub data: Value,
}

impl VerbOutput {
    /// A result with both renderings.
    pub fn new(text: impl Into<String>, data: Value) -> Self {
        VerbOutput { text: text.into(), data }
    }
}

/// What every verb function receives besides its own arguments: the uniform
/// flags, already resolved.
pub struct Ctx {
    /// `--json` was given.
    pub json: bool,
    /// `--dry-run` was given. Always `false` on a non-writer, which has no
    /// such flag to give.
    pub dry_run: bool,
    /// `--data-dir`, else `$IDL1_DATA_DIR`, else unset.
    data_dir: Option<PathBuf>,
}

impl Ctx {
    /// Reads the uniform flags out of a parsed subcommand.
    fn from_matches(m: &ArgMatches) -> Ctx {
        let data_dir = m
            .try_get_one::<PathBuf>("data-dir")
            .ok()
            .flatten()
            .cloned()
            .or_else(|| std::env::var_os(DATA_DIR_ENV).map(PathBuf::from));
        Ctx {
            json: m.try_get_one::<bool>("json").ok().flatten().copied().unwrap_or(false),
            dry_run: m.try_get_one::<bool>("dry-run").ok().flatten().copied().unwrap_or(false),
            data_dir,
        }
    }

    /// The data directory, or a usage error naming both ways to supply it.
    ///
    /// # Errors
    /// [`ErrorKind::Usage`] when neither `--data-dir` nor `IDL1_DATA_DIR` is set.
    pub fn data_dir(&self) -> Result<&Path, CliError> {
        self.data_dir.as_deref().ok_or_else(|| {
            CliError::usage(format!("no data directory: pass --data-dir or set {DATA_DIR_ENV}"))
        })
    }
}

/// Adds one clap subcommand per noun in the table to `cmd`.
///
/// The nouns that predate R230 and are not in the table (`table`, and the
/// noun-less legacy verbs) are declared by `main.rs`'s derive as before;
/// this only ever adds.
pub fn augment(cmd: ClapCommand) -> ClapCommand {
    let mut cmd = cmd;
    for noun in nouns_in_use() {
        let mut group = ClapCommand::new(noun)
            .about(noun_about(noun))
            .subcommand_required(true)
            .arg_required_else_help(true);
        for row in rows_for(noun) {
            group = group.subcommand(clap_for(row));
        }
        cmd = cmd.subcommand(group);
    }
    cmd
}

/// The one-line description of a noun's group, printed in `idl-rs --help`.
fn noun_about(noun: &str) -> &'static str {
    match noun {
        "session" => "Recorded sessions: list, inspect, import, and edit their metadata",
        "workbook" => "Workbooks: create, validate, evaluate, and export",
        "track" => "Tracks and the lap detection that binds them to sessions",
        "catalog" => "The catalog index: verify it, rebuild it",
        "library" => "Bulk library management for a whole data directory",
        "docs" => "Regenerate documentation from the engine's own catalogs",
        _ => "",
    }
}

/// Builds one row's clap subcommand: its positionals, its own flags, then
/// R230 item 3's uniform ones.
fn clap_for(row: &CommandRow) -> ClapCommand {
    let mut cmd = ClapCommand::new(row.verb).about(row.help);
    for arg in row.args {
        cmd = cmd.arg(clap_arg(arg));
    }
    for flag in row.all_flags() {
        cmd = cmd.arg(clap_flag(&flag));
    }
    cmd
}

/// One positional argument.
fn clap_arg(arg: &Arg) -> ClapArg {
    let mut a = ClapArg::new(arg.name).help(arg.help).required(arg.required);
    if arg.repeatable {
        a = a.num_args(0..).action(ArgAction::Append);
    }
    typed(a, arg.kind, arg.choices)
}

/// One long flag: a switch, or a value of the row's declared kind.
fn clap_flag(flag: &Flag) -> ClapArg {
    let mut a = ClapArg::new(flag.long).long(flag.long).help(flag.help);
    match flag.kind {
        None => a = a.action(ArgAction::SetTrue),
        Some(kind) => {
            if flag.repeatable {
                a = a.action(ArgAction::Append);
            }
            if let Some(default) = flag.default {
                a = a.default_value(default);
            }
            a = typed(a, kind, flag.choices);
        }
    }
    a
}

/// Attaches the value parser for a [`ValueKind`], so clap rejects a bad
/// value with its own usage error rather than the command doing it later.
fn typed(a: ClapArg, kind: ValueKind, choices: &'static [&'static str]) -> ClapArg {
    match kind {
        ValueKind::Path => a.value_parser(clap::value_parser!(PathBuf)),
        ValueKind::Integer => a.value_parser(clap::value_parser!(i64)),
        ValueKind::Choice => a.value_parser(clap::builder::PossibleValuesParser::new(choices)),
        ValueKind::Text => a.value_parser(clap::value_parser!(String)),
    }
}

/// Runs one `(noun, verb)` pair, having been handed the verb's own matches.
///
/// Returns `None` when `noun` names no table group, which is how `main.rs`
/// tells a generated subcommand apart from one of its own.
pub fn dispatch(noun: &str, matches: &ArgMatches) -> Option<ExitCode> {
    if !nouns_in_use().contains(&noun) {
        return None;
    }
    let (verb, m) = matches.subcommand()?;
    let row = find(noun, verb)?;
    let ctx = Ctx::from_matches(m);

    let result = match (noun, verb) {
        ("session", "list") => session::list(&ctx, m),
        ("session", "show") => session::show(&ctx, m),
        ("session", "laps") => session::laps(&ctx, m),
        ("session", "set-start") => session::set_start(&ctx, m),
        ("session", "set-meta") => session::set_meta(&ctx, m),
        ("session", "import") => session::import(&ctx, m),
        ("session", "synth") => session::synth(&ctx, m),
        ("workbook", "new") => workbook::new(&ctx, m),
        ("workbook", "check") => workbook::check(&ctx, m),
        ("workbook", "cells") => workbook::cells(&ctx, m),
        ("workbook", "eval") => workbook::eval(&ctx, m),
        ("workbook", "data") => workbook::data(&ctx, m),
        ("workbook", "export") => workbook::export(&ctx, m),
        ("track", "list") => catalog::track_list(&ctx, m),
        ("track", "detect") => catalog::track_detect(&ctx, m),
        ("catalog", "verify") => catalog::verify(&ctx, m),
        ("catalog", "rebuild") => catalog::rebuild(&ctx, m),
        ("library", _) => return Some(store::library(row, &ctx, m)),
        ("docs", _) => store::docs(row, &ctx, m),
        _ => Err(CliError::new(
            ErrorKind::Internal,
            format!("`{noun} {verb}` is in the command table but has no implementation"),
        )),
    };

    Some(emit(row, &ctx, result))
}

/// Prints one verb's result and returns its exit code.
///
/// Text mode prints [`VerbOutput::text`] to stdout and a bare `error: …` to
/// stderr. JSON mode prints a success envelope to stdout and a typed error
/// envelope to stderr.
pub fn emit(row: &CommandRow, ctx: &Ctx, result: Result<VerbOutput, CliError>) -> ExitCode {
    match result {
        Ok(out) => {
            if ctx.json {
                println!("{}", pretty(&success_envelope(row, &out.data)));
            } else if !out.text.is_empty() {
                println!("{}", out.text);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            let usage = error.kind == ErrorKind::Usage;
            if ctx.json {
                eprintln!("{}", pretty(&error_envelope(row, error)));
            } else {
                eprintln!("error: {} {}: {}", row.noun, row.verb, error.message);
            }
            if usage {
                ExitCode::from(USAGE_EXIT)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

/// The success envelope: `schema_version`, the command, and its `data`.
fn success_envelope(row: &CommandRow, data: &Value) -> Value {
    json!({
        "schema_version": JSON_SCHEMA_VERSION,
        "ok": true,
        "command": format!("{} {}", row.noun, row.verb),
        "engine": env!("CARGO_PKG_VERSION"),
        "data": data,
    })
}

/// The error envelope: the same header, a typed `error`, no `data`.
fn error_envelope(row: &CommandRow, error: CliError) -> Value {
    json!({
        "schema_version": JSON_SCHEMA_VERSION,
        "ok": false,
        "command": format!("{} {}", row.noun, row.verb),
        "engine": env!("CARGO_PKG_VERSION"),
        "error": error,
    })
}

/// Pretty-print. Serializing an envelope of plain data cannot fail.
fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).expect("envelope serialize")
}

// ---------------------------------------------------------------------------
// Shared argument readers. Every verb reads its arguments through these, so
// "the table said this is a path" and "the code asked for a path" cannot
// drift into a panic.
// ---------------------------------------------------------------------------

/// A required `Text` positional or flag.
pub fn text(m: &ArgMatches, name: &str) -> Result<String, CliError> {
    m.get_one::<String>(name)
        .cloned()
        .ok_or_else(|| CliError::usage(format!("missing <{name}>")))
}

/// An optional `Text` flag.
pub fn opt_text(m: &ArgMatches, name: &str) -> Option<String> {
    m.get_one::<String>(name).cloned()
}

/// A required `Path` positional or flag.
pub fn path(m: &ArgMatches, name: &str) -> Result<PathBuf, CliError> {
    m.get_one::<PathBuf>(name)
        .cloned()
        .ok_or_else(|| CliError::usage(format!("missing <{name}>")))
}

/// An optional `Path` flag.
pub fn opt_path(m: &ArgMatches, name: &str) -> Option<PathBuf> {
    m.get_one::<PathBuf>(name).cloned()
}

/// A required `Integer` positional or flag.
pub fn integer(m: &ArgMatches, name: &str) -> Result<i64, CliError> {
    m.get_one::<i64>(name)
        .copied()
        .ok_or_else(|| CliError::usage(format!("missing <{name}>")))
}

/// An optional `Integer` flag.
pub fn opt_integer(m: &ArgMatches, name: &str) -> Option<i64> {
    m.get_one::<i64>(name).copied()
}

/// A repeatable positional or flag; empty when absent.
pub fn many(m: &ArgMatches, name: &str) -> Vec<String> {
    m.get_many::<String>(name).map(|v| v.cloned().collect()).unwrap_or_default()
}

/// A boolean switch.
pub fn switch(m: &ArgMatches, name: &str) -> bool {
    m.get_one::<bool>(name).copied().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::commands::COMMANDS;

    /// The whole generated tree, with a root that declares nothing itself.
    fn tree() -> ClapCommand {
        augment(ClapCommand::new("idl-rs"))
    }

    #[test]
    fn every_table_row_becomes_a_reachable_subcommand() {
        // Arrange
        let tree = tree();

        // Act / Assert — R230: the clap tree is generated from the rows.
        for row in COMMANDS {
            let group = tree
                .get_subcommands()
                .find(|c| c.get_name() == row.noun)
                .unwrap_or_else(|| panic!("no `{}` group", row.noun));
            assert!(
                group.get_subcommands().any(|c| c.get_name() == row.verb),
                "no `{row}` subcommand"
            );
        }
    }

    #[test]
    fn every_generated_subcommand_accepts_json() {
        // Arrange
        let tree = tree();

        // Act / Assert
        for row in COMMANDS {
            let parsed = tree
                .clone()
                .try_get_matches_from(sample_argv(row, &["--json"]))
                .unwrap_or_else(|e| panic!("`{row} --json` did not parse: {e}"));
            let (_, m) = parsed.subcommand().unwrap();
            let (_, m) = m.subcommand().unwrap();
            assert!(switch(m, "json"), "`{row} --json` did not set the flag");
        }
    }

    #[test]
    fn every_writer_accepts_dry_run_and_no_reader_does() {
        // Arrange
        let tree = tree();

        // Act / Assert
        for row in COMMANDS {
            let parsed = tree.clone().try_get_matches_from(sample_argv(row, &["--dry-run"]));
            assert_eq!(parsed.is_ok(), row.writer, "`{row} --dry-run` acceptance is wrong");
        }
    }

    #[test]
    fn every_data_dir_command_accepts_data_dir_and_no_other_does() {
        // Arrange
        let tree = tree();

        // Act / Assert
        for row in COMMANDS {
            let parsed =
                tree.clone().try_get_matches_from(sample_argv(row, &["--data-dir", "d"]));
            assert_eq!(parsed.is_ok(), row.data_dir, "`{row} --data-dir` acceptance is wrong");
        }
    }

    #[test]
    fn a_choice_flag_rejects_a_value_outside_its_closed_list() {
        // Arrange
        let tree = tree();

        // Act
        let parsed =
            tree.try_get_matches_from(["idl-rs", "workbook", "new", "wb.idl0wb", "--template", "racecar"]);

        // Assert
        assert!(parsed.is_err());
    }

    #[test]
    fn a_choice_flag_takes_the_default_the_table_declares() {
        // Arrange
        let tree = tree();

        // Act
        let parsed = tree.try_get_matches_from(["idl-rs", "workbook", "new", "wb.idl0wb"]).unwrap();

        // Assert
        let (_, m) = parsed.subcommand().unwrap();
        let (_, m) = m.subcommand().unwrap();
        assert_eq!(opt_text(m, "template").as_deref(), Some("blank"));
    }

    #[test]
    fn an_integer_argument_rejects_a_non_numeric_value() {
        // Arrange
        let tree = tree();

        // Act
        let parsed =
            tree.try_get_matches_from(["idl-rs", "session", "set-start", "s1", "yesterday"]);

        // Assert
        assert!(parsed.is_err());
    }

    #[test]
    fn a_repeatable_positional_collects_every_value() {
        // Arrange
        let tree = tree();

        // Act
        let parsed = tree
            .try_get_matches_from(["idl-rs", "library", "rebuild", "s1", "s2", "s3"])
            .unwrap();

        // Assert
        let (_, m) = parsed.subcommand().unwrap();
        let (_, m) = m.subcommand().unwrap();
        assert_eq!(many(m, "sessions"), vec!["s1", "s2", "s3"]);
    }

    #[test]
    fn every_workbook_verb_that_takes_a_track_also_takes_a_main_lap() {
        // Arrange
        let tree = tree();
        // A track *artifact*, not `session list --track`, which is a text
        // filter on a track id and has no laps to number.
        let rows: Vec<&CommandRow> = COMMANDS
            .iter()
            .filter(|r| {
                r.flags.iter().any(|f| f.long == "track" && f.kind == Some(ValueKind::Path))
            })
            .collect();

        // Act / Assert — a track without a main lap means lap-scoped
        // expressions silently read the whole session, so the two flags
        // travel together.
        assert!(!rows.is_empty());
        for row in rows {
            let parsed =
                tree.clone().try_get_matches_from(sample_argv(row, &["--main-lap", "2"]));
            assert!(parsed.is_ok(), "`{row} --main-lap` did not parse");
        }
    }

    #[test]
    fn a_missing_required_positional_is_a_parse_error() {
        // Arrange
        let tree = tree();

        // Act
        let parsed = tree.try_get_matches_from(["idl-rs", "session", "show"]);

        // Assert
        assert!(parsed.is_err());
    }

    #[test]
    fn a_verb_outside_the_table_is_a_parse_error() {
        // Arrange
        let tree = tree();

        // Act
        let parsed = tree.try_get_matches_from(["idl-rs", "session", "destroy", "s1"]);

        // Assert
        assert!(parsed.is_err());
    }

    #[test]
    fn the_data_dir_falls_back_to_the_environment_variable_when_the_flag_is_absent() {
        // Arrange
        let tree = tree();
        let parsed = tree.try_get_matches_from(["idl-rs", "catalog", "verify"]).unwrap();
        let (_, m) = parsed.subcommand().unwrap();
        let (_, m) = m.subcommand().unwrap();

        // Act — the variable is process-wide, so this test sets and clears it
        // around one read rather than relying on the ambient environment.
        std::env::set_var(DATA_DIR_ENV, "from-env");
        let ctx = Ctx::from_matches(m);
        std::env::remove_var(DATA_DIR_ENV);

        // Assert
        assert_eq!(ctx.data_dir().unwrap(), Path::new("from-env"));
    }

    #[test]
    fn the_flag_wins_over_the_environment_variable() {
        // Arrange
        let tree = tree();
        let parsed = tree
            .try_get_matches_from(["idl-rs", "catalog", "verify", "--data-dir", "from-flag"])
            .unwrap();
        let (_, m) = parsed.subcommand().unwrap();
        let (_, m) = m.subcommand().unwrap();

        // Act
        std::env::set_var(DATA_DIR_ENV, "from-env");
        let ctx = Ctx::from_matches(m);
        std::env::remove_var(DATA_DIR_ENV);

        // Assert
        assert_eq!(ctx.data_dir().unwrap(), Path::new("from-flag"));
    }

    #[test]
    fn a_missing_data_dir_is_a_usage_error_naming_both_ways_to_set_it() {
        // Arrange
        let ctx = Ctx { json: false, dry_run: false, data_dir: None };

        // Act
        let err = ctx.data_dir().unwrap_err();

        // Assert
        assert_eq!(err.kind, ErrorKind::Usage);
        assert!(err.message.contains("--data-dir"));
        assert!(err.message.contains(DATA_DIR_ENV));
    }

    #[test]
    fn the_success_envelope_states_its_schema_version_and_command() {
        // Arrange
        let row = find("session", "list").unwrap();

        // Act
        let envelope = success_envelope(row, &json!({ "sessions": [] }));

        // Assert
        assert_eq!(envelope["schema_version"], JSON_SCHEMA_VERSION);
        assert_eq!(envelope["ok"], true);
        assert_eq!(envelope["command"], "session list");
    }

    #[test]
    fn the_error_envelope_carries_the_typed_kind_and_no_data() {
        // Arrange
        let row = find("session", "show").unwrap();

        // Act
        let envelope = error_envelope(row, CliError::new(ErrorKind::NotFound, "no such session"));

        // Assert
        assert_eq!(envelope["ok"], false);
        assert_eq!(envelope["error"]["kind"], "not_found");
        assert!(envelope.get("data").is_none());
    }

    /// A minimal argv for `row` — every required positional filled with a
    /// placeholder of the right shape — plus `extra`.
    fn sample_argv(row: &CommandRow, extra: &[&str]) -> Vec<String> {
        let mut argv = vec!["idl-rs".to_string(), row.noun.to_string(), row.verb.to_string()];
        for arg in row.args.iter().filter(|a| a.required) {
            argv.push(match arg.kind {
                ValueKind::Integer => "1".to_string(),
                ValueKind::Choice => arg.choices.first().copied().unwrap_or("x").to_string(),
                _ => "x".to_string(),
            });
        }
        argv.extend(extra.iter().map(|s| (*s).to_string()));
        argv
    }
}
