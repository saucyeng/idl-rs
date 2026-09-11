//! The command table itself (ruling R230): the row shape, the closed noun
//! and verb vocabularies, and [`COMMANDS`].
//!
//! # Why the vocabularies are two lists, not one
//!
//! R230 closes the verb vocabulary and says "a new verb needs a ruling".
//! [`VERBS`] is that closed list verbatim. [`VERBS_RULED`] holds the verbs a
//! *later* ruling added, each with the ruling that added it — R229 rules in
//! `set-start`, `set-meta`, `cells`, `data`, `detect` and `laps` by writing
//! them out, R222 shipped `docs workbook`, R230 item 2 itself writes
//! `docs cli`, and R197's `library` verbs keep their names. A verb in
//! neither list fails [`tests`]' enumeration, which is the check R230 asks
//! for; adding one means editing [`VERBS_RULED`] and naming the ruling,
//! which is deliberate rather than accidental.
//!
//! # Deprecated rows
//!
//! The verbs that predate R230 (`idl-rs info`, `idl-rs fft`, top-level
//! `import`, …) are not in the table: they are noun-less and R230's grammar
//! has no place for them. They stay registered by the CLI as deprecated
//! aliases — R230 allows breaking changes only at a major version, and sets
//! the precedent itself for top-level `import`. A row that *is* in the table
//! but superseded carries [`Status::Deprecated`] and is skipped by the
//! vocabulary test.
//!
//! # Uniform behaviour
//!
//! R230 item 3's uniform flags are not repeated on every row. A row declares
//! what it *is* — [`CommandRow::data_dir`] for a command that works against
//! a data directory, [`CommandRow::writer`] for one that writes — and
//! [`CommandRow::all_flags`] materialises `--json`, `--data-dir` and
//! `--dry-run` from those. Uniformity is then a property of the table, not a
//! thing each row has to remember.

use std::fmt;

/// What a positional argument or a flag's value is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// A filesystem path.
    Path,
    /// A free string (an id, a name, a comment).
    Text,
    /// A signed integer. Milliseconds where the name says so.
    Integer,
    /// One of [`Flag::choices`] / [`Arg::choices`].
    Choice,
}

impl ValueKind {
    /// The placeholder the help text and the generated Markdown print.
    pub fn placeholder(self) -> &'static str {
        match self {
            ValueKind::Path => "<path>",
            ValueKind::Text => "<text>",
            ValueKind::Integer => "<n>",
            ValueKind::Choice => "<choice>",
        }
    }

    /// The wire name used in the JSON export and the Markdown table.
    pub fn as_str(self) -> &'static str {
        match self {
            ValueKind::Path => "path",
            ValueKind::Text => "text",
            ValueKind::Integer => "integer",
            ValueKind::Choice => "choice",
        }
    }
}

/// One positional argument, in command-line order.
#[derive(Debug, Clone, Copy)]
pub struct Arg {
    /// Lower-kebab, as the help prints it.
    pub name: &'static str,
    pub kind: ValueKind,
    /// `false` makes this and every later argument optional.
    pub required: bool,
    /// The closed value set when `kind` is [`ValueKind::Choice`]; empty otherwise.
    pub choices: &'static [&'static str],
    /// One line, sentence case, no trailing full stop.
    pub help: &'static str,
}

/// One long flag. Short forms are not in the table: R230's uniform surface
/// is long flags only, so a flag means the same thing on every command.
#[derive(Debug, Clone, Copy)]
pub struct Flag {
    /// Without the leading `--`.
    pub long: &'static str,
    /// `None` for a boolean switch.
    pub kind: Option<ValueKind>,
    /// The flag may be given more than once, accumulating.
    pub repeatable: bool,
    /// The closed value set when `kind` is `Some(ValueKind::Choice)`; empty otherwise.
    pub choices: &'static [&'static str],
    /// Printed in the help, and used by the CLI as clap's default.
    pub default: Option<&'static str>,
    /// One line, sentence case, no trailing full stop.
    pub help: &'static str,
}

impl Flag {
    /// A boolean switch.
    const fn switch(long: &'static str, help: &'static str) -> Flag {
        Flag { long, kind: None, repeatable: false, choices: &[], default: None, help }
    }

    /// A flag taking one value of `kind`.
    const fn value(long: &'static str, kind: ValueKind, help: &'static str) -> Flag {
        Flag { long, kind: Some(kind), repeatable: false, choices: &[], default: None, help }
    }
}

/// How often a command is reached for. The same three values as the app's
/// `commandTiers.ts` (`CommandTier`), so a row that appears in both tables
/// can be compared field for field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Core,
    Occasional,
    Rare,
}

impl Tier {
    /// The wire name, matching the TypeScript union's members exactly.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Core => "core",
            Tier::Occasional => "occasional",
            Tier::Rare => "rare",
        }
    }
}

/// Whether a row is the current spelling or a superseded one kept for one
/// release (R230: breaking changes only at a major version).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Current,
    Deprecated,
}

impl Status {
    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Current => "current",
            Status::Deprecated => "deprecated",
        }
    }
}

/// One command: everything the CLI, the generated reference and (later) an
/// agent tool schema need to know about it.
#[derive(Debug, Clone, Copy)]
pub struct CommandRow {
    /// A member of [`NOUNS`].
    pub noun: &'static str,
    /// A member of [`VERBS`] or [`VERBS_RULED`].
    pub verb: &'static str,
    /// Positional arguments, in order.
    pub args: &'static [Arg],
    /// Command-specific flags. The uniform ones come from [`CommandRow::all_flags`].
    pub flags: &'static [Flag],
    /// The `idl_rs` path of the function that does the work, for the
    /// reference and for reviewers checking "no CLI-only logic".
    pub core_fn: &'static str,
    /// The name of the JSON shape `--json` emits, or `None` when the command
    /// writes an artifact rather than a payload. Where a C3 DTO exists the
    /// name is that DTO's.
    pub json_shape: Option<&'static str>,
    /// One line, sentence case, no trailing full stop.
    pub help: &'static str,
    pub tier: Tier,
    pub status: Status,
    /// The command works against a data directory, so it takes `--data-dir`
    /// and honours `IDL1_DATA_DIR` (R230 item 3).
    pub data_dir: bool,
    /// The command writes. Every writer takes `--dry-run` (R230 item 3).
    pub writer: bool,
}

impl CommandRow {
    /// The shared id, `noun.verbInCamelCase` — the spelling the app's
    /// `commandTiers.ts` uses, so the two tables can be compared on the ids
    /// they have in common (`workbook.new`, `library.rebuild`).
    pub fn id(&self) -> String {
        format!("{}.{}", self.noun, camel_case(self.verb))
    }

    /// How the command is typed, e.g. `session set-start <id> <utc-ms>`.
    pub fn usage(&self) -> String {
        let mut out = format!("{} {}", self.noun, self.verb);
        for arg in self.args {
            if arg.required {
                out.push_str(&format!(" <{}>", arg.name));
            } else {
                out.push_str(&format!(" [{}]", arg.name));
            }
        }
        out
    }

    /// This row's own flags followed by R230 item 3's uniform ones, in the
    /// order the help prints them. Never contains a duplicate `long`: a row
    /// that declares `--json` itself would shadow the uniform one, so the
    /// uniform flag is skipped and [`tests`] asserts no row does that.
    pub fn all_flags(&self) -> Vec<Flag> {
        let mut flags: Vec<Flag> = self.flags.to_vec();
        if self.data_dir {
            flags.push(DATA_DIR_FLAG);
        }
        if self.writer {
            flags.push(DRY_RUN_FLAG);
        }
        flags.push(JSON_FLAG);
        flags
    }
}

impl fmt::Display for CommandRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.usage())
    }
}

/// `--json`, on every command (R230 item 3).
pub const JSON_FLAG: Flag = Flag::switch(
    "json",
    "Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr",
);

/// `--data-dir`, on every command that works against a data directory. Falls
/// back to `IDL1_DATA_DIR` (R230 item 3).
pub const DATA_DIR_FLAG: Flag = Flag::value(
    "data-dir",
    ValueKind::Path,
    "Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR",
);

/// `--dry-run`, on every writer (R230 item 3).
pub const DRY_RUN_FLAG: Flag =
    Flag::switch("dry-run", "Report what would change and write nothing");

/// The environment variable [`DATA_DIR_FLAG`] falls back to.
pub const DATA_DIR_ENV: &str = "IDL1_DATA_DIR";

/// The `schema_version` every `--json` payload carries (R230 item 3).
pub const JSON_SCHEMA_VERSION: u32 = 1;

/// R230's closed noun vocabulary.
pub const NOUNS: &[&str] =
    &["session", "workbook", "track", "catalog", "library", "device", "docs"];

/// R230's closed verb vocabulary, verbatim.
pub const VERBS: &[&str] = &[
    "list", "show", "new", "check", "eval", "set", "import", "export", "scan", "fold-in", "index",
    "rebuild", "verify", "delete",
];

/// Verbs a ruling later than R230's list added, each with that ruling. R230
/// says "a new verb needs a ruling"; this is where the rulings are recorded.
pub const VERBS_RULED: &[(&str, &str)] = &[
    ("set-start", "R229"),
    ("set-meta", "R229"),
    ("cells", "R229"),
    ("data", "R229"),
    ("detect", "R229"),
    ("laps", "R229"),
    ("stale", "R197"),
    ("workbook", "R222"),
    ("cli", "R230 item 2"),
];

/// Whether `verb` is in either vocabulary.
pub fn verb_is_known(verb: &str) -> bool {
    VERBS.contains(&verb) || VERBS_RULED.iter().any(|(v, _)| *v == verb)
}

/// Whether `noun` is in [`NOUNS`].
pub fn noun_is_known(noun: &str) -> bool {
    NOUNS.contains(&noun)
}

/// `set-start` → `setStart`, `fold-in` → `foldIn`, `new` → `new`.
fn camel_case(verb: &str) -> String {
    let mut out = String::with_capacity(verb.len());
    let mut upper_next = false;
    for ch in verb.chars() {
        if ch == '-' {
            upper_next = true;
        } else if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// Look one row up by `noun` and `verb`.
pub fn find(noun: &str, verb: &str) -> Option<&'static CommandRow> {
    COMMANDS.iter().find(|r| r.noun == noun && r.verb == verb)
}

/// Every noun that has at least one row, in [`NOUNS`] order.
pub fn nouns_in_use() -> Vec<&'static str> {
    NOUNS.iter().copied().filter(|n| COMMANDS.iter().any(|r| r.noun == *n)).collect()
}

/// Every row for one noun, in table order.
pub fn rows_for(noun: &str) -> Vec<&'static CommandRow> {
    COMMANDS.iter().filter(|r| r.noun == noun).collect()
}

// ---------------------------------------------------------------------------
// Argument and flag definitions, shared where two rows take the same thing.
// ---------------------------------------------------------------------------

const SESSION_ID_ARG: Arg = Arg {
    name: "id",
    kind: ValueKind::Text,
    required: true,
    choices: &[],
    help: "Session id, as `session list` prints it",
};

const WORKBOOK_FILE_ARG: Arg = Arg {
    name: "file",
    kind: ValueKind::Path,
    required: true,
    choices: &[],
    help: "Path to an `.idl0wb` workbook",
};

/// `--session`, where a workbook command needs data to evaluate against.
const SESSION_PATH_FLAG: Flag = Flag::value(
    "session",
    ValueKind::Path,
    "Session to evaluate against: an `.idl0` file, or a session id inside --data-dir",
);

/// `--track`, where lap-bound rows need a track artifact.
const TRACK_PATH_FLAG: Flag = Flag::value(
    "track",
    ValueKind::Path,
    "`.idl0t` track artifact, required only when a cell is lap-bound",
);

/// The templates `workbook new` knows. Kept a closed set in the table so
/// adding one is a new value rather than a new verb.
pub const WORKBOOK_TEMPLATES: &[&str] = &["blank", "session"];

/// What `workbook export` can write.
pub const WORKBOOK_EXPORT_FORMATS: &[&str] = &["md", "json"];

// ---------------------------------------------------------------------------
// The table.
// ---------------------------------------------------------------------------

/// Every command, grouped by noun in [`NOUNS`] order.
pub const COMMANDS: &[CommandRow] = &[
    // --- session ----------------------------------------------------------
    CommandRow {
        noun: "session",
        verb: "list",
        args: &[],
        flags: &[
            Flag::value("venue", ValueKind::Text, "Only sessions whose venue matches, case-insensitively"),
            Flag::value("track", ValueKind::Text, "Only sessions that visited this track id"),
            Flag::value("since", ValueKind::Integer, "Only sessions starting at or after this Unix epoch millisecond"),
            Flag::value("until", ValueKind::Integer, "Only sessions starting at or before this Unix epoch millisecond"),
            Flag::value("tag", ValueKind::Text, "Only sessions carrying this tag, case-insensitively"),
        ],
        core_fn: "store::catalog_read::list_sessions + commands::session_ops::filter_sessions",
        json_shape: Some("SessionSummary[]"),
        help: "List catalogued sessions, most recent first",
        tier: Tier::Core,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "session",
        verb: "show",
        args: &[SESSION_ID_ARG],
        flags: &[],
        core_fn: "store::catalog_read::get_session",
        json_shape: Some("SessionDetail"),
        help: "Print one session's metadata, channels and laps",
        tier: Tier::Core,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "session",
        verb: "laps",
        args: &[SESSION_ID_ARG],
        flags: &[],
        core_fn: "store::catalog_read::list_laps",
        json_shape: Some("LapSummary[]"),
        help: "Print one session's indexed laps",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "session",
        verb: "set-start",
        args: &[
            SESSION_ID_ARG,
            Arg {
                name: "utc-ms",
                kind: ValueKind::Integer,
                required: true,
                choices: &[],
                help: "Recording start, Unix epoch milliseconds; must be greater than zero",
            },
        ],
        flags: &[],
        core_fn: "store::session_json::set_session_start",
        json_shape: Some("SessionJson"),
        help: "Set a session's recording start time by hand",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    CommandRow {
        noun: "session",
        verb: "set-meta",
        args: &[SESSION_ID_ARG],
        flags: &[
            Flag::value("venue", ValueKind::Text, "Venue name"),
            Flag::value("rider", ValueKind::Text, "Rider name"),
            Flag::value("bike", ValueKind::Text, "Bike name"),
            Flag::value("event", ValueKind::Text, "Event name"),
            Flag::value("event-session", ValueKind::Text, "Event session, e.g. `Qualifying 2`"),
            Flag::value("notes", ValueKind::Text, "Long comment"),
            Flag::value("tag", ValueKind::Text, "Tag"),
        ],
        core_fn: "commands::session_ops::apply_session_meta",
        json_shape: Some("SessionJson"),
        help: "Set a session's descriptive metadata; omitted fields are left alone",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    CommandRow {
        noun: "session",
        verb: "import",
        args: &[Arg {
            name: "file",
            kind: ValueKind::Path,
            required: true,
            choices: &[],
            help: "Log file to import (`.idl0`, `.fit`, `.gpx`, `.csv`)",
        }],
        flags: &[],
        core_fn: "store::import::import_file_path",
        json_shape: Some("ImportReport"),
        help: "Import one log file into the data directory",
        tier: Tier::Core,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    // --- workbook ---------------------------------------------------------
    CommandRow {
        noun: "workbook",
        verb: "new",
        args: &[Arg {
            name: "file",
            kind: ValueKind::Path,
            required: true,
            choices: &[],
            help: "Workbook file to create; refuses to overwrite an existing one",
        }],
        flags: &[
            Flag {
                long: "template",
                kind: Some(ValueKind::Choice),
                repeatable: false,
                choices: WORKBOOK_TEMPLATES,
                default: Some("blank"),
                help: "Skeleton to write: `blank` is front matter only, `session` adds a math and a table cell",
            },
            Flag::value("session", ValueKind::Text, "Session id to reference from the `session` template"),
        ],
        core_fn: "commands::workbook_ops::new_workbook",
        json_shape: Some("WorkbookNewReport"),
        help: "Create a new workbook from a built-in skeleton",
        tier: Tier::Core,
        status: Status::Current,
        data_dir: false,
        writer: true,
    },
    CommandRow {
        noun: "workbook",
        verb: "check",
        args: &[WORKBOOK_FILE_ARG],
        flags: &[SESSION_PATH_FLAG, TRACK_PATH_FLAG],
        core_fn: "workbook::v3::parse_workbook",
        json_shape: Some("WorkbookCheckReport"),
        help: "Parse a workbook and report every structural problem",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: false,
        writer: false,
    },
    CommandRow {
        noun: "workbook",
        verb: "cells",
        args: &[WORKBOOK_FILE_ARG],
        flags: &[],
        core_fn: "workbook::v3::parse_workbook",
        json_shape: Some("WorkbookCellSummary[]"),
        help: "List a workbook's cells in document order",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: false,
        writer: false,
    },
    CommandRow {
        noun: "workbook",
        verb: "eval",
        args: &[WORKBOOK_FILE_ARG],
        flags: &[SESSION_PATH_FLAG, TRACK_PATH_FLAG],
        core_fn: "workbook::v3::eval_cells",
        json_shape: Some("WorkbookEvalReport"),
        help: "Evaluate a workbook's math and table cells against a session",
        tier: Tier::Core,
        status: Status::Current,
        data_dir: false,
        writer: false,
    },
    CommandRow {
        noun: "workbook",
        verb: "data",
        args: &[
            WORKBOOK_FILE_ARG,
            Arg {
                name: "cell",
                kind: ValueKind::Text,
                required: true,
                choices: &[],
                help: "Cell id, as `workbook cells` prints it",
            },
        ],
        flags: &[SESSION_PATH_FLAG, TRACK_PATH_FLAG],
        core_fn: "workbook::v3::eval_cells",
        json_shape: Some("WorkbookCellData"),
        help: "Print one cell's evaluated series or table",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: false,
        writer: false,
    },
    CommandRow {
        noun: "workbook",
        verb: "export",
        args: &[WORKBOOK_FILE_ARG],
        flags: &[
            Flag::value("out", ValueKind::Path, "File to write; stdout when omitted"),
            Flag {
                long: "format",
                kind: Some(ValueKind::Choice),
                repeatable: false,
                choices: WORKBOOK_EXPORT_FORMATS,
                default: Some("md"),
                help: "`md` re-renders the workbook source, `json` writes its evaluated cells",
            },
            SESSION_PATH_FLAG,
            TRACK_PATH_FLAG,
        ],
        core_fn: "workbook::v3::render_workbook",
        json_shape: None,
        help: "Write a workbook out as Markdown or as evaluated JSON",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: false,
        writer: true,
    },
    // --- track ------------------------------------------------------------
    CommandRow {
        noun: "track",
        verb: "list",
        args: &[],
        flags: &[],
        core_fn: "store::catalog_read::list_tracks",
        json_shape: Some("TrackSummary[]"),
        help: "List the tracks in the library",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "track",
        verb: "detect",
        args: &[SESSION_ID_ARG],
        flags: &[],
        core_fn: "store::lap_index::index_session_laps",
        json_shape: Some("LapIndexReport"),
        help: "Re-detect track visits and laps for one session",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    // --- catalog ----------------------------------------------------------
    CommandRow {
        noun: "catalog",
        verb: "verify",
        args: &[],
        flags: &[],
        core_fn: "store::verify::verify",
        json_shape: Some("Finding[]"),
        help: "Run contract C4 §7's checks against the data directory",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "catalog",
        verb: "rebuild",
        args: &[],
        flags: &[],
        core_fn: "store::catalog::rebuild_catalog",
        json_shape: Some("RebuildReport"),
        help: "Rebuild the catalog index from the canonical files",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    // --- library (R197; these verbs keep the names they shipped with) -----
    CommandRow {
        noun: "library",
        verb: "fold-in",
        args: &[Arg {
            name: "folder",
            kind: ValueKind::Path,
            required: true,
            choices: &[],
            help: "Folder to fold into the library",
        }],
        flags: &[
            Flag::switch("move", "Delete each source file once its blob has verified in the store"),
            Flag::switch("recursive", "Descend into sub-directories"),
        ],
        core_fn: "store::import::import_file_path (per file)",
        json_shape: Some("FoldInReport"),
        help: "Import every importable file in a folder",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    CommandRow {
        noun: "library",
        verb: "scan",
        args: &[Arg {
            name: "folder",
            kind: ValueKind::Path,
            required: true,
            choices: &[],
            help: "Folder to scan",
        }],
        flags: &[Flag::switch("recursive", "Descend into sub-directories")],
        core_fn: "store::scan::scan_folder_with_blob_check",
        json_shape: Some("ScanReport"),
        help: "Preview what folding a folder in would import",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "library",
        verb: "stale",
        args: &[],
        flags: &[],
        core_fn: "store::catalog_read::list_stale_sessions",
        json_shape: Some("StaleSession[]"),
        help: "List sessions whose data.parquet predates this importer version",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: true,
        writer: false,
    },
    CommandRow {
        noun: "library",
        verb: "index",
        args: &[Arg {
            name: "sessions",
            kind: ValueKind::Text,
            required: false,
            choices: &[],
            help: "Session ids to index; with none, every session whose index is stale",
        }],
        flags: &[
            Flag::switch("all", "Consider every session, not only the stale ones"),
            Flag::switch("force", "Recompute even where the stamps are already current"),
            Flag::value("workers", ValueKind::Integer, "Pool width; defaults to physical cores minus one"),
        ],
        core_fn: "store::index_job::run_index_job",
        json_shape: Some("IndexJobReport"),
        help: "Detect track visits and laps for sessions whose index is missing or stale",
        tier: Tier::Occasional,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    CommandRow {
        noun: "library",
        verb: "rebuild",
        args: &[Arg {
            name: "sessions",
            kind: ValueKind::Text,
            required: false,
            choices: &[],
            help: "Session ids to rebuild; mutually exclusive with --all",
        }],
        flags: &[Flag::switch("all", "Rebuild every session `library stale` lists")],
        core_fn: "store::import::reimport_session",
        json_shape: Some("RebuildSessionsReport"),
        help: "Re-import listed sessions from their own blobs",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: true,
        writer: true,
    },
    // --- docs -------------------------------------------------------------
    CommandRow {
        noun: "docs",
        verb: "workbook",
        args: &[],
        flags: &[
            Flag::value("out", ValueKind::Path, "File to write; overwritten wholesale"),
            Flag::value("src", ValueKind::Path, "Directory of curated Markdown sections"),
        ],
        core_fn: "docs::render_workbook_reference",
        json_shape: None,
        help: "Regenerate the workbook reference from the engine's own catalogs",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: false,
        writer: true,
    },
    CommandRow {
        noun: "docs",
        verb: "cli",
        args: &[],
        flags: &[Flag::value("out", ValueKind::Path, "File to write; stdout when omitted")],
        core_fn: "commands::markdown::render_command_reference",
        json_shape: Some("CommandTable"),
        help: "Emit this command table as Markdown, or as JSON with --json",
        tier: Tier::Rare,
        status: Status::Current,
        data_dir: false,
        writer: true,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_row_uses_a_noun_from_the_closed_vocabulary() {
        // Arrange
        let rows = COMMANDS;

        // Act
        let strays: Vec<&str> =
            rows.iter().filter(|r| !noun_is_known(r.noun)).map(|r| r.noun).collect();

        // Assert — ruling R230's enumeration check, noun half.
        assert!(strays.is_empty(), "nouns outside NOUNS: {strays:?}");
    }

    #[test]
    fn every_row_uses_a_verb_from_the_closed_vocabulary_or_a_named_ruling() {
        // Arrange
        let rows = COMMANDS;

        // Act
        let strays: Vec<String> = rows
            .iter()
            .filter(|r| r.status == Status::Current && !verb_is_known(r.verb))
            .map(|r| r.usage())
            .collect();

        // Assert — ruling R230's enumeration check, verb half.
        assert!(strays.is_empty(), "verbs in neither VERBS nor VERBS_RULED: {strays:?}");
    }

    #[test]
    fn the_ruled_verb_list_is_exactly_what_the_rulings_added() {
        // Arrange
        let expected =
            ["set-start", "set-meta", "cells", "data", "detect", "laps", "stale", "workbook", "cli"];

        // Act
        let actual: Vec<&str> = VERBS_RULED.iter().map(|(v, _)| *v).collect();

        // Assert — adding a verb means editing this list and naming its ruling.
        assert_eq!(actual, expected);
    }

    #[test]
    fn no_ruled_verb_duplicates_one_already_in_the_closed_list() {
        // Arrange
        let closed: HashSet<&str> = VERBS.iter().copied().collect();

        // Act
        let overlap: Vec<&str> =
            VERBS_RULED.iter().map(|(v, _)| *v).filter(|v| closed.contains(v)).collect();

        // Assert
        assert!(overlap.is_empty(), "already in VERBS: {overlap:?}");
    }

    #[test]
    fn every_ruled_verb_names_the_ruling_that_added_it() {
        // Arrange
        let rows = VERBS_RULED;

        // Act
        let unnamed: Vec<&str> =
            rows.iter().filter(|(_, ruling)| !ruling.starts_with('R')).map(|(v, _)| *v).collect();

        // Assert
        assert!(unnamed.is_empty(), "no ruling recorded for: {unnamed:?}");
    }

    #[test]
    fn no_two_rows_share_a_noun_and_verb() {
        // Arrange
        let mut seen: HashSet<(&str, &str)> = HashSet::new();

        // Act
        let dupes: Vec<String> = COMMANDS
            .iter()
            .filter(|r| !seen.insert((r.noun, r.verb)))
            .map(|r| r.usage())
            .collect();

        // Assert
        assert!(dupes.is_empty(), "duplicate rows: {dupes:?}");
    }

    #[test]
    fn every_row_takes_json_and_every_writer_takes_dry_run() {
        // Arrange
        let rows = COMMANDS;

        // Act / Assert — R230 item 3's uniform behaviour.
        for row in rows {
            let flags = row.all_flags();
            let longs: Vec<&str> = flags.iter().map(|f| f.long).collect();

            assert!(longs.contains(&"json"), "{row} has no --json");
            assert_eq!(row.writer, longs.contains(&"dry-run"), "{row}'s --dry-run does not match `writer`");
            assert_eq!(row.data_dir, longs.contains(&"data-dir"), "{row}'s --data-dir does not match `data_dir`");
        }
    }

    #[test]
    fn no_row_declares_a_flag_that_shadows_a_uniform_one() {
        // Arrange
        let uniform = ["json", "data-dir", "dry-run"];

        // Act
        let shadowed: Vec<String> = COMMANDS
            .iter()
            .flat_map(|r| r.flags.iter().map(move |f| (r, f)))
            .filter(|(_, f)| uniform.contains(&f.long))
            .map(|(r, f)| format!("{r} --{}", f.long))
            .collect();

        // Assert
        assert!(shadowed.is_empty(), "rows redeclaring a uniform flag: {shadowed:?}");
    }

    #[test]
    fn no_row_has_a_duplicate_flag_or_argument_name() {
        // Arrange
        let rows = COMMANDS;

        // Act / Assert
        for row in rows {
            let mut flags = HashSet::new();
            for flag in row.all_flags() {
                assert!(flags.insert(flag.long), "{row} declares --{} twice", flag.long);
            }
            let mut args = HashSet::new();
            for arg in row.args {
                assert!(args.insert(arg.name), "{row} declares <{}> twice", arg.name);
            }
        }
    }

    #[test]
    fn a_required_argument_never_follows_an_optional_one() {
        // Arrange
        let rows = COMMANDS;

        // Act / Assert — clap cannot build the other order.
        for row in rows {
            let first_optional = row.args.iter().position(|a| !a.required);
            if let Some(at) = first_optional {
                assert!(
                    row.args[at..].iter().all(|a| !a.required),
                    "{row} has a required argument after an optional one"
                );
            }
        }
    }

    #[test]
    fn a_choice_valued_flag_lists_its_choices_and_defaults_to_one_of_them() {
        // Arrange
        let rows = COMMANDS;

        // Act / Assert
        for row in rows {
            for flag in row.flags {
                if flag.kind == Some(ValueKind::Choice) {
                    assert!(!flag.choices.is_empty(), "{row} --{} has no choices", flag.long);
                    if let Some(default) = flag.default {
                        assert!(
                            flag.choices.contains(&default),
                            "{row} --{}'s default `{default}` is not one of its choices",
                            flag.long
                        );
                    }
                } else {
                    assert!(flag.choices.is_empty(), "{row} --{} lists choices but is not a choice", flag.long);
                }
            }
        }
    }

    #[test]
    fn every_row_names_a_core_function_and_a_one_line_help() {
        // Arrange
        let rows = COMMANDS;

        // Act / Assert — R230: no CLI-only logic, and the help is one line.
        for row in rows {
            assert!(!row.core_fn.is_empty(), "{row} names no core function");
            assert!(!row.help.is_empty(), "{row} has no help");
            assert!(!row.help.contains('\n'), "{row}'s help is more than one line");
            assert!(!row.help.ends_with('.'), "{row}'s help ends with a full stop");
        }
    }

    #[test]
    fn the_shared_id_is_the_dotted_camel_case_spelling_the_app_table_uses() {
        // Arrange
        let new = find("workbook", "new").unwrap();
        let set_start = find("session", "set-start").unwrap();
        let fold_in = find("library", "fold-in").unwrap();

        // Act
        let ids = [new.id(), set_start.id(), fold_in.id()];

        // Assert
        assert_eq!(ids, ["workbook.new", "session.setStart", "library.foldIn"]);
    }

    #[test]
    fn usage_prints_required_arguments_in_angle_brackets_and_optional_in_square() {
        // Arrange
        let set_start = find("session", "set-start").unwrap();
        let index = find("library", "index").unwrap();

        // Act
        let usages = [set_start.usage(), index.usage()];

        // Assert
        assert_eq!(usages, ["session set-start <id> <utc-ms>", "library index [sessions]"]);
    }

    #[test]
    fn every_noun_in_use_has_at_least_one_row_and_device_has_none_yet() {
        // Arrange
        let in_use = nouns_in_use();

        // Act
        let has_device = in_use.contains(&"device");

        // Assert — `device` is in R230's vocabulary for the transport lane;
        // no row claims it yet, and that is not a failure.
        assert!(!has_device);
        assert_eq!(in_use, ["session", "workbook", "track", "catalog", "library", "docs"]);
    }

    #[test]
    fn rows_for_returns_a_nouns_rows_in_table_order() {
        // Arrange
        let noun = "catalog";

        // Act
        let verbs: Vec<&str> = rows_for(noun).iter().map(|r| r.verb).collect();

        // Assert
        assert_eq!(verbs, ["verify", "rebuild"]);
    }

    #[test]
    fn find_returns_nothing_for_a_noun_verb_pair_that_is_not_registered() {
        // Arrange
        let noun = "catalog";

        // Act
        let row = find(noun, "delete");

        // Assert
        assert!(row.is_none());
    }
}
