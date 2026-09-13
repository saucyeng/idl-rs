//! The one command table (ruling R230) and the core operations the rows
//! behind the newer verbs call.
//!
//! R230: `idl-rs <noun> <verb> [args] [--flags]`, with one table as the
//! source of truth for the CLI's subcommands, the generated reference, and
//! (later, R228 item 2) the agent tool schemas. Adding a command is one row
//! plus one core function; there is no CLI-only logic.
//!
//! This module lives in `core`, not `cli`, for two reasons. The table is
//! pure data — no clap, no I/O, no Tauri — so it breaks none of CLAUDE.md
//! §2's purity rule. And `idl-rs-tauri` needs the same rows to derive the
//! agent tool schemas, which it cannot take from the CLI binary.
//!
//! - [`table`] — the row shape, the closed vocabularies, and [`table::COMMANDS`].
//! - [`markdown`] — the table rendered as Markdown and as JSON (`docs cli`).
//! - [`lap_ops`] — assembling a `MathLapContext` from detected laps.
//! - [`session_ops`] — the filtering and metadata edits behind `session list`/`set-meta`.
//! - [`workbook_ops`] — the skeletons behind `workbook new`.

pub mod lap_ops;
pub mod markdown;
pub mod session_ops;
pub mod table;
pub mod workbook_ops;

pub use table::{Arg, CommandRow, Flag, Status, Tier, ValueKind, COMMANDS, NOUNS, VERBS};
