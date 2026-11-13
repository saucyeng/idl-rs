//! The core operations the CLI's newer verbs call (rulings R229, R230).
//!
//! R230: `idl-rs <noun> <verb> [args] [--flags]`. The command table itself
//! (the rows, the vocabularies, the generated reference) is the CLI's own
//! contract and lives in `idl-rs-cli`; what stays here is the engine-side
//! logic those rows call, so there is no CLI-only logic.
//!
//! - [`calibration_ops`] — calibration inputs and truth checks for a session.
//! - [`lap_ops`] — assembling a `MathLapContext` from detected laps.
//! - [`workbook_ops`] — the skeletons behind `workbook new`.

pub mod calibration_ops;
pub mod lap_ops;
pub mod workbook_ops;

/// The `schema_version` every CLI `--json` payload carries (R230 item 3),
/// shared by the command envelope and `docs workbook --json`. Unitless
/// integer, bumped when an envelope's shape changes.
pub const JSON_SCHEMA_VERSION: u32 = 1;
