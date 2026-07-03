//! Portable-workbook (`.idl0wb`) support for the engine: read the file and
//! apply its math channels to a session. Read-only on the format — authoring
//! stays in the app. See
//! `docs/superpowers/specs/2026-06-01-idl-rs-phase-3b-workbook-cli-design.md`.

pub mod apply;
pub mod model;
pub mod read;

pub use apply::{apply_workbook, ApplyReport, ChannelApplyResult};
pub use model::{Workbook, WorkbookTable, SUPPORTED_WORKBOOK_VERSION};
pub use read::{parse_workbook, read_workbook};
