//! Sidecar-ffmpeg export driver for idl-rs video overlays. The engine
//! (`idl-rs`) is pure and never spawns processes; this crate is the one
//! place that runs `ffprobe`/`ffmpeg`, consumed by both the CLI and the
//! FRB bridge. Engine-agnostic: rendered frames arrive through a closure.
//! See docs/IDL0_SPEC.md §33.5.

pub mod args;
pub mod export;
pub mod probe;

pub use args::ExportPlan;
pub use export::{run_export, ExportError, ExportErrorKind, Progress};
pub use probe::{parse_ffprobe_json, probe, VideoProbe};
