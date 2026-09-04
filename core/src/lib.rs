//! idl-rs — pure data-acquisition engine for IDL0 (.idl0) files.
//!
//! Parsing, data model, DSP, and analysis. No Tauri, no async runtime,
//! no clap, no I/O beyond std::fs. Consumed by the idl1 app (via
//! idl-rs-tauri) and the idl-rs CLI, and future Python/WASM bindings.

pub mod calibration;
pub mod chart_decimation;
pub mod clip_reconstruct;
pub mod colormap;
pub mod config;
pub mod estimate;
pub mod export;
pub mod fft;
pub mod filters;
pub mod gps;
pub mod histogram;
pub mod histogram2d;
pub mod integration;
pub mod laps;
pub mod math;
pub mod parse;
pub mod raster;
pub mod rotation;
pub mod scatter;
pub mod session;
pub mod spectrogram;
pub mod statistics;
pub mod store;
pub mod table;
pub mod tile;
pub mod track_artifact;
pub mod track_projection;
pub mod tracks;
pub mod variance;
pub mod workbook;

/// The engine's crate version, stamped into derived-file hashes and reported
/// over IPC so every device can prove it computes with the same engine.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
