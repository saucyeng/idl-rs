//! idl-rs — pure data-acquisition engine for IDL0 (.idl0) files.
//!
//! Parsing, data model, DSP, and analysis. No Flutter, no flutter_rust_bridge,
//! no clap, no I/O beyond std::fs. Consumed by the Flutter app (via
//! idl-rs-bridge), the idl-rs CLI, and future Python/WASM bindings.

pub mod calibration;
pub mod chart_decimation;
pub mod clip_reconstruct;
pub mod config;
pub mod estimate;
pub mod export;
pub mod fft;
pub mod filters;
pub mod gps;
pub mod histogram;
pub mod integration;
pub mod laps;
pub mod math;
pub mod parse;
pub mod rotation;
pub mod scatter;
pub mod session;
pub mod spectrogram;
pub mod statistics;
pub mod table;
pub mod track_artifact;
pub mod track_projection;
pub mod tracks;
pub mod variance;
pub mod workbook;
