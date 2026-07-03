//! FRB wrappers + external-type mirrors for `idl_rs::fft`.

pub use idl_rs::fft::{Averaging, Detrend, FftWindow, Scaling, WelchResult};

/// Mirror of [`idl_rs::fft::FftWindow`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(FftWindow))]
pub enum _FftWindow {
    Rectangular,
    Hann,
    Hamming,
}

/// Mirror of [`idl_rs::fft::Detrend`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(Detrend))]
pub enum _Detrend {
    None,
    Mean,
    Linear,
}

/// Mirror of [`idl_rs::fft::Averaging`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(Averaging))]
pub enum _Averaging {
    Mean,
    Median,
}

/// Mirror of [`idl_rs::fft::Scaling`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(Scaling))]
pub enum _Scaling {
    Magnitude,
    Density,
}

/// Mirror of [`idl_rs::fft::WelchResult`] for FRB codegen.
#[flutter_rust_bridge::frb(mirror(WelchResult))]
pub struct _WelchResult {
    pub freqs_hz: Vec<f64>,
    pub values: Vec<f64>,
}

// The former sync `fft`/`welch` wrappers were pruned (2026-06-11): the app
// computes spectra via the handle's `welch_channel` (session.rs), so samples
// never cross FFI, and a sync full-vector call would block the UI thread.
// This module survives for the mirrored types above, which `welch_channel`
// takes and returns.
