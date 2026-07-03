//! FRB mirror for `idl_rs::spectrogram::SpectrogramResult`. The flat row-major
//! `power` buffer + explicit dims cross FFI (2-D arrays cannot).

use flutter_rust_bridge::frb;

pub use idl_rs::spectrogram::SpectrogramResult;

/// Mirror of [`idl_rs::spectrogram::SpectrogramResult`] for FRB codegen.
/// `power` is flat row-major `n_times × n_freqs`: frame `t`, bin `f` is at
/// `power[t * n_freqs + f]`. 2-D arrays cannot cross FRB, hence the flat
/// buffer + explicit dims.
#[frb(mirror(SpectrogramResult))]
pub struct _SpectrogramResult {
    pub freqs_hz: Vec<f64>,
    pub times_secs: Vec<f64>,
    pub power: Vec<f64>,
    pub n_times: u32,
    pub n_freqs: u32,
}
