//! Evaluates workbook math expressions against the committed synthetic ride.
//!
//!   cargo run -p idl-rs --example math_channel
//!
//! Uses only `tests/fixtures/synth-3lap.idl0` (generated, not a real ride).
//! The expression is the same language a workbook `math` cell takes: here a
//! 2nd-order 5 Hz Butterworth low-pass of an accelerometer channel, then the
//! RMS of the filtered result.

use std::path::PathBuf;

use idl_rs::math::eval::{evaluate, MathLapContext};
use idl_rs::math::parse::parse;
use idl_rs::math::units::{infer, Unit};
use idl_rs::session::handle::SessionHandle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synth-3lap.idl0");
    let handle = SessionHandle::from_path(path.to_str().ok_or("fixture path is not UTF-8")?)?;
    let accel = "IMU0_AccelZ";
    let laps = MathLapContext::empty(); // no track, so lap-aware functions are not used here

    for expr in [
        format!("butter(2, 5, \"low\", [{accel}])"),
        format!("rms(butter(2, 5, \"low\", [{accel}]))"),
    ] {
        let out = evaluate(&expr, &handle, &laps)?;
        let (unit, _notes) = infer(&parse(&expr)?, &handle);
        let unit = match unit {
            Unit::Known(u) => u.to_string(),
            _ => "unit not determined".to_string(),
        };
        match out.samples.as_slice() {
            [v] => println!("{expr}\n  = {v:.6} {unit}"),
            s => println!("{expr}\n  = {} samples @ {} Hz, first {:.6} {unit}", s.len(), out.sample_rate_hz, s[0]),
        }
    }
    Ok(())
}
