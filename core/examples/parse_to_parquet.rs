//! Parses the committed synthetic ride and writes it out as `data.parquet`.
//!
//!   cargo run -p idl-rs --example parse_to_parquet [-- <out-dir>]
//!
//! Uses only `tests/fixtures/synth-3lap.idl0` (generated, not a real ride).
//! `<out-dir>` is the data root; the file lands at
//! `<out-dir>/sessions/<session_id>/data.parquet`. Default: a folder under
//! the system temp directory.

use std::path::PathBuf;

use idl_rs::parse::{parse, IDL0_IMPORTER_VERSION};
use idl_rs::session::synthesis::synthesize_base_channels;
use idl_rs::store::atomic::sha256_hex;
use idl_rs::store::parquet::write_session_parquet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synth-3lap.idl0");
    let out_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("idl-rs-example"));

    // Parse the raw log; the content hash is the session's provenance.
    let bytes = std::fs::read(&input)?;
    let mut session = parse(&bytes)?.session;
    session.blob_sha256 = sha256_hex(&bytes);
    synthesize_base_channels(&mut session);

    println!("session {}: {} channels", session.session_id, session.channels.len());
    for ch in &session.channels {
        println!("  {:<24} {:>8} samples @ {} Hz nominal", ch.channel_id, ch.t_us.len(), ch.nominal_rate_hz);
    }

    let path = write_session_parquet(&out_dir, &session, IDL0_IMPORTER_VERSION)?;
    println!("wrote {}", path.display());
    Ok(())
}
