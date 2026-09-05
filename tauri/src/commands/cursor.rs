//! Cursor command (C3 §3.7): the atomic cross-channel readout at one
//! instant, over L3's `idl_rs::cursor::cursor_readout`.
//!
//! Same idiom as `commands/workbook.rs`/`commands/device.rs`: the
//! `#[tauri::command]` is a one-line wrapper over a `_via`-suffixed plain
//! function taking `data_dir: &Path` — this module's own tests exercise the
//! `_via` function.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{IpcError, IpcErrorKind};
use crate::session_source::load_session;

/// `cursor_readout`'s return (C3 §3.7).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CursorReadout {
    /// Echoes the request, µs on the session's `t` axis (C1 §3.1).
    pub t_us: i64,
    /// channel_id → nearest recorded sample, `null` outside that channel's
    /// own recorded span (ledger R31).
    pub values: HashMap<String, Option<f64>>,
}

/// Core of `cursor_readout` (C3 §3.7). `t_us` is µs on the session's `t`
/// axis (C1 §3.1).
///
/// Existence-checks every id in `channels` against `session.channels`
/// **before** reading any value — the first id not found rejects the whole
/// call with [`IpcErrorKind::InvalidArgument`] and `detail.channel` naming
/// it, never a partial readout (C3 §3.7, spec 746-750: this command's
/// answer is one atomic instant, not per-cell math-tolerant like
/// `eval_workbook`). Requested channels are then materialized
/// (`Channel::materialize`/`Channel::t_us`) in request order and handed to
/// [`idl_rs::cursor::cursor_readout`], which owns the nearest-sample/`null`-
/// outside-span rule (ledger R31) — this function does not re-derive or
/// second-guess it, only folds the `(id, value)` pairs into C3's map.
pub fn cursor_readout_via(
    data_dir: &Path,
    session_id: &str,
    channels: &[String],
    t_us: i64,
) -> Result<CursorReadout, IpcError> {
    let session = load_session(data_dir, session_id)?;

    // Resolved once per requested id — the existence check and the lookup
    // used to materialize below are the same lookup, so there is no second
    // `.find()` later that could fail on an id this loop already accepted
    // (previously two separate lookups, the second pair asserted via
    // `.unwrap()` on the "it was already checked above" invariant; a typed
    // error here needs no such invariant to hold).
    let mut resolved = Vec::with_capacity(channels.len());
    for id in channels {
        let channel = session.channels.iter().find(|c| &c.channel_id == id).ok_or_else(|| {
            IpcError::with_detail(
                IpcErrorKind::InvalidArgument,
                format!("channel '{id}' not found on session '{session_id}'"),
                serde_json::json!({ "channel": id }),
            )
        })?;
        resolved.push(channel);
    }

    // Materialized samples must outlive the `&[f64]` slices borrowed into
    // `triples` below.
    let materialized: Vec<Vec<f64>> = resolved.iter().map(|c| c.materialize()).collect();
    let triples: Vec<(&str, &[i64], &[f64])> = channels
        .iter()
        .zip(resolved.iter())
        .zip(materialized.iter())
        .map(|((id, channel), samples)| (id.as_str(), channel.t_us.as_slice(), samples.as_slice()))
        .collect();

    let values = idl_rs::cursor::cursor_readout(&triples, t_us).into_iter().collect();
    Ok(CursorReadout { t_us, values })
}

/// C3 §3.7 `cursor_readout(session_id, channels, t_us)`. Settle-bound only,
/// never a hot path (C3 §4) — nothing in wave 1 calls it from a hover path.
// TODO(idl0): every call re-reads the session's whole `data.parquet` from
// disk via `load_session` — a session/tier cache is design §4's recorded
// deferral, not this task's (same deferral as `session_source::load_session`
// and Task 11's workbook commands).
#[tauri::command]
pub fn cursor_readout(
    session_id: String,
    channels: Vec<String>,
    t_us: i64,
    data_dir: tauri::State<'_, crate::state::DataDir>,
) -> Result<CursorReadout, IpcError> {
    cursor_readout_via(&data_dir.0, &session_id, &channels, t_us)
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat};
    use idl_rs::store::parquet::write_session_parquet;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-cursor-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds a session with three channels whose spans differ: `long` runs
    /// the full session, `short` stops early (the R31 "stops" case), `late`
    /// starts after the session start (the R31 "hasn't started yet" case).
    fn seed_two_channel_session(root: &Path, session_id: &str) {
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: String::new(),
            channels: vec![
                Channel {
                    channel_id: "long".to_string(),
                    t_us: vec![0, 1_000_000, 2_000_000],
                    t_recorded_us: None,
                    nominal_rate_hz: 1.0,
                    column: RawColumn::F64(vec![1.0, 2.0, 3.0]),
                    source_kind: "test".to_string(),
                    unit: String::new(),
                    gaps: Vec::new(),
                },
                Channel {
                    channel_id: "short".to_string(),
                    t_us: vec![0, 1_000_000],
                    t_recorded_us: None,
                    nominal_rate_hz: 1.0,
                    column: RawColumn::F64(vec![10.0, 20.0]),
                    source_kind: "test".to_string(),
                    unit: String::new(),
                    gaps: Vec::new(),
                },
                Channel {
                    channel_id: "late".to_string(),
                    t_us: vec![1_000_000, 2_000_000],
                    t_recorded_us: None,
                    nominal_rate_hz: 1.0,
                    column: RawColumn::F64(vec![100.0, 200.0]),
                    source_kind: "test".to_string(),
                    unit: String::new(),
                    gaps: Vec::new(),
                },
            ],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn cursor_readout_t_us_on_a_recorded_sample_returns_that_sample_for_every_channel() {
        // Arrange
        let root = temp_root();
        seed_two_channel_session(&root, "s1");

        // Act
        let out = cursor_readout_via(&root, "s1", &["long".to_string(), "short".to_string()], 1_000_000).unwrap();

        // Assert
        assert_eq!(out.t_us, 1_000_000);
        assert_eq!(out.values.get("long"), Some(&Some(2.0)));
        assert_eq!(out.values.get("short"), Some(&Some(20.0)));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_readout_t_us_between_two_samples_resolves_to_the_earlier_sample() {
        // Arrange
        let root = temp_root();
        seed_two_channel_session(&root, "s1");

        // Act
        let out = cursor_readout_via(&root, "s1", &["long".to_string()], 1_500_000).unwrap();

        // Assert
        assert_eq!(out.values.get("long"), Some(&Some(2.0)));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_readout_t_us_past_a_channels_last_sample_that_channel_is_null_others_still_read() {
        // Arrange — R31, the HR-strap-drops-at-minute-40 case: `short` ends
        // at 1_000_000 while `long` keeps going.
        let root = temp_root();
        seed_two_channel_session(&root, "s1");

        // Act
        let out = cursor_readout_via(&root, "s1", &["long".to_string(), "short".to_string()], 2_000_000).unwrap();

        // Assert
        assert_eq!(out.values.get("short"), Some(&None));
        assert_eq!(out.values.get("long"), Some(&Some(3.0)));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_readout_t_us_before_a_channels_first_sample_that_channel_is_null() {
        // Arrange — `late` doesn't start recording until 1_000_000.
        let root = temp_root();
        seed_two_channel_session(&root, "s1");

        // Act
        let out = cursor_readout_via(&root, "s1", &["late".to_string()], 0).unwrap();

        // Assert
        assert_eq!(out.values.get("late"), Some(&None));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_readout_unknown_channel_in_the_list_invalid_argument_naming_it_no_partial_readout() {
        // Arrange
        let root = temp_root();
        seed_two_channel_session(&root, "s1");

        // Act
        let err = cursor_readout_via(&root, "s1", &["long".to_string(), "nope".to_string()], 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "channel": "nope" })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_readout_unknown_session_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = cursor_readout_via(&root, "nope", &["long".to_string()], 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }
}
