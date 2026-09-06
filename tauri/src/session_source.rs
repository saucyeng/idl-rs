//! Session loading shared by every workbook command that needs a channel
//! lookup or a lap context (`eval_workbook`, and the tasks after it — C3
//! §3.4). A session is bound by id, not by front matter (C2 has none), so
//! this module is the one place that turns a `session_id` string into a
//! [`idl_rs::session::SessionHandle`] or a [`MathLapContext`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use idl_rs::math::{MathLapContext, MathOverlay};
use idl_rs::session::handle::SessionHandle;
use idl_rs::session::synthesis::synthesize_base_channels;
use idl_rs::session::Session;
use idl_rs::store::parquet::read_session_parquet;
use idl_rs::store::session_json::{read_session_json, LapJson};

use crate::commands::workbook::LapContext;
use crate::error::{IpcError, IpcErrorKind};

/// `<data>/sessions/<session_id>` (C4 §2).
pub fn session_dir(data_dir: &Path, session_id: &str) -> PathBuf {
    data_dir.join("sessions").join(session_id)
}

// TODO(idl0): every call re-reads the whole `data.parquet` from disk — a
// session cache is design §4's recorded deferral, not this task's.
/// Reads `session_id`'s `data.parquet` back into a [`Session`], running
/// [`synthesize_base_channels`] before returning (`read_session_parquet`
/// does not reconstruct `Time`/`Distance` itself — its own doc comment says
/// so). A missing session directory or `data.parquet` is
/// [`IpcErrorKind::NotFound`] (the id is named in the message); a parquet
/// read failure is [`IpcErrorKind::Io`].
pub fn load_session(data_dir: &Path, session_id: &str) -> Result<Session, IpcError> {
    let parquet_path = session_dir(data_dir, session_id).join("data.parquet");
    if !parquet_path.exists() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session '{session_id}' not found")));
    }
    let mut session = read_session_parquet(&parquet_path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {}", parquet_path.display(), e.message)))?;
    synthesize_base_channels(&mut session);
    Ok(session)
}

/// [`load_session`] wrapped in a [`SessionHandle`] via [`SessionHandle::
/// from_session`] (ledger R40 — carries `source_format`/`blob_sha256`/
/// per-channel `unit` through, unlike `from_channels`).
pub fn load_session_handle(data_dir: &Path, session_id: &str) -> Result<SessionHandle, IpcError> {
    let session = load_session(data_dir, session_id)?;
    Ok(SessionHandle::from_session(session))
}

/// Builds an [`IpcErrorKind::InvalidArgument`] naming the unresolvable lap
/// number, `detail: { "lap": lap }` (C3 §3.4).
fn unknown_lap(lap: u32) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::InvalidArgument,
        format!("lap {lap} not found in session.json's laps[]"),
        serde_json::json!({ "lap": lap }),
    )
}

/// Builds a [`MathLapContext`] from `session_id`'s `session.json` `laps[]`,
/// optionally validated and overridden by a caller-supplied `selection`
/// (C3 §3.4 `lap_context`, ruling R52 Q5; same-session `overlay_laps` per
/// lead ruling R64.1, 2026-09-05).
///
/// `selection = None` reproduces the function's original, pre-C3-`lap_context`
/// behaviour byte for byte: `main_lap_bounds`/`main_lap_number` come from
/// `session.json`'s own stored `laps[]`/`main_lap_number` verbatim, `overlay`
/// stays `None`. `session.json` absent or unreadable is not an error here in
/// either branch — it returns `Ok(`[`MathLapContext::empty`]`)` (C2 §3.5.B's
/// `NoLapContext` is the evaluator's own answer for "no lap context", not
/// this function's job to reject on).
///
/// `selection = Some(lc)` is the per-call UI designation (ledger R41 — not a
/// property of the file): every lap number named in `lc.main_lap`/
/// `lc.overlay_laps` must exist in `session.json`'s actual `laps[]`, checked
/// `main_lap` first, then `overlay_laps` in order — the first unresolvable
/// number returns `Err(`[`unknown_lap`]`)`. When every named lap resolves,
/// `main_lap_bounds`/`main_lap_number` are built from the resolved main lap
/// (falling back to every lap in `laps[]`/`session.json`'s own
/// `main_lap_number` when `lc.main_lap` is `None`, matching the `selection =
/// None` bounds) and, when `lc.overlay_laps` is non-empty, `overlay` is
/// built from `handle` — the **same session's own** `ChannelLookup`
/// (R64.1: wave 2 has no cross-session overlay; a future amendment carries
/// a `{ session_id, lap }[]` shape for that) — windowed to the *first*
/// entry of `lc.overlay_laps` (`MathOverlay` models one lap window; see its
/// doc comment). `laps[]` is always empty today (lap indexing has not
/// landed), so every non-empty `selection` rejects via [`unknown_lap`]
/// before this branch is ever reached in practice.
pub fn load_lap_context(
    data_dir: &Path,
    session_id: &str,
    handle: &SessionHandle,
    selection: Option<&LapContext>,
) -> Result<MathLapContext, IpcError> {
    let path = session_dir(data_dir, session_id).join("session.json");
    let Ok(doc) = read_session_json(&path) else {
        return Ok(MathLapContext::empty());
    };

    let Some(lc) = selection else {
        return Ok(MathLapContext {
            main_lap_bounds: doc.laps.iter().map(|l| (l.start_time_secs, l.end_time_secs)).collect(),
            main_lap_number: doc.main_lap_number,
            ..MathLapContext::empty()
        });
    };

    let find_lap = |n: u32| -> Option<&LapJson> { doc.laps.iter().find(|l| l.lap_number == n) };

    let main_lap_json = match lc.main_lap {
        Some(n) => Some(find_lap(n).ok_or_else(|| unknown_lap(n))?),
        None => None,
    };

    let mut overlay_lap_jsons = Vec::with_capacity(lc.overlay_laps.len());
    for &n in &lc.overlay_laps {
        overlay_lap_jsons.push(find_lap(n).ok_or_else(|| unknown_lap(n))?);
    }

    let (main_lap_bounds, main_lap_number) = match main_lap_json {
        Some(l) => (vec![(l.start_time_secs, l.end_time_secs)], Some(l.lap_number)),
        None => (doc.laps.iter().map(|l| (l.start_time_secs, l.end_time_secs)).collect(), doc.main_lap_number),
    };

    let overlay = overlay_lap_jsons.first().map(|l| MathOverlay {
        lookup: Arc::new(handle.clone()),
        lap_start_ms: l.start_timestamp_ms as f64,
        lap_end_ms: l.end_timestamp_ms as f64,
        lap_start_uniform_sec: l.start_time_secs,
    });

    Ok(MathLapContext { main_lap_bounds, main_sectors: Vec::new(), main_lap_number, overlay, baseline_row: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, SourceFormat};
    use idl_rs::store::parquet::write_session_parquet;
    use idl_rs::store::session_json::{empty_session_json, write_session_json, LapJson};
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-session-source-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seed_session(root: &Path, session_id: &str) {
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Speed".to_string(),
                t_us: vec![0, 500_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0]),
                source_kind: "gps".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn load_session_unknown_id_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = load_session(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_session_seeded_session_channels_and_units_survive_the_round_trip() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");

        // Act
        let session = load_session(&root, "s1").unwrap();

        // Assert
        assert_eq!(session.source_format, SourceFormat::Fit);
        assert_eq!(session.blob_sha256, "a".repeat(64));
        let speed = session.channels.iter().find(|c| c.channel_id == "Speed").unwrap();
        assert_eq!(speed.unit, "m/s");
        assert_eq!(speed.column, RawColumn::F64(vec![1.0, 2.0]));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_no_selection_session_json_with_two_laps_two_bounds() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        let mut doc = empty_session_json("s1");
        doc.main_lap_number = Some(2);
        doc.laps = vec![
            LapJson {
                lap_number: 1,
                start_timestamp_ms: 0,
                end_timestamp_ms: 1_000,
                raw_elapsed_ms: 1_000,
                lap_time_ms: 1_000,
                start_time_secs: 0.0,
                end_time_secs: 1.0,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
            LapJson {
                lap_number: 2,
                start_timestamp_ms: 1_000,
                end_timestamp_ms: 2_500,
                raw_elapsed_ms: 1_500,
                lap_time_ms: 1_500,
                start_time_secs: 1.0,
                end_time_secs: 2.5,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
        ];
        write_session_json(&root, "s1", &doc, None).unwrap();

        // Act — `selection = None` is byte-identical to the function's
        // original, pre-`lap_context` behaviour: session.json's own laps[]
        // win, no validation.
        let ctx = load_lap_context(&root, "s1", &handle, None).unwrap();

        // Assert
        assert_eq!(ctx.main_lap_bounds, vec![(0.0, 1.0), (1.0, 2.5)]);
        assert_eq!(ctx.main_lap_number, Some(2));
        assert!(ctx.overlay.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_no_session_json_empty_context() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();

        // Act
        let ctx = load_lap_context(&root, "nope", &handle, None).unwrap();

        // Assert
        assert!(ctx.main_lap_bounds.is_empty());
        assert_eq!(ctx.main_lap_number, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_selection_naming_a_main_lap_absent_from_laps_invalid_argument_with_detail_lap() {
        // Arrange — session.json has no laps[] at all (today's only
        // reachable state, C3 §3.4's "Note").
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        let doc = empty_session_json("s1");
        write_session_json(&root, "s1", &doc, None).unwrap();
        let selection = LapContext { main_lap: Some(1), overlay_laps: Vec::new() };

        // Act
        let err = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 1 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_selection_naming_overlay_laps_absent_from_laps_invalid_argument_names_the_first_offender() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        let doc = empty_session_json("s1");
        write_session_json(&root, "s1", &doc, None).unwrap();
        let selection = LapContext { main_lap: None, overlay_laps: vec![2, 3] };

        // Act
        let err = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap_err();

        // Assert — lap 2 is scanned before lap 3, so it is "the first
        // offending value" named in `detail.lap`.
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 2 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_explicit_empty_selection_matches_no_selection_output() {
        // Arrange — an explicit `LapContext { main_lap: None, overlay_laps:
        // vec![] }` is a distinct wire value from the argument's own
        // absence, but both must resolve to the same output for the same
        // session.json (C3 §3.4).
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        let doc = empty_session_json("s1");
        write_session_json(&root, "s1", &doc, None).unwrap();
        let selection = LapContext { main_lap: None, overlay_laps: Vec::new() };

        // Act
        let no_selection = load_lap_context(&root, "s1", &handle, None).unwrap();
        let empty_selection = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap();

        // Assert
        assert_eq!(no_selection.main_lap_bounds, empty_selection.main_lap_bounds);
        assert_eq!(no_selection.main_lap_number, empty_selection.main_lap_number);
        assert!(no_selection.overlay.is_none());
        assert!(empty_selection.overlay.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }
}
