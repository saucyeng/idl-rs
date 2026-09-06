//! Session loading shared by every workbook command that needs a channel
//! lookup or a lap context (`eval_workbook`, and the tasks after it — C3
//! §3.4). A session is bound by id, not by front matter (C2 has none), so
//! this module is the one place that turns a `session_id` string into a
//! [`idl_rs::session::SessionHandle`] or a [`MathLapContext`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use idl_rs::math::{ChannelLookup, MathLapContext, MathOverlay};
use idl_rs::session::handle::SessionHandle;
use idl_rs::session::synthesis::synthesize_base_channels;
use idl_rs::session::Session;
use idl_rs::store::parquet::read_session_parquet;
use idl_rs::store::session_json::{empty_session_json, read_session_json, LapJson, SessionJson};

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

/// Reads `session_id`'s `session.json`, `Err(())` if it is absent or fails
/// to parse. The one place [`load_lap_context`] and [`resolve_lap_window`]
/// both call to reach the file, so a path change or a parse-library swap
/// only has one call site to update (each caller still decides its own
/// meaning for "unreadable" — they are not required to agree).
fn try_read_session_json(data_dir: &Path, session_id: &str) -> Result<SessionJson, ()> {
    let path = session_dir(data_dir, session_id).join("session.json");
    read_session_json(&path).map_err(|_| ())
}

/// Resolves one lap number to its recording-time window, seconds, from
/// `session_id`'s `session.json` `laps[]` (C3 §3.6 `fetch_fft`). Shares
/// [`unknown_lap`]'s error shape with [`load_lap_context`] so a bad lap
/// number reports identically wherever it is named (C3 §3.4's
/// `invalid_argument` + `detail: { "lap": n }`). A missing or unparsable
/// `session.json` has no `laps[]` to resolve against — treated as zero
/// known laps, so every `lap` number is [`unknown_lap`], not a distinct
/// error (this function has no "no lap context" answer to give back, unlike
/// [`load_lap_context`]'s `Ok(MathLapContext::empty())`: a window is either
/// resolved or it is an error).
pub fn resolve_lap_window(data_root: &Path, session_id: &str, lap: u32) -> Result<(f64, f64), IpcError> {
    let doc = try_read_session_json(data_root, session_id).unwrap_or_else(|_| empty_session_json(session_id));
    doc.laps
        .iter()
        .find(|l| l.lap_number == lap)
        .map(|l| (l.start_time_secs, l.end_time_secs))
        .ok_or_else(|| unknown_lap(lap))
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
/// None` bounds) and `overlay` gets one [`MathOverlay`] per entry of
/// `lc.overlay_laps`, in order (empty `Vec` when `lc.overlay_laps` is empty)
/// — each windowed to that lap, all built from `handle`, the **same
/// session's own** `ChannelLookup` (R64.1: wave 2 has no cross-session
/// overlay; a future amendment carries a `{ session_id, lap }[]` shape for
/// that). Every entry shares one `Arc<dyn ChannelLookup>` over `handle`
/// (built once, cloned per entry — a cheap refcount bump, not a fresh
/// `SessionHandle` clone per overlay lap; R73's note) since they all read
/// the same session. The evaluator's `variance_time`/`variance_dist` fold
/// across every entry of `overlay` (`core::math::eval::mean_across_overlays`);
/// see their doc comments for what "several overlay laps" means to each.
pub fn load_lap_context(
    data_dir: &Path,
    session_id: &str,
    handle: &SessionHandle,
    selection: Option<&LapContext>,
) -> Result<MathLapContext, IpcError> {
    let Ok(doc) = try_read_session_json(data_dir, session_id) else {
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

    // One shared lookup Arc for every overlay entry — `Arc::clone` bumps a
    // refcount, it does not clone `handle` itself (R73's note: the old code
    // built a fresh `Arc::new(handle.clone())` per overlay).
    let overlay: Vec<MathOverlay> = if overlay_lap_jsons.is_empty() {
        Vec::new()
    } else {
        let shared: Arc<dyn ChannelLookup + Send + Sync> = Arc::new(handle.clone());
        overlay_lap_jsons
            .iter()
            .map(|l| MathOverlay {
                lookup: Arc::clone(&shared),
                lap_start_ms: l.start_timestamp_ms as f64,
                lap_end_ms: l.end_timestamp_ms as f64,
                lap_start_uniform_sec: l.start_time_secs,
            })
            .collect()
    };

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
        assert!(ctx.overlay.is_empty());

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
        // Arrange — this session's session.json has no laps[] at all.
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
        assert!(no_selection.overlay.is_empty());
        assert!(empty_selection.overlay.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Four laps at 1-second-per-lap boundaries, for the multi-overlay tests.
    fn four_lap_doc(session_id: &str) -> SessionJson {
        let mut doc = empty_session_json(session_id);
        doc.laps = (1..=4u32)
            .map(|n| LapJson {
                lap_number: n,
                start_timestamp_ms: (n as i64 - 1) * 1_000,
                end_timestamp_ms: n as i64 * 1_000,
                raw_elapsed_ms: 1_000,
                lap_time_ms: 1_000,
                start_time_secs: (n - 1) as f64,
                end_time_secs: n as f64,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            })
            .collect();
        doc
    }

    #[test]
    fn load_lap_context_overlay_laps_two_entries_two_overlays_in_order() {
        // Arrange — a 4-lap session, overlay_laps = [2, 3].
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let selection = LapContext { main_lap: None, overlay_laps: vec![2, 3] };

        // Act
        let ctx = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap();

        // Assert — two overlays, windows matching laps 2 and 3 in that order.
        assert_eq!(ctx.overlay.len(), 2);
        assert_eq!(ctx.overlay[0].lap_start_ms, 1_000.0);
        assert_eq!(ctx.overlay[0].lap_end_ms, 2_000.0);
        assert_eq!(ctx.overlay[1].lap_start_ms, 2_000.0);
        assert_eq!(ctx.overlay[1].lap_end_ms, 3_000.0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_overlay_laps_unknown_entry_after_valid_main_lap() {
        // Arrange — a valid main_lap, overlay_laps names a good lap then an
        // unresolvable one; main_lap's own validation must still run first
        // (it does not error here, proving it ran and passed before the
        // overlay scan reached the bad entry).
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let selection = LapContext { main_lap: Some(1), overlay_laps: vec![2, 99] };

        // Act
        let err = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 99 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_overlay_laps_empty_vec_no_error_no_overlay() {
        // Arrange — a 4-lap session, overlay_laps explicitly empty.
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let selection = LapContext { main_lap: Some(1), overlay_laps: Vec::new() };

        // Act
        let ctx = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap();

        // Assert
        assert!(ctx.overlay.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_overlay_laps_share_one_arc_lookup() {
        // Arrange — two overlay laps.
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let selection = LapContext { main_lap: None, overlay_laps: vec![2, 3] };

        // Act
        let ctx = load_lap_context(&root, "s1", &handle, Some(&selection)).unwrap();

        // Assert — both overlays' `lookup` point at the same allocation
        // (R73's note: one Arc built once and cloned, not one `SessionHandle`
        // clone per overlay).
        assert_eq!(ctx.overlay.len(), 2);
        assert!(std::sync::Arc::ptr_eq(&ctx.overlay[0].lookup, &ctx.overlay[1].lookup));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_lap_window_known_lap_number_its_two_seconds_values() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        let mut doc = empty_session_json("s1");
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

        // Act
        let window = resolve_lap_window(&root, "s1", 2).unwrap();

        // Assert
        assert_eq!(window, (1.0, 2.5));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_lap_window_unknown_lap_number_unknown_lap_with_detail() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        let doc = empty_session_json("s1");
        write_session_json(&root, "s1", &doc, None).unwrap();

        // Act
        let err = resolve_lap_window(&root, "s1", 99).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 99 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_lap_window_missing_session_json_unknown_lap_same_as_load_lap_context_mapping() {
        // Arrange — no `write_session_json` call at all: the file is absent,
        // same state `load_lap_context_no_session_json_empty_context`
        // exercises for its own `Ok(empty)` answer. `resolve_lap_window` has
        // no "no context" answer to give back, so an absent file behaves as
        // zero known laps: any `lap` number is `unknown_lap`, not a distinct
        // `not_found`/`io` error.
        let root = temp_root();
        seed_session(&root, "s1");

        // Act
        let err = resolve_lap_window(&root, "s1", 1).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 1 })));

        let _ = std::fs::remove_dir_all(&root);
    }
}
