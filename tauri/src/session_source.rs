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

/// Builds an [`IpcErrorKind::InvalidArgument`] for a `laps[]` entry whose
/// `start_time_secs >= end_time_secs` (ruling R130). `LapJson` deserialises
/// plain `f64`s with no ordering check, and `start_time_secs ==
/// end_time_secs == 0.0` is indistinguishable downstream from
/// `window_index_range`'s "no window selected" sentinel — it would
/// silently resolve to the whole channel rather than the invalid lap it
/// is. C1 treats `session.json` as data to validate, not trust: a
/// corrupted or hand-edited file (or one from an older engine) can produce
/// this even though the shipped lap detectors cannot. `detail: { "lap": n
/// }`, matching [`unknown_lap`]'s shape for the same `Lap` span kind.
fn invalid_lap_bounds(lap: u32) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::InvalidArgument,
        format!("lap {lap}'s bounds in session.json are not ordered (start_time_secs must be < end_time_secs)"),
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

/// One session's id plus the tagged-union span the UI picked (C1 §6.1, R115)
/// — the wire shape of a time window, `snake_case`, deserialised straight
/// from JS's `Window`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WindowDto {
    /// The session `span` is resolved against. A window is meaningless
    /// without it (C1 §6.1).
    pub session_id: String,
    /// Which portion of `session_id`'s recording this window names.
    pub span: SpanDto,
    /// A chart-token name (`--chart-1` … `--chart-8`), never a hex literal
    /// (C1 §6.1, R117.6). Not validated at this layer — see
    /// [`resolve_window`]'s doc comment for why.
    pub colour: String,
}

/// The tagged union naming which portion of a session's recording a
/// [`WindowDto`] resolves to (C1 §6.1). `kind` tags the wire JSON;
/// `snake_case` on both the tag and `lap_number`/`t0_us`/`t1_us`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpanDto {
    /// The whole session, start to end of its recorded samples.
    Session,
    /// One lap by its 1-based `lap_number`, matching `LapSummary.lap_number`
    /// / `session.json`'s `laps[]`.
    Lap {
        /// 1-based, as stored in `session.json`'s `laps[]`.
        lap_number: u32,
    },
    /// An explicit span. **Not** epoch time — session-relative.
    Range {
        /// Start, microseconds since `session_id`'s first sample (the same
        /// axis as `Channel.t_us`). C1 §6.1 states `t0_us < t1_us` as part
        /// of the type, not an unchecked producer invariant — a value
        /// violating it is rejected by [`resolve_window`] (ruling R120).
        t0_us: i64,
        /// End, microseconds since `session_id`'s first sample.
        t1_us: i64,
    },
}

/// The session's full recorded span, microseconds, **half-open**: the
/// earliest `t_us` across every channel of `session_id`'s `data.parquet`,
/// and one microsecond past the latest (ruling R126). Every resolved
/// window is half-open `[t0, t1)` (C1 §6.1), and `t_us` is integer
/// microseconds, so `last_us + 1` is the exact, minimal half-open bound
/// containing the sample at `last_us` — no recorded sample can fall in
/// `(last_us, last_us + 1)`. Using the last sample's own `t_us` as the end
/// (the pre-R126 behaviour) silently dropped that final sample from every
/// windowed reduction over a whole-session window.
/// `Channel.t_us` is µs since the session's first sample, but not every
/// channel starts (or ends) at the same sample — e.g. an event-driven
/// channel's first `t_us` can be well after `0` — so the session's own
/// start is the min across all of them and the end is one past the max,
/// not any one channel's own span (contrast [`idl_rs::session::handle::
/// SessionMeta::duration_ms`], which is the *longest single channel's*
/// span and does not track the start offset at all). `(0, 0)` for a
/// session with no channels or only empty channels. [`full_session_span_secs`]
/// is this in seconds, the unit
/// [`resolve_window`] returns; this µs form exists for
/// [`no_overlap`]'s `session_span_us` detail, which must name the same
/// integer axis as the caller's `t0_us`/`t1_us` rather than a lossy
/// seconds round-trip.
fn full_session_span_us(data_dir: &Path, session_id: &str) -> Result<(i64, i64), IpcError> {
    let session = load_session(data_dir, session_id)?;
    let mut start_us: Option<i64> = None;
    let mut last_us: Option<i64> = None;
    for c in &session.channels {
        if let (Some(&first), Some(&last)) = (c.t_us.first(), c.t_us.last()) {
            start_us = Some(start_us.map_or(first, |s| s.min(first)));
            last_us = Some(last_us.map_or(last, |e| e.max(last)));
        }
    }
    match (start_us, last_us) {
        (Some(start), Some(last)) => Ok((start, last + 1)),
        _ => Ok((0, 0)),
    }
}

/// [`full_session_span_us`] converted to seconds — the unit every
/// [`resolve_window`] bound is expressed in.
fn full_session_span_secs(data_dir: &Path, session_id: &str) -> Result<(f64, f64), IpcError> {
    let (start_us, end_us) = full_session_span_us(data_dir, session_id)?;
    Ok((start_us as f64 / 1e6, end_us as f64 / 1e6))
}

/// Builds an [`IpcErrorKind::InvalidArgument`] for a `Range` span that does
/// not overlap `session_id`'s recorded span at all (C1 §6.1, ruling R119).
/// `detail: { session_id, t0_us, t1_us, session_span_us }` names every value
/// the caller needs to explain the failure without re-deriving it —
/// `session_span_us` is [`full_session_span_us`]'s half-open `[start_us,
/// end_us)` (R126; `end_us` is one microsecond past the last recorded
/// sample, not that sample's own timestamp), the same integer axis as the
/// caller's own `t0_us`/`t1_us` (not a seconds round-trip, which would lose
/// precision `full_session_span_secs` already discards).
fn no_overlap(session_id: &str, t0_us: i64, t1_us: i64, session_span_us: (i64, i64)) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::InvalidArgument,
        format!(
            "range [{t0_us}, {t1_us}] µs does not overlap session '{session_id}''s recorded span [{}, {}] µs",
            session_span_us.0, session_span_us.1
        ),
        serde_json::json!({
            "session_id": session_id,
            "t0_us": t0_us,
            "t1_us": t1_us,
            "session_span_us": [session_span_us.0, session_span_us.1],
        }),
    )
}

/// Builds an [`IpcErrorKind::InvalidArgument`] for a `Range` span whose
/// `t0_us >= t1_us` (ruling R120, amending R119). C1 §6.1 states
/// `t0_us < t1_us` as part of the `Range` type, so a value violating it is
/// invalid input, not a degenerate-but-legal single-sample window — slicing
/// is inclusive at `t1`, so an in-span `(T, T)` would otherwise select
/// exactly one sample rather than the empty selection the user actually
/// asked for (e.g. dragging both boundary cursors together, decision 52).
/// `detail` shares [`no_overlap`]'s shape:
/// `{ session_id, t0_us, t1_us, session_span_us }`.
fn invalid_range_order(session_id: &str, t0_us: i64, t1_us: i64, session_span_us: (i64, i64)) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::InvalidArgument,
        format!(
            "range [{t0_us}, {t1_us}] µs is not ordered (t0_us must be < t1_us) for session '{session_id}'"
        ),
        serde_json::json!({
            "session_id": session_id,
            "t0_us": t0_us,
            "t1_us": t1_us,
            "session_span_us": [session_span_us.0, session_span_us.1],
        }),
    )
}

/// Resolves a [`WindowDto`]'s span to its recording-time bounds, seconds,
/// against `data_root`'s session store (C1 §6.1, R115; generalises the
/// lap-only [`resolve_lap_window`], which now forwards here).
///
/// **A resolved window is half-open `[t0, t1)`** (ruling R126, C1 §6.1):
/// `t1` is one microsecond past the window's last sample, never that
/// sample's own timestamp. `idl-rs`'s `window_index_range` (`core/src/math/
/// eval.rs`) relies on this to reduce over every sample in the window, and
/// `full_session_span_us`'s `last_us + 1` is what makes `Session` honour it.
///
/// - `Session` resolves to [`full_session_span_secs`] — the full recorded
///   span, `[first_us, last_us + 1)`.
/// - `Lap` resolves one lap number from `session_id`'s `session.json`
///   `laps[]`; a missing or unparsable `session.json` has no `laps[]` to
///   resolve against — treated as zero known laps, so every `lap_number` is
///   [`unknown_lap`] (`invalid_argument`, `detail: { "lap": n }`), shared
///   with [`load_lap_context`] so a bad lap number reports identically
///   wherever it is named.
/// - `Range` converts `t0_us`/`t1_us` to seconds against
///   [`full_session_span_secs`]'s bounds. A range that **overlaps** the
///   session's recorded span (including only partially) clamps each
///   endpoint independently to that span — a boundary cursor dragged past
///   the edge (decision 52), which is legitimate; the in-bounds edge of a
///   partial overlap is kept exactly, only the out-of-bounds edge moves. A
///   range that does **not overlap at all** is
///   [`no_overlap`] (`invalid_argument`, `detail: { session_id, t0_us,
///   t1_us, session_span_us }`), **not** a zero-width `(t, t)` window: this
///   crate's own slicing primitive
///   ([`idl_rs::session::handle::time_window_index_range`]) is inclusive at
///   `t1`, and the clamp target is always a real recorded sample's `t_us`,
///   so `(T, T)` would select exactly one sample rather than none — ruling
///   R119, C1 §6.1. `t0_us >= t1_us` is rejected the same way, via
///   [`invalid_range_order`], even when both endpoints fall inside the
///   session's span — C1 §6.1 states `t0_us < t1_us` as part of the
///   `Range` type, not an unchecked producer invariant, and an in-span
///   `(T, T)` is the same defect one step inward: inclusive-at-`t1`
///   slicing would still select exactly one sample rather than the empty
///   selection the caller asked for (ruling R120, amending R119).
///
/// `Session` and `Range` load the session's `data.parquet` (via
/// [`load_session`]) to know its span, so a missing session is
/// [`IpcErrorKind::NotFound`] for those two kinds. `Lap` never loads the
/// parquet — matching [`resolve_lap_window`]'s original behaviour of not
/// caring whether the session itself exists, only whether the lap number
/// resolves.
pub fn resolve_window(data_root: &Path, window: &WindowDto) -> Result<(f64, f64), IpcError> {
    match &window.span {
        SpanDto::Session => full_session_span_secs(data_root, &window.session_id),
        SpanDto::Lap { lap_number } => {
            let doc = try_read_session_json(data_root, &window.session_id)
                .unwrap_or_else(|_| empty_session_json(&window.session_id));
            let (start, end) = doc
                .laps
                .iter()
                .find(|l| l.lap_number == *lap_number)
                .map(|l| (l.start_time_secs, l.end_time_secs))
                .ok_or_else(|| unknown_lap(*lap_number))?;
            // R130: `session.json` is validated, not trusted — an unordered
            // (or degenerate) pair is rejected here rather than reaching
            // `window_index_range`, where `(0.0, 0.0)` is indistinguishable
            // from "no window selected" and would silently return the
            // whole channel.
            if start < end {
                Ok((start, end))
            } else {
                Err(invalid_lap_bounds(*lap_number))
            }
        }
        SpanDto::Range { t0_us, t1_us } => {
            let session_span_us = full_session_span_us(data_root, &window.session_id)?;
            if *t0_us >= *t1_us {
                return Err(invalid_range_order(&window.session_id, *t0_us, *t1_us, session_span_us));
            }
            let (session_start_us, session_end_us) = session_span_us;
            // Half-open overlap test (ruling R128, amending R126/R119):
            // `session_span_us` is now `[session_start_us, session_end_us)`
            // (R126), so the closed-bound `<`/`>` this used before R126
            // under-rejects at the boundary — a range starting exactly at
            // `session_end_us` has no sample it could possibly cover (no
            // recorded sample falls at or after `session_end_us`) and must
            // be `no_overlap`, not a zero-width clamp that `window_index_
            // range`'s `(0.0, 0.0)`-only sentinel then silently reads back
            // as "no window", i.e. the whole channel.
            if *t1_us <= session_start_us || *t0_us >= session_end_us {
                return Err(no_overlap(&window.session_id, *t0_us, *t1_us, session_span_us));
            }
            let session_start_s = session_start_us as f64 / 1e6;
            let session_end_s = session_end_us as f64 / 1e6;
            let t0_s = (*t0_us as f64 / 1e6).clamp(session_start_s, session_end_s);
            let t1_s = (*t1_us as f64 / 1e6).clamp(session_start_s, session_end_s);
            Ok((t0_s, t1_s))
        }
    }
}

/// Resolves one lap number to its recording-time window, seconds, from
/// `session_id`'s `session.json` `laps[]` (C3 §3.6 `fetch_fft`). A thin
/// wrapper over [`resolve_window`]'s `Lap` arm — kept as its own function
/// until Task 6 retires its last caller; `colour` is irrelevant to
/// resolution so an empty placeholder is passed through.
pub fn resolve_lap_window(data_root: &Path, session_id: &str, lap: u32) -> Result<(f64, f64), IpcError> {
    resolve_window(
        data_root,
        &WindowDto { session_id: session_id.to_string(), span: SpanDto::Lap { lap_number: lap }, colour: String::new() },
    )
}

/// Builds a [`MathLapContext`] scoped to a single resolved [`WindowDto`]
/// (C1 §6.1, R115) — the per-window replacement for [`load_lap_context`]'s
/// per-session lap selection, used by `eval_workbook_v2` and its siblings
/// (Task 4 onward).
///
/// `main_lap_bounds` is always a one-entry vec — [`resolve_window`]'s
/// bounds for `window` — and `main_lap_number` is always `Some(1)`,
/// regardless of `window.span`'s kind: a window is not "the Nth lap of the
/// session", it is *the* selected span, so lap number must never again be
/// read as a position in `main_lap_bounds` (the fix to `core`'s private
/// `main_lap_window`, this same commit, closes the indexing bug this
/// coupling caused). `main_sectors`/`overlay`/`baseline_row` are empty —
/// a window carries no overlay of its own; `handle` is accepted for
/// signature symmetry with [`load_lap_context`] and is not read here.
///
/// Errors are [`resolve_window`]'s: `not_found` for a missing session on
/// the `Session`/`Range` kinds, `invalid_argument` (`detail: { "lap": n }`)
/// for an unknown lap number on the `Lap` kind.
pub fn load_window_context(
    data_dir: &Path,
    window: &WindowDto,
    _handle: &SessionHandle,
) -> Result<MathLapContext, IpcError> {
    let bounds = resolve_window(data_dir, window)?;
    Ok(MathLapContext {
        main_lap_bounds: vec![bounds],
        main_sectors: Vec::new(),
        main_lap_number: Some(1),
        overlay: Vec::new(),
        baseline_row: None,
    })
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

    #[test]
    fn resolve_window_session_span_full_recorded_span_from_t_us() {
        // Arrange — `seed_session`'s "Speed" channel's last sample is at
        // 500_000 µs; the resolved span is half-open, one microsecond past
        // it (ruling R126), so the last sample is not silently dropped by a
        // downstream half-open reduction.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: String::new() };

        // Act
        let span = resolve_window(&root, &window).unwrap();

        // Assert
        assert_eq!(span, (0.0, 0.500_001));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_lap_span_matches_session_json_laps() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Lap { lap_number: 3 }, colour: String::new() };

        // Act
        let span = resolve_window(&root, &window).unwrap();

        // Assert
        assert_eq!(span, (2.0, 3.0));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_lap_span_unknown_lap_number_invalid_argument_with_detail() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Lap { lap_number: 99 }, colour: String::new() };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 99 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_lap_span_degenerate_zero_zero_bounds_invalid_argument_with_detail() {
        // Arrange — a `laps[]` entry with `start_time_secs == end_time_secs
        // == 0.0` (a corrupted or hand-edited `session.json`; the shipped
        // detectors cannot produce this). Unvalidated, `(0.0, 0.0)` is
        // exactly `main_lap_window`'s "no window selected" sentinel and
        // would silently resolve to the whole channel (ruling R130).
        let root = temp_root();
        seed_session(&root, "s1");
        let mut doc = empty_session_json("s1");
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            raw_elapsed_ms: 0,
            lap_time_ms: 0,
            start_time_secs: 0.0,
            end_time_secs: 0.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        write_session_json(&root, "s1", &doc, None).unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Lap { lap_number: 1 }, colour: String::new() };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert — a typed error, not a `(0.0, 0.0)` window read back as
        // the whole channel.
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 1 })));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_inside_session_converts_us_to_seconds() {
        // Arrange — within the 0..500_000 µs session span.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 100_000, t1_us: 400_000 },
            colour: String::new(),
        };

        // Act
        let span = resolve_window(&root, &window).unwrap();

        // Assert
        assert_eq!(span, (0.1, 0.4));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_partially_overlapping_before_clamps_only_the_out_of_bounds_edge() {
        // Arrange — session spans 0..500_000 µs; this range starts before
        // it and ends inside it, so only `t0_us` is out of bounds.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: -100_000, t1_us: 200_000 },
            colour: String::new(),
        };

        // Act
        let span = resolve_window(&root, &window).unwrap();

        // Assert — the in-bounds edge (t1_us = 200_000 µs = 0.2 s) is kept
        // exactly; only the out-of-bounds start clamps to the session start.
        assert_eq!(span, (0.0, 0.2));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_partially_overlapping_after_clamps_only_the_out_of_bounds_edge() {
        // Arrange — mirror of the "before" case: starts inside the session,
        // ends after it.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 300_000, t1_us: 900_000 },
            colour: String::new(),
        };

        // Act
        let span = resolve_window(&root, &window).unwrap();

        // Assert — the in-bounds edge (t0_us = 300_000 µs = 0.3 s) is kept
        // exactly; only the out-of-bounds end clamps to the session's
        // half-open end, one microsecond past its last sample (R126).
        assert_eq!(span, (0.3, 0.500_001));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_wholly_after_session_invalid_argument_with_detail() {
        // Arrange — session's half-open end is 500_001 µs (one past its
        // last sample, R126); this range starts after it, so there is no
        // overlap at all (ruling R119).
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 600_000, t1_us: 700_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert — a typed error, not a zero-width `(0.5, 0.5)` window.
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": 600_000,
                "t1_us": 700_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_starting_exactly_at_the_half_open_session_end_no_overlap() {
        // Arrange — R128 (amending R126): `session_span_us`'s end is
        // half-open (`last_us + 1` = 500_001 µs here), so a range starting
        // there covers no sample the session actually has — it must be
        // rejected as `no_overlap`, never accepted as an in-bounds range
        // that clamps to a zero-width `(X, X)` window (which
        // `window_index_range`'s sentinel would then silently read back as
        // the whole channel).
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 500_001, t1_us: 600_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert!(err.message.contains("does not overlap"), "message: {}", err.message);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": 500_001,
                "t1_us": 600_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_wholly_before_session_invalid_argument_with_detail() {
        // Arrange — session starts at 0 µs; this range ends before it, so
        // there is no overlap at all (ruling R119).
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: -500_000, t1_us: -100_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": -500_000,
                "t1_us": -100_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_equal_endpoints_inside_session_invalid_argument_with_detail() {
        // Arrange — t0_us == t1_us falls inside the session's recorded span
        // (0..500_000 µs), but slicing is inclusive at t1, so this would
        // otherwise select exactly one sample — rejected as an ordering
        // violation, not resolved as a legitimate single-sample overlap
        // (ruling R120, amending R119).
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 200_000, t1_us: 200_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": 200_000,
                "t1_us": 200_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_mis_ordered_and_wholly_outside_session_invalid_range_order_not_no_overlap() {
        // Arrange — both defects at once: `t0_us > t1_us` (R120) *and* the
        // range falls entirely outside the session's recorded span
        // (0..500_000 µs) as R119's `no_overlap` alone would also reject.
        // The two R120 tests above both use in-span endpoints, so they only
        // pin ordering precedence for an in-span pair — this pins that the
        // ordering check fires first regardless, per `resolve_window`'s own
        // check order.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 600_000, t1_us: 100_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert — `invalid_range_order` fired, not `no_overlap`: same
        // `detail` shape either way, so the distinguishing evidence is
        // `message` (`no_overlap`'s reads "does not overlap"; this must not).
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert!(err.message.contains("not ordered"), "message: {}", err.message);
        assert!(!err.message.contains("does not overlap"), "message: {}", err.message);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": 600_000,
                "t1_us": 100_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_window_range_span_t0_after_t1_invalid_argument_with_detail() {
        // Arrange — t0_us > t1_us, the general case R120 closes.
        let root = temp_root();
        seed_session(&root, "s1");
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 300_000, t1_us: 100_000 },
            colour: String::new(),
        };

        // Act
        let err = resolve_window(&root, &window).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(
            err.detail,
            Some(serde_json::json!({
                "session_id": "s1",
                "t0_us": 300_000,
                "t1_us": 100_000,
                "session_span_us": [0, 500_001],
            }))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_window_context_lap_span_one_entry_bounds_main_lap_number_always_one() {
        // Arrange — lap 3 of a 4-lap session; the resolved lap number (3)
        // must never leak into `main_lap_number`, which stays `Some(1)`
        // regardless of which lap or span kind was selected.
        let root = temp_root();
        seed_session(&root, "s1");
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let handle = load_session_handle(&root, "s1").unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Lap { lap_number: 3 }, colour: String::new() };

        // Act
        let ctx = load_window_context(&root, &window, &handle).unwrap();

        // Assert
        assert_eq!(ctx.main_lap_bounds, vec![(2.0, 3.0)]);
        assert_eq!(ctx.main_lap_number, Some(1));
        assert!(ctx.main_sectors.is_empty());
        assert!(ctx.overlay.is_empty());
        assert_eq!(ctx.baseline_row, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_window_context_session_span_bounds_the_whole_recorded_span() {
        // Arrange — `seed_session`'s "Speed" channel's last sample is at
        // 500_000 µs; the resolved bound is half-open, one microsecond past
        // it (ruling R126).
        let root = temp_root();
        seed_session(&root, "s1");
        let handle = load_session_handle(&root, "s1").unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: String::new() };

        // Act
        let ctx = load_window_context(&root, &window, &handle).unwrap();

        // Assert
        assert_eq!(ctx.main_lap_bounds, vec![(0.0, 0.500_001)]);
        assert_eq!(ctx.main_lap_number, Some(1));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_window_context_unknown_lap_number_invalid_argument_with_detail() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");
        write_session_json(&root, "s1", &four_lap_doc("s1"), None).unwrap();
        let handle = load_session_handle(&root, "s1").unwrap();
        let window = WindowDto { session_id: "s1".to_string(), span: SpanDto::Lap { lap_number: 99 }, colour: String::new() };

        // Act
        let err = load_window_context(&root, &window, &handle).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail, Some(serde_json::json!({ "lap": 99 })));

        let _ = std::fs::remove_dir_all(&root);
    }
}
