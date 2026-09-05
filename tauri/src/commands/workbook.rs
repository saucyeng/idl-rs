//! Workbook commands (C3 §3.4): `open_workbook`, `eval_workbook`,
//! `save_workbook`, `watch_workbook` — thin wrappers over L3's v3 parser/
//! evaluator (`idl_rs::workbook::v3`) and Task 3's file watcher
//! (`crate::watcher`).
//!
//! Same idiom as `commands/device.rs`/`commands/catalog.rs`: each
//! `#[tauri::command]` is a one-line wrapper over a `_via`-suffixed plain
//! function taking `data_dir: &Path` — this module's own tests exercise the
//! `_via` functions (`tauri::State`/`tauri::ipc::Channel` cannot be
//! constructed outside a running app).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use idl_rs::math::MathLapContext;
use idl_rs::session::handle::{SessionHandle, SessionMetaInput};
use idl_rs::store::atomic::{sha256_hex, write_atomic, AtomicWriteErrorKind};
use idl_rs::table::eval::evaluate_table;
use idl_rs::workbook::v3::front_matter::parse_front_matter;
use idl_rs::workbook::v3::{parse_workbook, CellDoc, CellError, CellKindToken, WorkbookError};

use crate::error::{IpcError, IpcErrorKind};
use crate::session_source::{load_lap_context, load_session_handle};
use crate::state::{DataDir, Hashes, Watchers};
use crate::watcher::{ExpectedHashSet, WorkbookWatcher};

/// `open_workbook`'s return (C3 §3.4).
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkbookHandle {
    pub id: String,
    pub name: String,
    pub path: String,
    /// u32
    pub cell_count: u32,
}

/// The light wire marker for a `HostChannel` (C3 §3.4, ledger R22/R45): the
/// full `{length, t, v}` shape (`idl_rs::workbook::v3::HostChannel`) stays
/// in-process — this is all that ever crosses IPC as JSON.
// TODO(idl0): design and ship the `HostChannel` binary byte path C3 §3.4
// assigns to L5/L6, deferred to wave 2 (ledger R45). Until that command
// exists, a `math`-cell definition's full sample array never crosses IPC —
// only this `{length, has_t}` marker does.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HostChannelRef {
    /// Number of values (`HostChannel::length`, always `v.len()`).
    pub length: u32,
    /// Whether the channel carries a recorded time axis (`!HostChannel.t.is_empty()`).
    pub has_t: bool,
}

impl From<&idl_rs::workbook::v3::HostChannel> for HostChannelRef {
    fn from(h: &idl_rs::workbook::v3::HostChannel) -> Self {
        Self { length: h.length as u32, has_t: !h.t.is_empty() }
    }
}

/// One `math`-cell definition's evaluated result (C3 §3.4).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CellDefResult {
    pub name: String,
    pub label: Option<String>,
    pub value: Option<HostChannelRef>,
    /// `math_*` kind only — a structural problem on this definition keeps it
    /// out of `defs` entirely (routed to the cell's own `errors` instead).
    pub error: Option<IpcError>,
}

/// One cell's evaluation result (C3 §3.4's `CellOutput`). A per-cell failure
/// never rejects `eval_workbook` — it appears here, in `errors` or in a
/// specific `defs[i].error`; other cells still evaluate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CellOutput {
    /// C2 fence-string id.
    pub cell_id: String,
    /// "math" | "table" | "js" — never "prose" (ledger R21; prose has no
    /// fence id and never gets its own `CellOutput`).
    pub kind: String,
    /// `null` for `math`/`js` (their results live in `defs`); for a `table`
    /// cell, `{ model, results }` on success (ledger R21) — `null` when no
    /// session is bound (nothing to evaluate against) or the table JSON
    /// itself failed to parse (that structural error is in `errors`).
    pub value: Option<serde_json::Value>,
    /// One entry per definition, `def_line` source order; empty for
    /// `table`/`js` cells.
    pub defs: Vec<CellDefResult>,
    /// Plural, always present, `[]` on success (ledger R22) — every
    /// structural (`workbook_*`) or evaluation (`math_*`) problem this cell
    /// carries.
    pub errors: Vec<IpcError>,
}

/// `save_workbook`'s return (C3 §3.4).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SaveResult {
    /// sha256 of the written bytes, hex.
    pub hash: String,
    /// i64, wall-clock at the moment the write completed.
    pub saved_utc_ms: i64,
}

/// One file-watcher event for a subscribed workbook (C3 §3.4, design §7).
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkbookEvent {
    /// "changed" | "conflict" — wave 1 only ever sends "changed" (no
    /// conflict detection lives in the watcher itself; a save conflict is
    /// `save_workbook`'s own `conflict` `IpcError`, not a watcher event).
    pub kind: String,
    /// Cells affected by this event (ledger, this task — L3 ships no diff
    /// function; see [`diff_cell_ids`]). May be empty when only prose
    /// changed.
    pub cell_ids: Vec<String>,
}

fn cell_kind_str(k: CellKindToken) -> &'static str {
    match k {
        CellKindToken::Math => "math",
        CellKindToken::Table => "table",
        CellKindToken::Js => "js",
    }
}

/// Maps `parse_workbook`'s `Err` arm (the two document-fatal kinds, fact 3
/// of this task's brief) to the matching `workbook_*` `IpcErrorKind`.
fn fatal_parse_error(errs: Vec<WorkbookError>) -> IpcError {
    match errs.first() {
        Some(e) => IpcError::from(e),
        None => IpcError::new(IpcErrorKind::Internal, "workbook front matter failed to parse"),
    }
}

/// Resolves C3 §3.4's `id_or_path` argument to a filesystem path, used by
/// `open_workbook`/`eval_workbook`/`watch_workbook`: (1) if the string is an
/// existing file path, use it; (2) otherwise scan `<data>/workbooks/
/// *.idl1wb`, parsing each file's front matter and matching `id`; (3)
/// otherwise `not_found`. Never consults the catalog (C4 §5 — it is an
/// index, never read for truth).
// TODO(idl0): the scan is O(workbooks) per call — acceptable at wave-1
// scale, not addressed here.
fn resolve_workbook_path(data_dir: &Path, id_or_path: &str) -> Result<PathBuf, IpcError> {
    let as_path = Path::new(id_or_path);
    if as_path.is_file() {
        return Ok(as_path.to_path_buf());
    }
    let workbooks_dir = data_dir.join("workbooks");
    if let Ok(entries) = std::fs::read_dir(&workbooks_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("idl1wb") {
                continue;
            }
            let Ok(markdown) = std::fs::read_to_string(&path) else { continue };
            if let Ok((front_matter, _)) = parse_front_matter(&markdown) {
                if front_matter.id == id_or_path {
                    return Ok(path);
                }
            }
        }
    }
    Err(IpcError::new(IpcErrorKind::NotFound, format!("workbook '{id_or_path}' not found")))
}

/// Filesystem-sanitises a display name into a `file_name` stem (C4 §2, SPEC
/// §15.1) — every character that is not alphanumeric, space, `-`, `_`, or
/// `.` becomes `_`; leading/trailing dots and whitespace (both illegal at a
/// Windows path-segment edge) are trimmed. Falls back to `"workbook"` when
/// that leaves nothing (e.g. a name made entirely of punctuation/emoji) — a
/// workbook always needs *some* file name. No core sanitiser exists yet to
/// reuse (only `session::filename::unique_file_base`'s collision-suffix
/// half is landed); this is scoped to workbooks, not a general-purpose
/// implementation.
fn sanitize_file_name_stem(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches(|c: char| c == '.' || c.is_whitespace());
    if trimmed.is_empty() || is_windows_reserved_name(trimmed) {
        "workbook".to_string()
    } else {
        trimmed.to_string()
    }
}

/// True iff `stem` case-insensitively matches one of Windows's reserved
/// device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`9`, `LPT1`-`9`) —
/// ledger R49's Minor: none of these fail to create on every filesystem
/// (verified empirically, not live on this machine), but the failure mode
/// where it *is* live is an opaque `io` error out of `write_atomic` rather
/// than a clear message, and a sanitised stem becomes a real path. Exact
/// match only (case-insensitive) — `stem` is already the whole file-name
/// stem by the time this runs, not a substring search, so `"CONTAINER"` is
/// not reserved.
fn is_windows_reserved_name(stem: &str) -> bool {
    const RESERVED: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1",
        "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    RESERVED.iter().any(|r| stem.eq_ignore_ascii_case(r))
}

/// Transport-agnostic core of `open_workbook`.
fn open_workbook_via(data_dir: &Path, id_or_path: &str) -> Result<WorkbookHandle, IpcError> {
    let path = resolve_workbook_path(data_dir, id_or_path)?;
    let markdown = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
    let (doc, _structural) = parse_workbook(&markdown).map_err(fatal_parse_error)?;
    Ok(WorkbookHandle { id: doc.id, name: doc.name, path: path.display().to_string(), cell_count: doc.cells.len() as u32 })
}

/// C3 §3.4 `WorkbookSource` — `read_workbook`'s return. Added post-sign
/// (2026-09-05, ruling R59) to close the gap that made `save_workbook`
/// unusable as specified: nothing previously let the editor read the file
/// it is about to save `based_on_hash` against.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkbookSource {
    /// The file's UTF-8 text, verbatim — never parsed by this command.
    pub markdown: String,
    /// sha256 of `markdown`'s bytes, hex — the `based_on_hash` a later
    /// `save_workbook` call passes.
    pub hash: String,
    /// Absolute path, under `<data>/workbooks/`.
    pub path: String,
}

/// Transport-agnostic core of `read_workbook`. Deliberately does **not**
/// call [`parse_workbook`] or [`parse_front_matter`] — a document whose
/// front matter is malformed enough for [`open_workbook_via`] to reject it
/// must still be readable here, so it can be repaired in the editor. That
/// separation from `open_workbook` is this command's entire reason to
/// exist; do not add a parse/validation step here.
fn read_workbook_via(data_dir: &Path, id_or_path: &str) -> Result<WorkbookSource, IpcError> {
    let path = resolve_workbook_path(data_dir, id_or_path)?;
    let markdown = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
    let hash = sha256_hex(markdown.as_bytes());
    Ok(WorkbookSource { markdown, hash, path: path.display().to_string() })
}

/// The "no session bound" `ChannelLookup` (ledger R41): every `[Channel]`
/// reference then surfaces as a per-cell `math_unknown_channel` rather than
/// rejecting the whole command.
fn empty_session_handle() -> SessionHandle {
    SessionHandle::from_channels(
        SessionMetaInput { session_id: String::new(), device_id: None, timestamp_utc_ms: 0, config_checksum: None },
        Vec::new(),
    )
}

/// Transport-agnostic core of `eval_workbook`. `session_id = None` evaluates
/// against [`empty_session_handle`] and [`MathLapContext::empty`] (ledger
/// R41); `session_id = Some(id)` for an id that does not exist is a caller
/// error (`not_found`), not a per-cell one.
fn eval_workbook_via(data_dir: &Path, id: &str, session_id: Option<&str>) -> Result<Vec<CellOutput>, IpcError> {
    let path = resolve_workbook_path(data_dir, id)?;
    let markdown = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
    let (doc, structural) = parse_workbook(&markdown).map_err(fatal_parse_error)?;

    let (handle, lap_ctx) = match session_id {
        Some(sid) => (load_session_handle(data_dir, sid)?, load_lap_context(data_dir, sid)),
        None => (empty_session_handle(), MathLapContext::empty()),
    };

    // `eval_cells` used to panic here whenever a definition name repeated
    // anywhere in the document (`resolve_workbook_defs`'s output was keyed
    // by bare name, so a duplicate collapsed to one entry and the second
    // owning cell's `math_cell_defs` found nothing left to remove). Fixed at
    // the root, ledger R47: `resolve_workbook_defs`/`math_cell_defs` now key
    // by `(cell_id, name)`, so every cell — including the offending one —
    // always gets its own result back; no defensive net needed here.
    let cell_results = idl_rs::workbook::v3::eval_cells(&doc, &structural, &handle, &lap_ctx);

    let out = cell_results
        .iter()
        .zip(doc.cells.iter())
        .map(|(cell_eval, cell_doc)| {
            let defs = cell_eval
                .defs
                .iter()
                .map(|d| CellDefResult {
                    name: d.name.clone(),
                    label: d.label.clone(),
                    value: d.value.as_ref().map(HostChannelRef::from),
                    error: d.error.clone().map(IpcError::from),
                })
                .collect();

            let errors = cell_eval
                .errors
                .iter()
                .map(|e| match e {
                    CellError::Structural(we) => IpcError::from(we),
                    CellError::Eval(me) => IpcError::from(me.clone()),
                })
                .collect();

            let value = table_cell_value(cell_eval.kind, cell_doc, session_id.is_some(), &handle);

            CellOutput { cell_id: cell_eval.cell_id.clone(), kind: cell_kind_str(cell_eval.kind).to_string(), value, defs, errors }
        })
        .collect();

    Ok(out)
}

/// A `table` cell's `value` (C3 §3.4): `{ model, results }` when a session is
/// bound and the cell's JSON parsed; `null` otherwise (no session bound —
/// nothing to evaluate against, not an error — or the JSON didn't parse,
/// whose structural error already lives in `CellOutput.errors`). `null` for
/// `math`/`js` cells unconditionally (their results live in `defs`).
/// Wave 1's `row_windows` is `vec![None; rows.len()]` — C2 §4's per-row
/// `context {sessionId, lapIndex}` binding has no C3 argument to carry it
/// yet.
// TODO(idl0): wire per-row `context` binding once a C3 argument exists for
// it — wave 1 evaluates every table row over the whole bound session.
fn table_cell_value(kind: CellKindToken, cell_doc: &CellDoc, session_bound: bool, handle: &SessionHandle) -> Option<serde_json::Value> {
    if kind != CellKindToken::Table || !session_bound {
        return None;
    }
    let model = cell_doc.table.as_ref()?;
    let row_windows = vec![None; model.rows.len()];
    let results = evaluate_table(handle, model, &row_windows);
    Some(serde_json::json!({ "model": model, "results": results }))
}

/// Transport-agnostic core of `save_workbook`. Order of operations (C4 §4):
/// resolve the target, parse (an `Err` writes nothing), hash, register the
/// expected hash **before** the write (step 3's load-bearing ordering), then
/// `write_atomic`.
fn save_workbook_via(
    data_dir: &Path,
    hashes: &ExpectedHashSet,
    id: &str,
    markdown: &str,
    based_on_hash: Option<&str>,
) -> Result<SaveResult, IpcError> {
    // Step 1 (reordered ahead of target resolution, ledger R48): parse; an
    // `Err` writes nothing. Parsing first (rather than after resolving the
    // target, the brief's original order) is required so a brand-new
    // workbook's file name can be derived from the parsed front matter's
    // `name` below — C2 §1 requires `name`, so a successful parse always
    // has one.
    let Ok((doc, _)) = parse_workbook(markdown) else {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, "workbook markdown front matter failed to parse"));
    };

    // Step 2: resolve the target. `id` matches an existing workbook (path or
    // scanned front-matter id) whenever one exists; when it does not *and*
    // `based_on_hash` is `None`, this is a genuinely new workbook — C4 §2
    // fixes its path as `workbooks/<file_name>.idl1wb`, `file_name` a
    // filesystem-sanitised form of the *display* name (never `id`, which
    // never names the file), with SPEC §15.1's `-2`/`-3`… collision suffix
    // (ledger R48).
    let target = match resolve_workbook_path(data_dir, id) {
        Ok(path) => path,
        Err(e) if based_on_hash.is_some() => return Err(e),
        Err(_) => {
            let workbooks_dir = data_dir.join("workbooks");
            let stem = sanitize_file_name_stem(&doc.name);
            let file_base = idl_rs::session::filename::unique_file_base(&stem, |candidate| {
                workbooks_dir.join(format!("{candidate}.idl1wb")).exists()
            });
            workbooks_dir.join(format!("{file_base}.idl1wb"))
        }
    };

    // Step 3: hash.
    let hash = sha256_hex(markdown.as_bytes());

    // Step 4: register before the write — C4 §4 step 3's load-bearing order.
    hashes.expect(target.clone(), hash.clone());

    // Step 5: write.
    write_atomic(data_dir, &target, markdown.as_bytes(), based_on_hash).map_err(|e| match e.kind {
        AtomicWriteErrorKind::RenameConflict => IpcError::with_detail(
            IpcErrorKind::Conflict,
            e.message.clone(),
            serde_json::json!({ "expected": based_on_hash, "found": e.message }),
        ),
        AtomicWriteErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
    })?;

    let saved_utc_ms =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
    Ok(SaveResult { hash, saved_utc_ms })
}

/// Cell ids affected by an edit between two parses (decided here — L3 ships
/// no diff function, `grep core/src/workbook/v3 diff`: no hits): every id
/// whose `raw_fence_body` changed, plus every id present in exactly one of
/// the two parses (added or removed). Ids are stable across an edit (C2
/// §2.2 puts them in the fence string), so this is a plain id-keyed compare,
/// not a content-similarity heuristic.
fn diff_cell_ids(before: &[CellDoc], after: &[CellDoc]) -> Vec<String> {
    let before_by_id: HashMap<&str, &CellDoc> = before.iter().map(|c| (c.id.as_str(), c)).collect();
    let after_by_id: HashMap<&str, &CellDoc> = after.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut ids: Vec<String> = Vec::new();
    for (id, cell) in &after_by_id {
        match before_by_id.get(id) {
            Some(prev) if prev.raw_fence_body != cell.raw_fence_body => ids.push((*id).to_string()),
            Some(_) => {}
            None => ids.push((*id).to_string()), // added
        }
    }
    for id in before_by_id.keys() {
        if !after_by_id.contains_key(id) {
            ids.push((*id).to_string()); // removed
        }
    }
    ids
}

/// Transport-agnostic core of `watch_workbook`: resolves `id`, keeps the
/// current parse's cells as a baseline, starts a [`WorkbookWatcher`] on
/// `<data>/workbooks`, and calls `on_event` with the cells that differ from
/// the baseline on every external change to this workbook's own path —
/// updating the baseline after each event. `hashes` is shared with
/// `save_workbook` so the app's own writes are suppressed (C4 §4).
fn watch_workbook_via(
    data_dir: &Path,
    hashes: Arc<ExpectedHashSet>,
    id: &str,
    on_event: impl Fn(WorkbookEvent) + Send + Sync + 'static,
) -> Result<WorkbookWatcher, IpcError> {
    let path = resolve_workbook_path(data_dir, id)?;
    let markdown = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
    let baseline_cells = parse_workbook(&markdown).map(|(doc, _)| doc.cells).unwrap_or_default();
    let baseline = std::sync::Mutex::new(baseline_cells);

    let workbooks_dir = data_dir.join("workbooks");
    let watch_path = path.clone();
    WorkbookWatcher::new(&workbooks_dir, hashes, move |changed_path: &Path| {
        if changed_path != watch_path {
            return; // not this workbook (design §7, watcher scope note)
        }
        let Ok(markdown) = std::fs::read_to_string(&watch_path) else { return };
        let Ok((doc, _)) = parse_workbook(&markdown) else { return };
        let mut prev = baseline.lock().unwrap();
        let cell_ids = diff_cell_ids(&prev, &doc.cells);
        *prev = doc.cells;
        drop(prev);
        on_event(WorkbookEvent { kind: "changed".to_string(), cell_ids });
    })
    .map_err(|e| IpcError::new(IpcErrorKind::Internal, format!("starting workbook watcher: {e}")))
}

/// C3 §3.4 `open_workbook(id_or_path)`.
#[tauri::command]
pub fn open_workbook(id_or_path: String, data_dir: tauri::State<'_, DataDir>) -> Result<WorkbookHandle, IpcError> {
    open_workbook_via(&data_dir.0, &id_or_path)
}

/// C3 §3.4 `read_workbook(id_or_path)` — returns the file's raw text and its
/// hash, without parsing. See [`read_workbook_via`] for why this must not
/// call `parse_workbook`.
#[tauri::command]
pub fn read_workbook(id_or_path: String, data_dir: tauri::State<'_, DataDir>) -> Result<WorkbookSource, IpcError> {
    read_workbook_via(&data_dir.0, &id_or_path)
}

/// C3 §3.4 `eval_workbook(id, session_id)`.
#[tauri::command]
pub fn eval_workbook(
    id: String,
    session_id: Option<String>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<Vec<CellOutput>, IpcError> {
    eval_workbook_via(&data_dir.0, &id, session_id.as_deref())
}

/// C3 §3.4 `save_workbook(id, markdown, based_on_hash)`.
#[tauri::command]
pub fn save_workbook(
    id: String,
    markdown: String,
    based_on_hash: Option<String>,
    data_dir: tauri::State<'_, DataDir>,
    hashes: tauri::State<'_, Hashes>,
) -> Result<SaveResult, IpcError> {
    save_workbook_via(&data_dir.0, &hashes.0, &id, &markdown, based_on_hash.as_deref())
}

/// C3 §3.4 `watch_workbook(id, channel)`. Parks the started watcher in
/// [`Watchers`] keyed by `id` for the app's lifetime — re-subscribing to the
/// same id replaces (and so stops) the previous one. No unsubscribe command
/// in wave 1 (Tauri v2 gives no observable channel-close signal).
#[tauri::command]
pub fn watch_workbook(
    id: String,
    channel: tauri::ipc::Channel<WorkbookEvent>,
    data_dir: tauri::State<'_, DataDir>,
    hashes: tauri::State<'_, Hashes>,
    watchers: tauri::State<'_, Watchers>,
) -> Result<(), IpcError> {
    let watcher = watch_workbook_via(&data_dir.0, hashes.0.clone(), &id, move |event| {
        let _ = channel.send(event);
    })?;
    watchers.0.lock().unwrap().insert(id, watcher);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-workbook-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_workbook(root: &Path, file_name: &str, markdown: &str) -> PathBuf {
        let dir = root.join("workbooks");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(file_name);
        std::fs::write(&path, markdown).unwrap();
        path
    }

    const WB_ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn two_cell_markdown() -> String {
        format!(
            "---\nid: {WB_ID}\nname: Fork tuning\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = 1\n```\n\n```js id=bbbbbbbb\n1 + 1\n```\n"
        )
    }

    fn seed_session(root: &Path, session_id: &str, channel_id: &str, samples: Vec<f64>, t_us: Vec<i64>) {
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: String::new(),
            channels: vec![Channel {
                channel_id: channel_id.to_string(),
                t_us,
                t_recorded_us: None,
                nominal_rate_hz: 10.0,
                column: RawColumn::F64(samples),
                source_kind: "test".to_string(),
                unit: String::new(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    // ---- Step 2: open_workbook ----

    #[test]
    fn open_workbook_a_file_with_front_matter_and_two_cells_handle_carries_id_name_and_cell_count_2() {
        // Arrange
        let root = temp_root();
        write_workbook(&root, "fork-tuning.idl1wb", &two_cell_markdown());

        // Act
        let handle = open_workbook_via(&root, WB_ID).unwrap();

        // Assert
        assert_eq!(handle.id, WB_ID);
        assert_eq!(handle.name, "Fork tuning");
        assert_eq!(handle.cell_count, 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_workbook_id_that_matches_no_file_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = open_workbook_via(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_workbook_front_matter_without_an_id_workbook_missing_front_matter_id() {
        // Arrange
        let root = temp_root();
        let markdown = "---\nname: No id\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = 1\n```\n";
        // Not resolvable by id (there is none), so this must be opened by path.
        let path = write_workbook(&root, "no-id.idl1wb", markdown);

        // Act
        let err = open_workbook_via(&root, path.to_str().unwrap()).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::WorkbookMissingFrontMatterId);

        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- Step 3: read_workbook ----

    #[test]
    fn read_workbook_via_lookup_by_id_returns_verbatim_markdown_and_matching_hash() {
        // Arrange
        let root = temp_root();
        let markdown = two_cell_markdown();
        let path = write_workbook(&root, "fork-tuning.idl1wb", &markdown);

        // Act
        let source = read_workbook_via(&root, WB_ID).unwrap();

        // Assert
        assert_eq!(source.markdown, markdown);
        assert_eq!(source.path, path.display().to_string());
        assert_eq!(source.hash, sha256_hex(markdown.as_bytes()));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_workbook_via_lookup_by_literal_path_returns_the_same_result_as_by_id() {
        // Arrange
        let root = temp_root();
        let markdown = two_cell_markdown();
        let path = write_workbook(&root, "fork-tuning.idl1wb", &markdown);

        // Act
        let source = read_workbook_via(&root, path.to_str().unwrap()).unwrap();

        // Assert
        assert_eq!(source.markdown, markdown);
        assert_eq!(source.path, path.display().to_string());
        assert_eq!(source.hash, sha256_hex(markdown.as_bytes()));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_workbook_via_unknown_id_or_path_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = read_workbook_via(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_workbook_via_malformed_front_matter_open_workbook_would_reject_still_succeeds_with_raw_text() {
        // Arrange — the exact fixture `open_workbook_front_matter_without_an_id_...`
        // above proves `open_workbook_via` rejects (missing `id:`); this test
        // proves `read_workbook_via` does not parse, so it reads it anyway.
        let root = temp_root();
        let markdown = "---\nname: No id\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = 1\n```\n";
        let path = write_workbook(&root, "no-id.idl1wb", markdown);

        // Act
        let source = read_workbook_via(&root, path.to_str().unwrap()).unwrap();

        // Assert
        assert_eq!(source.markdown, markdown);
        assert_eq!(source.hash, sha256_hex(markdown.as_bytes()));

        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- Step 2: save_workbook ----

    #[test]
    fn save_workbook_new_workbook_file_written_and_hash_matches_the_bytes() {
        // Arrange
        let root = temp_root();
        let hashes = ExpectedHashSet::new();
        let markdown = two_cell_markdown();

        // Act
        let result = save_workbook_via(&root, &hashes, WB_ID, &markdown, None).unwrap();

        // Assert — ledger R48: the file name comes from front matter `name`
        // ("Fork tuning", `two_cell_markdown`'s fixture), sanitised, never
        // from `id`.
        assert_eq!(result.hash, sha256_hex(markdown.as_bytes()));
        let target = root.join("workbooks").join("Fork tuning.idl1wb");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), markdown);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_workbook_creating_two_new_workbooks_with_the_same_front_matter_name_the_second_gets_a_collision_suffix() {
        // Arrange — ledger R48's SPEC §15.1 collision rule: same display
        // name, different workbook ids, second file becomes `-2`.
        let root = temp_root();
        let hashes = ExpectedHashSet::new();
        let other_id = "1a2b3c4d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";
        let first = two_cell_markdown();
        let second = format!(
            "---\nid: {other_id}\nname: Fork tuning\nversion: 3\n---\n\n```math id=cccccccc\ny = 1\n```\n"
        );

        // Act
        save_workbook_via(&root, &hashes, WB_ID, &first, None).unwrap();
        save_workbook_via(&root, &hashes, other_id, &second, None).unwrap();

        // Assert
        assert!(root.join("workbooks").join("Fork tuning.idl1wb").exists());
        assert!(root.join("workbooks").join("Fork tuning-2.idl1wb").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sanitize_file_name_stem_a_windows_reserved_device_name_case_insensitively_falls_back_to_workbook() {
        // Arrange / Act / Assert — ledger R49's Minor: a stem that would
        // otherwise sanitise to a reserved device name falls back the same
        // way the empty-after-trim case already does, not through to a raw
        // path a later `write_atomic` would fail on with an opaque `io`
        // error.
        assert_eq!(sanitize_file_name_stem("CON"), "workbook");
        assert_eq!(sanitize_file_name_stem("con"), "workbook");
        assert_eq!(sanitize_file_name_stem("Lpt3"), "workbook");
        // Exact match only — a name that merely contains a reserved word is
        // not reserved.
        assert_eq!(sanitize_file_name_stem("CONTAINER"), "CONTAINER");
        assert_eq!(sanitize_file_name_stem("Fork tuning"), "Fork tuning");
    }

    #[test]
    fn save_workbook_based_on_hash_matching_the_file_on_disk_succeeds_and_the_expected_hash_set_holds_the_new_hash_before_the_write(
    ) {
        // Arrange
        let root = temp_root();
        let hashes = ExpectedHashSet::new();
        let v1 = two_cell_markdown();
        let h1 = save_workbook_via(&root, &hashes, WB_ID, &v1, None).unwrap().hash;
        let target = root.join("workbooks").join("Fork tuning.idl1wb");
        let v2 = format!("{v1}\n<!-- edited -->\n");
        let h2 = sha256_hex(v2.as_bytes());

        // Act
        let result = save_workbook_via(&root, &hashes, WB_ID, &v2, Some(&h1)).unwrap();

        // Assert — the expected-hash set already held h2 before the write
        // happened (C4 §4 step 3): a self-write callback checking right now
        // would see it.
        assert!(hashes.check_and_consume(&target, &h2));
        assert_eq!(result.hash, h2);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), v2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_workbook_based_on_hash_stale_conflict_file_on_disk_unchanged() {
        // Arrange — R44 supersedes the pre-ruling `invalid_argument` guess:
        // a `RenameConflict` maps to the `conflict` kind.
        let root = temp_root();
        let hashes = ExpectedHashSet::new();
        let v1 = two_cell_markdown();
        let h1 = save_workbook_via(&root, &hashes, WB_ID, &v1, None).unwrap().hash;
        let target = root.join("workbooks").join("Fork tuning.idl1wb");
        let external = format!("{v1}\n<!-- external edit -->\n");
        std::fs::write(&target, &external).unwrap(); // changed underneath the caller

        // Act — caller still thinks the hash is h1; `my_write` parses fine
        // (Step 2 must succeed) so the conflict comes from Step 5's rename
        // check, not an early parse rejection.
        let my_write = format!("{v1}\n<!-- my write -->\n");
        let err = save_workbook_via(&root, &hashes, WB_ID, &my_write, Some(&h1)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Conflict);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), external);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_workbook_markdown_whose_front_matter_does_not_parse_invalid_argument_nothing_written() {
        // Arrange
        let root = temp_root();
        let hashes = ExpectedHashSet::new();

        // Act
        let err = save_workbook_via(&root, &hashes, WB_ID, "not a workbook at all", None).unwrap_err();

        // Assert — parsing fails before any file name can even be derived,
        // so nothing under `workbooks/` exists at all.
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert!(!root.join("workbooks").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- Step 3: eval_workbook ----

    #[test]
    fn eval_workbook_a_math_cell_over_a_bound_session_def_value_is_a_host_channel_ref_with_the_channels_length() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1", "ChanA", vec![1.0, 2.0, 3.0], vec![0, 100_000, 200_000]);
        let markdown = format!(
            "---\nid: {WB_ID}\nname: Test\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = [ChanA]\n```\n"
        );
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let out = eval_workbook_via(&root, WB_ID, Some("s1")).unwrap();

        // Assert
        assert_eq!(out.len(), 1);
        let x = &out[0].defs[0];
        assert_eq!(x.name, "x");
        assert!(x.error.is_none());
        assert_eq!(x.value.as_ref().unwrap().length, 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_workbook_a_math_cell_referencing_an_unknown_channel_command_succeeds_that_def_carries_math_unknown_channel(
    ) {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1", "ChanA", vec![1.0], vec![0]);
        let markdown = format!(
            "---\nid: {WB_ID}\nname: Test\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = [NopeChannel]\n```\n"
        );
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let out = eval_workbook_via(&root, WB_ID, Some("s1")).unwrap();

        // Assert
        let x = &out[0].defs[0];
        assert!(x.value.is_none());
        assert_eq!(x.error.as_ref().unwrap().kind, IpcErrorKind::MathUnknownChannel);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_workbook_duplicate_definitions_both_cells_returned_the_offending_cell_carries_workbook_duplicate_definition_in_errors(
    ) {
        // Arrange — ledger R47 fixed the root cause (`resolve_workbook_defs`/
        // `math_cell_defs` now key by `(cell_id, name)`), so this asserts
        // the brief's original, C3-correct shape: both cells returned, the
        // offending cell carries the structural error, neither cell's
        // command rejects.
        let root = temp_root();
        let markdown = format!(
            "---\nid: {WB_ID}\nname: Test\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\nx = 2\n```\n"
        );
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let out = eval_workbook_via(&root, WB_ID, None).unwrap();

        // Assert
        assert_eq!(out.len(), 2);
        assert!(out[1].errors.iter().any(|e| e.kind == IpcErrorKind::WorkbookDuplicateDefinition));
        // Both cells still produced their own def value — a duplicate name
        // never blanks the sibling cell's own result (CLAUDE.md §5).
        assert_eq!(out[0].defs[0].value.as_ref().unwrap().length, 1);
        assert_eq!(out[1].defs[0].value.as_ref().unwrap().length, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_workbook_no_session_bound_every_channel_ref_is_math_unknown_channel_command_still_succeeds() {
        // Arrange
        let root = temp_root();
        let markdown = format!(
            "---\nid: {WB_ID}\nname: Test\nversion: 3\n---\n\n```math id=aaaaaaaa\nx = [ChanA]\n```\n"
        );
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let out = eval_workbook_via(&root, WB_ID, None).unwrap();

        // Assert
        let x = &out[0].defs[0];
        assert_eq!(x.error.as_ref().unwrap().kind, IpcErrorKind::MathUnknownChannel);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_workbook_front_matter_version_2_command_rejects_with_workbook_unsupported_version() {
        // Arrange
        let root = temp_root();
        let markdown = format!("---\nid: {WB_ID}\nname: Test\nversion: 2\n---\n\n```math id=aaaaaaaa\nx = 1\n```\n");
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let err = eval_workbook_via(&root, WB_ID, None).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::WorkbookUnsupportedVersion);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_workbook_a_table_cell_value_carries_model_and_results() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1", "ChanA", vec![1.0, 2.0], vec![0, 100_000]);
        let table_json = r#"{"columns":[{"id":"c1","name":null,"template":null}],"rows":[{"id":"r1","context":null}],"cells":[[{"formula":null,"literal":5.0,"name":null}]]}"#;
        let markdown = format!("---\nid: {WB_ID}\nname: Test\nversion: 3\n---\n\n```table id=aaaaaaaa\n{table_json}\n```\n");
        write_workbook(&root, "test.idl1wb", &markdown);

        // Act
        let out = eval_workbook_via(&root, WB_ID, Some("s1")).unwrap();

        // Assert
        let value = out[0].value.as_ref().unwrap();
        assert!(value.get("model").is_some());
        let results = value.get("results").unwrap().as_array().unwrap();
        assert_eq!(results[0][0]["value"], serde_json::json!(5.0));

        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- Step 4: watch_workbook ----

    #[test]
    fn watch_workbook_external_edit_to_a_watched_workbook_event_names_only_the_changed_cell() {
        // Arrange
        let root = temp_root();
        let markdown = two_cell_markdown();
        let path = write_workbook(&root, "test.idl1wb", &markdown);
        let hashes = Arc::new(ExpectedHashSet::new());
        let (tx, rx) = std::sync::mpsc::channel::<WorkbookEvent>();

        let _watcher = watch_workbook_via(&root, hashes, WB_ID, move |e| {
            let _ = tx.send(e);
        })
        .unwrap();

        // Act — change only the math cell's body.
        let edited = markdown.replace("x = 1", "x = 2");
        std::fs::write(&path, &edited).unwrap();

        // Assert
        let event = rx.recv_timeout(std::time::Duration::from_millis(1000)).expect("callback fired");
        assert_eq!(event.kind, "changed");
        assert_eq!(event.cell_ids, vec!["aaaaaaaa".to_string()]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn watch_workbook_the_apps_own_save_hash_pre_registered_no_event() {
        // Arrange
        let root = temp_root();
        let markdown = two_cell_markdown();
        let path = write_workbook(&root, "test.idl1wb", &markdown);
        let hashes = Arc::new(ExpectedHashSet::new());
        let (tx, rx) = std::sync::mpsc::channel::<WorkbookEvent>();

        let _watcher = watch_workbook_via(&root, Arc::clone(&hashes), WB_ID, move |e| {
            let _ = tx.send(e);
        })
        .unwrap();

        // Act — the app's own write, hash registered before the write (C4 §4 step 3).
        let edited = markdown.replace("x = 1", "x = 2");
        hashes.expect(path.clone(), sha256_hex(edited.as_bytes()));
        std::fs::write(&path, &edited).unwrap();

        // Assert
        assert!(rx.recv_timeout(std::time::Duration::from_millis(500)).is_err(), "no event for a self-write");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn watch_workbook_unknown_id_not_found() {
        // Arrange
        let root = temp_root();
        std::fs::create_dir_all(root.join("workbooks")).unwrap();
        let hashes = Arc::new(ExpectedHashSet::new());

        // Act — `WorkbookWatcher`'s `Ok` variant is not `Debug`, so this
        // can't use `unwrap_err()`.
        let result = watch_workbook_via(&root, hashes, "nope", |_| {});

        // Assert
        match result {
            Err(e) => assert_eq!(e.kind, IpcErrorKind::NotFound),
            Ok(_) => panic!("expected not_found for an unresolvable id"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }
}
