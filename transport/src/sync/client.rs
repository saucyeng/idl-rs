//! The LAN sync client (PLAN §3/§4, this task's brief): fetches the peer's
//! manifest, asks core's pure [`plan_sync`] what moves, moves it —
//! resumably, reporting progress. Every decision (what to request next, how
//! to resume, when a response is stale) lives in [`plan_sync`]/core's
//! `install`, both pure; this module is the thin socket work around them —
//! it never parses a workbook, a `session.json` or a Parquet file itself
//! (that is core's `install`'s job, PLAN §2).

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use futures::StreamExt;
use tokio::io::AsyncWriteExt;

use idl_rs::store::atomic::sha256_hex;
use idl_rs::store::blob::blob_path;
use idl_rs::store::sync::apply::{install, InstallContext, InstallOutcome};
use idl_rs::store::sync::diff::{plan_sync, SyncAction, SyncClass, SyncItem};
use idl_rs::store::sync::ids::{is_valid_id, safe_join, IdClass};
use idl_rs::store::sync::manifest::{build_manifest, Manifest};

use crate::error::{TransportError, TransportErrorKind};
use crate::wifi_transport::{parse_content_range, range_header};

use super::wire::{PairRequest, PairResponse, Peer, PROTOCOL_VERSION};

/// What one sync run did (C3 §3.9's `SyncResult`, plus what the command
/// layer needs to report honestly — this task's brief interface). Every
/// count is "successfully applied this run", never a planned/attempted
/// count — a failed item (logged, not aborting the run, see
/// [`sync_with_peer`]'s doc comment) is simply absent from every field
/// here, visible only as the counts coming up short of the plan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncRunResult {
    /// Successful `SyncClass::Blob` transfers, pull and push combined.
    /// Content-addressed session channels (`SyncClass::Derived`) and
    /// `data.parquet` are counted under [`Self::sessions_updated`] instead —
    /// they are always session-scoped, unlike a raw blob.
    pub blobs_transferred: u32,
    /// Workbooks installed locally by a `Pull`/`PullForMerge` this run
    /// (`install`'s own first-sync-vs-merge distinction is not surfaced
    /// here — both count). A `Push` of a workbook is not counted: the merge
    /// it may cause happens on the peer, whose outcome this run never
    /// observes (the peer answers a bare `200`).
    pub workbooks_merged: u32,
    /// Conflict cells created across every workbook merge this run
    /// (`InstallOutcome::Merged { conflicts }`, summed).
    pub conflicts: u32,
    /// Distinct session ids that had at least one successful `data.parquet`,
    /// derived-channel, or `session.json` transfer this run (pull or push).
    /// Always `sessions_touched.len()` — the two fields can never disagree.
    pub sessions_updated: u32,
    /// The session ids counted by [`Self::sessions_updated`], sorted. A
    /// Rust-side detail only (lead ruling R104 addendum): the tauri command
    /// layer walks this to call `idl_rs::store::catalog::index_session` per
    /// id (mirroring `rescan_tracks_via`'s own per-session, non-rebuild
    /// re-index) — it is never forwarded to the frontend; C3 §3.9's
    /// `SyncResult` wire shape is unchanged by this field.
    pub sessions_touched: Vec<String>,
    /// Successful `SyncClass::Track` transfers, pull and push combined.
    pub tracks_updated: u32,
    /// Successful `SyncClass::Profile` transfers, pull and push combined.
    pub profiles_updated: u32,
}

/// One progress tick. `phase` is C3 §3.9's union as Task 1 widened it:
/// `"manifest"`, `"blobs"`, `"sessions"`, `"workbooks"`, `"tracks"`,
/// `"profiles"`, always in that order and never repeating (this task's
/// brief, "Key logic"). `total` is the planned action count for the current
/// phase; it is `None` only during `"manifest"`, before [`plan_sync`] has
/// run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncProgress {
    pub done: u64,
    pub total: Option<u64>,
    pub phase: &'static str,
}

/// The five post-manifest phases, in the fixed order [`SyncProgress::phase`]
/// moves through. A [`SyncClass`] maps onto exactly one via [`phase_of`].
const PHASES: [&str; 5] = ["blobs", "sessions", "workbooks", "tracks", "profiles"];

/// Builds a `TransportErrorKind::Sync` error with `message`.
fn sync_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Sync, message.into())
}

/// Client-side mirror of `server.rs`'s `MAX_DOCUMENT_BODY_BYTES` (R102): a
/// paired peer is not more trusted than a client — the same generous cap
/// bounds a `session.json`/`.idl1wb`/`.idl0t`/`.idl0p` body and the
/// manifest itself, whichever side sends it.
const MAX_DOCUMENT_BODY_BYTES: u64 = 16 * 1024 * 1024;

/// Client-side mirror of `server.rs`'s `MAX_RAW_FILE_BODY_BYTES` (R102): the
/// same cap that bounds an inbound blob/derived/`data.parquet` `PUT` on the
/// server bounds what this client will accept from a `GET` against a peer.
const MAX_RAW_FILE_BODY_BYTES: u64 = 512 * 1024 * 1024;

/// The per-class response-body cap a pulled item is held to (R102, the
/// reverse of `server.rs`'s per-route `read_body(_, MAX_*_BODY_BYTES)`
/// calls): `Blob`/`Derived`/`DataParquet` are the large/raw tier, every
/// other class is the small-document tier.
fn body_cap(class: SyncClass) -> u64 {
    match class {
        SyncClass::Blob | SyncClass::Derived | SyncClass::DataParquet => MAX_RAW_FILE_BODY_BYTES,
        SyncClass::SessionJson | SyncClass::Workbook | SyncClass::Track | SyncClass::Profile => MAX_DOCUMENT_BODY_BYTES,
    }
}

/// `true` if `item`'s `session_id`/`key` are shaped the way every other
/// id-addressed path in this crate requires (R102 Minor, R100 discipline):
/// `session_id` (present only for the session-scoped classes) as
/// `IdClass::Session`; `key` as `IdClass::Session` for a hash- or
/// session-keyed class (`Blob`/`Derived`/`DataParquet`/`SessionJson` — a
/// sha256 hex digest and a session id are both "lowercase hex, even
/// length, 16..=64 chars", the shape `IdClass::Session` checks), or
/// `IdClass::Uuid` for `Workbook`/`Track`/`Profile`. Called before any
/// peer-sourced identity is hashed into a local path (this task's own
/// [`tmp_part_path`]) — belt-and-braces the same way `safe_join` already
/// is; a peer manifest is untrusted input like any other.
fn item_shape_is_valid(item: &SyncItem) -> bool {
    if let Some(session_id) = &item.session_id {
        if !is_valid_id(session_id, IdClass::Session) {
            return false;
        }
    }
    let key_class = match item.class {
        SyncClass::Blob | SyncClass::Derived | SyncClass::DataParquet | SyncClass::SessionJson => IdClass::Session,
        SyncClass::Workbook | SyncClass::Track | SyncClass::Profile => IdClass::Uuid,
    };
    is_valid_id(&item.key, key_class)
}

/// Reads `response`'s full body, refusing (before any byte is handed back)
/// a body whose `Content-Length` already exceeds `cap_bytes`, and aborting
/// mid-stream the moment the running total would exceed it — the
/// in-memory counterpart to `download_item`'s on-disk cap check, used for
/// bodies this module never streams to a `.part` file (the manifest).
async fn bounded_bytes(response: reqwest::Response, cap_bytes: u64, what: &str) -> Result<Vec<u8>, TransportError> {
    if let Some(len) = response.content_length() {
        if len > cap_bytes {
            return Err(sync_error(format!("{what}: response is {len} bytes, exceeding the {cap_bytes}-byte cap")));
        }
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| sync_error(format!("{what}: stream error: {e}")))?;
        if buf.len() as u64 + chunk.len() as u64 > cap_bytes {
            return Err(sync_error(format!("{what}: response exceeded the {cap_bytes}-byte cap mid-stream")));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Which [`SyncProgress::phase`] a class's actions are reported under.
/// `DataParquet`/`Derived`/`SessionJson` all live under `"sessions"` — they
/// are always session-scoped and a peer's manifest walk treats them as one
/// unit (C4 §6) — while `Blob` (not session-scoped) gets its own phase.
fn phase_of(class: SyncClass) -> &'static str {
    match class {
        SyncClass::Blob => "blobs",
        SyncClass::DataParquet | SyncClass::Derived | SyncClass::SessionJson => "sessions",
        SyncClass::Workbook => "workbooks",
        SyncClass::Track => "tracks",
        SyncClass::Profile => "profiles",
    }
}

/// Borrows the [`SyncItem`] out of any [`SyncAction`] variant (mirrors
/// core's own private `diff::item_of` — not exported, so this module keeps
/// its own copy).
fn item_of(action: &SyncAction) -> &SyncItem {
    match action {
        SyncAction::Pull(i) | SyncAction::Push(i) | SyncAction::PullForMerge(i) => i,
    }
}

/// Runs one full sync against `peer`. `on_progress` is called on the
/// caller's task; it must not block (this task's brief interface).
///
/// Order of work: check `peer.protocol_version` (refusing before any
/// request if it differs from [`PROTOCOL_VERSION`]) → fetch the peer's
/// manifest → build the local manifest via core → [`plan_sync`] → pull
/// every planned `Pull`/`PullForMerge` action, then push every planned
/// `Push` action, phase by phase in [`PHASES`] order. A single item's
/// failure (a stale 404, a wrong hash, a network blip) is counted and
/// logged (its class and key) but never aborts the run — the shortfall
/// shows up only as the returned [`SyncRunResult`]'s counts coming up short
/// of what [`plan_sync`] planned. `Err` is reserved for a failure that
/// makes the whole run meaningless: a version mismatch, the manifest fetch
/// itself failing (bad token, peer unreachable), or the local manifest walk
/// failing.
pub async fn sync_with_peer(
    data_root: &Path,
    peer: &Peer,
    addr: SocketAddr,
    now_ms: i64,
    on_progress: &(dyn Fn(SyncProgress) + Send + Sync),
) -> Result<SyncRunResult, TransportError> {
    if peer.protocol_version != PROTOCOL_VERSION {
        return Err(sync_error(format!(
            "peer {} speaks protocol {}, this build speaks {PROTOCOL_VERSION}",
            peer.peer_id, peer.protocol_version
        )));
    }

    on_progress(SyncProgress { done: 0, total: None, phase: "manifest" });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}/idl1/v1");

    let remote_manifest = fetch_manifest(&client, &base_url, &peer.token).await?;
    let (local_manifest, local_skipped) =
        build_manifest(data_root, now_ms).map_err(|e| sync_error(format!("building local manifest: {e}")))?;

    let plan = plan_sync(&local_manifest, &remote_manifest, &local_skipped);

    let mut result = SyncRunResult::default();
    let mut sessions_touched: BTreeSet<String> = BTreeSet::new();

    for phase in PHASES {
        let actions: Vec<&SyncAction> = plan.actions.iter().filter(|a| phase_of(item_of(a).class) == phase).collect();
        let total = actions.len() as u64;
        on_progress(SyncProgress { done: 0, total: Some(total), phase });
        for (i, action) in actions.into_iter().enumerate() {
            run_one_action(&client, &base_url, peer, data_root, action, &remote_manifest, &local_manifest, now_ms, &mut result, &mut sessions_touched)
                .await;
            on_progress(SyncProgress { done: (i + 1) as u64, total: Some(total), phase });
        }
    }

    result.sessions_updated = sessions_touched.len() as u32;
    result.sessions_touched = sessions_touched.into_iter().collect(); // BTreeSet -> already sorted
    Ok(result)
}

/// `GET /idl1/v1/manifest` (PLAN §3). A non-success status (wrong token,
/// peer down), an over-cap body (R102: capped at [`MAX_DOCUMENT_BODY_BYTES`],
/// the same document tier `server.rs` holds `session.json`/workbook/track/
/// profile `PUT`s to), or malformed JSON is a typed [`TransportError`] —
/// this is one of the two failures that aborts the whole run (see
/// [`sync_with_peer`]'s doc comment).
async fn fetch_manifest(client: &reqwest::Client, base_url: &str, token: &str) -> Result<Manifest, TransportError> {
    let response = client
        .get(format!("{base_url}/manifest"))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| sync_error(format!("GET /manifest failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(sync_error(format!("GET /manifest returned status {status}")));
    }
    let bytes = bounded_bytes(response, MAX_DOCUMENT_BODY_BYTES, "GET /manifest").await?;
    serde_json::from_slice::<Manifest>(&bytes).map_err(|e| sync_error(format!("GET /manifest returned malformed JSON: {e}")))
}

/// `POST /idl1/v1/pair` (PLAN §3) — the *initiating* side of pairing:
/// redeems `request.code` against the peer at `addr`, which must be the
/// peer that is currently displaying/offering that code (ruling R104: the
/// caller resolves `addr` from a specific chosen [`super::DiscoveredPeer`],
/// never guessed or broadcast to every peer on the LAN). Unauthenticated —
/// this is the one route with no bearer token (PLAN §3) — so this function
/// takes no `token` argument, unlike [`fetch_manifest`]'s sibling calls.
/// Builds its own short-lived `reqwest::Client`, mirroring
/// [`sync_with_peer`]'s own top-level entry point rather than taking one as
/// a parameter, so no caller outside this crate ever needs a `reqwest`
/// dependency of its own (CLAUDE.md §2 — the wire stays in `idl-transport`).
///
/// A non-success status (malformed request body, wrong/expired code, a
/// protocol mismatch the peer's own `check_protocol_version` rejected) or
/// an over-cap/malformed response body is a typed [`TransportError`],
/// naming the status when the server supplied one.
pub async fn pair_with_peer(addr: SocketAddr, request: &PairRequest) -> Result<PairResponse, TransportError> {
    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}/idl1/v1");

    let response = client
        .post(format!("{base_url}/pair"))
        .json(request)
        .send()
        .await
        .map_err(|e| sync_error(format!("POST /pair failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(sync_error(format!("POST /pair returned status {status}: {body}")));
    }
    let bytes = bounded_bytes(response, MAX_DOCUMENT_BODY_BYTES, "POST /pair").await?;
    serde_json::from_slice::<PairResponse>(&bytes).map_err(|e| sync_error(format!("POST /pair returned malformed JSON: {e}")))
}

/// Runs one planned action, tallying its outcome into `result`/
/// `sessions_touched` on success or logging it (never propagating) on
/// failure — the per-item failure containment [`sync_with_peer`]'s doc
/// comment describes.
#[allow(clippy::too_many_arguments)]
async fn run_one_action(
    client: &reqwest::Client,
    base_url: &str,
    peer: &Peer,
    data_root: &Path,
    action: &SyncAction,
    remote_manifest: &Manifest,
    local_manifest: &Manifest,
    now_ms: i64,
    result: &mut SyncRunResult,
    sessions_touched: &mut BTreeSet<String>,
) {
    let item = item_of(action);
    let outcome = match action {
        SyncAction::Pull(_) | SyncAction::PullForMerge(_) => {
            pull_and_install(client, base_url, peer, data_root, item, remote_manifest, now_ms).await.map(|o| (true, Some(o)))
        }
        SyncAction::Push(_) => push_item(client, base_url, peer, data_root, item, local_manifest).await.map(|()| (false, None)),
    };
    match outcome {
        Ok((is_pull, install_outcome)) => tally_success(item, is_pull, install_outcome, result, sessions_touched),
        Err(e) => eprintln!("sync: {:?} {} failed: {e}", item.class, item.key),
    }
}

/// Adds one successful item's effect to the running totals (see each
/// [`SyncRunResult`] field's own doc comment for exactly what it counts).
fn tally_success(item: &SyncItem, is_pull: bool, install_outcome: Option<InstallOutcome>, result: &mut SyncRunResult, sessions_touched: &mut BTreeSet<String>) {
    match item.class {
        SyncClass::Blob => result.blobs_transferred += 1,
        SyncClass::DataParquet | SyncClass::Derived | SyncClass::SessionJson => {
            if let Some(session_id) = &item.session_id {
                sessions_touched.insert(session_id.clone());
            }
        }
        SyncClass::Workbook => {
            if is_pull {
                result.workbooks_merged += 1;
                if let Some(InstallOutcome::Merged { conflicts }) = install_outcome {
                    result.conflicts += conflicts;
                }
            }
        }
        SyncClass::Track => result.tracks_updated += 1,
        SyncClass::Profile => result.profiles_updated += 1,
    }
}

/// Builds the request path for `item` under `base_url` (PLAN §3's route
/// table). `Err` only for a session-scoped class whose `item.session_id` is
/// unexpectedly absent — a [`plan_sync`] invariant this function does not
/// re-derive, just refuses to silently ignore.
fn item_url(base_url: &str, item: &SyncItem) -> Result<String, TransportError> {
    Ok(match item.class {
        SyncClass::Blob => format!("{base_url}/blob/{}", item.key),
        SyncClass::Derived => format!("{base_url}/session/{}/derived/{}.parquet", session_id_of(item)?, item.key),
        SyncClass::DataParquet => format!("{base_url}/session/{}/data.parquet", session_id_of(item)?),
        SyncClass::SessionJson => format!("{base_url}/session/{}/session.json", session_id_of(item)?),
        SyncClass::Workbook => format!("{base_url}/workbook/{}", item.key),
        SyncClass::Track => format!("{base_url}/track/{}", item.key),
        SyncClass::Profile => format!("{base_url}/profile/{}", item.key),
    })
}

fn session_id_of(item: &SyncItem) -> Result<&str, TransportError> {
    item.session_id.as_deref().ok_or_else(|| sync_error(format!("{:?} item {} has no session_id", item.class, item.key)))
}

/// Deterministic `tmp/<digest>.part` path for `item` — the same logical
/// item always names the same partial file, run to run, so a `.part` left
/// behind by an interrupted pull is found and resumed rather than starting
/// over (this task's brief, "Key logic": "if a `.part` for the same item
/// already exists, resume"). `digest` is a sha256 of the item's own
/// identity (class, session id, key) — plain hex, so this is safe
/// regardless of what a hostile peer's manifest put in `item.key`/
/// `item.session_id` (no path-traversal surface: nothing peer-supplied ever
/// reaches a path segment here unescaped).
fn tmp_part_path(data_root: &Path, item: &SyncItem) -> PathBuf {
    let identity = format!("{:?}:{}:{}", item.class, item.session_id.as_deref().unwrap_or(""), item.key);
    let digest = sha256_hex(identity.as_bytes());
    data_root.join("tmp").join(format!("{digest}.part"))
}

/// Streams `item`'s bytes from the peer into its `tmp/<digest>.part` file,
/// resuming from the file's current length if it already exists, and
/// returns the assembled whole-file bytes once the stream completes. Never
/// installs — the caller does that, with `bytes` verified there (core's
/// `install`, this task's brief: "do not duplicate it").
///
/// Resume: a `206` whose `Content-Range` start matches what was asked for
/// is appended to the existing `.part` file — the file on disk is never
/// truncated or rewritten from byte 0 in this path, so a bad chunk mid-
/// stream (a dropped connection) leaves the `.part` file exactly as long as
/// it was before this call, ready to resume again. A `200` (the server
/// ignored `Range`, or none was sent because no `.part` existed) always
/// starts the file fresh. A `416` (R102: the peer's content is now shorter
/// than the local partial — the range this call asked for no longer
/// exists) or a `206` whose start does not match what was asked — either
/// way the existing `.part` can never validly resume — deletes the stale
/// `.part` before returning `Err`, so the next run starts the item over
/// from byte 0 instead of requesting the same doomed range forever.
///
/// The response body itself is bounded by `item.class`'s [`body_cap`]
/// (R102, the reverse of `server.rs`'s per-route `PUT` cap): a
/// `Content-Length` over the cap is refused before any byte is written,
/// and the running total is checked every chunk in case the peer omits
/// `Content-Length` or lies about it — an over-cap transfer deletes the
/// `.part` it was writing to and returns `Err` rather than leaving a
/// cap-sized (or larger) unusable partial on disk.
async fn download_item(client: &reqwest::Client, base_url: &str, token: &str, data_root: &Path, item: &SyncItem) -> Result<Vec<u8>, TransportError> {
    let url = item_url(base_url, item)?;
    let tmp_path = tmp_part_path(data_root, item);
    let tmp_dir = tmp_path.parent().expect("tmp_part_path always has a tmp/ parent");
    tokio::fs::create_dir_all(tmp_dir).await.map_err(|e| sync_error(format!("creating {}: {e}", tmp_dir.display())))?;

    let existing_len = tokio::fs::metadata(&tmp_path).await.map(|m| m.len()).unwrap_or(0);
    let cap = body_cap(item.class);

    let mut request = client.get(&url).bearer_auth(token);
    if existing_len > 0 {
        request = request.header(reqwest::header::RANGE, range_header(existing_len));
    }
    let response = request.send().await.map_err(|e| sync_error(format!("GET {url} failed: {e}")))?;
    let status = response.status();

    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(sync_error(format!("GET {url} returned 416 (peer's content no longer covers the local partial); discarded the stale .part")));
    }
    if !status.is_success() {
        return Err(sync_error(format!("GET {url} returned status {status}")));
    }

    if let Some(body_len) = response.content_length() {
        if existing_len.saturating_add(body_len) > cap {
            return Err(sync_error(format!(
                "GET {url}: response would assemble to {} bytes, exceeding the {cap}-byte cap for {:?}",
                existing_len.saturating_add(body_len),
                item.class
            )));
        }
    }

    let append = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| sync_error(format!("GET {url}: 206 response is missing a Content-Range header")))?
            .to_string();
        let (start_byte, _end_byte, _total_bytes) =
            parse_content_range(&content_range).ok_or_else(|| sync_error(format!("GET {url}: malformed Content-Range header: {content_range}")))?;
        if start_byte != existing_len {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(sync_error(format!(
                "GET {url}: peer resumed at byte {start_byte}, but {existing_len} was requested; discarded the stale .part"
            )));
        }
        true
    } else {
        false
    };

    let mut open_options = tokio::fs::OpenOptions::new();
    open_options.write(true).create(true);
    if append {
        open_options.append(true);
    } else {
        open_options.truncate(true);
    }
    let mut file = open_options.open(&tmp_path).await.map_err(|e| sync_error(format!("opening {}: {e}", tmp_path.display())))?;

    let mut total = existing_len;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| sync_error(format!("GET {url}: stream error: {e}")))?;
        total += chunk.len() as u64;
        if total > cap {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(sync_error(format!("GET {url}: response exceeded the {cap}-byte cap for {:?} mid-stream; discarded the .part", item.class)));
        }
        file.write_all(&chunk).await.map_err(|e| sync_error(format!("writing {}: {e}", tmp_path.display())))?;
    }
    file.flush().await.map_err(|e| sync_error(format!("flushing {}: {e}", tmp_path.display())))?;
    drop(file);

    tokio::fs::read(&tmp_path).await.map_err(|e| sync_error(format!("reading back {}: {e}", tmp_path.display())))
}

/// The manifest-sourced detail `item.class` needs beyond its bare identity
/// (`apply::InstallContext`'s own doc comment) — read from `remote_manifest`,
/// the same document [`plan_sync`] already consulted to plan this pull.
fn build_install_context(item: &SyncItem, remote_manifest: &Manifest) -> Result<InstallContext, TransportError> {
    match item.class {
        SyncClass::DataParquet => {
            let session_id = session_id_of(item)?;
            let entry = remote_manifest
                .sessions
                .iter()
                .find(|s| s.session_id == session_id)
                .and_then(|s| s.data_parquet.as_ref())
                .ok_or_else(|| sync_error(format!("remote manifest has no data.parquet entry for session {session_id}")))?;
            Ok(InstallContext { claimed_data_parquet_versions: Some((entry.importer_version.clone(), entry.seam_correction_version.clone())), ..Default::default() })
        }
        SyncClass::SessionJson => {
            let session_id = session_id_of(item)?;
            let entry = remote_manifest
                .sessions
                .iter()
                .find(|s| s.session_id == session_id)
                .and_then(|s| s.session_json.as_ref())
                .ok_or_else(|| sync_error(format!("remote manifest has no session.json entry for session {session_id}")))?;
            Ok(InstallContext { peer_session_json_updated_at_ms: Some(entry.updated_at_ms), ..Default::default() })
        }
        SyncClass::Workbook => {
            let entry = remote_manifest
                .workbooks
                .iter()
                .find(|w| w.workbook_id == item.key)
                .ok_or_else(|| sync_error(format!("remote manifest has no workbook entry for {}", item.key)))?;
            Ok(InstallContext { peer_workbook_file_name: Some(entry.file_name.clone()), ..Default::default() })
        }
        SyncClass::Blob | SyncClass::Derived | SyncClass::Track | SyncClass::Profile => Ok(InstallContext::default()),
    }
}

/// Downloads `item` (resumably, see [`download_item`]), hands the assembled
/// bytes to core's `install` — which verifies them, merges where the class
/// calls for it, and writes (this task's brief: "do not duplicate it").
///
/// `.part` lifecycle (R102 ruling): a failure while bytes are still
/// incomplete — the network call itself, a stale/mismatched resume
/// ([`download_item`]'s own `Err`s, which already discard a `.part` they
/// know is unrecoverable) — leaves whatever `.part` state `download_item`
/// left behind, so a later run can resume it (the whole point of resume).
/// Once `download_item` returns bytes, they are complete; any failure past
/// that point — a manifest that no longer has the entry
/// ([`build_install_context`]), or `install`'s own hash/parse/version
/// check rejecting the assembled bytes — deletes the `.part` before
/// returning, so a poisoned or mismatched prefix is never resumed against
/// again: the next run restarts the item from byte 0.
async fn pull_and_install(
    client: &reqwest::Client,
    base_url: &str,
    peer: &Peer,
    data_root: &Path,
    item: &SyncItem,
    remote_manifest: &Manifest,
    now_ms: i64,
) -> Result<InstallOutcome, TransportError> {
    if !item_shape_is_valid(item) {
        return Err(sync_error(format!("{:?} item has a malformed session_id/key shape: session_id={:?} key={}", item.class, item.session_id, item.key)));
    }
    let bytes = download_item(client, base_url, &peer.token, data_root, item).await?;

    let install_result = build_install_context(item, remote_manifest)
        .and_then(|ctx| install(data_root, item, &bytes, &peer.name, now_ms, &ctx).map_err(|e| sync_error(format!("install {:?} {}: {e}", item.class, item.key))));

    // The bytes are complete regardless of outcome from here on — discard
    // the `.part` either way (R102).
    let tmp_path = tmp_part_path(data_root, item);
    let _ = tokio::fs::remove_file(&tmp_path).await;

    install_result
}

/// Reads `item`'s bytes from wherever it lives locally, resolving a
/// workbook's `file_name` from `local_manifest` (the id is the identity,
/// C4 §6 — the on-disk file name is manifest-sourced detail, same pattern
/// as `server.rs`'s `resolve_workbook_file_name`). Every path is built
/// through [`safe_join`] (R100 belt-and-braces), even though every
/// component here comes from this machine's own trusted manifest walk, not
/// from the peer.
fn read_local_item_bytes(data_root: &Path, item: &SyncItem, local_manifest: &Manifest) -> Result<Vec<u8>, TransportError> {
    let path = match item.class {
        SyncClass::Blob => blob_path(data_root, &item.key),
        SyncClass::Derived => {
            let session_id = session_id_of(item)?;
            safe_join(data_root, &["sessions", session_id, "derived", &format!("{}.parquet", item.key)])
                .ok_or_else(|| sync_error("derived path would land outside data_root"))?
        }
        SyncClass::DataParquet => {
            let session_id = session_id_of(item)?;
            safe_join(data_root, &["sessions", session_id, "data.parquet"]).ok_or_else(|| sync_error("data.parquet path would land outside data_root"))?
        }
        SyncClass::SessionJson => {
            let session_id = session_id_of(item)?;
            safe_join(data_root, &["sessions", session_id, "session.json"]).ok_or_else(|| sync_error("session.json path would land outside data_root"))?
        }
        SyncClass::Workbook => {
            let file_name = local_manifest
                .workbooks
                .iter()
                .find(|w| w.workbook_id == item.key)
                .map(|w| w.file_name.clone())
                .ok_or_else(|| sync_error(format!("local manifest has no workbook entry for {}", item.key)))?;
            safe_join(data_root, &["workbooks", &format!("{file_name}.idl1wb")]).ok_or_else(|| sync_error("workbook path would land outside data_root"))?
        }
        SyncClass::Track => {
            safe_join(data_root, &["tracks", &format!("{}.idl0t", item.key)]).ok_or_else(|| sync_error("track path would land outside data_root"))?
        }
        SyncClass::Profile => {
            safe_join(data_root, &["profiles", &format!("{}.idl0p", item.key)]).ok_or_else(|| sync_error("profile path would land outside data_root"))?
        }
    };
    std::fs::read(&path).map_err(|e| sync_error(format!("reading local {:?} {}: {e}", item.class, item.key)))
}

/// `PUT`s `item`'s local bytes to the peer. A `200` — whether or not the
/// peer actually wrote anything (its `install` may find the bytes already
/// present) — counts as success (this task's brief: "a `200` reporting no
/// write still counts as success, not a transfer" — that distinction lives
/// in the peer's own `InstallOutcome`, which this response never carries).
async fn push_item(client: &reqwest::Client, base_url: &str, peer: &Peer, data_root: &Path, item: &SyncItem, local_manifest: &Manifest) -> Result<(), TransportError> {
    let bytes = read_local_item_bytes(data_root, item, local_manifest)?;
    let url = item_url(base_url, item)?;
    let response = client.put(&url).bearer_auth(&peer.token).body(bytes).send().await.map_err(|e| sync_error(format!("PUT {url} failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(sync_error(format!("PUT {url} returned status {status}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::pairing::PairingState;
    use crate::sync::server::{SyncServer, SyncServerConfig};
    use idl_rs::store::atomic::sha256_hex as digest;
    use std::net::Ipv4Addr;
    use std::sync::{Arc, Mutex};

    fn temp_data_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-transport-client-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn start_test_server(data_root: PathBuf, token: &str) -> SyncServer {
        let peers = Arc::new(Mutex::new(vec![Peer { peer_id: "server-side".to_string(), name: "Server".to_string(), token: token.to_string(), protocol_version: PROTOCOL_VERSION, paired_at_ms: 0 }]));
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let config =
            SyncServerConfig { data_root, port: 0, bind_addr: Ipv4Addr::LOCALHOST.into(), peer_id: "server-peer".to_string(), name: "Server".to_string() };
        SyncServer::start(config, pairing, peers).await.unwrap()
    }

    fn test_peer(token: &str) -> Peer {
        Peer { peer_id: "server-peer".to_string(), name: "Server".to_string(), token: token.to_string(), protocol_version: PROTOCOL_VERSION, paired_at_ms: 0 }
    }

    fn no_progress(_p: SyncProgress) {}

    /// Milliseconds since the Unix epoch — this test module's own copy of
    /// the same clock read `server.rs`'s tests keep locally, needed because
    /// `handle_pair` redeems against real wall-clock time, so a minted
    /// offer must be comparable to it.
    fn test_now_ms() -> i64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
    }

    #[tokio::test]
    async fn pair_with_peer_a_correct_code_returns_the_offering_peers_identity_and_a_token() {
        // Arrange
        let data_root = temp_data_root();
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let offer = pairing.lock().unwrap().offer(test_now_ms());
        let peers = Arc::new(Mutex::new(Vec::new()));
        let config = SyncServerConfig {
            data_root: data_root.clone(),
            port: 0,
            bind_addr: Ipv4Addr::LOCALHOST.into(),
            peer_id: "offering-peer".to_string(),
            name: "Pit Laptop".to_string(),
        };
        let server = SyncServer::start(config, pairing, peers).await.unwrap();
        let addr = server.local_addr();
        let request = PairRequest { code: offer.code, peer_id: "requesting-peer".to_string(), name: "Pit Tablet".to_string(), protocol_version: PROTOCOL_VERSION };

        // Act
        let response = pair_with_peer(addr, &request).await;

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);

        // Assert
        let response = response.unwrap();
        assert_eq!(response.peer_id, "offering-peer");
        assert_eq!(response.name, "Pit Laptop");
        assert_eq!(response.protocol_version, PROTOCOL_VERSION);
        assert!(!response.token.is_empty());
    }

    #[tokio::test]
    async fn pair_with_peer_a_wrong_code_is_a_sync_error_no_token_returned() {
        // Arrange
        let data_root = temp_data_root();
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let offer = pairing.lock().unwrap().offer(test_now_ms());
        let wrong_code = if offer.code == "000000" { "111111".to_string() } else { "000000".to_string() };
        let peers = Arc::new(Mutex::new(Vec::new()));
        let config = SyncServerConfig {
            data_root: data_root.clone(),
            port: 0,
            bind_addr: Ipv4Addr::LOCALHOST.into(),
            peer_id: "offering-peer".to_string(),
            name: "Pit Laptop".to_string(),
        };
        let server = SyncServer::start(config, pairing, peers).await.unwrap();
        let addr = server.local_addr();
        let request = PairRequest { code: wrong_code, peer_id: "requesting-peer".to_string(), name: "Pit Tablet".to_string(), protocol_version: PROTOCOL_VERSION };

        // Act
        let result = pair_with_peer(addr, &request).await;

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);

        // Assert
        assert_eq!(result.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn tally_success_two_session_scoped_items_for_the_same_session_sessions_touched_names_it_once() {
        // Arrange — R104 addendum: `SyncRunResult::sessions_touched` is the
        // id set `sessions_updated`'s count was always derived from,
        // exposed so `sync_now` can re-index exactly the sessions a run
        // touched rather than rebuilding the whole catalog.
        let mut result = SyncRunResult::default();
        let mut sessions_touched = BTreeSet::new();
        let data_parquet = SyncItem { class: SyncClass::DataParquet, key: "k".to_string(), session_id: Some("session-a".to_string()), size_bytes: 0 };
        let session_json = SyncItem { class: SyncClass::SessionJson, key: "session-a".to_string(), session_id: Some("session-a".to_string()), size_bytes: 0 };

        // Act
        tally_success(&data_parquet, true, None, &mut result, &mut sessions_touched);
        tally_success(&session_json, true, None, &mut result, &mut sessions_touched);
        result.sessions_updated = sessions_touched.len() as u32;
        result.sessions_touched = sessions_touched.into_iter().collect();

        // Assert
        assert_eq!(result.sessions_updated, 1);
        assert_eq!(result.sessions_touched, vec!["session-a".to_string()]);
    }

    #[tokio::test]
    async fn sync_with_peer_a_blob_only_on_the_peer_pulled_verified_counted() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"peer-only blob bytes";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        let result = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(result.blobs_transferred, 1);
        assert_eq!(std::fs::read(blob_path(&local_root, &sha)).unwrap(), content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_a_blob_only_locally_pushed() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"local-only blob bytes";
        let sha = digest(content);
        let path = blob_path(&local_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        let result = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(result.blobs_transferred, 1);
        assert_eq!(std::fs::read(blob_path(&remote_root, &sha)).unwrap(), content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_run_twice_the_second_run_transfers_nothing() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"blob bytes";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        let first = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();
        let second = sync_with_peer(&local_root, &peer, addr, 2000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(first.blobs_transferred, 1);
        assert_eq!(second, SyncRunResult::default());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_a_partial_part_file_present_resumed_and_the_final_bytes_match_the_whole_file() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Pre-seed the deterministic `.part` file with the true prefix of
        // the blob's bytes, exactly as an interrupted previous pull would
        // have left it.
        let item = SyncItem { class: SyncClass::Blob, key: sha.clone(), session_id: None, size_bytes: content.len() as u64 };
        let part_path = tmp_part_path(&local_root, &item);
        std::fs::create_dir_all(part_path.parent().unwrap()).unwrap();
        std::fs::write(&part_path, &content[..10]).unwrap();

        // Act
        let result = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(result.blobs_transferred, 1);
        assert_eq!(std::fs::read(blob_path(&local_root, &sha)).unwrap(), content);
        assert!(!part_path.exists());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_a_workbook_differing_merged_by_core_workbooks_merged_1() {
        // Arrange
        const WB_ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let wb = |body: &str| format!("---\nid: {WB_ID}\nname: Test\n---\n{body}");
        let local_src = wb("```math id=aaaaaaaa\nx = 1\n```\n");
        let peer_src = wb("```math id=aaaaaaaa\nx = 2\n```\n");
        let local_wb_dir = local_root.join("workbooks");
        std::fs::create_dir_all(&local_wb_dir).unwrap();
        std::fs::write(local_wb_dir.join("fork-tuning.idl1wb"), &local_src).unwrap();
        let remote_wb_dir = remote_root.join("workbooks");
        std::fs::create_dir_all(&remote_wb_dir).unwrap();
        std::fs::write(remote_wb_dir.join("fork-tuning.idl1wb"), &peer_src).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        let result = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(result.workbooks_merged, 1);
        let written = std::fs::read_to_string(local_wb_dir.join("fork-tuning.idl1wb")).unwrap();
        assert!(written.contains("x = 2"), "peer's change missing: {written}");

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_a_peer_whose_protocol_version_differs_refused_with_no_request_sent() {
        // Arrange — an address nothing listens on: any real request would
        // fail with a connection error, not the protocol-mismatch message
        // this test asserts, proving the refusal happens before any
        // request goes out.
        let local_root = temp_data_root();
        let unreachable_addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let mut peer = test_peer("tok-1");
        peer.protocol_version = PROTOCOL_VERSION + 1;

        // Act
        let err = sync_with_peer(&local_root, &peer, unreachable_addr, 1000, &no_progress).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(err.message.contains(&(PROTOCOL_VERSION + 1).to_string()), "{}", err.message);
        assert!(err.message.contains(&PROTOCOL_VERSION.to_string()), "{}", err.message);

        let _ = std::fs::remove_dir_all(&local_root);
    }

    #[tokio::test]
    async fn sync_with_peer_a_wrong_token_a_sync_error_nothing_installed() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"blob bytes";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let server = start_test_server(remote_root.clone(), "right-token").await;
        let addr = server.local_addr();
        let peer = test_peer("wrong-token");

        // Act
        let err = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(!blob_path(&local_root, &sha).is_file());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_one_item_404s_the_run_completes_and_the_others_land() {
        // Arrange — a track file whose own content id is not UUID-shaped
        // (`collect_tracks` only requires the file stem to match `track.id`,
        // no shape check) plans a `Pull`, but the server's `GET /track/<id>`
        // route (`is_valid_id(_, IdClass::Uuid)`, R100) refuses to serve it
        // — a real, deterministic `404` for a manifest-listed item,
        // alongside a blob that pulls normally.
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let blob_content = b"lands fine";
        let sha = digest(blob_content);
        let blob_write_path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(blob_write_path.parent().unwrap()).unwrap();
        std::fs::write(&blob_write_path, blob_content).unwrap();

        let track_id = "not-a-real-uuid";
        let track = idl_rs::track_artifact::model::Track {
            id: track_id.to_string(),
            name: "A-Line".to_string(),
            venue: "Whistler".to_string(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 1,
            updated_at_ms: 2,
        };
        idl_rs::track_artifact::write::write_track(&remote_root, &track).unwrap();

        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        let result = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert — the blob still landed even though the track's own `GET`
        // 404d (its id fails the server's `IdClass::Uuid` shape check).
        assert_eq!(result.blobs_transferred, 1);
        assert_eq!(result.tracks_updated, 0);
        assert_eq!(std::fs::read(blob_path(&local_root, &sha)).unwrap(), blob_content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_progress_phases_arrive_in_order_and_never_go_backwards() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"blob bytes";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);
        let seen: Arc<Mutex<Vec<SyncProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = seen.clone();
        let on_progress = move |p: SyncProgress| seen_cb.lock().unwrap().push(p);

        // Act
        sync_with_peer(&local_root, &peer, addr, 1000, &on_progress).await.unwrap();

        // Assert
        let ticks = seen.lock().unwrap();
        assert!(!ticks.is_empty());
        assert_eq!(ticks[0].phase, "manifest");
        assert_eq!(ticks[0].total, None);
        let phase_order = |p: &str| PHASES.iter().position(|x| *x == p).map(|i| i + 1).unwrap_or(0);
        let mut last_rank = 0usize;
        let mut last_done_in_phase = 0u64;
        let mut current_phase = "manifest";
        for tick in ticks.iter() {
            let rank = phase_order(tick.phase);
            if tick.phase != current_phase {
                assert!(rank >= last_rank, "phase went backwards: {current_phase} -> {}", tick.phase);
                current_phase = tick.phase;
                last_rank = rank;
                last_done_in_phase = 0;
            }
            assert!(tick.done >= last_done_in_phase, "done went backwards within {}", tick.phase);
            last_done_in_phase = tick.done;
        }

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    #[tokio::test]
    async fn sync_with_peer_after_a_successful_run_tmp_holds_no_part_files() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"blob bytes for tmp cleanup check";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        // Act
        sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        let tmp_dir = local_root.join("tmp");
        if tmp_dir.is_dir() {
            let leftovers: Vec<_> = std::fs::read_dir(&tmp_dir).unwrap().filter_map(|e| e.ok()).collect();
            assert!(leftovers.is_empty(), "leftover tmp entries: {leftovers:?}");
        }

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    /// R102 finding 2's first adversarial-resume case: the `.part`'s
    /// existing prefix does not actually match the peer's bytes at that
    /// offset (a second attempt whose content differs, or simply a stale
    /// leftover). The resumed append still satisfies `download_item`'s own
    /// `Content-Range` start check (the peer answers `206` starting exactly
    /// where asked), so the corruption is only caught once `install`
    /// hashes the assembled whole file — proving the `.part` is discarded
    /// on that failure, not just on a network-level one, and that the next
    /// run recovers cleanly.
    #[tokio::test]
    async fn sync_with_peer_a_part_file_whose_prefix_bytes_are_wrong_hash_mismatch_discards_it_and_the_next_run_succeeds() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        let item = SyncItem { class: SyncClass::Blob, key: sha.clone(), session_id: None, size_bytes: content.len() as u64 };
        let part_path = tmp_part_path(&local_root, &item);
        std::fs::create_dir_all(part_path.parent().unwrap()).unwrap();
        std::fs::write(&part_path, b"WRONGWRONG").unwrap();

        // Act — first run: the resumed prefix is wrong, so the assembled
        // bytes fail `install`'s hash check.
        let first = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert — the failed pull did not count, and the poisoned `.part`
        // is gone rather than left to be resumed again.
        assert_eq!(first.blobs_transferred, 0);
        assert!(!part_path.exists(), "poisoned .part left behind after an install failure");

        // Act — second run: with no `.part` to misresume, the item
        // restarts from byte 0 and succeeds.
        let second = sync_with_peer(&local_root, &peer, addr, 2000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(second.blobs_transferred, 1);
        assert_eq!(std::fs::read(blob_path(&local_root, &sha)).unwrap(), content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    /// R102 finding 2's second adversarial-resume case: the peer's content
    /// is now *shorter* than the local `.part` (it shrank or was
    /// replaced), so the resumed `Range: bytes=<len>-` request starts past
    /// the peer's own length and the server answers `416` — proving that
    /// case discards the stale `.part` too (not just an `install`
    /// failure), and that the next run recovers.
    #[tokio::test]
    async fn sync_with_peer_a_part_file_longer_than_the_peers_content_416_discards_it_and_the_next_run_succeeds() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let content = b"short";
        let sha = digest(content);
        let path = blob_path(&remote_root, &sha);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let peer = test_peer(token);

        let item = SyncItem { class: SyncClass::Blob, key: sha.clone(), session_id: None, size_bytes: content.len() as u64 };
        let part_path = tmp_part_path(&local_root, &item);
        std::fs::create_dir_all(part_path.parent().unwrap()).unwrap();
        std::fs::write(&part_path, b"way too long a prefix").unwrap();

        // Act — first run: the resumed range starts past the peer's
        // (shorter) content, so the server answers 416.
        let first = sync_with_peer(&local_root, &peer, addr, 1000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(first.blobs_transferred, 0);
        assert!(!part_path.exists(), "stale .part left behind after a 416");

        // Act — second run succeeds cleanly with no `.part` to misresume.
        let second = sync_with_peer(&local_root, &peer, addr, 2000, &no_progress).await.unwrap();

        // Assert
        assert_eq!(second.blobs_transferred, 1);
        assert_eq!(std::fs::read(blob_path(&local_root, &sha)).unwrap(), content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    /// R102 finding 1, document tier: `Track` is a small-document class
    /// (`server.rs`'s `MAX_DOCUMENT_BODY_BYTES`). Calls `download_item`
    /// directly (no `plan_sync` involved) so the oversized response is
    /// requested deliberately rather than relying on a manifest walk to
    /// surface it.
    #[tokio::test]
    async fn download_item_a_document_tier_response_over_the_cap_is_refused_and_nothing_is_written() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let track_id = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";
        let oversized = vec![b'a'; (MAX_DOCUMENT_BODY_BYTES + 1) as usize];
        let track_path = remote_root.join("tracks").join(format!("{track_id}.idl0t"));
        std::fs::create_dir_all(track_path.parent().unwrap()).unwrap();
        std::fs::write(&track_path, &oversized).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let base_url = format!("http://{addr}/idl1/v1");
        let item = SyncItem { class: SyncClass::Track, key: track_id.to_string(), session_id: None, size_bytes: oversized.len() as u64 };
        let client = reqwest::Client::new();

        // Act
        let err = download_item(&client, &base_url, token, &local_root, &item).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(err.message.contains("cap"), "{}", err.message);
        assert!(!tmp_part_path(&local_root, &item).exists());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    /// R102 finding 1, raw-file tier: `Blob` (`server.rs`'s
    /// `MAX_RAW_FILE_BODY_BYTES`) — the direct mirror of `server.rs`'s own
    /// `blob_put_over_the_raw_file_body_cap_is_413_and_nothing_is_written`,
    /// same size, other direction (a `GET` this client is pulling, not a
    /// `PUT` it is accepting).
    #[tokio::test]
    async fn download_item_a_raw_file_tier_response_over_the_cap_is_refused_and_nothing_is_written() {
        // Arrange
        let local_root = temp_data_root();
        let remote_root = temp_data_root();
        let placeholder_digest = "a".repeat(64);
        let oversized = vec![b'a'; (MAX_RAW_FILE_BODY_BYTES + 1) as usize];
        let blob_write_path = blob_path(&remote_root, &placeholder_digest);
        std::fs::create_dir_all(blob_write_path.parent().unwrap()).unwrap();
        std::fs::write(&blob_write_path, &oversized).unwrap();
        let token = "tok-1";
        let server = start_test_server(remote_root.clone(), token).await;
        let addr = server.local_addr();
        let base_url = format!("http://{addr}/idl1/v1");
        let item = SyncItem { class: SyncClass::Blob, key: placeholder_digest.clone(), session_id: None, size_bytes: oversized.len() as u64 };
        let client = reqwest::Client::new();

        // Act
        let err = download_item(&client, &base_url, token, &local_root, &item).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(err.message.contains("cap"), "{}", err.message);
        assert!(!tmp_part_path(&local_root, &item).exists());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&local_root);
        let _ = std::fs::remove_dir_all(&remote_root);
    }

    /// R102 Minor: a peer-supplied `session_id` containing a colon could
    /// otherwise collide with a different item's identity in
    /// `tmp_part_path`'s colon-joined digest input — `item_shape_is_valid`
    /// is the guard `pull_and_install` checks before ever calling
    /// `tmp_part_path` on a peer-sourced item.
    #[test]
    fn item_shape_is_valid_a_session_id_containing_a_colon_is_rejected() {
        // Arrange
        let item = SyncItem { class: SyncClass::Derived, key: "a".repeat(64), session_id: Some("s1:h".to_string()), size_bytes: 0 };

        // Act
        let valid = item_shape_is_valid(&item);

        // Assert
        assert!(!valid);
    }

    #[test]
    fn item_shape_is_valid_a_key_that_is_not_the_expected_shape_is_rejected() {
        // Arrange — a `Workbook` key must be `IdClass::Uuid`-shaped, not an
        // arbitrary string.
        let item = SyncItem { class: SyncClass::Workbook, key: "not-a-uuid".to_string(), session_id: None, size_bytes: 0 };

        // Act & Assert
        assert!(!item_shape_is_valid(&item));
    }

    #[test]
    fn item_shape_is_valid_a_well_formed_blob_item_is_accepted() {
        // Arrange
        let item = SyncItem { class: SyncClass::Blob, key: "a".repeat(64), session_id: None, size_bytes: 0 };

        // Act & Assert
        assert!(item_shape_is_valid(&item));
    }
}
