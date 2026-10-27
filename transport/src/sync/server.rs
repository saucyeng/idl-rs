//! The `axum` sync server (PLAN §3, ruling R88): routes under `/idl1/v1`,
//! bearer auth, hand-parsed `Range` support. Every handler here either
//! answers straight from the filesystem (byte routes) or hands received
//! bytes to `idl-rs`'s `store::sync::apply::install` — no merge logic
//! lives in this file (PLAN §2, this task's brief).
//!
//! **Judgment calls flagged for review**, both because the wire protocol
//! sketch (PLAN §3) names only paths, not the extra manifest-sourced
//! detail `idl-rs::store::sync::apply::InstallContext` needs for two
//! classes (that struct's own doc comment already flags this gap,
//! expecting "a future caller (Task 10/12)" to close it from a manifest it
//! already fetched — this server has no such manifest for an inbound
//! `PUT`):
//! - `PUT .../session/<id>/data.parquet` is installed with no claimed
//!   `(importer_version, seam_correction_version)` pair; `install` itself
//!   refuses this deterministically with a typed `Malformed` error (never a
//!   panic, never a bad write) until Task 10/12 thread that detail through.
//! - `PUT .../workbook/<id>` for a workbook with **no existing local
//!   copy** has nowhere to source `peer_workbook_file_name` from the URL
//!   alone; this handler falls back to the URL's own `<id>` as the file
//!   name (a workbook that already exists locally is unaffected — its
//!   real, current file name is looked up from the manifest and this
//!   fallback never triggers).

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::extract::{Path as AxumPath, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;

use idl_rs::store::blob::blob_path;
use idl_rs::store::sync::apply::{install, install_blob_file, InstallContext};
use idl_rs::store::sync::diff::{SyncClass, SyncItem};
use idl_rs::store::sync::ids::{is_valid_id, safe_join, IdClass};
use idl_rs::store::sync::manifest::build_manifest;

use crate::error::{TransportError, TransportErrorKind};

use super::pairing::PairingState;
use super::range::{parse_range, RangeErrorKind};
use super::wire::{PairRequest, PairResponse, Peer, PROTOCOL_VERSION};

/// Everything a route handler needs, shared behind `Arc`/`Mutex` so every
/// `axum` handler (which must be `Clone`) sees the one live state.
#[derive(Clone)]
struct ServerState {
    data_root: PathBuf,
    peer_id: String,
    name: String,
    pairing: Arc<Mutex<PairingState>>,
    peers: Arc<Mutex<Vec<Peer>>>,
}

/// Configuration for [`SyncServer::start`].
pub struct SyncServerConfig {
    /// Root of `<data>`; every served path is resolved under it.
    pub data_root: PathBuf,
    /// `0` asks the OS for an ephemeral port — how the tests bind.
    pub port: u16,
    /// The LAN address to bind. Production binds `0.0.0.0`; tests bind
    /// `127.0.0.1` (this task's brief, "Key logic").
    pub bind_addr: std::net::IpAddr,
    pub peer_id: String,
    pub name: String,
}

/// A running sync server. Dropping it stops serving (the background task
/// is aborted so no handle outlives its owner).
pub struct SyncServer {
    local_addr: SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl SyncServer {
    /// Binds and starts serving on the caller's runtime (this crate never
    /// spawns its own, CLAUDE.md §2 / this task's brief).
    pub async fn start(
        config: SyncServerConfig,
        pairing: Arc<Mutex<PairingState>>,
        peers: Arc<Mutex<Vec<Peer>>>,
    ) -> Result<Self, TransportError> {
        let state = ServerState { data_root: config.data_root, peer_id: config.peer_id, name: config.name, pairing, peers };

        let app = build_router(state);

        let listener = tokio::net::TcpListener::bind((config.bind_addr, config.port))
            .await
            .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("binding sync server: {e}")))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("reading bound address: {e}")))?;

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let join = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        Ok(Self { local_addr, shutdown_tx: Some(shutdown_tx), join: Some(join) })
    }

    /// The bound address — the real port when `port` was `0`.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Signals graceful shutdown and waits for the serving task to finish.
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl Drop for SyncServer {
    fn drop(&mut self) {
        // Best-effort: a caller that drops without awaiting `shutdown`
        // still stops serving — the signal is sent, the task is not
        // awaited (a `Drop` impl cannot `.await`), matching "dropping it
        // stops serving" (this task's brief interface doc comment).
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

fn build_router(state: ServerState) -> Router {
    Router::new()
        .route("/idl1/v1/pair", post(handle_pair))
        .route("/idl1/v1/manifest", get(handle_manifest))
        .route("/idl1/v1/library", get(handle_library))
        .route("/idl1/v1/blob/{hash}", get(handle_blob_get).put(handle_blob_put))
        .route("/idl1/v1/session/{id}/data.parquet", get(handle_data_parquet_get).put(handle_data_parquet_put))
        .route("/idl1/v1/session/{id}/session.json", get(handle_session_json_get).put(handle_session_json_put))
        // `matchit` (axum's router) allows only one `{param}` per path
        // segment — `{hash}.parquet` in one segment is rejected outright
        // (a router-construction panic, caught nowhere near a request), so
        // the whole `<sha256>.parquet` segment is captured and the
        // handlers strip the suffix themselves.
        .route(
            "/idl1/v1/session/{id}/derived/{file_name}",
            get(handle_derived_get).put(handle_derived_put),
        )
        .route("/idl1/v1/workbook/{id}", get(handle_workbook_get).put(handle_workbook_put))
        .route("/idl1/v1/track/{id}", get(handle_track_get).put(handle_track_put))
        .route("/idl1/v1/profile/{id}", get(handle_profile_get).put(handle_profile_put))
        .route("/idl1/v1/bike/{id}", get(handle_bike_get).put(handle_bike_put))
        .fallback(handle_not_found)
        // `route_layer`, not `layer`: auth wraps only the named routes, so an
        // unmatched path still falls through to the plain `404` fallback
        // above rather than a `401` (this task's brief: "an unknown route —
        // 404"). The extractor-tuple turbofish is required here: with a
        // named top-level `async fn` (rather than a closure) passed to
        // `from_fn_with_state`, rustc's obligation solver cannot pick the
        // one matching `FromFn<..., T>: Service<Request>` impl among the
        // macro-generated arities on its own (`T` never appears in any
        // concrete type it can unify against) and reports the whole bound
        // as unsatisfied; naming `T` explicitly resolves it deterministically.
        .route_layer(middleware::from_fn_with_state::<_, ServerState, (State<ServerState>, Request)>(
            state.clone(),
            auth_layer,
        ))
        .with_state(state)
}

async fn handle_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

/// Rejects every request but `POST /idl1/v1/pair` (PLAN §3) without a
/// bearer token matching a known peer. Token comparison is constant-time
/// (this task's brief, "Key logic"). No body detail on rejection.
async fn auth_layer(State(state): State<ServerState>, request: Request, next: Next) -> Response {
    if request.uri().path() == "/idl1/v1/pair" {
        return next.run(request).await;
    }

    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let Some(token) = token else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    let authorized = {
        let known = state.peers.lock().unwrap_or_else(|e| e.into_inner());
        known.iter().any(|p| constant_time_eq(&p.token, token))
    };

    if authorized {
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

/// Compares `a` and `b` without short-circuiting on length or the first
/// differing byte's position among the compared bytes (this task's brief:
/// "token comparison is constant-time").
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let len_diff = (a.len() != b.len()) as u8;
    let n = a.len().max(b.len());
    let mut diff = len_diff;
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

/// Milliseconds since the Unix epoch, read once per call — this server
/// owns no clock beyond this (this task's brief, "Key logic").
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

async fn handle_pair(State(state): State<ServerState>, body: axum::body::Bytes) -> Response {
    let Ok(req) = serde_json::from_slice::<PairRequest>(&body) else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    if let Err(e) = super::pairing::check_protocol_version(&req) {
        return (StatusCode::CONFLICT, e.message).into_response();
    }

    let redeemed = {
        let mut pairing = state.pairing.lock().unwrap_or_else(|e| e.into_inner());
        pairing.redeem(&req.code, now_ms())
    };
    if let Err(e) = redeemed {
        return (StatusCode::UNAUTHORIZED, e.message).into_response();
    }

    let token = uuid::Uuid::new_v4().to_string();
    {
        let mut peers = state.peers.lock().unwrap_or_else(|e| e.into_inner());
        peers.push(Peer {
            peer_id: req.peer_id.clone(),
            name: req.name.clone(),
            token: token.clone(),
            protocol_version: req.protocol_version,
            paired_at_ms: now_ms(),
            auto_sync: crate::sync::wire::AutoSync::Off,
        });
    }

    let response =
        PairResponse { peer_id: state.peer_id.clone(), name: state.name.clone(), token, protocol_version: PROTOCOL_VERSION };
    axum::Json(response).into_response()
}

async fn handle_manifest(State(state): State<ServerState>) -> Response {
    match build_manifest(&state.data_root, now_ms()) {
        Ok((manifest, _skipped)) => axum::Json(manifest).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.message).into_response(),
    }
}

/// The body of a `503` from `GET /idl1/v1/library`: the serving library's
/// catalog is not ready (library tiers spec 4a). Typed so a client can tell
/// "still indexing" from any other failure by `error`, not by prose.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LibraryUnavailableBody {
    /// Always `"catalog_unavailable"`.
    pub error: String,
    /// Human-readable detail from the catalog open.
    pub message: String,
}

/// `GET /idl1/v1/library` (library tiers spec 4a): the serving library's
/// session and workbook summary, from its catalog, read-only. `503` with a
/// [`LibraryUnavailableBody`] when the catalog is not ready.
async fn handle_library(State(state): State<ServerState>) -> Response {
    let root = state.data_root.clone();
    match tokio::task::spawn_blocking(move || idl_rs::store::library_summary::library_summary(&root)).await {
        Ok(Ok(summary)) => axum::Json(summary).into_response(),
        Ok(Err(e)) => {
            let body = LibraryUnavailableBody { error: "catalog_unavailable".to_string(), message: e.message };
            (StatusCode::SERVICE_UNAVAILABLE, axum::Json(body)).into_response()
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

/// `true` for a 64-hex-character sha256 digest, lowercase or uppercase
/// (this task's brief: "64 hex for a hash").
fn is_valid_hash(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Serves `path`'s bytes, honouring a single `Range` header (this task's
/// brief: `Accept-Ranges: bytes`; `206`/`Content-Range` for a valid range;
/// `416` for an unsatisfiable one). `404` if `path` is not a file — never
/// leaks whether a directory exists at that path vs. nothing at all.
/// Streamed from disk (lead ruling E1, 2026-10-18): a video blob can be
/// gigabytes, so the file is never read whole.
async fn serve_file(path: &Path, headers: &HeaderMap) -> Response {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Ok(meta) = file.metadata().await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if !meta.is_file() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let len = meta.len();

    let range_header = headers.get(header::RANGE).and_then(|v| v.to_str().ok());
    match parse_range(range_header, len) {
        Ok(None) => {
            let mut response = Response::new(Body::from_stream(reader_chunks(file)));
            response.headers_mut().insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
            response.headers_mut().insert(header::CONTENT_LENGTH, len.into());
            response
        }
        Ok(Some((start, end))) => {
            if file.seek(std::io::SeekFrom::Start(start)).await.is_err() {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            let window = file.take(end - start + 1);
            let mut response = Response::new(Body::from_stream(reader_chunks(window)));
            response.headers_mut().insert(header::CONTENT_LENGTH, (end - start + 1).into());
            *response.status_mut() = StatusCode::PARTIAL_CONTENT;
            response.headers_mut().insert(header::ACCEPT_RANGES, "bytes".parse().unwrap());
            response
                .headers_mut()
                .insert(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}").parse().unwrap());
            response
        }
        Err(e) if e.kind == RangeErrorKind::Unsatisfiable || e.kind == RangeErrorKind::Multiple => {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::RANGE_NOT_SATISFIABLE;
            response.headers_mut().insert(header::CONTENT_RANGE, format!("bytes */{len}").parse().unwrap());
            response
        }
        Err(_) => StatusCode::BAD_REQUEST.into_response(),
    }
}

/// An async reader (a file, or a ranged window of one) as a stream of
/// chunks of at most 256 KiB, for a response body that must not hold the
/// file in memory.
fn reader_chunks<R: tokio::io::AsyncRead + Unpin + Send + 'static>(reader: R) -> impl futures::Stream<Item = std::io::Result<Vec<u8>>> + Send + 'static {
    use tokio::io::AsyncReadExt;
    futures::stream::unfold(Some(reader), |state| async move {
        let mut reader = state?;
        let mut buf = vec![0u8; 256 * 1024];
        match reader.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok(buf), Some(reader)))
            }
            Err(e) => Some((Err(e), None)),
        }
    })
}

/// Generous cap for a small syncable document body — `session.json`,
/// `.idl1wb` workbook, `.idl0t` track, `.idl0p` profile — all plaintext
/// JSON/markdown-ish, never more than a few hundred KB in real use.
/// 16 MiB leaves orders-of-magnitude headroom (R100).
const MAX_DOCUMENT_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Generous cap for a raw/large syncable body — a content-addressed blob
/// (`store::blob`'s doc comment: "raw source files", i.e. a device's raw
/// capture), a cached `derived` Parquet, or `data.parquet` itself. IDL0_SPEC
/// notes a device's SD-card free-space threshold of ~200 MB at peak
/// logging rate, so a single legitimate session's raw capture can
/// plausibly be that large; 512 MiB leaves generous headroom above it
/// while still bounding memory (R100).
const MAX_RAW_FILE_BODY_BYTES: usize = 512 * 1024 * 1024;

/// Reads the full request body, refusing anything over `max_bytes` with
/// `413` — R100: `to_bytes(body, usize::MAX)` used to impose no real limit
/// at all, despite this function's own old doc comment claiming otherwise.
/// Any other body-read failure (a malformed chunked stream, a client
/// disconnect mid-upload) is `400`.
///
/// Distinguishing "the limit was hit" from "the stream broke some other
/// way": `to_bytes`'s own doc comment (`axum` 0.8.9, `body/mod.rs`) shows
/// the canonical way to tell — `std::error::Error::source` on the returned
/// `axum::Error` is a `http_body_util::LengthLimitError` when the limit was
/// exceeded. Recognised here by its fixed `Display` text (`"length limit
/// exceeded"`, `http-body-util` 0.1.5 — pinned by `axum`'s own dependency
/// tree, already in this workspace's `Cargo.lock`) rather than by naming
/// the type directly, so this file needs no new direct dependency (and no
/// `Cargo.lock` line of its own) just to recognise an error it never
/// constructs or matches structurally.
async fn read_body(body: Body, max_bytes: usize) -> Result<Vec<u8>, Response> {
    match to_bytes(body, max_bytes).await {
        Ok(bytes) => Ok(bytes.to_vec()),
        Err(err) => {
            let hit_the_limit = std::error::Error::source(&err).is_some_and(|source| source.to_string() == "length limit exceeded");
            if hit_the_limit {
                Err(StatusCode::PAYLOAD_TOO_LARGE.into_response())
            } else {
                Err(StatusCode::BAD_REQUEST.into_response())
            }
        }
    }
}

/// Runs `install` and turns its outcome/error into a response: `200` on
/// any [`idl_rs::store::sync::apply::InstallOutcome`], the typed error's
/// message with `422` otherwise (never a panic on malformed input, this
/// task's brief).
fn install_response(
    data_root: &Path,
    item: &SyncItem,
    bytes: &[u8],
    peer_name: &str,
    ctx: &InstallContext,
) -> Response {
    match install(data_root, item, bytes, peer_name, now_ms(), ctx) {
        Ok(_outcome) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, e.message).into_response(),
    }
}

async fn handle_blob_get(State(state): State<ServerState>, AxumPath(hash): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_hash(&hash) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let path = blob_path(&state.data_root, &hash);
    serve_file(&path, &headers).await
}

/// What this server does with a blob `PUT`, decided from its own merged
/// `session.json` files before any byte is read (C4 §6, lead ruling E2):
/// `None` declines it (`409 {"skipped": "video"}`); `Some(cap)` accepts up
/// to `cap` bytes. A linked video is declined on a device that does not
/// hold video, and anywhere only tombstones reference it; otherwise it has
/// no cap. Every other blob keeps [`MAX_RAW_FILE_BODY_BYTES`].
fn blob_put_cap(videos: &BTreeSet<String>, live: &BTreeSet<String>, hash: &str, holds_video: bool) -> Option<u64> {
    if !videos.contains(hash) {
        return Some(MAX_RAW_FILE_BODY_BYTES as u64);
    }
    (holds_video && live.contains(hash)).then_some(u64::MAX)
}

/// Streams a request body into `part`, refusing past `cap` bytes with `413`
/// (and removing `part`), or `400` on a broken stream. Never holds the body
/// in memory (lead ruling E1).
async fn receive_body_to_part(body: Body, part: &Path, cap: u64) -> Result<(), Response> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    if let Some(parent) = part.parent() {
        if tokio::fs::create_dir_all(parent).await.is_err() {
            return Err(StatusCode::INTERNAL_SERVER_ERROR.into_response());
        }
    }
    let Ok(mut file) = tokio::fs::File::create(part).await else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR.into_response());
    };
    let mut stream = body.into_data_stream();
    let mut total = 0u64;
    let failure = loop {
        let Some(chunk) = stream.next().await else { break None };
        let Ok(chunk) = chunk else { break Some(StatusCode::BAD_REQUEST) };
        total += chunk.len() as u64;
        if total > cap {
            break Some(StatusCode::PAYLOAD_TOO_LARGE);
        }
        if file.write_all(&chunk).await.is_err() {
            break Some(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let flushed = file.flush().await.is_ok();
    drop(file);
    match (failure, flushed) {
        (None, true) => Ok(()),
        (failure, _) => {
            let _ = tokio::fs::remove_file(part).await;
            Err(failure.unwrap_or(StatusCode::INTERNAL_SERVER_ERROR).into_response())
        }
    }
}

async fn handle_blob_put(State(state): State<ServerState>, AxumPath(hash): AxumPath<String>, request: Request) -> Response {
    if !is_valid_hash(&hash) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let root = state.data_root.clone();
    let linked = tokio::task::spawn_blocking(move || (idl_rs::store::video::video_blobs(&root), idl_rs::store::video::live_video_blobs(&root))).await;
    let Ok((videos, live)) = linked else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let Some(cap) = blob_put_cap(&videos, &live, &hash, super::client::holds_video()) else {
        return (StatusCode::CONFLICT, [(header::CONTENT_TYPE, "application/json")], SKIPPED_VIDEO_BODY).into_response();
    };
    let part = state.data_root.join("tmp").join(format!("put-{}.part", uuid::Uuid::new_v4()));
    if let Err(response) = receive_body_to_part(request.into_body(), &part, cap).await {
        return response;
    }
    let (root, key, file) = (state.data_root.clone(), hash.clone(), part.clone());
    let installed = tokio::task::spawn_blocking(move || install_blob_file(&root, &key, &file)).await;
    let _ = tokio::fs::remove_file(&part).await;
    match installed {
        Ok(Ok(_)) => StatusCode::OK.into_response(),
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, e.message).into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

/// C4 §6's answer to a declined video `PUT`.
const SKIPPED_VIDEO_BODY: &str = "{\"skipped\":\"video\"}";

/// Splits a `derived/<file_name>` route segment into its sha256 hash,
/// requiring the literal `.parquet` suffix `matchit` cannot itself capture
/// as a second parameter in the same segment (see `build_router`'s
/// comment). `None` for anything not shaped `<64 hex>.parquet`.
fn derived_hash_from_file_name(file_name: &str) -> Option<&str> {
    let hash = file_name.strip_suffix(".parquet")?;
    is_valid_hash(hash).then_some(hash)
}

async fn handle_derived_get(
    State(state): State<ServerState>,
    AxumPath((id, file_name)): AxumPath<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let Some(hash) = derived_hash_from_file_name(&file_name) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["sessions", &id, "derived", &format!("{hash}.parquet")]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_derived_put(
    State(state): State<ServerState>,
    AxumPath((id, file_name)): AxumPath<(String, String)>,
    request: Request,
) -> Response {
    let Some(hash) = derived_hash_from_file_name(&file_name) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let hash = hash.to_string();
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_RAW_FILE_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::Derived, key: hash, session_id: Some(id), size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

async fn handle_data_parquet_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["sessions", &id, "data.parquet"]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

/// See this module's doc comment: no claimed version pair reaches this
/// route from the URL alone, so `install` refuses with its own typed
/// error until a future task threads manifest-sourced detail through.
async fn handle_data_parquet_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_RAW_FILE_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::DataParquet, key: id.clone(), session_id: Some(id), size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

async fn handle_session_json_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["sessions", &id, "session.json"]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_session_json_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !is_valid_id(&id, IdClass::Session) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_DOCUMENT_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::SessionJson, key: id.clone(), session_id: Some(id), size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

/// Resolves `workbook_id` to its on-disk `file_name` via a fresh
/// `build_manifest` walk (the manifest is the only place this crate can
/// learn a workbook's current file name from its id, C4 §6: "id wins").
fn resolve_workbook_file_name(data_root: &Path, workbook_id: &str) -> Option<String> {
    let (manifest, _skipped) = build_manifest(data_root, now_ms()).ok()?;
    manifest.workbooks.into_iter().find(|w| w.workbook_id == workbook_id).map(|w| w.file_name)
}

async fn handle_workbook_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(file_name) = resolve_workbook_file_name(&state.data_root, &id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(path) = safe_join(&state.data_root, &["workbooks", &format!("{file_name}.idl1wb")]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_workbook_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_DOCUMENT_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    // See this module's doc comment: an existing local workbook's real
    // file name is looked up from the manifest; a brand-new workbook (no
    // local copy yet) falls back to the URL's own id as its file name.
    let file_name = resolve_workbook_file_name(&state.data_root, &id).unwrap_or_else(|| id.clone());
    let ctx = InstallContext { peer_workbook_file_name: Some(file_name), ..Default::default() };
    let item = SyncItem { class: SyncClass::Workbook, key: id, session_id: None, size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &ctx)
}

async fn handle_track_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["tracks", &format!("{id}.idl0t")]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_track_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_DOCUMENT_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::Track, key: id, session_id: None, size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

async fn handle_profile_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["profiles", &format!("{id}.idl0p")]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_profile_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !is_valid_id(&id, IdClass::Uuid) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_DOCUMENT_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::Profile, key: id, session_id: None, size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

async fn handle_bike_get(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, headers: HeaderMap) -> Response {
    if !idl_rs::store::bikelog::is_valid_id(&id) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(path) = safe_join(&state.data_root, &["bikes", &format!("{id}.json")]) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    serve_file(&path, &headers).await
}

async fn handle_bike_put(State(state): State<ServerState>, AxumPath(id): AxumPath<String>, request: Request) -> Response {
    if !idl_rs::store::bikelog::is_valid_id(&id) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let bytes = match read_body(request.into_body(), MAX_DOCUMENT_BODY_BYTES).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let item = SyncItem { class: SyncClass::Bike, key: id, session_id: None, size_bytes: bytes.len() as u64 };
    install_response(&state.data_root, &item, &bytes, &state.name, &InstallContext::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::store::sync::manifest::Manifest;
    use std::net::Ipv4Addr;

    async fn start_test_server(data_root: PathBuf, peers: Vec<Peer>) -> (SyncServer, Arc<Mutex<Vec<Peer>>>) {
        let peers = Arc::new(Mutex::new(peers));
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let config = SyncServerConfig {
            data_root,
            port: 0,
            bind_addr: Ipv4Addr::LOCALHOST.into(),
            peer_id: "server-peer".to_string(),
            name: "Server".to_string(),
        };
        let server = SyncServer::start(config, pairing, peers.clone()).await.unwrap();
        (server, peers)
    }

    fn temp_data_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-transport-server-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base_url(server: &SyncServer) -> String {
        format!("http://{}/idl1/v1", server.local_addr())
    }

    #[tokio::test]
    async fn manifest_with_a_valid_token_returns_the_c4_document() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok-1".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/manifest", base_url(&server))).bearer_auth(&token).send().await.unwrap();
        let status = response.status();
        let manifest: Manifest = response.json().await.unwrap();

        // Assert
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(manifest.schema_version, 1);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn manifest_with_no_token_is_401() {
        // Arrange
        let data_root = temp_data_root();
        let (server, _peers) = start_test_server(data_root.clone(), vec![]).await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/manifest", base_url(&server))).send().await.unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn manifest_with_a_wrong_token_is_401() {
        // Arrange
        let data_root = temp_data_root();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: "right".to_string(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/manifest", base_url(&server))).bearer_auth("wrong").send().await.unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_get_whole_body_matches_the_file() {
        // Arrange
        let data_root = temp_data_root();
        let content = b"hello sync world";
        let digest = idl_rs::store::atomic::sha256_hex(content);
        let path = blob_path(&data_root, &digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/blob/{digest}", base_url(&server))).bearer_auth(&token).send().await.unwrap();
        let status = response.status();
        let body = response.bytes().await.unwrap();

        // Assert
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(&body[..], content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_put_then_get_a_body_larger_than_the_stream_chunk_round_trips() {
        // Arrange — 1 MiB + 3 bytes: four 256 KiB chunks and a tail, both
        // ways (lead ruling E1). Small enough to hold in a test.
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();
        let content: Vec<u8> = (0..(1024 * 1024 + 3)).map(|i| (i % 251) as u8).collect();
        let digest = idl_rs::store::atomic::sha256_hex(&content);

        // Act
        let put = client.put(format!("{}/blob/{digest}", base_url(&server))).bearer_auth(&token).body(content.clone()).send().await.unwrap();
        let whole = client.get(format!("{}/blob/{digest}", base_url(&server))).bearer_auth(&token).send().await.unwrap();
        let whole_len = whole.headers().get("content-length").unwrap().to_str().unwrap().to_string();
        let whole_body = whole.bytes().await.unwrap();
        let range = client
            .get(format!("{}/blob/{digest}", base_url(&server)))
            .bearer_auth(&token)
            .header("Range", "bytes=262140-262150")
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        // Assert
        assert_eq!(put.status(), reqwest::StatusCode::OK);
        assert_eq!(std::fs::read(blob_path(&data_root, &digest)).unwrap(), content);
        assert_eq!(whole_len, content.len().to_string());
        assert_eq!(&whole_body[..], &content[..]);
        assert_eq!(&range[..], &content[262_140..=262_150], "a range across a chunk boundary");
        let leftovers = std::fs::read_dir(data_root.join("tmp")).map(|d| d.count()).unwrap_or(0);
        assert_eq!(leftovers, 0, "no put-*.part left behind");

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn blob_put_cap_a_log_keeps_the_raw_cap_a_live_video_has_none_and_a_video_can_be_declined() {
        // Arrange
        let videos: BTreeSet<String> = ["v".repeat(64), "t".repeat(64)].into_iter().collect();
        let live: BTreeSet<String> = ["v".repeat(64)].into_iter().collect();

        // Act + Assert
        assert_eq!(blob_put_cap(&videos, &live, &"l".repeat(64), true), Some(MAX_RAW_FILE_BODY_BYTES as u64));
        assert_eq!(blob_put_cap(&videos, &live, &"l".repeat(64), false), Some(MAX_RAW_FILE_BODY_BYTES as u64));
        assert_eq!(blob_put_cap(&videos, &live, &"v".repeat(64), true), Some(u64::MAX));
        assert_eq!(blob_put_cap(&videos, &live, &"v".repeat(64), false), None, "a phone declines video");
        assert_eq!(blob_put_cap(&videos, &live, &"t".repeat(64), true), None, "only tombstones reference it");
    }

    #[tokio::test]
    async fn blob_get_with_range_returns_206_and_content_range() {
        // Arrange
        let data_root = temp_data_root();
        let content = b"0123456789";
        let digest = idl_rs::store::atomic::sha256_hex(content);
        let path = blob_path(&data_root, &digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client
            .get(format!("{}/blob/{digest}", base_url(&server)))
            .bearer_auth(&token)
            .header("Range", "bytes=4-7")
            .send()
            .await
            .unwrap();
        let status = response.status();
        let content_range = response.headers().get("content-range").unwrap().to_str().unwrap().to_string();
        let body = response.bytes().await.unwrap();

        // Assert
        assert_eq!(status, reqwest::StatusCode::PARTIAL_CONTENT);
        assert_eq!(content_range, format!("bytes 4-7/{}", content.len()));
        assert_eq!(&body[..], b"4567");

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_get_unsatisfiable_range_is_416() {
        // Arrange
        let data_root = temp_data_root();
        let content = b"short";
        let digest = idl_rs::store::atomic::sha256_hex(content);
        let path = blob_path(&data_root, &digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client
            .get(format!("{}/blob/{digest}", base_url(&server)))
            .bearer_auth(&token)
            .header("Range", "bytes=100-200")
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::RANGE_NOT_SATISFIABLE);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_get_path_traversal_via_catalog_sqlite_is_404_and_never_opened() {
        // Arrange
        let data_root = temp_data_root();
        std::fs::write(data_root.join("catalog.sqlite"), b"secret").unwrap();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act: a hash-shaped segment that tries to escape via encoded traversal
        // is rejected by the router itself (axum normalises `..` segments away
        // from a single path parameter) or by this handler's own hash-shape
        // check; either way the file is never read.
        let response = client
            .get(format!("{}/blob/..%2F..%2Fcatalog.sqlite", base_url(&server)))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn workbook_get_id_with_path_separator_is_404() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client
            .get(format!("{}/workbook/..%2f..%2fetc", base_url(&server)))
            .bearer_auth(&token)
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_put_twice_with_same_body_both_200_same_state() {
        // Arrange
        let data_root = temp_data_root();
        let content = b"idempotent bytes";
        let digest = idl_rs::store::atomic::sha256_hex(content);
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("{}/blob/{digest}", base_url(&server));

        // Act
        let first = client.put(&url).bearer_auth(&token).body(content.to_vec()).send().await.unwrap();
        let second = client.put(&url).bearer_auth(&token).body(content.to_vec()).send().await.unwrap();

        // Assert
        assert_eq!(first.status(), reqwest::StatusCode::OK);
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        assert_eq!(std::fs::read(blob_path(&data_root, &digest)).unwrap(), content);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn blob_put_wrong_hash_is_rejected_never_written() {
        // Arrange
        let data_root = temp_data_root();
        let content = b"actual bytes";
        let wrong_hash = idl_rs::store::atomic::sha256_hex(b"not these bytes");
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response =
            client.put(format!("{}/blob/{wrong_hash}", base_url(&server))).bearer_auth(&token).body(content.to_vec()).send().await.unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!blob_path(&data_root, &wrong_hash).is_file());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn pair_with_the_current_code_returns_a_token_wrong_code_is_refused() {
        // Arrange
        let data_root = temp_data_root();
        // `handle_pair` redeems against real wall-clock time (`now_ms()`,
        // this task's brief: "the server does not own a clock beyond
        // [the manifest] call" — redemption is the one other place it
        // reads one), so the offer must be minted against it too, not a
        // fixed `0` that would already read as expired.
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let offer = pairing.lock().unwrap().offer(now_ms());
        let peers = Arc::new(Mutex::new(Vec::new()));
        let config = SyncServerConfig {
            data_root: data_root.clone(),
            port: 0,
            bind_addr: Ipv4Addr::LOCALHOST.into(),
            peer_id: "server-peer".to_string(),
            name: "Server".to_string(),
        };
        let server = SyncServer::start(config, pairing, peers).await.unwrap();
        let client = reqwest::Client::new();
        let request = PairRequest { code: offer.code.clone(), peer_id: "client".to_string(), name: "Client".to_string(), protocol_version: PROTOCOL_VERSION };

        // Act
        let wrong = client
            .post(format!("{}/pair", base_url(&server)))
            .json(&PairRequest { code: "000000".to_string(), ..request.clone() })
            .send()
            .await
            .unwrap();
        let wrong_status = wrong.status();

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);

        // The wrong code must not consume the real offer's slot in a way
        // that changes this assertion's meaning; re-check status only.
        assert_eq!(wrong_status, reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn pair_with_the_current_code_returns_a_token() {
        // Arrange
        let data_root = temp_data_root();
        // `handle_pair` redeems against real wall-clock time (`now_ms()`,
        // this task's brief: "the server does not own a clock beyond
        // [the manifest] call" — redemption is the one other place it
        // reads one), so the offer must be minted against it too, not a
        // fixed `0` that would already read as expired.
        let pairing = Arc::new(Mutex::new(PairingState::default()));
        let offer = pairing.lock().unwrap().offer(now_ms());
        let peers = Arc::new(Mutex::new(Vec::new()));
        let config = SyncServerConfig {
            data_root: data_root.clone(),
            port: 0,
            bind_addr: Ipv4Addr::LOCALHOST.into(),
            peer_id: "server-peer".to_string(),
            name: "Server".to_string(),
        };
        let server = SyncServer::start(config, pairing, peers).await.unwrap();
        let client = reqwest::Client::new();
        let request = PairRequest { code: offer.code, peer_id: "client".to_string(), name: "Client".to_string(), protocol_version: PROTOCOL_VERSION };

        // Act
        let response = client.post(format!("{}/pair", base_url(&server))).json(&request).send().await.unwrap();
        let status = response.status();
        let parsed: PairResponse = response.json().await.unwrap();

        // Assert
        assert_eq!(status, reqwest::StatusCode::OK);
        assert!(!parsed.token.is_empty());
        assert_eq!(parsed.peer_id, "server-peer");

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[tokio::test]
    async fn an_unknown_route_is_404() {
        // Arrange
        let data_root = temp_data_root();
        let (server, _peers) = start_test_server(data_root.clone(), vec![]).await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/nonexistent", base_url(&server))).send().await.unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    /// Regression test for the exact bug the `.layer` vs. `.route_layer`
    /// type error was hiding: with `.layer`, auth wraps the fallback too,
    /// so an unmatched path with no token would answer `401` instead of
    /// `404`. `.route_layer` wraps only the named routes.
    #[tokio::test]
    async fn unknown_route_is_404_without_a_token_while_a_known_route_is_401() {
        // Arrange
        let data_root = temp_data_root();
        let (server, _peers) = start_test_server(data_root.clone(), vec![]).await;
        let client = reqwest::Client::new();

        // Act
        let unknown = client.get(format!("{}/nonexistent", base_url(&server))).send().await.unwrap();
        let known_no_auth = client.get(format!("{}/manifest", base_url(&server))).send().await.unwrap();

        // Assert
        assert_eq!(unknown.status(), reqwest::StatusCode::NOT_FOUND);
        assert_eq!(known_no_auth.status(), reqwest::StatusCode::UNAUTHORIZED);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    /// R100 regression: `review-task8`'s Critical. A colon-prefixed segment
    /// (`C:evil`) is a legal, unencoded URL path segment that used to reach
    /// `PathBuf::join` unmodified — on Windows, joining a component with a
    /// drive prefix but no root **discards the whole base path**, landing
    /// outside `data_root` entirely. `is_valid_id`'s `IdClass::Uuid` shape
    /// now rejects it before any path is built.
    #[tokio::test]
    async fn track_get_windows_drive_relative_id_is_404_and_nothing_outside_data_root_is_read() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client.get(format!("{}/track/C:evil", base_url(&server))).bearer_auth(&token).send().await.unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    /// R100 regression, the brand-new-workbook write case
    /// `review-task8` named explicitly: `handle_workbook_put`'s fallback
    /// (`resolve_workbook_file_name(...).unwrap_or_else(|| id.clone())`)
    /// used to thread an unvalidated `id` straight into a file name when no
    /// local workbook existed yet — an authenticated peer could write a
    /// file outside `data_root/workbooks/`. The id-shape check now runs
    /// before `install` is ever called.
    #[tokio::test]
    async fn workbook_put_brand_new_windows_drive_relative_id_is_404_and_nothing_is_written() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();

        // Act
        let response = client
            .put(format!("{}/workbook/C:evil", base_url(&server)))
            .bearer_auth(&token)
            .body(b"---\nid: C:evil\nname: Evil\n---\n".to_vec())
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
        assert!(!data_root.join("workbooks").exists() || std::fs::read_dir(data_root.join("workbooks")).unwrap().next().is_none());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    /// R100 regression (amended: per-class caps, not one shared cap):
    /// `review-task8`'s Important — `read_body` used to call
    /// `to_bytes(body, usize::MAX)`, an explicit unlimited cap despite its
    /// own doc comment's claim otherwise. One test per cap tier — this one
    /// covers `MAX_DOCUMENT_BODY_BYTES` via `session.json` PUT; the raw-file
    /// tier (`MAX_RAW_FILE_BODY_BYTES`) is covered by `blob_put_...` below.
    /// Exactly one byte over the cap is `413`, and nothing is written.
    #[tokio::test]
    async fn session_json_put_over_the_document_body_cap_is_413_and_nothing_is_written() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();
        let session_id = "0123456789abcdef";
        let oversized = vec![b'a'; MAX_DOCUMENT_BODY_BYTES + 1];

        // Act
        let response = client
            .put(format!("{}/session/{session_id}/session.json", base_url(&server)))
            .bearer_auth(&token)
            .body(oversized)
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
        assert!(!data_root.join("sessions").join(session_id).join("session.json").exists());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }

    /// R100 amendment's second cap-tier test: `MAX_RAW_FILE_BODY_BYTES`,
    /// exercised via `blob` PUT (this crate's largest legitimate body
    /// class — a device's raw source file, `store::blob`'s own doc
    /// comment). The digest in the URL never needs to match the (rejected)
    /// body's real hash — `read_body`'s cap check runs before `install`
    /// ever sees the bytes, so a shape-valid placeholder is enough to
    /// reach the route.
    #[tokio::test]
    async fn blob_put_over_the_raw_file_body_cap_is_413_and_nothing_is_written() {
        // Arrange
        let data_root = temp_data_root();
        let token = "tok".to_string();
        let (server, _peers) = start_test_server(
            data_root.clone(),
            vec![Peer { peer_id: "p1".to_string(), name: "Peer".to_string(), token: token.clone(), protocol_version: 1, paired_at_ms: 0, auto_sync: crate::sync::wire::AutoSync::Off }],
        )
        .await;
        let client = reqwest::Client::new();
        let placeholder_digest = "a".repeat(64);
        let oversized = vec![b'a'; MAX_RAW_FILE_BODY_BYTES + 1];

        // Act
        let response = client
            .put(format!("{}/blob/{placeholder_digest}", base_url(&server)))
            .bearer_auth(&token)
            .body(oversized)
            .send()
            .await
            .unwrap();

        // Assert
        assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
        assert!(!blob_path(&data_root, &placeholder_digest).is_file());

        server.shutdown().await;
        let _ = std::fs::remove_dir_all(&data_root);
    }
}
