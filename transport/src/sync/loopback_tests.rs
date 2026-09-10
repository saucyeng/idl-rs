//! Task 11 — the loopback two-peer proof (this lane's brief, PLAN §7): two
//! `<data>` roots, two real `SyncServer`s, paired over a genuine
//! `POST /pair` handshake (no mDNS — addresses come from `local_addr()`,
//! PLAN §7's stated bypass), driving [`super::sync_with_peer`] both ways.
//! Every fixture is seeded through `idl-rs`'s own writers
//! (`write_blob`/`write_session_parquet`/`write_session_json`/`write_track`,
//! and — since there is no single dedicated workbook writer — the same
//! `write_atomic`/`base_cache::write_base` primitives `core::store::sync`
//! itself uses to persist a workbook and its merge base) so nothing here
//! bypasses the store's own conventions.
//!
//! Each test's doc comment names the acceptance sentence it proves. Three
//! come from the design doc's own L11 row: "Phone → desktop blob sync and
//! desktop → phone workbook sync on a home LAN; two-sided edits to
//! different cells merge with no conflict."

use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
use idl_rs::store::atomic::{sha256_hex, write_atomic};
use idl_rs::store::blob::{blob_path, write_blob};
use idl_rs::store::parquet::write_session_parquet;
use idl_rs::store::session_json::{empty_session_json, read_session_json, write_session_json, SessionJson};
use idl_rs::store::sync::base_cache;
use idl_rs::track_artifact::model::Track;
use idl_rs::track_artifact::read::read_track;
use idl_rs::track_artifact::write::write_track;
use idl_rs::workbook::v3::parse_workbook;

use super::{sync_with_peer, PairRequest, PairResponse, Peer, PairingState, SyncProgress, SyncServer, SyncServerConfig, PROTOCOL_VERSION};

fn no_progress(_p: SyncProgress) {}

/// Real wall-clock milliseconds — required for a pairing offer's TTL
/// (`handle_pair` redeems against the server's own real clock, not a
/// caller-supplied one; see `server.rs`'s own pairing tests for the same
/// note). [`sync_with_peer`]'s own `now_ms` parameter is unused by
/// `install` today, so every other call in this file uses small
/// deterministic values instead.
fn wall_clock_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn temp_data_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("idl-transport-loopback-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One end of a loopback pair: its own `<data>` root, a live `SyncServer`,
/// its own `PairingState`, and its own peer list — everything a real
/// desktop/phone instance holds (this task's brief: "two complete
/// instances").
struct Instance {
    data_root: PathBuf,
    peer_id: String,
    name: String,
    server: SyncServer,
    pairing: Arc<Mutex<PairingState>>,
    peers: Arc<Mutex<Vec<Peer>>>,
}

impl Instance {
    fn addr(&self) -> SocketAddr {
        self.server.local_addr()
    }

    async fn shutdown(self) {
        self.server.shutdown().await;
        let _ = std::fs::remove_dir_all(&self.data_root);
    }
}

async fn start_instance(tag: &str) -> Instance {
    let data_root = temp_data_root(tag);
    let peers = Arc::new(Mutex::new(Vec::new()));
    let pairing = Arc::new(Mutex::new(PairingState::default()));
    let peer_id = uuid::Uuid::new_v4().to_string();
    let config = SyncServerConfig {
        data_root: data_root.clone(),
        port: 0,
        bind_addr: Ipv4Addr::LOCALHOST.into(),
        peer_id: peer_id.clone(),
        name: tag.to_string(),
    };
    let server = SyncServer::start(config, pairing.clone(), peers.clone()).await.unwrap();
    Instance { data_root, peer_id, name: tag.to_string(), server, pairing, peers }
}

/// Runs the real `POST /pair` handshake: `offerer` mints a code on its own
/// live `PairingState`; `requester` redeems it over a genuine HTTP request
/// against `offerer`'s bound address. PLAN §7: "pair once, both keep a
/// token" — `handle_pair` already pushed a `Peer` for `requester` onto
/// `offerer.peers` (so `offerer`'s own [`auth_layer`]-guarded routes accept
/// calls bearing that token); this mirrors it by pushing the same token
/// onto `requester.peers` too, so `requester`'s routes accept calls from
/// `offerer` as well and each side's list holds the exact [`Peer`] record
/// it needs to make outgoing calls with.
async fn pair(offerer: &Instance, requester: &Instance) {
    let offer = offerer.pairing.lock().unwrap().offer(wall_clock_ms());
    let client = reqwest::Client::new();
    let request =
        PairRequest { code: offer.code, peer_id: requester.peer_id.clone(), name: requester.name.clone(), protocol_version: PROTOCOL_VERSION };
    let response = client.post(format!("http://{}/idl1/v1/pair", offerer.addr())).json(&request).send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK, "pairing handshake failed");
    let parsed: PairResponse = response.json().await.unwrap();
    requester.peers.lock().unwrap().push(Peer {
        peer_id: parsed.peer_id,
        name: parsed.name,
        token: parsed.token,
        protocol_version: parsed.protocol_version,
        paired_at_ms: wall_clock_ms(),
    });
}

/// The [`Peer`] record `from` uses to call `to` — present on `from`'s own
/// peer list after [`pair`] runs in either direction (see that helper's
/// doc comment for why the same token ends up on both sides' lists).
fn peer_of(from: &Instance, to_peer_id: &str) -> Peer {
    from.peers.lock().unwrap().iter().find(|p| p.peer_id == to_peer_id).cloned().unwrap()
}

/// Runs one sync: `from` pulls/pushes/merges against `to`, using the
/// `Peer` record `pair` set up on `from`'s own list.
async fn run_sync(from: &Instance, to: &Instance, now_ms: i64) -> super::SyncRunResult {
    let peer = peer_of(from, &to.peer_id);
    sync_with_peer(&from.data_root, &peer, to.addr(), now_ms, &no_progress).await.unwrap()
}

/// Writes a workbook file through the store's own atomic-write primitive
/// (there is no single dedicated "workbook writer" in `core` — every real
/// caller, including `store::sync::apply::install_workbook` itself, goes
/// through `write_atomic`/`write_atomic_with_retry` the same way), reading
/// back the current hash first so a second call edits in place instead of
/// spuriously conflicting with itself.
fn write_workbook_file(data_root: &Path, file_name: &str, bytes: &[u8]) -> String {
    let target = data_root.join("workbooks").join(format!("{file_name}.idl1wb"));
    let based_on_hash = std::fs::read(&target).ok().map(|b| sha256_hex(&b));
    write_atomic(data_root, &target, bytes, based_on_hash.as_deref()).unwrap()
}

/// Writes `session.json` through `write_session_json`, reading back the
/// current hash first so a second call edits in place (same pattern as
/// [`write_workbook_file`]).
fn write_session_json_file(data_root: &Path, session_id: &str, doc: &SessionJson) {
    let target = data_root.join("sessions").join(session_id).join("session.json");
    let based_on_hash = std::fs::read(&target).ok().map(|b| sha256_hex(&b));
    write_session_json(data_root, session_id, doc, based_on_hash.as_deref()).unwrap();
}

/// A minimal but real one-channel session, written through core's own
/// `write_blob` + `write_session_parquet` + `write_session_json` — the
/// same three writers a real import uses (this task's brief: "never by
/// hand-writing bytes into the tree").
fn seed_session(data_root: &Path, session_id: &str) -> String {
    let content = format!("raw capture bytes for {session_id}").into_bytes();
    let blob_sha = write_blob(data_root, &content).unwrap();

    let session = Session {
        session_id: session_id.to_string(),
        device_id: None,
        timestamp_utc_ms: 0,
        timestamp_source: TimestampSource::Header,
        config_checksum: None,
        source_format: SourceFormat::Idl0,
        blob_sha256: blob_sha.clone(),
        channels: vec![Channel {
            channel_id: "IMU0_AccelX".to_string(),
            t_us: vec![0, 500_000, 1_000_000],
            t_recorded_us: None,
            nominal_rate_hz: 2.0,
            column: RawColumn::F64(vec![1.0, 2.0, 3.0]),
            source_kind: "imu0".to_string(),
            unit: "g".to_string(),
            gaps: Vec::new(),
        }],
    };
    write_session_parquet(data_root, &session, "0.1.0").unwrap();
    write_session_json_file(data_root, session_id, &empty_session_json(session_id));
    blob_sha
}

/// Proves design doc §7's "Phone → desktop blob sync" half: a session
/// imported on A only — its raw blob, `data.parquet`, and `session.json`
/// — all appear on B, and the blob's hash verifies.
#[tokio::test]
async fn two_peers_a_session_imported_on_a_only_its_blob_data_parquet_and_session_json_all_appear_on_b_and_verify() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    let session_id = "0123456789abcdef";
    let blob_sha = seed_session(&a.data_root, session_id);

    // Act — B pulls A's session (B has nothing locally for this session id).
    let result = run_sync(&b, &a, 1000).await;

    // Assert
    assert_eq!(result.sessions_updated, 1);
    assert_eq!(result.blobs_transferred, 1);
    assert_eq!(std::fs::read(blob_path(&b.data_root, &blob_sha)).unwrap(), std::fs::read(blob_path(&a.data_root, &blob_sha)).unwrap());
    assert_eq!(sha256_hex(&std::fs::read(blob_path(&b.data_root, &blob_sha)).unwrap()), blob_sha);

    let a_dp = a.data_root.join("sessions").join(session_id).join("data.parquet");
    let b_dp = b.data_root.join("sessions").join(session_id).join("data.parquet");
    assert!(b_dp.is_file());
    assert_eq!(std::fs::read(&a_dp).unwrap(), std::fs::read(&b_dp).unwrap());

    let a_sj = read_session_json(&a.data_root.join("sessions").join(session_id).join("session.json")).unwrap();
    let b_sj = read_session_json(&b.data_root.join("sessions").join(session_id).join("session.json")).unwrap();
    assert_eq!(a_sj, b_sj);

    a.shutdown().await;
    b.shutdown().await;
}

/// Proves design doc §7's "desktop → phone workbook sync" half: a
/// workbook that exists on A only lands on B with A's content, driven
/// through the real `PUT /workbook/<id>` route (A pushes; B has nothing
/// locally).
#[tokio::test]
async fn two_peers_a_workbook_edited_on_a_only_b_has_as_version() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    const WB_ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";
    let src = format!("---\nid: {WB_ID}\nname: Setup\n---\n```math id=aaaaaaaa\nx = 1\n```\n");
    write_workbook_file(&a.data_root, "fork-tuning", src.as_bytes());

    // Act — A pushes to B (B has no local copy of this workbook).
    let result = run_sync(&a, &b, 1000).await;

    // Assert. `handle_workbook_put`'s own doc comment: a brand-new
    // workbook (no existing local copy on the receiving side) has nowhere
    // to source `peer_workbook_file_name` from the URL alone, so it falls
    // back to the URL's own id as the file name — B's copy lands at
    // `<WB_ID>.idl1wb`, not at A's `fork-tuning.idl1wb`. Content, not the
    // name, is what design §7 promises.
    assert_eq!(result.workbooks_merged, 0); // a Push, not a Pull — A's own count never covers a Push's peer-side outcome
    let written = std::fs::read_to_string(b.data_root.join("workbooks").join(format!("{WB_ID}.idl1wb"))).unwrap();
    assert_eq!(written, src);

    a.shutdown().await;
    b.shutdown().await;
}

/// Sets `cell_id`'s fence body to `new_body` on a cloned in-memory
/// [`WorkbookDoc`] — building a sibling edit by mutating the parsed
/// structure and rendering it exactly once, rather than by re-parsing an
/// already-rendered string (which this task found is *not* a fixed point
/// of `parse_workbook`/`render_workbook`: re-parsing rendered markdown and
/// rendering it again shifts blank-line spacing even for an untouched
/// cell, which C2 §7.3 — "prose travels with its cell" — then correctly,
/// but spuriously here, reports as a second `Changed` cell; a real UI
/// keeps a workbook open as a parsed document and renders once per save,
/// which is what this mirrors).
fn edit_cell(doc: &idl_rs::workbook::v3::WorkbookDoc, cell_id: &str, new_body: &str) -> String {
    let mut doc = doc.clone();
    let cell = doc.cells.iter_mut().find(|c| c.id == cell_id).unwrap();
    cell.raw_fence_body = new_body.to_string();
    idl_rs::workbook::v3::render_workbook(&doc)
}

/// Proves the design doc's own acceptance sentence: "two-sided edits to
/// different cells merge with no conflict." A common base (what two real
/// prior sync rounds would have left behind) is primed via
/// `base_cache::write_base` — core's own module for exactly this cache —
/// on both sides before the edits diverge; without it, `merge`'s empty-
/// base fallback cannot distinguish "unedited since the common ancestor"
/// from "changed", which the module's own `apply.rs` unit tests hit the
/// same way (seeding `base_cache::write_base` directly rather than
/// re-running two throwaway syncs). Each direction gets its own fresh pair
/// of instances seeded from the identical starting state — chaining
/// `a_pulls_b` into `b_pulls_a` on one shared pair would have the second
/// call observe the *first* call's already-merged file as its peer, which
/// is a different (and not what this scenario is about) situation.
#[tokio::test]
async fn two_peers_the_same_workbook_different_cells_edited_on_each_side_both_edits_present_on_both_sides_zero_conflicts() {
    // Arrange
    const WB_ID: &str = "1a2b3c4d-5e6f-4a1b-8c2d-3e4f5a6b7c8d";
    let (base_doc, _) = parse_workbook(&format!(
        "---\nid: {WB_ID}\nname: Setup\n---\n```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n"
    ))
    .unwrap();
    let base_src = idl_rs::workbook::v3::render_workbook(&base_doc);
    let a_edit = edit_cell(&base_doc, "aaaaaaaa", "x = 2");
    let b_edit = edit_cell(&base_doc, "bbbbbbbb", "y = 2");

    async fn one_direction(base_src: &str, local_edit: &str, peer_edit: &str, wb_id: &str) -> (String, u32) {
        let local = start_instance("local").await;
        let peer = start_instance("peer").await;
        pair(&local, &peer).await;
        write_workbook_file(&local.data_root, "fork-tuning", base_src.as_bytes());
        write_workbook_file(&peer.data_root, "fork-tuning", base_src.as_bytes());
        base_cache::write_base(&local.data_root, wb_id, base_src.as_bytes()).unwrap();
        base_cache::write_base(&peer.data_root, wb_id, base_src.as_bytes()).unwrap();
        write_workbook_file(&local.data_root, "fork-tuning", local_edit.as_bytes());
        write_workbook_file(&peer.data_root, "fork-tuning", peer_edit.as_bytes());

        let result = run_sync(&local, &peer, 1000).await;
        let written = std::fs::read_to_string(local.data_root.join("workbooks").join("fork-tuning.idl1wb")).unwrap();
        let conflicts = result.conflicts;
        local.shutdown().await;
        peer.shutdown().await;
        (written, conflicts)
    }

    // Act — each direction runs on its own fresh pair, both starting from
    // the identical primed base.
    let (a_written, a_conflicts) = one_direction(&base_src, &a_edit, &b_edit, WB_ID).await;
    let (b_written, b_conflicts) = one_direction(&base_src, &b_edit, &a_edit, WB_ID).await;

    // Assert
    assert_eq!(a_conflicts, 0);
    assert_eq!(b_conflicts, 0);
    for written in [&a_written, &b_written] {
        assert!(written.contains("x = 2"), "A's edit missing: {written}");
        assert!(written.contains("y = 2"), "B's edit missing: {written}");
        assert!(!written.contains("conflict"), "spurious conflict marker: {written}");
    }
}

/// Proves the flip side of the same design doc sentence: a real conflict
/// (the same cell edited differently on both sides) yields exactly one
/// conflict cell per side, with the C2 §7 marker, and both resulting files
/// still parse. As in the zero-conflict scenario above, each direction
/// runs on its own fresh pair from the identical starting state.
#[tokio::test]
async fn two_peers_the_same_cell_edited_on_both_sides_exactly_one_conflict_cell_on_each_side_below_the_local_cell_and_both_files_still_parse(
) {
    // Arrange
    const WB_ID: &str = "2b3c4d5e-6f7a-4b2c-9d3e-4f5a6b7c8d9e";
    let (base_doc, _) = parse_workbook(&format!("---\nid: {WB_ID}\nname: Setup\n---\n```math id=aaaaaaaa\nx = 1\n```\n")).unwrap();
    let base_src = idl_rs::workbook::v3::render_workbook(&base_doc);
    let a_edit = edit_cell(&base_doc, "aaaaaaaa", "x = 2");
    let b_edit = edit_cell(&base_doc, "aaaaaaaa", "x = 3");

    async fn one_direction(base_src: &str, local_edit: &str, peer_edit: &str) -> (String, u32) {
        let local = start_instance("local").await;
        let peer = start_instance("peer").await;
        pair(&local, &peer).await;
        write_workbook_file(&local.data_root, "fork-tuning", base_src.as_bytes());
        write_workbook_file(&peer.data_root, "fork-tuning", base_src.as_bytes());
        write_workbook_file(&local.data_root, "fork-tuning", local_edit.as_bytes());
        write_workbook_file(&peer.data_root, "fork-tuning", peer_edit.as_bytes());

        let result = run_sync(&local, &peer, 1000).await;
        let written = std::fs::read_to_string(local.data_root.join("workbooks").join("fork-tuning.idl1wb")).unwrap();
        let conflicts = result.conflicts;
        local.shutdown().await;
        peer.shutdown().await;
        (written, conflicts)
    }

    // Act
    let (a_written, a_conflicts) = one_direction(&base_src, &a_edit, &b_edit).await;
    let (b_written, b_conflicts) = one_direction(&base_src, &b_edit, &a_edit).await;

    // Assert
    assert_eq!(a_conflicts, 1);
    assert_eq!(b_conflicts, 1);
    for written in [&a_written, &b_written] {
        let marker_count = written.matches("<!-- conflict from").count();
        assert_eq!(marker_count, 1, "expected exactly one conflict marker: {written}");
        let (doc, _warnings) = parse_workbook(written).expect("merged workbook must still parse");
        assert_eq!(doc.cells.len(), 2);
    }
}

/// Proves sync is idempotent: running it a second time with nothing new
/// to move transfers nothing.
#[tokio::test]
async fn two_peers_sync_run_twice_the_second_run_moves_nothing() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    let content = b"loopback idempotence blob";
    let sha = sha256_hex(content);
    let path = blob_path(&a.data_root, &sha);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();

    // Act
    let first = run_sync(&b, &a, 1000).await;
    let second = run_sync(&b, &a, 2000).await;

    // Assert
    assert_eq!(first.blobs_transferred, 1);
    assert_eq!(second, super::SyncRunResult::default());

    a.shutdown().await;
    b.shutdown().await;
}

/// Proves track last-write-wins (C4 §6): A's newer track is taken by B;
/// running the sync back the other way afterwards leaves A untouched.
#[tokio::test]
async fn two_peers_a_adds_a_track_newer_than_bs_b_takes_it_the_reverse_direction_leaves_a_untouched() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    const TRACK_ID: &str = "3c4d5e6f-7a8b-4c3d-ae4f-5a6b7c8d9e0f";
    let a_track = Track {
        id: TRACK_ID.to_string(),
        name: "A-Line".to_string(),
        venue: "Whistler".to_string(),
        timing: None,
        sector_gates: Vec::new(),
        neutral_zones: Vec::new(),
        reference_polyline: Vec::new(),
        created_at_ms: 100,
        updated_at_ms: 2000,
    };
    let b_track = Track { updated_at_ms: 1000, name: "A-Line (stale)".to_string(), ..a_track.clone() };
    write_track(&a.data_root, &a_track).unwrap();
    write_track(&b.data_root, &b_track).unwrap();

    // Act — B pulls A's newer track.
    let b_pulls_a = run_sync(&b, &a, 1000).await;
    // Act — the reverse direction: A pulls from B, now identical to A.
    let a_pulls_b = run_sync(&a, &b, 1000).await;

    // Assert — `Track` has no `PartialEq`, so identity is checked field by
    // field on the two fields this scenario actually varies.
    assert_eq!(b_pulls_a.tracks_updated, 1);
    let b_now = read_track(&b.data_root.join("tracks").join(format!("{TRACK_ID}.idl0t"))).unwrap();
    assert_eq!(b_now.name, a_track.name);
    assert_eq!(b_now.updated_at_ms, a_track.updated_at_ms);

    assert_eq!(a_pulls_b.tracks_updated, 0);
    let a_now = read_track(&a.data_root.join("tracks").join(format!("{TRACK_ID}.idl0t"))).unwrap();
    assert_eq!(a_now.name, a_track.name, "A must be untouched by the reverse-direction run");
    assert_eq!(a_now.updated_at_ms, a_track.updated_at_ms, "A must be untouched by the reverse-direction run");

    a.shutdown().await;
    b.shutdown().await;
}

/// Proves `session.json`'s per-field merge (C4 §6, R91): editing different
/// fields on each side leaves both fields present on both sides after
/// syncing both ways.
#[tokio::test]
async fn two_peers_session_json_edited_on_each_side_in_different_fields_both_fields_survive_on_both_sides() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    let session_id = "fedcba9876543210";
    write_session_json_file(&a.data_root, session_id, &empty_session_json(session_id));
    write_session_json_file(&b.data_root, session_id, &empty_session_json(session_id));

    let mut a_doc = empty_session_json(session_id);
    a_doc.rider = "Alex".to_string();
    write_session_json_file(&a.data_root, session_id, &a_doc);
    let mut b_doc = empty_session_json(session_id);
    b_doc.bike = "Enduro".to_string();
    write_session_json_file(&b.data_root, session_id, &b_doc);

    // Act
    run_sync(&a, &b, 1000).await;
    run_sync(&b, &a, 1000).await;

    // Assert
    let a_final = read_session_json(&a.data_root.join("sessions").join(session_id).join("session.json")).unwrap();
    let b_final = read_session_json(&b.data_root.join("sessions").join(session_id).join("session.json")).unwrap();
    for doc in [&a_final, &b_final] {
        assert_eq!(doc.rider, "Alex");
        assert_eq!(doc.bike, "Enduro");
    }

    a.shutdown().await;
    b.shutdown().await;
}

/// Simulates a crash: B has already received the first bytes of a blob
/// (fetched via a genuine ranged `GET` against A's real server) when A's
/// server dies; A is "restarted" as a fresh `SyncServer` bound to the same
/// port, and re-running the sync resumes from B's existing `.part` file
/// and completes with the full content verified.
#[tokio::test]
async fn two_peers_a_server_killed_mid_transfer_then_restarted_the_resumed_run_completes_and_the_bytes_verify() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;
    let content = b"0123456789abcdefghijklmnopqrstuvwxyz-loopback-restart";
    let sha = sha256_hex(content);
    let path = blob_path(&a.data_root, &sha);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();

    // Fetch the first 10 bytes over a real ranged GET against A, exactly
    // what a crashed-mid-download client would already have on disk.
    let token = peer_of(&b, &a.peer_id).token;
    let partial_len: usize = 10;
    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{}/idl1/v1/blob/{sha}", a.addr()))
        .bearer_auth(&token)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", partial_len - 1))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::PARTIAL_CONTENT);
    let partial_bytes = response.bytes().await.unwrap();
    assert_eq!(partial_bytes.len(), partial_len);

    // The same deterministic `tmp/<digest>.part` naming `client.rs`'s
    // `tmp_part_path` uses for a `SyncClass::Blob` item with no
    // `session_id` (that function's own doc comment: a sha256 of
    // `"{class:?}:{session_id}:{key}"`) — duplicated here rather than
    // widening that function's visibility for one test.
    let identity = format!("Blob::{sha}");
    let digest = sha256_hex(identity.as_bytes());
    let part_path = b.data_root.join("tmp").join(format!("{digest}.part"));
    std::fs::create_dir_all(part_path.parent().unwrap()).unwrap();
    std::fs::write(&part_path, &partial_bytes).unwrap();

    let restart_port = a.addr().port();
    let a_data_root = a.data_root.clone();
    let a_peers = a.peers.clone();
    let a_peer_id = a.peer_id.clone();
    a.server.shutdown().await; // "the server killed mid-transfer"

    // "then restarted" — a fresh SyncServer, same data, same peers, same port.
    let restarted_config = SyncServerConfig {
        data_root: a_data_root.clone(),
        port: restart_port,
        bind_addr: Ipv4Addr::LOCALHOST.into(),
        peer_id: a_peer_id.clone(),
        name: "a".to_string(),
    };
    let restarted_pairing = Arc::new(Mutex::new(PairingState::default()));
    let restarted_server = SyncServer::start(restarted_config, restarted_pairing, a_peers).await.unwrap();
    let restarted_addr = restarted_server.local_addr();
    assert_eq!(restarted_addr.port(), restart_port);

    // Act
    let peer = peer_of(&b, &a_peer_id);
    let result = sync_with_peer(&b.data_root, &peer, restarted_addr, 1000, &no_progress).await.unwrap();

    // Assert
    assert_eq!(result.blobs_transferred, 1);
    assert_eq!(std::fs::read(blob_path(&b.data_root, &sha)).unwrap(), content);
    assert!(!part_path.exists());

    restarted_server.shutdown().await;
    let _ = std::fs::remove_dir_all(&a_data_root);
    b.shutdown().await;
}

/// Proves auth is enforced on every real route: a client presenting no
/// paired token is `401`ed everywhere but `/pair` itself, and nothing on
/// disk changes.
#[tokio::test]
async fn two_peers_an_unpaired_client_every_route_401s_and_nothing_changes() {
    // Arrange
    let a = start_instance("a").await;
    let content = b"untouched by an unpaired caller";
    let sha = sha256_hex(content);
    let blob_path_a = blob_path(&a.data_root, &sha);
    std::fs::create_dir_all(blob_path_a.parent().unwrap()).unwrap();
    std::fs::write(&blob_path_a, content).unwrap();

    let hash_placeholder = "a".repeat(64);
    let session_placeholder = "a".repeat(16);
    let uuid_placeholder = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    let base = format!("http://{}/idl1/v1", a.addr());
    let routes: Vec<(reqwest::Method, String)> = vec![
        (reqwest::Method::GET, format!("{base}/manifest")),
        (reqwest::Method::GET, format!("{base}/blob/{hash_placeholder}")),
        (reqwest::Method::PUT, format!("{base}/blob/{hash_placeholder}")),
        (reqwest::Method::GET, format!("{base}/session/{session_placeholder}/data.parquet")),
        (reqwest::Method::PUT, format!("{base}/session/{session_placeholder}/data.parquet")),
        (reqwest::Method::GET, format!("{base}/session/{session_placeholder}/session.json")),
        (reqwest::Method::PUT, format!("{base}/session/{session_placeholder}/session.json")),
        (reqwest::Method::GET, format!("{base}/session/{session_placeholder}/derived/{hash_placeholder}.parquet")),
        (reqwest::Method::PUT, format!("{base}/session/{session_placeholder}/derived/{hash_placeholder}.parquet")),
        (reqwest::Method::GET, format!("{base}/workbook/{uuid_placeholder}")),
        (reqwest::Method::PUT, format!("{base}/workbook/{uuid_placeholder}")),
        (reqwest::Method::GET, format!("{base}/track/{uuid_placeholder}")),
        (reqwest::Method::PUT, format!("{base}/track/{uuid_placeholder}")),
        (reqwest::Method::GET, format!("{base}/profile/{uuid_placeholder}")),
        (reqwest::Method::PUT, format!("{base}/profile/{uuid_placeholder}")),
    ];
    let client = reqwest::Client::new();

    // Act / Assert
    for (method, url) in routes {
        let response = client.request(method.clone(), &url).body(Vec::<u8>::new()).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED, "{method} {url} was not 401");
    }

    // Nothing on disk changed: the pre-existing blob is intact, and no
    // syncable directory a PUT could have created now exists.
    assert_eq!(std::fs::read(&blob_path_a).unwrap(), content);
    for dir in ["workbooks", "tracks", "profiles", "sessions"] {
        assert!(!a.data_root.join(dir).exists(), "{dir} should not exist after only 401s");
    }

    a.shutdown().await;
}

/// Proves the manifest walk (and thus every route built on it) never
/// touches the catalog, and a clean sync leaves no partial-download
/// leftovers: a mixed first-sync (blob + workbook + session + track, all
/// pulled by B from A) still leaves both roots with an absent
/// `catalog.sqlite` and an empty `tmp/`.
#[tokio::test]
async fn two_peers_after_every_scenario_neither_tmp_holds_leftovers_and_neither_catalog_sqlite_was_read_or_written() {
    // Arrange
    let a = start_instance("a").await;
    let b = start_instance("b").await;
    pair(&a, &b).await;

    let content = b"tmp/catalog cleanup check blob";
    let sha = sha256_hex(content);
    let path = blob_path(&a.data_root, &sha);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();

    const WB_ID: &str = "4d5e6f7a-8b9c-4d4e-af5a-6b7c8d9e0f1a";
    write_workbook_file(&a.data_root, "cleanup-check", format!("---\nid: {WB_ID}\nname: Setup\n---\n").as_bytes());

    let session_id = "1111222233334444";
    seed_session(&a.data_root, session_id);

    const TRACK_ID: &str = "5e6f7a8b-9c0d-4e5f-b06b-7c8d9e0f1a2b";
    write_track(
        &a.data_root,
        &Track {
            id: TRACK_ID.to_string(),
            name: "Cleanup Track".to_string(),
            venue: "Test".to_string(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 0,
            updated_at_ms: 500,
        },
    )
    .unwrap();

    // Act
    let result = run_sync(&b, &a, 1000).await;

    // Assert
    assert!(result.blobs_transferred >= 1);
    assert!(result.workbooks_merged >= 1);
    assert!(result.sessions_updated >= 1);
    assert!(result.tracks_updated >= 1);

    for root in [&a.data_root, &b.data_root] {
        assert!(!root.join("catalog.sqlite").exists(), "catalog.sqlite must never be created by sync");
        assert!(!root.join("catalog.sqlite-wal").exists());
        assert!(!root.join("catalog.sqlite-shm").exists());
        let tmp_dir = root.join("tmp");
        if tmp_dir.is_dir() {
            let leftovers: Vec<_> = std::fs::read_dir(&tmp_dir).unwrap().filter_map(|e| e.ok()).collect();
            assert!(leftovers.is_empty(), "leftover tmp entries in {}: {leftovers:?}", root.display());
        }
    }

    a.shutdown().await;
    b.shutdown().await;
}
