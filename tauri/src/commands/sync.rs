//! LAN sync commands (C3 §3.9, L11 Tasks 12–13): `sync_status`, `sync_now`,
//! `pair_peer`, `start_pairing`, `unpair_peer`, `set_sync_device_name`,
//! plus the `peer_appeared` event and the pure auto-trigger decision. Thin
//! over `idl-transport`'s
//! `sync` module — merging, diffing and installing stay in `idl-rs`; the
//! wire itself stays in `idl-transport` (CLAUDE.md §2, this task's brief).
//! No HTTP client of any kind lives in this crate — `idl_transport::sync::
//! pair_with_peer`/`sync_with_peer` each build their own short-lived
//! `reqwest::Client` internally (ruling R104).
//!
//! `pair_peer` takes `(peer_id, code)`, not `code` alone (ruling R104,
//! amending C3 §3.9): the caller names the specific discovered peer the
//! code came from — this command never guesses which online peer offered a
//! code, and never fans a pairing secret out to every unpaired peer on the
//! LAN.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

use idl_transport::sync::{DiscoveredPeer, PairRequest, PairingOffer, PairingState, Peer};

use crate::error::{IpcError, IpcErrorKind};
use crate::state::{DataDir, SyncState};

/// One paired peer's current status (C3 §3.9).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PeerStatusDto {
    pub peer_id: String,
    pub name: String,
    /// Currently visible on the LAN via mDNS.
    pub online: bool,
    /// The peer's `/idl1/v1`-style mDNS `v=` value at pairing time.
    pub protocol_version: u32,
    /// Epoch milliseconds when pairing completed.
    pub paired_at_ms: i64,
}

/// This device's own sync identity, as seen by peers (C3 §3.9, ruling
/// R172). Both fields are non-optional: `identity::load_or_create` mints
/// the identity on first launch, so there is no state where it is absent.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ThisDeviceDto {
    /// This device's stable id on the wire. Shown here — not guessed by
    /// the app — because ruling R104 requires a user pairing two of their
    /// own machines to read it off this device's own screen.
    pub peer_id: String,
    /// This device's display name, as sent in outgoing `PairRequest`s and
    /// as last set by `set_sync_device_name`.
    pub name: String,
}

/// One peer visible on the LAN via mDNS that this device has not paired
/// with (C3 §3.9, L11 Task 14). Lets a Settings pane prefill `peer_id` for
/// `pair_peer` (ruling R104) instead of Isaac typing a 32-character id read
/// off another screen.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DiscoveredPeerDto {
    pub peer_id: String,
    /// Display name advertised by the peer. May be empty.
    pub name: String,
    /// The peer's mDNS TXT-record `v` value, as broadcast (may differ from
    /// this device's own `PROTOCOL_VERSION`).
    pub protocol_version: u32,
    /// IP address the peer's sync server was last seen at.
    pub address: String,
    /// Port the peer's sync server was last seen at.
    pub port: u16,
}

/// Converts a raw LAN sighting into its DTO. Carries only what an unpaired
/// peer's row may show — no pairing token, nothing else a paired peer's
/// [`PeerStatusDto`] holds and an unpaired one has no business knowing.
fn discovered_peer_dto(peer: &DiscoveredPeer) -> DiscoveredPeerDto {
    DiscoveredPeerDto {
        peer_id: peer.peer_id.clone(),
        name: peer.name.clone(),
        protocol_version: peer.protocol_version,
        address: peer.addr.ip().to_string(),
        port: peer.addr.port(),
    }
}

/// `sync_status`'s return (C3 §3.9).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SyncStatusDto {
    pub paired_peers: Vec<PeerStatusDto>,
    /// Peers currently visible on the LAN via mDNS that are NOT in
    /// `paired_peers` (L11 Task 14). Disjoint from `paired_peers` by
    /// construction: a peer that is paired appears there and only there —
    /// letting a peer appear in both would leave every consumer to decide
    /// which list wins, and not all would decide the same way.
    pub discovered_peers: Vec<DiscoveredPeerDto>,
    /// `None` if never synced with any peer.
    pub last_sync_utc_ms: Option<i64>,
    /// This device's own id and name (ruling R172). Renaming via
    /// `set_sync_device_name` is not retroactive: it does not change what
    /// an already-paired peer displays for us, only what this field (and
    /// future pairings) reports from here on.
    pub this_device: ThisDeviceDto,
}

/// `sync_now`'s return (C3 §3.9, all six fields per ruling R102).
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize)]
pub struct SyncResultDto {
    pub blobs_transferred: u32,
    pub workbooks_merged: u32,
    pub conflicts: u32,
    pub sessions_updated: u32,
    pub tracks_updated: u32,
    pub profiles_updated: u32,
}

impl From<idl_transport::sync::SyncRunResult> for SyncResultDto {
    fn from(r: idl_transport::sync::SyncRunResult) -> Self {
        Self {
            blobs_transferred: r.blobs_transferred,
            workbooks_merged: r.workbooks_merged,
            conflicts: r.conflicts,
            sessions_updated: r.sessions_updated,
            tracks_updated: r.tracks_updated,
            profiles_updated: r.profiles_updated,
        }
    }
}

/// `start_pairing`'s return (C3 §3.9's `PairingCode`).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PairingOfferDto {
    /// Six decimal digits, leading zeros preserved.
    pub code: String,
    /// Epoch milliseconds after which the code no longer redeems.
    pub expires_at_ms: i64,
}

impl From<PairingOffer> for PairingOfferDto {
    fn from(o: PairingOffer) -> Self {
        Self { code: o.code, expires_at_ms: o.expires_at_ms }
    }
}

/// Milliseconds since the Unix epoch — this module owns no clock beyond
/// this (mirrors `idl_transport::sync::server`'s own `now_ms`).
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// The widened `peer_appeared` event's payload (C3 §3.9, L11 Task 14, lead
/// ruling on task 2): a LAN sighting that is either an already-paired peer
/// or one this device has not paired with, distinguished by the `status`
/// tag so a listener never has to make a second `sync_status` call to tell
/// the two apart. An enum, not a `paired: bool` on a single shape, because
/// `PeerStatusDto::paired_at_ms` is a fact an unpaired sighting has not
/// earned — flattening it to `Option<i64>` would leave every consumer
/// distinguishing "not paired" from "not paired but the flag lied" instead
/// of the compiler ruling it out.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PeerSightingDto {
    Paired(PeerStatusDto),
    Discovered(DiscoveredPeerDto),
}

/// Converts one paired `Peer` plus the latest browse results into its DTO.
pub(crate) fn peer_status_dto(peer: &Peer, discovered: &HashMap<String, DiscoveredPeer>) -> PeerStatusDto {
    PeerStatusDto {
        peer_id: peer.peer_id.clone(),
        name: peer.name.clone(),
        online: discovered.contains_key(&peer.peer_id),
        protocol_version: peer.protocol_version,
        paired_at_ms: peer.paired_at_ms,
    }
}

/// Builds the widened `peer_appeared` payload for one LAN sighting:
/// `Paired` if `sighting.peer_id` is in `peers`, `Discovered` otherwise.
/// The variant follows current membership, not the state at first
/// sighting — a peer that pairs mid-session arrives as `Paired` on its next
/// sighting regardless of how it arrived on its first.
pub(crate) fn peer_sighting_dto(sighting: &DiscoveredPeer, peers: &[Peer], discovered: &HashMap<String, DiscoveredPeer>) -> PeerSightingDto {
    match peers.iter().find(|p| p.peer_id == sighting.peer_id) {
        Some(paired) => PeerSightingDto::Paired(peer_status_dto(paired, discovered)),
        None => PeerSightingDto::Discovered(discovered_peer_dto(sighting)),
    }
}

/// Transport-agnostic core of `sync_status`: every paired peer with
/// `online` set from the latest browse results, every other browse
/// sighting as a `discovered_peers` entry (disjoint from `paired_peers` —
/// see [`SyncStatusDto::discovered_peers`]), the most recent successful
/// sync across every peer (`None` if none has ever completed), and this
/// device's own id and name (ruling R172).
fn sync_status_via(
    peers: &[Peer],
    discovered: &HashMap<String, DiscoveredPeer>,
    last_sync: &HashMap<String, i64>,
    this_peer_id: &str,
    this_name: &str,
) -> SyncStatusDto {
    let paired_peers = peers.iter().map(|p| peer_status_dto(p, discovered)).collect();
    let discovered_peers = discovered
        .values()
        .filter(|d| !peers.iter().any(|p| p.peer_id == d.peer_id))
        .map(discovered_peer_dto)
        .collect();
    let last_sync_utc_ms = last_sync.values().copied().max();
    let this_device = ThisDeviceDto { peer_id: this_peer_id.to_string(), name: this_name.to_string() };
    SyncStatusDto { paired_peers, discovered_peers, last_sync_utc_ms, this_device }
}

/// C3 §3.9 `sync_status()`.
#[tauri::command]
pub async fn sync_status(state: tauri::State<'_, SyncState>) -> Result<SyncStatusDto, IpcError> {
    let peers = state.peers.lock().unwrap_or_else(|e| e.into_inner());
    let discovered = state.discovered.lock().unwrap_or_else(|e| e.into_inner());
    let last_sync = state.last_sync.lock().unwrap_or_else(|e| e.into_inner());
    let name = state.name.lock().unwrap_or_else(|e| e.into_inner());
    Ok(sync_status_via(&peers, &discovered, &last_sync, &state.peer_id, &name))
}

/// Transport-agnostic core of `start_pairing`: mints a fresh offer on
/// `pairing`, replacing any outstanding one (PLAN §3, symmetric pairing).
fn start_pairing_via(pairing: &std::sync::Mutex<PairingState>, now_ms: i64) -> PairingOfferDto {
    let offer = pairing.lock().unwrap_or_else(|e| e.into_inner()).offer(now_ms);
    PairingOfferDto::from(offer)
}

/// C3 §3.9 `start_pairing()` — added post-sign, lead ruling R88, L11 Task 1.
#[tauri::command]
pub async fn start_pairing(state: tauri::State<'_, SyncState>) -> Result<PairingOfferDto, IpcError> {
    Ok(start_pairing_via(&state.pairing, now_ms()))
}

/// Transport-agnostic core of `unpair_peer`: drops `peer_id` from `peers`
/// and persists the shortened list to `peers_path`. An unknown `peer_id`
/// is `not_found` — nothing is written.
fn unpair_peer_via(
    peers_path: &std::path::Path,
    peers: &std::sync::Mutex<Vec<Peer>>,
    peer_id: &str,
) -> Result<(), IpcError> {
    let mut list = peers.lock().unwrap_or_else(|e| e.into_inner());
    let before = list.len();
    list.retain(|p| p.peer_id != peer_id);
    if list.len() == before {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("unknown peer_id: {peer_id}")));
    }
    idl_transport::sync::save_peers(peers_path, &list).map_err(IpcError::from)
}

/// C3 §3.9 `unpair_peer(peer_id)` — added post-sign, lead ruling R88, L11
/// Task 1. Forgets a paired peer: discards its stored token and drops it
/// from `sync_status`'s `paired_peers` list.
#[tauri::command]
pub async fn unpair_peer(peer_id: String, state: tauri::State<'_, SyncState>) -> Result<(), IpcError> {
    unpair_peer_via(&state.peers_path, &state.peers, &peer_id)
}

/// Transport-agnostic core of `set_sync_device_name`: rejects a
/// blank/whitespace-only name — unlike the hostname-seeded default
/// (`identity::load_or_create`, applied only on first launch), there is no
/// sensible fallback to apply silently once the user has explicitly chosen
/// to rename — then persists the renamed identity to `identity_path` via
/// `identity::set_name`, keeping `peer_id` unchanged, and updates the live
/// `name` lock so the rest of the process sees the new name immediately.
fn set_sync_device_name_via(
    identity_path: &std::path::Path,
    peer_id: &str,
    name_lock: &std::sync::Mutex<String>,
    name: String,
) -> Result<String, IpcError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, "device name must not be blank"));
    }

    let current = idl_transport::sync::Identity {
        peer_id: peer_id.to_string(),
        name: name_lock.lock().unwrap_or_else(|e| e.into_inner()).clone(),
    };
    let updated = idl_transport::sync::identity::set_name(identity_path, &current, trimmed.to_string()).map_err(IpcError::from)?;

    *name_lock.lock().unwrap_or_else(|e| e.into_inner()) = updated.name.clone();
    Ok(updated.name)
}

/// C3 §3.9 `set_sync_device_name(name)` — added post-sign (2026-09-07, lead
/// ruling R105, L11 Task 13). Renames this device: persists to
/// `identity.json` and updates the live `SyncState` so `pair_peer`'s
/// outgoing `PairRequest.name` reflects it for the rest of the process's
/// life. Does not retroactively change what an already-paired peer
/// displays for us — that name was copied into their own peer file at
/// pairing time, and re-sending it would need a wire message this contract
/// does not define.
#[tauri::command]
pub async fn set_sync_device_name(name: String, state: tauri::State<'_, SyncState>) -> Result<String, IpcError> {
    set_sync_device_name_via(&state.identity_path, &state.peer_id, &state.name, name)
}

/// `true` if `code` is exactly six ASCII digits (design §7, C3 §3.9). Kept
/// separate from `pair_peer`'s body so a malformed code fails before any
/// state is touched and before any request is sent (this task's brief's
/// test list).
fn validate_pairing_code(code: &str) -> Result<(), IpcError> {
    if code.len() == 6 && code.chars().all(|c| c.is_ascii_digit()) {
        Ok(())
    } else {
        Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("pairing code must be exactly six digits, got {code:?}")))
    }
}

/// Resolves `peer_id`'s current LAN address from the browse set — ruling
/// R104: the caller names a specific discovered peer, this function never
/// guesses among several. `not_found` when `peer_id` is not currently
/// visible.
fn resolve_discovered_addr(discovered: &HashMap<String, DiscoveredPeer>, peer_id: &str) -> Result<SocketAddr, IpcError> {
    discovered
        .get(peer_id)
        .map(|d| d.addr)
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, format!("peer {peer_id} not currently visible on the LAN")))
}

/// C3 §3.9 `pair_peer(peer_id, code)` (signature amended by ruling R104 —
/// originally `code` alone) — the *initiating* side: redeems `code`, the
/// six-digit pairing code shown on `peer_id`'s screen, against that
/// specific discovered peer over `POST /pair`
/// (`idl_transport::sync::pair_with_peer`). On success, stores the
/// returned token as a newly paired [`Peer`] (replacing any prior entry
/// for the same `peer_id`) and persists it to `peers_path`.
#[tauri::command]
pub async fn pair_peer(peer_id: String, code: String, state: tauri::State<'_, SyncState>) -> Result<PeerStatusDto, IpcError> {
    validate_pairing_code(&code)?;

    let addr = {
        let discovered = state.discovered.lock().unwrap_or_else(|e| e.into_inner());
        resolve_discovered_addr(&discovered, &peer_id)?
    };

    let request = PairRequest {
        code,
        peer_id: state.peer_id.clone(),
        name: state.name.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        protocol_version: idl_transport::sync::PROTOCOL_VERSION,
    };
    let response = idl_transport::sync::pair_with_peer(addr, &request).await.map_err(IpcError::from)?;

    let peer = Peer {
        peer_id: response.peer_id,
        name: response.name,
        token: response.token,
        protocol_version: response.protocol_version,
        paired_at_ms: now_ms(),
    };
    {
        let mut peers = state.peers.lock().unwrap_or_else(|e| e.into_inner());
        peers.retain(|p| p.peer_id != peer.peer_id);
        peers.push(peer.clone());
        idl_transport::sync::save_peers(&state.peers_path, &peers).map_err(IpcError::from)?;
    }

    let discovered = state.discovered.lock().unwrap_or_else(|e| e.into_inner());
    Ok(peer_status_dto(&peer, &discovered))
}

/// Transport-agnostic core of `sync_now`'s peer/address resolution: an
/// unknown `peer_id` (not in the paired list) or one not currently visible
/// on the LAN is `not_found` either way — both mean "there is nothing to
/// sync with right now" from the caller's point of view.
fn resolve_peer_and_addr(peer_id: &str, peers: &[Peer], discovered: &HashMap<String, DiscoveredPeer>) -> Result<(Peer, SocketAddr), IpcError> {
    let peer = peers
        .iter()
        .find(|p| p.peer_id == peer_id)
        .cloned()
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, format!("unknown peer_id: {peer_id}")))?;
    let addr = resolve_discovered_addr(discovered, peer_id)?;
    Ok((peer, addr))
}

/// Runs one full sync with `peer` at `addr` and re-indexes exactly the
/// sessions it touched — the shared core behind both `sync_now` (manual,
/// reports `Progress` on a channel) and `SyncState`'s background
/// auto-trigger task (silent — no open channel to report on). Marks
/// `peer_id` running for the duration (the auto-trigger decision's own
/// "never while a sync for that peer is already running" rule reads this
/// same set) and records the completion time in `last_sync` on success.
///
/// Catalog re-index is per touched session id
/// (`idl_rs::store::catalog::index_session`, mirroring `rescan_tracks_via`
/// — never a whole `rebuild_catalog`, R104 addendum) and only when
/// `catalog.sqlite` already exists; a re-index failure is swallowed for
/// the same reason `rescan_tracks_via` documents: the sync itself already
/// succeeded, and `rebuild_catalog` remains the recovery path.
pub(crate) async fn run_one_sync(
    running: &std::sync::Mutex<HashSet<String>>,
    last_sync: &std::sync::Mutex<HashMap<String, i64>>,
    data_root: &std::path::Path,
    peer: &Peer,
    addr: SocketAddr,
    on_progress: &(dyn Fn(idl_transport::sync::SyncProgress) + Send + Sync),
) -> Result<idl_transport::sync::SyncRunResult, IpcError> {
    let peer_id = peer.peer_id.clone();
    running.lock().unwrap_or_else(|e| e.into_inner()).insert(peer_id.clone());
    let result = idl_transport::sync::sync_with_peer(data_root, peer, addr, now_ms(), on_progress).await;
    running.lock().unwrap_or_else(|e| e.into_inner()).remove(&peer_id);
    let result = result.map_err(IpcError::from)?;

    let catalog_path = data_root.join("catalog.sqlite");
    if catalog_path.is_file() {
        if let Ok(conn) = idl_rs::store::catalog::open_catalog(&catalog_path) {
            for session_id in &result.sessions_touched {
                let _ = idl_rs::store::catalog::index_session(&conn, data_root, session_id);
            }
        }
    }

    last_sync.lock().unwrap_or_else(|e| e.into_inner()).insert(peer_id, now_ms());
    Ok(result)
}

/// C3 §3.9 `sync_now(peer_id, progress)`: resolves `peer_id`'s address from
/// the browse set and runs [`run_one_sync`], forwarding each tick as C3's
/// `Progress` on `progress`.
#[tauri::command]
pub async fn sync_now(
    peer_id: String,
    progress: tauri::ipc::Channel<crate::commands::device::Progress>,
    state: tauri::State<'_, SyncState>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<SyncResultDto, IpcError> {
    let (peer, addr) = {
        let peers = state.peers.lock().unwrap_or_else(|e| e.into_inner());
        let discovered = state.discovered.lock().unwrap_or_else(|e| e.into_inner());
        resolve_peer_and_addr(&peer_id, &peers, &discovered)?
    };

    let on_progress = |p: idl_transport::sync::SyncProgress| {
        let _ = progress.send(crate::commands::device::Progress { done: p.done, total: p.total, phase: p.phase.to_string() });
    };
    let result = run_one_sync(&state.running, &state.last_sync, &data_dir.0, &peer, addr, &on_progress).await?;
    Ok(SyncResultDto::from(result))
}

/// The subset of a browsed/paired peer's state
/// [`should_auto_sync`] needs — kept minimal so the decision itself
/// (this task's brief, "Key logic": "a pure function") is testable without
/// constructing a full [`DiscoveredPeer`]/[`Peer`] pair.
pub struct AutoSyncCandidate<'a> {
    pub peer_id: &'a str,
    /// `true` when this peer id is in the paired-peer list.
    pub paired: bool,
    /// The protocol version this sighting advertised.
    pub protocol_version: u32,
}

/// Whether a newly-seen `peer` should trigger an automatic sync right now
/// (PLAN §8 Q9, lead ruling R88): never for an unpaired or
/// protocol-incompatible peer, never while a manual or automatic sync for
/// that peer is already running, and at most once per peer per 60 s.
/// Pure — the background browse task (`SyncState`'s discovery loop) is the
/// only caller, never a command handler (this task's brief, "Do not").
pub fn should_auto_sync(
    peer: &AutoSyncCandidate<'_>,
    last_sync_ms: Option<i64>,
    now_ms: i64,
    running: &HashSet<String>,
) -> bool {
    const AUTO_SYNC_MIN_INTERVAL_MS: i64 = 60_000;

    if !peer.paired {
        return false;
    }
    if peer.protocol_version != idl_transport::sync::PROTOCOL_VERSION {
        return false;
    }
    if running.contains(peer.peer_id) {
        return false;
    }
    match last_sync_ms {
        Some(t) => now_ms.saturating_sub(t) >= AUTO_SYNC_MIN_INTERVAL_MS,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(id: &str) -> Peer {
        Peer { peer_id: id.to_string(), name: "Pit Tablet".to_string(), token: "tok".to_string(), protocol_version: idl_transport::sync::PROTOCOL_VERSION, paired_at_ms: 1_700_000_000_000 }
    }

    // -- sync_status ---------------------------------------------------

    #[test]
    fn sync_status_no_peers_an_empty_list_and_a_null_last_sync_utc_ms() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let last_sync = HashMap::new();

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");

        // Assert
        assert!(status.paired_peers.is_empty());
        assert_eq!(status.last_sync_utc_ms, None);
    }

    #[test]
    fn sync_status_a_paired_peer_seen_on_the_lan_is_online_and_last_sync_utc_ms_is_the_max() {
        // Arrange
        let peers = vec![peer("peer-1"), peer("peer-2")];
        let mut discovered = HashMap::new();
        discovered.insert(
            "peer-1".to_string(),
            DiscoveredPeer { peer_id: "peer-1".to_string(), name: "Pit Tablet".to_string(), protocol_version: 1, addr: "127.0.0.1:9000".parse().unwrap() },
        );
        let mut last_sync = HashMap::new();
        last_sync.insert("peer-1".to_string(), 1_000);
        last_sync.insert("peer-2".to_string(), 2_000);

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");

        // Assert
        let p1 = status.paired_peers.iter().find(|p| p.peer_id == "peer-1").unwrap();
        let p2 = status.paired_peers.iter().find(|p| p.peer_id == "peer-2").unwrap();
        assert!(p1.online);
        assert!(!p2.online);
        assert_eq!(status.last_sync_utc_ms, Some(2_000));
    }

    #[test]
    fn sync_status_a_discovered_peer_that_is_not_paired_appears_in_discovered_peers_not_paired_peers() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let mut discovered = HashMap::new();
        discovered.insert(
            "peer-1".to_string(),
            DiscoveredPeer { peer_id: "peer-1".to_string(), name: "Pit Tablet".to_string(), protocol_version: 1, addr: "127.0.0.1:9000".parse().unwrap() },
        );
        let last_sync = HashMap::new();

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");

        // Assert
        assert!(status.paired_peers.is_empty());
        assert_eq!(status.discovered_peers.len(), 1);
        let d = &status.discovered_peers[0];
        assert_eq!(d.peer_id, "peer-1");
        assert_eq!(d.name, "Pit Tablet");
        assert_eq!(d.protocol_version, 1);
        assert_eq!(d.address, "127.0.0.1");
        assert_eq!(d.port, 9000);
    }

    #[test]
    fn sync_status_a_discovered_peer_that_is_paired_appears_only_in_paired_peers() {
        // Arrange
        let peers = vec![peer("peer-1")];
        let mut discovered = HashMap::new();
        discovered.insert(
            "peer-1".to_string(),
            DiscoveredPeer { peer_id: "peer-1".to_string(), name: "Pit Tablet".to_string(), protocol_version: 1, addr: "127.0.0.1:9000".parse().unwrap() },
        );
        let last_sync = HashMap::new();

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");

        // Assert
        assert_eq!(status.paired_peers.len(), 1);
        assert!(status.discovered_peers.is_empty());
    }

    #[test]
    fn sync_status_nothing_discovered_discovered_peers_is_empty_not_absent() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let last_sync = HashMap::new();

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");
        let json = serde_json::to_value(&status).unwrap();

        // Assert
        assert!(status.discovered_peers.is_empty());
        assert!(json.get("discovered_peers").is_some());
        assert_eq!(json["discovered_peers"], serde_json::json!([]));
    }

    // -- peer_sighting_dto (the widened peer_appeared payload) ----------

    fn sighting(id: &str) -> DiscoveredPeer {
        DiscoveredPeer { peer_id: id.to_string(), name: "Pit Tablet".to_string(), protocol_version: 1, addr: "127.0.0.1:9000".parse().unwrap() }
    }

    #[test]
    fn peer_sighting_dto_an_unpaired_sighting_is_the_discovered_variant() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let s = sighting("peer-1");

        // Act
        let dto = peer_sighting_dto(&s, &peers, &discovered);
        let json = serde_json::to_value(&dto).unwrap();

        // Assert
        assert!(matches!(dto, PeerSightingDto::Discovered(_)));
        assert_eq!(json["status"], "discovered");
        assert_eq!(json["peer_id"], "peer-1");
        assert!(json.get("paired_at_ms").is_none());
    }

    #[test]
    fn peer_sighting_dto_a_paired_sighting_is_the_paired_variant() {
        // Arrange
        let peers = vec![peer("peer-1")];
        let discovered = HashMap::new();
        let s = sighting("peer-1");

        // Act
        let dto = peer_sighting_dto(&s, &peers, &discovered);
        let json = serde_json::to_value(&dto).unwrap();

        // Assert
        assert!(matches!(dto, PeerSightingDto::Paired(_)));
        assert_eq!(json["status"], "paired");
        assert_eq!(json["peer_id"], "peer-1");
        assert!(json.get("paired_at_ms").is_some());
    }

    #[test]
    fn peer_sighting_dto_a_peer_that_pairs_mid_session_arrives_as_paired_on_its_next_sighting() {
        // Arrange: same peer_id, first seen while unpaired
        let s = sighting("peer-1");
        let no_peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let first = peer_sighting_dto(&s, &no_peers, &discovered);

        // Act: now pairs, then is seen again
        let now_paired = vec![peer("peer-1")];
        let second = peer_sighting_dto(&s, &now_paired, &discovered);

        // Assert
        assert!(matches!(first, PeerSightingDto::Discovered(_)));
        assert!(matches!(second, PeerSightingDto::Paired(_)));
    }

    #[test]
    fn sync_status_a_freshly_minted_identity_this_device_carries_its_peer_id_and_name() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let last_sync = HashMap::new();

        // Act
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", "idl1");

        // Assert
        assert_eq!(status.this_device.peer_id, "peer-self");
        assert_eq!(status.this_device.name, "idl1");
    }

    #[test]
    fn sync_status_after_set_sync_device_name_reports_the_new_name() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let identity_path = dir.path().join("identity.json");
        let name_lock = std::sync::Mutex::new("idl1".to_string());
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let last_sync = HashMap::new();

        // Act
        set_sync_device_name_via(&identity_path, "peer-self", &name_lock, "Pit Wall Laptop".to_string()).unwrap();
        let name = name_lock.lock().unwrap().clone();
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", &name);

        // Assert
        assert_eq!(status.this_device.peer_id, "peer-self");
        assert_eq!(status.this_device.name, "Pit Wall Laptop");
    }

    #[test]
    fn sync_status_a_rejected_blank_rename_still_reports_the_old_name() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let identity_path = dir.path().join("identity.json");
        let name_lock = std::sync::Mutex::new("idl1".to_string());
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();
        let last_sync = HashMap::new();

        // Act
        let rename = set_sync_device_name_via(&identity_path, "peer-self", &name_lock, "   ".to_string());
        let name = name_lock.lock().unwrap().clone();
        let status = sync_status_via(&peers, &discovered, &last_sync, "peer-self", &name);

        // Assert
        assert_eq!(rename.unwrap_err().kind, IpcErrorKind::InvalidArgument);
        assert_eq!(status.this_device.name, "idl1");
    }

    // -- start_pairing --------------------------------------------------

    #[test]
    fn start_pairing_returns_a_six_digit_code_and_a_future_expiry() {
        // Arrange
        let pairing = std::sync::Mutex::new(PairingState::default());

        // Act
        let offer = start_pairing_via(&pairing, 1_000);

        // Assert
        assert_eq!(offer.code.len(), 6);
        assert!(offer.code.chars().all(|c| c.is_ascii_digit()));
        assert!(offer.expires_at_ms > 1_000);
    }

    // -- unpair_peer ------------------------------------------------------

    #[test]
    fn unpair_peer_an_unknown_peer_id_not_found_nothing_written() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let peers_path = dir.path().join("peers.json");
        let peers = std::sync::Mutex::new(vec![peer("peer-1")]);

        // Act
        let result = unpair_peer_via(&peers_path, &peers, "peer-unknown");

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::NotFound);
        assert!(!peers_path.exists());
        assert_eq!(peers.lock().unwrap().len(), 1);
    }

    #[test]
    fn unpair_peer_a_known_peer_id_removes_it_and_persists_the_shortened_list() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let peers_path = dir.path().join("peers.json");
        let peers = std::sync::Mutex::new(vec![peer("peer-1"), peer("peer-2")]);

        // Act
        let result = unpair_peer_via(&peers_path, &peers, "peer-1");

        // Assert
        assert!(result.is_ok());
        assert_eq!(peers.lock().unwrap().len(), 1);
        let persisted = idl_transport::sync::load_peers(&peers_path).unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].peer_id, "peer-2");
    }

    // -- set_sync_device_name ---------------------------------------------

    #[test]
    fn set_sync_device_name_blank_is_invalid_argument_nothing_written() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let identity_path = dir.path().join("identity.json");
        let name_lock = std::sync::Mutex::new("idl1".to_string());

        // Act
        let result = set_sync_device_name_via(&identity_path, "peer-1", &name_lock, "   ".to_string());

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::InvalidArgument);
        assert!(!identity_path.exists());
        assert_eq!(*name_lock.lock().unwrap(), "idl1");
    }

    #[test]
    fn set_sync_device_name_persists_the_trimmed_name_and_updates_the_live_lock() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let identity_path = dir.path().join("identity.json");
        let name_lock = std::sync::Mutex::new("idl1".to_string());

        // Act
        let result = set_sync_device_name_via(&identity_path, "peer-1", &name_lock, "  Pit Wall Laptop  ".to_string());

        // Assert
        assert_eq!(result.unwrap(), "Pit Wall Laptop");
        assert_eq!(*name_lock.lock().unwrap(), "Pit Wall Laptop");
        let persisted = idl_transport::sync::identity::load_or_create(&identity_path, None).unwrap();
        assert_eq!(persisted.peer_id, "peer-1");
        assert_eq!(persisted.name, "Pit Wall Laptop");
    }

    // -- pair_peer --------------------------------------------------------

    #[test]
    fn pair_peer_a_malformed_code_invalid_argument_no_request_sent() {
        // Arrange / Act — too short, and not sent to any peer: this
        // validator runs before `pair_peer` ever locks `state.discovered`
        // or builds a request.
        let result = validate_pairing_code("12a45");

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::InvalidArgument);
    }

    #[test]
    fn pair_peer_resolve_discovered_addr_an_unseen_peer_id_not_found() {
        // Arrange
        let discovered = HashMap::new();

        // Act
        let result = resolve_discovered_addr(&discovered, "peer-unseen");

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::NotFound);
    }

    // -- sync_now -----------------------------------------------------------

    #[test]
    fn sync_now_an_unknown_peer_id_the_not_found_kind() {
        // Arrange
        let peers: Vec<Peer> = Vec::new();
        let discovered = HashMap::new();

        // Act
        let result = resolve_peer_and_addr("peer-unknown", &peers, &discovered);

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::NotFound);
    }

    #[test]
    fn sync_now_a_paired_peer_not_currently_online_the_not_found_kind() {
        // Arrange — known to `peers` but absent from `discovered`: "unseen"
        // per this task's brief, distinct from "unknown" above but the
        // same `not_found` kind either way.
        let peers = vec![peer("peer-1")];
        let discovered = HashMap::new();

        // Act
        let result = resolve_peer_and_addr("peer-1", &peers, &discovered);

        // Assert
        assert_eq!(result.unwrap_err().kind, IpcErrorKind::NotFound);
    }

    // -- should_auto_sync -------------------------------------------------

    fn candidate(peer_id: &str) -> AutoSyncCandidate<'_> {
        AutoSyncCandidate { peer_id, paired: true, protocol_version: idl_transport::sync::PROTOCOL_VERSION }
    }

    #[test]
    fn should_auto_sync_an_unpaired_peer_no() {
        // Arrange
        let mut c = candidate("peer-1");
        c.paired = false;
        let running = HashSet::new();

        // Act
        let result = should_auto_sync(&c, None, 100_000, &running);

        // Assert
        assert!(!result);
    }

    #[test]
    fn should_auto_sync_a_paired_peer_last_synced_10_s_ago_no() {
        // Arrange
        let c = candidate("peer-1");
        let running = HashSet::new();
        let now = 100_000;
        let last = now - 10_000;

        // Act
        let result = should_auto_sync(&c, Some(last), now, &running);

        // Assert
        assert!(!result);
    }

    #[test]
    fn should_auto_sync_a_paired_peer_last_synced_90_s_ago_yes() {
        // Arrange
        let c = candidate("peer-1");
        let running = HashSet::new();
        let now = 100_000;
        let last = now - 90_000;

        // Act
        let result = should_auto_sync(&c, Some(last), now, &running);

        // Assert
        assert!(result);
    }

    #[test]
    fn should_auto_sync_a_peer_with_a_running_manual_sync_no() {
        // Arrange
        let c = candidate("peer-1");
        let mut running = HashSet::new();
        running.insert("peer-1".to_string());

        // Act
        let result = should_auto_sync(&c, None, 100_000, &running);

        // Assert
        assert!(!result);
    }

    #[test]
    fn should_auto_sync_an_incompatible_protocol_version_no() {
        // Arrange
        let mut c = candidate("peer-1");
        c.protocol_version = idl_transport::sync::PROTOCOL_VERSION + 1;
        let running = HashSet::new();

        // Act
        let result = should_auto_sync(&c, None, 100_000, &running);

        // Assert
        assert!(!result);
    }

    // -- DTOs: field names exactly match C3 §3.9 -------------------------

    #[test]
    fn dtos_serialise_with_exactly_c3_3_9s_field_names() {
        // Arrange
        let peer_status = PeerStatusDto { peer_id: "p1".to_string(), name: "Pit Tablet".to_string(), online: true, protocol_version: 1, paired_at_ms: 42 };
        let this_device = ThisDeviceDto { peer_id: "self".to_string(), name: "idl1".to_string() };
        let sync_status = SyncStatusDto { paired_peers: vec![peer_status.clone()], discovered_peers: Vec::new(), last_sync_utc_ms: Some(7), this_device };
        let sync_result = SyncResultDto { blobs_transferred: 1, workbooks_merged: 2, conflicts: 3, sessions_updated: 4, tracks_updated: 5, profiles_updated: 6 };
        let pairing_offer = PairingOfferDto { code: "004200".to_string(), expires_at_ms: 99 };

        // Act
        let peer_json = serde_json::to_value(&peer_status).unwrap();
        let status_json = serde_json::to_value(&sync_status).unwrap();
        let result_json = serde_json::to_value(&sync_result).unwrap();
        let offer_json = serde_json::to_value(&pairing_offer).unwrap();

        // Assert
        let mut peer_keys: Vec<&str> = peer_json.as_object().unwrap().keys().map(String::as_str).collect();
        peer_keys.sort();
        assert_eq!(peer_keys, vec!["name", "online", "paired_at_ms", "peer_id", "protocol_version"]);

        let mut status_keys: Vec<&str> = status_json.as_object().unwrap().keys().map(String::as_str).collect();
        status_keys.sort();
        assert_eq!(status_keys, vec!["discovered_peers", "last_sync_utc_ms", "paired_peers", "this_device"]);

        let mut result_keys: Vec<&str> = result_json.as_object().unwrap().keys().map(String::as_str).collect();
        result_keys.sort();
        assert_eq!(
            result_keys,
            vec!["blobs_transferred", "conflicts", "profiles_updated", "sessions_updated", "tracks_updated", "workbooks_merged"]
        );

        let mut offer_keys: Vec<&str> = offer_json.as_object().unwrap().keys().map(String::as_str).collect();
        offer_keys.sort();
        assert_eq!(offer_keys, vec!["code", "expires_at_ms"]);
    }
}
