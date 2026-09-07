//! Tauri-managed shared state (`app.manage(...)`), read by commands via
//! `tauri::State<T>`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The resolved `<data>` root (C4 §1), computed once at startup by
/// `paths::resolve_data_dir` and managed for the app's lifetime.
pub struct DataDir(pub PathBuf);

/// The C4 §4 self-write-suppression set (`crate::watcher::ExpectedHashSet`),
/// shared between `save_workbook` (registers the app's own writes before the
/// atomic rename) and `watch_workbook` (suppresses them so an external-edit
/// callback never fires for the app's own save).
pub struct Hashes(pub Arc<crate::watcher::ExpectedHashSet>);

/// Live `watch_workbook` subscriptions, keyed by workbook id. Dropping the
/// entry stops the watcher (`WorkbookWatcher`'s `Drop` tears down its
/// `notify` handle). Re-subscribing to the same id replaces the previous
/// entry, so a frontend remount cannot leak watchers — Tauri v2 gives no
/// channel-close signal this task can observe, so there is no unsubscribe
/// command in wave 1 (see the CHANGELOG entry).
pub struct Watchers(pub Mutex<HashMap<String, crate::watcher::WorkbookWatcher>>);

/// Live managed BLE connections, keyed by `device_id` (C3 §3.8's
/// `connect_device`/`disconnect_device`/`device_status`). The outer
/// `std::sync::Mutex` guards only the map's shape (insert/remove/lookup —
/// short, synchronous critical sections); each connection's own `BtleplugBle`
/// sits behind an `Arc<tokio::sync::Mutex<_>>` so a command can clone the
/// `Arc` out, drop the outer lock, then hold the inner async lock across its
/// own `.await`s without blocking every other command touching the map.
/// `connect_device` inserts an entry and leaves the link open; `device_status`/
/// `device_control`/`pull_config` use it when present and otherwise
/// connect-act-disconnect (C3 §3.8).
pub struct Connections(
    pub Mutex<HashMap<String, Arc<tokio::sync::Mutex<idl_transport::ble_transport::BtleplugBle>>>>,
);

/// LAN-sync managed state (C3 §3.9, PLAN §2 "server and discovery lifecycle
/// in state"): the running server, the browse task's latest peer set, the
/// pairing state, and the loaded paired-peer list. Constructed once at
/// startup (`SyncState::start`) and held for the app's lifetime; dropping it
/// stops serving and browsing.
pub struct SyncState {
    /// This instance's own stable id — advertised over mDNS and sent to a
    /// peer on `POST /pair`.
    pub peer_id: String,
    /// Display name advertised over mDNS and shown by a peer that pairs
    /// with us.
    pub name: String,
    /// Where the paired-peer list persists: `app_config_dir()/peers.json`
    /// (PLAN §8 Q7) — never under `<data>`, so a peer's bearer token never
    /// syncs.
    pub peers_path: PathBuf,
    /// The running sync server. Held only to keep it alive — `SyncServer`'s
    /// own `Drop` stops serving.
    pub server: idl_transport::sync::SyncServer,
    /// This instance's outstanding pairing offer, if any — the offering
    /// side of `start_pairing`/`POST /pair`.
    pub pairing: Arc<Mutex<idl_transport::sync::PairingState>>,
    /// Paired peers, persisted to `peers_path` on every mutation
    /// (`pair_peer`/`unpair_peer`).
    pub peers: Arc<Mutex<Vec<idl_transport::sync::Peer>>>,
    /// The most recent LAN sighting of each peer id, refreshed by the
    /// background browse task (`sync_status`'s `online` and `sync_now`'s
    /// address resolution both read this).
    pub discovered: Arc<Mutex<HashMap<String, idl_transport::sync::DiscoveredPeer>>>,
    /// Wall-clock completion time of the last successful sync with each
    /// peer id, epoch milliseconds — read by the auto-trigger decision and
    /// folded into `sync_status`'s `last_sync_utc_ms`.
    pub last_sync: Arc<Mutex<HashMap<String, i64>>>,
    /// Peer ids with a sync currently in flight, manual or automatic — the
    /// auto-trigger decision never starts a second run for a peer already
    /// here (this task's brief, "Key logic").
    pub running: Arc<Mutex<HashSet<String>>>,
    /// Keeps the background mDNS-browse task alive; dropping `SyncState`
    /// aborts it. `None` when `browse()` failed to start (a discovery
    /// failure never blocks startup — this task's brief, "Key logic";
    /// every peer simply shows offline).
    pub browse_task: Option<tauri::async_runtime::JoinHandle<()>>,
}

impl SyncState {
    /// Starts the sync server, loads the paired-peer list from
    /// `peers_path`, and spawns the background mDNS-browse task (PLAN §2's
    /// "server and discovery lifecycle in state"). Called once from the
    /// app crate's `.setup()` hook, the same shape `paths::resolve_data_dir`
    /// already follows for `DataDir` — the Tauri-specific parts
    /// (`app.path().app_config_dir()`, `app.manage(...)`) stay in
    /// `app/src-tauri`, one line each.
    ///
    /// A `browse()` failure (no multicast on this network, a firewall)
    /// never blocks startup (this task's brief, "Key logic": "surfaced
    /// through `sync_status`... never a startup crash") — `browse_task` is
    /// simply `None` and every peer shows offline until the next launch on
    /// a working network.
    pub async fn start<R: tauri::Runtime>(
        app: tauri::AppHandle<R>,
        data_root: PathBuf,
        peers_path: PathBuf,
        peer_id: String,
        name: String,
    ) -> Result<Self, crate::error::IpcError> {
        use crate::error::IpcError;

        let loaded_peers = idl_transport::sync::load_peers(&peers_path).map_err(IpcError::from)?;
        let peers = Arc::new(Mutex::new(loaded_peers));
        let pairing = Arc::new(Mutex::new(idl_transport::sync::PairingState::default()));

        let config = idl_transport::sync::SyncServerConfig {
            data_root: data_root.clone(),
            port: 0,
            bind_addr: std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
            peer_id: peer_id.clone(),
            name: name.clone(),
        };
        let server = idl_transport::sync::SyncServer::start(config, pairing.clone(), peers.clone()).await.map_err(IpcError::from)?;

        let discovered = Arc::new(Mutex::new(HashMap::new()));
        let last_sync = Arc::new(Mutex::new(HashMap::new()));
        let running = Arc::new(Mutex::new(HashSet::new()));

        let browse_task = match idl_transport::sync::browse() {
            Ok(rx) => Some(tauri::async_runtime::spawn(run_discovery_loop(
                app,
                rx,
                data_root,
                peers.clone(),
                discovered.clone(),
                last_sync.clone(),
                running.clone(),
            ))),
            Err(_) => None,
        };

        Ok(Self { peer_id, name, peers_path, server, pairing, peers, discovered, last_sync, running, browse_task })
    }
}

/// The background browse-consuming loop (PLAN §2's "the browse task's
/// latest peer set"): for every LAN sighting, refreshes `discovered`,
/// emits `peer_appeared` (C3 §3.9) for a sighting that is already paired,
/// and runs the pure `should_auto_sync` decision (`commands::sync`) —
/// starting a sync itself only when it says yes, never from a command
/// handler (this task's brief, "Do not"). Ends when `browse()`'s receiver
/// closes (the daemon shut down) — there is no other exit, matching
/// `SyncState`'s own lifetime.
async fn run_discovery_loop<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    mut rx: tokio::sync::mpsc::Receiver<idl_transport::sync::DiscoveredPeer>,
    data_root: PathBuf,
    peers: Arc<Mutex<Vec<idl_transport::sync::Peer>>>,
    discovered: Arc<Mutex<HashMap<String, idl_transport::sync::DiscoveredPeer>>>,
    last_sync: Arc<Mutex<HashMap<String, i64>>>,
    running: Arc<Mutex<HashSet<String>>>,
) {
    use tauri::Emitter;

    while let Some(sighting) = rx.recv().await {
        discovered.lock().unwrap_or_else(|e| e.into_inner()).insert(sighting.peer_id.clone(), sighting.clone());

        let paired_peer = peers.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|p| p.peer_id == sighting.peer_id).cloned();
        let Some(paired_peer) = paired_peer else { continue };

        let discovered_snapshot = discovered.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let _ = app.emit("peer_appeared", crate::commands::sync::peer_status_dto(&paired_peer, &discovered_snapshot));

        let last = last_sync.lock().unwrap_or_else(|e| e.into_inner()).get(&sighting.peer_id).copied();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
        let running_snapshot = running.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let candidate = crate::commands::sync::AutoSyncCandidate {
            peer_id: &sighting.peer_id,
            paired: true,
            protocol_version: sighting.protocol_version,
        };
        if crate::commands::sync::should_auto_sync(&candidate, last, now, &running_snapshot) {
            let addr = sighting.addr;
            let data_root = data_root.clone();
            let peer = paired_peer.clone();
            let running = running.clone();
            let last_sync = last_sync.clone();
            tauri::async_runtime::spawn(async move {
                let no_progress = |_p: idl_transport::sync::SyncProgress| {};
                let _ = crate::commands::sync::run_one_sync(&running, &last_sync, &data_root, &peer, addr, &no_progress).await;
            });
        }
    }
}
