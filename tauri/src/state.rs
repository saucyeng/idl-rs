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
/// entry stops the watcher (`WorkbookWatcher`'s `notify` field tears down
/// its handle on drop). `unwatch_workbook` removes an entry by id;
/// re-subscribing to the same id also replaces — and so stops — the
/// previous one, which stays the backstop for a frontend that never calls
/// unwatch (Tauri v2 gives no channel-close signal this task can observe).
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

/// The library-wide lap/track index job (rulings R207, R208 item 1): at
/// most one run at a time, its live progress, and the flag that stops it.
///
/// Managed state rather than a thread handle, because what the UI needs is
/// the *state* — a chip mounting halfway through a ten-minute run has to be
/// able to ask `index_status()` what is happening. The thread itself is
/// detached; it ends when the job finishes or the cancel flag is set, and a
/// process that exits mid-run loses at most the session in flight (every
/// finished session is already committed).
#[derive(Default)]
pub struct IndexJob {
    /// Live progress; see [`IndexJob::snapshot`].
    progress: Mutex<IndexJobProgress>,
    /// Polled by the core job between sessions and phases. Cleared by
    /// [`IndexJob::try_claim`] when a new run starts.
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
}

/// [`IndexJob`]'s mutable half, behind its mutex.
#[derive(Default)]
struct IndexJobProgress {
    running: bool,
    done: usize,
    total: usize,
    current_session_id: Option<String>,
    phase: Option<&'static str>,
    last_run: Option<crate::commands::index::IndexRunSummary>,
    last_error: Option<crate::error::IpcError>,
}

impl IndexJob {
    fn lock(&self) -> std::sync::MutexGuard<'_, IndexJobProgress> {
        self.progress.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Claims the single run slot: `true` when this caller may start a job,
    /// `false` when one is already in flight. Clears the cancel flag on a
    /// successful claim, so a run cancelled earlier does not stop the next
    /// one before it begins.
    pub fn try_claim(&self) -> bool {
        let mut progress = self.lock();
        if progress.running {
            return false;
        }
        progress.running = true;
        progress.done = 0;
        progress.total = 0;
        progress.current_session_id = None;
        progress.phase = None;
        progress.last_error = None;
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Records one `index_progress` observation from a worker thread.
    pub fn observe(&self, p: &idl_rs::store::index_job::IndexProgress) {
        let mut progress = self.lock();
        progress.done = p.done;
        progress.total = p.total;
        progress.current_session_id = Some(p.current_session_id.clone());
        progress.phase = Some(p.phase.as_str());
    }

    /// Releases the run slot and records how the run ended: either its
    /// counts (`report`) or the setup failure that stopped it before any
    /// session was considered (`error`). Exactly one of the two is `Some`.
    ///
    /// A background job has no promise to reject, so `error` is the only
    /// place a broken data root becomes visible — without it, an unreadable
    /// `<data>/tracks/` would be indistinguishable from "nothing to do"
    /// (CLAUDE.md §5).
    pub fn finish(
        &self,
        report: Option<&idl_rs::store::index_job::IndexJobReport>,
        error: Option<&crate::error::IpcError>,
    ) {
        let mut progress = self.lock();
        progress.running = false;
        progress.current_session_id = None;
        progress.phase = None;
        progress.last_error = error.cloned();
        if let Some(report) = report {
            progress.done = report.total;
            progress.total = report.total;
            progress.last_run = Some(crate::commands::index::IndexRunSummary::from(report));
        }
    }

    /// The C3 §3.2 `index_status()` value.
    pub fn snapshot(&self) -> crate::commands::index::IndexStatus {
        let progress = self.lock();
        crate::commands::index::IndexStatus {
            running: progress.running,
            done: progress.done,
            total: progress.total,
            current_session_id: progress.current_session_id.clone(),
            phase: progress.phase.map(str::to_string),
            last_run: progress.last_run.clone(),
            last_error: progress.last_error.clone(),
        }
    }
}

/// The background catalog rebuild (ruling R219 items 2–3): at most one run
/// at a time, and its live progress.
///
/// Managed state rather than a thread handle, for [`IndexJob`]'s reason: a
/// chip mounting halfway through a rebuild has to be able to ask
/// `rebuild_status()` what is happening. The thread is detached; a process
/// that exits mid-run loses the staging database in `tmp/` and nothing else,
/// since the swap onto `catalog.sqlite` is the last thing a run does (C4 §5).
///
/// There is no cancel flag here, unlike `IndexJob`: the index job commits
/// per session, so stopping it keeps what it has, whereas a rebuild's only
/// commit is its final atomic swap — stopping one early would throw away the
/// whole run's work, so the run is left to finish.
#[derive(Default)]
pub struct RebuildJob {
    progress: Mutex<RebuildJobProgress>,
}

/// [`RebuildJob`]'s mutable half, behind its mutex.
#[derive(Default)]
struct RebuildJobProgress {
    running: bool,
    done: usize,
    total: usize,
    phase: Option<&'static str>,
    last_run: Option<crate::commands::rebuild::RebuildRunSummary>,
    last_error: Option<crate::error::IpcError>,
}

impl RebuildJob {
    fn lock(&self) -> std::sync::MutexGuard<'_, RebuildJobProgress> {
        self.progress.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Claims the single run slot: `true` when this caller may start a
    /// rebuild, `false` when one is already in flight.
    pub fn try_claim(&self) -> bool {
        let mut progress = self.lock();
        if progress.running {
            return false;
        }
        progress.running = true;
        progress.done = 0;
        progress.total = 0;
        progress.phase = None;
        progress.last_error = None;
        true
    }

    /// Records one `rebuild_progress` observation from the job thread.
    pub fn observe(&self, p: &idl_rs::store::catalog::RebuildProgress) {
        let mut progress = self.lock();
        progress.done = p.done;
        progress.total = p.total;
        progress.phase = Some(p.phase.as_str());
    }

    /// Releases the run slot and records how the run ended: either its
    /// counts (`summary`) or the failure that stopped it (`error`). Exactly
    /// one of the two is `Some` — a background job has no promise to reject,
    /// so `error` is the only place a broken data root becomes visible
    /// (CLAUDE.md §5).
    pub fn finish(
        &self,
        summary: Option<crate::commands::rebuild::RebuildRunSummary>,
        error: Option<&crate::error::IpcError>,
    ) {
        let mut progress = self.lock();
        progress.running = false;
        progress.phase = None;
        progress.last_error = error.cloned();
        if summary.is_some() {
            progress.last_run = summary;
        }
    }

    /// The C3 §3.2 `rebuild_status()` value.
    pub fn snapshot(&self) -> crate::commands::rebuild::RebuildStatus {
        let progress = self.lock();
        crate::commands::rebuild::RebuildStatus {
            running: progress.running,
            done: progress.done,
            total: progress.total,
            phase: progress.phase.map(str::to_string),
            last_run: progress.last_run.clone(),
            last_error: progress.last_error.clone(),
        }
    }
}

/// The firmware/OTA state machine's current state (C3 §3.8, ruling R198).
/// One per app: only one device can be updated at a time, and the sequence
/// owns the BLE link while it runs. `push_firmware`/`confirm_firmware` write
/// it (and emit `ota_state_changed` with the same value); `ota_state` reads
/// it, so a UI mounting mid-flight sees where the sequence got to.
pub struct Ota(pub Mutex<crate::commands::firmware::OtaState>);

impl Default for Ota {
    fn default() -> Self {
        Self(Mutex::new(crate::commands::firmware::OtaState::Idle))
    }
}

/// LAN-sync managed state (C3 §3.9, PLAN §2 "server and discovery lifecycle
/// in state"): the running server, the browse task's latest peer set, the
/// pairing state, and the loaded paired-peer list. Constructed once at
/// startup (`SyncState::start`) and held for the app's lifetime; dropping it
/// stops serving and browsing.
pub struct SyncState {
    /// This instance's own stable id — advertised over mDNS and sent to a
    /// peer on `POST /pair`. Minted once into `identity.json` and never
    /// regenerated (ruling R105).
    pub peer_id: String,
    /// Display name advertised over mDNS and shown by a peer that pairs
    /// with us. Behind a `Mutex` (unlike `peer_id`, fixed for the process
    /// lifetime) because `set_sync_device_name` (C3 §3.9) changes it while
    /// the app runs.
    pub name: Mutex<String>,
    /// Where this device's own sync identity persists:
    /// `app_config_dir()/identity.json` (ruling R105) — never under
    /// `<data>`, so it never syncs. Read once at startup by
    /// [`Self::start`]; written again only by `set_sync_device_name`.
    pub identity_path: PathBuf,
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
    ///
    /// This device's own identity (`peer_id`/`name`) is read from
    /// `identity_path` here — minted and persisted on first launch, never
    /// re-minted after (ruling R105, L11 Task 13) — rather than passed in
    /// by the caller, so a corrupt `identity.json` fails `.setup()` the
    /// same explicit way a corrupt `settings.json` cannot (that file
    /// degrades to defaults by design; this one must not, or a fresh id
    /// would silently orphan every existing pairing).
    pub async fn start<R: tauri::Runtime>(
        app: tauri::AppHandle<R>,
        data_root: PathBuf,
        peers_path: PathBuf,
        identity_path: PathBuf,
    ) -> Result<Self, crate::error::IpcError> {
        use crate::error::IpcError;

        let identity = idl_transport::sync::identity::load_or_create(&identity_path, os_hostname().as_deref())
            .map_err(IpcError::from)?;
        let peer_id = identity.peer_id;
        let name = identity.name;

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

        Ok(Self {
            peer_id,
            name: Mutex::new(name),
            identity_path,
            peers_path,
            server,
            pairing,
            peers,
            discovered,
            last_sync,
            running,
            browse_task,
        })
    }
}

/// Best-effort OS hostname, used only to seed `identity.json`'s default
/// `name` on first launch (ruling R105) — never a hard requirement, since
/// [`idl_transport::sync::identity::load_or_create`] already falls back to
/// [`idl_transport::sync::DEFAULT_NAME`] on `None`. `std` only: Windows
/// always sets `COMPUTERNAME`; Unix has no equivalent environment
/// convention, so this shells out to the `hostname` command a terminal user
/// would run themselves, rather than adding a dependency for one lookup
/// (CLAUDE.md §1 judgment call — see this task's report).
fn os_hostname() -> Option<String> {
    if let Ok(name) = std::env::var("COMPUTERNAME") {
        let name = name.trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The background browse-consuming loop (PLAN §2's "the browse task's
/// latest peer set"): for every LAN sighting, refreshes `discovered` and
/// emits `peer_appeared` (C3 §3.9, widened by L11 Task 14 to any sighting,
/// paired or not) if — and only if — the sighting is new or differs from
/// what `discovered` already held for that `peer_id` (lead ruling R176).
/// A sighting byte-identical to the stored one is still inserted (refreshes
/// nothing observable, but keeps the map current) but emits nothing: the
/// underlying mDNS resolve re-fires far more often than a peer's actual
/// name/version/address/port changes, and re-announcing unchanged data on
/// every resolve would be a per-tick firehose into the frontend on any LAN
/// with several other devices on it. Also runs the pure `should_auto_sync`
/// decision (`commands::sync`) — starting a sync itself only when it says
/// yes, never from a command handler (this task's brief, "Do not"). Ends
/// when `browse()`'s receiver closes (the daemon shut down) — there is no
/// other exit, matching `SyncState`'s own lifetime.
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
        let previous = discovered.lock().unwrap_or_else(|e| e.into_inner()).insert(sighting.peer_id.clone(), sighting.clone());

        if should_emit_peer_appeared(previous.as_ref(), &sighting) {
            let peers_snapshot = peers.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let discovered_snapshot = discovered.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let payload = crate::commands::sync::peer_sighting_dto(&sighting, &peers_snapshot, &discovered_snapshot);
            let _ = app.emit("peer_appeared", payload);
        }

        let paired_peer = peers.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|p| p.peer_id == sighting.peer_id).cloned();
        let Some(paired_peer) = paired_peer else { continue };

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

/// Whether a LAN `sighting` should re-emit `peer_appeared`, given what
/// `discovered` already held for that `peer_id` immediately before this
/// sighting overwrote it (lead ruling R176, L11 Task 14). `true` for a
/// genuine appearance (`previous` is `None`) or a change in anything the
/// event's DTO carries (name, protocol_version, address, port — everything
/// `DiscoveredPeer`'s `PartialEq` compares); `false` for a byte-identical
/// re-resolve, so a peer that stays visible without changing does not
/// re-fire the event on every underlying mDNS resolve. Pure so this rule is
/// testable without the mDNS daemon or a Tauri `AppHandle`.
fn should_emit_peer_appeared(previous: Option<&idl_transport::sync::DiscoveredPeer>, sighting: &idl_transport::sync::DiscoveredPeer) -> bool {
    previous != Some(sighting)
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_transport::sync::DiscoveredPeer;

    // -- IndexJob (rulings R207/R208.1) ----------------------------------

    #[test]
    fn index_job_a_second_claim_while_one_run_is_in_flight_is_refused() {
        // Arrange
        let job = IndexJob::default();

        // Act
        let first = job.try_claim();
        let second = job.try_claim();

        // Assert
        assert!(first);
        assert!(!second);
        assert!(job.snapshot().running);
    }

    #[test]
    fn index_job_a_run_that_could_not_start_reports_its_error_not_silence() {
        // Arrange -- the failure mode a background job cannot reject a
        // promise for: an unreadable data root.
        let job = IndexJob::default();
        job.try_claim();
        let error = crate::error::IpcError::new(crate::error::IpcErrorKind::Io, "cannot read <data>/tracks/");

        // Act
        job.finish(None, Some(&error));

        // Assert
        let status = job.snapshot();
        assert!(!status.running);
        assert_eq!(status.last_error, Some(error));
        assert!(status.last_run.is_none());
    }

    #[test]
    fn index_job_a_new_run_clears_the_previous_runs_error_and_the_cancel_flag() {
        // Arrange -- a failed, cancelled run, then a fresh claim.
        let job = IndexJob::default();
        job.try_claim();
        job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        job.finish(None, Some(&crate::error::IpcError::new(crate::error::IpcErrorKind::Io, "gone")));

        // Act
        let claimed = job.try_claim();

        // Assert
        assert!(claimed);
        assert_eq!(job.snapshot().last_error, None);
        assert!(!job.cancel.load(std::sync::atomic::Ordering::Relaxed));
    }

    // -- RebuildJob (ruling R219 items 2-3) ------------------------------

    #[test]
    fn rebuild_job_a_second_claim_while_one_run_is_in_flight_is_refused() {
        // Arrange
        let job = RebuildJob::default();

        // Act
        let first = job.try_claim();
        let second = job.try_claim();

        // Assert
        assert!(first);
        assert!(!second);
        assert!(job.snapshot().running);
    }

    #[test]
    fn rebuild_job_a_run_that_failed_reports_its_error_and_keeps_no_summary() {
        // Arrange
        let job = RebuildJob::default();
        job.try_claim();
        let error = crate::error::IpcError::new(crate::error::IpcErrorKind::Io, "cannot read <data>/blobs/");

        // Act
        job.finish(None, Some(&error));

        // Assert
        let status = job.snapshot();
        assert!(!status.running);
        assert_eq!(status.last_error, Some(error));
        assert!(status.last_run.is_none());
    }

    #[test]
    fn rebuild_job_a_finished_run_keeps_its_counts_and_a_new_claim_clears_the_error() {
        // Arrange
        let job = RebuildJob::default();
        job.try_claim();
        job.finish(
            Some(crate::commands::rebuild::RebuildRunSummary { sessions_indexed: 159, blobs_carried: 158, ..Default::default() }),
            None,
        );
        job.try_claim();
        job.finish(None, Some(&crate::error::IpcError::new(crate::error::IpcErrorKind::Io, "gone")));

        // Act
        let claimed = job.try_claim();

        // Assert
        assert!(claimed);
        let status = job.snapshot();
        assert_eq!(status.last_error, None);
        assert_eq!(status.last_run.map(|r| (r.sessions_indexed, r.blobs_carried)), Some((159, 158)));
    }

    fn discovered_peer(addr: &str) -> DiscoveredPeer {
        DiscoveredPeer { peer_id: "peer-1".to_string(), name: "Pit Tablet".to_string(), protocol_version: 1, addr: addr.parse().unwrap() }
    }

    // -- should_emit_peer_appeared (lead ruling R176) --------------------

    #[test]
    fn should_emit_peer_appeared_a_genuine_first_appearance_yes() {
        // Arrange
        let sighting = discovered_peer("127.0.0.1:9000");

        // Act
        let result = should_emit_peer_appeared(None, &sighting);

        // Assert
        assert!(result);
    }

    #[test]
    fn should_emit_peer_appeared_seen_twice_unchanged_the_second_resolve_no() {
        // Arrange
        let first = discovered_peer("127.0.0.1:9000");
        let second = first.clone();

        // Act
        let first_result = should_emit_peer_appeared(None, &first);
        let second_result = should_emit_peer_appeared(Some(&first), &second);

        // Assert
        assert!(first_result);
        assert!(!second_result);
    }

    #[test]
    fn should_emit_peer_appeared_seen_twice_with_a_changed_address_both_times_yes() {
        // Arrange
        let first = discovered_peer("127.0.0.1:9000");
        let second = discovered_peer("127.0.0.1:9001");

        // Act
        let first_result = should_emit_peer_appeared(None, &first);
        let second_result = should_emit_peer_appeared(Some(&first), &second);

        // Assert
        assert!(first_result);
        assert!(second_result);
    }

    #[test]
    fn should_emit_peer_appeared_disappears_then_reappears_unchanged_yes() {
        // Arrange: `discovered` loses its entry between sightings (this
        // lane adds no disappearance event — sync_status polling is the
        // backstop — but the map entry itself can still be cleared, e.g. by
        // a future eviction), so the second sighting sees `previous: None`
        // again despite being identical in content to the first.
        let first = discovered_peer("127.0.0.1:9000");
        let reappearance = first.clone();

        // Act
        let result = should_emit_peer_appeared(None, &reappearance);

        // Assert
        assert!(result);
    }
}
