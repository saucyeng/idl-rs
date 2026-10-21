//! One owner per logger: the link actor (logger link design,
//! `docs/superpowers/specs/2026-10-19-idl1-logger-link-DRAFT.md`).
//!
//! Each paired logger gets one task that owns its Bluetooth link: it
//! connects, keeps the link's status stream, notices the link dying (the
//! stream ends, or goes silent past the logger's 1 Hz publish), and
//! reconnects — quickly when the logger is expected back, then patiently.
//! Nothing else opens a link to that logger. Commands that need the link
//! (config, files, firmware, WiFi mode) borrow it through the shared
//! [`ConnectionMap`], as before; a WiFi session *lends* the link for its
//! length ([`Intent::Lend`]) and hands it back ([`Intent::Return`]), so the
//! actor never races it.
//!
//! The UI holds no connection state: it renders the [`LinkView`] stream
//! (`link_watch`) and sends intent (`link_pair`, `link_record`, …).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use idl_transport::ble_control::ControlCommand;
use idl_transport::ble_transport::BleTransport;
use idl_transport::TransportError;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::commands::device::{ConnectionInfo, ConnectionMap, DeviceStatus};
use crate::error::{IpcError, IpcErrorKind};

/// What a logger's link is doing (design §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkPhase {
    /// Wanted, not connected; trying.
    Seeking,
    /// A connect attempt is running.
    Connecting,
    /// Connected; status streaming.
    Ready,
    /// A WiFi session has the logger with Bluetooth still up.
    Wifi,
    /// A WiFi session has the logger and its Bluetooth is off (SPEC §10.4).
    HandedOff,
    /// The WiFi session ended; reconnecting.
    Returning,
    /// Unpaired: the actor has stopped. Sent once, then the view is gone.
    Forgotten,
}

/// One logger's link, as the UI sees it (design §4 `LinkView`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LinkView {
    /// The logger's Bluetooth id.
    pub device_id: String,
    /// What connect reported, once it has connected.
    pub info: Option<ConnectionInfo>,
    /// Phase.
    pub phase: LinkPhase,
    /// The last status the logger sent.
    pub status: Option<DeviceStatus>,
    /// When `status` arrived, ms since the Unix epoch.
    pub status_at_ms: Option<u64>,
    /// Record was asked for and has not been carried out yet.
    pub pending_record: bool,
    /// The last connect failure, cleared on connect.
    pub error: Option<IpcError>,
}

impl LinkView {
    fn new(device_id: &str) -> Self {
        Self { device_id: device_id.to_string(), info: None, phase: LinkPhase::Seeking, status: None, status_at_ms: None, pending_record: false, error: None }
    }
}

/// What the rest of the app asks of a link actor.
pub(crate) enum Intent {
    /// Unpair: disconnect and end the actor.
    Stop,
    /// Start (`true`) or stop recording, whatever the link is doing.
    Record { start: bool, done: oneshot::Sender<Result<(), IpcError>> },
    /// A WiFi session or firmware push is taking the link over: stand back
    /// (no reconnects, no silence checks) until [`Intent::Return`].
    Lend,
    /// What the borrower is doing with it, for the view.
    Note(LinkPhase),
    /// The borrower is done. `boot`: the logger is restarting (firmware
    /// push); `quick`: it is back on Bluetooth now (WiFi mode was closed).
    Return { boot: bool, quick: bool },
    /// Try now (app back in front, Bluetooth switched on).
    Kick,
    /// Answer with the connection once Ready, or the next failure.
    WaitReady(oneshot::Sender<Result<ConnectionInfo, IpcError>>),
}

/// Timing (design §6). Separate so tests run in paused time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinkTiming {
    /// Retry interval while the logger is expected back.
    pub fast_retry: Duration,
    /// Retry interval otherwise.
    pub slow_retry: Duration,
    /// How long "expected back" lasts after a link break.
    pub fast_window: Duration,
    /// The longest one connect attempt may take.
    pub connect_limit: Duration,
    /// No status for this long means the link is dead (the logger
    /// publishes at 1 Hz).
    pub status_silence: Duration,
    /// Wait before the first attempt after a WiFi session closed cleanly.
    pub returning_wait: Duration,
    /// Wait before the first attempt after a firmware push (reboot).
    pub boot_wait: Duration,
    /// The longest a borrower may keep the link before the actor takes it
    /// back.
    pub lend_limit: Duration,
    /// How often a Ready link sends `CMD_KEEPALIVE`, so the logger can drop
    /// a link whose app died (SPEC §7.2 link liveness, 2026-10-21).
    pub keepalive: Duration,
}

impl Default for LinkTiming {
    fn default() -> Self {
        Self {
            fast_retry: Duration::from_millis(500),
            slow_retry: Duration::from_secs(5),
            fast_window: Duration::from_secs(30),
            connect_limit: Duration::from_secs(10),
            status_silence: Duration::from_secs(6),
            returning_wait: Duration::from_millis(500),
            boot_wait: Duration::from_secs(4),
            lend_limit: Duration::from_secs(20 * 60),
            keepalive: Duration::from_secs(5),
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// What the actor does when Record arrives while the link is lent: ask the
/// WiFi session to end with Record (`POST /record`) instead of WiFi off.
pub(crate) type OnRecordWhileLent<'a> = &'a (dyn Fn() + Send + Sync);

/// Drops `link` from `connections` if it is still the entry there, and
/// closes it in the background (bounded): a dead link's close can wait on a
/// read that never returns.
fn retire<B: BleTransport>(connections: &ConnectionMap<B>, device_id: &str, link: &Arc<tokio::sync::Mutex<B>>) {
    let mut map = connections.lock().unwrap();
    if map.get(device_id).is_some_and(|l| Arc::ptr_eq(l, link)) {
        map.remove(device_id);
    }
}

/// The actor (design §4). Runs until [`Intent::Stop`] or every sender is
/// dropped. `new_ble` builds a fresh transport per attempt; `publish` gets
/// every view change; `on_record_while_lent` asks a running WiFi session to
/// end with Record.
pub(crate) async fn run_link<B, F, Fut>(
    device_id: String,
    connections: &ConnectionMap<B>,
    new_ble: F,
    mut intents: mpsc::UnboundedReceiver<Intent>,
    publish: &(dyn Fn(LinkView) + Send + Sync),
    on_record_while_lent: OnRecordWhileLent<'_>,
    timing: LinkTiming,
) where
    B: BleTransport,
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<B, TransportError>>,
{
    let mut view = LinkView::new(&device_id);
    let mut fast_until = Instant::now() + timing.fast_window;
    let mut first_wait = Duration::ZERO;
    let mut waiters: Vec<oneshot::Sender<Result<ConnectionInfo, IpcError>>> = Vec::new();
    let mut record_waiters: Vec<(bool, oneshot::Sender<Result<(), IpcError>>)> = Vec::new();
    publish(view.clone());

    'outer: loop {
        // ---- not connected: wait (taking intents meanwhile), then try ----
        let wake = Instant::now() + first_wait;
        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(wake) => break,
                intent = intents.recv() => match intent {
                    None | Some(Intent::Stop) => break 'outer,
                    Some(Intent::Kick) => {
                        fast_until = Instant::now() + timing.fast_window;
                        break;
                    }
                    Some(Intent::WaitReady(tx)) => waiters.push(tx),
                    Some(Intent::Record { start, done }) => {
                        view.pending_record = start;
                        publish(view.clone());
                        record_waiters.push((start, done));
                    }
                    Some(Intent::Lend) | Some(Intent::Note(_)) | Some(Intent::Return { .. }) => {}
                },
            }
        }

        // Adopt a link someone else already made (a firmware push's own
        // reconnect); otherwise connect.
        let adopted = connections.lock().unwrap().get(&device_id).cloned();
        let link = match adopted {
            Some(link) => link,
            None => {
                view.phase = LinkPhase::Connecting;
                publish(view.clone());
                match connect_once(&device_id, &new_ble, timing.connect_limit).await {
                    Ok((ble, info)) => {
                        view.info = Some(info);
                        let link = Arc::new(tokio::sync::Mutex::new(ble));
                        connections.lock().unwrap().insert(device_id.clone(), link.clone());
                        link
                    }
                    Err(e) => {
                        view.phase = LinkPhase::Seeking;
                        view.error = Some(e.clone());
                        publish(view.clone());
                        for tx in waiters.drain(..) {
                            let _ = tx.send(Err(e.clone()));
                        }
                        first_wait = if Instant::now() < fast_until { timing.fast_retry } else { timing.slow_retry };
                        continue 'outer;
                    }
                }
            }
        };

        // ---- connected ----
        let mut status_rx = match link.lock().await.watch_status().await {
            Ok(rx) => Some(rx),
            Err(_) => None,
        };
        if let Ok(status) = link.lock().await.read_status().await {
            view.status = Some(status.into());
            view.status_at_ms = Some(now_ms());
        }
        view.phase = LinkPhase::Ready;
        view.error = None;
        publish(view.clone());
        if let Some(info) = view.info.clone() {
            for tx in waiters.drain(..) {
                let _ = tx.send(Ok(info.clone()));
            }
        }
        for (start, done) in record_waiters.drain(..) {
            let already = view.status.as_ref().and_then(|s| s.logging) == Some(start);
            let result = if already { Ok(()) } else { send_record(&link, start).await };
            let _ = done.send(result);
        }
        view.pending_record = false;
        publish(view.clone());

        // Silence is measured from the last status only: a keepalive tick or
        // an intent waking the loop must not restart it, or a link whose
        // status stopped would never be noticed.
        let mut silent_at = Instant::now() + timing.status_silence;
        let mut keepalive = tokio::time::interval_at(Instant::now() + timing.keepalive, timing.keepalive);
        keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                status = async { match status_rx.as_mut() { Some(rx) => rx.recv().await, None => std::future::pending().await } } => match status {
                    Some(status) => {
                        silent_at = Instant::now() + timing.status_silence;
                        view.status = Some(status.into());
                        view.status_at_ms = Some(now_ms());
                        publish(view.clone());
                    }
                    None => {
                        // The radio says the link is gone.
                        lost(connections, &device_id, &link, &mut view, publish);
                        fast_until = Instant::now() + timing.fast_window;
                        first_wait = timing.fast_retry;
                        continue 'outer;
                    }
                },
                _ = keepalive.tick() => {
                    // "Still here" (SPEC §7.2). The result is ignored: older
                    // firmware rejects the byte, and a dead link is the
                    // silence check's job. Bounded, so a borrower holding the
                    // link can't stall the actor.
                    let _ = tokio::time::timeout(timing.keepalive, async {
                        link.lock().await.send_command(ControlCommand::Keepalive).await
                    })
                    .await;
                }
                _ = tokio::time::sleep_until(silent_at) => {
                    // Silent past the 1 Hz publish: dead, whatever the
                    // stack says.
                    lost(connections, &device_id, &link, &mut view, publish);
                    fast_until = Instant::now() + timing.fast_window;
                    first_wait = timing.fast_retry;
                    continue 'outer;
                }
                intent = intents.recv() => match intent {
                    None | Some(Intent::Stop) => {
                        retire(connections, &device_id, &link);
                        let _ = tokio::time::timeout(Duration::from_secs(5), async { link.lock().await.disconnect().await }).await;
                        break 'outer;
                    }
                    Some(Intent::Record { start, done }) => {
                        let _ = done.send(send_record(&link, start).await);
                    }
                    Some(Intent::WaitReady(tx)) => {
                        let _ = tx.send(view.info.clone().ok_or_else(|| IpcError::new(IpcErrorKind::Ble, "connected without connection info")));
                    }
                    Some(Intent::Kick) | Some(Intent::Note(_)) | Some(Intent::Return { .. }) => {}
                    Some(Intent::Lend) => {
                        // ---- lent to a WiFi session or firmware push ----
                        drop(status_rx.take());
                        view.phase = LinkPhase::Wifi;
                        publish(view.clone());
                        let deadline = Instant::now() + timing.lend_limit;
                        let (boot, quick) = loop {
                            tokio::select! {
                                _ = tokio::time::sleep_until(deadline) => break (false, false),
                                intent = intents.recv() => match intent {
                                    None | Some(Intent::Stop) => break 'outer,
                                    Some(Intent::Return { boot, quick }) => break (boot, quick),
                                    Some(Intent::Note(phase)) => {
                                        view.phase = phase;
                                        publish(view.clone());
                                    }
                                    Some(Intent::Record { start, done }) => {
                                        if start {
                                            on_record_while_lent();
                                        }
                                        view.pending_record = start;
                                        publish(view.clone());
                                        record_waiters.push((start, done));
                                    }
                                    Some(Intent::WaitReady(tx)) => waiters.push(tx),
                                    Some(Intent::Kick) | Some(Intent::Lend) => {}
                                },
                            }
                        };
                        // The borrower may have replaced or dropped the link.
                        retire(connections, &device_id, &link);
                        view.phase = LinkPhase::Returning;
                        publish(view.clone());
                        fast_until = Instant::now() + timing.fast_window;
                        first_wait = if boot {
                            timing.boot_wait
                        } else if quick {
                            timing.returning_wait
                        } else {
                            timing.fast_retry
                        };
                        if !quick && !boot {
                            // WiFi mode may still be up: the logger leaves it
                            // within a minute (design §7.2).
                            fast_until = Instant::now() + Duration::from_secs(75);
                        }
                        continue 'outer;
                    }
                },
            }
        }
    }

    view.phase = LinkPhase::Forgotten;
    publish(view);
}

/// One bounded connect attempt. A timed-out attempt is closed, so Android
/// doesn't keep a pending GATT client per try.
async fn connect_once<B, F, Fut>(device_id: &str, new_ble: &F, limit: Duration) -> Result<(B, ConnectionInfo), IpcError>
where
    B: BleTransport,
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<B, TransportError>>,
{
    let mut ble = new_ble().await.map_err(IpcError::from)?;
    match tokio::time::timeout(limit, ble.connect(device_id)).await {
        Ok(Ok(info)) => {
            crate::platform::remember_name(device_id, &info.name);
            Ok((ble, info.into()))
        }
        Ok(Err(e)) => {
            let _ = tokio::time::timeout(Duration::from_secs(5), ble.disconnect()).await;
            Err(IpcError::from(e))
        }
        Err(_) => {
            let _ = tokio::time::timeout(Duration::from_secs(5), ble.disconnect()).await;
            Err(IpcError::new(IpcErrorKind::Ble, format!("no answer from the logger in {} s", limit.as_secs())))
        }
    }
}

async fn send_record<B: BleTransport>(link: &Arc<tokio::sync::Mutex<B>>, start: bool) -> Result<(), IpcError> {
    let cmd = if start { ControlCommand::StartLogging } else { ControlCommand::StopLogging };
    link.lock().await.send_command(cmd).await.map_err(IpcError::from)
}

fn lost<B: BleTransport>(
    connections: &ConnectionMap<B>,
    device_id: &str,
    link: &Arc<tokio::sync::Mutex<B>>,
    view: &mut LinkView,
    publish: &(dyn Fn(LinkView) + Send + Sync),
) {
    retire(connections, device_id, link);
    view.phase = LinkPhase::Seeking;
    publish(view.clone());
}

/// Every link actor and the latest view of each (managed state).
#[derive(Default)]
pub struct Links {
    actors: StdMutex<HashMap<String, mpsc::UnboundedSender<Intent>>>,
    views: StdMutex<HashMap<String, LinkView>>,
    watchers: StdMutex<Vec<tauri::ipc::Channel<LinkView>>>,
    record_requests: StdMutex<HashSet<String>>,
}

impl Links {
    /// Records `view` and sends it to every watcher (dropping closed ones).
    pub(crate) fn publish(&self, view: LinkView) {
        if view.phase == LinkPhase::Forgotten {
            self.views.lock().unwrap().remove(&view.device_id);
        } else {
            self.views.lock().unwrap().insert(view.device_id.clone(), view.clone());
        }
        self.watchers.lock().unwrap().retain(|ch| ch.send(view.clone()).is_ok());
    }

    /// Sends `intent` to `device_id`'s actor. `false` when it has none.
    pub(crate) fn send(&self, device_id: &str, intent: Intent) -> bool {
        match self.actors.lock().unwrap().get(device_id) {
            Some(tx) => tx.send(intent).is_ok(),
            None => false,
        }
    }

    /// `true` once for each Record asked while a WiFi session had the
    /// link: the session should end with `POST /record`.
    pub(crate) fn take_record_request(&self, device_id: &str) -> bool {
        self.record_requests.lock().unwrap().remove(device_id)
    }

    fn has(&self, device_id: &str) -> bool {
        self.actors.lock().unwrap().contains_key(device_id)
    }
}

/// Starts `device_id`'s actor on the platform transport, if it has none.
pub(crate) fn ensure_actor<R: tauri::Runtime>(app: &tauri::AppHandle<R>, device_id: &str) {
    use tauri::Manager;
    let links = app.state::<Links>();
    if links.has(device_id) {
        return;
    }
    let (tx, rx) = mpsc::unbounded_channel();
    links.actors.lock().unwrap().insert(device_id.to_string(), tx);
    let app = app.clone();
    let device_id = device_id.to_string();
    tauri::async_runtime::spawn(async move {
        let connections = app.state::<crate::state::Connections>();
        let links = app.state::<Links>();
        let transfers = app.state::<crate::state::WifiTransfers>();
        let publish = |view: LinkView| links.publish(view);
        let on_record = || {
            links.record_requests.lock().unwrap().insert(device_id.clone());
            if let Some(cancel) = transfers.0.lock().unwrap().get(&device_id) {
                cancel.notify_one();
            }
        };
        run_link(
            device_id.clone(),
            &connections.0,
            || crate::platform::PlatformBle::new(),
            rx,
            &publish,
            &on_record,
            LinkTiming::default(),
        )
        .await;
        links.actors.lock().unwrap().remove(&device_id);
    });
}

/// C3 §3.8 `link_watch(on_view)` (added 2026-10-19): every logger's current
/// [`LinkView`] now, then each change as it happens.
#[tauri::command]
pub fn link_watch(links: tauri::State<'_, Links>, on_view: tauri::ipc::Channel<LinkView>) {
    for view in links.views.lock().unwrap().values() {
        let _ = on_view.send(view.clone());
    }
    links.watchers.lock().unwrap().push(on_view);
}

/// C3 §3.8 `link_pair(device_id)` (added 2026-10-19): the app wants this
/// logger from now on; its actor connects and keeps it connected.
#[tauri::command]
pub fn link_pair<R: tauri::Runtime>(app: tauri::AppHandle<R>, device_id: String) {
    ensure_actor(&app, &device_id);
}

/// C3 §3.8 `link_forget(device_id)` (added 2026-10-19): the app no longer
/// wants this logger; its actor disconnects and stops.
#[tauri::command]
pub fn link_forget(links: tauri::State<'_, Links>, device_id: String) {
    links.send(&device_id, Intent::Stop);
}

/// C3 §3.8 `link_kick()` (added 2026-10-19): every actor that isn't
/// connected tries now (the app came to the front, Bluetooth came on).
#[tauri::command]
pub fn link_kick(links: tauri::State<'_, Links>) {
    let ids: Vec<String> = links.actors.lock().unwrap().keys().cloned().collect();
    for id in ids {
        links.send(&id, Intent::Kick);
    }
}

/// C3 §3.8 `link_record(device_id, start)` (added 2026-10-19): start or
/// stop recording whatever the link is doing — sent now when connected; a
/// running WiFi session ends with `POST /record`; otherwise sent the moment
/// the link is back.
#[tauri::command(async)]
pub async fn link_record(links: tauri::State<'_, Links>, device_id: String, start: bool) -> Result<(), IpcError> {
    let (tx, rx) = oneshot::channel();
    if !links.send(&device_id, Intent::Record { start, done: tx }) {
        return Err(IpcError::new(IpcErrorKind::NotFound, "that logger isn't paired"));
    }
    rx.await.unwrap_or_else(|_| Err(IpcError::new(IpcErrorKind::NotFound, "the logger was unpaired")))
}

/// Waits (up to `limit`) for `device_id`'s actor to be connected; starts
/// the actor if it has none. What `connect_device` is now.
pub(crate) async fn wait_ready<R: tauri::Runtime>(app: &tauri::AppHandle<R>, device_id: &str, limit: Duration) -> Result<ConnectionInfo, IpcError> {
    use tauri::Manager;
    ensure_actor(app, device_id);
    let (tx, rx) = oneshot::channel();
    app.state::<Links>().send(device_id, Intent::WaitReady(tx));
    match tokio::time::timeout(limit, rx).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(IpcError::new(IpcErrorKind::NotFound, "the logger was unpaired")),
        Err(_) => Err(IpcError::new(IpcErrorKind::Ble, "the logger didn't connect — is it on and close by?")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::device::tests::StubBle;
    use std::sync::atomic::Ordering;

    fn timing() -> LinkTiming {
        LinkTiming::default()
    }

    /// A stub logger that accepts the connection.
    fn reachable() -> StubBle {
        StubBle {
            connect_result: Ok(idl_transport::ConnectionInfo {
                device_id: "AA".to_string(),
                name: "IDL1-TEST".to_string(),
                firmware_version: "0.3.0".to_string(),
                connected: true,
            }),
            ..Default::default()
        }
    }

    async fn settle() {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn run_link_reachable_logger_connects_and_publishes_ready() {
        // Arrange
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let views = StdMutex::new(Vec::new());
        let publish = |v: LinkView| views.lock().unwrap().push(v.phase);
        let (tx, rx) = mpsc::unbounded_channel();

        // Act
        let actor = run_link("AA".into(), &connections, || async { Ok(reachable()) }, rx, &publish, &|| {}, timing());
        let drive = async {
            settle().await;
            let _ = tx.send(Intent::Stop);
        };
        tokio::join!(actor, drive);

        // Assert
        let phases = views.lock().unwrap().clone();
        assert!(phases.contains(&LinkPhase::Ready), "{phases:?}");
        assert_eq!(phases.last(), Some(&LinkPhase::Forgotten));
    }

    #[tokio::test(start_paused = true)]
    async fn run_link_silent_link_is_dropped_and_reconnected() {
        // Arrange — the stub never sends status, so the link goes silent.
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let connects = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let views = StdMutex::new(Vec::new());
        let publish = |v: LinkView| views.lock().unwrap().push(v.phase);
        let (tx, rx) = mpsc::unbounded_channel();
        let c = connects.clone();
        let new_ble = move || {
            c.fetch_add(1, Ordering::SeqCst);
            async { Ok(reachable()) }
        };

        // Act — past one silence window and a fast retry.
        let actor = run_link("AA".into(), &connections, new_ble, rx, &publish, &|| {}, timing());
        let drive = async {
            settle().await;
            tokio::time::sleep(timing().status_silence + timing().fast_retry + Duration::from_millis(100)).await;
            settle().await;
            let _ = tx.send(Intent::Stop);
        };
        tokio::join!(actor, drive);

        // Assert
        assert!(connects.load(Ordering::SeqCst) >= 2);
        let phases = views.lock().unwrap().clone();
        assert!(phases.contains(&LinkPhase::Seeking), "{phases:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn run_link_ready_link_sends_keepalives_and_a_silent_one_is_still_dropped() {
        // Arrange — the stub never sends status.
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let views = StdMutex::new(Vec::new());
        let publish = |v: LinkView| views.lock().unwrap().push(v.phase);
        let (tx, rx) = mpsc::unbounded_channel();

        // Act — just past one keepalive period, then past the silence window.
        let actor = run_link("AA".into(), &connections, || async { Ok(reachable()) }, rx, &publish, &|| {}, timing());
        let drive = async {
            settle().await;
            tokio::time::sleep(timing().keepalive + Duration::from_millis(100)).await;
            settle().await;
            let link = connections.lock().unwrap().get("AA").cloned().expect("connected");
            let sent = {
                let ble = link.lock().await;
                (ble.send_command_calls.load(Ordering::SeqCst), *ble.last_command.lock().unwrap())
            };
            tokio::time::sleep(timing().status_silence - timing().keepalive).await;
            settle().await;
            let _ = tx.send(Intent::Stop);
            sent
        };
        let ((), sent) = tokio::join!(actor, drive);

        // Assert
        assert_eq!(sent, (1, Some(ControlCommand::Keepalive)));
        let phases = views.lock().unwrap().clone();
        let ready = phases.iter().position(|p| *p == LinkPhase::Ready).expect("ready");
        assert!(phases[ready..].contains(&LinkPhase::Seeking), "silent link dropped despite keepalives: {phases:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn run_link_lent_link_is_left_alone_until_returned() {
        // Arrange
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let connects = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let publish = |_v: LinkView| {};
        let (tx, rx) = mpsc::unbounded_channel();
        let c = connects.clone();
        let new_ble = move || {
            c.fetch_add(1, Ordering::SeqCst);
            async { Ok(reachable()) }
        };

        // Act — lent for far longer than the silence window.
        let actor = run_link("AA".into(), &connections, new_ble, rx, &publish, &|| {}, timing());
        let drive = async {
            settle().await;
            let _ = tx.send(Intent::Lend);
            settle().await;
            tokio::time::sleep(Duration::from_secs(120)).await;
            let during = connects.load(Ordering::SeqCst);
            let _ = tx.send(Intent::Return { boot: false, quick: true });
            settle().await;
            tokio::time::sleep(timing().returning_wait + Duration::from_millis(50)).await;
            settle().await;
            let _ = tx.send(Intent::Stop);
            during
        };
        let ((), during) = tokio::join!(actor, drive);

        // Assert
        assert_eq!(during, 1, "no reconnect while lent");
        assert!(connects.load(Ordering::SeqCst) >= 2, "reconnects once returned");
    }

    #[tokio::test(start_paused = true)]
    async fn run_link_record_while_lent_asks_the_session_to_end_with_record() {
        // Arrange
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let asked = std::sync::atomic::AtomicBool::new(false);
        let on_record = || asked.store(true, Ordering::SeqCst);
        let publish = |_v: LinkView| {};
        let (tx, rx) = mpsc::unbounded_channel();

        // Act
        let actor = run_link("AA".into(), &connections, || async { Ok(reachable()) }, rx, &publish, &on_record, timing());
        let drive = async {
            settle().await;
            let _ = tx.send(Intent::Lend);
            settle().await;
            let (done, _rx) = oneshot::channel();
            let _ = tx.send(Intent::Record { start: true, done });
            settle().await;
            let _ = tx.send(Intent::Stop);
        };
        tokio::join!(actor, drive);

        // Assert
        assert!(asked.load(Ordering::SeqCst));
    }
}
