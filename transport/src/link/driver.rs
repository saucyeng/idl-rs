//! Runs a [`LinkMachine`] (SPEC §14b.3): performs its actions — network
//! requests through a platform [`NetworkBinder`], `/ping` and `/handoff`
//! over `ReqwestWifi`, the one pending timer — and feeds the outcomes back
//! as inputs. One task per logger, on whichever runtime spawned it (this
//! crate never creates a runtime).
//!
//! Callers talk to it through [`LinkHandle`]: report the device's mode, ask
//! for a verified base URL ([`LinkHandle::await_linked`], SPEC §14b.3's
//! 15 s operation gate), report successful operations as heartbeats.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use super::machine::{Action, FailReason, Input, LinkConfig, LinkMachine, LinkState, Transition};
use crate::wifi_transport::{ReqwestWifi, WifiTransport, DEVICE_AP_PASSWORD, DEVICE_BASE_URL};
use crate::{TransportError, TransportErrorKind};

/// SPEC §14b.3 operation gate: how long an operation waits for `linked`.
pub const OP_GATE: Duration = Duration::from_secs(15);

/// Brings the logger's network up and down (SPEC §6.2). Implementations
/// report outcomes as [`Input::Available`]/[`Input::Lost`]/
/// [`Input::Unavailable`] on `events`, possibly long after `request`
/// returns. A new `request` supersedes any live one.
pub trait NetworkBinder: Send + Sync + 'static {
    /// Starts requesting the network for `ssid`. Returns once the request is
    /// made; `Err` means it could not even be made.
    fn request(
        &self,
        ssid: &str,
        password: &str,
        events: mpsc::UnboundedSender<Input>,
    ) -> impl Future<Output = Result<(), TransportError>> + Send;

    /// Drops the live request, if any. Idempotent.
    fn release(&self) -> impl Future<Output = ()> + Send;
}

/// Desktop's binder (SPEC §14b.1): the user joins the AP in the OS, so the
/// "network" is always up at `192.168.4.1`; `verifying` finds out whether
/// the user actually joined it.
pub struct DirectBinder;

impl NetworkBinder for DirectBinder {
    async fn request(
        &self,
        _ssid: &str,
        _password: &str,
        events: mpsc::UnboundedSender<Input>,
    ) -> Result<(), TransportError> {
        let _ = events.send(Input::Available { base_url: DEVICE_BASE_URL.to_string() });
        Ok(())
    }

    async fn release(&self) {}
}

/// Why an operation could not get a link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// The device is not in WiFi mode (nothing asked for the link).
    NotWifiMode,
    /// The link gave up; see the reason. Needs Retry.
    Failed(FailReason),
    /// Still converging after [`OP_GATE`].
    Timeout,
}

impl From<LinkError> for TransportError {
    fn from(e: LinkError) -> Self {
        let message = match e {
            LinkError::NotWifiMode => "the logger is not in WiFi mode".to_string(),
            LinkError::Failed(FailReason::GaveUp) => "could not link to the logger over WiFi".to_string(),
            LinkError::Failed(FailReason::WrongDevice { found }) => {
                format!("the WiFi network answered as {found}, not this logger")
            }
            LinkError::Failed(FailReason::ProtocolMismatch { proto }) => {
                format!("the logger speaks WiFi protocol {proto}: update its firmware")
            }
            LinkError::Timeout => "the WiFi link did not come up in time".to_string(),
        };
        TransportError::new(TransportErrorKind::Wifi, message)
    }
}

/// The caller's side of one running link.
pub struct LinkHandle {
    inputs: mpsc::UnboundedSender<Input>,
    /// Mirrors the machine's desired state as soon as the caller reports a
    /// mode change, so `await_linked` never mistakes the not-yet-processed
    /// `unlinked` for "not in WiFi mode".
    wanted: Arc<AtomicBool>,
    /// Inputs sent so far, and how many the driver has processed: an
    /// `await_linked` right after a mode change must judge the state that
    /// change produced, not a stale `failed` from the previous session.
    sent: Arc<AtomicU64>,
    processed: watch::Receiver<u64>,
    state: watch::Receiver<LinkState>,
    journal: Arc<StdMutex<Vec<Transition>>>,
    task: JoinHandle<()>,
}

impl LinkHandle {
    fn send(&self, input: Input) {
        self.sent.fetch_add(1, Ordering::SeqCst);
        let _ = self.inputs.send(input);
    }

    /// The device entered WiFi mode.
    pub fn wifi_mode_entered(&self) {
        self.wanted.store(true, Ordering::SeqCst);
        self.send(Input::WifiModeEntered);
    }

    /// The device left WiFi mode.
    pub fn wifi_mode_left(&self) {
        self.wanted.store(false, Ordering::SeqCst);
        self.send(Input::WifiModeLeft);
    }

    /// The user pressed Retry, or the app resumed.
    pub fn retry(&self) {
        self.send(Input::UserRetry);
    }

    /// An operation over the link succeeded (counts as a heartbeat).
    pub fn op_succeeded(&self) {
        self.send(Input::OpSucceeded);
    }

    /// The current state.
    pub fn state(&self) -> LinkState {
        self.state.borrow().clone()
    }

    /// A receiver that sees every state change.
    pub fn watch(&self) -> watch::Receiver<LinkState> {
        self.state.clone()
    }

    /// The journal, oldest first (SPEC §14b.3, C3 `link_journal`).
    pub fn journal(&self) -> Vec<Transition> {
        self.journal.lock().unwrap().clone()
    }

    /// SPEC §14b.3's operation gate: the verified base URL once `linked`,
    /// waiting up to `gate` while the link converges; fails fast when it
    /// has failed or nothing wants it.
    pub async fn await_linked(&self, gate: Duration) -> Result<String, LinkError> {
        if !self.wanted.load(Ordering::SeqCst) {
            return Err(LinkError::NotWifiMode);
        }
        let target = self.sent.load(Ordering::SeqCst);
        let mut processed = self.processed.clone();
        let mut state = self.state.clone();
        let wanted = self.wanted.clone();
        let settled = tokio::time::timeout(gate, async {
            let _ = processed.wait_for(|&n| n >= target).await;
            state
                .wait_for(|s| match s {
                    LinkState::Linked { .. } | LinkState::Failed { .. } => true,
                    LinkState::Unlinked => !wanted.load(Ordering::SeqCst),
                    _ => false,
                })
                .await
                .map(|s| s.clone())
        })
        .await;
        match settled {
            Ok(Ok(LinkState::Linked { base_url, .. })) => Ok(base_url),
            Ok(Ok(LinkState::Failed { reason })) => Err(LinkError::Failed(reason)),
            Ok(Ok(_)) | Ok(Err(_)) => Err(LinkError::NotWifiMode),
            Err(_) => Err(LinkError::Timeout),
        }
    }
}

impl Drop for LinkHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Starts the link for the logger named `name` on the current runtime.
pub fn spawn_link<B: NetworkBinder>(name: impl Into<String>, config: LinkConfig, binder: Arc<B>) -> LinkHandle {
    let name = name.into();
    let (inputs, rx) = mpsc::unbounded_channel();
    let (state_tx, state) = watch::channel(LinkState::Unlinked);
    let (processed_tx, processed) = watch::channel(0u64);
    let sent = Arc::new(AtomicU64::new(0));
    let journal = Arc::new(StdMutex::new(Vec::new()));
    let machine = LinkMachine::new(name.clone(), config);
    let driver = Driver { binder, state_tx, journal: journal.clone(), processed_tx };
    let task = tokio::spawn(run(machine, driver, rx));
    LinkHandle { inputs, wanted: Arc::new(AtomicBool::new(false)), sent, processed, state, journal, task }
}

/// What the driver task owns besides the machine.
struct Driver<B> {
    binder: Arc<B>,
    state_tx: watch::Sender<LinkState>,
    journal: Arc<StdMutex<Vec<Transition>>>,
    /// Count of caller inputs processed (see `LinkHandle::sent`).
    processed_tx: watch::Sender<u64>,
}

/// The driver loop. Caller inputs (`calls`) and the driver's own feedback —
/// binder events, ping results, timers (`feedback`) — arrive on separate
/// channels so only the former count toward `processed`.
async fn run<B: NetworkBinder>(
    mut machine: LinkMachine,
    driver: Driver<B>,
    mut calls: mpsc::UnboundedReceiver<Input>,
) {
    let Driver { binder, state_tx, journal, processed_tx } = driver;
    let (tx, mut feedback) = mpsc::unbounded_channel::<Input>();
    let name = machine.expected_name().to_string();
    let mut timer: Option<JoinHandle<()>> = None;
    let mut processed = 0u64;
    loop {
        let (input, from_caller) = tokio::select! {
            call = calls.recv() => match call {
                Some(input) => (input, true),
                None => break,
            },
            Some(input) = feedback.recv() => (input, false),
        };
        let actions = machine.handle(input);
        state_tx.send_if_modified(|s| {
            let changed = s != machine.state();
            if changed {
                *s = machine.state().clone();
            }
            changed
        });
        *journal.lock().unwrap() = machine.journal().cloned().collect();

        for action in actions {
            match action {
                // Binder calls run in order, inline: a Release spawned
                // beside a Request could land after it and kill it.
                Action::Request => {
                    if binder.request(&name, DEVICE_AP_PASSWORD, tx.clone()).await.is_err() {
                        let _ = tx.send(Input::Unavailable);
                    }
                }
                Action::Release => binder.release().await,
                Action::Ping { base_url } => {
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let input = match ReqwestWifi::new(base_url).ping().await {
                            Ok(ping) => Input::PingOk { device: ping.device, proto: ping.proto_version },
                            Err(_) => Input::PingFailed,
                        };
                        let _ = tx.send(input);
                    });
                }
                Action::Handoff { base_url } => {
                    tokio::spawn(async move {
                        if let Err(e) = ReqwestWifi::new(base_url).handoff().await {
                            eprintln!("POST /handoff failed (best effort): {e}");
                        }
                    });
                }
                Action::Arm(which, after) => {
                    if let Some(old) = timer.take() {
                        old.abort();
                    }
                    let tx = tx.clone();
                    timer = Some(tokio::spawn(async move {
                        tokio::time::sleep(after).await;
                        let _ = tx.send(Input::Timer(which));
                    }));
                }
                Action::Disarm => {
                    if let Some(old) = timer.take() {
                        old.abort();
                    }
                }
            }
        }
        if from_caller {
            processed += 1;
            let _ = processed_tx.send(processed);
        }
    }
    if let Some(old) = timer.take() {
        old.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::wifi_transport::integration::spawn_mock_server;

    /// A binder that answers every request with `available` at a mock
    /// server, counting requests and releases.
    struct MockBinder {
        base_url: String,
        requests: AtomicU32,
        releases: AtomicU32,
    }

    impl NetworkBinder for MockBinder {
        async fn request(
            &self,
            _ssid: &str,
            _password: &str,
            events: mpsc::UnboundedSender<Input>,
        ) -> Result<(), TransportError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            let _ = events.send(Input::Available { base_url: self.base_url.clone() });
            Ok(())
        }

        async fn release(&self) {
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn ping_body(device: &str) -> Vec<u8> {
        format!(r#"{{"device":"{device}","fw":"1.4.0","proto":1,"battery":90,"sd":"OK","mode":"wifi","ble":"on"}}"#)
            .into_bytes()
    }

    async fn mock_logger(device: &'static str, handoffs: Arc<AtomicU32>) -> Arc<MockBinder> {
        let (addr, _server) = spawn_mock_server(move |path, _, _| match path {
            "/ping" => (200, "OK", vec![], ping_body(device)),
            "/handoff" => {
                handoffs.fetch_add(1, Ordering::SeqCst);
                (200, "OK", vec![], b"ok".to_vec())
            }
            _ => (404, "Not Found", vec![], vec![]),
        })
        .await;
        Arc::new(MockBinder { base_url: format!("http://{addr}"), requests: AtomicU32::new(0), releases: AtomicU32::new(0) })
    }

    #[tokio::test]
    async fn await_linked_wifi_mode_entered_links_verifies_and_hands_off_once() {
        // Arrange
        let handoffs = Arc::new(AtomicU32::new(0));
        let binder = mock_logger("IDL0-A3F2", handoffs.clone()).await;
        let link = spawn_link("IDL0-A3F2", LinkConfig::default(), binder.clone());

        // Act
        link.wifi_mode_entered();
        let base_url = link.await_linked(Duration::from_secs(5)).await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Assert
        assert_eq!(base_url.unwrap(), binder.base_url);
        assert_eq!(handoffs.load(Ordering::SeqCst), 1);
        assert_eq!(binder.requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn await_linked_another_logger_answers_fails_wrong_device_and_releases() {
        // Arrange
        let binder = mock_logger("IDL0-B000", Arc::new(AtomicU32::new(0))).await;
        let link = spawn_link("IDL0-A3F2", LinkConfig::default(), binder.clone());

        // Act
        link.wifi_mode_entered();
        let result = link.await_linked(Duration::from_secs(5)).await;

        // Assert
        assert_eq!(result, Err(LinkError::Failed(FailReason::WrongDevice { found: "IDL0-B000".to_string() })));
        assert!(binder.releases.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn await_linked_not_in_wifi_mode_fails_fast() {
        // Arrange
        let binder = mock_logger("IDL0-A3F2", Arc::new(AtomicU32::new(0))).await;
        let link = spawn_link("IDL0-A3F2", LinkConfig::default(), binder);

        // Act
        let started = tokio::time::Instant::now();
        let result = link.await_linked(Duration::from_secs(5)).await;

        // Assert
        assert_eq!(result, Err(LinkError::NotWifiMode));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn wifi_mode_left_releases_and_journal_records_the_session() {
        // Arrange
        let binder = mock_logger("IDL0-A3F2", Arc::new(AtomicU32::new(0))).await;
        let link = spawn_link("IDL0-A3F2", LinkConfig::default(), binder.clone());
        link.wifi_mode_entered();
        link.await_linked(Duration::from_secs(5)).await.unwrap();

        // Act
        link.wifi_mode_left();
        let mut state = link.watch();
        state.wait_for(|s| *s == LinkState::Unlinked).await.unwrap();

        // Assert
        assert!(binder.releases.load(Ordering::SeqCst) >= 1);
        let names: Vec<_> = link.journal().iter().map(|t| t.to.name()).collect();
        assert_eq!(names, vec!["requesting", "verifying", "linked", "unlinked"]);
    }

    #[tokio::test]
    async fn direct_binder_nobody_joined_the_ap_gives_up_after_backoff() {
        // Arrange: nothing listens on this port, as on a desktop that never
        // joined the AP; tiny numbers so the whole schedule runs in ms.
        let config = LinkConfig {
            ping_retry: Duration::from_millis(1),
            verify_attempts: 2,
            backoff: vec![Duration::from_millis(1)],
            ..LinkConfig::default()
        };
        struct Nowhere;
        impl NetworkBinder for Nowhere {
            async fn request(&self, _: &str, _: &str, events: mpsc::UnboundedSender<Input>) -> Result<(), TransportError> {
                let _ = events.send(Input::Available { base_url: "http://127.0.0.1:9".to_string() });
                Ok(())
            }
            async fn release(&self) {}
        }
        let link = spawn_link("IDL0-A3F2", config, Arc::new(Nowhere));

        // Act
        link.wifi_mode_entered();
        let result = link.await_linked(Duration::from_secs(10)).await;

        // Assert
        assert_eq!(result, Err(LinkError::Failed(FailReason::GaveUp)));
    }

    #[tokio::test]
    async fn await_linked_after_a_failed_session_judges_the_new_session_not_the_stale_failure() {
        // Arrange: the first /ping answers as another logger, later ones
        // as the right one.
        let pings = Arc::new(AtomicU32::new(0));
        let counter = pings.clone();
        let (addr, _server) = spawn_mock_server(move |path, _, _| match path {
            "/ping" if counter.fetch_add(1, Ordering::SeqCst) == 0 => (200, "OK", vec![], ping_body("IDL0-B000")),
            "/ping" => (200, "OK", vec![], ping_body("IDL0-A3F2")),
            _ => (200, "OK", vec![], b"ok".to_vec()),
        })
        .await;
        let binder = Arc::new(MockBinder {
            base_url: format!("http://{addr}"),
            requests: AtomicU32::new(0),
            releases: AtomicU32::new(0),
        });
        let link = spawn_link("IDL0-A3F2", LinkConfig::default(), binder);
        link.wifi_mode_entered();
        assert!(matches!(link.await_linked(Duration::from_secs(5)).await, Err(LinkError::Failed(_))));

        // Act
        link.retry();
        let result = link.await_linked(Duration::from_secs(5)).await;

        // Assert
        assert!(result.is_ok(), "{result:?}");
    }
}
