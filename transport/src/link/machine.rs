//! The SPEC §6.2 / §14b.3 transition table. See the module doc in `mod.rs`.

use std::collections::VecDeque;
use std::time::Duration;

/// SPEC §14b.3: how many transitions the journal keeps.
const JOURNAL_LEN: usize = 100;

/// SPEC §6.1 WiFi control-protocol major this app speaks.
const SUPPORTED_PROTO: u32 = 1;

/// The numbers SPEC §14b.3 fixes. `Default` is exactly that section.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkConfig {
    /// `requesting`: how long to wait for the network to come up.
    pub request_budget: Duration,
    /// `verifying`: gap between `/ping` attempts while the route warms up.
    pub ping_retry: Duration,
    /// `verifying`: `/ping` attempts before the request counts as failed
    /// (10 × 500 ms = SPEC's 5 s).
    pub verify_attempts: u32,
    /// `linked`: `/ping` cadence while idle.
    pub heartbeat: Duration,
    /// `linked`: consecutive heartbeat failures that force a relink.
    pub heartbeat_misses: u32,
    /// Waits before each re-request; its length bounds the retries before
    /// `failed`.
    pub backoff: Vec<Duration>,
    /// Network requests allowed per arming before `failed` regardless of
    /// `backoff` — Android 10 (API 29) may re-prompt on every request, so it
    /// gets `Some(2)` (one automatic retry). `None` elsewhere.
    pub max_requests: Option<u32>,
}

impl Default for LinkConfig {
    fn default() -> Self {
        Self {
            request_budget: Duration::from_secs(45),
            ping_retry: Duration::from_millis(500),
            verify_attempts: 10,
            heartbeat: Duration::from_secs(10),
            heartbeat_misses: 3,
            backoff: vec![Duration::from_secs(1), Duration::from_secs(2), Duration::from_secs(4)],
            max_requests: None,
        }
    }
}

/// Why the link gave up. `failed` waits for [`Input::UserRetry`] or WiFi-mode
/// re-entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailReason {
    /// Every allowed request/verify attempt failed.
    GaveUp,
    /// `/ping` answered as another logger (SPEC §6.1: every AP is
    /// `192.168.4.1`) — never talk to it.
    WrongDevice {
        /// The name `/ping` reported.
        found: String,
    },
    /// `/ping`'s `proto` major is not one this app speaks; the firmware
    /// needs an update.
    ProtocolMismatch {
        /// The `proto` `/ping` reported.
        proto: u32,
    },
}

/// SPEC §6.2's states. `attempt` counts network requests since the link was
/// last armed, from 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkState {
    /// Not wanted, or not yet asked for.
    Unlinked,
    /// Waiting for the platform binder's `available`.
    Requesting { attempt: u32 },
    /// Network up; checking `/ping` (`pings` failed so far).
    Verifying { attempt: u32, pings: u32, base_url: String },
    /// Verified; heartbeating (`misses` consecutive heartbeat failures).
    Linked { base_url: String, misses: u32 },
    /// Waiting before re-request number `attempt + 1`.
    Backoff { attempt: u32 },
    /// Gave up; see the reason.
    Failed { reason: FailReason },
}

impl LinkState {
    /// The C3 `LinkState.state` spelling (SPEC §14b.5).
    pub fn name(&self) -> &'static str {
        match self {
            LinkState::Unlinked => "unlinked",
            LinkState::Requesting { .. } => "requesting",
            LinkState::Verifying { .. } => "verifying",
            LinkState::Linked { .. } => "linked",
            LinkState::Backoff { .. } => "backoff",
            LinkState::Failed { .. } => "failed",
        }
    }

    /// The device base URL, while one is known to route.
    pub fn base_url(&self) -> Option<&str> {
        match self {
            LinkState::Verifying { base_url, .. } | LinkState::Linked { base_url, .. } => Some(base_url),
            _ => None,
        }
    }
}

/// The one timer a state can have pending; firing comes back as
/// [`Input::Timer`]. A timer that doesn't match the current state is stale
/// and ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timer {
    /// `requesting` gave up waiting for `available`.
    RequestTimeout,
    /// `verifying`: time for the next `/ping`.
    PingRetry,
    /// `linked`: time for the heartbeat `/ping`.
    Heartbeat,
    /// `backoff` is over.
    Backoff,
}

/// Everything the machine reacts to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// The device entered WiFi mode (desired state becomes `linked`).
    WifiModeEntered,
    /// The device left WiFi mode (desired state becomes `unlinked`).
    WifiModeLeft,
    /// The binder's network is up and the device is at `base_url`.
    Available { base_url: String },
    /// The binder's network went away.
    Lost,
    /// The binder could not bring the network up (declined, out of range).
    Unavailable,
    /// A `/ping` answered.
    PingOk { device: String, proto: u32 },
    /// A `/ping` did not answer (or answered garbage).
    PingFailed,
    /// Some other operation over the link succeeded (counts as a heartbeat).
    OpSucceeded,
    /// A timer armed by [`Action::Arm`] fired.
    Timer(Timer),
    /// The user pressed Retry, or the app resumed.
    UserRetry,
}

/// What the caller must do after an input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Request the logger's network (the binder supersedes any live request).
    Request,
    /// Release the network request, if any.
    Release,
    /// `GET /ping` at `base_url`; report [`Input::PingOk`]/[`Input::PingFailed`].
    Ping { base_url: String },
    /// `POST /handoff` at `base_url` (SPEC §10.4; idempotent, best effort).
    Handoff { base_url: String },
    /// Arm `timer` to fire after the duration, replacing any pending timer.
    Arm(Timer, Duration),
    /// Cancel the pending timer, if any.
    Disarm,
}

/// One journal entry (SPEC §14b.3, C3 `link_journal`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    /// State before the input.
    pub from: LinkState,
    /// What happened.
    pub input: Input,
    /// State after it.
    pub to: LinkState,
}

/// The link to one logger. See the module doc.
pub struct LinkMachine {
    expected_name: String,
    config: LinkConfig,
    state: LinkState,
    /// Desired state: the device is in WiFi mode.
    wanted: bool,
    /// `/handoff` already sent this WiFi session.
    handed_off: bool,
    journal: VecDeque<Transition>,
}

impl LinkMachine {
    /// A machine for the logger named `expected_name` (its SSID, and what
    /// `/ping` must report), starting `unlinked`.
    pub fn new(expected_name: impl Into<String>, config: LinkConfig) -> Self {
        Self {
            expected_name: expected_name.into(),
            config,
            state: LinkState::Unlinked,
            wanted: false,
            handed_off: false,
            journal: VecDeque::with_capacity(JOURNAL_LEN),
        }
    }

    /// The current state.
    pub fn state(&self) -> &LinkState {
        &self.state
    }

    /// The name this machine links to.
    pub fn expected_name(&self) -> &str {
        &self.expected_name
    }

    /// The last (up to) 100 transitions, oldest first. Inputs that changed
    /// nothing are not recorded.
    pub fn journal(&self) -> impl Iterator<Item = &Transition> {
        self.journal.iter()
    }

    /// Feeds one input; returns the actions to perform, in order.
    pub fn handle(&mut self, input: Input) -> Vec<Action> {
        let from = self.state.clone();
        let actions = self.step(&input);
        if self.state != from {
            if self.journal.len() == JOURNAL_LEN {
                self.journal.pop_front();
            }
            self.journal.push_back(Transition { from, input, to: self.state.clone() });
        }
        actions
    }

    /// Enters `requesting` for request number `attempt`.
    fn request(&mut self, attempt: u32) -> Vec<Action> {
        self.state = LinkState::Requesting { attempt };
        vec![Action::Request, Action::Arm(Timer::RequestTimeout, self.config.request_budget)]
    }

    /// Request `attempt` failed: back off before the next one, or give up.
    fn fail_attempt(&mut self, attempt: u32) -> Vec<Action> {
        let next = attempt + 1;
        let capped = self.config.max_requests.is_some_and(|max| next >= max);
        match self.config.backoff.get(attempt as usize) {
            Some(&wait) if !capped => {
                self.state = LinkState::Backoff { attempt };
                vec![Action::Release, Action::Arm(Timer::Backoff, wait)]
            }
            _ => self.fail(FailReason::GaveUp),
        }
    }

    fn fail(&mut self, reason: FailReason) -> Vec<Action> {
        self.state = LinkState::Failed { reason };
        vec![Action::Disarm, Action::Release]
    }

    fn leave(&mut self) -> Vec<Action> {
        self.wanted = false;
        self.state = LinkState::Unlinked;
        vec![Action::Disarm, Action::Release]
    }

    /// Checks a `/ping` answer against the expected logger and protocol.
    fn check_ping(&self, device: &str, proto: u32) -> Result<(), FailReason> {
        if device != self.expected_name {
            Err(FailReason::WrongDevice { found: device.to_string() })
        } else if proto != SUPPORTED_PROTO {
            Err(FailReason::ProtocolMismatch { proto })
        } else {
            Ok(())
        }
    }

    /// The transition table. Every arm not listed leaves the state alone
    /// and does nothing (a stale timer, a late event from a released
    /// request, a repeated mode report).
    fn step(&mut self, input: &Input) -> Vec<Action> {
        use Input as I;
        use LinkState as S;

        match (self.state.clone(), input) {
            // Leaving WiFi mode ends the link from every state.
            (S::Unlinked, I::WifiModeLeft) => {
                self.wanted = false;
                vec![]
            }
            (_, I::WifiModeLeft) => self.leave(),

            // Arming: WiFi mode entered, or Retry while it's wanted.
            (S::Unlinked | S::Failed { .. }, I::WifiModeEntered) => {
                self.wanted = true;
                self.handed_off = false;
                self.request(0)
            }
            (S::Failed { .. }, I::UserRetry) if self.wanted => self.request(0),
            (S::Unlinked, I::UserRetry) if self.wanted => self.request(0),

            // requesting
            (S::Requesting { attempt }, I::Available { base_url }) => {
                self.state = S::Verifying { attempt, pings: 0, base_url: base_url.clone() };
                vec![Action::Disarm, Action::Ping { base_url: base_url.clone() }]
            }
            (S::Requesting { attempt }, I::Unavailable | I::Lost | I::Timer(Timer::RequestTimeout)) => {
                self.fail_attempt(attempt)
            }

            // verifying
            (S::Verifying { base_url, .. }, I::PingOk { device, proto }) => match self.check_ping(device, *proto) {
                Ok(()) => {
                    self.state = S::Linked { base_url: base_url.clone(), misses: 0 };
                    let mut actions = vec![Action::Disarm];
                    if !self.handed_off {
                        self.handed_off = true;
                        actions.push(Action::Handoff { base_url: base_url.clone() });
                    }
                    actions.push(Action::Arm(Timer::Heartbeat, self.config.heartbeat));
                    actions
                }
                Err(reason) => self.fail(reason),
            },
            (S::Verifying { attempt, pings, base_url }, I::PingFailed) => {
                if pings + 1 < self.config.verify_attempts {
                    self.state = S::Verifying { attempt, pings: pings + 1, base_url };
                    vec![Action::Arm(Timer::PingRetry, self.config.ping_retry)]
                } else {
                    self.fail_attempt(attempt)
                }
            }
            (S::Verifying { base_url, .. }, I::Timer(Timer::PingRetry)) => vec![Action::Ping { base_url }],
            (S::Verifying { attempt, .. }, I::Lost | I::Unavailable) => {
                let mut actions = vec![Action::Disarm];
                actions.extend(self.fail_attempt(attempt));
                actions
            }

            // linked
            (S::Linked { base_url, .. }, I::Timer(Timer::Heartbeat)) => vec![Action::Ping { base_url }],
            (S::Linked { base_url, .. }, I::PingOk { device, proto }) => match self.check_ping(device, *proto) {
                Ok(()) => {
                    self.state = S::Linked { base_url, misses: 0 };
                    vec![Action::Arm(Timer::Heartbeat, self.config.heartbeat)]
                }
                Err(reason) => self.fail(reason),
            },
            (S::Linked { base_url, .. }, I::OpSucceeded) => {
                self.state = S::Linked { base_url, misses: 0 };
                vec![Action::Arm(Timer::Heartbeat, self.config.heartbeat)]
            }
            (S::Linked { base_url, misses }, I::PingFailed) => {
                if misses + 1 < self.config.heartbeat_misses {
                    self.state = S::Linked { base_url, misses: misses + 1 };
                    vec![Action::Arm(Timer::Heartbeat, self.config.heartbeat)]
                } else {
                    let mut actions = vec![Action::Release];
                    actions.extend(self.request(0));
                    actions
                }
            }
            (S::Linked { .. }, I::Lost | I::Unavailable) => {
                let mut actions = vec![Action::Disarm, Action::Release];
                actions.extend(self.request(0));
                actions
            }

            // backoff
            (S::Backoff { attempt }, I::Timer(Timer::Backoff)) => self.request(attempt + 1),

            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: &str = "IDL0-A3F2";
    const URL: &str = "http://127.0.0.1:40000";

    fn machine() -> LinkMachine {
        LinkMachine::new(NAME, LinkConfig::default())
    }

    fn ok_ping() -> Input {
        Input::PingOk { device: NAME.to_string(), proto: 1 }
    }

    fn available() -> Input {
        Input::Available { base_url: URL.to_string() }
    }

    /// Drives a fresh machine into `state_name`.
    fn in_state(state_name: &str) -> LinkMachine {
        let mut m = machine();
        match state_name {
            "unlinked" => {}
            "requesting" => {
                m.handle(Input::WifiModeEntered);
            }
            "verifying" => {
                m.handle(Input::WifiModeEntered);
                m.handle(available());
            }
            "linked" => {
                m.handle(Input::WifiModeEntered);
                m.handle(available());
                m.handle(ok_ping());
            }
            "backoff" => {
                m.handle(Input::WifiModeEntered);
                m.handle(Input::Unavailable);
            }
            "failed" => {
                m.handle(Input::WifiModeEntered);
                m.handle(available());
                m.handle(Input::PingOk { device: "IDL0-OTHER".to_string(), proto: 1 });
            }
            other => panic!("no such state {other}"),
        }
        assert_eq!(m.state().name(), state_name);
        m
    }

    fn all_inputs() -> Vec<(&'static str, Input)> {
        vec![
            ("wifi_mode_entered", Input::WifiModeEntered),
            ("wifi_mode_left", Input::WifiModeLeft),
            ("available", available()),
            ("lost", Input::Lost),
            ("unavailable", Input::Unavailable),
            ("ping_ok", ok_ping()),
            ("ping_failed", Input::PingFailed),
            ("op_succeeded", Input::OpSucceeded),
            ("timer_request_timeout", Input::Timer(Timer::RequestTimeout)),
            ("timer_ping_retry", Input::Timer(Timer::PingRetry)),
            ("timer_heartbeat", Input::Timer(Timer::Heartbeat)),
            ("timer_backoff", Input::Timer(Timer::Backoff)),
            ("user_retry", Input::UserRetry),
        ]
    }

    /// SPEC §14b.3: the transition table, state × input → next state, with
    /// full coverage. `=` means "unchanged".
    #[test]
    fn transition_table_every_state_every_input_lands_where_the_spec_says() {
        // Arrange
        let table: &[(&str, [&str; 13])] = &[
            //                entered       left        avail        lost          unavail       ping_ok   ping_fail    op_ok  t_req        t_ping t_hb  t_back       retry
            ("unlinked",   ["requesting", "=",        "=",         "=",          "=",          "=",      "=",         "=",   "=",         "=",   "=",  "=",         "="]),
            ("requesting", ["=",          "unlinked", "verifying", "backoff",    "backoff",    "=",      "=",         "=",   "backoff",   "=",   "=",  "=",         "="]),
            ("verifying",  ["=",          "unlinked", "=",         "backoff",    "backoff",    "linked", "=",         "=",   "=",         "=",   "=",  "=",         "="]),
            ("linked",     ["=",          "unlinked", "=",         "requesting", "requesting", "=",      "=",         "=",   "=",         "=",   "=",  "=",         "="]),
            ("backoff",    ["=",          "unlinked", "=",         "=",          "=",          "=",      "=",         "=",   "=",         "=",   "=",  "requesting","="]),
            ("failed",     ["requesting", "unlinked", "=",         "=",          "=",          "=",      "=",         "=",   "=",         "=",   "=",  "=",         "requesting"]),
        ];

        for (state_name, expected_row) in table {
            for ((input_name, input), expected) in all_inputs().into_iter().zip(expected_row.iter()) {
                // Act
                let mut m = in_state(state_name);
                m.handle(input);

                // Assert
                let want = if *expected == "=" { *state_name } else { *expected };
                assert_eq!(m.state().name(), want, "{state_name} × {input_name}");
            }
        }
    }

    #[test]
    fn first_verified_ping_hands_off_once_per_wifi_session() {
        // Arrange
        let mut m = in_state("verifying");

        // Act
        let first = m.handle(ok_ping());
        m.handle(Input::Lost);
        m.handle(available());
        let after_relink = m.handle(ok_ping());

        // Assert
        assert!(first.contains(&Action::Handoff { base_url: URL.to_string() }));
        assert!(!after_relink.iter().any(|a| matches!(a, Action::Handoff { .. })));
    }

    #[test]
    fn wifi_mode_reentered_hands_off_again() {
        // Arrange
        let mut m = in_state("linked");
        m.handle(Input::WifiModeLeft);

        // Act
        m.handle(Input::WifiModeEntered);
        m.handle(available());
        let actions = m.handle(ok_ping());

        // Assert
        assert!(actions.iter().any(|a| matches!(a, Action::Handoff { .. })));
    }

    #[test]
    fn wrong_device_fails_and_releases_never_links() {
        // Arrange
        let mut m = in_state("verifying");

        // Act
        let actions = m.handle(Input::PingOk { device: "IDL0-B000".to_string(), proto: 1 });

        // Assert
        assert_eq!(m.state(), &LinkState::Failed { reason: FailReason::WrongDevice { found: "IDL0-B000".to_string() } });
        assert!(actions.contains(&Action::Release));
    }

    #[test]
    fn protocol_major_mismatch_fails_with_the_proto() {
        // Arrange
        let mut m = in_state("verifying");

        // Act
        m.handle(Input::PingOk { device: NAME.to_string(), proto: 2 });

        // Assert
        assert_eq!(m.state(), &LinkState::Failed { reason: FailReason::ProtocolMismatch { proto: 2 } });
    }

    #[test]
    fn verifying_ping_fails_nine_times_retries_tenth_fails_the_attempt() {
        // Arrange
        let mut m = in_state("verifying");

        // Act
        for _ in 0..9 {
            let actions = m.handle(Input::PingFailed);
            assert_eq!(actions, vec![Action::Arm(Timer::PingRetry, Duration::from_millis(500))]);
            assert_eq!(m.state().name(), "verifying");
        }
        m.handle(Input::PingFailed);

        // Assert
        assert_eq!(m.state(), &LinkState::Backoff { attempt: 0 });
    }

    #[test]
    fn backoff_one_two_four_seconds_then_failed_gave_up() {
        // Arrange
        let mut m = in_state("requesting");
        let mut waits = Vec::new();

        // Act
        for _ in 0..4 {
            for action in m.handle(Input::Unavailable) {
                if let Action::Arm(Timer::Backoff, wait) = action {
                    waits.push(wait.as_secs());
                }
            }
            m.handle(Input::Timer(Timer::Backoff));
        }

        // Assert
        assert_eq!(waits, vec![1, 2, 4]);
        assert_eq!(m.state(), &LinkState::Failed { reason: FailReason::GaveUp });
    }

    #[test]
    fn android_10_cap_one_automatic_retry_then_failed() {
        // Arrange
        let config = LinkConfig { max_requests: Some(2), ..LinkConfig::default() };
        let mut m = LinkMachine::new(NAME, config);
        m.handle(Input::WifiModeEntered);

        // Act
        m.handle(Input::Unavailable);
        m.handle(Input::Timer(Timer::Backoff));
        m.handle(Input::Unavailable);

        // Assert
        assert_eq!(m.state(), &LinkState::Failed { reason: FailReason::GaveUp });
    }

    #[test]
    fn linked_three_heartbeat_misses_relink_two_do_not() {
        // Arrange
        let mut m = in_state("linked");

        // Act
        m.handle(Input::PingFailed);
        m.handle(Input::PingFailed);
        let after_two = m.state().name();
        let actions = m.handle(Input::PingFailed);

        // Assert
        assert_eq!(after_two, "linked");
        assert_eq!(m.state(), &LinkState::Requesting { attempt: 0 });
        assert_eq!(actions[0], Action::Release);
        assert!(actions.contains(&Action::Request));
    }

    #[test]
    fn linked_any_successful_op_resets_misses_and_rearms_heartbeat() {
        // Arrange
        let mut m = in_state("linked");
        m.handle(Input::PingFailed);
        m.handle(Input::PingFailed);

        // Act
        let actions = m.handle(Input::OpSucceeded);
        m.handle(Input::PingFailed);
        m.handle(Input::PingFailed);

        // Assert
        assert_eq!(actions, vec![Action::Arm(Timer::Heartbeat, Duration::from_secs(10))]);
        assert_eq!(m.state().name(), "linked");
    }

    #[test]
    fn stale_timer_from_a_previous_state_is_ignored() {
        // Arrange
        let mut m = in_state("linked");

        // Act
        let actions = m.handle(Input::Timer(Timer::RequestTimeout));

        // Assert
        assert!(actions.is_empty());
        assert_eq!(m.state().name(), "linked");
    }

    #[test]
    fn retry_while_not_in_wifi_mode_does_nothing() {
        // Arrange
        let mut m = in_state("failed");
        m.handle(Input::WifiModeLeft);

        // Act
        let actions = m.handle(Input::UserRetry);

        // Assert
        assert!(actions.is_empty());
        assert_eq!(m.state().name(), "unlinked");
    }

    #[test]
    fn journal_records_changes_only_and_keeps_the_last_hundred() {
        // Arrange
        let mut m = machine();

        // Act
        m.handle(Input::OpSucceeded); // no change: not journaled
        for _ in 0..60 {
            m.handle(Input::WifiModeEntered);
            m.handle(Input::WifiModeLeft);
        }

        // Assert
        let journal: Vec<_> = m.journal().collect();
        assert_eq!(journal.len(), 100);
        assert_eq!(journal.last().unwrap().to, LinkState::Unlinked);
        assert_eq!(journal.last().unwrap().input, Input::WifiModeLeft);
    }
}
