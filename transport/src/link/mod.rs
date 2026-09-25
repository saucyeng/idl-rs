//! SPEC §6.2 / §14b.3 link management: the state machine that owns the
//! WiFi link to one logger, on every platform.
//!
//! [`LinkMachine`] is pure — no I/O, no clock, no tasks. It takes
//! [`Input`]s (the platform binder's events, `/ping` outcomes, timers
//! firing, the device's mode) and returns the [`Action`]s the caller must
//! perform (request or release the network, ping, hand off, arm a timer).
//! The driver that performs them lives with the platform glue in
//! `idl-rs-tauri`; everything that decides lives here, behind one explicit
//! transition table with a test per (state, input) pair.

mod driver;
mod machine;

pub use machine::{Action, FailReason, Input, LinkConfig, LinkMachine, LinkState, Timer, Transition};
pub use driver::{spawn_link, DirectBinder, LinkError, LinkHandle, NetworkBinder, OP_GATE};
