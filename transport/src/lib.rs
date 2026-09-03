//! Device transport and LAN sync for idl1.
//!
//! This crate owns every byte that leaves or enters the machine: BLE to the
//! logger, WiFi transfer against the device's HTTP protocol (SPEC §6), config
//! push (SPEC §8), and peer sync on the pit-lane LAN. It never processes
//! signals — that is `idl-rs` — and `idl-rs` never depends on this crate.
//!
//! M0 ships only the typed error; lanes L4 and L11 fill it in.

pub mod error;

pub use error::{TransportError, TransportErrorKind};
