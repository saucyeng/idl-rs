//! Device transport and LAN sync for idl1.
//!
//! This crate owns every byte that leaves or enters the machine: BLE to the
//! logger, WiFi transfer against the device's HTTP protocol (SPEC §6), config
//! push (SPEC §8), and peer sync on the pit-lane LAN. It never processes
//! signals — that is `idl-rs` — and `idl-rs` never depends on this crate.
//!
//! M0 shipped only the typed error. L4 adds the desktop BLE/WiFi device
//! client. L11 adds LAN peer sync (`sync/`) beside it, starting with wire
//! DTOs and pairing.

pub mod ble_config;
pub mod ble_control;
pub mod ble_status;
pub mod ble_transport;
pub mod device;
pub mod firmware_catalog;
pub mod error;
pub mod sync;
pub mod wifi_transport;

pub use device::{ConnectionInfo, DeviceFile, DiscoveredDevice};
pub use error::{TransportError, TransportErrorKind};
