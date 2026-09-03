//! Plain data shared by the BLE and WiFi transports — no protocol logic, no
//! I/O. Kept separate so `ble_transport.rs`/`wifi_transport.rs` (I/O glue)
//! and the pure protocol modules can both depend on it without a cycle.

/// One device seen during a BLE scan (SPEC §7.4 step 1).
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredDevice {
    /// Platform BLE address/identifier — opaque, passed back verbatim to `connect`.
    pub device_id: String,
    /// Advertised local name, e.g. `"IDL0-A3F2"`.
    pub name: String,
    /// Received signal strength, dBm. Negative; closer to 0 is stronger.
    pub rssi_dbm: i32,
}

/// Result of a successful BLE connect + GATT setup (SPEC §7.4 steps 2–4).
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionInfo {
    /// Same identifier as the `DiscoveredDevice` passed to `connect`.
    pub device_id: String,
    /// From the first status read/notification's `Firmware:` line (SPEC §7.3).
    pub firmware_version: String,
    /// `true` once GATT setup (service/characteristic discovery, Status
    /// notifications enabled) completed successfully (SPEC §7.4 steps 2–4).
    pub connected: bool,
}

/// One entry from `GET /files` (SPEC §6.1).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DeviceFile {
    pub name: String,
    pub size_bytes: u64,
    /// 32 lowercase hex chars, or `None` if the file's header was unreadable
    /// (SPEC §6.1: "omitted only if the header is unreadable").
    #[serde(default)]
    pub session_id: Option<String>,
}
