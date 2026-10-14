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
    /// Service UUIDs from the device's advertisement, lowercase hyphenated
    /// form (e.g. `"000000ff-0000-1000-8000-00805f9b34fb"`). Lets a caller
    /// filter discovered devices by service client-side (e.g. an HRM search
    /// filtering to the standard heart-rate service by default) without a
    /// second round of GATT discovery. Empty if the platform's advertisement
    /// report carried none.
    pub service_uuids: Vec<String>,
}

/// Result of a successful BLE connect + GATT setup (SPEC §7.4 steps 2–4).
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectionInfo {
    /// Same identifier as the `DiscoveredDevice` passed to `connect`.
    pub device_id: String,
    /// Advertised name, e.g. `"IDL0-A3F2"` (SPEC §3.6) — the WiFi AP's SSID
    /// and the name `/ping` must report (SPEC §14b.3). Empty when the
    /// platform did not report one.
    pub name: String,
    /// From the first status read/notification's `Firmware:` line (SPEC §7.3).
    pub firmware_version: String,
    /// `true` once GATT setup (service/characteristic discovery, Status
    /// notifications enabled) completed successfully (SPEC §7.4 steps 2–4).
    pub connected: bool,
}

/// The `CMD_FILES_READ_BEGIN` payload (SPEC §7.2): the logger's newest
/// session files, newest first, read over BLE without WiFi.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RecentFiles {
    /// Every file in the logger's `/sessions/`, listed or not.
    pub total: u32,
    /// At most the ten newest files, newest first; same entries as `GET /files`.
    pub files: Vec<DeviceFile>,
}

/// Parses a `CMD_FILES_READ_BEGIN` read-back (SPEC §7.2). Malformed JSON is
/// a `TransportErrorKind::Ble` error: the bytes came over the BLE link.
pub fn parse_recent_files(bytes: &[u8]) -> Result<RecentFiles, crate::TransportError> {
    serde_json::from_slice(bytes).map_err(|e| {
        crate::TransportError::new(crate::TransportErrorKind::Ble, format!("recent-files read-back is not valid JSON: {e}"))
    })
}

/// One entry from `GET /files` (SPEC §6.1) and from the BLE recent-files
/// read-back (SPEC §7.2).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DeviceFile {
    /// File name on the device's SD card.
    pub name: String,
    /// File size, bytes. Open question 3: SPEC §6.1's wire key is the bare
    /// `size`; renamed here to this crate's unit-suffixed convention so C3
    /// §3.8's already-signed `DeviceFile.size_bytes: u64` matches without a
    /// second rename at the Tauri layer.
    #[serde(rename = "size")]
    pub size_bytes: u64,
    /// 32 lowercase hex chars, or `None` if the file's header was unreadable
    /// (SPEC §6.1: "omitted only if the header is unreadable").
    #[serde(default)]
    pub session_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recent_files_payload_with_and_without_session_id_both_entries_kept() {
        // Arrange
        let bytes = br#"{"total":3,"files":[{"name":"b.idl0","size":10,"session_id":"ab"},{"name":"a.idl0","size":5}]}"#;

        // Act
        let recent = parse_recent_files(bytes).unwrap();

        // Assert
        assert_eq!(recent.total, 3);
        assert_eq!(recent.files[0].session_id.as_deref(), Some("ab"));
        assert_eq!(recent.files[1].size_bytes, 5);
    }

    #[test]
    fn parse_recent_files_not_json_ble_error() {
        // Arrange
        let bytes = b"not json";

        // Act
        let err = parse_recent_files(bytes).unwrap_err();

        // Assert
        assert_eq!(err.kind, crate::TransportErrorKind::Ble);
    }
}
