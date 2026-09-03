//! `WifiTransport`: the trait L9's mobile plugins implement for the WiFi
//! side, and `ReqwestWifi`, the `reqwest`-backed desktop implementation
//! (SPEC §6). SPEC §6.2's Android network-binding proxy is explicitly out
//! of scope here (design §7: "Mobile ... WiFi-network binding are Tauri
//! mobile plugins ... Isaac's lane"; SPEC §6.2: "On every other platform
//! the app talks to 192.168.4.1 directly and the user joins the AP in
//! system settings") — desktop always talks to the fixed AP IP directly.

use crate::device::DeviceFile;
use crate::{TransportError, TransportErrorKind};

/// The device's fixed WiFi-mode IP (SPEC §6, AP mode, no router).
pub const DEVICE_BASE_URL: &str = "http://192.168.4.1";

/// Builds the `Range` request header for a resumed download starting at
/// `resume_from_bytes` bytes into the file (SPEC §6.1 `/download`: "`Range:
/// bytes=START[-END]` supported").
pub fn range_header(resume_from_bytes: u64) -> String {
    format!("bytes={resume_from_bytes}-")
}

/// Parses a `Content-Range` response header of the form `bytes start-end/total`
/// (or `bytes start-end/*` if the device doesn't report total size). Returns
/// `(start_byte, end_byte, total_bytes)` — `end_byte` is inclusive, matching
/// the wire format. Returns `None` for anything not matching that shape.
pub fn parse_content_range(header: &str) -> Option<(u64, u64, Option<u64>)> {
    let rest = header.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start_byte: u64 = start.parse().ok()?;
    let end_byte: u64 = end.parse().ok()?;
    let total_bytes = if total == "*" { None } else { total.parse().ok() };
    Some((start_byte, end_byte, total_bytes))
}

/// `GET /ping` response body (SPEC §6.1).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct PingResponse {
    /// `idl0_device_name()` — checked against the expected device by
    /// `verify_device_identity` before the app trusts the link.
    pub device: String,
    /// Running image's `esp_app_desc_t.version`, same value as the §7.3
    /// `Firmware:` line.
    pub fw: String,
    /// WiFi control-protocol version (currently `1`), not a byte/time
    /// quantity — an ordinal, so no `_bytes`/`_ms`/`_pct` suffix applies;
    /// renamed from the wire's bare `proto` for clarity at call sites (judgment
    /// call, Task 6 — SPEC §6.1 names the JSON key `proto`, the shape is
    /// otherwise unspecified). A major-version mismatch means the app must
    /// refuse operations and surface a firmware-update prompt (SPEC §6.1).
    #[serde(rename = "proto")]
    pub proto_version: u32,
    /// Battery charge, percent — SPEC §6.1's wire key is the bare `battery`;
    /// renamed here to match `ble_status::DeviceStatus::battery_pct`'s
    /// unit-suffixed convention (judgment call, Task 6, matching this
    /// crate's naming pattern flagged in prior reviews).
    #[serde(rename = "battery")]
    pub battery_pct: u8,
    /// SD card state, mirroring the §7.3 `SD:` line's raw string.
    pub sd: String,
    /// Device mode string (e.g. `"wifi"`).
    pub mode: String,
    /// `"on"` until `POST /handoff`, `"off"` after (SPEC §6.1).
    pub ble: String,
}

/// Checks `/ping`'s `device` field against the name the app expects before
/// trusting the link (SPEC §6.1: "the app verifies it against the expected
/// device before trusting the link — every IDL0 AP shares `192.168.4.1`").
pub fn verify_device_identity(
    ping: &PingResponse,
    expected_name: &str,
) -> Result<(), TransportError> {
    if ping.device != expected_name {
        return Err(TransportError::new(
            TransportErrorKind::Wifi,
            format!("connected to {}, expected {}", ping.device, expected_name),
        ));
    }
    Ok(())
}

/// Abstraction over the WiFi/HTTP link to one IDL0 device in WiFi mode
/// (SPEC §6). `ReqwestWifi` implements it for desktop; L9's mobile plugin
/// implements it wrapping the platform's network-binding proxy (SPEC §6.2)
/// behind the same trait.
pub trait WifiTransport {
    /// `GET /ping` (SPEC §6.1) — status + identity check.
    async fn ping(&self) -> Result<PingResponse, TransportError>;

    /// `POST /handoff` (SPEC §6.1) — idempotent.
    async fn handoff(&self) -> Result<(), TransportError>;

    /// `POST /wifi_off` (SPEC §6.1) — the normal WiFi-exit path.
    async fn wifi_off(&self) -> Result<(), TransportError>;

    /// `GET /files` (SPEC §6.1).
    async fn list_files(&self) -> Result<Vec<DeviceFile>, TransportError>;

    /// `GET /download?file=N`, resumable via `Range` (SPEC §6.1). Streams
    /// into `sink` starting at `resume_from_bytes` bytes (`0` for a fresh
    /// download); calls `on_progress(done_bytes, total_bytes)` as chunks
    /// arrive. Returns the total bytes written this call (not including
    /// `resume_from_bytes`). `sink`/`on_progress` are generic so this trait
    /// names no `idl-rs` core or Tauri type (CLAUDE.md layer rule) — L5
    /// supplies a temp-file sink and forwards progress into a
    /// `Channel<Progress>`.
    async fn download(
        &self,
        file_index: u32,
        resume_from_bytes: u64,
        sink: &mut (dyn tokio::io::AsyncWrite + Unpin + Send),
        on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<u64, TransportError>;

    /// `GET /delete?file=N` (SPEC §6.1).
    async fn delete(&self, file_index: u32) -> Result<(), TransportError>;

    /// `POST /config` (SPEC §6.1) — the WiFi fallback path; BLE
    /// (`BleTransport::push_config`) is primary (SPEC §7.2). Same opaque
    /// `&[u8]` contract as the BLE method — no schema knowledge here either.
    async fn push_config(&self, config_json: &[u8]) -> Result<(), TransportError>;

    /// `POST /ota` (SPEC §6.1) — raw firmware image body,
    /// `Content-Type: application/octet-stream`, `Content-Length` set.
    async fn push_ota(&self, firmware_image: &[u8]) -> Result<(), TransportError>;
}

/// `reqwest`-backed desktop `WifiTransport`, always against
/// `DEVICE_BASE_URL` (SPEC §6.2: desktop joins the AP in system settings,
/// no network-binding proxy).
pub struct ReqwestWifi {
    /// Shared HTTP client (connection pooling across calls).
    client: reqwest::Client,
    /// Base URL the device is reachable at, e.g. `DEVICE_BASE_URL`.
    base_url: String,
}

impl ReqwestWifi {
    /// `base_url` is a constructor parameter (not hardcoded to
    /// `DEVICE_BASE_URL`) so tests can point it at a local mock HTTP server
    /// (Task 7) without touching a real device.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { client: reqwest::Client::new(), base_url: base_url.into() }
    }
}

impl WifiTransport for ReqwestWifi {
    // Task 7 implementer fills in each method body against `reqwest`
    // 0.13.4's actual `Client`/`RequestBuilder`/`Response` API. No
    // real-server unit tests here (Task 7 adds the mock HTTP server and
    // integration-shaped tests); every body below is `unimplemented!()`
    // only so the crate compiles now — they panic if called, and nothing
    // calls them until Task 7.

    async fn ping(&self) -> Result<PingResponse, TransportError> {
        unimplemented!("Task 7 implementer: GET {{base_url}}/ping, .json::<PingResponse>().await, map non-2xx/decode failures to TransportErrorKind::Wifi")
    }

    async fn handoff(&self) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: POST {{base_url}}/handoff, map non-2xx to TransportErrorKind::Wifi")
    }

    async fn wifi_off(&self) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: POST {{base_url}}/wifi_off, map non-2xx to TransportErrorKind::Wifi")
    }

    async fn list_files(&self) -> Result<Vec<DeviceFile>, TransportError> {
        unimplemented!("Task 7 implementer: GET {{base_url}}/files, .json::<Vec<DeviceFile>>().await (resolve Open question 3's #[serde(rename = \"size\")] on DeviceFile::size_bytes)")
    }

    async fn download(
        &self,
        _file_index: u32,
        _resume_from_bytes: u64,
        _sink: &mut (dyn tokio::io::AsyncWrite + Unpin + Send),
        _on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<u64, TransportError> {
        unimplemented!("Task 7 implementer: GET {{base_url}}/download?file=N with a Range header from range_header(resume_from_bytes) when resume_from_bytes > 0, stream .bytes_stream() into sink via tokio::io::AsyncWriteExt::write_all, call on_progress per chunk, cross-check a 206 response's Content-Range via parse_content_range against the requested offset")
    }

    async fn delete(&self, _file_index: u32) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: GET {{base_url}}/delete?file=N, map non-2xx to TransportErrorKind::Wifi")
    }

    async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: POST {{base_url}}/config with config_json as the body, map non-2xx to TransportErrorKind::Wifi")
    }

    async fn push_ota(&self, _firmware_image: &[u8]) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: POST {{base_url}}/ota, Content-Type: application/octet-stream, Content-Length set, map non-2xx to TransportErrorKind::Wifi")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_header_resume_from_1000_bytes_formats_open_ended_range() {
        // Arrange
        let resume_from_bytes = 1000u64;

        // Act
        let header = range_header(resume_from_bytes);

        // Assert
        assert_eq!(header, "bytes=1000-");
    }

    #[test]
    fn range_header_resume_from_zero_formats_bytes_zero_dash() {
        // Arrange
        let resume_from_bytes = 0u64;

        // Act
        let header = range_header(resume_from_bytes);

        // Assert
        assert_eq!(header, "bytes=0-");
    }

    #[test]
    fn parse_content_range_known_total_returns_start_end_total() {
        // Arrange
        let header = "bytes 1000-1999/5000";

        // Act
        let parsed = parse_content_range(header);

        // Assert
        assert_eq!(parsed, Some((1000, 1999, Some(5000))));
    }

    #[test]
    fn parse_content_range_unknown_total_star_returns_none_total() {
        // Arrange
        let header = "bytes 0-999/*";

        // Act
        let parsed = parse_content_range(header);

        // Assert
        assert_eq!(parsed, Some((0, 999, None)));
    }

    #[test]
    fn parse_content_range_malformed_header_returns_none() {
        // Arrange
        let header = "not a content-range header";

        // Act
        let parsed = parse_content_range(header);

        // Assert
        assert_eq!(parsed, None);
    }

    #[test]
    fn verify_device_identity_matching_name_ok_mismatched_name_wifi_error() {
        // Arrange
        let ping = PingResponse {
            device: "IDL0-A3F2".to_string(),
            fw: "1.4.0".to_string(),
            proto_version: 1,
            battery_pct: 87,
            sd: "OK".to_string(),
            mode: "wifi".to_string(),
            ble: "on".to_string(),
        };

        // Act
        let matching = verify_device_identity(&ping, "IDL0-A3F2");
        let mismatched = verify_device_identity(&ping, "IDL0-B000");

        // Assert
        assert!(matching.is_ok());
        let err = mismatched.unwrap_err();
        assert_eq!(err.kind, TransportErrorKind::Wifi);
    }
}
