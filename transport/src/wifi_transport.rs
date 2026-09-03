//! `WifiTransport`: the trait L9's mobile plugins implement for the WiFi
//! side, and `ReqwestWifi`, the `reqwest`-backed desktop implementation
//! (SPEC §6). SPEC §6.2's Android network-binding proxy is explicitly out
//! of scope here (design §7: "Mobile ... WiFi-network binding are Tauri
//! mobile plugins ... Isaac's lane"; SPEC §6.2: "On every other platform
//! the app talks to 192.168.4.1 directly and the user joins the AP in
//! system settings") — desktop always talks to the fixed AP IP directly.

use futures::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::device::DeviceFile;
use crate::{TransportError, TransportErrorKind};

/// Builds a `TransportErrorKind::Wifi` error with `message`.
fn wifi_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Wifi, message)
}

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
    async fn ping(&self) -> Result<PingResponse, TransportError> {
        let response = self
            .client
            .get(format!("{}/ping", self.base_url))
            .send()
            .await
            .map_err(|e| wifi_error(format!("GET /ping failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("GET /ping returned an error status: {e}")))?;
        response
            .json::<PingResponse>()
            .await
            .map_err(|e| wifi_error(format!("GET /ping returned malformed JSON: {e}")))
    }

    async fn handoff(&self) -> Result<(), TransportError> {
        self.client
            .post(format!("{}/handoff", self.base_url))
            .send()
            .await
            .map_err(|e| wifi_error(format!("POST /handoff failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("POST /handoff returned an error status: {e}")))?;
        Ok(())
    }

    async fn wifi_off(&self) -> Result<(), TransportError> {
        self.client
            .post(format!("{}/wifi_off", self.base_url))
            .send()
            .await
            .map_err(|e| wifi_error(format!("POST /wifi_off failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("POST /wifi_off returned an error status: {e}")))?;
        Ok(())
    }

    async fn list_files(&self) -> Result<Vec<DeviceFile>, TransportError> {
        let response = self
            .client
            .get(format!("{}/files", self.base_url))
            .send()
            .await
            .map_err(|e| wifi_error(format!("GET /files failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("GET /files returned an error status: {e}")))?;
        response
            .json::<Vec<DeviceFile>>()
            .await
            .map_err(|e| wifi_error(format!("GET /files returned malformed JSON: {e}")))
    }

    /// SPEC §6.1 `/download`, resumable via `Range`/`Content-Range`. Sends
    /// the `Range` header only when resuming (`resume_from_bytes > 0`) —
    /// asking for `bytes=0-` on a fresh download is unnecessary and some
    /// HTTP servers respond `206` instead of `200` to any `Range` header,
    /// which would make the fresh-download and resumed-download code paths
    /// harder to tell apart than they need to be.
    async fn download(
        &self,
        file_index: u32,
        resume_from_bytes: u64,
        sink: &mut (dyn tokio::io::AsyncWrite + Unpin + Send),
        on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
    ) -> Result<u64, TransportError> {
        let url = format!("{}/download?file={file_index}", self.base_url);
        let mut request = self.client.get(&url);
        if resume_from_bytes > 0 {
            request = request.header(reqwest::header::RANGE, range_header(resume_from_bytes));
        }
        let response = request
            .send()
            .await
            .map_err(|e| wifi_error(format!("GET /download failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(wifi_error(format!("GET /download returned status {status}")));
        }

        // A 206 echoes exactly what range the server actually served — the
        // trait's own contract (`WifiTransport::download`'s doc comment)
        // says to error rather than silently trust a mismatch.
        let total_bytes = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let content_range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| wifi_error("206 response is missing a Content-Range header"))?
                .to_string();
            let (start_byte, _end_byte, total_bytes) = parse_content_range(&content_range)
                .ok_or_else(|| {
                    wifi_error(format!("malformed Content-Range header: {content_range}"))
                })?;
            if start_byte != resume_from_bytes {
                return Err(wifi_error(format!(
                    "server resumed at byte {start_byte}, but {resume_from_bytes} was requested"
                )));
            }
            total_bytes
        } else {
            None
        };

        let mut stream = response.bytes_stream();
        let mut done_bytes: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| wifi_error(format!("download stream error: {e}")))?;
            sink.write_all(&chunk)
                .await
                .map_err(|e| wifi_error(format!("writing downloaded bytes failed: {e}")))?;
            done_bytes += chunk.len() as u64;
            on_progress(done_bytes, total_bytes);
        }
        Ok(done_bytes)
    }

    async fn delete(&self, file_index: u32) -> Result<(), TransportError> {
        self.client
            .get(format!("{}/delete?file={file_index}", self.base_url))
            .send()
            .await
            .map_err(|e| wifi_error(format!("GET /delete failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("GET /delete returned an error status: {e}")))?;
        Ok(())
    }

    async fn push_config(&self, config_json: &[u8]) -> Result<(), TransportError> {
        self.client
            .post(format!("{}/config", self.base_url))
            .body(config_json.to_vec())
            .send()
            .await
            .map_err(|e| wifi_error(format!("POST /config failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("POST /config returned an error status: {e}")))?;
        Ok(())
    }

    async fn push_ota(&self, firmware_image: &[u8]) -> Result<(), TransportError> {
        self.client
            .post(format!("{}/ota", self.base_url))
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .body(firmware_image.to_vec())
            .send()
            .await
            .map_err(|e| wifi_error(format!("POST /ota failed: {e}")))?
            .error_for_status()
            .map_err(|e| wifi_error(format!("POST /ota returned an error status: {e}")))?;
        Ok(())
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

/// Integration-shaped tests: `ReqwestWifi` against a hand-rolled HTTP/1.1
/// mock device server (Task 7 Step 2/3), rather than a real IDL0 device —
/// that half of the proof is Task 9's manual step.
#[cfg(test)]
mod integration {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt as _};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    use super::*;

    /// One canned HTTP response: status code, reason phrase, extra headers
    /// (beyond `Content-Length`/`Connection`, which this server always
    /// sets itself), and body bytes.
    type MockResponse = (u16, &'static str, Vec<(String, String)>, Vec<u8>);

    /// Spins up a `TcpListener` on an OS-assigned free port and answers
    /// every request with whatever `route(path, headers)` returns. No HTTP
    /// framework: SPEC §6 needs only a handful of fixed GET/POST endpoints
    /// with no persistent connections, and a hand-rolled response is a
    /// handful of lines (Task 7 Step 2's own guidance) — this keeps the
    /// crate's dependency list to what SPEC actually requires. One request
    /// per accepted connection; `route` runs once per request.
    async fn spawn_mock_server<F>(route: F) -> (SocketAddr, JoinHandle<()>)
    where
        F: Fn(&str, &[(String, String)]) -> MockResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
        let addr = listener.local_addr().expect("mock server local addr");
        let route = Arc::new(route);

        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { break };
                let route = Arc::clone(&route);
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let mut total = 0;
                    loop {
                        let n = socket.read(&mut buf[total..]).await.unwrap_or(0);
                        if n == 0 {
                            return; // connection closed before a full request arrived
                        }
                        total += n;
                        if buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
                            break; // end of headers — every request here has no body
                        }
                    }
                    let text = String::from_utf8_lossy(&buf[..total]);
                    let mut lines = text.lines();
                    let request_line = lines.next().unwrap_or("");
                    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
                    let headers: Vec<(String, String)> = lines
                        .take_while(|line| !line.is_empty())
                        .filter_map(|line| line.split_once(':'))
                        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                        .collect();

                    let (status, reason, extra_headers, body) = route(path, &headers);
                    let mut response = format!(
                        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
                        body.len()
                    );
                    for (key, value) in extra_headers {
                        response.push_str(&format!("{key}: {value}\r\n"));
                    }
                    response.push_str("\r\n");
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.write_all(&body).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn ping_then_verify_device_identity_matching_ok_mismatched_wifi_error() {
        // Arrange
        let (addr, _server) = spawn_mock_server(|path, _headers| {
            assert_eq!(path, "/ping");
            let body = br#"{"device":"IDL0-A3F2","fw":"1.4.0","proto":1,"battery":87,"sd":"OK","mode":"wifi","ble":"on"}"#.to_vec();
            (
                200,
                "OK",
                vec![("Content-Type".to_string(), "application/json".to_string())],
                body,
            )
        })
        .await;
        let wifi = ReqwestWifi::new(format!("http://{addr}"));

        // Act
        let ping = wifi.ping().await.unwrap();
        let matching = verify_device_identity(&ping, "IDL0-A3F2");
        let mismatched = verify_device_identity(&ping, "IDL0-OTHER");

        // Assert
        assert_eq!(ping.device, "IDL0-A3F2");
        assert!(matching.is_ok());
        assert_eq!(mismatched.unwrap_err().kind, TransportErrorKind::Wifi);
    }

    #[tokio::test]
    async fn list_files_two_entries_one_missing_session_id_deserialises_both() {
        // Arrange
        let (addr, _server) = spawn_mock_server(|path, _headers| {
            assert_eq!(path, "/files");
            let body = br#"[
                {"name":"session_001.idl0","size":12345,"session_id":"0123456789abcdef0123456789abcdef"},
                {"name":"session_002.idl0","size":999}
            ]"#
            .to_vec();
            (
                200,
                "OK",
                vec![("Content-Type".to_string(), "application/json".to_string())],
                body,
            )
        })
        .await;
        let wifi = ReqwestWifi::new(format!("http://{addr}"));

        // Act
        let files = wifi.list_files().await.unwrap();

        // Assert
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].name, "session_001.idl0");
        assert_eq!(files[0].size_bytes, 12345);
        assert_eq!(
            files[0].session_id.as_deref(),
            Some("0123456789abcdef0123456789abcdef")
        );
        assert_eq!(files[1].size_bytes, 999);
        assert_eq!(files[1].session_id, None);
    }

    #[tokio::test]
    async fn download_full_then_resumed_from_midpoint_concatenates_to_original_bytes() {
        // Arrange
        const CONTENT: &[u8] = b"0123456789ABCDEFGHIJ"; // 20 bytes

        let (addr, _server) = spawn_mock_server(|path, headers| {
            assert!(path.starts_with("/download?file="));
            let range_value = headers
                .iter()
                .find(|(key, _)| key == "range")
                .map(|(_, value)| value.clone());
            match range_value {
                None => (200, "OK", Vec::new(), CONTENT.to_vec()),
                Some(range_value) => {
                    let start: usize = range_value
                        .trim_start_matches("bytes=")
                        .trim_end_matches('-')
                        .parse()
                        .expect("test-only Range header is well-formed");
                    let body = CONTENT[start..].to_vec();
                    let content_range =
                        format!("bytes {start}-{}/{}", CONTENT.len() - 1, CONTENT.len());
                    (
                        206,
                        "Partial Content",
                        vec![("Content-Range".to_string(), content_range)],
                        body,
                    )
                }
            }
        })
        .await;
        let wifi = ReqwestWifi::new(format!("http://{addr}"));

        // Act — fresh download, no resume
        let mut full_sink = Vec::new();
        let mut full_progress = Vec::new();
        let mut on_full_progress =
            |done_bytes: u64, total_bytes: Option<u64>| full_progress.push((done_bytes, total_bytes));
        let full_bytes_written = wifi
            .download(0, 0, &mut full_sink, &mut on_full_progress)
            .await
            .unwrap();

        // Act — resumed from the midpoint, simulating an interrupted download
        let mut resumed_sink = Vec::new();
        let mut on_resumed_progress = |_: u64, _: Option<u64>| {};
        let resumed_bytes_written = wifi
            .download(0, 10, &mut resumed_sink, &mut on_resumed_progress)
            .await
            .unwrap();

        // Assert
        assert_eq!(full_bytes_written, CONTENT.len() as u64);
        assert_eq!(full_sink, CONTENT);
        assert!(!full_progress.is_empty());
        assert!(full_progress.windows(2).all(|w| w[0].0 <= w[1].0));

        assert_eq!(resumed_bytes_written, (CONTENT.len() - 10) as u64);
        assert_eq!(resumed_sink, &CONTENT[10..]);

        let mut concatenated = full_sink[..10].to_vec();
        concatenated.extend_from_slice(&resumed_sink);
        assert_eq!(concatenated, CONTENT);
    }
}
