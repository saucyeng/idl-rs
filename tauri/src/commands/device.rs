//! Device commands (C3 §3.8): `ble_scan`, `ble_connect`, `list_device_files`,
//! `download_file`, `push_config` — wired to `idl-transport`'s desktop BLE/
//! WiFi implementations (`BtleplugBle`/`ReqwestWifi`, L4, SPEC §14a).
//!
//! Each `#[tauri::command]` is a thin wrapper: build the concrete transport
//! (`BtleplugBle::new()`, `ReqwestWifi::new(DEVICE_BASE_URL)`), then call a
//! `_via`-suffixed helper generic over `BleTransport`/`WifiTransport`. The
//! `_via` helpers are what this module's own tests exercise, against a
//! hand-written stub — L4 ships no cross-crate-visible test double (its own
//! `StubBle` lives in a `#[cfg(test)]`-private module of `idl-transport`),
//! and no real BLE adapter is available in CI (SPEC §14a, "Timeouts" /
//! design doc's own "proven at M2" risk note), so this layer's own tests are
//! argument-shape / error-mapping tests, not device round-trips.
//!
//! **Connection lifetime, a judgment call:** every command below connects,
//! acts, and disconnects within its own call — there is no managed,
//! cross-command BLE session (that would need new `app.manage()`-registered
//! state, out of this task's file list). `ble_connect`'s `ConnectionInfo.
//! connected` therefore reflects the state at the moment GATT setup
//! finished inside that one call, not a connection the caller can assume
//! still exists a moment later; a persistent connection manager (so the
//! Device tab can show live status between actions) is left to whichever
//! later task builds that UI (L6/L9), not fixed by C3 or SPEC §14a.

use std::path::Path;
use std::time::Duration;

use idl_transport::ble_control::ControlCommand;
use idl_transport::ble_transport::{BleTransport, BtleplugBle};
use idl_transport::wifi_transport::{ReqwestWifi, WifiTransport, DEVICE_BASE_URL};

use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// One device found during a `ble_scan` (C3 §3.8).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceDiscovered {
    /// Platform BLE address/identifier — passed back verbatim to `ble_connect`.
    pub device_id: String,
    /// Advertised local name, e.g. `"IDL0-A3F2"`.
    pub name: String,
    /// Received signal strength, dBm.
    pub rssi_dbm: i32,
}

impl From<idl_transport::DiscoveredDevice> for DeviceDiscovered {
    fn from(d: idl_transport::DiscoveredDevice) -> Self {
        Self { device_id: d.device_id, name: d.name, rssi_dbm: d.rssi_dbm }
    }
}

/// `ble_connect`'s return (C3 §3.8).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionInfo {
    pub device_id: String,
    pub firmware_version: String,
    pub connected: bool,
}

impl From<idl_transport::ConnectionInfo> for ConnectionInfo {
    fn from(c: idl_transport::ConnectionInfo) -> Self {
        Self { device_id: c.device_id, firmware_version: c.firmware_version, connected: c.connected }
    }
}

/// One file entry from `list_device_files` (C3 §3.8).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceFile {
    pub name: String,
    /// Bytes.
    pub size_bytes: u64,
    /// `None` if the device hasn't assigned one yet.
    pub session_id: Option<String>,
}

impl From<idl_transport::DeviceFile> for DeviceFile {
    fn from(f: idl_transport::DeviceFile) -> Self {
        Self { name: f.name, size_bytes: f.size_bytes, session_id: f.session_id }
    }
}

/// Progress payload streamed by long-running commands (C3 §1).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Progress {
    /// Units completed so far — bytes, for `download_file`.
    pub done: u64,
    /// Units expected in total, `None` when not known ahead of time.
    pub total: Option<u64>,
    /// Short machine-readable phase name. Not localized.
    pub phase: String,
}

/// `download_file`'s return (C3 §3.8).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DownloadResult {
    /// Where the blob landed under `<data>/blobs/sha256/`.
    pub path: String,
    /// 64 lowercase hex chars.
    pub sha256: String,
    /// Bytes.
    pub size_bytes: u64,
}

/// How many times to poll `read_status` while waiting for the device to
/// report `wifi_on == Some(true)` after `ControlCommand::WifiOn`, and how
/// long to wait between polls. SPEC §14a fixes no default for BLE timeouts
/// (its own "Timeouts" section, Open question 10) — chosen here as this
/// task's own judgment call: 10 attempts, 200 ms apart (2 s total), the same
/// poll-until-flag pattern L4's own composed sequencing test
/// (`ble_transport.rs::sequencing`) uses against its stub.
const WIFI_ON_POLL_ATTEMPTS: u32 = 10;
const WIFI_ON_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Hashes `bytes` as lowercase hex SHA-256 (C4 §3's blob-identity hash).
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Sends `ControlCommand::WifiOn` and polls `read_status` until the device
/// reports `wifi_on == Some(true)` (SPEC §14a's composed BLE→WiFi hand-off
/// flow) — the prerequisite for every WiFi-side call below (`list_files`,
/// `download`); `BleTransport` has no file-listing method of its own (SPEC
/// §6/§7 put file transfer on the WiFi side entirely).
async fn switch_to_wifi_mode(ble: &impl BleTransport) -> Result<(), IpcError> {
    ble.send_command(ControlCommand::WifiOn).await.map_err(IpcError::from)?;
    for _ in 0..WIFI_ON_POLL_ATTEMPTS {
        let status = ble.read_status().await.map_err(IpcError::from)?;
        if status.wifi_on == Some(true) {
            return Ok(());
        }
        tokio::time::sleep(WIFI_ON_POLL_INTERVAL).await;
    }
    Err(IpcError::new(IpcErrorKind::Ble, "device did not report wifi_on after switching to WiFi mode"))
}

/// Transport-agnostic core of `ble_scan`: drains `ble.scan(timeout)`,
/// calling `on_discovered` for each device as it arrives.
async fn scan_via(
    ble: &impl BleTransport,
    timeout: Duration,
    mut on_discovered: impl FnMut(DeviceDiscovered),
) -> Result<(), IpcError> {
    let mut rx = ble.scan(timeout).await.map_err(IpcError::from)?;
    while let Some(device) = rx.recv().await {
        on_discovered(device.into());
    }
    Ok(())
}

/// Transport-agnostic core of `ble_connect`: connects, then disconnects
/// (see this module's doc comment on connection lifetime) — the returned
/// `ConnectionInfo` is a snapshot, not a live handle.
async fn connect_via(ble: &mut impl BleTransport, device_id: &str) -> Result<ConnectionInfo, IpcError> {
    let info = ble.connect(device_id).await.map_err(IpcError::from)?;
    let _ = ble.disconnect().await;
    Ok(info.into())
}

/// Transport-agnostic core of `list_device_files`: switches to WiFi mode,
/// then lists files over HTTP.
async fn list_files_via(ble: &impl BleTransport, wifi: &impl WifiTransport) -> Result<Vec<DeviceFile>, IpcError> {
    switch_to_wifi_mode(ble).await?;
    let files = wifi.list_files().await.map_err(IpcError::from)?;
    Ok(files.into_iter().map(DeviceFile::from).collect())
}

/// Transport-agnostic core of `download_file`: switches to WiFi mode,
/// resolves `file_name` to the device's `file_index` (WiFi's `/download`
/// endpoint is index-addressed, not name-addressed — SPEC §6.1), streams
/// into a temp file under `<data>/tmp/`, then moves it into the
/// content-addressed blob store at `<data>/blobs/sha256/<2 hex>/<62 hex>`
/// (C4 §2's fixed sharded path convention — the first 2 lowercase hex chars
/// of the digest as a subdirectory, the remaining 62 as the filename, *not*
/// a flat `blobs/sha256/<64 hex>`; C4 §3: blob hash = SHA-256 of the raw
/// bytes; C4 §4: "a second write to the same hash is a verified no-op, skip
/// rather than overwrite"). This is the write-once blob case, simpler than
/// C4 §4's full atomic-write sequence for mutable files (no
/// optimistic-concurrency re-read needed — a blob never changes once
/// written), which stays L1's to build generally for
/// `session.json`/`data.parquet`/workbooks.
// TODO(idl0): this duplicates the sharded blob-path formula L1's
// `idl_rs::store::blob` module already implements correctly (not yet merged
// to `main` as of this task) — once L1 lands, replace this hand-rolled
// `data_dir.join(...)` split with a real call into that module's writer
// instead of reimplementing C4 §2's path convention here.
async fn download_via(
    ble: &impl BleTransport,
    wifi: &impl WifiTransport,
    file_name: &str,
    data_dir: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>) + Send,
) -> Result<DownloadResult, IpcError> {
    switch_to_wifi_mode(ble).await?;

    let files = wifi.list_files().await.map_err(IpcError::from)?;
    let file_index = files
        .iter()
        .position(|f| f.name == file_name)
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, format!("{file_name} not found on device")))?
        as u32;

    let tmp_dir = data_dir.join("tmp");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {}: {e}", tmp_dir.display())))?;
    let tmp_path = tmp_dir.join(uuid::Uuid::new_v4().to_string());
    let mut tmp_file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {}: {e}", tmp_path.display())))?;

    let size_bytes = wifi
        .download(file_index, 0, &mut tmp_file, &mut on_progress)
        .await
        .map_err(IpcError::from)?;
    tmp_file
        .sync_all()
        .await
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("fsync {}: {e}", tmp_path.display())))?;
    drop(tmp_file);

    let bytes = std::fs::read(&tmp_path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading back {}: {e}", tmp_path.display())))?;
    let sha256 = sha256_hex(&bytes);

    // C4 §2's fixed sharded convention: first 2 hex chars as a
    // subdirectory, remaining 62 as the filename.
    let blob_dir = data_dir.join("blobs/sha256").join(&sha256[..2]);
    std::fs::create_dir_all(&blob_dir)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {}: {e}", blob_dir.display())))?;
    let blob_path = blob_dir.join(&sha256[2..]);
    if blob_path.exists() {
        let _ = std::fs::remove_file(&tmp_path); // already have this content — verified no-op (C4 §4)
    } else {
        std::fs::rename(&tmp_path, &blob_path)
            .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("renaming into blob store: {e}")))?;
    }

    Ok(DownloadResult { path: blob_path.display().to_string(), sha256, size_bytes })
}

/// Checks `config_json` is at least syntactically valid JSON.
///
/// C3 §3.8 specifies full local validation "via the same `parse_config`
/// path core already has" (`idl_rs::config::parse_config::<T:
/// VersionedConfig>`), raising `config_parse`/`config_unsupported_version`
/// on failure. `idl_rs::config` (as of this task) has no `VersionedConfig`
/// implementor for SPEC §8's device-config schema — only the `.idl1wb`
/// workbook and `.idl0t` track-artifact schemas exist — and defining one is
/// `rust/core`'s job, out of this lane's file list (CLAUDE.md §7). Until
/// that type lands, this is the narrower check this layer can make without
/// guessing a schema: well-formed JSON, routed as `invalid_argument` (an
/// already-seeded cross-cutting kind), not the C3-named `config_parse`/
/// `config_unsupported_version` kinds, which stay unraised until the real
/// schema-validation path exists.
// TODO(idl0): swap this for `idl_rs::config::parse_config::<DeviceConfig>`
// once core defines that type, raising `config_parse`/`config_unsupported_version`
// per C3 §3.8 instead of `invalid_argument`.
fn validate_config_json_locally(config_json: &str) -> Result<(), IpcError> {
    serde_json::from_str::<serde_json::Value>(config_json)
        .map(|_| ())
        .map_err(|e| IpcError::new(IpcErrorKind::InvalidArgument, format!("config_json is not valid JSON: {e}")))
}

/// Transport-agnostic core of `push_config`: local syntax check, then BLE
/// push (SPEC §7.2's primary path; the WiFi fallback path, SPEC §6.1's
/// `POST /config`, is not wired here — this task's own scope note is "no
/// new error-mapping code, only argument/return adaptation", and no trigger
/// condition for falling back is fixed anywhere; left as a follow-up).
async fn push_config_via(ble: &impl BleTransport, config_json: &str) -> Result<(), IpcError> {
    validate_config_json_locally(config_json)?;
    ble.push_config(config_json.as_bytes()).await.map_err(IpcError::from)
}

/// Scans for `uuids::SERVICE` BLE devices for `timeout_ms`, streaming a
/// `DeviceDiscovered` message per device found; resolves with no value when
/// the scan window ends (C3 §3.8). Explicit user action on the Device tab —
/// never a hot path (C3 §4).
#[tauri::command]
pub async fn ble_scan(timeout_ms: u32, progress: tauri::ipc::Channel<DeviceDiscovered>) -> Result<(), IpcError> {
    let ble = BtleplugBle::new().await.map_err(IpcError::from)?;
    scan_via(&ble, Duration::from_millis(timeout_ms as u64), |d| {
        let _ = progress.send(d);
    })
    .await
}

/// Connects to `device_id` (from a prior `ble_scan`), completes GATT setup
/// and reads the initial firmware version (C3 §3.8).
#[tauri::command]
pub async fn ble_connect(device_id: String) -> Result<ConnectionInfo, IpcError> {
    let mut ble = BtleplugBle::new().await.map_err(IpcError::from)?;
    connect_via(&mut ble, &device_id).await
}

/// Lists files on `device_id`'s SD card, switching the device into WiFi
/// mode first (C3 §3.8).
#[tauri::command]
pub async fn list_device_files(device_id: String) -> Result<Vec<DeviceFile>, IpcError> {
    let mut ble = BtleplugBle::new().await.map_err(IpcError::from)?;
    ble.connect(&device_id).await.map_err(IpcError::from)?;
    let wifi = ReqwestWifi::new(DEVICE_BASE_URL);
    let result = list_files_via(&ble, &wifi).await;
    let _ = ble.disconnect().await;
    result
}

/// Downloads `file_name` from `device_id` into the content-addressed blob
/// store, streaming `Progress` (bytes done/total) as it goes (C3 §3.8).
#[tauri::command]
pub async fn download_file(
    device_id: String,
    file_name: String,
    progress: tauri::ipc::Channel<Progress>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<DownloadResult, IpcError> {
    let mut ble = BtleplugBle::new().await.map_err(IpcError::from)?;
    ble.connect(&device_id).await.map_err(IpcError::from)?;
    let wifi = ReqwestWifi::new(DEVICE_BASE_URL);
    let result = download_via(&ble, &wifi, &file_name, &data_dir.0, |done, total| {
        let _ = progress.send(Progress { done, total, phase: "download".to_string() });
    })
    .await;
    let _ = ble.disconnect().await;
    result
}

/// Validates `config_json` locally, then pushes it to `device_id` over BLE
/// (C3 §3.8, SPEC §7.2).
#[tauri::command]
pub async fn push_config(device_id: String, config_json: String) -> Result<(), IpcError> {
    let mut ble = BtleplugBle::new().await.map_err(IpcError::from)?;
    ble.connect(&device_id).await.map_err(IpcError::from)?;
    let result = push_config_via(&ble, &config_json).await;
    let _ = ble.disconnect().await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;

    use idl_transport::ble_status::DeviceStatus;
    use idl_transport::{TransportError, TransportErrorKind};
    use tokio::io::AsyncWriteExt;
    use tokio::sync::mpsc;

    /// `BleTransport` test double. Every method not configured by a given
    /// test returns a `Ble` error (never panics), so an unexpected call
    /// fails as a normal assertion rather than aborting the test binary —
    /// mirrors L4's own `StubBle` (`idl-transport`'s `ble_transport.rs::
    /// sequencing`, `#[cfg(test)]`-private to that crate, not reusable from
    /// here).
    struct StubBle {
        scan_devices: Vec<idl_transport::DiscoveredDevice>,
        connect_result: Result<idl_transport::ConnectionInfo, TransportError>,
        send_command_result: Result<(), TransportError>,
        push_config_result: Result<(), TransportError>,
        /// One entry consumed per `read_status` call; the last entry repeats
        /// once the queue is drained.
        wifi_on_reads: StdMutex<VecDeque<Option<bool>>>,
        disconnect_calls: AtomicUsize,
        send_command_calls: AtomicUsize,
    }

    impl Default for StubBle {
        fn default() -> Self {
            Self {
                scan_devices: Vec::new(),
                connect_result: Err(TransportError::new(TransportErrorKind::Ble, "StubBle::connect not configured")),
                send_command_result: Ok(()),
                push_config_result: Err(TransportError::new(
                    TransportErrorKind::Ble,
                    "StubBle::push_config not configured",
                )),
                wifi_on_reads: StdMutex::new(VecDeque::new()),
                disconnect_calls: AtomicUsize::new(0),
                send_command_calls: AtomicUsize::new(0),
            }
        }
    }

    impl BleTransport for StubBle {
        async fn scan(&self, _timeout: Duration) -> Result<mpsc::Receiver<idl_transport::DiscoveredDevice>, TransportError> {
            let (tx, rx) = mpsc::channel(self.scan_devices.len().max(1));
            for d in self.scan_devices.clone() {
                let _ = tx.send(d).await;
            }
            Ok(rx)
        }

        async fn connect(&mut self, _device_id: &str) -> Result<idl_transport::ConnectionInfo, TransportError> {
            self.connect_result.clone()
        }

        async fn disconnect(&mut self) -> Result<(), TransportError> {
            self.disconnect_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn read_status(&self) -> Result<DeviceStatus, TransportError> {
            let mut q = self.wifi_on_reads.lock().unwrap();
            let wifi_on = if q.len() > 1 { q.pop_front().unwrap() } else { q.front().copied().flatten() };
            Ok(DeviceStatus { wifi_on, ..DeviceStatus::default() })
        }

        async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError> {
            Err(TransportError::new(TransportErrorKind::Ble, "StubBle::watch_status not exercised"))
        }

        async fn send_command(&self, _cmd: ControlCommand) -> Result<(), TransportError> {
            self.send_command_calls.fetch_add(1, Ordering::SeqCst);
            self.send_command_result.clone()
        }

        async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
            self.push_config_result.clone()
        }

        async fn read_config(&self) -> Result<Vec<u8>, TransportError> {
            Err(TransportError::new(TransportErrorKind::Ble, "StubBle::read_config not exercised"))
        }
    }

    /// `WifiTransport` test double, same "unconfigured methods error"
    /// convention as `StubBle`.
    struct StubWifi {
        list_files_result: Result<Vec<idl_transport::DeviceFile>, TransportError>,
        download_bytes: Result<Vec<u8>, TransportError>,
    }

    impl Default for StubWifi {
        fn default() -> Self {
            Self {
                list_files_result: Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::list_files not configured")),
                download_bytes: Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::download not configured")),
            }
        }
    }

    impl WifiTransport for StubWifi {
        async fn ping(&self) -> Result<idl_transport::wifi_transport::PingResponse, TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::ping not exercised"))
        }

        async fn handoff(&self) -> Result<(), TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::handoff not exercised"))
        }

        async fn wifi_off(&self) -> Result<(), TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::wifi_off not exercised"))
        }

        async fn list_files(&self) -> Result<Vec<idl_transport::DeviceFile>, TransportError> {
            self.list_files_result.clone()
        }

        async fn download(
            &self,
            _file_index: u32,
            resume_from_bytes: u64,
            sink: &mut (dyn tokio::io::AsyncWrite + Unpin + Send),
            on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<u64, TransportError> {
            let bytes = self.download_bytes.clone()?;
            let slice = &bytes[resume_from_bytes as usize..];
            sink.write_all(slice)
                .await
                .map_err(|e| TransportError::new(TransportErrorKind::Wifi, e.to_string()))?;
            on_progress(slice.len() as u64, Some(bytes.len() as u64));
            Ok(slice.len() as u64)
        }

        async fn delete(&self, _file_index: u32) -> Result<(), TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::delete not exercised"))
        }

        async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::push_config not exercised"))
        }

        async fn push_ota(&self, _firmware_image: &[u8]) -> Result<(), TransportError> {
            Err(TransportError::new(TransportErrorKind::Wifi, "StubWifi::push_ota not exercised"))
        }
    }

    fn discovered(device_id: &str, name: &str, rssi_dbm: i32) -> idl_transport::DiscoveredDevice {
        idl_transport::DiscoveredDevice { device_id: device_id.to_string(), name: name.to_string(), rssi_dbm }
    }

    #[tokio::test]
    async fn scan_via_two_devices_on_the_channel_calls_on_discovered_for_each_in_order() {
        // Arrange
        let ble = StubBle { scan_devices: vec![discovered("AA:BB", "IDL0-A3F2", -40), discovered("CC:DD", "IDL0-9911", -70)], ..Default::default() };
        let mut seen = Vec::new();

        // Act
        scan_via(&ble, Duration::from_millis(10), |d| seen.push(d)).await.unwrap();

        // Assert
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].device_id, "AA:BB");
        assert_eq!(seen[0].rssi_dbm, -40);
        assert_eq!(seen[1].name, "IDL0-9911");
    }

    #[tokio::test]
    async fn connect_via_success_maps_connection_info_and_disconnects() {
        // Arrange
        let mut ble = StubBle {
            connect_result: Ok(idl_transport::ConnectionInfo {
                device_id: "AA:BB".to_string(),
                firmware_version: "1.5.0".to_string(),
                connected: true,
            }),
            ..Default::default()
        };

        // Act
        let info = connect_via(&mut ble, "AA:BB").await.unwrap();

        // Assert
        assert_eq!(info.device_id, "AA:BB");
        assert_eq!(info.firmware_version, "1.5.0");
        assert!(info.connected);
        assert_eq!(ble.disconnect_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn connect_via_transport_failure_maps_to_ble_ipc_error() {
        // Arrange
        let mut ble = StubBle {
            connect_result: Err(TransportError::new(TransportErrorKind::Ble, "device not found")),
            ..Default::default()
        };

        // Act
        let err = connect_via(&mut ble, "AA:BB").await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Ble);
        assert_eq!(err.message, "device not found");
    }

    #[tokio::test(start_paused = true)]
    async fn switch_to_wifi_mode_reports_on_after_two_polls_succeeds() {
        // Arrange
        let mut reads = VecDeque::new();
        reads.push_back(Some(false));
        reads.push_back(Some(true));
        let ble = StubBle { wifi_on_reads: StdMutex::new(reads), ..Default::default() };

        // Act
        let result = switch_to_wifi_mode(&ble).await;

        // Assert
        assert!(result.is_ok());
        assert_eq!(ble.send_command_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn switch_to_wifi_mode_never_reports_on_errors_ble_after_exhausting_polls() {
        // Arrange
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(false)])), ..Default::default() };

        // Act
        let err = switch_to_wifi_mode(&ble).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Ble);
    }

    #[tokio::test(start_paused = true)]
    async fn list_files_via_two_entries_maps_field_for_field() {
        // Arrange
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let wifi = StubWifi {
            list_files_result: Ok(vec![
                idl_transport::DeviceFile {
                    name: "session_001.idl0".to_string(),
                    size_bytes: 12345,
                    session_id: Some("0123456789abcdef0123456789abcdef".to_string()),
                },
                idl_transport::DeviceFile { name: "session_002.idl0".to_string(), size_bytes: 999, session_id: None },
            ]),
            ..Default::default()
        };

        // Act
        let files = list_files_via(&ble, &wifi).await.unwrap();

        // Assert
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].size_bytes, 12345);
        assert_eq!(files[0].session_id.as_deref(), Some("0123456789abcdef0123456789abcdef"));
        assert_eq!(files[1].session_id, None);
    }

    #[tokio::test(start_paused = true)]
    async fn list_files_via_wifi_error_maps_to_wifi_ipc_error() {
        // Arrange
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let wifi = StubWifi { list_files_result: Err(TransportError::new(TransportErrorKind::Wifi, "GET /files failed")), ..Default::default() };

        // Act
        let err = list_files_via(&ble, &wifi).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Wifi);
    }

    #[tokio::test(start_paused = true)]
    async fn download_via_writes_blob_and_returns_matching_hash_and_size() {
        // Arrange
        let data_dir = tempfile::tempdir().unwrap();
        let content = b"session bytes".to_vec();
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let wifi = StubWifi {
            list_files_result: Ok(vec![idl_transport::DeviceFile { name: "session_001.idl0".to_string(), size_bytes: content.len() as u64, session_id: None }]),
            download_bytes: Ok(content.clone()),
        };
        let mut progress_calls = Vec::new();

        // Act
        let result = download_via(&ble, &wifi, "session_001.idl0", data_dir.path(), |done, total| progress_calls.push((done, total)))
            .await
            .unwrap();

        // Assert
        let expected_hash = sha256_hex(&content);
        assert_eq!(result.sha256, expected_hash);
        assert_eq!(result.size_bytes, content.len() as u64);
        // C4 §2's fixed sharded convention: first 2 hex chars as a
        // subdirectory, remaining 62 as the filename — not a flat
        // `blobs/sha256/<64-hex>` path.
        let blob_path = data_dir.path().join("blobs/sha256").join(&expected_hash[..2]).join(&expected_hash[2..]);
        assert_eq!(result.path, blob_path.display().to_string());
        assert_eq!(std::fs::read(&blob_path).unwrap(), content);
        assert!(
            !data_dir.path().join("blobs/sha256").join(&expected_hash).exists(),
            "must not also land at the flat (unsharded) path"
        );
        assert_eq!(progress_calls, vec![(content.len() as u64, Some(content.len() as u64))]);
    }

    #[tokio::test(start_paused = true)]
    async fn download_via_shards_the_blob_path_by_the_first_two_hex_chars_of_the_digest() {
        // Arrange: content chosen so its SHA-256 digest starts with "ab",
        // matching C4 §2's example convention directly (found by brute-force
        // search over a counter suffix, not hand-picked to hide a bug).
        let content = (0u32..)
            .map(|n| format!("shard-fixture-{n}").into_bytes())
            .find(|candidate| sha256_hex(candidate).starts_with("ab"))
            .expect("some counter value hashes to a digest starting with ab");
        let expected_hash = sha256_hex(&content);
        assert!(expected_hash.starts_with("ab"));
        let data_dir = tempfile::tempdir().unwrap();
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let wifi = StubWifi {
            list_files_result: Ok(vec![idl_transport::DeviceFile { name: "shard.idl0".to_string(), size_bytes: content.len() as u64, session_id: None }]),
            download_bytes: Ok(content.clone()),
        };

        // Act
        let result = download_via(&ble, &wifi, "shard.idl0", data_dir.path(), |_, _| {}).await.unwrap();

        // Assert: lands at blobs/sha256/ab/<remaining 62 hex>, not blobs/sha256/<64 hex>.
        let expected_path = data_dir.path().join("blobs/sha256").join("ab").join(&expected_hash[2..]);
        assert_eq!(result.path, expected_path.display().to_string());
        assert_eq!(std::fs::read(&expected_path).unwrap(), content);
    }

    #[tokio::test(start_paused = true)]
    async fn download_via_redownloading_identical_content_is_a_no_op_second_write() {
        // Arrange
        let data_dir = tempfile::tempdir().unwrap();
        let content = b"same bytes twice".to_vec();
        let make_ble = || StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let make_wifi = || StubWifi {
            list_files_result: Ok(vec![idl_transport::DeviceFile { name: "a.idl0".to_string(), size_bytes: content.len() as u64, session_id: None }]),
            download_bytes: Ok(content.clone()),
        };

        // Act
        let first = download_via(&make_ble(), &make_wifi(), "a.idl0", data_dir.path(), |_, _| {}).await.unwrap();
        let second = download_via(&make_ble(), &make_wifi(), "a.idl0", data_dir.path(), |_, _| {}).await.unwrap();

        // Assert
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.path, second.path);
        let blob_path = Path::new(&second.path);
        assert_eq!(std::fs::read(blob_path).unwrap(), content);
        // The temp file from either download must not linger.
        let leftover: Vec<_> = std::fs::read_dir(data_dir.path().join("tmp")).unwrap().collect();
        assert!(leftover.is_empty(), "temp dir should be empty after both downloads: {leftover:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn download_via_unknown_file_name_errors_not_found_before_any_write() {
        // Arrange
        let data_dir = tempfile::tempdir().unwrap();
        let ble = StubBle { wifi_on_reads: StdMutex::new(VecDeque::from([Some(true)])), ..Default::default() };
        let wifi = StubWifi { list_files_result: Ok(vec![]), ..Default::default() };

        // Act
        let err = download_via(&ble, &wifi, "missing.idl0", data_dir.path(), |_, _| {}).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
        assert!(!data_dir.path().join("blobs/sha256").exists());
    }

    #[test]
    fn validate_config_json_locally_well_formed_ok_malformed_invalid_argument() {
        // Arrange
        let good = r#"{"config_version":1}"#;
        let bad = "{ not json";

        // Act
        let good_result = validate_config_json_locally(good);
        let bad_result = validate_config_json_locally(bad);

        // Assert
        assert!(good_result.is_ok());
        assert_eq!(bad_result.unwrap_err().kind, IpcErrorKind::InvalidArgument);
    }

    #[tokio::test]
    async fn push_config_via_malformed_json_rejected_before_any_transport_call() {
        // Arrange
        let ble = StubBle::default();

        // Act
        let err = push_config_via(&ble, "{ not json").await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(ble.send_command_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn push_config_via_valid_json_forwards_to_ble_push_config() {
        // Arrange
        let ble = StubBle { push_config_result: Ok(()), ..Default::default() };

        // Act
        let result = push_config_via(&ble, r#"{"config_version":1}"#).await;

        // Assert
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn push_config_via_device_rejection_maps_to_config_ipc_error() {
        // Arrange
        let ble = StubBle {
            push_config_result: Err(TransportError::new(TransportErrorKind::Config, "device rejected config")),
            ..Default::default()
        };

        // Act
        let err = push_config_via(&ble, r#"{"config_version":1}"#).await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Config);
    }
}
