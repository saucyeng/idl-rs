//! `AndroidBle`: SPEC §14a's `BleTransport` over the Kotlin plugin's GATT
//! commands (SPEC §14b.2). Unlike `btleplug` on Windows, the plugin hands
//! back the device's raw write status, so §7.2's ACK byte is read directly.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use base64::Engine as _;
use idl_transport::ble_config;
use idl_transport::ble_control::{AckCode, ControlCommand};
use idl_transport::ble_status::{self, DeviceStatus};
use idl_transport::ble_transport::{uuids, BleTransport};
use idl_transport::{ConnectionInfo, DiscoveredDevice, TransportError, TransportErrorKind};
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio::sync::mpsc;

use super::call;

/// SPEC §7.2's FF06 read loop ends on an empty read; this caps a device
/// that never sends one (idl0-app uses the same bound).
const MAX_CONFIG_READS: usize = 1024;

/// ATT write-request overhead, bytes (opcode + handle).
const ATT_WRITE_OVERHEAD: usize = 3;

/// One event from a plugin stream (`bleScan`/`bleConnect`'s `onEvent`).
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", rename_all_fields = "camelCase")]
enum BleEvent {
    Device { id: String, name: String, rssi_dbm: i32, service_uuids: Vec<String> },
    Done,
    Error { message: String },
    Status { value: String },
    Disconnected { status: i32 },
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ScanArgs {
    timeout_ms: u64,
    on_event: Channel<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ConnectArgs<'a> {
    id: &'a str,
    on_event: Channel<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct IdArgs<'a> {
    id: &'a str,
}

#[derive(serde::Serialize)]
struct CharArgs<'a> {
    id: &'a str,
    uuid: &'a str,
}

#[derive(serde::Serialize)]
struct WriteArgs<'a> {
    id: &'a str,
    uuid: &'a str,
    /// Base64 of the bytes.
    value: String,
}

#[derive(serde::Deserialize)]
struct Granted {
    granted: bool,
}

#[derive(serde::Deserialize)]
struct Connected {
    mtu: usize,
    name: String,
}

#[derive(serde::Deserialize)]
struct ReadResult {
    /// Base64 of the characteristic value.
    value: String,
}

#[derive(serde::Deserialize)]
struct WriteResult {
    /// Raw ATT/GATT status from `onCharacteristicWrite` (SPEC §14b.2).
    status: i32,
}

fn ble_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Ble, message)
}

fn decode(value: &str) -> Result<Vec<u8>, TransportError> {
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|e| ble_error(format!("the device plugin sent invalid base64: {e}")))
}

fn encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Asks for the Bluetooth runtime permissions if they aren't granted yet
/// (SPEC §14b.2); a refusal is `PermissionDenied`, not a BLE failure.
async fn ensure_permissions() -> Result<(), TransportError> {
    let result: Granted = call("bleEnsurePermissions", (), TransportErrorKind::Ble).await?;
    if result.granted {
        Ok(())
    } else {
        Err(TransportError::new(
            TransportErrorKind::PermissionDenied,
            "Bluetooth permission was not granted — allow Nearby devices for idl1 in system settings",
        ))
    }
}

/// Maps a Control write's raw status onto SPEC §7.2's ACK codes.
fn ack_result(cmd: &str, status: i32) -> Result<(), TransportError> {
    let code = u8::try_from(status).map(AckCode::from_byte).unwrap_or(AckCode::Unknown(u8::MAX));
    if code.is_success() {
        Ok(())
    } else {
        Err(ble_error(format!("the device refused {cmd}: ACK {status:#04x} ({code:?})")))
    }
}

/// State shared with the connection's event channel.
#[derive(Default)]
struct Link {
    /// `watch_status` subscribers; closed senders are dropped on send.
    watchers: Vec<mpsc::Sender<DeviceStatus>>,
    /// Set by the plugin's `disconnected` event.
    lost: bool,
}

/// Android `BleTransport` over the Kotlin `DevicePlugin`. `device_id` is
/// the logger's Bluetooth MAC address, as `bleScan` reports it.
pub struct AndroidBle {
    /// Set by `connect`, cleared by `disconnect`.
    connected: StdMutex<Option<Live>>,
    link: Arc<StdMutex<Link>>,
}

/// The live connection's identity and negotiated MTU, bytes.
#[derive(Clone)]
struct Live {
    device_id: String,
    mtu: usize,
}

impl AndroidBle {
    /// Matches `BtleplugBle::new`'s shape so `PlatformBle::new()` works on
    /// every platform. Cheap: the plugin is a process-wide singleton.
    pub async fn new() -> Result<Self, TransportError> {
        Ok(Self { connected: StdMutex::new(None), link: Arc::new(StdMutex::new(Link::default())) })
    }

    fn current(&self) -> Result<Live, TransportError> {
        if self.link.lock().unwrap().lost {
            return Err(ble_error("the device disconnected"));
        }
        self.connected.lock().unwrap().clone().ok_or_else(|| ble_error("not connected — call connect() first"))
    }

    async fn read(&self, uuid: &str) -> Result<Vec<u8>, TransportError> {
        let c = self.current()?;
        let result: ReadResult =
            call("bleRead", CharArgs { id: &c.device_id, uuid }, TransportErrorKind::Ble).await?;
        decode(&result.value)
    }

    /// Write with response; returns the raw status (SPEC §14b.2).
    async fn write(&self, uuid: &str, bytes: &[u8]) -> Result<i32, TransportError> {
        let c = self.current()?;
        let result: WriteResult = call(
            "bleWrite",
            WriteArgs { id: &c.device_id, uuid, value: encode(bytes) },
            TransportErrorKind::Ble,
        )
        .await?;
        Ok(result.status)
    }
}

impl BleTransport for AndroidBle {
    async fn scan(&self, timeout: Duration) -> Result<mpsc::Receiver<DiscoveredDevice>, TransportError> {
        ensure_permissions().await?;
        let (tx, rx) = mpsc::channel(32);
        // The sender lives in the channel handler until `done`, which drops it
        // and so ends the receiver's stream.
        let tx = StdMutex::new(Some(tx));
        let on_event = Channel::new(move |body: InvokeResponseBody| {
            let Ok(event) = body.deserialize::<BleEvent>() else { return Ok(()) };
            let mut guard = tx.lock().unwrap();
            match event {
                BleEvent::Device { id, name, rssi_dbm, service_uuids } => {
                    if let Some(tx) = guard.as_ref() {
                        let _ = tx.try_send(DiscoveredDevice { device_id: id, name, rssi_dbm, service_uuids });
                    }
                }
                BleEvent::Done => *guard = None,
                BleEvent::Error { message } => {
                    eprintln!("bleScan: {message}");
                    *guard = None;
                }
                _ => {}
            }
            Ok(())
        });
        call::<serde_json::Value>("bleScan", ScanArgs { timeout_ms: timeout.as_millis() as u64, on_event }, TransportErrorKind::Ble)
            .await?;
        Ok(rx)
    }

    async fn connect(&mut self, device_id: &str) -> Result<ConnectionInfo, TransportError> {
        ensure_permissions().await?;
        let link = Arc::new(StdMutex::new(Link::default()));
        let events = link.clone();
        let on_event = Channel::new(move |body: InvokeResponseBody| {
            let Ok(event) = body.deserialize::<BleEvent>() else { return Ok(()) };
            let mut link = events.lock().unwrap();
            match event {
                BleEvent::Status { value } => {
                    if let Ok(bytes) = decode(&value) {
                        let status = ble_status::parse_status(&String::from_utf8_lossy(&bytes));
                        link.watchers.retain(|w| w.try_send(status.clone()).is_ok() || !w.is_closed());
                    }
                }
                BleEvent::Disconnected { .. } => {
                    link.lost = true;
                    link.watchers.clear();
                }
                _ => {}
            }
            Ok(())
        });
        let connected: Connected =
            call("bleConnect", ConnectArgs { id: device_id, on_event }, TransportErrorKind::Ble).await?;
        self.link = link;
        *self.connected.lock().unwrap() =
            Some(Live { device_id: device_id.to_string(), mtu: connected.mtu });

        let status = self.read_status().await?;
        Ok(ConnectionInfo {
            device_id: device_id.to_string(),
            name: connected.name,
            firmware_version: status.firmware.unwrap_or_default(),
            connected: true,
        })
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        let taken = self.connected.lock().unwrap().take();
        if let Some(c) = taken {
            call::<serde_json::Value>("bleDisconnect", IdArgs { id: &c.device_id }, TransportErrorKind::Ble).await?;
        }
        Ok(())
    }

    async fn read_status(&self) -> Result<DeviceStatus, TransportError> {
        let bytes = self.read(uuids::STATUS).await?;
        Ok(ble_status::parse_status(&String::from_utf8_lossy(&bytes)))
    }

    async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError> {
        self.current()?;
        let (tx, rx) = mpsc::channel(32);
        self.link.lock().unwrap().watchers.push(tx);
        Ok(rx)
    }

    async fn send_command(&self, cmd: ControlCommand) -> Result<(), TransportError> {
        let status = self.write(uuids::CONTROL, &[cmd.as_byte()]).await?;
        ack_result(&format!("{cmd:?}"), status)
    }

    async fn push_config(&self, config_json: &[u8]) -> Result<(), TransportError> {
        ble_config::validate_config_size(config_json)?;
        self.send_command(ControlCommand::ConfigBegin).await?;
        let chunk_size_bytes = self.current()?.mtu.saturating_sub(ATT_WRITE_OVERHEAD).max(20);
        for chunk in ble_config::chunk_config(config_json, chunk_size_bytes) {
            let status = self.write(uuids::CONFIG_RX, chunk).await?;
            ack_result("a Config RX chunk", status)?;
        }
        self.send_command(ControlCommand::ConfigCommit).await
    }

    async fn read_config(&self) -> Result<Vec<u8>, TransportError> {
        self.send_command(ControlCommand::ConfigReadBegin).await?;
        let mut chunks = Vec::new();
        loop {
            if chunks.len() >= MAX_CONFIG_READS {
                return Err(ble_error("config read-back never ended (no empty FF06 read)"));
            }
            let chunk = self.read(uuids::CONFIG_TX).await?;
            let is_eof = chunk.is_empty();
            chunks.push(chunk);
            if is_eof {
                break;
            }
        }
        let mut chunks = chunks.into_iter();
        ble_config::reassemble_config_reads(move || Ok(chunks.next().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ack_result_success_ok_precondition_is_a_refusal_not_a_retry() {
        let ok = ack_result("WifiOn", 0);
        let refused = ack_result("ConfigReadBegin", 0x81);

        assert!(ok.is_ok());
        assert!(refused.unwrap_err().message.contains("0x81"));
    }

    #[test]
    fn ack_result_negative_status_is_unknown_not_success() {
        let result = ack_result("WifiOn", -1);

        assert!(result.is_err());
    }
}
