//! `BleTransport`: the trait L9's mobile plugins implement against, and
//! `BtleplugBle`, the `btleplug`-backed desktop implementation (SPEC §7).

use std::time::Duration;

use btleplug::api::{
    BDAddr, Central, CentralEvent, Characteristic, Manager as _, Peripheral as _, ScanFilter,
    WriteType,
};
use btleplug::platform::{Adapter, Manager, Peripheral, PeripheralId};
use futures::StreamExt;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::ble_config;
use crate::ble_control::ControlCommand;
use crate::ble_status::{self, DeviceStatus};
use crate::device::{ConnectionInfo, DiscoveredDevice};
use crate::{TransportError, TransportErrorKind};

/// SPEC §7.1 GATT service and characteristic UUIDs.
pub mod uuids {
    /// Service UUID scanned for (SPEC §7.4 step 1).
    pub const SERVICE: &str = "000000ff-0000-1000-8000-00805f9b34fb";
    /// IMU data characteristic (streamed samples, not parsed by this crate).
    pub const IMU_DATA: &str = "0000ff01-0000-1000-8000-00805f9b34fb";
    /// GPS data characteristic (streamed samples, not parsed by this crate).
    pub const GPS_DATA: &str = "0000ff02-0000-1000-8000-00805f9b34fb";
    /// Control characteristic (FF03) — single-byte commands (SPEC §7.2).
    pub const CONTROL: &str = "0000ff03-0000-1000-8000-00805f9b34fb";
    /// Status characteristic (FF04) — read/notify (SPEC §7.3).
    pub const STATUS: &str = "0000ff04-0000-1000-8000-00805f9b34fb";
    /// Config RX characteristic (FF05) — chunked config push (SPEC §7.2).
    pub const CONFIG_RX: &str = "0000ff05-0000-1000-8000-00805f9b34fb";
    /// Config TX characteristic (FF06) — chunked config read-back (SPEC §7.2).
    pub const CONFIG_TX: &str = "0000ff06-0000-1000-8000-00805f9b34fb";
}

/// Abstraction over the BLE control/status/config link to one IDL0 device
/// (SPEC §7). `BtleplugBle` (below) implements it for desktop; a Tauri
/// mobile plugin (L9) implements it against the platform BLE stack behind
/// the same trait, so `idl-rs-tauri`'s device commands (C3 §3.8) are one
/// code path on every platform — the desktop and mobile Tauri app crates
/// each pick their concrete implementation at compile time (separate build
/// targets), so this trait does not need to be `dyn`-safe; see Open
/// question 2.
pub trait BleTransport {
    /// Scans for `uuids::SERVICE` for up to `timeout`. Devices found are
    /// sent on the returned channel as they're discovered — resolves this
    /// lane's shape for C3 §3.8's `ble_scan` (C3 open question 6: "the
    /// transport crate's actual `btleplug` usage will make the natural
    /// shape obvious" — a live channel, not a batch return, because
    /// `btleplug`'s own scan API is event-based). L5 drains this channel
    /// into the command's `Channel<DeviceDiscovered>` argument.
    async fn scan(
        &self,
        timeout: Duration,
    ) -> Result<mpsc::Receiver<DiscoveredDevice>, TransportError>;

    /// Connects, negotiates MTU (see Open question 7 — `btleplug`'s MTU API
    /// is not uniform across platforms), enables Status notifications and
    /// reads the initial status (SPEC §7.4).
    async fn connect(&mut self, device_id: &str) -> Result<ConnectionInfo, TransportError>;

    /// Disconnects the current GATT link, if any (SPEC §7.4).
    async fn disconnect(&mut self) -> Result<(), TransportError>;

    /// One-shot status read (SPEC §7.4 step 4).
    async fn read_status(&self) -> Result<DeviceStatus, TransportError>;

    /// Subscribes to Status (FF04) notifications; each parsed `DeviceStatus`
    /// is sent on the returned channel until `disconnect` or the sender is
    /// dropped device-side.
    async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError>;

    /// Writes one command byte to Control (FF03), Write with Response, and
    /// maps the ACK (SPEC §7.2). Returns `Ok(())` only for `AckCode::Success`
    /// — any other code is a `TransportErrorKind::Ble` with the code in the
    /// message (mutex refusals per SPEC §7.2 "Mutex" included — this trait
    /// does not special-case them, callers read the message).
    async fn send_command(&self, cmd: ControlCommand) -> Result<(), TransportError>;

    /// Pushes an already-validated config JSON string over FF05, framed
    /// BEGIN/chunks/COMMIT (SPEC §7.2, `ble_config::chunk_config`). Returns
    /// once COMMIT is ACKed `Success` — the device then reboots; awaiting
    /// the reboot/reconnect and the read-back verify (SPEC §7.2) is the
    /// caller's job (L5), composed from this method plus `connect` and
    /// `read_config`, not folded into one call here.
    async fn push_config(&self, config_json: &[u8]) -> Result<(), TransportError>;

    /// Reads the live config back via CMD_CONFIG_READ_BEGIN + the FF06 read
    /// loop (SPEC §7.2, `ble_config::reassemble_config_reads`). Returns
    /// `TransportErrorKind::Ble` if READ_BEGIN ACKs `0x81` (no config file).
    async fn read_config(&self) -> Result<Vec<u8>, TransportError>;
}

/// Builds a `TransportErrorKind::Ble` error with `message`.
fn ble_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Ble, message)
}

/// Finds the characteristic `uuid` (one of the `uuids` module constants) on
/// `peripheral`'s already-discovered GATT table. Errors if
/// `discover_services` hasn't run (empty table) or the device doesn't
/// implement SPEC §7.1's expected characteristic set.
fn characteristic_by_uuid(
    peripheral: &Peripheral,
    uuid: &str,
) -> Result<Characteristic, TransportError> {
    let target = Uuid::parse_str(uuid).expect("uuids module constants are valid UUID strings");
    peripheral
        .characteristics()
        .into_iter()
        .find(|c| c.uuid == target)
        .ok_or_else(|| {
            ble_error(format!(
                "characteristic {uuid} not found on the connected device — was discover_services run?"
            ))
        })
}

/// `btleplug`-backed desktop `BleTransport`. Holds the platform `Adapter`
/// and, once connected, the `btleplug::platform::Peripheral`.
pub struct BtleplugBle {
    /// The Bluetooth adapter picked by `new()` (first one the platform
    /// reports). `btleplug`'s `Adapter` is cheaply `Clone` (an `Arc`
    /// handle), so this is shared with the background task `scan` spawns.
    adapter: Adapter,
    /// Set by `connect`, cleared by `disconnect`. A `Mutex` rather than a
    /// plain field because `connect`/`disconnect` take `&mut self` but every
    /// other method (`read_status`, `send_command`, ...) takes `&self` and
    /// needs to read it — `Peripheral` is itself a cheap `Clone` handle, so
    /// callers clone out of the lock rather than holding it.
    peripheral: Mutex<Option<Peripheral>>,
}

impl BtleplugBle {
    /// Creates a manager and picks the first available adapter. Returns
    /// `TransportErrorKind::Ble` if no Bluetooth adapter is present.
    pub async fn new() -> Result<Self, TransportError> {
        let manager = Manager::new()
            .await
            .map_err(|e| ble_error(format!("btleplug manager initialisation failed: {e}")))?;
        let adapters = manager
            .adapters()
            .await
            .map_err(|e| ble_error(format!("failed to list Bluetooth adapters: {e}")))?;
        let adapter = adapters
            .into_iter()
            .next()
            .ok_or_else(|| ble_error("no Bluetooth adapter available"))?;
        Ok(Self { adapter, peripheral: Mutex::new(None) })
    }

    /// Returns the currently-connected `Peripheral`, or a `Ble` error if
    /// `connect` hasn't succeeded (or `disconnect` has run since).
    async fn connected_peripheral(&self) -> Result<Peripheral, TransportError> {
        self.peripheral
            .lock()
            .await
            .clone()
            .ok_or_else(|| ble_error("not connected — call connect() first"))
    }
}

impl BleTransport for BtleplugBle {
    /// SPEC §7.4 step 1. `btleplug`'s scan surface is event-based
    /// (`Central::events`/`start_scan`, `CentralEvent::DeviceDiscovered`),
    /// which is exactly why the trait above returns a live channel instead
    /// of a batch `Vec` (Open question 6). This crate deliberately carries
    /// no owned tokio runtime (Task 1) — the way this method still feeds a
    /// channel *while* the scan is running, without blocking this call
    /// until `timeout` elapses, is a task spawned onto whichever runtime is
    /// already driving this call (the Tauri app's, at L5), never a runtime
    /// this crate creates itself. That's why Task 7 added tokio's `rt`
    /// feature to Cargo.toml — not present in Task 1's original list, and a
    /// genuine judgment call rather than something the plan fixed in
    /// advance (see the Cargo.toml comment and this task's report).
    async fn scan(
        &self,
        timeout: Duration,
    ) -> Result<mpsc::Receiver<DiscoveredDevice>, TransportError> {
        let service =
            Uuid::parse_str(uuids::SERVICE).expect("uuids::SERVICE is a valid UUID constant");
        let mut events = self
            .adapter
            .events()
            .await
            .map_err(|e| ble_error(format!("failed to subscribe to adapter events: {e}")))?;
        self.adapter
            .start_scan(ScanFilter { services: vec![service] })
            .await
            .map_err(|e| ble_error(format!("failed to start BLE scan: {e}")))?;

        let (tx, rx) = mpsc::channel(32);
        let adapter = self.adapter.clone();
        tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + timeout;
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let event = match tokio::time::timeout(remaining, events.next()).await {
                    Ok(Some(event)) => event,
                    // Ok(None): the event stream ended. Err(_): timeout elapsed.
                    Ok(None) | Err(_) => break,
                };
                let id = match event {
                    CentralEvent::DeviceDiscovered(id) | CentralEvent::DeviceUpdated(id) => id,
                    _ => continue,
                };
                let Ok(peripheral) = adapter.peripheral(&id).await else { continue };
                let Ok(Some(props)) = peripheral.properties().await else { continue };
                let device = DiscoveredDevice {
                    device_id: id.to_string(),
                    name: props.local_name.unwrap_or_default(),
                    // `props.rssi` is `None` on some backends until the
                    // first advertisement report lands; -127 dBm (the
                    // conventional BLE "RSSI not available" sentinel, e.g.
                    // HCI's own out-of-range value) rather than 0, since a
                    // real reading is always negative and 0 would read as
                    // an implausibly strong signal (judgment call — SPEC
                    // doesn't define a sentinel because DiscoveredDevice's
                    // rssi_dbm has no Option wrapper to express "unknown").
                    rssi_dbm: props.rssi.map(i32::from).unwrap_or(-127),
                    service_uuids: props.services.iter().map(Uuid::to_string).collect(),
                };
                if tx.send(device).await.is_err() {
                    break;
                }
            }
            let _ = adapter.stop_scan().await;
        });
        Ok(rx)
    }

    /// SPEC §7.4 steps 2–4: connect, discover services, subscribe to Status
    /// notifications, read the initial status for `ConnectionInfo::
    /// firmware_version`. `device_id` is whatever `scan` handed back
    /// (`PeripheralId`'s `Display`, a colon-delimited MAC on this platform)
    /// — parsed back via `BDAddr`'s `FromStr` rather than re-scanning, so
    /// the device must already be known to this adapter (from a prior
    /// `scan` call) or this errors.
    async fn connect(&mut self, device_id: &str) -> Result<ConnectionInfo, TransportError> {
        let addr: BDAddr = device_id
            .parse()
            .map_err(|e| ble_error(format!("device id {device_id:?} is not a BLE address: {e}")))?;
        let id: PeripheralId = addr.into();
        let peripheral = self.adapter.peripheral(&id).await.map_err(|e| {
            ble_error(format!("device {device_id} not known to this adapter — scan for it first: {e}"))
        })?;
        peripheral
            .connect()
            .await
            .map_err(|e| ble_error(format!("connect to {device_id} failed: {e}")))?;
        peripheral
            .discover_services()
            .await
            .map_err(|e| ble_error(format!("service discovery on {device_id} failed: {e}")))?;

        let status_char = characteristic_by_uuid(&peripheral, uuids::STATUS)?;
        peripheral
            .subscribe(&status_char)
            .await
            .map_err(|e| ble_error(format!("subscribing to Status notifications failed: {e}")))?;
        let raw = peripheral
            .read(&status_char)
            .await
            .map_err(|e| ble_error(format!("initial Status read failed: {e}")))?;
        let status = ble_status::parse_status(&String::from_utf8_lossy(&raw));
        let firmware_version = status.firmware.unwrap_or_default();

        *self.peripheral.lock().await = Some(peripheral);

        Ok(ConnectionInfo { device_id: device_id.to_string(), firmware_version, connected: true })
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        let maybe_peripheral = self.peripheral.lock().await.take();
        if let Some(peripheral) = maybe_peripheral {
            peripheral
                .disconnect()
                .await
                .map_err(|e| ble_error(format!("disconnect failed: {e}")))?;
        }
        Ok(())
    }

    async fn read_status(&self) -> Result<DeviceStatus, TransportError> {
        let peripheral = self.connected_peripheral().await?;
        let characteristic = characteristic_by_uuid(&peripheral, uuids::STATUS)?;
        let raw = peripheral
            .read(&characteristic)
            .await
            .map_err(|e| ble_error(format!("Status read failed: {e}")))?;
        Ok(ble_status::parse_status(&String::from_utf8_lossy(&raw)))
    }

    async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError> {
        let peripheral = self.connected_peripheral().await?;
        let status_uuid =
            Uuid::parse_str(uuids::STATUS).expect("uuids::STATUS is a valid UUID constant");
        let mut notifications = peripheral
            .notifications()
            .await
            .map_err(|e| ble_error(format!("failed to open the notification stream: {e}")))?;

        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            while let Some(notification) = notifications.next().await {
                if notification.uuid != status_uuid {
                    continue; // this peripheral's stream carries every subscribed characteristic
                }
                let text = String::from_utf8_lossy(&notification.value);
                let status = ble_status::parse_status(&text);
                if tx.send(status).await.is_err() {
                    break;
                }
            }
        });
        Ok(rx)
    }

    /// Writes one command byte to Control (FF03) with Write-with-Response.
    ///
    /// **Open question 8's real answer, found reading the pinned source:**
    /// `btleplug` 0.13.0's `Peripheral::write` returns `Result<()>`, not the
    /// ATT response byte — on Windows (`winrtble`), `write_value` calls
    /// `WriteValueWithOptionAsync` and only inspects the resulting
    /// `GattCommunicationStatus` (`Success`/`Unreachable`/`ProtocolError`/
    /// `AccessDenied`); the actual application error code SPEC §7.2 defines
    /// (`0x03`/`0x80`/`0x81`/`0x82`) is never read off the WinRT
    /// `GattWriteResult` and so never reaches this crate (see
    /// `winrtble/ble/characteristic.rs` in the pinned source, `write_value`).
    /// So `ble_control::AckCode::from_byte` (Task 3, pure and unit-tested)
    /// cannot be driven by a real byte here: `Ok(())` is the only signal we
    /// get for `AckCode::Success`, and every GATT-level failure — including
    /// SPEC's mutex refusals — surfaces as a `TransportErrorKind::Ble` whose
    /// message is `btleplug`'s own (coarser) error text, not a specific ACK
    /// code. This is a genuine platform-API limitation, not a design choice
    /// this crate could route around; flagged here for whoever reads
    /// SPEC §14a next (Task 8) and for L9's mobile plugins, who may find a
    /// richer readback on their own platform's BLE stack.
    async fn send_command(&self, cmd: ControlCommand) -> Result<(), TransportError> {
        let peripheral = self.connected_peripheral().await?;
        let characteristic = characteristic_by_uuid(&peripheral, uuids::CONTROL)?;
        peripheral
            .write(&characteristic, &[cmd.as_byte()], WriteType::WithResponse)
            .await
            .map_err(|e| ble_error(format!("Control write for {cmd:?} failed: {e}")))
    }

    async fn push_config(&self, config_json: &[u8]) -> Result<(), TransportError> {
        ble_config::validate_config_size(config_json)?;
        self.send_command(ControlCommand::ConfigBegin).await?;

        let peripheral = self.connected_peripheral().await?;
        let characteristic = characteristic_by_uuid(&peripheral, uuids::CONFIG_RX)?;
        // SPEC §7.2: "MTU-sized chunks". ATT write-request overhead is 3
        // bytes (1 opcode + 2 attribute handle). `Peripheral::mtu()` is a
        // *synchronous* readback (`btleplug::api::Peripheral::mtu`) of the
        // platform-negotiated MTU, defaulting to `DEFAULT_MTU_SIZE` (23)
        // until an exchange completes. On Windows the `winrtble` backend
        // updates this automatically on connect via a max-PDU-size callback
        // (see `winrtble/peripheral.rs::connect`) — no manual MTU-request
        // call exists or is needed on this platform, which resolves Open
        // question 7 for desktop: 23-3=20 is what actually ships whenever
        // this read races the callback, matching SPEC §14a's stated
        // fallback exactly (not a theoretical fallback path).
        let chunk_size_bytes = peripheral.mtu().saturating_sub(3).max(1) as usize;
        for chunk in ble_config::chunk_config(config_json, chunk_size_bytes) {
            peripheral
                .write(&characteristic, chunk, WriteType::WithResponse)
                .await
                .map_err(|e| ble_error(format!("Config RX chunk write failed: {e}")))?;
        }

        self.send_command(ControlCommand::ConfigCommit).await
    }

    async fn read_config(&self) -> Result<Vec<u8>, TransportError> {
        self.send_command(ControlCommand::ConfigReadBegin).await?;

        let peripheral = self.connected_peripheral().await?;
        let characteristic = characteristic_by_uuid(&peripheral, uuids::CONFIG_TX)?;

        // `ble_config::reassemble_config_reads` (Task 4, pure/tested) takes
        // a *synchronous* `FnMut` — real Config TX reads are async GATT
        // calls, so a sync closure can't drive them directly. Rather than
        // re-deriving the "concatenate chunks until an empty read" rule
        // here (Task 7's "no protocol decision gets made twice"), this
        // pre-fetches every chunk (including the terminating empty one)
        // through the real async read loop, then replays them through the
        // same pure concatenation logic — same GATT reads either way, just
        // decoupled from the concatenation step.
        let mut chunks = Vec::new();
        loop {
            let chunk = peripheral
                .read(&characteristic)
                .await
                .map_err(|e| ble_error(format!("Config TX read failed: {e}")))?;
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

/// Task 8: the composed download+config *sequencing* — BLE `send_command`
/// entering WiFi mode, polled via `read_status`, followed by the WiFi calls
/// L5's `download_file`/`list_device_files` Tauri commands (C3 §3.8)
/// actually run in that order. Proved once here with a canned-response
/// `BleTransport` stub (not real `btleplug` — Open question 9: this crate
/// has no in-process fake GATT peripheral) plus Task 7's mock HTTP device
/// server, so L5's own tests can focus on the IPC glue rather than
/// re-deriving this ordering.
#[cfg(test)]
mod sequencing {
    use std::sync::Mutex as StdMutex;

    use super::{ble_error, BleTransport};
    use crate::ble_control::ControlCommand;
    use crate::ble_status::DeviceStatus;
    use crate::device::{ConnectionInfo, DiscoveredDevice};
    use crate::wifi_transport::integration::spawn_mock_server;
    use crate::wifi_transport::{verify_device_identity, ReqwestWifi, WifiTransport};
    use crate::TransportError;

    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Canned `BleTransport`: `send_command(WifiOn)` flips an in-memory
    /// flag; `read_status` reports it back via `DeviceStatus::wifi_on`.
    /// Every other method is unused by this test and returns a `Ble` error
    /// rather than `unimplemented!()`, so an accidental call fails the test
    /// with a normal assertion instead of panicking the whole harness.
    struct StubBle {
        /// `true` once `send_command(WifiOn)` has been called.
        wifi_on: StdMutex<bool>,
    }

    impl StubBle {
        fn new() -> Self {
            Self { wifi_on: StdMutex::new(false) }
        }
    }

    impl BleTransport for StubBle {
        async fn scan(
            &self,
            _timeout: Duration,
        ) -> Result<mpsc::Receiver<DiscoveredDevice>, TransportError> {
            Err(ble_error("StubBle::scan is not exercised by this test"))
        }

        async fn connect(&mut self, _device_id: &str) -> Result<ConnectionInfo, TransportError> {
            Err(ble_error("StubBle::connect is not exercised by this test"))
        }

        async fn disconnect(&mut self) -> Result<(), TransportError> {
            Ok(())
        }

        async fn read_status(&self) -> Result<DeviceStatus, TransportError> {
            let wifi_on = *self.wifi_on.lock().expect("stub mutex is never poisoned");
            Ok(DeviceStatus { wifi_on: Some(wifi_on), ..DeviceStatus::default() })
        }

        async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError> {
            Err(ble_error("StubBle::watch_status is not exercised by this test"))
        }

        async fn send_command(&self, cmd: ControlCommand) -> Result<(), TransportError> {
            if cmd == ControlCommand::WifiOn {
                *self.wifi_on.lock().expect("stub mutex is never poisoned") = true;
            }
            Ok(())
        }

        async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
            Err(ble_error("StubBle::push_config is not exercised by this test"))
        }

        async fn read_config(&self) -> Result<Vec<u8>, TransportError> {
            Err(ble_error("StubBle::read_config is not exercised by this test"))
        }
    }

    #[tokio::test]
    async fn wifi_on_then_status_poll_then_wifi_download_runs_in_expected_order() {
        // Arrange
        let ble = StubBle::new();
        const CONTENT: &[u8] = b"session bytes";
        let (addr, _server) = spawn_mock_server(|path, _headers| {
            if path == "/ping" {
                let body = br#"{"device":"IDL0-A3F2","fw":"1.4.0","proto":1,"battery":80,"sd":"OK","mode":"wifi","ble":"on"}"#.to_vec();
                (
                    200,
                    "OK",
                    vec![("Content-Type".to_string(), "application/json".to_string())],
                    body,
                )
            } else if path == "/files" {
                let body = br#"[{"name":"session_001.idl0","size":13}]"#.to_vec();
                (
                    200,
                    "OK",
                    vec![("Content-Type".to_string(), "application/json".to_string())],
                    body,
                )
            } else {
                (200, "OK", Vec::new(), CONTENT.to_vec())
            }
        })
        .await;
        let wifi = ReqwestWifi::new(format!("http://{addr}"));

        // Act — the exact sequence L5's download-file command runs: BLE
        // WifiOn, poll status until wifi_on flips true, then the WiFi calls.
        ble.send_command(ControlCommand::WifiOn).await.unwrap();

        let mut wifi_on = false;
        for _ in 0..10 {
            let status = ble.read_status().await.unwrap();
            if status.wifi_on == Some(true) {
                wifi_on = true;
                break;
            }
        }
        assert!(wifi_on, "wifi_on never observed via read_status polling");

        let ping = wifi.ping().await.unwrap();
        verify_device_identity(&ping, "IDL0-A3F2").unwrap();
        let files = wifi.list_files().await.unwrap();

        let mut sink = Vec::new();
        let mut on_progress = |_: u64, _: Option<u64>| {};
        let bytes_written =
            wifi.download(0, 0, &mut sink, &mut on_progress).await.unwrap();

        // Assert
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "session_001.idl0");
        assert_eq!(bytes_written, CONTENT.len() as u64);
        assert_eq!(sink, CONTENT);
    }
}
