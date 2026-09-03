//! `BleTransport`: the trait L9's mobile plugins implement against, and
//! `BtleplugBle`, the `btleplug`-backed desktop implementation (SPEC §7).

use std::time::Duration;
use tokio::sync::mpsc;

use crate::ble_control::ControlCommand;
use crate::ble_status::DeviceStatus;
use crate::device::{ConnectionInfo, DiscoveredDevice};
use crate::TransportError;

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

/// `btleplug`-backed desktop `BleTransport`. Holds the platform `Manager`/
/// `Adapter` and, once connected, the `btleplug::platform::Peripheral`.
pub struct BtleplugBle {
    // btleplug::platform::{Manager, Adapter, Peripheral} fields — exact
    // shape follows btleplug 0.13.0's actual API once Task 7 fills this
    // struct in (not pinned by any contract; see Open question 8).
}

impl BtleplugBle {
    /// Creates a manager and picks the first available adapter. Returns
    /// `TransportErrorKind::Ble` if no Bluetooth adapter is present.
    pub async fn new() -> Result<Self, TransportError> {
        unimplemented!(
            "Task 7 implementer: btleplug::platform::Manager::new(), \
            .adapters().await, pick adapters[0] or error TransportErrorKind::Ble \
            \"no Bluetooth adapter\""
        )
    }
}

impl BleTransport for BtleplugBle {
    // Task 7 implementer fills in each method body against btleplug 0.13.0's
    // actual Central/Peripheral trait API (Manager::adapters, Adapter::
    // start_scan(ScanFilter), CentralEvent::DeviceDiscovered, Peripheral::
    // connect/discover_services/write/read/subscribe/notifications). No
    // real-hardware unit tests here (Task 9 verifies against the device);
    // keep each method a thin translation to/from Tasks 2–4's pure types
    // so review can check it against SPEC line by line. Every body below is
    // `unimplemented!()` only so the crate compiles now — they panic if
    // called, and nothing calls them until Task 7.

    async fn scan(
        &self,
        _timeout: Duration,
    ) -> Result<mpsc::Receiver<DiscoveredDevice>, TransportError> {
        unimplemented!("Task 7 implementer: Adapter::start_scan(ScanFilter {{ services: vec![uuids::SERVICE] }}), forward CentralEvent::DeviceDiscovered onto the returned channel until timeout")
    }

    async fn connect(&mut self, _device_id: &str) -> Result<ConnectionInfo, TransportError> {
        unimplemented!("Task 7 implementer: Peripheral::connect, discover_services, enable Status (FF04) notifications, read the initial status (SPEC §7.4)")
    }

    async fn disconnect(&mut self) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: Peripheral::disconnect")
    }

    async fn read_status(&self) -> Result<DeviceStatus, TransportError> {
        unimplemented!("Task 7 implementer: Peripheral::read(STATUS characteristic), then ble_status::parse_status on the UTF-8 payload")
    }

    async fn watch_status(&self) -> Result<mpsc::Receiver<DeviceStatus>, TransportError> {
        unimplemented!("Task 7 implementer: Peripheral::subscribe(STATUS characteristic), forward parsed ble_status::parse_status notifications onto the returned channel")
    }

    async fn send_command(&self, _cmd: ControlCommand) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: Peripheral::write(CONTROL characteristic, &[cmd.as_byte()], WriteType::WithResponse), map the ACK byte via ble_control::AckCode::from_byte")
    }

    async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
        unimplemented!("Task 7 implementer: send_command(ConfigBegin), ble_config::chunk_config over CONFIG_RX writes, send_command(ConfigCommit)")
    }

    async fn read_config(&self) -> Result<Vec<u8>, TransportError> {
        unimplemented!("Task 7 implementer: send_command(ConfigReadBegin), ble_config::reassemble_config_reads driven by CONFIG_TX reads")
    }
}
