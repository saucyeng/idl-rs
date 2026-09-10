//! Firmware / OTA commands (C3 §3.8, rulings R197/R198).
//!
//! **The state machine lives here, not in the app** (R198): Rust owns the
//! push → reboot → reconnect → confirm sequence, including the reconnect
//! retry loop, and the frontend draws whatever state it is told. Every
//! transition is both stored in `state::Ota` (readable at any time through
//! [`ota_state`]) and emitted as an `ota_state_changed` event, so a UI that
//! mounts mid-flight sees the same thing as one that watched from the start.
//!
//! The device half is SPEC §6.1 `POST /ota` → reboot → §7.2
//! `CMD_OTA_CONFIRM`, with §7.3's `OTA: PENDING_VERIFY` line as the signal
//! that the new image is running but uncommitted. The catalog half is
//! `idl_transport::firmware_catalog`.
//!
//! Same `_via`-helper shape as `commands::device`: the `#[tauri::command]`
//! wrappers build the concrete transports and this module's tests exercise
//! the helpers against that module's `StubBle`/`StubWifi`.

use std::sync::Arc;
use std::time::Duration;

use idl_transport::ble_control::ControlCommand;
use idl_transport::ble_transport::{BleTransport, BtleplugBle};
use idl_transport::firmware_catalog::{self, FirmwareChannel, FirmwareRelease};
use idl_transport::wifi_transport::{OtaPushErrorKind, ReqwestWifi, WifiTransport, DEVICE_BASE_URL};
use idl_transport::TransportError;

use crate::commands::device::{switch_to_wifi_mode, ConnectionMap, Progress};
use crate::error::{IpcError, IpcErrorKind};
use crate::state::{Connections, Ota};

/// Where a firmware image comes from (R198): a `.bin` the user picked off
/// disk, always available, or a release from the configured catalog.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FirmwareSource {
    /// A local file the user chose. Never arms auto-confirm — the app cannot
    /// know what is in it, so the user commits or power-cycles by hand.
    File {
        /// Absolute path to the `.bin`.
        path: String,
    },
    /// A release from the catalog, named by its version (the `version` field
    /// of a [`FirmwareRelease`] from [`firmware_catalog`]).
    Catalog {
        /// Semver text, no leading `v`.
        version: String,
    },
}

/// Where the OTA sequence has got to (R198's machine, plus `downloading`).
///
/// `downloading` is this lane's own addition to the ruling's list: R198
/// enumerates the *device-facing* states, and a catalog push spends real
/// time fetching the image before any of them begin — the Flutter card the
/// ruling says to copy showed that step too ("Downloading update… {pct}%").
/// It never occurs for a `file` source.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum OtaState {
    /// Nothing in flight. The state every session starts in, and the one a
    /// finished flow is reset to when the next push begins.
    Idle,
    /// Fetching a catalog image. `total_bytes`/`pct` are `null` until the
    /// server reports a length.
    Downloading {
        done_bytes: u64,
        total_bytes: Option<u64>,
        /// 0–100, `null` when `total_bytes` is unknown.
        pct: Option<u8>,
    },
    /// Streaming the image to the device's `/ota` endpoint.
    Pushing { done_bytes: u64, total_bytes: u64, pct: u8 },
    /// The device accepted the image and is restarting into it.
    Rebooting,
    /// Retrying the BLE link, `attempt` of `max_attempts` (R198: every 3 s
    /// for 60 s).
    Reconnecting { attempt: u32, max_attempts: u32 },
    /// Reconnected, and the device reports SPEC §7.3's `OTA: PENDING_VERIFY`
    /// — the new image is running but will roll back on the next reboot
    /// unless committed. `auto_confirm_armed` is `true` only for a catalog
    /// push whose sha256 verified (R198), in which case this state is passed
    /// through rather than waited in.
    PendingVerify { auto_confirm_armed: bool },
    /// `CMD_OTA_CONFIRM` sent and accepted (or the device came back already
    /// committed) — the new image is permanent.
    Confirmed,
    /// The device came back running a different version than the one pushed:
    /// the bootloader reverted. Nothing to do but tell the user.
    RolledBack,
    /// The sequence stopped. Carries the same typed error the command
    /// rejected with.
    Failed { error: IpcError },
}

/// What one completed [`push_firmware`] did.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OtaOutcome {
    /// The terminal state — `confirmed`, `pending_verify` (waiting on the
    /// user) or `rolled_back`. A failure rejects instead of returning here.
    pub state: OtaState,
    /// The version pushed, when it is known: the catalog release's version,
    /// or `null` for a `.bin` off disk.
    pub pushed_version: Option<String>,
    /// The version the device reports after reconnecting (SPEC §7.3
    /// `Firmware:`), or `null` if it reported none.
    pub device_version: Option<String>,
    /// Whether this flow sent `CMD_OTA_CONFIRM` on the user's behalf.
    pub auto_confirmed: bool,
    /// Whether the image's sha256 was checked against a published sidecar
    /// before the push. Always `false` for a `.bin` off disk.
    pub sha256_verified: bool,
}

/// How long the sequence waits at each step. A struct rather than constants
/// so tests run the whole machine in milliseconds instead of minutes.
#[derive(Debug, Clone, Copy)]
pub struct OtaTimings {
    /// Time allowed for the device to restart into the new image before the
    /// first reconnect attempt. SPEC §6.1 says the device reboots ~500 ms
    /// after the `200`; 5 s is the Flutter panel's own value, kept because
    /// the boot itself (mount SD, bring BLE up) is the slow part.
    pub boot_delay: Duration,
    /// Gap between reconnect attempts (R198: every 3 s).
    pub reconnect_interval: Duration,
    /// Total time reconnecting before giving up (R198: for 60 s).
    pub reconnect_window: Duration,
}

impl Default for OtaTimings {
    fn default() -> Self {
        Self {
            boot_delay: Duration::from_secs(5),
            reconnect_interval: Duration::from_secs(3),
            reconnect_window: Duration::from_secs(60),
        }
    }
}

impl OtaTimings {
    /// How many reconnect attempts the window allows, never fewer than one.
    fn max_attempts(&self) -> u32 {
        let interval_ms = self.reconnect_interval.as_millis().max(1);
        ((self.reconnect_window.as_millis() / interval_ms) as u32).max(1)
    }
}

/// A firmware image ready to push, plus what is known about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedImage {
    /// The raw `.bin` bytes.
    pub bytes: Vec<u8>,
    /// The version this image is expected to report once running, when
    /// known — a catalog release's version. `None` for a `.bin` off disk.
    pub version: Option<String>,
    /// Whether the bytes matched a published sha256 sidecar. Only a `true`
    /// here can arm auto-confirm (R198).
    pub sha256_verified: bool,
}

/// Percent of `total` that `done` is, clamped to 100.
fn pct_of(done_bytes: u64, total_bytes: u64) -> u8 {
    if total_bytes == 0 {
        return 100;
    }
    ((done_bytes.saturating_mul(100) / total_bytes).min(100)) as u8
}

/// Maps a `POST /ota` failure onto the IPC error shape, keeping the device's
/// own body text as structured `detail` so the UI can quote the firmware
/// rather than paraphrase it (R198).
fn ota_push_ipc_error(e: idl_transport::wifi_transport::OtaPushError) -> IpcError {
    let kind = match e.kind {
        // The device answered and refused: not a transport failure. `Wifi`
        // is still the right C3 §2 row — the alternative, `DeviceRejected`,
        // means a non-success BLE `AckCode` (SPEC §7.2), a different wire
        // and a different meaning — so the three-way split travels in
        // `detail.ota_error` instead of inventing a kind per HTTP status.
        OtaPushErrorKind::Rejected | OtaPushErrorKind::DeviceError | OtaPushErrorKind::Transport => {
            IpcErrorKind::Wifi
        }
    };
    IpcError::with_detail(
        kind,
        e.message.clone(),
        serde_json::json!({
            "ota_error": e.kind,
            "status_code": e.status_code,
            "device_body": e.detail,
        }),
    )
}

/// Reads `path` off disk as a firmware image. A missing file is `not_found`,
/// anything else about the read is `io` (C3 §2).
pub fn load_image_from_file(path: &str) -> Result<LoadedImage, IpcError> {
    match std::fs::read(path) {
        Ok(bytes) if bytes.is_empty() => {
            Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("{path} is empty — not a firmware image")))
        }
        Ok(bytes) => Ok(LoadedImage { bytes, version: None, sha256_verified: false }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(IpcError::new(IpcErrorKind::NotFound, format!("no such firmware file: {path}")))
        }
        Err(e) => Err(IpcError::new(IpcErrorKind::Io, format!("could not read {path}: {e}"))),
    }
}

/// Finds `version` in `releases`. A version the catalog does not carry is
/// `not_found` rather than a silent "latest" substitution — the user asked
/// for a specific build.
pub fn release_by_version<'a>(
    releases: &'a [FirmwareRelease],
    version: &str,
) -> Result<&'a FirmwareRelease, IpcError> {
    releases.iter().find(|release| release.version == version).ok_or_else(|| {
        IpcError::new(IpcErrorKind::NotFound, format!("the catalog has no firmware release {version}"))
    })
}

/// The precondition half of a push (R198), read off one status frame: the
/// device must not be recording. "BLE connected" is checked by the caller
/// (there must be a managed connection to read this status through at all),
/// and an unparseable firmware version is deliberately *not* a refusal — a
/// manual push is still allowed, the app just says "unknown version".
fn check_preconditions(status: &idl_transport::ble_status::DeviceStatus) -> Result<(), IpcError> {
    if status.logging == Some(true) {
        return Err(IpcError::new(
            IpcErrorKind::DeviceRejected,
            "the device is recording — stop the session before updating firmware",
        ));
    }
    Ok(())
}

/// Whether auto-confirm may fire for this push (R198): catalog source, and
/// its sha256 verified. Additionally, when both versions are known, the
/// device must have come back running the version that was pushed — a
/// device reporting anything else has already rolled back, and confirming
/// would commit the *old* image (SPEC §27.7's "Auto-confirm", kept).
pub fn auto_confirm_armed(
    image: &LoadedImage,
    source_is_catalog: bool,
    device_version: Option<&str>,
) -> bool {
    if !source_is_catalog || !image.sha256_verified {
        return false;
    }
    match (image.version.as_deref(), device_version) {
        (Some(pushed), Some(running)) => pushed == running,
        _ => true,
    }
}

/// The transport-agnostic core of [`push_firmware`]: everything from the
/// precondition read to the terminal state, driven against whatever
/// `BleTransport`/`WifiTransport` it is handed.
///
/// `on_state` is called for every transition, in order. `on_progress` is
/// called with `(done_bytes, total_bytes, phase)` for the push itself; the
/// download half happens before this function is called (it needs the
/// network, which this layer's tests do not have).
///
/// The managed connection is **taken out of `connections` before the push**:
/// the device reboots out from under the link, so leaving a dead
/// `BtleplugBle` in the map would have every other command reuse it. A new
/// one is inserted when the reconnect succeeds.
#[allow(clippy::too_many_arguments)]
pub async fn push_firmware_via<T, F, Fut>(
    connections: &ConnectionMap<T>,
    device_id: &str,
    image: &LoadedImage,
    source_is_catalog: bool,
    wifi: &impl WifiTransport,
    new_ble: F,
    timings: &OtaTimings,
    on_state: &mut (dyn FnMut(OtaState) + Send),
    on_progress: &mut (dyn FnMut(u64, u64, &str) + Send),
) -> Result<OtaOutcome, IpcError>
where
    T: BleTransport,
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, TransportError>>,
{
    let connected = connections.lock().unwrap().get(device_id).cloned().ok_or_else(|| {
        IpcError::new(IpcErrorKind::NotFound, "connect to the device first")
    })?;

    // Preconditions, then WiFi mode, then the push — all over the link that
    // is about to be rebooted away.
    {
        let ble = connected.lock().await;
        let status = ble.read_status().await.map_err(IpcError::from)?;
        check_preconditions(&status)?;
        switch_to_wifi_mode(&*ble).await?;
    }

    let total_bytes = image.bytes.len() as u64;
    on_state(OtaState::Pushing { done_bytes: 0, total_bytes, pct: 0 });
    {
        let mut on_push_progress = |done_bytes: u64, total_bytes: u64| {
            on_progress(done_bytes, total_bytes, "ota_push");
            on_state(OtaState::Pushing {
                done_bytes,
                total_bytes,
                pct: pct_of(done_bytes, total_bytes),
            });
        };
        wifi.push_ota(&image.bytes, &mut on_push_progress).await.map_err(ota_push_ipc_error)?;
    }

    // The image is accepted; the device restarts ~500 ms later (SPEC §6.1)
    // and this link dies with it. Drop it rather than leave a stale entry.
    {
        let entry = connections.lock().unwrap().remove(device_id);
        if let Some(ble) = entry {
            let _ = ble.lock().await.disconnect().await;
        }
    }
    on_state(OtaState::Rebooting);
    tokio::time::sleep(timings.boot_delay).await;

    let max_attempts = timings.max_attempts();
    let mut reconnected: Option<T> = None;
    for attempt in 1..=max_attempts {
        on_state(OtaState::Reconnecting { attempt, max_attempts });
        if let Ok(mut ble) = new_ble().await {
            if ble.connect(device_id).await.is_ok() {
                reconnected = Some(ble);
                break;
            }
            let _ = ble.disconnect().await;
        }
        if attempt < max_attempts {
            tokio::time::sleep(timings.reconnect_interval).await;
        }
    }
    let ble = reconnected.ok_or_else(|| {
        IpcError::new(
            IpcErrorKind::Ble,
            "the device did not come back after the update — power-cycle it and reconnect",
        )
    })?;
    let ble = Arc::new(tokio::sync::Mutex::new(ble));
    connections.lock().unwrap().insert(device_id.to_string(), Arc::clone(&ble));

    let status = ble.lock().await.read_status().await.map_err(IpcError::from)?;
    let device_version = status.firmware.clone();
    let mut outcome = OtaOutcome {
        state: OtaState::Idle,
        pushed_version: image.version.clone(),
        device_version: device_version.clone(),
        auto_confirmed: false,
        sha256_verified: image.sha256_verified,
    };

    if !status.ota_pending_verify {
        // Nothing is awaiting confirmation. Either the device committed on
        // its own, or the bootloader already reverted — the reported version
        // is what tells the two apart, when both ends are known.
        let rolled_back = match (image.version.as_deref(), device_version.as_deref()) {
            (Some(pushed), Some(running)) => pushed != running,
            _ => false,
        };
        outcome.state = if rolled_back { OtaState::RolledBack } else { OtaState::Confirmed };
        on_state(outcome.state.clone());
        return Ok(outcome);
    }

    let armed = auto_confirm_armed(image, source_is_catalog, device_version.as_deref());
    on_state(OtaState::PendingVerify { auto_confirm_armed: armed });
    if !armed {
        outcome.state = OtaState::PendingVerify { auto_confirm_armed: false };
        return Ok(outcome);
    }

    ble.lock().await.send_command(ControlCommand::OtaConfirm).await.map_err(IpcError::from)?;
    outcome.state = OtaState::Confirmed;
    outcome.auto_confirmed = true;
    on_state(OtaState::Confirmed);
    Ok(outcome)
}

/// The transport-agnostic core of [`confirm_firmware`]: sends SPEC §7.2's
/// `CMD_OTA_CONFIRM` over `device_id`'s managed link and reports the state
/// the device is in afterwards.
///
/// Requires a managed connection: confirming is a deliberate button press on
/// a device the user is already looking at, and silently opening a link to
/// send it would be a surprise.
pub async fn confirm_firmware_via<T: BleTransport>(
    connections: &ConnectionMap<T>,
    device_id: &str,
) -> Result<OtaState, IpcError> {
    let connected = connections
        .lock()
        .unwrap()
        .get(device_id)
        .cloned()
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, "connect to the device first"))?;
    let ble = connected.lock().await;
    ble.send_command(ControlCommand::OtaConfirm).await.map_err(IpcError::from)?;
    let status = ble.read_status().await.map_err(IpcError::from)?;
    // SPEC §7.2: the command is a no-op in any other state, so a device that
    // still reports PENDING_VERIFY did not take it.
    Ok(if status.ota_pending_verify {
        OtaState::PendingVerify { auto_confirm_armed: false }
    } else {
        OtaState::Confirmed
    })
}

/// Publishes `state`: stores it as the current OTA state and emits
/// `ota_state_changed` carrying it. Both, always — a UI that mounts
/// mid-flight reads the store, one that was already listening gets the
/// event, and the two never disagree.
fn publish_state<R: tauri::Runtime>(app: &tauri::AppHandle<R>, ota: &Ota, state: OtaState) {
    use tauri::Emitter;
    *ota.0.lock().unwrap() = state.clone();
    let _ = app.emit("ota_state_changed", state);
}

/// Downloads a catalog release, or reads a `.bin` off disk (C3 §3.8).
/// Emits `downloading` states for the catalog case.
async fn load_image<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    ota: &Ota,
    source: &FirmwareSource,
    firmware_repo: &str,
    channel: FirmwareChannel,
    progress: &tauri::ipc::Channel<Progress>,
) -> Result<LoadedImage, IpcError> {
    match source {
        FirmwareSource::File { path } => load_image_from_file(path),
        FirmwareSource::Catalog { version } => {
            let releases =
                firmware_catalog::fetch_releases(firmware_repo, channel).await.map_err(IpcError::from)?;
            let release = release_by_version(&releases, version)?.clone();
            let mut on_progress = |done_bytes: u64, total_bytes: Option<u64>| {
                let _ =
                    progress.send(Progress { done: done_bytes, total: total_bytes, phase: "download".to_string() });
                publish_state(
                    app,
                    ota,
                    OtaState::Downloading {
                        done_bytes,
                        total_bytes,
                        pct: total_bytes.map(|total| pct_of(done_bytes, total)),
                    },
                );
            };
            let downloaded = firmware_catalog::download_image(&release, &mut on_progress)
                .await
                .map_err(IpcError::from)?;
            Ok(LoadedImage {
                bytes: downloaded.bytes,
                version: Some(release.version),
                sha256_verified: downloaded.sha256_verified,
            })
        }
    }
}

/// Lists the firmware releases `firmware_repo` publishes on `channel`,
/// newest first (C3 §3.8, R198).
#[tauri::command]
pub async fn firmware_catalog(
    firmware_repo: String,
    channel: FirmwareChannel,
) -> Result<Vec<FirmwareRelease>, IpcError> {
    firmware_catalog::fetch_releases(&firmware_repo, channel).await.map_err(IpcError::from)
}

/// The current OTA state (C3 §3.8) — the same value the most recent
/// `ota_state_changed` event carried.
#[tauri::command]
pub fn ota_state(ota: tauri::State<'_, Ota>) -> OtaState {
    ota.0.lock().unwrap().clone()
}

/// Pushes firmware to `device_id` and drives the whole OTA sequence
/// (C3 §3.8, R198): load → push → reboot → reconnect → confirm.
///
/// `firmware_repo`/`channel` are read only for a `catalog` source; a `file`
/// source ignores them. `progress` streams the download and the push, with
/// `phase` telling them apart (`"download"`, `"ota_push"`); the coarser
/// state machine travels over `ota_state_changed`.
#[tauri::command]
pub async fn push_firmware<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    connections: tauri::State<'_, Connections>,
    ota: tauri::State<'_, Ota>,
    device_id: String,
    source: FirmwareSource,
    firmware_repo: String,
    channel: FirmwareChannel,
    progress: tauri::ipc::Channel<Progress>,
) -> Result<OtaOutcome, IpcError> {
    let source_is_catalog = matches!(source, FirmwareSource::Catalog { .. });
    publish_state(&app, &ota, OtaState::Idle);

    let result = async {
        let image = load_image(&app, &ota, &source, &firmware_repo, channel, &progress).await?;
        let wifi = ReqwestWifi::new(DEVICE_BASE_URL);
        let timings = OtaTimings::default();
        let mut on_state = |state: OtaState| publish_state(&app, &ota, state);
        let mut on_progress = |done: u64, total: u64, phase: &str| {
            let _ = progress.send(Progress { done, total: Some(total), phase: phase.to_string() });
        };
        push_firmware_via(
            &connections.0,
            &device_id,
            &image,
            source_is_catalog,
            &wifi,
            || async { BtleplugBle::new().await },
            &timings,
            &mut on_state,
            &mut on_progress,
        )
        .await
    }
    .await;

    if let Err(ref e) = result {
        publish_state(&app, &ota, OtaState::Failed { error: e.clone() });
    }
    result
}

/// Commits the running image after an OTA (C3 §3.8, SPEC §7.2
/// `CMD_OTA_CONFIRM`) — the "Confirm" button on the pending-verify card.
/// There is no matching roll-back command: rolling back is *not* confirming
/// and power-cycling the device, which the bootloader handles (R198).
#[tauri::command]
pub async fn confirm_firmware<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    connections: tauri::State<'_, Connections>,
    ota: tauri::State<'_, Ota>,
    device_id: String,
) -> Result<OtaState, IpcError> {
    let result = confirm_firmware_via(&connections.0, &device_id).await;
    match result {
        Ok(state) => {
            publish_state(&app, &ota, state.clone());
            Ok(state)
        }
        Err(e) => {
            publish_state(&app, &ota, OtaState::Failed { error: e.clone() });
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::device::tests::{StubBle, StubWifi};

    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    use idl_transport::ble_status::DeviceStatus;
    use idl_transport::wifi_transport::OtaPushError;
    use idl_transport::TransportErrorKind;

    /// Timings that run the whole machine inside a test: no boot wait, and
    /// three reconnect attempts a millisecond apart.
    fn fast_timings() -> OtaTimings {
        OtaTimings {
            boot_delay: Duration::from_millis(0),
            reconnect_interval: Duration::from_millis(1),
            reconnect_window: Duration::from_millis(3),
        }
    }

    /// A connection map holding one already-connected `StubBle` for
    /// `"dev-1"`, as `connect_device` would have left it.
    fn connected_map(ble: StubBle) -> ConnectionMap<StubBle> {
        let mut map = HashMap::new();
        map.insert("dev-1".to_string(), Arc::new(tokio::sync::Mutex::new(ble)));
        StdMutex::new(map)
    }

    /// A `StubBle` whose status reads report `wifi_on` and the given
    /// post-reboot picture.
    fn ready_ble(firmware: Option<&str>, ota_pending_verify: bool) -> StubBle {
        let ble = StubBle {
            connect_result: Ok(idl_transport::ConnectionInfo {
                device_id: "dev-1".to_string(),
                firmware_version: firmware.unwrap_or("").to_string(),
                connected: true,
            }),
            ..StubBle::default()
        };
        *ble.status_override.lock().unwrap() = Some(DeviceStatus {
            wifi_on: Some(true),
            logging: Some(false),
            firmware: firmware.map(str::to_string),
            ota_pending_verify,
            ..DeviceStatus::default()
        });
        ble
    }

    /// A catalog image whose sha256 verified, reporting version `version`.
    fn verified_image(version: &str) -> LoadedImage {
        LoadedImage {
            bytes: vec![7u8; 64],
            version: Some(version.to_string()),
            sha256_verified: true,
        }
    }

    #[test]
    fn pct_of_partial_complete_and_zero_length_never_exceeds_one_hundred() {
        // Arrange / Act / Assert
        assert_eq!(pct_of(0, 200), 0);
        assert_eq!(pct_of(50, 200), 25);
        assert_eq!(pct_of(200, 200), 100);
        assert_eq!(pct_of(999, 200), 100);
        assert_eq!(pct_of(0, 0), 100);
    }

    #[test]
    fn ota_state_serialises_with_a_snake_case_phase_tag_matching_the_c3_dto() {
        // Arrange
        let states = [
            OtaState::Idle,
            OtaState::Pushing { done_bytes: 5, total_bytes: 10, pct: 50 },
            OtaState::Reconnecting { attempt: 2, max_attempts: 20 },
            OtaState::PendingVerify { auto_confirm_armed: true },
            OtaState::RolledBack,
        ];

        // Act
        let json: Vec<String> = states.iter().map(|s| serde_json::to_string(s).unwrap()).collect();

        // Assert
        assert_eq!(json[0], r#"{"phase":"idle"}"#);
        assert_eq!(json[1], r#"{"phase":"pushing","done_bytes":5,"total_bytes":10,"pct":50}"#);
        assert_eq!(json[2], r#"{"phase":"reconnecting","attempt":2,"max_attempts":20}"#);
        assert_eq!(json[3], r#"{"phase":"pending_verify","auto_confirm_armed":true}"#);
        assert_eq!(json[4], r#"{"phase":"rolled_back"}"#);
    }

    #[test]
    fn firmware_source_deserialises_both_c3_variants_from_their_tagged_json() {
        // Arrange
        let file = r#"{"kind":"file","path":"C:/tmp/idl1.bin"}"#;
        let catalog = r#"{"kind":"catalog","version":"1.6.0"}"#;

        // Act
        let parsed_file: FirmwareSource = serde_json::from_str(file).unwrap();
        let parsed_catalog: FirmwareSource = serde_json::from_str(catalog).unwrap();

        // Assert
        assert_eq!(parsed_file, FirmwareSource::File { path: "C:/tmp/idl1.bin".to_string() });
        assert_eq!(parsed_catalog, FirmwareSource::Catalog { version: "1.6.0".to_string() });
    }

    #[test]
    fn ota_timings_default_allows_twenty_reconnect_attempts_over_r198s_sixty_seconds() {
        // Arrange
        let timings = OtaTimings::default();

        // Act
        let attempts = timings.max_attempts();

        // Assert
        assert_eq!(attempts, 20);
        assert_eq!(timings.reconnect_interval, Duration::from_secs(3));
    }

    #[test]
    fn auto_confirm_armed_only_for_a_verified_catalog_image_running_the_pushed_version() {
        // Arrange
        let verified = verified_image("1.6.0");
        let unverified = LoadedImage { sha256_verified: false, ..verified.clone() };
        let manual = LoadedImage { bytes: vec![1], version: None, sha256_verified: false };

        // Act / Assert
        assert!(auto_confirm_armed(&verified, true, Some("1.6.0")));
        assert!(!auto_confirm_armed(&verified, true, Some("1.5.0")), "a rolled-back device must not be confirmed");
        assert!(!auto_confirm_armed(&verified, false, Some("1.6.0")), "a file source never arms");
        assert!(!auto_confirm_armed(&unverified, true, Some("1.6.0")), "an unverified download never arms");
        assert!(!auto_confirm_armed(&manual, false, Some("1.6.0")));
    }

    #[test]
    fn check_preconditions_recording_device_is_device_rejected_idle_device_is_ok() {
        // Arrange
        let recording = DeviceStatus { logging: Some(true), ..DeviceStatus::default() };
        let idle = DeviceStatus { logging: Some(false), ..DeviceStatus::default() };
        let unreported = DeviceStatus::default();

        // Act
        let refused = check_preconditions(&recording);
        let allowed = check_preconditions(&idle);
        let silent = check_preconditions(&unreported);

        // Assert
        assert_eq!(refused.unwrap_err().kind, IpcErrorKind::DeviceRejected);
        assert!(allowed.is_ok());
        assert!(silent.is_ok(), "a device that reports no logging line is not evidence of recording");
    }

    #[test]
    fn load_image_from_file_missing_file_is_not_found_empty_file_is_invalid_argument() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.bin");
        std::fs::write(&empty, b"").unwrap();
        let good = dir.path().join("fw.bin");
        std::fs::write(&good, b"0123").unwrap();

        // Act
        let missing = load_image_from_file(&dir.path().join("nope.bin").to_string_lossy());
        let empty_result = load_image_from_file(&empty.to_string_lossy());
        let loaded = load_image_from_file(&good.to_string_lossy()).unwrap();

        // Assert
        assert_eq!(missing.unwrap_err().kind, IpcErrorKind::NotFound);
        assert_eq!(empty_result.unwrap_err().kind, IpcErrorKind::InvalidArgument);
        assert_eq!(loaded.bytes, b"0123");
        assert_eq!(loaded.version, None);
        assert!(!loaded.sha256_verified);
    }

    #[test]
    fn release_by_version_unknown_version_is_not_found_rather_than_the_newest() {
        // Arrange
        let releases = vec![FirmwareRelease {
            version: "1.5.0".to_string(),
            tag: "v1.5.0".to_string(),
            name: "1.5.0".to_string(),
            notes: String::new(),
            prerelease: false,
            published_at: String::new(),
            image_url: "https://example.invalid/1.5.0.bin".to_string(),
            image_size_bytes: 10,
            sha256_url: None,
        }];

        // Act
        let found = release_by_version(&releases, "1.5.0");
        let missing = release_by_version(&releases, "9.9.9");

        // Assert
        assert_eq!(found.unwrap().tag, "v1.5.0");
        assert_eq!(missing.unwrap_err().kind, IpcErrorKind::NotFound);
    }

    #[test]
    fn ota_push_ipc_error_carries_the_status_and_the_devices_body_text_as_detail() {
        // Arrange
        let push_error =
            OtaPushError::new(OtaPushErrorKind::Rejected, Some(400), "short upload", "the device rejected it");

        // Act
        let ipc = ota_push_ipc_error(push_error);

        // Assert
        assert_eq!(ipc.kind, IpcErrorKind::Wifi);
        let detail = ipc.detail.unwrap();
        assert_eq!(detail["ota_error"], "rejected");
        assert_eq!(detail["status_code"], 400);
        assert_eq!(detail["device_body"], "short upload");
    }

    #[tokio::test]
    async fn push_firmware_via_verified_catalog_image_walks_push_reboot_reconnect_and_auto_confirms() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.6.0"), true));
        let wifi = StubWifi { push_ota_result: Ok(()), ..StubWifi::default() };
        let image = verified_image("1.6.0");
        let mut states: Vec<OtaState> = Vec::new();
        let mut on_state = |state: OtaState| states.push(state);
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let outcome = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), true)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap();

        // Assert
        assert_eq!(outcome.state, OtaState::Confirmed);
        assert!(outcome.auto_confirmed);
        assert_eq!(outcome.pushed_version.as_deref(), Some("1.6.0"));
        assert!(matches!(states[0], OtaState::Pushing { .. }));
        assert!(states.contains(&OtaState::Rebooting));
        assert!(states.iter().any(|s| matches!(s, OtaState::Reconnecting { attempt: 1, .. })));
        assert!(states.contains(&OtaState::PendingVerify { auto_confirm_armed: true }));
        assert_eq!(states.last(), Some(&OtaState::Confirmed));
        assert!(connections.lock().unwrap().contains_key("dev-1"), "the reconnected link is managed again");
    }

    #[tokio::test]
    async fn push_firmware_via_manual_file_image_stops_at_pending_verify_without_confirming() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.6.0"), true));
        let wifi = StubWifi { push_ota_result: Ok(()), ..StubWifi::default() };
        let image = LoadedImage { bytes: vec![1, 2, 3], version: None, sha256_verified: false };
        let mut states: Vec<OtaState> = Vec::new();
        let mut on_state = |state: OtaState| states.push(state);
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let outcome = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            false,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), true)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap();

        // Assert
        assert_eq!(outcome.state, OtaState::PendingVerify { auto_confirm_armed: false });
        assert!(!outcome.auto_confirmed);
        assert_eq!(states.last(), Some(&OtaState::PendingVerify { auto_confirm_armed: false }));
    }

    #[tokio::test]
    async fn push_firmware_via_device_back_on_the_old_version_without_pending_verify_is_rolled_back() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.5.0"), false));
        let wifi = StubWifi { push_ota_result: Ok(()), ..StubWifi::default() };
        let image = verified_image("1.6.0");
        let mut states: Vec<OtaState> = Vec::new();
        let mut on_state = |state: OtaState| states.push(state);
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let outcome = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.5.0"), false)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap();

        // Assert
        assert_eq!(outcome.state, OtaState::RolledBack);
        assert_eq!(outcome.device_version.as_deref(), Some("1.5.0"));
        assert!(!outcome.auto_confirmed);
    }

    #[tokio::test]
    async fn push_firmware_via_reports_monotonic_push_progress_and_a_final_hundred_percent() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.6.0"), false));
        let wifi = StubWifi { push_ota_result: Ok(()), ..StubWifi::default() };
        let image = verified_image("1.6.0");
        let mut states: Vec<OtaState> = Vec::new();
        let mut on_state = |state: OtaState| states.push(state);
        let mut phases: Vec<String> = Vec::new();
        let mut on_progress = |_: u64, _: u64, phase: &str| phases.push(phase.to_string());

        // Act
        push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), false)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap();

        // Assert
        let pushing: Vec<u8> = states
            .iter()
            .filter_map(|s| match s {
                OtaState::Pushing { pct, .. } => Some(*pct),
                _ => None,
            })
            .collect();
        assert!(pushing.windows(2).all(|w| w[0] <= w[1]), "push percentages must be monotonic");
        assert_eq!(pushing.last(), Some(&100));
        assert!(phases.iter().all(|phase| phase == "ota_push"));
    }

    #[tokio::test]
    async fn push_firmware_via_recording_device_refuses_before_any_wifi_switch_or_push() {
        // Arrange
        let ble = ready_ble(Some("1.5.0"), false);
        *ble.status_override.lock().unwrap() =
            Some(DeviceStatus { logging: Some(true), ..DeviceStatus::default() });
        let connections = connected_map(ble);
        let wifi = StubWifi::default(); // push_ota unconfigured: calling it fails the test
        let image = verified_image("1.6.0");
        let mut on_state = |_: OtaState| {};
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let err = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), true)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::DeviceRejected);
        assert!(connections.lock().unwrap().contains_key("dev-1"), "a refused push leaves the link alone");
    }

    #[tokio::test]
    async fn push_firmware_via_no_managed_connection_is_not_found_before_anything_is_read() {
        // Arrange
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());
        let wifi = StubWifi::default();
        let image = verified_image("1.6.0");
        let mut on_state = |_: OtaState| {};
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let err = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), true)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
    }

    #[tokio::test]
    async fn push_firmware_via_rejected_image_surfaces_the_device_body_and_leaves_the_link_dropped() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.5.0"), false));
        let wifi = StubWifi {
            push_ota_result: Err(OtaPushError::new(
                OtaPushErrorKind::Rejected,
                Some(400),
                "image validation failed",
                "the device rejected the firmware image",
            )),
            ..StubWifi::default()
        };
        let image = verified_image("1.6.0");
        let mut on_state = |_: OtaState| {};
        let mut on_progress = |_: u64, _: u64, _: &str| {};

        // Act
        let err = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async { Ok(ready_ble(Some("1.6.0"), true)) },
            &fast_timings(),
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Wifi);
        assert_eq!(err.detail.unwrap()["device_body"], "image validation failed");
    }

    #[tokio::test]
    async fn push_firmware_via_device_that_never_comes_back_fails_after_the_whole_reconnect_window() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.5.0"), false));
        let wifi = StubWifi { push_ota_result: Ok(()), ..StubWifi::default() };
        let image = verified_image("1.6.0");
        let mut states: Vec<OtaState> = Vec::new();
        let mut on_state = |state: OtaState| states.push(state);
        let mut on_progress = |_: u64, _: u64, _: &str| {};
        let timings = fast_timings();

        // Act
        let err = push_firmware_via(
            &connections,
            "dev-1",
            &image,
            true,
            &wifi,
            || async {
                Err::<StubBle, _>(TransportError::new(TransportErrorKind::Ble, "no adapter"))
            },
            &timings,
            &mut on_state,
            &mut on_progress,
        )
        .await
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Ble);
        let attempts = states
            .iter()
            .filter(|s| matches!(s, OtaState::Reconnecting { .. }))
            .count();
        assert_eq!(attempts as u32, timings.max_attempts());
        assert!(!connections.lock().unwrap().contains_key("dev-1"), "the dead link is not left in the map");
    }

    #[tokio::test]
    async fn confirm_firmware_via_device_leaves_pending_verify_reports_confirmed() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.6.0"), false));

        // Act
        let state = confirm_firmware_via(&connections, "dev-1").await.unwrap();

        // Assert
        assert_eq!(state, OtaState::Confirmed);
    }

    #[tokio::test]
    async fn confirm_firmware_via_device_still_pending_verify_reports_that_rather_than_success() {
        // Arrange
        let connections = connected_map(ready_ble(Some("1.6.0"), true));

        // Act
        let state = confirm_firmware_via(&connections, "dev-1").await.unwrap();

        // Assert
        assert_eq!(state, OtaState::PendingVerify { auto_confirm_armed: false });
    }

    #[tokio::test]
    async fn confirm_firmware_via_unconnected_device_is_not_found() {
        // Arrange
        let connections: ConnectionMap<StubBle> = StdMutex::new(HashMap::new());

        // Act
        let err = confirm_firmware_via(&connections, "dev-1").await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
    }
}
