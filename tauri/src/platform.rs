//! The one place the concrete device transports are named (SPEC §14b.1):
//! `PlatformBle` is `BtleplugBle` on desktop and the Kotlin-plugin-backed
//! `AndroidBle` on Android, and `open_device_wifi` hands back an
//! identity-checked HTTP client for a logger already in WiFi mode.
//!
//! Step 1 of SPEC §14b.6: the per-command flow is kept, so the logger's
//! advertised name (the SSID, and what `/ping` must report) is remembered
//! here from scans and connects. Step 2's `DeviceSessions` replaces the
//! registry with per-device state.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use idl_transport::wifi_transport::{verify_device_identity, ReqwestWifi, WifiTransport};

use crate::error::{IpcError, IpcErrorKind};

/// The BLE transport this build uses.
#[cfg(not(target_os = "android"))]
pub type PlatformBle = idl_transport::ble_transport::BtleplugBle;

/// The BLE transport this build uses.
#[cfg(target_os = "android")]
pub type PlatformBle = crate::mobile::ble::AndroidBle;

/// WiFi control-protocol major version this app speaks (SPEC §6.1 `proto`).
const SUPPORTED_PROTO: u32 = 1;

/// SPEC §14b.3 `verifying`: the route is not usable the instant the network
/// comes up, so `/ping` is retried this often ...
const PING_RETRY_INTERVAL: Duration = Duration::from_millis(500);
/// ... for at most this long, ms.
const PING_BUDGET: Duration = Duration::from_secs(5);

/// `device_id` → advertised name, from every scan result and connect.
static NAMES: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

/// Remembers `name` for `device_id`; empty names are ignored.
pub(crate) fn remember_name(device_id: &str, name: &str) {
    if name.is_empty() {
        return;
    }
    let mut names = NAMES.lock().unwrap();
    names.get_or_insert_with(HashMap::new).insert(device_id.to_string(), name.to_string());
}

/// The advertised name last seen for `device_id`.
pub(crate) fn name_for(device_id: &str) -> Result<String, IpcError> {
    NAMES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|names| names.get(device_id).cloned())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorKind::Ble,
                format!("the name of device {device_id} is not known yet: scan for it or connect to it first"),
            )
        })
}

/// Opens an HTTP client to the logger named `expected_name`, which must
/// already be in WiFi mode, and checks with `/ping` that it is that logger
/// (SPEC §6.1: every IDL0 AP shares `192.168.4.1`) speaking protocol 1.
///
/// Android requests the AP and routes through the plugin's loopback proxy
/// (SPEC §6.2); desktop talks to `192.168.4.1` on whatever network the user
/// joined, so a `/ping` that never answers there means "join the AP".
pub(crate) async fn open_device_wifi(expected_name: &str) -> Result<ReqwestWifi, IpcError> {
    #[cfg(target_os = "android")]
    let base_url = crate::mobile::wifi::link(expected_name).await?;
    #[cfg(not(target_os = "android"))]
    let base_url = idl_transport::wifi_transport::DEVICE_BASE_URL.to_string();

    let wifi = ReqwestWifi::new(base_url);
    verify_link(&wifi, expected_name).await?;
    Ok(wifi)
}

/// SPEC §14b.3 `verifying`: `/ping` until it answers (bounded), then the
/// identity and protocol checks.
pub(crate) async fn verify_link(wifi: &impl WifiTransport, expected_name: &str) -> Result<(), IpcError> {
    let deadline = tokio::time::Instant::now() + PING_BUDGET;
    let ping = loop {
        match wifi.ping().await {
            Ok(ping) => break ping,
            Err(_) if tokio::time::Instant::now() + PING_RETRY_INTERVAL < deadline => {
                tokio::time::sleep(PING_RETRY_INTERVAL).await;
            }
            Err(e) => {
                return Err(IpcError::with_detail(
                    IpcErrorKind::Wifi,
                    format!("{expected_name} did not answer over WiFi: {}", e.message),
                    serde_json::json!({ "hint": "join_ap", "ssid": expected_name }),
                ));
            }
        }
    };
    verify_device_identity(&ping, expected_name).map_err(IpcError::from)?;
    if ping.proto_version != SUPPORTED_PROTO {
        return Err(IpcError::with_detail(
            IpcErrorKind::Wifi,
            format!(
                "{expected_name} speaks WiFi protocol {}, this app speaks {SUPPORTED_PROTO}: update the firmware",
                ping.proto_version
            ),
            serde_json::json!({ "hint": "firmware_update", "proto": ping.proto_version }),
        ));
    }
    Ok(())
}

/// Lets go of the logger's WiFi network once it has left WiFi mode (the
/// held Android request, SPEC §6.2). Nothing to release on desktop.
pub(crate) async fn release_device_wifi() {
    #[cfg(target_os = "android")]
    crate::mobile::wifi::release().await;
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use idl_transport::wifi_transport::{OtaPushError, PingResponse};
    use idl_transport::{DeviceFile, TransportError, TransportErrorKind};

    use super::*;

    /// A `WifiTransport` whose `/ping` answers are scripted, in order; every
    /// other operation is unused by `verify_link`.
    struct PingScript(Mutex<VecDeque<Result<PingResponse, TransportError>>>);

    impl PingScript {
        fn new(answers: Vec<Result<PingResponse, TransportError>>) -> Self {
            Self(Mutex::new(answers.into()))
        }
    }

    fn unused() -> TransportError {
        TransportError::new(TransportErrorKind::Wifi, "not used by verify_link")
    }

    impl WifiTransport for PingScript {
        async fn ping(&self) -> Result<PingResponse, TransportError> {
            self.0.lock().unwrap().pop_front().unwrap_or_else(|| Err(unreachable_ping()))
        }
        async fn handoff(&self) -> Result<(), TransportError> {
            Err(unused())
        }
        async fn wifi_off(&self) -> Result<(), TransportError> {
            Err(unused())
        }
        async fn list_files(&self) -> Result<Vec<DeviceFile>, TransportError> {
            Err(unused())
        }
        async fn download(
            &self,
            _file_index: u32,
            _resume_from_bytes: u64,
            _sink: &mut (dyn tokio::io::AsyncWrite + Unpin + Send),
            _on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
        ) -> Result<u64, TransportError> {
            Err(unused())
        }
        async fn delete(&self, _file_index: u32) -> Result<(), TransportError> {
            Err(unused())
        }
        async fn push_config(&self, _config_json: &[u8]) -> Result<(), TransportError> {
            Err(unused())
        }
        async fn push_ota(
            &self,
            _firmware_image: &[u8],
            _on_progress: &mut (dyn FnMut(u64, u64) + Send),
        ) -> Result<(), OtaPushError> {
            Err(OtaPushError::transport("not used by verify_link"))
        }
    }

    fn unreachable_ping() -> TransportError {
        TransportError::new(TransportErrorKind::Wifi, "GET /ping failed: connection refused")
    }

    fn ping(device: &str, proto_version: u32) -> PingResponse {
        PingResponse {
            device: device.to_string(),
            fw: "1.4.0".to_string(),
            proto_version,
            battery_pct: 87,
            sd: "OK".to_string(),
            mode: "wifi".to_string(),
            ble: "on".to_string(),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn verify_link_route_warms_up_then_answers_ok() {
        // Arrange
        let wifi = PingScript::new(vec![Err(unreachable_ping()), Err(unreachable_ping()), Ok(ping("IDL0-A3F2", 1))]);

        // Act
        let result = verify_link(&wifi, "IDL0-A3F2").await;

        // Assert
        assert!(result.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn verify_link_wrong_logger_answers_wifi_error_naming_both() {
        // Arrange
        let wifi = PingScript::new(vec![Ok(ping("IDL0-B000", 1))]);

        // Act
        let err = verify_link(&wifi, "IDL0-A3F2").await.unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::Wifi);
        assert!(err.message.contains("IDL0-B000") && err.message.contains("IDL0-A3F2"));
    }

    #[tokio::test(start_paused = true)]
    async fn verify_link_other_protocol_major_asks_for_a_firmware_update() {
        // Arrange
        let wifi = PingScript::new(vec![Ok(ping("IDL0-A3F2", 2))]);

        // Act
        let err = verify_link(&wifi, "IDL0-A3F2").await.unwrap_err();

        // Assert
        assert_eq!(err.detail.unwrap()["hint"], "firmware_update");
    }

    #[tokio::test(start_paused = true)]
    async fn verify_link_never_answers_within_budget_hints_join_ap_with_ssid() {
        // Arrange
        let wifi = PingScript::new(vec![]);

        // Act
        let started = tokio::time::Instant::now();
        let err = verify_link(&wifi, "IDL0-A3F2").await.unwrap_err();

        // Assert
        let detail = err.detail.unwrap();
        assert_eq!(detail["hint"], "join_ap");
        assert_eq!(detail["ssid"], "IDL0-A3F2");
        assert!(started.elapsed() <= PING_BUDGET);
    }

    #[test]
    fn name_for_after_remember_name_returns_it_empty_names_ignored() {
        // Arrange
        remember_name("AA:BB:CC:00:00:01", "IDL0-A3F2");
        remember_name("AA:BB:CC:00:00:01", "");

        // Act
        let name = name_for("AA:BB:CC:00:00:01");

        // Assert
        assert_eq!(name.unwrap(), "IDL0-A3F2");
    }
}
