//! Android glue over the Kotlin `DevicePlugin` (SPEC §14b.1–2): the plugin
//! handle, one typed call helper, and the plugin-backed transports. The
//! Kotlin side is a sensor/actuator only; every policy decision is here or
//! in `idl-transport`.

pub mod ble;
pub mod wifi;

use std::sync::OnceLock;

use idl_transport::{TransportError, TransportErrorKind};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tauri::plugin::PluginHandle;

/// Set once, by the app crate's plugin setup (`app/src-tauri/src/mobile.rs`).
static PLUGIN: OnceLock<PluginHandle<tauri::Wry>> = OnceLock::new();

/// Hands this crate the registered `DevicePlugin`. Called once at startup;
/// a second call is ignored.
pub fn install(handle: PluginHandle<tauri::Wry>) {
    let _ = PLUGIN.set(handle);
}

/// Runs `command` on the Kotlin plugin, failing as `kind` on any plugin or
/// JNI error. The plugin's reject message becomes the error's message.
pub(crate) async fn call<T: DeserializeOwned>(
    command: &str,
    payload: impl Serialize,
    kind: TransportErrorKind,
) -> Result<T, TransportError> {
    let plugin = PLUGIN
        .get()
        .ok_or_else(|| TransportError::new(kind, "the Android device plugin is not registered"))?;
    plugin
        .run_mobile_plugin_async(command, payload)
        .await
        .map_err(|e| TransportError::new(kind, format!("{command}: {e}")))
}
