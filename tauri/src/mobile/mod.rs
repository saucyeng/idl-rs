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

#[derive(serde::Serialize)]
struct CrashSetupArgs<'a> {
    dir: &'a str,
}

#[derive(serde::Serialize)]
struct ShareArgs<'a> {
    title: &'a str,
    text: &'a str,
}

/// Tells the plugin where crash reports go (its uncaught-exception handler
/// writes there) and has it turn the last run's exit reasons into reports
/// (crash reports spec, 2026-10-21). Idempotent; a failure only costs the
/// Android-side reports, so it is not returned.
pub async fn crash_setup(dir: &std::path::Path) {
    let dir = dir.to_string_lossy();
    // `call` wants a transport kind; this is neither Bluetooth nor WiFi, and
    // the result is dropped, so the kind never reaches anyone.
    let _ = call::<serde_json::Value>("crashSetup", CrashSetupArgs { dir: &dir }, TransportErrorKind::Ble).await;
}

/// Opens Android's share sheet with `text`. A plugin failure is `io`.
pub(crate) async fn share_text(title: &str, text: &str) -> Result<(), crate::error::IpcError> {
    call::<serde_json::Value>("shareText", ShareArgs { title, text }, TransportErrorKind::Ble)
        .await
        .map(|_| ())
        .map_err(|e| crate::error::IpcError::new(crate::error::IpcErrorKind::Io, e.message))
}

/// Runs `command` on the Kotlin plugin, failing as `kind` on any plugin or
/// JNI error. The plugin's reject message becomes the error's message.
///
/// The plugin call runs on its own task, so a caller that gives up (a
/// `tokio::time::timeout` around a connect) drops only the join handle and
/// the call still takes Kotlin's late answer. Dropping the pending call
/// itself made tauri's response handler unwrap a closed channel
/// (tauri 2.11.5 `plugin/mobile.rs:307`) inside a JNI callback, which
/// aborts the app: the crash ~10 s into a search (2026-10-21).
pub(crate) async fn call<T: DeserializeOwned>(
    command: &str,
    payload: impl Serialize,
    kind: TransportErrorKind,
) -> Result<T, TransportError> {
    let plugin = PLUGIN
        .get()
        .ok_or_else(|| TransportError::new(kind, "the Android device plugin is not registered"))?;
    let payload = serde_json::to_value(payload)
        .map_err(|e| TransportError::new(kind, format!("{command}: payload did not serialise: {e}")))?;
    let name = command.to_string();
    let task = tauri::async_runtime::spawn(async move {
        plugin.run_mobile_plugin_async::<serde_json::Value>(&name, payload).await.map_err(|e| e.to_string())
    });
    let value = task
        .await
        .map_err(|e| TransportError::new(kind, format!("{command}: the plugin call's task failed: {e}")))?
        .map_err(|e| TransportError::new(kind, format!("{command}: {e}")))?;
    serde_json::from_value(value).map_err(|e| TransportError::new(kind, format!("{command}: unexpected answer: {e}")))
}
