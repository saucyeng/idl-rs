//! SPEC §6.2 on Android: the logger's AP requested through the Kotlin
//! plugin, reached through its loopback proxy.
//!
//! Step 1 of SPEC §14b.6: one request is held for as long as the logger
//! stays in WiFi mode (idl0 learned that request/unregister cycles per
//! operation race on Android 10+: the second request never got a
//! callback), reused by every command, and released on `wifi_off`. Step 2's
//! link reconciler takes this over.

use std::sync::Mutex as StdMutex;
use std::time::Duration;

use idl_transport::wifi_transport::{DEVICE_AP_PASSWORD, DEVICE_BASE_URL};
use idl_transport::TransportErrorKind;
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio::sync::watch;

use super::call;
use crate::error::{IpcError, IpcErrorKind};

/// SPEC §14b.3 `requesting` budget: how long to wait for `available`
/// (the first request shows Android's approval dialog).
const REQUEST_BUDGET: Duration = Duration::from_secs(45);

/// What the plugin last said about the request.
#[derive(Clone, Debug, PartialEq)]
enum NetState {
    Pending,
    /// Up; `Some(port)` is the loopback proxy, `None` means direct
    /// `192.168.4.1` (API < 29, the user joined in Settings).
    Available(Option<u16>),
    Lost,
    Unavailable,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum NetEvent {
    Available { port: Option<u16> },
    Lost,
    Unavailable,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct RequestArgs<'a> {
    ssid: &'a str,
    password: &'a str,
    on_event: Channel<serde_json::Value>,
}

/// The one live request: its SSID and the plugin's latest state for it.
struct Held {
    ssid: String,
    state: watch::Receiver<NetState>,
}

static HELD: StdMutex<Option<Held>> = StdMutex::new(None);

fn wifi_error(message: impl Into<String>) -> IpcError {
    IpcError::new(IpcErrorKind::Wifi, message)
}

/// The base URL for the logger whose AP is `ssid`, requesting the network
/// first unless a live request for that SSID is already up.
pub(crate) async fn link(ssid: &str) -> Result<String, IpcError> {
    // A held request is reused while it is up or still coming up; a lost or
    // refused one is replaced. (No `watch` borrow may live across an
    // `.await`: it would make the command's future non-`Send`.)
    let reusable = HELD
        .lock()
        .unwrap()
        .as_ref()
        .filter(|held| held.ssid == ssid)
        .map(|held| held.state.clone())
        .filter(|state| matches!(*state.borrow(), NetState::Available(_) | NetState::Pending));
    let mut state = match reusable {
        Some(state) => state,
        None => request(ssid).await?,
    };

    let outcome = tokio::time::timeout(REQUEST_BUDGET, async {
        state.wait_for(|s| !matches!(s, NetState::Pending)).await.map(|s| s.clone())
    })
    .await;
    let current = match outcome {
        Ok(Ok(s)) => s,
        Ok(Err(_)) => NetState::Unavailable,
        Err(_) => {
            release().await;
            return Err(wifi_error(format!("{ssid} did not come up within {}s", REQUEST_BUDGET.as_secs())));
        }
    };
    match current {
        NetState::Available(Some(port)) => Ok(format!("http://127.0.0.1:{port}")),
        NetState::Available(None) => Ok(DEVICE_BASE_URL.to_string()),
        NetState::Lost => {
            release().await;
            Err(wifi_error(format!("lost the connection to {ssid}")))
        }
        NetState::Unavailable | NetState::Pending => {
            release().await;
            Err(wifi_error(format!("could not join {ssid} (declined, or out of range)")))
        }
    }
}

/// Supersedes any live request with a fresh one for `ssid`.
async fn request(ssid: &str) -> Result<watch::Receiver<NetState>, IpcError> {
    let (tx, rx) = watch::channel(NetState::Pending);
    let on_event = Channel::new(move |body: InvokeResponseBody| {
        if let Ok(event) = body.deserialize::<NetEvent>() {
            let _ = tx.send(match event {
                NetEvent::Available { port } => NetState::Available(port),
                NetEvent::Lost => NetState::Lost,
                NetEvent::Unavailable => NetState::Unavailable,
            });
        }
        Ok(())
    });
    call::<serde_json::Value>(
        "wifiRequest",
        RequestArgs { ssid, password: DEVICE_AP_PASSWORD, on_event },
        TransportErrorKind::Wifi,
    )
    .await
    .map_err(IpcError::from)?;
    *HELD.lock().unwrap() = Some(Held { ssid: ssid.to_string(), state: rx.clone() });
    Ok(rx)
}

/// Drops the live request, if any (the logger left WiFi mode). Best effort.
pub(crate) async fn release() {
    let had = HELD.lock().unwrap().take().is_some();
    if had {
        if let Err(e) = call::<serde_json::Value>("wifiRelease", (), TransportErrorKind::Wifi).await {
            eprintln!("wifiRelease: {e}");
        }
    }
}
