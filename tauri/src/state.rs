//! Tauri-managed shared state (`app.manage(...)`), read by commands via
//! `tauri::State<T>`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The resolved `<data>` root (C4 §1), computed once at startup by
/// `paths::resolve_data_dir` and managed for the app's lifetime.
pub struct DataDir(pub PathBuf);

/// The C4 §4 self-write-suppression set (`crate::watcher::ExpectedHashSet`),
/// shared between `save_workbook` (registers the app's own writes before the
/// atomic rename) and `watch_workbook` (suppresses them so an external-edit
/// callback never fires for the app's own save).
pub struct Hashes(pub Arc<crate::watcher::ExpectedHashSet>);

/// Live `watch_workbook` subscriptions, keyed by workbook id. Dropping the
/// entry stops the watcher (`WorkbookWatcher`'s `Drop` tears down its
/// `notify` handle). Re-subscribing to the same id replaces the previous
/// entry, so a frontend remount cannot leak watchers — Tauri v2 gives no
/// channel-close signal this task can observe, so there is no unsubscribe
/// command in wave 1 (see the CHANGELOG entry).
pub struct Watchers(pub Mutex<HashMap<String, crate::watcher::WorkbookWatcher>>);

/// Live managed BLE connections, keyed by `device_id` (C3 §3.8's
/// `connect_device`/`disconnect_device`/`device_status`). The outer
/// `std::sync::Mutex` guards only the map's shape (insert/remove/lookup —
/// short, synchronous critical sections); each connection's own `BtleplugBle`
/// sits behind an `Arc<tokio::sync::Mutex<_>>` so a command can clone the
/// `Arc` out, drop the outer lock, then hold the inner async lock across its
/// own `.await`s without blocking every other command touching the map.
/// `connect_device` inserts an entry and leaves the link open; `device_status`/
/// `device_control`/`pull_config` use it when present and otherwise
/// connect-act-disconnect (C3 §3.8).
pub struct Connections(
    pub Mutex<HashMap<String, Arc<tokio::sync::Mutex<idl_transport::ble_transport::BtleplugBle>>>>,
);
