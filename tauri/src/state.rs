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
