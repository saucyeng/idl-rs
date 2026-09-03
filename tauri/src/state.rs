//! Tauri-managed shared state (`app.manage(...)`), read by commands via
//! `tauri::State<T>`.

use std::path::PathBuf;

/// The resolved `<data>` root (C4 §1), computed once at startup by
/// `paths::resolve_data_dir` and managed for the app's lifetime.
pub struct DataDir(pub PathBuf);
