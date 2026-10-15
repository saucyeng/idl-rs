//! The app's own update check (C3 §3.10): glue over
//! `idl_transport::app_release`, which owns the feeds and the comparison.

use idl_transport::app_release::{self, AppUpdate, CurrentBuild};

use crate::error::IpcError;

/// Checks the running build's channel for a newer release (C3 §3.10).
/// `None` means up to date. The frontend passes what it was built as.
#[tauri::command]
pub async fn check_app_update(current: CurrentBuild) -> Result<Option<AppUpdate>, IpcError> {
    app_release::check_app_update(&current).await.map_err(IpcError::from)
}
