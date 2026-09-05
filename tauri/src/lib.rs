//! `idl-rs-tauri` — the only crate the idl1 frontend sees.
//!
//! Every function here is a `#[tauri::command]` wrapper over `idl-rs` (numbers)
//! or `idl-transport` (bytes on the wire). Heavy arrays return
//! `tauri::ipc::Response` raw bytes, never JSON (design §4). Register with:
//!
//! ```ignore
//! tauri::Builder::default().invoke_handler(idl_rs_tauri::handler())
//! ```

pub mod commands;
pub mod error;
pub mod paths;
pub mod session_source;
pub mod state;
pub mod watcher;
pub use error::{IpcError, IpcErrorKind};

/// The invoke handler covering every command in this crate. The app crate
/// passes it to `tauri::Builder::invoke_handler` so new commands never touch
/// `app/src-tauri`.
pub fn handler<R: tauri::Runtime>() -> impl Fn(tauri::ipc::Invoke<R>) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        commands::engine_version,
        commands::device::ble_scan,
        commands::device::ble_connect,
        commands::device::connect_device,
        commands::device::disconnect_device,
        commands::device::device_status,
        commands::device::list_device_files,
        commands::device::download_file,
        commands::device::push_config,
        commands::catalog::list_sessions,
        commands::catalog::get_session,
        commands::catalog::list_laps,
        commands::catalog::rebuild_catalog,
        commands::catalog::list_workbooks,
        commands::catalog::list_tracks,
        commands::catalog::get_track,
        commands::catalog::save_session_metadata,
        commands::catalog::delete_session,
        commands::workbook::open_workbook,
        commands::workbook::read_workbook,
        commands::workbook::eval_workbook,
        commands::workbook::save_workbook,
        commands::workbook::watch_workbook,
        commands::cursor::cursor_readout,
        commands::rasters::fetch_raster,
        commands::rasters::fetch_raster_meta,
        commands::tiles::fetch_tile,
        commands::import::list_importers,
        commands::import::import_file,
        commands::app::get_settings,
        commands::app::set_settings,
        commands::app::get_data_dir,
        commands::app::set_data_dir,
        commands::app::list_profiles,
        commands::app::save_profile,
        commands::app::delete_profile,
    ]
}
