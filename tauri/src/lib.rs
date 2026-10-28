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
/// Local crash reports (C3 §3.13, 2026-10-21).
pub mod crash;
/// Build-time guard over the engine's delete sites (ruling R196). Test-only
/// — it contains no runtime code, only the scan and the audited allowlist.
#[cfg(test)]
mod delete_guard;
pub mod error;
/// Shared, single-flight workbook evaluations keyed by their inputs
/// (evaluation-memory spec §4.2.A).
pub mod eval_registry;
/// `<data>/inbox` watching (C4 §2, ruling R191). Desktop only — the inbox
/// does not exist on mobile, where `inbox_status` returns
/// `unsupported_platform`.
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod inbox;
pub mod link;
pub mod memory;
#[cfg(target_os = "android")]
pub mod mobile;
pub mod paths;
pub mod platform;
pub mod session_cache;
pub mod session_source;
pub mod state;
/// The `idl1-video` URI scheme's request logic (C3 §3.12): playback bytes
/// for linked videos, registered on the builder in `app/src-tauri`.
pub mod video;
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
        commands::device::device_control,
        commands::device::pull_config,
        commands::device::list_recent_device_files,
        commands::device::preview_channel_registry,
        commands::app_update::check_app_update,
        commands::firmware::firmware_catalog,
        commands::firmware::push_firmware,
        commands::firmware::confirm_firmware,
        commands::firmware::ota_state,
        commands::device::list_device_files,
        commands::device::download_file,
        commands::device::download_files,
        commands::device::cancel_wifi_transfer,
        commands::device::power_off_device,
        link::link_watch,
        link::link_pair,
        link::link_forget,
        link::link_kick,
        link::link_record,
        commands::device::bluetooth_state,
        commands::device::request_bluetooth_on,
        commands::device::push_config,
        commands::catalog::list_sessions,
        commands::catalog::get_session,
        commands::catalog::list_laps,
        commands::catalog::rebuild_catalog,
        commands::catalog::list_workbooks,
        commands::catalog::list_tracks,
        commands::catalog::get_track,
        commands::catalog::save_track,
        commands::catalog::delete_track,
        commands::catalog::save_session_metadata,
        commands::catalog::ride_summary,
        commands::bikelog::list_bikes,
        commands::bikelog::bike_for_name,
        commands::bikelog::save_bike,
        commands::bikelog::get_bike_log,
        commands::bikelog::save_bike_log_entry,
        commands::bikelog::attach_bike_log,
        commands::catalog::delete_session,
        commands::catalog::rescan_tracks,
        commands::index::start_index_job,
        commands::index::index_status,
        commands::index::cancel_index_job,
        commands::rebuild::start_rebuild_job,
        commands::rebuild::rebuild_status,
        commands::workbook::open_workbook,
        commands::workbook::read_workbook,
        commands::workbook::eval_workbook,
        commands::workbook::eval_workbook_v2,
        commands::session_set::eval_session_set,
        commands::workbook::save_workbook,
        commands::workbook::create_workbook,
        commands::workbook::fetch_host_channel,
        commands::workbook::fetch_host_channel_v2,
        commands::workbook::list_math_builtins,
        commands::workbook::watch_workbook,
        commands::workbook::unwatch_workbook,
        commands::cursor::cursor_readout,
        commands::rasters::fetch_raster,
        commands::rasters::fetch_raster_meta,
        commands::rasters::fetch_raster_v2,
        commands::rasters::fetch_raster_meta_v2,
        commands::rasters::fetch_fft,
        commands::rasters::fetch_fft_v2,
        commands::histogram::fetch_histogram,
        commands::scatter::fetch_scatter,
        commands::gps::fetch_gps_trace_v2,
        commands::gps::fetch_gps_trace_meta,
        commands::tiles::fetch_tile,
        commands::seams::fetch_seams,
        commands::import::list_importers,
        commands::import::import_file,
        commands::library::set_session_start,
        commands::library::scan_folder,
        commands::library::list_stale_sessions,
        commands::library::reimport_sessions,
        commands::library::inbox_status,
        commands::app::get_settings,
        commands::app::set_settings,
        commands::docs::read_workbook_reference,
        commands::docs::read_cli_reference,
        commands::docs::open_agent_terminal,
        commands::app::get_data_dir,
        commands::app::set_data_dir,
        commands::app::move_data_dir,
        commands::app::list_profiles,
        commands::app::save_profile,
        commands::app::delete_profile,
        commands::maintenance::list_quarantine,
        commands::maintenance::resolve_quarantine,
        commands::maintenance::verify_data_dir,
        commands::sync::sync_status,
        commands::sync::sync_now,
        commands::sync::sync_selection,
        commands::sync::peer_library,
        commands::sync::set_peer_auto_sync,
        commands::sync::pair_peer,
        commands::sync::start_pairing,
        commands::sync::pairing_link,
        commands::sync::pair_with_link,
        commands::sync::unpair_peer,
        commands::sync::set_sync_device_name,
        commands::video::link_video,
        commands::video::match_videos,
        commands::video::set_video_offset,
        commands::video::reestimate_video_offset,
        commands::video::unlink_video,
        crash::crash_reports,
        crash::dismiss_crash_reports,
        crash::share_crash_report,
    ]
}
