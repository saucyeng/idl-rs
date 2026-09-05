//! App group (C3 §3.10): `get_settings`, `set_settings`, `get_data_dir`,
//! `set_data_dir` — thin wrappers over the already-landed
//! `idl_rs::store::settings` module and `crate::paths::resolve_data_dir`.
//! Satisfies wave-2 needs L7-6/L7-7 (ruling R53 Settings Q1/Q4).
//!
//! Same `_via`-suffixed idiom as `commands/catalog.rs`: each
//! `#[tauri::command]` resolves `app.path()...` (via `tauri::Manager`) and
//! delegates to a plain function taking paths, which this module's own
//! tests exercise directly with temp dirs — `tauri::AppHandle`/
//! `tauri::State` cannot be constructed outside a running app.
//!
//! Deviation from the plan's own interface sketch (documented, not a wire
//! shape change): `unit_system` is typed as
//! `idl_rs::store::settings::UnitSystem` directly rather than a plain
//! `String` with hand-written mapping functions. `UnitSystem` already
//! derives `Serialize`/`Deserialize` with `#[serde(rename_all =
//! "snake_case")]`, so it already serialises to exactly `"imperial"` /
//! `"metric"` and deserialises from exactly those two strings — an
//! unrecognised string becomes a Tauri-level argument-deserialisation
//! failure before the command body runs, rather than inventing silent-default
//! or reject fallback behaviour C3 §3.10 does not specify.

use std::path::{Path, PathBuf};

use tauri::Manager;

use idl_rs::store::settings::{AppSettings, SettingsError, SettingsErrorKind, UnitSystem};

use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// C3 §3.10 `AppSettings` — `get_settings`'s return and (as
/// [`AppSettingsArg`]) `set_settings`'s argument shape.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AppSettingsDto {
    /// The `<data>` root override, or `None` when the platform default is
    /// in use (C4 §1). Always echoes the current on-disk value — see
    /// `set_settings_via`, which ignores this field on its argument.
    pub data_dir: Option<String>,
    /// The rider's display name. `""` means not set (C4 §1).
    pub rider_name: String,
    /// Unit system used across the app. Engine default is `Imperial`.
    pub unit_system: UnitSystem,
}

impl From<AppSettings> for AppSettingsDto {
    fn from(s: AppSettings) -> Self {
        Self { data_dir: s.data_dir, rider_name: s.rider_name, unit_system: s.unit_system }
    }
}

/// `set_settings`'s argument. Same fields as [`AppSettingsDto`] — `data_dir`
/// is present on the wire (C3's `AppSettings` is one TS interface for both
/// directions) but ignored server-side; `set_data_dir` is the sole writer
/// of that key (ruling R59 Q5).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AppSettingsArg {
    /// Present on the wire for symmetry with [`AppSettingsDto`]; ignored by
    /// `set_settings_via`.
    pub data_dir: Option<String>,
    /// The rider's display name to persist. `""` means not set.
    pub rider_name: String,
    /// Unit system to persist.
    pub unit_system: UnitSystem,
}

/// C3 §3.10 `DataDirInfo`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DataDirInfo {
    /// The `<data>` root actually in use for this process — the managed
    /// `state::DataDir`, resolved once at startup and cached for the
    /// process lifetime (C4 §1).
    pub resolved_path: String,
    /// The override from `settings.json`, or `None` when the platform
    /// default is in use.
    pub override_path: Option<String>,
    /// `true` when re-resolving `data_dir` right now would give a
    /// different path than `resolved_path` — i.e. the override changed on
    /// disk since startup and the app has not yet restarted.
    pub restart_required: bool,
}

/// `app_config_dir()/settings.json` (C4 §1's existing bootstrap file).
fn settings_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("settings.json")
}

/// Maps [`SettingsError`] to the C3 §3.10 error rows (`io`, `internal`).
/// `SettingsErrorKind::Encode` folds to `internal` per C3 §2's folding rule
/// — `load` never fails, so only `save`'s errors reach this function.
fn map_settings_error(e: SettingsError) -> IpcError {
    match e.kind {
        SettingsErrorKind::Io => IpcError::new(IpcErrorKind::Io, e.message),
        SettingsErrorKind::Encode => IpcError::new(IpcErrorKind::Internal, e.message),
    }
}

/// `get_settings`'s transport-agnostic core. `load` never fails (C4 §1) —
/// a missing or malformed file yields defaults.
fn get_settings_via(settings_path: &Path) -> AppSettingsDto {
    idl_rs::store::settings::load(settings_path).into()
}

/// `set_settings`'s transport-agnostic core: a whole-document replace of
/// `rider_name`/`unit_system` only. `arg.data_dir` is deliberately not
/// applied — `set_data_dir_via` is the sole writer of that key (ruling R59
/// Q5). Re-reads after the write so the response reflects what is actually
/// on disk, not merely echoed from the argument.
fn set_settings_via(settings_path: &Path, arg: AppSettingsArg) -> Result<AppSettingsDto, IpcError> {
    let mut current = idl_rs::store::settings::load(settings_path);
    current.rider_name = arg.rider_name;
    current.unit_system = arg.unit_system;
    idl_rs::store::settings::save(settings_path, &current).map_err(map_settings_error)?;
    Ok(idl_rs::store::settings::load(settings_path).into())
}

/// `get_data_dir`'s transport-agnostic core. `resolved_data_dir` is always
/// the managed `DataDir` fixed at startup, never a freshly recomputed
/// value — comparing the two is precisely what makes `restart_required` a
/// real condition rather than defensive coding.
fn get_data_dir_via(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
) -> Result<DataDirInfo, IpcError> {
    let settings = idl_rs::store::settings::load(settings_path);
    let fresh = crate::paths::resolve_data_dir(app_data_dir, app_config_dir)?;
    Ok(DataDirInfo {
        resolved_path: resolved_data_dir.display().to_string(),
        override_path: settings.data_dir,
        restart_required: fresh != resolved_data_dir,
    })
}

/// `set_data_dir`'s transport-agnostic core: read-modify-write of only the
/// `data_dir` key, preserving `rider_name`/`unit_system` (ruling R59 Q5).
/// Does not move existing files. `path` must be an absolute path the app
/// can create the C4 §2 tree under; a relative path or an uncreatable path
/// is `invalid_argument`. The new tree is created *before* `settings.json`
/// is written — a failed create leaves nothing written.
fn set_data_dir_via(
    settings_path: &Path,
    app_data_dir: &Path,
    app_config_dir: &Path,
    resolved_data_dir: &Path,
    path: Option<String>,
) -> Result<DataDirInfo, IpcError> {
    if let Some(p) = &path {
        if !Path::new(p).is_absolute() {
            return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("'{p}' is not an absolute path")));
        }
        std::fs::create_dir_all(Path::new(p).join("data"))
            .map_err(|e| IpcError::new(IpcErrorKind::InvalidArgument, format!("cannot create '{p}': {e}")))?;
    }
    let mut current = idl_rs::store::settings::load(settings_path);
    current.data_dir = path.clone();
    idl_rs::store::settings::save(settings_path, &current).map_err(map_settings_error)?;
    let fresh = crate::paths::resolve_data_dir(app_data_dir, app_config_dir)?;
    Ok(DataDirInfo {
        resolved_path: resolved_data_dir.display().to_string(),
        override_path: path,
        restart_required: fresh != resolved_data_dir,
    })
}

/// Maps a `tauri::Error` from `app.path()...` itself (launch-time
/// path-resolution failing at command time) to `Internal` — no C3 §3.10
/// error row names it because it should never actually happen once
/// `.setup()` has run once at launch.
fn map_path_error(e: tauri::Error) -> IpcError {
    IpcError::new(IpcErrorKind::Internal, e.to_string())
}

/// C3 §3.10 `get_settings()`. Loads `settings.json` from
/// `app_config_dir()`; never fails on a missing/malformed file (C4 §1).
/// Generic over `R: tauri::Runtime` — `handler()` is itself generic, and a
/// non-generic `tauri::AppHandle` (fixed to the default runtime) does not
/// implement `CommandArg` for an arbitrary `R`.
#[tauri::command]
pub fn get_settings<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<AppSettingsDto, IpcError> {
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    Ok(get_settings_via(&settings_path(&app_config_dir)))
}

/// C3 §3.10 `set_settings(settings)`. Ignores `settings.data_dir` (ruling
/// R59 Q5 — `set_data_dir` is the sole writer of that key) and persists
/// `rider_name`/`unit_system`.
#[tauri::command]
pub fn set_settings<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    settings: AppSettingsArg,
) -> Result<AppSettingsDto, IpcError> {
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    set_settings_via(&settings_path(&app_config_dir), settings)
}

/// C3 §3.10 `get_data_dir()`. `resolved_path` is the managed `DataDir`
/// fixed at startup, not a freshly recomputed value.
#[tauri::command]
pub fn get_data_dir<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<DataDirInfo, IpcError> {
    let app_data_dir = app.path().app_data_dir().map_err(map_path_error)?;
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    get_data_dir_via(&settings_path(&app_config_dir), &app_data_dir, &app_config_dir, &data_dir.0)
}

/// C3 §3.10 `set_data_dir(path)`. Writes only the `data_dir` key,
/// read-modify-write, preserving `rider_name`/`unit_system`; does not move
/// existing files.
#[tauri::command]
pub fn set_data_dir<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    data_dir: tauri::State<'_, DataDir>,
    path: Option<String>,
) -> Result<DataDirInfo, IpcError> {
    let app_data_dir = app.path().app_data_dir().map_err(map_path_error)?;
    let app_config_dir = app.path().app_config_dir().map_err(map_path_error)?;
    set_data_dir_via(&settings_path(&app_config_dir), &app_data_dir, &app_config_dir, &data_dir.0, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// A fresh temp dir standing in for one of `app_data_dir()`/
    /// `app_config_dir()`/an arbitrary override root.
    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-app-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn get_settings_via_missing_file_returns_defaults() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);

        // Act
        let dto = get_settings_via(&path);

        // Assert
        assert_eq!(dto.data_dir, None);
        assert_eq!(dto.rider_name, "");
        assert_eq!(dto.unit_system, UnitSystem::Imperial);
    }

    #[test]
    fn get_settings_via_present_file_round_trips_every_field() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let written = AppSettings {
            data_dir: Some("D:\\race-data".to_string()),
            rider_name: "Isaac".to_string(),
            unit_system: UnitSystem::Metric,
        };
        idl_rs::store::settings::save(&path, &written).unwrap();

        // Act
        let dto = get_settings_via(&path);

        // Assert
        assert_eq!(dto.data_dir, written.data_dir);
        assert_eq!(dto.rider_name, written.rider_name);
        assert_eq!(dto.unit_system, written.unit_system);
    }

    #[test]
    fn set_settings_via_ignores_data_dir_in_the_argument() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let existing = AppSettings {
            data_dir: Some("D:\\existing-override".to_string()),
            rider_name: String::new(),
            unit_system: UnitSystem::Imperial,
        };
        idl_rs::store::settings::save(&path, &existing).unwrap();
        let arg = AppSettingsArg {
            data_dir: Some("D:\\attempted-override".to_string()),
            rider_name: "Isaac".to_string(),
            unit_system: UnitSystem::Metric,
        };

        // Act
        let dto = set_settings_via(&path, arg).unwrap();

        // Assert
        assert_eq!(dto.data_dir, Some("D:\\existing-override".to_string()));
    }

    #[test]
    fn set_settings_via_writes_rider_name_and_unit_system_and_reflects_the_reread_value() {
        // Arrange
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let arg =
            AppSettingsArg { data_dir: None, rider_name: "Isaac".to_string(), unit_system: UnitSystem::Metric };

        // Act
        let dto = set_settings_via(&path, arg).unwrap();

        // Assert
        assert_eq!(dto.rider_name, "Isaac");
        assert_eq!(dto.unit_system, UnitSystem::Metric);
        let reread = idl_rs::store::settings::load(&path);
        assert_eq!(reread.rider_name, "Isaac");
        assert_eq!(reread.unit_system, UnitSystem::Metric);
    }

    #[test]
    fn get_data_dir_via_no_override_matches_a_fresh_resolve_and_restart_is_not_required() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let info = get_data_dir_via(&path, &app_data, &app_config, &resolved).unwrap();

        // Assert
        assert_eq!(info.resolved_path, resolved.display().to_string());
        assert_eq!(info.override_path, None);
        assert!(!info.restart_required);
    }

    #[test]
    fn get_data_dir_via_override_changed_on_disk_reports_restart_required() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        // The managed value at startup — no override existed yet.
        let old_resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        // "Changed on disk, app not yet restarted": a new override root is
        // written directly to settings.json after that snapshot.
        let override_root = temp_root();
        idl_rs::store::settings::save(
            &path,
            &AppSettings {
                data_dir: Some(override_root.display().to_string()),
                rider_name: String::new(),
                unit_system: UnitSystem::Imperial,
            },
        )
        .unwrap();

        // Act
        let info = get_data_dir_via(&path, &app_data, &app_config, &old_resolved).unwrap();

        // Assert
        assert_eq!(info.resolved_path, old_resolved.display().to_string());
        assert_eq!(info.override_path, Some(override_root.display().to_string()));
        assert!(info.restart_required);
    }

    #[test]
    fn set_data_dir_via_relative_path_is_rejected_and_nothing_is_written() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let result = set_data_dir_via(&path, &app_data, &app_config, &resolved, Some("relative/dir".to_string()));

        // Assert
        assert!(matches!(result, Err(e) if e.kind == IpcErrorKind::InvalidArgument));
        assert!(!path.exists());
    }

    #[test]
    fn set_data_dir_via_absolute_path_succeeds_and_creates_the_data_subdir() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();
        let new_root = temp_root();
        let new_root_str = new_root.display().to_string();

        // Act
        let info = set_data_dir_via(&path, &app_data, &app_config, &resolved, Some(new_root_str.clone())).unwrap();

        // Assert
        assert!(new_root.join("data").is_dir());
        assert_eq!(info.override_path, Some(new_root_str));
        assert!(info.restart_required);
    }

    #[test]
    fn set_data_dir_via_none_clears_an_existing_override() {
        // Arrange
        let app_data = temp_root();
        let app_config = temp_root();
        let path = settings_path(&app_config);
        let override_root = temp_root();
        idl_rs::store::settings::save(
            &path,
            &AppSettings {
                data_dir: Some(override_root.display().to_string()),
                rider_name: String::new(),
                unit_system: UnitSystem::Imperial,
            },
        )
        .unwrap();
        let resolved = crate::paths::resolve_data_dir(&app_data, &app_config).unwrap();

        // Act
        let info = set_data_dir_via(&path, &app_data, &app_config, &resolved, None).unwrap();

        // Assert
        assert_eq!(info.override_path, None);
        let reread = idl_rs::store::settings::load(&path);
        assert_eq!(reread.data_dir, None);
    }
}
