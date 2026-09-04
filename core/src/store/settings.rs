//! App-wide settings persistence (port of `app_settings.dart`) — reuses
//! `app_config_dir()/settings.json`, the bootstrap file contract C4 §1
//! already fixes for `data_dir` (and, per lead ruling R15, `rider_name` /
//! `unit_system` too), rather than a second settings file. This module
//! reads/writes plain `std::fs` — it has no idea where `app_config_dir()`
//! resolves to on any platform (that's Tauri's path resolver, L5's layer);
//! it takes the resolved path as a parameter.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry, AtomicWriteError};

/// Rider-facing unit system (C4 §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitSystem {
    Imperial,
    Metric,
}

/// The `settings.json` document (C4 §1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(default)]
    pub data_dir: Option<String>, // C4 §1's existing key
    #[serde(default = "default_rider_name")]
    pub rider_name: String,
    #[serde(default = "default_unit_system")]
    pub unit_system: UnitSystem,
}

fn default_rider_name() -> String {
    String::new()
}
fn default_unit_system() -> UnitSystem {
    UnitSystem::Imperial
}

impl Default for AppSettings {
    fn default() -> Self {
        AppSettings { data_dir: None, rider_name: default_rider_name(), unit_system: default_unit_system() }
    }
}

/// Discriminant for [`SettingsError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsErrorKind {
    /// A filesystem operation failed (including exhausting C4 §4's
    /// bounded concurrency-retry).
    Io,
    /// The settings document did not encode to JSON.
    Encode,
}

/// Error from [`save`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
    pub kind: SettingsErrorKind,
    pub message: String,
}
impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for SettingsError {}

/// Loads `settings.json` at `path`; defaults (including `data_dir: None`,
/// matching C4 §1's "absent → platform default") when the file is absent
/// or malformed — settings are UI convenience, never a load-blocking
/// concern (CLAUDE.md §5).
pub fn load(path: &Path) -> AppSettings {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Writes `settings.json` atomically, replacing any existing content
/// (last-write-wins, via [`write_atomic_with_retry`] — the same C4 §4
/// step 4 pattern every other whole-document writer in this crate uses).
/// Unlike every other file class this crate writes, `settings.json` lives
/// **outside** `<data>` (C4 §1), so this function takes `path` directly
/// rather than a `data_root` — there is no `tmp/` beside it to stage
/// through; it stages in the same directory as `path` itself instead.
pub fn save(path: &Path, settings: &AppSettings) -> Result<(), SettingsError> {
    let bytes = serde_json::to_vec_pretty(settings)
        .map_err(|e| SettingsError { kind: SettingsErrorKind::Encode, message: e.to_string() })?;
    let parent = path
        .parent()
        .ok_or_else(|| SettingsError { kind: SettingsErrorKind::Io, message: "settings path has no parent".to_string() })?;
    let based_on = std::fs::read(path).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(parent, path, &bytes, based_on.as_deref(), |_current| bytes.clone())
        .map(|_| ())
        .map_err(|e: AtomicWriteError| SettingsError { kind: SettingsErrorKind::Io, message: e.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_settings_path() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("settings.json")
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        // Arrange
        let path = temp_settings_path();

        // Act
        let settings = load(&path);

        // Assert
        assert_eq!(settings, AppSettings::default());
    }

    #[test]
    fn save_then_load_round_trips() {
        // Arrange
        let path = temp_settings_path();
        let mut settings = AppSettings::default();
        settings.rider_name = "Isaac".to_string();
        settings.data_dir = Some("D:\\race-data".to_string());

        // Act
        save(&path, &settings).unwrap();
        let back = load(&path);

        // Assert
        assert_eq!(back, settings);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn malformed_settings_file_falls_back_to_defaults() {
        // Arrange
        let path = temp_settings_path();
        std::fs::write(&path, b"not json").unwrap();

        // Act
        let settings = load(&path);

        // Assert
        assert_eq!(settings, AppSettings::default());

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn save_overwrites_existing_settings_with_new_values() {
        // Arrange
        let path = temp_settings_path();
        let mut settings = AppSettings::default();
        settings.rider_name = "Isaac".to_string();
        save(&path, &settings).unwrap();

        // Act
        settings.rider_name = "Renamed".to_string();
        settings.unit_system = UnitSystem::Metric;
        save(&path, &settings).unwrap();
        let back = load(&path);

        // Assert
        assert_eq!(back.rider_name, "Renamed");
        assert_eq!(back.unit_system, UnitSystem::Metric);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
