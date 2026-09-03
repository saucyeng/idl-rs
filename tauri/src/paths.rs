//! Resolves `<data>` (C4 §1: `app_data_dir()/data`, with an optional
//! `settings.json` override at `app_config_dir()/settings.json`) and creates
//! the C4 §2 directory tree. Pure `std::fs` — the two inputs are plain
//! `Path`s, not Tauri's `AppHandle`, so this is testable with temp
//! directories standing in for `app_data_dir()`/`app_config_dir()`; the
//! Tauri-specific part (calling `app.path().app_data_dir()`) lives in
//! `app/src-tauri`'s `.setup()` hook, which is one line calling this.

use std::path::{Path, PathBuf};

use crate::error::{IpcError, IpcErrorKind};

/// `settings.json`'s shape (C4 §1). Deserialised leniently: a missing or
/// unparsable file, or a missing/empty `data_dir` key, all mean "use the
/// platform default" — this bootstrap file's own corruption must never
/// block the app from opening at all.
#[derive(Debug, Default, serde::Deserialize)]
struct Settings {
    data_dir: Option<String>,
}

/// Resolves `<data>` and ensures the C4 §2 tree exists under it
/// (`blobs/sha256/`, `sessions/`, `workbooks/`, `tracks/`, `tmp/quarantine/`).
/// Idempotent — safe to call on every launch.
pub fn resolve_data_dir(app_data_dir: &Path, app_config_dir: &Path) -> Result<PathBuf, IpcError> {
    let settings_path = app_config_dir.join("settings.json");
    let data_root = match std::fs::read_to_string(&settings_path) {
        Ok(text) => {
            let settings: Settings = serde_json::from_str(&text).unwrap_or_default();
            match settings.data_dir.filter(|d| !d.is_empty()) {
                Some(d) => PathBuf::from(d),
                None => app_data_dir.to_path_buf(),
            }
        }
        Err(_) => app_data_dir.to_path_buf(), // absent file, or unreadable — platform default (C4 §1)
    };
    let data = data_root.join("data");
    for sub in ["blobs/sha256", "sessions", "workbooks", "tracks", "tmp/quarantine"] {
        std::fs::create_dir_all(data.join(sub))
            .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {sub}: {e}")))?;
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_settings_file_present_resolves_to_app_data_dir_slash_data() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();

        // Act
        let data = resolve_data_dir(app_data.path(), app_config.path()).unwrap();

        // Assert
        assert_eq!(data, app_data.path().join("data"));
        assert!(data.join("blobs/sha256").is_dir());
        assert!(data.join("tmp/quarantine").is_dir());
    }

    #[test]
    fn settings_json_data_dir_override_is_honoured() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        let override_root = tempfile::tempdir().unwrap();
        std::fs::write(
            app_config.path().join("settings.json"),
            format!(r#"{{"data_dir":"{}"}}"#, override_root.path().display().to_string().replace('\\', "\\\\")),
        ).unwrap();

        // Act
        let data = resolve_data_dir(app_data.path(), app_config.path()).unwrap();

        // Assert
        assert_eq!(data, override_root.path().join("data"));
    }

    #[test]
    fn corrupt_settings_json_falls_back_to_platform_default() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        std::fs::write(app_config.path().join("settings.json"), "{ not json").unwrap();

        // Act
        let data = resolve_data_dir(app_data.path(), app_config.path()).unwrap();

        // Assert
        assert_eq!(data, app_data.path().join("data"));
    }
}
