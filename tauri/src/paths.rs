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

/// The `detail.reason` string a [`ResolveError::DataDirMissing`] carries on
/// the wire. The frontend routes its launch-time blocking screen on exactly
/// this value (C4 §1 "Missing root"), so it is a contract constant, not a
/// message.
pub const MISSING_ROOT_REASON: &str = "missing_root";

/// Why [`resolve_data_dir`] refused to hand back a `<data>` root.
///
/// Typed rather than a bare [`IpcError`] because the launch path in
/// `app/src-tauri` has to tell the one recoverable case (the override is
/// gone — show the blocking screen) apart from a genuine filesystem failure
/// under the platform default (nothing to recover to).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The `settings.json` override names a path that does not exist, is not
    /// a directory, or cannot be written to (C4 §1 "Missing root", ruling
    /// R196). The library is **not** opened: no fallback to the platform
    /// default, no creation of `path`, no write to the bootstrap file.
    DataDirMissing {
        /// The override root exactly as `settings.json` spells it.
        path: PathBuf,
        /// The underlying condition, for the message only — never routed on.
        detail: String,
    },
    /// Creating the C4 §2 tree failed for a root that is *not* a user
    /// override (the platform default). Not recoverable by choosing a
    /// different folder.
    Io {
        /// What failed, including the offending subdirectory.
        message: String,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::DataDirMissing { path, detail } => {
                write!(f, "the configured data folder '{}' is unavailable: {detail}", path.display())
            }
            ResolveError::Io { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ResolveError {}

impl From<ResolveError> for IpcError {
    /// C3 §2: both variants fold to `io`. `DataDirMissing` additionally
    /// carries `detail: { path, reason: "missing_root" }` — the discriminant
    /// the frontend switches on, since `kind` alone cannot distinguish it
    /// from any other filesystem failure.
    fn from(e: ResolveError) -> Self {
        match &e {
            ResolveError::DataDirMissing { path, .. } => IpcError::with_detail(
                IpcErrorKind::Io,
                e.to_string(),
                serde_json::json!({ "path": path.display().to_string(), "reason": MISSING_ROOT_REASON }),
            ),
            ResolveError::Io { message } => IpcError::new(IpcErrorKind::Io, message.clone()),
        }
    }
}

/// Resolves `<data>` and ensures the C4 §2 tree exists under it
/// (`blobs/sha256/`, `sessions/`, `workbooks/`, `tracks/`, `tmp/quarantine/`).
/// Idempotent — safe to call on every launch.
///
/// When `settings.json` carries a `data_dir` override (C4 §1), that root must
/// already exist as a writable directory. If it does not, this returns
/// [`ResolveError::DataDirMissing`] and touches nothing: the override root is
/// not created, the platform default is not used and not populated, and
/// `settings.json` is not rewritten (C4 §1 "Missing root", ruling R196). An
/// empty-looking library is the failure that rule exists to prevent.
pub fn resolve_data_dir(app_data_dir: &Path, app_config_dir: &Path) -> Result<PathBuf, ResolveError> {
    let settings_path = app_config_dir.join("settings.json");
    let override_root = match std::fs::read_to_string(&settings_path) {
        Ok(text) => {
            let text = text.trim_start_matches('\u{feff}');
            let settings: Settings = serde_json::from_str(text).unwrap_or_default();
            settings.data_dir.filter(|d| !d.is_empty()).map(PathBuf::from)
        }
        Err(_) => None, // absent file, or unreadable — platform default (C4 §1)
    };
    let data_root = override_root.clone().unwrap_or_else(|| app_data_dir.to_path_buf());

    // The override root itself is the user's folder — the app never creates
    // it. Only the tree *underneath* it is the app's to make.
    if let Some(root) = &override_root {
        if !root.is_dir() {
            return Err(ResolveError::DataDirMissing {
                path: root.clone(),
                detail: if root.exists() { "it is not a directory".to_string() } else { "it does not exist".to_string() },
            });
        }
    }

    let data = data_root.join("data");
    for sub in ["blobs/sha256", "sessions", "workbooks", "tracks", "tmp/quarantine"] {
        if let Err(e) = std::fs::create_dir_all(data.join(sub)) {
            // Under an override, a create failure means the folder is there
            // but unwritable (a read-only mount, a permissions change) — the
            // same recoverable condition as an absent one, per C4 §1, which
            // names "does not exist or is not writable" as one case.
            return Err(match &override_root {
                Some(root) => ResolveError::DataDirMissing { path: root.clone(), detail: format!("creating {sub}: {e}") },
                None => ResolveError::Io { message: format!("creating {sub}: {e}") },
            });
        }
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

    #[test]
    fn settings_json_data_dir_override_with_a_leading_bom_is_still_honoured() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        let override_root = tempfile::tempdir().unwrap();
        let json = format!(
            r#"{{"data_dir":"{}"}}"#,
            override_root.path().display().to_string().replace('\\', "\\\\"),
        );
        // '\u{feff}' prepended, matching a UTF-8 BOM as PowerShell's `Out-File`
        // default encoding writes it — the exact real-world failure mode this
        // task fixes (runs/2026-09-03/decisions.md, "settings.json BOM trap").
        let with_bom = format!("\u{feff}{json}");
        std::fs::write(app_config.path().join("settings.json"), with_bom).unwrap();

        // Act
        let data = resolve_data_dir(app_data.path(), app_config.path()).unwrap();

        // Assert
        assert_eq!(data, override_root.path().join("data"));
    }

    #[test]
    fn resolve_data_dir_bom_prefixed_invalid_json_falls_back_to_the_platform_default() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        // '\u{feff}' (UTF-8 BOM) prepended to text that is not valid JSON
        // even once the BOM is stripped — distinct from
        // `corrupt_settings_json_falls_back_to_platform_default` above,
        // which has no BOM.
        std::fs::write(app_config.path().join("settings.json"), "\u{feff}{ not json").unwrap();

        // Act
        let data = resolve_data_dir(app_data.path(), app_config.path()).unwrap();

        // Assert
        assert_eq!(data, app_data.path().join("data"));
    }

    #[test]
    fn resolve_data_dir_absent_override_root_errors_and_creates_nothing_anywhere() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        let gone = tempfile::tempdir().unwrap();
        let missing_root = gone.path().join("unplugged-drive");
        let settings_file = app_config.path().join("settings.json");
        let settings_json =
            format!(r#"{{"data_dir":"{}"}}"#, missing_root.display().to_string().replace('\\', "\\\\"));
        std::fs::write(&settings_file, &settings_json).unwrap();

        // Act
        let result = resolve_data_dir(app_data.path(), app_config.path());

        // Assert
        assert_eq!(
            result,
            Err(ResolveError::DataDirMissing { path: missing_root.clone(), detail: "it does not exist".to_string() })
        );
        // Never falls back: the platform default is not populated.
        assert!(!app_data.path().join("data").exists());
        // Never creates the override path.
        assert!(!missing_root.exists());
        // Never rewrites the bootstrap file.
        assert_eq!(std::fs::read_to_string(&settings_file).unwrap(), settings_json);
    }

    #[test]
    fn resolve_data_dir_override_root_that_is_a_file_errors_as_missing_root() {
        // Arrange
        let app_data = tempfile::tempdir().unwrap();
        let app_config = tempfile::tempdir().unwrap();
        let holder = tempfile::tempdir().unwrap();
        let not_a_dir = holder.path().join("race-data");
        std::fs::write(&not_a_dir, b"i am a file").unwrap();
        std::fs::write(
            app_config.path().join("settings.json"),
            format!(r#"{{"data_dir":"{}"}}"#, not_a_dir.display().to_string().replace('\\', "\\\\")),
        )
        .unwrap();

        // Act
        let result = resolve_data_dir(app_data.path(), app_config.path());

        // Assert
        assert_eq!(
            result,
            Err(ResolveError::DataDirMissing { path: not_a_dir, detail: "it is not a directory".to_string() })
        );
        assert!(!app_data.path().join("data").exists());
    }

    #[test]
    fn data_dir_missing_maps_to_io_with_the_missing_root_detail_the_frontend_routes_on() {
        // Arrange
        let err = ResolveError::DataDirMissing {
            path: PathBuf::from("D:\\race-data"),
            detail: "it does not exist".to_string(),
        };

        // Act
        let ipc: IpcError = err.into();

        // Assert
        assert_eq!(ipc.kind, IpcErrorKind::Io);
        let detail = ipc.detail.expect("DataDirMissing must carry structured detail");
        assert_eq!(detail["reason"], serde_json::json!("missing_root"));
        assert_eq!(detail["path"], serde_json::json!("D:\\race-data"));
    }

    #[test]
    fn resolve_error_io_maps_to_io_with_no_detail() {
        // Arrange
        let err = ResolveError::Io { message: "creating sessions: disk full".to_string() };

        // Act
        let ipc: IpcError = err.into();

        // Assert
        assert_eq!(ipc.kind, IpcErrorKind::Io);
        assert_eq!(ipc.detail, None);
    }
}
