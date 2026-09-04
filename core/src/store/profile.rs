//! Bike-profile persistence (contract C4 §2 — `profiles/<profile_id>.idl0p`,
//! added post-sign, lead ruling R6). Port of `bike_profile.dart` /
//! `profile_store.dart`: one JSON file per profile, atomic writes; a
//! malformed file is skipped on load and reported to the caller (never
//! printed — core is PURE, CLAUDE.md §2), not a failure of the whole load.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry, AtomicWriteError};

/// One rider/bike profile snapshot (C4 §2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BikeProfile {
    pub profile_id: String,
    #[serde(default)]
    pub profile_name: String,
    #[serde(default)]
    pub created_at_ms: i64,
    #[serde(default)]
    pub updated_at_ms: i64,
    /// The device-config JSON payload (SPEC §8), pushed verbatim.
    pub config: serde_json::Value,
}

/// Discriminant for [`ProfileError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileErrorKind {
    /// A filesystem operation failed (including exhausting C4 §4's
    /// bounded concurrency-retry).
    Io,
    /// The profile did not encode to JSON.
    Encode,
}

/// Error from [`save`] / [`delete`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileError {
    pub kind: ProfileErrorKind,
    pub message: String,
}
impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ProfileError {}
impl From<AtomicWriteError> for ProfileError {
    fn from(e: AtomicWriteError) -> Self {
        ProfileError { kind: ProfileErrorKind::Io, message: e.to_string() }
    }
}

fn profiles_dir(data_root: &Path) -> PathBuf {
    data_root.join("profiles")
}

/// The result of [`load_all`]: every profile that parsed, plus every file
/// that didn't (its path and the reason) — a malformed profile never fails
/// the whole load (matches `profile_store.dart`'s behaviour).
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileLoad {
    /// Successfully-parsed profiles, sorted by `profile_name` ascending.
    pub profiles: Vec<BikeProfile>,
    /// `(path, reason)` for every `*.idl0p` file that failed to parse.
    pub skipped: Vec<(PathBuf, String)>,
}

/// Loads every `<data_root>/profiles/*.idl0p`. Malformed files land in
/// [`ProfileLoad::skipped`] rather than being printed or aborting the load
/// — the caller decides how (or whether) to surface them.
pub fn load_all(data_root: &Path) -> ProfileLoad {
    let dir = profiles_dir(data_root);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return ProfileLoad { profiles: Vec::new(), skipped: Vec::new() };
    };
    let mut profiles = Vec::new();
    let mut skipped = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl0p") {
            continue;
        }
        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<BikeProfile>(&bytes) {
                Ok(p) => profiles.push(p),
                Err(e) => skipped.push((path, e.to_string())),
            },
            Err(e) => skipped.push((path, e.to_string())),
        }
    }
    profiles.sort_by(|a, b| a.profile_name.cmp(&b.profile_name));
    ProfileLoad { profiles, skipped }
}

/// Writes `profile` atomically to `<data_root>/profiles/<profile_id>.idl0p`,
/// replacing any existing file for the same id (last-write-wins, consistent
/// with C4 §6's LWW-by-`updated_at_ms`) via [`write_atomic_with_retry`]
/// (C4 §4 step 4).
pub fn save(data_root: &Path, profile: &BikeProfile) -> Result<(), ProfileError> {
    let bytes = serde_json::to_vec_pretty(profile)
        .map_err(|e| ProfileError { kind: ProfileErrorKind::Encode, message: e.to_string() })?;
    let target = profiles_dir(data_root).join(format!("{}.idl0p", profile.profile_id));
    let based_on = std::fs::read(&target).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(data_root, &target, &bytes, based_on.as_deref(), |_current| bytes.clone())?;
    Ok(())
}

/// Removes a profile's file. No-op when absent.
pub fn delete(data_root: &Path, profile_id: &str) -> Result<(), ProfileError> {
    let path = profiles_dir(data_root).join(format!("{profile_id}.idl0p"));
    if path.is_file() {
        std::fs::remove_file(&path).map_err(|e| ProfileError { kind: ProfileErrorKind::Io, message: e.to_string() })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_then_load_all_round_trips() {
        // Arrange
        let root = temp_root();
        let p = BikeProfile {
            profile_id: "p1".to_string(),
            profile_name: "Trek Session 2024".to_string(),
            created_at_ms: 1,
            updated_at_ms: 2,
            config: serde_json::json!({"wheel_circumference_front_mm": 2300}),
        };

        // Act
        save(&root, &p).unwrap();
        let loaded = load_all(&root);

        // Assert
        assert_eq!(loaded.profiles, vec![p]);
        assert!(loaded.skipped.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_all_skips_a_malformed_file_without_failing_the_load() {
        // Arrange
        let root = temp_root();
        std::fs::create_dir_all(profiles_dir(&root)).unwrap();
        std::fs::write(profiles_dir(&root).join("bad.idl0p"), b"not json").unwrap();
        let good = BikeProfile {
            profile_id: "p1".to_string(),
            profile_name: "Good".to_string(),
            created_at_ms: 0,
            updated_at_ms: 0,
            config: serde_json::json!({}),
        };
        save(&root, &good).unwrap();

        // Act
        let loaded = load_all(&root);

        // Assert
        assert_eq!(loaded.profiles, vec![good]);
        assert_eq!(loaded.skipped.len(), 1);
        assert_eq!(loaded.skipped[0].0.file_name().unwrap().to_str().unwrap(), "bad.idl0p");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_removes_the_file_and_is_a_no_op_when_absent() {
        // Arrange
        let root = temp_root();
        let p = BikeProfile {
            profile_id: "p1".to_string(),
            profile_name: String::new(),
            created_at_ms: 0,
            updated_at_ms: 0,
            config: serde_json::json!({}),
        };
        save(&root, &p).unwrap();

        // Act
        delete(&root, "p1").unwrap();
        delete(&root, "does-not-exist").unwrap();

        // Assert
        assert!(load_all(&root).profiles.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_overwrites_an_existing_profile_with_new_values() {
        // Arrange
        let root = temp_root();
        let mut p = BikeProfile {
            profile_id: "p1".to_string(),
            profile_name: "Original".to_string(),
            created_at_ms: 0,
            updated_at_ms: 0,
            config: serde_json::json!({}),
        };
        save(&root, &p).unwrap();

        // Act
        p.profile_name = "Renamed".to_string();
        save(&root, &p).unwrap();
        let loaded = load_all(&root);

        // Assert
        assert_eq!(loaded.profiles.len(), 1);
        assert_eq!(loaded.profiles[0].profile_name, "Renamed");

        let _ = std::fs::remove_dir_all(&root);
    }
}
