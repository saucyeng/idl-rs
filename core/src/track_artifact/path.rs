//! Where a track lives in a data directory (C4 §2): `tracks/<track_id>.idl1t`.
//!
//! idl1 writes `.idl1t`; idl0 and builds before 2026-11-05 wrote `.idl0t`,
//! in the same format. Every reader goes through this module, so the rule is
//! in one place: a track is read from `.idl1t` and falls back to `.idl0t`,
//! when both exist for one id the `.idl1t` wins, and
//! [`migrate_legacy_track_files`] renames the old files once.

use std::path::{Path, PathBuf};

/// The extension idl1 writes.
pub const TRACK_EXT: &str = "idl1t";
/// The extension idl0 (and idl1 before 2026-11-05) wrote; still read.
pub const LEGACY_TRACK_EXT: &str = "idl0t";

/// `<data_root>/tracks`.
pub fn tracks_dir(data_root: &Path) -> PathBuf {
    data_root.join("tracks")
}

/// `<data_root>/tracks/<track_id>.idl1t` — where a track is written. The
/// caller validates `track_id` as a file-name component first.
pub fn track_file(data_root: &Path, track_id: &str) -> PathBuf {
    tracks_dir(data_root).join(format!("{track_id}.{TRACK_EXT}"))
}

/// `<data_root>/tracks/<track_id>.idl0t`.
pub fn legacy_track_file(data_root: &Path, track_id: &str) -> PathBuf {
    tracks_dir(data_root).join(format!("{track_id}.{LEGACY_TRACK_EXT}"))
}

/// The file `track_id` is read from: its `.idl1t`, else its `.idl0t`, else
/// `None`.
pub fn existing_track_file(data_root: &Path, track_id: &str) -> Option<PathBuf> {
    [track_file(data_root, track_id), legacy_track_file(data_root, track_id)].into_iter().find(|p| p.is_file())
}

/// True for a `.idl1t` or `.idl0t` path.
pub fn is_track_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some(TRACK_EXT) | Some(LEGACY_TRACK_EXT))
}

/// Every track file in `dir`, one per file stem (the `.idl1t` when both
/// exist), sorted by path. A missing directory is an empty list; any other
/// read failure is the `io::Error`.
pub fn list_track_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if is_track_file(&path) {
            paths.push(path);
        }
    }
    let has_current = |legacy: &Path| legacy.with_extension(TRACK_EXT).is_file();
    paths.retain(|p| p.extension().and_then(|e| e.to_str()) == Some(TRACK_EXT) || !has_current(p));
    paths.sort();
    Ok(paths)
}

/// Renames each `tracks/<id>.idl0t` that has no `.idl1t` beside it to
/// `.idl1t`. Run once when a library opens. Returns one warning per file it
/// could not rename; those are still read through the `.idl0t` fallback, so
/// a failure here is never an error.
pub fn migrate_legacy_track_files(data_root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(tracks_dir(data_root)) else { return Vec::new() };
    let mut warnings = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(LEGACY_TRACK_EXT) {
            continue;
        }
        let target = path.with_extension(TRACK_EXT);
        if target.exists() {
            continue;
        }
        if let Err(e) = std::fs::rename(&path, &target) {
            warnings.push(format!("could not rename {} to .{TRACK_EXT}: {e}", path.display()));
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("tracks")).unwrap();
        dir
    }

    #[test]
    fn existing_track_file_both_extensions_present_prefers_idl1t() {
        // Arrange
        let root = temp_root();
        std::fs::write(legacy_track_file(&root, "a"), b"old").unwrap();
        std::fs::write(track_file(&root, "a"), b"new").unwrap();
        std::fs::write(legacy_track_file(&root, "b"), b"old").unwrap();

        // Act
        let a = existing_track_file(&root, "a");
        let b = existing_track_file(&root, "b");
        let c = existing_track_file(&root, "c");

        // Assert
        assert_eq!(a, Some(track_file(&root, "a")));
        assert_eq!(b, Some(legacy_track_file(&root, "b")));
        assert_eq!(c, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_track_files_mixed_extensions_lists_one_per_id_and_skips_other_files() {
        // Arrange
        let root = temp_root();
        std::fs::write(legacy_track_file(&root, "a"), b"").unwrap();
        std::fs::write(track_file(&root, "a"), b"").unwrap();
        std::fs::write(legacy_track_file(&root, "b"), b"").unwrap();
        std::fs::write(track_file(&root, "c"), b"").unwrap();
        std::fs::write(tracks_dir(&root).join("notes.txt"), b"").unwrap();

        // Act
        let files = list_track_files(&tracks_dir(&root)).unwrap();

        // Assert
        assert_eq!(files, vec![track_file(&root, "a"), legacy_track_file(&root, "b"), track_file(&root, "c")]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_track_files_missing_directory_is_empty() {
        // Act
        let files = list_track_files(&std::env::temp_dir().join(format!("idl-rs-none-{}", uuid::Uuid::new_v4()))).unwrap();

        // Assert
        assert!(files.is_empty());
    }

    #[test]
    fn migrate_legacy_track_files_renames_only_ids_without_an_idl1t() {
        // Arrange
        let root = temp_root();
        std::fs::write(legacy_track_file(&root, "a"), b"old-a").unwrap();
        std::fs::write(legacy_track_file(&root, "b"), b"old-b").unwrap();
        std::fs::write(track_file(&root, "b"), b"new-b").unwrap();

        // Act
        let warnings = migrate_legacy_track_files(&root);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(std::fs::read(track_file(&root, "a")).unwrap(), b"old-a");
        assert!(!legacy_track_file(&root, "a").exists());
        assert_eq!(std::fs::read(track_file(&root, "b")).unwrap(), b"new-b");
        assert!(legacy_track_file(&root, "b").exists());

        let _ = std::fs::remove_dir_all(&root);
    }
}
