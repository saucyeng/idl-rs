//! `#[tauri::command]` handlers. Thin: convert, call the engine, return.

pub mod app;
pub mod catalog;
pub mod cursor;
pub mod device;
pub mod import;
pub mod rasters;
pub mod tiles;
pub mod workbook;

/// Reports the engine crate version so the UI (and sync peers) can prove
/// they compute with the same engine.
#[tauri::command]
pub fn engine_version() -> String {
    idl_rs::VERSION.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_version_matches_core_crate_version() {
        // Arrange / Act
        let v = engine_version();

        // Assert
        assert_eq!(v, idl_rs::VERSION);
        assert!(!v.is_empty());
    }
}
