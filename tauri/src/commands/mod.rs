//! `#[tauri::command]` handlers. Thin: convert, call the engine, return.

pub mod device;

/// Reports the engine crate version so the UI (and sync peers) can prove
/// they compute with the same engine.
#[tauri::command]
pub fn engine_version() -> String {
    idl_rs::VERSION.to_string()
}

/// Encodes `values` as bare little-endian `f32` bytes.
///
/// M0 smoke layout only — contract C3 defines the real tile layout (header,
/// min/max pairs, per-column stats). Kept as a pure function so it is testable
/// without a Tauri runtime.
pub fn encode_f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Returns `n` ascending `f32` values `0.0, 1.0, …` as raw bytes, proving the
/// binary IPC path (`tauri::ipc::Response` → `ArrayBuffer` → `Float32Array`).
#[tauri::command]
pub fn smoke_tile(n: u32) -> tauri::ipc::Response {
    let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
    tauri::ipc::Response::new(encode_f32_le(&values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_f32_le_four_values_roundtrips_bytes() {
        // Arrange
        let values = [0.0f32, 1.0, 2.0, 3.0];

        // Act
        let bytes = encode_f32_le(&values);
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Assert
        assert_eq!(bytes.len(), 16);
        assert_eq!(decoded, values);
    }

    #[test]
    fn engine_version_matches_core_crate_version() {
        // Arrange / Act
        let v = engine_version();

        // Assert
        assert_eq!(v, idl_rs::VERSION);
        assert!(!v.is_empty());
    }
}
