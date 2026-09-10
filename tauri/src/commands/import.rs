//! Import commands (C3 §3.3): `import_file`, `list_importers`. Bridges C3's
//! `importer_id` vocabulary to `idl_rs::store::import`'s extension-keyed
//! `import_idl0`/`import_file` entry points (L2 Task 6, L2-R13) and
//! `idl_rs::import::importers()`'s FIT/GPX/CSV registry (L2 Task 7, R51 Q2).
//!
//! Same `_via`-function idiom as every other module in this directory: each
//! `#[tauri::command]` is a thin wrapper over a plain function this module's
//! own tests exercise directly (`tauri::State`/`tauri::ipc::Channel` cannot
//! be constructed outside a running app).

use std::path::Path;

use idl_rs::store::import as core_import;

use crate::commands::catalog::SessionSummary;
use crate::commands::device::Progress;
use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// C3 §3.3 `ImporterInfo` — one importer `list_importers` advertises and
/// `import_file`'s `importer_id` argument accepts. `extensions` are
/// dot-prefixed on the wire (e.g. `".gpx"`), matching C3's own worked
/// example, even though core's `idl_rs::import::ImporterInfo::extensions`
/// stores them bare (matching `idl_rs::import::importer_for_extension`'s own
/// `ext` convention) — this `From` impl is the one place that mismatch is
/// resolved.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImporterInfo {
    /// Stable id (e.g. `"gpx"`) — what `import_file`'s `importer_id` forces.
    pub id: String,
    /// Human-readable label (e.g. `"GPX track"`).
    pub label: String,
    /// Dot-prefixed lowercase extensions (e.g. `[".gpx"]`).
    pub extensions: Vec<String>,
}

impl From<&idl_rs::import::ImporterInfo> for ImporterInfo {
    fn from(i: &idl_rs::import::ImporterInfo) -> Self {
        Self {
            id: i.id.to_string(),
            label: i.label.to_string(),
            extensions: i.extensions.iter().map(|e| format!(".{e}")).collect(),
        }
    }
}

/// C3 §3.3 `import_file`'s return (lead ruling R60, 2026-09-05): wraps the
/// catalog row with the importer's own recovered warnings (a dropped
/// duplicate GPX timestamp, a `.idl0` truncation recovery) — a
/// `SessionSummary` alone (a fixed catalog-row shape) has no field to carry
/// a per-import advisory, and dropping one would violate CLAUDE.md §5.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportOutcome {
    /// The freshly-imported session's catalog row.
    pub session: SessionSummary,
    /// Non-fatal advisory messages raised while importing (never silently
    /// dropped) — empty on a clean import.
    pub warnings: Vec<String>,
}

/// Transport-agnostic core of `list_importers` (C3 §3.3): the `"idl0"` row
/// (`idl_rs::import::importers()` deliberately excludes it, R51 Q1/Q2 —
/// `.idl0` has its own entry point outside that registry's scope) plus every
/// row `idl_rs::import::importers()` enumerates, with extensions
/// dot-prefixed for the wire.
fn list_importers_via() -> Vec<ImporterInfo> {
    let mut out = vec![ImporterInfo {
        id: "idl0".to_string(),
        label: "IDL0 log".to_string(),
        extensions: vec![".idl0".to_string()],
    }];
    out.extend(idl_rs::import::importers().iter().map(ImporterInfo::from));
    out
}

/// C3 §3.3 `list_importers()`.
#[tauri::command]
pub fn list_importers() -> Vec<ImporterInfo> {
    list_importers_via()
}

/// Resolves `path`/`importer_id` to the lowercase, dot-free extension string
/// `idl_rs::store::import::{import_idl0, import_file}` dispatch on (C3 §3.3:
/// `importer_id` is `null` for extension-based auto-detection, or one of the
/// ids `list_importers` returns to "force a specific importer"). Every id in
/// this vocabulary (`"idl0"` plus every `idl_rs::import::importers()` id —
/// `"fit"`/`"gpx"`/`"csv"`) is already spelled identically to the extension
/// `import_idl0`/`import_file` expect, so forcing by id needs no separate
/// translation table; forcing ignores the file's own actual extension
/// entirely, by design.
fn resolve_extension(path: &str, importer_id: Option<&str>) -> Result<String, IpcError> {
    match importer_id {
        Some(id) => {
            if id == "idl0" || idl_rs::import::importers().iter().any(|i| i.id == id) {
                Ok(id.to_string())
            } else {
                Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("unrecognised importer_id '{id}'")))
            }
        }
        None => {
            let ext = Path::new(path).extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
            ext.ok_or_else(|| {
                IpcError::new(IpcErrorKind::InvalidArgument, format!("'{path}' has no file extension to auto-detect an importer from"))
            })
        }
    }
}

/// Transport-agnostic core of `import_file` (C3 §3.3). Reads `path` from
/// disk — the user's own pasted path (ledger R55), never resolved against
/// `data_dir` — dispatches to `import_idl0` (the `"idl0"` branch) or
/// `import_file` (every other resolved extension; core's own `import_file`
/// refuses `"idl0"` by design, L2-R13) by [`resolve_extension`]'s result,
/// rebuilds the catalog (C4 §5's rebuild is cheap and idempotent —
/// incremental catalog indexing does not exist yet), and reads the
/// freshly-imported row back from `catalog_read::list_sessions` by
/// `session_id`. `on_progress` is called with `"reading"` before the file
/// is sized and mapped, `"decoding"` before the importer call, and
/// `"materializing"` after a successful import and before the catalog
/// rebuild — never called again once a call has failed.
fn import_file_via(
    data_dir: &Path,
    path: &str,
    importer_id: Option<&str>,
    mut on_progress: impl FnMut(&str),
) -> Result<ImportOutcome, IpcError> {
    let ext = resolve_extension(path, importer_id)?;

    on_progress("reading");
    // Sized before anything is read (ruling R203.4). An import's heap peak
    // is the parsed `Session` plus one channel's Arrow column plus the
    // encoded output — the raw log itself is a memory map, not heap, since
    // R203.3 — so twice the file size is a conservative ceiling for it.
    let file_len = std::fs::metadata(path)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                IpcError::new(IpcErrorKind::NotFound, format!("'{path}' does not exist"))
            } else {
                IpcError::new(IpcErrorKind::Io, format!("reading {path}: {e}"))
            }
        })?
        .len();
    crate::memory::ensure_fits(file_len.saturating_mul(2), &format!("import '{path}'"))?;

    on_progress("decoding");
    // The path forms map the file rather than copying it onto the heap
    // (ruling R203.3).
    let report = if ext == "idl0" {
        core_import::import_idl0_path(data_dir, Path::new(path))
    } else {
        core_import::import_file_path(data_dir, &ext, Path::new(path))
    }
    .map_err(IpcError::from)?;

    on_progress("materializing");
    let session_id = report.session_id.clone();
    idl_rs::store::catalog_read::rebuild_catalog_report(data_dir)?;
    let session = idl_rs::store::catalog_read::list_sessions(data_dir)?
        .into_iter()
        .find(|s| s.session_id == session_id)
        .ok_or_else(|| {
            IpcError::new(IpcErrorKind::Internal, format!("session {session_id} missing from the catalog immediately after import"))
        })?;

    let mut warnings = report.import_warnings;
    if let Some(truncation) = report.truncation_warning {
        warnings.push(truncation);
    }

    Ok(ImportOutcome { session: SessionSummary::from(session), warnings })
}

/// C3 §3.3 `import_file(path, importer_id, progress)`.
#[tauri::command(async)]
pub fn import_file(
    path: String,
    importer_id: Option<String>,
    progress: tauri::ipc::Channel<Progress>,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, crate::session_cache::SessionCache>,
) -> Result<ImportOutcome, IpcError> {
    let outcome = import_file_via(&data_dir.0, &path, importer_id.as_deref(), |phase| {
        let _ = progress.send(Progress { done: 0, total: None, phase: phase.to_string() });
    })?;
    // A re-import under an existing `session_id` rewrites that session's
    // `data.parquet`, so any channel decoded from the old one is stale
    // (ruling R203.2). Harmless on a first import — nothing is resident
    // under a session id that did not exist a moment ago.
    cache.invalidate_session(&outcome.session.session_id);
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::parse::test_buffers::{frame, session_end, v3_registry_entry, Header};
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-import-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes `contents` to a fresh temp file named `name` and returns its
    /// path as a `String` (the shape `import_file`'s `path` argument takes).
    fn temp_file(name: &str, contents: &[u8]) -> String {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-import-src-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_string()
    }

    const GPX_TWO_POINTS: &str = r#"<gpx><trk><trkseg><trkpt lat="45.0" lon="-90.0"><time>2026-01-01T00:00:00Z</time></trkpt><trkpt lat="45.001" lon="-90.001"><time>2026-01-01T00:00:01Z</time></trkpt></trkseg></trk></gpx>"#;

    /// A minimal valid `.idl0` schema-3 buffer: one IMU axis, one sample of
    /// `axis_value`, then `SESSION_END` — same construction as
    /// `parse::mod`'s own `auto_detects_v3_and_applies_scaling` dispatch
    /// test. Two calls with different `axis_value`s share the same header
    /// (hence the same `session_id`, taken from the header's own UUID field
    /// — see `parse::v3`) but hash to different `blob_sha256`s, exercising
    /// [`idl_rs::store::import::ImportPlan::Collision`].
    fn minimal_idl0_buffer(axis_value: i16) -> Vec<u8> {
        let accel_scale: f32 = 32.0 / 32768.0;
        let mut buf = Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }
            .build(&[v3_registry_entry(0, 4, 800, accel_scale, 0.0, "IMU0_AccelX", "g")]);
        // IMU_SAMPLE payload: [imu_index:u8][ts_us:i64][axis i16 ...].
        let mut payload = vec![0u8];
        payload.extend_from_slice(&1_250_i64.to_le_bytes());
        payload.extend_from_slice(&axis_value.to_le_bytes());
        buf.extend(frame(0x01, &payload));
        buf.extend(session_end());
        buf
    }

    #[test]
    fn list_importers_via_includes_idl0_and_every_core_importer_with_dotted_extensions() {
        // Arrange
        let core_count = idl_rs::import::importers().len();

        // Act
        let out = list_importers_via();

        // Assert
        assert_eq!(out.len(), 1 + core_count);
        let idl0 = out.iter().find(|i| i.id == "idl0").unwrap();
        assert_eq!(idl0.extensions, vec![".idl0".to_string()]);
        for info in out.iter().filter(|i| i.id != "idl0") {
            for ext in &info.extensions {
                assert!(ext.starts_with('.'), "extension {ext} was not dot-prefixed");
            }
        }
    }

    #[test]
    fn import_file_via_gpx_happy_path_returns_a_session_summary() {
        // Arrange
        let data_root = temp_root();
        let path = temp_file("track.gpx", GPX_TWO_POINTS.as_bytes());
        let mut phases = Vec::new();

        // Act
        let outcome = import_file_via(&data_root, &path, None, |p| phases.push(p.to_string())).unwrap();

        // Assert
        assert_eq!(outcome.session.source_format, "gpx");
        assert_eq!(phases, vec!["reading", "decoding", "materializing"]);
        let sessions = idl_rs::store::catalog_read::list_sessions(&data_root).unwrap();
        assert_eq!(sessions.len(), 1);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_forced_importer_id_ignores_the_actual_extension() {
        // Arrange — a GPX fixture saved with a `.txt` extension.
        let data_root = temp_root();
        let path = temp_file("track.txt", GPX_TWO_POINTS.as_bytes());

        // Act
        let outcome = import_file_via(&data_root, &path, Some("gpx"), |_| {}).unwrap();

        // Assert
        assert_eq!(outcome.session.source_format, "gpx");

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_unrecognised_importer_id_is_invalid_argument() {
        // Arrange — a path that would raise `NotFound` if validation ran
        // after the read, proving validation happens first.
        let data_root = temp_root();

        // Act
        let err = import_file_via(&data_root, "/nonexistent/x.xml", Some("xml"), |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_missing_path_is_not_found() {
        // Arrange
        let data_root = temp_root();

        // Act
        let err = import_file_via(&data_root, "/nonexistent/x.gpx", None, |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_malformed_gpx_maps_to_import_gpx_malformed_xml() {
        // Arrange — `</gpx>` closes `<trk>` while it's still open; quick-xml's
        // `check_end_names` rejects this (same fixture as core's own
        // `gpx_importer_malformed_xml_returns_typed_error`).
        let data_root = temp_root();
        let path = temp_file("bad.gpx", b"<gpx><trk></gpx>");

        // Act
        let err = import_file_via(&data_root, &path, None, |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ImportGpxMalformedXml);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_idl0_source_routes_to_import_idl0_not_import_file() {
        // Arrange — core's own `import_file` refuses the extension "idl0"
        // (L2-R13); this proves the idl0 branch is actually taken instead.
        let data_root = temp_root();
        let path = temp_file("session.idl0", &minimal_idl0_buffer(16_384));
        let mut phases = Vec::new();

        // Act
        let outcome = import_file_via(&data_root, &path, None, |p| phases.push(p.to_string())).unwrap();

        // Assert
        assert_eq!(outcome.session.source_format, "idl0");
        assert_eq!(phases, vec!["reading", "decoding", "materializing"]);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_bad_magic_idl0_maps_to_parse_invalid_magic_bytes() {
        // Arrange
        let data_root = temp_root();
        let path = temp_file("bad.idl0", b"NOPE not an idl0 file at all, but long enough");

        // Act
        let err = import_file_via(&data_root, &path, None, |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ParseInvalidMagicBytes);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_unrecognised_extension_with_no_forced_id_is_invalid_argument() {
        // Arrange — `UnknownExtension` (core's own registry has no `.xyz`
        // importer) maps to the cross-cutting `InvalidArgument`, not a new
        // `IpcErrorKind` (L2's own brief-task6 anticipated this mapping).
        let data_root = temp_root();
        let path = temp_file("mystery.xyz", b"anything");

        // Act
        let err = import_file_via(&data_root, &path, None, |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn import_file_via_collision_maps_to_import_collision() {
        // Arrange — for `.idl0`, `session_id` comes from the header's own
        // 16-byte UUID field (`parse::v3`), not the blob hash — so two
        // buffers sharing one header but differing only in their sample data
        // share a `session_id` while hashing to different `blob_sha256`s,
        // the real-world condition this error exists for (C4 §3: a
        // truncated download and the full file share the device UUID).
        let data_root = temp_root();
        let path1 = temp_file("s1.idl0", &minimal_idl0_buffer(16_384));
        import_file_via(&data_root, &path1, None, |_| {}).unwrap();

        let path2 = temp_file("s1.idl0", &minimal_idl0_buffer(16_385));

        // Act
        let err = import_file_via(&data_root, &path2, None, |_| {}).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ImportCollision);

        let _ = std::fs::remove_dir_all(&data_root);
    }
}
