//! Maintenance group: quarantine review (C3 §3.2) and whole-`<data>`
//! verify/repair (C3 §3.10) — thin wrappers over the landed
//! `idl_rs::store::quarantine`/`idl_rs::store::verify` modules (Task 6, C4
//! §7, ruling R86).

use std::path::Path;

use idl_rs::store::quarantine::ResolveAction;
use idl_rs::store::verify::Severity;

use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// C3 §3.2 `QuarantineEntry` — mirrors
/// `idl_rs::store::quarantine::QuarantineEntry` field for field. Named the
/// same as the core type; call sites disambiguate with the full core path
/// (this module never `use`s the core type unqualified).
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuarantineEntry {
    pub entry_id: String,
    pub path: String,
    pub original_path: String,
    pub reason: String,
    pub quarantined_at_ms: i64,
}

impl From<idl_rs::store::quarantine::QuarantineEntry> for QuarantineEntry {
    fn from(e: idl_rs::store::quarantine::QuarantineEntry) -> Self {
        Self {
            entry_id: e.entry_id,
            path: e.path,
            original_path: e.original_path,
            reason: e.reason,
            quarantined_at_ms: e.quarantined_at_ms,
        }
    }
}

/// Maps a core [`Severity`] to the exact C3 §3.10 wire string. Explicit
/// match, never `Debug` formatting — a renamed/reordered core variant must
/// not silently change the wire shape.
fn severity_str(s: Severity) -> &'static str {
    match s {
        Severity::Info => "info",
        Severity::Warning => "warning",
        Severity::Error => "error",
    }
}

/// One C4 §7 [`idl_rs::store::verify::Finding`] as it crosses the wire.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyFinding {
    /// `"info" | "warning" | "error"` (C3 §3.10).
    pub severity: String,
    pub path: String,
    pub message: String,
}

impl From<idl_rs::store::verify::Finding> for VerifyFinding {
    fn from(f: idl_rs::store::verify::Finding) -> Self {
        Self { severity: severity_str(f.severity).to_string(), path: f.path.display().to_string(), message: f.message }
    }
}

/// C3 §3.10 `VerifyReport` — `verify_data_dir`'s return.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyReport {
    pub findings: Vec<VerifyFinding>,
    /// Empty unless `repair` was true.
    pub quarantined: Vec<QuarantineEntry>,
    pub elapsed_ms: u32,
}

/// `list_quarantine`'s transport-agnostic core.
fn list_quarantine_via(data_dir: &Path) -> Result<Vec<QuarantineEntry>, IpcError> {
    Ok(idl_rs::store::quarantine::list_quarantine(data_dir)?.into_iter().map(Into::into).collect())
}

/// `resolve_quarantine`'s transport-agnostic core. `action` maps
/// `"restore"`/`"discard"` to [`ResolveAction`]; anything else — including
/// the stub's former `"retry"` (ruling R86 Q2) — is `invalid_argument`
/// with the received value in `detail`, never silently aliased.
fn resolve_quarantine_via(data_dir: &Path, entry_id: &str, action: &str) -> Result<(), IpcError> {
    let action = match action {
        "restore" => ResolveAction::Restore,
        "discard" => ResolveAction::Discard,
        other => {
            return Err(IpcError::with_detail(
                IpcErrorKind::InvalidArgument,
                format!("resolve_quarantine: unknown action '{other}'"),
                serde_json::json!({ "action": other }),
            ))
        }
    };
    idl_rs::store::quarantine::resolve_quarantine(data_dir, entry_id, action)?;
    Ok(())
}

/// `verify_data_dir`'s transport-agnostic core. `ids`/`now_ms` are injected
/// (Task 6's `verify_and_repair` requires them — core mints neither uuids
/// nor clock reads on its own). `repair: false` never calls
/// `verify_and_repair`'s repair path — only the read-only `verify` — so no
/// write happens.
fn verify_data_dir_via(
    data_dir: &Path,
    repair: bool,
    ids: &mut dyn FnMut() -> String,
    now_ms: i64,
) -> Result<VerifyReport, IpcError> {
    let start = std::time::Instant::now();
    let (findings, quarantined) = if repair {
        idl_rs::store::verify::verify_and_repair(data_dir, ids, now_ms)
    } else {
        (idl_rs::store::verify::verify(data_dir), Vec::new())
    };
    Ok(VerifyReport {
        findings: findings.into_iter().map(Into::into).collect(),
        quarantined: quarantined.into_iter().map(Into::into).collect(),
        elapsed_ms: start.elapsed().as_millis() as u32,
    })
}

/// C3 §3.2 `list_quarantine()`. Thin over `store::quarantine::list_quarantine`.
#[tauri::command(async)]
pub fn list_quarantine(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<QuarantineEntry>, IpcError> {
    list_quarantine_via(&data_dir.0)
}

/// C3 §3.2 `resolve_quarantine(entry_id, action)`. `action` is
/// `"restore" | "discard"` — anything else, including the retired
/// `"retry"`, is `invalid_argument`.
#[tauri::command(async)]
pub fn resolve_quarantine(
    entry_id: String,
    action: String,
    data_dir: tauri::State<'_, DataDir>,
) -> Result<(), IpcError> {
    resolve_quarantine_via(&data_dir.0, &entry_id, &action)
}

/// C3 §3.10 `verify_data_dir(repair)`. `repair: false` is `store::verify`
/// unchanged (read-only); `repair: true` additionally runs the C4 §7
/// repair pass, minting a fresh uuid v4 per repair and stamping every
/// repair with the current wall-clock time.
#[tauri::command(async)]
pub fn verify_data_dir(repair: bool, data_dir: tauri::State<'_, DataDir>) -> Result<VerifyReport, IpcError> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let mut ids = || uuid::Uuid::new_v4().to_string();
    verify_data_dir_via(&data_dir.0, repair, &mut ids, now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::store::blob::{blob_exists, blob_path, write_blob};
    use idl_rs::store::quarantine::quarantine_file;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-maintenance-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn uuid_string() -> String {
        Uuid::new_v4().to_string()
    }

    #[test]
    fn list_quarantine_via_an_empty_directory_an_empty_list() {
        // Arrange
        let root = temp_root();

        // Act
        let entries = list_quarantine_via(&root).unwrap();

        // Assert
        assert!(entries.is_empty());
    }

    #[test]
    fn list_quarantine_via_two_entries_both_newest_first() {
        // Arrange
        let root = temp_root();
        let a = root.join("a.bin");
        let b = root.join("b.bin");
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(&b, b"b").unwrap();
        quarantine_file(&root, &a, "hash mismatch", &uuid_string(), 100).unwrap();
        quarantine_file(&root, &b, "hash mismatch", &uuid_string(), 200).unwrap();

        // Act
        let entries = list_quarantine_via(&root).unwrap();

        // Assert
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].quarantined_at_ms, 200);
        assert_eq!(entries[1].quarantined_at_ms, 100);
    }

    #[test]
    fn resolve_quarantine_via_restore_the_file_is_back() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("c".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"payload").unwrap();
        let entry_id = uuid_string();
        quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();

        // Act
        resolve_quarantine_via(&root, &entry_id, "restore").unwrap();

        // Assert
        assert_eq!(std::fs::read(&source).unwrap(), b"payload");
        assert!(list_quarantine_via(&root).unwrap().is_empty());
    }

    #[test]
    fn resolve_quarantine_via_discard_both_files_are_gone() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("d".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"payload").unwrap();
        let entry_id = uuid_string();
        let entry = quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();

        // Act
        resolve_quarantine_via(&root, &entry_id, "discard").unwrap();

        // Assert
        assert!(!std::path::Path::new(&entry.path).exists());
        assert!(!source.exists());
        assert!(list_quarantine_via(&root).unwrap().is_empty());
    }

    #[test]
    fn resolve_quarantine_via_retry_invalid_argument() {
        // Arrange — locks Q2 in: "retry" is the retired stub action, never
        // silently aliased to "restore".
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("e".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"payload").unwrap();
        let entry_id = uuid_string();
        quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();

        // Act
        let err = resolve_quarantine_via(&root, &entry_id, "retry").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
    }

    #[test]
    fn resolve_quarantine_via_an_unknown_entry_id_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = resolve_quarantine_via(&root, &uuid_string(), "discard").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
    }

    #[test]
    fn resolve_quarantine_via_restore_onto_an_occupied_path_invalid_argument() {
        // Arrange
        let root = temp_root();
        let source = root.join("blobs").join("sha256").join("ab").join("f".repeat(62));
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"original").unwrap();
        let entry_id = uuid_string();
        quarantine_file(&root, &source, "hash mismatch", &entry_id, 1).unwrap();
        std::fs::write(&source, b"something else now lives here").unwrap();

        // Act
        let err = resolve_quarantine_via(&root, &entry_id, "restore").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
    }

    fn fixed_ids(seed: &'static str) -> impl FnMut() -> String {
        let mut n = 0;
        move || {
            n += 1;
            format!("{seed}{n:0>35}")
        }
    }

    #[test]
    fn verify_data_dir_via_repair_false_on_a_corrupt_blob_the_finding_is_reported_quarantined_is_empty_and_the_blob_is_untouched(
    ) {
        // Arrange
        let root = temp_root();
        let digest = write_blob(&root, b"original").unwrap();
        let path = blob_path(&root, &digest);
        std::fs::write(&path, b"corrupted").unwrap();
        let mut ids = fixed_ids("1");

        // Act
        let report = verify_data_dir_via(&root, false, &mut ids, 1).unwrap();

        // Assert
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].severity, "error");
        assert!(report.quarantined.is_empty());
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), b"corrupted");
    }

    #[test]
    fn verify_data_dir_via_repair_true_on_the_same_tree_the_blob_is_quarantined_and_appears_in_both_quarantined_and_list_quarantine(
    ) {
        // Arrange
        let root = temp_root();
        let digest = write_blob(&root, b"original").unwrap();
        let path = blob_path(&root, &digest);
        std::fs::write(&path, b"corrupted").unwrap();
        let mut ids = fixed_ids("2");

        // Act
        let report = verify_data_dir_via(&root, true, &mut ids, 42).unwrap();

        // Assert
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.quarantined.len(), 1);
        assert!(!blob_exists(&root, &digest));
        let listed = list_quarantine_via(&root).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].entry_id, report.quarantined[0].entry_id);
    }

    #[test]
    fn verify_data_dir_via_a_healthy_tree_no_error_severity_findings() {
        // Arrange
        let root = temp_root();
        write_blob(&root, b"clean blob").unwrap();
        let mut ids = fixed_ids("3");

        // Act
        let report = verify_data_dir_via(&root, false, &mut ids, 1).unwrap();

        // Assert
        assert!(!report.findings.iter().any(|f| f.severity == "error"));
    }

    #[test]
    fn verify_data_dir_via_serialised_verify_report_severity_strings_match_c3() {
        // Arrange
        let report = VerifyReport {
            findings: vec![
                VerifyFinding { severity: "info".to_string(), path: "p".to_string(), message: "m".to_string() },
                VerifyFinding { severity: "warning".to_string(), path: "p".to_string(), message: "m".to_string() },
                VerifyFinding { severity: "error".to_string(), path: "p".to_string(), message: "m".to_string() },
            ],
            quarantined: Vec::new(),
            elapsed_ms: 5,
        };

        // Act
        let json = serde_json::to_value(&report).unwrap();

        // Assert
        assert_eq!(json["findings"][0]["severity"], serde_json::json!("info"));
        assert_eq!(json["findings"][1]["severity"], serde_json::json!("warning"));
        assert_eq!(json["findings"][2]["severity"], serde_json::json!("error"));
        assert_eq!(json["elapsed_ms"], serde_json::json!(5));
    }
}
