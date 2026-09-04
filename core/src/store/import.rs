//! The `.idl0` import pipeline (blob write → parse → base-channel synthesis
//! → `data.parquet` → `session.json`), ruling R18: this belongs in core, not
//! the CLI, so both the CLI and (later) L5's Tauri `import_file` command
//! share one implementation. Does **not** touch the catalog — refreshing it
//! is the caller's job (C4 §5's rebuild is cheap and idempotent; incremental
//! catalog indexing does not exist yet).

use std::fmt;
use std::path::{Path, PathBuf};

use crate::session::ParseError;
use crate::store::blob::{self, BlobStoreError};
use crate::store::parquet::{read_session_metadata, write_session_parquet, ParquetStoreError, SessionParquetMetadata};
use crate::store::session_json::{empty_session_json, write_session_json, SessionJsonError};

/// What [`plan_import`] decided to do, and (via [`ImportReport::plan`]) what
/// [`import_idl0`] actually did — `Collision` never appears in a report; that
/// arm returns [`ImportError`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportPlan {
    /// No `data.parquet` exists yet for this session — write one.
    Write,
    /// `data.parquet` already reflects this exact blob under the
    /// currently-running build's `importer_version`/`seam_correction_version`
    /// — idempotent re-import, nothing to do (C4 §3).
    Skip,
    /// `data.parquet` reflects this blob but was written by an older
    /// `importer_version` or `seam_correction_version` — delete and rewrite
    /// (C1 §4.3's regeneration rule).
    Regenerate,
    /// `data.parquet` exists for this `session_id` but under a *different*
    /// blob — refuse to overwrite (real for `.idl0`: a truncated download
    /// and the full file share the device UUID, C4 §3).
    Collision { existing_blob_sha256: String },
}

/// Decides what an import should do to `data.parquet`, given what's already
/// on disk (`None` if this session has never been imported). Pure — no I/O.
///
/// - no existing metadata → [`ImportPlan::Write`].
/// - existing blob matches `new_blob_sha256` and both versions match the
///   running build's → [`ImportPlan::Skip`].
/// - existing blob matches but either version differs → [`ImportPlan::Regenerate`].
/// - existing blob differs from `new_blob_sha256` → [`ImportPlan::Collision`].
pub fn plan_import(
    existing: Option<&SessionParquetMetadata>,
    new_blob_sha256: &str,
    importer_version: &str,
    seam_correction_version: &str,
) -> ImportPlan {
    match existing {
        None => ImportPlan::Write,
        Some(meta) if meta.blob_sha256 == new_blob_sha256 => {
            if meta.importer_version == importer_version && meta.seam_correction_version == seam_correction_version {
                ImportPlan::Skip
            } else {
                ImportPlan::Regenerate
            }
        }
        Some(meta) => ImportPlan::Collision { existing_blob_sha256: meta.blob_sha256.clone() },
    }
}

/// Discriminant for [`ImportError`]. Named after C3 §2's `import_*` set
/// where an error class corresponds to one of its kinds, plus `Collision`
/// (new — this pipeline's own re-import refusal, not yet in C3 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportErrorKind {
    /// A filesystem or Arrow/Parquet I/O operation failed, or an on-disk
    /// file this pipeline itself wrote no longer parses as its own schema
    /// (data corruption, not user input) — C3 §2's cross-cutting `io` kind.
    Io,
    /// Mirrors `ParseError::InvalidMagicBytes` (C3 §2 `parse_invalid_magic_bytes`).
    ParseInvalidMagicBytes,
    /// Mirrors `ParseError::UnsupportedSchemaVersion` (C3 §2 `parse_unsupported_schema_version`).
    ParseUnsupportedSchemaVersion,
    /// Mirrors `ParseError::TruncatedRecord` (C3 §2 `parse_truncated_record`) —
    /// only when the buffer is too short to read even the magic/schema bytes;
    /// a log that parses but ends mid-record instead surfaces via
    /// [`ImportReport::truncation_warning`], not this error.
    ParseTruncatedRecord,
    /// [`plan_import`] returned [`ImportPlan::Collision`] — refusing to
    /// overwrite a `data.parquet` written from a different blob.
    Collision,
}

/// Error from [`import_idl0`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportError {
    pub kind: ImportErrorKind,
    pub message: String,
}

impl ImportError {
    fn new(kind: ImportErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ImportError {}

impl From<BlobStoreError> for ImportError {
    fn from(e: BlobStoreError) -> Self {
        ImportError::new(ImportErrorKind::Io, e.to_string())
    }
}

impl From<ParseError> for ImportError {
    fn from(e: ParseError) -> Self {
        let kind = match &e {
            ParseError::InvalidMagicBytes(_) => ImportErrorKind::ParseInvalidMagicBytes,
            ParseError::UnsupportedSchemaVersion(_) => ImportErrorKind::ParseUnsupportedSchemaVersion,
            ParseError::TruncatedRecord(_) => ImportErrorKind::ParseTruncatedRecord,
            ParseError::Io(_) => ImportErrorKind::Io,
        };
        ImportError::new(kind, e.to_string())
    }
}

impl From<ParquetStoreError> for ImportError {
    fn from(e: ParquetStoreError) -> Self {
        ImportError::new(ImportErrorKind::Io, e.to_string())
    }
}

impl From<SessionJsonError> for ImportError {
    fn from(e: SessionJsonError) -> Self {
        ImportError::new(ImportErrorKind::Io, e.to_string())
    }
}

/// Outcome of one successful [`import_idl0`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    pub session_id: String,
    pub blob_sha256: String,
    pub data_parquet: PathBuf,
    /// What actually happened. Never [`ImportPlan::Collision`] — that arm
    /// returns [`ImportError`] instead of a report.
    pub plan: ImportPlan,
    /// `true` if this call created `session.json` (it never overwrites an
    /// existing one).
    pub session_json_created: bool,
    /// `Some` when the `.idl0` buffer ended mid-record (C1 §8's "recover
    /// what's readable" rule) — the underlying [`ParseError::TruncatedRecord`]'s
    /// message, not treated as a hard failure.
    pub truncation_warning: Option<String>,
}

/// Imports one `.idl0` buffer into `data_root` (contract C4 §2 layout):
/// writes the raw bytes to the CAS blob store, parses them, synthesizes the
/// base `Time`/`Distance` channels, decides via [`plan_import`] whether
/// `data.parquet` needs writing/skipping/regenerating, and creates an empty
/// `session.json` if none exists yet (never overwriting one that does).
///
/// Does not touch the catalog — the caller rebuilds/refreshes it afterward.
pub fn import_idl0(data_root: &Path, bytes: &[u8]) -> Result<ImportReport, ImportError> {
    let blob_sha256 = blob::write_blob(data_root, bytes)?;

    let mut result = crate::parse::parse(bytes)?;
    result.session.blob_sha256 = blob_sha256.clone();
    crate::session::synthesis::synthesize_base_channels(&mut result.session);

    let session_id = result.session.session_id.clone();
    let data_parquet_path = data_root.join("sessions").join(&session_id).join("data.parquet");
    let existing = if data_parquet_path.is_file() { Some(read_session_metadata(&data_parquet_path)?) } else { None };

    let plan = plan_import(
        existing.as_ref(),
        &blob_sha256,
        crate::parse::IDL0_IMPORTER_VERSION,
        crate::session::seam_correction::SEAM_CORRECTION_VERSION,
    );

    let executed_plan = match plan {
        ImportPlan::Collision { existing_blob_sha256 } => {
            return Err(ImportError::new(
                ImportErrorKind::Collision,
                format!(
                    "session {session_id} already has data.parquet for blob {existing_blob_sha256}, \
                     this import is blob {blob_sha256} — refusing to overwrite. \
                     Remove sessions/{session_id}/ to re-import."
                ),
            ));
        }
        ImportPlan::Write => {
            write_session_parquet(data_root, &result.session, crate::parse::IDL0_IMPORTER_VERSION)?;
            ImportPlan::Write
        }
        ImportPlan::Regenerate => {
            // C1 §4.3's regeneration rule: delete then rewrite. Explicit
            // because `parquet::write_session_parquet` writes via
            // `write_atomic(.., None)` by design — `data.parquet` is
            // write-once, so a stale file must be removed first (R16's
            // audit table). `derived/` is deliberately left untouched: its
            // hashes include input column hashes, so stale files simply
            // become orphans, not silently-wrong data.
            std::fs::remove_file(&data_parquet_path)
                .map_err(|e| ImportError::new(ImportErrorKind::Io, format!("removing stale {}: {e}", data_parquet_path.display())))?;
            write_session_parquet(data_root, &result.session, crate::parse::IDL0_IMPORTER_VERSION)?;
            ImportPlan::Regenerate
        }
        ImportPlan::Skip => ImportPlan::Skip,
    };

    let sj_path = data_root.join("sessions").join(&session_id).join("session.json");
    let session_json_created = if !sj_path.is_file() {
        let doc = empty_session_json(&session_id);
        write_session_json(data_root, &session_id, &doc, None)?;
        true
    } else {
        false
    };

    Ok(ImportReport {
        session_id,
        blob_sha256,
        data_parquet: data_parquet_path,
        plan: executed_plan,
        session_json_created,
        truncation_warning: result.truncation_warning.map(|w| w.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::test_buffers::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn meta(blob_sha256: &str, importer_version: &str, seam_correction_version: &str) -> SessionParquetMetadata {
        SessionParquetMetadata {
            session_id: "s1".to_string(),
            timestamp_utc_ms: 0,
            device_id: None,
            config_checksum: None,
            blob_sha256: blob_sha256.to_string(),
            source_format: "idl0".to_string(),
            importer_version: importer_version.to_string(),
            engine_version: "0.1.0".to_string(),
            seam_correction_version: seam_correction_version.to_string(),
        }
    }

    #[test]
    fn plan_import_no_existing_metadata_writes() {
        // Arrange / Act
        let plan = plan_import(None, "abc", "0.1.0", "v1");

        // Assert
        assert_eq!(plan, ImportPlan::Write);
    }

    #[test]
    fn plan_import_same_blob_and_versions_skips() {
        // Arrange
        let existing = meta("abc", "0.1.0", "v1");

        // Act
        let plan = plan_import(Some(&existing), "abc", "0.1.0", "v1");

        // Assert
        assert_eq!(plan, ImportPlan::Skip);
    }

    #[test]
    fn plan_import_same_blob_different_importer_version_regenerates() {
        // Arrange
        let existing = meta("abc", "0.0.1", "v1");

        // Act
        let plan = plan_import(Some(&existing), "abc", "0.1.0", "v1");

        // Assert
        assert_eq!(plan, ImportPlan::Regenerate);
    }

    #[test]
    fn plan_import_same_blob_different_seam_correction_version_regenerates() {
        // Arrange
        let existing = meta("abc", "0.1.0", "v0");

        // Act
        let plan = plan_import(Some(&existing), "abc", "0.1.0", "v1");

        // Assert
        assert_eq!(plan, ImportPlan::Regenerate);
    }

    #[test]
    fn plan_import_different_blob_collides() {
        // Arrange
        let existing = meta("abc", "0.1.0", "v1");

        // Act
        let plan = plan_import(Some(&existing), "def", "0.1.0", "v1");

        // Assert
        assert_eq!(plan, ImportPlan::Collision { existing_blob_sha256: "abc".to_string() });
    }

    /// One minimal, valid `.idl0` v3 buffer: one IMU sample, session UUID
    /// all-`0xAB` (matches `Header::default()` — every buffer built with the
    /// same UUID bytes belongs to the same `session_id`, matching real
    /// devices where the UUID never extends, C4 §3).
    fn synthetic_idl0_bytes() -> Vec<u8> {
        let accel: f32 = 32.0 / 32768.0;
        cat(&[
            Header { schema_version: 3, imu_mask: 0x3F, ..Default::default() }
                .build(&v3_imu_axes_registry(0, 0, 800, accel, 500.0)),
            frame(0x01, &imu_payload(0, 0, &[16384, 0, 0, 0, 0, 0])),
            frame(0x01, &imu_payload(0, 1250, &[16384, 0, 0, 0, 0, 0])),
            session_end(),
        ])
    }

    #[test]
    fn import_idl0_fresh_root_writes_parquet_and_session_json() {
        // Arrange
        let root = temp_root();
        let bytes = synthetic_idl0_bytes();

        // Act
        let report = import_idl0(&root, &bytes).unwrap();

        // Assert
        assert_eq!(report.plan, ImportPlan::Write);
        assert!(report.session_json_created);
        assert!(report.data_parquet.is_file());
        let sj_path = root.join("sessions").join(&report.session_id).join("session.json");
        assert!(sj_path.is_file());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_idl0_same_bytes_twice_skips_and_leaves_session_json_untouched() {
        // Arrange
        let root = temp_root();
        let bytes = synthetic_idl0_bytes();
        let first = import_idl0(&root, &bytes).unwrap();
        let sj_path = root.join("sessions").join(&first.session_id).join("session.json");
        let sj_bytes_before = std::fs::read(&sj_path).unwrap();

        // Act
        let second = import_idl0(&root, &bytes).unwrap();

        // Assert
        assert_eq!(second.plan, ImportPlan::Skip);
        assert!(!second.session_json_created);
        let sj_bytes_after = std::fs::read(&sj_path).unwrap();
        assert_eq!(sj_bytes_before, sj_bytes_after);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_idl0_stale_importer_version_regenerates() {
        // Arrange — a `data.parquet` for this same blob already exists, but
        // written by an older importer_version than the running build's.
        let root = temp_root();
        let bytes = synthetic_idl0_bytes();
        let mut parsed = crate::parse::parse(&bytes).unwrap();
        let blob_sha256 = blob::write_blob(&root, &bytes).unwrap();
        parsed.session.blob_sha256 = blob_sha256;
        crate::session::synthesis::synthesize_base_channels(&mut parsed.session);
        write_session_parquet(&root, &parsed.session, "0.0.1").unwrap();

        // Act
        let report = import_idl0(&root, &bytes).unwrap();

        // Assert
        assert_eq!(report.plan, ImportPlan::Regenerate);
        let meta = read_session_metadata(&report.data_parquet).unwrap();
        assert_eq!(meta.importer_version, crate::parse::IDL0_IMPORTER_VERSION);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Two buffers sharing the same session UUID (`Header::default()`'s
    /// `uuid`) but different record payloads — different blob hashes, same
    /// `session_id`. Real for `.idl0`: a truncated download and the full
    /// file share the device UUID (C4 §3 — the id never extends).
    #[test]
    fn import_idl0_different_blob_same_session_id_collides_and_leaves_first_parquet_untouched() {
        // Arrange
        let root = temp_root();
        let bytes_a = synthetic_idl0_bytes();
        let accel: f32 = 32.0 / 32768.0;
        let bytes_b = cat(&[
            Header { schema_version: 3, imu_mask: 0x3F, ..Default::default() }
                .build(&v3_imu_axes_registry(0, 0, 800, accel, 500.0)),
            frame(0x01, &imu_payload(0, 0, &[1, 0, 0, 0, 0, 0])), // different sample value -> different bytes
            session_end(),
        ]);
        assert_ne!(bytes_a, bytes_b, "fixtures must actually differ to exercise a real collision");

        let first = import_idl0(&root, &bytes_a).unwrap();
        let parquet_bytes_before = std::fs::read(&first.data_parquet).unwrap();

        // Act
        let err = import_idl0(&root, &bytes_b).unwrap_err();

        // Assert
        assert_eq!(err.kind, ImportErrorKind::Collision);
        let parquet_bytes_after = std::fs::read(&first.data_parquet).unwrap();
        assert_eq!(parquet_bytes_before, parquet_bytes_after);

        let _ = std::fs::remove_dir_all(&root);
    }
}
