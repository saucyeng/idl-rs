//! The `.idl0` import pipeline (blob write → parse → base-channel synthesis
//! → `data.parquet` → `session.json`), ruling R18: this belongs in core, not
//! the CLI, so both the CLI and (later) L5's Tauri `import_file` command
//! share one implementation. Also updates `catalog.sqlite` for this one
//! session, incrementally, when a catalog already exists (C4 §5's
//! `index_session`, task L2b T4) — it never creates a catalog itself; a
//! bare data root stays catalog-less until something rebuilds one.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::import::hook::PostImportHook;
use crate::import::ImporterError;
use crate::session::handle::SessionHandle;
use crate::session::{ParseError, Session};
use crate::store::blob::{self, BlobStoreError};
use crate::store::catalog::{index_session, open_catalog};
use crate::store::lap_index::{index_laps, LapIndexReport};
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
    /// Mirrors `ImporterError::FitMalformed` (C3 §2 `import_fit_malformed`).
    ImportFitMalformed,
    /// Mirrors `ImporterError::GpxMalformedXml` (C3 §2 `import_gpx_malformed_xml`).
    ImportGpxMalformedXml,
    /// Mirrors `ImporterError::GpxNoTrackpoints` (C3 §2 `import_gpx_no_trackpoints`).
    ImportGpxNoTrackpoints,
    /// Mirrors `ImporterError::GpxMissingLatLon` (C3 §2 `import_gpx_missing_lat_lon`).
    ImportGpxMissingLatLon,
    /// Mirrors `ImporterError::GpxUnparseableLatLon` (C3 §2 `import_gpx_unparseable_lat_lon`).
    ImportGpxUnparseableLatLon,
    /// Mirrors `ImporterError::CsvMalformed` (C3 §2 `import_csv_malformed`).
    ImportCsvMalformed,
    /// Mirrors `ImporterError::NotUtf8` (C3 §2 `import_not_utf8`).
    ImportNotUtf8,
    /// [`crate::import::importer_for_extension`] returned `None` — no
    /// importer covers this file extension. Core-internal: no C3 §2 row of
    /// its own (L5's Tauri layer maps this to C3's cross-cutting
    /// `invalid_argument` at that layer, not this one).
    UnknownExtension,
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

impl From<ImporterError> for ImportError {
    fn from(e: ImporterError) -> Self {
        let kind = match &e {
            ImporterError::FitMalformed(_) => ImportErrorKind::ImportFitMalformed,
            ImporterError::GpxMalformedXml(_) => ImportErrorKind::ImportGpxMalformedXml,
            ImporterError::GpxNoTrackpoints => ImportErrorKind::ImportGpxNoTrackpoints,
            ImporterError::GpxMissingLatLon(_) => ImportErrorKind::ImportGpxMissingLatLon,
            ImporterError::GpxUnparseableLatLon(_) => ImportErrorKind::ImportGpxUnparseableLatLon,
            ImporterError::CsvMalformed(_) => ImportErrorKind::ImportCsvMalformed,
            ImporterError::NotUtf8(_) => ImportErrorKind::ImportNotUtf8,
        };
        ImportError::new(kind, e.to_string())
    }
}

/// What [`import_idl0`] actually did to `data.parquet` on a successful call
/// — unlike [`ImportPlan`], this has no `Collision` arm: a collision returns
/// [`ImportError`] instead of an [`ImportReport`], so the impossible state
/// (a report claiming `Collision`) is unrepresentable rather than merely
/// documented-unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportOutcome {
    /// No `data.parquet` existed yet — one was written.
    Written,
    /// `data.parquet` already matched this blob and build — nothing written.
    Skipped,
    /// `data.parquet` matched this blob but an older build — deleted and rewritten.
    Regenerated,
}

/// Outcome of one successful [`import_idl0`] call. Not `Clone`/`PartialEq` —
/// [`LapIndexReport`] carries neither, and nothing needs to compare or
/// duplicate a whole report (tests compare individual fields instead).
#[derive(Debug)]
pub struct ImportReport {
    pub session_id: String,
    pub blob_sha256: String,
    pub data_parquet: PathBuf,
    /// What actually happened.
    pub outcome: ImportOutcome,
    /// `true` if this call created `session.json` (it never overwrites an
    /// existing one).
    pub session_json_created: bool,
    /// `Some` when the `.idl0` buffer ended mid-record (C1 §8's "recover
    /// what's readable" rule) — the underlying [`ParseError::TruncatedRecord`]'s
    /// message, not treated as a hard failure.
    pub truncation_warning: Option<String>,
    /// Non-fatal advisory messages raised while importing — from
    /// `crate::session::ParseResult::import_warnings` on the `.idl0` path,
    /// or the importer's own `crate::import::ImporterWarning`s on the
    /// [`import_file`] path. Never silently dropped (G0.6).
    pub import_warnings: Vec<String>,
    /// Outcome of the import-time lap-index step (IDL0_SPEC §17.4), run on
    /// every plan this function reaches (`Write`/`Regenerate`/`Skip` — not
    /// `Collision`, which never produces a report at all). `None` only when
    /// the step itself returned `Err`; see [`Self::lap_index_warning`].
    pub lap_index: Option<LapIndexReport>,
    /// Set to [`crate::store::lap_index::LapIndexError`]'s `Display` when the
    /// lap-index step failed; the import itself still succeeds (this field
    /// and [`Self::lap_index`] are never both `Some`). Recoverable by
    /// `idl-rs rescan`.
    pub lap_index_warning: Option<String>,
    /// Set to [`crate::store::catalog::CatalogError`]'s `Display` when
    /// `catalog.sqlite` exists but [`crate::store::catalog::index_session`]
    /// failed for this session (C4 §5, task L2b T4) — the import itself
    /// still succeeds; the catalog just falls behind until the next
    /// successful import or a `rebuild_catalog`. Always `None` when no
    /// catalog exists yet, since this pipeline never creates one.
    pub catalog_index_warning: Option<String>,
}

/// Imports one `.idl0` buffer into `data_root` (contract C4 §2 layout):
/// writes the raw bytes to the CAS blob store, parses them, synthesizes the
/// base `Time`/`Distance` channels, decides via [`plan_import`] whether
/// `data.parquet` needs writing/skipping/regenerating, and creates an empty
/// `session.json` if none exists yet (never overwriting one that does).
///
/// Does not touch the catalog — the caller rebuilds/refreshes it afterward.
pub fn import_idl0(data_root: &Path, bytes: &[u8]) -> Result<ImportReport, ImportError> {
    // Parse first — an unparseable `.idl0` file must leave nothing in the
    // CAS (the hash is computed from `bytes` directly, independent of
    // parsing, so ordering here costs nothing on the success path but
    // avoids a permanent orphan blob on every failed import).
    let mut result = crate::parse::parse(bytes)?;
    let blob_sha256 = blob::write_blob(data_root, bytes)?;
    result.session.blob_sha256 = blob_sha256.clone();
    crate::session::synthesis::synthesize_base_channels(&mut result.session);

    let truncation_warning = result.truncation_warning.map(|w| w.to_string());
    let import_warnings = result.import_warnings.iter().map(|w| w.message.clone()).collect();

    let mut report = finish_import(
        data_root,
        result.session,
        blob_sha256,
        crate::parse::IDL0_IMPORTER_VERSION,
        import_warnings,
    )?;
    report.truncation_warning = truncation_warning;
    Ok(report)
}

/// Imports one non-`.idl0` source buffer (FIT/GPX/CSV, ledger R23 L2-R13)
/// into `data_root`, generalising [`import_idl0`]'s blob/parquet/
/// `session.json` pipeline to any format [`crate::import::importer_for_extension`]
/// covers. Does not handle `"idl0"` — that stays [`import_idl0`]'s own entry
/// point, called separately by whoever routes `.idl0` files.
///
/// `extension` is a lowercase file extension without the dot (e.g. `"gpx"`).
/// Preserves the R18-addendum write-ordering invariant: `bytes` is hashed
/// and parsed before anything is written to the CAS, so a malformed buffer
/// leaves nothing behind. Runs [`crate::session::synthesis::synthesize_base_channels`]
/// on the parsed session (this is what makes the `Time` synthesis fallback,
/// ledger R23 Q2, actually reach FIT/GPX/CSV sessions) and, on success,
/// [`crate::import::hook::NoopPostImportHook::on_imported`] — the extension
/// point a real materialisation hook replaces at a future task; this
/// function itself takes no hook parameter.
pub fn import_file(data_root: &Path, extension: &str, bytes: &[u8]) -> Result<ImportReport, ImportError> {
    let Some(importer) = crate::import::importer_for_extension(extension) else {
        return Err(ImportError::new(
            ImportErrorKind::UnknownExtension,
            format!("no importer covers file extension \"{extension}\""),
        ));
    };

    // Hashed before parsing — `Importer::import` takes the digest as an
    // argument (to derive `session_id`) and does not write to the CAS
    // itself. One extra hash beyond what `import_idl0` needs (that path
    // only hashes once, inside `write_blob`, after parsing) — an accepted,
    // documented cost of this trait's pure signature.
    let blob_sha256 = crate::store::atomic::sha256_hex(bytes);

    // On `Err`, return immediately — nothing has been written to the CAS
    // yet (ordering preserved, R18 addendum).
    let mut outcome = importer.import(bytes, &blob_sha256)?;
    crate::session::synthesis::synthesize_base_channels(&mut outcome.session);
    crate::import::hook::NoopPostImportHook.on_imported(&outcome.session);

    // Now writes; its returned digest is guaranteed identical to the one
    // computed above (same bytes, same hash function) — use it as the
    // canonical blob_sha256 from here on.
    let blob_sha256 = blob::write_blob(data_root, bytes)?;
    outcome.session.blob_sha256 = blob_sha256.clone();

    let warnings = outcome.warnings.into_iter().map(|w| w.message).collect();

    finish_import(data_root, outcome.session, blob_sha256, importer.importer_version(), warnings)
}

/// Shared tail of [`import_idl0`] and [`import_file`]: decides via
/// [`plan_import`] whether `data.parquet` needs writing/skipping/
/// regenerating, writes it accordingly, and creates an empty `session.json`
/// if none exists yet (never overwriting one that does). `warnings` becomes
/// [`ImportReport::import_warnings`] verbatim; [`ImportReport::truncation_warning`]
/// is left `None` here — only [`import_idl0`] ever sets it.
fn finish_import(
    data_root: &Path,
    session: Session,
    blob_sha256: String,
    importer_version: &str,
    warnings: Vec<String>,
) -> Result<ImportReport, ImportError> {
    let session_id = session.session_id.clone();
    let data_parquet_path = data_root.join("sessions").join(&session_id).join("data.parquet");
    let existing = if data_parquet_path.is_file() { Some(read_session_metadata(&data_parquet_path)?) } else { None };

    let plan = plan_import(
        existing.as_ref(),
        &blob_sha256,
        importer_version,
        crate::session::seam_correction::SEAM_CORRECTION_VERSION,
    );

    let outcome = match plan {
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
            write_session_parquet(data_root, &session, importer_version)?;
            ImportOutcome::Written
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
            write_session_parquet(data_root, &session, importer_version)?;
            ImportOutcome::Regenerated
        }
        ImportPlan::Skip => ImportOutcome::Skipped,
    };

    let sj_path = data_root.join("sessions").join(&session_id).join("session.json");
    let session_json_created = if !sj_path.is_file() {
        let doc = empty_session_json(&session_id);
        write_session_json(data_root, &session_id, &doc, None)?;
        true
    } else {
        false
    };

    // Lap indexing (IDL0_SPEC §17.4), non-fatal (mirrors idl0's
    // `_detectAndSaveVisits`): a failure here must not fail the import, and
    // recovers on the next import or an explicit `rescan`. Runs with
    // `force = false` on every plan reaching this point (`Collision` already
    // returned above) — on `Skip` this costs one hash + one read and lets a
    // session imported before this lane pick up laps on its next import.
    // Building the handle from `session` (moved, not cloned) is why this is
    // the last thing this function does with it — the sessions this pipeline
    // imports are hundreds of MB, so a clone here is not an option.
    let handle = SessionHandle::from_session(session);
    let (lap_index, lap_index_warning) = match index_laps(data_root, &session_id, &handle, false) {
        Ok(report) => (Some(report), None),
        Err(e) => (None, Some(e.to_string())),
    };

    // Incremental catalog update (C4 §5, task L2b T4), non-fatal like the
    // lap-index step above: only when `catalog.sqlite` already exists —
    // this pipeline never creates one (a bare `<data>` stays catalog-less
    // until something runs `rebuild_catalog`, per the catalog's own
    // "deletable, rebuildable, never synced" contract). A missing catalog
    // is therefore not a warning either; it's simply not this function's
    // concern.
    let catalog_path = data_root.join("catalog.sqlite");
    let catalog_index_warning = if catalog_path.is_file() {
        match open_catalog(&catalog_path).and_then(|conn| index_session(&conn, data_root, &session_id)) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    } else {
        None
    };

    Ok(ImportReport {
        session_id,
        blob_sha256,
        data_parquet: data_parquet_path,
        outcome,
        session_json_created,
        truncation_warning: None,
        import_warnings: warnings,
        lap_index,
        lap_index_warning,
        catalog_index_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::GpsFix;
    use crate::laps::model::{Gate, LapTiming};
    use crate::parse::test_buffers::*;
    use crate::store::parquet::read_session_parquet;
    use crate::store::session_json::read_session_json;
    use crate::track_artifact::{write_track, Track};
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

    #[test]
    fn import_idl0_bad_magic_bytes_fails_and_leaves_no_orphan_blob() {
        // Arrange — a buffer that fails `parse::parse` before any session
        // work happens; its would-be blob digest must never land in the CAS.
        let root = temp_root();
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let would_be_digest = crate::store::atomic::sha256_hex(&bytes);

        // Act
        let result = import_idl0(&root, &bytes);

        // Assert
        assert!(matches!(
            result,
            Err(ImportError { kind: ImportErrorKind::ParseInvalidMagicBytes, .. })
        ));
        assert!(!crate::store::blob::blob_exists(&root, &would_be_digest));
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
        assert_eq!(report.outcome, ImportOutcome::Written);
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
        assert_eq!(second.outcome, ImportOutcome::Skipped);
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
        assert_eq!(report.outcome, ImportOutcome::Regenerated);
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

    #[test]
    fn import_idl0_populates_import_warnings_from_parse_result() {
        // Arrange — direct unit test on the new plumbing (G0.6): a
        // `ParseResult` with a non-empty `import_warnings` must survive
        // into the returned `ImportReport`, not be silently dropped as it
        // was before this task.
        let root = temp_root();
        let bytes = synthetic_idl0_bytes();
        let mut parsed = crate::parse::parse(&bytes).unwrap();
        parsed.import_warnings.push(crate::session::ImportWarning {
            kind: crate::session::ImportWarningKind::NonPositiveEffectivePeriod,
            message: "a burst-seam correction warning".to_string(),
        });
        let blob_sha256 = blob::write_blob(&root, &bytes).unwrap();
        parsed.session.blob_sha256 = blob_sha256.clone();
        crate::session::synthesis::synthesize_base_channels(&mut parsed.session);
        let import_warnings: Vec<String> = parsed.import_warnings.iter().map(|w| w.message.clone()).collect();

        // Act
        let report = finish_import(&root, parsed.session, blob_sha256, crate::parse::IDL0_IMPORTER_VERSION, import_warnings).unwrap();

        // Assert
        assert_eq!(report.import_warnings, vec!["a burst-seam correction warning".to_string()]);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A minimal, valid GPX buffer: three trackpoints, the last two sharing
    /// a `<time>` so the importer drops one with a warning (L2-R7's
    /// duplicate/non-monotonic rule) — proves `import_file` plumbs
    /// `ImporterWarning`s into `ImportReport.import_warnings`, not just the
    /// isolated importer call.
    fn minimal_gpx_with_warning() -> Vec<u8> {
        r#"<gpx><trk><trkseg>
            <trkpt lat="1.0" lon="2.0"><time>2026-01-01T00:00:00Z</time></trkpt>
            <trkpt lat="1.1" lon="2.1"><time>2026-01-01T00:00:01Z</time></trkpt>
            <trkpt lat="1.2" lon="2.2"><time>2026-01-01T00:00:01Z</time></trkpt>
        </trkseg></trk></gpx>"#
            .as_bytes()
            .to_vec()
    }

    /// FIT CRC-16 — same table-driven algorithm `import::fit`'s own test
    /// module uses; duplicated here rather than reached into (Task 5's
    /// `fit.rs` is under review, out of this task's scope to touch or
    /// import test-only items from).
    fn fit_crc16(data: &[u8]) -> u16 {
        const TABLE: [u16; 16] = [
            0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401, 0xA001, 0x6C00, 0x7800, 0xB401, 0x5000,
            0x9C01, 0x8801, 0x4400,
        ];
        let mut crc: u16 = 0;
        for &byte in data {
            let tmp = TABLE[(crc & 0xF) as usize];
            crc = (crc >> 4) & 0x0FFF;
            crc = crc ^ tmp ^ TABLE[(byte & 0xF) as usize];
            let tmp = TABLE[(crc & 0xF) as usize];
            crc = (crc >> 4) & 0x0FFF;
            crc = crc ^ tmp ^ TABLE[((byte >> 4) & 0xF) as usize];
        }
        crc
    }

    /// A minimal, valid `.fit` buffer: one `record` (global msg 20)
    /// definition with two fields (`timestamp`, `heart_rate`), and three
    /// data messages where the last two share a timestamp — the importer
    /// drops one with a duplicate-timestamp warning, same shape as
    /// `import::fit`'s own golden fixture, rebuilt minimally here for the
    /// same out-of-lane-scope reason as `fit_crc16` above.
    fn minimal_fit_with_warning() -> Vec<u8> {
        let mut body = Vec::new();
        body.push(0x40); // definition, local_type 0
        body.push(0x00); // reserved
        body.push(0x00); // architecture: little-endian
        body.extend_from_slice(&20u16.to_le_bytes()); // global_mesg_num = record
        body.push(2); // field count
        for &(num, size, base) in &[(253u8, 4u8, 0x86u8), (3, 1, 0x02)] {
            // timestamp: uint32, heart_rate: uint8
            body.push(num);
            body.push(size);
            body.push(base);
        }
        let t0: u32 = 1_000_000_000;
        for &(timestamp, hr) in &[(t0, 140u8), (t0 + 1, 142), (t0 + 1, 145)] {
            body.push(0x00); // data message, local_type 0
            body.extend_from_slice(&timestamp.to_le_bytes());
            body.push(hr);
        }

        let mut out = Vec::with_capacity(14 + body.len() + 2);
        out.push(14); // header size
        out.push(0x20); // protocol version 2.0
        out.extend_from_slice(&2100u16.to_le_bytes()); // profile version
        out.extend_from_slice(&(body.len() as u32).to_le_bytes()); // data size
        out.extend_from_slice(b".FIT");
        let header_crc = fit_crc16(&out[0..12]);
        out.extend_from_slice(&header_crc.to_le_bytes());
        out.extend_from_slice(&body);
        let file_crc = fit_crc16(&out);
        out.extend_from_slice(&file_crc.to_le_bytes());
        out
    }

    /// A minimal CSV buffer whose last two rows share `t_seconds` — the
    /// importer drops one with a duplicate/non-monotonic warning.
    fn minimal_csv_with_warning() -> Vec<u8> {
        b"t_seconds,ch1\n0,1.0\n1,2.0\n1,3.0\n".to_vec()
    }

    #[test]
    fn import_file_gpx_writes_parquet_and_session_json_with_warnings_and_time_channel() {
        // Arrange
        let root = temp_root();
        let bytes = minimal_gpx_with_warning();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        assert_eq!(report.outcome, ImportOutcome::Written);
        assert!(report.data_parquet.is_file());
        let sj_path = root.join("sessions").join(&report.session_id).join("session.json");
        assert!(sj_path.is_file());
        assert!(!report.import_warnings.is_empty());

        // `write_session_parquet` never persists synthesized `Time`/
        // `Distance` (C1 §2) — re-run the same synthesis the real reader
        // (`rust/tauri/src/session_source.rs::load_session`) does on every
        // read-back, proving Q2's fallback survives the parquet round trip.
        let mut session = read_session_parquet(&report.data_parquet).unwrap();
        crate::session::synthesis::synthesize_base_channels(&mut session);
        assert!(session.channels.iter().any(|c| c.channel_id == "Time"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_fit_writes_parquet_and_session_json_with_warnings_and_time_channel() {
        // Arrange
        let root = temp_root();
        let bytes = minimal_fit_with_warning();

        // Act
        let report = import_file(&root, "fit", &bytes).unwrap();

        // Assert
        assert_eq!(report.outcome, ImportOutcome::Written);
        assert!(report.data_parquet.is_file());
        let sj_path = root.join("sessions").join(&report.session_id).join("session.json");
        assert!(sj_path.is_file());
        assert!(!report.import_warnings.is_empty());

        // `write_session_parquet` never persists synthesized `Time`/
        // `Distance` (C1 §2) — re-run the same synthesis the real reader
        // (`rust/tauri/src/session_source.rs::load_session`) does on every
        // read-back, proving Q2's fallback survives the parquet round trip.
        let mut session = read_session_parquet(&report.data_parquet).unwrap();
        crate::session::synthesis::synthesize_base_channels(&mut session);
        assert!(session.channels.iter().any(|c| c.channel_id == "Time"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_csv_writes_parquet_and_session_json_with_warnings_and_time_channel() {
        // Arrange
        let root = temp_root();
        let bytes = minimal_csv_with_warning();

        // Act
        let report = import_file(&root, "csv", &bytes).unwrap();

        // Assert
        assert_eq!(report.outcome, ImportOutcome::Written);
        assert!(report.data_parquet.is_file());
        let sj_path = root.join("sessions").join(&report.session_id).join("session.json");
        assert!(sj_path.is_file());
        assert!(!report.import_warnings.is_empty());

        // `write_session_parquet` never persists synthesized `Time`/
        // `Distance` (C1 §2) — re-run the same synthesis the real reader
        // (`rust/tauri/src/session_source.rs::load_session`) does on every
        // read-back, proving Q2's fallback survives the parquet round trip.
        let mut session = read_session_parquet(&report.data_parquet).unwrap();
        crate::session::synthesis::synthesize_base_channels(&mut session);
        assert!(session.channels.iter().any(|c| c.channel_id == "Time"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_unknown_extension_returns_unknown_extension_kind() {
        // Arrange
        let root = temp_root();

        // Act
        let err = import_file(&root, "xyz", b"whatever").unwrap_err();

        // Assert
        assert_eq!(err.kind, ImportErrorKind::UnknownExtension);

        let _ = std::fs::remove_dir_all(&root);
    }

    // --- lap-index-at-import fixtures ---------------------------------------
    //
    // A there-and-back-and-there GPS track: 3 legs of 100 one-second
    // trackpoints, lat sweeping 0.000→0.099→0.000→0.099 at a fixed longitude
    // — the same geometry `store::lap_index`'s own tests use (not reachable
    // from here: that module's fixtures live inside its own `#[cfg(test)]`),
    // duplicated minimally rather than reached into, same precedent as
    // `fit_crc16` above.

    /// `2026-01-01T00:MM:SSZ` for `offset_secs` since midnight (never wraps
    /// an hour for this fixture's < 300 s span).
    fn iso_time(offset_secs: i64) -> String {
        format!("2026-01-01T{:02}:{:02}:{:02}Z", offset_secs / 3600, (offset_secs % 3600) / 60, offset_secs % 60)
    }

    /// A GPX file crossing lat 0.05 three times over 300 s — enough for
    /// `circuit_track_fixture`'s start/finish gate to detect 3 laps.
    fn gpx_three_lap_bytes() -> Vec<u8> {
        let mut trkpts = String::new();
        for leg in 0..3i64 {
            let up = leg % 2 == 0;
            for i in 0..100i64 {
                let lat = if up { i as f64 * 0.001 } else { 0.099 - i as f64 * 0.001 };
                trkpts.push_str(&format!(
                    "<trkpt lat=\"{lat}\" lon=\"0.0005\"><time>{}</time></trkpt>\n",
                    iso_time(leg * 100 + i)
                ));
            }
        }
        format!("<gpx><trk><trkseg>\n{trkpts}</trkseg></trk></gpx>").into_bytes()
    }

    /// A circuit track whose reference polyline covers the same lat range as
    /// [`gpx_three_lap_bytes`], with a start/finish gate at lat 0.05.
    fn circuit_track_fixture(id: &str) -> Track {
        let polyline: Vec<GpsFix> =
            (0..=100).map(|i| GpsFix { timestamp_ms: i * 1000, lat: i as f64 * 0.001, lon: 0.0005 }).collect();
        Track {
            id: id.to_string(),
            name: "Loop".to_string(),
            venue: String::new(),
            timing: Some(LapTiming::Circuit { start_finish: Gate { lat1: 0.05, lon1: -0.001, lat2: 0.05, lon2: 0.001 } }),
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: polyline,
            created_at_ms: 0,
            updated_at_ms: 1,
        }
    }

    #[test]
    fn import_file_gpx_with_matching_track_indexes_laps_into_session_json() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track_fixture("loop-1")).unwrap();
        let bytes = gpx_three_lap_bytes();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        let lap_index = report.lap_index.as_ref().expect("lap_index should be Some on a successful step");
        assert!(report.lap_index_warning.is_none());
        assert_eq!(lap_index.visits_indexed, 1);
        assert_eq!(lap_index.laps_indexed, 3);
        let doc = read_session_json(&root.join("sessions").join(&report.session_id).join("session.json")).unwrap();
        assert_eq!(doc.laps.len(), 3);
        assert_eq!(doc.track_visits.len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_gpx_with_no_tracks_dir_is_honest_empty_lap_index() {
        // Arrange — no `tracks/` at all.
        let root = temp_root();
        let bytes = minimal_gpx_with_warning();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        let lap_index = report.lap_index.as_ref().unwrap();
        assert_eq!(lap_index.laps_indexed, 0);
        assert!(lap_index.warnings.is_empty());
        let doc = read_session_json(&root.join("sessions").join(&report.session_id).join("session.json")).unwrap();
        assert!(doc.laps.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_gpx_reimport_same_bytes_skips_lap_index_and_leaves_session_json_untouched() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track_fixture("loop-1")).unwrap();
        let bytes = gpx_three_lap_bytes();
        let first = import_file(&root, "gpx", &bytes).unwrap();
        let sj_path = root.join("sessions").join(&first.session_id).join("session.json");
        let bytes_before = std::fs::read(&sj_path).unwrap();

        // Act — the `Skip` plan (same bytes, same build).
        let second = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        assert_eq!(second.outcome, ImportOutcome::Skipped);
        assert!(second.lap_index.as_ref().unwrap().skipped_up_to_date);
        assert_eq!(std::fs::read(&sj_path).unwrap(), bytes_before);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_gpx_with_malformed_track_artifact_still_indexes_readable_tracks_and_warns() {
        // Arrange — one valid `.idl0t`, one that isn't JSON.
        let root = temp_root();
        write_track(&root, &circuit_track_fixture("loop-1")).unwrap();
        std::fs::write(root.join("tracks").join("bad.idl0t"), b"not json").unwrap();
        let bytes = gpx_three_lap_bytes();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        let lap_index = report.lap_index.as_ref().unwrap();
        assert_eq!(lap_index.laps_indexed, 3);
        assert!(lap_index.warnings.iter().any(|w| w.contains("bad.idl0t")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_gpx_regenerate_plan_still_reindexes_laps() {
        // Arrange — a stale `data.parquet` (older importer_version) already
        // exists for this exact blob, so the next import takes the
        // `Regenerate` plan.
        let root = temp_root();
        write_track(&root, &circuit_track_fixture("loop-1")).unwrap();
        let bytes = gpx_three_lap_bytes();
        let importer = crate::import::importer_for_extension("gpx").unwrap();
        let blob_sha256 = crate::store::atomic::sha256_hex(&bytes);
        let mut outcome = importer.import(&bytes, &blob_sha256).unwrap();
        crate::session::synthesis::synthesize_base_channels(&mut outcome.session);
        let written_blob = blob::write_blob(&root, &bytes).unwrap();
        outcome.session.blob_sha256 = written_blob;
        write_session_parquet(&root, &outcome.session, "0.0.1").unwrap();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert
        assert_eq!(report.outcome, ImportOutcome::Regenerated);
        let lap_index = report.lap_index.as_ref().unwrap();
        assert!(!lap_index.skipped_up_to_date);
        assert_eq!(lap_index.laps_indexed, 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_malformed_gpx_leaves_no_orphan_blob() {
        // Arrange — non-UTF-8 bytes for a "gpx" extension, mirroring
        // `import_idl0`'s own "bad magic bytes leaves nothing in the CAS"
        // ordering test for the non-.idl0 path.
        let root = temp_root();
        let bytes: Vec<u8> = vec![0xFF, 0xFE, 0xFD];
        let would_be_digest = crate::store::atomic::sha256_hex(&bytes);

        // Act
        let result = import_file(&root, "gpx", &bytes);

        // Assert
        assert!(matches!(result, Err(ImportError { kind: ImportErrorKind::ImportNotUtf8, .. })));
        assert!(!crate::store::blob::blob_exists(&root, &would_be_digest));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_into_a_root_with_an_existing_catalog_indexes_laps_without_a_rebuild() {
        // Arrange — a catalog already exists (built on the empty tree,
        // before this session's blob ever existed) so `finish_import`'s
        // `index_session` call must insert this session's own `blobs` row
        // itself (ruling R84) rather than rely on a prior `rebuild_catalog`
        // scan of the CAS.
        let root = temp_root();
        crate::store::catalog::rebuild_catalog(&root).unwrap();
        write_track(&root, &circuit_track_fixture("loop-1")).unwrap();
        let bytes = gpx_three_lap_bytes();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert — no error/warning recording a failed catalog update, and
        // the laps are queryable immediately, with no intervening
        // `rebuild_catalog` call.
        assert!(report.catalog_index_warning.is_none());
        let laps = crate::store::catalog_read::list_laps(&root, &report.session_id).unwrap();
        assert_eq!(laps.len(), 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_file_into_a_root_with_no_catalog_succeeds_and_creates_none() {
        // Arrange — no `catalog.sqlite` at all under `root`.
        let root = temp_root();
        let bytes = minimal_gpx_with_warning();

        // Act
        let report = import_file(&root, "gpx", &bytes).unwrap();

        // Assert — this pipeline never creates a catalog as a side effect.
        assert!(report.catalog_index_warning.is_none());
        assert!(!root.join("catalog.sqlite").is_file());

        let _ = std::fs::remove_dir_all(&root);
    }
}
