//! Non-device source importers (FIT, GPX, CSV) producing the canonical
//! [`Session`] model (contract C1 §2). Pure: no I/O beyond bytes the caller
//! already read, no Tauri, no network (CLAUDE.md §2 — this is `core`).
//!
//! Format-specific code runs once, at import (design doc principle
//! "canonicalise on ingest", `docs/superpowers/specs/2026-09-02-idl1-rewrite-design.md`
//! §3) — everything downstream of [`Importer::import`] reads the same
//! [`Session`]/[`Channel`](crate::session::Channel) shape regardless of
//! source. `.idl0` import (`crate::parse` plus L1's burst-seam correction,
//! C1 §3.3) does not implement this trait in this module — see
//! `docs/IDL0_SPEC.md` §15a.1.

pub mod csv;
pub mod fit;
pub mod gpx;
pub mod hook;

mod error;

pub use error::ImporterError;

use crate::session::{Session, SourceFormat};

/// Converts one immutable source blob into the canonical [`Session`] model.
/// One implementation per non-device [`SourceFormat`] (C1 §2,
/// `docs/superpowers/specs/2026-09-03-idl1-c1-session-schema.md`). L1's
/// store writes whatever a conforming implementation returns straight to
/// `data.parquet` (C1 §4) — no format-specific code runs downstream of this.
pub trait Importer {
    /// Which [`SourceFormat`] this importer produces.
    fn source_format(&self) -> SourceFormat;

    /// Parses `bytes` — the exact, immutable content of the source blob, as
    /// read from disk by the caller — into a [`Session`]. `blob_sha256` is
    /// the caller-computed SHA-256 hex digest of `bytes` (CAS hashing is
    /// C4's/L1's job, not repeated here); this function only stamps it onto
    /// [`Session::blob_sha256`] and derives [`Session::session_id`] from it
    /// via [`session_id_from_blob_hash`].
    fn import(&self, bytes: &[u8], blob_sha256: &str) -> Result<ImportedSession, ImporterError>;

    /// This importer implementation's own version string (C1 §4.3 —
    /// `write_session_parquet`'s `importer_version` argument). Each
    /// format's own module defines its version const; this method just
    /// surfaces it through the trait object.
    fn importer_version(&self) -> &'static str;
}

/// One [`Importer::import`] call's result: the parsed session plus any
/// non-fatal warnings raised while recovering readable data. Never silently
/// drops a warning-worthy condition and never panics on it either
/// (CLAUDE.md §5). Renamed from the plan's original `ImportOutcome` to avoid
/// a name collision with landed [`crate::store::import::ImportOutcome`]
/// (SPEC §15a.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedSession {
    /// The parsed session.
    pub session: Session,
    /// Advisory messages surfaced during import — e.g. a dropped duplicate
    /// timestamp (C1 §3.4). Empty when the source parsed with no concerns.
    pub warnings: Vec<ImporterWarning>,
}

/// One non-fatal advisory raised during import. CLAUDE.md §5 — recover what
/// is readable, never crash on bad data, never silently drop it either.
/// Renamed from the plan's original `ImportWarning` to avoid a name
/// collision with landed [`crate::session::ImportWarning`] (SPEC §15a.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImporterWarning {
    /// Human-readable text describing what was recovered/dropped and why.
    pub message: String,
}

impl ImporterWarning {
    /// Builds a warning with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        ImporterWarning { message: message.into() }
    }
}

/// Derives `session_id` for a non-device source (C4 §3): the first 16
/// lowercase hex characters of `blob_sha256`. Collision extension (18, 20,
/// ... hex characters, on a real catalog collision) needs catalog state
/// this pure function does not have — L1's job at catalog-insert time.
///
/// Precondition (C4 §3, not validated here): `blob_sha256` must already be
/// a valid SHA-256 hex digest (64 lowercase hex characters). Every real
/// caller passes a value it just computed via `sha2`/`write_blob`, so this
/// function stays infallible rather than rippling `?` through every
/// importer for a case that cannot occur in practice — it only normalizes
/// case.
pub fn session_id_from_blob_hash(blob_sha256: &str) -> String {
    debug_assert!(
        blob_sha256.len() >= 16,
        "blob_sha256 must be at least 16 hex characters (C4 §3)"
    );

    blob_sha256.to_ascii_lowercase().chars().take(16).collect()
}

/// Selects an importer by lowercase file extension (without the dot).
/// `None` for anything this module does not (yet) cover — the caller
/// (CLI/L1 store) decides how to report an unrecognised extension.
pub fn importer_for_extension(ext: &str) -> Option<Box<dyn Importer>> {
    match ext {
        "gpx" => Some(Box::new(gpx::GpxImporter)),
        "fit" => Some(Box::new(fit::FitImporter)),
        "csv" => Some(Box::new(csv::CsvImporter)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the trait's own shape (object-safety, method signatures)
    /// independently of any real format parser — GPX/FIT/CSV each get their
    /// own golden tests in their own modules (Tasks 3-5).
    struct StubImporter;
    impl Importer for StubImporter {
        fn source_format(&self) -> SourceFormat {
            SourceFormat::Fit
        }
        fn import(&self, bytes: &[u8], blob_sha256: &str) -> Result<ImportedSession, ImporterError> {
            Ok(ImportedSession {
                session: Session {
                    session_id: session_id_from_blob_hash(blob_sha256),
                    device_id: None,
                    timestamp_utc_ms: 0,
                    config_checksum: None,
                    source_format: SourceFormat::Fit,
                    blob_sha256: blob_sha256.to_string(),
                    channels: Vec::new(),
                },
                warnings: if bytes.is_empty() {
                    vec![ImporterWarning::new("empty input")]
                } else {
                    Vec::new()
                },
            })
        }
        fn importer_version(&self) -> &'static str {
            "0.0.0-stub"
        }
    }

    #[test]
    fn session_id_from_blob_hash_takes_first_16_hex_chars() {
        // Arrange
        let hash = format!("abcdef0123456789{}", "0".repeat(48));
        assert_eq!(hash.len(), 64);

        // Act
        let id = session_id_from_blob_hash(&hash);

        // Assert
        assert_eq!(id, "abcdef0123456789");
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn session_id_from_blob_hash_uppercase_input_is_lowercased() {
        // Arrange
        let hash = "AB".repeat(32);
        assert_eq!(hash.len(), 64);

        // Act
        let id = session_id_from_blob_hash(&hash);

        // Assert
        assert_eq!(id, "abababababababab");
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn stub_importer_via_trait_object_produces_session_with_derived_id() {
        // Arrange
        let importer: Box<dyn Importer> = Box::new(StubImporter);
        let hash = "aa00".repeat(16);
        assert_eq!(hash.len(), 64);

        // Act
        let outcome = importer.import(b"x", &hash).unwrap();

        // Assert
        assert_eq!(outcome.session.session_id, "aa00aa00aa00aa00");
        assert_eq!(outcome.session.source_format, SourceFormat::Fit);
        assert!(outcome.warnings.is_empty());
    }

    #[test]
    fn stub_importer_empty_bytes_surfaces_warning_not_error() {
        // Arrange
        let importer = StubImporter;
        let hash = "bb".repeat(32);
        assert_eq!(hash.len(), 64);

        // Act
        let outcome = importer.import(b"", &hash).unwrap();

        // Assert
        assert_eq!(outcome.warnings.len(), 1);
        assert_eq!(outcome.warnings[0].message, "empty input");
    }
}
