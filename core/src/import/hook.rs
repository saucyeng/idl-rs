//! Post-import materialisation hook (design doc §5's "materialised derived
//! channels" extension point,
//! `docs/superpowers/specs/2026-09-02-idl1-rewrite-design.md`). This module
//! ships only the hook's shape — the real estimator wiring (the iEKF chain)
//! is L1/L3's, once the materialised-channel store exists. See
//! `docs/IDL0_SPEC.md` §15a.5 and this plan's Open Questions. Per L2-R12,
//! this file does not include an `import_with_hook` call-site wrapper — the
//! one real call site (`crate::store::import::import_file`) wires
//! [`NoopPostImportHook`] directly.

use crate::session::Session;

/// Runs once, immediately after a successful [`super::Importer::import`],
/// with the freshly produced [`Session`]. The extension point L1's store
/// (or the CLI) uses to trigger materialised-channel computation without
/// `import` itself depending on the estimator crate.
pub trait PostImportHook {
    /// Called with the imported `session`. Must not panic on ordinary bad
    /// data (CLAUDE.md §5) — there is no channel back to the caller for a
    /// hook-raised problem today; see Open Questions.
    fn on_imported(&self, session: &Session);
}

/// The hook that does nothing — the default until L1/L3 wire a real one.
pub struct NoopPostImportHook;

impl PostImportHook for NoopPostImportHook {
    fn on_imported(&self, _session: &Session) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SourceFormat;

    #[test]
    fn noop_post_import_hook_runs_without_panicking() {
        // Arrange
        let session = Session {
            session_id: String::new(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Csv,
            blob_sha256: String::new(),
            channels: Vec::new(),
        };

        // Act / Assert — exists only to prove the trait has a working
        // zero-cost default implementer.
        NoopPostImportHook.on_imported(&session);
    }
}
