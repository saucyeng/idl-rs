//! [`ImporterError`] — typed failures raised while importing a non-device
//! source (FIT, GPX, CSV). Lives in its own file per contract C3 §2's
//! post-sign note (`docs/superpowers/specs/2026-09-03-idl1-c3-ipc-surface.md`),
//! not inline in `import/mod.rs`.

/// Failures raised while importing a non-device source. Variant names are
/// written to fit the `parse_*` kind-vocabulary prefix contract C3 §2
/// assigns to importer/parser errors (mirroring
/// [`crate::session::ParseError`]'s existing `parse_*` kinds for `.idl0`);
/// C3 §2 does not yet carry rows for these variants — see
/// `docs/IDL0_SPEC.md` §15a.5 and the L2 plan's Open Questions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImporterError {
    /// The FIT byte stream failed CRC validation, has no usable `record`
    /// messages, or `fitparser` otherwise rejected it. Carries `fitparser`'s
    /// message.
    FitMalformed(String),
    /// The GPX document is not well-formed XML.
    GpxMalformedXml(String),
    /// The GPX document has no `<trkpt>` elements.
    GpxNoTrackpoints,
    /// A `<trkpt>` is missing a required `lat` or `lon` attribute.
    GpxMissingLatLon(String),
    /// A `<trkpt>`'s `lat`/`lon` attribute is not a parseable number.
    GpxUnparseableLatLon(String),
    /// The CSV header row is missing, malformed, or has no data rows.
    CsvMalformed(String),
    /// The source bytes could not be read as UTF-8 text (GPX/CSV only —
    /// FIT is binary and never raises this).
    NotUtf8(String),
}

impl std::fmt::Display for ImporterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImporterError::FitMalformed(m) => write!(f, "FitMalformed: {m}"),
            ImporterError::GpxMalformedXml(m) => write!(f, "GpxMalformedXml: {m}"),
            ImporterError::GpxNoTrackpoints => write!(f, "GpxNoTrackpoints"),
            ImporterError::GpxMissingLatLon(m) => write!(f, "GpxMissingLatLon: {m}"),
            ImporterError::GpxUnparseableLatLon(m) => write!(f, "GpxUnparseableLatLon: {m}"),
            ImporterError::CsvMalformed(m) => write!(f, "CsvMalformed: {m}"),
            ImporterError::NotUtf8(m) => write!(f, "NotUtf8: {m}"),
        }
    }
}

impl std::error::Error for ImporterError {}
