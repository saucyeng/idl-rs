//! The single typed error every `idl-rs-tauri` command returns (C3 §2).
//! `IpcErrorKind` grows additively as each lane's core error enum lands
//! (C3 §5, "IpcError.kind values are additive-only") — this file seeds the
//! four cross-cutting kinds and the four kinds sourced from
//! `idl_transport::TransportErrorKind` (already shipped, M0 Task 2); L1–L4
//! each add their own prefixed variants (`parse_*`, `math_*`, `config_*`,
//! `export_*`) in their own Group B task here, never editing another lane's
//! variant.

/// Machine-readable failure class. Frontend code routes on the serialized
/// string, never on `IpcError::message`. Variants are additive-only once
/// shipped (C3 §5) — never renamed or removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IpcErrorKind {
    /// `TransportErrorKind::Ble` — device: `ble_scan`, `ble_connect`, `list_device_files`, `download_file`, `push_config`.
    Ble,
    /// `TransportErrorKind::Wifi` — device: `list_device_files`, `download_file`, `push_config`.
    Wifi,
    /// `TransportErrorKind::Config` — device rejected or malformed a pushed config over the wire.
    Config,
    /// `TransportErrorKind::Sync` — LAN sync (L11, wave 2; no command uses this kind yet).
    Sync,
    /// Cross-cutting: the named entity (session, workbook, channel, peer…) does not exist.
    NotFound,
    /// Cross-cutting: a caller-supplied argument fails local validation.
    InvalidArgument,
    /// Cross-cutting: a filesystem read/write failed. Folds every core `Io(...)` variant (C3 §2 folding rule).
    Io,
    /// Cross-cutting: an unexpected/programmer-error condition, not the caller's fault. Folds `ExportError::Json`.
    Internal,
    /// Cross-cutting (added post-sign, 2026-09-04, lead ruling R44): an
    /// optimistic-concurrency check failed (`store::atomic::RenameConflict`)
    /// — the file on disk is no longer the version the caller based its edit
    /// on. Not the caller's fault and not a bug (so not folded into
    /// `InvalidArgument`) — a recoverable condition the UI must present
    /// differently ("this file changed elsewhere — reload?"). Raised by
    /// Workbook: `save_workbook`.
    Conflict,
    /// `WorkbookErrorKind::DuplicateCellId` — workbook: `eval_workbook` (per cell only, C2 §3.5.A).
    WorkbookDuplicateCellId,
    /// `WorkbookErrorKind::DuplicateDefinition` — workbook: `eval_workbook` (per cell only).
    WorkbookDuplicateDefinition,
    /// `WorkbookErrorKind::DuplicateConstant` — workbook: `eval_workbook` (per cell only).
    WorkbookDuplicateConstant,
    /// `WorkbookErrorKind::InvalidIdentifier` — workbook: `eval_workbook` (per cell only).
    WorkbookInvalidIdentifier,
    /// `WorkbookErrorKind::ReservedName` — workbook: `eval_workbook` (per cell only).
    WorkbookReservedName,
    /// `WorkbookErrorKind::MissingFrontMatterId` — workbook: `open_workbook`, `eval_workbook`
    /// (fatal — no `WorkbookDoc` is constructable without a valid `id`, C2 §3.5.A).
    WorkbookMissingFrontMatterId,
    /// `WorkbookErrorKind::UnsupportedWorkbookVersion` — workbook: `open_workbook`, `eval_workbook`
    /// (fatal — explicit `version` != 3, C2 §1).
    WorkbookUnsupportedVersion,
    /// `WorkbookErrorKind::InvalidTableJson` — workbook: `eval_workbook` (per cell only, C2 §4).
    WorkbookInvalidTableJson,
    /// `MathEvalErrorKind::Parse` — workbook: `eval_workbook` (per definition).
    MathParse,
    /// `MathEvalErrorKind::UnknownFunction` — workbook: `eval_workbook` (per definition).
    MathUnknownFunction,
    /// `MathEvalErrorKind::UnknownChannel` — workbook: `eval_workbook` (per definition).
    MathUnknownChannel,
    /// `MathEvalErrorKind::ArgCount` — workbook: `eval_workbook` (per definition).
    MathArgCount,
    /// `MathEvalErrorKind::Type` — workbook: `eval_workbook` (per definition).
    MathType,
    /// `MathEvalErrorKind::DivisionByZero` — workbook: `eval_workbook` (per definition).
    MathDivisionByZero,
    /// `MathEvalErrorKind::NoLapContext` — workbook: `eval_workbook` (per definition).
    MathNoLapContext,
    /// `MathEvalErrorKind::NotImplemented` — workbook: `eval_workbook` (per definition).
    MathNotImplemented,
    /// `MathEvalErrorKind::Runtime` — workbook: `eval_workbook` (per definition).
    MathRuntime,
    /// `idl_rs::store::import::ImportErrorKind::ParseInvalidMagicBytes`
    /// (mirrors `ParseError::InvalidMagicBytes`) — Import: `import_file`
    /// (`.idl0` source, C3 §2 `parse_invalid_magic_bytes`).
    ParseInvalidMagicBytes,
    /// `ImportErrorKind::ParseUnsupportedSchemaVersion` — Import:
    /// `import_file` (`.idl0` source, C3 §2 `parse_unsupported_schema_version`).
    ParseUnsupportedSchemaVersion,
    /// `ImportErrorKind::ParseTruncatedRecord` — Import: `import_file`
    /// (`.idl0` source, buffer too short to read even the magic/schema
    /// bytes; a log that parses but ends mid-record instead surfaces via
    /// `ImportOutcome::warnings`, never this kind — C3 §2 `parse_truncated_record`).
    ParseTruncatedRecord,
    /// `ImportErrorKind::ImportFitMalformed` — Import: `import_file` (`.fit`
    /// source, C3 §2 `import_fit_malformed`, R7).
    ImportFitMalformed,
    /// `ImportErrorKind::ImportGpxMalformedXml` — Import: `import_file`
    /// (`.gpx` source, C3 §2 `import_gpx_malformed_xml`).
    ImportGpxMalformedXml,
    /// `ImportErrorKind::ImportGpxNoTrackpoints` — Import: `import_file`
    /// (`.gpx` source, C3 §2 `import_gpx_no_trackpoints`).
    ImportGpxNoTrackpoints,
    /// `ImportErrorKind::ImportGpxMissingLatLon` — Import: `import_file`
    /// (`.gpx` source, C3 §2 `import_gpx_missing_lat_lon`).
    ImportGpxMissingLatLon,
    /// `ImportErrorKind::ImportGpxUnparseableLatLon` — Import: `import_file`
    /// (`.gpx` source, C3 §2 `import_gpx_unparseable_lat_lon`).
    ImportGpxUnparseableLatLon,
    /// `ImportErrorKind::ImportCsvMalformed` — Import: `import_file` (`.csv`
    /// source, C3 §2 `import_csv_malformed`).
    ImportCsvMalformed,
    /// `ImportErrorKind::ImportNotUtf8` — Import: `import_file` (`.gpx`/
    /// `.csv` source, C3 §2 `import_not_utf8`).
    ImportNotUtf8,
    /// `ImportErrorKind::Collision` (added post-sign, 2026-09-05, lead
    /// ruling R60) — Import: `import_file`, a re-import of a different blob
    /// under an existing `session_id` (C4 §3). Kept distinct from the
    /// cross-cutting `Conflict` kind — same precedent R44 set for
    /// `Conflict` itself — because this is not an optimistic-concurrency
    /// failure (C3 §2 `import_collision`).
    ImportCollision,
    /// C3 §2 (added post-sign, 2026-09-05, lead ruling R59, extended by R63):
    /// a device refused a control transition by returning a non-success
    /// `AckCode` over the Control characteristic (SPEC §7.2) — distinct from
    /// a transport-level failure (`Ble`), which covers "the write/read
    /// itself didn't complete". **Platform-limited today** (R63/R63.1):
    /// `btleplug`'s desktop backends surface only `Result<(), TransportError>`
    /// from a Control write, never the raw ACK byte
    /// (`idl_transport::ble_transport::BtleplugBle::send_command`'s own doc
    /// comment) — so no call site in `idl-rs-tauri` can construct this kind
    /// from a real `AckCode` yet. Device: `device_control` (see the
    /// `// TODO(idl0):` at its `send_command` call site).
    DeviceRejected,
}

/// One JSON error crossing every fallible command (C3 §2). `detail`'s shape
/// depends on `kind`; absent when there is nothing structured to add.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IpcError {
    pub kind: IpcErrorKind,
    /// Human-readable text. No stack traces (CLAUDE.md §5).
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

impl IpcError {
    /// Builds an `IpcError` with no structured detail.
    pub fn new(kind: IpcErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), detail: None }
    }

    /// Builds an `IpcError` carrying structured `detail`.
    pub fn with_detail(kind: IpcErrorKind, message: impl Into<String>, detail: serde_json::Value) -> Self {
        Self { kind, message: message.into(), detail: Some(detail) }
    }
}

impl From<idl_transport::TransportError> for IpcError {
    fn from(e: idl_transport::TransportError) -> Self {
        let kind = match e.kind {
            idl_transport::TransportErrorKind::Ble => IpcErrorKind::Ble,
            idl_transport::TransportErrorKind::Wifi => IpcErrorKind::Wifi,
            idl_transport::TransportErrorKind::Config => IpcErrorKind::Config,
            idl_transport::TransportErrorKind::Sync => IpcErrorKind::Sync,
        };
        IpcError::new(kind, e.message)
    }
}

/// C3 §3.2 (Catalog). `idl_rs::store::catalog::CatalogError` folds three
/// ways (ruling R46): `NotFound` (the requested entity does not exist) maps
/// to `IpcErrorKind::NotFound`; `Sql` (a genuine SQLite failure — e.g. a
/// corrupt `catalog.sqlite`) maps to `IpcErrorKind::Internal`, pointing the
/// caller at `rebuild_catalog` rather than telling them their data is
/// missing; `Io` maps to `IpcErrorKind::Io`. C3 §2's catalog rows are
/// `not_found`/`io`/`internal` only — no new `IpcErrorKind` variant is
/// added here.
impl From<idl_rs::store::catalog::CatalogError> for IpcError {
    fn from(e: idl_rs::store::catalog::CatalogError) -> Self {
        use idl_rs::store::catalog::CatalogErrorKind;
        let kind = match e.kind {
            CatalogErrorKind::NotFound => IpcErrorKind::NotFound,
            CatalogErrorKind::Sql => IpcErrorKind::Internal,
            CatalogErrorKind::Io => IpcErrorKind::Io,
        };
        IpcError::new(kind, e.message)
    }
}

/// C3 §3.4 (Workbook). `idl_rs::math::MathEvalError` (evaluation-time,
/// per-definition failures, C2 §3.5.B) — every variant gets its own
/// `math_*` `IpcErrorKind` (C3 §2's naming rule: prefix every source enum's
/// variants, even where no collision exists today).
impl From<idl_rs::math::MathEvalError> for IpcError {
    fn from(e: idl_rs::math::MathEvalError) -> Self {
        use idl_rs::math::MathEvalErrorKind;
        let kind = match e.kind {
            MathEvalErrorKind::Parse => IpcErrorKind::MathParse,
            MathEvalErrorKind::UnknownFunction => IpcErrorKind::MathUnknownFunction,
            MathEvalErrorKind::UnknownChannel => IpcErrorKind::MathUnknownChannel,
            MathEvalErrorKind::ArgCount => IpcErrorKind::MathArgCount,
            MathEvalErrorKind::Type => IpcErrorKind::MathType,
            MathEvalErrorKind::DivisionByZero => IpcErrorKind::MathDivisionByZero,
            MathEvalErrorKind::NoLapContext => IpcErrorKind::MathNoLapContext,
            MathEvalErrorKind::NotImplemented => IpcErrorKind::MathNotImplemented,
            MathEvalErrorKind::Runtime => IpcErrorKind::MathRuntime,
        };
        IpcError::new(kind, e.message)
    }
}

/// C3 §3.4 (Workbook). `idl_rs::workbook::v3::error::WorkbookErrorKind`
/// (parse-time/structural failures, C2 §3.5.A) — only the eight variants
/// the landed enum actually has (C3 §2's `workbook_invalid_front_matter`/
/// `workbook_invalid_cell_id` rows have no source variant here; see this
/// lane's report). Takes `&WorkbookError` (rather than by value) so a
/// caller holding a `&WorkbookError` borrowed from a `Vec` (e.g.
/// `eval_workbook`'s per-cell `errors` routing) does not need to clone
/// first.
impl From<&idl_rs::workbook::v3::WorkbookError> for IpcError {
    fn from(e: &idl_rs::workbook::v3::WorkbookError) -> Self {
        use idl_rs::workbook::v3::WorkbookErrorKind;
        let kind = match e.kind {
            WorkbookErrorKind::DuplicateCellId => IpcErrorKind::WorkbookDuplicateCellId,
            WorkbookErrorKind::DuplicateDefinition => IpcErrorKind::WorkbookDuplicateDefinition,
            WorkbookErrorKind::DuplicateConstant => IpcErrorKind::WorkbookDuplicateConstant,
            WorkbookErrorKind::InvalidIdentifier => IpcErrorKind::WorkbookInvalidIdentifier,
            WorkbookErrorKind::ReservedName => IpcErrorKind::WorkbookReservedName,
            WorkbookErrorKind::MissingFrontMatterId => IpcErrorKind::WorkbookMissingFrontMatterId,
            WorkbookErrorKind::UnsupportedWorkbookVersion => IpcErrorKind::WorkbookUnsupportedVersion,
            WorkbookErrorKind::InvalidTableJson => IpcErrorKind::WorkbookInvalidTableJson,
        };
        IpcError::new(kind, e.message.clone())
    }
}

/// C3 §3.3 (Import). `idl_rs::store::import::ImportError` — the boundary
/// type both `import_idl0` and `import_file` return (L2-R13); it already
/// wraps `ParseError`/`ImporterError` internally, so this is the one `From`
/// impl this task adds (**not** `From<idl_rs::session::ParseError>`, which
/// this file's own type never sees directly). `UnknownExtension` folds into
/// the cross-cutting `InvalidArgument` (no dedicated C3 §2 row — L2's own
/// brief-task6 anticipated this mapping); every other variant gets its own
/// `IpcErrorKind`, matching C3 §2's `import_*`/`parse_*` rows one for one.
impl From<idl_rs::store::import::ImportError> for IpcError {
    fn from(e: idl_rs::store::import::ImportError) -> Self {
        use idl_rs::store::import::ImportErrorKind;
        let kind = match e.kind {
            ImportErrorKind::Io => IpcErrorKind::Io,
            ImportErrorKind::ParseInvalidMagicBytes => IpcErrorKind::ParseInvalidMagicBytes,
            ImportErrorKind::ParseUnsupportedSchemaVersion => IpcErrorKind::ParseUnsupportedSchemaVersion,
            ImportErrorKind::ParseTruncatedRecord => IpcErrorKind::ParseTruncatedRecord,
            ImportErrorKind::Collision => IpcErrorKind::ImportCollision,
            ImportErrorKind::ImportFitMalformed => IpcErrorKind::ImportFitMalformed,
            ImportErrorKind::ImportGpxMalformedXml => IpcErrorKind::ImportGpxMalformedXml,
            ImportErrorKind::ImportGpxNoTrackpoints => IpcErrorKind::ImportGpxNoTrackpoints,
            ImportErrorKind::ImportGpxMissingLatLon => IpcErrorKind::ImportGpxMissingLatLon,
            ImportErrorKind::ImportGpxUnparseableLatLon => IpcErrorKind::ImportGpxUnparseableLatLon,
            ImportErrorKind::ImportCsvMalformed => IpcErrorKind::ImportCsvMalformed,
            ImportErrorKind::ImportNotUtf8 => IpcErrorKind::ImportNotUtf8,
            ImportErrorKind::UnknownExtension => IpcErrorKind::InvalidArgument,
        };
        IpcError::new(kind, e.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_error_kind_serialises_snake_case_matching_c3() {
        // Arrange
        let err = IpcError::new(IpcErrorKind::NotFound, "no such session");

        // Act
        let json = serde_json::to_string(&err).unwrap();

        // Assert
        assert_eq!(json, r#"{"kind":"not_found","message":"no such session"}"#);
    }

    #[test]
    fn ipc_error_with_detail_serialises_the_detail_object() {
        // Arrange
        let err = IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            "unknown channel",
            serde_json::json!({ "channel": "fork_travel" }),
        );

        // Act
        let json = serde_json::to_string(&err).unwrap();

        // Assert
        assert_eq!(
            json,
            r#"{"kind":"invalid_argument","message":"unknown channel","detail":{"channel":"fork_travel"}}"#
        );
    }

    #[test]
    fn catalog_error_sql_kind_converts_to_internal_not_not_found() {
        // Arrange — R46: a genuine SQLite failure (corrupt catalog.sqlite)
        // must surface as `internal`, pointing at `rebuild_catalog`, not
        // `not_found`, which would wrongly suggest the data is missing.
        let ce = idl_rs::store::catalog::CatalogError {
            kind: idl_rs::store::catalog::CatalogErrorKind::Sql,
            message: "file is not a database".to_string(),
        };

        // Act
        let ie: IpcError = ce.into();

        // Assert
        assert_eq!(ie.kind, IpcErrorKind::Internal);
    }

    #[test]
    fn catalog_error_not_found_kind_converts_to_not_found() {
        // Arrange
        let ce = idl_rs::store::catalog::CatalogError {
            kind: idl_rs::store::catalog::CatalogErrorKind::NotFound,
            message: "session nope not found".to_string(),
        };

        // Act
        let ie: IpcError = ce.into();

        // Assert
        assert_eq!(ie.kind, IpcErrorKind::NotFound);
    }

    #[test]
    fn conflict_kind_serialises_matching_c3_r44() {
        // Arrange
        let err = IpcError::with_detail(
            IpcErrorKind::Conflict,
            "workbooks/w1.idl1wb changed since it was read",
            serde_json::json!({ "expected": "h0", "found": "h1" }),
        );

        // Act
        let json = serde_json::to_string(&err).unwrap();

        // Assert
        assert_eq!(
            json,
            r#"{"kind":"conflict","message":"workbooks/w1.idl1wb changed since it was read","detail":{"expected":"h0","found":"h1"}}"#
        );
    }

    #[test]
    fn math_eval_error_converts_kind_preserving_message() {
        // Arrange
        let e = idl_rs::math::MathEvalError::new(idl_rs::math::MathEvalErrorKind::UnknownChannel, "Channel '[Nope]' not in this session");

        // Act
        let ie: IpcError = e.into();

        // Assert
        assert_eq!(ie.kind, IpcErrorKind::MathUnknownChannel);
        assert_eq!(ie.message, "Channel '[Nope]' not in this session");
    }

    #[test]
    fn workbook_error_ref_converts_kind_preserving_message() {
        // Arrange
        let e = idl_rs::workbook::v3::error::duplicate_definition("aaaaaaaa", "x");

        // Act
        let ie: IpcError = (&e).into();

        // Assert
        assert_eq!(ie.kind, IpcErrorKind::WorkbookDuplicateDefinition);
        assert_eq!(ie.message, "'x' is defined more than once");
    }

    #[test]
    fn transport_error_converts_kind_preserving_message() {
        // Arrange
        let te = idl_transport::TransportError::new(idl_transport::TransportErrorKind::Wifi, "timeout");

        // Act
        let ie: IpcError = te.into();

        // Assert
        assert_eq!(ie.kind, IpcErrorKind::Wifi);
        assert_eq!(ie.message, "timeout");
    }

    /// `From<idl_rs::store::import::ImportError> for IpcError` conversion
    /// tests — nested under `import` (rather than flat in `tests`) so the
    /// crate's own `import::` test filter (Task 9's compute rule) picks
    /// these up alongside `commands::import`'s tests.
    mod import {
        use super::*;

        /// Builds a core `ImportError` for `kind` — `ImportError`'s fields
        /// are `pub` (its own constructor is crate-private), so a test
        /// outside `idl_rs::store::import` still builds one directly via
        /// struct literal.
        fn import_error(kind: idl_rs::store::import::ImportErrorKind, message: &str) -> idl_rs::store::import::ImportError {
            idl_rs::store::import::ImportError { kind, message: message.to_string() }
        }

        #[test]
        fn import_error_io_kind_converts_to_io() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::Io, "disk full");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::Io);
            assert_eq!(ie.message, "disk full");
        }

        #[test]
        fn import_error_parse_invalid_magic_bytes_converts_to_parse_invalid_magic_bytes() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ParseInvalidMagicBytes, "not IDL0");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ParseInvalidMagicBytes);
        }

        #[test]
        fn import_error_parse_unsupported_schema_version_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ParseUnsupportedSchemaVersion, "schema 9");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ParseUnsupportedSchemaVersion);
        }

        #[test]
        fn import_error_parse_truncated_record_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ParseTruncatedRecord, "too short");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ParseTruncatedRecord);
        }

        #[test]
        fn import_error_collision_converts_to_import_collision_not_conflict() {
            // Arrange — R60: kept distinct from the cross-cutting `Conflict`
            // kind (same precedent R44 set for `Conflict` itself).
            let e = import_error(idl_rs::store::import::ImportErrorKind::Collision, "blob mismatch");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportCollision);
        }

        #[test]
        fn import_error_import_fit_malformed_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportFitMalformed, "bad crc");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportFitMalformed);
        }

        #[test]
        fn import_error_import_gpx_malformed_xml_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportGpxMalformedXml, "bad xml");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportGpxMalformedXml);
        }

        #[test]
        fn import_error_import_gpx_no_trackpoints_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportGpxNoTrackpoints, "no trkpt");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportGpxNoTrackpoints);
        }

        #[test]
        fn import_error_import_gpx_missing_lat_lon_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportGpxMissingLatLon, "missing lat");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportGpxMissingLatLon);
        }

        #[test]
        fn import_error_import_gpx_unparseable_lat_lon_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportGpxUnparseableLatLon, "not a number");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportGpxUnparseableLatLon);
        }

        #[test]
        fn import_error_import_csv_malformed_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportCsvMalformed, "bad header");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportCsvMalformed);
        }

        #[test]
        fn import_error_import_not_utf8_converts() {
            // Arrange
            let e = import_error(idl_rs::store::import::ImportErrorKind::ImportNotUtf8, "invalid utf8");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::ImportNotUtf8);
        }

        #[test]
        fn import_error_unknown_extension_folds_into_invalid_argument() {
            // Arrange — no dedicated C3 §2 row for this core-internal condition.
            let e = import_error(idl_rs::store::import::ImportErrorKind::UnknownExtension, "no importer for .xyz");

            // Act
            let ie: IpcError = e.into();

            // Assert
            assert_eq!(ie.kind, IpcErrorKind::InvalidArgument);
        }
    }
}
