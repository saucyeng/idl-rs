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
}
