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
