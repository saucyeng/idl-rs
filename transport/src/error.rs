//! Typed transport failures. Every fallible transport operation returns
//! [`TransportError`]; free-form `Err(String)` is not allowed (CLAUDE.md §5).

use std::fmt;

/// Which subsystem failed. Coarse on purpose: the UI routes on this, and the
/// message carries the detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportErrorKind {
    /// BLE scan, connect, GATT or transfer failure.
    Ble,
    /// WiFi network binding or HTTP transfer failure against the device.
    Wifi,
    /// Config push rejected or malformed (SPEC §8).
    Config,
    /// LAN peer sync failure (pairing, manifest, blob transfer).
    Sync,
}

/// A transport failure: a kind the UI can route on plus a human-readable
/// message. Serialisable so it crosses IPC unchanged (contract C3).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TransportError {
    /// Subsystem that failed.
    pub kind: TransportErrorKind,
    /// What went wrong, for the user. No stack traces.
    pub message: String,
}

impl TransportError {
    /// Builds an error of `kind` with `message`.
    pub fn new(kind: TransportErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for TransportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_error_display_carries_kind_and_message() {
        // Arrange
        let err = TransportError::new(TransportErrorKind::Ble, "device not found");

        // Act
        let text = err.to_string();

        // Assert
        assert_eq!(text, "Ble: device not found");
    }

    #[test]
    fn transport_error_serialises_kind_as_snake_case() {
        // Arrange
        let err = TransportError::new(TransportErrorKind::Wifi, "timeout");

        // Act
        let json = serde_json::to_string(&err).unwrap();

        // Assert
        assert_eq!(json, r#"{"kind":"wifi","message":"timeout"}"#);
    }
}
