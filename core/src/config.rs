//! Shared reader for versioned JSON config files (`.idl0wb` workbooks,
//! `.idl0t` track artifacts). One typed error, one version-gate — so each
//! config format states its schema and reuses the read/parse logic.

use std::fmt;
use std::path::Path;

/// Discriminant for [`ConfigError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigErrorKind {
    /// Filesystem read failed.
    Io,
    /// JSON did not parse / required fields were missing.
    Parse,
    /// The file's schema version exceeds what this engine supports.
    UnsupportedVersion,
}

/// Error returned by [`parse_config`] / [`read_config`]. Freezed-free shape
/// (unit-enum `kind` + message), matching the `MathEvalError` precedent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub kind: ConfigErrorKind,
    pub message: String,
}

impl ConfigError {
    pub fn new(kind: ConfigErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ConfigError {}

/// A JSON config that carries a schema version this engine bounds.
pub trait VersionedConfig: serde::de::DeserializeOwned {
    /// Highest schema version this build understands.
    const SUPPORTED_VERSION: u32;
    /// Human label for error messages, e.g. `"workbook"`.
    const LABEL: &'static str;
    /// The version declared by this instance.
    fn version(&self) -> u32;
}

/// Parse a versioned config from JSON bytes, rejecting a too-new schema.
pub fn parse_config<T: VersionedConfig>(bytes: &[u8]) -> Result<T, ConfigError> {
    let value: T = serde_json::from_slice(bytes)
        .map_err(|e| ConfigError::new(ConfigErrorKind::Parse, format!("malformed {} JSON: {e}", T::LABEL)))?;
    if value.version() > T::SUPPORTED_VERSION {
        return Err(ConfigError::new(
            ConfigErrorKind::UnsupportedVersion,
            format!("{} version {} exceeds supported {}", T::LABEL, value.version(), T::SUPPORTED_VERSION),
        ));
    }
    Ok(value)
}

/// Read a versioned config from disk, then [`parse_config`].
pub fn read_config<T: VersionedConfig>(path: &Path) -> Result<T, ConfigError> {
    let bytes = std::fs::read(path)
        .map_err(|e| ConfigError::new(ConfigErrorKind::Io, format!("cannot read {}: {e}", path.display())))?;
    parse_config(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Demo {
        v: u32,
        #[serde(default)]
        name: String,
    }
    impl VersionedConfig for Demo {
        const SUPPORTED_VERSION: u32 = 2;
        const LABEL: &'static str = "demo";
        fn version(&self) -> u32 {
            self.v
        }
    }

    #[test]
    fn parse_config_accepts_supported_version() {
        // Act
        let d: Demo = parse_config(br#"{"v":2,"name":"ok"}"#).unwrap();

        // Assert
        assert_eq!(d.name, "ok");
    }

    #[test]
    fn parse_config_rejects_too_new_version() {
        // Act
        let err = parse_config::<Demo>(br#"{"v":3}"#).unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::UnsupportedVersion);
    }

    #[test]
    fn parse_config_reports_parse_error() {
        // Act
        let err = parse_config::<Demo>(b"not json").unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::Parse);
    }
}
