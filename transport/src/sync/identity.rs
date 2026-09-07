//! This device's own sync identity — `identity.json` (ruling R105 item 1,
//! L11 Task 13). Holds `{ peer_id, name }`: `peer_id` is a `uuid` v4 minted
//! once on first read and never regenerated (regenerating it would orphan
//! every pairing this instance already has); `name` defaults to the OS
//! hostname when readable, else [`DEFAULT_NAME`], and the user may change
//! it (`set_sync_device_name`, C3 §3.9).
//!
//! Lives beside the peer file (`peers.json`) in `app_config_dir()`, not in
//! `settings.json` — that file is C4's user-facing store and syncs
//! semantics this file must not (ruling R105). `identity_path` is passed in
//! by the caller (never resolved from a Tauri app handle here), matching
//! `pairing::save_peers`' "no Tauri dependency for a path" rule (CLAUDE.md
//! §2); outside `<data>` so it never syncs (ruling R88).

use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{TransportError, TransportErrorKind};

use super::pairing::write_json_atomic;

/// The display name used when the OS hostname cannot be read (ruling R105).
pub const DEFAULT_NAME: &str = "idl1";

/// This device's sync identity — `identity.json`'s exact shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Identity {
    /// Stable identifier for this app instance, minted once
    /// (`uuid::Uuid::new_v4()`) and never regenerated.
    pub peer_id: String,
    /// Display name advertised to peers and shown in their peer lists.
    pub name: String,
}

/// Loads `identity_path`, minting and persisting a fresh [`Identity`] if the
/// file does not yet exist. `hostname` seeds the minted name — the caller
/// resolves the OS hostname (this crate never does; see the module doc) and
/// passes it in; an absent or blank hostname falls back to [`DEFAULT_NAME`].
///
/// A present-but-corrupt or otherwise unreadable file is a typed
/// [`TransportError`], **never** a silently minted replacement — a fresh id
/// here would orphan every peer this instance is already paired with (this
/// task's "Critical behaviour").
pub fn load_or_create(identity_path: &Path, hostname: Option<&str>) -> Result<Identity, TransportError> {
    match std::fs::read(identity_path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
            TransportError::new(
                TransportErrorKind::Sync,
                format!("malformed identity file {}: {e}", identity_path.display()),
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let name = hostname.map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_NAME).to_string();
            let identity = Identity { peer_id: Uuid::new_v4().to_string(), name };
            write_json_atomic(identity_path, &identity)?;
            Ok(identity)
        }
        Err(e) => Err(TransportError::new(
            TransportErrorKind::Sync,
            format!("reading identity file {}: {e}", identity_path.display()),
        )),
    }
}

/// Renames this device: writes a copy of `identity` with `name` replaced,
/// keeping `peer_id` unchanged, and persists it to `identity_path`. Used by
/// `set_sync_device_name` (C3 §3.9).
pub fn set_name(identity_path: &Path, identity: &Identity, name: String) -> Result<Identity, TransportError> {
    let updated = Identity { peer_id: identity.peer_id.clone(), name };
    write_json_atomic(identity_path, &updated)?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("idl-transport-test-identity-{}-{}", name, Uuid::new_v4()))
    }

    #[test]
    fn load_or_create_absent_file_mints_and_persists_an_identity() {
        // Arrange
        let path = tmp_path("absent");

        // Act
        let minted = load_or_create(&path, Some("workshop-pc")).unwrap();
        let reloaded: Identity = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();

        // Assert
        assert!(Uuid::parse_str(&minted.peer_id).is_ok());
        assert_eq!(minted.name, "workshop-pc");
        assert_eq!(reloaded, minted);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_or_create_present_file_is_read_unchanged() {
        // Arrange
        let path = tmp_path("present");
        let existing = Identity { peer_id: "fixed-peer-id".to_string(), name: "Pit Tablet".to_string() };
        write_json_atomic(&path, &existing).unwrap();

        // Act
        let loaded = load_or_create(&path, Some("some-other-host")).unwrap();

        // Assert — the hostname hint is ignored once a file already exists.
        assert_eq!(loaded, existing);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_or_create_corrupt_file_is_a_typed_error_and_never_overwrites() {
        // Arrange
        let path = tmp_path("corrupt");
        std::fs::write(&path, b"{ not json").unwrap();

        // Act
        let result = load_or_create(&path, Some("workshop-pc"));

        // Assert
        assert!(matches!(result, Err(e) if e.kind == TransportErrorKind::Sync));
        // The corrupt bytes are untouched — a fresh id was never minted.
        assert_eq!(std::fs::read(&path).unwrap(), b"{ not json");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_or_create_absent_hostname_falls_back_to_default_name() {
        // Arrange
        let path = tmp_path("no-hostname");

        // Act
        let minted = load_or_create(&path, None).unwrap();

        // Assert
        assert_eq!(minted.name, DEFAULT_NAME);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_or_create_blank_hostname_falls_back_to_default_name() {
        // Arrange
        let path = tmp_path("blank-hostname");

        // Act
        let minted = load_or_create(&path, Some("   ")).unwrap();

        // Assert
        assert_eq!(minted.name, DEFAULT_NAME);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn set_name_persists_the_new_name_and_keeps_peer_id() {
        // Arrange
        let path = tmp_path("set-name");
        let original = Identity { peer_id: "fixed-peer-id".to_string(), name: "idl1".to_string() };
        write_json_atomic(&path, &original).unwrap();

        // Act
        let updated = set_name(&path, &original, "Pit Wall Laptop".to_string()).unwrap();
        let reloaded: Identity = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();

        // Assert
        assert_eq!(updated.peer_id, original.peer_id);
        assert_eq!(updated.name, "Pit Wall Laptop");
        assert_eq!(reloaded, updated);

        let _ = std::fs::remove_file(&path);
    }
}
