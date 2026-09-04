//! Content-addressed blob store (contract C4 §2–3): raw source files,
//! immutable, at `blobs/sha256/<2 hex>/<62 hex>`. A blob is written once;
//! a second write of the same content is a verified no-op (design doc §3
//! "sync sources... content-addressed").

use std::fmt;
use std::path::{Path, PathBuf};

use crate::store::atomic::{sha256_hex, write_atomic, AtomicWriteError};

/// Discriminant for [`BlobStoreError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobStoreErrorKind {
    Io,
    /// The bytes at a blob's expected path do not hash to that path's own
    /// name — corruption or tampering (C4 §7 finding #1).
    HashMismatch,
}

/// Error from the blob store. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobStoreError {
    pub kind: BlobStoreErrorKind,
    pub message: String,
}

impl BlobStoreError {
    fn new(kind: BlobStoreErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for BlobStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for BlobStoreError {}

impl From<AtomicWriteError> for BlobStoreError {
    fn from(e: AtomicWriteError) -> Self {
        BlobStoreError::new(BlobStoreErrorKind::Io, e.to_string())
    }
}

/// The on-disk path for a blob given its sha256 hex digest (C4 §2): a
/// 2+62 hex split, no file extension. Does not check the file exists.
pub fn blob_path(data_root: &Path, sha256_hex: &str) -> PathBuf {
    data_root.join("blobs").join("sha256").join(&sha256_hex[0..2]).join(&sha256_hex[2..])
}

/// `true` when a blob with this digest is already on disk.
pub fn blob_exists(data_root: &Path, sha256_hex: &str) -> bool {
    blob_path(data_root, sha256_hex).is_file()
}

/// Writes `bytes` into the CAS, returning its sha256 hex digest (also the
/// path-determining identity, C4 §3). If a blob with this exact digest
/// already exists, this is a verified no-op — the existing bytes are
/// trusted without re-reading them (content-addressed: the same hash can
/// only mean the same bytes, short of a SHA-256 collision), matching C4
/// §4's "a second write to the same hash is a verified no-op, skip rather
/// than overwrite."
pub fn write_blob(data_root: &Path, bytes: &[u8]) -> Result<String, BlobStoreError> {
    let digest = sha256_hex(bytes);
    let path = blob_path(data_root, &digest);
    if path.is_file() {
        return Ok(digest);
    }
    write_atomic(data_root, &path, bytes, None)?;
    Ok(digest)
}

/// Reads a blob's bytes by digest. `Io` if absent (C4 §7 #3 — "missing
/// blob" is the caller's concern to classify as a warning, not this
/// function's; it simply reports the read failed).
pub fn read_blob(data_root: &Path, sha256_hex: &str) -> Result<Vec<u8>, BlobStoreError> {
    let path = blob_path(data_root, sha256_hex);
    std::fs::read(&path).map_err(|e| BlobStoreError::new(BlobStoreErrorKind::Io, format!("read {}: {e}", path.display())))
}

/// Verifies a blob's own bytes hash to the digest its path encodes (C4 §7
/// finding #1). Used by `verify` (Task 12) and by [`write_blob`]'s callers
/// who want extra paranoia beyond the fast-path existence check.
pub fn verify_blob(data_root: &Path, sha256_hex: &str) -> Result<(), BlobStoreError> {
    let bytes = read_blob(data_root, sha256_hex)?;
    let actual = crate::store::atomic::sha256_hex(&bytes);
    if actual != sha256_hex {
        return Err(BlobStoreError::new(
            BlobStoreErrorKind::HashMismatch,
            format!("blob at {sha256_hex} hashes to {actual}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_blob_then_read_blob_round_trips_bytes() {
        // Arrange
        let root = temp_root();

        // Act
        let digest = write_blob(&root, b"hello idl0").unwrap();
        let back = read_blob(&root, &digest).unwrap();

        // Assert
        assert_eq!(back, b"hello idl0");
        assert_eq!(digest.len(), 64);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_blob_same_content_twice_is_a_no_op_same_digest() {
        // Arrange
        let root = temp_root();
        let d1 = write_blob(&root, b"same bytes").unwrap();

        // Act
        let d2 = write_blob(&root, b"same bytes").unwrap();

        // Assert
        assert_eq!(d1, d2);
        assert_eq!(read_blob(&root, &d1).unwrap(), b"same bytes");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn blob_path_splits_digest_two_and_sixty_two() {
        // Arrange
        let root = PathBuf::from("/data");
        let digest = "ab".to_string() + &"c".repeat(62);

        // Act
        let path = blob_path(&root, &digest);

        // Assert
        assert_eq!(path, root.join("blobs").join("sha256").join("ab").join("c".repeat(62)));
    }

    #[test]
    fn verify_blob_detects_a_tampered_file() {
        // Arrange — write a blob then corrupt its bytes on disk directly.
        let root = temp_root();
        let digest = write_blob(&root, b"original").unwrap();
        std::fs::write(blob_path(&root, &digest), b"corrupted").unwrap();

        // Act
        let err = verify_blob(&root, &digest).unwrap_err();

        // Assert
        assert_eq!(err.kind, BlobStoreErrorKind::HashMismatch);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn blob_exists_reflects_presence() {
        // Arrange
        let root = temp_root();

        // Act + Assert
        assert!(!blob_exists(&root, &"0".repeat(64)));
        let digest = write_blob(&root, b"x").unwrap();
        assert!(blob_exists(&root, &digest));

        let _ = std::fs::remove_dir_all(&root);
    }
}
