//! Shared id validation and safe path joining for every id-addressed sync
//! route (workbook/session/track/profile), ruling R100. `review-task8`'s
//! Critical: `transport`'s old `is_valid_id` rejected `/`, `\` and `..` but
//! not a Windows drive-relative segment (`C:evil`) — and `PathBuf::join`
//! on a component that has a prefix but no root **discards the base path
//! entirely** (confirmed: `PathBuf::from(r"C:\data").join("tracks").join(
//! format!("{}.idl0t", "C:evil"))` yields `C:evil.idl0t`, outside
//! `data_root`). One validator, here in `core`, used by both `idl-transport`
//! (routing) and `idl-rs`'s own `store::sync::apply` (install), so no
//! id-addressed path is ever built from an unchecked string in either
//! crate — plus [`safe_join`], a second, independent layer that checks the
//! *result* of a join is still inside `data_root` regardless of whether the
//! id that produced it was validated at all.

use std::path::{Component, Path, PathBuf};

/// The shape of id this module accepts, one per syncable class (SPEC
/// references below) — allow-list, not deny-list: every accepted id is
/// ASCII, has no path separator, no `..`, no `:`, and no control byte by
/// construction, since each shape permits only `[0-9a-f-]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdClass {
    /// Workbook, Track, and Profile ids: a UUID's canonical textual form —
    /// 36 ASCII characters, lowercase hex in `8-4-4-4-12` groups separated
    /// by `-` (IDL0_SPEC: workbook front matter `id` is "UUIDv4"; track/
    /// profile ids are documented as `"uuid"`). Uppercase hex is rejected
    /// (canonical form is lowercase; accepting both would let two strings
    /// name the same id, which `find_local_workbook`'s exact string
    /// comparison — and every other id equality check in this crate —
    /// assumes cannot happen).
    Uuid,
    /// Session ids: lowercase hex, no separators. IDL0_SPEC §24: a
    /// device-sourced session id is "the 16-byte header UUID as 32
    /// lowercase hex chars (no dashes)"; a non-device session id (C4 §3)
    /// is "the first 16 lowercase hex characters of `blob_sha256`",
    /// extended two characters at a time on a catalog collision, up to the
    /// full 64-character digest. Any even length from 16 to 64 lowercase
    /// hex characters is accepted, covering every id either path can
    /// produce.
    Session,
}

/// `true` if `id` matches `class`'s shape exactly. This is the only check
/// most callers need — see [`safe_join`] for the second, independent
/// layer applied after path construction.
pub fn is_valid_id(id: &str, class: IdClass) -> bool {
    match class {
        IdClass::Uuid => is_uuid_shape(id),
        IdClass::Session => is_session_shape(id),
    }
}

/// `8-4-4-4-12` lowercase hex groups, `-` at exactly positions 8/13/18/23,
/// 36 ASCII bytes total.
fn is_uuid_shape(id: &str) -> bool {
    let b = id.as_bytes();
    if b.len() != 36 || !id.is_ascii() {
        return false;
    }
    for (i, &c) in b.iter().enumerate() {
        let want_dash = matches!(i, 8 | 13 | 18 | 23);
        if want_dash {
            if c != b'-' {
                return false;
            }
        } else if !c.is_ascii_hexdigit() || c.is_ascii_uppercase() {
            return false;
        }
    }
    true
}

/// Lowercase hex, even length, `16..=64` characters.
fn is_session_shape(id: &str) -> bool {
    let len = id.len();
    if !(16..=64).contains(&len) || len % 2 != 0 || !id.is_ascii() {
        return false;
    }
    id.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Collapses `.`/`..` components lexically (no filesystem access — the
/// target of a join may not exist yet, e.g. a brand-new workbook), so the
/// result can be compared against a similarly-collapsed `data_root`
/// without requiring either path to exist. A `..` past the start of the
/// path (nothing left to pop) is simply dropped, matching how a real
/// filesystem would resolve it at the root.
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => out.push(component.as_os_str()),
        }
    }
    out
}

/// Joins `data_root` with `segments` (each pushed as its own path
/// component — the caller does not pre-format a `/`-joined string) and
/// returns the joined path only if it is still lexically inside
/// `data_root` afterwards (R100's "belt and braces": independent of
/// whether `segments` were validated by [`is_valid_id`] at all — this is
/// what actually closes the Windows drive-relative-join bug, since
/// `PathBuf::join` silently discarding `data_root` for a drive-relative
/// segment means the *joined* path no longer starts with `data_root` once
/// both sides are lexically normalised). `None` on escape; the caller
/// answers `404`, never `403` (R100 — nothing about what exists outside
/// `data_root` should leak).
pub fn safe_join(data_root: &Path, segments: &[&str]) -> Option<PathBuf> {
    let mut joined = data_root.to_path_buf();
    for segment in segments {
        joined = joined.join(segment);
    }
    let normalized_joined = normalize_lexically(&joined);
    let normalized_root = normalize_lexically(data_root);
    normalized_joined.starts_with(&normalized_root).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_id_uuid_class_accepts_a_canonical_lowercase_uuid() {
        // Arrange
        let id = "550e8400-e29b-41d4-a716-446655440000";

        // Act
        let result = is_valid_id(id, IdClass::Uuid);

        // Assert
        assert!(result);
    }

    #[test]
    fn is_valid_id_uuid_class_rejects_a_windows_drive_relative_prefix() {
        // Arrange
        let id = "C:foo";

        // Act
        let result = is_valid_id(id, IdClass::Uuid);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_session_class_rejects_a_windows_drive_relative_prefix() {
        // Arrange
        let id = "C:foo";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_rejects_a_windows_verbatim_unc_style_prefix() {
        // Arrange
        let id = r"\\?\C:\evil";

        // Act
        let result = is_valid_id(id, IdClass::Uuid);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_rejects_a_parent_dir_traversal() {
        // Arrange
        let id = r"..\..";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_rejects_a_bare_dot() {
        // Arrange
        let id = ".";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_rejects_an_over_long_id() {
        // Arrange: one character past the 64-char session ceiling, and
        // nowhere near the 36-char UUID shape either.
        let id = "a".repeat(65);

        // Act
        let uuid_result = is_valid_id(&id, IdClass::Uuid);
        let session_result = is_valid_id(&id, IdClass::Session);

        // Assert
        assert!(!uuid_result);
        assert!(!session_result);
    }

    #[test]
    fn is_valid_id_session_class_accepts_a_sixteen_char_hex_prefix() {
        // Arrange: the minimum, un-extended blob-hash-prefix session id.
        let id = "0123456789abcdef";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(result);
    }

    #[test]
    fn is_valid_id_session_class_accepts_a_thirty_two_char_device_uuid_hex() {
        // Arrange: SPEC §24's device-sourced shape, 32 lowercase hex, no dashes.
        let id = "0102030405060708090a0b0c0d0e0f10";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(result);
    }

    #[test]
    fn is_valid_id_session_class_rejects_uppercase_hex() {
        // Arrange
        let id = "0123456789ABCDEF";

        // Act
        let result = is_valid_id(id, IdClass::Session);

        // Assert
        assert!(!result);
    }

    #[test]
    fn is_valid_id_uuid_class_rejects_uppercase_hex() {
        // Arrange
        let id = "550E8400-E29B-41D4-A716-446655440000";

        // Act
        let result = is_valid_id(id, IdClass::Uuid);

        // Assert
        assert!(!result);
    }

    #[test]
    fn safe_join_a_normal_id_stays_inside_data_root() {
        // Arrange
        let root = Path::new(r"C:\data");
        let id = "550e8400-e29b-41d4-a716-446655440000";

        // Act
        let result = safe_join(root, &["tracks", &format!("{id}.idl0t")]);

        // Assert
        assert_eq!(result, Some(root.join("tracks").join(format!("{id}.idl0t"))));
    }

    #[test]
    fn safe_join_catches_a_drive_relative_escape_the_shape_check_would_also_catch() {
        // Arrange: this segment would already fail `is_valid_id` — safe_join
        // is checked here on its own, called directly (as R100 requires: a
        // second, independent layer, not merely a restatement of the shape
        // check).
        let root = Path::new(r"C:\data");

        // Act
        let result = safe_join(root, &["tracks", "C:evil.idl0t"]);

        // Assert
        assert_eq!(result, None);
    }

    #[test]
    fn safe_join_catches_a_parent_dir_escape_the_shape_check_alone_might_miss_if_ever_skipped() {
        // Arrange: a hypothetical caller that built its segments by
        // concatenation rather than validated components — safe_join is
        // the layer that still holds even when a segment reaches it
        // unchecked (the belt-and-braces R100 asks for).
        let root = Path::new(r"C:\data");

        // Act
        let result = safe_join(root, &["sessions", "..", "..", "etc", "passwd"]);

        // Assert
        assert_eq!(result, None);
    }

    #[test]
    fn safe_join_a_valid_session_id_multi_segment_path_stays_inside_data_root() {
        // Arrange
        let root = Path::new(r"C:\data");
        let id = "0123456789abcdef";
        let hash = "aa".repeat(32);

        // Act
        let result = safe_join(root, &["sessions", id, "derived", &hash]);

        // Assert
        assert!(result.is_some());
        assert!(result.unwrap().starts_with(root));
    }
}
