//! The rule that turns a git state into the engine identity persisted set
//! results are keyed by (ruling Q4 of 2026-10-28). `build.rs` includes this
//! file (`#[path]`) to stamp `IDL_RS_GIT_COMMIT`; it lives here so the rule
//! is tested. No dependencies: `build.rs` compiles it on its own.

/// FNV-1a, 64-bit: a stable hash with no dependency (std's `DefaultHasher` is
/// not promised to be stable across Rust releases).
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The engine identity of a build from `commit` whose crate files differ
/// from it by `diff` (`git diff HEAD -- .` output):
///
/// - no commit → `""` (no identity; the disk cache is off);
/// - a clean tree → the commit;
/// - uncommitted changes → `<commit>-dirty-<fnv64 of the diff>`, so every
///   distinct uncommitted engine is its own identity and a dirty build never
///   reuses another build's cached numbers;
/// - a diff that could not be taken (`None`) → `""`: it cannot be ruled clean.
pub fn identity_from(commit: Option<&str>, diff: Option<&[u8]>) -> String {
    match (commit, diff) {
        (None, _) | (_, None) => String::new(),
        (Some(c), Some(d)) if d.is_empty() => c.to_string(),
        (Some(c), Some(d)) => format!("{c}-dirty-{:016x}", fnv64(d)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_identity_clean_tree_the_commit() {
        // Act + Assert
        assert_eq!(identity_from(Some("abc123"), Some(b"")), "abc123");
    }

    #[test]
    fn engine_identity_dirty_tree_marked_and_distinct_per_diff() {
        // Act
        let a = identity_from(Some("abc123"), Some(b"-x\n+y\n"));
        let b = identity_from(Some("abc123"), Some(b"-x\n+z\n"));

        // Assert
        assert!(a.starts_with("abc123-dirty-"));
        assert_ne!(a, b);
        assert_ne!(a, "abc123");
    }

    #[test]
    fn engine_identity_no_commit_or_no_diff_no_identity() {
        // Act + Assert
        assert_eq!(identity_from(None, Some(b"")), "");
        assert_eq!(identity_from(Some("abc123"), None), "");
    }
}
