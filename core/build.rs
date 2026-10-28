//! Stamps the engine identity persisted set results are keyed by into
//! `IDL_RS_GIT_COMMIT` (ruling Q4 of 2026-10-28;
//! `store::set_job::engine_identity`). `idl_rs::VERSION` never changes, so it
//! cannot invalidate a result computed by an older engine.
//!
//! The identity is the git commit this crate is built from. When this crate's
//! files differ from that commit (a dev build with uncommitted changes), it is
//! `<commit>-dirty-<fnv64 of the diff>`: every distinct uncommitted engine is
//! a distinct identity, so a dirty build never reuses another build's cached
//! numbers. `src/` is watched, so an edit re-runs this script and re-hashes.
//!
//! An `IDL_RS_GIT_COMMIT` already in the environment wins (a packaging build
//! outside a checkout can supply one). A build with neither leaves the
//! variable empty, and such a build does not use the disk cache.

use std::process::Command;

fn git(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then_some(out.stdout)
}

fn git_text(args: &[&str]) -> Option<String> {
    let text = String::from_utf8(git(args)?).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}

#[path = "src/store/engine_identity.rs"]
#[allow(dead_code)]
mod engine_identity;

fn main() {
    println!("cargo:rerun-if-env-changed=IDL_RS_GIT_COMMIT");
    if let Ok(given) = std::env::var("IDL_RS_GIT_COMMIT") {
        println!("cargo:rustc-env=IDL_RS_GIT_COMMIT={given}");
        return;
    }

    let commit = git_text(&["rev-parse", "HEAD"]);
    let diff = commit.as_ref().and_then(|_| git(&["diff", "HEAD", "--", "."]));
    let identity = engine_identity::identity_from(commit.as_deref(), diff.as_deref());
    println!("cargo:rustc-env=IDL_RS_GIT_COMMIT={identity}");

    // Re-run when HEAD moves (HEAD itself, the branch ref it points at,
    // packed-refs) and when any source file changes (the dirty hash). Only
    // paths that exist are named: cargo treats a missing one as always
    // changed.
    let mut watched = vec![
        Some("src".to_string()),
        Some("Cargo.toml".to_string()),
        git_text(&["rev-parse", "--git-path", "HEAD"]),
        git_text(&["rev-parse", "--git-path", "packed-refs"]),
        git_text(&["rev-parse", "--git-path", "index"]),
    ];
    if let Some(reference) = git_text(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(git_text(&["rev-parse", "--git-path", &reference]));
    }
    for path in watched.into_iter().flatten() {
        if std::path::Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
