//! Stamps the git commit this engine is built from into `IDL_RS_GIT_COMMIT`
//! (ruling Q4 of 2026-10-28): the engine identity persisted set results are
//! keyed by (`store::set_job::engine_identity`). `idl_rs::VERSION` never
//! changes, so it cannot invalidate a result computed by an older engine.
//!
//! An `IDL_RS_GIT_COMMIT` already in the environment wins (a packaging build
//! outside a checkout can supply one). A build with neither leaves the
//! variable empty, and such a build does not use the disk cache. Uncommitted
//! changes are not part of the identity.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn main() {
    println!("cargo:rerun-if-env-changed=IDL_RS_GIT_COMMIT");
    if let Ok(given) = std::env::var("IDL_RS_GIT_COMMIT") {
        println!("cargo:rustc-env=IDL_RS_GIT_COMMIT={given}");
        return;
    }

    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_default();
    println!("cargo:rustc-env=IDL_RS_GIT_COMMIT={commit}");

    // Re-run when HEAD moves: HEAD itself (a checkout), the branch ref it
    // points at (a commit), and packed-refs. Only files that exist are named,
    // since cargo treats a missing one as always changed.
    let mut watched = vec![git(&["rev-parse", "--git-path", "HEAD"]), git(&["rev-parse", "--git-path", "packed-refs"])];
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(git(&["rev-parse", "--git-path", &reference]));
    }
    for path in watched.into_iter().flatten() {
        if std::path::Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
