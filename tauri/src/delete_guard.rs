//! Build-time guard over every place the engine deletes something (ruling
//! R196). Test-only: it scans `core/src`, `transport/src` and `tauri/src` as
//! *text* for `remove_dir_all`, `remove_dir(` and `remove_file(` outside
//! `#[cfg(test)]`/`#[test]` code, and asserts the resulting set of
//! `(file, function)` pairs is exactly the allowlist below — the one audited
//! in `runs/2026-09-10/DELETE-AUDIT.md`.
//!
//! The point is not that deleting is forbidden. It is that a *new* delete
//! site cannot appear without someone editing this list, which forces the
//! audit question ("what does this delete, and when could it be wrong?") to
//! be answered before the code lands rather than after a user's library is
//! gone. Renaming an allowlisted function fails the test too, deliberately:
//! the audit names functions, so the audit entry has to move with it.
//!
//! Text, not AST: a `syn` dependency for a guard whose whole job is to be
//! trivially auditable would be worse. The cost is that the `#[cfg(test)]`
//! skipping is brace-counting rather than parsing — good enough for a repo
//! whose test modules are all `#[cfg(test)] mod tests { … }`.
//!
//! One known sharp edge: a delete call inside a nested `fn` is attributed to
//! that nested name, which is usually meaningless in an allowlist. The test
//! still fails — it just names something an auditor cannot find — so the fix
//! is to hoist the helper out, not to allowlist the inner name.

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    /// The audited delete sites (`runs/2026-09-10/DELETE-AUDIT.md`), as
    /// `(path relative to the idl-rs workspace root, enclosing function)`.
    /// One comment per entry says what it deletes and why that is safe.
    const ALLOWLIST: &[(&str, &str)] = &[
        // (a) tmp/ scratch — staging, never a final path.
        // `write_atomic`: drops its own `tmp/<uuid>` scratch file when a
        // rename loses a race, before returning the error.
        ("core/src/store/atomic.rs", "write_atomic"),
        // `resolve_quarantine`/`move_file`: the explicit, user-triggered
        // resolution of an existing quarantine entry (C4 §2) — Discard
        // removes the quarantined payload, Restore moves it out.
        ("core/src/store/quarantine.rs", "move_file"),
        ("core/src/store/quarantine.rs", "resolve_quarantine"),
        // Sync verify: writes then removes a `tmp/sync-verify-<uuid>` file.
        ("core/src/store/sync/apply.rs", "read_data_parquet_versions_from_bytes"),
        // Workbook merge install: removes the *old*-named `.idl1wb` only when
        // a peer's `workbook_id` wins the rename (C4 §6).
        ("core/src/store/sync/apply.rs", "install_workbook"),
        // `download_via`: removes/renames only the `tmp/<uuid>` download
        // scratch file on its way into `blobs/`.
        ("tauri/src/commands/device.rs", "download_via"),
        // Resumable download `.part` files under `tmp/`, dropped on a cap
        // breach and again once the bytes are complete (R102).
        ("transport/src/sync/client.rs", "download_item_with_cap"),
        ("transport/src/sync/client.rs", "pull_and_install"),
        // `write_json_atomic`: tmp-file write/rename/cleanup for
        // peers.json/identity.json, which live outside `<data>`.
        ("transport/src/sync/pairing.rs", "write_json_atomic"),
        // (b) one session's derived/ or data.parquet during a rebuild.
        // `remove_session_derived`: `sessions/<id>/derived/` only, after a
        // `data.parquet` rebuild invalidates the cached derived files.
        ("core/src/store/derived.rs", "remove_session_derived"),
        // `ImportPlan::Regenerate`: deletes `sessions/<id>/data.parquet`
        // immediately before rewriting it (C1 §4.3).
        ("core/src/store/import.rs", "finish_import"),
        // (c) the inbox: removes the dropped file only *after* a successful
        // import (R191); a failed import renames it into `inbox/failed/`.
        ("tauri/src/inbox.rs", "import_one"),
        // (d) catalog.sqlite and its -wal/-shm sidecars — a rebuildable
        // index, never `blobs/` or `sessions/`.
        ("core/src/store/catalog.rs", "rebuild_catalog"),
        // (g) an explicit single-profile delete, no-op if absent.
        ("core/src/store/profile.rs", "delete"),
        // `delete_track`: one `tracks/<id>.idl0t`, the explicit "delete
        // track" action, path-validated first. Added to the audit on
        // 2026-09-10 — this scan is what found it missing from it.
        ("core/src/track_artifact/write.rs", "delete_track"),
        // (h) `move_data_dir` (C3 §3.10 as amended by R197: move, not copy).
        // `copy_verify_delete` removes a source file only after its copy has
        // verified at the destination; `move_data_dir_via` removes a source
        // whose destination copy already verifies, resuming an interrupted
        // run; `prune_emptied_dirs` uses `remove_dir`, which fails on a
        // non-empty directory by construction, so it can only remove what the
        // move itself emptied.
        ("tauri/src/commands/app.rs", "copy_verify_delete"),
        ("tauri/src/commands/app.rs", "move_data_dir_via"),
        ("tauri/src/commands/app.rs", "prune_emptied_dirs"),
        // (e) the one and only blob-deleting path: `delete_session` with
        // `delete_blob: true`, after confirming no other session references
        // the same content hash.
        ("tauri/src/commands/catalog.rs", "delete_session_via"),
    ];

    /// The idl-rs workspace root (`tauri/`'s parent).
    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("tauri/ has a parent").to_path_buf()
    }

    /// Every `.rs` file under `dir`, recursively, sorted.
    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// The function name a `fn` line declares, if it declares one.
    fn function_name(line: &str) -> Option<String> {
        let trimmed = line.trim_start();
        let after = trimmed
            .strip_prefix("pub(crate) ")
            .or_else(|| trimmed.strip_prefix("pub(super) "))
            .or_else(|| trimmed.strip_prefix("pub "))
            .unwrap_or(trimmed);
        let after = after.strip_prefix("const ").unwrap_or(after);
        let after = after.strip_prefix("async ").unwrap_or(after);
        let after = after.strip_prefix("unsafe ").unwrap_or(after);
        let after = after.strip_prefix("extern \"C\" ").unwrap_or(after);
        let rest = after.strip_prefix("fn ")?;
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    /// `true` when the line calls one of the three delete APIs the audit
    /// covers. `remove_dir_all` needs no paren: it is never a local name.
    fn is_delete_call(line: &str) -> bool {
        line.contains("remove_dir_all") || line.contains("remove_dir(") || line.contains("remove_file(")
    }

    /// The file a `#[cfg(test)] mod name;` declaration in `declaring_file`
    /// pulls in — that whole file is test code. Both layouts are checked
    /// (`name.rs` beside the declaring module, and `name/mod.rs`).
    fn test_only_module_files(declaring_file: &Path, text: &str) -> Vec<PathBuf> {
        let dir = match declaring_file.file_name().and_then(|n| n.to_str()) {
            // `foo/mod.rs` declares siblings inside `foo/`; `foo.rs` declares
            // them inside `foo/`, a directory named after the file.
            Some("mod.rs") | Some("lib.rs") | Some("main.rs") => declaring_file.parent().map(|p| p.to_path_buf()),
            _ => declaring_file.parent().map(|p| p.join(declaring_file.file_stem().unwrap_or_default())),
        };
        let Some(dir) = dir else { return Vec::new() };
        let lines: Vec<&str> = text.lines().collect();
        let mut out = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if line.trim() != "#[cfg(test)]" {
                continue;
            }
            let Some(next) = lines.get(index + 1) else { continue };
            let next = next.trim();
            let Some(rest) = next.strip_prefix("mod ") else { continue };
            let Some(name) = rest.strip_suffix(';') else { continue };
            out.push(dir.join(format!("{name}.rs")));
            out.push(dir.join(name).join("mod.rs"));
        }
        out
    }

    /// Index of the first line of the file's trailing `#[cfg(test)] mod … {`
    /// block, if it has one. Everything from there on is test code.
    ///
    /// Matching the *inline module* form specifically (an attribute line
    /// followed by a `mod … {` line) is what makes this exact without a
    /// brace-counting parser: `#[cfg(test)] mod name;` declarations end in
    /// `;` and are handled by [`test_only_module_files`] instead, and this
    /// repo places its inline test module last in every file.
    fn test_module_start(text: &str) -> Option<usize> {
        let lines: Vec<&str> = text.lines().collect();
        lines.iter().enumerate().find_map(|(index, line)| {
            if line.trim() != "#[cfg(test)]" {
                return None;
            }
            let next = lines.get(index + 1)?.trim();
            if next.starts_with("mod ") && next.ends_with('{') {
                Some(index)
            } else {
                None
            }
        })
    }

    /// Scans one file, returning `(relative path, function)` for every delete
    /// call in non-test code.
    fn scan(root: &Path, path: &Path, text: &str) -> Vec<(String, String)> {
        let relative = path.strip_prefix(root).unwrap_or(path).display().to_string().replace('\\', "/");
        let end = test_module_start(text).unwrap_or(usize::MAX);
        let mut hits = Vec::new();
        let mut current_fn = String::from("(module)");
        for (index, line) in text.lines().enumerate() {
            if index >= end {
                break;
            }
            if line.trim_start().starts_with("//") {
                continue;
            }
            if let Some(name) = function_name(line) {
                current_fn = name;
            }
            if is_delete_call(line) {
                hits.push((relative.clone(), current_fn.clone()));
            }
        }
        hits
    }

    #[test]
    fn every_non_test_delete_site_is_one_the_delete_audit_already_covers() {
        // Arrange
        let root = workspace_root();
        let mut files = Vec::new();
        for crate_dir in ["core/src", "transport/src", "tauri/src"] {
            rust_files(&root.join(crate_dir), &mut files);
        }
        assert!(files.len() > 20, "the scanner found almost no source files — it is looking in the wrong place");
        let expected: BTreeSet<(String, String)> =
            ALLOWLIST.iter().map(|(f, n)| ((*f).to_string(), (*n).to_string())).collect();
        let sources: Vec<(PathBuf, String)> =
            files.iter().filter_map(|p| std::fs::read_to_string(p).ok().map(|t| (p.clone(), t))).collect();
        let test_only: BTreeSet<PathBuf> =
            sources.iter().flat_map(|(path, text)| test_only_module_files(path, text)).collect();

        // Act
        let found: BTreeSet<(String, String)> = sources
            .iter()
            .filter(|(path, _)| !test_only.contains(path))
            .flat_map(|(path, text)| scan(&root, path, text))
            .collect();

        // Assert
        let added: Vec<_> = found.difference(&expected).collect();
        let gone: Vec<_> = expected.difference(&found).collect();
        assert!(
            added.is_empty() && gone.is_empty(),
            "the set of non-test delete sites changed.\n\
             New, unaudited sites (add them to runs/2026-09-10/DELETE-AUDIT.md first, then to ALLOWLIST): {added:#?}\n\
             Allowlisted sites that no longer exist (remove them from ALLOWLIST): {gone:#?}",
        );
    }
}
