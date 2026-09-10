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

    /// The audited delete sites, as
    ///
    /// **Where the audit lives:** `runs/2026-09-10/DELETE-AUDIT.md` is in the
    /// *superproject* (`idl1-app`), not in this repository — `idl-rs` is a
    /// submodule of it. A reader with only `idl-rs` checked out cannot open
    /// the cited source, and nothing mechanically ties the two, so an entry
    /// added here must be added there in the same change by hand.
    ///
    /// The entries are
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
        // verified at the destination; `move_data_dir_with` removes a source
        // whose destination copy already verifies, resuming an interrupted
        // run; `prune_emptied_dirs` uses `remove_dir`, which fails on a
        // non-empty directory by construction, so it can only remove what the
        // move itself emptied.
        ("tauri/src/commands/app.rs", "copy_verify_delete"),
        ("tauri/src/commands/app.rs", "move_data_dir_with"),
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

    /// `true` when line `index` opens an inline `#[cfg(test)] mod … {` block
    /// — an attribute line followed by a `mod … {` line.
    ///
    /// Matching that shape specifically is what makes the skip exact without
    /// a brace-counting parser: `#[cfg(test)] mod name;` *declarations* end
    /// in `;` and are handled by [`test_only_module_files`] instead.
    fn opens_inline_test_module(lines: &[&str], index: usize) -> bool {
        if lines[index].trim() != "#[cfg(test)]" {
            return false;
        }
        match lines.get(index + 1) {
            Some(next) => {
                let next = next.trim();
                next.starts_with("mod ") && next.ends_with('{')
            }
            None => false,
        }
    }

    /// Scans one file, returning `(relative path, function)` for every delete
    /// call in non-test code.
    ///
    /// Each inline test module is skipped individually, from its
    /// `#[cfg(test)]` line to the next line that is exactly `}` in column 0
    /// — this repo's test modules are top-level items, so their closing brace
    /// is the only unindented `}` that can end them. Skipping each block
    /// rather than everything from the first one to end of file matters:
    /// several files here carry two test modules back to back, and one file
    /// could carry production code between them.
    ///
    /// Erring is one-directional by design. A `#[cfg(test)]` *function* is
    /// still scanned, so a delete inside one would be reported as
    /// unaudited — noisy, but visible. Nothing production is ever skipped.
    fn scan(root: &Path, path: &Path, text: &str) -> Vec<(String, String)> {
        let relative = path.strip_prefix(root).unwrap_or(path).display().to_string().replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        let mut hits = Vec::new();
        let mut current_fn = String::from("(module)");
        let mut in_test_module = false;
        for index in 0..lines.len() {
            let line = lines[index];
            if in_test_module {
                if line == "}" {
                    in_test_module = false;
                }
                continue;
            }
            if opens_inline_test_module(&lines, index) {
                in_test_module = true;
                continue;
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
    fn scan_production_code_between_two_inline_test_modules_is_still_scanned() {
        // Arrange — the shape that would break a "first test module to end of
        // file" skip: two test modules with a production function between
        // them, each containing a delete call.
        let source = "fn early() {\n    std::fs::remove_file(&p);\n}\n\
                      #[cfg(test)]\nmod tests {\n    fn helper() {\n        std::fs::remove_dir_all(&r);\n    }\n}\n\
                      fn between() {\n    std::fs::remove_dir(&d);\n}\n\
                      #[cfg(test)]\nmod more_tests {\n    fn other() {\n        std::fs::remove_file(&q);\n    }\n}\n";

        // Act
        let hits = scan(Path::new("/repo"), Path::new("/repo/core/src/thing.rs"), source);

        // Assert — both production sites, neither test site.
        assert_eq!(
            hits,
            vec![
                ("core/src/thing.rs".to_string(), "early".to_string()),
                ("core/src/thing.rs".to_string(), "between".to_string()),
            ]
        );
    }

    #[test]
    fn scan_a_cfg_test_mod_declaration_is_not_mistaken_for_an_inline_test_module() {
        // Arrange — `#[cfg(test)] mod name;` ends in `;`, pulls in a separate
        // file, and must not start a skip that swallows the rest of this one.
        let source = "#[cfg(test)]\nmod loopback_tests;\n\
                      fn after() {\n    std::fs::remove_file(&p);\n}\n";

        // Act
        let hits = scan(Path::new("/repo"), Path::new("/repo/transport/src/sync/mod.rs"), source);

        // Assert
        assert_eq!(hits, vec![("transport/src/sync/mod.rs".to_string(), "after".to_string())]);
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
