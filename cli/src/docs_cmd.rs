//! The `docs` command group: regenerating documentation from the engine's
//! own catalogs (ruling R222 item 1).
//!
//! One action today, `docs workbook`, which renders
//! `docs/WORKBOOK-REFERENCE.md` — the generated builtin catalog and
//! retired-name table from [`idl_rs::docs`], followed by the curated
//! Markdown files under `--src` in filename order.
//!
//! CI runs this and then `git diff --exit-code`, so the output has to be
//! byte-stable for a given engine and curated source set: the curated files
//! are read in sorted order (never the filesystem's own), their bodies are
//! spliced verbatim, and the renderer itself is pure. The file is written
//! with `\n` line endings on every platform for the same reason — a
//! Windows-generated file and a Linux-generated one must be identical bytes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Subcommand;

use idl_rs::docs::{render_workbook_reference, CuratedSection};

use crate::envelope::{emit_bulk, CliError, ErrorKind};

/// The `docs` sub-actions.
#[derive(Subcommand)]
pub enum DocsAction {
    /// Regenerate the workbook reference: every math builtin grouped by
    /// category, the retired-name table, then the curated sections.
    Workbook {
        /// File to write. Overwritten wholesale.
        #[arg(long)]
        out: PathBuf,
        /// Directory of curated Markdown sections, appended in filename
        /// order after the generated half. Missing is not an error — the
        /// generated half alone is a valid reference.
        #[arg(long, default_value = "docs/reference-src")]
        src: PathBuf,
    },
}

/// Runs one `docs` action. Bulk-shaped: on success it writes its artifact
/// and prints one line to stderr; on failure it emits an error envelope.
pub fn run(action: DocsAction) -> ExitCode {
    match action {
        DocsAction::Workbook { out, src } => emit_bulk("docs workbook", workbook(&out, &src)),
    }
}

/// Renders the reference from `src` and writes it to `out`.
fn workbook(out: &Path, src: &Path) -> Result<(), CliError> {
    let curated = read_curated(src)?;
    let rendered = render_workbook_reference(&curated);

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::io(format!("creating {}: {e}", parent.display())))?;
        }
    }
    // `as_bytes`, not a text write: the string already holds `\n` only, and
    // going through `write` keeps it that way on Windows too.
    std::fs::write(out, rendered.as_bytes())
        .map_err(|e| CliError::io(format!("writing {}: {e}", out.display())))?;

    eprintln!("wrote {} ({} curated sections)", out.display(), curated.len());
    Ok(())
}

/// Reads every `*.md` under `src`, sorted by file name.
///
/// Sorting is on the file name rather than the full path so the curated
/// files' own numeric prefixes (`10-…`, `20-…`) decide the document's
/// section order, and so the order does not depend on where the directory
/// happens to sit.
fn read_curated(src: &Path) -> Result<Vec<CuratedSection>, CliError> {
    if !src.is_dir() {
        return Ok(Vec::new());
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(src)
        .map_err(|e| CliError::io(format!("reading {}: {e}", src.display())))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|e| e == "md"))
        .collect();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut sections = Vec::with_capacity(files.len());
    for path in files {
        let body = std::fs::read_to_string(&path)
            .map_err(|e| CliError::new(ErrorKind::Io, format!("reading {}: {e}", path.display())))?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        // A CRLF checkout must not produce different bytes than an LF one:
        // the generated file is compared byte for byte by CI.
        sections.push(CuratedSection { name, body: body.replace("\r\n", "\n") });
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_dir() -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("idl-rs-docs-cmd-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn read_curated_on_a_missing_directory_returns_no_sections() {
        // Arrange
        let missing = temp_dir().join("not-here");

        // Act
        let sections = read_curated(&missing).unwrap();

        // Assert
        assert!(sections.is_empty());
    }

    #[test]
    fn read_curated_orders_sections_by_file_name_not_directory_order() {
        // Arrange
        let dir = temp_dir();
        std::fs::write(dir.join("30-c.md"), b"## C\n").unwrap();
        std::fs::write(dir.join("10-a.md"), b"## A\n").unwrap();
        std::fs::write(dir.join("20-b.md"), b"## B\n").unwrap();

        // Act
        let sections = read_curated(&dir).unwrap();

        // Assert
        let names: Vec<&str> = sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["10-a", "20-b", "30-c"]);
    }

    #[test]
    fn read_curated_ignores_files_that_are_not_markdown() {
        // Arrange
        let dir = temp_dir();
        std::fs::write(dir.join("10-a.md"), b"## A\n").unwrap();
        std::fs::write(dir.join("notes.txt"), b"scratch\n").unwrap();

        // Act
        let sections = read_curated(&dir).unwrap();

        // Assert
        assert_eq!(sections.len(), 1);
    }

    #[test]
    fn read_curated_normalises_crlf_so_the_output_is_checkout_independent() {
        // Arrange
        let dir = temp_dir();
        std::fs::write(dir.join("10-a.md"), b"## A\r\n\r\nbody\r\n").unwrap();

        // Act
        let sections = read_curated(&dir).unwrap();

        // Assert
        assert_eq!(sections[0].body, "## A\n\nbody\n");
    }

    #[test]
    fn workbook_writes_a_file_containing_every_builtin() {
        // Arrange
        let dir = temp_dir();
        let out = dir.join("nested").join("REFERENCE.md");

        // Act
        workbook(&out, &dir.join("no-curated")).unwrap();

        // Assert
        let text = std::fs::read_to_string(&out).unwrap();
        for entry in idl_rs::math::math_builtin_catalog() {
            assert!(text.contains(&format!("#### {}\n", entry.name)), "{} missing", entry.name);
        }
    }

    #[test]
    fn workbook_run_twice_writes_identical_bytes() {
        // Arrange
        let dir = temp_dir();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src").join("10-a.md"), b"## Alpha\n\nbody\n").unwrap();
        let out = dir.join("REFERENCE.md");

        // Act
        workbook(&out, &dir.join("src")).unwrap();
        let first = std::fs::read(&out).unwrap();
        workbook(&out, &dir.join("src")).unwrap();
        let second = std::fs::read(&out).unwrap();

        // Assert — CI's gate is `git diff --exit-code` over exactly this.
        assert_eq!(first, second);
    }

    #[test]
    fn workbook_output_contains_no_carriage_returns() {
        // Arrange
        let dir = temp_dir();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src").join("10-a.md"), b"## Alpha\r\n").unwrap();
        let out = dir.join("REFERENCE.md");

        // Act
        workbook(&out, &dir.join("src")).unwrap();

        // Assert — a Windows-generated file and a Linux-generated one have
        // to be the same bytes, or CI fails on the dev machine's own commit.
        let bytes = std::fs::read(&out).unwrap();
        assert!(!bytes.contains(&b'\r'));
    }
}
