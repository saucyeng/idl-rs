//! Renders `docs/WORKBOOK-REFERENCE.md` — the one document an author (or an
//! agent) reads to write a workbook (ruling R222 item 1).
//!
//! Two halves, in this order:
//!
//! 1. **Generated.** Every entry of [`crate::math::math_builtin_catalog`],
//!    grouped by its `category` column, plus the retired-name table read
//!    from [`crate::math::math_name_migrations`]. Nothing in this half is
//!    hand-written, so it cannot drift from the engine: CI regenerates the
//!    file and fails on any diff.
//! 2. **Curated.** The prose sections that have no machine-readable source —
//!    definition annotations, windows and laps, the JS host variables, the
//!    `plotForm` subset — kept as ordinary Markdown files under
//!    `docs/reference-src/` and appended in filename order by the caller.
//!
//! **Why the retired-name table is generated rather than curated**, despite
//! ruling R222 item 1 listing it among the curated sections: C2 §3.8 states
//! outright that `rust/core/src/math/alias.rs`'s `math_name_migrations()`
//! is the authoritative list and that its own table "must not drift". A
//! third hand-kept copy of a list that already has a stated single source
//! would be the drift the ruling's own CI gate exists to catch, so the
//! table is rendered from `alias.rs` and the curated file beside it carries
//! only the prose around it (lane self-ruling, `runs/2026-09-11`).
//!
//! Pure and deterministic: same catalog in, byte-identical Markdown out.
//! No I/O — the caller (`idl-rs docs workbook`) reads the curated files and
//! writes the result.

use crate::math::{math_builtin_catalog, math_name_migrations, MathBuiltinStatus};

/// One curated section to append after the generated half: the file's
/// stem (used only for ordering by the caller) and its verbatim Markdown
/// body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CuratedSection {
    /// The source file's stem, e.g. `"10-annotations"`. Carried for the
    /// caller's own ordering and error messages; never rendered.
    pub name: String,
    /// The file's Markdown, spliced in verbatim. Heading levels are the
    /// file's own responsibility — `##` for a top-level section, to sit
    /// beside the generated half's own `##` headings.
    pub body: String,
}

/// The anchor a heading gets in the rendered document, in GitHub's own
/// slug rules as far as this document needs them: lowercased, spaces to
/// hyphens, `-` and `_` kept, every other character dropped.
///
/// Exposed because the app's help panel scrolls to `builtin_anchor(name)`
/// when the editor asks for a function's entry (ruling R222 item 2), and
/// that anchor has to be computed the same way on both sides.
pub fn slugify(heading: &str) -> String {
    let mut out = String::with_capacity(heading.len());
    for ch in heading.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if ch == '_' || ch == '-' {
            // Kept as-is: GitHub's own slugger preserves both, and the
            // generated index links have to resolve when the file is read
            // on GitHub as well as in the app's help panel.
            out.push(ch);
        } else if ch == ' ' {
            out.push('-');
        }
    }
    out
}

/// The anchor of one builtin's entry in the generated reference — what the
/// editor's `F1` and "Docs" button scroll the help panel to.
pub fn builtin_anchor(name: &str) -> String {
    slugify(name)
}

/// Renders the whole reference: generated builtin sections, the generated
/// retired-name table, then `curated` in the order given.
///
/// Deterministic — category order is the catalog's own first-appearance
/// order (which is C2 §3.3's table order), never a hash map's iteration
/// order, so CI's `git diff --exit-code` gate only ever fires on a real
/// change.
pub fn render_workbook_reference(curated: &[CuratedSection]) -> String {
    let mut out = String::with_capacity(64 * 1024);

    out.push_str("# Workbook reference\n\n");
    out.push_str(
        "Everything a `.idl1wb` workbook can say: every math builtin, the\n\
         annotations a definition accepts, the variables a `js` cell is given,\n\
         and the chart grammar the Properties panel reads and writes.\n\n",
    );
    out.push_str(
        "**This file is generated in part.** The builtin catalog and the\n\
         retired-name table below are rendered from the engine itself by\n\
         `idl-rs docs workbook --out docs/WORKBOOK-REFERENCE.md`; CI regenerates\n\
         them and fails if the committed file differs. Edit the engine, or the\n\
         curated sources under `docs/reference-src/`, never this file.\n\n",
    );

    render_builtins(&mut out);
    render_retired_names(&mut out);

    for section in curated {
        out.push_str(section.body.trim_end());
        out.push_str("\n\n");
    }

    // One trailing newline, never two: the file is compared byte for byte.
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

/// The generated builtin half: a category index, then one `###` entry per
/// function under its `##` category heading.
fn render_builtins(out: &mut String) {
    let catalog = math_builtin_catalog();

    // First-appearance order, not sorted: C2 §3.3's own table order groups
    // related functions (every filter, then every time-domain operator) in
    // a sequence a reader can skim, which alphabetical order would shred.
    let mut categories: Vec<&'static str> = Vec::new();
    for entry in catalog {
        if !categories.contains(&entry.category) {
            categories.push(entry.category);
        }
    }

    out.push_str("## Math builtins\n\n");
    out.push_str(&format!(
        "{} functions a `math` cell's expression can call, grouped by category\n\
         (C2 §3.3). A function marked **not implemented** parses and validates —\n\
         it is part of the committed language surface — but evaluating it is an\n\
         error today.\n\n",
        catalog.len()
    ));

    for category in &categories {
        let names: Vec<String> = catalog
            .iter()
            .filter(|b| b.category == *category)
            .map(|b| format!("[`{}`](#{})", b.name, builtin_anchor(b.name)))
            .collect();
        out.push_str(&format!("- **{}** — {}\n", category, names.join(", ")));
    }
    out.push('\n');

    for category in &categories {
        out.push_str(&format!("### {}\n\n", category));
        for entry in catalog.iter().filter(|b| b.category == *category) {
            out.push_str(&format!("#### {}\n\n", entry.name));
            out.push_str(&format!("```\n{}\n```\n\n", entry.signature));
            out.push_str(&format!("{}\n\n", entry.description));
            out.push_str("| | |\n|---|---|\n");
            out.push_str(&format!("| Shape | `{}` |\n", entry.shape));
            out.push_str(&format!("| Unit rule | `{}` |\n", entry.unit_rule));
            out.push_str(&format!(
                "| Status | {} |\n",
                match entry.status {
                    MathBuiltinStatus::Implemented => "implemented",
                    MathBuiltinStatus::NotImplemented => "**not implemented**",
                }
            ));
            let arity = entry.arity.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" or ");
            out.push_str(&format!("| Arguments | {} |\n\n", arity));
            out.push_str(&format!("Example:\n\n```\nresult = {}\n```\n\n", entry.example));
        }
    }
}

/// The generated retired-name table, from `alias.rs` (see the module doc
/// for why this half is generated and not curated).
fn render_retired_names(out: &mut String) {
    out.push_str("## Retired names\n\n");
    out.push_str(
        "Names the language no longer uses, and what replaced them. A `version: 3`\n\
         workbook is migrated on read and rewritten on save; a `version: 4`\n\
         workbook using one is a typed error naming its replacement (C2 §3.8).\n\n",
    );
    out.push_str("| Retired | Replacement |\n|---|---|\n");
    for migration in math_name_migrations() {
        out.push_str(&format!("| `{}` | `{}` |\n", migration.old, migration.new));
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_workbook_reference_called_twice_is_byte_identical() {
        // Arrange
        let curated = vec![CuratedSection { name: "10-x".to_string(), body: "## X\n\nbody\n".to_string() }];

        // Act
        let first = render_workbook_reference(&curated);
        let second = render_workbook_reference(&curated);

        // Assert — CI's whole gate is `git diff --exit-code` over this
        // output, so non-determinism here would be a permanently red build.
        assert_eq!(first, second);
    }

    #[test]
    fn render_workbook_reference_every_builtin_has_its_own_heading() {
        // Arrange
        let rendered = render_workbook_reference(&[]);

        // Act / Assert
        for entry in math_builtin_catalog() {
            assert!(
                rendered.contains(&format!("#### {}\n", entry.name)),
                "{} has no heading in the rendered reference",
                entry.name
            );
        }
    }

    #[test]
    fn render_workbook_reference_a_deferred_builtin_is_marked_not_implemented() {
        // Arrange
        let rendered = render_workbook_reference(&[]);

        // Act
        let spectrogram = rendered
            .split("#### ")
            .find(|s| s.starts_with("spectrogram\n"))
            .expect("spectrogram section");

        // Assert
        assert!(spectrogram.contains("| Status | **not implemented** |"));
    }

    #[test]
    fn render_workbook_reference_curated_sections_follow_the_generated_half() {
        // Arrange
        let curated = vec![
            CuratedSection { name: "10-a".to_string(), body: "## Alpha\n".to_string() },
            CuratedSection { name: "20-b".to_string(), body: "## Beta\n".to_string() },
        ];

        // Act
        let rendered = render_workbook_reference(&curated);

        // Assert
        let builtins = rendered.find("## Math builtins").unwrap();
        let alpha = rendered.find("## Alpha").unwrap();
        let beta = rendered.find("## Beta").unwrap();
        assert!(builtins < alpha && alpha < beta);
    }

    #[test]
    fn render_workbook_reference_every_retired_name_appears_in_the_table() {
        // Arrange
        let rendered = render_workbook_reference(&[]);

        // Act / Assert
        for migration in math_name_migrations() {
            assert!(
                rendered.contains(&format!("| `{}` | `{}` |", migration.old, migration.new)),
                "{} missing from the retired-name table",
                migration.old
            );
        }
    }

    #[test]
    fn render_workbook_reference_output_ends_with_exactly_one_newline() {
        // Arrange
        let curated = vec![CuratedSection { name: "10-a".to_string(), body: "## Alpha\n\n\n".to_string() }];

        // Act
        let rendered = render_workbook_reference(&curated);

        // Assert
        assert!(rendered.ends_with("\n"));
        assert!(!rendered.ends_with("\n\n"));
    }

    #[test]
    fn slugify_a_heading_with_punctuation_drops_it() {
        // Arrange / Act
        let slug = slugify("Estimator (diagnostic)");

        // Assert
        assert_eq!(slug, "estimator-diagnostic");
    }

    #[test]
    fn builtin_anchor_a_builtin_name_is_the_name_itself() {
        // Arrange / Act / Assert — every catalog name is already
        // `[a-z0-9_]`, so the anchor is the name unchanged.
        assert_eq!(builtin_anchor("lap_delta_time"), "lap_delta_time");
        assert_eq!(builtin_anchor("welch"), "welch");
    }
}
