//! Workbook v3 (`.idl1wb`) — the Markdown/front-matter parser and cell-id
//! assignment (C2 §1–§2). This task (Task 1 of the workbook-v3 lane) lands
//! only the document container: front matter, fence scanning, cell ids and
//! prose spans. Math-cell grammar (C2 §3), table cells (C2 §4), JS cells
//! (C2 §5) and evaluation are later tasks; the fence bodies here are stored
//! raw and unparsed (`CellDoc::raw_fence_body`).

use std::collections::{HashMap, HashSet};

pub mod cell;
pub mod constants;
pub mod error;
pub mod eval;
pub mod front_matter;
pub mod host;
pub mod host_channel_wire;
pub mod js_cell;
pub mod math_cell;
pub mod prose;
pub mod resolve;
pub mod table_cell;

pub use cell::{CellDoc, CellKindToken};
pub use constants::merge_constants;
pub use error::{WorkbookError, WorkbookErrorKind};
pub use eval::{eval_cells, CellDefResult, CellError, CellEvalResult};
pub use front_matter::{ConstantRaw, FrontMatter, UnitsPref};
pub use host::{channel, host_constants, host_laps, host_session, to_host_channel, HostChannel, HostLap, HostSession};
pub use host_channel_wire::encode_host_channel_idlh;
pub use js_cell::{find_inline_exprs, InlineExpr};
pub use math_cell::{parse_math_cell_body, MathCellLine};
pub use prose::{render_prose_html, ProseSpanRef, RenderedProse};
pub use resolve::resolve_workbook_defs;
pub use table_cell::parse_table_cell;

/// One `const` line collected from any `math` cell (C2 §3.1), flattened
/// across the whole document. Workbook-scoped like a front-matter constant
/// (C2 §3.1: "not scoped to their own cell") — `cell_id` records only where
/// it was *declared*, for error messages. Handed to Task 4's
/// `merge_constants`, this lane's single `DuplicateConstant`/`ReservedName`
/// enforcement point for constants (this module raises neither for `Const`
/// lines — see [`parse_workbook`]'s doc comment).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstLine {
    /// The `math` cell this `const` line was declared in.
    pub cell_id: String,
    /// The `const` line's identifier (C2 §3.1) — already validated against
    /// `identifier`/[`error::RESERVED_NAMES`] by [`parse_math_cell_body`].
    pub name: String,
    /// The constant's scalar value — unitless (a `const` line's
    /// right-hand side is always a bare `number`, C2 §3.1).
    pub value: f64,
    /// Always `None` — a `const` line has no unit-suffix syntax (C2 §3.1);
    /// kept for shape symmetry with
    /// [`front_matter::ConstantRaw::WithUnit`].
    pub unit_display: Option<String>,
}

/// One flat-namespace `def_line` (C2 §2.4, §3.1), flattened across the whole
/// document — Task 6's `resolve_workbook_defs` input. `order` is this
/// definition's index within its own cell's `Def` lines (not a document-wide
/// index): document order plus per-cell `order` together give Task 9 "which
/// cell declared each identifier, in source order" (L3-R16).
#[derive(Debug, Clone, PartialEq)]
pub struct MathCellDef {
    /// The `math` cell this definition was declared in.
    pub cell_id: String,
    /// The `def_line`'s identifier (C2 §3.1) — already validated against
    /// `identifier`/[`error::RESERVED_NAMES`] by [`parse_math_cell_body`].
    pub name: String,
    /// The unparsed right-hand side (C2 §3.2 expression parsing happens at
    /// evaluation time, not here).
    pub expr_text: String,
    /// The definition's display name, from a `# label: <text>` trailing
    /// comment (C2 §3.1). `None` when the line has no such comment.
    pub label: Option<String>,
    /// Index within this definition's own cell's `Def` lines (source order).
    pub order: usize,
}

/// A parsed `.idl1wb` document (C2 §1–§2): front-matter identity plus the
/// cells and prose the body contains, in document order.
#[derive(Debug, Clone)]
pub struct WorkbookDoc {
    /// Stable workbook identity (C2 §1) — a UUIDv4 string.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Front-matter `constants` map, unresolved (name → raw value, C2 §1,
    /// §3.1). Task 3's flat constants table (front matter + `const` lines)
    /// builds on this.
    pub constants_raw: HashMap<String, ConstantRaw>,
    /// Editor unit-system preference (C2 §1); no effect on parsing/eval.
    pub units_pref: UnitsPref,
    /// Schema version — always `3` for a document this function returns
    /// `Ok` for (a non-`3` explicit value is fatal, see [`parse_workbook`]).
    pub version: u32,
    /// Every recognised `math`/`table`/`js` cell, in document order.
    pub cells: Vec<CellDoc>,
    /// Whole-document prose when [`Self::cells`] is empty (C2 §2.4) —
    /// `None` whenever there is at least one cell (trailing text then lives
    /// on that last cell's [`CellDoc::prose_after`] instead).
    pub trailing_prose: Option<String>,
    /// Every `const` line from every `math` cell, flattened document-wide
    /// (C2 §2.4, §3.1) — Task 4's `merge_constants` input; see
    /// [`ConstLine`].
    pub const_lines: Vec<ConstLine>,
    /// Every `def_line` from every `math` cell, flattened document-wide (C2
    /// §2.4) in document-cell order, then `def_line` order within a cell —
    /// Task 6's `resolve_workbook_defs` input (L3-R16); see [`MathCellDef`].
    pub defs: Vec<MathCellDef>,
    /// The flat constants table (C2 §3.1): [`merge_constants`]'s merged
    /// `name → f64` output over `constants_raw` and `const_lines`, run once
    /// inside [`parse_workbook`] (L3-R16/17).
    pub constants: HashMap<String, f64>,
}

/// Parses a `.idl1wb` document's front matter, cell fences and prose spans
/// (C2 §1–§2).
///
/// `Err` only for front-matter-fatal problems —
/// [`WorkbookErrorKind::MissingFrontMatterId`] (which also covers YAML that
/// doesn't parse at all, see [`front_matter::parse_front_matter`]'s doc
/// comment) or [`WorkbookErrorKind::UnsupportedWorkbookVersion`]. Every
/// other structural problem (`DuplicateCellId` here; Task 2 adds more) is
/// collected into the `Ok` tuple's `Vec<WorkbookError>` alongside a
/// best-effort [`WorkbookDoc`], so an editor can still show every other
/// cell. This is a deliberate design choice beyond what C2 states verbatim
/// (C2 fixes the error *kinds*, not whether parsing is all-or-nothing) —
/// stated here so Task 2 and Task 9 build on one consistent rule:
/// **front-matter identity/version is fatal; everything else is
/// collected.**
///
/// Every `math` cell's fence body is parsed here too (C2 §3.1) and its
/// `Def` lines flattened into one document-wide namespace (C2 §2.4): a
/// repeated name raises [`WorkbookErrorKind::DuplicateDefinition`]. `Const`
/// lines are only *collected*, into [`WorkbookDoc::const_lines`] — this
/// function does not raise `DuplicateConstant` for them (G3.6): Task 4's
/// `merge_constants` is this lane's single enforcement point for that kind,
/// the same "single enforcement point" shape already used for
/// `ReservedName`.
pub fn parse_workbook(markdown: &str) -> Result<(WorkbookDoc, Vec<WorkbookError>), Vec<WorkbookError>> {
    let (front_matter, body) = front_matter::parse_front_matter(markdown).map_err(|e| vec![e])?;

    if front_matter.version != 3 {
        return Err(vec![error::unsupported_workbook_version(front_matter.version)]);
    }

    let (mut cells, trailing_prose, mut errors) = cell::scan_cells(body);

    let mut const_lines = Vec::new();
    let mut defs = Vec::new();
    let mut def_names = HashSet::new();
    for cell in &mut cells {
        match cell.kind_token {
            CellKindToken::Math => {
                let (lines, cell_errors) = math_cell::parse_math_cell_body(&cell.id, &cell.raw_fence_body);
                errors.extend(cell_errors);
                let mut order = 0;
                for line in lines {
                    match line {
                        MathCellLine::Def { name, expr_text, label } => {
                            if !def_names.insert(name.clone()) {
                                errors.push(error::duplicate_definition(&cell.id, &name));
                            }
                            defs.push(MathCellDef { cell_id: cell.id.clone(), name, expr_text, label, order });
                            order += 1;
                        }
                        MathCellLine::Const { name, value, unit_display } => {
                            const_lines.push(ConstLine { cell_id: cell.id.clone(), name, value, unit_display });
                        }
                        MathCellLine::Blank | MathCellLine::Comment => {}
                    }
                }
            }
            CellKindToken::Table => {
                // C2 §4: a table cell's JSON either parses or it doesn't —
                // eagerly parsed here (unlike a math cell's lazy text),
                // since there is no per-line partial result to preserve.
                match table_cell::parse_table_cell(&cell.id, &cell.raw_fence_body) {
                    Ok(table) => cell.table = Some(table),
                    Err(e) => errors.push(e),
                }
            }
            CellKindToken::Js => {}
        }
    }

    // L3-R16: merge_constants runs exactly once here — its errors half joins
    // the collected Vec<WorkbookError>, its table half becomes
    // WorkbookDoc.constants (Task 8's host_constants source).
    let (constants, constant_errors) = merge_constants(&front_matter.constants, &const_lines);
    errors.extend(constant_errors);

    let doc = WorkbookDoc {
        id: front_matter.id,
        name: front_matter.name,
        constants_raw: front_matter.constants,
        units_pref: front_matter.units,
        version: front_matter.version,
        cells,
        trailing_prose,
        const_lines,
        defs,
        constants,
    };

    Ok((doc, errors))
}

/// Renders a [`WorkbookDoc`] back to `.idl1wb` source text (C2 §1's
/// `document ::= front_matter body` grammar) — the inverse of
/// [`parse_workbook`], needed by L11 Task 6's sync `install` to write a
/// merged document to disk (`workbook::merge::merge` only ever produces a
/// structured [`WorkbookDoc`], never text). Every byte this function emits
/// already lives verbatim in `doc`'s fields (`raw_fence_body`,
/// `prose_before`/`prose_after`, `trailing_prose`) — this is pure
/// reassembly, never a re-derivation of content `parse_workbook` already
/// captured, so it is exact for any `doc` that came from `parse_workbook` or
/// from `workbook::merge::merge` (which only ever copies cells verbatim
/// from one of its three inputs, plus freshly-rendered marker/prose text
/// built with this same grammar in mind, C2 §7.2/§7.3).
///
/// No prior task needed this direction: ordinary editing round-trips the
/// author's own markdown text unchanged, re-parsing it only for evaluation.
/// A fence body that does not already end in `\n` gets one inserted before
/// the closing fence — `parse_workbook`'s own fence-body capture is not
/// documented as always including that trailing newline (pulldown-cmark's
/// `Text` event boundary is not part of its stated public contract, C2
/// §2.2's own cell-id comment makes the same caveat about that crate), so
/// this guards the one byte a future upgrade could silently drop instead of
/// asserting a convention this module does not own.
///
/// **R103 fixed point (`runs/2026-09-03/decisions.md`):** the closing fence
/// marker itself is emitted *without* a trailing `\n` — `scan_cells`'s
/// underlying pulldown-cmark code-block byte range ends right at the
/// closing `` ``` ``, before that line's own line-ending newline, so that
/// newline is already the first byte of the following `prose_before` /
/// `prose_after` / `trailing_prose` span (or, for the very last cell with no
/// following text at all, simply absent because the source itself has no
/// trailing newline there). A hard-coded `"```\n"` here used to double that
/// byte, inserting a spurious blank line around every fence on every
/// render — `render_workbook(parse_workbook(s))` was therefore never a
/// fixed point for a document that closes any fence, and C2 §7.2's
/// `Unchanged`/`Changed` cell classification (byte-identical content) then
/// misread an untouched cell as edited after a single round trip through
/// the editor's own save path.
///
/// **Front matter is the one deliberate exception**, not part of this
/// invariant: `front_matter::render_front_matter`'s own doc comment already
/// states its contract is "round-trips through [`parse_front_matter`] back
/// to the same `FrontMatter`... not the exact bytes emitted" (`serde_yaml_ng`
/// picks its own YAML style). This is safe for C2 §7.1's merge, which reads
/// front matter "per top-level key" as structured values (`id`, `name`,
/// `units`, `constants`), never as raw YAML bytes — so a reformatted-but
/// -equivalent front-matter block cannot manufacture the spurious-`Changed`
/// bug R103 describes, unlike a cell's byte-for-byte-compared body. The
/// fixed-point invariant this function upholds is therefore: byte-identical
/// for the whole document whenever the front matter is already in
/// `render_front_matter`'s own canonical form, and byte-identical for
/// `body` alone (everything after the front matter's closing `---`) in
/// every case — see the `tests` module's `render_workbook_is_a_fixed_point_*`
/// suite.
pub fn render_workbook(doc: &WorkbookDoc) -> String {
    let fm = FrontMatter {
        id: doc.id.clone(),
        name: doc.name.clone(),
        constants: doc.constants_raw.clone(),
        units: doc.units_pref,
        version: doc.version,
    };
    let mut out = front_matter::render_front_matter(&fm);

    if doc.cells.is_empty() {
        if let Some(prose) = &doc.trailing_prose {
            out.push_str(prose);
        }
        return out;
    }

    let last_index = doc.cells.len() - 1;
    for (i, cell) in doc.cells.iter().enumerate() {
        if let Some(prose_before) = &cell.prose_before {
            out.push_str(prose_before);
        }
        let kind = match cell.kind_token {
            CellKindToken::Math => "math",
            CellKindToken::Table => "table",
            CellKindToken::Js => "js",
        };
        out.push_str("```");
        out.push_str(kind);
        out.push_str(" id=");
        out.push_str(&cell.id);
        out.push('\n');
        out.push_str(&cell.raw_fence_body);
        if !cell.raw_fence_body.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("```");
        if i == last_index {
            if let Some(prose_after) = &cell.prose_after {
                out.push_str(prose_after);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests_pipeline;

#[cfg(test)]
mod tests {
    use super::*;

    /// C2 §2.5's worked example, restated literally (copied verbatim from
    /// the spec — already the contract's own checked example).
    const WORKED_EXAMPLE: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Fork tuning\nconstants: { rider_mass_kg: 82 }\n---\n# Fork tuning \u{2014} Whistler, 2026-08-30\n\n```math id=a1b2c3d4\nfork_velocity = differentiate([fork_travel])\nfork_bottom_out = [fork_travel] > 195\n```\n\n```js id=e5f6a7b8\nPlot.plot({ marks: [Plot.lineY(channel(\"fork_velocity\"), { x: \"t\", y: \"v\" })] })\n```\n\nBottom-outs this lap: ${fork_bottom_out.v.filter(Boolean).length}\n";

    #[test]
    fn parse_workbook_c2_5_worked_example_parses_id_version_and_both_cells() {
        // Act
        let (doc, errors) = parse_workbook(WORKED_EXAMPLE).unwrap();

        // Assert
        assert!(errors.is_empty());
        assert_eq!(doc.id, "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d");
        assert_eq!(doc.version, 3);
        assert_eq!(doc.cells.len(), 2);
        assert_eq!(doc.cells[0].id, "a1b2c3d4");
        assert_eq!(doc.cells[0].kind_token, CellKindToken::Math);
        assert!(doc.cells[0].raw_fence_body.contains("fork_velocity = differentiate([fork_travel])"));
        assert!(doc.cells[0].raw_fence_body.contains("fork_bottom_out = [fork_travel] > 195"));
        assert_eq!(doc.cells[1].id, "e5f6a7b8");
        assert_eq!(doc.cells[1].kind_token, CellKindToken::Js);
    }

    #[test]
    fn same_identifier_defined_in_two_different_math_cells_duplicate_definition_both_cells_still_parse() {
        // Arrange
        let markdown = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Test\n---\n\n```math id=aaaaaaaa\nroll_deg = [Roll]\n```\n\n```math id=bbbbbbbb\nroll_deg = [Roll2]\n```\n";

        // Act
        let (doc, errors) = parse_workbook(markdown).unwrap();

        // Assert
        assert_eq!(doc.cells.len(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::DuplicateDefinition);
        assert_eq!(errors[0].cell_id, "bbbbbbbb");
    }

    #[test]
    fn render_workbook_the_c2_5_worked_example_re_parses_to_the_same_cells() {
        // Arrange
        let (doc, _errors) = parse_workbook(WORKED_EXAMPLE).unwrap();

        // Act
        let rendered = render_workbook(&doc);
        let (back, errors) = parse_workbook(&rendered).unwrap();

        // Assert
        assert!(errors.is_empty());
        assert_eq!(back.id, doc.id);
        assert_eq!(back.name, doc.name);
        assert_eq!(back.cells.len(), doc.cells.len());
        assert_eq!(back.cells[0].id, doc.cells[0].id);
        assert_eq!(back.cells[0].kind_token, doc.cells[0].kind_token);
        assert_eq!(back.cells[0].raw_fence_body.trim(), doc.cells[0].raw_fence_body.trim());
        assert_eq!(back.cells[1].id, doc.cells[1].id);
        assert_eq!(back.trailing_prose, doc.trailing_prose);
    }

    #[test]
    fn render_workbook_a_pure_prose_document_round_trips_the_trailing_prose() {
        // Arrange
        let markdown = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Notes\n---\nJust a written note, no cells here.\n";
        let (doc, _errors) = parse_workbook(markdown).unwrap();

        // Act
        let rendered = render_workbook(&doc);
        let (back, errors) = parse_workbook(&rendered).unwrap();

        // Assert
        assert!(errors.is_empty());
        assert!(back.cells.is_empty());
        assert_eq!(back.trailing_prose, doc.trailing_prose);
    }

    #[test]
    fn render_workbook_a_cell_with_no_prose_before_or_after_still_parses_back_to_one_cell() {
        // Arrange
        let markdown = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Test\n---\n```math id=aaaaaaaa\nx = 1\n```\n";
        let (doc, _errors) = parse_workbook(markdown).unwrap();

        // Act
        let rendered = render_workbook(&doc);
        let (back, errors) = parse_workbook(&rendered).unwrap();

        // Assert
        assert!(errors.is_empty());
        assert_eq!(back.cells.len(), 1);
        assert_eq!(back.cells[0].id, "aaaaaaaa");
        assert!(back.cells[0].raw_fence_body.contains("x = 1"));
    }

    /// R103's fixed-point suite (`runs/2026-09-03/decisions.md`): every
    /// fixture here asserts `render_workbook(parse_workbook(s)) == s` on
    /// `body` alone (everything after the front matter's closing `---\n`,
    /// via [`body_of`]) — the byte range [`render_workbook`]'s doc comment
    /// promises unconditionally — and, where the fixture's own front matter
    /// is already in [`front_matter::render_front_matter`]'s canonical
    /// form (built with [`canonical_doc`]), the *whole* document
    /// byte-for-byte too.
    /// Splits `markdown` into `(front_matter_block, body)` on the first
    /// `\n---\n` after the opening `---\n` — mirrors
    /// [`front_matter::parse_front_matter`]'s own split, kept independent of
    /// it so a bug in that function can't also hide a bug in this test.
    fn body_of(markdown: &str) -> &str {
        let rest = markdown.strip_prefix("---\n").expect("fixture must open with '---\\n'");
        let (_yaml, body) = rest.split_once("\n---\n").expect("fixture must have a closing '---\\n'");
        body
    }

    /// Builds a document whose front matter is already
    /// [`front_matter::render_front_matter`]'s own canonical byte form (no
    /// `constants`, SI units, explicit `version: 3`), so the whole-document
    /// fixed point holds, not just the `body` slice — used by fixtures that
    /// don't care about exercising the front-matter grammar itself.
    fn canonical_doc(body: &str) -> String {
        let fm = FrontMatter {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Test".to_string(),
            constants: HashMap::new(),
            units: UnitsPref::Si,
            version: 3,
        };
        format!("{}{}", front_matter::render_front_matter(&fm), body)
    }

    /// Runs one fixed-point assertion: `body_of` matches byte-for-byte
    /// always; the whole document matches too whenever `markdown`'s front
    /// matter is already canonical (checked by rendering `doc`'s own front
    /// matter back and comparing against `markdown`'s own front-matter
    /// block, so a non-canonical fixture doesn't have to pre-compute
    /// whether it happens to be canonical).
    fn assert_fixed_point(markdown: &str) {
        let (doc, errors) = parse_workbook(markdown).unwrap();
        assert!(errors.is_empty(), "fixture must parse cleanly: {errors:?}");

        let rendered = render_workbook(&doc);

        assert_eq!(body_of(&rendered), body_of(markdown), "body diverged for {markdown:?}");

        let canonical_front_matter = front_matter::render_front_matter(&FrontMatter {
            id: doc.id.clone(),
            name: doc.name.clone(),
            constants: doc.constants_raw.clone(),
            units: doc.units_pref,
            version: doc.version,
        });
        if markdown.starts_with(&canonical_front_matter) {
            assert_eq!(rendered, markdown, "whole document diverged for already-canonical front matter {markdown:?}");
        }
    }

    #[test]
    fn render_workbook_is_a_fixed_point_c2_2_5_worked_example() {
        assert_fixed_point(WORKED_EXAMPLE);
    }

    #[test]
    fn render_workbook_is_a_fixed_point_prose_only_document() {
        assert_fixed_point(&canonical_doc("Just a written note, no cells here.\n"));
    }

    #[test]
    fn render_workbook_is_a_fixed_point_two_adjacent_fences_no_blank_line_between() {
        assert_fixed_point(&canonical_doc(
            "```math id=aaaaaaaa\nx = 1\n```\n```js id=bbbbbbbb\ny\n```\n",
        ));
    }

    #[test]
    fn render_workbook_is_a_fixed_point_fence_then_prose_with_two_blank_lines() {
        assert_fixed_point(&canonical_doc(
            "```math id=aaaaaaaa\nx = 1\n```\n\n\nTwo blank lines above this line.\n",
        ));
    }

    #[test]
    fn render_workbook_is_a_fixed_point_trailing_whitespace_on_a_prose_line() {
        assert_fixed_point(&canonical_doc("```math id=aaaaaaaa\nx = 1\n```\n\nTrailing spaces below.   \n"));
    }

    #[test]
    fn render_workbook_is_a_fixed_point_no_trailing_newline_at_all() {
        assert_fixed_point(&canonical_doc("```math id=aaaaaaaa\nx = 1\n```"));
    }

    #[test]
    fn render_workbook_is_a_fixed_point_front_matter_with_every_optional_field() {
        let fm = FrontMatter {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Fork tuning".to_string(),
            constants: HashMap::from([
                ("rider_mass_kg".to_string(), ConstantRaw::WithUnit { value: 82.0, unit_display: "kg".to_string() }),
                ("gravity_m_s2".to_string(), ConstantRaw::Number(9.80665)),
            ]),
            units: UnitsPref::Imperial,
            version: 3,
        };
        let markdown = format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front_matter::render_front_matter(&fm));
        assert_fixed_point(&markdown);
    }
}

