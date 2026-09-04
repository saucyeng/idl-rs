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
pub mod front_matter;
pub mod math_cell;

pub use cell::{CellDoc, CellKindToken};
pub use constants::merge_constants;
pub use error::{WorkbookError, WorkbookErrorKind};
pub use front_matter::{ConstantRaw, FrontMatter, UnitsPref};
pub use math_cell::{parse_math_cell_body, MathCellLine};

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

    let (cells, trailing_prose, mut errors) = cell::scan_cells(body);

    let mut const_lines = Vec::new();
    let mut def_names = HashSet::new();
    for cell in &cells {
        if cell.kind_token != CellKindToken::Math {
            continue;
        }
        let (lines, cell_errors) = math_cell::parse_math_cell_body(&cell.id, &cell.raw_fence_body);
        errors.extend(cell_errors);
        for line in lines {
            match line {
                MathCellLine::Def { name, .. } => {
                    if !def_names.insert(name.clone()) {
                        errors.push(error::duplicate_definition(&cell.id, &name));
                    }
                }
                MathCellLine::Const { name, value, unit_display } => {
                    const_lines.push(ConstLine { cell_id: cell.id.clone(), name, value, unit_display });
                }
                MathCellLine::Blank | MathCellLine::Comment => {}
            }
        }
    }

    let doc = WorkbookDoc {
        id: front_matter.id,
        name: front_matter.name,
        constants_raw: front_matter.constants,
        units_pref: front_matter.units,
        version: front_matter.version,
        cells,
        trailing_prose,
        const_lines,
    };

    Ok((doc, errors))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C2 §2.5's worked example, restated literally (copied verbatim from
    /// the spec — already the contract's own checked example).
    const WORKED_EXAMPLE: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Fork tuning\nconstants: { g: 9.80665, rider_mass_kg: 82 }\n---\n# Fork tuning \u{2014} Whistler, 2026-08-30\n\n```math id=a1b2c3d4\nfork_velocity = differentiate([fork_travel])\nfork_bottom_out = [fork_travel] > 195\n```\n\n```js id=e5f6a7b8\nPlot.plot({ marks: [Plot.lineY(channel(\"fork_velocity\"), { x: \"t\", y: \"v\" })] })\n```\n\nBottom-outs this lap: ${fork_bottom_out.v.filter(Boolean).length}\n";

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
}
