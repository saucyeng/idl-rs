//! Workbook v3 (`.idl1wb`) — the Markdown/front-matter parser and cell-id
//! assignment (C2 §1–§2). This task (Task 1 of the workbook-v3 lane) lands
//! only the document container: front matter, fence scanning, cell ids and
//! prose spans. Math-cell grammar (C2 §3), table cells (C2 §4), JS cells
//! (C2 §5) and evaluation are later tasks; the fence bodies here are stored
//! raw and unparsed (`CellDoc::raw_fence_body`).

use std::collections::HashMap;

pub mod cell;
pub mod error;
pub mod front_matter;

pub use cell::{CellDoc, CellKindToken};
pub use error::{WorkbookError, WorkbookErrorKind};
pub use front_matter::{ConstantRaw, FrontMatter, UnitsPref};

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
pub fn parse_workbook(markdown: &str) -> Result<(WorkbookDoc, Vec<WorkbookError>), Vec<WorkbookError>> {
    let (front_matter, body) = front_matter::parse_front_matter(markdown).map_err(|e| vec![e])?;

    if front_matter.version != 3 {
        return Err(vec![WorkbookError::front_matter(
            WorkbookErrorKind::UnsupportedWorkbookVersion,
            format!("Workbook version {} is not supported (expected 3)", front_matter.version),
        )]);
    }

    let (cells, trailing_prose, errors) = cell::scan_cells(body);

    let doc = WorkbookDoc {
        id: front_matter.id,
        name: front_matter.name,
        constants_raw: front_matter.constants,
        units_pref: front_matter.units,
        version: front_matter.version,
        cells,
        trailing_prose,
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
}
