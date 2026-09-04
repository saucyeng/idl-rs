//! Per-cell evaluation orchestrator (C3 §3.4) — the Tauri-free Rust
//! equivalent of `CellOutput`: given a parsed [`super::WorkbookDoc`] and the
//! structural errors [`super::parse_workbook`] collected alongside it, wires
//! Task 6's resolved math definitions and Step 2's routed structural errors
//! into **one [`CellEvalResult`] per cell, in document order** (C3 §3.4's own
//! wording, literally — not one per definition; that was the plan's stated
//! default, closed the other way by lead ruling L3-R25,
//! `runs/2026-09-03/decisions.md` R21, because C3 §3.4 already says
//! otherwise and "one per definition" makes `cell_id` non-unique across the
//! returned `Vec`). L5's `eval_workbook` Tauri command (C3 §3.4) calls this
//! once per open workbook and converts each [`CellEvalResult`] into C3's
//! `CellOutput` JSON shape — that mapping, and `MathEvalErrorKind`'s mapping
//! through C3 §2's `math_*` prefix, is L5's job, not built here.

use std::collections::HashMap;

use crate::math::MathEvalError;

use super::host::to_host_channel;
use super::resolve::resolve_workbook_defs;
use super::{CellDoc, CellKindToken, WorkbookDoc, WorkbookError};

use crate::math::eval::{ChannelLookup, MathLapContext};

/// One `math`-cell definition's evaluated result, in `def_line` source order
/// within its cell ([`super::MathCellDef::order`]). Exactly one of `value`/
/// `error` is `Some` — [`resolve_workbook_defs`] always produces `Ok` xor
/// `Err` for a def that reaches evaluation (CLAUDE.md §5: a sibling
/// definition's failure never suppresses this one's own result).
#[derive(Debug, Clone, PartialEq)]
pub struct CellDefResult {
    /// The definition's identifier (C2 §3.1) — unique document-wide.
    pub name: String,
    /// The definition's `# label: <text>` display name, if any (C2 §3.1).
    pub label: Option<String>,
    /// The resolved channel, host-variable shape (C2 §5.1), on success.
    pub value: Option<crate::workbook::v3::HostChannel>,
    /// The evaluation failure, on failure — reuses [`MathEvalError`]
    /// verbatim (C2 §3.5.B) so C3's `math_*` IPC kind prefixes need no
    /// translation layer.
    pub error: Option<MathEvalError>,
}

/// One error attached to a [`CellEvalResult`]: either a structural
/// (parse-time, C2 §3.5.A) problem routed here by [`eval_cells`]'s Step 2,
/// or a cell-level evaluation (C2 §3.5.B) problem. `Eval` is not populated
/// by this module today — every `math`-cell evaluation failure this task
/// produces lives on its own [`CellDefResult::error`] instead — but is part
/// of L3-R25's shape (`runs/2026-09-03/decisions.md` R21) for a future
/// cell-level (not per-definition) evaluation failure, e.g. a `table`/`js`
/// cell's own evaluation error once L5/L6 produce one.
#[derive(Debug, Clone, PartialEq)]
pub enum CellError {
    /// A structural (parse-time) error scoped to this cell (C2 §3.5.A).
    Structural(WorkbookError),
    /// A cell-level (non-per-definition) evaluation error (C2 §3.5.B).
    Eval(MathEvalError),
}

/// One cell's evaluation result (C3 §3.4's `CellOutput`, Tauri-free Rust
/// equivalent) — one entry per [`super::WorkbookDoc::cells`] entry, in
/// document order (C3 §3.4 literally; L3-R25).
#[derive(Debug, Clone, PartialEq)]
pub struct CellEvalResult {
    /// The cell's fence id (C2 §2.2) — unique across the returned `Vec`
    /// (unlike a per-definition scheme, which would repeat it for a
    /// multi-definition `math` cell).
    pub cell_id: String,
    /// `math` / `table` / `js` (C2 §2.1).
    pub kind: CellKindToken,
    /// This cell's `math`-definition results, in `def_line` source order
    /// (empty for `table`/`js` cells). **A `table` cell's success value does
    /// NOT live here** (G9.4) — it lives on [`CellDoc::table`] plus
    /// `table::eval`'s own per-cell `CellResult` grid; this field carries
    /// only structural errors for a `table` cell, via [`Self::errors`]. L5
    /// must read `CellDoc.table` for a table cell's value, not this struct.
    pub defs: Vec<CellDefResult>,
    /// This cell's structural errors (C2 §3.5.A, routed here by
    /// [`eval_cells`]'s Step 2 from the `structural` parameter) plus any
    /// cell-level evaluation errors (C2 §3.5.B — see [`CellError::Eval`]).
    /// This is how a `table` cell's `InvalidTableJson` and a `math` cell's
    /// `DuplicateDefinition`/`InvalidIdentifier`/`ReservedName` all reach a
    /// cell — the routing is generic, not special-cased per kind.
    pub errors: Vec<CellError>,
}

/// Builds one [`CellEvalResult`] per cell in `doc.cells` (C3 §3.4, L3-R25):
/// wires Task 6's [`resolve_workbook_defs`] output back onto each `math`
/// cell's own definitions (Step 1) and routes `structural`'s
/// [`WorkbookError`]s to their owning cell by `cell_id` (Step 2, L3-R2); a
/// `"front-matter"`-scoped structural error is dropped here (G9.1) — it is
/// not cell-scoped and is already fatal or returned separately by
/// [`super::parse_workbook`]'s caller. `table`/`js` cells get an empty
/// `defs` (see [`CellEvalResult::defs`]'s doc comment, G9.4).
pub fn eval_cells(
    doc: &WorkbookDoc,
    structural: &[WorkbookError],
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> Vec<CellEvalResult> {
    let mut resolved = resolve_workbook_defs(&doc.defs, &doc.constants, lookup, lap_ctx);

    // Step 2: group structural errors by owning cell_id up front, dropping
    // the front-matter-scoped ones (G9.1) — a single pass over `structural`
    // rather than one pass per cell.
    let mut errors_by_cell: HashMap<&str, Vec<CellError>> = HashMap::new();
    for err in structural {
        if err.cell_id == "front-matter" {
            continue;
        }
        errors_by_cell.entry(err.cell_id.as_str()).or_default().push(CellError::Structural(err.clone()));
    }

    doc.cells
        .iter()
        .map(|cell| {
            let defs = match cell.kind_token {
                CellKindToken::Math => math_cell_defs(doc, cell, &mut resolved),
                CellKindToken::Table | CellKindToken::Js => Vec::new(),
            };
            let errors = errors_by_cell.remove(cell.id.as_str()).unwrap_or_default();
            CellEvalResult { cell_id: cell.id.clone(), kind: cell.kind_token, defs, errors }
        })
        .collect()
}

/// Step 1: this `math` cell's own [`super::MathCellDef`]s, in `def_line`
/// source order ([`super::MathCellDef::order`]), each paired with its
/// resolved result taken out of `resolved` (a def's name is unique
/// document-wide, so a `remove` never collides across cells).
fn math_cell_defs(
    doc: &WorkbookDoc,
    cell: &CellDoc,
    resolved: &mut HashMap<String, Result<crate::math::eval::EvalOutput, MathEvalError>>,
) -> Vec<CellDefResult> {
    let mut own: Vec<&super::MathCellDef> = doc.defs.iter().filter(|d| d.cell_id == cell.id).collect();
    own.sort_by_key(|d| d.order);

    own.into_iter()
        .map(|def| {
            // resolve_workbook_defs stores an entry for every def it is
            // handed, Ok or Err, regardless of whether anything references
            // it (its own doc comment) — a missing entry here is that
            // invariant broken, not a value this function can recover from.
            let result = resolved.remove(&def.name).expect("resolve_workbook_defs stores an entry for every def");
            match result {
                Ok(out) => CellDefResult {
                    name: def.name.clone(),
                    label: def.label.clone(),
                    value: Some(to_host_channel(&out.t_us, &out.samples)),
                    error: None,
                },
                Err(err) => {
                    CellDefResult { name: def.name.clone(), label: def.label.clone(), value: None, error: Some(err) }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::MathEvalErrorKind;
    use crate::workbook::v3::error::WorkbookErrorKind;
    use crate::workbook::v3::{ConstLine, MathCellDef, UnitsPref};

    struct EmptyLookup;
    impl ChannelLookup for EmptyLookup {
        fn lookup(&self, _name: &str) -> Option<crate::math::eval::LookupChannel> {
            None
        }
    }

    fn no_laps() -> MathLapContext {
        MathLapContext::empty()
    }

    fn math_cell(id: &str) -> CellDoc {
        CellDoc {
            id: id.to_string(),
            kind_token: CellKindToken::Math,
            prose_before: None,
            prose_after: None,
            raw_fence_body: String::new(),
            table: None,
        }
    }

    fn table_cell(id: &str) -> CellDoc {
        CellDoc { kind_token: CellKindToken::Table, ..math_cell(id) }
    }

    fn js_cell(id: &str) -> CellDoc {
        CellDoc { kind_token: CellKindToken::Js, ..math_cell(id) }
    }

    fn def(cell_id: &str, name: &str, expr_text: &str, order: usize) -> MathCellDef {
        MathCellDef { cell_id: cell_id.to_string(), name: name.to_string(), expr_text: expr_text.to_string(), label: None, order }
    }

    fn doc(cells: Vec<CellDoc>, defs: Vec<MathCellDef>) -> WorkbookDoc {
        WorkbookDoc {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Test".to_string(),
            constants_raw: HashMap::new(),
            units_pref: UnitsPref::Si,
            version: 3,
            cells,
            trailing_prose: None,
            const_lines: Vec::<ConstLine>::new(),
            defs,
            constants: HashMap::new(),
        }
    }

    // ---- Step 1/3: one entry per cell, in document order (L3-R25) ----

    #[test]
    fn two_math_cells_one_two_def_one_single_a_table_cell_a_js_cell_one_entry_per_cell_in_document_order_two_def_cell_has_two_defrslts(
    ) {
        // Arrange
        let cells = vec![math_cell("aaaaaaaa"), math_cell("bbbbbbbb"), table_cell("cccccccc"), js_cell("dddddddd")];
        let defs = vec![
            def("aaaaaaaa", "A", "1", 0),
            def("aaaaaaaa", "B", "2", 1),
            def("bbbbbbbb", "C", "3", 0),
        ];
        let d = doc(cells, defs);

        // Act
        let got = eval_cells(&d, &[], &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(got.len(), 4);
        assert_eq!(got[0].cell_id, "aaaaaaaa");
        assert_eq!(got[0].kind, CellKindToken::Math);
        assert_eq!(got[0].defs.len(), 2);
        assert_eq!(got[0].defs[0].name, "A");
        assert_eq!(got[0].defs[1].name, "B");
        assert_eq!(got[1].cell_id, "bbbbbbbb");
        assert_eq!(got[1].defs.len(), 1);
        assert_eq!(got[2].cell_id, "cccccccc");
        assert_eq!(got[2].kind, CellKindToken::Table);
        assert!(got[2].defs.is_empty());
        assert_eq!(got[3].cell_id, "dddddddd");
        assert_eq!(got[3].kind, CellKindToken::Js);
        assert!(got[3].defs.is_empty());
    }

    #[test]
    fn multi_definition_cells_defs_are_in_def_line_order_regardless_of_doc_defs_storage_order() {
        // Arrange — doc.defs deliberately out of `order` to prove the
        // orchestrator sorts, not just passes through storage order.
        let cells = vec![math_cell("aaaaaaaa")];
        let defs = vec![def("aaaaaaaa", "second", "2", 1), def("aaaaaaaa", "first", "1", 0)];
        let d = doc(cells, defs);

        // Act
        let got = eval_cells(&d, &[], &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(got[0].defs[0].name, "first");
        assert_eq!(got[0].defs[1].name, "second");
    }

    // ---- Step 1: def-level values/errors (CLAUDE.md §5 at this layer) ----

    #[test]
    fn one_definition_errors_its_cells_other_definition_still_returns_a_value() {
        // Arrange — "bad" references an unknown channel, "ok" is a scalar.
        let cells = vec![math_cell("aaaaaaaa")];
        let defs = vec![def("aaaaaaaa", "bad", "[Nope]", 0), def("aaaaaaaa", "ok", "5", 1)];
        let d = doc(cells, defs);

        // Act
        let got = eval_cells(&d, &[], &EmptyLookup, &no_laps());

        // Assert
        let bad = &got[0].defs[0];
        assert_eq!(bad.name, "bad");
        assert!(bad.value.is_none());
        assert_eq!(bad.error.as_ref().unwrap().kind, MathEvalErrorKind::UnknownChannel);

        let ok = &got[0].defs[1];
        assert_eq!(ok.name, "ok");
        assert_eq!(ok.value.as_ref().unwrap().v, vec![5.0]);
        assert!(ok.error.is_none());
    }

    // ---- Step 2: structural error routing by cell_id (L3-R2) ----

    #[test]
    fn a_cell_with_a_duplicate_definition_structural_error_the_error_appears_in_that_cells_errors_siblings_still_evaluate(
    ) {
        // Arrange
        let cells = vec![math_cell("aaaaaaaa"), math_cell("bbbbbbbb")];
        let defs = vec![def("aaaaaaaa", "x", "1", 0), def("bbbbbbbb", "y", "2", 0)];
        let d = doc(cells, defs);
        let structural = vec![WorkbookError::new("aaaaaaaa", WorkbookErrorKind::DuplicateDefinition, "'x' is defined more than once")];

        // Act
        let got = eval_cells(&d, &structural, &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(got[0].errors.len(), 1);
        assert_eq!(got[0].errors[0], CellError::Structural(structural[0].clone()));
        assert!(got[1].errors.is_empty());
        assert_eq!(got[1].defs[0].value.as_ref().unwrap().v, vec![2.0]);
    }

    #[test]
    fn a_front_matter_scoped_structural_error_dropped_appears_on_no_cell() {
        // Arrange
        let cells = vec![math_cell("aaaaaaaa")];
        let d = doc(cells, Vec::new());
        let structural = vec![WorkbookError::front_matter(WorkbookErrorKind::UnsupportedWorkbookVersion, "Workbook version 5 is not supported (expected 3)")];

        // Act
        let got = eval_cells(&d, &structural, &EmptyLookup, &no_laps());

        // Assert
        assert!(got[0].errors.is_empty());
    }

    #[test]
    fn a_table_cells_invalid_table_json_structural_error_appears_in_its_errors_defs_stays_empty() {
        // Arrange
        let cells = vec![table_cell("aaaaaaaa")];
        let d = doc(cells, Vec::new());
        let structural = vec![WorkbookError::new(
            "aaaaaaaa",
            WorkbookErrorKind::InvalidTableJson,
            "Table cell JSON is malformed: expected value at line 1 column 1",
        )];

        // Act
        let got = eval_cells(&d, &structural, &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(got[0].errors, vec![CellError::Structural(structural[0].clone())]);
        assert!(got[0].defs.is_empty());
    }
}
