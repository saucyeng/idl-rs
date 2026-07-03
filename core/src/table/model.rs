//! Table model: a grid of cells whose formulas reference other cells and
//! row-windowed channels. Serialized model only; values are computed by
//! [`crate::table::eval`].
//!
//! The model types derive `serde::{Serialize, Deserialize}` with camelCase keys
//! so the `TableModel` is the **portable artifact** — the CLI / Python / WASM
//! read it straight from the `.idl0wb` (the keys match the Dart `toJson`), not
//! merely an FRB mirror. See design §9a.

/// A column. `name` (when set) is the `{name}` / `{name[]}` reference target;
/// `template` is a formula applied to every cell in the column that has no own
/// formula (evaluated in each row's context).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Column {
    pub id: String,
    pub name: Option<String>,
    pub template: Option<String>,
}

/// A row. `context` binds a lap/session window so a cell's `[Channel]` refs
/// resolve to that window's samples.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub id: String,
    pub context: Option<RowContext>,
}

/// Binds a row to a lap of a session. The `[t0, t1]` window itself is supplied
/// to [`crate::table::evaluate_table`] (resolved app-side from the lap cache) —
/// the model only records which lap.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RowContext {
    pub session_id: String,
    pub lap_index: u32,
}

/// One cell. A `literal` short-circuits evaluation; otherwise the effective
/// formula is `formula` or the column's `template`. `name` lets a single cell
/// be a `{name}` target.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cell {
    pub formula: Option<String>,
    pub literal: Option<f64>,
    pub name: Option<String>,
}

/// The full table. `cells[r][c]` is the cell at row `r`, column `c`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableModel {
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
    pub cells: Vec<Vec<Cell>>,
}

/// Per-cell evaluation outcome: a value or a human-readable error.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CellResult {
    pub value: Option<f64>,
    pub error: Option<String>,
}

/// A structural problem found in a table by [`crate::table::validate`]. `row`
/// and `col` locate the offending cell when applicable (both `None` for a
/// whole-table problem such as a dimension mismatch or a cycle).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TableProblem {
    pub row: Option<usize>,
    pub col: Option<usize>,
    /// "dimension_mismatch" | "parse_error" | "unknown_reference" | "cycle".
    pub kind: String,
    pub message: String,
}
