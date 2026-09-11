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
///
/// **`lap_number` is 1-based** (C2 §4, ruling R217 item 2.4), matching C3's
/// `LapSummary.lap_number` and [`crate::laps::model::Lap::lap_number`]. It was
/// spelled `lapIndex` with no documented base until 2026-09-11, which left a
/// reader unable to tell whether `1` meant the first lap or the second; the
/// old key is still **accepted on read** (`serde(alias)`), so no stored table
/// fails to parse, and the writer emits the new one.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RowContext {
    pub session_id: String,
    #[serde(alias = "lapIndex")]
    pub lap_number: u32,
}

/// Where a table's rows come from (C2 §4, ruling R217 item 2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RowSource {
    /// The rows in [`TableModel::rows`], exactly as before this field
    /// existed. The default and the absent value, so every landed table is
    /// unchanged.
    #[default]
    Authored,
    /// One row per lap of each selected window, in window order then lap
    /// order. [`TableModel::rows`] is ignored and every cell evaluates its
    /// column's `template`.
    WindowLaps,
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
    /// Where the rows come from (C2 §4, ruling R217 item 2.1). Absent in
    /// every table written before 2026-09-11, and absent means
    /// [`RowSource::Authored`], so nothing landed changes.
    #[serde(default, skip_serializing_if = "is_default_row_source")]
    pub row_source: RowSource,
    /// The id of the row `main({col[]})` compares against — C2 §4's Main row,
    /// which populates `MathLapContext::baseline_row` (ruling R217 item 2.2).
    /// `None` leaves it unset and `main(...)` is `NaN`, exactly as before this
    /// field existed. The literal `"fastest"` is reserved and legal only under
    /// [`RowSource::WindowLaps`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main_row_id: Option<String>,
}

/// The reserved [`TableModel::main_row_id`] naming the fastest derived row
/// (C2 §4) — idl0's own default Main row.
pub const MAIN_ROW_FASTEST: &str = "fastest";

/// `skip_serializing_if` for [`TableModel::row_source`]: an absent key and
/// `"authored"` mean the same thing, and writing the default back would
/// rewrite every landed table's bytes for no change in meaning.
fn is_default_row_source(source: &RowSource) -> bool {
    *source == RowSource::Authored
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
