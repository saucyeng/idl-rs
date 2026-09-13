//! Table evaluation engine (model + cell-formula evaluation). Reuses
//! `crate::math` — no second evaluator.
pub mod eval;
pub mod model;

pub use eval::{
    evaluate_table, evaluate_table_multi, plan_rows, resolve_baseline_row, validate, RowBinding,
    WindowLaps,
};
pub use model::{Cell, CellResult, Column, Row, RowContext, RowSource, TableModel, TableProblem, MAIN_ROW_FASTEST};
