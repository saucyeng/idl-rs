//! Table evaluation engine (model + cell-formula evaluation). Reuses
//! `crate::math` — no second evaluator.
pub mod eval;
pub mod model;

pub use eval::{evaluate_table, evaluate_table_multi, lap_windows, validate};
pub use model::{Cell, CellResult, Column, Row, RowContext, TableModel, TableProblem};
