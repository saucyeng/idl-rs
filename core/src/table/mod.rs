//! Table evaluation engine (model + cell-formula evaluation). Reuses
//! `crate::math` — no second evaluator. See design
//! `docs/superpowers/specs/2026-06-15-modular-tables-design.md`.
pub mod eval;
pub mod model;

pub use eval::{evaluate_table, evaluate_table_multi, lap_windows, validate};
pub use model::{Cell, CellResult, Column, Row, RowContext, TableModel, TableProblem};
