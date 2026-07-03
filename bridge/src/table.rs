//! FRB external-type mirrors for `idl_rs::table` value types.
//!
//! The `evaluate_table` wrapper itself lives in `session.rs` (beside the other
//! `SessionHandle` accessors) so it shares the canonical opaque-handle type —
//! see the opaque-type-duplication note in `session.rs`.

pub use idl_rs::table::{Cell, CellResult, Column, Row, RowContext, TableModel};

/// Mirror of [`idl_rs::table::Column`].
#[flutter_rust_bridge::frb(mirror(Column))]
pub struct _Column {
    pub id: String,
    pub name: Option<String>,
    pub template: Option<String>,
}

/// Mirror of [`idl_rs::table::RowContext`].
#[flutter_rust_bridge::frb(mirror(RowContext))]
pub struct _RowContext {
    pub session_id: String,
    pub lap_index: u32,
}

/// Mirror of [`idl_rs::table::Row`].
#[flutter_rust_bridge::frb(mirror(Row))]
pub struct _Row {
    pub id: String,
    pub context: Option<RowContext>,
}

/// Mirror of [`idl_rs::table::Cell`].
#[flutter_rust_bridge::frb(mirror(Cell))]
pub struct _Cell {
    pub formula: Option<String>,
    pub literal: Option<f64>,
    pub name: Option<String>,
}

/// Mirror of [`idl_rs::table::TableModel`].
#[flutter_rust_bridge::frb(mirror(TableModel))]
pub struct _TableModel {
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
    pub cells: Vec<Vec<Cell>>,
}

/// Mirror of [`idl_rs::table::CellResult`].
#[flutter_rust_bridge::frb(mirror(CellResult))]
pub struct _CellResult {
    pub value: Option<f64>,
    pub error: Option<String>,
}
