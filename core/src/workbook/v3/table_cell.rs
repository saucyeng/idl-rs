//! `table` cell fence-body parsing (C2 §4) — a fence body is one JSON object,
//! the existing `idl_rs::table::TableModel` verbatim. Structural (parse-time)
//! only; dimension/cycle validation is evaluation-time (`table::validate`,
//! `table/eval.rs`), not this module's job.

use crate::table::TableModel;

use super::error::{self, WorkbookError};

// TODO(idl0): `TableModel` has no `#[serde(deny_unknown_fields)]` (an unknown
// key silently vanishes) and no `#[serde(skip_serializing_if)]` on its
// `Option` fields (a re-serialized empty `Cell` becomes
// `{"formula":null,"literal":null,"name":null}` instead of `{}`) — both gaps
// live in `table/model.rs`, shared with v2's `.idl0wb`, out of this lane's
// scope. Load-bearing for the save path and for C2 §7's per-cell merge,
// where every table cell would otherwise read as changed after one no-op
// save.

/// Parses a `table` cell's fence body (C2 §4) as `TableModel` JSON.
/// `cell_id` scopes a failure to the owning cell (L3-R2); `fence_body` is
/// the fence's raw, unparsed text. `table::validate`'s dimension/cycle
/// checks are not run here — they are evaluation-time.
pub fn parse_table_cell(cell_id: &str, fence_body: &str) -> Result<TableModel, WorkbookError> {
    serde_json::from_str::<TableModel>(fence_body).map_err(|e| error::invalid_table_json(cell_id, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::WorkbookErrorKind;
    use crate::table::{Cell, RowContext};

    /// C2 §4's minimal worked example, restated literally.
    const WORKED_EXAMPLE: &str = r#"{
  "columns": [
    { "id": "c0", "name": "lap" },
    { "id": "c1", "name": "fork_max", "template": "max([Fork travel])" }
  ],
  "rows": [
    { "id": "r0", "context": { "sessionId": "s1", "lapIndex": 1 } },
    { "id": "r1", "context": { "sessionId": "s1", "lapIndex": 2 } }
  ],
  "cells": [
    [ { "literal": 1 }, {} ],
    [ { "literal": 2 }, {} ]
  ]
}"#;

    #[test]
    fn c2_4_worked_example_parses_ok_columns_rows_and_cells_match() {
        // Act
        let table = parse_table_cell("aaaaaaaa", WORKED_EXAMPLE).unwrap();

        // Assert
        assert_eq!(table.columns[0].template, None);
        assert_eq!(table.rows[0].context, Some(RowContext { session_id: "s1".to_string(), lap_number: 1 }));
        assert_eq!(table.cells[0][1], Cell::default());
    }

    #[test]
    fn row_context_the_retired_lap_index_key_and_the_current_lap_number_key_parse_to_the_same_row() {
        // Arrange — the same table twice, spelled the old way and the new way.
        let old = r#"{ "columns": [], "rows": [ { "id": "r0", "context": { "sessionId": "s1", "lapIndex": 4 } } ], "cells": [] }"#;
        let new = r#"{ "columns": [], "rows": [ { "id": "r0", "context": { "sessionId": "s1", "lapNumber": 4 } } ], "cells": [] }"#;

        // Act
        let from_old = parse_table_cell("aaaaaaaa", old).unwrap();
        let from_new = parse_table_cell("aaaaaaaa", new).unwrap();

        // Assert — the rename fixed the name, not the numbers (C2 §4, R217 item 2.4).
        assert_eq!(from_old.rows[0].context, from_new.rows[0].context);
        assert_eq!(from_old.rows[0].context.as_ref().unwrap().lap_number, 4);
    }

    #[test]
    fn table_model_row_source_defaults_to_authored_and_main_row_id_to_unset() {
        // Arrange — a table written before either field existed.
        let json = r#"{ "columns": [], "rows": [], "cells": [] }"#;

        // Act
        let table = parse_table_cell("aaaaaaaa", json).unwrap();

        // Assert — nothing landed changes meaning (C2 §4, R217 items 2.1-2.2).
        assert_eq!(table.row_source, crate::table::RowSource::Authored);
        assert_eq!(table.main_row_id, None);
    }

    #[test]
    fn table_model_window_laps_and_a_main_row_round_trip_through_the_fence() {
        // Arrange
        let json = r#"{ "columns": [], "rows": [], "cells": [], "rowSource": "windowLaps", "mainRowId": "fastest" }"#;

        // Act
        let table = parse_table_cell("aaaaaaaa", json).unwrap();

        // Assert
        assert_eq!(table.row_source, crate::table::RowSource::WindowLaps);
        assert_eq!(table.main_row_id.as_deref(), Some(crate::table::MAIN_ROW_FASTEST));
    }

    #[test]
    fn fence_body_not_valid_json_err_invalid_table_json_message_includes_serde_error() {
        // Arrange
        let fence_body = "{ not valid json";

        // Act
        let err = parse_table_cell("aaaaaaaa", fence_body).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::InvalidTableJson);
        assert_eq!(err.cell_id, "aaaaaaaa");
        assert!(err.message.starts_with("Table cell JSON is malformed: "));
    }
}
