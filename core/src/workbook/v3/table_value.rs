//! A `table` cell evaluated in one window (C2 §4): the planned rows and their
//! results, with `[name]` resolving to the workbook's definitions first and
//! the session's channels second (ruling Q6, 2026-10-28).
//!
//! **Where Q6 was lost.** idl0's tables saw math channels because the app
//! evaluated every math channel *into the session's math store*
//! (`SessionHandle::store_math`, `workbook::apply`) before a table read the
//! same handle, so `[Name]` found them as channels. Workbook v3 resolves
//! definitions in their own overlay (`resolve.rs`) and never writes them to
//! the handle, and the first v3 table wiring (`idl-rs-tauri`
//! `table_cell_value`, 2026-09-04, `8d737fe`) evaluated tables against the
//! bare handle — so from then on a table saw channels only. C2 §4 kept
//! idl0's "`[Channel]` refs" wording, which read as channels-only once
//! definitions were no longer channels. This module is the shared fix: the
//! app's per-window path and the CLI both build table values here.

use crate::math::eval::MathLapContext;
use crate::math::ValueAxis;
use crate::session::handle::SessionHandle;
use crate::table::{
    evaluate_table_with_definitions, plan_rows, resolve_baseline_row, CellResult, DefinitionValue, DefinitionValues,
    TableModel, WindowLaps,
};

use super::host_channel_wire::AxisKind;
use super::CellEvalResult;

/// Every evaluated definition in `cells`, by name, as a table cell reads it:
/// its value, or its error message when it failed. A name defined twice
/// keeps its first definition in document order (R49) whatever that one's
/// outcome, as a math cell's `[name]` does — a failed first definition is
/// never replaced by a later one, nor by a channel.
pub fn definition_values(cells: &[CellEvalResult]) -> DefinitionValues {
    let mut out = DefinitionValues::default();
    for def in cells.iter().flat_map(|c| c.defs.iter()) {
        if out.0.contains_key(&def.name) {
            continue;
        }
        let entry = match (&def.value, &def.error) {
            (Some(view), _) => Ok(DefinitionValue {
                samples: view.v.clone(),
                sample_rate_hz: def.sample_rate_hz.unwrap_or(0.0),
                t_us: view.t_us.clone(),
                axis: if view.axis == AxisKind::Lap { ValueAxis::Lap } else { ValueAxis::Time },
            }),
            (None, Some(e)) => Err(e.message.clone()),
            (None, None) => Err("no value".to_string()),
        };
        out.0.insert(def.name.clone(), entry);
    }
    out
}

/// Evaluates table `model` in the window `handle`/`lap_ctx` describe: rows
/// planned by C2 §4's `rowSource` rule over the window's laps, each row's
/// `[name]` resolving to `definitions` first. Returns the planned model (the
/// rows actually evaluated) and one result per cell. A `rowSource:
/// "sessions"` table plans no rows here; it is a set table
/// ([`super::set_eval`]).
pub fn evaluate_table_in_window(
    model: &TableModel,
    definitions: &DefinitionValues,
    handle: &SessionHandle,
    lap_ctx: &MathLapContext,
) -> (TableModel, Vec<Vec<CellResult>>) {
    let windows = [WindowLaps { session_id: handle.metadata().session_id, laps: lap_ctx.laps.clone() }];
    let (planned, bindings) = plan_rows(model, &lap_ctx.laps, &windows);
    let baseline = resolve_baseline_row(&planned, &bindings);
    let results = evaluate_table_with_definitions(handle, definitions, &planned, &bindings, baseline);
    (planned, results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionMetaInput};
    use crate::table::{Cell, Column, Row};
    use crate::workbook::v3::{eval_cells, parse_workbook};

    fn channel(id: &str, samples: Vec<f64>) -> ChannelInput {
        let t_us = (0..samples.len() as i64).map(|i| i * 1_000_000).collect();
        ChannelInput { channel_id: id.into(), sample_rate_hz: 1.0, samples, t_us, source_kind: "test".into() }
    }

    fn handle(samples: Vec<f64>) -> SessionHandle {
        SessionHandle::from_channels(
            SessionMetaInput { session_id: "s1".into(), device_id: None, timestamp_utc_ms: 0, config_checksum: None },
            vec![channel("Fork", samples), channel("Speed", vec![1.0, 2.0, 3.0])],
        )
    }

    fn one_row_table(template: &str) -> TableModel {
        TableModel {
            columns: vec![Column { id: "c0".into(), name: Some("x".into()), template: Some(template.into()) }],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell::default()]],
            row_source: Default::default(),
            main_row_id: None,
            row_group: None,
        }
    }

    const MD: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: T\n---\n\n```math id=aaaaaaaa\nfork_max = max([Fork])\nSpeed = 42\ndoubled = [Fork] * 2\n```\n";

    #[test]
    fn table_template_naming_a_definition_reads_the_definition_not_an_unknown_channel() {
        // Arrange
        let h = handle(vec![1.0, 5.0, 3.0]);
        let (doc, structural) = parse_workbook(MD).unwrap();
        let cells = eval_cells(&doc, &structural, &h, &MathLapContext::empty());

        // Act
        let (_, results) =
            evaluate_table_in_window(&one_row_table("[fork_max] + 1"), &definition_values(&cells), &h, &MathLapContext::empty());

        // Assert
        assert_eq!(results[0][0], CellResult { value: Some(6.0), error: None });
    }

    #[test]
    fn table_template_a_full_rate_definition_is_reduced_in_the_cell() {
        // Arrange
        let h = handle(vec![1.0, 5.0, 3.0]);
        let (doc, structural) = parse_workbook(MD).unwrap();
        let cells = eval_cells(&doc, &structural, &h, &MathLapContext::empty());

        // Act
        let (_, results) =
            evaluate_table_in_window(&one_row_table("max([doubled])"), &definition_values(&cells), &h, &MathLapContext::empty());

        // Assert
        assert_eq!(results[0][0].value, Some(10.0));
    }

    #[test]
    fn table_template_a_definition_named_like_a_channel_wins_over_the_channel() {
        // Arrange
        let h = handle(vec![1.0, 5.0, 3.0]);
        let (doc, structural) = parse_workbook(MD).unwrap();
        let cells = eval_cells(&doc, &structural, &h, &MathLapContext::empty());

        // Act
        let (_, results) =
            evaluate_table_in_window(&one_row_table("max([Speed])"), &definition_values(&cells), &h, &MathLapContext::empty());

        // Assert
        assert_eq!(results[0][0].value, Some(42.0));
    }

    #[test]
    fn table_template_a_failed_definition_named_like_a_channel_is_its_error_never_the_channel() {
        // Arrange — `Speed` fails (unknown function), and a `Speed` channel exists.
        let md = MD.replace("Speed = 42", "Speed = nosuchfn(1)");
        let h = handle(vec![1.0, 5.0, 3.0]);
        let (doc, structural) = parse_workbook(&md).unwrap();
        let cells = eval_cells(&doc, &structural, &h, &MathLapContext::empty());

        // Act
        let (_, results) =
            evaluate_table_in_window(&one_row_table("max([Speed])"), &definition_values(&cells), &h, &MathLapContext::empty());

        // Assert
        assert_eq!(results[0][0].value, None);
        assert!(results[0][0].error.as_deref().unwrap().starts_with("[Speed] failed:"), "{:?}", results[0][0]);
    }

    #[test]
    fn definition_values_a_duplicated_name_whose_first_definition_failed_keeps_the_failure() {
        // Arrange
        let md = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: T\n---\n\n```math id=aaaaaaaa\nSpeed = nosuchfn(1)\n```\n\n```math id=bbbbbbbb\nSpeed = 7\n```\n";
        let h = handle(vec![1.0]);
        let (doc, structural) = parse_workbook(md).unwrap();
        let cells = eval_cells(&doc, &structural, &h, &MathLapContext::empty());

        // Act
        let defs = definition_values(&cells);

        // Assert
        assert!(defs.0["Speed"].is_err());
    }

    #[test]
    fn table_template_a_name_that_is_no_definition_still_reads_the_channel() {
        // Arrange
        let h = handle(vec![1.0, 5.0, 3.0]);

        // Act
        let (_, results) = evaluate_table_in_window(
            &one_row_table("max([Fork])"),
            &DefinitionValues::default(),
            &h,
            &MathLapContext::empty(),
        );

        // Assert
        assert_eq!(results[0][0].value, Some(5.0));
    }
}
