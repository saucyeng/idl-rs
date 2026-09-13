//! Table evaluation: cell addressing, dependency ordering, and per-cell
//! evaluation. Reuses the `idl-rs` math evaluator — there is no second
//! expression engine. Each cell is evaluated with [`crate::math::evaluate_scalar`]
//! against a [`CellLookup`] that resolves `{cell}` references to already-computed
//! values and `[Channel]` references to the row's lap-windowed samples.

use std::collections::HashMap;

use crate::math::eval::LapSpan;
use crate::math::parse::parse;
use crate::math::{evaluate_scalar, ChannelLookup, LookupChannel, MathLapContext};
use crate::session::handle::SessionHandle;
use crate::table::model::{Cell, CellResult, Row, RowContext, RowSource, TableModel, TableProblem, MAIN_ROW_FASTEST};

/// A cell coordinate, `(row, col)`.
pub type Addr = (usize, usize);

/// Effective formula for cell `(r, c)`: its own formula, else the column
/// template. `None` for a literal/blank cell.
pub(crate) fn effective_formula(t: &TableModel, r: usize, c: usize) -> Option<&str> {
    let cell = &t.cells[r][c];
    if cell.literal.is_some() {
        return None;
    }
    cell.formula.as_deref().or(t.columns[c].template.as_deref())
}

/// Parse the `{ … }` bodies in `expr` (mirrors the brace scan in the tokenizer).
fn cell_ref_bodies(expr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = expr;
    while let Some(o) = rest.find('{') {
        rest = &rest[o + 1..];
        match rest.find('}') {
            Some(cl) => {
                let b = &rest[..cl];
                if !b.is_empty() {
                    out.push(b.to_string());
                }
                rest = &rest[cl + 1..];
            }
            None => break,
        }
    }
    out
}

/// `A`→0, `B`→1, …, `Z`→25, `AA`→26, … `None` if not all-uppercase-ASCII.
fn col_letters_to_index(s: &str) -> Option<usize> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_uppercase()) {
        return None;
    }
    let mut n = 0usize;
    for b in s.bytes() {
        n = n * 26 + (b - b'A' + 1) as usize;
    }
    Some(n - 1)
}

/// Resolve a `{body}` (from cell at `(r, c)`) to the cell addresses it depends
/// on. `{A1}`→one cell; `{name}`→same-row named column; `{name[]}`→all cells in
/// the named column.
fn deps_of_body(t: &TableModel, r: usize, body: &str) -> Vec<Addr> {
    let col_by_name = |name: &str| t.columns.iter().position(|c| c.name.as_deref() == Some(name));
    if let Some(name) = body.strip_suffix("[]") {
        if let Some(ci) = col_by_name(name) {
            return (0..t.rows.len()).map(|ri| (ri, ci)).collect();
        }
        return vec![];
    }
    // A1 form: leading uppercase letters then digits.
    let split = body.find(|ch: char| ch.is_ascii_digit());
    if let Some(i) = split {
        let (letters, digits) = body.split_at(i);
        let letters = letters.replace('$', "");
        let digits = digits.replace('$', "");
        if let (Some(ci), Ok(rn)) = (col_letters_to_index(&letters), digits.parse::<usize>()) {
            if rn >= 1 && rn - 1 < t.rows.len() && ci < t.columns.len() {
                return vec![(rn - 1, ci)];
            }
        }
        return vec![];
    }
    // Named column, same row.
    col_by_name(body).map(|ci| vec![(r, ci)]).unwrap_or_default()
}

/// Direct dependency addresses of cell `(r, c)`.
pub(crate) fn deps_of(t: &TableModel, r: usize, c: usize) -> Vec<Addr> {
    match effective_formula(t, r, c) {
        Some(f) => cell_ref_bodies(f).into_iter().flat_map(|b| deps_of_body(t, r, &b)).collect(),
        None => vec![],
    }
}

/// Topological order of all cells, dependencies first. `Err` lists the cells on
/// a cycle.
pub(crate) fn topo_order(t: &TableModel) -> Result<Vec<Addr>, Vec<Addr>> {
    let mut order = Vec::new();
    let mut state: HashMap<Addr, u8> = HashMap::new(); // 0=unseen,1=visiting,2=done
    let mut cycle = Vec::new();
    fn visit(
        t: &TableModel,
        a: Addr,
        state: &mut HashMap<Addr, u8>,
        order: &mut Vec<Addr>,
        cycle: &mut Vec<Addr>,
    ) -> bool {
        match state.get(&a).copied().unwrap_or(0) {
            2 => return true,
            1 => {
                cycle.push(a);
                return false;
            }
            _ => {}
        }
        state.insert(a, 1);
        for d in deps_of(t, a.0, a.1) {
            if !visit(t, d, state, order, cycle) {
                cycle.push(a);
                return false;
            }
        }
        state.insert(a, 2);
        order.push(a);
        true
    }
    for r in 0..t.rows.len() {
        for c in 0..t.columns.len() {
            if !visit(t, (r, c), &mut state, &mut order, &mut cycle) {
                return Err(cycle);
            }
        }
    }
    Ok(order)
}

/// Lookup for one cell: cell values from already-evaluated cells, plus channels
/// sliced to this row's window. `[Channel]` → row-windowed samples; `{cell}` →
/// a prior result; `{col[]}` → a whole column's prior results.
struct CellLookup<'a> {
    handle: &'a SessionHandle,
    window: Option<(f64, f64)>,
    /// Resolved scalar per evaluated cell address.
    values: &'a HashMap<Addr, f64>,
    /// Column index by name (for `{name}` same-row and `{name[]}`).
    col_by_name: &'a HashMap<String, usize>,
    /// This cell's row (for `{name}` same-row resolution).
    row: usize,
    /// Total row count (for `{col[]}` column gather).
    rows: usize,
}

impl ChannelLookup for CellLookup<'_> {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        let samples = match self.window {
            Some((t0, t1)) => self.handle.slice_by_time(name, t0, t1),
            None => self.handle.materialize_f64(name, 0, u32::MAX),
        };
        if samples.is_empty() {
            return None;
        }
        // Rate is whatever the source channel reports; aggregates ignore it, and
        // a cell must reduce to a scalar so the rate is never surfaced. Rate-0
        // presentation — t_us is the "no time axis" marker (L3-R12), never a
        // synthetic ramp.
        Some(LookupChannel {
            samples: std::sync::Arc::from(samples),
            sample_rate_hz: 0.0,
            t_us: std::sync::Arc::from(&[] as &[i64]),
        })
    }
    /// Populate the estimator's outputs into the session store via the handle,
    /// then read the channel back through [`Self::lookup`] so a table cell
    /// still honours its row's time window — a raw forward would hand back the
    /// whole session regardless of the window.
    fn estimator_channel(&self, channel_id: &str) -> Option<LookupChannel> {
        self.handle.estimator_channel(channel_id)?;
        self.lookup(channel_id)
    }

    fn lookup_cell(&self, body: &str) -> Option<f64> {
        let addr = single_addr(self, body)?;
        self.values.get(&addr).copied()
    }
    fn lookup_cell_column(&self, name: &str) -> Option<Vec<f64>> {
        let ci = *self.col_by_name.get(name)?;
        Some((0..self.rows).filter_map(|ri| self.values.get(&(ri, ci)).copied()).collect())
    }
}

/// Resolve a non-`[]` `{body}` to one address relative to the lookup's row.
fn single_addr(l: &CellLookup, body: &str) -> Option<Addr> {
    let split = body.find(|ch: char| ch.is_ascii_digit());
    if let Some(i) = split {
        let (letters, digits) = body.split_at(i);
        let ci = col_letters_to_index(&letters.replace('$', ""))?;
        let rn: usize = digits.replace('$', "").parse().ok()?;
        return Some((rn.checked_sub(1)?, ci));
    }
    l.col_by_name.get(body).map(|&ci| (l.row, ci))
}

/// What one table row is evaluated against: the time window its `[Channel]`
/// references resolve in, and the lap it is bound to when it is bound to one.
///
/// The two are not the same fact and are not derivable from each other. A row
/// can have a window without a lap (a range binding), and the lap is what
/// makes `lap_time()` a scalar in that row rather than a `[lap]` series
/// (`MathLapContext::row_lap`, ruling R217 item 3) — a window alone cannot say
/// which lap number a cell is speaking for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowBinding {
    /// The row's `[t0, t1)` in session-relative seconds, or `None` to read the
    /// whole channel.
    pub window: Option<(f64, f64)>,
    /// The lap this row is a row *of*, when it is one.
    pub lap: Option<LapSpan>,
}

/// One selected window, resolved to the laps it covers — the input to C2 §4's
/// `rowSource: "windowLaps"` derivation. The caller resolves the selection
/// (which is a UI fact, and out here a `WindowDto`) into this; `core` never
/// reads a selection itself.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowLaps {
    /// The session the window names, for the derived rows' `RowContext`.
    pub session_id: String,
    /// That window's laps, in lap order.
    pub laps: Vec<LapSpan>,
}

/// Applies C2 §4's `rowSource` rule, returning the table as it is actually
/// evaluated plus one [`RowBinding`] per row.
///
/// Under [`RowSource::Authored`] (the default, and every table written before
/// the field existed) this is the identity on the model, and the bindings come
/// from matching each row's `context.lap_number` against `session_laps`.
///
/// Under [`RowSource::WindowLaps`] the row set is **derived**: one row per lap
/// of each selected window, in window order then lap order, and the authored
/// `rows`/`cells` are ignored — not merged and not appended, because two row
/// sets would have two orderings and no rule for interleaving them. A derived
/// row has no authored cells, so every one of its cells falls through to its
/// column's `template`; a column with no template renders empty in every
/// derived row. The returned model's `cells` is therefore a rows × columns
/// grid of blank [`Cell`]s, which is exactly what `effective_formula` reads as
/// "use the template".
///
/// `session_laps` is unused under `WindowLaps` (the windows carry their own
/// laps) and is the authored path's lap source; `windows` is unused under
/// `Authored`. Both are taken so one call site covers both row sources.
pub fn plan_rows(
    table: &TableModel,
    session_laps: &[LapSpan],
    windows: &[WindowLaps],
) -> (TableModel, Vec<RowBinding>) {
    match table.row_source {
        RowSource::Authored => {
            let bindings = table
                .rows
                .iter()
                .map(|row| match &row.context {
                    Some(ctx) => {
                        // Matched by lap *number*, never by position (C2 §4,
                        // ruling R217 item 2.4). Indexing `session_laps`
                        // directly would silently make the field 0-based while
                        // every other lap-numbered surface — C3's
                        // `LapSummary.lap_number`, `Lap::lap_number`,
                        // `current_lap()` — is 1-based, and matching by number
                        // also survives an ignored or renumbered lap, where a
                        // position quietly slides to a neighbour instead of
                        // resolving to nothing.
                        match session_laps.iter().find(|l| l.lap_number == ctx.lap_number) {
                            Some(lap) => RowBinding {
                                window: Some((lap.start_secs, lap.end_secs)),
                                lap: Some(lap.clone()),
                            },
                            None => RowBinding::default(),
                        }
                    }
                    None => RowBinding::default(),
                })
                .collect();
            (table.clone(), bindings)
        }
        RowSource::WindowLaps => {
            let mut rows = Vec::new();
            let mut bindings = Vec::new();
            for window in windows {
                for lap in &window.laps {
                    rows.push(Row {
                        // Stable and readable: the session and lap a row
                        // stands for are exactly what identifies it, and a
                        // derived row has no authored id to keep.
                        id: format!("{}#{}", window.session_id, lap.lap_number),
                        context: Some(RowContext {
                            session_id: window.session_id.clone(),
                            lap_number: lap.lap_number,
                        }),
                    });
                    bindings.push(RowBinding {
                        window: Some((lap.start_secs, lap.end_secs)),
                        lap: Some(lap.clone()),
                    });
                }
            }
            let cells = vec![vec![Cell::default(); table.columns.len()]; rows.len()];
            (TableModel { columns: table.columns.clone(), rows, cells, ..table.clone() }, bindings)
        }
    }
}

/// Resolves C2 §4's `mainRowId` to the row index `main({col[]})` reads from
/// (`MathLapContext::baseline_row`). `None` when the field is absent, names no
/// row, or asks for something this row source cannot provide.
///
/// The reserved literal `"fastest"` is the row with the smallest `lap_time()`
/// among the derived rows — idl0's own default Main row — and is legal **only**
/// under [`RowSource::WindowLaps`]. Under `Authored` it resolves to nothing and
/// [`validate`] reports it: an authored table's rows are named, so a magic id
/// there would shadow a real row id.
///
/// A lap whose recorded time is `NaN` is skipped rather than compared; ties go
/// to the lowest index, so the answer does not depend on iteration luck.
pub fn resolve_baseline_row(table: &TableModel, bindings: &[RowBinding]) -> Option<usize> {
    let id = table.main_row_id.as_deref()?;
    if id == MAIN_ROW_FASTEST {
        if table.row_source != RowSource::WindowLaps {
            return None;
        }
        return bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.lap.as_ref().map(|l| (i, l.lap_time_secs)))
            .filter(|(_, t)| !t.is_nan())
            .min_by(|(ia, a), (ib, b)| a.partial_cmp(b).unwrap().then(ia.cmp(ib)))
            .map(|(i, _)| i);
    }
    table.rows.iter().position(|r| r.id == id)
}

/// Evaluate every cell. `bindings[r]` is row `r`'s window and lap (see
/// [`RowBinding`]). Single-handle convenience over [`evaluate_table_multi`].
pub fn evaluate_table(
    handle: &SessionHandle,
    table: &TableModel,
    bindings: &[RowBinding],
) -> Vec<Vec<CellResult>> {
    let row_handles = vec![0usize; table.rows.len()];
    evaluate_table_multi(&[handle], &row_handles, table, bindings, None)
}

/// Evaluate a table whose rows may bind different sessions. `handles` is the
/// distinct session pool; `row_handles[r]` indexes it for row `r`'s `[Channel]`
/// resolution. `{cell}` / `{col[]}` cross-row references resolve from the global
/// values map exactly as in the single-handle path — one pass over the whole
/// grid. `baseline_row`, when set, is the row `main({col[]})` reads from.
pub fn evaluate_table_multi(
    handles: &[&SessionHandle],
    row_handles: &[usize],
    table: &TableModel,
    bindings: &[RowBinding],
    baseline_row: Option<usize>,
) -> Vec<Vec<CellResult>> {
    let cols = table.columns.len();
    let mut out: Vec<Vec<CellResult>> =
        table.rows.iter().map(|_| vec![CellResult { value: None, error: None }; cols]).collect();

    let col_by_name: HashMap<String, usize> = table
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.name.clone().map(|n| (n, i)))
        .collect();

    let order = match topo_order(table) {
        Ok(o) => o,
        Err(cycle) => {
            for (r, c) in cycle {
                out[r][c] = CellResult { value: None, error: Some("Circular reference".into()) };
            }
            return out;
        }
    };

    let mut values: HashMap<Addr, f64> = HashMap::new();
    // One context per row, not one per table: a row bound to a lap makes the
    // lap-shaped builtins scalars *for that row* (ruling R217 item 3), and the
    // lap they speak for differs row to row. `main_lap_bounds` stays empty —
    // the row's window narrows channels in `CellLookup` (which reports rate 0,
    // so `window_index_range` never narrows again) and setting it here would
    // window an already-windowed slice a second time.
    let row_ctx = |binding: &RowBinding| MathLapContext {
        baseline_row,
        laps: binding.lap.clone().into_iter().collect(),
        row_lap: binding.lap.as_ref().map(|l| l.lap_number),
        ..MathLapContext::empty()
    };
    for (r, c) in order {
        // Literal short-circuit.
        if let Some(v) = table.cells[r][c].literal {
            values.insert((r, c), v);
            out[r][c] = CellResult { value: Some(v), error: None };
            continue;
        }
        let Some(formula) = effective_formula(table, r, c) else {
            continue; // blank cell
        };
        // Row r resolves `[Channel]` against its own session handle.
        let handle = handles[row_handles.get(r).copied().unwrap_or(0)];
        let binding = bindings.get(r).cloned().unwrap_or_default();
        let lap_ctx = row_ctx(&binding);
        let lookup = CellLookup {
            handle,
            window: binding.window,
            values: &values,
            col_by_name: &col_by_name,
            row: r,
            rows: table.rows.len(),
        };
        match evaluate_scalar(formula, &lookup, &lap_ctx) {
            Ok(v) => {
                values.insert((r, c), v);
                out[r][c] = CellResult { value: Some(v), error: None };
            }
            Err(e) => {
                out[r][c] = CellResult { value: None, error: Some(e.message) };
            }
        }
    }
    out
}


/// Static validation of a table (no session): `cells` is rows×cols, every
/// effective formula parses, every `{…}` reference resolves to a real
/// column/cell, and there is no dependency cycle. Channel-existence
/// (`[Channel]`) is *not* checked here — that needs a session, so the CLI's
/// `check` runs an eval pass for it.
pub fn validate(table: &TableModel) -> Vec<TableProblem> {
    let mut problems = Vec::new();
    let cols = table.columns.len();

    // 1. Dimensions. A mismatch makes per-cell indexing unsafe, so bail early.
    if table.cells.len() != table.rows.len() {
        problems.push(TableProblem {
            row: None,
            col: None,
            kind: "dimension_mismatch".into(),
            message: format!("cells has {} rows, expected {}", table.cells.len(), table.rows.len()),
        });
        return problems;
    }
    for (r, row_cells) in table.cells.iter().enumerate() {
        if row_cells.len() != cols {
            problems.push(TableProblem {
                row: Some(r),
                col: None,
                kind: "dimension_mismatch".into(),
                message: format!("row {r} has {} cells, expected {cols}", row_cells.len()),
            });
        }
    }
    if !problems.is_empty() {
        return problems;
    }

    // 2. Per-cell: parse the effective formula, then resolve each `{…}` ref.
    for r in 0..table.rows.len() {
        for c in 0..cols {
            let Some(formula) = effective_formula(table, r, c) else { continue };
            if let Err(e) = parse(formula) {
                problems.push(TableProblem {
                    row: Some(r),
                    col: Some(c),
                    kind: "parse_error".into(),
                    message: e.message,
                });
                continue;
            }
            for body in cell_ref_bodies(formula) {
                if deps_of_body(table, r, &body).is_empty() {
                    problems.push(TableProblem {
                        row: Some(r),
                        col: Some(c),
                        kind: "unknown_reference".into(),
                        message: format!("reference {{{body}}} does not resolve"),
                    });
                }
            }
        }
    }

    // 3. The reserved Main row id (C2 §4 rule 2). `"fastest"` names the
    //    fastest *derived* row, so it only means anything under
    //    `rowSource: "windowLaps"`. Under `"authored"` it is reported rather
    //    than silently ignored: an authored table's rows are named, so a magic
    //    id there would shadow a real row id, and a Main row that quietly
    //    resolved to nothing turns every `main(...)` into NaN with no clue why.
    if table.main_row_id.as_deref() == Some(MAIN_ROW_FASTEST)
        && table.row_source != RowSource::WindowLaps
    {
        problems.push(TableProblem {
            row: None,
            col: None,
            kind: "invalid_main_row".into(),
            message: format!(
                "mainRowId '{MAIN_ROW_FASTEST}' is reserved for rowSource \"windowLaps\";                  name one of this table's own row ids instead"
            ),
        });
    }

    // 4. Cycle detection over the whole grid.
    if let Err(cycle) = topo_order(table) {
        let cells: Vec<String> = cycle.iter().map(|(r, c)| format!("({r},{c})")).collect();
        problems.push(TableProblem {
            row: None,
            col: None,
            kind: "cycle".into(),
            message: format!("circular reference among cells: {}", cells.join(", ")),
        });
    }

    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::lap_ops::lap_spans;
    use crate::laps::model::Lap;

    /// A row bound to a plain time window and no lap — what most of these
    /// tests need, since they are about cell addressing, not lap shapes.
    fn windowed(t0: f64, t1: f64) -> RowBinding {
        RowBinding { window: Some((t0, t1)), lap: None }
    }
    use crate::session::handle::{ChannelInput, SessionMetaInput};
    use crate::table::model::*;

    fn lap(n: u32, t0: f64, t1: f64) -> Lap {
        Lap {
            lap_number: n + 1,
            start_ms: 0,
            end_ms: 0,
            start_time_secs: t0,
            end_time_secs: t1,
            raw_elapsed_ms: 0,
            lap_time_ms: 0,
            sectors: vec![],
            neutral_zone_visits: vec![],
        }
    }

    #[test]
    fn plan_rows_authored_matches_the_1_based_lap_number_and_is_unbound_for_no_or_unknown_context() {
        // Arrange — laps numbered 1 and 2; rows bound to lap 1, unbound, and
        // to a lap number no lap carries.
        let laps = lap_spans(&[lap(0, 10.0, 20.0), lap(1, 20.0, 35.0)]);
        let t = TableModel {
            columns: vec![],
            rows: vec![
                Row { id: "r0".into(), context: Some(RowContext { session_id: "s".into(), lap_number: 1 }) },
                Row { id: "r1".into(), context: None },
                Row { id: "r2".into(), context: Some(RowContext { session_id: "s".into(), lap_number: 9 }) },
            ],
            cells: vec![vec![], vec![], vec![]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act
        let (planned, bindings) = plan_rows(&t, &laps, &[]);

        // Assert — lap *number* 1 is the first lap (C2 §4, R217 item 2.4); it
        // was the second one while this field was an index into `laps`.
        assert_eq!(planned.rows, t.rows, "authored rows pass through untouched");
        assert_eq!(bindings[0].window, Some((10.0, 20.0)));
        assert_eq!(bindings[0].lap.as_ref().unwrap().lap_number, 1);
        assert_eq!(bindings[1], RowBinding::default());
        assert_eq!(bindings[2], RowBinding::default());
    }

    #[test]
    fn validate_flags_dimension_mismatch() {
        // Arrange — 1 row declared but 0 cell-rows.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act + Assert
        assert!(validate(&t).iter().any(|p| p.kind == "dimension_mismatch"));
    }

    #[test]
    fn validate_flags_parse_error() {
        // Arrange — an unbalanced bracket.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell { formula: Some("max([Fork)".into()), ..Default::default() }]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act + Assert
        assert!(validate(&t).iter().any(|p| p.kind == "parse_error"));
    }

    #[test]
    fn validate_flags_unknown_reference() {
        // Arrange — `{nope}` references a column/cell that does not exist.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell { formula: Some("{nope}".into()), ..Default::default() }]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act + Assert
        assert!(validate(&t).iter().any(|p| p.kind == "unknown_reference"));
    }

    #[test]
    fn validate_flags_cycle() {
        // Arrange — A1 = {A2}, A2 = {A1}.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }, Row { id: "r1".into(), context: None }],
            cells: vec![
                vec![Cell { formula: Some("{A2}".into()), ..Default::default() }],
                vec![Cell { formula: Some("{A1}".into()), ..Default::default() }],
            ],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act + Assert
        assert!(validate(&t).iter().any(|p| p.kind == "cycle"));
    }

    #[test]
    fn validate_clean_table_has_no_problems() {
        // Arrange — a single literal cell.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell { literal: Some(1.0), ..Default::default() }]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act + Assert
        assert!(validate(&t).is_empty());
    }

    fn col(id: &str, name: &str, tmpl: Option<&str>) -> Column {
        Column { id: id.into(), name: Some(name.into()), template: tmpl.map(Into::into) }
    }

    #[test]
    fn topo_orders_dependencies_before_dependents() {
        // 1 col "v", 2 rows. cell(1,0) = {A1} + 1  (depends on cell(0,0)).
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![
                Row { id: "r0".into(), context: None },
                Row { id: "r1".into(), context: None },
            ],
            cells: vec![
                vec![Cell { literal: Some(10.0), ..Default::default() }],
                vec![Cell { formula: Some("{A1} + 1".into()), ..Default::default() }],
            ],
            row_source: RowSource::Authored,
            main_row_id: None,
        };
        let order = topo_order(&t).unwrap();
        let p0 = order.iter().position(|&a| a == (0, 0)).unwrap();
        let p1 = order.iter().position(|&a| a == (1, 0)).unwrap();
        assert!(p0 < p1, "dependency must come first");
    }

    #[test]
    fn cycle_is_detected() {
        // A1 = {A2}, A2 = {A1}.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![
                Row { id: "r0".into(), context: None },
                Row { id: "r1".into(), context: None },
            ],
            cells: vec![
                vec![Cell { formula: Some("{A2}".into()), ..Default::default() }],
                vec![Cell { formula: Some("{A1}".into()), ..Default::default() }],
            ],
            row_source: RowSource::Authored,
            main_row_id: None,
        };
        assert!(matches!(topo_order(&t), Err(ref c) if !c.is_empty()));
    }

    #[test]
    fn evaluate_table_aggregates_per_row_window_and_cross_cell() {
        // 10 Hz "Fork" ramp 0..9 (1 s). Row 0 = window [0,1]; metric col "fmax"
        // template max([Fork]); delta col template {fmax} - min({fmax[]}).
        let meta = SessionMetaInput {
            session_id: "s".into(),
            device_id: Some("d".into()),
            timestamp_utc_ms: 0,
            config_checksum: None,
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "Fork".into(),
                sample_rate_hz: 10.0,
                samples: (0..10).map(|i| i as f64).collect(),
                t_us: (0..10).map(|i| i * 100_000).collect(),
                source_kind: "fork".into(),
            }],
        );
        let t = TableModel {
            columns: vec![
                Column { id: "c0".into(), name: Some("fmax".into()), template: Some("max([Fork])".into()) },
                Column {
                    id: "c1".into(),
                    name: Some("delta".into()),
                    template: Some("{fmax} - min({fmax[]})".into()),
                },
            ],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell::default(), Cell::default()]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };
        let res = evaluate_table(&h, &t, &[windowed(0.0, 1.0)]);
        assert_eq!(res[0][0].value, Some(9.0)); // max over [0..9]
        assert_eq!(res[0][1].value, Some(0.0)); // 9 - min(column {9}) = 0
    }

    #[test]
    fn evaluate_table_marks_cycle_cells() {
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![
                Row { id: "r0".into(), context: None },
                Row { id: "r1".into(), context: None },
            ],
            cells: vec![
                vec![Cell { formula: Some("{A2}".into()), ..Default::default() }],
                vec![Cell { formula: Some("{A1}".into()), ..Default::default() }],
            ],
            row_source: RowSource::Authored,
            main_row_id: None,
        };
        // No channels needed — cycle short-circuits before evaluation.
        let meta = SessionMetaInput {
            session_id: "s".into(),
            device_id: Some("d".into()),
            timestamp_utc_ms: 0,
            config_checksum: None,
        };
        let h = SessionHandle::from_channels(meta, vec![]);
        let res = evaluate_table(&h, &t, &[RowBinding::default(), RowBinding::default()]);
        assert!(res.iter().flatten().any(|c| c.error.as_deref() == Some("Circular reference")));
    }

    #[test]
    fn evaluate_table_multi_resolves_each_row_against_its_own_handle() {
        // Arrange — two sessions, each a single 10 Hz "Fork" channel. Session A
        // ramps 0..9 (max 9); session B is constant 4 (max 4). One table, two rows
        // (row0→A, row1→B), columns: fmax = max([Fork]); delta = {fmax} - main({fmax[]}).
        fn handle(id: &str, samples: Vec<f64>) -> SessionHandle {
            let meta = SessionMetaInput {
                session_id: id.into(),
                device_id: Some("d".into()),
                timestamp_utc_ms: 0,
                config_checksum: None,
            };
            let t_us = (0..samples.len() as i64).map(|i| i * 100_000).collect();
            SessionHandle::from_channels(
                meta,
                vec![ChannelInput {
                    channel_id: "Fork".into(),
                    sample_rate_hz: 10.0,
                    samples,
                    t_us,
                    source_kind: "fork".into(),
                }],
            )
        }
        let a = handle("A", (0..10).map(|i| i as f64).collect());
        let b = handle("B", vec![4.0; 10]);
        let t = TableModel {
            columns: vec![
                Column { id: "c0".into(), name: Some("fmax".into()), template: Some("max([Fork])".into()) },
                Column {
                    id: "c1".into(),
                    name: Some("delta".into()),
                    template: Some("{fmax} - main({fmax[]})".into()),
                },
            ],
            rows: vec![
                Row { id: "r0".into(), context: None },
                Row { id: "r1".into(), context: None },
            ],
            cells: vec![
                vec![Cell::default(), Cell::default()],
                vec![Cell::default(), Cell::default()],
            ],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act — full-channel windows; Main row = 0 (session A, fmax 9).
        let res = evaluate_table_multi(&[&a, &b], &[0, 1], &t, &[RowBinding::default(), RowBinding::default()], Some(0));

        // Assert — fmax: A=9, B=4. delta vs Main(A=9): A→0, B→-5.
        assert_eq!(res[0][0].value, Some(9.0));
        assert_eq!(res[1][0].value, Some(4.0));
        assert_eq!(res[0][1].value, Some(0.0));
        assert_eq!(res[1][1].value, Some(-5.0));
    }

    #[test]
    fn evaluate_table_delegates_to_multi_unchanged() {
        // Arrange — single-handle path still works through the delegate.
        let meta = SessionMetaInput {
            session_id: "s".into(),
            device_id: Some("d".into()),
            timestamp_utc_ms: 0,
            config_checksum: None,
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "Fork".into(),
                sample_rate_hz: 10.0,
                samples: (0..10).map(|i| i as f64).collect(),
                t_us: (0..10).map(|i| i * 100_000).collect(),
                source_kind: "fork".into(),
            }],
        );
        let t = TableModel {
            columns: vec![Column {
                id: "c0".into(),
                name: Some("fmax".into()),
                template: Some("max([Fork])".into()),
            }],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![vec![Cell::default()]],
            row_source: RowSource::Authored,
            main_row_id: None,
        };

        // Act
        let res = evaluate_table(&h, &t, &[windowed(0.0, 1.0)]);

        // Assert
        assert_eq!(res[0][0].value, Some(9.0));
    }

    // ---- C2 §4 row derivation (ruling R217 item 2, R233) ----

    fn span(lap_number: u32, start: f64, end: f64, lap_time: f64) -> LapSpan {
        LapSpan { lap_number, start_secs: start, end_secs: end, lap_time_secs: lap_time, sectors: vec![] }
    }

    /// A one-column table whose column has a template, and one authored row
    /// that derivation must ignore.
    fn derived_table(main_row_id: Option<&str>) -> TableModel {
        TableModel {
            columns: vec![Column { id: "c0".into(), name: Some("best".into()), template: Some("lap_time()".into()) }],
            rows: vec![Row { id: "authored".into(), context: None }],
            cells: vec![vec![Cell::default()]],
            row_source: RowSource::WindowLaps,
            main_row_id: main_row_id.map(str::to_string),
        }
    }

    #[test]
    fn plan_rows_window_laps_is_one_row_per_lap_in_window_order_then_lap_order() {
        // Arrange — two windows over two sessions, three laps between them.
        let t = derived_table(None);
        let windows = vec![
            WindowLaps { session_id: "s1".into(), laps: vec![span(2, 0.0, 90.0, 90.0), span(3, 90.0, 175.0, 85.0)] },
            WindowLaps { session_id: "s2".into(), laps: vec![span(1, 0.0, 88.0, 88.0)] },
        ];

        // Act
        let (planned, bindings) = plan_rows(&t, &[], &windows);

        // Assert — the authored row is gone, not merged or appended.
        let ids: Vec<&str> = planned.rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["s1#2", "s1#3", "s2#1"]);
        assert_eq!(
            planned.rows[2].context,
            Some(RowContext { session_id: "s2".into(), lap_number: 1 })
        );
        assert_eq!(bindings[1].window, Some((90.0, 175.0)));
        assert_eq!(bindings.len(), 3);
    }

    #[test]
    fn plan_rows_window_laps_gives_every_derived_cell_the_columns_template() {
        // Arrange
        let t = derived_table(None);
        let windows = vec![WindowLaps { session_id: "s1".into(), laps: vec![span(1, 0.0, 90.0, 90.0)] }];

        // Act
        let (planned, _) = plan_rows(&t, &[], &windows);

        // Assert — a derived row carries no authored cells, so every cell
        // falls through to the column's template.
        assert_eq!(planned.cells, vec![vec![Cell::default()]]);
        assert_eq!(effective_formula(&planned, 0, 0), Some("lap_time()"));
    }

    #[test]
    fn resolve_baseline_row_fastest_is_the_smallest_recorded_lap_time_among_derived_rows() {
        // Arrange — lap 3 is the quickest, and it is not the first row.
        let t = derived_table(Some(MAIN_ROW_FASTEST));
        let windows = vec![WindowLaps {
            session_id: "s1".into(),
            laps: vec![span(1, 0.0, 90.0, 90.0), span(3, 90.0, 175.0, 85.0), span(4, 175.0, 268.0, 93.0)],
        }];
        let (planned, bindings) = plan_rows(&t, &[], &windows);

        // Act
        let baseline = resolve_baseline_row(&planned, &bindings);

        // Assert
        assert_eq!(baseline, Some(1));
    }

    #[test]
    fn resolve_baseline_row_fastest_under_authored_rows_resolves_to_nothing() {
        // Arrange — the reserved id is meaningless here; validate() reports it.
        let t = TableModel { row_source: RowSource::Authored, ..derived_table(Some(MAIN_ROW_FASTEST)) };
        let (planned, bindings) = plan_rows(&t, &[], &[]);

        // Act / Assert — never a guess at which authored row was meant.
        assert_eq!(resolve_baseline_row(&planned, &bindings), None);
    }

    #[test]
    fn resolve_baseline_row_an_ordinary_id_names_the_row_that_carries_it() {
        // Arrange
        let t = TableModel {
            row_source: RowSource::Authored,
            main_row_id: Some("r1".into()),
            rows: vec![Row { id: "r0".into(), context: None }, Row { id: "r1".into(), context: None }],
            cells: vec![vec![Cell::default()], vec![Cell::default()]],
            ..derived_table(None)
        };

        // Act / Assert
        assert_eq!(resolve_baseline_row(&t, &[RowBinding::default(); 0]), Some(1));
    }

    #[test]
    fn validate_flags_the_reserved_fastest_id_under_authored_rows() {
        // Arrange
        let t = TableModel {
            row_source: RowSource::Authored,
            rows: vec![],
            cells: vec![],
            ..derived_table(Some(MAIN_ROW_FASTEST))
        };

        // Act
        let problems = validate(&t);

        // Assert
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].kind, "invalid_main_row");
    }

    #[test]
    fn validate_accepts_the_reserved_fastest_id_under_window_laps() {
        // Arrange — the same id, on the row source it is reserved for.
        let t = TableModel { rows: vec![], cells: vec![], ..derived_table(Some(MAIN_ROW_FASTEST)) };

        // Act
        let problems = validate(&t);

        // Assert
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn a_derived_rows_lap_time_cell_is_that_laps_own_time_as_a_scalar() {
        // Arrange — the end-to-end shape: derived rows, a `lap_time()`
        // template, evaluated through the table path.
        let h = SessionHandle::from_channels(
            crate::session::handle::SessionMetaInput {
                session_id: "s1".into(),
                device_id: None,
                timestamp_utc_ms: 0,
                config_checksum: None,
            },
            vec![],
        );
        let t = derived_table(None);
        let windows = vec![WindowLaps {
            session_id: "s1".into(),
            laps: vec![span(1, 0.0, 90.0, 86.0), span(2, 90.0, 178.5, 88.5)],
        }];
        let (planned, bindings) = plan_rows(&t, &[], &windows);

        // Act
        let res = evaluate_table(&h, &planned, &bindings);

        // Assert — one value per row, each its own lap's recorded time.
        assert_eq!(res[0][0].value, Some(86.0));
        assert_eq!(res[1][0].value, Some(88.5));
    }
}
