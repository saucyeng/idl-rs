//! Table evaluation: cell addressing, dependency ordering, and per-cell
//! evaluation. Reuses the `idl-rs` math evaluator — there is no second
//! expression engine. Each cell is evaluated with [`crate::math::evaluate_scalar`]
//! against a [`CellLookup`] that resolves `{cell}` references to already-computed
//! values and `[Channel]` references to the row's lap-windowed samples.

use std::collections::HashMap;

use crate::laps::model::Lap;
use crate::math::parse::parse;
use crate::math::{evaluate_scalar, ChannelLookup, LookupChannel, MathLapContext};
use crate::session::handle::SessionHandle;
use crate::table::model::{CellResult, TableModel, TableProblem};

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
        // a cell must reduce to a scalar so the rate is never surfaced.
        Some(LookupChannel { samples: std::sync::Arc::from(samples), sample_rate_hz: 0.0 })
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

/// Evaluate every cell. `row_windows[r]` is row `r`'s `[t0, t1]` (or `None` for
/// the full channel). Single-handle convenience over [`evaluate_table_multi`].
pub fn evaluate_table(
    handle: &SessionHandle,
    table: &TableModel,
    row_windows: &[Option<(f64, f64)>],
) -> Vec<Vec<CellResult>> {
    let row_handles = vec![0usize; table.rows.len()];
    evaluate_table_multi(&[handle], &row_handles, table, row_windows, None)
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
    row_windows: &[Option<(f64, f64)>],
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
    let lap_ctx = MathLapContext { baseline_row, ..MathLapContext::empty() };
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
        let lookup = CellLookup {
            handle,
            window: row_windows.get(r).copied().flatten(),
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

/// Map each row's lap binding to its recording-time `(t0, t1)` window. Returns
/// `None` for a row with no `context` or whose `lap_index` is past the end of
/// `laps`. The result is the `row_windows` argument of [`evaluate_table`].
pub fn lap_windows(table: &TableModel, laps: &[Lap]) -> Vec<Option<(f64, f64)>> {
    table
        .rows
        .iter()
        .map(|row| {
            row.context.as_ref().and_then(|ctx| {
                laps.get(ctx.lap_index as usize).map(|l| (l.start_time_secs, l.end_time_secs))
            })
        })
        .collect()
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

    // 3. Cycle detection over the whole grid.
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
    fn lap_windows_maps_index_and_none_for_unbound_or_oob() {
        // Arrange — 2 laps; rows bound to lap 1, unbound, and out-of-range lap 9.
        let laps = vec![lap(0, 10.0, 20.0), lap(1, 20.0, 35.0)];
        let t = TableModel {
            columns: vec![],
            rows: vec![
                Row { id: "r0".into(), context: Some(RowContext { session_id: "s".into(), lap_index: 1 }) },
                Row { id: "r1".into(), context: None },
                Row { id: "r2".into(), context: Some(RowContext { session_id: "s".into(), lap_index: 9 }) },
            ],
            cells: vec![vec![], vec![], vec![]],
        };

        // Act
        let w = lap_windows(&t, &laps);

        // Assert
        assert_eq!(w[0], Some((20.0, 35.0)));
        assert_eq!(w[1], None);
        assert_eq!(w[2], None);
    }

    #[test]
    fn validate_flags_dimension_mismatch() {
        // Arrange — 1 row declared but 0 cell-rows.
        let t = TableModel {
            columns: vec![col("c", "v", None)],
            rows: vec![Row { id: "r0".into(), context: None }],
            cells: vec![],
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
        };
        assert!(matches!(topo_order(&t), Err(ref c) if !c.is_empty()));
    }

    #[test]
    fn evaluate_table_aggregates_per_row_window_and_cross_cell() {
        // 10 Hz "Fork" ramp 0..9 (1 s). Row 0 = window [0,1]; metric col "fmax"
        // template max([Fork]); delta col template {fmax} - min({fmax[]}).
        let meta = SessionMetaInput {
            session_id: "s".into(),
            device_id: "d".into(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "Fork".into(),
                sample_rate_hz: 10.0,
                samples: (0..10).map(|i| i as f64).collect(),
                sample_times_secs: None,
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
        };
        let res = evaluate_table(&h, &t, &[Some((0.0, 1.0))]);
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
        };
        // No channels needed — cycle short-circuits before evaluation.
        let meta = SessionMetaInput {
            session_id: "s".into(),
            device_id: "d".into(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![]);
        let res = evaluate_table(&h, &t, &[None, None]);
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
                device_id: "d".into(),
                timestamp_utc_ms: 0,
                config_checksum: String::new(),
            };
            SessionHandle::from_channels(
                meta,
                vec![ChannelInput {
                    channel_id: "Fork".into(),
                    sample_rate_hz: 10.0,
                    samples,
                    sample_times_secs: None,
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
        };

        // Act — full-channel windows; Main row = 0 (session A, fmax 9).
        let res = evaluate_table_multi(&[&a, &b], &[0, 1], &t, &[None, None], Some(0));

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
            device_id: "d".into(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "Fork".into(),
                sample_rate_hz: 10.0,
                samples: (0..10).map(|i| i as f64).collect(),
                sample_times_secs: None,
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
        };

        // Act
        let res = evaluate_table(&h, &t, &[Some((0.0, 1.0))]);

        // Assert
        assert_eq!(res[0][0].value, Some(9.0));
    }
}
