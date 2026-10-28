//! Session-set evaluation (session-sets spec §4, phase 1): **map, then
//! combine**.
//!
//! - [`plan_set`] computes, statically from the parsed document, which
//!   definitions a set needs: the closure of every `rowSource: "sessions"`
//!   table's column templates (the *demand*), each template compiled as an
//!   anonymous session-level definition named `table:<cell id>:<column id>`.
//! - [`evaluate_member`] evaluates that demand in **one** member session and
//!   keeps only reduced results (rank 0 and `[lap]`). Every full-rate buffer
//!   is dropped before it returns, so nothing larger than a `[lap]` vector
//!   ever leaves a member.
//! - [`set_tables`] combines the members' kept results into the set tables,
//!   in set order. It reads no session data.
//!
//! The member key ([`member_key`]) addresses one member's results in the
//! on-disk cache (`store::eval_cache`); the runner that walks the members is
//! `store::set_job`.

use std::collections::{BTreeMap, HashMap, HashSet};

use sha2::{Digest, Sha256};

use crate::math::eval::{ChannelLookup, LookupChannel, MathLapContext};
use crate::math::units::UnitLabel;
use crate::math::{channel_refs, MathEvalError, ValueAxis};
use crate::table::{evaluate_table_rows, Addr, CellResult, Row, RowBinding, RowContext, RowSource, TableModel};

use super::eval::{EvalCancelled, EvalObserver};
use super::plan::{Demand, EvalPlan, PlanInputs};
use super::resolve::{resolve_planned, resolve_workbook_units, ResolvedValue};
use super::session_set::SessionSet;
use super::{CellKindToken, MathCellDef, WorkbookDoc};

/// Prefix of an anonymous template definition's name. A `:` can never
/// appear in a definition identifier (C2 §3.1), so these names never collide
/// with an authored one and nothing can reference them.
pub const TEMPLATE_DEF_PREFIX: &str = "table:";

/// Which definitions a plan demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanScope {
    /// The closure of the session-row tables' templates only (the app's
    /// `eval_session_set`, spec §4.2).
    Tables,
    /// Every definition as well (the CLI's `workbook eval --values`, gap N7).
    AllDefinitions,
}

/// One session-row table column whose template is evaluated inside each
/// member, as the anonymous definition [`Self::def_name`].
#[derive(Debug, Clone, PartialEq)]
pub struct MemberTemplate {
    /// The table cell's id.
    pub cell_id: String,
    /// The column's index in the table.
    pub column: usize,
    /// The anonymous definition's name, `table:<cell id>:<column id>`.
    pub def_name: String,
}

/// What a session set needs from each member (spec §4.2), computed once per
/// document.
#[derive(Debug, Clone, PartialEq)]
pub struct SetPlan {
    /// The definitions evaluated in each member: the demanded workbook
    /// definitions' closure, in document order, then one anonymous definition
    /// per [`MemberTemplate`].
    pub defs: Vec<MathCellDef>,
    /// The templates evaluated inside members.
    pub templates: Vec<MemberTemplate>,
    /// Every name a member's results must hold for this plan: every entry of
    /// [`Self::defs`], once each (the anonymous templates, the workbook
    /// definitions `{…}` templates read at combine time, their closure, and
    /// under [`PlanScope::AllDefinitions`] every definition). A cache hit must
    /// hold all of them, so a run never serves a narrower earlier run's
    /// subset.
    pub needed: Vec<String>,
    /// Whether any demanded definition reads a stored channel. `false` means
    /// a member can be evaluated from `session.json` alone (lap and sector
    /// statistics).
    pub reads_channels: bool,
    /// The phase-1 key text (spec §4.3): every definition and constant of the
    /// document in canonical order, plus every session-row column template.
    /// Prose, js cells and `graph` are not in it.
    pub maths_text: String,
}

/// True when `template` reads other cells (`{…}`), which can only be
/// evaluated once every member's row exists.
fn reads_cells(template: &str) -> bool {
    template.contains('{')
}

/// Builds the [`SetPlan`] of `doc` (spec §4.2): the anonymous template
/// definitions of every `rowSource: "sessions"` table, the workbook
/// definitions their `[name]`s reach (transitively, through the same
/// `[name]` edges the resolver follows; a name defined twice resolves to its
/// first definition, R49), and — under [`PlanScope::AllDefinitions`] — every
/// definition.
pub fn plan_set(doc: &WorkbookDoc, scope: PlanScope) -> SetPlan {
    let mut first_def: HashMap<&str, &MathCellDef> = HashMap::new();
    for d in &doc.defs {
        first_def.entry(d.name.as_str()).or_insert(d);
    }

    let mut templates = Vec::new();
    let mut template_defs = Vec::new();
    let mut roots: Vec<String> = Vec::new();
    let mut template_text = Vec::new();
    for cell in doc.cells.iter().filter(|c| c.kind_token == CellKindToken::Table) {
        let Some(table) = &cell.table else { continue };
        if table.row_source != RowSource::Sessions {
            continue;
        }
        for (ci, col) in table.columns.iter().enumerate() {
            let Some(template) = col.template.as_deref() else { continue };
            template_text.push(format!("{}:{}={}", cell.id, col.id, template));
            let refs = channel_refs(template);
            if reads_cells(template) {
            } else {
                let def_name = format!("{TEMPLATE_DEF_PREFIX}{}:{}", cell.id, col.id);
                template_defs.push(MathCellDef {
                    cell_id: cell.id.clone(),
                    name: def_name.clone(),
                    expr_text: template.to_string(),
                    label: col.name.clone(),
                    unit_annotation: None,
                    order: ci,
                });
                templates.push(MemberTemplate { cell_id: cell.id.clone(), column: ci, def_name });
            }
            roots.extend(refs);
        }
    }

    // Closure over `[name]` edges between definitions.
    let mut demanded: HashSet<String> = HashSet::new();
    if scope == PlanScope::AllDefinitions {
        demanded.extend(doc.defs.iter().map(|d| d.name.clone()));
    }
    let mut stack: Vec<String> = roots.into_iter().chain(demanded.iter().cloned()).collect();
    while let Some(name) = stack.pop() {
        let Some(def) = first_def.get(name.as_str()) else { continue };
        demanded.insert(name.clone());
        for r in channel_refs(&def.expr_text) {
            if first_def.contains_key(r.as_str()) && !demanded.contains(&r) {
                stack.push(r);
            }
        }
    }

    let mut defs: Vec<MathCellDef> = Vec::new();
    let mut taken = HashSet::new();
    for d in &doc.defs {
        if demanded.contains(&d.name) && taken.insert(d.name.clone()) {
            defs.push(d.clone());
        }
    }
    let reads_channels = defs
        .iter()
        .chain(template_defs.iter())
        .flat_map(|d| channel_refs(&d.expr_text))
        .chain(
            doc.cells
                .iter()
                .filter_map(|c| c.table.as_ref())
                .filter(|t| t.row_source == RowSource::Sessions)
                .flat_map(|t| t.columns.iter().filter_map(|c| c.template.as_deref()).filter(|t| reads_cells(t)))
                .flat_map(channel_refs),
        )
        .any(|r| !first_def.contains_key(r.as_str()));
    defs.extend(template_defs);
    let mut needed: Vec<String> = Vec::new();
    for d in &defs {
        if !needed.contains(&d.name) {
            needed.push(d.name.clone());
        }
    }

    SetPlan { defs, templates, needed, reads_channels, maths_text: maths_text(doc, &template_text) }
}

/// The canonical maths text of `doc` (spec §4.3 phase-1 key): definition
/// lines sorted by name (`name = expr # unit: u`), constants sorted by name
/// with their exact bits, then the session-row templates sorted.
fn maths_text(doc: &WorkbookDoc, templates: &[String]) -> String {
    // Sorted by name, but a repeated name keeps its document-order position
    // among its own duplicates (`name#k`): the first definition of a name is
    // the one every reference reads (R49), so swapping two duplicates changes
    // results and must change the key.
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let mut defs: Vec<String> = doc
        .defs
        .iter()
        .map(|d| {
            let k = seen.entry(d.name.as_str()).or_insert(0);
            *k += 1;
            format!("{}#{}={}#{}", d.name, *k - 1, d.expr_text, d.unit_annotation.as_deref().unwrap_or(""))
        })
        .collect();
    defs.sort();
    let mut constants: Vec<String> =
        doc.constants.iter().map(|(k, v)| format!("{k}={:016x}", v.to_bits())).collect();
    constants.sort();
    let mut templates = templates.to_vec();
    templates.sort();
    let mut out = String::new();
    for section in [defs, constants, templates] {
        for line in section {
            out.push_str(&line);
            out.push('\n');
        }
        out.push('\u{1e}');
    }
    out
}

/// The phase-1 member key (spec §4.3): SHA-256 over the maths text, the
/// member's data identity (`blob_sha256|importer_version|seam_correction_version`),
/// its laps identity ([`laps_identity`]) and the engine identity (the build's
/// git commit, Q4). Each part is length-prefixed so no two different inputs
/// concatenate to the same bytes.
pub fn member_key(plan: &SetPlan, data_identity: &str, laps_identity: &str, engine_identity: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in ["idl1-member-v1", plan.maths_text.as_str(), data_identity, laps_identity, engine_identity] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    h.finalize().into()
}

/// A member's laps identity (spec §4.3): its `session.json` `laps[]` (bounds,
/// times and sectors) and `lap_detector_version`, as canonical JSON. Every
/// other `session.json` field (comments, rider, tags) is left out, so editing
/// one recomputes nothing. An absent `session.json` is the empty string.
pub fn laps_identity(doc: Option<&crate::store::session_json::SessionJson>) -> String {
    match doc {
        None => String::new(),
        Some(d) => serde_json::json!({ "laps": d.laps, "lap_detector_version": d.lap_detector_version }).to_string(),
    }
}

/// The key of one result row in the cache: SHA-256 of the member key and the
/// result's name.
pub fn result_key(member_key: &[u8; 32], name: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(member_key);
    h.update(name.as_bytes());
    h.finalize().into()
}

/// One member-level result as the set keeps it.
#[derive(Debug, Clone, PartialEq)]
pub enum MemberValue {
    /// A rank-0 value.
    Scalar(f64),
    /// A `[lap]` value: `laps[i]` is the 1-based lap number of `values[i]`.
    Laps {
        /// Lap numbers.
        laps: Vec<u32>,
        /// One value per lap.
        values: Vec<f64>,
    },
    /// A full-rate (`[t]`) value. Its samples were dropped at the end of the
    /// member (spec §4.1); only the fact that it was full-rate is kept.
    FullRate,
    /// The definition failed in this member (a deterministic `math_*`
    /// error).
    Error(MathEvalError),
}

/// One kept result: its value and its inferred (or declared) unit.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberResult {
    /// The value.
    pub value: MemberValue,
    /// The unit, C3 `UnitLabel`.
    pub unit: UnitLabel,
}

/// Everything one member keeps, by definition name (anonymous template
/// definitions included).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemberResults(pub BTreeMap<String, MemberResult>);

/// The evaluation plan of one member (evaluation-memory spec §4.2.B): the
/// set plan's definitions scheduled against that member's stored channels
/// (`inputs`, from its `data.parquet` index), keeping **no** full-rate value
/// beyond its last reader ([`Demand::reductions`]). Its estimate is what the
/// member is admitted against.
pub fn member_eval_plan(plan: &SetPlan, constants: &HashMap<String, f64>, inputs: &PlanInputs) -> EvalPlan {
    EvalPlan::new(&plan.defs, constants, inputs, &Demand::reductions(plan.defs.iter().map(|d| d.name.clone())))
}

/// Evaluates `plan`'s definitions in one member session (`lookup` is that
/// session's channels, `lap_ctx` its whole-session lap context) following
/// `eval_plan` ([`member_eval_plan`]), and keeps only reduced results (spec
/// §4.1): every full-rate buffer is released after its last reader, so the
/// member's peak is the planner's, not the sum of its values. Definitions
/// resolve exactly as in a math cell (definitions first, then channels —
/// ruling Q6).
///
/// # Errors
/// [`EvalCancelled`] when `observer` cancels between two definitions.
pub fn evaluate_member_planned(
    plan: &SetPlan,
    constants: &HashMap<String, f64>,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
    eval_plan: &EvalPlan,
    observer: Option<&dyn EvalObserver>,
) -> Result<MemberResults, EvalCancelled> {
    let mut values = resolve_planned(&plan.defs, constants, lookup, lap_ctx, eval_plan, observer)?;
    let mut units = resolve_workbook_units(&plan.defs, constants, lookup);
    let mut out = MemberResults::default();
    for def in &plan.defs {
        let key = (def.cell_id.clone(), def.name.clone());
        let unit = units.remove(&key).map(|(u, _)| UnitLabel::from(&u)).unwrap_or(UnitLabel::Dimensionless);
        let value = match values.remove(&key) {
            Some(Ok(v)) => reduce(&v),
            Some(Err(e)) => MemberValue::Error(e),
            None => continue,
        };
        out.0.insert(def.name.clone(), MemberResult { value, unit });
    }
    Ok(out)
}

/// [`evaluate_member_planned`] with a plan that knows no channel sizes and no
/// observer — for a member with no data and for tests.
pub fn evaluate_member(
    plan: &SetPlan,
    constants: &HashMap<String, f64>,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> MemberResults {
    let eval_plan = member_eval_plan(plan, constants, &PlanInputs::default());
    evaluate_member_planned(plan, constants, lookup, lap_ctx, &eval_plan, None).unwrap_or_default()
}

/// The kept form of one evaluated value: rank 0 is `evaluate_scalar`'s rule
/// (one sample, rate 0, on no lap axis); a released or longer time-axis
/// value is [`MemberValue::FullRate`].
fn reduce(v: &ResolvedValue) -> MemberValue {
    let out = &v.out;
    match out.axis {
        ValueAxis::Lap => MemberValue::Laps {
            laps: out.t_us.iter().map(|&n| n.max(0) as u32).collect(),
            values: out.samples.to_vec(),
        },
        ValueAxis::Time if !v.released && out.samples.len() == 1 && out.sample_rate_hz == 0.0 => {
            MemberValue::Scalar(out.samples[0])
        }
        ValueAxis::Time => MemberValue::FullRate,
    }
}

/// Why a member has no results, as the combine step reports it in that
/// member's rows.
#[derive(Debug, Clone, PartialEq)]
pub enum MemberAbsence {
    /// Its data is not on this device and nothing is cached (Q3: rows show
    /// as missing; set-level reductions run over the members present).
    Missing,
    /// No such session in the library at all.
    NotFound,
    /// Evaluation could not run (a decode refused or failed); the message.
    Failed(String),
}

/// One member as the combine step sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct CombineMember<'a> {
    /// Session id.
    pub session_id: &'a str,
    /// Display label.
    pub label: &'a str,
    /// Index into the set's groups.
    pub group: usize,
    /// Its kept results, or why there are none.
    pub results: Result<&'a MemberResults, MemberAbsence>,
}

/// One evaluated session-row table.
#[derive(Debug, Clone, PartialEq)]
pub struct SetTable {
    /// The table cell's id.
    pub cell_id: String,
    /// The planned model: one row per member (filtered by `rowGroup`), row id
    /// = session id, context `{ sessionId }`.
    pub model: TableModel,
    /// One result per planned cell.
    pub results: Vec<Vec<CellResult>>,
    /// For each planned row, the member's index in set order.
    pub row_members: Vec<usize>,
}

/// A non-fatal note about a set evaluation (Q3 of 2026-10-28): a value that
/// reduces over the set was computed over the members present on this
/// device only. `missing` names each member that was left out, so the app
/// can offer to fetch it.
#[derive(Debug, Clone, PartialEq)]
pub struct SetWarning {
    /// The table cell the warning is about; `None` for the whole set.
    pub cell_id: Option<String>,
    /// Session ids of the members left out, in set order.
    pub missing: Vec<String>,
    /// Display-ready English.
    pub message: String,
}

/// A member's kept results as a `[name]` lookup, for the templates that are
/// evaluated at combine time (`{…}` templates): a rank-0 result reads as a
/// scalar, a `[lap]` result as its per-lap values; full-rate and failed
/// results do not resolve.
struct MemberLookup<'a> {
    results: &'a MemberResults,
}

impl ChannelLookup for MemberLookup<'_> {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        match &self.results.0.get(name)?.value {
            MemberValue::Scalar(v) => Some(LookupChannel {
                samples: std::sync::Arc::from(vec![*v]),
                sample_rate_hz: 0.0,
                t_us: std::sync::Arc::from(&[] as &[i64]),
            }),
            MemberValue::Laps { laps, values } => Some(LookupChannel {
                samples: std::sync::Arc::from(values.clone()),
                sample_rate_hz: 0.0,
                t_us: std::sync::Arc::from(laps.iter().map(|&n| n as i64).collect::<Vec<_>>()),
            }),
            MemberValue::FullRate | MemberValue::Error(_) => None,
        }
    }
}

/// The cell result a member-evaluated template has in `member`'s row.
fn template_cell(member: &CombineMember, def_name: &str) -> CellResult {
    let fail = |message: String| CellResult { value: None, error: Some(message) };
    match &member.results {
        Err(MemberAbsence::Missing) => fail(format!("{} is not on this device", member.label)),
        Err(MemberAbsence::NotFound) => fail(format!("{}: no such session in the library", member.label)),
        Err(MemberAbsence::Failed(m)) => fail(format!("{} could not be evaluated: {m}", member.label)),
        Ok(results) => match results.0.get(def_name).map(|r| &r.value) {
            Some(MemberValue::Scalar(v)) => CellResult { value: Some(*v), error: None },
            Some(MemberValue::Laps { .. }) => fail("reduce to a single number: this is a [lap] value".into()),
            Some(MemberValue::FullRate) => fail("reduce to a single number: this is a [t] value".into()),
            Some(MemberValue::Error(e)) => fail(e.message.clone()),
            None => fail("not evaluated".into()),
        },
    }
}

/// Combines the members' kept results into every `rowSource: "sessions"`
/// table of `doc`, in document order, plus the Q3 warnings (spec §4.1's
/// "combine"). `members` is every member of `set`, in set order.
///
/// A row whose member has no results shows that absence in every cell. A
/// `{col[]}` reference reads the rows that have a value, so a column
/// aggregate over a table with a missing member is computed over the members
/// present and the table carries a [`SetWarning`] naming the missing ones.
/// A `rowGroup` naming no group of the set plans no rows.
pub fn set_tables(
    doc: &WorkbookDoc,
    set: &SessionSet,
    plan: &SetPlan,
    members: &[CombineMember],
) -> (Vec<SetTable>, Vec<SetWarning>) {
    let mut tables = Vec::new();
    let mut warnings = Vec::new();
    for cell in doc.cells.iter().filter(|c| c.kind_token == CellKindToken::Table) {
        let Some(table) = &cell.table else { continue };
        if table.row_source != RowSource::Sessions {
            continue;
        }
        let group = table.row_group.as_deref().map(|name| set.group_index(name));
        let row_members: Vec<usize> = members
            .iter()
            .enumerate()
            .filter(|(_, m)| match group {
                None => true,
                Some(Some(g)) => m.group == g,
                Some(None) => false,
            })
            .map(|(i, _)| i)
            .collect();
        let rows: Vec<Row> = row_members
            .iter()
            .map(|&i| Row {
                id: members[i].session_id.to_string(),
                context: Some(RowContext { session_id: members[i].session_id.to_string(), lap_number: None }),
            })
            .collect();
        let cols = table.columns.len();
        let model = TableModel {
            rows,
            cells: vec![vec![Default::default(); cols]; row_members.len()],
            ..table.clone()
        };

        let mut preset: HashMap<Addr, CellResult> = HashMap::new();
        for (r, &i) in row_members.iter().enumerate() {
            let member = &members[i];
            for t in plan.templates.iter().filter(|t| t.cell_id == cell.id) {
                preset.insert((r, t.column), template_cell(member, &t.def_name));
            }
            if member.results.is_err() {
                for c in 0..cols {
                    preset.entry((r, c)).or_insert_with(|| template_cell(member, ""));
                }
            }
        }
        let lookups: Vec<Option<MemberLookup>> = row_members
            .iter()
            .map(|&i| members[i].results.as_ref().ok().map(|results| MemberLookup { results }))
            .collect();
        let row_lookups: Vec<&dyn ChannelLookup> = lookups
            .iter()
            .map(|l| match l {
                Some(l) => l as &dyn ChannelLookup,
                None => &EMPTY_LOOKUP as &dyn ChannelLookup,
            })
            .collect();
        let bindings = vec![RowBinding::default(); row_members.len()];
        let results = evaluate_table_rows(&row_lookups, &model, &bindings, None, &preset);

        // Q3: a `{col[]}` total runs over the rows that have a value. Every
        // row it leaves out is named, with its reason — not only members
        // missing from this device.
        let gathered: Vec<usize> = table
            .columns
            .iter()
            .filter_map(|c| c.template.as_deref())
            .flat_map(gathered_columns)
            .filter_map(|name| table.columns.iter().position(|c| c.name.as_deref() == Some(name.as_str())))
            .collect();
        let mut left_out: Vec<(usize, String)> = Vec::new();
        for (r, &i) in row_members.iter().enumerate() {
            let reason = gathered.iter().find_map(|&c| {
                let cell = &results[r][c];
                cell.value.is_none().then(|| cell.error.clone().unwrap_or_else(|| "no value".into()))
            });
            if let Some(reason) = reason {
                left_out.push((i, reason));
            }
        }
        if !left_out.is_empty() {
            let missing = left_out
                .iter()
                .filter(|(i, _)| matches!(members[*i].results, Err(MemberAbsence::Missing)))
                .map(|(i, _)| members[*i].session_id.to_string())
                .collect();
            let named: Vec<String> =
                left_out.iter().map(|(i, reason)| format!("{} ({reason})", members[*i].label)).collect();
            warnings.push(SetWarning {
                cell_id: Some(cell.id.clone()),
                message: format!("Column totals in this table leave out {} session(s): {}", left_out.len(), named.join(", ")),
                missing,
            });
        }
        if matches!(group, Some(None)) {
            warnings.push(SetWarning {
                cell_id: Some(cell.id.clone()),
                missing: Vec::new(),
                message: format!(
                    "rowGroup '{}' names no group of the workbook's sessions, so this table has no rows",
                    table.row_group.as_deref().unwrap_or("")
                ),
            });
        }
        tables.push(SetTable { cell_id: cell.id.clone(), model, results, row_members });
    }

    // The whole set: every member that has no results, with its reason.
    let absent: Vec<(&CombineMember, String)> = members
        .iter()
        .filter_map(|m| match &m.results {
            Ok(_) => None,
            Err(MemberAbsence::Missing) => Some((m, "not on this device".to_string())),
            Err(MemberAbsence::NotFound) => Some((m, "no such session".to_string())),
            Err(MemberAbsence::Failed(e)) => Some((m, e.clone())),
        })
        .collect();
    if !absent.is_empty() {
        warnings.push(SetWarning {
            cell_id: None,
            missing: absent
                .iter()
                .filter(|(m, _)| matches!(m.results, Err(MemberAbsence::Missing)))
                .map(|(m, _)| m.session_id.to_string())
                .collect(),
            message: format!(
                "{} of {} sessions have no results and are left out: {}",
                absent.len(),
                members.len(),
                absent.iter().map(|(m, r)| format!("{} ({r})", m.label)).collect::<Vec<_>>().join(", ")
            ),
        });
    }
    (tables, warnings)
}

/// The column names a template gathers whole (`{name[]}`).
fn gathered_columns(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(o) = rest.find('{') {
        rest = &rest[o + 1..];
        let Some(close) = rest.find('}') else { break };
        if let Some(name) = rest[..close].strip_suffix("[]") {
            out.push(name.to_string());
        }
        rest = &rest[close + 1..];
    }
    out
}

/// The lookup of a row that has nothing to resolve against.
struct EmptyLookup;

impl ChannelLookup for EmptyLookup {
    fn lookup(&self, _name: &str) -> Option<LookupChannel> {
        None
    }
}

static EMPTY_LOOKUP: EmptyLookup = EmptyLookup;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};
    use crate::workbook::v3::parse_workbook;
    use crate::workbook::v3::session_set::parse_session_set;

    const HEAD: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Set\nsessions:\n  - group: Day 1\n    members:\n      - { id: s1, label: A }\n      - { id: s2, label: B }\n  - group: Day 2\n    members:\n      - { id: s3, label: C }\n---\n";

    fn workbook(math: &str, table: &str) -> WorkbookDoc {
        let md = format!("{HEAD}\n```math id=aaaaaaaa\n{math}\n```\n\n```table id=bbbbbbbb\n{table}\n```\n");
        parse_workbook(&md).unwrap().0
    }

    fn session(id: &str, fork: Vec<f64>) -> SessionHandle {
        let t_us = (0..fork.len() as i64).map(|i| i * 1_000_000).collect();
        SessionHandle::from_channels(
            SessionMetaInput { session_id: id.into(), device_id: None, timestamp_utc_ms: 0, config_checksum: None },
            vec![ChannelInput {
                channel_id: "Fork".into(),
                sample_rate_hz: 1.0,
                samples: fork,
                t_us,
                source_kind: "test".into(),
            }],
        )
    }

    fn evaluate(doc: &WorkbookDoc, handles: &[SessionHandle]) -> Vec<MemberResults> {
        let plan = plan_set(doc, PlanScope::Tables);
        handles.iter().map(|h| evaluate_member(&plan, &doc.constants, h, &MathLapContext::empty())).collect()
    }

    fn combine<'a>(set: &'a SessionSet, results: &'a [Result<MemberResults, MemberAbsence>]) -> Vec<CombineMember<'a>> {
        set.members()
            .into_iter()
            .zip(results)
            .map(|(m, r)| CombineMember {
                session_id: set.groups[m.group].members.iter().find(|x| x.id == m.id).map(|x| x.id.as_str()).unwrap(),
                label: set.groups[m.group].members.iter().find(|x| x.id == m.id).and_then(|x| x.label.as_deref()).unwrap(),
                group: m.group,
                results: r.as_ref().map_err(|e| e.clone()),
            })
            .collect()
    }

    const TABLE: &str = r#"{"rowSource":"sessions","columns":[{"id":"c0","name":"med","template":"[med]"},{"id":"c1","name":"dbl","template":"2 * {med}"}],"rows":[],"cells":[]}"#;

    #[test]
    fn session_row_table_template_med_over_two_synthetic_sessions_each_row_is_its_own_sessions_med() {
        // Arrange
        let doc = workbook("med = median([Fork])", TABLE);
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let results: Vec<Result<MemberResults, MemberAbsence>> =
            evaluate(&doc, &[session("s1", vec![1.0, 2.0, 3.0]), session("s2", vec![10.0, 20.0, 30.0]), session("s3", vec![5.0])])
                .into_iter()
                .map(Ok)
                .collect();
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, warnings) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        let t = &tables[0];
        assert_eq!(t.model.rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["s1", "s2", "s3"]);
        assert_eq!(t.results.iter().map(|r| r[0].value).collect::<Vec<_>>(), vec![Some(2.0), Some(20.0), Some(5.0)]);
        assert_eq!(t.results[1][1].value, Some(40.0));
        assert!(warnings.is_empty());
    }

    #[test]
    fn session_row_table_row_group_names_the_second_group_only_its_members_in_set_order() {
        // Arrange
        let doc = workbook(
            "med = median([Fork])",
            r#"{"rowSource":"sessions","rowGroup":"Day 2","columns":[{"id":"c0","template":"[med]"}],"rows":[],"cells":[]}"#,
        );
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let results: Vec<Result<MemberResults, MemberAbsence>> =
            evaluate(&doc, &[session("s1", vec![1.0]), session("s2", vec![2.0]), session("s3", vec![3.0])])
                .into_iter()
                .map(Ok)
                .collect();
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, _) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        assert_eq!(tables[0].row_members, vec![2]);
        assert_eq!(tables[0].results[0][0].value, Some(3.0));
    }

    #[test]
    fn session_row_table_a_full_rate_template_is_a_reduce_to_a_single_number_cell_error() {
        // Arrange
        let doc = workbook(
            "dbl = [Fork] * 2",
            r#"{"rowSource":"sessions","columns":[{"id":"c0","template":"[dbl]"}],"rows":[],"cells":[]}"#,
        );
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let results: Vec<Result<MemberResults, MemberAbsence>> =
            evaluate(&doc, &[session("s1", vec![1.0, 2.0]), session("s2", vec![2.0]), session("s3", vec![3.0, 4.0])])
                .into_iter()
                .map(Ok)
                .collect();
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, _) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        assert!(tables[0].results[0][0].error.as_deref().unwrap().starts_with("reduce to a single number"));
    }

    #[test]
    fn session_row_table_a_missing_member_its_row_says_so_and_a_column_total_is_over_the_rest_with_a_warning() {
        // Arrange
        let doc = workbook(
            "med = median([Fork])",
            r#"{"rowSource":"sessions","columns":[{"id":"c0","name":"med","template":"[med]"},{"id":"c1","name":"tot","template":"sum({med[]})"}],"rows":[],"cells":[]}"#,
        );
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let mut evaluated = evaluate(&doc, &[session("s1", vec![1.0]), session("s3", vec![3.0])]).into_iter();
        let results = vec![Ok(evaluated.next().unwrap()), Err(MemberAbsence::Missing), Ok(evaluated.next().unwrap())];
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, warnings) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        assert_eq!(tables[0].results[1][0].error.as_deref(), Some("B is not on this device"));
        assert_eq!(tables[0].results[0][1].value, Some(4.0));
        assert_eq!(warnings[0].cell_id.as_deref(), Some("bbbbbbbb"));
        assert_eq!(warnings[0].missing, vec!["s2".to_string()]);
        assert_eq!(warnings.last().unwrap().cell_id, None);
    }

    #[test]
    fn session_row_table_a_failed_member_and_a_member_whose_definition_errored_both_named_in_the_totals_warning() {
        // Arrange — s1 fine, s2 failed (resource_exhausted), s3's `med` errors.
        let doc = workbook(
            "med = median([Fork])",
            r#"{"rowSource":"sessions","columns":[{"id":"c0","name":"med","template":"[med]"},{"id":"c1","name":"tot","template":"sum({med[]})"}],"rows":[],"cells":[]}"#,
        );
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let mut evaluated = evaluate(&doc, &[session("s1", vec![1.0])]).into_iter();
        let mut broken = MemberResults::default();
        broken.0.insert("table:bbbbbbbb:c0".into(), MemberResult {
            value: MemberValue::Error(MathEvalError::new(crate::math::MathEvalErrorKind::UnknownChannel, "no Fork")),
            unit: UnitLabel::Dimensionless,
        });
        let results = vec![
            Ok(evaluated.next().unwrap()),
            Err(MemberAbsence::Failed("needs 3.0 GB, budget 2.0 GB".into())),
            Ok(broken),
        ];
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, warnings) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        assert_eq!(tables[0].results[0][1].value, Some(1.0));
        let table_warning = &warnings[0];
        assert_eq!(table_warning.cell_id.as_deref(), Some("bbbbbbbb"));
        assert!(table_warning.message.contains("B (") && table_warning.message.contains("C (no Fork)"), "{}", table_warning.message);
        assert!(table_warning.missing.is_empty());
        assert!(warnings.last().unwrap().message.contains("needs 3.0 GB"));
    }

    #[test]
    fn session_row_table_row_group_naming_no_group_no_rows_and_a_warning() {
        // Arrange
        let doc = workbook(
            "med = median([Fork])",
            r#"{"rowSource":"sessions","rowGroup":"Day 9","columns":[{"id":"c0","template":"[med]"}],"rows":[],"cells":[]}"#,
        );
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        let results: Vec<Result<MemberResults, MemberAbsence>> =
            evaluate(&doc, &[session("s1", vec![1.0]), session("s2", vec![2.0]), session("s3", vec![3.0])])
                .into_iter()
                .map(Ok)
                .collect();
        let plan = plan_set(&doc, PlanScope::Tables);

        // Act
        let (tables, warnings) = set_tables(&doc, &set, &plan, &combine(&set, &results));

        // Assert
        assert!(tables[0].model.rows.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("rowGroup 'Day 9'")));
    }

    #[test]
    fn member_key_two_duplicate_definitions_swapped_key_changes() {
        // Act
        let a = key_of("x = 1\nx = 2", "", TABLE);
        let b = key_of("x = 2\nx = 1", "", TABLE);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn plan_needed_lists_every_planned_definition() {
        // Arrange
        let doc = workbook("dbl = [Fork] * 2\npeak = max([dbl])\nother = 3", TABLE.replace("[med]", "[peak]").as_str());

        // Act
        let plan = plan_set(&doc, PlanScope::AllDefinitions);

        // Assert
        for d in &plan.defs {
            assert!(plan.needed.contains(&d.name), "{}", d.name);
        }
        assert!(plan.needed.contains(&"other".to_string()));
    }

    #[test]
    fn member_evaluation_full_rate_definitions_only_rank_0_and_lap_results_survive_the_member() {
        // Arrange
        let doc = workbook(
            "dbl = [Fork] * 2\npeak = max([dbl])",
            r#"{"rowSource":"sessions","columns":[{"id":"c0","template":"[peak]"}],"rows":[],"cells":[]}"#,
        );

        // Act
        let results = evaluate(&doc, &[session("s1", vec![1.0, 2.0, 3.0])]).remove(0);

        // Assert — the [t] value is a marker, not samples.
        assert_eq!(results.0["dbl"].value, MemberValue::FullRate);
        assert_eq!(results.0["peak"].value, MemberValue::Scalar(6.0));
        assert_eq!(results.0["table:bbbbbbbb:c0"].value, MemberValue::Scalar(6.0));
    }

    #[test]
    fn demand_a_table_of_lap_statistics_only_reads_no_channel() {
        // Arrange
        let doc = workbook(
            "n_laps = count(lap_time())\nunused = max([Fork])",
            r#"{"rowSource":"sessions","columns":[{"id":"c0","template":"[n_laps]"}],"rows":[],"cells":[]}"#,
        );

        // Act
        let plan = plan_set(&doc, PlanScope::Tables);

        // Assert
        assert!(!plan.reads_channels);
        assert_eq!(plan.defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), vec!["n_laps", "table:bbbbbbbb:c0"]);
    }

    #[test]
    fn plan_all_definitions_demands_every_definition() {
        // Arrange
        let doc = workbook("a = 1\nb = max([Fork])", r#"{"columns":[],"rows":[],"cells":[]}"#);

        // Act
        let plan = plan_set(&doc, PlanScope::AllDefinitions);

        // Assert
        assert_eq!(plan.defs.len(), 2);
        assert!(plan.reads_channels);
    }

    fn key_of(md_math: &str, prose: &str, table: &str) -> [u8; 32] {
        let md = format!("{HEAD}\n{prose}\n```math id=aaaaaaaa\n{md_math}\n```\n\n```table id=bbbbbbbb\n{table}\n```\n");
        let doc = parse_workbook(&md).unwrap().0;
        member_key(&plan_set(&doc, PlanScope::Tables), "blob|1|1", "laps", "abc123")
    }

    #[test]
    fn member_key_prose_js_or_graph_edited_key_unchanged() {
        // Act
        let a = key_of("med = median([Fork])", "Some prose.", TABLE);
        let b = key_of("med = median([Fork])", "Other prose ${med}.", TABLE);

        // Assert
        assert_eq!(a, b);
    }

    #[test]
    fn member_key_a_definition_edited_key_changes() {
        // Act
        let a = key_of("med = median([Fork])", "", TABLE);
        let b = key_of("med = mean([Fork])", "", TABLE);

        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn member_key_session_json_long_comment_edited_key_unchanged() {
        // Arrange
        let mut doc = crate::store::session_json::empty_session_json("s1");
        let before = laps_identity(Some(&doc));

        // Act
        doc.long_comment = "great day".into();

        // Assert
        assert_eq!(laps_identity(Some(&doc)), before);
    }

    #[test]
    fn member_key_a_lap_bound_edited_key_changes() {
        // Arrange
        let lap = crate::store::session_json::LapJson {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            raw_elapsed_ms: 0,
            lap_time_ms: 1000,
            start_time_secs: 1.0,
            end_time_secs: 2.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        };
        let mut doc = crate::store::session_json::empty_session_json("s1");
        doc.laps = vec![lap];
        let before = laps_identity(Some(&doc));

        // Act
        doc.laps[0].end_time_secs = 2.5;

        // Assert
        assert_ne!(laps_identity(Some(&doc)), before);
    }
}
