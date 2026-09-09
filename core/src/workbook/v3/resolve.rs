//! Cross-cell resolver for workbook v3's flat namespace (C2 §2.4). Every
//! `def_line`, in any `math` cell, evaluates deps-first — a definition
//! referenced by two others is computed once — and gets its own entry in the
//! returned map **regardless of whether anything else references it**. This
//! is the behavioral difference from [`crate::math::resolve::resolve_dependencies`]
//! (v2's resolver): that function's "best-effort, swallowed, cycle-guarded by
//! silent skip" contract is exactly right for a math expression's own
//! transparent dependency needs, but wrong here — v3 needs one
//! `CellOutput`-worthy result per definition, always, and a failure must be
//! *visible* per-definition (C2 §3.5.B, CLAUDE.md §5), never swallowed. This
//! is therefore a genuinely new function, not a v3-mode branch of the
//! existing one.
//!
//! **Definitions win over session channels** (L3-R18a) — deliberately the
//! *inverse* of [`crate::session::handle::SessionHandle::lookup`]'s "base +
//! synthesized channels win over the math store" precedent: in v2, the math
//! store is a cache the base session always shadows; in v3, the workbook
//! document is the source of truth, so a definition named after a base
//! channel overrides it, not the other way around. A consequence
//! (L3-R18/R21 addition): a definition that references itself under a base
//! channel's own name (e.g. `IMU0_AccelZ = [IMU0_AccelZ] * 9.81`) is a
//! **cycle**, not a read-through to the base channel — it never resolves and
//! surfaces as the same leftover `UnknownChannel` every other cycle member
//! gets.

use std::collections::HashMap;
use std::sync::Arc;

use crate::math::eval::{evaluate_with_constants, ChannelLookup, EvalOutput, LookupChannel, MathLapContext};
use crate::math::resolve::channel_refs;
use crate::math::{MathEvalError, MathEvalErrorKind};

use super::MathCellDef;

/// Overlay lookup layering a fixed-point pass's already-resolved definitions
/// *over* `base` (L3-R18a: definitions win). `t_us` is stored as `Arc<[i64]>`
/// once per resolved definition, at insertion — a lookup hit then clones the
/// `Arc` (a counter bump), never the underlying `Vec<i64>` (L3-R18b).
struct OverlayLookup<'a> {
    base: &'a dyn ChannelLookup,
    resolved: &'a HashMap<String, (Arc<[f64]>, f64, Arc<[i64]>)>,
}

impl ChannelLookup for OverlayLookup<'_> {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        if let Some((samples, sample_rate_hz, t_us)) = self.resolved.get(name) {
            return Some(LookupChannel {
                samples: samples.clone(),
                sample_rate_hz: *sample_rate_hz,
                t_us: t_us.clone(),
            });
        }
        self.base.lookup(name)
    }

    fn estimator_channel(&self, channel_id: &str) -> Option<LookupChannel> {
        self.base.estimator_channel(channel_id)
    }

    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        self.base.best_time_base_dims()
    }

    fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
        if let Some((samples, sample_rate_hz, _)) = self.resolved.get(name) {
            return Some((samples.len(), *sample_rate_hz));
        }
        self.base.channel_dims(name)
    }

    fn lookup_cell(&self, name: &str) -> Option<f64> {
        self.base.lookup_cell(name)
    }

    fn lookup_cell_column(&self, name: &str) -> Option<Vec<f64>> {
        self.base.lookup_cell_column(name)
    }

    fn sample_times(&self, name: &str) -> Option<Vec<f64>> {
        self.base.sample_times(name)
    }
}

/// One [`MathCellDef`]'s identity in [`resolve_workbook_defs`]'s returned
/// map: `(cell_id, name)`, not `name` alone (ledger R47) — C2 §3.5.A's
/// `DuplicateDefinition` means two different cells can declare the *same*
/// name; each such definition is still evaluated and still gets its own
/// entry here, so the owning cell it panicked recovering from before this
/// fix (`eval.rs`'s `math_cell_defs`) can now always find its own result.
pub type DefKey = (String, String);

/// Resolves every flat-namespace `def_line` in `defs` deps-first (C2 §2.4),
/// memoizing so a definition referenced by two others is computed once, and
/// stores every definition's own result — `Ok` or `Err` — in the returned
/// map, keyed by [`DefKey`] (`(cell_id, name)`, ledger R47 — not `name`
/// alone, which would collapse two same-named definitions in different
/// cells to one entry and silently drop the other), regardless of whether
/// anything else references it (the property that makes this a new
/// function rather than a v3-mode branch of
/// [`crate::math::resolve::resolve_dependencies`]; see the module doc
/// comment).
///
/// **Cross-referencing is still by bare name**, via the internal `resolved`
/// overlay: a `[Name]` reference from a third definition can only ever name
/// one target. When two definitions share a name, **the first in document
/// order wins** (ledger R49) — pinned by explicit construction
/// (`primary_for_name` below), not left as a side effect of fixed-point
/// pass timing, which resolved it order-dependently before this ruling (an
/// unrelated edit could silently swing which duplicate a reference saw).
/// This mirrors [`super::constants::merge_constants`]'s own rule for the
/// same shape of collision (L3-R16/R17: the first declaration of a name
/// wins its table entry, every later one reported) — one document format
/// should not resolve the same kind of collision two different ways. The
/// rejected alternative was making every referencing cell error too; that
/// buries the one signal the user needs (the `DuplicateDefinition` error
/// naming the actual mistake, C2 §3.5.A) under cascading errors on every
/// cell that merely mentions the name. The non-winning duplicate still gets
/// its own result in the returned map (R47) — only the cross-reference
/// overlay ignores it.
///
/// Dependency edges come from [`channel_refs`] filtered to names present in
/// `defs` — a `[Name]` reference to a name *not* in `defs` is a base/session
/// channel and falls through to `lookup` unchanged, exactly as v2's resolver
/// already does. Evaluation is fixed-point: every definition whose
/// dependencies (edges within `defs`) are already resolved gets evaluated
/// via [`evaluate_with_constants`] (never `parse` + [`crate::math::eval::eval`]
/// by hand — that would bypass the private per-pass `MemoLookup`, G6.3)
/// against an [`OverlayLookup`] of everything resolved so far, layered over
/// `lookup` (L3-R18a: definitions win). Anything left over once a pass makes
/// no further progress — a cycle, or a dependent of a definition whose own
/// evaluation failed — gets [`MathEvalErrorKind::UnknownChannel`] (C2 §3.5.A
/// has no dedicated cycle kind), the message naming the one unresolved
/// dependency that blocked it. A definition that references itself under a
/// base channel's own name is one such cycle (L3-R18/R21 addition): because
/// definitions win over session channels, its own name always resolves to
/// *itself* first, so it can never read the base channel — it simply never
/// leaves the leftover set.
pub fn resolve_workbook_defs(
    defs: &[MathCellDef],
    constants: &HashMap<String, f64>,
    lookup: &dyn ChannelLookup,
    lap_ctx: &MathLapContext,
) -> HashMap<DefKey, Result<EvalOutput, MathEvalError>> {
    let def_names: std::collections::HashSet<&str> = defs.iter().map(|d| d.name.as_str()).collect();

    // Each def's within-defs dependency names, computed once up front. Keyed
    // by `DefKey` (ledger R47), not bare name — two defs sharing a name
    // would otherwise collapse to one `deps` entry here too, and a def could
    // end up reading a sibling duplicate's dependency list instead of its
    // own.
    let deps: HashMap<DefKey, Vec<String>> = defs
        .iter()
        .map(|d| {
            let names = channel_refs(&d.expr_text)
                .into_iter()
                .filter(|n| def_names.contains(n.as_str()))
                .collect();
            ((d.cell_id.clone(), d.name.clone()), names)
        })
        .collect();

    // R47/R49: for a duplicated name, the *first* declaration in document
    // order (`defs`'s own order) is the one cross-references see — pinned
    // here, by identity, independent of which duplicate's evaluation
    // happens to finish first in the fixed-point loop below (`.entry(...)
    // .or_insert(d)` keeps only the first occurrence per name, since `defs`
    // is iterated in document order).
    let primary_for_name: HashMap<&str, &MathCellDef> =
        defs.iter().fold(HashMap::new(), |mut m, d| {
            m.entry(d.name.as_str()).or_insert(d);
            m
        });

    let mut resolved: HashMap<String, (Arc<[f64]>, f64, Arc<[i64]>)> = HashMap::new();
    let mut results: HashMap<DefKey, Result<EvalOutput, MathEvalError>> = HashMap::new();
    let mut remaining: Vec<&MathCellDef> = defs.iter().collect();

    loop {
        let mut made_progress = false;
        let mut still_remaining = Vec::new();

        for def in remaining {
            let key = (def.cell_id.clone(), def.name.clone());
            let ready = deps[&key].iter().all(|dep| resolved.contains_key(dep));
            if !ready {
                still_remaining.push(def);
                continue;
            }

            let overlay = OverlayLookup { base: lookup, resolved: &resolved };
            let out = evaluate_with_constants(&def.expr_text, constants, &overlay, lap_ctx);
            match &out {
                Ok(eval_out) => {
                    // Cross-referencing overlay: still name-keyed (see this
                    // function's doc comment) — but only the name's `primary`
                    // (first-in-document-order) definition ever writes here
                    // (R49), regardless of fixed-point pass timing. A
                    // non-primary duplicate still gets its own `results`
                    // entry below (so its own cell shows a real value, not a
                    // suppressed one) — it just never becomes what a third
                    // cell's `[Name]` reference sees.
                    if std::ptr::eq(primary_for_name[def.name.as_str()], def) {
                        resolved.insert(
                            def.name.clone(),
                            (
                                Arc::from(eval_out.samples.as_slice()),
                                eval_out.sample_rate_hz,
                                Arc::from(eval_out.t_us.as_slice()),
                            ),
                        );
                    }
                }
                Err(_) => {
                    // A failed definition still counts as "settled" for the
                    // fixed point — its dependents fall through to the
                    // leftover sweep below rather than looping forever.
                }
            }
            results.insert(key, out);
            made_progress = true;
        }

        remaining = still_remaining;
        if !made_progress || remaining.is_empty() {
            break;
        }
    }

    // Leftovers: cycle members, and dependents of a def whose own evaluation
    // never inserted into `resolved` (either it errored, or it is itself
    // stuck in a cycle). Each gets UnknownChannel naming one blocking dep.
    for def in remaining {
        let key = (def.cell_id.clone(), def.name.clone());
        let blocking = deps[&key]
            .iter()
            .find(|dep| !resolved.contains_key(dep.as_str()))
            .cloned()
            .unwrap_or_else(|| def.name.clone());
        results.insert(
            key,
            Err(MathEvalError::new(
                MathEvalErrorKind::UnknownChannel,
                format!("Channel '[{blocking}]' not in this session"),
            )),
        );
    }

    results
}

/// A [`ChannelLookup`] layering already-inferred sibling-definition units
/// *over* `base`, for [`resolve_workbook_units`]'s fixed-point pass — the
/// unit-inference analogue of [`OverlayLookup`] above. Only carries
/// [`Unit::Known`], non-dimensionless entries (rendered back to a C1-style
/// string, the only shape [`ChannelLookup::unit_of`] can carry): a sibling
/// definition whose own inferred unit is `Scalar`, dimensionless, or
/// `Unknown` has no string to hand back, so a reference to it here falls
/// through to `base` and reports [`UnknownReason::NoSourceUnit`] rather than
/// the more specific state it actually has. This is a conservative
/// approximation, not a wrong one (R152: no unit is never a lie) — a
/// dimensionless or `Scalar` sibling definition should ideally propagate
/// its own state through a `[Name]` reference too, but doing that exactly
/// needs `infer` to accept something richer than `ChannelLookup`'s
/// `Option<String>`, which is future work, not guessed here.
struct UnitOverlayLookup<'a> {
    base: &'a dyn ChannelLookup,
    /// name → rendered unit string, only for `Known` non-dimensionless
    /// sibling definitions already resolved this pass.
    units: &'a HashMap<String, String>,
}

impl ChannelLookup for UnitOverlayLookup<'_> {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        self.base.lookup(name)
    }

    fn unit_of(&self, name: &str) -> Option<String> {
        self.units.get(name).cloned().or_else(|| self.base.unit_of(name))
    }
}

/// Resolves every `def_line`'s inferred unit (R154 §3's "math definition"
/// row: `[Name]` resolving to another definition takes that definition's
/// own inferred unit), in the same dependency order as
/// [`resolve_workbook_defs`] — a definition referenced by two others infers
/// its unit once. Deliberately **independent of value evaluation**: a
/// definition's unit is inferred from its `Ast` alone (R154's "inference is
/// a separate pass"), never from `resolve_workbook_defs`'s `results`, so a
/// definition can report a determined unit even when its own evaluation
/// fails (a mismatched-rate error, say), and vice versa.
///
/// Cross-referencing is by bare name, first-declaration-wins, mirroring
/// [`resolve_workbook_defs`]'s `primary_for_name` rule (R47/R49) — the same
/// document should not resolve a same-name collision two different ways for
/// its value and its unit.
///
/// A dependency cycle among defs (the fixed point makes no further
/// progress) yields [`UnknownReason::Cycle`] for every member — unlike
/// [`resolve_workbook_defs`]'s leftover sweep, which names the one blocking
/// dependency, a cycle has no single blocker to name for a unit.
pub fn resolve_workbook_units(
    defs: &[MathCellDef],
    constants: &HashMap<String, f64>,
    lookup: &dyn ChannelLookup,
) -> HashMap<DefKey, (crate::math::units::Unit, Vec<crate::math::units::UnitNote>)> {
    use crate::math::units::{infer, Unit, UnknownReason};

    let def_names: std::collections::HashSet<&str> = defs.iter().map(|d| d.name.as_str()).collect();

    let deps: HashMap<DefKey, Vec<String>> = defs
        .iter()
        .map(|d| {
            let names = channel_refs(&d.expr_text)
                .into_iter()
                .filter(|n| def_names.contains(n.as_str()))
                .collect();
            ((d.cell_id.clone(), d.name.clone()), names)
        })
        .collect();

    let primary_for_name: HashMap<&str, &MathCellDef> =
        defs.iter().fold(HashMap::new(), |mut m, d| {
            m.entry(d.name.as_str()).or_insert(d);
            m
        });

    let mut resolved_units: HashMap<String, String> = HashMap::new();
    let mut settled_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut results: HashMap<DefKey, (Unit, Vec<crate::math::units::UnitNote>)> = HashMap::new();
    let mut remaining: Vec<&MathCellDef> = defs.iter().collect();

    loop {
        let mut made_progress = false;
        let mut still_remaining = Vec::new();

        for def in remaining {
            let key = (def.cell_id.clone(), def.name.clone());
            let ready = deps[&key].iter().all(|dep| settled_names.contains(dep));
            if !ready {
                still_remaining.push(def);
                continue;
            }

            let overlay = UnitOverlayLookup { base: lookup, units: &resolved_units };
            let (unit, notes) = match crate::math::parse::parse_with_constants(&def.expr_text, constants) {
                Ok(ast) => infer(&ast, &overlay),
                // The definition does not even parse — `resolve_workbook_defs`
                // already surfaces the real MathEvalError on `CellDefResult::error`;
                // this pass only needs a unit state to publish alongside it.
                Err(_) => {
                    (Unit::Unknown(UnknownReason::Propagated { of: "a definition with a parse error".to_string() }), Vec::new())
                }
            };

            if std::ptr::eq(primary_for_name[def.name.as_str()], def) {
                if let Unit::Known(u) = &unit {
                    if !u.is_dimensionless() {
                        resolved_units.insert(def.name.clone(), u.to_string());
                    }
                }
                settled_names.insert(def.name.clone());
            }
            results.insert(key, (unit, notes));
            made_progress = true;
        }

        remaining = still_remaining;
        if !made_progress || remaining.is_empty() {
            break;
        }
    }

    // Leftovers are cycle members — every one of them gets Unknown(Cycle),
    // no single blocking dependency to name (unlike resolve_workbook_defs's
    // leftover sweep, which reports one).
    for def in remaining {
        results.insert(
            (def.cell_id.clone(), def.name.clone()),
            (Unit::Unknown(UnknownReason::Cycle), Vec::new()),
        );
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptyLookup;
    impl ChannelLookup for EmptyLookup {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
    }

    fn def(cell_id: &str, name: &str, expr_text: &str, order: usize) -> MathCellDef {
        MathCellDef {
            cell_id: cell_id.to_string(),
            name: name.to_string(),
            expr_text: expr_text.to_string(),
            label: None,
            order,
        }
    }

    fn no_laps() -> MathLapContext {
        MathLapContext::empty()
    }

    /// Looks up a [`resolve_workbook_defs`] result by `(cell_id, name)`
    /// (ledger R47) — every test below uses a single `cell_id` ("c1") per
    /// def, so this is a thin convenience over the real `DefKey`.
    fn get<'a>(
        out: &'a HashMap<DefKey, Result<EvalOutput, MathEvalError>>,
        cell_id: &str,
        name: &str,
    ) -> &'a Result<EvalOutput, MathEvalError> {
        &out[&(cell_id.to_string(), name.to_string())]
    }

    // A lookup double with one named base channel, for the overlay-precedence
    // and base-fallback tests below.
    struct OneChannel {
        name: &'static str,
        samples: Vec<f64>,
        rate: f64,
    }
    impl ChannelLookup for OneChannel {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            if name == self.name {
                let t_us = (0..self.samples.len())
                    .map(|i| (i as f64 * 1_000_000.0 / self.rate) as i64)
                    .collect();
                Some(LookupChannel { samples: self.samples.clone().into(), sample_rate_hz: self.rate, t_us })
            } else {
                None
            }
        }
    }

    // A lookup double with one named base channel carrying a C1 unit
    // (`unit_of`), for `resolve_workbook_units`'s tests below.
    struct UnitChannel {
        name: &'static str,
        unit: &'static str,
    }
    impl ChannelLookup for UnitChannel {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
        fn unit_of(&self, name: &str) -> Option<String> {
            if name == self.name {
                Some(self.unit.to_string())
            } else {
                None
            }
        }
    }

    /// Looks up a [`resolve_workbook_units`] result by `(cell_id, name)`.
    fn get_unit<'a>(
        out: &'a HashMap<DefKey, (crate::math::units::Unit, Vec<crate::math::units::UnitNote>)>,
        cell_id: &str,
        name: &str,
    ) -> &'a (crate::math::units::Unit, Vec<crate::math::units::UnitNote>) {
        &out[&(cell_id.to_string(), name.to_string())]
    }

    #[test]
    fn resolve_units_channel_ref_takes_the_base_channel_unit() {
        // Arrange
        let defs = vec![def("c1", "TravelMm", "[Travel]", 0)];
        let lk = UnitChannel { name: "Travel", unit: "mm" };

        // Act
        let out = resolve_workbook_units(&defs, &HashMap::new(), &lk);

        // Assert
        assert_eq!(
            get_unit(&out, "c1", "TravelMm").0,
            crate::math::units::Unit::Known(crate::math::units::UnitExpr::atom("mm"))
        );
    }

    #[test]
    fn resolve_units_a_references_b_takes_bs_own_inferred_unit() {
        // Arrange — R154 §3: a math-definition [Name] reference takes that
        // definition's own inferred unit, in dependency order.
        let defs = vec![def("c1", "A", "[B] * 2", 0), def("c1", "B", "[Travel]", 1)];
        let lk = UnitChannel { name: "Travel", unit: "mm" };

        // Act
        let out = resolve_workbook_units(&defs, &HashMap::new(), &lk);

        // Assert
        assert_eq!(
            get_unit(&out, "c1", "A").0,
            crate::math::units::Unit::Known(crate::math::units::UnitExpr::atom("mm"))
        );
    }

    #[test]
    fn resolve_units_cycle_reports_unknown_cycle_for_every_member_no_infinite_loop() {
        // Arrange
        let defs = vec![def("c1", "A", "[B] + 1", 0), def("c1", "B", "[A] + 1", 1)];
        let lk = UnitChannel { name: "nothing", unit: "mm" };

        // Act — must terminate.
        let out = resolve_workbook_units(&defs, &HashMap::new(), &lk);

        // Assert
        assert_eq!(
            get_unit(&out, "c1", "A").0,
            crate::math::units::Unit::Unknown(crate::math::units::UnknownReason::Cycle)
        );
        assert_eq!(
            get_unit(&out, "c1", "B").0,
            crate::math::units::Unit::Unknown(crate::math::units::UnknownReason::Cycle)
        );
    }

    #[test]
    fn a_depends_on_b_depends_on_a_both_report_unknown_channel_no_panic_no_infinite_loop() {
        // Arrange
        let defs = vec![def("c1", "A", "[B] + 1", 0), def("c1", "B", "[A] + 1", 1)];

        // Act — must terminate.
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "A").as_ref().unwrap_err().kind, MathEvalErrorKind::UnknownChannel);
        assert_eq!(get(&out, "c1", "B").as_ref().unwrap_err().kind, MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn a_depends_on_b_depends_on_c_no_cycle_resolves_in_one_pass_c_before_b_before_a() {
        // Arrange — C is a plain scalar; B = [C] + 1; A = [B] + 1.
        let defs = vec![def("c1", "A", "[B] + 1", 0), def("c1", "B", "[C] + 1", 1), def("c1", "C", "10", 2)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "C").as_ref().unwrap().samples, vec![10.0]);
        assert_eq!(get(&out, "c1", "B").as_ref().unwrap().samples, vec![11.0]);
        assert_eq!(get(&out, "c1", "A").as_ref().unwrap().samples, vec![12.0]);
    }

    #[test]
    fn self_reference_against_a_base_channel_of_the_same_name_is_a_cycle_leftover_unknown_channel_naming_it_never_loops_never_reads_the_base_channel(
    ) {
        // Arrange — L3-R18/R21: a definition named after a base channel that
        // references itself. Definitions win over session channels, so
        // `[IMU0_AccelZ]` inside the definition resolves to the definition
        // itself, not the base channel — a cycle, not a read-through.
        let base = OneChannel { name: "IMU0_AccelZ", samples: vec![1.0, 1.0], rate: 10.0 };
        let defs = vec![def("c1", "IMU0_AccelZ", "[IMU0_AccelZ] * 9.81", 0)];

        // Act — must terminate, not read the base channel's [1.0, 1.0].
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &base, &no_laps());

        // Assert
        let err = get(&out, "c1", "IMU0_AccelZ").as_ref().unwrap_err();
        assert_eq!(err.kind, MathEvalErrorKind::UnknownChannel);
        assert!(err.message.contains("IMU0_AccelZ"), "{}", err.message);
    }

    #[test]
    fn definition_named_after_a_base_channel_the_definitions_value_is_returned_not_the_base_channels() {
        // Arrange — L3-R18a: a definition and a base channel share a name;
        // the definition wins.
        let base = OneChannel { name: "Speed", samples: vec![1.0, 2.0], rate: 10.0 };
        let defs = vec![def("c1", "Speed", "99", 0)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &base, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "Speed").as_ref().unwrap().samples, vec![99.0]);
    }

    #[test]
    fn three_independent_definitions_none_referencing_each_other_all_three_resolve() {
        // Arrange
        let defs = vec![def("c1", "A", "1", 0), def("c1", "B", "2", 1), def("c1", "C", "3", 2)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "A").as_ref().unwrap().samples, vec![1.0]);
        assert_eq!(get(&out, "c1", "B").as_ref().unwrap().samples, vec![2.0]);
        assert_eq!(get(&out, "c1", "C").as_ref().unwrap().samples, vec![3.0]);
    }

    #[test]
    fn one_definitions_expression_has_a_parse_error_its_own_entry_is_err_parse_sibling_definitions_still_resolve()
    {
        // Arrange — "1 +" is an incomplete expression (Parse error); "ok" is fine.
        let defs = vec![def("c1", "bad", "1 +", 0), def("c1", "ok", "5", 1)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "bad").as_ref().unwrap_err().kind, MathEvalErrorKind::Parse);
        assert_eq!(get(&out, "c1", "ok").as_ref().unwrap().samples, vec![5.0]);
    }

    #[test]
    fn definition_referencing_a_base_session_channel_not_in_defs_falls_through_to_lookup_as_today_s_v2_resolver_does(
    ) {
        // Arrange
        let base = OneChannel { name: "IMU0_AccelZ", samples: vec![1.0, 2.0], rate: 10.0 };
        let defs = vec![def("c1", "scaled", "[IMU0_AccelZ] * 2", 0)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &base, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "scaled").as_ref().unwrap().samples, vec![2.0, 4.0]);
    }

    #[test]
    fn a_definition_referenced_by_two_others_is_stored_once_and_shared() {
        // Arrange — B and C both reference A; A itself gets its own entry too
        // (v3's "store every definition, referenced or not" rule).
        let defs = vec![def("c1", "A", "10", 0), def("c1", "B", "[A] + 1", 1), def("c1", "C", "[A] + 2", 2)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out.len(), 3);
        assert_eq!(get(&out, "c1", "A").as_ref().unwrap().samples, vec![10.0]);
        assert_eq!(get(&out, "c1", "B").as_ref().unwrap().samples, vec![11.0]);
        assert_eq!(get(&out, "c1", "C").as_ref().unwrap().samples, vec![12.0]);
    }

    #[test]
    fn constants_table_is_passed_through_to_evaluate_with_constants() {
        // Arrange
        let defs = vec![def("c1", "scaled", "rider_mass_kg * 2", 0)];
        let constants = HashMap::from([("rider_mass_kg".to_string(), 82.0)]);

        // Act
        let out = resolve_workbook_defs(&defs, &constants, &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "scaled").as_ref().unwrap().samples, vec![164.0]);
    }

    #[test]
    fn a_name_repeated_in_two_different_cells_both_still_get_their_own_result_no_panic_no_dropped_entry() {
        // Arrange — ledger R47: `DuplicateDefinition` (C2 §3.5.A) means two
        // cells can share a name; before this fix the second cell's
        // `math_cell_defs` panicked recovering its own result because
        // `results` was keyed by bare name and the first cell's entry had
        // already been removed.
        let defs = vec![def("c1", "x", "1", 0), def("c2", "x", "2", 0)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert — both keyed entries present, each cell's own value intact.
        assert_eq!(out.len(), 2);
        assert_eq!(get(&out, "c1", "x").as_ref().unwrap().samples, vec![1.0]);
        assert_eq!(get(&out, "c2", "x").as_ref().unwrap().samples, vec![2.0]);
    }

    #[test]
    fn a_duplicated_name_a_third_definitions_reference_resolves_to_the_first_in_document_order_not_the_second(
    ) {
        // Arrange — ledger R49: two cells both define "x" (c1's first in
        // document order, c2's second, deliberately different values so the
        // assertion below fails outright if the rule ever flips to
        // "last wins" or reverts to pass-timing-dependent behaviour); a
        // third cell's `y = [x] + 1` must see c1's "x" (10), never c2's (20).
        let defs = vec![def("c1", "x", "10", 0), def("c2", "x", "20", 0), def("c3", "y", "[x] + 1", 0)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(get(&out, "c1", "x").as_ref().unwrap().samples, vec![10.0]);
        assert_eq!(get(&out, "c2", "x").as_ref().unwrap().samples, vec![20.0]);
        assert_eq!(get(&out, "c3", "y").as_ref().unwrap().samples, vec![11.0], "y must resolve against the first-declared x (10), not the second (20)");
    }
}
