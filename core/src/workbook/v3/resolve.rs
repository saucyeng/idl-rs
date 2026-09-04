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

/// Resolves every flat-namespace `def_line` in `defs` deps-first (C2 §2.4),
/// memoizing so a definition referenced by two others is computed once, and
/// stores every definition's own result — `Ok` or `Err` — in the returned
/// map, keyed by name, regardless of whether anything else references it
/// (the property that makes this a new function rather than a v3-mode
/// branch of [`crate::math::resolve::resolve_dependencies`]; see the module
/// doc comment).
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
) -> HashMap<String, Result<EvalOutput, MathEvalError>> {
    let def_names: std::collections::HashSet<&str> = defs.iter().map(|d| d.name.as_str()).collect();

    // Each def's within-defs dependency names, computed once up front.
    let deps: HashMap<&str, Vec<String>> = defs
        .iter()
        .map(|d| {
            let names = channel_refs(&d.expr_text)
                .into_iter()
                .filter(|n| def_names.contains(n.as_str()))
                .collect();
            (d.name.as_str(), names)
        })
        .collect();

    let mut resolved: HashMap<String, (Arc<[f64]>, f64, Arc<[i64]>)> = HashMap::new();
    let mut results: HashMap<String, Result<EvalOutput, MathEvalError>> = HashMap::new();
    let mut remaining: Vec<&MathCellDef> = defs.iter().collect();

    loop {
        let mut made_progress = false;
        let mut still_remaining = Vec::new();

        for def in remaining {
            let ready = deps[def.name.as_str()].iter().all(|dep| resolved.contains_key(dep));
            if !ready {
                still_remaining.push(def);
                continue;
            }

            let overlay = OverlayLookup { base: lookup, resolved: &resolved };
            let out = evaluate_with_constants(&def.expr_text, constants, &overlay, lap_ctx);
            match &out {
                Ok(eval_out) => {
                    resolved.insert(
                        def.name.clone(),
                        (
                            Arc::from(eval_out.samples.as_slice()),
                            eval_out.sample_rate_hz,
                            Arc::from(eval_out.t_us.as_slice()),
                        ),
                    );
                }
                Err(_) => {
                    // A failed definition still counts as "settled" for the
                    // fixed point — its dependents fall through to the
                    // leftover sweep below rather than looping forever.
                }
            }
            results.insert(def.name.clone(), out);
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
        let blocking = deps[def.name.as_str()]
            .iter()
            .find(|dep| !resolved.contains_key(dep.as_str()))
            .cloned()
            .unwrap_or_else(|| def.name.clone());
        results.insert(
            def.name.clone(),
            Err(MathEvalError::new(
                MathEvalErrorKind::UnknownChannel,
                format!("Channel '[{blocking}]' not in this session"),
            )),
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

    #[test]
    fn a_depends_on_b_depends_on_a_both_report_unknown_channel_no_panic_no_infinite_loop() {
        // Arrange
        let defs = vec![def("c1", "A", "[B] + 1", 0), def("c1", "B", "[A] + 1", 1)];

        // Act — must terminate.
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out["A"].as_ref().unwrap_err().kind, MathEvalErrorKind::UnknownChannel);
        assert_eq!(out["B"].as_ref().unwrap_err().kind, MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn a_depends_on_b_depends_on_c_no_cycle_resolves_in_one_pass_c_before_b_before_a() {
        // Arrange — C is a plain scalar; B = [C] + 1; A = [B] + 1.
        let defs = vec![def("c1", "A", "[B] + 1", 0), def("c1", "B", "[C] + 1", 1), def("c1", "C", "10", 2)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out["C"].as_ref().unwrap().samples, vec![10.0]);
        assert_eq!(out["B"].as_ref().unwrap().samples, vec![11.0]);
        assert_eq!(out["A"].as_ref().unwrap().samples, vec![12.0]);
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
        let err = out["IMU0_AccelZ"].as_ref().unwrap_err();
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
        assert_eq!(out["Speed"].as_ref().unwrap().samples, vec![99.0]);
    }

    #[test]
    fn three_independent_definitions_none_referencing_each_other_all_three_resolve() {
        // Arrange
        let defs = vec![def("c1", "A", "1", 0), def("c1", "B", "2", 1), def("c1", "C", "3", 2)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out["A"].as_ref().unwrap().samples, vec![1.0]);
        assert_eq!(out["B"].as_ref().unwrap().samples, vec![2.0]);
        assert_eq!(out["C"].as_ref().unwrap().samples, vec![3.0]);
    }

    #[test]
    fn one_definitions_expression_has_a_parse_error_its_own_entry_is_err_parse_sibling_definitions_still_resolve()
    {
        // Arrange — "1 +" is an incomplete expression (Parse error); "ok" is fine.
        let defs = vec![def("c1", "bad", "1 +", 0), def("c1", "ok", "5", 1)];

        // Act
        let out = resolve_workbook_defs(&defs, &HashMap::new(), &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out["bad"].as_ref().unwrap_err().kind, MathEvalErrorKind::Parse);
        assert_eq!(out["ok"].as_ref().unwrap().samples, vec![5.0]);
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
        assert_eq!(out["scaled"].as_ref().unwrap().samples, vec![2.0, 4.0]);
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
        assert_eq!(out["A"].as_ref().unwrap().samples, vec![10.0]);
        assert_eq!(out["B"].as_ref().unwrap().samples, vec![11.0]);
        assert_eq!(out["C"].as_ref().unwrap().samples, vec![12.0]);
    }

    #[test]
    fn constants_table_is_passed_through_to_evaluate_with_constants() {
        // Arrange
        let defs = vec![def("c1", "scaled", "rider_mass_kg * 2", 0)];
        let constants = HashMap::from([("rider_mass_kg".to_string(), 82.0)]);

        // Act
        let out = resolve_workbook_defs(&defs, &constants, &EmptyLookup, &no_laps());

        // Assert
        assert_eq!(out["scaled"].as_ref().unwrap().samples, vec![164.0]);
    }
}
