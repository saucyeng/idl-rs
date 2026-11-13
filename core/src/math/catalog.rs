//! The math builtin catalog, for `list_math_builtins` (C3 §3.4, ruling
//! R64.2) and the generated workbook reference: one row per builtin, read
//! from [`crate::math::registry::Registry::builtins`] (spec
//! `2026-11-13-idl-rs-builtin-registry-v2.md`), so the documented set is the
//! dispatched set by construction — the second hand copy R64.2 warned about
//! no longer exists.
//!
//! `arity` is derived from each function's parameter types, or declared at
//! registration for a function with several call forms (`rms(ch)` /
//! `rms(ch, w)` → `&[1, 2]`). `status` collapses C2 §3.3's Status column to
//! R64.2's two values: a function whose *only* unimplemented form is a
//! secondary arity (`median`'s 2-arg rolling form) is
//! [`MathBuiltinStatus::Implemented`], because its 1-arg form is real.
//!
//! Excludes `main(col[])` (table-cell only, C2 §4, not §3.3) and the
//! grammar keywords `and`/`or`/`not` (operators in `math::parse`). Order is
//! C2 §3.3's table row order, the order `math::builtins` registers in.

/// Whether a catalog entry's function is implemented by its registered body
/// today. Mirrors C2 §3.3's Status column,
/// collapsed to the two-value split that column's own text totals (63
/// Implemented / 6 NotImplemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBuiltinStatus {
    /// The function's documented call form(s) all dispatch to real logic.
    Implemented,
    /// The function parses and validates but its match arm returns
    /// `MathEvalErrorKind::NotImplemented` (C2 §3.3's committed-but-deferred
    /// surface, e.g. `spectrogram`, `envelope`).
    NotImplemented,
}

/// One math builtin's catalog metadata: its name, the set of argument
/// counts C2 §3.3's signature documents as valid, whether it is
/// implemented, and (added 2026-09-11, ruling R222 item 1) the
/// documentation columns the generated workbook reference and the
/// notebook editor's hover card are both rendered from. See the module
/// doc for provenance and the R64.2 ruling this exists for.
///
/// The six documentation fields are transcribed from the same C2 §3.3
/// table row as `arity`/`status` — `signature` and `category` verbatim
/// from those columns, `unit_rule` from the "Unit rule (§3.3.1)" column
/// with its explanatory tail dropped, `shape` from the signature's own
/// result annotation cross-checked against §3.6, and `description` and
/// `example` written here (no column carries either). One source, two
/// consumers: `idl-rs docs workbook` renders them into
/// `docs/WORKBOOK-REFERENCE.md` and `list_math_builtins` puts the same
/// strings on the wire for the editor's hover, so the reference and the
/// tooltip can never disagree.
#[derive(Debug, Clone, Copy)]
pub struct MathBuiltin {
    /// The function name as written in a `math` cell expression.
    pub name: &'static str,
    /// Valid argument counts (more than one entry when C2 §3.3's signature
    /// documents multiple call forms).
    pub arity: &'static [u32],
    /// Whether the function is implemented (see [`MathBuiltinStatus`]).
    pub status: MathBuiltinStatus,
    /// C2 §3.3's Category column, verbatim — the grouping heading the
    /// generated reference sorts entries under.
    pub category: &'static str,
    /// C2 §3.3's Signature column, verbatim: the call form(s) as a human
    /// reads them, `|`-separated when there is more than one.
    pub signature: &'static str,
    /// C2 §3.3's "Unit rule (§3.3.1)" column, the rule expression only
    /// (the column's prose explanation is not carried).
    pub unit_rule: &'static str,
    /// The result's value shape in §3.6's vocabulary: `scalar`, `[t]`
    /// (a time-axis channel), `[f]`, `[t,f]`, `Vec3`, or a `|`-separated
    /// pair when the shape depends on the call form.
    pub shape: &'static str,
    /// One line saying what the function computes. Written for this
    /// catalog; no C2 column carries it.
    pub description: &'static str,
    /// One runnable example call, as it would be typed on the right of a
    /// `math` definition's `=`. Written for this catalog.
    pub example: &'static str,
}

/// Returns the full 77-entry math builtin catalog (C2 §3.3, minus
/// `main(col[])` and the `and`/`or`/`not` grammar keywords — see module
/// doc), in C2 §3.3's table row order.
pub fn math_builtin_catalog() -> &'static [MathBuiltin] {
    static CATALOG: std::sync::OnceLock<Vec<MathBuiltin>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| crate::math::registry::Registry::shared().catalog())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{evaluate, ChannelLookup, LookupChannel, MathEvalErrorKind, MathLapContext};

    /// A lookup with no channels — every catalog entry's dispatch check
    /// calls its function with zero arguments, so no channel reference is
    /// ever reached.
    struct EmptyLookup;
    impl ChannelLookup for EmptyLookup {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
    }

    #[test]
    fn math_builtin_catalog_len_is_77_after_the_scipy_splits_the_lap_shaped_trio_deglitch_pulses_and_keep() {
        // Arrange / Act
        let n = math_builtin_catalog().len();

        // Assert — was 69 (C2 §3.3's original stated total: 63 Implemented +
        // 6 NotImplemented); the scipy-alignment lane adds three entries
        // without removing any (`runs/2026-09-08/scipy-alignment-plan.md`):
        // `fft` split into `periodogram` + `welch` (net +1, task 6),
        // `cumtrapz` added as `cumulative_trapezoid`'s permanent second
        // spelling (net +1, task 10, R151 item 6), and `gradient` added
        // alongside `differentiate` (net +1, task 11, R151 item 3) —
        // 69 + 3 = 72; then `lap_number`/`lap_time`/`sector_time`, which C2
        // §3.3 has listed all along and this catalog lacked (R233), 72 + 3 = 75;
        // then `deglitch_pulses` for wheel-pulse channels (2026-11-05), 76;
        // then `keep` for event lists (2026-11-05), 77.
        assert_eq!(n, 77);
    }

    #[test]
    fn implemented_and_not_implemented_counts_split_71_and_5() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let not_implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::NotImplemented).count();
        let implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::Implemented).count();

        // Assert — was 63/6; the scipy-alignment lane's three net-new entries
        // and the lap-shaped trio (see the length test above) are all
        // Implemented, so only that side of the split moved (69/6). R242 then
        // implements `resample`, moving one entry across: 70/5. `deglitch_pulses`
        // arrives implemented: 71/5; `keep` too: 72/5.
        assert_eq!(not_implemented, 5);
        assert_eq!(implemented, 72);
    }

    #[test]
    fn multi_arity_entries_have_more_than_one_valid_arity() {
        // Arrange
        let catalog = math_builtin_catalog();
        let arity_of = |name: &str| catalog.iter().find(|b| b.name == name).unwrap().arity;

        // Act / Assert
        for name in ["rms", "mean", "std", "min", "max"] {
            assert!(arity_of(name).len() > 1, "{name} should have more than one valid arity");
        }
    }

    #[test]
    fn butter_has_a_single_fixed_arity_of_four() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let butter = catalog.iter().find(|b| b.name == "butter").unwrap();

        // Assert
        assert_eq!(butter.arity, &[4]);
    }

    #[test]
    fn no_duplicate_names_in_the_catalog() {
        // Arrange
        let mut names: Vec<&str> = math_builtin_catalog().iter().map(|b| b.name).collect();
        let before = names.len();

        // Act
        names.sort_unstable();
        names.dedup();

        // Assert
        assert_eq!(names.len(), before, "duplicate name in math_builtin_catalog");
    }

    #[test]
    fn every_catalog_name_dispatches_in_call_functions_real_match() {
        // Arrange — see module doc for why this is the check R64.2 asks
        // for rather than a second hand-typed name list.
        let lookup = EmptyLookup;
        let lap_ctx = MathLapContext::empty();

        for entry in math_builtin_catalog() {
            // Act — a zero-arg call is always syntactically valid; every
            // match arm in `call_function` validates argument count/type
            // before indexing `args`, so this never panics regardless of
            // the function's real arity.
            let result = evaluate(&format!("{}()", entry.name), &lookup, &lap_ctx);

            // Assert — any error other than UnknownFunction proves the
            // name reached the dispatch match (ArgCount/Type/NotImplemented
            // etc. are expected and fine here).
            if let Err(e) = result {
                assert_ne!(
                    e.kind,
                    MathEvalErrorKind::UnknownFunction,
                    "{}: not found in eval::call_function's dispatch table",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn every_entry_carries_all_six_documentation_columns() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act / Assert — a blank column would render as an empty cell in
        // `docs::render_workbook_reference`'s table and as an empty hover
        // card in the editor, neither of which fails loudly on its own.
        for entry in catalog {
            assert!(!entry.category.is_empty(), "{}: empty category", entry.name);
            assert!(!entry.signature.is_empty(), "{}: empty signature", entry.name);
            assert!(!entry.unit_rule.is_empty(), "{}: empty unit_rule", entry.name);
            assert!(!entry.shape.is_empty(), "{}: empty shape", entry.name);
            assert!(!entry.description.is_empty(), "{}: empty description", entry.name);
            assert!(!entry.example.is_empty(), "{}: empty example", entry.name);
        }
    }

    #[test]
    fn every_signature_and_example_opens_with_the_entrys_own_name() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act / Assert — the cheapest guard against a copy-paste row, which
        // is the realistic transcription error in a 75-row hand-typed table.
        for entry in catalog {
            assert!(
                entry.signature.starts_with(&format!("{}(", entry.name)),
                "{}: signature does not open with the function's own name",
                entry.name
            );
            assert!(
                entry.example.starts_with(&format!("{}(", entry.name)),
                "{}: example does not call the function it documents",
                entry.name
            );
        }
    }

    #[test]
    fn not_implemented_entries_say_so_in_their_description() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let silent: Vec<&str> = catalog
            .iter()
            .filter(|b| b.status == MathBuiltinStatus::NotImplemented)
            .filter(|b| !b.description.contains("Not implemented"))
            .map(|b| b.name)
            .collect();

        // Assert — the generated reference prints the description under the
        // heading; a deferred function whose prose reads like a working one
        // is exactly the false friend C2 §3.8's naming policy exists to stop.
        assert_eq!(silent, Vec::<&str>::new());
    }
}
