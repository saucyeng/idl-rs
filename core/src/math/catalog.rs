//! Static catalog of every math-expression builtin function, for
//! `list_math_builtins` (C3 §3.4, lead ruling R64.2,
//! `runs/2026-09-03/decisions.md`). Hand transcribed from C2 §3.3's
//! "Builtin catalog" table
//! (`docs/superpowers/specs/2026-09-03-idl1-c2-workbook-v3.md`) — the same
//! source L6's `functionCatalog.ts` was transcribed from — because
//! `eval::call_function` is a dispatch `match` with no attached metadata to
//! extract mechanically; there is nothing else to transcribe from.
//!
//! `arity` is transcribed from C2 §3.3's Signature column: the set of valid
//! argument counts for that name (more than one entry when a name has more
//! than one call form, e.g. `rms(ch)` / `rms(ch, w)` → `&[1, 2]`). `status`
//! mirrors C2 §3.3's Status column collapsed to R64.2's two-value wire
//! enum: a function whose *only* unimplemented form is a secondary arity
//! (`median`'s 2-arg rolling form) is still [`MathBuiltinStatus::Implemented`]
//! here, because its 1-arg form dispatches to real logic.
//!
//! Excludes `main(col[])` (table-cell only, documented in C2 §4, not §3.3)
//! and the grammar keywords `and`/`or`/`not` (parsed as operators in
//! `math::parse`, never reach `call_function`'s dispatch at all) — the same
//! three exclusions L6's `functionCatalog.ts` transcription already made,
//! per that file's own doc comment. C2 §3.3 states 69 named functions total
//! (63 `Implemented` + 6 `NotImplemented`); this catalog has exactly 69
//! entries, none of the four exclusions above being §3.3 rows to begin
//! with.
//!
//! R64.2 requires this catalog's *name* set be checked against
//! `eval::call_function`'s real dispatch table, "never a second hand
//! copy." `call_function` itself is private to `eval.rs` and its match
//! arms aren't enumerable from outside that file without literally
//! retyping every arm's pattern — which would be exactly the "second hand
//! copy" the ruling forbids. This module's own `#[cfg(test)]` below
//! resolves that by calling [`crate::math::evaluate`] (the crate's public
//! parse+eval entry point, which reaches `call_function`) with `"{name}()"`
//! for every catalog entry, and asserting the failure is never
//! [`MathEvalErrorKind::UnknownFunction`]. That proves every catalog name
//! really dispatches; it can't prove the converse (that `call_function`
//! dispatches no *other* name) without the forbidden second copy — flagged
//! in L8w Task 12b's report as a judgment call for the lead to confirm.

/// Whether a catalog entry's function is implemented in
/// `math::eval::call_function` today. Mirrors C2 §3.3's Status column,
/// collapsed to the two-value split that column's own text totals (63
/// Implemented / 6 NotImplemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBuiltinStatus {
    /// The function's documented call form(s) all dispatch to real logic.
    Implemented,
    /// The function parses and validates but its match arm returns
    /// `MathEvalErrorKind::NotImplemented` (C2 §3.3's committed-but-deferred
    /// surface, e.g. `spectrogram`, `hilbert`).
    NotImplemented,
}

/// One math builtin's catalog metadata: its name, the set of argument
/// counts C2 §3.3's signature documents as valid, and whether it is
/// implemented. See the module doc for provenance and the R64.2 ruling
/// this exists for.
#[derive(Debug, Clone, Copy)]
pub struct MathBuiltin {
    /// The function name as written in a `math` cell expression.
    pub name: &'static str,
    /// Valid argument counts (more than one entry when C2 §3.3's signature
    /// documents multiple call forms).
    pub arity: &'static [u32],
    /// Whether the function is implemented (see [`MathBuiltinStatus`]).
    pub status: MathBuiltinStatus,
}

/// Returns the full 69-entry math builtin catalog (C2 §3.3, minus
/// `main(col[])` and the `and`/`or`/`not` grammar keywords — see module
/// doc). Order matches C2 §3.3's table row order.
pub fn math_builtin_catalog() -> &'static [MathBuiltin] {
    use MathBuiltinStatus::{Implemented as I, NotImplemented as N};
    &[
        MathBuiltin { name: "butter", arity: &[4], status: I },
        MathBuiltin { name: "sosfilt", arity: &[2], status: N },
        MathBuiltin { name: "declip", arity: &[1], status: I },
        MathBuiltin { name: "integrate", arity: &[1], status: I },
        MathBuiltin { name: "differentiate", arity: &[1], status: I },
        MathBuiltin { name: "detrend", arity: &[1, 2], status: I },
        MathBuiltin { name: "rms", arity: &[1, 2], status: I },
        MathBuiltin { name: "mean", arity: &[1, 2], status: I },
        MathBuiltin { name: "std", arity: &[1, 2], status: I },
        MathBuiltin { name: "median", arity: &[1], status: I },
        MathBuiltin { name: "sum", arity: &[1], status: I },
        MathBuiltin { name: "count", arity: &[1], status: I },
        MathBuiltin { name: "first", arity: &[1], status: I },
        MathBuiltin { name: "last", arity: &[1], status: I },
        MathBuiltin { name: "p", arity: &[2], status: I },
        MathBuiltin { name: "abs", arity: &[1], status: I },
        MathBuiltin { name: "sqrt", arity: &[1], status: I },
        MathBuiltin { name: "sign", arity: &[1], status: I },
        MathBuiltin { name: "floor", arity: &[1], status: I },
        MathBuiltin { name: "ceil", arity: &[1], status: I },
        MathBuiltin { name: "round", arity: &[1], status: I },
        MathBuiltin { name: "pow", arity: &[2], status: I },
        MathBuiltin { name: "min", arity: &[1, 2], status: I },
        MathBuiltin { name: "max", arity: &[1, 2], status: I },
        MathBuiltin { name: "clamp", arity: &[3], status: I },
        MathBuiltin { name: "sin", arity: &[1], status: I },
        MathBuiltin { name: "cos", arity: &[1], status: I },
        MathBuiltin { name: "tan", arity: &[1], status: I },
        MathBuiltin { name: "asin", arity: &[1], status: I },
        MathBuiltin { name: "acos", arity: &[1], status: I },
        MathBuiltin { name: "atan", arity: &[1], status: I },
        MathBuiltin { name: "atan2", arity: &[2], status: I },
        MathBuiltin { name: "sinh", arity: &[1], status: I },
        MathBuiltin { name: "cosh", arity: &[1], status: I },
        MathBuiltin { name: "tanh", arity: &[1], status: I },
        MathBuiltin { name: "deg2rad", arity: &[1], status: I },
        MathBuiltin { name: "rad2deg", arity: &[1], status: I },
        MathBuiltin { name: "fft", arity: &[2], status: I },
        MathBuiltin { name: "spectrogram", arity: &[1], status: N },
        MathBuiltin { name: "hilbert", arity: &[1], status: N },
        MathBuiltin { name: "correlate", arity: &[2], status: N },
        MathBuiltin { name: "convolve", arity: &[2], status: N },
        MathBuiltin { name: "resample", arity: &[2], status: N },
        MathBuiltin { name: "if", arity: &[3], status: I },
        MathBuiltin { name: "current_lap", arity: &[0], status: I },
        MathBuiltin { name: "lap_start_time", arity: &[1], status: I },
        MathBuiltin { name: "lap_start_distance", arity: &[1], status: I },
        MathBuiltin { name: "sector_number", arity: &[0], status: I },
        MathBuiltin { name: "lap_delta_time", arity: &[1], status: I },
        MathBuiltin { name: "lap_delta_dist", arity: &[1], status: I },
        MathBuiltin { name: "attitude", arity: &[1], status: I },
        MathBuiltin { name: "body_accel", arity: &[1], status: I },
        MathBuiltin { name: "wheel_travel", arity: &[1], status: I },
        MathBuiltin { name: "wheel_velocity", arity: &[1], status: I },
        MathBuiltin { name: "vec", arity: &[3], status: I },
        MathBuiltin { name: "vx", arity: &[1], status: I },
        MathBuiltin { name: "vy", arity: &[1], status: I },
        MathBuiltin { name: "vz", arity: &[1], status: I },
        MathBuiltin { name: "vadd", arity: &[2], status: I },
        MathBuiltin { name: "vsub", arity: &[2], status: I },
        MathBuiltin { name: "vscale", arity: &[2], status: I },
        MathBuiltin { name: "cross", arity: &[2], status: I },
        MathBuiltin { name: "dot", arity: &[2], status: I },
        MathBuiltin { name: "norm", arity: &[1], status: I },
        MathBuiltin { name: "normalize", arity: &[1], status: I },
        MathBuiltin { name: "angle_between", arity: &[2], status: I },
        MathBuiltin { name: "rotate_mat", arity: &[10], status: I },
        MathBuiltin { name: "rotate_axis", arity: &[5], status: I },
        MathBuiltin { name: "rotate_euler", arity: &[4], status: I },
    ]
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
    fn math_builtin_catalog_len_is_69_matching_c2_3_3s_stated_total() {
        // Arrange / Act
        let n = math_builtin_catalog().len();

        // Assert — C2 §3.3 states 69 named functions total (63 Implemented +
        // 6 NotImplemented); `main`/`and`/`or`/`not` are not §3.3 rows, so
        // there is no further subtraction to make.
        assert_eq!(n, 69);
    }

    #[test]
    fn implemented_and_not_implemented_counts_split_63_and_6() {
        // Arrange
        let catalog = math_builtin_catalog();

        // Act
        let not_implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::NotImplemented).count();
        let implemented =
            catalog.iter().filter(|b| b.status == MathBuiltinStatus::Implemented).count();

        // Assert
        assert_eq!(not_implemented, 6);
        assert_eq!(implemented, 63);
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
}
