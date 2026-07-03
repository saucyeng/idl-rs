//! Math-channel dependency resolver. Ported verbatim from the Dart
//! `_resolveDependenciesIntoHandle`
//! (`app/lib/providers/math_channel_provider.dart`): walk the `[Name]`
//! references in an expression, and for each that names another math channel,
//! evaluate it deps-first and write the result into the handle's math store so
//! the outer evaluation reads it Rust-side. Best-effort — per-dependency errors
//! are swallowed; the outer eval surfaces a clean error if a consumer needed
//! the channel. A `visited` set guards cycles.

use std::collections::{HashMap, HashSet};

use crate::math::channel_def::MathChannelDef;
use crate::math::eval::{evaluate, MathLapContext};
use crate::session::handle::SessionHandle;

/// Extract the `[Name]` channel references in `expr`, in order of appearance.
///
/// Mirrors the Dart resolver's regex `\[([^\[\]]+)\]`: each maximal run of
/// non-bracket characters delimited by `[` … `]`. Best-effort and total — an
/// unbalanced or nested bracket simply yields no match for that fragment rather
/// than erroring, preserving the Dart "swallow and let the outer eval report
/// it" behavior. Empty `[]` is skipped.
pub(crate) fn channel_refs(expr: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut rest = expr;
    while let Some(open) = rest.find('[') {
        rest = &rest[open + 1..];
        match rest.find(']') {
            Some(close) => {
                let name = &rest[..close];
                // `[^\[\]]+`: non-empty, no nested '['.
                if !name.is_empty() && !name.contains('[') {
                    refs.push(name.to_string());
                }
                rest = &rest[close + 1..];
            }
            None => break, // unterminated '[' — nothing more to extract
        }
    }
    refs
}

/// Resolve the transitive math-channel dependencies referenced by `expression`
/// into `handle`'s math store. For each `[Name]` reference that names a math
/// channel in `defs`, evaluate it deps-first via [`crate::math::evaluate`] and
/// write the result with [`SessionHandle::store_math`], so the outer
/// evaluation reads it Rust-side without marshalling samples.
///
/// A `[Name]` not in `defs` is left alone — it is either a base/synthesized
/// channel (which `evaluate` looks up directly) or genuinely unknown (which
/// `evaluate` errors on). Because math-channel names are distinct from base
/// channel ids by design, `defs` membership is equivalent to the Dart
/// resolver's "skip base channels, then look up by name" path.
///
/// Best-effort: a failed sub-evaluation is swallowed. `visited` is the cycle
/// guard — seed it with the target channel's own name before the first call so
/// `A` referencing `[A]` is not resolved as its own dependency.
pub fn resolve_dependencies(
    handle: &SessionHandle,
    expression: &str,
    defs: &HashMap<String, MathChannelDef>,
    lap_ctx: &MathLapContext,
    visited: &mut HashSet<String>,
) {
    for name in channel_refs(expression) {
        if visited.contains(&name) {
            continue;
        }
        let Some(def) = defs.get(&name) else {
            continue; // base/synth channel or unknown — not a math dependency
        };
        visited.insert(name.clone());
        resolve_dependencies(handle, &def.expression, defs, lap_ctx, visited);
        if let Ok(out) = evaluate(&def.expression, handle, lap_ctx) {
            handle.store_math(&name, out.sample_rate_hz, out.samples);
        }
        visited.remove(&name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::eval::ChannelLookup;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn base(id: &str, samples: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 10.0, samples, sample_times_secs: None }
    }

    fn defs(pairs: &[(&str, &str)]) -> HashMap<String, MathChannelDef> {
        pairs
            .iter()
            .map(|(n, e)| {
                (n.to_string(), MathChannelDef { name: n.to_string(), expression: e.to_string() })
            })
            .collect()
    }

    #[test]
    fn extracts_single_reference() {
        // Act + Assert
        assert_eq!(channel_refs("differentiate([ForkTravel])"), vec!["ForkTravel"]);
    }

    #[test]
    fn extracts_multiple_references_in_order() {
        // Act + Assert
        assert_eq!(channel_refs("[A] + [B] * [A]"), vec!["A", "B", "A"]);
    }

    #[test]
    fn ignores_empty_and_unterminated_brackets() {
        // Act + Assert — `[]` skipped; trailing `[C` has no closer.
        assert_eq!(channel_refs("[] + 1 + [C"), Vec::<String>::new());
    }

    #[test]
    fn no_references_returns_empty() {
        // Act + Assert
        assert_eq!(channel_refs("sqrt(2) + 3"), Vec::<String>::new());
    }

    #[test]
    fn resolves_transitive_dependency_into_store() {
        // Arrange — base X; B = [X] * 2; A = [B] + 1. Resolving A's expression
        // must leave B (and not A) in the store.
        let h = handle_with(vec![base("X", vec![1.0, 2.0, 3.0])]);
        let d = defs(&[("A", "[B] + 1"), ("B", "[X] * 2")]);
        let mut visited = HashSet::from(["A".to_string()]);

        // Act
        resolve_dependencies(&h, "[B] + 1", &d, &MathLapContext::empty(), &mut visited);

        // Assert — B is now resolvable from the handle; A is not (it is the target).
        assert_eq!(h.lookup("B").unwrap().samples, vec![2.0, 4.0, 6.0].into());
        assert!(h.lookup("A").is_none());
    }

    #[test]
    fn cycle_terminates_without_panic() {
        // Arrange — A = [B], B = [A]. Resolving A's expression must not recurse
        // forever.
        let h = handle_with(vec![base("X", vec![1.0])]);
        let d = defs(&[("A", "[B]"), ("B", "[A]")]);
        let mut visited = HashSet::from(["A".to_string()]);

        // Act — terminates (the visited guard breaks the cycle).
        resolve_dependencies(&h, "[B]", &d, &MathLapContext::empty(), &mut visited);

        // Assert — neither side resolves (each depends on an unresolved other).
        assert!(h.lookup("B").is_none());
    }

    #[test]
    fn missing_reference_is_swallowed() {
        // Arrange — A = [Nope] references a channel that is neither base nor a def.
        let h = handle_with(vec![base("X", vec![1.0])]);
        let d = defs(&[("A", "[Nope] + 1")]);
        let mut visited = HashSet::from(["A".to_string()]);

        // Act — does not panic; resolves nothing.
        resolve_dependencies(&h, "[Nope] + 1", &d, &MathLapContext::empty(), &mut visited);

        // Assert
        assert!(h.lookup("Nope").is_none());
    }
}
