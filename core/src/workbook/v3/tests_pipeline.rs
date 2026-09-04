//! Integration-level parity tests for the v3 workbook pipeline (L3 Task 15,
//! the lane's done-criteria proof task; L3-R36 replaces the original
//! v2-vs-migration parity gate with a v3-vs-direct-evaluator one, since
//! Task 13 (`migrate_workbook`) is cut, R30). Two independent proofs:
//! Step 1 that the cell pipeline hands back the evaluator's own numbers,
//! unaltered; Step 2 that a tile built from a real evaluated result matches
//! `chart_decimation::decimate_channel` called directly on the same array.
//! Neither proof touches migration — see R36's mapping of the lane's
//! Done-when (1) onto this file.

use super::*;
use crate::math::eval::{evaluate_with_constants, MathLapContext};
use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

/// Two channels, both with **explicit, non-uniform** `t_us` — the recorded
/// axis must survive the pipeline unchanged, so a synthesized uniform axis
/// would hide exactly the bug this test exists to catch.
fn synthetic_session() -> SessionHandle {
    let chan_a = ChannelInput {
        channel_id: "ChanA".to_string(),
        sample_rate_hz: 100.0,
        samples: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        t_us: vec![0, 9_000, 21_000, 34_000, 50_000],
        source_kind: "test".to_string(),
    };
    let chan_b = ChannelInput {
        channel_id: "ChanB".to_string(),
        sample_rate_hz: 50.0,
        samples: vec![10.0, 20.0, 30.0],
        t_us: vec![0, 18_000, 41_000],
        source_kind: "test".to_string(),
    };
    let meta = SessionMetaInput {
        session_id: String::new(),
        device_id: None,
        timestamp_utc_ms: 0,
        config_checksum: None,
    };
    SessionHandle::from_channels(meta, vec![chan_a, chan_b])
}

/// One v3 document, four `math`-cell definitions (per L3-R41 minus its
/// migration-only fixture item, G15.1):
/// `a` — plain arithmetic over `ChanA`; `b` — a `[Name]` cross-reference to
/// `a`; `c` — a front-matter constant, the v3-only case (SPEC:1793: idl0
/// inlines user constants as numeric literals, so no v2 arm exists to
/// compare against); `avg` — a scalar (`mean(...)`), the axis-less case
/// (L3-R12/L3-R21).
const WORKBOOK_MD: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Test\nversion: 3\nconstants: { rider_mass_kg: 82 }\n---\n\n```math id=aaaaaaaa\na = [ChanA] * 2\nb = [a] + 1\nc = [ChanA] * rider_mass_kg\navg = mean([ChanA])\n```\n";

/// v3 cell pipeline — four definitions over a synthetic session — every
/// value equals a direct `evaluate` on the inlined expression, bit-for-bit.
///
/// Path A routes through the real pipeline: `parse_workbook` →
/// `eval_cells`. Path B calls `math::eval::evaluate_with_constants` directly
/// against the bare `SessionHandle`, with any `[Name]` cross-reference
/// (definition `b`'s `[a]`) **textually inlined** — calling
/// `resolve_workbook_defs` for path B would just be path A again, so
/// inlining is what makes path B an independent proof (L3-R36(ii)).
///
/// Scope: this proves *the v3 cell pipeline routes to the evaluator without
/// altering values* — identical `f64` samples out of the same evaluator on
/// the same inputs. It does not re-prove *the evaluator produces idl0's
/// numbers*; that is `core/src/math/tests_parity.rs`'s job (R36(i)). The
/// constants table and the recorded `t_us` axis are new information here,
/// not parity regressions.
#[test]
fn workbook_v3_pipeline_values_match_a_direct_evaluate_bit_for_bit() {
    // Arrange
    let handle = synthetic_session();
    let lap_ctx = MathLapContext::empty();

    // Act — Path A: the real pipeline.
    let (doc, structural) = parse_workbook(WORKBOOK_MD).expect("front matter is well-formed");
    assert!(structural.is_empty(), "fixture must parse with no structural errors: {structural:?}");
    let cells = eval_cells(&doc, &structural, &handle, &lap_ctx);
    assert_eq!(cells.len(), 1);
    let defs = &cells[0].defs;
    assert_eq!(defs.len(), 4);
    let by_name = |name: &str| -> &CellDefResult { defs.iter().find(|d| d.name == name).unwrap() };

    // Act — Path B: the bare evaluator, `[a]` textually inlined into `b`'s
    // expression so this path never calls `resolve_workbook_defs` itself.
    let mut constants = std::collections::HashMap::new();
    constants.insert("rider_mass_kg".to_string(), 82.0);
    let direct_a = evaluate_with_constants("[ChanA] * 2", &constants, &handle, &lap_ctx).unwrap();
    let direct_b = evaluate_with_constants("([ChanA] * 2) + 1", &constants, &handle, &lap_ctx).unwrap();
    let direct_c = evaluate_with_constants("[ChanA] * rider_mass_kg", &constants, &handle, &lap_ctx).unwrap();
    let direct_avg = evaluate_with_constants("mean([ChanA])", &constants, &handle, &lap_ctx).unwrap();

    // Assert — per definition: bit-for-bit samples, `length == v.len()`,
    // `t.len() ∈ {0, length}` with `0` only for the scalar, and the µs→s
    // conversion (`t_us` in µs, `t` in seconds) is the only transform.
    for (name, direct) in [("a", &direct_a), ("b", &direct_b), ("c", &direct_c), ("avg", &direct_avg)] {
        let value = by_name(name).value.as_ref().unwrap_or_else(|| panic!("{name} must evaluate to a value"));

        assert_eq!(value.v.len(), direct.samples.len(), "{name}: sample count");
        for i in 0..value.v.len() {
            if value.v[i].is_nan() {
                assert!(direct.samples[i].is_nan(), "{name}[{i}]: expected NaN to match NaN");
            } else {
                assert_eq!(value.v[i], direct.samples[i], "{name}[{i}]: bit-for-bit sample mismatch");
            }
        }
        assert_eq!(value.length, value.v.len(), "{name}: length == v.len() (L3-R21)");

        if name == "avg" {
            assert_eq!(value.t.len(), 0, "avg: scalar result must have an empty axis (L3-R12/L3-R21)");
        } else {
            assert_eq!(value.t.len(), value.length, "{name}: channel result must carry a full axis");
            for i in 0..value.t.len() {
                assert_eq!(value.t[i], direct.t_us[i] as f64 / 1e6, "{name}: t[{i}] is t_us[{i}] in µs / 1e6");
            }
        }
    }
}

/// tile bytes — built from an evaluated math-cell result — sample region
/// equals `decimate_channel` cast to `f32`, element-for-element.
///
/// Takes definition `a`'s evaluated `(v, t_us)` (never the scalar `avg`; it
/// has no `t_us` and a tile has no axis to write), runs it through
/// `tile::build_tile_bytes` as Task 10 landed it, decodes the sample region
/// back out as `f32` LE, and compares it to `decimate_channel` called
/// directly on the same array. This is the integration-level pass Task 10
/// Step 4 promised at the unit level, on a real evaluated result rather than
/// a hand-built array.
#[test]
fn tile_built_from_an_evaluated_math_cell_matches_decimate_channel_directly() {
    // Arrange
    let handle = synthetic_session();
    let lap_ctx = MathLapContext::empty();
    let (doc, structural) = parse_workbook(WORKBOOK_MD).expect("front matter is well-formed");
    let cells = eval_cells(&doc, &structural, &handle, &lap_ctx);
    let a = cells[0].defs.iter().find(|d| d.name == "a").unwrap().value.as_ref().unwrap();
    let tier = 0u32;
    let tile_index = 0u32;
    let column_count = 4u32;

    // Act
    let a_t_us: Vec<i64> = a.t.iter().map(|&s| (s * 1e6).round() as i64).collect();
    let bytes = crate::tile::build_tile_bytes(&a.v, &a_t_us, tier, tile_index, column_count);
    let want = crate::chart_decimation::decimate_channel(&a.v, tier, tile_index);

    // Assert — decode the sample region (32-byte header, then
    // `sample_count * 8` bytes of interleaved f32 [min, max] pairs) and
    // compare element-for-element against `decimate_channel` cast to f32.
    let sample_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(sample_count, want.len() / 2, "sample_count must match decimate_channel's bucket count");
    let sample_off = 32usize;
    for i in 0..want.len() {
        let base = sample_off + i * 4;
        let got = f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
        let want_f32 = want[i] as f32;
        if want_f32.is_nan() {
            assert!(got.is_nan(), "sample region[{i}]: expected NaN to match NaN");
        } else {
            assert_eq!(got, want_f32, "sample region[{i}]: mismatch against decimate_channel");
        }
    }
}
