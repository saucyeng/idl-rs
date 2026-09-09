//! Host-variable data for `js` cells (C2 §5.1): the `{length, t, v}` channel
//! shape, the general `channel()` lookup, and the thin `laps`/`session`/
//! `constants` mappings. Every value here is an **in-process Rust shape**
//! consumed by L6's Runtime injection (`postMessage` into the sandboxed
//! iframe) — none of it crosses Tauri IPC (design's "no IPC on the
//! interaction path" principle; C3 has no byte path for a [`HostChannel`]
//! yet, a lead-owned flag batched into ledger R21's contract-amendment
//! list). `host_session`/`host_laps`/`host_constants` have no caller in
//! wave 1 — host-variable injection is L6's job (plan 96–101) — so nothing
//! here threads a [`Session`] into Task 9's `eval_cells`.

use std::collections::HashMap;

use serde::Serialize;

use crate::math::eval::{ChannelLookup, LookupChannel, MathLapContext};
use crate::math::{MathEvalError, MathEvalErrorKind};
use crate::session::Session;

use super::WorkbookDoc;

/// Column-oriented (SoA) channel shape handed to JS host code (C2 §5.1) —
/// matches `@observablehq/plot`'s tabular-data protocol so `Plot.lineY(x,
/// {x: "t", y: "v"})` addresses columns by name. **Not an IPC payload** —
/// it never crosses Tauri IPC as bytes or JSON; decimation to the current
/// tile budget (C2 §5.1) is the host's job in L6, not [`to_host_channel`]'s
/// (lead-owned flag, ledger R21).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostChannel {
    /// Number of values — always `v.len()`; values are never dropped to
    /// match a shorter or absent time axis.
    pub length: usize,
    /// Per-sample recording time, **seconds** since the session's first
    /// sample. `t.len() == length` for any channel with a recorded axis;
    /// `t` is empty for a scalar or table-sourced result (no recorded
    /// axis, L3-R12), and a consumer must not plot such a value against
    /// `t`. A time axis is never synthesized (C1 §3.5 invariant 4).
    pub t: Vec<f64>,
    /// The channel's values, verbatim.
    pub v: Vec<f64>,
    /// This channel's unit (R164) — a `math`-cell definition's own
    /// [`crate::math::units::infer`]red unit, or (for [`channel`]'s direct
    /// name lookup) the C1 §4.1 unit recorded on the base/session channel.
    /// Always present as one of the three states (R154) — never a bare
    /// `Option<String>`, which would recreate the ambiguity this model
    /// exists to remove.
    pub unit: crate::math::units::UnitLabel,
}

/// Converts a resolved channel's `(t_us, v)` pair into the [`HostChannel`]
/// host-variable shape (C2 §5.1) — the one conversion site every host
/// channel goes through. `length` is always `v.len()`; values are never
/// dropped to match a shorter/absent `t_us` (L3-R21 — supersedes the
/// original plan's "empty input → length 0" rule, which would silently
/// zero out every rate-0/scalar/table-sourced definition, G8.1). `t` is
/// `t_us[i] as f64 / 1e6` (**µs → seconds**) over `min(t_us.len(),
/// v.len())` entries, so `t` is empty whenever `t_us` is empty (no
/// recorded axis) — never a synthesized ramp. `unit` (R164) is threaded
/// through verbatim, never computed here — this function has no `Ast` and
/// no `ChannelLookup`, so it cannot infer one; each caller supplies its own
/// (a `math`-cell definition's already-inferred [`crate::math::units::UnitLabel`],
/// or [`channel`]'s direct base-channel lookup).
pub fn to_host_channel(t_us: &[i64], v: &[f64], unit: crate::math::units::UnitLabel) -> HostChannel {
    let n = t_us.len().min(v.len());
    let t = t_us[..n].iter().map(|&us| us as f64 / 1e6).collect();
    HostChannel { length: v.len(), t, v: v.to_vec(), unit }
}

/// General host-variable channel lookup (C2 §5.1's `channel(name, {lap,
/// session})`): resolves `name`, optionally windows to one lap's bounds,
/// and converts via [`to_host_channel`]. `lookup` is the caller's plain
/// `ChannelLookup` — Task 9's orchestrator registers every resolved `math`
/// definition into an overlay over the base session lookup before calling
/// this (mirroring `resolve.rs`'s `OverlayLookup`, L3-R18a/R22 precedent)
/// so `channel()` itself only ever needs one plain lookup per session.
///
/// `other_session`, when `Some((id, other_lookup))`, is the cross-session
/// lookup to use *instead of* `lookup` — this signature has no separate
/// "requested session id" field, so a caller asking for a specific
/// session's channel must already have resolved that session's id and
/// lookup and hand both in here; `None` means "this session only". `name`
/// is looked up exclusively in whichever lookup applies — no fallback
/// between them (an overlay/compare feature scopes to one session at a
/// time, not a merge).
///
/// **The `(id, lookup)` pairing is trusted, not verified here** (ledger
/// R35, `runs/2026-09-03/decisions.md`): `id` itself is never read by this
/// function, only `other_lookup` is — so a caller that wires session A's id
/// to session B's lookup gets session B's data back under session A's
/// label, silently. The caller that resolves `other_session` from a
/// session id string is the one place that pairing can be checked, and it
/// must guarantee the lookup it hands in really does belong to the id it
/// hands in alongside it.
///
/// # Errors
/// [`MathEvalErrorKind::UnknownChannel`] when `name` is not resolvable in
/// the applicable lookup (this is also the outcome when a session-scoped
/// request has no `other_session` data to satisfy it, or `name` simply
/// isn't in that session — both surface as "not found", since this
/// signature carries no separate expected-id to compare against a
/// mismatch). [`MathEvalErrorKind::NoLapContext`] when `lap` is given but
/// `lap_ctx.main_lap_bounds` has no matching lap (empty, or `lap` out of
/// range — both mean "no lap context available for that lap"; the
/// message names how many laps this session actually has (L3-R34b), so
/// "lap 7 of a 3-lap session" does not read as a "no laps at all" lie).
// TODO(idl0): land the `(id, lookup)` pairing check with the wave-2 caller
// (L6) that first threads a real `Session`/session map in here —
// `ChannelLookup` has no way to report its own session id today, so the
// caller must compare ids it already holds itself rather than asking the
// trait to confirm (R35).
pub fn channel(
    lookup: &dyn ChannelLookup,
    name: &str,
    lap: Option<u32>,
    lap_ctx: &MathLapContext,
    other_session: Option<(&str, &dyn ChannelLookup)>,
) -> Result<HostChannel, MathEvalError> {
    let unit_lookup: &dyn ChannelLookup = match other_session {
        Some((_, other_lookup)) => other_lookup,
        None => lookup,
    };
    let resolved = match other_session {
        Some((_, other_lookup)) => other_lookup.lookup(name),
        None => lookup.lookup(name),
    };
    let LookupChannel { samples, t_us, .. } = resolved.ok_or_else(|| {
        MathEvalError::new(MathEvalErrorKind::UnknownChannel, format!("Channel '[{name}]' not in this session"))
    })?;
    // A direct `[Name]` lookup, not an expression — reuse `infer`'s own
    // ChannelRef rule (parse/None handling included) rather than
    // duplicating it here.
    let unit = {
        let ast = crate::math::parse::Ast::ChannelRef(name.to_string());
        let (unit, _) = crate::math::units::infer(&ast, unit_lookup);
        crate::math::units::UnitLabel::from(&unit)
    };

    let Some(lap_number) = lap else {
        return Ok(to_host_channel(&t_us, &samples, unit));
    };

    let window = (lap_number as usize).checked_sub(1).and_then(|i| lap_ctx.main_lap_bounds.get(i));
    let &(start_s, end_s) = window.ok_or_else(|| {
        // L3-R34b: name the recorded lap count so "lap 7 of a 3-lap
        // session" reads differently from "no laps recorded at all".
        let lap_count = lap_ctx.main_lap_bounds.len();
        let laps_recorded =
            if lap_count == 0 { "no laps recorded".to_string() } else { format!("{lap_count} laps recorded") };
        MathEvalError::new(
            MathEvalErrorKind::NoLapContext,
            format!(
                "channel(\"{name}\", lap: {lap_number}): no lap {lap_number} in this session's lap table ({laps_recorded})"
            ),
        )
    })?;

    // Seconds → µs, once, at this one conversion site — inclusive at both
    // ends, matching `SessionHandle::slice_by_time` (`handle.rs:664-667`).
    let start_us = (start_s * 1e6).round() as i64;
    let end_us = (end_s * 1e6).round() as i64;
    let (win_t_us, win_v): (Vec<i64>, Vec<f64>) = t_us
        .iter()
        .zip(samples.iter())
        .filter(|(&t, _)| t >= start_us && t <= end_us)
        .map(|(&t, &v)| (t, v))
        .unzip();
    Ok(to_host_channel(&win_t_us, &win_v, unit))
}

/// One lap of the active session's lap table, as exposed to JS host code
/// (C2 §5.1). Field names are camelCase (`startT`/`endT`) — a deliberate,
/// host-variable-only exception to C3 §1's snake_case IPC rule (L3-R23):
/// these are Observable module-scope bindings, not IPC payload fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostLap {
    pub number: u32,
    /// Lap start, seconds (session-relative, `MathLapContext`'s own axis).
    pub start_t: f64,
    /// Lap end, seconds.
    pub end_t: f64,
}

/// Maps `lap_ctx.main_lap_bounds` to the JS `laps` host variable (C2
/// §5.1). Numbers laps `index + 1` — the engine's own 1-based index into
/// `main_lap_bounds` (`eval.rs:567`'s `.get(n - 1)` convention) — equal to
/// `Lap.lap_number` only when the caller built `main_lap_bounds` from an
/// unfiltered ascending lap list (a session with `ignored_lap_numbers` can
/// make the two diverge, L3-R23).
pub fn host_laps(lap_ctx: &MathLapContext) -> Vec<HostLap> {
    lap_ctx
        .main_lap_bounds
        .iter()
        .enumerate()
        .map(|(i, &(start_t, end_t))| HostLap { number: (i + 1) as u32, start_t, end_t })
        .collect()
}

/// Active session metadata, as exposed to JS host code (C2 §5.1). Field
/// names are camelCase (`timestampUtcMs`) for the same reason as
/// [`HostLap`] (L3-R23).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostSession {
    pub id: String,
    pub name: Option<String>,
    /// Session start, UTC milliseconds since the Unix epoch (`0` = unknown,
    /// [`Session::timestamp_utc_ms`]'s own sentinel convention).
    pub timestamp_utc_ms: i64,
}

/// Maps a [`Session`] to the JS `session` host variable (C2 §5.1). `name`
/// is always `None` — `Session` (`session/mod.rs:341-362`) has no name
/// field, and neither does `SessionJson`, C3's `SessionSummary`, or
/// `SessionDetail` (Q2, `runs/2026-09-03/decisions.md` R21). No display
/// string is synthesized here.
// TODO(idl0): source `HostSession.name` once Isaac rules on Q2's naming
// convention (rider/bike/venue_name/event_name/event_session precedence).
pub fn host_session(session: &Session) -> HostSession {
    HostSession { id: session.session_id.clone(), name: None, timestamp_utc_ms: session.timestamp_utc_ms }
}

/// Maps `doc.constants` (Task 6's merged flat table, L3-R16) to the JS
/// `constants` host variable (C2 §5.1) verbatim.
pub fn host_constants(doc: &WorkbookDoc) -> HashMap<String, f64> {
    doc.constants.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SourceFormat;

    struct MapLookup(HashMap<&'static str, (Vec<f64>, f64, Vec<i64>)>);
    impl ChannelLookup for MapLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            self.0.get(name).map(|(samples, rate, t_us)| LookupChannel {
                samples: samples.clone().into(),
                sample_rate_hz: *rate,
                t_us: t_us.clone().into(),
            })
        }
    }

    fn no_laps() -> MathLapContext {
        MathLapContext::empty()
    }

    // ---- Step 1: to_host_channel (L3-R21) ----

    #[test]
    fn recorded_axis_t_us_converts_microseconds_to_seconds() {
        // Arrange
        let t_us = [0i64, 1_000_000, 2_500_000];
        let v = [10.0, 20.0, 30.0];

        // Act
        let got = to_host_channel(&t_us, &v, crate::math::units::UnitLabel::Dimensionless);

        // Assert
        assert_eq!(got.t, vec![0.0, 1.0, 2.5]);
        assert_eq!(got.length, 3);
        assert_eq!(got.v, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn empty_t_us_three_values_length_3_t_empty_v_preserved() {
        // Arrange — L3-R12: a scalar/table-sourced result has no axis.
        let t_us: [i64; 0] = [];
        let v = [1.0, 2.0, 3.0];

        // Act
        let got = to_host_channel(&t_us, &v, crate::math::units::UnitLabel::Dimensionless);

        // Assert
        assert_eq!(got.length, 3);
        assert!(got.t.is_empty());
        assert_eq!(got.v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn both_empty_length_0() {
        // Arrange
        let t_us: [i64; 0] = [];
        let v: [f64; 0] = [];

        // Act
        let got = to_host_channel(&t_us, &v, crate::math::units::UnitLabel::Dimensionless);

        // Assert
        assert_eq!(got.length, 0);
        assert!(got.t.is_empty());
        assert!(got.v.is_empty());
    }

    // ---- Step 2: channel() (L3-R22) ----

    #[test]
    fn channel_with_no_lap_or_session_same_as_lookup_converted_via_to_host_channel() {
        // Arrange
        let lookup = MapLookup(HashMap::from([("X", (vec![1.0, 2.0], 10.0, vec![0, 100_000]))]));

        // Act
        let got = channel(&lookup, "X", None, &no_laps(), None).unwrap();

        // Assert
        assert_eq!(
            got,
            to_host_channel(
                &[0, 100_000],
                &[1.0, 2.0],
                crate::math::units::UnitLabel::Unknown { reason: "no unit recorded for this channel".to_string() }
            )
        );
    }

    #[test]
    fn channel_with_lap_windows_t_and_v_to_that_laps_bounds_inclusive_both_ends() {
        // Arrange — 10 Hz, samples 0..10 (sample i at i/10 s); lap 2 = [0.2, 0.5]s.
        let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t_us: Vec<i64> = (0..10).map(|i| i * 100_000).collect();
        let lookup = MapLookup(HashMap::from([("X", (samples, 10.0, t_us))]));
        let lap_ctx = MathLapContext { main_lap_bounds: vec![(0.0, 0.1), (0.2, 0.5)], ..MathLapContext::empty() };

        // Act — window [0.2, 0.5]s → indices 2..=5 (µs boundaries inclusive).
        let got = channel(&lookup, "X", Some(2), &lap_ctx, None).unwrap();

        // Assert
        assert_eq!(got.v, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(got.t, vec![0.2, 0.3, 0.4, 0.5]);
        assert_eq!(got.length, 4);
    }

    #[test]
    fn channel_nonexistent_name_unknown_channel() {
        // Arrange
        let lookup = MapLookup(HashMap::new());

        // Act
        let err = channel(&lookup, "nonexistent", None, &no_laps(), None).unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn channel_session_scoped_request_with_no_other_session_data_unknown_channel() {
        // Arrange — "requesting" a cross-session channel is only expressible
        // by handing `other_session`'s lookup in; with `other_session: None`
        // there is no session data to satisfy the request, so the plain
        // "not found in the lookup actually available" path applies (L3-R22
        // doc comment: this signature carries no separate expected-id).
        let lookup = MapLookup(HashMap::from([("X", (vec![1.0], 10.0, vec![0]))]));

        // Act — "X" lives only in the (unavailable) other session.
        let err = channel(&lookup, "OtherSessionOnlyChannel", None, &no_laps(), None).unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::UnknownChannel);
    }

    #[test]
    fn channel_lap_requested_with_empty_main_lap_bounds_no_lap_context_message_says_no_laps_recorded() {
        // Arrange — L3-R34b: an empty lap table names itself as such, not
        // as "0 laps recorded".
        let lookup = MapLookup(HashMap::from([("X", (vec![1.0, 2.0], 10.0, vec![0, 100_000]))]));

        // Act
        let err = channel(&lookup, "X", Some(1), &no_laps(), None).unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::NoLapContext);
        assert_eq!(
            err.message,
            "channel(\"X\", lap: 1): no lap 1 in this session's lap table (no laps recorded)"
        );
    }

    #[test]
    fn channel_lap_out_of_range_of_a_non_empty_lap_table_no_lap_context_message_names_the_recorded_lap_count() {
        // Arrange — L3-R34b: 3 laps recorded, lap 7 requested.
        let lookup = MapLookup(HashMap::from([("X", (vec![1.0, 2.0], 10.0, vec![0, 100_000]))]));
        let lap_ctx = MathLapContext {
            main_lap_bounds: vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
            ..MathLapContext::empty()
        };

        // Act
        let err = channel(&lookup, "X", Some(7), &lap_ctx, None).unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::NoLapContext);
        assert_eq!(
            err.message,
            "channel(\"X\", lap: 7): no lap 7 in this session's lap table (3 laps recorded)"
        );
    }

    #[test]
    fn channel_seconds_to_microseconds_conversion_site_rounds_and_is_inclusive_at_both_boundary_samples() {
        // Arrange — a boundary sample lands exactly at the (rounded) lap-end µs.
        let t_us = vec![0i64, 500_000, 1_000_000, 1_500_000];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let lookup = MapLookup(HashMap::from([("X", (v, 2.0, t_us))]));
        let lap_ctx = MathLapContext { main_lap_bounds: vec![(0.5, 1.0)], ..MathLapContext::empty() };

        // Act — [0.5s, 1.0s] → [500_000us, 1_000_000us] inclusive.
        let got = channel(&lookup, "X", Some(1), &lap_ctx, None).unwrap();

        // Assert
        assert_eq!(got.v, vec![20.0, 30.0]);
    }

    // ---- Step 3: host_laps / host_session / host_constants (L3-R23) ----

    #[test]
    fn host_laps_numbers_from_1_field_for_field_against_main_lap_bounds() {
        // Arrange
        let lap_ctx =
            MathLapContext { main_lap_bounds: vec![(0.0, 10.0), (10.0, 22.5)], ..MathLapContext::empty() };

        // Act
        let got = host_laps(&lap_ctx);

        // Assert
        assert_eq!(
            got,
            vec![
                HostLap { number: 1, start_t: 0.0, end_t: 10.0 },
                HostLap { number: 2, start_t: 10.0, end_t: 22.5 },
            ]
        );
    }

    #[test]
    fn host_session_maps_id_and_timestamp_name_is_always_none() {
        // Arrange
        let session = Session {
            session_id: "abc123".to_string(),
            device_id: None,
            timestamp_utc_ms: 1_700_000_000_000,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: "0".repeat(64),
            channels: Vec::new(),
        };

        // Act
        let got = host_session(&session);

        // Assert
        assert_eq!(got, HostSession { id: "abc123".to_string(), name: None, timestamp_utc_ms: 1_700_000_000_000 });
    }

    #[test]
    fn host_constants_reads_workbook_doc_constants_verbatim() {
        // Arrange
        let doc = WorkbookDoc {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Fork tuning".to_string(),
            constants_raw: HashMap::new(),
            units_pref: crate::workbook::v3::UnitsPref::Si,
            version: 3,
            cells: Vec::new(),
            trailing_prose: None,
            const_lines: Vec::new(),
            defs: Vec::new(),
            constants: HashMap::from([("g".to_string(), 9.80665), ("rider_mass_kg".to_string(), 82.0)]),
            front_matter_unknown: std::collections::BTreeMap::new(),
        };

        // Act
        let got = host_constants(&doc);

        // Assert
        assert_eq!(got, doc.constants);
    }
}
