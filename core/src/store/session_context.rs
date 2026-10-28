//! A stored session's lap context for one window (C1 §6.1), from
//! `session.json` — the part of `idl-rs-tauri`'s `load_window_context_from`
//! that is not about the wire (session-sets spec §6.2), so the app's
//! per-window path, the session-set runner and the CLI build the same
//! [`MathLapContext`] for the same window.

use std::path::Path;

use crate::math::eval::{LapSpan, MathLapContext};
use crate::store::parquet::{read_time_axis_bounds, ParquetStoreError};
use crate::store::session_json::{LapJson, SessionJson};

/// One [`LapJson`] as the `[lap]` axis reads it (C2 §3.6.1, ruling R233).
/// `lap_time_secs` comes from the recorded `lap_time_ms`, **not** from the
/// bounds: a neutral-zone visit is already subtracted from `lap_time_ms`, so
/// the two differ on exactly the laps where it matters.
pub fn lap_span(lap: &LapJson) -> LapSpan {
    LapSpan {
        lap_number: lap.lap_number,
        start_secs: lap.start_time_secs,
        end_secs: lap.end_time_secs,
        lap_time_secs: lap.lap_time_ms as f64 / 1000.0,
        sectors: lap.sectors.iter().map(|s| (s.start_time_secs, s.end_time_secs)).collect(),
    }
}

/// The laps of `laps` that a window `[start, end)` (seconds) covers, in lap
/// order. Overlap, not containment, and half-open on both sides: a lap counts
/// when it shares any time with the window.
pub fn laps_in_window(laps: &[LapJson], (start, end): (f64, f64)) -> Vec<LapSpan> {
    laps.iter().filter(|l| l.start_time_secs < end && l.end_time_secs > start).map(lap_span).collect()
}

/// The [`MathLapContext`] of one resolved window `bounds` (seconds, session
/// relative) over a session whose `session.json` is `doc` (`None` when absent
/// or unreadable — a session with no laps).
///
/// `main_lap_bounds` is the one window and `main_lap_number` is `Some(1)`: a
/// window is *the* selected span, not "the Nth lap". `laps` is every lap the
/// window covers, each keeping its number; `session_laps` is every lap of the
/// session, so the lap functions number against the whole session whatever
/// the window (memory spec §7 Q1). Never a row context.
pub fn window_lap_context(doc: Option<&SessionJson>, bounds: (f64, f64)) -> MathLapContext {
    let laps = doc.map(|d| laps_in_window(&d.laps, bounds)).unwrap_or_default();
    MathLapContext {
        main_lap_bounds: vec![bounds],
        main_sectors: Vec::new(),
        main_lap_number: Some(1),
        overlay: Vec::new(),
        baseline_row: None,
        laps,
        row_lap: None,
        session_laps: Some(doc.map(|d| d.laps.iter().map(lap_span).collect()).unwrap_or_default()),
    }
}

/// The whole recorded span of the session stored in `session_dir`, seconds:
/// `[first sample, last sample + 1 µs)`, from `data.parquet`'s `t` column
/// statistics (a footer read, no decode). `(0, 0)` for a file with no rows.
///
/// # Errors
/// The file cannot be opened or its footer read.
pub fn whole_session_bounds(session_dir: &Path) -> Result<(f64, f64), ParquetStoreError> {
    Ok(match read_time_axis_bounds(session_dir)? {
        Some((start, last)) => (start as f64 / 1e6, (last + 1) as f64 / 1e6),
        None => (0.0, 0.0),
    })
}

/// [`window_lap_context`] for a whole-session window (C1 §6.1 `span =
/// session`) when `data.parquet` is not available: every lap of `doc` is in
/// the window. Used for a member whose demanded definitions read no channel
/// (lap and sector statistics come from `session.json` alone, session-sets
/// spec §4.2).
pub fn whole_session_context_without_data(doc: Option<&SessionJson>) -> MathLapContext {
    window_lap_context(doc, (f64::NEG_INFINITY, f64::INFINITY))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lap(n: u32, start: f64, end: f64) -> LapJson {
        LapJson {
            lap_number: n,
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            raw_elapsed_ms: 0,
            lap_time_ms: ((end - start) * 1000.0) as i64,
            start_time_secs: start,
            end_time_secs: end,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }
    }

    #[test]
    fn laps_in_window_a_lap_ending_at_the_window_start_is_not_in_it() {
        // Act
        let got = laps_in_window(&[lap(1, 0.0, 10.0), lap(2, 10.0, 20.0)], (10.0, 30.0));

        // Assert
        assert_eq!(got.iter().map(|l| l.lap_number).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn whole_session_context_without_data_every_lap_and_never_a_row() {
        // Arrange
        let mut doc = crate::store::session_json::empty_session_json("s");
        doc.laps = vec![lap(1, 5.0, 10.0), lap(2, 10.0, 20.0)];

        // Act
        let ctx = whole_session_context_without_data(Some(&doc));

        // Assert
        assert_eq!(ctx.laps.len(), 2);
        assert_eq!(ctx.session_laps.as_ref().map(Vec::len), Some(2));
        assert_eq!(ctx.row_lap, None);
    }
}
