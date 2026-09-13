//! Assembling a [`MathLapContext`] from detected laps.
//!
//! Lap-aware math reads five things off the context: the per-lap bounds, the
//! sectors flattened across laps, the designated main lap, the table baseline
//! row, and the per-lap spans the `[lap]` axis runs over. Filling only the first of those is not a partial answer —
//! it is a *wrong* one. `main_lap_window` treats an absent
//! `main_lap_number` as "no window selected" (ruling R128) and every scalar
//! aggregate then reads the whole session, so a `mean()` a table cell scoped
//! to the main lap silently returns the whole-session mean instead.
//!
//! This module is where the context is built, so the CLI and any other
//! headless caller build it the same way and the rule above is stated once.

use crate::laps::model::Lap;
use crate::math::eval::{LapSpan, MathLapContext};

/// What [`lap_context`] was asked to do about the main lap, and what it
/// could actually do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainLap {
    /// No main lap was designated. Lap-scoped functions have no window, and
    /// a caller that cares must say so to its user — silence here reads as
    /// "the whole session", which is not what a lap-scoped expression asked
    /// for.
    None,
    /// The 1-based lap number designated as main.
    Number(u32),
}

/// Every lap as the `[lap]` axis reads it (C2 §3.6.1, ruling R233) — the one
/// place a [`Lap`] becomes a [`LapSpan`], so the table evaluator, the CLI and
/// [`lap_context`] cannot disagree about what a lap's time is.
///
/// `lap_time_secs` is the recorded `lap_time_ms`, **not** `end - start`: a
/// neutral-zone visit is already subtracted from it, and the two agree on
/// every lap without one, which is what makes the wrong choice hard to see.
pub fn lap_spans(laps: &[Lap]) -> Vec<LapSpan> {
    laps.iter()
        .map(|lap| LapSpan {
            lap_number: lap.lap_number,
            start_secs: lap.start_time_secs,
            end_secs: lap.end_time_secs,
            lap_time_secs: lap.lap_time_ms as f64 / 1000.0,
            sectors: lap.sectors.iter().map(|s| (s.start_time_secs, s.end_time_secs)).collect(),
        })
        .collect()
}

/// Builds the context from `laps`.
///
/// `main_lap` names the designated main lap. Bounds are in session-relative
/// seconds, in lap order; sectors are flattened across laps in arrival
/// order, which is the order `sector_number()` counts in.
///
/// # Errors
/// Returns `Err` with the available lap numbers when `main_lap` names a lap
/// `laps` does not contain — a main lap that does not exist would otherwise
/// become "no main lap", i.e. the whole session.
pub fn lap_context(laps: &[Lap], main_lap: MainLap) -> Result<MathLapContext, Vec<u32>> {
    if let MainLap::Number(n) = main_lap {
        if !laps.iter().any(|lap| lap.lap_number == n) {
            return Err(laps.iter().map(|lap| lap.lap_number).collect());
        }
    }

    let mut ctx = MathLapContext::empty();
    ctx.main_lap_bounds = laps.iter().map(|lap| (lap.start_time_secs, lap.end_time_secs)).collect();
    ctx.main_sectors = laps
        .iter()
        .flat_map(|lap| lap.sectors.iter())
        .map(|sector| (sector.start_time_secs, sector.end_time_secs))
        .collect();
    ctx.main_lap_number = match main_lap {
        MainLap::None => None,
        MainLap::Number(n) => Some(n),
    };
    // The `[lap]` axis's domain (C2 §3.6.1, ruling R233) — every lap, each
    // keeping its own number and its own sectors. Not derivable from
    // `main_lap_bounds`/`main_sectors` above: those flatten the numbering away
    // and flatten the sectors across laps, which is right for the windowing
    // they serve and wrong for a per-lap value.
    ctx.laps = lap_spans(laps);
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laps::model::Sector;

    fn sector(name: &str, start_time_secs: f64, end_time_secs: f64) -> Sector {
        Sector {
            name: name.to_string(),
            start_ms: 0,
            end_ms: 0,
            start_time_secs,
            end_time_secs,
        }
    }

    fn lap(lap_number: u32, start_time_secs: f64, end_time_secs: f64, sectors: Vec<Sector>) -> Lap {
        Lap {
            lap_number,
            start_ms: 0,
            end_ms: 0,
            start_time_secs,
            end_time_secs,
            raw_elapsed_ms: 0,
            lap_time_ms: 0,
            sectors,
            neutral_zone_visits: Vec::new(),
        }
    }

    #[test]
    fn the_context_carries_one_bound_pair_per_lap_in_lap_order() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![]), lap(2, 90.0, 178.5, vec![])];

        // Act
        let ctx = lap_context(&laps, MainLap::None).unwrap();

        // Assert — seconds, session-relative.
        assert_eq!(ctx.main_lap_bounds, vec![(0.0, 90.0), (90.0, 178.5)]);
    }

    #[test]
    fn the_context_flattens_sectors_across_laps_in_arrival_order() {
        // Arrange
        let laps = vec![
            lap(1, 0.0, 90.0, vec![sector("s1", 0.0, 30.0), sector("s2", 30.0, 90.0)]),
            lap(2, 90.0, 180.0, vec![sector("s1", 90.0, 119.0)]),
        ];

        // Act
        let ctx = lap_context(&laps, MainLap::None).unwrap();

        // Assert — `sector_number()` counts in this order.
        assert_eq!(ctx.main_sectors, vec![(0.0, 30.0), (30.0, 90.0), (90.0, 119.0)]);
    }

    #[test]
    fn a_designated_main_lap_reaches_the_context() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![]), lap(2, 90.0, 180.0, vec![])];

        // Act
        let ctx = lap_context(&laps, MainLap::Number(2)).unwrap();

        // Assert
        assert_eq!(ctx.main_lap_number, Some(2));
    }

    #[test]
    fn no_designated_main_lap_leaves_the_number_unset() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![])];

        // Act
        let ctx = lap_context(&laps, MainLap::None).unwrap();

        // Assert — the caller has to tell its user; this is not "lap 1".
        assert!(ctx.main_lap_number.is_none());
    }

    #[test]
    fn a_main_lap_that_does_not_exist_is_an_error_listing_the_ones_that_do() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![]), lap(2, 90.0, 180.0, vec![])];

        // Act
        let available = lap_context(&laps, MainLap::Number(7)).unwrap_err();

        // Assert — silently becoming "no main lap" would mean whole-session
        // answers to a lap-scoped question.
        assert_eq!(available, vec![1, 2]);
    }

    #[test]
    fn a_main_lap_named_against_no_laps_at_all_is_an_error() {
        // Arrange
        let laps: Vec<Lap> = Vec::new();

        // Act
        let available = lap_context(&laps, MainLap::Number(1)).unwrap_err();

        // Assert
        assert!(available.is_empty());
    }

    #[test]
    fn an_empty_lap_set_with_no_main_lap_is_the_empty_context() {
        // Arrange
        let laps: Vec<Lap> = Vec::new();

        // Act
        let ctx = lap_context(&laps, MainLap::None).unwrap();

        // Assert
        assert!(ctx.main_lap_bounds.is_empty());
        assert!(ctx.main_sectors.is_empty());
        assert!(ctx.main_lap_number.is_none());
        assert!(ctx.overlay.is_empty());
    }

    #[test]
    fn the_context_carries_one_lap_span_per_lap_with_its_number_and_its_own_sectors() {
        // Arrange — lap 2 has two sectors, lap 5 has one; the numbers are not
        // positions, which is the whole reason `LapSpan` carries them.
        let laps = vec![
            lap(2, 0.0, 90.0, vec![sector("s1", 0.0, 30.0), sector("s2", 30.0, 90.0)]),
            Lap { lap_time_ms: 86_000, ..lap(5, 90.0, 180.0, vec![sector("s1", 90.0, 119.0)]) },
        ];

        // Act
        let ctx = lap_context(&laps, MainLap::None).unwrap();

        // Assert — per-lap sectors, not the flattened `main_sectors` view.
        assert_eq!(ctx.laps.len(), 2);
        assert_eq!(ctx.laps[0].lap_number, 2);
        assert_eq!(ctx.laps[0].sectors, vec![(0.0, 30.0), (30.0, 90.0)]);
        assert_eq!(ctx.laps[1].lap_number, 5);
        assert_eq!(ctx.laps[1].sectors, vec![(90.0, 119.0)]);
        assert_eq!((ctx.laps[1].start_secs, ctx.laps[1].end_secs), (90.0, 180.0));
        assert_eq!(ctx.laps[1].lap_time_secs, 86.0, "the recorded lap time, not the bounds span");
    }

    #[test]
    fn the_row_lap_is_never_set_from_laps() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![])];

        // Act
        let ctx = lap_context(&laps, MainLap::Number(1)).unwrap();

        // Assert — `row_lap` marks a table row's single-lap scope; a session's
        // laps say nothing about whether an evaluation is in one.
        assert!(ctx.row_lap.is_none());
    }

    #[test]
    fn the_baseline_row_is_never_set_from_laps() {
        // Arrange
        let laps = vec![lap(1, 0.0, 90.0, vec![])];

        // Act
        let ctx = lap_context(&laps, MainLap::Number(1)).unwrap();

        // Assert — `baseline_row` is a table's Main *row*, not a lap; it is
        // the table evaluator's to set, and `main()` is NaN without it.
        assert!(ctx.baseline_row.is_none());
    }
}
