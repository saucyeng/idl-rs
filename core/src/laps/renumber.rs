//! Session-wide lap renumbering (port of `cached_session_laps.dart`). The
//! engine's lap detector emits **per-visit** numbering (each `TrackVisit`
//! restarts at 1, `crate::laps::detect_laps`); this assigns the
//! **session-wide** identity `session.json`'s `ignored_lap_numbers` etc.
//! match against — laps from every visit, sorted by start time, renumbered
//! 1-based across the whole session. (This independently confirms ruling
//! R14 item 2's choice to join `laps.track_id` by timestamp containment
//! rather than by `visit.laps[*].lap_number`, since per-visit lap numbers
//! are not the session-wide identity anything else keys off.)

use crate::store::session_json::{LapJson, TrackVisitJson};

/// One session-wide-renumbered lap, its originating visit's `track_id`, and
/// whether it's in `ignored_lap_numbers`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenumberedLap {
    pub lap: LapJson,
    pub track_id: Option<String>,
    pub is_ignored: bool,
}

/// Renumbers every lap across every `track_visits` entry, 1-based, in
/// start-time order. Pure — no I/O.
pub fn renumber_session_laps(track_visits: &[TrackVisitJson], ignored_lap_numbers: &[u32]) -> Vec<RenumberedLap> {
    let mut collected: Vec<(String, LapJson)> = Vec::new();
    for visit in track_visits {
        for lap in &visit.laps {
            collected.push((visit.track_id.clone(), lap.clone()));
        }
    }
    collected.sort_by_key(|(_, lap)| lap.start_timestamp_ms);

    collected
        .into_iter()
        .enumerate()
        .map(|(i, (track_id, src))| {
            let lap_number = (i + 1) as u32;
            RenumberedLap {
                lap: LapJson { lap_number, ..src },
                track_id: Some(track_id),
                is_ignored: ignored_lap_numbers.contains(&lap_number),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lap(number: u32, start_ms: i64) -> LapJson {
        LapJson {
            lap_number: number,
            start_timestamp_ms: start_ms,
            end_timestamp_ms: start_ms + 1000,
            raw_elapsed_ms: 1000,
            lap_time_ms: 1000,
            start_time_secs: 0.0,
            end_time_secs: 1.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }
    }

    fn visit(track_id: &str, laps: Vec<LapJson>) -> TrackVisitJson {
        TrackVisitJson {
            visit_id: "v".to_string(),
            track_id: track_id.to_string(),
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            laps,
        }
    }

    #[test]
    fn renumbers_across_visits_in_start_time_order() {
        // Arrange -- visit B's per-visit lap 1 starts before visit A's, so it
        // must come first in the session-wide numbering.
        let visits = vec![
            visit("track-A", vec![lap(1, 5000), lap(2, 6000)]),
            visit("track-B", vec![lap(1, 1000)]),
        ];

        // Act
        let out = renumber_session_laps(&visits, &[]);

        // Assert
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].lap.lap_number, 1);
        assert_eq!(out[0].track_id.as_deref(), Some("track-B"));
        assert_eq!(out[1].lap.lap_number, 2);
        assert_eq!(out[1].track_id.as_deref(), Some("track-A"));
        assert_eq!(out[2].lap.lap_number, 3);
    }

    #[test]
    fn is_ignored_matches_the_renumbered_session_wide_number() {
        // Arrange
        let visits = vec![visit("t", vec![lap(1, 0), lap(2, 1000), lap(3, 2000)])];

        // Act
        let out = renumber_session_laps(&visits, &[2]);

        // Assert -- the *second* session-wide lap is ignored, regardless of
        // its original per-visit lap_number.
        assert!(!out[0].is_ignored);
        assert!(out[1].is_ignored);
        assert!(!out[2].is_ignored);
    }

    #[test]
    fn no_visits_renumbers_to_an_empty_list() {
        // Arrange / Act
        let out = renumber_session_laps(&[], &[]);

        // Assert
        assert!(out.is_empty());
    }

    #[test]
    fn equal_start_timestamps_keep_their_original_visit_order() {
        // Arrange -- two laps with the same start time, from different
        // visits; `sort_by_key` is stable, so ties resolve by original
        // (visit-then-lap) order rather than being left unspecified the
        // way Dart's `List.sort` comparator leaves them.
        let visits = vec![visit("track-A", vec![lap(1, 1000)]), visit("track-B", vec![lap(1, 1000)])];

        // Act
        let out = renumber_session_laps(&visits, &[]);

        // Assert
        assert_eq!(out[0].track_id.as_deref(), Some("track-A"));
        assert_eq!(out[1].track_id.as_deref(), Some("track-B"));
    }
}
