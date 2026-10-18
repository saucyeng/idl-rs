//! `session.json`'s per-field merge (C4 §6, ruling R88; C1 §6 names the
//! user-owned fields) — the receiving side's pure merge of two full
//! `session.json` documents. Never touches disk, never merges the L2b lap
//! cache (`laps`/`track_visits`/`track_visits_library_hash`/
//! `lap_detector_version`, ruling R83) — those are always kept verbatim from
//! `local` (the receiver's own copy), re-derived locally rather than
//! reconciled from a peer.
//!
//! `local_updated_at_ms`/`peer_updated_at_ms` are each side's `session.json`
//! file's own last-modified time
//! (`store::sync::manifest::SessionJsonEntry::updated_at_ms` — C1 §6's
//! schema carries no such field of its own, confirmed by that entry's own
//! doc comment). [`merge_session_json`] takes them as plain parameters
//! rather than reading either file's metadata itself, to stay pure
//! (CLAUDE.md §4; this lane's brief calls this function out as "Pure").
//! `apply::install` (this module's caller) reads `local`'s file mtime from
//! disk before overwriting it; the peer's is carried down from whichever
//! caller already holds the peer's manifest entry (a deviation from this
//! task's literal two-parameter interface sketch — see this task's report).
//!
//! **A field's "changed" test.** C1 §6 keeps no per-sync base for
//! `session.json` (unlike the workbook's `.sync-base` cache, C2 §7) — there
//! is no prior synced snapshot to diff either side against. This module
//! reads the brief's own "equal to neither side's default" wording
//! literally: a user-owned field's *type default* (`""` for a string,
//! `None` for an option, `[]` for a list — exactly
//! [`crate::store::session_json::empty_session_json`]'s own values) stands
//! in for "never touched," since every field's real-world unset state
//! already coincides with its zero value (C1 §6: "`""` = not set, no null
//! representation"). This resolves every field without needing the brief's
//! "differs from the receiver's current value" no-base fallback — noted
//! here rather than invoked.

use crate::store::session_json::SessionJson;
use crate::video::VideoLink;

/// One user-owned field's per-field merge (C4 §6): "changed" means "not
/// equal to this type's zero value" (see this module's doc comment); a
/// field changed on exactly one side takes that side; changed on both to
/// the same value is not a conflict (either value is returned); changed on
/// both to different values takes the side whose file is newer
/// (`local_newer`) — a per-field application of C4 §6's LWW tiebreak,
/// rather than a whole-document one. A tie (`local_newer` computed as
/// `local_updated_at_ms >= peer_updated_at_ms`) keeps local, a deliberate,
/// deterministic choice C4 §6 does not itself state.
fn merge_field<T: PartialEq + Default>(local: T, peer: T, local_newer: bool) -> T {
    let default = T::default();
    let local_changed = local != default;
    let peer_changed = peer != default;
    match (local_changed, peer_changed) {
        (false, false) => local,
        (true, false) => local,
        (false, true) => peer,
        (true, true) => {
            if local == peer || local_newer {
                local
            } else {
                peer
            }
        }
    }
}

/// C4 §6's `videos` rule (added 2026-10-16): merged by entry, keyed by
/// `blob_sha256`, never as one field. An entry on one side only is kept; an
/// entry on both takes the side whose entry-level `updated_at_ms` is newer,
/// as a whole, and an exact tie keeps local. Tombstones (`unlinked: true`)
/// are entries like any other, which is what stops the union bringing an
/// unlinked video back. Order: local's entries in local order, then the
/// peer-only ones in peer order.
fn merge_videos(local: &[VideoLink], peer: &[VideoLink]) -> Vec<VideoLink> {
    let mut out: Vec<VideoLink> = local
        .iter()
        .map(|l| match peer.iter().find(|p| p.blob_sha256 == l.blob_sha256) {
            Some(p) if p.updated_at_ms > l.updated_at_ms => p.clone(),
            _ => l.clone(),
        })
        .collect();
    for p in peer {
        if !local.iter().any(|l| l.blob_sha256 == p.blob_sha256) {
            out.push(p.clone());
        }
    }
    out
}

/// C4 §6's `session.json` per-field merge. `local`/`peer` are each side's
/// full parsed document; `local_updated_at_ms`/`peer_updated_at_ms` are each
/// side's file's own last-modified time (see this module's doc comment).
/// `schema_version`/`session_id` and the L2b lap cache
/// (`laps`/`track_visits`/`track_visits_library_hash`/`lap_detector_version`
/// — ruling R83) are never merged: the result always carries `local`'s
/// verbatim copy. Every other (user-owned, C1 §6) field merges
/// independently per [`merge_field`].
pub fn merge_session_json(
    local: &SessionJson,
    peer: &SessionJson,
    local_updated_at_ms: i64,
    peer_updated_at_ms: i64,
) -> SessionJson {
    let local_newer = local_updated_at_ms >= peer_updated_at_ms;
    // The bike and what it took from the log move together: mixing one
    // side's bike with the other's entries would be wrong.
    let (bike_id, bike_log) =
        merge_field((local.bike_id.clone(), local.bike_log.clone()), (peer.bike_id.clone(), peer.bike_log.clone()), local_newer);

    SessionJson {
        schema_version: local.schema_version,
        session_id: local.session_id.clone(),
        rider: merge_field(local.rider.clone(), peer.rider.clone(), local_newer),
        bike: merge_field(local.bike.clone(), peer.bike.clone(), local_newer),
        bike_comment: merge_field(local.bike_comment.clone(), peer.bike_comment.clone(), local_newer),
        venue_name: merge_field(local.venue_name.clone(), peer.venue_name.clone(), local_newer),
        event_name: merge_field(local.event_name.clone(), peer.event_name.clone(), local_newer),
        event_session: merge_field(local.event_session.clone(), peer.event_session.clone(), local_newer),
        short_comment: merge_field(local.short_comment.clone(), peer.short_comment.clone(), local_newer),
        long_comment: merge_field(local.long_comment.clone(), peer.long_comment.clone(), local_newer),
        tag: merge_field(local.tag.clone(), peer.tag.clone(), local_newer),
        bike_profile_snapshot: merge_field(local.bike_profile_snapshot.clone(), peer.bike_profile_snapshot.clone(), local_newer),
        lap_gates: merge_field(local.lap_gates.clone(), peer.lap_gates.clone(), local_newer),
        sector_gates: merge_field(local.sector_gates.clone(), peer.sector_gates.clone(), local_newer),
        laps: local.laps.clone(),
        reference_lap_number: merge_field(local.reference_lap_number, peer.reference_lap_number, local_newer),
        ignored_lap_numbers: merge_field(local.ignored_lap_numbers.clone(), peer.ignored_lap_numbers.clone(), local_newer),
        main_lap_number: merge_field(local.main_lap_number, peer.main_lap_number, local_newer),
        overlay_lap_key: merge_field(local.overlay_lap_key.clone(), peer.overlay_lap_key.clone(), local_newer),
        starred_lap_number: merge_field(local.starred_lap_number, peer.starred_lap_number, local_newer),
        track_visits: local.track_visits.clone(),
        track_visits_library_hash: local.track_visits_library_hash.clone(),
        // User-owned via `set_session_start` (C3 §3.3), not part of the
        // never-merged L2b lap cache above — merges like every other C1 §6
        // user field (R194 doesn't rule on sync merge explicitly; this
        // follows the existing `merge_field` precedent for user-set data).
        timestamp_utc_ms: merge_field(local.timestamp_utc_ms, peer.timestamp_utc_ms, local_newer),
        timestamp_source: merge_field(local.timestamp_source, peer.timestamp_source, local_newer),
        lap_detector_version: local.lap_detector_version.clone(),
        bike_id,
        bike_log,
        videos: merge_videos(&local.videos, &peer.videos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::session_json::{empty_session_json, LapJson};

    #[test]
    fn merge_session_json_peer_set_rider_local_set_venue_both_survive() {
        // Arrange
        let mut local = empty_session_json("s1");
        local.venue_name = "Whistler".to_string();
        let mut peer = empty_session_json("s1");
        peer.rider = "Isaac".to_string();

        // Act
        let merged = merge_session_json(&local, &peer, 100, 100);

        // Assert
        assert_eq!(merged.venue_name, "Whistler");
        assert_eq!(merged.rider, "Isaac");
    }

    #[test]
    fn merge_session_json_both_set_rider_peer_newer_the_peers_rider() {
        // Arrange
        let mut local = empty_session_json("s1");
        local.rider = "Local Rider".to_string();
        let mut peer = empty_session_json("s1");
        peer.rider = "Peer Rider".to_string();

        // Act
        let merged = merge_session_json(&local, &peer, 100, 200);

        // Assert
        assert_eq!(merged.rider, "Peer Rider");
    }

    #[test]
    fn merge_session_json_both_set_rider_local_newer_locals_rider() {
        // Arrange — the mirror of the case above, pinning the tiebreak
        // direction rather than just "peer wins when peer is named newer".
        let mut local = empty_session_json("s1");
        local.rider = "Local Rider".to_string();
        let mut peer = empty_session_json("s1");
        peer.rider = "Peer Rider".to_string();

        // Act
        let merged = merge_session_json(&local, &peer, 200, 100);

        // Assert
        assert_eq!(merged.rider, "Local Rider");
    }

    #[test]
    fn merge_session_json_the_peer_carries_different_laps_and_stamps_the_receivers_laps_and_both_stamps_are_untouched() {
        // Arrange
        let mut local = empty_session_json("s1");
        local.lap_detector_version = Some("v-local".to_string());
        local.track_visits_library_hash = Some("hash-local".to_string());
        let mut peer = empty_session_json("s1");
        peer.lap_detector_version = Some("v-peer".to_string());
        peer.track_visits_library_hash = Some("hash-peer".to_string());
        peer.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 1000,
            raw_elapsed_ms: 1000,
            lap_time_ms: 1000,
            start_time_secs: 0.0,
            end_time_secs: 1.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];

        // Act — peer is given a far "newer" timestamp; it must not matter,
        // since the lap cache never merges regardless of recency.
        let merged = merge_session_json(&local, &peer, 100, 999);

        // Assert
        assert_eq!(merged.lap_detector_version, Some("v-local".to_string()));
        assert_eq!(merged.track_visits_library_hash, Some("hash-local".to_string()));
        assert!(merged.laps.is_empty());
    }

    fn video(sha: char, offset_s: f64, unlinked: bool, updated_at_ms: i64) -> VideoLink {
        VideoLink {
            blob_sha256: sha.to_string().repeat(64),
            size_bytes: 1,
            file_name: format!("{sha}.mp4"),
            video_codec: "h264".to_string(),
            width_px: 1920,
            height_px: 1080,
            fps: 30.0,
            duration_s: 60.0,
            offset_s,
            sync_method: crate::video::SyncMethod::Manual,
            unlinked,
            updated_at_ms,
        }
    }

    #[test]
    fn merge_session_json_videos_linked_on_different_sides_both_survive() {
        // Arrange
        let mut local = empty_session_json("s1");
        local.videos = vec![video('a', 0.0, false, 10)];
        let mut peer = empty_session_json("s1");
        peer.videos = vec![video('b', 0.0, false, 20)];

        // Act — the file-level recency must not matter to the entry rule.
        let merged = merge_session_json(&local, &peer, 999, 1);

        // Assert
        assert_eq!(merged.videos, vec![video('a', 0.0, false, 10), video('b', 0.0, false, 20)]);
    }

    #[test]
    fn merge_session_json_the_same_video_on_both_sides_the_newer_entry_wins_whole() {
        // Arrange — a manual offset made on the peer later.
        let mut local = empty_session_json("s1");
        local.videos = vec![video('a', 1.0, false, 10)];
        let mut peer = empty_session_json("s1");
        peer.videos = vec![video('a', 7.5, false, 30)];

        // Act
        let merged = merge_session_json(&local, &peer, 100, 1);

        // Assert
        assert_eq!(merged.videos, vec![video('a', 7.5, false, 30)]);
    }

    #[test]
    fn merge_session_json_the_same_video_with_equal_stamps_keeps_local() {
        // Arrange
        let mut local = empty_session_json("s1");
        local.videos = vec![video('a', 1.0, false, 10)];
        let mut peer = empty_session_json("s1");
        peer.videos = vec![video('a', 2.0, false, 10)];

        // Act
        let merged = merge_session_json(&local, &peer, 1, 100);

        // Assert
        assert_eq!(merged.videos, vec![video('a', 1.0, false, 10)]);
    }

    #[test]
    fn merge_session_json_a_newer_tombstone_unlinks_and_an_older_one_does_not() {
        // Arrange — the peer unlinked `a` after the local edit; local
        // re-linked `b` after the peer unlinked it.
        let mut local = empty_session_json("s1");
        local.videos = vec![video('a', 0.0, false, 10), video('b', 0.0, false, 50)];
        let mut peer = empty_session_json("s1");
        peer.videos = vec![video('a', 0.0, true, 20), video('b', 0.0, true, 40)];

        // Act
        let merged = merge_session_json(&local, &peer, 100, 100);

        // Assert
        assert_eq!(merged.videos, vec![video('a', 0.0, true, 20), video('b', 0.0, false, 50)]);
    }

    #[test]
    fn merge_session_json_the_peers_ignored_lap_numbers_only_adopted() {
        // Arrange
        let local = empty_session_json("s1");
        let mut peer = empty_session_json("s1");
        peer.ignored_lap_numbers = vec![2, 4];

        // Act
        let merged = merge_session_json(&local, &peer, 100, 100);

        // Assert
        assert_eq!(merged.ignored_lap_numbers, vec![2, 4]);
    }
}
