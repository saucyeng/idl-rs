//! The core operations behind `session list`'s filters and `session
//! set-meta` (ruling R229, via R230's "no CLI-only logic").
//!
//! Filtering and the metadata patch are pure functions over already-read
//! values, so they are unit-testable without a data directory; only
//! [`set_session_meta`] touches the filesystem, and it does so through the
//! same read-hash-write path [`crate::store::session_json::set_session_start`]
//! uses, so two writers racing the same `session.json` still lose one write
//! loudly rather than silently.

use std::path::Path;

use crate::store::atomic::sha256_hex;
use crate::store::catalog_read::{SessionDetail, SessionSummary};
use crate::store::session_json::{
    parse_session_json, write_session_json, SessionJson, SessionJsonError, SessionJsonErrorKind,
};

/// `session list`'s filters. Every field is "no opinion" when `None`; a
/// filter with every field `None` keeps every row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionFilter {
    /// Matches [`SessionSummary::venue_name`], case-insensitively, whole.
    pub venue: Option<String>,
    /// Matches [`SessionSummary::tag`], case-insensitively, whole.
    pub tag: Option<String>,
    /// Keeps rows whose `timestamp_utc_ms` is at or after this, Unix epoch
    /// milliseconds. A row with `timestamp_utc_ms == 0` ("unknown", C1 §3.1)
    /// is dropped by any time bound rather than counted as the epoch.
    pub since_utc_ms: Option<i64>,
    /// Keeps rows whose `timestamp_utc_ms` is at or before this, Unix epoch
    /// milliseconds. Same treatment of `0` as [`SessionFilter::since_utc_ms`].
    pub until_utc_ms: Option<i64>,
}

impl SessionFilter {
    /// `true` when no field is set, so the caller can skip the pass.
    pub fn is_empty(&self) -> bool {
        *self == SessionFilter::default()
    }

    /// Whether one row survives this filter.
    pub fn matches(&self, row: &SessionSummary) -> bool {
        if let Some(venue) = &self.venue {
            if !row.venue_name.eq_ignore_ascii_case(venue) {
                return false;
            }
        }
        if let Some(tag) = &self.tag {
            if !row.tag.eq_ignore_ascii_case(tag) {
                return false;
            }
        }
        if self.since_utc_ms.is_some() || self.until_utc_ms.is_some() {
            // `0` means "start unknown" (C1 §3.1), not 1970: a time-bounded
            // query must not sweep every undated session into its earliest
            // bucket.
            if row.timestamp_utc_ms == 0 {
                return false;
            }
            if let Some(since) = self.since_utc_ms {
                if row.timestamp_utc_ms < since {
                    return false;
                }
            }
            if let Some(until) = self.until_utc_ms {
                if row.timestamp_utc_ms > until {
                    return false;
                }
            }
        }
        true
    }
}

/// Every row that survives `filter`, in the order given.
pub fn filter_sessions(rows: &[SessionSummary], filter: &SessionFilter) -> Vec<SessionSummary> {
    rows.iter().filter(|row| filter.matches(row)).cloned().collect()
}

/// Whether a session visited `track_id`.
///
/// `session list --track` needs this, and a [`SessionSummary`] does not
/// carry track visits — the caller reads the [`SessionDetail`] per candidate
/// row and applies this, which is why the flag costs a file read per session
/// and the other filters do not.
pub fn visited_track(detail: &SessionDetail, track_id: &str) -> bool {
    detail.track_visits.iter().any(|visit| visit.track_id == track_id)
}

/// `session set-meta`'s patch: the descriptive `session.json` fields a user
/// can set from the command line. `None` leaves the field alone; `Some("")`
/// clears it, which is how C1 §6 spells "not set".
///
/// There is deliberately no `track` field. A session's track association is
/// detected from its GPS (`track detect`), never typed in, so a
/// user-supplied track string would be a second source of truth for
/// something the engine already computes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionMetaPatch {
    pub venue_name: Option<String>,
    pub rider: Option<String>,
    pub bike: Option<String>,
    pub event_name: Option<String>,
    pub event_session: Option<String>,
    /// `session.json`'s `long_comment` — the CLI spells it `--notes`.
    pub long_comment: Option<String>,
    pub tag: Option<String>,
}

impl SessionMetaPatch {
    /// `true` when the patch sets nothing, so the caller can refuse a
    /// `set-meta` that would be a no-op write.
    pub fn is_empty(&self) -> bool {
        *self == SessionMetaPatch::default()
    }
}

/// Applies `patch` to `doc`, returning the names of the fields it actually
/// changed, in a stable order.
///
/// The returned list is empty when every set field already held the given
/// value — which is what makes `session set-meta` idempotent (R230 item 3):
/// running it twice writes once.
pub fn apply_session_meta(doc: &mut SessionJson, patch: &SessionMetaPatch) -> Vec<&'static str> {
    let mut changed = Vec::new();
    let mut set = |field: &'static str, target: &mut String, value: &Option<String>| {
        if let Some(value) = value {
            if target != value {
                *target = value.clone();
                changed.push(field);
            }
        }
    };

    set("venue_name", &mut doc.venue_name, &patch.venue_name);
    set("rider", &mut doc.rider, &patch.rider);
    set("bike", &mut doc.bike, &patch.bike);
    set("event_name", &mut doc.event_name, &patch.event_name);
    set("event_session", &mut doc.event_session, &patch.event_session);
    set("long_comment", &mut doc.long_comment, &patch.long_comment);
    set("tag", &mut doc.tag, &patch.tag);

    changed
}

/// What [`set_session_meta`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionMetaReport {
    /// The document as it now stands on disk (or as it would stand, under
    /// `dry_run`).
    pub doc: SessionJson,
    /// The fields this call changed; empty when the values were already set.
    pub changed: Vec<&'static str>,
    /// `true` when nothing was written, either because `dry_run` was set or
    /// because `changed` was empty.
    pub unchanged: bool,
}

/// Reads `<data_root>/sessions/<session_id>/session.json`, applies `patch`,
/// and writes it back atomically using the hash of the bytes it just read as
/// the optimistic-concurrency `based_on_hash`.
///
/// Under `dry_run` the patched document is returned and nothing is written.
/// A patch that changes no field writes nothing either way, so calling this
/// twice with the same values touches the file once.
///
/// # Errors
/// [`SessionJsonErrorKind::InvalidArgument`] when `patch` sets no field;
/// `Io`/`Parse`/`UnsupportedVersion` propagate from the read and the write.
pub fn set_session_meta(
    data_root: &Path,
    session_id: &str,
    patch: &SessionMetaPatch,
    dry_run: bool,
) -> Result<SessionMetaReport, SessionJsonError> {
    if patch.is_empty() {
        return Err(SessionJsonError {
            kind: SessionJsonErrorKind::InvalidArgument,
            message: "set-meta needs at least one field to set".to_string(),
        });
    }

    let target = data_root.join("sessions").join(session_id).join("session.json");
    let raw = std::fs::read(&target)
        .map_err(|e| SessionJsonError { kind: SessionJsonErrorKind::Io, message: e.to_string() })?;
    let based_on_hash = sha256_hex(&raw);
    let mut doc = parse_session_json(&raw)?;

    let changed = apply_session_meta(&mut doc, patch);
    let unchanged = dry_run || changed.is_empty();
    if !unchanged {
        write_session_json(data_root, session_id, &doc, Some(&based_on_hash))?;
    }

    Ok(SessionMetaReport { doc, changed, unchanged })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::session_json::empty_session_json;

    fn summary(venue: &str, tag: &str, timestamp_utc_ms: i64) -> SessionSummary {
        SessionSummary {
            session_id: format!("{venue}-{tag}-{timestamp_utc_ms}"),
            blob_sha256: String::new(),
            source_format: "idl0".to_string(),
            device_id: None,
            config_checksum: None,
            importer_version: "0.1.0".to_string(),
            seam_correction_version: "v1".to_string(),
            engine_version: "0.1.0".to_string(),
            timestamp_utc_ms,
            created_at_ms: 0,
            rider: String::new(),
            bike: String::new(),
            venue_name: venue.to_string(),
            event_name: String::new(),
            event_session: String::new(),
            short_comment: String::new(),
            tag: tag.to_string(),
            lap_count: None,
            duration_ms: None,
        }
    }

    #[test]
    fn an_empty_filter_keeps_every_row() {
        // Arrange
        let rows = vec![summary("Cadwell", "wet", 1_000), summary("Donington", "dry", 2_000)];

        // Act
        let kept = filter_sessions(&rows, &SessionFilter::default());

        // Assert
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn the_venue_filter_matches_case_insensitively() {
        // Arrange
        let rows = vec![summary("Cadwell", "", 1_000), summary("Donington", "", 2_000)];
        let filter = SessionFilter { venue: Some("cadwell".to_string()), ..Default::default() };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].venue_name, "Cadwell");
    }

    #[test]
    fn the_venue_filter_does_not_match_a_prefix() {
        // Arrange
        let rows = vec![summary("Cadwell Park", "", 1_000)];
        let filter = SessionFilter { venue: Some("Cadwell".to_string()), ..Default::default() };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert — whole-value matching, so a filter never quietly widens.
        assert!(kept.is_empty());
    }

    #[test]
    fn the_tag_filter_matches_case_insensitively() {
        // Arrange
        let rows = vec![summary("Cadwell", "Wet", 1_000), summary("Cadwell", "dry", 2_000)];
        let filter = SessionFilter { tag: Some("WET".to_string()), ..Default::default() };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].tag, "Wet");
    }

    #[test]
    fn the_since_and_until_bounds_are_both_inclusive() {
        // Arrange
        let rows = vec![summary("a", "", 1_000), summary("b", "", 2_000), summary("c", "", 3_000)];
        let filter = SessionFilter {
            since_utc_ms: Some(1_000),
            until_utc_ms: Some(2_000),
            ..Default::default()
        };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn a_session_with_an_unknown_start_is_dropped_by_any_time_bound() {
        // Arrange
        let rows = vec![summary("a", "", 0), summary("b", "", 2_000)];
        let filter = SessionFilter { since_utc_ms: Some(1_000), ..Default::default() };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert — `0` is "unknown" (C1 §3.1), never 1970.
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].venue_name, "b");
    }

    #[test]
    fn a_session_with_an_unknown_start_survives_when_no_time_bound_is_given() {
        // Arrange
        let rows = vec![summary("Cadwell", "", 0)];
        let filter = SessionFilter { venue: Some("Cadwell".to_string()), ..Default::default() };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn filters_combine_as_and_not_or() {
        // Arrange
        let rows = vec![summary("Cadwell", "wet", 1_000), summary("Cadwell", "dry", 2_000)];
        let filter = SessionFilter {
            venue: Some("Cadwell".to_string()),
            tag: Some("dry".to_string()),
            ..Default::default()
        };

        // Act
        let kept = filter_sessions(&rows, &filter);

        // Assert
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].tag, "dry");
    }

    #[test]
    fn an_empty_patch_reports_itself_empty() {
        // Arrange
        let patch = SessionMetaPatch::default();

        // Act
        let empty = patch.is_empty();

        // Assert
        assert!(empty);
    }

    #[test]
    fn applying_a_patch_sets_only_the_fields_it_names() {
        // Arrange
        let mut doc = empty_session_json("s1");
        doc.rider = "Isaac".to_string();
        let patch =
            SessionMetaPatch { venue_name: Some("Cadwell".to_string()), ..Default::default() };

        // Act
        let changed = apply_session_meta(&mut doc, &patch);

        // Assert
        assert_eq!(changed, vec!["venue_name"]);
        assert_eq!(doc.venue_name, "Cadwell");
        assert_eq!(doc.rider, "Isaac");
    }

    #[test]
    fn applying_a_patch_that_changes_nothing_reports_no_changed_fields() {
        // Arrange
        let mut doc = empty_session_json("s1");
        doc.venue_name = "Cadwell".to_string();
        let patch =
            SessionMetaPatch { venue_name: Some("Cadwell".to_string()), ..Default::default() };

        // Act
        let changed = apply_session_meta(&mut doc, &patch);

        // Assert — this is what makes `set-meta` idempotent.
        assert!(changed.is_empty());
    }

    #[test]
    fn an_empty_string_clears_a_field() {
        // Arrange
        let mut doc = empty_session_json("s1");
        doc.tag = "wet".to_string();
        let patch = SessionMetaPatch { tag: Some(String::new()), ..Default::default() };

        // Act
        let changed = apply_session_meta(&mut doc, &patch);

        // Assert — C1 §6: `""` is how "not set" is spelled.
        assert_eq!(changed, vec!["tag"]);
        assert_eq!(doc.tag, "");
    }

    #[test]
    fn applying_a_patch_reports_every_changed_field_in_declaration_order() {
        // Arrange
        let mut doc = empty_session_json("s1");
        let patch = SessionMetaPatch {
            venue_name: Some("Cadwell".to_string()),
            bike: Some("R6".to_string()),
            tag: Some("wet".to_string()),
            ..Default::default()
        };

        // Act
        let changed = apply_session_meta(&mut doc, &patch);

        // Assert
        assert_eq!(changed, vec!["venue_name", "bike", "tag"]);
    }

    #[test]
    fn the_notes_flag_writes_the_long_comment_field() {
        // Arrange
        let mut doc = empty_session_json("s1");
        let patch = SessionMetaPatch { long_comment: Some("gearing".to_string()), ..Default::default() };

        // Act
        apply_session_meta(&mut doc, &patch);

        // Assert
        assert_eq!(doc.long_comment, "gearing");
    }

    #[test]
    fn set_session_meta_rejects_a_patch_that_sets_nothing() {
        // Arrange
        let root = std::env::temp_dir().join("idl-rs-session-ops-empty");

        // Act
        let err =
            set_session_meta(&root, "s1", &SessionMetaPatch::default(), false).unwrap_err();

        // Assert
        assert_eq!(err.kind, SessionJsonErrorKind::InvalidArgument);
    }
}
