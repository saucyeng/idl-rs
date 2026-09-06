//! `session.json` (contract C1 §6) — replaces `.idl0w`. Metadata, lap gates,
//! cached laps, lap flags, and track visits for one session.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{parse_config, read_config, ConfigError, VersionedConfig};
use crate::store::atomic::{write_atomic, AtomicWriteError};

/// The schema version this build of `idl-rs` writes and accepts for
/// `session.json` (C1 §6).
pub const SESSION_JSON_SCHEMA_VERSION: u32 = 1;

/// One session's metadata, lap gates, cached laps, lap flags, and track
/// visits (C1 §6). Replaces idl0's `.idl0w` workbook file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionJson {
    /// Schema version of this document (see [`SESSION_JSON_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The session this file belongs to, matching the containing directory
    /// name (`<data>/sessions/<session_id>/`).
    pub session_id: String,

    /// Rider name. `""` = not set.
    #[serde(default)]
    pub rider: String,
    /// Bike name. `""` = not set.
    #[serde(default)]
    pub bike: String,
    /// Free-text comment about the bike setup. `""` = not set.
    #[serde(default)]
    pub bike_comment: String,
    /// Venue/track name. `""` = not set.
    #[serde(default)]
    pub venue_name: String,
    /// Event name. `""` = not set.
    #[serde(default)]
    pub event_name: String,
    /// Event session label (e.g. "Q1"). `""` = not set.
    #[serde(default)]
    pub event_session: String,
    /// Short free-text comment. `""` = not set.
    #[serde(default)]
    pub short_comment: String,
    /// Long free-text comment. `""` = not set.
    #[serde(default)]
    pub long_comment: String,
    /// Free-text tag. `""` = not set.
    #[serde(default)]
    pub tag: String,

    /// A frozen copy of the bike profile in effect when this session was
    /// recorded/edited, opaque to this crate.
    #[serde(default)]
    pub bike_profile_snapshot: Option<serde_json::Value>,

    /// Lap-start/finish gates for this session.
    #[serde(default)]
    pub lap_gates: Vec<LapGateJson>,
    /// Sector gates for this session.
    #[serde(default)]
    pub sector_gates: Vec<SectorGateJson>,

    /// Cached lap boundaries and derived timings.
    #[serde(default)]
    pub laps: Vec<LapJson>,

    /// The lap number used as the comparison reference, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_lap_number: Option<u32>,
    /// Lap numbers excluded from aggregate analysis.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored_lap_numbers: Vec<u32>,
    /// The lap number flagged as this session's "main" lap, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main_lap_number: Option<u32>,
    /// Cross-session overlay reference lap, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_lap_key: Option<OverlayLapKeyJson>,
    /// The lap number starred by the rider, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starred_lap_number: Option<u32>,

    /// Track-library visits recorded for this session.
    #[serde(default)]
    pub track_visits: Vec<TrackVisitJson>,
    /// Sha256 hex of the track library state used to resolve
    /// [`SessionJson::track_visits`], if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_visits_library_hash: Option<String>,
    /// The `store::lap_index::LAP_DETECTOR_VERSION` stamped by whichever
    /// import/rescan last wrote [`SessionJson::laps`]/
    /// [`SessionJson::track_visits`], if any. Additive C1 §6 field (ruling
    /// R83 Q2) — an older file with no `lap_detector_version` parses
    /// unchanged (`None`), and a mismatch against the running build's
    /// `LAP_DETECTOR_VERSION` (or a track library whose
    /// `track_visits_library_hash` no longer matches) marks the cached laps
    /// stale, triggering a re-index on next import or an explicit rescan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lap_detector_version: Option<String>,
}

impl VersionedConfig for SessionJson {
    const SUPPORTED_VERSION: u32 = SESSION_JSON_SCHEMA_VERSION;
    const LABEL: &'static str = "session.json";
    fn version(&self) -> u32 {
        self.schema_version
    }
}

/// A single lap-start/finish gate, decimal degrees (C1 §6, settled by ruling
/// R8: `session.json` is a human-legible file, not an internal struct, so it
/// keeps decimal degrees. Ruling R27 moved `crate::laps::model::Gate`/
/// `crate::gps::GpsFix` to the same physical decimal-degree scale, so the
/// conversion this JSON boundary once did is gone — it is now a plain field
/// copy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LapGateJson {
    /// First gate endpoint latitude, decimal degrees.
    pub lat1_deg: f64,
    /// First gate endpoint longitude, decimal degrees.
    pub lon1_deg: f64,
    /// Second gate endpoint latitude, decimal degrees.
    pub lat2_deg: f64,
    /// Second gate endpoint longitude, decimal degrees.
    pub lon2_deg: f64,
    /// Gate display name. `""` = not set.
    #[serde(default)]
    pub name: String,
}

/// A named sector boundary gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectorGateJson {
    /// Sector display name.
    pub name: String,
    /// The gate geometry marking this sector's boundary.
    pub gate: LapGateJson,
}

/// A cached lap's boundaries and derived timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LapJson {
    /// 1-based lap number within the session.
    pub lap_number: u32,
    /// Lap start, milliseconds since the Unix epoch.
    pub start_timestamp_ms: i64,
    /// Lap end, milliseconds since the Unix epoch.
    pub end_timestamp_ms: i64,
    /// Raw elapsed time from start to end, milliseconds.
    pub raw_elapsed_ms: i64,
    /// Lap time after any adjustments (e.g. neutral-zone exclusion),
    /// milliseconds.
    pub lap_time_ms: i64,
    /// Lap start, seconds into the session.
    pub start_time_secs: f64,
    /// Lap end, seconds into the session.
    pub end_time_secs: f64,
    /// Sector splits within this lap.
    #[serde(default)]
    pub sectors: Vec<SectorJson>,
    /// Neutral-zone visits within this lap.
    #[serde(default)]
    pub neutral_zone_visits: Vec<NeutralZoneVisitJson>,
}

/// A single sector split within a lap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectorJson {
    /// Sector display name.
    pub name: String,
    /// Sector start, milliseconds since the Unix epoch.
    pub start_ms: i64,
    /// Sector end, milliseconds since the Unix epoch.
    pub end_ms: i64,
    /// Sector start, seconds into the session.
    pub start_time_secs: f64,
    /// Sector end, seconds into the session.
    pub end_time_secs: f64,
}

/// A single neutral-zone visit within a lap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeutralZoneVisitJson {
    /// Neutral-zone display name.
    pub name: String,
    /// Entry time, milliseconds since the Unix epoch.
    pub enter_ms: i64,
    /// Exit time, milliseconds since the Unix epoch.
    pub exit_ms: i64,
}

/// A reference to a specific lap in another session, used for cross-session
/// overlay comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayLapKeyJson {
    /// The referenced session's `session_id`.
    pub session_id: String,
    /// The referenced lap number within that session.
    pub lap_number: u32,
}

/// A single visit to a track-library entry, with its own cached laps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackVisitJson {
    /// Unique id for this visit.
    pub visit_id: String,
    /// The track-library entry this visit refers to.
    pub track_id: String,
    /// Visit start, milliseconds since the Unix epoch.
    pub start_timestamp_ms: i64,
    /// Visit end, milliseconds since the Unix epoch.
    pub end_timestamp_ms: i64,
    /// Cached laps recorded during this visit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub laps: Vec<LapJson>,
}

/// Discriminant for [`SessionJsonError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionJsonErrorKind {
    /// A filesystem operation failed.
    Io,
    /// JSON did not parse / required fields were missing.
    Parse,
    /// The file's schema version exceeds what this engine supports.
    UnsupportedVersion,
}

/// Error from [`read_session_json`] / [`parse_session_json`] /
/// [`write_session_json`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionJsonError {
    pub kind: SessionJsonErrorKind,
    pub message: String,
}

impl fmt::Display for SessionJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for SessionJsonError {}

impl From<ConfigError> for SessionJsonError {
    fn from(e: ConfigError) -> Self {
        let kind = match e.kind {
            crate::config::ConfigErrorKind::Io => SessionJsonErrorKind::Io,
            crate::config::ConfigErrorKind::Parse => SessionJsonErrorKind::Parse,
            crate::config::ConfigErrorKind::UnsupportedVersion => SessionJsonErrorKind::UnsupportedVersion,
        };
        SessionJsonError { kind, message: e.message }
    }
}
impl From<AtomicWriteError> for SessionJsonError {
    fn from(e: AtomicWriteError) -> Self {
        SessionJsonError { kind: SessionJsonErrorKind::Io, message: e.to_string() }
    }
}

/// Reads `<data>/sessions/<session_id>/session.json`, reusing the crate's
/// existing `VersionedConfig` machinery (the same one `.idl0t` uses).
pub fn read_session_json(path: &Path) -> Result<SessionJson, SessionJsonError> {
    Ok(read_config::<SessionJson>(path)?)
}

/// Parses `session.json` bytes without touching disk (used by `verify`,
/// Task 12, and by tests).
pub fn parse_session_json(bytes: &[u8]) -> Result<SessionJson, SessionJsonError> {
    Ok(parse_config::<SessionJson>(bytes)?)
}

/// Writes `session.json` atomically (C4 §4) to
/// `<data_root>/sessions/<session_id>/session.json`. Returns the new
/// content's sha256 hex.
pub fn write_session_json(
    data_root: &Path,
    session_id: &str,
    doc: &SessionJson,
    based_on_hash: Option<&str>,
) -> Result<String, SessionJsonError> {
    let bytes = serde_json::to_vec_pretty(doc)
        .map_err(|e| SessionJsonError { kind: SessionJsonErrorKind::Parse, message: e.to_string() })?;
    let target = data_root.join("sessions").join(session_id).join("session.json");
    Ok(write_atomic(data_root, &target, &bytes, based_on_hash)?)
}

/// A fresh, empty `session.json` for a newly-imported session — every
/// string field `""`, every collection empty, matching C1 §6's stated
/// defaults exactly ("`""` = not set, no null representation").
pub fn empty_session_json(session_id: impl Into<String>) -> SessionJson {
    SessionJson {
        schema_version: SESSION_JSON_SCHEMA_VERSION,
        session_id: session_id.into(),
        rider: String::new(),
        bike: String::new(),
        bike_comment: String::new(),
        venue_name: String::new(),
        event_name: String::new(),
        event_session: String::new(),
        short_comment: String::new(),
        long_comment: String::new(),
        tag: String::new(),
        bike_profile_snapshot: None,
        lap_gates: Vec::new(),
        sector_gates: Vec::new(),
        laps: Vec::new(),
        reference_lap_number: None,
        ignored_lap_numbers: Vec::new(),
        main_lap_number: None,
        overlay_lap_key: None,
        starred_lap_number: None,
        track_visits: Vec::new(),
        track_visits_library_hash: None,
        lap_detector_version: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn empty_session_json_round_trips_through_write_and_read() {
        // Arrange
        let root = temp_root();
        let doc = empty_session_json("abc123");

        // Act
        write_session_json(&root, "abc123", &doc, None).unwrap();
        let back = read_session_json(&root.join("sessions").join("abc123").join("session.json")).unwrap();

        // Assert
        assert_eq!(back, doc);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_then_write_again_uses_the_optimistic_concurrency_hash() {
        // Arrange
        let root = temp_root();
        let mut doc = empty_session_json("abc123");
        let h1 = write_session_json(&root, "abc123", &doc, None).unwrap();

        // Act
        doc.rider = "Isaac".to_string();
        let h2 = write_session_json(&root, "abc123", &doc, Some(&h1)).unwrap();

        // Assert
        assert_ne!(h1, h2);
        let back = read_session_json(&root.join("sessions").join("abc123").join("session.json")).unwrap();
        assert_eq!(back.rider, "Isaac");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn omitted_optional_fields_are_absent_from_the_json_not_null() {
        // Arrange
        let doc = empty_session_json("abc123");

        // Act
        let bytes = serde_json::to_vec(&doc).unwrap();
        let text = String::from_utf8(bytes).unwrap();

        // Assert — C1 §6: "omitted (not an empty-array string / null)".
        assert!(!text.contains("reference_lap_number"));
        assert!(!text.contains("ignored_lap_numbers"));
        assert!(!text.contains("track_visits_library_hash"));
    }

    #[test]
    fn malformed_json_is_a_typed_parse_error_not_a_panic() {
        // Act
        let err = parse_session_json(b"not json").unwrap_err();

        // Assert
        assert_eq!(err.kind, SessionJsonErrorKind::Parse);
    }

    #[test]
    fn future_schema_version_is_rejected_typed() {
        // Arrange
        let json = r#"{"schema_version":99,"session_id":"x"}"#;

        // Act
        let err = parse_session_json(json.as_bytes()).unwrap_err();

        // Assert
        assert_eq!(err.kind, SessionJsonErrorKind::UnsupportedVersion);
    }
}
