//! Pure lap-indexing core (IDL0_SPEC §17.4): load the track library, hash it,
//! detect this session's track visits and the laps within each, and
//! session-wide-renumber the top-level lap list. Reads only a
//! [`SessionHandle`] already in memory and a slice of [`Track`]s already read
//! from disk — no filesystem access, no clock, no RNG, no writes.
//! `store::import`/a future `reindex_laps` (Task 2) own the I/O around this.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::laps::{detect_laps, renumber_session_laps};
use crate::session::handle::SessionHandle;
use crate::store::atomic::sha256_hex;
use crate::store::parquet::open_session_lazy;
use crate::store::session_json::{
    empty_session_json, parse_session_json, write_session_json, LapJson, NeutralZoneVisitJson, SectorJson,
    TrackVisitJson,
};
use crate::track_artifact::{read_track, Track};
use crate::tracks::detect_visits;

/// Discriminant for [`LapIndexError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapIndexErrorKind {
    /// A filesystem operation failed (reading `<data_root>/tracks/`).
    Io,
    /// A `.idl0t` track artifact was malformed in a way that stops the whole
    /// library load rather than being skipped (currently unreached —
    /// per-track parse failures are skipped into warnings instead; kept for
    /// a future fatal-track case rather than widening `Io`).
    Track,
}

/// Error from [`load_track_library`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LapIndexError {
    /// Discriminant.
    pub kind: LapIndexErrorKind,
    /// Human-readable detail, including the offending path where relevant.
    pub message: String,
}

impl LapIndexError {
    fn new(kind: LapIndexErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for LapIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for LapIndexError {}

/// Opaque stamp over the track library; callers must not parse it. Port of
/// idl0's `trackLibraryHash` (`track_provider.dart`), with idl0's `sha1`
/// swapped for this crate's own `sha2` dependency (no `sha1` crate is
/// vendored here) — same algorithm shape, `"sha256:"` prefix instead of
/// `"sha1:"`. Builds `"<track_id>:<updated_at_ms>"` per track, sorts the
/// strings so track order never affects the hash, joins with `'|'`, and
/// hashes the UTF-8 bytes. An empty library hashes the empty string.
pub fn track_library_hash(tracks: &[Track]) -> String {
    let mut pairs: Vec<String> = tracks.iter().map(|t| format!("{}:{}", t.id, t.updated_at_ms)).collect();
    pairs.sort();
    let joined = pairs.join("|");
    let digest = Sha256::digest(joined.as_bytes());
    format!("sha256:{digest:x}")
}

/// Every `<data_root>/tracks/*.idl0t`, sorted by track id. A missing
/// `tracks/` directory is `Ok(vec![])`, not an error. An unreadable or
/// version-rejected artifact is skipped, and a warning naming its file
/// records why.
pub fn load_track_library(data_root: &Path) -> Result<(Vec<Track>, Vec<String>), LapIndexError> {
    let tracks_dir = data_root.join("tracks");
    if !tracks_dir.is_dir() {
        return Ok((Vec::new(), Vec::new()));
    }
    let entries = fs::read_dir(&tracks_dir)
        .map_err(|e| LapIndexError::new(LapIndexErrorKind::Io, format!("cannot read {}: {e}", tracks_dir.display())))?;

    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| LapIndexError::new(LapIndexErrorKind::Io, format!("cannot read {}: {e}", tracks_dir.display())))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("idl0t") {
            paths.push(path);
        }
    }

    let mut tracks: Vec<Track> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for path in paths {
        match read_track(&path) {
            Ok(t) => tracks.push(t),
            Err(e) => warnings.push(format!("skipped track artifact {}: {e}", path.display())),
        }
    }
    tracks.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((tracks, warnings))
}

/// Deterministic visit identity: lowercase hex of
/// `sha256("<track_id>:<start_ms>:<end_ms>")`, truncated to 16 characters.
/// Deterministic (not idl0's random UUID, ruling R83 Q4) so a rescan that
/// finds the same visit again does not churn a synced `session.json`.
pub fn visit_id(track_id: &str, start_ms: i64, end_ms: i64) -> String {
    let digest = Sha256::digest(format!("{track_id}:{start_ms}:{end_ms}").as_bytes());
    format!("{digest:x}")[..16].to_string()
}

/// Result of [`compute_lap_index`]: the session's track visits (each with its
/// own cached, per-visit-numbered laps), the session-wide-renumbered
/// top-level lap list, the track-library stamp used to resolve them, and any
/// non-fatal warnings (a visit whose track could not be resolved, or whose
/// track has no lap timing).
pub struct LapIndex {
    /// Each visited track's window and its own per-visit-numbered laps.
    pub track_visits: Vec<TrackVisitJson>,
    /// The top-level, session-wide-renumbered lap list (per-visit numbers
    /// live only on [`Self::track_visits`]).
    pub laps: Vec<LapJson>,
    /// [`track_library_hash`]'s stamp over the `tracks` slice this was
    /// resolved against.
    pub track_library_hash: String,
    /// Non-fatal messages — a visit whose track could not be resolved, or
    /// whose track has no lap timing.
    pub warnings: Vec<String>,
}

/// Detects this session's track visits and laps against `tracks`. Pure over
/// `handle` + `tracks`: no filesystem access, no clock, no RNG. Never an
/// error — an empty `tracks` slice, no surviving visit window, or a resolved
/// track with no lap timing all fold into an honest-empty [`LapIndex`]
/// (IDL0_SPEC §17.4) rather than failing.
///
/// 1. `detect_visits` runs with `VisitParams::default()` — tuning lives once
///    in the engine (IDL0_SPEC §17.3), this function takes no overrides.
/// 2. Each window's `Track` is resolved by id. A window whose track is
///    missing from `tracks`, or whose track has `timing: None`, still
///    produces a [`TrackVisitJson`] (with `laps: []`) and a warning — the
///    visit is recorded even though it has no laps.
/// 3. Otherwise `detect_laps` runs restricted to the visit's window, and each
///    `laps::model::Lap`/`Sector`/`NeutralZoneVisit` converts field for field
///    into the matching `session_json` DTO.
/// 4. `renumber_session_laps` assigns the top-level, session-wide lap
///    numbers; each visit's own `laps[]` keeps its per-visit numbering
///    (`renumber_session_laps`'s doc comment: per-visit numbers are not the
///    session-wide identity).
pub fn compute_lap_index(handle: &SessionHandle, tracks: &[Track], ignored_lap_numbers: &[u32]) -> LapIndex {
    let hash = track_library_hash(tracks);
    let mut warnings: Vec<String> = Vec::new();

    if tracks.is_empty() {
        return LapIndex { track_visits: Vec::new(), laps: Vec::new(), track_library_hash: hash, warnings };
    }

    let refs: Vec<crate::tracks::TrackRef> = tracks.iter().map(Track::track_ref).collect();
    let windows = detect_visits(handle, &refs, Default::default());

    let mut track_visits: Vec<TrackVisitJson> = Vec::with_capacity(windows.len());
    for window in windows {
        let (visit, warning) = resolve_visit(handle, window, tracks);
        track_visits.push(visit);
        warnings.extend(warning);
    }

    let laps = renumber_session_laps(&track_visits, ignored_lap_numbers).into_iter().map(|r| r.lap).collect();

    LapIndex { track_visits, laps, track_library_hash: hash, warnings }
}

/// Bumped whenever a change to the detection algorithm alters its output
/// enough that already-cached `session.json` entries must be recomputed even
/// though [`track_library_hash`] has not changed. Stamped into
/// `session.json.lap_detector_version` (C1 §6, additive field, ruling R83
/// Q2) by [`index_laps`]; a mismatch (including a file with no stamp at all)
/// marks the cache stale exactly like a changed [`track_library_hash`].
pub const LAP_DETECTOR_VERSION: &str = "1";

/// Outcome of one [`index_laps`]/[`reindex_laps`] call.
#[derive(Debug)]
pub struct LapIndexReport {
    /// The session indexed.
    pub session_id: String,
    /// `track_visits[]` length written this call. `0` when
    /// [`Self::skipped_up_to_date`] is true (nothing was recomputed, so
    /// nothing was written).
    pub visits_indexed: usize,
    /// Top-level `laps[]` length written this call. `0` when
    /// [`Self::skipped_up_to_date`] is true.
    pub laps_indexed: usize,
    /// True when the stamp was already current and nothing was recomputed —
    /// `session.json` is left byte-for-byte untouched.
    pub skipped_up_to_date: bool,
    /// Lap-flag fields cleared because their lap number no longer exists
    /// after renumbering (ruling R83 Q3): any of `"main_lap_number"`,
    /// `"reference_lap_number"`, `"starred_lap_number"`, or
    /// `"ignored_lap_numbers"` (the last when at least one of its entries
    /// was dropped). `overlay_lap_key` is never in this list — it names a
    /// lap in *another* session, which this session's own renumbering
    /// cannot invalidate.
    pub flags_cleared: Vec<String>,
    /// Non-fatal warnings from [`load_track_library`]/[`compute_lap_index`].
    /// Empty when [`Self::skipped_up_to_date`] is true (neither ran).
    pub warnings: Vec<String>,
}

/// Indexes laps for a session whose [`SessionHandle`] the caller already
/// holds (the import path, `store::import::finish_import`) and merges the
/// result into `session.json` (C1 §6). Never touches `data.parquet`;
/// [`reindex_laps`] is the entry point that rebuilds `handle` from disk
/// first.
///
/// **Staleness.** Recomputes when `force`, or when the freshly-loaded track
/// library's hash differs from the stamped `track_visits_library_hash`, or
/// when the stamped `lap_detector_version` differs from
/// [`LAP_DETECTOR_VERSION`] (including a file with no stamp at all).
/// Otherwise returns early with `skipped_up_to_date: true` and **writes
/// nothing**.
///
/// **Missing `session.json`.** Starts from [`empty_session_json`] rather
/// than erroring — `finish_import` may run before or after this call.
///
/// **Merge.** Only `track_visits`, `laps`, `track_visits_library_hash`,
/// `lap_detector_version`, and the lap-flag fields reconciled below are
/// overwritten; every other field (rider, bike, comments, gates,
/// `bike_profile_snapshot`) is carried through by value. `ignored_lap_numbers`
/// is read *before* reconciliation and passed to [`compute_lap_index`],
/// since `renumber_session_laps` takes it as an input to renumbering itself.
///
/// **Flag reconciliation (ruling R83 Q3).** After the new `laps[]` is built,
/// `main_lap_number`/`reference_lap_number`/`starred_lap_number` are cleared
/// to `None` when they name a lap number no longer present, and
/// `ignored_lap_numbers` is filtered to the surviving numbers.
pub fn index_laps(
    data_root: &Path,
    session_id: &str,
    handle: &SessionHandle,
    force: bool,
) -> Result<LapIndexReport, LapIndexError> {
    let sj_path = data_root.join("sessions").join(session_id).join("session.json");
    let (mut doc, based_on_hash) = if sj_path.is_file() {
        let bytes = fs::read(&sj_path)
            .map_err(|e| LapIndexError::new(LapIndexErrorKind::Io, format!("reading {}: {e}", sj_path.display())))?;
        let doc = parse_session_json(&bytes)
            .map_err(|e| LapIndexError::new(LapIndexErrorKind::Io, format!("parsing {}: {e}", sj_path.display())))?;
        (doc, Some(sha256_hex(&bytes)))
    } else {
        (empty_session_json(session_id), None)
    };

    let (tracks, mut warnings) = load_track_library(data_root)?;
    let fresh_hash = track_library_hash(&tracks);

    let stale = force
        || doc.track_visits_library_hash.as_deref() != Some(fresh_hash.as_str())
        || doc.lap_detector_version.as_deref() != Some(LAP_DETECTOR_VERSION);

    if !stale {
        return Ok(LapIndexReport {
            session_id: session_id.to_string(),
            visits_indexed: 0,
            laps_indexed: 0,
            skipped_up_to_date: true,
            flags_cleared: Vec::new(),
            warnings: Vec::new(),
        });
    }

    let index = compute_lap_index(handle, &tracks, &doc.ignored_lap_numbers);
    warnings.extend(index.warnings);

    let valid: HashSet<u32> = index.laps.iter().map(|l| l.lap_number).collect();
    let mut flags_cleared: Vec<String> = Vec::new();

    if matches!(doc.main_lap_number, Some(n) if !valid.contains(&n)) {
        doc.main_lap_number = None;
        flags_cleared.push("main_lap_number".to_string());
    }
    if matches!(doc.reference_lap_number, Some(n) if !valid.contains(&n)) {
        doc.reference_lap_number = None;
        flags_cleared.push("reference_lap_number".to_string());
    }
    if matches!(doc.starred_lap_number, Some(n) if !valid.contains(&n)) {
        doc.starred_lap_number = None;
        flags_cleared.push("starred_lap_number".to_string());
    }
    let ignored_before = doc.ignored_lap_numbers.len();
    doc.ignored_lap_numbers.retain(|n| valid.contains(n));
    if doc.ignored_lap_numbers.len() != ignored_before {
        flags_cleared.push("ignored_lap_numbers".to_string());
    }
    // `overlay_lap_key` names a lap in another session's own numbering —
    // this session's renumbering cannot invalidate it, so it is left alone.

    doc.laps = index.laps;
    doc.track_visits = index.track_visits;
    doc.track_visits_library_hash = Some(index.track_library_hash);
    doc.lap_detector_version = Some(LAP_DETECTOR_VERSION.to_string());

    let visits_indexed = doc.track_visits.len();
    let laps_indexed = doc.laps.len();

    write_session_json(data_root, session_id, &doc, based_on_hash.as_deref())
        .map_err(|e| LapIndexError::new(LapIndexErrorKind::Io, e.to_string()))?;

    Ok(LapIndexReport {
        session_id: session_id.to_string(),
        visits_indexed,
        laps_indexed,
        skipped_up_to_date: false,
        flags_cleared,
        warnings,
    })
}

/// IDL0_SPEC §17.4's "Rescan Tracks": rebuilds the handle from
/// `sessions/<id>/data.parquet` (so a rescan sees the same signal data
/// import saw, without re-parsing the original source blob), then calls
/// [`index_laps`] with `force = true` so a rescan always recomputes even
/// when the stamp still matches — the whole point of an explicit rescan is
/// to let the rider confirm newly-added or newly-edited tracks took effect.
/// A missing `data.parquet` is [`LapIndexErrorKind::Io`] with the path in
/// the message, never a panic.
pub fn reindex_laps(data_root: &Path, session_id: &str) -> Result<LapIndexReport, LapIndexError> {
    let session_dir = data_root.join("sessions").join(session_id);
    let handle = open_session_lazy(&session_dir).map_err(|e| {
        LapIndexError::new(LapIndexErrorKind::Io, format!("reading {}: {e}", session_dir.join("data.parquet").display()))
    })?;
    index_laps(data_root, session_id, &handle, true)
}

/// Resolves one detected visit window against the track library: finds its
/// `Track` by id and, if found with lap timing configured, detects its laps.
/// A window whose `track_id` has no match in `tracks` (structurally
/// unreachable through [`compute_lap_index`] itself, since its `refs` are
/// built from that same slice — kept as documented defence, precedent
/// `list_math_builtins`'s unreachable branch) or whose matched track has no
/// lap timing both return an empty `laps` and a warning; the visit is always
/// returned (IDL0_SPEC §17.4's "visits present, laps absent").
fn resolve_visit(handle: &SessionHandle, window: crate::tracks::VisitWindow, tracks: &[Track]) -> (TrackVisitJson, Option<String>) {
    let id = visit_id(&window.track_id, window.start_timestamp_ms, window.end_timestamp_ms);
    let track = tracks.iter().find(|t| t.id == window.track_id);

    let (laps, warning) = match track.and_then(|t| t.timing.as_ref().map(|timing| (t, timing))) {
        Some((t, timing)) => (
            detect_laps(
                handle,
                timing,
                &t.sector_gates,
                &t.neutral_zones,
                Some((window.start_timestamp_ms, window.end_timestamp_ms)),
            )
            .into_iter()
            .map(lap_to_json)
            .collect(),
            None,
        ),
        None => {
            let message = if track.is_none() {
                format!("visit {id}: track {} not found in the library, laps not detected", window.track_id)
            } else {
                format!("visit {id}: track {} has no lap timing configured, laps not detected", window.track_id)
            };
            (Vec::new(), Some(message))
        }
    };

    (
        TrackVisitJson {
            visit_id: id,
            track_id: window.track_id,
            start_timestamp_ms: window.start_timestamp_ms,
            end_timestamp_ms: window.end_timestamp_ms,
            laps,
        },
        warning,
    )
}

/// Converts an engine [`crate::laps::model::Lap`] into its `session.json`
/// wire DTO, field for field.
fn lap_to_json(lap: crate::laps::model::Lap) -> LapJson {
    LapJson {
        lap_number: lap.lap_number,
        start_timestamp_ms: lap.start_ms,
        end_timestamp_ms: lap.end_ms,
        raw_elapsed_ms: lap.raw_elapsed_ms,
        lap_time_ms: lap.lap_time_ms,
        start_time_secs: lap.start_time_secs,
        end_time_secs: lap.end_time_secs,
        sectors: lap.sectors.into_iter().map(sector_to_json).collect(),
        neutral_zone_visits: lap.neutral_zone_visits.into_iter().map(neutral_zone_visit_to_json).collect(),
    }
}

/// Converts an engine [`crate::laps::model::Sector`] into its wire DTO.
fn sector_to_json(s: crate::laps::model::Sector) -> SectorJson {
    SectorJson {
        name: s.name,
        start_ms: s.start_ms,
        end_ms: s.end_ms,
        start_time_secs: s.start_time_secs,
        end_time_secs: s.end_time_secs,
    }
}

/// Converts an engine [`crate::laps::model::NeutralZoneVisit`] into its wire
/// DTO.
fn neutral_zone_visit_to_json(z: crate::laps::model::NeutralZoneVisit) -> NeutralZoneVisitJson {
    NeutralZoneVisitJson { name: z.name, enter_ms: z.enter_ms, exit_ms: z.exit_ms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laps::model::{Gate, LapTiming};
    use crate::session::handle::{ChannelInput, SessionMetaInput};
    use crate::gps::GpsFix;
    use crate::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{read_session_json, OverlayLapKeyJson};
    use crate::track_artifact::write_track;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-lapidx-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sj_path(root: &Path, session_id: &str) -> std::path::PathBuf {
        root.join("sessions").join(session_id).join("session.json")
    }

    fn track_stub(id: &str, updated_at_ms: i64) -> Track {
        Track {
            id: id.to_string(),
            name: String::new(),
            venue: String::new(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 0,
            updated_at_ms,
        }
    }

    #[test]
    fn track_library_hash_is_order_independent() {
        // Arrange -- two shuffled orderings of the same three tracks.
        let a = vec![track_stub("t1", 100), track_stub("t2", 200), track_stub("t3", 300)];
        let b = vec![track_stub("t3", 300), track_stub("t1", 100), track_stub("t2", 200)];

        // Act
        let hash_a = track_library_hash(&a);
        let hash_b = track_library_hash(&b);

        // Assert
        assert_eq!(hash_a, hash_b);
        assert!(hash_a.starts_with("sha256:"));
    }

    #[test]
    fn track_library_hash_changes_when_an_updated_at_ms_changes() {
        // Arrange
        let before = vec![track_stub("t1", 100)];
        let after = vec![track_stub("t1", 999)];

        // Act
        let hash_before = track_library_hash(&before);
        let hash_after = track_library_hash(&after);

        // Assert
        assert_ne!(hash_before, hash_after);
    }

    #[test]
    fn track_library_hash_of_empty_library_hashes_the_empty_string() {
        // Act
        let hash = track_library_hash(&[]);

        // Assert -- the well-known SHA-256 digest of the empty byte string,
        // an independent oracle rather than a second call to `Sha256::digest`.
        assert_eq!(hash, "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn visit_id_is_stable_across_calls_and_differs_for_different_bounds() {
        // Act
        let a1 = visit_id("t1", 1000, 2000);
        let a2 = visit_id("t1", 1000, 2000);
        let b = visit_id("t1", 1000, 3000);

        // Assert
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_eq!(a1.len(), 16);
    }

    #[test]
    fn load_track_library_on_a_root_with_no_tracks_dir_is_empty_ok() {
        // Arrange -- a fresh temp dir with no `tracks/` subdirectory.
        let dir = std::env::temp_dir().join(format!("lap_index_test_notracks_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Act
        let (tracks, warnings) = load_track_library(&dir).unwrap();

        // Assert
        assert!(tracks.is_empty());
        assert!(warnings.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_track_library_skips_an_unparseable_artifact_and_keeps_the_readable_ones() {
        // Arrange -- one valid `.idl0t`, one malformed.
        let dir = std::env::temp_dir().join(format!("lap_index_test_skip_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let tracks_dir = dir.join("tracks");
        fs::create_dir_all(&tracks_dir).unwrap();
        let good_json = r#"{"track_artifact_version":1,"track":{"track_id":"good","name":"n","created_at_ms":0,"updated_at_ms":0}}"#;
        fs::write(tracks_dir.join("good.idl0t"), good_json).unwrap();
        fs::write(tracks_dir.join("bad.idl0t"), b"not json").unwrap();

        // Act
        let (tracks, warnings) = load_track_library(&dir).unwrap();

        // Assert
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, "good");
        assert_eq!(warnings.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    /// Builds a handle whose GPS channels replay `fixes` verbatim (idiom
    /// shared with `laps::detect::tests`/`tracks::detect::tests`).
    fn handle_from_fixes(fixes: &[GpsFix]) -> SessionHandle {
        let meta = SessionMetaInput { session_id: String::new(), device_id: None, timestamp_utc_ms: 0, config_checksum: None };
        let lat: Vec<f64> = fixes.iter().map(|f| f.lat).collect();
        let lon: Vec<f64> = fixes.iter().map(|f| f.lon).collect();
        let epoch: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
        let ch = |id: &str, s: Vec<f64>| {
            let t_us = (0..s.len() as i64).map(|i| i * 1_000_000).collect();
            ChannelInput { channel_id: id.to_string(), sample_rate_hz: 1.0, samples: s, t_us, source_kind: id.to_lowercase() }
        };
        SessionHandle::from_channels(meta, vec![ch("GPS_Latitude", lat), ch("GPS_Longitude", lon), ch("GPS_EpochMs", epoch)])
    }

    /// Same fixture as [`handle_from_fixes`], but as an on-disk-writable
    /// [`Session`] (for [`write_session_parquet`]) rather than a
    /// [`SessionHandle`] — used by `reindex_laps`'s tests, which need a real
    /// `data.parquet` to read back.
    fn session_from_fixes(session_id: &str, fixes: &[GpsFix]) -> Session {
        let lat: Vec<f64> = fixes.iter().map(|f| f.lat).collect();
        let lon: Vec<f64> = fixes.iter().map(|f| f.lon).collect();
        let epoch: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
        let ch = |id: &str, s: Vec<f64>| {
            let t_us: Vec<i64> = (0..s.len() as i64).map(|i| i * 1_000_000).collect();
            Channel {
                channel_id: id.to_string(),
                t_us,
                t_recorded_us: None,
                nominal_rate_hz: 1.0,
                column: RawColumn::F64(s),
                source_kind: id.to_lowercase(),
                unit: String::new(),
                gaps: Vec::new(),
            }
        };
        Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Gpx,
            blob_sha256: String::new(),
            channels: vec![ch("GPS_Latitude", lat), ch("GPS_Longitude", lon), ch("GPS_EpochMs", epoch)],
        }
    }

    /// A there-and-back-and-there GPS track: 3 legs of 100 one-second fixes,
    /// lat sweeping 0.000→0.099→0.000→0.099 at a fixed longitude, crossing a
    /// gate at lat 0.05 three times (so a circuit start/finish gate there
    /// detects three laps). Long enough (300 s) to clear `min_visit_s`.
    fn three_lap_fixes() -> Vec<GpsFix> {
        let mut fixes = Vec::new();
        let leg = |t0: i64, up: bool| -> Vec<GpsFix> {
            (0..100)
                .map(|i| {
                    let lat = if up { i as f64 * 0.001 } else { 0.099 - i as f64 * 0.001 };
                    GpsFix { timestamp_ms: t0 + i * 1000, lat, lon: 0.0005 }
                })
                .collect()
        };
        fixes.extend(leg(0, true));
        fixes.extend(leg(100_000, false));
        fixes.extend(leg(200_000, true));
        fixes
    }

    /// A track whose reference polyline covers the same lat range/longitude
    /// as [`three_lap_fixes`], with a circuit start/finish gate at lat 0.05.
    fn circuit_track(id: &str) -> Track {
        let polyline: Vec<GpsFix> =
            (0..=100).map(|i| GpsFix { timestamp_ms: i * 1000, lat: i as f64 * 0.001, lon: 0.0005 }).collect();
        Track {
            id: id.to_string(),
            name: "Loop".to_string(),
            venue: String::new(),
            timing: Some(LapTiming::Circuit { start_finish: Gate { lat1: 0.05, lon1: -0.001, lat2: 0.05, lon2: 0.001 } }),
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: polyline,
            created_at_ms: 0,
            updated_at_ms: 1,
        }
    }

    #[test]
    fn compute_lap_index_one_circuit_track_lapped_three_times() {
        // Arrange
        let handle = handle_from_fixes(&three_lap_fixes());
        let tracks = vec![circuit_track("loop-1")];

        // Act
        let index = compute_lap_index(&handle, &tracks, &[]);

        // Assert -- one visit, three laps numbered 1..3, no warnings.
        assert_eq!(index.track_visits.len(), 1);
        assert_eq!(index.laps.len(), 3);
        assert_eq!(index.laps[0].lap_number, 1);
        assert_eq!(index.laps[1].lap_number, 2);
        assert_eq!(index.laps[2].lap_number, 3);
        assert!(index.warnings.is_empty());
        assert!(!index.track_library_hash.is_empty());
    }

    #[test]
    fn compute_lap_index_track_with_no_timing_records_visit_with_no_laps() {
        // Arrange -- same GPS/track geometry, but the track has no lap timing.
        let handle = handle_from_fixes(&three_lap_fixes());
        let mut track = circuit_track("loop-1");
        track.timing = None;
        let tracks = vec![track];

        // Act
        let index = compute_lap_index(&handle, &tracks, &[]);

        // Assert -- the visit is still recorded, with no laps and a warning.
        assert_eq!(index.track_visits.len(), 1);
        assert!(index.track_visits[0].laps.is_empty());
        assert!(index.laps.is_empty());
        assert_eq!(index.warnings.len(), 1);
    }

    #[test]
    fn resolve_visit_window_track_id_absent_from_library_records_visit_with_no_laps() {
        // Arrange -- a detected window naming a track id that isn't in the
        // library at all. This case is structurally unreachable through
        // `compute_lap_index` itself (its `refs` for `detect_visits` are
        // built from the same `tracks` slice this resolves against, so a
        // returned window's `track_id` is always present) -- tested directly
        // against the extracted resolver instead of contriving an
        // end-to-end scenario that cannot occur; see `resolve_visit`'s doc
        // comment.
        let handle = handle_from_fixes(&[]);
        let window = crate::tracks::VisitWindow {
            track_id: "loop-1".to_string(),
            start_timestamp_ms: 0,
            end_timestamp_ms: 1000,
        };
        let tracks = vec![track_stub("not-loop-1", 1)];

        // Act
        let (visit, warning) = resolve_visit(&handle, window, &tracks);

        // Assert -- the visit is still recorded, laps empty, one warning.
        assert_eq!(visit.track_id, "loop-1");
        assert!(visit.laps.is_empty());
        assert!(warning.is_some());
    }

    #[test]
    fn compute_lap_index_empty_track_library_is_all_empty_with_hash_set() {
        // Arrange
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        let index = compute_lap_index(&handle, &[], &[]);

        // Assert
        assert!(index.track_visits.is_empty());
        assert!(index.laps.is_empty());
        assert!(index.warnings.is_empty());
        assert!(!index.track_library_hash.is_empty());
    }

    #[test]
    fn compute_lap_index_two_visits_renumber_session_wide_but_keep_per_visit_numbering() {
        // Arrange -- track A gets one 3-lap visit at t=0..300s; track B gets a
        // separate, later single-lap visit, at a different longitude so the
        // two tracks don't overlap geometrically.
        let mut fixes = three_lap_fixes();
        let b_leg: Vec<GpsFix> =
            (0..100).map(|i| GpsFix { timestamp_ms: 400_000 + i * 1000, lat: i as f64 * 0.001, lon: 0.9005 }).collect();
        fixes.extend(b_leg);
        let handle = handle_from_fixes(&fixes);

        let track_a = circuit_track("track-a");
        let mut track_b = circuit_track("track-b");
        track_b.reference_polyline =
            (0..=100).map(|i| GpsFix { timestamp_ms: i * 1000, lat: i as f64 * 0.001, lon: 0.9005 }).collect();
        // The start/finish gate must sit at track B's own longitude, not
        // `circuit_track`'s default 0.0005 -- `find_crossings` tests each
        // consecutive GPS segment against the gate's own line segment, which
        // never crosses a fixed lat line far from where the GPS track runs.
        track_b.timing =
            Some(LapTiming::Circuit { start_finish: Gate { lat1: 0.05, lon1: 0.8995, lat2: 0.05, lon2: 0.9015 } });

        // Act
        let index = compute_lap_index(&handle, &[track_a, track_b], &[]);

        // Assert -- two visits; four laps total, top-level numbered 1..4 in
        // start-time order; each visit's own laps restart at 1.
        assert_eq!(index.track_visits.len(), 2);
        assert_eq!(index.laps.len(), 4);
        for (i, lap) in index.laps.iter().enumerate() {
            assert_eq!(lap.lap_number, (i + 1) as u32);
        }
        let visit_a = index.track_visits.iter().find(|v| v.track_id == "track-a").unwrap();
        assert_eq!(visit_a.laps.len(), 3);
        assert_eq!(visit_a.laps[0].lap_number, 1);
        let visit_b = index.track_visits.iter().find(|v| v.track_id == "track-b").unwrap();
        assert_eq!(visit_b.laps.len(), 1);
        assert_eq!(visit_b.laps[0].lap_number, 1);
    }

    #[test]
    fn index_laps_fresh_session_no_session_json_writes_one_with_laps_and_stamps() {
        // Arrange -- one circuit track, lapped three times, no session.json yet.
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert!(!report.skipped_up_to_date);
        assert_eq!(report.visits_indexed, 1);
        assert_eq!(report.laps_indexed, 3);
        let doc = read_session_json(&sj_path(&root, session_id)).unwrap();
        assert_eq!(doc.laps.len(), 3);
        assert!(doc.track_visits_library_hash.is_some());
        assert_eq!(doc.lap_detector_version.as_deref(), Some(LAP_DETECTOR_VERSION));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_existing_session_json_carries_unrelated_fields_through_verbatim() {
        // Arrange -- rider/bike/comments already set before indexing runs.
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let mut doc = empty_session_json(session_id);
        doc.rider = "Isaac".to_string();
        doc.bike = "SV650".to_string();
        doc.bike_comment = "new forks".to_string();
        write_session_json(&root, session_id, &doc, None).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert -- unrelated fields untouched, laps populated.
        assert_eq!(report.laps_indexed, 3);
        let after = read_session_json(&sj_path(&root, session_id)).unwrap();
        assert_eq!(after.rider, "Isaac");
        assert_eq!(after.bike, "SV650");
        assert_eq!(after.bike_comment, "new forks");
        assert_eq!(after.laps.len(), 3);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_second_call_unchanged_library_skips_and_leaves_file_untouched() {
        // Arrange
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());
        index_laps(&root, session_id, &handle, false).unwrap();
        let path = sj_path(&root, session_id);
        let bytes_before = fs::read(&path).unwrap();

        // Act -- same library, no force.
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert!(report.skipped_up_to_date);
        assert_eq!(fs::read(&path).unwrap(), bytes_before);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_second_call_after_track_updated_at_ms_changes_recomputes() {
        // Arrange
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());
        index_laps(&root, session_id, &handle, false).unwrap();
        let mut bumped = circuit_track("loop-1");
        bumped.updated_at_ms = 999;
        write_track(&root, &bumped).unwrap();

        // Act -- the library hash changed, so this recomputes despite force=false.
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert!(!report.skipped_up_to_date);
        assert_eq!(report.laps_indexed, 3);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_second_call_with_stale_detector_version_recomputes() {
        // Arrange -- the library is unchanged, but the stamped detector
        // version does not match LAP_DETECTOR_VERSION.
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());
        index_laps(&root, session_id, &handle, false).unwrap();
        let path = sj_path(&root, session_id);
        let mut doc = read_session_json(&path).unwrap();
        doc.lap_detector_version = Some("0".to_string());
        let based_on = sha256_hex(&fs::read(&path).unwrap());
        write_session_json(&root, session_id, &doc, Some(&based_on)).unwrap();

        // Act
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert!(!report.skipped_up_to_date);
        let after = read_session_json(&path).unwrap();
        assert_eq!(after.lap_detector_version.as_deref(), Some(LAP_DETECTOR_VERSION));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_force_true_recomputes_even_when_stamp_is_current() {
        // Arrange
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());
        index_laps(&root, session_id, &handle, false).unwrap();

        // Act
        let report = index_laps(&root, session_id, &handle, true).unwrap();

        // Assert
        assert!(!report.skipped_up_to_date);
        assert_eq!(report.laps_indexed, 3);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_unresolvable_lap_flags_are_cleared_and_reported() {
        // Arrange -- main_lap_number names a lap that will not exist once
        // only 3 laps are detected; ignored_lap_numbers names one that
        // survives (1) and one that does not (9).
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let mut doc = empty_session_json(session_id);
        doc.main_lap_number = Some(9);
        doc.ignored_lap_numbers = vec![1, 9];
        write_session_json(&root, session_id, &doc, None).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert_eq!(report.laps_indexed, 3);
        assert!(report.flags_cleared.contains(&"main_lap_number".to_string()));
        assert!(report.flags_cleared.contains(&"ignored_lap_numbers".to_string()));
        let after = read_session_json(&sj_path(&root, session_id)).unwrap();
        assert_eq!(after.main_lap_number, None);
        assert_eq!(after.ignored_lap_numbers, vec![1]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_overlay_lap_key_survives_untouched() {
        // Arrange -- overlay_lap_key names a lap in a different session.
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let mut doc = empty_session_json(session_id);
        doc.overlay_lap_key = Some(OverlayLapKeyJson { session_id: "other".to_string(), lap_number: 5 });
        write_session_json(&root, session_id, &doc, None).unwrap();
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        let after = read_session_json(&sj_path(&root, session_id)).unwrap();
        assert_eq!(after.overlay_lap_key, Some(OverlayLapKeyJson { session_id: "other".to_string(), lap_number: 5 }));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_laps_empty_track_library_writes_honest_empty_with_stamps_set() {
        // Arrange -- no tracks/ directory at all.
        let root = temp_root();
        let session_id = "s1";
        let handle = handle_from_fixes(&three_lap_fixes());

        // Act
        let report = index_laps(&root, session_id, &handle, false).unwrap();

        // Assert
        assert!(!report.skipped_up_to_date);
        assert_eq!(report.visits_indexed, 0);
        assert_eq!(report.laps_indexed, 0);
        let doc = read_session_json(&sj_path(&root, session_id)).unwrap();
        assert!(doc.laps.is_empty());
        assert!(doc.track_visits.is_empty());
        assert!(doc.track_visits_library_hash.is_some());
        assert_eq!(doc.lap_detector_version.as_deref(), Some(LAP_DETECTOR_VERSION));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn reindex_laps_reproduces_index_laps_and_errors_typed_on_missing_data_parquet() {
        // Arrange -- a real data.parquet, read back and re-indexed.
        let root = temp_root();
        let session_id = "s1";
        write_track(&root, &circuit_track("loop-1")).unwrap();
        let session = session_from_fixes(session_id, &three_lap_fixes());
        write_session_parquet(&root, &session, "test-importer").unwrap();

        // Act
        let report = reindex_laps(&root, session_id).unwrap();

        // Assert -- same result compute_lap_index/index_laps produce directly
        // from the equivalent in-memory handle.
        assert!(!report.skipped_up_to_date);
        assert_eq!(report.visits_indexed, 1);
        assert_eq!(report.laps_indexed, 3);

        // Act -- no data.parquet for this session at all.
        let err = reindex_laps(&root, "nope").unwrap_err();

        // Assert -- typed Io error naming the path, not a panic.
        assert_eq!(err.kind, LapIndexErrorKind::Io);
        assert!(err.message.contains("data.parquet"));

        let _ = fs::remove_dir_all(&root);
    }
}
