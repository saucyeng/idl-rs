//! The `track` and `catalog` verbs (ruling R229): `track list`,
//! `track detect`, `catalog verify`, `catalog rebuild`.
//!
//! `track detect` re-runs lap indexing through `lap_index::reindex_laps`.

use clap::ArgMatches;
use serde_json::{json, Value};

use idl_library::catalog::{rebuild_catalog, RebuildReport};
use idl_library::catalog_read::TrackSummary;
use idl_rs::track_artifact::path::{list_track_files, tracks_dir};
use idl_rs::track_artifact::read::read_track;
use idl_library::lap_index::{self, LapIndexReport};
use idl_library::verify::{self, Finding, Severity};

use crate::envelope::CliError;
use crate::verbs::{text, Ctx, VerbOutput};

/// `track list` — the tracks in the library, read from the track files in
/// `<data>/tracks/` (`.idl1t`, else legacy `.idl0t`) rather than the catalog,
/// which is only an index and may not have seen a new file yet (CLAUDE.md §3).
pub fn track_list(ctx: &Ctx, _m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;

    let tracks = list_tracks_from_files(root)?;

    let text = tracks
        .iter()
        .map(|t| format!("{}  {}  {}", t.track_id, t.name, t.venue_name))
        .chain(std::iter::once(format!("({} track(s))", tracks.len())))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(VerbOutput::new(
        text,
        json!({ "tracks": tracks.iter().map(track_json).collect::<Vec<_>>() }),
    ))
}

/// Every readable track file in `<root>/tracks/` as a [`TrackSummary`],
/// ordered by name like the catalog's `list_tracks`. A file that does not
/// parse is skipped, as the catalog rebuild skips it.
fn list_tracks_from_files(root: &std::path::Path) -> Result<Vec<TrackSummary>, CliError> {
    let dir = tracks_dir(root);
    let files = list_track_files(&dir).map_err(|e| CliError::io(format!("reading {}: {e}", dir.display())))?;
    let mut tracks: Vec<TrackSummary> = files
        .iter()
        .filter_map(|path| read_track(path).ok())
        .map(|t| TrackSummary {
            track_id: t.id,
            name: t.name,
            venue_name: t.venue,
            created_at_ms: t.created_at_ms,
            updated_at_ms: t.updated_at_ms,
        })
        .collect();
    tracks.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.track_id.cmp(&b.track_id)));
    Ok(tracks)
}

/// `track detect` — re-run lap indexing for one session.
pub fn track_detect(ctx: &Ctx, m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;
    let id = text(m, "id")?;

    if ctx.dry_run {
        // `reindex_laps` writes `session.json` unconditionally when it
        // recomputes, and there is no read-only form of it in core, so a dry
        // run reports the intent rather than half-running the detector.
        return Ok(VerbOutput::new(
            format!("would re-detect laps for {id}"),
            json!({ "session_id": id, "written": false }),
        ));
    }

    let report = lap_index::reindex_laps(root, &id)?;

    let text = if report.skipped_up_to_date {
        format!("{}: up to date, nothing recomputed", report.session_id)
    } else {
        format!(
            "{}: {} visit(s), {} lap(s)",
            report.session_id, report.visits_indexed, report.laps_indexed
        )
    };

    Ok(VerbOutput::new(text, index_json(&report)))
}

/// `catalog verify` — contract C4 §7's checks.
///
/// Exits `1` when any finding is an error, so a script can gate on it.
pub fn verify(ctx: &Ctx, _m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;

    let findings = verify::verify(root);

    let errors = findings.iter().filter(|f| f.severity == Severity::Error).count();
    let text = findings
        .iter()
        .map(|f| format!("[{:?}] {}: {}", f.severity, f.path.display(), f.message))
        .chain(std::iter::once(format!("({} finding(s), {errors} error(s))", findings.len())))
        .collect::<Vec<_>>()
        .join("\n");

    let data = json!({
        "findings": findings.iter().map(finding_json).collect::<Vec<_>>(),
        "error_count": errors,
    });

    if errors > 0 {
        // A failed verification is a failed operation, not a broken tool:
        // the findings are the useful output, so they go in `details`.
        return Err(CliError::with_details(
            crate::envelope::ErrorKind::InvalidInput,
            format!("{errors} error-severity finding(s)"),
            data,
        ));
    }

    Ok(VerbOutput::new(text, data))
}

/// `catalog rebuild` — rebuild the index from the canonical files.
pub fn rebuild(ctx: &Ctx, _m: &ArgMatches) -> Result<VerbOutput, CliError> {
    let root = ctx.data_dir()?;

    if ctx.dry_run {
        // The catalog is an index — deletable and rebuildable — so there is
        // nothing to preview beyond saying which directory would be reindexed.
        return Ok(VerbOutput::new(
            format!("would rebuild the catalog under {}", root.display()),
            json!({ "data_dir": root.display().to_string(), "written": false }),
        ));
    }

    let report = rebuild_catalog(root)?;

    let text = format!(
        "{} session(s), {} blob(s) ({} carried, {} hashed), {} lap(s), {} skipped",
        report.sessions_indexed,
        report.blobs_indexed,
        report.blobs_carried,
        report.blobs_hashed,
        report.laps_indexed,
        report.skipped.len()
    );

    Ok(VerbOutput::new(text, rebuild_json(&report)))
}

// ---------------------------------------------------------------------------
// Projections.
// ---------------------------------------------------------------------------

/// C3's `TrackSummary` shape.
fn track_json(track: &TrackSummary) -> Value {
    json!({
        "track_id": track.track_id,
        "name": track.name,
        "venue_name": track.venue_name,
        "created_at_ms": track.created_at_ms,
        "updated_at_ms": track.updated_at_ms,
    })
}

/// `LapIndexReport` as the `track detect` JSON object.
fn index_json(report: &LapIndexReport) -> Value {
    json!({
        "session_id": report.session_id,
        "visits_indexed": report.visits_indexed,
        "laps_indexed": report.laps_indexed,
        "skipped_up_to_date": report.skipped_up_to_date,
        "flags_cleared": report.flags_cleared,
        "warnings": report.warnings,
        "written": !report.skipped_up_to_date,
    })
}

/// One `verify` finding.
fn finding_json(finding: &Finding) -> Value {
    json!({
        "severity": format!("{:?}", finding.severity).to_lowercase(),
        "path": finding.path.display().to_string(),
        "message": finding.message,
    })
}

/// `RebuildReport`.
fn rebuild_json(report: &RebuildReport) -> Value {
    json!({
        "blobs_indexed": report.blobs_indexed,
        "blobs_carried": report.blobs_carried,
        "blobs_hashed": report.blobs_hashed,
        "tracks_indexed": report.tracks_indexed,
        "sessions_indexed": report.sessions_indexed,
        "laps_indexed": report.laps_indexed,
        "skipped": report.skipped,
        "written": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn a_finding_serialises_its_severity_in_lower_case() {
        // Arrange
        let finding = Finding {
            severity: Severity::Error,
            path: PathBuf::from("blobs/sha256/aa/bb"),
            message: "content hash mismatch".to_string(),
        };

        // Act
        let value = finding_json(&finding);

        // Assert
        assert_eq!(value["severity"], "error");
        assert_eq!(value["message"], "content hash mismatch");
    }

    #[test]
    fn an_up_to_date_index_report_says_nothing_was_written() {
        // Arrange
        let report = LapIndexReport {
            session_id: "s1".to_string(),
            visits_indexed: 0,
            laps_indexed: 0,
            skipped_up_to_date: true,
            flags_cleared: Vec::new(),
            warnings: Vec::new(),
        };

        // Act
        let value = index_json(&report);

        // Assert — `written` is what a caller checks before re-syncing.
        assert_eq!(value["written"], false);
    }

    #[test]
    fn a_recomputed_index_report_says_something_was_written() {
        // Arrange
        let report = LapIndexReport {
            session_id: "s1".to_string(),
            visits_indexed: 2,
            laps_indexed: 11,
            skipped_up_to_date: false,
            flags_cleared: Vec::new(),
            warnings: Vec::new(),
        };

        // Act
        let value = index_json(&report);

        // Assert
        assert_eq!(value["written"], true);
        assert_eq!(value["laps_indexed"], 11);
    }

    #[test]
    fn list_tracks_from_files_new_current_and_legacy_files_appear_without_a_catalog_rebuild() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-rs-track-list-{}", std::process::id()));
        let dir = root.join("tracks");
        std::fs::create_dir_all(&dir).unwrap();
        let body = |id: &str, name: &str| {
            format!(
                r#"{{"track_artifact_version":1,"track":{{"track_id":"{id}","name":"{name}","venue_name":"Whistler","created_at_ms":1,"updated_at_ms":2}}}}"#
            )
        };
        std::fs::write(dir.join("t-b.idl1t"), body("t-b", "B-Line")).unwrap();
        std::fs::write(dir.join("t-a.idl0t"), body("t-a", "A-Line")).unwrap();
        std::fs::write(dir.join("broken.idl1t"), b"not json").unwrap();

        // Act
        let tracks = list_tracks_from_files(&root).unwrap();

        // Assert
        let ids: Vec<&str> = tracks.iter().map(|t| t.track_id.as_str()).collect();
        assert_eq!(ids, vec!["t-a", "t-b"]);
        assert_eq!(tracks[0].venue_name, "Whistler");
        assert!(!root.join("catalog.sqlite").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn list_tracks_from_files_no_tracks_directory_is_an_empty_list() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-rs-track-list-none-{}", std::process::id()));

        // Act
        let tracks = list_tracks_from_files(&root).unwrap();

        // Assert
        assert!(tracks.is_empty());
    }
}
