//! `verify` (contract C4 §7): walks `<data>`, reports every check as a
//! [`Finding`], never auto-repairs structured content. Scope is `<data>`
//! only — `app_config_dir()/settings.json` is outside it on every platform
//! (C4 §1) and is never scanned.

use std::path::{Path, PathBuf};

/// How serious a [`Finding`] is (C4 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// One `verify` observation: never a repair action, only a report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub severity: Severity,
    pub path: PathBuf,
    pub message: String,
}

/// Runs every C4 §7 check against `<data_root>`, in the contract's order.
/// Implements #1, #2, #3, #4, #5, #8, #10. Checks #6 (derived-file orphan
/// detection), #7 (workbook parse) and #9 (catalog-row-points-at-missing-
/// file) are **not implemented by this task** — #6 needs the session's
/// current `data.parquet` plus the *live* estimator config to recompute the
/// expected hash (materialisation-time knowledge this store module has no
/// generic way to know; left to the estimator glue that calls
/// `store::derived::write_derived_parquet`, which already has everything
/// needed to recompute and compare), #7 needs C2/L3's `.idl1wb` parser (out
/// of L1's scope), #9 needs a live catalog connection cross-referenced
/// against every table's file-reference columns (mechanically
/// straightforward once L1's catalog, Task 12, is the only writer, but not
/// written here to keep this task's own scope to what L1 alone can verify
/// without guessing at another lane's not-yet-existing format).
///
/// A malformed `blobs/sha256/*/*` filename (not a well-formed 64-hex digest)
/// can be reported twice: once by #1 ([`check_blobs`], whose reconstructed
/// path fails content/length verification) and once by #10
/// ([`check_unexpected_paths`], whose [`matches_layout`] rejects the
/// non-conforming name). Both facts are independently true and at different
/// severities — this overlap is intentional, not deduplicated.
pub fn verify(data_root: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();
    check_blobs(data_root, &mut findings); // #1
    check_sessions(data_root, &mut findings); // #2, #3, #4
    check_derived(data_root, &mut findings); // #5 (orphan detection, #6, deferred — see doc comment)
    check_tracks(data_root, &mut findings); // #8
    check_unexpected_paths(data_root, &mut findings); // #10
    findings
}

fn check_blobs(data_root: &Path, out: &mut Vec<Finding>) {
    let dir = data_root.join("blobs").join("sha256");
    let Ok(shards) = std::fs::read_dir(&dir) else { return };
    for shard in shards.flatten() {
        let Ok(entries) = std::fs::read_dir(shard.path()) else { continue };
        for entry in entries.flatten() {
            let prefix = shard.file_name().to_string_lossy().into_owned();
            let suffix = entry.file_name().to_string_lossy().into_owned();
            let digest = format!("{prefix}{suffix}");
            if let Err(e) = crate::store::blob::verify_blob(data_root, &digest) {
                out.push(Finding { severity: Severity::Error, path: entry.path(), message: e.to_string() });
            }
        }
    }
}

fn check_sessions(data_root: &Path, out: &mut Vec<Finding>) {
    let dir = data_root.join("sessions");
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for entry in entries.flatten() {
        let session_dir = entry.path();
        if !session_dir.is_dir() {
            continue;
        }
        let dir_session_id = entry.file_name().to_string_lossy().into_owned();
        let sj_path = session_dir.join("session.json");
        if !sj_path.is_file() {
            continue; // transient import state, C4 §2 — not corruption
        }
        if let Err(e) = crate::store::session_json::read_session_json(&sj_path) {
            out.push(Finding { severity: Severity::Error, path: sj_path, message: e.to_string() }); // #2
        }

        let dp_path = session_dir.join("data.parquet");
        if !dp_path.is_file() {
            continue; // transient import state, C4 §2 — not corruption (also, #3/#4 need data.parquet)
        }
        match crate::store::catalog::read_data_parquet_session_fields(&dp_path) {
            Ok(fields) => {
                if fields.session_id != dir_session_id {
                    out.push(Finding {
                        severity: Severity::Error,
                        path: dp_path.clone(),
                        message: format!(
                            "data.parquet metadata session_id {} does not match directory name {}",
                            fields.session_id, dir_session_id
                        ),
                    }); // #4
                }
                if !crate::store::blob::blob_exists(data_root, &fields.blob_sha256) {
                    out.push(Finding {
                        severity: Severity::Warning,
                        path: dp_path,
                        message: format!(
                            "blob {} unavailable — re-import or re-sync from a peer that still has it",
                            fields.blob_sha256
                        ),
                    }); // #3
                }
            }
            // A data.parquet that exists but whose own file-level metadata
            // can't even be read is exactly #4's own example of a
            // "corrupted file, e.g. an interrupted sync" — session_id can't
            // be compared, so this is reported as the identity check's
            // failure rather than silently skipped.
            Err(e) => out.push(Finding { severity: Severity::Error, path: dp_path, message: e.to_string() }), // #4
        }
    }
}

fn check_derived(data_root: &Path, out: &mut Vec<Finding>) {
    let sessions_dir = data_root.join("sessions");
    let Ok(sessions) = std::fs::read_dir(&sessions_dir) else { return };
    for session in sessions.flatten() {
        let derived_dir = session.path().join("derived");
        let Ok(entries) = std::fs::read_dir(&derived_dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let actual = crate::store::atomic::sha256_hex(&bytes);
            if actual != stem {
                out.push(Finding {
                    severity: Severity::Error,
                    message: format!("derived filename {stem} does not match its own content hash {actual}"),
                    path,
                }); // #5
            }
            // Orphan detection (#6, info-severity) needs the session's
            // current data.parquet + the live estimator config to
            // recompute the expected hash — this is a materialisation-time
            // concern (which estimator, which config) this store module has
            // no way to know generically; left to the estimator glue that
            // calls store::derived::write_derived_parquet in the first
            // place (it already has everything needed to recompute and
            // compare). Flagged, not silently dropped — see this fn's doc.
        }
    }
}

fn check_tracks(data_root: &Path, out: &mut Vec<Finding>) {
    let dir = data_root.join("tracks");
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl0t") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        match crate::track_artifact::read::read_track(&path) {
            Ok(track) if track.id == stem => {}
            Ok(_) => out.push(Finding {
                severity: Severity::Error,
                path,
                message: "filename does not match the track_id inside its own JSON".to_string(),
            }),
            Err(e) => out.push(Finding { severity: Severity::Error, path, message: e.to_string() }),
        }
    }
}

/// Whether `rel` (a path relative to `<data_root>`) is part of C4 §2's known
/// layout — either a file matching one of the fixed patterns
/// (`blobs/sha256/<2 hex>/<62 hex>`, `sessions/<id>/session.json`,
/// `sessions/<id>/data.parquet`, `sessions/<id>/derived/<64 hex>.parquet`,
/// `workbooks/*.idl1wb`, `tracks/*.idl0t`, `profiles/*.idl0p`,
/// `catalog.sqlite(-wal|-shm)`, anything under `tmp/`), or a directory that
/// is a legitimate ancestor of one (e.g. `sessions`, `sessions/<id>`,
/// `sessions/<id>/derived`, `blobs/sha256/<2 hex>`). Used only by
/// [`check_unexpected_paths`] (#10) — never by identity/content checks.
fn matches_layout(rel: &Path) -> bool {
    let parts: Vec<&str> = rel.iter().map(|c| c.to_str().unwrap_or("")).collect();
    match parts.as_slice() {
        [] => true, // <data_root> itself
        ["tmp"] => true,
        ["tmp", ..] => true, // anything under tmp/ (C4 §2)
        ["blobs"] => true,
        ["blobs", "sha256"] => true,
        ["blobs", "sha256", shard] => is_lower_hex(shard, 2),
        ["blobs", "sha256", shard, name] => is_lower_hex(shard, 2) && is_lower_hex(name, 62),
        ["sessions"] => true,
        ["sessions", _id] => true,
        ["sessions", _id, "session.json"] => true,
        ["sessions", _id, "data.parquet"] => true,
        ["sessions", _id, "derived"] => true,
        ["sessions", _id, "derived", name] => {
            name.strip_suffix(".parquet").is_some_and(|stem| is_lower_hex(stem, 64))
        }
        ["workbooks"] => true,
        ["workbooks", name] => name.ends_with(".idl1wb"),
        ["tracks"] => true,
        ["tracks", name] => name.ends_with(".idl0t"),
        ["profiles"] => true,
        ["profiles", name] => name.ends_with(".idl0p"),
        ["catalog.sqlite"] => true,
        ["catalog.sqlite-wal"] => true,
        ["catalog.sqlite-shm"] => true,
        _ => false,
    }
}

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Runs [`verify`] and then repairs C4 §7's two auto-repairable findings —
/// a corrupted blob (#1) or a corrupted derived parquet (#5) — by moving
/// each into quarantine via [`crate::store::quarantine::quarantine_file`].
/// `verify` itself is unchanged and stays read-only; this is the only
/// caller of quarantine's repair path (C4 §7, ruling R86 Q1/Q8 —
/// `verify_data_dir(repair: true)`, C3 §3.10). `ids` mints one `entry_id`
/// per repair and `now_ms` timestamps every repair; both are injected so
/// this function has no clock and no randomness of its own — core never
/// generates uuids or reads the clock (CLAUDE.md §2, this lane's brief).
///
/// Findings #2, #3, #4, #7, #8, #9, #10 are never quarantined here — C4 §7
/// names them surfaced-only (structured content, a missing file, a stale
/// catalog row, or a bystander path) where an automatic move risks losing
/// information a human or a migration tool needs. Which findings are #1/#5
/// is decided structurally, from each finding's own path shape (the same
/// C4 §2 shapes [`matches_layout`] already knows), not by matching message
/// text, so a look-alike message on an unrelated path can never be
/// mis-repaired.
pub fn verify_and_repair(
    data_root: &Path,
    ids: &mut dyn FnMut() -> String,
    now_ms: i64,
) -> (Vec<Finding>, Vec<crate::store::quarantine::QuarantineEntry>) {
    let findings = verify(data_root);
    let mut quarantined = Vec::new();

    for finding in &findings {
        if finding.severity != Severity::Error || !is_repairable_finding_path(data_root, &finding.path) {
            continue;
        }
        let entry_id = ids();
        if let Ok(entry) = crate::store::quarantine::quarantine_file(data_root, &finding.path, &finding.message, &entry_id, now_ms)
        {
            quarantined.push(entry);
        }
        // A failed repair (e.g. the path was already moved by an earlier
        // finding on the same file) is not surfaced as a second error here
        // — the finding itself, already in `findings`, is the report.
    }

    (findings, quarantined)
}

/// Whether `path` (absolute, taken from a [`Finding`]) is one of C4 §7's
/// two auto-repairable shapes: a blob under `blobs/sha256/<2 hex>/<62
/// hex>` (finding #1) or a `sessions/<id>/derived/<64 hex>.parquet`
/// (finding #5).
fn is_repairable_finding_path(data_root: &Path, path: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(data_root) else { return false };
    let parts: Vec<&str> = rel.iter().map(|c| c.to_str().unwrap_or("")).collect();
    match parts.as_slice() {
        ["blobs", "sha256", shard, name] => is_lower_hex(shard, 2) && is_lower_hex(name, 62),
        ["sessions", _id, "derived", name] => name.strip_suffix(".parquet").is_some_and(|stem| is_lower_hex(stem, 64)),
        _ => false,
    }
}

fn check_unexpected_paths(data_root: &Path, out: &mut Vec<Finding>) {
    walk_unexpected(data_root, data_root, out);
}

/// Recurses through `<data_root>`, reporting one Info [`Finding`] per path
/// that [`matches_layout`] rejects. Legitimate directories (including
/// session directories, which always exist regardless of what junk might
/// also be inside them) are descended into; an already-flagged unexpected
/// directory is not, to avoid one finding per file underneath it.
fn walk_unexpected(data_root: &Path, dir: &Path, out: &mut Vec<Finding>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(data_root) else { continue };
        let Ok(file_type) = entry.file_type() else { continue };
        if matches_layout(rel) {
            if file_type.is_dir() {
                walk_unexpected(data_root, &path, out);
            }
            continue;
        }
        out.push(Finding {
            severity: Severity::Info,
            path,
            message: "path under <data> matches none of the C4 §2 patterns".to_string(),
        }); // #10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Channel, RawColumn, Session, SourceFormat};
    use crate::store::blob::write_blob;
    use crate::store::derived::{write_derived_parquet, DerivedOutput};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{empty_session_json, write_session_json};
    use crate::track_artifact::model::Track;
    use crate::track_artifact::write::write_track;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a minimal but real session (`session.json` + `data.parquet`,
    /// whose file metadata names `blob_sha256`) under `<root>/sessions/<id>`.
    fn write_full_session_with_blob(root: &Path, session_id: &str, blob_sha256: String) {
        let doc = empty_session_json(session_id);
        write_session_json(root, session_id, &doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256,
            channels: vec![Channel {
                channel_id: "IMU0_AccelX".to_string(),
                t_us: vec![0, 500_000, 1_000_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0, 3.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn verify_clean_tree_reports_nothing() {
        // Arrange
        let root = temp_root();
        write_blob(&root, b"clean blob").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert!(findings.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_detects_a_corrupted_blob() {
        // Arrange
        let root = temp_root();
        let digest = write_blob(&root, b"original").unwrap();
        std::fs::write(crate::store::blob::blob_path(&root, &digest), b"corrupted").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_reports_an_unrecognised_top_level_path_as_info() {
        // Arrange
        let root = temp_root();
        std::fs::write(root.join("Thumbs.db"), b"").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Info);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_reports_unexpected_paths_recursively_including_nested_ones() {
        // Arrange — one unexpected path at top level, one nested inside an
        // otherwise-legitimate session directory.
        let root = temp_root();
        std::fs::write(root.join("Thumbs.db"), b"").unwrap();
        let session_dir = root.join("sessions").join("s1");
        std::fs::create_dir_all(&session_dir).unwrap();
        std::fs::write(session_dir.join(".DS_Store"), b"").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 2);
        assert!(findings.iter().all(|f| f.severity == Severity::Info));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_reports_a_missing_blob_as_a_single_warning() {
        // Arrange — session.json + data.parquet exist, but the blob its
        // data.parquet names was never written.
        let root = temp_root();
        write_full_session_with_blob(&root, "s1", "0".repeat(64));

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Warning);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_detects_a_malformed_session_json_as_an_error() {
        // Arrange — session.json exists but is not valid JSON; no
        // data.parquet yet, so only #2 fires (#3/#4 both need data.parquet).
        let root = temp_root();
        let session_dir = root.join("sessions").join("s1");
        std::fs::create_dir_all(&session_dir).unwrap();
        std::fs::write(session_dir.join("session.json"), b"not json").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);
        assert_eq!(findings[0].path, session_dir.join("session.json"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_detects_a_derived_file_whose_content_does_not_match_its_filename_hash() {
        // Arrange — a real derived/<hash>.parquet, then its content is
        // overwritten in place (filename, hence claimed hash, unchanged) —
        // same corruption shape as `verify_detects_a_corrupted_blob`.
        let root = temp_root();
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 500_000],
            values: vec![1.0, 2.0],
            nominal_rate_hz: 2.0,
            unit: "deg".to_string(),
        }];
        let path = write_derived_parquet(&root, "s1", "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();
        std::fs::write(&path, b"corrupted").unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_detects_a_track_file_whose_filename_does_not_match_its_track_id() {
        // Arrange — write a valid `.idl0t` for "t-1", then rename it so the
        // filename no longer matches the `track_id` inside its own JSON.
        let root = temp_root();
        let track = Track {
            id: "t-1".to_string(),
            name: "A-Line".to_string(),
            venue: "Whistler".to_string(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 1,
            updated_at_ms: 2,
        };
        let path = write_track(&root, &track).unwrap();
        let renamed = path.parent().unwrap().join("t-2.idl0t");
        std::fs::rename(&path, &renamed).unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_reports_a_data_parquet_session_id_mismatch_as_an_error() {
        // Arrange — data.parquet's own file metadata says session_id "s1",
        // but the directory it lives in is renamed to "s2" afterward.
        let root = temp_root();
        let blob_sha256 = write_blob(&root, b"raw bytes").unwrap();
        write_full_session_with_blob(&root, "s1", blob_sha256);
        std::fs::rename(root.join("sessions").join("s1"), root.join("sessions").join("s2")).unwrap();

        // Act
        let findings = verify(&root);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Deterministic `ids` closure for [`verify_and_repair`] tests: hands
    /// out fixed, distinguishable ids of the same 36-char length a real
    /// uuid has (core mints neither uuids nor clock reads on its own — the
    /// caller injects both).
    fn fixed_ids(seed: &'static str) -> impl FnMut() -> String {
        let mut n = 0;
        move || {
            n += 1;
            format!("{seed}{n:0>35}")
        }
    }

    #[test]
    fn verify_and_repair_a_blob_whose_bytes_do_not_match_its_path_the_blob_is_quarantined_and_the_finding_is_still_returned(
    ) {
        // Arrange
        let root = temp_root();
        let digest = write_blob(&root, b"original").unwrap();
        let blob_path = crate::store::blob::blob_path(&root, &digest);
        std::fs::write(&blob_path, b"corrupted").unwrap();
        let mut ids = fixed_ids("1");

        // Act
        let (findings, quarantined) = verify_and_repair(&root, &mut ids, 42);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);
        assert_eq!(quarantined.len(), 1);
        assert!(!blob_path.exists());
        assert_eq!(quarantined[0].quarantined_at_ms, 42);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_and_repair_a_healthy_tree_no_quarantine_entries_and_the_directory_stays_empty() {
        // Arrange
        let root = temp_root();
        write_blob(&root, b"clean blob").unwrap();
        let mut ids = fixed_ids("2");

        // Act
        let (findings, quarantined) = verify_and_repair(&root, &mut ids, 1);

        // Assert
        assert!(findings.is_empty());
        assert!(quarantined.is_empty());
        assert!(crate::store::quarantine::list_quarantine(&root).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_and_repair_a_missing_blob_finding_3_not_quarantined() {
        // Arrange — session.json + data.parquet exist, but the blob its
        // data.parquet names was never written (warning-severity #3).
        let root = temp_root();
        write_full_session_with_blob(&root, "s1", "0".repeat(64));
        let mut ids = fixed_ids("3");

        // Act
        let (findings, quarantined) = verify_and_repair(&root, &mut ids, 1);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert!(quarantined.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_and_repair_a_malformed_session_json_error_severity_finding_2_not_quarantined() {
        // Arrange — session.json is unreadable JSON (error-severity #2),
        // a path shape `is_repairable_finding_path` never matches (it only
        // matches blob and derived-parquet shapes) — this exercises the
        // structural guard on an Error-severity finding, unlike the sibling
        // "missing blob" test above, which is excluded one guard earlier by
        // being Warning-severity.
        let root = temp_root();
        let session_dir = root.join("sessions").join("s1");
        std::fs::create_dir_all(&session_dir).unwrap();
        std::fs::write(session_dir.join("session.json"), b"not json").unwrap();
        let mut ids = fixed_ids("4");

        // Act
        let (findings, quarantined) = verify_and_repair(&root, &mut ids, 1);

        // Assert
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, Severity::Error);
        assert!(quarantined.is_empty());
        assert!(session_dir.join("session.json").exists());

        let _ = std::fs::remove_dir_all(&root);
    }
}
