//! The C4 §6 sync manifest: the typed document `GET /idl1/v1/manifest`
//! answers, and the pure `std::fs` walk of `<data>` that builds it. No
//! network, no async, no clock and no randomness of its own — `now_ms` is
//! injected by the command layer (CLAUDE.md §2). The diff/merge logic that
//! consumes this document is a later task in this lane.

use std::fmt;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::store::atomic::sha256_hex;
use crate::track_artifact::read::read_track;
use crate::workbook::v3::front_matter::parse_front_matter;

/// One content-addressed blob's manifest entry (C4 §6). `sha256` is both
/// the identity key and the CAS path (`blobs/sha256/<2>/<62>`) — no
/// conflict is possible for this class, sync is a set difference by hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobEntry {
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// One `sessions/<id>/derived/<sha256>.parquet` manifest entry — content-
/// addressed like [`BlobEntry`], never overwritten (C4 §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedEntry {
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// One `sessions/<id>/data.parquet` manifest entry. `importer_version` and
/// `seam_correction_version` are the conflict key (C4 §6); `engine_version`
/// is informational provenance only (C1 §4.3), never part of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataParquetEntry {
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SemVer 2.0.0 string — the importer module that produced this file
    /// (C1 §4.3).
    pub importer_version: String,
    /// The seam-correction algorithm version applied (C1 §4.3), e.g. `v1`.
    pub seam_correction_version: String,
    /// SemVer 2.0.0 string — `idl-rs`'s own build version (C1 §4.3).
    /// Informational provenance only — never part of the conflict key
    /// (C4 §6).
    pub engine_version: String,
}

/// One `sessions/<id>/session.json` manifest entry. `updated_at_ms` is the
/// file's own mtime (the schema carries no such field, C1 §6) — the same
/// convention this crate already uses for workbooks (`store::catalog`'s
/// `index_workbook_file`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionJsonEntry {
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified time, milliseconds since the Unix epoch.
    pub updated_at_ms: i64,
}

/// One session's manifest entry, nested by session (C4 §6). Any of the
/// three file classes may be absent — a session mid-import, or one whose
/// `data.parquet` failed to parse (malformed file metadata never aborts
/// the walk; the session still appears with `data_parquet: None`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub data_parquet: Option<DataParquetEntry>,
    #[serde(default)]
    pub derived: Vec<DerivedEntry>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_json: Option<SessionJsonEntry>,
}

/// One `workbooks/<file_name>.idl1wb` manifest entry. `workbook_id` (front
/// matter, C2) is the identity key, not `file_name` — a rename is not a
/// new workbook (C4 §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbookEntry {
    pub workbook_id: String,
    /// The file's own stem (without `.idl1wb`) — names the file, not the
    /// workbook's identity.
    pub file_name: String,
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified time, milliseconds since the Unix epoch.
    pub updated_at_ms: i64,
}

/// One `tracks/<id>.idl0t` manifest entry. Last-write-wins by
/// `updated_at_ms` (C4 §6, SPEC §16.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackEntry {
    pub track_id: String,
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified time, milliseconds since the Unix epoch.
    pub updated_at_ms: i64,
}

/// One `profiles/<id>.idl0p` manifest entry. Last-write-wins by
/// `updated_at_ms`, same class as [`TrackEntry`] (C4 §6, ruling R6/R88).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub profile_id: String,
    pub sha256: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last-modified time, milliseconds since the Unix epoch.
    pub updated_at_ms: i64,
}

/// The full sync manifest (C4 §6) — the exact body `GET /idl1/v1/manifest`
/// answers, so its field names are the wire contract, not an internal
/// convenience shape. Every entry list is sorted by its own identity key
/// so two runs over an unchanged tree serialise byte-identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest document schema version (currently `1`).
    pub schema_version: u32,
    /// When this manifest was built, milliseconds since the Unix epoch.
    pub generated_at_ms: i64,
    pub blobs: Vec<BlobEntry>,
    pub sessions: Vec<SessionEntry>,
    pub workbooks: Vec<WorkbookEntry>,
    pub tracks: Vec<TrackEntry>,
    pub profiles: Vec<ProfileEntry>,
}

/// Discriminant for [`SyncError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncErrorKind {
    Io,
    Malformed,
}

/// Error from [`build_manifest`]. Never `Err(String)` (CLAUDE.md §5). Only
/// raised for a failure that makes the whole walk meaningless (e.g.
/// `data_root` cannot be read at all as a directory) — a malformed
/// individual file is reported by omission from the manifest instead
/// (see each entry type's doc comment), never by aborting the build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncError {
    pub kind: SyncErrorKind,
    pub message: String,
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for SyncError {}

/// Walks `<data_root>` (C4 §2) and builds the [`Manifest`] a sync peer
/// serves at `GET /idl1/v1/manifest` (C4 §6). Excludes, unconditionally:
/// `catalog.sqlite` and its `-wal`/`-shm` sidecars, `tmp/` (including
/// `tmp/quarantine/`), and `workbooks/.sync-base/` — plus any other
/// dotfile or dot-directory under `workbooks/`. `now_ms` is injected; this
/// module has no clock of its own (CLAUDE.md §2, this lane's brief).
///
/// A malformed individual file (unreadable, or missing the metadata its
/// class requires) never aborts the walk — it is simply omitted from the
/// manifest (a session's `data_parquet`/`session_json` becomes `None`; a
/// malformed workbook/track/profile file does not appear in its list) so
/// the rest of the tree still syncs. [`SyncError`] is reserved for a
/// failure that makes the whole walk meaningless, such as `data_root`
/// existing but not being readable as a directory at all.
pub fn build_manifest(data_root: &Path, now_ms: i64) -> Result<Manifest, SyncError> {
    if data_root.exists() && !data_root.is_dir() {
        return Err(SyncError { kind: SyncErrorKind::Io, message: format!("{}: not a directory", data_root.display()) });
    }

    let mut blobs = collect_blobs(data_root);
    blobs.sort_by(|a, b| a.sha256.cmp(&b.sha256));

    let mut sessions = collect_sessions(data_root);
    sessions.sort_by(|a, b| a.session_id.cmp(&b.session_id));

    let mut workbooks = collect_workbooks(data_root);
    workbooks.sort_by(|a, b| a.workbook_id.cmp(&b.workbook_id));

    let mut tracks = collect_tracks(data_root);
    tracks.sort_by(|a, b| a.track_id.cmp(&b.track_id));

    let mut profiles = collect_profiles(data_root);
    profiles.sort_by(|a, b| a.profile_id.cmp(&b.profile_id));

    Ok(Manifest { schema_version: 1, generated_at_ms: now_ms, blobs, sessions, workbooks, tracks, profiles })
}

/// Last-modified time of `meta`, milliseconds since the Unix epoch. `0` if
/// unavailable (matches `store::catalog`'s own `file_mtime_ms` — the two
/// are kept as separate private helpers rather than sharing one `pub(crate)`
/// symbol, since each module's need is a one-line computation over
/// [`std::fs::Metadata`] it already has in hand).
fn file_mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn collect_blobs(data_root: &Path) -> Vec<BlobEntry> {
    let mut out = Vec::new();
    let shards_dir = data_root.join("blobs").join("sha256");
    let Ok(shards) = std::fs::read_dir(&shards_dir) else { return out };
    for shard in shards.flatten() {
        let Ok(entries) = std::fs::read_dir(shard.path()) else { continue };
        for entry in entries.flatten() {
            let Ok(bytes) = std::fs::read(entry.path()) else { continue };
            out.push(BlobEntry { sha256: sha256_hex(&bytes), size_bytes: bytes.len() as u64 });
        }
    }
    out
}

/// Reads `data.parquet`'s file-level key-value metadata (C1 §4.3) for the
/// three fields the manifest needs. `None` on any read/parse failure or a
/// missing key — the caller treats that as "session malformed, omit
/// `data_parquet`" rather than aborting the walk. A separate reader from
/// `store::catalog::read_data_parquet_session_fields` (whose fields this
/// module has no access to — they are private to that module, and its
/// field set differs) rather than widening that struct's visibility for a
/// three-field subset.
fn read_data_parquet_manifest_fields(path: &Path) -> Option<(String, String, String)> {
    let file = std::fs::File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let kv = builder.metadata().file_metadata().key_value_metadata()?;
    let get = |k: &str| kv.iter().find(|e| e.key == k).and_then(|e| e.value.clone());
    Some((get("importer_version")?, get("seam_correction_version")?, get("engine_version")?))
}

fn collect_sessions(data_root: &Path) -> Vec<SessionEntry> {
    let mut out = Vec::new();
    let sessions_dir = data_root.join("sessions");
    let Ok(entries) = std::fs::read_dir(&sessions_dir) else { return out };
    for entry in entries.flatten() {
        let session_dir = entry.path();
        if !session_dir.is_dir() {
            continue;
        }
        let session_id = entry.file_name().to_string_lossy().into_owned();

        let dp_path = session_dir.join("data.parquet");
        let data_parquet = if dp_path.is_file() {
            match (std::fs::read(&dp_path), read_data_parquet_manifest_fields(&dp_path)) {
                (Ok(bytes), Some((importer_version, seam_correction_version, engine_version))) => Some(DataParquetEntry {
                    sha256: sha256_hex(&bytes),
                    size_bytes: bytes.len() as u64,
                    importer_version,
                    seam_correction_version,
                    engine_version,
                }),
                // Missing/unparseable metadata — malformed, omitted; the
                // rest of this session (and every other session) still
                // builds (this task's brief, "Key logic").
                _ => None,
            }
        } else {
            None
        };

        let mut derived = Vec::new();
        let derived_dir = session_dir.join("derived");
        if let Ok(derived_entries) = std::fs::read_dir(&derived_dir) {
            for d in derived_entries.flatten() {
                let path = d.path();
                if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                    continue;
                }
                let Ok(bytes) = std::fs::read(&path) else { continue };
                derived.push(DerivedEntry { sha256: sha256_hex(&bytes), size_bytes: bytes.len() as u64 });
            }
        }
        derived.sort_by(|a, b| a.sha256.cmp(&b.sha256));

        let sj_path = session_dir.join("session.json");
        let session_json = if sj_path.is_file() {
            match (std::fs::read(&sj_path), std::fs::metadata(&sj_path)) {
                (Ok(bytes), Ok(meta)) => Some(SessionJsonEntry {
                    sha256: sha256_hex(&bytes),
                    size_bytes: bytes.len() as u64,
                    updated_at_ms: file_mtime_ms(&meta),
                }),
                _ => None,
            }
        } else {
            None
        };

        out.push(SessionEntry { session_id, data_parquet, derived, session_json });
    }
    out
}

/// `true` for a workbooks-directory entry name that must be skipped
/// entirely (never descended into, never manifested) — a dotfile or
/// dot-directory, `workbooks/.sync-base/` (C2 §7's base cache) included as
/// one instance of that general rule, per C4 §6's amendment.
fn is_dot_name(name: &std::ffi::OsStr) -> bool {
    name.to_str().is_some_and(|s| s.starts_with('.'))
}

fn collect_workbooks(data_root: &Path) -> Vec<WorkbookEntry> {
    let mut out = Vec::new();
    let dir = data_root.join("workbooks");
    let Ok(entries) = std::fs::read_dir(&dir) else { return out };
    for entry in entries.flatten() {
        if is_dot_name(&entry.file_name()) {
            continue; // workbooks/.sync-base/ and any other dotfile/dot-dir
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl1wb") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(markdown) = std::str::from_utf8(&bytes) else { continue };
        // A workbook whose front matter will not parse is skipped, not
        // silently dropped from the walk's overall result — the rest of
        // the manifest still builds (this task's brief, "Key logic").
        let Ok((front_matter, _)) = parse_front_matter(markdown) else { continue };
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        out.push(WorkbookEntry {
            workbook_id: front_matter.id,
            file_name,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
            updated_at_ms: file_mtime_ms(&meta),
        });
    }
    out
}

fn collect_tracks(data_root: &Path) -> Vec<TrackEntry> {
    let mut out = Vec::new();
    let dir = data_root.join("tracks");
    let Ok(entries) = std::fs::read_dir(&dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl0t") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let Ok(track) = read_track(&path) else { continue };
        if track.id != stem {
            continue; // filename/content-id mismatch — malformed, omitted
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        out.push(TrackEntry {
            track_id: stem,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
            updated_at_ms: track.updated_at_ms,
        });
    }
    out
}

fn collect_profiles(data_root: &Path) -> Vec<ProfileEntry> {
    let mut out = Vec::new();
    let dir = data_root.join("profiles");
    let Ok(entries) = std::fs::read_dir(&dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl0p") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(profile) = serde_json::from_slice::<crate::store::profile::BikeProfile>(&bytes) else { continue };
        if profile.profile_id != stem {
            continue; // filename/content-id mismatch — malformed, omitted
        }
        out.push(ProfileEntry {
            profile_id: stem,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
            updated_at_ms: profile.updated_at_ms,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Channel, RawColumn, Session, SourceFormat};
    use crate::store::blob::write_blob;
    use crate::store::parquet::write_session_parquet;
    use crate::store::profile::BikeProfile;
    use crate::store::session_json::{empty_session_json, write_session_json};
    use crate::track_artifact::model::Track;
    use crate::track_artifact::write::write_track;
    use crate::workbook::v3::front_matter::{render_front_matter, FrontMatter};
    use std::collections::HashMap;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_full_session(root: &Path, session_id: &str, blob_sha256: String) {
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
    fn build_manifest_an_empty_data_root_every_class_empty_schema_version_1() {
        // Arrange
        let root = temp_root();

        // Act
        let manifest = build_manifest(&root, 1000).unwrap();

        // Assert
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.generated_at_ms, 1000);
        assert!(manifest.blobs.is_empty());
        assert!(manifest.sessions.is_empty());
        assert!(manifest.workbooks.is_empty());
        assert!(manifest.tracks.is_empty());
        assert!(manifest.profiles.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_manifest_one_blob_one_session_with_data_parquet_and_session_json_entries_carry_the_right_hashes_and_sizes() {
        // Arrange
        let root = temp_root();
        let blob_bytes = b"raw bytes".to_vec();
        let blob_sha256 = write_blob(&root, &blob_bytes).unwrap();
        write_full_session(&root, "s1", blob_sha256);

        // Act
        let manifest = build_manifest(&root, 2000).unwrap();

        // Assert
        assert_eq!(manifest.blobs.len(), 1);
        assert_eq!(manifest.blobs[0].sha256, sha256_hex(&blob_bytes));
        assert_eq!(manifest.blobs[0].size_bytes, blob_bytes.len() as u64);

        assert_eq!(manifest.sessions.len(), 1);
        let session = &manifest.sessions[0];
        assert_eq!(session.session_id, "s1");
        let dp = session.data_parquet.as_ref().unwrap();
        let dp_bytes = std::fs::read(root.join("sessions").join("s1").join("data.parquet")).unwrap();
        assert_eq!(dp.sha256, sha256_hex(&dp_bytes));
        assert_eq!(dp.size_bytes, dp_bytes.len() as u64);
        assert_eq!(dp.importer_version, "0.1.0");
        assert!(!dp.seam_correction_version.is_empty());
        assert!(!dp.engine_version.is_empty());
        let sj = session.session_json.as_ref().unwrap();
        let sj_bytes = std::fs::read(root.join("sessions").join("s1").join("session.json")).unwrap();
        assert_eq!(sj.sha256, sha256_hex(&sj_bytes));
        assert_eq!(sj.size_bytes, sj_bytes.len() as u64);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_manifest_catalog_sqlite_its_wal_shm_and_tmp_present_none_appear() {
        // Arrange
        let root = temp_root();
        std::fs::write(root.join("catalog.sqlite"), b"x").unwrap();
        std::fs::write(root.join("catalog.sqlite-wal"), b"x").unwrap();
        std::fs::write(root.join("catalog.sqlite-shm"), b"x").unwrap();
        std::fs::create_dir_all(root.join("tmp").join("quarantine")).unwrap();
        std::fs::write(root.join("tmp").join("quarantine").join("leftover"), b"x").unwrap();

        // Act
        let manifest = build_manifest(&root, 1).unwrap();

        // Assert — every class stays empty; catalog/tmp are simply never
        // walked by this module (unlike `store::verify`, this walk has no
        // "unexpected path" reporting to accidentally trip on them).
        assert!(manifest.blobs.is_empty());
        assert!(manifest.sessions.is_empty());
        assert!(manifest.workbooks.is_empty());
        assert!(manifest.tracks.is_empty());
        assert!(manifest.profiles.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    fn workbook_front_matter(id: &str, name: &str) -> FrontMatter {
        FrontMatter { id: id.to_string(), name: name.to_string(), constants: HashMap::new(), units: Default::default(), version: 3 }
    }

    #[test]
    fn build_manifest_workbooks_sync_base_id_idl1wb_present_not_in_workbooks() {
        // Arrange
        let root = temp_root();
        let workbooks_dir = root.join("workbooks");
        let sync_base_dir = workbooks_dir.join(".sync-base");
        std::fs::create_dir_all(&sync_base_dir).unwrap();
        let id = Uuid::new_v4().to_string();
        std::fs::write(sync_base_dir.join(format!("{id}.idl1wb")), render_front_matter(&workbook_front_matter(&id, "Base"))).unwrap();

        // Act
        let manifest = build_manifest(&root, 1).unwrap();

        // Assert
        assert!(manifest.workbooks.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_manifest_a_workbook_whose_file_name_differs_from_its_workbook_id_the_entry_keys_on_workbook_id() {
        // Arrange
        let root = temp_root();
        let workbooks_dir = root.join("workbooks");
        std::fs::create_dir_all(&workbooks_dir).unwrap();
        let id = Uuid::new_v4().to_string();
        std::fs::write(workbooks_dir.join("fork-tuning.idl1wb"), render_front_matter(&workbook_front_matter(&id, "Fork tuning"))).unwrap();

        // Act
        let manifest = build_manifest(&root, 1).unwrap();

        // Assert
        assert_eq!(manifest.workbooks.len(), 1);
        assert_eq!(manifest.workbooks[0].workbook_id, id);
        assert_eq!(manifest.workbooks[0].file_name, "fork-tuning");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_manifest_data_parquet_without_importer_metadata_the_session_is_reported_malformed_and_the_other_session_still_appears(
    ) {
        // Arrange — "s1" has a genuine data.parquet; "s2" has a hand-written
        // file under that name with no Parquet file metadata at all, so it
        // can never yield importer_version/seam_correction_version/
        // engine_version.
        let root = temp_root();
        let blob_sha256 = write_blob(&root, b"raw bytes").unwrap();
        write_full_session(&root, "s1", blob_sha256);
        let s2_dir = root.join("sessions").join("s2");
        std::fs::create_dir_all(&s2_dir).unwrap();
        std::fs::write(s2_dir.join("data.parquet"), b"not a real parquet file").unwrap();

        // Act
        let manifest = build_manifest(&root, 1).unwrap();

        // Assert
        assert_eq!(manifest.sessions.len(), 2);
        let s1 = manifest.sessions.iter().find(|s| s.session_id == "s1").unwrap();
        let s2 = manifest.sessions.iter().find(|s| s.session_id == "s2").unwrap();
        assert!(s1.data_parquet.is_some());
        assert!(s2.data_parquet.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_manifest_run_twice_over_one_tree_identical_serialised_json() {
        // Arrange
        let root = temp_root();
        let blob_sha256 = write_blob(&root, b"raw bytes").unwrap();
        write_full_session(&root, "s1", blob_sha256);
        let workbooks_dir = root.join("workbooks");
        std::fs::create_dir_all(&workbooks_dir).unwrap();
        let id = Uuid::new_v4().to_string();
        std::fs::write(workbooks_dir.join("fork-tuning.idl1wb"), render_front_matter(&workbook_front_matter(&id, "Fork tuning"))).unwrap();
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
        write_track(&root, &track).unwrap();
        std::fs::create_dir_all(root.join("profiles")).unwrap();
        let profile =
            BikeProfile { profile_id: "p-1".to_string(), profile_name: "Bike".to_string(), created_at_ms: 1, updated_at_ms: 2, config: serde_json::json!({}) };
        std::fs::write(root.join("profiles").join("p-1.idl0p"), serde_json::to_vec(&profile).unwrap()).unwrap();

        // Act
        let m1 = build_manifest(&root, 42).unwrap();
        let m2 = build_manifest(&root, 42).unwrap();

        // Assert
        assert_eq!(serde_json::to_string(&m1).unwrap(), serde_json::to_string(&m2).unwrap());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn manifest_round_trips_through_serde_json_with_c4_6_field_names() {
        // Arrange
        let manifest = Manifest {
            schema_version: 1,
            generated_at_ms: 1_756_857_600_000,
            blobs: vec![BlobEntry { sha256: "a".repeat(64), size_bytes: 12_345_678 }],
            sessions: vec![SessionEntry {
                session_id: "s1".to_string(),
                data_parquet: Some(DataParquetEntry {
                    sha256: "b".repeat(64),
                    size_bytes: 98_765,
                    importer_version: "0.3.0".to_string(),
                    seam_correction_version: "v1".to_string(),
                    engine_version: "0.7.2".to_string(),
                }),
                derived: vec![DerivedEntry { sha256: "c".repeat(64), size_bytes: 4321 }],
                session_json: Some(SessionJsonEntry { sha256: "d".repeat(64), size_bytes: 512, updated_at_ms: 1_756_857_600_000 }),
            }],
            workbooks: vec![WorkbookEntry {
                workbook_id: "wb-1".to_string(),
                file_name: "fork-tuning".to_string(),
                sha256: "e".repeat(64),
                size_bytes: 2048,
                updated_at_ms: 1_756_857_600_000,
            }],
            tracks: vec![TrackEntry { track_id: "t-1".to_string(), sha256: "f".repeat(64), size_bytes: 900, updated_at_ms: 1_756_857_600_000 }],
            profiles: vec![ProfileEntry { profile_id: "p-1".to_string(), sha256: "0".repeat(64), size_bytes: 640, updated_at_ms: 1_756_857_600_000 }],
        };

        // Act
        let json = serde_json::to_value(&manifest).unwrap();
        let back: Manifest = serde_json::from_value(json.clone()).unwrap();

        // Assert
        assert_eq!(back, manifest);
        assert!(json["sessions"][0].get("data_parquet").is_some());
        assert_eq!(json["sessions"][0]["data_parquet"]["seam_correction_version"], "v1");
        assert_eq!(json["profiles"][0]["profile_id"], "p-1");
        assert_eq!(json["workbooks"][0]["workbook_id"], "wb-1");
    }
}
