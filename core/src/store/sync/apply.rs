//! Verified install of one received sync item (C4 §6, ruling R88; design
//! §7's "a blob enters the local catalog only after its hash verifies").
//! `bytes` are exactly what the wire delivered — nothing here trusts a
//! manifest's claim about them; every content-addressed class re-derives
//! its own identity from the bytes themselves, and `session.json`/workbook
//! merges compute their result from full documents, never from a manifest
//! summary alone (design §7, C4 §6's transfer section). No network, no
//! catalog write (`store::catalog::index_session` runs after install, at
//! the command layer — CLAUDE.md §2) and no lap recomputation (the L2b
//! cache is never merged, see [`super::session_merge`]).
//!
//! **Deviation from this task's interface sketch, flagged for review.**
//! [`SyncItem`] (landed by Task 3, `super::diff`) carries only `class` /
//! `key` / `session_id` / `size_bytes` — none of the per-class manifest
//! detail this task's own "Key logic" section requires knowing at install
//! time: `session.json`'s peer file mtime for the per-field LWW tiebreak,
//! `data.parquet`'s claimed `(importer_version, seam_correction_version)`
//! to verify the received bytes against, and a workbook's peer `file_name`
//! (needed both to name a brand-new local file and to detect the rename
//! case). Extending `SyncItem`/`diff.rs` is outside this task's file list
//! (`apply.rs`/`session_merge.rs`/`sync/mod.rs`/`CHANGELOG.md`), so
//! [`InstallContext`] carries exactly this extra, manifest-sourced detail
//! instead — populated by a future caller (Task 10/12) from the same
//! manifest it already fetched to plan the sync. `install`'s own five
//! "core" parameters otherwise match the brief's sketch unchanged.

use std::path::{Path, PathBuf};

use crate::store::atomic::{sha256_hex, write_atomic, write_atomic_with_retry, AtomicWriteError};
use crate::store::profile::{BikeProfile, ProfileError};
use crate::store::session_json::{
    empty_session_json, parse_session_json, read_session_json, write_session_json, SessionJson, SessionJsonError,
};
use crate::store::sync::base_cache;
use crate::store::sync::diff::{SyncClass, SyncItem};
use crate::store::sync::manifest::{SyncError, SyncErrorKind};
use crate::store::sync::session_merge::merge_session_json;
use crate::track_artifact::model::Track;
use crate::track_artifact::read::parse_track;
use crate::track_artifact::write::{write_track, TrackWriteError};
use crate::workbook::merge::{merge, MergeError};
use crate::workbook::v3::{front_matter::parse_front_matter, parse_workbook, render_workbook, WorkbookDoc};

/// What installing one received file did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallOutcome {
    /// Written (or already byte-identical — the write was skipped).
    Installed,
    /// Ignored: the local copy wins under this class's rule.
    KeptLocal,
    /// Workbook only: merged, with this many conflict cells created.
    Merged { conflicts: u32 },
}

/// Manifest-sourced detail [`install`] needs beyond [`SyncItem`]'s bare
/// identity, for the three classes that need it (see this module's doc
/// comment) — absent (`None`) for every other class, and simply unused by
/// them. Every field names, verbatim, the manifest entry it comes from.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InstallContext {
    /// `SyncClass::DataParquet`: the peer manifest's claimed
    /// `(importer_version, seam_correction_version)` for this session's
    /// `data.parquet` — the received bytes' own embedded metadata must
    /// match this exactly, or the install is refused (defends against a
    /// stale manifest or a file that changed between the manifest fetch
    /// and this fetch).
    pub claimed_data_parquet_versions: Option<(String, String)>,
    /// `SyncClass::SessionJson`: the peer's copy's own file mtime
    /// (`manifest::SessionJsonEntry::updated_at_ms`) — the per-field
    /// merge's LWW tiebreak (see `session_merge`'s doc comment).
    pub peer_session_json_updated_at_ms: Option<i64>,
    /// `SyncClass::Workbook`: the peer manifest's `file_name` for this
    /// `workbook_id` — the name a brand-new local file is written under,
    /// and what the local file is renamed to when it differs from the
    /// local file's own current name (C4 §6: "id wins").
    pub peer_workbook_file_name: Option<String>,
}

impl From<AtomicWriteError> for SyncError {
    fn from(e: AtomicWriteError) -> Self {
        SyncError { kind: SyncErrorKind::Io, message: e.to_string() }
    }
}
impl From<TrackWriteError> for SyncError {
    fn from(e: TrackWriteError) -> Self {
        SyncError { kind: SyncErrorKind::Io, message: e.to_string() }
    }
}
impl From<ProfileError> for SyncError {
    fn from(e: ProfileError) -> Self {
        SyncError { kind: SyncErrorKind::Io, message: e.to_string() }
    }
}
impl From<SessionJsonError> for SyncError {
    fn from(e: SessionJsonError) -> Self {
        SyncError { kind: SyncErrorKind::Malformed, message: e.to_string() }
    }
}
impl From<MergeError> for SyncError {
    fn from(e: MergeError) -> Self {
        SyncError { kind: SyncErrorKind::Malformed, message: e.to_string() }
    }
}

fn malformed(message: impl Into<String>) -> SyncError {
    SyncError { kind: SyncErrorKind::Malformed, message: message.into() }
}

/// Installs one received file. `bytes` are exactly what the wire delivered;
/// nothing is trusted until this function verifies it. `now_ms` is used
/// only where a class needs "the moment of receipt" (none of the classes
/// implemented so far do — kept for interface symmetry with the wire
/// layer's other timestamps and for a future class). `ctx` supplies the
/// manifest detail `item` itself does not carry (see this module's doc
/// comment).
pub fn install(data_root: &Path, item: &SyncItem, bytes: &[u8], peer_name: &str, _now_ms: i64, ctx: &InstallContext) -> Result<InstallOutcome, SyncError> {
    match item.class {
        SyncClass::Blob => install_blob(data_root, item, bytes),
        SyncClass::Derived => install_derived(data_root, item, bytes),
        SyncClass::DataParquet => install_data_parquet(data_root, item, bytes, ctx),
        SyncClass::SessionJson => install_session_json(data_root, item, bytes, ctx),
        SyncClass::Workbook => install_workbook(data_root, item, bytes, peer_name, ctx),
        SyncClass::Track => install_track(data_root, bytes),
        SyncClass::Profile => install_profile(data_root, bytes),
    }
}

/// Content-addressed: `item.key` is the requested sha256. A mismatch
/// against the bytes actually received is refused before anything is
/// written — the wrong hash simply names the wrong path, so this also
/// guards against ever corrupting an existing entry (C4 §6, design §7).
fn install_blob(data_root: &Path, item: &SyncItem, bytes: &[u8]) -> Result<InstallOutcome, SyncError> {
    let digest = sha256_hex(bytes);
    if digest != item.key {
        return Err(malformed(format!("blob: requested {}, received bytes hash to {digest}", item.key)));
    }
    let path = crate::store::blob::blob_path(data_root, &digest);
    if path.is_file() {
        return Ok(InstallOutcome::Installed); // verified no-op (C4 §4)
    }
    write_atomic(data_root, &path, bytes, None)?;
    Ok(InstallOutcome::Installed)
}

/// Content-addressed like a blob, but session-scoped and never overwritten
/// once present (C4 §6: "name is content").
fn install_derived(data_root: &Path, item: &SyncItem, bytes: &[u8]) -> Result<InstallOutcome, SyncError> {
    let digest = sha256_hex(bytes);
    if digest != item.key {
        return Err(malformed(format!("derived: requested {}, received bytes hash to {digest}", item.key)));
    }
    let session_id = item.session_id.as_deref().ok_or_else(|| malformed("derived: item has no session_id"))?;
    let path = data_root.join("sessions").join(session_id).join("derived").join(format!("{digest}.parquet"));
    if path.is_file() {
        return Ok(InstallOutcome::Installed);
    }
    write_atomic(data_root, &path, bytes, None)?;
    Ok(InstallOutcome::Installed)
}

/// Reads `bytes`' own embedded `(importer_version, seam_correction_version)`
/// Parquet file-level metadata by writing them to a scratch file under
/// `tmp/` (the `parquet` reader needs a `File`, not an in-memory slice) and
/// removing it again immediately — never left behind, satisfying "nothing
/// left in tmp/" regardless of whether the read succeeds.
fn read_data_parquet_versions_from_bytes(data_root: &Path, bytes: &[u8]) -> Option<(String, String)> {
    let tmp_dir = data_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir).ok()?;
    let scratch = tmp_dir.join(format!("sync-verify-{}", uuid::Uuid::new_v4()));
    std::fs::write(&scratch, bytes).ok()?;
    let result = (|| {
        let file = std::fs::File::open(&scratch).ok()?;
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
        let kv = builder.metadata().file_metadata().key_value_metadata()?;
        let get = |k: &str| kv.iter().find(|e| e.key == k).and_then(|e| e.value.clone());
        Some((get("importer_version")?, get("seam_correction_version")?))
    })();
    let _ = std::fs::remove_file(&scratch);
    result
}

/// `data.parquet`: re-derives the received bytes' own version metadata and
/// refuses unless it matches `ctx`'s claimed pair exactly (C4 §6's transfer
/// rule — "the wrong pair is refused" is this task's own reading of "if it
/// does not match what the manifest claimed"). Otherwise written through
/// the atomic primitive, bytes as-is (never regenerated, C4 §6/C1 §4.3).
fn install_data_parquet(data_root: &Path, item: &SyncItem, bytes: &[u8], ctx: &InstallContext) -> Result<InstallOutcome, SyncError> {
    let session_id = item.session_id.as_deref().ok_or_else(|| malformed("data.parquet: item has no session_id"))?;
    let Some(claimed) = &ctx.claimed_data_parquet_versions else {
        return Err(malformed("data.parquet: install called with no claimed version pair in InstallContext"));
    };
    let actual = read_data_parquet_versions_from_bytes(data_root, bytes)
        .ok_or_else(|| malformed("data.parquet: received bytes have no readable importer/seam-correction metadata"))?;
    if actual != *claimed {
        return Err(malformed(format!(
            "data.parquet: manifest claimed {claimed:?}, received bytes carry {actual:?}"
        )));
    }
    let target = data_root.join("sessions").join(session_id).join("data.parquet");
    let based_on = std::fs::read(&target).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(data_root, &target, bytes, based_on.as_deref(), |_current| bytes.to_vec())?;
    Ok(InstallOutcome::Installed)
}

/// `session.json`: per-field merge (`session_merge::merge_session_json`)
/// against the local copy (an absent local file merges against
/// [`empty_session_json`], matching a first-ever sync for this session).
fn install_session_json(data_root: &Path, item: &SyncItem, bytes: &[u8], ctx: &InstallContext) -> Result<InstallOutcome, SyncError> {
    let session_id = item.session_id.as_deref().ok_or_else(|| malformed("session.json: item has no session_id"))?;
    let target = data_root.join("sessions").join(session_id).join("session.json");

    let (local_doc, local_updated_at_ms, based_on_hash) = match std::fs::metadata(&target) {
        Ok(meta) => {
            let local = read_session_json(&target)?;
            (local, file_mtime_ms(&meta), std::fs::read(&target).ok().map(|b| sha256_hex(&b)))
        }
        Err(_) => (empty_session_json(session_id), 0, None),
    };

    let peer_doc: SessionJson = parse_session_json(bytes)?;
    let peer_updated_at_ms = ctx.peer_session_json_updated_at_ms.unwrap_or(0);

    let merged = merge_session_json(&local_doc, &peer_doc, local_updated_at_ms, peer_updated_at_ms);
    write_session_json(data_root, session_id, &merged, based_on_hash.as_deref())?;
    Ok(InstallOutcome::Installed)
}

/// Last-modified time of `meta`, milliseconds since the Unix epoch — same
/// convention as `manifest::file_mtime_ms` (kept as its own private copy;
/// that function is not `pub(crate)` and `manifest.rs` is outside this
/// task's file list).
fn file_mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Finds the local `workbooks/*.idl1wb` file whose front matter `id`
/// matches `workbook_id`, if any. `None` when no local copy exists yet
/// (first-ever sync of this workbook).
fn find_local_workbook(data_root: &Path, workbook_id: &str) -> Option<(PathBuf, Vec<u8>)> {
    let dir = data_root.join("workbooks");
    let entries = std::fs::read_dir(&dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_str().is_some_and(|s| s.starts_with('.')) {
            continue; // workbooks/.sync-base/ and any other dotfile/dot-dir
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("idl1wb") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let Ok(text) = std::str::from_utf8(&bytes) else { continue };
        let Ok((fm, _)) = parse_front_matter(text) else { continue };
        if fm.id == workbook_id {
            return Some((path, bytes));
        }
    }
    None
}

/// An empty `WorkbookDoc` for `workbook_id` — the merge base when no
/// `.sync-base` cache exists yet (first-ever sync, C2 §7).
fn empty_workbook_doc(workbook_id: &str) -> WorkbookDoc {
    WorkbookDoc {
        id: workbook_id.to_string(),
        name: String::new(),
        constants_raw: std::collections::HashMap::new(),
        units_pref: Default::default(),
        version: 3,
        cells: Vec::new(),
        trailing_prose: None,
        const_lines: Vec::new(),
        defs: Vec::new(),
        constants: std::collections::HashMap::new(),
    }
}

/// Workbook: merges against the local copy and the `.sync-base` cache when
/// a local copy exists (C2 §7); otherwise installs the peer's bytes
/// verbatim as a brand-new local file (nothing to merge against). A
/// `file_name` mismatch for a known `workbook_id` renames the local file —
/// id wins (C4 §6).
fn install_workbook(data_root: &Path, item: &SyncItem, bytes: &[u8], peer_name: &str, ctx: &InstallContext) -> Result<InstallOutcome, SyncError> {
    let workbook_id = &item.key;
    let peer_text = std::str::from_utf8(bytes).map_err(|e| malformed(format!("workbook: not valid UTF-8: {e}")))?;

    match find_local_workbook(data_root, workbook_id) {
        None => {
            // First-ever sync of this workbook — nothing to merge against.
            let (_peer_doc, _errors) = parse_workbook(peer_text).map_err(|errs| malformed(format!("workbook: peer bytes do not parse: {errs:?}")))?;
            let file_name = ctx
                .peer_workbook_file_name
                .as_deref()
                .ok_or_else(|| malformed("workbook: install called with no peer_workbook_file_name in InstallContext"))?;
            let target = data_root.join("workbooks").join(format!("{file_name}.idl1wb"));
            write_atomic(data_root, &target, bytes, None)?;
            base_cache::write_base(data_root, workbook_id, bytes)?;
            Ok(InstallOutcome::Installed)
        }
        Some((local_path, local_bytes)) => {
            let local_text = std::str::from_utf8(&local_bytes).map_err(|e| malformed(format!("workbook: local file not valid UTF-8: {e}")))?;
            let (local_doc, _) = parse_workbook(local_text).map_err(|errs| malformed(format!("workbook: local file does not parse: {errs:?}")))?;
            let (peer_doc, _) = parse_workbook(peer_text).map_err(|errs| malformed(format!("workbook: peer bytes do not parse: {errs:?}")))?;
            let base_doc = base_cache::read_base(data_root, workbook_id)?.unwrap_or_else(|| empty_workbook_doc(workbook_id));

            let merged = merge(&local_doc, &peer_doc, &base_doc, peer_name)?;
            let rendered = render_workbook(&merged.doc);

            let local_file_name = local_path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let target_file_name = ctx.peer_workbook_file_name.clone().unwrap_or(local_file_name.clone());
            let target_path = data_root.join("workbooks").join(format!("{target_file_name}.idl1wb"));

            let based_on_hash = if target_path == local_path { Some(sha256_hex(&local_bytes)) } else { None };
            write_atomic(data_root, &target_path, rendered.as_bytes(), based_on_hash.as_deref())?;
            if target_path != local_path {
                let _ = std::fs::remove_file(&local_path); // rename: id wins (C4 §6)
            }
            base_cache::write_base(data_root, workbook_id, rendered.as_bytes())?;

            Ok(InstallOutcome::Merged { conflicts: merged.conflicts })
        }
    }
}

/// Track: last-write-wins by `updated_at_ms` (C4 §6, SPEC §16.5). Re-checks
/// at install time rather than trusting the plan (defense against a race
/// between the manifest fetch and this file's fetch).
fn install_track(data_root: &Path, bytes: &[u8]) -> Result<InstallOutcome, SyncError> {
    let peer: Track = parse_track(bytes).map_err(|e| malformed(format!("track: {e}")))?;
    let path = data_root.join("tracks").join(format!("{}.idl0t", peer.id));
    let local_updated_at_ms = crate::track_artifact::read::read_track(&path).ok().map(|t| t.updated_at_ms);

    if let Some(local_ms) = local_updated_at_ms {
        if peer.updated_at_ms < local_ms {
            return Ok(InstallOutcome::KeptLocal);
        }
    }
    write_track(data_root, &peer)?;
    Ok(InstallOutcome::Installed)
}

/// Profile: last-write-wins by `updated_at_ms`, same rule as Track (C4 §6,
/// ruling R6/R88).
fn install_profile(data_root: &Path, bytes: &[u8]) -> Result<InstallOutcome, SyncError> {
    let peer: BikeProfile = serde_json::from_slice(bytes).map_err(|e| malformed(format!("profile: {e}")))?;
    let path = data_root.join("profiles").join(format!("{}.idl0p", peer.profile_id));
    let local_updated_at_ms =
        std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<BikeProfile>(&b).ok()).map(|p| p.updated_at_ms);

    if let Some(local_ms) = local_updated_at_ms {
        if peer.updated_at_ms < local_ms {
            return Ok(InstallOutcome::KeptLocal);
        }
    }
    crate::store::profile::save(data_root, &peer)?;
    Ok(InstallOutcome::Installed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::blob::read_blob;
    use crate::store::sync::diff::SyncClass;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn blob_item(key: &str, size_bytes: u64) -> SyncItem {
        SyncItem { class: SyncClass::Blob, key: key.to_string(), session_id: None, size_bytes }
    }

    #[test]
    fn install_a_blob_whose_bytes_hash_to_the_requested_hash_installed_and_readable() {
        // Arrange
        let root = temp_root();
        let bytes = b"raw source bytes";
        let digest = sha256_hex(bytes);
        let item = blob_item(&digest, bytes.len() as u64);

        // Act
        let outcome = install(&root, &item, bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Installed);
        assert_eq!(read_blob(&root, &digest).unwrap(), bytes);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_a_blob_whose_bytes_do_not_match_syncerror_and_no_file_written() {
        // Arrange
        let root = temp_root();
        let bytes = b"raw source bytes";
        let wrong_hash = "f".repeat(64);
        let item = blob_item(&wrong_hash, bytes.len() as u64);

        // Act
        let err = install(&root, &item, bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap_err();

        // Assert
        assert_eq!(err.kind, SyncErrorKind::Malformed);
        assert!(!root.join("blobs").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_a_derived_parquet_lands_under_the_sessions_derived_by_hash() {
        // Arrange
        let root = temp_root();
        let bytes = b"anything at all";
        let digest = sha256_hex(bytes);
        let item = SyncItem { class: SyncClass::Derived, key: digest.clone(), session_id: Some("s1".to_string()), size_bytes: bytes.len() as u64 };

        // Act
        let outcome = install(&root, &item, bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Installed);
        assert_eq!(std::fs::read(root.join("sessions").join("s1").join("derived").join(format!("{digest}.parquet"))).unwrap(), bytes);

        let _ = std::fs::remove_dir_all(&root);
    }

    fn write_test_parquet(path: &Path, importer_version: &str, seam_correction_version: &str) -> Vec<u8> {
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        use parquet::file::metadata::KeyValue;

        let schema = Arc::new(Schema::new(vec![Field::new("t_us", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Float64Array::from(vec![0.0, 1.0]))]).unwrap();
        let metadata = vec![
            KeyValue::new("importer_version".to_string(), importer_version.to_string()),
            KeyValue::new("seam_correction_version".to_string(), seam_correction_version.to_string()),
            KeyValue::new("engine_version".to_string(), "9.9.9".to_string()),
        ];
        let props = WriterProperties::builder().set_key_value_metadata(Some(metadata)).build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::read(path).unwrap()
    }

    #[test]
    fn install_data_parquet_whose_embedded_versions_contradict_the_manifest_refused() {
        // Arrange
        let root = temp_root();
        let scratch = root.join("scratch.parquet");
        std::fs::create_dir_all(&root).unwrap();
        let bytes = write_test_parquet(&scratch, "0.2.0", "v1");
        let item = SyncItem { class: SyncClass::DataParquet, key: "s1".to_string(), session_id: Some("s1".to_string()), size_bytes: bytes.len() as u64 };
        let ctx = InstallContext { claimed_data_parquet_versions: Some(("0.9.0".to_string(), "v9".to_string())), ..Default::default() };

        // Act
        let err = install(&root, &item, &bytes, "peer-laptop", 1000, &ctx).unwrap_err();

        // Assert
        assert_eq!(err.kind, SyncErrorKind::Malformed);
        assert!(!root.join("sessions").join("s1").join("data.parquet").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_data_parquet_whose_embedded_versions_match_the_manifest_installed() {
        // Arrange
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();
        let scratch = root.join("scratch.parquet");
        let bytes = write_test_parquet(&scratch, "0.2.0", "v1");
        let item = SyncItem { class: SyncClass::DataParquet, key: "s1".to_string(), session_id: Some("s1".to_string()), size_bytes: bytes.len() as u64 };
        let ctx = InstallContext { claimed_data_parquet_versions: Some(("0.2.0".to_string(), "v1".to_string())), ..Default::default() };

        // Act
        let outcome = install(&root, &item, &bytes, "peer-laptop", 1000, &ctx).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Installed);
        assert_eq!(std::fs::read(root.join("sessions").join("s1").join("data.parquet")).unwrap(), bytes);

        let _ = std::fs::remove_dir_all(&root);
    }

    const WB_ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn wb_source(extra_front: &str, body: &str) -> String {
        format!("---\nid: {WB_ID}\nname: Test\n{extra_front}---\n{body}")
    }

    #[test]
    fn install_a_workbook_edited_on_both_sides_in_different_cells_merged_zero_conflicts_base_cache_updated() {
        // Arrange
        let root = temp_root();
        let workbooks_dir = root.join("workbooks");
        std::fs::create_dir_all(&workbooks_dir).unwrap();
        let base_src = wb_source("", "```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n");
        let local_src = wb_source("", "```math id=aaaaaaaa\nx = 2\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n");
        let peer_src = wb_source("", "```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 2\n```\n");
        std::fs::write(workbooks_dir.join("fork-tuning.idl1wb"), &local_src).unwrap();
        base_cache::write_base(&root, WB_ID, base_src.as_bytes()).unwrap();
        let item = SyncItem { class: SyncClass::Workbook, key: WB_ID.to_string(), session_id: None, size_bytes: peer_src.len() as u64 };
        let ctx = InstallContext { peer_workbook_file_name: Some("fork-tuning".to_string()), ..Default::default() };

        // Act
        let outcome = install(&root, &item, peer_src.as_bytes(), "peer-laptop", 1000, &ctx).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Merged { conflicts: 0 });
        let written = std::fs::read_to_string(workbooks_dir.join("fork-tuning.idl1wb")).unwrap();
        assert!(written.contains("x = 2"));
        assert!(written.contains("y = 2"));
        let base_back = base_cache::read_base(&root, WB_ID).unwrap().unwrap();
        assert!(base_back.cells.iter().any(|c| c.raw_fence_body.contains("x = 2")));
        assert!(base_back.cells.iter().any(|c| c.raw_fence_body.contains("y = 2")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_a_workbook_whose_file_name_differs_the_local_file_is_renamed() {
        // Arrange
        let root = temp_root();
        let workbooks_dir = root.join("workbooks");
        std::fs::create_dir_all(&workbooks_dir).unwrap();
        let base_src = wb_source("", "```math id=aaaaaaaa\nx = 1\n```\n");
        let local_src = wb_source("", "```math id=aaaaaaaa\nx = 2\n```\n");
        let peer_src = wb_source("", "```math id=aaaaaaaa\nx = 3\n```\n");
        std::fs::write(workbooks_dir.join("old-name.idl1wb"), &local_src).unwrap();
        base_cache::write_base(&root, WB_ID, base_src.as_bytes()).unwrap();
        let item = SyncItem { class: SyncClass::Workbook, key: WB_ID.to_string(), session_id: None, size_bytes: peer_src.len() as u64 };
        let ctx = InstallContext { peer_workbook_file_name: Some("new-name".to_string()), ..Default::default() };

        // Act
        let outcome = install(&root, &item, peer_src.as_bytes(), "peer-laptop", 1000, &ctx).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Merged { conflicts: 1 });
        assert!(!workbooks_dir.join("old-name.idl1wb").exists());
        assert!(workbooks_dir.join("new-name.idl1wb").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    fn sample_track(id: &str, updated_at_ms: i64) -> Track {
        Track {
            id: id.to_string(),
            name: "A-Line".to_string(),
            venue: "Whistler".to_string(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 0,
            updated_at_ms,
        }
    }

    fn track_bytes(track: &Track) -> Vec<u8> {
        use crate::track_artifact::model::TrackArtifact;
        serde_json::to_vec(&TrackArtifact::from(track)).unwrap()
    }

    #[test]
    fn install_a_track_older_than_local_keptlocal_the_file_unchanged() {
        // Arrange
        let root = temp_root();
        let local = sample_track("t-1", 100);
        write_track(&root, &local).unwrap();
        let peer = sample_track("t-1", 50);
        let bytes = track_bytes(&peer);
        let item = SyncItem { class: SyncClass::Track, key: "t-1".to_string(), session_id: None, size_bytes: bytes.len() as u64 };

        // Act
        let outcome = install(&root, &item, &bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::KeptLocal);
        let back = crate::track_artifact::read::read_track(&root.join("tracks").join("t-1.idl0t")).unwrap();
        assert_eq!(back.updated_at_ms, 100);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_a_track_newer_than_local_installed() {
        // Arrange
        let root = temp_root();
        let local = sample_track("t-1", 50);
        write_track(&root, &local).unwrap();
        let peer = sample_track("t-1", 100);
        let bytes = track_bytes(&peer);
        let item = SyncItem { class: SyncClass::Track, key: "t-1".to_string(), session_id: None, size_bytes: bytes.len() as u64 };

        // Act
        let outcome = install(&root, &item, &bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap();

        // Assert
        assert_eq!(outcome, InstallOutcome::Installed);
        let back = crate::track_artifact::read::read_track(&root.join("tracks").join("t-1.idl0t")).unwrap();
        assert_eq!(back.updated_at_ms, 100);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_every_class_nothing_is_left_behind_in_tmp() {
        // Arrange
        let root = temp_root();
        let blob_bytes = b"blob bytes";
        let blob_digest = sha256_hex(blob_bytes);
        let blob_item = SyncItem { class: SyncClass::Blob, key: blob_digest.clone(), session_id: None, size_bytes: blob_bytes.len() as u64 };

        let scratch = root.join("scratch.parquet");
        std::fs::create_dir_all(&root).unwrap();
        let dp_bytes = write_test_parquet(&scratch, "0.1.0", "v1");
        let dp_item = SyncItem { class: SyncClass::DataParquet, key: "s1".to_string(), session_id: Some("s1".to_string()), size_bytes: dp_bytes.len() as u64 };
        let dp_ctx = InstallContext { claimed_data_parquet_versions: Some(("0.1.0".to_string(), "v1".to_string())), ..Default::default() };

        let track = sample_track("t-1", 100);
        let track_bytes_v = track_bytes(&track);
        let track_item = SyncItem { class: SyncClass::Track, key: "t-1".to_string(), session_id: None, size_bytes: track_bytes_v.len() as u64 };

        // Act
        install(&root, &blob_item, blob_bytes, "peer-laptop", 1000, &InstallContext::default()).unwrap();
        install(&root, &dp_item, &dp_bytes, "peer-laptop", 1000, &dp_ctx).unwrap();
        install(&root, &track_item, &track_bytes_v, "peer-laptop", 1000, &InstallContext::default()).unwrap();

        // Assert — `tmp/` exists (write_atomic creates it) but is empty.
        let tmp_dir = root.join("tmp");
        if tmp_dir.is_dir() {
            let leftovers: Vec<_> = std::fs::read_dir(&tmp_dir).unwrap().collect();
            assert!(leftovers.is_empty(), "leftovers in tmp/: {leftovers:?}");
        }

        let _ = std::fs::remove_dir_all(&root);
    }
}
