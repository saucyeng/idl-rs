//! The pure sync diff (C4 §6, ruling R88/R89): `plan_sync` turns two sides'
//! [`Manifest`]s into the set of transfers a sync run must perform. No I/O,
//! no clock, no network — every conflict rule below is quoted from C4 §6's
//! entry table, not invented (CLAUDE.md §1). `local` is this machine; the
//! function is otherwise symmetric in its two arguments (swapping them
//! mirrors every `Pull`/`Push` pair, tested below).

use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::manifest::{DataParquetEntry, Manifest, ProfileEntry, SessionEntry, TrackEntry, WorkbookEntry};

/// One unit of work a sync run performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncAction {
    /// Fetch this file from the peer and install it locally.
    Pull(SyncItem),
    /// Offer this file to the peer.
    Push(SyncItem),
    /// Both sides hold it; fetch the peer's copy so the receiver can merge
    /// locally (workbooks and `session.json` only — the merge needs both
    /// full documents).
    PullForMerge(SyncItem),
}

/// Which file, in which class. `session_id` is set only for session-scoped
/// classes (`data.parquet`, derived channels, `session.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncItem {
    pub class: SyncClass,
    /// `sha256` | `session_id` | `workbook_id` | `track_id` | `profile_id`,
    /// per [`SyncClass`].
    pub key: String,
    pub session_id: Option<String>,
    /// Size, in bytes, of the copy this action transfers (the side actually
    /// being fetched or offered — for `PullForMerge` and a version-losing
    /// `data.parquet`, this is the remote/authoritative side's size).
    pub size_bytes: u64,
}

/// A manifest entry's file class (C4 §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncClass {
    Blob,
    DataParquet,
    Derived,
    SessionJson,
    Workbook,
    Track,
    Profile,
}

/// Why nothing is transferred for an item that differs on both sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncNote {
    pub class: SyncClass,
    pub key: String,
    pub reason: SyncNoteReason,
}

/// A session's two `data.parquet` version pairs, `(importer_version,
/// seam_correction_version)` — carried by [`SyncNoteReason::IncomparableDataParquetVersion`]
/// so the note names exactly what could not be ordered (ruling R89).
pub type VersionPair = (String, String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncNoteReason {
    /// `data.parquet`: version pair matches, hashes differ — a last-ulp
    /// cross-CPU difference (design §5). Keep local, transfer nothing.
    EquivalentDataParquet,
    /// `data.parquet`: one side's `importer_version` or
    /// `seam_correction_version` does not parse under ruling R89's ordering
    /// rule, so the pair cannot be compared. Neither side is authoritative;
    /// nothing transfers for this session's `data.parquet` (ruling R89).
    IncomparableDataParquetVersion { local: VersionPair, remote: VersionPair },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SyncPlan {
    pub actions: Vec<SyncAction>,
    pub notes: Vec<SyncNote>,
}

/// Computes what a sync between these two sides must move. Pure and
/// deterministic; `local` is this machine. Never treats "absent from one
/// side's manifest" as a deletion (Task 2's note) — every class here either
/// transfers the missing copy (no deletion is ever propagated by this
/// function) or, for `data.parquet`/workbook/`session.json` present on both
/// sides, applies that class's own conflict rule.
pub fn plan_sync(local: &Manifest, remote: &Manifest) -> SyncPlan {
    let mut actions = Vec::new();
    let mut notes = Vec::new();

    diff_blobs(local, remote, &mut actions);

    let session_ids: BTreeSet<&str> =
        local.sessions.iter().map(|s| s.session_id.as_str()).chain(remote.sessions.iter().map(|s| s.session_id.as_str())).collect();
    for session_id in session_ids {
        let l = find_session(local, session_id);
        let r = find_session(remote, session_id);

        let (dp_action, dp_note) =
            diff_data_parquet(session_id, l.and_then(|s| s.data_parquet.as_ref()), r.and_then(|s| s.data_parquet.as_ref()));
        actions.extend(dp_action);
        notes.extend(dp_note);

        diff_derived(session_id, l.map(|s| s.derived.as_slice()).unwrap_or(&[]), r.map(|s| s.derived.as_slice()).unwrap_or(&[]), &mut actions);

        if let Some(a) = diff_session_json(session_id, l.and_then(|s| s.session_json.as_ref()), r.and_then(|s| s.session_json.as_ref())) {
            actions.push(a);
        }
    }

    let workbook_ids: BTreeSet<&str> =
        local.workbooks.iter().map(|w| w.workbook_id.as_str()).chain(remote.workbooks.iter().map(|w| w.workbook_id.as_str())).collect();
    for workbook_id in workbook_ids {
        let l = local.workbooks.iter().find(|w| w.workbook_id == workbook_id);
        let r = remote.workbooks.iter().find(|w| w.workbook_id == workbook_id);
        if let Some(a) = diff_workbook(workbook_id, l, r) {
            actions.push(a);
        }
    }

    let track_ids: BTreeSet<&str> = local.tracks.iter().map(|t| t.track_id.as_str()).chain(remote.tracks.iter().map(|t| t.track_id.as_str())).collect();
    for track_id in track_ids {
        let l = local.tracks.iter().find(|t| t.track_id == track_id);
        let r = remote.tracks.iter().find(|t| t.track_id == track_id);
        if let Some(a) = lww_diff(SyncClass::Track, track_id, l.map(TrackLww), r.map(TrackLww)) {
            actions.push(a);
        }
    }

    let profile_ids: BTreeSet<&str> =
        local.profiles.iter().map(|p| p.profile_id.as_str()).chain(remote.profiles.iter().map(|p| p.profile_id.as_str())).collect();
    for profile_id in profile_ids {
        let l = local.profiles.iter().find(|p| p.profile_id == profile_id);
        let r = remote.profiles.iter().find(|p| p.profile_id == profile_id);
        if let Some(a) = lww_diff(SyncClass::Profile, profile_id, l.map(ProfileLww), r.map(ProfileLww)) {
            actions.push(a);
        }
    }

    actions.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
    notes.sort_by(|a, b| (class_rank(a.class), a.key.as_str()).cmp(&(class_rank(b.class), b.key.as_str())));

    SyncPlan { actions, notes }
}

fn find_session<'a>(m: &'a Manifest, session_id: &str) -> Option<&'a SessionEntry> {
    m.sessions.iter().find(|s| s.session_id == session_id)
}

fn class_rank(c: SyncClass) -> u8 {
    match c {
        SyncClass::Blob => 0,
        SyncClass::DataParquet => 1,
        SyncClass::Derived => 2,
        SyncClass::SessionJson => 3,
        SyncClass::Workbook => 4,
        SyncClass::Track => 5,
        SyncClass::Profile => 6,
    }
}

fn item_of(action: &SyncAction) -> &SyncItem {
    match action {
        SyncAction::Pull(i) | SyncAction::Push(i) | SyncAction::PullForMerge(i) => i,
    }
}

fn sort_key(action: &SyncAction) -> (u8, &str, &str) {
    let item = item_of(action);
    (class_rank(item.class), item.key.as_str(), item.session_id.as_deref().unwrap_or(""))
}

fn diff_blobs(local: &Manifest, remote: &Manifest, actions: &mut Vec<SyncAction>) {
    let local_hashes: BTreeSet<&str> = local.blobs.iter().map(|b| b.sha256.as_str()).collect();
    let remote_hashes: BTreeSet<&str> = remote.blobs.iter().map(|b| b.sha256.as_str()).collect();

    for hash in remote_hashes.difference(&local_hashes) {
        let entry = remote.blobs.iter().find(|b| b.sha256 == *hash).expect("hash came from remote.blobs");
        actions.push(SyncAction::Pull(SyncItem { class: SyncClass::Blob, key: hash.to_string(), session_id: None, size_bytes: entry.size_bytes }));
    }
    for hash in local_hashes.difference(&remote_hashes) {
        let entry = local.blobs.iter().find(|b| b.sha256 == *hash).expect("hash came from local.blobs");
        actions.push(SyncAction::Push(SyncItem { class: SyncClass::Blob, key: hash.to_string(), session_id: None, size_bytes: entry.size_bytes }));
    }
}

fn diff_derived(session_id: &str, local: &[super::manifest::DerivedEntry], remote: &[super::manifest::DerivedEntry], actions: &mut Vec<SyncAction>) {
    let local_hashes: BTreeSet<&str> = local.iter().map(|d| d.sha256.as_str()).collect();
    let remote_hashes: BTreeSet<&str> = remote.iter().map(|d| d.sha256.as_str()).collect();

    for hash in remote_hashes.difference(&local_hashes) {
        let entry = remote.iter().find(|d| d.sha256 == *hash).expect("hash came from remote derived list");
        actions.push(SyncAction::Pull(SyncItem {
            class: SyncClass::Derived,
            key: hash.to_string(),
            session_id: Some(session_id.to_string()),
            size_bytes: entry.size_bytes,
        }));
    }
    for hash in local_hashes.difference(&remote_hashes) {
        let entry = local.iter().find(|d| d.sha256 == *hash).expect("hash came from local derived list");
        actions.push(SyncAction::Push(SyncItem {
            class: SyncClass::Derived,
            key: hash.to_string(),
            session_id: Some(session_id.to_string()),
            size_bytes: entry.size_bytes,
        }));
    }
}

/// Ruling R89's `(importer_version, seam_correction_version)` ordering:
/// `importer_version` compared first as SemVer 2.0.0 core (`major.minor.patch`,
/// pre-release/build metadata ignored — no fixture in this codebase carries
/// either); only if equal, `seam_correction_version` compared as the integer
/// after its leading `v`. A value that fails to parse under its own rule
/// makes the whole pair incomparable — never falls back to another
/// comparison.
fn compare_data_parquet_versions(local: &DataParquetEntry, remote: &DataParquetEntry) -> Option<Ordering> {
    let local_importer = parse_semver_core(&local.importer_version)?;
    let remote_importer = parse_semver_core(&remote.importer_version)?;
    match local_importer.cmp(&remote_importer) {
        Ordering::Equal => {
            let local_seam = parse_seam_version(&local.seam_correction_version)?;
            let remote_seam = parse_seam_version(&remote.seam_correction_version)?;
            Some(local_seam.cmp(&remote_seam))
        }
        other => Some(other),
    }
}

/// Parses a SemVer 2.0.0 string's `major.minor.patch` core for ordering
/// (ruling R89). Pre-release/build metadata after `-`/`+` is stripped, not
/// compared. `None` if the core is not exactly three numeric components.
fn parse_semver_core(s: &str) -> Option<(u64, u64, u64)> {
    let core = s.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Parses `seam_correction_version` (e.g. `v1`) as the integer after its
/// leading `v` (ruling R89). `None` if there is no leading `v` or the rest
/// is not a plain non-negative integer.
fn parse_seam_version(s: &str) -> Option<u64> {
    s.strip_prefix('v')?.parse().ok()
}

fn diff_data_parquet(
    session_id: &str,
    local: Option<&DataParquetEntry>,
    remote: Option<&DataParquetEntry>,
) -> (Option<SyncAction>, Option<SyncNote>) {
    match (local, remote) {
        (None, None) => (None, None),
        (None, Some(r)) => (
            Some(SyncAction::Pull(SyncItem {
                class: SyncClass::DataParquet,
                key: session_id.to_string(),
                session_id: Some(session_id.to_string()),
                size_bytes: r.size_bytes,
            })),
            None,
        ),
        (Some(l), None) => (
            Some(SyncAction::Push(SyncItem {
                class: SyncClass::DataParquet,
                key: session_id.to_string(),
                session_id: Some(session_id.to_string()),
                size_bytes: l.size_bytes,
            })),
            None,
        ),
        (Some(l), Some(r)) => match compare_data_parquet_versions(l, r) {
            None => (
                None,
                Some(SyncNote {
                    class: SyncClass::DataParquet,
                    key: session_id.to_string(),
                    reason: SyncNoteReason::IncomparableDataParquetVersion {
                        local: (l.importer_version.clone(), l.seam_correction_version.clone()),
                        remote: (r.importer_version.clone(), r.seam_correction_version.clone()),
                    },
                }),
            ),
            Some(Ordering::Equal) => {
                if l.sha256 == r.sha256 {
                    (None, None)
                } else {
                    (None, Some(SyncNote { class: SyncClass::DataParquet, key: session_id.to_string(), reason: SyncNoteReason::EquivalentDataParquet }))
                }
            }
            Some(Ordering::Less) => (
                Some(SyncAction::Pull(SyncItem {
                    class: SyncClass::DataParquet,
                    key: session_id.to_string(),
                    session_id: Some(session_id.to_string()),
                    size_bytes: r.size_bytes,
                })),
                None,
            ),
            Some(Ordering::Greater) => (
                Some(SyncAction::Push(SyncItem {
                    class: SyncClass::DataParquet,
                    key: session_id.to_string(),
                    session_id: Some(session_id.to_string()),
                    size_bytes: l.size_bytes,
                })),
                None,
            ),
        },
    }
}

fn diff_session_json(
    session_id: &str,
    local: Option<&super::manifest::SessionJsonEntry>,
    remote: Option<&super::manifest::SessionJsonEntry>,
) -> Option<SyncAction> {
    match (local, remote) {
        (None, None) => None,
        (None, Some(r)) => Some(SyncAction::Pull(SyncItem {
            class: SyncClass::SessionJson,
            key: session_id.to_string(),
            session_id: Some(session_id.to_string()),
            size_bytes: r.size_bytes,
        })),
        (Some(l), None) => Some(SyncAction::Push(SyncItem {
            class: SyncClass::SessionJson,
            key: session_id.to_string(),
            session_id: Some(session_id.to_string()),
            size_bytes: l.size_bytes,
        })),
        (Some(l), Some(r)) => {
            if l.sha256 == r.sha256 {
                None
            } else {
                Some(SyncAction::PullForMerge(SyncItem {
                    class: SyncClass::SessionJson,
                    key: session_id.to_string(),
                    session_id: Some(session_id.to_string()),
                    size_bytes: r.size_bytes,
                }))
            }
        }
    }
}

/// Workbooks key on `workbook_id`, never `file_name` (C4 §6) — a rename
/// with unchanged content hashes identically since `file_name` is not part
/// of the hashed bytes, so it naturally falls into the "equal hashes ⇒ no
/// action" arm below without special-casing the rename.
fn diff_workbook(workbook_id: &str, local: Option<&WorkbookEntry>, remote: Option<&WorkbookEntry>) -> Option<SyncAction> {
    match (local, remote) {
        (None, None) => None,
        (None, Some(r)) => {
            Some(SyncAction::Pull(SyncItem { class: SyncClass::Workbook, key: workbook_id.to_string(), session_id: None, size_bytes: r.size_bytes }))
        }
        (Some(l), None) => {
            Some(SyncAction::Push(SyncItem { class: SyncClass::Workbook, key: workbook_id.to_string(), session_id: None, size_bytes: l.size_bytes }))
        }
        (Some(l), Some(r)) => {
            if l.sha256 == r.sha256 {
                None
            } else {
                Some(SyncAction::PullForMerge(SyncItem {
                    class: SyncClass::Workbook,
                    key: workbook_id.to_string(),
                    session_id: None,
                    size_bytes: r.size_bytes,
                }))
            }
        }
    }
}

/// Common shape [`lww_diff`] needs from [`TrackEntry`]/[`ProfileEntry`],
/// borrowed rather than duplicating the whole match arm per class.
trait LwwFields {
    fn sha256(&self) -> &str;
    fn size_bytes(&self) -> u64;
    fn updated_at_ms(&self) -> i64;
}

struct TrackLww<'a>(&'a TrackEntry);
impl LwwFields for TrackLww<'_> {
    fn sha256(&self) -> &str {
        &self.0.sha256
    }
    fn size_bytes(&self) -> u64 {
        self.0.size_bytes
    }
    fn updated_at_ms(&self) -> i64 {
        self.0.updated_at_ms
    }
}

struct ProfileLww<'a>(&'a ProfileEntry);
impl LwwFields for ProfileLww<'_> {
    fn sha256(&self) -> &str {
        &self.0.sha256
    }
    fn size_bytes(&self) -> u64 {
        self.0.size_bytes
    }
    fn updated_at_ms(&self) -> i64 {
        self.0.updated_at_ms
    }
}

/// Track/Profile conflict rule (C4 §6): last-write-wins by `updated_at_ms`.
/// Equal timestamps with differing hashes keeps local, no action — the
/// same rule extends naturally to "present on one side only" (nothing to
/// compare against, so the present side simply transfers).
fn lww_diff<T: LwwFields>(class: SyncClass, key: &str, local: Option<T>, remote: Option<T>) -> Option<SyncAction> {
    match (local, remote) {
        (None, None) => None,
        (None, Some(r)) => Some(SyncAction::Pull(SyncItem { class, key: key.to_string(), session_id: None, size_bytes: r.size_bytes() })),
        (Some(l), None) => Some(SyncAction::Push(SyncItem { class, key: key.to_string(), session_id: None, size_bytes: l.size_bytes() })),
        (Some(l), Some(r)) => {
            if l.sha256() == r.sha256() {
                return None;
            }
            match l.updated_at_ms().cmp(&r.updated_at_ms()) {
                Ordering::Less => Some(SyncAction::Pull(SyncItem { class, key: key.to_string(), session_id: None, size_bytes: r.size_bytes() })),
                Ordering::Greater => Some(SyncAction::Push(SyncItem { class, key: key.to_string(), session_id: None, size_bytes: l.size_bytes() })),
                Ordering::Equal => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::sync::manifest::{BlobEntry, DerivedEntry, SessionJsonEntry};

    fn empty_manifest() -> Manifest {
        Manifest { schema_version: 1, generated_at_ms: 0, blobs: Vec::new(), sessions: Vec::new(), workbooks: Vec::new(), tracks: Vec::new(), profiles: Vec::new() }
    }

    fn dp(sha256: &str, importer_version: &str, seam_correction_version: &str) -> DataParquetEntry {
        DataParquetEntry {
            sha256: sha256.to_string(),
            size_bytes: 100,
            importer_version: importer_version.to_string(),
            seam_correction_version: seam_correction_version.to_string(),
            engine_version: "9.9.9".to_string(),
        }
    }

    #[test]
    fn plan_sync_a_blob_only_on_the_remote_one_pull_only_local_one_push() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        remote.blobs.push(BlobEntry { sha256: "r".repeat(64), size_bytes: 10 });
        local.blobs.push(BlobEntry { sha256: "l".repeat(64), size_bytes: 20 });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert_eq!(plan.actions.len(), 2);
        assert!(plan.actions.contains(&SyncAction::Pull(SyncItem { class: SyncClass::Blob, key: "r".repeat(64), session_id: None, size_bytes: 10 })));
        assert!(plan.actions.contains(&SyncAction::Push(SyncItem { class: SyncClass::Blob, key: "l".repeat(64), session_id: None, size_bytes: 20 })));
    }

    #[test]
    fn plan_sync_identical_manifests_an_empty_plan() {
        // Arrange
        let mut m = empty_manifest();
        m.blobs.push(BlobEntry { sha256: "a".repeat(64), size_bytes: 5 });
        m.sessions.push(SessionEntry {
            session_id: "s1".to_string(),
            data_parquet: Some(dp("h1", "0.1.0", "v1")),
            derived: vec![DerivedEntry { sha256: "d1".repeat(8), size_bytes: 1 }],
            session_json: Some(SessionJsonEntry { sha256: "sj1".to_string(), size_bytes: 2, updated_at_ms: 10 }),
        });

        // Act
        let plan = plan_sync(&m, &m);

        // Assert
        assert!(plan.actions.is_empty());
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn plan_sync_data_parquet_same_version_pair_different_hash_no_action_and_one_equivalent_note() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.3.0", "v1")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h2", "0.3.0", "v1")), derived: Vec::new(), session_json: None });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert!(plan.actions.is_empty());
        assert_eq!(plan.notes, vec![SyncNote { class: SyncClass::DataParquet, key: "s1".to_string(), reason: SyncNoteReason::EquivalentDataParquet }]);
    }

    #[test]
    fn plan_sync_data_parquet_newer_remote_importer_version_pull() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v9")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h2", "0.2.0", "v1")), derived: Vec::new(), session_json: None });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert — importer newer wins regardless of seam (ruling R89).
        assert_eq!(
            plan.actions,
            vec![SyncAction::Pull(SyncItem { class: SyncClass::DataParquet, key: "s1".to_string(), session_id: Some("s1".to_string()), size_bytes: 100 })]
        );
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn plan_sync_data_parquet_equal_importer_newer_seam_wins() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v1")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h2", "0.1.0", "v2")), derived: Vec::new(), session_json: None });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert_eq!(
            plan.actions,
            vec![SyncAction::Pull(SyncItem { class: SyncClass::DataParquet, key: "s1".to_string(), session_id: Some("s1".to_string()), size_bytes: 100 })]
        );
    }

    #[test]
    fn plan_sync_data_parquet_unparsable_seam_warning_and_no_item() {
        // Arrange — importer_version equal, so ordering falls to
        // seam_correction_version, and remote's does not parse (ruling R89).
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v1")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry {
            session_id: "s1".to_string(),
            data_parquet: Some(dp("h2", "0.1.0", "not-a-version")),
            derived: Vec::new(),
            session_json: None,
        });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert!(plan.actions.is_empty());
        assert_eq!(
            plan.notes,
            vec![SyncNote {
                class: SyncClass::DataParquet,
                key: "s1".to_string(),
                reason: SyncNoteReason::IncomparableDataParquetVersion {
                    local: ("0.1.0".to_string(), "v1".to_string()),
                    remote: ("0.1.0".to_string(), "not-a-version".to_string()),
                },
            }]
        );
    }

    #[test]
    fn plan_sync_data_parquet_equal_pair_equal_hash_nothing() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v1")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v1")), derived: Vec::new(), session_json: None });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert!(plan.actions.is_empty());
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn plan_sync_session_json_differing_pull_for_merge_never_a_bare_pull() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry {
            session_id: "s1".to_string(),
            data_parquet: None,
            derived: Vec::new(),
            session_json: Some(SessionJsonEntry { sha256: "l".to_string(), size_bytes: 1, updated_at_ms: 1 }),
        });
        remote.sessions.push(SessionEntry {
            session_id: "s1".to_string(),
            data_parquet: None,
            derived: Vec::new(),
            session_json: Some(SessionJsonEntry { sha256: "r".to_string(), size_bytes: 2, updated_at_ms: 2 }),
        });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert_eq!(
            plan.actions,
            vec![SyncAction::PullForMerge(SyncItem {
                class: SyncClass::SessionJson,
                key: "s1".to_string(),
                session_id: Some("s1".to_string()),
                size_bytes: 2,
            })]
        );
    }

    #[test]
    fn plan_sync_a_workbook_differing_pull_for_merge() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.workbooks.push(WorkbookEntry { workbook_id: "w1".to_string(), file_name: "a".to_string(), sha256: "l".to_string(), size_bytes: 1, updated_at_ms: 1 });
        remote.workbooks.push(WorkbookEntry {
            workbook_id: "w1".to_string(),
            file_name: "a".to_string(),
            sha256: "r".to_string(),
            size_bytes: 2,
            updated_at_ms: 2,
        });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert_eq!(
            plan.actions,
            vec![SyncAction::PullForMerge(SyncItem { class: SyncClass::Workbook, key: "w1".to_string(), session_id: None, size_bytes: 2 })]
        );
    }

    #[test]
    fn plan_sync_a_workbook_with_the_same_id_and_a_different_file_name_no_transfer_action() {
        // Arrange — same bytes (so same hash), different file_name: a
        // rename, reconciled locally by Task 6, not a sync transfer.
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.workbooks.push(WorkbookEntry {
            workbook_id: "w1".to_string(),
            file_name: "old-name".to_string(),
            sha256: "same".to_string(),
            size_bytes: 1,
            updated_at_ms: 1,
        });
        remote.workbooks.push(WorkbookEntry {
            workbook_id: "w1".to_string(),
            file_name: "new-name".to_string(),
            sha256: "same".to_string(),
            size_bytes: 1,
            updated_at_ms: 2,
        });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn plan_sync_a_track_newer_on_the_remote_pull_newer_locally_push_equal_updated_at_ms_different_hashes_no_action() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.tracks.push(TrackEntry { track_id: "t-newer-remote".to_string(), sha256: "l".to_string(), size_bytes: 1, updated_at_ms: 1 });
        remote.tracks.push(TrackEntry { track_id: "t-newer-remote".to_string(), sha256: "r".to_string(), size_bytes: 2, updated_at_ms: 2 });
        local.tracks.push(TrackEntry { track_id: "t-newer-local".to_string(), sha256: "l".to_string(), size_bytes: 3, updated_at_ms: 5 });
        remote.tracks.push(TrackEntry { track_id: "t-newer-local".to_string(), sha256: "r".to_string(), size_bytes: 4, updated_at_ms: 1 });
        local.tracks.push(TrackEntry { track_id: "t-tie".to_string(), sha256: "l".to_string(), size_bytes: 5, updated_at_ms: 9 });
        remote.tracks.push(TrackEntry { track_id: "t-tie".to_string(), sha256: "r".to_string(), size_bytes: 6, updated_at_ms: 9 });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert
        assert_eq!(plan.actions.len(), 2);
        assert!(plan.actions.contains(&SyncAction::Pull(SyncItem {
            class: SyncClass::Track,
            key: "t-newer-remote".to_string(),
            session_id: None,
            size_bytes: 2,
        })));
        assert!(plan.actions.contains(&SyncAction::Push(SyncItem {
            class: SyncClass::Track,
            key: "t-newer-local".to_string(),
            session_id: None,
            size_bytes: 3,
        })));
    }

    #[test]
    fn plan_sync_derived_channels_differing_per_session_the_right_session_id_on_every_item() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: None, derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry {
            session_id: "s1".to_string(),
            data_parquet: None,
            derived: vec![DerivedEntry { sha256: "shared-hash".to_string(), size_bytes: 7 }],
            session_json: None,
        });
        local.sessions.push(SessionEntry {
            session_id: "s2".to_string(),
            data_parquet: None,
            derived: vec![DerivedEntry { sha256: "shared-hash".to_string(), size_bytes: 7 }],
            session_json: None,
        });
        remote.sessions.push(SessionEntry { session_id: "s2".to_string(), data_parquet: None, derived: Vec::new(), session_json: None });

        // Act
        let plan = plan_sync(&local, &remote);

        // Assert — the same sha256 appears once per session, each tagged
        // with its own session_id, not merged into one item.
        assert_eq!(plan.actions.len(), 2);
        assert!(plan.actions.contains(&SyncAction::Pull(SyncItem {
            class: SyncClass::Derived,
            key: "shared-hash".to_string(),
            session_id: Some("s1".to_string()),
            size_bytes: 7,
        })));
        assert!(plan.actions.contains(&SyncAction::Push(SyncItem {
            class: SyncClass::Derived,
            key: "shared-hash".to_string(),
            session_id: Some("s2".to_string()),
            size_bytes: 7,
        })));
    }

    #[test]
    fn plan_sync_swapping_the_arguments_pull_and_push_mirror_exactly() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.blobs.push(BlobEntry { sha256: "l".repeat(64), size_bytes: 1 });
        remote.blobs.push(BlobEntry { sha256: "r".repeat(64), size_bytes: 2 });
        local.tracks.push(TrackEntry { track_id: "t1".to_string(), sha256: "l".to_string(), size_bytes: 3, updated_at_ms: 1 });
        remote.tracks.push(TrackEntry { track_id: "t1".to_string(), sha256: "r".to_string(), size_bytes: 4, updated_at_ms: 2 });

        // Act
        let forward = plan_sync(&local, &remote);
        let backward = plan_sync(&remote, &local);

        // Assert
        assert_eq!(forward.actions.len(), backward.actions.len());
        for action in &forward.actions {
            let mirrored = match action {
                SyncAction::Pull(item) => SyncAction::Push(item.clone()),
                SyncAction::Push(item) => SyncAction::Pull(item.clone()),
                SyncAction::PullForMerge(item) => SyncAction::PullForMerge(item.clone()),
            };
            assert!(backward.actions.contains(&mirrored), "missing mirror of {action:?} in {backward:?}");
        }
    }

    #[test]
    fn plan_sync_run_twice_identical_plans() {
        // Arrange
        let mut local = empty_manifest();
        let mut remote = empty_manifest();
        local.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h1", "0.1.0", "v1")), derived: Vec::new(), session_json: None });
        remote.sessions.push(SessionEntry { session_id: "s1".to_string(), data_parquet: Some(dp("h2", "0.2.0", "v1")), derived: Vec::new(), session_json: None });
        local.tracks.push(TrackEntry { track_id: "t1".to_string(), sha256: "l".to_string(), size_bytes: 1, updated_at_ms: 1 });
        remote.tracks.push(TrackEntry { track_id: "t1".to_string(), sha256: "r".to_string(), size_bytes: 2, updated_at_ms: 2 });

        // Act
        let first = plan_sync(&local, &remote);
        let second = plan_sync(&local, &remote);

        // Assert
        assert_eq!(first, second);
    }
}
