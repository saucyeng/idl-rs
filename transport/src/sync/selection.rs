//! Selective sync (library tiers Phase 1, spec 4a): which of [`plan_sync`]'s
//! actions a run may perform. [`filter_plan`] is a pure function over
//! `(plan, selection, local manifest, remote manifest)` — no I/O — applied by
//! [`super::sync_with_peer`] right after planning. It narrows a plan; it never
//! adds an action, so every conflict rule stays core's.
//!
//! [`plan_sync`]: idl_rs::store::sync::diff::plan_sync

use std::collections::BTreeSet;

use idl_rs::store::sync::diff::{SyncAction, SyncClass, SyncItem, SyncPlan};
use idl_rs::store::sync::manifest::{Manifest, BLOB_KIND_VIDEO};
use serde::{Deserialize, Serialize};

/// Which side's actions a run keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncDirection {
    /// Pull and push (today's sync).
    #[default]
    Both,
    /// Only fetch from the peer (`Pull` and `PullForMerge`); every `Push` is dropped.
    Pull,
    /// Only send to the peer; every `Pull` and `PullForMerge` is dropped.
    Push,
}

/// What one sync run is allowed to move (library tiers spec 4a).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncSelection {
    /// Which side's actions are kept.
    pub direction: SyncDirection,
    /// `None` = every session. `Some(ids)` = only these sessions' closures: their
    /// `data.parquet`, derived channels, `session.json` and the blob named by the
    /// `blob_sha256` of whichever manifest holds the session.
    #[serde(default)]
    pub sessions: Option<BTreeSet<String>>,
    /// `None` = every workbook. `Some(ids)` = only these workbook ids.
    #[serde(default)]
    pub workbooks: Option<BTreeSet<String>>,
    /// Push only items the peer does not have at all (never an update or a merge):
    /// the "Send new" rule. Pulls are not affected by this flag.
    #[serde(default)]
    pub absent_only: bool,
}

impl SyncSelection {
    /// Today's sync: both directions, every session and workbook, no
    /// absent-only restriction.
    pub fn everything() -> Self {
        SyncSelection { direction: SyncDirection::Both, sessions: None, workbooks: None, absent_only: false }
    }
}

impl Default for SyncSelection {
    fn default() -> Self {
        Self::everything()
    }
}

/// `true` for the small classes that always travel in the run's direction(s)
/// whatever the selection: tracks, profiles and the bike log (spec 4a).
fn is_small_class(class: SyncClass) -> bool {
    matches!(class, SyncClass::Track | SyncClass::Profile | SyncClass::Bike)
}

fn has_video_blob(m: &Manifest, sha256: &str) -> bool {
    m.blobs.iter().any(|b| b.sha256 == sha256 && b.kind.as_deref() == Some(BLOB_KIND_VIDEO))
}

/// The source-blob hashes of the selected sessions, from whichever manifest
/// holds each. A selected session with no usable `blob_sha256` (an older peer)
/// or in neither manifest is logged and contributes nothing.
fn selected_blobs(ids: &BTreeSet<String>, local: &Manifest, remote: &Manifest) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for id in ids {
        let entries: Vec<_> = [local, remote].iter().filter_map(|m| m.sessions.iter().find(|s| &s.session_id == id)).collect();
        if entries.is_empty() {
            eprintln!("sync: selected session {id} is in neither manifest; ignored");
            continue;
        }
        let hashes: Vec<&str> =
            entries.iter().filter_map(|s| s.data_parquet.as_ref()).map(|d| d.blob_sha256.as_str()).filter(|h| !h.is_empty()).collect();
        if hashes.is_empty() {
            eprintln!("sync: selected session {id} has no blob_sha256 in its manifest entry (older peer); its blob is not moved");
        }
        out.extend(hashes.into_iter().map(str::to_string));
    }
    out
}

/// `true` if the peer (`remote`) already has `item`'s identity, for the
/// `absent_only` rule. Blobs and derived channels are content-addressed, so a
/// planned push of one is by construction an item the peer lacks.
fn remote_has(remote: &Manifest, item: &SyncItem) -> bool {
    let session = || item.session_id.as_deref().and_then(|id| remote.sessions.iter().find(|s| s.session_id == id));
    match item.class {
        SyncClass::Blob | SyncClass::Derived => false,
        SyncClass::DataParquet => session().is_some_and(|s| s.data_parquet.is_some()),
        SyncClass::SessionJson => session().is_some_and(|s| s.session_json.is_some()),
        SyncClass::Workbook => remote.workbooks.iter().any(|w| w.workbook_id == item.key),
        SyncClass::Track => remote.tracks.iter().any(|t| t.track_id == item.key),
        SyncClass::Profile => remote.profiles.iter().any(|p| p.profile_id == item.key),
        SyncClass::Bike => remote.bikes.iter().any(|b| b.bike_id == item.key),
    }
}

/// Narrows `plan` to `selection`. Pure; keeps action order; leaves `notes`
/// untouched. Rules, in order, per action:
///
/// 1. Direction: `Pull` drops `Push`; `Push` drops `Pull` and `PullForMerge`.
/// 2. `absent_only`: a `Push` of an item the peer already holds is dropped.
/// 3. Small classes (track, profile, bike) pass.
/// 4. Session-scoped classes pass iff `sessions` is `None` or names the session.
/// 5. Workbooks pass iff `workbooks` is `None` or names the workbook.
/// 6. Blobs: with `sessions` set, a log blob passes iff a selected session's
///    `blob_sha256` names it; a video blob (`kind: "video"`, either manifest)
///    passes unchanged so today's liveness rule (`keep_live_videos`) decides.
pub fn filter_plan(plan: SyncPlan, selection: &SyncSelection, local: &Manifest, remote: &Manifest) -> SyncPlan {
    let blobs = selection.sessions.as_ref().map(|ids| selected_blobs(ids, local, remote));

    let actions = plan
        .actions
        .into_iter()
        .filter(|action| {
            let (item, is_push) = match action {
                SyncAction::Push(i) => (i, true),
                SyncAction::Pull(i) | SyncAction::PullForMerge(i) => (i, false),
            };
            match (selection.direction, is_push) {
                (SyncDirection::Pull, true) | (SyncDirection::Push, false) => return false,
                _ => {}
            }
            if is_push && selection.absent_only && remote_has(remote, item) {
                return false;
            }
            match item.class {
                c if is_small_class(c) => true,
                SyncClass::DataParquet | SyncClass::Derived | SyncClass::SessionJson => match (&selection.sessions, &item.session_id) {
                    (None, _) => true,
                    (Some(ids), Some(id)) => ids.contains(id),
                    (Some(_), None) => false,
                },
                SyncClass::Workbook => selection.workbooks.as_ref().is_none_or(|ids| ids.contains(&item.key)),
                SyncClass::Blob => match &blobs {
                    None => true,
                    Some(hashes) => hashes.contains(&item.key) || has_video_blob(local, &item.key) || has_video_blob(remote, &item.key),
                },
                _ => true,
            }
        })
        .collect();

    SyncPlan { actions, notes: plan.notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::store::sync::manifest::{BikeEntry, BlobEntry, DataParquetEntry, DerivedEntry, ProfileEntry, SessionEntry, SessionJsonEntry, TrackEntry, WorkbookEntry};

    fn empty() -> Manifest {
        Manifest { schema_version: 1, generated_at_ms: 0, blobs: vec![], sessions: vec![], workbooks: vec![], tracks: vec![], profiles: vec![], bikes: vec![] }
    }

    fn item(class: SyncClass, key: &str, session_id: Option<&str>) -> SyncItem {
        SyncItem { class, key: key.to_string(), session_id: session_id.map(str::to_string), size_bytes: 10 }
    }

    fn session(id: &str, blob: &str) -> SessionEntry {
        SessionEntry {
            session_id: id.to_string(),
            data_parquet: Some(DataParquetEntry {
                sha256: "d".repeat(64),
                size_bytes: 1,
                importer_version: "1.0.0".to_string(),
                seam_correction_version: "v1".to_string(),
                engine_version: "1.0.0".to_string(),
                blob_sha256: blob.to_string(),
            }),
            derived: vec![DerivedEntry { sha256: "e".repeat(64), size_bytes: 1 }],
            session_json: Some(SessionJsonEntry { sha256: "f".repeat(64), size_bytes: 1, updated_at_ms: 0 }),
        }
    }

    fn blob(sha: &str, kind: Option<&str>) -> BlobEntry {
        BlobEntry { sha256: sha.to_string(), size_bytes: 1, kind: kind.map(str::to_string) }
    }

    fn set(ids: &[&str]) -> Option<BTreeSet<String>> {
        Some(ids.iter().map(|s| s.to_string()).collect())
    }

    /// A plan that pulls two sessions (a, b) completely and pushes one (c).
    fn two_pulled_one_pushed() -> (SyncPlan, Manifest, Manifest) {
        let (ba, bb, bc) = ("a1".repeat(32), "b2".repeat(32), "c3".repeat(32));
        let mut remote = empty();
        remote.sessions = vec![session("aaaa", &ba), session("bbbb", &bb)];
        remote.blobs = vec![blob(&ba, None), blob(&bb, None)];
        let mut local = empty();
        local.sessions = vec![session("cccc", &bc)];
        local.blobs = vec![blob(&bc, None)];
        let mut actions = Vec::new();
        for (id, b) in [("aaaa", &ba), ("bbbb", &bb)] {
            actions.push(SyncAction::Pull(item(SyncClass::Blob, b, None)));
            actions.push(SyncAction::Pull(item(SyncClass::DataParquet, id, Some(id))));
            actions.push(SyncAction::Pull(item(SyncClass::Derived, &"e".repeat(64), Some(id))));
            actions.push(SyncAction::Pull(item(SyncClass::SessionJson, id, Some(id))));
        }
        actions.push(SyncAction::Push(item(SyncClass::Blob, &bc, None)));
        actions.push(SyncAction::Push(item(SyncClass::DataParquet, "cccc", Some("cccc"))));
        actions.push(SyncAction::Push(item(SyncClass::SessionJson, "cccc", Some("cccc"))));
        (SyncPlan { actions, notes: vec![] }, local, remote)
    }

    fn keys(plan: &SyncPlan) -> Vec<String> {
        plan.actions
            .iter()
            .map(|a| match a {
                SyncAction::Pull(i) => format!("pull:{:?}:{}", i.class, i.key),
                SyncAction::Push(i) => format!("push:{:?}:{}", i.class, i.key),
                SyncAction::PullForMerge(i) => format!("merge:{:?}:{}", i.class, i.key),
            })
            .collect()
    }

    #[test]
    fn filter_plan_everything_leaves_the_plan_unchanged() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();

        // Act
        let out = filter_plan(plan.clone(), &SyncSelection::everything(), &local, &remote);

        // Assert
        assert_eq!(out, plan);
    }

    #[test]
    fn filter_plan_one_selected_session_keeps_its_closure_and_drops_the_others() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();
        let sel = SyncSelection { sessions: set(&["aaaa"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        let k = keys(&out);
        assert_eq!(k.len(), 4);
        assert!(k.contains(&format!("pull:Blob:{}", "a1".repeat(32))));
        assert!(k.contains(&"pull:DataParquet:aaaa".to_string()));
        assert!(k.contains(&"pull:SessionJson:aaaa".to_string()));
        assert!(k.iter().any(|s| s.starts_with("pull:Derived:")));
        assert!(!k.iter().any(|s| s.contains("bbbb") || s.contains("cccc")));
    }

    #[test]
    fn filter_plan_a_selected_session_held_only_locally_pushes_its_closure() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();
        let sel = SyncSelection { sessions: set(&["cccc"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert_eq!(keys(&out), vec![format!("push:Blob:{}", "c3".repeat(32)), "push:DataParquet:cccc".to_string(), "push:SessionJson:cccc".to_string()]);
    }

    #[test]
    fn filter_plan_an_empty_session_selection_keeps_no_session_or_blob_action() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();
        let sel = SyncSelection { sessions: set(&[]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert!(out.actions.is_empty());
    }

    #[test]
    fn filter_plan_an_unknown_session_id_is_ignored_and_selects_nothing() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();
        let sel = SyncSelection { sessions: set(&["ffff"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert!(out.actions.is_empty());
    }

    #[test]
    fn filter_plan_a_session_whose_manifest_lacks_blob_sha256_moves_without_its_blob() {
        // Arrange
        let (plan, local, mut remote) = two_pulled_one_pushed();
        remote.sessions[0].data_parquet.as_mut().unwrap().blob_sha256 = String::new();
        let sel = SyncSelection { sessions: set(&["aaaa"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        let k = keys(&out);
        assert_eq!(k.len(), 3);
        assert!(!k.iter().any(|s| s.contains(":Blob:")));
    }

    #[test]
    fn filter_plan_the_blob_hash_is_taken_from_the_local_manifest_when_only_it_has_one() {
        // Arrange
        let (plan, mut local, mut remote) = two_pulled_one_pushed();
        remote.sessions[0].data_parquet.as_mut().unwrap().blob_sha256 = String::new();
        local.sessions.push(session("aaaa", &"a1".repeat(32)));
        let sel = SyncSelection { sessions: set(&["aaaa"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert!(keys(&out).contains(&format!("pull:Blob:{}", "a1".repeat(32))));
    }

    #[test]
    fn filter_plan_direction_pull_drops_pushes_and_keeps_pulls_and_merges() {
        // Arrange
        let (mut plan, local, remote) = two_pulled_one_pushed();
        plan.actions.push(SyncAction::PullForMerge(item(SyncClass::SessionJson, "aaaa", Some("aaaa"))));
        let sel = SyncSelection { direction: SyncDirection::Pull, ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert!(out.actions.iter().all(|a| !matches!(a, SyncAction::Push(_))));
        assert!(out.actions.iter().any(|a| matches!(a, SyncAction::PullForMerge(_))));
        assert_eq!(out.actions.len(), 9);
    }

    #[test]
    fn filter_plan_direction_push_drops_pulls_and_merges_and_keeps_pushes() {
        // Arrange
        let (mut plan, local, remote) = two_pulled_one_pushed();
        plan.actions.push(SyncAction::PullForMerge(item(SyncClass::SessionJson, "aaaa", Some("aaaa"))));
        let sel = SyncSelection { direction: SyncDirection::Push, ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &local, &remote);

        // Assert
        assert_eq!(out.actions.len(), 3);
        assert!(out.actions.iter().all(|a| matches!(a, SyncAction::Push(_))));
    }

    #[test]
    fn filter_plan_small_classes_travel_in_the_run_direction_whatever_the_selection() {
        // Arrange
        let mut plan = SyncPlan::default();
        plan.actions = vec![
            SyncAction::Pull(item(SyncClass::Track, "t1", None)),
            SyncAction::Push(item(SyncClass::Profile, "p1", None)),
            SyncAction::PullForMerge(item(SyncClass::Bike, "gt", None)),
            SyncAction::Push(item(SyncClass::Bike, "gt2", None)),
        ];
        let sel = SyncSelection { sessions: set(&[]), workbooks: set(&[]), ..SyncSelection::everything() };
        let pull_sel = SyncSelection { direction: SyncDirection::Pull, ..sel.clone() };

        // Act
        let all = filter_plan(plan.clone(), &sel, &empty(), &empty());
        let pull = filter_plan(plan, &pull_sel, &empty(), &empty());

        // Assert
        assert_eq!(all.actions.len(), 4);
        assert_eq!(pull.actions.len(), 2);
    }

    #[test]
    fn filter_plan_workbooks_filter_keeps_only_the_named_ids() {
        // Arrange
        let plan = SyncPlan {
            actions: vec![
                SyncAction::Pull(item(SyncClass::Workbook, "w1", None)),
                SyncAction::PullForMerge(item(SyncClass::Workbook, "w2", None)),
                SyncAction::Push(item(SyncClass::Workbook, "w3", None)),
            ],
            notes: vec![],
        };
        let sel = SyncSelection { workbooks: set(&["w2", "w3"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &empty(), &empty());

        // Assert
        assert_eq!(keys(&out), vec!["merge:Workbook:w2".to_string(), "push:Workbook:w3".to_string()]);
    }

    #[test]
    fn filter_plan_a_workbook_selection_does_not_narrow_sessions() {
        // Arrange
        let (plan, local, remote) = two_pulled_one_pushed();
        let sel = SyncSelection { workbooks: set(&[]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan.clone(), &sel, &local, &remote);

        // Assert
        assert_eq!(out, plan);
    }

    #[test]
    fn filter_plan_absent_only_drops_pushes_of_items_the_peer_has() {
        // Arrange
        let mut remote = empty();
        remote.sessions = vec![session("aaaa", &"a1".repeat(32))];
        remote.workbooks = vec![WorkbookEntry { workbook_id: "w1".into(), file_name: "w".into(), sha256: "0".into(), size_bytes: 1, updated_at_ms: 0 }];
        remote.tracks = vec![TrackEntry { track_id: "t1".into(), sha256: "0".into(), size_bytes: 1, updated_at_ms: 0 }];
        remote.profiles = vec![ProfileEntry { profile_id: "p1".into(), sha256: "0".into(), size_bytes: 1, updated_at_ms: 0 }];
        remote.bikes = vec![BikeEntry { bike_id: "gt".into(), sha256: "0".into(), size_bytes: 1 }];
        let plan = SyncPlan {
            actions: vec![
                SyncAction::Push(item(SyncClass::DataParquet, "aaaa", Some("aaaa"))),
                SyncAction::Push(item(SyncClass::SessionJson, "aaaa", Some("aaaa"))),
                SyncAction::Push(item(SyncClass::Track, "t1", None)),
                SyncAction::Push(item(SyncClass::Profile, "p1", None)),
                SyncAction::Push(item(SyncClass::Bike, "gt", None)),
                SyncAction::Push(item(SyncClass::Workbook, "w1", None)),
                SyncAction::Push(item(SyncClass::DataParquet, "bbbb", Some("bbbb"))),
                SyncAction::Push(item(SyncClass::Track, "t2", None)),
                SyncAction::Push(item(SyncClass::Blob, &"b2".repeat(32), None)),
            ],
            notes: vec![],
        };
        let sel = SyncSelection { direction: SyncDirection::Push, absent_only: true, ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &empty(), &remote);

        // Assert
        assert_eq!(keys(&out), vec!["push:DataParquet:bbbb".to_string(), "push:Track:t2".to_string(), format!("push:Blob:{}", "b2".repeat(32))]);
    }

    #[test]
    fn filter_plan_absent_only_with_direction_both_leaves_pulls_alone() {
        // Arrange
        let plan = SyncPlan {
            actions: vec![SyncAction::Pull(item(SyncClass::Track, "t1", None)), SyncAction::PullForMerge(item(SyncClass::Workbook, "w1", None))],
            notes: vec![],
        };
        let sel = SyncSelection { absent_only: true, ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan.clone(), &sel, &empty(), &empty());

        // Assert
        assert_eq!(out, plan);
    }

    #[test]
    fn filter_plan_a_video_blob_passes_a_session_selection_for_the_liveness_rule_to_decide() {
        // Arrange
        let vid = "9a".repeat(32);
        let mut remote = empty();
        remote.blobs = vec![blob(&vid, Some(BLOB_KIND_VIDEO)), blob(&"77".repeat(32), None)];
        let plan = SyncPlan {
            actions: vec![SyncAction::Pull(item(SyncClass::Blob, &vid, None)), SyncAction::Pull(item(SyncClass::Blob, &"77".repeat(32), None))],
            notes: vec![],
        };
        let sel = SyncSelection { sessions: set(&["aaaa"]), ..SyncSelection::everything() };

        // Act
        let out = filter_plan(plan, &sel, &empty(), &remote);

        // Assert
        assert_eq!(keys(&out), vec![format!("pull:Blob:{vid}")]);
    }

    #[test]
    fn filter_plan_notes_are_left_untouched() {
        // Arrange
        let note = idl_rs::store::sync::diff::SyncNote {
            class: SyncClass::DataParquet,
            key: "k".into(),
            reason: idl_rs::store::sync::diff::SyncNoteReason::EquivalentDataParquet,
        };
        let plan = SyncPlan { actions: vec![], notes: vec![note.clone()] };

        // Act
        let out = filter_plan(plan, &SyncSelection { sessions: set(&[]), ..SyncSelection::everything() }, &empty(), &empty());

        // Assert
        assert_eq!(out.notes, vec![note]);
    }
}
