//! Per-cell merge decisions (C2 §7.2): a cell id's state on each side
//! relative to `base`, and the sixteen-cell decision table that turns a
//! pair of states into a [`CellOutcome`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::workbook::v3::{CellDoc, WorkbookDoc};

use super::{CellOutcome, MergeWarning};

/// A cell's state relative to `base` on one side (C2 §7.2): `Unchanged`
/// (present in base, byte-identical content), `Changed` (present in base,
/// different content), `Deleted` (present in base, absent from this side),
/// `Added` (absent from base, present in this side).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Present in `base`, byte-identical content on this side.
    Unchanged,
    /// Present in `base`, different content on this side.
    Changed,
    /// Absent from `base`, present on this side.
    Added,
    /// Present in `base`, absent from this side.
    Deleted,
}

/// `true` when `a` and `b` carry the same content for merge purposes: fence
/// kind, raw fence body, and both prose spans. C2 §7.3's closing paragraph
/// — "prose travels with its cell" — means a prose-only edit (fence body
/// unchanged) must still compare unequal here, so `Changed` is reported for
/// it exactly as for a fence-body edit.
fn same_content(a: &CellDoc, b: &CellDoc) -> bool {
    a.kind_token == b.kind_token
        && a.raw_fence_body == b.raw_fence_body
        && a.prose_before == b.prose_before
        && a.prose_after == b.prose_after
}

/// Indexes `doc`'s cells by id for `O(1)` lookup during the per-id merge
/// loop. C2 §2.2 guarantees ids are meant to be unique; a document that
/// (erroneously) has a duplicate id keeps only the last cell scanned under
/// that id — the same last-wins behaviour a `HashMap` gives for free, and
/// no worse than picking arbitrarily between two already-invalid states.
fn index_by_id(doc: &WorkbookDoc) -> HashMap<&str, &CellDoc> {
    doc.cells.iter().map(|c| (c.id.as_str(), c)).collect()
}

/// One side's [`CellState`] given whether `base` and that side each hold a
/// cell for this id. Only ever called here with `base_cell: Some(_)` (see
/// [`decide_cells`]) — the `None` arm is kept so the match is total rather
/// than partial, never so it need be reached.
fn side_state(base_cell: Option<&CellDoc>, side_cell: Option<&CellDoc>) -> CellState {
    match (base_cell, side_cell) {
        (Some(b), Some(s)) => {
            if same_content(b, s) {
                CellState::Unchanged
            } else {
                CellState::Changed
            }
        }
        (Some(_), None) => CellState::Deleted,
        (None, Some(_)) => CellState::Added,
        (None, None) => CellState::Unchanged,
    }
}

/// The `Changed`×`Changed` (and, defensively, `Added`×`Added`) rule: keep
/// local unless the two sides' content is byte-identical despite both
/// having diverged from `base` independently — C2 §7.2's explicit "byte-
/// identical content on both sides is never a conflict" clause.
fn conflict_or_keep(local_cell: Option<&CellDoc>, peer_cell: Option<&CellDoc>) -> CellOutcome {
    match (local_cell, peer_cell) {
        (Some(l), Some(p)) if same_content(l, p) => CellOutcome::KeepLocal,
        _ => CellOutcome::Conflict,
    }
}

/// C2 §7.2's sixteen-cell decision table, applied once `base` is known to
/// hold this id (so both states are drawn from `{Unchanged, Changed,
/// Deleted}` in the reachable case). The six `Added`-crossed cells are
/// structurally impossible from a single shared `base` — [`decide_cells`]
/// never produces them — and are implemented here only as the spec's own
/// defensive fallback for a caller that somehow constructs them directly
/// (this function's own unit tests do, exercising the branch C2 §7.2 says
/// must never panic or silently drop data).
fn decide_one(
    id: &str,
    local_state: CellState,
    peer_state: CellState,
    local_cell: Option<&CellDoc>,
    peer_cell: Option<&CellDoc>,
) -> (CellOutcome, Option<MergeWarning>) {
    use CellState::*;

    let inconsistency = || Some(MergeWarning::MergeStateInconsistency { cell_id: id.to_string() });

    match (local_state, peer_state) {
        (Unchanged, Unchanged) => (CellOutcome::KeepLocal, None),
        (Unchanged, Changed) => (CellOutcome::TakePeer, None),
        (Unchanged, Deleted) => (CellOutcome::Drop, None),
        // Impossible: `Unchanged` presupposes `base` has this id, `Added`
        // presupposes it doesn't. C2 §7.2's defensive handling: trust the
        // peer's concrete content and adopt it, same as `Changed`.
        (Unchanged, Added) => (CellOutcome::TakePeer, inconsistency()),

        (Changed, Unchanged) => (CellOutcome::KeepLocal, None),
        (Changed, Changed) => (conflict_or_keep(local_cell, peer_cell), None),
        (Changed, Deleted) => (CellOutcome::MarkedDeletion { deleted_by_peer: true }, None),
        // Impossible, same reasoning. C2 §7.2's defensive handling: fall
        // back to the ordinary same-id conflict rule.
        (Changed, Added) => (conflict_or_keep(local_cell, peer_cell), inconsistency()),

        // Impossible (the mirror image of the two cells above — `local`
        // claims `Added` while `peer`'s state presupposes `base` has the
        // id). Not part of C2 §7.2's own worked defensive text (which only
        // spells out the `Unchanged`/`Changed`×`Added` direction), but
        // handled the same way here for symmetry: never panic, never drop,
        // and warn since the labels are contradictory — kept local (the
        // side that actually claims to have added it) wins.
        (Added, Unchanged) => (CellOutcome::KeepLocal, inconsistency()),
        (Added, Changed) => (CellOutcome::KeepLocal, inconsistency()),
        (Added, Added) => (conflict_or_keep(local_cell, peer_cell), None),
        (Added, Deleted) => (CellOutcome::KeepLocal, inconsistency()),

        (Deleted, Unchanged) => (CellOutcome::Drop, None),
        (Deleted, Changed) => (CellOutcome::MarkedDeletion { deleted_by_peer: false }, None),
        // Impossible, same reasoning as the `Added` row. Peer's claimed
        // addition wins (mirrors the `Unchanged`/`Changed`×`Added` cells).
        (Deleted, Added) => (CellOutcome::TakePeer, inconsistency()),
        (Deleted, Deleted) => (CellOutcome::Drop, None),
    }
}

/// Decides every cell id's merge outcome (C2 §7.2), independently, over
/// three parsed documents. Deterministic and side-effect-free: no file I/O,
/// no clock, and no randomness — the same three documents always produce
/// the same result (their ids are already fixed in each `WorkbookDoc`, this
/// function mints none).
///
/// An id absent from `base` and present on only one side is C2 §7.2's
/// unary "union unconditionally" rule, not a table lookup — it is the
/// ordinary, common case of a brand-new cell and never raises
/// [`MergeWarning::MergeStateInconsistency`]. An id absent from `base` and
/// present, with the same content, on **both** sides is the true (and only
/// reachable) `Added`×`Added` cell — an id collision (C2 §2.2's ~2⁻³² case)
/// — resolved by the same identical-content-is-never-a-conflict rule as
/// `Changed`×`Changed`.
pub fn decide_cells(
    local: &WorkbookDoc,
    peer: &WorkbookDoc,
    base: &WorkbookDoc,
) -> (BTreeMap<String, CellOutcome>, Vec<MergeWarning>) {
    let base_by_id = index_by_id(base);
    let local_by_id = index_by_id(local);
    let peer_by_id = index_by_id(peer);

    let mut ids: BTreeSet<&str> = BTreeSet::new();
    ids.extend(base_by_id.keys().copied());
    ids.extend(local_by_id.keys().copied());
    ids.extend(peer_by_id.keys().copied());

    let mut outcomes = BTreeMap::new();
    let mut warnings = Vec::new();

    for id in ids {
        let base_cell = base_by_id.get(id).copied();
        let local_cell = local_by_id.get(id).copied();
        let peer_cell = peer_by_id.get(id).copied();

        let outcome = match (base_cell, local_cell, peer_cell) {
            // Never reached (`id` came from one of the three maps above),
            // but never a panic: nothing exists anywhere, so there is
            // nothing to keep.
            (None, None, None) => CellOutcome::Drop,

            // `base` lacks it; only one side has it — C2 §7.2's unary
            // "union unconditionally" rule, the everyday brand-new-cell
            // case. No `MergeStateInconsistency`: nothing is contradictory
            // here, `local`/`peer` simply have no opinion about an id they
            // never contained.
            (None, Some(_), None) => CellOutcome::KeepLocal,
            (None, None, Some(_)) => CellOutcome::TakePeer,

            // `base` lacks it; both sides independently created a cell
            // carrying the same random id (C2 §2.2's collision case).
            (None, Some(l), Some(p)) => conflict_or_keep(Some(l), Some(p)),

            // `base` has it: the real 3×3 `Unchanged`/`Changed`/`Deleted`
            // grid (plus its structurally-impossible corners, handled
            // defensively by `decide_one` but never reached from here).
            (Some(b), l, p) => {
                let local_state = side_state(Some(b), l);
                let peer_state = side_state(Some(b), p);
                let (outcome, warning) = decide_one(id, local_state, peer_state, l, p);
                if let Some(w) = warning {
                    warnings.push(w);
                }
                outcome
            }
        };

        outcomes.insert(id.to_string(), outcome);
    }

    (outcomes, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workbook::v3::CellKindToken;

    fn cell(id: &str, body: &str) -> CellDoc {
        CellDoc {
            id: id.to_string(),
            kind_token: CellKindToken::Math,
            prose_before: None,
            prose_after: None,
            raw_fence_body: body.to_string(),
            table: None,
        }
    }

    fn doc(cells: Vec<CellDoc>) -> WorkbookDoc {
        WorkbookDoc {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Test".to_string(),
            constants_raw: HashMap::new(),
            units_pref: Default::default(),
            version: 3,
            cells,
            trailing_prose: None,
            const_lines: Vec::new(),
            defs: Vec::new(),
            constants: HashMap::new(),
        }
    }

    fn empty_doc() -> WorkbookDoc {
        doc(Vec::new())
    }

    #[test]
    fn decide_cells_an_empty_base_every_cell_on_both_sides_is_added() {
        // Arrange
        let base = empty_doc();
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = doc(vec![cell("bbbbbbbb", "y = 2")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::KeepLocal));
        assert_eq!(outcomes.get("bbbbbbbb"), Some(&CellOutcome::TakePeer));
    }

    #[test]
    fn decide_cells_unchanged_on_both_sides_keeplocal() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 1")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::KeepLocal));
    }

    #[test]
    fn decide_cells_unchanged_locally_changed_by_the_peer_takepeer() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 2")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::TakePeer));
    }

    #[test]
    fn decide_cells_unchanged_locally_deleted_by_the_peer_drop() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = empty_doc();

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::Drop));
    }

    #[test]
    fn decide_cells_deleted_locally_unchanged_by_the_peer_drop() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = empty_doc();
        let peer = doc(vec![cell("aaaaaaaa", "x = 1")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::Drop));
    }

    #[test]
    fn decide_cells_both_sides_made_the_identical_edit_keeplocal_no_conflict() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 2")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 2")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::KeepLocal));
    }

    #[test]
    fn decide_cells_a_prose_only_edit_on_one_side_that_cell_is_changed() {
        // Arrange — same fence body, different `prose_before`: still a
        // `Changed` cell (C2 §7.3's "prose travels with its cell").
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let mut edited = cell("aaaaaaaa", "x = 1");
        edited.prose_before = Some("New heading\n".to_string());
        let local = doc(vec![edited]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 1")]);

        // Act
        let (outcomes, _warnings) = decide_cells(&local, &peer, &base);

        // Assert — local changed (prose), peer unchanged: local's version
        // already wins by staying put.
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::KeepLocal));
    }

    #[test]
    fn decide_cells_run_twice_identical_outcomes() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 2"), cell("cccccccc", "z = 9")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 3")]);

        // Act
        let (first, first_warnings) = decide_cells(&local, &peer, &base);
        let (second, second_warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert_eq!(first, second);
        assert_eq!(first_warnings, second_warnings);
    }

    #[test]
    fn decide_cells_changed_both_sides_to_different_content_is_a_conflict() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 2")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 3")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::Conflict));
    }

    #[test]
    fn decide_cells_local_edited_peer_deleted_marked_deletion_local_survives() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = doc(vec![cell("aaaaaaaa", "x = 2")]);
        let peer = empty_doc();

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::MarkedDeletion { deleted_by_peer: true }));
    }

    #[test]
    fn decide_cells_local_deleted_peer_edited_marked_deletion_peer_survives() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = empty_doc();
        let peer = doc(vec![cell("aaaaaaaa", "x = 2")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::MarkedDeletion { deleted_by_peer: false }));
    }

    #[test]
    fn decide_cells_both_sides_deleted_drop_no_warning() {
        // Arrange
        let base = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let local = empty_doc();
        let peer = empty_doc();

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::Drop));
    }

    #[test]
    fn decide_cells_added_on_both_sides_with_the_same_content_no_conflict() {
        // Arrange — C2 §2.2's collision case: base never had this id.
        let base = empty_doc();
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 1")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::KeepLocal));
    }

    #[test]
    fn decide_cells_added_on_both_sides_with_different_content_is_a_conflict() {
        // Arrange
        let base = empty_doc();
        let local = doc(vec![cell("aaaaaaaa", "x = 1")]);
        let peer = doc(vec![cell("aaaaaaaa", "x = 2")]);

        // Act
        let (outcomes, warnings) = decide_cells(&local, &peer, &base);

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(outcomes.get("aaaaaaaa"), Some(&CellOutcome::Conflict));
    }

    /// C2 §7.2's "impossible" `Unchanged`×`Added` cell cannot be produced by
    /// [`decide_cells`] itself (a single `base` document can't simultaneously
    /// contain and lack the same id) — exercised directly against the
    /// private decision-table function instead, which is where the
    /// contract's defensive handling actually lives.
    #[test]
    fn decide_one_an_added_by_peer_times_unchanged_by_local_pair_takes_peer_and_warns() {
        // Arrange
        let local_cell = cell("aaaaaaaa", "x = 1");
        let peer_cell = cell("aaaaaaaa", "y = 2");

        // Act
        let (outcome, warning) =
            decide_one("aaaaaaaa", CellState::Unchanged, CellState::Added, Some(&local_cell), Some(&peer_cell));

        // Assert
        assert_eq!(outcome, CellOutcome::TakePeer);
        assert_eq!(warning, Some(MergeWarning::MergeStateInconsistency { cell_id: "aaaaaaaa".to_string() }));
    }
}
