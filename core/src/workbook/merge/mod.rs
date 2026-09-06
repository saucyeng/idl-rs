//! Workbook merge (C2 §7): a pure function over three parsed `.idl1wb`
//! documents (`local`, `peer`, `base`) that decides *content* — front-matter
//! fields (§7.1) and, per cell id, which side's content the merged document
//! should carry (§7.2). Ordering the merged document, minting fresh
//! conflict-cell ids and rendering the conflict-marker prose text is Task
//! 5's job ([`crate::workbook::merge`] itself never touches a file, never
//! reorders a cell, and never mints a cell id).
//!
//! `base` is the last document both sides successfully synced (design
//! §7.1); the caller (Task 5, then `store::sync::apply`) is responsible for
//! reading it from `<data>/workbooks/.sync-base/<id>.idl1wb` and for
//! substituting the empty document when no cache exists yet (first-ever
//! sync) — this module takes `base` as an ordinary parameter and is correct
//! for an empty `base` (every cell then classifies `Added` on both sides,
//! C2 §7).

pub mod cells;
pub mod front_matter;
pub mod order;

pub use cells::{decide_cells, CellState};
pub use front_matter::merge_front_matter;

use std::collections::HashMap;
use std::fmt;

use crate::workbook::v3::{merge_constants, ConstLine, FrontMatter, MathCellDef, MathCellLine};
use crate::workbook::v3::{parse_math_cell_body, CellKindToken, WorkbookDoc};

/// What the merge decided for one cell id (C2 §7.2). Content only — the
/// caller (Task 5) is responsible for turning this into an actual edit of
/// the merged document (dropping a cell, minting a conflict cell's fresh
/// id, writing a marker line into `prose_before`, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellOutcome {
    /// Keep local's cell as-is.
    KeepLocal,
    /// Adopt the peer's content for this id.
    TakePeer,
    /// The cell is gone from the merged document (both sides agree it's
    /// deleted, or one side deleted it and the other never touched it).
    Drop,
    /// Local's cell survives; the peer's is appended below as a conflict
    /// copy with a fresh id (C2 §2.2 collision-avoidance path) — both sides
    /// edited this id (or independently minted the same random id) to
    /// different content.
    Conflict,
    /// One side edited this cell, the other deleted it: the edited side's
    /// content survives, marked. `deleted_by_peer: true` means the surviving
    /// content is **local's** (peer deleted it, C2 §7.2's Changed×Deleted
    /// cell); `false` means the surviving content is **peer's** (local
    /// deleted it, the Deleted×Changed cell).
    MarkedDeletion { deleted_by_peer: bool },
}

/// A merge that produced something a human should look at (C2 §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeWarning {
    /// C2 §7.2's "impossible" grid cells reached — a cell id's state on one
    /// side (`Unchanged`/`Changed`/`Deleted`, which all presuppose `base`
    /// contains the id) contradicts the other side's state (`Added`, which
    /// presupposes `base` lacks it). Never produced by this module's own
    /// derivation (a single `base` document can't disagree with itself
    /// about whether it contains an id) — kept as defensive handling for a
    /// caller that constructs states some other way, per C2 §7.2's "filled
    /// with the contradiction and its defensive handling, not left blank."
    /// Names the cell id.
    MergeStateInconsistency { cell_id: String },
    /// A front-matter scalar (`name` or `units`) changed on both sides
    /// since `base`, to different values — the peer's was discarded
    /// (C2 §7.1). `peer_value` is the discarded value's display text.
    FrontMatterConflict { key: String, peer_value: String },
    /// A `constants` entry changed on both sides since `base`, to different
    /// values — the peer's was discarded (C2 §7.1). `peer_value` is the
    /// discarded value's display text, or `"<deleted>"` when the peer's
    /// side of the conflict was a deletion rather than a differing value.
    ConstantConflict { name: String, peer_value: String },
}

/// Front matter's `id` or `version` differs between `local` and `peer` (C2
/// §7.1) — either not the same workbook, or not the same schema version, so
/// sync must refuse to merge rather than silently overwrite one side with
/// the other's document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeError {
    /// `local.id != peer.id` — not the same workbook.
    IdMismatch {
        /// `local`'s front-matter `id`.
        local_id: String,
        /// `peer`'s front-matter `id`.
        peer_id: String,
    },
    /// `local.version != peer.version`. Unreachable in practice today — the
    /// landed parser is fatal on any front-matter `version` other than 3, so
    /// two documents that both parsed successfully already agree — but kept
    /// as C2 §7.1's named defensive check against a future relaxation of
    /// that parser rule letting a real mismatch through unmerged and
    /// unreported.
    VersionMismatch {
        /// `local`'s front-matter `version`.
        local_version: u32,
        /// `peer`'s front-matter `version`.
        peer_version: u32,
    },
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergeError::IdMismatch { local_id, peer_id } => write!(
                f,
                "workbook front-matter ids differ: local is '{local_id}', peer is '{peer_id}' — not the same workbook"
            ),
            MergeError::VersionMismatch { local_version, peer_version } => write!(
                f,
                "workbook front-matter versions differ: local is {local_version}, peer is {peer_version}"
            ),
        }
    }
}

impl std::error::Error for MergeError {}

/// The result of merging one workbook (C2 §7's whole contract, folded into
/// one pure function by [`merge`]).
///
/// `doc`'s `const_lines`/`defs`/`constants` fields are recomputed from
/// `doc.cells`' merged content by the same logic
/// [`crate::workbook::v3::parse_workbook`] runs (`parse_math_cell_body`
/// per `math` cell, then [`merge_constants`]) — kept internally consistent
/// with a document that had gone through an actual save-then-reparse
/// round trip, even though `merge` itself never renders to text. A
/// `DuplicateDefinition`/`ReservedName` raised by that recomputation is
/// intentionally discarded here (never surfaced as a [`MergeWarning`]):
/// any such duplication already existed in at least one of `local`/`peer`
/// and would have been reported by that document's own original parse.
#[derive(Debug, Clone)]
pub struct MergedDoc {
    pub doc: WorkbookDoc,
    /// Conflict cells created (C3 §3.9's `SyncResult.conflicts` counts
    /// these) — `Conflict`-outcome cells only; a `MarkedDeletion` never
    /// invents a cell, so is never counted here.
    pub conflicts: u32,
    pub warnings: Vec<MergeWarning>,
}

/// Reads a [`WorkbookDoc`]'s identity/scalar fields out as a standalone
/// [`FrontMatter`] value, `merge_front_matter`'s own input shape.
fn to_front_matter(doc: &WorkbookDoc) -> FrontMatter {
    FrontMatter {
        id: doc.id.clone(),
        name: doc.name.clone(),
        constants: doc.constants_raw.clone(),
        units: doc.units_pref,
        version: doc.version,
    }
}

/// C2 §7.1's front-matter conflict-marker text: `<!-- conflict from
/// <peer>: <key> was "<peer value>" -->`, one line per
/// [`MergeWarning::FrontMatterConflict`]/[`MergeWarning::ConstantConflict`]
/// in `warnings` (in the order `merge_front_matter` produced them).
fn front_matter_conflict_markers(peer_name: &str, warnings: &[MergeWarning]) -> Vec<String> {
    warnings
        .iter()
        .filter_map(|w| match w {
            MergeWarning::FrontMatterConflict { key, peer_value } => {
                Some(format!("<!-- conflict from {peer_name}: {key} was \"{peer_value}\" -->"))
            }
            MergeWarning::ConstantConflict { name, peer_value } => {
                Some(format!("<!-- conflict from {peer_name}: {name} was \"{peer_value}\" -->"))
            }
            _ => None,
        })
        .collect()
}

/// Recomputes [`WorkbookDoc::const_lines`]/`defs`/`constants` for `cells`
/// and `constants_raw` — see [`MergedDoc`]'s doc comment for why. Mirrors
/// [`crate::workbook::v3::parse_workbook`]'s own per-cell loop, minus that
/// function's `WorkbookError` collection (deliberately discarded here).
fn recompute_derived_fields(
    cells: &[crate::workbook::v3::CellDoc],
    constants_raw: &HashMap<String, crate::workbook::v3::ConstantRaw>,
) -> (Vec<ConstLine>, Vec<MathCellDef>, HashMap<String, f64>) {
    let mut const_lines = Vec::new();
    let mut defs = Vec::new();

    for cell in cells {
        if cell.kind_token != CellKindToken::Math {
            continue;
        }
        let (lines, _cell_errors) = parse_math_cell_body(&cell.id, &cell.raw_fence_body);
        let mut order = 0usize;
        for line in lines {
            match line {
                MathCellLine::Def { name, expr_text, label } => {
                    defs.push(MathCellDef { cell_id: cell.id.clone(), name, expr_text, label, order });
                    order += 1;
                }
                MathCellLine::Const { name, value, unit_display } => {
                    const_lines.push(ConstLine { cell_id: cell.id.clone(), name, value, unit_display });
                }
                MathCellLine::Blank | MathCellLine::Comment => {}
            }
        }
    }

    let (constants, _constant_errors) = merge_constants(constants_raw, &const_lines);
    (const_lines, defs, constants)
}

/// C2 §7's whole merge contract, in one pure function: front matter (§7.1)
/// merged independently of cells, cell content decided per id (§7.2) and
/// ordered/conflict-rendered into an actual document (§7.3), or — for a
/// workbook that has never carried a fenced cell on any of the three sides
/// — §7.3's closing paragraph's plain three-way text merge instead.
///
/// `base` is the empty document (front matter only, zero cells,
/// `trailing_prose: None`) when no `.sync-base` cache exists yet
/// (first-ever sync) — see [`crate::store::sync::base_cache::read_base`].
/// `peer_name` is the human-readable label this merge's conflict markers
/// name (C2 §7.1/§7.2).
///
/// `Err` only when `local.id != peer.id` — not the same workbook, refused
/// rather than silently merged (nothing is decided or rendered in that
/// case).
pub fn merge(local: &WorkbookDoc, peer: &WorkbookDoc, base: &WorkbookDoc, peer_name: &str) -> Result<MergedDoc, MergeError> {
    let local_fm = to_front_matter(local);
    let peer_fm = to_front_matter(peer);
    let base_fm = to_front_matter(base);

    let (merged_fm, mut warnings) = merge_front_matter(&local_fm, &peer_fm, &base_fm, peer_name)?;
    let fm_marker_lines = front_matter_conflict_markers(peer_name, &warnings);

    let is_pure_prose = local.cells.is_empty() && peer.cells.is_empty() && base.cells.is_empty();

    let (mut cells, mut trailing_prose, conflicts) = if is_pure_prose {
        let (merged_text, had_conflict) = order::merge_pure_prose(
            local.trailing_prose.as_deref(),
            peer.trailing_prose.as_deref(),
            base.trailing_prose.as_deref(),
            peer_name,
        );
        (Vec::new(), merged_text, u32::from(had_conflict))
    } else {
        let (outcomes, cell_warnings) = decide_cells(local, peer, base);
        warnings.extend(cell_warnings);
        let (cells, conflicts) = order::build_merged_cells(local, peer, base, &outcomes, peer_name);
        (cells, None, conflicts)
    };

    if !fm_marker_lines.is_empty() {
        if let Some(first) = cells.first_mut() {
            first.prose_before = order::prepend_marker_lines(first.prose_before.as_deref(), &fm_marker_lines);
        } else {
            trailing_prose = order::prepend_marker_lines(trailing_prose.as_deref(), &fm_marker_lines);
        }
    }

    let (const_lines, defs, constants) = recompute_derived_fields(&cells, &merged_fm.constants);

    let doc = WorkbookDoc {
        id: merged_fm.id,
        name: merged_fm.name,
        constants_raw: merged_fm.constants,
        units_pref: merged_fm.units,
        version: merged_fm.version,
        cells,
        trailing_prose,
        const_lines,
        defs,
        constants,
    };

    Ok(MergedDoc { doc, conflicts, warnings })
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use crate::workbook::v3::parse_workbook;

    const ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn wb(markdown: &str) -> WorkbookDoc {
        parse_workbook(markdown).unwrap().0
    }

    fn front(extra: &str) -> String {
        format!("---\nid: {ID}\nname: Test\n{extra}---\n")
    }

    fn empty_doc() -> WorkbookDoc {
        wb(&front(""))
    }

    #[test]
    fn merge_two_sides_editing_different_cells_both_edits_present_zero_conflicts() {
        // Arrange
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n", front("")));
        let local = wb(&format!("{}```math id=aaaaaaaa\nx = 2\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n", front("")));
        let peer = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 2\n```\n", front("")));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.conflicts, 0);
        assert!(merged.warnings.is_empty());
        assert_eq!(merged.doc.cells.len(), 2);
        assert!(merged.doc.cells[0].raw_fence_body.contains("x = 2"));
        assert!(merged.doc.cells[1].raw_fence_body.contains("y = 2"));
    }

    #[test]
    fn merge_two_sides_editing_the_same_cell_differently_one_conflict_cell_below_locals_fresh_id_marker_first_line() {
        // Arrange
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = wb(&format!("{}```math id=aaaaaaaa\nx = 2\n```\n", front("")));
        let peer = wb(&format!("{}```math id=aaaaaaaa\nx = 3\n```\n", front("")));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.conflicts, 1);
        assert_eq!(merged.doc.cells.len(), 2);
        assert_eq!(merged.doc.cells[0].id, "aaaaaaaa");
        assert!(merged.doc.cells[0].raw_fence_body.contains("x = 2"));
        let conflict_cell = &merged.doc.cells[1];
        assert_ne!(conflict_cell.id, "aaaaaaaa");
        assert_eq!(conflict_cell.id.len(), 8);
        assert!(conflict_cell.raw_fence_body.contains("x = 3"));
        let prose = conflict_cell.prose_before.as_deref().unwrap();
        assert_eq!(prose.lines().next(), Some("<!-- conflict from peer-laptop -->"));
    }

    #[test]
    fn merge_the_same_cell_edited_identically_on_both_sides_zero_conflicts() {
        // Arrange
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = wb(&format!("{}```math id=aaaaaaaa\nx = 2\n```\n", front("")));
        let peer = wb(&format!("{}```math id=aaaaaaaa\nx = 2\n```\n", front("")));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.conflicts, 0);
        assert_eq!(merged.doc.cells.len(), 1);
    }

    #[test]
    fn merge_a_cell_added_on_each_side_at_the_same_anchor_locals_first() {
        // Arrange — both sides add a new cell right after `aaaaaaaa`, base
        // only has `aaaaaaaa`.
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = wb(&format!(
            "{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=cccccccc\nz = 1\n```\n",
            front("")
        ));
        let peer = wb(&format!(
            "{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=dddddddd\nw = 1\n```\n",
            front("")
        ));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.conflicts, 0);
        let ids: Vec<&str> = merged.doc.cells.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["aaaaaaaa", "cccccccc", "dddddddd"]);
    }

    #[test]
    fn merge_a_cell_deleted_on_one_side_untouched_on_the_other_it_is_gone() {
        // Arrange
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = empty_doc();
        let peer = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert!(merged.doc.cells.is_empty());
        assert_eq!(merged.conflicts, 0);
    }

    #[test]
    fn merge_a_cell_deleted_by_the_peer_edited_locally_locals_survives_with_deleted_upstream_marker() {
        // Arrange
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = wb(&format!("{}```math id=aaaaaaaa\nx = 2\n```\n", front("")));
        let peer = empty_doc();

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.doc.cells.len(), 1);
        assert!(merged.doc.cells[0].raw_fence_body.contains("x = 2"));
        let prose = merged.doc.cells[0].prose_before.as_deref().unwrap();
        assert_eq!(prose.lines().next(), Some("<!-- conflict from peer-laptop: deleted upstream -->"));
        assert_eq!(merged.conflicts, 0);
    }

    #[test]
    fn merge_base_cells_reordered_locally_c2_7_3_order_deterministically() {
        // Arrange — local reorders `aaaaaaaa`/`bbbbbbbb`; base's order wins.
        let base = wb(&format!(
            "{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n",
            front("")
        ));
        let local = wb(&format!(
            "{}```math id=bbbbbbbb\ny = 1\n```\n\n```math id=aaaaaaaa\nx = 1\n```\n",
            front("")
        ));
        let peer = wb(&format!(
            "{}```math id=aaaaaaaa\nx = 1\n```\n\n```math id=bbbbbbbb\ny = 1\n```\n",
            front("")
        ));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        let ids: Vec<&str> = merged.doc.cells.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["aaaaaaaa", "bbbbbbbb"]);
    }

    #[test]
    fn merge_a_pure_prose_document_changed_on_both_sides_the_three_way_text_result_not_a_cell_table() {
        // Arrange
        let base = wb(&front(""));
        let mut local = base.clone();
        local.trailing_prose = Some("Local notes.\n".to_string());
        let mut peer = base.clone();
        peer.trailing_prose = Some("Peer notes.\n".to_string());

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert!(merged.doc.cells.is_empty());
        assert_eq!(merged.conflicts, 1);
        let text = merged.doc.trailing_prose.as_deref().unwrap();
        assert!(text.starts_with("Local notes."));
        assert!(text.contains("<!-- conflict from peer-laptop -->"));
        assert!(text.contains("Peer notes."));
    }

    #[test]
    fn merge_differing_workbook_ids_mergeerror() {
        // Arrange
        let base = empty_doc();
        let local = empty_doc();
        let mut peer = empty_doc();
        peer.id = "00000000-0000-4000-8000-000000000000".to_string();

        // Act
        let err = merge(&local, &peer, &base, "peer-laptop").unwrap_err();

        // Assert
        assert_eq!(
            err,
            MergeError::IdMismatch {
                local_id: ID.to_string(),
                peer_id: "00000000-0000-4000-8000-000000000000".to_string(),
            }
        );
    }

    #[test]
    fn merge_a_const_line_changed_via_conflict_outcome_constants_map_reflects_the_merged_value() {
        // Arrange — same cell edited on both sides to different `const k`
        // values: a Conflict outcome keeps local's cell content (`k = 2`)
        // and appends peer's as a fresh conflict cell (`k = 3`) below it.
        // `recompute_derived_fields` must derive `constants["k"]` from the
        // *merged* cell set, not from either side's stale pre-merge value.
        let base = wb(&format!("{}```math id=aaaaaaaa\nconst k = 1\n```\n", front("")));
        let local = wb(&format!("{}```math id=aaaaaaaa\nconst k = 2\n```\n", front("")));
        let peer = wb(&format!("{}```math id=aaaaaaaa\nconst k = 3\n```\n", front("")));

        // Act
        let merged = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.conflicts, 1);
        // The surviving (non-conflict-copy) cell is local's — its `k = 2`
        // is the one `merge_constants` should have folded into the map,
        // since a duplicate `k` from the conflict-copy cell is dropped by
        // `merge_constants`'s own duplicate handling, not by this test's
        // logic.
        assert_eq!(merged.doc.constants.get("k"), Some(&2.0));
        assert_eq!(merged.doc.const_lines.len(), 2);
        assert!(merged.doc.const_lines.iter().any(|l| l.name == "k" && l.value == 2.0));
        assert!(merged.doc.const_lines.iter().any(|l| l.name == "k" && l.value == 3.0));
    }

    #[test]
    fn merge_merging_twice_identical_output_idempotent() {
        // Arrange — no conflicts, so no randomly-minted id makes two runs
        // diverge.
        let base = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));
        let local = wb(&format!(
            "{}```math id=aaaaaaaa\nx = 2\n```\n\n```math id=cccccccc\nz = 1\n```\n",
            front("")
        ));
        let peer = wb(&format!("{}```math id=aaaaaaaa\nx = 1\n```\n", front("")));

        // Act
        let first = merge(&local, &peer, &base, "peer-laptop").unwrap();
        let second = merge(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        let first_ids: Vec<&str> = first.doc.cells.iter().map(|c| c.id.as_str()).collect();
        let second_ids: Vec<&str> = second.doc.cells.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(first_ids, second_ids);
        assert_eq!(first.conflicts, second.conflicts);
        assert_eq!(first.warnings, second.warnings);
    }
}
