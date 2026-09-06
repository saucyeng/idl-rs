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

pub use cells::{decide_cells, CellState};
pub use front_matter::merge_front_matter;

use std::fmt;

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

/// Front matter's `id` differs between `local` and `peer` (C2 §7.1) — not
/// the same workbook, so sync must refuse to merge rather than silently
/// overwrite one side with the other's unrelated document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeError {
    /// `local`'s front-matter `id`.
    pub local_id: String,
    /// `peer`'s front-matter `id`.
    pub peer_id: String,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "workbook front-matter ids differ: local is '{}', peer is '{}' — not the same workbook",
            self.local_id, self.peer_id
        )
    }
}

impl std::error::Error for MergeError {}
