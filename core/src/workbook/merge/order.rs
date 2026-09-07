//! Ordering the merged cell list, conflict-copy id minting and marker-text
//! rendering (C2 §7.2's conflict-copy text, §7.3's ordering rules), and the
//! pure-prose three-way merge (§7.3's closing paragraph) — the part of
//! Task 5's job [`super`]'s doc comment describes as "turning a decision
//! into an actual edit of the merged document."

use std::collections::{HashMap, HashSet};

use crate::workbook::v3::{CellDoc, WorkbookDoc};

use super::CellOutcome;

/// C2 §2.2's collision-avoidance path, reused here for a conflict copy's
/// fresh id: 4 random bytes, lowercase-hex encoded, regenerated on a
/// collision against any id already spoken for. `existing` is mutated to
/// include the minted id, so two conflict copies minted in the same merge
/// never collide with each other either.
fn mint_conflict_id(existing: &mut HashSet<String>) -> String {
    loop {
        let random = uuid::Uuid::new_v4();
        let candidate: String = random.as_bytes()[..4].iter().map(|b| format!("{b:02x}")).collect();
        if existing.insert(candidate.clone()) {
            return candidate;
        }
    }
}

/// The bare conflict-copy marker (C2 §7.2's Changed×Changed / Added×Added
/// rows): `<!-- conflict from <peer> -->`.
fn conflict_marker(peer_name: &str) -> String {
    format!("<!-- conflict from {peer_name} -->")
}

/// C2 §7.2's Changed×Deleted row marker: local's edit survives, peer
/// deleted it.
fn deleted_upstream_marker(peer_name: &str) -> String {
    format!("<!-- conflict from {peer_name}: deleted upstream -->")
}

/// C2 §7.2's Deleted×Changed row marker: peer's edit survives, local
/// deleted it — written from the perspective of the document being
/// produced, so "deleted locally" names *this* document's side.
fn deleted_locally_marker(peer_name: &str) -> String {
    format!("<!-- conflict from {peer_name}: deleted locally -->")
}

/// Prepends `markers` (already-rendered `<!-- … -->` lines, one per line)
/// ahead of `existing` prose, each on its own line — C2 §7.1/§7.2's "first
/// line of prose_before"/"first line of body prose" placement. `existing`
/// unchanged (as `Some`/`None`) when `markers` is empty.
pub(super) fn prepend_marker_lines(existing: Option<&str>, markers: &[String]) -> Option<String> {
    if markers.is_empty() {
        return existing.map(str::to_string);
    }
    let mut out = String::new();
    for marker in markers {
        out.push_str(marker);
        out.push('\n');
    }
    if let Some(e) = existing {
        out.push_str(e);
    }
    Some(out)
}

/// C2 §7.3's closing paragraph: a document with zero fenced cells on all
/// three sides merges as plain three-way text rather than the cell table.
/// Returns `(merged_text, conflict)` — `conflict` is `true` only when both
/// sides changed the text, to different values, in which case the result is
/// local's text, the marker, then peer's text in full. `None` inputs are
/// treated as empty text.
pub(super) fn merge_pure_prose(
    local: Option<&str>,
    peer: Option<&str>,
    base: Option<&str>,
    peer_name: &str,
) -> (Option<String>, bool) {
    let local_s = local.unwrap_or("");
    let peer_s = peer.unwrap_or("");
    let base_s = base.unwrap_or("");

    let local_changed = local_s != base_s;
    let peer_changed = peer_s != base_s;

    match (local_changed, peer_changed) {
        (false, true) => (none_if_empty(peer_s), false),
        (true, true) if local_s != peer_s => {
            let merged = format!("{local_s}\n{}\n{peer_s}", conflict_marker(peer_name));
            (Some(merged), true)
        }
        _ => (none_if_empty(local_s), false),
    }
}

fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Builds the merged document's cell list (C2 §7.3's ordering, §7.2's
/// conflict-copy rendering) from `outcomes` (already decided, per id, by
/// [`super::decide_cells`]). Returns `(cells, conflicts)` — `conflicts`
/// counts conflict copies actually appended (`CellOutcome::Conflict`
/// entries only; `MarkedDeletion` never invents a cell, so is never
/// counted).
pub(super) fn build_merged_cells(
    local: &WorkbookDoc,
    peer: &WorkbookDoc,
    base: &WorkbookDoc,
    outcomes: &std::collections::BTreeMap<String, CellOutcome>,
    peer_name: &str,
) -> (Vec<CellDoc>, u32) {
    let base_ids: HashSet<&str> = base.cells.iter().map(|c| c.id.as_str()).collect();
    let local_index: HashMap<&str, &CellDoc> = local.cells.iter().map(|c| (c.id.as_str(), c)).collect();
    let peer_index: HashMap<&str, &CellDoc> = peer.cells.iter().map(|c| (c.id.as_str(), c)).collect();

    // A base id "survives" when its own outcome isn't `Drop` — only a
    // surviving base id is a valid anchor for a same-side insertion (§7.3).
    let survivors: HashSet<&str> = base_ids
        .iter()
        .copied()
        .filter(|id| !matches!(outcomes.get(*id), Some(CellOutcome::Drop) | None))
        .collect();

    // Ids absent from `base`: local-added (present in `local`, regardless of
    // whether `peer` also independently minted the same id — §7.2's
    // Added×Added collision is anchored and content-owned by `local`) and
    // pure peer-added (present in `peer`, absent from `local`).
    let local_added: HashSet<&str> =
        outcomes.keys().map(String::as_str).filter(|id| !base_ids.contains(id) && local_index.contains_key(id)).collect();
    let peer_added: HashSet<&str> = outcomes
        .keys()
        .map(String::as_str)
        .filter(|id| !base_ids.contains(id) && peer_index.contains_key(id) && !local_index.contains_key(id))
        .collect();

    let local_buckets = collect_added_by_anchor(&local.cells, &survivors, &local_added);
    let peer_buckets = collect_added_by_anchor(&peer.cells, &survivors, &peer_added);

    // C2 §2.2's collision-avoidance path must avoid every id already in
    // play anywhere in this merge, not just the two documents being
    // combined at one anchor point.
    let mut minted: HashSet<String> = local.cells.iter().map(|c| c.id.clone()).collect();
    minted.extend(peer.cells.iter().map(|c| c.id.clone()));
    minted.extend(base.cells.iter().map(|c| c.id.clone()));

    let mut cells = Vec::new();
    let mut conflicts = 0u32;

    // §7.3 rule (2)/(3)'s anchor points, in document order: "before every
    // base cell" (`None`) first, then each surviving base id in `base`'s
    // own order.
    let anchor_sequence: Vec<Option<String>> = std::iter::once(None)
        .chain(base.cells.iter().filter(|c| survivors.contains(c.id.as_str())).map(|c| Some(c.id.clone())))
        .collect();

    for anchor in &anchor_sequence {
        if let Some(base_id) = anchor {
            render_base_cell(
                base_id,
                outcomes.get(base_id).expect("survivor ids always have a decided outcome"),
                &local_index,
                &peer_index,
                peer_name,
                &mut minted,
                &mut cells,
                &mut conflicts,
            );
        }

        // Rule (4): local's insertion(s) at this anchor sort before peer's.
        if let Some(ids) = local_buckets.get(anchor) {
            for id in ids {
                let outcome = outcomes.get(id).expect("local-added ids always have a decided outcome");
                let local_cell = (*local_index.get(id.as_str()).expect("local-added id is in local")).clone();
                cells.push(local_cell);
                if *outcome == CellOutcome::Conflict {
                    let peer_cell = peer_index.get(id.as_str()).expect("Added×Added conflict implies peer has this id");
                    push_conflict_copy(peer_cell, peer_name, &mut minted, &mut cells);
                    conflicts += 1;
                }
            }
        }

        if let Some(ids) = peer_buckets.get(anchor) {
            for id in ids {
                let peer_cell = (*peer_index.get(id.as_str()).expect("peer-added id is in peer")).clone();
                cells.push(peer_cell);
            }
        }
    }

    (cells, conflicts)
}

/// Groups `doc_cells`' added ids by the nearest preceding surviving base id
/// in `doc_cells`' own order (`None` when an added id precedes every
/// surviving base cell in this document) — C2 §7.3 rules (2)/(3). A single
/// forward pass: `last_seen` only ever advances past a surviving base id,
/// so a base id that didn't survive (dropped on both sides) is skipped
/// exactly as if it were never there, and insertion order within a bucket
/// is preserved for free by appending in document order.
fn collect_added_by_anchor(
    doc_cells: &[CellDoc],
    survivors: &HashSet<&str>,
    added_ids: &HashSet<&str>,
) -> HashMap<Option<String>, Vec<String>> {
    let mut buckets: HashMap<Option<String>, Vec<String>> = HashMap::new();
    let mut last_seen: Option<String> = None;
    for cell in doc_cells {
        if survivors.contains(cell.id.as_str()) {
            last_seen = Some(cell.id.clone());
        }
        if added_ids.contains(cell.id.as_str()) {
            buckets.entry(last_seen.clone()).or_default().push(cell.id.clone());
        }
    }
    buckets
}

/// Renders one base-derived cell's content per its [`CellOutcome`] and
/// pushes it (plus, for `Conflict`, the peer's conflict copy immediately
/// after) onto `cells`.
#[allow(clippy::too_many_arguments)]
fn render_base_cell(
    base_id: &str,
    outcome: &CellOutcome,
    local_index: &HashMap<&str, &CellDoc>,
    peer_index: &HashMap<&str, &CellDoc>,
    peer_name: &str,
    minted: &mut HashSet<String>,
    cells: &mut Vec<CellDoc>,
    conflicts: &mut u32,
) {
    match outcome {
        CellOutcome::KeepLocal => {
            let cell = (*local_index.get(base_id).expect("KeepLocal implies local has this id")).clone();
            cells.push(cell);
        }
        CellOutcome::TakePeer => {
            let cell = (*peer_index.get(base_id).expect("TakePeer implies peer has this id")).clone();
            cells.push(cell);
        }
        CellOutcome::Conflict => {
            let local_cell = (*local_index.get(base_id).expect("Conflict implies local has this id")).clone();
            cells.push(local_cell);
            let peer_cell = peer_index.get(base_id).expect("Conflict implies peer has this id");
            push_conflict_copy(peer_cell, peer_name, minted, cells);
            *conflicts += 1;
        }
        CellOutcome::MarkedDeletion { deleted_by_peer: true } => {
            let mut cell = (*local_index.get(base_id).expect("MarkedDeletion(peer) implies local has this id")).clone();
            cell.prose_before = prepend_marker_lines(cell.prose_before.as_deref(), &[deleted_upstream_marker(peer_name)]);
            cells.push(cell);
        }
        CellOutcome::MarkedDeletion { deleted_by_peer: false } => {
            let mut cell = (*peer_index.get(base_id).expect("MarkedDeletion(local) implies peer has this id")).clone();
            cell.prose_before = prepend_marker_lines(cell.prose_before.as_deref(), &[deleted_locally_marker(peer_name)]);
            cells.push(cell);
        }
        CellOutcome::Drop => {
            // Never reached — `base_id` only appears here when it survived
            // (see `survivors` in `build_merged_cells`).
        }
    }
}

/// Appends `peer_cell`'s content as a conflict copy immediately after
/// whatever was just pushed: a fresh id (never `local`'s or `peer`'s,
/// C2 §2.2) and the marker as the first line of `prose_before`.
fn push_conflict_copy(peer_cell: &CellDoc, peer_name: &str, minted: &mut HashSet<String>, cells: &mut Vec<CellDoc>) {
    let mut copy = peer_cell.clone();
    copy.id = mint_conflict_id(minted);
    copy.prose_before = prepend_marker_lines(copy.prose_before.as_deref(), &[conflict_marker(peer_name)]);
    cells.push(copy);
}
