//! Front-matter merge (C2 §7.1): each top-level key merged independently of
//! the cell table, three-way against `base`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::workbook::v3::front_matter::{ConstantRaw, FrontMatter, UnitsPref};

use super::{MergeError, MergeWarning};

/// C2 §7.1's shared three-way scalar rule, generic over any field that's
/// `Clone + PartialEq`: unchanged-on-one-side takes the other's value;
/// changed-on-both keeps `local`'s and returns `peer`'s (discarded) value —
/// unless the two sides changed to the *same* value, which is no conflict
/// at all. Returns `(merged_value, discarded_peer_value)`.
fn merge_scalar<T: Clone + PartialEq>(local: &T, peer: &T, base: &T) -> (T, Option<T>) {
    let local_changed = local != base;
    let peer_changed = peer != base;

    match (local_changed, peer_changed) {
        (false, true) => (peer.clone(), None),
        (true, true) if local != peer => (local.clone(), Some(peer.clone())),
        _ => (local.clone(), None),
    }
}

/// Display text for a [`UnitsPref`] value, matching the lowercase form
/// [`crate::workbook::v3::front_matter::render_front_matter`] would emit for
/// it (C2 §1's `units` key).
fn units_display(units: UnitsPref) -> &'static str {
    match units {
        UnitsPref::Si => "si",
        UnitsPref::Imperial => "imperial",
    }
}

/// Display text for a front-matter constant's raw value (C2 §3.1), for use
/// in a [`MergeWarning::ConstantConflict`]'s `peer_value`.
fn constant_display(raw: &ConstantRaw) -> String {
    match raw {
        ConstantRaw::Number(v) => v.to_string(),
        ConstantRaw::WithUnit { value, unit_display } => format!("{value} {unit_display}"),
    }
}

/// Merges `local` and `peer` front matter against their common `base` (C2
/// §7.1, amended by R151 item 10 / plan §3.4 for the retired-name migration,
/// `runs/2026-09-08/scipy-alignment-plan.md`). `id` is immutable identity: a
/// mismatched `id` is refused outright (`Err`) rather than merged — not the
/// same workbook. `version` is **not** refused on a mismatch: a `version: 3`
/// side that has not yet been re-saved through the migration meeting a
/// `version: 4` peer that has is not a different-file situation, it's what
/// the migration in flight looks like — the merged document's `version` is
/// the higher of the two (the migrating side wins, never the other way:
/// nothing here ever produces a merged `version: 3` from a `version: 4`
/// input). [`MergeError::VersionMismatch`] is kept as a typed error for a
/// version pairing outside `{3, 4}`, which the parser's own version gate
/// (C2 §3.5) should already prevent from reaching here — defensive, not
/// reachable in practice today.
///
/// `peer_name` (a human-readable label for the peer device/workbook,
/// distinct from the `name` *field* being merged) is threaded through for
/// Task 5's conflict-marker text ("`<!-- conflict from <peer>: … -->`", C2
/// §7.1) — this function's own returned [`MergeWarning`]s carry only the
/// discarded value, not the marker's rendered text, since rendering is
/// Task 5's job.
pub fn merge_front_matter(
    local: &FrontMatter,
    peer: &FrontMatter,
    base: &FrontMatter,
    peer_name: &str,
) -> Result<(FrontMatter, Vec<MergeWarning>), MergeError> {
    let _ = peer_name; // reserved for Task 5's marker-text rendering; unused here.

    if local.id != peer.id {
        return Err(MergeError::IdMismatch { local_id: local.id.clone(), peer_id: peer.id.clone() });
    }
    // C2 §3.5's version gate only ever lets `3` or `4` reach a parsed
    // `FrontMatter` — this is the defensive fallback for anything else, see
    // this function's doc comment.
    let known_version = |v: u32| v == 3 || v == 4;
    if !known_version(local.version) || !known_version(peer.version) {
        return Err(MergeError::VersionMismatch { local_version: local.version, peer_version: peer.version });
    }
    let version = local.version.max(peer.version);

    let mut warnings = Vec::new();

    let (name, discarded_name) = merge_scalar(&local.name, &peer.name, &base.name);
    if let Some(discarded) = discarded_name {
        warnings.push(MergeWarning::FrontMatterConflict { key: "name".to_string(), peer_value: discarded });
    }

    let (units, discarded_units) = merge_scalar(&local.units, &peer.units, &base.units);
    if let Some(discarded) = discarded_units {
        warnings.push(MergeWarning::FrontMatterConflict {
            key: "units".to_string(),
            peer_value: units_display(discarded).to_string(),
        });
    }

    let mut keys: BTreeSet<&String> = BTreeSet::new();
    keys.extend(local.constants.keys());
    keys.extend(peer.constants.keys());

    let mut constants = HashMap::new();
    for key in keys {
        let base_value = base.constants.get(key).cloned();
        let local_value = local.constants.get(key).cloned();
        let peer_value = peer.constants.get(key).cloned();

        let (merged_value, discarded) = merge_scalar(&local_value, &peer_value, &base_value);
        if let Some(value) = merged_value {
            constants.insert(key.clone(), value);
        }
        if let Some(discarded_value) = discarded {
            let peer_display = match discarded_value {
                Some(raw) => constant_display(&raw),
                None => "<deleted>".to_string(),
            };
            warnings.push(MergeWarning::ConstantConflict { name: key.clone(), peer_value: peer_display });
        }
    }

    let unknown = merge_unknown(&local.unknown, &peer.unknown, &base.unknown, &mut warnings);

    Ok((FrontMatter { id: local.id.clone(), name, constants, units, version, unknown }, warnings))
}

/// Merges every front-matter key this contract does not itself define (C2
/// §1's `unknown` catch-all), one entry at a time, on the same
/// unchanged-on-one-side/changed-on-both-keeps-local rule as `constants`
/// above — a coarse, whole-value merge per key rather than a per-key rule
/// tailored to what that key means, since this function has no way to know.
/// A key with its own merge rule (C2 §3.7.2's `graph`, merged per *entry
/// within* `nodes`/`cells` and never a conflict) gets that finer rule from
/// its own future task; this is the generic fallback that guarantees no
/// unrecognised key is ever silently dropped by a sync merge — the same
/// durability guarantee ruling R135 requires of parse/render, extended here
/// to the merge step those bytes also pass through.
fn merge_unknown(
    local: &BTreeMap<String, serde_yaml_ng::Value>,
    peer: &BTreeMap<String, serde_yaml_ng::Value>,
    base: &BTreeMap<String, serde_yaml_ng::Value>,
    warnings: &mut Vec<MergeWarning>,
) -> BTreeMap<String, serde_yaml_ng::Value> {
    let mut keys: BTreeSet<&String> = BTreeSet::new();
    keys.extend(local.keys());
    keys.extend(peer.keys());

    let mut merged = BTreeMap::new();
    for key in keys {
        let base_value = base.get(key).cloned();
        let local_value = local.get(key).cloned();
        let peer_value = peer.get(key).cloned();

        let (merged_value, discarded) = merge_scalar(&local_value, &peer_value, &base_value);
        if let Some(value) = merged_value {
            merged.insert(key.clone(), value);
        }
        if let Some(discarded_value) = discarded {
            let peer_display = discarded_value
                .map(|v| serde_yaml_ng::to_string(&v).unwrap_or_default().trim().to_string())
                .unwrap_or_else(|| "<deleted>".to_string());
            warnings.push(MergeWarning::FrontMatterConflict { key: key.clone(), peer_value: peer_display });
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn fm(name: &str, constants: HashMap<String, ConstantRaw>) -> FrontMatter {
        FrontMatter {
            id: ID.to_string(),
            name: name.to_string(),
            constants,
            units: UnitsPref::Si,
            version: 3,
            unknown: BTreeMap::new(),
        }
    }

    #[test]
    fn merge_front_matter_differing_ids_mergeerror_nothing_merged() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning", HashMap::new());
        let mut peer = fm("Fork tuning", HashMap::new());
        peer.id = "00000000-0000-4000-8000-000000000000".to_string();

        // Act
        let err = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap_err();

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
    fn merge_front_matter_differing_versions_mergeerror_nothing_merged() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning", HashMap::new());
        let mut peer = fm("Fork tuning", HashMap::new());
        peer.version = 2;

        // Act
        let err = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap_err();

        // Assert
        assert_eq!(err, MergeError::VersionMismatch { local_version: 3, peer_version: 2 });
    }

    #[test]
    fn merge_front_matter_a_3_vs_4_version_pairing_merges_to_4_never_a_refusal() {
        // Arrange — R151 item 10 / plan §3.4: one side has been re-saved
        // through the retired-name migration (now `version: 4`), the other
        // has not (`version: 3`). Not a different-file situation.
        let base = fm("Fork tuning", HashMap::new());
        let mut local = fm("Fork tuning", HashMap::new());
        local.version = 4;
        let peer = fm("Fork tuning", HashMap::new()); // still version 3

        // Act
        let (merged, _warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert — the migrating side wins; never the other way around.
        assert_eq!(merged.version, 4);
    }

    #[test]
    fn merge_front_matter_name_changed_on_the_peer_only_the_peers_name() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning", HashMap::new());
        let peer = fm("Fork tuning v2", HashMap::new());

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.name, "Fork tuning v2");
        assert!(warnings.is_empty());
    }

    #[test]
    fn merge_front_matter_name_changed_on_both_locals_name_and_one_warning_carrying_the_peers_value() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning — local", HashMap::new());
        let peer = fm("Fork tuning — peer", HashMap::new());

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.name, "Fork tuning — local");
        assert_eq!(
            warnings,
            vec![MergeWarning::FrontMatterConflict {
                key: "name".to_string(),
                peer_value: "Fork tuning — peer".to_string(),
            }]
        );
    }

    #[test]
    fn merge_front_matter_name_changed_on_both_to_the_same_value_no_conflict() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning v2", HashMap::new());
        let peer = fm("Fork tuning v2", HashMap::new());

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.name, "Fork tuning v2");
        assert!(warnings.is_empty());
    }

    #[test]
    fn merge_front_matter_a_constant_added_on_each_side_both_present() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning", HashMap::from([("rider_mass_kg".to_string(), ConstantRaw::Number(82.0))]));
        let peer = fm("Fork tuning", HashMap::from([("sag_target".to_string(), ConstantRaw::Number(0.3))]));

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert!(warnings.is_empty());
        assert_eq!(merged.constants.get("rider_mass_kg"), Some(&ConstantRaw::Number(82.0)));
        assert_eq!(merged.constants.get("sag_target"), Some(&ConstantRaw::Number(0.3)));
    }

    #[test]
    fn merge_front_matter_a_constant_changed_on_both_local_wins_one_warning() {
        // Arrange
        let base = fm("Fork tuning", HashMap::from([("rider_mass_kg".to_string(), ConstantRaw::Number(80.0))]));
        let local = fm("Fork tuning", HashMap::from([("rider_mass_kg".to_string(), ConstantRaw::Number(82.0))]));
        let peer = fm("Fork tuning", HashMap::from([("rider_mass_kg".to_string(), ConstantRaw::Number(84.0))]));

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.constants.get("rider_mass_kg"), Some(&ConstantRaw::Number(82.0)));
        assert_eq!(
            warnings,
            vec![MergeWarning::ConstantConflict { name: "rider_mass_kg".to_string(), peer_value: "84".to_string() }]
        );
    }

    #[test]
    fn merge_front_matter_an_unknown_key_added_on_the_peer_only_survives_merged() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let local = fm("Fork tuning", HashMap::new());
        let mut peer = fm("Fork tuning", HashMap::new());
        peer.unknown.insert("_migrate_charts".to_string(), serde_yaml_ng::to_value(true).unwrap());

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.unknown.get("_migrate_charts"), Some(&serde_yaml_ng::to_value(true).unwrap()));
        assert!(warnings.is_empty());
    }

    #[test]
    fn merge_front_matter_an_unknown_key_changed_on_both_sides_local_wins_one_warning() {
        // Arrange
        let base = fm("Fork tuning", HashMap::new());
        let mut local = fm("Fork tuning", HashMap::new());
        local.unknown.insert("_migrate_charts".to_string(), serde_yaml_ng::to_value("local").unwrap());
        let mut peer = fm("Fork tuning", HashMap::new());
        peer.unknown.insert("_migrate_charts".to_string(), serde_yaml_ng::to_value("peer").unwrap());

        // Act
        let (merged, warnings) = merge_front_matter(&local, &peer, &base, "peer-laptop").unwrap();

        // Assert
        assert_eq!(merged.unknown.get("_migrate_charts"), Some(&serde_yaml_ng::to_value("local").unwrap()));
        assert_eq!(
            warnings,
            vec![MergeWarning::FrontMatterConflict { key: "_migrate_charts".to_string(), peer_value: "peer".to_string() }]
        );
    }
}
