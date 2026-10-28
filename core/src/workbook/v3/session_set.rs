//! The front-matter `sessions` key (C2 §1, session-sets spec §2.2): the
//! workbook's **session set**, an ordered list of named groups of whole
//! sessions. The set is part of the file, independent of the app's
//! selection, so a table over it is a function of the file and the library
//! only.
//!
//! The key stays in [`super::WorkbookDoc::front_matter_unknown`] verbatim
//! (ruling R135: an older build round-trips it untouched); this module reads
//! it into a typed [`SessionSet`] on demand and renders one back.
//!
//! ```yaml
//! sessions:
//!   - group: Day 1 (19 Sep)
//!     members:
//!       - { id: 388740a19b3c013e0ec66d276c001d04, label: D1 S1 }
//!   - group: Day 2 (24 Oct)
//!     members:
//!       - { id: 1a5d2cdbb0efddfd9a1796d83fe00351, label: D2 S1 }
//! ```
//!
//! A list of bare members (`- { id: …, label: … }`, no `members` key) is one
//! unnamed group.

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

use super::error::{self, WorkbookError};

/// The front-matter key a session set lives under (C2 §1).
pub const SESSIONS_KEY: &str = "sessions";

/// A workbook's session set: groups in order, each holding its members in
/// order. A session id appears at most once across the whole set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionSet {
    /// The groups, in the order the file lists them.
    pub groups: Vec<SetGroup>,
}

/// One named group of a [`SessionSet`] ("Day 1").
#[derive(Debug, Clone, PartialEq)]
pub struct SetGroup {
    /// The group's name, unique within the set; `None` for the one unnamed
    /// group a list of bare members makes.
    pub name: Option<String>,
    /// The group's sessions, in the order the file lists them.
    pub members: Vec<SetMember>,
}

/// One session of a [`SetGroup`].
#[derive(Debug, Clone, PartialEq)]
pub struct SetMember {
    /// The session id (C4 `<data>/sessions/<id>`), 32 hex digits.
    pub id: String,
    /// Display text; `None` means the default (`session.json`'s
    /// `event_session` when non-empty, else the id's first 8 hex digits —
    /// see [`default_member_label`]).
    pub label: Option<String>,
}

/// One member as the set evaluates it: its position in set order and the
/// index of the group it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatMember {
    /// The session id.
    pub id: String,
    /// The authored label, if any.
    pub label: Option<String>,
    /// Index into [`SessionSet::groups`].
    pub group: usize,
}

impl SessionSet {
    /// Every member in set order (group order, then member order), each
    /// carrying its group index.
    pub fn members(&self) -> Vec<FlatMember> {
        self.groups
            .iter()
            .enumerate()
            .flat_map(|(g, group)| {
                group.members.iter().map(move |m| FlatMember { id: m.id.clone(), label: m.label.clone(), group: g })
            })
            .collect()
    }

    /// The index of the group named `name`, if there is one.
    pub fn group_index(&self, name: &str) -> Option<usize> {
        self.groups.iter().position(|g| g.name.as_deref() == Some(name))
    }

    /// This set as the YAML value the `sessions` key holds. Always the
    /// grouped form, `label` omitted when `None`; parses back through
    /// [`parse_session_set_value`] to an equal set.
    pub fn to_yaml(&self) -> serde_yaml_ng::Value {
        use serde_yaml_ng::{Mapping, Value};
        let groups = self
            .groups
            .iter()
            .map(|g| {
                let mut map = Mapping::new();
                if let Some(name) = &g.name {
                    map.insert(Value::from("group"), Value::from(name.as_str()));
                }
                let members = g
                    .members
                    .iter()
                    .map(|m| {
                        let mut mm = Mapping::new();
                        mm.insert(Value::from("id"), Value::from(m.id.as_str()));
                        if let Some(label) = &m.label {
                            mm.insert(Value::from("label"), Value::from(label.as_str()));
                        }
                        Value::Mapping(mm)
                    })
                    .collect();
                map.insert(Value::from("members"), Value::Sequence(members));
                Value::Mapping(map)
            })
            .collect();
        Value::Sequence(groups)
    }
}

/// The default display label of a member with no authored `label`
/// (session-sets spec §2.2): `event_session` when non-empty, else the first
/// 8 characters of the id.
pub fn default_member_label(session_id: &str, event_session: Option<&str>) -> String {
    match event_session {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => session_id.chars().take(8).collect(),
    }
}

/// One `members` entry as YAML writes it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemberYaml {
    id: String,
    #[serde(default)]
    label: Option<String>,
}

/// One top-level list entry: a group, or a bare member of the unnamed group.
#[derive(Deserialize)]
#[serde(untagged)]
enum ItemYaml {
    Group {
        #[serde(default)]
        group: Option<String>,
        members: Vec<MemberYaml>,
    },
    Member(MemberYaml),
}

/// Reads the `sessions` key out of a document's unrecognised front-matter
/// keys. `Ok(None)` when the key is absent.
///
/// # Errors
/// [`super::WorkbookErrorKind::InvalidSessionSet`] for a value of the wrong
/// shape (not a list; an entry that is neither a group nor a member; groups
/// mixed with bare members; a repeated group name);
/// [`super::WorkbookErrorKind::DuplicateSetMember`] for a session id listed
/// twice anywhere in the set.
pub fn parse_session_set(
    front_matter_unknown: &BTreeMap<String, serde_yaml_ng::Value>,
) -> Result<Option<SessionSet>, WorkbookError> {
    match front_matter_unknown.get(SESSIONS_KEY) {
        None => Ok(None),
        Some(value) => parse_session_set_value(value).map(Some),
    }
}

/// [`parse_session_set`] over the key's value itself.
pub fn parse_session_set_value(value: &serde_yaml_ng::Value) -> Result<SessionSet, WorkbookError> {
    let items: Vec<ItemYaml> = serde_yaml_ng::from_value(value.clone()).map_err(error::invalid_session_set)?;

    let group_count = items.iter().filter(|i| matches!(i, ItemYaml::Group { .. })).count();
    if group_count != 0 && group_count != items.len() {
        return Err(error::invalid_session_set("groups and bare members are mixed in one list"));
    }

    let to_member = |m: MemberYaml| SetMember { id: m.id, label: m.label };
    let groups: Vec<SetGroup> = if group_count == 0 {
        let members: Vec<SetMember> = items
            .into_iter()
            .filter_map(|i| match i {
                ItemYaml::Member(m) => Some(to_member(m)),
                ItemYaml::Group { .. } => None,
            })
            .collect();
        if members.is_empty() { Vec::new() } else { vec![SetGroup { name: None, members }] }
    } else {
        items
            .into_iter()
            .filter_map(|i| match i {
                ItemYaml::Group { group, members } => {
                    Some(SetGroup { name: group, members: members.into_iter().map(to_member).collect() })
                }
                ItemYaml::Member(_) => None,
            })
            .collect()
    };

    let mut names = HashSet::new();
    for g in &groups {
        if let Some(name) = &g.name {
            if !names.insert(name.as_str()) {
                return Err(error::invalid_session_set(format!("group '{name}' is listed more than once")));
            }
        }
    }
    if groups.iter().filter(|g| g.name.is_none()).count() > 1 {
        return Err(error::invalid_session_set("more than one group has no name"));
    }

    let mut seen = HashSet::new();
    for m in groups.iter().flat_map(|g| g.members.iter()) {
        if !seen.insert(m.id.as_str()) {
            return Err(error::duplicate_set_member(&m.id));
        }
    }

    Ok(SessionSet { groups })
}

/// Three-way merge of a `sessions` key (C2 §7.1 amendment, session-sets spec
/// §2.2). Never a conflict, as for `graph`:
///
/// - **groups** merge by name: a group added on one side carries, a group
///   removed on one side carries;
/// - **members** within a kept group merge as a three-way set against that
///   group in `base`: added on one side carries, removed on one side
///   carries; a label changed on both sides keeps local's;
/// - **order** is local's, with what only the peer added appended in the
///   peer's order.
///
/// A session that ends up in two groups (moved on one side, kept on the
/// other) stays only in the first, so the result still has every id once.
/// `None` on a side means that side has no `sessions` key.
pub fn merge_session_sets(
    local: Option<&SessionSet>,
    peer: Option<&SessionSet>,
    base: Option<&SessionSet>,
) -> Option<SessionSet> {
    let empty = SessionSet::default();
    let (l, p, b) = (local.unwrap_or(&empty), peer.unwrap_or(&empty), base.unwrap_or(&empty));
    let find = |set: &SessionSet, name: &Option<String>| set.groups.iter().find(|g| &g.name == name).cloned();

    let mut names: Vec<Option<String>> = l.groups.iter().map(|g| g.name.clone()).collect();
    for g in &p.groups {
        if !names.contains(&g.name) {
            names.push(g.name.clone());
        }
    }

    let mut groups = Vec::new();
    for name in names {
        let (lg, pg, bg) = (find(l, &name), find(p, &name), find(b, &name));
        let keep = match (&lg, &pg) {
            (Some(_), Some(_)) => true,
            (Some(_), None) | (None, Some(_)) => bg.is_none(),
            (None, None) => false,
        };
        if !keep {
            continue;
        }
        let no_members = Vec::new();
        let lm = lg.as_ref().map(|g| &g.members).unwrap_or(&no_members);
        let pm = pg.as_ref().map(|g| &g.members).unwrap_or(&no_members);
        let bm = bg.as_ref().map(|g| &g.members).unwrap_or(&no_members);
        let get = |ms: &Vec<SetMember>, id: &str| ms.iter().find(|m| m.id == id).cloned();

        let mut ids: Vec<String> = lm.iter().map(|m| m.id.clone()).collect();
        for m in pm {
            if !ids.contains(&m.id) {
                ids.push(m.id.clone());
            }
        }
        let mut members = Vec::new();
        for id in ids {
            let (a, c, o) = (get(lm, &id), get(pm, &id), get(bm, &id));
            let merged = match (a, c) {
                (Some(a), Some(c)) => {
                    let base_label = o.and_then(|o| o.label);
                    let label = if a.label == base_label && c.label != base_label { c.label } else { a.label };
                    Some(SetMember { id, label })
                }
                (Some(a), None) => if o.is_none() { Some(a) } else { None },
                (None, Some(c)) => if o.is_none() { Some(c) } else { None },
                (None, None) => None,
            };
            members.extend(merged);
        }
        groups.push(SetGroup { name, members });
    }

    let mut seen = HashSet::new();
    for g in &mut groups {
        g.members.retain(|m| seen.insert(m.id.clone()));
    }

    if local.is_none() && peer.is_none() {
        return None;
    }
    if groups.is_empty() && (local.is_none() || peer.is_none()) && base.is_some() {
        // The key itself was deleted on one side and the other left it alone.
        return None;
    }
    Some(SessionSet { groups })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workbook::v3::{parse_workbook, render_workbook, WorkbookErrorKind};

    const ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn doc_with(sessions_yaml: &str) -> String {
        format!("---\nid: {ID}\nname: Sets\n{sessions_yaml}---\n\nBody.\n")
    }

    fn unknown_of(markdown: &str) -> BTreeMap<String, serde_yaml_ng::Value> {
        parse_workbook(markdown).unwrap().0.front_matter_unknown
    }

    #[test]
    fn sessions_front_matter_two_groups_groups_and_members_in_file_order() {
        // Arrange
        let md = doc_with(
            "sessions:\n  - group: Day 1\n    members:\n      - { id: aaaa1111, label: D1 S1 }\n      - { id: bbbb2222 }\n  - group: Day 2\n    members:\n      - { id: cccc3333, label: D2 S1 }\n",
        );

        // Act
        let set = parse_session_set(&unknown_of(&md)).unwrap().unwrap();

        // Assert
        assert_eq!(set.groups.len(), 2);
        assert_eq!(set.groups[0].name.as_deref(), Some("Day 1"));
        assert_eq!(set.groups[0].members[1], SetMember { id: "bbbb2222".into(), label: None });
        let flat = set.members();
        assert_eq!(flat.iter().map(|m| (m.id.as_str(), m.group)).collect::<Vec<_>>(), vec![
            ("aaaa1111", 0),
            ("bbbb2222", 0),
            ("cccc3333", 1)
        ]);
    }

    #[test]
    fn sessions_front_matter_bare_members_one_unnamed_group() {
        // Arrange
        let md = doc_with("sessions:\n  - { id: aaaa1111 }\n  - { id: bbbb2222, label: B }\n");

        // Act
        let set = parse_session_set(&unknown_of(&md)).unwrap().unwrap();

        // Assert
        assert_eq!(set.groups.len(), 1);
        assert_eq!(set.groups[0].name, None);
        assert_eq!(set.groups[0].members.len(), 2);
    }

    #[test]
    fn sessions_front_matter_absent_none() {
        // Act
        let set = parse_session_set(&unknown_of(&doc_with(""))).unwrap();

        // Assert
        assert_eq!(set, None);
    }

    #[test]
    fn sessions_front_matter_duplicate_member_workbook_duplicate_set_member() {
        // Arrange
        let md = doc_with(
            "sessions:\n  - group: A\n    members: [{ id: aaaa1111 }]\n  - group: B\n    members: [{ id: aaaa1111 }]\n",
        );

        // Act
        let (_, structural) = parse_workbook(&md).unwrap();
        let err = parse_session_set(&unknown_of(&md)).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::DuplicateSetMember);
        assert_eq!(err.cell_id, "front-matter");
        assert!(structural.iter().any(|e| e.kind == WorkbookErrorKind::DuplicateSetMember));
    }

    #[test]
    fn sessions_front_matter_repeated_group_name_invalid_session_set() {
        // Arrange
        let md = doc_with(
            "sessions:\n  - group: A\n    members: [{ id: aaaa1111 }]\n  - group: A\n    members: [{ id: bbbb2222 }]\n",
        );

        // Act
        let err = parse_session_set(&unknown_of(&md)).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::InvalidSessionSet);
    }

    #[test]
    fn sessions_front_matter_not_a_list_invalid_session_set() {
        // Act
        let err = parse_session_set(&unknown_of(&doc_with("sessions: 3\n"))).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::InvalidSessionSet);
    }

    #[test]
    fn sessions_front_matter_round_trip_bytes_unchanged() {
        // Arrange — the canonical form `render_workbook` writes.
        let md = doc_with(
            "sessions:\n  - group: Day 1\n    members:\n      - { id: aaaa1111, label: D1 S1 }\n  - group: Day 2\n    members:\n      - { id: cccc3333 }\n",
        );
        let once = render_workbook(&parse_workbook(&md).unwrap().0);

        // Act
        let twice = render_workbook(&parse_workbook(&once).unwrap().0);

        // Assert
        assert_eq!(once, twice);
        assert_eq!(parse_session_set(&unknown_of(&once)).unwrap(), parse_session_set(&unknown_of(&md)).unwrap());
    }

    #[test]
    fn session_set_to_yaml_parses_back_equal() {
        // Arrange
        let set = SessionSet {
            groups: vec![SetGroup {
                name: Some("Day 1".into()),
                members: vec![SetMember { id: "aaaa1111".into(), label: Some("S1".into()) }, SetMember {
                    id: "bbbb2222".into(),
                    label: None,
                }],
            }],
        };

        // Act
        let back = parse_session_set_value(&set.to_yaml()).unwrap();

        // Assert
        assert_eq!(back, set);
    }


    fn set(groups: &[(&str, &[&str])]) -> SessionSet {
        SessionSet {
            groups: groups
                .iter()
                .map(|(name, ids)| SetGroup {
                    name: Some(name.to_string()),
                    members: ids.iter().map(|id| SetMember { id: id.to_string(), label: None }).collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn merge_member_added_on_one_side_removed_on_the_other_both_apply() {
        // Arrange
        let base = set(&[("Day 1", &["a", "b"])]);
        let local = set(&[("Day 1", &["a", "b", "c"])]);
        let peer = set(&[("Day 1", &["b"])]);

        // Act
        let merged = merge_session_sets(Some(&local), Some(&peer), Some(&base)).unwrap();

        // Assert
        assert_eq!(merged, set(&[("Day 1", &["b", "c"])]));
    }

    #[test]
    fn merge_group_added_on_the_peer_carries_after_local_groups() {
        // Arrange
        let base = set(&[("Day 1", &["a"])]);
        let local = set(&[("Day 1", &["a"])]);
        let peer = set(&[("Day 1", &["a"]), ("Day 2", &["d"])]);

        // Act
        let merged = merge_session_sets(Some(&local), Some(&peer), Some(&base)).unwrap();

        // Assert
        assert_eq!(merged, peer);
    }

    #[test]
    fn merge_order_conflict_keeps_local_order() {
        // Arrange
        let base = set(&[("G", &["a", "b"])]);
        let local = set(&[("G", &["b", "a"])]);
        let peer = set(&[("G", &["a", "b"])]);

        // Act
        let merged = merge_session_sets(Some(&local), Some(&peer), Some(&base)).unwrap();

        // Assert
        assert_eq!(merged, local);
    }

    #[test]
    fn default_member_label_empty_event_session_first_8_of_the_id() {
        // Act + Assert
        assert_eq!(default_member_label("388740a19b3c013e", Some(" ")), "388740a1");
        assert_eq!(default_member_label("388740a19b3c013e", Some("Session 4")), "Session 4");
        assert_eq!(default_member_label("388740a19b3c013e", None), "388740a1");
    }
}
