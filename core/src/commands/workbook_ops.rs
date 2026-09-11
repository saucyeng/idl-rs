//! The skeletons behind `workbook new` (ruling R229, via R230's "no
//! CLI-only logic").
//!
//! R225 ratified that there is no template *concept* yet — "New from
//! template" and "Create" fold into one `workbook.new`. A template here is
//! therefore nothing more than a named starting document: a closed set of
//! source skeletons, named in
//! [`crate::commands::table::WORKBOOK_TEMPLATES`], so adding one is a new
//! value in that list rather than a new verb or a new ruling.
//!
//! Skeletons are emitted as workbook *source text*, not as a constructed
//! `WorkbookDoc`, because source text is what the file holds and what the
//! app's editor opens. Every skeleton is round-tripped through
//! [`crate::workbook::v3::parse_workbook`] in this module's tests, so a
//! template that would not parse fails the build rather than the user.

use std::collections::{BTreeMap, HashMap};

use crate::workbook::v3::front_matter::render_front_matter;
use crate::workbook::v3::{FrontMatter, UnitsPref};

/// The workbook schema version a freshly created workbook is written at.
/// `3` documents still parse (and migrate on read); nothing new is written
/// at `3`.
pub const NEW_WORKBOOK_VERSION: u32 = 4;

/// Which skeleton [`new_workbook_source`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkbookTemplate {
    /// Front matter and a title. Nothing else — the default.
    Blank,
    /// Front matter, a title naming the session, and one `math` cell holding
    /// a worked example definition to edit.
    Session,
}

impl WorkbookTemplate {
    /// The name as the CLI spells it, matching
    /// [`crate::commands::table::WORKBOOK_TEMPLATES`].
    pub fn as_str(self) -> &'static str {
        match self {
            WorkbookTemplate::Blank => "blank",
            WorkbookTemplate::Session => "session",
        }
    }

    /// Parses the CLI spelling. `None` for a name no template has.
    pub fn from_str(name: &str) -> Option<WorkbookTemplate> {
        match name {
            "blank" => Some(WorkbookTemplate::Blank),
            "session" => Some(WorkbookTemplate::Session),
            _ => None,
        }
    }
}

/// Everything [`new_workbook_source`] needs that it cannot invent: the
/// workbook's identity, its display name, and the session the `session`
/// template refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewWorkbook {
    /// The workbook's stable id (C2 §1) — a UUIDv4 string. Taken as a
    /// parameter rather than generated inside, so the same inputs always
    /// produce the same bytes and the function stays testable.
    pub id: String,
    /// Display name, also the document's title heading.
    pub name: String,
    /// The session id the `session` template names. Ignored by `Blank`.
    pub session_id: Option<String>,
}

/// A fresh UUIDv4 for [`NewWorkbook::id`].
pub fn new_workbook_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Renders one template as workbook source text.
///
/// The result always ends in a newline and never contains a carriage
/// return, so a workbook created on Windows and one created on Linux are
/// the same bytes.
pub fn new_workbook_source(template: WorkbookTemplate, spec: &NewWorkbook) -> String {
    let front = FrontMatter {
        id: spec.id.clone(),
        name: spec.name.clone(),
        constants: HashMap::new(),
        units: UnitsPref::default(),
        version: NEW_WORKBOOK_VERSION,
        unknown: BTreeMap::new(),
    };

    let mut out = render_front_matter(&front);
    out.push('\n');
    out.push_str(&format!("# {}\n", spec.name));

    match template {
        WorkbookTemplate::Blank => {}
        WorkbookTemplate::Session => {
            out.push('\n');
            match &spec.session_id {
                Some(session_id) => {
                    out.push_str(&format!("Analysis of session `{session_id}`.\n"));
                }
                None => out.push_str("Analysis of a session.\n"),
            }
            out.push('\n');
            out.push_str("```math id=00000001\n");
            // Plain ASCII in a generated file: a workbook is edited by hand
            // in whatever editor the user has, and the template should not
            // be the thing that introduces a non-ASCII byte.
            out.push_str("# Replace [speed] with a channel this session has.\n");
            out.push_str("# `idl-rs session show <id>` lists them.\n");
            out.push_str("speed_kmh = [speed] * 3.6\n");
            out.push_str("top_speed_kmh = max([speed_kmh])\n");
            out.push_str("```\n");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::table::WORKBOOK_TEMPLATES;
    use crate::workbook::v3::parse_workbook;

    fn spec() -> NewWorkbook {
        NewWorkbook {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Cadwell test day".to_string(),
            session_id: Some("s-123".to_string()),
        }
    }

    #[test]
    fn every_template_name_in_the_command_table_parses_to_a_template() {
        // Arrange
        let names = WORKBOOK_TEMPLATES;

        // Act
        let parsed: Vec<Option<WorkbookTemplate>> =
            names.iter().map(|n| WorkbookTemplate::from_str(n)).collect();

        // Assert — the table's closed value list and this enum cannot drift.
        assert!(parsed.iter().all(|t| t.is_some()), "unparsed template names in {names:?}");
    }

    #[test]
    fn every_template_round_trips_its_own_name() {
        // Arrange
        let templates = [WorkbookTemplate::Blank, WorkbookTemplate::Session];

        // Act / Assert
        for template in templates {
            assert_eq!(WorkbookTemplate::from_str(template.as_str()), Some(template));
        }
    }

    #[test]
    fn an_unknown_template_name_parses_to_nothing() {
        // Arrange
        let name = "racecar";

        // Act
        let template = WorkbookTemplate::from_str(name);

        // Assert
        assert!(template.is_none());
    }

    #[test]
    fn the_blank_template_parses_as_a_workbook_with_no_cells() {
        // Arrange
        let source = new_workbook_source(WorkbookTemplate::Blank, &spec());

        // Act
        let (doc, errors) = parse_workbook(&source).expect("blank template must parse");

        // Assert
        assert!(errors.is_empty(), "{errors:?}");
        assert!(doc.cells.is_empty());
        assert_eq!(doc.name, "Cadwell test day");
        assert_eq!(doc.version, NEW_WORKBOOK_VERSION);
    }

    #[test]
    fn the_session_template_parses_as_a_workbook_with_one_math_cell() {
        // Arrange
        let source = new_workbook_source(WorkbookTemplate::Session, &spec());

        // Act
        let (doc, errors) = parse_workbook(&source).expect("session template must parse");

        // Assert
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(doc.cells.len(), 1);
        assert_eq!(doc.defs.len(), 2);
    }

    #[test]
    fn the_session_template_names_the_session_it_was_created_for() {
        // Arrange
        let source = new_workbook_source(WorkbookTemplate::Session, &spec());

        // Act
        let mentions = source.contains("`s-123`");

        // Assert
        assert!(mentions);
    }

    #[test]
    fn the_session_template_without_a_session_id_still_parses() {
        // Arrange
        let spec = NewWorkbook { session_id: None, ..spec() };

        // Act
        let source = new_workbook_source(WorkbookTemplate::Session, &spec);

        // Assert
        let (doc, errors) = parse_workbook(&source).expect("must parse without a session");
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(doc.cells.len(), 1);
    }

    #[test]
    fn a_rendered_template_carries_the_id_it_was_given() {
        // Arrange
        let spec = spec();

        // Act
        let source = new_workbook_source(WorkbookTemplate::Blank, &spec);

        // Assert
        let (doc, _) = parse_workbook(&source).unwrap();
        assert_eq!(doc.id, spec.id);
    }

    #[test]
    fn a_rendered_template_is_ascii_apart_from_what_the_caller_supplied() {
        // Arrange
        let spec = NewWorkbook { name: "Plain".to_string(), ..spec() };

        // Act
        let source = new_workbook_source(WorkbookTemplate::Session, &spec);

        // Assert — the skeleton itself introduces no non-ASCII byte.
        assert!(source.is_ascii(), "template is not ASCII: {source}");
    }

    #[test]
    fn a_rendered_template_contains_no_carriage_returns() {
        // Arrange
        let templates = [WorkbookTemplate::Blank, WorkbookTemplate::Session];

        // Act / Assert — a workbook made on Windows and one made on Linux
        // have to be the same bytes.
        for template in templates {
            let source = new_workbook_source(template, &spec());
            assert!(!source.contains('\r'), "{} has a carriage return", template.as_str());
        }
    }

    #[test]
    fn rendering_the_same_spec_twice_gives_identical_bytes() {
        // Arrange
        let spec = spec();

        // Act
        let first = new_workbook_source(WorkbookTemplate::Session, &spec);
        let second = new_workbook_source(WorkbookTemplate::Session, &spec);

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn a_generated_id_is_a_distinct_uuid_each_time() {
        // Arrange
        let first = new_workbook_id();

        // Act
        let second = new_workbook_id();

        // Assert
        assert_ne!(first, second);
        assert_eq!(first.len(), 36);
    }
}
