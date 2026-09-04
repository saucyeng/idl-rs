//! Structural (parse-time) error model for workbook v3 — distinct from
//! [`crate::math::MathEvalError`] (evaluation-time, per-definition failures,
//! reused verbatim by later tasks per C2 §3.5.B). Every structural error is
//! scoped to a cell (`cell_id`) or to the literal `"front-matter"` for a
//! front-matter-level problem (C2 §3.5).

use std::fmt;

/// Discriminant for [`WorkbookError`]. Task 1 (front matter, fence scanning,
/// cell-id assignment) landed the first three kinds; Task 2 adds the
/// remaining four C2 §3.5.A lists (`DuplicateDefinition`, `DuplicateConstant`,
/// `InvalidIdentifier`, `ReservedName`) — raised by Tasks 3/4's math-cell and
/// constants parsers, not by this task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkbookErrorKind {
    /// Two fenced cells share an `id=` attribute (C2 §2.2, §3.5.A).
    DuplicateCellId,
    /// The same identifier is defined more than once, in any `math` cell, on
    /// any line (C2 §3.5.A).
    DuplicateDefinition,
    /// The same constant name is declared more than once — front matter and
    /// a `const` line, or two `const` lines (C2 §3.5.A).
    DuplicateConstant,
    /// A `def_line`/`const_line` name doesn't match the §3.1 `identifier`
    /// grammar (C2 §3.5.A) — also the default for a math-cell line matching
    /// none of §3.1's four `math_line` forms (ledger R7, R20).
    InvalidIdentifier,
    /// A definition or constant is named `const`, one of the four universal
    /// constants, a JS-cell host-variable name, or one of the two
    /// engine-synthesized channel names `Time`/`Distance` (C2 §3.5.A,
    /// amended by lead ruling R20 — see [`RESERVED_NAMES`]).
    ReservedName,
    /// Front matter lacks a well-formed UUIDv4 `id` — including the case
    /// where the front-matter YAML doesn't parse at all, since no `id` can
    /// be trusted either way (C2 §1, §3.5.A).
    MissingFrontMatterId,
    /// Front matter's `version` key is present and not `3` (C2 §1 — only
    /// *absence* of the key defaults to 3; a wrong value never does).
    UnsupportedWorkbookVersion,
}

/// Names a `math`-cell definition or a `const` declaration may not use (C2
/// §3.5.A `ReservedName`): the `const` keyword itself, the four universal
/// math constants (`pi`, `tau`, `e`, `g`), the eight JS-cell host variables
/// (C2 §5.1 — `Plot`, `d3`, `Inputs`, `html`, `laps`, `session`, `constants`,
/// `channel`), and the two engine-synthesized channel names `Time`/
/// `Distance` (lead ruling R20, `runs/2026-09-03/decisions.md` — amended
/// into C2 §3.5.A post-sign; a future host-variable addition must be added
/// here too, or it will silently shadow rather than being refused). 15
/// entries, case-sensitive.
pub const RESERVED_NAMES: [&str; 15] = [
    "const", "pi", "tau", "e", "g", "Plot", "d3", "Inputs", "html", "laps", "session", "constants", "channel", "Time",
    "Distance",
];

/// A structural workbook error: a `kind` discriminant, the owning cell's id
/// (or the literal `"front-matter"`), and a human-readable message. Mirrors
/// [`crate::math::MathEvalError`]'s `kind` + `message` shape, with the
/// `cell_id` C2 §3.5's error model adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkbookError {
    pub cell_id: String,
    pub kind: WorkbookErrorKind,
    pub message: String,
}

impl WorkbookError {
    /// Builds an error scoped to `cell_id`.
    pub fn new(cell_id: impl Into<String>, kind: WorkbookErrorKind, message: impl Into<String>) -> Self {
        Self { cell_id: cell_id.into(), kind, message: message.into() }
    }

    /// Builds a front-matter-level error (C2 §3.5's reserved `"front-matter"`
    /// cell id).
    pub fn front_matter(kind: WorkbookErrorKind, message: impl Into<String>) -> Self {
        Self::new("front-matter", kind, message)
    }
}

impl fmt::Display for WorkbookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for WorkbookError {}

/// `DuplicateCellId` (C2 §3.5.A) — `id` is both the offending fence id and
/// this error's `cell_id`, since the duplicated id itself names the cell.
pub fn duplicate_cell_id(id: &str) -> WorkbookError {
    WorkbookError::new(id, WorkbookErrorKind::DuplicateCellId, format!("Cell id '{id}' used by more than one cell"))
}

/// `DuplicateDefinition` (C2 §3.5.A) — `cell_id` is the cell holding the
/// second (repeat) definition of `name`.
pub fn duplicate_definition(cell_id: &str, name: &str) -> WorkbookError {
    WorkbookError::new(cell_id, WorkbookErrorKind::DuplicateDefinition, format!("'{name}' is defined more than once"))
}

/// `DuplicateConstant` (C2 §3.5.A) — `cell_id` is the owning cell for a
/// `const`-line collision, or the literal `"front-matter"` when the
/// collision involves the front-matter `constants` map.
pub fn duplicate_constant(cell_id: &str, name: &str) -> WorkbookError {
    WorkbookError::new(
        cell_id,
        WorkbookErrorKind::DuplicateConstant,
        format!("Constant '{name}' is declared more than once"),
    )
}

/// `InvalidIdentifier` (C2 §3.5.A) — `name` is the offending text: a
/// `def_line`/`const_line` name that fails the §3.1 `identifier` grammar, or
/// (ledger R7) a stray math-cell line's trimmed text up to the first `=` (or
/// the whole trimmed line, if none). Exactly one template for both cases —
/// do not introduce a second `InvalidIdentifier` message.
pub fn invalid_identifier(cell_id: &str, name: &str) -> WorkbookError {
    WorkbookError::new(
        cell_id,
        WorkbookErrorKind::InvalidIdentifier,
        format!("'{name}' is not a valid definition name \u{2014} use letters, digits, underscore, and don't start with a digit"),
    )
}

/// `ReservedName` (C2 §3.5.A) — `name` matched an entry in
/// [`RESERVED_NAMES`].
pub fn reserved_name(cell_id: &str, name: &str) -> WorkbookError {
    WorkbookError::new(
        cell_id,
        WorkbookErrorKind::ReservedName,
        format!("'{name}' is reserved and can't be used as a definition or constant name"),
    )
}

/// `MissingFrontMatterId` (C2 §3.5.A).
pub fn missing_front_matter_id() -> WorkbookError {
    WorkbookError::front_matter(WorkbookErrorKind::MissingFrontMatterId, "Workbook front matter is missing a valid 'id'")
}

/// `UnsupportedWorkbookVersion` (C2 §3.5.A) — `n` is front matter's actual
/// `version` value.
pub fn unsupported_workbook_version(n: u32) -> WorkbookError {
    WorkbookError::front_matter(
        WorkbookErrorKind::UnsupportedWorkbookVersion,
        format!("Workbook version {n} is not supported (expected 3)"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_message_and_front_matter_helper_sets_reserved_cell_id() {
        // Arrange
        let e = WorkbookError::front_matter(
            WorkbookErrorKind::MissingFrontMatterId,
            "Workbook front matter is missing a valid 'id'",
        );

        // Act
        let s = format!("{e}");

        // Assert
        assert_eq!(s, "Workbook front matter is missing a valid 'id'");
        assert_eq!(e.cell_id, "front-matter");
        assert_eq!(e.kind, WorkbookErrorKind::MissingFrontMatterId);
    }

    #[test]
    fn duplicate_cell_id_message_matches_c2_exactly() {
        // Act
        let e = duplicate_cell_id("aaaaaaaa");

        // Assert
        assert_eq!(e.message, "Cell id 'aaaaaaaa' used by more than one cell");
        assert_eq!(e.kind, WorkbookErrorKind::DuplicateCellId);
        assert_eq!(e.cell_id, "aaaaaaaa");
    }

    #[test]
    fn duplicate_definition_message_matches_c2_exactly() {
        // Act
        let e = duplicate_definition("aaaaaaaa", "fork_velocity");

        // Assert
        assert_eq!(e.message, "'fork_velocity' is defined more than once");
        assert_eq!(e.kind, WorkbookErrorKind::DuplicateDefinition);
        assert_eq!(e.cell_id, "aaaaaaaa");
    }

    #[test]
    fn duplicate_constant_message_matches_c2_exactly() {
        // Act
        let e = duplicate_constant("aaaaaaaa", "rider_mass_kg");

        // Assert
        assert_eq!(e.message, "Constant 'rider_mass_kg' is declared more than once");
        assert_eq!(e.kind, WorkbookErrorKind::DuplicateConstant);
        assert_eq!(e.cell_id, "aaaaaaaa");
    }

    #[test]
    fn invalid_identifier_message_matches_c2_exactly() {
        // Act
        let e = invalid_identifier("aaaaaaaa", "3invalid");

        // Assert
        assert_eq!(
            e.message,
            "'3invalid' is not a valid definition name \u{2014} use letters, digits, underscore, and don't start with a digit"
        );
        assert_eq!(e.kind, WorkbookErrorKind::InvalidIdentifier);
        assert_eq!(e.cell_id, "aaaaaaaa");
    }

    #[test]
    fn reserved_name_message_matches_c2_exactly() {
        // Act
        let e = reserved_name("aaaaaaaa", "pi");

        // Assert
        assert_eq!(e.message, "'pi' is reserved and can't be used as a definition or constant name");
        assert_eq!(e.kind, WorkbookErrorKind::ReservedName);
        assert_eq!(e.cell_id, "aaaaaaaa");
    }

    #[test]
    fn missing_front_matter_id_message_matches_c2_exactly() {
        // Act
        let e = missing_front_matter_id();

        // Assert
        assert_eq!(e.message, "Workbook front matter is missing a valid 'id'");
        assert_eq!(e.kind, WorkbookErrorKind::MissingFrontMatterId);
        assert_eq!(e.cell_id, "front-matter");
    }

    #[test]
    fn unsupported_workbook_version_message_matches_c2_exactly() {
        // Act
        let e = unsupported_workbook_version(5);

        // Assert
        assert_eq!(e.message, "Workbook version 5 is not supported (expected 3)");
        assert_eq!(e.kind, WorkbookErrorKind::UnsupportedWorkbookVersion);
        assert_eq!(e.cell_id, "front-matter");
    }

    #[test]
    fn reserved_names_has_exactly_15_entries() {
        // Assert
        assert_eq!(RESERVED_NAMES.len(), 15);
    }

    #[test]
    fn workbook_error_clone_and_equality_hold_for_a_constructed_error() {
        // Arrange
        let a = duplicate_cell_id("aaaaaaaa");

        // Act
        let b = a.clone();

        // Assert
        assert_eq!(a, b);
        assert!(!format!("{a:?}").is_empty());
    }
}
