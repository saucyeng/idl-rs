//! Structural (parse-time) error model for workbook v3 — distinct from
//! [`crate::math::MathEvalError`] (evaluation-time, per-definition failures,
//! reused verbatim by later tasks per C2 §3.5.B). Every structural error is
//! scoped to a cell (`cell_id`) or to the literal `"front-matter"` for a
//! front-matter-level problem (C2 §3.5).

use std::fmt;

/// Discriminant for [`WorkbookError`]. This task (front matter, fence
/// scanning, cell-id assignment) defines only the kinds it produces; Task 2
/// extends this enum with the remaining structural kinds C2 §3.5.A lists
/// (`DuplicateDefinition`, `DuplicateConstant`, `InvalidIdentifier`,
/// `ReservedName`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkbookErrorKind {
    /// Two fenced cells share an `id=` attribute (C2 §2.2, §3.5.A).
    DuplicateCellId,
    /// Front matter lacks a well-formed UUIDv4 `id` — including the case
    /// where the front-matter YAML doesn't parse at all, since no `id` can
    /// be trusted either way (C2 §1, §3.5.A).
    MissingFrontMatterId,
    /// Front matter's `version` key is present and not `3` (C2 §1 — only
    /// *absence* of the key defaults to 3; a wrong value never does).
    UnsupportedWorkbookVersion,
}

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
}
