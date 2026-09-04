//! Flat constants table (C2 §3.1): merges the two constant sources — the
//! front-matter `constants` map and every workbook-wide `const` line — into
//! one `name → f64` table for
//! [`crate::math::parse::parse_with_constants`]. This module is workbook
//! v3's single enforcement point for both the full C2 §3.5.A `ReservedName`
//! set and `DuplicateConstant`, applied to constants from **either**
//! source — [`super::parse_workbook`] raises neither of these kinds for a
//! `Const` line itself (see that function's doc comment).

use std::collections::HashMap;

use super::error::{self, WorkbookError};
use super::front_matter::ConstantRaw;
use super::ConstLine;

/// Which source first claimed a name in the merged table — tracked only so
/// a later collision picks the right `cell_id` for
/// [`error::duplicate_constant`] (see [`merge_constants`]'s doc comment).
enum Owner {
    FrontMatter,
    ConstLine,
}

/// Builds the flat constants table (C2 §3.1) and validates it: a name in
/// [`error::RESERVED_NAMES`] — Task 2's full 15-entry set, not just the four
/// universal math constants — is reported ([`error::reserved_name`]) and
/// **not** merged, from either source; a name declared twice (front matter +
/// a `const` line, or two `const` lines) is [`error::duplicate_constant`].
/// Front matter is processed first (order doesn't affect the final table,
/// but duplicate detection must see both sources) — the *first* declaration
/// of a name wins its table entry, and every later colliding declaration is
/// reported without replacing it.
///
/// **`cell_id` on a `DuplicateConstant`:** when the name's first claim came
/// from front matter, every later collision (from any `const` line) is
/// reported at the front-matter pseudo-cell id — matching
/// [`error::duplicate_constant`]'s own doc comment ("the literal
/// `"front-matter"` when the collision involves the front-matter `constants`
/// map"). When the first claim came from a `const` line, a later collision
/// is reported at the *colliding* line's own `cell_id` — the same "cell
/// holding the second occurrence" rule [`error::duplicate_definition`]
/// already uses.
///
/// For a `ConstantRaw::Number(v)` the merged value is `v`; for
/// `ConstantRaw::WithUnit { value, .. }` it is `value` — `unit_display` is
/// display metadata only (C2 §3.1) and is never consulted here. Constant
/// values are unitless scalars throughout this function.
///
/// **Front-matter names with spaces (C2 §3.1).** Unlike a `const` line's
/// name, a front-matter constant name is not restricted to `identifier` — a
/// name like `"rider mass"` is legal and lands in the returned table as-is.
/// This is correct, not a bug: such a name is only reachable from the JS
/// `constants` object (`constants["rider mass"]`, C2 §5.1), never as a bare
/// identifier — the math-cell tokenizer cannot produce a spaced bare
/// identifier, so `parse_with_constants` can never resolve it via
/// `[Name]`-free substitution from *inside* a `math`/`const` expression.
/// Stated here so the next reader doesn't rediscover it as a defect.
pub fn merge_constants(
    front_matter: &HashMap<String, ConstantRaw>,
    const_lines: &[ConstLine],
) -> (HashMap<String, f64>, Vec<WorkbookError>) {
    let mut table = HashMap::new();
    let mut owners: HashMap<String, Owner> = HashMap::new();
    let mut errors = Vec::new();

    for (name, raw) in front_matter {
        if error::RESERVED_NAMES.contains(&name.as_str()) {
            errors.push(error::reserved_name("front-matter", name));
            continue;
        }
        let value = match raw {
            ConstantRaw::Number(v) => *v,
            ConstantRaw::WithUnit { value, .. } => *value,
        };
        table.insert(name.clone(), value);
        owners.insert(name.clone(), Owner::FrontMatter);
    }

    for line in const_lines {
        if error::RESERVED_NAMES.contains(&line.name.as_str()) {
            errors.push(error::reserved_name(&line.cell_id, &line.name));
            continue;
        }
        match owners.get(&line.name) {
            Some(Owner::FrontMatter) => {
                errors.push(error::duplicate_constant("front-matter", &line.name));
            }
            Some(Owner::ConstLine) => {
                errors.push(error::duplicate_constant(&line.cell_id, &line.name));
            }
            None => {
                table.insert(line.name.clone(), line.value);
                owners.insert(line.name.clone(), Owner::ConstLine);
            }
        }
    }

    (table, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::WorkbookErrorKind;

    fn const_line(cell_id: &str, name: &str, value: f64) -> ConstLine {
        ConstLine { cell_id: cell_id.to_string(), name: name.to_string(), value, unit_display: None }
    }

    #[test]
    fn same_name_in_front_matter_and_a_const_line_duplicate_constant() {
        // Arrange
        let front_matter = HashMap::from([("k".to_string(), ConstantRaw::Number(5.0))]);
        let const_lines = [const_line("aaaaaaaa", "k", 9.0)];

        // Act
        let (table, errors) = merge_constants(&front_matter, &const_lines);

        // Assert — front matter's value wins the table entry; the collision
        // is reported at the front-matter pseudo-cell id.
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::DuplicateConstant);
        assert_eq!(errors[0].cell_id, "front-matter");
        assert_eq!(table.get("k"), Some(&5.0));
    }

    #[test]
    fn two_const_lines_named_k_duplicate_constant_second_occurrence_reported() {
        // Arrange
        let front_matter = HashMap::new();
        let const_lines = [const_line("aaaaaaaa", "k", 1.0), const_line("bbbbbbbb", "k", 2.0)];

        // Act
        let (table, errors) = merge_constants(&front_matter, &const_lines);

        // Assert — first declaration's value wins; the second's own cell
        // reports the collision.
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::DuplicateConstant);
        assert_eq!(errors[0].cell_id, "bbbbbbbb");
        assert_eq!(table.get("k"), Some(&1.0));
    }

    #[test]
    fn front_matter_constant_named_g_reserved_name_not_merged() {
        // Arrange
        let front_matter = HashMap::from([("g".to_string(), ConstantRaw::Number(9.80665))]);

        // Act
        let (table, errors) = merge_constants(&front_matter, &[]);

        // Assert
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
        assert_eq!(errors[0].cell_id, "front-matter");
        assert!(!table.contains_key("g"));
    }

    #[test]
    fn front_matter_constant_named_session_reserved_name_not_merged() {
        // Arrange — L3-R10's explicit host-var-collision test (C2 §5.1's
        // `session` host variable).
        let front_matter = HashMap::from([("session".to_string(), ConstantRaw::Number(1.0))]);

        // Act
        let (table, errors) = merge_constants(&front_matter, &[]);

        // Assert
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
        assert!(!table.contains_key("session"));
    }

    #[test]
    fn front_matter_constant_named_const_reserved_name() {
        // Arrange
        let front_matter = HashMap::from([("const".to_string(), ConstantRaw::Number(1.0))]);

        // Act
        let (table, errors) = merge_constants(&front_matter, &[]);

        // Assert
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
        assert!(!table.contains_key("const"));
    }

    #[test]
    fn constant_name_with_a_space_front_matter_only_allowed_present_in_table() {
        // Arrange — C2 §3.1: front-matter constant names are unrestricted,
        // unlike a `const` line's `identifier`-restricted name.
        let front_matter = HashMap::from([("rider mass".to_string(), ConstantRaw::Number(82.0))]);

        // Act
        let (table, errors) = merge_constants(&front_matter, &[]);

        // Assert
        assert!(errors.is_empty());
        assert_eq!(table.get("rider mass"), Some(&82.0));
    }
}
