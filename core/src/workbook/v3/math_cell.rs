//! Math-cell definition/const-line grammar (C2 §3.1). Splits a `math` cell's
//! fence body into classified lines and validates `def_line`/`const_line`
//! names against the `identifier` grammar and [`super::error::RESERVED_NAMES`]
//! — hand-written scanners throughout (ledger R20, ruling L3-R5: C2's `/…/`
//! terminals are specifications, not an instruction to add the `regex`
//! crate), matching [`crate::math::token`]'s and [`crate::math::resolve`]'s
//! existing house style.

use crate::math::token::{tokenize, Token, TokenKind};

use super::error::{self, WorkbookError};

/// One classified line of a `math` cell's fence body (C2 §3.1's
/// `math_line ::= blank_line | comment_line | const_line | def_line`).
#[derive(Debug, Clone, PartialEq)]
pub enum MathCellLine {
    /// `blank_line` — no-op.
    Blank,
    /// `comment_line` — the whole line is a `#`-prefixed comment; no-op.
    Comment,
    /// `const_line` — `const NAME = value`, workbook-scoped once flattened
    /// (C2 §3.1).
    Const {
        /// The `identifier` on the left of `=` (already validated).
        name: String,
        /// The scalar right-hand side — unitless (a `const` line's value has
        /// no physical unit attached by the grammar itself).
        value: f64,
        /// Always `None` from this parser — a `const` line's right-hand side
        /// is always a bare `number` (C2 §3.1 has no unit-suffix alternative
        /// here); kept for shape symmetry with
        /// [`super::front_matter::ConstantRaw::WithUnit`].
        unit_display: Option<String>,
    },
    /// `def_line` — `NAME = expression`, one JS host variable per definition
    /// (C2 §5.1).
    Def {
        /// The `identifier` on the left of `=` (already validated).
        name: String,
        /// The unparsed right-hand side — expression parsing (C2 §3.2) is an
        /// evaluation-time concern, not this task's.
        expr_text: String,
        /// The definition's display name, from a `# label: <text>` trailing
        /// comment (C2 §3.1). `None` when the line has no such comment.
        label: Option<String>,
    },
}

/// Parses a `math` cell's fence body (C2 §3.1's `math_body ::= math_line*`)
/// into classified lines plus any structural errors. `cell_id` is the owning
/// cell's fence id — every error this call raises is scoped to it (L3-R2).
///
/// Line-level errors (`InvalidIdentifier`, `ReservedName`) are collected,
/// never fatal to sibling lines — CLAUDE.md §5's "missing math channel
/// reference → inline validation error, don't block other channels" extends
/// to structural parsing (L3-R6). A line that fails to classify contributes
/// no [`MathCellLine`] to the returned vector — only its error.
pub fn parse_math_cell_body(cell_id: &str, body: &str) -> (Vec<MathCellLine>, Vec<WorkbookError>) {
    let mut lines = Vec::new();
    let mut errors = Vec::new();

    for raw_line in body.split('\n') {
        let (line, err) = classify_line(cell_id, raw_line);
        if let Some(line) = line {
            lines.push(line);
        }
        if let Some(err) = err {
            errors.push(err);
        }
    }

    (lines, errors)
}

/// Classifies one line per L3-R6's exact order: strip a trailing comment,
/// then blank / leading-`#` comment / `const `-prefixed / contains-`=` / the
/// `InvalidIdentifier` catch-all, in that order — not `starts_with`
/// heuristics (G3.2: `trimmed.starts_with("const")` alone would misclassify
/// `constant = [X]` as a malformed const line, since `"constant"` also
/// starts with `"const"`).
fn classify_line(cell_id: &str, raw_line: &str) -> (Option<MathCellLine>, Option<WorkbookError>) {
    let trimmed = raw_line.trim();
    let (before_hash, comment) = split_trailing_comment(trimmed);
    let main = before_hash.trim_end();

    if main.is_empty() {
        return (Some(MathCellLine::Blank), None);
    }
    if main.starts_with('#') {
        return (Some(MathCellLine::Comment), None);
    }
    if let Some(rest) = main.strip_prefix("const") {
        if rest.starts_with(' ') || rest.starts_with('\t') {
            return classify_const_line(cell_id, rest.trim_start(), main);
        }
    }
    if let Some(eq_idx) = main.find('=') {
        let name = main[..eq_idx].trim();
        let expr_text = main[eq_idx + 1..].trim().to_string();
        return classify_def_line(cell_id, name, expr_text, comment);
    }

    (None, Some(error::invalid_identifier(cell_id, fallback_name(main))))
}

/// Splits a trailing `# comment` off an already-trimmed `line` (C2 §3.1's
/// `trailing_comment`), scanning for the first `#` that is outside a
/// double-quoted string and immediately preceded by whitespace (closes
/// G3.3 — a `#` inside a `"low"`/`"none"` string argument must not split the
/// line). `line` being pre-trimmed means a whole-line comment's leading `#`
/// (no preceding whitespace at all) is never mistaken for a trailing one.
/// Returns `(before, Some(after))` — `after` excludes the `#` itself — or
/// `(line, None)` when no such `#` exists.
fn split_trailing_comment(line: &str) -> (&str, Option<&str>) {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut prev_is_space = false;

    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' {
            in_string = !in_string;
        } else if b == b'#' && !in_string && prev_is_space {
            return (&line[..i], Some(&line[i + 1..]));
        }
        prev_is_space = b == b' ' || b == b'\t';
    }

    (line, None)
}

/// L3-R6 rule 6's `<name>` extraction: the line's trimmed text up to the
/// first `=`, or the whole trimmed line if it has none. Shared by the
/// top-level catch-all and by a `const`-prefixed line that fails to parse
/// beyond the keyword (no `=`, or a right-hand side that isn't a bare
/// `number`) — both cases are "this line doesn't parse as `const NAME =
/// value` or `NAME = expr`", the same `InvalidIdentifier` template C2 §3.5.A
/// gives exactly one of.
fn fallback_name(main: &str) -> &str {
    match main.find('=') {
        Some(idx) => main[..idx].trim(),
        None => main,
    }
}

/// Classifies a line already known to start with `"const" /[ \t]+/` — `rest`
/// is the text after that keyword and its separating whitespace; `main` is
/// the whole comment-stripped line, for [`fallback_name`] if parsing fails.
fn classify_const_line(cell_id: &str, rest: &str, main: &str) -> (Option<MathCellLine>, Option<WorkbookError>) {
    let Some(eq_idx) = rest.find('=') else {
        return (None, Some(error::invalid_identifier(cell_id, fallback_name(main))));
    };

    let name = rest[..eq_idx].trim();
    let value_text = rest[eq_idx + 1..].trim();

    if let Some(err) = validate_identifier(cell_id, name) {
        return (None, Some(err));
    }

    match parse_bare_number(value_text) {
        Some(value) => (Some(MathCellLine::Const { name: name.to_string(), value, unit_display: None }), None),
        None => (None, Some(error::invalid_identifier(cell_id, fallback_name(main)))),
    }
}

/// Classifies a line already known to contain `=` — `name` is the trimmed
/// text before it, `expr_text` the trimmed text after, `comment` the
/// already-stripped trailing comment (if any), consulted only for a `#
/// label:` display name (C2 §3.1).
fn classify_def_line(
    cell_id: &str,
    name: &str,
    expr_text: String,
    comment: Option<&str>,
) -> (Option<MathCellLine>, Option<WorkbookError>) {
    if let Some(err) = validate_identifier(cell_id, name) {
        return (None, Some(err));
    }

    let label = comment.and_then(extract_label);
    (Some(MathCellLine::Def { name: name.to_string(), expr_text, label }), None)
}

/// C2 §3.1's display-name annotation: a trailing comment (already stripped
/// of its leading `#`) is a display name only when it matches `[ \t]*
/// "label" [ \t]* ":" .*` — the literal word `label`, optional whitespace, a
/// colon, then free text to end of line. Any other trailing comment has no
/// display name.
fn extract_label(comment: &str) -> Option<String> {
    let rest = comment.trim_start().strip_prefix("label")?;
    let rest = rest.trim_start().strip_prefix(':')?;
    Some(rest.trim().to_string())
}

/// Validates `name` against C2 §3.1's `identifier` grammar and
/// [`error::RESERVED_NAMES`] (ledger R20's `Time`/`Distance` widening,
/// L3-R7) — checked for both `def_line` and `const_line` names,
/// case-sensitive.
fn validate_identifier(cell_id: &str, name: &str) -> Option<WorkbookError> {
    if !is_identifier(name) {
        return Some(error::invalid_identifier(cell_id, name));
    }
    if error::RESERVED_NAMES.contains(&name) {
        return Some(error::reserved_name(cell_id, name));
    }
    None
}

/// C2 §3.1's `identifier ::= /[A-Za-z_][A-Za-z0-9_]*/` — hand-written scan
/// (ruling L3-R5), no `regex` crate.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// C2 §3.1's `const_line` `number` terminal — by reference to
/// [`crate::math::token`]'s `Number` literal (int/float, optional
/// exponent). Reuses the tokenizer itself rather than re-implementing its
/// scan, so the two can never drift; `s` must tokenize to exactly one
/// `Number` token followed by `Eof`, **or** (lead ruling R24) a leading
/// `Minus` token immediately before that same shape — the tokenizer always
/// emits a leading `-` as its own `Minus` token (unary negation is a
/// parser-level concern, C2 §3.2, not part of the `Number` terminal itself),
/// so a signed `const` value is accepted by matching that two-token prefix
/// and negating. The tokenizer skips whitespace uniformly, so `-1.5` and
/// `- 1.5` tokenize identically and are both accepted here — there is no
/// lower-level distinction available to tell them apart.
fn parse_bare_number(s: &str) -> Option<f64> {
    let tokens = tokenize(s).ok()?;
    match tokens.as_slice() {
        [Token { kind: TokenKind::Number, num_val, .. }, Token { kind: TokenKind::Eof, .. }] => Some(*num_val),
        [Token { kind: TokenKind::Minus, .. }, Token { kind: TokenKind::Number, num_val, .. }, Token { kind: TokenKind::Eof, .. }] => {
            Some(-*num_val)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::WorkbookErrorKind;

    #[test]
    fn def_line_roll_deg_eq_roll_bracket_def_with_name_roll_deg() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "roll_deg = [Roll]");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Def { name: "roll_deg".to_string(), expr_text: "[Roll]".to_string(), label: None }]
        );
    }

    #[test]
    fn def_line_with_hash_label_comment_label_captured() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "x = 1 # label: My Label");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Def {
                name: "x".to_string(),
                expr_text: "1".to_string(),
                label: Some("My Label".to_string())
            }]
        );
    }

    #[test]
    fn def_line_trailing_plain_comment_without_label_label_is_none() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "x = 1 # just a note");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Def { name: "x".to_string(), expr_text: "1".to_string(), label: None }]
        );
    }

    #[test]
    fn const_line_const_k_eq_9_81_const() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const k = 9.81");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(lines, vec![MathCellLine::Const { name: "k".to_string(), value: 9.81, unit_display: None }]);
    }

    #[test]
    fn const_line_no_equals_invalid_identifier() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const k");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::InvalidIdentifier);
        assert_eq!(errors[0].cell_id, "aaaaaaaa");
    }

    #[test]
    fn const_line_non_numeric_value_invalid_identifier() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const k = abc");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::InvalidIdentifier);
        assert_eq!(errors[0].cell_id, "aaaaaaaa");
    }

    #[test]
    fn const_line_negative_literal_value_is_negative() {
        // Arrange — R24: a `const` line's number accepts an optional leading
        // `-`; the tokenizer yields `[Minus, Number, Eof]` for it (unary
        // negation is otherwise a parser-level concern, C2 §3.2), so
        // `parse_bare_number` special-cases that one shape and negates.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const offset = -1.5");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Const { name: "offset".to_string(), value: -1.5, unit_display: None }]
        );
    }

    #[test]
    fn const_line_minus_5_with_a_space_still_accepted() {
        // Arrange — the tokenizer skips whitespace uniformly (`token.rs`'s
        // main loop advances past ' '/'\t' before dispatching on the next
        // char), so it cannot distinguish "-5" from "- 5": both scan to the
        // identical `[Minus, Number(5.0), Eof]` token stream. Since
        // `parse_bare_number` matches on that shape, not on source text, a
        // spaced sign is accepted too — there's no lower-level distinction to
        // reject it on.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const k = - 5");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(lines, vec![MathCellLine::Const { name: "k".to_string(), value: -5.0, unit_display: None }]);
    }

    #[test]
    fn const_line_constant_eq_bracket_x_def_not_misclassified_as_const_line() {
        // Arrange — G3.2: "constant" starts with "const" but is not the
        // `const` keyword followed by whitespace.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "constant = [X]");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Def { name: "constant".to_string(), expr_text: "[X]".to_string(), label: None }]
        );
    }

    #[test]
    fn def_line_with_a_string_argument_containing_hash_not_split_at_the_in_string_hash() {
        // Arrange — G3.3: the '#' inside "no#ne" must not end the line early;
        // the real trailing comment starts at the '#' after the closing `)`.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "x = detrend([X], \"no#ne\") # trailing comment");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(
            lines,
            vec![MathCellLine::Def {
                name: "x".to_string(),
                expr_text: "detrend([X], \"no#ne\")".to_string(),
                label: None
            }]
        );
    }

    #[test]
    fn identifier_starting_with_digit_invalid_identifier() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "3x = 1");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::InvalidIdentifier);
        assert_eq!(errors[0].cell_id, "aaaaaaaa");
    }

    #[test]
    fn name_const_reserved_name() {
        // Arrange — C2 §3.1's own example: `const const = 1`.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const const = 1");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
    }

    #[test]
    fn name_pi_reserved_name() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "pi = 3.14");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
    }

    #[test]
    fn name_channel_reserved_name_host_var_collision() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "channel = 1");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
    }

    #[test]
    fn def_line_name_time_reserved_name() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "Time = [Speed]");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
    }

    #[test]
    fn const_line_name_distance_reserved_name() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "const Distance = 100");

        // Assert
        assert!(lines.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::ReservedName);
    }

    #[test]
    fn blank_line_and_comment_only_line_no_error_produce_blank_comment() {
        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "\n# just a comment");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(lines, vec![MathCellLine::Blank, MathCellLine::Comment]);
    }

    #[test]
    fn indented_comment_only_line_produces_comment_not_blank() {
        // Arrange — the trailing-comment strip must not eat a whole-line
        // comment's own '#' just because it has leading indentation.

        // Act
        let (lines, errors) = parse_math_cell_body("aaaaaaaa", "    # indented comment");

        // Assert
        assert!(errors.is_empty());
        assert_eq!(lines, vec![MathCellLine::Comment]);
    }
}
