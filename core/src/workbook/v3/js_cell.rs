//! Inline `${…}` span extraction from prose (C2 §5.2's `inline_expr ::= "${"
//! js_expression "}"` grammar). Used to find every splice point a `js` cell's
//! host-mediated scope evaluates and re-evaluates reactively (C2 §5.2) —
//! parsing/evaluating the JavaScript itself is L6's job; this module only
//! finds the byte spans and hands back the raw expression text.

use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser, Tag};

/// One `${…}` span found in a cell's prose (C2 §5.2). `start`/`end` are
/// **byte offsets into the `prose` slice passed to [`find_inline_exprs`]**,
/// not into the containing document — L6 splices the evaluated result in
/// using these, L11 merges near them (L3-R24).
#[derive(Debug, Clone, PartialEq)]
pub struct InlineExpr {
    /// Byte offset of the leading `$` of `${` in `prose`.
    pub start: usize,
    /// Byte offset one past the closing `}` in `prose`.
    pub end: usize,
    /// The JavaScript expression text between the outermost `{` and `}`
    /// (exclusive of both), verbatim including interior whitespace.
    pub js_expr: String,
}

/// Byte ranges in `prose` that `pulldown-cmark` reports as an inline code
/// span (`` `…` ``) or a code block (fenced/indented) — `${…}` text inside
/// these is inert markdown/code, not a template expression (mirrors
/// `cell.rs`'s `scan_cells`, which likewise leaves an inert fence's `${…}`
/// untouched inside `prose_before`, L3-R24). Reuses the same
/// `into_offset_iter()` parse-pass style `scan_cells` already establishes.
fn inert_ranges(prose: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    for (event, range) in Parser::new_ext(prose, Options::empty()).into_offset_iter() {
        match event {
            Event::Code(_) => ranges.push(range),
            Event::Start(Tag::CodeBlock(_)) => ranges.push(range),
            _ => {}
        }
    }
    ranges
}

/// Scans `prose` for `${…}` spans (C2 §5.2), skipping any byte range
/// `pulldown-cmark` reports as inline code or a code block (L3-R24). A
/// brace-depth counter tracks nesting past the opening `${`, honouring
/// `'…'`/`"…"`/`` `…` `` string literals (with `\`-escapes) so a `}`
/// inside a JS string does not close the span early (G8.9) — unlike a
/// naive `{`/`}` counter. Nested `${…}` inside a template literal is out
/// of scope: the scanner tracks `{`/`}` depth only, never a literal `${`
/// once already inside a span (L3-R24).
pub fn find_inline_exprs(prose: &str) -> Vec<InlineExpr> {
    let inert = inert_ranges(prose);
    let is_inert = |pos: usize| inert.iter().any(|r| r.contains(&pos));

    let bytes = prose.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_inert(i) {
            i += 1;
            continue;
        }
        if bytes[i] == b'$' && bytes.get(i + 1) == Some(&b'{') {
            let start = i;
            let mut j = i + 2;
            let mut depth = 1i32;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'{' => {
                        depth += 1;
                        j += 1;
                    }
                    b'}' => {
                        depth -= 1;
                        j += 1;
                    }
                    quote @ (b'\'' | b'"' | b'`') => {
                        j += 1;
                        while j < bytes.len() && bytes[j] != quote {
                            // `\`-escape: skip the escaped byte too, so an
                            // escaped quote never closes the string early.
                            if bytes[j] == b'\\' && j + 1 < bytes.len() {
                                j += 2;
                            } else {
                                j += 1;
                            }
                        }
                        if j < bytes.len() {
                            j += 1; // consume the closing quote
                        }
                    }
                    _ => j += 1,
                }
            }
            if depth == 0 {
                let js_expr = prose[start + 2..j - 1].to_string();
                out.push(InlineExpr { start, end: j, js_expr });
                i = j;
                continue;
            }
            // Unterminated `${…}` (no matching close brace) — not a valid
            // span; stop scanning rather than misreading the remainder.
            break;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_span_in_prose_one_inline_expr_js_expr_matches() {
        // Arrange
        let prose = "Bottom-outs: ${count(x)}";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].js_expr, "count(x)");
        assert_eq!(&prose[got[0].start..got[0].end], "${count(x)}");
    }

    #[test]
    fn two_spans_two_inline_exprs_in_document_order() {
        // Arrange
        let prose = "${a} and ${b}";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].js_expr, "a");
        assert_eq!(got[1].js_expr, "b");
        assert!(got[0].start < got[1].start);
    }

    #[test]
    fn nested_object_literal_brace_balanced_not_closed_early() {
        // Arrange — the depth-counter case that motivates it over a naive regex.
        let prose = "${ {a: 1} }";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].js_expr, " {a: 1} ");
    }

    #[test]
    fn no_dollar_brace_present_empty_vec() {
        // Arrange
        let prose = "Just plain prose, no expressions here.";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert!(got.is_empty());
    }

    #[test]
    fn brace_inside_a_string_literal_is_ignored_by_the_depth_counter() {
        // Arrange — L3-R24: `}` inside a JS string must not close the span.
        let prose = "${ x + \"}\" }";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].js_expr, " x + \"}\" ");
    }

    #[test]
    fn dollar_brace_inside_a_fenced_code_block_in_prose_is_not_extracted() {
        // Arrange — mirrors `cell.rs`'s fence-in-prose test: an inert fence's
        // body must not be scanned for `${…}` (L3-R24).
        let prose = "Before.\n\n```bash\necho ${not_an_expr}\n```\n\nAfter: ${real(x)}\n";

        // Act
        let got = find_inline_exprs(prose);

        // Assert
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].js_expr, "real(x)");
    }
}
