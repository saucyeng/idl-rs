//! The retired-builtin-name migration table and its rewriter (R143/R146,
//! `runs/2026-09-08/scipy-alignment-plan.md` §3, lead ruling R151). Decision
//! 75: an update **migrates** workbooks, it does not break them — a name
//! retired here keeps parsing for one revision (C2 §3.5's `version: 3`
//! branch), is rewritten to its current spelling on save, and the change is
//! reported rather than left silent.
//!
//! **Safety property (R151 item 1):** every entry in this table changes what
//! a call is *called*, never what it computes. A rename whose new spelling
//! would also need different arguments (e.g. `fft`'s eventual split into
//! `periodogram`/`welch`, R146) is [`MigrationKind::Rewrite`] instead, and
//! the migration pins the rewritten call to the option that reproduces
//! today's numbers exactly — no existing workbook's numbers are allowed to
//! move under cover of a naming change.
//!
//! The rewriter is a byte splice, not a re-serialisation: it walks `src`
//! looking for an identifier immediately followed (ignoring whitespace) by
//! `(`, skipping over anything inside a `'…'`/`"…"` string, a `[Channel]`
//! name or a `{cell}` reference so those are never mistaken for a call site.
//! Every other byte is copied through untouched — comments, spacing and the
//! author's own formatting survive, so one rename does not turn into a
//! whole-line reflow (which would make every migrated cell look like an
//! unrelated edit to L11's merge).

/// Whether a table entry is a same-arguments rename, or a rewrite to a call
/// with different arguments (reserved for a future entry like `fft`'s split
/// — R146; no entry uses this today).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationKind {
    /// The call is re-spelled; its arguments are untouched.
    Rename,
    /// The call is re-spelled *and* its arguments change shape.
    #[allow(dead_code)] // no table entry uses this yet (task 6 will)
    Rewrite,
}

/// One retired builtin name → its replacement (see the module doc for the
/// safety property this table exists to uphold).
#[derive(Debug, Clone, Copy)]
pub struct NameMigration {
    /// The retired spelling, as it appears in a `version: 3` workbook.
    pub old: &'static str,
    /// The current spelling, as `math::eval::call_function` dispatches it.
    pub new: &'static str,
    /// [`MigrationKind::Rename`] for every entry today.
    pub kind: MigrationKind,
}

/// The full retired-name table (C2 §3.8's "Retired names" list, hand
/// transcribed here — see [`crate::math::catalog`]'s module doc for why this
/// codebase transcribes rather than derives its function metadata). Consulted
/// by every migration door named in the plan: in-memory evaluation of a
/// `version: 3` document, [`migrate_expression`]/[`migrate_document`], and a
/// `version: 4` document's [`crate::math::error::MathEvalErrorKind::UnknownFunction`]
/// message for a name that appears here.
pub fn math_name_migrations() -> &'static [NameMigration] {
    use MigrationKind::{Rename, Rewrite};
    &[
        // `variance_time`/`variance_dist` never compute a variance (σ²) —
        // they compute a lap's delta against an overlay lap (R143's worst
        // false friend in the catalog: "variance" has one meaning in every
        // statistics library a model has read).
        NameMigration { old: "variance_time", new: "lap_delta_time", kind: Rename },
        NameMigration { old: "variance_dist", new: "lap_delta_dist", kind: Rename },
        // `fft(ch, window)` was one un-normalised windowed magnitude
        // spectrum, no segmentation, no averaging — not the Welch spectrum
        // the charts already compute under the same word (R146). Split into
        // `periodogram` (this shape, scipy-named/scaled) and `welch`
        // (segmented/averaged); `fft` itself is retired and reserved for a
        // true complex DFT. `rewrite_call_args` pins every migrated call to
        // `scaling="raw_magnitude"` so no existing workbook's spectrum
        // moves (R151 item 1) — `new` here is the migration's primary
        // target, not the only function `fft` retired in favour of;
        // `unknown_function_error`'s message additionally names `welch`.
        NameMigration { old: "fft", new: "periodogram", kind: Rewrite },
        // `numpy.angle` is complex phase, not the angle between two vectors
        // — a false friend (R143/R151 item 4).
        NameMigration { old: "angle", new: "angle_between", kind: Rename },
    ]
}

/// One rename applied by [`migrate_expression`], for the caller to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedRename {
    /// The retired spelling that was found.
    pub old: String,
    /// What it was rewritten to.
    pub new: String,
    /// The byte offset of the identifier's first byte within the `src`
    /// passed to [`migrate_expression`].
    pub position: usize,
}

/// Rewrites every retired builtin call in `src` to its current spelling,
/// leaving every other byte untouched (see the module doc's "byte splice,
/// not a re-serialisation"). A [`MigrationKind::Rename`] entry only ever
/// touches the identifier itself; a [`MigrationKind::Rewrite`] entry (only
/// `fft`, task 6) replaces the whole call — identifier *and* argument list —
/// with [`rewrite_call_args`]'s reshaped text, still leaving every byte
/// outside that one call site untouched. Idempotent: running this on
/// already-current source finds nothing and returns `(src.to_string(),
/// vec![])`.
pub fn migrate_expression(src: &str) -> (String, Vec<AppliedRename>) {
    let table = math_name_migrations();
    let mut out = String::with_capacity(src.len());
    let mut renames = Vec::new();
    let mut copied_to = 0usize;

    for (start, end) in call_site_idents(src) {
        let ident = &src[start..end];
        let Some(m) = table.iter().find(|m| m.old == ident) else {
            continue;
        };
        let (call_end, replacement) = match m.kind {
            MigrationKind::Rename => (end, m.new.to_string()),
            MigrationKind::Rewrite => match rewrite_call_args(src, end, m.old, m.new) {
                Some((call_end, text)) => (call_end, text),
                // Defensive: the call's own argument shape doesn't match
                // what this migration expects (e.g. already hand-edited to
                // a wrong arity) — leave the whole call untouched rather
                // than emit a half-rewritten one; it surfaces as a normal
                // typed evaluation error instead (`unknown_function_error`,
                // `eval::call_function`) rather than a corrupted document.
                None => continue,
            },
        };
        out.push_str(&src[copied_to..start]);
        out.push_str(&replacement);
        renames.push(AppliedRename { old: m.old.to_string(), new: m.new.to_string(), position: start });
        copied_to = call_end;
    }
    out.push_str(&src[copied_to..]);

    (out, renames)
}

/// [`MigrationKind::Rewrite`]'s whole-call reshaping, for the one entry that
/// uses it today: `fft(ch, window)` → `periodogram(ch, window=window,
/// detrend="none", scaling="raw_magnitude")` (R151 item 1) — pinning every
/// existing call to the scaling *and* detrend that reproduce its old,
/// un-normalised numbers exactly, so adopting scipy's name never moves a
/// workbook's existing values. `detrend="none"` is load-bearing here: the
/// legacy `fft()` never detrended, but `periodogram`'s own new-caller
/// default is scipy's `detrend="constant"` — leaving it off the migrated
/// call would silently subtract each segment's mean under cover of a
/// rename, exactly the defect class this migration exists to prevent.
/// `ident_end` is the byte offset right after `old`'s identifier (the `(`
/// that must follow it, per [`call_site_idents`], may have whitespace
/// before it). Returns `(byte offset right after the call's closing ')',
/// replacement text)`, or `None` when the call's argument list isn't the
/// shape this rewrite expects (see [`migrate_expression`]'s fallback).
fn rewrite_call_args(src: &str, ident_end: usize, old: &str, new: &str) -> Option<(usize, String)> {
    // Only `fft`'s 2-positional-argument shape exists today; a future
    // second `Rewrite` entry would need its own arm here.
    if old != "fft" {
        return None;
    }
    let open = src[ident_end..].find('(')? + ident_end;
    let close = matching_close_paren(src, open)?;
    let args = split_top_level_args(&src[open + 1..close]);
    let [channel_arg, window_arg] = args.as_slice() else { return None };
    let text = format!("{new}({channel_arg}, window={window_arg}, detrend=\"none\", scaling=\"raw_magnitude\")");
    Some((close + 1, text))
}

/// Finds the `)` matching the `(` at byte offset `open` in `src`, skipping
/// the interior of any `'…'`/`"…"` string, `[…]` channel reference or
/// `{…}` cell reference (none can contain an unbalanced paren of their
/// own) and accounting for nested nested parens — an argument can itself be
/// a call, e.g. `fft(butter(...), "hann")`.
fn matching_close_paren(src: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut chars = src[open..].char_indices();
    while let Some((rel, c)) = chars.next() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + rel);
                }
            }
            '\'' | '"' => {
                let quote = c;
                for (_, c2) in chars.by_ref() {
                    if c2 == quote {
                        break;
                    }
                }
            }
            '[' => {
                for (_, c2) in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                }
            }
            '{' => {
                for (_, c2) in chars.by_ref() {
                    if c2 == '}' {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Splits `text` (a call's argument-list interior, exclusive of the
/// surrounding parens) at top-level commas — i.e. not inside a nested
/// `(...)`, `[...]`, `{...}`, or `'...'`/`"..."` string — trimming
/// surrounding whitespace off each piece. Used only by [`rewrite_call_args`]
/// today, where every produced piece is re-embedded verbatim into a brand
/// new call, so trimming (rather than preserving original inter-argument
/// spacing) is the right call: the whole call site is being reshaped, not
/// spliced byte-for-byte.
fn split_top_level_args(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '\'' | '"' => {
                let quote = c;
                for (_, c2) in chars.by_ref() {
                    if c2 == quote {
                        break;
                    }
                }
            }
            ',' if depth == 0 => {
                parts.push(text[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() || !parts.is_empty() {
        parts.push(tail);
    }
    parts
}

/// Finds every `(start, end)` byte span of an identifier immediately
/// followed, ignoring `' '`/`'\t'`/`'\n'`/`'\r'`, by `(` — i.e. every call
/// site — while skipping the interior of any `'…'`/`"…"` string, `[…]`
/// channel reference or `{…}` cell reference, none of which can contain a
/// call site of its own (`math::token::tokenize` treats all three the same
/// way: verbatim content, never re-scanned for sub-tokens).
fn call_site_idents(src: &str) -> Vec<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut spans = Vec::new();
    let mut chars = src.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        match c {
            '\'' | '"' => {
                let quote = c;
                for (_, c2) in chars.by_ref() {
                    if c2 == quote {
                        break;
                    }
                }
            }
            '[' => {
                for (_, c2) in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                }
            }
            '{' => {
                for (_, c2) in chars.by_ref() {
                    if c2 == '}' {
                        break;
                    }
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                let mut end = i + c.len_utf8();
                while let Some(&(j, c2)) = chars.peek() {
                    if c2.is_ascii_alphanumeric() || c2 == '_' {
                        end = j + c2.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                let mut k = end;
                while k < bytes.len() && matches!(bytes[k], b' ' | b'\t' | b'\n' | b'\r') {
                    k += 1;
                }
                if bytes.get(k) == Some(&b'(') {
                    spans.push((start, end));
                }
            }
            _ => {}
        }
    }

    spans
}

/// One rename applied by [`migrate_document`] (or [`migrate_body`]), scoped
/// to the cell it was found in — the shape `save_workbook` (C3 §3.4) reports
/// through `SaveResult.migrations`. `line` is the 1-based line number within
/// the cell's raw fence body for a `math` cell rename, or the 1-based line
/// number within the raw JSON for a `table` cell rename (there is no
/// `def_line` to anchor to there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRename {
    /// The fence's `id=hex8` (C2 §2.2).
    pub cell_id: String,
    /// 1-based line number, scoped as described above.
    pub line: usize,
    /// The retired spelling that was found.
    pub old: String,
    /// What it was rewritten to.
    pub new: String,
}

/// Rewrites every retired builtin call in `markdown`'s `math`-cell
/// def_lines and `table`-cell `formula`/`template` strings to its current
/// spelling, and bumps front matter `version` to `4`. A no-op on a document
/// that is already `version: 4` (or whose front matter fails to parse —
/// callers see that failure through the ordinary read path, not here):
/// returns `(markdown.to_string(), vec![])`. Idempotent: running this twice
/// reports nothing the second time (C2 §3.7).
///
/// This is the "save" side of R151 item 9 — the door that actually changes
/// the bytes on disk. The "open" side (a passive, read-only notice) and the
/// merge-time normalisation (§3.4 of the plan) both need only
/// [`migrate_body`]'s renames, never this function's version bump or its
/// front-matter rewrite.
pub fn migrate_document(markdown: &str) -> (String, Vec<DocumentRename>) {
    let Ok((front_matter, body)) = crate::workbook::v3::front_matter::parse_front_matter(markdown) else {
        return (markdown.to_string(), Vec::new());
    };
    if front_matter.version != 3 {
        return (markdown.to_string(), Vec::new());
    }

    let (new_body, renames) = migrate_body(body);
    if renames.is_empty() {
        return (markdown.to_string(), Vec::new());
    }

    let front_matter_end = markdown.len() - body.len();
    let new_front_matter = bump_version_to_4(&markdown[..front_matter_end]);
    (format!("{new_front_matter}{new_body}"), renames)
}

/// Byte-splices a literal `version: 3` line in a front-matter block to
/// `version: 4`, preserving every other byte (including comments and key
/// order) — a full YAML round-trip would risk reflowing the whole block for
/// a one-character change. When the key is absent (defaulted to `3`, C2 §1),
/// inserts an explicit `version: 4` as the block's first key, right after
/// the opening `---\n`.
fn bump_version_to_4(front_matter_block: &str) -> String {
    for line_start in front_matter_block.match_indices('\n').map(|(i, _)| i + 1).chain(std::iter::once(0)) {
        let rest = &front_matter_block[line_start..];
        if let Some(after) = rest.strip_prefix("version:") {
            let value_start = line_start + "version:".len();
            let trimmed = after.trim_start();
            let leading_ws = after.len() - trimmed.len();
            let digits_len = trimmed.bytes().take_while(u8::is_ascii_digit).count();
            if trimmed[..digits_len].trim() == "3" {
                let digit_start = value_start + leading_ws;
                let digit_end = digit_start + digits_len;
                return format!(
                    "{}4{}",
                    &front_matter_block[..digit_start],
                    &front_matter_block[digit_end..]
                );
            }
        }
    }
    // No explicit `version:` key — insert one right after the opening fence.
    match front_matter_block.find('\n') {
        Some(i) => format!("{}version: 4\n{}", &front_matter_block[..=i], &front_matter_block[i + 1..]),
        None => front_matter_block.to_string(),
    }
}

/// One cell's kind dispatched to the matching rewriter — [`migrate_body`]'s
/// per-cell primitive, exposed standalone for the merge-time normalisation
/// (plan §3.4), which already holds parsed [`crate::workbook::v3::CellDoc`]
/// values (`kind_token`/`id`/`raw_fence_body`) rather than a raw markdown
/// document to re-scan. `js` cells are returned unchanged (R145: textual
/// rewriting of arbitrary JS is out of scope).
pub fn migrate_cell_body(
    kind: crate::workbook::v3::CellKindToken,
    cell_id: &str,
    body: &str,
) -> (String, Vec<DocumentRename>) {
    use crate::workbook::v3::CellKindToken;

    let mut renames = Vec::new();
    let new_body = match kind {
        CellKindToken::Math => rewrite_math_fence(cell_id, body, &mut renames),
        CellKindToken::Table => rewrite_table_fence(cell_id, body, &mut renames),
        CellKindToken::Js => body.to_string(),
    };
    (new_body, renames)
}

/// The document-walk half of [`migrate_document`], operating on `body` (the
/// text after front matter, as [`crate::workbook::v3::front_matter::parse_front_matter`]
/// returns it). Every `math` and `table` cell's fence body is walked in
/// document order via [`migrate_cell_body`] and spliced back at its own byte
/// range in `body`; `js` cells and every byte outside a fence (front matter
/// already excluded, prose, fence delimiters) are copied through unchanged.
pub fn migrate_body(body: &str) -> (String, Vec<DocumentRename>) {
    use crate::workbook::v3::cell::scan_cells;

    let (cells, _trailing_prose, _errors) = scan_cells(body);
    let mut out = String::with_capacity(body.len());
    let mut copied_to = 0usize;
    let mut search_from = 0usize;
    let mut renames = Vec::new();

    for cell in &cells {
        if cell.raw_fence_body.is_empty() {
            continue;
        }
        // The fence body is captured verbatim by `scan_cells` but without its
        // own byte range; recovered here by finding it starting from where
        // the previous cell's fence body ended — document order plus a
        // monotonically advancing search start makes this unambiguous.
        let Some(rel_start) = body[search_from..].find(cell.raw_fence_body.as_str()) else {
            continue; // defensive: should be unreachable, `scan_cells` read this exact text from `body`
        };
        let start = search_from + rel_start;
        let end = start + cell.raw_fence_body.len();

        let (new_fence_body, cell_renames) = migrate_cell_body(cell.kind_token, &cell.id, &cell.raw_fence_body);
        renames.extend(cell_renames);

        out.push_str(&body[copied_to..start]);
        out.push_str(&new_fence_body);
        copied_to = end;
        search_from = end;
    }
    out.push_str(&body[copied_to..]);

    (out, renames)
}

/// Rewrites one `math` cell's def_lines via
/// [`crate::workbook::v3::math_cell::rewrite_math_cell_body`], collecting a
/// [`DocumentRename`] per name actually changed.
fn rewrite_math_fence(cell_id: &str, body: &str, renames: &mut Vec<DocumentRename>) -> String {
    crate::workbook::v3::math_cell::rewrite_math_cell_body(body, |line_no, expr| {
        let (new_expr, applied) = migrate_expression(expr);
        for a in applied {
            renames.push(DocumentRename { cell_id: cell_id.to_string(), line: line_no, old: a.old, new: a.new });
        }
        new_expr
    })
}

/// Rewrites one `table` cell's raw JSON, splicing [`migrate_expression`]
/// over every `"formula"` and `"template"` JSON string value (C2 §4) without
/// parsing the JSON structurally — `TableModel`'s serialized form loses
/// whitespace/key-order fidelity on a round trip, which the splice approach
/// (see module doc) must not do. A JSON string's escape sequences (`\"`,
/// `\\`, …) are left exactly as they appear; `migrate_expression`'s own
/// string-literal skip (`'…'`/`"…"`) already treats an escaped `\"` inside a
/// formula's own quoted argument (e.g. `butter(…, "low", …)`) as an ordinary
/// quote pair, so nothing inside it is misread as a call site.
fn rewrite_table_fence(cell_id: &str, body: &str, renames: &mut Vec<DocumentRename>) -> String {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for key in ["formula", "template"] {
        spans.extend(json_string_value_spans(body, key));
    }
    spans.sort_unstable();

    let mut out = String::with_capacity(body.len());
    let mut copied_to = 0usize;
    for (start, end) in spans {
        let (new_value, applied) = migrate_expression(&body[start..end]);
        if applied.is_empty() {
            continue;
        }
        let line = body[..start].matches('\n').count() + 1;
        for a in applied {
            renames.push(DocumentRename { cell_id: cell_id.to_string(), line, old: a.old, new: a.new });
        }
        out.push_str(&body[copied_to..start]);
        out.push_str(&new_value);
        copied_to = end;
    }
    out.push_str(&body[copied_to..]);
    out
}

/// Finds every `"key": "…"` JSON string value's inner-content byte span in
/// `text` — a hand-written scanner (matching this crate's existing house
/// style for grammars simple enough not to need a JSON path/range library,
/// e.g. `workbook::v3::math_cell`'s line classifier) rather than a
/// structural JSON parse, because a structural parse's own serialization
/// would reflow the document. Skips a `"key": null` value (no span emitted).
/// Does not attempt to unescape the content — see [`rewrite_table_fence`]'s
/// doc comment for why that is safe here.
fn json_string_value_spans(text: &str, key: &str) -> Vec<(usize, usize)> {
    let pat = format!("\"{key}\"");
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find(pat.as_str()) {
        let key_start = search_from + rel;
        let mut i = key_start + pat.len();
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b':') {
            search_from = key_start + pat.len();
            continue;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if bytes.get(i) == Some(&b'"') {
            let content_start = i + 1;
            let mut j = content_start;
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' {
                    j += 2;
                } else {
                    j += 1;
                }
            }
            let content_end = j.min(bytes.len());
            spans.push((content_start, content_end));
            search_from = content_end + 1;
        } else {
            search_from = key_start + pat.len();
        }
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_expression_rewrites_a_bare_call_and_reports_the_rename() {
        // Arrange / Act
        let (out, applied) = migrate_expression("variance_time([LapTime])");

        // Assert
        assert_eq!(out, "lap_delta_time([LapTime])");
        assert_eq!(
            applied,
            vec![AppliedRename { old: "variance_time".to_string(), new: "lap_delta_time".to_string(), position: 0 }]
        );
    }

    #[test]
    fn migrate_expression_rewrites_a_call_nested_inside_an_operator_expression() {
        // Arrange / Act
        let (out, applied) = migrate_expression("2 * variance_dist([LapTime]) + 1");

        // Assert
        assert_eq!(out, "2 * lap_delta_dist([LapTime]) + 1");
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].old, "variance_dist");
    }

    #[test]
    fn migrate_expression_does_not_touch_a_channel_name_that_matches_a_retired_call_name() {
        // Arrange — `[variance_time]` is a channel reference, not a call;
        // the identifier inside `[...]` must never be treated as a call site.

        // Act
        let (out, applied) = migrate_expression("[variance_time] + 1");

        // Assert
        assert_eq!(out, "[variance_time] + 1");
        assert!(applied.is_empty());
    }

    #[test]
    fn migrate_expression_does_not_touch_a_string_argument_that_matches_a_retired_call_name() {
        // Act
        let (out, applied) = migrate_expression("p([X], \"variance_time\")");

        // Assert
        assert_eq!(out, "p([X], \"variance_time\")");
        assert!(applied.is_empty());
    }

    #[test]
    fn migrate_expression_is_idempotent_on_already_current_source() {
        // Arrange
        let src = "lap_delta_time([LapTime]) + lap_delta_dist([LapTime])";

        // Act
        let (out, applied) = migrate_expression(src);

        // Assert
        assert_eq!(out, src);
        assert!(applied.is_empty());
    }

    #[test]
    fn migrate_expression_tolerates_whitespace_between_the_name_and_the_paren() {
        // Act
        let (out, applied) = migrate_expression("variance_time  ([LapTime])");

        // Assert — the replaced span is only the identifier; the original
        // whitespace before `(` survives untouched.
        assert_eq!(out, "lap_delta_time  ([LapTime])");
        assert_eq!(applied.len(), 1);
    }

    #[test]
    fn math_name_migrations_table_has_no_duplicate_old_names() {
        // Arrange
        let table = math_name_migrations();
        let mut olds: Vec<&str> = table.iter().map(|m| m.old).collect();
        let before = olds.len();

        // Act
        olds.sort_unstable();
        olds.dedup();

        // Assert
        assert_eq!(olds.len(), before);
    }

    #[test]
    fn migrate_expression_rewrites_angle_to_angle_between() {
        // Act
        let (out, applied) = migrate_expression("angle([A], [B])");

        // Assert
        assert_eq!(out, "angle_between([A], [B])");
        assert_eq!(
            applied,
            vec![AppliedRename { old: "angle".to_string(), new: "angle_between".to_string(), position: 0 }]
        );
    }

    #[test]
    fn migrate_expression_rewrites_fft_to_periodogram_pinning_the_old_numbers() {
        // Arrange / Act — R151 item 1: the migration must pin both the
        // scaling *and* the detrend that reproduce `fft()`'s old numbers,
        // not just rename the call.
        let (out, applied) = migrate_expression("fft([X], \"hann\")");

        // Assert
        assert_eq!(out, "periodogram([X], window=\"hann\", detrend=\"none\", scaling=\"raw_magnitude\")");
        assert_eq!(applied, vec![AppliedRename { old: "fft".to_string(), new: "periodogram".to_string(), position: 0 }]);
    }

    #[test]
    fn migrate_expression_rewrites_fft_nested_inside_an_operator_expression() {
        // Act
        let (out, applied) = migrate_expression("2 * fft([X], \"rect\") + 1");

        // Assert — only the call site is reshaped; the surrounding operator
        // expression's bytes are untouched.
        assert_eq!(out, "2 * periodogram([X], window=\"rect\", detrend=\"none\", scaling=\"raw_magnitude\") + 1");
        assert_eq!(applied.len(), 1);
    }

    #[test]
    fn migrate_expression_fft_with_a_channel_argument_that_is_itself_a_call_result() {
        // Arrange — the argument-splitting must respect nested parens, not
        // just split on every top-level-looking comma naively.
        let (out, applied) = migrate_expression("fft(butter(2, 0.3, \"low\", [X]), \"hann\")");

        // Assert
        assert_eq!(
            out,
            "periodogram(butter(2, 0.3, \"low\", [X]), window=\"hann\", detrend=\"none\", scaling=\"raw_magnitude\")"
        );
        assert_eq!(applied.len(), 1);
    }

    #[test]
    fn migrate_expression_an_fft_call_with_the_wrong_arity_is_left_untouched() {
        // Arrange — defensive: a malformed call (already-hand-edited, wrong
        // arg count) is left byte-for-byte rather than half-rewritten.
        let src = "fft([X])";

        // Act
        let (out, applied) = migrate_expression(src);

        // Assert
        assert_eq!(out, src);
        assert!(applied.is_empty());
    }

    #[test]
    fn migrate_expression_is_idempotent_after_migrating_fft() {
        // Arrange
        let (once, _) = migrate_expression("fft([X], \"hann\")");

        // Act
        let (twice, applied) = migrate_expression(&once);

        // Assert
        assert_eq!(twice, once);
        assert!(applied.is_empty());
    }

    fn wb(version_line: &str, body: &str) -> String {
        format!(
            "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Fork tuning\n{version_line}---\n{body}"
        )
    }

    #[test]
    fn migrate_document_rewrites_a_math_cells_def_line_and_bumps_the_version() {
        // Arrange
        let src = wb("version: 3\n", "```math id=a1b2c3d4\nv = variance_time([LapTime])\n```\n");

        // Act
        let (out, renames) = migrate_document(&src);

        // Assert
        assert!(out.contains("version: 4"));
        assert!(!out.contains("version: 3"));
        assert!(out.contains("v = lap_delta_time([LapTime])"));
        assert_eq!(renames.len(), 1);
        assert_eq!(renames[0].cell_id, "a1b2c3d4");
        assert_eq!(renames[0].old, "variance_time");
        assert_eq!(renames[0].new, "lap_delta_time");
    }

    #[test]
    fn migrate_document_is_a_no_op_on_an_already_v4_document() {
        // Arrange
        let src = wb("version: 4\n", "```math id=a1b2c3d4\nv = variance_time([LapTime])\n```\n");

        // Act
        let (out, renames) = migrate_document(&src);

        // Assert — a v4 document's old names are a typed error at eval time
        // (R151 item 10), never silently migrated.
        assert_eq!(out, src);
        assert!(renames.is_empty());
    }

    #[test]
    fn migrate_document_running_twice_reports_nothing_the_second_time() {
        // Arrange
        let src = wb("version: 3\n", "```math id=a1b2c3d4\nv = variance_time([LapTime])\n```\n");
        let (once, _) = migrate_document(&src);

        // Act
        let (twice, renames) = migrate_document(&once);

        // Assert
        assert_eq!(twice, once);
        assert!(renames.is_empty());
    }

    #[test]
    fn migrate_document_leaves_a_document_with_no_retired_names_unchanged_even_at_v3() {
        // Arrange — no rename found means no reason to touch the file at
        // all, including its version key (nothing to report on save either).
        let src = wb("version: 3\n", "```math id=a1b2c3d4\nv = mean([X])\n```\n");

        // Act
        let (out, renames) = migrate_document(&src);

        // Assert
        assert_eq!(out, src);
        assert!(renames.is_empty());
    }

    #[test]
    fn migrate_document_preserves_a_trailing_label_comment_and_a_comment_line() {
        // Arrange
        let src = wb(
            "version: 3\n",
            "```math id=a1b2c3d4\n# the old variance_time path\nv = variance_time([LapTime])  # label: Delta\n```\n",
        );

        // Act
        let (out, _renames) = migrate_document(&src);

        // Assert — the prose comment mentioning the retired name in English
        // is untouched; only the call site is rewritten; the label survives.
        assert!(out.contains("# the old variance_time path"));
        assert!(out.contains("v = lap_delta_time([LapTime])  # label: Delta"));
    }

    #[test]
    fn migrate_document_rewrites_a_table_cells_formula_and_template() {
        // Arrange
        let table_json = r#"{"columns":[{"id":"c1","name":null,"template":"variance_time([LapTime])"}],"rows":[{"id":"r1","context":null}],"cells":[[{"formula":"variance_dist([LapTime])","literal":null,"name":null}]]}"#;
        let src = wb("version: 3\n", &format!("```table id=b2c3d4e5\n{table_json}\n```\n"));

        // Act
        let (out, renames) = migrate_document(&src);

        // Assert
        assert!(out.contains("\"template\":\"lap_delta_time([LapTime])\""));
        assert!(out.contains("\"formula\":\"lap_delta_dist([LapTime])\""));
        assert_eq!(renames.len(), 2);
        assert!(renames.iter().all(|r| r.cell_id == "b2c3d4e5"));
    }

    #[test]
    fn migrate_document_leaves_a_js_cell_untouched() {
        // Arrange — R145: textual rewriting of arbitrary JS is out of scope;
        // a `js` cell is never a migration surface even if it happens to
        // spell a retired name.
        let src = wb(
            "version: 3\n",
            "```math id=a1b2c3d4\nv = variance_time([LapTime])\n```\n\n```js id=e5f6a7b8\nvariance_time(1)\n```\n",
        );

        // Act
        let (out, renames) = migrate_document(&src);

        // Assert
        assert!(out.contains("```js id=e5f6a7b8\nvariance_time(1)\n```"));
        assert_eq!(renames.len(), 1);
    }

    #[test]
    fn bump_version_to_4_replaces_an_explicit_version_3_in_place() {
        // Act
        let out = bump_version_to_4("---\nid: x\nversion: 3\nname: y\n---\n");

        // Assert — only the digit changed; every other byte, including key
        // order, is untouched.
        assert_eq!(out, "---\nid: x\nversion: 4\nname: y\n---\n");
    }

    #[test]
    fn bump_version_to_4_inserts_the_key_when_it_was_defaulted() {
        // Act
        let out = bump_version_to_4("---\nid: x\nname: y\n---\n");

        // Assert
        assert_eq!(out, "---\nversion: 4\nid: x\nname: y\n---\n");
    }
}
