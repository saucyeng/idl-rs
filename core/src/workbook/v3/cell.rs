//! Fence scanning, `id=hex8` assignment/generation, and prose-span
//! attachment for workbook v3 (C2 §2.2–§2.4). Walks the document body (the
//! text after front matter) with `pulldown-cmark`'s event stream to find
//! fenced `math`/`table`/`js` cells; every other fence (including a bare
//! ` ``` `) is inert and produces no [`CellDoc`] (C2 §1).

use std::collections::HashSet;
use std::ops::Range;

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

use super::error::{WorkbookError, WorkbookErrorKind};

/// A cell's fence-language token (C2 §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKindToken {
    /// ` ```math ` — an expression-definition cell (C2 §3).
    Math,
    /// ` ```table ` — a `TableModel` JSON cell (C2 §4).
    Table,
    /// ` ```js ` — an Observable Runtime cell (C2 §5).
    Js,
}

/// One fenced cell (C2 §2). Prose itself is not a cell (§2.4) — it attaches
/// to the cell that follows it (`prose_before`), except for text after the
/// very last cell, which attaches to that cell as `prose_after`.
#[derive(Debug, Clone, PartialEq)]
pub struct CellDoc {
    /// 8 lowercase hex characters — either the fence's own `id=` attribute
    /// or a freshly generated one (C2 §2.2's "assignment on first save").
    /// Writing a generated id back into the source file is a future save
    /// path's job, not this task's — [`scan_cells`] only ever returns an
    /// in-memory id.
    pub id: String,
    /// `math` / `table` / `js`.
    pub kind_token: CellKindToken,
    /// Raw Markdown text (not rendered — this lane does not render prose)
    /// from the end of the previous cell (or end of front matter, for the
    /// first cell) up to this cell's fence-open. `None` when there is none.
    pub prose_before: Option<String>,
    /// Raw Markdown text after this cell's fence-close, up to the next
    /// cell's fence-open or the end of the document. Only ever `Some` on
    /// the last cell in document order (C2 §2.4) — every other cell's is
    /// always `None`.
    pub prose_after: Option<String>,
    /// The fence body's literal source text, unparsed — kind-specific
    /// grammars (C2 §3/§4/§5) are later tasks' job.
    pub raw_fence_body: String,
}

/// Parses a fence's info string against C2 §2.2's `fence_open` grammar:
/// `cell_kind (" " attr)*`, `attr ::= "id=" hex8`, `hex8 ::= [0-9a-f]{8}`.
/// Returns `(kind, id)` when `info` names a math/table/js cell — `id` is
/// `Some` only when a well-formed `id=<hex8>` attribute was present.
/// Any other attribute is the "reserved attribute namespace" C2 §2.2
/// describes (room for a future `key=value`); this task neither interprets
/// it nor has anywhere to preserve it verbatim (no write path exists yet),
/// so it is simply ignored. Returns `None` for any other fence language —
/// that fence is inert (C2 §1).
fn parse_fence_open(info: &str) -> Option<(CellKindToken, Option<String>)> {
    let mut parts = info.split_whitespace();
    let kind = match parts.next()? {
        "math" => CellKindToken::Math,
        "table" => CellKindToken::Table,
        "js" => CellKindToken::Js,
        _ => return None,
    };

    let mut id = None;
    for attr in parts {
        if let Some(value) = attr.strip_prefix("id=") {
            let is_hex8 = value.len() == 8 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
            if is_hex8 {
                id = Some(value.to_string());
            }
        }
    }
    Some((kind, id))
}

/// Generates a fresh cell id: 4 random bytes from the crate's existing
/// `uuid` dependency, lowercase-hex encoded to 8 characters (C2 §2.2).
/// Per lead ruling (`runs/2026-09-03/decisions.md`, "L3 Task 1 dispatched"),
/// this reuses `uuid::Uuid::new_v4()` rather than adding a `rand`
/// dependency — collision probability at workbook scale is astronomically
/// small either way (~2⁻³² per pair, C2 §2.2).
fn generate_cell_id() -> String {
    let random = uuid::Uuid::new_v4();
    random.as_bytes()[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// Consumes events up to and including the next `End(TagEnd::CodeBlock)`,
/// concatenating any `Text` content seen along the way. Used both to
/// capture a recognised cell's `raw_fence_body` and to drain (and discard)
/// an inert fence's body without misreading it as top-level prose.
fn code_block_text<'a, I>(events: &mut I) -> String
where
    I: Iterator<Item = (Event<'a>, Range<usize>)>,
{
    let mut text = String::new();
    for (event, _) in events {
        match event {
            Event::Text(t) => text.push_str(&t),
            Event::End(TagEnd::CodeBlock) => break,
            _ => {}
        }
    }
    text
}

/// Fence-scans `body` (the document text after front matter, C2 §1) into
/// its cells, attaching prose spans (C2 §2.4) and assigning/validating cell
/// ids (C2 §2.2) as it goes. Returns `(cells, trailing_prose, errors)`:
/// `trailing_prose` is `Some` only when `cells` is empty (a pure-prose
/// document, C2 §2.4) — otherwise any text after the last cell lives on
/// that cell's own `prose_after`. Duplicate-id problems are collected into
/// `errors`, never fatal (C2 §3.5.A; this module's caller,
/// [`super::parse_workbook`], is the only place front-matter-fatal errors
/// short-circuit).
pub fn scan_cells(body: &str) -> (Vec<CellDoc>, Option<String>, Vec<WorkbookError>) {
    let mut cells: Vec<CellDoc> = Vec::new();
    let mut errors: Vec<WorkbookError> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut segment_start = 0usize;

    let mut events = Parser::new_ext(body, Options::empty()).into_offset_iter();
    while let Some((event, range)) = events.next() {
        let Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(info))) = event else {
            continue;
        };

        let Some((kind_token, explicit_id)) = parse_fence_open(&info) else {
            // Inert fence (C2 §1): drain its body without capturing it, and
            // leave the source text in place for whatever prose span
            // eventually attaches to the next recognised cell.
            let _ = code_block_text(&mut events);
            continue;
        };

        let prose_before = body[segment_start..range.start].to_string();
        let prose_before = if prose_before.is_empty() { None } else { Some(prose_before) };

        let raw_fence_body = code_block_text(&mut events);

        let id = match explicit_id {
            Some(id) => {
                if !seen_ids.insert(id.clone()) {
                    errors.push(WorkbookError::new(
                        id.clone(),
                        WorkbookErrorKind::DuplicateCellId,
                        format!("Cell id '{id}' used by more than one cell"),
                    ));
                }
                id
            }
            None => {
                // Loops only on a hash collision against an id already seen
                // in this document (~2⁻³² per pair, C2 §2.2) — astronomically
                // unlikely, but the generated-id path must uphold the same
                // "never two cells share an id" invariant the explicit-id
                // branch above enforces, so a hit regenerates rather than
                // silently aliasing two cells.
                loop {
                    let generated = generate_cell_id();
                    if seen_ids.insert(generated.clone()) {
                        break generated;
                    }
                }
            }
        };

        cells.push(CellDoc { id, kind_token, prose_before, prose_after: None, raw_fence_body });
        // `range` is the *Start* event's byte range, but it is used here as
        // the fence-close boundary (the next cell's `prose_before` starts
        // right after it). This relies on pulldown-cmark 0.13.4 giving
        // `Start(CodeBlock)` and the eventual `End(CodeBlock)` the same
        // range — both are the tree node's full `item.start..item.end`
        // (verified against that version's `parse.rs`'s `OffsetIter::next`)
        // — which is not part of pulldown-cmark's stated public contract. A
        // future upgrade that changes this should re-read the range from the
        // `End` event instead of assuming it still matches `Start`'s.
        segment_start = range.end;
    }

    let remainder = &body[segment_start..];
    let remainder = if remainder.is_empty() { None } else { Some(remainder.to_string()) };

    let trailing_prose = if let Some(last) = cells.last_mut() {
        last.prose_after = remainder;
        None
    } else {
        remainder
    };

    (cells, trailing_prose, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_cells_same_id_duplicate_cell_id_collected_both_cells_still_returned() {
        // Arrange
        let body = "```math id=aaaaaaaa\nx = 1\n```\n\n```js id=aaaaaaaa\nx\n```\n";

        // Act
        let (cells, _trailing, errors) = scan_cells(body);

        // Assert
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].id, "aaaaaaaa");
        assert_eq!(cells[1].id, "aaaaaaaa");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, WorkbookErrorKind::DuplicateCellId);
        assert_eq!(errors[0].message, "Cell id 'aaaaaaaa' used by more than one cell");
    }

    #[test]
    fn fence_with_no_id_id_assigned_8_lowercase_hex_chars() {
        // Arrange
        let body = "```math\nx = 1\n```\n";

        // Act
        let (cells, _trailing, errors) = scan_cells(body);

        // Assert
        assert!(errors.is_empty());
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].id.len(), 8);
        assert!(cells[0].id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }

    #[test]
    fn unrecognised_fence_language_produces_no_cell_round_trips_as_inert() {
        // Arrange
        let body = "```bash\necho hi\n```\n\n```math id=aaaaaaaa\nx = 1\n```\n";

        // Act
        let (cells, _trailing, errors) = scan_cells(body);

        // Assert
        assert!(errors.is_empty());
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].kind_token, CellKindToken::Math);
        assert!(cells[0].prose_before.as_deref().unwrap().contains("```bash"));
    }

    #[test]
    fn prose_before_first_math_cell_becomes_that_cells_prose_before() {
        // Arrange
        let body = "# Fork tuning\n\n```math id=aaaaaaaa\nx = 1\n```\n";

        // Act
        let (cells, _trailing, _errors) = scan_cells(body);

        // Assert
        assert_eq!(cells[0].prose_before.as_deref(), Some("# Fork tuning\n\n"));
    }

    #[test]
    fn trailing_prose_after_last_cell_becomes_prose_after_on_the_last_cell() {
        // Arrange
        let body = "```math id=aaaaaaaa\nx = 1\n```\n\nBottom-outs: 3\n";

        // Act
        let (cells, trailing, _errors) = scan_cells(body);

        // Assert
        assert_eq!(cells[0].prose_after.as_deref(), Some("\n\nBottom-outs: 3\n"));
        assert_eq!(trailing, None);
    }

    #[test]
    fn zero_fenced_cells_cells_empty_trailing_prose_is_some() {
        // Arrange
        let body = "Just a written note, no cells here.\n";

        // Act
        let (cells, trailing, errors) = scan_cells(body);

        // Assert
        assert!(cells.is_empty());
        assert!(errors.is_empty());
        assert_eq!(trailing.as_deref(), Some("Just a written note, no cells here.\n"));
    }
}
