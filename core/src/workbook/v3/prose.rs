//! Renders a cell's prose Markdown (`CellDoc::prose_before`/`prose_after`,
//! C2 §2.4) to HTML for the sandbox iframe to display (ledger R69). Every
//! `${…}` inline span (C2 §5.2) found by [`find_inline_exprs`] — reused
//! here, not re-scanned — becomes a `<span data-span-id="…"></span>`
//! placeholder for the sandbox to fill once it evaluates the expression
//! (L6 Task 13b, not this module's job). Raw HTML the author typed is
//! escaped rather than passed through: R69's model is that prose HTML lands
//! in the sandbox iframe, and a `.idl1wb` file can arrive over LAN sync (a
//! peer's workbook, C4 §6) — a literal `<script>` or `onerror=` attribute
//! must never execute, only render as visible text.

use pulldown_cmark::{html, Event, Options, Parser};

use super::find_inline_exprs;

/// One `${…}` inline span (C2 §5.2) as it travels with a rendered prose
/// block's HTML: `id` is the exact `data-span-id` value the matching
/// `<span>` placeholder in [`RenderedProse::html`] carries; `expr` is the
/// verbatim JavaScript text `find_inline_exprs` found inside the braces.
/// Handed back alongside the HTML (ledger R70) so a sandbox consumer (L6
/// Task 13b) never needs its own `${…}` scan of the raw prose text.
#[derive(Debug, Clone, PartialEq)]
pub struct ProseSpanRef {
    /// Matches the `data-span-id` attribute of this span's placeholder
    /// `<span>` in [`RenderedProse::html`] — stable per (cell, index): the
    /// same `prose` text and `span_id_prefix` always produce the same ids.
    pub id: String,
    /// The JavaScript expression text between `${` and `}`, verbatim
    /// (C2 §5.2) — [`super::InlineExpr::js_expr`] passed through unchanged.
    pub expr: String,
}

/// [`render_prose_html`]'s return: the rendered HTML plus every `${…}` span
/// it placeholdered, in document order.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedProse {
    /// `text` rendered to HTML via `pulldown-cmark`, with each `${…}` span
    /// replaced by a `<span data-span-id="{span_id_prefix}:{i}"></span>`
    /// placeholder (`i` 0-indexed in document order) and all raw HTML the
    /// author typed escaped to visible literal text (ledger R69) — see
    /// [`escape_html_text`]'s doc comment for why.
    pub html: String,
    /// One entry per placeholder in `html`, same order, `id` matching that
    /// placeholder's `data-span-id` exactly (ledger R70).
    pub spans: Vec<ProseSpanRef>,
}

/// Renders one block of prose Markdown (`CellDoc::prose_before`/
/// `prose_after`, C2 §2.4) to HTML via `pulldown-cmark`, with every `${…}`
/// inline span (C2 §5.2, found via [`find_inline_exprs`] — not re-scanned)
/// replaced by a `<span data-span-id="{span_id_prefix}:{i}"></span>`
/// placeholder, `i` 0-indexed in document order — matching `ProseSpan.tsx`'s
/// `extractInlineSpans` numbering convention exactly, so an interim
/// consumer already keyed off that convention needs no new plumbing.
///
/// Raw HTML the author typed in `text` (a literal `<script>`, `<b>`, an
/// HTML comment — CommonMark's "raw HTML" grammar) is **escaped, not
/// passed through** (ledger R69's sandbox security boundary): every
/// `Event::Html`/`Event::InlineHtml` from the parse is turned into an
/// `Event::Text` carrying the same raw string *before* `push_html` runs.
/// `push_html`'s own `Text` handling already calls
/// `pulldown_cmark_escape::escape_html_body_text` on every `Event::Text`
/// (verified against the vendored 0.13.4 source, `html.rs`) — reusing that
/// existing escaping step, rather than hand-rolling a second one here, is
/// what actually turns `<script>` into visible `&lt;script&gt;` text
/// instead of live markup. (An earlier draft of this function additionally
/// hand-escaped the string before wrapping it in `Event::Text` — that
/// double-escapes, since `push_html` escapes `Event::Text` unconditionally;
/// caught by this module's own raw-HTML test, fixed by dropping the
/// redundant step rather than adding a second, non-standard escaper.)
///
/// `text` and `span_id_prefix` deterministically produce the same
/// [`RenderedProse::spans`] ids every call (ledger R70) — a fresh per-call
/// random nonce only insulates the internal sentinel substitution from
/// colliding with the prose's own text, it never reaches the returned ids.
pub fn render_prose_html(text: &str, span_id_prefix: &str) -> RenderedProse {
    // 1. Find spans with the existing, correct scanner -- do not rescan.
    let found = find_inline_exprs(text);

    // 2. Substitute each span's exact byte range with a plain-alphanumeric
    //    sentinel token, unique to this call, so the markdown parser sees
    //    ordinary text there (no markdown-special characters, no risk of
    //    the sentinel itself forming accidental emphasis/HTML syntax).
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let sentinel = |i: usize| format!("IDL1WBSPAN{nonce}_{i}");
    let mut with_sentinels = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (i, span) in found.iter().enumerate() {
        with_sentinels.push_str(&text[cursor..span.start]);
        with_sentinels.push_str(&sentinel(i));
        cursor = span.end;
    }
    with_sentinels.push_str(&text[cursor..]);

    // 3. Parse + render, neutralising raw HTML per the doc comment above.
    let events: Vec<Event> = Parser::new_ext(&with_sentinels, Options::empty())
        .map(|ev| match ev {
            // Reclassify raw HTML as plain text -- `push_html` escapes every
            // `Event::Text` unconditionally, so this alone is what neutralises
            // the markup (see this function's doc comment for why no separate
            // hand-rolled escape step is applied here).
            Event::Html(s) | Event::InlineHtml(s) => Event::Text(s),
            other => other,
        })
        .collect();
    let mut html_out = String::new();
    html::push_html(&mut html_out, events.into_iter());

    // 4. Swap each sentinel for its real placeholder element, in order.
    //    Sentinels contain no markdown/HTML-special characters, so they
    //    survive steps 2-3 byte-identical and this plain string replace is
    //    exact -- no risk of a partial/false match against real content.
    let mut spans = Vec::with_capacity(found.len());
    for (i, span) in found.iter().enumerate() {
        let id = format!("{span_id_prefix}:{i}");
        html_out = html_out.replace(&sentinel(i), &format!(r#"<span data-span-id="{id}"></span>"#));
        spans.push(ProseSpanRef { id, expr: span.js_expr.clone() });
    }

    RenderedProse { html: html_out, spans }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prose_html_heading_bold_and_two_spans_placeholders_in_document_order() {
        // Arrange
        let text = "# Title\n\nSome **bold ${a} text** and ${b}.\n";

        // Act
        let got = render_prose_html(text, "prefix");

        // Assert -- exact pulldown-cmark output isn't pinned; substring and
        // ordering assertions instead.
        assert!(got.html.contains("<h1>Title</h1>"));
        assert!(got.html.contains("<strong>bold"));
        let i0 = got.html.find(r#"data-span-id="prefix:0""#).unwrap();
        let i1 = got.html.find(r#"data-span-id="prefix:1""#).unwrap();
        assert!(i0 < i1);
        assert!(!got.html.contains("${"));
        assert_eq!(got.spans, vec![
            ProseSpanRef { id: "prefix:0".to_string(), expr: "a".to_string() },
            ProseSpanRef { id: "prefix:1".to_string(), expr: "b".to_string() },
        ]);
    }

    #[test]
    fn render_prose_html_raw_html_the_author_typed_is_escaped_not_passed_through() {
        // Arrange
        let text = "Some <b>bold</b> text and <!-- hi -->.";

        // Act
        let got = render_prose_html(text, "prefix");

        // Assert
        assert!(got.html.contains("&lt;b&gt;"));
        assert!(!got.html.contains("<b>"));
        assert!(got.html.contains("&lt;!-- hi --&gt;"));
    }

    #[test]
    fn render_prose_html_no_spans_no_placeholder_substitution_ordinary_rendering() {
        // Arrange
        let text = "Just plain prose, no expressions here.\n";

        // Act
        let got = render_prose_html(text, "prefix");

        // Assert
        assert!(got.html.contains("Just plain prose, no expressions here."));
        assert!(got.spans.is_empty());
        assert!(!got.html.contains("data-span-id"));
    }

    #[test]
    fn render_prose_html_an_inert_dollar_brace_inside_a_code_span_is_not_treated_as_a_placeholder() {
        // Arrange -- reuses find_inline_exprs's own already-tested inert-code
        // behaviour; this only confirms render_prose_html doesn't break it.
        let text = "Use `${not_an_expr}` here, but ${real(x)} is live.";

        // Act
        let got = render_prose_html(text, "prefix");

        // Assert
        assert_eq!(got.spans.len(), 1);
        assert_eq!(got.spans[0].expr, "real(x)");
        assert!(got.html.contains("${not_an_expr}"));
    }
}
