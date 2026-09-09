//! The scipy-alignment migration's production-shaped fixture (plan §3.7,
//! `runs/2026-09-08/scipy-alignment-plan.md`, task 2). Exercises
//! [`crate::math::migrate_document`] against one `.idl1wb` document carrying
//! every surface the plan calls out, rather than a synthesised one-liner:
//! front matter with a `constants` map; a `math` cell with several
//! `def_line`s (one with a trailing `# label:` comment, one a `const`
//! line, one a comment line that mentions a retired name in prose, one a
//! call nested inside an operator expression); a `table` cell whose JSON
//! carries both a `columns[].template` and a per-cell `formula`; a `js`
//! cell; and a prose paragraph with an inline `${…}` span.

use super::*;

const FIXTURE: &str = concat!(
    "---\n",
    "id: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\n",
    "name: Fork tuning\n",
    "constants: { rider_mass_kg: 82 }\n",
    "version: 3\n",
    "---\n",
    "# Fork tuning \u{2014} Whistler, 2026-08-30\n",
    "\n",
    "```math id=a1b2c3d4\n",
    "# the old variance_time path is retired\n",
    "const k = 2\n",
    "delta = variance_time([LapTime])  # label: Lap delta\n",
    "scaled = k * variance_dist([LapTime]) + 1\n",
    "```\n",
    "\n",
    "```table id=b2c3d4e5\n",
    "{\"columns\":[{\"id\":\"c1\",\"name\":null,\"template\":\"variance_time([LapTime])\"}],",
    "\"rows\":[{\"id\":\"r1\",\"context\":null}],",
    "\"cells\":[[{\"formula\":\"variance_dist([LapTime])\",\"literal\":null,\"name\":null}]]}\n",
    "```\n",
    "\n",
    "```js id=e5f6a7b8\n",
    "Plot.plot({ marks: [Plot.lineY(channel(\"delta\"), { x: \"t\", y: \"v\" })] })\n",
    "```\n",
    "\n",
    "Peak delta this lap: ${delta.v.length}\n",
);

#[test]
fn migrate_document_rewrites_both_expression_bearing_surfaces_of_the_production_fixture() {
    // Act
    let (out, renames) = crate::math::migrate_document(FIXTURE);

    // Assert — every call site in the math cell and the table cell moved.
    assert!(out.contains("delta = lap_delta_time([LapTime])  # label: Lap delta"));
    assert!(out.contains("scaled = k * lap_delta_dist([LapTime]) + 1"));
    assert!(out.contains("\"template\":\"lap_delta_time([LapTime])\""));
    assert!(out.contains("\"formula\":\"lap_delta_dist([LapTime])\""));
    assert_eq!(renames.len(), 4);
}

#[test]
fn migrate_document_leaves_the_prose_comment_and_js_cell_byte_identical() {
    // Act
    let (out, _renames) = crate::math::migrate_document(FIXTURE);

    // Assert — a comment mentioning the retired name in English, and the
    // `js` cell (R145: never a textual-rewrite surface), are untouched.
    assert!(out.contains("# the old variance_time path is retired"));
    assert!(out.contains("Plot.plot({ marks: [Plot.lineY(channel(\"delta\"), { x: \"t\", y: \"v\" })] })"));
}

#[test]
fn migrate_document_leaves_the_const_line_and_label_and_inline_span_byte_identical() {
    // Act
    let (out, _renames) = crate::math::migrate_document(FIXTURE);

    // Assert
    assert!(out.contains("const k = 2"));
    assert!(out.contains("# label: Lap delta"));
    assert!(out.contains("Peak delta this lap: ${delta.v.length}"));
}

#[test]
fn migrate_document_bumps_version_and_the_result_still_parses_as_a_v4_workbook() {
    // Act
    let (out, _renames) = crate::math::migrate_document(FIXTURE);
    let (doc, errors) = parse_workbook(&out).unwrap();

    // Assert
    assert!(out.contains("version: 4"));
    assert_eq!(doc.version, 4);
    assert!(errors.is_empty());
}

#[test]
fn migrate_document_on_the_fixture_a_second_time_reports_nothing() {
    // Arrange
    let (once, _) = crate::math::migrate_document(FIXTURE);

    // Act
    let (twice, renames) = crate::math::migrate_document(&once);

    // Assert
    assert_eq!(twice, once);
    assert!(renames.is_empty());
}

#[test]
fn a_v3_fixture_evaluates_using_the_new_dispatch_without_being_saved() {
    // Arrange — the fixture is still on disk (in this test, in memory) as
    // `version: 3` with retired names; parse_workbook must migrate it before
    // handing definitions to the resolver/evaluator (R151 item 9).

    // Act
    let (doc, errors) = parse_workbook(FIXTURE).unwrap();

    // Assert — every def's expr_text already carries the current spelling.
    assert!(errors.is_empty());
    let delta = doc.defs.iter().find(|d| d.name == "delta").unwrap();
    let scaled = doc.defs.iter().find(|d| d.name == "scaled").unwrap();
    assert_eq!(delta.expr_text, "lap_delta_time([LapTime])");
    assert_eq!(scaled.expr_text, "k * lap_delta_dist([LapTime]) + 1");
}
