//! The command table rendered for readers and for machines: Markdown for
//! the generated reference, JSON for the app's checked-in copy.
//!
//! Both are pure functions of [`super::table::COMMANDS`], and both are
//! byte-stable for a given table — CI regenerates them and fails on a diff,
//! the same gate `docs workbook` already runs under. Every line ends `\n` on
//! every platform for that reason.

use serde_json::{json, Value};

use super::table::{
    CommandRow, Status, COMMANDS, DATA_DIR_ENV, JSON_SCHEMA_VERSION, NOUNS, VERBS, VERBS_RULED,
};

/// Renders the whole table as Markdown: the grammar, the vocabularies, the
/// uniform behaviour, then one section per noun.
pub fn render_command_reference() -> String {
    let mut out = String::new();

    out.push_str("# idl-rs command reference\n\n");
    out.push_str("Generated from the one command table (`idl_rs::commands::table`, ruling R230).\n");
    out.push_str("Do not edit by hand: run `idl-rs docs cli --out <file>`.\n\n");

    out.push_str("## Grammar\n\n");
    out.push_str("```\nidl-rs <noun> <verb> [args] [--flags]\n```\n\n");

    out.push_str("**Nouns.** ");
    out.push_str(&joined_code(NOUNS));
    out.push_str("\n\n**Verbs.** ");
    out.push_str(&joined_code(VERBS));
    out.push_str("\n\nVerbs added by a later ruling: ");
    let ruled: Vec<String> =
        VERBS_RULED.iter().map(|(verb, ruling)| format!("`{verb}` ({ruling})")).collect();
    out.push_str(&ruled.join(", "));
    out.push_str(".\n\n");

    out.push_str("## Uniform behaviour\n\n");
    out.push_str("- `--json` on every command. The payload carries ");
    out.push_str(&format!("`schema_version: {JSON_SCHEMA_VERSION}`"));
    out.push_str(", and a failure becomes a typed JSON error on stderr.\n");
    out.push_str("- `--data-dir` on every command that works against a data directory, falling back to `");
    out.push_str(DATA_DIR_ENV);
    out.push_str("`.\n");
    out.push_str("- `--dry-run` on every writer.\n");
    out.push_str("- Exit `0` on success, `1` on a failed operation, `2` on a usage error.\n\n");

    for noun in NOUNS {
        let rows: Vec<&CommandRow> = COMMANDS.iter().filter(|r| r.noun == *noun).collect();
        if rows.is_empty() {
            continue;
        }
        out.push_str(&format!("## `{noun}`\n\n"));
        for row in rows {
            out.push_str(&render_row(row));
        }
    }

    out
}

/// One command's section: usage, help, arguments, flags, and the core
/// function it calls.
fn render_row(row: &CommandRow) -> String {
    let mut out = String::new();

    out.push_str(&format!("### `idl-rs {}`\n\n", row.usage()));
    out.push_str(row.help);
    out.push_str(".\n\n");

    if row.status == Status::Deprecated {
        out.push_str("> Deprecated. Kept for one release; see the grammar above for the current spelling.\n\n");
    }

    if !row.args.is_empty() {
        out.push_str("| Argument | Type | Required | Meaning |\n| --- | --- | --- | --- |\n");
        for arg in row.args {
            out.push_str(&format!(
                "| `{}` | {} | {} | {} |\n",
                arg.name,
                arg.kind.as_str(),
                if arg.required { "yes" } else { "no" },
                arg.help
            ));
        }
        out.push('\n');
    }

    let flags = row.all_flags();
    out.push_str("| Flag | Value | Meaning |\n| --- | --- | --- |\n");
    for flag in &flags {
        let value = match flag.kind {
            None => "switch".to_string(),
            Some(kind) if flag.choices.is_empty() => kind.as_str().to_string(),
            Some(_) => joined_code(flag.choices),
        };
        let default = match flag.default {
            Some(d) => format!(" (default `{d}`)"),
            None => String::new(),
        };
        out.push_str(&format!("| `--{}` | {} | {}{} |\n", flag.long, value, flag.help, default));
    }
    out.push('\n');

    out.push_str(&format!("Calls `{}`", row.core_fn));
    match row.json_shape {
        Some(shape) => out.push_str(&format!("; `--json` emits `{shape}`.\n\n")),
        None => out.push_str("; writes an artifact rather than a payload.\n\n"),
    }

    out
}

/// `` `a`, `b`, `c` `` — the vocabularies and choice lists print the same way.
fn joined_code(items: &[&str]) -> String {
    items.iter().map(|i| format!("`{i}`")).collect::<Vec<_>>().join(", ")
}

/// The whole table as JSON — the shape `idl-rs docs cli --json` emits and
/// `app/src/shell/cliTable.json` holds, so the app's own command table can
/// be tested against it (R230 item 2).
pub fn command_table_json() -> Value {
    json!({
        "schema_version": JSON_SCHEMA_VERSION,
        "nouns": NOUNS,
        "verbs": VERBS,
        "verbs_ruled": VERBS_RULED
            .iter()
            .map(|(verb, ruling)| json!({ "verb": verb, "ruling": ruling }))
            .collect::<Vec<_>>(),
        "commands": COMMANDS.iter().map(row_json).collect::<Vec<_>>(),
    })
}

/// One row as JSON.
fn row_json(row: &CommandRow) -> Value {
    json!({
        "id": row.id(),
        "noun": row.noun,
        "verb": row.verb,
        "usage": row.usage(),
        "help": row.help,
        "tier": row.tier.as_str(),
        "status": row.status.as_str(),
        "core_fn": row.core_fn,
        "json_shape": row.json_shape,
        "data_dir": row.data_dir,
        "writer": row.writer,
        "args": row.args.iter().map(|arg| json!({
            "name": arg.name,
            "kind": arg.kind.as_str(),
            "required": arg.required,
            "repeatable": arg.repeatable,
            "choices": arg.choices,
            "help": arg.help,
        })).collect::<Vec<_>>(),
        "flags": row.all_flags().iter().map(|flag| json!({
            "long": flag.long,
            "kind": flag.kind.map(|k| k.as_str()),
            "repeatable": flag.repeatable,
            "choices": flag.choices,
            "default": flag.default,
            "help": flag.help,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::table::find;

    #[test]
    fn the_reference_names_every_registered_command() {
        // Arrange
        let rows = COMMANDS;

        // Act
        let rendered = render_command_reference();

        // Assert
        for row in rows {
            assert!(rendered.contains(&format!("### `idl-rs {}`\n", row.usage())), "{row} missing");
        }
    }

    #[test]
    fn the_reference_is_byte_stable_across_two_renders() {
        // Arrange
        let first = render_command_reference();

        // Act
        let second = render_command_reference();

        // Assert — CI's gate is `git diff --exit-code` over exactly this.
        assert_eq!(first, second);
    }

    #[test]
    fn the_reference_contains_no_carriage_returns() {
        // Arrange
        let rendered = render_command_reference();

        // Act
        let has_cr = rendered.contains('\r');

        // Assert — a Windows-generated file and a Linux-generated one must match.
        assert!(!has_cr);
    }

    #[test]
    fn the_reference_prints_the_uniform_flags_on_a_writer_that_takes_a_data_dir() {
        // Arrange
        let row = find("session", "import").unwrap();

        // Act
        let rendered = render_row(row);

        // Assert
        assert!(rendered.contains("| `--json` |"));
        assert!(rendered.contains("| `--data-dir` |"));
        assert!(rendered.contains("| `--dry-run` |"));
    }

    #[test]
    fn the_reference_omits_dry_run_from_a_read_only_command() {
        // Arrange
        let row = find("session", "list").unwrap();

        // Act
        let rendered = render_row(row);

        // Assert
        assert!(!rendered.contains("| `--dry-run` |"));
    }

    #[test]
    fn the_json_export_carries_one_entry_per_row_keyed_by_the_shared_id() {
        // Arrange
        let value = command_table_json();

        // Act
        let ids: Vec<&str> =
            value["commands"].as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap()).collect();

        // Assert
        assert_eq!(ids.len(), COMMANDS.len());
        assert!(ids.contains(&"workbook.new"));
        assert!(ids.contains(&"library.rebuild"));
    }

    #[test]
    fn the_json_export_states_its_schema_version() {
        // Arrange
        let value = command_table_json();

        // Act
        let version = value["schema_version"].as_u64().unwrap();

        // Assert
        assert_eq!(version, u64::from(JSON_SCHEMA_VERSION));
    }

    #[test]
    fn the_json_export_carries_the_uniform_flags_on_every_row() {
        // Arrange
        let value = command_table_json();

        // Act / Assert
        for command in value["commands"].as_array().unwrap() {
            let longs: Vec<&str> = command["flags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["long"].as_str().unwrap())
                .collect();
            assert!(longs.contains(&"json"), "{} has no --json", command["id"]);
        }
    }

    #[test]
    fn the_json_export_is_byte_stable_across_two_renders() {
        // Arrange
        let first = serde_json::to_string_pretty(&command_table_json()).unwrap();

        // Act
        let second = serde_json::to_string_pretty(&command_table_json()).unwrap();

        // Assert — the app's checked-in copy is regenerated by CI.
        assert_eq!(first, second);
    }
}
