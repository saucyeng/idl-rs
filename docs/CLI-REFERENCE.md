# idl-rs command reference

Generated from the one command table (`idl_rs::commands::table`, ruling R230).
Do not edit by hand: run `idl-rs docs cli --out <file>`.

## Grammar

```
idl-rs <noun> <verb> [args] [--flags]
```

**Nouns.** `session`, `workbook`, `track`, `bike`, `catalog`, `library`, `device`, `docs`

**Verbs.** `list`, `show`, `new`, `check`, `eval`, `set`, `import`, `export`, `scan`, `fold-in`, `index`, `rebuild`, `verify`, `delete`

Verbs added by a later ruling: `set-start` (R229), `set-meta` (R229), `cells` (R229), `data` (R229), `detect` (R229), `laps` (R229), `stale` (R197), `workbook` (R222), `cli` (R230 item 2), `synth` (R187), `calibrate` (R238), `note` (R251).

## Uniform behaviour

- `--json` on every command. The payload carries `schema_version: 1`, and a failure becomes a typed JSON error on stderr.
- `--data-dir` on every command that works against a data directory, falling back to `IDL1_DATA_DIR`.
- `--dry-run` on every writer.
- Exit `0` on success, `1` on a failed operation, `2` on a usage error.

## `session`

### `idl-rs session list`

List catalogued sessions, most recent first.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--venue` | text | Only sessions whose venue matches, case-insensitively |
| `--track` | text | Only sessions that visited this track id |
| `--since` | integer | Only sessions starting at or after this Unix epoch millisecond |
| `--until` | integer | Only sessions starting at or before this Unix epoch millisecond |
| `--tag` | text | Only sessions carrying this tag, case-insensitively |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog_read::list_sessions + idl_library::session_ops::filter_sessions`; `--json` emits `SessionSummary[]`.

### `idl-rs session show <id>`

Print one session's metadata, channels and laps.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | text | yes | Session id, as `session list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog_read::get_session`; `--json` emits `SessionDetail`.

### `idl-rs session laps <id>`

Print one session's indexed laps.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | text | yes | Session id, as `session list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog_read::list_laps`; `--json` emits `LapSummary[]`.

### `idl-rs session set-start <id> <utc-ms>`

Set a session's recording start time by hand.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | text | yes | Session id, as `session list` prints it |
| `utc-ms` | integer | yes | Recording start, Unix epoch milliseconds; must be greater than zero |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `store::session_json::set_session_start`; `--json` emits `SessionJson`.

### `idl-rs session set-meta <id>`

Set a session's descriptive metadata; omitted fields are left alone.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | text | yes | Session id, as `session list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--venue` | text | Venue name |
| `--rider` | text | Rider name |
| `--bike` | text | Bike name |
| `--event` | text | Event name |
| `--event-session` | text | Event session, e.g. `Qualifying 2` |
| `--notes` | text | Long comment |
| `--tag` | text | Tag |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::session_ops::apply_session_meta`; `--json` emits `SessionJson`.

### `idl-rs session import <file>`

Import one log file into the data directory.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Log file to import (`.idl0`, `.fit`, `.gpx`, `.csv`) |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::import::import_idl0_path (.idl0) or import_file_path (.fit/.gpx/.csv)`; `--json` emits `ImportReport`.

### `idl-rs session synth`

Generate a synthetic .idl0 session with a ground-truth JSON beside it (ruling R187).

| Flag | Value | Meaning |
| --- | --- | --- |
| `--out` | path | File to write the `.idl0` log to; the truth JSON goes beside it |
| `--laps` | integer | Complete circuits of the loop (default `3`) |
| `--lap-length-m` | integer | Loop perimeter in metres (default `400`) |
| `--rate-hz` | integer | IMU output data rate in Hz (default `800`) |
| `--gps-hz` | integer | GPS fix rate in Hz (default `5`) |
| `--seed` | integer | PRNG seed; the same seed and flags give byte-identical output (default `1`) |
| `--imu-count` | integer | How many of the three sensors to instrument (1-3) (default `3`) |
| `--noise` | text | Multiplies every sensor noise sigma; 0 gives a noiseless recording (default `1.0`) |
| `--protocol` | `loop`, `calibration` | Motion to simulate: `loop` rides a circuit, `calibration` is the two-body rigid-body manoeuvre (default `loop`) |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `synth::generate`; `--json` emits `SynthReport`.

### `idl-rs session calibrate <file>`

Fit the rigid-body IMU calibration to a held-in-the-air session (ruling R238).

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | `.idl0` log of the calibration manoeuvre to fit |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--truth` | path | Ground-truth JSON from `session synth`; adds the error against it to the report |
| `--imu` | integer | Fit only these IMU indices (repeat for several); default every complete one |
| `--rigid` | integer | Treat this IMU as rigid with the frame (a fork pod over a window with the bars still) |
| `--from-ms` | integer | Start of the window to fit, ms from the first common sample |
| `--to-ms` | integer | End of the window to fit, ms from the first common sample |
| `--min-rate-mrad-s` | integer | Lower the tumble-rate gate (default 2000 mrad/s) to study a slower tumble |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `calibration::rigid::calibrate`; `--json` emits `CalibrationReport`.

## `workbook`

### `idl-rs workbook new <file>`

Create a new workbook from a built-in skeleton.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Workbook file to create; refuses to overwrite an existing one |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--template` | `blank`, `session` | Skeleton to write: `blank` is front matter only, `session` adds a math and a table cell (default `blank`) |
| `--session` | text | Session id to reference from the `session` template |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `commands::workbook_ops::new_workbook`; `--json` emits `WorkbookNewReport`.

### `idl-rs workbook check <file>`

Parse a workbook and report every structural problem.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Path to an `.idl0wb` workbook |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--session` | path | Session to evaluate against: an `.idl0` file, or a session id inside --data-dir (its data.parquet, laps from session.json); repeat it to evaluate the workbook's session tables over those sessions |
| `--track` | path | `.idl0t` track artifact, required only when a cell is lap-bound |
| `--main-lap` | integer | 1-based lap number that lap-scoped expressions mean; needs --track |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `workbook::v3::parse_workbook`; `--json` emits `WorkbookCheckReport`.

### `idl-rs workbook cells <file>`

List a workbook's cells in document order.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Path to an `.idl0wb` workbook |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `workbook::v3::parse_workbook`; `--json` emits `WorkbookCellSummary[]`.

### `idl-rs workbook eval <file>`

Evaluate a workbook against a session, or its session tables over its `sessions` set.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Path to an `.idl0wb` workbook |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--session` | path | Session to evaluate against: an `.idl0` file, or a session id inside --data-dir (its data.parquet, laps from session.json); repeat it to evaluate the workbook's session tables over those sessions |
| `--track` | path | `.idl0t` track artifact, required only when a cell is lap-bound |
| `--main-lap` | integer | 1-based lap number that lap-scoped expressions mean; needs --track |
| `--values` | switch | Print every reduced (rank-0) and per-lap result as `name  value  unit` |
| `--no-cache` | switch | Do not read or write the session-set result cache (writes nothing to the data directory) |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `workbook::v3::eval_cells`; `--json` emits `WorkbookEvalReport`.

### `idl-rs workbook data <file> <cell>`

Print one cell's evaluated series or table.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Path to an `.idl0wb` workbook |
| `cell` | text | yes | Cell id, as `workbook cells` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--session` | path | Session to evaluate against: an `.idl0` file, or a session id inside --data-dir (its data.parquet, laps from session.json); repeat it to evaluate the workbook's session tables over those sessions |
| `--track` | path | `.idl0t` track artifact, required only when a cell is lap-bound |
| `--main-lap` | integer | 1-based lap number that lap-scoped expressions mean; needs --track |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `workbook::v3::eval_cells`; `--json` emits `WorkbookCellData`.

### `idl-rs workbook export <file>`

Write a workbook out as Markdown or as evaluated JSON.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | path | yes | Path to an `.idl0wb` workbook |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--out` | path | File to write; stdout when omitted |
| `--format` | `md`, `json` | `md` re-renders the workbook source, `json` writes its evaluated cells (default `md`) |
| `--session` | path | Session to evaluate against: an `.idl0` file, or a session id inside --data-dir (its data.parquet, laps from session.json); repeat it to evaluate the workbook's session tables over those sessions |
| `--track` | path | `.idl0t` track artifact, required only when a cell is lap-bound |
| `--main-lap` | integer | 1-based lap number that lap-scoped expressions mean; needs --track |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `workbook::v3::render_workbook`; writes an artifact rather than a payload.

## `track`

### `idl-rs track list`

List the tracks in the library.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog_read::list_tracks`; `--json` emits `TrackSummary[]`.

### `idl-rs track detect <id>`

Re-detect track visits and laps for one session.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | text | yes | Session id, as `session list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::lap_index::index_session_laps`; `--json` emits `LapIndexReport`.

## `bike`

### `idl-rs bike list`

List the bikes in the library.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::bikelog::load_bikes`; `--json` emits `Bike[]`.

### `idl-rs bike show <bike_id>`

Show a bike's log and its setup sheet.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `bike_id` | text | yes | Bike id, as `bike list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--at` | integer | Work the setup sheet out as of this Unix epoch millisecond; default now |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::bikelog::running_sheet`; `--json` emits `BikeShow`.

### `idl-rs bike note <bike_id>`

Add a note to a bike's log; its changes update the setup sheet.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `bike_id` | text | yes | Bike id, as `bike list` prints it |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--text` | text | The note, in the rider's words |
| `--change` | text | A changed setup value as `key=value`, e.g. `fork.lsc=9`; repeat for several |
| `--at` | integer | When it happened, Unix epoch milliseconds; default now |
| `--device` | text | Who wrote it, e.g. `agent`; default `cli` |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::bikelog::save_entry`; `--json` emits `Entry`.

## `catalog`

### `idl-rs catalog verify`

Run contract C4 §7's checks against the data directory.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::verify::verify`; `--json` emits `Finding[]`.

### `idl-rs catalog rebuild`

Rebuild the catalog index from the canonical files.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog::rebuild_catalog`; `--json` emits `RebuildReport`.

## `library`

### `idl-rs library fold-in <folder>`

Import every importable file in a folder.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `folder` | path | yes | Folder to fold into the library |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--move` | switch | Delete each source file once its blob has verified in the store |
| `--recursive` | switch | Descend into sub-directories |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::import::import_file_path (per file)`; `--json` emits `FoldInReport`.

### `idl-rs library scan <folder>`

Preview what folding a folder in would import.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `folder` | path | yes | Folder to scan |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--recursive` | switch | Descend into sub-directories |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::scan::scan_folder_with_blob_check`; `--json` emits `ScanReport`.

### `idl-rs library stale`

List sessions whose data.parquet predates this importer version.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::catalog_read::list_stale_sessions`; `--json` emits `StaleSession[]`.

### `idl-rs library index [sessions]`

Detect track visits and laps for sessions whose index is missing or stale.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `sessions` | text | no | Session ids to index; with none, every session whose index is stale |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--all` | switch | Consider every session, not only the stale ones |
| `--force` | switch | Recompute even where the stamps are already current |
| `--workers` | integer | Pool width; defaults to physical cores minus one |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::index_job::run_index_job`; `--json` emits `IndexJobReport`.

### `idl-rs library rebuild [sessions]`

Re-import listed sessions from their own blobs.

| Argument | Type | Required | Meaning |
| --- | --- | --- | --- |
| `sessions` | text | no | Session ids to rebuild; mutually exclusive with --all |

| Flag | Value | Meaning |
| --- | --- | --- |
| `--all` | switch | Rebuild every session `library stale` lists |
| `--data-dir` | path | Data directory root (contract C4 §1's <data>); defaults to $IDL1_DATA_DIR |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `idl_library::import::reimport_session`; `--json` emits `RebuildSessionsReport`.

## `docs`

### `idl-rs docs workbook`

Regenerate the workbook reference from the engine's own catalogs, or its builtin catalog as JSON with --json (ruling R249).

| Flag | Value | Meaning |
| --- | --- | --- |
| `--out` | path | File to write; overwritten wholesale |
| `--src` | path | Directory of curated Markdown sections |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `docs::render_workbook_reference`; writes an artifact rather than a payload.

### `idl-rs docs cli`

Emit this command table as Markdown, or as JSON with --json.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--out` | path | File to write; stdout when omitted |
| `--dry-run` | switch | Report what would change and write nothing |
| `--json` | switch | Emit the result as JSON on stdout (schema_version 1); errors become typed JSON on stderr |

Calls `commands::markdown::render_command_reference`; `--json` emits `CommandTable`.

