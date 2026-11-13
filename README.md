# idl-rs

<!-- ISAAC: one or two sentences: what idl-rs is, in your words -->

## Why

<!-- ISAAC: why does it exist, and who is it for? A few sentences in your words. -->

## Quick start

Install the `idl-rs` command-line tool from a checkout (publishing to crates.io is coming):

```
cargo install --path cli
```

Generate a synthetic ride (with a ground-truth JSON beside it), import it into a data directory, where it becomes `sessions/<id>/data.parquet`, and list what is there:

```
idl-rs session synth --out ride.idl0
idl-rs session import ride.idl0 --data-dir ./data
idl-rs session list --data-dir ./data
```

`.fit`, `.gpx` and `.csv` files import the same way. The `data.parquet` files are plain Parquet, readable by DuckDB, Polars or pandas.

Two runnable library examples use the committed synthetic ride `core/tests/fixtures/synth-3lap.idl0`:

```
cargo run -p idl-rs --example parse_to_parquet
cargo run -p idl-rs --example math_channel
```

`parse_to_parquet` parses the log and writes `data.parquet`; `math_channel` low-pass filters an accelerometer channel with a Butterworth filter and takes its RMS.

## Crates

| Crate | Directory | What |
|-------|-----------|------|
| `idl-rs` | `core/` | Filters, FFT, integration, rotation, statistics, `.idl0` parsing, importers, math-channel evaluator, Parquet store, suspension-kinematics estimation. No Tauri, no network; file I/O through `std::fs` only. |
| `idl-library` | `library/` | Ride library manager: SQLite catalog, content-addressed blob store, import and index jobs, bike log, LAN-sync model. Depends on `idl-rs`. |
| `idl-rs-cli` | `cli/` | The `idl-rs` binary, grammar `idl-rs <noun> <verb>`. Depends on `idl-rs` and `idl-library`. |

## Documentation

- [Workbook reference](docs/WORKBOOK-REFERENCE.md): every math builtin, annotation, host variable and the chart grammar.
- [CLI reference](docs/CLI-REFERENCE.md): every `idl-rs` command and flag.
- [Signal pipeline](docs/signal_pipeline.md): what happens to a sample between the sensor and a result.

## Build and test

```
cargo test -p idl-rs -p idl-library -p idl-rs-cli
```

Never `--workspace`. The two reference files above are generated; see `docs/reference-src/` for the hand-written parts.

## License

AGPL-3.0-or-later; see [LICENSE](LICENSE). Contributions require the CLA ([CLA.md](CLA.md)), which keeps commercial dual-licensing available.

<!-- ISAAC: one sentence on commercial licensing: who to contact, and what it covers? -->
