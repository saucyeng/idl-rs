# idl-rs

idl-rs is a cross-platform Rust library and command-line tool for processing data logs, built for MTB and motocross racing. It parses, converts, and processes time series data through `nalgebra`, `sci-rs`, and `rustfft`, and stores results in Parquet so long sessions stay compact and fast to query.

## Why

While tools like NumPy/SciPy, MATLAB, and Julia have been the go-to in scientific computing for years, the mobile options are few and far between. Until now, there was no complete, high-level pipeline that ran the same on mobile as on desktop.

idl-rs is an FFmpeg-inspired, cross-platform tool that puts a full toolbox of scientific computing capabilities on a racer's or mechanic's phone for trackside analysis.

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
| `idl-rs` | `core/` | Filters, FFT, integration, rotation, statistics, `.idl0` parsing, importers, math-channel evaluator, Parquet store, suspension-kinematics estimation. No network; file I/O through `std::fs` only. |
| `idl-library` | `library/` | Ride library manager: SQLite catalog, content-addressed blob store, import and index jobs, bike log, LAN-sync model. Depends on `idl-rs`. |
| `idl-rs-cli` | `cli/` | The `idl-rs` binary, grammar `idl-rs <noun> <verb>`. Depends on `idl-rs` and `idl-library`. |

## Documentation

- [Workbook reference](docs/WORKBOOK-REFERENCE.md): every math builtin, annotation, host variable and the chart grammar.
- [CLI reference](docs/CLI-REFERENCE.md): every `idl-rs` command and flag.
- [Signal pipeline](docs/signal_pipeline.md): what happens to a sample between the sensor and a result.

## Build and test

```
cargo test
```

The two reference files above are generated; see `docs/reference-src/` for the hand-written parts.

## License

AGPL-3.0-or-later; see [LICENSE](LICENSE). Contributions require the CLA ([CLA.md](CLA.md)), which keeps commercial dual-licensing available.

idl-rs is available for free under an AGPL-3.0 license, meaning you may use it, modify it, and distribute it as you please, as long as your project remains open source. Commercial licenses are available for purchase if you would like to use this pipeline in a closed source project. Please reach out to isaac@marlboroelectronics.com.
