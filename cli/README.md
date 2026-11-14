# idl-rs-cli

idl-rs-cli is a command line tool to use the idl-rs engine, designed for a user that just wants to process log files through the terminal.

## Install

```
cargo install idl-rs-cli
```

This installs the `idl-rs` command.

## Quick start

Generate a synthetic ride (with a ground-truth JSON beside it), import it into a data directory, and list what is there:

```
idl-rs session synth --out ride.idl0
idl-rs session import ride.idl0 --data-dir ./data
idl-rs session list --data-dir ./data
```

`.fit`, `.gpx` and `.csv` files import the same way. Each session becomes `sessions/<id>/data.parquet`, plain Parquet readable by DuckDB, Polars or pandas.

Commands follow `idl-rs <noun> <verb>`; `idl-rs --help` lists them, and the [CLI reference](https://github.com/saucyeng/idl-rs/blob/main/docs/CLI-REFERENCE.md) covers every command and flag.

Part of the [idl-rs repository](https://github.com/saucyeng/idl-rs). AGPL-3.0-or-later.
