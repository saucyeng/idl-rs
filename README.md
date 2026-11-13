# idl-rs

The IDL0 signal-processing engine — a pure Rust workspace that owns all DSP,
`.idl0` binary parsing, the math-channel evaluator, and the suspension-kinematics
estimator. Consumed by the idl1 app via `idl1-bridge` and by the `idl-rs` CLI.

| Crate | What |
|-------|------|
| `core/` | `idl-rs` — filters, FFT, integration, rotation, statistics, estimation (sci-rs, nalgebra). Pure: no Tauri, no I/O beyond `std::fs`. |
| `library/` | `idl-library` — the ride library manager: SQLite catalog, content-addressed blob store, import and index jobs, bike log, LAN-sync model. Depends on `idl-rs`; the engine never depends on it. |
| `cli/` | `idl-rs-cli` — the standalone `idl-rs` binary. |

The app-side crates (`idl1-bridge`, the `#[tauri::command]` glue, and `idl1-transport`, BLE/WiFi/LAN sync) now live in the idl1-app repo under `crates/`.

## Build & test

```
cargo test --workspace
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Contributions require the CLA
(see the app repo's `CLA.md`), which keeps commercial dual-licensing available.
