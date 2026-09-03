# idl-rs

The IDL0 signal-processing engine — a pure Rust workspace that owns all DSP,
`.idl0` binary parsing, the math-channel evaluator, and the suspension-kinematics
estimator. Consumed by the idl1 app via `idl-rs-tauri` and by the `idl-rs` CLI.

| Crate | What |
|-------|------|
| `core/` | `idl-rs` — filters, FFT, integration, rotation, statistics, estimation (sci-rs, nalgebra). Pure: no Tauri, no I/O beyond `std::fs`. |
| `transport/` | `idl-transport` — BLE, WiFi transfer, config push, LAN sync. Never depends on Tauri; never does DSP. |
| `tauri/` | `idl-rs-tauri` — `#[tauri::command]` glue over `core` and `transport`; the only crate the frontend sees. |
| `cli/` | `idl-rs-cli` — the standalone `idl-rs` binary. |

## Build & test

```
cargo test --workspace
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Contributions require the CLA
(see the app repo's `CLA.md`), which keeps commercial dual-licensing available.
