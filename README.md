# idl-rs

The IDL0 signal-processing engine — a pure Rust workspace that owns all DSP,
`.idl0` binary parsing, the math-channel evaluator, and the suspension-kinematics
estimator. Consumed by the IDL0 app via `flutter_rust_bridge` and by the `idl-rs` CLI.

| Crate | What |
|-------|------|
| `core/` | `idl-rs` — filters, FFT, integration, rotation, statistics, estimation (sci-rs, nalgebra). Pure: no Flutter, no I/O beyond `std::fs`. |
| `bridge/` | `idl_rs_bridge` — thin `#[frb]` wrappers over `core`; the only crate Flutter sees. |
| `cli/` | `idl-rs-cli` — the standalone `idl-rs` binary. |

## Build & test

```
cargo test --workspace
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Contributions require the CLA
(see the app repo's `CLA.md`), which keeps commercial dual-licensing available.
