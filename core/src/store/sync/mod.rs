//! LAN sync (contract C4 §6, design doc §7): the manifest model and the
//! pure walk of `<data>` that builds it. No network, no async runtime — the
//! wire transfer, pairing and the diff/merge logic that consume this
//! document live in `idl-transport` and this lane's later core tasks
//! (`store::sync::diff`, `workbook::merge`). `std::fs` only (CLAUDE.md §2).

pub mod manifest;
