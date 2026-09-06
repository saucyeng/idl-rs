//! LAN sync (contract C4 §6, design doc §7): the manifest model, the pure
//! walk of `<data>` that builds it, and the pure diff over two manifests.
//! No network, no async runtime — the wire transfer and the workbook/
//! `session.json` merge logic that consume a [`diff::SyncPlan`] live in
//! `idl-transport` and this lane's later core tasks (`workbook::merge`).
//! `std::fs` only (CLAUDE.md §2).

pub mod diff;
pub mod manifest;
