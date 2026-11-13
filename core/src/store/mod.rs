//! `idl-rs`'s storage primitives: the atomic-write primitive,
//! `data.parquet`/`derived/<hash>.parquet` Arrow/Parquet r/w, and the
//! `session.json` schema. Contracts C1 (session schema) and C4 (data
//! directory) fix what these implement. The library manager (CAS blob store,
//! SQLite catalog, import and index jobs, bike log, sync model, `verify`)
//! lives in the `idl-library` crate, which depends on this one. Pure:
//! `std::fs`/`std::path` only — no Tauri, no async, no network (CLAUDE.md §2).

pub mod atomic;
pub mod engine_identity;
pub mod parquet;
pub mod session_json;
