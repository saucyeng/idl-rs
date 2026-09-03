//! `idl-rs`'s storage layer: CAS blob store, atomic-write primitive,
//! `data.parquet`/`derived/<hash>.parquet` Arrow/Parquet r/w, `session.json`,
//! the SQLite catalog, `verify`, and bike-profile/app-settings persistence.
//! Contracts C1 (session schema) and C4 (data directory) fix everything this
//! module implements. Pure: `std::fs`/`std::path` only — no Tauri, no async,
//! no network (CLAUDE.md §2).

pub mod atomic;
pub mod blob;
pub mod parquet;
