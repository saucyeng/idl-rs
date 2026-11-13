//! idl-library — the idl-rs ride library: the on-disk `<data>` directory
//! that holds every imported session. The content-addressed blob store, the
//! SQLite catalog (an index, deletable and rebuildable), import and index
//! jobs, the bike log, the LAN-sync model and `verify`. It depends on the
//! `idl-rs` engine for parsing, `data.parquet` and `session.json`; the engine
//! never depends on it (contracts C1 and C4). `std::fs` only: no Tauri, no
//! async runtime, no network (CLAUDE.md §2).

/// `<data>/AGENTS.md`, the orientation file an agent reads before it
/// touches the library (ruling R222 item 4).
pub mod agents_md;
pub mod bikelog;
pub mod blob;
pub mod catalog;
pub mod catalog_read;
pub mod derived;
pub mod eval_cache;
pub mod import;
pub mod index_job;
pub mod lap_index;
/// The read-only library summary served to paired peers (library tiers Phase 1).
pub mod library_summary;
pub mod profile;
pub mod quarantine;
pub mod scan;
pub mod session_context;
/// The filtering and metadata edits behind `session list`/`set-meta`.
pub mod session_ops;
pub mod set_job;
pub mod settings;
pub mod sync;
pub mod verify;
/// Linking MP4s to sessions (video spec §3–§4, C1 §6 `videos`).
pub mod video;

/// The engine's per-thread allocation counter measures the memory-shape
/// tests in `import`.
#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: idl_rs::counting_alloc::Counting = idl_rs::counting_alloc::Counting;
