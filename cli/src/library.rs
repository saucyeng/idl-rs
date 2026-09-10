//! The `library` command group (ruling R197 item 1): bulk folder management
//! for a C4 §2 data directory from the shell — `fold-in`, `scan`, `stale`,
//! `rebuild`.
//!
//! Every action is a wrapper over the same `idl-rs` core functions the app's
//! C3 commands call (`store::scan::scan_folder`, `store::import::{import_idl0,
//! import_file, reimport_session}`, `store::catalog_read::list_stale_sessions`),
//! so the CLI and the app can never drift. No import, hashing or staleness
//! logic lives here: this module walks directories, formats tables, and
//! counts outcomes.
//!
//! `--move` deletes a source file only after its blob's sha256 has verified
//! in the store (R197) — never a rename, so it behaves identically within a
//! volume and across volumes.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Subcommand;
use serde_json::{json, Value};

use idl_rs::store::atomic::sha256_hex;
use idl_rs::store::blob::blob_exists;
use idl_rs::store::catalog::{self, CatalogError};
use idl_rs::store::catalog_read::{list_stale_sessions, StaleSession};
use idl_rs::store::import::{self, ImportError, ImportOutcome};
use idl_rs::store::scan::{scan_folder, ScanEntry, ScanError};

/// The `library` sub-actions.
#[derive(Subcommand)]
pub enum LibraryAction {
    /// Import every importable file in a folder into the data directory,
    /// optionally moving (not copying) each source file once its blob has
    /// verified in the store.
    FoldIn {
        /// Folder to fold into the library.
        folder: PathBuf,
        /// Data directory root (contract C4 §1's `<data>`).
        #[arg(long)]
        data_dir: PathBuf,
        /// Delete each source file after its blob's sha256 has verified in
        /// the store. A file that fails to import is never deleted.
        #[arg(long)]
        r#move: bool,
        /// Descend into sub-directories (core's scan is per-folder; this
        /// walks the tree and scans each folder it finds).
        #[arg(long)]
        recursive: bool,
        /// Print the preview table and exit without importing anything.
        #[arg(long)]
        dry_run: bool,
        /// Emit one JSON object instead of the human tables.
        #[arg(long)]
        json: bool,
    },
    /// Print the preview table for a folder — no import, no catalog write.
    Scan {
        /// Folder to scan.
        folder: PathBuf,
        /// Data directory root (contract C4 §1's `<data>`), whose `blobs/`
        /// decides the already-imported column.
        #[arg(long)]
        data_dir: PathBuf,
        /// Descend into sub-directories.
        #[arg(long)]
        recursive: bool,
        /// Emit one JSON object instead of the human table.
        #[arg(long)]
        json: bool,
    },
    /// List every catalogued session whose `data.parquet` was written by a
    /// different importer version than this build runs.
    Stale {
        /// Data directory root (contract C4 §1's `<data>`).
        #[arg(long)]
        data_dir: PathBuf,
        /// Emit one JSON object instead of the human table.
        #[arg(long)]
        json: bool,
    },
    /// Re-import listed sessions from their own blobs (`reimport_session`).
    Rebuild {
        /// Session ids to rebuild. Mutually exclusive with `--all`.
        session_ids: Vec<String>,
        /// Data directory root (contract C4 §1's `<data>`).
        #[arg(long)]
        data_dir: PathBuf,
        /// Rebuild every session `library stale` lists, instead of naming ids.
        #[arg(long, conflicts_with = "session_ids")]
        all: bool,
        /// Emit one JSON object instead of the human progress lines.
        #[arg(long)]
        json: bool,
    },
}

/// Everything that can go wrong in this module, typed (CLAUDE.md §5) —
/// the wrapped core errors plus this module's own filesystem walk.
#[derive(Debug)]
pub enum LibraryError {
    /// [`scan_folder`] failed for the folder named.
    Scan(ScanError),
    /// A catalog open or query failed.
    Catalog(CatalogError),
    /// Walking the folder tree failed (`--recursive` only).
    Walk { path: PathBuf, message: String },
    /// `--all` was given with no catalog to read, or the arguments name no
    /// session at all.
    NoSessions,
}

impl fmt::Display for LibraryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LibraryError::Scan(e) => write!(f, "{e}"),
            LibraryError::Catalog(e) => write!(f, "{e}"),
            LibraryError::Walk { path, message } => write!(f, "walking {}: {message}", path.display()),
            LibraryError::NoSessions => write!(f, "no sessions named: pass session ids or --all"),
        }
    }
}

impl std::error::Error for LibraryError {}

impl From<ScanError> for LibraryError {
    fn from(e: ScanError) -> Self {
        LibraryError::Scan(e)
    }
}

impl From<CatalogError> for LibraryError {
    fn from(e: CatalogError) -> Self {
        LibraryError::Catalog(e)
    }
}

/// One source file's fate in a [`fold_in`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldOutcome {
    /// The importer wrote (or regenerated) this session's `data.parquet`.
    Imported,
    /// The blob and `data.parquet` were already current — nothing written.
    Skipped,
    /// No importer covers this file's extension; it was never opened.
    NoImporter,
    /// Reading or importing the file failed; the message is the typed core
    /// error's `Display`.
    Failed,
}

/// What happened to one source file, in scan order.
#[derive(Debug, Clone)]
pub struct FoldResult {
    pub path: PathBuf,
    pub outcome: FoldOutcome,
    /// The core error's `Display` when `outcome` is [`FoldOutcome::Failed`].
    pub error: Option<String>,
    /// `true` when `--move` deleted the source after its blob verified.
    pub source_removed: bool,
    /// Non-fatal advisories the importer raised for this file.
    pub warnings: Vec<String>,
}

/// Counts for the summary line, plus the per-file results behind them.
#[derive(Debug, Clone)]
pub struct FoldSummary {
    pub results: Vec<FoldResult>,
    pub imported: usize,
    pub skipped: usize,
    pub failed: usize,
    /// Files whose extension no importer covers — listed, never imported.
    pub no_importer: usize,
    /// Scanned files with no readable session start (`unknown` in the table).
    pub unknown_start: usize,
    pub moved: usize,
}

/// One session's fate in a [`rebuild`] run.
#[derive(Debug, Clone)]
pub struct RebuildResult {
    pub session_id: String,
    /// The core error's `Display` when the rebuild failed.
    pub error: Option<String>,
    /// What `reimport_session` did, when it succeeded.
    pub outcome: Option<ImportOutcome>,
}

/// Dispatches one `library` action, printing to stdout/stderr and returning
/// the process exit code: failure if any file or session failed.
pub fn run(action: LibraryAction) -> ExitCode {
    match action {
        LibraryAction::Scan { folder, data_dir, recursive, json } => {
            match collect_entries(&data_dir, &folder, recursive) {
                Ok(entries) => {
                    if json {
                        print_json(&json!({ "scan": entries_json(&entries) }))
                    } else {
                        print_scan_table(&entries);
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => fail(&e),
            }
        }
        LibraryAction::FoldIn { folder, data_dir, r#move, recursive, dry_run, json } => {
            let entries = match collect_entries(&data_dir, &folder, recursive) {
                Ok(entries) => entries,
                Err(e) => return fail(&e),
            };
            if dry_run {
                return if json {
                    print_json(&json!({ "fold_in": { "dry_run": true, "scan": entries_json(&entries) } }))
                } else {
                    print_scan_table(&entries);
                    println!("dry run: nothing imported");
                    ExitCode::SUCCESS
                };
            }
            if !json {
                print_scan_table(&entries);
            }
            let summary = fold_in(&data_dir, &entries, r#move, |result| {
                if !json {
                    println!("{}", fold_line(result));
                }
            });
            // One catalog rebuild for the whole run, not one per file: the
            // catalog is an index, deletable and rebuildable (CLAUDE.md §3).
            let catalog_error = catalog::rebuild_catalog(&data_dir).err().map(|e| e.to_string());
            if json {
                let mut object = fold_summary_json(&summary);
                object["catalog_error"] = catalog_error.clone().map_or(Value::Null, Value::String);
                let code = print_json(&json!({ "fold_in": object }));
                if summary.failed > 0 || catalog_error.is_some() {
                    return ExitCode::FAILURE;
                }
                code
            } else {
                print_fold_summary(&summary);
                if let Some(e) = &catalog_error {
                    eprintln!("error: catalog rebuild: {e}");
                    return ExitCode::FAILURE;
                }
                if summary.failed > 0 {
                    ExitCode::FAILURE
                } else {
                    ExitCode::SUCCESS
                }
            }
        }
        LibraryAction::Stale { data_dir, json } => match stale(&data_dir) {
            Ok(rows) => {
                if json {
                    print_json(&json!({ "stale": stale_json(&rows) }))
                } else {
                    print_stale_table(&rows);
                    ExitCode::SUCCESS
                }
            }
            Err(e) => fail(&e),
        },
        LibraryAction::Rebuild { session_ids, data_dir, all, json } => {
            let ids = if all {
                match stale(&data_dir) {
                    Ok(rows) => rows.into_iter().map(|r| r.session_id).collect(),
                    Err(e) => return fail(&e),
                }
            } else {
                session_ids
            };
            if ids.is_empty() && !all {
                return fail(&LibraryError::NoSessions);
            }
            let results = rebuild(&data_dir, &ids, |result| {
                if !json {
                    println!("{}", rebuild_line(result));
                }
            });
            let failed = results.iter().filter(|r| r.error.is_some()).count();
            if json {
                let code = print_json(&json!({ "rebuild": rebuild_json(&results) }));
                if failed > 0 {
                    return ExitCode::FAILURE;
                }
                code
            } else {
                println!("rebuilt {} session(s), {failed} failed", results.len() - failed);
                if failed > 0 {
                    ExitCode::FAILURE
                } else {
                    ExitCode::SUCCESS
                }
            }
        }
    }
}

/// Prints a typed error and returns the failure exit code.
fn fail(error: &dyn std::error::Error) -> ExitCode {
    eprintln!("error: {error}");
    ExitCode::FAILURE
}

/// Serialises one JSON object to stdout.
fn print_json(value: &Value) -> ExitCode {
    match serde_json::to_string_pretty(value) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── scan ─────────────────────────────────────────────────────────────────────

/// [`scan_folder`] over one folder, or over the folder and every
/// sub-directory beneath it when `recursive`. The recursion is this module's
/// only addition to core's scan: it walks the tree and calls `scan_folder`
/// per directory, so extension→importer mapping, the blob-digest
/// already-imported test and the header peek all still happen in core.
/// Ordered by path for a deterministic preview.
pub fn collect_entries(data_root: &Path, folder: &Path, recursive: bool) -> Result<Vec<ScanEntry>, LibraryError> {
    let mut folders = vec![folder.to_path_buf()];
    if recursive {
        collect_subdirectories(folder, &mut folders)?;
    }

    let mut entries = Vec::new();
    for dir in &folders {
        entries.extend(scan_folder(data_root, dir)?);
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

/// Appends every directory beneath `folder`, depth-first, to `out`.
fn collect_subdirectories(folder: &Path, out: &mut Vec<PathBuf>) -> Result<(), LibraryError> {
    let dir = std::fs::read_dir(folder)
        .map_err(|e| LibraryError::Walk { path: folder.to_path_buf(), message: e.to_string() })?;
    let mut children = Vec::new();
    for item in dir {
        let item = item.map_err(|e| LibraryError::Walk { path: folder.to_path_buf(), message: e.to_string() })?;
        let path = item.path();
        if path.is_dir() {
            children.push(path);
        }
    }
    children.sort();
    for child in children {
        collect_subdirectories(&child, out)?;
        out.push(child);
    }
    Ok(())
}

/// `true` when this entry's session start is missing or has no clock value
/// (`0` — the header carried no time), i.e. the table shows `unknown`.
fn start_unknown(entry: &ScanEntry) -> bool {
    !matches!(entry.session_start_utc_ms, Some(ms) if ms != 0)
}

/// The preview table (file, importer, size, already-imported, start time).
fn print_scan_table(entries: &[ScanEntry]) {
    println!("{:<40}  {:<8}  {:>12}  {:<9}  {}", "FILE", "IMPORTER", "SIZE", "IMPORTED", "START (UTC ms)");
    for e in entries {
        let start = if start_unknown(e) { "unknown".to_string() } else { e.session_start_utc_ms.unwrap_or(0).to_string() };
        println!(
            "{:<40}  {:<8}  {:>12}  {:<9}  {}",
            e.file_name,
            e.importer_id.as_deref().unwrap_or("-"),
            e.size_bytes,
            if e.already_imported { "yes" } else { "no" },
            start
        );
    }
    println!("({} file(s))", entries.len());
}

/// The scan rows as JSON (C3 §3.3 `ScanEntry` field names).
fn entries_json(entries: &[ScanEntry]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|e| {
                json!({
                    "path": e.path.display().to_string(),
                    "file_name": e.file_name,
                    "size_bytes": e.size_bytes,
                    "importer_id": e.importer_id,
                    "already_imported": e.already_imported,
                    "session_start_utc_ms": e.session_start_utc_ms,
                })
            })
            .collect(),
    )
}

// ── fold-in ──────────────────────────────────────────────────────────────────

/// Imports every scanned entry an importer covers, through the same
/// `import_idl0`/`import_file` dispatch the app's `import_file` command uses.
///
/// With `move_sources`, a source file is deleted only once (a) its import
/// returned a report, (b) `sha256(source bytes)` equals the report's
/// `blob_sha256`, and (c) that blob is present under `<data>/blobs` — never
/// a rename, so the behaviour is identical within a volume and across one
/// (R197). A file that fails to import is never deleted.
///
/// `on_file` is called once per attempted file, in scan order, for progress
/// output. The catalog is not touched here — the caller rebuilds it once.
pub fn fold_in(
    data_root: &Path,
    entries: &[ScanEntry],
    move_sources: bool,
    mut on_file: impl FnMut(&FoldResult),
) -> FoldSummary {
    let mut summary = FoldSummary {
        results: Vec::new(),
        imported: 0,
        skipped: 0,
        failed: 0,
        no_importer: 0,
        unknown_start: entries.iter().filter(|e| start_unknown(e)).count(),
        moved: 0,
    };

    for entry in entries {
        let Some(importer_id) = entry.importer_id.as_deref() else {
            summary.no_importer += 1;
            let result = FoldResult {
                path: entry.path.clone(),
                outcome: FoldOutcome::NoImporter,
                error: None,
                source_removed: false,
                warnings: Vec::new(),
            };
            on_file(&result);
            summary.results.push(result);
            continue;
        };

        let mut result = match import_one(data_root, &entry.path, importer_id) {
            Ok((report, bytes)) => {
                let outcome = match report.outcome {
                    ImportOutcome::Skipped => FoldOutcome::Skipped,
                    _ => FoldOutcome::Imported,
                };
                let source_removed = move_sources && remove_verified_source(data_root, &entry.path, &report.blob_sha256, &bytes);
                let mut warnings = report.import_warnings.clone();
                if let Some(w) = &report.truncation_warning {
                    warnings.push(w.clone());
                }
                FoldResult { path: entry.path.clone(), outcome, error: None, source_removed, warnings }
            }
            Err(message) => FoldResult {
                path: entry.path.clone(),
                outcome: FoldOutcome::Failed,
                error: Some(message),
                source_removed: false,
                warnings: Vec::new(),
            },
        };

        match result.outcome {
            FoldOutcome::Imported => summary.imported += 1,
            FoldOutcome::Skipped => summary.skipped += 1,
            FoldOutcome::Failed => summary.failed += 1,
            FoldOutcome::NoImporter => summary.no_importer += 1,
        }
        if result.source_removed {
            summary.moved += 1;
        }
        result.warnings.dedup();
        on_file(&result);
        summary.results.push(result);
    }

    summary
}

/// Reads and imports one file, returning its report alongside the bytes that
/// produced it (the `--move` verification needs both). The error arm is the
/// typed core error's `Display`, never a bare string built here.
fn import_one(data_root: &Path, path: &Path, importer_id: &str) -> Result<(import::ImportReport, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let report: Result<import::ImportReport, ImportError> = if importer_id == "idl0" {
        import::import_idl0(data_root, &bytes)
    } else {
        import::import_file(data_root, importer_id, &bytes)
    };
    match report {
        Ok(report) => Ok((report, bytes)),
        Err(e) => Err(e.to_string()),
    }
}

/// Deletes `path` only if its bytes hash to `blob_sha256` and that blob is
/// present in the store. Returns `false` (leaving the file in place) on any
/// mismatch or delete failure — losing a source file is unrecoverable, so
/// every doubt keeps it.
fn remove_verified_source(data_root: &Path, path: &Path, blob_sha256: &str, bytes: &[u8]) -> bool {
    if sha256_hex(bytes) != blob_sha256 || !blob_exists(data_root, blob_sha256) {
        return false;
    }
    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("warning: imported but could not remove {}: {e}", path.display());
            false
        }
    }
}

/// One progress line for a folded file.
fn fold_line(result: &FoldResult) -> String {
    let verb = match result.outcome {
        FoldOutcome::Imported => "imported",
        FoldOutcome::Skipped => "skipped (already imported)",
        FoldOutcome::NoImporter => "no importer",
        FoldOutcome::Failed => "FAILED",
    };
    let moved = if result.source_removed { " (source removed)" } else { "" };
    match &result.error {
        Some(e) => format!("{verb}: {}{moved}: {e}", result.path.display()),
        None => format!("{verb}: {}{moved}", result.path.display()),
    }
}

/// The summary line, then every failed path.
fn print_fold_summary(summary: &FoldSummary) {
    for result in &summary.results {
        for warning in &result.warnings {
            eprintln!("warning: {}: {warning}", result.path.display());
        }
    }
    println!(
        "imported {}, skipped {}, failed {}, no importer {}, unknown start {}, moved {}",
        summary.imported, summary.skipped, summary.failed, summary.no_importer, summary.unknown_start, summary.moved
    );
    for result in summary.results.iter().filter(|r| r.outcome == FoldOutcome::Failed) {
        eprintln!("  failed: {}: {}", result.path.display(), result.error.as_deref().unwrap_or(""));
    }
}

/// The fold-in summary and per-file results as JSON.
fn fold_summary_json(summary: &FoldSummary) -> Value {
    json!({
        "imported": summary.imported,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "no_importer": summary.no_importer,
        "unknown_start": summary.unknown_start,
        "moved": summary.moved,
        "files": summary.results.iter().map(|r| json!({
            "path": r.path.display().to_string(),
            "outcome": match r.outcome {
                FoldOutcome::Imported => "imported",
                FoldOutcome::Skipped => "skipped",
                FoldOutcome::NoImporter => "no_importer",
                FoldOutcome::Failed => "failed",
            },
            "error": r.error,
            "source_removed": r.source_removed,
            "warnings": r.warnings,
        })).collect::<Vec<_>>(),
        "failed_paths": summary.results.iter()
            .filter(|r| r.outcome == FoldOutcome::Failed)
            .map(|r| r.path.display().to_string())
            .collect::<Vec<_>>(),
    })
}

// ── stale / rebuild ──────────────────────────────────────────────────────────

/// [`list_stale_sessions`] against `<data>/catalog.sqlite`. The catalog is an
/// index: this reads it, never rebuilds it.
pub fn stale(data_root: &Path) -> Result<Vec<StaleSession>, LibraryError> {
    let conn = catalog::open_catalog(&data_root.join("catalog.sqlite"))?;
    Ok(list_stale_sessions(&conn)?)
}

/// The stale table (session id, importer, stored vs current version).
fn print_stale_table(rows: &[StaleSession]) {
    println!("{:<36}  {:<8}  {:<12}  {}", "SESSION", "IMPORTER", "STORED", "CURRENT");
    for r in rows {
        println!("{:<36}  {:<8}  {:<12}  {}", r.session_id, r.importer_id, r.stored_version, r.current_version);
    }
    println!("({} stale session(s))", rows.len());
}

/// The stale rows as JSON (C3 §3.3 `StaleSession` field names).
fn stale_json(rows: &[StaleSession]) -> Value {
    Value::Array(
        rows.iter()
            .map(|r| {
                json!({
                    "session_id": r.session_id,
                    "importer_id": r.importer_id,
                    "stored_version": r.stored_version,
                    "current_version": r.current_version,
                })
            })
            .collect(),
    )
}

/// [`import::reimport_session`] per id, in the order given, never stopping at
/// the first failure — one unrebuildable session must not strand the rest.
pub fn rebuild(data_root: &Path, session_ids: &[String], mut on_session: impl FnMut(&RebuildResult)) -> Vec<RebuildResult> {
    let mut results = Vec::new();
    for session_id in session_ids {
        let result = match import::reimport_session(data_root, session_id) {
            Ok(report) => RebuildResult { session_id: session_id.clone(), error: None, outcome: Some(report.outcome) },
            Err(e) => RebuildResult { session_id: session_id.clone(), error: Some(e.to_string()), outcome: None },
        };
        on_session(&result);
        results.push(result);
    }
    results
}

/// One progress line for a rebuilt session.
fn rebuild_line(result: &RebuildResult) -> String {
    match &result.error {
        Some(e) => format!("FAILED: {}: {e}", result.session_id),
        None => format!("rebuilt: {} ({:?})", result.session_id, result.outcome),
    }
}

/// The rebuild results as JSON.
fn rebuild_json(results: &[RebuildResult]) -> Value {
    Value::Array(
        results
            .iter()
            .map(|r| {
                json!({
                    "session_id": r.session_id,
                    "rebuilt": r.error.is_none(),
                    "error": r.error,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::parse::test_buffers::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("idl-rs-cli-library-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A minimal valid `.idl0` v3 buffer: one IMU sample, a header carrying
    /// `session_start_ms` and a session UUID derived from `uuid_byte` (so
    /// two fixtures are two different sessions).
    fn idl0_bytes(session_start_ms: i64, uuid_byte: u8) -> Vec<u8> {
        let accel: f32 = 32.0 / 32768.0;
        cat(&[
            Header {
                schema_version: 3,
                uuid: vec![uuid_byte; 16],
                session_start_ms,
                imu_mask: 0x01,
                ..Default::default()
            }
            .build(&[v3_registry_entry(0, 4, 800, accel, 0.0, "IMU0_AccelX", "g")]),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ])
    }

    #[test]
    fn collect_entries_recursive_includes_sub_directories_and_orders_by_path() {
        // Arrange
        let data_root = temp_dir("cr-data");
        let folder = temp_dir("cr-src");
        std::fs::write(folder.join("a.idl0"), idl0_bytes(0, 0x11)).unwrap();
        std::fs::create_dir_all(folder.join("nested")).unwrap();
        std::fs::write(folder.join("nested").join("b.idl0"), idl0_bytes(0, 0x22)).unwrap();

        // Act
        let flat = collect_entries(&data_root, &folder, false).unwrap();
        let deep = collect_entries(&data_root, &folder, true).unwrap();

        // Assert
        assert_eq!(flat.iter().map(|e| e.file_name.as_str()).collect::<Vec<_>>(), vec!["a.idl0"]);
        assert_eq!(deep.iter().map(|e| e.file_name.as_str()).collect::<Vec<_>>(), vec!["a.idl0", "b.idl0"]);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn collect_entries_a_missing_folder_returns_a_typed_scan_error() {
        // Arrange
        let data_root = temp_dir("miss-data");
        let folder = data_root.join("nope");

        // Act
        let result = collect_entries(&data_root, &folder, false);

        // Assert
        assert!(matches!(result, Err(LibraryError::Scan(_))));

        let _ = std::fs::remove_dir_all(&data_root);
    }

    #[test]
    fn fold_in_imports_idl0_files_skips_unknown_extensions_and_counts_unknown_starts() {
        // Arrange
        let data_root = temp_dir("fi-data");
        let folder = temp_dir("fi-src");
        std::fs::write(folder.join("a.idl0"), idl0_bytes(1_700_000_000_000, 0x11)).unwrap();
        std::fs::write(folder.join("b.idl0"), idl0_bytes(0, 0x22)).unwrap();
        std::fs::write(folder.join("c.txt"), b"notes").unwrap();
        let entries = collect_entries(&data_root, &folder, false).unwrap();

        // Act
        let summary = fold_in(&data_root, &entries, false, |_| {});

        // Assert — two imports, the .txt listed but never opened, and only
        // the zero-start .idl0 counted as unknown.
        assert_eq!(summary.imported, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.no_importer, 1);
        assert_eq!(summary.unknown_start, 2, "b.idl0 (start 0) and c.txt (no peek)");
        assert_eq!(summary.moved, 0);
        assert!(folder.join("a.idl0").is_file(), "no --move: sources stay");

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn fold_in_rerun_over_the_same_folder_reports_skipped_not_imported() {
        // Arrange
        let data_root = temp_dir("re-data");
        let folder = temp_dir("re-src");
        std::fs::write(folder.join("a.idl0"), idl0_bytes(1_700_000_000_000, 0x11)).unwrap();
        let first = collect_entries(&data_root, &folder, false).unwrap();
        fold_in(&data_root, &first, false, |_| {});

        // Act
        let second = collect_entries(&data_root, &folder, false).unwrap();
        let summary = fold_in(&data_root, &second, false, |_| {});

        // Assert
        assert!(second[0].already_imported);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.imported, 0);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn fold_in_move_deletes_the_source_only_after_its_blob_verifies() {
        // Arrange
        let data_root = temp_dir("mv-data");
        let folder = temp_dir("mv-src");
        std::fs::write(folder.join("good.idl0"), idl0_bytes(1_700_000_000_000, 0x11)).unwrap();
        std::fs::write(folder.join("bad.idl0"), b"not an idl0 file at all").unwrap();
        let entries = collect_entries(&data_root, &folder, false).unwrap();

        // Act
        let summary = fold_in(&data_root, &entries, true, |_| {});

        // Assert — the good file moved, the failed one is still on disk.
        assert_eq!(summary.moved, 1);
        assert_eq!(summary.failed, 1);
        assert!(!folder.join("good.idl0").exists(), "verified source deleted");
        assert!(folder.join("bad.idl0").is_file(), "a failed import never deletes its source");

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn fold_in_a_failed_file_records_the_typed_core_error_and_keeps_going() {
        // Arrange
        let data_root = temp_dir("err-data");
        let folder = temp_dir("err-src");
        std::fs::write(folder.join("a-bad.idl0"), b"garbage").unwrap();
        std::fs::write(folder.join("b-good.idl0"), idl0_bytes(1_700_000_000_000, 0x33)).unwrap();
        let entries = collect_entries(&data_root, &folder, false).unwrap();

        // Act
        let summary = fold_in(&data_root, &entries, false, |_| {});

        // Assert
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.imported, 1, "the run continues past a failure");
        let failed = summary.results.iter().find(|r| r.outcome == FoldOutcome::Failed).unwrap();
        assert!(failed.error.as_deref().unwrap_or("").contains("MagicBytes"), "{:?}", failed.error);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn stale_a_freshly_imported_data_dir_lists_nothing() {
        // Arrange
        let data_root = temp_dir("st-data");
        let folder = temp_dir("st-src");
        std::fs::write(folder.join("a.idl0"), idl0_bytes(1_700_000_000_000, 0x11)).unwrap();
        let entries = collect_entries(&data_root, &folder, false).unwrap();
        fold_in(&data_root, &entries, false, |_| {});
        catalog::rebuild_catalog(&data_root).unwrap();

        // Act
        let rows = stale(&data_root).unwrap();

        // Assert
        assert!(rows.is_empty(), "{rows:?}");

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn rebuild_an_imported_session_succeeds_and_an_unknown_id_fails_without_stopping_the_run() {
        // Arrange
        let data_root = temp_dir("rb-data");
        let folder = temp_dir("rb-src");
        std::fs::write(folder.join("a.idl0"), idl0_bytes(1_700_000_000_000, 0x11)).unwrap();
        let entries = collect_entries(&data_root, &folder, false).unwrap();
        let summary = fold_in(&data_root, &entries, false, |_| {});
        let session_id = read_only_session_id(&data_root);
        assert_eq!(summary.imported, 1);

        // Act
        let ids = vec!["no-such-session".to_string(), session_id.clone()];
        let results = rebuild(&data_root, &ids, |_| {});

        // Assert
        assert_eq!(results.len(), 2);
        assert!(results[0].error.is_some(), "unknown id fails");
        assert!(results[1].error.is_none(), "{:?}", results[1].error);

        let _ = std::fs::remove_dir_all(&data_root);
        let _ = std::fs::remove_dir_all(&folder);
    }

    /// The single session id under `<data>/sessions`, for tests that import
    /// exactly one file.
    fn read_only_session_id(data_root: &Path) -> String {
        let mut ids: Vec<String> = std::fs::read_dir(data_root.join("sessions"))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        ids.sort();
        assert_eq!(ids.len(), 1, "{ids:?}");
        ids.remove(0)
    }
}
