//! Library-wide lap/track indexing as one resumable, parallel, memory-bounded
//! job (rulings R207 and R208 item 1).
//!
//! Indexing a session means two things, in this order:
//!
//! 1. **tracks** — detect this session's track visits and the laps within
//!    them against the track library, and merge the result into
//!    `session.json` (`store::lap_index::index_laps_with_tracks`);
//! 2. **laps** — refresh that session's `sessions`/`laps`/`lap_summary` rows
//!    in `catalog.sqlite` (`store::catalog::index_session`), in that
//!    session's own transaction.
//!
//! Both commit per session, so a killed process loses at most the session in
//! flight, and a restarted job skips every session whose stamped
//! `lap_detector_version`/`track_visits_library_hash` are already current
//! (ruling R208.1 item 2). Sessions are independent, so they run on a pool of
//! `physical cores − 1` workers, each holding a byte reservation from the
//! caller's [`DecodeBudget`] for the duration of its decode so N workers can
//! never exceed the process's memory budget (ruling R208.1 item 3). Workers
//! **wait** for memory; they never fail for it (ruling R211).
//!
//! This module never blocks a user: `store::index_job` is what the app runs
//! in the background and what the CLI runs at the end of a fold-in. Opening
//! one session's workbook goes through [`index_one_session`], which indexes
//! exactly that session.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use rusqlite::Connection;

use crate::store::catalog::{index_session, open_catalog, CatalogError};
use crate::store::lap_index::{
    index_laps_with_tracks, load_track_library, session_index_is_current, track_library_hash, LapIndexError,
};
use crate::store::parquet::{estimate_channel_bytes, open_session_lazy};
use crate::track_artifact::Track;

/// The channels track-visit and lap detection read (`gps::fixes_from`), and
/// therefore the only ones one session's indexing decodes. Sized for the
/// [`DecodeBudget`] reservation; everything else in `data.parquet` is left
/// on disk.
const INDEXED_CHANNELS: [&str; 3] = ["GPS_Latitude", "GPS_Longitude", "GPS_EpochMs"];

/// Bytes charged for a session whose channel index cannot be read (a
/// mid-import directory, a truncated footer): 64 MiB. A guess is needed
/// because the reservation is taken before the file is opened; erring high
/// costs a little concurrency, erring low costs the budget its meaning.
const UNKNOWN_SESSION_BYTES: u64 = 64 * 1024 * 1024;

// ── the memory budget ────────────────────────────────────────────────────

/// An in-flight byte reservation. Dropping it releases the bytes. Opaque:
/// a caller holds it and drops it, and never looks inside.
pub trait BudgetGuard: Send {}

/// The no-op guard [`NoBudget`] hands out.
impl BudgetGuard for () {}

/// A process-wide byte semaphore an indexing worker reserves against before
/// it decodes (ruling R208.1 item 3, ruling R211's "wait, don't fail").
///
/// The app implements this over `idl-rs-tauri`'s `SessionCache`, so indexing
/// decodes queue behind the same budget as every command's; the CLI uses
/// [`ByteBudget`]; tests use [`NoBudget`].
pub trait DecodeBudget: Sync {
    /// Blocks until `bytes` are available, then returns a guard holding
    /// them. Never fails: a worker that cannot have memory yet waits for a
    /// worker that has it to finish. `hint` names the work being sized, for
    /// whatever diagnostics the implementation keeps.
    fn acquire<'a>(&'a self, bytes: u64, hint: &str) -> Box<dyn BudgetGuard + 'a>;
}

/// A [`DecodeBudget`] that grants everything immediately — for tests and for
/// a caller that bounds memory some other way. Never use it in the app: the
/// whole point of the budget is that N workers share one ceiling.
pub struct NoBudget;

impl DecodeBudget for NoBudget {
    fn acquire<'a>(&'a self, _bytes: u64, _hint: &str) -> Box<dyn BudgetGuard + 'a> {
        Box::new(())
    }
}

/// A plain counting byte semaphore: at most `budget_bytes` reserved across
/// all threads at once, waiters served as reservations are dropped. The
/// CLI's [`DecodeBudget`] (the app has `SessionCache`, which counts the same
/// bytes against its own residency budget).
///
/// A single request larger than the whole budget is granted rather than
/// deadlocking forever — it is the only thing running when it is granted,
/// which is the closest this type can get to honouring "wait, never fail".
pub struct ByteBudget {
    budget_bytes: u64,
    /// Bytes currently reserved across all threads.
    in_flight: Mutex<u64>,
    /// Signalled every time a guard is dropped.
    released: Condvar,
}

impl ByteBudget {
    /// A budget of `budget_bytes` bytes, floored at 1 byte so an accidental
    /// zero cannot make every acquire wait forever.
    pub fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes: budget_bytes.max(1), in_flight: Mutex::new(0), released: Condvar::new() }
    }

    /// The ceiling this budget was built with, bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Bytes currently reserved across all threads.
    pub fn in_flight_bytes(&self) -> u64 {
        *self.in_flight.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl DecodeBudget for ByteBudget {
    fn acquire<'a>(&'a self, bytes: u64, _hint: &str) -> Box<dyn BudgetGuard + 'a> {
        let mut in_flight = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // `*in_flight == 0` is the oversized-request escape: nothing else
            // holds the budget, so granting it is the least-bad option.
            if *in_flight == 0 || in_flight.saturating_add(bytes) <= self.budget_bytes {
                *in_flight = in_flight.saturating_add(bytes);
                return Box::new(ByteGuard { budget: self, bytes });
            }
            in_flight = self.released.wait(in_flight).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// [`ByteBudget`]'s RAII guard: gives the bytes back and wakes one waiter.
struct ByteGuard<'a> {
    budget: &'a ByteBudget,
    bytes: u64,
}

impl BudgetGuard for ByteGuard<'_> {}

impl Drop for ByteGuard<'_> {
    fn drop(&mut self) {
        let mut in_flight = self.budget.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        *in_flight = in_flight.saturating_sub(self.bytes);
        drop(in_flight);
        self.budget.released.notify_all();
    }
}

// ── progress, reports, errors ────────────────────────────────────────────

/// Which half of one session's indexing is running (C3 §3.2
/// `index_progress.phase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPhase {
    /// Detecting this session's track visits and their laps, and writing
    /// `session.json`.
    Tracks,
    /// Writing this session's catalog `sessions`/`laps`/`lap_summary` rows.
    Laps,
}

impl IndexPhase {
    /// The wire spelling (C3 §3.2): `"tracks"` or `"laps"`. Byte-exact — the
    /// frontend compares against these strings.
    pub fn as_str(self) -> &'static str {
        match self {
            IndexPhase::Tracks => "tracks",
            IndexPhase::Laps => "laps",
        }
    }
}

impl fmt::Display for IndexPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One `index_progress` observation (C3 §3.2). Emitted when a session's work
/// enters a phase, so `done` is the number of sessions **finished** before
/// this one — `done + 1` is the one being worked on, the way a person counts
/// a stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProgress {
    /// Sessions finished so far (indexed, skipped-current or failed).
    pub done: usize,
    /// Sessions this run was asked to consider.
    pub total: usize,
    /// The session entering [`Self::phase`].
    pub current_session_id: String,
    /// Which half of that session's work is starting.
    pub phase: IndexPhase,
}

/// What one session's indexing did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionIndexOutcome {
    /// Recomputed and written.
    Indexed,
    /// The stamps were already current; nothing was read or written.
    SkippedUpToDate,
}

/// Why one session could not be indexed. The run continues: one unreadable
/// session never stops the library (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexFailure {
    /// The session that failed.
    pub session_id: String,
    /// The typed core error's `Display`.
    pub message: String,
}

/// Outcome of [`index_sessions`]/[`index_library`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexJobReport {
    /// Sessions this run was asked to consider.
    pub total: usize,
    /// Sessions recomputed and written.
    pub indexed: usize,
    /// Sessions whose stamps were already current.
    pub skipped_up_to_date: usize,
    /// Per-session failures, in completion order.
    pub failed: Vec<IndexFailure>,
    /// True when the cancel flag was set before every session finished.
    pub cancelled: bool,
}

/// Discriminant for [`IndexJobError`] — the errors that stop a whole run
/// before any session is considered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexJobErrorKind {
    /// Reading `<data>/sessions/` or `<data>/tracks/` failed.
    Io,
    /// Opening `catalog.sqlite` failed.
    Catalog,
}

/// Error from [`index_library`]'s setup or [`index_one_session`]. Never
/// `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexJobError {
    /// Discriminant.
    pub kind: IndexJobErrorKind,
    /// Human-readable detail, naming the offending path where relevant.
    pub message: String,
}

impl IndexJobError {
    fn new(kind: IndexJobErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for IndexJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for IndexJobError {}

impl From<LapIndexError> for IndexJobError {
    fn from(e: LapIndexError) -> Self {
        IndexJobError::new(IndexJobErrorKind::Io, e.message)
    }
}

impl From<CatalogError> for IndexJobError {
    fn from(e: CatalogError) -> Self {
        IndexJobError::new(IndexJobErrorKind::Catalog, e.message)
    }
}

// ── the job ──────────────────────────────────────────────────────────────

/// Workers the index pool uses: physical cores − 1, at least 1 (ruling
/// R208.1 item 3). One core is deliberately left for the UI thread and the
/// webview; on a single-core machine the job is serial.
pub fn worker_count() -> usize {
    num_cpus::get_physical().saturating_sub(1).max(1)
}

/// Everything a run needs beyond the data root: how wide to go, what to
/// reserve memory against, how to be stopped, and where to report.
pub struct IndexJobOptions<'a> {
    /// Pool width. [`worker_count`] is the app's and the CLI's default; a
    /// test or a `--workers 1` measurement passes its own.
    pub workers: usize,
    /// Reserved against before each session's decode.
    pub budget: &'a dyn DecodeBudget,
    /// Polled before each session and between phases; a set flag ends the
    /// run with [`IndexJobReport::cancelled`] true and every finished
    /// session's work already committed.
    pub cancel: &'a AtomicBool,
    /// Called as each session enters each phase, from whichever worker owns
    /// that session — so it must be cheap and thread-safe.
    pub progress: &'a (dyn Fn(IndexProgress) + Sync),
}

impl<'a> IndexJobOptions<'a> {
    /// A run at [`worker_count`] width with no cancellation and no progress
    /// reporting, against `budget`. `cancel` is borrowed because the flag
    /// has to outlive the options; pass a `&AtomicBool::new(false)` for a
    /// run nothing will stop.
    pub fn new(budget: &'a dyn DecodeBudget, cancel: &'a AtomicBool) -> Self {
        Self { workers: worker_count(), budget, cancel, progress: &|_| {} }
    }
}

/// Every session directory under `<data_root>/sessions/`, sorted by id.
/// A missing `sessions/` directory is an empty list, not an error.
pub fn list_session_ids(data_root: &Path) -> Result<Vec<String>, IndexJobError> {
    let dir = data_root.join("sessions");
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| IndexJobError::new(IndexJobErrorKind::Io, format!("cannot read {}: {e}", dir.display())))?;
    let mut ids: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            ids.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    ids.sort();
    Ok(ids)
}

/// Every session whose lap/track index is not current against the track
/// library as it stands right now — the work [`index_library`] would do.
/// Reads only `session.json` per session, so it is cheap enough to call on
/// launch before deciding whether to start a job at all.
pub fn stale_session_ids(data_root: &Path) -> Result<Vec<String>, IndexJobError> {
    let (tracks, _) = load_track_library(data_root)?;
    let hash = track_library_hash(&tracks);
    let ids = list_session_ids(data_root)?;
    Ok(ids.into_iter().filter(|id| !session_index_is_current(data_root, id, &hash)).collect())
}

/// Indexes every session under `data_root` whose index is not current.
/// Equivalent to [`stale_session_ids`] followed by [`index_sessions`], but
/// loads the track library once for both.
pub fn index_library(data_root: &Path, opts: &IndexJobOptions<'_>) -> Result<IndexJobReport, IndexJobError> {
    let ids = list_session_ids(data_root)?;
    index_sessions(data_root, &ids, false, opts)
}

/// Indexes the named sessions, skipping any whose stamps are already
/// current unless `force`.
///
/// The track library is loaded once and shared by every worker. Each session
/// runs on the pool: its `session.json` merge and then its catalog
/// transaction, both committed before the next session's work is counted.
/// A per-session failure lands in [`IndexJobReport::failed`] and the run
/// continues.
///
/// Errors only when the run cannot start at all — an unreadable
/// `<data>/tracks/`, or a `catalog.sqlite` that will not open.
pub fn index_sessions(
    data_root: &Path,
    session_ids: &[String],
    force: bool,
    opts: &IndexJobOptions<'_>,
) -> Result<IndexJobReport, IndexJobError> {
    let (tracks, warnings) = load_track_library(data_root)?;
    let hash = track_library_hash(&tracks);

    let conn = open_catalog(&data_root.join("catalog.sqlite"))?;
    let conn = Mutex::new(conn);

    let total = session_ids.len();
    let done = AtomicUsize::new(0);
    let results: Mutex<IndexJobReport> = Mutex::new(IndexJobReport { total, ..Default::default() });

    let run_one = |session_id: &String| {
        if opts.cancel.load(Ordering::Relaxed) {
            return;
        }
        let outcome = index_one(
            data_root,
            session_id,
            &tracks,
            &warnings,
            &hash,
            force,
            opts.budget,
            &conn,
            &done,
            total,
            opts.progress,
        );
        done.fetch_add(1, Ordering::Relaxed);
        let mut report = results.lock().unwrap_or_else(|e| e.into_inner());
        match outcome {
            Ok(SessionIndexOutcome::Indexed) => report.indexed += 1,
            Ok(SessionIndexOutcome::SkippedUpToDate) => report.skipped_up_to_date += 1,
            Err(e) => report.failed.push(IndexFailure { session_id: session_id.clone(), message: e.to_string() }),
        }
    };

    let workers = opts.workers.max(1);
    if workers == 1 || total <= 1 {
        session_ids.iter().for_each(run_one);
    } else {
        // A private pool, not the global one: this job must not inherit (or
        // impose) another caller's width, and `physical cores − 1` is the
        // whole point of the ruling.
        match rayon::ThreadPoolBuilder::new().num_threads(workers).build() {
            Ok(pool) => pool.install(|| {
                use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
                session_ids.par_iter().for_each(run_one);
            }),
            // A pool that will not build (a thread-spawn failure on an
            // exhausted machine) degrades to serial rather than failing the
            // run: indexing slowly is strictly better than not at all.
            Err(_) => session_ids.iter().for_each(run_one),
        }
    }

    let mut report = results.into_inner().unwrap_or_else(|e| e.into_inner());
    report.cancelled = opts.cancel.load(Ordering::Relaxed);
    Ok(report)
}

/// Indexes exactly one session, now, on the calling thread — the path
/// "opening a workbook or a session needs only that session's index"
/// (ruling R207/R208.1 item 1) takes when that session's index is missing.
///
/// Cheap enough to call before every open, but not free: it reads
/// `<data>/tracks/` (every `.idl0t`, to compute the hash the stamps are
/// compared against) and opens `catalog.sqlite` before it can decide
/// anything. What it skips when the stamps are current is the expensive
/// part — opening `data.parquet`, decoding GPS, detecting visits and laps,
/// and rewriting `session.json` and the catalog rows.
pub fn index_one_session(
    data_root: &Path,
    session_id: &str,
    budget: &dyn DecodeBudget,
) -> Result<SessionIndexOutcome, IndexJobError> {
    let (tracks, warnings) = load_track_library(data_root)?;
    let hash = track_library_hash(&tracks);
    let conn = Mutex::new(open_catalog(&data_root.join("catalog.sqlite"))?);
    let done = AtomicUsize::new(0);
    let no_progress = |_: IndexProgress| {};
    index_one(
        data_root,
        session_id,
        &tracks,
        &warnings,
        &hash,
        false,
        budget,
        &conn,
        &done,
        1,
        &no_progress,
    )
}

/// Bytes one session's indexing decodes: its three GPS channels, summed
/// from `data.parquet`'s own footer. A session whose footer cannot be read
/// is charged [`UNKNOWN_SESSION_BYTES`] rather than nothing — a
/// reservation of zero would let unbounded workers through.
fn index_bytes(session_dir: &Path) -> u64 {
    let mut total: u64 = 0;
    for channel in INDEXED_CHANNELS {
        match estimate_channel_bytes(session_dir, channel) {
            Ok(bytes) => total = total.saturating_add(bytes),
            // A session with no GPS at all is legitimate (an IMU-only log):
            // it contributes nothing to the estimate and detects no visits.
            Err(_) => continue,
        }
    }
    if total == 0 {
        // Distinguish "no GPS columns" (cheap, and `read_channel_index`
        // succeeded) from "footer unreadable" (unknown, charge the guess).
        if crate::store::parquet::estimate_session_bytes(session_dir).is_err() {
            return UNKNOWN_SESSION_BYTES;
        }
    }
    total
}

/// One session's whole unit of work: the tracks phase (detect, write
/// `session.json`) then the laps phase (the catalog transaction). The byte
/// reservation is held across both, since the decoded GPS samples stay
/// resident for the whole of it.
#[allow(clippy::too_many_arguments)]
fn index_one(
    data_root: &Path,
    session_id: &str,
    tracks: &[Track],
    library_warnings: &[String],
    library_hash: &str,
    force: bool,
    budget: &dyn DecodeBudget,
    conn: &Mutex<Connection>,
    done: &AtomicUsize,
    total: usize,
    progress: &(dyn Fn(IndexProgress) + Sync),
) -> Result<SessionIndexOutcome, IndexJobError> {
    if !force && session_index_is_current(data_root, session_id, library_hash) {
        return Ok(SessionIndexOutcome::SkippedUpToDate);
    }

    let session_dir: PathBuf = data_root.join("sessions").join(session_id);
    let report = |phase: IndexPhase| {
        progress(IndexProgress {
            done: done.load(Ordering::Relaxed),
            total,
            current_session_id: session_id.to_string(),
            phase,
        });
    };

    report(IndexPhase::Tracks);
    let _permit = budget.acquire(index_bytes(&session_dir), &format!("index session '{session_id}'"));

    let handle = open_session_lazy(&session_dir).map_err(|e| {
        IndexJobError::new(
            IndexJobErrorKind::Io,
            format!("reading {}: {e}", session_dir.join("data.parquet").display()),
        )
    })?;
    index_laps_with_tracks(data_root, session_id, &handle, tracks, library_warnings, force)?;

    report(IndexPhase::Laps);
    let conn = conn.lock().unwrap_or_else(|e| e.into_inner());
    index_session(&conn, data_root, session_id)?;

    Ok(SessionIndexOutcome::Indexed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::GpsFix;
    use crate::laps::model::{Gate, LapTiming};
    use crate::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use crate::store::blob::write_blob;
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::read_session_json;
    use crate::track_artifact::write_track;
    use std::sync::Arc;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-idxjob-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The same there-and-back-and-there GPS shape `lap_index`'s tests use:
    /// 3 legs of 100 one-second fixes crossing a gate at lat 0.05 three
    /// times, so a circuit track there detects three laps.
    fn three_lap_fixes() -> Vec<GpsFix> {
        let leg = |t0: i64, up: bool| -> Vec<GpsFix> {
            (0..100)
                .map(|i| {
                    let lat = if up { i as f64 * 0.001 } else { 0.099 - i as f64 * 0.001 };
                    GpsFix { timestamp_ms: t0 + i * 1000, lat, lon: 0.0005 }
                })
                .collect()
        };
        let mut fixes = leg(0, true);
        fixes.extend(leg(100_000, false));
        fixes.extend(leg(200_000, true));
        fixes
    }

    fn circuit_track(id: &str) -> crate::track_artifact::Track {
        let polyline: Vec<GpsFix> =
            (0..=100).map(|i| GpsFix { timestamp_ms: i * 1000, lat: i as f64 * 0.001, lon: 0.0005 }).collect();
        crate::track_artifact::Track {
            id: id.to_string(),
            name: "Loop".to_string(),
            venue: String::new(),
            timing: Some(LapTiming::Circuit { start_finish: Gate { lat1: 0.05, lon1: -0.001, lat2: 0.05, lon2: 0.001 } }),
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: polyline,
            created_at_ms: 0,
            updated_at_ms: 1,
        }
    }

    /// Writes `sessions/<id>/data.parquet` plus the blob its catalog row
    /// references, so `index_session` can insert it.
    fn write_session(root: &Path, session_id: &str, fixes: &[GpsFix]) {
        let blob_sha = write_blob(root, format!("source bytes for {session_id}").as_bytes()).unwrap();
        let ch = |id: &str, s: Vec<f64>| Channel {
            channel_id: id.to_string(),
            t_us: (0..s.len() as i64).map(|i| i * 1_000_000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1.0,
            column: RawColumn::F64(s),
            source_kind: id.to_lowercase(),
            unit: String::new(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Gpx,
            blob_sha256: blob_sha,
            channels: vec![
                ch("GPS_Latitude", fixes.iter().map(|f| f.lat).collect()),
                ch("GPS_Longitude", fixes.iter().map(|f| f.lon).collect()),
                ch("GPS_EpochMs", fixes.iter().map(|f| f.timestamp_ms as f64).collect()),
            ],
        };
        write_session_parquet(root, &session, "test-importer").unwrap();
    }

    // ── the budget ───────────────────────────────────────────────────────

    #[test]
    fn byte_budget_two_requests_that_do_not_both_fit_never_overlap() {
        // Arrange -- a budget that holds one 600 KB request but not two.
        let budget = Arc::new(ByteBudget::new(1_000_000));
        let peak = Arc::new(Mutex::new(0u64));

        // Act -- four threads each reserve 600 KB and observe the in-flight
        // total while holding it.
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let budget = budget.clone();
                let peak = peak.clone();
                scope.spawn(move || {
                    let guard = budget.acquire(600_000, "test");
                    let seen = budget.in_flight_bytes();
                    let mut peak = peak.lock().unwrap();
                    *peak = (*peak).max(seen);
                    drop(peak);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    drop(guard);
                });
            }
        });

        // Assert
        assert_eq!(*peak.lock().unwrap(), 600_000);
        assert_eq!(budget.in_flight_bytes(), 0);
    }

    #[test]
    fn byte_budget_a_request_larger_than_the_whole_budget_is_granted_rather_than_deadlocking() {
        // Arrange
        let budget = ByteBudget::new(1_000);

        // Act
        let guard = budget.acquire(50_000, "one oversized decode");

        // Assert
        assert_eq!(budget.in_flight_bytes(), 50_000);
        drop(guard);
        assert_eq!(budget.in_flight_bytes(), 0);
    }

    // ── the job ──────────────────────────────────────────────────────────

    #[test]
    fn index_library_three_sessions_indexes_every_one_and_writes_their_laps() {
        // Arrange -- one circuit track and three sessions that each lap it.
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        for id in ["s1", "s2", "s3"] {
            write_session(&root, id, &three_lap_fixes());
        }
        let cancel = AtomicBool::new(false);
        let opts = IndexJobOptions::new(&NoBudget, &cancel);

        // Act
        let report = index_library(&root, &opts).unwrap();

        // Assert
        assert_eq!((report.total, report.indexed, report.skipped_up_to_date), (3, 3, 0));
        assert_eq!(report.failed, Vec::new());
        assert!(!report.cancelled);
        for id in ["s1", "s2", "s3"] {
            let doc = read_session_json(&root.join("sessions").join(id).join("session.json")).unwrap();
            assert_eq!(doc.laps.len(), 3);
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_library_run_again_with_an_unchanged_library_skips_every_session() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        for id in ["s1", "s2"] {
            write_session(&root, id, &three_lap_fixes());
        }
        let cancel = AtomicBool::new(false);
        let opts = IndexJobOptions::new(&NoBudget, &cancel);
        index_library(&root, &opts).unwrap();

        // Act -- the resume path: nothing has changed since.
        let report = index_library(&root, &opts).unwrap();

        // Assert -- `index_library` filters nothing itself; each session is
        // considered and skipped on its own stamp.
        assert_eq!((report.total, report.indexed, report.skipped_up_to_date), (2, 0, 2));
        assert!(stale_session_ids(&root).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_sessions_one_worker_and_the_pool_produce_the_same_report() {
        // Arrange -- the same library indexed twice from scratch.
        let serial_root = temp_root();
        let pool_root = temp_root();
        for root in [&serial_root, &pool_root] {
            write_track(root, &circuit_track("loop-1")).unwrap();
            for id in ["s1", "s2", "s3", "s4"] {
                write_session(root, id, &three_lap_fixes());
            }
        }
        let cancel = AtomicBool::new(false);
        let ids: Vec<String> = ["s1", "s2", "s3", "s4"].iter().map(|s| s.to_string()).collect();

        // Act
        let serial = index_sessions(
            &serial_root,
            &ids,
            false,
            &IndexJobOptions { workers: 1, ..IndexJobOptions::new(&NoBudget, &cancel) },
        )
        .unwrap();
        let parallel = index_sessions(
            &pool_root,
            &ids,
            false,
            &IndexJobOptions { workers: 4, ..IndexJobOptions::new(&NoBudget, &cancel) },
        )
        .unwrap();

        // Assert
        assert_eq!(serial, parallel);
        assert_eq!(serial.indexed, 4);

        let _ = std::fs::remove_dir_all(&serial_root);
        let _ = std::fs::remove_dir_all(&pool_root);
    }

    #[test]
    fn index_sessions_a_cancel_set_before_the_run_indexes_nothing_and_reports_cancelled() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        write_session(&root, "s1", &three_lap_fixes());
        let cancel = AtomicBool::new(true);

        // Act
        let report =
            index_sessions(&root, &["s1".to_string()], false, &IndexJobOptions::new(&NoBudget, &cancel)).unwrap();

        // Assert -- nothing was indexed, and the session is still stale, so
        // a resumed run picks it up.
        assert!(report.cancelled);
        assert_eq!((report.indexed, report.skipped_up_to_date), (0, 0));
        assert_eq!(stale_session_ids(&root).unwrap(), vec!["s1".to_string()]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_sessions_progress_reports_both_phases_for_every_session_in_range() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        for id in ["s1", "s2"] {
            write_session(&root, id, &three_lap_fixes());
        }
        let cancel = AtomicBool::new(false);
        let seen: Mutex<Vec<IndexProgress>> = Mutex::new(Vec::new());
        let record = |p: IndexProgress| seen.lock().unwrap_or_else(|e| e.into_inner()).push(p);
        let ids: Vec<String> = vec!["s1".to_string(), "s2".to_string()];

        // Act
        index_sessions(
            &root,
            &ids,
            false,
            &IndexJobOptions { workers: 1, progress: &record, ..IndexJobOptions::new(&NoBudget, &cancel) },
        )
        .unwrap();

        // Assert -- two phases per session, `done` never past `total`.
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0].phase, IndexPhase::Tracks);
        assert_eq!(seen[1].phase, IndexPhase::Laps);
        assert_eq!(seen[0].current_session_id, "s1");
        assert_eq!(seen[2].done, 1);
        assert!(seen.iter().all(|p| p.total == 2 && p.done < p.total));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_sessions_a_session_with_no_data_parquet_fails_alone_and_the_others_still_index() {
        // Arrange -- "ghost" has a directory but nothing in it.
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        write_session(&root, "s1", &three_lap_fixes());
        std::fs::create_dir_all(root.join("sessions").join("ghost")).unwrap();
        let cancel = AtomicBool::new(false);

        // Act
        let report = index_library(&root, &IndexJobOptions::new(&NoBudget, &cancel)).unwrap();

        // Assert
        assert_eq!(report.indexed, 1);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].session_id, "ghost");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_one_session_indexes_only_the_named_session_then_skips_on_the_second_call() {
        // Arrange
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        write_session(&root, "s1", &three_lap_fixes());
        write_session(&root, "s2", &three_lap_fixes());

        // Act
        let first = index_one_session(&root, "s1", &NoBudget).unwrap();
        let second = index_one_session(&root, "s1", &NoBudget).unwrap();

        // Assert -- s1 is done, s2 untouched (still stale).
        assert_eq!(first, SessionIndexOutcome::Indexed);
        assert_eq!(second, SessionIndexOutcome::SkippedUpToDate);
        assert_eq!(stale_session_ids(&root).unwrap(), vec!["s2".to_string()]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stale_session_ids_after_a_track_edit_lists_every_session_again() {
        // Arrange -- a fully indexed library.
        let root = temp_root();
        write_track(&root, &circuit_track("loop-1")).unwrap();
        write_session(&root, "s1", &three_lap_fixes());
        let cancel = AtomicBool::new(false);
        index_library(&root, &IndexJobOptions::new(&NoBudget, &cancel)).unwrap();
        assert!(stale_session_ids(&root).unwrap().is_empty());

        // Act -- the track's `updated_at_ms` changes, so the library hash does.
        let mut bumped = circuit_track("loop-1");
        bumped.updated_at_ms = 999;
        write_track(&root, &bumped).unwrap();

        // Assert
        assert_eq!(stale_session_ids(&root).unwrap(), vec!["s1".to_string()]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn worker_count_is_at_least_one_and_under_the_machines_physical_core_count() {
        // Act
        let workers = worker_count();

        // Assert
        assert!(workers >= 1);
        assert!(workers <= num_cpus::get_physical().max(1));
    }

    #[test]
    fn index_phase_wire_spellings_are_the_c3_strings() {
        // Act + Assert -- byte-exact, the frontend compares against these.
        assert_eq!(IndexPhase::Tracks.as_str(), "tracks");
        assert_eq!(IndexPhase::Laps.as_str(), "laps");
    }
}
