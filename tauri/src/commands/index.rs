//! C3 §3.2 library indexing: `start_index_job`, `index_status`,
//! `cancel_index_job`, and the `index_progress` event (rulings R207 and
//! R208 item 1).
//!
//! The job itself is `idl_rs::store::index_job` — the same core function the
//! CLI runs at the end of a fold-in, on the same pool. This module is the
//! glue: it owns the background thread, the managed state a UI mounting
//! mid-flight reads, the cancel flag, and the event.
//!
//! **Indexing never sits between the user and a workbook** (ruling R207
//! item 1): nothing here is awaited by a command that opens something. The
//! per-session path a session open needs is
//! `idl_rs::store::index_job::index_one_session`, called from
//! `commands::catalog::list_laps`.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tauri::{Emitter, Manager};

use idl_rs::store::index_job::{
    self, DecodeBudget, IndexJobOptions, IndexJobReport, IndexPhase, IndexProgress,
};

use crate::error::IpcError;
use crate::state::{DataDir, IndexJob};

/// C3 §3.2 `index_progress` — the event payload, emitted as each session
/// enters each phase. Field names and `phase`'s two spellings are the
/// contract; they are what the import chip renders.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexProgressEvent {
    /// Sessions finished before this one.
    pub done: usize,
    /// Sessions this run is working through.
    pub total: usize,
    /// The session now being indexed.
    pub current_session_id: String,
    /// `"tracks"` (detecting visits and laps) or `"laps"` (writing the
    /// catalog rows).
    pub phase: &'static str,
}

impl From<&IndexProgress> for IndexProgressEvent {
    fn from(p: &IndexProgress) -> Self {
        Self {
            done: p.done,
            total: p.total,
            current_session_id: p.current_session_id.clone(),
            phase: p.phase.as_str(),
        }
    }
}

/// C3 §3.2 `index_status()`'s return — what the chip shows, whether or not
/// the frontend was mounted when the job started.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IndexStatus {
    /// True while the background job is working.
    pub running: bool,
    /// Sessions finished in the current (or last) run.
    pub done: usize,
    /// Sessions in the current (or last) run.
    pub total: usize,
    /// The session being indexed right now, `null` when idle.
    pub current_session_id: Option<String>,
    /// `"tracks"`/`"laps"` right now, `null` when idle.
    pub phase: Option<String>,
    /// The last finished run's counts, `null` before the first one.
    pub last_run: Option<IndexRunSummary>,
    /// Why the last run could not start at all (an unreadable
    /// `<data>/sessions/` or `<data>/tracks/`, a `catalog.sqlite` that will
    /// not open), `null` when the last run started. A background job has no
    /// promise to reject, so this is where its setup failure surfaces —
    /// without it a broken data root would look exactly like "nothing to
    /// do" (CLAUDE.md §5: never silence on bad data).
    pub last_error: Option<crate::error::IpcError>,
}

/// C3 §3.2: a finished run's counts.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IndexRunSummary {
    /// Sessions recomputed and written.
    pub indexed: usize,
    /// Sessions whose stamps were already current.
    pub skipped_up_to_date: usize,
    /// Sessions that could not be indexed; the run continued past each.
    pub failed: usize,
    /// True when the run stopped on a cancel rather than finishing.
    pub cancelled: bool,
}

impl From<&IndexJobReport> for IndexRunSummary {
    fn from(r: &IndexJobReport) -> Self {
        Self {
            indexed: r.indexed,
            skipped_up_to_date: r.skipped_up_to_date,
            failed: r.failed.len(),
            cancelled: r.cancelled,
        }
    }
}

/// C3 §3.2 `index_status()`. Managed state only — no filesystem, no
/// compute, so it stays a synchronous command (CLAUDE.md §3's R201 rule).
#[tauri::command]
pub fn index_status(job: tauri::State<'_, IndexJob>) -> IndexStatus {
    job.snapshot()
}

/// C3 §3.2 `cancel_index_job()`. Sets the flag the job polls between
/// sessions; every session already committed stays committed, and the next
/// run resumes from what is left (ruling R208.1 item 2). Returns whether a
/// run was actually in flight.
#[tauri::command]
pub fn cancel_index_job(job: tauri::State<'_, IndexJob>) -> bool {
    let was_running = job.snapshot().running;
    job.cancel.store(true, Ordering::Relaxed);
    was_running
}

/// C3 §3.2 `start_index_job()`. Starts the library-wide lap/track index in
/// the background and returns immediately with `true`, or `false` when a
/// run is already in flight (a second call is a no-op, not an error — the
/// app calls this on launch and `rebuild_catalog` calls it again).
///
/// The work runs on its own OS thread, not the async runtime: it is a
/// blocking, CPU-bound pool of `physical cores − 1` workers and must not
/// occupy runtime threads for minutes (CLAUDE.md §3, R201).
///
/// **This never returns `Err`**, and that is deliberate: the job outlives
/// the call, so a failure it hits after this has returned has no promise
/// left to reject. A run that cannot start at all lands in
/// [`IndexStatus::last_error`], which is where the UI reads it.
#[tauri::command(async)]
pub fn start_index_job<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> bool {
    spawn_index_job(app)
}

/// Starts the job if one is not already running, returning whether it
/// started. Shared by the `start_index_job` command and `rebuild_catalog`,
/// which starts one itself (ruling R207 item 1: library-wide indexing
/// follows a rebuild).
pub fn spawn_index_job<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> bool {
    {
        let job = app.state::<IndexJob>();
        if !job.try_claim() {
            return false;
        }
    }
    let data_root = app.state::<DataDir>().0.clone();

    // `std::thread`, not `tauri::async_runtime::spawn`: this is minutes of
    // blocking CPU work and would starve the runtime's worker threads.
    std::thread::spawn(move || {
        let report = run_index_job(&app, &data_root);
        let job = app.state::<IndexJob>();
        job.finish(report.as_ref().ok(), report.as_ref().err());
        // The terminal observation, in the same shape as every other one
        // (one event, one payload type): `done == total` with no session in
        // flight is how the chip knows the run is over. A run that could
        // not start at all reports 0 / 0, which is also "nothing left to
        // do" — its error is in the command's own result, not the event.
        let finished = report.as_ref().map_or(0, |r| r.total);
        let _ = app.emit(
            "index_progress",
            IndexProgressEvent {
                done: finished,
                total: finished,
                current_session_id: String::new(),
                phase: IndexPhase::Laps.as_str(),
            },
        );
    });
    true
}

/// The job body, on the background thread: reserve through the app's
/// `SessionCache` budget (ruling R208.1 item 3 — indexing decodes queue
/// behind the same ceiling as every command's), report every phase into
/// managed state and out as `index_progress`, and stop on the cancel flag.
fn run_index_job<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    data_root: &std::path::Path,
) -> Result<IndexJobReport, IpcError> {
    let cache = app.state::<crate::session_cache::SessionCache>();
    let job = app.state::<IndexJob>();
    let cancel: Arc<std::sync::atomic::AtomicBool> = job.cancel.clone();

    let on_progress = |p: IndexProgress| {
        let job = app.state::<IndexJob>();
        job.observe(&p);
        let _ = app.emit("index_progress", IndexProgressEvent::from(&p));
    };

    let budget: &dyn DecodeBudget = cache.inner();
    let opts = IndexJobOptions {
        workers: index_job::worker_count(),
        budget,
        cancel: &cancel,
        progress: &on_progress,
    };
    Ok(index_job::index_library(data_root, &opts)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_progress_event_carries_the_c3_phase_spellings_verbatim() {
        // Arrange
        let progress = IndexProgress {
            done: 11,
            total: 159,
            current_session_id: "2026-09-07_09-43-52".to_string(),
            phase: IndexPhase::Tracks,
        };

        // Act
        let event = IndexProgressEvent::from(&progress);

        // Assert
        assert_eq!(event.phase, "tracks");
        assert_eq!((event.done, event.total), (11, 159));
        assert_eq!(event.current_session_id, "2026-09-07_09-43-52");
    }

    #[test]
    fn index_run_summary_counts_failures_by_length_not_by_message() {
        // Arrange
        let report = IndexJobReport {
            total: 3,
            indexed: 1,
            skipped_up_to_date: 1,
            failed: vec![idl_rs::store::index_job::IndexFailure {
                session_id: "ghost".to_string(),
                message: "no data.parquet".to_string(),
            }],
            cancelled: false,
        };

        // Act
        let summary = IndexRunSummary::from(&report);

        // Assert
        assert_eq!((summary.indexed, summary.skipped_up_to_date, summary.failed), (1, 1, 1));
        assert!(!summary.cancelled);
    }
}
