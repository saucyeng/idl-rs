//! C3 §3.2 background catalog rebuild: `start_rebuild_job`,
//! `rebuild_status`, and the `rebuild_progress` event (ruling R219 items
//! 2–3).
//!
//! The work itself is `idl_rs::store::catalog::rebuild_catalog_with_progress`
//! — the same incremental C4 §5 scan the CLI runs. This module is the glue:
//! the background thread, the managed state a UI mounting mid-flight reads,
//! and the event.
//!
//! **No route awaits a rebuild** (ruling R219 item 3). The staged database
//! swaps in atomically at the end, so every reader keeps seeing the old
//! catalog until the new one lands — which is precisely why nothing has to
//! wait: the app opens on whatever the catalog already has and refreshes on
//! the run's completion event.

use tauri::{Emitter, Manager};

use idl_rs::store::catalog::{rebuild_catalog_with_progress, RebuildProgress, RebuildReport as CoreRebuildReport};

use crate::error::IpcError;
use crate::state::{DataDir, RebuildJob};

/// C3 §3.2 `rebuild_progress` — the event payload, emitted as each entity in
/// each C4 §5 phase is finished. Field names and `phase`'s five spellings
/// are the contract; they are what the status chip renders.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RebuildProgressEvent {
    /// Entities finished in this phase.
    pub done: usize,
    /// Entities this phase has to get through.
    pub total: usize,
    /// `"blobs"`, `"tracks"`, `"sessions"`, `"laps"` or `"workbooks"`.
    pub phase: &'static str,
    /// `true` on the one observation emitted after the staged database has
    /// swapped in, and on nothing else.
    ///
    /// This flag exists because `done == total` cannot mean "the run is
    /// over": **every** phase ends that way, and the last workbook of the
    /// last phase reaches it while the swap has still not happened. A
    /// listener keyed on the shape alone would call the run finished, then
    /// read a `rebuild_status()` whose `last_run` is still the previous
    /// run's.
    pub finished: bool,
}

impl From<&RebuildProgress> for RebuildProgressEvent {
    fn from(p: &RebuildProgress) -> Self {
        Self { done: p.done, total: p.total, phase: p.phase.as_str(), finished: false }
    }
}

/// C3 §3.2: a finished rebuild's counts. The four fields of the synchronous
/// `rebuild_catalog`'s `RebuildReport`, plus ruling R219 item 1's two blob
/// counts — how much of step 1 was carried over from the previous catalog
/// and how much had to be hashed.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RebuildRunSummary {
    pub sessions_indexed: u32,
    pub workbooks_indexed: u32,
    pub tracks_indexed: u32,
    /// Blobs copied from the previous catalog without re-hashing.
    pub blobs_carried: u32,
    /// Blobs actually read and hashed this run.
    pub blobs_hashed: u32,
    /// Wall-clock milliseconds the run took.
    pub duration_ms: u64,
}

impl RebuildRunSummary {
    /// The core report plus the wall-clock time the caller measured (C4 §5's
    /// rebuild has no `duration_ms` of its own).
    pub fn from_report(report: &CoreRebuildReport, duration_ms: u64) -> Self {
        Self {
            sessions_indexed: report.sessions_indexed as u32,
            workbooks_indexed: report.workbooks_indexed as u32,
            tracks_indexed: report.tracks_indexed as u32,
            blobs_carried: report.blobs_carried as u32,
            blobs_hashed: report.blobs_hashed as u32,
            duration_ms,
        }
    }
}

/// C3 §3.2 `rebuild_status()`'s return — what the chip shows, whether or not
/// the frontend was mounted when the job started.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RebuildStatus {
    /// True while the background rebuild is working.
    pub running: bool,
    /// Entities finished in the current (or last) phase.
    pub done: usize,
    /// Entities in the current (or last) phase.
    pub total: usize,
    /// The phase running right now, `null` when idle.
    pub phase: Option<String>,
    /// The last finished run's counts, `null` before the first one.
    pub last_run: Option<RebuildRunSummary>,
    /// Why the last run failed outright (an unreadable `<data>`, a staging
    /// database that will not open), `null` when it finished. A background
    /// job has no promise to reject, so this is where its failure surfaces —
    /// without it a broken data root would look exactly like a rebuild that
    /// found nothing (CLAUDE.md §5).
    pub last_error: Option<IpcError>,
}

/// C3 §3.2 `rebuild_status()`. Managed state only — no filesystem, no
/// compute, so it stays a synchronous command (CLAUDE.md §3's R201 rule).
#[tauri::command]
pub fn rebuild_status(job: tauri::State<'_, RebuildJob>) -> RebuildStatus {
    job.snapshot()
}

/// C3 §3.2 `start_rebuild_job()`. Starts the incremental catalog rebuild in
/// the background and returns immediately with `true`, or `false` when a run
/// is already in flight (a second call is a no-op, not an error — the
/// notebook's empty state and the maintenance panel both call it).
///
/// The work runs on its own OS thread, not the async runtime: it is a
/// blocking scan of the whole data directory and must not occupy runtime
/// threads (CLAUDE.md §3, R201).
///
/// **This never returns `Err`**, deliberately: the job outlives the call, so
/// a failure it hits after this has returned has no promise left to reject.
/// A run that fails lands in [`RebuildStatus::last_error`], which is where
/// the UI reads it.
#[tauri::command(async)]
pub fn start_rebuild_job<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> bool {
    spawn_rebuild_job(app)
}

/// Starts the rebuild if one is not already running, returning whether it
/// started. Shared by the `start_rebuild_job` command and any future caller
/// that wants a rebuild without waiting for one.
pub fn spawn_rebuild_job<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> bool {
    {
        let job = app.state::<RebuildJob>();
        if !job.try_claim() {
            return false;
        }
    }
    let data_root = app.state::<DataDir>().0.clone();

    // `std::thread`, not `tauri::async_runtime::spawn`: this is a blocking
    // whole-tree scan and would starve the runtime's worker threads.
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let on_progress = |p: RebuildProgress| {
            let job = app.state::<RebuildJob>();
            job.observe(&p);
            let _ = app.emit("rebuild_progress", RebuildProgressEvent::from(&p));
        };
        let outcome = rebuild_catalog_with_progress(&data_root, &on_progress).map_err(IpcError::from);
        let duration_ms = started.elapsed().as_millis() as u64;
        let summary = outcome.as_ref().ok().map(|r| RebuildRunSummary::from_report(r, duration_ms));

        let job = app.state::<RebuildJob>();
        job.finish(summary, outcome.as_ref().err());

        // The terminal observation — the only one carrying `finished: true`,
        // emitted after `job.finish` above, so a listener that acts on it can
        // read this run's own counts from `rebuild_status()` rather than the
        // previous run's. Its `done`/`total` are the last phase's, kept only
        // so the payload shape never varies; a failed run reports 0 / 0 and
        // its reason is in `rebuild_status().last_error`, not in the event.
        let workbooks = outcome.as_ref().map_or(0, |r| r.workbooks_indexed);
        let _ = app.emit(
            "rebuild_progress",
            RebuildProgressEvent {
                done: workbooks,
                total: workbooks,
                phase: idl_rs::store::catalog::RebuildPhase::Workbooks.as_str(),
                finished: true,
            },
        );

        // A freshly rebuilt catalog has no laps to copy until the library
        // index job has run (ruling R207 item 1) — the same follow-on the
        // synchronous `rebuild_catalog` command starts, kept here so the
        // background path is not the weaker one.
        crate::commands::index::spawn_index_job(app);
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::store::catalog::RebuildPhase;

    #[test]
    fn rebuild_progress_event_carries_the_c3_phase_spellings_verbatim() {
        // Arrange
        let progress = RebuildProgress { phase: RebuildPhase::Sessions, done: 12, total: 159 };

        // Act
        let event = RebuildProgressEvent::from(&progress);

        // Assert
        assert_eq!(event.phase, "sessions");
        assert_eq!((event.done, event.total), (12, 159));
        assert!(!event.finished);
    }

    #[test]
    fn rebuild_progress_event_a_phases_last_item_is_not_the_runs_end() {
        // Arrange -- the final workbook of the final phase: `done == total`,
        // but the staged database has not swapped in yet.
        let progress = RebuildProgress { phase: RebuildPhase::Workbooks, done: 3, total: 3 };

        // Act
        let event = RebuildProgressEvent::from(&progress);

        // Assert
        assert!(!event.finished);
    }

    #[test]
    fn rebuild_run_summary_carries_both_blob_counts_from_the_core_report() {
        // Arrange
        let report = CoreRebuildReport {
            blobs_indexed: 159,
            blobs_carried: 158,
            blobs_hashed: 1,
            tracks_indexed: 0,
            sessions_indexed: 159,
            laps_indexed: 0,
            lap_summary_indexed: 0,
            workbooks_indexed: 3,
            skipped: Vec::new(),
        };

        // Act
        let summary = RebuildRunSummary::from_report(&report, 420);

        // Assert
        assert_eq!((summary.blobs_carried, summary.blobs_hashed), (158, 1));
        assert_eq!((summary.sessions_indexed, summary.workbooks_indexed, summary.duration_ms), (159, 3, 420));
    }
}
