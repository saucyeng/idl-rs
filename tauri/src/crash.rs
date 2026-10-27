//! Local crash reports (`docs/superpowers/specs/2026-10-21-idl1-crash-reports.md`,
//! C3 §3.13). A panic hook writes a text report into `<app data dir>/crash/`
//! before the default hook runs; the next launch lists the reports and the UI
//! offers to share them. Nothing leaves the device unless the user shares it.
//! A dismissed report moves to `crash/seen/`; nothing is deleted.
//!
//! A panic hook cannot see an allocation-failure abort or a kill by the OS,
//! so [`install`] also keeps a run marker (`crash/running.json`, the process
//! id and start time): written at startup, removed by [`mark_clean_exit`], and
//! found still in place, naming a dead process, by the next launch, which
//! reports it once as an abnormal exit (evaluation-memory spec §4.2.H).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{IpcError, IpcErrorKind};

/// At most this many reports are returned by [`pending`], newest first.
pub const MAX_REPORTS: usize = 10;

/// A report's text is cut to this many bytes when listed.
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

/// The sub-directory a dismissed report moves to.
const SEEN_DIR: &str = "seen";

/// The run marker's file name in the crash directory. Not a report name
/// (`<kind>-<ms>.txt`), so [`pending`] never lists it.
const MARKER_FILE: &str = "running.json";

/// The first line of the report an abnormal exit produces.
pub const ABNORMAL_EXIT_HEADLINE: &str = "idl1 closed unexpectedly (likely out of memory)";

/// What `running.json` holds: the process that wrote it. Present only while
/// that process is meant to be running.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunMarker {
    /// Operating-system process id of the run that wrote the marker.
    pub pid: u32,
    /// The app version that run was.
    pub version: String,
    /// When the run started, ms since the Unix epoch.
    pub started_at_ms: u64,
}

/// Where reports live (managed state).
pub struct CrashDir(pub PathBuf);

/// One pending report, as the UI sees it (C3 §3.13).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CrashReport {
    /// The report's file name in `crash/`.
    pub id: String,
    /// When it was written, ms since the Unix epoch.
    pub at_ms: u64,
    /// The report, at most [`MAX_REPORT_BYTES`].
    pub text: String,
}

/// The crash directory under `app_data_dir`.
pub fn dir_for(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("crash")
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The text of a panic report.
pub fn format_panic_report(version: &str, at_ms: u64, thread: &str, message: &str, location: &str, backtrace: &str) -> String {
    format!(
        "idl1 crash report (Rust panic)\n\
         app: {version} ({os}/{arch})\n\
         time: {at_ms} ms since the Unix epoch\n\
         thread: {thread}\n\
         panic: {message}\n\
         at: {location}\n\
         \n\
         backtrace:\n{backtrace}\n",
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
    )
}

/// The panic's message, whatever its payload type.
fn panic_message(info: &std::panic::PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(no message)".to_string()
    }
}

/// Installs the panic hook (once; later calls are ignored). Each panic writes
/// `panic-<ms>.txt` into `dir`, then runs the hook that was there before. A
/// report that can't be written is lost silently: the hook must never panic.
pub fn install(dir: PathBuf, version: String) {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    begin_run(&dir, &version);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let at_ms = now_ms();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let text = format_panic_report(&version, at_ms, &thread, &panic_message(info), &location, &backtrace);
        let _ = std::fs::write(dir.join(format!("panic-{at_ms}.txt")), text);
        previous(info);
    }));
}

/// Whether a process with id `pid` exists right now.
fn process_alive(pid: u32) -> bool {
    let pid = sysinfo::Pid::from_u32(pid);
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    sys.process(pid).is_some()
}

/// The text of an abnormal-exit report for the run `marker` describes.
pub fn format_abnormal_exit_report(marker: &RunMarker, noticed_at_ms: u64) -> String {
    format!(
        "{ABNORMAL_EXIT_HEADLINE}
         app: {version} ({os}/{arch})
         the previous run (pid {pid}) started {started} ms since the Unix epoch and left its run marker behind
         noticed: {noticed_at_ms} ms since the Unix epoch
         
         The process ended without a panic report or a clean shutdown: the operating system          killed it or an allocation failed. Nothing more is known.
",
        version = marker.version,
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        pid = marker.pid,
        started = marker.started_at_ms,
    )
}

/// Starts a run: reports a marker left by a dead process, then writes this
/// process's own marker. Called by [`install`]; separate so tests can drive it
/// against a temporary directory.
///
/// A marker naming this very process, or a process that is still alive, is
/// not reported (another copy of the app is running; or this is a second call
/// from the same process). Either way the marker is replaced, so a report is
/// made **once**. An unreadable marker is dropped silently, and nothing here
/// returns an error or panics: a failure to write is a lost breadcrumb, not a
/// reason to stop the app.
pub fn begin_run(dir: &Path, version: &str) {
    let marker_path = dir.join(MARKER_FILE);
    let now = now_ms();
    let own_pid = std::process::id();
    if let Some(old) = std::fs::read(&marker_path).ok().and_then(|b| serde_json::from_slice::<RunMarker>(&b).ok()) {
        if old.pid != own_pid && !process_alive(old.pid) {
            let text = format_abnormal_exit_report(&old, now);
            let _ = std::fs::write(dir.join(format!("abnormal-{now}.txt")), text);
        }
    }
    let marker = RunMarker { pid: own_pid, version: version.to_string(), started_at_ms: now };
    if let Ok(bytes) = serde_json::to_vec(&marker) {
        let _ = std::fs::write(&marker_path, bytes);
    }
}

/// Ends a run cleanly: removes the run marker so the next launch reports
/// nothing. The app calls this when the event loop exits (`RunEvent::Exit`);
/// a missing marker is not an error.
pub fn mark_clean_exit(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(MARKER_FILE));
}

/// The ms timestamp in a report's file name (`<kind>-<ms>.txt`).
fn report_ms(name: &str) -> Option<u64> {
    name.strip_suffix(".txt")?.rsplit_once('-')?.1.parse().ok()
}

/// Pending reports in `dir`, newest first, at most [`MAX_REPORTS`], each
/// text cut to [`MAX_REPORT_BYTES`]. A missing `dir` is no reports.
pub fn pending(dir: &Path) -> Result<Vec<CrashReport>, std::io::Error> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut found: Vec<(u64, String, PathBuf)> = Vec::new();
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(ms) = report_ms(&name) {
            found.push((ms, name, entry.path()));
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    found.truncate(MAX_REPORTS);
    let mut reports = Vec::with_capacity(found.len());
    for (at_ms, id, path) in found {
        let bytes = std::fs::read(&path)?;
        let cut = &bytes[..bytes.len().min(MAX_REPORT_BYTES)];
        reports.push(CrashReport { id, at_ms, text: String::from_utf8_lossy(cut).into_owned() });
    }
    Ok(reports)
}

/// Whether `id` is a plain file name: no separators, no `..`, not empty.
fn is_plain_name(id: &str) -> bool {
    !id.is_empty() && id != "." && id != ".." && !id.contains(['/', '\\']) && Path::new(id).file_name().is_some_and(|n| n == id)
}

/// Moves each report `ids` names to `dir/seen/`. An id already gone is not
/// an error; an id that is not a plain file name is `invalid_argument`.
pub fn dismiss(dir: &Path, ids: &[String]) -> Result<(), IpcError> {
    if let Some(bad) = ids.iter().find(|id| !is_plain_name(id)) {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("not a crash report id: {bad}")));
    }
    let seen = dir.join(SEEN_DIR);
    std::fs::create_dir_all(&seen).map_err(|e| IpcError::new(IpcErrorKind::Io, format!("creating {}: {e}", seen.display())))?;
    for id in ids {
        match std::fs::rename(dir.join(id), seen.join(id)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(IpcError::new(IpcErrorKind::Io, format!("moving crash report {id}: {e}"))),
        }
    }
    Ok(())
}

/// C3 §3.13 `crash_reports()` (added 2026-10-21): pending reports, newest
/// first. On Android the last run's exit reasons are collected first.
#[tauri::command(async)]
pub async fn crash_reports(dir: tauri::State<'_, CrashDir>) -> Result<Vec<CrashReport>, IpcError> {
    #[cfg(target_os = "android")]
    crate::mobile::crash_setup(&dir.0).await;
    pending(&dir.0).map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", dir.0.display())))
}

/// C3 §3.13 `dismiss_crash_reports(ids)` (added 2026-10-21): moves those
/// reports to `crash/seen/`.
#[tauri::command(async)]
pub async fn dismiss_crash_reports(dir: tauri::State<'_, CrashDir>, ids: Vec<String>) -> Result<(), IpcError> {
    dismiss(&dir.0, &ids)
}

/// C3 §3.13 `share_crash_report(text)` (added 2026-10-21): Android's share
/// sheet with `text`; `unsupported_platform` elsewhere.
#[tauri::command(async)]
pub async fn share_crash_report(text: String) -> Result<(), IpcError> {
    #[cfg(target_os = "android")]
    {
        crate::mobile::share_text("idl1 crash report", &text).await
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = text;
        Err(IpcError::new(IpcErrorKind::UnsupportedPlatform, "sharing is Android only; copy the report instead"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) {
        std::fs::write(dir.join(name), text).unwrap();
    }

    #[test]
    fn pending_reports_of_every_kind_newest_first_other_files_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "panic-1000.txt", "old");
        write(tmp.path(), "kotlin-3000.txt", "newest");
        write(tmp.path(), "exit-2000.txt", "middle");
        write(tmp.path(), "exit-seen", "3000");
        write(tmp.path(), "notes.txt", "not a report");

        let reports = pending(tmp.path()).unwrap();

        let ids: Vec<&str> = reports.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["kotlin-3000.txt", "exit-2000.txt", "panic-1000.txt"]);
        assert_eq!(reports[0].at_ms, 3000);
        assert_eq!(reports[0].text, "newest");
    }

    #[test]
    fn pending_more_than_the_cap_newest_kept_and_long_text_cut() {
        let tmp = tempfile::tempdir().unwrap();
        for ms in 0..(MAX_REPORTS as u64 + 3) {
            write(tmp.path(), &format!("panic-{ms}.txt"), "x");
        }
        write(tmp.path(), "panic-99999.txt", &"y".repeat(MAX_REPORT_BYTES + 10));

        let reports = pending(tmp.path()).unwrap();

        assert_eq!(reports.len(), MAX_REPORTS);
        assert_eq!(reports[0].text.len(), MAX_REPORT_BYTES);
        assert!(reports.iter().all(|r| r.at_ms >= 4));
    }

    #[test]
    fn pending_a_missing_directory_is_no_reports() {
        let tmp = tempfile::tempdir().unwrap();

        let reports = pending(&tmp.path().join("crash")).unwrap();

        assert!(reports.is_empty());
    }

    #[test]
    fn dismiss_moves_reports_to_seen_and_a_gone_id_is_fine() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "panic-1.txt", "a");

        dismiss(tmp.path(), &["panic-1.txt".to_string(), "panic-2.txt".to_string()]).unwrap();

        assert!(pending(tmp.path()).unwrap().is_empty());
        assert_eq!(std::fs::read_to_string(tmp.path().join("seen").join("panic-1.txt")).unwrap(), "a");
    }

    #[test]
    fn dismiss_a_path_for_an_id_is_invalid_argument_and_moves_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "panic-1.txt", "a");

        let result = dismiss(tmp.path(), &["panic-1.txt".to_string(), "../panic-1.txt".to_string()]);

        assert_eq!(result.unwrap_err().kind, IpcErrorKind::InvalidArgument);
        assert_eq!(pending(tmp.path()).unwrap().len(), 1);
    }

    #[test]
    fn crash_marker_left_by_a_dead_pid_reported_once_as_an_abnormal_exit() {
        // Arrange — a marker whose process cannot exist.
        let tmp = tempfile::tempdir().unwrap();
        let dead = RunMarker { pid: u32::MAX - 1, version: "0.3.0".to_string(), started_at_ms: 1000 };
        std::fs::write(tmp.path().join(MARKER_FILE), serde_json::to_vec(&dead).unwrap()).unwrap();

        // Act — two launches in a row.
        begin_run(tmp.path(), "0.4.0");
        begin_run(tmp.path(), "0.4.0");

        // Assert — one report, naming the run that died; the marker is now ours.
        let reports = pending(tmp.path()).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].text.starts_with(ABNORMAL_EXIT_HEADLINE));
        assert!(reports[0].text.contains(&format!("pid {}", u32::MAX - 1)));
        let marker: RunMarker = serde_json::from_slice(&std::fs::read(tmp.path().join(MARKER_FILE)).unwrap()).unwrap();
        assert_eq!(marker.pid, std::process::id());
        assert_eq!(marker.version, "0.4.0");
    }

    #[test]
    fn crash_marker_removed_on_a_clean_exit_next_launch_reports_nothing() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        begin_run(tmp.path(), "0.4.0");

        // Act
        mark_clean_exit(tmp.path());
        begin_run(tmp.path(), "0.4.0");

        // Assert
        assert!(pending(tmp.path()).unwrap().is_empty());
    }

    #[test]
    fn crash_marker_unreadable_is_dropped_without_a_report() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(MARKER_FILE), b"not json").unwrap();

        // Act
        begin_run(tmp.path(), "0.4.0");

        // Assert
        assert!(pending(tmp.path()).unwrap().is_empty());
    }

    #[test]
    fn format_panic_report_carries_version_thread_message_and_location() {
        let text = format_panic_report("0.4.0", 1234, "idl1-ble", "boom", "src/x.rs:7", "0: frame");

        assert!(text.contains("app: 0.4.0 ("));
        assert!(text.contains("time: 1234 ms"));
        assert!(text.contains("thread: idl1-ble"));
        assert!(text.contains("panic: boom"));
        assert!(text.contains("at: src/x.rs:7"));
        assert!(text.contains("backtrace:\n0: frame"));
    }
}
