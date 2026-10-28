//! Local crash reports (`docs/superpowers/specs/2026-10-21-idl1-crash-reports.md`,
//! C3 §3.13). A panic hook writes a text report into `<app data dir>/crash/`
//! before the default hook runs; the next launch lists the reports and the UI
//! offers to share them. Nothing leaves the device unless the user shares it.
//! A dismissed report moves to `crash/seen/`; nothing is deleted.
//!
//! A panic hook cannot see an allocation-failure abort or a kill by the OS,
//! so [`install`] also keeps a run marker (`crash/running-<pid>.json`, the process
//! id and start time): written at startup, removed by [`mark_clean_exit`], and
//! found still in place, naming a dead process, by the next launch, which
//! reports it once as an abnormal exit (evaluation-memory spec §4.2.H).

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{IpcError, IpcErrorKind};

/// At most this many reports are returned by [`pending`], newest first.
pub const MAX_REPORTS: usize = 10;

/// A report's text is cut to this many bytes when listed.
pub const MAX_REPORT_BYTES: usize = 64 * 1024;

/// The sub-directory a dismissed report moves to.
const SEEN_DIR: &str = "seen";

/// Prefix of a run marker's file name in the crash directory
/// (`running-<pid>.json`). Not a report name (`<kind>-<ms>.txt`), so
/// [`pending`] never lists one.
const MARKER_PREFIX: &str = "running-";

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
    open_breadcrumb(&dir);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let at_ms = now_ms();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let text = format_panic_report(&version, at_ms, &thread, &panic_message(info), &location, &backtrace);
        let _ = std::fs::write(dir.join(format!("panic-{at_ms}.txt")), text);
        // The panic report says what happened; a leftover marker would add a
        // second, wrong "closed unexpectedly" report at the next launch.
        mark_clean_exit(&dir);
        previous(info);
    }));
}

// ── the allocation breadcrumb (evaluation-memory spec §4.2.H) ────────────

/// Prefix of the file the allocator writes when an allocation fails
/// (`alloc-<pid>.log`). Not a report name (`<kind>-<ms>.txt`), so [`pending`]
/// never lists it; [`begin_run`] folds it into the abnormal-exit report of
/// the run that wrote it.
const BREADCRUMB_PREFIX: &str = "alloc-";

/// Bytes the Rust heap holds right now, kept by [`BreadcrumbAllocator`].
static LIVE_HEAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// The open breadcrumb file, set once by [`open_breadcrumb`]. Opened ahead
/// of time because writing to it must not allocate.
static BREADCRUMB: OnceLock<std::fs::File> = OnceLock::new();

/// Bytes the Rust heap holds right now, as [`BreadcrumbAllocator`] counts
/// them; `0` when it is not the process's global allocator (tests, the CLI).
/// The memory budget credits this as the process's own commit.
pub fn live_heap_bytes() -> u64 {
    LIVE_HEAP_BYTES.load(Ordering::Relaxed)
}

/// The app's global allocator: the system allocator, counting live bytes,
/// and leaving a breadcrumb when an allocation fails.
///
/// A failed allocation does not panic, so the panic hook never sees it:
/// Rust aborts the process. Before returning the null that causes that
/// abort, this writes `alloc of N bytes failed, live=M` to the breadcrumb
/// file opened at startup, with a stack buffer and no allocation, so the
/// next launch's abnormal-exit report can say what killed the run. The
/// `static` lives in the app crate (`#[global_allocator]`).
pub struct BreadcrumbAllocator;

// SAFETY: every call forwards to `System` unchanged; the counter only
// observes sizes, and the failure path writes from a stack buffer to a file
// opened earlier, never allocating.
unsafe impl GlobalAlloc for BreadcrumbAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        record(p, layout.size());
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        record(p, layout.size());
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE_HEAP_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if p.is_null() {
            write_breadcrumb(new_size);
        } else {
            LIVE_HEAP_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            LIVE_HEAP_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        p
    }
}

/// Counts a successful allocation of `size` bytes, or leaves the breadcrumb
/// for a failed one.
fn record(p: *mut u8, size: usize) {
    if p.is_null() {
        write_breadcrumb(size);
    } else {
        LIVE_HEAP_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

/// Writes `alloc of <size> bytes failed, live=<live>\n` to the breadcrumb
/// file, if one is open. Allocation-free.
fn write_breadcrumb(size: usize) {
    use std::io::Write;
    let Some(mut file) = BREADCRUMB.get() else { return };
    let mut buf = [0u8; 96];
    let n = format_breadcrumb(&mut buf, size as u64, live_heap_bytes());
    let _ = file.write_all(&buf[..n]);
}

/// The breadcrumb line for a failed allocation of `size` bytes with `live`
/// bytes held, into `buf`; returns its length. Allocation-free.
fn format_breadcrumb(buf: &mut [u8; 96], size: u64, live: u64) -> usize {
    let mut n = 0;
    for part in [Part::Text(b"alloc of "), Part::Num(size), Part::Text(b" bytes failed, live="), Part::Num(live), Part::Text(b"\n")] {
        match part {
            Part::Text(t) => {
                buf[n..n + t.len()].copy_from_slice(t);
                n += t.len();
            }
            Part::Num(mut v) => {
                let mut digits = [0u8; 20];
                let mut d = 0;
                loop {
                    digits[d] = b'0' + (v % 10) as u8;
                    d += 1;
                    v /= 10;
                    if v == 0 {
                        break;
                    }
                }
                for i in (0..d).rev() {
                    buf[n] = digits[i];
                    n += 1;
                }
            }
        }
    }
    n
}

/// One piece of [`format_breadcrumb`]'s line.
enum Part {
    Text(&'static [u8]),
    Num(u64),
}

/// The breadcrumb file for process `pid`.
fn breadcrumb_path(dir: &Path, pid: u32) -> PathBuf {
    dir.join(format!("{BREADCRUMB_PREFIX}{pid}.log"))
}

/// Opens this process's breadcrumb file for [`BreadcrumbAllocator`] to write
/// to. Called by [`install`]; a failure to open leaves the allocator without
/// a breadcrumb, never stops the app.
fn open_breadcrumb(dir: &Path) {
    if let Ok(file) = std::fs::File::create(breadcrumb_path(dir, std::process::id())) {
        let _ = BREADCRUMB.set(file);
    }
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
        "{ABNORMAL_EXIT_HEADLINE}\n\
         app: {version} ({os}/{arch})\n\
         the previous run (pid {pid}) started {started} ms since the Unix epoch and left its run marker behind\n\
         noticed: {noticed_at_ms} ms since the Unix epoch\n\
         \n\
         The process ended without a panic report or a clean shutdown: the operating system killed it \
         or an allocation failed. Nothing more is known.\n",
        version = marker.version,
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        pid = marker.pid,
        started = marker.started_at_ms,
    )
}

/// The marker file for process `pid`. One per process, so two running copies
/// of the app never overwrite each other's.
fn marker_path(dir: &Path, pid: u32) -> PathBuf {
    dir.join(format!("{MARKER_PREFIX}{pid}.json"))
}

/// Starts a run: reports every marker left by a dead process of this same
/// version, then writes this process's own marker. Called by [`install`];
/// separate so tests can drive it against a temporary directory.
///
/// Every `running-<pid>.json` in `dir` is examined. A marker is deleted
/// without a report when it names this process, a process that is still alive
/// (another copy of the app is running), or a different app version (an update
/// or restart, not a crash). Otherwise it is reported once as an abnormal exit
/// and deleted, so it cannot be reported again. An unreadable marker is
/// deleted silently. Nothing here returns an error or panics: a failure to
/// write is a lost breadcrumb, not a reason to stop the app.
///
/// The new marker is written to a temporary file and renamed into place, so a
/// kill mid-write cannot leave a half-written marker.
pub fn begin_run(dir: &Path, version: &str) {
    let now = now_ms();
    let own_pid = std::process::id();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !(name.starts_with(MARKER_PREFIX) && name.ends_with(".json")) {
                continue;
            }
            let path = entry.path();
            let marker = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<RunMarker>(&b).ok());
            if let Some(old) = marker {
                if old.pid == own_pid || process_alive(old.pid) {
                    continue;
                }
                if old.version == version {
                    let mut text = format_abnormal_exit_report(&old, now);
                    if let Ok(crumb) = std::fs::read_to_string(breadcrumb_path(dir, old.pid)) {
                        if !crumb.is_empty() {
                            text.push_str(&format!("\nThe allocator's last word: {crumb}"));
                        }
                    }
                    let _ = std::fs::write(dir.join(format!("abnormal{}-{now}.txt", old.pid)), text);
                }
                let _ = std::fs::remove_file(breadcrumb_path(dir, old.pid));
            }
            let _ = std::fs::remove_file(&path);
        }
    }
    let marker = RunMarker { pid: own_pid, version: version.to_string(), started_at_ms: now };
    if let Ok(bytes) = serde_json::to_vec(&marker) {
        let target = marker_path(dir, own_pid);
        let tmp = target.with_extension("json.tmp");
        if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &target).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Ends a run cleanly: removes this process's run marker so the next launch
/// reports nothing. The app calls this when the event loop exits
/// (`RunEvent::Exit`); the panic hook calls it after writing its report. A
/// missing marker is not an error.
pub fn mark_clean_exit(dir: &Path) {
    let _ = std::fs::remove_file(marker_path(dir, std::process::id()));
    let _ = std::fs::remove_file(breadcrumb_path(dir, std::process::id()));
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
        let dead = RunMarker { pid: u32::MAX - 1, version: "0.4.0".to_string(), started_at_ms: 1000 };
        std::fs::write(marker_path(tmp.path(), dead.pid), serde_json::to_vec(&dead).unwrap()).unwrap();

        // Act — two launches in a row.
        begin_run(tmp.path(), "0.4.0");
        begin_run(tmp.path(), "0.4.0");

        // Assert — one report, naming the run that died; the marker is now ours.
        let reports = pending(tmp.path()).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].text.starts_with(ABNORMAL_EXIT_HEADLINE));
        assert!(reports[0].text.contains(&format!("pid {}", u32::MAX - 1)));
        let marker: RunMarker = serde_json::from_slice(&std::fs::read(marker_path(tmp.path(), std::process::id())).unwrap()).unwrap();
        assert_eq!(marker.pid, std::process::id());
        assert_eq!(marker.version, "0.4.0");
    }

    #[test]
    fn crash_marker_left_by_a_dead_pid_of_another_version_is_an_update_not_a_crash() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let old = RunMarker { pid: u32::MAX - 1, version: "0.3.0".to_string(), started_at_ms: 1000 };
        std::fs::write(marker_path(tmp.path(), old.pid), serde_json::to_vec(&old).unwrap()).unwrap();

        // Act
        begin_run(tmp.path(), "0.4.0");

        // Assert
        assert!(pending(tmp.path()).unwrap().is_empty());
        let marker: RunMarker = serde_json::from_slice(&std::fs::read(marker_path(tmp.path(), std::process::id())).unwrap()).unwrap();
        assert_eq!(marker.version, "0.4.0");
    }

    #[test]
    fn crash_marker_two_dead_runs_each_leave_their_own_marker_and_both_are_reported() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        for pid in [u32::MAX - 1, u32::MAX - 2] {
            let m = RunMarker { pid, version: "0.4.0".to_string(), started_at_ms: 1 };
            std::fs::write(marker_path(tmp.path(), pid), serde_json::to_vec(&m).unwrap()).unwrap();
        }

        // Act
        begin_run(tmp.path(), "0.4.0");

        // Assert
        assert_eq!(pending(tmp.path()).unwrap().len(), 2);
        assert!(!marker_path(tmp.path(), u32::MAX - 1).exists());
        assert!(marker_path(tmp.path(), std::process::id()).exists());
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
        std::fs::write(marker_path(tmp.path(), 12345), b"not json").unwrap();

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

    #[test]
    fn format_breadcrumb_a_failed_allocation_names_its_size_and_the_live_heap() {
        // Arrange
        let mut buf = [0u8; 96];

        // Act
        let n = format_breadcrumb(&mut buf, 1_694_793_000, 0);

        // Assert
        assert_eq!(&buf[..n], b"alloc of 1694793000 bytes failed, live=0\n");
    }

    #[test]
    fn begin_run_a_dead_runs_breadcrumb_is_folded_into_its_abnormal_exit_report() {
        // Arrange
        let dir = std::env::temp_dir().join(format!("idl1-crumb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let dead = RunMarker { pid: u32::MAX - 7, version: "9.9.9".to_string(), started_at_ms: 1 };
        std::fs::write(marker_path(&dir, dead.pid), serde_json::to_vec(&dead).unwrap()).unwrap();
        std::fs::write(breadcrumb_path(&dir, dead.pid), "alloc of 4096 bytes failed, live=123\n").unwrap();

        // Act
        begin_run(&dir, "9.9.9");

        // Assert
        let reports = pending(&dir).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].text.contains("alloc of 4096 bytes failed, live=123"));
        assert!(!breadcrumb_path(&dir, dead.pid).exists());

        mark_clean_exit(&dir);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
