//! `<data>/inbox` — the drop folder (C4 §2, ruling R191). Desktop only.
//!
//! A file dropped into `<data>/inbox` is imported through the same path
//! `import_file` uses, then **deleted**: its bytes are already in `blobs/`,
//! content-addressed, so the copy in the inbox is redundant (R191 — the one
//! irreversible choice here, safe precisely because the import succeeded).
//! A file that fails to import moves to `inbox/failed/<name>` beside a
//! `<name>.error.txt` holding the C3 §2 `IpcError` as JSON, and is never
//! retried automatically.
//!
//! A file counts as ready only after two seconds of unchanged size (R191) —
//! a large `.idl0` copied in over a slow link must not be imported
//! half-written. This is a second watcher struct, deliberately not a
//! generalisation of [`crate::watcher::WorkbookWatcher`] (R191), whose
//! self-write suppression and hash bookkeeping answer a different question.
//!
//! The inbox is not synced, not catalogued and ignored by repair: nothing
//! outside this module ever looks at `<data>/inbox`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::{RecursiveMode, Watcher};

use crate::error::{IpcError, IpcErrorKind};

/// How long a file's size must stay unchanged before it is imported (R191).
pub const SETTLE: Duration = Duration::from_secs(2);

/// How long to keep waiting for a file that never settles (a partial copy
/// that was abandoned): the file is left in the inbox untouched and picked
/// up again by the next launch scan.
const SETTLE_GIVE_UP: Duration = Duration::from_secs(600);

/// Directory name, inside the inbox, holding files that failed to import.
const FAILED_DIR: &str = "failed";

/// One failed inbox file (C3 §3.3 `InboxStatus.failed`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct InboxFailure {
    /// The failed file's name as it now sits in `inbox/failed/`.
    pub file_name: String,
    /// The error its import returned, as recorded in `<name>.error.txt`.
    pub error: IpcError,
}

/// C3 §3.3 `InboxStatus`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InboxStatus {
    /// Absolute path of `<data>/inbox` (C4 §2).
    pub path: String,
    /// Files this process has imported from the inbox since launch,
    /// including the launch scan's own.
    pub imported_since_launch: u32,
    /// Everything currently sitting in `inbox/failed/`.
    pub failed: Vec<InboxFailure>,
}

/// Live inbox state: the folder, the running watcher, and the counter
/// `inbox_status` reports. Managed by the app crate at startup
/// (`app.manage(...)`) and read by the `inbox_status` command; dropping it
/// stops the watcher.
pub struct InboxState {
    /// `<data>/inbox`.
    pub path: PathBuf,
    /// Imports completed since this struct was built.
    imported_since_launch: Arc<Mutex<u32>>,
    /// Paths currently being settled or imported, so a burst of filesystem
    /// events for one file starts exactly one import.
    in_flight: Arc<Mutex<HashSet<PathBuf>>>,
    /// Kept alive for the process lifetime — `notify`'s watcher stops on
    /// drop.
    _watcher: Option<notify::RecommendedWatcher>,
}

impl InboxState {
    /// Creates `<data>/inbox` (and `inbox/failed/`) if absent, scans it once
    /// for files already sitting there, and starts watching it. Errors only
    /// when the folder cannot be created or watched; a file that fails to
    /// import is recorded under `failed/`, never returned from here.
    pub fn start(data_dir: &Path) -> std::io::Result<Self> {
        Self::start_with_settle(data_dir, SETTLE)
    }

    /// [`Self::start`] with the stable-size window as an argument — the
    /// production value is [`SETTLE`]; tests pass a few milliseconds so a
    /// test does not sleep for two seconds per file.
    pub fn start_with_settle(data_dir: &Path, settle: Duration) -> std::io::Result<Self> {
        let path = data_dir.join("inbox");
        std::fs::create_dir_all(path.join(FAILED_DIR))?;

        let imported_since_launch = Arc::new(Mutex::new(0u32));
        let in_flight: Arc<Mutex<HashSet<PathBuf>>> = Arc::new(Mutex::new(HashSet::new()));

        // Launch scan (R191): everything already in the folder, handled
        // exactly as a freshly dropped file is.
        for entry in std::fs::read_dir(&path)?.flatten() {
            let file = entry.path();
            if file.is_file() {
                spawn_handler(data_dir, file, settle, &imported_since_launch, &in_flight);
            }
        }

        let data_dir = data_dir.to_path_buf();
        let counter = Arc::clone(&imported_since_launch);
        let flight = Arc::clone(&in_flight);
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let Ok(event) = res else { return };
            if !matches!(event.kind, notify::EventKind::Create(_) | notify::EventKind::Modify(_)) {
                return;
            }
            for file in event.paths {
                if file.is_file() {
                    spawn_handler(&data_dir, file, settle, &counter, &flight);
                }
            }
        })
        .map_err(|e| std::io::Error::other(format!("watching {}: {e}", path.display())))?;
        // Non-recursive: `failed/` is this module's own output and must
        // never be re-imported (R191 — failures are never auto-retried).
        watcher
            .watch(&path, RecursiveMode::NonRecursive)
            .map_err(|e| std::io::Error::other(format!("watching {}: {e}", path.display())))?;

        Ok(Self { path, imported_since_launch, in_flight, _watcher: Some(watcher) })
    }

    /// C3 §3.3 `inbox_status()` — the folder, the count of imports since
    /// launch, and everything currently in `inbox/failed/`.
    pub fn status(&self) -> InboxStatus {
        InboxStatus {
            path: self.path.to_string_lossy().into_owned(),
            imported_since_launch: *self.imported_since_launch.lock().unwrap(),
            failed: read_failures(&self.path),
        }
    }

    /// True while at least one dropped file is still settling or importing.
    /// The seam this module's own tests wait on, and the honest answer to
    /// "is the inbox idle" for any future caller that needs one.
    pub fn busy(&self) -> bool {
        !self.in_flight.lock().unwrap().is_empty()
    }
}

/// Starts one background handler for `file` unless one is already running
/// for that exact path (a single copy raises several `Create`/`Modify`
/// events). Skips anything inside `failed/` and the `.error.txt` sidecars.
fn spawn_handler(
    data_dir: &Path,
    file: PathBuf,
    settle: Duration,
    imported_since_launch: &Arc<Mutex<u32>>,
    in_flight: &Arc<Mutex<HashSet<PathBuf>>>,
) {
    if file.parent().and_then(|p| p.file_name()).is_some_and(|n| n == FAILED_DIR) {
        return;
    }
    if !in_flight.lock().unwrap().insert(file.clone()) {
        return;
    }

    let data_dir = data_dir.to_path_buf();
    let counter = Arc::clone(imported_since_launch);
    let flight = Arc::clone(in_flight);
    std::thread::spawn(move || {
        if wait_until_stable(&file, settle) {
            match import_one(&data_dir, &file) {
                Ok(()) => *counter.lock().unwrap() += 1,
                Err(e) => move_to_failed(&data_dir.join("inbox"), &file, &e),
            }
        }
        flight.lock().unwrap().remove(&file);
    });
}

/// Blocks until `file`'s size has been unchanged for `settle`, returning
/// `true` when it settled and `false` when the file vanished (someone took
/// it away again) or never settled within [`SETTLE_GIVE_UP`].
fn wait_until_stable(file: &Path, settle: Duration) -> bool {
    let deadline = std::time::Instant::now() + SETTLE_GIVE_UP;
    let mut last = match std::fs::metadata(file) {
        Ok(m) => m.len(),
        Err(_) => return false,
    };
    while std::time::Instant::now() < deadline {
        std::thread::sleep(settle);
        let now = match std::fs::metadata(file) {
            Ok(m) => m.len(),
            Err(_) => return false,
        };
        if now == last {
            return true;
        }
        last = now;
    }
    false
}

/// Imports one settled inbox file through the same core entry points
/// `import_file` uses, then deletes it (R191). The catalog row is updated
/// incrementally by core's import pipeline; nothing else here touches it.
fn import_one(data_dir: &Path, file: &Path) -> Result<(), IpcError> {
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .ok_or_else(|| {
            IpcError::new(
                IpcErrorKind::InvalidArgument,
                format!("{} has no file extension to choose an importer from", file.display()),
            )
        })?;
    if ext != "idl0" && idl_rs::import::importer_for_extension(&ext).is_none() {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("no importer covers '.{ext}' files")));
    }

    let bytes = std::fs::read(file)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", file.display())))?;
    if ext == "idl0" {
        idl_rs::store::import::import_idl0(data_dir, &bytes)
    } else {
        idl_rs::store::import::import_file(data_dir, &ext, &bytes)
    }
    .map_err(IpcError::from)?;

    std::fs::remove_file(file)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("removing {}: {e}", file.display())))?;
    Ok(())
}

/// Moves a failed file to `inbox/failed/<name>` and writes
/// `inbox/failed/<name>.error.txt` beside it (R191). Best-effort: a failure
/// to move leaves the file where it is, to be retried at the next launch
/// scan — the alternative (deleting it) would lose the user's data.
fn move_to_failed(inbox: &Path, file: &Path, error: &IpcError) {
    let failed_dir = inbox.join(FAILED_DIR);
    if std::fs::create_dir_all(&failed_dir).is_err() {
        return;
    }
    let Some(name) = file.file_name() else { return };
    let target = failed_dir.join(name);
    if std::fs::rename(file, &target).is_err() {
        return;
    }
    let sidecar = failed_dir.join(format!("{}.error.txt", name.to_string_lossy()));
    let json = serde_json::to_string_pretty(error).unwrap_or_else(|_| error.message.clone());
    let _ = std::fs::write(sidecar, json);
}

/// Reads `inbox/failed/` back into C3's `failed` list: one entry per file
/// that is not itself an `.error.txt` sidecar, paired with the error its
/// sidecar records. A missing or unreadable sidecar reads as `internal` —
/// the file did fail, the reason is just no longer on disk.
fn read_failures(inbox: &Path) -> Vec<InboxFailure> {
    let mut out: Vec<InboxFailure> = std::fs::read_dir(inbox.join(FAILED_DIR))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && !p.to_string_lossy().ends_with(".error.txt"))
        .map(|p| {
            let file_name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            let sidecar = p.with_file_name(format!("{file_name}.error.txt"));
            let error = std::fs::read_to_string(&sidecar)
                .ok()
                .and_then(|s| serde_json::from_str::<IpcError>(&s).ok())
                .unwrap_or_else(|| {
                    IpcError::new(IpcErrorKind::Internal, format!("{file_name} failed to import; no error record kept"))
                });
            InboxFailure { file_name, error }
        })
        .collect();
    out.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::parse::test_buffers::{cat, frame, imu_payload, session_end, v3_registry_entry, Header};

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-inbox-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn idl0_bytes() -> Vec<u8> {
        let accel: f32 = 32.0 / 32768.0;
        cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }
                .build(&[v3_registry_entry(0, 4, 800, accel, 0.0, "IMU0_AccelX", "g")]),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ])
    }

    /// Blocks until nothing is in flight, or the timeout expires. Polling a
    /// background thread's own completion flag, not a fixed sleep.
    fn wait_idle(state: &InboxState) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
            if !state.busy() {
                return;
            }
        }
        panic!("inbox handlers did not finish within 10 s");
    }

    #[test]
    fn inbox_a_dropped_idl0_file_is_imported_counted_and_deleted() {
        // Arrange
        let root = temp_root();
        let state = InboxState::start_with_settle(&root, Duration::from_millis(20)).unwrap();
        let dropped = state.path.join("ride.idl0");

        // Act
        std::fs::write(&dropped, idl0_bytes()).unwrap();
        // The watcher's event may arrive after this write returns; give the
        // handler a chance to register before waiting for idleness.
        std::thread::sleep(Duration::from_millis(200));
        wait_idle(&state);

        // Assert — the session landed, the inbox copy is gone (its bytes are
        // in `blobs/`), and the counter moved.
        assert!(!dropped.exists());
        assert_eq!(state.status().imported_since_launch, 1);
        assert_eq!(idl_rs::store::catalog_read::rebuild_catalog_report(&root).unwrap().sessions_indexed, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn inbox_a_file_present_at_launch_is_imported_by_the_launch_scan() {
        // Arrange — the file is there before the watcher ever starts.
        let root = temp_root();
        std::fs::create_dir_all(root.join("inbox")).unwrap();
        std::fs::write(root.join("inbox").join("ride.idl0"), idl0_bytes()).unwrap();

        // Act
        let state = InboxState::start_with_settle(&root, Duration::from_millis(20)).unwrap();
        wait_idle(&state);

        // Assert
        assert_eq!(state.status().imported_since_launch, 1);
        assert!(!root.join("inbox").join("ride.idl0").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn inbox_a_file_that_fails_to_import_moves_to_failed_with_an_error_sidecar() {
        // Arrange
        let root = temp_root();
        let state = InboxState::start_with_settle(&root, Duration::from_millis(20)).unwrap();

        // Act — valid extension, invalid content.
        std::fs::write(state.path.join("broken.idl0"), b"not an idl0 file at all").unwrap();
        std::thread::sleep(Duration::from_millis(200));
        wait_idle(&state);

        // Assert — moved, sidecar written, and reported by `status()` with
        // the kind the importer raised.
        let status = state.status();
        assert_eq!(status.imported_since_launch, 0);
        assert_eq!(status.failed.len(), 1);
        assert_eq!(status.failed[0].file_name, "broken.idl0");
        assert_eq!(status.failed[0].error.kind, IpcErrorKind::ParseInvalidMagicBytes);
        assert!(state.path.join("failed").join("broken.idl0").is_file());
        assert!(state.path.join("failed").join("broken.idl0.error.txt").is_file());
        assert!(!state.path.join("broken.idl0").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn inbox_a_file_no_importer_covers_fails_with_invalid_argument() {
        // Arrange
        let root = temp_root();
        let state = InboxState::start_with_settle(&root, Duration::from_millis(20)).unwrap();

        // Act
        std::fs::write(state.path.join("notes.txt"), b"just some notes").unwrap();
        std::thread::sleep(Duration::from_millis(200));
        wait_idle(&state);

        // Assert
        let status = state.status();
        assert_eq!(status.failed.len(), 1);
        assert_eq!(status.failed[0].error.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn inbox_status_on_a_fresh_data_dir_is_empty_and_names_the_inbox_path() {
        // Arrange
        let root = temp_root();

        // Act
        let state = InboxState::start_with_settle(&root, Duration::from_millis(20)).unwrap();
        let status = state.status();

        // Assert
        assert_eq!(status.path, root.join("inbox").to_string_lossy());
        assert_eq!(status.imported_since_launch, 0);
        assert!(status.failed.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }
}
