//! `<data>/workbooks` file watching (C4 §4, design §7). Distinguishes the
//! app's own writes (temp-file + rename, hash pre-registered before the
//! rename — C4 §4 step 3) from external edits, debounces ~100 ms, and calls
//! back with the changed path. Cell-level diffing and re-evaluation are the
//! caller's job (`watch_workbook`, gated on L3) — this module only answers
//! "did this path just change, and was it us."

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::{RecursiveMode, Watcher};

const EXPECTED_HASH_TTL: Duration = Duration::from_secs(5);
const DEBOUNCE: Duration = Duration::from_millis(100);

/// The C4 §4 self-write-suppression primitive: `(path -> (hex sha256,
/// inserted_at))`, entries expiring after `EXPECTED_HASH_TTL` so a stale
/// entry can never misclassify a later, genuinely external write.
#[derive(Default)]
pub struct ExpectedHashSet(Mutex<HashMap<PathBuf, (String, Instant)>>);

impl ExpectedHashSet {
    /// Builds an empty set with nothing pre-registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the hash the app is about to write to `path`, **before**
    /// the atomic rename (C4 §4 step 3 — this ordering is load-bearing).
    pub fn expect(&self, path: PathBuf, sha256_hex: String) {
        self.0.lock().unwrap().insert(path, (sha256_hex, Instant::now()));
    }

    /// True iff `path` has a live, matching expected hash — meaning this
    /// change was the app's own write. A single logical write can surface as
    /// more than one filesystem event with identical bytes (observed on
    /// Windows: `Create` then `Modify`, both carrying the final content), so
    /// a matching entry is deliberately **not** removed on first match —
    /// removal happens only via `EXPECTED_HASH_TTL` expiry, checked here
    /// lazily. A later, genuinely external write to the same path carries
    /// different content (a different hash), so it still cannot be
    /// misclassified while a matched entry lingers.
    pub fn check_and_consume(&self, path: &Path, actual_sha256_hex: &str) -> bool {
        let mut map = self.0.lock().unwrap();
        let Some((expected, at)) = map.get(path) else { return false };
        if at.elapsed() >= EXPECTED_HASH_TTL {
            map.remove(path);
            return false;
        }
        expected == actual_sha256_hex
    }
}

/// Hex-encoded SHA-256 of `bytes`, used to compare a just-written file's
/// content against a pre-registered expected hash (C4 §4 step 3).
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Watches `workbooks_dir` (non-recursive — C4 §2's `workbooks/` is flat) for
/// create/rename events, filters out the app's own writes via `hashes`, and
/// calls `on_external_change(path)` after a ~100 ms debounce per path.
pub struct WorkbookWatcher {
    _inner: notify::RecommendedWatcher, // kept alive for the watcher's lifetime
}

impl WorkbookWatcher {
    /// Starts watching `workbooks_dir`. `hashes` is shared with the writer
    /// side (whoever performs the app's own temp-file + rename writes) so
    /// self-writes are suppressed; `on_external_change` fires once per
    /// externally-changed path, debounced ~100 ms, on a background thread.
    pub fn new(
        workbooks_dir: &Path,
        hashes: Arc<ExpectedHashSet>,
        on_external_change: impl Fn(&Path) + Send + Sync + 'static,
    ) -> notify::Result<Self> {
        let on_external_change: Arc<dyn Fn(&Path) + Send + Sync> = Arc::new(on_external_change);
        let pending: Arc<Mutex<HashMap<PathBuf, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let Ok(event) = res else { return };
            use notify::EventKind::*;
            if !matches!(event.kind, Create(_) | Modify(_)) {
                return;
            }
            for path in event.paths {
                let Ok(bytes) = std::fs::read(&path) else { continue }; // gone again before we read it
                let hash = sha256_hex(&bytes);
                if hashes.check_and_consume(&path, &hash) {
                    continue; // our own write — matches a live expected hash, do not re-parse (C4 §4)
                }
                pending.lock().unwrap().insert(path.clone(), Instant::now());
                let pending = Arc::clone(&pending);
                let on_external_change = Arc::clone(&on_external_change);
                let path_for_thread = path.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(DEBOUNCE);
                    let mut map = pending.lock().unwrap();
                    if let Some(&t) = map.get(&path_for_thread) {
                        if t.elapsed() >= DEBOUNCE {
                            map.remove(&path_for_thread);
                            on_external_change(&path_for_thread);
                        }
                    }
                });
            }
        })?;
        watcher.watch(workbooks_dir, RecursiveMode::NonRecursive)?;
        Ok(Self { _inner: watcher })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn external_write_to_workbooks_dir_fires_callback_with_the_path() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();
        let (tx, rx) = mpsc::channel::<PathBuf>();
        let hashes = Arc::new(ExpectedHashSet::new());
        let _watcher = WorkbookWatcher::new(dir.path(), hashes, move |p| { let _ = tx.send(p.to_path_buf()); }).unwrap();
        let target = dir.path().join("dummy.idl1wb");

        // Act
        std::fs::write(&target, "---\nid: test\n---\n# Dummy\n").unwrap();

        // Assert
        let seen = rx.recv_timeout(Duration::from_millis(500)).expect("callback fired");
        assert_eq!(seen, target);
    }

    #[test]
    fn self_write_with_pre_registered_hash_never_fires_callback() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (tx, rx) = mpsc::channel::<PathBuf>();
        let hashes = Arc::new(ExpectedHashSet::new());
        let target = dir.path().join("dummy.idl1wb");
        let content = b"---\nid: test\n---\n# Dummy\n";
        hashes.expect(target.clone(), sha256_hex(content)); // registered before the write, per C4 §4 step 3
        let _watcher = WorkbookWatcher::new(dir.path(), Arc::clone(&hashes), move |p| { let _ = tx.send(p.to_path_buf()); }).unwrap();

        // Act
        std::fs::write(&target, content).unwrap();

        // Assert
        assert!(rx.recv_timeout(Duration::from_millis(500)).is_err(), "callback must not fire for a self-write");
    }
}
