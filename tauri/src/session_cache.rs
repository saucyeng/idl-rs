//! One decode per session channel, bounded by bytes (ruling R203.2).
//!
//! Before this cache every tile, cursor and raster call re-read the whole
//! `data.parquet` from disk and threw all but one channel away — dozens of
//! full-session-sized allocations a minute while panning. [`SessionCache`]
//! holds `session_id -> channel -> Arc<ChannelSamples>` instead: a channel
//! is decoded at most once while it is resident, and every consumer shares
//! the same `Arc` rather than its own copy.
//!
//! Residency is capped in bytes, not entries — channels differ in size by
//! orders of magnitude, so an entry count would bound nothing. The budget
//! is `crate::memory::budget_bytes()`, read once at startup, and eviction
//! is least-recently-used: inserting past the budget drops the coldest
//! entries until the new one fits. A single channel larger than the whole
//! budget is returned to the caller and simply not retained, rather than
//! evicting the world for something that cannot fit.
//!
//! Invalidation is by session: a reimport, a delete or a `session_forgotten`
//! event drops every channel of that session, because `data.parquet` is a
//! function of (blob, importer version) and a changed file makes every
//! decode of it stale at once.
//!
//! The lock is held to look a channel up and to insert it, but **not**
//! across the decode in between, so two commands racing for the same
//! not-yet-resident channel (a `fetch_tile` and the `cursor_readout` that
//! settles behind it) can both decode it once. Both results are correct
//! and byte accounting stays consistent — the loser's copy is simply
//! dropped — so this costs one redundant decode on a cold channel, never
//! correctness. Holding the lock across the decode instead would serialise
//! every sample-serving command in the app behind the slowest read.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use idl_rs::session::ChannelSamples;
use idl_rs::store::parquet::{estimate_channel_bytes, read_channel, ParquetStoreErrorKind};

use crate::error::{IpcError, IpcErrorKind};
use crate::memory::{budget_bytes, ensure_fits};

/// Cache key: which session, which channel.
type Key = (String, String);

/// Everything the cache mutates, behind one lock.
///
/// `order` is the LRU chain, coldest first: the key of every resident entry
/// exactly once. A hit moves its key to the back; an insert pushes to the
/// back and pops from the front until `resident_bytes <= budget_bytes`.
/// A `Vec` rather than a linked list because the entry count is small
/// (channels per session, tens) and a `retain`/`push` pair on it is
/// cheaper in practice than the allocation a list would need per node.
struct Inner {
    entries: HashMap<Key, Arc<ChannelSamples>>,
    order: Vec<Key>,
    resident_bytes: u64,
    budget_bytes: u64,
}

/// Per-channel decoded-sample cache, LRU by bytes (ruling R203.2).
///
/// Managed by the app for its lifetime (`app.manage(SessionCache::new())`)
/// and reached by commands through `tauri::State<SessionCache>`. Every
/// command that serves samples goes through [`SessionCache::channel`];
/// nothing else calls `read_channel` directly.
pub struct SessionCache(Mutex<Inner>);

impl Default for SessionCache {
    fn default() -> Self {
        Self::with_budget(budget_bytes())
    }
}

impl SessionCache {
    /// A cache budgeted at [`crate::memory::budget_bytes`] — `min(2 GiB,
    /// 25 % of physical RAM)`, read once.
    pub fn new() -> Self {
        Self::default()
    }

    /// A cache with an explicit byte budget. Exists for tests, which need a
    /// budget small enough to force eviction deterministically.
    pub fn with_budget(budget_bytes: u64) -> Self {
        SessionCache(Mutex::new(Inner {
            entries: HashMap::new(),
            order: Vec::new(),
            resident_bytes: 0,
            budget_bytes,
        }))
    }

    /// The budget this cache was built with, bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.lock().budget_bytes
    }

    /// Bytes currently resident.
    pub fn resident_bytes(&self) -> u64 {
        self.lock().resident_bytes
    }

    /// Number of resident channels, across every session.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// `true` when nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The lock, recovered from a poisoned mutex rather than propagated.
    ///
    /// A panic while the cache lock was held would otherwise turn every
    /// later sample request into a panic of its own. The cache is a
    /// rebuildable index over `data.parquet` (design §5's rule for the
    /// catalog applies here too), so the worst a poisoned-and-recovered
    /// state can be is stale-free and slow, never wrong.
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One channel's samples, decoded at most once while resident.
    ///
    /// `session_dir` is `<data>/sessions/<session_id>` (C4 §2) and
    /// `session_id` is the cache's own key for it — they are passed
    /// separately so the key never depends on how the path was spelled.
    ///
    /// On a miss the decode is sized first (ruling R203.4): an estimate
    /// past the budget is refused as `resource_exhausted` **before** any
    /// allocation, so a session too large for this machine produces a toast
    /// rather than an abort. On a hit nothing is read and nothing is
    /// allocated beyond the `Arc` clone.
    ///
    /// An unknown session or channel is [`IpcErrorKind::NotFound`]; a
    /// corrupt or unreadable `data.parquet` is [`IpcErrorKind::Io`].
    pub fn channel(&self, session_dir: &Path, session_id: &str, channel: &str) -> Result<Arc<ChannelSamples>, IpcError> {
        let key = (session_id.to_string(), channel.to_string());
        if let Some(hit) = self.lock().touch(&key) {
            return Ok(hit);
        }

        let needed = estimate_channel_bytes(session_dir, channel).map_err(|e| store_error(session_id, channel, e))?;
        ensure_fits(needed, &format!("decode channel '{channel}' of session '{session_id}'"))?;

        let samples = Arc::new(read_channel(session_dir, channel).map_err(|e| store_error(session_id, channel, e))?);
        self.lock().insert(key, samples.clone());
        Ok(samples)
    }

    /// Drops every resident channel of `session_id`.
    ///
    /// Called when the file the decodes came from is gone or has changed:
    /// `delete_session`, a reimport/rebuild, and the `session_forgotten`
    /// path. Dropping the whole session rather than one channel is
    /// deliberate — `data.parquet` is rewritten as a unit, so one changed
    /// channel means every decode of that file is stale.
    pub fn invalidate_session(&self, session_id: &str) {
        let mut inner = self.lock();
        inner.retain(|(sid, _)| sid != session_id);
    }

    /// Drops everything. For a data-directory switch, where every session
    /// id may now mean a different file.
    pub fn invalidate_all(&self) {
        let mut inner = self.lock();
        inner.retain(|_| false);
    }
}

impl Inner {
    /// The entry for `key`, marking it most-recently-used, or `None`.
    fn touch(&mut self, key: &Key) -> Option<Arc<ChannelSamples>> {
        let hit = self.entries.get(key)?.clone();
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
        Some(hit)
    }

    /// Inserts `samples` as most-recently-used, then evicts the coldest
    /// entries until residency is inside the budget.
    ///
    /// An entry bigger than the whole budget is not retained at all: the
    /// caller already holds its `Arc`, and admitting it would evict every
    /// other channel to hold something that still does not fit.
    fn insert(&mut self, key: Key, samples: Arc<ChannelSamples>) {
        let bytes = samples.resident_bytes() as u64;
        if bytes > self.budget_bytes {
            return;
        }
        if let Some(old) = self.entries.insert(key.clone(), samples) {
            self.resident_bytes = self.resident_bytes.saturating_sub(old.resident_bytes() as u64);
            self.order.retain(|k| k != &key);
        }
        self.resident_bytes += bytes;
        self.order.push(key);
        self.evict_to_budget();
    }

    /// Drops coldest-first until `resident_bytes <= budget_bytes`.
    fn evict_to_budget(&mut self) {
        while self.resident_bytes > self.budget_bytes && !self.order.is_empty() {
            let coldest = self.order.remove(0);
            if let Some(dropped) = self.entries.remove(&coldest) {
                self.resident_bytes = self.resident_bytes.saturating_sub(dropped.resident_bytes() as u64);
            }
        }
    }

    /// Keeps only the entries whose key `keep` accepts, re-deriving
    /// residency from what survives.
    fn retain(&mut self, keep: impl Fn(&Key) -> bool) {
        let entries = &mut self.entries;
        let mut freed = 0u64;
        entries.retain(|k, v| {
            let kept = keep(k);
            if !kept {
                freed += v.resident_bytes() as u64;
            }
            kept
        });
        self.order.retain(|k| keep(k));
        self.resident_bytes = self.resident_bytes.saturating_sub(freed);
    }
}

/// Maps a parquet-store failure to its C3 §2 kind: a missing session
/// directory, file or channel is `not_found`; everything else is `io`.
fn store_error(session_id: &str, channel: &str, e: idl_rs::store::parquet::ParquetStoreError) -> IpcError {
    match e.kind {
        ParquetStoreErrorKind::NotFound => IpcError::with_detail(
            IpcErrorKind::NotFound,
            format!("channel '{channel}' not found on session '{session_id}'"),
            serde_json::json!({ "session_id": session_id, "channel": channel }),
        ),
        _ => IpcError::new(IpcErrorKind::Io, format!("reading channel '{channel}' of session '{session_id}': {}", e.message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-cache-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds `session_id` with three 1 kHz f64 channels of `n` samples
    /// each, so every channel has the same, known resident size.
    fn seed(root: &Path, session_id: &str, n: usize) {
        let channel = |id: &str| Channel {
            channel_id: id.to_string(),
            t_us: (0..n).map(|i| i as i64 * 1000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1000.0,
            column: RawColumn::F64((0..n).map(|i| i as f64).collect()),
            source_kind: "wheel".to_string(),
            unit: "m/s".to_string(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![channel("A"), channel("B"), channel("C")],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn session_dir(root: &Path, session_id: &str) -> PathBuf {
        root.join("sessions").join(session_id)
    }

    #[test]
    fn channel_a_second_request_for_the_same_channel_is_a_hit_returning_the_same_allocation() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let first = cache.channel(&dir, "s1", "A").unwrap();
        let second = cache.channel(&dir, "s1", "A").unwrap();

        // Assert
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_a_miss_decodes_the_channel_and_accounts_its_bytes_as_resident() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let samples = cache.channel(&dir, "s1", "A").unwrap();

        // Assert — 100 f64 samples plus 100 i64 timestamps.
        assert_eq!(samples.len(), 100);
        assert_eq!(cache.resident_bytes(), samples.resident_bytes() as u64);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_inserting_past_the_budget_evicts_the_least_recently_used_channel_first() {
        // Arrange — a budget holding exactly two of the three channels.
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let one = SessionCache::with_budget(1 << 30).channel(&dir, "s1", "A").unwrap().resident_bytes() as u64;
        let cache = SessionCache::with_budget(one * 2);

        // Act — A, then B, then touch A so B is coldest, then C.
        cache.channel(&dir, "s1", "A").unwrap();
        cache.channel(&dir, "s1", "B").unwrap();
        cache.channel(&dir, "s1", "A").unwrap();
        cache.channel(&dir, "s1", "C").unwrap();

        // Assert — B was the least recently used, so B is the one gone.
        assert_eq!(cache.len(), 2);
        assert!(cache.lock().entries.contains_key(&("s1".to_string(), "A".to_string())));
        assert!(cache.lock().entries.contains_key(&("s1".to_string(), "C".to_string())));
        assert!(!cache.lock().entries.contains_key(&("s1".to_string(), "B".to_string())));
        assert!(cache.resident_bytes() <= cache.budget_bytes());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_a_channel_larger_than_the_whole_budget_is_returned_but_never_retained() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(8);

        // Act
        let samples = cache.channel(&dir, "s1", "A").unwrap();

        // Assert
        assert_eq!(samples.len(), 100);
        assert!(cache.is_empty());
        assert_eq!(cache.resident_bytes(), 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalidate_session_drops_that_sessions_channels_and_leaves_every_other_sessions_alone() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        seed(&root, "s2", 100);
        let cache = SessionCache::with_budget(1 << 30);
        cache.channel(&session_dir(&root, "s1"), "s1", "A").unwrap();
        cache.channel(&session_dir(&root, "s1"), "s1", "B").unwrap();
        let kept = cache.channel(&session_dir(&root, "s2"), "s2", "A").unwrap();

        // Act
        cache.invalidate_session("s1");

        // Assert
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_bytes(), kept.resident_bytes() as u64);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalidate_session_a_reimport_between_two_reads_serves_the_rewritten_samples() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);
        assert_eq!(cache.channel(&dir, "s1", "A").unwrap().len(), 100);

        // Act — the file is rewritten with a different length, and the
        // cache is told, exactly as an import/delete path tells it.
        std::fs::remove_dir_all(&dir).unwrap();
        seed(&root, "s1", 42);
        cache.invalidate_session("s1");

        // Assert
        assert_eq!(cache.channel(&dir, "s1", "A").unwrap().len(), 42);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_an_unknown_channel_is_not_found_and_nothing_becomes_resident() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let err = cache.channel(&dir, "s1", "NopeChannel").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);
        assert!(cache.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_an_unknown_session_is_not_found() {
        // Arrange
        let root = temp_root();
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let err = cache.channel(&session_dir(&root, "nope"), "nope", "A").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalidate_all_drops_every_session() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", 100);
        seed(&root, "s2", 100);
        let cache = SessionCache::with_budget(1 << 30);
        cache.channel(&session_dir(&root, "s1"), "s1", "A").unwrap();
        cache.channel(&session_dir(&root, "s2"), "s2", "A").unwrap();

        // Act
        cache.invalidate_all();

        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.resident_bytes(), 0);

        let _ = std::fs::remove_dir_all(&root);
    }
}
