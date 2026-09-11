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
//! entries until the new one fits.
//!
//! The same budget is also a **byte-counting semaphore** over decodes in
//! flight (ruling R211.2). Sizing each request on its own was not enough:
//! nine notebook cells each passed a check that nine of them together could
//! not honour, and the sum is what the allocator refused. A decode now
//! reserves its estimated bytes before it allocates and gives them back when
//! it is done; a decode that does not fit waits (bounded by
//! [`RESERVE_TIMEOUT`]) instead of failing, and only a request too large for
//! the whole budget on its own is refused outright.
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
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use idl_rs::session::ChannelSamples;
use idl_rs::store::parquet::{estimate_channel_bytes, read_channel, ParquetStoreErrorKind};

use crate::error::{IpcError, IpcErrorKind};
use crate::memory::{budget_bytes, resource_exhausted, with_estimate_margin};

/// How long a decode waits for the budget to free up before giving up
/// (ruling R211.2). Long enough that a notebook full of cells binding at
/// once serialises and every one of them succeeds — the whole point of
/// waiting rather than failing — and short enough that a genuinely stuck
/// app tells the user instead of hanging.
pub const RESERVE_TIMEOUT: Duration = Duration::from_secs(30);

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
#[derive(Debug)]
struct Inner {
    entries: HashMap<Key, Arc<ChannelSamples>>,
    order: Vec<Key>,
    resident_bytes: u64,
    budget_bytes: u64,
    /// Bytes promised to decodes that have started but not finished
    /// (ruling R211.2). The semaphore half of the budget: `resident_bytes +
    /// in_flight_bytes` is what the app has committed itself to, and a new
    /// decode may only begin when its own estimate still fits under
    /// `budget_bytes` alongside both.
    in_flight_bytes: u64,
}

/// Everything a cache is, behind one `Arc` — see [`SessionCache`]'s note on
/// cloning.
#[derive(Debug)]
struct Shared {
    inner: Mutex<Inner>,
    /// Signalled whenever bytes are given back — a reservation dropped, or
    /// an entry evicted — so waiting decodes re-check the budget.
    released: Condvar,
}

/// Per-channel decoded-sample cache, LRU by bytes (ruling R203.2), and the
/// process's one memory budget (ruling R211.2).
///
/// Managed by the app for its lifetime (`app.manage(SessionCache::new())`)
/// and reached by commands through `tauri::State<SessionCache>`. Every
/// command that serves samples goes through [`SessionCache::channel`];
/// nothing else calls `read_channel` directly.
///
/// **A clone is the same cache**, not a copy of it: entries, residency and
/// the in-flight byte counter all live behind one `Arc`. That is what lets
/// a lazy `SessionHandle`'s channel source hold a cache for as long as the
/// handle lives (`session_source::CachedChannelSource`) while commands go on
/// borrowing it from `tauri::State`, and it is what makes "one budget for
/// the process" true rather than aspirational — two clones cannot each
/// spend the whole budget.
#[derive(Debug, Clone)]
pub struct SessionCache {
    shared: Arc<Shared>,
}

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
        SessionCache {
            shared: Arc::new(Shared {
                inner: Mutex::new(Inner {
                    entries: HashMap::new(),
                    order: Vec::new(),
                    resident_bytes: 0,
                    budget_bytes,
                    in_flight_bytes: 0,
                }),
                released: Condvar::new(),
            }),
        }
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
        self.shared.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Bytes currently promised to decodes in flight (ruling R211.2).
    pub fn in_flight_bytes(&self) -> u64 {
        self.lock().in_flight_bytes
    }

    /// Claims `needed_bytes` of the budget for one decode, **waiting** for
    /// other decodes to finish rather than failing when it does not fit
    /// (ruling R211.2).
    ///
    /// This is the fix for what killed the app: every request used to be
    /// sized on its own against the whole budget, so N concurrent requests
    /// each passed a check that N of them together could not honour, and
    /// their sum was what the allocator could not find. Here the budget is
    /// one counter for the whole process's worth of caches: a request that
    /// does not fit *right now* sleeps until one does, which turns a burst
    /// of notebook cells binding at once into a queue instead of an abort.
    ///
    /// The estimate is inflated by
    /// [`crate::memory::with_estimate_margin`] first, exactly as the old
    /// per-request check inflated it, and the reservation is that inflated
    /// figure — the margin is what covers a decode's transient peak, so it
    /// must be held for the decode's duration, not merely compared once.
    ///
    /// What is counted is **decodes in flight**, not residency: the LRU keeps
    /// resident bytes under the same budget separately, so the app's ceiling
    /// is one budget of cached channels plus one budget of decodes running
    /// right now. Charging residency to the same counter was tried and
    /// rejected — a decode whose estimate approaches the budget would then
    /// have to evict the entire cache to start, which turns every large
    /// channel into a cache flush and re-reads the file on the next hover.
    /// The transient peak is what the incident ran out of, and that is what
    /// this bounds.
    ///
    /// # Errors
    /// [`IpcErrorKind::ResourceExhausted`] in two cases, deliberately the
    /// same kind because the UI's answer to both is the same toast:
    /// - **this one request alone exceeds the whole budget** — no amount of
    ///   waiting would help, so it fails immediately;
    /// - **[`RESERVE_TIMEOUT`] elapsed** with other work still holding the
    ///   budget. `detail` then carries `waited_ms` alongside C3 §1's
    ///   `{ needed_bytes, budget_bytes, hint }`, so the two are
    ///   distinguishable by a caller that cares.
    pub fn reserve(&self, needed_bytes: u64, hint: &str) -> Result<Reservation<'_>, IpcError> {
        self.reserve_with_timeout(needed_bytes, hint, RESERVE_TIMEOUT)
    }

    /// [`Self::reserve`] with an explicit wait. Exists so tests can pin the
    /// timeout behaviour in milliseconds instead of
    /// [`RESERVE_TIMEOUT`]'s thirty seconds; production always uses
    /// `reserve`.
    pub fn reserve_with_timeout(
        &self,
        needed_bytes: u64,
        hint: &str,
        timeout: Duration,
    ) -> Result<Reservation<'_>, IpcError> {
        let needed = with_estimate_margin(needed_bytes);
        let mut inner = self.lock();
        let budget = inner.budget_bytes;
        if needed > budget {
            return Err(resource_exhausted(needed, budget, hint));
        }

        let deadline = Instant::now() + timeout;
        loop {
            if inner.in_flight_bytes + needed <= budget {
                inner.in_flight_bytes += needed;
                return Ok(Reservation { cache: self, bytes: needed });
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(reserve_timed_out(needed, budget, hint, timeout));
            }
            let (guard, _) = self
                .shared
                .released
                .wait_timeout(inner, remaining)
                .unwrap_or_else(|e| e.into_inner());
            inner = guard;
        }
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

        let needed = estimate_channel_bytes(session_dir, channel).map_err(|e| store_error(session_id, channel, 0, e))?;
        let reservation = self.reserve(needed, &format!("decode channel '{channel}' of session '{session_id}'"))?;

        let samples = Arc::new(read_channel(session_dir, channel).map_err(|e| store_error(session_id, channel, needed, e))?);
        // Insert first, then give the reservation back: releasing it before
        // the bytes are accounted as resident would let a waiter through on
        // a budget this decode is still occupying.
        self.lock().insert(key, samples.clone());
        drop(reservation);
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
        self.lock().retain(|(sid, _)| sid != session_id);
        self.shared.released.notify_all();
    }

    /// Drops everything. For a data-directory switch, where every session
    /// id may now mean a different file.
    pub fn invalidate_all(&self) {
        self.lock().retain(|_| false);
        self.shared.released.notify_all();
    }
}

/// One decode's claim on the budget, released when it is dropped (ruling
/// R211.2).
///
/// Held for exactly as long as the allocation it covers: taken before
/// `read_channel` is called, given back once the result is accounted as
/// resident. Dropping it wakes every waiting decode, so a queue of cells
/// drains as fast as the decodes finish.
#[derive(Debug)]
pub struct Reservation<'a> {
    cache: &'a SessionCache,
    bytes: u64,
}

impl Reservation<'_> {
    /// Bytes this reservation holds — the caller's estimate with
    /// [`crate::memory::with_estimate_margin`] applied.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        {
            let mut inner = self.cache.lock();
            inner.in_flight_bytes = inner.in_flight_bytes.saturating_sub(self.bytes);
        }
        self.cache.shared.released.notify_all();
    }
}

/// The C3 §1 `resource_exhausted` error for a decode that waited
/// [`RESERVE_TIMEOUT`] for the budget and never got it (ruling R211.2).
///
/// Same kind and same three `detail` keys as
/// [`crate::memory::resource_exhausted`] — the UI's answer is the same
/// toast — plus `waited_ms`, which is what distinguishes "the app is busy"
/// from "this session is too big for this machine".
fn reserve_timed_out(needed_bytes: u64, budget: u64, hint: &str, waited: Duration) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::ResourceExhausted,
        format!(
            "Timed out waiting for memory: {:.1} GB needed, {:.1} GB budget, all of it in use",
            needed_bytes as f64 / 1e9,
            budget as f64 / 1e9
        ),
        serde_json::json!({
            "needed_bytes": needed_bytes,
            "budget_bytes": budget,
            "hint": hint,
            "waited_ms": waited.as_millis() as u64,
        }),
    )
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
    ///
    /// Evicting only stops the cache *counting* those bytes: a channel a
    /// command is still holding an `Arc` to keeps its memory until that
    /// command returns. That is the honest limit of a byte budget over
    /// shared, immutable data, and it is why the reservation half of
    /// [`SessionCache::reserve`] exists — the in-flight counter bounds what
    /// is actually being allocated right now, which this counter cannot.
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
/// directory, file or channel is `not_found`; an allocation the machine
/// refused is `resource_exhausted` (ruling R211.4); everything else is
/// `io`.
fn store_error(session_id: &str, channel: &str, needed_bytes: u64, e: idl_rs::store::parquet::ParquetStoreError) -> IpcError {
    match e.kind {
        ParquetStoreErrorKind::NotFound => IpcError::with_detail(
            IpcErrorKind::NotFound,
            format!("channel '{channel}' not found on session '{session_id}'"),
            serde_json::json!({ "session_id": session_id, "channel": channel }),
        ),
        ParquetStoreErrorKind::ResourceExhausted => resource_exhausted(
            needed_bytes,
            budget_bytes(),
            &format!("decode channel '{channel}' of session '{session_id}'"),
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

    /// Seeds `session_id` with four 1 kHz f64 channels of `n` samples
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
            channels: vec![channel("A"), channel("B"), channel("C"), channel("D")],
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
        // Arrange — a budget holding three of the four channels. It must
        // also clear one channel's decode *estimate*, which is a
        // deliberately generous ceiling over its resident size, or no decode
        // could start at all (ruling R211.2).
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let one = SessionCache::with_budget(1 << 30).channel(&dir, "s1", "A").unwrap().resident_bytes() as u64;
        let estimate = with_estimate_margin(estimate_channel_bytes(&dir, "A").unwrap());
        let cache = SessionCache::with_budget((one * 3).max(estimate));

        // Act — A, B, C, then touch A so B is coldest, then D.
        cache.channel(&dir, "s1", "A").unwrap();
        cache.channel(&dir, "s1", "B").unwrap();
        cache.channel(&dir, "s1", "C").unwrap();
        cache.channel(&dir, "s1", "A").unwrap();
        cache.channel(&dir, "s1", "D").unwrap();

        // Assert — B was the least recently used, so B is the one gone.
        assert!(cache.lock().entries.contains_key(&("s1".to_string(), "A".to_string())));
        assert!(cache.lock().entries.contains_key(&("s1".to_string(), "D".to_string())));
        assert!(!cache.lock().entries.contains_key(&("s1".to_string(), "B".to_string())));
        assert!(cache.resident_bytes() <= cache.budget_bytes());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_a_channel_whose_own_estimate_exceeds_the_whole_budget_is_refused_as_resource_exhausted() {
        // Arrange — a budget no single decode of this session could fit.
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(8);

        // Act
        let err = cache.channel(&dir, "s1", "A").unwrap_err();

        // Assert — R211.2's one non-waiting refusal: waiting for other work
        // to finish could never make room, so the caller is told now.
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        assert!(cache.is_empty());
        assert_eq!(cache.in_flight_bytes(), 0);

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
    fn reserve_a_request_larger_than_the_whole_budget_fails_immediately_as_resource_exhausted() {
        // Arrange
        let cache = SessionCache::with_budget(1_000_000);

        // Act
        let started = Instant::now();
        let err = cache.reserve(10_000_000, "decode channel Speed").unwrap_err();

        // Assert — no amount of waiting would help, so it does not wait.
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        assert!(started.elapsed() < Duration::from_secs(1));
        let detail = err.detail.unwrap();
        assert_eq!(detail["hint"], "decode channel Speed");
        assert!(detail.get("waited_ms").is_none());
        assert_eq!(cache.in_flight_bytes(), 0);
    }

    #[test]
    fn reserve_a_waiter_that_never_gets_the_budget_times_out_as_resource_exhausted() {
        // Arrange — one reservation holding the whole budget, never released
        // for the duration of the wait.
        let cache = SessionCache::with_budget(1_000_000);
        let _held = cache.reserve(800_000, "the decode in front").unwrap();

        // Act
        let err = cache
            .reserve_with_timeout(800_000, "the decode behind it", Duration::from_millis(50))
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        let detail = err.detail.unwrap();
        assert_eq!(detail["waited_ms"], 50u64);
        assert_eq!(detail["hint"], "the decode behind it");
    }

    #[test]
    fn reserve_dropping_a_reservation_lets_the_next_waiter_through() {
        // Arrange
        let cache = SessionCache::with_budget(1_000_000);
        let held = cache.reserve(700_000, "first").unwrap();

        // Act
        drop(held);
        let second = cache.reserve_with_timeout(700_000, "second", Duration::from_millis(50));

        // Assert
        assert!(second.is_ok());
        assert!(cache.in_flight_bytes() > 0);
    }

    #[test]
    fn reserve_four_concurrent_requests_summing_past_the_budget_all_succeed_by_serialising() {
        // Arrange — four requests of 400 kB (500 kB with the margin) against
        // a 1 MB budget: any two fit, all four together do not. Before
        // R211.2 each passed its own check and all four allocated at once.
        let cache = SessionCache::with_budget(1_000_000);
        let peak = Arc::new(Mutex::new(0u64));

        // Act
        let mut handles = Vec::new();
        for i in 0..4 {
            let cache = cache.clone();
            let peak = Arc::clone(&peak);
            handles.push(std::thread::spawn(move || {
                let reservation = cache.reserve(400_000, &format!("decode {i}")).unwrap();
                let mut seen = peak.lock().unwrap();
                *seen = (*seen).max(cache.in_flight_bytes());
                drop(seen);
                std::thread::sleep(Duration::from_millis(20));
                drop(reservation);
            }));
        }
        let outcomes: Vec<bool> = handles.into_iter().map(|h| h.join().is_ok()).collect();

        // Assert — every one of them got served, and the budget was never
        // oversubscribed while they ran.
        assert_eq!(outcomes, vec![true; 4]);
        assert!(*peak.lock().unwrap() <= cache.budget_bytes());
        assert_eq!(cache.in_flight_bytes(), 0);
    }

    #[test]
    fn clone_two_handles_on_one_cache_share_the_budget_rather_than_each_getting_their_own() {
        // Arrange
        let cache = SessionCache::with_budget(1_000_000);
        let twin = cache.clone();

        // Act
        let _held = cache.reserve(700_000, "held by the original").unwrap();
        let err = twin
            .reserve_with_timeout(700_000, "asked for through the clone", Duration::from_millis(50))
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        assert_eq!(twin.in_flight_bytes(), cache.in_flight_bytes());
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
