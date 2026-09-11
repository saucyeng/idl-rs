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
//! A session's **timestamp axes are cached apart from its channels**
//! (ruling R232.1). `data.parquet`'s union `t` column and each source's
//! `<source>_t_recorded_us` companion used to be decompressed inside every
//! channel decode, so six `imu0` channels of a 516 MB session paid for the
//! same two i64 columns six times — the cost the R221 measurement found
//! behind the two silent minutes. They are now their own entries, decoded
//! once per session (`t`) and once per source (the companion) and borrowed
//! by every channel that reads through them. An axis is counted in
//! residency exactly once, and is never evicted while a channel that
//! borrows it is resident.
//!
//! The lock is held to look a channel up and to insert it, but **not**
//! across the decode in between, so every sample-serving command in the app
//! is not serialised behind the slowest read. A channel already being
//! decoded by another thread is **not** decoded a second time (ruling
//! R232.2): the key is marked in flight, later callers wait on
//! [`Shared::decoded`] and take the result the leader inserts. One decode,
//! many waiters — which is what makes a request for nine channels at once
//! safe to fan out.
//!
//! [`SessionCache::channels`] is that fan-out: several channels of one
//! request (a cell bind, a report, a raster) decode on a shared rayon pool,
//! each worker reserving through the same byte semaphore, so they queue on
//! memory rather than failing on it.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use idl_rs::session::ChannelSamples;
use idl_rs::store::index_job::worker_count;
use idl_rs::store::parquet::{
    estimate_channel_bytes, read_channel_index, read_channel_sharing_axes, read_channel_with_progress,
    recorded_axis_column, AxisColumn, BorrowedAxes, ParquetStoreErrorKind,
};

use crate::error::{IpcError, IpcErrorKind};
use crate::memory::{budget_bytes, resource_exhausted, with_estimate_margin};

/// How long a decode waits for the budget to free up before giving up
/// (ruling R211.2). Long enough that a notebook full of cells binding at
/// once serialises and every one of them succeeds — the whole point of
/// waiting rather than failing — and short enough that a genuinely stuck
/// app tells the user instead of hanging.
pub const RESERVE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a decode runs before it starts reporting progress (ruling R221
/// item 1's "more than ~200 ms").
///
/// A decode is not timed in advance — its duration is only known by living
/// through it — so the rule is applied as a delay: nothing is reported until
/// the decode has already been running this long, and a decode that finishes
/// first reports nothing at all. That is the point: a ring that appears and
/// vanishes inside two frames is noise, and every hover and pan in the app
/// goes through this same function.
pub const PROGRESS_AFTER: Duration = Duration::from_millis(200);

/// Minimum gap between two `decode_progress` events for one decode.
///
/// The reader hands back a `RecordBatch` every 1024 rows or so, which on a
/// multi-million-row session is thousands of observations a second —
/// far more than a ring redrawing at 60 Hz can use, and every one of them
/// crosses IPC. Ten a second is enough to look continuous.
pub const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// C3 §3.2 `decode_progress` — one observation of one channel's decode
/// (ruling R221 item 1). Field names are the contract; `app/src/ipc/
/// decode_progress.ts` mirrors them byte for byte.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DecodeProgressEvent {
    /// The session whose `data.parquet` is being decoded.
    pub session_id: String,
    /// The channel being decoded, as the cell's binding spells it.
    pub channel: String,
    /// Rows decoded so far, over every pass this decode makes (a
    /// synthesized `Distance` makes two — see
    /// `idl_rs::store::parquet::read_channel_with_progress`).
    pub done_rows: u64,
    /// Rows this decode has to get through in total. Never `0` on an event
    /// the app renders, so `done_rows / total_rows` is always defined.
    pub total_rows: u64,
    /// `true` on the one terminal observation for this decode, whether it
    /// succeeded or failed. A consumer that only ever saw `false` would
    /// leave a ring spinning forever on a decode that errored, so this is
    /// emitted on the error path too (with whatever counts were last seen).
    pub finished: bool,
}

/// Where [`SessionCache`] sends its [`DecodeProgressEvent`]s — the Tauri
/// event emitter in the app, a recording closure in tests.
///
/// `Arc<dyn Fn>` rather than a `tauri::AppHandle`: this module is the one
/// piece of decode plumbing that is otherwise free of Tauri, and a cache
/// that could only report through an `AppHandle` could not be tested without
/// a running app.
pub type ProgressSink = Arc<dyn Fn(DecodeProgressEvent) + Send + Sync>;

/// Cache key: which session, which channel.
type Key = (String, String);

/// Cache key for a shared timestamp axis (ruling R232.1): which session,
/// which axis column — `"t"` for the union axis, or the
/// `<source>_t_recorded_us` spelling
/// [`idl_rs::store::parquet::recorded_axis_column`] builds for a source.
///
/// The same `(String, String)` shape as [`Key`] and deliberately a distinct
/// map: an axis is not a channel, is never served to a caller as samples,
/// and never reaches a DTO.
type AxisKey = (String, String);

/// What one decode has claimed so later callers wait for it instead of
/// repeating it (ruling R232.2).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Pending {
    /// A channel decode, by its [`Key`].
    Channel(Key),
    /// A shared axis decode, by its [`AxisKey`].
    Axis(AxisKey),
}

/// How long a waiter sleeps before re-checking whether the decode it is
/// waiting on has landed.
///
/// Not a deadline — a waiter never gives up, because the thread that marked
/// the key in flight always clears it (the mark is released by a `Drop`
/// guard, on the error path and on unwind alike). The interval only bounds
/// how long a missed notification could go unnoticed.
const PENDING_POLL: Duration = Duration::from_secs(5);

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
    /// The shared timestamp axes (ruling R232.1), keyed apart from channels
    /// so one entry serves every channel that borrows it and is charged to
    /// `resident_bytes` exactly once.
    axes: HashMap<AxisKey, Arc<AxisColumn>>,
    /// Insertion order of `axes`, coldest first — the eviction order for
    /// axes no resident channel still borrows.
    axis_order: Vec<AxisKey>,
    /// How many axis columns this cache has actually decoded, ever. The
    /// observable that makes "one axis decode per (session, source)"
    /// testable: six channels of one source must move it by two (the union
    /// `t` and that source's companion), not by twelve.
    axis_decodes: u64,
    /// Decodes marked in flight, so a second caller for the same channel or
    /// axis waits for the first rather than repeating it (ruling R232.2).
    pending: HashSet<Pending>,
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
struct Shared {
    inner: Mutex<Inner>,
    /// Signalled whenever bytes are given back — a reservation dropped, or
    /// an entry evicted — so waiting decodes re-check the budget.
    released: Condvar,
    /// Signalled whenever a decode marked in flight finishes, succeeds or
    /// fails, so the callers that deduped onto it re-check (ruling R232.2).
    decoded: Condvar,
    /// Where decode progress goes and how often, cloned out once per decode
    /// and never held across one.
    progress: Mutex<Progress>,
}

/// The decode-progress reporting settings — see
/// [`SessionCache::set_progress_sink`].
#[derive(Clone)]
struct Progress {
    /// `None` when nobody is listening: the CLI, every test that has not
    /// installed one, and the app before its setup hook runs. A decode then
    /// behaves exactly as it did before ruling R221.
    sink: Option<ProgressSink>,
    /// [`PROGRESS_AFTER`], unless a test shortened it.
    after: Duration,
    /// [`PROGRESS_INTERVAL`], unless a test shortened it.
    interval: Duration,
}

impl Default for Progress {
    fn default() -> Self {
        Progress { sink: None, after: PROGRESS_AFTER, interval: PROGRESS_INTERVAL }
    }
}

/// Hand-written because a [`ProgressSink`] is a `dyn Fn` and cannot derive
/// it; the sink is reported as present or absent rather than printed.
impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_sink = self.progress.lock().map(|p| p.sink.is_some()).unwrap_or(false);
        f.debug_struct("Shared").field("inner", &self.inner).field("has_progress_sink", &has_sink).finish()
    }
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
                    axes: HashMap::new(),
                    axis_order: Vec::new(),
                    axis_decodes: 0,
                    pending: HashSet::new(),
                    resident_bytes: 0,
                    budget_bytes,
                    in_flight_bytes: 0,
                }),
                released: Condvar::new(),
                decoded: Condvar::new(),
                progress: Mutex::new(Progress::default()),
            }),
        }
    }

    /// Installs the sink every later decode reports progress to (ruling
    /// R221 item 1), replacing any previous one.
    ///
    /// Called once, from the app's setup hook, with a closure that emits the
    /// C3 §3.2 `decode_progress` event. A cache with no sink decodes exactly
    /// as it did before this ruling: the callback threaded into
    /// `read_channel_with_progress` is never even built.
    pub fn set_progress_sink(&self, sink: ProgressSink) {
        self.set_progress_sink_with_timings(sink, PROGRESS_AFTER, PROGRESS_INTERVAL);
    }

    /// [`Self::set_progress_sink`] with explicit thresholds. Exists so tests
    /// can pin the delay and the throttle in milliseconds instead of waiting
    /// out [`PROGRESS_AFTER`]; production always uses `set_progress_sink`.
    pub fn set_progress_sink_with_timings(&self, sink: ProgressSink, after: Duration, interval: Duration) {
        let mut slot = self.shared.progress.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Progress { sink: Some(sink), after, interval };
    }

    /// The current reporting settings. Cloned per decode so the lock is
    /// never held while one runs.
    fn progress(&self) -> Progress {
        self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()).clone()
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
        loop {
            // Either this thread owns the decode, or another one finished it
            // while this one waited (ruling R232.2) — in which case `claim`
            // yields nothing to do and the entry is simply resident.
            let in_flight = self.claim(Pending::Channel(key.clone()), |inner| inner.touch(&key));
            if let Some(hit) = self.lock().touch(&key) {
                return Ok(hit);
            }
            // No entry and no claim: the leader failed and left nothing
            // behind. Go round again and lead the next attempt rather than
            // inherit a failure this thread never saw.
            let Some(_in_flight) = in_flight else { continue };

            let needed = estimate_channel_bytes(session_dir, channel).map_err(|e| store_error(session_id, channel, 0, e))?;
            let reservation = self.reserve(needed, &format!("decode channel '{channel}' of session '{session_id}'"))?;

            let decoded = self.decode_reporting_progress(session_dir, session_id, channel);
            let samples = Arc::new(decoded.map_err(|e| store_error(session_id, channel, needed, e))?);
            // Insert first, then give the reservation back: releasing it
            // before the bytes are accounted as resident would let a waiter
            // through on a budget this decode is still occupying.
            self.lock().insert(key, samples.clone());
            drop(reservation);
            return Ok(samples);
        }
    }

    /// Several channels of one session at once, decoded **in parallel**
    /// (ruling R232.2).
    ///
    /// This is what a cell bind, a report or a raster asks for: a set of
    /// channels, most of them cold, all wanted before anything can be drawn.
    /// Served one at a time they cost the sum of their decodes — the 44
    /// seconds the R221 measurement recorded for one 516 MB session. Here
    /// the misses fan out over [`decode_pool`]'s workers, each reserving
    /// through the same byte semaphore [`Self::reserve`] hands out, so they
    /// queue on memory rather than exhaust it.
    ///
    /// Returns one result per requested channel, **in the order asked** —
    /// a per-channel failure is that channel's own `Err`, never the
    /// request's, because one unknown name in a notebook must not cost the
    /// other eight their samples. Resident channels are returned without
    /// touching the pool at all.
    pub fn channels(
        &self,
        session_dir: &Path,
        session_id: &str,
        channels: &[&str],
    ) -> Vec<Result<Arc<ChannelSamples>, IpcError>> {
        if channels.len() < 2 {
            return channels.iter().map(|c| self.channel(session_dir, session_id, c)).collect();
        }
        match decode_pool() {
            Some(pool) => pool.install(|| {
                use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
                channels.par_iter().map(|c| self.channel(session_dir, session_id, c)).collect()
            }),
            // No pool on this machine (a thread-spawn failure) degrades to
            // serial, exactly as the indexing job's does: decoding slowly is
            // strictly better than not decoding at all.
            None => channels.iter().map(|c| self.channel(session_dir, session_id, c)).collect(),
        }
    }

    /// Decodes `channels` into residency for their own sake, discarding both
    /// the samples and any failure (ruling R232.2's "requests wait, never
    /// fail").
    ///
    /// The call a command makes when it knows up front which channels an
    /// evaluation will ask for one at a time: they arrive together, in
    /// parallel, and the lazy lookups that follow are all cache hits. A
    /// name this session does not have is simply not decoded — a prefetch
    /// reports nothing, because the lookup that really needs the channel
    /// runs afterwards and reports it properly.
    pub fn prefetch(&self, session_dir: &Path, session_id: &str, channels: &[&str]) {
        let _ = self.channels(session_dir, session_id, channels);
    }

    /// Marks `what` as being decoded by this thread, or waits for the thread
    /// that already is (ruling R232.2).
    ///
    /// `hit` is re-checked under the lock on every wake, so a waiter takes
    /// the leader's result rather than repeating its work: `Ok(None)` means
    /// "it is resident now, ask again". `Ok(Some(guard))` means this thread
    /// owns the decode and must do it; dropping the guard clears the mark
    /// and wakes everyone waiting, on the error path and on unwind alike.
    ///
    /// A leader that fails leaves nothing resident, so the first waiter to
    /// wake becomes the next leader and tries once itself. That costs a
    /// second attempt at a genuinely broken decode, and it is what keeps a
    /// transient failure (a file being rewritten under the reader) from
    /// being cached as a permanent one.
    fn claim<T>(&self, what: Pending, hit: impl Fn(&mut Inner) -> Option<T>) -> Option<InFlight<'_>> {
        let mut inner = self.lock();
        loop {
            if hit(&mut inner).is_some() {
                return None;
            }
            if !inner.pending.contains(&what) {
                inner.pending.insert(what.clone());
                return Some(InFlight { cache: self, what });
            }
            let (guard, _) = self
                .shared
                .decoded
                .wait_timeout(inner, PENDING_POLL)
                .unwrap_or_else(|e| e.into_inner());
            inner = guard;
        }
    }

    /// One decode of `channel`, borrowing this session's shared timestamp
    /// axes rather than decompressing them again (ruling R232.1).
    ///
    /// The union `t` column and the channel's `<source>_t_recorded_us`
    /// companion are taken from the cache when resident. When they are not,
    /// **this decode reads them in its own single pass** and leaves them
    /// behind for every later channel of that session and source --
    /// `read_channel_sharing_axes` projects the axes alongside the value
    /// column, so priming the cache costs no extra pass over the file.
    ///
    /// A cold axis is claimed before the read (`t` first, then the recorded
    /// companion -- one order for every caller, so two threads can never
    /// each hold the axis the other is waiting for). A thread that does not
    /// win the claim waits for the axis to land and then reads only its own
    /// column. That is what makes a cold fan-out worth doing: the first
    /// channel pays for the axes once, and the rest decode in parallel
    /// against them.
    ///
    /// A synthesized `Time`/`Distance`, and any name this file does not
    /// carry, falls through to the ordinary whole-decode read, which owns
    /// the answer and the error message for both.
    fn decode_borrowing_axes(
        &self,
        session_dir: &Path,
        session_id: &str,
        channel: &str,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<ChannelSamples, idl_rs::store::parquet::ParquetStoreError> {
        let index = read_channel_index(session_dir)?;
        let Some(info) = index.iter().find(|c| c.channel_id == channel) else {
            return read_channel_with_progress(session_dir, channel, on_progress);
        };
        let t_key: AxisKey = (session_id.to_string(), "t".to_string());
        let recorded_key: AxisKey = (session_id.to_string(), recorded_axis_column(&info.source_kind));

        // Claimed in this order by every caller, always: `t`, then the
        // recorded companion. A second order would let two threads each
        // wait on the axis the other holds.
        let t_claim = self.claim(Pending::Axis(t_key.clone()), |inner| inner.axis(&t_key));
        let recorded_claim = self.claim(Pending::Axis(recorded_key.clone()), |inner| inner.axis(&recorded_key));
        let (held_t, held_recorded) = {
            let inner = self.lock();
            (inner.axis(&t_key), inner.axis(&recorded_key))
        };

        let decoded = read_channel_sharing_axes(
            session_dir,
            channel,
            BorrowedAxes { t: held_t.as_deref(), recorded: held_recorded.as_deref() },
            on_progress,
        )?;

        {
            let mut inner = self.lock();
            if let Some(t) = decoded.t {
                inner.axis_decodes += 1;
                inner.insert_axis(t_key, Arc::new(t));
            }
            if let Some(recorded) = decoded.recorded {
                inner.axis_decodes += 1;
                inner.insert_axis(recorded_key, Arc::new(recorded));
            }
        }
        drop(t_claim);
        drop(recorded_claim);
        Ok(decoded.samples)
    }

    /// How many axis columns this cache has decoded since it was built
    /// (ruling R232.1's "count reads").
    pub fn axis_decodes(&self) -> u64 {
        self.lock().axis_decodes
    }

    /// Number of resident axis columns, across every session.
    pub fn axes_len(&self) -> usize {
        self.lock().axes.len()
    }

    /// One `read_channel`, reporting `decode_progress` to the installed sink
    /// (ruling R221 item 1).
    ///
    /// Two rules shape what is sent, both in this function rather than in
    /// core: nothing at all until the decode has been running
    /// [`PROGRESS_AFTER`] (a hover-speed decode must not flash a ring), and
    /// at most one event per [`PROGRESS_INTERVAL`] after that (a reader
    /// yielding a batch per 1024 rows would otherwise flood IPC). The one
    /// exception to the interval is the terminal `finished` event, which is
    /// always sent when any progress was — including on the error path,
    /// since a ring whose decode failed must stop, not spin.
    ///
    /// With no sink installed this is exactly `read_channel`.
    fn decode_reporting_progress(
        &self,
        session_dir: &Path,
        session_id: &str,
        channel: &str,
    ) -> Result<ChannelSamples, idl_rs::store::parquet::ParquetStoreError> {
        let settings = self.progress();
        let Some(sink) = settings.sink else {
            return self.decode_borrowing_axes(session_dir, session_id, channel, &mut |_, _| {});
        };

        let started = Instant::now();
        let mut last_sent: Option<Instant> = None;
        let mut last_counts = (0u64, 0u64);

        let result = {
            let mut report = |done: usize, total: usize| {
                last_counts = (done as u64, total as u64);
                let now = Instant::now();
                let due = match last_sent {
                    None => now.duration_since(started) >= settings.after,
                    Some(sent) => now.duration_since(sent) >= settings.interval,
                };
                if !due || total == 0 {
                    return;
                }
                last_sent = Some(now);
                (*sink)(DecodeProgressEvent {
                    session_id: session_id.to_string(),
                    channel: channel.to_string(),
                    done_rows: done as u64,
                    total_rows: total as u64,
                    finished: false,
                });
            };
            self.decode_borrowing_axes(session_dir, session_id, channel, &mut report)
        };

        if last_sent.is_some() {
            (*sink)(DecodeProgressEvent {
                session_id: session_id.to_string(),
                channel: channel.to_string(),
                done_rows: last_counts.0,
                total_rows: last_counts.1,
                finished: true,
            });
        }
        result
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

/// Points the app's managed [`SessionCache`] at the C3 §3.2
/// `decode_progress` event (ruling R221 item 1).
///
/// Called once from the app's setup hook, after `app.manage(SessionCache)`.
/// It lives here rather than in `app/src-tauri` so that the event name and
/// the payload type stay in the same file as the struct that defines them —
/// the app crate only says *when*, never *what*.
///
/// Emission failures are ignored, exactly as every other event in this crate
/// ignores them: a webview that has gone away is not a decode failure, and
/// there is no promise left to reject on a background decode.
pub fn install_progress_sink<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    use tauri::{Emitter, Manager};

    let handle = app.clone();
    let sink: ProgressSink = Arc::new(move |event: DecodeProgressEvent| {
        let _ = handle.emit("decode_progress", event);
    });
    app.state::<SessionCache>().set_progress_sink(sink);
}

/// The pool several channels of one request decode on (ruling R232.2),
/// built once for the process.
///
/// `None` when the pool would not build — a thread-spawn failure on an
/// exhausted machine — in which case [`SessionCache::channels`] decodes
/// serially instead, the same degradation the indexing job takes.
///
/// Its own pool, not rayon's global one, and the same width the indexing
/// job uses (`physical cores − 1`, [`worker_count`]): decodes must not
/// inherit another caller's width, and must not widen past what R208.1
/// decided this machine can carry. Width is not what bounds memory — the
/// byte semaphore is, and every worker here reserves through it.
fn decode_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(worker_count().max(1)).build().ok())
        .as_ref()
}

/// One thread's claim on a decode, clearing the mark and waking every
/// waiter when it is dropped (ruling R232.2).
///
/// Held for exactly as long as the decode: taken before the estimate, given
/// back when the entry is resident *or* the decode has failed. A `Drop`
/// impl rather than an explicit release so an early `?` and an unwinding
/// panic both free the mark — a leaked mark would park every later caller
/// for that channel forever.
#[derive(Debug)]
struct InFlight<'a> {
    cache: &'a SessionCache,
    what: Pending,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.cache.lock().pending.remove(&self.what);
        self.cache.shared.decoded.notify_all();
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

/// The fraction of the budget one indexing worker's reservation is clamped
/// to, as `(numerator, denominator)` — 4/5, chosen so that
/// [`crate::memory::with_estimate_margin`]'s own 5/4 lifts it back to
/// exactly the budget and never past it. See [`DecodeBudget for
/// SessionCache`](SessionCache#impl-DecodeBudget-for-SessionCache).
const INDEX_RESERVATION_CAP: (u64, u64) = (4, 5);

/// The indexing job's memory gate (ruling R208.1 item 3): a worker about to
/// decode a session's GPS reserves against the *same* budget every command's
/// decode reserves against, so N indexing workers plus whatever the notebook
/// is doing can never exceed one ceiling.
///
/// **Waits, never fails** (ruling R211's rule for the indexing path, where
/// there is no user to show a toast to). Two things make that safe:
///
/// - the request is clamped to [`INDEX_RESERVATION_CAP`] of the budget
///   before [`SessionCache::reserve`] applies its margin, so the
///   "larger than the whole budget" branch is unreachable and no amount of
///   waiting is ever futile;
/// - a [`RESERVE_TIMEOUT`] elapsing is retried rather than surfaced, because
///   a background job that gives up on a busy minute would leave the library
///   half-indexed for no reason.
///
/// A clamped reservation under-charges a genuinely enormous session, but
/// such a session can only run when nothing else holds the budget — which
/// is the same guarantee an unclamped wait would have given.
impl idl_rs::store::index_job::DecodeBudget for SessionCache {
    fn acquire<'a>(&'a self, bytes: u64, hint: &str) -> Box<dyn idl_rs::store::index_job::BudgetGuard + 'a> {
        let (num, den) = INDEX_RESERVATION_CAP;
        let capped = bytes.min(self.budget_bytes() / den * num).max(1);
        loop {
            if let Ok(reservation) = self.reserve(capped, hint) {
                return Box::new(reservation);
            }
        }
    }
}

impl idl_rs::store::index_job::BudgetGuard for Reservation<'_> {}

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

    /// The resident axis for `key`, or `None`. Axes have no LRU position of
    /// their own — what keeps one alive is a channel that borrows it
    /// ([`Self::axis_is_borrowed`]), not how recently it was touched.
    fn axis(&self, key: &AxisKey) -> Option<Arc<AxisColumn>> {
        self.axes.get(key).cloned()
    }

    /// Admits a decoded axis, charging its bytes to residency once.
    ///
    /// An axis larger than the whole budget is not retained, exactly as an
    /// oversized channel is not: the decode that produced it already holds
    /// its `Arc`, and admitting it would evict everything else to hold
    /// something that still does not fit.
    fn insert_axis(&mut self, key: AxisKey, axis: Arc<AxisColumn>) {
        let bytes = axis.resident_bytes() as u64;
        if bytes > self.budget_bytes || self.axes.contains_key(&key) {
            return;
        }
        self.axes.insert(key.clone(), axis);
        self.axis_order.push(key);
        self.resident_bytes += bytes;
        self.evict_to_budget();
    }

    /// `true` when some resident channel still reads through the axis
    /// `key` names — the union `t` of that session, or a
    /// `<source>_t_recorded_us` whose source kind that channel carries.
    ///
    /// …or when a decode that is using it right now still holds its own
    /// `Arc` — the cache's own handle is the only one when nothing else
    /// does, so `strong_count > 1` *is* "a decode is reading through this",
    /// with no bookkeeping to keep in step.
    ///
    /// This is R232.1's pin: evicting an axis out from under a resident
    /// channel would free nothing (the channel's `Arc` on it keeps the
    /// memory) and would make the next channel of that source decode it
    /// again, which is the exact cost the ruling exists to remove.
    fn axis_is_borrowed(&self, key: &AxisKey) -> bool {
        let (session_id, axis) = key;
        let Some(resident) = self.axes.get(key) else {
            return false;
        };
        if Arc::strong_count(resident) > 1 {
            return true;
        }
        self.entries.iter().any(|((sid, _), samples)| {
            sid == session_id && (axis == "t" || *axis == recorded_axis_column(&samples.source_kind))
        })
    }

    /// Drops every axis of the sessions no resident channel names any more.
    fn evict_unborrowed_axes(&mut self) {
        let stale: Vec<AxisKey> = self.axis_order.iter().filter(|k| !self.axis_is_borrowed(k)).cloned().collect();
        for key in stale {
            self.drop_axis(&key);
        }
    }

    /// Removes one axis entry and gives its bytes back.
    fn drop_axis(&mut self, key: &AxisKey) {
        if let Some(axis) = self.axes.remove(key) {
            self.resident_bytes = self.resident_bytes.saturating_sub(axis.resident_bytes() as u64);
        }
        self.axis_order.retain(|k| k != key);
    }

    /// Drops coldest-first until `resident_bytes <= budget_bytes`.
    ///
    /// Evicting only stops the cache *counting* those bytes: a channel a
    /// command is still holding an `Arc` to keeps its memory until that
    /// command returns. That is the honest limit of a byte budget over
    /// shared, immutable data, and it is why the reservation half of
    /// [`SessionCache::reserve`] exists — the in-flight counter bounds what
    /// is actually being allocated right now, which this counter cannot.
    /// Channels go first, coldest first; an axis is only ever dropped once
    /// the last channel borrowing it has gone (ruling R232.1), so a source
    /// whose channels are all resident keeps its axis however tight the
    /// budget gets.
    fn evict_to_budget(&mut self) {
        while self.resident_bytes > self.budget_bytes && !self.order.is_empty() {
            let coldest = self.order.remove(0);
            if let Some(dropped) = self.entries.remove(&coldest) {
                self.resident_bytes = self.resident_bytes.saturating_sub(dropped.resident_bytes() as u64);
            }
            self.evict_unborrowed_axes();
        }
        if self.resident_bytes > self.budget_bytes {
            self.evict_unborrowed_axes();
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
        // A session whose channels are gone has no axis worth keeping —
        // `data.parquet` is invalidated as a unit, so its axes are as stale
        // as its channels (ruling R232.1 follows R203.2's rule here).
        self.evict_unborrowed_axes();
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

        // Assert — 100 f64 samples plus 100 i64 timestamps, and the
        // session's shared axes charged once alongside them (R232.1).
        assert_eq!(samples.len(), 100);
        let axes: u64 = cache.lock().axes.values().map(|a| a.resident_bytes() as u64).sum();
        assert_eq!(cache.resident_bytes(), samples.resident_bytes() as u64 + axes);

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

        // Assert — only s2's channel survives, and only s2's axes with it.
        assert_eq!(cache.len(), 1);
        let axes: u64 = cache.lock().axes.values().map(|a| a.resident_bytes() as u64).sum();
        assert_eq!(cache.resident_bytes(), kept.resident_bytes() as u64 + axes);
        assert!(cache.lock().axes.keys().all(|(sid, _)| sid == "s2"));

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

    /// A sink that records every event it is handed, and the recording.
    fn recording_sink() -> (ProgressSink, Arc<Mutex<Vec<DecodeProgressEvent>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let into_sink = Arc::clone(&seen);
        let sink: ProgressSink = Arc::new(move |e: DecodeProgressEvent| {
            into_sink.lock().unwrap().push(e);
        });
        (sink, seen)
    }

    #[test]
    fn decode_progress_a_decode_shorter_than_the_delay_reports_nothing_at_all() {
        // Arrange -- production thresholds against a four-channel toy
        // session: nothing here takes 200 ms.
        let root = temp_root();
        seed(&root, "s1", 100);
        let cache = SessionCache::with_budget(1 << 30);
        let (sink, seen) = recording_sink();
        cache.set_progress_sink(sink);

        // Act
        cache.channel(&session_dir(&root, "s1"), "s1", "A").unwrap();

        // Assert -- a ring that appears and vanishes within a frame is noise.
        assert!(seen.lock().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn decode_progress_a_decode_past_the_delay_reports_the_session_channel_and_a_single_finished_event() {
        // Arrange -- zero delay, so every observation is past it.
        let root = temp_root();
        seed(&root, "s1", 100);
        let cache = SessionCache::with_budget(1 << 30);
        let (sink, seen) = recording_sink();
        cache.set_progress_sink_with_timings(sink, Duration::ZERO, Duration::ZERO);

        // Act
        cache.channel(&session_dir(&root, "s1"), "s1", "B").unwrap();

        // Assert
        let events = seen.lock().unwrap().clone();
        assert!(events.len() >= 2);
        assert!(events.iter().all(|e| e.session_id == "s1" && e.channel == "B" && e.total_rows > 0));
        assert!(events.iter().rev().skip(1).all(|e| !e.finished));
        let last = events.last().unwrap();
        assert!(last.finished);
        assert_eq!(last.done_rows, last.total_rows);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn decode_progress_a_cache_hit_reports_nothing_because_it_decodes_nothing() {
        // Arrange -- the first decode is warmed with no sink installed.
        let root = temp_root();
        seed(&root, "s1", 100);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);
        cache.channel(&dir, "s1", "C").unwrap();
        let (sink, seen) = recording_sink();
        cache.set_progress_sink_with_timings(sink, Duration::ZERO, Duration::ZERO);

        // Act
        cache.channel(&dir, "s1", "C").unwrap();

        // Assert
        assert!(seen.lock().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn decode_progress_a_failed_decode_still_sends_its_finished_event() {
        // Arrange -- `Distance` needs a GPS speed channel the seed has no
        // trace of, so its second pass fails after the first has reported.
        let root = temp_root();
        seed(&root, "s1", 100);
        let cache = SessionCache::with_budget(1 << 30);
        let (sink, seen) = recording_sink();
        cache.set_progress_sink_with_timings(sink, Duration::ZERO, Duration::ZERO);

        // Act
        let err = cache.channel(&session_dir(&root, "s1"), "s1", "Distance").unwrap_err();

        // Assert -- a ring whose decode failed must stop, not spin.
        assert_eq!(err.kind, IpcErrorKind::NotFound);
        let events = seen.lock().unwrap().clone();
        assert!(events.last().is_some_and(|e| e.finished));

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Seeds `session_id` with six `imu0` channels that record hardware
    /// stamps and two `gps` channels that do not — the shape ruling R232.1
    /// is about: many channels, few sources, one union axis.
    fn seed_two_sources(root: &Path, session_id: &str, n: usize) {
        let channel = |id: &str, source: &str, recorded: bool| Channel {
            channel_id: id.to_string(),
            t_us: (0..n).map(|i| i as i64 * 1000).collect(),
            t_recorded_us: if recorded { Some((0..n).map(|i| i as i64 * 1000 + 7).collect()) } else { None },
            nominal_rate_hz: 1000.0,
            column: RawColumn::F64((0..n).map(|i| i as f64).collect()),
            source_kind: source.to_string(),
            unit: "m/s".to_string(),
            gaps: Vec::new(),
        };
        let mut channels: Vec<Channel> = (0..6).map(|i| channel(&format!("IMU0_C{i}"), "imu0", true)).collect();
        channels.push(channel("GPS_SpeedKmh", "gps", false));
        channels.push(channel("GPS_Alt", "gps", false));
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels,
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn channel_six_channels_of_one_source_decode_that_sources_axis_exactly_once() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        for i in 0..6 {
            cache.channel(&dir, "s1", &format!("IMU0_C{i}")).unwrap();
        }

        // Assert — ruling R232.1: the union `t` and `imu0_t_recorded_us`,
        // twice in total, not twelve times.
        assert_eq!(cache.axis_decodes(), 2);
        assert_eq!(cache.axes_len(), 2);
        assert_eq!(cache.len(), 6);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_channels_of_a_second_source_reuse_the_union_axis_and_add_only_their_own() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        cache.channel(&dir, "s1", "IMU0_C0").unwrap();
        cache.channel(&dir, "s1", "GPS_SpeedKmh").unwrap();
        cache.channel(&dir, "s1", "GPS_Alt").unwrap();

        // Assert — three axis columns for eight channels: the union `t`
        // once, and one recorded companion per source (the writer emits one
        // for every source, C1 §3.2), never one per channel.
        assert_eq!(cache.axis_decodes(), 3);
        assert_eq!(cache.axes_len(), 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_borrowing_a_shared_axis_returns_the_same_samples_as_decoding_it_alone() {
        // Arrange — one cache warms the axes, a second decodes each channel
        // on its own; the two must not disagree by one sample.
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let shared = SessionCache::with_budget(1 << 30);
        shared.channel(&dir, "s1", "IMU0_C0").unwrap();

        // Act
        let borrowed = shared.channel(&dir, "s1", "IMU0_C3").unwrap();
        let alone = SessionCache::with_budget(1 << 30).channel(&dir, "s1", "IMU0_C3").unwrap();

        // Assert
        assert_eq!(borrowed.t_us, alone.t_us);
        assert_eq!(borrowed.t_recorded_us, alone.t_recorded_us);
        assert_eq!(borrowed.materialize(), alone.materialize());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resident_bytes_counts_every_channel_plus_each_axis_exactly_once() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let decoded: Vec<Arc<ChannelSamples>> =
            (0..6).map(|i| cache.channel(&dir, "s1", &format!("IMU0_C{i}")).unwrap()).collect();

        // Assert — six channels plus two axes, each charged once (R232.1).
        let channels: u64 = decoded.iter().map(|c| c.resident_bytes() as u64).sum();
        let axes: u64 = cache.lock().axes.values().map(|a| a.resident_bytes() as u64).sum();
        assert_eq!(cache.resident_bytes(), channels + axes);
        assert!(axes > 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn invalidate_session_drops_that_sessions_axes_alongside_its_channels() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);
        cache.channel(&dir, "s1", "IMU0_C0").unwrap();
        assert_eq!(cache.axes_len(), 2);

        // Act
        cache.invalidate_session("s1");

        // Assert — `data.parquet` is invalidated as a unit, so its axes are
        // as stale as its channels.
        assert_eq!(cache.axes_len(), 0);
        assert_eq!(cache.resident_bytes(), 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn evict_to_budget_never_drops_an_axis_a_resident_channel_still_borrows() {
        // Arrange — a budget tight enough to force eviction of channels.
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let sized = SessionCache::with_budget(1 << 30);
        let one = sized.channel(&dir, "s1", "IMU0_C0").unwrap().resident_bytes() as u64;
        let estimate = with_estimate_margin(estimate_channel_bytes(&dir, "IMU0_C0").unwrap());
        let cache = SessionCache::with_budget((one * 2).max(estimate));

        // Act
        for i in 0..6 {
            cache.channel(&dir, "s1", &format!("IMU0_C{i}")).unwrap();
        }

        // Assert — channels were evicted, but every resident channel still
        // has the axis it reads through (ruling R232.1's pin), so no later
        // channel of this source re-decodes it.
        assert!(cache.len() < 6);
        assert!(cache.len() > 0);
        assert_eq!(cache.axes_len(), 2);
        assert_eq!(cache.axis_decodes(), 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channels_a_batch_of_cold_channels_decodes_each_exactly_once_and_returns_them_in_order() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);
        let names: Vec<String> = (0..6).map(|i| format!("IMU0_C{i}")).collect();
        let wanted: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        // Act
        let got = cache.channels(&dir, "s1", &wanted);

        // Assert — one entry per request, in order, and the axes still
        // decoded twice between all six workers (rulings R232.1, R232.2).
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|r| r.is_ok()));
        assert_eq!(cache.len(), 6);
        assert_eq!(cache.axis_decodes(), 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channels_one_unknown_name_fails_only_its_own_slot() {
        // Arrange
        let root = temp_root();
        seed_two_sources(&root, "s1", 64);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);

        // Act
        let got = cache.channels(&dir, "s1", &["IMU0_C0", "NopeChannel", "GPS_Alt"]);

        // Assert — one unknown name in a notebook must not cost the others
        // their samples.
        assert!(got[0].is_ok());
        assert_eq!(got[1].as_ref().unwrap_err().kind, IpcErrorKind::NotFound);
        assert!(got[2].is_ok());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn channel_eight_threads_racing_for_one_cold_channel_decode_it_exactly_once() {
        // Arrange — a recording sink with no delay, so every decode that
        // actually runs leaves exactly one `finished` event behind.
        let root = temp_root();
        seed_two_sources(&root, "s1", 4096);
        let dir = session_dir(&root, "s1");
        let cache = SessionCache::with_budget(1 << 30);
        let (sink, seen) = recording_sink();
        cache.set_progress_sink_with_timings(sink, Duration::ZERO, Duration::ZERO);

        // Act
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let dir = dir.clone();
                std::thread::spawn(move || cache.channel(&dir, "s1", "IMU0_C0").map(|s| s.len()))
            })
            .collect();
        let lengths: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap().unwrap()).collect();

        // Assert — one decode, many waiters (ruling R232.2).
        let finished = seen.lock().unwrap().iter().filter(|e| e.finished && e.channel == "IMU0_C0").count();
        assert_eq!(finished, 1);
        assert_eq!(lengths, vec![4096; 8]);
        assert_eq!(cache.len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn decode_progress_a_decode_that_reads_its_axes_first_reports_monotonically_to_its_total() {
        // Arrange — nothing resident, so this decode makes three passes:
        // the union axis, the recorded companion, then its own column.
        let root = temp_root();
        seed_two_sources(&root, "s1", 4096);
        let cache = SessionCache::with_budget(1 << 30);
        let (sink, seen) = recording_sink();
        cache.set_progress_sink_with_timings(sink, Duration::ZERO, Duration::ZERO);

        // Act
        cache.channel(&session_dir(&root, "s1"), "s1", "IMU0_C0").unwrap();

        // Assert — ruling R232.3: one ring, advancing over real work, never
        // restarting at a column boundary.
        let events = seen.lock().unwrap().clone();
        assert!(events.len() >= 2);
        assert!(events.windows(2).all(|w| w[1].done_rows >= w[0].done_rows));
        assert!(events.iter().all(|e| e.done_rows <= e.total_rows));
        let last = events.last().unwrap();
        assert!(last.finished);
        assert_eq!(last.done_rows, last.total_rows);

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

