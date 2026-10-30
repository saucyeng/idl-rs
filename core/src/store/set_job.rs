//! The session-set runner (session-sets spec §4.1, §4.5; ruling Q8 of
//! 2026-10-28: it lives in core, beside [`crate::store::index_job`], so the
//! app and the CLI cannot schedule members differently).
//!
//! For each member, in set order, **one at a time** (phase 1, P = 1):
//!
//! 1. look its results up in the cache (`<data>/cache/eval/results.sqlite`)
//!    under its member key — a full hit opens no `data.parquet` at all;
//! 2. otherwise evaluate the plan's demand in that member alone, through its
//!    own channel source (never the app's `SessionCache`, so a set run never
//!    evicts what the user is looking at), keep only reduced results, and
//!    drop the member's data before the next one starts;
//! 3. store the results, in that member's own transaction.
//!
//! Then combine every member's results into the set tables
//! ([`crate::workbook::v3::set_eval::set_tables`]).
//!
//! Members are independent, so a member that cannot be evaluated is reported
//! and the run continues. Cancellation is checked at member boundaries.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::session::handle::{ChannelSource, SessionHandle, SessionMetaInput};
use crate::session::{Channel, ChannelSamples};
use crate::store::eval_cache::{catalog_identity, open_catalog_read_only, EvalCache, PRUNE_ABOVE_BYTES};
use crate::store::index_job::{MemoryBudget, NeverFits};
use crate::store::parquet::{
    axis_bytes_ceiling, open_session_lazy_with, read_session_metadata, ParquetStoreErrorKind, SessionColumnReader,
};
use crate::workbook::v3::plan::{PlanInputs, ESTIMATE_MARGIN};
use crate::workbook::v3::EvalObserver;
use crate::store::session_context::{whole_session_bounds, whole_session_context_without_data, window_lap_context};
use crate::store::session_json::{parse_session_json, SessionJson};
use crate::workbook::v3::session_set::{default_member_label, SessionSet};
use crate::workbook::v3::set_eval::{
    evaluate_member, evaluate_member_planned, laps_identity, member_eval_plan, member_key, plan_set, set_tables, CombineMember, MemberAbsence, MemberResults,
    PlanScope, SetPlan, SetTable, SetWarning,
};
use crate::workbook::v3::WorkbookDoc;

/// The build's engine identity for persisted results (ruling Q4 of
/// 2026-10-28): the git commit the engine was built from, stamped by
/// `build.rs`, or `None` for a build made outside a git checkout — such a
/// build does not use the disk cache.
pub fn engine_identity() -> Option<&'static str> {
    option_env!("IDL_RS_GIT_COMMIT").filter(|s| !s.is_empty())
}

/// What one run does.
#[derive(Clone, Copy)]
pub struct SetJobOptions<'a> {
    /// Read and write `<data>/cache/eval/results.sqlite`. `false` reads
    /// nothing from it and writes nothing to the data directory (the CLI's
    /// `--no-cache`).
    pub use_cache: bool,
    /// The engine identity results are keyed by; `None` disables the cache
    /// whatever `use_cache` says. Normally [`engine_identity`].
    pub engine_identity: Option<&'a str>,
    /// Which definitions each member evaluates.
    pub scope: PlanScope,
    /// Polled between members, while a member waits for memory, and
    /// between a member's definitions; set, the run stops there and the
    /// member in flight is discarded (members already finished stay cached).
    pub cancel: Option<&'a AtomicBool>,
    /// The memory ledger each member is admitted against before it decodes
    /// (evaluation-memory spec §4.2.G): its planned estimate is held for the
    /// member's evaluation and released when the member ends. The app passes
    /// its `SessionCache`, the CLI a [`crate::store::index_job::ByteBudget`];
    /// `None` admits everything (tests).
    pub budget: Option<&'a dyn MemoryBudget>,
}

/// One member's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberStatus {
    /// Evaluated in this run.
    Computed,
    /// Served from the cache; its data is here too.
    Cached,
    /// Served from the cache; its data is not on this device.
    CachedDataElsewhere,
    /// Its data is not on this device, nothing is cached, and its demand
    /// reads channels (Q3: its rows show as missing).
    Missing,
    /// No such session in the library (no `session.json`, no data).
    NotFound,
    /// Evaluation could not run: see [`MemberReport::error`].
    Error,
}

impl MemberStatus {
    /// The C3 `SessionSetResult.members[].status` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            MemberStatus::Computed => "computed",
            MemberStatus::Cached => "cached",
            MemberStatus::CachedDataElsewhere => "cached_data_elsewhere",
            MemberStatus::Missing => "missing",
            MemberStatus::NotFound => "not_found",
            MemberStatus::Error => "error",
        }
    }
}

/// Why a member's evaluation could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetMemberErrorKind {
    /// `data.parquet` could not be opened or a channel could not be read.
    Io,
    /// The allocator refused a buffer a decode needed (C3 `resource_exhausted`).
    ResourceExhausted,
}

/// A member-level failure (never stored in the cache).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetMemberError {
    /// What kind of failure.
    pub kind: SetMemberErrorKind,
    /// Display-ready English.
    pub message: String,
}

/// One member as the run reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberReport {
    /// Session id.
    pub session_id: String,
    /// Display label: the authored one, else the default
    /// ([`default_member_label`]).
    pub label: String,
    /// Index into the set's groups.
    pub group: usize,
    /// Session start, ms since the Unix epoch, when known.
    pub start_utc_ms: Option<i64>,
    /// Outcome.
    pub status: MemberStatus,
    /// Set when `status` is [`MemberStatus::Error`].
    pub error: Option<SetMemberError>,
    /// Channel decodes this member performed (0 when served from the cache
    /// or from `session.json` alone).
    pub decoded_channels: usize,
    /// Where this member's time went, for profiling.
    pub timing: MemberTiming,
}

/// Wall-clock time one member spent in each step, ms. Decoding happens
/// lazily inside the evaluation, so `evaluate_ms` excludes `decode_ms`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MemberTiming {
    /// Opening `data.parquet`, planning, and waiting for admission.
    pub open_ms: f64,
    /// Channel decodes (`read_channel`), summed.
    pub decode_ms: f64,
    /// The evaluation itself, decodes excluded (unit inference included).
    pub evaluate_ms: f64,
    /// Writing the member's rows to the result cache.
    pub cache_write_ms: f64,
    /// The five slowest definitions, `(name, ms)`, slowest first; each
    /// includes the channel decodes it triggered.
    pub slowest: Vec<(String, f64)>,
}

/// Which step a member is in, for progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetPhase {
    /// Served from the cache.
    Cached,
    /// Opening its data.
    Decoding,
    /// Evaluating its demand.
    Evaluating,
}

impl SetPhase {
    /// The C3 `SetProgress.phase` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            SetPhase::Cached => "cached",
            SetPhase::Decoding => "decoding",
            SetPhase::Evaluating => "evaluating",
        }
    }
}

/// One progress report: emitted as each member enters each phase.
#[derive(Debug, Clone, PartialEq)]
pub struct SetProgress {
    /// Members finished before this one.
    pub done: usize,
    /// Members in the set.
    pub total: usize,
    /// The member now in progress.
    pub session_id: String,
    /// Its phase.
    pub phase: SetPhase,
}

/// Everything one run produced.
#[derive(Debug, Clone)]
pub struct SetJobReport {
    /// SHA-256 hex of the resolved set and the maths text: the identity of
    /// what was evaluated.
    pub set_key: String,
    /// The plan each member evaluated.
    pub plan: SetPlan,
    /// One report per member, in set order.
    pub members: Vec<MemberReport>,
    /// Each member's kept results, in set order (`None` when it has none).
    pub results: Vec<Option<MemberResults>>,
    /// Every session-row table, in document order.
    pub tables: Vec<SetTable>,
    /// Q3 warnings.
    pub warnings: Vec<SetWarning>,
    /// True when the run stopped on a cancel; members after the boundary are
    /// absent from `members` and nothing is combined.
    pub cancelled: bool,
    /// Why the cache could not be used, when it could not (the run went on
    /// without it).
    pub cache_error: Option<String>,
}

/// A member's channels, read straight from its `data.parquet` (no shared
/// cache) through one [`SessionColumnReader`] -- the footer parsed once, the
/// time axes decoded once -- counting decodes and keeping the first failure
/// that is not "no such channel" — a [`ChannelSource`] cannot return an
/// error, and a refused decode must not read as an unknown channel (R211.2),
/// nor be cached.
#[derive(Debug)]
struct MemberSource {
    reader: SessionColumnReader,
    decodes: AtomicUsize,
    /// Time spent in `SessionColumnReader::read_channel`, µs, summed.
    decode_us: std::sync::atomic::AtomicU64,
    first_error: Mutex<Option<SetMemberError>>,
}

impl ChannelSource for MemberSource {
    fn channel(&self, channel_id: &str) -> Option<Channel> {
        self.channel_samples(channel_id).map(|s| s.to_channel())
    }

    fn channel_samples(&self, channel_id: &str) -> Option<Arc<ChannelSamples>> {
        self.decodes.fetch_add(1, Ordering::Relaxed);
        let started = std::time::Instant::now();
        let read = self.reader.read_channel(channel_id);
        self.decode_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        match read {
            Ok(s) => Some(Arc::new(s)),
            Err(e) if e.kind == ParquetStoreErrorKind::NotFound => None,
            Err(e) => {
                let kind = if e.kind == ParquetStoreErrorKind::ResourceExhausted {
                    SetMemberErrorKind::ResourceExhausted
                } else {
                    SetMemberErrorKind::Io
                };
                let mut slot = self.first_error.lock().unwrap_or_else(|p| p.into_inner());
                slot.get_or_insert(SetMemberError { kind, message: format!("{channel_id}: {}", e.message) });
                None
            }
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// An empty session for a member evaluated from `session.json` alone.
fn no_data_handle(session_id: &str) -> SessionHandle {
    SessionHandle::from_channels(
        SessionMetaInput { session_id: session_id.to_string(), device_id: None, timestamp_utc_ms: 0, config_checksum: None },
        Vec::new(),
    )
}

/// SHA-256 hex of the resolved set and the plan's maths text.
fn set_key(set: &SessionSet, plan: &SetPlan) -> String {
    let set_text = serde_yaml_ng::to_string(&set.to_yaml()).unwrap_or_default();
    crate::store::atomic::sha256_hex(format!("{set_text}\u{1e}{}", plan.maths_text).as_bytes())
}

/// Evaluates `set` over `doc` against the library at `data_dir` (spec §4.1):
/// members one at a time, in set order, then the combine. `progress` is
/// called as each member enters each phase.
///
/// Never fails as a whole: each member's failure is in its
/// [`MemberReport`], and a cache that cannot be opened is reported in
/// [`SetJobReport::cache_error`] while the run goes on without it.
pub fn run_session_set(
    data_dir: &Path,
    doc: &WorkbookDoc,
    set: &SessionSet,
    options: SetJobOptions,
    progress: &mut dyn FnMut(&SetProgress),
) -> SetJobReport {
    let plan = plan_set(doc, options.scope);
    let set_key = set_key(set, &plan);
    let flat = set.members();
    let total = flat.len();

    let mut cache_error = None;
    let mut cache = match (options.use_cache, options.engine_identity) {
        (true, Some(_)) => match EvalCache::open(data_dir) {
            Ok(c) => Some(c),
            Err(e) => {
                cache_error = Some(e.message);
                None
            }
        },
        (true, None) => {
            cache_error = Some("result cache off: this engine build has no clean git commit to key results by".into());
            None
        }
        (false, _) => None,
    };
    let catalog = open_catalog_read_only(data_dir);

    let mut members = Vec::with_capacity(total);
    let mut results: Vec<Option<MemberResults>> = Vec::with_capacity(total);
    let mut cancelled = false;

    for (done, member) in flat.iter().enumerate() {
        if options.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            cancelled = true;
            break;
        }
        let dir = data_dir.join("sessions").join(&member.id);
        let json_bytes = std::fs::read(dir.join("session.json")).ok();
        let json: Option<SessionJson> = json_bytes.as_deref().and_then(|b| parse_session_json(b).ok());
        let parquet = dir.join("data.parquet");
        let has_data = parquet.exists();

        let catalog_row = catalog.as_ref().and_then(|c| catalog_identity(c, &member.id));
        // The data's identity comes from the file itself whenever the file is
        // here: the catalog is a deletable index that can lag a re-import, and
        // a lagging row would key a stale result. The catalog row speaks only
        // for a member whose data is elsewhere. An unreadable footer is no
        // identity at all (no cache), never the catalog's.
        let identity = if has_data {
            read_session_metadata(&parquet)
                .ok()
                .map(|m| format!("{}|{}|{}", m.blob_sha256, m.importer_version, m.seam_correction_version))
        } else {
            catalog_row.as_ref().map(|(i, _)| i.clone())
        };
        let start_utc_ms = catalog_row.as_ref().map(|(_, t)| *t);
        let label = member
            .label
            .clone()
            .unwrap_or_else(|| default_member_label(&member.id, json.as_ref().map(|j| j.event_session.as_str())));
        let mut report = MemberReport {
            session_id: member.id.clone(),
            label,
            group: member.group,
            start_utc_ms,
            status: MemberStatus::Computed,
            error: None,
            decoded_channels: 0,
            timing: MemberTiming::default(),
        };

        if json.is_none() && !has_data {
            report.status = MemberStatus::NotFound;
            members.push(report);
            results.push(None);
            continue;
        }

        let key = match (&identity, options.engine_identity) {
            (Some(identity), Some(engine)) => Some(member_key(&plan, identity, &laps_identity(json.as_ref()), engine)),
            _ => None,
        };
        if let (Some(c), Some(k)) = (&cache, &key) {
            if let Ok(Some(hit)) = c.get_member(k, &plan.needed, now_ms()) {
                progress(&SetProgress { done, total, session_id: member.id.clone(), phase: SetPhase::Cached });
                report.status = if has_data { MemberStatus::Cached } else { MemberStatus::CachedDataElsewhere };
                members.push(report);
                results.push(Some(hit));
                continue;
            }
        }

        if !has_data && plan.reads_channels {
            report.status = MemberStatus::Missing;
            members.push(report);
            results.push(None);
            continue;
        }

        let evaluated = if !has_data || !plan.reads_channels {
            // Lap and sector statistics only: `session.json` is enough, and no
            // channel is ever decoded (spec §4.2).
            progress(&SetProgress { done, total, session_id: member.id.clone(), phase: SetPhase::Evaluating });
            let ctx = match has_data.then(|| whole_session_bounds(&dir).ok()).flatten() {
                Some(bounds) => window_lap_context(json.as_ref(), bounds),
                None => whole_session_context_without_data(json.as_ref()),
            };
            Ok(evaluate_member(&plan, &doc.constants, &no_data_handle(&member.id), &ctx))
        } else {
            progress(&SetProgress { done, total, session_id: member.id.clone(), phase: SetPhase::Decoding });
            match evaluate_with_data(&dir, &plan, doc, json.as_ref(), &options, &mut report, |phase| {
                progress(&SetProgress { done, total, session_id: member.id.clone(), phase })
            }) {
                Ok(r) => Ok(r),
                Err(MemberRunError::Failed(e)) => Err(e),
                Err(MemberRunError::Cancelled) => {
                    cancelled = true;
                    break;
                }
            }
        };

        match evaluated {
            Ok(member_results) => {
                if let (Some(c), Some(k)) = (&mut cache, &key) {
                    let started = std::time::Instant::now();
                    if let Err(e) = c.put_member(&member.id, k, &member_results, now_ms()) {
                        cache_error.get_or_insert(e.message);
                    }
                    report.timing.cache_write_ms = started.elapsed().as_secs_f64() * 1e3;
                }
                members.push(report);
                results.push(Some(member_results));
            }
            Err(error) => {
                report.status = MemberStatus::Error;
                report.error = Some(error);
                members.push(report);
                results.push(None);
            }
        }
    }

    if let Some(c) = &cache {
        if let Err(e) = c.prune(PRUNE_ABOVE_BYTES) {
            cache_error.get_or_insert(e.message);
        }
    }

    let (tables, warnings) = if cancelled {
        (Vec::new(), Vec::new())
    } else {
        let combine: Vec<CombineMember> = members
            .iter()
            .zip(&results)
            .map(|(m, r)| CombineMember {
                session_id: &m.session_id,
                label: &m.label,
                group: m.group,
                results: match (r, m.status) {
                    (Some(r), _) => Ok(r),
                    (None, MemberStatus::NotFound) => Err(MemberAbsence::NotFound),
                    (None, MemberStatus::Error) => {
                        Err(MemberAbsence::Failed(m.error.as_ref().map(|e| e.message.clone()).unwrap_or_default()))
                    }
                    (None, _) => Err(MemberAbsence::Missing),
                },
            })
            .collect();
        set_tables(doc, set, &plan, &combine)
    };

    SetJobReport { set_key, plan, members, results, tables, warnings, cancelled, cache_error }
}

/// Why [`evaluate_with_data`] produced no results.
enum MemberRunError {
    /// The member could not be evaluated (reported on the member).
    Failed(SetMemberError),
    /// The run was cancelled while the member waited or evaluated.
    Cancelled,
}

/// The run's cancel flag as an [`EvalObserver`]. It is asked before each
/// scheduled definition starts, so it also stamps the time of each ask: the
/// gap between two asks is one definition's time.
struct CancelFlag<'a> {
    cancel: Option<&'a AtomicBool>,
    marks: Mutex<Vec<std::time::Instant>>,
}

impl EvalObserver for CancelFlag<'_> {
    fn cancelled(&self) -> bool {
        self.marks.lock().unwrap_or_else(|p| p.into_inner()).push(std::time::Instant::now());
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }
}

/// How often a cancellable admission asks again, ms.
const ADMIT_POLL_MS: u64 = 50;

/// Opens one member's `data.parquet` through its own [`MemberSource`], plans
/// its evaluation against the file's channel index, admits the plan's
/// estimate against `options.budget` (every input counts as not resident:
/// this source shares no cache), evaluates in the whole-session window, and
/// drops every decoded channel and the admission before returning.
fn evaluate_with_data(
    dir: &Path,
    plan: &SetPlan,
    doc: &WorkbookDoc,
    json: Option<&SessionJson>,
    options: &SetJobOptions,
    report: &mut MemberReport,
    mut phase: impl FnMut(SetPhase),
) -> Result<MemberResults, MemberRunError> {
    let io = |message: String| MemberRunError::Failed(SetMemberError { kind: SetMemberErrorKind::Io, message });
    let opened = std::time::Instant::now();
    let reader = SessionColumnReader::open(dir)
        .map_err(|e| io(format!("reading {}: {}", dir.join("data.parquet").display(), e.message)))?;
    let inputs = PlanInputs::from_channel_index(reader.index());
    let eval_plan = member_eval_plan(plan, &doc.constants, &inputs);
    // The reader keeps the session's time axes for the member's life, so they
    // are charged once, here, on top of the plan's per-channel inputs.
    let axes = axis_bytes_ceiling(reader.index(), eval_plan.inputs.iter().map(|i| i.channel.as_str()));
    let (margin_num, margin_den) = ESTIMATE_MARGIN;
    let needed = eval_plan
        .estimate_with_missing_inputs(|_| true)
        .saturating_add(axes.saturating_mul(margin_num) / margin_den);
    let hint = format!("evaluate session-set member '{}'", report.session_id);
    let observer = CancelFlag { cancel: options.cancel, marks: Mutex::new(Vec::new()) };

    let _admission = match options.budget {
        None => None,
        Some(budget) => {
            let exhausted = |e: NeverFits| {
                MemberRunError::Failed(SetMemberError { kind: SetMemberErrorKind::ResourceExhausted, message: e.to_string() })
            };
            if options.cancel.is_none() {
                Some(budget.acquire(needed, &hint).map_err(exhausted)?)
            } else {
                loop {
                    if observer.cancelled() {
                        return Err(MemberRunError::Cancelled);
                    }
                    if let Some(guard) = budget.try_acquire(needed, &hint).map_err(exhausted)? {
                        break Some(guard);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(ADMIT_POLL_MS));
                }
            }
        }
    };

    let source = Arc::new(MemberSource {
        reader,
        decodes: AtomicUsize::new(0),
        decode_us: std::sync::atomic::AtomicU64::new(0),
        first_error: Mutex::new(None),
    });
    let handle = open_session_lazy_with(dir, source.clone() as Arc<dyn ChannelSource>)
        .map_err(|e| io(format!("reading {}: {}", dir.join("data.parquet").display(), e.message)))?;
    let bounds = whole_session_bounds(dir).map_err(|e| io(e.message))?;
    let ctx = window_lap_context(json, bounds);
    report.timing.open_ms = opened.elapsed().as_secs_f64() * 1e3;
    phase(SetPhase::Evaluating);
    observer.marks.lock().unwrap_or_else(|p| p.into_inner()).clear();
    let evaluating = std::time::Instant::now();
    let results = evaluate_member_planned(plan, &doc.constants, &handle, &ctx, &eval_plan, Some(&observer));
    drop(handle);
    let finished = std::time::Instant::now();
    let marks = std::mem::take(&mut *observer.marks.lock().unwrap_or_else(|p| p.into_inner()));
    let mut per_def: Vec<(String, f64)> = marks
        .iter()
        .enumerate()
        .filter_map(|(k, start)| {
            let end = marks.get(k + 1).copied().unwrap_or(finished);
            let name = plan.defs.get(*eval_plan.schedule.get(k)?)?.name.clone();
            Some((name, (end - *start).as_secs_f64() * 1e3))
        })
        .collect();
    per_def.sort_by(|a, b| b.1.total_cmp(&a.1));
    per_def.truncate(5);
    report.timing.slowest = per_def;
    report.timing.decode_ms = source.decode_us.load(Ordering::Relaxed) as f64 / 1e3;
    report.timing.evaluate_ms = evaluating.elapsed().as_secs_f64() * 1e3 - report.timing.decode_ms;
    report.decoded_channels = source.decodes.load(Ordering::Relaxed);
    let results = results.map_err(|_| MemberRunError::Cancelled)?;
    let failure = source.first_error.lock().unwrap_or_else(|p| p.into_inner()).take();
    match failure {
        Some(e) => Err(MemberRunError::Failed(e)),
        None => Ok(results),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::workbook::v3::parse_workbook;
    use crate::workbook::v3::session_set::parse_session_set;

    fn temp_root() -> PathBuf {
        let d = std::env::temp_dir().join(format!("idl-rs-set-job-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Writes `sessions/<id>/data.parquet` with one 1 Hz `Fork` channel and a
    /// `session.json` with one lap of `lap_s` seconds.
    fn seed(root: &Path, id: &str, fork: Vec<f64>, lap_s: f64) {
        use crate::session::{RawColumn, Session, SourceFormat, TimestampSource};
        let dir = root.join("sessions").join(id);
        let session = Session {
            session_id: id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: format!("blob-{id}"),
            channels: vec![Channel {
                channel_id: "Fork".to_string(),
                t_us: (0..fork.len() as i64).map(|i| i * 1_000_000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1.0,
                column: RawColumn::F64(fork),
                source_kind: "test".to_string(),
                unit: String::new(),
                gaps: Vec::new(),
            }],
        };
        crate::store::parquet::write_session_parquet(root, &session, "0.1.0").unwrap();
        let mut doc = crate::store::session_json::empty_session_json(id);
        doc.laps = vec![crate::store::session_json::LapJson {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 0,
            raw_elapsed_ms: (lap_s * 1000.0) as i64,
            lap_time_ms: (lap_s * 1000.0) as i64,
            start_time_secs: 0.0,
            end_time_secs: lap_s,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        std::fs::write(dir.join("session.json"), serde_json::to_vec(&doc).unwrap()).unwrap();
    }

    const MD: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Set\nsessions:\n  - group: G\n    members:\n      - { id: s1 }\n      - { id: s2 }\n      - { id: s3 }\n---\n\n```math id=aaaaaaaa\npeak = max([Fork])\nlt = sum(lap_time())\n```\n\n```table id=bbbbbbbb\n{\"rowSource\":\"sessions\",\"columns\":[{\"id\":\"c0\",\"template\":\"[peak]\"}],\"rows\":[],\"cells\":[]}\n```\n\n```table id=cccccccc\n{\"rowSource\":\"sessions\",\"columns\":[{\"id\":\"c0\",\"template\":\"[lt]\"}],\"rows\":[],\"cells\":[]}\n```\n";

    fn options<'a>(use_cache: bool) -> SetJobOptions<'a> {
        SetJobOptions { use_cache, engine_identity: Some("test-engine"), scope: PlanScope::Tables, cancel: None, budget: None }
    }

    fn run(root: &Path, md: &str, opts: SetJobOptions) -> SetJobReport {
        let doc = parse_workbook(md).unwrap().0;
        let set = parse_session_set(&doc.front_matter_unknown).unwrap().unwrap();
        run_session_set(root, &doc, &set, opts, &mut |_| {})
    }

    #[test]
    fn set_job_three_members_one_absent_status_missing_and_the_other_rows_computed() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0, 4.0], 10.0);
        seed(&root, "s3", vec![2.0, 9.0], 12.0);

        // Act
        let report = run(&root, MD, options(false));

        // Assert
        let statuses: Vec<_> = report.members.iter().map(|m| m.status).collect();
        assert_eq!(statuses, vec![MemberStatus::Computed, MemberStatus::NotFound, MemberStatus::Computed]);
        let peaks: Vec<_> = report.tables[0].results.iter().map(|r| r[0].value).collect();
        assert_eq!(peaks, vec![Some(4.0), None, Some(9.0)]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_data_parquet_absent_but_session_json_present_status_missing() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);
        seed(&root, "s2", vec![1.0], 10.0);
        seed(&root, "s3", vec![1.0], 10.0);
        std::fs::remove_file(root.join("sessions/s2/data.parquet")).unwrap();

        // Act
        let report = run(&root, MD, options(false));

        // Assert
        assert_eq!(report.members[1].status, MemberStatus::Missing);
        assert_eq!(report.members[0].status, MemberStatus::Computed);
        assert!(report.warnings.iter().any(|w| w.missing == vec!["s2".to_string()]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn demand_a_table_of_lap_statistics_only_no_channel_is_opened() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);
        seed(&root, "s2", vec![1.0], 11.0);
        seed(&root, "s3", vec![1.0], 12.0);
        let lap_only = MD.replace("[peak]", "[lt]");

        // Act
        let report = run(&root, &lap_only, options(false));

        // Assert
        assert!(report.members.iter().all(|m| m.decoded_channels == 0 && m.status == MemberStatus::Computed));
        let lts: Vec<_> = report.tables[0].results.iter().map(|r| r[0].value).collect();
        assert_eq!(lts, vec![Some(10.0), Some(11.0), Some(12.0)]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn result_cache_reopen_with_every_member_cached_zero_data_parquet_opens() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0, 4.0], 10.0);
        seed(&root, "s2", vec![3.0], 10.0);
        seed(&root, "s3", vec![2.0, 9.0], 12.0);
        let first = run(&root, MD, options(true));

        // Act
        let second = run(&root, MD, options(true));

        // Assert
        assert!(first.members.iter().all(|m| m.status == MemberStatus::Computed && m.decoded_channels > 0));
        assert!(second.members.iter().all(|m| m.status == MemberStatus::Cached && m.decoded_channels == 0));
        assert_eq!(first.tables, second.tables);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn result_cache_a_refused_decode_not_stored() {
        // Arrange — s2's file is truncated after its footer was read, so the
        // channel decode fails.
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);
        seed(&root, "s2", vec![2.0], 10.0);
        seed(&root, "s3", vec![3.0], 10.0);
        let bad = root.join("sessions/s2/data.parquet");
        let bytes = std::fs::read(&bad).unwrap();
        let mut corrupt = bytes.clone();
        for b in corrupt.iter_mut().take(bytes.len() / 2).skip(4) {
            *b = 0xFF;
        }
        std::fs::write(&bad, &corrupt).unwrap();

        // Act
        let first = run(&root, MD, options(true));
        std::fs::write(&bad, &bytes).unwrap();
        let second = run(&root, MD, options(true));

        // Assert
        assert_eq!(first.members[1].status, MemberStatus::Error);
        assert_eq!(second.members[1].status, MemberStatus::Computed);
        assert_eq!(second.members[0].status, MemberStatus::Cached);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_cancelled_before_the_first_member_stops_at_the_boundary() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);
        let cancel = AtomicBool::new(true);

        // Act
        let report = run(&root, MD, SetJobOptions { cancel: Some(&cancel), ..options(false) });

        // Assert
        assert!(report.cancelled);
        assert!(report.members.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_a_member_larger_than_the_budget_ceiling_is_resource_exhausted_and_the_rest_still_run() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0; 4], 10.0);
        let tiny = crate::store::index_job::ByteBudget::new(1);

        // Act
        let report = run(&root, MD, SetJobOptions { budget: Some(&tiny), ..options(false) });

        // Assert
        assert_eq!(report.members[0].status, MemberStatus::Error);
        assert_eq!(report.members[0].error.as_ref().map(|e| e.kind), Some(SetMemberErrorKind::ResourceExhausted));
        assert_eq!(report.members[1].status, MemberStatus::NotFound);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_admitted_against_a_budget_that_fits_computes_and_releases_the_budget() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0, 4.0], 10.0);
        let budget = crate::store::index_job::ByteBudget::new(1 << 30);

        // Act
        let report = run(&root, MD, SetJobOptions { budget: Some(&budget), ..options(false) });

        // Assert
        assert_eq!(report.members[0].status, MemberStatus::Computed);
        assert_eq!(budget.in_flight_bytes(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn result_cache_a_narrower_run_cached_then_every_definition_asked_for_all_of_them_served() {
        // Arrange — a Tables run stores the table's closure only.
        let root = temp_root();
        seed(&root, "s1", vec![1.0, 4.0], 10.0);
        seed(&root, "s2", vec![3.0], 10.0);
        seed(&root, "s3", vec![2.0, 9.0], 12.0);
        let md = MD.replace("lt = sum(lap_time())", "lt = sum(lap_time())\nextra = min([Fork])");
        let narrow = run(&root, &md, options(true));

        // Act
        let wide = run(&root, &md, SetJobOptions { scope: PlanScope::AllDefinitions, ..options(true) });

        // Assert
        assert!(narrow.members.iter().all(|m| m.status == MemberStatus::Computed));
        assert!(wide.members.iter().all(|m| m.status == MemberStatus::Computed));
        for r in wide.results.iter().flatten() {
            assert!(r.0.contains_key("extra"), "{:?}", r.0.keys());
        }
        let again = run(&root, &md, SetJobOptions { scope: PlanScope::AllDefinitions, ..options(true) });
        assert!(again.members.iter().all(|m| m.status == MemberStatus::Cached));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn data_identity_data_parquet_present_and_a_stale_catalog_row_the_footer_keys_the_result() {
        // Arrange — cache a run, then a catalog row claiming other data
        // appears. The file did not change, so the cached result still holds.
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);
        let first = run(&root, MD, options(true));
        let conn = rusqlite::Connection::open(root.join("catalog.sqlite")).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (session_id TEXT, blob_sha256 TEXT, importer_version TEXT, seam_correction_version TEXT, timestamp_utc_ms INTEGER);
             INSERT INTO sessions VALUES ('s1', 'stale-blob', '0.0.1', 'v0', 0);",
        )
        .unwrap();
        drop(conn);

        // Act
        let second = run(&root, MD, options(true));

        // Assert
        assert_eq!(first.members[0].status, MemberStatus::Computed);
        assert_eq!(second.members[0].status, MemberStatus::Cached);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_no_engine_identity_cache_off_and_said_so() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);

        // Act
        let report = run(&root, MD, SetJobOptions { engine_identity: None, ..options(true) });

        // Assert
        assert!(report.cache_error.as_deref().unwrap_or("").contains("result cache off"));
        assert!(!root.join("cache").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn set_job_no_cache_writes_nothing_under_the_data_dir() {
        // Arrange
        let root = temp_root();
        seed(&root, "s1", vec![1.0], 10.0);

        // Act
        let _ = run(&root, MD, options(false));

        // Assert
        assert!(!root.join("cache").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
