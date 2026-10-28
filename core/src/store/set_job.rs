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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::session::handle::{ChannelSource, SessionHandle, SessionMetaInput};
use crate::session::{Channel, ChannelSamples};
use crate::store::eval_cache::{catalog_identity, open_catalog_read_only, EvalCache, PRUNE_ABOVE_BYTES};
use crate::store::parquet::{open_session_lazy_with, read_channel, read_session_metadata, ParquetStoreErrorKind};
use crate::store::session_context::{whole_session_bounds, whole_session_context_without_data, window_lap_context};
use crate::store::session_json::{parse_session_json, SessionJson};
use crate::workbook::v3::session_set::{default_member_label, SessionSet};
use crate::workbook::v3::set_eval::{
    evaluate_member, laps_identity, member_key, plan_set, set_tables, CombineMember, MemberAbsence, MemberResults,
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
#[derive(Debug, Clone, Copy)]
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
    /// Polled between members; set, the run stops at the next boundary.
    pub cancel: Option<&'a AtomicBool>,
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
/// cache), counting decodes and keeping the first failure that is not "no
/// such channel" — a [`ChannelSource`] cannot return an error, and a refused
/// decode must not read as an unknown channel (R211.2), nor be cached.
#[derive(Debug)]
struct MemberSource {
    session_dir: PathBuf,
    decodes: AtomicUsize,
    first_error: Mutex<Option<SetMemberError>>,
}

impl ChannelSource for MemberSource {
    fn channel(&self, channel_id: &str) -> Option<Channel> {
        self.channel_samples(channel_id).map(|s| s.to_channel())
    }

    fn channel_samples(&self, channel_id: &str) -> Option<Arc<ChannelSamples>> {
        self.decodes.fetch_add(1, Ordering::Relaxed);
        match read_channel(&self.session_dir, channel_id) {
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
        _ => None,
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
        let identity = catalog_row.as_ref().map(|(i, _)| i.clone()).or_else(|| {
            has_data.then(|| read_session_metadata(&parquet).ok()).flatten().map(|m| {
                format!("{}|{}|{}", m.blob_sha256, m.importer_version, m.seam_correction_version)
            })
        });
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
            evaluate_with_data(&dir, &plan, doc, json.as_ref(), &mut report, |phase| {
                progress(&SetProgress { done, total, session_id: member.id.clone(), phase })
            })
        };

        match evaluated {
            Ok(member_results) => {
                if let (Some(c), Some(k)) = (&mut cache, &key) {
                    if let Err(e) = c.put_member(&member.id, k, &member_results, now_ms()) {
                        cache_error.get_or_insert(e.message);
                    }
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

/// Opens one member's `data.parquet` through its own [`MemberSource`],
/// evaluates the plan in its whole-session window, and drops every decoded
/// channel before returning.
fn evaluate_with_data(
    dir: &Path,
    plan: &SetPlan,
    doc: &WorkbookDoc,
    json: Option<&SessionJson>,
    report: &mut MemberReport,
    mut phase: impl FnMut(SetPhase),
) -> Result<MemberResults, SetMemberError> {
    let io = |message: String| SetMemberError { kind: SetMemberErrorKind::Io, message };
    let source = Arc::new(MemberSource {
        session_dir: dir.to_path_buf(),
        decodes: AtomicUsize::new(0),
        first_error: Mutex::new(None),
    });
    let handle = open_session_lazy_with(dir, source.clone() as Arc<dyn ChannelSource>)
        .map_err(|e| io(format!("reading {}: {}", dir.join("data.parquet").display(), e.message)))?;
    let bounds = whole_session_bounds(dir).map_err(|e| io(e.message))?;
    let ctx = window_lap_context(json, bounds);
    phase(SetPhase::Evaluating);
    let results = evaluate_member(plan, &doc.constants, &handle, &ctx);
    drop(handle);
    report.decoded_channels = source.decodes.load(Ordering::Relaxed);
    let failure = source.first_error.lock().unwrap_or_else(|p| p.into_inner()).take();
    match failure {
        Some(e) => Err(e),
        None => Ok(results),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        SetJobOptions { use_cache, engine_identity: Some("test-engine"), scope: PlanScope::Tables, cancel: None }
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
