//! C3 §3.4 `eval_session_set` (session-sets spec §5.2, phase 1): evaluates a
//! workbook's session set and returns its session-row tables.
//!
//! Glue only: the runner is `idl_rs::store::set_job`, the same core function
//! the CLI's `workbook eval --data-dir` runs. It decodes each member through
//! its own source, never the app's `SessionCache`, so evaluating a set never
//! evicts the session the user is looking at (spec §6.2).
//!
//! **Supersession.** A newer call for the same workbook sets the running
//! call's cancel flag; the running call stops while its member waits for
//! memory, between that member's definitions, or at the next member boundary,
//! and resolves with `cancelled: true` (the caller ignores it — a newer
//! answer is on its way). Members already finished stay in the cache.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use idl_rs::store::set_job::{engine_identity, run_session_set, SetJobOptions, SetJobReport, SetMemberErrorKind, SetProgress};
use idl_rs::table::{CellResult, TableModel};
use idl_rs::workbook::v3::parse_workbook;
use idl_rs::workbook::v3::session_set::parse_session_set;
use idl_rs::workbook::v3::set_eval::PlanScope;

use crate::commands::workbook::{fatal_parse_error, resolve_workbook_path};
use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

/// The running `eval_session_set` call per workbook id, by its cancel flag.
/// Managed for the app's lifetime (`app.manage(SetRuns::default())`).
#[derive(Default)]
pub struct SetRuns(Mutex<HashMap<String, Arc<AtomicBool>>>);

impl SetRuns {
    /// Registers a new run of `workbook_id`, cancelling the one it supersedes.
    fn begin(&self, workbook_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        let mut runs = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(previous) = runs.insert(workbook_id.to_string(), flag.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
        flag
    }

    /// Forgets `flag`'s run of `workbook_id`, unless a newer one replaced it.
    fn end(&self, workbook_id: &str, flag: &Arc<AtomicBool>) {
        let mut runs = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if runs.get(workbook_id).is_some_and(|f| Arc::ptr_eq(f, flag)) {
            runs.remove(workbook_id);
        }
    }
}

/// C3 `SetProgress`: emitted as each member enters each phase.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SetProgressDto {
    /// Members finished before this one.
    pub done: usize,
    /// Members in the set.
    pub total: usize,
    /// The member in progress.
    pub session_id: String,
    /// `"cached"` | `"decoding"` | `"evaluating"`.
    pub phase: &'static str,
}

impl From<&SetProgress> for SetProgressDto {
    fn from(p: &SetProgress) -> Self {
        Self { done: p.done, total: p.total, session_id: p.session_id.clone(), phase: p.phase.as_str() }
    }
}

/// C3 `SessionSetResult.groups[]`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SetGroupDto {
    /// The group's name; `null` for the one unnamed group.
    pub name: Option<String>,
}

/// C3 `SessionSetResult.members[]`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SetMemberDto {
    /// Session id.
    pub session_id: String,
    /// Display label.
    pub label: String,
    /// Index into `groups`.
    pub group: usize,
    /// Session start, ms since the Unix epoch; `null` when the catalog does
    /// not know it.
    pub start_utc_ms: Option<i64>,
    /// `"computed" | "cached" | "cached_data_elsewhere" | "missing" |
    /// "not_found" | "error"`.
    pub status: &'static str,
    /// Set when `status` is `"error"`: `io` or `resource_exhausted`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<IpcError>,
}

/// C3 `SessionSetResult.tables[]`: one evaluated session-row table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SetTableDto {
    /// The table cell's id.
    pub cell_id: String,
    /// The planned model: one row per member (row id = session id).
    pub model: TableModel,
    /// One result per planned cell.
    pub results: Vec<Vec<CellResult>>,
    /// Each row's member index into `members`, in row order — the app reads
    /// the row's group (for group headers) and status from it.
    pub row_members: Vec<usize>,
}

/// C3 `SessionSetResult.warnings[]` (ruling Q3 of 2026-10-28): a value over
/// the set was computed over the members present; `missing` names each one
/// left out, for the app's offer to fetch it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SetWarningDto {
    /// The table cell concerned; `null` for the whole set.
    pub cell_id: Option<String>,
    /// Session ids left out, in set order.
    pub missing: Vec<String>,
    /// Display-ready English.
    pub message: String,
}

/// C3 `SessionSetResult`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSetResult {
    /// SHA-256 hex of the resolved set and the maths text.
    pub set_key: String,
    /// The set's groups, in order.
    pub groups: Vec<SetGroupDto>,
    /// Every member, in set order.
    pub members: Vec<SetMemberDto>,
    /// Every session-row table, in document order.
    pub tables: Vec<SetTableDto>,
    /// Q3 warnings.
    pub warnings: Vec<SetWarningDto>,
    /// True when a newer call superseded this one; nothing else is complete.
    pub cancelled: bool,
}

fn result_from(report: SetJobReport, groups: Vec<SetGroupDto>) -> SessionSetResult {
    SessionSetResult {
        set_key: report.set_key,
        groups,
        members: report
            .members
            .into_iter()
            .map(|m| SetMemberDto {
                status: m.status.as_str(),
                error: m.error.map(|e| {
                    let kind = match e.kind {
                        SetMemberErrorKind::Io => IpcErrorKind::Io,
                        SetMemberErrorKind::ResourceExhausted => IpcErrorKind::ResourceExhausted,
                    };
                    IpcError::new(kind, e.message)
                }),
                session_id: m.session_id,
                label: m.label,
                group: m.group,
                start_utc_ms: m.start_utc_ms,
            })
            .collect(),
        tables: report
            .tables
            .into_iter()
            .map(|t| SetTableDto { cell_id: t.cell_id, model: t.model, results: t.results, row_members: t.row_members })
            .collect(),
        warnings: report
            .warnings
            .into_iter()
            .map(|w| SetWarningDto { cell_id: w.cell_id, missing: w.missing, message: w.message })
            .collect(),
        cancelled: report.cancelled,
    }
}

/// Transport-agnostic core of `eval_session_set`.
///
/// # Errors
/// `not_found` for an unknown workbook; `io` for an unreadable file; the
/// fatal workbook kinds for a front matter that does not parse;
/// `workbook_duplicate_set_member` / `workbook_invalid_session_set` for a
/// malformed `sessions` key; `invalid_argument` when the workbook declares
/// no session set. Everything per member is in `members`.
pub fn eval_session_set_via(
    data_dir: &Path,
    budget: &dyn idl_rs::store::index_job::MemoryBudget,
    runs: &SetRuns,
    workbook_id: &str,
    progress: &mut dyn FnMut(&SetProgress),
) -> Result<SessionSetResult, IpcError> {
    let path = resolve_workbook_path(data_dir, workbook_id)?;
    let markdown = std::fs::read_to_string(&path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {e}", path.display())))?;
    let (doc, _) = parse_workbook(&markdown).map_err(fatal_parse_error)?;
    let set = parse_session_set(&doc.front_matter_unknown).map_err(|e| IpcError::from(&e))?.ok_or_else(|| {
        IpcError::new(IpcErrorKind::InvalidArgument, format!("workbook '{workbook_id}' declares no sessions"))
    })?;
    let groups = set.groups.iter().map(|g| SetGroupDto { name: g.name.clone() }).collect();

    let cancel = runs.begin(workbook_id);
    let options = SetJobOptions {
        use_cache: true,
        engine_identity: engine_identity(),
        scope: PlanScope::Tables,
        cancel: Some(&cancel),
        budget: Some(budget),
    };
    let report = run_session_set(data_dir, &doc, &set, options, progress);
    runs.end(workbook_id, &cancel);
    Ok(result_from(report, groups))
}

/// C3 §3.4 `eval_session_set(workbook_id, on_progress)`. Async: it reads
/// every member's data and computes over it (CLAUDE.md §3, R201).
#[tauri::command(async)]
pub fn eval_session_set(
    workbook_id: String,
    on_progress: tauri::ipc::Channel<SetProgressDto>,
    data_dir: tauri::State<'_, DataDir>,
    runs: tauri::State<'_, SetRuns>,
    cache: tauri::State<'_, crate::session_cache::SessionCache>,
) -> Result<SessionSetResult, IpcError> {
    // Each member is admitted against the app's one memory ledger, the same
    // budget its own evaluations, decodes and cache count against.
    eval_session_set_via(&data_dir.0, &*cache, &runs, &workbook_id, &mut |p| {
        let _ = on_progress.send(SetProgressDto::from(p));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-set-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("workbooks")).unwrap();
        dir
    }

    fn seed(root: &Path, id: &str, value: f64) {
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
                t_us: vec![0, 1_000_000],
                t_recorded_us: None,
                nominal_rate_hz: 1.0,
                column: RawColumn::F64(vec![value, value * 2.0]),
                source_kind: "test".to_string(),
                unit: String::new(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
        let json = idl_rs::store::session_json::empty_session_json(id);
        std::fs::write(root.join("sessions").join(id).join("session.json"), serde_json::to_vec(&json).unwrap()).unwrap();
    }

    const WB: &str = "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: Set\nsessions:\n  - group: G\n    members:\n      - { id: s1, label: A }\n      - { id: s2, label: B }\n      - { id: s3, label: C }\n---\n\n```math id=aaaaaaaa\npeak = max([Fork])\n```\n\n```table id=bbbbbbbb\n{\"rowSource\":\"sessions\",\"columns\":[{\"id\":\"c0\",\"template\":\"[peak]\"}],\"rows\":[],\"cells\":[]}\n```\n";

    #[test]
    fn eval_session_set_via_three_member_fixture_one_data_parquet_absent_status_missing_the_other_rows_computed() {
        // Arrange
        let root = temp_root();
        std::fs::write(root.join("workbooks/set.idl1wb"), WB).unwrap();
        seed(&root, "s1", 1.0);
        seed(&root, "s2", 2.0);
        seed(&root, "s3", 3.0);
        std::fs::remove_file(root.join("sessions/s2/data.parquet")).unwrap();
        let mut phases = Vec::new();

        // Act
        let result = eval_session_set_via(&root, &crate::session_cache::SessionCache::new(), &SetRuns::default(), "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d", &mut |p| {
            phases.push((p.done, p.session_id.clone()))
        })
        .unwrap();

        // Assert
        let statuses: Vec<_> = result.members.iter().map(|m| m.status).collect();
        assert_eq!(statuses[0], "computed");
        assert_eq!(statuses[1], "missing");
        assert_eq!(statuses[2], "computed");
        let values: Vec<_> = result.tables[0].results.iter().map(|r| r[0].value).collect();
        assert_eq!(values, vec![Some(2.0), None, Some(6.0)]);
        assert_eq!(result.warnings.last().unwrap().missing, vec!["s2".to_string()]);
        // Members evaluate one at a time, in set order.
        assert!(phases.windows(2).all(|w| w[0].0 <= w[1].0));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_session_set_via_superseded_by_a_newer_call_stops_the_member_in_flight_and_the_rest() {
        // Arrange
        let root = temp_root();
        std::fs::write(root.join("workbooks/set.idl1wb"), WB).unwrap();
        seed(&root, "s1", 1.0);
        seed(&root, "s2", 2.0);
        seed(&root, "s3", 3.0);
        let runs = SetRuns::default();
        let id = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

        // Act — a second call starts while the first is in its first member.
        let mut started_second = false;
        let first = eval_session_set_via(&root, &crate::session_cache::SessionCache::new(), &runs, id, &mut |p| {
            if p.done == 0 && !started_second {
                started_second = true;
                runs.begin(id);
            }
        })
        .unwrap();

        // Assert
        assert!(first.cancelled);
        assert!(first.members.is_empty());
        assert!(first.tables.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_session_set_via_a_workbook_without_sessions_invalid_argument() {
        // Arrange
        let root = temp_root();
        std::fs::write(root.join("workbooks/x.idl1wb"), "---\nid: 9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d\nname: X\n---\n").unwrap();

        // Act
        let err = eval_session_set_via(&root, &crate::session_cache::SessionCache::new(), &SetRuns::default(), "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d", &mut |_| {})
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eval_session_set_via_duplicate_member_workbook_duplicate_set_member() {
        // Arrange
        let root = temp_root();
        let md = WB.replace("{ id: s2, label: B }", "{ id: s1, label: B }");
        std::fs::write(root.join("workbooks/set.idl1wb"), md).unwrap();

        // Act
        let err = eval_session_set_via(&root, &crate::session_cache::SessionCache::new(), &SetRuns::default(), "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d", &mut |_| {})
            .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::WorkbookDuplicateSetMember);
        let _ = std::fs::remove_dir_all(&root);
    }
}
