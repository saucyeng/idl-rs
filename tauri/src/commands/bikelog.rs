//! Bike log and setup sheet commands (C3 §3.11, bike-log spec
//! `2026-10-16-idl1-bike-log-DRAFT.md`): thin glue over
//! `idl_rs::store::bikelog`. Every number the screens show (the running
//! sheet, which version is current, which notes are in conflict) is worked
//! out here, never in JavaScript.

use std::path::Path;

use idl_rs::store::bikelog::{self, Bike, BikeLogError, BikeLogErrorKind, Entry, EntryVersion, RunningSheet, SessionBikeLog};

use crate::error::{IpcError, IpcErrorKind};
use crate::state::DataDir;

impl From<BikeLogError> for IpcError {
    fn from(e: BikeLogError) -> Self {
        let kind = match e.kind {
            BikeLogErrorKind::Io => IpcErrorKind::Io,
            BikeLogErrorKind::Encode => IpcErrorKind::Internal,
            BikeLogErrorKind::InvalidId => IpcErrorKind::InvalidArgument,
        };
        IpcError::new(kind, e.message)
    }
}

/// UTC milliseconds now.
fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// One log entry as the screens need it: the stored entry plus what core
/// works out from its versions.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BikeLogEntryView {
    /// The entry with every version.
    pub entry: Entry,
    /// The version to show (the newest head).
    pub current: Option<EntryVersion>,
    /// Edited on two devices and not yet resolved: [`Self::heads`] holds
    /// each side's text to pick from.
    pub in_conflict: bool,
    /// The versions nothing was edited from, newest first.
    pub heads: Vec<EntryVersion>,
    /// Deleted (by its current version).
    pub deleted: bool,
}

/// `get_bike_log`'s return.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BikeLogView {
    /// Live entries and deleted ones, sorted by entry id.
    pub entries: Vec<BikeLogEntryView>,
    /// The running sheet now.
    pub running_sheet: RunningSheet,
    /// Files in the log folder that didn't parse, as paths.
    pub skipped: Vec<String>,
}

fn get_bike_log_via(data_dir: &Path, bike_id: &str, at_ms: i64) -> Result<BikeLogView, IpcError> {
    if !bikelog::is_valid_id(bike_id) {
        return Err(IpcError::new(IpcErrorKind::InvalidArgument, format!("{bike_id:?} is not a bike id")));
    }
    let load = bikelog::load_log(data_dir, bike_id);
    let running_sheet = bikelog::running_sheet(&load.entries, at_ms);
    let entries = load
        .entries
        .into_iter()
        .map(|e| BikeLogEntryView {
            current: e.current().cloned(),
            in_conflict: e.in_conflict(),
            heads: e.heads().into_iter().cloned().collect(),
            deleted: e.is_deleted(),
            entry: e,
        })
        .collect();
    Ok(BikeLogView { entries, running_sheet, skipped: load.skipped.into_iter().map(|(p, _)| p.display().to_string()).collect() })
}

/// C3 §3.11 `list_bikes()`: every bike, sorted by name.
#[tauri::command(async)]
pub fn list_bikes(data_dir: tauri::State<'_, DataDir>) -> Result<Vec<Bike>, IpcError> {
    Ok(bikelog::load_bikes(&data_dir.0).bikes)
}

/// C3 §3.11 `bike_for_name(name)`: the bike with that name (case ignored),
/// created when there is none. The logger's bike name comes in here.
#[tauri::command(async)]
pub fn bike_for_name(name: String, data_dir: tauri::State<'_, DataDir>) -> Result<Bike, IpcError> {
    Ok(bikelog::bike_for_name(&data_dir.0, &name, now_ms())?)
}

/// C3 §3.11 `save_bike(bike)`: saves a bike, merged with the copy on disk;
/// returns what was written.
#[tauri::command(async)]
pub fn save_bike(bike: Bike, data_dir: tauri::State<'_, DataDir>) -> Result<Bike, IpcError> {
    Ok(bikelog::save_bike(&data_dir.0, &bike)?)
}

/// C3 §3.11 `get_bike_log(bike_id)`: the bike's log and its running sheet
/// now.
#[tauri::command(async)]
pub fn get_bike_log(bike_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<BikeLogView, IpcError> {
    get_bike_log_via(&data_dir.0, &bike_id, now_ms())
}

/// C3 §3.11 `save_bike_log_entry(entry)`: saves an entry (a new one, or new
/// versions of one), merged with the copy on disk so no version is lost;
/// returns what was written.
#[tauri::command(async)]
pub fn save_bike_log_entry(entry: Entry, data_dir: tauri::State<'_, DataDir>) -> Result<Entry, IpcError> {
    Ok(bikelog::save_entry(&data_dir.0, &entry)?)
}

/// C3 §3.11 `attach_bike_log(session_id, bike_id)`: records that the
/// session was ridden on the bike and gives it the log entries since that
/// bike's previous session plus the running sheet at its start. The
/// session's start and end come from the catalog.
#[tauri::command(async)]
pub fn attach_bike_log(session_id: String, bike_id: String, data_dir: tauri::State<'_, DataDir>) -> Result<SessionBikeLog, IpcError> {
    let summary = idl_rs::store::catalog_read::list_sessions(&data_dir.0)?
        .into_iter()
        .find(|s| s.session_id == session_id)
        .ok_or_else(|| IpcError::new(IpcErrorKind::NotFound, format!("session {session_id} not found")))?;
    let start_ms = summary.timestamp_utc_ms;
    let end_ms = start_ms + summary.duration_ms.unwrap_or(0);
    let doc = bikelog::attach_session(&data_dir.0, &session_id, &bike_id, start_ms, end_ms)?;
    Ok(doc.bike_log.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::store::bikelog::EntryKind;

    fn version(id: &str, parents: &[&str], updated_at_ms: i64, text: &str) -> EntryVersion {
        EntryVersion {
            version_id: id.to_string(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            at_ms: 10,
            updated_at_ms,
            device: "phone".to_string(),
            text: text.to_string(),
            changes: Vec::new(),
            sheet: Default::default(),
            deleted: false,
        }
    }

    #[test]
    fn get_bike_log_via_reports_current_version_and_conflicts() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-tauri-bikelog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let entry = Entry {
            entry_id: "e1".into(),
            bike_id: "gt".into(),
            kind: EntryKind::Note,
            versions: vec![version("v1", &[], 1, "a"), version("p", &["v1"], 2, "phone"), version("d", &["v1"], 3, "desk")],
        };
        bikelog::save_entry(&root, &entry).unwrap();

        // Act
        let view = get_bike_log_via(&root, "gt", 100).unwrap();

        // Assert
        assert_eq!(view.entries.len(), 1);
        assert!(view.entries[0].in_conflict);
        assert_eq!(view.entries[0].current.as_ref().unwrap().text, "desk");
        assert_eq!(view.entries[0].heads.iter().map(|v| v.text.as_str()).collect::<Vec<_>>(), vec!["desk", "phone"]);
        assert_eq!(view.running_sheet.since, vec!["e1"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn get_bike_log_via_an_unsafe_bike_id_is_invalid_argument() {
        // Arrange
        let root = std::env::temp_dir();

        // Act
        let err = get_bike_log_via(&root, "../x", 0).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
    }
}
