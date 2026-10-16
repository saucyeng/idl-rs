//! The bike log and setup sheet (spec `2026-10-16-idl1-bike-log-DRAFT.md`).
//!
//! Each bike has a log: one file per entry under
//! `<data>/bikelog/<bike_id>/`, named `note-<entry_id>.json` or
//! `sheet-<entry_id>.json` so the kind shows in the folder listing and the
//! confirmed sheets can be read without opening every note (Isaac,
//! 2026-10-16), plus the bike itself at
//! `<data>/bikes/<bike_id>.json`. An entry is a **note** (text, optionally
//! exact field changes) or a confirmed **sheet** (every value). An entry
//! keeps every version of itself, so two devices' copies merge as a union
//! of versions and no edit is lost (§4). The running setup sheet is never
//! stored: [`running_sheet`] works it out from the log.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry, AtomicWriteError};

/// One bike (spec §4): its name and which setup-sheet fields it has. The
/// two parts merge separately, each by its own `*_updated_at_ms`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bike {
    /// Stable id, also the file and log-folder name.
    pub bike_id: String,
    /// Display name, e.g. "GT Force".
    #[serde(default)]
    pub name: String,
    /// When `name` last changed, UTC ms.
    #[serde(default)]
    pub name_updated_at_ms: i64,
    /// The sheet fields this bike has (dotted keys, e.g. `"fork.lsc"`), in
    /// display order. A field not listed doesn't apply to this bike.
    #[serde(default)]
    pub fields: Vec<String>,
    /// When `fields` last changed, UTC ms.
    #[serde(default)]
    pub fields_updated_at_ms: i64,
}

impl Bike {
    /// Merges two copies of the same bike part by part, the newer of each
    /// part winning; a tie keeps `self`'s.
    pub fn merge(&self, other: &Bike) -> Bike {
        let mut out = self.clone();
        if other.name_updated_at_ms > self.name_updated_at_ms {
            out.name = other.name.clone();
            out.name_updated_at_ms = other.name_updated_at_ms;
        }
        if other.fields_updated_at_ms > self.fields_updated_at_ms {
            out.fields = other.fields.clone();
            out.fields_updated_at_ms = other.fields_updated_at_ms;
        }
        out
    }
}

/// What an entry is (spec §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// Text, optionally with exact field changes: "something changed".
    Note,
    /// Every value on the sheet: a fresh baseline.
    Sheet,
}

/// One exact field change carried by a note, e.g. fork LSC to 12 clicks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldChange {
    /// Dotted sheet key, e.g. `"fork.lsc"`.
    pub field: String,
    /// The new absolute value (clicks from fully closed, psi, mm, text…).
    pub value: serde_json::Value,
}

/// One version of an entry. Versions are never edited or removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryVersion {
    /// Unique id of this version.
    pub version_id: String,
    /// The version(s) this one was edited from: none for the first, two for
    /// a merge of a conflict.
    #[serde(default)]
    pub parents: Vec<String>,
    /// When the noted thing happened, UTC ms (editable).
    pub at_ms: i64,
    /// When this version was written, UTC ms.
    pub updated_at_ms: i64,
    /// Which device wrote it, e.g. "phone".
    #[serde(default)]
    pub device: String,
    /// The note's text (a sheet may carry a comment here).
    #[serde(default)]
    pub text: String,
    /// A note's exact field changes, in order.
    #[serde(default)]
    pub changes: Vec<FieldChange>,
    /// A sheet's values, by dotted key. Empty for a note.
    #[serde(default)]
    pub sheet: BTreeMap<String, serde_json::Value>,
    /// This version deletes the entry.
    #[serde(default)]
    pub deleted: bool,
}

/// One log entry: its identity and every version of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Unique id, also the file name; sorts by creation time when it is a ULID.
    pub entry_id: String,
    /// The bike whose log holds it.
    pub bike_id: String,
    /// Note or sheet; fixed for the entry's life.
    pub kind: EntryKind,
    /// Every version, kept in (`updated_at_ms`, `version_id`) order.
    pub versions: Vec<EntryVersion>,
}

impl Entry {
    /// The versions no other version was edited from. One head is a clean
    /// entry; more than one means it was edited on two devices before they
    /// synced. Newest first.
    pub fn heads(&self) -> Vec<&EntryVersion> {
        let parents: HashSet<&str> = self.versions.iter().flat_map(|v| v.parents.iter().map(String::as_str)).collect();
        let mut heads: Vec<&EntryVersion> = self.versions.iter().filter(|v| !parents.contains(v.version_id.as_str())).collect();
        heads.sort_by(|a, b| (b.updated_at_ms, &b.version_id).cmp(&(a.updated_at_ms, &a.version_id)));
        heads
    }

    /// The version to show: the newest head. `None` only for an entry with
    /// no versions, which a well-formed file never has.
    pub fn current(&self) -> Option<&EntryVersion> {
        self.heads().into_iter().next()
    }

    /// Edited on two devices and not yet resolved (spec §4).
    pub fn in_conflict(&self) -> bool {
        self.heads().len() > 1
    }

    /// Deleted, judged by its current version.
    pub fn is_deleted(&self) -> bool {
        self.current().map_or(true, |v| v.deleted)
    }

    /// Merges two copies of the same entry: the union of their versions, so
    /// neither side's edit is lost. Versions with the same id are the same
    /// version; `self`'s copy is kept.
    pub fn merge(&self, other: &Entry) -> Entry {
        let mut out = self.clone();
        let have: HashSet<String> = out.versions.iter().map(|v| v.version_id.clone()).collect();
        out.versions.extend(other.versions.iter().filter(|v| !have.contains(&v.version_id)).cloned());
        out.versions.sort_by(|a, b| (a.updated_at_ms, &a.version_id).cmp(&(b.updated_at_ms, &b.version_id)));
        out
    }
}

/// The running setup sheet at one moment (spec §2).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RunningSheet {
    /// The confirmed sheet it starts from, or `None` before the first one.
    #[serde(default)]
    pub baseline: Option<String>,
    /// Every value: the baseline's, with later notes' changes applied.
    #[serde(default)]
    pub values: BTreeMap<String, serde_json::Value>,
    /// Notes since the baseline, in time order: their changes are applied,
    /// and those without changes are the sheet's "changed since baseline"
    /// list.
    #[serde(default)]
    pub since: Vec<String>,
}

/// What a session took from its bike's log, stored in its `session.json`
/// (spec §4, §6).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SessionBikeLog {
    /// The entries attached, in time order.
    #[serde(default)]
    pub entry_ids: Vec<String>,
    /// The window entries were taken from: after `after_ms` (the bike's
    /// previous session's `until_ms`; `None` for its first) up to and
    /// including `until_ms` (this session's end), UTC ms.
    #[serde(default)]
    pub after_ms: Option<i64>,
    /// See [`SessionBikeLog::after_ms`].
    pub until_ms: i64,
    /// The running sheet at the session's start: R190's per-session
    /// snapshot.
    #[serde(default)]
    pub setup_sheet: RunningSheet,
}

/// The running sheet at `at_ms`: the newest live confirmed sheet at or
/// before `at_ms`, then the changes of every live note after it, up to and
/// including `at_ms`, in `at_ms` order (ties by entry id). Deleted entries
/// count for nothing; an entry in conflict counts by its newest head.
pub fn running_sheet(entries: &[Entry], at_ms: i64) -> RunningSheet {
    let mut live: Vec<(&Entry, &EntryVersion)> = entries
        .iter()
        .filter_map(|e| e.current().filter(|v| !v.deleted && v.at_ms <= at_ms).map(|v| (e, v)))
        .collect();
    live.sort_by(|a, b| (a.1.at_ms, &a.0.entry_id).cmp(&(b.1.at_ms, &b.0.entry_id)));

    let mut out = RunningSheet::default();
    let start = live.iter().rposition(|(e, _)| e.kind == EntryKind::Sheet);
    if let Some(i) = start {
        out.baseline = Some(live[i].0.entry_id.clone());
        out.values = live[i].1.sheet.clone();
    }
    for (e, v) in live.iter().skip(start.map_or(0, |i| i + 1)) {
        if e.kind != EntryKind::Note {
            continue;
        }
        for c in &v.changes {
            out.values.insert(c.field.clone(), c.value.clone());
        }
        out.since.push(e.entry_id.clone());
    }
    out
}

/// The entries a session takes at import (spec §6): every live entry whose
/// `at_ms` is after `after_ms` (the bike's previous session's end, or
/// `None` for its first session) and at or before `until_ms` (this
/// session's end), in `at_ms` order.
pub fn entries_for_session(entries: &[Entry], after_ms: Option<i64>, until_ms: i64) -> Vec<String> {
    let mut picked: Vec<(i64, &str)> = entries
        .iter()
        .filter_map(|e| e.current().filter(|v| !v.deleted).map(|v| (v.at_ms, e.entry_id.as_str())))
        .filter(|(t, _)| after_ms.map_or(true, |a| *t > a) && *t <= until_ms)
        .collect();
    picked.sort();
    picked.into_iter().map(|(_, id)| id.to_string()).collect()
}

/// The bike named `name` (trimmed, case ignored), creating it when there
/// is none. A new bike's id is the name in lowercase letters, digits and
/// `-` ("GT Force" is `gt-force`), with `-2`, `-3`… added if taken.
pub fn bike_for_name(data_root: &Path, name: &str, now_ms: i64) -> Result<Bike, BikeLogError> {
    let wanted = name.trim();
    let load = load_bikes(data_root);
    if let Some(b) = load.bikes.iter().find(|b| b.name.trim().eq_ignore_ascii_case(wanted)) {
        return Ok(b.clone());
    }
    let mut slug = String::new();
    for c in wanted.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
    }
    let slug = slug.trim_end_matches('-').to_string();
    let base = if slug.is_empty() { "bike".to_string() } else { slug };
    let taken = |id: &str| load.bikes.iter().any(|b| b.bike_id == id) || bike_path(data_root, id).exists();
    let mut bike_id = base.clone();
    let mut n = 2;
    while taken(&bike_id) {
        bike_id = format!("{base}-{n}");
        n += 1;
    }
    save_bike(data_root, &Bike { bike_id, name: wanted.to_string(), name_updated_at_ms: now_ms, fields: Vec::new(), fields_updated_at_ms: 0 })
}

/// Records that `session_id` was ridden on `bike_id` (spec §6): sets its
/// `session.json`'s `bike_id`, `bike` (the bike's name) and `bike_log` —
/// every live entry after the bike's previous session (the latest other
/// session of this bike whose window ends at or before `start_ms`) up to
/// `end_ms`, and the running sheet at `start_ms`. Running it again gives
/// the same result for an unchanged log, so the ride form can re-run it
/// after changing the bike.
pub fn attach_session(data_root: &Path, session_id: &str, bike_id: &str, start_ms: i64, end_ms: i64) -> Result<crate::store::session_json::SessionJson, BikeLogError> {
    use crate::store::session_json::{parse_session_json, write_session_json};
    let io = |message: String| BikeLogError { kind: BikeLogErrorKind::Io, message };
    check_id(bike_id)?;
    let bike: Bike = std::fs::read(bike_path(data_root, bike_id))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| BikeLogError { kind: BikeLogErrorKind::InvalidId, message: format!("no bike {bike_id:?}") })?;

    let mut after_ms: Option<i64> = None;
    for dir in std::fs::read_dir(data_root.join("sessions")).into_iter().flatten().flatten() {
        if dir.file_name().to_str() == Some(session_id) {
            continue;
        }
        let Some(doc) = std::fs::read(dir.path().join("session.json")).ok().and_then(|b| parse_session_json(&b).ok()) else { continue };
        if doc.bike_id != bike_id {
            continue;
        }
        if let Some(prev) = doc.bike_log.filter(|l| l.until_ms <= start_ms) {
            after_ms = Some(after_ms.map_or(prev.until_ms, |a| a.max(prev.until_ms)));
        }
    }

    let entries = load_log(data_root, bike_id).entries;
    let attached = SessionBikeLog {
        entry_ids: entries_for_session(&entries, after_ms, end_ms),
        after_ms,
        until_ms: end_ms,
        setup_sheet: running_sheet(&entries, start_ms),
    };

    let target = data_root.join("sessions").join(session_id).join("session.json");
    let raw = std::fs::read(&target).map_err(|e| io(format!("reading {}: {e}", target.display())))?;
    let mut doc = parse_session_json(&raw).map_err(|e| io(e.message))?;
    doc.bike_id = bike_id.to_string();
    doc.bike = bike.name.clone();
    doc.bike_log = Some(attached);
    write_session_json(data_root, session_id, &doc, Some(&sha256_hex(&raw))).map_err(|e| io(e.message))?;
    Ok(doc)
}

/// Discriminant for [`BikeLogError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BikeLogErrorKind {
    /// A filesystem operation failed.
    Io,
    /// The record did not encode to JSON.
    Encode,
    /// An id would not make a safe file name.
    InvalidId,
}

/// Error from the bike-log store. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BikeLogError {
    /// What went wrong.
    pub kind: BikeLogErrorKind,
    /// Detail for the person reading it.
    pub message: String,
}

impl fmt::Display for BikeLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for BikeLogError {}

impl From<AtomicWriteError> for BikeLogError {
    fn from(e: AtomicWriteError) -> Self {
        BikeLogError { kind: BikeLogErrorKind::Io, message: e.to_string() }
    }
}

/// An id is used as a file or folder name: ASCII letters, digits, `-` and
/// `_` only, and not empty.
fn check_id(id: &str) -> Result<(), BikeLogError> {
    if !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        Ok(())
    } else {
        Err(BikeLogError { kind: BikeLogErrorKind::InvalidId, message: format!("{id:?} is not a valid id (letters, digits, - and _ only)") })
    }
}

/// Whether `id` is usable as a bike or entry id (a safe file name).
pub fn is_valid_id(id: &str) -> bool {
    check_id(id).is_ok()
}

/// The sync key of a log entry file: `<bike_id>/<kind>-<entry_id>`, its
/// path under `bikelog/` without `.json`.
pub fn log_key(bike_id: &str, kind: EntryKind, entry_id: &str) -> String {
    format!("{bike_id}/{}{entry_id}", kind.file_prefix())
}

/// Splits a sync key from [`log_key`] into `(bike_id, kind, entry_id)`;
/// `None` for anything else, including unsafe ids.
pub fn parse_log_key(key: &str) -> Option<(&str, EntryKind, &str)> {
    let (bike_id, stem) = key.split_once('/')?;
    let (kind, entry_id) = if let Some(id) = stem.strip_prefix(EntryKind::Note.file_prefix()) {
        (EntryKind::Note, id)
    } else {
        (EntryKind::Sheet, stem.strip_prefix(EntryKind::Sheet.file_prefix())?)
    };
    (is_valid_id(bike_id) && is_valid_id(entry_id)).then_some((bike_id, kind, entry_id))
}

/// `<data>/bikes/<bike_id>.json`.
pub fn bike_path(data_root: &Path, bike_id: &str) -> PathBuf {
    data_root.join("bikes").join(format!("{bike_id}.json"))
}

impl EntryKind {
    /// The file-name prefix for this kind: `note-` or `sheet-`.
    pub fn file_prefix(self) -> &'static str {
        match self {
            EntryKind::Note => "note-",
            EntryKind::Sheet => "sheet-",
        }
    }
}

/// `<data>/bikelog/<bike_id>/<kind>-<entry_id>.json`.
pub fn entry_path(data_root: &Path, bike_id: &str, kind: EntryKind, entry_id: &str) -> PathBuf {
    data_root.join("bikelog").join(bike_id).join(format!("{}{entry_id}.json", kind.file_prefix()))
}

/// Writes `bytes` to `target` atomically (C4 §4).
fn write(data_root: &Path, target: &Path, bytes: Vec<u8>) -> Result<(), BikeLogError> {
    let based_on = std::fs::read(target).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(data_root, target, &bytes, based_on.as_deref(), |_current| bytes.clone())?;
    Ok(())
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, BikeLogError> {
    serde_json::to_vec_pretty(value).map_err(|e| BikeLogError { kind: BikeLogErrorKind::Encode, message: e.to_string() })
}

/// Saves `bike`, merged with the copy already on disk (so a stale writer
/// can't undo a newer rename).
pub fn save_bike(data_root: &Path, bike: &Bike) -> Result<Bike, BikeLogError> {
    check_id(&bike.bike_id)?;
    let target = bike_path(data_root, &bike.bike_id);
    let merged = match std::fs::read(&target).ok().and_then(|b| serde_json::from_slice::<Bike>(&b).ok()) {
        Some(on_disk) => on_disk.merge(bike),
        None => bike.clone(),
    };
    write(data_root, &target, encode(&merged)?)?;
    Ok(merged)
}

/// Saves `entry`, merged with the copy already on disk, so a version
/// written elsewhere in the meantime is kept. Returns what was written.
pub fn save_entry(data_root: &Path, entry: &Entry) -> Result<Entry, BikeLogError> {
    check_id(&entry.bike_id)?;
    check_id(&entry.entry_id)?;
    let target = entry_path(data_root, &entry.bike_id, entry.kind, &entry.entry_id);
    let merged = match std::fs::read(&target).ok().and_then(|b| serde_json::from_slice::<Entry>(&b).ok()) {
        Some(on_disk) => on_disk.merge(entry),
        None => entry.merge(&Entry { versions: Vec::new(), ..entry.clone() }),
    };
    write(data_root, &target, encode(&merged)?)?;
    Ok(merged)
}

/// Every bike, plus the files that didn't parse.
#[derive(Debug, Clone, PartialEq)]
pub struct BikesLoad {
    /// Parsed bikes, sorted by name.
    pub bikes: Vec<Bike>,
    /// `(path, reason)` for each file that didn't parse.
    pub skipped: Vec<(PathBuf, String)>,
}

/// One bike's log, plus the files that didn't parse.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLoad {
    /// Parsed entries, deleted ones included, sorted by entry id.
    pub entries: Vec<Entry>,
    /// `(path, reason)` for each file that didn't parse.
    pub skipped: Vec<(PathBuf, String)>,
}

/// Reads every `<prefix>*.json` in `dir` as `T`; a bad file is skipped,
/// never fatal.
fn load_dir<T: for<'de> Deserialize<'de>>(dir: &Path, prefix: &str) -> (Vec<T>, Vec<(PathBuf, String)>) {
    let Ok(read) = std::fs::read_dir(dir) else { return (Vec::new(), Vec::new()) };
    let mut ok = Vec::new();
    let mut skipped = Vec::new();
    for item in read.flatten() {
        let path = item.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with(prefix) || path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<T>(&b).map_err(|e| e.to_string())) {
            Ok(v) => ok.push(v),
            Err(e) => skipped.push((path, e)),
        }
    }
    (ok, skipped)
}

/// Loads every bike.
pub fn load_bikes(data_root: &Path) -> BikesLoad {
    let (mut bikes, skipped) = load_dir::<Bike>(&data_root.join("bikes"), "");
    bikes.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.bike_id.cmp(&b.bike_id)));
    BikesLoad { bikes, skipped }
}

/// Loads `bike_id`'s log, notes and sheets. An unknown bike has an empty log.
pub fn load_log(data_root: &Path, bike_id: &str) -> LogLoad {
    load_log_files(data_root, bike_id, "")
}

/// Loads only `bike_id`'s confirmed sheets (`sheet-*.json`), without
/// opening a single note.
pub fn load_sheets(data_root: &Path, bike_id: &str) -> LogLoad {
    load_log_files(data_root, bike_id, EntryKind::Sheet.file_prefix())
}

fn load_log_files(data_root: &Path, bike_id: &str, prefix: &str) -> LogLoad {
    if check_id(bike_id).is_err() {
        return LogLoad { entries: Vec::new(), skipped: Vec::new() };
    }
    let (mut entries, skipped) = load_dir::<Entry>(&data_root.join("bikelog").join(bike_id), prefix);
    entries.sort_by(|a, b| a.entry_id.cmp(&b.entry_id));
    LogLoad { entries, skipped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-bikelog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn version(id: &str, parents: &[&str], at_ms: i64, updated_at_ms: i64, text: &str) -> EntryVersion {
        EntryVersion {
            version_id: id.to_string(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            at_ms,
            updated_at_ms,
            device: "phone".to_string(),
            text: text.to_string(),
            changes: Vec::new(),
            sheet: BTreeMap::new(),
            deleted: false,
        }
    }

    fn note(id: &str, at_ms: i64, changes: &[(&str, serde_json::Value)]) -> Entry {
        let mut v = version(&format!("{id}-v1"), &[], at_ms, at_ms, id);
        v.changes = changes.iter().map(|(f, val)| FieldChange { field: f.to_string(), value: val.clone() }).collect();
        Entry { entry_id: id.to_string(), bike_id: "gt".to_string(), kind: EntryKind::Note, versions: vec![v] }
    }

    fn sheet(id: &str, at_ms: i64, values: &[(&str, serde_json::Value)]) -> Entry {
        let mut v = version(&format!("{id}-v1"), &[], at_ms, at_ms, "");
        v.sheet = values.iter().map(|(f, val)| (f.to_string(), val.clone())).collect();
        Entry { entry_id: id.to_string(), bike_id: "gt".to_string(), kind: EntryKind::Sheet, versions: vec![v] }
    }

    #[test]
    fn entry_merge_edits_on_two_devices_keep_both_and_show_a_conflict() {
        // Arrange — both devices edit v1 before syncing.
        let base = version("v1", &[], 10, 10, "fork service");
        let phone = Entry { entry_id: "e".into(), bike_id: "gt".into(), kind: EntryKind::Note, versions: vec![base.clone(), version("v2p", &["v1"], 10, 20, "fork service, new seals")] };
        let desk = Entry { versions: vec![base, version("v2d", &["v1"], 10, 30, "fork service at shop")], ..phone.clone() };

        // Act
        let merged = phone.merge(&desk);

        // Assert
        assert_eq!(merged.versions.len(), 3);
        assert!(merged.in_conflict());
        assert_eq!(merged.current().unwrap().text, "fork service at shop");
        assert_eq!(merged, desk.merge(&phone));
    }

    #[test]
    fn entry_merge_a_version_resolving_the_conflict_clears_it() {
        // Arrange
        let e = Entry {
            entry_id: "e".into(),
            bike_id: "gt".into(),
            kind: EntryKind::Note,
            versions: vec![version("v1", &[], 10, 10, "a"), version("v2p", &["v1"], 10, 20, "b"), version("v2d", &["v1"], 10, 30, "c")],
        };
        let fix = Entry { versions: vec![version("v3", &["v2p", "v2d"], 10, 40, "b and c")], ..e.clone() };

        // Act
        let merged = e.merge(&fix);

        // Assert
        assert!(!merged.in_conflict());
        assert_eq!(merged.current().unwrap().text, "b and c");
    }

    #[test]
    fn entry_delete_is_a_version_and_survives_a_merge_with_an_older_copy() {
        // Arrange
        let old = note("e", 10, &[]);
        let mut del = version("v2", &["e-v1"], 10, 20, "");
        del.deleted = true;
        let deleted = Entry { versions: vec![old.versions[0].clone(), del], ..old.clone() };

        // Act
        let merged = old.merge(&deleted);

        // Assert
        assert!(merged.is_deleted());
    }

    #[test]
    fn running_sheet_baseline_plus_later_note_changes_in_time_order() {
        // Arrange
        let entries = vec![
            sheet("s1", 100, &[("fork.lsc", json!(10)), ("fork.psi", json!(70))]),
            note("n1", 50, &[("fork.lsc", json!(3))]),
            note("n3", 300, &[("fork.lsc", json!(12))]),
            note("n2", 200, &[("fork.lsc", json!(11)), ("rear.psi", json!(28))]),
            note("n4", 250, &[]),
            note("n5", 900, &[("fork.lsc", json!(20))]),
        ];

        // Act
        let s = running_sheet(&entries, 400);

        // Assert
        assert_eq!(s.baseline.as_deref(), Some("s1"));
        assert_eq!(s.values["fork.lsc"], json!(12));
        assert_eq!(s.values["fork.psi"], json!(70));
        assert_eq!(s.values["rear.psi"], json!(28));
        assert_eq!(s.since, vec!["n2", "n4", "n3"]);
    }

    #[test]
    fn running_sheet_a_newer_sheet_replaces_the_baseline_and_deleted_entries_count_for_nothing() {
        // Arrange
        let mut gone = sheet("s2", 200, &[("fork.lsc", json!(99))]);
        let mut del = version("s2-v2", &["s2-v1"], 200, 210, "");
        del.deleted = true;
        gone.versions.push(del);
        let entries = vec![sheet("s1", 100, &[("fork.lsc", json!(10))]), gone, sheet("s3", 150, &[("fork.lsc", json!(8))])];

        // Act
        let s = running_sheet(&entries, 400);

        // Assert
        assert_eq!(s.baseline.as_deref(), Some("s3"));
        assert_eq!(s.values["fork.lsc"], json!(8));
    }

    #[test]
    fn entries_for_session_takes_live_entries_since_the_previous_session() {
        // Arrange
        let mut gone = note("gone", 150, &[]);
        let mut del = version("gone-v2", &["gone-v1"], 150, 160, "");
        del.deleted = true;
        gone.versions.push(del);
        let entries = vec![note("before", 100, &[]), note("b", 300, &[]), note("a", 200, &[]), gone, note("after", 600, &[]), note("edge", 500, &[])];

        // Act
        let picked = entries_for_session(&entries, Some(100), 500);
        let first = entries_for_session(&entries, None, 150);

        // Assert
        assert_eq!(picked, vec!["a", "b", "edge"]);
        assert_eq!(first, vec!["before"]);
    }

    #[test]
    fn bike_merge_takes_the_newer_of_each_part() {
        // Arrange
        let a = Bike { bike_id: "gt".into(), name: "GT".into(), name_updated_at_ms: 20, fields: vec!["fork.lsc".into()], fields_updated_at_ms: 5 };
        let b = Bike { bike_id: "gt".into(), name: "Old".into(), name_updated_at_ms: 10, fields: vec!["fork.lsc".into(), "fork.hsc".into()], fields_updated_at_ms: 30 };

        // Act
        let m = a.merge(&b);

        // Assert
        assert_eq!(m.name, "GT");
        assert_eq!(m.fields, vec!["fork.lsc", "fork.hsc"]);
    }

    #[test]
    fn save_entry_merges_with_the_copy_on_disk_and_load_log_reads_it_back() {
        // Arrange
        let root = temp_root();
        let first = note("e1", 10, &[]);
        let edit = Entry { versions: vec![version("e1-v2", &["e1-v1"], 10, 20, "edited")], ..first.clone() };
        std::fs::create_dir_all(root.join("bikelog/gt")).unwrap();
        std::fs::write(root.join("bikelog/gt/bad.json"), b"nope").unwrap();

        // Act
        save_entry(&root, &first).unwrap();
        save_entry(&root, &edit).unwrap();
        let log = load_log(&root, "gt");

        // Assert
        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].versions.len(), 2);
        assert_eq!(log.entries[0].current().unwrap().text, "edited");
        assert_eq!(log.skipped.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_bike_and_entry_refuse_ids_that_are_not_safe_file_names() {
        // Arrange
        let root = temp_root();
        let bike = Bike { bike_id: "../x".into(), name: String::new(), name_updated_at_ms: 0, fields: Vec::new(), fields_updated_at_ms: 0 };
        let entry = Entry { entry_id: "a/b".into(), ..note("e", 1, &[]) };

        // Act
        let b = save_bike(&root, &bike).unwrap_err();
        let e = save_entry(&root, &entry).unwrap_err();

        // Assert
        assert_eq!(b.kind, BikeLogErrorKind::InvalidId);
        assert_eq!(e.kind, BikeLogErrorKind::InvalidId);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_entry_names_the_file_by_kind_and_load_sheets_reads_only_sheets() {
        // Arrange
        let root = temp_root();
        let n = note("01A", 10, &[]);
        let s = sheet("01B", 20, &[("fork.lsc", json!(10))]);

        // Act
        save_entry(&root, &n).unwrap();
        save_entry(&root, &s).unwrap();
        let sheets = load_sheets(&root, "gt");
        let all = load_log(&root, "gt");

        // Assert
        assert!(root.join("bikelog/gt/note-01A.json").is_file());
        assert!(root.join("bikelog/gt/sheet-01B.json").is_file());
        assert_eq!(sheets.entries.iter().map(|e| e.entry_id.as_str()).collect::<Vec<_>>(), vec!["01B"]);
        assert_eq!(all.entries.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_log_key_round_trips_log_key_and_refuses_anything_else() {
        // Arrange
        let key = log_key("gt-force", EntryKind::Sheet, "01ABC");

        // Act
        let parsed = parse_log_key(&key);

        // Assert
        assert_eq!(key, "gt-force/sheet-01ABC");
        assert_eq!(parsed, Some(("gt-force", EntryKind::Sheet, "01ABC")));
        assert_eq!(parse_log_key("gt/note-01"), Some(("gt", EntryKind::Note, "01")));
        for bad in ["gt", "gt/other-01", "../x/note-01", "gt/note-", "gt/note-a/b", "gt/note-a.b"] {
            assert_eq!(parse_log_key(bad), None, "{bad}");
        }
    }

    #[test]
    fn bike_for_name_finds_by_name_ignoring_case_and_creates_a_new_one_with_a_free_slug() {
        // Arrange
        let root = temp_root();
        save_bike(&root, &Bike { bike_id: "gt-force".into(), name: "GT Force".into(), name_updated_at_ms: 1, fields: Vec::new(), fields_updated_at_ms: 0 }).unwrap();

        // Act
        let found = bike_for_name(&root, "  gt force ", 5).unwrap();
        let other = bike_for_name(&root, "GT-Force!", 5).unwrap();
        let blank = bike_for_name(&root, "  ", 5).unwrap();

        // Assert
        assert_eq!(found.bike_id, "gt-force");
        assert_eq!(other.bike_id, "gt-force-2");
        assert_eq!(other.name, "GT-Force!");
        assert_eq!(blank.bike_id, "bike");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn attach_session_takes_entries_since_the_previous_session_and_the_sheet_at_start() {
        // Arrange
        use crate::store::session_json::{empty_session_json, read_session_json, write_session_json};
        let root = temp_root();
        save_bike(&root, &Bike { bike_id: "gt".into(), name: "GT".into(), name_updated_at_ms: 1, fields: Vec::new(), fields_updated_at_ms: 0 }).unwrap();
        for e in [
            sheet("s1", 50, &[("fork.lsc", json!(10))]),
            note("n1", 150, &[("fork.lsc", json!(9))]),
            note("n2", 350, &[]),
            note("during", 450, &[("fork.lsc", json!(8))]),
            note("later", 900, &[]),
        ] {
            save_entry(&root, &e).unwrap();
        }
        for id in ["ride1", "ride2"] {
            std::fs::create_dir_all(root.join("sessions").join(id)).unwrap();
            write_session_json(&root, id, &empty_session_json(id), None).unwrap();
        }

        // Act
        attach_session(&root, "ride1", "gt", 100, 200).unwrap();
        let doc = attach_session(&root, "ride2", "gt", 400, 500).unwrap();

        // Assert
        let log = doc.bike_log.unwrap();
        assert_eq!(doc.bike, "GT");
        assert_eq!(log.after_ms, Some(200));
        assert_eq!(log.entry_ids, vec!["n2", "during"]);
        assert_eq!(log.setup_sheet.values["fork.lsc"], json!(9));
        let ride1 = read_session_json(&root.join("sessions/ride1/session.json")).unwrap().bike_log.unwrap();
        assert_eq!(ride1.entry_ids, vec!["s1", "n1"]);
        assert_eq!(ride1.after_ms, None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_bike_then_load_bikes_round_trips() {
        // Arrange
        let root = temp_root();
        let bike = Bike { bike_id: "gt-force".into(), name: "GT Force".into(), name_updated_at_ms: 1, fields: vec!["fork.lsc".into()], fields_updated_at_ms: 1 };

        // Act
        save_bike(&root, &bike).unwrap();
        let load = load_bikes(&root);

        // Assert
        assert_eq!(load.bikes, vec![bike]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
