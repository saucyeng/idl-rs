//! The bike log and setup sheet (spec `2026-10-16-idl1-bike-log-DRAFT.md`).
//!
//! One file per bike, `<data>/bikes/<bike_id>.json`, holds the bike (its
//! name and sheet fields) and its whole log, oldest entry first: the one
//! readable history of that bike (Isaac, 2026-10-16). An entry is a
//! **note** (text, plus only the values it changed) or a confirmed
//! **sheet** (every value). Each entry keeps every version of itself: the
//! current one sits at the top of the entry, older wording in its
//! `history`, so two devices' copies merge as a union of entries and, within
//! an entry, of versions; no edit is lost (§4). The full setup at any moment
//! is never stored: [`running_sheet`] works it out from the log.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry, AtomicWriteError};

/// One bike (spec §4): its name and which setup-sheet fields it has. The
/// two parts merge separately, each by its own `*_updated_at_ms`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bike {
    /// Stable id, also the file name.
    pub bike_id: String,
    /// Display name, e.g. "GT Force".
    #[serde(default)]
    pub name: String,
    /// When `name` last changed, UTC ms.
    #[serde(default)]
    pub name_updated_at_ms: i64,
    /// The sheet fields this bike has (dotted keys, e.g. `"fork.lsc"`), in
    /// display order. Empty means every field.
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

/// One version of an entry. Versions are never edited or removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryVersion {
    /// Unique id of this version.
    pub version_id: String,
    /// The version(s) this one was edited from: none for the first, two for
    /// a merge of a conflict.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    /// A note's changes: only the values it set, by dotted key (clicks from
    /// fully closed, psi, mm, text…).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub changes: BTreeMap<String, serde_json::Value>,
    /// A sheet's values, by dotted key. Empty for a note.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sheet: BTreeMap<String, serde_json::Value>,
    /// This version deletes the entry.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deleted: bool,
}

/// One log entry: its identity and every version of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Unique id; sorts by creation time.
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

/// One bike's file: the bike and its log.
#[derive(Debug, Clone, PartialEq)]
pub struct BikeFile {
    /// The bike's name and fields.
    pub bike: Bike,
    /// Every entry, deleted ones included, oldest first (by current `at_ms`,
    /// then id).
    pub entries: Vec<Entry>,
}

/// An entry as written in the file: the current version's fields at the
/// top, older versions in `history`.
#[derive(Serialize, Deserialize)]
struct StoredEntry {
    entry_id: String,
    kind: EntryKind,
    #[serde(flatten)]
    current: EntryVersion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    history: Vec<EntryVersion>,
}

/// The file's shape: the bike's fields, then `entries`.
#[derive(Serialize, Deserialize)]
struct StoredBikeFile {
    #[serde(flatten)]
    bike: Bike,
    #[serde(default)]
    entries: Vec<StoredEntry>,
}

impl BikeFile {
    /// A new bike with an empty log.
    pub fn new(bike: Bike) -> BikeFile {
        BikeFile { bike, entries: Vec::new() }
    }

    /// Parses a bike file.
    pub fn from_slice(bytes: &[u8]) -> Result<BikeFile, BikeLogError> {
        let stored: StoredBikeFile =
            serde_json::from_slice(bytes).map_err(|e| BikeLogError { kind: BikeLogErrorKind::Parse, message: e.to_string() })?;
        let bike_id = stored.bike.bike_id.clone();
        let mut file = BikeFile { bike: stored.bike, entries: Vec::new() };
        for s in stored.entries {
            let mut versions = s.history;
            versions.push(s.current);
            let entry = Entry { entry_id: s.entry_id, bike_id: bike_id.clone(), kind: s.kind, versions };
            file.upsert(&entry);
        }
        Ok(file)
    }

    /// The file's bytes: pretty JSON, entries oldest first, each with its
    /// current version at the top and older ones in `history`.
    pub fn to_vec(&self) -> Result<Vec<u8>, BikeLogError> {
        let entries = self
            .entries
            .iter()
            .filter_map(|e| {
                let current = e.current()?.clone();
                let history = e.versions.iter().filter(|v| v.version_id != current.version_id).cloned().collect();
                Some(StoredEntry { entry_id: e.entry_id.clone(), kind: e.kind, current, history })
            })
            .collect();
        serde_json::to_vec_pretty(&StoredBikeFile { bike: self.bike.clone(), entries })
            .map_err(|e| BikeLogError { kind: BikeLogErrorKind::Encode, message: e.to_string() })
    }

    /// Adds `entry`, or merges it into the entry with the same id, keeping
    /// the log in order.
    pub fn upsert(&mut self, entry: &Entry) {
        let entry = Entry { bike_id: self.bike.bike_id.clone(), ..entry.clone() };
        match self.entries.iter_mut().find(|e| e.entry_id == entry.entry_id) {
            Some(e) => *e = e.merge(&entry),
            None => self.entries.push(entry.merge(&Entry { versions: Vec::new(), ..entry.clone() })),
        }
        self.entries.sort_by_key(|e| (e.current().map_or(i64::MIN, |v| v.at_ms), e.entry_id.clone()));
    }

    /// Merges two copies of the same bike's file: the bike part by part,
    /// the log as a union of entries and versions. Symmetric, so two devices
    /// converge once each has merged the other's copy.
    pub fn merge(&self, other: &BikeFile) -> BikeFile {
        let mut out = BikeFile { bike: self.bike.merge(&other.bike), entries: self.entries.clone() };
        for e in &other.entries {
            out.upsert(e);
        }
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
        for (field, value) in &v.changes {
            out.values.insert(field.clone(), value.clone());
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

/// A new id that sorts by creation time: `now_ms` in base 36, zero-padded
/// to nine characters, then six random lowercase letters and digits. Only
/// letters and digits, so it is a safe id ([`is_valid_id`]); the app makes
/// its ids the same way.
pub fn new_id(now_ms: i64) -> String {
    const ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut n = now_ms.max(0) as u64;
    let mut head = Vec::new();
    while n > 0 {
        head.push(ALPHABET[(n % 36) as usize]);
        n /= 36;
    }
    while head.len() < 9 {
        head.push(b'0');
    }
    head.reverse();
    let random = uuid::Uuid::new_v4();
    let tail = random.as_bytes()[..6].iter().map(|b| ALPHABET[(*b as usize) % 36]);
    String::from_utf8(head.into_iter().chain(tail).collect()).expect("ascii")
}

/// A new note for `bike_id`: its text, the values it sets, when it
/// happened and who wrote it. Not saved: hand it to [`save_entry`].
pub fn new_note(bike_id: &str, text: &str, changes: BTreeMap<String, serde_json::Value>, at_ms: i64, device: &str, now_ms: i64) -> Entry {
    Entry {
        entry_id: new_id(now_ms),
        bike_id: bike_id.to_string(),
        kind: EntryKind::Note,
        versions: vec![EntryVersion {
            version_id: new_id(now_ms),
            parents: Vec::new(),
            at_ms,
            updated_at_ms: now_ms,
            device: device.to_string(),
            text: text.trim().to_string(),
            changes,
            sheet: BTreeMap::new(),
            deleted: false,
        }],
    }
}

/// Parses one `key=value` setup change: the value is a number when it
/// reads as one, else text. `None` without an `=` or with an empty key.
pub fn parse_change(spec: &str) -> Option<(String, serde_json::Value)> {
    let (key, value) = spec.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    let value = value.trim();
    let parsed = value.parse::<f64>().ok().and_then(serde_json::Number::from_f64).map(|n| {
        if n.as_f64().is_some_and(|f| f.fract() == 0.0 && f.abs() < 9e15) {
            serde_json::Value::from(n.as_f64().unwrap() as i64)
        } else {
            serde_json::Value::Number(n)
        }
    });
    Some((key.to_string(), parsed.unwrap_or_else(|| serde_json::Value::String(value.to_string()))))
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
    let file = read_bike_file(data_root, bike_id)?;

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

    let attached = SessionBikeLog {
        entry_ids: entries_for_session(&file.entries, after_ms, end_ms),
        after_ms,
        until_ms: end_ms,
        setup_sheet: running_sheet(&file.entries, start_ms),
    };

    let target = data_root.join("sessions").join(session_id).join("session.json");
    let raw = std::fs::read(&target).map_err(|e| io(format!("reading {}: {e}", target.display())))?;
    let mut doc = parse_session_json(&raw).map_err(|e| io(e.message))?;
    doc.bike_id = bike_id.to_string();
    doc.bike = file.bike.name.clone();
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
    /// A bike file did not parse.
    Parse,
    /// An id would not make a safe file name, or names no bike.
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

/// An id is used as a file name: ASCII letters, digits, `-` and `_` only,
/// and not empty.
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

/// `<data>/bikes/<bike_id>.json`.
pub fn bike_path(data_root: &Path, bike_id: &str) -> PathBuf {
    data_root.join("bikes").join(format!("{bike_id}.json"))
}

/// Reads `bike_id`'s file. A missing bike is `InvalidId`.
pub fn read_bike_file(data_root: &Path, bike_id: &str) -> Result<BikeFile, BikeLogError> {
    check_id(bike_id)?;
    let path = bike_path(data_root, bike_id);
    let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => BikeLogError { kind: BikeLogErrorKind::InvalidId, message: format!("no bike {bike_id:?}") },
        _ => BikeLogError { kind: BikeLogErrorKind::Io, message: format!("reading {}: {e}", path.display()) },
    })?;
    BikeFile::from_slice(&bytes)
}

/// Writes `file` merged with the copy on disk, if any, so nothing written
/// elsewhere in the meantime is lost. Returns what was written.
pub fn save_bike_file(data_root: &Path, file: &BikeFile) -> Result<BikeFile, BikeLogError> {
    check_id(&file.bike.bike_id)?;
    for e in &file.entries {
        check_id(&e.entry_id)?;
    }
    let target = bike_path(data_root, &file.bike.bike_id);
    let on_disk = std::fs::read(&target).ok();
    let merged = match on_disk.as_deref().map(BikeFile::from_slice) {
        Some(Ok(existing)) => existing.merge(file),
        _ => file.merge(&BikeFile::new(file.bike.clone())),
    };
    let bytes = merged.to_vec()?;
    let based_on = on_disk.as_deref().map(sha256_hex);
    write_atomic_with_retry(data_root, &target, &bytes, based_on.as_deref(), |_current| bytes.clone())?;
    Ok(merged)
}

/// Saves `bike`'s name and fields (merged part by part with the copy on
/// disk, so a stale writer can't undo a newer rename), keeping its log.
pub fn save_bike(data_root: &Path, bike: &Bike) -> Result<Bike, BikeLogError> {
    Ok(save_bike_file(data_root, &BikeFile::new(bike.clone()))?.bike)
}

/// Saves `entry` (new, or new versions of one) into its bike's log, merged
/// with what is there. The bike must exist. Returns the merged entry.
pub fn save_entry(data_root: &Path, entry: &Entry) -> Result<Entry, BikeLogError> {
    let mut file = read_bike_file(data_root, &entry.bike_id)?;
    file.entries.clear();
    file.upsert(entry);
    let merged = save_bike_file(data_root, &file)?;
    Ok(merged.entries.into_iter().find(|e| e.entry_id == entry.entry_id).expect("just saved"))
}

/// Every bike, plus the files that didn't parse.
#[derive(Debug, Clone, PartialEq)]
pub struct BikesLoad {
    /// Parsed bikes, sorted by name.
    pub bikes: Vec<Bike>,
    /// `(path, reason)` for each file that didn't parse.
    pub skipped: Vec<(PathBuf, String)>,
}

/// One bike's log.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLoad {
    /// Every entry, deleted ones included, oldest first.
    pub entries: Vec<Entry>,
    /// `(path, reason)` when the bike's file didn't parse.
    pub skipped: Vec<(PathBuf, String)>,
}

/// Loads every bike's name and fields.
pub fn load_bikes(data_root: &Path) -> BikesLoad {
    let mut bikes = Vec::new();
    let mut skipped = Vec::new();
    for item in std::fs::read_dir(data_root.join("bikes")).into_iter().flatten().flatten() {
        let path = item.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| BikeFile::from_slice(&b).map_err(|e| e.message)) {
            Ok(f) => bikes.push(f.bike),
            Err(e) => skipped.push((path, e)),
        }
    }
    bikes.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.bike_id.cmp(&b.bike_id)));
    BikesLoad { bikes, skipped }
}

/// Loads `bike_id`'s log. An unknown bike has an empty log; a file that
/// doesn't parse is reported in `skipped`.
pub fn load_log(data_root: &Path, bike_id: &str) -> LogLoad {
    match read_bike_file(data_root, bike_id) {
        Ok(f) => LogLoad { entries: f.entries, skipped: Vec::new() },
        Err(e) if e.kind == BikeLogErrorKind::Parse => LogLoad { entries: Vec::new(), skipped: vec![(bike_path(data_root, bike_id), e.message)] },
        Err(_) => LogLoad { entries: Vec::new(), skipped: Vec::new() },
    }
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

    fn gt() -> Bike {
        Bike { bike_id: "gt".into(), name: "GT".into(), name_updated_at_ms: 1, fields: Vec::new(), fields_updated_at_ms: 0 }
    }

    fn version(id: &str, parents: &[&str], at_ms: i64, updated_at_ms: i64, text: &str) -> EntryVersion {
        EntryVersion {
            version_id: id.to_string(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            at_ms,
            updated_at_ms,
            device: "phone".to_string(),
            text: text.to_string(),
            changes: BTreeMap::new(),
            sheet: BTreeMap::new(),
            deleted: false,
        }
    }

    fn note(id: &str, at_ms: i64, changes: &[(&str, serde_json::Value)]) -> Entry {
        let mut v = version(&format!("{id}-v1"), &[], at_ms, at_ms, id);
        v.changes = changes.iter().map(|(f, val)| (f.to_string(), val.clone())).collect();
        Entry { entry_id: id.to_string(), bike_id: "gt".to_string(), kind: EntryKind::Note, versions: vec![v] }
    }

    fn sheet(id: &str, at_ms: i64, values: &[(&str, serde_json::Value)]) -> Entry {
        let mut v = version(&format!("{id}-v1"), &[], at_ms, at_ms, "");
        v.sheet = values.iter().map(|(f, val)| (f.to_string(), val.clone())).collect();
        Entry { entry_id: id.to_string(), bike_id: "gt".to_string(), kind: EntryKind::Sheet, versions: vec![v] }
    }

    fn deleted(mut e: Entry, at: i64) -> Entry {
        let parent = e.versions[0].version_id.clone();
        let mut d = version(&format!("{parent}-del"), &[&parent], e.versions[0].at_ms, at, "");
        d.deleted = true;
        e.versions.push(d);
        e
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
        let gone = deleted(old.clone(), 20);

        // Act
        let merged = old.merge(&gone);

        // Assert
        assert!(merged.is_deleted());
    }

    #[test]
    fn bike_file_round_trips_with_current_on_top_and_older_versions_in_history() {
        // Arrange
        let mut file = BikeFile::new(gt());
        let mut edited = note("n1", 20, &[("rear_tyre.psi", json!(26))]);
        edited.versions.push(version("n1-v2", &["n1-v1"], 20, 30, "rear 26 psi"));
        file.upsert(&edited);
        file.upsert(&sheet("s1", 10, &[("fork.lsc", json!(10))]));

        // Act
        let bytes = file.to_vec().unwrap();
        let back = BikeFile::from_slice(&bytes).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Assert
        assert_eq!(back, file);
        assert_eq!(json["bike_id"], "gt");
        assert_eq!(json["entries"][0]["entry_id"], "s1");
        assert_eq!(json["entries"][1]["text"], "rear 26 psi");
        assert_eq!(json["entries"][1]["history"][0]["text"], "n1");
        assert!(json["entries"][0].get("history").is_none());
    }

    #[test]
    fn bike_file_merge_is_a_union_of_entries_and_versions_from_both_sides() {
        // Arrange
        let mut phone = BikeFile::new(gt());
        phone.upsert(&note("a", 10, &[]));
        phone.upsert(&note("shared", 20, &[]));
        let mut desk = BikeFile::new(Bike { name: "GT Force".into(), name_updated_at_ms: 5, ..gt() });
        desk.upsert(&note("b", 15, &[]));
        let mut shared = note("shared", 20, &[]);
        shared.versions.push(version("shared-v2", &["shared-v1"], 20, 40, "edited on desk"));
        desk.upsert(&shared);

        // Act
        let merged = phone.merge(&desk);

        // Assert
        assert_eq!(merged.entries.iter().map(|e| e.entry_id.as_str()).collect::<Vec<_>>(), vec!["a", "b", "shared"]);
        assert_eq!(merged.entries[2].current().unwrap().text, "edited on desk");
        assert_eq!(merged.bike.name, "GT Force");
        assert_eq!(merged, desk.merge(&phone));
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
        let entries = vec![sheet("s1", 100, &[("fork.lsc", json!(10))]), deleted(sheet("s2", 200, &[("fork.lsc", json!(99))]), 210), sheet("s3", 150, &[("fork.lsc", json!(8))])];

        // Act
        let s = running_sheet(&entries, 400);

        // Assert
        assert_eq!(s.baseline.as_deref(), Some("s3"));
        assert_eq!(s.values["fork.lsc"], json!(8));
    }

    #[test]
    fn entries_for_session_takes_live_entries_since_the_previous_session() {
        // Arrange
        let entries = vec![note("before", 100, &[]), note("b", 300, &[]), note("a", 200, &[]), deleted(note("gone", 150, &[]), 160), note("after", 600, &[]), note("edge", 500, &[])];

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
    fn save_entry_adds_to_the_bikes_one_file_and_merges_new_versions() {
        // Arrange
        let root = temp_root();
        save_bike(&root, &gt()).unwrap();
        let first = note("e1", 10, &[]);
        let edit = Entry { versions: vec![version("e1-v2", &["e1-v1"], 10, 20, "edited")], ..first.clone() };

        // Act
        save_entry(&root, &first).unwrap();
        let merged = save_entry(&root, &edit).unwrap();
        save_bike(&root, &Bike { name: "GT Force".into(), name_updated_at_ms: 50, ..gt() }).unwrap();
        let log = load_log(&root, "gt");

        // Assert
        assert_eq!(merged.versions.len(), 2);
        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].current().unwrap().text, "edited");
        assert_eq!(load_bikes(&root).bikes[0].name, "GT Force");
        assert_eq!(std::fs::read_dir(root.join("bikes")).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn save_entry_for_an_unknown_bike_or_with_an_unsafe_id_is_invalid_id() {
        // Arrange
        let root = temp_root();
        save_bike(&root, &gt()).unwrap();
        let unknown = Entry { bike_id: "nope".into(), ..note("e", 1, &[]) };
        let unsafe_id = Entry { entry_id: "a/b".into(), ..note("e", 1, &[]) };
        let bad_bike = Bike { bike_id: "../x".into(), ..gt() };

        // Act
        let errors = [save_entry(&root, &unknown).unwrap_err().kind, save_entry(&root, &unsafe_id).unwrap_err().kind, save_bike(&root, &bad_bike).unwrap_err().kind];

        // Assert
        assert_eq!(errors, [BikeLogErrorKind::InvalidId; 3]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_log_reports_a_bike_file_that_does_not_parse() {
        // Arrange
        let root = temp_root();
        std::fs::create_dir_all(root.join("bikes")).unwrap();
        std::fs::write(root.join("bikes/gt.json"), b"nope").unwrap();

        // Act
        let log = load_log(&root, "gt");
        let bikes = load_bikes(&root);

        // Assert
        assert!(log.entries.is_empty());
        assert_eq!(log.skipped.len(), 1);
        assert_eq!(bikes.skipped.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn new_id_sorts_by_time_and_is_a_valid_id() {
        // Arrange
        let early = new_id(1_000);
        let late = new_id(1_760_000_000_000);

        // Act
        let mut sorted = vec![late.clone(), early.clone()];
        sorted.sort();

        // Assert
        assert_eq!(sorted, vec![early.clone(), late.clone()]);
        assert_eq!(late.len(), 15);
        assert!(is_valid_id(&late) && is_valid_id(&early));
    }

    #[test]
    fn parse_change_numbers_text_and_bad_specs() {
        // Arrange
        let specs = ["fork.lsc=9", "rear_tyre.psi = 26.5", "frame.flip_chip=Long", "nokey", "=5"];

        // Act
        let parsed: Vec<Option<(String, serde_json::Value)>> = specs.iter().map(|s| parse_change(s)).collect();

        // Assert
        assert_eq!(parsed[0], Some(("fork.lsc".to_string(), json!(9))));
        assert_eq!(parsed[1], Some(("rear_tyre.psi".to_string(), json!(26.5))));
        assert_eq!(parsed[2], Some(("frame.flip_chip".to_string(), json!("Long"))));
        assert_eq!(parsed[3], None);
        assert_eq!(parsed[4], None);
    }

    #[test]
    fn new_note_saved_changes_the_running_sheet() {
        // Arrange
        let root = temp_root();
        save_bike(&root, &gt()).unwrap();
        save_entry(&root, &sheet("s1", 10, &[("fork.lsc", json!(10))])).unwrap();
        let changes = [("fork.lsc".to_string(), json!(9))].into_iter().collect();

        // Act
        let saved = save_entry(&root, &new_note("gt", "+1 LSC fork", changes, 20, "agent", 30)).unwrap();
        let log = load_log(&root, "gt");

        // Assert
        assert_eq!(saved.versions[0].device, "agent");
        assert_eq!(running_sheet(&log.entries, 100).values["fork.lsc"], json!(9));
        let _ = std::fs::remove_dir_all(&root);
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
        save_bike(&root, &gt()).unwrap();
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
}
