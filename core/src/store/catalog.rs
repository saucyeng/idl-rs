//! SQLite catalog (contract C4 §5) — a rebuildable index over `<data>`,
//! never itself the source of truth (design principle: "the catalog is an
//! index"). Six tables: `sessions`, `blobs`, `workbooks`, `tracks`, `laps`,
//! `lap_summary`.

use std::fmt;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rusqlite::Connection;
use uuid::Uuid;

use crate::store::atomic::{sha256_hex, write_atomic_with_retry};
use crate::store::blob::{blob_path, verify_blob};
use crate::store::catalog_read::WorkbookSummary;
use crate::store::session_json::{read_session_json, LapJson, TrackVisitJson};
use crate::track_artifact::read::read_track;
use crate::workbook::v3::front_matter::parse_front_matter;

/// Schema version this build of `idl-rs` writes/expects for `catalog.sqlite`
/// (C4 §5). A mismatch on open means the catalog is deleted and rebuilt, not
/// migrated in place — the catalog holds no data that doesn't also live,
/// canonically, in a file under `<data>`.
pub const CATALOG_SCHEMA_VERSION: i64 = 1;

/// Discriminant for [`CatalogError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogErrorKind {
    /// A filesystem operation failed.
    Io,
    /// A SQLite operation failed (a genuine driver/schema/corruption error —
    /// distinct from [`CatalogErrorKind::NotFound`], which is "the query
    /// ran fine and found nothing").
    Sql,
    /// The requested entity (session, lap, track, …) does not exist. Added
    /// by ruling R46: `idl_rs::store::catalog_read` (L5) used to reuse `Sql`
    /// for this, which meant a corrupt `catalog.sqlite` surfaced as
    /// "not found" (inviting a re-import) rather than "internal" (pointing
    /// at `rebuild_catalog`, the actual fix).
    NotFound,
}

/// Error from the catalog. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogError {
    pub kind: CatalogErrorKind,
    pub message: String,
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for CatalogError {}

impl From<rusqlite::Error> for CatalogError {
    fn from(e: rusqlite::Error) -> Self {
        CatalogError { kind: CatalogErrorKind::Sql, message: e.to_string() }
    }
}

const DDL: &str = r#"
CREATE TABLE blobs (
  sha256        TEXT PRIMARY KEY,
  size_bytes    INTEGER NOT NULL,
  mtime_ms      INTEGER NOT NULL
);

CREATE TABLE tracks (
  track_id      TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  venue_name    TEXT NOT NULL DEFAULT '',
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  full_json     TEXT NOT NULL
);
CREATE INDEX idx_tracks_venue ON tracks(venue_name);

CREATE TABLE sessions (
  session_id        TEXT PRIMARY KEY,
  blob_sha256       TEXT NOT NULL REFERENCES blobs(sha256) ON DELETE RESTRICT,
  source_format     TEXT NOT NULL CHECK (source_format IN ('idl0','fit','gpx','csv')),
  device_id         TEXT,
  config_checksum   TEXT,
  importer_version  TEXT NOT NULL,
  seam_correction_version TEXT NOT NULL,
  engine_version    TEXT NOT NULL,
  timestamp_utc_ms  INTEGER NOT NULL,
  created_at_ms     INTEGER NOT NULL,
  rider             TEXT NOT NULL DEFAULT '',
  bike              TEXT NOT NULL DEFAULT '',
  venue_name        TEXT NOT NULL DEFAULT '',
  event_name        TEXT NOT NULL DEFAULT '',
  event_session     TEXT NOT NULL DEFAULT '',
  short_comment     TEXT NOT NULL DEFAULT '',
  tag               TEXT NOT NULL DEFAULT '',
  lap_count         INTEGER,
  duration_ms       INTEGER
);
CREATE INDEX idx_sessions_timestamp ON sessions(timestamp_utc_ms);
CREATE INDEX idx_sessions_venue     ON sessions(venue_name);
CREATE INDEX idx_sessions_tag       ON sessions(tag);
CREATE INDEX idx_sessions_blob      ON sessions(blob_sha256);

CREATE TABLE workbooks (
  workbook_id   TEXT PRIMARY KEY,
  file_name     TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  size_bytes    INTEGER NOT NULL
);
CREATE INDEX idx_workbooks_name ON workbooks(name);

CREATE TABLE laps (
  session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  lap_number   INTEGER NOT NULL,
  lap_time_ms  INTEGER NOT NULL,
  track_id     TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
  PRIMARY KEY (session_id, lap_number)
);
CREATE INDEX idx_laps_track ON laps(track_id);
CREATE INDEX idx_laps_time  ON laps(lap_time_ms);

CREATE TABLE lap_summary (
  session_id    TEXT NOT NULL,
  lap_number    INTEGER NOT NULL,
  channel_id    TEXT NOT NULL,
  derived_hash  TEXT NOT NULL,
  min_value     REAL NOT NULL,
  max_value     REAL NOT NULL,
  mean_value    REAL NOT NULL,
  PRIMARY KEY (session_id, lap_number, channel_id),
  FOREIGN KEY (session_id, lap_number) REFERENCES laps(session_id, lap_number) ON DELETE CASCADE
);
CREATE INDEX idx_lap_summary_channel ON lap_summary(channel_id);
"#;

/// Opens (or creates) the catalog at `path`, applying C4 §5's PRAGMAs on
/// every connection (they are per-connection, not persisted by SQLite).
/// Does not create the schema — callers that may be opening a brand-new,
/// empty file call [`create_schema`] (internal) via [`rebuild_catalog`].
pub fn open_catalog(path: &Path) -> Result<Connection, CatalogError> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    // A brand-new file (first launch on a fresh data root, ruling R200) has
    // no tables and `user_version` 0: bootstrap the schema here rather than
    // leaving every query to fail with "no such table" until someone runs a
    // rebuild. Only a *completely empty* database is touched; a populated
    // one, whatever its version, is left to the rebuild/migration paths.
    let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if user_version == 0 {
        let tables: i64 =
            conn.query_row("SELECT count(*) FROM sqlite_master WHERE type = 'table'", [], |r| r.get(0))?;
        if tables == 0 {
            create_schema(&conn)?;
        }
    }
    Ok(conn)
}

/// Creates the schema and sets `user_version` on a freshly-opened,
/// empty database.
fn create_schema(conn: &Connection) -> Result<(), CatalogError> {
    // Idempotent: `open_catalog` may already have bootstrapped an empty file
    // (R200), and callers that open-then-create must not fail on it.
    let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if user_version == CATALOG_SCHEMA_VERSION {
        return Ok(());
    }
    conn.execute_batch(DDL)?;
    conn.pragma_update(None, "user_version", CATALOG_SCHEMA_VERSION)?;
    Ok(())
}

/// Outcome of [`rebuild_catalog`] — counts plus non-fatal per-entity
/// problems (a parse failure skips that entity, not the whole scan, CLAUDE.md §5).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RebuildReport {
    pub blobs_indexed: usize,
    pub tracks_indexed: usize,
    pub sessions_indexed: usize,
    pub laps_indexed: usize,
    /// Rows inserted into `lap_summary` (C4 §5 step 5) — one per
    /// `(lap, channel)` pair with at least one finite sample in the lap's
    /// time window.
    pub lap_summary_indexed: usize,
    pub workbooks_indexed: usize,
    pub skipped: Vec<String>, // human-readable "<path>: <reason>" entries
}

/// Rebuilds `<data_root>/catalog.sqlite` from a full tree scan (C4 §5).
/// Never mutates the live catalog in place — builds
/// `tmp/catalog-rebuild-<uuid>.sqlite`, then atomically swaps it over
/// `catalog.sqlite` via [`crate::store::atomic::write_atomic_with_retry`]
/// (the WAL sidecars are checkpointed into the main file before the swap, so
/// no `-wal`/`-shm` files need to move separately; any stale sidecars left
/// beside the *previous* `catalog.sqlite` are removed after a successful
/// swap). The swap overwrites unconditionally — a rebuild supersedes
/// whatever catalog was already there (C4 §5).
///
/// **Precondition:** no connection to the live `catalog.sqlite` may be open
/// while this runs. C4 §5's rebuild is an offline swap, not a live
/// replace-under-a-reader — the caller is responsible for not holding a
/// [`open_catalog`] connection across a call to this function.
pub fn rebuild_catalog(data_root: &Path) -> Result<RebuildReport, CatalogError> {
    let tmp_dir = data_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(io_err)?;
    let staging_path = tmp_dir.join(format!("catalog-rebuild-{}.sqlite", Uuid::new_v4()));

    let conn = open_catalog(&staging_path)?;
    create_schema(&conn)?;
    let mut report = RebuildReport::default();

    // 1. blobs
    let blobs_dir = data_root.join("blobs").join("sha256");
    if blobs_dir.is_dir() {
        for shard in std::fs::read_dir(&blobs_dir).map_err(io_err)?.flatten() {
            if !shard.path().is_dir() {
                continue;
            }
            let prefix = shard.file_name().to_string_lossy().into_owned();
            for entry in std::fs::read_dir(shard.path()).map_err(io_err)?.flatten() {
                let suffix = entry.file_name().to_string_lossy().into_owned();
                let sha256 = format!("{prefix}{suffix}");
                if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                    report.skipped.push(format!("{}: path does not encode a 64-hex sha256", entry.path().display()));
                    continue;
                }
                match verify_blob(data_root, &sha256) {
                    Ok(()) => {
                        let meta = entry.metadata().map_err(io_err)?;
                        let mtime_ms = file_mtime_ms(&meta);
                        conn.execute(
                            "INSERT INTO blobs (sha256, size_bytes, mtime_ms) VALUES (?1, ?2, ?3)",
                            rusqlite::params![sha256, meta.len() as i64, mtime_ms],
                        )?;
                        report.blobs_indexed += 1;
                    }
                    Err(e) => report.skipped.push(format!("{}: {e}", entry.path().display())),
                }
            }
        }
    }

    // 2. tracks
    let tracks_dir = data_root.join("tracks");
    if tracks_dir.is_dir() {
        for entry in std::fs::read_dir(&tracks_dir).map_err(io_err)?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("idl0t") {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            match read_track(&path) {
                Ok(track) if track.id == stem => {
                    let full_json = std::fs::read_to_string(&path).map_err(io_err)?;
                    conn.execute(
                        "INSERT INTO tracks (track_id, name, venue_name, created_at_ms, updated_at_ms, full_json) VALUES (?1,?2,?3,?4,?5,?6)",
                        rusqlite::params![track.id, track.name, track.venue, track.created_at_ms, track.updated_at_ms, full_json],
                    )?;
                    report.tracks_indexed += 1;
                }
                Ok(_) => report.skipped.push(format!("{}: filename does not match its own track_id", path.display())),
                Err(e) => report.skipped.push(format!("{}: {e}", path.display())),
            }
        }
    }

    // 3. sessions (+ 4. laps, inline per session; 5. lap_summary, inline per session)
    let sessions_dir = data_root.join("sessions");
    if sessions_dir.is_dir() {
        for entry in std::fs::read_dir(&sessions_dir).map_err(io_err)?.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let session_id = entry.file_name().to_string_lossy().into_owned();
            match index_session_body(&conn, &dir, &session_id, &mut report.laps_indexed, &mut report.lap_summary_indexed, &mut report.skipped)?
            {
                SessionBodyOutcome::Indexed => report.sessions_indexed += 1,
                SessionBodyOutcome::TransientNoData | SessionBodyOutcome::Skipped => {}
            }
        }
    }

    // 6. workbooks — L3 shipped `.idl1wb` front-matter parsing (ruling R87):
    // walk `workbooks/*.idl1wb`, parse each file's front matter for
    // `workbook_id`/`name`, and upsert its row. A file that fails to parse
    // is skipped and reported, not inserted (same non-fatal-per-entity rule
    // as steps 2/3 above).
    let workbooks_dir = data_root.join("workbooks");
    if workbooks_dir.is_dir() {
        for entry in std::fs::read_dir(&workbooks_dir).map_err(io_err)?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("idl1wb") {
                continue;
            }
            match index_workbook_file(&conn, &path) {
                Ok(()) => report.workbooks_indexed += 1,
                Err(e) => report.skipped.push(format!("{}: {e}", path.display())),
            }
        }
    }

    // 7. schema version already set in create_schema.

    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    drop(conn);

    let bytes = std::fs::read(&staging_path).map_err(io_err)?;
    let final_path = data_root.join("catalog.sqlite");
    // The rebuild supersedes whatever is already at `final_path` (C4 §5: the
    // staging file is "atomically renamed over catalog.sqlite") — this is an
    // overwrite by design, not a caller-vs-peer-edit race, so `based_on`
    // tracks whatever is currently there and `rederive` always re-offers this
    // rebuild's own bytes rather than merging with a concurrent writer's.
    let based_on = std::fs::read(&final_path).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(data_root, &final_path, &bytes, based_on.as_deref(), |_current| bytes.clone())
        .map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;
    let _ = std::fs::remove_file(&staging_path);
    // Sidecars belong to the *previous* database — the new file just swapped
    // in owns none of its own WAL/SHM state yet (checkpointed above). Not a
    // C4 §5 contract line, just prudence: leaving a stale -wal/-shm next to a
    // brand-new catalog.sqlite could otherwise be replayed against it.
    let _ = std::fs::remove_file(data_root.join("catalog.sqlite-wal"));
    let _ = std::fs::remove_file(data_root.join("catalog.sqlite-shm"));

    Ok(report)
}

/// What [`index_session_body`] did with one session directory — the caller
/// (either [`rebuild_catalog`]'s loop or [`index_session`]) decides what
/// each outcome means for its own report shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionBodyOutcome {
    /// A `sessions` row (and its `laps`/`lap_summary` rows, if any) was
    /// inserted.
    Indexed,
    /// `session.json` or `data.parquet` doesn't exist yet under this
    /// directory — a transient import-in-progress state (C4 §2), not
    /// corruption; nothing was inserted and nothing was reported.
    TransientNoData,
    /// A parse failure or a rejected `sessions` insert (e.g. its blob is
    /// missing from `blobs`) — reported into `skipped`; nothing was
    /// inserted.
    Skipped,
}

/// C4 §5 steps 3–5 for exactly one session directory: reads `session.json`
/// and `data.parquet`'s file metadata, inserts the `sessions` row, then each
/// `laps` row and its `lap_summary` rows. Lifted out of [`rebuild_catalog`]'s
/// per-session loop body (task L2b T4) so [`index_session`] (incremental,
/// one session) and `rebuild_catalog` (full tree walk) share one
/// implementation — the loop's own observable behaviour is unchanged, since
/// this is the same code that used to run inline. `laps_indexed` and
/// `lap_summary_indexed` are incremented in place rather than returned,
/// matching how the two call sites already carry running totals
/// ([`RebuildReport`]'s fields, or [`SessionIndexReport`]'s).
fn index_session_body(
    conn: &Connection,
    dir: &Path,
    session_id: &str,
    laps_indexed: &mut usize,
    lap_summary_indexed: &mut usize,
    skipped: &mut Vec<String>,
) -> Result<SessionBodyOutcome, CatalogError> {
    let sj_path = dir.join("session.json");
    if !sj_path.is_file() {
        return Ok(SessionBodyOutcome::TransientNoData);
    }
    let doc = match read_session_json(&sj_path) {
        Ok(doc) => doc,
        Err(e) => {
            skipped.push(format!("{}: {e}", sj_path.display()));
            return Ok(SessionBodyOutcome::Skipped);
        }
    };
    let dp_path = dir.join("data.parquet");
    if !dp_path.is_file() {
        return Ok(SessionBodyOutcome::TransientNoData);
    }
    // `data.parquet`'s own file-level metadata (C1 §4.3) is the canonical
    // source for the `sessions` row's provenance columns — read here
    // without materializing any column (same technique as
    // [`index_lap_summary`]'s `timestamp_utc_ms` read below).
    let fields = match read_data_parquet_session_fields(&dp_path) {
        Ok(f) => f,
        Err(e) => {
            skipped.push(format!("{}: {e}", dp_path.display()));
            return Ok(SessionBodyOutcome::Skipped);
        }
    };
    // Session length = the actual recorded span of `data.parquet`'s `t`
    // column (µs), not any epoch-scale lap timestamp — mirrors
    // `Channel::duration_ms` (`session/mod.rs`) exactly.
    let duration_ms = match read_data_parquet_duration_ms(&dp_path) {
        Ok(d) => d,
        Err(e) => {
            skipped.push(format!("{}: {e}", dp_path.display()));
            return Ok(SessionBodyOutcome::Skipped);
        }
    };
    let lap_count = doc.laps.len() as i64;
    // TODO(idl0): C1 §6 has no imported_at_ms field; created_at_ms cannot
    // mean "import time" across rebuilds until C1 grows one
    // (runs/2026-09-03/decisions.md R14 item 3). Interim value:
    // `session.json`'s own filesystem mtime.
    let sj_meta = std::fs::metadata(&sj_path).map_err(io_err)?;
    let created_at_ms = file_mtime_ms(&sj_meta);
    // A user-supplied start (`doc.timestamp_source == "user"`) overrides
    // the parquet origin for display/sorting (R194); every other source
    // catalogues the parquet value unchanged.
    let effective_timestamp_utc_ms =
        crate::store::session_json::effective_start_ms(&doc, fields.timestamp_utc_ms);
    let inserted = conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, blob_sha256, source_format, device_id, config_checksum, importer_version, seam_correction_version, engine_version, timestamp_utc_ms, created_at_ms, rider, bike, venue_name, event_name, event_session, short_comment, tag, lap_count, duration_ms) \
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
        rusqlite::params![
            session_id,
            fields.blob_sha256,
            fields.source_format,
            fields.device_id,
            fields.config_checksum,
            fields.importer_version,
            fields.seam_correction_version,
            fields.engine_version,
            effective_timestamp_utc_ms,
            created_at_ms,
            doc.rider,
            doc.bike,
            doc.venue_name,
            doc.event_name,
            doc.event_session,
            doc.short_comment,
            doc.tag,
            lap_count,
            duration_ms
        ],
    );
    // Insert-order dependency (C4 §5 scan order): `sessions.blob_sha256`
    // `REFERENCES blobs(sha256)` under `PRAGMA foreign_keys = ON` — a
    // session whose blob is missing from `blobs` is rejected by the
    // constraint itself and lands here as a reported finding, not a
    // dangling reference (C4 §5's own stated mechanism). `rebuild_catalog`
    // guarantees this by running its own blobs scan (step 1) first;
    // `index_session` guarantees it by inserting this session's own blob
    // row itself before calling this function (ruling R84).
    match inserted {
        Ok(_) => {
            for lap in &doc.laps {
                // `track_id` from the session's track visits (C4 §5 step
                // 4): the first visit containing the lap, or deliberately
                // NULL if none does (a session with visits recorded for
                // only part of it, or none — not corruption).
                let track_id = match lap_track_id(lap, &doc.track_visits) {
                    Some(tid) => {
                        // `laps.track_id REFERENCES tracks(track_id)` would
                        // reject the insert under `PRAGMA foreign_keys = ON`
                        // if `tid` isn't already in `tracks` (inserted in
                        // step 2) — check first so the lap row itself still
                        // lands.
                        let exists = conn
                            .query_row("SELECT 1 FROM tracks WHERE track_id = ?1", rusqlite::params![tid], |_| Ok(()))
                            .is_ok();
                        if exists {
                            Some(tid)
                        } else {
                            skipped.push(format!("{session_id} lap {}: track_id {tid} not in tracks/", lap.lap_number));
                            None
                        }
                    }
                    None => None,
                };
                conn.execute(
                    "INSERT INTO laps (session_id, lap_number, lap_time_ms, track_id) VALUES (?1,?2,?3,?4)",
                    rusqlite::params![session_id, lap.lap_number, lap.lap_time_ms, track_id],
                )?;
                *laps_indexed += 1;
            }
            // Deliberately the raw parquet `fields.timestamp_utc_ms`, NOT
            // `effective_timestamp_utc_ms`: the cached laps' epoch stamps
            // were computed against the parquet origin, so swapping in a
            // user-supplied start here would shift every lap window.
            index_lap_summary(conn, dir, session_id, fields.timestamp_utc_ms, &doc.laps, lap_summary_indexed, skipped)?;
            Ok(SessionBodyOutcome::Indexed)
        }
        Err(e) => {
            skipped.push(format!("{}: {e}", sj_path.display()));
            Ok(SessionBodyOutcome::Skipped)
        }
    }
}

/// Outcome of [`index_session`] — counts plus non-fatal per-entity problems,
/// the same shape [`RebuildReport`] uses for its own steps 3–5 fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionIndexReport {
    pub laps_indexed: usize,
    /// Rows inserted into `lap_summary` (C4 §5 step 5).
    pub lap_summary_indexed: usize,
    pub skipped: Vec<String>,
}

/// Indexes (or re-indexes) exactly one session into an already-open
/// `catalog.sqlite` (C4 §5 steps 3–5): the `sessions` row and its `laps` /
/// `lap_summary` rows. Idempotent — any existing rows for `session_id` are
/// removed first (deleting the `sessions` row cascades to `laps`, and
/// `laps`' own cascade takes `lap_summary` with it, per this file's `DDL`),
/// so calling this twice in a row leaves the same state rather than
/// duplicating or erroring on a primary-key conflict. `tracks` rows are NOT
/// touched — step 2 is the track library's own concern, and a lap whose
/// visit names a `track_id` not yet in `tracks` simply gets a NULL
/// `laps.track_id` (`ON DELETE SET NULL` already covers a track deleted
/// later).
///
/// Also inserts this session's own `blobs` row if it isn't already there
/// (ruling R84): unlike [`rebuild_catalog`], which always runs its step 1
/// blobs scan before reaching any session, an incremental caller (import)
/// may be indexing a session whose blob was written to the CAS just now and
/// has never been scanned into `blobs` — without this, `sessions.blob_sha256
/// REFERENCES blobs(sha256)` would reject the insert under `PRAGMA
/// foreign_keys = ON` on every first-time import into an existing catalog.
///
/// The delete, the blob upsert, and the steps-3–5 insert all run in one
/// transaction, so a failure partway through cannot leave this session
/// half-indexed.
///
/// Errors with [`CatalogErrorKind::NotFound`] if `session_id` has no
/// `sessions/<session_id>/session.json` or no `data.parquet` yet — unlike
/// `rebuild_catalog`'s tree walk (which treats a mid-import directory as
/// transient and silently skips it while scanning everything else),
/// `index_session` names one specific, already-known session_id, so "there
/// is nothing to index yet" is reported to that specific caller rather than
/// swallowed.
pub fn index_session(conn: &Connection, data_root: &Path, session_id: &str) -> Result<SessionIndexReport, CatalogError> {
    let dir = data_root.join("sessions").join(session_id);
    if !dir.join("session.json").is_file() {
        return Err(CatalogError {
            kind: CatalogErrorKind::NotFound,
            message: format!("{session_id}: no session.json under {}", dir.display()),
        });
    }
    if !dir.join("data.parquet").is_file() {
        return Err(CatalogError {
            kind: CatalogErrorKind::NotFound,
            message: format!("{session_id}: no data.parquet under {}", dir.display()),
        });
    }
    let fields = read_data_parquet_session_fields(&dir.join("data.parquet"))?;

    conn.execute_batch("BEGIN IMMEDIATE;")?;
    let result = index_session_inner(conn, data_root, &dir, session_id, &fields.blob_sha256);
    if result.is_ok() {
        conn.execute_batch("COMMIT;")?;
    } else {
        // Best-effort — if the transaction is already gone (e.g. the error
        // came from a prior statement failure that SQLite itself aborted),
        // there is nothing left to roll back.
        let _ = conn.execute_batch("ROLLBACK;");
    }
    result
}

/// The transactional body of [`index_session`], factored out so the
/// `BEGIN`/`COMMIT`/`ROLLBACK` wrapper above has one `Result` to branch on
/// regardless of which step inside fails.
fn index_session_inner(
    conn: &Connection,
    data_root: &Path,
    dir: &Path,
    session_id: &str,
    blob_sha256: &str,
) -> Result<SessionIndexReport, CatalogError> {
    // Idempotence: remove this session's existing rows before re-inserting.
    // `laps`/`lap_summary` cascade from this single delete (see the DDL).
    conn.execute("DELETE FROM sessions WHERE session_id = ?1", rusqlite::params![session_id])?;

    ensure_blob_row(conn, data_root, blob_sha256)?;

    let mut report = SessionIndexReport::default();
    match index_session_body(conn, dir, session_id, &mut report.laps_indexed, &mut report.lap_summary_indexed, &mut report.skipped)? {
        SessionBodyOutcome::Indexed => {}
        // Both already checked by `index_session` before opening the
        // transaction — reaching either here would mean the files were
        // removed concurrently, which this pipeline's own write-once
        // contracts (C4 §2/§3) don't allow for a session mid-index.
        SessionBodyOutcome::TransientNoData | SessionBodyOutcome::Skipped => {
            return Err(CatalogError {
                kind: CatalogErrorKind::Sql,
                message: format!("{session_id}: session.json/data.parquet became unreadable during indexing"),
            });
        }
    }
    Ok(report)
}

/// Inserts `sha256`'s row into `blobs` if it isn't already there — the one
/// piece of C4 §5 step 1 [`index_session`] needs (ruling R84). Verifies the
/// blob's own bytes first (same check step 1's scan makes), rather than
/// trusting the caller's claim that this hash is the file's real digest.
fn ensure_blob_row(conn: &Connection, data_root: &Path, sha256: &str) -> Result<(), CatalogError> {
    verify_blob(data_root, sha256).map_err(|e| CatalogError { kind: CatalogErrorKind::Sql, message: e.to_string() })?;
    let path = blob_path(data_root, sha256);
    let meta = std::fs::metadata(&path).map_err(io_err)?;
    let mtime_ms = file_mtime_ms(&meta);
    conn.execute(
        "INSERT OR IGNORE INTO blobs (sha256, size_bytes, mtime_ms) VALUES (?1, ?2, ?3)",
        rusqlite::params![sha256, meta.len() as i64, mtime_ms],
    )?;
    Ok(())
}

/// C4 §5 step 5: for every `derived/<hash>.parquet` under `session_dir`, for
/// every channel column in the file and every already-inserted lap, compute
/// `(min, max, mean)` over that lap's time window and insert one
/// `lap_summary` row per `(lap, channel)` — skipped (not inserted, not
/// reported) when the window contains no finite sample, since that is not
/// itself corruption. `timestamp_utc_ms` (the session's own `data.parquet`
/// file metadata, already read by the caller) converts `session.json`'s
/// epoch-ms lap boundaries into the derived file's session-relative-µs `t`
/// axis (C1 §3.5); a session with laps but no `derived/` directory yet (no
/// estimator has run) contributes nothing here, silently — not corruption.
fn index_lap_summary(
    conn: &Connection,
    session_dir: &Path,
    session_id: &str,
    timestamp_utc_ms: i64,
    laps: &[LapJson],
    lap_summary_indexed: &mut usize,
    skipped: &mut Vec<String>,
) -> Result<(), CatalogError> {
    if laps.is_empty() {
        return Ok(());
    }
    let derived_dir = session_dir.join("derived");
    if !derived_dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(&derived_dir).map_err(io_err)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let derived_hash = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let (t_us, channels) = match read_derived_channels(&path) {
            Ok(v) => v,
            Err(e) => {
                skipped.push(format!("{}: {e}", path.display()));
                continue;
            }
        };

        for lap in laps {
            let start_us = (lap.start_timestamp_ms - timestamp_utc_ms) * 1000;
            let end_us = (lap.end_timestamp_ms - timestamp_utc_ms) * 1000;
            for (channel_id, values) in &channels {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                let mut sum = 0.0;
                let mut count = 0u64;
                for (&t, &v) in t_us.iter().zip(values.iter()) {
                    if t < start_us || t > end_us || !v.is_finite() {
                        continue;
                    }
                    min = min.min(v);
                    max = max.max(v);
                    sum += v;
                    count += 1;
                }
                if count == 0 {
                    continue;
                }
                let mean = sum / count as f64;
                conn.execute(
                    "INSERT OR REPLACE INTO lap_summary (session_id, lap_number, channel_id, derived_hash, min_value, max_value, mean_value) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    rusqlite::params![session_id, lap.lap_number, channel_id, derived_hash, min, max, mean],
                )?;
                *lap_summary_indexed += 1;
            }
        }
    }
    Ok(())
}

/// The `sessions` DDL columns that live in `data.parquet`'s own file-level
/// key-value metadata (C1 §4.3), read once per session. `pub(crate)` (not
/// private): `store::verify`'s checks #3 (missing blob) and #4 (session_id
/// identity) reuse this same read rather than a second, parallel parse of
/// the same file-level metadata (decisions.md R15 item 4).
pub(crate) struct DataParquetSessionFields {
    /// `data.parquet`'s own `session_id` metadata key — compared against
    /// the containing directory's name by `store::verify` check #4.
    pub(crate) session_id: String,
    pub(crate) blob_sha256: String,
    /// One of `"idl0"`/`"fit"`/`"gpx"`/`"csv"` (the `sessions.source_format`
    /// `CHECK` constraint's exact allowed set, C4 §5).
    source_format: String,
    device_id: Option<String>,
    config_checksum: Option<String>,
    importer_version: String,
    seam_correction_version: String,
    engine_version: String,
    /// Recording start, UTC milliseconds since the Unix epoch.
    timestamp_utc_ms: i64,
}

// TODO(idl0): delegate to store::parquet::read_session_metadata (R18 item 3)
/// Reads `data.parquet`'s file-level key-value metadata (C1 §4.3) — no
/// row-group/column materialization — into the fields the `sessions` DDL
/// requires. `Sql`-kind [`CatalogError`] when the file doesn't parse as
/// Parquet or a required key is missing/malformed (this session's row is
/// then reported as a skip, not inserted with fabricated values).
pub(crate) fn read_data_parquet_session_fields(path: &Path) -> Result<DataParquetSessionFields, CatalogError> {
    let bad = |msg: String| CatalogError { kind: CatalogErrorKind::Sql, message: msg };
    let file = std::fs::File::open(path).map_err(io_err)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| bad(e.to_string()))?;
    let kv = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .ok_or_else(|| bad(format!("{}: no file-level key-value metadata", path.display())))?;
    let get = |k: &str| -> Result<String, CatalogError> {
        kv.iter()
            .find(|e| e.key == k)
            .and_then(|e| e.value.clone())
            .ok_or_else(|| bad(format!("{}: missing required file metadata key {k}", path.display())))
    };
    let opt = |k: &str| -> Option<String> { kv.iter().find(|e| e.key == k).and_then(|e| e.value.clone()) };
    let timestamp_utc_ms: i64 =
        get("timestamp_utc_ms")?.parse().map_err(|_| bad(format!("{}: timestamp_utc_ms did not parse as i64", path.display())))?;
    Ok(DataParquetSessionFields {
        session_id: get("session_id")?,
        blob_sha256: get("blob_sha256")?,
        source_format: get("source_format")?,
        device_id: opt("device_id"),
        config_checksum: opt("config_checksum"),
        importer_version: get("importer_version")?,
        seam_correction_version: get("seam_correction_version")?,
        engine_version: get("engine_version")?,
        timestamp_utc_ms,
    })
}

/// Session length: `round((max(t) - min(t)) / 1000)` ms over `data.parquet`'s
/// own `t` column (session-relative µs, C1 §4.1) — mirrors
/// [`crate::session::Channel::duration_ms`] exactly, so this is a real
/// elapsed-time value, never an epoch-scale timestamp. Reads only the `t`
/// column via [`ProjectionMask`] (no other column is materialized), and
/// only the first and last row it reads (batches stream in `t`-ascending
/// order, C1 §3.5). `None` when the file has fewer than 2 rows.
fn read_data_parquet_duration_ms(path: &Path) -> Result<Option<i64>, CatalogError> {
    let file = std::fs::File::open(path).map_err(io_err)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| CatalogError { kind: CatalogErrorKind::Sql, message: e.to_string() })?;
    let mask = ProjectionMask::columns(builder.parquet_schema(), ["t"]);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;

    let mut first: Option<i64> = None;
    let mut last: Option<i64> = None;
    let mut n_rows: usize = 0;
    for batch in reader {
        let batch = batch.map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CatalogError { kind: CatalogErrorKind::Sql, message: format!("{}: t column is not Int64", path.display()) })?;
        n_rows += col.len();
        if col.len() > 0 {
            if first.is_none() {
                first = Some(col.value(0));
            }
            last = Some(col.value(col.len() - 1));
        }
    }
    if n_rows < 2 {
        return Ok(None);
    }
    let (first, last) = (first.unwrap(), last.unwrap());
    Ok(Some(((last - first) as f64 / 1000.0).round() as i64))
}

/// `lap`'s `track_id` by containment (C4 §5 step 4): the first entry in
/// `visits` (file order) whose `[start_timestamp_ms, end_timestamp_ms]`
/// window fully contains the lap's own window. `None` when no visit
/// contains the lap — deliberate, not an error: a session can have track
/// visits recorded for only part of it, or none at all.
fn lap_track_id(lap: &LapJson, visits: &[TrackVisitJson]) -> Option<String> {
    visits
        .iter()
        .find(|v| v.start_timestamp_ms <= lap.start_timestamp_ms && lap.end_timestamp_ms <= v.end_timestamp_ms)
        .map(|v| v.track_id.clone())
}

/// Reads a `derived/<hash>.parquet` file (contract C1 §5's shape: `t` as
/// `Int64`, every other column `Float64`, no `scale`/`offset`) into its
/// session-relative-µs time axis and `(channel_id, values)` pairs, in
/// column order. Materializes the whole file via `collect` +
/// `concat_batches` rather than folding row-group-by-row-group — a known
/// simplification of C4 §5 step 5's "streaming reduce" framing, accepted at
/// the per-session file sizes `derived/` files reach in practice.
fn read_derived_channels(path: &Path) -> Result<(Vec<i64>, Vec<(String, Vec<f64>)>), CatalogError> {
    let file = std::fs::File::open(path).map_err(io_err)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;
    let batches: Vec<RecordBatch> =
        reader.collect::<Result<Vec<_>, _>>().map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;
    let batch = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() })?;

    let t_col = batch
        .column_by_name("t")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>().map(|a| a.values().to_vec()))
        .ok_or_else(|| CatalogError { kind: CatalogErrorKind::Sql, message: format!("{}: missing/non-Int64 t column", path.display()) })?;

    let mut channels = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        if name == "t" {
            continue;
        }
        let col = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| CatalogError { kind: CatalogErrorKind::Sql, message: format!("{}: column {name} is not Float64", path.display()) })?;
        channels.push((name.clone(), col.values().to_vec()));
    }
    Ok((t_col, channels))
}

/// Deletes `session_id`'s row from `sessions`, and (via the schema's
/// cascading foreign keys) every row in `laps`/`lap_summary` for it in the
/// same statement: `laps.session_id` is `REFERENCES sessions(session_id)
/// ON DELETE CASCADE`, and `lap_summary`'s foreign key onto `laps` is
/// likewise `ON DELETE CASCADE` (see this file's `DDL`) — `open_catalog`
/// enables `PRAGMA foreign_keys = ON` per connection, so both cascades fire
/// for a connection built through it. Deliberately a single targeted
/// `DELETE`, not a `rebuild_catalog` call (needlessly expensive per delete,
/// R68) — the tauri command layer (`idl-rs-tauri`'s `delete_session`) calls
/// this after removing `<data>/sessions/<session_id>/` from disk.
///
/// Returns `true` if a `sessions` row existed and was removed, `false` if
/// `session_id` had no row (a no-op, not an error — the caller already
/// knows whether the session directory existed).
pub fn delete_session(conn: &Connection, session_id: &str) -> Result<bool, CatalogError> {
    let rows_deleted = conn.execute("DELETE FROM sessions WHERE session_id = ?1", rusqlite::params![session_id])?;
    Ok(rows_deleted > 0)
}

/// Updates `session_id`'s `sessions.timestamp_utc_ms` in place — the
/// catalog-agrees-with-the-file half of `set_session_start` (C3 §3.3,
/// ruling R194), so sorting/listing see a user-supplied start without a
/// full re-index. Returns `true` if a `sessions` row existed and was
/// updated, `false` if `session_id` had no row.
pub fn update_session_timestamp(conn: &Connection, session_id: &str, timestamp_utc_ms: i64) -> Result<bool, CatalogError> {
    let rows_updated = conn.execute(
        "UPDATE sessions SET timestamp_utc_ms = ?2 WHERE session_id = ?1",
        rusqlite::params![session_id, timestamp_utc_ms],
    )?;
    Ok(rows_updated > 0)
}

/// Upserts one `tracks` row for `track` (C4 §5, ruling R86 §4 — `save_track`
/// calls this only when `catalog.sqlite` already exists). `full_json` is the
/// exact `.idl0t` file text just written by
/// [`write_track`](crate::track_artifact::write::write_track), matching
/// `rebuild_catalog`'s own `full_json` column verbatim.
///
/// Deliberately `INSERT ... ON CONFLICT(track_id) DO UPDATE`, never a
/// delete-then-insert: `laps.track_id` is `REFERENCES tracks(track_id) ON
/// DELETE SET NULL` (this file's `DDL`), so deleting an existing `tracks`
/// row — even to immediately reinsert it — would null out every lap that
/// already names this track. `ON CONFLICT DO UPDATE` never deletes the row,
/// so that cascade never fires on an edit.
pub fn upsert_track(conn: &Connection, track: &crate::track_artifact::Track, full_json: &str) -> Result<(), CatalogError> {
    conn.execute(
        "INSERT INTO tracks (track_id, name, venue_name, created_at_ms, updated_at_ms, full_json) VALUES (?1,?2,?3,?4,?5,?6) \
         ON CONFLICT(track_id) DO UPDATE SET name = excluded.name, venue_name = excluded.venue_name, \
         created_at_ms = excluded.created_at_ms, updated_at_ms = excluded.updated_at_ms, full_json = excluded.full_json",
        rusqlite::params![track.id, track.name, track.venue, track.created_at_ms, track.updated_at_ms, full_json],
    )?;
    Ok(())
}

/// Deletes `track_id`'s row from `tracks` (ruling R86, `delete_track` C3
/// §3.2). `laps.track_id` is `REFERENCES tracks(track_id) ON DELETE SET
/// NULL` (this file's `DDL`), and `open_catalog` enables `PRAGMA
/// foreign_keys = ON` per connection, so this single `DELETE` also nulls
/// out `track_id` on every lap row that named this track — the tauri
/// command layer (`idl-rs-tauri`'s `delete_track`) calls this after removing
/// `<data>/tracks/<track_id>.idl0t` from disk. Deliberately a single
/// targeted `DELETE`, not a `rebuild_catalog` call, matching
/// [`delete_session`]'s own R68 reasoning.
///
/// Returns `true` if a `tracks` row existed and was removed, `false` if
/// `track_id` had no row (a no-op, not an error).
pub fn delete_track(conn: &Connection, track_id: &str) -> Result<bool, CatalogError> {
    let rows_deleted = conn.execute("DELETE FROM tracks WHERE track_id = ?1", rusqlite::params![track_id])?;
    Ok(rows_deleted > 0)
}

/// Inserts or updates one `workbooks` row (C4 §5, ruling R87) keyed on
/// `workbook_id` — a rename of the file keeps the id, so a later upsert with
/// the same `workbook_id` but a different `file_name`/`name` updates the
/// existing row rather than creating a second one. Called by
/// [`rebuild_catalog`]'s step 6 and by `idl-rs-tauri`'s `create_workbook`/
/// `save_workbook` commands right after their own atomic write, so
/// `list_workbooks` (C3 §3.2) reflects a new/edited workbook without waiting
/// for the next rebuild.
pub fn upsert_workbook(conn: &Connection, row: &WorkbookSummary) -> Result<(), CatalogError> {
    conn.execute(
        "INSERT INTO workbooks (workbook_id, file_name, name, updated_at_ms, size_bytes) VALUES (?1,?2,?3,?4,?5) \
         ON CONFLICT(workbook_id) DO UPDATE SET file_name = excluded.file_name, name = excluded.name, \
         updated_at_ms = excluded.updated_at_ms, size_bytes = excluded.size_bytes",
        rusqlite::params![row.workbook_id, row.file_name, row.name, row.updated_at_ms, row.size_bytes as i64],
    )?;
    Ok(())
}

/// Parses `path`'s front matter for `workbook_id`/`name`, reads its own file
/// metadata for `updated_at_ms` (mtime) and `size_bytes`, and upserts the
/// row. `file_name` is the file's own stem, not anything from the front
/// matter (C4 §2: `file_name` names the file; `id`/`name` come from inside
/// it). Returns the failure's message on a parse error — the caller
/// ([`rebuild_catalog`]'s step 6) reports and skips, it does not abort the
/// scan (CLAUDE.md §5).
fn index_workbook_file(conn: &Connection, path: &Path) -> Result<(), String> {
    let markdown = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let (front_matter, _) = parse_front_matter(&markdown).map_err(|e| e.to_string())?;
    let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let row = WorkbookSummary {
        workbook_id: front_matter.id,
        file_name,
        name: front_matter.name,
        updated_at_ms: file_mtime_ms(&meta),
        size_bytes: meta.len(),
    };
    upsert_workbook(conn, &row).map_err(|e| e.to_string())
}

fn io_err(e: std::io::Error) -> CatalogError {
    CatalogError { kind: CatalogErrorKind::Io, message: e.to_string() }
}

/// Filesystem mtime, UTC milliseconds since the Unix epoch. `0` if the
/// platform doesn't report one or it predates the epoch (not expected in
/// practice) — same fallback the `blobs.mtime_ms` insert has always used.
fn file_mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::store::blob::write_blob;
    use crate::store::derived::{write_derived_parquet, DerivedOutput};
    use crate::store::parquet::write_session_parquet;
    use crate::store::session_json::{empty_session_json, write_session_json, LapJson, TrackVisitJson};
    use crate::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn open_catalog_creates_file_and_applies_pragmas() {
        // Arrange
        let root = temp_root();
        let path = root.join("catalog.sqlite");

        // Act
        let conn = open_catalog(&path).unwrap();
        create_schema(&conn).unwrap();

        // Assert
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
        assert_eq!(journal_mode.to_lowercase(), "wal");
        let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(user_version, CATALOG_SCHEMA_VERSION);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// As [`write_full_session`], but takes an already-resolved
    /// `blob_sha256` instead of writing a fresh blob for it — lets a test
    /// reference a blob that was never written via [`write_blob`] (the
    /// missing-blob path).
    fn write_full_session_with_blob(
        root: &Path,
        session_id: &str,
        timestamp_utc_ms: i64,
        doc: &crate::store::session_json::SessionJson,
        blob_sha256: String,
    ) {
        write_session_json(root, session_id, doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256,
            channels: vec![Channel {
                channel_id: "IMU0_AccelX".to_string(),
                t_us: vec![0, 500_000, 1_000_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0, 3.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    /// Writes a session's `blob_sha256`-referenced blob, its `session.json`,
    /// and its `data.parquet` (whose `blob_sha256` file metadata matches the
    /// just-written blob, satisfying the `sessions` table's foreign key
    /// under `PRAGMA foreign_keys = ON`, C4 §5).
    fn write_full_session(root: &Path, session_id: &str, timestamp_utc_ms: i64, doc: &crate::store::session_json::SessionJson) {
        let blob_sha256 = write_blob(root, format!("raw bytes for {session_id}").as_bytes()).unwrap();
        write_full_session_with_blob(root, session_id, timestamp_utc_ms, doc, blob_sha256);
    }

    #[test]
    fn rebuild_catalog_indexes_a_blob_and_a_session() {
        // Arrange
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 0, &doc);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.blobs_indexed, 1);
        assert_eq!(report.sessions_indexed, 1);
        assert!(report.skipped.is_empty());

        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_json_says_user_the_sessions_row_carries_the_user_value() {
        // Arrange — parquet start is `0` (unknown), but `session.json`
        // carries a user-supplied start.
        let root = temp_root();
        let mut doc = empty_session_json("s1");
        doc.timestamp_utc_ms = Some(1_700_000_000_000);
        doc.timestamp_source = Some(TimestampSource::User);
        write_full_session(&root, "s1", 0, &doc);

        // Act
        rebuild_catalog(&root).unwrap();

        // Assert
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let timestamp_utc_ms: i64 =
            conn.query_row("SELECT timestamp_utc_ms FROM sessions WHERE session_id = 's1'", [], |r| r.get(0)).unwrap();
        assert_eq!(timestamp_utc_ms, 1_700_000_000_000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_skips_a_malformed_session_json_without_aborting_the_scan() {
        // Arrange
        let root = temp_root();
        let good = empty_session_json("s-good");
        write_full_session(&root, "s-good", 0, &good);
        let bad_dir = root.join("sessions").join("s-bad");
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::write(bad_dir.join("session.json"), b"not json").unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 1);
        assert_eq!(report.skipped.len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_json_without_a_data_parquet_yet_is_not_indexed_or_reported() {
        // Arrange — session.json alone (no data.parquet): a transient
        // import-in-progress state (C4 §2), not corruption.
        let root = temp_root();
        let doc = empty_session_json("s-mid-import");
        write_session_json(&root, "s-mid-import", &doc, None).unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 0);
        assert!(report.skipped.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_on_an_empty_tree_produces_an_empty_report_not_an_error() {
        // Arrange
        let root = temp_root();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report, RebuildReport::default());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_indexes_two_valid_workbooks_and_skips_one_malformed() {
        // Arrange — C4 §5 step 6: two valid `.idl1wb` front-matter files and
        // one that fails to parse (no `---`-delimited block at all).
        use crate::workbook::v3::front_matter::{render_front_matter, FrontMatter};

        let root = temp_root();
        let workbooks_dir = root.join("workbooks");
        std::fs::create_dir_all(&workbooks_dir).unwrap();
        let fm_a = FrontMatter {
            id: "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Fork tuning".to_string(),
            constants: Default::default(),
            units: Default::default(),
            version: 3,
            unknown: Default::default(),
        };
        let fm_b = FrontMatter {
            id: "1a2b3c4d-4b6a-4f1c-9c3d-2a7e8f9b0c1d".to_string(),
            name: "Session review".to_string(),
            constants: Default::default(),
            units: Default::default(),
            version: 3,
            unknown: Default::default(),
        };
        std::fs::write(workbooks_dir.join("Fork tuning.idl1wb"), render_front_matter(&fm_a)).unwrap();
        std::fs::write(workbooks_dir.join("Session review.idl1wb"), render_front_matter(&fm_b)).unwrap();
        std::fs::write(workbooks_dir.join("Broken.idl1wb"), b"not front matter at all").unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.workbooks_indexed, 2);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].contains("Broken.idl1wb"));

        let rows = crate::store::catalog_read::list_workbooks(&root).unwrap();
        assert_eq!(rows.len(), 2);
        // C3 §3.2 `list_workbooks` orders by `name` ASC.
        assert_eq!(rows[0].name, "Fork tuning");
        assert_eq!(rows[0].workbook_id, "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d");
        assert_eq!(rows[0].file_name, "Fork tuning");
        assert_eq!(rows[1].name, "Session review");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn upsert_workbook_a_rename_keeps_the_id() {
        // Arrange
        let root = temp_root();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        create_schema(&conn).unwrap();
        let row = crate::store::catalog_read::WorkbookSummary {
            workbook_id: "wb-1".to_string(),
            file_name: "Old name".to_string(),
            name: "Old name".to_string(),
            updated_at_ms: 1000,
            size_bytes: 10,
        };
        upsert_workbook(&conn, &row).unwrap();

        // Act — same `workbook_id`, new `file_name`/`name` (a rename).
        let renamed = crate::store::catalog_read::WorkbookSummary {
            workbook_id: "wb-1".to_string(),
            file_name: "New name".to_string(),
            name: "New name".to_string(),
            updated_at_ms: 2000,
            size_bytes: 20,
        };
        upsert_workbook(&conn, &renamed).unwrap();

        // Assert — one row, updated in place.
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM workbooks", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
        let (file_name, updated_at_ms): (String, i64) = conn
            .query_row("SELECT file_name, updated_at_ms FROM workbooks WHERE workbook_id = 'wb-1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(file_name, "New name");
        assert_eq!(updated_at_ms, 2000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_tracks_row_carries_the_wire_created_and_updated_at_ms() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        let json = r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line",
            "venue_name":"Whistler","created_at_ms":1111,"updated_at_ms":2222}}"#;
        std::fs::write(tracks_dir.join("t-1.idl0t"), json).unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.tracks_indexed, 1);
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let (created, updated): (i64, i64) = conn
            .query_row("SELECT created_at_ms, updated_at_ms FROM tracks WHERE track_id = 't-1'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(created, 1111);
        assert_eq!(updated, 2222);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_indexes_lap_summary_min_max_mean_within_the_lap_window() {
        // Arrange — session-relative t axis 0..=1_000_000us; the recording
        // started at epoch 10_000ms, so lap 1 (epoch [10_000, 10_500]ms)
        // covers session-relative [0, 500_000]us — samples 1.0 and 2.0 only.
        let root = temp_root();
        let session_id = "sess-1";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 10_000,
            end_timestamp_ms: 10_500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        write_full_session(&root, session_id, 10_000, &doc);

        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 500_000, 1_000_000],
            values: vec![1.0, 2.0, 3.0],
            nominal_rate_hz: 2.0,
            unit: "deg".to_string(),
        }];
        write_derived_parquet(&root, session_id, "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.lap_summary_indexed, 1);
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let (min, max, mean): (f64, f64, f64) = conn
            .query_row(
                "SELECT min_value, max_value, mean_value FROM lap_summary WHERE session_id = ?1 AND lap_number = 1 AND channel_id = 'Roll (deg)'",
                rusqlite::params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(min, 1.0);
        assert_eq!(max, 2.0);
        assert_eq!(mean, 1.5);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_removes_the_session_row_and_cascades_to_laps_and_lap_summary() {
        // Arrange — a session with one lap (-> `laps`) and one derived
        // channel within that lap's window (-> `lap_summary`), rebuilt so
        // all three tables actually hold a row for it before deletion.
        let root = temp_root();
        let session_id = "sess-1";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 10_000,
            end_timestamp_ms: 10_500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        write_full_session(&root, session_id, 10_000, &doc);
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 500_000, 1_000_000],
            values: vec![1.0, 2.0, 3.0],
            nominal_rate_hz: 2.0,
            unit: "deg".to_string(),
        }];
        write_derived_parquet(&root, session_id, "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();
        rebuild_catalog(&root).unwrap();

        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let sessions_before: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        let laps_before: i64 = conn.query_row("SELECT COUNT(*) FROM laps", [], |r| r.get(0)).unwrap();
        let lap_summary_before: i64 = conn.query_row("SELECT COUNT(*) FROM lap_summary", [], |r| r.get(0)).unwrap();
        assert_eq!((sessions_before, laps_before, lap_summary_before), (1, 1, 1));

        // Act
        let deleted = delete_session(&conn, session_id).unwrap();

        // Assert
        assert!(deleted);
        let sessions_after: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        let laps_after: i64 = conn.query_row("SELECT COUNT(*) FROM laps", [], |r| r.get(0)).unwrap();
        let lap_summary_after: i64 = conn.query_row("SELECT COUNT(*) FROM lap_summary", [], |r| r.get(0)).unwrap();
        assert_eq!((sessions_after, laps_after, lap_summary_after), (0, 0, 0));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_session_unknown_session_id_returns_false_and_deletes_nothing() {
        // Arrange
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 0, &doc);
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();

        // Act
        let deleted = delete_session(&conn, "nope").unwrap();

        // Assert
        assert!(!deleted);
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn update_session_timestamp_an_existing_row_takes_the_user_supplied_start() {
        // Arrange — a catalogued session whose importer left the start at 0
        // (C1 §3.1: unknown), the case `set_session_start` exists for.
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 0, &doc);
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();

        // Act
        let updated = update_session_timestamp(&conn, "s1", 1_700_000_000_000).unwrap();

        // Assert
        assert!(updated);
        let stored: i64 = conn
            .query_row("SELECT timestamp_utc_ms FROM sessions WHERE session_id = 's1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(stored, 1_700_000_000_000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn update_session_timestamp_an_unknown_session_id_returns_false_and_changes_nothing() {
        // Arrange
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 10_000, &doc);
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();

        // Act
        let updated = update_session_timestamp(&conn, "nope", 1_700_000_000_000).unwrap();

        // Assert
        assert!(!updated);
        let stored: i64 = conn
            .query_row("SELECT timestamp_utc_ms FROM sessions WHERE session_id = 's1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(stored, 10_000);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Minimal domain `Track` for [`upsert_track`] tests — `store::catalog`
    /// has no reason to exercise gate/timing fields, only the five scalar
    /// columns `upsert_track` writes.
    fn minimal_track(id: &str, name: &str) -> crate::track_artifact::Track {
        crate::track_artifact::Track {
            id: id.to_string(),
            name: name.to_string(),
            venue: "Whistler".to_string(),
            timing: None,
            sector_gates: Vec::new(),
            neutral_zones: Vec::new(),
            reference_polyline: Vec::new(),
            created_at_ms: 111,
            updated_at_ms: 222,
        }
    }

    #[test]
    fn upsert_track_on_an_empty_catalog_inserts_the_row() {
        // Arrange
        let root = temp_root();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        create_schema(&conn).unwrap();
        let track = minimal_track("t-1", "A-Line");

        // Act
        upsert_track(&conn, &track, "{}").unwrap();

        // Assert
        let (name, venue, created, updated, full_json): (String, String, i64, i64, String) = conn
            .query_row(
                "SELECT name, venue_name, created_at_ms, updated_at_ms, full_json FROM tracks WHERE track_id = ?1",
                rusqlite::params!["t-1"],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();
        assert_eq!(name, "A-Line");
        assert_eq!(venue, "Whistler");
        assert_eq!(created, 111);
        assert_eq!(updated, 222);
        assert_eq!(full_json, "{}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn upsert_track_called_twice_updates_the_row_rather_than_duplicating_it() {
        // Arrange
        let root = temp_root();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        create_schema(&conn).unwrap();
        upsert_track(&conn, &minimal_track("t-1", "A-Line"), "{}").unwrap();

        // Act
        let mut edited = minimal_track("t-1", "B-Line");
        edited.updated_at_ms = 333;
        upsert_track(&conn, &edited, "{\"edited\":true}").unwrap();

        // Assert
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM tracks", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
        let (name, updated): (String, i64) =
            conn.query_row("SELECT name, updated_at_ms FROM tracks WHERE track_id = ?1", rusqlite::params!["t-1"], |r| {
                Ok((r.get(0)?, r.get(1)?))
            }).unwrap();
        assert_eq!(name, "B-Line");
        assert_eq!(updated, 333);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn upsert_track_updating_a_track_with_an_existing_lap_reference_does_not_null_it_out() {
        // Arrange — `laps.track_id` is `ON DELETE SET NULL`; a naive
        // delete-then-insert upsert would fire that cascade on every edit of
        // an already-visited track. This is the regression test for using
        // `ON CONFLICT ... DO UPDATE` instead.
        let root = temp_root();
        let session_id = "s1";
        write_full_session(&root, session_id, 0, &empty_session_json(session_id));
        rebuild_catalog(&root).unwrap();
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        upsert_track(&conn, &minimal_track("t-1", "A-Line"), "{}").unwrap();
        conn.execute(
            "INSERT INTO laps (session_id, lap_number, lap_time_ms, track_id) VALUES (?1, 1, 1000, ?2)",
            rusqlite::params![session_id, "t-1"],
        )
        .unwrap();

        // Act
        upsert_track(&conn, &minimal_track("t-1", "A-Line-Renamed"), "{}").unwrap();

        // Assert
        let track_id: Option<String> = conn
            .query_row("SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1", rusqlite::params![session_id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(track_id.as_deref(), Some("t-1"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_with_laps_but_no_derived_dir_indexes_no_lap_summary() {
        // Arrange
        let root = temp_root();
        let session_id = "sess-2";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 0,
            end_timestamp_ms: 500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        write_full_session(&root, session_id, 0, &doc);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert — data.parquet exists (the session is indexed), but there is
        // no `derived/` dir: nothing to summarize, and this is not reported
        // as a problem.
        assert_eq!(report.sessions_indexed, 1);
        assert_eq!(report.lap_summary_indexed, 0);
        assert!(report.skipped.is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_duration_ms_is_the_t_span_not_epoch_time() {
        // Arrange — `data.parquet`'s `t` axis spans 0..=12_345_000us;
        // `duration_ms` must reflect that span, not any epoch-scale lap
        // timestamp (the old, wrong formula).
        let root = temp_root();
        let session_id = "sess-dur";
        let doc = empty_session_json(session_id);
        let blob_sha256 = write_blob(&root, b"raw bytes for sess-dur").unwrap();
        write_session_json(&root, session_id, &doc, None).unwrap();
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 1_700_000_000_000,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256,
            channels: vec![Channel {
                channel_id: "IMU0_AccelX".to_string(),
                t_us: vec![0, 12_345_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0]),
                source_kind: "imu0".to_string(),
                unit: "g".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 1);
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let duration_ms: i64 = conn
            .query_row("SELECT duration_ms FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0))
            .unwrap();
        assert_eq!(duration_ms, 12_345);
        assert!(duration_ms < 1_000_000, "duration_ms must be a small elapsed time, not an epoch-scale value");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_lap_track_id_from_a_containing_track_visit() {
        // Arrange
        let root = temp_root();
        let tracks_dir = root.join("tracks");
        std::fs::create_dir_all(&tracks_dir).unwrap();
        let track_json = r#"{"track_artifact_version":1,"track":{"track_id":"t-1","name":"A-Line",
            "venue_name":"Whistler","created_at_ms":1,"updated_at_ms":2}}"#;
        std::fs::write(tracks_dir.join("t-1.idl0t"), track_json).unwrap();

        let session_id = "sess-visit";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 1_000,
            end_timestamp_ms: 1_500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        doc.track_visits = vec![TrackVisitJson {
            visit_id: "v-1".to_string(),
            track_id: "t-1".to_string(),
            start_timestamp_ms: 500,
            end_timestamp_ms: 2_000,
            laps: Vec::new(),
        }];
        write_full_session(&root, session_id, 0, &doc);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.laps_indexed, 1);
        assert!(report.skipped.is_empty());
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let track_id: Option<String> = conn
            .query_row(
                "SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1",
                rusqlite::params![session_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(track_id, Some("t-1".to_string()));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_lap_outside_every_track_visit_has_null_track_id() {
        // Arrange — the recorded visit does not cover the lap's window.
        let root = temp_root();
        let session_id = "sess-no-visit";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 10_000,
            end_timestamp_ms: 10_500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        doc.track_visits = vec![TrackVisitJson {
            visit_id: "v-1".to_string(),
            track_id: "t-1".to_string(),
            start_timestamp_ms: 0,
            end_timestamp_ms: 1_000,
            laps: Vec::new(),
        }];
        write_full_session(&root, session_id, 0, &doc);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.laps_indexed, 1);
        assert!(report.skipped.is_empty());
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let track_id: Option<String> = conn
            .query_row(
                "SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1",
                rusqlite::params![session_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(track_id, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_lap_visit_references_a_track_id_not_in_tracks_dir() {
        // Arrange — no `tracks/t-missing.idl0t` is ever written, so the FK
        // would reject the lap insert if `track_id` were passed through
        // unconditionally.
        let root = temp_root();
        let session_id = "sess-dangling-visit";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![LapJson {
            lap_number: 1,
            start_timestamp_ms: 1_000,
            end_timestamp_ms: 1_500,
            raw_elapsed_ms: 500,
            lap_time_ms: 500,
            start_time_secs: 0.0,
            end_time_secs: 0.5,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }];
        doc.track_visits = vec![TrackVisitJson {
            visit_id: "v-1".to_string(),
            track_id: "t-missing".to_string(),
            start_timestamp_ms: 0,
            end_timestamp_ms: 2_000,
            laps: Vec::new(),
        }];
        write_full_session(&root, session_id, 0, &doc);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert — lap row still inserted, track_id NULL, one skip entry.
        assert_eq!(report.laps_indexed, 1);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].contains("t-missing"));
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let track_id: Option<String> = conn
            .query_row(
                "SELECT track_id FROM laps WHERE session_id = ?1 AND lap_number = 1",
                rusqlite::params![session_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(track_id, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_created_at_ms_is_session_json_mtime_not_zero() {
        // Arrange
        let root = temp_root();
        let session_id = "sess-created";
        let doc = empty_session_json(session_id);
        write_full_session(&root, session_id, 0, &doc);
        let sj_path = root.join("sessions").join(session_id).join("session.json");
        let stat_mtime_ms = std::fs::metadata(&sj_path)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 1);
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let created_at_ms: i64 = conn
            .query_row("SELECT created_at_ms FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0))
            .unwrap();
        assert!(created_at_ms > 0);
        assert!(created_at_ms >= stat_mtime_ms);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_session_referencing_a_missing_blob_is_skipped_scan_continues() {
        // Arrange — `s-missing`'s `data.parquet` claims a `blob_sha256` that
        // was never written via `write_blob`; `s-good` is a normal session
        // in the same tree.
        let root = temp_root();
        let good = empty_session_json("s-good");
        write_full_session(&root, "s-good", 0, &good);

        let missing_doc = empty_session_json("s-missing");
        let fake_sha256 = "f".repeat(64);
        write_full_session_with_blob(&root, "s-missing", 0, &missing_doc, fake_sha256);

        // Act
        let report = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(report.sessions_indexed, 1);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].contains("s-missing"));

        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
        let good_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions WHERE session_id = 's-good'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(good_count, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rebuild_catalog_twice_on_the_same_root_overwrites_the_previous_catalog() {
        // Arrange — R16: `rebuild_catalog`'s final swap must overwrite an
        // already-existing catalog.sqlite (C4 §5's rebuild procedure), not
        // treat every rebuild after the first as a conflict.
        let root = temp_root();
        let doc = empty_session_json("s1");
        write_full_session(&root, "s1", 0, &doc);

        // Act
        let first = rebuild_catalog(&root).unwrap();
        let second = rebuild_catalog(&root).unwrap();

        // Assert
        assert_eq!(second.blobs_indexed, first.blobs_indexed);
        assert_eq!(second.sessions_indexed, first.sessions_indexed);
        assert_eq!(second.tracks_indexed, first.tracks_indexed);
        assert_eq!(second.laps_indexed, first.laps_indexed);
        assert_eq!(second.lap_summary_indexed, first.lap_summary_indexed);
        assert_eq!(second.workbooks_indexed, first.workbooks_indexed);

        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(user_version, CATALOG_SCHEMA_VERSION);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A minimal, valid lap over `[start_ms, end_ms]`, no sectors/neutral
    /// zones — the shape `index_session`'s own tests need without pulling
    /// in `laps::detect_laps`'s real detection pipeline.
    fn simple_lap(lap_number: u32, start_ms: i64, end_ms: i64) -> LapJson {
        LapJson {
            lap_number,
            start_timestamp_ms: start_ms,
            end_timestamp_ms: end_ms,
            raw_elapsed_ms: end_ms - start_ms,
            lap_time_ms: end_ms - start_ms,
            start_time_secs: start_ms as f64 / 1000.0,
            end_time_secs: end_ms as f64 / 1000.0,
            sectors: Vec::new(),
            neutral_zone_visits: Vec::new(),
        }
    }

    /// An already-open, freshly-created (empty) catalog at `<root>/catalog.sqlite`
    /// — the fixture `index_session`'s own tests use when they want a catalog
    /// present but call `index_session` directly rather than through
    /// `rebuild_catalog` or `finish_import`.
    fn empty_catalog(root: &Path) -> Connection {
        let conn = open_catalog(&root.join("catalog.sqlite")).unwrap();
        create_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn index_session_three_laps_indexes_three_lap_rows_and_sets_lap_count() {
        // Arrange
        let root = temp_root();
        let session_id = "sess-idx";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![simple_lap(1, 0, 500), simple_lap(2, 500, 1000), simple_lap(3, 1000, 1500)];
        write_full_session(&root, session_id, 0, &doc);
        let conn = empty_catalog(&root);

        // Act
        let report = index_session(&conn, &root, session_id).unwrap();

        // Assert
        assert_eq!(report.laps_indexed, 3);
        assert!(report.skipped.is_empty());
        let lap_count: Option<i64> =
            conn.query_row("SELECT lap_count FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(lap_count, Some(3));
        let laps_rows: i64 =
            conn.query_row("SELECT COUNT(*) FROM laps WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(laps_rows, 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_session_called_twice_is_idempotent_no_primary_key_error() {
        // Arrange
        let root = temp_root();
        let session_id = "sess-idem";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![simple_lap(1, 0, 500)];
        write_full_session(&root, session_id, 0, &doc);
        let conn = empty_catalog(&root);

        // Act
        index_session(&conn, &root, session_id).unwrap();
        let second = index_session(&conn, &root, session_id).unwrap();

        // Assert
        assert_eq!(second.laps_indexed, 1);
        let laps_rows: i64 =
            conn.query_row("SELECT COUNT(*) FROM laps WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(laps_rows, 1);
        let sessions_rows: i64 = conn.query_row("SELECT COUNT(*) FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(sessions_rows, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_session_after_laps_shrink_from_three_to_two_drops_the_stale_lap_and_its_summary() {
        // Arrange — three laps, one derived channel spanning all three, so
        // lap 3 gets a `lap_summary` row that must disappear along with the
        // `laps` row once `session.json` is rewritten down to two laps.
        let root = temp_root();
        let session_id = "sess-shrink";
        let mut doc = empty_session_json(session_id);
        doc.laps = vec![simple_lap(1, 0, 500), simple_lap(2, 500, 1000), simple_lap(3, 1000, 1500)];
        write_full_session(&root, session_id, 0, &doc);
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 500_000, 1_000_000, 1_500_000],
            values: vec![1.0, 2.0, 3.0, 4.0],
            nominal_rate_hz: 2.0,
            unit: "deg".to_string(),
        }];
        write_derived_parquet(&root, session_id, "test_kind", &[], &serde_json::json!({}), &outputs, 0).unwrap();
        let conn = empty_catalog(&root);
        index_session(&conn, &root, session_id).unwrap();
        let lap3_summary_before: i64 = conn
            .query_row("SELECT COUNT(*) FROM lap_summary WHERE session_id = ?1 AND lap_number = 3", rusqlite::params![session_id], |r| r.get(0))
            .unwrap();
        assert_eq!(lap3_summary_before, 1);

        // Act — rewrite session.json with only laps 1-2, re-index.
        let sj_path = root.join("sessions").join(session_id).join("session.json");
        let current_hash = sha256_hex(&std::fs::read(&sj_path).unwrap());
        doc.laps = vec![simple_lap(1, 0, 500), simple_lap(2, 500, 1000)];
        write_session_json(&root, session_id, &doc, Some(&current_hash)).unwrap();
        let report = index_session(&conn, &root, session_id).unwrap();

        // Assert
        assert_eq!(report.laps_indexed, 2);
        let laps_rows: i64 =
            conn.query_row("SELECT COUNT(*) FROM laps WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(laps_rows, 2);
        let lap3_rows: i64 =
            conn.query_row("SELECT COUNT(*) FROM laps WHERE session_id = ?1 AND lap_number = 3", rusqlite::params![session_id], |r| r.get(0))
                .unwrap();
        assert_eq!(lap3_rows, 0);
        let lap3_summary_after: i64 = conn
            .query_row("SELECT COUNT(*) FROM lap_summary WHERE session_id = ?1 AND lap_number = 3", rusqlite::params![session_id], |r| r.get(0))
            .unwrap();
        assert_eq!(lap3_summary_after, 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_session_unknown_session_id_is_not_found() {
        // Arrange
        let root = temp_root();
        let conn = empty_catalog(&root);

        // Act
        let err = index_session(&conn, &root, "does-not-exist").unwrap_err();

        // Assert
        assert_eq!(err.kind, CatalogErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn index_session_inserts_its_own_missing_blob_row_before_the_session_row() {
        // Arrange — ruling R84: a catalog created before this session's blob
        // ever existed (as `rebuild_catalog` would leave it on an earlier,
        // empty tree) has no `blobs` row for it; `index_session` must
        // insert one itself rather than fail the `sessions.blob_sha256`
        // foreign key.
        let root = temp_root();
        let conn = empty_catalog(&root); // catalog exists before the session does
        let session_id = "sess-blob";
        let doc = empty_session_json(session_id);
        write_full_session(&root, session_id, 0, &doc);
        let blobs_before: i64 = conn.query_row("SELECT COUNT(*) FROM blobs", [], |r| r.get(0)).unwrap();
        assert_eq!(blobs_before, 0);

        // Act
        let report = index_session(&conn, &root, session_id).unwrap();

        // Assert
        assert!(report.skipped.is_empty());
        let sessions_rows: i64 = conn.query_row("SELECT COUNT(*) FROM sessions WHERE session_id = ?1", rusqlite::params![session_id], |r| r.get(0)).unwrap();
        assert_eq!(sessions_rows, 1);
        let blobs_after: i64 = conn.query_row("SELECT COUNT(*) FROM blobs", [], |r| r.get(0)).unwrap();
        assert_eq!(blobs_after, 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_catalog_on_a_fresh_empty_file_creates_the_schema() {
        // Arrange
        let root = temp_root();
        let path = root.join("catalog.sqlite");

        // Act
        let conn = open_catalog(&path).unwrap();

        // Assert
        let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(user_version, CATALOG_SCHEMA_VERSION);
        let workbooks: i64 = conn.query_row("SELECT count(*) FROM workbooks", [], |r| r.get(0)).unwrap();
        assert_eq!(workbooks, 0);
    }
}
