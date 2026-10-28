//! The session-set result cache (C4 §2/§6 amendment, session-sets spec §4.3):
//! `<data>/cache/eval/results.sqlite`, one row per (member, result name),
//! keyed by the content of every input that produced it.
//!
//! Local only (ruling Q5 of 2026-10-28): never in the sync manifest. Like
//! the catalog it is an index: deletable at any time, and a missing or
//! unreadable file is an empty cache, never an error the user sees.
//!
//! Only deterministic results are stored: values, and `math_*` errors. A
//! member whose decode was refused or failed stores nothing (the runner
//! never calls [`EvalCache::put_member`] for it).

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OpenFlags};

use crate::math::units::UnitLabel;
use crate::math::{MathEvalError, MathEvalErrorKind};
use crate::workbook::v3::set_eval::{result_key, MemberResult, MemberResults, MemberValue};

/// Bytes above which [`EvalCache::prune`] evicts least-recently-used rows:
/// 256 MiB (spec §4.3).
pub const PRUNE_ABOVE_BYTES: u64 = 256 * 1024 * 1024;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS results (
  key          BLOB PRIMARY KEY,
  session_id   TEXT NOT NULL,
  name         TEXT NOT NULL,
  shape        TEXT NOT NULL,
  unit_json    TEXT NOT NULL,
  payload      BLOB,
  error_json   TEXT,
  last_used_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS results_by_session ON results(session_id);
";

/// The cache file's path under data root `data_dir`.
pub fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("cache").join("eval").join("results.sqlite")
}

/// Why the cache could not be used. Callers treat every one of these as "no
/// cache" and evaluate; it is reported, never fatal.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalCacheError {
    /// Display-ready English.
    pub message: String,
}

impl From<rusqlite::Error> for EvalCacheError {
    fn from(e: rusqlite::Error) -> Self {
        EvalCacheError { message: format!("eval cache: {e}") }
    }
}

/// An open `results.sqlite`.
pub struct EvalCache {
    conn: Connection,
}

impl EvalCache {
    /// Opens (creating the file and its directory when absent) the cache
    /// under `data_dir`.
    ///
    /// # Errors
    /// The directory cannot be created or the file cannot be opened.
    pub fn open(data_dir: &Path) -> Result<Self, EvalCacheError> {
        let path = cache_path(data_dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EvalCacheError { message: format!("eval cache: {e}") })?;
        }
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE)?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.execute_batch(SCHEMA)?;
        Ok(EvalCache { conn })
    }

    /// Every result in `names` for the member whose key is `member_key`, or
    /// `None` unless **all** of them are stored (a partial hit is a miss: the
    /// member is evaluated again). Touches `last_used_ms` on a hit.
    ///
    /// # Errors
    /// A query failed.
    pub fn get_member(
        &self,
        member_key: &[u8; 32],
        names: &[String],
        now_ms: i64,
    ) -> Result<Option<MemberResults>, EvalCacheError> {
        let mut out = MemberResults::default();
        let mut stmt = self.conn.prepare_cached("SELECT shape, unit_json, payload, error_json FROM results WHERE key = ?1")?;
        for name in names {
            let key = result_key(member_key, name);
            let row = stmt
                .query_row(params![&key[..]], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<Vec<u8>>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                })
                .map(Some)
                .or_else(|e| if e == rusqlite::Error::QueryReturnedNoRows { Ok(None) } else { Err(e) })?;
            let Some((shape, unit_json, payload, error_json)) = row else { return Ok(None) };
            let Some(result) = decode(&shape, &unit_json, payload.as_deref(), error_json.as_deref()) else {
                return Ok(None);
            };
            out.0.insert(name.clone(), result);
        }
        let mut touch = self.conn.prepare_cached("UPDATE results SET last_used_ms = ?2 WHERE key = ?1")?;
        for name in names {
            touch.execute(params![&result_key(member_key, name)[..], now_ms])?;
        }
        Ok(Some(out))
    }

    /// Stores every result of one member in that member's own transaction, so
    /// a cancelled run keeps what it finished (spec §4.3).
    ///
    /// # Errors
    /// A write failed; nothing of this member is stored.
    pub fn put_member(
        &mut self,
        session_id: &str,
        member_key: &[u8; 32],
        results: &MemberResults,
        now_ms: i64,
    ) -> Result<(), EvalCacheError> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO results (key, session_id, name, shape, unit_json, payload, error_json, last_used_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (name, result) in &results.0 {
                let (shape, payload, error_json) = encode(&result.value);
                let unit_json = serde_json::to_string(&result.unit).unwrap_or_else(|_| "null".into());
                stmt.execute(params![
                    &result_key(member_key, name)[..],
                    session_id,
                    name,
                    shape,
                    unit_json,
                    payload,
                    error_json,
                    now_ms
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Deletes least-recently-used rows until the stored payloads total at
    /// most `limit_bytes`. Returns the number of rows deleted.
    ///
    /// # Errors
    /// A query failed.
    pub fn prune(&self, limit_bytes: u64) -> Result<usize, EvalCacheError> {
        let total: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(COALESCE(LENGTH(payload), 0) + LENGTH(unit_json) + COALESCE(LENGTH(error_json), 0) + 64), 0) FROM results",
            [],
            |r| r.get(0),
        )?;
        if (total as u64) <= limit_bytes {
            return Ok(0);
        }
        let excess = total as u64 - limit_bytes;
        let mut stmt = self.conn.prepare(
            "SELECT key, COALESCE(LENGTH(payload), 0) + LENGTH(unit_json) + COALESCE(LENGTH(error_json), 0) + 64 FROM results ORDER BY last_used_ms ASC",
        )?;
        let mut freed = 0u64;
        let mut doomed: Vec<Vec<u8>> = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            if freed >= excess {
                break;
            }
            doomed.push(row.get(0)?);
            freed += row.get::<_, i64>(1)? as u64;
        }
        drop(rows);
        for key in &doomed {
            self.conn.execute("DELETE FROM results WHERE key = ?1", params![key])?;
        }
        Ok(doomed.len())
    }
}

/// `(shape, payload, error_json)` of one value. A `[t]` value is stored as a
/// marker row (`shape` `"[t]"`, no payload) so a member whose demand names one
/// can still be a full hit.
fn encode(value: &MemberValue) -> (&'static str, Option<Vec<u8>>, Option<String>) {
    match value {
        MemberValue::Scalar(v) => ("[]", Some(v.to_le_bytes().to_vec()), None),
        MemberValue::Laps { laps, values } => {
            let mut bytes = Vec::with_capacity(16 * laps.len());
            for &n in laps {
                bytes.extend_from_slice(&(n as f64).to_le_bytes());
            }
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            ("[lap]", Some(bytes), None)
        }
        MemberValue::FullRate => ("[t]", None, None),
        MemberValue::Error(e) => {
            ("[]", None, Some(serde_json::json!({ "kind": kind_name(e.kind), "message": e.message }).to_string()))
        }
    }
}

/// The inverse of [`encode`]; `None` for a row this build cannot read (the
/// member is then evaluated again).
fn decode(shape: &str, unit_json: &str, payload: Option<&[u8]>, error_json: Option<&str>) -> Option<MemberResult> {
    let unit: UnitLabel = serde_json::from_str(unit_json).ok()?;
    let f64s = |b: &[u8]| -> Vec<f64> {
        b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk"))).collect()
    };
    let value = match (shape, payload, error_json) {
        (_, _, Some(err)) => {
            let v: serde_json::Value = serde_json::from_str(err).ok()?;
            let kind = kind_from_name(v.get("kind")?.as_str()?)?;
            MemberValue::Error(MathEvalError::new(kind, v.get("message")?.as_str()?.to_string()))
        }
        ("[]", Some(b), None) if b.len() == 8 => MemberValue::Scalar(f64s(b)[0]),
        ("[lap]", Some(b), None) if b.len() % 16 == 0 => {
            let all = f64s(b);
            let (laps, values) = all.split_at(all.len() / 2);
            MemberValue::Laps { laps: laps.iter().map(|&n| n as u32).collect(), values: values.to_vec() }
        }
        ("[t]", None, None) => MemberValue::FullRate,
        _ => return None,
    };
    Some(MemberResult { value, unit })
}

/// A stable name for a [`MathEvalErrorKind`] in the cache file.
fn kind_name(kind: MathEvalErrorKind) -> &'static str {
    match kind {
        MathEvalErrorKind::Parse => "parse",
        MathEvalErrorKind::UnknownFunction => "unknown_function",
        MathEvalErrorKind::UnknownChannel => "unknown_channel",
        MathEvalErrorKind::ArgCount => "arg_count",
        MathEvalErrorKind::Type => "type",
        MathEvalErrorKind::DivisionByZero => "division_by_zero",
        MathEvalErrorKind::NoLapContext => "no_lap_context",
        MathEvalErrorKind::NotImplemented => "not_implemented",
        MathEvalErrorKind::Runtime => "runtime",
    }
}

/// The inverse of [`kind_name`].
fn kind_from_name(name: &str) -> Option<MathEvalErrorKind> {
    Some(match name {
        "parse" => MathEvalErrorKind::Parse,
        "unknown_function" => MathEvalErrorKind::UnknownFunction,
        "unknown_channel" => MathEvalErrorKind::UnknownChannel,
        "arg_count" => MathEvalErrorKind::ArgCount,
        "type" => MathEvalErrorKind::Type,
        "division_by_zero" => MathEvalErrorKind::DivisionByZero,
        "no_lap_context" => MathEvalErrorKind::NoLapContext,
        "not_implemented" => MathEvalErrorKind::NotImplemented,
        "runtime" => MathEvalErrorKind::Runtime,
        _ => return None,
    })
}

/// Opens `<data>/catalog.sqlite` **read-only**, for data identities (spec
/// §4.3: the catalog's `sessions` row holds the three fields the footer
/// would give). `None` when there is no catalog or it will not open.
pub fn open_catalog_read_only(data_dir: &Path) -> Option<Connection> {
    let path = data_dir.join("catalog.sqlite");
    if !path.exists() {
        return None;
    }
    Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).ok()
}

/// What the catalog knows of one session: its data identity
/// (`blob_sha256|importer_version|seam_correction_version`) and its start,
/// ms since the Unix epoch.
pub fn catalog_identity(conn: &Connection, session_id: &str) -> Option<(String, i64)> {
    conn.query_row(
        "SELECT blob_sha256, importer_version, seam_correction_version, timestamp_utc_ms FROM sessions WHERE session_id = ?1",
        params![session_id],
        |r| {
            Ok((
                format!("{}|{}|{}", r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?),
                r.get::<_, i64>(3)?,
            ))
        },
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("idl-rs-eval-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn results() -> MemberResults {
        let mut r = MemberResults::default();
        r.0.insert("a".into(), MemberResult { value: MemberValue::Scalar(1.5), unit: UnitLabel::Known { text: "s".into() } });
        r.0.insert("b".into(), MemberResult {
            value: MemberValue::Laps { laps: vec![2, 3], values: vec![10.0, f64::NAN] },
            unit: UnitLabel::Dimensionless,
        });
        r.0.insert("c".into(), MemberResult { value: MemberValue::FullRate, unit: UnitLabel::Dimensionless });
        r.0.insert("d".into(), MemberResult {
            value: MemberValue::Error(MathEvalError::new(MathEvalErrorKind::NoLapContext, "no laps")),
            unit: UnitLabel::Dimensionless,
        });
        r
    }

    #[test]
    fn result_cache_a_math_error_stored_and_served_again() {
        // Arrange
        let dir = temp_dir();
        let mut cache = EvalCache::open(&dir).unwrap();
        let key = [7u8; 32];
        cache.put_member("s1", &key, &results(), 1).unwrap();
        let names: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();

        // Act
        let got = cache.get_member(&key, &names, 2).unwrap().unwrap();

        // Assert
        assert_eq!(got.0["a"], results().0["a"]);
        assert_eq!(got.0["c"].value, MemberValue::FullRate);
        assert_eq!(got.0["d"], results().0["d"]);
        match &got.0["b"].value {
            MemberValue::Laps { laps, values } => {
                assert_eq!(laps, &vec![2, 3]);
                assert!(values[1].is_nan());
            }
            other => panic!("{other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn result_cache_one_name_absent_partial_hit_is_a_miss() {
        // Arrange
        let dir = temp_dir();
        let mut cache = EvalCache::open(&dir).unwrap();
        cache.put_member("s1", &[1u8; 32], &results(), 1).unwrap();

        // Act
        let got = cache.get_member(&[1u8; 32], &["a".to_string(), "zz".to_string()], 2).unwrap();

        // Assert
        assert_eq!(got, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn result_cache_prune_over_the_limit_evicts_least_recently_used_first() {
        // Arrange
        let dir = temp_dir();
        let mut cache = EvalCache::open(&dir).unwrap();
        cache.put_member("old", &[1u8; 32], &results(), 1).unwrap();
        cache.put_member("new", &[2u8; 32], &results(), 100).unwrap();

        // Act
        let deleted = cache.prune(500).unwrap();

        // Assert
        assert!(deleted > 0);
        assert!(cache.get_member(&[2u8; 32], &["a".to_string()], 101).unwrap().is_some());
        assert!(cache.get_member(&[1u8; 32], &["a".to_string()], 101).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
