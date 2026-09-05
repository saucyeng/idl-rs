//! Session loading shared by every workbook command that needs a channel
//! lookup or a lap context (`eval_workbook`, and the tasks after it — C3
//! §3.4). A session is bound by id, not by front matter (C2 has none), so
//! this module is the one place that turns a `session_id` string into a
//! [`idl_rs::session::SessionHandle`] or a [`MathLapContext`].

use std::path::{Path, PathBuf};

use idl_rs::math::MathLapContext;
use idl_rs::session::handle::SessionHandle;
use idl_rs::session::synthesis::synthesize_base_channels;
use idl_rs::session::Session;
use idl_rs::store::parquet::read_session_parquet;
use idl_rs::store::session_json::read_session_json;

use crate::error::{IpcError, IpcErrorKind};

/// `<data>/sessions/<session_id>` (C4 §2).
pub fn session_dir(data_dir: &Path, session_id: &str) -> PathBuf {
    data_dir.join("sessions").join(session_id)
}

// TODO(idl0): every call re-reads the whole `data.parquet` from disk — a
// session cache is design §4's recorded deferral, not this task's.
/// Reads `session_id`'s `data.parquet` back into a [`Session`], running
/// [`synthesize_base_channels`] before returning (`read_session_parquet`
/// does not reconstruct `Time`/`Distance` itself — its own doc comment says
/// so). A missing session directory or `data.parquet` is
/// [`IpcErrorKind::NotFound`] (the id is named in the message); a parquet
/// read failure is [`IpcErrorKind::Io`].
pub fn load_session(data_dir: &Path, session_id: &str) -> Result<Session, IpcError> {
    let parquet_path = session_dir(data_dir, session_id).join("data.parquet");
    if !parquet_path.exists() {
        return Err(IpcError::new(IpcErrorKind::NotFound, format!("session '{session_id}' not found")));
    }
    let mut session = read_session_parquet(&parquet_path)
        .map_err(|e| IpcError::new(IpcErrorKind::Io, format!("reading {}: {}", parquet_path.display(), e.message)))?;
    synthesize_base_channels(&mut session);
    Ok(session)
}

/// [`load_session`] wrapped in a [`SessionHandle`] via [`SessionHandle::
/// from_session`] (ledger R40 — carries `source_format`/`blob_sha256`/
/// per-channel `unit` through, unlike `from_channels`).
pub fn load_session_handle(data_dir: &Path, session_id: &str) -> Result<SessionHandle, IpcError> {
    let session = load_session(data_dir, session_id)?;
    Ok(SessionHandle::from_session(session))
}

/// Builds a [`MathLapContext`] from `session_id`'s `session.json` `laps[]`
/// (`start_time_secs`/`end_time_secs` per lap → `main_lap_bounds`;
/// `main_lap_number` copied verbatim). `session.json` absent or unreadable
/// is not an error here — it returns [`MathLapContext::empty`] (C2 §3.5.B's
/// `NoLapContext` is the evaluator's own answer for "no lap context", not
/// this function's job to reject on).
pub fn load_lap_context(data_dir: &Path, session_id: &str) -> MathLapContext {
    let path = session_dir(data_dir, session_id).join("session.json");
    let Ok(doc) = read_session_json(&path) else {
        return MathLapContext::empty();
    };
    MathLapContext {
        main_lap_bounds: doc.laps.iter().map(|l| (l.start_time_secs, l.end_time_secs)).collect(),
        main_lap_number: doc.main_lap_number,
        ..MathLapContext::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use idl_rs::session::{Channel, RawColumn, SourceFormat};
    use idl_rs::store::parquet::write_session_parquet;
    use idl_rs::store::session_json::{empty_session_json, write_session_json, LapJson};
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-session-source-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seed_session(root: &Path, session_id: &str) {
        let session = Session {
            session_id: session_id.to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Speed".to_string(),
                t_us: vec![0, 500_000],
                t_recorded_us: None,
                nominal_rate_hz: 2.0,
                column: RawColumn::F64(vec![1.0, 2.0]),
                source_kind: "gps".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    #[test]
    fn load_session_unknown_id_not_found() {
        // Arrange
        let root = temp_root();

        // Act
        let err = load_session(&root, "nope").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_session_seeded_session_channels_and_units_survive_the_round_trip() {
        // Arrange
        let root = temp_root();
        seed_session(&root, "s1");

        // Act
        let session = load_session(&root, "s1").unwrap();

        // Assert
        assert_eq!(session.source_format, SourceFormat::Fit);
        assert_eq!(session.blob_sha256, "a".repeat(64));
        let speed = session.channels.iter().find(|c| c.channel_id == "Speed").unwrap();
        assert_eq!(speed.unit, "m/s");
        assert_eq!(speed.column, RawColumn::F64(vec![1.0, 2.0]));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_session_json_with_two_laps_two_bounds() {
        // Arrange
        let root = temp_root();
        let mut doc = empty_session_json("s1");
        doc.main_lap_number = Some(2);
        doc.laps = vec![
            LapJson {
                lap_number: 1,
                start_timestamp_ms: 0,
                end_timestamp_ms: 1_000,
                raw_elapsed_ms: 1_000,
                lap_time_ms: 1_000,
                start_time_secs: 0.0,
                end_time_secs: 1.0,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
            LapJson {
                lap_number: 2,
                start_timestamp_ms: 1_000,
                end_timestamp_ms: 2_500,
                raw_elapsed_ms: 1_500,
                lap_time_ms: 1_500,
                start_time_secs: 1.0,
                end_time_secs: 2.5,
                sectors: Vec::new(),
                neutral_zone_visits: Vec::new(),
            },
        ];
        write_session_json(&root, "s1", &doc, None).unwrap();

        // Act
        let ctx = load_lap_context(&root, "s1");

        // Assert
        assert_eq!(ctx.main_lap_bounds, vec![(0.0, 1.0), (1.0, 2.5)]);
        assert_eq!(ctx.main_lap_number, Some(2));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn load_lap_context_no_session_json_empty_context() {
        // Arrange
        let root = temp_root();

        // Act
        let ctx = load_lap_context(&root, "nope");

        // Assert
        assert!(ctx.main_lap_bounds.is_empty());
        assert_eq!(ctx.main_lap_number, None);

        let _ = std::fs::remove_dir_all(&root);
    }
}
