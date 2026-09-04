//! Encode side of `.idl0t` (contract C4 §2) — the inventory's noted gap
//! ("encode missing in engine"). Serialises through the same private
//! `TrackDto` (and nested DTOs) that `track_artifact::read` already reads
//! (`track_artifact::model`'s `From<&Track> for TrackArtifact`), so the
//! reader stays the single authority on the `.idl0t` wire shape in both
//! directions.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::store::atomic::{sha256_hex, write_atomic_with_retry, AtomicWriteError};
use crate::track_artifact::model::{Track, TrackArtifact};

/// Discriminant for [`TrackWriteError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackWriteErrorKind {
    /// A filesystem operation failed (including exhausting C4 §4's
    /// bounded concurrency-retry).
    Io,
    /// The track did not encode to JSON.
    Encode,
}

/// Error from [`write_track`]. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackWriteError {
    pub kind: TrackWriteErrorKind,
    pub message: String,
}
impl fmt::Display for TrackWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for TrackWriteError {}
impl From<AtomicWriteError> for TrackWriteError {
    fn from(e: AtomicWriteError) -> Self {
        TrackWriteError { kind: TrackWriteErrorKind::Io, message: e.to_string() }
    }
}

/// Serialises `track` to the `.idl0t` wire JSON shape (mirroring
/// [`read_track`](crate::track_artifact::read::read_track) exactly, via the
/// shared `TrackDto`) and writes it atomically to
/// `<data_root>/tracks/<track_id>.idl0t`, replacing any existing file for
/// the same id (last-write-wins, consistent with C4 §6's LWW-by-
/// `updated_at_ms`) via [`write_atomic_with_retry`] (C4 §4 step 4).
/// `track.created_at_ms`/`updated_at_ms` are written verbatim — the caller
/// owns picking real timestamps.
pub fn write_track(data_root: &Path, track: &Track) -> Result<PathBuf, TrackWriteError> {
    let artifact = TrackArtifact::from(track);
    let bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|e| TrackWriteError { kind: TrackWriteErrorKind::Encode, message: e.to_string() })?;
    let target = data_root.join("tracks").join(format!("{}.idl0t", track.id));
    let based_on = std::fs::read(&target).ok().map(|b| sha256_hex(&b));
    write_atomic_with_retry(data_root, &target, &bytes, based_on.as_deref(), |_current| bytes.clone())?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gps::GpsFix;
    use crate::laps::{Gate, LapTiming, NeutralZone, SectorGate};
    use crate::track_artifact::read::read_track;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_track() -> Track {
        Track {
            id: "t-1".to_string(),
            name: "A-Line".to_string(),
            venue: "Whistler".to_string(),
            timing: Some(LapTiming::Circuit {
                start_finish: Gate { lat1: 1.0, lon1: 2.0, lat2: 3.0, lon2: 4.0 },
            }),
            sector_gates: vec![SectorGate {
                name: "S1".to_string(),
                gate: Gate { lat1: 5.0, lon1: 6.0, lat2: 7.0, lon2: 8.0 },
            }],
            neutral_zones: vec![NeutralZone {
                name: "Pit".to_string(),
                enter: Gate { lat1: 9.0, lon1: 10.0, lat2: 11.0, lon2: 12.0 },
                exit: Gate { lat1: 13.0, lon1: 14.0, lat2: 15.0, lon2: 16.0 },
            }],
            reference_polyline: vec![GpsFix { timestamp_ms: 1000, lat: 501163000.0, lon: -1229574000.0 }],
            created_at_ms: 111,
            updated_at_ms: 222,
        }
    }

    #[test]
    fn write_then_read_round_trips_every_field() {
        // Arrange
        let root = temp_root();
        let track = sample_track();

        // Act
        let path = write_track(&root, &track).unwrap();
        let back = read_track(&path).unwrap();

        // Assert
        assert_eq!(back.id, track.id);
        assert_eq!(back.name, track.name);
        assert_eq!(back.venue, track.venue);
        assert_eq!(back.created_at_ms, track.created_at_ms);
        assert_eq!(back.updated_at_ms, track.updated_at_ms);
        match (&back.timing, &track.timing) {
            (Some(LapTiming::Circuit { start_finish: a }), Some(LapTiming::Circuit { start_finish: b })) => {
                assert_eq!(a, b);
            }
            other => panic!("expected circuit timing to round-trip, got {other:?}"),
        }
        assert_eq!(back.sector_gates.len(), 1);
        assert_eq!(back.sector_gates[0].name, track.sector_gates[0].name);
        assert_eq!(back.sector_gates[0].gate, track.sector_gates[0].gate);
        assert_eq!(back.neutral_zones.len(), 1);
        assert_eq!(back.neutral_zones[0].name, track.neutral_zones[0].name);
        assert_eq!(back.neutral_zones[0].enter, track.neutral_zones[0].enter);
        assert_eq!(back.neutral_zones[0].exit, track.neutral_zones[0].exit);
        assert_eq!(back.reference_polyline, track.reference_polyline);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_track_overwrites_an_existing_file_with_new_values() {
        // Arrange
        let root = temp_root();
        let mut track = sample_track();
        write_track(&root, &track).unwrap();

        // Act
        track.name = "B-Line".to_string();
        track.updated_at_ms = 999;
        write_track(&root, &track).unwrap();
        let back = read_track(&root.join("tracks").join(format!("{}.idl0t", track.id))).unwrap();

        // Assert
        assert_eq!(back.name, "B-Line");
        assert_eq!(back.updated_at_ms, 999);

        let _ = std::fs::remove_dir_all(&root);
    }
}
