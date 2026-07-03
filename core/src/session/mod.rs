//! Parser-output data model for `.idl0` sessions.
//!
//! Mirrors the fields the Dart `BinaryParser` produces (see
//! `app/lib/data/session_model.dart` `Session` / `ChannelData`). Only the
//! parser-populated fields are modelled here — `bikeProfileSnapshot` and `laps`
//! in the Dart `Session` come from the `.idl0w` workspace, not the parser, and
//! are out of scope for the engine's parse output (roadmap Phase 5).

pub mod column;
pub mod handle;
pub mod synthesis;

pub use column::RawColumn;

use std::fmt;

/// Errors raised while parsing an `.idl0` binary log.
///
/// Mirrors the Dart exception hierarchy in `app/lib/data/exceptions.dart`.
/// [`ParseError::TruncatedRecord`] is *recoverable*: the parser returns the
/// data read before the truncation point in [`ParseResult::session`] and
/// surfaces the error via [`ParseResult::truncation_warning`] rather than
/// failing the whole parse (CLAUDE.md §5 — recover what's readable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Magic bytes were neither `ESPL` (v1) nor `IDL0` (v2/v3).
    InvalidMagicBytes(String),
    /// Magic was `IDL0` but the schema version byte is unsupported.
    UnsupportedSchemaVersion(String),
    /// The buffer ended before a record/header field could be fully read.
    TruncatedRecord(String),
    /// The file could not be read from disk (path entry point only).
    Io(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidMagicBytes(m) => write!(f, "InvalidMagicBytes: {m}"),
            ParseError::UnsupportedSchemaVersion(m) => {
                write!(f, "UnsupportedSchemaVersion: {m}")
            }
            ParseError::TruncatedRecord(m) => write!(f, "TruncatedRecord: {m}"),
            ParseError::Io(m) => write!(f, "Io: {m}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// A channel declared in the file header registry. See IDL0_SPEC §5.2.
///
/// v2 files (32-byte entries) have `scale` `1.0` and `offset` `0.0` by
/// convention — those fields are not on the wire and are filled in by the v2
/// registry reader so the `physical = stored × scale + offset` formula works
/// for both schema versions. v3 files (40-byte entries) carry explicit
/// `scale`/`offset` per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelRegistryEntry {
    /// Unique channel ID within this session, referenced by 0x03 records.
    pub channel_id: u8,
    /// Data type code: 0=u8 1=u16 2=u32 3=i8 4=i16 5=i32 6=f32 7=f64.
    pub data_type: u8,
    /// Nominal sample rate in Hz. 0 = event-driven.
    pub sample_rate_hz: u16,
    /// Scale factor applied to the raw stored value (`physical = stored × scale + offset`).
    pub scale: f64,
    /// Offset added after scaling.
    pub offset: f64,
    /// Null-terminated ASCII channel name, e.g. `IMU0_AccelX`.
    pub name: String,
    /// Null-terminated ASCII unit string, e.g. `g`, `dps`, `pulse`.
    pub units: String,
}

/// A contiguous run of synthesized samples on a reconciled IMU grid.
///
/// `start` is the grid-slot index of the first synthesized sample; `len` is the
/// run length in slots. Produced by IMU drop reconciliation (§15): each run is
/// either a linearly-interpolated interior fill (a real dropped-sample event) or
/// a held-edge leading/trailing pad. Coordinates are grid slots — the same as a
/// channel's sample indices — and a run is **shared across an IMU's six axes**
/// (the same drops affect every axis). It is the honest record of every region
/// the parser filled to put all IMUs on one nominal grid. No consumers ship in
/// the change that introduced it (see the drop-reconciliation design §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapSpan {
    /// Grid-slot index of the first synthesized sample in the run.
    pub start: usize,
    /// Number of consecutive synthesized samples.
    pub len: usize,
}

/// Time-series data for a single sensor channel within a session. See §15.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// Registry name for this channel, e.g. `IMU0_AccelZ` or `WheelFront`.
    pub channel_id: String,
    /// Nominal sample rate in Hz. 0 indicates event-driven (variable rate).
    pub sample_rate_hz: f64,
    /// Compact, typed sample storage. Physical f64 is materialized on demand via
    /// [`Channel::materialize`] (registry `scale`/`offset` applied lazily; GPS,
    /// synthesized, and math channels are stored verbatim f64). See the
    /// compact-raw-storage design spec.
    pub column: RawColumn,
    /// Per-sample times in seconds for event-driven channels (`sample_rate_hz == 0`),
    /// relative to session t=0 (earliest record `timestamp_us`). `None` for
    /// fixed-rate channels, whose sample `i` is implicitly at `i / sample_rate_hz`.
    pub sample_times_secs: Option<Vec<f64>>,
    /// Synthesized-sample runs from IMU drop reconciliation (§15), in grid-slot
    /// coordinates. Empty for every non-IMU channel and for any IMU channel with
    /// no drops. Shared across an IMU's six axes. Recorded, not yet consumed.
    pub gaps: Vec<GapSpan>,
}

impl Channel {
    /// Construct a channel from physical f64 samples (verbatim `RawColumn::F64`).
    /// The construction path for synthesized/GPX/math channels and tests.
    pub fn from_f64(
        channel_id: impl Into<String>,
        sample_rate_hz: f64,
        samples: Vec<f64>,
        sample_times_secs: Option<Vec<f64>>,
    ) -> Self {
        Channel {
            channel_id: channel_id.into(),
            sample_rate_hz,
            column: RawColumn::F64(samples),
            sample_times_secs,
            gaps: Vec::new(),
        }
    }

    /// Number of samples in the channel.
    pub fn len(&self) -> usize {
        self.column.len()
    }

    /// `true` when the channel holds no samples.
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }

    /// Widen all samples to physical f64 (transient — never resident).
    pub fn materialize(&self) -> Vec<f64> {
        self.column.materialize()
    }

    /// Widen the half-open index window `[start, end)` to physical f64, clamped.
    pub fn materialize_range(&self, start: usize, end: usize) -> Vec<f64> {
        self.column.materialize_range(start, end)
    }

    /// Physical value at index `i`, or `None` if out of range.
    pub fn value_at(&self, i: usize) -> Option<f64> {
        self.column.value_at(i)
    }

    /// Finite (min, max) of the physical samples; `None` when empty/all-non-finite.
    pub fn min_max(&self) -> Option<(f64, f64)> {
        self.column.min_max()
    }

    /// Duration of this channel's data in milliseconds.
    ///
    /// Fixed-rate: `len / sample_rate_hz × 1000`. Event-driven
    /// (`sample_rate_hz == 0`): last `sample_times_secs` entry in ms, or 0
    /// when no per-sample times are available. Matches Dart `ChannelData.durationMs`.
    pub fn duration_ms(&self) -> i64 {
        if self.sample_rate_hz == 0.0 {
            match &self.sample_times_secs {
                Some(times) if !times.is_empty() => (times[times.len() - 1] * 1000.0).round() as i64,
                _ => 0,
            }
        } else {
            ((self.len() as f64 / self.sample_rate_hz) * 1000.0).round() as i64
        }
    }
}

/// In-memory representation of a parsed `.idl0` session. The `.idl0` file is the
/// source of truth; this is the parsed view. Lap/sector/workspace data lives in
/// the companion `.idl0w` file (Phase 5) and is not part of the parse output.
#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    /// UUID matching the session header, 32-char lowercase hex (empty for v1).
    pub session_id: String,
    /// Device ID, 12-char lowercase hex (empty for v1).
    pub device_id: String,
    /// Session start in UTC milliseconds, GPS-anchored when available.
    pub timestamp_utc_ms: i64,
    /// CRC32 of `idl0_config.json` at recording time, 8-char lowercase hex (empty for v1).
    pub config_checksum: String,
    /// Parsed channel data, one entry per enabled channel, in first-seen order.
    pub channels: Vec<Channel>,
}

/// Result of parsing an `.idl0` buffer.
///
/// Always contains a valid [`Session`]. When the file was truncated mid-record,
/// [`truncation_warning`](ParseResult::truncation_warning) is `Some` and the
/// session holds all data parsed before the truncation point.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseResult {
    /// The parsed session. May be partial when [`Self::truncation_warning`] is set.
    pub session: Session,
    /// `Some` when the file ended mid-record. Surface as
    /// "Log incomplete — showing data to <timestamp>".
    pub truncation_warning: Option<ParseError>,
}

impl ParseResult {
    /// `true` when the file parsed cleanly with no truncation.
    pub fn is_complete(&self) -> bool {
        self.truncation_warning.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_ms_fixed_rate_channel_uses_sample_count_over_rate() {
        // Arrange — 800 samples at 800 Hz = 1000 ms.
        let ch = Channel::from_f64("IMU0_AccelX", 800.0, vec![0.0; 800], None);

        // Act
        let ms = ch.duration_ms();

        // Assert
        assert_eq!(ms, 1000);
    }

    #[test]
    fn duration_ms_event_driven_channel_uses_last_sample_time() {
        // Arrange — event channel, last sample at 1.3 s → 1300 ms.
        let ch = Channel::from_f64("HR_RR", 0.0, vec![1000.0, 900.0, 850.0], Some(vec![0.5, 1.0, 1.3]));

        // Act
        let ms = ch.duration_ms();

        // Assert
        assert_eq!(ms, 1300);
    }

    #[test]
    fn duration_ms_event_driven_without_times_is_zero() {
        // Arrange
        let ch = Channel::from_f64("HR_RR", 0.0, vec![1.0], None);

        // Act + Assert
        assert_eq!(ch.duration_ms(), 0);
    }

    #[test]
    fn is_complete_reflects_truncation_warning() {
        // Arrange
        let session = Session {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
            channels: Vec::new(),
        };

        // Act + Assert
        let clean = ParseResult { session: session.clone(), truncation_warning: None };
        assert!(clean.is_complete());
        let partial = ParseResult {
            session,
            truncation_warning: Some(ParseError::TruncatedRecord("eof".to_string())),
        };
        assert!(!partial.is_complete());
    }

    #[test]
    fn io_error_displays_with_prefix() {
        // Arrange
        let e = ParseError::Io("no such file".to_string());

        // Act
        let s = format!("{e}");

        // Assert
        assert_eq!(s, "Io: no such file");
    }
}
