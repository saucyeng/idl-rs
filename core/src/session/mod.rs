//! Parser-output data model for `.idl0` sessions.
//!
//! Mirrors the fields the Dart `BinaryParser` produces (see
//! `app/lib/data/session_model.dart` `Session` / `ChannelData`). Only the
//! parser-populated fields are modelled here — `bikeProfileSnapshot` and `laps`
//! in the Dart `Session` come from the `.idl0w` workspace, not the parser, and
//! are out of scope for the engine's parse output (roadmap Phase 5).

pub mod column;
pub mod filename;
pub mod handle;
pub mod seam_correction;
pub mod synthesis;
pub mod time_map;

pub use column::RawColumn;
pub use seam_correction::{ImportWarning, ImportWarningKind};

use std::fmt;

use serde::{Deserialize, Serialize};

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

/// A contiguous run of synthesized samples on a reconciled grid.
///
/// `start` is the grid-slot index of the first synthesized sample; `len` is the
/// run length in slots. Produced by IMU drop reconciliation (SPEC §15.2): each
/// run is either a linearly-interpolated interior fill (a real dropped-sample
/// event) or a held-edge leading/trailing pad, computed on the burst-seam-corrected
/// grid (contract C1 §3.3) rather than the nominal one. Coordinates are grid
/// slots — the same as a channel's sample indices — and a run is **shared
/// across an IMU's six axes** (the same drops affect every axis).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapSpan {
    /// Grid-slot index of the first synthesized sample in the run.
    pub start: usize,
    /// Number of consecutive synthesized samples.
    pub len: usize,
}

/// Which importer produced a [`Session`]. Serializes to the `source_format`
/// `data.parquet` file-metadata string (contract C1 §4.3) lowercase, verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// `.idl0` binary log from an IDL0 device.
    Idl0,
    /// Garmin/Wahoo/etc. `.fit` activity file.
    Fit,
    /// Garmin Connect/Strava-style `.gpx` track.
    Gpx,
    /// Generic `.csv` import (low priority — design doc D4).
    Csv,
}

impl SourceFormat {
    /// The lowercase wire token this variant serializes to (contract C1 §4.3
    /// `source_format` file-metadata value). `const fn` so `core::import`'s
    /// `IMPORTER_TABLE` (R51 Q2) can call it inside a `const` initializer —
    /// a `match` over a `Copy` enum returning `&'static str` is
    /// const-fn-eligible and this change is behaviour-preserving for every
    /// existing call site.
    pub const fn as_str(&self) -> &'static str {
        match self {
            SourceFormat::Idl0 => "idl0",
            SourceFormat::Fit => "fit",
            SourceFormat::Gpx => "gpx",
            SourceFormat::Csv => "csv",
        }
    }
}

/// Provenance of a [`Session`]'s `timestamp_utc_ms` (contract C1 §2). Only
/// `User` makes a reader prefer `session.json`'s `timestamp_utc_ms` over
/// `data.parquet`'s (see [`crate::store::session_json::effective_start_ms`]);
/// every other variant means the parquet value is the truth. Never written
/// into `data.parquet` file metadata — it lives only in `session.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampSource {
    /// Recovered from the `.idl0` file header's `session_start_ms` field.
    Header,
    /// Header value was `0`/missing and the start was recovered from the
    /// first GPS fix instead (SPEC §5.6 back-fill).
    GpsBackfill,
    /// Taken from a non-`.idl0` source file's own timestamp (FIT/GPX/CSV).
    SourceFile,
    /// Set explicitly by the user via `set_session_start`.
    User,
}

impl TimestampSource {
    /// The lowercase wire token this variant serializes to in `session.json`
    /// (contract C1 §6). `const fn` for parity with [`SourceFormat::as_str`].
    pub const fn as_str(&self) -> &'static str {
        match self {
            TimestampSource::Header => "header",
            TimestampSource::GpsBackfill => "gps_backfill",
            TimestampSource::SourceFile => "source_file",
            TimestampSource::User => "user",
        }
    }
}

/// Time-series data for a single sensor channel within a session. See §15.2
/// and contract C1 §2/§3.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// Registry name for this channel, e.g. `IMU0_AccelZ` or `WheelFront`.
    pub channel_id: String,
    /// Per-sample time, **microseconds since the session's first sample**
    /// (contract C1 §3.1). One entry per sample in `column`;
    /// `t_us.len() == column.len()`. Strictly increasing (C1 §3.5 invariant
    /// 1) — traces back to a recorded device/GPS/FIT/GPX timestamp, verbatim
    /// or burst-corrected, **never** `i / nominal_rate_hz` (invariant 4),
    /// except the documented synthesis exception on `Time`/`Distance` (see
    /// `session::synthesis`'s doc comment).
    pub t_us: Vec<i64>,
    /// Verbatim recorded time, **before** burst-seam correction (contract
    /// C1 §3.2's `<source>_t_recorded_us`), same length/units/origin as
    /// `t_us`. `None` when identical to `t_us` element-for-element by
    /// construction (every non-IMU source, and any IMU channel session
    /// with no burst correction applied yet) — avoids duplicating identical
    /// data in RAM; `Some` only once §3.3 correction has actually moved a
    /// value (Task 6). **This field is not explicit in C1 §2's struct
    /// listing** — C1 §1 states the module layout implementing §2–§7 is
    /// L1's own call, and C1 §4.1 requires both `t` (corrected) and
    /// `_t_recorded_us` (verbatim) as separate, independently-readable
    /// `data.parquet` columns; once correction diverges them for IMU
    /// sources, the in-memory model needs to carry both, or the verbatim
    /// value is unrecoverable at Parquet-write time. See this plan's Open
    /// questions.
    pub t_recorded_us: Option<Vec<i64>>,
    /// Nominal sample rate in Hz — **metadata only**, never used to derive a
    /// sample's time. `0.0` for event-driven channels.
    pub nominal_rate_hz: f64,
    /// Compact, typed sample storage. Physical f64 is materialized on demand
    /// via [`Channel::materialize`].
    pub column: RawColumn,
    /// Which recorded-timestamp source this channel's `t_us` was derived
    /// from — one of the `source_kind` tokens contract C1 §4.2 enumerates
    /// (`imu0`, `imu1`, `imu2`, `gps`, `wheel_front`, `wheel_rear`,
    /// `pressure_front`, `pressure_rear`, `hr_bpm`, `hr_rr`, `fit`, `gpx`),
    /// or `"synthesized"` for the engine's own `Time`/`Distance` channels
    /// (not one of C1's wire `source_kind` tokens — those never round-trip
    /// through `data.parquet` at all, so there is no metadata-key collision
    /// to avoid; see C1 §2's `RawColumn` round-trip table).
    pub source_kind: String,
    /// Physical unit string (contract C1 §4.1's per-channel `unit` values,
    /// e.g. `g`, `dps`, `km/h`, `deg`, `pulse`, `bar`, `bpm`). **Not
    /// explicit in C1 §2's struct listing** — C1 §4.2 mandates a `unit`
    /// column-metadata value on every `data.parquet` channel column
    /// *always*, and the only place that value already exists today is the
    /// registry's `ChannelRegistryEntry.units` (currently read at parse
    /// time and discarded); this field carries it forward. Empty string for
    /// engine-synthesized channels (`Time`: `s`, `Distance`: `m` — set
    /// explicitly, not left empty, since both have an unambiguous physical
    /// unit) and for any channel this session has no unit information for.
    /// See this plan's Open questions.
    pub unit: String,
    /// Synthesized-sample runs from drop reconciliation (SPEC §15.2),
    /// unchanged semantics from the pre-idl1 engine: empty for every channel
    /// with no drops.
    pub gaps: Vec<GapSpan>,
}

impl Channel {
    /// Construct a channel from physical f64 samples with **synthetic
    /// uniform** per-sample time (`t_us[i] = round(i * 1e6 / rate)` for
    /// `rate > 0`, all-zero for `rate == 0` with an explicit `t_us` override
    /// via [`Channel::from_f64_with_times`]).
    ///
    /// **Scoped, documented exception to C1 §3.5 invariant 4:** this
    /// constructor is for the interior-mutable derived-channel store
    /// (`SessionHandle`'s math-output and lap-slice entries) and test
    /// fixtures — ephemeral, in-process channels that are never written to
    /// `data.parquet` and are not subject to the *imported/canonical-file*
    /// time invariant, which governs what a session's *persisted* channels
    /// may claim about recorded time. A math-channel output computed
    /// pointwise from a real channel should prefer
    /// [`Channel::from_f64_with_times`] with the source's own `t_us` so its
    /// samples stay aligned to real recorded time; `from_f64` remains for
    /// callers (today: `SessionHandle::store_math`'s legacy call sites,
    /// synthesized test channels) where no source `t_us` is at hand. See
    /// this plan's Open questions for the reasoning.
    pub fn from_f64(channel_id: impl Into<String>, nominal_rate_hz: f64, samples: Vec<f64>) -> Self {
        let t_us = if nominal_rate_hz > 0.0 {
            (0..samples.len())
                .map(|i| (i as f64 * 1_000_000.0 / nominal_rate_hz).round() as i64)
                .collect()
        } else {
            vec![0; samples.len()]
        };
        Channel {
            channel_id: channel_id.into(),
            t_us,
            t_recorded_us: None,
            nominal_rate_hz,
            column: RawColumn::F64(samples),
            source_kind: "synthesized".to_string(),
            unit: String::new(),
            gaps: Vec::new(),
        }
    }

    /// Construct a channel from physical f64 samples with explicit,
    /// caller-supplied `t_us` (µs since the session's first sample) — the
    /// path for event-driven and imported channels, and for math outputs
    /// that carry a real source's per-sample time forward. `t_us.len()` must
    /// equal `samples.len()`; not enforced here (a length mismatch degrades
    /// gracefully — [`Channel::len`] reads `column.len()`, so extra/missing
    /// `t_us` entries are simply unreachable/absent rather than panicking,
    /// CLAUDE.md §5).
    pub fn from_f64_with_times(
        channel_id: impl Into<String>,
        nominal_rate_hz: f64,
        samples: Vec<f64>,
        t_us: Vec<i64>,
        source_kind: impl Into<String>,
    ) -> Self {
        Channel {
            channel_id: channel_id.into(),
            t_us,
            t_recorded_us: None,
            nominal_rate_hz,
            column: RawColumn::F64(samples),
            source_kind: source_kind.into(),
            unit: String::new(),
            gaps: Vec::new(),
        }
    }

    /// Like [`Channel::from_f64_with_times`], but with an explicit,
    /// possibly-different `t_recorded_us` (post burst-seam-correction
    /// construction path — Task 6).
    pub fn from_f64_corrected(
        channel_id: impl Into<String>,
        nominal_rate_hz: f64,
        samples: Vec<f64>,
        t_us: Vec<i64>,
        t_recorded_us: Vec<i64>,
        source_kind: impl Into<String>,
    ) -> Self {
        Channel {
            channel_id: channel_id.into(),
            t_us,
            t_recorded_us: Some(t_recorded_us),
            nominal_rate_hz,
            column: RawColumn::F64(samples),
            source_kind: source_kind.into(),
            unit: String::new(),
            gaps: Vec::new(),
        }
    }

    /// Sets `unit` (builder-style, since every other constructor defaults it
    /// to empty — most callers that care about a real unit are the parser,
    /// which knows it only after construction, from the channel registry).
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }

    /// The verbatim recorded time for this channel (contract C1 §3.2) —
    /// `t_recorded_us` when correction actually diverged it, else `t_us`
    /// itself (the two are identical by construction whenever
    /// `t_recorded_us` is `None`).
    pub fn t_recorded_us_or_t_us(&self) -> &[i64] {
        self.t_recorded_us.as_deref().unwrap_or(&self.t_us)
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

    /// Duration of this channel's own data span, in milliseconds:
    /// `(t_us.last() - t_us.first()) / 1000`, rounded. `0` when the channel
    /// has fewer than 2 samples. **Changed from the pre-idl1 formula**
    /// (`len / sample_rate_hz × 1000` for fixed-rate, last event time for
    /// event-driven) — `t_us` is now mandatory and session-relative (C1
    /// §3.1), so both cases collapse into one formula that reads real
    /// recorded/corrected time instead of assuming a rate.
    pub fn duration_ms(&self) -> i64 {
        match (self.t_us.first(), self.t_us.last()) {
            (Some(&first), Some(&last)) if last > first => {
                ((last - first) as f64 / 1000.0).round() as i64
            }
            _ => 0,
        }
    }
}

/// In-memory representation of one imported session — the parsed/converted
/// view of one immutable source blob (contract C1 §2).
#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    /// Stable identity. The device's session UUID (32-char lowercase hex)
    /// for `.idl0` sources; a prefix of `blob_sha256` for FIT/GPX/CSV
    /// sources (exact derivation: contract C4 §3).
    pub session_id: String,
    /// 12-char lowercase hex MAC-derived device id. `None` for FIT/GPX/CSV
    /// — there is no device.
    pub device_id: Option<String>,
    /// Session start, UTC milliseconds since the Unix epoch. `0` means
    /// "unknown" (SPEC §5.1 sentinel convention, unchanged).
    pub timestamp_utc_ms: i64,
    /// Where `timestamp_utc_ms` came from (contract C1 §2, ruling R194):
    /// `Header` (`.idl0` header `Session start UTC`), `GpsBackfill` (§3.1's
    /// first-fix formula), `SourceFile` (FIT/GPX/CSV earliest converted
    /// instant), or `User` (`set_session_start`, C3 §3.3). Set by each
    /// importer and carried in `session.json` — deliberately never written
    /// into `data.parquet` file metadata, so no parquet is invalidated by
    /// its introduction.
    pub timestamp_source: TimestampSource,
    /// CRC32 of `idl0_config.json` at recording time, 8-char lowercase hex.
    /// `None` for FIT/GPX/CSV — there is no device config.
    pub config_checksum: Option<String>,
    /// Which importer produced this session.
    pub source_format: SourceFormat,
    /// SHA-256 of the raw source file bytes exactly as imported, 64
    /// lowercase hex chars — the CAS blob this session's `data.parquet` is
    /// a function of (design doc §5).
    pub blob_sha256: String,
    /// Parsed channel data, one entry per channel present in this session.
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
    /// Non-fatal import-time anomalies collected during parsing — today,
    /// exclusively burst-seam correction's fallback cases (contract C1
    /// §3.3; [`crate::session::seam_correction::correct_burst_seams`]).
    /// Empty for a clean parse. Never silently dropped (CLAUDE.md §5) — a
    /// caller that discards `ParseResult` without reading this is the one
    /// place a warning could still go unseen; every constructor of a
    /// [`SessionHandle`](crate::session::handle::SessionHandle) threads it
    /// through instead.
    pub import_warnings: Vec<ImportWarning>,
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
    fn duration_ms_uses_first_and_last_t_us_span() {
        // Arrange — 800 samples at 800 Hz, t_us spanning exactly 1_000_000 µs.
        let ch = Channel::from_f64("IMU0_AccelX", 800.0, vec![0.0; 800]);

        // Act
        let ms = ch.duration_ms();

        // Assert
        assert_eq!(ms, 999); // (799/800)*1e6 µs span, rounded
    }

    #[test]
    fn duration_ms_event_driven_channel_uses_t_us_span() {
        // Arrange — event channel, t_us at 0.5s, 1.0s, 1.3s → span 800ms.
        let ch = Channel::from_f64_with_times(
            "HR_RR", 0.0, vec![1000.0, 900.0, 850.0],
            vec![500_000, 1_000_000, 1_300_000], "hr_rr",
        );

        // Act
        let ms = ch.duration_ms();

        // Assert
        assert_eq!(ms, 800);
    }

    #[test]
    fn duration_ms_single_sample_is_zero() {
        // Arrange
        let ch = Channel::from_f64("HR_RR", 0.0, vec![1.0]);

        // Act + Assert
        assert_eq!(ch.duration_ms(), 0);
    }

    #[test]
    fn is_complete_reflects_truncation_warning() {
        // Arrange
        let session = Session {
            session_id: String::new(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::Header,
            config_checksum: None,
            source_format: SourceFormat::Idl0,
            blob_sha256: String::new(),
            channels: Vec::new(),
        };

        // Act + Assert
        let clean = ParseResult {
            session: session.clone(),
            truncation_warning: None,
            import_warnings: Vec::new(),
        };
        assert!(clean.is_complete());
        let partial = ParseResult {
            session,
            truncation_warning: Some(ParseError::TruncatedRecord("eof".to_string())),
            import_warnings: Vec::new(),
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
