//! Owned, parsed-session handle. The bridge wraps this in `RustOpaque`; the CLI
//! uses it directly. Holds the parsed `Session` plus synthesized base channels
//! and serves FFI-shaped views (metadata, channel list, per-channel samples).
//! No decimation tile cache this phase — that moves here in Phase 3 alongside
//! retiring `chart_decimation`'s global registry.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;

use crate::parse::parse;
use crate::session::synthesis::synthesize_base_channels;
use crate::session::{Channel, ParseError, ParseResult, RawColumn, Session};

/// Compact session summary. Mirrors roadmap §11 seam #2 (single summary
/// extractor consumed by the SQLite catalog and any future cloud index).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionMeta {
    /// UUID matching the session header, 32-char lowercase hex (empty for v1).
    pub session_id: String,
    /// Device ID, 12-char lowercase hex (empty for v1).
    pub device_id: String,
    /// Session start in UTC milliseconds, GPS-anchored when available.
    pub timestamp_utc_ms: i64,
    /// CRC32 of `idl0_config.json`, 8-char lowercase hex (empty for v1).
    pub config_checksum: String,
    /// Number of channels (including synthesized `Time`/`Distance`).
    pub channel_count: u32,
    /// Canonical duration = longest channel span in ms (C9). 0 when empty.
    pub duration_ms: i64,
    /// `Some` when the file was truncated mid-record (recoverable).
    pub truncation_warning: Option<String>,
}

/// Per-channel metadata; no samples cross with this.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelMeta {
    /// Registry name, e.g. `IMU0_AccelX`, `GPS_Latitude`, `Time`.
    pub channel_id: String,
    /// Nominal sample rate in Hz. 0 = event-driven.
    pub sample_rate_hz: f64,
    /// Number of samples in the channel.
    pub length: u32,
    /// `true` when `sample_rate_hz == 0` (per-sample times apply).
    pub is_event_driven: bool,
    /// `true` for engine-synthesized channels (`Time`, `Distance`).
    pub synthesized: bool,
}

/// Header fields for [`SessionHandle::from_channels`] (GPX path).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionMetaInput {
    pub session_id: String,
    pub device_id: String,
    pub timestamp_utc_ms: i64,
    pub config_checksum: String,
}

/// One channel of caller-parsed data (GPX path).
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelInput {
    pub channel_id: String,
    pub sample_rate_hz: f64,
    pub samples: Vec<f64>,
    pub sample_times_secs: Option<Vec<f64>>,
}

/// Owned parsed session. Synthesis runs in every constructor. The `derived`
/// store is interior-mutable so the math evaluator's resolver can write
/// resolved dependency outputs back without re-marshalling samples across FFI
/// (Phase 3a, spec §6). Keyed by [`DerivedKey`] so a lap slice can never
/// overwrite a math output or shadow a base channel. Parsed+synthesized
/// `session.channels` stay immutable so the exporter's `channel_data()` borrow
/// is unaffected.
#[derive(Debug)]
pub struct SessionHandle {
    session: Session,
    synthesized_ids: Vec<String>,
    truncation_warning: Option<String>,
    derived: RwLock<HashMap<DerivedKey, Channel>>,
}

impl Clone for SessionHandle {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            synthesized_ids: self.synthesized_ids.clone(),
            truncation_warning: self.truncation_warning.clone(),
            derived: RwLock::new(self.derived.read().unwrap().clone()),
        }
    }
}

/// Lap-comparison role of a lap-windowed slice. Half of a
/// [`DerivedKey::LapSlice`] identity: `Main` is the focused lap, `Overlay` the
/// comparison lap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceRole {
    Main,
    Overlay,
}

/// Identity of a derived (runtime-written) channel in a handle's overlay store.
/// The enum variant is the channel's *kind* — a math-channel output keyed by its
/// workbook name, or a lap-windowed slice keyed by its `(source, role, lap)`
/// triple. Because the two are different variants, a lap slice can never occupy
/// a math output's key (or vice-versa): the cross-kind collision the old flat
/// string store allowed is impossible by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DerivedKey {
    /// A math-channel output, addressed by its workbook name.
    Math(String),
    /// A lap-windowed, sample-0-rebased slice of `source` (a base or math
    /// channel name).
    LapSlice { source: String, role: SliceRole, lap: u32 },
}

/// Reserved control character prefixing every lap-slice storage token. A
/// channel name can never contain it, so slice tokens and math/base names form
/// disjoint string namespaces; the decimation path (`with_channel`) routes on it.
///
/// This is a **reserved-character invariant** assumed of every channel name —
/// firmware base ids, synthesized `Time`/`Distance`, and user math-channel names
/// are all `U+0001`-free (it is a non-printable SOH control char). The invariant
/// is not separately enforced: a pathological name that both contained the
/// sentinel *and* matched the `\u{1}<src>\u{1}<m|o><digits>` token grammar would
/// be written under `Math(name)` but read back via `from_token` as a `LapSlice`
/// key, so that one channel would render empty. That is graceful degradation
/// (one channel blank, no crash, no effect on others — CLAUDE.md §5), reachable
/// only by deliberately malformed input (e.g. a hand-edited `.idl0wb`), never by
/// any typed name.
const SLICE_TOKEN_SENTINEL: char = '\u{1}';

impl DerivedKey {
    /// The opaque storage token: the `channel_id` string this entry is addressed
    /// by on the decimation path. A `Math` token is the plain name; a `LapSlice`
    /// token is `\u{1}<source>\u{1}<m|o><lap>`, so the namespaces never collide.
    fn token(&self) -> String {
        match self {
            DerivedKey::Math(name) => name.clone(),
            DerivedKey::LapSlice { source, role, lap } => {
                let r = match role {
                    SliceRole::Main => 'm',
                    SliceRole::Overlay => 'o',
                };
                format!("{SLICE_TOKEN_SENTINEL}{source}{SLICE_TOKEN_SENTINEL}{r}{lap}")
            }
        }
    }

    /// Parse a storage token back into its key. A token without the sentinel
    /// prefix is a [`DerivedKey::Math`] name (the base/math addressing path); a
    /// sentinel-prefixed token parses to its [`DerivedKey::LapSlice`]. A
    /// malformed sentinel token falls back to `Math` (it simply matches nothing).
    fn from_token(token: &str) -> DerivedKey {
        if let Some(rest) = token.strip_prefix(SLICE_TOKEN_SENTINEL) {
            if let Some((source, role_lap)) = rest.rsplit_once(SLICE_TOKEN_SENTINEL) {
                let mut chars = role_lap.chars();
                if let Some(role_char) = chars.next() {
                    let role = if role_char == 'o' {
                        SliceRole::Overlay
                    } else {
                        SliceRole::Main
                    };
                    if let Ok(lap) = chars.as_str().parse::<u32>() {
                        return DerivedKey::LapSlice {
                            source: source.to_string(),
                            role,
                            lap,
                        };
                    }
                }
            }
        }
        DerivedKey::Math(token.to_string())
    }
}

impl SessionHandle {
    /// Parse `.idl0` bytes, synthesize base channels, and own the result.
    /// Hard failures (`InvalidMagicBytes`/`UnsupportedSchemaVersion`) → `Err`;
    /// truncation is recoverable and surfaced via [`SessionMeta::truncation_warning`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ParseError> {
        let ParseResult { mut session, truncation_warning } = parse(bytes)?;
        let synthesized_ids = synthesize_base_channels(&mut session);
        Ok(Self {
            session,
            synthesized_ids,
            truncation_warning: truncation_warning.map(|e| e.to_string()),
            derived: RwLock::new(HashMap::new()),
        })
    }

    /// Read `.idl0` from disk then [`Self::from_bytes`]. IO errors → [`ParseError::Io`].
    pub fn from_path(path: &str) -> Result<Self, ParseError> {
        let bytes = std::fs::read(path).map_err(|e| ParseError::Io(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Build a handle from caller-parsed channels (GPX). Synthesis still runs.
    pub fn from_channels(meta: SessionMetaInput, channels: Vec<ChannelInput>) -> Self {
        let mut session = Session {
            session_id: meta.session_id,
            device_id: meta.device_id,
            timestamp_utc_ms: meta.timestamp_utc_ms,
            config_checksum: meta.config_checksum,
            channels: channels
                .into_iter()
                .map(|c| Channel {
                    channel_id: c.channel_id,
                    sample_rate_hz: c.sample_rate_hz,
                    column: RawColumn::F64(c.samples),
                    sample_times_secs: c.sample_times_secs,
                    gaps: Vec::new(),
                })
                .collect(),
        };
        let synthesized_ids = synthesize_base_channels(&mut session);
        Self {
            session,
            synthesized_ids,
            truncation_warning: None,
            derived: RwLock::new(HashMap::new()),
        }
    }

    /// Compact summary for the catalog / library list.
    pub fn metadata(&self) -> SessionMeta {
        SessionMeta {
            session_id: self.session.session_id.clone(),
            device_id: self.session.device_id.clone(),
            timestamp_utc_ms: self.session.timestamp_utc_ms,
            config_checksum: self.session.config_checksum.clone(),
            channel_count: self.session.channels.len() as u32,
            duration_ms: self
                .session
                .channels
                .iter()
                .map(|c| c.duration_ms())
                .max()
                .unwrap_or(0),
            truncation_warning: self.truncation_warning.clone(),
        }
    }

    /// Metadata for every channel (no samples).
    pub fn channels(&self) -> Vec<ChannelMeta> {
        self.session
            .channels
            .iter()
            .map(|c| ChannelMeta {
                channel_id: c.channel_id.clone(),
                sample_rate_hz: c.sample_rate_hz,
                length: c.len() as u32,
                is_event_driven: c.sample_rate_hz == 0.0,
                synthesized: self.synthesized_ids.iter().any(|id| id == &c.channel_id),
            })
            .collect()
    }

    /// Samples for `channel_id`, or empty when absent.
    pub fn channel_samples(&self, channel_id: &str) -> Vec<f64> {
        self.session
            .channels
            .iter()
            .find(|c| c.channel_id == channel_id)
            .map(|c| c.materialize())
            .unwrap_or_default()
    }

    /// Event-driven per-sample times for `channel_id`; `None` for fixed-rate or absent.
    pub fn channel_sample_times(&self, channel_id: &str) -> Option<Vec<f64>> {
        self.session
            .channels
            .iter()
            .find(|c| c.channel_id == channel_id)
            .and_then(|c| c.sample_times_secs.clone())
    }

    /// Convert wall-clock epoch-ms timestamps to uniform-Time seconds for this
    /// session.
    ///
    /// When `GPS_EpochMs` is present and fixed-rate, binary-search-interpolate
    /// each value's fractional sample index within the channel and divide by the
    /// GPS rate (values at/below the first epoch → `0.0`; at/above the last →
    /// `(len - 1) / rate`). Otherwise fall back to `(epoch_ms - origin) / 1000`
    /// using the back-filled session origin (`timestamp_utc_ms`). Returns
    /// seconds (grid-independent), one per input, input order preserved.
    ///
    /// `GPS_EpochMs` is assumed monotonic after GPS lock (matching the lap
    /// detector's crossing epochs, which are always post-lock).
    pub fn epoch_ms_to_time_secs(&self, epochs_ms: &[f64]) -> Vec<f64> {
        let gps = self
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_EpochMs")
            .filter(|c| c.sample_rate_hz > 0.0 && !c.is_empty());
        match gps {
            Some(c) => {
                let samples = c.materialize();
                epochs_ms
                    .iter()
                    .map(|&e| epoch_to_time_one(&samples, c.sample_rate_hz, e))
                    .collect()
            }
            None => {
                let origin = self.session.timestamp_utc_ms as f64;
                epochs_ms.iter().map(|&e| (e - origin) / 1000.0).collect()
            }
        }
    }

    /// One value per GPS fix, in the exact order [`crate::gps::build_gps_track`]
    /// returns fixes: `channel_id` resampled (nearest-sample, no interpolation)
    /// to each fix's recording-time instant. `NaN` where the fix falls outside
    /// the channel's sample span or the channel is absent. Empty when the
    /// session has no GPS fixes.
    ///
    /// The resampled vector is small (one f64 per fix; GPS ≈ 10 Hz) and is the
    /// only thing that crosses FFI — the channel's full sample column stays in
    /// the handle. Used to colour a GPS polyline by an arbitrary channel.
    ///
    /// Reaches base, synthesized, and derived (math / lap-slice) channels via
    /// [`Self::with_channel`].
    pub fn gps_channel_values(&self, channel_id: &str) -> Vec<f64> {
        let fixes = crate::gps::build_gps_track(self);
        if fixes.is_empty() {
            return Vec::new();
        }
        // Map each fix's wall-clock epoch to this session's recording-time
        // seconds — the common axis every channel is indexed on.
        let epochs: Vec<f64> = fixes.iter().map(|f| f.timestamp_ms as f64).collect();
        let times = self.epoch_ms_to_time_secs(&epochs);

        self.with_channel(channel_id, |c| {
            let samples = c.materialize();
            if samples.is_empty() {
                return vec![f64::NAN; times.len()];
            }
            match &c.sample_times_secs {
                // Event-driven: nearest entry in the per-sample time array.
                Some(st) => times
                    .iter()
                    .map(|&t| nearest_by_times(&samples, st, t))
                    .collect(),
                // Fixed-rate: sample i is at i / sample_rate_hz.
                None => times
                    .iter()
                    .map(|&t| nearest_by_rate(&samples, c.sample_rate_hz, t))
                    .collect(),
            }
        })
        .unwrap_or_else(|| vec![f64::NAN; times.len()])
    }

    /// Total bytes resident in this handle's sample storage — parsed +
    /// synthesized columns, event-time arrays, and the math store. Feeds the
    /// app's byte-budgeted residency policy (§15.3). Vec over-capacity slack
    /// is not counted (capacity is an allocator detail); lazy columns count
    /// only their base storage (`Ramp` 0, `Interp` its GPS-rate base).
    pub fn resident_bytes(&self) -> u64 {
        fn channel_bytes(c: &Channel) -> u64 {
            let col = match &c.column {
                RawColumn::I16 { data, .. } => data.len() * 2,
                RawColumn::I32 { data, .. } => data.len() * 4,
                RawColumn::F32 { data, .. } => data.len() * 4,
                RawColumn::F64(data) => data.len() * 8,
                RawColumn::Ramp { .. } => 0,
                RawColumn::Interp { base, .. } => base.len() * 8,
            };
            let times = c.sample_times_secs.as_ref().map_or(0, |t| t.len() * 8);
            (col + times) as u64
        }
        let cols: u64 = self.session.channels.iter().map(channel_bytes).sum();
        let math: u64 = self.derived.read().unwrap().values().map(channel_bytes).sum();
        cols + math
    }

    /// Borrow the parsed + synthesized channels. In-core only (not bridged);
    /// the exporter uses this to stream samples without cloning.
    pub fn channel_data(&self) -> &[Channel] {
        &self.session.channels
    }

    /// Borrow the ids of engine-synthesized channels (`Time`, `Distance`).
    pub fn synthesized_channel_ids(&self) -> &[String] {
        &self.synthesized_ids
    }

    /// Insert or replace a math-channel result by name (upsert). Takes `&self`
    /// via interior mutability so the Dart resolver can write back resolved
    /// dependency channels between `eval_math` calls (spec §6). Fixed-rate
    /// (no per-sample times); `sample_rate_hz` 0.0 denotes a scalar-as-channel.
    pub fn store_math(&self, channel_id: &str, sample_rate_hz: f64, samples: Vec<f64>) {
        let ch = Channel::from_f64(channel_id, sample_rate_hz, samples, None);
        self.derived
            .write()
            .unwrap()
            .insert(DerivedKey::Math(channel_id.to_string()), ch);
    }

    /// Run `f` against the [`Channel`] for `channel_id`, wherever it lives —
    /// parsed/synthesized `session.channels` or the interior-mutable math store.
    /// The math-store read lock is held only for the closure's duration. `None`
    /// if the channel is absent. The single channel-find path, shared by the
    /// seam accessors and the `ChannelLookup` impl. Passing `&Channel` (not a
    /// materialized slice) lets each accessor use the narrowest column op — a
    /// tile range, a min/max fold over raw — without widening the whole channel.
    fn with_channel<R>(&self, channel_id: &str, f: impl FnOnce(&Channel) -> R) -> Option<R> {
        if let Some(c) = self.session.channels.iter().find(|c| c.channel_id == channel_id) {
            return Some(f(c));
        }
        let store = self.derived.read().unwrap();
        store.get(&DerivedKey::from_token(channel_id)).map(|c| f(c))
    }

    /// Decimate the chart tile at (`tier`, `tile_index`) for `channel_id`.
    /// All-NaN when the channel is absent. Folds min/max per bucket directly
    /// over the raw column ([`RawColumn::min_max_range`]) — no f64 window is
    /// ever materialized, so the cost is one pass over the tile's raw samples
    /// at any tier (master design §4 seam). NaN semantics match
    /// [`crate::chart_decimation::decimate_tile_pure`]: past-end and all-NaN
    /// buckets emit `[NaN, NaN]`; mixed buckets fold finite samples only.
    pub fn decimate_tile(&self, channel_id: &str, tier: u32, tile_index: u32) -> Vec<f64> {
        self.with_channel(channel_id, |c| {
            let bucket = crate::chart_decimation::TIER_BASE.pow(tier) as usize;
            let n_buckets = crate::chart_decimation::TILE_SIZE_BUCKETS as usize;
            let tile_start = (tile_index as usize).saturating_mul(n_buckets).saturating_mul(bucket);
            let mut out = Vec::with_capacity(n_buckets * 2);
            for b in 0..n_buckets {
                let lo = tile_start.saturating_add(b * bucket);
                match c.column.min_max_range(lo, lo.saturating_add(bucket)) {
                    Some((mn, mx)) => {
                        out.push(mn);
                        out.push(mx);
                    }
                    None => {
                        out.push(f64::NAN);
                        out.push(f64::NAN);
                    }
                }
            }
            out
        })
        .unwrap_or_else(crate::chart_decimation::empty_tile)
    }

    /// Finite min/max of `channel_id`'s samples — the Y-axis auto-scale bound
    /// (master design §4 seam). `None` when absent or all-non-finite. Folds over
    /// the raw column and scales the extremum pair (no full materialization).
    pub fn channel_min_max(&self, channel_id: &str) -> Option<(f64, f64)> {
        self.with_channel(channel_id, |c| c.min_max()).flatten()
    }

    /// Physical samples for the half-open index window `[start, end)` of
    /// `channel_id` (master design §4 seam). Bounds clamp to the channel length;
    /// an empty window or absent channel yields an empty `Vec`. Only the window
    /// is widened — the f64 form exists only for this returned range.
    pub fn materialize_f64(&self, channel_id: &str, start: u32, end: u32) -> Vec<f64> {
        self.with_channel(channel_id, |c| c.materialize_range(start as usize, end as usize))
            .unwrap_or_default()
    }

    /// Physical samples within the inclusive time window `[t0_secs, t1_secs]`
    /// (master design §4 seam — the lap-window slice). Fixed-rate channels map
    /// seconds→indices via the nominal rate; event-driven channels filter on
    /// their per-sample times. The seconds↔index conversion lives here in the
    /// engine, not scattered across Dart widgets. Empty for an absent channel or
    /// a window that covers no sample.
    pub fn slice_by_time(&self, channel_id: &str, t0_secs: f64, t1_secs: f64) -> Vec<f64> {
        self.with_channel(channel_id, |c| slice_channel_by_time(c, t0_secs, t1_secs))
            .unwrap_or_default()
    }

    /// Slice `channel_id` to `[t0_secs, t1_secs]`, sample-0-rebase it, and upsert
    /// it into the derived store under a typed [`DerivedKey::LapSlice`]
    /// (`source = channel_id`). Returns `(token, length)`: the opaque storage
    /// token the chart decimates by, and the slice length (0 = no sample in the
    /// window — nothing stored, but the token is still returned). The slice never
    /// crosses FFI (spec §15.3 seam). The token is stable across edits, so
    /// re-slicing the same identity *replaces* the entry — no per-edit leak.
    pub fn slice_lap_into_store(
        &self,
        channel_id: &str,
        role: SliceRole,
        lap: u32,
        t0_secs: f64,
        t1_secs: f64,
    ) -> (String, usize) {
        let key = DerivedKey::LapSlice {
            source: channel_id.to_string(),
            role,
            lap,
        };
        let token = key.token();
        // Slice inside with_channel, store after it returns: with_channel holds
        // the derived-store READ lock when the source lives there, and the insert
        // below takes the WRITE lock — nesting them deadlocks.
        let sliced = self.with_channel(channel_id, |c| {
            (slice_channel_by_time(c, t0_secs, t1_secs), c.sample_rate_hz)
        });
        match sliced {
            Some((slice, rate)) if !slice.is_empty() => {
                let n = slice.len();
                let ch = Channel::from_f64(&token, rate, slice, None);
                self.derived.write().unwrap().insert(key, ch);
                (token, n)
            }
            _ => (token, 0),
        }
    }

    /// Drop every derived entry not backed by reality: a [`DerivedKey::Math`]
    /// survives only when its name is in `live_sources` (the current
    /// math-channel names); a [`DerivedKey::LapSlice`] survives only when its
    /// `source` is a live math name **or** a base channel. Called on the eval
    /// path with the live channel-name list, so a deleted/renamed math channel's
    /// output and slices are reclaimed with no per-name delete wiring (spec §4).
    pub fn retain_derived(&self, live_sources: &[String]) {
        use std::collections::HashSet;
        let live: HashSet<&str> = live_sources.iter().map(String::as_str).collect();
        let base: HashSet<&str> = self
            .session
            .channels
            .iter()
            .map(|c| c.channel_id.as_str())
            .collect();
        self.derived.write().unwrap().retain(|k, _| match k {
            DerivedKey::Math(name) => live.contains(name.as_str()),
            DerivedKey::LapSlice { source, .. } => {
                live.contains(source.as_str()) || base.contains(source.as_str())
            }
        });
    }

    /// Welch PSD/spectrum for `channel_id`, computed entirely in the engine from
    /// the channel's samples (materialized transiently from the compact column,
    /// never resident in Dart). Empty [`WelchResult`](crate::fft::WelchResult)
    /// for an absent channel. Wraps [`crate::fft::welch`] — the FFT input never
    /// crosses FFI; only the spectrum does (master design §4 seam, Phase D-drain).
    #[allow(clippy::too_many_arguments)]
    pub fn welch_channel(
        &self,
        channel_id: &str,
        window: crate::fft::FftWindow,
        nperseg: usize,
        noverlap: usize,
        detrend: crate::fft::Detrend,
        averaging: crate::fft::Averaging,
        scaling: crate::fft::Scaling,
    ) -> crate::fft::WelchResult {
        match self.with_channel(channel_id, |c| (c.materialize(), c.sample_rate_hz)) {
            Some((samples, rate)) => crate::fft::welch(
                samples, rate, window, nperseg, noverlap, detrend, averaging, scaling,
            ),
            None => crate::fft::WelchResult { freqs_hz: Vec::new(), values: Vec::new() },
        }
    }

    /// Welch spectrum for `channel_id` over the inclusive time window
    /// `[t0_secs, t1_secs]`. Slices the channel by time (reusing
    /// [`slice_channel_by_time`]) then runs [`crate::fft::welch`] at the channel's
    /// rate — the slice never crosses FFI, only the [`WelchResult`](crate::fft::WelchResult)
    /// does. Empty result for an absent channel or a window covering no sample.
    /// Pass `[0, duration]` for a whole-channel spectrum (identical to `welch_channel`).
    #[allow(clippy::too_many_arguments)]
    pub fn welch_channel_windowed(
        &self,
        channel_id: &str,
        t0_secs: f64,
        t1_secs: f64,
        window: crate::fft::FftWindow,
        nperseg: usize,
        noverlap: usize,
        detrend: crate::fft::Detrend,
        averaging: crate::fft::Averaging,
        scaling: crate::fft::Scaling,
    ) -> crate::fft::WelchResult {
        match self.with_channel(channel_id, |c| (slice_channel_by_time(c, t0_secs, t1_secs), c.sample_rate_hz)) {
            Some((slice, rate)) if !slice.is_empty() => {
                crate::fft::welch(slice, rate, window, nperseg, noverlap, detrend, averaging, scaling)
            }
            _ => crate::fft::WelchResult { freqs_hz: Vec::new(), values: Vec::new() },
        }
    }

    /// Spectrogram for `channel_id` over the inclusive time window
    /// `[t0_secs, t1_secs]`. Slices by time then runs
    /// [`crate::spectrogram::spectrogram`] at the channel's rate; the returned
    /// `times_secs` are shifted to **absolute session seconds** (slice-relative +
    /// `t0_secs`) so the heatmap's X axis matches the time-series charts. Empty
    /// result for an absent channel or an empty window.
    #[allow(clippy::too_many_arguments)]
    pub fn spectrogram_channel(
        &self,
        channel_id: &str,
        t0_secs: f64,
        t1_secs: f64,
        window: crate::fft::FftWindow,
        nperseg: usize,
        noverlap: usize,
        detrend: crate::fft::Detrend,
        scaling: crate::fft::Scaling,
    ) -> crate::spectrogram::SpectrogramResult {
        match self.with_channel(channel_id, |c| (slice_channel_by_time(c, t0_secs, t1_secs), c.sample_rate_hz)) {
            Some((slice, rate)) if !slice.is_empty() => {
                let mut s = crate::spectrogram::spectrogram(slice, rate, window, nperseg, noverlap, detrend, scaling);
                for t in s.times_secs.iter_mut() {
                    *t += t0_secs;
                }
                s
            }
            _ => crate::spectrogram::SpectrogramResult::empty(),
        }
    }

    /// Value-distribution histogram for `channel_id`, computed in the engine
    /// from the channel's samples (materialized transiently from the compact
    /// column, never resident in Dart). Wraps [`crate::histogram::histogram`] —
    /// the samples never cross FFI, only the small
    /// [`HistogramResult`](crate::histogram::HistogramResult) does (master
    /// design §4 seam). Empty result for an absent channel. `symmetric` centres
    /// the range on zero (signed velocity distributions); range otherwise the
    /// data min/max.
    ///
    /// `range` pins the binning extent (the overlay path passes a range shared
    /// across every series so their bins align); `None` falls back to the data
    /// min/max, widened to `[−m, m]` when `symmetric`.
    pub fn histogram(
        &self,
        channel_id: &str,
        bins: usize,
        symmetric: bool,
        range: Option<(f64, f64)>,
    ) -> crate::histogram::HistogramResult {
        match self.with_channel(channel_id, |c| c.materialize()) {
            Some(samples) => crate::histogram::histogram(&samples, bins, symmetric, range),
            None => crate::histogram::HistogramResult::empty(),
        }
    }
}

/// Slices a channel to the inclusive time window `[t0, t1]` (seconds).
///
/// Fixed-rate (`rate > 0`): sample `i` is at `i / rate`; returns the contiguous
/// run `ceil(t0·rate)..=floor(t1·rate)` widened from the raw column via
/// [`RawColumn::materialize_range`] — the full channel is never materialized.
/// Event-driven (`rate == 0` with per-sample times): returns samples whose
/// time is in `[t0, t1]` (sparse by design, so the transient materialize is
/// small). Empty when the window is inverted, the data is empty, or no sample
/// falls inside.
fn slice_channel_by_time(c: &Channel, t0: f64, t1: f64) -> Vec<f64> {
    if c.is_empty() || t1 < t0 {
        return Vec::new();
    }
    if c.sample_rate_hz > 0.0 {
        return match fixed_rate_slice_range(c.len(), c.sample_rate_hz, t0, t1) {
            Some((lo, hi)) => c.column.materialize_range(lo, hi + 1),
            None => Vec::new(),
        };
    }
    match c.sample_times_secs.as_deref() {
        Some(ts) => {
            let samples = c.materialize();
            samples
                .iter()
                .zip(ts.iter())
                .filter(|(_, &t)| t >= t0 && t <= t1)
                .map(|(&s, _)| s)
                .collect()
        }
        None => Vec::new(),
    }
}

/// Inclusive-window index range for a fixed-rate channel: `ceil(t0·rate) ..=
/// floor(t1·rate)`, clamped; `None` when no sample falls inside. The 1e-9
/// epsilon absorbs float boundary error so an exact grid point isn't dropped
/// (same arithmetic as the previous whole-channel slice — outputs are
/// bit-identical).
fn fixed_rate_slice_range(len: usize, rate: f64, t0: f64, t1: f64) -> Option<(usize, usize)> {
    if len == 0 || t1 < t0 {
        return None;
    }
    const EPS: f64 = 1e-9;
    let lo_f = (t0 * rate - EPS).ceil();
    let hi_f = (t1 * rate + EPS).floor();
    if hi_f < 0.0 {
        return None;
    }
    let lo = lo_f.max(0.0) as usize;
    let hi = (hi_f as usize).min(len - 1);
    if lo > hi || lo >= len {
        return None;
    }
    Some((lo, hi))
}

/// Maps one epoch-ms `target` to uniform-Time seconds via a bracketing binary
/// search over the (assumed monotonic) `GPS_EpochMs` `samples`, then `/ rate`.
/// Faithful port of the Dart `epochToUniformTimeSec` interpolation.
fn epoch_to_time_one(samples: &[f64], rate: f64, target: f64) -> f64 {
    let last = samples.len() - 1;
    if target <= samples[0] {
        return 0.0;
    }
    if target >= samples[last] {
        return last as f64 / rate;
    }
    let (mut lo, mut hi) = (0usize, last);
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if samples[mid] <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = samples[hi] - samples[lo];
    let frac = if span == 0.0 { 0.0 } else { (target - samples[lo]) / span };
    (lo as f64 + frac) / rate
}

impl crate::math::eval::ChannelLookup for SessionHandle {
    fn lookup(&self, name: &str) -> Option<crate::math::eval::LookupChannel> {
        // Base + synthesized channels win over the math store (with_channel
        // checks session.channels first). The evaluator needs the whole array.
        self.with_channel(name, |c| crate::math::eval::LookupChannel {
            samples: Arc::from(c.materialize()),
            sample_rate_hz: c.sample_rate_hz,
        })
    }

    fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
        // (len, rate) without materializing — the closed-form time base reads
        // this so the zero-storage `Time` ramp is never widened.
        self.with_channel(name, |c| (c.len(), c.sample_rate_hz))
    }

    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        // Highest-rate non-event channel with the most samples, across base +
        // math store. Mirrors Dart `_resolveTimeBase`'s fallback scan.
        let mut best: Option<(usize, f64)> = None;
        let mut consider = |len: usize, rate: f64| {
            if rate > 0.0 && best.map_or(true, |(blen, _)| len > blen) {
                best = Some((len, rate));
            }
        };
        for c in &self.session.channels {
            consider(c.len(), c.sample_rate_hz);
        }
        for c in self.derived.read().unwrap().values() {
            consider(c.len(), c.sample_rate_hz);
        }
        best
    }

    fn sample_times(&self, name: &str) -> Option<Vec<f64>> {
        self.channel_sample_times(name)
    }
}

/// Value of a fixed-rate channel at recording-time `t_secs` — the nearest
/// sample (sample `i` is at `i / rate_hz`). `NaN` for a non-positive rate,
/// negative time, or an index past the last sample.
fn nearest_by_rate(samples: &[f64], rate_hz: f64, t_secs: f64) -> f64 {
    if rate_hz <= 0.0 || t_secs < 0.0 {
        return f64::NAN;
    }
    let idx = (t_secs * rate_hz).round();
    if idx < 0.0 || idx as usize >= samples.len() {
        return f64::NAN;
    }
    samples[idx as usize]
}

/// Value of an event-driven channel at recording-time `t_secs` — the sample
/// whose `times` entry (assumed ascending) is closest. `NaN` when the arrays
/// are empty. Times beyond the ends clamp to the first / last sample.
fn nearest_by_times(samples: &[f64], times: &[f64], t_secs: f64) -> f64 {
    let n = samples.len().min(times.len());
    if n == 0 {
        return f64::NAN;
    }
    let pos = times[..n].partition_point(|&x| x < t_secs);
    let hi = pos.min(n - 1);
    let lo = pos.saturating_sub(1);
    let pick = if (times[lo] - t_secs).abs() <= (times[hi] - t_secs).abs() {
        lo
    } else {
        hi
    };
    samples[pick]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_channel(id: &str, rate: f64, samples: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: rate, samples, sample_times_secs: None }
    }

    fn test_meta() -> SessionMetaInput {
        SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        }
    }

    #[test]
    fn derived_key_token_round_trips_for_math_and_slice() {
        // Arrange
        let math = DerivedKey::Math("Fork velocity".to_string());
        let main = DerivedKey::LapSlice {
            source: "Fork velocity".to_string(),
            role: SliceRole::Main,
            lap: 7,
        };
        let overlay = DerivedKey::LapSlice {
            source: "IMU0_AccelZ".to_string(),
            role: SliceRole::Overlay,
            lap: 12,
        };

        // Act + Assert — token() then from_token() is the identity.
        assert_eq!(DerivedKey::from_token(&math.token()), math);
        assert_eq!(DerivedKey::from_token(&main.token()), main);
        assert_eq!(DerivedKey::from_token(&overlay.token()), overlay);
    }

    #[test]
    fn math_token_is_plain_name_slice_token_is_sentinel_prefixed_and_distinct() {
        // Arrange / Act / Assert
        assert_eq!(DerivedKey::Math("Speed".to_string()).token(), "Speed");
        let slice = DerivedKey::LapSlice {
            source: "Speed".to_string(),
            role: SliceRole::Main,
            lap: 3,
        };
        assert!(slice.token().starts_with('\u{1}'));
        // A math output named "Speed" and a lap slice of "Speed" produce
        // different tokens — they can never collide in the store.
        assert_ne!(DerivedKey::Math("Speed".to_string()).token(), slice.token());
    }

    #[test]
    fn from_channels_runs_synthesis_and_reports_metadata() {
        // Arrange — one 10 Hz channel of 20 samples (= 2000 ms).
        let meta = SessionMetaInput {
            session_id: "abc".to_string(),
            device_id: "dev".to_string(),
            timestamp_utc_ms: 1700,
            config_checksum: "crc".to_string(),
        };

        // Act
        let h = SessionHandle::from_channels(meta, vec![input_channel("Main", 10.0, vec![0.0; 20])]);
        let m = h.metadata();

        // Assert — Time was synthesized; duration is the max span; no truncation.
        assert_eq!(m.session_id, "abc");
        assert_eq!(m.duration_ms, 2000);
        assert_eq!(m.truncation_warning, None);
        assert!(h.channels().iter().any(|c| c.channel_id == "Time" && c.synthesized));
        assert_eq!(m.channel_count, h.channels().len() as u32);
    }

    #[test]
    fn channel_samples_returns_data_and_empty_for_absent() {
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 1.0, vec![1.0, 2.0, 3.0])]);

        // Act + Assert
        assert_eq!(h.channel_samples("X"), vec![1.0, 2.0, 3.0]);
        assert_eq!(h.channel_samples("nope"), Vec::<f64>::new());
    }

    #[test]
    fn resident_bytes_counts_columns_times_and_math_store() {
        // Arrange — one event-driven F64 channel (10 samples + 10 times) plus
        // a 5-sample math entry. Event-only session → no Time/Distance.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![ChannelInput {
                channel_id: "E".to_string(),
                sample_rate_hz: 0.0,
                samples: vec![1.0; 10],
                sample_times_secs: Some(vec![0.5; 10]),
            }],
        );
        h.store_math("M", 1.0, vec![2.0; 5]);

        // Act + Assert — 10×8 (samples) + 10×8 (times) + 5×8 (math) = 200.
        assert_eq!(h.resident_bytes(), 200);
    }

    #[test]
    fn resident_bytes_lazy_columns_count_base_storage_only() {
        // Arrange — fixed-rate channel synthesizes Time (Ramp → 0 bytes).
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![0.0; 8])]);

        // Act + Assert — X = 8×8 = 64; synthesized Time ramp adds nothing.
        assert_eq!(h.resident_bytes(), 64);
    }

    #[test]
    fn slice_by_time_fixed_rate_compact_column_matches_materialized_window() {
        // Arrange — compact i16 channel, 10 Hz, scale 0.5: the slice must be
        // computed from the index window, never a whole-channel widen.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let mut h = SessionHandle::from_channels(meta, vec![]);
        h.session.channels.push(Channel {
            channel_id: "C".to_string(),
            sample_rate_hz: 10.0,
            column: RawColumn::I16 { data: (0..100).collect(), scale: 0.5, offset: 0.0 },
            sample_times_secs: None,
            gaps: Vec::new(),
        });

        // Act
        let got = h.slice_by_time("C", 1.0, 2.5);

        // Assert — ceil(1.0·10)=10 ..= floor(2.5·10)=25, physical = raw × 0.5.
        let want: Vec<f64> = (10..=25).map(|r| r as f64 * 0.5).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn slice_lap_into_store_stores_under_typed_key_addressable_by_token() {
        // Arrange — source in the derived store; the slice read and the store
        // write must not hold the RwLock simultaneously (deadlock regression).
        let h = SessionHandle::from_channels(test_meta(), vec![]);
        h.store_math("M", 10.0, (0..100).map(|i| i as f64).collect());

        // Act — slice [1,2] s = indices 10..=20 → 11 samples.
        let (token, len) = h.slice_lap_into_store("M", SliceRole::Main, 7, 1.0, 2.0);

        // Assert — stored at the source rate, addressable by the returned token.
        use crate::math::eval::ChannelLookup;
        assert_eq!(len, 11);
        assert!(token.starts_with('\u{1}'));
        let stored = h.lookup(&token).unwrap();
        assert_eq!(stored.samples, (10..=20).map(|i| i as f64).collect::<Vec<_>>().into());
        assert_eq!(stored.sample_rate_hz, 10.0);
    }

    #[test]
    fn slice_lap_into_store_does_not_collide_with_same_named_math_output() {
        // Arrange
        let h = SessionHandle::from_channels(test_meta(), vec![]);
        h.store_math("M", 10.0, (0..100).map(|i| i as f64).collect());

        // Act — slice M into a lap slice whose source is also "M".
        let (token, _) = h.slice_lap_into_store("M", SliceRole::Main, 1, 1.0, 2.0);

        // Assert — the math output "M" survives intact; the slice has its own id.
        assert_eq!(h.channel_min_max("M"), Some((0.0, 99.0)));
        assert_ne!(token, "M");
    }

    #[test]
    fn slice_lap_into_store_replaces_same_identity_in_place() {
        // Arrange
        let h = SessionHandle::from_channels(test_meta(), vec![]);
        h.store_math("M", 10.0, (0..100).map(|i| i as f64).collect());

        // Act — same (source, role, lap), two different windows.
        let (t1, _) = h.slice_lap_into_store("M", SliceRole::Main, 5, 1.0, 2.0);
        let b1 = h.channel_min_max(&t1);
        let (t2, _) = h.slice_lap_into_store("M", SliceRole::Main, 5, 3.0, 4.0);
        let b2 = h.channel_min_max(&t2);

        // Assert — stable token, content replaced in place (not appended).
        assert_eq!(t1, t2);
        assert_ne!(b1, b2);
    }

    #[test]
    fn slice_lap_into_store_empty_window_stores_nothing() {
        // Arrange
        let h = SessionHandle::from_channels(test_meta(), vec![input_channel("X", 10.0, vec![0.0; 10])]);

        // Act + Assert — window past the data, and an absent channel: length 0,
        // nothing addressable under the token.
        let (token, len) = h.slice_lap_into_store("X", SliceRole::Main, 1, 5.0, 6.0);
        assert_eq!(len, 0);
        assert!(h.channel_min_max(&token).is_none());
        let (_, absent) = h.slice_lap_into_store("nope", SliceRole::Main, 1, 0.0, 1.0);
        assert_eq!(absent, 0);
    }

    #[test]
    fn retain_derived_keeps_live_and_base_sources_drops_orphans() {
        // Arrange — base channel "B"; math outputs "Keep" and "Gone"; a lap
        // slice of each math output and one of the base channel.
        let h = SessionHandle::from_channels(test_meta(), vec![input_channel("B", 10.0, vec![1.0; 20])]);
        h.store_math("Keep", 10.0, vec![3.0; 20]);
        h.store_math("Gone", 10.0, vec![4.0; 20]);
        let (keep_slice, _) = h.slice_lap_into_store("Keep", SliceRole::Main, 1, 0.0, 0.5);
        let (gone_slice, _) = h.slice_lap_into_store("Gone", SliceRole::Main, 1, 0.0, 0.5);
        let (base_slice, _) = h.slice_lap_into_store("B", SliceRole::Overlay, 2, 0.0, 0.5);

        // Act — only "Keep" remains a live math channel.
        h.retain_derived(&["Keep".to_string()]);

        // Assert — live output + its slice + the base-sourced slice survive;
        // the orphaned output and its slice are dropped.
        assert!(h.channel_min_max("Keep").is_some());
        assert!(h.channel_min_max(&keep_slice).is_some());
        assert!(h.channel_min_max(&base_slice).is_some());
        assert!(h.channel_min_max("Gone").is_none());
        assert!(h.channel_min_max(&gone_slice).is_none());
    }

    #[test]
    fn decimate_tile_matches_pure_fold_over_materialized_samples() {
        // Arrange — compact i16 column with scale/offset, length not
        // bucket-aligned, so extremum-pair scaling and NaN right-edge padding
        // are both exercised.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let mut h = SessionHandle::from_channels(meta, vec![]);
        let raws: Vec<i16> = (0..5000).map(|i| ((i * 37) % 1000) as i16 - 500).collect();
        h.session.channels.push(Channel {
            channel_id: "C".to_string(),
            sample_rate_hz: 100.0,
            column: RawColumn::I16 { data: raws, scale: 0.5, offset: 1.0 },
            sample_times_secs: None,
            gaps: Vec::new(),
        });

        // Act + Assert — every tier/tile pair matches the reference fold over
        // the fully materialized samples (including NaN padding past the end).
        let samples = h.channel_samples("C");
        for tier in 0..=6u32 {
            for tile in 0..2u32 {
                let got = h.decimate_tile("C", tier, tile);
                let want = crate::chart_decimation::decimate_channel(&samples, tier, tile);
                assert_eq!(got.len(), want.len(), "tier {tier} tile {tile}");
                for (g, w) in got.iter().zip(want.iter()) {
                    assert!(
                        (g.is_nan() && w.is_nan()) || g == w,
                        "tier {tier} tile {tile}: got {g}, want {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn from_path_missing_file_is_io_error() {
        // Act
        let r = SessionHandle::from_path("definitely/not/a/real/file.idl0");

        // Assert
        assert!(matches!(r, Err(ParseError::Io(_))));
    }

    #[test]
    fn channel_data_borrows_parsed_and_synthesized_channels() {
        // Arrange — one fixed-rate channel; synthesis adds "Time".
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![1.0, 2.0])]);

        // Act
        let data = h.channel_data();

        // Assert — the borrow exposes the same channels channels() reports.
        assert_eq!(data.len(), h.channels().len());
        assert!(data.iter().any(|c| c.channel_id == "X"));
        assert!(data.iter().any(|c| c.channel_id == "Time"));
    }

    #[test]
    fn store_math_then_lookup_finds_it() {
        use crate::math::eval::ChannelLookup;
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![1.0, 2.0])]);

        // Act — write a math channel via &self, then resolve it.
        h.store_math("BrakePower", 10.0, vec![7.0, 8.0]);
        let got = h.lookup("BrakePower").unwrap();

        // Assert
        assert_eq!(got.samples, vec![7.0, 8.0].into());
        assert_eq!(got.sample_rate_hz, 10.0);
    }

    #[test]
    fn store_math_upserts_by_name() {
        use crate::math::eval::ChannelLookup;
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 1.0, vec![0.0])]);

        // Act — same name twice; the second write replaces the first.
        h.store_math("M", 1.0, vec![1.0]);
        h.store_math("M", 1.0, vec![2.0, 3.0]);

        // Assert
        assert_eq!(h.lookup("M").unwrap().samples, vec![2.0, 3.0].into());
    }

    #[test]
    fn lookup_prefers_base_channel_over_math_store() {
        use crate::math::eval::ChannelLookup;
        // Arrange — a base channel "X" and (illegally) a same-named math entry.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 1.0, vec![1.0, 1.0])]);
        h.store_math("X", 1.0, vec![9.0, 9.0]);

        // Act / Assert — base wins.
        assert_eq!(h.lookup("X").unwrap().samples, vec![1.0, 1.0].into());
    }

    #[test]
    fn best_time_base_dims_picks_highest_count_channel() {
        use crate::math::eval::ChannelLookup;
        // Arrange — two channels; the longer one wins the time base.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![
                input_channel("Lo", 10.0, vec![0.0; 5]),
                input_channel("Hi", 100.0, vec![0.0; 50]),
            ],
        );

        // Act
        let (len, rate) = h.best_time_base_dims().unwrap();

        // Assert
        assert_eq!(len, 50);
        assert_eq!(rate, 100.0);
    }

    #[test]
    fn decimate_tile_base_channel_returns_bucket_pairs() {
        // Arrange — base channel "X" = 0..16.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let samples: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, samples)]);

        // Act — tier 0 (raw): bucket k is [sample k, sample k].
        let out = h.decimate_tile("X", 0, 0);

        // Assert
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn decimate_tile_math_store_channel_is_decimated() {
        // Arrange — a math channel written via store_math (not a base channel).
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![0.0; 4])]);
        h.store_math("M", 10.0, (0..16).map(|i| i as f64).collect());

        // Act
        let out = h.decimate_tile("M", 1, 0);

        // Assert — tier 1, bucket size 8: bucket 0 = 0..8 (0,7), bucket 1 = 8..16 (8,15).
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 7.0);
        assert_eq!(out[2], 8.0);
        assert_eq!(out[3], 15.0);
    }

    #[test]
    fn decimate_tile_absent_channel_returns_all_nan() {
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![1.0])]);

        // Act
        let out = h.decimate_tile("nope", 0, 0);

        // Assert
        assert_eq!(out.len(), 1024 * 2);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn synthesized_channel_ids_lists_only_engine_added_channels() {
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![1.0, 2.0])]);

        // Act
        let syn = h.synthesized_channel_ids();

        // Assert — "Time" is synthesized; "X" is not.
        assert!(syn.iter().any(|id| id == "Time"));
        assert!(!syn.iter().any(|id| id == "X"));
    }

    #[test]
    fn epoch_ms_to_time_secs_interpolates_against_gps_epoch() {
        // Arrange — GPS_EpochMs at 10 Hz: epoch 1000,1100,1200,... ms.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 1000,
            config_checksum: String::new(),
        };
        let epochs: Vec<f64> = (0..10).map(|i| 1000.0 + i as f64 * 100.0).collect();
        let h = SessionHandle::from_channels(meta, vec![input_channel("GPS_EpochMs", 10.0, epochs)]);

        // Act — midpoint between sample 2 (1200ms) and 3 (1300ms) → index 2.5 → 0.25 s.
        let out = h.epoch_ms_to_time_secs(&[1000.0, 1250.0, 1900.0]);

        // Assert — first sample → 0.0; 1250 → 0.25 s; last sample → 9/10 = 0.9 s.
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 0.25).abs() < 1e-9);
        assert!((out[2] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn epoch_ms_to_time_secs_clamps_outside_gps_range() {
        // Arrange
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(
            meta,
            vec![input_channel("GPS_EpochMs", 10.0, vec![1000.0, 1100.0, 1200.0])],
        );

        // Act / Assert — below first → 0.0; above last → (len-1)/rate = 2/10.
        assert_eq!(h.epoch_ms_to_time_secs(&[500.0])[0], 0.0);
        assert!((h.epoch_ms_to_time_secs(&[5000.0])[0] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn epoch_ms_to_time_secs_falls_back_to_backfilled_origin_without_gps() {
        // Arrange — no GPS_EpochMs; origin = timestamp_utc_ms = 2000.
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 2000,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 10.0, vec![0.0; 4])]);

        // Act — (5000 - 2000)/1000 = 3.0 s.
        let out = h.epoch_ms_to_time_secs(&[5000.0, 2000.0]);

        // Assert
        assert!((out[0] - 3.0).abs() < 1e-9);
        assert!((out[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn epoch_ms_to_time_secs_empty_input_is_empty() {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        let h = SessionHandle::from_channels(meta, vec![input_channel("X", 1.0, vec![0.0])]);
        assert!(h.epoch_ms_to_time_secs(&[]).is_empty());
    }

    // ---- Phase 0 seam: channel_min_max ---------------------------------------

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    #[test]
    fn channel_min_max_returns_finite_min_and_max() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 1.0, vec![3.0, 1.0, 4.0, 1.0, 5.0])]);

        // Act
        let (min, max) = h.channel_min_max("X").unwrap();

        // Assert
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn channel_min_max_ignores_non_finite_values() {
        // Arrange — NaN and infinities must not poison the bounds.
        let h = handle_with(vec![input_channel(
            "X",
            1.0,
            vec![f64::NAN, 2.0, f64::INFINITY, -3.0, f64::NEG_INFINITY],
        )]);

        // Act
        let (min, max) = h.channel_min_max("X").unwrap();

        // Assert
        assert_eq!(min, -3.0);
        assert_eq!(max, 2.0);
    }

    #[test]
    fn channel_min_max_absent_channel_is_none() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 1.0, vec![1.0])]);

        // Act + Assert
        assert_eq!(h.channel_min_max("nope"), None);
    }

    #[test]
    fn channel_min_max_all_non_finite_is_none() {
        // Arrange — no finite sample → no bounds.
        let h = handle_with(vec![input_channel("X", 1.0, vec![f64::NAN, f64::INFINITY])]);

        // Act + Assert
        assert_eq!(h.channel_min_max("X"), None);
    }

    #[test]
    fn channel_min_max_reads_math_store_channel() {
        // Arrange — bounds must work for upserted math channels too.
        let h = handle_with(vec![input_channel("X", 1.0, vec![0.0])]);
        h.store_math("M", 1.0, vec![-2.0, 7.0, 3.0]);

        // Act
        let (min, max) = h.channel_min_max("M").unwrap();

        // Assert
        assert_eq!(min, -2.0);
        assert_eq!(max, 7.0);
    }

    // ---- Phase 0 seam: materialize_f64 ---------------------------------------

    #[test]
    fn materialize_f64_returns_half_open_index_window() {
        // Arrange — X = 0..5.
        let h = handle_with(vec![input_channel("X", 1.0, vec![0.0, 1.0, 2.0, 3.0, 4.0])]);

        // Act — [1, 4) → samples 1,2,3.
        let out = h.materialize_f64("X", 1, 4);

        // Assert
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn materialize_f64_clamps_end_past_length() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 1.0, vec![0.0, 1.0, 2.0, 3.0, 4.0])]);

        // Act — end clamps to len.
        let out = h.materialize_f64("X", 3, 100);

        // Assert
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn materialize_f64_start_at_or_past_end_is_empty() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 1.0, vec![0.0, 1.0, 2.0])]);

        // Act + Assert — start >= end, and start past length, both empty.
        assert!(h.materialize_f64("X", 2, 2).is_empty());
        assert!(h.materialize_f64("X", 3, 1).is_empty());
        assert!(h.materialize_f64("X", 10, 20).is_empty());
    }

    #[test]
    fn materialize_f64_absent_channel_is_empty() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 1.0, vec![0.0])]);

        // Act + Assert
        assert!(h.materialize_f64("nope", 0, 1).is_empty());
    }

    // ---- Phase 0 seam: slice_by_time -----------------------------------------

    #[test]
    fn slice_by_time_fixed_rate_returns_samples_in_window_inclusive() {
        // Arrange — 10 Hz, samples 0..10 (sample i at i/10 s).
        let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let h = handle_with(vec![input_channel("X", 10.0, samples)]);

        // Act — window [0.2, 0.5] s → indices 2..=5.
        let out = h.slice_by_time("X", 0.2, 0.5);

        // Assert
        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn slice_by_time_event_driven_uses_sample_times() {
        // Arrange — event channel with explicit times.
        let ch = ChannelInput {
            channel_id: "E".to_string(),
            sample_rate_hz: 0.0,
            samples: vec![10.0, 20.0, 30.0, 40.0, 50.0],
            sample_times_secs: Some(vec![0.0, 0.5, 1.0, 1.5, 2.0]),
        };
        let h = handle_with(vec![ch]);

        // Act — window [0.5, 1.5] s → samples at 0.5, 1.0, 1.5.
        let out = h.slice_by_time("E", 0.5, 1.5);

        // Assert
        assert_eq!(out, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_by_time_window_outside_data_is_empty() {
        // Arrange
        let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let h = handle_with(vec![input_channel("X", 10.0, samples)]);

        // Act + Assert — window starts after the last sample (0.9 s).
        assert!(h.slice_by_time("X", 5.0, 6.0).is_empty());
    }

    #[test]
    fn slice_by_time_absent_channel_is_empty() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 10.0, vec![0.0; 4])]);

        // Act + Assert
        assert!(h.slice_by_time("nope", 0.0, 1.0).is_empty());
    }

    #[test]
    fn welch_channel_matches_welch_over_materialized_samples() {
        // Arrange — a channel and explicit Welch params.
        let samples: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let h = handle_with(vec![input_channel("X", 100.0, samples)]);

        // Act — handle path vs. materialize-then-welch must be bit-identical.
        let via_handle = h.welch_channel(
            "X",
            crate::fft::FftWindow::Hann,
            64,
            32,
            crate::fft::Detrend::Mean,
            crate::fft::Averaging::Mean,
            crate::fft::Scaling::Density,
        );
        let direct = crate::fft::welch(
            h.channel_samples("X"),
            100.0,
            crate::fft::FftWindow::Hann,
            64,
            32,
            crate::fft::Detrend::Mean,
            crate::fft::Averaging::Mean,
            crate::fft::Scaling::Density,
        );

        // Assert
        assert_eq!(via_handle.freqs_hz, direct.freqs_hz);
        assert_eq!(via_handle.values, direct.values);
    }

    #[test]
    fn welch_channel_absent_is_empty() {
        // Arrange
        let h = handle_with(vec![input_channel("X", 100.0, vec![0.0; 8])]);

        // Act
        let r = h.welch_channel(
            "nope",
            crate::fft::FftWindow::Hann,
            4,
            2,
            crate::fft::Detrend::None,
            crate::fft::Averaging::Mean,
            crate::fft::Scaling::Magnitude,
        );

        // Assert
        assert!(r.freqs_hz.is_empty() && r.values.is_empty());
    }

    #[test]
    fn welch_channel_windowed_matches_welch_over_sliced_samples() {
        // Arrange — 10 Hz channel, 200 samples; window [2.0, 6.0] s → indices 20..=60.
        let samples: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let h = handle_with(vec![input_channel("X", 10.0, samples)]);

        // Act — windowed handle path vs. manual slice → welch.
        let via_handle = h.welch_channel_windowed(
            "X", 2.0, 6.0, crate::fft::FftWindow::Hann, 0, 0,
            crate::fft::Detrend::Mean, crate::fft::Averaging::Mean, crate::fft::Scaling::Density,
        );
        let sliced = h.slice_by_time("X", 2.0, 6.0);
        let direct = crate::fft::welch(
            sliced, 10.0, crate::fft::FftWindow::Hann, 0, 0,
            crate::fft::Detrend::Mean, crate::fft::Averaging::Mean, crate::fft::Scaling::Density,
        );

        // Assert
        assert_eq!(via_handle.freqs_hz, direct.freqs_hz);
        assert_eq!(via_handle.values, direct.values);
    }

    #[test]
    fn spectrogram_channel_offsets_times_to_absolute_session_seconds() {
        // Arrange — 64 Hz channel, 256 samples; window [1.0, 4.0] s.
        let samples: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
        let h = handle_with(vec![input_channel("X", 64.0, samples)]);

        // Act
        let s = h.spectrogram_channel(
            "X", 1.0, 4.0, crate::fft::FftWindow::Hann, 64, 32,
            crate::fft::Detrend::None, crate::fft::Scaling::Magnitude,
        );

        // Assert — first frame centre is >= window start (1.0 s), not slice-relative 0.
        assert!(s.n_times > 0);
        assert!(s.times_secs[0] >= 1.0, "times must be absolute: {}", s.times_secs[0]);
    }

    #[test]
    fn spectrogram_channel_absent_is_empty() {
        let h = handle_with(vec![input_channel("X", 64.0, vec![0.0; 64])]);
        let s = h.spectrogram_channel(
            "nope", 0.0, 1.0, crate::fft::FftWindow::Hann, 0, 0,
            crate::fft::Detrend::None, crate::fft::Scaling::Magnitude,
        );
        assert_eq!(s.n_times, 0);
        assert!(s.power.is_empty());
    }

    #[test]
    fn sample_times_trait_returns_times_for_event_driven_and_none_for_fixed_rate() {
        use crate::math::eval::ChannelLookup;

        // Arrange — one event-driven channel with explicit per-sample times,
        // one fixed-rate channel with no per-sample times.
        let event_ch = ChannelInput {
            channel_id: "GPS_Latitude".to_string(),
            sample_rate_hz: 0.0,
            samples: vec![51.5, 51.6, 51.7],
            sample_times_secs: Some(vec![0.0, 1.5, 3.2]),
        };
        let fixed_ch = ChannelInput {
            channel_id: "IMU0_AccelX".to_string(),
            sample_rate_hz: 200.0,
            samples: vec![0.1, 0.2, 0.3],
            sample_times_secs: None,
        };
        let h = SessionHandle::from_channels(test_meta(), vec![event_ch, fixed_ch]);

        // Act
        let event_times = h.sample_times("GPS_Latitude");
        let fixed_times = h.sample_times("IMU0_AccelX");
        let absent_times = h.sample_times("nope");

        // Assert — event-driven channel returns the exact per-sample times;
        // fixed-rate channel returns None; absent channel returns None.
        assert_eq!(event_times, Some(vec![0.0, 1.5, 3.2]));
        assert_eq!(fixed_times, None);
        assert_eq!(absent_times, None);
    }
}

#[cfg(test)]
mod gps_channel_values_tests {
    use super::*;

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        SessionHandle::from_channels(
            SessionMetaInput {
                session_id: String::new(),
                device_id: String::new(),
                timestamp_utc_ms: 0,
                config_checksum: String::new(),
            },
            channels,
        )
    }

    fn ch(id: &str, rate: f64, samples: Vec<f64>) -> ChannelInput {
        ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: rate,
            samples,
            sample_times_secs: None,
        }
    }

    #[test]
    fn gps_channel_values_resamples_fixed_rate_channel_at_fix_times() {
        // Arrange — GPS 1 Hz from epoch 1000ms: fixes at recording-secs 0,1,2.
        // A 2 Hz target channel: sample i at i/2 s → [10,11,12,13,14] covers
        // secs 0..2. Nearest at 0→idx0(10), 1→idx2(12), 2→idx4(14).
        let h = handle_with(vec![
            ch("GPS_Latitude", 1.0, vec![10.0, 11.0, 12.0]),
            ch("GPS_Longitude", 1.0, vec![5.0, 6.0, 7.0]),
            ch("GPS_EpochMs", 1.0, vec![1000.0, 2000.0, 3000.0]),
            ch("Fork", 2.0, vec![10.0, 11.0, 12.0, 13.0, 14.0]),
        ]);

        // Act
        let v = h.gps_channel_values("Fork");

        // Assert — one value per fix, in build_gps_track order.
        assert_eq!(v.len(), 3);
        assert_eq!(v, vec![10.0, 12.0, 14.0]);
    }

    #[test]
    fn gps_channel_values_nan_for_absent_channel() {
        // Arrange — two valid fixes, no such channel.
        let h = handle_with(vec![
            ch("GPS_Latitude", 1.0, vec![10.0, 11.0]),
            ch("GPS_Longitude", 1.0, vec![5.0, 6.0]),
            ch("GPS_EpochMs", 1.0, vec![1000.0, 2000.0]),
        ]);

        // Act
        let v = h.gps_channel_values("Nope");

        // Assert — length matches fixes; all NaN.
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn gps_channel_values_nan_past_channel_span() {
        // Arrange — fixes at secs 0,1,2 but target channel only spans sec 0
        // (1 Hz, length 1). Fix 0 → sample 0; fixes 1,2 → out of span → NaN.
        let h = handle_with(vec![
            ch("GPS_Latitude", 1.0, vec![10.0, 11.0, 12.0]),
            ch("GPS_Longitude", 1.0, vec![5.0, 6.0, 7.0]),
            ch("GPS_EpochMs", 1.0, vec![1000.0, 2000.0, 3000.0]),
            ch("Short", 1.0, vec![42.0]),
        ]);

        // Act
        let v = h.gps_channel_values("Short");

        // Assert
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 42.0);
        assert!(v[1].is_nan() && v[2].is_nan());
    }

    #[test]
    fn gps_channel_values_empty_when_no_gps() {
        // Arrange — no GPS channels at all.
        let h = handle_with(vec![ch("Fork", 1.0, vec![1.0, 2.0])]);

        // Act + Assert — no fixes → empty result.
        assert!(h.gps_channel_values("Fork").is_empty());
    }
}
