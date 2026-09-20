//! Shared record-parsing helpers used by the v2 and v3 parsers.
//!
//! Ported from the helper methods in `binary_parser.dart`: the GPS_FIX record
//! parser (identical wire format across v2/v3), the registry-entry readers,
//! typed-value reader, channel-name canonicalization, rate resolution, and the
//! small `_TimeOrigin` / `_GpsAnchor` state holders. Channel sample collection
//! uses an insertion-ordered accumulator so the output channel order matches
//! Dart's `Map<String, List<double>>` (which preserves insertion order).

use std::collections::HashMap;

use crate::parse::reader::ByteReader;
use crate::session::{ChannelRegistryEntry, GapSpan, ParseError, RawColumn};

/// IMU channel names indexed by `[imu_index][axis 0..5]`.
/// Axis order: AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ. See §5.4.
pub const IMU_CHANNEL_NAMES: [[&str; 6]; 3] = [
    ["IMU0_AccelX", "IMU0_AccelY", "IMU0_AccelZ", "IMU0_GyroX", "IMU0_GyroY", "IMU0_GyroZ"],
    ["IMU1_AccelX", "IMU1_AccelY", "IMU1_AccelZ", "IMU1_GyroX", "IMU1_GyroY", "IMU1_GyroZ"],
    ["IMU2_AccelX", "IMU2_AccelY", "IMU2_AccelZ", "IMU2_GyroX", "IMU2_GyroY", "IMU2_GyroZ"],
];

/// Value width in bytes for each data_type code (index = code). See §5.2.
pub const DATA_TYPE_WIDTHS: [usize; 8] = [1, 2, 4, 1, 2, 4, 4, 8];

/// Typed raw buffer backing one accumulator slot. Compact `I16` carries the
/// channel's `(scale, offset)` and stores raw wire values (the IMU hot path);
/// `F64` stores already-physical values (GPS, generic registry channels).
enum RawBuf {
    I16 { data: Vec<i16>, scale: f64, offset: f64 },
    I32 { data: Vec<i32>, scale: f64, offset: f64 },
    F32 { data: Vec<f32>, scale: f64, offset: f64 },
    F64(Vec<f64>),
}

impl RawBuf {
    fn into_column(self) -> RawColumn {
        match self {
            RawBuf::I16 { data, scale, offset } => RawColumn::I16 { data, scale, offset },
            RawBuf::I32 { data, scale, offset } => RawColumn::I32 { data, scale, offset },
            RawBuf::F32 { data, scale, offset } => RawColumn::F32 { data, scale, offset },
            RawBuf::F64(data) => RawColumn::F64(data),
        }
    }
}

/// Insertion-ordered, typed channel sample accumulator.
///
/// Mirrors Dart `Map<String, List<double>>.putIfAbsent(..).add(..)` semantics
/// with guaranteed first-seen ordering of channel names. Each slot is either a
/// compact `i16` buffer (IMU axes — raw values + `(scale, offset)`) or an `f64`
/// buffer (GPS, generic channels — already-physical values), so the resident
/// IMU column is 2 bytes/sample instead of 8.
#[derive(Default)]
pub struct ChannelAccumulator {
    order: Vec<String>,
    index: HashMap<String, usize>,
    bufs: Vec<RawBuf>,
}

impl ChannelAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    fn new_slot(&mut self, name: &str, buf: RawBuf) -> usize {
        let i = self.order.len();
        self.order.push(name.to_string());
        self.index.insert(name.to_string(), i);
        self.bufs.push(buf);
        i
    }

    /// Returns the bucket index for `name`, creating an **f64** bucket on first
    /// sight. Used by GPS / generic channels that push already-physical values.
    ///
    /// Hashes the name at most once per channel; callers cache the returned
    /// index and route subsequent samples through [`push_at`](Self::push_at),
    /// keeping the per-sample hot path integer-keyed (SPEC §5.2).
    pub fn slot_for(&mut self, name: &str) -> usize {
        match self.index.get(name) {
            Some(&i) => i,
            None => self.new_slot(name, RawBuf::F64(Vec::new())),
        }
    }

    /// Returns the bucket index for `name`, creating a compact **i16** bucket
    /// (carrying `(scale, offset)`) on first sight. Subsequent samples push raw
    /// i16 via [`push_i16_at`](Self::push_i16_at); physical f64 is materialized
    /// lazily as `(raw as f64) * scale + offset`. The IMU hot path.
    pub fn slot_for_i16(&mut self, name: &str, scale: f64, offset: f64) -> usize {
        match self.index.get(name) {
            Some(&i) => i,
            None => self.new_slot(name, RawBuf::I16 { data: Vec::new(), scale, offset }),
        }
    }

    /// Compact **i32** bucket for `name` (i32 registry channels — e.g. raw GPS
    /// coordinate-scale sensors). Subsequent samples push raw i32 via
    /// [`push_i32_at`](Self::push_i32_at).
    pub fn slot_for_i32(&mut self, name: &str, scale: f64, offset: f64) -> usize {
        match self.index.get(name) {
            Some(&i) => i,
            None => self.new_slot(name, RawBuf::I32 { data: Vec::new(), scale, offset }),
        }
    }

    /// Compact **f32** bucket for `name` (f32 registry channels). Subsequent
    /// samples push raw f32 via [`push_f32_at`](Self::push_f32_at).
    pub fn slot_for_f32(&mut self, name: &str, scale: f64, offset: f64) -> usize {
        match self.index.get(name) {
            Some(&i) => i,
            None => self.new_slot(name, RawBuf::F32 { data: Vec::new(), scale, offset }),
        }
    }

    /// Appends a physical `value` to the f64 bucket at `slot` — no hashing.
    /// `slot` MUST be a valid index returned by [`slot_for`](Self::slot_for).
    pub fn push_at(&mut self, slot: usize, value: f64) {
        match &mut self.bufs[slot] {
            RawBuf::F64(data) => data.push(value),
            _ => unreachable!("push_at: f64 into a compact (i16/i32/f32) slot"),
        }
    }

    /// Appends a raw `value` to the i16 bucket at `slot` — no hashing, no scale.
    /// `slot` MUST be a valid index returned by [`slot_for_i16`](Self::slot_for_i16).
    pub fn push_i16_at(&mut self, slot: usize, value: i16) {
        match &mut self.bufs[slot] {
            RawBuf::I16 { data, .. } => data.push(value),
            _ => unreachable!("push_i16_at: wrong slot type"),
        }
    }

    /// Appends a raw `value` to the i32 bucket at `slot`. `slot` MUST come from
    /// [`slot_for_i32`](Self::slot_for_i32).
    pub fn push_i32_at(&mut self, slot: usize, value: i32) {
        match &mut self.bufs[slot] {
            RawBuf::I32 { data, .. } => data.push(value),
            _ => unreachable!("push_i32_at: wrong slot type"),
        }
    }

    /// Appends a raw `value` to the f32 bucket at `slot`. `slot` MUST come from
    /// [`slot_for_f32`](Self::slot_for_f32).
    pub fn push_f32_at(&mut self, slot: usize, value: f32) {
        match &mut self.bufs[slot] {
            RawBuf::F32 { data, .. } => data.push(value),
            _ => unreachable!("push_f32_at: wrong slot type"),
        }
    }

    /// Appends a physical `value` to channel `name`, creating an f64 channel on
    /// first use. Equivalent to `let s = self.slot_for(name); self.push_at(s, value);`.
    pub fn push(&mut self, name: &str, value: f64) {
        let slot = self.slot_for(name);
        self.push_at(slot, value);
    }

    /// `true` if `name` has at least one sample.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Consumes the accumulator into `(name, column)` pairs in first-seen order.
    pub fn into_entries(self) -> Vec<(String, RawColumn)> {
        self.order
            .into_iter()
            .zip(self.bufs.into_iter().map(RawBuf::into_column))
            .collect()
    }
}

/// Tracks the earliest record `timestamp_us` seen — the session time origin
/// (t=0) for event-driven channel sample times. Mirrors Dart `_TimeOrigin`.
#[derive(Default)]
pub struct TimeOrigin {
    pub min_us: Option<i64>,
}

impl TimeOrigin {
    /// Folds `ts_us` into the running minimum.
    pub fn observe(&mut self, ts_us: i64) {
        match self.min_us {
            Some(cur) if ts_us >= cur => {}
            _ => self.min_us = Some(ts_us),
        }
    }
}

/// Wall-clock anchor captured from the first non-zero GPS_FIX. Mirrors Dart
/// `_GpsAnchor`; used to back-fill `session_start_utc_ms` per §5.6.
#[derive(Default)]
pub struct GpsAnchor {
    pub gps_epoch_ms: Option<i64>,
    pub device_ts_us: Option<i64>,
}

/// Renders bytes as a lowercase hex string (UUID/device-id encoding).
pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Decodes a null-terminated ASCII string from a fixed-width field.
pub fn null_term_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Maps a legacy registry name to its canonical form. Currently the pre-rename
/// HRM channel `HeartRate` surfaces as `HR_BPM`. See §5.2.
pub fn canonical_channel_name(raw: &str) -> String {
    if raw == "HeartRate" {
        "HR_BPM".to_string()
    } else {
        raw.to_string()
    }
}

/// Reads a typed value per the registry `data_type` code, widened to `f64`.
pub fn read_typed_value(reader: &mut ByteReader, data_type: u8) -> Result<f64, ParseError> {
    Ok(match data_type {
        0 => reader.u8("u8 value")? as f64,
        1 => reader.u16("u16 value")? as f64,
        2 => reader.u32("u32 value")? as f64,
        3 => reader.i8("i8 value")? as f64,
        4 => reader.i16("i16 value")? as f64,
        5 => reader.i32("i32 value")? as f64,
        6 => reader.f32("f32 value")? as f64,
        7 => reader.f64("f64 value")?,
        other => {
            return Err(ParseError::TruncatedRecord(format!(
                "Unknown data type {other}"
            )))
        }
    })
}

/// Reads a 40-byte v3 registry entry (explicit `scale`/`offset` after rate).
pub fn read_registry_entry_v3(reader: &mut ByteReader) -> Result<ChannelRegistryEntry, ParseError> {
    let channel_id = reader.u8("registry channel_id")?;
    let data_type = reader.u8("registry data_type")?;
    let sample_rate_hz = reader.u16("registry sample_rate_hz")?;
    let scale = reader.f32("registry scale")? as f64;
    let offset = reader.f32("registry offset")? as f64;
    let name_bytes = reader.bytes(20, "registry name")?;
    let name = canonical_channel_name(&null_term_str(name_bytes));
    let unit_bytes = reader.bytes(8, "registry units")?;
    let units = null_term_str(unit_bytes);
    Ok(ChannelRegistryEntry {
        channel_id,
        data_type,
        sample_rate_hz,
        scale,
        offset,
        name,
        units,
    })
}

/// Parses a GPS_FIX (0x02) record — identical wire format in v2 and v3.
///
/// Emits the eight GPS channels, with lat/lon/alt/heading baked to physical
/// units at parse time (ruling R27) and no metadata scale/offset. Optionally seeds the
/// `anchor` from the first non-zero `gps_epoch_ms` (for §5.6 back-fill) and
/// folds `device_timestamp_us` into `origin` (the event-time zero). Pushes
/// this fix's own `device_ts_us` into `gps_ts` only once every fallible field
/// read below has succeeded — every GPS channel shares this one per-fix
/// timestamp as its `t_us` source, and a mid-record truncation must not leave
/// `gps_ts` ahead of the `GPS_*` columns (C1 §2's mandatory
/// `t_us.len() == column.len()`).
pub fn parse_gps_record(
    reader: &mut ByteReader,
    payload_len: usize,
    out: &mut ChannelAccumulator,
    anchor: Option<&mut GpsAnchor>,
    origin: Option<&mut TimeOrigin>,
    gps_ts: &mut Vec<i64>,
) -> Result<(), ParseError> {
    let payload_start = reader.position();
    let gps_epoch_ms = reader.i64("gps_epoch_ms")?;
    let device_ts_us = reader.i64("device_timestamp_us")?;
    if let Some(o) = origin {
        o.observe(device_ts_us);
    }
    if let Some(a) = anchor {
        if a.gps_epoch_ms.is_none() && gps_epoch_ms > 0 {
            a.gps_epoch_ms = Some(gps_epoch_ms);
            a.device_ts_us = Some(device_ts_us);
        }
    }
    let latitude = reader.i32("latitude")?;
    let longitude = reader.i32("longitude")?;
    let altitude = reader.i16("altitude")?;
    let speed = reader.u16("speed")?;
    let heading = reader.u16("heading")?;
    let fix_quality = reader.u8("fix_quality")?;
    let satellites = reader.u8("satellites")?;

    // Pushed only once every fallible field above has succeeded, mirroring
    // `parse_channel`'s `channel_ts_us` ordering (v3.rs) — a `.idl0` buffer
    // truncated mid-GPS-record must not leave `gps_ts` one entry ahead of the
    // `GPS_*` columns below (C1 §2's mandatory `t_us.len() == column.len()`).
    gps_ts.push(device_ts_us);

    out.push("GPS_EpochMs", gps_epoch_ms as f64);
    // GPS_Latitude/Longitude/Altitude/Heading are baked to physical units at
    // parse time (ruling R27), the same convention already used by
    // `WheelFront`/`HR_RR`: no metadata `scale`/`offset` key, the division
    // happens once here instead of in every consumer.
    out.push("GPS_Latitude", latitude as f64 * 1e-7);
    out.push("GPS_Longitude", longitude as f64 * 1e-7);
    out.push("GPS_Altitude", altitude as f64 * 0.1);
    // GPS_SpeedKmh is the one GPS-fix channel returned via metadata scale. The
    // firmware logs km/h × 100 (§5.6); a 0.01 scale on a compact i32 column makes
    // `materialize()` yield km/h, so Distance synthesis, math expressions, and the
    // colour-by-channel scale all get physical speed without each consumer
    // dividing (§5.7).
    let speed_slot = out.slot_for_i32("GPS_SpeedKmh", 0.01, 0.0);
    out.push_i32_at(speed_slot, speed as i32);
    out.push("GPS_Heading", heading as f64 * 0.01);
    out.push("GPS_FixQuality", fix_quality as f64);
    out.push("GPS_Satellites", satellites as f64);

    let consumed = reader.position() - payload_start;
    if consumed < payload_len {
        reader.skip(payload_len - consumed, "GPS payload remainder")?;
    }
    Ok(())
}

/// Resolves a channel's sample rate. **All** IMU channels share the single
/// `imu_nominal` rate (drop reconciliation — design §4.1; the dropped samples
/// that made each IMU's `(n-1)/span` differ are now filled onto a shared grid).
/// GPS channels use the header GPS rate; registry channels use their declared
/// rate; otherwise 0. Mirrors Dart `_resolveRate`.
pub fn resolve_rate(
    channel_id: &str,
    gps_hz: u8,
    registry: &HashMap<u8, ChannelRegistryEntry>,
    imu_nominal: f64,
) -> f64 {
    if imu_index_of(channel_id).is_some() {
        return imu_nominal;
    }
    if channel_id.starts_with("GPS") {
        return gps_hz as f64;
    }
    for e in registry.values() {
        if e.name == channel_id {
            return e.sample_rate_hz as f64;
        }
    }
    0.0
}

/// Nominal IMU grid period in microseconds from the configured ODR, mirroring
/// the firmware back-count exactly: `period_us = if odr > 0 { 1_000_000 / odr }
/// else { 10_000 }` (integer division — using a float period would slowly
/// misalign placement against the firmware's integer-period stamping). See
/// SPEC §5.5 and the drop-reconciliation design §3.
pub fn imu_period_us(odr_hz: u16) -> i64 {
    if odr_hz > 0 {
        1_000_000 / odr_hz as i64
    } else {
        10_000
    }
}

/// Nominal IMU sample rate in Hz: `1e6 / period_us`. The single rate shared by
/// every IMU channel (the per-IMU `(n-1)/span` value is the drop-induced
/// artifact this replaces — design §4.1).
pub fn imu_nominal_rate(period_us: i64) -> f64 {
    1e6 / period_us as f64
}

/// Maps a generic registry channel name to its `source_kind` token (contract
/// C1 §4.2). Falls back to the lower-cased channel name for any future
/// sensor the registry adds without a schema change (SPEC §5.2's
/// forward-compatibility philosophy, carried into C1 §4.2's `source_kind`
/// definition verbatim).
pub fn generic_source_kind(channel_id: &str) -> String {
    match channel_id {
        "WheelFront" => "wheel_front",
        "WheelRear" => "wheel_rear",
        "PressureFront" => "pressure_front",
        "PressureRear" => "pressure_rear",
        "HR_BPM" => "hr_bpm",
        "HR_RR" => "hr_rr",
        other => return other.to_lowercase(),
    }
    .to_string()
}

/// IMU index `0..=2` parsed from an `IMU{n}_` channel name, else `None`.
pub fn imu_index_of(name: &str) -> Option<usize> {
    let b = name.as_bytes();
    if b.len() >= 5 && &b[0..3] == b"IMU" && b[4] == b'_' {
        match b[3] {
            b'0' => Some(0),
            b'1' => Some(1),
            b'2' => Some(2),
            _ => None,
        }
    } else {
        None
    }
}

/// Rebuilds one IMU axis column onto the shared nominal grid in a single O(n)
/// pass. `gaps_received` is `(received_index, missing)` per drop (sorted by
/// received index). Front is padded to the session `t0` with `leading` held
/// copies of the first value; interior drops are linearly interpolated in raw
/// `i16` space between the bracketing real samples; the tail is padded to
/// `target_len` with held copies of the last value. Design §4.2.
///
/// Since ruling R241 the plan passes each IMU's **own** occupied length as
/// `target_len`, so the tail branch is a no-op on the import path: a stream
/// ends at its last recorded sample. The branch stays because the function is
/// public and its contract ("pad to `target_len`") is what a caller asking for
/// a longer grid would still get.
pub fn rebuild_i16(
    raw: &[i16],
    gaps_received: &[(usize, usize)],
    leading: usize,
    target_len: usize,
) -> Vec<i16> {
    if raw.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<i16> = Vec::with_capacity(target_len);
    // Leading pad: held copies of the first value (no bracket to interpolate).
    let first_val = raw[0];
    out.resize(leading, first_val);
    out.push(first_val);

    // Walk received samples, inserting `missing` linear fills before each drop.
    let mut gi = 0usize;
    for k in 1..raw.len() {
        let missing = if gi < gaps_received.len() && gaps_received[gi].0 == k {
            let m = gaps_received[gi].1;
            gi += 1;
            m
        } else {
            0
        };
        if missing >= 1 {
            let v0 = raw[k - 1] as f64;
            let v1 = raw[k] as f64;
            let denom = (missing + 1) as f64;
            for j in 1..=missing {
                out.push((v0 + (v1 - v0) * (j as f64) / denom).round() as i16);
            }
        }
        out.push(raw[k]);
    }

    // Tail pad: held copies of the last value out to the session-wide length.
    if let Some(&last_val) = out.last() {
        if out.len() < target_len {
            out.resize(target_len, last_val);
        }
    }
    out
}

/// Builds the grid-slot [`GapSpan`] list for one IMU from its leading pad, drop
/// list, occupied length, and the grid length `target_len`. Shared across the
/// IMU's six axes. Design §4.2/§4.3. With `target_len == occupied` — what the
/// plan passes since ruling R241 — there is no trailing span, because there is
/// no trailing pad to mark.
pub fn build_spans(
    leading: usize,
    gaps_received: &[(usize, usize)],
    occupied: usize,
    target_len: usize,
) -> Vec<GapSpan> {
    let mut spans = Vec::new();
    if leading > 0 {
        spans.push(GapSpan { start: 0, len: leading });
    }
    // Interior fills. Sample `recv_idx` sits at slot `leading + recv_idx +
    // cum_missing`; its fills occupy the `missing` slots immediately before it.
    let mut cum_missing = 0usize;
    for &(recv_idx, missing) in gaps_received {
        spans.push(GapSpan { start: leading + recv_idx + cum_missing, len: missing });
        cum_missing += missing;
    }
    if occupied < target_len {
        spans.push(GapSpan { start: occupied, len: target_len - occupied });
    }
    spans
}

/// Drop-reconciliation plan for the three IMUs, derived from each IMU's
/// **burst-seam-corrected** stamp sequence (contract C1 §3.3 — gap detection
/// runs on the corrected grid, never the nominal one; C1 §3.3/§8 item 1,
/// ruled). Anchors each IMU's leading pad at the earliest corrected
/// first-sample across every reconciled IMU (`t0`, absolute device-clock
/// domain — the caller subtracts the session-wide origin uniformly afterward,
/// the same way it already does for every non-IMU channel).
///
/// **The grid is shared at its start, not at its end (ruling R241).** Every
/// IMU's slot 0 still sits at `t0`, so the three streams remain index-aligned
/// from the front; but each ends at its own last recorded sample. The old
/// session-wide length padded an IMU that stopped early out to the longest
/// stream's, with a forward-extrapolated tail that pushed `duration_ms` and
/// the union `t` axis past the end of the recording (up to ~52 s on a real
/// session). The leading pad is kept: it is bounded by the spread of the
/// IMUs' start instants — one FIFO drain, milliseconds — where the tail was
/// bounded by nothing, and dropping it would also drop the front-of-stream
/// index alignment C1 §3.3's reconciliation premise still rests on. Both
/// pads are marked in `gaps` either way.
pub struct ImuGridPlan {
    /// Per-IMU corrected period (µs): `effective_period_us` from seam
    /// correction, or `nominal_period_us` when unreconciled.
    period_us: [i64; 3],
    /// Leading pad length (slots) per IMU — `round((first - t0) / period)`.
    leading: [usize; 3],
    /// Whether each IMU has ≥2 samples and is therefore reconciled.
    reconciled: [bool; 3],
    /// Per-IMU grid length: that IMU's own occupied length — leading pad +
    /// received samples + interior fills — and nothing beyond it (ruling
    /// R241). There is no session-wide length: an IMU that stopped early
    /// ends at its own last recorded sample rather than being padded out to
    /// the longest stream's, which used to put up to ~52 s of synthesized
    /// tail into `duration_ms` and the union `t` axis.
    grid_len: [usize; 3],
    /// Per-IMU drop list `(received_index, missing)`, used by the rebuild.
    gaps_received: [Vec<(usize, usize)>; 3],
    /// Per-IMU grid-slot gap spans, shared across that IMU's six axes.
    spans: [Vec<GapSpan>; 3],
    /// The corrected (pre-grid, dense-but-ungapped) stamps this plan was
    /// built from — the source of each kept slot's `t_us`, and of the grid
    /// itself.
    corrected: [Vec<i64>; 3],
    /// The **raw, pre-correction** device stamps, index-aligned with
    /// [`Self::corrected`] — the source of each kept slot's `t_recorded_us`
    /// (contract C1 §3.2, ruling R240). Kept separately because §3.2's column
    /// is verbatim by definition: once the two are the same array, the seam
    /// detector `fetch_seams` runs has nothing left to detect against (C1
    /// §3.3's step 1 reads recorded deltas, whose within-burst spacing the
    /// correction deliberately flattens), and no column anywhere retains a
    /// pre-correction stamp.
    raw: [Vec<i64>; 3],
}

impl ImuGridPlan {
    /// Builds the plan. `corrected[i]` is IMU `i`'s burst-seam-corrected
    /// stamp sequence (`seam_correction::correct_burst_seams(..).corrected_us`,
    /// or the raw stamps verbatim for an IMU with <2 samples — correction is
    /// a no-op there per the burst-correction module). `effective_period_us[i]`
    /// is that same call's `effective_period_us`. `raw[i]` is the same IMU's
    /// **verbatim** recorded stamp sequence — the input correction ran on,
    /// index-aligned with `corrected[i]` and the same length — which becomes
    /// `t_recorded_us` (C1 §3.2, R240). `nominal_period_us` is the fallback
    /// for an IMU with <2 samples (no correction ran, so no effective period
    /// exists; `raw` and `corrected` are then the same values).
    pub fn build_from_corrected(
        corrected: [Vec<i64>; 3],
        raw: [Vec<i64>; 3],
        effective_period_us: [i64; 3],
        nominal_period_us: i64,
    ) -> Self {
        let mut period_us = [nominal_period_us; 3];
        let mut leading = [0usize; 3];
        let mut occupied = [0usize; 3];
        let mut reconciled = [false; 3];
        let mut gaps_received: [Vec<(usize, usize)>; 3] = Default::default();

        let t0 = corrected
            .iter()
            .filter(|c| c.len() >= 2)
            .filter_map(|c| c.first().copied())
            .min();

        for i in 0..3 {
            if corrected[i].len() < 2 {
                continue;
            }
            let Some(t0v) = t0 else { continue };
            reconciled[i] = true;
            period_us[i] = effective_period_us[i];
            let first = corrected[i][0];
            leading[i] = (((first - t0v) as f64) / period_us[i] as f64).round().max(0.0) as usize;

            // Gap detection, once, against the corrected stamps and this
            // IMU's own effective period — same absolute-slot rule as the
            // pre-idl1 hot loop (SPEC §15.2), just relocated per C1 §3.3.
            // Corrected stamps are already strictly increasing (burst
            // correction's monotonicity guarantee), so there is no
            // backward-step case to handle here — that was filtered
            // pre-correction, in the parser's hot loop.
            let mut last_abs_slot: i64 = 0;
            for k in 1..corrected[i].len() {
                let delta = corrected[i][k] - corrected[i][k - 1];
                let abs_slot = if delta == period_us[i] {
                    last_abs_slot + 1
                } else {
                    ((corrected[i][k] - first) as f64 / period_us[i] as f64).round() as i64
                };
                let missing = (abs_slot - last_abs_slot - 1).max(0) as usize;
                if missing >= 1 {
                    gaps_received[i].push((k, missing));
                }
                last_abs_slot = abs_slot;
            }
            let total_missing: usize = gaps_received[i].iter().map(|g| g.1).sum();
            occupied[i] = leading[i] + corrected[i].len() + total_missing;
        }

        // No session-wide `target_len` (R241): each IMU's grid is exactly its
        // own occupied length, so `build_spans` never appends a trailing span
        // and the rebuilds never pad past the last real sample.
        let grid_len = occupied;
        let mut spans: [Vec<GapSpan>; 3] = Default::default();
        for i in 0..3 {
            if reconciled[i] {
                spans[i] = build_spans(leading[i], &gaps_received[i], occupied[i], grid_len[i]);
            }
        }

        ImuGridPlan {
            period_us,
            leading,
            reconciled,
            gaps_received,
            spans,
            corrected,
            raw,
            grid_len,
        }
    }

    /// Reconciles one channel by name: rebuilds its value column onto the
    /// grid (unchanged fill logic — held-edge pads, linear interior fills),
    /// and returns `(column, t_us, t_recorded_us, gaps)` — `t_us` is the
    /// **recorded** time of each slot (absolute device-clock µs, same domain
    /// as `t0`): the slot's own burst-seam-corrected stamp wherever the slot
    /// holds a real sample, and a neighbour-interpolated placeholder inside a
    /// gap ([`slot_times_from_corrected`], C1 §3.1/§3.5 invariant 4 — a
    /// sample's time is never `i / nominal_rate_hz`). `t_recorded_us` is
    /// dense and equals the **raw, pre-correction** device stamp at every kept
    /// slot (C1 §3.2, ruling R240 — it is not a second copy of `t_us`), and is
    /// **unspecified at every slot inside `gaps`** (callers must consult
    /// `gaps`, never infer nullness from content — see this module's doc for
    /// the reasoning). Every non-IMU channel passes through with
    /// `t_us`/`t_recorded_us` empty and an empty gap list (the caller already
    /// has real `t_us` for those from its own recorded-time bookkeeping).
    pub fn reconcile(&self, name: &str, column: RawColumn) -> (RawColumn, Vec<i64>, Vec<i64>, Vec<GapSpan>) {
        match imu_index_of(name) {
            Some(i) if self.reconciled[i] => {
                let t_us = slot_times_from_corrected(
                    &self.corrected[i], &self.gaps_received[i], self.leading[i], self.grid_len[i], self.period_us[i],
                );
                let t_recorded_us = rebuild_i64_grid_or_real(
                    &self.raw[i], &self.gaps_received[i], self.leading[i], self.grid_len[i], &t_us,
                );
                match column {
                    RawColumn::I16 { data, scale, offset } => {
                        let rebuilt = rebuild_i16(&data, &self.gaps_received[i], self.leading[i], self.grid_len[i]);
                        (RawColumn::I16 { data: rebuilt, scale, offset }, t_us, t_recorded_us, self.spans[i].clone())
                    }
                    // IMU axes are always compact i16 on the parse path; pass any
                    // other variant through untouched rather than panic.
                    other => (other, t_us, t_recorded_us, self.spans[i].clone()),
                }
            }
            _ => (column, Vec::new(), Vec::new(), Vec::new()),
        }
    }
}

/// Builds one IMU's dense per-slot time axis (`t_us`, absolute device-clock
/// µs) from its burst-seam-corrected stamps — contract C1 §3.1 ("`t_us[i] =
/// corrected_timestamp_us[i] − t0_us`"; the caller subtracts the session-wide
/// `t0_us`) and §3.5 invariant 4 ("`nominal_rate_hz` never derives a sample's
/// time, anywhere"). Walks the same fill pattern as [`rebuild_i16`]:
///
/// - a slot holding a real sample takes that sample's corrected stamp
///   verbatim, so a recorded cadence that drifts from the session median, and
///   a multi-second FIFO dropout, both appear in `t_us` exactly as recorded;
/// - an **interior** gap slot is linearly interpolated (rounded to µs) between
///   the corrected stamps bracketing the drop, so `t_us` stays monotone across
///   the dropout and each fill sits between its neighbours;
/// - the **leading** pad extrapolates backward from the first corrected stamp
///   at `period_us`, and the **trailing** pad (when a caller asks for a grid
///   longer than the stream — not the import path since ruling R241, where
///   `target_len` is the stream's own occupied length) forward from the last
///   one — there is no bracketing stamp on those edges, exactly as
///   `rebuild_i16` has no bracketing value there.
///
/// A final pass enforces C1 §3.5 invariant 1 (strictly increasing `t_us`): a
/// slot that does not advance past its predecessor is clamped to
/// `previous + 1 µs`. Burst-seam correction already guarantees strictly
/// increasing corrected stamps, so this is a defence against a degenerate
/// interpolation (a sub-slot-per-µs drop), never the normal path.
///
/// `period_us` is that IMU's corrected `effective_period_us` (µs).
fn slot_times_from_corrected(
    corrected: &[i64],
    gaps_received: &[(usize, usize)],
    leading: usize,
    target_len: usize,
    period_us: i64,
) -> Vec<i64> {
    if corrected.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<i64> = Vec::with_capacity(target_len);

    // Leading pad: backward extrapolation from the first corrected stamp.
    let first = corrected[0];
    for slot in 0..leading {
        out.push(first - ((leading - slot) as i64) * period_us);
    }
    out.push(first);

    let mut gi = 0usize;
    for k in 1..corrected.len() {
        let missing = if gi < gaps_received.len() && gaps_received[gi].0 == k {
            let m = gaps_received[gi].1;
            gi += 1;
            m
        } else {
            0
        };
        if missing >= 1 {
            let t0 = corrected[k - 1] as f64;
            let t1 = corrected[k] as f64;
            let denom = (missing + 1) as f64;
            for j in 1..=missing {
                out.push((t0 + (t1 - t0) * (j as f64) / denom).round() as i64);
            }
        }
        out.push(corrected[k]);
    }

    // Tail pad: forward extrapolation from the last corrected stamp out to
    // `target_len`. Dead on the import path since R241 (`target_len` is this
    // stream's own length there); see this function's doc.
    if out.len() < target_len {
        let last = *corrected.last().unwrap_or(&first);
        let occupied = out.len();
        for j in 1..=(target_len - occupied) {
            out.push(last + (j as i64) * period_us);
        }
    }

    for i in 1..out.len() {
        if out[i] <= out[i - 1] {
            out[i] = out[i - 1] + 1;
        }
    }
    out
}

/// Builds a dense `t_recorded_us` array for one IMU's grid: at a slot that
/// holds a real (non-synthesized) sample, that sample's **verbatim recorded**
/// timestamp (`stamps` is the raw, pre-correction sequence — contract C1 §3.2,
/// ruling R240); at every other slot (leading pad, interior fill, trailing pad
/// — all covered by a `GapSpan`), the corresponding value from `slot_t_us` (a
/// harmless, documented placeholder — see [`ImuGridPlan::reconcile`]'s doc; a
/// synthesized slot has no recorded stamp to carry, and the gap list, not the
/// content, is what says so). Mirrors `rebuild_i16`'s fill-pattern walk
/// exactly, but placing real/placeholder timestamps rather than interpolating
/// values.
///
/// `stamps` is indexed in the same received-sample space as `gaps_received`,
/// which is why the raw and corrected arrays must stay index-aligned.
fn rebuild_i64_grid_or_real(
    stamps: &[i64],
    gaps_received: &[(usize, usize)],
    leading: usize,
    target_len: usize,
    slot_t_us: &[i64],
) -> Vec<i64> {
    if stamps.is_empty() {
        return Vec::new();
    }
    // Real/placeholder is tracked per index (not by sentinel-value
    // collision), so a real recorded timestamp of exactly `0` — never
    // physically possible for a device clock, but not worth relying on
    // implicitly — cannot be mistaken for a placeholder.
    let mut out: Vec<i64> = Vec::with_capacity(target_len);
    let mut is_real: Vec<bool> = Vec::with_capacity(target_len);
    out.resize(leading, 0);
    is_real.resize(leading, false);
    out.push(stamps[0]);
    is_real.push(true);
    let mut gi = 0usize;
    for k in 1..stamps.len() {
        let missing = if gi < gaps_received.len() && gaps_received[gi].0 == k {
            let m = gaps_received[gi].1;
            gi += 1;
            m
        } else {
            0
        };
        for _ in 0..missing {
            out.push(0);
            is_real.push(false);
        }
        out.push(stamps[k]);
        is_real.push(true);
    }
    if out.len() < target_len {
        out.resize(target_len, 0);
        is_real.resize(target_len, false);
    }
    for (i, real) in is_real.iter().enumerate() {
        if !*real && i < slot_t_us.len() {
            out[i] = slot_t_us[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_preserves_first_seen_order() {
        // Arrange
        let mut acc = ChannelAccumulator::new();

        // Act — push out of alphabetical order
        acc.push("Zebra", 1.0);
        acc.push("Apple", 2.0);
        acc.push("Zebra", 3.0);

        // Assert — order is first-seen, samples grouped
        let entries = acc.into_entries();
        assert_eq!(entries[0].0, "Zebra");
        assert_eq!(entries[0].1.materialize(), vec![1.0, 3.0]);
        assert_eq!(entries[1].0, "Apple");
        assert_eq!(entries[1].1.materialize(), vec![2.0]);
    }

    #[test]
    fn slot_for_caches_index_and_push_at_appends_without_rehash() {
        // Arrange
        let mut acc = ChannelAccumulator::new();

        // Act — first sight assigns a slot; same name returns the same slot.
        let zebra = acc.slot_for("Zebra");
        let apple = acc.slot_for("Apple");
        assert_eq!(acc.slot_for("Zebra"), zebra);
        acc.push_at(zebra, 1.0);
        acc.push_at(apple, 2.0);
        acc.push_at(zebra, 3.0);

        // Assert — first-seen order preserved, samples grouped by slot, identical
        // to what push() would have produced.
        let entries = acc.into_entries();
        assert_eq!(entries[0].0, "Zebra");
        assert_eq!(entries[0].1.materialize(), vec![1.0, 3.0]);
        assert_eq!(entries[1].0, "Apple");
        assert_eq!(entries[1].1.materialize(), vec![2.0]);
    }

    #[test]
    fn typed_slots_yield_compact_columns_with_scale_offset() {
        // Arrange — one i16, one i32, one f32, one f64 channel.
        let mut acc = ChannelAccumulator::new();
        let i16s = acc.slot_for_i16("Accel", 0.5, 1.0);
        let i32s = acc.slot_for_i32("Coord", 2.0, 0.0);
        let f32s = acc.slot_for_f32("Temp", 1.0, 0.0);

        // Act
        acc.push_i16_at(i16s, 100);
        acc.push_i32_at(i32s, 1_000_000);
        acc.push_f32_at(f32s, 1.5);
        acc.push("Brake", 7.0); // f64 path

        // Assert — variants and materialized physical values match the formula.
        let entries = acc.into_entries();
        assert!(matches!(entries[0].1, RawColumn::I16 { .. }));
        assert_eq!(entries[0].1.materialize(), vec![100.0 * 0.5 + 1.0]);
        assert!(matches!(entries[1].1, RawColumn::I32 { .. }));
        assert_eq!(entries[1].1.materialize(), vec![1_000_000.0 * 2.0]);
        assert!(matches!(entries[2].1, RawColumn::F32 { .. }));
        assert_eq!(entries[2].1.materialize(), vec![1.5_f32 as f64]);
        assert!(matches!(entries[3].1, RawColumn::F64(_)));
        assert_eq!(entries[3].1.materialize(), vec![7.0]);
    }

    #[test]
    fn canonical_name_maps_heartrate_to_hr_bpm() {
        // Arrange / Act / Assert
        assert_eq!(canonical_channel_name("HeartRate"), "HR_BPM");
        assert_eq!(canonical_channel_name("WheelFront"), "WheelFront");
    }

    #[test]
    fn null_term_str_stops_at_first_zero() {
        // Arrange
        let bytes = b"IMU0_AccelX\0\0\0\0";

        // Act + Assert
        assert_eq!(null_term_str(bytes), "IMU0_AccelX");
    }

    #[test]
    fn imu_period_us_integer_divides_like_firmware() {
        // Arrange / Act / Assert — integer division, byte-for-byte the firmware
        // back-count step (833 Hz → 1200 µs, not 1200.48).
        assert_eq!(imu_period_us(833), 1200);
        assert_eq!(imu_period_us(800), 1250);
        assert_eq!(imu_period_us(1000), 1000);
    }

    #[test]
    fn imu_period_us_zero_odr_falls_back_to_100hz() {
        // Arrange / Act / Assert — odr 0 → 10_000 µs (100 Hz), the firmware fallback.
        assert_eq!(imu_period_us(0), 10_000);
    }

    #[test]
    fn imu_nominal_rate_is_reciprocal_of_period() {
        // Arrange / Act / Assert — 1250 µs → exactly 800 Hz; 1200 µs → 833.33 Hz.
        assert_eq!(imu_nominal_rate(1250), 800.0);
        assert!((imu_nominal_rate(1200) - 1e6 / 1200.0).abs() < 1e-9);
    }

    #[test]
    fn imu_index_of_parses_prefix_else_none() {
        // Arrange / Act / Assert
        assert_eq!(imu_index_of("IMU0_AccelX"), Some(0));
        assert_eq!(imu_index_of("IMU1_GyroZ"), Some(1));
        assert_eq!(imu_index_of("IMU2_AccelZ"), Some(2));
        assert_eq!(imu_index_of("IMU3_AccelX"), None); // only 0..=2 exist
        assert_eq!(imu_index_of("GPS_Latitude"), None);
        assert_eq!(imu_index_of("IMU"), None);
    }

    #[test]
    fn rebuild_i16_interpolates_a_single_interior_gap() {
        // Arrange — 3 received [0, 30, 40]; 2 missing before received index 1.
        // leading 0, target = 0 + 3 + 2 = 5.
        let raw = vec![0i16, 30, 40];
        let gaps = vec![(1usize, 2usize)];

        // Act
        let out = rebuild_i16(&raw, &gaps, 0, 5);

        // Assert — linear fill 0→30 over 3 steps inserts 10, 20.
        assert_eq!(out, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn rebuild_i16_pads_front_and_tail_with_held_edge_values() {
        // Arrange — 2 received [7, 9], leading 2, target 6, no interior gaps.
        let raw = vec![7i16, 9];

        // Act
        let out = rebuild_i16(&raw, &[], 2, 6);

        // Assert — front holds 7 (no bracket to interpolate), tail holds 9.
        assert_eq!(out, vec![7, 7, 7, 9, 9, 9]);
    }

    #[test]
    fn build_spans_records_leading_interior_and_tail_runs() {
        // Arrange — leading 2; one interior drop of 2 missing before received
        // index 1; occupied = 2 + 3 + 2 = 7; target 9.
        let gaps = vec![(1usize, 2usize)];

        // Act
        let spans = build_spans(2, &gaps, 7, 9);

        // Assert — leading {0,2}; interior fill at slot 2+1+0 = 3 len 2; tail {7,2}.
        assert_eq!(
            spans,
            vec![
                GapSpan { start: 0, len: 2 },
                GapSpan { start: 3, len: 2 },
                GapSpan { start: 7, len: 2 },
            ]
        );
    }

    /// A plan whose raw and corrected stamps are the same sequence — the
    /// shape of an IMU the correction was a no-op on (C1 §3.3: a stream
    /// already at its nominal period). Tests that care about the two columns
    /// differing build the plan directly.
    fn plan_from(
        corrected: [Vec<i64>; 3],
        effective_period_us: [i64; 3],
        nominal_period_us: i64,
    ) -> ImuGridPlan {
        let raw = corrected.clone();
        ImuGridPlan::build_from_corrected(corrected, raw, effective_period_us, nominal_period_us)
    }

    #[test]
    fn build_spans_clean_imu_has_no_spans() {
        // Arrange / Act / Assert — leading 0, no drops, occupied == target.
        assert!(build_spans(0, &[], 5, 5).is_empty());
    }

    #[test]
    fn plan_reconciles_single_imu_axis_onto_grid() {
        // Arrange — IMU0 corrected stamps: 0, 1000 (exact period), 3000 (a
        // 2000 µs jump → 1 sample missing before received index 2).
        let corrected = [vec![0i64, 1000, 3000], Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1000, 1000, 1000], 1000);

        // Act — raw [0, 10, 30]; one fill (20) between 10 and 30.
        let (col, t_us, t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![0, 10, 30], scale: 1.0, offset: 0.0 },
        );

        // Assert — value fill unchanged; t_us is the recorded time at every
        // kept slot and the neighbour interpolation inside the gap (here
        // identical to the old uniform grid, because this stream's recorded
        // spacing *is* exactly the period — the existing-behaviour
        // regression case); t_recorded_us carries the real corrected stamp
        // at every kept slot.
        assert_eq!(col.materialize(), vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(spans, vec![GapSpan { start: 2, len: 1 }]);
        assert_eq!(t_us, vec![0, 1000, 2000, 3000]);
        assert_eq!(t_recorded_us[0], 0);
        assert_eq!(t_recorded_us[1], 1000);
        assert_eq!(t_recorded_us[3], 3000);
    }

    #[test]
    fn plan_gives_each_imu_its_own_length_rather_than_padding_to_the_longest() {
        // Arrange — IMU0: 4 corrected stamps with a 2-sample gap between
        // received index 0 and 1 (occupied 6). IMU1: 5 received, no drops
        // (occupied 5). Ruling R241: no shared target — 6 and 5.
        let corrected = [
            vec![0i64, 3000, 4000, 5000],
            vec![0i64, 1000, 2000, 3000, 4000],
            Vec::new(),
        ];
        let plan = plan_from(corrected, [1000, 1000, 1000], 1000);

        // Act
        let (c0, _, _, _) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![0, 10, 20, 30], scale: 1.0, offset: 0.0 },
        );
        let (c1, _, _, _) = plan.reconcile(
            "IMU1_AccelX",
            RawColumn::I16 { data: vec![1, 2, 3, 4, 5], scale: 1.0, offset: 0.0 },
        );

        // Assert — each rebuilt to its own occupied length: IMU0's interior
        // fills count, IMU1 gets no tail it never recorded.
        assert_eq!(c0.len(), 6);
        assert_eq!(c1.len(), 5);
    }

    #[test]
    fn plan_passes_through_non_imu_channels_unchanged() {
        // Arrange
        let corrected = [vec![0i64, 1000, 2000], Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1000, 1000, 1000], 1000);

        // Act
        let (col, t_us, t_recorded_us, spans) =
            plan.reconcile("GPS_Latitude", RawColumn::F64(vec![1.0, 2.0]));

        // Assert
        assert_eq!(col.materialize(), vec![1.0, 2.0]);
        assert!(t_us.is_empty());
        assert!(t_recorded_us.is_empty());
        assert!(spans.is_empty());
    }

    #[test]
    fn plan_leaves_under_two_sample_imu_unreconciled() {
        // Arrange — a single-sample IMU is not reconciled (design §6).
        let corrected = [vec![42i64], Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1000, 1000, 1000], 1000);

        // Act
        let (col, t_us, t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![42], scale: 1.0, offset: 0.0 },
        );

        // Assert — column unchanged, no gaps, no grid/recorded time (unreconciled).
        assert_eq!(col.materialize(), vec![42.0]);
        assert!(t_us.is_empty());
        assert!(t_recorded_us.is_empty());
        assert!(spans.is_empty());
    }

    #[test]
    fn parse_gps_record_truncated_mid_record_keeps_gps_ts_in_sync_with_columns() {
        // Arrange — a full first GPS fix, followed by a second fix whose
        // buffer physically ends after `device_timestamp_us` but before the
        // remaining five fallible fields (`latitude`..`satellites`) can be
        // read — the real "buffer ended mid-record" recovery path `parse_v3`
        // already documents.
        use crate::parse::test_buffers::gps_payload;

        let full = gps_payload(1_704_110_400_000, 1_000, 10, 20, 30, 40, 50, 1, 8);
        let mut second = gps_payload(0, 2_000, 0, 0, 0, 0, 0, 0, 0);
        second.truncate(18); // gps_epoch_ms(8) + device_ts_us(8) + 2 of latitude's 4 bytes
        let mut buf = full.clone();
        buf.extend_from_slice(&second);
        let mut reader = ByteReader::new(&buf);
        let mut acc = ChannelAccumulator::new();
        let mut gps_ts: Vec<i64> = Vec::new();

        // Act
        parse_gps_record(&mut reader, full.len(), &mut acc, None, None, &mut gps_ts).unwrap();
        let err = parse_gps_record(&mut reader, full.len(), &mut acc, None, None, &mut gps_ts);

        // Assert — the second call reports the truncation and every GPS_*
        // column still has exactly as many samples as `gps_ts` (C1 §2).
        assert!(matches!(err, Err(ParseError::TruncatedRecord(_))));
        assert_eq!(gps_ts.len(), 1);
        let entries = acc.into_entries();
        assert_eq!(entries.len(), 8);
        for (name, col) in &entries {
            assert_eq!(
                col.materialize().len(),
                gps_ts.len(),
                "{name}: t_us/column length mismatch"
            );
        }
    }

    /// One IMU's corrected stamps for the drift-and-dropout cases below:
    /// `n_before` samples at `period_a` µs, then a `dropout_us` hole, then
    /// `n_after` samples at `period_b` µs. Returns the corrected sequence
    /// (absolute device-clock µs).
    fn drifting_stamps_with_dropout(
        n_before: usize,
        period_a: i64,
        dropout_us: i64,
        n_after: usize,
        period_b: i64,
    ) -> Vec<i64> {
        let mut out = Vec::with_capacity(n_before + n_after);
        let mut t = 1_000_000i64;
        for _ in 0..n_before {
            out.push(t);
            t += period_a;
        }
        t += dropout_us;
        for _ in 0..n_after {
            out.push(t);
            t += period_b;
        }
        out
    }

    #[test]
    fn imu_slot_time_drifting_period_with_a_dropout_tracks_the_recorded_stamps() {
        // Arrange — 2000 samples whose true period (1200 µs) drifts from the
        // session median the reconciler is handed (1250 µs), with a 3 s
        // dropout in the middle. Under the old `t0 + slot × period` grid the
        // end of this stream drifted ≈ 2000 × 50 µs = 100 ms away from the
        // hardware stamps; the recorded-time rule must not.
        let stamps = drifting_stamps_with_dropout(1000, 1200, 3_000_000, 1000, 1200);
        let last_stamp = *stamps.last().unwrap();
        let first_stamp = stamps[0];
        let corrected = [stamps, Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1250, 1250, 1250], 1250);

        // Act
        let (_col, t_us, t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![7; 2000], scale: 1.0, offset: 0.0 },
        );

        // Assert — a single reconciled IMU anchors the grid on its own first
        // stamp and sets the grid length, so slot 0 and the last slot are
        // both real: each carries its recorded stamp, well inside one sample
        // period of it. The 3 s dropout is present in `t_us` as a jump.
        let last = t_us.len() - 1;
        assert!(
            (t_us[last] - t_recorded_us[last]).abs() < 1250,
            "t drifted from the recorded stamp: {} vs {}",
            t_us[last],
            t_recorded_us[last]
        );
        assert_eq!(t_us[last], last_stamp);
        assert_eq!(t_us[0], first_stamp);
        let interior = spans
            .iter()
            .find(|s| s.start > 0 && s.start + s.len < t_us.len())
            .expect("the dropout should be an interior gap span");
        let jump = t_us[interior.start + interior.len] - t_us[interior.start - 1];
        assert!(jump > 3_000_000, "the dropout should appear in t, got {jump} µs");
    }

    #[test]
    fn imu_slot_time_drifting_period_with_a_dropout_is_strictly_increasing() {
        // Arrange — same stream, with the two halves running at different
        // true periods so neither matches the reconciler's median.
        let stamps = drifting_stamps_with_dropout(500, 1180, 2_500_000, 500, 1230);
        let corrected = [stamps, Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1250, 1250, 1250], 1250);

        // Act
        let (_col, t_us, _t_recorded_us, _spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![7; 1000], scale: 1.0, offset: 0.0 },
        );

        // Assert — C1 §3.5 invariant 1, per source.
        assert!(t_us.len() >= 1000);
        assert!(t_us.windows(2).all(|w| w[1] > w[0]), "t_us must be strictly increasing");
    }

    #[test]
    fn imu_slot_time_pad_slots_lie_strictly_between_their_neighbours() {
        // Arrange — one interior dropout, so every padded slot has a real
        // corrected stamp on both sides.
        let stamps = drifting_stamps_with_dropout(10, 1200, 100_000, 10, 1200);
        let corrected = [stamps, Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1250, 1250, 1250], 1250);

        // Act
        let (_col, t_us, _t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![7; 20], scale: 1.0, offset: 0.0 },
        );

        // Assert — inside each interior span, every slot is strictly between
        // the real stamps bracketing it.
        let interior: Vec<&GapSpan> =
            spans.iter().filter(|s| s.start > 0 && s.start + s.len < t_us.len()).collect();
        assert!(!interior.is_empty(), "the dropout should produce an interior gap span");
        for s in interior {
            let before = t_us[s.start - 1];
            let after = t_us[s.start + s.len];
            for slot in s.start..s.start + s.len {
                assert!(t_us[slot] > before && t_us[slot] < after, "pad slot {slot} is not between its neighbours");
            }
        }
    }

    /// C1 §3.3's worked example as the two sequences an import really holds:
    /// the raw stamps the firmware wrote (nominal 1250 µs walk-back inside
    /// each burst, a 1050 µs step at each seam) and the corrected axis the
    /// same data produces (a uniform 1200 µs).
    fn worked_example_raw_and_corrected() -> (Vec<i64>, Vec<i64>) {
        let raw = vec![
            96250, 97500, 98750, 100000, //
            101050, 102300, 103550, 104800, //
            105850, 107100, 108350, 109600, //
            110650, 111900, 113150, 114400,
        ];
        let corrected =
            crate::session::seam_correction::correct_burst_seams(&raw, 1250).corrected_us;
        (raw, corrected)
    }

    #[test]
    fn reconcile_a_corrected_imu_puts_the_raw_stamp_in_t_recorded_us_and_the_corrected_one_in_t_us() {
        // Arrange — ruling R240: the two columns must mean different things.
        let (raw, corrected) = worked_example_raw_and_corrected();
        let plan = ImuGridPlan::build_from_corrected(
            [corrected.clone(), Vec::new(), Vec::new()],
            [raw.clone(), Vec::new(), Vec::new()],
            [1200, 1200, 1200],
            1250,
        );

        // Act
        let (_col, t_us, t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![1; 16], scale: 1.0, offset: 0.0 },
        );

        // Assert — no drops here, so every slot is real: t_us is the
        // corrected axis, t_recorded_us is the raw one, and they differ.
        assert!(spans.is_empty());
        assert_eq!(t_us, corrected);
        assert_eq!(t_recorded_us, raw);
        assert_ne!(t_us, t_recorded_us);
    }

    #[test]
    fn reconcile_a_gap_slot_keeps_the_grid_time_in_t_recorded_us_because_nothing_was_recorded_there() {
        // Arrange — one interior dropout: the raw stamps are the corrected
        // ones offset by a constant, so a real slot is recognisable by its
        // offset and a synthesized one by the absence of it.
        let corrected = drifting_stamps_with_dropout(10, 1200, 100_000, 10, 1200);
        let raw: Vec<i64> = corrected.iter().map(|t| t + 7).collect();
        let plan = ImuGridPlan::build_from_corrected(
            [corrected, Vec::new(), Vec::new()],
            [raw, Vec::new(), Vec::new()],
            [1200, 1200, 1200],
            1200,
        );

        // Act
        let (_col, t_us, t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![7; 20], scale: 1.0, offset: 0.0 },
        );

        // Assert — inside every gap span, t_recorded_us repeats t_us (the
        // documented placeholder); outside one, it is the raw stamp.
        assert!(!spans.is_empty(), "the dropout should produce a gap span");
        let mut in_gap = vec![false; t_us.len()];
        for s in &spans {
            for slot in s.start..s.start + s.len {
                in_gap[slot] = true;
            }
        }
        for slot in 0..t_us.len() {
            if in_gap[slot] {
                assert_eq!(t_recorded_us[slot], t_us[slot], "gap slot {slot}");
            } else {
                assert_eq!(t_recorded_us[slot], t_us[slot] + 7, "real slot {slot}");
            }
        }
    }

    #[test]
    fn seam_spans_over_an_import_shaped_channel_finds_the_seams_the_correction_flattened() {
        // Arrange — exactly what `fetch_seams_via` passes after an import:
        // the channel's own `t_recorded_us` and `t_us`, straight out of
        // `reconcile` (review finding 2's collateral note — the tauri tests
        // hand-build a channel and so never exercised this shape).
        let (raw, corrected) = worked_example_raw_and_corrected();
        let plan = ImuGridPlan::build_from_corrected(
            [corrected, Vec::new(), Vec::new()],
            [raw, Vec::new(), Vec::new()],
            [1200, 1200, 1200],
            1250,
        );
        let (_col, t_us, t_recorded_us, _spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![1; 16], scale: 1.0, offset: 0.0 },
        );

        // Act — nominal period, as the command derives it from the rate.
        let seams = crate::session::seam_correction::seam_spans(&t_recorded_us, &t_us, 1250);

        // Assert — the example's three burst boundaries, reported on the
        // corrected axis. Fed the corrected array in both arguments (what
        // the importer produced before R240) the same call reads every
        // sample as its own burst — 15 spurious seams instead of 3 — which
        // is the regression this test exists to hold shut.
        assert_eq!(seams, vec![(100000, 101200), (104800, 106000), (109600, 110800)]);
        assert_eq!(crate::session::seam_correction::seam_spans(&t_us, &t_us, 1250).len(), 15);
    }

    #[test]
    fn an_imu_that_stops_early_ends_at_its_own_last_recorded_sample() {
        // Arrange — ruling R241: IMU0 runs for 100 samples, IMU1 stops after
        // 10. Both at 1000 µs, starting together.
        let long: Vec<i64> = (0..100).map(|i| 1_000_000 + i * 1000).collect();
        let short: Vec<i64> = (0..10).map(|i| 1_000_000 + i * 1000).collect();
        let last_short = *short.last().unwrap();
        let plan = plan_from([long, short, Vec::new()], [1000, 1000, 1000], 1000);

        // Act
        let (_c0, t0_us, _r0, spans0) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![1; 100], scale: 1.0, offset: 0.0 },
        );
        let (c1, t1_us, _r1, spans1) = plan.reconcile(
            "IMU1_AccelX",
            RawColumn::I16 { data: vec![2; 10], scale: 1.0, offset: 0.0 },
        );

        // Assert — the short IMU has 10 slots, not 100, ends at its own last
        // stamp, and carries no trailing gap span, because there is no
        // synthesized tail to mark.
        assert_eq!(t0_us.len(), 100);
        assert_eq!(t1_us.len(), 10);
        assert_eq!(c1.len(), 10);
        assert_eq!(*t1_us.last().unwrap(), last_short);
        assert!(spans0.is_empty());
        assert!(spans1.is_empty(), "no pad means no span: {spans1:?}");
    }

    #[test]
    fn imu_slot_time_gap_free_constant_rate_stream_is_the_uniform_grid() {
        // Arrange — the existing-behaviour regression: a stream whose
        // recorded spacing is exactly the reconciler's period has no gaps,
        // so recorded time and the old `t0 + slot × period` grid agree.
        let stamps: Vec<i64> = (0..50).map(|i| 500_000 + i * 1000).collect();
        let corrected = [stamps, Vec::new(), Vec::new()];
        let plan = plan_from(corrected, [1000, 1000, 1000], 1000);

        // Act
        let (_col, t_us, _t_recorded_us, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![3; 50], scale: 1.0, offset: 0.0 },
        );

        // Assert
        assert!(spans.is_empty());
        assert_eq!(t_us, (0..50).map(|i| 500_000 + i * 1000).collect::<Vec<i64>>());
    }

    #[test]
    fn rebuild_i64_grid_or_real_fills_a_gap_even_when_the_first_real_stamp_is_exactly_zero() {
        // Arrange — a real corrected timestamp of exactly 0 at index 0 (a
        // legitimate, if rare, value — this plan's Open questions flagged a
        // 0-sentinel placeholder-then-overwrite approach as a rough edge
        // precisely because a naive "is it 0?" check would treat this real
        // value as a placeholder and, depending on implementation, disable
        // filling for the whole array). Received [0, 1000, 3000] with a
        // 1-sample gap before received index 2 (period 1000).
        let corrected = vec![0i64, 1000, 3000];
        let gaps = vec![(2usize, 1usize)];
        let grid_t_us = vec![0i64, 1000, 2000, 3000];

        // Act
        let out = rebuild_i64_grid_or_real(&corrected, &gaps, 0, 4, &grid_t_us);

        // Assert — the real 0 at slot 0 is preserved as a real value, AND the
        // synthesized slot 2 is correctly filled from the grid (2000), not
        // left at the internal placeholder.
        assert_eq!(out, vec![0, 1000, 2000, 3000]);
    }
}
