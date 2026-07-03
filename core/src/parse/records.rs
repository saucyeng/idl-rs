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
/// Emits the eight raw GPS channels (no scale/offset). Optionally seeds the
/// `anchor` from the first non-zero `gps_epoch_ms` (for §5.6 back-fill) and
/// folds `device_timestamp_us` into `origin` (the event-time zero).
pub fn parse_gps_record(
    reader: &mut ByteReader,
    payload_len: usize,
    out: &mut ChannelAccumulator,
    anchor: Option<&mut GpsAnchor>,
    origin: Option<&mut TimeOrigin>,
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

    out.push("GPS_EpochMs", gps_epoch_ms as f64);
    out.push("GPS_Latitude", latitude as f64);
    out.push("GPS_Longitude", longitude as f64);
    out.push("GPS_Altitude", altitude as f64);
    // GPS_SpeedKmh is the one GPS-fix channel returned in physical units. The
    // firmware logs km/h × 100 (§5.6); a 0.01 scale on a compact i32 column makes
    // `materialize()` yield km/h, so Distance synthesis, math expressions, and the
    // colour-by-channel scale all get physical speed without each consumer
    // dividing (§5.7). Lat/lon/alt/heading stay raw — their consumers apply the
    // documented ÷1e7 / ÷10 / ÷100.
    let speed_slot = out.slot_for_i32("GPS_SpeedKmh", 0.01, 0.0);
    out.push_i32_at(speed_slot, speed as i32);
    out.push("GPS_Heading", heading as f64);
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
/// list, occupied length, and the session-wide `target_len`. Shared across the
/// IMU's six axes. Design §4.2/§4.3.
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

/// Drop-reconciliation plan for the three IMUs, derived once at parse
/// finalization from the per-IMU first-sample timestamps, received counts, and
/// drop lists collected in the hot loop. Anchors a single grid at the earliest
/// IMU first-sample (`t0`), assigns the nominal rate, and rebuilds each IMU axis
/// onto that grid so all IMU channels end equal-length and time-aligned. Design
/// §4.
pub struct ImuGridPlan {
    /// Nominal grid period (µs) shared by all IMUs.
    pub period_us: i64,
    /// Nominal rate (Hz) assigned to every IMU channel.
    pub nominal_rate: f64,
    /// Leading pad length (slots) per IMU — `round((first - t0) / period)`.
    leading: [usize; 3],
    /// Whether each IMU has ≥2 samples and is therefore reconciled.
    reconciled: [bool; 3],
    /// Session-wide grid length — the max occupied length across reconciled IMUs.
    target_len: usize,
    /// Per-IMU drop list `(received_index, missing)`, used by the rebuild.
    gaps_received: [Vec<(usize, usize)>; 3],
    /// Per-IMU grid-slot gap spans, shared across that IMU's six axes.
    spans: [Vec<GapSpan>; 3],
}

impl ImuGridPlan {
    /// Builds the plan from the hot-loop state. `period_us` comes from
    /// [`imu_period_us`]. IMUs with fewer than two samples are left
    /// unreconciled (design §6).
    pub fn build(
        first: &[Option<i64>; 3],
        count: &[usize; 3],
        gaps_received: [Vec<(usize, usize)>; 3],
        period_us: i64,
    ) -> Self {
        let nominal_rate = imu_nominal_rate(period_us);
        // Session t0 = earliest IMU first-sample timestamp across all IMUs.
        let t0 = first.iter().filter_map(|f| *f).min();

        let mut leading = [0usize; 3];
        let mut occupied = [0usize; 3];
        let mut reconciled = [false; 3];
        let mut target_len = 0usize;
        for i in 0..3 {
            // < 2 samples → no reconciliation (design §6).
            if count[i] < 2 {
                continue;
            }
            let (Some(f), Some(t0v)) = (first[i], t0) else { continue };
            reconciled[i] = true;
            // Leading offset onto the shared grid; f ≥ t0v so the diff is ≥ 0.
            leading[i] = (((f - t0v) as f64) / period_us as f64).round().max(0.0) as usize;
            let total_missing: usize = gaps_received[i].iter().map(|g| g.1).sum();
            occupied[i] = leading[i] + count[i] + total_missing;
            target_len = target_len.max(occupied[i]);
        }

        // Gap spans depend on the session-wide target_len, so build them after it
        // is known.
        let mut spans: [Vec<GapSpan>; 3] = Default::default();
        for i in 0..3 {
            if reconciled[i] {
                spans[i] = build_spans(leading[i], &gaps_received[i], occupied[i], target_len);
            }
        }

        ImuGridPlan {
            period_us,
            nominal_rate,
            leading,
            reconciled,
            target_len,
            gaps_received,
            spans,
        }
    }

    /// Reconciles one channel by name. IMU axes of a reconciled IMU are rebuilt
    /// onto the shared grid and returned with that IMU's gap spans; every other
    /// channel passes through unchanged with an empty gap list.
    pub fn reconcile(&self, name: &str, column: RawColumn) -> (RawColumn, Vec<GapSpan>) {
        match imu_index_of(name) {
            Some(i) if self.reconciled[i] => match column {
                RawColumn::I16 { data, scale, offset } => {
                    let rebuilt =
                        rebuild_i16(&data, &self.gaps_received[i], self.leading[i], self.target_len);
                    (RawColumn::I16 { data: rebuilt, scale, offset }, self.spans[i].clone())
                }
                // IMU axes are always compact i16 on the parse path; pass any
                // other variant through untouched rather than panic.
                other => (other, Vec::new()),
            },
            _ => (column, Vec::new()),
        }
    }
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

    #[test]
    fn build_spans_clean_imu_has_no_spans() {
        // Arrange / Act / Assert — leading 0, no drops, occupied == target.
        assert!(build_spans(0, &[], 5, 5).is_empty());
    }

    #[test]
    fn plan_reconciles_single_imu_axis_onto_grid() {
        // Arrange — IMU0 is t0; 3 samples; 1 missing before received index 2.
        let gaps = [vec![(2usize, 1usize)], Vec::new(), Vec::new()];
        let plan = ImuGridPlan::build(&[Some(0), None, None], &[3, 0, 0], gaps, 1000);

        // Act — raw [0, 10, 30]; one fill (20) between 10 and 30.
        let (col, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![0, 10, 30], scale: 1.0, offset: 0.0 },
        );

        // Assert
        assert_eq!(col.materialize(), vec![0.0, 10.0, 20.0, 30.0]);
        assert_eq!(spans, vec![GapSpan { start: 2, len: 1 }]);
        assert_eq!(plan.nominal_rate, 1000.0);
    }

    #[test]
    fn plan_makes_two_imus_with_different_drops_equal_length() {
        // Arrange — IMU0: 4 received, a 2-sample drop (occupied 6).
        //           IMU1: 5 received, no drops (occupied 5). target = 6.
        let gaps = [vec![(1usize, 2usize)], Vec::new(), Vec::new()];
        let plan = ImuGridPlan::build(&[Some(0), Some(0), None], &[4, 5, 0], gaps, 1000);

        // Act
        let (c0, _) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![0, 10, 20, 30], scale: 1.0, offset: 0.0 },
        );
        let (c1, _) = plan.reconcile(
            "IMU1_AccelX",
            RawColumn::I16 { data: vec![1, 2, 3, 4, 5], scale: 1.0, offset: 0.0 },
        );

        // Assert — both rebuilt to the shared grid length 6 (IMU1 tail-padded).
        assert_eq!(c0.len(), 6);
        assert_eq!(c1.len(), 6);
    }

    #[test]
    fn plan_passes_through_non_imu_channels_unchanged() {
        // Arrange
        let plan =
            ImuGridPlan::build(&[Some(0), None, None], &[3, 0, 0], Default::default(), 1000);

        // Act
        let (col, spans) = plan.reconcile("GPS_Latitude", RawColumn::F64(vec![1.0, 2.0]));

        // Assert
        assert_eq!(col.materialize(), vec![1.0, 2.0]);
        assert!(spans.is_empty());
    }

    #[test]
    fn plan_leaves_under_two_sample_imu_unreconciled() {
        // Arrange — a single-sample IMU is not reconciled (design §6).
        let plan =
            ImuGridPlan::build(&[Some(0), None, None], &[1, 0, 0], Default::default(), 1000);

        // Act
        let (col, spans) = plan.reconcile(
            "IMU0_AccelX",
            RawColumn::I16 { data: vec![42], scale: 1.0, offset: 0.0 },
        );

        // Assert — column unchanged, no gaps.
        assert_eq!(col.materialize(), vec![42.0]);
        assert!(spans.is_empty());
    }
}
