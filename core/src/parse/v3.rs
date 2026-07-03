//! v3 (`IDL0` schema 3) parser. Ported from `BinaryParser.parseV3`.
//!
//! v3 differs from v2 only in 40-byte registry entries (explicit `scale`/
//! `offset`) and per-channel `physical = raw × scale + offset` scaling applied
//! to every value, including individually-registered IMU axes.

use std::collections::HashMap;

use crate::parse::reader::ByteReader;
use crate::parse::records::*;
use crate::session::{Channel, ChannelRegistryEntry, ParseError, ParseResult, Session};

/// Per-session hot-loop routing, built once after the header so the v3 record
/// loop never hashes a channel name per sample (SPEC §5.2). Holds the IMU
/// `(scale, offset)` + axis→slot table and the generic-channel `channel_id →
/// slot` cache.
struct HotRouting {
    imu: ImuRouting,
    /// Accumulator slot for each generic-channel `channel_id` (0x03 records),
    /// filled lazily on the first sample. Indexed directly by the `u8` id.
    channel_slot: [Option<usize>; 256],
}

impl HotRouting {
    fn new(registry_by_name: &HashMap<String, ChannelRegistryEntry>) -> Self {
        Self {
            imu: ImuRouting::new(registry_by_name),
            channel_slot: [None; 256],
        }
    }
}

/// Per-`[imu_index][axis]` routing for the IMU hot loop. `scale_offset` holds
/// the registry `(scale, offset)` for each axis — `(1.0, 0.0)` when the axis is
/// unregistered, matching the raw-passthrough fallback. `slot` caches the
/// accumulator bucket index, filled lazily on the first sample for that axis.
struct ImuRouting {
    scale_offset: [[(f64, f64); 6]; 3],
    slot: [[Option<usize>; 6]; 3],
}

impl ImuRouting {
    /// Resolves `(scale, offset)` for every `[imu][axis]` from the by-name
    /// registry once, up front. Axes absent from the registry keep `(1.0, 0.0)`
    /// so `raw × 1.0 + 0.0 == raw` reproduces the old per-sample raw fallback.
    fn new(registry_by_name: &HashMap<String, ChannelRegistryEntry>) -> Self {
        let mut scale_offset = [[(1.0_f64, 0.0_f64); 6]; 3];
        for (imu, names) in IMU_CHANNEL_NAMES.iter().enumerate() {
            for (axis, name) in names.iter().enumerate() {
                if let Some(e) = registry_by_name.get(*name) {
                    scale_offset[imu][axis] = (e.scale, e.offset);
                }
            }
        }
        Self { scale_offset, slot: [[None; 6]; 3] }
    }
}

/// Parses a v3 `.idl0` buffer. Returns the parsed session plus an optional
/// truncation warning (the buffer ended mid-record).
pub fn parse_v3(bytes: &[u8]) -> Result<ParseResult, ParseError> {
    let mut reader = ByteReader::new(bytes);

    let magic = String::from_utf8_lossy(reader.bytes(4, "magic")?).into_owned();
    if magic != "IDL0" {
        return Err(ParseError::InvalidMagicBytes(format!(
            "Expected IDL0, got: {magic}"
        )));
    }
    let schema = reader.u8("schema version")?;
    if schema != 3 {
        return Err(ParseError::UnsupportedSchemaVersion(format!(
            "parseV3 called with schema v{schema} (expected 3)"
        )));
    }

    let session_id = to_hex(reader.bytes(16, "UUID")?);
    let device_id = to_hex(reader.bytes(6, "device ID")?);
    let session_start_ms = reader.i64("session start UTC ms")?;
    let config_crc = reader.u32("config CRC32")?;
    let imu_mask = reader.u32("IMU channel mask")?;
    let _imu_count = reader.u8("IMU count")?;
    let imu_sample_rate_hz = reader.u16("IMU sample rate Hz")?;
    let gps_sample_rate_hz = reader.u8("GPS sample rate Hz")?;

    let registry_count = reader.u8("channel registry count")?;
    let mut registry: HashMap<u8, ChannelRegistryEntry> = HashMap::new();
    let mut registry_by_name: HashMap<String, ChannelRegistryEntry> = HashMap::new();
    for _ in 0..registry_count {
        let entry = read_registry_entry_v3(&mut reader)?;
        registry.insert(entry.channel_id, entry.clone());
        registry_by_name.insert(entry.name.clone(), entry);
    }

    let marker = reader.u32("header end marker")?;
    if marker != 0xDEAD_BEEF {
        return Err(ParseError::TruncatedRecord(
            "v3 header end marker missing or corrupt".to_string(),
        ));
    }

    let mut acc = ChannelAccumulator::new();
    let mut routing = HotRouting::new(&registry_by_name);
    let mut event_ts_us: HashMap<String, Vec<i64>> = HashMap::new();
    let mut origin = TimeOrigin::default();
    let mut first: [Option<i64>; 3] = [None; 3];
    let mut last: [Option<i64>; 3] = [None; 3];
    let mut count: [usize; 3] = [0; 3];
    // Nominal IMU grid period (firmware back-counts each FIFO drain at this step,
    // SPEC §5.5). The hot loop records a drop only when a timestamp jump deviates
    // from it — a clean log never touches `imu_gaps`.
    let period_us = imu_period_us(imu_sample_rate_hz);
    let mut imu_gaps: [Vec<(usize, usize)>; 3] = Default::default();
    // Absolute grid slot (relative to each IMU's first sample) of the last *kept*
    // sample. Placement anchors to absolute time so per-IMU drop/backstep history
    // never accumulates cross-IMU drift (§15.2).
    let mut last_abs_slot: [i64; 3] = [0; 3];
    let mut gps_anchor = GpsAnchor::default();
    let mut truncation: Option<ParseError> = None;

    while reader.has_more() {
        match read_record(
            &mut reader,
            imu_mask,
            &registry,
            &mut routing,
            &mut acc,
            &mut event_ts_us,
            &mut origin,
            &mut first,
            &mut last,
            &mut count,
            period_us,
            &mut imu_gaps,
            &mut last_abs_slot,
            &mut gps_anchor,
        ) {
            Ok(true) => {}
            Ok(false) => break, // SESSION_END
            Err(e @ ParseError::TruncatedRecord(_)) => {
                truncation = Some(e);
                break;
            }
            Err(e) => return Err(e),
        }
    }

    // Drop reconciliation (SPEC §15): all IMU channels share one nominal rate and
    // a single grid anchored at the earliest IMU first-sample; each IMU column is
    // rebuilt onto that grid (drops linear-filled, edges held), so cross-IMU
    // element-wise math no longer sees mismatched rates/lengths.
    let plan = ImuGridPlan::build(&first, &count, imu_gaps, period_us);
    let mut channels = Vec::new();
    for (name, column) in acc.into_entries() {
        let rate = resolve_rate(&name, gps_sample_rate_hz, &registry, plan.nominal_rate);
        let sample_times_secs = match event_ts_us.get(&name) {
            Some(ts) if !ts.is_empty() => {
                let o = origin.min_us.unwrap_or(ts[0]);
                Some(ts.iter().map(|&t| (t - o) as f64 / 1e6).collect())
            }
            _ => None,
        };
        let (column, gaps) = plan.reconcile(&name, column);
        channels.push(Channel {
            channel_id: name,
            sample_rate_hz: rate,
            column,
            sample_times_secs,
            gaps,
        });
    }

    // §5.6 back-fill: firmware writes 0 in the header, so recover the session
    // start from the first non-zero GPS fix. The device timestamp is monotonic
    // since *boot*, not since this recording — anchoring on the GPS fix alone
    // (`epoch - dev/1000`) yields the boot instant, which is identical for every
    // recording in one power cycle. Subtract only the offset from the
    // recording's first sample (`origin.min_us`) to that GPS fix, giving the
    // wall clock at recording start:
    //   `gps_epoch - (gps_device_ts - first_sample_ts) / 1000`.
    let mut effective_start_ms = session_start_ms;
    if effective_start_ms == 0 {
        if let (Some(epoch), Some(dev)) = (gps_anchor.gps_epoch_ms, gps_anchor.device_ts_us) {
            let first_sample_us = origin.min_us.unwrap_or(0);
            effective_start_ms =
                epoch - ((dev - first_sample_us) as f64 / 1000.0).round() as i64;
        }
    }

    Ok(ParseResult {
        session: Session {
            session_id,
            device_id,
            timestamp_utc_ms: effective_start_ms,
            config_checksum: format!("{config_crc:08x}"),
            channels,
        },
        truncation_warning: truncation,
    })
}

/// Reads one record. Returns `Ok(true)` to continue, `Ok(false)` on SESSION_END.
#[allow(clippy::too_many_arguments)]
fn read_record(
    reader: &mut ByteReader,
    imu_mask: u32,
    registry: &HashMap<u8, ChannelRegistryEntry>,
    routing: &mut HotRouting,
    acc: &mut ChannelAccumulator,
    event_ts_us: &mut HashMap<String, Vec<i64>>,
    origin: &mut TimeOrigin,
    first: &mut [Option<i64>; 3],
    last: &mut [Option<i64>; 3],
    count: &mut [usize; 3],
    period_us: i64,
    imu_gaps: &mut [Vec<(usize, usize)>; 3],
    last_abs_slot: &mut [i64; 3],
    gps_anchor: &mut GpsAnchor,
) -> Result<bool, ParseError> {
    let type_ = reader.u8("record type")?;
    let payload_len = reader.u16("payload_len")? as usize;
    match type_ {
        0xFF => Ok(false),
        0x01 => {
            parse_imu(
                reader, payload_len, imu_mask, &mut routing.imu, acc, first, last, count, period_us,
                imu_gaps, last_abs_slot, origin,
            )?;
            Ok(true)
        }
        0x02 => {
            parse_gps_record(reader, payload_len, acc, Some(gps_anchor), Some(origin))?;
            Ok(true)
        }
        0x03 => {
            parse_channel(reader, payload_len, registry, &mut routing.channel_slot, acc, event_ts_us, origin)?;
            Ok(true)
        }
        _ => {
            reader.skip(payload_len, "unknown record payload")?;
            Ok(true)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_imu(
    reader: &mut ByteReader,
    payload_len: usize,
    imu_mask: u32,
    routing: &mut ImuRouting,
    acc: &mut ChannelAccumulator,
    first: &mut [Option<i64>; 3],
    last: &mut [Option<i64>; 3],
    count: &mut [usize; 3],
    period_us: i64,
    imu_gaps: &mut [Vec<(usize, usize)>; 3],
    last_abs_slot: &mut [i64; 3],
    origin: &mut TimeOrigin,
) -> Result<(), ParseError> {
    let payload_start = reader.position();
    let imu_index = reader.u8("imu_index")?;
    let ts_us = reader.i64("timestamp_us")?;
    origin.observe(ts_us);
    let idx = imu_index as usize;
    // Absolute-grid placement (§15.2): each sample lands on slot
    // `round((ts - first) / period)`, so co-temporal events across IMUs share a
    // slot regardless of differing drop histories — no per-step drift. Fast path
    // (exact nominal Δ, the ~99% case) is one i64 compare + increment, no divide.
    // A sample whose slot does not advance past the last kept one (a backward
    // step / duplicate at a FIFO drain boundary) is dropped: not counted, its
    // axes not stored. A forward jump records the missing run for the rebuild.
    let mut drop_sample = false;
    if idx < 3 {
        match (first[idx], last[idx]) {
            (Some(f), Some(prev)) => {
                let delta = ts_us - prev;
                let abs_slot = if delta == period_us {
                    last_abs_slot[idx] + 1
                } else {
                    ((ts_us - f) as f64 / period_us as f64).round() as i64
                };
                if abs_slot <= last_abs_slot[idx] {
                    drop_sample = true;
                } else {
                    let missing = (abs_slot - last_abs_slot[idx] - 1) as usize;
                    if missing >= 1 {
                        imu_gaps[idx].push((count[idx], missing));
                    }
                    last_abs_slot[idx] = abs_slot;
                    last[idx] = Some(ts_us);
                    count[idx] += 1;
                }
            }
            _ => {
                // First sample of this IMU anchors relative slot 0.
                first[idx] = Some(ts_us);
                last[idx] = Some(ts_us);
                last_abs_slot[idx] = 0;
                count[idx] += 1;
            }
        }
    }

    if !drop_sample && idx < IMU_CHANNEL_NAMES.len() {
        let names = IMU_CHANNEL_NAMES[idx];
        for axis in 0..6u32 {
            let mask_bit = imu_index as u32 * 6 + axis;
            if (imu_mask >> mask_bit) & 1 == 1 {
                // Honor the record's own `payload_len`: stop if it cannot hold
                // another i16. Some logs carry fewer axes than the header mask
                // claims (observed in recovered/older logs with a 6-axis mask but
                // 4-axis records); reading the mask's count would overrun into the
                // next record and desync the whole stream after record 1.
                if reader.position() - payload_start + 2 > payload_len {
                    break;
                }
                // Read the raw sample first — the byte must be consumed whether or
                // not the axis is registered (parity with the old code path).
                // Store the raw i16 compactly; (scale, offset) is captured once at
                // slot creation and applied lazily on materialize — no per-sample
                // multiply on the hot path.
                let raw = reader.i16("IMU axis")?;
                let a = axis as usize;
                let slot = match routing.slot[idx][a] {
                    Some(s) => s,
                    None => {
                        let (scale, offset) = routing.scale_offset[idx][a];
                        let s = acc.slot_for_i16(names[a], scale, offset);
                        routing.slot[idx][a] = Some(s);
                        s
                    }
                };
                acc.push_i16_at(slot, raw);
            }
        }
    }

    let consumed = reader.position() - payload_start;
    if consumed < payload_len {
        reader.skip(payload_len - consumed, "IMU payload remainder")?;
    }
    Ok(())
}

fn parse_channel(
    reader: &mut ByteReader,
    payload_len: usize,
    registry: &HashMap<u8, ChannelRegistryEntry>,
    channel_slot: &mut [Option<usize>; 256],
    acc: &mut ChannelAccumulator,
    event_ts_us: &mut HashMap<String, Vec<i64>>,
    origin: &mut TimeOrigin,
) -> Result<(), ParseError> {
    let payload_start = reader.position();
    let channel_id = reader.u8("channel_id")?;
    let ts_us = reader.i64("timestamp_us")?;
    origin.observe(ts_us);

    let entry = match registry.get(&channel_id) {
        Some(e) => e,
        None => {
            let consumed = reader.position() - payload_start;
            if consumed < payload_len {
                reader.skip(payload_len - consumed, "unknown channel payload")?;
            }
            return Ok(());
        }
    };

    if (entry.data_type as usize) < DATA_TYPE_WIDTHS.len() {
        // Resolve the accumulator slot once per channel_id; route by index after.
        // Compact registry types (i16/i32/f32 = codes 4/5/6) store the raw wire
        // value + lazy (scale, offset); other types store the already-physical
        // f64. Materialize reproduces `(raw as f64) * scale + offset` either way.
        let slot = match channel_slot[channel_id as usize] {
            Some(s) => s,
            None => {
                let s = match entry.data_type {
                    4 => acc.slot_for_i16(&entry.name, entry.scale, entry.offset),
                    5 => acc.slot_for_i32(&entry.name, entry.scale, entry.offset),
                    6 => acc.slot_for_f32(&entry.name, entry.scale, entry.offset),
                    _ => acc.slot_for(&entry.name),
                };
                channel_slot[channel_id as usize] = Some(s);
                s
            }
        };
        match entry.data_type {
            4 => acc.push_i16_at(slot, reader.i16("i16 value")?),
            5 => acc.push_i32_at(slot, reader.i32("i32 value")?),
            6 => acc.push_f32_at(slot, reader.f32("f32 value")?),
            _ => acc.push_at(
                slot,
                read_typed_value(reader, entry.data_type)? * entry.scale + entry.offset,
            ),
        }
        if entry.sample_rate_hz == 0 {
            // Event-driven channel (low-rate): record the per-sample timestamp.
            // The name is only cloned here, never on the high-rate path.
            event_ts_us.entry(entry.name.clone()).or_default().push(ts_us);
        }
    }

    let consumed = reader.position() - payload_start;
    if consumed < payload_len {
        reader.skip(payload_len - consumed, "channel payload remainder")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_v3;
    use crate::parse::test_buffers::*;
    use crate::session::{Channel, GapSpan, ParseError, ParseResult, RawColumn};
    use approx::assert_relative_eq;

    const ACCEL_SCALE: f32 = 32.0 / 32768.0;
    const GYRO_SCALE: f32 = 2000.0 / 32768.0;

    fn find<'a>(r: &'a ParseResult, name: &str) -> &'a Channel {
        r.session
            .channels
            .iter()
            .find(|c| c.channel_id == name)
            .unwrap_or_else(|| panic!("channel {name} not found"))
    }

    fn has(r: &ParseResult, name: &str) -> bool {
        r.session.channels.iter().any(|c| c.channel_id == name)
    }

    #[test]
    fn wrong_magic_returns_invalid_magic_bytes() {
        let mut buf = vec![0u8; 54];
        buf[0..4].copy_from_slice(b"ESPL");
        assert!(matches!(parse_v3(&buf), Err(ParseError::InvalidMagicBytes(_))));
    }

    #[test]
    fn idl0_schema_2_returns_unsupported_schema_version() {
        let buf = cat(&[Header { schema_version: 2, ..Default::default() }.build(&[]), session_end()]);
        assert!(matches!(parse_v3(&buf), Err(ParseError::UnsupportedSchemaVersion(_))));
    }

    #[test]
    fn full_round_trip_header_imu_gps_channel() {
        // Arrange
        let uuid: Vec<u8> = (1..=16).collect();
        let dev = vec![0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5];
        let mut registry = v3_imu_axes_registry(0, 0, 800, ACCEL_SCALE, GYRO_SCALE);
        registry.push(v3_registry_entry(18, 2, 0, 1.0, 0.0, "WheelFront", "pulse"));
        let buf = cat(&[
            Header {
                schema_version: 3,
                uuid,
                device_id: dev,
                session_start_ms: RMC_UTC_MS,
                config_crc: 0xCAFE_BABE,
                imu_mask: 0x3F,
                ..Default::default()
            }
            .build(&registry),
            frame(0x01, &imu_payload(0, 1250, &[16384, -8192, 0, 1000, -500, 0])),
            frame(0x02, &gps_payload(RMC_UTC_MS, 1250, 515_250_000, -1_234_567, 500, 1000, 18000, 1, 8)),
            frame(0x03, &channel_payload_u32(18, 2_000_000, 99)),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — metadata
        assert!(r.is_complete());
        assert_eq!(r.session.session_id, "0102030405060708090a0b0c0d0e0f10");
        assert_eq!(r.session.device_id, "b0b1b2b3b4b5");
        assert_eq!(r.session.timestamp_utc_ms, RMC_UTC_MS);
        assert_eq!(r.session.config_checksum, "cafebabe");

        // IMU scaled
        assert_relative_eq!(find(&r, "IMU0_AccelX").materialize()[0], 16.0, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU0_AccelY").materialize()[0], -8192.0 * ACCEL_SCALE as f64, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU0_GyroX").materialize()[0], 1000.0 * GYRO_SCALE as f64, epsilon = 1e-3);

        // GPS raw (lat/lon/epoch/sats are verbatim wire integers)
        assert_eq!(find(&r, "GPS_EpochMs").materialize()[0], RMC_UTC_MS as f64);
        assert_eq!(find(&r, "GPS_Latitude").materialize()[0], 515_250_000.0);
        assert_eq!(find(&r, "GPS_Longitude").materialize()[0], -1_234_567.0);
        assert_eq!(find(&r, "GPS_Satellites").materialize()[0], 8.0);
        // GPS_SpeedKmh is engine-scaled to physical km/h: raw 1000 (km/h × 100)
        // → 10.0 km/h via the 0.01 column scale (§5.7).
        assert_relative_eq!(
            find(&r, "GPS_SpeedKmh").materialize()[0],
            10.0,
            epsilon = 1e-9
        );

        // Generic channel (scale 1, offset 0)
        assert_eq!(find(&r, "WheelFront").materialize()[0], 99.0);

        // Compact storage: IMU axes are raw i16; GPS_SpeedKmh is a scaled i32
        // (km/h × 100 stored, 0.01 scale); lat/lon and generic channels stay
        // verbatim f64. Locks the memory win against silent regression.
        assert!(matches!(find(&r, "IMU0_AccelX").column, RawColumn::I16 { .. }));
        assert!(matches!(find(&r, "IMU0_GyroX").column, RawColumn::I16 { .. }));
        assert!(matches!(find(&r, "GPS_SpeedKmh").column, RawColumn::I32 { .. }));
        assert!(matches!(find(&r, "GPS_Latitude").column, RawColumn::F64(_)));
        assert!(matches!(find(&r, "WheelFront").column, RawColumn::F64(_)));
    }

    #[test]
    fn backfill_session_start_is_recording_start_not_boot() {
        // Arrange — the firmware leaves the header start time at 0, so the start
        // is back-filled from the first GPS fix (§5.6). The device timestamp is
        // monotonic since *boot*, not since this recording: this file's first
        // sample is at t = 600 s and its first GPS fix lands 5 s later at
        // t = 605 s carrying wall clock RMC_UTC_MS. The recording therefore
        // started at RMC_UTC_MS − 5 s — NOT at the boot instant
        // (RMC_UTC_MS − 605 s), which is what every recording in this power
        // cycle would otherwise collapse to.
        const FIRST_SAMPLE_US: i64 = 600_000_000; // 600 s after boot
        const GPS_FIX_US: i64 = 605_000_000; // first fix, 5 s into the recording
        let registry = v3_imu_axes_registry(0, 0, 800, ACCEL_SCALE, GYRO_SCALE);
        let buf = cat(&[
            Header {
                schema_version: 3,
                session_start_ms: 0, // firmware leaves the header time at 0
                imu_mask: 0x3F,
                ..Default::default()
            }
            .build(&registry),
            frame(0x01, &imu_payload(0, FIRST_SAMPLE_US, &[16384, 0, 0, 0, 0, 0])),
            frame(
                0x02,
                &gps_payload(RMC_UTC_MS, GPS_FIX_US, 515_250_000, -1_234_567, 500, 1000, 18000, 1, 8),
            ),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — start is the wall clock at the first sample (RMC_UTC_MS − 5 s),
        // not the boot instant (RMC_UTC_MS − 605 s).
        assert_eq!(r.session.timestamp_utc_ms, RMC_UTC_MS - 5_000);
    }

    #[test]
    fn imu_record_with_fewer_axes_than_mask_stays_aligned() {
        // Arrange — header mask claims 6 axes (0x3F), but each IMU record carries
        // only 4 (payload_len = 17). Mirrors the recovered 4-axis logs: previously
        // the parser trusted the mask, over-read 4 bytes, and desynced after the
        // first record.
        let registry = v3_imu_axes_registry(0, 0, 833, ACCEL_SCALE, GYRO_SCALE);
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x3F, imu_sample_rate_hz: 833, ..Default::default() }
                .build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[10, 20, 30, 40])),
            frame(0x01, &imu_payload(0, 1_001_200, &[11, 21, 31, 41])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — no desync: both records parse, first 4 axes get 2 samples each,
        // the 2 axes the records omit (GyroY/GyroZ) are absent.
        assert!(r.is_complete());
        assert_eq!(find(&r, "IMU0_AccelX").len(), 2);
        assert_eq!(find(&r, "IMU0_GyroX").len(), 2); // 4th axis present
        assert!(!has(&r, "IMU0_GyroY")); // 5th — not supplied
        assert!(!has(&r, "IMU0_GyroZ")); // 6th — not supplied
    }

    #[test]
    fn default_range_scaling() {
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x3F, ..Default::default() }
                .build(&v3_imu_axes_registry(0, 0, 800, ACCEL_SCALE, GYRO_SCALE)),
            frame(0x01, &imu_payload(0, 1250, &[16384, 0, 0, 1000, 0, 0])),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_relative_eq!(find(&r, "IMU0_AccelX").materialize()[0], 16.0, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU0_GyroX").materialize()[0], 61.0352, epsilon = 1e-3);
    }

    #[test]
    fn mixed_range_same_raw_different_physical() {
        let accel16: f32 = 16.0 / 32768.0;
        let mut registry = v3_imu_axes_registry(0, 0, 800, ACCEL_SCALE, GYRO_SCALE);
        registry.extend(v3_imu_axes_registry(1, 6, 800, accel16, GYRO_SCALE));
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x3F | 0xFC0, imu_count: 2, ..Default::default() }.build(&registry),
            frame(0x01, &imu_payload(0, 1250, &[16384, 0, 0, 0, 0, 0])),
            frame(0x01, &imu_payload(1, 1250, &[16384, 0, 0, 0, 0, 0])),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_relative_eq!(find(&r, "IMU0_AccelX").materialize()[0], 16.0, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU1_AccelX").materialize()[0], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn disabled_axis_absent_from_output() {
        // Mask 0x1F = bits 0-4 (GyroZ bit 5 clear); registry has 5 entries.
        let registry = vec![
            v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g"),
            v3_registry_entry(1, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelY", "g"),
            v3_registry_entry(2, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelZ", "g"),
            v3_registry_entry(3, 4, 800, GYRO_SCALE, 0.0, "IMU0_GyroX", "dps"),
            v3_registry_entry(4, 4, 800, GYRO_SCALE, 0.0, "IMU0_GyroY", "dps"),
        ];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x1F, ..Default::default() }.build(&registry),
            frame(0x01, &imu_payload(0, 1250, &[100, 200, 300, 400, 500])),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert!(has(&r, "IMU0_GyroY"));
        assert!(!has(&r, "IMU0_GyroZ"));
    }

    #[test]
    fn imu_channel_reports_nominal_rate_not_a_drop_skewed_average() {
        // Arrange — IMU0 at ODR 1666 (integer period 600 µs); 6 samples exactly
        // on the grid. The reconciler assigns the nominal rate (1e6/600), not the
        // observed (n-1)/span.
        let registry = vec![
            v3_registry_entry(0, 4, 1666, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g"),
            v3_registry_entry(1, 4, 1666, ACCEL_SCALE, 0.0, "IMU0_AccelY", "g"),
            v3_registry_entry(2, 4, 1666, ACCEL_SCALE, 0.0, "IMU0_AccelZ", "g"),
        ];
        let mut parts = vec![Header {
            schema_version: 3,
            imu_mask: 0x07,
            imu_sample_rate_hz: 1666,
            ..Default::default()
        }
        .build(&registry)];
        for i in 0..6i64 {
            parts.push(frame(0x01, &imu_payload(0, 1_000_000 + i * 600, &[100 + i as i16, 200, 300])));
        }
        parts.push(session_end());

        // Act
        let r = parse_v3(&cat(&parts)).unwrap();

        // Assert — nominal rate, no drops → length unchanged, empty gap list.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(ch.len(), 6);
        assert!(ch.gaps.is_empty());
        assert_relative_eq!(ch.sample_rate_hz, 1e6 / 600.0, epsilon = 1e-6);
    }

    #[test]
    fn clean_single_imu_stream_is_a_noop_with_nominal_rate() {
        // Arrange — IMU0 at 1000 Hz (period 1000 µs), 4 samples exactly on grid.
        let registry = vec![v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw")];
        let mut parts = vec![Header {
            schema_version: 3,
            imu_mask: 0x01,
            imu_sample_rate_hz: 1000,
            ..Default::default()
        }
        .build(&registry)];
        for i in 0..4i64 {
            parts.push(frame(0x01, &imu_payload(0, 1_000_000 + i * 1000, &[(10 * (i + 1)) as i16])));
        }
        parts.push(session_end());

        // Act
        let r = parse_v3(&cat(&parts)).unwrap();

        // Assert — no fills, empty gap list, length == received, nominal rate.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(ch.len(), 4);
        assert!(ch.gaps.is_empty());
        assert_eq!(ch.materialize(), vec![10.0, 20.0, 30.0, 40.0]);
        assert_relative_eq!(ch.sample_rate_hz, 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn single_imu_drop_is_linearly_filled_and_recorded() {
        // Arrange — IMU0 at 1000 Hz; the 3rd sample arrives 2 periods after the
        // 2nd (one sample dropped between received indices 1 and 2).
        let registry = vec![v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[10])),
            frame(0x01, &imu_payload(0, 1_003_000, &[30])), // 2000 µs jump → 1 missing
            frame(0x01, &imu_payload(0, 1_004_000, &[40])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — one linear fill (20) between 10 and 30; length 5; one GapSpan.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(ch.materialize(), vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(ch.gaps, vec![GapSpan { start: 2, len: 1 }]);
        assert_eq!(ch.len(), 5);
    }

    #[test]
    fn two_imus_with_different_drops_align_a_shared_spike_to_the_same_slot() {
        // Arrange — IMU0 and IMU1 at 1000 Hz on one clock. Both record a spike
        // (1000) at the same timestamp (1_004_000), but IMU0 drops a sample
        // before it while IMU1 does not. Reconciliation must make them
        // equal-length and land the spike on the same slot so `[A] - [B]` works.
        let registry = vec![
            v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw"),
            v3_registry_entry(6, 4, 1000, 1.0, 0.0, "IMU1_AccelX", "raw"),
        ];
        // mask: IMU0 axis0 = bit 0, IMU1 axis0 = bit 6.
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x41, imu_count: 2, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            // IMU0 — drops one sample between 1_001_000 and 1_003_000.
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[0])),
            frame(0x01, &imu_payload(0, 1_003_000, &[0])),
            frame(0x01, &imu_payload(0, 1_004_000, &[1000])), // spike
            frame(0x01, &imu_payload(0, 1_005_000, &[0])),
            // IMU1 — no drops.
            frame(0x01, &imu_payload(1, 1_000_000, &[0])),
            frame(0x01, &imu_payload(1, 1_001_000, &[0])),
            frame(0x01, &imu_payload(1, 1_002_000, &[0])),
            frame(0x01, &imu_payload(1, 1_003_000, &[0])),
            frame(0x01, &imu_payload(1, 1_004_000, &[1000])), // spike, same timestamp
            frame(0x01, &imu_payload(1, 1_005_000, &[0])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();
        let a = find(&r, "IMU0_AccelX");
        let b = find(&r, "IMU1_AccelX");

        // Assert — equal length, equal nominal rate, spike on the same slot, and
        // the two reconciled columns are identical (so the difference is all 0).
        assert_eq!(a.len(), b.len());
        assert_eq!(a.sample_rate_hz, b.sample_rate_hz);
        assert_eq!(a.materialize()[4], 1000.0);
        assert_eq!(b.materialize()[4], 1000.0);
        assert_eq!(a.materialize(), b.materialize());
        // IMU0 carries the recorded fill; IMU1 has none.
        assert_eq!(a.gaps, vec![GapSpan { start: 2, len: 1 }]);
        assert!(b.gaps.is_empty());
    }

    #[test]
    fn all_imu_channels_report_the_single_nominal_rate_despite_different_drops() {
        // Arrange — same two-IMU stream; IMU0 drops one, IMU1 drops none. The old
        // (n-1)/span formula gave 800 vs 1000 Hz; the nominal rate is identical.
        let registry = vec![
            v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw"),
            v3_registry_entry(6, 4, 1000, 1.0, 0.0, "IMU1_AccelX", "raw"),
        ];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x41, imu_count: 2, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[0])),
            frame(0x01, &imu_payload(0, 1_003_000, &[0])),
            frame(0x01, &imu_payload(0, 1_004_000, &[0])),
            frame(0x01, &imu_payload(0, 1_005_000, &[0])),
            frame(0x01, &imu_payload(1, 1_000_000, &[0])),
            frame(0x01, &imu_payload(1, 1_001_000, &[0])),
            frame(0x01, &imu_payload(1, 1_002_000, &[0])),
            frame(0x01, &imu_payload(1, 1_003_000, &[0])),
            frame(0x01, &imu_payload(1, 1_004_000, &[0])),
            frame(0x01, &imu_payload(1, 1_005_000, &[0])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert
        let r0 = find(&r, "IMU0_AccelX").sample_rate_hz;
        let r1 = find(&r, "IMU1_AccelX").sample_rate_hz;
        assert_eq!(r0, r1);
        assert_relative_eq!(r0, 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn backward_timestamp_sample_is_dropped_to_preserve_alignment() {
        // Arrange — IMU0 at 1000 Hz with one backward step (drain-boundary jitter).
        // The out-of-order sample (value 20) maps to an already-occupied slot, so
        // it is dropped rather than pushed forward (which would drift the grid).
        let registry = vec![v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[10])),
            frame(0x01, &imu_payload(0, 1_000_500, &[20])), // backward step → dropped
            frame(0x01, &imu_payload(0, 1_001_500, &[30])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — the backstep sample is dropped; the rest stays on its true slot.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(ch.materialize(), vec![0.0, 10.0, 30.0]);
        assert_eq!(ch.len(), 3);
        assert!(ch.gaps.is_empty());
    }

    #[test]
    fn backsteps_in_one_imu_do_not_shift_a_later_co_temporal_spike() {
        // Arrange — IMU0 and IMU1 at 1000 Hz on one clock; both spike (1000) at
        // the same timestamp (1_004_000). IMU0 has a backward step before the
        // spike; IMU1 is clean. With per-step advance accumulation the backstep
        // would push IMU0's spike to a later slot than IMU1's (the drift bug);
        // absolute-slot placement must land both spikes on the same slot.
        let registry = vec![
            v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw"),
            v3_registry_entry(6, 4, 1000, 1.0, 0.0, "IMU1_AccelX", "raw"),
        ];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x41, imu_count: 2, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            // IMU0 — a backward step at 1_001_500 before the spike.
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[0])),
            frame(0x01, &imu_payload(0, 1_002_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_500, &[0])), // backstep → dropped
            frame(0x01, &imu_payload(0, 1_003_000, &[0])),
            frame(0x01, &imu_payload(0, 1_004_000, &[1000])), // spike
            frame(0x01, &imu_payload(0, 1_005_000, &[0])),
            // IMU1 — clean.
            frame(0x01, &imu_payload(1, 1_000_000, &[0])),
            frame(0x01, &imu_payload(1, 1_001_000, &[0])),
            frame(0x01, &imu_payload(1, 1_002_000, &[0])),
            frame(0x01, &imu_payload(1, 1_003_000, &[0])),
            frame(0x01, &imu_payload(1, 1_004_000, &[1000])), // spike, same timestamp
            frame(0x01, &imu_payload(1, 1_005_000, &[0])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();
        let a = find(&r, "IMU0_AccelX");
        let b = find(&r, "IMU1_AccelX");

        // Assert — equal length, both spikes on slot 4, identical reconciled
        // columns (so [A] - [B] is zero everywhere, including the spike).
        assert_eq!(a.len(), b.len());
        assert_eq!(a.materialize()[4], 1000.0);
        assert_eq!(b.materialize()[4], 1000.0);
        assert_eq!(a.materialize(), b.materialize());
    }

    #[test]
    fn channel_sample_applies_scale_and_offset() {
        let registry = vec![v3_registry_entry(0, 4, 100, 0.001, 0.5, "PressureFront", "bar")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x00, ..Default::default() }.build(&registry),
            frame(0x03, &channel_payload_i16(0, 5_000_000, 1000)),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_relative_eq!(find(&r, "PressureFront").materialize()[0], 1.5, epsilon = 1e-6);
        // i16 registry channel (data_type 4) is stored compactly.
        assert!(matches!(find(&r, "PressureFront").column, RawColumn::I16 { .. }));
    }

    #[test]
    fn repeated_channel_id_routes_all_samples_to_one_channel_in_order() {
        // Arrange — one fixed-rate generic channel, three samples.
        let registry = vec![v3_registry_entry(7, 4, 100, 0.5, 1.0, "Brake", "bar")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x00, ..Default::default() }.build(&registry),
            frame(0x03, &channel_payload_i16(7, 1_000_000, 100)),
            frame(0x03, &channel_payload_i16(7, 1_010_000, 200)),
            frame(0x03, &channel_payload_i16(7, 1_020_000, 300)),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — one channel, three samples in arrival order, scale+offset applied.
        let brake = find(&r, "Brake");
        assert_eq!(brake.len(), 3);
        assert_relative_eq!(brake.materialize()[0], 100.0 * 0.5 + 1.0, epsilon = 1e-6);
        assert_relative_eq!(brake.materialize()[1], 200.0 * 0.5 + 1.0, epsilon = 1e-6);
        assert_relative_eq!(brake.materialize()[2], 300.0 * 0.5 + 1.0, epsilon = 1e-6);
        // Fixed-rate channel → no per-sample event timestamps.
        assert!(brake.sample_times_secs.is_none());
    }

    #[test]
    fn unknown_channel_id_skipped() {
        let registry = vec![v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }.build(&registry),
            frame(0x03, &channel_payload_i16(99, 1000, 999)),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert!(r.is_complete());
        assert_eq!(find(&r, "IMU0_AccelX").len(), 1);
        assert_relative_eq!(find(&r, "IMU0_AccelX").materialize()[0], 16.0, epsilon = 1e-6);
    }

    #[test]
    fn missing_registry_entry_stores_raw_fallback() {
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }.build(&[]),
            frame(0x01, &imu_payload(0, 1250, &[12345])),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_eq!(find(&r, "IMU0_AccelX").materialize()[0], 12345.0);
    }

    #[test]
    fn session_end_then_junk_ignored() {
        let registry = vec![v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }.build(&registry),
            frame(0x01, &imu_payload(0, 1250, &[16384])),
            session_end(),
            vec![0xFF, 0xFF, 0xFF, 0xFF],
        ]);
        let r = parse_v3(&buf).unwrap();
        assert!(r.is_complete());
        assert_eq!(find(&r, "IMU0_AccelX").len(), 1);
    }

    #[test]
    fn session_start_back_filled_from_first_nonzero_gps() {
        let real_epoch = 1_735_732_800_000i64;
        let device_ts = 5_000_000i64;
        let registry = vec![v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g")];
        // The first record — the epoch=0 GPS fix — sits at device_ts =
        // 1_000_000 µs, so it (not boot) is the recording origin. Back-fill
        // subtracts the offset from that first sample to the first non-zero
        // fix, not the raw since-boot device timestamp (§5.6).
        let first_sample_us = 1_000_000i64;
        let buf = cat(&[
            Header { schema_version: 3, session_start_ms: 0, imu_mask: 0x01, ..Default::default() }.build(&registry),
            frame(0x02, &gps_payload(0, first_sample_us, 0, 0, 0, 0, 0, 0, 0)),
            frame(0x02, &gps_payload(real_epoch, device_ts, 0, 0, 0, 0, 0, 1, 8)),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_eq!(
            r.session.timestamp_utc_ms,
            real_epoch - (device_ts - first_sample_us) / 1000,
        );
    }

    #[test]
    fn hr_rr_event_sample_times_relative_to_earliest_record() {
        let rr_scale: f32 = 1000.0 / 1024.0;
        let registry = vec![
            v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g"),
            v3_registry_entry(23, 1, 0, rr_scale, 0.0, "HR_RR", "ms"),
        ];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }.build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[16384])), // origin
            frame(0x03, &channel_payload_u16(23, 1_500_000, 1024)),
            frame(0x03, &channel_payload_u16(23, 2_000_000, 900)),
            frame(0x03, &channel_payload_u16(23, 2_300_000, 850)),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        let rr = find(&r, "HR_RR");
        assert_eq!(rr.sample_rate_hz, 0.0);
        let times = rr.sample_times_secs.as_ref().unwrap();
        assert_eq!(times.len(), 3);
        assert_relative_eq!(times[0], 0.5, epsilon = 1e-9);
        assert_relative_eq!(times[1], 1.0, epsilon = 1e-9);
        assert_relative_eq!(times[2], 1.3, epsilon = 1e-9);
        assert_relative_eq!(rr.materialize()[0], 1000.0, epsilon = 1e-6);
        assert!(find(&r, "IMU0_AccelX").sample_times_secs.is_none());
    }

    #[test]
    fn event_channel_duration_reflects_last_timestamp() {
        let rr_scale: f32 = 1000.0 / 1024.0;
        let registry = vec![
            v3_registry_entry(0, 4, 800, ACCEL_SCALE, 0.0, "IMU0_AccelX", "g"),
            v3_registry_entry(23, 1, 0, rr_scale, 0.0, "HR_RR", "ms"),
        ];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, ..Default::default() }.build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[16384])),
            frame(0x03, &channel_payload_u16(23, 1_500_000, 1024)),
            frame(0x03, &channel_payload_u16(23, 2_000_000, 900)),
            frame(0x03, &channel_payload_u16(23, 2_300_000, 850)),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        assert_eq!(find(&r, "HR_RR").duration_ms(), 1300);
    }

    #[test]
    fn hrm_only_session_origin_is_first_event() {
        let rr_scale: f32 = 1000.0 / 1024.0;
        let registry = vec![v3_registry_entry(23, 1, 0, rr_scale, 0.0, "HR_RR", "ms")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x00, ..Default::default() }.build(&registry),
            frame(0x03, &channel_payload_u16(23, 5_000_000, 1000)),
            frame(0x03, &channel_payload_u16(23, 5_400_000, 980)),
            session_end(),
        ]);
        let r = parse_v3(&buf).unwrap();
        let times = find(&r, "HR_RR").sample_times_secs.as_ref().unwrap();
        assert_relative_eq!(times[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(times[1], 0.4, epsilon = 1e-9);
    }
}
