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
///
/// `session.blob_sha256` is left empty — this function only sees the decoded
/// record stream, not the file's raw bytes, so it cannot compute the hash
/// itself. The caller with file access (`SessionHandle::from_path`/
/// `from_bytes`) computes `sha256(bytes)` over the same buffer and fills it
/// in after `parse()` returns.
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
    let mut channel_ts_us: HashMap<String, Vec<i64>> = HashMap::new();
    let mut origin = TimeOrigin::default();
    let mut first: [Option<i64>; 3] = [None; 3];
    let mut last: [Option<i64>; 3] = [None; 3];
    let mut count: [usize; 3] = [0; 3];
    // Nominal IMU grid period (firmware back-counts each FIFO drain at this step,
    // SPEC §5.5) — the fallback period for an IMU with too few samples to run
    // burst-seam correction, and `correct_burst_seams`'s own burst-detection
    // tolerance window.
    let period_us = imu_period_us(imu_sample_rate_hz);
    // Every kept IMU record's own device timestamp, per IMU — the raw recorded
    // time burst-seam correction reconciles against (contract C1 §3.3), once
    // per parse, after the whole stream is read.
    let mut imu_recorded_ts: [Vec<i64>; 3] = Default::default();
    // Per-fix GPS device timestamp — every GPS channel shares this as its
    // `t_us` source (C1 §2; no burst structure to reconcile for GPS).
    let mut gps_ts: Vec<i64> = Vec::new();
    let mut gps_anchor = GpsAnchor::default();
    let mut truncation: Option<ParseError> = None;

    while reader.has_more() {
        match read_record(
            &mut reader,
            imu_mask,
            &registry,
            &mut routing,
            &mut acc,
            &mut channel_ts_us,
            &mut origin,
            &mut first,
            &mut last,
            &mut count,
            &mut imu_recorded_ts,
            &mut gps_ts,
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

    // Burst-seam correction (contract C1 §3.3): recover each IMU's true
    // (possibly off-nominal) sample cadence from its recorded read-instant
    // stamps and re-space every burst to a monotonic, uniform corrected
    // axis, before gap detection ever runs — C1 §3.3's "ruled" ordering
    // (§8 item 1). An IMU with fewer than 2 samples has no burst structure
    // to correct; its (possibly empty) raw stamps pass through verbatim at
    // the nominal period.
    let mut corrected: [Vec<i64>; 3] = Default::default();
    let mut effective_period_us = [period_us; 3];
    let mut import_warnings: Vec<crate::session::seam_correction::ImportWarning> = Vec::new();
    for i in 0..3 {
        if imu_recorded_ts[i].len() >= 2 {
            let seam = crate::session::seam_correction::correct_burst_seams(&imu_recorded_ts[i], period_us);
            effective_period_us[i] = seam.effective_period_us;
            corrected[i] = seam.corrected_us;
            // Never silently drop a non-fatal anomaly (CLAUDE.md §5) — tag
            // each with its source IMU so a caller reading the flattened
            // list can tell which stream it came from.
            import_warnings.extend(seam.warnings.into_iter().map(|w| {
                crate::session::seam_correction::ImportWarning {
                    kind: w.kind,
                    message: format!("IMU{i}: {}", w.message),
                }
            }));
        } else {
            corrected[i] = imu_recorded_ts[i].clone();
        }
    }

    // Drop reconciliation (contract C1 §3.3): gap detection runs once, here,
    // against each IMU's own corrected stamps and effective period — never
    // the nominal period, and never inline in the hot loop (that was the
    // phantom-drop mechanism C1's worked example demonstrates).
    let plan = ImuGridPlan::build_from_corrected(corrected, effective_period_us, period_us);
    let t0_us = origin.min_us.unwrap_or(0);
    let mut channels = Vec::new();
    for (name, column) in acc.into_entries() {
        if let Some(imu_idx) = imu_index_of(&name) {
            let (rebuilt_column, t_us_abs, t_recorded_us_abs, gaps) = plan.reconcile(&name, column);
            let t_us = t_us_abs.iter().map(|&t| t - t0_us).collect();
            let t_recorded_us = if t_recorded_us_abs.is_empty() {
                None
            } else {
                Some(t_recorded_us_abs.iter().map(|&t| t - t0_us).collect())
            };
            channels.push(Channel {
                channel_id: name.clone(),
                t_us,
                t_recorded_us,
                nominal_rate_hz: plan_nominal_rate_for(imu_idx, &effective_period_us),
                column: rebuilt_column,
                source_kind: format!("imu{imu_idx}"),
                unit: unit_for(&name, &registry_by_name),
                gaps,
            });
            continue;
        }
        let rate = resolve_rate(&name, gps_sample_rate_hz, &registry, 0.0);
        let (column, _t_us, _t_recorded_us, _gaps) = plan.reconcile(&name, column);
        let (t_us, source_kind) = if name.starts_with("GPS") {
            let t = gps_ts.iter().map(|&ts| ts - t0_us).collect();
            (t, "gps".to_string())
        } else if let Some(ts) = channel_ts_us.get(&name) {
            let t = ts.iter().map(|&ts| ts - t0_us).collect();
            (t, generic_source_kind(&name))
        } else {
            // No recorded timestamp captured for this name (should not happen
            // for a real registry channel) — empty t_us degrades gracefully
            // rather than panicking (CLAUDE.md §5); surfaced by the round-trip
            // test in Task 9 if it ever fires.
            (Vec::new(), generic_source_kind(&name))
        };
        channels.push(Channel {
            channel_id: name.clone(),
            t_us,
            t_recorded_us: None,
            nominal_rate_hz: rate,
            column,
            source_kind,
            unit: unit_for(&name, &registry_by_name),
            gaps: Vec::new(),
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
            device_id: Some(device_id),
            timestamp_utc_ms: effective_start_ms,
            config_checksum: Some(format!("{config_crc:08x}")),
            source_format: crate::session::SourceFormat::Idl0,
            // Filled by the caller, not here — `parse_v3` only sees the
            // decoded record stream, not the raw file bytes it came from.
            // `SessionHandle::from_bytes` computes `sha256(bytes)` (the
            // `sha2` dependency, Task 1) and overwrites this field after
            // `parse()` returns, before synthesis runs (see store::blob,
            // Task 8).
            blob_sha256: String::new(),
            channels,
        },
        truncation_warning: truncation,
        import_warnings,
    })
}

/// Per-IMU nominal rate (Hz) after burst-seam correction: `1e6 /
/// effective_period_us[imu_idx]`. Replaces the old single session-wide
/// `ImuGridPlan::nominal_rate` — each IMU can now have its own corrected
/// period, so its `nominal_rate_hz` metadata must be its own (contract C1
/// §4.2 already treats `nominal_rate_hz` as per-channel; see this plan's
/// Open questions for the widening this represents from the pre-idl1
/// single-shared-rate model).
fn plan_nominal_rate_for(imu_idx: usize, effective_period_us: &[i64; 3]) -> f64 {
    1e6 / effective_period_us[imu_idx] as f64
}

/// Resolves a channel's physical unit string (contract C1 §4.1). IMU axes and
/// GPS channels get a small hardcoded table (the registry doesn't self-describe
/// a useful unit for them today); every other registry channel falls back to
/// `ChannelRegistryEntry.units` verbatim (SPEC §5.2); unknown names get an
/// empty string.
fn unit_for(channel_id: &str, registry_by_name: &HashMap<String, ChannelRegistryEntry>) -> String {
    if let Some(u) = imu_axis_unit(channel_id) {
        return u.to_string();
    }
    if let Some(u) = gps_channel_unit(channel_id) {
        return u.to_string();
    }
    registry_by_name.get(channel_id).map(|e| e.units.clone()).unwrap_or_default()
}

/// IMU axis unit (`g` for accel, `dps` for gyro), or `None` for a non-IMU
/// channel name.
fn imu_axis_unit(channel_id: &str) -> Option<&'static str> {
    if imu_index_of(channel_id).is_none() {
        return None;
    }
    if channel_id.contains("Accel") {
        Some("g")
    } else if channel_id.contains("Gyro") {
        Some("dps")
    } else {
        None
    }
}

/// GPS channel unit, per contract C1 §4.1's table verbatim, or `None` for a
/// non-GPS channel name.
fn gps_channel_unit(channel_id: &str) -> Option<&'static str> {
    match channel_id {
        "GPS_SpeedKmh" => Some("km/h"),
        "GPS_EpochMs" => Some("ms_raw"),
        "GPS_Latitude" | "GPS_Longitude" => Some("deg"),
        "GPS_Altitude" => Some("m"),
        "GPS_Heading" => Some("deg"),
        "GPS_FixQuality" => Some("enum_raw"),
        "GPS_Satellites" => Some("count"),
        _ => None,
    }
}

/// Reads one record. Returns `Ok(true)` to continue, `Ok(false)` on SESSION_END.
#[allow(clippy::too_many_arguments)]
fn read_record(
    reader: &mut ByteReader,
    imu_mask: u32,
    registry: &HashMap<u8, ChannelRegistryEntry>,
    routing: &mut HotRouting,
    acc: &mut ChannelAccumulator,
    channel_ts_us: &mut HashMap<String, Vec<i64>>,
    origin: &mut TimeOrigin,
    first: &mut [Option<i64>; 3],
    last: &mut [Option<i64>; 3],
    count: &mut [usize; 3],
    imu_recorded_ts: &mut [Vec<i64>; 3],
    gps_ts: &mut Vec<i64>,
    gps_anchor: &mut GpsAnchor,
) -> Result<bool, ParseError> {
    let type_ = reader.u8("record type")?;
    let payload_len = reader.u16("payload_len")? as usize;
    match type_ {
        0xFF => Ok(false),
        0x01 => {
            parse_imu(
                reader, payload_len, imu_mask, &mut routing.imu, acc, first, last, count,
                imu_recorded_ts, origin,
            )?;
            Ok(true)
        }
        0x02 => {
            parse_gps_record(reader, payload_len, acc, Some(gps_anchor), Some(origin), gps_ts)?;
            Ok(true)
        }
        0x03 => {
            parse_channel(reader, payload_len, registry, &mut routing.channel_slot, acc, channel_ts_us, origin)?;
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
    imu_recorded_ts: &mut [Vec<i64>; 3],
    origin: &mut TimeOrigin,
) -> Result<(), ParseError> {
    let payload_start = reader.position();
    let imu_index = reader.u8("imu_index")?;
    let ts_us = reader.i64("timestamp_us")?;
    origin.observe(ts_us);
    let idx = imu_index as usize;
    // A raw wire timestamp that does not advance is a genuine duplicate/
    // backstep read at a FIFO drain boundary (SPEC §5.5) — safe to drop
    // unconditionally: within-burst deltas are always exact at the
    // *nominal* cadence regardless of true ODR (C1 §3.3), so this
    // comparison needs no period knowledge and cannot itself manufacture a
    // phantom drop. Gap detection against the *corrected* period happens
    // once, after the whole stream is read (ImuGridPlan::build_from_corrected).
    let mut drop_sample = false;
    if idx < 3 {
        match last[idx] {
            Some(prev) if ts_us <= prev => {
                drop_sample = true;
            }
            _ => {
                if first[idx].is_none() {
                    first[idx] = Some(ts_us);
                }
                last[idx] = Some(ts_us);
                count[idx] += 1;
            }
        }
    }

    if !drop_sample && idx < IMU_CHANNEL_NAMES.len() {
        // One push per kept record (not per axis) — the raw recorded
        // timestamp Task 6's burst-seam correction reconciles against.
        imu_recorded_ts[idx].push(ts_us);
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
    channel_ts_us: &mut HashMap<String, Vec<i64>>,
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
        // Every CHANNEL_SAMPLE record's own timestamp, regardless of the
        // registry's declared rate — contract C1 §2 makes per-sample time
        // mandatory on every channel, not just event-driven ones.
        channel_ts_us.entry(entry.name.clone()).or_default().push(ts_us);
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
        assert_eq!(r.session.device_id.as_deref(), Some("b0b1b2b3b4b5"));
        assert_eq!(r.session.timestamp_utc_ms, RMC_UTC_MS);
        assert_eq!(r.session.config_checksum.as_deref(), Some("cafebabe"));

        // IMU scaled
        assert_relative_eq!(find(&r, "IMU0_AccelX").materialize()[0], 16.0, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU0_AccelY").materialize()[0], -8192.0 * ACCEL_SCALE as f64, epsilon = 1e-6);
        assert_relative_eq!(find(&r, "IMU0_GyroX").materialize()[0], 1000.0 * GYRO_SCALE as f64, epsilon = 1e-3);

        // GPS: epoch/sats are verbatim wire integers; lat/lon are baked to
        // physical decimal degrees at parse time (ruling R27): raw ×1e-7.
        assert_eq!(find(&r, "GPS_EpochMs").materialize()[0], RMC_UTC_MS as f64);
        assert_relative_eq!(
            find(&r, "GPS_Latitude").materialize()[0],
            51.525,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            find(&r, "GPS_Longitude").materialize()[0],
            -0.1234567,
            epsilon = 1e-9
        );
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
        assert_relative_eq!(ch.nominal_rate_hz, 1e6 / 600.0, epsilon = 1e-6);
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
        assert_relative_eq!(ch.nominal_rate_hz, 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn single_imu_drop_is_linearly_filled_and_recorded() {
        // Arrange — IMU0 at 1000 Hz; the 3rd sample arrives 2 periods after the
        // 2nd (one sample dropped between received indices 1 and 2), followed by
        // 9 more single-sample "bursts" of ordinary ±2 µs read-instant jitter (5
        // at 998 µs, 4 at 1002 µs) — realistic surrounding burst context so the
        // drop's own skewed 1500 µs burst-to-burst estimate is one of 10 total
        // estimates, not the only one. C1 §3.3's median-not-mean robustness
        // guarantee only holds with enough other estimates for the skewed one to
        // be an outlier the median ignores (R12):
        // median(998, 998, 998, 998, 998, 1002, 1002, 1002, 1002, 1500) = 1000 µs,
        // exactly nominal, so the drop is still detected as a genuine gap rather
        // than reinterpreted as an off-nominal true ODR.
        let registry = vec![v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw")];
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x01, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[10])),
            frame(0x01, &imu_payload(0, 1_003_000, &[30])), // 2000 µs jump → 1 missing
            frame(0x01, &imu_payload(0, 1_004_000, &[40])),
            // Ordinary jittery burst context appended after the drop (no
            // further drops) — 5 seams at nominal-2 µs, 4 at nominal+2 µs.
            frame(0x01, &imu_payload(0, 1_004_998, &[50])),
            frame(0x01, &imu_payload(0, 1_005_996, &[60])),
            frame(0x01, &imu_payload(0, 1_006_994, &[70])),
            frame(0x01, &imu_payload(0, 1_007_992, &[80])),
            frame(0x01, &imu_payload(0, 1_008_990, &[90])),
            frame(0x01, &imu_payload(0, 1_009_992, &[100])),
            frame(0x01, &imu_payload(0, 1_010_994, &[110])),
            frame(0x01, &imu_payload(0, 1_011_996, &[120])),
            frame(0x01, &imu_payload(0, 1_012_998, &[130])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert — one linear fill (20) between 10 and 30, the rest verbatim;
        // one GapSpan at the drop's own location, unaffected by the extra
        // context appended after it.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(
            ch.materialize(),
            vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        );
        assert_eq!(ch.gaps, vec![GapSpan { start: 2, len: 1 }]);
        assert_eq!(ch.len(), 14);
    }

    #[test]
    fn two_imus_with_different_drops_align_a_shared_spike_to_the_same_slot() {
        // Arrange — IMU0 and IMU1 at 1000 Hz on one clock. Both record a spike
        // (1000) at the same timestamp (1_004_000), but IMU0 drops a sample
        // before it while IMU1 does not. Both streams then carry 9 more samples
        // of realistic surrounding burst context: IMU0 as ordinary jittery
        // single-sample "bursts" (±2 µs seams, no further drops), IMU1 as plain
        // nominal continuation — so IMU0's median has enough burst-to-burst
        // estimates for its drop's skewed one to be an ignorable outlier (R12):
        // median(1333.33, 998×5, 1002×4) = 1000 µs, matching IMU1's trivial
        // (single-burst, no-estimate) nominal fallback exactly. Reconciliation
        // must make them equal-length and land the spike on the same slot so
        // `[A] - [B]` works.
        let registry = vec![
            v3_registry_entry(0, 4, 1000, 1.0, 0.0, "IMU0_AccelX", "raw"),
            v3_registry_entry(6, 4, 1000, 1.0, 0.0, "IMU1_AccelX", "raw"),
        ];
        // mask: IMU0 axis0 = bit 0, IMU1 axis0 = bit 6.
        let buf = cat(&[
            Header { schema_version: 3, imu_mask: 0x41, imu_count: 2, imu_sample_rate_hz: 1000, ..Default::default() }
                .build(&registry),
            // IMU0 — drops one sample between 1_001_000 and 1_003_000, then 9
            // ordinary jittery bursts (no further drops).
            frame(0x01, &imu_payload(0, 1_000_000, &[0])),
            frame(0x01, &imu_payload(0, 1_001_000, &[0])),
            frame(0x01, &imu_payload(0, 1_003_000, &[0])),
            frame(0x01, &imu_payload(0, 1_004_000, &[1000])), // spike
            frame(0x01, &imu_payload(0, 1_005_000, &[0])),
            frame(0x01, &imu_payload(0, 1_005_998, &[0])),
            frame(0x01, &imu_payload(0, 1_006_996, &[0])),
            frame(0x01, &imu_payload(0, 1_007_994, &[0])),
            frame(0x01, &imu_payload(0, 1_008_992, &[0])),
            frame(0x01, &imu_payload(0, 1_009_990, &[0])),
            frame(0x01, &imu_payload(0, 1_010_992, &[0])),
            frame(0x01, &imu_payload(0, 1_011_994, &[0])),
            frame(0x01, &imu_payload(0, 1_012_996, &[0])),
            frame(0x01, &imu_payload(0, 1_013_998, &[0])),
            // IMU1 — no drops, plain nominal continuation.
            frame(0x01, &imu_payload(1, 1_000_000, &[0])),
            frame(0x01, &imu_payload(1, 1_001_000, &[0])),
            frame(0x01, &imu_payload(1, 1_002_000, &[0])),
            frame(0x01, &imu_payload(1, 1_003_000, &[0])),
            frame(0x01, &imu_payload(1, 1_004_000, &[1000])), // spike, same timestamp
            frame(0x01, &imu_payload(1, 1_005_000, &[0])),
            frame(0x01, &imu_payload(1, 1_006_000, &[0])),
            frame(0x01, &imu_payload(1, 1_007_000, &[0])),
            frame(0x01, &imu_payload(1, 1_008_000, &[0])),
            frame(0x01, &imu_payload(1, 1_009_000, &[0])),
            frame(0x01, &imu_payload(1, 1_010_000, &[0])),
            frame(0x01, &imu_payload(1, 1_011_000, &[0])),
            frame(0x01, &imu_payload(1, 1_012_000, &[0])),
            frame(0x01, &imu_payload(1, 1_013_000, &[0])),
            frame(0x01, &imu_payload(1, 1_014_000, &[0])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();
        let a = find(&r, "IMU0_AccelX");
        let b = find(&r, "IMU1_AccelX");

        // Assert — equal length, equal nominal rate, spike on the same slot, and
        // the two reconciled columns are identical (so the difference is all 0).
        assert_eq!(a.len(), b.len());
        assert_eq!(a.nominal_rate_hz, b.nominal_rate_hz);
        assert_eq!(a.materialize()[4], 1000.0);
        assert_eq!(b.materialize()[4], 1000.0);
        assert_eq!(a.materialize(), b.materialize());
        // IMU0 carries the recorded fill; IMU1 has none.
        assert_eq!(a.gaps, vec![GapSpan { start: 2, len: 1 }]);
        assert!(b.gaps.is_empty());
    }

    #[test]
    fn all_imu_channels_report_the_single_nominal_rate_despite_different_drops() {
        // Arrange — same two-IMU stream as the shared-spike test above (minus
        // the spike itself): IMU0 drops one, IMU1 drops none, each carrying 9
        // more samples of realistic surrounding burst context (IMU0 jittery
        // single-sample "bursts", IMU1 plain nominal continuation) so IMU0's
        // median has enough burst-to-burst estimates for its drop's skewed one
        // to be an ignorable outlier (R12) rather than the only estimate. The
        // old (n-1)/span formula gave 800 vs 1000 Hz; the nominal rate is
        // identical.
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
            frame(0x01, &imu_payload(0, 1_005_998, &[0])),
            frame(0x01, &imu_payload(0, 1_006_996, &[0])),
            frame(0x01, &imu_payload(0, 1_007_994, &[0])),
            frame(0x01, &imu_payload(0, 1_008_992, &[0])),
            frame(0x01, &imu_payload(0, 1_009_990, &[0])),
            frame(0x01, &imu_payload(0, 1_010_992, &[0])),
            frame(0x01, &imu_payload(0, 1_011_994, &[0])),
            frame(0x01, &imu_payload(0, 1_012_996, &[0])),
            frame(0x01, &imu_payload(0, 1_013_998, &[0])),
            frame(0x01, &imu_payload(1, 1_000_000, &[0])),
            frame(0x01, &imu_payload(1, 1_001_000, &[0])),
            frame(0x01, &imu_payload(1, 1_002_000, &[0])),
            frame(0x01, &imu_payload(1, 1_003_000, &[0])),
            frame(0x01, &imu_payload(1, 1_004_000, &[0])),
            frame(0x01, &imu_payload(1, 1_005_000, &[0])),
            frame(0x01, &imu_payload(1, 1_006_000, &[0])),
            frame(0x01, &imu_payload(1, 1_007_000, &[0])),
            frame(0x01, &imu_payload(1, 1_008_000, &[0])),
            frame(0x01, &imu_payload(1, 1_009_000, &[0])),
            frame(0x01, &imu_payload(1, 1_010_000, &[0])),
            frame(0x01, &imu_payload(1, 1_011_000, &[0])),
            frame(0x01, &imu_payload(1, 1_012_000, &[0])),
            frame(0x01, &imu_payload(1, 1_013_000, &[0])),
            frame(0x01, &imu_payload(1, 1_014_000, &[0])),
            session_end(),
        ]);

        // Act
        let r = parse_v3(&buf).unwrap();

        // Assert
        let r0 = find(&r, "IMU0_AccelX").nominal_rate_hz;
        let r1 = find(&r, "IMU1_AccelX").nominal_rate_hz;
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
        // Every channel carries real, strictly increasing t_us now (C1 §2/§3.5
        // invariant 1) — fixed-rate channels are no longer distinguished by a
        // "times present/absent" signal.
        assert!(brake.t_us.windows(2).all(|w| w[1] > w[0]));
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
        assert_eq!(rr.nominal_rate_hz, 0.0);
        // Tightened to exact µs checks on t_us directly — the old
        // sample_times_secs-presence signal ("this channel carries real
        // per-sample time") is now true of every channel by construction.
        assert_eq!(rr.t_us, vec![500_000, 1_000_000, 1_300_000]);
        assert_relative_eq!(rr.materialize()[0], 1000.0, epsilon = 1e-6);
    }

    #[test]
    fn event_channel_duration_spans_its_own_first_to_last_sample() {
        // duration_ms is (t_us.last() - t_us.first()) / 1000 (session::Channel
        // doc comment) — the channel's own data span, not "time since the
        // session origin to the last sample". HR_RR's own samples run from
        // 500_000 µs to 1_300_000 µs (session-relative, origin = the IMU
        // record at 1_000_000), an 800 ms span.
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
        assert_eq!(find(&r, "HR_RR").duration_ms(), 800);
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
        let t_us = &find(&r, "HR_RR").t_us;
        assert_eq!(t_us, &vec![0, 400_000]);
    }

    #[test]
    fn imu_burst_off_nominal_odr_reports_the_corrected_rate_and_spacing() {
        // Arrange — contract C1 §3.3's worked example's *raw* stamps
        // verbatim: nominal 1250 µs (800 Hz configured), true period
        // 1200 µs (≈833.3 Hz), 4 bursts of N=4. IMU0, single axis.
        let raw_stamps: [i64; 16] = [
            96250, 97500, 98750, 100000, // burst 0
            101050, 102300, 103550, 104800, // burst 1
            105850, 107100, 108350, 109600, // burst 2
            110650, 111900, 113150, 114400, // burst 3
        ];
        let registry = vec![v3_registry_entry(0, 4, 800, 1.0, 0.0, "IMU0_AccelX", "raw")];
        let mut parts = vec![Header {
            schema_version: 3,
            imu_mask: 0x01,
            imu_sample_rate_hz: 800, // imu_period_us(800) == 1250, the nominal period.
            ..Default::default()
        }
        .build(&registry)];
        for &ts in raw_stamps.iter() {
            parts.push(frame(0x01, &imu_payload(0, ts, &[10])));
        }
        parts.push(session_end());

        // Act
        let r = parse_v3(&cat(&parts)).unwrap();

        // Assert — nominal_rate_hz reflects the *corrected* 1200 µs period,
        // not the configured-ODR 1250 µs one; every sample lands cleanly on
        // the corrected grid (no drops, since the worked example's corrected
        // stamps are exactly evenly spaced), so t_us advances by exactly
        // 1200 µs at every step.
        let ch = find(&r, "IMU0_AccelX");
        assert_eq!(ch.len(), 16);
        assert!(ch.gaps.is_empty());
        assert_relative_eq!(ch.nominal_rate_hz, 1e6 / 1200.0, epsilon = 1e-6);
        assert!(ch.t_us.windows(2).all(|w| w[1] - w[0] == 1200));

        // t_recorded_us is present (correction actually diverged it from the
        // nominal-grid formula the pre-Task-6 parser used) and, since this
        // IMU has no drops, advances at the same 1200 µs corrected spacing
        // as t_us.
        let t_recorded = ch.t_recorded_us.as_ref().expect("burst correction should set t_recorded_us");
        assert_eq!(t_recorded.len(), 16);
        assert!(t_recorded.windows(2).all(|w| w[1] - w[0] == 1200));
    }
}
