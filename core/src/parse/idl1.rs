//! `.idl1` session file parser (DRAFT spec 2026-10-15,
//! `docs/superpowers/specs/2026-10-15-idl1-file-format-DRAFT.md`).
//!
//! IDL1 hardware's own format, separate from `.idl0` (Isaac, 2026-10-15).
//! Records keep `.idl0`'s `[type:u8][len:u16][payload]` framing, so unknown
//! types are skipped by length. IMU sample times come from the chips' own
//! clocks ([`super::idl1_timing`]); every other channel carries its own
//! recorded timestamp.
//!
//! Output is the same C1 [`Session`] model as `.idl0`: per-channel `t_us`
//! from the session's first sample, lost IMU samples filled linearly and
//! listed as [`GapSpan`]s.

use std::collections::HashMap;

use crate::parse::idl1_timing::{self, ClockPair, TsMark};
use crate::parse::reader::ByteReader;
use crate::parse::records::{null_term_str, parse_gps_record, read_typed_value, to_hex, ChannelAccumulator, GpsAnchor, DATA_TYPE_WIDTHS};
use crate::session::{Channel, GapSpan, ImportWarning, ImportWarningKind, ParseError, ParseResult, RawColumn, Session, SourceFormat, TimestampSource};

/// The `.idl1` format version this parser reads.
pub const IDL1_FORMAT_VERSION: u16 = 1;

/// Header bytes before the registry: magic 4, format_version 2,
/// header_len 4, session_uuid 16, device_id 6, firmware_version 24,
/// session_start_utc_ms 8, config_crc32 4, imu_freq_fine 3,
/// registry_count 2.
const FIXED_HEADER_BYTES: usize = 73;

/// One registry entry, bytes (spec §4.2).
const REGISTRY_ENTRY_BYTES: usize = 52;

/// The header's trailing CRC32, bytes.
const HEADER_CRC_BYTES: usize = 4;

/// IMU_BLOCK's fixed part (spec §4.4): imu_index, flags, read_us,
/// read_chip_ts, n_samples, n_hg, n_ts, bytes.
const IMU_BLOCK_FIXED_BYTES: usize = 1 + 1 + 8 + 4 + 2 + 2 + 1;

/// One FIFO timestamp word in IMU_BLOCK: after_sample, after_hg, chip_ts.
const TS_WORD_BYTES: usize = 8;

/// GPS_FIX payload: the `.idl0` 32 bytes plus `hdop_x100:u16`.
const GPS_FIX_BYTES: usize = 34;

/// Width of the clock-map fit windows, µs of ESP32 time (spec §3; a parser
/// setting, tuned from bench data).
const CLOCK_FIT_WINDOW_US: i64 = 10_000_000;

/// Registry `kind` values (spec §4.2).
const KIND_IMU_AXIS: u8 = 1;
/// Registry `path` values for an IMU axis.
const PATH_HIGH_G: u8 = 1;

/// Record types (spec §4.3).
const REC_GPS_FIX: u8 = 0x02;
const REC_CHANNEL_SAMPLE: u8 = 0x03;
const REC_IMU_BLOCK: u8 = 0x10;
const REC_LOGGER_EVENT: u8 = 0x20;
const REC_SESSION_SUMMARY: u8 = 0x21;
const REC_GPS_CONFIG: u8 = 0x22;
const REC_SESSION_END: u8 = 0xFF;

/// One `.idl1` registry entry (spec §4.2).
#[derive(Debug, Clone)]
struct RegistryEntry {
    channel_id: u16,
    data_type: u8,
    kind: u8,
    source: u8,
    axis: u8,
    path: u8,
    rate_mhz: u32,
    scale: f64,
    offset: f64,
    name: String,
    units: String,
}

fn read_registry_entry(r: &mut ByteReader) -> Result<RegistryEntry, ParseError> {
    let channel_id = r.u16("registry channel_id")?;
    let data_type = r.u8("registry data_type")?;
    let kind = r.u8("registry kind")?;
    let source = r.u8("registry source")?;
    let axis = r.u8("registry axis")?;
    let path = r.u8("registry path")?;
    let _reserved = r.u8("registry reserved")?;
    let rate_mhz = r.u32("registry rate_mhz")?;
    let scale = r.f32("registry scale")? as f64;
    let offset = r.f32("registry offset")? as f64;
    let name = null_term_str(r.bytes(24, "registry name")?);
    let units = null_term_str(r.bytes(8, "registry units")?);
    Ok(RegistryEntry { channel_id, data_type, kind, source, axis, path, rate_mhz, scale, offset, name, units })
}

/// One IMU stream (low-g/gyro or high-g) as read: its registry entries in
/// axis order, the raw values per axis, the timestamp marks and per-block
/// counts.
#[derive(Default)]
struct Stream {
    /// Registry entries of the recorded axes, in axis order.
    axes: Vec<RegistryEntry>,
    /// Raw values per recorded axis, received samples only.
    values: Vec<Vec<i16>>,
    /// Received samples so far.
    count: usize,
    /// `(received-sample index, raw chip_ts)` per FIFO timestamp word.
    marks: Vec<(usize, u32)>,
    /// `(read_us, samples in that block)` per block, for the read-instant
    /// fallback.
    blocks: Vec<(i64, usize)>,
}

/// One IMU: its two streams and the per-drain clock pairs.
#[derive(Default)]
struct Imu {
    low: Stream,
    high: Stream,
    /// `(raw read_chip_ts, read_us)` per block.
    reads: Vec<(u32, i64)>,
    /// Every chip timestamp this IMU produced, in the order produced, tagged
    /// with where it goes once unwrapped — one counter, unwrapped as one
    /// sequence.
    chip_order: Vec<ChipUse>,
    /// Blocks with the FIFO-overrun flag set.
    overrun_blocks: usize,
}

impl Imu {
    /// An empty segment with this IMU's recorded axes, for after a restart.
    fn fresh_segment(&self) -> Imu {
        let mut fresh = Imu::default();
        for (to, from) in [(&mut fresh.low, &self.low), (&mut fresh.high, &self.high)] {
            to.axes = from.axes.clone();
            to.values = vec![Vec::new(); from.axes.len()];
        }
        fresh
    }
}

/// LOGGER_EVENT code for "IMU recovered" (spec §4.3): the firmware re-ran the
/// IMU's init, so its FIFO and TIMESTAMP counter restarted. Blocks after it
/// are a new clock segment.
const EVT_IMU_RECOVERED: u8 = 5;

/// Where an IMU's chip timestamp is used, so one unwrap pass can route it.
#[derive(Clone, Copy)]
enum ChipUse {
    Read(usize),
    LowMark(usize),
    HighMark(usize),
}

/// Parses an `.idl1` buffer (spec §4). Returns the session plus an optional
/// truncation warning when the buffer ends mid-record.
///
/// `session.blob_sha256` is left empty: the caller with the file's bytes
/// fills it in, as for `.idl0`.
pub fn parse_idl1(bytes: &[u8]) -> Result<ParseResult, ParseError> {
    let mut r = ByteReader::new(bytes);
    let magic = r.bytes(4, "magic")?;
    if magic != b"IDL1" {
        return Err(ParseError::InvalidMagicBytes(format!("Expected IDL1, got: {}", String::from_utf8_lossy(magic))));
    }
    let version = r.u16("format version")?;
    if version != IDL1_FORMAT_VERSION {
        return Err(ParseError::UnsupportedSchemaVersion(format!(
            "Update the app to open this file (.idl1 format v{version}, only v{IDL1_FORMAT_VERSION} supported)"
        )));
    }
    let header_len = r.u32("header_len")? as usize;
    if header_len < FIXED_HEADER_BYTES + HEADER_CRC_BYTES || header_len > bytes.len() {
        return Err(ParseError::TruncatedRecord(format!("header_len {header_len} does not fit a {}-byte file", bytes.len())));
    }
    let stored_crc = u32::from_le_bytes(bytes[header_len - 4..header_len].try_into().expect("4 bytes"));
    let computed_crc = crate::synth::rng::crc32(&bytes[..header_len - 4]);
    if stored_crc != computed_crc {
        return Err(ParseError::TruncatedRecord(format!(
            "header checksum mismatch (stored {stored_crc:08x}, computed {computed_crc:08x}): the file is damaged"
        )));
    }

    let session_id = to_hex(r.bytes(16, "session UUID")?);
    let device_id = to_hex(r.bytes(6, "device ID")?);
    let firmware_version = null_term_str(r.bytes(24, "firmware version")?);
    let session_start_ms = r.i64("session start UTC ms")?;
    let config_crc = r.u32("config CRC32")?;
    let _freq_fine = r.bytes(3, "IMU FREQ_FINE")?;
    let registry_count = r.u16("registry count")? as usize;
    if FIXED_HEADER_BYTES + registry_count * REGISTRY_ENTRY_BYTES + HEADER_CRC_BYTES > header_len {
        return Err(ParseError::TruncatedRecord(format!("{registry_count} registry entries do not fit header_len {header_len}")));
    }
    let mut registry: HashMap<u16, RegistryEntry> = HashMap::new();
    let mut imus: [Imu; 3] = Default::default();
    for _ in 0..registry_count {
        let e = read_registry_entry(&mut r)?;
        if e.kind == KIND_IMU_AXIS && (e.source as usize) < 3 {
            let imu = &mut imus[e.source as usize];
            let stream = if e.path == PATH_HIGH_G { &mut imu.high } else { &mut imu.low };
            stream.axes.push(e.clone());
        }
        registry.insert(e.channel_id, e);
    }
    for imu in imus.iter_mut() {
        for stream in [&mut imu.low, &mut imu.high] {
            stream.axes.sort_by_key(|e| e.axis);
            stream.values = vec![Vec::new(); stream.axes.len()];
        }
    }
    // Header fields this parser doesn't know (a later version's) sit between
    // the registry and the CRC; skip them.
    r.skip(header_len - r.position(), "header remainder")?;
    // Each IMU as clock segments: a new one starts at every "IMU recovered"
    // event, because the restarted chip's TIMESTAMP counter starts over.
    let mut imus: [Vec<Imu>; 3] = imus.map(|imu| vec![imu]);

    let mut acc = ChannelAccumulator::new();
    let mut channel_slot: HashMap<u16, usize> = HashMap::new();
    let mut channel_ts: HashMap<String, Vec<i64>> = HashMap::new();
    let mut gps_ts: Vec<i64> = Vec::new();
    let mut gps_anchor = GpsAnchor::default();
    let mut warnings: Vec<ImportWarning> = Vec::new();
    let mut truncation: Option<ParseError> = None;
    let mut ended = false;

    while r.has_more() && !ended {
        if r.remaining() < 3 {
            truncation = Some(ParseError::TruncatedRecord("file ends inside a record header".to_string()));
            break;
        }
        let kind = r.u8("record type")?;
        let len = r.u16("record length")? as usize;
        if r.remaining() < len {
            truncation = Some(ParseError::TruncatedRecord(format!("file ends inside a 0x{kind:02x} record")));
            break;
        }
        let start = r.position();
        match kind {
            REC_SESSION_END => ended = true,
            REC_IMU_BLOCK => read_imu_block(&mut r, len, &mut imus)?,
            REC_GPS_FIX if len >= GPS_FIX_BYTES => {
                parse_gps_record(&mut r, 32, &mut acc, Some(&mut gps_anchor), None, &mut gps_ts)?;
                let hdop_x100 = r.u16("hdop_x100")?;
                acc.push("GPS_Hdop", hdop_x100 as f64 * 0.01);
            }
            REC_CHANNEL_SAMPLE => read_channel_sample(&mut r, len, &registry, &mut acc, &mut channel_slot, &mut channel_ts)?,
            REC_LOGGER_EVENT => {
                let (warning, event) = read_logger_event(&mut r, len)?;
                warnings.extend(warning);
                if let Some((EVT_IMU_RECOVERED, source)) = event {
                    if let Some(segments) = imus.get_mut(source as usize) {
                        let fresh = segments[0].fresh_segment();
                        segments.push(fresh);
                    }
                }
            }
            REC_SESSION_SUMMARY => warnings.extend(read_session_summary(&mut r, len)?),
            REC_GPS_CONFIG => {
                if let Some(cfg) = read_gps_config(&mut r, len)? {
                    if !cfg.setup_ok() {
                        warnings.push(ImportWarning {
                            kind: ImportWarningKind::LoggerDiagnostic,
                            message: format!(
                                "GPS setup failed ({}): the module recorded on earlier settings at {} Hz, not the {} Hz configured",
                                cfg.result_name(),
                                cfg.applied_rate_hz,
                                cfg.requested_rate_hz
                            ),
                        });
                    }
                }
            }
            _ => {}
        }
        let consumed = r.position() - start;
        if consumed < len {
            r.skip(len - consumed, "record remainder")?;
        }
    }

    // Each IMU's sample times on the ESP32 clock, segment by segment, then
    // joined: (imu, high-g?, times, values per axis, gaps, rate).
    let mut timed: Vec<(usize, bool, Vec<f64>, Vec<Vec<i16>>, Vec<GapSpan>, f64)> = Vec::new();
    for (i, segments) in imus.iter().enumerate() {
        let overruns: usize = segments.iter().map(|s| s.overrun_blocks).sum();
        if overruns > 0 {
            warnings.push(ImportWarning {
                kind: ImportWarningKind::LoggerDiagnostic,
                message: format!("IMU{i}: FIFO overrun flagged in {overruns} reads"),
            });
        }
        for high in [false, true] {
            let mut joined: Option<(Vec<f64>, Vec<Vec<i16>>, Vec<GapSpan>, f64, usize)> = None;
            for imu in segments {
                let stream = if high { &imu.high } else { &imu.low };
                if stream.count == 0 || stream.axes.is_empty() {
                    continue;
                }
                let (esp, source, gaps, rate) = time_stream(i, high, imu, &mut warnings);
                let (times, values, all_gaps, best_rate, best_len) =
                    joined.get_or_insert_with(|| (Vec::new(), vec![Vec::new(); stream.axes.len()], Vec::new(), rate, 0));
                let base = times.len();
                all_gaps.extend(gaps.into_iter().map(|g| GapSpan { start: g.start + base, len: g.len }));
                for (a, axis) in values.iter_mut().enumerate() {
                    axis.extend(fill_lost(&stream.values[a], &source));
                }
                times.extend(esp);
                // The rate of the longest segment speaks for the channel.
                if stream.count > *best_len {
                    *best_len = stream.count;
                    *best_rate = rate;
                }
            }
            if let Some((times, values, gaps, rate, _)) = joined {
                timed.push((i, high, times, values, gaps, rate));
            }
        }
    }

    // Session origin: the earliest time of any source.
    let t0 = timed
        .iter()
        .filter_map(|t| t.2.first().copied())
        .map(|t| t.round() as i64)
        .chain(gps_ts.iter().copied())
        .chain(channel_ts.values().filter_map(|v| v.first().copied()))
        .min()
        .unwrap_or(0);

    let mut channels: Vec<Channel> = Vec::new();
    for (i, high, esp, values, gaps, rate) in timed {
        let stream = if high { &imus[i][0].high } else { &imus[i][0].low };
        let t_us = strictly_increasing(esp.iter().map(|&t| t.round() as i64 - t0).collect());
        let source_kind = if high { format!("imu{i}_hg") } else { format!("imu{i}") };
        for (a, entry) in stream.axes.iter().enumerate() {
            channels.push(Channel {
                channel_id: entry.name.clone(),
                t_us: t_us.clone(),
                t_recorded_us: None,
                nominal_rate_hz: rate,
                column: RawColumn::I16 { data: values[a].clone(), scale: entry.scale, offset: entry.offset },
                source_kind: source_kind.clone(),
                unit: entry.units.clone(),
                gaps: gaps.clone(),
            });
        }
    }
    let units_by_name: HashMap<&str, &str> = registry.values().map(|e| (e.name.as_str(), e.units.as_str())).collect();
    let rate_by_name: HashMap<&str, f64> = registry.values().map(|e| (e.name.as_str(), e.rate_mhz as f64 / 1000.0)).collect();
    for (name, column) in acc.into_entries() {
        let (t_us, source_kind, rate) = if name.starts_with("GPS") {
            (gps_ts.iter().map(|&t| t - t0).collect(), "gps".to_string(), 0.0)
        } else {
            let ts = channel_ts.get(&name).map(|v| v.iter().map(|&t| t - t0).collect()).unwrap_or_default();
            (ts, crate::parse::records::generic_source_kind(&name), rate_by_name.get(name.as_str()).copied().unwrap_or(0.0))
        };
        let unit = gps_unit(&name).map(str::to_string).unwrap_or_else(|| units_by_name.get(name.as_str()).map(|u| u.to_string()).unwrap_or_default());
        channels.push(Channel { channel_id: name, t_us, t_recorded_us: None, nominal_rate_hz: rate, column, source_kind, unit, gaps: Vec::new() });
    }

    merge_high_g(&mut channels);

    // Start time: the header's, else back-filled from the first GPS fix
    // (as `.idl0` §5.6).
    let (timestamp_utc_ms, timestamp_source) = match (session_start_ms, gps_anchor.gps_epoch_ms, gps_anchor.device_ts_us) {
        (0, Some(epoch), Some(dev)) => (epoch - ((dev - t0) as f64 / 1000.0).round() as i64, TimestampSource::GpsBackfill),
        (start, _, _) => (start, TimestampSource::Header),
    };
    if !firmware_version.is_empty() && warnings.iter().any(|w| w.kind == ImportWarningKind::LoggerDiagnostic) {
        warnings.push(ImportWarning { kind: ImportWarningKind::LoggerDiagnostic, message: format!("recorded by firmware {firmware_version}") });
    }

    Ok(ParseResult {
        session: Session {
            session_id,
            device_id: Some(device_id),
            timestamp_utc_ms,
            timestamp_source,
            config_checksum: Some(format!("{config_crc:08x}")),
            source_format: SourceFormat::Idl1,
            blob_sha256: String::new(),
            channels,
        },
        truncation_warning: truncation,
        import_warnings: warnings,
    })
}

/// Low-g readings at or above this fraction of full scale count as
/// saturated, and the merge takes the high-g reading there (Isaac,
/// 2026-10-15: "95% seems right").
const LOW_G_SATURATION: f64 = 0.95;

/// Merges each IMU accel axis's low-g and high-g streams at import (Isaac,
/// 2026-10-14/15; `.idl1` DRAFT spec §4.5). For every axis with both:
///
/// - the merged channel takes the low-g name (`IMU1_AccelX`), so workbooks
///   and charts keep working, and runs on whichever stream has the higher
///   rate, with that stream's times and gaps;
/// - at each of those times it takes the low-g reading, unless a low-g
///   sample around that moment is at or above [`LOW_G_SATURATION`] of the
///   low-g full scale, where it takes the high-g reading instead; the
///   other stream is read by linear interpolation;
/// - high-g is taken as linear and scaled by its registry scale, with no
///   per-session calibration (Isaac: "assume it's scaled correctly and
///   linear... we can play with this after the fact");
/// - the raw streams stay, the low-g one renamed `IMU1_LgAccelX` and the
///   high-g one as recorded (`IMU1_HgAccelX`), so the merge can be checked.
fn merge_high_g(channels: &mut Vec<Channel>) {
    for imu in 0..3 {
        for axis in ["X", "Y", "Z"] {
            let low_name = format!("IMU{imu}_Accel{axis}");
            let high_name = format!("IMU{imu}_HgAccel{axis}");
            let (Some(li), Some(hi)) = (
                channels.iter().position(|c| c.channel_id == low_name),
                channels.iter().position(|c| c.channel_id == high_name),
            ) else {
                continue;
            };
            let full_scale = match channels[li].column {
                RawColumn::I16 { scale, .. } => scale.abs() * 32768.0,
                _ => continue,
            };
            let merged = {
                let (low, high) = (&channels[li], &channels[hi]);
                let (low_v, high_v) = (low.materialize(), high.materialize());
                let grid = if high.nominal_rate_hz > low.nominal_rate_hz { high } else { low };
                let threshold = LOW_G_SATURATION * full_scale;
                let data: Vec<f32> = grid
                    .t_us
                    .iter()
                    .map(|&t| {
                        let (lv, saturated) = sample_at(&low.t_us, &low_v, t, threshold);
                        if saturated {
                            sample_at(&high.t_us, &high_v, t, f64::INFINITY).0 as f32
                        } else {
                            lv as f32
                        }
                    })
                    .collect();
                Channel {
                    channel_id: low_name.clone(),
                    t_us: grid.t_us.clone(),
                    t_recorded_us: None,
                    nominal_rate_hz: grid.nominal_rate_hz,
                    column: RawColumn::F32 { data, scale: 1.0, offset: 0.0 },
                    source_kind: grid.source_kind.clone(),
                    unit: "g".to_string(),
                    gaps: grid.gaps.clone(),
                }
            };
            channels[li].channel_id = format!("IMU{imu}_LgAccel{axis}");
            channels.push(merged);
        }
    }
}

/// `values` (on times `t_us`, sorted) at time `t`, by linear interpolation
/// between the two samples around it (the nearest one past either end),
/// and whether either of those samples is at or above `threshold` in
/// magnitude.
fn sample_at(t_us: &[i64], values: &[f64], t: i64, threshold: f64) -> (f64, bool) {
    if t_us.is_empty() || values.is_empty() {
        return (0.0, false);
    }
    let n = t_us.len().min(values.len());
    let i = t_us[..n].partition_point(|&x| x < t);
    if i == 0 {
        return (values[0], values[0].abs() >= threshold);
    }
    if i >= n {
        return (values[n - 1], values[n - 1].abs() >= threshold);
    }
    let (t0, t1, v0, v1) = (t_us[i - 1], t_us[i], values[i - 1], values[i]);
    let saturated = v0.abs() >= threshold || v1.abs() >= threshold;
    if t1 == t0 {
        return (v1, saturated);
    }
    (v0 + (v1 - v0) * (t - t0) as f64 / (t1 - t0) as f64, saturated)
}

/// GPS channel units, as `.idl0` (C1 §4.1), plus HDOP (unitless).
fn gps_unit(name: &str) -> Option<&'static str> {
    match name {
        "GPS_SpeedKmh" => Some("km/h"),
        "GPS_EpochMs" => Some("ms_raw"),
        "GPS_Latitude" | "GPS_Longitude" | "GPS_Heading" => Some("deg"),
        "GPS_Altitude" => Some("m"),
        "GPS_FixQuality" => Some("enum_raw"),
        "GPS_Satellites" => Some("count"),
        "GPS_Hdop" => Some(""),
        _ => None,
    }
}

/// One clock segment's stream timed on the ESP32 clock: sample times, the
/// received sample behind each slot (`None` = lost), lost runs, and the
/// measured rate in Hz. Falls back to read instants, with a warning, when
/// the segment has too few chip timestamps.
fn time_stream(i: usize, high: bool, imu: &Imu, warnings: &mut Vec<ImportWarning>) -> (Vec<f64>, Vec<Option<usize>>, Vec<GapSpan>, f64) {
    let stream = if high { &imu.high } else { &imu.low };
    let (pairs, low_marks, high_marks) = unwrap_chip_times(imu);
    let marks = if high { high_marks } else { low_marks };
    let rate_hz = stream.axes[0].rate_mhz as f64 / 1000.0;
    let map = idl1_timing::fit_clock_map(&pairs, CLOCK_FIT_WINDOW_US);
    match (map.as_ref(), idl1_timing::place_samples(stream.count, &marks)) {
        (Some(map), Some(placed)) => {
            let esp: Vec<f64> = placed.chip.iter().map(|&c| map.map(c)).collect();
            let gaps = placed.lost.iter().map(|l| GapSpan { start: l.start, len: l.len }).collect();
            let period_us = placed.period_ticks * map.mean_slope();
            let rate = if period_us > 0.0 { 1e6 / period_us } else { rate_hz };
            (esp, placed.source, gaps, rate)
        }
        _ => {
            let label = if high { format!("IMU{i} high-g") } else { format!("IMU{i}") };
            let reason = if map.is_some() && marks.len() >= 2 {
                "its chip timestamps imply far more lost samples than arrived (a broken clock, not loss), so it was not timed by the chip clock"
            } else {
                "too few chip timestamps to time samples by the chip clock"
            };
            warnings.push(ImportWarning {
                kind: ImportWarningKind::TimingFallback,
                message: format!("{label}: {reason}; timed from read instants at the nominal {rate_hz} Hz"),
            });
            let esp = read_instant_times(&stream.blocks, rate_hz);
            let source = (0..stream.count).map(Some).collect();
            (esp, source, Vec::new(), rate_hz)
        }
    }
}

/// Reads one IMU_BLOCK (spec §4.4) into the current segment of its IMU. A block for an IMU index
/// past 2, or one too short for its own counts, is skipped whole.
fn read_imu_block(r: &mut ByteReader, len: usize, imus: &mut [Vec<Imu>; 3]) -> Result<(), ParseError> {
    if len < IMU_BLOCK_FIXED_BYTES {
        return Ok(());
    }
    let idx = r.u8("imu_index")? as usize;
    let flags = r.u8("flags")?;
    let read_us = r.i64("read_us")?;
    let read_chip_ts = r.u32("read_chip_ts")?;
    let n_samples = r.u16("n_samples")? as usize;
    let n_hg = r.u16("n_hg")? as usize;
    let n_ts = r.u8("n_ts")? as usize;
    let Some(imu) = imus.get_mut(idx).and_then(|segments| segments.last_mut()) else { return Ok(()) };
    let need = IMU_BLOCK_FIXED_BYTES + n_samples * 2 * imu.low.axes.len() + n_hg * 2 * imu.high.axes.len() + n_ts * TS_WORD_BYTES;
    if need > len {
        return Ok(());
    }
    if flags & 1 == 1 {
        imu.overrun_blocks += 1;
    }
    for stream_n in [(false, n_samples), (true, n_hg)] {
        let stream = if stream_n.0 { &mut imu.high } else { &mut imu.low };
        for _ in 0..stream_n.1 {
            for axis in stream.values.iter_mut() {
                axis.push(r.i16("IMU sample")?);
            }
        }
    }
    let (low_base, high_base) = (imu.low.count, imu.high.count);
    for _ in 0..n_ts {
        let after_sample = r.u16("ts after_sample")? as usize;
        let after_hg = r.u16("ts after_hg")? as usize;
        let chip_ts = r.u32("ts chip_ts")?;
        // A word stamps the next low-g sample and, by FIFO order, the next
        // high-g sample.
        if after_sample < n_samples {
            imu.chip_order.push(ChipUse::LowMark(imu.low.marks.len()));
            imu.low.marks.push((low_base + after_sample, chip_ts));
        }
        if after_hg < n_hg {
            imu.chip_order.push(ChipUse::HighMark(imu.high.marks.len()));
            imu.high.marks.push((high_base + after_hg, chip_ts));
        }
    }
    imu.chip_order.push(ChipUse::Read(imu.reads.len()));
    imu.reads.push((read_chip_ts, read_us));
    imu.low.count += n_samples;
    imu.high.count += n_hg;
    imu.low.blocks.push((read_us, n_samples));
    imu.high.blocks.push((read_us, n_hg));
    Ok(())
}

/// Unwraps one IMU's chip timestamps as the single sequence they were
/// produced in, then splits them into clock pairs and the two streams'
/// marks.
fn unwrap_chip_times(imu: &Imu) -> (Vec<ClockPair>, Vec<TsMark>, Vec<TsMark>) {
    let mut raw: Vec<u32> = imu
        .chip_order
        .iter()
        .map(|u| match *u {
            ChipUse::Read(k) => imu.reads[k].0,
            ChipUse::LowMark(k) => imu.low.marks[k].1,
            ChipUse::HighMark(k) => imu.high.marks[k].1,
        })
        .collect();
    unshift_read_ts(&imu.chip_order, &mut raw);
    let unwrapped = idl1_timing::unwrap_u32(&raw);
    let (mut pairs, mut low, mut high) = (Vec::new(), Vec::new(), Vec::new());
    for (u, chip) in imu.chip_order.iter().zip(unwrapped) {
        match *u {
            ChipUse::Read(k) => pairs.push(ClockPair { chip, esp_us: imu.reads[k].1 }),
            ChipUse::LowMark(k) => low.push(TsMark { sample_index: imu.low.marks[k].0, chip }),
            ChipUse::HighMark(k) => high.push(TsMark { sample_index: imu.high.marks[k].0, chip }),
        }
    }
    (pairs, low, high)
}

/// Firmware through 0.3.0-beta.16 wrote each block's `read_chip_ts` as the
/// TIMESTAMP register shifted left one byte (ST's vendored driver shifted
/// once too often; fixed in idl1-firmware 2026-11-02): low byte zero, top
/// byte lost. Unwrapped as is, every read→mark step looked like a 2^32-tick
/// counter wrap. Recognised when every read (at least 4) has a zero low byte
/// and its upper three bytes match the nearest earlier mark's lower three
/// to within 2^16 ticks (≈1.4 s); each read is then rebuilt next to that
/// mark. Correct data never matches: its reads' low bytes are not all zero.
fn unshift_read_ts(order: &[ChipUse], raw: &mut [u32]) {
    const MATCH_TICKS: i64 = 1 << 16;
    let first_mark = order.iter().zip(raw.iter()).find(|(u, _)| !matches!(u, ChipUse::Read(_))).map(|(_, &v)| v);
    let Some(mut mark) = first_mark else { return };
    // Per read: its index and its offset from the mark, from the 24 bits kept.
    let mut fixes = Vec::new();
    for (i, u) in order.iter().enumerate() {
        if !matches!(u, ChipUse::Read(_)) {
            mark = raw[i];
            continue;
        }
        if raw[i] & 0xFF != 0 {
            return;
        }
        let d24 = ((raw[i] >> 8).wrapping_sub(mark) & 0xFF_FFFF) as i64;
        let d = if d24 >= 1 << 23 { d24 - (1 << 24) } else { d24 };
        if d.abs() >= MATCH_TICKS {
            return;
        }
        fixes.push((i, mark.wrapping_add(d as i32 as u32)));
    }
    if fixes.len() < 4 {
        return;
    }
    for (i, v) in fixes {
        raw[i] = v;
    }
}

/// The fallback without chip timing: each block's newest sample at its read
/// instant, earlier ones at the nominal period before it (the `.idl0` §5.5
/// rule).
fn read_instant_times(blocks: &[(i64, usize)], rate_hz: f64) -> Vec<f64> {
    let period = if rate_hz > 0.0 { 1e6 / rate_hz } else { 0.0 };
    blocks
        .iter()
        .flat_map(|&(read_us, n)| (0..n).map(move |k| read_us as f64 - (n - 1 - k) as f64 * period))
        .collect()
}

/// `values` laid out on `source`'s slots: received samples where `source`
/// names one, lost slots filled linearly between their neighbours (held
/// at the edges).
fn fill_lost(values: &[i16], source: &[Option<usize>]) -> Vec<i16> {
    let mut out: Vec<i16> = Vec::with_capacity(source.len());
    let mut i = 0;
    while i < source.len() {
        match source[i] {
            Some(k) => {
                out.push(values.get(k).copied().unwrap_or(0));
                i += 1;
            }
            None => {
                let run_end = (i..source.len()).find(|&j| source[j].is_some()).unwrap_or(source.len());
                let before = out.last().copied();
                let after = source.get(run_end).copied().flatten().and_then(|k| values.get(k).copied());
                let len = run_end - i;
                for j in 0..len {
                    let v = match (before, after) {
                        (Some(a), Some(b)) => a as f64 + (b as f64 - a as f64) * (j + 1) as f64 / (len + 1) as f64,
                        (Some(a), None) => a as f64,
                        (None, Some(b)) => b as f64,
                        (None, None) => 0.0,
                    };
                    out.push(v.round() as i16);
                }
                i = run_end;
            }
        }
    }
    out
}

/// `t` made strictly increasing (C1 §3.5 invariant 1): a value not past its
/// predecessor is moved 1 µs past it.
fn strictly_increasing(mut t: Vec<i64>) -> Vec<i64> {
    for i in 1..t.len() {
        if t[i] <= t[i - 1] {
            t[i] = t[i - 1] + 1;
        }
    }
    t
}

/// Reads one CHANNEL_SAMPLE (spec §4.3: `.idl0` §5.7 with a u16 id). A
/// sample for an id the registry doesn't list is skipped.
fn read_channel_sample(
    r: &mut ByteReader,
    len: usize,
    registry: &HashMap<u16, RegistryEntry>,
    acc: &mut ChannelAccumulator,
    slots: &mut HashMap<u16, usize>,
    channel_ts: &mut HashMap<String, Vec<i64>>,
) -> Result<(), ParseError> {
    if len < 10 {
        return Ok(());
    }
    let id = r.u16("channel_id")?;
    let ts = r.i64("timestamp_us")?;
    let Some(e) = registry.get(&id) else { return Ok(()) };
    let Some(width) = DATA_TYPE_WIDTHS.get(e.data_type as usize) else { return Ok(()) };
    if 10 + width > len {
        return Ok(());
    }
    let slot = *slots.entry(id).or_insert_with(|| match e.data_type {
        4 => acc.slot_for_i16(&e.name, e.scale, e.offset),
        5 => acc.slot_for_i32(&e.name, e.scale, e.offset),
        6 => acc.slot_for_f32(&e.name, e.scale, e.offset),
        _ => acc.slot_for(&e.name),
    });
    match e.data_type {
        4 => acc.push_i16_at(slot, r.i16("i16 value")?),
        5 => acc.push_i32_at(slot, r.i32("i32 value")?),
        6 => acc.push_f32_at(slot, r.f32("f32 value")?),
        _ => acc.push_at(slot, read_typed_value(r, e.data_type)? * e.scale + e.offset),
    }
    channel_ts.entry(e.name.clone()).or_default().push(ts);
    Ok(())
}

/// LOGGER_EVENT codes (spec §4.3).
fn event_name(code: u8) -> &'static str {
    match code {
        1 => "FIFO overrun",
        2 => "writer buffer full, data dropped",
        3 => "bus error",
        4 => "IMU stopped answering",
        5 => "IMU recovered",
        _ => "unknown logger event",
    }
}

/// Reads one LOGGER_EVENT (spec §4.3: code u8, source u16, count u32,
/// timestamp_us i64) as an import warning, plus its `(code, source)`.
fn read_logger_event(r: &mut ByteReader, len: usize) -> Result<(Option<ImportWarning>, Option<(u8, u16)>), ParseError> {
    if len < 15 {
        return Ok((None, None));
    }
    let code = r.u8("event code")?;
    let source = r.u16("event source")?;
    let count = r.u32("event count")?;
    let ts = r.i64("event timestamp_us")?;
    let warning = ImportWarning {
        kind: ImportWarningKind::LoggerDiagnostic,
        message: format!("logger: {} (source {source}) ×{count} at device time {:.3} s", event_name(code), ts as f64 / 1e6),
    };
    Ok((Some(warning), Some((code, source))))
}

/// Reads SESSION_SUMMARY (spec §4.3: n_imu u8; per IMU overruns u32,
/// bus_errors u32, samples u32; then writer_drops u32, slowest_cycle_us
/// u32). Becomes one warning when anything went wrong, none otherwise.
fn read_session_summary(r: &mut ByteReader, len: usize) -> Result<Option<ImportWarning>, ParseError> {
    if len < 1 {
        return Ok(None);
    }
    let n = r.u8("summary n_imu")? as usize;
    if len < 1 + n * 12 + 8 {
        return Ok(None);
    }
    let mut parts = Vec::new();
    for i in 0..n {
        let overruns = r.u32("summary overruns")?;
        let bus_errors = r.u32("summary bus_errors")?;
        let _samples = r.u32("summary samples")?;
        if overruns > 0 || bus_errors > 0 {
            parts.push(format!("IMU{i} {overruns} overruns, {bus_errors} bus errors"));
        }
    }
    let writer_drops = r.u32("summary writer_drops")?;
    let slowest = r.u32("summary slowest_cycle_us")?;
    if writer_drops > 0 {
        parts.push(format!("{writer_drops} writer drops"));
    }
    Ok((!parts.is_empty()).then(|| ImportWarning {
        kind: ImportWarningKind::LoggerDiagnostic,
        message: format!("logger session summary: {}; slowest poll cycle {slowest} µs", parts.join(", ")),
    }))
}

/// The GPS_CONFIG record (`.idl1` DRAFT spec §4.3, 0x22): the SAM-M10Q
/// settings a session recorded with (SPEC §8 "SAM-M10Q (IDL1)") and whether
/// the module accepted them.
#[derive(Debug, Clone, PartialEq)]
pub struct GpsConfig {
    /// Setup result code: 0 not run, 1 OK, 2 no response, 3 baud change
    /// failed, 4 refused (NAK), 5 no acknowledgement, 6 encode error.
    pub result: u8,
    /// Hz, as configured.
    pub requested_rate_hz: u8,
    /// Hz, as the module acknowledged (1, factory, when setup failed first).
    pub applied_rate_hz: u8,
    /// Bit 0 GPS, 1 Galileo, 2 BeiDou, 3 GLONASS, 4 QZSS, 5 SBAS.
    pub signals: u8,
    /// u-blox CFG-NAVSPG-DYNMODEL value.
    pub dynamic_model: u8,
    /// Degrees.
    pub min_elevation_deg: i8,
    /// dBHz.
    pub min_cno_dbhz: u8,
    pub min_satellites: u8,
    pub max_satellites: u8,
    /// 0 = off.
    pub fix_gate_satellites: u8,
    /// dBHz.
    pub fix_gate_cno_dbhz: u8,
    /// Position DOP mask (the record's × 10 value, scaled back).
    pub pdop_mask: f64,
    /// Metres.
    pub position_accuracy_mask_m: u16,
    pub high_precision_nmea: bool,
    /// Bit 0 GLL, 1 GSA, 2 GSV, 3 VTG, 4 ZDA.
    pub nmea_extra: u8,
    /// UBX-MON-VER swVersion; empty when the module did not answer.
    pub module_version: String,
}

/// GPS_CONFIG payload bytes for record version 1.
const GPS_CONFIG_BYTES: usize = 48;

impl GpsConfig {
    /// True when the module acknowledged every setting.
    pub fn setup_ok(&self) -> bool {
        self.result == 1
    }

    /// The §7.3 `GPSConfig` token for `result`.
    pub fn result_name(&self) -> &'static str {
        match self.result {
            1 => "OK",
            2 => "NO_RESPONSE",
            3 => "BAUD_FAILED",
            4 => "NAK",
            5 => "NO_ACK",
            6 => "ENCODE",
            _ => "NOT_RUN",
        }
    }

    /// The SPEC §8 `gps.dynamic_model` name for `dynamic_model`.
    pub fn dynamic_model_name(&self) -> String {
        match self.dynamic_model {
            0 => "portable".to_string(),
            2 => "stationary".to_string(),
            3 => "pedestrian".to_string(),
            4 => "automotive".to_string(),
            5 => "sea".to_string(),
            6 => "airborne".to_string(),
            7 => "airborne_2g".to_string(),
            8 => "airborne_4g".to_string(),
            other => format!("model {other}"),
        }
    }

    /// The enabled constellations and augmentations, e.g. `"GPS+Galileo, QZSS, SBAS"`.
    pub fn signals_text(&self) -> String {
        let names = ["GPS", "Galileo", "BeiDou", "GLONASS"];
        let major: Vec<&str> = names.iter().enumerate().filter(|(i, _)| self.signals & (1 << i) != 0).map(|(_, n)| *n).collect();
        let mut parts = vec![if major.is_empty() { "none".to_string() } else { major.join("+") }];
        if self.signals & (1 << 4) != 0 {
            parts.push("QZSS".to_string());
        }
        if self.signals & (1 << 5) != 0 {
            parts.push("SBAS".to_string());
        }
        parts.join(", ")
    }
}

impl std::fmt::Display for GpsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Hz (module accepted {} Hz), {}, {}, elevation >= {} deg, C/N0 >= {} dBHz, satellites {}-{}",
            self.requested_rate_hz,
            self.applied_rate_hz,
            self.signals_text(),
            self.dynamic_model_name(),
            self.min_elevation_deg,
            self.min_cno_dbhz,
            self.min_satellites,
            self.max_satellites
        )?;
        if self.fix_gate_satellites > 0 {
            write!(f, ", fix needs {} above {} dBHz", self.fix_gate_satellites, self.fix_gate_cno_dbhz)?;
        }
        write!(f, ", PDOP <= {:.1}, accuracy <= {} m", self.pdop_mask, self.position_accuracy_mask_m)?;
        if self.high_precision_nmea {
            write!(f, ", high-precision NMEA")?;
        }
        write!(f, "; setup {}", self.result_name())?;
        if !self.module_version.is_empty() {
            write!(f, "; module {}", self.module_version)?;
        }
        Ok(())
    }
}

/// Reads GPS_CONFIG. A record of an unknown version, or shorter than
/// version 1's payload, is skipped (None).
fn read_gps_config(r: &mut ByteReader, len: usize) -> Result<Option<GpsConfig>, ParseError> {
    if len < GPS_CONFIG_BYTES {
        return Ok(None);
    }
    if r.u8("gps_config version")? != 1 {
        return Ok(None);
    }
    Ok(Some(GpsConfig {
        result: r.u8("gps_config result")?,
        requested_rate_hz: r.u8("gps_config requested_rate_hz")?,
        applied_rate_hz: r.u8("gps_config applied_rate_hz")?,
        signals: r.u8("gps_config signals")?,
        dynamic_model: r.u8("gps_config dynamic_model")?,
        min_elevation_deg: r.i8("gps_config min_elevation_deg")?,
        min_cno_dbhz: r.u8("gps_config min_cno_dbhz")?,
        min_satellites: r.u8("gps_config min_satellites")?,
        max_satellites: r.u8("gps_config max_satellites")?,
        fix_gate_satellites: r.u8("gps_config fix_gate_satellites")?,
        fix_gate_cno_dbhz: r.u8("gps_config fix_gate_cno_dbhz")?,
        pdop_mask: r.u16("gps_config pdop_mask_x10")? as f64 / 10.0,
        position_accuracy_mask_m: r.u16("gps_config position_accuracy_mask_m")?,
        high_precision_nmea: r.u8("gps_config nmea_flags")? & 1 != 0,
        nmea_extra: r.u8("gps_config nmea_extra")?,
        module_version: null_term_str(r.bytes(30, "gps_config module_version")?),
    }))
}

/// The GPS_CONFIG record of an `.idl1` file, if it has one. Walks the
/// records from the end of the header; the firmware writes GPS_CONFIG
/// before any data record, so this stops at the first one it finds. None
/// for a file that is not `.idl1`, predates the record, or is damaged.
pub fn read_idl1_gps_config(bytes: &[u8]) -> Option<GpsConfig> {
    if bytes.len() < 10 || &bytes[..4] != b"IDL1" {
        return None;
    }
    let header_len = u32::from_le_bytes(bytes[6..10].try_into().ok()?) as usize;
    let mut r = ByteReader::new(bytes.get(header_len..)?);
    while r.remaining() >= 3 {
        let kind = r.u8("record type").ok()?;
        let len = r.u16("record length").ok()? as usize;
        if r.remaining() < len {
            return None;
        }
        if kind == REC_GPS_CONFIG {
            return read_gps_config(&mut r, len).ok()?;
        }
        if kind == REC_SESSION_END {
            return None;
        }
        r.skip(len, "record").ok()?;
    }
    None
}

/// Builds `.idl1` files for tests (spec §4). Exposed under the `test-fixtures`
/// feature so `idl-library`'s import tests can build logger files.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_builder {
    use super::*;

    /// Builds `.idl1` files for tests (spec §4).
    pub struct Builder {
        registry: Vec<Vec<u8>>,
        records: Vec<u8>,
        /// Session start written to the header, UTC ms.
        pub session_start_ms: i64,
    }

    impl Builder {
        /// Test builder step `new` (see the `.idl1` spec §4 for the record it writes).
        pub fn new() -> Self {
            Self { registry: Vec::new(), records: Vec::new(), session_start_ms: 1_760_000_000_000 }
        }

        /// Test builder step `channel` (see the `.idl1` spec §4 for the record it writes).
        #[allow(clippy::too_many_arguments)]
        pub fn channel(&mut self, id: u16, data_type: u8, kind: u8, source: u8, axis: u8, path: u8, rate_hz: u32, scale: f32, name: &str, units: &str) -> &mut Self {
            let mut e = Vec::new();
            e.extend(id.to_le_bytes());
            e.extend([data_type, kind, source, axis, path, 0]);
            e.extend((rate_hz * 1000).to_le_bytes());
            e.extend(scale.to_le_bytes());
            e.extend(0f32.to_le_bytes());
            let mut n = [0u8; 24];
            n[..name.len()].copy_from_slice(name.as_bytes());
            e.extend(n);
            let mut u = [0u8; 8];
            u[..units.len()].copy_from_slice(units.as_bytes());
            e.extend(u);
            assert_eq!(e.len(), REGISTRY_ENTRY_BYTES);
            self.registry.push(e);
            self
        }

        /// The six low-g/gyro axes of IMU `imu` at `rate_hz`.
        pub fn imu_axes(&mut self, imu: u8, rate_hz: u32) -> &mut Self {
            let names = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"];
            for (a, n) in names.iter().enumerate() {
                let (scale, units) = if a < 3 { (0.000488f32, "g") } else { (0.07f32, "dps") };
                self.channel(imu as u16 * 6 + a as u16, 4, KIND_IMU_AXIS, imu, a as u8, 0, rate_hz, scale, &format!("IMU{imu}_{n}"), units);
            }
            self
        }

        /// The three high-g axes of IMU `imu` at `rate_hz`, ±64 g.
        pub fn hg_axes(&mut self, imu: u8, rate_hz: u32) -> &mut Self {
            for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
                self.channel(18 + imu as u16 * 3 + a as u16, 4, KIND_IMU_AXIS, imu, a as u8, PATH_HIGH_G, rate_hz, 0.001952, &format!("IMU{imu}_HgAccel{n}"), "g");
            }
            self
        }

        fn record(&mut self, kind: u8, payload: &[u8]) -> &mut Self {
            self.records.push(kind);
            self.records.extend((payload.len() as u16).to_le_bytes());
            self.records.extend(payload);
            self
        }

        /// An IMU_BLOCK with `samples` (6 axes each), `hg` triplets and
        /// timestamp words `(after_sample, after_hg, chip_ts)`.
        pub fn imu_block(&mut self, imu: u8, flags: u8, read_us: i64, read_chip: u32, samples: &[[i16; 6]], hg: &[[i16; 3]], ts: &[(u16, u16, u32)]) -> &mut Self {
            let mut p = vec![imu, flags];
            p.extend(read_us.to_le_bytes());
            p.extend(read_chip.to_le_bytes());
            p.extend((samples.len() as u16).to_le_bytes());
            p.extend((hg.len() as u16).to_le_bytes());
            p.push(ts.len() as u8);
            for s in samples {
                for v in s {
                    p.extend(v.to_le_bytes());
                }
            }
            for s in hg {
                for v in s {
                    p.extend(v.to_le_bytes());
                }
            }
            for &(a, h, c) in ts {
                p.extend(a.to_le_bytes());
                p.extend(h.to_le_bytes());
                p.extend(c.to_le_bytes());
            }
            self.record(REC_IMU_BLOCK, &p)
        }

        /// Test builder step `channel_sample_u32` (see the `.idl1` spec §4 for the record it writes).
        pub fn channel_sample_u32(&mut self, id: u16, ts_us: i64, value: u32) -> &mut Self {
            let mut p = id.to_le_bytes().to_vec();
            p.extend(ts_us.to_le_bytes());
            p.extend(value.to_le_bytes());
            self.record(REC_CHANNEL_SAMPLE, &p)
        }

        /// Test builder step `gps_fix` (see the `.idl1` spec §4 for the record it writes).
        pub fn gps_fix(&mut self, epoch_ms: i64, device_us: i64, hdop_x100: u16) -> &mut Self {
            let mut p = epoch_ms.to_le_bytes().to_vec();
            p.extend(device_us.to_le_bytes());
            p.extend(515_250_000i32.to_le_bytes());
            p.extend((-1_234_567i32).to_le_bytes());
            p.extend(500i16.to_le_bytes());
            p.extend(1000u16.to_le_bytes());
            p.extend(18000u16.to_le_bytes());
            p.extend([1u8, 9u8]);
            p.extend(hdop_x100.to_le_bytes());
            self.record(REC_GPS_FIX, &p)
        }

        /// Test builder step `logger_event` (see the `.idl1` spec §4 for the record it writes).
        pub fn logger_event(&mut self, code: u8, source: u16, count: u32, ts_us: i64) -> &mut Self {
            let mut p = vec![code];
            p.extend(source.to_le_bytes());
            p.extend(count.to_le_bytes());
            p.extend(ts_us.to_le_bytes());
            self.record(REC_LOGGER_EVENT, &p)
        }

        /// A GPS_CONFIG record, laid out as idl1-firmware's
        /// `idl1_format_gps_config` writes it: 8 Hz GPS+Galileo+BeiDou,
        /// automotive, factory filters.
        pub fn gps_config(&mut self, result: u8, applied_rate_hz: u8) -> &mut Self {
            let mut p = vec![1u8, result, 8, applied_rate_hz, 0x07 | 0x10 | 0x20, 4, 5, 6, 3, 32, 0, 0];
            p.extend(250u16.to_le_bytes());
            p.extend(100u16.to_le_bytes());
            p.extend([1u8, 0u8]);
            let mut version = [0u8; 30];
            version[..12].copy_from_slice(b"ROM SPG 5.10");
            p.extend(version);
            self.record(REC_GPS_CONFIG, &p)
        }

        /// Test builder step `end` (see the `.idl1` spec §4 for the record it writes).
        pub fn end(&mut self) -> &mut Self {
            self.record(REC_SESSION_END, &[])
        }

        /// Test builder step `build` (see the `.idl1` spec §4 for the record it writes).
        pub fn build(&self) -> Vec<u8> {
            let header_len = FIXED_HEADER_BYTES + self.registry.len() * REGISTRY_ENTRY_BYTES + HEADER_CRC_BYTES;
            let mut h = b"IDL1".to_vec();
            h.extend(IDL1_FORMAT_VERSION.to_le_bytes());
            h.extend((header_len as u32).to_le_bytes());
            h.extend((1..=16u8).collect::<Vec<_>>());
            h.extend([0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5]);
            let mut fw = [0u8; 24];
            fw[..5].copy_from_slice(b"0.3.0");
            h.extend(fw);
            h.extend(self.session_start_ms.to_le_bytes());
            h.extend(0xCAFE_BABEu32.to_le_bytes());
            h.extend([0u8, 0, 0]);
            h.extend((self.registry.len() as u16).to_le_bytes());
            for e in &self.registry {
                h.extend(e);
            }
            assert_eq!(h.len() + 4, header_len);
            let crc = crate::synth::rng::crc32(&h);
            h.extend(crc.to_le_bytes());
            h.extend(&self.records);
            h
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) use super::test_builder::Builder;

    /// `seconds` of one IMU at 480 Hz, chip tick 21.75 µs, drained every
    /// 20 ms (10 samples per read at this rate, rounding), a timestamp word
    /// every 32 samples. Sample k's value is k on every axis.
    fn imu_stream(b: &mut Builder, imu: u8, seconds: i64, lose: Option<(usize, usize)>) {
        let tick_us = 21.75_f64;
        let period_us = 1e6 / 480.0;
        let total = (seconds as f64 * 480.0) as usize;
        let mut emitted = 0usize; // received samples written so far
        let mut k = 0usize; // true sample index on the chip
        let mut read_us = 20_000i64;
        while k < total {
            let mut samples = Vec::new();
            let mut ts = Vec::new();
            while k < total && (k as f64 * period_us) < read_us as f64 {
                let is_lost = matches!(lose, Some((at, n)) if k >= at && k < at + n);
                if !is_lost {
                    if k % 32 == 0 {
                        ts.push((samples.len() as u16, 0, ((k as f64 * period_us) / tick_us).round() as u32));
                    }
                    samples.push([k as i16; 6]);
                    emitted += 1;
                }
                k += 1;
            }
            let chip_at_read = (read_us as f64 / tick_us).round() as u32;
            b.imu_block(imu, 0, read_us + 300, chip_at_read, &samples, &[], &ts);
            read_us += 20_000;
        }
        let _ = emitted;
    }

    /// One second of IMU `imu` with low-g at 480 Hz and high-g at 960 Hz.
    /// The true acceleration on every axis is 5 g, except 30 g from 0.4 s to
    /// 0.5 s: the low-g stream (±16 g) clips there, the high-g (±64 g)
    /// doesn't.
    fn imu_stream_with_hg(b: &mut Builder, imu: u8) {
        let tick_us = 21.75_f64;
        let period_us = 1e6 / 480.0;
        let accel = |t_us: f64| if (400_000.0..500_000.0).contains(&t_us) { 30.0 } else { 5.0 };
        let low_raw = |g: f64| (g / 0.000488).round().clamp(-32768.0, 32767.0) as i16;
        let high_raw = |g: f64| (g / 0.001952).round() as i16;
        let mut k = 0usize;
        let mut read_us = 20_000i64;
        while k < 480 {
            let (mut samples, mut hg, mut ts) = (Vec::new(), Vec::new(), Vec::new());
            while k < 480 && (k as f64 * period_us) < read_us as f64 {
                let t = k as f64 * period_us;
                if k % 32 == 0 {
                    ts.push((samples.len() as u16, hg.len() as u16, (t / tick_us).round() as u32));
                }
                let g = accel(t);
                samples.push([low_raw(g), low_raw(g), low_raw(g), 0, 0, 0]);
                hg.push([high_raw(g); 3]);
                hg.push([high_raw(accel(t + period_us / 2.0)); 3]);
                k += 1;
            }
            b.imu_block(imu, 0, read_us + 300, (read_us as f64 / tick_us).round() as u32, &samples, &hg, &ts);
            read_us += 20_000;
        }
    }

    #[test]
    fn parse_idl1_high_g_merges_into_the_accel_channel_where_low_g_saturates() {
        // Arrange
        let mut b = Builder::new();
        b.imu_axes(1, 480).hg_axes(1, 960);
        imu_stream_with_hg(&mut b, 1);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert — the merged channel runs at the high-g rate, reads ~5 g
        // outside the clipped stretch and ~30 g inside it.
        let merged = find(&r, "IMU1_AccelX");
        assert_eq!(merged.t_us.len(), 960);
        assert_eq!(merged.unit, "g");
        let v = merged.materialize();
        let at = |t_us: i64| v[merged.t_us.iter().position(|&t| t >= t_us).unwrap()];
        assert!((at(200_000) - 5.0).abs() < 0.01, "quiet: {}", at(200_000));
        assert!((at(450_000) - 30.0).abs() < 0.05, "clipped stretch: {}", at(450_000));
        // Raw streams kept for checking the merge.
        assert!(find(&r, "IMU1_LgAccelX").materialize().iter().any(|&g| g > 15.0));
        assert_eq!(find(&r, "IMU1_HgAccelX").t_us.len(), 960);
        // Gyro is untouched.
        assert!(r.session.channels.iter().any(|c| c.channel_id == "IMU1_GyroX"));
    }

    /// Written by the firmware's own encoders (idl1-firmware
    /// `test/gen_idl1_golden.c`), so the parser and the firmware can't drift
    /// apart unnoticed. Regenerate with that program after an encoder change.
    const FIRMWARE_GOLDEN: &[u8] = include_bytes!("testdata/firmware_golden.idl1");

    #[test]
    fn parse_idl1_firmware_golden_file_reads_back_what_the_firmware_wrote() {
        // Act
        let r = parse_idl1(FIRMWARE_GOLDEN).unwrap();

        // Assert — identity and start time.
        assert_eq!(r.session.session_id, "101112131415161718191a1b1c1d1e1f");
        assert_eq!(r.session.device_id.as_deref(), Some("b0b1b2b3b4b5"));
        assert_eq!(r.session.config_checksum.as_deref(), Some("12345678"));
        assert_eq!(r.session.timestamp_source, TimestampSource::GpsBackfill);
        assert!(r.truncation_warning.is_none());
        assert!(!r.import_warnings.iter().any(|w| w.kind == ImportWarningKind::TimingFallback), "{:?}", r.import_warnings);
        assert!(r.import_warnings.iter().any(|w| w.message.contains("FIFO overrun")));
        assert!(r.import_warnings.iter().any(|w| w.message.contains("session summary")));

        // Low-g: 2 s at 480 Hz, chip-timed, value k at sample k.
        let lg = find(&r, "IMU1_LgAccelX");
        assert_eq!(lg.t_us.len(), 960);
        assert!(lg.gaps.is_empty());
        assert!((lg.nominal_rate_hz - 480.0).abs() < 0.5, "rate {}", lg.nominal_rate_hz);
        let spacing = (lg.t_us[959] - lg.t_us[0]) as f64 / 959.0;
        assert!((spacing - 1e6 / 480.0).abs() < 0.5, "spacing {spacing}");
        assert!((lg.materialize()[500] - 500.0 * 0.000488).abs() < 1e-6);

        // High-g: 1920 samples, value 2k + h; the merged channel runs on it.
        let hg = find(&r, "IMU1_HgAccelX");
        assert_eq!(hg.t_us.len(), 1920);
        assert_eq!(hg.source_kind, "imu1_hg");
        assert_eq!(find(&r, "IMU1_AccelX").t_us.len(), 1920);

        // Slow channels and GPS keep their own times and values.
        assert_eq!(find(&r, "HR_BPM").materialize(), vec![142.0]);
        assert_eq!(find(&r, "WheelFront").materialize(), vec![3.0]);
        assert!((find(&r, "GPS_Hdop").materialize()[0] - 0.85).abs() < 1e-12);
    }

    #[test]
    fn sample_at_interpolates_and_flags_saturated_neighbours() {
        // Arrange
        let t = [0i64, 10, 20];
        let v = [0.0, 10.0, 16.0];

        // Act / Assert
        assert_eq!(sample_at(&t, &v, 5, 15.0), (5.0, false));
        assert_eq!(sample_at(&t, &v, 15, 15.0), (13.0, true));
        assert_eq!(sample_at(&t, &v, 30, 15.0), (16.0, true));
        assert_eq!(sample_at(&t, &v, -5, 15.0), (0.0, false));
    }

    fn find<'a>(r: &'a ParseResult, name: &str) -> &'a Channel {
        r.session.channels.iter().find(|c| c.channel_id == name).unwrap_or_else(|| panic!("no channel {name}"))
    }

    #[test]
    fn parse_idl1_clean_stream_times_every_sample_on_the_chip_clock() {
        // Arrange
        let mut b = Builder::new();
        b.imu_axes(0, 480);
        imu_stream(&mut b, 0, 5, None);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert — 2400 samples, ~2083.3 µs apart, no gaps, offset of the
        // 300 µs read latency absorbed by the clock map's intercept.
        let ax = find(&r, "IMU0_AccelX");
        assert_eq!(ax.t_us.len(), 2400);
        assert!(ax.gaps.is_empty());
        assert!((ax.nominal_rate_hz - 480.0).abs() < 0.5, "rate {}", ax.nominal_rate_hz);
        let spacing = (ax.t_us[2399] - ax.t_us[0]) as f64 / 2399.0;
        assert!((spacing - 1e6 / 480.0).abs() < 0.5, "spacing {spacing}");
        assert_eq!(ax.source_kind, "imu0");
        assert_eq!(r.session.source_format, SourceFormat::Idl1);
        assert_eq!(r.session.session_id, "0102030405060708090a0b0c0d0e0f10");
        assert!(r.import_warnings.is_empty(), "{:?}", r.import_warnings);
    }

    /// `seconds` of IMU `imu` at 480 Hz starting at ESP32 time `start_us`,
    /// on a chip whose TIMESTAMP counter read 0 at `chip_zero_us`. Sample k's
    /// value is `first_value + k`.
    fn imu_stream_at(b: &mut Builder, imu: u8, seconds: i64, start_us: i64, chip_zero_us: i64, first_value: i16) {
        imu_stream_written(b, imu, seconds, start_us, chip_zero_us, first_value, |c| c);
    }

    /// [`imu_stream_at`], each block's `read_chip_ts` passed through
    /// `written` (how a firmware wrote it).
    fn imu_stream_written(b: &mut Builder, imu: u8, seconds: i64, start_us: i64, chip_zero_us: i64, first_value: i16, written: fn(u32) -> u32) {
        let tick_us = 21.75_f64;
        let period_us = 1e6 / 480.0;
        let total = (seconds as f64 * 480.0) as usize;
        let chip = |t_us: f64| ((t_us - chip_zero_us as f64) / tick_us).round() as u32;
        let mut k = 0usize;
        let mut read_us = start_us + 20_000;
        while k < total {
            let (mut samples, mut ts) = (Vec::new(), Vec::new());
            while k < total && (start_us as f64 + k as f64 * period_us) < read_us as f64 {
                let t = start_us as f64 + k as f64 * period_us;
                if k % 32 == 0 {
                    ts.push((samples.len() as u16, 0, chip(t)));
                }
                samples.push([first_value + k as i16; 6]);
                k += 1;
            }
            b.imu_block(imu, 0, read_us + 300, written(chip(read_us as f64)), &samples, &[], &ts);
            read_us += 20_000;
        }
    }

    #[test]
    fn parse_idl1_beta16_shifted_read_chip_ts_is_rebuilt_and_chip_timed() {
        // Arrange — 3 s at 480 Hz, the chip counter ~2.8e8 ticks in (the
        // 2026-10-31 ride's), each block's read_chip_ts written as
        // 0.3.0-beta.16 did: shifted left a byte, top byte lost.
        let mut b = Builder::new();
        b.imu_axes(1, 480);
        imu_stream_written(&mut b, 1, 3, 0, -6_000_000_000, 0, |c| c << 8);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert — chip-timed at 480 Hz with nothing invented, as if the
        // reads had been written right.
        let ax = find(&r, "IMU1_AccelX");
        assert_eq!(ax.t_us.len(), 1440);
        assert!(ax.gaps.is_empty());
        assert!(!r.import_warnings.iter().any(|w| w.kind == ImportWarningKind::TimingFallback), "{:?}", r.import_warnings);
        let spacing = (ax.t_us[1439] - ax.t_us[0]) as f64 / 1439.0;
        assert!((spacing - 1e6 / 480.0).abs() < 0.5, "spacing {spacing}");
    }

    #[test]
    fn parse_idl1_imu_restart_mid_session_starts_a_new_clock_segment() {
        // Arrange — 3 s, then the firmware re-inits IMU1 (its TIMESTAMP
        // counter starts over at 4 s), then 3 s more.
        let mut b = Builder::new();
        b.imu_axes(1, 480);
        imu_stream_at(&mut b, 1, 3, 0, 0, 0);
        b.logger_event(4, 1, 1, 3_100_000);
        b.logger_event(5, 1, 1, 3_990_000);
        imu_stream_at(&mut b, 1, 3, 4_000_000, 4_000_000, 2000);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert — both segments on the chip clock, in order, the second
        // starting ~4 s after the first, each spaced at 480 Hz.
        let ax = find(&r, "IMU1_AccelX");
        assert_eq!(ax.t_us.len(), 2880);
        assert!(!r.import_warnings.iter().any(|w| w.kind == ImportWarningKind::TimingFallback), "{:?}", r.import_warnings);
        assert!(ax.t_us.windows(2).all(|w| w[1] > w[0]));
        let restart = (ax.t_us[1440] - ax.t_us[0]) as f64;
        assert!((restart - 4e6).abs() < 100.0, "second segment at {restart} µs");
        for seg in [&ax.t_us[..1440], &ax.t_us[1440..]] {
            let spacing = (seg[1439] - seg[0]) as f64 / 1439.0;
            assert!((spacing - 1e6 / 480.0).abs() < 0.5, "spacing {spacing}");
        }
        assert!((ax.nominal_rate_hz - 480.0).abs() < 0.5, "rate {}", ax.nominal_rate_hz);
        let v = ax.materialize();
        assert!((v[1440] - 2000.0 * 0.000488).abs() < 1e-6);
        assert!(r.import_warnings.iter().any(|w| w.message.contains("IMU recovered")));
    }

    #[test]
    fn parse_idl1_lost_samples_are_counted_filled_and_listed_as_a_gap() {
        // Arrange — 5 samples lost from true index 1000.
        let mut b = Builder::new();
        b.imu_axes(0, 480);
        imu_stream(&mut b, 0, 5, Some((1000, 5)));
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert — back to 2400 slots, one 5-slot gap, linear fill.
        let ax = find(&r, "IMU0_AccelX");
        assert_eq!(ax.t_us.len(), 2400);
        assert_eq!(ax.gaps.len(), 1);
        assert_eq!(ax.gaps[0].len, 5);
        let values = ax.materialize();
        assert!((values[1004] - 1004.0 * 0.000488).abs() < 0.01, "filled value {}", values[1004]);
    }

    #[test]
    fn parse_idl1_gps_hdop_and_generic_channel_keep_their_own_times() {
        // Arrange
        let mut b = Builder::new();
        b.session_start_ms = 0;
        b.channel(32, 2, 0, 0, 0, 0, 0, 1.0, "WheelFront", "pulse");
        b.gps_fix(1_760_000_123_000, 1_000_000, 85);
        b.channel_sample_u32(32, 1_500_000, 7);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert
        let hdop = find(&r, "GPS_Hdop").materialize();
        assert_eq!(hdop.len(), 1);
        assert!((hdop[0] - 0.85).abs() < 1e-12);
        assert_eq!(find(&r, "WheelFront").t_us, vec![500_000]);
        assert_eq!(r.session.timestamp_source, TimestampSource::GpsBackfill);
        assert_eq!(r.session.timestamp_utc_ms, 1_760_000_123_000);
    }

    #[test]
    fn read_idl1_gps_config_reads_the_record_after_the_header() {
        // Arrange
        let mut b = Builder::new();
        b.gps_config(1, 8);
        b.logger_event(2, 0, 3, 4_000_000);
        b.end();

        // Act
        let cfg = read_idl1_gps_config(&b.build()).expect("GPS_CONFIG present");

        // Assert
        assert!(cfg.setup_ok());
        assert_eq!((cfg.requested_rate_hz, cfg.applied_rate_hz), (8, 8));
        assert_eq!(cfg.signals_text(), "GPS+Galileo+BeiDou, QZSS, SBAS");
        assert_eq!(cfg.dynamic_model_name(), "automotive");
        assert_eq!((cfg.min_elevation_deg, cfg.min_cno_dbhz, cfg.min_satellites, cfg.max_satellites), (5, 6, 3, 32));
        assert_eq!((cfg.pdop_mask, cfg.position_accuracy_mask_m), (25.0, 100));
        assert!(cfg.high_precision_nmea);
        assert_eq!(cfg.module_version, "ROM SPG 5.10");
        assert!(cfg.to_string().starts_with("8 Hz (module accepted 8 Hz), GPS+Galileo+BeiDou, QZSS, SBAS, automotive"));
    }

    #[test]
    fn read_idl1_gps_config_a_file_without_the_record_is_none() {
        let mut b = Builder::new();
        b.logger_event(2, 0, 3, 4_000_000);
        b.end();
        assert_eq!(read_idl1_gps_config(&b.build()), None);
        assert_eq!(read_idl1_gps_config(b"IDL0 not an idl1 file"), None);
    }

    #[test]
    fn parse_idl1_a_failed_gps_setup_becomes_an_import_warning_and_a_good_one_does_not() {
        // Arrange
        let mut failed = Builder::new();
        failed.gps_config(4, 1);
        failed.end();
        let mut ok = Builder::new();
        ok.gps_config(1, 8);
        ok.end();

        // Act
        let failed = parse_idl1(&failed.build()).unwrap();
        let ok = parse_idl1(&ok.build()).unwrap();

        // Assert
        assert!(failed.import_warnings.iter().any(|w| w.kind == ImportWarningKind::LoggerDiagnostic && w.message.contains("GPS setup failed (NAK)") && w.message.contains("at 1 Hz, not the 8 Hz")));
        assert!(!ok.import_warnings.iter().any(|w| w.message.contains("GPS")));
    }

    #[test]
    fn parse_idl1_logger_event_becomes_an_import_warning() {
        // Arrange
        let mut b = Builder::new();
        b.logger_event(2, 0, 3, 4_000_000);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert
        assert!(r.import_warnings.iter().any(|w| w.kind == ImportWarningKind::LoggerDiagnostic && w.message.contains("writer buffer full") && w.message.contains("×3")));
    }

    #[test]
    fn parse_idl1_damaged_header_is_refused_and_a_newer_version_asks_for_an_update() {
        // Arrange
        let mut b = Builder::new();
        b.imu_axes(0, 480).end();
        let mut damaged = b.build();
        damaged[30] ^= 0xFF;
        let mut newer = b.build();
        newer[4] = 2;

        // Act
        let e1 = parse_idl1(&damaged).unwrap_err();
        let e2 = parse_idl1(&newer).unwrap_err();

        // Assert
        assert!(matches!(e1, ParseError::TruncatedRecord(ref m) if m.contains("checksum")));
        assert!(matches!(e2, ParseError::UnsupportedSchemaVersion(_)));
    }

    #[test]
    fn parse_idl1_no_timestamp_words_falls_back_to_read_instants_with_a_warning() {
        // Arrange — one block, no timestamp words.
        let mut b = Builder::new();
        b.imu_axes(1, 480);
        b.imu_block(1, 0, 20_000, 900, &[[1; 6], [2; 6], [3; 6]], &[], &[]);
        b.end();

        // Act
        let r = parse_idl1(&b.build()).unwrap();

        // Assert
        let ax = find(&r, "IMU1_AccelX");
        assert_eq!(ax.t_us.len(), 3);
        assert!(r.import_warnings.iter().any(|w| w.kind == ImportWarningKind::TimingFallback));
    }

    #[test]
    fn parse_idl1_cut_off_mid_record_keeps_what_came_before() {
        // Arrange
        let mut b = Builder::new();
        b.imu_axes(0, 480);
        imu_stream(&mut b, 0, 2, None);
        let full = b.build();
        let cut = &full[..full.len() - 20];

        // Act
        let r = parse_idl1(cut).unwrap();

        // Assert
        assert!(r.truncation_warning.is_some());
        assert!(find(&r, "IMU0_AccelX").t_us.len() > 900);
    }

    #[test]
    fn fill_lost_interpolates_between_neighbours_and_holds_at_edges() {
        // Arrange
        let values = [0i16, 40];
        let source = [Some(0), None, None, None, Some(1), None];

        // Act
        let out = fill_lost(&values, &source);

        // Assert
        assert_eq!(out, vec![0, 10, 20, 30, 40, 40]);
    }
}
