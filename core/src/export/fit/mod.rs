//! FIT (Garmin) activity export. Reads GPS + heart rate from a `SessionHandle`,
//! maps them to FIT messages, and writes a complete `.fit` file. The low-level
//! byte framing lives in `encoder`.

mod encoder;

use std::fmt;
use std::io;

use encoder::{BaseType, FieldDef, FitWriter, FIT_EPOCH_OFFSET_SECS};

use crate::session::handle::SessionHandle;

/// Strava / Garmin sport classification for the activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSport {
    Generic,
    Running,
    Cycling,
    Motorcycling,
}

impl FitSport {
    /// FIT `sport` enum value (confirmed against the FIT profile).
    fn enum_value(self) -> u8 {
        match self {
            FitSport::Generic => 0,
            FitSport::Running => 1,
            FitSport::Cycling => 2,
            FitSport::Motorcycling => 22,
        }
    }
}

/// One lap's timing for a FIT `lap` message. All fields are milliseconds.
/// `elapsed_ms` is the split duration Strava displays — IDL0's effective lap
/// time (neutral zones removed).
#[derive(Debug, Clone, PartialEq)]
pub struct FitLap {
    /// Lap start, unix epoch milliseconds (wall clock).
    pub start_ms: i64,
    /// Lap end, unix epoch milliseconds (wall clock).
    pub end_ms: i64,
    /// Effective lap duration in milliseconds (the displayed split time).
    pub elapsed_ms: i64,
}

/// Controls FIT export. `laps` is `None` for a single whole-ride lap, or a
/// pre-detected lap list (the CLI runs detection from a track artifact; the app
/// passes its cached laps).
pub struct FitOptions {
    pub sport: FitSport,
    pub laps: Option<Vec<FitLap>>,
}

/// Errors raised while writing a FIT file.
#[derive(Debug)]
pub enum FitExportError {
    /// The session has no usable GPS fixes — a position-less activity is not
    /// meaningful for upload.
    NoGpsData,
    /// The output sink failed.
    Io(io::Error),
}

impl fmt::Display for FitExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitExportError::NoGpsData => {
                write!(f, "session has no GPS data; cannot build a FIT activity")
            }
            FitExportError::Io(e) => write!(f, "write failed: {e}"),
        }
    }
}

impl std::error::Error for FitExportError {}

impl From<io::Error> for FitExportError {
    fn from(e: io::Error) -> Self {
        FitExportError::Io(e)
    }
}

// FIT global message numbers.
const MSG_FILE_ID: u16 = 0;
const MSG_SESSION: u16 = 18;
const MSG_LAP: u16 = 19;
const MSG_RECORD: u16 = 20;
const MSG_DEVICE_INFO: u16 = 23;
const MSG_ACTIVITY: u16 = 34;

// uint8 / uint16 "invalid" sentinels per the FIT base-type table (used for
// optional heart-rate / speed / altitude fields with no value).
const U8_INVALID: u8 = 0xFF;
const U16_INVALID: u16 = 0xFFFF;

/// Convert a unix-epoch millisecond timestamp to a FIT timestamp (seconds since
/// 1989-12-31). Values before the FIT epoch clamp to 0.
fn fit_timestamp(epoch_ms: i64) -> u32 {
    let secs = epoch_ms / 1000 - FIT_EPOCH_OFFSET_SECS;
    secs.max(0) as u32
}

/// Convert a raw `GPS_Latitude`/`GPS_Longitude` sample (degrees × 1e7) to FIT
/// semicircles: `degrees × 2^31 / 180`.
fn to_semicircles(raw_deg_e7: f64) -> i32 {
    let degrees = raw_deg_e7 / 1e7;
    (degrees * (2f64.powi(31) / 180.0)).round() as i32
}

/// Convert a raw `GPS_Altitude` sample (metres × 10) to the FIT record
/// altitude field: `(metres + 500) × 5`, clamped to the uint16 range.
fn altitude_stored(raw_m_x10: f64) -> u16 {
    let metres = raw_m_x10 / 10.0;
    let v = ((metres + 500.0) * 5.0).round();
    v.clamp(0.0, u16::MAX as f64) as u16
}

/// Convert a `GPS_SpeedKmh` sample (physical km/h — the channel is engine-scaled
/// by 0.01, §5.7) to the FIT speed field: metres-per-second × 1000, clamped to
/// the uint16 range.
fn speed_stored(kmh: f64) -> u16 {
    let mps = kmh / 3.6;
    (mps * 1000.0).round().clamp(0.0, u16::MAX as f64) as u16
}

/// Great-circle distance in metres between two raw (deg × 1e7) coordinates.
fn haversine_m(lat1_e7: f64, lon1_e7: f64, lat2_e7: f64, lon2_e7: f64) -> f64 {
    const R: f64 = 6_371_000.0; // Earth radius, metres
    let to_rad = |deg_e7: f64| (deg_e7 / 1e7).to_radians();
    let (la1, lo1, la2, lo2) =
        (to_rad(lat1_e7), to_rad(lon1_e7), to_rad(lat2_e7), to_rad(lon2_e7));
    let dlat = la2 - la1;
    let dlon = lo2 - lo1;
    let a = (dlat / 2.0).sin().powi(2) + la1.cos() * la2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().atan2((1.0 - a).sqrt()) * R
}

/// Store an optional m/s speed in the FIT uint16 speed field (× 1000), or the
/// invalid sentinel when absent.
fn speed_mps_stored(mps: Option<f64>) -> u16 {
    match mps {
        Some(v) => (v * 1000.0).round().clamp(0.0, u16::MAX as f64) as u16,
        None => U16_INVALID,
    }
}

/// A single GPS fix prepared for FIT export. Coordinates stay at the raw
/// channel scale (deg × 1e7); altitude is the raw channel sample (m × 10).
/// `speed_kmh` is the engine-scaled physical speed in km/h (§5.7). Each optional
/// field is `None` when its source channel is absent.
struct Fix {
    epoch_ms: i64,
    lat_e7: f64,
    lon_e7: f64,
    alt_m_x10: Option<f64>,
    speed_kmh: Option<f64>,
}

/// Heart-rate beats on the session wall clock, ascending. `at` does a
/// carry-forward lookup: the most recent beat at-or-before a timestamp.
struct HrSeries {
    wall_ms: Vec<i64>,
    bpm: Vec<u8>,
}

impl HrSeries {
    /// The bpm of the last beat at or before `epoch_ms`, or `None` before the
    /// first beat / when empty.
    fn at(&self, epoch_ms: i64) -> Option<u8> {
        // partition_point returns the count of beats with wall_ms <= epoch_ms.
        let idx = self.wall_ms.partition_point(|&t| t <= epoch_ms);
        if idx == 0 {
            None
        } else {
            Some(self.bpm[idx - 1])
        }
    }
}

/// Build the GPS fix list from the handle's GPS channels: zip lat/lon/epoch to
/// the shortest, drop `(0,0)` fix-not-acquired sentinels, attach altitude/speed
/// where present. Empty when any of lat/lon/epoch is missing.
fn collect_fixes(handle: &SessionHandle) -> Vec<Fix> {
    let lat = handle.channel_samples("GPS_Latitude");
    let lon = handle.channel_samples("GPS_Longitude");
    let epoch = handle.channel_samples("GPS_EpochMs");
    if lat.is_empty() || lon.is_empty() || epoch.is_empty() {
        return Vec::new();
    }
    let alt = handle.channel_samples("GPS_Altitude");
    let speed = handle.channel_samples("GPS_SpeedKmh");
    let has_alt = !alt.is_empty();
    let has_speed = !speed.is_empty();

    let n = lat.len().min(lon.len()).min(epoch.len());
    let mut fixes = Vec::with_capacity(n);
    for i in 0..n {
        if lat[i] == 0.0 && lon[i] == 0.0 {
            continue; // fix-not-acquired sentinel
        }
        fixes.push(Fix {
            epoch_ms: epoch[i] as i64,
            lat_e7: lat[i],
            lon_e7: lon[i],
            alt_m_x10: if has_alt { alt.get(i).copied() } else { None },
            speed_kmh: if has_speed { speed.get(i).copied() } else { None },
        });
    }
    fixes
}

/// Build the heart-rate series on the session wall clock from `HR_BPM`, mapping
/// each sample's session-relative time to wall clock via
/// `timestamp_utc_ms + sample_time_secs × 1000`.
///
/// Handles both shapes the channel can take: **event-driven** (per-sample times
/// from `sample_times_secs`) and **fixed-rate** (the firmware registers
/// `HR_BPM` at 1 Hz, so there are no per-sample times — derive them from the
/// channel rate). Empty only when the channel is absent. A fixed-rate channel
/// with no usable rate falls back to 1 Hz so the HR still lands on the timeline.
fn collect_hr(handle: &SessionHandle, session_start_ms: i64) -> HrSeries {
    let bpm = handle.channel_samples("HR_BPM");
    if bpm.is_empty() {
        return HrSeries { wall_ms: Vec::new(), bpm: Vec::new() };
    }
    let times_secs: Vec<f64> = match handle.channel_sample_times("HR_BPM") {
        Some(t) => t,
        None => {
            let rate = handle
                .channels()
                .into_iter()
                .find(|c| c.channel_id == "HR_BPM")
                .map(|c| c.sample_rate_hz)
                .filter(|r| *r > 0.0)
                .unwrap_or(1.0);
            (0..bpm.len()).map(|i| i as f64 / rate).collect()
        }
    };
    let n = bpm.len().min(times_secs.len());
    let mut wall_ms = Vec::with_capacity(n);
    let mut out_bpm = Vec::with_capacity(n);
    for i in 0..n {
        wall_ms.push(session_start_ms + (times_secs[i] * 1000.0).round() as i64);
        out_bpm.push(bpm[i].round().clamp(0.0, 254.0) as u8);
    }
    HrSeries { wall_ms, bpm: out_bpm }
}

/// A lap's timing for the FIT `lap` message.
struct LapSpec {
    start_ms: i64,
    end_ms: i64,
    elapsed_s: f64,
}

/// Write a complete FIT activity for `handle` to `w`.
///
/// Emits file_id, device_info (fixed "IDL0" branding), one record per GPS fix
/// (with altitude/speed/heart-rate fields included only when that data exists),
/// lap message(s), a session, and an activity. Heart rate is carry-forward
/// merged onto the record stream.
///
/// Returns [`FitExportError::NoGpsData`] when the session has no GPS fixes.
pub fn write_fit(
    handle: &SessionHandle,
    options: &FitOptions,
    w: &mut impl io::Write,
) -> Result<(), FitExportError> {
    let fixes = collect_fixes(handle);
    if fixes.is_empty() {
        return Err(FitExportError::NoGpsData);
    }
    let session_start_ms = handle.metadata().timestamp_utc_ms;
    let hr = collect_hr(handle, session_start_ms);

    let has_alt = fixes.iter().any(|f| f.alt_m_x10.is_some());
    let has_speed = fixes.iter().any(|f| f.speed_kmh.is_some());
    let has_hr = !hr.wall_ms.is_empty();

    let mut fw = FitWriter::new();

    let first_ts = fit_timestamp(fixes.first().unwrap().epoch_ms);
    let last_ts = fit_timestamp(fixes.last().unwrap().epoch_ms);

    write_file_id(&mut fw, first_ts);
    write_device_info(&mut fw);

    // ---- record definition (local type 0), chosen once ----
    let mut record_fields = vec![
        FieldDef { num: 253, base: BaseType::Uint32 }, // timestamp
        FieldDef { num: 0, base: BaseType::Sint32 },   // position_lat
        FieldDef { num: 1, base: BaseType::Sint32 },   // position_long
        FieldDef { num: 5, base: BaseType::Uint32 },   // distance
    ];
    if has_alt {
        record_fields.push(FieldDef { num: 2, base: BaseType::Uint16 }); // altitude
    }
    if has_speed {
        record_fields.push(FieldDef { num: 6, base: BaseType::Uint16 }); // speed
    }
    if has_hr {
        record_fields.push(FieldDef { num: 3, base: BaseType::Uint8 }); // heart_rate
    }
    fw.definition(0, MSG_RECORD, &record_fields);

    // ---- record data ----
    let mut cumulative_m = 0.0f64;
    let mut prev: Option<&Fix> = None;
    let mut hr_values: Vec<u8> = Vec::new();
    for fix in &fixes {
        if let Some(p) = prev {
            cumulative_m += haversine_m(p.lat_e7, p.lon_e7, fix.lat_e7, fix.lon_e7);
        }
        fw.data_header(0);
        fw.push_u32(fit_timestamp(fix.epoch_ms));
        fw.push_i32(to_semicircles(fix.lat_e7));
        fw.push_i32(to_semicircles(fix.lon_e7));
        fw.push_u32((cumulative_m * 100.0).round().clamp(0.0, u32::MAX as f64) as u32);
        if has_alt {
            fw.push_u16(fix.alt_m_x10.map(altitude_stored).unwrap_or(U16_INVALID));
        }
        if has_speed {
            fw.push_u16(fix.speed_kmh.map(speed_stored).unwrap_or(U16_INVALID));
        }
        if has_hr {
            let bpm = hr.at(fix.epoch_ms);
            if let Some(b) = bpm {
                hr_values.push(b);
            }
            fw.push_u8(bpm.unwrap_or(U8_INVALID));
        }
        prev = Some(fix);
    }

    let total_distance_m = cumulative_m;
    let total_elapsed_s =
        (fixes.last().unwrap().epoch_ms - fixes.first().unwrap().epoch_ms).max(0) as f64 / 1000.0;

    // average / max speed in m/s over fixes that reported speed (km/h ÷ 3.6).
    let speeds_mps: Vec<f64> = fixes
        .iter()
        .filter_map(|f| f.speed_kmh)
        .map(|kmh| kmh / 3.6)
        .collect();
    let avg_speed_mps = if speeds_mps.is_empty() {
        None
    } else {
        Some(speeds_mps.iter().sum::<f64>() / speeds_mps.len() as f64)
    };
    let max_speed_mps = speeds_mps.iter().cloned().fold(None, |m: Option<f64>, v| {
        Some(m.map_or(v, |cur| cur.max(v)))
    });

    let avg_hr = if hr_values.is_empty() {
        None
    } else {
        Some((hr_values.iter().map(|&b| b as u32).sum::<u32>() / hr_values.len() as u32) as u8)
    };
    let max_hr = hr_values.iter().copied().max();

    // ---- laps ----
    let lap_specs: Vec<LapSpec> = match &options.laps {
        Some(laps) if !laps.is_empty() => laps
            .iter()
            .map(|l| LapSpec {
                start_ms: l.start_ms,
                end_ms: l.end_ms,
                elapsed_s: l.elapsed_ms.max(0) as f64 / 1000.0,
            })
            .collect(),
        _ => vec![LapSpec {
            start_ms: fixes.first().unwrap().epoch_ms,
            end_ms: fixes.last().unwrap().epoch_ms,
            elapsed_s: total_elapsed_s,
        }],
    };
    let num_laps = lap_specs.len() as u16;
    for (i, spec) in lap_specs.iter().enumerate() {
        write_lap(&mut fw, i as u16, spec);
    }

    write_session(
        &mut fw,
        first_ts,
        last_ts,
        total_elapsed_s,
        total_distance_m,
        options.sport,
        num_laps,
        avg_speed_mps,
        max_speed_mps,
        avg_hr,
        max_hr,
    );
    write_activity(&mut fw, last_ts, total_elapsed_s);

    let bytes = fw.finish();
    w.write_all(&bytes)?;
    Ok(())
}

fn write_file_id(fw: &mut FitWriter, time_created: u32) {
    fw.definition(
        1,
        MSG_FILE_ID,
        &[
            FieldDef { num: 0, base: BaseType::Enum },    // type
            FieldDef { num: 1, base: BaseType::Uint16 },  // manufacturer
            FieldDef { num: 2, base: BaseType::Uint16 },  // product
            FieldDef { num: 3, base: BaseType::Uint32z }, // serial_number
            FieldDef { num: 4, base: BaseType::Uint32 },  // time_created
        ],
    );
    fw.data_header(1);
    fw.push_enum(4); // file type: activity
    fw.push_u16(255); // manufacturer: development
    fw.push_u16(0); // product
    fw.push_u32(if time_created == 0 { 1 } else { time_created }); // serial (uint32z: 0 = invalid)
    fw.push_u32(time_created);
}

fn write_device_info(fw: &mut FitWriter) {
    const NAME_LEN: u8 = 5; // "IDL0" + null
    fw.definition(
        2,
        MSG_DEVICE_INFO,
        &[
            FieldDef { num: 0, base: BaseType::Uint8 },             // device_index
            FieldDef { num: 2, base: BaseType::Uint16 },            // manufacturer
            FieldDef { num: 27, base: BaseType::String(NAME_LEN) }, // product_name
        ],
    );
    fw.data_header(2);
    fw.push_u8(0); // device_index: creator
    fw.push_u16(255); // manufacturer: development
    fw.push_string("IDL0", NAME_LEN);
}

fn write_lap(fw: &mut FitWriter, index: u16, spec: &LapSpec) {
    fw.definition(
        3,
        MSG_LAP,
        &[
            FieldDef { num: 254, base: BaseType::Uint16 }, // message_index
            FieldDef { num: 253, base: BaseType::Uint32 }, // timestamp (end)
            FieldDef { num: 2, base: BaseType::Uint32 },   // start_time
            FieldDef { num: 7, base: BaseType::Uint32 },   // total_elapsed_time
            FieldDef { num: 8, base: BaseType::Uint32 },   // total_timer_time
        ],
    );
    fw.data_header(3);
    fw.push_u16(index);
    fw.push_u32(fit_timestamp(spec.end_ms));
    fw.push_u32(fit_timestamp(spec.start_ms));
    let elapsed = (spec.elapsed_s * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
    fw.push_u32(elapsed);
    fw.push_u32(elapsed);
}

#[allow(clippy::too_many_arguments)]
fn write_session(
    fw: &mut FitWriter,
    first_ts: u32,
    last_ts: u32,
    total_elapsed_s: f64,
    total_distance_m: f64,
    sport: FitSport,
    num_laps: u16,
    avg_speed_mps: Option<f64>,
    max_speed_mps: Option<f64>,
    avg_hr: Option<u8>,
    max_hr: Option<u8>,
) {
    fw.definition(
        4,
        MSG_SESSION,
        &[
            FieldDef { num: 254, base: BaseType::Uint16 }, // message_index
            FieldDef { num: 253, base: BaseType::Uint32 }, // timestamp
            FieldDef { num: 2, base: BaseType::Uint32 },   // start_time
            FieldDef { num: 7, base: BaseType::Uint32 },   // total_elapsed_time
            FieldDef { num: 8, base: BaseType::Uint32 },   // total_timer_time
            FieldDef { num: 9, base: BaseType::Uint32 },   // total_distance
            FieldDef { num: 5, base: BaseType::Enum },     // sport
            FieldDef { num: 6, base: BaseType::Enum },     // sub_sport
            FieldDef { num: 25, base: BaseType::Uint16 },  // first_lap_index
            FieldDef { num: 26, base: BaseType::Uint16 },  // num_laps
            FieldDef { num: 14, base: BaseType::Uint16 },  // avg_speed
            FieldDef { num: 15, base: BaseType::Uint16 },  // max_speed
            FieldDef { num: 16, base: BaseType::Uint8 },   // avg_heart_rate
            FieldDef { num: 17, base: BaseType::Uint8 },   // max_heart_rate
        ],
    );
    fw.data_header(4);
    fw.push_u16(0); // message_index
    fw.push_u32(last_ts);
    fw.push_u32(first_ts);
    let elapsed = (total_elapsed_s * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
    fw.push_u32(elapsed);
    fw.push_u32(elapsed);
    fw.push_u32((total_distance_m * 100.0).round().clamp(0.0, u32::MAX as f64) as u32);
    fw.push_enum(sport.enum_value());
    fw.push_enum(0); // sub_sport: generic
    fw.push_u16(0); // first_lap_index
    fw.push_u16(num_laps);
    fw.push_u16(speed_mps_stored(avg_speed_mps));
    fw.push_u16(speed_mps_stored(max_speed_mps));
    fw.push_u8(avg_hr.unwrap_or(U8_INVALID));
    fw.push_u8(max_hr.unwrap_or(U8_INVALID));
}

fn write_activity(fw: &mut FitWriter, last_ts: u32, total_timer_s: f64) {
    fw.definition(
        5,
        MSG_ACTIVITY,
        &[
            FieldDef { num: 253, base: BaseType::Uint32 }, // timestamp
            FieldDef { num: 0, base: BaseType::Uint32 },   // total_timer_time
            FieldDef { num: 1, base: BaseType::Uint16 },   // num_sessions
            FieldDef { num: 2, base: BaseType::Enum },     // type
            FieldDef { num: 3, base: BaseType::Enum },     // event
            FieldDef { num: 4, base: BaseType::Enum },     // event_type
        ],
    );
    fw.data_header(5);
    fw.push_u32(last_ts);
    fw.push_u32((total_timer_s * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32);
    fw.push_u16(1); // num_sessions
    fw.push_enum(0); // type: manual
    fw.push_enum(26); // event: activity
    fw.push_enum(1); // event_type: stop
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 1_700_000_000_000, // arbitrary recent unix ms
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn fixed(id: &str, samples: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 1.0, samples, sample_times_secs: None }
    }

    fn event(id: &str, samples: Vec<f64>, times: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 0.0, samples, sample_times_secs: Some(times) }
    }

    #[test]
    fn fit_timestamp_subtracts_fit_epoch() {
        // Arrange — exactly the FIT epoch in unix ms → 0 FIT seconds.
        assert_eq!(fit_timestamp(FIT_EPOCH_OFFSET_SECS * 1000), 0);
        // One second later → 1.
        assert_eq!(fit_timestamp((FIT_EPOCH_OFFSET_SECS + 1) * 1000), 1);
    }

    #[test]
    fn semicircles_round_trip_known_value() {
        // Arrange — 45.0 degrees stored as 45.0e7.
        // 45 deg × 2^31/180 = 536870912.
        assert_eq!(to_semicircles(45.0 * 1e7), 536_870_912);
        assert_eq!(to_semicircles(0.0), 0);
    }

    #[test]
    fn altitude_applies_fit_scale_and_offset() {
        // Arrange — 100.0 m stored as 1000 (m × 10). (100 + 500) × 5 = 3000.
        assert_eq!(altitude_stored(1000.0), 3000);
    }

    #[test]
    fn speed_converts_kmh_to_mps_times_1000() {
        // Arrange — 36 km/h (engine-scaled physical) = 10 m/s → 10000.
        assert_eq!(speed_stored(36.0), 10_000);
    }

    #[test]
    fn haversine_one_degree_latitude_is_about_111km() {
        // Act
        let d = haversine_m(0.0, 0.0, 1.0 * 1e7, 0.0);

        // Assert — ~111.19 km within 1 km.
        assert!((d - 111_195.0).abs() < 1000.0, "got {d}");
    }

    #[test]
    fn collect_fixes_drops_zero_sentinels_and_zips_to_shortest() {
        // Arrange — 3 lat/lon (index 0 = (0,0) sentinel), 2 epochs.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![0.0, 45.0e7, 46.0e7]),
            fixed("GPS_Longitude", vec![0.0, -1.0e7, -2.0e7]),
            fixed("GPS_EpochMs", vec![1000.0, 2000.0]),
        ]);

        // Act
        let fixes = collect_fixes(&h);

        // Assert — sentinel dropped; zipped to the 2 epochs; index 1 survives.
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].epoch_ms, 2000);
        assert_eq!(fixes[0].lat_e7, 45.0e7);
        assert!(fixes[0].alt_m_x10.is_none());
        assert!(fixes[0].speed_kmh.is_none());
    }

    #[test]
    fn collect_fixes_carries_altitude_and_speed_when_present() {
        // Arrange
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7]),
            fixed("GPS_Longitude", vec![-1.0e7]),
            fixed("GPS_EpochMs", vec![5000.0]),
            fixed("GPS_Altitude", vec![1000.0]),
            fixed("GPS_SpeedKmh", vec![36.0]),
        ]);

        // Act
        let fixes = collect_fixes(&h);

        // Assert
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].alt_m_x10, Some(1000.0));
        assert_eq!(fixes[0].speed_kmh, Some(36.0));
    }

    #[test]
    fn hr_at_carries_forward_last_beat_at_or_before() {
        // Arrange — beats at session-relative 1s and 3s; session start 1000 ms.
        // Wall clocks: 2000 ms and 4000 ms.
        let series = HrSeries {
            wall_ms: vec![2000, 4000],
            bpm: vec![140, 150],
        };

        // Act + Assert
        assert_eq!(series.at(1500), None); // before first beat
        assert_eq!(series.at(2000), Some(140)); // exactly first beat
        assert_eq!(series.at(3999), Some(140)); // carry-forward
        assert_eq!(series.at(4000), Some(150)); // second beat
        assert_eq!(series.at(9999), Some(150)); // carry-forward past last
    }

    #[test]
    fn write_fit_errors_when_no_gps() {
        // Arrange — a session with no GPS channels.
        let h = handle_with(vec![fixed("IMU0_AccelX", vec![1.0, 2.0])]);
        let mut buf: Vec<u8> = Vec::new();

        // Act
        let r = write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: None }, &mut buf);

        // Assert
        assert!(matches!(r, Err(FitExportError::NoGpsData)));
    }

    #[test]
    fn write_fit_produces_a_valid_fit_file() {
        // Arrange — two GPS fixes + two HR beats.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7, 45.001e7]),
            fixed("GPS_Longitude", vec![-1.0e7, -1.001e7]),
            fixed("GPS_EpochMs", vec![1_700_000_001_000.0, 1_700_000_002_000.0]),
            fixed("GPS_Altitude", vec![1000.0, 1010.0]),
            fixed("GPS_SpeedKmh", vec![36.0, 36.0]),
            event("HR_BPM", vec![140.0, 150.0], vec![1.0, 2.0]),
        ]);
        let mut buf: Vec<u8> = Vec::new();

        // Act
        write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: None }, &mut buf).unwrap();

        // Assert — header + trailing CRC frame is self-consistent.
        assert_eq!(&buf[8..12], b".FIT");
        let n = buf.len();
        let file_crc = u16::from_le_bytes([buf[n - 2], buf[n - 1]]);
        assert_eq!(file_crc, encoder::crc16(&buf[..n - 2]));
        // Data size in the header matches the body length.
        let data_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        assert_eq!(data_size, n - 14 - 2);
    }

    #[test]
    fn fit_output_parses_back_with_fitparser() {
        // Arrange — two fixes, two HR beats, cycling.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7, 45.001e7]),
            fixed("GPS_Longitude", vec![-1.0e7, -1.001e7]),
            fixed("GPS_EpochMs", vec![1_700_000_001_000.0, 1_700_000_002_000.0]),
            fixed("GPS_Altitude", vec![1000.0, 1010.0]),
            fixed("GPS_SpeedKmh", vec![36.0, 36.0]),
            event("HR_BPM", vec![140.0, 150.0], vec![1.0, 2.0]),
        ]);
        let mut buf: Vec<u8> = Vec::new();
        write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: None }, &mut buf).unwrap();

        // Act — fitparser validates the CRC on parse and decodes every message.
        let records = fitparser::from_bytes(&buf).expect("valid FIT file");

        // Assert — exactly two `record` messages (one per GPS fix).
        use fitparser::profile::MesgNum;
        let record_count = records
            .iter()
            .filter(|d| d.kind() == MesgNum::Record)
            .count();
        assert_eq!(record_count, 2);

        // A record carries a heart_rate field with our first value.
        let first_record = records
            .iter()
            .find(|d| d.kind() == MesgNum::Record)
            .unwrap();
        let hr_field = first_record
            .fields()
            .iter()
            .find(|f| f.name() == "heart_rate");
        assert!(hr_field.is_some(), "heart_rate field present");
    }

    #[test]
    fn fit_output_includes_heart_rate_from_fixed_rate_channel() {
        // Arrange — HR_BPM as a FIXED-RATE 1 Hz channel (how the firmware
        // registers channel 22), with no per-sample event times. Regression:
        // HR was silently dropped from the FIT for this (real-world) shape.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7, 45.001e7]),
            fixed("GPS_Longitude", vec![-1.0e7, -1.001e7]),
            fixed("GPS_EpochMs", vec![1_700_000_001_000.0, 1_700_000_002_000.0]),
            fixed("HR_BPM", vec![140.0, 150.0]),
        ]);
        let mut buf: Vec<u8> = Vec::new();
        write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: None }, &mut buf).unwrap();

        // Act
        let records = fitparser::from_bytes(&buf).expect("valid FIT file");

        // Assert — at least one record carries a heart_rate field.
        use fitparser::profile::MesgNum;
        let any_hr = records
            .iter()
            .filter(|d| d.kind() == MesgNum::Record)
            .any(|d| d.fields().iter().any(|f| f.name() == "heart_rate"));
        assert!(any_hr, "heart_rate present for a fixed-rate HR_BPM channel");
    }

    #[test]
    fn fit_output_omits_heart_rate_when_absent() {
        // Arrange — GPS only, no HR.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7, 45.001e7]),
            fixed("GPS_Longitude", vec![-1.0e7, -1.001e7]),
            fixed("GPS_EpochMs", vec![1_700_000_001_000.0, 1_700_000_002_000.0]),
        ]);
        let mut buf: Vec<u8> = Vec::new();
        write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: None }, &mut buf).unwrap();

        // Act
        let records = fitparser::from_bytes(&buf).expect("valid FIT file");

        // Assert — no record carries a heart_rate field.
        use fitparser::profile::MesgNum;
        let any_hr = records
            .iter()
            .filter(|d| d.kind() == MesgNum::Record)
            .any(|d| d.fields().iter().any(|f| f.name() == "heart_rate"));
        assert!(!any_hr, "no heart_rate field when HR channel absent");
    }

    #[test]
    fn fit_output_writes_one_lap_message_per_fit_lap() {
        // Arrange — two GPS fixes spanning 4 s, and two explicit laps.
        let h = handle_with(vec![
            fixed("GPS_Latitude", vec![45.0e7, 45.001e7]),
            fixed("GPS_Longitude", vec![-1.0e7, -1.001e7]),
            fixed("GPS_EpochMs", vec![1_700_000_001_000.0, 1_700_000_005_000.0]),
        ]);
        let laps = vec![
            FitLap { start_ms: 1_700_000_001_000, end_ms: 1_700_000_003_000, elapsed_ms: 2_000 },
            FitLap { start_ms: 1_700_000_003_000, end_ms: 1_700_000_005_000, elapsed_ms: 2_000 },
        ];
        let mut buf: Vec<u8> = Vec::new();

        // Act
        write_fit(&h, &FitOptions { sport: FitSport::Cycling, laps: Some(laps) }, &mut buf).unwrap();

        // Assert — fitparser sees exactly two lap messages.
        use fitparser::profile::MesgNum;
        let records = fitparser::from_bytes(&buf).expect("valid FIT file");
        let lap_count = records.iter().filter(|d| d.kind() == MesgNum::Lap).count();
        assert_eq!(lap_count, 2);
    }
}
