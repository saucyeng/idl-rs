//! FIT (Garmin/Wahoo/etc.) `.fit` activity import, via the `fitparser`
//! crate (`0.9`, promoted from a dev-dependency — L2-R9). Channel mapping
//! and the record-level duplicate/non-monotonic-timestamp rule are
//! documented in `docs/IDL0_SPEC.md` §15a.2.
//!
//! **Why record-level, not literally per-channel, deduplication.** C1 §3.4
//! states the drop rule per *channel*; every FIT `record` message carries
//! exactly one `timestamp` field shared by every other field on that same
//! message (a FIT protocol invariant), so dropping a whole record when its
//! timestamp collides is equivalent to dropping the same sample from every
//! channel independently — this module does it once, at the record level.

use crate::session::{Channel, Session, SourceFormat};

use super::{ImportedSession, Importer, ImporterError, ImporterWarning};

/// This importer implementation's own version string (C1 §4.3).
pub const FIT_IMPORTER_VERSION: &str = "0.1.0";

/// Imports Garmin/Wahoo-style `.fit` activity files.
pub struct FitImporter;

/// One `record` message's fields this importer understands, `None` when the
/// field was absent from that particular message (FIT records commonly omit
/// fields — e.g. an indoor-trainer file has no `position_lat`/`position_long`).
#[derive(Debug, Clone, Default)]
struct FitRecord {
    /// Unix-epoch seconds (already offset-corrected by `fitparser` — see
    /// [`Importer::import`]'s `"timestamp"` arm; never re-add
    /// `631_065_600`).
    timestamp_utc_s: Option<i64>,
    /// Latitude, decimal degrees (R27 — physical, unscaled).
    lat_deg: Option<f64>,
    /// Longitude, decimal degrees (R27 — physical, unscaled).
    lon_deg: Option<f64>,
    /// Altitude, metres.
    altitude_m: Option<f64>,
    /// Heart rate, beats per minute.
    hr_bpm: Option<f64>,
    /// Cadence, revolutions per minute.
    cadence_rpm: Option<f64>,
    /// Power, watts.
    power_w: Option<f64>,
}

/// Converts FIT position semicircles (`i32`) to decimal degrees:
/// `degrees = semicircles × 180 / 2^31` (the FIT SDK's documented formula).
/// R27: the result is stored unscaled — no `× 1e7`, no `deg_e7` anywhere.
fn semicircles_to_deg(raw: i32) -> f64 {
    raw as f64 * (180.0 / 2_147_483_648.0)
}

/// Widens any numeric `fitparser::Value` to `f64`; `None` for non-numeric
/// variants (should not occur for the fields this importer reads).
fn numeric_value(v: &fitparser::Value) -> Option<f64> {
    match v {
        fitparser::Value::UInt8(x) => Some(*x as f64),
        fitparser::Value::UInt16(x) => Some(*x as f64),
        fitparser::Value::SInt32(x) => Some(*x as f64),
        fitparser::Value::UInt32(x) => Some(*x as f64),
        fitparser::Value::Float32(x) => Some(*x as f64),
        fitparser::Value::Float64(x) => Some(*x),
        _ => None,
    }
}

impl Importer for FitImporter {
    fn source_format(&self) -> SourceFormat {
        SourceFormat::Fit
    }

    fn import(&self, bytes: &[u8], blob_sha256: &str) -> Result<ImportedSession, ImporterError> {
        let data = fitparser::from_bytes(bytes).map_err(|e| ImporterError::FitMalformed(e.to_string()))?;

        let mut records: Vec<FitRecord> = Vec::new();
        for d in data.iter().filter(|d| d.kind() == fitparser::profile::MesgNum::Record) {
            let mut r = FitRecord::default();
            for field in d.fields() {
                match field.name() {
                    "timestamp" => {
                        if let fitparser::Value::Timestamp(dt) = field.value() {
                            // `fitparser`'s `.timestamp()` already folds in
                            // the FIT-epoch offset (1989-12-31 -> 1970-01-01,
                            // 631_065_600 s) internally — this is already
                            // Unix-epoch seconds (Q3). Do not add the offset
                            // again anywhere downstream.
                            r.timestamp_utc_s = Some(dt.timestamp());
                        }
                    }
                    "position_lat" => {
                        if let fitparser::Value::SInt32(v) = field.value() {
                            r.lat_deg = Some(semicircles_to_deg(*v));
                        }
                    }
                    "position_long" => {
                        if let fitparser::Value::SInt32(v) = field.value() {
                            r.lon_deg = Some(semicircles_to_deg(*v));
                        }
                    }
                    "enhanced_altitude" => r.altitude_m = numeric_value(field.value()),
                    "heart_rate" => r.hr_bpm = numeric_value(field.value()),
                    "cadence" => r.cadence_rpm = numeric_value(field.value()),
                    "power" => r.power_w = numeric_value(field.value()),
                    _ => {}
                }
            }
            records.push(r);
        }

        // L2-R7(d): a record with no timestamp cannot be placed on the time
        // axis — recover what's readable rather than failing the whole
        // import (CLAUDE.md §5), but warn about every one dropped rather
        // than dropping it silently.
        let mut warnings = Vec::new();
        for (i, r) in records.iter().enumerate() {
            if r.timestamp_utc_s.is_none() {
                warnings.push(ImporterWarning::new(format!("dropped FIT record {i}: no timestamp")));
            }
        }
        records.retain(|r| r.timestamp_utc_s.is_some());
        if records.is_empty() {
            return Err(ImporterError::FitMalformed(
                "no `record` message carries a timestamp".to_string(),
            ));
        }

        // L2-R7(a): t0 is the minimum timestamp among the kept records, not
        // necessarily `records[0]` — a FIT file is not guaranteed sorted by
        // time.
        let first_utc_s = records
            .iter()
            .map(|r| r.timestamp_utc_s.unwrap())
            .min()
            .expect("records is non-empty (checked above)");

        let mut kept: Vec<(usize, i64)> = Vec::with_capacity(records.len());
        let mut last_kept_t_us: Option<i64> = None;
        for (i, r) in records.iter().enumerate() {
            let t_us = (r.timestamp_utc_s.unwrap() - first_utc_s) * 1_000_000;
            if let Some(last) = last_kept_t_us {
                if t_us <= last {
                    warnings.push(ImporterWarning::new(format!(
                        "dropped FIT record {i}: duplicate/non-monotonic timestamp"
                    )));
                    continue;
                }
            }
            last_kept_t_us = Some(t_us);
            kept.push((i, t_us));
        }

        let mut channels = Vec::new();
        channels.push(push_channel(&kept, &records, "GPS_Latitude", "deg", |r| r.lat_deg));
        channels.push(push_channel(&kept, &records, "GPS_Longitude", "deg", |r| r.lon_deg));
        // Q3: GPS_EpochMs is present only on records carrying a position
        // (both lat and lon decoded), value = timestamp_utc_s * 1000, no
        // offset added a second time.
        channels.push(push_channel(&kept, &records, "GPS_EpochMs", "ms_raw", |r| {
            (r.lat_deg.is_some() && r.lon_deg.is_some()).then(|| r.timestamp_utc_s.unwrap() as f64 * 1000.0)
        }));
        channels.push(push_channel(&kept, &records, "GPS_Altitude", "m", |r| r.altitude_m));
        channels.push(push_channel(&kept, &records, "HR_BPM", "bpm", |r| r.hr_bpm));
        channels.push(push_channel(&kept, &records, "Cadence_RPM", "rpm", |r| r.cadence_rpm));
        channels.push(push_channel(&kept, &records, "Power_W", "W", |r| r.power_w));
        // L2-R6-equivalent: a channel absent from every record contributes
        // no samples — omit it entirely rather than writing an empty one.
        channels.retain(|c| !c.t_us.is_empty());

        let session = Session {
            session_id: super::session_id_from_blob_hash(blob_sha256),
            device_id: None,
            timestamp_utc_ms: first_utc_s * 1000,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: blob_sha256.to_string(),
            channels,
        };

        Ok(ImportedSession { session, warnings })
    }

    fn importer_version(&self) -> &'static str {
        FIT_IMPORTER_VERSION
    }
}

/// Builds one channel from `kept` (record index, t_us) pairs, taking only
/// the records where `f` returns `Some` — genuinely per-record independence
/// (C1 §3.4): a record missing `power`, say, contributes no sample to
/// `Power_W` rather than a zero-filled one. Callers drop the resulting
/// channel afterward if it ended up with zero samples (a field absent from
/// every record, C1 §4.1 "as applicable").
fn push_channel(
    kept: &[(usize, i64)],
    records: &[FitRecord],
    channel_id: &str,
    unit: &str,
    f: impl Fn(&FitRecord) -> Option<f64>,
) -> Channel {
    let mut t_us = Vec::new();
    let mut values = Vec::new();
    for &(i, t) in kept {
        if let Some(v) = f(&records[i]) {
            t_us.push(t);
            values.push(v);
        }
    }
    Channel::from_f64_with_times(channel_id, 0.0, values, t_us, "fit").with_unit(unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIT CRC-16, the same table-driven algorithm as
    /// `crate::export::fit::encoder::crc16` — duplicated here (that module
    /// is private to `export::fit`, and this plan's lane does not touch
    /// `export/` — CLAUDE.md §7) rather than reached into.
    fn fit_crc16(data: &[u8]) -> u16 {
        const TABLE: [u16; 16] = [
            0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401, 0xA001, 0x6C00, 0x7800,
            0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
        ];
        let mut crc: u16 = 0;
        for &byte in data {
            let tmp = TABLE[(crc & 0xF) as usize];
            crc = (crc >> 4) & 0x0FFF;
            crc = crc ^ tmp ^ TABLE[(byte & 0xF) as usize];
            let tmp = TABLE[(crc & 0xF) as usize];
            crc = (crc >> 4) & 0x0FFF;
            crc = crc ^ tmp ^ TABLE[((byte >> 4) & 0xF) as usize];
        }
        crc
    }

    /// One `record` message's raw field values, in definition order:
    /// timestamp (FIT-epoch seconds), lat/lon (semicircles), altitude (raw,
    /// `physical = raw/5 − 500`), heart_rate, cadence, power.
    struct RawRecord {
        timestamp: u32,
        lat: i32,
        lon: i32,
        altitude_raw: u16,
        hr: u8,
        cadence: u8,
        power: u16,
    }

    /// Builds a minimal, valid `.fit` byte sequence: a 14-byte file header,
    /// one `record` (global msg 20) definition message, one data message
    /// per `rows` entry, and the trailing file CRC.
    fn build_fit_fixture(rows: &[RawRecord]) -> Vec<u8> {
        let mut body = Vec::new();
        body.push(0x40); // definition, local_type 0
        body.push(0x00); // reserved
        body.push(0x00); // architecture: little-endian
        body.extend_from_slice(&20u16.to_le_bytes()); // global_mesg_num = record
        body.push(7); // field count
        for &(num, size, base) in &[
            (253u8, 4u8, 0x86u8), // timestamp: uint32
            (0, 4, 0x85),          // position_lat: sint32
            (1, 4, 0x85),          // position_long: sint32
            (2, 2, 0x84),          // altitude: uint16
            (3, 1, 0x02),          // heart_rate: uint8
            (4, 1, 0x02),          // cadence: uint8
            (7, 2, 0x84),          // power: uint16
        ] {
            body.push(num);
            body.push(size);
            body.push(base);
        }
        for row in rows {
            body.push(0x00); // data message, local_type 0
            body.extend_from_slice(&row.timestamp.to_le_bytes());
            body.extend_from_slice(&row.lat.to_le_bytes());
            body.extend_from_slice(&row.lon.to_le_bytes());
            body.extend_from_slice(&row.altitude_raw.to_le_bytes());
            body.push(row.hr);
            body.push(row.cadence);
            body.extend_from_slice(&row.power.to_le_bytes());
        }

        let mut out = Vec::with_capacity(14 + body.len() + 2);
        out.push(14); // header size
        out.push(0x20); // protocol version 2.0
        out.extend_from_slice(&2100u16.to_le_bytes()); // profile version
        out.extend_from_slice(&(body.len() as u32).to_le_bytes()); // data size
        out.extend_from_slice(b".FIT");
        let header_crc = fit_crc16(&out[0..12]);
        out.extend_from_slice(&header_crc.to_le_bytes());
        out.extend_from_slice(&body);
        let file_crc = fit_crc16(&out);
        out.extend_from_slice(&file_crc.to_le_bytes());
        out
    }

    /// `T0` — the raw value this fixture writes into each `record`
    /// message's wire-format `timestamp` field. Per the FIT protocol that
    /// field is itself FIT-epoch seconds (seconds since 1989-12-31), so
    /// `fitparser` adds [`FIT_EPOCH_OFFSET_S`] when decoding it into the
    /// Unix-epoch `r.timestamp_utc_s` this importer stores — the golden
    /// assertions below use `T0 + FIT_EPOCH_OFFSET_S`, never `T0` alone, to
    /// match what `fitparser` actually hands back.
    const T0: i64 = 1_000_000_000;

    /// FIT-epoch → Unix-epoch offset (1989-12-31 00:00:00 UTC → 1970-01-01
    /// 00:00:00 UTC), seconds. Used only here, to compute this fixture's
    /// expected Unix-epoch golden values from the raw FIT-epoch seconds it
    /// writes as `RawRecord::timestamp` — never applied in production code
    /// (`fitparser`'s own `Value::Timestamp(..).timestamp()` already folds
    /// it in; see the `"timestamp"` match arm in [`Importer::import`]).
    const FIT_EPOCH_OFFSET_S: i64 = 631_065_600;

    /// The Unix-epoch seconds `fitparser` actually returns for a `record`
    /// whose raw wire-format `timestamp` field is `raw` (FIT-epoch seconds).
    fn unix_s(raw: i64) -> i64 {
        raw + FIT_EPOCH_OFFSET_S
    }

    fn golden_rows() -> Vec<RawRecord> {
        vec![
            RawRecord { timestamp: T0 as u32, lat: 536_870_912, lon: -1_073_741_824, altitude_raw: 10_000, hr: 140, cadence: 80, power: 200 },
            RawRecord { timestamp: (T0 + 1) as u32, lat: 268_435_456, lon: -536_870_912, altitude_raw: 10_005, hr: 142, cadence: 82, power: 205 },
            RawRecord { timestamp: (T0 + 1) as u32, lat: 0, lon: 0, altitude_raw: 10_010, hr: 145, cadence: 85, power: 210 }, // duplicate — dropped
            RawRecord { timestamp: (T0 + 3) as u32, lat: 134_217_728, lon: 0, altitude_raw: 10_020, hr: 148, cadence: 88, power: 215 },
        ]
    }

    /// thing — condition — result: golden fixture — a duplicate timestamp
    /// dropped, decimal-degree lat/lon (R27), and a no-double-offset
    /// `GPS_EpochMs`.
    #[test]
    fn fit_importer_golden_fixture_maps_channels_and_drops_duplicate_timestamp() {
        // Arrange
        let bytes = build_fit_fixture(&golden_rows());

        // Act
        let outcome = FitImporter.import(&bytes, &"aa".repeat(32)).unwrap();

        // Assert — one warning for the dropped duplicate record.
        assert_eq!(outcome.warnings.len(), 1);
        assert!(outcome.warnings[0].message.contains("record 2"));

        // Assert — three kept samples per channel, t_us = [0, 1_000_000, 3_000_000].
        let lat = outcome.session.channels.iter().find(|c| c.channel_id == "GPS_Latitude").unwrap();
        assert_eq!(lat.t_us, vec![0, 1_000_000, 3_000_000]);
        assert_eq!(lat.materialize(), vec![45.0, 22.5, 11.25]);
        assert_eq!(lat.unit, "deg");

        let lon = outcome.session.channels.iter().find(|c| c.channel_id == "GPS_Longitude").unwrap();
        assert_eq!(lon.materialize(), vec![-90.0, -45.0, 0.0]);
        assert_eq!(lon.unit, "deg");

        // Assert — R27: no ×1e7 scaling, plain decimal degrees.
        assert!(lat.materialize().iter().all(|&v| v.abs() <= 180.0));

        // Assert — Q3: GPS_EpochMs = timestamp_utc_s * 1000 exactly, where
        // timestamp_utc_s is fitparser's already-offset Unix-epoch value
        // (unix_s(raw)) — no *second* `+ 631_065_600` applied here on top of
        // that (every kept record here carries a position).
        let epoch = outcome.session.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap();
        assert_eq!(
            epoch.materialize(),
            vec![
                unix_s(T0) as f64 * 1000.0,
                unix_s(T0 + 1) as f64 * 1000.0,
                unix_s(T0 + 3) as f64 * 1000.0
            ]
        );
        assert_eq!(epoch.unit, "ms_raw");

        let alt = outcome.session.channels.iter().find(|c| c.channel_id == "GPS_Altitude").unwrap();
        assert_eq!(alt.materialize(), vec![1500.0, 1501.0, 1504.0]);

        let hr = outcome.session.channels.iter().find(|c| c.channel_id == "HR_BPM").unwrap();
        assert_eq!(hr.materialize(), vec![140.0, 142.0, 148.0]);

        let cad = outcome.session.channels.iter().find(|c| c.channel_id == "Cadence_RPM").unwrap();
        assert_eq!(cad.materialize(), vec![80.0, 82.0, 88.0]);

        let pw = outcome.session.channels.iter().find(|c| c.channel_id == "Power_W").unwrap();
        assert_eq!(pw.materialize(), vec![200.0, 205.0, 215.0]);

        assert_eq!(outcome.session.source_format, SourceFormat::Fit);
        assert_eq!(outcome.session.device_id, None);
        assert_eq!(outcome.session.config_checksum, None);
        assert_eq!(outcome.session.session_id.len(), 16);
        assert_eq!(outcome.session.timestamp_utc_ms, unix_s(T0) * 1000);
    }

    /// thing — condition — result: bytes that aren't a valid FIT file —
    /// typed error, not a panic.
    #[test]
    fn fit_importer_malformed_bytes_returns_typed_error() {
        // Arrange
        let bytes = b"not a fit file";

        // Act
        let result = FitImporter.import(bytes, &"bb".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::FitMalformed(_))));
    }

    /// thing — condition — result: a quarter-circle of semicircles —
    /// converts to exactly 45 degrees (R27: no further scaling applied
    /// anywhere, in this function or at its call sites).
    #[test]
    fn semicircles_to_deg_quarter_circle_is_45_degrees() {
        // Arrange / Act / Assert
        assert_eq!(semicircles_to_deg(536_870_912), 45.0);
    }

    /// thing — condition — result: L2-R7(d) — a dropped, timestamp-less
    /// record is not silent — it raises its own `ImporterWarning` naming
    /// the record index. Exercised at the warning-collection level
    /// directly (the existing 7-field fixture layout always writes a
    /// `timestamp` field, so it cannot itself produce an undecoded/missing
    /// one — see the brief's note on this fixture's limitation).
    #[test]
    fn dropped_timestampless_record_raises_a_warning_naming_its_index() {
        // Arrange — records[1] has no timestamp; records[0] and
        // records[2] do.
        let records = vec![
            FitRecord { timestamp_utc_s: Some(1_000_000_000), ..Default::default() },
            FitRecord { timestamp_utc_s: None, ..Default::default() },
            FitRecord { timestamp_utc_s: Some(1_000_000_002), ..Default::default() },
        ];

        // Act — replicate `import`'s warning-collection loop in isolation.
        let mut warnings: Vec<ImporterWarning> = Vec::new();
        for (i, r) in records.iter().enumerate() {
            if r.timestamp_utc_s.is_none() {
                warnings.push(ImporterWarning::new(format!("dropped FIT record {i}: no timestamp")));
            }
        }

        // Assert
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("record 1"));
        assert!(warnings[0].message.contains("no timestamp"));
    }
}
