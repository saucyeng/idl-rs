//! GPX (`.gpx`) import — port of the Dart `GpxParser`
//! (`app/lib/data/gpx_parser.dart`, idl0-app). Channel mapping and
//! derivation rules are documented in `docs/IDL0_SPEC.md` §15a.3.

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use crate::session::{Channel, Session, SourceFormat};

use super::{ImportedSession, Importer, ImporterError, ImporterWarning};

/// This importer implementation's own version string (C1 §4.3).
pub const GPX_IMPORTER_VERSION: &str = "0.1.0";

/// Imports Garmin/Strava-style `.gpx` track exports.
pub struct GpxImporter;

/// One `<trkpt>`'s raw field values, `None` when the element/attribute was
/// absent. Mirrors the Dart parser's per-point extraction.
#[derive(Debug, Clone, Default)]
struct TrkPt {
    /// Latitude, decimal degrees (R27 — physical, unscaled).
    lat_deg: Option<f64>,
    /// Longitude, decimal degrees (R27 — physical, unscaled).
    lon_deg: Option<f64>,
    /// Elevation, metres.
    ele_m: Option<f64>,
    /// Recorded time, UTC milliseconds since the Unix epoch.
    time_utc_ms: Option<i64>,
    /// Heart rate, beats per minute (from a `TrackPointExtension`).
    hr_bpm: Option<f64>,
    /// Cadence, revolutions per minute (from a `TrackPointExtension`).
    cadence_rpm: Option<f64>,
    /// Power, watts (from a `TrackPointExtension`).
    power_w: Option<f64>,
}

impl Importer for GpxImporter {
    fn source_format(&self) -> SourceFormat {
        SourceFormat::Gpx
    }

    fn import(&self, bytes: &[u8], blob_sha256: &str) -> Result<ImportedSession, ImporterError> {
        let text = std::str::from_utf8(bytes).map_err(|e| ImporterError::NotUtf8(e.to_string()))?;
        let trkpts = parse_trkpts(text)?;
        if trkpts.is_empty() {
            return Err(ImporterError::GpxNoTrackpoints);
        }

        let mut warnings = Vec::new();

        // L2-R7: partition into case (b) — no trackpoint anywhere has a
        // parseable <time> — and case (c) — some do, some don't.
        let any_timestamped = trkpts.iter().any(|p| p.time_utc_ms.is_some());

        let (kept, timestamp_utc_ms, create_epoch_ms): (Vec<(usize, i64)>, i64, bool) = if !any_timestamped
        {
            // Case (b): synthesize a 1 Hz index from t_us = 0, one warning
            // for the whole file, no GPS_EpochMs (synthesized index-ms must
            // never share an axis or a column with real epoch-ms).
            let n = trkpts.len();
            warnings.push(ImporterWarning::new(format!(
                "{n} GPX trackpoints have no parseable <time> — synthesized a 1 Hz index"
            )));
            let kept: Vec<(usize, i64)> = (0..n).map(|i| (i, i as i64 * 1_000_000)).collect();
            (kept, 0, false)
        } else {
            // Case (c): the timestamped points alone define the axis; every
            // untimestamped point is dropped, each with its own warning.
            let mut timestamped: Vec<(usize, i64)> = Vec::with_capacity(trkpts.len());
            for (i, p) in trkpts.iter().enumerate() {
                match p.time_utc_ms {
                    Some(ms) => timestamped.push((i, ms)),
                    None => warnings.push(ImporterWarning::new(format!(
                        "dropped GPX trackpoint {i}: no parseable <time>"
                    ))),
                }
            }
            // L2-R7(a): t0 is the minimum timestamp among the kept points,
            // not the first one encountered — a GPX file is not guaranteed
            // sorted by time.
            let t0_ms = timestamped
                .iter()
                .map(|&(_, ms)| ms)
                .min()
                .expect("any_timestamped guarantees at least one entry");

            // Duplicate/non-monotonic dedup (C1 §3.4), scoped to the
            // timestamped, kept points.
            let mut kept: Vec<(usize, i64)> = Vec::with_capacity(timestamped.len());
            let mut last_t_us: Option<i64> = None;
            for (i, ms) in timestamped {
                let t_us = (ms - t0_ms) * 1000;
                if let Some(last) = last_t_us {
                    if t_us <= last {
                        warnings.push(ImporterWarning::new(format!(
                            "dropped GPX trackpoint {i}: duplicate/non-monotonic timestamp"
                        )));
                        continue;
                    }
                }
                last_t_us = Some(t_us);
                kept.push((i, t_us));
            }
            (kept, t0_ms, true)
        };

        let mut channels = Vec::new();
        // GPS_Latitude/GPS_Longitude have no Option semantics (R27) — every
        // kept trackpoint has both, or parse_trkpts already raised
        // GpxMissingLatLon.
        channels.push(push_channel(&kept, &trkpts, "GPS_Latitude", "deg", |p| {
            p.lat_deg
        }));
        channels.push(push_channel(&kept, &trkpts, "GPS_Longitude", "deg", |p| {
            p.lon_deg
        }));

        // L2-R6: optional per-field channels, gated on the kept set — no
        // zero-filling, a dropped point's field never fabricates a channel
        // with zero real samples.
        if kept.iter().any(|&(i, _)| trkpts[i].ele_m.is_some()) {
            channels.push(push_channel(&kept, &trkpts, "GPS_Altitude", "m", |p| {
                p.ele_m
            }));
        }
        if create_epoch_ms {
            channels.push(push_channel(&kept, &trkpts, "GPS_EpochMs", "ms_raw", |p| {
                p.time_utc_ms.map(|v| v as f64)
            }));
        }
        if kept.iter().any(|&(i, _)| trkpts[i].hr_bpm.is_some()) {
            channels.push(push_channel(&kept, &trkpts, "HR_BPM", "bpm", |p| p.hr_bpm));
        }
        if kept.iter().any(|&(i, _)| trkpts[i].cadence_rpm.is_some()) {
            channels.push(push_channel(&kept, &trkpts, "Cadence_RPM", "rpm", |p| {
                p.cadence_rpm
            }));
        }
        if kept.iter().any(|&(i, _)| trkpts[i].power_w.is_some()) {
            channels.push(push_channel(&kept, &trkpts, "Power_W", "W", |p| p.power_w));
        }

        let session = Session {
            session_id: super::session_id_from_blob_hash(blob_sha256),
            device_id: None,
            timestamp_utc_ms,
            config_checksum: None,
            source_format: SourceFormat::Gpx,
            blob_sha256: blob_sha256.to_string(),
            channels,
        };

        Ok(ImportedSession { session, warnings })
    }

    fn importer_version(&self) -> &'static str {
        GPX_IMPORTER_VERSION
    }
}

/// Builds one channel from the accessor's values at the kept trackpoints —
/// `kept` holds `(trackpoint index, its t_us)` pairs (L2-R7's axis). Keeps
/// only samples where `f` returns `Some` (L2-R6 — filter-and-collect, never
/// map-with-default): an absent field contributes no sample, not a `0.0`
/// one. `GPS_Latitude`/`GPS_Longitude` are always `Some` over the kept set
/// by construction, so this reduces to "every kept point" for those two.
fn push_channel(
    kept: &[(usize, i64)],
    trkpts: &[TrkPt],
    channel_id: &str,
    unit: &str,
    f: impl Fn(&TrkPt) -> Option<f64>,
) -> Channel {
    let mut t_us = Vec::with_capacity(kept.len());
    let mut values = Vec::with_capacity(kept.len());
    for &(i, t) in kept {
        if let Some(v) = f(&trkpts[i]) {
            t_us.push(t);
            values.push(v);
        }
    }
    Channel::from_f64_with_times(channel_id, 0.0, values, t_us, "gpx").with_unit(unit)
}

/// Streams `text` once, collecting every `<trkpt>` into a [`TrkPt`].
/// Namespace-prefix-agnostic local-name matching (`quick_xml`'s
/// `local_name()`), matching the Dart parser's approach.
fn parse_trkpts(text: &str) -> Result<Vec<TrkPt>, ImporterError> {
    let mut reader = Reader::from_str(text);
    {
        let cfg = reader.config_mut();
        cfg.trim_text_start = true;
        cfg.trim_text_end = true;
    }

    let mut trkpts = Vec::new();
    let mut in_trkpt = false;
    let mut current = TrkPt::default();
    let mut path: Vec<String> = Vec::new();

    loop {
        let event = reader
            .read_event()
            .map_err(|e| ImporterError::GpxMalformedXml(e.to_string()))?;
        match event {
            Event::Eof => break,
            Event::Start(e) => {
                let local = local_str(&e);
                if local == "trkpt" {
                    in_trkpt = true;
                    current = TrkPt::default();
                    read_lat_lon(&e, &mut current)?;
                } else if in_trkpt {
                    path.push(local);
                }
            }
            Event::Empty(e) => {
                // L2-R8: `<trkpt lat lon/>` is valid GPX with valid
                // coordinates and no children — not a GpxMissingLatLon
                // error. It falls into L2-R7's missing-timestamp handling
                // like any other untimestamped point.
                if local_str(&e) == "trkpt" {
                    let mut pt = TrkPt::default();
                    read_lat_lon(&e, &mut pt)?;
                    trkpts.push(pt);
                }
            }
            Event::Text(t) => {
                if in_trkpt {
                    if let Some(field) = path.last().cloned() {
                        let decoded = t
                            .decode()
                            .map_err(|e| ImporterError::GpxMalformedXml(e.to_string()))?;
                        assign_field(&mut current, &field, decoded.trim());
                    }
                }
            }
            Event::CData(t) => {
                // A `<time><![CDATA[...]]></time>` must not silently yield
                // no timestamp (L2-R8) — same handling as Event::Text.
                if in_trkpt {
                    if let Some(field) = path.last().cloned() {
                        let decoded = t
                            .decode()
                            .map_err(|e| ImporterError::GpxMalformedXml(e.to_string()))?;
                        assign_field(&mut current, &field, decoded.trim());
                    }
                }
            }
            Event::End(e) => {
                let local = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                if local == "trkpt" {
                    in_trkpt = false;
                    trkpts.push(current.clone());
                } else if in_trkpt && path.last().map(|s| s.as_str()) == Some(local.as_str()) {
                    path.pop();
                }
            }
            _ => {}
        }
    }

    Ok(trkpts)
}

/// The local (namespace-prefix-stripped) name of a start/empty tag.
fn local_str(e: &BytesStart) -> String {
    String::from_utf8_lossy(e.local_name().as_ref()).into_owned()
}

/// Reads `lat`/`lon` attributes off `e` into `out`. Shared by
/// `Event::Start` and `Event::Empty` handling (L2-R8) since both carry the
/// same attribute set for a `<trkpt>` tag.
fn read_lat_lon(e: &BytesStart, out: &mut TrkPt) -> Result<(), ImporterError> {
    for attr in e.attributes() {
        let attr = attr.map_err(|err| ImporterError::GpxMalformedXml(err.to_string()))?;
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        if key == "lat" || key == "lon" {
            let val = attr
                .unescape_value()
                .map_err(|err| ImporterError::GpxMalformedXml(err.to_string()))?
                .into_owned();
            let parsed = val
                .trim()
                .parse::<f64>()
                .map_err(|_| ImporterError::GpxUnparseableLatLon(format!("{key}=\"{val}\"")))?;
            if key == "lat" {
                out.lat_deg = Some(parsed);
            } else {
                out.lon_deg = Some(parsed);
            }
        }
    }
    if out.lat_deg.is_none() || out.lon_deg.is_none() {
        return Err(ImporterError::GpxMissingLatLon(
            "<trkpt> missing required lat/lon attribute".to_string(),
        ));
    }
    Ok(())
}

/// Assigns one decoded text-node value to the field its element `field`
/// names, namespace-prefix-agnostic (matches by local name only). Unknown
/// fields (any element name not listed) are ignored.
fn assign_field(p: &mut TrkPt, field: &str, text: &str) {
    match field {
        "ele" => p.ele_m = text.parse().ok(),
        "time" => p.time_utc_ms = parse_iso8601_utc_ms(text),
        "hr" => p.hr_bpm = text.parse().ok(),
        "cad" => p.cadence_rpm = text.parse().ok(),
        "power" => p.power_w = text.parse().ok(),
        _ => {}
    }
}

/// Parses a GPX `<time>` value (`YYYY-MM-DDTHH:MM:SS[.fraction]Z`, always
/// UTC per the GPX schema) into UTC milliseconds since the Unix epoch,
/// rounding any sub-millisecond fraction to the nearest millisecond, ties
/// away from zero (C1 §3.4). `None` if the string does not match this shape.
fn parse_iso8601_utc_ms(s: &str) -> Option<i64> {
    let s = s.trim().strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;

    let (hms, frac) = match time.split_once('.') {
        Some((hms, frac)) => (hms, Some(frac)),
        None => (time, None),
    };
    let mut time_parts = hms.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next()?.parse().ok()?;

    let days = days_since_epoch(year, month, day)?;
    let mut ms = ((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000;

    if let Some(frac) = frac {
        let digits: Vec<u32> = frac.chars().filter_map(|c| c.to_digit(10)).collect();
        let ms_frac: i64 = digits
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, &d)| d as i64 * 10i64.pow(2 - i as u32))
            .sum();
        let round_up = digits.get(3).map(|&d| d >= 5).unwrap_or(false);
        ms += ms_frac + if round_up { 1 } else { 0 };
    }

    Some(ms)
}

/// Days from the Unix epoch (1970-01-01) to `year-month-day`, proleptic
/// Gregorian — Howard Hinnant's "days from civil" algorithm (public domain,
/// howardhinnant.github.io/date_algorithms.html).
fn days_since_epoch(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (month + 9) % 12; // [0, 11], Mar=0 .. Feb=11
    let doy = (153 * mp + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    Some(era * 146097 + doe - 719468)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_GPX: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="idl1-golden-fixture" xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
  <metadata><name>Golden Loop</name></metadata>
  <trk><trkseg>
    <trkpt lat="45.0" lon="-90.0">
      <ele>1500.0</ele>
      <time>2026-06-01T12:00:00Z</time>
      <extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>140</gpxtpx:hr></gpxtpx:TrackPointExtension></extensions>
    </trkpt>
    <trkpt lat="45.001" lon="-90.001">
      <ele>1501.0</ele>
      <time>2026-06-01T12:00:01Z</time>
      <extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>142</gpxtpx:hr></gpxtpx:TrackPointExtension></extensions>
    </trkpt>
    <trkpt lat="45.002" lon="-90.002">
      <ele>1502.0</ele>
      <time>2026-06-01T12:00:01Z</time>
      <extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>145</gpxtpx:hr></gpxtpx:TrackPointExtension></extensions>
    </trkpt>
    <trkpt lat="45.003" lon="-90.003">
      <time>2026-06-01T12:00:03Z</time>
    </trkpt>
  </trkseg></trk>
</gpx>"#;

    /// thing — condition — result: golden fixture — a duplicate timestamp
    /// and a point with no `<ele>`/`<hr>` — maps channels per L2-R6/R27 and
    /// drops the duplicate.
    #[test]
    fn gpx_importer_golden_fixture_maps_channels_and_drops_duplicate_timestamp() {
        // Arrange / Act
        let outcome = GpxImporter
            .import(GOLDEN_GPX.as_bytes(), &"aa".repeat(32))
            .unwrap();

        // Assert — one warning for the dropped duplicate trackpoint.
        assert_eq!(outcome.warnings.len(), 1);
        assert!(outcome.warnings[0].message.contains("trackpoint 2"));

        // Assert — three kept samples, t_us = [0, 1_000_000, 3_000_000].
        let lat = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Latitude")
            .unwrap();
        assert_eq!(lat.t_us, vec![0, 1_000_000, 3_000_000]);
        assert_eq!(lat.materialize(), vec![45.0, 45.001, 45.003]);
        assert_eq!(lat.unit, "deg");

        let lon = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Longitude")
            .unwrap();
        assert_eq!(lon.materialize(), vec![-90.0, -90.001, -90.003]);
        assert_eq!(lon.unit, "deg");

        // Assert — L2-R6: row 3 has no <ele>: contributes no sample (not
        // zero-filled) — two samples, not three.
        let alt = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Altitude")
            .unwrap();
        assert_eq!(alt.t_us, vec![0, 1_000_000]);
        assert_eq!(alt.materialize(), vec![1500.0, 1501.0]);
        assert_eq!(alt.unit, "m");

        // Assert — row 3 has no <hr> extension: contributes no sample.
        let hr = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "HR_BPM")
            .unwrap();
        assert_eq!(hr.materialize(), vec![140.0, 142.0]);
        assert_eq!(hr.unit, "bpm");

        // Assert — no <cad>/<power> anywhere: channels absent entirely.
        assert!(outcome
            .session
            .channels
            .iter()
            .all(|c| c.channel_id != "Cadence_RPM"));
        assert!(outcome
            .session
            .channels
            .iter()
            .all(|c| c.channel_id != "Power_W"));

        assert_eq!(outcome.session.source_format, SourceFormat::Gpx);
        assert_eq!(outcome.session.device_id, None);
        assert_eq!(outcome.session.config_checksum, None);
        assert_eq!(outcome.session.session_id.len(), 16);
        assert_eq!(outcome.session.timestamp_utc_ms, 1_780_315_200_000);
    }

    /// thing — condition — result: no `<trkpt>` elements — typed error.
    #[test]
    fn gpx_importer_no_trackpoints_returns_typed_error() {
        // Arrange
        let gpx = r#"<gpx><trk><trkseg></trkseg></trk></gpx>"#;

        // Act
        let result = GpxImporter.import(gpx.as_bytes(), &"bb".repeat(32));

        // Assert
        assert_eq!(result, Err(ImporterError::GpxNoTrackpoints));
    }

    /// thing — condition — result: `<trkpt>` missing `lat` — typed error.
    #[test]
    fn gpx_importer_missing_lat_returns_typed_error() {
        // Arrange
        let gpx = r#"<gpx><trk><trkseg><trkpt lon="1.0"><time>2026-01-01T00:00:00Z</time></trkpt></trkseg></trk></gpx>"#;

        // Act
        let result = GpxImporter.import(gpx.as_bytes(), &"cc".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::GpxMissingLatLon(_))));
    }

    /// thing — condition — result: mismatched closing tag — typed error.
    #[test]
    fn gpx_importer_malformed_xml_returns_typed_error() {
        // Arrange — `</gpx>` closes `<trk>` while it's still open (quick-xml
        // 0.41's `check_end_names`, on by default, rejects this: an
        // unclosed-at-EOF document like `<gpx><trk>` is not itself an error
        // for a streaming reader, since no End event with a mismatched
        // name is ever produced).
        let gpx = "<gpx><trk></gpx>";

        // Act
        let result = GpxImporter.import(gpx.as_bytes(), &"dd".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::GpxMalformedXml(_))));
    }

    /// thing — condition — result: L2-R7 case (b) — no trackpoint anywhere
    /// has a parseable `<time>` — synthesizes a 1 Hz index, warns once, and
    /// never creates `GPS_EpochMs`.
    #[test]
    fn gpx_importer_no_timestamps_at_all_synthesizes_once_and_omits_epoch_ms() {
        // Arrange
        let gpx = r#"<gpx><trk><trkseg><trkpt lat="1.0" lon="2.0"><ele>10.0</ele></trkpt><trkpt lat="1.1" lon="2.1"><ele>11.0</ele></trkpt></trkseg></trk></gpx>"#;

        // Act
        let outcome = GpxImporter.import(gpx.as_bytes(), &"ee".repeat(32)).unwrap();

        // Assert — one warning for the whole file, not one per point.
        assert_eq!(outcome.warnings.len(), 1);
        assert!(outcome.warnings[0].message.contains('2'));

        let lat = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Latitude")
            .unwrap();
        assert_eq!(lat.t_us, vec![0, 1_000_000]);

        assert_eq!(outcome.session.timestamp_utc_ms, 0);
        assert!(outcome
            .session
            .channels
            .iter()
            .all(|c| c.channel_id != "GPS_EpochMs"));
    }

    /// thing — condition — result: L2-R7 case (c) — one of several points
    /// lacks `<time>` — that point is dropped with its own warning, the
    /// kept points' axis starts at 0 relative to the minimum kept
    /// timestamp, and `GPS_EpochMs` is present.
    #[test]
    fn gpx_importer_some_missing_timestamps_drops_them_and_keeps_epoch_ms() {
        // Arrange — the first point has no <time>; the other two do, in
        // ascending time order (the minimum kept timestamp is also the
        // first encountered, so this isolates the drop-and-keep behaviour
        // from the dedup pass's own document-order stepping).
        let gpx = r#"<gpx><trk><trkseg>
            <trkpt lat="1.0" lon="2.0"><ele>10.0</ele></trkpt>
            <trkpt lat="1.1" lon="2.1"><time>2000-01-01T00:00:00Z</time></trkpt>
            <trkpt lat="1.2" lon="2.2"><time>2000-01-01T00:00:02Z</time></trkpt>
        </trkseg></trk></gpx>"#;

        // Act
        let outcome = GpxImporter.import(gpx.as_bytes(), &"ff".repeat(32)).unwrap();

        // Assert — one warning naming the dropped (untimestamped) point.
        assert_eq!(outcome.warnings.len(), 1);
        assert!(outcome.warnings[0].message.contains("trackpoint 0"));
        assert!(outcome.warnings[0].message.contains("no parseable"));

        // Assert — t0 is the minimum kept timestamp (point index 1's
        // 00:00:00Z), kept points' t_us starts at 0.
        let lat = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Latitude")
            .unwrap();
        assert_eq!(lat.t_us, vec![0, 2_000_000]);
        assert_eq!(lat.materialize(), vec![1.1, 1.2]);
        assert_eq!(outcome.session.timestamp_utc_ms, 946_684_800_000);

        // Assert — GPS_EpochMs is present (case (c), real kept timestamps).
        let epoch = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_EpochMs")
            .unwrap();
        assert_eq!(epoch.materialize(), vec![946_684_800_000.0, 946_684_802_000.0]);
    }

    /// thing — condition — result: L2-R8 — a self-closing `<trkpt/>` with
    /// valid lat/lon imports successfully rather than raising
    /// `GpxMissingLatLon`.
    #[test]
    fn gpx_importer_self_closing_trkpt_with_valid_lat_lon_imports_successfully() {
        // Arrange
        let gpx = r#"<gpx><trk><trkseg><trkpt lat="1.0" lon="2.0"/></trkseg></trk></gpx>"#;

        // Act
        let outcome = GpxImporter.import(gpx.as_bytes(), &"11".repeat(32)).unwrap();

        // Assert — falls into case (b): no trackpoint has a <time>.
        assert_eq!(outcome.warnings.len(), 1);
        let lat = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Latitude")
            .unwrap();
        assert_eq!(lat.materialize(), vec![1.0]);
        let lon = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "GPS_Longitude")
            .unwrap();
        assert_eq!(lon.materialize(), vec![2.0]);
    }

    /// thing — condition — result: `<time>` wrapped in CDATA — parses the
    /// same as plain text, not the missing-timestamp path.
    #[test]
    fn gpx_importer_cdata_wrapped_time_parses_like_plain_text() {
        // Arrange
        let gpx = r#"<gpx><trk><trkseg><trkpt lat="1.0" lon="2.0"><time><![CDATA[2000-01-01T00:00:00Z]]></time></trkpt></trkseg></trk></gpx>"#;

        // Act
        let outcome = GpxImporter.import(gpx.as_bytes(), &"22".repeat(32)).unwrap();

        // Assert — real timestamp recognized: case (c)'s GPS_EpochMs path
        // (a single timestamped point, nothing dropped).
        assert_eq!(outcome.warnings.len(), 0);
        assert_eq!(outcome.session.timestamp_utc_ms, 946_684_800_000);
        assert!(outcome
            .session
            .channels
            .iter()
            .any(|c| c.channel_id == "GPS_EpochMs"));
    }

    /// thing — condition — result: year-2000 anchor — matches the
    /// widely-verified Unix epoch second count.
    #[test]
    fn parse_iso8601_utc_ms_at_the_year_2000_anchor_matches_known_epoch() {
        // Arrange — 2000-01-01T00:00:00Z is a widely-verified 946684800 s.

        // Act
        let ms = parse_iso8601_utc_ms("2000-01-01T00:00:00Z");

        // Assert
        assert_eq!(ms, Some(946_684_800_000));
    }

    /// thing — condition — result: a 4th fractional digit of 5 or more —
    /// rounds the millisecond up, ties away from zero.
    #[test]
    fn parse_iso8601_utc_ms_rounds_subsecond_fraction_ties_away_from_zero() {
        // Arrange — .1235 s → 123 ms plus a 4th-digit-5 round-up to 124 ms.

        // Act
        let ms = parse_iso8601_utc_ms("2000-01-01T00:00:00.1235Z");

        // Assert
        assert_eq!(ms, Some(946_684_800_124));
    }
}
