//! GPMF (GoPro Metadata Format) KLV parsing → camera telemetry.
//! See docs/IDL0_SPEC.md §33.3 and GoPro's published gpmf-parser format doc.
//!
//! KLV: 4-byte FourCC key, 1-byte type char, 1-byte struct size, 2-byte BE
//! repeat count; payload = size × repeat bytes, padded to 4-byte alignment.
//! Type 0x00 = nested container. GPS lives under `DEVC` → `STRM` as `GPS5`
//! (i32×5: lat, lon, alt, speed2d, speed3d; scaled by `SCAL`) or `GPS9`
//! (per-sample date/time included), with `GPSU` (UTC string) as time anchor.

use crate::video::mp4box::GpmdSample;
use crate::video::VideoError;

/// One camera GPS fix on the video clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryFix {
    /// Video-relative time in seconds.
    pub t_video_s: f64,
    /// Latitude in degrees (already descaled).
    pub lat_deg: f64,
    /// Longitude in degrees.
    pub lon_deg: f64,
    /// 2D ground speed in m/s.
    pub speed_mps: f64,
}

/// Camera telemetry extracted from GPMF.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VideoTelemetry {
    /// (video time s, UTC epoch ms) from the first `GPSU`/`GPS9` stamp.
    pub utc_anchor: Option<(f64, i64)>,
    pub fixes: Vec<TelemetryFix>,
}

fn err_parse(msg: &str) -> VideoError {
    VideoError::parse(format!("gpmf: {msg}"))
}

/// One decoded KLV item borrowed from the payload.
struct Klv<'a> {
    key: [u8; 4],
    type_char: u8,
    struct_size: usize,
    repeat: usize,
    value: &'a [u8],
}

/// Walk the KLV items of `data`, calling `f` for each.
fn walk_klv<'a>(
    data: &'a [u8],
    f: &mut impl FnMut(Klv<'a>) -> Result<(), VideoError>,
) -> Result<(), VideoError> {
    let mut off = 0usize;
    while off < data.len() {
        if off + 8 > data.len() {
            return Err(err_parse("truncated KLV header"));
        }
        let key: [u8; 4] = data[off..off + 4].try_into().unwrap();
        let type_char = data[off + 4];
        let struct_size = data[off + 5] as usize;
        let repeat = u16::from_be_bytes(data[off + 6..off + 8].try_into().unwrap()) as usize;
        let len = struct_size * repeat;
        let padded = len.div_ceil(4) * 4;
        let value = data
            .get(off + 8..off + 8 + len)
            .ok_or_else(|| err_parse("KLV value overruns payload"))?;
        f(Klv {
            key,
            type_char,
            struct_size,
            repeat,
            value,
        })?;
        off += 8 + padded;
    }
    Ok(())
}

/// Parse a `GPSU` UTC string (`yymmddhhmmss.sss`, ASCII) → UTC epoch ms.
/// Years are 20xx.
fn parse_gpsu(s: &[u8]) -> Option<i64> {
    let txt = std::str::from_utf8(s).ok()?.trim_end_matches('\0').trim();
    if txt.len() < 12 {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<i64> { txt.get(r)?.parse::<i64>().ok() };
    let year = 2000 + num(0..2)?;
    let month = num(2..4)?;
    let day = num(4..6)?;
    let hour = num(6..8)?;
    let minute = num(8..10)?;
    let sec_f: f64 = txt.get(10..)?.parse().ok()?;
    let days = days_from_civil(year, month, day)?;
    let ms = (((days * 24 + hour) * 60 + minute) * 60) as f64 * 1000.0 + sec_f * 1000.0;
    Some(ms as i64)
}

/// Days since the Unix epoch for a civil date (proleptic Gregorian).
/// Howard Hinnant's `days_from_civil` algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> Option<i64> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// Per-STRM accumulated GPS state.
#[derive(Default)]
struct StreamGps {
    /// Raw GPS5 rows (lat, lon, alt, speed2d, speed3d).
    gps5: Vec<[i64; 5]>,
    /// Raw GPS9 rows (lat, lon, alt, speed2d, speed3d, days, secs, dop, fix).
    gps9: Vec<[i64; 9]>,
    /// SCAL divisors, one per column.
    scal: Vec<f64>,
    /// GPSU epoch ms.
    gpsu_ms: Option<i64>,
}

fn read_i32s(value: &[u8], struct_size: usize) -> Vec<Vec<i64>> {
    let cols = struct_size / 4;
    value
        .chunks_exact(struct_size)
        .map(|row| {
            (0..cols)
                .map(|c| i32::from_be_bytes(row[c * 4..c * 4 + 4].try_into().unwrap()) as i64)
                .collect()
        })
        .collect()
}

/// Parse the GPMF payloads of every `gpmd` sample into telemetry. GPS5 rows
/// have no per-sample time: a payload's N fixes spread evenly across
/// `[sample.t, next_sample.t)` (the last payload reuses the previous span).
pub fn parse_gpmf(samples: &[GpmdSample]) -> Result<VideoTelemetry, VideoError> {
    let mut telemetry = VideoTelemetry::default();

    for (i, sample) in samples.iter().enumerate() {
        if sample.payload.is_empty() {
            continue;
        }
        let span_s = match samples.get(i + 1) {
            Some(next) => (next.t_video_s - sample.t_video_s).max(0.0),
            None => samples
                .get(i.wrapping_sub(1))
                .map(|prev| (sample.t_video_s - prev.t_video_s).max(0.0))
                .unwrap_or(1.0),
        };

        let mut streams: Vec<StreamGps> = Vec::new();
        collect_streams(&sample.payload, &mut streams)?;

        for s in streams {
            // Anchor: prefer GPSU; else derive from the first GPS9 row.
            let anchor_ms = s.gpsu_ms.or_else(|| {
                s.gps9.first().map(|row| {
                    // GPS9: days since 2000-01-01 (col 5), seconds-of-day
                    // scaled (col 6). Scale factors come from SCAL cols 5/6.
                    let day_ms = (days_from_civil(2000, 1, 1).unwrap() + row[5]) * 86_400_000;
                    let sec_scale = s.scal.get(6).copied().unwrap_or(1000.0);
                    day_ms + (row[6] as f64 / sec_scale * 1000.0) as i64
                })
            });
            if telemetry.utc_anchor.is_none() {
                if let Some(ms) = anchor_ms {
                    telemetry.utc_anchor = Some((sample.t_video_s, ms));
                }
            }

            let scal = |col: usize| {
                s.scal
                    .get(col)
                    .copied()
                    .filter(|&v| v != 0.0)
                    .unwrap_or(1.0)
            };
            let rows: Vec<[i64; 3]> = if !s.gps5.is_empty() {
                s.gps5.iter().map(|r| [r[0], r[1], r[3]]).collect()
            } else {
                s.gps9.iter().map(|r| [r[0], r[1], r[3]]).collect()
            };
            let n = rows.len();
            for (j, row) in rows.into_iter().enumerate() {
                let t = sample.t_video_s + span_s * j as f64 / n as f64;
                telemetry.fixes.push(TelemetryFix {
                    t_video_s: t,
                    lat_deg: row[0] as f64 / scal(0),
                    lon_deg: row[1] as f64 / scal(1),
                    speed_mps: row[2] as f64 / scal(3),
                });
            }
        }
    }
    Ok(telemetry)
}

/// Recurse DEVC containers collecting GPS-bearing STRMs.
fn collect_streams(payload: &[u8], out: &mut Vec<StreamGps>) -> Result<(), VideoError> {
    walk_klv(payload, &mut |klv| {
        match &klv.key {
            b"DEVC" => collect_streams(klv.value, out)?,
            b"STRM" => {
                let mut s = StreamGps::default();
                walk_klv(klv.value, &mut |inner| {
                    match (&inner.key, inner.type_char) {
                        (b"GPS5", b'l') if inner.struct_size == 20 => {
                            s.gps5 = read_i32s(inner.value, 20)
                                .into_iter()
                                .map(|r| [r[0], r[1], r[2], r[3], r[4]])
                                .collect();
                        }
                        (b"GPS9", _) if inner.struct_size >= 36 => {
                            // GoPro packs GPS9 as mixed types but the first 7
                            // columns are 4-byte values; read as i32 grid.
                            s.gps9 = read_i32s(inner.value, inner.struct_size)
                                .into_iter()
                                .filter(|r| r.len() >= 9)
                                .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]])
                                .collect();
                        }
                        (b"SCAL", b'l') => {
                            s.scal = read_i32s(inner.value, 4)
                                .into_iter()
                                .map(|r| r[0] as f64)
                                .collect();
                        }
                        (b"GPSU", b'U') | (b"GPSU", b'c') => {
                            s.gpsu_ms = parse_gpsu(inner.value);
                        }
                        _ => {}
                    }
                    let _ = inner.repeat;
                    Ok(())
                })?;
                if !s.gps5.is_empty() || !s.gps9.is_empty() {
                    out.push(s);
                }
            }
            _ => {}
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::VideoErrorKind;

    /// Build one KLV item (padded to 4-byte alignment).
    fn klv(key: &[u8; 4], type_char: u8, struct_size: u8, repeat: u16, value: &[u8]) -> Vec<u8> {
        assert_eq!(value.len(), struct_size as usize * repeat as usize);
        let mut v = Vec::new();
        v.extend_from_slice(key);
        v.push(type_char);
        v.push(struct_size);
        v.extend_from_slice(&repeat.to_be_bytes());
        v.extend_from_slice(value);
        while v.len() % 4 != 0 {
            v.push(0);
        }
        v
    }

    fn nest(key: &[u8; 4], children: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = children.concat();
        // Nested containers use type 0x00 with struct_size 1.
        klv(key, 0x00, 1, body.len() as u16, &body)
    }

    fn gps5_box(rows: &[[i32; 5]]) -> Vec<u8> {
        let mut body = Vec::new();
        for row in rows {
            for v in row {
                body.extend_from_slice(&v.to_be_bytes());
            }
        }
        klv(b"GPS5", b'l', 20, rows.len() as u16, &body)
    }

    fn scal_box(divisors: &[i32]) -> Vec<u8> {
        let mut body = Vec::new();
        for d in divisors {
            body.extend_from_slice(&d.to_be_bytes());
        }
        klv(b"SCAL", b'l', 4, divisors.len() as u16, &body)
    }

    fn gpsu_box(s: &str) -> Vec<u8> {
        klv(b"GPSU", b'U', s.len() as u8, 1, s.as_bytes())
    }

    #[test]
    fn parse_gpmf_gps5_with_scal_and_gpsu_yields_scaled_fixes_and_anchor() {
        // Arrange — one payload at t=2.0 s, next at 3.0 s.
        let payload = nest(
            b"DEVC",
            &[nest(
                b"STRM",
                &[
                    gpsu_box("240607143025.000"),
                    scal_box(&[10_000_000, 10_000_000, 1000, 1000, 100]),
                    gps5_box(&[
                        [471_234_567, 82_345_678, 0, 5_000, 0],
                        [471_234_667, 82_345_778, 0, 6_000, 0],
                    ]),
                ],
            )],
        );
        let samples = vec![
            GpmdSample {
                t_video_s: 2.0,
                payload,
            },
            GpmdSample {
                t_video_s: 3.0,
                payload: vec![],
            },
        ];

        // Act
        let t = parse_gpmf(&samples).unwrap();

        // Assert
        assert_eq!(t.fixes.len(), 2);
        assert!((t.fixes[0].lat_deg - 47.1234567).abs() < 1e-9);
        assert!((t.fixes[0].lon_deg - 8.2345678).abs() < 1e-9);
        assert!((t.fixes[0].speed_mps - 5.0).abs() < 1e-9);
        assert!((t.fixes[0].t_video_s - 2.0).abs() < 1e-9);
        assert!(
            (t.fixes[1].t_video_s - 2.5).abs() < 1e-9,
            "2 fixes spread over [2,3)"
        );
        let (t0, epoch_ms) = t.utc_anchor.unwrap();
        assert_eq!(t0, 2.0);
        assert_eq!(epoch_ms, 1_717_770_625_000); // 2024-06-07T14:30:25Z
    }

    #[test]
    fn parse_gpmf_stream_without_gps_yields_empty_telemetry() {
        // Arrange — accelerometer-only stream.
        let accl = klv(b"ACCL", b's', 6, 1, &[0, 1, 0, 2, 0, 3]);
        let payload = nest(b"DEVC", &[nest(b"STRM", &[accl])]);
        let samples = vec![GpmdSample {
            t_video_s: 0.0,
            payload,
        }];

        // Act
        let t = parse_gpmf(&samples).unwrap();

        // Assert
        assert!(t.fixes.is_empty());
        assert!(t.utc_anchor.is_none());
    }

    #[test]
    fn parse_gpmf_truncated_klv_is_parse_error_not_panic() {
        // Arrange — half a KLV header.
        let samples = vec![GpmdSample {
            t_video_s: 0.0,
            payload: vec![0x47, 0x50, 0x53],
        }];

        // Act
        let err = parse_gpmf(&samples).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::Parse);
    }

    #[test]
    fn parse_gpsu_string_converts_to_utc_epoch_ms() {
        // Arrange + Act + Assert
        assert_eq!(parse_gpsu(b"240607143025.000"), Some(1_717_770_625_000));
        assert_eq!(parse_gpsu(b"700101000000.000"), Some(3_155_760_000_000)); // 2070-01-01
        assert_eq!(parse_gpsu(b"garbage"), None);
    }
}
