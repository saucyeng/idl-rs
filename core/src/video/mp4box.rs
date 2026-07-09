//! Minimal ISO-BMFF (MP4/MOV) walker — just what the video feature needs:
//! the `gpmd` (GPMF) track's sample payloads with video-relative timestamps,
//! `mvhd` creation time, and video-track width/height/fps/duration. No
//! ffmpeg; every read is bounds-checked. See docs/IDL0_SPEC.md §33.3.
//!
//! Hand-rolled rather than the `mp4` crate: its typed `TrackType` rejects
//! tracks whose handler is not video/audio/subtitle, which is exactly what a
//! GoPro `gpmd` metadata track is.

use crate::video::{VideoError, VideoErrorKind};

/// Seconds between the MP4 epoch (1904-01-01) and the Unix epoch (1970-01-01).
const MP4_EPOCH_TO_UNIX_S: u64 = 2_082_844_800;

/// Container facts read without ffmpeg. `fps` and `duration_s` come from the
/// video track's sample table; `creation_time_utc_ms` is `None` when the
/// `mvhd` field is zero or pre-1970.
#[derive(Debug, Clone, PartialEq)]
pub struct Mp4Info {
    /// Coded video width in pixels (from `tkhd`, 16.16 fixed point).
    pub width: u32,
    /// Coded video height in pixels.
    pub height: u32,
    /// Nominal frame rate in frames/second.
    pub fps: f64,
    /// Presentation duration in seconds (from `mvhd`).
    pub duration_s: f64,
    /// Container creation time as UTC milliseconds since the Unix epoch.
    pub creation_time_utc_ms: Option<i64>,
    /// True when a `gpmd` (GPMF telemetry) track is present.
    pub has_gpmd: bool,
}

/// One `gpmd` sample: video-relative decode time (seconds) + raw GPMF bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct GpmdSample {
    /// Sample decode time on the video clock, in seconds.
    pub t_video_s: f64,
    /// Raw GPMF KLV payload.
    pub payload: Vec<u8>,
}

/// Everything collected for one `trak`.
#[derive(Debug, Default, Clone)]
struct Trak {
    /// `hdlr` handler_type FourCC.
    handler: [u8; 4],
    /// First `stsd` entry's sample-format FourCC.
    stsd_format: [u8; 4],
    /// `mdhd` media timescale (ticks/second).
    mdhd_timescale: u32,
    /// `mdhd` media duration in timescale ticks.
    mdhd_duration: u64,
    /// `tkhd` presentation width/height in pixels (from 16.16 fixed).
    width: u32,
    height: u32,
    /// `stts` runs: (sample_count, sample_delta_ticks).
    stts: Vec<(u32, u32)>,
    /// `stsc` runs: (first_chunk 1-based, samples_per_chunk).
    stsc: Vec<(u32, u32)>,
    /// Per-sample sizes in bytes (expanded from `stsz`).
    sizes: Vec<u64>,
    /// Chunk file offsets (`stco`/`co64`).
    chunk_offsets: Vec<u64>,
}

/// Parsed top-level facts.
struct Parsed {
    mvhd_timescale: u32,
    mvhd_duration: u64,
    mvhd_creation_1904_s: u64,
    traks: Vec<Trak>,
}

fn err_parse(msg: &str) -> VideoError {
    VideoError::parse(format!("mp4: {msg}"))
}

/// Bounds-checked big-endian readers over the file buffer.
fn be_u32(b: &[u8], o: usize) -> Result<u32, VideoError> {
    Ok(u32::from_be_bytes(
        b.get(o..o + 4)
            .ok_or_else(|| err_parse("truncated u32"))?
            .try_into()
            .unwrap(),
    ))
}
fn be_u64(b: &[u8], o: usize) -> Result<u64, VideoError> {
    Ok(u64::from_be_bytes(
        b.get(o..o + 8)
            .ok_or_else(|| err_parse("truncated u64"))?
            .try_into()
            .unwrap(),
    ))
}
fn fourcc(b: &[u8], o: usize) -> Result<[u8; 4], VideoError> {
    Ok(b.get(o..o + 4)
        .ok_or_else(|| err_parse("truncated fourcc"))?
        .try_into()
        .unwrap())
}

/// Iterate child boxes of `data[start..end]`, calling `f(kind, payload)`.
fn walk_boxes(
    data: &[u8],
    start: usize,
    end: usize,
    f: &mut impl FnMut([u8; 4], &[u8]) -> Result<(), VideoError>,
) -> Result<(), VideoError> {
    let mut off = start;
    while off + 8 <= end {
        let size32 = be_u32(data, off)? as u64;
        let kind = fourcc(data, off + 4)?;
        let (header, size) = if size32 == 1 {
            (16usize, be_u64(data, off + 8)?)
        } else if size32 == 0 {
            // Box extends to end of enclosing scope.
            (8usize, (end - off) as u64)
        } else {
            (8usize, size32)
        };
        if size < header as u64 || off as u64 + size > end as u64 {
            return Err(err_parse("box overruns its container"));
        }
        let body = &data[off + header..off + size as usize];
        f(kind, body)?;
        off += size as usize;
    }
    Ok(())
}

/// Parse a full-box `mvhd`/`mdhd`-style (version, timescale, duration,
/// creation) tuple. Returns (creation_s_1904, timescale, duration_ticks).
fn parse_timed_header(body: &[u8]) -> Result<(u64, u32, u64), VideoError> {
    let version = *body
        .first()
        .ok_or_else(|| err_parse("empty timed header"))?;
    if version == 1 {
        let creation = be_u64(body, 4)?;
        let timescale = be_u32(body, 20)?;
        let duration = be_u64(body, 24)?;
        Ok((creation, timescale, duration))
    } else {
        let creation = be_u32(body, 4)? as u64;
        let timescale = be_u32(body, 12)?;
        let duration = be_u32(body, 16)? as u64;
        Ok((creation, timescale, duration))
    }
}

fn parse(data: &[u8]) -> Result<Parsed, VideoError> {
    let mut parsed = Parsed {
        mvhd_timescale: 0,
        mvhd_duration: 0,
        mvhd_creation_1904_s: 0,
        traks: Vec::new(),
    };
    let mut saw_moov = false;

    walk_boxes(data, 0, data.len(), &mut |kind, body| {
        if &kind == b"moov" {
            saw_moov = true;
            let base = body.as_ptr() as usize - data.as_ptr() as usize;
            walk_boxes(data, base, base + body.len(), &mut |k2, b2| {
                match &k2 {
                    b"mvhd" => {
                        let (creation, timescale, duration) = parse_timed_header(b2)?;
                        parsed.mvhd_creation_1904_s = creation;
                        parsed.mvhd_timescale = timescale;
                        parsed.mvhd_duration = duration;
                    }
                    b"trak" => {
                        let base2 = b2.as_ptr() as usize - data.as_ptr() as usize;
                        parsed
                            .traks
                            .push(parse_trak(data, base2, base2 + b2.len())?);
                    }
                    _ => {}
                }
                Ok(())
            })?;
        }
        Ok(())
    })?;

    if !saw_moov || parsed.mvhd_timescale == 0 {
        return Err(err_parse("no moov/mvhd found"));
    }
    Ok(parsed)
}

fn parse_trak(data: &[u8], start: usize, end: usize) -> Result<Trak, VideoError> {
    let mut trak = Trak::default();
    walk_boxes(data, start, end, &mut |kind, body| {
        match &kind {
            b"tkhd" => {
                let version = *body.first().ok_or_else(|| err_parse("empty tkhd"))?;
                // width/height are the last two 16.16 values of the box.
                let base = if version == 1 { 88 } else { 76 };
                trak.width = (be_u32(body, base)? >> 16) & 0xFFFF;
                trak.height = (be_u32(body, base + 4)? >> 16) & 0xFFFF;
            }
            b"mdia" => {
                let base = body.as_ptr() as usize - data.as_ptr() as usize;
                walk_boxes(data, base, base + body.len(), &mut |k2, b2| {
                    match &k2 {
                        b"mdhd" => {
                            let (_, timescale, duration) = parse_timed_header(b2)?;
                            trak.mdhd_timescale = timescale;
                            trak.mdhd_duration = duration;
                        }
                        b"hdlr" => {
                            trak.handler = fourcc(b2, 8)?;
                        }
                        b"minf" => {
                            let base2 = b2.as_ptr() as usize - data.as_ptr() as usize;
                            walk_boxes(data, base2, base2 + b2.len(), &mut |k3, b3| {
                                if &k3 == b"stbl" {
                                    let base3 = b3.as_ptr() as usize - data.as_ptr() as usize;
                                    parse_stbl(data, base3, base3 + b3.len(), &mut trak)?;
                                }
                                Ok(())
                            })?;
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(trak)
}

fn parse_stbl(data: &[u8], start: usize, end: usize, trak: &mut Trak) -> Result<(), VideoError> {
    walk_boxes(data, start, end, &mut |kind, body| {
        match &kind {
            b"stsd" => {
                let count = be_u32(body, 4)?;
                if count > 0 && body.len() >= 16 {
                    // First entry: size(4) + format fourcc(4).
                    trak.stsd_format = fourcc(body, 12)?;
                }
            }
            b"stts" => {
                let count = be_u32(body, 4)? as usize;
                for i in 0..count {
                    let o = 8 + i * 8;
                    trak.stts.push((be_u32(body, o)?, be_u32(body, o + 4)?));
                }
            }
            b"stsc" => {
                let count = be_u32(body, 4)? as usize;
                for i in 0..count {
                    let o = 8 + i * 12;
                    trak.stsc.push((be_u32(body, o)?, be_u32(body, o + 4)?));
                }
            }
            b"stsz" => {
                let fixed = be_u32(body, 4)? as u64;
                let count = be_u32(body, 8)? as usize;
                if fixed != 0 {
                    trak.sizes = vec![fixed; count];
                } else {
                    for i in 0..count {
                        trak.sizes.push(be_u32(body, 12 + i * 4)? as u64);
                    }
                }
            }
            b"stco" => {
                let count = be_u32(body, 4)? as usize;
                for i in 0..count {
                    trak.chunk_offsets.push(be_u32(body, 8 + i * 4)? as u64);
                }
            }
            b"co64" => {
                let count = be_u32(body, 4)? as usize;
                for i in 0..count {
                    trak.chunk_offsets.push(be_u64(body, 8 + i * 8)?);
                }
            }
            _ => {}
        }
        Ok(())
    })
}

impl Trak {
    fn is_gpmd(&self) -> bool {
        &self.stsd_format == b"gpmd" || &self.handler == b"gpmd"
    }

    fn is_video(&self) -> bool {
        &self.handler == b"vide"
    }

    fn sample_count(&self) -> u64 {
        self.stts.iter().map(|&(n, _)| n as u64).sum()
    }

    /// Per-sample decode times in seconds (cumulative `stts`).
    fn sample_times_s(&self) -> Vec<f64> {
        let mut out = Vec::new();
        let mut ticks: u64 = 0;
        let ts = self.mdhd_timescale.max(1) as f64;
        for &(count, delta) in &self.stts {
            for _ in 0..count {
                out.push(ticks as f64 / ts);
                ticks += delta as u64;
            }
        }
        out
    }

    /// Absolute file offset of every sample, via `stsc` × `stco` × `stsz`.
    fn sample_offsets(&self) -> Result<Vec<u64>, VideoError> {
        let n_samples = self.sizes.len();
        let mut offsets = Vec::with_capacity(n_samples);
        let mut sample = 0usize;
        let n_chunks = self.chunk_offsets.len();
        for chunk_idx in 0..n_chunks {
            let chunk_no = (chunk_idx + 1) as u32;
            // samples_per_chunk from the last stsc run whose first_chunk <= chunk_no.
            let spc = self
                .stsc
                .iter()
                .take_while(|&&(first, _)| first <= chunk_no)
                .last()
                .map(|&(_, spc)| spc)
                .ok_or_else(|| err_parse("stsc has no run for chunk"))?
                as usize;
            let mut within: u64 = 0;
            for _ in 0..spc {
                if sample >= n_samples {
                    break;
                }
                offsets.push(self.chunk_offsets[chunk_idx] + within);
                within += self.sizes[sample];
                sample += 1;
            }
        }
        if sample < n_samples {
            return Err(err_parse("chunk table covers fewer samples than stsz"));
        }
        Ok(offsets)
    }
}

/// Read container facts from MP4/MOV bytes.
pub fn read_info(bytes: &[u8]) -> Result<Mp4Info, VideoError> {
    let parsed = parse(bytes)?;
    let video = parsed
        .traks
        .iter()
        .find(|t| t.is_video())
        .ok_or_else(|| err_parse("no video track"))?;
    let media_dur_s = video.mdhd_duration as f64 / video.mdhd_timescale.max(1) as f64;
    let fps = if media_dur_s > 0.0 {
        video.sample_count() as f64 / media_dur_s
    } else {
        0.0
    };
    let creation_time_utc_ms = (parsed.mvhd_creation_1904_s > MP4_EPOCH_TO_UNIX_S)
        .then(|| ((parsed.mvhd_creation_1904_s - MP4_EPOCH_TO_UNIX_S) * 1000) as i64);
    Ok(Mp4Info {
        width: video.width,
        height: video.height,
        fps,
        duration_s: parsed.mvhd_duration as f64 / parsed.mvhd_timescale as f64,
        creation_time_utc_ms,
        has_gpmd: parsed.traks.iter().any(|t| t.is_gpmd()),
    })
}

/// Extract every `gpmd` sample (payload + decode time) from MP4/MOV bytes.
pub fn read_gpmd_samples(bytes: &[u8]) -> Result<Vec<GpmdSample>, VideoError> {
    let parsed = parse(bytes)?;
    let trak = parsed
        .traks
        .iter()
        .find(|t| t.is_gpmd())
        .ok_or_else(|| VideoError::new(VideoErrorKind::NoGpmf, "no gpmd track in container"))?;
    let times = trak.sample_times_s();
    let offsets = trak.sample_offsets()?;
    let mut out = Vec::with_capacity(offsets.len());
    for (i, (&off, t)) in offsets.iter().zip(times).enumerate() {
        let size = *trak
            .sizes
            .get(i)
            .ok_or_else(|| err_parse("stsz shorter than offsets"))? as usize;
        let payload = bytes
            .get(off as usize..off as usize + size)
            .ok_or_else(|| err_parse("gpmd sample overruns file"))?
            .to_vec();
        out.push(GpmdSample {
            t_video_s: t,
            payload,
        });
    }
    Ok(out)
}

/// `std::fs` convenience over [`read_info`].
pub fn read_info_path(path: &str) -> Result<Mp4Info, VideoError> {
    let bytes = std::fs::read(path).map_err(|e| VideoError::io(format!("read {path}: {e}")))?;
    read_info(&bytes)
}

/// `std::fs` convenience over [`read_gpmd_samples`].
pub fn read_gpmd_samples_path(path: &str) -> Result<Vec<GpmdSample>, VideoError> {
    let bytes = std::fs::read(path).map_err(|e| VideoError::io(format!("read {path}: {e}")))?;
    read_gpmd_samples(&bytes)
}

/// Synthetic-MP4 builder for tests (no real footage exists yet — design-doc
/// gap). Exposed under the `test-fixtures` feature so the CLI's integration
/// tests can write fixture files.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixture {
    /// Plain box: size + fourcc + payload.
    pub fn boxb(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + payload.len());
        v.extend_from_slice(&((payload.len() as u32 + 8).to_be_bytes()));
        v.extend_from_slice(kind);
        v.extend_from_slice(payload);
        v
    }

    /// Full box payload: version + 3 flag bytes + body.
    pub fn full(version: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![version, 0, 0, 0];
        v.extend_from_slice(body);
        v
    }

    fn mvhd(creation_1904_s: u64, timescale: u32, duration: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(creation_1904_s as u32).to_be_bytes()); // creation
        b.extend_from_slice(&0u32.to_be_bytes()); // modification
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&duration.to_be_bytes());
        b.extend_from_slice(&[0u8; 80]); // rate..next_track_id
        boxb(b"mvhd", &full(0, &b))
    }

    fn tkhd(width: u32, height: u32) -> Vec<u8> {
        // v0 tkhd: creation(4) mod(4) id(4) reserved(4) duration(4)
        // reserved(8) layer(2) alt(2) volume(2) reserved(2) matrix(36) w(4) h(4)
        let mut b = vec![0u8; 72];
        b.extend_from_slice(&(width << 16).to_be_bytes());
        b.extend_from_slice(&(height << 16).to_be_bytes());
        boxb(b"tkhd", &full(0, &b))
    }

    fn mdhd(timescale: u32, duration: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&duration.to_be_bytes());
        b.extend_from_slice(&[0u8; 4]); // language + pre_defined
        boxb(b"mdhd", &full(0, &b))
    }

    fn hdlr(handler: &[u8; 4]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&[0u8; 4]); // pre_defined
        b.extend_from_slice(handler);
        b.extend_from_slice(&[0u8; 12]); // reserved
        b.push(0); // empty name
        boxb(b"hdlr", &full(0, &b))
    }

    fn stsd(format: &[u8; 4]) -> Vec<u8> {
        // One minimal entry: size(4) + format(4) + 8 reserved bytes.
        let entry_len = 16u32;
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&entry_len.to_be_bytes());
        b.extend_from_slice(format);
        b.extend_from_slice(&[0u8; 8]);
        boxb(b"stsd", &full(0, &b))
    }

    fn stts(count: u32, delta: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&count.to_be_bytes());
        b.extend_from_slice(&delta.to_be_bytes());
        boxb(b"stts", &full(0, &b))
    }

    fn stsc(samples_per_chunk: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        b.extend_from_slice(&samples_per_chunk.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        boxb(b"stsc", &full(0, &b))
    }

    fn stsz(sizes: &[u32]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // variable sizes
        b.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
        for s in sizes {
            b.extend_from_slice(&s.to_be_bytes());
        }
        boxb(b"stsz", &full(0, &b))
    }

    fn stco(offset: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&offset.to_be_bytes());
        boxb(b"stco", &full(0, &b))
    }

    fn trak(
        handler: &[u8; 4],
        format: &[u8; 4],
        width: u32,
        height: u32,
        timescale: u32,
        duration: u32,
        n_samples: u32,
        delta: u32,
        sizes: &[u32],
        chunk_offset: u32,
    ) -> Vec<u8> {
        let stbl = boxb(
            b"stbl",
            &[
                stsd(format),
                stts(n_samples, delta),
                stsc(n_samples.max(1)),
                stsz(sizes),
                stco(chunk_offset),
            ]
            .concat(),
        );
        let minf = boxb(b"minf", &stbl);
        let mdia = boxb(
            b"mdia",
            &[mdhd(timescale, duration), hdlr(handler), minf].concat(),
        );
        boxb(b"trak", &[tkhd(width, height), mdia].concat())
    }

    /// Build a synthetic MP4: one 1920x1080 video trak (100 samples over
    /// 10 s → 10 fps) and, when `gpmd_payloads` is non-empty, one `gpmd`
    /// trak at 1 Hz whose payloads live in the mdat. `creation_1904_s` is
    /// the mvhd creation time (seconds since 1904; 0 = unset).
    pub fn synthetic_mp4(creation_1904_s: u64, gpmd_payloads: &[&[u8]]) -> Vec<u8> {
        let ftyp = boxb(b"ftyp", b"isom\x00\x00\x02\x00isomiso2");
        let mdat_data: Vec<u8> = gpmd_payloads.concat();
        let sizes: Vec<u32> = gpmd_payloads.iter().map(|p| p.len() as u32).collect();
        let n = gpmd_payloads.len() as u32;

        // Two passes: placeholder chunk offsets first (fixed-width u32), then
        // rebuild with the real mdat data offset.
        let build_moov = |gpmd_off: u32| -> Vec<u8> {
            let mut traks = trak(
                b"vide",
                b"avc1",
                1920,
                1080,
                1000,
                10_000,
                100,
                100,
                &vec![0u32; 100],
                gpmd_off,
            );
            if n > 0 {
                traks.extend_from_slice(&trak(
                    b"meta",
                    b"gpmd",
                    0,
                    0,
                    1000,
                    n * 1000,
                    n,
                    1000,
                    &sizes,
                    gpmd_off,
                ));
            }
            boxb(
                b"moov",
                &[mvhd(creation_1904_s, 1000, 10_000), traks].concat(),
            )
        };

        let moov_len = build_moov(0).len();
        let mdat_data_off = (ftyp.len() + moov_len + 8) as u32;
        let moov = build_moov(mdat_data_off);
        assert_eq!(moov.len(), moov_len, "placeholder pass must be size-stable");
        let mdat = boxb(b"mdat", &mdat_data);
        [ftyp, moov, mdat].concat()
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::synthetic_mp4;
    use super::*;

    #[test]
    fn read_info_synthetic_two_track_mp4_reports_dims_fps_duration_creation() {
        // Arrange — creation = 1970 epoch + 1000 s.
        let bytes = synthetic_mp4(MP4_EPOCH_TO_UNIX_S + 1_000, &[b"x"]);

        // Act
        let info = read_info(&bytes).unwrap();

        // Assert
        assert_eq!((info.width, info.height), (1920, 1080));
        assert!((info.fps - 10.0).abs() < 0.01);
        assert!((info.duration_s - 10.0).abs() < 1e-6);
        assert_eq!(info.creation_time_utc_ms, Some(1_000_000));
        assert!(info.has_gpmd);
    }

    #[test]
    fn read_info_zero_creation_time_maps_to_none() {
        // Arrange
        let bytes = synthetic_mp4(0, &[]);

        // Act
        let info = read_info(&bytes).unwrap();

        // Assert
        assert_eq!(info.creation_time_utc_ms, None);
        assert!(!info.has_gpmd);
    }

    #[test]
    fn read_gpmd_samples_three_payloads_roundtrip_times_and_bytes() {
        // Arrange
        let bytes = synthetic_mp4(0, &[b"aaaa", b"bb", b"cccccc"]);

        // Act
        let samples = read_gpmd_samples(&bytes).unwrap();

        // Assert
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].payload, b"aaaa");
        assert_eq!(samples[1].payload, b"bb");
        assert_eq!(samples[2].payload, b"cccccc");
        assert!((samples[0].t_video_s - 0.0).abs() < 1e-9);
        assert!((samples[1].t_video_s - 1.0).abs() < 1e-9, "1 Hz gpmd track");
        assert!((samples[2].t_video_s - 2.0).abs() < 1e-9);
    }

    #[test]
    fn read_gpmd_samples_without_gpmd_track_is_no_gpmf_error() {
        // Arrange
        let bytes = synthetic_mp4(0, &[]);

        // Act
        let err = read_gpmd_samples(&bytes).unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::NoGpmf);
    }

    #[test]
    fn read_info_truncated_garbage_is_parse_error_not_panic() {
        // Arrange + Act
        let err = read_info(&[0u8; 16]).unwrap_err();
        let err2 = read_info(b"").unwrap_err();

        // Assert
        assert_eq!(err.kind, VideoErrorKind::Parse);
        assert_eq!(err2.kind, VideoErrorKind::Parse);
    }
}
