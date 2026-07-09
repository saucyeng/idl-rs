//! Raw-device `.idl0` recovery (self-contained CLI utility).
//!
//! Salvages sessions whose filesystem metadata is truncated or orphaned — e.g.
//! a power loss during logging leaves the directory entry's file size frozen
//! while the sample data keeps landing in clusters on the card. Two modes:
//!
//! - [`run`]   — recover one session (the first, or a matching `--session-id`).
//! - [`scan_all`] — sweep the whole device for *every* `IDL0` session and list
//!   them; with an output dir, also write each recovered session out.
//!
//! Both read the source **read-only** and reconstruct sessions by walking the
//! record stream's own `type`/`len` framing — independent of the (stale)
//! filesystem size.
//!
//! Self-contained on purpose: it touches no engine code and depends only on
//! `std` plus the CLI's error envelope, so it is trivially removable once the
//! firmware commits metadata periodically (`fsync`) and this corruption mode
//! can no longer occur.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;

use crate::envelope::{CliError, ErrorKind as CliErrorKind};

/// `IDL0` magic at the start of every log header.
const MAGIC: &[u8; 4] = b"IDL0";
/// Little-endian `0xDEADBEEF` header-end marker that precedes the records.
const MARKER: u32 = 0xDEAD_BEEF;
/// Bytes from the magic to the `registry_count` byte, inclusive end — i.e. the
/// offset at which the registry entries begin (matches the v3 header layout).
const HEADER_PREFIX: usize = 48;
/// Size of one v3 channel-registry entry.
const REGISTRY_ENTRY: usize = 40;
/// Upper bound on a sane record payload; a larger declared length means we have
/// walked off the data into garbage.
const MAX_RECORD_PAYLOAD: usize = 4096;
/// Disk sector size; raw `\\.\PhysicalDrive` reads must be sector-aligned.
const SECTOR: u64 = 512;
/// Read size while scanning for a header.
const SCAN_CHUNK: usize = 8 * 1024 * 1024;
/// Read size while collecting a record stream.
const COLLECT_STEP: usize = 16 * 1024 * 1024;

/// Default cap on how far to scan for a header before giving up (16 GiB).
pub const DEFAULT_SCAN_LIMIT: u64 = 16 * 1024 * 1024 * 1024;
/// Default cap on how many bytes to recover from a header onward (512 MiB).
pub const DEFAULT_WINDOW: usize = 512 * 1024 * 1024;

/// A located, validated session region within a scanned buffer.
#[derive(Debug, Clone, PartialEq)]
struct RecoveredRegion {
    /// Length in bytes from the header that forms a valid header + record stream.
    len: usize,
    /// Number of data records walked (excludes the header and SESSION_END).
    records: usize,
    /// Lowercase-hex session UUID from the header.
    session_id: String,
    /// `true` if a SESSION_END (0xFF) record terminated the stream cleanly.
    clean_end: bool,
    /// `true` if the walk stopped only because it ran out of buffer mid-record
    /// (not at a definitive end). Signals the caller to read more if the device
    /// has more — otherwise a record straddling the read window looks like the
    /// end of the data.
    buffer_bound: bool,
}

/// Round `x` up to the next sector boundary.
fn align_up(x: u64) -> u64 {
    ((x + SECTOR - 1) / SECTOR) * SECTOR
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Find the next `IDL0` v3 header at or after `from`. When `want_session` is
/// set, only a header whose UUID matches (case-insensitive hex) is accepted —
/// which makes a false-positive magic in sample data astronomically unlikely.
fn find_header(buf: &[u8], from: usize, want_session: Option<&str>) -> Option<usize> {
    let want = want_session.map(|s| s.trim().to_ascii_lowercase());
    let mut i = from;
    while i + HEADER_PREFIX <= buf.len() {
        if &buf[i..i + 4] == MAGIC && buf[i + 4] == 3 {
            match &want {
                Some(w) => {
                    if i + 21 <= buf.len() && &hex(&buf[i + 5..i + 21]) == w {
                        return Some(i);
                    }
                }
                None => return Some(i),
            }
        }
        i += 1;
    }
    None
}

/// Walk the header + record stream at `buf[start..]`. Returns `None` if `start`
/// is not a valid v3 header (bad magic/schema/marker, or the buffer is too
/// short to reach the marker). Otherwise walks records by their `type`/`len`
/// framing, stopping at the first byte that cannot be a valid record (unknown
/// type, zero/implausible length, or a record that runs past the buffer) or at
/// SESSION_END.
fn recover_region(buf: &[u8], start: usize) -> Option<RecoveredRegion> {
    let h = buf.get(start..)?;
    if h.len() < HEADER_PREFIX || &h[0..4] != MAGIC || h[4] != 3 {
        return None;
    }
    let rc = h[47] as usize;
    let marker_off = HEADER_PREFIX + rc * REGISTRY_ENTRY;
    if h.len() < marker_off + 4 {
        return None; // buffer too short to confirm the header
    }
    let marker = u32::from_le_bytes([h[marker_off], h[marker_off + 1], h[marker_off + 2], h[marker_off + 3]]);
    if marker != MARKER {
        return None;
    }
    let session_id = hex(&h[5..21]);

    let mut pos = marker_off + 4; // first record
    let mut records = 0usize;
    let mut clean_end = false;
    let mut buffer_bound = false;
    loop {
        if pos + 3 > h.len() {
            buffer_bound = true; // no room for another header — more may follow
            break;
        }
        let type_ = h[pos];
        let len = u16::from_le_bytes([h[pos + 1], h[pos + 2]]) as usize;
        if type_ == 0xFF {
            clean_end = true;
            pos += 3; // SESSION_END carries no payload
            break;
        }
        if !matches!(type_, 0x01 | 0x02 | 0x03) || len == 0 || len > MAX_RECORD_PAYLOAD {
            break; // definitive end: unknown type / implausible length (garbage/zeros)
        }
        if pos + 3 + len > h.len() {
            buffer_bound = true; // real record, but it runs past what we've read
            break;
        }
        pos += 3 + len;
        records += 1;
    }

    Some(RecoveredRegion { len: pos, records, session_id, clean_end, buffer_bound })
}

/// Read `buf.len()` bytes from `off` into `buf`, returning how many were read
/// (short only at end-of-device/file). `off` and `buf.len()` should be
/// sector-aligned for raw `\\.\PhysicalDrive` access.
fn read_block(f: &mut File, off: u64, buf: &mut [u8]) -> Result<usize, String> {
    f.seek(SeekFrom::Start(off)).map_err(|e| format!("seek to {off} failed: {e}"))?;
    let mut total = 0;
    while total < buf.len() {
        match f.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("read at offset {} failed: {e}", off + total as u64)),
        }
    }
    Ok(total)
}

/// Find the next `IDL0` header at or after `from` (sector-aligned), carrying a
/// small tail between chunks so a header straddling a chunk boundary is still
/// found. `Ok(None)` when none is found before `scan_limit` / end of source.
fn find_next_header(
    f: &mut File,
    from: u64,
    want_session: Option<&str>,
    scan_limit: u64,
) -> Result<Option<u64>, String> {
    let mut chunk = vec![0u8; SCAN_CHUNK];
    let mut carry: Vec<u8> = Vec::new();
    let mut file_off = from - (from % SECTOR);
    let report_step: u64 = 2 << 30; // progress line every 2 GiB scanned
    let mut next_report = (file_off / report_step + 1) * report_step;
    loop {
        if file_off >= scan_limit {
            return Ok(None);
        }
        if file_off >= next_report {
            eprintln!("  …scanned {} GiB", file_off >> 30);
            next_report += report_step;
        }
        let n = read_block(f, file_off, &mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        let hay_base = file_off - carry.len() as u64;
        let mut hay = Vec::with_capacity(carry.len() + n);
        hay.extend_from_slice(&carry);
        hay.extend_from_slice(&chunk[..n]);
        if let Some(rel) = find_header(&hay, 0, want_session) {
            let abs = hay_base + rel as u64;
            if abs >= from {
                return Ok(Some(abs));
            }
        }
        let keep = hay.len().min(HEADER_PREFIX - 1);
        carry = hay[hay.len() - keep..].to_vec();
        file_off += n as u64;
        if n < chunk.len() {
            return Ok(None);
        }
    }
}

/// From a located header offset, read forward (sector-aligned) and validate the
/// record stream, stopping as soon as the data ends or `window_max` is reached.
/// `Ok(Some(...))` with the region, its buffer, and the header's local index;
/// `Ok(None)` if the magic did not validate as a header (false positive).
fn collect_region(
    f: &mut File,
    header_abs: u64,
    window_max: usize,
) -> Result<Option<(RecoveredRegion, Vec<u8>, usize)>, String> {
    let read_base = header_abs - (header_abs % SECTOR);
    let local = (header_abs - read_base) as usize;
    let mut buf: Vec<u8> = Vec::new();
    let mut off = read_base;
    let mut block = vec![0u8; COLLECT_STEP];

    loop {
        let n = read_block(f, off, &mut block)?;
        if n == 0 {
            break; // end of source
        }
        buf.extend_from_slice(&block[..n]);
        off += n as u64;
        let device_more = n == block.len(); // a full read suggests more follows

        match recover_region(&buf, local) {
            Some(region) => {
                // Stop when the stream ended cleanly, or at a definitive
                // non-record byte within the buffer. Only keep reading when the
                // walk stopped solely because a record ran past the read window
                // (buffer_bound) and the device still has more to give.
                if region.clean_end || !region.buffer_bound {
                    return Ok(Some((region, buf, local)));
                }
                if !device_more || buf.len().saturating_sub(local) >= window_max {
                    return Ok(Some((region, buf, local)));
                }
            }
            None => {
                if buf.len() >= COLLECT_STEP || !device_more {
                    return Ok(None); // false-positive magic, or ran out before a valid header
                }
            }
        }
    }

    Ok(recover_region(&buf, local).map(|r| (r, buf, local)))
}

/// Recover a single session from `device` (a raw `\\.\PhysicalDrive`, an image,
/// or an `.idl0` file) to `output`. Read-only on the source. `want_session`
/// restricts to a matching session UUID; `scan_limit`/`window_max` bound work.
pub fn run(
    device: &Path,
    output: &Path,
    want_session: Option<&str>,
    scan_limit: u64,
    window_max: usize,
    from: u64,
) -> Result<(), CliError> {
    let mut f = open_ro(device).map_err(CliError::io)?;
    eprintln!("opened {} (read-only)", device.display());

    let header_abs = find_next_header(&mut f, from, want_session, scan_limit)
        .map_err(CliError::io)?
        .ok_or_else(|| CliError::new(CliErrorKind::NotFound, "no matching IDL0 header found"))?;
    eprintln!("found IDL0 header at byte offset {header_abs}");

    let (region, buf, local) = collect_region(&mut f, header_abs, window_max)
        .map_err(CliError::io)?
        .ok_or_else(|| {
            CliError::new(
                CliErrorKind::InvalidInput,
                "located magic is not a valid IDL0 header (try --session-id)",
            )
        })?;
    let data = &buf[local..local + region.len];
    fs::write(output, data)
        .map_err(|e| CliError::io(format!("cannot write {}: {e}", output.display())))?;

    eprintln!(
        "recovered {} bytes, {} records, session {}{}",
        region.len,
        region.records,
        region.session_id,
        end_note(region.clean_end)
    );
    eprintln!("wrote {}", output.display());
    eprintln!(
        "next: idl-rs info \"{out}\"  /  idl-rs channels \"{out}\"  /  idl-rs fit \"{out}\"",
        out = output.display()
    );
    Ok(())
}

/// Sweep the whole device/image for every `IDL0` session, list them, and (when
/// `out_dir` is set) write each recovered session to `<session>_<offset>.idl0`.
/// Read-only on the source.
pub fn scan_all(device: &Path, out_dir: Option<&Path>, scan_limit: u64) -> Result<(), CliError> {
    let mut f = open_ro(device).map_err(CliError::io)?;
    eprintln!(
        "scanning {} for IDL0 sessions (read-only) — reads the device sequentially; this can take a while on a large card...",
        device.display()
    );
    let found = scan_regions(&mut f, scan_limit).map_err(CliError::io)?;

    if found.is_empty() {
        eprintln!("no IDL0 sessions found");
        return Ok(());
    }

    println!("{:<34} {:>16} {:>13} {:>12} {:>6}", "SESSION", "OFFSET", "BYTES", "RECORDS", "END");
    for (abs, region, _) in &found {
        println!(
            "{:<34} {:>16} {:>13} {:>12} {:>6}",
            region.session_id,
            abs,
            region.len,
            region.records,
            if region.clean_end { "clean" } else { "cut" }
        );
    }

    match out_dir {
        Some(dir) => {
            fs::create_dir_all(dir)
                .map_err(|e| CliError::io(format!("cannot create {}: {e}", dir.display())))?;
            for (abs, region, data) in &found {
                let path = dir.join(format!("{}_{}.idl0", region.session_id, abs));
                fs::write(&path, data)
                    .map_err(|e| CliError::io(format!("cannot write {}: {e}", path.display())))?;
            }
            eprintln!("wrote {} session(s) to {}", found.len(), dir.display());
        }
        None => eprintln!(
            "{} session(s) found. Re-run with --out-dir <DIR> to write them all, or `recover --session-id <ID>` for one.",
            found.len()
        ),
    }
    Ok(())
}

/// Walk the whole source, returning every recoverable session as
/// `(offset, region, bytes)`. Skips false-positive magics; advances past each
/// recovered region so records are never re-scanned as headers.
fn scan_regions(f: &mut File, scan_limit: u64) -> Result<Vec<(u64, RecoveredRegion, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut pos = 0u64;
    loop {
        match find_next_header(f, pos, None, scan_limit)? {
            None => break,
            Some(abs) => match collect_region(f, abs, DEFAULT_WINDOW)? {
                Some((region, buf, local)) => {
                    let data = buf[local..local + region.len].to_vec();
                    let next = align_up(abs + region.len as u64).max(abs + SECTOR);
                    out.push((abs, region, data));
                    pos = next;
                }
                None => pos = abs + SECTOR, // false positive; step past the magic
            },
        }
    }
    Ok(out)
}

fn open_ro(device: &Path) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .open(device)
        .map_err(|e| format!("cannot open {} read-only: {e}", device.display()))
}

fn end_note(clean: bool) -> &'static str {
    if clean {
        " (clean SESSION_END)"
    } else {
        " (no SESSION_END — recovered up to the last intact record)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Build a minimal valid v3 header with `rc` zeroed registry entries, the
    /// DEADBEEF marker, then the given record bytes.
    fn build(uuid: [u8; 16], rc: u8, records: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.push(3); // schema
        b.extend_from_slice(&uuid); // 5..21
        b.extend_from_slice(&[0u8; 6]); // device_id 21..27
        b.extend_from_slice(&0i64.to_le_bytes()); // start_ms 27..35
        b.extend_from_slice(&0u32.to_le_bytes()); // crc 35..39
        b.extend_from_slice(&0u32.to_le_bytes()); // imu_mask 39..43
        b.push(0); // imu_count 43
        b.extend_from_slice(&0u16.to_le_bytes()); // imu_rate 44..46
        b.push(0); // gps_rate 46
        b.push(rc); // registry_count 47
        for _ in 0..rc {
            b.extend_from_slice(&[0u8; REGISTRY_ENTRY]);
        }
        b.extend_from_slice(&MARKER.to_le_bytes());
        b.extend_from_slice(records);
        b
    }

    /// One framed record: type + u16 little-endian length + payload.
    fn rec(type_: u8, payload: &[u8]) -> Vec<u8> {
        let mut r = vec![type_];
        r.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        r.extend_from_slice(payload);
        r
    }

    const UUID: [u8; 16] = [
        0x33, 0x3e, 0xf6, 0x08, 0xcb, 0x9c, 0xee, 0x47, 0x88, 0x2b, 0x48, 0x00, 0x0a, 0x38, 0xf2, 0x47,
    ];
    const UUID_HEX: &str = "333ef608cb9cee47882b48000a38f247";

    #[test]
    fn recover_region_walks_records_and_detects_clean_end() {
        // Arrange — two IMU-ish records (4-axis: payload 17) then SESSION_END.
        let mut records = Vec::new();
        records.extend(rec(0x01, &[0u8; 17]));
        records.extend(rec(0x01, &[0u8; 17]));
        records.extend(rec(0xFF, &[]));
        let buf = build(UUID, 6, &records);

        // Act
        let r = recover_region(&buf, 0).unwrap();

        // Assert — both records counted, clean end, length spans the whole buffer.
        assert_eq!(r.records, 2);
        assert!(r.clean_end);
        assert!(!r.buffer_bound);
        assert_eq!(r.len, buf.len());
        assert_eq!(r.session_id, UUID_HEX);
    }

    #[test]
    fn recover_region_flags_buffer_bound_when_last_record_is_cut() {
        // Arrange — one complete record, then a record header claiming a 17-byte
        // payload but with only 5 bytes present (a read-window cut).
        let mut records = Vec::new();
        records.extend(rec(0x01, &[0u8; 17])); // complete
        records.push(0x01); // next record type...
        records.extend(&(17u16).to_le_bytes()); // ...claims 17 bytes...
        records.extend(vec![0u8; 5]); // ...but only 5 are here
        let buf = build(UUID, 0, &records);

        // Act
        let r = recover_region(&buf, 0).unwrap();

        // Assert — first record counted; stopped because the 2nd ran past the buffer.
        assert_eq!(r.records, 1);
        assert!(!r.clean_end);
        assert!(r.buffer_bound);
    }

    #[test]
    fn recover_region_stops_at_garbage_without_clean_end() {
        // Arrange — one valid record, then a zeroed "cluster" (type 0x00 = end).
        let mut records = Vec::new();
        records.extend(rec(0x02, &[0u8; 32])); // GPS-sized record
        let valid_len = records.len();
        records.extend(vec![0u8; 64]); // garbage / erased space
        let buf = build(UUID, 0, &records);
        let header_len = buf.len() - records.len();

        // Act
        let r = recover_region(&buf, 0).unwrap();

        // Assert — stops at the zeros; definitive end (not buffer-bound).
        assert_eq!(r.records, 1);
        assert!(!r.clean_end);
        assert!(!r.buffer_bound);
        assert_eq!(r.len, header_len + valid_len);
    }

    #[test]
    fn recover_region_rejects_non_header() {
        // Arrange — wrong magic.
        let buf = b"NOPEnot a header............................................".to_vec();

        // Act + Assert
        assert!(recover_region(&buf, 0).is_none());
    }

    #[test]
    fn find_header_locates_magic_at_offset_with_session_match() {
        // Arrange — 100 bytes of filler, then a header.
        let mut buf = vec![0xABu8; 100];
        buf.extend(build(UUID, 0, &rec(0xFF, &[])));

        // Act
        let off = find_header(&buf, 0, Some(UUID_HEX)).unwrap();

        // Assert
        assert_eq!(off, 100);
    }

    #[test]
    fn find_header_skips_wrong_session_id() {
        // Arrange — a real header, but we ask for a different session.
        let buf = build(UUID, 0, &rec(0xFF, &[]));

        // Act + Assert — no match for a different UUID; matches when unconstrained.
        assert!(find_header(&buf, 0, Some("00000000000000000000000000000000")).is_none());
        assert_eq!(find_header(&buf, 0, None), Some(0));
    }

    #[test]
    fn scan_regions_finds_multiple_sessions_on_a_device() {
        // Arrange — zeros, session A (clean), zeros, session B (cut), trailing zeros.
        let a = build(UUID, 0, &{
            let mut r = Vec::new();
            r.extend(rec(0x01, &[0u8; 17]));
            r.extend(rec(0xFF, &[]));
            r
        });
        let uuid_b = {
            let mut u = UUID;
            u[0] = 0xAA;
            u
        };
        let b = build(uuid_b, 0, &rec(0x02, &[0u8; 32])); // no SESSION_END

        let mut dev = Vec::new();
        dev.extend(vec![0u8; 600]);
        let a_off = dev.len();
        dev.extend(&a);
        dev.extend(vec![0u8; 1000]);
        let b_off = dev.len();
        dev.extend(&b);
        dev.extend(vec![0u8; 500]);

        let path = std::env::temp_dir().join("idlrs_recover_scan_test.bin");
        File::create(&path).unwrap().write_all(&dev).unwrap();
        let mut f = OpenOptions::new().read(true).open(&path).unwrap();

        // Act
        let found = scan_regions(&mut f, u64::MAX).unwrap();

        // Assert — both sessions found, in order, at their offsets.
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].0, a_off as u64);
        assert_eq!(found[0].1.session_id, UUID_HEX);
        assert!(found[0].1.clean_end);
        assert_eq!(found[1].0, b_off as u64);
        assert_eq!(found[1].1.session_id, hex(&uuid_b));
        assert!(!found[1].1.clean_end);

        let _ = fs::remove_file(&path);
    }
}
