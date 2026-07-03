//! Dev tool: per-IMU drop / backstep report for one or more `.idl0` sessions.
//!
//! Walks the raw IMU_SAMPLE timestamps (independent of reconciliation) and, on the
//! nominal grid (`period = 1_000_000 / ODR`), reports received samples, true drops
//! (forward gaps), drop rate, out-of-order backsteps, and the absolute-vs-counted
//! span drift that the per-step reconciler used to accumulate.
//!
//!   cargo run -q --example drop_report -- <session.idl0> [more.idl0 ...]

use std::convert::TryInto;

fn le_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn le_i64(b: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Returns (odr_hz, per-IMU timestamp vectors) for a schema-3 `.idl0` buffer.
fn read_imu_timestamps(b: &[u8]) -> Option<(u16, [Vec<i64>; 3])> {
    if b.len() < 48 || &b[0..4] != b"IDL0" || b[4] != 3 {
        return None;
    }
    // Header offsets: magic=0, schema=4, uuid=5, dev=21, start=27, crc=35,
    // mask=39, imu_count=43, imu_rate=44, gps_rate=46, reg_count=47, entries=48.
    let odr = le_u16(b, 44);
    let reg_count = b[47] as usize;
    let mut pos = 48 + reg_count * 40;
    if pos + 4 > b.len() {
        return None;
    }
    pos += 4; // 0xDEADBEEF marker

    let mut ts: [Vec<i64>; 3] = Default::default();
    while pos + 3 <= b.len() {
        let t = b[pos];
        let len = le_u16(b, pos + 1) as usize;
        pos += 3;
        if t == 0xFF {
            break;
        }
        if pos + len > b.len() {
            break; // truncated
        }
        if t == 0x01 && len >= 9 {
            let idx = b[pos] as usize;
            let tus = le_i64(b, pos + 1);
            if idx < 3 {
                ts[idx].push(tus);
            }
        }
        pos += len;
    }
    Some((odr, ts))
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: cargo run -q --example drop_report -- <session.idl0> [...]");
        std::process::exit(2);
    }
    for path in &paths {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                println!("=== {name} ===\n  read error: {e}\n");
                continue;
            }
        };
        let Some((odr, ts)) = read_imu_timestamps(&bytes) else {
            println!("=== {name} ===\n  not a schema-3 IDL0 log (skipped)\n");
            continue;
        };
        let period = if odr > 0 { 1_000_000 / odr as i64 } else { 10_000 };
        println!(
            "=== {name} ===  ODR={odr} Hz (period {period} µs)  {:.1} MB",
            bytes.len() as f64 / 1e6
        );
        for (imu, v) in ts.iter().enumerate() {
            if v.len() < 2 {
                if !v.is_empty() {
                    println!("  IMU{imu}: {} sample(s) — too few to analyse", v.len());
                }
                continue;
            }
            let first = v[0];
            let last = v[v.len() - 1];
            let dur_s = (last - first) as f64 / 1e6;
            // Absolute-grid placement: slot = round((ts - first)/period). A forward
            // jump of >1 slot is a real drop (one "gap site" of `gap` samples); a
            // slot that does not advance is an out-of-order backstep/duplicate that
            // the reconciler drops (a "collision").
            let mut fills = 0i64; // total synthesized (dropped) samples
            let mut collisions = 0i64;
            let mut backsteps = 0i64;
            let mut sites = 0i64; // distinct gap events
            let mut max_gap = 0i64;
            // gap-size histogram: [1] [2-3] [4-9] [10-49] [50+]
            let mut bucket = [0i64; 5];
            let mut prev_slot = 0i64;
            for k in 1..v.len() {
                if v[k] < v[k - 1] {
                    backsteps += 1;
                }
                let slot = (((v[k] - first) as f64) / period as f64).round() as i64;
                if slot <= prev_slot {
                    collisions += 1;
                } else {
                    let gap = slot - prev_slot - 1;
                    if gap > 0 {
                        fills += gap;
                        sites += 1;
                        max_gap = max_gap.max(gap);
                        let b = if gap == 1 {
                            0
                        } else if gap <= 3 {
                            1
                        } else if gap <= 9 {
                            2
                        } else if gap <= 49 {
                            3
                        } else {
                            4
                        };
                        bucket[b] += 1;
                    }
                    prev_slot = slot;
                }
            }
            let recv = v.len() as i64;
            let kept = recv - collisions;
            let nominal = kept + fills; // grid slots this IMU spans
            let drop_pct = 100.0 * fills as f64 / nominal as f64;
            let avg = if sites > 0 { fills as f64 / sites as f64 } else { 0.0 };
            println!(
                "  IMU{imu}: recv={recv:>7} kept={kept:>7} coll={collisions:>5} (back={backsteps})  fills={fills:>6} ({drop_pct:4.1}%)  sites={sites:>5} avg={avg:4.1} max={max_gap:>4}  span={dur_s:6.1}s"
            );
            println!(
                "         gaps: [1]={:<5} [2-3]={:<5} [4-9]={:<5} [10-49]={:<5} [50+]={}",
                bucket[0], bucket[1], bucket[2], bucket[3], bucket[4]
            );
        }
        println!();
    }
}
