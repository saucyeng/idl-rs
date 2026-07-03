//! Dev tool: summarise the firmware on-SD diagnostic log (`idl0_debug.log`).
//!
//! Parses the per-session IMU drain stats the firmware flushes at session end
//! (`imu_task.c::idl0_imu_diag_flush`) plus boot markers and heap samples, and
//! tabulates where the IMU poll cycle's time goes — the key being `ovr` (FIFO
//! overruns: >0 ⇒ drops are real overflow / throughput; 0 ⇒ drops are upstream
//! of the FIFO, e.g. pairing/desync) and `read_us` (per-IMU bus cost).
//!
//!   cargo run -q --example diag_report -- <idl0_debug.log>
//!
//! Line formats (diag_log.c):
//!   t=<s> EVT IMU<i> <spi|i2c> drains=.. pairs=.. ovr=.. read_us=avg/max tot_us=avg/max
//!   t=<s> EVT IMU cycle_max_us=.. writer_drops=.. poll_ms=.. cap=..
//!   t=<s> heap=.. min=.. frag=.. wifi=.. ble_susp=.. sd=..
//!   ==== BOOT reason=<R> heap=.. ====

use std::collections::BTreeMap;

/// Value for `key` in a whitespace-tokenised `key=value` line, if present.
fn kv<'a>(tokens: &[&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).filter(|r| r.starts_with('=')).map(|r| &r[1..]))
}

/// Splits an `avg/max` field into its two parts.
fn avg_max(s: &str) -> (&str, &str) {
    s.split_once('/').unwrap_or((s, "?"))
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: cargo run -q --example diag_report -- <idl0_debug.log>");
            std::process::exit(2);
        }
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read error: {e}");
            std::process::exit(1);
        }
    };

    let mut boots: BTreeMap<String, u32> = BTreeMap::new();
    let mut imu_lines = 0u64;
    let mut last_brownout_t: Option<String> = None;
    let mut sessions: Vec<String> = Vec::new(); // formatted blocks, in file order

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("==== BOOT") {
            let toks: Vec<&str> = line.split_whitespace().collect();
            let reason = kv(&toks, "reason").unwrap_or("?").to_string();
            *boots.entry(reason).or_insert(0) += 1;
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        let t = toks.iter().find_map(|x| x.strip_prefix("t=")).unwrap_or("?");
        // Per-IMU drain line: "... EVT IMU<i> <bus> drains=.. ovr=.. read_us=a/m tot_us=a/m"
        if line.contains("EVT IMU") && line.contains("drains=") {
            imu_lines += 1;
            // IMU index + bus are the two tokens after "EVT".
            let evt = toks.iter().position(|&x| x == "EVT").unwrap_or(0);
            let imu = toks.get(evt + 1).copied().unwrap_or("IMU?");
            let bus = toks.get(evt + 2).copied().unwrap_or("?");
            let ovr = kv(&toks, "ovr").unwrap_or("?");
            let pairs = kv(&toks, "pairs").unwrap_or("?");
            let drains = kv(&toks, "drains").unwrap_or("?");
            // reuse/unk added with the lossless-pairing fix; "-" on older logs.
            let reuse = kv(&toks, "reuse").unwrap_or("-");
            let unk = kv(&toks, "unk").unwrap_or("-");
            let (read_a, read_m) = avg_max(kv(&toks, "read_us").unwrap_or("?/?"));
            let (tot_a, tot_m) = avg_max(kv(&toks, "tot_us").unwrap_or("?/?"));
            let flag = if ovr != "0" && ovr != "?" {
                "  <-- OVERRUNS (FIFO overflow)"
            } else if unk != "0" && unk != "-" && unk != "?" {
                "  <-- UNKNOWN TAGS (bus corruption)"
            } else {
                ""
            };
            sessions.push(format!(
                "  t={t:>6} {imu} {bus:<3} drains={drains:<6} pairs={pairs:<9} ovr={ovr:<4} reuse={reuse:<8} unk={unk:<5} read_us={read_a:>4}/{read_m:<5} tot_us={tot_a:>4}/{tot_m:<5}{flag}"
            ));
        } else if line.contains("EVT IMU cycle_max_us=") {
            let cyc = kv(&toks, "cycle_max_us").unwrap_or("?");
            let wd = kv(&toks, "writer_drops").unwrap_or("?");
            let poll = kv(&toks, "poll_ms").unwrap_or("?");
            let cap = kv(&toks, "cap").unwrap_or("?");
            let flag = if wd != "0" && wd != "?" { "  <-- WRITER DROPS" } else { "" };
            sessions.push(format!(
                "  t={t:>6} cycle_max_us={cyc} writer_drops={wd} poll_ms={poll} cap={cap}{flag}\n"
            ));
        } else if line.contains("BROWNOUT") {
            last_brownout_t = Some(t.to_string());
        }
    }

    println!("=== {} ===", path.rsplit(['/', '\\']).next().unwrap_or(&path));
    print!("boots: ");
    for (r, n) in &boots {
        print!("{r}×{n} ");
    }
    println!();
    let brownouts = boots.get("BROWNOUT").copied().unwrap_or(0);
    if brownouts > 0 {
        println!(
            "  ⚠ {brownouts} BROWNOUT reset(s){} — battery/supply sag can stall the SD/bus and cause drops",
            last_brownout_t.map(|t| format!(" (last at t={t})")).unwrap_or_default()
        );
    }
    if imu_lines == 0 {
        println!("  (no per-session IMU drain stats found — session may not have closed cleanly)");
    } else {
        println!("per-session IMU drain stats (read ovr first: >0 = real FIFO overflow):");
        for s in &sessions {
            println!("{s}");
        }
    }
}
