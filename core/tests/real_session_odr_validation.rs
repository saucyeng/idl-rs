//! C1 §8 item 8: validates the burst-seam correction (C1 §3.3) against a
//! real `.idl0` session with GPS and IMU both enabled, cross-checking the
//! corrected effective ODR against an independent estimate (GPS-anchored
//! recording duration ÷ IMU sample count).
//!
//! **Not a checked-in fixture.** The input file is real session data
//! Isaac supplied for this one validation, gitignored (see this plan's
//! Task 1) rather than added as a test fixture — idl-rs has no
//! fixture-directory convention and real session data isn't source-
//! controlled fixture material. The file lives at the idl1-app repo root,
//! never copied into this worktree; its path is read from the
//! `IDL_RS_REAL_SESSION_IDL0` environment variable (absolute path) rather
//! than hard-coded, so this test is a no-op (prints a skip notice naming
//! the variable, exits success) on any machine where the variable is
//! unset or does not point at a real file, and never blocks `cargo test`
//! for anyone but the machine that has it.

#[test]
fn real_session_burst_seam_correction_matches_an_independent_odr_estimate() {
    let path = match std::env::var("IDL_RS_REAL_SESSION_IDL0") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => {
            eprintln!(
                "skipping: IDL_RS_REAL_SESSION_IDL0 not set (not a checked-in fixture, see this test's module doc)"
            );
            return;
        }
    };
    if !path.is_file() {
        eprintln!(
            "skipping: IDL_RS_REAL_SESSION_IDL0={} does not point at a file (not a checked-in fixture, see this test's module doc)",
            path.display()
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read the real session file");
    let result = idl_rs::parse::parse(&bytes).expect("parse the real .idl0 file");
    let session = result.session;

    // Contract C1 §8 item 8's precondition: both GPS and IMU enabled.
    let has_gps = session.channels.iter().any(|c| c.channel_id.starts_with("GPS") && !c.is_empty());
    let has_imu = session
        .channels
        .iter()
        .any(|c| idl_rs::parse::records::imu_index_of(&c.channel_id).is_some() && !c.is_empty());

    if !has_gps || !has_imu {
        eprintln!(
            "SKIP + OPEN QUESTION: real session at {} does not have both GPS (present={has_gps}) \
             and IMU (present={has_imu}) enabled — C1 §8 item 8's cross-check needs both. \
             Logged as an open question for Isaac rather than failing the build (see this plan's \
             Open questions and the task brief's explicit instruction not to block L1 on this).",
            path.display()
        );
        return;
    }

    // Independent estimate's wall-clock anchor: the GPS fix's own epoch-ms
    // channel. If no GPS channel carries wall-clock time, this validation's
    // design (GPS-anchored duration ÷ IMU count) cannot run at all — that is
    // a real finding, not a skip.
    let gps = session
        .channels
        .iter()
        .find(|c| c.channel_id == "GPS_EpochMs" && !c.is_empty())
        .unwrap_or_else(|| {
            panic!(
                "STOP: no GPS_EpochMs (wall-clock) channel found among this session's GPS_* \
                 channels at {} — the independent ODR estimate needs a GPS wall-clock column; \
                 this changes the validation design, report to the lead instead of guessing",
                path.display()
            )
        });

    let imu0 = session
        .channels
        .iter()
        .find(|c| c.source_kind == "imu0")
        .expect("an imu0 channel exists (has_imu checked above)");

    // GPS wall-clock span (s): last epoch-ms fix minus first, in seconds.
    let gps_values = gps.materialize();
    let gps_first_ms = gps_values[0];
    let gps_last_ms = *gps_values.last().unwrap();
    let gps_span_s = (gps_last_ms - gps_first_ms) / 1000.0;

    // Independent estimate: only imu0 samples inside the GPS fix window
    // (the GPS channel's own `t_us` bounds) — imu0 typically records before
    // the first fix and after the last, so a whole-file count would bias
    // this estimate low by exactly that margin (R19 item 3).
    let gps_t_us_first = gps.t_us[0];
    let gps_t_us_last = *gps.t_us.last().unwrap();
    let count_in_window =
        imu0.t_us.iter().filter(|&&t| t >= gps_t_us_first && t <= gps_t_us_last).count();
    let independent_hz = count_in_window as f64 / gps_span_s;

    // Corrected ODR, measured directly from what §3.3 actually produced —
    // imu0's own corrected `t_us` span — rather than trusting that Task 6
    // rewrote `nominal_rate_hz` correctly (R19 item 2). `nominal_rate_hz`
    // is printed alongside for reference only, never asserted on.
    let imu0_n = imu0.t_us.len();
    let corrected_hz = (imu0_n - 1) as f64 / ((imu0.t_us[imu0_n - 1] - imu0.t_us[0]) as f64 / 1e6);
    let nominal_hz = imu0.nominal_rate_hz;

    let relative_error = ((corrected_hz - independent_hz) / independent_hz).abs();
    println!(
        "real-session ODR validation: corrected={corrected_hz:.3} Hz, nominal={nominal_hz:.3} Hz, \
         independent={independent_hz:.3} Hz, relative_error={:.4}%, imu0 n={imu0_n}, \
         count_in_window={count_in_window}, gps_span_s={gps_span_s:.3}",
        relative_error * 100.0
    );

    // A generous tolerance (5%) — GPS-duration ÷ IMU-count is itself a
    // coarse estimate (whole-file span, no sub-sample precision at either
    // end), not a second ground-truth clock; this is a sanity cross-check,
    // not a bit-exact assertion the way the synthetic worked example
    // (Task 5) is. Isaac reviews the printed numbers regardless of pass/fail.
    assert!(
        relative_error < 0.05,
        "corrected ODR {corrected_hz:.3} Hz disagrees with the independent GPS-based estimate \
         {independent_hz:.3} Hz by {:.2}% — investigate before trusting §3.3 on this file",
        relative_error * 100.0
    );
}
