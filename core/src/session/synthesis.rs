//! Synthesizes the `Time` and `Distance` base channels appended to a parsed
//! session, mirroring the Dart `channelDataProvider` synthesis
//! (app/lib/providers/channel_provider.dart §57-176). One source of truth so
//! the app and the CLI derive identical base channels.

use crate::integration::integrate;
use crate::session::{Channel, RawColumn, Session};

/// What [`synthesize_base_channels`] adds to a session, decided from each
/// channel's id, nominal rate (Hz) and sample count alone — so a reader that
/// has only a file's metadata (`idl_library::catalog_read::get_session`) lists the
/// same base channels the engine will build, without decoding anything.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasePlan {
    /// Index (into the dims given) of the channel whose real `t_us` `Time`
    /// carries.
    pub time_source: usize,
    /// `Time`'s (and `Distance`'s) nominal rate, Hz; `0.0` when no channel
    /// is fixed-rate (never a fabricated rate, ledger R23 Q2).
    pub nominal_rate_hz: f64,
    /// `Time`'s (and `Distance`'s) sample count.
    pub len: usize,
    /// Whether `Distance` is added (a non-empty fixed-rate `GPS_SpeedKmh`).
    pub distance: bool,
}

/// The [`BasePlan`] for channels given as `(channel_id, nominal_rate_hz,
/// sample_count)`, or `None` when there are none or every one is empty.
/// `Time` follows the highest fixed-rate channel (the longest at that rate);
/// with no fixed-rate channel, the channel with the most samples.
pub fn plan_base_channels(dims: &[(&str, f64, usize)]) -> Option<BasePlan> {
    let mut best: Option<(usize, f64, usize)> = None; // (index, rate, len)
    for (i, &(_, rate, len)) in dims.iter().enumerate() {
        if rate <= 0.0 {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, r, l)) => rate > r || (rate == r && len > l),
        };
        if better {
            best = Some((i, rate, len));
        }
    }
    // No fixed-rate channel — fall back to the channel with the most
    // samples, using its own real `t_us` (ledger R23 Q2). This is what
    // makes every FIT/GPX/CSV session (every channel `nominal_rate_hz:
    // 0.0`) actually gain a `Time` channel; the rate stays `0.0` here,
    // never a fabricated rate — `channel_kind` is `event` iff
    // `nominal_rate_hz == 0.0` (`store/parquet.rs:158`), and an honest
    // `Time` channel for event-driven data must itself read `event`.
    if best.is_none() {
        for (i, &(_, _, len)) in dims.iter().enumerate() {
            if len > best.map_or(0, |(_, _, l)| l) {
                best = Some((i, 0.0, len));
            }
        }
    }
    let (time_source, nominal_rate_hz, len) = best?;
    if len == 0 {
        return None;
    }
    let distance = dims.iter().any(|&(id, rate, n)| id == "GPS_SpeedKmh" && rate > 0.0 && n > 0);
    Some(BasePlan { time_source, nominal_rate_hz, len, distance })
}

/// Appends a synthesized `Time` channel (and, when `GPS_SpeedKmh` is present, a
/// cumulative `Distance` channel) to `session`. Returns the ids it appended
/// (`["Time"]`, `["Time", "Distance"]`, or `[]`).
///
/// `Time`: built at the highest fixed-rate channel's rate, length = the longest
/// sample count at that rate; `samples[i] = t_us[i] / 1e6`, where `t_us` is
/// **that winning channel's own real `t_us`** — not `i / rate` (contract C1
/// §3.5 invariant 4 forbids deriving `Time`'s values from the nominal rate).
/// This is a **deliberate, documented departure from the pre-idl1
/// zero-storage `Ramp` representation**: `Ramp`'s `value(i) = i / rate`
/// formula is exactly the derivation invariant 4 forbids once a channel's
/// `t_us` is not perfectly uniform (true after Task 6's burst correction, or
/// for any channel with drops), so `Time` costs 8 B/sample again under idl1
/// (`RawColumn::F64`, not `Ramp`).
///
/// When no channel has a positive `nominal_rate_hz` (every FIT/GPX/CSV
/// channel today), falls back to the channel with the most samples, still
/// using its own real `t_us` — the synthesized `Time` channel's
/// `nominal_rate_hz` is `0.0` in this branch, never a fabricated rate
/// (ledger R23 Q2): declaring a fake rate would make `channel_kind` lie
/// about an irregular source (`channel_kind` is `event` iff
/// `nominal_rate_hz == 0.0`, `store/parquet.rs:158`). Omitted (returns
/// `[]`) only when `session.channels` is empty or every channel is empty.
///
/// `Distance` (metres): trapezoidal-integrate `GPS_SpeedKmh / 3.6` (km/h → m/s)
/// at the GPS rate, then linear-interpolate onto the `Time` grid, clamped at
/// both ends; presented on `Time`'s own `t_us` (they share one time axis by
/// construction). Omitted when `GPS_SpeedKmh` is absent/empty.
pub fn synthesize_base_channels(session: &mut Session) -> Vec<String> {
    let dims: Vec<(&str, f64, usize)> =
        session.channels.iter().map(|c| (c.channel_id.as_str(), c.nominal_rate_hz, c.len())).collect();
    let Some(plan) = plan_base_channels(&dims) else {
        return Vec::new();
    };
    let (time_source_idx, max_rate, max_rate_len) = (plan.time_source, plan.nominal_rate_hz, plan.len);

    let mut added = vec!["Time".to_string()];

    // Distance base from GPS_SpeedKmh, if present and usable.
    let distance = synthesize_distance_base(&session.channels);

    // The winning channel's own real t_us — cloned once, out from under the
    // borrow of session.channels, so it can feed both the Time push below
    // and (if present) Distance's push.
    let time_t_us = session.channels[time_source_idx].t_us.clone();
    let time_values: Vec<f64> = time_t_us.iter().map(|&t| t as f64 / 1_000_000.0).collect();

    session.channels.push(Channel {
        channel_id: "Time".to_string(),
        t_us: time_t_us.clone(),
        t_recorded_us: None,
        nominal_rate_hz: max_rate,
        column: RawColumn::F64(time_values),
        source_kind: "synthesized".to_string(),
        unit: "s".to_string(),
        gaps: Vec::new(),
    });
    if let Some((base, base_rate)) = distance {
        // Distance stores only the GPS-rate metres; presentation on the Time
        // grid is lazy (RawColumn::Interp reproduces the former eager
        // clamp-lerp bit-for-bit).
        session.channels.push(Channel {
            channel_id: "Distance".to_string(),
            t_us: time_t_us,
            t_recorded_us: None,
            nominal_rate_hz: max_rate,
            column: RawColumn::Interp {
                base,
                base_rate,
                out_rate: max_rate,
                len: max_rate_len,
            },
            source_kind: "synthesized".to_string(),
            unit: "m".to_string(),
            gaps: Vec::new(),
        });
        added.push("Distance".to_string());
    }
    added
}

/// Cumulative distance (metres) at the GPS rate plus that rate, or `None` when
/// `GPS_SpeedKmh` is absent/empty. Trapezoidal-integrates speed/3.6 (km/h →
/// m/s) via [`integrate`]; interpolation onto the Time grid is lazy
/// ([`RawColumn::Interp`]). Mirrors Dart `_synthesiseDistance` step 1.
fn synthesize_distance_base(channels: &[Channel]) -> Option<(Vec<f64>, f64)> {
    let speed = channels.iter().find(|c| {
        c.channel_id == "GPS_SpeedKmh" && c.nominal_rate_hz > 0.0 && !c.is_empty()
    })?;
    let ms: Vec<f64> = speed.materialize().into_iter().map(|s| s / 3.6).collect();
    Some((integrate(&ms, speed.nominal_rate_hz), speed.nominal_rate_hz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ch(id: &str, rate: f64, samples: Vec<f64>) -> Channel {
        Channel::from_f64(id, rate, samples)
    }
    fn session(channels: Vec<Channel>) -> Session {
        Session {
            session_id: String::new(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: crate::session::TimestampSource::Header,
            config_checksum: None,
            source_format: crate::session::SourceFormat::Idl0,
            blob_sha256: String::new(),
            channels,
        }
    }

    #[test]
    fn plan_base_channels_event_only_channels_fall_back_to_the_longest_at_rate_zero() {
        // Arrange — a GPX-like session: every channel irregular.
        let dims = [("GPS_Latitude", 0.0, 80), ("WheelFront", 0.0, 900), ("GPS_SpeedKmh", 0.0, 80)];

        // Act
        let plan = plan_base_channels(&dims).unwrap();

        // Assert — Time follows WheelFront; no Distance without a fixed-rate GPS speed.
        assert_eq!(plan, BasePlan { time_source: 1, nominal_rate_hz: 0.0, len: 900, distance: false });
    }

    #[test]
    fn plan_base_channels_every_channel_empty_plans_nothing() {
        // Act
        let plan = plan_base_channels(&[("A", 100.0, 0), ("B", 0.0, 0)]);

        // Assert
        assert_eq!(plan, None);
    }

    #[test]
    fn time_channel_uses_highest_fixed_rate_and_longest_length() {
        // Arrange — 100 Hz/3 samples and 200 Hz/5 samples; Time should follow 200 Hz, len 5.
        let mut s = session(vec![ch("A", 100.0, vec![0.0; 3]), ch("B", 200.0, vec![0.0; 5])]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert_eq!(added, vec!["Time".to_string()]);
        let time = s.channels.iter().find(|c| c.channel_id == "Time").unwrap();
        assert_eq!(time.nominal_rate_hz, 200.0);
        assert_eq!(time.len(), 5);
        // `ch`/`from_f64` builds its channels with synthetic-uniform t_us
        // (t_us[i] = round(i * 1e6 / rate)), so these values numerically
        // coincide with i/rate for this fixture — but Time's real formula is
        // "the winning channel's own t_us / 1e6", not "i/rate" (C1 §3.5
        // invariant 4); the two only agree because the *source* channel here
        // happens to be perfectly uniform.
        assert_relative_eq!(time.materialize()[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(time.materialize()[4], 4.0 / 200.0, epsilon = 1e-9);
    }

    #[test]
    fn no_fixed_rate_channel_falls_back_to_its_own_t_us_with_zero_rate() {
        // Arrange — only an event-driven channel (rate 0). Ledger R23 Q2:
        // this is no longer "nothing to synthesize" — the fallback picks
        // this channel (it's the only one, trivially "most samples") and
        // carries its own real t_us into Time, at nominal_rate_hz 0.0.
        let mut s = session(vec![Channel::from_f64_with_times(
            "HR_RR",
            0.0,
            vec![1.0, 2.0],
            vec![500_000, 1_000_000],
            "hr_rr",
        )]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert_eq!(added, vec!["Time".to_string()]);
        let time = s.channels.iter().find(|c| c.channel_id == "Time").unwrap();
        assert_eq!(time.nominal_rate_hz, 0.0);
        assert_eq!(time.t_us, vec![500_000, 1_000_000]);
        assert_relative_eq!(time.materialize()[0], 0.5, epsilon = 1e-9);
        assert_relative_eq!(time.materialize()[1], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn event_driven_fallback_uses_longer_channels_own_t_us_not_a_ramp() {
        // Arrange — two event-driven channels (rate 0), different lengths
        // and different t_us. Ledger R23 Q2: Time follows the longer
        // channel's own t_us exactly, not i/rate and not a synthesized
        // ramp, with nominal_rate_hz 0.0 (never a fabricated rate).
        let mut s = session(vec![
            Channel::from_f64_with_times("Short", 0.0, vec![10.0, 20.0], vec![100, 200], "short"),
            Channel::from_f64_with_times(
                "Long",
                0.0,
                vec![1.0, 2.0, 3.0],
                vec![7_000, 9_000, 11_000],
                "long",
            ),
        ]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert_eq!(added, vec!["Time".to_string()]);
        let time = s.channels.iter().find(|c| c.channel_id == "Time").unwrap();
        assert_eq!(time.nominal_rate_hz, 0.0);
        assert_eq!(time.t_us, vec![7_000, 9_000, 11_000]);
        assert_relative_eq!(time.materialize()[0], 0.007, epsilon = 1e-9);
        assert_relative_eq!(time.materialize()[1], 0.009, epsilon = 1e-9);
        assert_relative_eq!(time.materialize()[2], 0.011, epsilon = 1e-9);
    }

    #[test]
    fn distance_integrates_speed_and_interpolates_to_time_grid() {
        // Arrange — Time at 2 Hz over 4 samples; GPS_SpeedKmh at 1 Hz, 3.6 km/h
        // constant (= 1 m/s). Distance at t seconds = t metres.
        let mut s = session(vec![
            ch("Main", 2.0, vec![0.0; 4]),                // Time → 2 Hz, len 4 (t = 0,0.5,1,1.5)
            ch("GPS_SpeedKmh", 1.0, vec![3.6, 3.6, 3.6]), // 1 m/s, len 3 (t = 0,1,2)
        ]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert_eq!(added, vec!["Time".to_string(), "Distance".to_string()]);
        let dist = s.channels.iter().find(|c| c.channel_id == "Distance").unwrap();
        assert_eq!(dist.nominal_rate_hz, 2.0);
        assert_eq!(dist.len(), 4);
        // distAtGps = [0,1,2] m; interp to t=0,0.5,1,1.5 → 0,0.5,1,1.5
        assert_relative_eq!(dist.materialize()[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(dist.materialize()[1], 0.5, epsilon = 1e-9);
        assert_relative_eq!(dist.materialize()[2], 1.0, epsilon = 1e-9);
        assert_relative_eq!(dist.materialize()[3], 1.5, epsilon = 1e-9);
    }

    #[test]
    fn distance_omitted_when_no_gps_speed() {
        // Arrange
        let mut s = session(vec![ch("Main", 10.0, vec![0.0; 10])]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert_eq!(added, vec!["Time".to_string()]);
        assert!(s.channels.iter().all(|c| c.channel_id != "Distance"));
    }
}
