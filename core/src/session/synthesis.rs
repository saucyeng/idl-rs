//! Synthesizes the `Time` and `Distance` base channels appended to a parsed
//! session, mirroring the Dart `channelDataProvider` synthesis
//! (app/lib/providers/channel_provider.dart §57-176). One source of truth so
//! the app and the CLI derive identical base channels.

use crate::integration::integrate;
use crate::session::{Channel, RawColumn, Session};

/// Appends a synthesized `Time` channel (and, when `GPS_SpeedKmh` is present, a
/// cumulative `Distance` channel) to `session`. Returns the ids it appended
/// (`["Time"]`, `["Time", "Distance"]`, or `[]`).
///
/// `Time`: built at the highest fixed-rate channel's rate, length = the longest
/// sample count at that rate, `samples[i] = i / rate`. Omitted (returns `[]`)
/// when the session has no fixed-rate channel.
///
/// `Distance` (metres): trapezoidal-integrate `GPS_SpeedKmh / 3.6` (km/h → m/s)
/// at the GPS rate, then linear-interpolate onto the `Time` grid, clamped at
/// both ends. Omitted when `GPS_SpeedKmh` is absent/empty.
pub fn synthesize_base_channels(session: &mut Session) -> Vec<String> {
    // Highest fixed-rate channel; longest sample count at that rate.
    let mut max_rate = 0.0_f64;
    let mut max_rate_len = 0usize;
    for c in &session.channels {
        if c.sample_rate_hz <= 0.0 {
            continue;
        }
        if c.sample_rate_hz > max_rate {
            max_rate = c.sample_rate_hz;
            max_rate_len = c.len();
        } else if c.sample_rate_hz == max_rate && c.len() > max_rate_len {
            max_rate_len = c.len();
        }
    }
    if max_rate <= 0.0 || max_rate_len == 0 {
        return Vec::new();
    }

    let mut added = vec!["Time".to_string()];

    // Distance base from GPS_SpeedKmh, if present and usable.
    let distance = synthesize_distance_base(&session.channels);

    // Time is a pure function of (len, rate) — RawColumn::Ramp stores no
    // samples (the eager 8 B/sample ramp cost ~800 MB at 100M samples).
    session.channels.push(Channel {
        channel_id: "Time".to_string(),
        sample_rate_hz: max_rate,
        column: RawColumn::Ramp { len: max_rate_len, rate: max_rate },
        sample_times_secs: None,
        gaps: Vec::new(),
    });
    if let Some((base, base_rate)) = distance {
        // Distance stores only the GPS-rate metres; presentation on the Time
        // grid is lazy (RawColumn::Interp reproduces the former eager
        // clamp-lerp bit-for-bit).
        session.channels.push(Channel {
            channel_id: "Distance".to_string(),
            sample_rate_hz: max_rate,
            column: RawColumn::Interp {
                base,
                base_rate,
                out_rate: max_rate,
                len: max_rate_len,
            },
            sample_times_secs: None,
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
        c.channel_id == "GPS_SpeedKmh" && c.sample_rate_hz > 0.0 && !c.is_empty()
    })?;
    let ms: Vec<f64> = speed.materialize().into_iter().map(|s| s / 3.6).collect();
    Some((integrate(&ms, speed.sample_rate_hz), speed.sample_rate_hz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ch(id: &str, rate: f64, samples: Vec<f64>) -> Channel {
        Channel::from_f64(id, rate, samples, None)
    }
    fn session(channels: Vec<Channel>) -> Session {
        Session {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
            channels,
        }
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
        assert_eq!(time.sample_rate_hz, 200.0);
        assert_eq!(time.len(), 5);
        assert_relative_eq!(time.materialize()[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(time.materialize()[4], 4.0 / 200.0, epsilon = 1e-9);
    }

    #[test]
    fn no_fixed_rate_channel_synthesizes_nothing() {
        // Arrange — only an event-driven channel (rate 0).
        let mut s = session(vec![Channel::from_f64(
            "HR_RR",
            0.0,
            vec![1.0, 2.0],
            Some(vec![0.5, 1.0]),
        )]);

        // Act
        let added = synthesize_base_channels(&mut s);

        // Assert
        assert!(added.is_empty());
        assert!(s.channels.iter().all(|c| c.channel_id != "Time"));
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
        assert_eq!(dist.sample_rate_hz, 2.0);
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
