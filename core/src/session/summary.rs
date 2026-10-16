//! Whole-ride totals for the ride form's facts strip (ride-form mockup,
//! 2026-10-16): heart rate and distance, worked out here so no number
//! reaches the screen from JavaScript (CLAUDE.md §2).

use serde::Serialize;

use crate::session::handle::SessionHandle;

/// Below this GPS speed a fix-to-fix step is taken as standing still, km/h:
/// a parked receiver wanders a few metres and would otherwise add distance.
const STATIONARY_KMH: f64 = 2.0;

/// A ride's totals. Each is `None` when the session has no data for it.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct RideSummary {
    /// Mean of the heart-rate samples, bpm.
    pub avg_hr_bpm: Option<f64>,
    /// Highest heart-rate sample, bpm.
    pub max_hr_bpm: Option<f64>,
    /// Ground distance along the GPS track, metres: great-circle steps
    /// between consecutive fixes, leaving out steps made below
    /// [`STATIONARY_KMH`] when the receiver reports speed.
    pub distance_m: Option<f64>,
}

/// The totals for `handle`. Heart rate uses `HR_BPM` samples above zero (a
/// strap reports 0 before it locks on); distance uses `GPS_Latitude` and
/// `GPS_Longitude`, skipping the (0, 0) no-fix sentinel.
pub fn ride_summary(handle: &SessionHandle) -> RideSummary {
    let hr: Vec<f64> = handle.channel_samples("HR_BPM").into_iter().filter(|b| b.is_finite() && *b > 0.0).collect();
    let (avg_hr_bpm, max_hr_bpm) = if hr.is_empty() {
        (None, None)
    } else {
        (Some(hr.iter().sum::<f64>() / hr.len() as f64), hr.iter().copied().reduce(f64::max))
    };

    let lat = handle.channel_samples("GPS_Latitude");
    let lon = handle.channel_samples("GPS_Longitude");
    let speed = handle.channel_samples("GPS_SpeedKmh");
    let n = lat.len().min(lon.len());
    let mut distance = 0.0;
    let mut fixes = 0usize;
    let mut prev: Option<(f64, f64)> = None;
    for i in 0..n {
        if (lat[i] == 0.0 && lon[i] == 0.0) || !lat[i].is_finite() || !lon[i].is_finite() {
            continue;
        }
        fixes += 1;
        let moving = speed.get(i).map_or(true, |s| !s.is_finite() || *s >= STATIONARY_KMH);
        if let Some((plat, plon)) = prev {
            if moving {
                distance += crate::export::fit::haversine_m(plat, plon, lat[i], lon[i]);
            }
        }
        prev = Some((lat[i], lon[i]));
    }
    let distance_m = (fixes >= 2).then_some(distance);

    RideSummary { avg_hr_bpm, max_hr_bpm, distance_m }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionMetaInput};

    fn channel(id: &str, samples: Vec<f64>) -> ChannelInput {
        let t_us = (0..samples.len() as i64).map(|i| i * 1_000_000).collect();
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 1.0, samples, t_us, source_kind: "gps".to_string() }
    }

    fn handle(channels: Vec<ChannelInput>) -> SessionHandle {
        SessionHandle::from_channels(SessionMetaInput { session_id: "s".into(), device_id: None, timestamp_utc_ms: 0, config_checksum: None }, channels)
    }

    #[test]
    fn ride_summary_heart_rate_ignores_zero_samples() {
        // Arrange
        let h = handle(vec![channel("HR_BPM", vec![0.0, 140.0, 150.0, 160.0])]);

        // Act
        let s = ride_summary(&h);

        // Assert
        assert_eq!(s.avg_hr_bpm, Some(150.0));
        assert_eq!(s.max_hr_bpm, Some(160.0));
        assert_eq!(s.distance_m, None);
    }

    #[test]
    fn ride_summary_distance_sums_moving_steps_and_skips_no_fix_and_standing_still() {
        // Arrange — 0.001° of latitude is about 111 m. The second step is
        // made standing still and the (0, 0) row is the no-fix sentinel.
        let h = handle(vec![
            channel("GPS_Latitude", vec![50.0, 0.0, 50.001, 50.002, 50.003]),
            channel("GPS_Longitude", vec![-120.0, 0.0, -120.0, -120.0, -120.0]),
            channel("GPS_SpeedKmh", vec![20.0, 0.0, 20.0, 0.5, 20.0]),
        ]);

        // Act
        let s = ride_summary(&h);

        // Assert — two moving steps of ~111.2 m.
        let d = s.distance_m.unwrap();
        assert!((d - 222.4).abs() < 1.0, "got {d}");
        assert_eq!(s.avg_hr_bpm, None);
    }
}
