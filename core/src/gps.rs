//! General GPS utilities: assemble a fix list from the session handle's GPS
//! channels. Shared by `laps` (gate crossings) and `tracks` (visit detection).
//! Coordinates are copied at the raw channel-sample scale (degrees × 1e7); no
//! rescaling happens here.

use crate::session::handle::SessionHandle;

/// A GPS position with timestamp. `lat`/`lon` are the raw channel-sample scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsFix {
    pub timestamp_ms: i64,
    pub lat: f64,
    pub lon: f64,
}

/// Build the fix list from the handle's `GPS_Latitude`/`GPS_Longitude`/
/// `GPS_EpochMs` channels: drop `(0,0)` fix-not-acquired sentinels, iterate to
/// the shortest of the three. Empty when any channel is absent. Samples are
/// copied verbatim (no rescaling).
pub fn build_gps_track(handle: &SessionHandle) -> Vec<GpsFix> {
    let chans = handle.channel_data();
    let find = |id: &str| chans.iter().find(|c| c.channel_id == id);
    let (lat, lon, epoch) = match (find("GPS_Latitude"), find("GPS_Longitude"), find("GPS_EpochMs")) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Vec::new(),
    };
    let lat_s = lat.materialize();
    let lon_s = lon.materialize();
    let epoch_s = epoch.materialize();
    let n = lat_s.len().min(lon_s.len()).min(epoch_s.len());
    let mut fixes = Vec::with_capacity(n);
    for i in 0..n {
        let la = lat_s[i];
        let lo = lon_s[i];
        if la == 0.0 && lo == 0.0 {
            continue;
        }
        fixes.push(GpsFix { timestamp_ms: epoch_s[i] as i64, lat: la, lon: lo });
    }
    fixes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn ch(id: &str, samples: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 1.0, samples, sample_times_secs: None }
    }

    #[test]
    fn build_gps_track_drops_zero_sentinels_and_zips_to_shortest() {
        // Arrange — 3 lat, 3 lon (index 0 is (0,0)), 2 epoch → zip to 2; drop sentinel.
        let h = handle_with(vec![
            ch("GPS_Latitude", vec![0.0, 10.0, 20.0]),
            ch("GPS_Longitude", vec![0.0, 5.0, 6.0]),
            ch("GPS_EpochMs", vec![1000.0, 2000.0]),
        ]);

        // Act
        let fixes = build_gps_track(&h);

        // Assert — index 0 is (0,0) sentinel → dropped; index 1 kept; index 2 beyond epoch.
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0], GpsFix { timestamp_ms: 2000, lat: 10.0, lon: 5.0 });
    }

    #[test]
    fn build_gps_track_empty_when_channel_absent() {
        // Arrange — no GPS_EpochMs.
        let h = handle_with(vec![ch("GPS_Latitude", vec![1.0]), ch("GPS_Longitude", vec![1.0])]);

        // Act + Assert
        assert!(build_gps_track(&h).is_empty());
    }
}
