//! General GPS utilities: assemble a fix list from the session handle's GPS
//! channels. Shared by `laps` (gate crossings) and `tracks` (visit detection).
//! Coordinates are copied at the channel-sample scale — physical decimal
//! degrees, ruling R27; no rescaling happens here.

use crate::session::handle::SessionHandle;

/// A GPS position with timestamp. `lat`/`lon` are physical decimal degrees.
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
///
/// Resolves the three by name ([`SessionHandle::channel_samples`]) rather
/// than scanning `channel_data()`, so a lazy handle decodes exactly these
/// three columns and no others (ruling R211.1/.3) — lap indexing after an
/// import is this function's largest caller. Resolving by name also reaches
/// the derived store, which the old scan did not; no caller writes a
/// derived channel under a `GPS_*` name.
pub fn build_gps_track(handle: &SessionHandle) -> Vec<GpsFix> {
    let lat_s = handle.channel_samples("GPS_Latitude");
    let lon_s = handle.channel_samples("GPS_Longitude");
    let epoch_s = handle.channel_samples("GPS_EpochMs");
    if lat_s.is_empty() || lon_s.is_empty() || epoch_s.is_empty() {
        return Vec::new();
    }
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
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn ch(id: &str, samples: Vec<f64>) -> ChannelInput {
        let t_us = (0..samples.len() as i64).map(|i| i * 1_000_000).collect();
        ChannelInput {
            channel_id: id.to_string(),
            sample_rate_hz: 1.0,
            samples,
            t_us,
            source_kind: id.to_lowercase(),
        }
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
