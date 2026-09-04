//! Long/tidy CSV writer: header `channel,time_s,value`, one row per sample.
//! Hand-rolled (no `csv` crate) — the schema is three controlled columns.

use std::io::Write;

use super::{csv_field, sample_time_secs, select_channels, ExportError, ExportOptions};
use crate::session::Channel;

/// Stream `channels` (filtered by `options`) to `w` as long/tidy CSV.
pub(crate) fn write_csv(
    channels: &[Channel],
    w: &mut impl Write,
    options: &ExportOptions,
) -> Result<(), ExportError> {
    let selected = select_channels(channels, options)?;
    writeln!(w, "channel,time_s,value")?;
    for ch in selected {
        let id = csv_field(&ch.channel_id);
        for (i, value) in ch.materialize().iter().enumerate() {
            let t = sample_time_secs(ch, i);
            writeln!(w, "{id},{t},{value}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::export::{write_csv_to_string_for_test, ExportOptions};
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

    fn only(ids: &[&str]) -> ExportOptions {
        ExportOptions { channels: ids.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn csv_fixed_rate_channel_emits_index_over_rate_times() {
        // Arrange — 1000 Hz, three samples.
        let h = handle_with(vec![ChannelInput {
            channel_id: "IMU0_AccelX".to_string(),
            sample_rate_hz: 1000.0,
            samples: vec![0.121, 0.13, 0.142],
            t_us: vec![0, 1_000, 2_000],
            source_kind: "imu0".to_string(),
        }]);

        // Act
        let out = write_csv_to_string_for_test(&h, &only(&["IMU0_AccelX"]));

        // Assert
        assert_eq!(
            out,
            "channel,time_s,value\n\
             IMU0_AccelX,0,0.121\n\
             IMU0_AccelX,0.001,0.13\n\
             IMU0_AccelX,0.002,0.142\n"
        );
    }

    #[test]
    fn csv_event_driven_channel_uses_per_sample_times() {
        // Arrange
        let h = handle_with(vec![ChannelInput {
            channel_id: "HR_RR".to_string(),
            sample_rate_hz: 0.0,
            samples: vec![1000.0, 900.0],
            t_us: vec![500_000, 1_250_000],
            source_kind: "hr_rr".to_string(),
        }]);

        // Act
        let out = write_csv_to_string_for_test(&h, &only(&["HR_RR"]));

        // Assert
        assert_eq!(
            out,
            "channel,time_s,value\n\
             HR_RR,0.5,1000\n\
             HR_RR,1.25,900\n"
        );
    }

    #[test]
    fn csv_quotes_channel_id_with_comma() {
        // Arrange — pathological id with a comma.
        let h = handle_with(vec![ChannelInput {
            channel_id: "a,b".to_string(),
            sample_rate_hz: 1.0,
            samples: vec![7.0],
            t_us: vec![0],
            source_kind: "a_b".to_string(),
        }]);

        // Act
        let out = write_csv_to_string_for_test(&h, &only(&["a,b"]));

        // Assert
        assert_eq!(out, "channel,time_s,value\n\"a,b\",0,7\n");
    }

    #[test]
    fn csv_exports_synthesized_time_channel() {
        // Arrange — selecting "Time" proves synthesized channels flow through.
        let h = handle_with(vec![ChannelInput {
            channel_id: "X".to_string(),
            sample_rate_hz: 10.0,
            samples: vec![0.0, 0.0],
            t_us: vec![0, 100_000],
            source_kind: "x".to_string(),
        }]);

        // Act
        let out = write_csv_to_string_for_test(&h, &only(&["Time"]));

        // Assert — Time channel present with rows.
        assert!(out.starts_with("channel,time_s,value\n"));
        assert!(out.contains("Time,"));
    }
}
