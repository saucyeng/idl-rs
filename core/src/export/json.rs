//! Nested per-channel JSON writer. Lossless: each channel carries its samples
//! (and per-sample times for event-driven channels), so no rate-alignment
//! compromise is needed. Uses serde_json's pretty writer for human inspection.

use std::io::Write;

use serde::Serialize;

use super::{select_channels, ExportError, ExportOptions};
use crate::session::handle::SessionMeta;
use crate::session::Channel;

#[derive(Serialize)]
struct JsonExport<'a> {
    session: SessionMeta,
    channels: Vec<ChannelJson<'a>>,
}

#[derive(Serialize)]
struct ChannelJson<'a> {
    channel_id: &'a str,
    sample_rate_hz: f64,
    synthesized: bool,
    is_event_driven: bool,
    /// Materialized physical samples (transient, owned — the channel stores a
    /// compact `RawColumn`, so there is no resident `&[f64]` to borrow).
    samples: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_times_secs: Option<&'a [f64]>,
}

/// Stream `channels` (filtered by `options`) to `w` as nested JSON. `meta`
/// populates the `session` block; `synthesized_ids` flags synthesized channels.
pub(crate) fn write_json(
    meta: &SessionMeta,
    synthesized_ids: &[String],
    channels: &[Channel],
    w: &mut impl Write,
    options: &ExportOptions,
) -> Result<(), ExportError> {
    let selected = select_channels(channels, options)?;
    let channels = selected
        .iter()
        .map(|c| ChannelJson {
            channel_id: &c.channel_id,
            sample_rate_hz: c.sample_rate_hz,
            synthesized: synthesized_ids.iter().any(|id| id == &c.channel_id),
            is_event_driven: c.sample_rate_hz == 0.0,
            samples: c.materialize(),
            sample_times_secs: c.sample_times_secs.as_deref(),
        })
        .collect();
    let export = JsonExport { session: meta.clone(), channels };
    serde_json::to_writer_pretty(w, &export).map_err(ExportError::Json)
}

#[cfg(test)]
mod tests {
    use crate::export::{write_json_to_string_for_test, ExportOptions};
    use crate::session::handle::{ChannelInput, SessionHandle, SessionMetaInput};

    fn handle_with(channels: Vec<ChannelInput>) -> SessionHandle {
        let meta = SessionMetaInput {
            session_id: "abc".to_string(),
            device_id: "dev".to_string(),
            timestamp_utc_ms: 1700,
            config_checksum: "crc".to_string(),
        };
        SessionHandle::from_channels(meta, channels)
    }

    fn only(ids: &[&str]) -> ExportOptions {
        ExportOptions { channels: ids.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn json_fixed_rate_channel_has_no_sample_times_field() {
        // Arrange
        let h = handle_with(vec![ChannelInput {
            channel_id: "X".to_string(),
            sample_rate_hz: 10.0,
            samples: vec![1.0, 2.0],
            sample_times_secs: None,
        }]);

        // Act
        let v: serde_json::Value =
            serde_json::from_str(&write_json_to_string_for_test(&h, &only(&["X"]))).unwrap();

        // Assert
        let ch = &v["channels"][0];
        assert_eq!(ch["channel_id"], "X");
        assert_eq!(ch["is_event_driven"], false);
        assert_eq!(ch["synthesized"], false);
        assert_eq!(ch["samples"], serde_json::json!([1.0, 2.0]));
        assert!(ch.get("sample_times_secs").is_none());
    }

    #[test]
    fn json_event_channel_includes_sample_times() {
        // Arrange
        let h = handle_with(vec![ChannelInput {
            channel_id: "E".to_string(),
            sample_rate_hz: 0.0,
            samples: vec![5.0],
            sample_times_secs: Some(vec![0.5]),
        }]);

        // Act
        let v: serde_json::Value =
            serde_json::from_str(&write_json_to_string_for_test(&h, &only(&["E"]))).unwrap();

        // Assert
        let ch = &v["channels"][0];
        assert_eq!(ch["is_event_driven"], true);
        assert_eq!(ch["sample_times_secs"], serde_json::json!([0.5]));
    }

    #[test]
    fn json_session_meta_and_synthesized_flag() {
        // Arrange
        let h = handle_with(vec![ChannelInput {
            channel_id: "X".to_string(),
            sample_rate_hz: 10.0,
            samples: vec![0.0, 0.0],
            sample_times_secs: None,
        }]);

        // Act — export all channels (includes synthesized "Time").
        let v: serde_json::Value =
            serde_json::from_str(&write_json_to_string_for_test(&h, &ExportOptions::default()))
                .unwrap();

        // Assert — session meta surfaced; a synthesized channel is flagged.
        assert_eq!(v["session"]["session_id"], "abc");
        assert_eq!(v["session"]["timestamp_utc_ms"], 1700);
        let time = v["channels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["channel_id"] == "Time")
            .unwrap();
        assert_eq!(time["synthesized"], true);
    }
}
