//! Export the engine's channel set to text formats. Pure and streaming: every
//! writer takes a caller-provided `io::Write`, so the core never opens a sink
//! (stays within the "no I/O beyond std::fs" rule). Borrows the SessionHandle's
//! channels — no per-channel clones. Reused by the CLI, the app, and future
//! Python/WASM bindings (roadmap D8 / §11: export buffer as a first-class
//! engine output).

mod csv;
pub mod fit;
mod json;

pub use fit::{write_fit, FitExportError, FitLap, FitOptions, FitSport};

use std::borrow::Cow;
use std::fmt;
use std::io;

use crate::session::handle::{SessionHandle, SessionMeta};
use crate::session::Channel;

/// Output serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Long/tidy CSV: `channel,time_s,value`, one row per sample.
    Csv,
    /// Nested per-channel JSON.
    Json,
}

/// Controls what is exported. `Default` selects all channels.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Allow-list of channel ids, in the order to emit. Empty = all channels.
    pub channels: Vec<String>,
}

/// Errors raised while exporting.
#[derive(Debug)]
pub enum ExportError {
    /// A requested channel id is not present in the session.
    UnknownChannel(String),
    /// The output sink failed.
    Io(io::Error),
    /// JSON serialization failed.
    Json(serde_json::Error),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::UnknownChannel(id) => write!(f, "unknown channel '{id}'"),
            ExportError::Io(e) => write!(f, "write failed: {e}"),
            ExportError::Json(e) => write!(f, "json serialization failed: {e}"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<io::Error> for ExportError {
    fn from(e: io::Error) -> Self {
        ExportError::Io(e)
    }
}

/// Resolve the channels to export, in requested order. Empty options = all.
/// Returns [`ExportError::UnknownChannel`] for the first id not in the session.
pub(crate) fn select_channels<'a>(
    channels: &'a [Channel],
    options: &ExportOptions,
) -> Result<Vec<&'a Channel>, ExportError> {
    if options.channels.is_empty() {
        return Ok(channels.iter().collect());
    }
    let mut out = Vec::with_capacity(options.channels.len());
    for name in &options.channels {
        match channels.iter().find(|c| &c.channel_id == name) {
            Some(c) => out.push(c),
            None => return Err(ExportError::UnknownChannel(name.clone())),
        }
    }
    Ok(out)
}

/// Time in seconds of sample `i`: per-sample time for event-driven channels
/// (`sample_rate_hz == 0`), else `i / sample_rate_hz`.
pub(crate) fn sample_time_secs(ch: &Channel, i: usize) -> f64 {
    if ch.sample_rate_hz == 0.0 {
        ch.sample_times_secs
            .as_ref()
            .and_then(|t| t.get(i))
            .copied()
            .unwrap_or(0.0)
    } else {
        i as f64 / ch.sample_rate_hz
    }
}

/// CSV-escape a field: quote and double internal quotes only when the field
/// contains a comma, quote, CR, or LF. Channel ids are normally `[A-Za-z0-9_]`,
/// so this is defensive.
pub(crate) fn csv_field(s: &str) -> Cow<'_, str> {
    if s.contains([',', '"', '\n', '\r']) {
        Cow::Owned(format!("\"{}\"", s.replace('"', "\"\"")))
    } else {
        Cow::Borrowed(s)
    }
}

/// Stream `handle`'s selected channels to `w` in `format`.
///
/// Returns [`ExportError::UnknownChannel`] if `options.channels` names a
/// channel the session does not contain. Truncated sessions export normally;
/// the caller decides whether to warn (the truncation flag is on
/// [`SessionHandle::metadata`]).
pub fn write(
    handle: &SessionHandle,
    w: &mut impl io::Write,
    format: ExportFormat,
    options: &ExportOptions,
) -> Result<(), ExportError> {
    write_channels(
        &handle.metadata(),
        handle.synthesized_channel_ids(),
        handle.channel_data(),
        w,
        format,
        options,
    )
}

/// Stream an explicit channel slice to `w`. `meta` and `synthesized_ids` feed
/// the JSON writer's session block and per-channel `synthesized` flag, so a
/// caller can export channels the handle does not own (e.g. the derived
/// channels from `workbook::apply_workbook`, with `synthesized_ids = &[]`).
///
/// Returns [`ExportError::UnknownChannel`] if `options.channels` names a channel
/// not in `channels`.
pub fn write_channels(
    meta: &SessionMeta,
    synthesized_ids: &[String],
    channels: &[Channel],
    w: &mut impl io::Write,
    format: ExportFormat,
    options: &ExportOptions,
) -> Result<(), ExportError> {
    match format {
        ExportFormat::Csv => csv::write_csv(channels, w, options),
        ExportFormat::Json => json::write_json(meta, synthesized_ids, channels, w, options),
    }
}

#[cfg(test)]
pub(crate) fn write_csv_to_string_for_test(
    handle: &SessionHandle,
    options: &ExportOptions,
) -> String {
    let mut buf: Vec<u8> = Vec::new();
    csv::write_csv(handle.channel_data(), &mut buf, options).unwrap();
    String::from_utf8(buf).unwrap()
}

#[cfg(test)]
pub(crate) fn write_json_to_string_for_test(
    handle: &SessionHandle,
    options: &ExportOptions,
) -> String {
    let mut buf: Vec<u8> = Vec::new();
    json::write_json(
        &handle.metadata(),
        handle.synthesized_channel_ids(),
        handle.channel_data(),
        &mut buf,
        options,
    )
    .unwrap();
    String::from_utf8(buf).unwrap()
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

    fn fixed(id: &str, rate: f64, samples: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: rate, samples, sample_times_secs: None }
    }

    fn event(id: &str, samples: Vec<f64>, times: Vec<f64>) -> ChannelInput {
        ChannelInput { channel_id: id.to_string(), sample_rate_hz: 0.0, samples, sample_times_secs: Some(times) }
    }

    #[test]
    fn select_channels_empty_options_returns_all() {
        // Arrange
        let h = handle_with(vec![fixed("X", 10.0, vec![1.0])]);

        // Act
        let sel = select_channels(h.channel_data(), &ExportOptions::default()).unwrap();

        // Assert — all channels, including synthesized "Time".
        assert_eq!(sel.len(), h.channel_data().len());
    }

    #[test]
    fn select_channels_subset_preserves_requested_order() {
        // Arrange
        let h = handle_with(vec![fixed("A", 10.0, vec![1.0]), fixed("B", 10.0, vec![2.0])]);
        let opts = ExportOptions { channels: vec!["B".to_string(), "A".to_string()] };

        // Act
        let sel = select_channels(h.channel_data(), &opts).unwrap();

        // Assert
        assert_eq!(sel.iter().map(|c| c.channel_id.as_str()).collect::<Vec<_>>(), vec!["B", "A"]);
    }

    #[test]
    fn select_channels_unknown_id_is_error() {
        // Arrange
        let h = handle_with(vec![fixed("A", 10.0, vec![1.0])]);
        let opts = ExportOptions { channels: vec!["nope".to_string()] };

        // Act
        let r = select_channels(h.channel_data(), &opts);

        // Assert
        assert!(matches!(r, Err(ExportError::UnknownChannel(ref s)) if s == "nope"));
    }

    #[test]
    fn sample_time_secs_fixed_rate_is_index_over_rate() {
        // Arrange — 1000 Hz: sample 1 is at 1 ms = 0.001 s.
        let h = handle_with(vec![fixed("X", 1000.0, vec![0.0, 0.0])]);
        let ch = &h.channel_data()[0];

        // Act + Assert
        assert_eq!(sample_time_secs(ch, 0), 0.0);
        assert_eq!(sample_time_secs(ch, 1), 0.001);
    }

    #[test]
    fn sample_time_secs_event_driven_uses_per_sample_time() {
        // Arrange
        let h = handle_with(vec![event("E", vec![5.0, 6.0], vec![0.5, 1.25])]);
        let ch = h.channel_data().iter().find(|c| c.channel_id == "E").unwrap();

        // Act + Assert
        assert_eq!(sample_time_secs(ch, 1), 1.25);
    }

    #[test]
    fn csv_field_quotes_only_when_needed() {
        // Act + Assert
        assert_eq!(csv_field("IMU0_AccelX"), "IMU0_AccelX");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn write_dispatches_to_csv() {
        // Arrange
        let h = handle_with(vec![fixed("X", 10.0, vec![1.0])]);
        let mut buf: Vec<u8> = Vec::new();

        // Act
        write(&h, &mut buf, ExportFormat::Csv, &ExportOptions::default()).unwrap();

        // Assert — CSV header.
        assert!(String::from_utf8(buf).unwrap().starts_with("channel,time_s,value\n"));
    }

    #[test]
    fn write_dispatches_to_json() {
        // Arrange
        let h = handle_with(vec![fixed("X", 10.0, vec![1.0])]);
        let mut buf: Vec<u8> = Vec::new();

        // Act
        write(&h, &mut buf, ExportFormat::Json, &ExportOptions::default()).unwrap();

        // Assert — parseable JSON object with a session key.
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(v["session"].is_object());
    }

    #[test]
    fn write_propagates_unknown_channel_error() {
        // Arrange
        let h = handle_with(vec![fixed("X", 10.0, vec![1.0])]);
        let mut buf: Vec<u8> = Vec::new();
        let opts = ExportOptions { channels: vec!["nope".to_string()] };

        // Act
        let r = write(&h, &mut buf, ExportFormat::Csv, &opts);

        // Assert
        assert!(matches!(r, Err(ExportError::UnknownChannel(_))));
    }

    #[test]
    fn write_channels_exports_an_explicit_owned_slice() {
        // Arrange — a "derived" channel not owned by any handle, plus empty meta.
        let meta = SessionMeta {
            session_id: String::new(),
            device_id: String::new(),
            timestamp_utc_ms: 0,
            config_checksum: String::new(),
            channel_count: 1,
            duration_ms: 0,
            truncation_warning: None,
        };
        let derived = vec![Channel::from_f64("ForkVelocity", 10.0, vec![1.5, 2.5], None)];
        let mut buf: Vec<u8> = Vec::new();

        // Act
        write_channels(&meta, &[], &derived, &mut buf, ExportFormat::Csv, &ExportOptions::default())
            .unwrap();

        // Assert
        let out = String::from_utf8(buf).unwrap();
        assert_eq!(out, "channel,time_s,value\nForkVelocity,0,1.5\nForkVelocity,0.1,2.5\n");
    }
}
