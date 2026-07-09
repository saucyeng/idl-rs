//! Overlay layout model. Stored in the workbook (`.idl0wb` v2,
//! `overlay_layouts`); rects are normalized [x, y, w, h] canvas fractions;
//! `canvas` ("1920x1080") is design-space for stroke/font scaling only.
//! See docs/IDL0_SPEC.md §33.1.

use serde::{Deserialize, Serialize};

/// Normalized placement rectangle: fractions of the canvas in [0, 1].
/// Serialized as the JSON array `[x, y, w, h]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f32; 4]", into = "[f32; 4]")]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl From<[f32; 4]> for Rect {
    fn from(a: [f32; 4]) -> Self {
        Rect {
            x: a[0],
            y: a[1],
            w: a[2],
            h: a[3],
        }
    }
}

impl From<Rect> for [f32; 4] {
    fn from(r: Rect) -> Self {
        [r.x, r.y, r.w, r.h]
    }
}

/// Gauge visual style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GaugeStyle {
    Numeric,
    Bar,
    Dial,
}

/// Attitude indicator style for signed, zero-centered channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttitudeStyle {
    /// Tilting horizon/bike glyph + degree readout (roll angle).
    Roll,
    /// Zero-centered needle/arc + degree readout (steering angle).
    Steer,
}

/// One positioned overlay element. `channel` names resolve like chart
/// channels (raw, synthesized, or math); a missing channel degrades the
/// element to its no-data state, never fails the render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OverlayElement {
    /// Single-value readout. `min`/`max` bound the bar/dial travel (channel
    /// units); `label` is free text (typically the unit).
    Gauge {
        rect: Rect,
        channel: String,
        style: GaugeStyle,
        label: String,
        min: f64,
        max: f64,
    },
    /// Signed zero-centered indicator; `range_deg` is the full-scale
    /// deflection in degrees.
    Attitude {
        rect: Rect,
        channel: String,
        style: AttitudeStyle,
        range_deg: f64,
    },
    /// Scrolling time-series strip: trailing `window_s` seconds, "now" at the
    /// right edge, y normalized to each channel's session min/max.
    TraceStrip {
        rect: Rect,
        channels: Vec<String>,
        window_s: f64,
    },
    /// Session GPS outline with current-position dot.
    TrackMap { rect: Rect },
    /// Current/last/best lap readout.
    LapPanel { rect: Rect },
}

/// A named overlay layout: workbook asset, canvas-agnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayLayout {
    /// Stable UUIDv4.
    pub id: String,
    /// Display name; the CLI `--layout` selector.
    pub name: String,
    /// Design-space size as "WxH" pixels, e.g. "1920x1080".
    pub canvas: String,
    pub elements: Vec<OverlayElement>,
}

impl OverlayLayout {
    /// Parse `canvas` ("WxH" pixels); malformed input falls back to
    /// (1920, 1080).
    pub fn canvas_size(&self) -> (u32, u32) {
        let mut it = self.canvas.split('x');
        match (
            it.next().and_then(|s| s.trim().parse::<u32>().ok()),
            it.next().and_then(|s| s.trim().parse::<u32>().ok()),
        ) {
            (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
            _ => (1920, 1080),
        }
    }

    /// Every channel any element references — unique, document order.
    pub fn referenced_channels(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let push = |out: &mut Vec<String>, name: &str| {
            if !out.iter().any(|c| c == name) {
                out.push(name.to_string());
            }
        };
        for e in &self.elements {
            match e {
                OverlayElement::Gauge { channel, .. }
                | OverlayElement::Attitude { channel, .. } => push(&mut out, channel),
                OverlayElement::TraceStrip { channels, .. } => {
                    channels.iter().for_each(|c| push(&mut out, c))
                }
                OverlayElement::TrackMap { .. } | OverlayElement::LapPanel { .. } => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUT_JSON: &str = r#"{
      "id": "11111111-2222-3333-4444-555555555555",
      "name": "MTB default",
      "canvas": "1920x1080",
      "elements": [
        { "type": "gauge", "rect": [0.02, 0.80, 0.14, 0.16], "channel": "GPS_SpeedKmh",
          "style": "numeric", "label": "km/h", "min": 0, "max": 80 },
        { "type": "attitude", "rect": [0.18, 0.80, 0.10, 0.16], "channel": "Roll_deg",
          "style": "roll", "range_deg": 60 },
        { "type": "trace_strip", "rect": [0.30, 0.82, 0.40, 0.15],
          "channels": ["TravelFront_mm", "TravelRear_mm"], "window_s": 8.0 },
        { "type": "track_map", "rect": [0.84, 0.04, 0.14, 0.25] },
        { "type": "lap_panel", "rect": [0.02, 0.04, 0.16, 0.14] }
      ]
    }"#;

    #[test]
    fn deserialize_full_layout_json_parses_all_five_element_kinds() {
        // Arrange: LAYOUT_JSON above

        // Act
        let l: OverlayLayout = serde_json::from_str(LAYOUT_JSON).unwrap();

        // Assert
        assert_eq!(l.name, "MTB default");
        assert_eq!(l.elements.len(), 5);
        assert!(matches!(
            &l.elements[0],
            OverlayElement::Gauge {
                style: GaugeStyle::Numeric,
                ..
            }
        ));
        assert!(matches!(
            &l.elements[1],
            OverlayElement::Attitude {
                style: AttitudeStyle::Roll,
                ..
            }
        ));
        match &l.elements[2] {
            OverlayElement::TraceStrip {
                rect,
                channels,
                window_s,
            } => {
                assert_eq!(rect.x, 0.30f32);
                assert_eq!(channels.len(), 2);
                assert_eq!(*window_s, 8.0);
            }
            other => panic!("expected TraceStrip, got {other:?}"),
        }
        assert!(matches!(&l.elements[3], OverlayElement::TrackMap { .. }));
        assert!(matches!(&l.elements[4], OverlayElement::LapPanel { .. }));
    }

    #[test]
    fn roundtrip_serialize_then_deserialize_is_identical() {
        // Arrange
        let l: OverlayLayout = serde_json::from_str(LAYOUT_JSON).unwrap();

        // Act
        let back: OverlayLayout =
            serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();

        // Assert
        assert_eq!(l, back);
    }

    #[test]
    fn canvas_size_well_formed_parses_and_malformed_defaults() {
        // Arrange
        let mut l: OverlayLayout = serde_json::from_str(LAYOUT_JSON).unwrap();

        // Act + Assert
        assert_eq!(l.canvas_size(), (1920, 1080));
        l.canvas = "garbage".into();
        assert_eq!(l.canvas_size(), (1920, 1080));
        l.canvas = "3840x2160".into();
        assert_eq!(l.canvas_size(), (3840, 2160));
    }

    #[test]
    fn referenced_channels_dedupes_across_elements_in_document_order() {
        // Arrange
        let l: OverlayLayout = serde_json::from_str(LAYOUT_JSON).unwrap();

        // Act
        let chans = l.referenced_channels();

        // Assert
        assert_eq!(
            chans,
            vec![
                "GPS_SpeedKmh",
                "Roll_deg",
                "TravelFront_mm",
                "TravelRear_mm"
            ]
        );
    }
}
