//! Overlay-frame rasterization: tiny-skia canvas + embedded IBM Plex Mono
//! text (OFL — license vendored beside the font files). Deterministic:
//! `(layout, sample) → pixels`, golden-image tested. See docs/IDL0_SPEC.md
//! §33.4.
//!
//! Visual spec (v1, deliberately plain): every element sits on a rounded
//! panel (`rgba(10,10,14,168)` fill, hairline border); values in SemiBold,
//! labels in Regular; amber accent. All strokes/fonts scale by
//! `s = output_h / canvas_h` so one layout serves 1080p and 4K.

use std::sync::OnceLock;

use fontdue::Font;
use tiny_skia::{
    Color, FillRule, Paint, PathBuilder, Pixmap, PremultipliedColorU8, Shader, Stroke, Transform,
};

use crate::overlay::model::{AttitudeStyle, GaugeStyle, OverlayElement, OverlayLayout};
use crate::overlay::sample::{ElementSample, FrameSample, LapState};

/// Panel background fill (straight RGBA).
const PANEL_FILL: [u8; 4] = [10, 10, 14, 168];
/// Panel hairline border.
const PANEL_BORDER: [u8; 4] = [255, 255, 255, 64];
/// Primary text.
const TEXT: [u8; 4] = [255, 255, 255, 255];
/// Accent (amber) — needles, fills, best lap.
const ACCENT: [u8; 4] = [255, 179, 0, 255];
/// No-data glyph color (50 % white).
const NO_DATA: [u8; 4] = [255, 255, 255, 128];
/// Trace-series palette: amber, sky, green, pink.
const SERIES: [[u8; 4]; 4] = [
    [255, 179, 0, 255],
    [64, 196, 255, 255],
    [120, 255, 120, 255],
    [255, 120, 180, 255],
];
/// Track polyline (70 % white).
const TRACK: [u8; 4] = [255, 255, 255, 178];

/// Embedded IBM Plex Mono Regular (matches app branding; OFL license).
pub(crate) fn font_regular() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(
            include_bytes!("../../assets/fonts/IBMPlexMono-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .expect("embedded IBMPlexMono-Regular.ttf is a valid font")
    })
}

/// Embedded IBM Plex Mono SemiBold (value readouts).
pub(crate) fn font_semibold() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(
            include_bytes!("../../assets/fonts/IBMPlexMono-SemiBold.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .expect("embedded IBMPlexMono-SemiBold.ttf is a valid font")
    })
}

/// Draw `text` left-aligned at (`x`, baseline `y`) in `px` pixel size onto
/// the (premultiplied) pixmap, alpha-blended. `color` is straight RGBA.
/// Returns the advanced width in pixels. Monospaced: Plex Mono's constant
/// advance keeps layout trivial.
pub(crate) fn draw_text(
    pm: &mut Pixmap,
    font: &Font,
    text: &str,
    x: f32,
    y: f32,
    px: f32,
    color: [u8; 4],
) -> f32 {
    let width = pm.width() as i32;
    let height = pm.height() as i32;
    let mut advance = 0.0f32;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, px);
        let gx = (x + advance) as i32 + metrics.xmin;
        // `ymin` is the bitmap bottom's offset from the baseline (up = +).
        let gy = y as i32 - metrics.height as i32 - metrics.ymin;
        for row in 0..metrics.height as i32 {
            let py = gy + row;
            if py < 0 || py >= height {
                continue;
            }
            for col in 0..metrics.width as i32 {
                let pxx = gx + col;
                if pxx < 0 || pxx >= width {
                    continue;
                }
                let cov = bitmap[(row * metrics.width as i32 + col) as usize] as u32;
                if cov == 0 {
                    continue;
                }
                // Coverage × text alpha, in 0..=255.
                let a_src = cov * color[3] as u32 / 255;
                let idx = (py * width + pxx) as usize;
                let pixels = pm.pixels_mut();
                let dst = pixels[idx];
                let inv = 255 - a_src;
                // Premultiplied over-blend, channel = src_c*a + dst_c*(1-a).
                let r = (color[0] as u32 * a_src + dst.red() as u32 * inv) / 255;
                let g = (color[1] as u32 * a_src + dst.green() as u32 * inv) / 255;
                let b = (color[2] as u32 * a_src + dst.blue() as u32 * inv) / 255;
                let a = a_src + dst.alpha() as u32 * inv / 255;
                pixels[idx] = PremultipliedColorU8::from_rgba(r as u8, g as u8, b as u8, a as u8)
                    .unwrap_or(dst);
            }
        }
        advance += metrics.advance_width;
    }
    advance
}

/// Monospace advance for one glyph at `px` (Plex Mono is fixed-pitch).
fn mono_advance(font: &Font, px: f32) -> f32 {
    font.metrics('0', px).advance_width
}

fn color(c: [u8; 4]) -> Color {
    Color::from_rgba8(c[0], c[1], c[2], c[3])
}

fn paint(c: [u8; 4]) -> Paint<'static> {
    Paint {
        shader: Shader::SolidColor(color(c)),
        anti_alias: true,
        ..Paint::default()
    }
}

fn stroke(width: f32) -> Stroke {
    Stroke {
        width,
        ..Stroke::default()
    }
}

/// An element's pixel-space box.
#[derive(Debug, Clone, Copy)]
struct Box2 {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Box2 {
    fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }
    fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }
    /// Inset on all sides.
    fn inset(&self, d: f32) -> Box2 {
        Box2 {
            x: self.x + d,
            y: self.y + d,
            w: (self.w - 2.0 * d).max(1.0),
            h: (self.h - 2.0 * d).max(1.0),
        }
    }
}

/// Rounded-rect path (quarter-circle cubics, kappa ≈ 0.5523).
fn rrect_path(b: Box2, r: f32) -> Option<tiny_skia::Path> {
    let r = r.min(b.w / 2.0).min(b.h / 2.0).max(0.0);
    let k = 0.552_284_75 * r;
    let (x0, y0, x1, y1) = (b.x, b.y, b.x + b.w, b.y + b.h);
    let mut p = PathBuilder::new();
    p.move_to(x0 + r, y0);
    p.line_to(x1 - r, y0);
    p.cubic_to(x1 - r + k, y0, x1, y0 + r - k, x1, y0 + r);
    p.line_to(x1, y1 - r);
    p.cubic_to(x1, y1 - r + k, x1 - r + k, y1, x1 - r, y1);
    p.line_to(x0 + r, y1);
    p.cubic_to(x0 + r - k, y1, x0, y1 - r + k, x0, y1 - r);
    p.line_to(x0, y0 + r);
    p.cubic_to(x0, y0 + r - k, x0 + r - k, y0, x0 + r, y0);
    p.close();
    p.finish()
}

fn polyline_path(pts: &[(f32, f32)]) -> Option<tiny_skia::Path> {
    if pts.len() < 2 {
        return None;
    }
    let mut p = PathBuilder::new();
    p.move_to(pts[0].0, pts[0].1);
    for &(x, y) in &pts[1..] {
        p.line_to(x, y);
    }
    p.finish()
}

/// Panel background + border every element sits on.
fn draw_panel(pm: &mut Pixmap, b: Box2, s: f32) {
    if let Some(path) = rrect_path(b, 8.0 * s) {
        pm.fill_path(
            &path,
            &paint(PANEL_FILL),
            FillRule::Winding,
            Transform::identity(),
            None,
        );
        pm.stroke_path(
            &path,
            &paint(PANEL_BORDER),
            &stroke(1.5 * s),
            Transform::identity(),
            None,
        );
    }
}

/// Centered no-data glyph.
fn draw_no_data(pm: &mut Pixmap, b: Box2, _s: f32) {
    let px = 0.3 * b.h;
    let adv = mono_advance(font_regular(), px);
    draw_text(
        pm,
        font_regular(),
        "—",
        b.cx() - adv / 2.0,
        b.cy() + px * 0.35,
        px,
        NO_DATA,
    );
}

/// Value formatting: 1 decimal ≥ 10 in magnitude, 2 below.
fn fmt_value(v: f64) -> String {
    if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// `m:ss.d` for an in-progress elapsed time (seconds).
fn fmt_elapsed(secs: f64) -> String {
    let m = (secs / 60.0).floor() as i64;
    let s = secs - m as f64 * 60.0;
    format!("{m}:{s:04.1}")
}

/// `m:ss.ss` for a completed lap time (milliseconds).
fn fmt_lap_ms(ms: i64) -> String {
    let m = ms / 60_000;
    let s = (ms % 60_000) as f64 / 1000.0;
    format!("{m}:{s:05.2}")
}

fn draw_gauge(
    pm: &mut Pixmap,
    b: Box2,
    s: f32,
    style: GaugeStyle,
    label: &str,
    min: f64,
    max: f64,
    value: Option<f64>,
) {
    let Some(v) = value else {
        draw_no_data(pm, b, s);
        return;
    };
    let pad = 8.0 * s;
    let inner = b.inset(pad);
    let frac = if max > min {
        ((v - min) / (max - min)).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    match style {
        GaugeStyle::Numeric => {
            let vpx = 0.42 * b.h;
            draw_text(
                pm,
                font_semibold(),
                &fmt_value(v),
                inner.x,
                inner.y + vpx,
                vpx,
                TEXT,
            );
            let lpx = 0.14 * b.h;
            draw_text(
                pm,
                font_regular(),
                label,
                inner.x,
                b.y + b.h - pad,
                lpx,
                TEXT,
            );
        }
        GaugeStyle::Bar => {
            let lpx = 0.14 * b.h;
            draw_text(pm, font_regular(), label, inner.x, inner.y + lpx, lpx, TEXT);
            let vpx = 0.2 * b.h;
            let vtxt = fmt_value(v);
            let vw = mono_advance(font_semibold(), vpx) * vtxt.chars().count() as f32;
            draw_text(
                pm,
                font_semibold(),
                &vtxt,
                inner.x + inner.w - vw,
                inner.y + lpx + vpx,
                vpx,
                TEXT,
            );
            let track_h = 0.18 * b.h;
            let ty = b.y + 0.65 * b.h;
            let track = Box2 {
                x: inner.x,
                y: ty,
                w: inner.w,
                h: track_h,
            };
            if let Some(path) = rrect_path(track, 3.0 * s) {
                pm.fill_path(
                    &path,
                    &paint([255, 255, 255, 40]),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
            let fill = Box2 {
                x: inner.x,
                y: ty,
                w: (inner.w * frac).max(1.0),
                h: track_h,
            };
            if let Some(path) = rrect_path(fill, 3.0 * s) {
                pm.fill_path(
                    &path,
                    &paint(ACCENT),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }
        GaugeStyle::Dial => {
            let cx = b.cx();
            let cy = b.y + 0.52 * b.h;
            let radius = (b.w.min(b.h * 1.1) / 2.0 - pad).max(4.0);
            // Classic speedometer arc: 135° → 405° (y-down screen coords).
            let arc: Vec<(f32, f32)> = (0..=48)
                .map(|i| {
                    let deg = 135.0 + 270.0 * i as f32 / 48.0;
                    let rad = deg.to_radians();
                    (cx + radius * rad.cos(), cy + radius * rad.sin())
                })
                .collect();
            if let Some(path) = polyline_path(&arc) {
                pm.stroke_path(
                    &path,
                    &paint([255, 255, 255, 90]),
                    &stroke(3.0 * s),
                    Transform::identity(),
                    None,
                );
            }
            let needle_rad = (135.0 + 270.0 * frac).to_radians();
            let needle = [
                (cx, cy),
                (
                    cx + 0.8 * radius * needle_rad.cos(),
                    cy + 0.8 * radius * needle_rad.sin(),
                ),
            ];
            if let Some(path) = polyline_path(&needle) {
                pm.stroke_path(
                    &path,
                    &paint(ACCENT),
                    &stroke(3.0 * s),
                    Transform::identity(),
                    None,
                );
            }
            let vpx = 0.16 * b.h;
            let vtxt = fmt_value(v);
            let vw = mono_advance(font_semibold(), vpx) * vtxt.chars().count() as f32;
            draw_text(
                pm,
                font_semibold(),
                &vtxt,
                cx - vw / 2.0,
                b.y + 0.78 * b.h,
                vpx,
                TEXT,
            );
            let lpx = 0.12 * b.h;
            let lw = mono_advance(font_regular(), lpx) * label.chars().count() as f32;
            draw_text(
                pm,
                font_regular(),
                label,
                cx - lw / 2.0,
                b.y + b.h - pad,
                lpx,
                TEXT,
            );
        }
    }
}

fn draw_attitude(
    pm: &mut Pixmap,
    b: Box2,
    s: f32,
    style: AttitudeStyle,
    range_deg: f64,
    value: Option<f64>,
) {
    let Some(v) = value else {
        draw_no_data(pm, b, s);
        return;
    };
    let pad = 8.0 * s;
    let clamped = v.clamp(-range_deg, range_deg);
    let readout = format!("{clamped:+.0}°");
    let rpx = 0.18 * b.h;
    let rw = mono_advance(font_regular(), rpx) * readout.chars().count() as f32;
    match style {
        AttitudeStyle::Roll => {
            // Horizon line through the panel center, rotated by -v degrees.
            let half = 0.4 * b.w;
            let rad = (-clamped as f32).to_radians();
            let (dx, dy) = (half * rad.cos(), half * rad.sin());
            let line = [(b.cx() - dx, b.cy() - dy), (b.cx() + dx, b.cy() + dy)];
            if let Some(path) = polyline_path(&line) {
                pm.stroke_path(
                    &path,
                    &paint(ACCENT),
                    &stroke(3.0 * s),
                    Transform::identity(),
                    None,
                );
            }
            // Fixed center marker triangle.
            let t = 5.0 * s;
            let mut p = PathBuilder::new();
            p.move_to(b.cx(), b.cy() - t);
            p.line_to(b.cx() + t, b.cy() + t);
            p.line_to(b.cx() - t, b.cy() + t);
            p.close();
            if let Some(path) = p.finish() {
                pm.fill_path(
                    &path,
                    &paint(TEXT),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
            draw_text(
                pm,
                font_regular(),
                &readout,
                b.cx() - rw / 2.0,
                b.y + b.h - pad,
                rpx,
                TEXT,
            );
        }
        AttitudeStyle::Steer => {
            // Needle pivoting at bottom-center; 0° = straight up.
            let pivot = (b.cx(), b.y + b.h - pad);
            let len = 0.62 * b.h;
            let rad = ((clamped / range_deg) as f32 * 90.0 - 90.0).to_radians();
            let tip = (pivot.0 + len * rad.cos(), pivot.1 + len * rad.sin());
            // Zero tick.
            let tick = [(b.cx(), pivot.1 - len), (b.cx(), pivot.1 - len + 6.0 * s)];
            if let Some(path) = polyline_path(&tick) {
                pm.stroke_path(
                    &path,
                    &paint([255, 255, 255, 90]),
                    &stroke(2.0 * s),
                    Transform::identity(),
                    None,
                );
            }
            if let Some(path) = polyline_path(&[pivot, tip]) {
                pm.stroke_path(
                    &path,
                    &paint(ACCENT),
                    &stroke(3.0 * s),
                    Transform::identity(),
                    None,
                );
            }
            draw_text(
                pm,
                font_regular(),
                &readout,
                b.cx() - rw / 2.0,
                b.y + rpx + 4.0 * s,
                rpx,
                TEXT,
            );
        }
    }
}

fn draw_trace(pm: &mut Pixmap, b: Box2, s: f32, series: &[Vec<(f32, f32)>]) {
    let inner = b.inset(8.0 * s);
    if series.iter().all(|pts| pts.is_empty()) {
        draw_no_data(pm, b, s);
        return;
    }
    for (i, pts) in series.iter().enumerate() {
        let mapped: Vec<(f32, f32)> = pts
            .iter()
            .map(|&(x, y)| (inner.x + x * inner.w, inner.y + (1.0 - y) * inner.h))
            .collect();
        if let Some(path) = polyline_path(&mapped) {
            pm.stroke_path(
                &path,
                &paint(SERIES[i % SERIES.len()]),
                &stroke(2.0 * s),
                Transform::identity(),
                None,
            );
        }
    }
    // "Now" hairline at the right edge.
    let edge = [
        (inner.x + inner.w, inner.y),
        (inner.x + inner.w, inner.y + inner.h),
    ];
    if let Some(path) = polyline_path(&edge) {
        pm.stroke_path(
            &path,
            &paint(NO_DATA),
            &stroke(1.0 * s),
            Transform::identity(),
            None,
        );
    }
}

fn draw_track_map(
    pm: &mut Pixmap,
    b: Box2,
    s: f32,
    polyline: &[(f32, f32)],
    pos: Option<(f32, f32)>,
) {
    let inner = b.inset(8.0 * s);
    if polyline.len() < 2 {
        draw_no_data(pm, b, s);
        return;
    }
    // Letterbox: square drawing area centered in the panel (normalized track
    // coordinates are 0..1 on both axes).
    let side = inner.w.min(inner.h);
    let ox = inner.x + (inner.w - side) / 2.0;
    let oy = inner.y + (inner.h - side) / 2.0;
    let map = |&(x, y): &(f32, f32)| (ox + x * side, oy + (1.0 - y) * side);
    let mapped: Vec<(f32, f32)> = polyline.iter().map(map).collect();
    if let Some(path) = polyline_path(&mapped) {
        pm.stroke_path(
            &path,
            &paint(TRACK),
            &stroke(2.0 * s),
            Transform::identity(),
            None,
        );
    }
    if let Some(p) = pos {
        let (cx, cy) = map(&p);
        let mut pb = PathBuilder::new();
        pb.push_circle(cx, cy, 4.0 * s);
        if let Some(path) = pb.finish() {
            pm.fill_path(
                &path,
                &paint(ACCENT),
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
}

fn draw_lap_panel(pm: &mut Pixmap, b: Box2, s: f32, laps: &LapState) {
    let pad = 8.0 * s;
    let inner = b.inset(pad);
    let lh = 0.22 * b.h;
    let px = 0.17 * b.h;
    let row = |i: f32| inner.y + lh * (i + 1.0);
    let current = match laps.current_lap {
        Some(n) => format!("LAP {n}  {}", fmt_elapsed(laps.lap_elapsed_s)),
        None => "LAP —".to_string(),
    };
    let last = format!(
        "LAST {}",
        laps.last_lap_ms.map(fmt_lap_ms).unwrap_or("—".into())
    );
    let best = format!(
        "BEST {}",
        laps.best_lap_ms.map(fmt_lap_ms).unwrap_or("—".into())
    );
    draw_text(pm, font_semibold(), &current, inner.x, row(0.0), px, TEXT);
    draw_text(pm, font_regular(), &last, inner.x, row(1.0), px, TEXT);
    draw_text(pm, font_regular(), &best, inner.x, row(2.0), px, ACCENT);
}

/// Rasterize one overlay frame at (`w`, `h`). `polyline` is the normalized
/// session track from `SampleContext::track_polyline`. Returns **straight**
/// (un-premultiplied) RGBA bytes, `w*h*4`, row-major — ffmpeg `rawvideo
/// rgba` order. `sample.elements` must parallel `layout.elements` (as
/// produced by `SampleContext::sample`); mismatched entries are skipped.
pub fn render_overlay_frame(
    layout: &OverlayLayout,
    sample: &FrameSample,
    polyline: &[(f32, f32)],
    w: u32,
    h: u32,
) -> Vec<u8> {
    let mut pm = Pixmap::new(w.max(1), h.max(1)).expect("nonzero pixmap dims");
    let (_, canvas_h) = layout.canvas_size();
    let s = h as f32 / canvas_h as f32;

    for (elem, es) in layout.elements.iter().zip(sample.elements.iter()) {
        let rect = match elem {
            OverlayElement::Gauge { rect, .. }
            | OverlayElement::Attitude { rect, .. }
            | OverlayElement::TraceStrip { rect, .. }
            | OverlayElement::TrackMap { rect }
            | OverlayElement::LapPanel { rect } => rect,
        };
        let b = Box2 {
            x: rect.x * w as f32,
            y: rect.y * h as f32,
            w: rect.w * w as f32,
            h: rect.h * h as f32,
        };
        draw_panel(&mut pm, b, s);
        match (elem, es) {
            (
                OverlayElement::Gauge {
                    style,
                    label,
                    min,
                    max,
                    ..
                },
                ElementSample::Value(v),
            ) => draw_gauge(&mut pm, b, s, *style, label, *min, *max, *v),
            (
                OverlayElement::Attitude {
                    style, range_deg, ..
                },
                ElementSample::Value(v),
            ) => draw_attitude(&mut pm, b, s, *style, *range_deg, *v),
            (OverlayElement::TraceStrip { .. }, ElementSample::Trace(series)) => {
                draw_trace(&mut pm, b, s, series)
            }
            (OverlayElement::TrackMap { .. }, ElementSample::MapPos(pos)) => {
                draw_track_map(&mut pm, b, s, polyline, *pos)
            }
            (OverlayElement::LapPanel { .. }, ElementSample::Laps(laps)) => {
                draw_lap_panel(&mut pm, b, s, laps)
            }
            _ => {}
        }
    }

    // Demultiply on the way out: ffmpeg's overlay filter expects straight
    // alpha for rawvideo rgba input.
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for p in pm.pixels() {
        let c = p.demultiply();
        out.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    out
}

#[cfg(test)]
mod text_tests {
    use super::*;

    #[test]
    fn draw_text_hello_covers_pixels_and_is_deterministic() {
        // Arrange
        let mut pm = Pixmap::new(200, 60).unwrap();

        // Act
        let w = draw_text(
            &mut pm,
            font_regular(),
            "HELLO",
            4.0,
            40.0,
            24.0,
            [255, 255, 255, 255],
        );
        let lit = pm.data().chunks(4).filter(|p| p[0] > 0).count();

        // Assert
        assert!(w > 50.0, "monospace advance accumulates, got {w}");
        assert!(lit > 100, "glyph coverage rendered, got {lit}");
        let mut pm2 = Pixmap::new(200, 60).unwrap();
        draw_text(
            &mut pm2,
            font_regular(),
            "HELLO",
            4.0,
            40.0,
            24.0,
            [255, 255, 255, 255],
        );
        assert_eq!(pm.data(), pm2.data(), "deterministic");
    }

    #[test]
    fn draw_text_clips_at_canvas_edges_without_panic() {
        // Arrange
        let mut pm = Pixmap::new(20, 20).unwrap();

        // Act — baseline above the canvas top and text running off the right.
        draw_text(
            &mut pm,
            font_semibold(),
            "999999",
            -5.0,
            2.0,
            48.0,
            [255, 0, 0, 255],
        );

        // Assert — reaching here without a panic is the contract.
        assert_eq!(pm.width(), 20);
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::overlay::sample::{ElementSample, FrameSample, LapState};

    /// The five-element layout used by both goldens.
    fn layout() -> OverlayLayout {
        serde_json::from_str(
            r#"{
          "id": "L", "name": "golden", "canvas": "1920x1080",
          "elements": [
            { "type": "gauge", "rect": [0.02, 0.78, 0.15, 0.18], "channel": "GPS_SpeedKmh",
              "style": "numeric", "label": "km/h", "min": 0, "max": 80 },
            { "type": "attitude", "rect": [0.19, 0.78, 0.11, 0.18], "channel": "Roll_deg",
              "style": "roll", "range_deg": 60 },
            { "type": "trace_strip", "rect": [0.32, 0.80, 0.40, 0.16],
              "channels": ["TravelFront_mm", "TravelRear_mm"], "window_s": 8.0 },
            { "type": "track_map", "rect": [0.82, 0.04, 0.16, 0.28] },
            { "type": "lap_panel", "rect": [0.02, 0.04, 0.18, 0.16] }
          ]
        }"#,
        )
        .unwrap()
    }

    /// Deterministic triangle wave in [0.1, 0.9] (no libm — exact in f32).
    fn tri(x: f32) -> f32 {
        let t = (x * 2.0).fract();
        let up = if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
        0.1 + 0.8 * up
    }

    fn full_sample() -> FrameSample {
        let series_a: Vec<(f32, f32)> = (0..=64)
            .map(|i| (i as f32 / 64.0, tri(i as f32 / 64.0)))
            .collect();
        let series_b: Vec<(f32, f32)> = (0..=64)
            .map(|i| (i as f32 / 64.0, 1.0 - tri(i as f32 / 64.0)))
            .collect();
        FrameSample {
            t_secs: 95.0,
            elements: vec![
                ElementSample::Value(Some(42.3)),
                ElementSample::Value(Some(-15.0)),
                ElementSample::Trace(vec![series_a, series_b]),
                ElementSample::MapPos(Some((0.3, 0.7))),
                ElementSample::Laps(LapState {
                    current_lap: Some(3),
                    lap_elapsed_s: 12.3,
                    last_lap_ms: Some(83_450),
                    best_lap_ms: Some(79_120),
                }),
            ],
        }
    }

    fn nodata_sample() -> FrameSample {
        FrameSample {
            t_secs: 0.0,
            elements: vec![
                ElementSample::Value(None),
                ElementSample::Value(None),
                ElementSample::Trace(vec![vec![], vec![]]),
                ElementSample::MapPos(None),
                ElementSample::Laps(LapState::default()),
            ],
        }
    }

    /// A small closed loop as the track polyline (exact fractions).
    fn poly() -> Vec<(f32, f32)> {
        vec![
            (0.1, 0.1),
            (0.9, 0.15),
            (0.85, 0.8),
            (0.4, 0.9),
            (0.1, 0.5),
            (0.1, 0.1),
        ]
    }

    /// Compare `rgba` against the checked-in PNG golden; regenerate it when
    /// GOLDEN_WRITE=1 is set.
    fn golden(name: &str, rgba: &[u8], w: u32, h: u32) {
        let mut pm = Pixmap::new(w, h).unwrap();
        for (i, px) in pm.pixels_mut().iter_mut().enumerate() {
            let c = &rgba[i * 4..i * 4 + 4];
            let straight = tiny_skia::ColorU8::from_rgba(c[0], c[1], c[2], c[3]);
            *px = straight.premultiply();
        }
        let encoded = pm.encode_png().unwrap();
        let path = format!("{}/tests/golden/{name}.png", env!("CARGO_MANIFEST_DIR"));
        if std::env::var("GOLDEN_WRITE").is_ok() {
            std::fs::create_dir_all(format!("{}/tests/golden", env!("CARGO_MANIFEST_DIR")))
                .unwrap();
            std::fs::write(&path, &encoded).unwrap();
            return;
        }
        let want = std::fs::read(&path)
            .unwrap_or_else(|_| panic!("golden {path} missing — rerun with GOLDEN_WRITE=1"));
        assert_eq!(
            encoded, want,
            "golden mismatch for {name} — inspect and regenerate if intended"
        );
    }

    #[test]
    fn render_overlay_frame_full_sample_matches_golden() {
        // Arrange
        let layout = layout();
        let sample = full_sample();

        // Act
        let rgba = render_overlay_frame(&layout, &sample, &poly(), 640, 360);

        // Assert
        assert_eq!(rgba.len(), 640 * 360 * 4);
        golden("overlay_full", &rgba, 640, 360);
    }

    #[test]
    fn render_overlay_frame_all_no_data_matches_golden_without_panic() {
        // Arrange
        let layout = layout();
        let sample = nodata_sample();

        // Act
        let rgba = render_overlay_frame(&layout, &sample, &[], 640, 360);

        // Assert
        golden("overlay_nodata", &rgba, 640, 360);
    }

    #[test]
    fn render_overlay_frame_output_is_straight_alpha() {
        // Arrange — panel edges are antialiased, so some pixel has 0<a<255.
        let layout = layout();
        let sample = full_sample();

        // Act
        let rgba = render_overlay_frame(&layout, &sample, &poly(), 640, 360);

        // Assert — in straight alpha a bright channel can exceed alpha;
        // impossible in premultiplied encoding.
        let found = rgba
            .chunks(4)
            .any(|p| p[3] > 0 && p[3] < 255 && (p[0] > p[3] || p[1] > p[3] || p[2] > p[3]));
        assert!(
            found,
            "expected straight-alpha pixels (channel > alpha at partial coverage)"
        );
    }

    #[test]
    fn render_overlay_frame_dial_and_bar_styles_render_without_panic() {
        // Arrange
        let layout: OverlayLayout = serde_json::from_str(
            r#"{
          "id": "L2", "name": "styles", "canvas": "1920x1080",
          "elements": [
            { "type": "gauge", "rect": [0.05, 0.05, 0.2, 0.3], "channel": "A",
              "style": "dial", "label": "rpm", "min": 0, "max": 100 },
            { "type": "gauge", "rect": [0.3, 0.05, 0.3, 0.15], "channel": "B",
              "style": "bar", "label": "throttle", "min": 0, "max": 1 },
            { "type": "attitude", "rect": [0.65, 0.05, 0.15, 0.25], "channel": "C",
              "style": "steer", "range_deg": 45 }
          ]
        }"#,
        )
        .unwrap();
        let sample = FrameSample {
            t_secs: 0.0,
            elements: vec![
                ElementSample::Value(Some(63.0)),
                ElementSample::Value(Some(0.42)),
                ElementSample::Value(Some(-12.0)),
            ],
        };

        // Act
        let rgba = render_overlay_frame(&layout, &sample, &[], 640, 360);

        // Assert — content rendered (some accent-colored pixels present).
        let lit = rgba.chunks(4).filter(|p| p[3] > 0).count();
        assert!(
            lit > 1000,
            "expected panels + needles to cover pixels, got {lit}"
        );
    }
}
