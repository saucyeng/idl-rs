//! Overlay-frame rasterization: tiny-skia canvas + embedded IBM Plex Mono
//! text (OFL — license vendored beside the font files). Deterministic:
//! `(layout, sample) → pixels`, golden-image tested. See docs/IDL0_SPEC.md
//! §33.4. Element renderers land in the next task.

use std::sync::OnceLock;

use fontdue::Font;
use tiny_skia::{Pixmap, PremultipliedColorU8};

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
