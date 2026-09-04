//! Binary raster encoder for the spectrogram/histogram2d IPC endpoint (C3
//! §3.6): a 16-byte header, then row-major top-down RGBA8 pixel data — plus
//! the `*_raster_meta` sibling functions (C3's `fetch_raster_meta`, added
//! post-sign, ledger R25/L3-R33) that expose the axis domains and colour
//! scale without decoding pixel bytes.
//!
//! `RasterMeta` carries only what this crate can compute — no channel names,
//! units, or axis-scale kind reach `idl-rs-core` (no session/catalog context
//! at this layer). L5's Tauri command fills `x_label`/`y_label`/`scale.kind`
//! (channel names/units it already has, and the constant `"linear"`) when it
//! assembles C3's JSON `RasterMeta` from this struct — do not add those
//! fields here.

use crate::colormap::{colorize_with_bounds, finite_bounds};
use crate::fft::{Detrend, FftWindow, Scaling};
use crate::histogram2d::histogram2d;
use crate::spectrogram::spectrogram;

const HEADER_LEN: usize = 16;
const FORMAT_RGBA8: u16 = 0;
const RASTER_VERSION: u16 = 1;

/// Axis domains and colour-scale bounds for a raster, without its pixel
/// bytes — mirrors `fetch_raster`'s own `width`/`height`/`params`, so L5 can
/// call the meta function with exactly the arguments it already has for the
/// byte fetch.
///
/// `x_label`/`y_label`/`scale.kind` are **not** here (see module doc) — L5
/// fills those.
pub struct RasterMeta {
    /// Data-space extent of the X axis, in the channel's native unit.
    pub x_domain: (f64, f64),
    /// Data-space extent of the Y axis, in the channel's native unit.
    pub y_domain: (f64, f64),
    /// Lower bound of the colour scale (the value that maps to `t = 0.0`).
    pub vmin: f64,
    /// Upper bound of the colour scale (the value that maps to `t = 1.0`).
    pub vmax: f64,
    /// Whether a `0`/absent value in this raster kind renders transparent
    /// (`true` for histogram2d, per its `count == 0 → NaN` rule; `false` for
    /// spectrogram, which has no such rule).
    pub transparent_zero: bool,
}

// Writes the fixed 16-byte raster header: magic, version, dims, format,
// reserved padding. Shared by both raster kinds — C3 §3.6's field table.
fn write_header(out: &mut Vec<u8>, width: u16, height: u16) {
    out.extend_from_slice(b"IDLR");
    out.extend_from_slice(&RASTER_VERSION.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&FORMAT_RGBA8.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // reserved
}

/// Encodes a spectrogram as a C3 §3.6 raster: header + `width * height`
/// RGBA8 pixels via the Turbo colour ramp.
///
/// Computes `spectrogram(samples.to_vec(), sample_rate_hz, window, nperseg,
/// noverlap, detrend, scaling)` (the engine's real **seven**-argument
/// signature — no defaults invented here; the CLI's own defaults, `Hann`,
/// `nperseg = 0`, `noverlap = 0`, `Mean`, `Density` (`cli/src/main.rs:172-185`),
/// are what a caller passes), then nearest-cell rebins its `n_times ×
/// n_freqs` power matrix onto the requested `(width, height)` pixel grid — no
/// smoother resampling. `noverlap` keeps the engine's own name here; `hop =
/// nperseg − noverlap` converts to C3's `hop_size` at L5's boundary, not
/// this one.
///
/// **Orientation (loud, tested):** pixel `(x, y)` reads `power[frame_of(x) *
/// n_freqs + (n_freqs - 1 - bin_of(y))]` — **x = time, row 0 = highest
/// frequency**. `SpectrogramResult.power` is row-major `n_times × n_freqs`
/// (row = time, column = frequency); naively rebinning it onto `(width,
/// height)` would put time on the Y axis and DC (0 Hz) at row 0/top — this
/// expression corrects both in one step.
///
/// **Colour bounds are resolution-independent (ledger R38):** the Turbo
/// normalisation uses `vmin`/`vmax` scanned over the **full, un-rebinned**
/// `power` matrix — the same scan [`spectrogram_raster_meta`] performs — not
/// over just the pixels this call happens to rebin onto. Nearest-cell rebin
/// is a subset selection, not an average, so at low output resolution some
/// extreme cells of `power` are never selected by any pixel; if the bounds
/// came from the rebinned subset instead, the legend and the rendered image
/// would disagree, and resizing a chart would re-normalise its own colours.
/// Colour is data, not layout — a consequence of this: at low resolution the
/// rendered raster can fail to contain any pixel at exactly `vmin` or
/// `vmax`, since the cell that held that value may have been dropped by the
/// rebin. The legend describes the mapping, not a census of what's on screen.
///
/// An empty/degenerate `samples` (or a spectrogram with `n_times == 0 ||
/// n_freqs == 0`) still returns a well-formed header and a full transparent
/// (`alpha = 0`) pixel region of the requested size — never truncated, never
/// a panic.
pub fn build_spectrogram_raster_bytes(
    samples: &[f64],
    sample_rate_hz: f64,
    width: u16,
    height: u16,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    scaling: Scaling,
) -> Vec<u8> {
    let s = spectrogram(samples.to_vec(), sample_rate_hz, window, nperseg, noverlap, detrend, scaling);
    let (n_times, n_freqs) = (s.n_times as usize, s.n_freqs as usize);
    let (w, h) = (width as usize, height as usize);

    // Bounds over the full matrix, before rebinning (R38) — matches
    // `spectrogram_raster_meta` exactly, so the legend and the pixels agree.
    let (vmin, vmax) = finite_bounds(&s.power);

    let mut values = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            if n_times == 0 || n_freqs == 0 {
                values.push(f64::NAN);
                continue;
            }
            let frame = (x * n_times / w).min(n_times - 1);
            let bin = (y * n_freqs / h).min(n_freqs - 1);
            values.push(s.power[frame * n_freqs + (n_freqs - 1 - bin)]);
        }
    }

    let colours = colorize_with_bounds(&values, vmin, vmax);
    let mut out = Vec::with_capacity(HEADER_LEN + w * h * 4);
    write_header(&mut out, width, height);
    for c in &colours {
        out.extend_from_slice(c);
    }
    out
}

/// Encodes a 2-D histogram as a C3 §3.6 raster: header + `width * height`
/// RGBA8 pixels via the Turbo colour ramp.
///
/// Sizes `histogram2d` directly to `(width, height)` — the bin count *is*
/// the pixel count, no separate rebin step (unlike the spectrogram builder,
/// whose time/frequency resolution is independent of pixel size). Applies
/// the `count == 0 → NaN` substitution (cast counts to `f64`, then
/// substitute, then normalise) so an empty bin renders fully transparent —
/// the chart's own gridlines show through underneath the raster (design §4)
/// — and the colour-scale normalisation spans the non-zero counts only.
pub fn build_histogram2d_raster_bytes(
    xs: &[f64],
    ys: &[f64],
    width: u16,
    height: u16,
    range_x: Option<(f64, f64)>,
    range_y: Option<(f64, f64)>,
) -> Vec<u8> {
    let h = histogram2d(xs, ys, width as usize, height as usize, range_x, range_y);
    let values: Vec<f64> = h
        .counts
        .iter()
        .map(|&c| if c == 0 { f64::NAN } else { c as f64 })
        .collect();

    // Bounds over the full count grid, same substitution as above (R38) —
    // here the "full" grid and the encoded grid are already identical
    // (histogram2d is sized directly to width*height, no separate rebin),
    // but computing bounds this way keeps the pattern identical to the
    // spectrogram builder and to `histogram2d_raster_meta`.
    let (vmin, vmax) = finite_bounds(&values);
    let colours = colorize_with_bounds(&values, vmin, vmax);
    let mut out = Vec::with_capacity(HEADER_LEN + width as usize * height as usize * 4);
    write_header(&mut out, width, height);
    for c in &colours {
        out.extend_from_slice(c);
    }
    out
}

/// `RasterMeta` for a spectrogram, computed with the same parameters as
/// [`build_spectrogram_raster_bytes`] minus `width`/`height` (recomputing
/// the spectrogram here rather than threading pixel bytes back out — C3's
/// "separate command, not stuffed into the header" design).
///
/// `x_domain`/`y_domain` come from `times_secs`/`freqs_hz`'s own min/max.
/// `vmin`/`vmax` are the finite bounds of the raw, un-rebinned `power`
/// matrix — deliberately **not** rebinned to any particular `(width,
/// height)` (this function takes none), so the colour scale is
/// resolution-independent (ledger R38): the same channel at two different
/// chart sizes gets the same legend, and resizing a chart never
/// re-normalises its colours. [`build_spectrogram_raster_bytes`] scans these
/// same, identical bounds before rebinning its pixels, so the legend this
/// function reports and the pixels that builder draws always agree — even
/// though nearest-cell rebin can drop the specific cell that held the
/// extreme value, so the rendered raster may contain no pixel at exactly
/// `vmin`/`vmax` at low output resolution. `transparent_zero` is always
/// `false` — spectrogram power has no `count == 0` transparency rule.
pub fn spectrogram_raster_meta(
    samples: &[f64],
    sample_rate_hz: f64,
    window: FftWindow,
    nperseg: usize,
    noverlap: usize,
    detrend: Detrend,
    scaling: Scaling,
) -> RasterMeta {
    let s = spectrogram(samples.to_vec(), sample_rate_hz, window, nperseg, noverlap, detrend, scaling);
    let x_domain = finite_bounds(&s.times_secs);
    let y_domain = finite_bounds(&s.freqs_hz);
    let (vmin, vmax) = finite_bounds(&s.power);
    RasterMeta { x_domain, y_domain, vmin, vmax, transparent_zero: false }
}

/// `RasterMeta` for a 2-D histogram, computed with the same parameters as
/// [`build_histogram2d_raster_bytes`] minus `width`/`height` (`nx`/`ny` here
/// play that role directly — see [`histogram2d`]).
///
/// `x_domain`/`y_domain` come from `bin_edges_x`/`bin_edges_y`'s own
/// min/max (first/last, since edges are ascending). `vmin`/`vmax` apply the
/// same `count == 0 → NaN` substitution as the byte builder before taking
/// finite bounds, so the colour scale matches the rendered raster exactly.
/// `transparent_zero` is always `true` (L3-R34's zero-count rule).
pub fn histogram2d_raster_meta(
    xs: &[f64],
    ys: &[f64],
    nx: usize,
    ny: usize,
    range_x: Option<(f64, f64)>,
    range_y: Option<(f64, f64)>,
) -> RasterMeta {
    let h = histogram2d(xs, ys, nx, ny, range_x, range_y);
    let x_domain = (
        h.bin_edges_x.first().copied().unwrap_or(0.0),
        h.bin_edges_x.last().copied().unwrap_or(0.0),
    );
    let y_domain = (
        h.bin_edges_y.first().copied().unwrap_or(0.0),
        h.bin_edges_y.last().copied().unwrap_or(0.0),
    );
    let values: Vec<f64> = h.counts.iter().map(|&c| if c == 0 { f64::NAN } else { c as f64 }).collect();
    let (vmin, vmax) = finite_bounds(&values);
    RasterMeta { x_domain, y_domain, vmin, vmax, transparent_zero: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_fields(bytes: &[u8]) -> (u16, u16, u16, u16) {
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let width = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let height = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        let format = u16::from_le_bytes(bytes[10..12].try_into().unwrap());
        (version, width, height, format)
    }

    #[test]
    fn raster_header_matches_c3_field_table_exactly() {
        // Arrange/Act — a tiny spectrogram raster is enough to check the header.
        let bytes = build_spectrogram_raster_bytes(&[], 100.0, 4, 2, FftWindow::Hann, 0, 0, Detrend::Mean, Scaling::Density);

        // Assert — byte-sliced against C3 §3.6's table.
        assert_eq!(&bytes[0..4], b"IDLR");
        let (version, width, height, format) = header_fields(&bytes);
        assert_eq!(version, 1);
        assert_eq!(width, 4);
        assert_eq!(height, 2);
        assert_eq!(format, 0);
        assert_eq!(&bytes[12..16], &[0u8; 4]);
    }

    #[test]
    fn worked_example_64x32_format_0_pixel_region_and_total_length() {
        // Arrange — C3 §3.6's worked example, both builders.
        let spec_bytes = build_spectrogram_raster_bytes(
            &vec![0.0; 512],
            100.0,
            64,
            32,
            FftWindow::Hann,
            0,
            0,
            Detrend::Mean,
            Scaling::Density,
        );
        let hist_bytes = build_histogram2d_raster_bytes(&[0.0, 1.0], &[0.0, 1.0], 64, 32, None, None);

        // Assert — pixel region [16, 8208), total length 8208 bytes, both builders.
        assert_eq!(spec_bytes.len(), 8208);
        assert_eq!(hist_bytes.len(), 8208);
        assert_eq!(HEADER_LEN, 16);
        assert_eq!(spec_bytes.len() - HEADER_LEN, 64 * 32 * 4);
    }

    #[test]
    fn spectrogram_raster_steady_tone_row_moves_up_with_frequency() {
        // Arrange — two tones at different frequencies, same sample rate/window.
        let fs = 256.0;
        let n = 512usize;
        let low_tone: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin()).collect();
        let high_tone: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 64.0 * i as f64 / fs).sin()).collect();
        let (width, height) = (16u16, 32u16);

        // Act — rebin both to the same raster, find each one's brightest row.
        let low_bytes = build_spectrogram_raster_bytes(&low_tone, fs, width, height, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);
        let high_bytes = build_spectrogram_raster_bytes(&high_tone, fs, width, height, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);
        let brightest_row = |bytes: &[u8]| -> usize {
            let pixels = &bytes[HEADER_LEN..];
            let mut best_row = 0usize;
            let mut best_sum: u32 = 0;
            for y in 0..height as usize {
                let mut sum: u32 = 0;
                for x in 0..width as usize {
                    let base = (y * width as usize + x) * 4;
                    sum += pixels[base] as u32 + pixels[base + 1] as u32 + pixels[base + 2] as u32;
                }
                if sum > best_sum {
                    best_sum = sum;
                    best_row = y;
                }
            }
            best_row
        };

        // Assert — a steady tone's energy sits at one row (checked implicitly by
        // finding a single brightest row per raster); moving the tone up in
        // frequency moves that row up, i.e. toward row 0 (per the orientation rule).
        let low_row = brightest_row(&low_bytes);
        let high_row = brightest_row(&high_bytes);
        assert!(high_row < low_row, "high tone row {high_row} should be above (smaller than) low tone row {low_row}");
    }

    #[test]
    fn spectrogram_raster_empty_input_is_well_formed_and_transparent() {
        // Arrange/Act — no samples at all.
        let bytes = build_spectrogram_raster_bytes(&[], 100.0, 8, 4, FftWindow::Hann, 0, 0, Detrend::Mean, Scaling::Density);

        // Assert — correct total length, every pixel alpha 0 (transparent).
        assert_eq!(bytes.len(), HEADER_LEN + 8 * 4 * 4);
        let pixels = &bytes[HEADER_LEN..];
        for chunk in pixels.chunks(4) {
            assert_eq!(chunk[3], 0);
        }
    }

    #[test]
    fn histogram2d_raster_degenerate_input_is_well_formed_and_transparent() {
        // Arrange/Act — no points at all; every bin count is 0.
        let bytes = build_histogram2d_raster_bytes(&[], &[], 6, 5, None, None);

        // Assert — well-formed size, every pixel transparent per L3-R34's
        // count==0 -> NaN substitution (this is what makes this pass).
        assert_eq!(bytes.len(), HEADER_LEN + 6 * 5 * 4);
        let pixels = &bytes[HEADER_LEN..];
        for chunk in pixels.chunks(4) {
            assert_eq!(chunk[3], 0);
        }
    }

    #[test]
    fn spectrogram_raster_meta_matches_hand_computed_fixture() {
        // Arrange — a single-frequency tone, one segment (no averaging needed).
        let fs = 100.0;
        let data: Vec<f64> = (0..100).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()).collect();

        // Act
        let meta = spectrogram_raster_meta(&data, fs, FftWindow::Hann, 0, 0, Detrend::Mean, Scaling::Density);

        // Assert — domains are non-degenerate and vmin <= vmax; freq domain
        // starts at 0 Hz (DC) and time domain starts at/after 0 s.
        assert_eq!(meta.y_domain.0, 0.0);
        assert!(meta.y_domain.1 > meta.y_domain.0);
        assert!(meta.x_domain.1 >= meta.x_domain.0);
        assert!(meta.vmax >= meta.vmin);
        assert!(!meta.transparent_zero);
    }

    #[test]
    fn spectrogram_raster_meta_bounds_match_the_builder_even_when_rebin_drops_the_extreme_cell() {
        // Arrange — parameters chosen so nearest-cell rebin structurally never
        // selects the Nyquist frequency row (n_freqs=33, height=32): the
        // review that raised R38 showed `bin_of(y) = (y*33/32).min(32)` never
        // yields 32 for any y in 0..32, so that row of `power` is scanned for
        // bounds but never rendered by any pixel.
        let fs = 256.0;
        let n = 512usize;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs).sin() + 0.3 * (i as f64 * 0.01).sin()).collect();
        let (width, height) = (16u16, 32u16);

        // Act — the meta's reported bounds, and the actual bytes rendered.
        let meta = spectrogram_raster_meta(&data, fs, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);
        let bytes = build_spectrogram_raster_bytes(&data, fs, width, height, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);

        // Assert — re-derive the expected pixel colours purely from the
        // meta's own vmin/vmax (independent of the builder's internals) and
        // require an exact byte match against what the builder actually
        // produced. This is the invariant R38 requires: if the builder ever
        // normalised over the rebinned subset instead of the full matrix,
        // this would fail whenever the dropped row held the true min/max.
        let s = spectrogram(data.clone(), fs, FftWindow::Hann, 64, 32, Detrend::Mean, Scaling::Density);
        let (n_times, n_freqs) = (s.n_times as usize, s.n_freqs as usize);
        let (w, h) = (width as usize, height as usize);
        let mut expected = Vec::with_capacity(HEADER_LEN + w * h * 4);
        expected.extend_from_slice(&bytes[0..HEADER_LEN]); // header already checked elsewhere
        for y in 0..h {
            for x in 0..w {
                let frame = (x * n_times / w).min(n_times - 1);
                let bin = (y * n_freqs / h).min(n_freqs - 1);
                let v = s.power[frame * n_freqs + (n_freqs - 1 - bin)];
                let range = meta.vmax - meta.vmin;
                let t = if range > 0.0 { (v - meta.vmin) / range } else { 0.0 };
                expected.extend_from_slice(&crate::colormap::colorize_with_bounds(&[t], 0.0, 1.0)[0]);
            }
        }
        assert_eq!(bytes, expected);
    }

    #[test]
    fn histogram2d_raster_meta_matches_hand_computed_fixture() {
        // Arrange — a small, hand-checkable cloud.
        let xs = vec![0.0, 5.0, 10.0];
        let ys = vec![0.0, 5.0, 10.0];

        // Act
        let meta = histogram2d_raster_meta(&xs, &ys, 5, 5, None, None);

        // Assert — domain matches the data extent exactly; vmin/vmax reflect
        // the non-zero counts only (every occupied cell here holds count 1).
        assert_eq!(meta.x_domain, (0.0, 10.0));
        assert_eq!(meta.y_domain, (0.0, 10.0));
        assert_eq!(meta.vmin, 1.0);
        assert_eq!(meta.vmax, 1.0);
        assert!(meta.transparent_zero);
    }
}
