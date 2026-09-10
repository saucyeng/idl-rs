//! Turbo colour ramp for raster charts (C3 §3.6's spectrogram/histogram2d
//! pixel encoding). Hand-rolled — `core/Cargo.toml` carries no `colorous`-style
//! LUT crate, and the Turbo ramp has a well-known, small polynomial fit
//! (Mikhailov, Google, 2019) that avoids pulling one in for ~10 lines of math.

/// Maps `t` in `[0.0, 1.0]` to an RGBA8 Turbo colour via the polynomial fit.
///
/// `t` is clamped into range before evaluation. Non-finite `t` (NaN, ±∞) is
/// the ramp's one transparency signal: returns `[0, 0, 0, 0]`, alpha `0`, so a
/// raster's non-finite pixels (an empty histogram bin post-substitution, an
/// empty spectrogram frame) show the chart's own background/gridlines through
/// instead of a false colour. A merely *out-of-range* finite `t` (clamped, not
/// NaN) still gets an opaque endpoint colour — only `NaN` is transparent.
pub fn turbo_rgba8(t: f64) -> [u8; 4] {
    if !t.is_finite() {
        return [0, 0, 0, 0];
    }
    let x = t.clamp(0.0, 1.0);

    // Mikhailov/Google Turbo polynomial approximation: two dot products of
    // (1, x, x^2, x^3) against a 4-vector plus (x^4, x^5) against a 2-vector,
    // flattened here into one polynomial per channel.
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let x5 = x4 * x;

    let red = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3
        - 152.94239396 * x4
        + 59.28637943 * x5;
    let green = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3
        + 4.27729857 * x4
        + 2.82956604 * x5;
    let blue = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3
        - 89.90310912 * x4
        + 27.34824973 * x5;

    [to_u8(red), to_u8(green), to_u8(blue), 255]
}

// Scales a [0.0, 1.0]-ish channel value to a rounded, clamped u8.
fn to_u8(v: f64) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Finds the finite min/max of `values`, or `(0.0, 0.0)` when none are finite
/// (empty input or all-NaN/∞) — the shared "0/0" guard both
/// [`normalize_to_colormap`] and the `*_raster_meta` functions in `raster.rs`
/// use so the bounds-scan logic lives in exactly one place.
pub(crate) fn finite_bounds(values: &[f64]) -> (f64, f64) {
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in values {
        if v.is_finite() {
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    if mn > mx {
        (0.0, 0.0)
    } else {
        (mn, mx)
    }
}

// TODO(idl0): linear min-max only — log/percentile scaling for heavily-skewed
// density counts (G11.8) is a deferred L6 display decision, not implemented here.
/// Linearly normalises `values` to Turbo RGBA8 colours.
///
/// Non-finite values (NaN, ±∞) map to `turbo_rgba8`'s transparent `[0,0,0,0]`
/// — this is how `raster.rs`'s `count == 0 → NaN` substitution becomes a
/// transparent pixel. The min/max span is taken over the *finite* values only
/// (`finite_bounds`), so substituted `NaN`s never widen the range.
///
/// Guards the degenerate case explicitly: an all-equal or empty finite span
/// (`max == min`, including "no finite values at all") maps every finite
/// value to `t = 0.0` — the LUT's zero-point *colour*, a visible flat colour,
/// **not** transparency. Only `NaN` is transparent; this is a different rule
/// from that one.
pub fn normalize_to_colormap(values: &[f64]) -> Vec<[u8; 4]> {
    let (mn, mx) = finite_bounds(values);
    colorize_with_bounds(values, mn, mx)
}

/// Samples the Turbo ramp at `n` evenly spaced `t`, both endpoints included:
/// stop `i` is `turbo_rgba8(i / (n - 1))`, so stop `0` is `t = 0.0` and stop
/// `n - 1` is `t = 1.0`. Each stop is opaque RGBA8 (`a = 255`) — the ramp's
/// transparent non-finite case (`turbo_rgba8(NaN)`) is never a stop.
///
/// This is the one place a *legend* may learn what the ramp looks like
/// (C3 §3.6 `RasterMeta.ramp_stops`, ruling R177): the app builds its colour
/// bar from stops it was given and never reimplements Turbo in TypeScript,
/// so the bar cannot drift from the pixels it describes.
///
/// `n < 2` (0 or 1) cannot express "evenly spaced including both endpoints",
/// so it returns exactly the two endpoint stops — a two-stop ramp is still a
/// truthful, if coarse, gradient, and no caller gets an empty or one-sided
/// legend.
pub fn turbo_stops(n: usize) -> Vec<[u8; 4]> {
    if n < 2 {
        return vec![turbo_rgba8(0.0), turbo_rgba8(1.0)];
    }
    (0..n).map(|i| turbo_rgba8(i as f64 / (n - 1) as f64)).collect()
}

/// Like [`normalize_to_colormap`], but against caller-supplied `(lo, hi)`
/// bounds instead of scanning `values` for its own min/max.
///
/// For a raster builder that rebins a larger source grid onto fewer output
/// pixels (`raster.rs`'s spectrogram path, ledger R38), the bounds must come
/// from the **full, un-rebinned** source data — nearest-cell rebinning is a
/// subset selection, not an average, so scanning only the rebinned pixels can
/// silently drop the true extreme value and produce a colour scale that
/// disagrees with `*_raster_meta`'s reported `vmin`/`vmax`. This function is
/// the shared mapping step both the meta function and the byte builder can
/// call against the same bounds, so they can never disagree.
///
/// Same NaN/degenerate rules as `normalize_to_colormap`: non-finite `v` is
/// transparent; `hi <= lo` (an empty or all-equal span) maps every finite `v`
/// to `t = 0.0`, a flat visible colour, not transparency.
pub(crate) fn colorize_with_bounds(values: &[f64], lo: f64, hi: f64) -> Vec<[u8; 4]> {
    let range = hi - lo;
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return turbo_rgba8(f64::NAN);
            }
            let t = if range > 0.0 { (v - lo) / range } else { 0.0 };
            turbo_rgba8(t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_rgba8_pins_the_luts_zero_endpoint_colour() {
        // Act
        let c = turbo_rgba8(0.0);

        // Assert — the polynomial's literal value at t=0.0; a future LUT
        // change is then a visible test diff, not a silent drift.
        assert_eq!(c, [35, 23, 27, 255]);
    }

    #[test]
    fn turbo_rgba8_pins_the_luts_one_endpoint_colour() {
        // Act
        let c = turbo_rgba8(1.0);

        // Assert — the polynomial's literal value at t=1.0.
        assert_eq!(c, [144, 13, 0, 255]);
    }

    #[test]
    fn turbo_stops_length_matches_the_requested_count() {
        // Act
        let stops = turbo_stops(16);

        // Assert
        assert_eq!(stops.len(), 16);
    }

    #[test]
    fn turbo_stops_endpoints_are_the_ramps_own_endpoints() {
        // Arrange
        let stops = turbo_stops(16);

        // Act
        let (first, last) = (stops[0], stops[15]);

        // Assert
        assert_eq!(first, turbo_rgba8(0.0));
        assert_eq!(last, turbo_rgba8(1.0));
    }

    #[test]
    fn turbo_stops_every_stop_is_opaque() {
        // Act
        let stops = turbo_stops(16);

        // Assert — the transparent non-finite case is never a stop.
        assert!(stops.iter().all(|s| s[3] == 255));
    }

    #[test]
    fn turbo_stops_sample_t_evenly_and_match_direct_ramp_calls() {
        // Arrange
        let n = 9;

        // Act
        let stops = turbo_stops(n);

        // Assert
        for (i, stop) in stops.iter().enumerate() {
            assert_eq!(*stop, turbo_rgba8(i as f64 / (n - 1) as f64), "stop {i}");
        }
    }

    #[test]
    fn turbo_stops_below_two_returns_the_two_endpoints() {
        // Act
        let zero = turbo_stops(0);
        let one = turbo_stops(1);

        // Assert
        assert_eq!(zero, vec![turbo_rgba8(0.0), turbo_rgba8(1.0)]);
        assert_eq!(one, vec![turbo_rgba8(0.0), turbo_rgba8(1.0)]);
    }

    #[test]
    fn turbo_stops_are_not_all_the_same_colour() {
        // Act
        let stops = turbo_stops(16);

        // Assert — a legend built from these is a gradient, not a flat block.
        assert!(stops.iter().any(|s| *s != stops[0]));
    }

    #[test]
    fn turbo_rgba8_nan_is_transparent() {
        // Act/Assert
        assert_eq!(turbo_rgba8(f64::NAN), [0, 0, 0, 0]);
    }

    #[test]
    fn normalize_to_colormap_all_equal_input_is_one_flat_colour_no_panic() {
        // Arrange — a constant, non-empty input.
        let values = vec![5.0, 5.0, 5.0];

        // Act
        let colours = normalize_to_colormap(&values);

        // Assert — every pixel the LUT's zero-point colour, all identical, no panic.
        assert_eq!(colours.len(), 3);
        assert!(colours.iter().all(|&c| c == colours[0]));
        assert_eq!(colours[0], turbo_rgba8(0.0));
    }
}
