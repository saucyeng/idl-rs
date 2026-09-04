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
    let range = mx - mn;
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return turbo_rgba8(f64::NAN);
            }
            let t = if range > 0.0 { (v - mn) / range } else { 0.0 };
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
