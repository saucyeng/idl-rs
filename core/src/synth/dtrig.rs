//! Deterministic trigonometry for the synthetic generator (C1 §9.7 rule 1).
//!
//! The platform's `f64::sin` is *not* required to be correctly rounded — its
//! last unit in the last place is unspecified, and glibc, musl and the MSVC
//! CRT disagree on some arguments. That is harmless for a display value and
//! fatal for a generator whose output bytes are committed as a fixture and
//! diffed across machines: one differing ULP before the `i16` quantisation
//! step is enough to flip an LSB.
//!
//! Everything here is built from `+ − × ÷`, comparison, `floor`/`ceil` and
//! `sqrt`, all of which IEEE-754 requires to be exactly rounded and which are
//! therefore bit-identical on every target Rust supports. Accuracy is
//! deliberately traded for reproducibility: these functions agree with `std`
//! to better than 1e-12 absolute (asserted in [`tests`]), which is ~10 orders
//! of magnitude finer than the sensor LSBs this feeds.
//!
//! Not a general-purpose math library. Arguments are assumed to be modest in
//! magnitude (|x| below ~1e6 radians); range reduction is a single
//! subtraction of a multiple of 2π and loses precision beyond that.

/// π to `f64` precision.
pub const PI: f64 = std::f64::consts::PI;
/// 2π to `f64` precision.
pub const TAU: f64 = std::f64::consts::TAU;
/// π/2 to `f64` precision.
pub const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

/// Nearest integer, halfway away from zero. `floor`/`ceil` are exact, so this
/// is exact — and so it is the rounding the generator uses everywhere it
/// quantises a physical value to an integer wire field, not `f64::round`
/// (which is the same function, but stating it here keeps the "no libm"
/// claim checkable in one place).
pub fn round_half_away(x: f64) -> f64 {
    if x >= 0.0 {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }
}

/// Maclaurin series for `sin` on |x| ≤ π/2, Horner in x².
///
/// Truncated after x¹⁹/19!; the first dropped term is x²¹/21!, which at the
/// largest argument (π/2) is ~3e-16 — below the `f64` resolution of the
/// result itself.
fn sin_core(x: f64) -> f64 {
    const C3: f64 = -1.0 / 6.0;
    const C5: f64 = 1.0 / 120.0;
    const C7: f64 = -1.0 / 5040.0;
    const C9: f64 = 1.0 / 362_880.0;
    const C11: f64 = -1.0 / 39_916_800.0;
    const C13: f64 = 1.0 / 6_227_020_800.0;
    const C15: f64 = -1.0 / 1_307_674_368_000.0;
    const C17: f64 = 1.0 / 355_687_428_096_000.0;
    const C19: f64 = -1.0 / 121_645_100_408_832_000.0;

    let x2 = x * x;
    let p = C19;
    let p = p * x2 + C17;
    let p = p * x2 + C15;
    let p = p * x2 + C13;
    let p = p * x2 + C11;
    let p = p * x2 + C9;
    let p = p * x2 + C7;
    let p = p * x2 + C5;
    let p = p * x2 + C3;
    x + x * x2 * p
}

/// Maclaurin series for `cos` on |x| ≤ π/2, Horner in x².
///
/// Truncated after x²⁰/20!; the first dropped term is x²²/22!, ~3e-17 at π/2.
fn cos_core(x: f64) -> f64 {
    const C2: f64 = -1.0 / 2.0;
    const C4: f64 = 1.0 / 24.0;
    const C6: f64 = -1.0 / 720.0;
    const C8: f64 = 1.0 / 40320.0;
    const C10: f64 = -1.0 / 3_628_800.0;
    const C12: f64 = 1.0 / 479_001_600.0;
    const C14: f64 = -1.0 / 87_178_291_200.0;
    const C16: f64 = 1.0 / 20_922_789_888_000.0;
    const C18: f64 = -1.0 / 6_402_373_705_728_000.0;
    const C20: f64 = 1.0 / 2_432_902_008_176_640_000.0;

    let x2 = x * x;
    let p = C20;
    let p = p * x2 + C18;
    let p = p * x2 + C16;
    let p = p * x2 + C14;
    let p = p * x2 + C12;
    let p = p * x2 + C10;
    let p = p * x2 + C8;
    let p = p * x2 + C6;
    let p = p * x2 + C4;
    let p = p * x2 + C2;
    1.0 + x2 * p
}

/// Reduces `x` radians to `[-π, π]` and reports whether the caller must
/// negate the result of a `cos_core`/`sin_core` evaluated on the returned
/// half-range value.
///
/// Returns `(y, negate)` with |y| ≤ π/2.
fn reduce(x: f64) -> (f64, bool) {
    // To (-π, π].
    let k = round_half_away(x / TAU);
    let mut y = x - k * TAU;
    // To [-π/2, π/2], recording the sign flip the reflection costs.
    let mut negate = false;
    if y > HALF_PI {
        y = PI - y;
        negate = true;
    } else if y < -HALF_PI {
        y = -PI - y;
        negate = true;
    }
    (y, negate)
}

/// `sin(x)`, x in radians. Deterministic across platforms.
///
/// Reflection about ±π/2 preserves `sin` in magnitude and sign, so the
/// `negate` flag from [`reduce`] is deliberately unused here — it exists for
/// [`cos`], which the same reflection *does* negate.
pub fn sin(x: f64) -> f64 {
    let (y, _) = reduce(x);
    sin_core(y)
}

/// `cos(x)`, x in radians. Deterministic across platforms.
pub fn cos(x: f64) -> f64 {
    let (y, negate) = reduce(x);
    let c = cos_core(y);
    if negate {
        -c
    } else {
        c
    }
}

/// `(sin(x), cos(x))` from one range reduction.
pub fn sin_cos(x: f64) -> (f64, f64) {
    let (y, negate) = reduce(x);
    let c = cos_core(y);
    (sin_core(y), if negate { -c } else { c })
}

/// Maclaurin series for `atan` on |z| ≤ tan(π/32) ≈ 0.0985, Horner in z².
///
/// Truncated after z¹⁷/17, whose value at that argument is ~8e-18.
fn atan_small(z: f64) -> f64 {
    let z2 = z * z;
    let p = 1.0 / 17.0;
    let p = p * z2 - 1.0 / 15.0;
    let p = p * z2 + 1.0 / 13.0;
    let p = p * z2 - 1.0 / 11.0;
    let p = p * z2 + 1.0 / 9.0;
    let p = p * z2 - 1.0 / 7.0;
    let p = p * z2 + 1.0 / 5.0;
    let p = p * z2 - 1.0 / 3.0;
    z + z * z2 * p
}

/// One application of the half-angle identity
/// `atan(z) = 2·atan(z / (1 + sqrt(1 + z²)))`.
///
/// `sqrt` is exactly rounded by IEEE-754, so this is as deterministic as the
/// arithmetic around it.
fn atan_halve(z: f64) -> f64 {
    z / (1.0 + (1.0 + z * z).sqrt())
}

/// `atan(z)`. Deterministic across platforms.
///
/// Above |z| = 1 the reciprocal identity is used, so the series is never
/// evaluated where it converges slowly; below it, three halvings shrink the
/// argument into [`atan_small`]'s range before the series runs.
pub fn atan(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z > 1.0 {
        return HALF_PI - atan(1.0 / z);
    }
    if z < -1.0 {
        return -HALF_PI - atan(1.0 / z);
    }
    let h = atan_halve(atan_halve(atan_halve(z)));
    8.0 * atan_small(h)
}

/// `atan2(y, x)`, the angle of the vector `(x, y)` in `(-π, π]`.
/// Deterministic across platforms.
///
/// `atan2(0, 0)` is `0.0`, matching `f64::atan2`.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x > 0.0 {
        atan(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            atan(y / x) + PI
        } else {
            atan(y / x) - PI
        }
    } else if y > 0.0 {
        HALF_PI
    } else if y < 0.0 {
        -HALF_PI
    } else {
        0.0
    }
}

/// `hypot(x, y)` without the platform's own implementation: scales by the
/// larger magnitude so the intermediate square never overflows, then one
/// exactly-rounded `sqrt`.
pub fn hypot(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let (big, small) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if big == 0.0 {
        return 0.0;
    }
    let r = small / big;
    big * (1.0 + r * r).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2001 arguments spanning ±20 radians — several full periods either side
    /// of zero, so range reduction and both reflection branches are hit.
    fn sweep() -> Vec<f64> {
        (0..2001).map(|i| -20.0 + (i as f64) * 0.02).collect()
    }

    #[test]
    fn sin_and_cos_agree_with_std_over_several_periods() {
        // Arrange
        let xs = sweep();

        // Act / Assert
        for x in xs {
            assert!((sin(x) - x.sin()).abs() < 1e-12, "sin({x})");
            assert!((cos(x) - x.cos()).abs() < 1e-12, "cos({x})");
        }
    }

    #[test]
    fn sin_cos_matches_the_separate_functions() {
        // Arrange
        let xs = sweep();

        // Act / Assert
        for x in xs {
            let (s, c) = sin_cos(x);
            assert_eq!(s, sin(x));
            assert_eq!(c, cos(x));
        }
    }

    #[test]
    fn atan_agrees_with_std_across_the_reciprocal_boundary() {
        // Arrange — dense through ±1 where the identity switches, plus tails.
        let mut zs: Vec<f64> = (0..401).map(|i| -2.0 + (i as f64) * 0.01).collect();
        zs.extend([-1e6, -1000.0, -50.0, 50.0, 1000.0, 1e6]);

        // Act / Assert
        for z in zs {
            assert!((atan(z) - z.atan()).abs() < 1e-12, "atan({z})");
        }
    }

    #[test]
    fn atan2_agrees_with_std_in_every_quadrant() {
        // Arrange
        let vals = [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0];

        // Act / Assert
        for y in vals {
            for x in vals {
                let got = atan2(y, x);
                let want = y.atan2(x);
                // std's atan2(0, -0.0) is +π while this returns +π/2 for the
                // x == 0 column; the column tested here uses exact +0.0, where
                // both agree.
                assert!((got - want).abs() < 1e-12, "atan2({y}, {x}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn atan2_of_the_origin_is_zero() {
        // Arrange / Act
        let got = atan2(0.0, 0.0);

        // Assert
        assert_eq!(got, 0.0);
    }

    #[test]
    fn hypot_agrees_with_std_including_a_zero_and_a_huge_component() {
        // Arrange
        let pairs = [(3.0, 4.0), (0.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (1e200, 1e200), (1e-200, 3e-200)];

        // Act / Assert
        for (x, y) in pairs {
            let got = hypot(x, y);
            let want = x.hypot(y);
            assert!((got - want).abs() <= 1e-12 * want.max(1.0), "hypot({x}, {y})");
        }
    }

    #[test]
    fn results_are_identical_when_recomputed() {
        // Arrange
        let xs = sweep();

        // Act
        let first: Vec<f64> = xs.iter().map(|x| sin(*x) + cos(*x) + atan(*x)).collect();
        let second: Vec<f64> = xs.iter().map(|x| sin(*x) + cos(*x) + atan(*x)).collect();

        // Assert — bit equality, not approximate.
        assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(),
                   second.iter().map(|v| v.to_bits()).collect::<Vec<u64>>());
    }
}
