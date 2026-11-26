//! The force laws as plain functions generic over [`Real`], shared by the
//! simulator and the physics estimator (force-model spec phase F0,
//! idl1-app `docs/superpowers/specs/2026-11-25-idl1-force-model-DRAFT.md`
//! §3.2; whole-bike sim spec §5.3).
//!
//! Every switch (a tyre leaving the ground, a damper's knee, compression
//! against rebound, a stop) takes a smoothing width `w` in the switch's own
//! unit. At `w = 0` the switch is exact, which is what the simulator uses;
//! the estimator passes a small width so Gauss-Newton (and later the NPU)
//! sees smooth functions with no branches on the data. The parameters are
//! [`Real`] too, so a fitted damper or tyre gets its derivatives from the
//! same call.

use crate::estimate::physics::la::{c, Real};

/// `max(x, 0)`, smoothed over width `w` (same unit as `x`) as a softplus,
/// `w ln(1 + e^(x/w))`. Exact at `w = 0`.
pub fn relu<S: Real>(x: S, w: f64) -> S {
    if w <= 0.0 {
        return if x.re() > 0.0 { x } else { c(0.0) };
    }
    let u = x.re() / w;
    if u > 30.0 {
        x
    } else if u < -30.0 {
        (x / w).exp() * w
    } else {
        ((x / w).exp() + 1.0).ln() * w
    }
}

/// The step `x > 0`, smoothed over width `w` as a logistic, `1 / (1 + e^(−x/w))`.
/// Exact at `w = 0` (0 at `x = 0`).
pub fn step<S: Real>(x: S, w: f64) -> S {
    if w <= 0.0 {
        return c(if x.re() > 0.0 { 1.0 } else { 0.0 });
    }
    let u = x.re() / w;
    if u > 40.0 {
        c(1.0)
    } else if u < -40.0 {
        (x / w).exp()
    } else {
        ((-x / w).exp() + 1.0).recip()
    }
}

/// A stop: stiffness `k_n_m` (N/m) once `x` (m) is outside `[lo_m, hi_m]`,
/// pushing it back in, N (positive pushes `x` down). Each side is a
/// [`relu`] of width `w_m`.
pub fn stop_n<S: Real>(x: S, lo_m: f64, hi_m: f64, k_n_m: f64, w_m: f64) -> S {
    (relu(x - hi_m, w_m) - relu(-(x - lo_m), w_m)) * k_n_m
}

/// A damper's force–speed curve: linear up to a knee, then a shallower line
/// (the high-speed circuit), in compression and in rebound. Damping in
/// N·s/m, the knee in m/s. The same law as [`super::ride::Damper`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DamperLaw<S> {
    /// Low-speed compression, N·s/m.
    pub lsc_ns_m: S,
    /// High-speed compression past the knee, N·s/m.
    pub hsc_ns_m: S,
    /// Low-speed rebound, N·s/m.
    pub lsr_ns_m: S,
    /// High-speed rebound past the knee, N·s/m.
    pub hsr_ns_m: S,
    /// Speed where high speed starts, m/s.
    pub knee_mps: S,
}

impl DamperLaw<f64> {
    /// The same law with its numbers lifted to `T` (e.g. a dual type).
    pub fn lift<T: Real>(&self) -> DamperLaw<T> {
        DamperLaw { lsc_ns_m: c(self.lsc_ns_m), hsc_ns_m: c(self.hsc_ns_m), lsr_ns_m: c(self.lsr_ns_m), hsr_ns_m: c(self.hsr_ns_m), knee_mps: c(self.knee_mps) }
    }
}

impl<S: Real> DamperLaw<S> {
    /// The damping force, N, resisting speed `v_mps` (positive compressing),
    /// with the knee and the compression / rebound switch smoothed over
    /// `w_mps` (m/s; 0 exact).
    pub fn force_n(&self, v_mps: S, w_mps: f64) -> S {
        // each side's curve continued linearly through zero, then blended by
        // the direction: at w = 0 this is exactly the piecewise law
        let comp = v_mps * self.lsc_ns_m - (self.lsc_ns_m - self.hsc_ns_m) * relu(v_mps - self.knee_mps, w_mps);
        let reb = v_mps * self.lsr_ns_m + (self.lsr_ns_m - self.hsr_ns_m) * relu(-v_mps - self.knee_mps, w_mps);
        let k = step(v_mps, w_mps);
        comp * k + reb * (-k + 1.0)
    }
}

/// Seal friction, N: Coulomb `f_c_n` against speed `v_mps` (positive
/// compressing), smoothed as `f_c tanh(v / v0)` with `v0_mps` (m/s; 0 exact,
/// with 0 at rest).
pub fn friction_n<S: Real>(f_c_n: S, v_mps: S, v0_mps: f64) -> S {
    if v0_mps <= 0.0 {
        let s = v_mps.re();
        return if s > 0.0 { f_c_n } else if s < 0.0 { -f_c_n } else { c(0.0) };
    }
    // past 20, tanh is 1 to 16 digits; and a dual tanh there divides cosh's infinities (NaN)
    let u = v_mps.re() / v0_mps;
    if u.abs() > 20.0 {
        return f_c_n * u.signum();
    }
    f_c_n * (v_mps / v0_mps).tanh()
}

/// An air spring's chambers, SI: the law of [`super::suspension::AirSpring`]
/// with the gauge pressure (and so its drift with heat) a [`Real`].
///
/// `area_m2` piston area, `positive_m3` / `negative_m3` the chambers at
/// top-out (spacers out), compressing with polytropic index `n`.
pub fn air_force_n<S: Real>(area_m2: f64, positive_m3: f64, negative_m3: f64, gauge_pa: S, x_m: S, n: f64) -> S {
    let p0 = gauge_pa + super::suspension::ATM_PA;
    let swept = x_m * area_m2;
    let pos = floor(-swept + positive_m3, 0.01 * positive_m3);
    let neg = floor(swept + negative_m3, 1e-9);
    let p_pos = p0 * (pos.recip() * positive_m3).powf(n);
    let p_neg = p0 * (neg.recip() * negative_m3).powf(n);
    (p_pos - p_neg) * area_m2
}

/// The air spring's force, N, at stroke `x_m` after resting at `x_ref_m`
/// (isothermal from top-out there), then moved quickly with index `n`: the
/// spring over a bump ridden from sag. The law of
/// [`super::suspension::AirSpring::force_about_n`].
pub fn air_force_about_n<S: Real>(area_m2: f64, positive_m3: f64, negative_m3: f64, gauge_pa: S, x_ref_m: S, x_m: S, n: f64) -> S {
    let p0 = gauge_pa + super::suspension::ATM_PA;
    let lo = 0.01 * positive_m3;
    let (pos_ref, neg_ref) = (floor(-(x_ref_m * area_m2) + positive_m3, lo), floor(x_ref_m * area_m2 + negative_m3, 1e-9));
    let (pp, pn) = (p0 * positive_m3 / pos_ref, p0 * negative_m3 / neg_ref);
    let (pos, neg) = (floor(-(x_m * area_m2) + positive_m3, lo), floor(x_m * area_m2 + negative_m3, 1e-9));
    (pp * (pos_ref / pos).powf(n) - pn * (neg_ref / neg).powf(n)) * area_m2
}

/// `max(x, lo)` on the value (the derivative is cut below `lo`): the air
/// spring's guard against a chamber of no volume, never reached in a sane
/// solve.
fn floor<S: Real>(x: S, lo: f64) -> S {
    if x.re() > lo { x } else { c(lo) }
}

/// A tyre's radial law: air and casing as a linear spring, a foam insert
/// that is touched once the deflection closes the gap above it and stiffens
/// toward solid, and the rim pinching the tyre as a stiff stop. The law of
/// [`super::tyre::Tyre::force`], with the fitted parts [`Real`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TyreLaw<S> {
    /// Air and casing stiffness, N/m (`2π (p + p_c) √(R r)`, [`super::tyre::Tyre::stiffness_n_m`]).
    pub stiffness_n_m: S,
    /// Deflection where the insert is touched, m (None: no insert).
    pub insert_touch_m: Option<f64>,
    /// The insert's initial crush stiffness, N/m.
    pub insert_n_m: S,
    /// Crush where the insert is solid, m (70 % of its height).
    pub insert_solid_m: f64,
    /// Deflection where the rim pinches the tyre, m.
    pub rim_m: f64,
    /// Stiffness of the rim's stop, N/m.
    pub rim_n_m: f64,
}

impl<S: Real> TyreLaw<S> {
    /// The law of a simulator tyre (its fixed numbers lifted to `S`).
    pub fn from_tyre(t: &super::tyre::Tyre) -> Self {
        TyreLaw {
            stiffness_n_m: c(t.stiffness_n_m()),
            insert_touch_m: t.insert.map(|i| (t.section_height_m - t.casing_thickness_m - i.height_m).max(0.0)),
            insert_n_m: c(t.insert.map_or(0.0, |i| i.stiffness_n_m)),
            insert_solid_m: t.insert.map_or(1.0, |i| 0.7 * i.height_m),
            rim_m: t.rim_strike_m(),
            rim_n_m: super::tyre::RIM_STOP_N_M,
        }
    }

    /// The radial force, N, at deflection `d_m` (m, positive into the tyre;
    /// tension-free), with lift-off and each contact smoothed over `w_m` (m;
    /// 0 exact). No damping.
    pub fn force_n(&self, d_m: S, w_m: f64) -> S {
        let mut f = self.stiffness_n_m * relu(d_m, w_m);
        if let Some(touch) = self.insert_touch_m {
            let e = relu(d_m - touch, w_m);
            // crushed toward solid; past 98 % of it the foam is a stop
            let cap = 0.98 * self.insert_solid_m;
            let crush = if e.re() < cap { e } else { c(cap) };
            f += self.insert_n_m * crush / (-crush / self.insert_solid_m + 1.0) + relu(e - cap, w_m) * self.rim_n_m;
        }
        f + relu(d_m - self.rim_m, w_m) * self.rim_n_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::ride::Damper;
    use crate::sim::suspension::{AirSpring, N_DYNAMIC, PSI_PA};
    use crate::sim::tyre::Tyre;
    use num_dual::Dual64;

    fn damper_law(d: &Damper) -> DamperLaw<f64> {
        DamperLaw { lsc_ns_m: d.lsc_ns_m, hsc_ns_m: d.hsc_ns_m, lsr_ns_m: d.lsr_ns_m, hsr_ns_m: d.hsr_ns_m, knee_mps: d.knee_mps }
    }

    #[test]
    fn relu_and_step_width_zero_exact_smooth_close() {
        // Arrange
        let xs: [f64; 5] = [-0.3, -0.01, 0.0, 0.01, 0.3];

        // Act / Assert
        for x in xs {
            assert_eq!(relu(x, 0.0), x.max(0.0));
            assert_eq!(step(x, 0.0), if x > 0.0 { 1.0 } else { 0.0 });
            assert!((relu(x, 1e-3) - x.max(0.0)).abs() < 1e-3 * 0.7, "{x}");
        }
        assert!((relu(0.0, 1e-3) - 1e-3 * std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn damper_law_width_zero_is_the_piecewise_law() {
        // Arrange
        let law = damper_law(&Damper::default());

        // Act / Assert: linear below the knee, the high-speed slope past it, each direction its own
        assert!((law.force_n(0.1, 0.0) - 4000.0 * 0.1).abs() < 1e-9);
        assert!((law.force_n(0.4, 0.0) - (4000.0 * 0.15 + 1200.0 * 0.25)).abs() < 1e-9);
        assert!((law.force_n(-0.1, 0.0) + 9000.0 * 0.1).abs() < 1e-9);
        assert!((law.force_n(-0.4, 0.0) + (9000.0 * 0.15 + 3500.0 * 0.25)).abs() < 1e-9);
        assert_eq!(law.force_n(0.0, 0.0), 0.0);
    }

    #[test]
    fn damper_law_smoothed_is_close_and_zero_at_rest() {
        // Arrange
        let law = damper_law(&Damper::default());

        // Act
        let rest = law.force_n(0.0, 0.005);

        // Assert
        assert!(rest.abs() < 1e-6);
        for v in [-1.0, -0.3, 0.3, 1.0] {
            assert!((law.force_n(v, 0.005) - law.force_n(v, 0.0)).abs() < 0.01 * law.force_n(v, 0.0).abs(), "{v}");
        }
    }

    #[test]
    fn damper_law_dual_derivative_matches_finite_difference() {
        // Arrange
        let law = damper_law(&Damper::default());
        let (v, h, w) = (0.12, 1e-7, 0.01);

        // Act
        let ad = law.lift::<Dual64>().force_n(Dual64::from(v).derivative(), w).eps;
        let fd = (law.force_n(v + h, w) - law.force_n(v - h, w)) / (2.0 * h);

        // Assert
        assert!((ad - fd).abs() < 1e-3 * fd.abs(), "{ad} {fd}");
    }

    #[test]
    fn friction_width_zero_is_coulomb_and_smooth_is_tanh() {
        // Arrange / Act / Assert
        assert_eq!(friction_n(30.0f64, 0.2, 0.0), 30.0);
        assert_eq!(friction_n(30.0f64, -0.2, 0.0), -30.0);
        assert_eq!(friction_n(30.0f64, 0.0, 0.0), 0.0);
        assert!((friction_n(30.0f64, 0.01, 0.01) - 30.0 * 1f64.tanh()).abs() < 1e-12);
    }

    #[test]
    fn stop_pushes_back_only_outside_its_range() {
        // Arrange / Act / Assert
        assert_eq!(stop_n(0.05f64, 0.0, 0.17, 2e6, 0.0), 0.0);
        assert!((stop_n(0.171f64, 0.0, 0.17, 2e6, 0.0) - 2000.0).abs() < 1e-6);
        assert!((stop_n(-0.001f64, 0.0, 0.17, 2e6, 0.0) + 2000.0).abs() < 1e-6);
    }

    #[test]
    fn air_law_balanced_at_top_out_and_about_its_rest_point_isothermal() {
        // Arrange
        let s = AirSpring { area_m2: 0.00095, positive_m3: 1.6e-4, negative_m3: 9e-5, stroke_m: 0.065 };
        let p = 220.0 * PSI_PA;

        // Act
        let top = air_force_n(s.area_m2, s.positive_m3, s.negative_m3, p, 0.0, N_DYNAMIC);
        let rest = air_force_about_n(s.area_m2, s.positive_m3, s.negative_m3, p, 0.018, 0.018, N_DYNAMIC);
        let iso = air_force_n(s.area_m2, s.positive_m3, s.negative_m3, p, 0.018, 1.0);
        let fast = air_force_about_n(s.area_m2, s.positive_m3, s.negative_m3, p, 0.018, 0.05, N_DYNAMIC);
        let slow = air_force_about_n(s.area_m2, s.positive_m3, s.negative_m3, p, 0.018, 0.05, 1.0);

        // Assert: equal chambers at top-out; at rest the spring is at its isothermal force; fast is stiffer
        assert!(top.abs() < 1e-9);
        assert!((rest - iso).abs() < 1e-6 * iso);
        assert!(fast > slow);
    }

    #[test]
    fn tyre_law_linear_until_the_insert_then_stiffer_then_the_rim() {
        // Arrange
        let t = Tyre::mtb_29(2.5, 24.0, true, true);
        let law = TyreLaw::<f64>::from_tyre(&t);
        let touch = law.insert_touch_m.unwrap();

        // Act
        let below = law.force_n(0.5 * touch, 0.0);
        let on = law.force_n(touch + 0.005, 0.0) - law.stiffness_n_m * (touch + 0.005);
        let rim = law.force_n(law.rim_m + 0.001, 0.0) - law.force_n(law.rim_m, 0.0);

        // Assert
        assert!((below - law.stiffness_n_m * 0.5 * touch).abs() < 1e-9);
        assert!(on > 0.0);
        assert!(rim > 0.9 * law.rim_n_m * 0.001);
        assert_eq!(law.force_n(-0.01, 0.0), 0.0);
    }

    #[test]
    fn friction_dual_at_a_fast_stroke_is_finite_and_flat() {
        // Arrange: 4 m/s against a 0.01 m/s smoothing (a fork bottoming out)
        let v = Dual64::from(4.0).derivative();

        // Act
        let f = friction_n(Dual64::from(30.0), v, 0.01);

        // Assert
        assert_eq!(f.re, 30.0);
        assert!(f.eps.is_finite() && f.eps.abs() < 1e-9);
    }
}
