//! Springs and sag (whole-bike sim spec §9, stride 3): the shock's stroke
//! and leverage through the rear wheel's travel, air springs with negative
//! chambers for the shock and the fork, the static load on each wheel from
//! the centre of mass, the sag those give, and the spring's unknowns fitted
//! to sag measured on the bike.
//!
//! The gas force is the pressure difference across the air piston times
//! its area, isothermal for a slow (static) compression and adiabatic for a
//! fast one (Dixon, *The Shock Absorber Handbook*, 2007, §7.4, PDF p. 287).
//! The force at the wheel is the shock's force times the stroke's rate with
//! travel, by virtual work, which is Cossalter's reduced stiffness
//! (*Motorcycle Dynamics*, 2006, §5.3, PDF pp. 177–179) for a linkage.
//!
//! Travel is the rear axle's vertical travel, metres, as [`SimBike::pose`]
//! takes it; a stroke is the shock's (or the fork's) shortening from top-out.

use crate::calibration::params::{number, Params};

use super::bike::SimBike;

/// Atmospheric pressure, Pa.
pub const ATM_PA: f64 = 101_325.0;
/// One psi, Pa.
pub const PSI_PA: f64 = 6_894.757;
/// Standard gravity, m/s².
pub const G: f64 = 9.806_65;
/// The polytropic index for a slow, static compression (isothermal).
pub const N_STATIC: f64 = 1.0;
/// The polytropic index for a fast compression over a bump (adiabatic, air).
pub const N_DYNAMIC: f64 = 1.4;

/// An air spring: a positive chamber the piston compresses and a negative
/// chamber it expands, equalised at top-out (the usual self-equalising
/// negative spring). SI units throughout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirSpring {
    /// Air piston area, m².
    pub area_m2: f64,
    /// Positive chamber volume at top-out, spacers already taken out, m³.
    pub positive_m3: f64,
    /// Negative chamber volume at top-out, m³.
    pub negative_m3: f64,
    /// Stroke, m.
    pub stroke_m: f64,
}

impl AirSpring {
    /// The force extending the spring, N, at stroke `x_m` with the chambers
    /// filled to `gauge_pa` (gauge) at top-out, compressing with polytropic
    /// index `n`.
    pub fn force_n(&self, gauge_pa: f64, x_m: f64, n: f64) -> f64 {
        super::laws::air_force_n(self.area_m2, self.positive_m3, self.negative_m3, gauge_pa, x_m, n)
    }

    /// The force, N, at stroke `x_m` when the spring sat at `x_ref_m` long
    /// enough to be at ambient temperature (isothermal from top-out there),
    /// then moved quickly from it with polytropic index `n`: the spring over
    /// a bump ridden from sag.
    pub fn force_about_n(&self, gauge_pa: f64, x_ref_m: f64, x_m: f64, n: f64) -> f64 {
        super::laws::air_force_about_n(self.area_m2, self.positive_m3, self.negative_m3, gauge_pa, x_ref_m, x_m, n)
    }

    /// The stroke, m, where the spring at `gauge_pa` holds `load_n` (slow,
    /// isothermal), clamped to [0, stroke].
    pub fn stroke_under(&self, gauge_pa: f64, load_n: f64) -> f64 {
        bisect(|x| self.force_n(gauge_pa, x, N_STATIC) - load_n, 0.0, self.stroke_m)
    }
}

/// The root of increasing `f` on [lo, hi] by bisection; an end when `f` has
/// no sign change there.
fn bisect(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let (mut a, mut b) = (lo, hi);
    if f(a) >= 0.0 {
        return a;
    }
    if f(b) <= 0.0 {
        return b;
    }
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        if f(m) < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// Which end of the bike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    /// The fork.
    Fork,
    /// The shock.
    Shock,
}

impl End {
    /// The parameter family's key prefix.
    pub fn prefix(self) -> &'static str {
        match self {
            End::Fork => "fork",
            End::Shock => "shock",
        }
    }
}

/// Air piston area when the parameters don't say, m²: a 230×65 shock's air
/// can (about 45 mm across), a 36 mm fork's air piston (about 28 mm).
pub fn default_area_m2(end: End) -> f64 {
    match end {
        End::Shock => 16.0e-4,
        End::Fork => 6.2e-4,
    }
}

/// The spring's chambers when the parameters don't say, as multiples of
/// the swept volume (area × stroke): positive, negative. Estimates to be
/// replaced by a fit to measured sag ([`fit`]); fork first, then shock.
const DEFAULT_RATIOS: [(f64, f64); 2] = [(2.5, 0.4), (2.6, 0.6)];

/// The spring at `end` from the bike's parameters (`*.piston_area_cm2`,
/// `*.positive_volume_cc`, `*.negative_volume_cc`, spacers or reducers and
/// their volume), with stroke `stroke_m`.
pub fn air_spring(p: &Params, end: End, stroke_m: f64) -> AirSpring {
    let k = end.prefix();
    let area_m2 = number(p, &format!("{k}.piston_area_cm2")).filter(|v| *v > 0.0).map_or(default_area_m2(end), |v| v * 1e-4);
    let swept = area_m2 * stroke_m;
    let (pr, nr) = DEFAULT_RATIOS[end as usize];
    let cc = |key: &str| number(p, &format!("{k}.{key}")).filter(|v| *v > 0.0).map(|v| v * 1e-6);
    let (spacers, per) = match end {
        End::Shock => (number(p, "shock.volume_spacers"), cc("spacer_cc").unwrap_or(3.0e-6)),
        End::Fork => (number(p, "fork.volume_reducers"), cc("reducer_cc").unwrap_or(10.0e-6)),
    };
    let positive = cc("positive_volume_cc").unwrap_or(pr * swept) - spacers.unwrap_or(0.0).max(0.0) * per;
    AirSpring { area_m2, positive_m3: positive.max(1.05 * swept), negative_m3: cc("negative_volume_cc").unwrap_or(nr * swept), stroke_m }
}

/// The shock's eye-to-eye length at `travel_m`, m; `None` without both eyes.
pub fn shock_length(bike: &SimBike, travel_m: f64) -> Option<f64> {
    let (f, r) = (bike.points.shock_frame?, bike.points.shock_rocker?);
    let pose = bike.pose(travel_m);
    let e = bike.rocker_point(&pose, r);
    Some((e[0] - f[0]).hypot(e[1] - f[1]))
}

/// The shock's stroke at `travel_m`, m: how much shorter it is than at top-out.
pub fn shock_stroke(bike: &SimBike, travel_m: f64) -> Option<f64> {
    Some(shock_length(bike, 0.0)? - shock_length(bike, travel_m)?)
}

/// The leverage ratio at `travel_m`: wheel travel per shock stroke (a
/// central difference, one-sided at the ends).
pub fn leverage_ratio(bike: &SimBike, travel_m: f64) -> Option<f64> {
    let h = 1e-4;
    let (a, b) = ((travel_m - h).max(0.0), (travel_m + h).min(bike.travel_max_m));
    let ds = shock_stroke(bike, b)? - shock_stroke(bike, a)?;
    (ds.abs() > 1e-12).then(|| (b - a) / ds)
}

/// The static vertical load on each wheel, N, (front, rear): `mass_kg` on
/// level ground with its centre of mass `cog_x_m` ahead of the rear axle.
pub fn wheel_loads_n(mass_kg: f64, cog_x_m: f64, wheelbase_m: f64) -> (f64, f64) {
    let w = mass_kg * G;
    let front = w * (cog_x_m / wheelbase_m).clamp(0.0, 1.0);
    (front, w - front)
}

/// The rear wheel's vertical force, N, with the shock at `gauge_pa`, at
/// `travel_m`, compressing with polytropic index `n`.
pub fn rear_wheel_force_n(bike: &SimBike, spring: &AirSpring, gauge_pa: f64, travel_m: f64, n: f64) -> Option<f64> {
    Some(spring.force_n(gauge_pa, shock_stroke(bike, travel_m)?, n) / leverage_ratio(bike, travel_m)?)
}

/// The rear travel, m, where the shock at `gauge_pa` holds `load_n` at the wheel.
pub fn rear_sag_travel_m(bike: &SimBike, spring: &AirSpring, gauge_pa: f64, load_n: f64) -> Option<f64> {
    rear_wheel_force_n(bike, spring, gauge_pa, 0.0, N_STATIC)?;
    Some(bisect(|s| rear_wheel_force_n(bike, spring, gauge_pa, s, N_STATIC).unwrap_or(f64::INFINITY) - load_n, 0.0, bike.travel_max_m))
}

/// The share of a vertical load at the front axle that pushes along the
/// fork's legs, with the head angle `head_angle_deg` from horizontal.
pub fn fork_axial_share(head_angle_deg: f64) -> f64 {
    head_angle_deg.to_radians().sin()
}

/// A sag measurement: the spring's pressure, the load it held and the stroke
/// it sagged to, with the spacers fitted then.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SagTest {
    /// Gauge pressure, Pa.
    pub gauge_pa: f64,
    /// The load along the spring's axis, at the shock (rear: at the wheel,
    /// turned into shock force through the leverage at the measured stroke).
    pub load_n: f64,
    /// Measured stroke, m.
    pub stroke_m: f64,
    /// Spacers or reducers fitted.
    pub spacers: f64,
}

/// What [`fit`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    /// The fitted spring (spacers of the first test taken out).
    pub spring: AirSpring,
    /// Root-mean-square difference between measured and predicted stroke, m.
    pub rms_m: f64,
    /// How many unknowns were fitted: 1 (area) or 2 (area and positive volume).
    pub unknowns: usize,
}

/// Fits the spring's unknowns to sag measurements (the "identify" step of
/// the measure → estimate → identify → simulate loop). One measurement
/// fixes the air piston's effective area (the force's level), keeping the
/// chambers' ratio to the swept volume; two or more at different loads or
/// pressures fix the positive chamber's volume too (the progression).
/// `spacer_m3` is the volume one spacer takes. `None` without a measurement.
pub fn fit(start: &AirSpring, tests: &[SagTest], spacer_m3: f64) -> Option<Fit> {
    if tests.is_empty() {
        return None;
    }
    let base_spacers = tests[0].spacers;
    let swept0 = start.area_m2 * start.stroke_m;
    let (pr, nr) = (start.positive_m3 / swept0, start.negative_m3 / swept0);
    let make = |area: f64, pos_ratio: f64| {
        let swept = area * start.stroke_m;
        AirSpring { area_m2: area, positive_m3: pos_ratio * swept, negative_m3: nr * swept, stroke_m: start.stroke_m }
    };
    let cost = |s: &AirSpring| -> f64 {
        tests
            .iter()
            .map(|t| {
                let extra = (t.spacers - base_spacers) * spacer_m3;
                let at = AirSpring { positive_m3: (s.positive_m3 - extra).max(1.05 * s.area_m2 * s.stroke_m), ..*s };
                (at.stroke_under(t.gauge_pa, t.load_n) - t.stroke_m).powi(2)
            })
            .sum::<f64>()
    };
    let distinct = tests.iter().any(|t| (t.load_n - tests[0].load_n).abs() > 1.0 || (t.gauge_pa - tests[0].gauge_pa).abs() > 100.0);
    let best_area = |pos_ratio: f64| golden(|ln_a| cost(&make(ln_a.exp(), pos_ratio)), (start.area_m2 / 4.0).ln(), (start.area_m2 * 4.0).ln());
    let (area, pos_ratio, unknowns) = if tests.len() >= 2 && distinct {
        let r = golden(|r| cost(&make(best_area(r).exp(), r)), 1.15, 8.0);
        (best_area(r).exp(), r, 2)
    } else {
        (best_area(pr).exp(), pr, 1)
    };
    let spring = make(area, pos_ratio);
    Some(Fit { spring, rms_m: (cost(&spring) / tests.len() as f64).sqrt(), unknowns })
}

/// The minimiser of unimodal `f` on [a, b], golden-section search.
fn golden(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let r = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::bike::tests::{gt_force_drivetrain, gt_force_linkage};

    fn shock() -> AirSpring {
        AirSpring { area_m2: 12e-4, positive_m3: 200e-6, negative_m3: 50e-6, stroke_m: 0.065 }
    }

    #[test]
    fn air_spring_at_top_out_equalised_pushes_nothing() {
        // Arrange
        let s = shock();

        // Act
        let f = s.force_n(220.0 * PSI_PA, 0.0, N_STATIC);

        // Assert
        assert!(f.abs() < 1e-9, "{f}");
    }

    #[test]
    fn air_spring_force_rises_through_the_stroke_and_faster_when_adiabatic() {
        // Arrange
        let s = shock();
        let p = 220.0 * PSI_PA;

        // Act
        let (mid, end) = (s.force_n(p, 0.03, N_STATIC), s.force_n(p, 0.06, N_STATIC));
        let fast = s.force_n(p, 0.06, N_DYNAMIC);

        // Assert
        assert!(0.0 < mid && mid < end && end < fast, "{mid} {end} {fast}");
    }

    #[test]
    fn air_spring_isothermal_positive_chamber_matches_boyle() {
        // Arrange: no negative chamber to speak of (huge, so its pressure barely falls)
        let s = AirSpring { negative_m3: 1.0, ..shock() };
        let p0 = 200.0 * PSI_PA + ATM_PA;
        let x = 0.04;

        // Act
        let f = s.force_n(200.0 * PSI_PA, x, N_STATIC);

        // Assert: A (p0 V/(V - A x) - p0), the negative side's tiny drop aside
        let want = s.area_m2 * (p0 * s.positive_m3 / (s.positive_m3 - s.area_m2 * x) - p0);
        assert!((f - want).abs() / want < 1e-3, "{f} {want}");
    }

    #[test]
    fn stroke_under_holds_the_load() {
        // Arrange
        let s = shock();
        let p = 200.0 * PSI_PA;

        // Act
        let x = s.stroke_under(p, 1500.0);

        // Assert
        assert!((s.force_n(p, x, N_STATIC) - 1500.0).abs() < 1e-6, "{x}");
    }

    #[test]
    fn leverage_ratio_is_travel_over_stroke_on_average() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let n = 400;

        // Act: integrate 1 / LR over the travel
        let h = bike.travel_max_m / n as f64;
        let stroke: f64 = (0..n).map(|i| h / leverage_ratio(&bike, (i as f64 + 0.5) * h).unwrap()).sum();

        // Assert
        let want = shock_stroke(&bike, bike.travel_max_m).unwrap();
        assert!((stroke - want).abs() < 1e-5, "{stroke} {want}");
    }

    #[test]
    fn rear_sag_wheel_force_equals_the_load() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let stroke = shock_stroke(&bike, bike.travel_max_m).unwrap();
        let s = AirSpring { stroke_m: stroke, ..shock() };
        let p = 220.0 * PSI_PA;

        // Act
        let sag = rear_sag_travel_m(&bike, &s, p, 550.0).unwrap();

        // Assert
        assert!(sag > 0.0 && sag < bike.travel_max_m);
        assert!((rear_wheel_force_n(&bike, &s, p, sag, N_STATIC).unwrap() - 550.0).abs() < 1e-3);
    }

    #[test]
    fn wheel_loads_split_by_the_centre_of_mass() {
        // Arrange / Act
        let (f, r) = wheel_loads_n(100.0, 0.512, 1.28);

        // Assert
        assert!((f - 0.4 * 100.0 * G).abs() < 1e-9 && (f + r - 100.0 * G).abs() < 1e-9);
    }

    #[test]
    fn fit_one_measurement_recovers_the_area() {
        // Arrange: a "true" spring and the sag it gives
        let truth = AirSpring { area_m2: 10e-4, positive_m3: 2.6 * 10e-4 * 0.065, negative_m3: 0.6 * 10e-4 * 0.065, stroke_m: 0.065 };
        let p = 220.0 * PSI_PA;
        let tests = [SagTest { gauge_pa: p, load_n: 1400.0, stroke_m: truth.stroke_under(p, 1400.0), spacers: 0.0 }];
        let start = AirSpring { area_m2: 12e-4, positive_m3: 2.6 * 12e-4 * 0.065, negative_m3: 0.6 * 12e-4 * 0.065, stroke_m: 0.065 };

        // Act
        let fit = fit(&start, &tests, 3e-6).unwrap();

        // Assert
        assert_eq!(fit.unknowns, 1);
        assert!((fit.spring.area_m2 - truth.area_m2).abs() / truth.area_m2 < 1e-3, "{:?}", fit.spring);
        assert!(fit.rms_m < 1e-5);
    }

    #[test]
    fn fit_two_loads_recover_area_and_positive_volume() {
        // Arrange
        let truth = AirSpring { area_m2: 11e-4, positive_m3: 3.4 * 11e-4 * 0.065, negative_m3: 0.6 * 11e-4 * 0.065, stroke_m: 0.065 };
        let p = 220.0 * PSI_PA;
        let tests: Vec<SagTest> = [1300.0, 1600.0].iter().map(|&l| SagTest { gauge_pa: p, load_n: l, stroke_m: truth.stroke_under(p, l), spacers: 0.0 }).collect();
        let start = AirSpring { area_m2: 12e-4, positive_m3: 2.6 * 12e-4 * 0.065, negative_m3: 0.6 * 12e-4 * 0.065, stroke_m: 0.065 };

        // Act
        let fit = fit(&start, &tests, 3e-6).unwrap();

        // Assert
        assert_eq!(fit.unknowns, 2);
        assert!(fit.rms_m < 2e-4, "{}", fit.rms_m);
        assert!((fit.spring.area_m2 - truth.area_m2).abs() / truth.area_m2 < 0.05, "{:?}", fit.spring);
    }

    #[test]
    fn air_spring_from_params_takes_spacers_out_of_the_positive_chamber() {
        // Arrange
        let mut p = Params::new();
        p.insert("shock.positive_volume_cc".into(), serde_json::json!(200.0));
        p.insert("shock.spacer_cc".into(), serde_json::json!(5.0));
        p.insert("shock.volume_spacers".into(), serde_json::json!(2));

        // Act
        let s = air_spring(&p, End::Shock, 0.065);

        // Assert
        assert!((s.positive_m3 - 190e-6).abs() < 1e-12, "{}", s.positive_m3);
    }
}
