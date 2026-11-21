//! The tyre between the wheel and the ground: its radial force through
//! deflection (air, casing, insert, the rim striking), and the ground it
//! feels, filtered by the contact patch (enveloping).
//!
//! **Radial force.** The air holds the load through the contact patch, so
//! the force is pressure × patch area. A torus of radius `R` (the tyre's
//! outer radius) and section radius `r` (half the casing width) pressed `δ`
//! into a flat touches an ellipse of half-axes √(2Rδ) and √(2rδ), area
//! 2πδ√(Rr): a stiffness 2π(p + p_c)√(Rr), linear in pressure, with the
//! casing's own stiffness `p_c` written as an equivalent pressure. A 29 ×
//! 2.5" tyre at 25 psi gives about 145 kN/m with a DH casing, inside the
//! 100-200 kN/m Cossalter gives for motorcycle carcasses (Motorcycle
//! Dynamics §2.8, PDF p. 84).
//!
//! **Insert.** A foam insert of radial height `h_i` sits on the rim; it is
//! touched once the deflection closes the gap above it and stiffens as it
//! is crushed (`k e / (1 − e/e_max)`, solid at 70 % of its height).
//!
//! **Rim strike.** At the section height less both casings the rim pinches
//! the tyre on the ground: a stiff stop (2 MN/m), counted as a rim strike.
//!
//! **Enveloping** (Pacejka, Tyre and Vehicle Dynamics §10.1, PDF pp. 498-506):
//! a tyre doesn't follow the ground's profile but an effective road plane.
//! The ground is first rolled by a rigid circle of the tyre's radius (the
//! basic curve of a step then has length √(r₀² − (r₀ − h)²), eq. 10.6), then
//! read by a two-point follower spaced `l_s`, about 80 % of the contact
//! length (p. 506): the effective height is the follower's midpoint, its
//! slope the follower's angle. The circle stands in for the fitted ellipse
//! cams of Schmeitz's model; the parameters aren't fitted to MTB tyres.

/// The rim's stop stiffness when it pinches the tyre on the ground, N/m.
pub const RIM_STOP_N_M: f64 = 2.0e6;

/// Pa per psi.
const PSI_PA: f64 = 6_894.757;

/// A foam insert on the rim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Insert {
    /// Radial height from the rim bed, m.
    pub height_m: f64,
    /// Initial crush stiffness, N/m.
    pub stiffness_n_m: f64,
}

/// One tyre on its rim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tyre {
    /// Outer radius, unloaded, m.
    pub radius_m: f64,
    /// Casing width, m.
    pub width_m: f64,
    /// Section height, rim bead seat to tread, m.
    pub section_height_m: f64,
    /// Thickness of one casing wall under the tread, m.
    pub casing_thickness_m: f64,
    /// Gauge pressure, Pa.
    pub gauge_pa: f64,
    /// The casing's own stiffness as an equivalent pressure, Pa.
    pub casing_pa: f64,
    /// The insert, if fitted.
    pub insert: Option<Insert>,
    /// Radial damping, N·s/m.
    pub damping_ns_m: f64,
}

/// What the tyre is doing at a deflection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TyreForce {
    /// Radial force, N (0 off the ground).
    pub force_n: f64,
    /// The insert is touched.
    pub on_insert: bool,
    /// The rim is pinching the tyre on the ground.
    pub rim_strike: bool,
}

impl Tyre {
    /// A 29" MTB tyre of `width_in` inches at `psi`: section height 0.9 ×
    /// the width, 1.5 mm casing walls (3 mm for a DH casing), the casing worth
    /// 3 psi (6 psi DH), and an insert 25 mm tall with a 60 kN/m crush when
    /// `insert` (all assumed: no MTB tyre has been measured yet).
    pub fn mtb_29(width_in: f64, psi: f64, dh_casing: bool, insert: bool) -> Tyre {
        let width_m = width_in.clamp(1.5, 3.5) * 0.0254;
        let section_height_m = 0.9 * width_m;
        Tyre {
            radius_m: 0.3683 - 0.0635 + section_height_m,
            width_m,
            section_height_m,
            casing_thickness_m: if dh_casing { 0.003 } else { 0.0015 },
            gauge_pa: psi.max(0.0) * PSI_PA,
            casing_pa: if dh_casing { 6.0 } else { 3.0 } * PSI_PA,
            insert: insert.then_some(Insert { height_m: 0.025, stiffness_n_m: 60_000.0 }),
            damping_ns_m: 100.0,
        }
    }

    /// The air and casing's stiffness, N/m.
    pub fn stiffness_n_m(&self) -> f64 {
        std::f64::consts::TAU * (self.gauge_pa + self.casing_pa) * (self.radius_m * 0.5 * self.width_m).sqrt()
    }

    /// Deflection at which the rim pinches the tyre, m.
    pub fn rim_strike_m(&self) -> f64 {
        (self.section_height_m - 2.0 * self.casing_thickness_m).max(0.005)
    }

    /// Radial force at deflection `d_m` (no damping).
    pub fn force(&self, d_m: f64) -> TyreForce {
        if d_m <= 0.0 {
            return TyreForce { force_n: 0.0, on_insert: false, rim_strike: false };
        }
        let rim = self.rim_strike_m();
        let mut f = self.stiffness_n_m() * d_m;
        let mut on_insert = false;
        if let Some(i) = self.insert {
            // the gap above the insert closes, then the foam is crushed toward solid
            let touch = (self.section_height_m - self.casing_thickness_m - i.height_m).max(0.0);
            let e = d_m - touch;
            if e > 0.0 {
                on_insert = true;
                let e_max = 0.7 * i.height_m;
                let crush = e.min(0.98 * e_max);
                f += i.stiffness_n_m * crush / (1.0 - crush / e_max) + if e > crush { RIM_STOP_N_M * (e - crush) } else { 0.0 };
            }
        }
        let rim_strike = d_m > rim;
        if rim_strike {
            f += RIM_STOP_N_M * (d_m - rim);
        }
        TyreForce { force_n: f, on_insert, rim_strike }
    }

    /// Deflection holding `load_n` statically, m.
    pub fn deflection_under(&self, load_n: f64) -> f64 {
        let (mut a, mut b) = (0.0, self.section_height_m + 0.05);
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if self.force(m).force_n < load_n {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    }

    /// Contact length at deflection `d_m`, m: the chord of the outer circle.
    pub fn contact_length_m(&self, d_m: f64) -> f64 {
        let d = d_m.clamp(0.0, self.radius_m);
        2.0 * (2.0 * self.radius_m * d - d * d).sqrt()
    }
}

/// The ground as the tyre feels it, sampled along the trail.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    /// Distance of the first sample, m.
    pub from_m: f64,
    /// Sample spacing, m.
    pub step_m: f64,
    /// Effective height per sample, m.
    pub height_m: Vec<f64>,
    /// Effective slope per sample (rise over run).
    pub slope: Vec<f64>,
}

impl Envelope {
    /// The ground `ground(x)` from `from_m` to `to_m` as `tyre` feels it at a
    /// static deflection `static_m` (which sets the follower's spacing).
    pub fn new(tyre: &Tyre, ground: &dyn Fn(f64) -> f64, from_m: f64, to_m: f64, static_m: f64) -> Envelope {
        let step = 0.002;
        let r = tyre.radius_m.max(0.05);
        let ls = (0.8 * tyre.contact_length_m(static_m)).max(step);
        let half = (ls / 2.0 / step).round() as i64;
        // the rigid circle's centre height (less r) over the ground, on a grid padded by the follower
        let n = ((to_m - from_m) / step).ceil().max(1.0) as i64 + 1;
        let reach = (r / step).ceil() as i64;
        let raw: Vec<f64> = (-reach - half..n + reach + half).map(|i| ground(from_m + i as f64 * step)).collect();
        let off = reach + half;
        let cam = |i: i64| -> f64 {
            let mut best = f64::NEG_INFINITY;
            for j in -reach..=reach {
                let k = i + j + off;
                if k < 0 || k as usize >= raw.len() {
                    continue;
                }
                let dx = j as f64 * step;
                let lift = (r * r - dx * dx).max(0.0).sqrt() - r;
                best = best.max(raw[k as usize] + lift);
            }
            best
        };
        let cams: Vec<f64> = (-half..n + half).map(cam).collect();
        let mut height_m = Vec::with_capacity(n as usize);
        let mut slope = Vec::with_capacity(n as usize);
        for i in 0..n as usize {
            let (a, b) = (cams[i], cams[i + 2 * half as usize]);
            height_m.push(0.5 * (a + b));
            slope.push((b - a) / (2 * half) as f64 / step);
        }
        Envelope { from_m, step_m: step, height_m, slope }
    }

    /// Effective height and slope at `x_m` (linear between samples, held past the ends).
    pub fn at(&self, x_m: f64) -> (f64, f64) {
        let u = ((x_m - self.from_m) / self.step_m).max(0.0);
        let i = (u.floor() as usize).min(self.height_m.len() - 1);
        let j = (i + 1).min(self.height_m.len() - 1);
        let f = (u - i as f64).clamp(0.0, 1.0);
        (self.height_m[i] + (self.height_m[j] - self.height_m[i]) * f, self.slope[i] + (self.slope[j] - self.slope[i]) * f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtb_29_stiffness_at_25_psi_is_within_the_carcass_range() {
        // Arrange
        let t = Tyre::mtb_29(2.5, 25.0, true, false);

        // Act
        let k = t.stiffness_n_m();

        // Assert: Cossalter's 100-200 kN/m, and more pressure is stiffer
        assert!((100e3..200e3).contains(&k), "{k}");
        assert!(Tyre::mtb_29(2.5, 30.0, true, false).stiffness_n_m() > k);
    }

    #[test]
    fn force_an_insert_stiffens_before_the_rim_and_holds_it_off() {
        // Arrange
        let bare = Tyre::mtb_29(2.5, 25.0, true, false);
        let with = Tyre::mtb_29(2.5, 25.0, true, true);
        let d = 0.045;

        // Act
        let (fb, fw) = (bare.force(d), with.force(d));

        // Assert
        assert!(fw.on_insert && !fb.on_insert);
        assert!(fw.force_n > fb.force_n * 1.2, "{} {}", fw.force_n, fb.force_n);
        assert!(!fb.rim_strike && bare.force(bare.rim_strike_m() + 0.001).rim_strike);
        // the load that strikes the bare rim leaves the insert's rim clear
        let strike = bare.force(bare.rim_strike_m() + 0.001).force_n;
        assert!(with.deflection_under(strike) < with.rim_strike_m());
    }

    #[test]
    fn deflection_under_inverts_force() {
        // Arrange
        let t = Tyre::mtb_29(2.4, 22.0, false, true);

        // Act
        let d = t.deflection_under(600.0);

        // Assert
        assert!((t.force(d).force_n - 600.0).abs() < 1e-3);
        assert_eq!(t.force(-0.01).force_n, 0.0);
    }

    #[test]
    fn envelope_a_flat_ground_stays_flat_and_level() {
        // Arrange
        let t = Tyre::mtb_29(2.5, 25.0, true, false);

        // Act
        let e = Envelope::new(&t, &|_| 0.1, 0.0, 1.0, 0.004);

        // Assert
        let (h, s) = e.at(0.5);
        assert!((h - 0.1).abs() < 1e-12 && s.abs() < 1e-12);
    }

    #[test]
    fn envelope_a_square_step_is_ramped_over_the_basic_curve_length_and_keeps_its_height() {
        // Arrange: a 50 mm kerb at 1 m
        let t = Tyre::mtb_29(2.5, 25.0, true, false);
        let step = |x: f64| if x >= 1.0 { 0.05 } else { 0.0 };
        let lb = (t.radius_m.powi(2) - (t.radius_m - 0.05).powi(2)).sqrt();

        // Act
        let e = Envelope::new(&t, &step, 0.0, 2.0, 0.004);

        // Assert: it starts rising about lb (plus half the follower) ahead, is level after, never overshoots
        let ls = 0.8 * t.contact_length_m(0.004);
        assert!(e.at(1.0 - lb - ls / 2.0 - 0.01).0.abs() < 1e-9);
        assert!(e.at(1.0 - lb / 2.0).0 > 0.0 && e.at(1.0 - lb / 2.0).0 < 0.05);
        assert!((e.at(1.0 + ls).0 - 0.05).abs() < 1e-9);
        assert!(e.height_m.iter().all(|h| *h <= 0.05 + 1e-12));
        assert!(e.at(1.0 - lb / 2.0).1 > 0.0);
    }

    #[test]
    fn envelope_a_narrow_slot_is_bridged() {
        // Arrange: a 30 mm wide, 40 mm deep slot
        let t = Tyre::mtb_29(2.5, 25.0, true, false);
        let slot = |x: f64| if (1.0..1.03).contains(&x) { -0.04 } else { 0.0 };

        // Act
        let e = Envelope::new(&t, &slot, 0.0, 2.0, 0.004);

        // Assert: the tyre dips far less than the slot
        let dip = e.height_m.iter().cloned().fold(0.0, f64::min);
        assert!(dip > -0.005, "{dip}");
    }
}
