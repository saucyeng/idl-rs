//! The rear end riding over terrain (whole-bike sim spec §9, stride 3): the
//! rear wheel's travel comes out of the springs, the damper and the tyre
//! instead of being prescribed. A two-mass model in the vertical: the
//! rear's share of the sprung mass on the shock (through the leverage), the
//! unsprung mass on the tyre ([`super::tyre`]: air, casing, insert, rim
//! strike), the tyre on the ground it envelops. The frame stays level
//! (no pitch) and the front is not modelled yet.
//!
//! The result is an [`Excitation`], so the chain's dynamics
//! ([`super::dynamics`]) run on it unchanged: suspension drives the chain.
//! The chain's force back on the suspension (anti-squat while pedalling,
//! chain tension) is not yet fed back.

use super::bike::SimBike;
use super::excite::{Excitation, G};
use super::suspension::{leverage_ratio, shock_stroke, AirSpring, N_DYNAMIC};
use super::tyre::{Envelope, Tyre};
use super::{SimError, SimErrorKind};

/// A damper at the shock shaft: linear up to a knee, then a shallower line
/// (the high-speed circuit), in compression and in rebound. N·s/m at the shaft.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Damper {
    /// Low-speed compression, N·s/m.
    pub lsc_ns_m: f64,
    /// High-speed compression past the knee, N·s/m.
    pub hsc_ns_m: f64,
    /// Low-speed rebound, N·s/m.
    pub lsr_ns_m: f64,
    /// High-speed rebound past the knee, N·s/m.
    pub hsr_ns_m: f64,
    /// Shaft speed where high speed starts, m/s.
    pub knee_mps: f64,
}

impl Default for Damper {
    /// A mid-range air shock's damping at the shaft, middle clicks: an
    /// estimate until the shock is dyno'd or identified from a ride.
    fn default() -> Self {
        Damper { lsc_ns_m: 4000.0, hsc_ns_m: 1200.0, lsr_ns_m: 9000.0, hsr_ns_m: 3500.0, knee_mps: 0.15 }
    }
}

impl Damper {
    /// The damping force at the shaft, N, resisting shaft speed `v_mps`
    /// (positive compressing).
    pub fn force_n(&self, v_mps: f64) -> f64 {
        self.law().force_n(v_mps, 0.0)
    }

    /// The shared force law ([`super::laws::DamperLaw`]) with these numbers.
    pub fn law(&self) -> super::laws::DamperLaw<f64> {
        super::laws::DamperLaw { lsc_ns_m: self.lsc_ns_m, hsc_ns_m: self.hsc_ns_m, lsr_ns_m: self.lsr_ns_m, hsr_ns_m: self.hsr_ns_m, knee_mps: self.knee_mps }
    }
}

/// The rear end, ready to ride.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RearEnd {
    /// The rear wheel's share of the sprung mass, kg.
    pub sprung_kg: f64,
    /// Rear unsprung mass (wheel, part of the swingarm), kg.
    pub unsprung_kg: f64,
    /// The shock's air spring.
    pub spring: AirSpring,
    /// Its gauge pressure, Pa.
    pub gauge_pa: f64,
    /// The shock's damper.
    pub damper: Damper,
    /// The rear tyre.
    pub tyre: Tyre,
}

/// One bump or drop on the trail.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Feature {
    /// Where it starts, metres along the trail.
    pub at_m: f64,
    /// Its length along the trail, metres.
    pub length_m: f64,
    /// Its height, metres: positive a bump (half-cosine up and down),
    /// negative a step down (a half-cosine ramp to the lower level).
    pub height_m: f64,
}

/// The trail's surface: flat with [`Feature`]s on it.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Terrain {
    /// The features, in any order.
    pub features: Vec<Feature>,
}

impl Terrain {
    /// Ground height at `x_m` along the trail, metres.
    pub fn height(&self, x_m: f64) -> f64 {
        let pi = std::f64::consts::PI;
        self.features
            .iter()
            .map(|f| {
                let u = (x_m - f.at_m) / f.length_m.max(1e-6);
                if f.height_m >= 0.0 {
                    if (0.0..1.0).contains(&u) { f.height_m * 0.5 * (1.0 - (2.0 * pi * u).cos()) } else { 0.0 }
                } else if u <= 0.0 {
                    0.0
                } else if u >= 1.0 {
                    f.height_m
                } else {
                    f.height_m * 0.5 * (1.0 - (pi * u).cos())
                }
            })
            .sum()
    }

    /// `count` bumps `height_m` high and `length_m` long, `spacing_m` apart, the first at `lead_m`.
    pub fn bumps(count: u32, height_m: f64, length_m: f64, spacing_m: f64, lead_m: f64) -> Terrain {
        Terrain { features: (0..count).map(|i| Feature { at_m: lead_m + i as f64 * spacing_m, length_m, height_m }).collect() }
    }
}

/// What a ride produced beside its excitation.
#[derive(Debug, Clone, PartialEq)]
pub struct RideOut {
    /// The travel and chassis motion, for the chain's dynamics.
    pub excitation: Excitation,
    /// Tyre force on the ground at each sample, N (0 in the air).
    pub tyre_n: Vec<f64>,
    /// Shock force at the shaft at each sample, N.
    pub shock_n: Vec<f64>,
    /// Ground height under the tyre at each sample, m.
    pub ground_m: Vec<f64>,
    /// Tyre deflection at each sample, m (0 in the air).
    pub tyre_deflection_m: Vec<f64>,
    /// Times the rim pinched the tyre on the ground.
    pub rim_strikes: u32,
    /// Static sag the ride starts from, m of wheel travel.
    pub sag_m: f64,
}

/// The suspension's force at the wheel, N, at travel `s` and travel speed
/// `v` (positive compressing): spring and damper through the leverage, with
/// stiff stops at top-out and bottom-out. The air compresses adiabatically
/// about the shock's stroke at sag, `x_sag`.
fn wheel_force(bike: &SimBike, rear: &RearEnd, x_sag: f64, s: f64, v: f64) -> f64 {
    let max = bike.travel_max_m;
    let sc = s.clamp(0.0, max);
    let lr = leverage_ratio(bike, sc).unwrap_or(3.0).max(0.5);
    let x = shock_stroke(bike, sc).unwrap_or(sc / lr);
    let spring = rear.spring.force_about_n(rear.gauge_pa, x_sag, x, N_DYNAMIC);
    let damper = rear.damper.force_n(v / lr);
    let stop = if s < 0.0 { 2.0e6 * s } else if s > max { 2.0e6 * (s - max) } else { 0.0 };
    (spring + damper) / lr + stop
}

/// The rider pedalling through a ride: crank torque with two power strokes
/// a revolution, the drive force it puts on the ground, and the load it
/// moves onto the rear wheel (drive force × centre of mass height ÷
/// wheelbase). The linkage carries the anti-squat share of that load
/// straight to the wheel; the rest compresses the suspension (Cossalter
/// §6.1.3: 100 % neither squats nor extends).
#[derive(Debug, Clone, PartialEq)]
pub struct Pedalling {
    /// Mean crank torque, N·m.
    pub torque_nm: f64,
    /// Cadence, rpm.
    pub cadence_rpm: f64,
    /// Chainring and cog teeth.
    pub chainring_teeth: u32,
    /// See `chainring_teeth`.
    pub cog_teeth: u32,
    /// Rear wheel's rolling radius, m.
    pub wheel_radius_m: f64,
    /// Centre of mass height, m.
    pub cog_height_m: f64,
    /// Wheelbase, m.
    pub wheelbase_m: f64,
    /// Anti-squat through travel in the gear: (travel m, %), increasing travel.
    pub anti_squat: Vec<(f64, f64)>,
}

impl Pedalling {
    /// Crank torque at `t_s`, N·m: `torque_nm` (1 − cos 2θ), zero at the dead centres.
    pub fn torque_at(&self, t_s: f64) -> f64 {
        let w = self.cadence_rpm * std::f64::consts::TAU / 60.0;
        self.torque_nm * (1.0 - (2.0 * w * t_s).cos())
    }

    /// The drive force on the ground at `t_s`, N.
    pub fn drive_n(&self, t_s: f64) -> f64 {
        self.torque_at(t_s) * self.cog_teeth as f64 / self.chainring_teeth.max(1) as f64 / self.wheel_radius_m.max(0.1)
    }

    /// Anti-squat at `travel_m`, % (linear between samples, held at the ends; 100 without a curve).
    pub fn anti_squat_at(&self, travel_m: f64) -> f64 {
        let a = &self.anti_squat;
        match a.iter().position(|p| p.0 >= travel_m) {
            None => a.last().map_or(100.0, |p| p.1),
            Some(0) => a[0].1,
            Some(i) => {
                let (p, q) = (a[i - 1], a[i]);
                p.1 + (q.1 - p.1) * (travel_m - p.0) / (q.0 - p.0).max(1e-12)
            }
        }
    }

    /// The pedalling's squat force on the suspension at `t_s` and `travel_m`, N
    /// (positive compressing), and the part the linkage takes straight to the wheel.
    pub fn squat_n(&self, t_s: f64, travel_m: f64) -> (f64, f64) {
        let transfer = self.drive_n(t_s) * self.cog_height_m / self.wheelbase_m.max(0.5);
        let as_share = self.anti_squat_at(travel_m) / 100.0;
        (transfer * (1.0 - as_share), transfer * as_share)
    }
}

/// Rides the rear end over `terrain` at `speed_mps` for `duration_s`,
/// sampling the result at `rate_hz`, from rest at sag on the flat, coasting.
/// Errors: `Excitation` for a non-positive speed, duration or rate, or a
/// rear end that can't hold its load (it sits on the stop).
pub fn ride(bike: &SimBike, rear: &RearEnd, terrain: &Terrain, speed_mps: f64, duration_s: f64, rate_hz: f64) -> Result<RideOut, SimError> {
    ride_pedalling(bike, rear, terrain, speed_mps, duration_s, rate_hz, None)
}

/// [`ride`], pedalling as `pedal` says.
pub fn ride_pedalling(bike: &SimBike, rear: &RearEnd, terrain: &Terrain, speed_mps: f64, duration_s: f64, rate_hz: f64, pedal: Option<&Pedalling>) -> Result<RideOut, SimError> {
    if !(speed_mps > 0.0 && duration_s > 0.0 && rate_hz > 0.0) {
        return Err(SimError::new(SimErrorKind::Excitation, "a ride needs a positive speed, duration and sample rate"));
    }
    let (ms, mu) = (rear.sprung_kg, rear.unsprung_kg.max(0.5));
    // static sag: the wheel force holding the sprung weight (slow spring)
    let hold = |s: f64| {
        let lr = leverage_ratio(bike, s).unwrap_or(3.0).max(0.5);
        rear.spring.force_n(rear.gauge_pa, shock_stroke(bike, s).unwrap_or(s / lr), 1.0) / lr - ms * G
    };
    let (mut a, mut b) = (0.0, bike.travel_max_m);
    if hold(b) < 0.0 {
        return Err(SimError::new(SimErrorKind::Excitation, "the shock can't hold the rear's weight: it sits bottomed out (more pressure)"));
    }
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if hold(m) < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    let sag = 0.5 * (a + b);
    let x_sag = shock_stroke(bike, sag).unwrap_or(0.0);
    // axle height above the flat (the tyre's centre over its undeformed radius) and the frame's datum
    let squash0 = rear.tyre.deflection_under((ms + mu) * G);
    // the ground as the tyre feels it (enveloping), a metre either side of the ride
    let env = Envelope::new(&rear.tyre, &|x| terrain.height(x), -1.0, speed_mps * duration_s + 1.0, squash0);
    let mut yu = -squash0;
    let mut vu = 0.0;
    let mut ys = yu - sag;
    let mut vs = 0.0;

    let dt = 2.0e-5;
    let steps_per_sample = ((1.0 / rate_hz) / dt).round().max(1.0) as usize;
    let dt = 1.0 / rate_hz / steps_per_sample as f64;
    let n = (duration_s * rate_hz).ceil() as usize + 1;
    let mut out = RideOut {
        excitation: Excitation { t_s: Vec::with_capacity(n), travel_m: Vec::with_capacity(n), fx: vec![0.0; n], fy: Vec::with_capacity(n), omega: vec![0.0; n], speed_mps: vec![speed_mps; n] },
        tyre_n: Vec::with_capacity(n),
        shock_n: Vec::with_capacity(n),
        ground_m: Vec::with_capacity(n),
        tyre_deflection_m: Vec::with_capacity(n),
        rim_strikes: 0,
        sag_m: sag,
    };
    let mut striking = false;
    let forces = |t: f64, yu: f64, vu: f64, ys: f64, vs: f64| {
        let x = speed_mps * t;
        let (g, slope) = env.at(x);
        let g_rate = slope * speed_mps;
        let pen = g - yu;
        let tyre = if pen > 0.0 { (rear.tyre.force(pen).force_n + rear.tyre.damping_ns_m * (g_rate - vu)).max(0.0) } else { 0.0 };
        let s = yu - ys;
        let fw = wheel_force(bike, rear, x_sag, s, vu - vs);
        let (squat, direct) = pedal.map_or((0.0, 0.0), |p| p.squat_n(t, s.clamp(0.0, bike.travel_max_m)));
        // the squat share pushes the frame down on the suspension, the rest goes down through the linkage
        // and the wheel carries the whole transfer: the suspension's share through the spring, the rest directly
        (tyre, fw - squat, g, direct + squat)
    };
    let mut t = 0.0;
    for k in 0..n {
        let (tyre, fw, g, _) = forces(t, yu, vu, ys, vs);
        let acc_s = fw / ms - G;
        out.excitation.t_s.push(k as f64 / rate_hz);
        out.excitation.travel_m.push((yu - ys).clamp(0.0, bike.travel_max_m));
        out.excitation.fy.push(acc_s + G);
        out.tyre_n.push(tyre);
        let lr = leverage_ratio(bike, (yu - ys).clamp(0.0, bike.travel_max_m)).unwrap_or(3.0);
        let squat = pedal.map_or(0.0, |p| p.squat_n(t, (yu - ys).clamp(0.0, bike.travel_max_m)).0);
        out.shock_n.push((fw + squat) * lr);
        out.ground_m.push(g);
        let pen = (g - yu).max(0.0);
        out.tyre_deflection_m.push(pen);
        let strike = rear.tyre.force(pen).rim_strike;
        if strike && !striking {
            out.rim_strikes += 1;
        }
        striking = strike;
        if k + 1 == n {
            break;
        }
        for _ in 0..steps_per_sample {
            let (tyre, fw, _, direct) = forces(t, yu, vu, ys, vs);
            // semi-implicit Euler: velocities from the forces, then positions from the new velocities
            vs += dt * (fw / ms - G);
            vu += dt * ((tyre - fw - direct) / mu - G);
            ys += dt * vs;
            yu += dt * vu;
            t += dt;
        }
        if !(ys.is_finite() && yu.is_finite()) {
            return Err(SimError::new(SimErrorKind::Excitation, "the ride diverged"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::bike::tests::{gt_force_drivetrain, gt_force_linkage};
    use crate::sim::suspension::PSI_PA;

    fn rear(bike: &SimBike) -> RearEnd {
        let st = shock_stroke(bike, bike.travel_max_m).unwrap();
        RearEnd {
            sprung_kg: 55.0,
            unsprung_kg: 4.5,
            spring: AirSpring { area_m2: 16e-4, positive_m3: 2.6 * 16e-4 * st, negative_m3: 0.6 * 16e-4 * st, stroke_m: st },
            gauge_pa: 220.0 * PSI_PA,
            damper: Damper::default(),
            tyre: Tyre::mtb_29(2.5, 25.0, true, false),
        }
    }

    #[test]
    fn ride_on_the_flat_stays_at_sag() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let r = rear(&bike);

        // Act
        let out = ride(&bike, &r, &Terrain::default(), 5.0, 0.5, 500.0).unwrap();

        // Assert
        let drift = out.excitation.travel_m.iter().map(|s| (s - out.sag_m).abs()).fold(0.0, f64::max);
        assert!(drift < 1e-3, "{drift}");
        assert!(out.sag_m > 0.01 && out.sag_m < 0.08, "{}", out.sag_m);
    }

    #[test]
    fn ride_a_bump_compresses_then_settles_back_to_sag() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let r = rear(&bike);
        let terrain = Terrain::bumps(1, 0.08, 0.5, 1.0, 1.0);

        // Act
        let out = ride(&bike, &r, &terrain, 5.0, 1.5, 1000.0).unwrap();

        // Assert
        let peak = out.excitation.travel_m.iter().cloned().fold(0.0, f64::max);
        assert!(peak > out.sag_m + 0.02, "peak {peak} sag {}", out.sag_m);
        let last = *out.excitation.travel_m.last().unwrap();
        assert!((last - out.sag_m).abs() < 0.01, "{last}");
    }

    #[test]
    fn damper_rebound_resists_harder_than_compression_at_low_speed() {
        // Arrange
        let d = Damper::default();

        // Act / Assert
        assert!(d.force_n(-0.1).abs() > d.force_n(0.1));
        assert!(d.force_n(0.5) - d.force_n(0.4) < d.force_n(0.1) - d.force_n(0.0) + 1e-9);
    }

    #[test]
    fn terrain_bump_rises_and_falls_a_drop_stays_down() {
        // Arrange
        let t = Terrain { features: vec![Feature { at_m: 1.0, length_m: 0.4, height_m: 0.1 }, Feature { at_m: 3.0, length_m: 0.2, height_m: -0.3 }] };

        // Act / Assert
        assert!((t.height(1.2) - 0.1).abs() < 1e-12);
        assert_eq!(t.height(2.0), 0.0);
        assert!((t.height(4.0) + 0.3).abs() < 1e-12);
    }

    #[test]
    fn ride_pedalling_with_no_anti_squat_sits_deeper_than_with_full() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let r = rear(&bike);
        let pedal = |pct: f64| Pedalling { torque_nm: 60.0, cadence_rpm: 80.0, chainring_teeth: 30, cog_teeth: 28, wheel_radius_m: 0.37, cog_height_m: 1.1, wheelbase_m: 1.28, anti_squat: vec![(0.0, pct), (0.16, pct)] };
        let mean = |o: &RideOut| o.excitation.travel_m.iter().skip(500).sum::<f64>() / (o.excitation.travel_m.len() - 500) as f64;

        // Act
        let none = ride_pedalling(&bike, &r, &Terrain::default(), 5.0, 2.0, 500.0, Some(&pedal(0.0))).unwrap();
        let full = ride_pedalling(&bike, &r, &Terrain::default(), 5.0, 2.0, 500.0, Some(&pedal(100.0))).unwrap();

        // Assert: 100 % holds the sag; none squats under the transfer
        assert!((mean(&full) - full.sag_m).abs() < 1e-3, "{} {}", mean(&full), full.sag_m);
        assert!(mean(&none) > none.sag_m + 2e-3, "{} {}", mean(&none), none.sag_m);
    }

    #[test]
    fn pedalling_torque_is_zero_at_dead_centre_and_averages_the_mean() {
        // Arrange
        let p = Pedalling { torque_nm: 50.0, cadence_rpm: 60.0, chainring_teeth: 30, cog_teeth: 30, wheel_radius_m: 0.37, cog_height_m: 1.1, wheelbase_m: 1.28, anti_squat: Vec::new() };

        // Act
        let n = 1000;
        let avg = (0..n).map(|k| p.torque_at(k as f64 / n as f64)).sum::<f64>() / n as f64;

        // Assert
        assert!(p.torque_at(0.0).abs() < 1e-12);
        assert!((avg - 50.0).abs() < 1e-6, "{avg}");
        assert_eq!(p.anti_squat_at(0.05), 100.0);
    }

    #[test]
    fn ride_a_sharp_bump_with_an_insert_strikes_the_rim_less_than_without() {
        // Arrange: a sharp 80 mm bump 100 mm long at 8 m/s, low pressure
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let bare = RearEnd { tyre: Tyre::mtb_29(2.5, 18.0, false, false), ..rear(&bike) };
        let with = RearEnd { tyre: Tyre::mtb_29(2.5, 18.0, false, true), ..rear(&bike) };
        let kerb = Terrain { features: vec![Feature { at_m: 2.0, length_m: 0.1, height_m: 0.08 }] };

        // Act
        let a = ride(&bike, &bare, &kerb, 8.0, 0.6, 2000.0).unwrap();
        let b = ride(&bike, &with, &kerb, 8.0, 0.6, 2000.0).unwrap();

        // Assert: the insert only adds force, so the tyre squashes no further and strikes no more
        let peak = |o: &RideOut| o.tyre_deflection_m.iter().cloned().fold(0.0, f64::max);
        assert!(peak(&b) <= peak(&a) + 1e-9, "{} {}", peak(&a), peak(&b));
        assert!(b.rim_strikes <= a.rim_strikes);
    }

    #[test]
    fn ride_tyre_deflection_on_the_flat_is_the_static_squash() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();
        let r = rear(&bike);

        // Act
        let o = ride(&bike, &r, &Terrain::default(), 5.0, 0.5, 500.0).unwrap();

        // Assert
        let want = r.tyre.deflection_under((r.sprung_kg + r.unsprung_kg) * G);
        assert!((o.tyre_deflection_m[100] - want).abs() < 2e-4, "{} {want}", o.tyre_deflection_m[100]);
        assert_eq!(o.rim_strikes, 0);
    }
}
