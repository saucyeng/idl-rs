//! The synthetic loop and the body's motion along it (contract C1 §9.3).
//!
//! A planar ellipse in a local east/north metre frame, traversed
//! anticlockwise, at the speed a fixed lateral-acceleration limit allows.
//! Arc length and elapsed time are tabulated once over the loop parameter and
//! inverted by interpolation; everything else is evaluated analytically at the
//! recovered parameter, so a sample's position is never an interpolation of
//! two table rows.
//!
//! Pure and libm-free: every transcendental call goes through
//! [`super::dtrig`] (C1 §9.7 rule 1).

use super::dtrig::{atan2, sin_cos, TAU};

/// Steps in the arc-length / time tables. Part of the output format: change
/// it and every generated file's bytes change (C1 §9.3).
pub const ARC_TABLE_STEPS: usize = 4096;

/// Semi-minor over semi-major. Fixed, not configurable: the ratio is what
/// gives the loop its curvature variation, and a caller that could set it to
/// 1.0 would silently produce the degenerate constant-yaw-rate case C1 §9.3
/// exists to avoid.
pub const AXIS_RATIO: f64 = 0.5;

/// Speed cap on the low-curvature flanks, m/s (≈ 43 km/h).
pub const V_MAX_M_S: f64 = 12.0;

/// Lateral-acceleration limit that sets the speed in the tight ends, m/s².
pub const A_LAT_MAX_M_S2: f64 = 6.0;

/// The body's state at one instant within a lap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    /// Loop parameter, radians in `[0, 2π)`.
    pub u: f64,
    /// Position east of the loop centre, metres.
    pub east_m: f64,
    /// Position north of the loop centre, metres.
    pub north_m: f64,
    /// Path tangent heading, radians anticlockwise from east (this is the
    /// body's yaw ψ — the loop is planar and the body does not slip).
    pub heading_rad: f64,
    /// Ground speed, m/s.
    pub speed_m_s: f64,
    /// Yaw rate, rad/s. Positive (turning left) everywhere: the loop is
    /// traversed anticlockwise and `dψ/ds = +κ`.
    pub yaw_rate_rad_s: f64,
    /// Distance travelled from the lap's start, metres.
    pub arc_m: f64,
}

/// The tabulated loop.
#[derive(Debug, Clone)]
pub struct Loop {
    /// Semi-major axis (east), metres.
    pub semi_major_m: f64,
    /// Semi-minor axis (north), metres.
    pub semi_minor_m: f64,
    /// Total arc length of one circuit, metres. Equals the requested lap
    /// length to within the table's own quadrature error.
    pub perimeter_m: f64,
    /// Time to complete one circuit, seconds.
    pub lap_time_s: f64,
    /// Cumulative arc length at each table node, metres.
    arc: Vec<f64>,
    /// Cumulative elapsed time at each table node, seconds.
    time: Vec<f64>,
}

/// `|d(east, north)/du|` at `u` for semi-axes `a`, `b`.
fn speed_of_parameter(a: f64, b: f64, u: f64) -> f64 {
    let (s, c) = sin_cos(u);
    let de = -a * s;
    let dn = b * c;
    (de * de + dn * dn).sqrt()
}

/// Signed curvature of the ellipse at `u`, 1/m. Always positive for an
/// anticlockwise traversal.
fn curvature(a: f64, b: f64, u: f64) -> f64 {
    let sp = speed_of_parameter(a, b, u);
    a * b / (sp * sp * sp)
}

/// Ground speed at `u`, m/s: the lateral-acceleration limit, capped.
fn speed_at(a: f64, b: f64, u: f64) -> f64 {
    let k = curvature(a, b, u);
    let limited = (A_LAT_MAX_M_S2 / k).sqrt();
    if limited < V_MAX_M_S {
        limited
    } else {
        V_MAX_M_S
    }
}

/// Cumulative trapezoid integral of `f` over the table grid, node by node.
/// Returns `ARC_TABLE_STEPS + 1` values starting at zero.
fn cumulative_trapezoid(step: f64, values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0.0;
    out.push(0.0);
    for i in 1..values.len() {
        acc += 0.5 * (values[i - 1] + values[i]) * step;
        out.push(acc);
    }
    out
}

impl Loop {
    /// Builds the loop whose perimeter is `lap_length_m`.
    ///
    /// The unit ellipse (`a = 1`) is integrated first to get its perimeter,
    /// then both axes are scaled by the ratio needed — the arc length of an
    /// ellipse is linear in its scale, so one integration suffices and the
    /// requested length is hit exactly (to quadrature error, which at 4096
    /// steps is ~1e-11 relative).
    pub fn build(lap_length_m: f64) -> Loop {
        let n = ARC_TABLE_STEPS;
        let du = TAU / n as f64;

        let unit_speeds: Vec<f64> =
            (0..=n).map(|i| speed_of_parameter(1.0, AXIS_RATIO, i as f64 * du)).collect();
        // `last().unwrap()` here and below: `cumulative_trapezoid` always
        // pushes at least the leading zero, so its result is never empty.
        let unit_perimeter = *cumulative_trapezoid(du, &unit_speeds).last().unwrap();

        let semi_major_m = lap_length_m / unit_perimeter;
        let semi_minor_m = semi_major_m * AXIS_RATIO;

        let du_speeds: Vec<f64> =
            (0..=n).map(|i| speed_of_parameter(semi_major_m, semi_minor_m, i as f64 * du)).collect();
        let arc = cumulative_trapezoid(du, &du_speeds);

        let dt_du: Vec<f64> = (0..=n)
            .map(|i| {
                let u = i as f64 * du;
                du_speeds[i] / speed_at(semi_major_m, semi_minor_m, u)
            })
            .collect();
        let time = cumulative_trapezoid(du, &dt_du);

        let perimeter_m = *arc.last().unwrap();
        let lap_time_s = *time.last().unwrap();

        Loop { semi_major_m, semi_minor_m, perimeter_m, lap_time_s, arc, time }
    }

    /// The loop parameter at elapsed time `tau_s` within one lap, by binary
    /// search and linear interpolation over the time table. `tau_s` outside
    /// `[0, lap_time_s]` is clamped.
    fn parameter_at(&self, tau_s: f64) -> (f64, f64) {
        let n = ARC_TABLE_STEPS;
        let du = TAU / n as f64;

        if tau_s <= 0.0 {
            return (0.0, 0.0);
        }
        if tau_s >= self.lap_time_s {
            return (TAU, self.perimeter_m);
        }

        // First node whose cumulative time exceeds `tau_s`; the bracket is
        // [hi - 1, hi], and `hi` is in 1..=n because of the guards above.
        let hi = self.time.partition_point(|t| *t <= tau_s);
        let lo = hi - 1;
        let span = self.time[hi] - self.time[lo];
        let frac = if span > 0.0 { (tau_s - self.time[lo]) / span } else { 0.0 };

        let u = (lo as f64 + frac) * du;
        let arc = self.arc[lo] + frac * (self.arc[hi] - self.arc[lo]);
        (u, arc)
    }

    /// The body's state at elapsed time `tau_s` within one lap.
    pub fn state_at(&self, tau_s: f64) -> BodyState {
        let (u, arc_m) = self.parameter_at(tau_s);
        let (a, b) = (self.semi_major_m, self.semi_minor_m);
        let (su, cu) = sin_cos(u);

        let east_m = a * cu;
        let north_m = b * su;
        // Tangent d(east, north)/du = (−a sin u, b cos u).
        let heading_rad = atan2(b * cu, -a * su);
        let speed_m_s = speed_at(a, b, u);
        let yaw_rate_rad_s = curvature(a, b, u) * speed_m_s;

        BodyState { u, east_m, north_m, heading_rad, speed_m_s, yaw_rate_rad_s, arc_m }
    }

    /// The start/finish gate: the loop point at `u = 0` and the inward unit
    /// normal there (C1 §9.6). A lap boundary is a crossing of the line
    /// through that point perpendicular to the tangent.
    pub fn gate(&self) -> (f64, f64, f64, f64) {
        let state = self.state_at(0.0);
        // Tangent at u = 0 is (0, +b) — due north — so the gate normal is
        // the heading direction itself, taken from the state rather than
        // re-derived so the two can never disagree.
        let (s, c) = sin_cos(state.heading_rad);
        (state.east_m, state.north_m, c, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_loop_has_the_requested_perimeter() {
        // Arrange
        let target = 400.0;

        // Act
        let lp = Loop::build(target);

        // Assert
        assert!((lp.perimeter_m - target).abs() < 1e-6, "{}", lp.perimeter_m);
    }

    #[test]
    fn semi_axes_hold_the_fixed_ratio() {
        // Arrange / Act
        let lp = Loop::build(400.0);

        // Assert
        assert!((lp.semi_minor_m / lp.semi_major_m - AXIS_RATIO).abs() < 1e-12);
    }

    #[test]
    fn state_at_zero_and_at_one_lap_are_the_same_point() {
        // Arrange
        let lp = Loop::build(400.0);

        // Act
        let start = lp.state_at(0.0);
        let end = lp.state_at(lp.lap_time_s);

        // Assert
        assert!((start.east_m - end.east_m).abs() < 1e-9);
        assert!((start.north_m - end.north_m).abs() < 1e-9);
    }

    #[test]
    fn arc_length_advances_monotonically_over_a_lap() {
        // Arrange
        let lp = Loop::build(400.0);
        let steps = 500;

        // Act
        let arcs: Vec<f64> =
            (0..=steps).map(|i| lp.state_at(lp.lap_time_s * i as f64 / steps as f64).arc_m).collect();

        // Assert
        for pair in arcs.windows(2) {
            assert!(pair[1] >= pair[0], "{pair:?}");
        }
        assert!((arcs[steps] - lp.perimeter_m).abs() < 1e-6);
    }

    #[test]
    fn speed_is_capped_on_the_flanks_and_lower_in_the_ends() {
        // Arrange — u = 0 is the end of the major axis (tightest corner),
        // u = π/2 is the flank (loosest).
        let lp = Loop::build(400.0);

        // Act
        let corner = speed_at(lp.semi_major_m, lp.semi_minor_m, 0.0);
        let flank = speed_at(lp.semi_major_m, lp.semi_minor_m, super::super::dtrig::HALF_PI);

        // Assert
        assert!(corner < flank, "corner {corner} flank {flank}");
        assert!(flank <= V_MAX_M_S);
    }

    #[test]
    fn yaw_rate_is_positive_everywhere_on_an_anticlockwise_loop() {
        // Arrange
        let lp = Loop::build(400.0);

        // Act / Assert
        for i in 0..=400 {
            let st = lp.state_at(lp.lap_time_s * i as f64 / 400.0);
            assert!(st.yaw_rate_rad_s > 0.0, "u {} rate {}", st.u, st.yaw_rate_rad_s);
        }
    }

    #[test]
    fn integrating_speed_over_a_lap_recovers_the_perimeter() {
        // Arrange
        let lp = Loop::build(400.0);
        let steps = 20_000;
        let dt = lp.lap_time_s / steps as f64;

        // Act — midpoint rule over the sampled speed.
        let distance: f64 =
            (0..steps).map(|i| lp.state_at((i as f64 + 0.5) * dt).speed_m_s * dt).sum();

        // Assert — 0.1 % of the loop, which is the midpoint rule's own error
        // at this step count, not a modelling tolerance.
        assert!((distance - lp.perimeter_m).abs() < lp.perimeter_m * 1e-3, "{distance}");
    }

    #[test]
    fn a_shorter_lap_is_faster() {
        // Arrange / Act
        let small = Loop::build(120.0);
        let big = Loop::build(400.0);

        // Assert
        assert!(small.lap_time_s < big.lap_time_s);
    }

    #[test]
    fn build_is_deterministic_across_calls() {
        // Arrange / Act
        let first = Loop::build(400.0);
        let second = Loop::build(400.0);

        // Assert
        let bits = |lp: &Loop| {
            (0..1000)
                .map(|i| lp.state_at(lp.lap_time_s * i as f64 / 1000.0).east_m.to_bits())
                .collect::<Vec<u64>>()
        };
        assert_eq!(bits(&first), bits(&second));
    }
}
