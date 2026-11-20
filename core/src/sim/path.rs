//! The chain's path round its wheels: straight spans on the common tangents
//! between consecutive wheels, wrap arcs on each wheel, the rollers the
//! chain would otherwise cross, and the cage angle that makes the path as
//! long as the chain.
//!
//! The loop, in the direction the chain moves when pedalling forward: cog
//! (CW) → top-span rollers → chainring (CW) → lower-span rollers → lower
//! jockey (CW) → upper jockey (CCW) → cog.

use std::f64::consts::TAU;

use crate::calibration::drivetrain::{Span, CHAIN_PITCH_M};
use crate::calibration::linkage::{LinkagePose, P2};

use super::bike::{SimBike, Wheel, WheelRole};
use super::{SimError, SimErrorKind};

/// The chain wrapped on one wheel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    /// The wheel.
    pub wheel: Wheel,
    /// Angle of the point where the chain arrives, rad (from the centre).
    pub arrive: f64,
    /// Angle of the point where the chain leaves, rad.
    pub depart: f64,
    /// How far round the wheel the chain wraps, rad (0..2π).
    pub sweep: f64,
    /// Chain length on the wheel, metres.
    pub len: f64,
}

/// A straight span from one wheel to the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    /// Where it leaves the wheel before it, metres.
    pub from: P2,
    /// Where it meets the wheel after it, metres.
    pub to: P2,
    /// Length, metres.
    pub len: f64,
}

/// A closed chain path: `arcs[i]`, then `lines[i]` to `arcs[i + 1]` (the
/// last line back to `arcs[0]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPath {
    /// Wraps, in the chain's direction of travel.
    pub arcs: Vec<Arc>,
    /// Spans, `lines[i]` leaving `arcs[i]`.
    pub lines: Vec<Line>,
    /// Total length along the pin line, metres.
    pub length: f64,
}

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn len(a: P2) -> f64 {
    dot(a, a).sqrt()
}
fn ang(c: P2, p: P2) -> f64 {
    (p[1] - c[1]).atan2(p[0] - c[0])
}

/// The common tangent the chain takes from wheel `a` to wheel `b`, as
/// (leaving point on `a`, meeting point on `b`), metres. None when the
/// wheels are too close for it (they overlap the wrong way).
pub fn tangent(a: &Wheel, b: &Wheel) -> Option<(P2, P2)> {
    // signed radii: +r when the centre is on the right of the travel (CW)
    let (ra, rb) = (a.wrap.sign() * a.radius, b.wrap.sign() * b.radius);
    let d = sub(b.centre, a.centre);
    let dist = len(d);
    let s = (ra - rb) / dist;
    if !s.is_finite() || s.abs() >= 1.0 {
        return None;
    }
    let gamma = d[1].atan2(d[0]) - s.asin();
    // the chain's left normal; a point on a wheel is centre + signed radius × normal
    let n = [-gamma.sin(), gamma.cos()];
    Some(([a.centre[0] + ra * n[0], a.centre[1] + ra * n[1]], [b.centre[0] + rb * n[0], b.centre[1] + rb * n[1]]))
}

/// The closed path through `wheels` in order.
pub fn path_through(wheels: &[Wheel]) -> Result<ChainPath, SimError> {
    let n = wheels.len();
    let mut lines = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (&wheels[i], &wheels[(i + 1) % n]);
        let (p, q) = tangent(a, b).ok_or_else(|| SimError::new(SimErrorKind::NoPath, format!("the chain can't run from the {:?} to the {:?}: they overlap", a.role, b.role)))?;
        lines.push(Line { from: p, to: q, len: len(sub(q, p)) });
    }
    let mut arcs = Vec::with_capacity(n);
    for i in 0..n {
        let w = wheels[i];
        let arrive = ang(w.centre, lines[(i + n - 1) % n].to);
        let depart = ang(w.centre, lines[i].from);
        let raw = match w.wrap {
            super::bike::Wrap::Cw => arrive - depart,
            super::bike::Wrap::Ccw => depart - arrive,
        };
        let sweep = raw.rem_euclid(TAU);
        arcs.push(Arc { wheel: w, arrive, depart, sweep, len: sweep * w.len_radius });
    }
    let length = lines.iter().map(|l| l.len).sum::<f64>() + arcs.iter().map(|a| a.len).sum::<f64>();
    Ok(ChainPath { arcs, lines, length })
}

impl ChainPath {
    /// The wrap on the wheel with role `role`.
    pub fn arc(&self, role: WheelRole) -> Option<(usize, &Arc)> {
        self.arcs.iter().enumerate().find(|(_, a)| a.wheel.role == role)
    }

    /// The point `u` metres along the path from where it arrives on `arcs[0]`.
    pub fn point_at(&self, u: f64) -> P2 {
        let mut u = u.rem_euclid(self.length.max(1e-12));
        for (a, l) in self.arcs.iter().zip(&self.lines) {
            if u <= a.len {
                let t = a.arrive - a.wheel.wrap.sign() * u / a.wheel.len_radius;
                return [a.wheel.centre[0] + a.wheel.radius * t.cos(), a.wheel.centre[1] + a.wheel.radius * t.sin()];
            }
            u -= a.len;
            if u <= l.len {
                let f = if l.len > 0.0 { u / l.len } else { 0.0 };
                return [l.from[0] + f * (l.to[0] - l.from[0]), l.from[1] + f * (l.to[1] - l.from[1])];
            }
            u -= l.len;
        }
        self.arcs[0].wheel.centre
    }

    /// The path as a polyline for drawing, a point at least every `step` metres, closed.
    pub fn polyline(&self, step: f64) -> Vec<P2> {
        let n = (self.length / step.max(1e-4)).ceil() as usize;
        (0..=n).map(|k| self.point_at(self.length * k as f64 / n as f64)).collect()
    }
}

/// Signed distance from `c` to the right of the line `l`, metres, and
/// whether `c` projects inside it.
fn right_of(l: &Line, c: P2) -> (f64, bool) {
    let d = sub(l.to, l.from);
    let ll = len(d).max(1e-12);
    let u = [d[0] / ll, d[1] / ll];
    let w = sub(c, l.from);
    let along = dot(w, u);
    (-cross(u, w), along > 0.0 && along < ll)
}

/// The chain's path at `pose` in gear `gear` with the cage at absolute
/// rotation `cage_abs`: round every fitted threaded idler, and round each
/// fitted guide (`enabled`) the chain would otherwise cross.
pub fn chain_path(bike: &SimBike, pose: &LinkagePose, gear: usize, cage_abs: f64, enabled: &dyn Fn(usize) -> bool) -> Result<ChainPath, SimError> {
    let [cog, ring, lower, upper] = bike.fixed_wheels(pose, gear, cage_abs)?;
    // threaded idlers are always in the path, in order along their span; guides
    // join only where the chain would otherwise cross them
    let along = |from: P2, to: P2, c: P2| (c[0] - from[0]) * (to[0] - from[0]) + (c[1] - from[1]) * (to[1] - from[1]);
    let threaded = |span: Span, from: P2, to: P2| {
        let mut v: Vec<Wheel> = bike.rollers_on(span, enabled).into_iter().filter(|&i| bike.drivetrain.rollers[i].threaded).map(|i| bike.roller(pose, i)).collect();
        v.sort_by(|a, b| along(from, to, a.centre).partial_cmp(&along(from, to, b.centre)).unwrap_or(std::cmp::Ordering::Equal));
        v
    };
    let mut top: Vec<Wheel> = threaded(Span::Top, cog.centre, ring.centre);
    let mut low: Vec<Wheel> = threaded(Span::Lower, ring.centre, lower.centre);
    let guide = |i: &usize| !bike.drivetrain.rollers[*i].threaded;
    let mut pending: Vec<(Span, Wheel)> = bike
        .rollers_on(Span::Top, enabled)
        .into_iter()
        .filter(guide)
        .map(|i| (Span::Top, bike.roller(pose, i)))
        .chain(bike.rollers_on(Span::Lower, enabled).into_iter().filter(guide).map(|i| (Span::Lower, bike.roller(pose, i))))
        .collect();
    loop {
        let wheels: Vec<Wheel> = std::iter::once(cog).chain(top.iter().copied()).chain(std::iter::once(ring)).chain(low.iter().copied()).chain([lower, upper]).collect();
        let path = path_through(&wheels)?;
        // the first roller (in path order) the current path cuts through goes in
        let mut hit: Option<(usize, usize)> = None;
        for (k, (span, w)) in pending.iter().enumerate() {
            let (first, last) = match span {
                Span::Top => (0, top.len()),
                Span::Lower => (top.len() + 1, top.len() + 1 + low.len()),
            };
            for li in first..=last {
                let (right, inside) = right_of(&path.lines[li], w.centre);
                if inside && w.wrap.sign() * right < w.radius {
                    hit = Some((k, li - first));
                    break;
                }
            }
            if hit.is_some() {
                break;
            }
        }
        match hit {
            None => return Ok(path),
            Some((k, slot)) => {
                let (span, w) = pending.remove(k);
                match span {
                    Span::Top => top.insert(slot, w),
                    Span::Lower => low.insert(slot, w),
                }
            }
        }
    }
}

/// The cage angle (forward from drawn, relative to the seatstay, rad) that
/// makes the path as long as the chain at `pose` in gear `gear`, and that
/// path. Searches -40° to +100° and takes the root nearest the drawn angle.
pub fn close_cage(bike: &SimBike, pose: &LinkagePose, gear: usize, enabled: &dyn Fn(usize) -> bool) -> Result<(f64, ChainPath), SimError> {
    close_cage_near(bike, pose, gear, 0.0, enabled)
}

/// [`close_cage`], taking the root nearest `near_fwd` (rad): through travel,
/// the previous sample's angle, so the cage follows one branch instead of
/// jumping to another. A sign change where the path's length jumps (a guide
/// joining or leaving the chain) is not a root and is skipped.
pub fn close_cage_near(bike: &SimBike, pose: &LinkagePose, gear: usize, near_fwd: f64, enabled: &dyn Fn(usize) -> bool) -> Result<(f64, ChainPath), SimError> {
    let target = bike.drivetrain.chain_length_m();
    let f = |fwd: f64| chain_path(bike, pose, gear, bike.cage_abs(pose, fwd), enabled).map(|p| p.length - target);
    let (lo, hi, step) = (-0.7_f64, 1.75_f64, 0.01_f64);
    let mut brackets: Vec<(f64, f64, f64)> = Vec::new();
    let mut prev: Option<(f64, f64)> = None;
    let mut x = lo;
    while x <= hi {
        if let Ok(v) = f(x) {
            if let Some((px, pv)) = prev {
                if pv.signum() != v.signum() {
                    brackets.push((px, x, pv));
                }
            }
            prev = Some((x, v));
        } else {
            prev = None;
        }
        x += step;
    }
    brackets.sort_by(|p, q| (0.5 * (p.0 + p.1) - near_fwd).abs().total_cmp(&(0.5 * (q.0 + q.1) - near_fwd).abs()));
    for (mut a, mut b, fa) in brackets {
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            let fm = f(m)?;
            if fm.signum() == fa.signum() {
                a = m;
            } else {
                b = m;
            }
        }
        let fwd = 0.5 * (a + b);
        let path = chain_path(bike, pose, gear, bike.cage_abs(pose, fwd), enabled)?;
        if (path.length - target).abs() < CLOSE_TOL_M {
            return Ok((fwd, path));
        }
    }
    Err(SimError::new(SimErrorKind::NoPath, format!("no cage angle fits a {}-link chain in gear {} at this travel: the chain is too short or too long", bike.drivetrain.chain_links, gear + 1)))
}

/// How near the chain's length a closed loop's path must be, m.
const CLOSE_TOL_M: f64 = 1e-4;

/// The chain lengths (links, as a fractional range) that close the loop in
/// every gear from top-out to full travel, with the cage anywhere it can
/// reach (-40° to +100° from drawn). None when no length fits them all.
pub fn chain_fit(bike: &SimBike, enabled: &dyn Fn(usize) -> bool) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for gear in 0..bike.drivetrain.cassette_teeth.len() {
        for s in [0.0, 0.5 * bike.travel_max_m, bike.travel_max_m] {
            let pose = bike.pose(s);
            let (mut mn, mut mx) = (f64::INFINITY, 0.0f64);
            let mut k = -0.7;
            while k <= 1.75 {
                if let Ok(p) = chain_path(bike, &pose, gear, bike.cage_abs(&pose, k), enabled) {
                    mn = mn.min(p.length);
                    mx = mx.max(p.length);
                }
                k += 0.01;
            }
            lo = lo.max(mn / CHAIN_PITCH_M);
            hi = hi.min(mx / CHAIN_PITCH_M);
        }
    }
    (lo <= hi).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::super::bike::tests::{gt_force_drivetrain, gt_force_linkage};
    use super::super::bike::Wrap;
    use super::*;
    use crate::calibration::drivetrain::{Body, ChainSide, Roller};

    fn wheel(c: P2, r: f64, wrap: Wrap) -> Wheel {
        Wheel { role: WheelRole::Cog, centre: c, radius: r, len_radius: r, wrap, toothed: false }
    }

    #[test]
    fn two_equal_wheels_make_a_stadium_two_spans_and_two_half_wraps() {
        // Arrange
        let a = wheel([0.0, 0.0], 0.1, Wrap::Cw);
        let b = wheel([1.0, 0.0], 0.1, Wrap::Cw);

        // Act
        let p = path_through(&[a, b]).unwrap();

        // Assert: 2 × 1 m + 2π × 0.1
        assert!((p.length - (2.0 + TAU * 0.1)).abs() < 1e-12, "{}", p.length);
        assert!((p.arcs[0].sweep - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn crossed_wheels_take_the_internal_tangent() {
        // Arrange: a figure of eight, CW then CCW
        let a = wheel([0.0, 0.0], 0.1, Wrap::Cw);
        let b = wheel([1.0, 0.0], 0.1, Wrap::Ccw);

        // Act
        let (p, q) = tangent(&a, &b).unwrap();

        // Assert: the span crosses the centre line at its middle, at length √(1 − 0.2²)
        assert!((p[1] + q[1]).abs() < 1e-12);
        assert!((len(sub(q, p)) - (1.0f64 - 0.04).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn point_at_walks_the_whole_loop_back_to_the_start() {
        // Arrange
        let p = path_through(&[wheel([0.0, 0.0], 0.1, Wrap::Cw), wheel([1.0, 0.2], 0.05, Wrap::Cw)]).unwrap();

        // Act
        let (start, end) = (p.point_at(0.0), p.point_at(p.length - 1e-12));

        // Assert
        assert!(len(sub(start, end)) < 1e-9);
    }

    #[test]
    fn close_cage_makes_the_gt_force_path_the_chain_length_in_every_gear_through_travel() {
        // Arrange
        let bike = SimBike::new(&gt_force_linkage(), &gt_force_drivetrain()).unwrap();

        for gear in [0, 5, 11] {
            for s in [0.0, 0.08, 0.16] {
                // Act
                let (_, path) = close_cage(&bike, &bike.pose(s), gear, &|_| true).unwrap();

                // Assert
                assert!((path.length - bike.drivetrain.chain_length_m()).abs() < 1e-9, "gear {gear} at {s}: {}", path.length);
            }
        }
    }

    #[test]
    fn a_roller_clear_of_the_chain_is_not_in_the_path_and_one_in_its_way_is() {
        // Arrange: one roller far above the top span, one pushing up into it
        let mut d = gt_force_drivetrain();
        let roller = |y: f64| Roller { name: "r".into(), body: Body::FrontTriangle, centre_m: [0.25, y], teeth: Some(16), radius_m: 0.0, span: Span::Top, side: ChainSide::Under, inertia_kgm2: 2e-6, enabled: true, threaded: false };
        d.rollers = vec![roller(0.3), roller(0.02)];
        d.rollers[1].side = ChainSide::Over;
        let bike = SimBike::new(&gt_force_linkage(), &d).unwrap();

        // Act
        let (_, path) = close_cage(&bike, &bike.pose(0.0), 5, &|_| true).unwrap();

        // Assert
        assert!(path.arc(WheelRole::Roller(0)).is_none());
        assert!(path.arc(WheelRole::Roller(1)).is_some());
    }
}
