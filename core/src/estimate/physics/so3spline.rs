//! The chassis attitude as a cumulative uniform cubic B-spline on SO(3)
//! (spec `2026-11-24-idl1-one-model-estimator-DRAFT.md` §2): between knots k
//! and k+1, with u in [0, 1) the fraction of the knot span,
//!
//! ```text
//! R(u) = R_k Exp(B1(u) W1) Exp(B2(u) W2) Exp(B3(u) W3),   Wj = Log(R_{k+j-1}^T R_{k+j})
//! ```
//!
//! with the cumulative cubic basis B1..B3. B1 + B2 + B3 = 1 + u, so control
//! rotations a constant rotation apart give exactly that constant body rate.
//! Generic over [`Real`]: lift u to hyper-dual numbers (u' = 1 / span) and the
//! rotation's first and second time derivatives come out with it, so the body
//! rate is vee(R^T R') and the angular acceleration vee(R^T R'') (R'^T R' is
//! symmetric and drops out of the vee).

use super::la::{c, exp, vee, M3, Real, V3};

/// Rotation matrix to rotation vector, rad. Smooth near the identity (the
/// spline's neighbouring control rotations are a knot span apart, far from pi);
/// not meant for angles near pi.
pub fn log<S: Real>(r: M3<S>) -> V3<S> {
    let s_v = vee(r); // sin(theta) * axis
    let s2 = s_v.dot(s_v);
    let co = (r.0[0][0] + r.0[1][1] + r.0[2][2] - c::<S>(1.0)) * 0.5;
    // branch on s^2: sqrt's derivative at 0 is infinite, so an exact identity would give NaN rates
    let k = if s2.re() < 1e-12 {
        c::<S>(1.0) + s2 * (1.0 / 6.0)
    } else {
        let s = s2.sqrt();
        s.atan2(co) / s
    };
    s_v.scale(k)
}

/// The cumulative cubic basis B1, B2, B3 at fraction `u` of a knot span.
pub fn cumulative_basis<S: Real>(u: S) -> [S; 3] {
    let u2 = u * u;
    let u3 = u2 * u;
    let six = 1.0 / 6.0;
    [
        (c::<S>(5.0) + u * 3.0 - u2 * 3.0 + u3) * six,
        (c::<S>(1.0) + u * 3.0 + u2 * 3.0 - u3 * 2.0) * six,
        u3 * six,
    ]
}

/// The spline's rotation (body -> nav) at fraction `u` of the span starting at
/// control rotation `ctrl[0]`.
pub fn rotation<S: Real>(ctrl: &[M3<S>; 4], u: S) -> M3<S> {
    let b = cumulative_basis(u);
    let mut r = ctrl[0];
    for j in 0..3 {
        let w = log(ctrl[j].t() * ctrl[j + 1]);
        r = r * exp(w.scale(b[j]));
    }
    r
}

/// Rotation, body rate (rad/s, body axes) and angular acceleration (rad/s^2,
/// body axes) at fraction `u` of a span of `span_s` seconds.
pub fn rotation_rate_accel<S: Real>(ctrl: &[M3<S>; 4], u: S, span_s: f64) -> (M3<S>, V3<S>, V3<S>) {
    use num_dual::HyperDual;
    let du = 1.0 / span_s;
    let uh = HyperDual::new(u, c::<S>(du), c::<S>(du), c::<S>(0.0));
    let ch: [M3<HyperDual<S>>; 4] = ctrl.map(|m| m.map(HyperDual::from_re));
    let rh = rotation(&ch, uh);
    let r = rh.map(|x| x.re);
    let rd = rh.map(|x| x.eps1);
    let rdd = rh.map(|x| x.eps1eps2);
    (r, vee(r.t() * rd), vee(r.t() * rdd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: V3<f64>, b: V3<f64>, tol: f64) -> bool {
        (0..3).all(|i| (a.0[i] - b.0[i]).abs() < tol)
    }

    fn controls(seed: f64) -> [M3<f64>; 4] {
        // four rotations a few degrees apart, about changing axes
        let mut r = exp(V3::from_f64([0.3, -0.2, 1.1]));
        std::array::from_fn(|i| {
            let k = i as f64 + seed;
            r = r * exp(V3::from_f64([0.02 * k.sin(), 0.03 * (1.3 * k).cos(), -0.025 + 0.01 * k]));
            r
        })
    }

    #[test]
    fn log_inverts_exp_near_and_away_from_the_identity() {
        // Arrange
        for w in [[1e-9, -2e-9, 3e-9], [0.01, 0.02, -0.03], [0.4, -1.1, 0.7]] {
            let w = V3::from_f64(w);

            // Act
            let back = log(exp(w));

            // Assert
            assert!(close(back, w, 1e-12), "{w:?} -> {back:?}");
        }
    }

    #[test]
    fn spline_of_a_constant_rotation_rate_gives_that_rate_and_no_acceleration() {
        // Arrange: controls 10 ms apart, turning at w about a fixed body axis
        let w = V3::from_f64([0.8, -1.5, 2.0]);
        let span = 0.01;
        let r0 = exp(V3::from_f64([0.2, 0.1, -0.4]));
        let ctrl: [M3<f64>; 4] = std::array::from_fn(|i| r0 * exp(w.scale(span * i as f64)));

        for u in [0.0, 0.3, 0.77] {
            // Act
            let (_, rate, acc) = rotation_rate_accel(&ctrl, u, span);

            // Assert
            assert!(close(rate, w, 1e-9), "u {u}: {rate:?}");
            assert!(close(acc, V3::zero(), 1e-6), "u {u}: {acc:?}");
        }
    }

    #[test]
    fn body_rate_and_acceleration_match_finite_differences_of_the_rotation() {
        // Arrange
        let ctrl = controls(0.4);
        let span = 0.005;
        let (u, h) = (0.41, 1e-4);

        // Act
        let (r, rate, acc) = rotation_rate_accel(&ctrl, u, span);
        let rate_at = |u: f64| rotation_rate_accel(&ctrl, u, span).1;

        // Assert: rate from the rotation's central difference, acceleration from the rate's
        let (rp, rm) = (rotation(&ctrl, u + h), rotation(&ctrl, u - h));
        let fd_rate = log(rm.t() * rp).scale(1.0 / (2.0 * h * span));
        assert!(close(rate, fd_rate, 1e-5 * (1.0 + rate.norm())), "{rate:?} vs {fd_rate:?}");
        let fd_acc = (rate_at(u + h) - rate_at(u - h)).scale(1.0 / (2.0 * h * span));
        assert!(close(acc, fd_acc, 1e-4 * (1.0 + acc.norm())), "{acc:?} vs {fd_acc:?}");
        let rrt = r * r.t();
        assert!((0..3).all(|i| (0..3).all(|j| (rrt.0[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12)));
    }

    #[test]
    fn spline_is_continuous_across_a_knot() {
        // Arrange: five controls, two spans sharing controls 1..4
        let c5 = {
            let a = controls(0.0);
            let b = controls(1.0);
            [a[0], a[1], a[2], a[3], a[3] * a[2].t() * b[3]]
        };
        let left = [c5[0], c5[1], c5[2], c5[3]];
        let right = [c5[1], c5[2], c5[3], c5[4]];
        let span = 0.01;

        // Act
        let (rl, wl, al) = rotation_rate_accel(&left, 1.0, span);
        let (rr, wr, ar) = rotation_rate_accel(&right, 0.0, span);

        // Assert: rotation, rate and acceleration continuous (a cubic spline is C2)
        assert!(close(log(rl.t() * rr), V3::zero(), 1e-12));
        assert!(close(wl, wr, 1e-9));
        assert!(close(al, ar, 1e-6 * (1.0 + al.norm())));
    }
}
