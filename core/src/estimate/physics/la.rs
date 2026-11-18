//! Three-vectors and 3x3 matrices over any [`Real`] scalar: `f64` for values,
//! num-dual's dual and hyper-dual numbers for exact derivatives. Small and
//! `Copy`, so the forward model can be written once and evaluated in any of
//! them. Matrices map a body's axes into its parent's (`v_parent = R * v_body`).

use num_dual::DualNum;
use std::ops::{Add, Mul, Neg, Sub};

/// A scalar the physics can be evaluated in: `f64`, or a (hyper-)dual number
/// over `f64` that carries derivatives alongside the value.
pub trait Real: DualNum<Primitive = f64> + Copy {}
impl<T: DualNum<Primitive = f64> + Copy> Real for T {}

/// The scalar `x` as any [`Real`] (derivative parts zero).
#[inline]
pub fn c<S: Real>(x: f64) -> S {
    S::from(x)
}

/// A 3-vector (units are the caller's: m, m/s, rad, rad/s ...).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3<S>(pub [S; 3]);

/// A 3x3 matrix, row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct M3<S>(pub [[S; 3]; 3]);

impl<S: Real> V3<S> {
    /// The zero vector.
    pub fn zero() -> Self {
        V3([c(0.0); 3])
    }
    /// From plain numbers.
    pub fn from_f64(v: [f64; 3]) -> Self {
        V3([c(v[0]), c(v[1]), c(v[2])])
    }
    /// Each component mapped through `f` (e.g. lifting to a dual type).
    pub fn map<T: Real>(self, f: impl Fn(S) -> T) -> V3<T> {
        V3([f(self.0[0]), f(self.0[1]), f(self.0[2])])
    }
    /// Dot product.
    pub fn dot(self, o: Self) -> S {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    /// Cross product.
    pub fn cross(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        V3([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    }
    /// Scaled by `k`.
    pub fn scale(self, k: S) -> Self {
        V3([self.0[0] * k, self.0[1] * k, self.0[2] * k])
    }
    /// Euclidean length.
    pub fn norm(self) -> S {
        self.dot(self).sqrt()
    }
    /// Unit vector along `self`.
    pub fn unit(self) -> Self {
        self.scale(self.norm().recip())
    }
}

impl<S: Real> Add for V3<S> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        V3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
}
impl<S: Real> Sub for V3<S> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        V3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}
impl<S: Real> Neg for V3<S> {
    type Output = Self;
    fn neg(self) -> Self {
        V3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl<S: Real> M3<S> {
    /// The identity.
    pub fn eye() -> Self {
        let (o, z) = (c(1.0), c(0.0));
        M3([[o, z, z], [z, o, z], [z, z, o]])
    }
    /// From plain numbers, row-major.
    pub fn from_f64(m: [[f64; 3]; 3]) -> Self {
        M3(m.map(|r| r.map(c)))
    }
    /// Each entry mapped through `f`.
    pub fn map<T: Real>(self, f: impl Fn(S) -> T) -> M3<T> {
        M3(self.0.map(|r| r.map(&f)))
    }
    /// Transpose (the inverse, for a rotation).
    pub fn t(self) -> Self {
        let m = self.0;
        M3([[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]])
    }
    /// Scaled by `k`.
    pub fn scale(self, k: S) -> Self {
        M3(self.0.map(|r| r.map(|x| x * k)))
    }
}

impl<S: Real> Add for M3<S> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut r = self.0;
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = r[i][j] + o.0[i][j];
            }
        }
        M3(r)
    }
}
impl<S: Real> Mul<V3<S>> for M3<S> {
    type Output = V3<S>;
    fn mul(self, v: V3<S>) -> V3<S> {
        let m = self.0;
        V3([0, 1, 2].map(|i| m[i][0] * v.0[0] + m[i][1] * v.0[1] + m[i][2] * v.0[2]))
    }
}
impl<S: Real> Mul for M3<S> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let (a, b) = (self.0, o.0);
        M3([0, 1, 2].map(|i| [0, 1, 2].map(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])))
    }
}

/// The cross-product matrix: `skew(v) * x = v x x`.
pub fn skew<S: Real>(v: V3<S>) -> M3<S> {
    let [x, y, z] = v.0;
    let o = c(0.0);
    M3([[o, -z, y], [z, o, -x], [-y, x, o]])
}

/// Inverse of [`skew`], from the skew-symmetric part of `m`.
pub fn vee<S: Real>(m: M3<S>) -> V3<S> {
    let h = c::<S>(0.5);
    let m = m.0;
    V3([(m[2][1] - m[1][2]) * h, (m[0][2] - m[2][0]) * h, (m[1][0] - m[0][1]) * h])
}

/// Rotation by `angle` (rad) about the unit `axis` (Rodrigues). Smooth at zero.
pub fn rot_axis<S: Real>(axis: V3<S>, angle: S) -> M3<S> {
    let k = skew(axis);
    let (s, co) = angle.sin_cos();
    M3::eye() + k.scale(s) + (k * k).scale(c::<S>(1.0) - co)
}

/// Rotation vector (rad) to matrix; small-angle safe.
pub fn exp<S: Real>(w: V3<S>) -> M3<S> {
    let th2 = w.dot(w);
    let (a, b) = if th2.re() < 1e-12 {
        (c::<S>(1.0) - th2 * (1.0 / 6.0), c::<S>(0.5) - th2 * (1.0 / 24.0))
    } else {
        let th = th2.sqrt();
        (th.sin() / th, (c::<S>(1.0) - th.cos()) / th2)
    };
    let k = skew(w);
    M3::eye() + k.scale(a) + (k * k).scale(b)
}

/// Unit quaternion `[w, x, y, z]` (the bike calibration's convention) to matrix.
pub fn quat_to_mat(q: [f64; 4]) -> M3<f64> {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [w, x, y, z] = q.map(|v| v / n);
    M3([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_rot_axis_and_is_a_rotation() {
        // Arrange
        let axis = V3::<f64>::from_f64([0.3, -0.5, 0.8]).unit();
        let angle = 0.7;

        // Act
        let a = exp(axis.scale(angle));
        let b = rot_axis(axis, angle);

        // Assert
        for i in 0..3 {
            for j in 0..3 {
                assert!((a.0[i][j] - b.0[i][j]).abs() < 1e-12);
                let rrt = (a * a.t()).0[i][j];
                assert!((rrt - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn vee_inverts_skew() {
        // Arrange
        let v = V3::<f64>::from_f64([1.0, -2.0, 0.5]);

        // Act
        let back = vee(skew(v));

        // Assert
        assert_eq!(back, v);
    }
}
