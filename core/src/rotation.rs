//! Rotation matrix application — nalgebra 3×3 matrix multiply on sensor vectors.
//!
//! Maps sensor body frame → vehicle frame (ISO 8855: X=forward, Y=left, Z=up).
//! Rotation matrix is produced by calibration::rotation_from_gravity() and stored
//! in idl0_config.json as `imu.orientation`.
//!
//! See docs/calibration.md and IDL0_SPEC.md §10, §18.

use nalgebra::{Matrix3, Quaternion, Unit, UnitQuaternion, Vector3};

/// Applies a 3×3 rotation matrix to a sensor vector, mapping sensor body frame
/// → vehicle frame (ISO 8855: X=forward, Y=left, Z=up).
///
/// `sensor_vec`: 3-element [x, y, z] in sensor body frame — raw LSB counts,
///               or any units consistent with the calibration matrix
/// `rotation`: 9-element row-major rotation matrix produced by
///             calibration::rotation_from_gravity()
///             [R00, R01, R02, R10, R11, R12, R20, R21, R22]
///
/// Returns 3-element [x, y, z] in vehicle frame, same units as `sensor_vec`.
///
/// nalgebra: Matrix3::from_row_slice() reconstructs the matrix from row-major
/// layout, then multiplies via standard matrix × vector product.
pub fn apply_rotation(sensor_vec: Vec<f64>, rotation: Vec<f64>) -> Vec<f64> {
    let r = Matrix3::from_row_slice(&rotation);
    let v = Vector3::new(sensor_vec[0], sensor_vec[1], sensor_vec[2]);
    let result = r * v;
    vec![result.x, result.y, result.z]
}

/// The skew-symmetric "hat" matrix `[v]_×` of a 3-vector, i.e. the matrix such
/// that `[v]_× w == v × w` for all `w`. Used to assemble SO(3) Jacobians and the
/// matrix exponential. Dimensionless operator: output units follow the operands.
///
/// nalgebra: built directly as a `Matrix3` (the canonical cross-product matrix).
pub fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y, //
        v.z, 0.0, -v.x, //
        -v.y, v.x, 0.0,
    )
}

/// SO(3) exponential map: rotation by angle `‖ω‖` (radians) about axis `ω/‖ω‖`,
/// from a rotation vector (scaled axis). This is the `⊞` retraction's rotation
/// half under the pinned **right-perturbation** convention `R · exp([δθ]_×)`.
///
/// nalgebra: `UnitQuaternion::from_scaled_axis` (quaternion form, numerically
/// stable for composing many small rotations along a trajectory).
pub fn exp_so3(omega: Vector3<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(omega)
}

/// SO(3) logarithm map: the rotation vector (scaled axis, radians) of `q`, the
/// inverse of [`exp_so3`]. This is the `⊟` (boxminus) rotation half. Returns the
/// minimal rotation vector (`‖·‖ ≤ π`).
///
/// nalgebra: `UnitQuaternion::scaled_axis`.
pub fn log_so3(q: &UnitQuaternion<f64>) -> Vector3<f64> {
    q.scaled_axis()
}

/// The minimal (shortest-arc) rotation that maps direction `from` onto direction
/// `to` (magnitudes ignored — only directions matter). Used for gravity-leveling
/// (align measured "up" to vehicle Z) and any vector-to-vector alignment.
///
/// Degenerate antiparallel case (`from ≈ -to`): the shortest rotation is a 180°
/// flip about *any* axis perpendicular to `from` — non-unique, so a stable
/// perpendicular axis is chosen. (For gravity-leveling this is the sensor-exactly-
/// upside-down case; the in-plane axis is unobservable from gravity anyway.)
///
/// nalgebra: `UnitQuaternion::rotation_between` (returns `None` only when
/// antiparallel/degenerate, where the fallback flip is supplied).
pub fn rotation_between(from: Vector3<f64>, to: Vector3<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::rotation_between(&from, &to).unwrap_or_else(|| {
        // Antiparallel: 180° about a perpendicular to `from`. cross with X unless
        // `from` is ~parallel to X, in which case cross with Y.
        let perp = from.cross(&Vector3::x());
        let axis = if perp.norm() > 1e-6 { perp } else { from.cross(&Vector3::y()) };
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), std::f64::consts::PI)
    })
}

/// Inverse of [`skew`]: extracts the 3-vector `v` from its skew-symmetric matrix
/// `[v]_×`. Reads the three independent off-diagonal entries; the matrix is
/// assumed skew-symmetric (Jacobian/log-map plumbing always supplies one).
pub fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

/// The SO(3) adjoint `Adj(R)`: the linear map on `so(3)` satisfying
/// `R [ω]_× Rᵀ == [Adj(R) ω]_×`. For SO(3) it is simply the rotation matrix `R`.
/// Used to transport process noise and assemble the attitude block of the
/// error-state transition Jacobian `F`.
pub fn adjoint(q: &UnitQuaternion<f64>) -> Matrix3<f64> {
    q.to_rotation_matrix().into_inner()
}

/// The SO(3) **right Jacobian** `J_r(φ)`: the linear map satisfying
/// `Exp(φ + δφ) ≈ Exp(φ) · Exp(J_r(φ)·δφ)` to first order. It relates a
/// perturbation of a rotation vector to the right-tangent increment of the
/// resulting rotation, and appears in the error-state transition Jacobian as the
/// gyro-bias → attitude coupling (`∂δθ⁺/∂δb_g = −J_r(ω dt)·dt`).
///
/// Closed form: `J_r = I − ((1−cosθ)/θ²)[φ]× + ((θ−sinθ)/θ³)[φ]×²`, `θ = ‖φ‖`,
/// with the `θ→0` limit `I − ½[φ]×` (series, avoids the 0/0). Dimensionless.
pub fn right_jacobian_so3(phi: Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    let k = skew(phi);
    if theta < 1e-8 {
        // Series limit: J_r ≈ I − ½[φ]× (the [φ]×² term is O(θ²), negligible here).
        Matrix3::identity() - 0.5 * k
    } else {
        let a = (1.0 - theta.cos()) / (theta * theta);
        let b = (theta - theta.sin()) / (theta * theta * theta);
        Matrix3::identity() - a * k + b * k * k
    }
}

/// Swing–twist decomposition of `q` about unit `axis`: returns
/// `(twist_angle_rad, swing)` with `q = swing · twist`, where `twist` is the
/// rotation component about `axis` and `swing` is the residual off-axis rotation.
/// Extracts the **steering scalar** (twist about the known, tilted steer axis);
/// `twist_angle` is signed about `axis`, in radians. Valid away from a 180° twist.
pub fn swing_twist(q: &UnitQuaternion<f64>, axis: Vector3<f64>) -> (f64, UnitQuaternion<f64>) {
    let a = axis.normalize();
    let quat = q.quaternion();
    let proj = quat.imag().dot(&a) * a;
    let twist = UnitQuaternion::from_quaternion(Quaternion::from_parts(quat.scalar(), proj));
    let twist_angle = 2.0 * quat.imag().dot(&a).atan2(quat.scalar());
    let swing = q * twist.inverse();
    (twist_angle, swing)
}

/// Rigid-body acceleration transfer: the linear acceleration at a point offset by
/// `lever` (m) from a reference point on the **same rigid body**, given the body's
/// angular velocity `omega` (rad/s) and angular acceleration `omega_dot` (rad/s²):
/// `a = a_ref + ω̇×L + ω×(ω×L)`. The `ω̇×L` (tangential/Euler) and `ω×(ω×L)`
/// (centripetal) terms are **mandatory** — omitting them aliases body rotation into
/// false suspension travel. All vectors share one frame; output units match
/// `a_ref` (m/s²).
pub fn lever_arm_accel(
    a_ref: Vector3<f64>,
    omega: Vector3<f64>,
    omega_dot: Vector3<f64>,
    lever: Vector3<f64>,
) -> Vector3<f64> {
    a_ref + omega_dot.cross(&lever) + omega.cross(&omega.cross(&lever))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn skew_times_vector_equals_cross_product() {
        // Arrange — the hat operator [a]_× must satisfy [a]_× b == a × b.
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);

        // Act
        let via_skew = skew(a) * b;

        // Assert
        assert_relative_eq!(via_skew, a.cross(&b), epsilon = 1e-12);
    }

    #[test]
    fn vee_inverts_skew() {
        // Arrange
        let v = Vector3::new(-2.0, 7.0, 0.5);

        // Act + Assert — vee is the inverse of the hat operator.
        assert_relative_eq!(vee(&skew(v)), v, epsilon = 1e-12);
    }

    #[test]
    fn exp_so3_rotates_90_about_z_maps_x_to_y() {
        // Arrange — a π/2 rotation vector about +Z.
        let omega = Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2);

        // Act
        let rotated = exp_so3(omega) * Vector3::new(1.0, 0.0, 0.0);

        // Assert — physically: 90° about Z maps X → Y.
        assert_relative_eq!(rotated, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn exp_log_so3_round_trips() {
        // Arrange — a generic rotation vector, ‖ω‖ ≈ 1.34 rad (well within ±π).
        let omega = Vector3::new(0.3, -0.7, 1.1);

        // Act + Assert — log is the inverse of exp on SO(3).
        assert_relative_eq!(log_so3(&exp_so3(omega)), omega, epsilon = 1e-12);
    }

    #[test]
    fn rotation_between_maps_from_onto_to() {
        // Arrange — the minimal rotation taking +X onto +Z.
        let from = Vector3::new(1.0, 0.0, 0.0);
        let to = Vector3::new(0.0, 0.0, 1.0);

        // Act
        let mapped = rotation_between(from, to) * from;

        // Assert
        assert_relative_eq!(mapped, to, epsilon = 1e-12);
    }

    #[test]
    fn rotation_between_handles_antiparallel_flip() {
        // Arrange — +Z onto -Z is the degenerate (antiparallel) case where the
        // minimal rotation is non-unique; the fallback must still be a valid flip.
        let from = Vector3::new(0.0, 0.0, 1.0);
        let to = Vector3::new(0.0, 0.0, -1.0);

        // Act
        let mapped = rotation_between(from, to) * from;

        // Assert
        assert_relative_eq!(mapped, to, epsilon = 1e-12);
    }

    #[test]
    fn adjoint_conjugates_the_hat_operator() {
        // Arrange — the SO(3) adjoint must satisfy R [ω]_× Rᵀ == [Adj(R) ω]_×.
        let q = exp_so3(Vector3::new(0.2, -0.5, 0.9));
        let omega = Vector3::new(1.0, -2.0, 0.5);
        let r = q.to_rotation_matrix();

        // Act
        let lhs = r.matrix() * skew(omega) * r.matrix().transpose();
        let rhs = skew(adjoint(&q) * omega);

        // Assert
        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
    }

    #[test]
    fn right_jacobian_at_zero_is_identity() {
        // Arrange + Act — J_r(0) = I (the small-angle limit).
        let jr = right_jacobian_so3(Vector3::zeros());

        // Assert
        assert_relative_eq!(jr, Matrix3::identity(), epsilon = 1e-12);
    }

    #[test]
    fn right_jacobian_satisfies_its_defining_perturbation_identity() {
        // Arrange — J_r(φ) must satisfy Exp(φ)ᵀ Exp(φ+εv) ≈ Exp(J_r(φ)·εv) to
        // first order, i.e. log(Exp(φ)ᵀ Exp(φ+εv)) ≈ J_r(φ)·εv.
        let phi = Vector3::new(0.2, -0.3, 0.5);
        let v = Vector3::new(1.0, -0.5, 0.25);
        let eps = 1e-6;

        // Act — finite-difference the right-tangent increment.
        let fd = log_so3(&(exp_so3(phi).inverse() * exp_so3(phi + eps * v))) / eps;
        let analytic = right_jacobian_so3(phi) * v;

        // Assert
        assert_relative_eq!(fd, analytic, epsilon = 1e-6);
    }

    #[test]
    fn swing_twist_recovers_twist_angle_about_axis() {
        // Arrange — a pure rotation of ψ about a tilted steer-like axis.
        let axis = Vector3::new(0.3, 0.0, 0.95).normalize();
        let psi = 0.6_f64;
        let q = exp_so3(axis * psi);

        // Act
        let (twist_angle, _swing) = swing_twist(&q, axis);

        // Assert — the twist about the axis equals ψ.
        assert_relative_eq!(twist_angle, psi, epsilon = 1e-9);
    }

    #[test]
    fn lever_arm_accel_no_rotation_equals_reference_accel() {
        // Arrange — no angular velocity/accel: the offset point sees a_ref unchanged.
        let a_ref = Vector3::new(1.0, -2.0, 9.81);

        // Act
        let a = lever_arm_accel(
            a_ref,
            Vector3::zeros(),
            Vector3::zeros(),
            Vector3::new(0.1, 0.2, -0.3),
        );

        // Assert
        assert_relative_eq!(a, a_ref, epsilon = 1e-12);
    }

    #[test]
    fn lever_arm_accel_is_centripetal_under_steady_spin() {
        // Arrange — point at lever [0.5,0,0], steady spin ω=3 about +Z, no a_ref,
        // no angular accel. Centripetal accel = −ω²·lever = [−4.5, 0, 0].
        let omega = Vector3::new(0.0, 0.0, 3.0);
        let lever = Vector3::new(0.5, 0.0, 0.0);

        // Act
        let a = lever_arm_accel(Vector3::zeros(), omega, Vector3::zeros(), lever);

        // Assert
        assert_relative_eq!(a, Vector3::new(-4.5, 0.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn apply_rotation_identity_matrix_returns_unchanged_vector() {
        // Arrange
        let rotation = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let input = vec![3.0, -7.0, 42.0];

        // Act
        let result = apply_rotation(input.clone(), rotation);

        // Assert — identity rotation leaves the vector unchanged
        assert_relative_eq!(result[0], input[0], epsilon = 1e-9);
        assert_relative_eq!(result[1], input[1], epsilon = 1e-9);
        assert_relative_eq!(result[2], input[2], epsilon = 1e-9);
    }

    #[test]
    fn apply_rotation_90_degrees_about_z_swaps_x_and_y() {
        // Arrange — 90° rotation around Z: X→Y, Y→-X, Z→Z
        // Row-major: [[0,-1,0],[1,0,0],[0,0,1]]
        let rotation = vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let input = vec![1.0, 0.0, 0.0]; // pointing along X axis

        // Act
        let result = apply_rotation(input, rotation);

        // Assert — physically: 90° about Z maps X → Y
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-9);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn apply_rotation_90_degrees_about_z_maps_y_to_negative_x() {
        // Arrange — same 90° Z rotation, input along Y axis
        let rotation = vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let input = vec![0.0, 1.0, 0.0];

        // Act
        let result = apply_rotation(input, rotation);

        // Assert — physically: 90° about Z maps Y → -X
        assert_relative_eq!(result[0], -1.0, epsilon = 1e-9);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-9);
    }
}
