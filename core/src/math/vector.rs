//! Vector & rotation primitives for the math-channel expression language.
//!
//! Adds an internal [`Vec3`](Value::Vec3) value type and the vector/rotation
//! function set (SPEC §19, "Vector" / "Rotation" rows). All operations work
//! element-wise over the component buffers and reuse the evaluator's existing
//! broadcasting rules ([`elemwise`](crate::math::eval::elemwise) /
//! [`map_value`](crate::math::eval::map_value)) — a scalar component broadcasts
//! across every sample, channel components must share rate + length.
//!
//! `Vec3` is an intermediate value only: the top-level result of an expression
//! must reduce to a scalar channel (`vx`/`vy`/`vz` or `norm`).

use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

use crate::math::eval::{elemwise, map_value};
use crate::math::value::{ChannelValue, Value, Vec3Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

fn err(kind: MathEvalErrorKind, msg: impl Into<String>) -> MathEvalError {
    MathEvalError::new(kind, msg)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Channel(_) => "channel",
        Value::Scalar(_) => "scalar",
        Value::Str(_) => "string",
        Value::Vec3(_) => "vec3",
    }
}

/// Borrows the [`Vec3Value`] inside a `Value`, or a typed error otherwise.
fn require_vec3<'a>(v: &'a Value, ctx: &str) -> Result<&'a Vec3Value, MathEvalError> {
    match v {
        Value::Vec3(b) => Ok(b),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected a 3-vector (e.g. vec(x,y,z)), got {}", type_name(other)),
        )),
    }
}

/// Validates that a value is usable as a vector component or scale (scalar or
/// channel — not a string or nested vector) and clones it.
fn require_scalar_or_channel(v: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    match v {
        Value::Scalar(_) | Value::Channel(_) => Ok(v.clone()),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected a scalar or channel, got {}", type_name(other)),
        )),
    }
}

// Element-wise component combinators built on the evaluator's broadcasting
// rules: scalar↔channel broadcast, channel↔channel requires equal rate+length.
fn add2(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    elemwise(a.clone(), b.clone(), ctx, |p, q| Ok(p + q))
}
fn sub2(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    elemwise(a.clone(), b.clone(), ctx, |p, q| Ok(p - q))
}
fn mul2(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    elemwise(a.clone(), b.clone(), ctx, |p, q| Ok(p * q))
}
fn scale1(v: &Value, k: f64) -> Result<Value, MathEvalError> {
    map_value(v.clone(), move |x| x * k)
}

/// Assembles a [`Value::Vec3`] from three component values. Each component must
/// be a scalar (broadcast across all samples) or a channel — a string or nested
/// vector is rejected.
pub fn make_vec3(x: &Value, y: &Value, z: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: require_scalar_or_channel(x, ctx)?,
        y: require_scalar_or_channel(y, ctx)?,
        z: require_scalar_or_channel(z, ctx)?,
    })))
}

/// Extracts component `idx` (0=x, 1=y, 2=z) of a vector, returning the
/// component's underlying scalar or channel value. Backs `vx`/`vy`/`vz`.
pub fn component(v: &Value, idx: usize, ctx: &str) -> Result<Value, MathEvalError> {
    let b = require_vec3(v, ctx)?;
    Ok(match idx {
        0 => b.x.clone(),
        1 => b.y.clone(),
        2 => b.z.clone(),
        _ => return Err(err(MathEvalErrorKind::Runtime, format!("{ctx}: component index {idx} out of range"))),
    })
}

/// Element-wise vector sum `a + b`.
pub fn vadd(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let (a, b) = (require_vec3(a, ctx)?, require_vec3(b, ctx)?);
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: add2(&a.x, &b.x, ctx)?,
        y: add2(&a.y, &b.y, ctx)?,
        z: add2(&a.z, &b.z, ctx)?,
    })))
}

/// Element-wise vector difference `a - b`.
pub fn vsub(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let (a, b) = (require_vec3(a, ctx)?, require_vec3(b, ctx)?);
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: sub2(&a.x, &b.x, ctx)?,
        y: sub2(&a.y, &b.y, ctx)?,
        z: sub2(&a.z, &b.z, ctx)?,
    })))
}

/// Scales every component of `v` by the scalar/channel `s`.
pub fn vscale(v: &Value, s: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let v = require_vec3(v, ctx)?;
    let s = require_scalar_or_channel(s, ctx)?;
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: mul2(&v.x, &s, ctx)?,
        y: mul2(&v.y, &s, ctx)?,
        z: mul2(&v.z, &s, ctx)?,
    })))
}

/// Vector cross product `a × b` (right-handed).
pub fn cross(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let (a, b) = (require_vec3(a, ctx)?, require_vec3(b, ctx)?);
    // cx = ay·bz − az·by ; cy = az·bx − ax·bz ; cz = ax·by − ay·bx
    let cx = sub2(&mul2(&a.y, &b.z, ctx)?, &mul2(&a.z, &b.y, ctx)?, ctx)?;
    let cy = sub2(&mul2(&a.z, &b.x, ctx)?, &mul2(&a.x, &b.z, ctx)?, ctx)?;
    let cz = sub2(&mul2(&a.x, &b.y, ctx)?, &mul2(&a.y, &b.x, ctx)?, ctx)?;
    Ok(Value::Vec3(Box::new(Vec3Value { x: cx, y: cy, z: cz })))
}

/// Vector dot product `a · b` → scalar/channel.
pub fn dot(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let (a, b) = (require_vec3(a, ctx)?, require_vec3(b, ctx)?);
    let xx = mul2(&a.x, &b.x, ctx)?;
    let yy = mul2(&a.y, &b.y, ctx)?;
    let zz = mul2(&a.z, &b.z, ctx)?;
    add2(&add2(&xx, &yy, ctx)?, &zz, ctx)
}

/// Euclidean magnitude `|v| = sqrt(v · v)` → scalar/channel.
pub fn norm(v: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let d = dot(v, v, ctx)?;
    map_value(d, f64::sqrt)
}

/// Unit vector `v / |v|`. A zero-length vector yields `NaN` components
/// (direction undefined) rather than an error, matching the `NaN`-on-bad-sample
/// convention of the other functions.
pub fn normalize(v: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let n = norm(v, ctx)?;
    let recip = map_value(n, |x| if x == 0.0 { f64::NAN } else { 1.0 / x })?;
    vscale(v, &recip, ctx)
}

/// Angle between `a` and `b` in radians, in `[0, π]`. Computed as
/// `atan2(|a × b|, a · b)` — numerically robust near 0 and π, unlike
/// `acos(dot / (|a||b|))` which loses precision and can stray outside `[-1, 1]`.
pub fn angle(a: &Value, b: &Value, ctx: &str) -> Result<Value, MathEvalError> {
    let cross_mag = norm(&cross(a, b, ctx)?, ctx)?;
    let d = dot(a, b, ctx)?;
    elemwise(cross_mag, d, ctx, |y, x| Ok(y.atan2(x)))
}

// ---- rotations ----

/// Applies a constant 3×3 rotation/transform matrix to `v` as the linear
/// combination `out_i = Σ_j m[i][j]·v_j`. Reusing `map_value`/`elemwise` (vs a
/// per-sample `nalgebra` matrix-vector product) preserves the scalar/channel
/// kind of each component for free and avoids rebuilding a `Matrix3` per sample.
/// `m` is indexed logically `m[(row, col)]`, so callers pass a row-major matrix.
fn apply_const_matrix(v: &Vec3Value, m: &Matrix3<f64>, ctx: &str) -> Result<Value, MathEvalError> {
    let row = |r0: f64, r1: f64, r2: f64| -> Result<Value, MathEvalError> {
        let a = scale1(&v.x, r0)?;
        let b = scale1(&v.y, r1)?;
        let c = scale1(&v.z, r2)?;
        add2(&add2(&a, &b, ctx)?, &c, ctx)
    };
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: row(m[(0, 0)], m[(0, 1)], m[(0, 2)])?,
        y: row(m[(1, 0)], m[(1, 1)], m[(1, 2)])?,
        z: row(m[(2, 0)], m[(2, 1)], m[(2, 2)])?,
    })))
}

/// Applies a constant 3×3 row-major matrix to `v`. `m = [r00, r01, r02, r10,
/// …, r22]`, the same row-major layout as `rotation::apply_rotation` /
/// `rotation_from_gravity` output (SPEC §19 "Rust API notes").
pub fn rotate_mat(v: &Value, m: &[f64; 9], ctx: &str) -> Result<Value, MathEvalError> {
    let v = require_vec3(v, ctx)?;
    // nalgebra Matrix3::new takes arguments in row-major order.
    let mat = Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    apply_const_matrix(v, &mat, ctx)
}

/// Rotates `v` about the (constant) axis `(ax, ay, az)` by `angle` radians.
/// nalgebra: `Rotation3::from_axis_angle` (axis is normalized internally).
pub fn rotate_axis(
    v: &Value,
    ax: f64,
    ay: f64,
    az: f64,
    angle: f64,
    ctx: &str,
) -> Result<Value, MathEvalError> {
    let v = require_vec3(v, ctx)?;
    let axis = Vector3::new(ax, ay, az);
    if axis.norm() == 0.0 {
        return Err(err(
            MathEvalErrorKind::Runtime,
            format!("{ctx}: rotation axis (ax, ay, az) must be non-zero"),
        ));
    }
    let mat = Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle).into_inner();
    apply_const_matrix(v, &mat, ctx)
}

/// Rotates `v` by intrinsic roll/pitch/yaw (radians) via nalgebra
/// `Rotation3::from_euler_angles`. When all three angles are scalars the
/// rotation is constant (one matrix); when any is a channel the rotation is
/// rebuilt per sample (time-varying orientation), with the vector components
/// broadcast to the channel length.
pub fn rotate_euler(
    v: &Value,
    roll: &Value,
    pitch: &Value,
    yaw: &Value,
    ctx: &str,
) -> Result<Value, MathEvalError> {
    let vv = require_vec3(v, ctx)?;
    if let (Value::Scalar(r), Value::Scalar(p), Value::Scalar(y)) = (roll, pitch, yaw) {
        let mat = Rotation3::from_euler_angles(*r, *p, *y).into_inner();
        return apply_const_matrix(vv, &mat, ctx);
    }
    rotate_euler_varying(vv, roll, pitch, yaw, ctx)
}

// Per-sample Euler rotation. All channel operands must share rate + length;
// scalar operands broadcast. Output components are channels at the common rate.
fn rotate_euler_varying(
    v: &Vec3Value,
    roll: &Value,
    pitch: &Value,
    yaw: &Value,
    ctx: &str,
) -> Result<Value, MathEvalError> {
    let (n, rate) = broadcast_shape(&[&v.x, &v.y, &v.z, roll, pitch, yaw], ctx)?;
    let vx = materialize(&v.x, n, ctx)?;
    let vy = materialize(&v.y, n, ctx)?;
    let vz = materialize(&v.z, n, ctx)?;
    let rr = materialize(roll, n, ctx)?;
    let pp = materialize(pitch, n, ctx)?;
    let yy = materialize(yaw, n, ctx)?;
    let mut ox = vec![0.0; n];
    let mut oy = vec![0.0; n];
    let mut oz = vec![0.0; n];
    for i in 0..n {
        let rot = Rotation3::from_euler_angles(rr[i], pp[i], yy[i]);
        let out = rot * Vector3::new(vx[i], vy[i], vz[i]);
        ox[i] = out.x;
        oy[i] = out.y;
        oz[i] = out.z;
    }
    Ok(Value::Vec3(Box::new(Vec3Value {
        x: chan_or_scalar(ox, rate),
        y: chan_or_scalar(oy, rate),
        z: chan_or_scalar(oz, rate),
    })))
}

// Common (length, sample_rate) across a set of operands: every channel must
// agree on both; scalars impose no shape. No channel → (1, 0.0).
fn broadcast_shape(values: &[&Value], ctx: &str) -> Result<(usize, f64), MathEvalError> {
    let mut shape: Option<(usize, f64)> = None;
    for v in values {
        match v {
            Value::Scalar(_) => {}
            Value::Channel(c) => match shape {
                None => shape = Some((c.samples.len(), c.sample_rate_hz)),
                Some((len, rate)) => {
                    if len != c.samples.len() || rate != c.sample_rate_hz {
                        return Err(err(
                            MathEvalErrorKind::Runtime,
                            format!(
                                "{ctx}: channel arguments must share sample rate and length \
                                 ({rate} Hz × {len} vs {} Hz × {})",
                                c.sample_rate_hz,
                                c.samples.len()
                            ),
                        ));
                    }
                }
            },
            other => {
                return Err(err(
                    MathEvalErrorKind::Type,
                    format!("{ctx}: expected scalar or channel, got {}", type_name(other)),
                ))
            }
        }
    }
    Ok(shape.unwrap_or((1, 0.0)))
}

// Expands a scalar to `n` copies or returns a channel's samples (length must be
// `n` — guaranteed by a prior broadcast_shape check, re-verified here).
fn materialize(v: &Value, n: usize, ctx: &str) -> Result<Vec<f64>, MathEvalError> {
    match v {
        Value::Scalar(x) => Ok(vec![*x; n]),
        Value::Channel(c) if c.samples.len() == n => Ok(c.samples.to_vec()),
        Value::Channel(c) => Err(err(
            MathEvalErrorKind::Runtime,
            format!("{ctx}: channel length {} does not match {n}", c.samples.len()),
        )),
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("{ctx}: expected scalar or channel, got {}", type_name(other)),
        )),
    }
}

// A buffer at a positive rate is a channel; a rate-0 single sample is a scalar.
fn chan_or_scalar(samples: Vec<f64>, rate: f64) -> Value {
    if rate > 0.0 {
        Value::Channel(ChannelValue { samples: std::sync::Arc::from(samples), sample_rate_hz: rate, channel_id: None })
    } else {
        Value::Scalar(samples[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn chan(samples: Vec<f64>, rate: f64) -> Value {
        Value::Channel(ChannelValue { samples: std::sync::Arc::from(samples), sample_rate_hz: rate, channel_id: None })
    }
    fn vec3(x: f64, y: f64, z: f64) -> Value {
        make_vec3(&Value::Scalar(x), &Value::Scalar(y), &Value::Scalar(z), "test").unwrap()
    }
    fn sx(v: &Value) -> f64 {
        match component(v, 0, "test").unwrap() {
            Value::Scalar(x) => x,
            _ => panic!("expected scalar component"),
        }
    }
    fn sy(v: &Value) -> f64 {
        match component(v, 1, "test").unwrap() {
            Value::Scalar(x) => x,
            _ => panic!("expected scalar component"),
        }
    }
    fn sz(v: &Value) -> f64 {
        match component(v, 2, "test").unwrap() {
            Value::Scalar(x) => x,
            _ => panic!("expected scalar component"),
        }
    }

    // ---- build / access ----

    #[test]
    fn make_vec3_from_scalars_yields_extractable_components() {
        // Act
        let v = vec3(1.0, 2.0, 3.0);

        // Assert
        assert_eq!(sx(&v), 1.0);
        assert_eq!(sy(&v), 2.0);
        assert_eq!(sz(&v), 3.0);
    }

    #[test]
    fn make_vec3_rejects_string_component() {
        // Act
        let e = make_vec3(&Value::Str("x".into()), &Value::Scalar(0.0), &Value::Scalar(0.0), "vec")
            .unwrap_err();

        // Assert
        assert_eq!(e.kind, MathEvalErrorKind::Type);
    }

    #[test]
    fn component_of_non_vector_is_type_error() {
        // Act
        let e = component(&Value::Scalar(1.0), 0, "vx").unwrap_err();

        // Assert
        assert_eq!(e.kind, MathEvalErrorKind::Type);
    }

    #[test]
    fn component_preserves_channel_buffer() {
        // Arrange — a vector whose x is a channel.
        let v = make_vec3(&chan(vec![7.0, 8.0], 50.0), &Value::Scalar(0.0), &Value::Scalar(0.0), "vec")
            .unwrap();

        // Act
        let x = component(&v, 0, "vx").unwrap();

        // Assert
        match x {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![7.0, 8.0].into());
                assert_eq!(c.sample_rate_hz, 50.0);
            }
            _ => panic!("expected channel"),
        }
    }

    // ---- algebra ----

    #[test]
    fn vadd_sums_components() {
        // Act
        let v = vadd(&vec3(1.0, 2.0, 3.0), &vec3(10.0, 20.0, 30.0), "vadd").unwrap();

        // Assert
        assert_eq!((sx(&v), sy(&v), sz(&v)), (11.0, 22.0, 33.0));
    }

    #[test]
    fn vsub_subtracts_components() {
        // Act
        let v = vsub(&vec3(10.0, 20.0, 30.0), &vec3(1.0, 2.0, 3.0), "vsub").unwrap();

        // Assert
        assert_eq!((sx(&v), sy(&v), sz(&v)), (9.0, 18.0, 27.0));
    }

    #[test]
    fn vscale_multiplies_every_component() {
        // Act
        let v = vscale(&vec3(1.0, 2.0, 3.0), &Value::Scalar(2.0), "vscale").unwrap();

        // Assert
        assert_eq!((sx(&v), sy(&v), sz(&v)), (2.0, 4.0, 6.0));
    }

    #[test]
    fn cross_x_cross_y_is_z() {
        // Act — right-hand rule: x̂ × ŷ = ẑ.
        let v = cross(&vec3(1.0, 0.0, 0.0), &vec3(0.0, 1.0, 0.0), "cross").unwrap();

        // Assert
        assert_relative_eq!(sx(&v), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sy(&v), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sz(&v), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cross_is_anticommutative() {
        // Act — ŷ × x̂ = -ẑ.
        let v = cross(&vec3(0.0, 1.0, 0.0), &vec3(1.0, 0.0, 0.0), "cross").unwrap();

        // Assert
        assert_relative_eq!(sz(&v), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn dot_sums_componentwise_products() {
        // Act — 1·4 + 2·5 + 3·6 = 32.
        let d = dot(&vec3(1.0, 2.0, 3.0), &vec3(4.0, 5.0, 6.0), "dot").unwrap();

        // Assert
        assert!(matches!(d, Value::Scalar(x) if (x - 32.0).abs() < 1e-12));
    }

    #[test]
    fn dot_of_orthogonal_vectors_is_zero() {
        // Act
        let d = dot(&vec3(1.0, 0.0, 0.0), &vec3(0.0, 5.0, 0.0), "dot").unwrap();

        // Assert
        assert!(matches!(d, Value::Scalar(x) if x.abs() < 1e-12));
    }

    #[test]
    fn norm_of_3_4_0_is_5() {
        // Act
        let n = norm(&vec3(3.0, 4.0, 0.0), "norm").unwrap();

        // Assert
        assert!(matches!(n, Value::Scalar(x) if (x - 5.0).abs() < 1e-12));
    }

    #[test]
    fn normalize_yields_unit_vector() {
        // Act
        let u = normalize(&vec3(0.0, 3.0, 0.0), "normalize").unwrap();

        // Assert
        assert_relative_eq!(sx(&u), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sy(&u), 1.0, epsilon = 1e-12);
        assert_relative_eq!(sz(&u), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn angle_between_orthogonal_axes_is_half_pi() {
        // Act
        let a = angle(&vec3(1.0, 0.0, 0.0), &vec3(0.0, 1.0, 0.0), "angle").unwrap();

        // Assert
        assert!(matches!(a, Value::Scalar(x) if (x - PI / 2.0).abs() < 1e-9));
    }

    #[test]
    fn angle_of_parallel_vectors_is_zero() {
        // Act
        let a = angle(&vec3(2.0, 0.0, 0.0), &vec3(5.0, 0.0, 0.0), "angle").unwrap();

        // Assert
        assert!(matches!(a, Value::Scalar(x) if x.abs() < 1e-9));
    }

    #[test]
    fn dot_with_channel_component_returns_channel() {
        // Arrange — a = ([1,2], 0, 0), b = ([3,4], 0, 0); dot = [3, 8].
        let a = make_vec3(&chan(vec![1.0, 2.0], 10.0), &Value::Scalar(0.0), &Value::Scalar(0.0), "vec")
            .unwrap();
        let b = make_vec3(&chan(vec![3.0, 4.0], 10.0), &Value::Scalar(0.0), &Value::Scalar(0.0), "vec")
            .unwrap();

        // Act
        let d = dot(&a, &b, "dot").unwrap();

        // Assert
        match d {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![3.0, 8.0].into());
                assert_eq!(c.sample_rate_hz, 10.0);
            }
            _ => panic!("expected channel"),
        }
    }

    // ---- rotations ----

    #[test]
    fn rotate_mat_identity_leaves_vector_unchanged() {
        // Arrange
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Act
        let v = rotate_mat(&vec3(3.0, -7.0, 42.0), &id, "rotate_mat").unwrap();

        // Assert
        assert_relative_eq!(sx(&v), 3.0, epsilon = 1e-12);
        assert_relative_eq!(sy(&v), -7.0, epsilon = 1e-12);
        assert_relative_eq!(sz(&v), 42.0, epsilon = 1e-12);
    }

    #[test]
    fn rotate_mat_90_about_z_maps_x_to_y() {
        // Arrange — row-major 90° about Z: [[0,-1,0],[1,0,0],[0,0,1]].
        let r = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        // Act
        let v = rotate_mat(&vec3(1.0, 0.0, 0.0), &r, "rotate_mat").unwrap();

        // Assert
        assert_relative_eq!(sx(&v), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sy(&v), 1.0, epsilon = 1e-12);
        assert_relative_eq!(sz(&v), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rotate_axis_about_z_maps_x_to_y() {
        // Act — 90° about +Z.
        let v = rotate_axis(&vec3(1.0, 0.0, 0.0), 0.0, 0.0, 1.0, PI / 2.0, "rotate_axis").unwrap();

        // Assert
        assert_relative_eq!(sx(&v), 0.0, epsilon = 1e-9);
        assert_relative_eq!(sy(&v), 1.0, epsilon = 1e-9);
        assert_relative_eq!(sz(&v), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn rotate_euler_yaw_maps_x_to_y() {
        // Act — yaw 90° about Z, roll=pitch=0.
        let v = rotate_euler(
            &vec3(1.0, 0.0, 0.0),
            &Value::Scalar(0.0),
            &Value::Scalar(0.0),
            &Value::Scalar(PI / 2.0),
            "rotate_euler",
        )
        .unwrap();

        // Assert
        assert_relative_eq!(sx(&v), 0.0, epsilon = 1e-9);
        assert_relative_eq!(sy(&v), 1.0, epsilon = 1e-9);
        assert_relative_eq!(sz(&v), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn rotate_euler_with_channel_yaw_is_per_sample() {
        // Arrange — yaw channel [0, π/2] applied to (1,0,0): sample0 unchanged,
        // sample1 rotated to (0,1,0). Output components are channels.
        let v = rotate_euler(
            &vec3(1.0, 0.0, 0.0),
            &Value::Scalar(0.0),
            &Value::Scalar(0.0),
            &chan(vec![0.0, PI / 2.0], 20.0),
            "rotate_euler",
        )
        .unwrap();

        // Assert
        match (component(&v, 0, "vx").unwrap(), component(&v, 1, "vy").unwrap()) {
            (Value::Channel(cx), Value::Channel(cy)) => {
                assert_relative_eq!(cx.samples[0], 1.0, epsilon = 1e-9);
                assert_relative_eq!(cy.samples[0], 0.0, epsilon = 1e-9);
                assert_relative_eq!(cx.samples[1], 0.0, epsilon = 1e-9);
                assert_relative_eq!(cy.samples[1], 1.0, epsilon = 1e-9);
                assert_eq!(cx.sample_rate_hz, 20.0);
            }
            _ => panic!("expected channel components"),
        }
    }

    #[test]
    fn centripetal_term_matches_omega_squared_r() {
        // Arrange — ω = (0,0,2), r = (3,0,0). ω×(ω×r) = -ω²r = (-12, 0, 0).
        let omega = vec3(0.0, 0.0, 2.0);
        let r = vec3(3.0, 0.0, 0.0);

        // Act
        let inner = cross(&omega, &r, "cross").unwrap();
        let centripetal = cross(&omega, &inner, "cross").unwrap();

        // Assert
        assert_relative_eq!(sx(&centripetal), -12.0, epsilon = 1e-12);
        assert_relative_eq!(sy(&centripetal), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sz(&centripetal), 0.0, epsilon = 1e-12);
    }
}
