//! Vector and rotation primitives over [`crate::math::vector`] (SPEC §19).
//! Arm bodies moved verbatim from `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

/// The `vec` builtin.
pub(crate) fn vec(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 3)?;
    crate::math::vector::make_vec3(&args[0], &args[1], &args[2], name)
}

/// The `vx` builtin.
pub(crate) fn vx(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    crate::math::vector::component(&args[0], 0, name)
}

/// The `vy` builtin.
pub(crate) fn vy(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    crate::math::vector::component(&args[0], 1, name)
}

/// The `vz` builtin.
pub(crate) fn vz(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    crate::math::vector::component(&args[0], 2, name)
}

/// The `vadd` builtin.
pub(crate) fn vadd(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::vadd(&args[0], &args[1], name)
}

/// The `vsub` builtin.
pub(crate) fn vsub(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::vsub(&args[0], &args[1], name)
}

/// The `vscale` builtin.
pub(crate) fn vscale(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::vscale(&args[0], &args[1], name)
}

/// The `cross` builtin.
pub(crate) fn cross(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::cross(&args[0], &args[1], name)
}

/// The `dot` builtin.
pub(crate) fn dot(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::dot(&args[0], &args[1], name)
}

/// The `norm` builtin.
pub(crate) fn norm(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    crate::math::vector::norm(&args[0], name)
}

/// The `normalize` builtin.
pub(crate) fn normalize(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 1)?;
    crate::math::vector::normalize(&args[0], name)
}

// Retired as `angle` (R143/R151 item 4): `numpy.angle` is complex
// phase, not the angle between two vectors — a false friend.
// `vector::angle`'s own Rust fn name is unchanged (internal, R151's
// "internal fn names may stay" precedent from `variance_geom.rs`).
/// The `angle_between` builtin.
pub(crate) fn angle_between(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    require_arg_count(name, &args, 2)?;
    crate::math::vector::angle(&args[0], &args[1], name)
}

/// The `rotate_mat` builtin.
pub(crate) fn rotate_mat(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    // v + 9 row-major matrix entries (constant scalars).
    require_arg_count(name, &args, 10)?;
    let mut m = [0.0_f64; 9];
    for (slot, arg) in m.iter_mut().zip(&args[1..]) {
        *slot = require_scalar(arg, name)?;
    }
    crate::math::vector::rotate_mat(&args[0], &m, name)
}

/// The `rotate_axis` builtin.
pub(crate) fn rotate_axis(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    // v, ax, ay, az, angle — constant scalars.
    require_arg_count(name, &args, 5)?;
    let ax = require_scalar(&args[1], name)?;
    let ay = require_scalar(&args[2], name)?;
    let az = require_scalar(&args[3], name)?;
    let ang = require_scalar(&args[4], name)?;
    crate::math::vector::rotate_axis(&args[0], ax, ay, az, ang, name)
}

/// The `rotate_euler` builtin.
pub(crate) fn rotate_euler(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, .. } = call;
    // v, roll, pitch, yaw — angle args may be channels (per-sample).
    require_arg_count(name, &args, 4)?;
    crate::math::vector::rotate_euler(&args[0], &args[1], &args[2], &args[3], name)
}
