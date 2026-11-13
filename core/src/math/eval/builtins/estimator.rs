//! Estimator-backed virtual sensors (`wheel_travel`, `wheel_velocity`,
//! `attitude`, `body_accel`). Arm body moved verbatim from
//! `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

// The offline geometry-constrained estimator (`estimate::run`) is run
// once per session by the lookup and cached; these are store reads.
/// The `wheel_travel` / `wheel_velocity` / `attitude` / `body_accel` builtin: one evaluator for every spelling, which reads `call.name` for its messages.
pub(crate) fn virtual_sensor(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, .. } = call;
    require_arg_count(name, &args, 1)?;
    let arg = require_string(&args[0], name)?;
    let channel_id = estimator_channel_id(name, arg)?;
    let ch = lookup.estimator_channel(channel_id).ok_or_else(|| {
        err(
            MathEvalErrorKind::Runtime,
            format!(
                "{name}(\"{arg}\"): the suspension/attitude estimator could not run for \
                 this session — it needs IMU0 accel and gyro channels"
            ),
        )
    })?;
    Ok(channel(ch.samples.to_vec(), ch.sample_rate_hz, ch.t_us))
}
