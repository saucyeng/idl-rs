//! Lap-aware builtins: they read the injected [`crate::math::MathLapContext`]
//! (and, for the deltas, the overlay laps' handles).

use std::sync::Arc;

use crate::math::eval::{channel, channel_shared, err, lap_shaped, main_lap_window, mean_across_overlays, require_ref_channel, require_scalar, resolve_time_base, type_name, ChannelLookup, MathOverlay};
use crate::math::registry::Ctx;
use crate::math::value::{ChannelValue, Value, ValueAxis};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// `current_lap()`: the 1-based lap number at each sample of the time base.
pub(super) fn current_lap(ctx: &Ctx) -> Result<Value, MathEvalError> {
    let (len, rate) = resolve_time_base(ctx.lookup());
    let out: Arc<[f64]> = (0..len)
        .map(|i| ctx.lap().lap_number_at(i as f64 / rate) as f64)
        .collect();
    // Built off resolve_time_base's closed-form (len, rate), never a
    // widened source — there is no real t_us to inherit here, and
    // synthesizing one from i/rate would be exactly the fabricated
    // axis L3-R12 forbids. Empty (no established axis).
    Ok(channel_shared(out, rate, Arc::from(&[] as &[i64])))
}

/// `lap_start_time(n)`: lap `n`'s start, in seconds; a channel of lap
/// numbers maps sample for sample (NaN where a sample is not a lap number).
pub(super) fn lap_start_time(ctx: &Ctx, n: Value) -> Result<Value, MathEvalError> {
    let lap_ctx = ctx.lap();
    match &n {
        Value::Scalar(n) => Ok(Value::Scalar(lap_ctx.lap_start_secs(n.round() as u32))),
        Value::Channel(c) => {
            let out = c
                .samples
                .iter()
                .map(|&n| {
                    if n.is_nan() || n <= 0.0 || !n.is_finite() {
                        f64::NAN
                    } else {
                        lap_ctx.lap_start_secs(n.round() as u32)
                    }
                })
                .collect();
            // 1:1 map over c's samples — same positions, same t_us.
            Ok(channel(out, c.sample_rate_hz, c.t_us.clone()))
        }
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("lap_start_time(): expected scalar or channel, got {}", type_name(other)),
        )),
    }
}

/// `lap_start_distance(n)`: `[Distance]` at lap `n`'s start, in its unit;
/// a channel of lap numbers maps sample for sample.
pub(super) fn lap_start_distance(ctx: &Ctx, n: Value) -> Result<Value, MathEvalError> {
    let lap_ctx = ctx.lap();
    let distance = ctx.lookup().lookup("Distance").ok_or_else(|| {
        err(
            MathEvalErrorKind::Runtime,
            "lap_start_distance(): no [Distance] channel in this session (GPS_SpeedKmh missing).",
        )
    })?;
    if distance.samples.is_empty() {
        return Err(err(
            MathEvalErrorKind::Runtime,
            "lap_start_distance(): no [Distance] channel in this session (GPS_SpeedKmh missing).",
        ));
    }
    let dist_rate = distance.sample_rate_hz;
    let dist_samples = distance.samples; // owned Vec<f64>, moved in
    let distance_at_lap = |n: i64| -> f64 {
        if n <= 0 {
            return f64::NAN;
        }
        let t_sec = lap_ctx.lap_start_secs(n as u32);
        if t_sec.is_nan() {
            return f64::NAN;
        }
        let idx = (t_sec * dist_rate).round() as i64;
        if idx < 0 || idx as usize >= dist_samples.len() {
            return f64::NAN;
        }
        dist_samples[idx as usize]
    };
    match &n {
        Value::Scalar(n) => Ok(Value::Scalar(distance_at_lap(n.round() as i64))),
        Value::Channel(c) => {
            let out = c
                .samples
                .iter()
                .map(|&n| {
                    if n.is_nan() || n <= 0.0 || !n.is_finite() {
                        f64::NAN
                    } else {
                        distance_at_lap(n.round() as i64)
                    }
                })
                .collect();
            // 1:1 map over c's samples — same positions, same t_us.
            Ok(channel(out, c.sample_rate_hz, c.t_us.clone()))
        }
        other => Err(err(
            MathEvalErrorKind::Type,
            format!("lap_start_distance(): expected scalar or channel, got {}", type_name(other)),
        )),
    }
}

/// `sector_number()`: the 0-based sector index at each sample of the time
/// base, NaN outside every sector.
pub(super) fn sector_number(ctx: &Ctx) -> Result<Value, MathEvalError> {
    let (len, rate) = resolve_time_base(ctx.lookup());
    // Same closed-form-only basis as current_lap() — no real t_us to
    // inherit, so both branches are empty (no established axis).
    let sectors = ctx.lap().numbering_sectors();
    if sectors.is_empty() {
        return Ok(channel(vec![f64::NAN; len], rate, Arc::from(&[] as &[i64])));
    }
    let out = (0..len)
        .map(|i| {
            let idx = crate::variance::sector_number_at(&sectors, i as f64 / rate);
            if idx == u32::MAX {
                f64::NAN
            } else {
                idx as f64
            }
        })
        .collect();
    Ok(channel(out, rate, Arc::from(&[] as &[i64])))
}

// ---- C2 §3.6 lap-shaped builtins (R217 item 3, R233) ----
// Each is a scalar in a table row bound to a lap and a `[lap]` value in a
// `math` cell; `lap_shaped` holds that rule.

/// `lap_number()`: the 1-based lap number.
pub(super) fn lap_number(ctx: &Ctx) -> Result<Value, MathEvalError> {
    lap_shaped("lap_number", ctx.lap(), |lap| lap.lap_number as f64)
}

/// `lap_time()`: the lap's duration, in seconds.
pub(super) fn lap_time(ctx: &Ctx) -> Result<Value, MathEvalError> {
    lap_shaped("lap_time", ctx.lap(), |lap| lap.lap_time_secs)
}

/// `sector_time(i)`: sector `i`'s duration in seconds, `i` **0-based** as
/// C2 §3.3's row states, matching `sector_number()`. Lap *numbers* are
/// 1-based and sector indices are not: a lap number is recorded in the
/// file, a sector index is a position in a list.
pub(super) fn sector_time(ctx: &Ctx, i: Value) -> Result<Value, MathEvalError> {
    let i = require_scalar(&i, "sector_time()")?;
    if !i.is_finite() || i < 0.0 {
        return Err(err(
            MathEvalErrorKind::Runtime,
            format!("sector_time(): sector index is 0-based and cannot be negative; got {i}"),
        ));
    }
    let idx = i.round() as usize;
    // A lap with fewer sectors than asked for yields NaN, not an
    // error: a lap cut short has genuinely no third sector, and one
    // such lap must not blank the whole definition (C2 §3.5.B).
    lap_shaped("sector_time", ctx.lap(), move |lap| {
        lap.sectors.get(idx).map(|(start, end)| end - start).unwrap_or(f64::NAN)
    })
}

/// One overlay's variance series against the main lap: `eval_variance_time`
/// or `eval_variance_dist`.
type VarianceFn = fn(&[f64], f64, &[i64], &str, &dyn ChannelLookup, &MathOverlay, Option<(f64, f64)>) -> Result<Value, MathEvalError>;

/// The shared body of `lap_delta_time`/`lap_delta_dist` (B2): with N overlay
/// laps designated, each contributes its own delta series (main vs that one
/// overlay) and the value is the elementwise mean across all N
/// (`mean_across_overlays`, R73) — a rider comparing against several ghosts
/// sees the average deviation, not just the first ghost's.
fn lap_delta(ctx: &Ctx, x: &Value, variance: VarianceFn) -> Result<Value, MathEvalError> {
    let (name, lap_ctx) = (ctx.name(), ctx.lap());
    if lap_ctx.overlay.is_empty() || lap_ctx.main_lap_number.is_none() {
        return Err(err(
            MathEvalErrorKind::NoLapContext,
            format!(
                "{name}(): requires a main lap AND at least one overlay lap to be \
                 designated. Pick both in the Analyze lap table."
            ),
        ));
    }
    let (main_samples, main_rate, main_t_us, channel_id) = require_ref_channel(x, name)?;
    let window = main_lap_window(lap_ctx);
    let mut series = Vec::with_capacity(lap_ctx.overlay.len());
    for overlay in &lap_ctx.overlay {
        match variance(&main_samples, main_rate, &main_t_us, &channel_id, ctx.lookup(), overlay, window)? {
            Value::Channel(c) => series.push(c.samples.to_vec()),
            _ => unreachable!("{name}'s variance always returns a Channel"),
        }
    }
    Ok(Value::Channel(ChannelValue {
        samples: Arc::from(mean_across_overlays(&series)),
        sample_rate_hz: main_rate,
        channel_id: None,
        t_us: main_t_us,
        axis: ValueAxis::Time,
    }))
}

/// `lap_delta_time([ch])`: main minus overlay at equal elapsed time, in the
/// channel's unit.
pub(super) fn lap_delta_time(ctx: &Ctx, x: Value) -> Result<Value, MathEvalError> {
    lap_delta(ctx, &x, crate::math::variance_geom::eval_variance_time)
}

/// `lap_delta_dist([ch])`: main minus overlay at equal distance, in the
/// channel's unit.
pub(super) fn lap_delta_dist(ctx: &Ctx, x: Value) -> Result<Value, MathEvalError> {
    lap_delta(ctx, &x, crate::math::variance_geom::eval_variance_dist)
}
