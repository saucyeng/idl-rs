//! Lap-aware builtins: they consume the injected [`MathLapContext`]. Arm
//! bodies moved verbatim from `eval::call_function`'s match.

use crate::math::eval::*;
use crate::math::registry::Call;

/// The `current_lap` builtin.
pub(crate) fn current_lap(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, lap_ctx, .. } = call;
    require_arg_count(name, &args, 0)?;
    let (len, rate) = resolve_time_base(lookup);
    let out: Arc<[f64]> = (0..len)
        .map(|i| lap_ctx.lap_number_at(i as f64 / rate) as f64)
        .collect();
    // Built off resolve_time_base's closed-form (len, rate), never a
    // widened source — there is no real t_us to inherit here, and
    // synthesizing one from i/rate would be exactly the fabricated
    // axis L3-R12 forbids. Empty (no established axis).
    Ok(channel_shared(out, rate, Arc::from(&[] as &[i64])))
}

/// The `lap_start_time` builtin.
pub(crate) fn lap_start_time(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    match &args[0] {
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

/// The `lap_start_distance` builtin.
pub(crate) fn lap_start_distance(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    let distance = lookup.lookup("Distance").ok_or_else(|| {
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
    match &args[0] {
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

/// The `sector_number` builtin.
pub(crate) fn sector_number(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, lap_ctx, .. } = call;
    require_arg_count(name, &args, 0)?;
    let (len, rate) = resolve_time_base(lookup);
    // Same closed-form-only basis as current_lap() — no real t_us to
    // inherit, so both branches are empty (no established axis).
    let sectors = lap_ctx.numbering_sectors();
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

// Each is a scalar in a table row bound to a lap and a `[lap]` value
// in a `math` cell; `lap_shaped` holds that rule.
/// The `lap_number` builtin.
pub(crate) fn lap_number(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    require_arg_count(name, &args, 0)?;
    lap_shaped(name, lap_ctx, |lap| lap.lap_number as f64)
}

/// The `lap_time` builtin.
pub(crate) fn lap_time(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    require_arg_count(name, &args, 0)?;
    lap_shaped(name, lap_ctx, |lap| lap.lap_time_secs)
}

/// The `sector_time` builtin.
pub(crate) fn sector_time(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    let i = require_scalar(&args[0], "sector_time()")?;
    // **0-based**, as C2 §3.3's own row states, matching
    // `sector_number()`'s "0-based sector index at each sample". Lap
    // *numbers* are 1-based and sector indices are not, which reads
    // as an inconsistency until you notice they come from different
    // places: a lap number is recorded in the file, a sector index is
    // a position in a list.
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
    lap_shaped(name, lap_ctx, move |lap| {
        lap.sectors.get(idx).map(|(start, end)| end - start).unwrap_or(f64::NAN)
    })
}

// Both fold: with N overlay laps designated, each contributes its own
// delta series (main vs that one overlay) and the cell's value is the
// elementwise mean across all N (mean_across_overlays, R73) — a rider
// comparing against several ghosts sees the average deviation, not
// just the first ghost's.
/// The `lap_delta_time` builtin.
pub(crate) fn lap_delta_time(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    if lap_ctx.overlay.is_empty() || lap_ctx.main_lap_number.is_none() {
        return Err(err(
            MathEvalErrorKind::NoLapContext,
            "lap_delta_time(): requires a main lap AND at least one overlay lap to be \
             designated. Pick both in the Analyze lap table.",
        ));
    }
    let (main_samples, main_rate, main_t_us, channel_id) =
        require_ref_channel(&args[0], "lap_delta_time")?;
    let window = main_lap_window(lap_ctx);
    let mut series = Vec::with_capacity(lap_ctx.overlay.len());
    for overlay in &lap_ctx.overlay {
        let v = crate::math::variance_geom::eval_variance_time(
            &main_samples,
            main_rate,
            &main_t_us,
            &channel_id,
            lookup,
            overlay,
            window,
        )?;
        match v {
            Value::Channel(c) => series.push(c.samples.to_vec()),
            _ => unreachable!("eval_variance_time always returns a Channel"),
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

/// The `lap_delta_dist` builtin.
pub(crate) fn lap_delta_dist(call: Call<'_>) -> Result<Value, MathEvalError> {
    let Call { name, args, lookup, lap_ctx, .. } = call;
    require_arg_count(name, &args, 1)?;
    if lap_ctx.overlay.is_empty() || lap_ctx.main_lap_number.is_none() {
        return Err(err(
            MathEvalErrorKind::NoLapContext,
            "lap_delta_dist(): requires a main lap AND at least one overlay lap to be \
             designated. Pick both in the Analyze lap table.",
        ));
    }
    let (main_samples, main_rate, main_t_us, channel_id) =
        require_ref_channel(&args[0], "lap_delta_dist")?;
    let window = main_lap_window(lap_ctx);
    let mut series = Vec::with_capacity(lap_ctx.overlay.len());
    for overlay in &lap_ctx.overlay {
        let v = crate::math::variance_geom::eval_variance_dist(
            &main_samples,
            main_rate,
            &main_t_us,
            &channel_id,
            lookup,
            overlay,
            window,
        )?;
        match v {
            Value::Channel(c) => series.push(c.samples.to_vec()),
            _ => unreachable!("eval_variance_dist always returns a Channel"),
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
