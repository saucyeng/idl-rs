//! GPS tangent-plane geometry for `variance_time`/`variance_dist`.
//!
//! Ported verbatim from the Dart evaluator's variance helpers
//! (`app/lib/data/math_channel_evaluator.dart` `_toDeg`,
//! `_buildOverlayReference`, `_buildMainPositions`, `_cumulativeArc`, and the
//! `_callVarianceTimeRust`/`_callVarianceDistRust` assembly). Reads GPS channels
//! through `ChannelLookup` — main from the primary handle, overlay from the
//! second handle (spec §6). Pure geometry; data in, data out.

use crate::math::eval::{ChannelLookup, LookupChannel, MathOverlay};
use crate::math::value::{ChannelValue, Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// GPS_Latitude/Longitude are stored as degrees×1e7 (firmware i32) on binary
/// imports; GPX uses raw degrees. Magnitude check is robust to both (real
/// lat/lon never exceed 180 in degree form). Mirrors Dart `_toDeg`.
pub fn to_deg(x: f64) -> f64 {
    if x.abs() > 1000.0 {
        x / 1e7
    } else {
        x
    }
}

/// Cumulative arc length (metres) along a polyline given by parallel E/N
/// arrays. `out[0] = 0.0`; length matches `min(e.len(), n.len())`. Mirrors
/// Dart `_cumulativeArc`.
pub fn cumulative_arc(e: &[f64], n: &[f64]) -> Vec<f64> {
    let len = e.len().min(n.len());
    let mut out = vec![0.0; len];
    let mut s = 0.0;
    for i in 1..len {
        let de = e[i] - e[i - 1];
        let dn = n[i] - n[i - 1];
        s += (de * de + dn * dn).sqrt();
        out[i] = s;
    }
    out
}

/// Overlay-lap reference polyline + shared tangent-plane parameters.
/// `e`/`n` are metres-from-origin; `t_lap` is overlay-lap-relative seconds;
/// `lap_start_sec` is the overlay lap start in uniform-time (passed in,
/// pre-converted Dart-side). `lat0`/`lon0` (degrees) and `lat_scale`/`lon_scale`
/// (metres/degree at the lap's mean latitude) define the frame BOTH sides
/// project into. Mirrors Dart `_buildOverlayReference`.
pub struct OverlayReference {
    pub e: Vec<f64>,
    pub n: Vec<f64>,
    pub t_lap: Vec<f64>,
    pub lap_start_sec: f64,
    pub lat0: f64,
    pub lon0: f64,
    pub lat_scale: f64,
    pub lon_scale: f64,
}

/// Builds the overlay reference from the overlay handle's GPS channels.
///
/// `overlay`: lookup over the overlay session (GPS_Latitude/Longitude/EpochMs).
/// `lap_start_ms`/`lap_end_ms`: overlay lap window in raw epoch ms (selects GPS
///   samples). `overlay_lap_start_uniform_sec`: the overlay lap start in
///   uniform-time, pre-converted Dart-side.
/// Returns `None` on missing GPS or a window with < 2 samples (Dart parity).
pub fn build_overlay_reference(
    overlay: &dyn ChannelLookup,
    lap_start_ms: f64,
    lap_end_ms: f64,
    overlay_lap_start_uniform_sec: f64,
) -> Option<OverlayReference> {
    let lat = overlay.lookup("GPS_Latitude")?;
    let lon = overlay.lookup("GPS_Longitude")?;
    let epoch = overlay.lookup("GPS_EpochMs")?;
    let len = lat.samples.len().min(lon.samples.len()).min(epoch.samples.len());
    if len < 2 {
        return None;
    }

    let indices: Vec<usize> = (0..len)
        .filter(|&i| epoch.samples[i] >= lap_start_ms && epoch.samples[i] <= lap_end_ms)
        .collect();
    if indices.len() < 2 {
        return None;
    }

    // Mean latitude over the overlay lap drives the lon scale; the lap's first
    // GPS sample is the origin. Both choices are re-used for main_positions.
    let mut sum_lat = 0.0;
    for &i in &indices {
        sum_lat += to_deg(lat.samples[i]);
    }
    let mean_lat_deg = sum_lat / indices.len() as f64;
    let mean_lat_rad = mean_lat_deg * std::f64::consts::PI / 180.0;
    let lat_scale = 111320.0;
    let lon_scale = 111320.0 * mean_lat_rad.cos();

    let lat0 = to_deg(lat.samples[indices[0]]);
    let lon0 = to_deg(lon.samples[indices[0]]);

    let rate = epoch.sample_rate_hz;
    let mut e = Vec::with_capacity(indices.len());
    let mut n = Vec::with_capacity(indices.len());
    let mut t_lap = Vec::with_capacity(indices.len());
    for &i in &indices {
        e.push((to_deg(lon.samples[i]) - lon0) * lon_scale);
        n.push((to_deg(lat.samples[i]) - lat0) * lat_scale);
        t_lap.push(i as f64 / rate - overlay_lap_start_uniform_sec);
    }

    Some(OverlayReference {
        e,
        n,
        t_lap,
        lap_start_sec: overlay_lap_start_uniform_sec,
        lat0,
        lon0,
        lat_scale,
        lon_scale,
    })
}

/// Main-session positions (E, N, heading_rad) in the overlay's tangent plane.
/// Heading from GPS_Heading when present (compass→math radians:
/// `(90 - compassDeg) * PI/180`), else finite-differenced from E/N. Returns
/// `None` when GPS_Latitude/Longitude are missing or < 2 samples. Mirrors Dart
/// `_buildMainPositions`.
pub struct MainPositions {
    pub e: Vec<f64>,
    pub n: Vec<f64>,
    pub heading: Vec<f64>,
}

pub fn build_main_positions(
    main: &dyn ChannelLookup,
    lat0: f64,
    lon0: f64,
    lat_scale: f64,
    lon_scale: f64,
) -> Option<MainPositions> {
    let lat = main.lookup("GPS_Latitude")?;
    let lon = main.lookup("GPS_Longitude")?;
    let len = lat.samples.len().min(lon.samples.len());
    if len < 2 {
        return None;
    }

    let mut e = vec![0.0; len];
    let mut n = vec![0.0; len];
    for i in 0..len {
        e[i] = (to_deg(lon.samples[i]) - lon0) * lon_scale;
        n[i] = (to_deg(lat.samples[i]) - lat0) * lat_scale;
    }

    let mut heading = vec![0.0; len];
    if let Some(h) = main.lookup("GPS_Heading") {
        if h.samples.len() >= len {
            for i in 0..len {
                // Compass degrees (0=N, clockwise) → math radians (0=E, CCW).
                let compass_deg = h.samples[i];
                heading[i] = (90.0 - compass_deg) * std::f64::consts::PI / 180.0;
            }
            return Some(MainPositions { e, n, heading });
        }
    }
    // Finite-difference fallback when no usable GPS_Heading.
    for i in 0..len {
        let j = if i + 1 < len { i + 1 } else { i - 1 };
        let de = e[j] - e[i];
        let dn = n[j] - n[i];
        heading[i] = dn.atan2(de);
    }
    Some(MainPositions { e, n, heading })
}

fn runtime_err(msg: impl Into<String>) -> MathEvalError {
    MathEvalError::new(MathEvalErrorKind::Runtime, msg)
}

/// Assembles `variance_time` inputs and delegates to
/// [`crate::variance::variance_time`]. Mirrors `_callVarianceTimeRust`.
pub fn eval_variance_time(
    main_samples: &[f64],
    main_rate: f64,
    channel_id: &str,
    main: &dyn ChannelLookup,
    overlay: &MathOverlay,
    main_window: (f64, f64),
) -> Result<Value, MathEvalError> {
    let overlay_ch = overlay.lookup.lookup(channel_id).ok_or_else(|| {
        runtime_err(format!(
            "variance_time(): channel \"{channel_id}\" not present in overlay session."
        ))
    })?;
    let r = build_overlay_reference(
        overlay.lookup.as_ref(),
        overlay.lap_start_ms,
        overlay.lap_end_ms,
        overlay.lap_start_uniform_sec,
    )
    .ok_or_else(|| {
        runtime_err(
            "variance_time(): overlay session lap could not be resolved (missing GPS or empty lap window).",
        )
    })?;
    let main_pos = build_main_positions(main, r.lat0, r.lon0, r.lat_scale, r.lon_scale)
        .ok_or_else(|| {
            runtime_err("variance_time(): main session is missing GPS_Latitude / GPS_Longitude.")
        })?;

    let result = variance_time_against(&r, &overlay_ch, &main_pos, main_samples, main_rate, main_window);
    Ok(Value::Channel(ChannelValue { samples: std::sync::Arc::from(result), sample_rate_hz: main_rate, channel_id: None }))
}

/// Computes one target's `variance_time` series against a PREBUILT reference and
/// prebuilt target positions — so a batch ([`crate::variance::variance_traces`])
/// can build the reference once and reuse it across targets. Same numbers as the
/// inline assembly in [`eval_variance_time`].
pub fn variance_time_against(
    reference: &OverlayReference,
    overlay_ch: &LookupChannel,
    main_pos: &MainPositions,
    main_samples: &[f64],
    main_rate: f64,
    main_window: (f64, f64),
) -> Vec<f64> {
    let positions = align_main_positions(main_samples.len(), main_pos);
    let refpts: Vec<(f64, f64, f64)> = reference
        .e
        .iter()
        .zip(&reference.n)
        .zip(&reference.t_lap)
        .map(|((e, n), t)| (*e, *n, *t))
        .collect();
    crate::variance::variance_time(
        main_samples,
        main_rate,
        main_window.0,
        main_window.1,
        &positions,
        refpts,
        &overlay_ch.samples,
        overlay_ch.sample_rate_hz,
        reference.lap_start_sec,
    )
}

/// Assembles `variance_dist` inputs and delegates to
/// [`crate::variance::variance_dist`]. Mirrors `_callVarianceDistRust`.
pub fn eval_variance_dist(
    main_samples: &[f64],
    main_rate: f64,
    channel_id: &str,
    main: &dyn ChannelLookup,
    overlay: &MathOverlay,
    main_window: (f64, f64),
) -> Result<Value, MathEvalError> {
    let overlay_ch = overlay.lookup.lookup(channel_id).ok_or_else(|| {
        runtime_err(format!(
            "variance_dist(): channel \"{channel_id}\" not present in overlay session."
        ))
    })?;
    let r = build_overlay_reference(
        overlay.lookup.as_ref(),
        overlay.lap_start_ms,
        overlay.lap_end_ms,
        overlay.lap_start_uniform_sec,
    )
    .ok_or_else(|| runtime_err("variance_dist(): overlay session lap could not be resolved."))?;
    let main_pos = build_main_positions(main, r.lat0, r.lon0, r.lat_scale, r.lon_scale)
        .ok_or_else(|| {
            runtime_err("variance_dist(): main session is missing GPS_Latitude / GPS_Longitude.")
        })?;

    let overlay_arc = cumulative_arc(&r.e, &r.n);
    let overlay_samples = subsample_to_arc(&overlay_ch.samples, &overlay_arc);
    let result =
        variance_dist_against(&overlay_arc, &overlay_samples, &main_pos, main_samples, main_rate, main_window);
    Ok(Value::Channel(ChannelValue { samples: std::sync::Arc::from(result), sample_rate_hz: main_rate, channel_id: None }))
}

/// Computes one target's `variance_dist` series against a PREBUILT reference
/// arc-length + subsampled-channel pair (built once by the caller) and prebuilt
/// target positions. Same numbers as the inline assembly in [`eval_variance_dist`].
pub fn variance_dist_against(
    overlay_arc: &[f64],
    overlay_samples: &[f64],
    main_pos: &MainPositions,
    main_samples: &[f64],
    main_rate: f64,
    main_window: (f64, f64),
) -> Vec<f64> {
    let main_arc = cumulative_arc(&main_pos.e, &main_pos.n);
    // Pad/sample main arc to main_samples length (channel may outrate GPS).
    let mut main_arc_len = vec![0.0; main_samples.len()];
    if !main_arc.is_empty() {
        let ratio = main_samples.len() as f64 / main_arc.len() as f64;
        for i in 0..main_samples.len() {
            let j = ((i as f64 / ratio).floor() as usize).min(main_arc.len() - 1);
            main_arc_len[i] = main_arc[j];
        }
    }
    crate::variance::variance_dist(
        main_samples,
        main_rate,
        main_window.0,
        main_window.1,
        &main_arc_len,
        overlay_samples,
        overlay_arc,
    )
}

/// Subsamples an overlay channel to align with its arc-length array (the loop
/// previously inlined in [`eval_variance_dist`]). Reused by the batch so the
/// reference channel is subsampled once.
pub fn subsample_to_arc(overlay_ch: &[f64], overlay_arc: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    if !overlay_ch.is_empty() && !overlay_arc.is_empty() {
        let ratio = overlay_ch.len() as f64 / overlay_arc.len() as f64;
        for i in 0..overlay_arc.len() {
            let j = ((i as f64 * ratio).floor() as usize).min(overlay_ch.len() - 1);
            out.push(overlay_ch[j]);
        }
    }
    out
}

// Aligns the (E, N, heading) main positions (one per GPS sample) to the length
// of the channel under comparison by subsampling at the rate ratio. Mirrors the
// Dart ratio loop in `_callVarianceTimeRust`.
fn align_main_positions(channel_len: usize, main_pos: &MainPositions) -> Vec<(f64, f64, f64)> {
    let pos_len = main_pos.e.len();
    let ratio_f =
        if channel_len == 0 || pos_len == 0 { 1.0 } else { channel_len as f64 / pos_len as f64 };
    let divisor = (ratio_f as i64).clamp(1, channel_len.max(1) as i64) as usize;
    let mut out = Vec::with_capacity(channel_len);
    for i in 0..channel_len {
        let j = (i / divisor).min(pos_len.saturating_sub(1));
        out.push((main_pos.e[j], main_pos.n[j], main_pos.heading[j]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_deg_passes_through_degree_values_and_scales_e7() {
        // Arrange / Act / Assert — firmware degrees×1e7 vs raw degrees.
        assert!((to_deg(-37.81) - -37.81).abs() < 1e-9);
        assert!((to_deg(-378_100_000.0) - -37.81).abs() < 1e-6);
    }

    #[test]
    fn cumulative_arc_sums_segment_lengths() {
        // Arrange — 3-4-5 triangle legs: (0,0)->(3,0)->(3,4).
        let e = vec![0.0, 3.0, 3.0];
        let n = vec![0.0, 0.0, 4.0];

        // Act
        let arc = cumulative_arc(&e, &n);

        // Assert — 0, 3, 7.
        assert_eq!(arc, vec![0.0, 3.0, 7.0]);
    }
}
