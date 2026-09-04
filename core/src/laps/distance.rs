//! Per-sample lap-distance normalisation onto a canonical polyline (port of
//! `lap_distance_accumulator.dart`), confidence-anchor redistribution.
//!
//! **Unit correction vs idl0 (ruling R17):** `lap_distance_accumulator.dart`
//! (lines 79-80, 88-89) feeds ×1e7-scale latitude straight into
//! `cos(meanLat · π/180)` and multiplies ×1e7 coordinate deltas by the raw
//! `111_320` m/deg constant — its "metres" are ~1e7× too large, so the
//! documented 5 m confidence-anchor residual threshold can never fire and
//! the projection's longitude scale is the cosine of a meaningless angle.
//! This port fixes both: `M_PER_UNIT = 111_320.0 / 1e7` converts a raw
//! (`crate::gps::GpsFix`) ×1e7-scale coordinate delta directly to metres,
//! and the mean-latitude angle is divided by `1e7` before `cos()`. The
//! algorithm's structure (projection, residual, tangent agreement, anchor
//! selection, arc-fraction redistribution) is otherwise a verbatim port.

use crate::gps::GpsFix;

/// Metres per raw GPS-fix coordinate unit (`crate::gps::GpsFix`'s degrees ×
/// 1e7 scale): `111_320.0` m/deg (WGS-84 mean) ÷ `1e7` units/deg. See this
/// module's doc comment — this is the constant idl0's Dart port omits the
/// `/ 1e7` from.
const M_PER_UNIT: f64 = 111_320.0 / 1e7;

/// Maximum perpendicular residual (metres) for a sample to qualify as a
/// confidence anchor.
const ANCHOR_RESIDUAL_METRES: f64 = 5.0;

/// Minimum tangent-agreement cosine (cos(30°) ~= 0.866) for a sample to
/// qualify as a confidence anchor.
const ANCHOR_TANGENT_COS: f64 = 0.866;

/// Minimum speed (km/h) for a sample to qualify as a confidence anchor.
/// Below this, GPS direction is unreliable.
const ANCHOR_MIN_SPEED_KMH: f64 = 5.0;

/// Discriminant for [`LapDistanceError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapDistanceErrorKind {
    /// `speed_kmh.len() != samples.len()`.
    LengthMismatch,
    /// A [`GateCrossing::sample_index`] was out of bounds for `samples`.
    IndexOutOfBounds,
}

/// Error from [`LapDistanceAccumulator::compute`]. Never `Err(String)`
/// (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LapDistanceError {
    pub kind: LapDistanceErrorKind,
    pub message: String,
}
impl std::fmt::Display for LapDistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for LapDistanceError {}

/// One lap's per-sample distance map, normalised to `polyline`.
#[derive(Debug, Clone, PartialEq)]
pub struct LapDistanceAccumulator {
    /// Per-sample arc length on `polyline`, metres, anchor-corrected. Same
    /// length as `samples`. Strictly monotonic in well-formed input.
    pub normalised_distance: Vec<f64>,
    /// Per-sample perpendicular residual from `polyline`, metres.
    pub residual: Vec<f64>,
    /// Per-sample tangent agreement (cosine of angle between the rider's
    /// local direction and the polyline tangent at the projection).
    pub tangent_agreement: Vec<f64>,
}

/// A `(sample_index, known_polyline_distance)` gate crossing. `known_distance`
/// is in metres along `polyline`.
pub struct GateCrossing {
    pub sample_index: usize,
    pub known_distance: f64,
}

impl LapDistanceAccumulator {
    /// Computes the normalised distance map. `samples`/`speed_kmh` are
    /// parallel, chronological; `speed_kmh` is km/h. `start_gate_distance`/
    /// `finish_gate_distance` are metres along `polyline`, defaulting to
    /// `0.0`/`polyline_length` when `None`.
    ///
    /// `samples` and `polyline` are `crate::gps::GpsFix`'s native ×1e7
    /// scale (see this module's doc comment).
    ///
    /// Returns [`LapDistanceErrorKind::LengthMismatch`] when
    /// `speed_kmh.len() != samples.len()`, or
    /// [`LapDistanceErrorKind::IndexOutOfBounds`] when any
    /// `gate_crossings` entry's `sample_index >= samples.len()`. Otherwise
    /// never panics on caller input.
    pub fn compute(
        samples: &[GpsFix],
        polyline: &[GpsFix],
        speed_kmh: &[f64],
        start_gate_distance: Option<f64>,
        finish_gate_distance: Option<f64>,
        gate_crossings: &[GateCrossing],
    ) -> Result<Self, LapDistanceError> {
        let n = samples.len();
        if speed_kmh.len() != n {
            return Err(LapDistanceError {
                kind: LapDistanceErrorKind::LengthMismatch,
                message: format!("speed_kmh has {} entries, samples has {n}", speed_kmh.len()),
            });
        }
        for g in gate_crossings {
            if g.sample_index >= n {
                return Err(LapDistanceError {
                    kind: LapDistanceErrorKind::IndexOutOfBounds,
                    message: format!("gate crossing sample_index {} out of bounds for {n} samples", g.sample_index),
                });
            }
        }

        if n == 0 || polyline.len() < 2 {
            return Ok(LapDistanceAccumulator {
                normalised_distance: vec![0.0; n],
                residual: vec![0.0; n],
                tangent_agreement: vec![0.0; n],
            });
        }

        let mean_lat_e7 = polyline.iter().map(|f| f.lat).sum::<f64>() / polyline.len() as f64;
        let mean_lat_rad = (mean_lat_e7 / 1e7) * std::f64::consts::PI / 180.0;
        let lon_scale = M_PER_UNIT * mean_lat_rad.cos();

        let mut polyline_cum = vec![0.0f64; polyline.len()];
        for k in 1..polyline.len() {
            let dx_lon = (polyline[k].lon - polyline[k - 1].lon) * lon_scale;
            let dy_lat = (polyline[k].lat - polyline[k - 1].lat) * M_PER_UNIT;
            polyline_cum[k] = polyline_cum[k - 1] + (dx_lon * dx_lon + dy_lat * dy_lat).sqrt();
        }
        let polyline_length = *polyline_cum.last().unwrap();

        let mut polyline_distance = vec![0.0f64; n];
        let mut residual = vec![0.0f64; n];
        let mut tangent_agreement = vec![0.0f64; n];
        for i in 0..n {
            let s = &samples[i];
            let mut best_sq = f64::INFINITY;
            let mut best_k = 0usize;
            let mut best_t = 0.0f64;
            for k in 0..polyline.len() - 1 {
                let a = &polyline[k];
                let b = &polyline[k + 1];
                let dx_lon = (b.lon - a.lon) * lon_scale;
                let dy_lat = (b.lat - a.lat) * M_PER_UNIT;
                let len_sq = dx_lon * dx_lon + dy_lat * dy_lat;
                let t = if len_sq == 0.0 {
                    0.0
                } else {
                    let px_lon = (s.lon - a.lon) * lon_scale;
                    let py_lat = (s.lat - a.lat) * M_PER_UNIT;
                    ((px_lon * dx_lon + py_lat * dy_lat) / len_sq).clamp(0.0, 1.0)
                };
                let cx_lon = a.lon * lon_scale + t * dx_lon;
                let cy_lat = a.lat * M_PER_UNIT + t * dy_lat;
                let ex = s.lon * lon_scale - cx_lon;
                let ey = s.lat * M_PER_UNIT - cy_lat;
                let dist_sq = ex * ex + ey * ey;
                if dist_sq < best_sq {
                    best_sq = dist_sq;
                    best_k = k;
                    best_t = t;
                }
            }
            let seg_len = polyline_cum[best_k + 1] - polyline_cum[best_k];
            polyline_distance[i] = polyline_cum[best_k] + best_t * seg_len;
            residual[i] = best_sq.sqrt();

            if i > 0 && i < n - 1 {
                let stx = (samples[i + 1].lon - samples[i - 1].lon) * lon_scale;
                let sty = (samples[i + 1].lat - samples[i - 1].lat) * M_PER_UNIT;
                let st_len = (stx * stx + sty * sty).sqrt();
                let ptx = (polyline[best_k + 1].lon - polyline[best_k].lon) * lon_scale;
                let pty = (polyline[best_k + 1].lat - polyline[best_k].lat) * M_PER_UNIT;
                let pt_len = (ptx * ptx + pty * pty).sqrt();
                if st_len > 1e-6 && pt_len > 1e-6 {
                    tangent_agreement[i] = (stx * ptx + sty * pty) / (st_len * pt_len);
                }
            }
        }

        let mut cumulative_arc = vec![0.0f64; n];
        for i in 1..n {
            let dx_lon = (samples[i].lon - samples[i - 1].lon) * lon_scale;
            let dy_lat = (samples[i].lat - samples[i - 1].lat) * M_PER_UNIT;
            cumulative_arc[i] = cumulative_arc[i - 1] + (dx_lon * dx_lon + dy_lat * dy_lat).sqrt();
        }

        let mut anchors: Vec<(usize, f64)> = vec![(0, start_gate_distance.unwrap_or(0.0))];
        for g in gate_crossings {
            anchors.push((g.sample_index, g.known_distance));
        }
        for i in 0..n {
            if residual[i] < ANCHOR_RESIDUAL_METRES
                && tangent_agreement[i] > ANCHOR_TANGENT_COS
                && speed_kmh[i] > ANCHOR_MIN_SPEED_KMH
            {
                anchors.push((i, polyline_distance[i]));
            }
        }
        anchors.push((n - 1, finish_gate_distance.unwrap_or(polyline_length)));
        anchors.sort_by_key(|&(idx, _)| idx);

        let mut normalised_distance = vec![0.0f64; n];
        for w in anchors.windows(2) {
            let (lo_idx, lo_dist) = w[0];
            let (hi_idx, hi_dist) = w[1];
            normalised_distance[lo_idx] = lo_dist;
            if hi_idx <= lo_idx {
                continue;
            }
            let span = cumulative_arc[hi_idx] - cumulative_arc[lo_idx];
            for k in (lo_idx + 1)..=hi_idx {
                normalised_distance[k] = if span < 1e-6 {
                    lo_dist
                } else {
                    let frac = (cumulative_arc[k] - cumulative_arc[lo_idx]) / span;
                    lo_dist + frac * (hi_dist - lo_dist)
                };
            }
        }

        Ok(LapDistanceAccumulator { normalised_distance, residual, tangent_agreement })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(lat: f64, lon: f64) -> GpsFix {
        GpsFix { timestamp_ms: 0, lat, lon }
    }

    #[test]
    fn empty_samples_or_short_polyline_returns_all_zero() {
        // Arrange / Act
        let a = LapDistanceAccumulator::compute(&[], &[fix(0.0, 0.0), fix(10_000_000.0, 10_000_000.0)], &[], None, None, &[]).unwrap();
        let b = LapDistanceAccumulator::compute(&[fix(0.0, 0.0)], &[fix(0.0, 0.0)], &[10.0], None, None, &[]).unwrap();

        // Assert
        assert!(a.normalised_distance.is_empty());
        assert_eq!(b.normalised_distance, vec![0.0]);
    }

    #[test]
    fn straight_line_polyline_normalised_distance_increases_monotonically() {
        // Arrange -- samples exactly on a straight-line polyline (raw x1e7
        // scale, ~1_000-unit == ~11 m steps), moving fast enough to qualify
        // as confidence anchors throughout.
        let polyline: Vec<GpsFix> = (0..10).map(|i| fix(i as f64 * 1_000.0, 0.0)).collect();
        let samples = polyline.clone();
        let speed = vec![30.0; samples.len()];

        // Act
        let acc = LapDistanceAccumulator::compute(&samples, &polyline, &speed, None, None, &[]).unwrap();

        // Assert
        assert!(acc.normalised_distance.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(acc.normalised_distance.first(), Some(&0.0));
    }

    #[test]
    fn gate_crossing_pins_an_exact_known_distance_at_its_sample_index() {
        // Arrange
        let polyline: Vec<GpsFix> = (0..10).map(|i| fix(i as f64 * 1_000.0, 0.0)).collect();
        let samples = polyline.clone();
        let speed = vec![1.0; samples.len()]; // below anchor threshold -- only the gate crossing anchors

        // Act
        let acc = LapDistanceAccumulator::compute(
            &samples, &polyline, &speed, None, None,
            &[GateCrossing { sample_index: 5, known_distance: 123.45 }],
        ).unwrap();

        // Assert
        assert_eq!(acc.normalised_distance[5], 123.45);
    }

    #[test]
    fn projected_residual_reflects_real_metres_not_e7_scaled_metres() {
        // Arrange -- a due-north polyline at ~50 deg N (raw x1e7 GPS-fix
        // scale); one sample offset exactly 3 m due east of the polyline's
        // midpoint (offset computed with the same M_PER_UNIT/cos(lat)
        // formula the implementation uses). Only a correct x1e7->metres
        // conversion recovers a ~3 m residual -- the idl0-bug-faithful math
        // (feeding raw x1e7 into cos()/* 111_320 without the /1e7) is off
        // by a factor of ~1e7 and would not come remotely close.
        let polyline: Vec<GpsFix> = (0..=10).map(|i| fix(500_000_000.0 + i as f64 * 1_000.0, 100_000_000.0)).collect();
        let mean_lat_e7 = polyline.iter().map(|f| f.lat).sum::<f64>() / polyline.len() as f64;
        let mean_lat_rad = (mean_lat_e7 / 1e7) * std::f64::consts::PI / 180.0;
        let lon_scale = M_PER_UNIT * mean_lat_rad.cos();
        let delta_lon_e7 = 3.0 / lon_scale;
        assert!((delta_lon_e7 - 419.0).abs() < 5.0, "sanity check on the computed offset: {delta_lon_e7}");
        let sample = fix(mean_lat_e7, 100_000_000.0 + delta_lon_e7);
        let speed = vec![30.0];

        // Act
        let acc = LapDistanceAccumulator::compute(&[sample], &polyline, &speed, None, None, &[]).unwrap();

        // Assert
        assert!((acc.residual[0] - 3.0).abs() < 0.1, "residual = {}", acc.residual[0]);
    }

    #[test]
    fn on_line_fast_sample_is_an_anchor_whose_distance_matches_its_arc_length() {
        // Arrange -- samples exactly on a straight due-north polyline,
        // moving fast (30 km/h) so every interior sample qualifies as a
        // confidence anchor (residual ~= 0, tangent_agreement ~= 1, speed
        // above the anchor threshold).
        let polyline: Vec<GpsFix> = (0..=10).map(|i| fix(500_000_000.0 + i as f64 * 1_000.0, 100_000_000.0)).collect();
        let samples = polyline.clone();
        let speed = vec![30.0; samples.len()];

        // Act
        let acc = LapDistanceAccumulator::compute(&samples, &polyline, &speed, None, None, &[]).unwrap();

        // Assert -- sample 5's tangent agrees almost exactly with the
        // polyline's own tangent, and its normalised distance matches its
        // arc length: 5 steps x (1_000 units x 111_320/1e7 m/unit).
        assert!(acc.tangent_agreement[5] > 0.999, "tangent_agreement = {}", acc.tangent_agreement[5]);
        let expected = 5.0 * 1_000.0 * M_PER_UNIT;
        assert!((acc.normalised_distance[5] - expected).abs() < 0.1, "normalised_distance = {}", acc.normalised_distance[5]);
    }

    #[test]
    fn speed_kmh_length_mismatch_is_a_typed_error() {
        // Arrange
        let polyline = vec![fix(0.0, 0.0), fix(1_000.0, 0.0)];
        let samples = vec![fix(0.0, 0.0)];
        let speed = vec![1.0, 2.0]; // wrong length

        // Act
        let err = LapDistanceAccumulator::compute(&samples, &polyline, &speed, None, None, &[]).unwrap_err();

        // Assert
        assert_eq!(err.kind, LapDistanceErrorKind::LengthMismatch);
    }

    #[test]
    fn gate_crossing_index_out_of_bounds_is_a_typed_error() {
        // Arrange
        let polyline = vec![fix(0.0, 0.0), fix(1_000.0, 0.0)];
        let samples = vec![fix(0.0, 0.0), fix(500.0, 0.0)];
        let speed = vec![1.0, 1.0];

        // Act
        let err = LapDistanceAccumulator::compute(
            &samples, &polyline, &speed, None, None,
            &[GateCrossing { sample_index: 5, known_distance: 1.0 }],
        ).unwrap_err();

        // Assert
        assert_eq!(err.kind, LapDistanceErrorKind::IndexOutOfBounds);
    }
}
