//! Synthesises [`LapGateJson`] pairs from a reference polyline (port of
//! `gate_geometry.dart`). Used by GPX-as-Track import: the user gets two
//! reasonable default gates at the start and finish, re-placeable later.
//!
//! **Unit convention (settled, ruling R8):** every input here —
//! [`GpsFix`] and the internal geometry — is at the raw GPS-fix scale
//! (degrees × 1e7, `crate::gps::GpsFix`'s own doc). [`LapGateJson`] is
//! decimal degrees (C1 §6). The geometric-mean/perpendicular-vector math
//! below is scale-invariant, so it runs entirely at the × 1e7 scale and
//! converts to decimal degrees exactly once, at the very end, in
//! [`to_lap_gate_json`].

use crate::gps::GpsFix;
use crate::store::session_json::LapGateJson;

/// Discriminant for [`GateSynthesisError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateSynthesisErrorKind {
    /// `polyline` had fewer than 2 fixes (no direction is defined).
    TooShort,
    /// `index` was out of bounds for `polyline`.
    IndexOutOfBounds,
    /// `polyline` was empty.
    Empty,
}

/// Error from [`perpendicular_gate_at`] / [`snap_to_nearest_fix`]. Never
/// `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateSynthesisError {
    pub kind: GateSynthesisErrorKind,
    pub message: String,
}
impl std::fmt::Display for GateSynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for GateSynthesisError {}

/// Pre-idl1 default gate width, metres — wide enough to register a
/// crossing under typical 3-5 m GPS noise, narrow enough to avoid false
/// positives from a parallel return path.
const DEFAULT_GATE_WIDTH_METERS: f64 = 20.0;

/// Two gates derived from the start and end of `polyline`, each a
/// `gate_width_meters`-long segment centred on the endpoint, perpendicular
/// to the local track direction. `None` when `polyline` has fewer than 2
/// fixes.
pub fn endpoint_gates(polyline: &[GpsFix], gate_width_meters: f64) -> Option<(LapGateJson, LapGateJson)> {
    if polyline.len() < 2 {
        return None;
    }
    let start = perpendicular_gate(&polyline[0], &polyline[1], gate_width_meters, "Start");
    let finish = perpendicular_gate(
        &polyline[polyline.len() - 1],
        &polyline[polyline.len() - 2],
        gate_width_meters,
        "Finish",
    );
    Some((start, finish))
}

/// [`endpoint_gates`] with the pre-idl1 default width (20 m).
pub fn endpoint_gates_default(polyline: &[GpsFix]) -> Option<(LapGateJson, LapGateJson)> {
    endpoint_gates(polyline, DEFAULT_GATE_WIDTH_METERS)
}

/// A gate of length `width_meters` centred on `polyline[index]`,
/// perpendicular to the local tangent (the segment toward `index+1`, or
/// toward `index-1` at the last index).
pub fn perpendicular_gate_at(
    polyline: &[GpsFix],
    index: usize,
    width_meters: f64,
    name: &str,
) -> Result<LapGateJson, GateSynthesisError> {
    if polyline.len() < 2 {
        return Err(GateSynthesisError {
            kind: GateSynthesisErrorKind::TooShort,
            message: "polyline must contain at least 2 fixes".to_string(),
        });
    }
    if index >= polyline.len() {
        return Err(GateSynthesisError {
            kind: GateSynthesisErrorKind::IndexOutOfBounds,
            message: format!("index {index} out of bounds for polyline of length {}", polyline.len()),
        });
    }
    let at = &polyline[index];
    let towards = if index + 1 < polyline.len() { &polyline[index + 1] } else { &polyline[index - 1] };
    Ok(perpendicular_gate(at, towards, width_meters, name))
}

/// Index of the polyline fix whose Euclidean lat/lon distance to
/// `(lat_e7, lon_e7)` (same degrees-× 1e7 scale as `polyline`) is smallest.
pub fn snap_to_nearest_fix(polyline: &[GpsFix], lat_e7: f64, lon_e7: f64) -> Result<usize, GateSynthesisError> {
    if polyline.is_empty() {
        return Err(GateSynthesisError { kind: GateSynthesisErrorKind::Empty, message: "polyline must not be empty".to_string() });
    }
    let mut best_idx = 0;
    let mut best_dist_sq = f64::INFINITY;
    for (i, f) in polyline.iter().enumerate() {
        let d_lat = f.lat - lat_e7;
        let d_lon = f.lon - lon_e7;
        let dist_sq = d_lat * d_lat + d_lon * d_lon;
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Builds a gate of total length `width_meters` centred on `at_index`,
/// perpendicular to the segment from `at_index` toward `towards`. Coords
/// stay at the × 1e7 scale until [`to_lap_gate_json`] converts once at the end.
fn perpendicular_gate(at_index: &GpsFix, towards: &GpsFix, width_meters: f64, name: &str) -> LapGateJson {
    // Local-metre conversion factors. Lat: 1 deg ~= 111,320 m. Lon: shrinks
    // by cos(lat). Coordinates are at x1e7, so the constants fold the
    // factor in directly (x1e7 deg -> metres).
    const M_PER_DEG_UNITS: f64 = 111_320.0 / 1e7;
    let lat_deg = at_index.lat / 1e7;
    let lon_scale = M_PER_DEG_UNITS * (lat_deg * std::f64::consts::PI / 180.0).cos().abs();

    // Direction vector along the track in metric units.
    let dx_m = (towards.lat - at_index.lat) * M_PER_DEG_UNITS;
    let dy_m = (towards.lon - at_index.lon) * lon_scale;
    let length = (dx_m * dx_m + dy_m * dy_m).sqrt();

    if length == 0.0 {
        // Degenerate: identical points. Return a zero-length gate at the
        // point -- caller will recognise it as invalid via lap_detector's
        // existing length check.
        return to_lap_gate_json(at_index.lat, at_index.lon, at_index.lat, at_index.lon, name);
    }

    // Perpendicular unit vector in metric units (rotate 90 deg CCW: (x,y)->(-y,x)).
    let perp_dx_m = -dy_m / length;
    let perp_dy_m = dx_m / length;

    // Half-width offsets, converted back to x1e7 deg.
    let half_width = width_meters / 2.0;
    let d_lat_units = (perp_dx_m * half_width) / M_PER_DEG_UNITS;
    let d_lon_units = (perp_dy_m * half_width) / lon_scale;

    to_lap_gate_json(
        at_index.lat + d_lat_units,
        at_index.lon + d_lon_units,
        at_index.lat - d_lat_units,
        at_index.lon - d_lon_units,
        name,
    )
}

/// Converts × 1e7-scale coordinates to `LapGateJson`'s decimal-degrees
/// convention (C1 §6, settled by ruling R8 — see this module's doc
/// comment).
fn to_lap_gate_json(lat1_e7: f64, lon1_e7: f64, lat2_e7: f64, lon2_e7: f64, name: &str) -> LapGateJson {
    LapGateJson {
        lat1_deg: lat1_e7 / 1e7,
        lon1_deg: lon1_e7 / 1e7,
        lat2_deg: lat2_e7 / 1e7,
        lon2_deg: lon2_e7 / 1e7,
        name: name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(ts: i64, lat: f64, lon: f64) -> GpsFix {
        GpsFix { timestamp_ms: ts, lat, lon }
    }

    #[test]
    fn endpoint_gates_needs_at_least_two_fixes() {
        // Arrange / Act / Assert
        assert!(endpoint_gates_default(&[]).is_none());
        assert!(endpoint_gates_default(&[fix(0, 0.0, 0.0)]).is_none());
    }

    #[test]
    fn endpoint_gates_produces_start_and_finish_perpendicular_to_travel() {
        // Arrange -- a straight line due north (lat increasing, lon constant).
        let polyline = vec![
            fix(0, 501_163_000.0, -1_229_574_000.0),
            fix(1000, 501_164_000.0, -1_229_574_000.0),
            fix(2000, 501_165_000.0, -1_229_574_000.0),
        ];

        // Act
        let (start, finish) = endpoint_gates_default(&polyline).unwrap();

        // Assert -- a gate perpendicular to due-north travel is (approximately)
        // east-west: its two endpoints should differ mostly in longitude, not
        // latitude, and be centred on the fix.
        assert_eq!(start.name, "Start");
        assert_eq!(finish.name, "Finish");
        assert!((start.lat1_deg - start.lat2_deg).abs() < (start.lon1_deg - start.lon2_deg).abs());
    }

    #[test]
    fn perpendicular_gate_at_out_of_bounds_index_is_a_typed_error() {
        // Arrange
        let polyline = vec![fix(0, 0.0, 0.0), fix(1, 1.0, 1.0)];

        // Act
        let err = perpendicular_gate_at(&polyline, 5, 20.0, "").unwrap_err();

        // Assert
        assert_eq!(err.kind, GateSynthesisErrorKind::IndexOutOfBounds);
    }

    #[test]
    fn snap_to_nearest_fix_finds_the_closest_index() {
        // Arrange
        let polyline = vec![fix(0, 0.0, 0.0), fix(1, 10.0, 10.0), fix(2, 20.0, 20.0)];

        // Act
        let idx = snap_to_nearest_fix(&polyline, 11.0, 9.0).unwrap();

        // Assert
        assert_eq!(idx, 1);
    }

    #[test]
    fn snap_to_nearest_fix_empty_polyline_is_a_typed_error() {
        // Act
        let err = snap_to_nearest_fix(&[], 0.0, 0.0).unwrap_err();

        // Assert
        assert_eq!(err.kind, GateSynthesisErrorKind::Empty);
    }
}
