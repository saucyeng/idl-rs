//! Gate-crossing geometry: a flat-earth 2D segment intersection — scale-
//! invariant, so it works on the raw degrees × 1e7 coordinates without
//! conversion. Verbatim port of the Dart `LapDetector._findCrossings`.

use crate::gps::GpsFix;
use crate::laps::model::Gate;

/// Interpolated UTC-ms timestamps for every crossing of `gate` along `track`,
/// in chronological order. `u` (parameter along the track segment) is half-open
/// on `(0, 1]` so a crossing landing exactly on a fix is counted once (on the
/// arriving segment), not twice.
pub fn find_crossings(track: &[GpsFix], gate: &Gate) -> Vec<i64> {
    let mut crossings = Vec::new();
    let (px, py, qx, qy) = (gate.lat1, gate.lon1, gate.lat2, gate.lon2);
    let (rx, ry) = (qx - px, qy - py); // gate direction
    for w in track.windows(2) {
        let (prev, curr) = (w[0], w[1]);
        let (ax, ay) = (prev.lat, prev.lon);
        let (sx, sy) = (curr.lat - ax, curr.lon - ay); // track segment direction
        let rxs = rx * sy - ry * sx;
        if rxs.abs() < 1e-15 {
            continue; // parallel / collinear
        }
        let (apx, apy) = (ax - px, ay - py);
        let t = (apx * sy - apy * sx) / rxs; // along gate
        let u = (apx * ry - apy * rx) / rxs; // along track
        if t < 0.0 || t > 1.0 || u <= 0.0 || u > 1.0 {
            continue;
        }
        let dt = (curr.timestamp_ms - prev.timestamp_ms) as f64;
        crossings.push(prev.timestamp_ms + (u * dt).round() as i64);
    }
    crossings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_crossings_counts_a_single_perpendicular_crossing_once() {
        // Arrange — track moves north along lon 0; gate is east-west at lat 0.05.
        let track: Vec<GpsFix> = (0..100)
            .map(|i| GpsFix { timestamp_ms: 1000 + i * 1000, lat: i as f64 * 0.001, lon: 0.0 })
            .collect();
        let gate = Gate { lat1: 0.05, lon1: -0.001, lat2: 0.05, lon2: 0.001 };

        // Act
        let xs = find_crossings(&track, &gate);

        // Assert — exactly one crossing, at the fix where lat == 0.05 (fix 50).
        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0], 1000 + 50 * 1000);
    }

    #[test]
    fn find_crossings_parallel_segment_yields_none() {
        // Arrange — track runs along lat 0.0; gate is also east-west (parallel).
        let track = vec![
            GpsFix { timestamp_ms: 0, lat: 0.0, lon: 0.0 },
            GpsFix { timestamp_ms: 1000, lat: 0.0, lon: 1.0 },
        ];
        let gate = Gate { lat1: 1.0, lon1: -1.0, lat2: 1.0, lon2: 1.0 };

        // Act + Assert
        assert!(find_crossings(&track, &gate).is_empty());
    }
}
