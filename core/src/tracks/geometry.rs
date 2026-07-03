//! Pure planar geometry for track matching: closest-point-on-polyline and an
//! axis-aligned bounding box. Scale-agnostic — inputs share one consistent
//! unit (raw degrees × 1e7 for the bbox pre-filter; local metres for the
//! distance projection). Verbatim port of Dart `PolylineGeometry` / `_Bbox`.

use crate::gps::GpsFix;

/// Closest-point projection of `(px, py)` onto polyline `(ref_lat[i], ref_lon[i])`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestPoint {
    pub segment_index: usize,
    pub t: f64,
    pub dist_sq: f64,
}

/// Project `(px, py)` onto the polyline; returns the nearest segment index, the
/// clamped parameter `t ∈ [0, 1]` along it, and the squared distance. A polyline
/// with < 2 points yields `dist_sq = INFINITY`. A degenerate (zero-length)
/// segment treats `t = 0` (collapses to its start point).
pub fn closest_point_on_polyline(px: f64, py: f64, ref_lat: &[f64], ref_lon: &[f64]) -> ClosestPoint {
    debug_assert_eq!(ref_lat.len(), ref_lon.len());
    if ref_lat.len() < 2 {
        return ClosestPoint { segment_index: 0, t: 0.0, dist_sq: f64::INFINITY };
    }
    let mut best = ClosestPoint { segment_index: 0, t: 0.0, dist_sq: f64::INFINITY };
    for k in 0..ref_lat.len() - 1 {
        let (ax, ay) = (ref_lat[k], ref_lon[k]);
        let (bx, by) = (ref_lat[k + 1], ref_lon[k + 1]);
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let t = if len2 == 0.0 { 0.0 } else { (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0) };
        let (cx, cy) = (ax + t * dx, ay + t * dy);
        let (ex, ey) = (px - cx, py - cy);
        let dist2 = ex * ex + ey * ey;
        if dist2 < best.dist_sq {
            best = ClosestPoint { segment_index: k, t, dist_sq: dist2 };
        }
    }
    best
}

/// Axis-aligned lat/lon bounding box in the input scale.
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl Bbox {
    pub fn of(fixes: &[GpsFix]) -> Self {
        let mut b = Bbox {
            min_lat: f64::INFINITY,
            max_lat: f64::NEG_INFINITY,
            min_lon: f64::INFINITY,
            max_lon: f64::NEG_INFINITY,
        };
        for f in fixes {
            if f.lat < b.min_lat {
                b.min_lat = f.lat;
            }
            if f.lat > b.max_lat {
                b.max_lat = f.lat;
            }
            if f.lon < b.min_lon {
                b.min_lon = f.lon;
            }
            if f.lon > b.max_lon {
                b.max_lon = f.lon;
            }
        }
        b
    }

    /// `true` when the boxes share any point (boundary touch counts as overlap).
    pub fn overlaps(&self, other: &Bbox) -> bool {
        self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
            && self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_on_segment_returns_perpendicular_foot() {
        // Arrange — polyline east along x; query point above the 3.0 mark.
        let lat = vec![0.0, 10.0];
        let lon = vec![0.0, 0.0];

        // Act
        let r = closest_point_on_polyline(3.0, 4.0, &lat, &lon);

        // Assert — foot at (3,0), 4 away → dist² = 16; t = 0.3 of the 10-long seg.
        assert_eq!(r.segment_index, 0);
        assert!((r.t - 0.3).abs() < 1e-9);
        assert!((r.dist_sq - 16.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_clamps_beyond_endpoint() {
        // Arrange — query point past the far end of the single segment.
        let lat = vec![0.0, 10.0];
        let lon = vec![0.0, 0.0];

        // Act — x = 20 is beyond b; clamp to t = 1 (point b).
        let r = closest_point_on_polyline(20.0, 0.0, &lat, &lon);

        // Assert
        assert!((r.t - 1.0).abs() < 1e-9);
        assert!((r.dist_sq - 100.0).abs() < 1e-9); // 10 units past b
    }

    #[test]
    fn closest_point_degenerate_segment_collapses_to_start() {
        // Arrange — two identical points (zero-length segment).
        let lat = vec![5.0, 5.0];
        let lon = vec![5.0, 5.0];

        // Act
        let r = closest_point_on_polyline(8.0, 9.0, &lat, &lon);

        // Assert — t = 0, distance from (8,9) to (5,5) → 9 + 16 = 25.
        assert_eq!(r.t, 0.0);
        assert!((r.dist_sq - 25.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_picks_nearest_of_multiple_segments() {
        // Arrange — L-shaped polyline; query nearest the second segment.
        let lat = vec![0.0, 10.0, 10.0];
        let lon = vec![0.0, 0.0, 10.0];

        // Act — point near the vertical second segment at (10,5).
        let r = closest_point_on_polyline(11.0, 5.0, &lat, &lon);

        // Assert — segment 1 (index 1), dist² = 1.
        assert_eq!(r.segment_index, 1);
        assert!((r.dist_sq - 1.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_fewer_than_two_points_is_infinite() {
        // Arrange + Act
        let r = closest_point_on_polyline(0.0, 0.0, &[1.0], &[1.0]);

        // Assert
        assert_eq!(r.dist_sq, f64::INFINITY);
    }

    #[test]
    fn bbox_overlap_and_disjoint() {
        // Arrange
        let a = Bbox::of(&[
            GpsFix { timestamp_ms: 0, lat: 0.0, lon: 0.0 },
            GpsFix { timestamp_ms: 1, lat: 10.0, lon: 10.0 },
        ]);
        let touching = Bbox { min_lat: 10.0, max_lat: 20.0, min_lon: 10.0, max_lon: 20.0 };
        let disjoint = Bbox { min_lat: 11.0, max_lat: 20.0, min_lon: 11.0, max_lon: 20.0 };

        // Act + Assert — boundary touch overlaps; fully separated does not.
        assert!(a.overlaps(&touching));
        assert!(!a.overlaps(&disjoint));
    }
}
