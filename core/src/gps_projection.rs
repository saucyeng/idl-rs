//! Projection of GPS fixes into a local ENU (east/north) frame, and
//! perpendicular-distance decimation of the resulting path (C2 §5.3's map
//! cell, C3 §3.5's `fetch_gps_trace_v2`, ruling R217 item 1).
//!
//! **Why this is `core` and not the renderer.** A map cell draws metres, not
//! degrees: equal aspect is a property of the projection, and the comparison
//! between two laps' lines through a corner is a number the sync model
//! depends on. Projecting in JavaScript would put that number in the picture
//! layer and duplicate [`crate::track_projection`] (CLAUDE.md §2). The app
//! never sees a latitude.
//!
//! **Equirectangular, not UTM or Web Mercator.** A session covers a track, a
//! kilometre or two across; over that span an equirectangular projection about
//! the session's own mean latitude is accurate to well under a metre, and it
//! is the same approximation [`crate::laps::distance`] already normalises lap
//! distance with. A conformal global projection would buy nothing here and
//! would make two nearby sessions' frames disagree.
//!
//! **Decimation is Douglas–Peucker, not stride** — see [`decimate_path`].

use crate::gps::GpsFix;

/// Metres per degree of latitude, and per degree of longitude at the equator.
/// The same constant [`crate::laps::distance`] uses, for the same reason: one
/// spherical-earth scale, so a distance computed there and a position
/// projected here agree.
const M_PER_DEGREE: f64 = 111_320.0;

/// A local east/north frame anchored at one origin, in metres.
///
/// Construct it once per request and project every point of the trace *and*
/// every point of the track underlay through the same value — two frames in
/// one picture would draw the trace beside the track instead of on it
/// (C2 §5.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnuFrame {
    /// Origin latitude, decimal degrees.
    pub origin_lat: f64,
    /// Origin longitude, decimal degrees.
    pub origin_lon: f64,
    /// Metres per degree of longitude at this origin's latitude.
    lon_scale: f64,
}

impl EnuFrame {
    /// A frame anchored at `(origin_lat, origin_lon)`, decimal degrees.
    pub fn new(origin_lat: f64, origin_lon: f64) -> Self {
        let lon_scale = M_PER_DEGREE * (origin_lat * std::f64::consts::PI / 180.0).cos();
        Self { origin_lat, origin_lon, lon_scale }
    }

    /// A frame anchored at the mean of `fixes`. `None` for an empty list —
    /// a session with no GPS fixes has no frame, which is a true answer and
    /// not a failure (C3 §3.5).
    ///
    /// The **session's** mean, never the track's: a track outlives any one
    /// session and a session may run only part of it, so anchoring on the
    /// session is what keeps the trace centred.
    pub fn from_fixes(fixes: &[GpsFix]) -> Option<Self> {
        if fixes.is_empty() {
            return None;
        }
        let n = fixes.len() as f64;
        let lat = fixes.iter().map(|f| f.lat).sum::<f64>() / n;
        let lon = fixes.iter().map(|f| f.lon).sum::<f64>() / n;
        Some(Self::new(lat, lon))
    }

    /// Projects `(lat, lon)` decimal degrees to `(east, north)` metres from
    /// this frame's origin.
    pub fn project(&self, lat: f64, lon: f64) -> (f64, f64) {
        ((lon - self.origin_lon) * self.lon_scale, (lat - self.origin_lat) * M_PER_DEGREE)
    }
}

/// Indices of the points [`decimate_path`] keeps, or every index when the
/// path already fits the budget.
///
/// **Perpendicular distance, not stride** (C2 §5.3, ruling R217 item 1). A
/// path is geometric: a uniform stride drops whichever fixes fall between its
/// steps, and a hairpin taken slowly is exactly where the fixes are dense —
/// striding a hairpin turns it into a corner. Douglas–Peucker keeps the
/// points that carry the shape and drops the ones that sit on a straight, so
/// the result is the shape rather than a sample of it.
///
/// The returned indices are strictly increasing and always include the first
/// and last point of a non-empty path. `budget` is a maximum, not a target: a
/// straight line under any budget returns two points, not `budget` of them.
///
/// The tolerance that meets `budget` is found by bisection over
/// `[0, bounding-box diagonal]` — the simplification itself has no closed
/// form in the number of points it keeps, and 40 halvings resolve the
/// tolerance to well below a GPS receiver's own noise.
pub fn decimate_path(x: &[f64], y: &[f64], budget: usize) -> Vec<usize> {
    let n = x.len().min(y.len());
    if n == 0 {
        return Vec::new();
    }
    if n <= budget.max(2) {
        return (0..n).collect();
    }

    let (mut lo, mut hi) = (0.0_f64, bbox_diagonal(&x[..n], &y[..n]).max(1.0));
    let mut best = simplify(&x[..n], &y[..n], hi);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let kept = simplify(&x[..n], &y[..n], mid);
        if kept.len() <= budget {
            best = kept;
            hi = mid;
        } else {
            lo = mid;
        }
    }
    best
}

/// The diagonal of the path's bounding box, metres — the largest useful
/// tolerance, since simplifying at it always reduces to the two endpoints.
fn bbox_diagonal(x: &[f64], y: &[f64]) -> f64 {
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            x0 = x0.min(x[i]);
            x1 = x1.max(x[i]);
            y0 = y0.min(y[i]);
            y1 = y1.max(y[i]);
        }
    }
    if x0 > x1 {
        return 0.0;
    }
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

/// Douglas–Peucker at a fixed `tolerance` (metres), iteratively rather than
/// recursively so a long path cannot overflow the stack. Returns the kept
/// indices in increasing order.
fn simplify(x: &[f64], y: &[f64], tolerance: f64) -> Vec<usize> {
    let n = x.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let mut worst = (a, -1.0_f64);
        for i in a + 1..b {
            let d = perpendicular_distance(x[i], y[i], x[a], y[a], x[b], y[b]);
            if d > worst.1 {
                worst = (i, d);
            }
        }
        if worst.1 > tolerance {
            keep[worst.0] = true;
            stack.push((a, worst.0));
            stack.push((worst.0, b));
        }
    }

    (0..n).filter(|&i| keep[i]).collect()
}

/// Distance from `(px, py)` to the segment `(ax, ay)-(bx, by)`, metres. A
/// degenerate segment falls back to the distance from its single point, so a
/// run of identical fixes (a stationary bike) cannot make this `NaN`.
fn perpendicular_distance(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (dx, dy) = (bx - ax, by - ay);
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let tau = (((px - ax) * dx + (py - ay) * dy) / len_sq).clamp(0.0, 1.0);
    let (qx, qy) = (ax + tau * dx, ay + tau * dy);
    ((px - qx).powi(2) + (py - qy).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(lat: f64, lon: f64) -> GpsFix {
        GpsFix { timestamp_ms: 0, lat, lon }
    }

    #[test]
    fn enu_frame_projects_its_own_origin_to_the_frame_origin() {
        // Arrange
        let frame = EnuFrame::new(50.0, -120.0);

        // Act
        let (e, n) = frame.project(50.0, -120.0);

        // Assert
        assert_eq!((e, n), (0.0, 0.0));
    }

    #[test]
    fn enu_frame_one_degree_of_latitude_is_the_spherical_earth_scale() {
        // Arrange
        let frame = EnuFrame::new(0.0, 0.0);

        // Act
        let (_, n) = frame.project(1.0, 0.0);

        // Assert
        assert!((n - M_PER_DEGREE).abs() < 1e-6, "got {n} m for one degree of latitude");
    }

    #[test]
    fn enu_frame_longitude_shrinks_with_the_cosine_of_latitude() {
        // Arrange — 60° N, where a degree of longitude is half its equatorial width.
        let frame = EnuFrame::new(60.0, 0.0);

        // Act
        let (e, _) = frame.project(60.0, 1.0);

        // Assert
        assert!((e - M_PER_DEGREE * 0.5).abs() < 1.0, "got {e} m for one degree of longitude at 60 N");
    }

    #[test]
    fn enu_frame_from_fixes_anchors_on_the_mean_and_is_none_for_an_empty_list() {
        // Arrange
        let fixes = vec![fix(10.0, 20.0), fix(12.0, 24.0)];

        // Act
        let frame = EnuFrame::from_fixes(&fixes).unwrap();

        // Assert
        assert_eq!((frame.origin_lat, frame.origin_lon), (11.0, 22.0));
        assert!(EnuFrame::from_fixes(&[]).is_none());
    }

    #[test]
    fn decimate_path_a_path_already_under_budget_keeps_every_point() {
        // Arrange
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![0.0; 10];

        // Act
        let kept = decimate_path(&x, &y, 100);

        // Assert
        assert_eq!(kept, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn decimate_path_a_straight_line_reduces_to_its_two_endpoints_not_to_the_budget() {
        // Arrange — 1000 collinear points; none of the interior ones carries shape.
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y = vec![0.0; 1000];

        // Act
        let kept = decimate_path(&x, &y, 100);

        // Assert
        assert_eq!(kept, vec![0, 999]);
    }

    #[test]
    fn decimate_path_keeps_the_hairpin_a_uniform_stride_would_cut_off() {
        // Arrange — a long straight run east, then a single sharp spike north
        // at one point. A stride of 10 would miss the spike entirely.
        let mut x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let mut y = vec![0.0; 1000];
        x.push(999.0);
        y.push(400.0);
        x.push(1000.0);
        y.push(0.0);

        // Act
        let kept = decimate_path(&x, &y, 8);

        // Assert — the apex survives.
        assert!(kept.len() <= 8, "kept {} points for a budget of 8", kept.len());
        assert!(kept.contains(&1000), "the hairpin apex must survive decimation: {kept:?}");
    }

    #[test]
    fn decimate_path_never_exceeds_the_budget_and_keeps_both_endpoints() {
        // Arrange — a noisy circle, every point carrying some shape.
        let n = 5000;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos() * 100.0 + (i % 7) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin() * 100.0 + (i % 5) as f64).collect();

        // Act
        let kept = decimate_path(&x, &y, 256);

        // Assert
        assert!(kept.len() <= 256, "kept {}", kept.len());
        assert_eq!(kept.first().copied(), Some(0));
        assert_eq!(kept.last().copied(), Some(n - 1));
        assert!(kept.windows(2).all(|w| w[0] < w[1]), "indices must be strictly increasing");
    }

    #[test]
    fn decimate_path_an_empty_path_decimates_to_nothing() {
        // Arrange / Act / Assert
        assert!(decimate_path(&[], &[], 100).is_empty());
    }

    #[test]
    fn decimate_path_a_stationary_run_of_identical_fixes_does_not_produce_nan() {
        // Arrange — 500 identical points: every segment is degenerate.
        let x = vec![5.0; 500];
        let y = vec![7.0; 500];

        // Act
        let kept = decimate_path(&x, &y, 10);

        // Assert
        assert_eq!(kept, vec![0, 499]);
    }
}
