//! Position-based reference matching with directional projection.
//! See lap-delta-rewrite spec §4.1.

#[derive(Clone, Copy, Debug)]
pub struct ReferencePoint {
    pub e: f64,
    pub n: f64,
    pub t_lap: f64,    // seconds, relative to overlay-lap start
    pub s_cum: f64,    // cumulative arc length from start, metres
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionResult {
    pub s_star: f64,
    pub t_ref: f64,
    pub distance: f64,
    pub segment_index: usize,
}

pub struct Projector {
    reference: Vec<ReferencePoint>,
    prev_segment: Option<usize>,
}

const LOCAL_SEARCH_RADIUS: usize = 10;
const HEADING_TOLERANCE_RAD: f64 = std::f64::consts::FRAC_PI_2; // ±90°

impl Projector {
    pub fn new(samples: Vec<(f64, f64, f64)>) -> Self {
        let mut reference = Vec::with_capacity(samples.len());
        let mut s = 0.0;
        for (i, (e, n, t)) in samples.iter().enumerate() {
            if i > 0 {
                let de = e - samples[i - 1].0;
                let dn = n - samples[i - 1].1;
                s += (de * de + dn * dn).sqrt();
            }
            reference.push(ReferencePoint { e: *e, n: *n, t_lap: *t, s_cum: s });
        }
        Self { reference, prev_segment: None }
    }

    pub fn project(&mut self, e: f64, n: f64, heading_rad: f64)
        -> Option<ProjectionResult>
    {
        if self.reference.len() < 2 {
            return None;
        }
        let (start, end) = self.search_window();

        let mut best: Option<(usize, f64, f64)> = None; // (idx, dist_sq, tau)
        for k in start..end {
            let a = &self.reference[k];
            let b = &self.reference[k + 1];
            let dx = b.e - a.e;
            let dy = b.n - a.n;
            let len_sq = dx * dx + dy * dy;
            if len_sq < 1e-12 { continue; }

            // Heading match: segment heading vs sample heading.
            let seg_heading = dy.atan2(dx);
            let heading_diff = wrap_pi(heading_rad - seg_heading);
            if heading_diff.abs() > HEADING_TOLERANCE_RAD { continue; }

            // Perpendicular projection.
            let tau = ((e - a.e) * dx + (n - a.n) * dy) / len_sq;
            let tau_clamped = tau.clamp(0.0, 1.0);
            let qx = a.e + tau_clamped * dx;
            let qy = a.n + tau_clamped * dy;
            let dist_sq = (e - qx).powi(2) + (n - qy).powi(2);
            match best {
                None => best = Some((k, dist_sq, tau_clamped)),
                Some((_, prev_dist_sq, _)) if dist_sq < prev_dist_sq => {
                    best = Some((k, dist_sq, tau_clamped));
                }
                _ => {}
            }
        }

        let (k, dist_sq, tau) = best?;
        self.prev_segment = Some(k);
        let a = &self.reference[k];
        let b = &self.reference[k + 1];
        let seg_len = b.s_cum - a.s_cum;
        Some(ProjectionResult {
            s_star: a.s_cum + tau * seg_len,
            t_ref: a.t_lap + tau * (b.t_lap - a.t_lap),
            distance: dist_sq.sqrt(),
            segment_index: k,
        })
    }

    fn search_window(&self) -> (usize, usize) {
        let n_segments = self.reference.len() - 1;
        match self.prev_segment {
            None => (0, n_segments),
            Some(p) => {
                let start = p.saturating_sub(LOCAL_SEARCH_RADIUS);
                let end = (p + LOCAL_SEARCH_RADIUS + 1).min(n_segments);
                (start, end)
            }
        }
    }
}

fn wrap_pi(x: f64) -> f64 {
    use std::f64::consts::PI;
    let mut y = x;
    while y > PI { y -= 2.0 * PI; }
    while y < -PI { y += 2.0 * PI; }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_lap() -> Vec<(f64, f64, f64)> {
        // Straight east, 100 m total, samples every 10 m. t_lap = 0..10s.
        (0..=10).map(|i| (i as f64 * 10.0, 0.0, i as f64)).collect()
    }

    #[test]
    fn identity_projection() {
        // Arrange
        let mut p = Projector::new(ref_lap());

        // Act
        let r = p.project(50.0, 0.0, 0.0).unwrap();

        // Assert — sample sits exactly at the 50 m point on the reference.
        assert!((r.s_star - 50.0).abs() < 1e-6);
        assert!((r.distance).abs() < 1e-6);
        assert!((r.t_ref - 5.0).abs() < 1e-6);
    }

    #[test]
    fn offline_projection() {
        // Arrange
        let mut p = Projector::new(ref_lap());

        // Act — sample 5 m north of the 30 m point.
        let r = p.project(30.0, 5.0, 0.0).unwrap();

        // Assert
        assert!((r.s_star - 30.0).abs() < 1e-6);
        assert!((r.distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn switchback_heading_rejection() {
        // Arrange — reference goes east. Sample is on the line but heading west.
        let mut p = Projector::new(ref_lap());

        // Act — heading π (west) is 180° from segment heading 0 (east).
        let r = p.project(50.0, 0.0, std::f64::consts::PI);

        // Assert — projection rejected.
        assert!(r.is_none());
    }
}
