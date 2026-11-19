//! What drives a dynamic run: the front triangle's motion, the rear travel
//! and the wheel's speed over time (spec §5). Synthetic inputs now; a
//! recorded ride's IMU0, physics travel and GPS later.

use super::{SimError, SimErrorKind};

/// Standard gravity, m/s².
pub const G: f64 = 9.806_65;

/// The prescribed inputs, sampled on increasing time.
#[derive(Debug, Clone, PartialEq)]
pub struct Excitation {
    /// Sample times, seconds, increasing.
    pub t_s: Vec<f64>,
    /// Rear travel, metres.
    pub travel_m: Vec<f64>,
    /// Specific force at IMU0 in the side plane (what an accelerometer there
    /// reads, gravity included), x toward the front axle, m/s².
    pub fx: Vec<f64>,
    /// Specific force at IMU0, y up, m/s² (+g at rest on level ground).
    pub fy: Vec<f64>,
    /// Front triangle's rotation rate in the side plane, rad/s, CCW (nose up).
    pub omega: Vec<f64>,
    /// Ground speed, m/s.
    pub speed_mps: Vec<f64>,
}

/// One instant of an [`Excitation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Rear travel, metres.
    pub travel_m: f64,
    /// Specific force at IMU0, m/s².
    pub f: [f64; 2],
    /// Rotation rate, rad/s (CCW).
    pub omega: f64,
    /// Rotation acceleration, rad/s² (CCW).
    pub alpha: f64,
    /// Ground speed, m/s.
    pub speed_mps: f64,
}

impl Excitation {
    /// Checks the series line up and time increases.
    pub fn validate(&self) -> Result<(), SimError> {
        let n = self.t_s.len();
        if n < 2 || [self.travel_m.len(), self.fx.len(), self.fy.len(), self.omega.len(), self.speed_mps.len()].iter().any(|&l| l != n) {
            return Err(SimError::new(SimErrorKind::Excitation, "the excitation needs at least two samples of every input"));
        }
        if self.t_s.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(SimError::new(SimErrorKind::Excitation, "the excitation's time must increase"));
        }
        Ok(())
    }

    /// Start and end time, seconds.
    pub fn span(&self) -> (f64, f64) {
        (self.t_s[0], *self.t_s.last().unwrap())
    }

    /// The inputs at time `t` (linear between samples, held past the ends).
    pub fn at(&self, t: f64) -> Sample {
        let n = self.t_s.len();
        let i = match self.t_s.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
            Ok(i) => i.min(n - 2),
            Err(0) => 0,
            Err(i) => (i - 1).min(n - 2),
        };
        let (t0, t1) = (self.t_s[i], self.t_s[i + 1]);
        let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        let lerp = |v: &[f64]| v[i] + f * (v[i + 1] - v[i]);
        Sample {
            travel_m: lerp(&self.travel_m),
            f: [lerp(&self.fx), lerp(&self.fy)],
            omega: lerp(&self.omega),
            alpha: (self.omega[i + 1] - self.omega[i]) / (t1 - t0),
            speed_mps: lerp(&self.speed_mps),
        }
    }
}

/// A train of identical hits: the rear compresses `depth_m` in `compress_s`
/// (a half-cosine), rebounds in `rebound_s`, while the front triangle is
/// pushed up with a half-sine of `chassis_g` peak over the compression.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BumpTrain {
    /// Ground speed, m/s.
    pub speed_mps: f64,
    /// Rear travel at the bottom of each hit, metres.
    pub depth_m: f64,
    /// Static sag (travel between hits), metres.
    pub sag_m: f64,
    /// Compression time, seconds.
    pub compress_s: f64,
    /// Rebound time, seconds.
    pub rebound_s: f64,
    /// Front triangle's extra upward acceleration at the peak of a hit, g.
    pub chassis_g: f64,
    /// Number of hits.
    pub count: u32,
    /// Time from one hit's start to the next, seconds.
    pub spacing_s: f64,
    /// Quiet time before the first hit and after the last, seconds.
    pub lead_s: f64,
}

impl Default for BumpTrain {
    fn default() -> Self {
        BumpTrain { speed_mps: 6.0, depth_m: 0.11, sag_m: 0.045, compress_s: 0.06, rebound_s: 0.14, chassis_g: 3.0, count: 4, spacing_s: 0.35, lead_s: 0.25 }
    }
}

impl BumpTrain {
    /// The train as an excitation sampled at `rate_hz`.
    pub fn excitation(&self, rate_hz: f64) -> Excitation {
        let total = 2.0 * self.lead_s + self.spacing_s * self.count.saturating_sub(1) as f64 + self.compress_s + self.rebound_s;
        let n = (total * rate_hz).ceil() as usize + 1;
        let mut e = Excitation { t_s: Vec::with_capacity(n), travel_m: Vec::with_capacity(n), fx: vec![0.0; n], fy: Vec::with_capacity(n), omega: vec![0.0; n], speed_mps: vec![self.speed_mps; n] };
        let pi = std::f64::consts::PI;
        for k in 0..n {
            let t = k as f64 / rate_hz;
            let mut s = self.sag_m;
            let mut a = 0.0;
            for h in 0..self.count {
                let t0 = self.lead_s + h as f64 * self.spacing_s;
                let u = t - t0;
                if u >= 0.0 && u < self.compress_s {
                    s = self.sag_m + (self.depth_m - self.sag_m) * 0.5 * (1.0 - (pi * u / self.compress_s).cos());
                    a += self.chassis_g * G * (pi * u / self.compress_s).sin();
                } else if u >= self.compress_s && u < self.compress_s + self.rebound_s {
                    let v = (u - self.compress_s) / self.rebound_s;
                    s = self.sag_m + (self.depth_m - self.sag_m) * 0.5 * (1.0 + (pi * v).cos());
                }
            }
            e.t_s.push(t);
            e.travel_m.push(s);
            e.fy.push(G + a);
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_train_rests_at_sag_and_reaches_depth_at_each_hit() {
        // Arrange
        let b = BumpTrain::default();

        // Act
        let e = b.excitation(1000.0);

        // Assert
        e.validate().unwrap();
        assert!((e.travel_m[0] - b.sag_m).abs() < 1e-12);
        let peak = e.travel_m.iter().cloned().fold(0.0, f64::max);
        assert!((peak - b.depth_m).abs() < 1e-4, "{peak}");
        assert!((e.at(0.0).f[1] - G).abs() < 1e-12);
    }

    #[test]
    fn at_interpolates_between_samples_and_holds_past_the_ends() {
        // Arrange
        let e = Excitation { t_s: vec![0.0, 1.0], travel_m: vec![0.0, 0.1], fx: vec![0.0; 2], fy: vec![G; 2], omega: vec![0.0, 2.0], speed_mps: vec![5.0; 2] };

        // Act
        let (mid, past) = (e.at(0.5), e.at(3.0));

        // Assert
        assert!((mid.travel_m - 0.05).abs() < 1e-12 && (mid.alpha - 2.0).abs() < 1e-12);
        assert!((past.travel_m - 0.1).abs() < 1e-12);
    }
}
