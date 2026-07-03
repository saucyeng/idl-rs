//! Fixed per-bike geometry the estimator consumes as a per-run input (data-in,
//! data-out). All directions are unit vectors in the **chassis (IMU0 vehicle)
//! frame**, ISO 8855: X = forward, Y = left, Z = up. Lengths are metres.
//!
//! This is the geometry the design (§6, §8) authors from the 4-bar pivot points;
//! for M2a the rear axle path is carried in its already-derived **sampled** form
//! (the design's allowed "rear 4-bar points *or* sampled axle-path") — the full
//! linkage solver that produces these samples is an authoring-time concern and is
//! deferred. The front (telescoping fork) is a straight prismatic path with a
//! constant tangent.
//!
//! See docs/superpowers/specs/2026-06-23-suspension-estimator-design.md §6.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Suspension topology — drives the geometry-derived state schema (rear states and
/// the rear-unsprung IMU drop out on a hardtail).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    /// Front + rear suspension (this reference bike).
    FullSuspension,
    /// Front suspension only — no rear axle path, no IMU2.
    Hardtail,
}

/// Pose of one IMU relative to the chassis (IMU0) frame, at the suspension neutral
/// (sag) configuration.
#[derive(Debug, Clone)]
pub struct ImuPose {
    /// Rotation mapping this sensor's body frame → chassis frame at neutral.
    pub mount: UnitQuaternion<f64>,
    /// Offset from the IMU0 origin to this IMU, in the chassis frame, metres.
    pub lever: Vector3<f64>,
}

impl ImuPose {
    /// An IMU mounted coincident with and aligned to the chassis frame.
    pub fn aligned_at_origin() -> Self {
        ImuPose { mount: UnitQuaternion::identity(), lever: Vector3::zeros() }
    }
}

/// A rear axle path sampled as (travel arc-length `s` in metres, axle position in
/// the chassis frame in metres, relative to the neutral axle location). The
/// **tangent** `t(s) = d(position)/ds` (normalized) is the measurement/parameter
/// axis for rear wheel travel (§6); it rotates through travel.
#[derive(Debug, Clone)]
pub struct AxlePath {
    /// Strictly increasing `s` samples paired with axle positions (chassis frame).
    samples: Vec<(f64, Vector3<f64>)>,
}

impl AxlePath {
    /// Builds a path from `(s, position)` samples. `s` must be strictly increasing
    /// and start at 0; positions are chassis-frame, metres, relative to neutral.
    pub fn from_samples(samples: Vec<(f64, Vector3<f64>)>) -> Self {
        AxlePath { samples }
    }

    /// Unit tangent `t(s)` at travel `s` (metres): the direction the axle moves per
    /// unit increase in travel. Piecewise from the bracketing sample secant, clamped
    /// to the sampled range. Chassis frame, dimensionless unit vector.
    pub fn tangent(&self, s: f64) -> Vector3<f64> {
        // Find the segment [s_i, s_{i+1}] containing s; clamp to the first/last
        // segment outside the sampled range. The tangent is the segment secant
        // (P_{i+1} − P_i) normalized.
        let n = self.samples.len();
        let mut i = 0;
        while i + 2 < n && s > self.samples[i + 1].0 {
            i += 1;
        }
        let secant = self.samples[i + 1].1 - self.samples[i].1;
        secant.normalize()
    }
}

/// Fixed geometry for one bike. Consumed per estimator run (§8: sourced from the
/// session's `.idl0w`, optionally stamped from a `.idl0p` profile).
#[derive(Debug, Clone)]
pub struct BikeGeometry {
    /// Suspension topology.
    pub topology: Topology,
    /// Steer axis (pointing up), chassis frame unit vector. Tilted back by the
    /// head-angle complement (90° − head angle). Used by steering (M2b).
    pub steer_axis: Vector3<f64>,
    /// Front fork compression tangent (direction the front axle moves per unit
    /// travel), chassis frame unit vector. Constant (telescoping fork).
    pub front_axis: Vector3<f64>,
    /// Maximum front wheel travel, metres.
    pub front_travel_max: f64,
    /// Front static sag as a fraction of `front_travel_max` (0..1).
    pub front_sag: f64,
    /// Rear axle path (None on a hardtail).
    pub rear_path: Option<AxlePath>,
    /// Maximum rear wheel travel, metres.
    pub rear_travel_max: f64,
    /// Rear static sag as a fraction of `rear_travel_max` (0..1).
    pub rear_sag: f64,
    /// Chassis/sprung IMU pose (reference frame; nominally aligned at origin).
    pub imu0: ImuPose,
    /// Front-unsprung IMU pose (fork lower).
    pub imu1: ImuPose,
    /// Rear-unsprung IMU pose (None on a hardtail).
    pub imu2: Option<ImuPose>,
}

impl BikeGeometry {
    /// The reference full-suspension bike from the design doc: 170 mm front /
    /// 160 mm rear travel, 27% sag, 63.5° head angle, curved rear axle path
    /// (rearward to ~13 mm then returning). IMU lever arms are nominal placeholders
    /// pending authored geometry (refined in the M4 editor).
    pub fn reference_bike() -> BikeGeometry {
        // Head angle 63.5° → 26.5° off vertical; compression/steer axes point up
        // (+Z) and rearward (−X) in the sagittal plane.
        let off_vertical = (90.0_f64 - 63.5).to_radians();
        let tilt_back = Vector3::new(-off_vertical.sin(), 0.0, off_vertical.cos());

        // Rear axle path: rises ~vertically, leaning rearward to ~13 mm at mid
        // travel then returning (design doc). Sampled (s, position) in metres.
        let rear_path = AxlePath::from_samples(vec![
            (0.000, Vector3::new(0.000, 0.0, 0.000)),
            (0.040, Vector3::new(-0.008, 0.0, 0.040)),
            (0.080, Vector3::new(-0.013, 0.0, 0.080)),
            (0.120, Vector3::new(-0.010, 0.0, 0.120)),
            (0.160, Vector3::new(-0.004, 0.0, 0.160)),
        ]);

        // CALIBRATED unsprung mounts (§7), baked from real data — used as-is by
        // `run()` (per-session refinement is opt-in via `refine_mounts`, default off,
        // so a ride log's parked lean / flopped bars can't corrupt them). Provenance:
        // - Tilt: the flat-ground motionless capture (2026-06-20_18-10-32.idl0,
        //   10.6 s stationary window), each sensor's mean gravity aligned to the
        //   MOUNTED IMU0's window-mean up (the bike leaned 13.2° even in that
        //   controlled capture — aligning to chassis +Z would have baked that in).
        // - Yaw datum (which gravity cannot observe): disambiguated on a riding log
        //   (2026-06-13_10-21-26.idl0) by horizontal diff-accel RMS — shared braking/
        //   cornering accelerations cancel under the correct yaw and double under a
        //   180° error. Both original coarse picks lost (23.4→17.8, 18.7→15.5 m/s²):
        //   they had inherited a 180° yaw from the pre-fix identity IMU0 frame.
        // As axis maps (chassis = R · sensor), post-calibration: IMU1 (fork lower)
        // sensor +X→up, +Y→rearward, +Z→right; IMU2 (seatstay) +Y→up, +Z→right,
        // +X→forward — each plus a few degrees of fitted static link tilt (3.3°
        // front, 10.1° rear). Recompute by re-running the bake (see CHANGELOG entry)
        // if a sensor is remounted.
        let imu1_mount = UnitQuaternion::from_quaternion(Quaternion::new(
            -4.78101903822033492e-1,
            -5.01109229357781349e-1,
            5.18398525639093610e-1,
            -5.01568617867397371e-1,
        ));
        let imu2_mount = UnitQuaternion::from_quaternion(Quaternion::new(
            -7.08515127451586846e-1,
            -7.00155077592052089e-1,
            6.73954169987484863e-2,
            -5.69827979472209215e-2,
        ));

        BikeGeometry {
            topology: Topology::FullSuspension,
            steer_axis: tilt_back,
            front_axis: tilt_back,
            front_travel_max: 0.170,
            front_sag: 0.27,
            rear_path: Some(rear_path),
            rear_travel_max: 0.160,
            rear_sag: 0.27,
            // IMU0 (sprung/chassis reference) is mounted **X-rear, Y-right** (Z up) —
            // a 180° yaw from the ISO chassis frame (X-forward, Y-left, Z-up) the lever
            // arms, fork/steer axes, and rear path are expressed in. Encoding that here
            // (not identity) rotates IMU0's gyro/accel into the ISO frame so the lever-
            // arm transport `ω̇×L + ω×(ω×L)` cancels with the correct sign. Identity (the
            // old value) left ωx/ωy and the horizontal specific force inverted, which
            // *doubled* the rotational transport into wheel travel instead of cancelling
            // it — railing front travel to the barrier on hard pitch (landings).
            imu0: ImuPose {
                mount: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI),
                lever: Vector3::zeros(),
            },
            // Lever arms (metres, chassis frame, from IMU0): X = the 4-bar axle
            // positions (front +0.835, rear −0.445 m from the BB), Z ≈ −0.4 (IMU0 sits
            // ~0.4 m above the axles). Approximate (the unsprung IMU sits along the
            // fork/stay, not exactly at the axle). Mounts are the coarse picks above,
            // tilt-refined per session by the orientation step.
            imu1: ImuPose { mount: imu1_mount, lever: Vector3::new(0.835, 0.0, -0.4) },
            imu2: Some(ImuPose { mount: imu2_mount, lever: Vector3::new(-0.445, 0.0, -0.4) }),
        }
    }

    /// Front fork compression tangent (constant). Chassis frame unit vector.
    pub fn front_tangent(&self) -> Vector3<f64> {
        self.front_axis
    }

    /// Rear axle-path tangent at rear travel `s` (metres). Panics if called on a
    /// hardtail (no rear path) — the state schema gates this off.
    pub fn rear_tangent(&self, s: f64) -> Vector3<f64> {
        self.rear_path
            .as_ref()
            .expect("rear_tangent called on a bike with no rear axle path")
            .tangent(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reference_bike_has_design_travel_and_sag() {
        // Arrange + Act
        let g = BikeGeometry::reference_bike();

        // Assert — design doc: 170 mm front, 160 mm rear, 27% sag, full-sus.
        assert_eq!(g.topology, Topology::FullSuspension);
        assert_relative_eq!(g.front_travel_max, 0.170, epsilon = 1e-9);
        assert_relative_eq!(g.rear_travel_max, 0.160, epsilon = 1e-9);
        assert_relative_eq!(g.front_sag, 0.27, epsilon = 1e-9);
        assert_relative_eq!(g.rear_sag, 0.27, epsilon = 1e-9);
        assert!(g.imu2.is_some());
        assert!(g.rear_path.is_some());
    }

    #[test]
    fn reference_bike_imu0_mount_maps_x_rear_y_right_into_iso_frame() {
        // Ground truth (measured on the bike): IMU0 sits X-rear, Y-right, Z-up — a 180°
        // yaw from the ISO chassis frame (X-forward, Y-left, Z-up) the levers/axes/path
        // are expressed in. The mount must rotate IMU0's sensor axes into that ISO frame
        // so the lever-arm transport cancels with the right sign. Non-circular: this
        // checks the mount against the physical orientation, not against the estimator
        // math that consumes it.
        let m = BikeGeometry::reference_bike().imu0.mount;

        // sensor +X (rear) → chassis −X; +Y (right) → −Y; +Z (up) → +Z.
        assert_relative_eq!(m * Vector3::x(), -Vector3::x(), epsilon = 1e-12);
        assert_relative_eq!(m * Vector3::y(), -Vector3::y(), epsilon = 1e-12);
        assert_relative_eq!(m * Vector3::z(), Vector3::z(), epsilon = 1e-12);
    }

    #[test]
    fn front_tangent_is_unit_and_tilted_back_by_head_angle_complement() {
        // Arrange — telescoping fork tilts back 26.5° (= 90° − 63.5° head angle):
        // compression moves the axle up (+Z) and rearward (−X).
        let g = BikeGeometry::reference_bike();

        // Act
        let t = g.front_tangent();
        let off_vertical = t.z.acos().to_degrees(); // angle from +Z

        // Assert
        assert_relative_eq!(t.norm(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(off_vertical, 26.5, epsilon = 0.1);
        assert!(t.x < 0.0, "compression tangent points rearward (−X)");
        assert_relative_eq!(t.y, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn steer_axis_is_unit_and_tilted_back_by_head_angle_complement() {
        // Arrange
        let g = BikeGeometry::reference_bike();

        // Act
        let s = g.steer_axis;
        let off_vertical = s.z.acos().to_degrees();

        // Assert — steer axis (pointing up) tilts back 26.5° off vertical.
        assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(off_vertical, 26.5, epsilon = 0.1);
        assert!(s.x < 0.0);
    }

    #[test]
    fn rear_tangent_is_unit_and_near_vertical_at_zero_travel() {
        // Arrange — the seatstay axle path is mostly vertical with a small rearward
        // component; at the start it rises with a slight rearward lean.
        let g = BikeGeometry::reference_bike();

        // Act
        let t = g.rear_tangent(0.0);

        // Assert
        assert_relative_eq!(t.norm(), 1.0, epsilon = 1e-9);
        assert!(t.z > 0.9, "rear path is predominantly vertical near neutral");
        assert!(t.x < 0.0, "rear axle leans rearward early in travel");
    }

    #[test]
    fn axle_path_tangent_equals_segment_secant_direction() {
        // Arrange — a simple two-segment path: straight up, then up-and-back.
        let path = AxlePath::from_samples(vec![
            (0.0, Vector3::new(0.0, 0.0, 0.0)),
            (0.1, Vector3::new(0.0, 0.0, 0.1)),
            (0.2, Vector3::new(-0.05, 0.0, 0.2)),
        ]);

        // Act — a query inside the second segment uses that segment's secant.
        let t = path.tangent(0.15);

        // Assert — secant of segment 2 is (−0.05,0,0.1) → normalized.
        let expected = Vector3::new(-0.05, 0.0, 0.1).normalize();
        assert_relative_eq!(t, expected, epsilon = 1e-9);
        assert_relative_eq!(t.norm(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn axle_path_tangent_clamps_below_first_sample() {
        // Arrange
        let path = AxlePath::from_samples(vec![
            (0.0, Vector3::new(0.0, 0.0, 0.0)),
            (0.1, Vector3::new(-0.02, 0.0, 0.1)),
        ]);

        // Act — a negative query clamps to the first segment.
        let t = path.tangent(-0.5);

        // Assert
        let expected = Vector3::new(-0.02, 0.0, 0.1).normalize();
        assert_relative_eq!(t, expected, epsilon = 1e-9);
    }
}
