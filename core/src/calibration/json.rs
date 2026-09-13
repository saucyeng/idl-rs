//! The calibration record as JSON — the `calibration.*` reserved block of the
//! session's setup sheet (spec §6, question 1).
//!
//! [`rigid::CalibrationRecord`] is the maths layer's type, carrying `nalgebra`
//! quaternions and vectors. This module is its serialised form: plain arrays,
//! units in every field name, and **every unobserved field simply absent**
//! rather than zero (R190 — blank means "not applicable", and a zero lever arm
//! is a claim, not a blank). Round-tripping is exact, so a stored record can
//! be read back into the estimator without re-fitting.
//!
//! `nalgebra` is not built with its `serde-serialize` feature here, which is
//! why this is a separate representation rather than a derive on the record
//! itself. That is no loss: the stored shape is a contract with the setup
//! sheet and deserves to be written out where it can be read.

use serde::{Deserialize, Serialize};

use super::rigid::{
    AccelBiasDifference, Body, CalibrationRecord, CalibrationSource, ExcitationShortfall,
    MountOrigin, Quality, SensorCalibration,
};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// One session's `calibration.*` block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationJson {
    /// `rigid::MODEL_VERSION` at the time of the fit.
    pub model_version: u32,
    /// Capture time, RFC 3339. Absent when the caller supplied none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_utc: Option<String>,
    /// `rigid_body` or `rest_only`.
    pub source: String,
    /// Per sensor, ascending by `imu_index`.
    pub sensors: Vec<SensorJson>,
    /// `β_ij` per rear-body pair. Empty when no lever arm was fitted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accel_bias_differences: Vec<AccelBiasDifferenceJson>,
    /// Steering-axis unit vector in the rear body frame, dimensionless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer_axis: Option<[f64; 3]>,
    /// `C₀` as `(w, x, y, z)`, dimensionless — the spec §1.2a identity gauge.
    pub steer_datum: [f64; 4],
    /// The §3 metrics.
    pub quality: QualityJson,
}

/// One sensor's entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorJson {
    /// `imu_index` as it appears in the file's records.
    pub imu_index: u8,
    /// `rear` or `front`.
    pub body: String,
    /// Sensor frame → body frame as `(w, x, y, z)`, dimensionless.
    pub mount: [f64; 4],
    /// `measured` or `gauge`.
    pub mount_origin: String,
    /// Lever arm from the body origin, body frame, metres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lever_m: Option<[f64; 3]>,
    /// Gyro bias in the sensor frame, rad/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gyro_bias_rad_s: Option<[f64; 3]>,
}

/// `β_ij = R_i b_a,i − R_j b_a,j`, body frame, m/s².
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccelBiasDifferenceJson {
    pub from_imu: u8,
    pub to_imu: u8,
    pub value_m_s2: [f64; 3],
}

/// The §3 metrics and whatever they ruled out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityJson {
    /// `cond(Σ ω ωᵀ)` over the tumble, dimensionless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_condition: Option<f64>,
    /// Median `‖ω_R‖` over the tumble, rad/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_rate_rad_s: Option<f64>,
    /// `min eig(Σ ΩᵀΩ)/N`, s⁻⁴.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lever_richness_s4: Option<f64>,
    /// Peak-to-peak `δ` over the bar turn, rad.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer_peak_to_peak_rad: Option<f64>,
    /// Stationary hold, seconds.
    pub rest_duration_s: f64,
    /// Tumble, seconds.
    pub datum_duration_s: f64,
    /// RMS rear-body gyro-consistency residual after the fit, rad/s (§7.6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rear_body_residual_rad_s: Option<f64>,
    /// Levenberg–Marquardt steps accepted.
    pub lm_iterations: u32,
    /// Every gate that failed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shortfalls: Vec<ShortfallJson>,
}

/// One failed gate, in a uniform shape so a consumer need not branch on the
/// metric to read the numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortfallJson {
    /// Which gate: `rotation_richness`, `rate_magnitude`, `lever_arm_richness`,
    /// `steer_richness`, `rest_hold`, `duration`, `no_front_sensor` or
    /// `no_rear_pair`.
    pub metric: String,
    /// What the session showed, in [`Self::units`]. Absent for the two
    /// structural shortfalls, which are not measurements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seen: Option<f64>,
    /// What the gate needed, in [`Self::units`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needed: Option<f64>,
    /// Units of [`Self::seen`] and [`Self::needed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    /// The operator-facing sentence, which always says both numbers.
    pub message: String,
}

fn quaternion_array(q: &UnitQuaternion<f64>) -> [f64; 4] {
    [q.w, q.i, q.j, q.k]
}

fn vector_array(v: &Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

impl From<&ExcitationShortfall> for ShortfallJson {
    fn from(shortfall: &ExcitationShortfall) -> ShortfallJson {
        let message = shortfall.to_string();
        let (metric, seen, needed, units) = match *shortfall {
            ExcitationShortfall::RotationRichness { seen, needed } => {
                ("rotation_richness", Some(seen), Some(needed), Some(""))
            }
            ExcitationShortfall::RateMagnitude { seen, needed } => {
                ("rate_magnitude", Some(seen), Some(needed), Some("rad/s"))
            }
            ExcitationShortfall::LeverArmRichness { seen, needed } => {
                ("lever_arm_richness", Some(seen), Some(needed), Some("s^-4"))
            }
            ExcitationShortfall::SteerRichness { seen, needed } => {
                ("steer_richness", Some(seen), Some(needed), Some("rad"))
            }
            ExcitationShortfall::RestHold { seen_s, needed_s } => {
                ("rest_hold", Some(seen_s), Some(needed_s), Some("s"))
            }
            ExcitationShortfall::Duration { seen_s, needed_s } => {
                ("duration", Some(seen_s), Some(needed_s), Some("s"))
            }
            ExcitationShortfall::NoFrontSensor => ("no_front_sensor", None, None, None),
            ExcitationShortfall::NoRearPair => ("no_rear_pair", None, None, None),
        };
        ShortfallJson {
            metric: metric.to_string(),
            seen,
            needed,
            units: units.map(str::to_string),
            message,
        }
    }
}

impl From<&Quality> for QualityJson {
    fn from(quality: &Quality) -> QualityJson {
        QualityJson {
            rotation_condition: quality.rotation_condition,
            median_rate_rad_s: quality.median_rate_rad_s,
            lever_richness_s4: quality.lever_richness,
            steer_peak_to_peak_rad: quality.steer_peak_to_peak_rad,
            rest_duration_s: quality.rest_duration_s,
            datum_duration_s: quality.datum_duration_s,
            rear_body_residual_rad_s: quality.rear_body_residual_rad_s,
            lm_iterations: quality.lm_iterations,
            shortfalls: quality.shortfalls.iter().map(ShortfallJson::from).collect(),
        }
    }
}

impl From<&SensorCalibration> for SensorJson {
    fn from(sensor: &SensorCalibration) -> SensorJson {
        SensorJson {
            imu_index: sensor.imu_index,
            body: match sensor.body {
                Body::Rear => "rear",
                Body::Front => "front",
            }
            .to_string(),
            mount: quaternion_array(&sensor.mount),
            mount_origin: match sensor.mount_origin {
                MountOrigin::Measured => "measured",
                MountOrigin::Gauge => "gauge",
            }
            .to_string(),
            lever_m: sensor.lever_m.as_ref().map(vector_array),
            gyro_bias_rad_s: sensor.gyro_bias_rad_s.as_ref().map(vector_array),
        }
    }
}

impl From<&AccelBiasDifference> for AccelBiasDifferenceJson {
    fn from(difference: &AccelBiasDifference) -> AccelBiasDifferenceJson {
        AccelBiasDifferenceJson {
            from_imu: difference.from_imu,
            to_imu: difference.to_imu,
            value_m_s2: vector_array(&difference.value_m_s2),
        }
    }
}

impl From<&CalibrationRecord> for CalibrationJson {
    fn from(record: &CalibrationRecord) -> CalibrationJson {
        CalibrationJson {
            model_version: record.model_version,
            captured_utc: record.captured_utc.clone(),
            source: match record.source {
                CalibrationSource::RigidBody => "rigid_body",
                CalibrationSource::RestOnly => "rest_only",
            }
            .to_string(),
            sensors: record.sensors.iter().map(SensorJson::from).collect(),
            accel_bias_differences: record
                .accel_bias_differences
                .iter()
                .map(AccelBiasDifferenceJson::from)
                .collect(),
            steer_axis: record.steer_axis.as_ref().map(vector_array),
            steer_datum: quaternion_array(&record.steer_datum),
            quality: QualityJson::from(&record.quality),
        }
    }
}

impl CalibrationJson {
    /// The record as a quaternion, in the form the estimator consumes. Only
    /// the fields a consumer of the *calibration* needs come back; the quality
    /// block stays in the JSON, where a reviewer reads it.
    pub fn mount_of(&self, imu_index: u8) -> Option<UnitQuaternion<f64>> {
        let sensor = self.sensors.iter().find(|s| s.imu_index == imu_index)?;
        let [w, x, y, z] = sensor.mount;
        Some(UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)))
    }

    /// The lever arm of `imu_index`, metres, when one was measured.
    pub fn lever_of(&self, imu_index: u8) -> Option<Vector3<f64>> {
        let sensor = self.sensors.iter().find(|s| s.imu_index == imu_index)?;
        sensor.lever_m.map(Vector3::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::rigid::{self, CalibrationInput, SensorStream};
    use crate::estimate::noise::ImuNoise;

    fn minimal_record() -> CalibrationRecord {
        // A stationary-only session: the §4 degenerate case, which is the one
        // that exercises every "absent, never zero" field at once.
        let input = CalibrationInput {
            sample_rate_hz: 100.0,
            noise: ImuNoise {
                gyro_arw: 0.003,
                accel_vrw: 0.05,
                gyro_bias_rw: 0.0,
                accel_bias_rw: 0.0,
            },
            captured_utc: Some("2026-01-01T00:00:00Z".to_string()),
            sensors: vec![SensorStream {
                imu_index: 0,
                body: rigid::Body::Rear,
                gyro_rad_s: vec![Vector3::new(0.001, -0.002, 0.003); 1000],
                accel_m_s2: vec![Vector3::new(0.0, 0.0, 9.80665); 1000],
            }],
        };

        rigid::calibrate(&input).expect("a rest-only session still produces a record")
    }

    #[test]
    fn rest_only_record_omits_every_unobserved_field_rather_than_zeroing_it() {
        // Arrange
        let record = minimal_record();

        // Act
        let json = serde_json::to_string(&CalibrationJson::from(&record)).unwrap();

        // Assert — R190: blank is "not applicable", and a zero lever arm would
        // be a claim about where the sensor sits.
        assert!(json.contains(r#""source":"rest_only""#), "{json}");
        assert!(!json.contains("lever_m"), "{json}");
        assert!(!json.contains("steer_axis"), "{json}");
        assert!(json.contains("gyro_bias_rad_s"), "{json}");
    }

    #[test]
    fn the_record_round_trips_through_json_unchanged() {
        // Arrange — a record written out by hand rather than fitted, so the
        // test is about the serialised shape and not about whatever float a
        // particular solve happened to land on.
        let record = CalibrationJson {
            model_version: 1,
            captured_utc: Some("2026-01-01T00:00:00Z".to_string()),
            source: "rigid_body".to_string(),
            sensors: vec![
                SensorJson {
                    imu_index: 0,
                    body: "rear".to_string(),
                    mount: [1.0, 0.0, 0.0, 0.0],
                    mount_origin: "gauge".to_string(),
                    lever_m: None,
                    gyro_bias_rad_s: Some([0.001, -0.0015, 0.0007]),
                },
                SensorJson {
                    imu_index: 2,
                    body: "rear".to_string(),
                    mount: [0.25, 0.5, -0.75, 0.125],
                    mount_origin: "measured".to_string(),
                    lever_m: Some([-0.43, 0.02, -0.26]),
                    gyro_bias_rad_s: Some([0.0005, 0.0021, -0.0011]),
                },
            ],
            accel_bias_differences: vec![AccelBiasDifferenceJson {
                from_imu: 2,
                to_imu: 0,
                value_m_s2: [0.0125, -0.03125, 0.0625],
            }],
            steer_axis: Some([-0.4462, 0.0, 0.8949]),
            steer_datum: [1.0, 0.0, 0.0, 0.0],
            quality: QualityJson {
                rotation_condition: Some(2.875),
                median_rate_rad_s: Some(2.96875),
                lever_richness_s4: Some(114.5),
                steer_peak_to_peak_rad: Some(1.5),
                rest_duration_s: 9.5,
                datum_duration_s: 32.5,
                rear_body_residual_rad_s: Some(0.125),
                lm_iterations: 3,
                shortfalls: vec![ShortfallJson::from(&ExcitationShortfall::RestHold {
                    seen_s: 1.5,
                    needed_s: 2.0,
                })],
            },
        };

        // Act
        let text = serde_json::to_string(&record).unwrap();
        let parsed: CalibrationJson = serde_json::from_str(&text).unwrap();

        // Assert
        assert_eq!(parsed, record);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), text);
    }

    #[test]
    fn the_rest_hold_gives_back_the_gyro_bias_it_was_generated_with() {
        // Arrange — §4: with ω ≡ 0 the per-sensor mean *is* the bias.
        let record = minimal_record();

        // Act
        let json = CalibrationJson::from(&record);
        let bias = json.sensors[0].gyro_bias_rad_s.expect("bias present");

        // Assert
        assert!((bias[0] - 0.001).abs() < 1e-12);
        assert!((bias[1] + 0.002).abs() < 1e-12);
        assert!((bias[2] - 0.003).abs() < 1e-12);
        assert_eq!(json.sensors[0].mount_origin, "gauge");
    }

    #[test]
    fn every_shortfall_serialises_with_its_numbers_its_units_and_its_sentence() {
        // Arrange
        let shortfall = ExcitationShortfall::SteerRichness { seen: 0.31, needed: 0.5 };

        // Act
        let json = ShortfallJson::from(&shortfall);

        // Assert
        assert_eq!(json.metric, "steer_richness");
        assert_eq!(json.units.as_deref(), Some("rad"));
        assert_eq!(json.seen, Some(0.31));
        assert_eq!(json.needed, Some(0.5));
        assert!(json.message.contains("bars did not turn far enough"), "{}", json.message);
    }

    #[test]
    fn a_structural_shortfall_carries_a_sentence_but_no_measurement() {
        // Arrange — "there is no front sensor" is not a number.
        let shortfall = ExcitationShortfall::NoFrontSensor;

        // Act
        let json = ShortfallJson::from(&shortfall);

        // Assert
        assert_eq!(json.metric, "no_front_sensor");
        assert_eq!(json.seen, None);
        assert_eq!(json.units, None);
        assert!(!json.message.is_empty());
    }
}
