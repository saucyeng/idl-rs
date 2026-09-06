//! Predicts the SPEC §5.2 fixed-ID subset of a device's channel registry
//! from `idl0_config.json` alone, with no device I/O (C3 §3.8
//! `preview_channel_registry`, ruling R63 (2)).
//!
//! **Scope limit (R63 (2) / decisions.md R64.5):** SPEC §5.2's "Current
//! channels in registry at v3 launch" table fixes an exact `channel_id`
//! only for the 18 IMU axes, the 2 wheel-speed counters, the 2 pressure
//! channels, and the 2 heart-rate channels. Every other `analog.channels[]`/
//! `digital.channels[]` entry has no fixed `channel_id` — SPEC §5.2's own
//! "24+ anything, TBD" row says so — so this module emits no row for them.
//! [`preview_channel_registry`] also emits no row for the two pressure
//! channels (20/21): SPEC §8's own `idl0_config.json` schema has no
//! dedicated `pressure` block naming their scale/offset explicitly, and
//! guessing a source here would be exactly the kind of invented
//! `channel_id` this scope limit exists to avoid (open question carried to
//! Isaac, not resolved by this task).

use crate::config::VersionedConfig;

/// One predicted SPEC §5.2 registry row, derived from config alone (no
/// device I/O). Shares field *names* with
/// [`crate::session::ChannelRegistryEntry`] (the *wire* registry read off a
/// recorded file) but is a distinct type with different field types on the
/// wire (`channel_id: u8`, `data_type: u8` code there vs. `u16`/`&'static str`
/// here, matching C3 §3.8's `RegistryRow` interface instead): this row is a
/// *prediction*, never read from or written to a session.
pub struct RegistryPreviewRow {
    /// Matches SPEC §5.2's `channel_id` column for this row's channel.
    pub channel_id: u16,
    /// One of `"i16"`, `"i32"`, `"u8"`, `"u16"`, `"u32"` — SPEC §5.2's Type
    /// column, restricted to the values this fixed subset actually uses.
    pub data_type: &'static str,
    /// Nominal sample rate, Hz. `0.0` marks an event-driven channel (SPEC
    /// §5.2's "0 = event-driven").
    pub sample_rate_hz: f64,
    /// `physical = stored × scale + offset` (SPEC §5.2).
    pub scale: f64,
    /// Added after scaling (SPEC §5.2).
    pub offset: f64,
    /// SPEC §5.2's Name column, e.g. `"IMU0_AccelX"`, `"WheelFront"`.
    pub name: String,
    /// SPEC §5.2's Units column, e.g. `"g"`, `"dps"`, `"pulse"`, `"bpm"`.
    pub units: String,
}

/// The relevant subset of SPEC §8's `idl0_config.json` schema this
/// derivation needs. Does not round-trip or validate the whole document —
/// that is the TS validator's job (SPEC §8, R53 Device Q1).
#[derive(Debug, serde::Deserialize)]
pub struct DeviceConfig {
    /// SPEC §8 `config_version`. Compared against [`VersionedConfig::SUPPORTED_VERSION`].
    pub config_version: u32,
    /// SPEC §8 `imu` block.
    pub imu: ImuConfig,
    /// SPEC §8 `wheel_speed` block. Absent is equivalent to both slots disabled.
    pub wheel_speed: Option<WheelSpeedConfig>,
    /// SPEC §8 `heart_rate_monitor` block. Absent is equivalent to `enabled: false` (SPEC §8).
    pub heart_rate_monitor: Option<HrmConfig>,
}

/// SPEC §8 `imu` block's fields relevant to registry-row derivation.
#[derive(Debug, serde::Deserialize)]
pub struct ImuConfig {
    /// Shared IMU sample rate, Hz — SPEC §8's single `imu.sample_rate_hz`
    /// (no per-IMU override exists in the schema). This is what SPEC §5.2's
    /// "(configured)" Rate-column notation for the 18 IMU axis rows names.
    pub sample_rate_hz: f64,
    /// Top-level default accelerometer range, g. Overridden per-IMU when an `imuN.accel_range_g` is present.
    pub accel_range_g: f64,
    /// Top-level default gyroscope range, deg/s. Overridden per-IMU when an `imuN.gyro_range_dps` is present.
    pub gyro_range_dps: f64,
    /// `imu0` sub-block. Entirely absent means IMU0 is not recorded (SPEC §8's "presence is derived from `imu.imuN.enabled`").
    pub imu0: Option<ImuSubConfig>,
    /// `imu1` sub-block. Entirely absent means IMU1 is not recorded.
    pub imu1: Option<ImuSubConfig>,
    /// `imu2` sub-block. Entirely absent means IMU2 is not recorded.
    pub imu2: Option<ImuSubConfig>,
}

/// One `imuN` sub-block (SPEC §8 "Per-IMU range resolution").
#[derive(Debug, serde::Deserialize)]
pub struct ImuSubConfig {
    /// Whether this IMU is enabled. Not used directly for row emission — an
    /// absent sub-block (not merely `enabled: false`) is what suppresses
    /// this IMU's rows, per SPEC §8's own presence rule; kept here only
    /// because SPEC §8's schema names it.
    #[serde(default)]
    pub enabled: bool,
    /// This IMU's accelerometer range override, g. Falls back to `ImuConfig::accel_range_g` when absent.
    pub accel_range_g: Option<f64>,
    /// This IMU's gyroscope range override, deg/s. Falls back to `ImuConfig::gyro_range_dps` when absent.
    pub gyro_range_dps: Option<f64>,
    /// Per-axis enable flags (SPEC §8 `imuN.channels`).
    pub channels: ImuChannelsConfig,
}

/// SPEC §8 `imuN.channels` — one enable flag per IMU axis.
#[derive(Debug, serde::Deserialize)]
pub struct ImuChannelsConfig {
    /// Enables this IMU's `AccelX` row.
    pub accel_x: bool,
    /// Enables this IMU's `AccelY` row.
    pub accel_y: bool,
    /// Enables this IMU's `AccelZ` row.
    pub accel_z: bool,
    /// Enables this IMU's `GyroX` row.
    pub gyro_x: bool,
    /// Enables this IMU's `GyroY` row.
    pub gyro_y: bool,
    /// Enables this IMU's `GyroZ` row.
    pub gyro_z: bool,
}

/// SPEC §8 `wheel_speed` block.
#[derive(Debug, serde::Deserialize)]
pub struct WheelSpeedConfig {
    /// Front wheel-speed slot (channel 18, `WheelFront`).
    pub front: WheelSlot,
    /// Rear wheel-speed slot (channel 19, `WheelRear`).
    pub rear: WheelSlot,
}

/// One wheel-speed slot (SPEC §8 `wheel_speed.front`/`wheel_speed.rear`).
#[derive(Debug, serde::Deserialize)]
pub struct WheelSlot {
    /// Whether this wheel-speed counter is enabled. Defaults to `false` (SPEC §8 "Wheel speed defaults").
    #[serde(default)]
    pub enabled: bool,
}

/// SPEC §8 `heart_rate_monitor` block.
#[derive(Debug, serde::Deserialize)]
pub struct HrmConfig {
    /// Whether an HR monitor is paired and active. Defaults to `false` (SPEC §8).
    #[serde(default)]
    pub enabled: bool,
}

impl VersionedConfig for DeviceConfig {
    const SUPPORTED_VERSION: u32 = 1;
    const LABEL: &'static str = "device config";
    fn version(&self) -> u32 {
        self.config_version
    }
}

/// One IMU's 6 axes, in SPEC §5.2's fixed per-IMU order.
const IMU_AXIS_NAMES: [&str; 6] = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"];

/// Appends `imu_slot`'s enabled-axis rows (if present) at `base_id` (SPEC
/// §5.2's IMU0 = 0, IMU1 = 6, IMU2 = 12) into `out`. Applies per-IMU range
/// resolution (SPEC §8 "Per-IMU range resolution"): `imu_slot`'s own
/// `accel_range_g`/`gyro_range_dps` when present, else `imu`'s top-level
/// default. An absent `imu_slot` contributes no rows at all.
fn push_imu_rows(out: &mut Vec<RegistryPreviewRow>, imu: &ImuConfig, imu_slot: &Option<ImuSubConfig>, base_id: u16, imu_index: u8) {
    let Some(slot) = imu_slot else { return };
    let accel_range_g = slot.accel_range_g.unwrap_or(imu.accel_range_g);
    let gyro_range_dps = slot.gyro_range_dps.unwrap_or(imu.gyro_range_dps);
    let accel_scale = accel_range_g / 32768.0;
    let gyro_scale = gyro_range_dps / 32768.0;

    let axis_enabled = [
        slot.channels.accel_x,
        slot.channels.accel_y,
        slot.channels.accel_z,
        slot.channels.gyro_x,
        slot.channels.gyro_y,
        slot.channels.gyro_z,
    ];
    for (i, name) in IMU_AXIS_NAMES.iter().enumerate() {
        if !axis_enabled[i] {
            continue;
        }
        let is_accel = i < 3;
        out.push(RegistryPreviewRow {
            channel_id: base_id + i as u16,
            data_type: "i16",
            // SPEC §5.2's Rate column for IMU axes is "(configured)" —
            // `imu.sample_rate_hz`, shared across all three IMUs (SPEC §8
            // has no per-IMU rate override).
            sample_rate_hz: imu.sample_rate_hz,
            scale: if is_accel { accel_scale } else { gyro_scale },
            offset: 0.0,
            name: format!("IMU{imu_index}_{name}"),
            units: if is_accel { "g".to_string() } else { "dps".to_string() },
        });
    }
}

/// Predicts the SPEC §5.2 fixed-ID subset of the channel registry a device
/// would produce recording under `config`: the 18 IMU axes, 2 wheel
/// counters, 2 pressure channels (only if their scale/offset are named
/// explicitly in `config` — see this module's own scope-limit doc note,
/// currently never, since SPEC §8's schema has no such field), and 2 HR
/// channels. Emits **no row** for any `analog.channels[]`/
/// `digital.channels[]` entry beyond those two named pressure slots — SPEC
/// §5.2 itself does not fix a `channel_id` scheme for them (its own "24+
/// anything, TBD" row); inventing one here would be a guess this function
/// deliberately declines to make (ruling R63 (2)).
pub fn preview_channel_registry(config: &DeviceConfig) -> Vec<RegistryPreviewRow> {
    let mut rows = Vec::new();

    push_imu_rows(&mut rows, &config.imu, &config.imu.imu0, 0, 0);
    push_imu_rows(&mut rows, &config.imu, &config.imu.imu1, 6, 1);
    push_imu_rows(&mut rows, &config.imu, &config.imu.imu2, 12, 2);

    if let Some(wheel) = &config.wheel_speed {
        if wheel.front.enabled {
            rows.push(RegistryPreviewRow {
                channel_id: 18,
                data_type: "u32",
                sample_rate_hz: 0.0,
                scale: 1.0,
                offset: 0.0,
                name: "WheelFront".to_string(),
                units: "pulse".to_string(),
            });
        }
        if wheel.rear.enabled {
            rows.push(RegistryPreviewRow {
                channel_id: 19,
                data_type: "u32",
                sample_rate_hz: 0.0,
                scale: 1.0,
                offset: 0.0,
                name: "WheelRear".to_string(),
                units: "pulse".to_string(),
            });
        }
    }

    // Channels 20/21 (PressureFront/PressureRear): SPEC §8's schema has no
    // field naming their scale/offset explicitly (no dedicated `pressure`
    // block; `analog.channels[]` is generic and unnamed by SPEC §5.2's
    // fixed-ID table) — no row emitted, per this module's scope limit.

    if let Some(hrm) = &config.heart_rate_monitor {
        if hrm.enabled {
            rows.push(RegistryPreviewRow {
                channel_id: 22,
                data_type: "u8",
                sample_rate_hz: 1.0,
                scale: 1.0,
                offset: 0.0,
                name: "HR_BPM".to_string(),
                units: "bpm".to_string(),
            });
            rows.push(RegistryPreviewRow {
                channel_id: 23,
                data_type: "u16",
                sample_rate_hz: 0.0,
                scale: 1000.0 / 1024.0,
                offset: 0.0,
                name: "HR_RR".to_string(),
                units: "ms".to_string(),
            });
        }
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{parse_config, ConfigErrorKind};

    /// Builds a minimal valid config JSON string with all three IMUs
    /// enabled at the top-level default range, all six axes on per IMU, and
    /// `wheel_speed`/`heart_rate_monitor` omitted.
    fn base_config_json() -> String {
        r#"{
            "config_version": 1,
            "imu": {
                "sample_rate_hz": 833,
                "accel_range_g": 32,
                "gyro_range_dps": 2000,
                "imu0": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu1": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu2": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } }
            }
        }"#
        .to_string()
    }

    #[test]
    fn preview_channel_registry_all_axes_enabled_produces_18_imu_rows_with_per_imu_range_override() {
        // Arrange — imu1's accel_range_g (16) differs from the top-level default (32).
        let json = r#"{
            "config_version": 1,
            "imu": {
                "sample_rate_hz": 833,
                "accel_range_g": 32,
                "gyro_range_dps": 2000,
                "imu0": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu1": { "enabled": true, "accel_range_g": 16, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu2": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } }
            }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        let imu_rows: Vec<_> = rows.iter().filter(|r| r.channel_id < 18).collect();
        assert_eq!(imu_rows.len(), 18);
        let imu1_accel_x = rows.iter().find(|r| r.channel_id == 6).unwrap();
        assert_eq!(imu1_accel_x.scale, 16.0 / 32768.0);
        let imu0_accel_x = rows.iter().find(|r| r.channel_id == 0).unwrap();
        assert_eq!(imu0_accel_x.scale, 32.0 / 32768.0);
    }

    #[test]
    fn preview_channel_registry_absent_imu_sub_block_emits_no_rows_for_that_imu() {
        // Arrange
        let json = r#"{
            "config_version": 1,
            "imu": {
                "sample_rate_hz": 833,
                "accel_range_g": 32,
                "gyro_range_dps": 2000,
                "imu0": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu2": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } }
            }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        assert_eq!(rows.iter().filter(|r| r.channel_id < 18).count(), 12);
        assert!(rows.iter().all(|r| !r.name.starts_with("IMU1_")));
    }

    #[test]
    fn preview_channel_registry_disabled_single_axis_omits_only_that_row() {
        // Arrange
        let json = r#"{
            "config_version": 1,
            "imu": {
                "sample_rate_hz": 833,
                "accel_range_g": 32,
                "gyro_range_dps": 2000,
                "imu0": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": false } },
                "imu1": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } },
                "imu2": { "enabled": true, "channels": { "accel_x": true, "accel_y": true, "accel_z": true, "gyro_x": true, "gyro_y": true, "gyro_z": true } }
            }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        assert_eq!(rows.iter().filter(|r| r.channel_id < 18).count(), 17);
        assert!(!rows.iter().any(|r| r.name == "IMU0_GyroZ"));
    }

    #[test]
    fn preview_channel_registry_wheel_and_hr_rows_present_only_per_enabled_flags() {
        // Arrange
        let json = r#"{
            "config_version": 1,
            "imu": { "sample_rate_hz": 833, "accel_range_g": 32, "gyro_range_dps": 2000 },
            "wheel_speed": { "front": { "enabled": true }, "rear": { "enabled": false } },
            "heart_rate_monitor": { "enabled": true }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        assert!(rows.iter().any(|r| r.channel_id == 18 && r.name == "WheelFront"));
        assert!(!rows.iter().any(|r| r.channel_id == 19));
        assert!(rows.iter().any(|r| r.channel_id == 22 && r.name == "HR_BPM"));
        assert!(rows.iter().any(|r| r.channel_id == 23 && r.name == "HR_RR"));
    }

    #[test]
    fn preview_channel_registry_absent_wheel_and_hr_blocks_emit_no_rows_for_18_19_22_23() {
        // Arrange
        let config: DeviceConfig = parse_config(base_config_json().as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        assert!(!rows.iter().any(|r| matches!(r.channel_id, 18 | 19 | 22 | 23)));
    }

    #[test]
    fn preview_channel_registry_hr_rr_scale_is_spec_exact_value() {
        // Arrange
        let json = r#"{
            "config_version": 1,
            "imu": { "sample_rate_hz": 833, "accel_range_g": 32, "gyro_range_dps": 2000 },
            "heart_rate_monitor": { "enabled": true }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        let hr_rr = rows.iter().find(|r| r.channel_id == 23).unwrap();
        assert_eq!(hr_rr.scale, 1000.0 / 1024.0);
    }

    #[test]
    fn preview_channel_registry_never_emits_pressure_rows_absent_an_explicit_config_source() {
        // Arrange — only wheel_speed/heart_rate_monitor set, no config field
        // names channels 20/21's scale/offset (expected: none exists).
        let json = r#"{
            "config_version": 1,
            "imu": { "sample_rate_hz": 833, "accel_range_g": 32, "gyro_range_dps": 2000 },
            "wheel_speed": { "front": { "enabled": true }, "rear": { "enabled": true } },
            "heart_rate_monitor": { "enabled": true }
        }"#;
        let config: DeviceConfig = parse_config(json.as_bytes()).unwrap();

        // Act
        let rows = preview_channel_registry(&config);

        // Assert
        assert!(!rows.iter().any(|r| matches!(r.channel_id, 20 | 21)));
    }

    #[test]
    fn preview_channel_registry_malformed_json_returns_parse_kind() {
        // Act
        let err = parse_config::<DeviceConfig>(b"not json").unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::Parse);
    }

    #[test]
    fn preview_channel_registry_unsupported_version_returns_unsupported_version_kind() {
        // Arrange
        let json = r#"{
            "config_version": 99,
            "imu": { "sample_rate_hz": 833, "accel_range_g": 32, "gyro_range_dps": 2000 }
        }"#;

        // Act
        let err = parse_config::<DeviceConfig>(json.as_bytes()).unwrap_err();

        // Assert
        assert_eq!(err.kind, ConfigErrorKind::UnsupportedVersion);
    }
}
