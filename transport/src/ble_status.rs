//! SPEC §7.3 status-characteristic parsing. UTF-8, newline-delimited,
//! case-insensitive keys, unknown lines ignored so the set may grow without
//! breaking older parsers. Never fails — a block with no recognised lines
//! yields `DeviceStatus::default()` (CLAUDE.md §5: never a crash on bad data).

/// One snapshot of device status, decoded from either the FF04 notify
/// payload (idle/recording modes) or `/ping`'s JSON body flattened to the
/// same line shape by the caller (SPEC §6.1, §7.3) — both carry the same
/// field set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceStatus {
    /// `true` when the WiFi radio is on (`WiFi: ON` line), `false` for `OFF`.
    pub wifi_on: Option<bool>,
    /// `true` while a recording session is active (`Logging: RUNNING` line).
    pub logging: Option<bool>,
    /// Main battery charge, percent (`Battery: NN%` line).
    pub battery_pct: Option<u8>,
    /// SD card state (`SD:` line).
    pub sd: Option<SdState>,
    /// GPS fix state (`GPS:` line).
    pub gps: Option<GpsState>,
    /// IMU health state (`IMU:` line).
    pub imu: Option<ImuState>,
    /// Running image's `esp_app_desc_t.version`, e.g. `"1.5.0"`.
    pub firmware: Option<String>,
    /// `true` only while the `OTA: PENDING_VERIFY` line is present.
    pub ota_pending_verify: bool,
    /// Raw HR line value (`ABSENT`, `SEARCHING`, `"CONNECTED 132"`, …) — kept
    /// as a string pending a typed enum decision, see Open question 4.
    pub hr: Option<String>,
    /// Heart-rate strap battery, percent (`HR_Battery: NN%` line).
    pub hr_battery_pct: Option<u8>,
    /// Seconds elapsed since the current logging session started
    /// (`LoggingElapsed: N` line), device-monotonic (`esp_timer`). Present
    /// only while `logging` is `Some(true)` (SPEC §7.3, R113); the device
    /// clock is not wall-clock-anchored until first GPS fix, so the app uses
    /// this elapsed counter rather than a start timestamp. `None` when the
    /// line is absent — the caller falls back to client-observed elapsed
    /// time, rendered dimmed.
    pub logging_elapsed_s: Option<u32>,
    /// Unscaled main battery ADC count (`BatteryRaw: N` line) — whatever the
    /// pin/ADC returns, no units. `Battery: NN%` is deprecated (SPEC §7.3):
    /// the device cannot know pack chemistry or board revision, so scaling
    /// to a percentage is the app's job, per board revision. `None` only
    /// while the field is unreported by older firmware — SPEC §7.3 says this
    /// line is otherwise always present.
    pub battery_raw: Option<u32>,
    /// Free space on the mounted SD card, MiB (`SDFreeMiB: N` line).
    /// Present when `sd` is `Ok` or `Full`; `None` for `Error`/`Absent` or
    /// when unreported.
    pub sd_free_mib: Option<u32>,
    /// Raw NMEA GGA fix-quality field (`GPSFix: N` line): 0 none, 1 GPS,
    /// 2 DGPS, 3 PPS or better. Present whenever `gps` is not `Absent`;
    /// `None` when unreported.
    pub gps_fix_quality: Option<u8>,
    /// Satellites used in the GPS solution (`GPSSats: N` line, GGA field 7).
    /// Present under the same condition as `gps_fix_quality`; `0` is a valid
    /// reported value ("searching"), distinct from `None` ("unknown" — the
    /// line was absent entirely).
    pub gps_sats: Option<u32>,
    /// HDOP × 100 (`GPSHDOP: N` line, GGA field 8 scaled — e.g. 0.9 → 90).
    /// Present when `gps_fix_quality` is >= 1; optional even then (SPEC
    /// §7.3: "optional if the GPS driver does not expose it cheaply"), so
    /// `None` here is always "unknown", never a real HDOP of zero.
    pub gps_hdop_x100: Option<u32>,
    /// Per-IMU state for IMU index 0 (`IMU0:` line, SPEC §7.3).
    pub imu0: Option<ImuState>,
    /// Per-IMU state for IMU index 1 (`IMU1:` line, SPEC §7.3).
    pub imu1: Option<ImuState>,
    /// Per-IMU state for IMU index 2 (`IMU2:` line, SPEC §7.3).
    pub imu2: Option<ImuState>,
}

/// SD card state, decoded from the status block's `SD:` line (SPEC §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdState {
    /// Card present and writable.
    Ok,
    /// Card present but has no free space left.
    Full,
    /// Card present but unreadable/unwritable.
    Error,
    /// No card inserted.
    Absent,
}

/// GPS fix state, decoded from the status block's `GPS:` line (SPEC §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpsState {
    /// GPS module has a valid position fix.
    Fix,
    /// GPS module is powered but has not yet acquired a fix.
    NoFix,
    /// No GPS module detected.
    Absent,
}

/// IMU health state, decoded from the status block's `IMU:` line (aggregate)
/// or an `IMU0:`/`IMU1:`/`IMU2:` line (per-sensor) — SPEC §7.3. `Off` is a
/// per-sensor-only value (disabled in the loaded config); the aggregate
/// `IMU:` line never reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuState {
    /// All configured IMU sensors are reporting.
    Ok,
    /// Some but not all configured IMU sensors are reporting.
    Partial,
    /// IMU present but reporting a fault (per-sensor: detected but failing
    /// reads/FIFO).
    Error,
    /// No IMU detected (per-sensor: enabled in config but not found on the bus).
    Absent,
    /// Per-sensor only: disabled in the loaded config.
    Off,
}

/// Parses a §7.3-shaped status block. Unknown lines are ignored; a
/// malformed value for a known key (e.g. `Battery: NN%` with non-numeric
/// `NN`) leaves that field `None` rather than failing the whole parse.
pub fn parse_status(text: &str) -> DeviceStatus {
    let mut status = DeviceStatus::default();
    for line in text.lines() {
        let Some((key, value)) = line.trim().split_once(':') else { continue };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        match key.as_str() {
            "wifi" => status.wifi_on = Some(value.eq_ignore_ascii_case("on")),
            "logging" => status.logging = Some(value.eq_ignore_ascii_case("running")),
            "battery" => status.battery_pct = value.trim_end_matches('%').parse().ok(),
            "sd" => status.sd = parse_sd_state(value),
            "gps" => status.gps = parse_gps_state(value),
            "imu" => status.imu = parse_imu_state(value),
            "firmware" => status.firmware = Some(value.to_string()),
            "ota" => status.ota_pending_verify = value.eq_ignore_ascii_case("pending_verify"),
            "hr" => status.hr = Some(value.to_string()),
            "hr_battery" => status.hr_battery_pct = value.trim_end_matches('%').parse().ok(),
            "loggingelapsed" => status.logging_elapsed_s = value.parse().ok(),
            "batteryraw" => status.battery_raw = value.parse().ok(),
            "sdfreemib" => status.sd_free_mib = value.parse().ok(),
            "gpsfix" => status.gps_fix_quality = value.parse().ok(),
            "gpssats" => status.gps_sats = value.parse().ok(),
            "gpshdop" => status.gps_hdop_x100 = value.parse().ok(),
            "imu0" => status.imu0 = parse_imu_state(value),
            "imu1" => status.imu1 = parse_imu_state(value),
            "imu2" => status.imu2 = parse_imu_state(value),
            _ => {} // unknown line — SPEC §7.3, ignored so the set may grow
        }
    }
    status
}

fn parse_sd_state(v: &str) -> Option<SdState> {
    match v.to_ascii_uppercase().as_str() {
        "OK" => Some(SdState::Ok),
        "FULL" => Some(SdState::Full),
        "ERROR" => Some(SdState::Error),
        "ABSENT" => Some(SdState::Absent),
        _ => None,
    }
}

fn parse_gps_state(v: &str) -> Option<GpsState> {
    match v.to_ascii_uppercase().as_str() {
        "FIX" => Some(GpsState::Fix),
        "NOFIX" => Some(GpsState::NoFix),
        "ABSENT" => Some(GpsState::Absent),
        _ => None,
    }
}

/// Parses an `IMU:`/`IMU0:`/`IMU1:`/`IMU2:` line value. `OFF` only appears on
/// the per-sensor lines in practice (SPEC §7.3), but this parser accepts it
/// wherever it's seen rather than special-casing which key it came from.
fn parse_imu_state(v: &str) -> Option<ImuState> {
    match v.to_ascii_uppercase().as_str() {
        "OK" => Some(ImuState::Ok),
        "PARTIAL" => Some(ImuState::Partial),
        "ERROR" => Some(ImuState::Error),
        "ABSENT" => Some(ImuState::Absent),
        "OFF" => Some(ImuState::Off),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_status_full_block_recording_mode_reads_every_field() {
        // Arrange
        let text = "WiFi: OFF\nLogging: RUNNING\nBattery: 87%\nSD: OK\nGPS: FIX\nIMU: PARTIAL\n\
                    Firmware: 1.4.0\nHR: CONNECTED 132\nHR_Battery: 91%";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.wifi_on, Some(false));
        assert_eq!(status.logging, Some(true));
        assert_eq!(status.battery_pct, Some(87));
        assert_eq!(status.sd, Some(SdState::Ok));
        assert_eq!(status.gps, Some(GpsState::Fix));
        assert_eq!(status.imu, Some(ImuState::Partial));
        assert_eq!(status.firmware, Some("1.4.0".to_string()));
        assert_eq!(status.hr, Some("CONNECTED 132".to_string()));
        assert_eq!(status.hr_battery_pct, Some(91));
        assert!(!status.ota_pending_verify);
    }

    #[test]
    fn parse_status_ota_pending_verify_line_present_sets_flag() {
        // Arrange
        let text = "WiFi: OFF\nOTA: PENDING_VERIFY";

        // Act
        let status = parse_status(text);

        // Assert
        assert!(status.ota_pending_verify);
    }

    #[test]
    fn parse_status_unknown_line_ignored_known_fields_still_parse() {
        // Arrange
        let text = "WiFi: ON\nSomeFutureField: 42\nBattery: 50%";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.wifi_on, Some(true));
        assert_eq!(status.battery_pct, Some(50));
    }

    #[test]
    fn parse_status_lowercase_keys_and_values_parse_case_insensitively() {
        // Arrange
        let text = "wifi: on\nsd: ok\ngps: nofix";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.wifi_on, Some(true));
        assert_eq!(status.sd, Some(SdState::Ok));
        assert_eq!(status.gps, Some(GpsState::NoFix));
    }

    #[test]
    fn parse_status_additive_fields_present_all_parse() {
        // Arrange
        let text = "Logging: RUNNING\nLoggingElapsed: 305\nBatteryRaw: 2731\n\
                    SD: OK\nSDFreeMiB: 14208\nGPS: FIX\nGPSFix: 1\nGPSSats: 7\n\
                    GPSHDOP: 90\nIMU0: OK\nIMU1: ERROR\nIMU2: OFF";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.logging_elapsed_s, Some(305));
        assert_eq!(status.battery_raw, Some(2731));
        assert_eq!(status.sd_free_mib, Some(14208));
        assert_eq!(status.gps_fix_quality, Some(1));
        assert_eq!(status.gps_sats, Some(7));
        assert_eq!(status.gps_hdop_x100, Some(90));
        assert_eq!(status.imu0, Some(ImuState::Ok));
        assert_eq!(status.imu1, Some(ImuState::Error));
        assert_eq!(status.imu2, Some(ImuState::Off));
    }

    #[test]
    fn parse_status_additive_fields_absent_are_none_never_zero() {
        // Arrange: none of the additive lines are present at all.
        let text = "WiFi: ON\nLogging: STOPPED\nSD: ABSENT\nGPS: ABSENT";

        // Act
        let status = parse_status(text);

        // Assert: absent means "unknown", so every additive field is `None`
        // — not a zero/false default that would be mistaken for a real
        // reading (e.g. `GPSSats: 0` means "searching", not "unreported").
        assert_eq!(status.logging_elapsed_s, None);
        assert_eq!(status.battery_raw, None);
        assert_eq!(status.sd_free_mib, None);
        assert_eq!(status.gps_fix_quality, None);
        assert_eq!(status.gps_sats, None);
        assert_eq!(status.gps_hdop_x100, None);
        assert_eq!(status.imu0, None);
        assert_eq!(status.imu1, None);
        assert_eq!(status.imu2, None);
    }

    #[test]
    fn parse_status_logging_elapsed_absent_while_stopped_is_normal() {
        // Arrange: R113 — LoggingElapsed is present only while RUNNING.
        let text = "Logging: STOPPED";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.logging, Some(false));
        assert_eq!(status.logging_elapsed_s, None);
    }

    #[test]
    fn parse_status_gps_sats_zero_while_searching_is_distinct_from_absent() {
        // Arrange: `GPSSats: 0` is a real reported value ("searching"), not
        // the same as the line being missing entirely.
        let text = "GPS: NOFIX\nGPSFix: 0\nGPSSats: 0";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.gps_sats, Some(0));
        assert_ne!(status.gps_sats, None);
    }

    #[test]
    fn parse_status_malformed_additive_line_loses_only_that_field() {
        // Arrange: a non-numeric value for a known additive key.
        let text = "BatteryRaw: not-a-number\nSDFreeMiB: 500";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status.battery_raw, None);
        assert_eq!(status.sd_free_mib, Some(500));
    }

    #[test]
    fn parse_status_empty_block_returns_default() {
        // Arrange
        let text = "";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status, DeviceStatus::default());
    }
}
