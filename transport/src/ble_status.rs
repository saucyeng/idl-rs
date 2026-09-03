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

/// IMU health state, decoded from the status block's `IMU:` line (SPEC §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuState {
    /// All configured IMU sensors are reporting.
    Ok,
    /// Some but not all configured IMU sensors are reporting.
    Partial,
    /// IMU present but reporting a fault.
    Error,
    /// No IMU detected.
    Absent,
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

fn parse_imu_state(v: &str) -> Option<ImuState> {
    match v.to_ascii_uppercase().as_str() {
        "OK" => Some(ImuState::Ok),
        "PARTIAL" => Some(ImuState::Partial),
        "ERROR" => Some(ImuState::Error),
        "ABSENT" => Some(ImuState::Absent),
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
    fn parse_status_empty_block_returns_default() {
        // Arrange
        let text = "";

        // Act
        let status = parse_status(text);

        // Assert
        assert_eq!(status, DeviceStatus::default());
    }
}
