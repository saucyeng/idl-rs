//! SPEC §7.2 control-characteristic (FF03) command bytes and the ACK
//! protocol every write to it returns.

/// Single-byte commands written to the Control characteristic (FF03),
/// Write with Response (SPEC §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlCommand {
    WifiOn = 0x01,
    WifiOff = 0x02,
    StartLogging = 0x03,
    StopLogging = 0x04,
    CalibrateImu = 0x05,
    OtaConfirm = 0x06,
    ConfigBegin = 0x07,
    ConfigCommit = 0x08,
    ConfigReadBegin = 0x09,
}

impl ControlCommand {
    /// The single byte written to FF03 for this command.
    pub fn as_byte(self) -> u8 {
        self as u8
    }
}

/// The GATT write-response ACK code every Control write returns (SPEC
/// §7.2). `0x00` means only "accepted and dispatched" — not "completed";
/// completion is the corresponding FF04 status notify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckCode {
    Success,
    /// `0x03` — mutex or precondition refusal (SPEC §7.2 "Mutex").
    WriteNotPermitted,
    /// `0x80` — reserved (SPEC §7.2 `IDL0_ACK_BUSY`).
    Busy,
    /// `0x81` — reserved (SPEC §7.2 `IDL0_ACK_PRECONDITION`).
    Precondition,
    /// `0x82` — reserved (SPEC §7.2 `IDL0_ACK_NOT_IMPLEMENTED`).
    NotImplemented,
    /// A code SPEC §7.2 doesn't document — SPEC reserves the space for
    /// firmware growth; never treated as success.
    Unknown(u8),
}

impl AckCode {
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x00 => AckCode::Success,
            0x03 => AckCode::WriteNotPermitted,
            0x80 => AckCode::Busy,
            0x81 => AckCode::Precondition,
            0x82 => AckCode::NotImplemented,
            other => AckCode::Unknown(other),
        }
    }

    /// `true` only for `0x00` — every other code, including `Unknown`, is a refusal.
    pub fn is_success(self) -> bool {
        matches!(self, AckCode::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_command_as_byte_matches_spec_table_for_every_variant() {
        // Arrange
        let commands = [
            (ControlCommand::WifiOn, 0x01),
            (ControlCommand::WifiOff, 0x02),
            (ControlCommand::StartLogging, 0x03),
            (ControlCommand::StopLogging, 0x04),
            (ControlCommand::CalibrateImu, 0x05),
            (ControlCommand::OtaConfirm, 0x06),
            (ControlCommand::ConfigBegin, 0x07),
            (ControlCommand::ConfigCommit, 0x08),
            (ControlCommand::ConfigReadBegin, 0x09),
        ];

        // Act / Assert
        for (cmd, byte) in commands {
            assert_eq!(cmd.as_byte(), byte);
        }
    }

    #[test]
    fn ack_code_from_byte_documented_codes_map_correctly() {
        // Arrange
        let codes = [
            (0x00u8, AckCode::Success),
            (0x03, AckCode::WriteNotPermitted),
            (0x80, AckCode::Busy),
            (0x81, AckCode::Precondition),
            (0x82, AckCode::NotImplemented),
        ];

        // Act / Assert
        for (byte, expected) in codes {
            assert_eq!(AckCode::from_byte(byte), expected);
        }
    }

    #[test]
    fn ack_code_from_byte_undocumented_code_is_unknown_not_success() {
        // Arrange
        let byte = 0x7f;

        // Act
        let code = AckCode::from_byte(byte);

        // Assert
        assert_eq!(code, AckCode::Unknown(0x7f));
        assert!(!code.is_success());
    }

    #[test]
    fn ack_code_is_success_true_only_for_0x00() {
        // Arrange
        let codes = [
            AckCode::Success,
            AckCode::WriteNotPermitted,
            AckCode::Busy,
            AckCode::Precondition,
            AckCode::NotImplemented,
            AckCode::Unknown(0x99),
        ];

        // Act
        let successes: Vec<bool> = codes.iter().map(|c| c.is_success()).collect();

        // Assert
        assert_eq!(successes, [true, false, false, false, false, false]);
    }
}
