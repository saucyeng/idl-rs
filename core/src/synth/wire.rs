//! `.idl0` schema-3 byte emission for the synthetic generator (contract C1
//! §9.5, `docs/IDL0_SPEC.md` §5).
//!
//! This is the *only* writer of `.idl0` bytes in the engine — the device is
//! the other one, and it is firmware. Everything here mirrors SPEC §5 field
//! for field and in SPEC's order; where a value is a fixed literal rather
//! than a generated one, the literal is named in C1 §9.5 too, so the two can
//! be diffed.
//!
//! Pure: builds and returns a `Vec<u8>`, touches no filesystem.

/// `docs/IDL0_SPEC.md` §5.1's fixed header prefix length, before the channel
/// registry: magic 4 + schema 1 + uuid 16 + device id 6 + start ms 8 +
/// config crc 4 + imu mask 4 + imu count 1 + imu rate 2 + gps rate 1 +
/// registry count 1.
pub const HEADER_PREFIX_LEN: usize = 48;

/// SPEC §5.2's registry entry width.
pub const REGISTRY_ENTRY_LEN: usize = 40;

/// SPEC §5.1's header end marker, little-endian.
pub const END_MARKER: u32 = 0xDEAD_BEEF;

/// Record type tags (SPEC §5.3).
pub const IMU_SAMPLE: u8 = 0x01;
/// See [`IMU_SAMPLE`].
pub const GPS_FIX: u8 = 0x02;
/// See [`IMU_SAMPLE`].
pub const SESSION_END: u8 = 0xFF;

/// SPEC §5.2's `data_type` code for `i16`.
pub const DATA_TYPE_I16: u8 = 4;

/// The header fields SPEC §5.1 fixes, ready to serialise.
pub struct Header {
    /// 16 raw bytes; the app renders them as 32 lowercase hex characters.
    pub uuid: [u8; 16],
    /// 6 raw bytes; the app renders them as 12 lowercase hex characters.
    pub device_id: [u8; 6],
    /// Session start, UTC milliseconds since the Unix epoch.
    pub session_start_ms: i64,
    /// CRC-32 of the recording configuration.
    pub config_crc: u32,
    /// SPEC §5.4's enabled-axis mask.
    pub imu_mask: u32,
    /// Number of IMUs whose records appear in the file.
    pub imu_count: u8,
    /// IMU output data rate, Hz.
    pub imu_sample_rate_hz: u16,
    /// GPS fix rate, Hz.
    pub gps_sample_rate_hz: u8,
}

/// Writes SPEC §5.1's header: the 48-byte prefix, `registry` verbatim, then
/// the end marker.
///
/// # Panics
///
/// If `registry` holds more than 255 entries — SPEC §5.1's registry count is
/// a `u8`, and a caller that built more has a bug the file format cannot
/// express. The generator's own maximum is 18.
pub fn write_header(header: &Header, registry: &[[u8; REGISTRY_ENTRY_LEN]]) -> Vec<u8> {
    assert!(registry.len() <= u8::MAX as usize, "registry count exceeds the header's u8");

    let mut buf = Vec::with_capacity(HEADER_PREFIX_LEN + registry.len() * REGISTRY_ENTRY_LEN + 4);
    buf.extend_from_slice(b"IDL0");
    buf.push(3);
    buf.extend_from_slice(&header.uuid);
    buf.extend_from_slice(&header.device_id);
    buf.extend_from_slice(&header.session_start_ms.to_le_bytes());
    buf.extend_from_slice(&header.config_crc.to_le_bytes());
    buf.extend_from_slice(&header.imu_mask.to_le_bytes());
    buf.push(header.imu_count);
    buf.extend_from_slice(&header.imu_sample_rate_hz.to_le_bytes());
    buf.push(header.gps_sample_rate_hz);
    buf.push(registry.len() as u8);
    debug_assert_eq!(buf.len(), HEADER_PREFIX_LEN);

    for entry in registry {
        buf.extend_from_slice(entry);
    }
    buf.extend_from_slice(&END_MARKER.to_le_bytes());
    buf
}

/// Builds one SPEC §5.2 registry entry. `name` is truncated to 19 bytes and
/// `units` to 7, each leaving room for the null terminator the format
/// requires.
pub fn registry_entry(
    channel_id: u8,
    data_type: u8,
    sample_rate_hz: u16,
    scale: f32,
    offset: f32,
    name: &str,
    units: &str,
) -> [u8; REGISTRY_ENTRY_LEN] {
    let mut buf = [0u8; REGISTRY_ENTRY_LEN];
    buf[0] = channel_id;
    buf[1] = data_type;
    buf[2..4].copy_from_slice(&sample_rate_hz.to_le_bytes());
    buf[4..8].copy_from_slice(&scale.to_le_bytes());
    buf[8..12].copy_from_slice(&offset.to_le_bytes());
    write_ascii(&mut buf[12..32], name, 19);
    write_ascii(&mut buf[32..40], units, 7);
    buf
}

/// Copies up to `max` bytes of `text` into `field`, leaving the rest zero so
/// the value stays null-terminated.
fn write_ascii(field: &mut [u8], text: &str, max: usize) {
    for (i, byte) in text.as_bytes().iter().take(max).enumerate() {
        field[i] = *byte;
    }
}

/// Wraps a payload in SPEC §5.3's `[type:u8][payload_len:u16][payload]`
/// framing, appending to `out`.
///
/// # Panics
///
/// If `payload` is longer than `u16::MAX`. No record this generator emits
/// exceeds 32 bytes.
pub fn push_record(out: &mut Vec<u8>, record_type: u8, payload: &[u8]) {
    assert!(payload.len() <= u16::MAX as usize, "record payload exceeds the framing's u16");
    out.push(record_type);
    out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    out.extend_from_slice(payload);
}

/// SPEC §5.5's `IMU_SAMPLE` payload for a sensor with all six axes enabled.
pub fn imu_payload(imu_index: u8, timestamp_us: i64, axes: [i16; 6]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9 + 12);
    buf.push(imu_index);
    buf.extend_from_slice(&timestamp_us.to_le_bytes());
    for axis in axes {
        buf.extend_from_slice(&axis.to_le_bytes());
    }
    buf
}

/// SPEC §5.6's 32-byte `GPS_FIX` payload — the legacy record, without the
/// `docs/HARDWARE_M10_SETUP.md` §3 fields, which SPEC §5.6 has no room for
/// (contract C1 §9.5).
#[allow(clippy::too_many_arguments)]
pub fn gps_payload(
    gps_epoch_ms: i64,
    device_timestamp_us: i64,
    latitude_e7: i32,
    longitude_e7: i32,
    altitude_dm: i16,
    speed_kmh_e2: u16,
    heading_deg_e2: u16,
    fix_quality: u8,
    satellites: u8,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&gps_epoch_ms.to_le_bytes());
    buf.extend_from_slice(&device_timestamp_us.to_le_bytes());
    buf.extend_from_slice(&latitude_e7.to_le_bytes());
    buf.extend_from_slice(&longitude_e7.to_le_bytes());
    buf.extend_from_slice(&altitude_dm.to_le_bytes());
    buf.extend_from_slice(&speed_kmh_e2.to_le_bytes());
    buf.extend_from_slice(&heading_deg_e2.to_le_bytes());
    buf.push(fix_quality);
    buf.push(satellites);
    debug_assert_eq!(buf.len(), 32);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            uuid: [0xAB; 16],
            device_id: [0xCD; 6],
            session_start_ms: 1_767_225_600_000,
            config_crc: 0x1234_5678,
            imu_mask: 0x3F,
            imu_count: 1,
            imu_sample_rate_hz: 800,
            gps_sample_rate_hz: 5,
        }
    }

    #[test]
    fn header_prefix_is_forty_eight_bytes_before_the_registry() {
        // Arrange
        let h = header();

        // Act
        let bytes = write_header(&h, &[]);

        // Assert
        assert_eq!(bytes.len(), HEADER_PREFIX_LEN + 4);
        assert_eq!(&bytes[0..4], b"IDL0");
        assert_eq!(bytes[4], 3);
        assert_eq!(&bytes[HEADER_PREFIX_LEN..], &END_MARKER.to_le_bytes());
    }

    #[test]
    fn header_grows_by_forty_bytes_per_registry_entry() {
        // Arrange
        let h = header();
        let entry = registry_entry(0, DATA_TYPE_I16, 800, 0.001, 0.0, "IMU0_AccelX", "g");

        // Act
        let bytes = write_header(&h, &[entry, entry]);

        // Assert
        assert_eq!(bytes.len(), HEADER_PREFIX_LEN + 2 * REGISTRY_ENTRY_LEN + 4);
        assert_eq!(bytes[47], 2);
    }

    #[test]
    fn registry_entry_null_terminates_a_name_at_the_field_width() {
        // Arrange — 25 characters, longer than the 20-byte name field.
        let long = "IMU0_AccelXAndThenSomeMore";

        // Act
        let entry = registry_entry(3, DATA_TYPE_I16, 800, 1.0, 0.0, long, "dps");

        // Assert
        assert_eq!(&entry[12..31], &long.as_bytes()[..19]);
        assert_eq!(entry[31], 0);
    }

    #[test]
    fn framing_carries_the_payload_length_little_endian() {
        // Arrange
        let mut out = Vec::new();
        let payload = vec![7u8; 300];

        // Act
        push_record(&mut out, IMU_SAMPLE, &payload);

        // Assert
        assert_eq!(out[0], IMU_SAMPLE);
        assert_eq!(u16::from_le_bytes([out[1], out[2]]), 300);
        assert_eq!(out.len(), 303);
    }

    #[test]
    fn imu_payload_is_twenty_one_bytes_with_six_axes() {
        // Arrange / Act
        let payload = imu_payload(2, 4_000_000, [1, -2, 3, -4, 5, -6]);

        // Assert
        assert_eq!(payload.len(), 21);
        assert_eq!(payload[0], 2);
        assert_eq!(i64::from_le_bytes(payload[1..9].try_into().unwrap()), 4_000_000);
        assert_eq!(i16::from_le_bytes(payload[9..11].try_into().unwrap()), 1);
    }

    #[test]
    fn gps_payload_is_thirty_two_bytes() {
        // Arrange / Act
        let payload = gps_payload(1, 2, 515_000_000, -15_000_000, 1000, 1234, 9000, 1, 12);

        // Assert
        assert_eq!(payload.len(), 32);
        assert_eq!(payload[30], 1);
        assert_eq!(payload[31], 12);
    }
}
