//! Byte-buffer builders for parser parity tests.
//!
//! Direct port of the helpers in `app/test/data/binary_parser_test.dart` so the
//! Rust parser is exercised against byte-identical inputs and the same expected
//! outputs — the parity gate for the Dart→Rust migration.
#![allow(dead_code)]

/// 2024-01-01 12:00:00 UTC in ms since Unix epoch (the Dart `_rmcUtcMs`).
pub const RMC_UTC_MS: i64 = 1_704_110_400_000;

fn write_str(buf: &mut [u8], offset: usize, s: &str, max: usize) {
    for (i, b) in s.as_bytes().iter().enumerate() {
        if i >= max {
            break;
        }
        buf[offset + i] = *b;
    }
}

/// `[type:u8][payload_len:u16 LE][payload]` framing (shared by all versions).
pub fn frame(type_: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 + payload.len());
    out.push(type_);
    out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// 4-byte little-endian `0xDEADBEEF` header end marker.
pub fn end_marker() -> Vec<u8> {
    0xDEAD_BEEF_u32.to_le_bytes().to_vec()
}

/// SESSION_END record (type 0xFF, empty payload).
pub fn session_end() -> Vec<u8> {
    frame(0xFF, &[])
}

// ── IDL0 header ──────────────────────────────────────────────────────────────

/// IDL0 header (48 fixed bytes + registry entries + 0xDEADBEEF), used to build
/// schema-3 buffers for the v3 parser tests.
pub struct Header {
    pub schema_version: u8,
    pub uuid: Vec<u8>,
    pub device_id: Vec<u8>,
    pub session_start_ms: i64,
    pub config_crc: u32,
    pub imu_mask: u32,
    pub imu_count: u8,
    pub imu_sample_rate_hz: u16,
    pub gps_sample_rate_hz: u8,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            schema_version: 3,
            uuid: vec![0xAB; 16],
            device_id: vec![0xCD; 6],
            session_start_ms: RMC_UTC_MS,
            config_crc: 0xABCD_1234,
            imu_mask: 0x3F,
            imu_count: 1,
            imu_sample_rate_hz: 800,
            gps_sample_rate_hz: 5,
        }
    }
}

impl Header {
    pub fn build(&self, registry: &[Vec<u8>]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"IDL0");
        buf.push(self.schema_version);
        buf.extend_from_slice(&self.uuid);
        buf.extend_from_slice(&self.device_id);
        buf.extend_from_slice(&self.session_start_ms.to_le_bytes());
        buf.extend_from_slice(&self.config_crc.to_le_bytes());
        buf.extend_from_slice(&self.imu_mask.to_le_bytes());
        buf.push(self.imu_count);
        buf.extend_from_slice(&self.imu_sample_rate_hz.to_le_bytes());
        buf.push(self.gps_sample_rate_hz);
        buf.push(registry.len() as u8);
        debug_assert_eq!(buf.len(), 48);
        for entry in registry {
            buf.extend_from_slice(entry);
        }
        buf.extend_from_slice(&end_marker());
        buf
    }
}

// ── record payloads ──────────────────────────────────────────────────────────

/// IMU_SAMPLE payload: `[imu_index:u8][ts_us:i64][axis i16 ...]`.
pub fn imu_payload(imu_index: u8, ts_us: i64, axes: &[i16]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9 + axes.len() * 2);
    buf.push(imu_index);
    buf.extend_from_slice(&ts_us.to_le_bytes());
    for a in axes {
        buf.extend_from_slice(&a.to_le_bytes());
    }
    buf
}

/// GPS_FIX payload (32 bytes). See §5.6.
#[allow(clippy::too_many_arguments)]
pub fn gps_payload(
    gps_epoch_ms: i64,
    device_ts_us: i64,
    latitude: i32,
    longitude: i32,
    altitude: i16,
    speed: u16,
    heading: u16,
    fix_quality: u8,
    satellites: u8,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&gps_epoch_ms.to_le_bytes());
    buf.extend_from_slice(&device_ts_us.to_le_bytes());
    buf.extend_from_slice(&latitude.to_le_bytes());
    buf.extend_from_slice(&longitude.to_le_bytes());
    buf.extend_from_slice(&altitude.to_le_bytes());
    buf.extend_from_slice(&speed.to_le_bytes());
    buf.extend_from_slice(&heading.to_le_bytes());
    buf.push(fix_quality);
    buf.push(satellites);
    buf
}

/// CHANNEL_SAMPLE payload with a u32 value: `[id:u8][ts_us:i64][value:u32]`.
pub fn channel_payload_u32(channel_id: u8, ts_us: i64, value: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(13);
    buf.push(channel_id);
    buf.extend_from_slice(&ts_us.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    buf
}

/// CHANNEL_SAMPLE payload with an i16 value: `[id:u8][ts_us:i64][value:i16]`.
pub fn channel_payload_i16(channel_id: u8, ts_us: i64, value: i16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11);
    buf.push(channel_id);
    buf.extend_from_slice(&ts_us.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    buf
}

/// CHANNEL_SAMPLE payload with a u16 value: `[id:u8][ts_us:i64][value:u16]`.
pub fn channel_payload_u16(channel_id: u8, ts_us: i64, value: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11);
    buf.push(channel_id);
    buf.extend_from_slice(&ts_us.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    buf
}

// ── v3 registry ───────────────────────────────────────────────────────────────

/// 40-byte v3 registry entry: id(1)+type(1)+rate(2)+scale(f32@4)+offset(f32@8)+name(20@12)+units(8@32).
#[allow(clippy::too_many_arguments)]
pub fn v3_registry_entry(
    channel_id: u8,
    data_type: u8,
    sample_rate_hz: u16,
    scale: f32,
    offset: f32,
    name: &str,
    units: &str,
) -> Vec<u8> {
    let mut buf = vec![0u8; 40];
    buf[0] = channel_id;
    buf[1] = data_type;
    buf[2..4].copy_from_slice(&sample_rate_hz.to_le_bytes());
    buf[4..8].copy_from_slice(&scale.to_le_bytes());
    buf[8..12].copy_from_slice(&offset.to_le_bytes());
    write_str(&mut buf, 12, name, 19);
    write_str(&mut buf, 32, units, 7);
    buf
}

/// Six v3 registry entries for one IMU's axes (AccelX..GyroZ from `start_id`).
pub fn v3_imu_axes_registry(
    imu_index: u8,
    start_id: u8,
    sample_rate_hz: u16,
    accel_scale: f32,
    gyro_scale: f32,
) -> Vec<Vec<u8>> {
    let p = format!("IMU{imu_index}_");
    vec![
        v3_registry_entry(start_id, 4, sample_rate_hz, accel_scale, 0.0, &format!("{p}AccelX"), "g"),
        v3_registry_entry(start_id + 1, 4, sample_rate_hz, accel_scale, 0.0, &format!("{p}AccelY"), "g"),
        v3_registry_entry(start_id + 2, 4, sample_rate_hz, accel_scale, 0.0, &format!("{p}AccelZ"), "g"),
        v3_registry_entry(start_id + 3, 4, sample_rate_hz, gyro_scale, 0.0, &format!("{p}GyroX"), "dps"),
        v3_registry_entry(start_id + 4, 4, sample_rate_hz, gyro_scale, 0.0, &format!("{p}GyroY"), "dps"),
        v3_registry_entry(start_id + 5, 4, sample_rate_hz, gyro_scale, 0.0, &format!("{p}GyroZ"), "dps"),
    ]
}

/// Concatenates buffer fragments.
pub fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}
