//! Low-level FIT (Garmin) binary encoder: CRC-16, base types, and `FitWriter`
//! (file header, definition/data message records, trailing CRC). Pure — knows
//! nothing about GPS or heart rate; the message stream is assembled in the
//! parent `mod.rs`. FIT format reference: developer.garmin.com/fit/protocol.

/// Seconds between the unix epoch (1970-01-01) and the FIT epoch (1989-12-31).
pub const FIT_EPOCH_OFFSET_SECS: i64 = 631_065_600;

/// FIT CRC-16 over `data`, seed 0. Uses the 16-entry nibble table from the FIT
/// protocol spec (each byte folded low-nibble then high-nibble). The FIT file
/// trailer and header both carry this checksum.
pub fn crc16(data: &[u8]) -> u16 {
    const TABLE: [u16; 16] = [
        0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
    ];
    let mut crc: u16 = 0;
    for &byte in data {
        // lower nibble
        let tmp = TABLE[(crc & 0xF) as usize];
        crc = (crc >> 4) & 0x0FFF;
        crc = crc ^ tmp ^ TABLE[(byte & 0xF) as usize];
        // upper nibble
        let tmp = TABLE[(crc & 0xF) as usize];
        crc = (crc >> 4) & 0x0FFF;
        crc = crc ^ tmp ^ TABLE[((byte >> 4) & 0xF) as usize];
    }
    crc
}

/// FIT base types used by this encoder. `String(n)` carries its fixed byte
/// width `n` (null-padded). The type byte's high bit marks endian-aware
/// (multi-byte) types, per the FIT spec base-type table.
#[derive(Debug, Clone, Copy)]
pub enum BaseType {
    Enum,
    Uint8,
    Uint16,
    Sint32,
    Uint32,
    Uint32z,
    /// Fixed-width, null-padded ASCII string of `n` bytes.
    String(u8),
}

impl BaseType {
    /// The FIT base-type byte written into a definition message field.
    pub fn type_byte(self) -> u8 {
        match self {
            BaseType::Enum => 0x00,
            BaseType::Uint8 => 0x02,
            BaseType::Uint16 => 0x84,
            BaseType::Sint32 => 0x85,
            BaseType::Uint32 => 0x86,
            BaseType::Uint32z => 0x8C,
            BaseType::String(_) => 0x07,
        }
    }

    /// Field width in bytes.
    pub fn size(self) -> u8 {
        match self {
            BaseType::Enum | BaseType::Uint8 => 1,
            BaseType::Uint16 => 2,
            BaseType::Sint32 | BaseType::Uint32 | BaseType::Uint32z => 4,
            BaseType::String(n) => n,
        }
    }
}

/// One field in a definition message: field-definition number + base type.
#[derive(Debug, Clone, Copy)]
pub struct FieldDef {
    pub num: u8,
    pub base: BaseType,
}

/// Accumulates FIT data records (definition + data messages). `finish` wraps
/// them in the 14-byte file header (with data size + header CRC) and appends
/// the file CRC. Architecture is little-endian throughout.
pub struct FitWriter {
    data: Vec<u8>,
}

impl FitWriter {
    pub fn new() -> Self {
        FitWriter { data: Vec::new() }
    }

    /// The raw data-record bytes written so far (no header, no trailing CRC).
    #[cfg(test)]
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn push_u8(&mut self, v: u8) {
        self.data.push(v);
    }

    pub fn push_enum(&mut self, v: u8) {
        self.data.push(v);
    }

    pub fn push_u16(&mut self, v: u16) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_u32(&mut self, v: u32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_i32(&mut self, v: i32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    /// Write `s` as a fixed `size`-byte, null-padded ASCII field. Non-ASCII
    /// bytes pass through verbatim; output is truncated to `size`, always
    /// null-terminated within the field when it fills it.
    pub fn push_string(&mut self, s: &str, size: u8) {
        let mut buf = vec![0u8; size as usize];
        for (i, b) in s.bytes().take(size as usize).enumerate() {
            buf[i] = b;
        }
        // Guarantee a trailing null when the string fills the field.
        let last = (size as usize).saturating_sub(1);
        if s.len() >= size as usize {
            buf[last] = 0;
        }
        self.data.extend_from_slice(&buf);
    }

    /// Write a definition message for `local_type` describing `global_msg`'s
    /// `fields`. Header byte = 0x40 | local_type; architecture is little-endian
    /// (0). Subsequent `data_header(local_type)` records must push the same
    /// fields, in this order, at these widths.
    pub fn definition(&mut self, local_type: u8, global_msg: u16, fields: &[FieldDef]) {
        self.data.push(0x40 | (local_type & 0x0F)); // definition record header
        self.data.push(0x00); // reserved
        self.data.push(0x00); // architecture: 0 = little-endian
        self.data.extend_from_slice(&global_msg.to_le_bytes());
        self.data.push(fields.len() as u8);
        for f in fields {
            self.data.push(f.num);
            self.data.push(f.base.size());
            self.data.push(f.base.type_byte());
        }
    }

    /// Write a data message record header for `local_type` (high bits clear).
    /// Follow with `push_*` calls matching the definition's field order.
    pub fn data_header(&mut self, local_type: u8) {
        self.data.push(local_type & 0x0F);
    }

    /// Frame the accumulated data records into a complete FIT file: a 14-byte
    /// header (size, protocol 2.0, profile version, data size, ".FIT", header
    /// CRC) followed by the data records and a 2-byte file CRC over everything
    /// preceding it. Consumes the writer.
    pub fn finish(self) -> Vec<u8> {
        const HEADER_SIZE: u8 = 14;
        const PROTOCOL_VERSION: u8 = 0x20; // 2.0
        const PROFILE_VERSION: u16 = 2100;

        let data_size = self.data.len() as u32;
        let mut out = Vec::with_capacity(14 + self.data.len() + 2);
        out.push(HEADER_SIZE);
        out.push(PROTOCOL_VERSION);
        out.extend_from_slice(&PROFILE_VERSION.to_le_bytes());
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(b".FIT");
        let header_crc = crc16(&out[0..12]);
        out.extend_from_slice(&header_crc.to_le_bytes());

        out.extend_from_slice(&self.data);

        let file_crc = crc16(&out);
        out.extend_from_slice(&file_crc.to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc16_of_empty_is_zero() {
        // Arrange / Act / Assert — FIT CRC seed is 0; no bytes → 0.
        assert_eq!(crc16(&[]), 0);
    }

    #[test]
    fn crc16_matches_known_vector() {
        // Arrange — the ASCII bytes ".FIT".
        let data = b".FIT";

        // Act
        let crc = crc16(data);

        // Assert — value produced by the reference FIT CRC algorithm.
        assert_eq!(crc, 0x92DE);
    }

    #[test]
    fn base_type_byte_and_size_are_correct() {
        // Assert — type byte has bit7 set for multi-byte (endian-aware) types.
        assert_eq!((BaseType::Enum.type_byte(), BaseType::Enum.size()), (0x00, 1));
        assert_eq!((BaseType::Uint8.type_byte(), BaseType::Uint8.size()), (0x02, 1));
        assert_eq!((BaseType::Uint16.type_byte(), BaseType::Uint16.size()), (0x84, 2));
        assert_eq!((BaseType::Sint32.type_byte(), BaseType::Sint32.size()), (0x85, 4));
        assert_eq!((BaseType::Uint32.type_byte(), BaseType::Uint32.size()), (0x86, 4));
        assert_eq!((BaseType::Uint32z.type_byte(), BaseType::Uint32z.size()), (0x8C, 4));
        assert_eq!((BaseType::String(5).type_byte(), BaseType::String(5).size()), (0x07, 5));
    }

    #[test]
    fn pushes_are_little_endian() {
        // Arrange
        let mut w = FitWriter::new();

        // Act — write a u16 and an i32 into the data buffer.
        w.push_u16(0x0102);
        w.push_i32(-2);

        // Assert — little-endian byte order.
        assert_eq!(w.data_bytes(), &[0x02, 0x01, 0xFE, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn definition_message_layout() {
        // Arrange — a local-type-0 definition for global msg 0 (file_id) with one
        // enum field (num 0).
        let mut w = FitWriter::new();

        // Act
        w.definition(0, 0, &[FieldDef { num: 0, base: BaseType::Enum }]);

        // Assert — header 0x40, reserved 0, arch 0 (LE), global 0x0000, 1 field,
        // then [field_num=0, size=1, base_type=0x00].
        assert_eq!(
            w.data_bytes(),
            &[0x40, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00]
        );
    }

    #[test]
    fn data_header_uses_local_type_with_high_bits_clear() {
        // Arrange
        let mut w = FitWriter::new();

        // Act
        w.data_header(0);
        w.push_enum(4);

        // Assert — data record header is just the local type; then the value.
        assert_eq!(w.data_bytes(), &[0x00, 0x04]);
    }

    #[test]
    fn finish_frames_header_and_trailing_crc() {
        // Arrange — a minimal file with one data record byte sequence.
        let mut w = FitWriter::new();
        w.data_header(0);
        w.push_enum(4); // 2 data bytes total: [0x00, 0x04]

        // Act
        let bytes = w.finish();

        // Assert — 14-byte header + 2 data bytes + 2 CRC bytes = 18.
        assert_eq!(bytes.len(), 18);
        // Header: size=14, protocol=0x20, profile=2100 (LE), data_size=2 (LE),
        // ".FIT", header CRC (LE).
        assert_eq!(bytes[0], 14);
        assert_eq!(bytes[1], 0x20);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 2100);
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 2);
        assert_eq!(&bytes[8..12], b".FIT");
        // Header CRC is over the first 12 bytes.
        let hdr_crc = u16::from_le_bytes([bytes[12], bytes[13]]);
        assert_eq!(hdr_crc, crc16(&bytes[0..12]));
        // File CRC is over everything except the final 2 bytes.
        let file_crc = u16::from_le_bytes([bytes[16], bytes[17]]);
        assert_eq!(file_crc, crc16(&bytes[0..16]));
    }
}
