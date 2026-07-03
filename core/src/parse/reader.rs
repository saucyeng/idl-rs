//! Sequential little-endian byte reader.
//!
//! Mirrors Dart `_ByteReader` in `binary_parser.dart`. Every read is
//! bounds-checked and returns [`ParseError::TruncatedRecord`] when the buffer
//! ends before the requested number of bytes — the parser catches this to
//! terminate the record loop and surface a partial session.

use crate::session::ParseError;

/// A cursor over a byte slice that reads little-endian primitives.
pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Creates a reader positioned at the start of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current read offset in bytes from the start.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes remaining between the cursor and the end of the buffer.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// `true` while the cursor has not reached the end of the buffer.
    pub fn has_more(&self) -> bool {
        self.pos < self.data.len()
    }

    fn require(&self, n: usize, context: &str) -> Result<(), ParseError> {
        if self.remaining() < n {
            return Err(ParseError::TruncatedRecord(format!(
                "Unexpected end of file at offset {} (need {} bytes for {}, have {})",
                self.pos,
                n,
                context,
                self.remaining()
            )));
        }
        Ok(())
    }

    /// Reads an unsigned 8-bit integer.
    pub fn u8(&mut self, context: &str) -> Result<u8, ParseError> {
        self.require(1, context)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    /// Reads an unsigned little-endian 16-bit integer.
    pub fn u16(&mut self, context: &str) -> Result<u16, ParseError> {
        self.require(2, context)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    /// Reads an unsigned little-endian 32-bit integer.
    pub fn u32(&mut self, context: &str) -> Result<u32, ParseError> {
        self.require(4, context)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    /// Reads a signed 8-bit integer.
    pub fn i8(&mut self, context: &str) -> Result<i8, ParseError> {
        self.require(1, context)?;
        let v = self.data[self.pos] as i8;
        self.pos += 1;
        Ok(v)
    }

    /// Reads a signed little-endian 16-bit integer.
    pub fn i16(&mut self, context: &str) -> Result<i16, ParseError> {
        self.require(2, context)?;
        let v = i16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    /// Reads a signed little-endian 32-bit integer.
    pub fn i32(&mut self, context: &str) -> Result<i32, ParseError> {
        self.require(4, context)?;
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    /// Reads a signed little-endian 64-bit integer.
    pub fn i64(&mut self, context: &str) -> Result<i64, ParseError> {
        self.require(8, context)?;
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    /// Reads a little-endian 32-bit float.
    pub fn f32(&mut self, context: &str) -> Result<f32, ParseError> {
        self.require(4, context)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    /// Reads a little-endian 64-bit float.
    pub fn f64(&mut self, context: &str) -> Result<f64, ParseError> {
        self.require(8, context)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    /// Returns a borrowed slice of the next `n` bytes and advances the cursor.
    pub fn bytes(&mut self, n: usize, context: &str) -> Result<&'a [u8], ParseError> {
        self.require(n, context)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Advances the cursor `n` bytes without returning them.
    pub fn skip(&mut self, n: usize, context: &str) -> Result<(), ParseError> {
        self.require(n, context)?;
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_each_width_little_endian() {
        // Arrange — u8=0x01, u16=0x0302, u32=0x07060504, i16=-1 (0xFFFF), f32=1.0
        let buf: Vec<u8> = vec![
            0x01, // u8
            0x02, 0x03, // u16 = 0x0302
            0x04, 0x05, 0x06, 0x07, // u32 = 0x07060504
            0xFF, 0xFF, // i16 = -1
            0x00, 0x00, 0x80, 0x3F, // f32 = 1.0
        ];
        let mut r = ByteReader::new(&buf);

        // Act + Assert
        assert_eq!(r.u8("a").unwrap(), 0x01);
        assert_eq!(r.u16("b").unwrap(), 0x0302);
        assert_eq!(r.u32("c").unwrap(), 0x0706_0504);
        assert_eq!(r.i16("d").unwrap(), -1);
        assert_eq!(r.f32("e").unwrap(), 1.0);
        assert!(!r.has_more());
    }

    #[test]
    fn underrun_returns_truncated_record() {
        // Arrange — only 2 bytes, ask for an i64.
        let buf = vec![0x00, 0x01];
        let mut r = ByteReader::new(&buf);

        // Act
        let err = r.i64("eight").unwrap_err();

        // Assert
        assert!(matches!(err, ParseError::TruncatedRecord(_)));
        // Cursor unmoved after a failed read.
        assert_eq!(r.position(), 0);
    }

    #[test]
    fn bytes_and_skip_advance_cursor() {
        // Arrange
        let buf = vec![1, 2, 3, 4, 5];
        let mut r = ByteReader::new(&buf);

        // Act
        let head = r.bytes(2, "head").unwrap().to_vec();
        r.skip(1, "mid").unwrap();
        let tail = r.bytes(2, "tail").unwrap().to_vec();

        // Assert
        assert_eq!(head, vec![1, 2]);
        assert_eq!(tail, vec![4, 5]);
        assert!(!r.has_more());
    }
}
