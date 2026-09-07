//! Hand-parsed `Range: bytes=...` header (PLAN §3, ruling R88: no
//! `tower-http`). Only a single range is supported; a header naming more
//! than one range is refused rather than silently answering the first.

use std::fmt;

/// Why a `Range` header could not be honoured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeErrorKind {
    /// Not `bytes=...`, or the numeric part does not parse.
    Malformed,
    /// More than one range requested — this server answers a single range
    /// only (PLAN §3).
    Multiple,
    /// The requested start lies at or past the resource's length (RFC 7233
    /// "unsatisfiable" — the caller answers `416`).
    Unsatisfiable,
}

/// A `Range` header this server could not honour. Never `Err(String)`
/// (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeError {
    pub kind: RangeErrorKind,
    pub message: String,
}

impl RangeError {
    fn new(kind: RangeErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for RangeError {}

/// Parses one `Range: bytes=a-b` header against a resource of `len` bytes.
/// `None` for an absent header (the caller answers the whole body). The
/// returned pair is `(start, end)`, both inclusive, always within
/// `0..len` — the caller answers `206` with a `Content-Range: bytes
/// start-end/len` header. `bytes=a-` means "from `a` to the end";
/// `bytes=-n` means "the last `n` bytes". A start past the end, or a
/// header naming more than one range, is a typed [`RangeError`] the
/// caller turns into a `416`.
pub fn parse_range(header: Option<&str>, len: u64) -> Result<Option<(u64, u64)>, RangeError> {
    let Some(header) = header else { return Ok(None) };

    let Some(spec) = header.strip_prefix("bytes=") else {
        return Err(RangeError::new(RangeErrorKind::Malformed, format!("not a byte range: {header}")));
    };

    if spec.contains(',') {
        return Err(RangeError::new(RangeErrorKind::Multiple, "multiple ranges are not supported"));
    }

    let (start_str, end_str) = spec
        .split_once('-')
        .ok_or_else(|| RangeError::new(RangeErrorKind::Malformed, format!("missing '-' in range: {spec}")))?;

    if start_str.is_empty() {
        // "bytes=-n": the last n bytes.
        let suffix_len = parse_range_number(end_str, "suffix length")?;
        if suffix_len == 0 || len == 0 {
            return Err(RangeError::new(RangeErrorKind::Unsatisfiable, "suffix range on an empty resource"));
        }
        let start = len.saturating_sub(suffix_len);
        return Ok(Some((start, len - 1)));
    }

    let start = parse_range_number(start_str, "start")?;

    if start >= len {
        return Err(RangeError::new(
            RangeErrorKind::Unsatisfiable,
            format!("start {start} is at or past the resource's length {len}"),
        ));
    }

    let end = if end_str.is_empty() {
        len - 1
    } else {
        let end = parse_range_number(end_str, "end")?;
        if end < start {
            return Err(RangeError::new(RangeErrorKind::Malformed, format!("end {end} precedes start {start}")));
        }
        end.min(len - 1)
    };

    Ok(Some((start, end)))
}

/// Parses one `Range` numeric field (`start`, `end`, or a suffix length) as
/// `u64`. A value that fails to parse *because it overflows `u64`* is
/// [`RangeErrorKind::Unsatisfiable`], not [`RangeErrorKind::Malformed`]
/// (review-task8 Minor / R100): every number this large is, by
/// construction, at or past any real resource's length, so it is the same
/// class of error as "start past the end" — `416`, not `400`. Any other
/// parse failure (non-numeric, empty) stays `Malformed` — `400`.
fn parse_range_number(s: &str, what: &str) -> Result<u64, RangeError> {
    s.parse::<u64>().map_err(|e| match e.kind() {
        std::num::IntErrorKind::PosOverflow => {
            RangeError::new(RangeErrorKind::Unsatisfiable, format!("{what} {s} overflows u64, unsatisfiable by any real resource"))
        }
        _ => RangeError::new(RangeErrorKind::Malformed, format!("bad {what}: {s}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_absent_header_is_none() {
        // Arrange
        let header = None;

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn parse_range_bytes_0_9_is_the_full_ten_bytes() {
        // Arrange
        let header = Some("bytes=0-9");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result, Ok(Some((0, 9))));
    }

    #[test]
    fn parse_range_bytes_5_dash_is_five_to_the_end() {
        // Arrange
        let header = Some("bytes=5-");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result, Ok(Some((5, 9))));
    }

    #[test]
    fn parse_range_suffix_range_is_the_last_n_bytes() {
        // Arrange
        let header = Some("bytes=-3");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result, Ok(Some((7, 9))));
    }

    #[test]
    fn parse_range_start_past_the_end_is_unsatisfiable() {
        // Arrange
        let header = Some("bytes=100-200");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result.unwrap_err().kind, RangeErrorKind::Unsatisfiable);
    }

    #[test]
    fn parse_range_multi_range_header_is_refused() {
        // Arrange
        let header = Some("bytes=0-1,2-3");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result.unwrap_err().kind, RangeErrorKind::Multiple);
    }

    #[test]
    fn parse_range_malformed_header_is_refused() {
        // Arrange
        let header = Some("chunks=0-1");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result.unwrap_err().kind, RangeErrorKind::Malformed);
    }

    #[test]
    fn parse_range_a_start_that_overflows_u64_is_unsatisfiable_not_malformed() {
        // Arrange: R100 / review-task8 Minor — this exact example.
        let header = Some("bytes=0-99999999999999999999");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result.unwrap_err().kind, RangeErrorKind::Unsatisfiable);
    }

    #[test]
    fn parse_range_a_suffix_length_that_overflows_u64_is_unsatisfiable_not_malformed() {
        // Arrange
        let header = Some("bytes=-99999999999999999999");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result.unwrap_err().kind, RangeErrorKind::Unsatisfiable);
    }

    #[test]
    fn parse_range_end_clamped_to_the_resource_length() {
        // Arrange
        let header = Some("bytes=5-1000");

        // Act
        let result = parse_range(header, 10);

        // Assert
        assert_eq!(result, Ok(Some((5, 9))));
    }
}
