//! Math-evaluation error model. Reuses the Phase-1 freezed-free FFI pattern:
//! a unit-enum `kind` + a `message`. The bridge mirrors this to Dart as
//! `MathEvalFailure`; the app maps `kind` onto the `MathChannelException`
//! hierarchy in `app/lib/data/exceptions.dart`.

use std::fmt;

/// Discriminant for [`MathEvalError`]. Unit enum → plain Dart enum (no freezed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathEvalErrorKind {
    /// Tokenizer / parser failure (syntax).
    Parse,
    /// Call to a function name not in the dispatch table.
    UnknownFunction,
    /// `[Name]` reference not resolvable via the channel lookup.
    UnknownChannel,
    /// Wrong number of arguments to a function.
    ArgCount,
    /// Wrong argument/operand value type (e.g. scalar where channel required).
    Type,
    /// `/` with a zero divisor.
    DivisionByZero,
    /// Lap-aware function called with no lap context available.
    NoLapContext,
    /// A deferred function stub (`spectrogram`, `hilbert`, …).
    NotImplemented,
    /// Any other runtime failure (mismatched rates/lengths, bad window, …).
    Runtime,
}

/// An evaluation error: a `kind` discriminant plus a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathEvalError {
    pub kind: MathEvalErrorKind,
    pub message: String,
}

impl MathEvalError {
    pub fn new(kind: MathEvalErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for MathEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for MathEvalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_message() {
        // Arrange
        let e = MathEvalError::new(MathEvalErrorKind::UnknownChannel, "Channel '[X]' not in this session");

        // Act
        let s = format!("{e}");

        // Assert
        assert_eq!(s, "Channel '[X]' not in this session");
        assert_eq!(e.kind, MathEvalErrorKind::UnknownChannel);
    }
}
