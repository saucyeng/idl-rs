//! Builtin evaluators, one `fn` per [`crate::math::registry::Builtin`] entry
//! (spellings that shared an arm share one fn). Each body is the arm it
//! replaced in `call_function`'s old `match`, moved verbatim, so every error
//! kind and message is unchanged. Children of `eval` so they keep using its
//! private helpers (`require_*`, `channel`, `timed`, ...) unchanged.

pub(crate) mod dsp;
pub(crate) mod elementwise;
pub(crate) mod estimator;
pub(crate) mod laps;
pub(crate) mod vector;
pub(crate) mod windowed;
