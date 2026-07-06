//! Math-channel expression engine — tokenizer, recursive-descent parser,
//! evaluator, and function set. Ported from the Dart `MathChannelEvaluator`
//! (`app/lib/data/math_channel_evaluator.dart`). Pure: data in, data out.

pub mod aggregate;
pub mod channel_def;
pub mod error;
pub mod eval;
pub mod parse;
pub mod resolve;
pub mod token;
pub mod value;
pub mod variance_geom;
pub mod vector;

pub use channel_def::MathChannelDef;
pub use error::{MathEvalError, MathEvalErrorKind};
pub use eval::{
    evaluate, evaluate_scalar, ChannelLookup, EvalOutput, LookupChannel, MathLapContext,
    MathOverlay,
};
pub use resolve::resolve_dependencies;
pub use value::{ChannelValue, Value, Vec3Value};

#[cfg(test)]
mod tests_parity;
