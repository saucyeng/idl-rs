//! Math-channel expression engine — tokenizer, recursive-descent parser,
//! evaluator, and function set. Ported from the Dart `MathChannelEvaluator`
//! (`app/lib/data/math_channel_evaluator.dart`). Pure: data in, data out.

pub mod aggregate;
pub mod alias;
pub mod catalog;
pub mod channel_def;
pub mod error;
pub mod eval;
pub mod parse;
pub mod resolve;
pub mod token;
pub mod units;
pub mod value;
pub mod variance_geom;
pub mod vector;

pub use alias::{
    math_name_migrations, migrate_body, migrate_cell_body, migrate_document, migrate_expression,
    AppliedRename, DocumentRename, MigrationKind, NameMigration,
};
pub use catalog::{math_builtin_catalog, MathBuiltin, MathBuiltinStatus};
pub use channel_def::MathChannelDef;
pub use error::{MathEvalError, MathEvalErrorKind};
pub use eval::{
    evaluate, evaluate_scalar, ChannelLookup, EvalOutput, LookupChannel, MathLapContext,
    MathOverlay,
};
pub use resolve::resolve_dependencies;
pub use value::{ChannelValue, Value, Vec3Value};

#[cfg(test)]
mod tests_ahrs;
#[cfg(test)]
mod tests_parity;
