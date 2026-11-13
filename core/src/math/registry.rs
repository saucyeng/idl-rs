//! The builtin registry (spec `2026-11-13-idl-rs-builtin-registry-v2.md`): a
//! builtin is a typed Rust function. Its parameter types are its declaration:
//! [`FromArg`] turns each positional [`Value`] into the parameter's type
//! through the evaluator's own `require_*` helpers, so every error kind and
//! message is the one the hand-written `call_function` arms gave, and
//! [`IntoBuiltin`] derives the arity from the parameter list. A function that
//! needs the channel lookup, the lap context, keyword arguments or the
//! spelling it was called by takes a leading `&Ctx`, which consumes no
//! argument.
//!
//! A [`Registry`] is a value: [`Registry::builtins`] is idl-rs's own set, and
//! a program that embeds idl-rs adds its own functions with [`Registry::add`]
//! and evaluates through [`crate::math::eval::eval_with`]. The generated
//! references read only idl-rs's own set.
//!
//! Prototype: the converted builtins live in `math::builtins`; every other
//! name still falls through to `eval::call_function`'s match.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::OnceLock;

use crate::math::catalog::{MathBuiltin, MathBuiltinStatus};
use crate::math::eval::{err, reject_any_kwargs, reject_unknown_kwargs, require_arg_count, ChannelLookup, Kwargs, MathLapContext};
use crate::math::value::{ChannelValue, Value, Vec3Value};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// A builtin's hand-written documentation and keyword list: everything a
/// parameter type cannot say. Rendered unchanged into the generated workbook
/// reference and the editor's catalog (see [`MathBuiltin`] for each column).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinMeta {
    /// The spelling a workbook calls it by.
    pub name: &'static str,
    /// The reference's grouping heading.
    pub category: &'static str,
    /// The call form(s) as a human reads them; names literal choices a type can't.
    pub signature: &'static str,
    /// The unit rule expression (`units.rs` vocabulary).
    pub unit_rule: &'static str,
    /// The result's value shape (`scalar`, `[t]`, `[f]`, `[lap]`, `Vec3`, ...).
    pub shape: &'static str,
    /// One line saying what the function computes.
    pub description: &'static str,
    /// One runnable example call.
    pub example: &'static str,
    /// The keyword arguments the body reads through [`Ctx::kwarg`]; any other
    /// keyword is refused before the call.
    pub kwargs: &'static [&'static str],
}

/// What a builtin body may read besides its positional arguments.
pub struct Ctx<'a> {
    name: &'a str,
    lookup: &'a dyn ChannelLookup,
    lap: &'a MathLapContext,
    kwargs: &'a Kwargs,
}

impl<'a> Ctx<'a> {
    /// The spelling the call used (`cumtrapz` or `cumulative_trapezoid`), for
    /// a body shared by several names and for its error messages.
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// The session's channels, estimator outputs and time base.
    pub fn lookup(&self) -> &'a dyn ChannelLookup {
        self.lookup
    }

    /// The lap structure and the selected window.
    pub fn lap(&self) -> &'a MathLapContext {
        self.lap
    }

    /// A keyword argument's value, `None` when the call did not pass it. Only
    /// names listed in [`BuiltinMeta::kwargs`] ever reach the body.
    pub fn kwarg(&self, key: &str) -> Option<&'a Value> {
        self.kwargs.get(key)
    }
}

/// One positional argument converted to a parameter type. The conversions
/// call the evaluator's `require_*` helpers, so a wrong kind fails with the
/// same typed error a hand-written arm gave; `name` is the spelling called.
pub trait FromArg: Sized {
    /// Whether the parameter may be left off the end of the call (`Option<T>`).
    const OPTIONAL: bool = false;

    /// Converts the argument, or fails with a typed error naming `name`.
    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError>;

    /// The value of a trailing optional parameter the call left off.
    fn absent() -> Option<Self> {
        None
    }
}

impl FromArg for f64 {
    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError> {
        crate::math::eval::require_scalar(v, name)
    }
}

impl FromArg for String {
    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError> {
        crate::math::eval::require_string(v, name).map(str::to_owned)
    }
}

impl FromArg for ChannelValue {
    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError> {
        crate::math::eval::require_channel(v, name)
    }
}

impl FromArg for Vec3Value {
    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError> {
        crate::math::vector::require_vec3(v, name).cloned()
    }
}

impl FromArg for Value {
    fn from_arg(v: &Value, _name: &str) -> Result<Self, MathEvalError> {
        Ok(v.clone())
    }
}

impl<T: FromArg> FromArg for Option<T> {
    const OPTIONAL: bool = true;

    fn from_arg(v: &Value, name: &str) -> Result<Self, MathEvalError> {
        T::from_arg(v, name).map(Some)
    }

    fn absent() -> Option<Self> {
        Some(None)
    }
}

/// A builtin body after its parameters are erased: context and raw arguments in.
pub type Body = Box<dyn Fn(&Ctx<'_>, &[Value]) -> Result<Value, MathEvalError> + Send + Sync>;

/// A Rust function usable as a builtin: one of 0..=6 [`FromArg`] parameters,
/// optionally after a leading `&Ctx`, returning `Result<Value, MathEvalError>`.
/// `Marker` only keeps the impls apart; callers never name it.
pub trait IntoBuiltin<Marker> {
    /// `(min, max)` positional argument count: trailing `Option` parameters
    /// lower the minimum.
    fn arity() -> (usize, usize);

    /// Erases the parameter types into a body over raw arguments.
    fn into_body(self) -> Body;
}

/// [`IntoBuiltin`] marker for a function of positional parameters `T` only.
pub struct Plain<T>(PhantomData<T>);

/// [`IntoBuiltin`] marker for a function taking `&Ctx` then parameters `T`.
pub struct WithCtx<T>(PhantomData<T>);

/// `(min, max)` from each parameter's [`FromArg::OPTIONAL`], in order.
fn arity_of(optional: &[bool]) -> (usize, usize) {
    let min = optional.iter().rposition(|o| !o).map_or(0, |i| i + 1);
    (min, optional.len())
}

/// The next positional argument as `T`, advancing `i`.
fn next_arg<T: FromArg>(name: &str, args: &[Value], i: &mut usize) -> Result<T, MathEvalError> {
    let v = args.get(*i);
    *i += 1;
    match v {
        Some(v) => T::from_arg(v, name),
        None => T::absent().ok_or_else(|| err(MathEvalErrorKind::ArgCount, format!("{name}: missing argument {i}"))),
    }
}

macro_rules! into_builtin {
    ($($T:ident),*) => {
        #[allow(non_snake_case, unused_mut, unused_variables)]
        impl<F, $($T: FromArg + 'static),*> IntoBuiltin<Plain<($($T,)*)>> for F
        where
            F: Fn($($T),*) -> Result<Value, MathEvalError> + Send + Sync + 'static,
        {
            fn arity() -> (usize, usize) {
                arity_of(&[$($T::OPTIONAL),*])
            }

            fn into_body(self) -> Body {
                Box::new(move |ctx: &Ctx<'_>, args: &[Value]| {
                    let mut i = 0;
                    $(let $T = next_arg::<$T>(ctx.name, args, &mut i)?;)*
                    self($($T),*)
                })
            }
        }

        #[allow(non_snake_case, unused_mut, unused_variables)]
        impl<F, $($T: FromArg + 'static),*> IntoBuiltin<WithCtx<($($T,)*)>> for F
        where
            F: Fn(&Ctx<'_>, $($T),*) -> Result<Value, MathEvalError> + Send + Sync + 'static,
        {
            fn arity() -> (usize, usize) {
                arity_of(&[$($T::OPTIONAL),*])
            }

            fn into_body(self) -> Body {
                Box::new(move |ctx: &Ctx<'_>, args: &[Value]| {
                    let mut i = 0;
                    $(let $T = next_arg::<$T>(ctx.name, args, &mut i)?;)*
                    self(ctx, $($T),*)
                })
            }
        }
    };
}

into_builtin!();
into_builtin!(A);
into_builtin!(A, B);
into_builtin!(A, B, C);
into_builtin!(A, B, C, D);
into_builtin!(A, B, C, D, E);
into_builtin!(A, B, C, D, E, G);

/// One registered builtin.
struct Entry {
    meta: BuiltinMeta,
    arity: (usize, usize),
    body: Body,
}

/// The set of builtins an evaluation can call, in registration order.
#[derive(Default)]
pub struct Registry {
    entries: Vec<Entry>,
    index: HashMap<&'static str, usize>,
}

/// Argument counts `0..=6` as `u32`s, so a contiguous `min..=max` arity is a
/// `'static` subslice for [`MathBuiltin::arity`].
static COUNTS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// idl-rs's own builtins: the set the CLI, the app and the generated
    /// references use.
    pub fn builtins() -> Self {
        let mut registry = Self::new();
        crate::math::builtins::register(&mut registry);
        registry
    }

    /// The one [`Registry::builtins`] instance behind the evaluator entry
    /// points that take no registry, and behind `math_builtin_catalog`.
    pub(crate) fn shared() -> &'static Registry {
        static SHARED: OnceLock<Registry> = OnceLock::new();
        SHARED.get_or_init(Registry::builtins)
    }

    /// Adds `f` under `meta.name`; a name already registered is replaced, so a
    /// program can override one of idl-rs's builtins with its own.
    pub fn add<M, F: IntoBuiltin<M>>(&mut self, meta: BuiltinMeta, f: F) -> &mut Self {
        let entry = Entry { meta, arity: F::arity(), body: f.into_body() };
        match self.index.get(meta.name) {
            Some(&i) => self.entries[i] = entry,
            None => {
                self.index.insert(meta.name, self.entries.len());
                self.entries.push(entry);
            }
        }
        self
    }

    /// The registered builtins' metadata, in registration order.
    pub fn metas(&self) -> impl Iterator<Item = &BuiltinMeta> {
        self.entries.iter().map(|e| &e.meta)
    }

    /// `name`'s catalog row (documentation plus derived arity), `None` when
    /// `name` is not registered.
    pub fn catalog_entry(&self, name: &str) -> Option<MathBuiltin> {
        let e = &self.entries[*self.index.get(name)?];
        let m = e.meta;
        Some(MathBuiltin {
            name: m.name,
            arity: &COUNTS[e.arity.0..=e.arity.1],
            status: MathBuiltinStatus::Implemented,
            category: m.category,
            signature: m.signature,
            unit_rule: m.unit_rule,
            shape: m.shape,
            description: m.description,
            example: m.example,
        })
    }

    /// Calls `name`, `None` when it is not registered. Keyword arguments are
    /// refused before the arity check for a function that takes none, and
    /// after it for one that lists some — the order the hand-written arms used.
    pub(crate) fn call(
        &self,
        name: &str,
        args: &[Value],
        kwargs: &Kwargs,
        lookup: &dyn ChannelLookup,
        lap: &MathLapContext,
    ) -> Option<Result<Value, MathEvalError>> {
        let e = &self.entries[*self.index.get(name)?];
        Some(e.call(name, args, kwargs, lookup, lap))
    }
}

impl Entry {
    /// Checks keywords and arity, then runs the body.
    fn call(
        &self,
        name: &str,
        args: &[Value],
        kwargs: &Kwargs,
        lookup: &dyn ChannelLookup,
        lap: &MathLapContext,
    ) -> Result<Value, MathEvalError> {
        if self.meta.kwargs.is_empty() {
            reject_any_kwargs(name, kwargs)?;
        }
        let (min, max) = self.arity;
        if min == max {
            require_arg_count(name, args, min)?;
        } else if args.len() < min || args.len() > max {
            return Err(err(
                MathEvalErrorKind::ArgCount,
                format!("{name}: expected {min} to {max} argument(s), got {}", args.len()),
            ));
        }
        if !self.meta.kwargs.is_empty() {
            reject_unknown_kwargs(name, kwargs, self.meta.kwargs)?;
        }
        (self.body)(&Ctx { name, lookup, lap, kwargs }, args)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::math::catalog::math_builtin_catalog;
    use crate::math::eval::{eval_with, require_channel, require_scalar, require_string, LookupChannel};
    use crate::math::parse::parse;
    use crate::math::value::ValueAxis;
    use crate::session::handle::SessionHandle;

    /// The committed synthetic fixture, with the channel names the documented
    /// examples use mapped onto its own (the examples are written for a real
    /// bike; the generated reference prints them unchanged).
    struct ExampleLookup(SessionHandle);

    impl ChannelLookup for ExampleLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            match name {
                "Fork" | "AccelZ" => self.0.lookup("IMU0_AccelZ"),
                // The fixture has no wheel sensor: its GPS fixes stand in as a
                // pulse event channel, the same samples at their recorded times
                // with no rate.
                "WheelFront" => self.0.lookup("GPS_SpeedKmh").map(|c| LookupChannel { sample_rate_hz: 0.0, ..c }),
                other => self.0.lookup(other),
            }
        }

        fn estimator_channel(&self, channel_id: &str) -> Option<LookupChannel> {
            self.0.estimator_channel(channel_id)
        }
    }

    /// A lookup with no channels, for functions that read none.
    struct NoChannels;

    impl ChannelLookup for NoChannels {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
    }

    fn fixture() -> ExampleLookup {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/synth-3lap.idl0");
        ExampleLookup(SessionHandle::from_path(path).expect("committed fixture is present"))
    }

    fn run(registry: &Registry, lookup: &dyn ChannelLookup, src: &str) -> Result<Value, MathEvalError> {
        eval_with(&parse(src)?, registry, lookup, &MathLapContext::empty())
    }

    /// Positional parameters in a signature's first call form: top-level
    /// commas inside its parentheses, keyword parameters (`k=v`) not counted.
    fn positional_count(signature: &str) -> usize {
        let open = signature.find('(').expect("a signature opens a parenthesis");
        let (mut depth, mut start, mut parts) = (0, open + 1, Vec::new());
        for (i, c) in signature.char_indices().skip(open + 1) {
            match c {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    parts.push(&signature[start..i]);
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&signature[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        parts.iter().filter(|p| !p.trim().is_empty() && !p.contains('=')).count()
    }

    /// An outside function: a program's own builtin, not one of idl-rs's.
    fn double_it(x: f64) -> Result<Value, MathEvalError> {
        Ok(Value::Scalar(2.0 * x))
    }

    /// An outside function with a trailing optional parameter.
    fn scale(x: f64, k: Option<f64>) -> Result<Value, MathEvalError> {
        Ok(Value::Scalar(x * k.unwrap_or(1.0)))
    }

    fn outside_meta(name: &'static str) -> BuiltinMeta {
        BuiltinMeta {
            name, category: "Outside", signature: "", unit_rule: "SameAsArg(0)", shape: "scalar",
            description: "", example: "", kwargs: &[],
        }
    }

    #[test]
    fn registry_every_example_on_the_fixture_evaluates_to_its_declared_shape() {
        // Arrange
        let registry = Registry::builtins();
        let lookup = fixture();

        for meta in registry.metas() {
            // Act
            let value = run(&registry, &lookup, meta.example);

            // Assert
            let shape = match value {
                Ok(Value::Channel(c)) if c.axis == ValueAxis::Time => "[t]",
                Ok(Value::Channel(_)) => "[lap]",
                Ok(Value::Scalar(_)) => "scalar",
                Ok(Value::Vec3(_)) => "Vec3",
                Ok(Value::Str(_)) => "string",
                Err(e) => panic!("{}: {e:?}", meta.example),
            };
            assert_eq!(shape, meta.shape, "{}", meta.example);
        }
    }

    #[test]
    fn registry_declared_keyword_is_accepted_and_an_undeclared_one_is_refused_with_todays_message() {
        // Arrange
        let registry = Registry::builtins();
        let lookup = fixture();

        for meta in registry.metas() {
            let open = &meta.example[..meta.example.len() - 1];

            // Act — every declared keyword today (`max_fraction`) is a scalar.
            let accepted: Vec<_> = meta.kwargs.iter().map(|k| run(&registry, &lookup, &format!("{open}, {k}=0.35)"))).collect();
            let refused = run(&registry, &lookup, &format!("{open}, zzz=1)")).unwrap_err();

            // Assert
            for result in accepted {
                assert!(result.is_ok(), "{}: {:?}", meta.name, result.err());
            }
            let expected = if meta.kwargs.is_empty() {
                format!("{0}: unknown keyword argument \"zzz\" — {0} takes no keyword arguments", meta.name)
            } else {
                format!("{}: unknown keyword argument \"zzz\" (accepts: {})", meta.name, meta.kwargs.join(", "))
            };
            assert_eq!(refused.kind, MathEvalErrorKind::Runtime);
            assert_eq!(refused.message, expected);
        }
    }

    #[test]
    fn registry_signature_argument_count_is_within_the_functions_arity() {
        // Arrange
        let registry = Registry::builtins();

        for e in &registry.entries {
            // Act
            let n = positional_count(e.meta.signature);

            // Assert
            assert!(e.arity.0 <= n && n <= e.arity.1, "{}: signature has {n}, arity {:?}", e.meta.name, e.arity);
        }
    }

    #[test]
    fn registry_every_name_takes_exactly_one_row_of_the_catalog() {
        // Arrange
        let registry = Registry::builtins();

        for meta in registry.metas() {
            // Act
            let rows: Vec<&MathBuiltin> = math_builtin_catalog().iter().filter(|b| b.name == meta.name).collect();

            // Assert — once, and with the registry's documentation, not a placeholder.
            assert_eq!(rows.len(), 1, "{}", meta.name);
            assert_eq!(rows[0].signature, meta.signature, "{}", meta.name);
        }
    }

    #[test]
    fn from_arg_accepts_its_own_kind_and_refuses_the_others_with_the_require_message() {
        // Arrange
        let ch = Value::Channel(ChannelValue {
            samples: Arc::from(vec![1.0, 2.0]),
            sample_rate_hz: 10.0,
            channel_id: None,
            t_us: Arc::from(vec![0, 100_000]),
            axis: ValueAxis::Time,
        });
        let v3 = Value::Vec3(Box::new(Vec3Value { x: Value::Scalar(1.0), y: Value::Scalar(2.0), z: Value::Scalar(3.0) }));
        let kinds = [ch, Value::Scalar(2.0), Value::Str("low".into()), v3];

        for v in &kinds {
            // Act
            let scalar = f64::from_arg(v, "f");
            let string = String::from_arg(v, "f");
            let channel = ChannelValue::from_arg(v, "f");
            let vec3 = Vec3Value::from_arg(v, "f");
            let value = Value::from_arg(v, "f");
            let optional = Option::<f64>::from_arg(v, "f");

            // Assert — a scalar is also a one-sample untimed channel, as before.
            assert_eq!(scalar.is_ok(), matches!(v, Value::Scalar(_)));
            assert_eq!(scalar.err(), require_scalar(v, "f").err());
            assert_eq!(string.is_ok(), matches!(v, Value::Str(_)));
            assert_eq!(string.err(), require_string(v, "f").err());
            assert_eq!(channel.is_ok(), matches!(v, Value::Channel(_) | Value::Scalar(_)));
            assert_eq!(channel.err(), require_channel(v, "f").err());
            assert_eq!(vec3.is_ok(), matches!(v, Value::Vec3(_)));
            assert_eq!(vec3.err(), crate::math::vector::require_vec3(v, "f").err());
            assert_eq!(value, Ok(v.clone()));
            assert_eq!(optional.err(), require_scalar(v, "f").err());
        }
        assert_eq!(Option::<f64>::absent(), Some(None));
        assert_eq!(f64::absent(), None);
    }

    #[test]
    fn registry_fixed_arity_mismatch_gives_require_arg_counts_error() {
        // Arrange
        let registry = Registry::builtins();

        // Act
        let e = run(&registry, &NoChannels, "butter(2, 5)").unwrap_err();

        // Assert
        assert_eq!(e.kind, MathEvalErrorKind::ArgCount);
        assert_eq!(e.message, "butter: expected 4 argument(s), got 2");
    }

    #[test]
    fn registry_outside_function_evaluates_and_is_absent_from_the_generated_references() {
        // Arrange
        let mut registry = Registry::builtins();
        registry.add(outside_meta("double_it"), double_it);

        // Act
        let v = run(&registry, &NoChannels, "double_it(21)");
        let reference = crate::docs::render_workbook_reference(&[]);
        let json = crate::docs::workbook_catalog_json().to_string();

        // Assert
        assert_eq!(v, Ok(Value::Scalar(42.0)));
        assert!(math_builtin_catalog().iter().all(|b| b.name != "double_it"));
        assert!(!reference.contains("double_it"));
        assert!(!json.contains("double_it"));
    }

    #[test]
    fn registry_trailing_option_parameter_lowers_the_minimum_arity() {
        // Arrange
        let mut registry = Registry::new();
        registry.add(outside_meta("scale"), scale);

        // Act
        let one = run(&registry, &NoChannels, "scale(2)");
        let two = run(&registry, &NoChannels, "scale(2, 3)");
        let none = run(&registry, &NoChannels, "scale()").unwrap_err();

        // Assert
        assert_eq!(one, Ok(Value::Scalar(2.0)));
        assert_eq!(two, Ok(Value::Scalar(6.0)));
        assert_eq!(none.kind, MathEvalErrorKind::ArgCount);
        assert_eq!(none.message, "scale: expected 1 to 2 argument(s), got 0");
    }
}
