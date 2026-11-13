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
//! idl-rs's own builtins are registered in `math::builtins`, in the catalog's
//! order (C2 §3.3), which the generated references keep.

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

    /// Every keyword argument the call passed, for the evaluator's own
    /// keyword helpers (`kwarg_string`, `take_kwarg`).
    pub(crate) fn kwargs(&self) -> &'a Kwargs {
        self.kwargs
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

/// How a builtin's positional arguments are counted.
#[derive(Debug, Clone, Copy)]
enum Arity {
    /// `(min, max)` from the parameter types, checked before the body runs.
    Typed(usize, usize),
    /// The valid counts, declared at registration; the body checks them,
    /// because its arity errors name its own call forms.
    Declared(&'static [u32]),
}

/// One registered builtin.
struct Entry {
    meta: BuiltinMeta,
    arity: Arity,
    status: MathBuiltinStatus,
    /// Listed in [`Registry::catalog`] (false only for `main`, a table-cell form).
    documented: bool,
    body: Body,
}

/// Why a builtin could not be registered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// A builtin with this spelling is already registered.
    DuplicateName(&'static str),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::DuplicateName(name) => write!(f, "builtin \"{name}\" is already registered"),
        }
    }
}

impl std::error::Error for RegistryError {}

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
        crate::math::builtins::register(&mut registry).expect("idl-rs builtins have distinct names");
        registry
    }

    /// The one [`Registry::builtins`] instance behind the evaluator entry
    /// points that take no registry, and behind `math_builtin_catalog`.
    pub(crate) fn shared() -> &'static Registry {
        static SHARED: OnceLock<Registry> = OnceLock::new();
        SHARED.get_or_init(Registry::builtins)
    }

    /// Adds `f` under `meta.name`, its arity taken from its parameter types.
    ///
    /// # Errors
    /// [`RegistryError::DuplicateName`] when the name is already registered.
    pub fn add<M, F: IntoBuiltin<M>>(&mut self, meta: BuiltinMeta, f: F) -> Result<&mut Self, RegistryError> {
        let (min, max) = F::arity();
        self.push(meta, Arity::Typed(min, max), MathBuiltinStatus::Implemented, true, f.into_body())
    }

    /// Adds a body over the raw positional arguments, for a function with
    /// several call forms whose arity errors name those forms: `arity` lists
    /// the valid counts for the catalog, and the body checks them itself.
    ///
    /// # Errors
    /// [`RegistryError::DuplicateName`] when the name is already registered.
    pub fn add_raw<F>(&mut self, meta: BuiltinMeta, arity: &'static [u32], f: F) -> Result<&mut Self, RegistryError>
    where
        F: Fn(&Ctx<'_>, &[Value]) -> Result<Value, MathEvalError> + Send + Sync + 'static,
    {
        self.push(meta, Arity::Declared(arity), MathBuiltinStatus::Implemented, true, Box::new(f))
    }

    /// Adds a name the language reserves but does not implement yet: it
    /// parses, and evaluating it is a `NotImplemented` error naming it.
    ///
    /// # Errors
    /// [`RegistryError::DuplicateName`] when the name is already registered.
    pub fn add_not_implemented(&mut self, meta: BuiltinMeta, arity: &'static [u32]) -> Result<&mut Self, RegistryError> {
        let body: Body = Box::new(|ctx: &Ctx<'_>, _: &[Value]| {
            Err(err(MathEvalErrorKind::NotImplemented, format!("not yet implemented: {}", ctx.name)))
        });
        self.push(meta, Arity::Declared(arity), MathBuiltinStatus::NotImplemented, true, body)
    }

    /// [`Registry::add_raw`] for a callable name left out of the catalog.
    pub(crate) fn add_undocumented<F>(&mut self, name: &'static str, arity: &'static [u32], f: F) -> Result<&mut Self, RegistryError>
    where
        F: Fn(&Ctx<'_>, &[Value]) -> Result<Value, MathEvalError> + Send + Sync + 'static,
    {
        let meta = BuiltinMeta { name, category: "", signature: "", unit_rule: "", shape: "", description: "", example: "", kwargs: &[] };
        self.push(meta, Arity::Declared(arity), MathBuiltinStatus::Implemented, false, Box::new(f))
    }

    fn push(&mut self, meta: BuiltinMeta, arity: Arity, status: MathBuiltinStatus, documented: bool, body: Body) -> Result<&mut Self, RegistryError> {
        if self.index.contains_key(meta.name) {
            return Err(RegistryError::DuplicateName(meta.name));
        }
        self.index.insert(meta.name, self.entries.len());
        self.entries.push(Entry { meta, arity, status, documented, body });
        Ok(self)
    }

    /// The documented builtins' metadata, in registration order.
    pub fn metas(&self) -> impl Iterator<Item = &BuiltinMeta> {
        self.entries.iter().filter(|e| e.documented).map(|e| &e.meta)
    }

    /// The documented builtins as catalog rows, in registration order: what
    /// the generated references and the editor's catalog render.
    pub fn catalog(&self) -> Vec<MathBuiltin> {
        self.entries.iter().filter(|e| e.documented).map(|e| {
            let m = e.meta;
            let arity = match e.arity {
                Arity::Typed(min, max) => &COUNTS[min..=max],
                Arity::Declared(counts) => counts,
            };
            MathBuiltin {
                name: m.name, arity, status: e.status, category: m.category, signature: m.signature,
                unit_rule: m.unit_rule, shape: m.shape, description: m.description, example: m.example,
            }
        }).collect()
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
        match self.arity {
            Arity::Typed(min, max) if min == max => require_arg_count(name, args, min)?,
            Arity::Typed(min, max) if args.len() < min || args.len() > max => {
                return Err(err(
                    MathEvalErrorKind::ArgCount,
                    format!("{name}: expected {min} to {max} argument(s), got {}", args.len()),
                ));
            }
            _ => {}
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
    use crate::math::eval::{eval_with, require_channel, require_scalar, require_string, LapSpan, LookupChannel, MathOverlay};
    use crate::math::parse::parse;
    use crate::math::value::ValueAxis;
    use crate::session::handle::SessionHandle;

    /// The committed synthetic fixture, with the channel names the documented
    /// examples use mapped onto its own sensors (the examples are written for
    /// a real bike; the generated reference prints them unchanged).
    struct ExampleLookup(SessionHandle);

    impl ChannelLookup for ExampleLookup {
        fn lookup(&self, name: &str) -> Option<LookupChannel> {
            let stand_in = match name {
                "Fork" | "AccelZ" | "Speed" => "IMU0_AccelZ",
                "Shock" | "AccelY" => "IMU0_AccelY",
                "AccelX" | "Ratio" | "Power" | "gap_ratio" => "IMU0_AccelX",
                "Roll" => "IMU0_GyroX",
                "Pitch" => "IMU0_GyroY",
                "Yaw" => "IMU0_GyroZ",
                // No wheel sensor on the fixture: its GPS fixes stand in as a
                // pulse event channel, the same samples at their recorded
                // times with no rate.
                "WheelFront" => return self.0.lookup("GPS_SpeedKmh").map(|c| LookupChannel { sample_rate_hz: 0.0, ..c }),
                other => other,
            };
            self.0.lookup(stand_in)
        }

        fn estimator_channel(&self, channel_id: &str) -> Option<LookupChannel> {
            self.0.estimator_channel(channel_id)
        }

        fn best_time_base_dims(&self) -> Option<(usize, f64)> {
            self.0.best_time_base_dims()
        }

        fn channel_dims(&self, name: &str) -> Option<(usize, f64)> {
            self.0.channel_dims(name)
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

    /// Three equal laps of two sectors each over the fixture's span, lap 1
    /// the main lap and the whole fixture again as the overlay, so every lap
    /// builtin has something to read.
    fn fixture_laps(lookup: &ExampleLookup) -> MathLapContext {
        let t_us = lookup.lookup("IMU0_AccelZ").expect("the fixture has IMU0").t_us;
        let end = *t_us.last().expect("the fixture is not empty") as f64 / 1e6;
        let laps: Vec<LapSpan> = (0..3u32)
            .map(|k| {
                let (start, stop) = (end * k as f64 / 3.0, end * (k + 1) as f64 / 3.0);
                let mid = (start + stop) / 2.0;
                LapSpan { lap_number: k + 1, start_secs: start, end_secs: stop, lap_time_secs: stop - start, sectors: vec![(start, mid), (mid, stop)] }
            })
            .collect();
        MathLapContext {
            main_lap_bounds: laps.iter().map(|l| (l.start_secs, l.end_secs)).collect(),
            main_sectors: laps.iter().flat_map(|l| l.sectors.clone()).collect(),
            main_lap_number: Some(1),
            overlay: vec![MathOverlay { lookup: Arc::new(fixture()), lap_start_ms: f64::MIN, lap_end_ms: f64::MAX, lap_start_uniform_sec: 0.0 }],
            laps,
            ..MathLapContext::empty()
        }
    }

    fn run(registry: &Registry, lookup: &dyn ChannelLookup, laps: &MathLapContext, src: &str) -> Result<Value, MathEvalError> {
        eval_with(&parse(src)?, registry, lookup, laps)
    }

    /// An example with its bare placeholder names (`accel`, `a`, `b`,
    /// `kernel`, `sos` — definitions a real workbook would hold, not channels)
    /// replaced by an expression of that kind, so it can run.
    fn runnable(example: &str) -> String {
        let (mut out, mut word, mut quoted, mut bracketed) = (String::new(), String::new(), false, false);
        for c in example.chars().chain(std::iter::once('\0')) {
            if c.is_ascii_alphanumeric() || c == '_' {
                word.push(c);
                continue;
            }
            let stand_in = match word.as_str() {
                _ if quoted || bracketed || c == '(' => None,
                "accel" | "a" | "b" => Some("vec([AccelX], [AccelY], [AccelZ])"),
                "kernel" => Some("[Fork]"),
                "sos" => Some("1"),
                _ => None,
            };
            out.push_str(stand_in.unwrap_or(&word));
            word.clear();
            match c {
                '"' => quoted = !quoted,
                '[' => bracketed = true,
                ']' => bracketed = false,
                _ => {}
            }
            if c != '\0' {
                out.push(c);
            }
        }
        out
    }

    /// A call's top-level arguments: the text between its first `(` and the
    /// matching `)`, split on commas outside nested parentheses.
    fn top_level_args(call: &str) -> Vec<&str> {
        let open = call.find('(').expect("a call opens a parenthesis");
        let (mut depth, mut start, mut parts) = (0, open + 1, Vec::new());
        for (i, c) in call.char_indices().skip(open + 1) {
            match c {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    parts.push(&call[start..i]);
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&call[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        parts.into_iter().map(str::trim).filter(|p| !p.is_empty()).collect()
    }

    /// A value each declared keyword accepts.
    fn kwarg_value(name: &str, key: &str) -> &'static str {
        match (name, key) {
            ("mean", "window") => "64",
            (_, "window") => "\"hann\"",
            (_, "detrend") => "\"linear\"",
            (_, "scaling") => "\"spectrum\"",
            (_, "average") => "\"median\"",
            (_, "nperseg") => "128",
            (_, "noverlap") => "64",
            (_, "max_fraction") => "0.35",
            _ => panic!("{name}: no test value for keyword {key}"),
        }
    }

    /// The declared shapes a value can be: `[f]` and `[t,f]` are carried as a
    /// time-axis channel too, so a channel on the time axis answers to all three.
    fn shapes_of(v: &Value) -> &'static [&'static str] {
        match v {
            Value::Channel(c) if c.axis == ValueAxis::Lap => &["[lap]"],
            Value::Channel(_) => &["[t]", "[f]", "[t,f]"],
            Value::Scalar(_) => &["scalar"],
            Value::Vec3(_) => &["Vec3"],
            Value::Str(_) => &["string"],
        }
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
    fn registry_every_example_on_the_fixture_evaluates_to_its_declared_shape_or_says_not_implemented() {
        // Arrange
        let registry = Registry::builtins();
        let lookup = fixture();
        let laps = fixture_laps(&lookup);

        for e in registry.entries.iter().filter(|e| e.documented) {
            // Act
            let example = runnable(e.meta.example);
            let value = run(&registry, &lookup, &laps, &example);

            // Assert
            match (e.status, value) {
                (MathBuiltinStatus::NotImplemented, Err(error)) => assert_eq!(error.kind, MathEvalErrorKind::NotImplemented, "{example}: {error:?}"),
                (MathBuiltinStatus::NotImplemented, Ok(_)) => panic!("{example}: a not-implemented builtin returned a value"),
                (MathBuiltinStatus::Implemented, Err(error)) => panic!("{example}: {error:?}"),
                (MathBuiltinStatus::Implemented, Ok(v)) => {
                    let declared: Vec<&str> = e.meta.shape.split('|').map(str::trim).collect();
                    assert!(declared.iter().any(|d| shapes_of(&v).contains(d)), "{example}: declared {}, got {:?}", e.meta.shape, shapes_of(&v));
                }
            }
        }
    }

    #[test]
    fn registry_declared_keyword_is_accepted_and_an_undeclared_one_is_refused_with_todays_message() {
        // Arrange
        let registry = Registry::builtins();
        let lookup = fixture();
        let laps = fixture_laps(&lookup);

        for meta in registry.metas() {
            let example = runnable(meta.example);
            let open = &example[..example.len() - 1];
            let first = top_level_args(&example).first().map(|a| a.to_string()).unwrap_or_default();

            // Act
            let accepted: Vec<_> = meta.kwargs.iter()
                .map(|k| run(&registry, &lookup, &laps, &format!("{}({first}, {k}={})", meta.name, kwarg_value(meta.name, k))))
                .collect();
            let refused = if open.ends_with('(') { format!("{open}zzz=1)") } else { format!("{open}, zzz=1)") };
            let refused = run(&registry, &lookup, &laps, &refused).unwrap_err();

            // Assert
            for result in accepted {
                assert!(result.is_ok(), "{}: {:?}", meta.name, result.err());
            }
            let expected = if meta.kwargs.is_empty() {
                format!("{0}: unknown keyword argument \"zzz\" — {0} takes no keyword arguments", meta.name)
            } else {
                format!("{}: unknown keyword argument \"zzz\" (accepts: {})", meta.name, meta.kwargs.join(", "))
            };
            assert_eq!(refused.kind, MathEvalErrorKind::Runtime, "{}", meta.name);
            assert_eq!(refused.message, expected);
        }
    }

    #[test]
    fn registry_signature_argument_count_is_within_the_functions_arity() {
        // Arrange
        let registry = Registry::builtins();

        // A signature that abbreviates a run of parameters (`rotate_mat(v, m00..m22)`) can't be
        // counted from its text, so it is left out.
        for e in registry.entries.iter().filter(|e| e.documented && !e.meta.signature.contains("..")) {
            // Act — the first call form's positional parameters (`k=v` is a keyword).
            let n = top_level_args(e.meta.signature).iter().filter(|p| !p.contains('=')).count();

            // Assert
            let within = match e.arity {
                Arity::Typed(min, max) => min <= n && n <= max,
                Arity::Declared(counts) => counts.contains(&(n as u32)),
            };
            assert!(within, "{}: signature has {n}, arity {:?}", e.meta.name, e.arity);
        }
    }

    #[test]
    fn registry_catalog_is_every_documented_builtin_in_registration_order() {
        // Arrange
        let registry = Registry::builtins();

        // Act
        let names: Vec<&str> = math_builtin_catalog().iter().map(|b| b.name).collect();

        // Assert — `main` is callable but table-cell only, so not listed.
        assert_eq!(names, registry.metas().map(|m| m.name).collect::<Vec<_>>());
        assert!(!names.contains(&"main"));
        assert!(registry.index.contains_key("main"));
    }

    #[test]
    fn registry_add_of_a_name_already_registered_is_a_duplicate_name_error() {
        // Arrange
        let mut registry = Registry::builtins();

        // Act
        let result = registry.add(outside_meta("butter"), double_it).map(|_| ());

        // Assert
        assert_eq!(result, Err(RegistryError::DuplicateName("butter")));
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
        let e = run(&registry, &NoChannels, &MathLapContext::empty(), "butter(2, 5)").unwrap_err();

        // Assert
        assert_eq!(e.kind, MathEvalErrorKind::ArgCount);
        assert_eq!(e.message, "butter: expected 4 argument(s), got 2");
    }

    #[test]
    fn registry_outside_function_evaluates_and_is_absent_from_the_generated_references() {
        // Arrange
        let mut registry = Registry::builtins();
        registry.add(outside_meta("double_it"), double_it).expect("a new name");

        // Act
        let v = run(&registry, &NoChannels, &MathLapContext::empty(), "double_it(21)");
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
        registry.add(outside_meta("scale"), scale).expect("a new name");
        let laps = MathLapContext::empty();

        // Act
        let one = run(&registry, &NoChannels, &laps, "scale(2)");
        let two = run(&registry, &NoChannels, &laps, "scale(2, 3)");
        let none = run(&registry, &NoChannels, &laps, "scale()").unwrap_err();

        // Assert
        assert_eq!(one, Ok(Value::Scalar(2.0)));
        assert_eq!(two, Ok(Value::Scalar(6.0)));
        assert_eq!(none.kind, MathEvalErrorKind::ArgCount);
        assert_eq!(none.message, "scale: expected 1 to 2 argument(s), got 0");
    }
}
