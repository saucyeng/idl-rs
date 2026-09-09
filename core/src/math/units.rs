//! Unit algebra (ruling R154 / `runs/2026-09-08/unit-model.md`). A small
//! symbolic model over the unit strings C1 §4.1 already records: opaque
//! atoms (`mm`, `g`, `km`, `h`, `Hz`, …) taken verbatim, mapped to rational
//! exponents, never canonicalised and never converted. `mm` and `m` stay two
//! different atoms — that is what lets `[travel_mm] + [altitude_m]` be
//! caught, which a dimension-vector model cannot do (R154 §1(b)).
//!
//! Task 1 is the algebra (`UnitExpr`, `Ratio`) above; task 2 below adds the
//! `Unit` lattice and `infer`, a second, cheap walk of the same `Ast`
//! `evaluate` walks — not a field threaded through `Value`/`ChannelValue`,
//! per R154's "inference is a separate pass" ruling. Task 2 covers
//! literals, `[Name]`, unary/binary operators and the mismatch diagnostic;
//! every `Ast::Call` yields `Unknown(Propagated)` until task 3 installs the
//! per-function rule table.
//!
//! `g`/`pi`/`tau`/`e` survive parsing as `Ast::Constant { name, value }`
//! (ruling R162) rather than collapsing to a bare `Ast::Number` — treating
//! `g` as dimensionless would make `[body_accel] / g` infer the
//! accelerometer's own unit, a confidently wrong label R152 forbids.
//! `constant_unit` below is this module's only reader of `name`.

use std::collections::BTreeMap;
use std::fmt;

use crate::math::eval::ChannelLookup;
use crate::math::parse::{Ast, BinOp, UnOp};

/// A small rational number used as a unit exponent. `den` is always
/// positive; the fraction is kept in lowest terms. Needed because `sqrt`
/// must be total over any exponent (`g/√Hz` halves an already-negative
/// exponent) and `pow(x, n)` must be exact for any integer `n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ratio {
    num: i64,
    den: i64,
}

impl Ratio {
    /// The exponent `0` — an atom with this exponent carries no information
    /// and is dropped from a [`UnitExpr`]'s map.
    pub const ZERO: Ratio = Ratio { num: 0, den: 1 };
    /// The exponent `1` — an atom to the first power.
    pub const ONE: Ratio = Ratio { num: 1, den: 1 };

    /// Builds a rational `num/den` in lowest terms.
    ///
    /// # Panics
    /// Panics if `den` is zero — callers only ever pass a parsed integer
    /// literal as a denominator, never a computed value that could be zero.
    pub fn new(num: i64, den: i64) -> Ratio {
        assert!(den != 0, "Ratio denominator must not be zero");
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(num.unsigned_abs(), den.unsigned_abs()).max(1);
        Ratio { num: num / g as i64, den: den / g as i64 }
    }

    /// Builds the rational `n/1`.
    pub fn from_int(n: i64) -> Ratio {
        Ratio { num: n, den: 1 }
    }

    /// True when this exponent is exactly zero.
    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    /// True when this exponent is positive (used to sort an atom into a
    /// [`UnitExpr`]'s numerator vs. denominator when rendering).
    pub fn is_positive(self) -> bool {
        self.num > 0
    }

    /// True when this exponent has no fractional part — used by
    /// [`UnitExpr`]'s `Display` to render `g^2` unparenthesised but
    /// `Hz^(1/2)` parenthesised, so the fraction re-parses as one `^`
    /// fragment rather than an atom exponent followed by a stray `/den`.
    pub fn is_integer(self) -> bool {
        self.den == 1
    }

    /// The negation `-self`.
    pub fn neg(self) -> Ratio {
        Ratio { num: -self.num, den: self.den }
    }

    /// `self + other`.
    pub fn add(self, other: Ratio) -> Ratio {
        Ratio::new(self.num * other.den + other.num * self.den, self.den * other.den)
    }

    /// `self * other`, used by `pow(x, n)` and `sqrt` (`n = 1/2`) to scale
    /// every atom's exponent in a [`UnitExpr`] at once.
    pub fn mul(self, other: Ratio) -> Ratio {
        Ratio::new(self.num * other.num, self.den * other.den)
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Euclid's algorithm on unsigned magnitudes, used to keep [`Ratio`] in
/// lowest terms.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// A unit as a product of named atoms with rational exponents — `km/h` is
/// `{km: 1, h: -1}`, `g/√Hz` is `{g: 1, Hz: -1/2}`. The empty map is
/// dimensionless. Atoms are exactly the tokens C1 §4.1 records (`mm`, `g`,
/// `dps`, `pulse`, …); this type never rewrites, reorders semantically, or
/// converts between them — only algebraically combines exponents on
/// matching atoms. The `BTreeMap` keeps atoms in a canonical (alphabetical)
/// order, so two `UnitExpr`s built in different orders compare and render
/// identically.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitExpr(BTreeMap<String, Ratio>);

/// Why a unit string failed to parse. Carried so a caller (task 2's
/// inference pass) can turn it into an `Unknown` reason rather than a panic
/// — a malformed unit string in a session must never crash evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitParseError {
    /// The string was empty — C1 §4.1 uses `""` for "no unit recorded",
    /// which is a distinct case from a dimensionless [`UnitExpr`] and is
    /// handled by the caller, not by this parser.
    Empty,
    /// A factor had no atom name (e.g. `"^2"`, `"/h"`, or a stray `·`).
    MissingAtom(String),
    /// A `^` exponent was not an integer or `n/d` rational literal.
    BadExponent(String),
}

impl fmt::Display for UnitParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitParseError::Empty => write!(f, "unit string is empty"),
            UnitParseError::MissingAtom(s) => write!(f, "no atom name in unit fragment '{s}'"),
            UnitParseError::BadExponent(s) => write!(f, "not a valid exponent: '{s}'"),
        }
    }
}

impl std::error::Error for UnitParseError {}

impl UnitExpr {
    /// The dimensionless unit — the empty product.
    pub fn dimensionless() -> UnitExpr {
        UnitExpr(BTreeMap::new())
    }

    /// The unit consisting of a single atom to the first power (e.g.
    /// `UnitExpr::atom("mm")` is `mm`).
    pub fn atom(name: &str) -> UnitExpr {
        let mut m = BTreeMap::new();
        m.insert(name.to_string(), Ratio::ONE);
        UnitExpr(m)
    }

    /// True when this is the empty product — no atoms at all.
    pub fn is_dimensionless(&self) -> bool {
        self.0.is_empty()
    }

    /// Parses a C1 §4.1 unit string (`"mm"`, `"km/h"`, `"g"`, …) into its
    /// atom/exponent map. Grammar: a term is one or more atoms joined by
    /// `·` or `*`, each optionally raised to an integer or `n/d` exponent
    /// with `^` (e.g. `Hz^2`, `Hz^(1/2)`); a unit is one term, or two terms
    /// separated by a single `/` (numerator and denominator). This is a
    /// superset of every string C1 §4.1 emits today (a bare atom, or
    /// `atom/atom`) and is also the grammar [`UnitExpr::to_string`]
    /// renders, so parse and render round-trip on inferred compound units
    /// too.
    ///
    /// Never converts and never validates that an atom is a "real" unit —
    /// a typo'd atom parses fine and simply fails to equal the atom the
    /// author meant (R154 §1(c), point 3).
    pub fn parse(s: &str) -> Result<UnitExpr, UnitParseError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(UnitParseError::Empty);
        }
        let mut halves = s.splitn(2, '/');
        let numer = halves.next().unwrap();
        let denom = halves.next();

        let mut map = parse_term(numer)?;
        if let Some(d) = denom {
            for (atom, exp) in parse_term(d)? {
                let entry = map.entry(atom).or_insert(Ratio::ZERO);
                *entry = entry.add(exp.neg());
            }
        }
        map.retain(|_, exp| !exp.is_zero());
        Ok(UnitExpr(map))
    }

    /// `self * other` — exponents on matching atoms add, an atom present in
    /// only one operand keeps its exponent, and an atom whose combined
    /// exponent reaches zero is dropped (so `[a]/[a]` renders as
    /// dimensionless, and `mm·s/s` renders as `mm`).
    pub fn mul(&self, other: &UnitExpr) -> UnitExpr {
        self.combine(other, false)
    }

    /// `self / other` — `other`'s exponents are negated before combining,
    /// so `mm / (mm/s)` is `s`.
    pub fn div(&self, other: &UnitExpr) -> UnitExpr {
        self.combine(other, true)
    }

    fn combine(&self, other: &UnitExpr, negate_other: bool) -> UnitExpr {
        let mut map = self.0.clone();
        for (atom, exp) in &other.0 {
            let exp = if negate_other { exp.neg() } else { *exp };
            let entry = map.entry(atom.clone()).or_insert(Ratio::ZERO);
            *entry = entry.add(exp);
        }
        map.retain(|_, exp| !exp.is_zero());
        UnitExpr(map)
    }

    /// Raises every atom's exponent to the power `n` — `pow(x, n)`'s unit
    /// rule. `n` is a [`Ratio`] so `sqrt` (`n = 1/2`) reuses this directly.
    /// The dimensionless unit raised to any power stays dimensionless.
    pub fn pow(&self, n: Ratio) -> UnitExpr {
        let mut map: BTreeMap<String, Ratio> =
            self.0.iter().map(|(atom, exp)| (atom.clone(), exp.mul(n))).collect();
        map.retain(|_, exp| !exp.is_zero());
        UnitExpr(map)
    }

    /// `self^(1/2)` — halves every exponent. Total: `g/√Hz` (`{g: 1, Hz:
    /// -1/2}`) is a real accelerometer-PSD unit, not a degenerate case,
    /// which is why exponents are rational rather than integer.
    pub fn sqrt(&self) -> UnitExpr {
        self.pow(Ratio::new(1, 2))
    }
}

/// Parses one side of a `/` split — one or more atoms joined by `·`/`*`,
/// each with an optional `^exponent`.
fn parse_term(s: &str) -> Result<BTreeMap<String, Ratio>, UnitParseError> {
    let mut map: BTreeMap<String, Ratio> = BTreeMap::new();
    for factor in s.split(['·', '*']) {
        let factor = factor.trim();
        let (atom, exp) = parse_factor(factor)?;
        let entry = map.entry(atom).or_insert(Ratio::ZERO);
        *entry = entry.add(exp);
    }
    Ok(map)
}

/// Parses one atom, with its optional `^exponent` suffix.
fn parse_factor(s: &str) -> Result<(String, Ratio), UnitParseError> {
    let (atom, exp) = match s.find('^') {
        Some(idx) => (&s[..idx], parse_exponent(&s[idx + 1..])?),
        None => (s, Ratio::ONE),
    };
    if atom.is_empty() {
        return Err(UnitParseError::MissingAtom(s.to_string()));
    }
    Ok((atom.to_string(), exp))
}

/// Parses a `^` suffix: a bare integer (`"2"`, `"-1"`) or a parenthesised
/// `n/d` rational (`"(1/2)"`).
fn parse_exponent(s: &str) -> Result<Ratio, UnitParseError> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    if let Some((n, d)) = inner.split_once('/') {
        let n: i64 = n.trim().parse().map_err(|_| UnitParseError::BadExponent(s.to_string()))?;
        let d: i64 = d.trim().parse().map_err(|_| UnitParseError::BadExponent(s.to_string()))?;
        if d == 0 {
            return Err(UnitParseError::BadExponent(s.to_string()));
        }
        Ok(Ratio::new(n, d))
    } else {
        let n: i64 = inner.parse().map_err(|_| UnitParseError::BadExponent(s.to_string()))?;
        Ok(Ratio::from_int(n))
    }
}

impl fmt::Display for UnitExpr {
    /// Renders `mm` as `"mm"`, `{km:1, h:-1}` as `"km/h"`, `{g:1, Hz:-1,2}`
    /// (i.e. `Hz` to `-1/2`) as `"g/Hz^(1/2)"`. Atoms within the numerator
    /// and within the denominator are joined in the map's canonical
    /// (alphabetical) order — this does not reorder how C1 emits a single
    /// bare or `a/b` unit, since those have at most one atom per side.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut numer: Vec<(&str, Ratio)> = Vec::new();
        let mut denom: Vec<(&str, Ratio)> = Vec::new();
        for (atom, exp) in &self.0 {
            if exp.is_positive() {
                numer.push((atom, *exp));
            } else {
                denom.push((atom, exp.neg()));
            }
        }

        if numer.is_empty() && denom.is_empty() {
            return write!(f, "");
        }

        let render_side = |side: &[(&str, Ratio)]| -> String {
            side.iter()
                .map(|(atom, exp)| {
                    if *exp == Ratio::ONE {
                        atom.to_string()
                    } else if exp.is_integer() {
                        format!("{atom}^{exp}")
                    } else {
                        // A fractional exponent is parenthesised so it
                        // re-parses as one `^` fragment, not `atom^n` then
                        // a stray `/den` (see parse_exponent's grammar).
                        format!("{atom}^({exp})")
                    }
                })
                .collect::<Vec<_>>()
                .join("·")
        };

        if denom.is_empty() {
            write!(f, "{}", render_side(&numer))
        } else if numer.is_empty() {
            write!(f, "1/{}", render_side(&denom))
        } else {
            write!(f, "{}/{}", render_side(&numer), render_side(&denom))
        }
    }
}

/// The unit lattice used during inference (R154 §1). `Scalar` is internal
/// only — it never crosses the wire; a top-level `Scalar` result reports
/// dimensionless (task 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unit {
    /// A bare numeric literal, or arithmetic between literals only. Adapts
    /// to the other operand's unit under `+ - min max clamp if`;
    /// dimensionless under `* /`. Without this state, a literal is either
    /// dimensionless (so every offset expression like `[travel] + 10`
    /// reports a mismatch) or unknown (so every workbook loses its unit at
    /// the first constant) — neither is honest.
    Scalar,
    /// A determined unit. `Known(UnitExpr::dimensionless())` is genuinely
    /// dimensionless (a ratio, a count, a comparison result) — distinct
    /// from `Scalar`, which is a number that has not yet met a unit.
    Known(UnitExpr),
    /// Could not be worked out; carries why, for display (task 4).
    Unknown(UnknownReason),
}

/// Why [`Unit::infer`] could not determine a unit. See
/// `runs/2026-09-08/unit-model.md` §4 — this set is closed; a new source of
/// "unknown" is a design decision, not a call this module makes locally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownReason {
    /// The channel's C1 §4.1 unit is `""` (no unit recorded), or a
    /// `{cell}`/`{col[]}` reference (which has no recorded unit at all).
    NoSourceUnit,
    /// A `+`/`-` (or another unit-checked construct) combined two `Known`
    /// operands whose units differ. `op` is the operator's display text,
    /// e.g. `"+"`.
    Mismatch { left: UnitExpr, right: UnitExpr, op: &'static str },
    /// `pow(x, n)` where `n` is not a literal and `x`'s base is not
    /// dimensionless — task 3's function table produces this; task 2 does
    /// not yet, since every call yields `Propagated`.
    NonLiteralExponent,
    /// An operand was already `Unknown`; `of` names the channel or
    /// sub-expression it started from (best-effort, for display — not
    /// guaranteed unique). Also used, for now, as the yield of every
    /// `Ast::Call` — task 3 replaces that with a real per-function rule.
    Propagated { of: String },
    /// A math-definition dependency cycle (task 4, once inference runs in
    /// dependency order over `resolve_workbook_defs`). Unused by this
    /// module's own `infer`, which does not walk definition references.
    Cycle,
    /// The AST node is not a numeric construct (currently: a string
    /// literal, which only appears as a function argument, e.g.
    /// `spectrogram(x, "density")`). Not one of the design's five listed
    /// reasons — added because `infer` must be total over any `Ast` node
    /// reachable from a `Call`'s `args`, and a bare string has no unit to
    /// report. Small, safe judgment call (CLAUDE.md §1); flagged for the
    /// lead rather than folded silently into an existing reason.
    NotNumeric,
}

/// A non-fatal unit diagnostic produced during inference — most often the
/// `+`/`-` mismatch of R154 §2.1. Never an evaluation failure: the value in
/// `CellDefResult::value` (task 4) is unaffected by a `UnitNote`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitNote {
    /// Display-ready English, e.g. "`+`: units differ (`bpm` and `km/h`);
    /// result unit withheld".
    pub message: String,
}

impl UnknownReason {
    /// Display-ready English for this reason — what `UnitLabel::Unknown`'s
    /// `reason` (and, later, a UI tooltip) shows. Never references internal
    /// type names; a rider or mechanic reads this, not a developer.
    pub fn describe(&self) -> String {
        match self {
            UnknownReason::NoSourceUnit => "no unit recorded for this channel".to_string(),
            UnknownReason::Mismatch { left, right, op } => {
                format!("`{op}`: units differ (`{left}` and `{right}`)")
            }
            UnknownReason::NonLiteralExponent => {
                "exponent is not a literal number".to_string()
            }
            UnknownReason::Propagated { of } => format!("unit of {of} could not be determined"),
            UnknownReason::Cycle => "definition cycle".to_string(),
            UnknownReason::NotNumeric => "not a numeric value".to_string(),
        }
    }
}

/// The unit as it crosses to the outer layers (R154 §5) — a three-state
/// rendering of [`Unit`] with `Scalar` folded into `Dimensionless`, since
/// `Scalar` is inference-internal only (it never crosses the wire; a
/// top-level `Scalar` result, e.g. `count(x)`, is genuinely dimensionless as
/// far as any consumer is concerned). `unit: string | null` must not ship in
/// its place (R154) — `None` cannot mean both "not applicable" and "we
/// could not work it out".
// `serde::Serialize` here (unusually, for a `core` type — see CLAUDE.md
// §2) mirrors `HostChannel`'s own already-established exception
// (`core/src/workbook/v3/host.rs`): a host-variable value crosses to the
// JS sandbox as JSON via `postMessage`, never through Tauri IPC, so there
// is no `idl-rs-tauri` layer between this type and its wire form the way
// `CellDefResult`'s own `UnitLabel` mirror (`tauri/src/commands/workbook.rs`)
// has. Same shape as that mirror (`tag = "state"`) so the two paths render
// identically JS-side.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum UnitLabel {
    /// A determined, non-empty unit, rendered from [`UnitExpr`] — `"mm"`,
    /// `"km/h"`. A struct variant (not a newtype) because
    /// `#[serde(tag = "state")]` (internal tagging, for a clean
    /// discriminated union on the JS side) requires one — a tuple variant
    /// cannot be internally tagged.
    Known { text: String },
    /// Genuinely no unit: a ratio, a count, a comparison result, or a
    /// top-level [`Unit::Scalar`].
    Dimensionless,
    /// Not determinable. `reason` is [`UnknownReason::describe`]'s text.
    Unknown { reason: String },
}

impl From<&Unit> for UnitLabel {
    fn from(unit: &Unit) -> UnitLabel {
        match unit {
            Unit::Scalar => UnitLabel::Dimensionless,
            Unit::Known(u) if u.is_dimensionless() => UnitLabel::Dimensionless,
            Unit::Known(u) => UnitLabel::Known { text: u.to_string() },
            Unit::Unknown(reason) => UnitLabel::Unknown { reason: reason.describe() },
        }
    }
}

/// `a op b`'s unit under `+`/`-` (R154 §2's table row for `a + b`, `a - b`):
/// `Scalar` on either side adopts the other's unit; two different `Known`s
/// mismatch; anything touching `Unknown` propagates. Shared by both `Add`
/// and `Sub`, which have the same unit rule.
fn add_sub_unit(left: Unit, right: Unit, op: &'static str) -> (Unit, Option<UnitNote>) {
    match (left, right) {
        (Unit::Scalar, Unit::Scalar) => (Unit::Scalar, None),
        (Unit::Scalar, Unit::Known(u)) | (Unit::Known(u), Unit::Scalar) => (Unit::Known(u), None),
        (Unit::Known(l), Unit::Known(r)) if l == r => (Unit::Known(l), None),
        (Unit::Known(l), Unit::Known(r)) => {
            let note = UnitNote {
                message: format!(
                    "`{op}`: units differ (`{l}` and `{r}`); result unit withheld"
                ),
            };
            (Unit::Unknown(UnknownReason::Mismatch { left: l, right: r, op }), Some(note))
        }
        (Unit::Unknown(_), other) | (other, Unit::Unknown(_)) => {
            (Unit::Unknown(UnknownReason::Propagated { of: describe(&other) }), None)
        }
    }
}

/// A short, best-effort label for a `Unit` used only inside a `Propagated`
/// reason's `of` field — not a full render, just enough for a diagnostic to
/// point somewhere.
fn describe(u: &Unit) -> String {
    match u {
        Unit::Scalar => "a scalar".to_string(),
        Unit::Known(expr) if expr.is_dimensionless() => "a dimensionless value".to_string(),
        Unit::Known(expr) => expr.to_string(),
        Unit::Unknown(_) => "an unknown-unit value".to_string(),
    }
}

/// `a * b` / `a / b`'s unit (R154 §2): `Scalar` combines as dimensionless,
/// `Known` combines via [`UnitExpr::mul`]/[`UnitExpr::div`], anything
/// touching `Unknown` propagates. `divide` selects `*` vs `/`.
fn mul_div_unit(left: Unit, right: Unit, divide: bool) -> Unit {
    let combine = |l: &UnitExpr, r: &UnitExpr| if divide { l.div(r) } else { l.mul(r) };
    match (left, right) {
        (Unit::Scalar, Unit::Scalar) => Unit::Scalar,
        (Unit::Scalar, Unit::Known(u)) => {
            Unit::Known(combine(&UnitExpr::dimensionless(), &u))
        }
        (Unit::Known(u), Unit::Scalar) => Unit::Known(u),
        (Unit::Known(l), Unit::Known(r)) => Unit::Known(combine(&l, &r)),
        (Unit::Unknown(_), other) | (other, Unit::Unknown(_)) => {
            Unit::Unknown(UnknownReason::Propagated { of: describe(&other) })
        }
    }
}

/// The unit of a universal named constant (R162): `pi`/`tau`/`e` are
/// genuinely dimensionless numbers; `g` is standard gravity and carries
/// `m/s²` (C2 §3.2). `name` is always one of `constant_value`'s four
/// canonical names — an `Ast::Constant` the parser produces can never carry
/// anything else — so an unrecognised name here is unreachable in practice,
/// not a case this function guesses at.
fn constant_unit(name: &str) -> Unit {
    match name {
        "g" => Unit::Known(UnitExpr::parse("m/s^2").expect("m/s^2 is a valid unit string")),
        _ => Unit::Known(UnitExpr::dimensionless()),
    }
}

/// Infers the unit of `ast` by walking it once, resolving `[Name]`
/// references against `lookup` (`ChannelLookup::unit_of`, R154). Returns
/// the inferred [`Unit`] plus any non-fatal diagnostics collected from
/// nested sub-expressions (R154 §2.1) — a diagnostic never stops this
/// function from also returning a `Unit` for the whole tree.
///
/// This is task 2 of the design's 8-task plan: literals, `[Name]`, unary
/// `-`/`not`, `+ - * /`, comparisons, `and`/`or`. Every `Ast::Call` yields
/// `Unknown(Propagated)` regardless of function — task 3 installs the real
/// per-function rule table. Does not resolve a math-definition `[Name]` to
/// that definition's own inferred unit (R154 §3's second `[Name]` row) —
/// that requires dependency-order memoization over the workbook and is
/// task 4's `resolve.rs`/`eval.rs` wiring, not this per-expression pass.
pub fn infer(ast: &Ast, lookup: &dyn ChannelLookup) -> (Unit, Vec<UnitNote>) {
    match ast {
        Ast::Number(_) => (Unit::Scalar, Vec::new()),
        Ast::Constant { name, .. } => (constant_unit(name), Vec::new()),
        Ast::Str(_) => (Unit::Unknown(UnknownReason::NotNumeric), Vec::new()),
        Ast::ChannelRef(name) => {
            let unit = match lookup.unit_of(name) {
                Some(s) => match UnitExpr::parse(&s) {
                    Ok(expr) => Unit::Known(expr),
                    // A malformed C1 unit string is treated the same as no
                    // unit at all — never a panic, never a guess at what
                    // the author meant.
                    Err(_) => Unit::Unknown(UnknownReason::NoSourceUnit),
                },
                None => Unit::Unknown(UnknownReason::NoSourceUnit),
            };
            (unit, Vec::new())
        }
        Ast::CellRef(_) => (Unit::Unknown(UnknownReason::NoSourceUnit), Vec::new()),
        Ast::Unary { op: UnOp::Neg, expr } => infer(expr, lookup),
        Ast::Unary { op: UnOp::Not, expr } => {
            let (_, notes) = infer(expr, lookup);
            (Unit::Known(UnitExpr::dimensionless()), notes)
        }
        Ast::Binary { op, left, right } => {
            let (lu, mut notes) = infer(left, lookup);
            let (ru, rnotes) = infer(right, lookup);
            notes.extend(rnotes);
            match op {
                BinOp::Add | BinOp::Sub => {
                    let op_sym = if *op == BinOp::Add { "+" } else { "-" };
                    let (unit, note) = add_sub_unit(lu, ru, op_sym);
                    notes.extend(note);
                    (unit, notes)
                }
                BinOp::Mul => (mul_div_unit(lu, ru, false), notes),
                BinOp::Div => (mul_div_unit(lu, ru, true), notes),
                BinOp::Lt
                | BinOp::Gt
                | BinOp::LtEq
                | BinOp::GtEq
                | BinOp::EqEq
                | BinOp::BangEq => {
                    // The result is always Known(dimensionless) — a
                    // comparison is a truth value — but the operands are
                    // still checked, so a `mm` vs. `m` comparison raises
                    // the same mismatch diagnostic `+` would, purely for
                    // its side-effect note.
                    let (_, note) = add_sub_unit(lu, ru, comparison_symbol(*op));
                    notes.extend(note);
                    (Unit::Known(UnitExpr::dimensionless()), notes)
                }
                BinOp::And | BinOp::Or => {
                    // Operands are truthiness tests, not checked against
                    // each other — only their own nested notes (already
                    // collected above) surface.
                    (Unit::Known(UnitExpr::dimensionless()), notes)
                }
            }
        }
        Ast::Call { name, args, kwargs } => {
            // Task 3 (C2 §3.3/§3.3.1, transcribed — R162's own ruling that
            // the rule column, not this table, is normative where the two
            // could ever disagree): each positional argument is inferred
            // first (so a nested mismatch always surfaces, function rule
            // or not), then the named function's own rule combines them.
            let mut notes = Vec::new();
            let arg_units: Vec<Unit> = args
                .iter()
                .map(|arg| {
                    let (u, arg_notes) = infer(arg, lookup);
                    notes.extend(arg_notes);
                    u
                })
                .collect();
            for (_, expr) in kwargs {
                let (_, kw_notes) = infer(expr, lookup);
                notes.extend(kw_notes);
            }

            // `pow(x, y)` is the one rule whose branch depends on `y`'s own
            // literal *value*, not any argument's unit — not modelled as a
            // generic `FnUnitRule` case (nothing else needs that), handled
            // directly here instead.
            let unit = if name == "pow" {
                pow_call_unit(&arg_units, args.get(1))
            } else {
                let (rule, checks) = function_unit_rule(name, args.len());
                apply_fn_checks(&checks, &arg_units, &mut notes);
                let (unit, rule_notes) = eval_fn_rule(&rule, &arg_units, kwargs);
                notes.extend(rule_notes);
                unit
            };
            (unit, notes)
        }
    }
}

/// A named-function unit rule (C2 §3.3.1) — this module's transcription of
/// §3.3's **Unit rule** column, which is the normative source (R162): where
/// this table and that column could ever disagree, the column wins and
/// this table is the bug. Argument positions are 0-based over the
/// positional arguments, matching §3.3.1 exactly.
enum FnUnitRule {
    /// The output carries positional argument `n`'s unit.
    SameAsArg(usize),
    /// A constant unit regardless of the arguments — a C1-style string,
    /// parsed once per evaluation (not cached: this is metadata inference,
    /// never the hot per-sample path).
    Fixed(&'static str),
    /// The empty unit product. Never a stand-in for "we don't know".
    Dimensionless,
    /// Exponents added.
    Product(Box<FnUnitRule>, Box<FnUnitRule>),
    /// Exponents subtracted.
    Quotient(Box<FnUnitRule>, Box<FnUnitRule>),
    /// The inner rule's exponents multiplied by the rational `k`.
    PowN(Box<FnUnitRule>, Ratio),
    /// Every listed positional argument must share one unit, which is also
    /// the output's — a mismatch is the §3.3.1 diagnostic (a `UnitNote`)
    /// and the output is `Unknown(Mismatch)`.
    AllMatch(Vec<usize>),
    /// A string-literal **keyword** argument (`kwarg`) selects which rule
    /// applies. Absent → `default`'s rule. Present but not a string
    /// literal → `Unknown(NonLiteralExponent)`-shaped (this module's
    /// nearest existing reason; §3.3.1 calls it "non-literal selector",
    /// the same family of problem). Present, a literal, but not one of
    /// `cases` — not reachable through a workbook that also evaluates,
    /// since `eval`'s own scaling parser rejects it first; falls back to
    /// `default`'s rule rather than inventing a new reason for a state
    /// that cannot survive to a rendered cell.
    SelectByLiteral { kwarg: &'static str, cases: Vec<(&'static str, FnUnitRule)>, default: &'static str },
    /// Not derivable — reserved for a function this table cannot yet rule
    /// on (none, today; §3.3 covers all 72).
    Unknown,
}

/// A non-fatal §3.3.1 "check" — constrains what an argument *should* be
/// without ever changing the output unit or the evaluated value. A
/// violation is a `UnitNote`, the same non-fatal treatment as a mismatched
/// `+` (R154 §2.1).
enum FnUnitCheck {
    /// Argument `n`, if `Known`, should be the atom `unit`.
    Expect(usize, &'static str),
    /// Argument `n`, if `Known`, should be dimensionless.
    ExpectDimensionless(usize),
    /// Arguments `a` and `b`, if both `Known`, should share a unit.
    SameUnit(usize, usize),
}

fn same(n: usize) -> Box<FnUnitRule> {
    Box::new(FnUnitRule::SameAsArg(n))
}

fn fixed(u: &'static str) -> Box<FnUnitRule> {
    Box::new(FnUnitRule::Fixed(u))
}

/// C2 §3.3's per-function unit rule, by name and positional argument count
/// (only `min`/`max` vary by count — 1-arg aggregate vs. 2-arg
/// elementwise). Every one of the 72 `math_builtin_catalog` entries has a
/// case; task 3's own catalog-completeness test (below) fails loudly if a
/// future function is added here without one, rather than silently
/// defaulting every unlisted name to `Unknown`.
fn function_unit_rule(name: &str, arg_count: usize) -> (FnUnitRule, Vec<FnUnitCheck>) {
    use FnUnitRule::*;
    let no_checks = Vec::new();
    match name {
        "butter" => (SameAsArg(3), no_checks),
        "sosfilt" => (SameAsArg(1), no_checks),
        "declip" => (SameAsArg(0), no_checks),
        "cumulative_trapezoid" | "cumtrapz" => (Product(same(0), fixed("s")), no_checks),
        "differentiate" | "gradient" => (Quotient(same(0), fixed("s")), no_checks),
        "detrend" => (SameAsArg(0), no_checks),
        "rms" | "mean" | "std" => (SameAsArg(0), no_checks),
        "median" => (SameAsArg(0), no_checks),
        "sum" => (SameAsArg(0), no_checks),
        "count" => (Dimensionless, no_checks),
        "first" | "last" => (SameAsArg(0), no_checks),
        "percentile" => (SameAsArg(0), no_checks),
        "abs" => (SameAsArg(0), no_checks),
        "sqrt" => (PowN(same(0), Ratio::new(1, 2)), no_checks),
        "sign" => (Dimensionless, no_checks),
        "floor" | "ceil" | "round" => (SameAsArg(0), no_checks),
        "pow" => (pow_rule(), no_checks),
        "min" | "max" => {
            if arg_count <= 1 {
                (SameAsArg(0), no_checks)
            } else {
                (AllMatch(vec![0, 1]), no_checks)
            }
        }
        "clip" => (SameAsArg(0), no_checks),
        "sin" | "cos" | "tan" => (Dimensionless, vec![FnUnitCheck::Expect(0, "rad")]),
        "asin" | "acos" | "atan" => (Fixed("rad"), vec![FnUnitCheck::ExpectDimensionless(0)]),
        "atan2" => (Fixed("rad"), vec![FnUnitCheck::SameUnit(0, 1)]),
        "sinh" | "cosh" | "tanh" => (Dimensionless, vec![FnUnitCheck::ExpectDimensionless(0)]),
        "deg2rad" => (Fixed("rad"), vec![FnUnitCheck::Expect(0, "deg")]),
        "rad2deg" => (Fixed("deg"), vec![FnUnitCheck::Expect(0, "rad")]),
        "periodogram" | "welch" | "spectrogram" => (spectral_rule(), no_checks),
        // Retired from `hilbert` (R151 item 8, C2 3.8): with no complex type
        // in the language, this was always going to return an envelope, not
        // scipy's analytic signal — the scipy name was a false friend before
        // a single line of it existed (R146). `NotImplemented`, so the
        // rename cost zero migration.
        "envelope" => (SameAsArg(0), no_checks),
        "correlate" | "convolve" => (Product(same(0), same(1)), no_checks),
        // `resample(ch, n)` — `n` is a target sample count (scipy's own
        // parameterisation), not a rate (R146/R151 item 8): a count is
        // `Dimensionless` and carries no unit check of its own, so this rule
        // is unaffected by the rename — only the C2 §3.3 signature/prose
        // changed, not the argument this rule reads.
        "resample" => (SameAsArg(0), no_checks),
        "where" => (AllMatch(vec![1, 2]), no_checks),
        "current_lap" => (Dimensionless, no_checks),
        "lap_start_time" => (Fixed("s"), no_checks),
        "lap_start_distance" => (Fixed("m"), no_checks),
        "sector_number" => (Dimensionless, no_checks),
        // A difference of two [ch] series, not a duration/distance —
        // corrected from the original brief's Fixed(s)/Fixed(m) (lead
        // ruling, this dispatch's message).
        "lap_delta_time" | "lap_delta_dist" => (SameAsArg(0), no_checks),
        "attitude" => (Fixed("deg"), no_checks),
        "body_accel" => (Fixed("g"), no_checks),
        "wheel_travel" => (Fixed("mm"), no_checks),
        "wheel_velocity" => (Fixed("mm/s"), no_checks),
        "vec" => (AllMatch(vec![0, 1, 2]), no_checks),
        "vx" | "vy" | "vz" => (SameAsArg(0), no_checks),
        "vadd" | "vsub" => (AllMatch(vec![0, 1]), no_checks),
        "vscale" => (Product(same(0), same(1)), no_checks),
        "cross" | "dot" => (Product(same(0), same(1)), no_checks),
        // Formally PowN(Product(0, 0), 1/2) (§3.3.1); simplified here to
        // the equivalent SameAsArg(0) — squaring an atom's exponents then
        // halving them is the identity, and a Vec3 carries exactly one
        // unit, so there is nothing PowN/Product could disagree with
        // SameAsArg about. Documented rather than expanded literally,
        // since expanding it changes no observable output.
        "norm" => (SameAsArg(0), no_checks),
        "normalize" => (Dimensionless, no_checks),
        // Formally Fixed(rad) because atan2's Product(0, 1) operands are
        // identical and cancel (§3.3.1); the cancellation has no
        // observable effect, so this is Fixed(rad) directly.
        "angle_between" => (Fixed("rad"), no_checks),
        "rotate_mat" | "rotate_axis" | "rotate_euler" => (SameAsArg(0), no_checks),
        _ => (Unknown, no_checks),
    }
}

/// `pow(x, y)`'s catalog placeholder — never evaluated. `infer`'s `Ast::Call`
/// arm special-cases `"pow"` before ever consulting `function_unit_rule`
/// (see [`pow_call_unit`]), because `y`'s value, not any argument's *unit*,
/// selects the branch (§3.3: `PowN(0, y)` when `y` is a numeric literal;
/// otherwise `Dimensionless` if `x` is dimensionless, else
/// `Unknown(NonLiteralExponent)`) — no other rule needs a raw `Ast` value,
/// so it is not modelled generically in `FnUnitRule`. This entry exists
/// only so the catalog-completeness test below finds `"pow"` accounted
/// for, not `Unknown` by omission.
fn pow_rule() -> FnUnitRule {
    FnUnitRule::Unknown
}

/// `pow(x, y)`'s real rule (§3.3), computed directly from the call's own
/// `Ast` rather than through [`FnUnitRule`] (see [`pow_rule`]'s doc
/// comment for why). `y_ast` is `args.get(1)` — absent when `pow` was
/// called with the wrong arity, which is a separate `ArgCount` evaluation
/// error this function does not need to duplicate; it simply reports
/// `Unknown` in that case.
fn pow_call_unit(arg_units: &[Unit], y_ast: Option<&Ast>) -> Unit {
    let x = arg_units.first().cloned().unwrap_or(Unit::Unknown(UnknownReason::NoSourceUnit));
    let literal_exponent = match y_ast {
        Some(Ast::Number(y)) => ratio_from_f64(*y),
        _ => None,
    };
    match literal_exponent {
        Some(k) => pow_unit(x, k),
        None => match x {
            Unit::Scalar => Unit::Scalar,
            Unit::Known(u) if u.is_dimensionless() => Unit::Known(u),
            _ => Unit::Unknown(UnknownReason::NonLiteralExponent),
        },
    }
}

/// Recovers a small exact [`Ratio`] from a literal `pow` exponent, when one
/// exists within floating-point tolerance — an integer (`2`, `-1`) or a
/// half-integer (`0.5`, `-1.5`, the `sqrt`/`cbrt`-shaped exponents authors
/// actually write via `pow` instead of `sqrt`). `y` is an arbitrary `f64`
/// in general (any workbook literal), which has no exact rational form in
/// general — rather than guess a rational for something like `0.301` (an
/// author's `log10`-flavoured exponent, not really an exact rational),
/// this recovers only the two shapes real workbooks demonstrably need and
/// reports `None` otherwise, which `pow_call_unit` turns into the honest
/// `Unknown(NonLiteralExponent)` rather than a wrong-looking unit. A
/// judgment call (CLAUDE.md §1) narrower than "any rational `y`", not a
/// guess at what that broader rule should be.
fn ratio_from_f64(y: f64) -> Option<Ratio> {
    const TOL: f64 = 1e-9;
    if (y - y.round()).abs() < TOL {
        return Some(Ratio::from_int(y.round() as i64));
    }
    let doubled = y * 2.0;
    if (doubled - doubled.round()).abs() < TOL {
        return Some(Ratio::new(doubled.round() as i64, 2));
    }
    None
}

/// `periodogram`/`welch`/`spectrogram`'s shared three-scaling rule (§3.3.1):
/// `scaling="density"` → `[ch]²/Hz`, `"spectrum"` → `[ch]²`,
/// `"raw_magnitude"` → `[ch]` (magnitude, unnormalised). Default
/// `"density"`, matching the engine's own default. `spectrogram`'s rule is
/// contingent (§3.3.1 Open 1 — it is `NotImplemented`, so this is read off
/// `periodogram`'s prose, not an implementation) and shares this function
/// rather than duplicating it, so the day it lands the two do not silently
/// drift apart.
fn spectral_rule() -> FnUnitRule {
    FnUnitRule::SelectByLiteral {
        kwarg: "scaling",
        cases: vec![
            ("density", FnUnitRule::Quotient(Box::new(FnUnitRule::PowN(same(0), Ratio::from_int(2))), fixed("Hz"))),
            ("spectrum", FnUnitRule::PowN(same(0), Ratio::from_int(2))),
            ("raw_magnitude", FnUnitRule::SameAsArg(0)),
        ],
        default: "density",
    }
}

/// `periodogram`/`welch`/`spectrogram`'s output unit (§3.3.1), for a caller
/// that reaches a spectrum without walking an `Ast` — the raster/`fetch_fft`
/// path (`idl_rs::fft::welch`/`spectrogram`, called directly on a channel's
/// materialized samples) has no math-cell call site for [`infer`] to visit,
/// so it cannot see [`function_unit_rule`] (a private, `Ast`-shaped table).
/// This reuses [`spectral_rule`] itself — never a second hard-coded copy of
/// §3.3.1's table (R163) — over a synthesized single-argument unit and a
/// synthesized `scaling` keyword, so the two paths can never drift apart.
///
/// `source_unit` is the source channel's raw C1 §4.1 unit string (`""` means
/// no unit recorded, the same `NoSourceUnit` reason a `[Name]` reference with
/// no unit gets); `scaling` is one of §3.3.1's three literals (`"density"`,
/// `"spectrum"`, `"raw_magnitude"`) — an absent selection is the caller's own
/// choice to omit, expressed the same way `infer` sees an omitted keyword
/// argument: pass `"density"`, the rule's own default, explicitly.
pub fn spectral_output_unit(source_unit: &str, scaling: &str) -> UnitLabel {
    let source = if source_unit.is_empty() {
        Unit::Unknown(UnknownReason::NoSourceUnit)
    } else {
        match UnitExpr::parse(source_unit) {
            Ok(expr) => Unit::Known(expr),
            // A malformed C1 unit string is treated the same as no unit at
            // all — the same fallback `infer`'s own `Ast::ChannelRef` arm
            // takes, never a panic or a guess at what the author meant.
            Err(_) => Unit::Unknown(UnknownReason::NoSourceUnit),
        }
    };
    let kwargs = vec![("scaling".to_string(), Ast::Str(scaling.to_string()))];
    let (unit, _notes) = eval_fn_rule(&spectral_rule(), &[source], &kwargs);
    UnitLabel::from(&unit)
}

/// Reads a keyword argument's value as a string literal, if `kwargs`
/// carries `name` and its value is a bare `Ast::Str`.
fn literal_kwarg<'a>(kwargs: &'a [(String, Ast)], name: &str) -> Option<&'a str> {
    kwargs.iter().find(|(k, _)| k == name).and_then(|(_, v)| match v {
        Ast::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

/// Evaluates `rule` against `arg_units` (already-inferred positional
/// argument units) and `kwargs` (for `SelectByLiteral`'s selector).
fn eval_fn_rule(rule: &FnUnitRule, arg_units: &[Unit], kwargs: &[(String, Ast)]) -> (Unit, Vec<UnitNote>) {
    match rule {
        FnUnitRule::SameAsArg(n) => (
            arg_units
                .get(*n)
                .cloned()
                .unwrap_or_else(|| Unit::Unknown(UnknownReason::Propagated { of: "a missing argument".to_string() })),
            Vec::new(),
        ),
        FnUnitRule::Fixed(u) => (
            Unit::Known(UnitExpr::parse(u).unwrap_or_else(|_| panic!("fixed unit literal '{u}' must parse"))),
            Vec::new(),
        ),
        FnUnitRule::Dimensionless => (Unit::Known(UnitExpr::dimensionless()), Vec::new()),
        FnUnitRule::Product(a, b) => {
            let (ua, mut notes) = eval_fn_rule(a, arg_units, kwargs);
            let (ub, notes_b) = eval_fn_rule(b, arg_units, kwargs);
            notes.extend(notes_b);
            (mul_div_unit(ua, ub, false), notes)
        }
        FnUnitRule::Quotient(a, b) => {
            let (ua, mut notes) = eval_fn_rule(a, arg_units, kwargs);
            let (ub, notes_b) = eval_fn_rule(b, arg_units, kwargs);
            notes.extend(notes_b);
            (mul_div_unit(ua, ub, true), notes)
        }
        FnUnitRule::PowN(inner, k) => {
            let (u, notes) = eval_fn_rule(inner, arg_units, kwargs);
            (pow_unit(u, *k), notes)
        }
        FnUnitRule::AllMatch(idxs) => {
            let (unit, note) = all_match(idxs, arg_units);
            (unit, note.into_iter().collect())
        }
        FnUnitRule::SelectByLiteral { kwarg, cases, default } => {
            let present = kwargs.iter().any(|(k, _)| k == kwarg);
            match literal_kwarg(kwargs, kwarg) {
                Some(selector) => {
                    let picked = cases.iter().find(|(lit, _)| *lit == selector).map(|(_, r)| r).unwrap_or_else(|| {
                        // A literal but not one of the recognised options —
                        // eval's own scaling parser rejects this before a
                        // cell can render, so fall back to `default` rather
                        // than invent a reason nothing can ever observe.
                        &cases.iter().find(|(lit, _)| lit == default).expect("default case must exist").1
                    });
                    eval_fn_rule(picked, arg_units, kwargs)
                }
                None if present => (Unit::Unknown(UnknownReason::NonLiteralExponent), Vec::new()),
                None => {
                    let default_rule = &cases.iter().find(|(lit, _)| lit == default).expect("default case must exist").1;
                    eval_fn_rule(default_rule, arg_units, kwargs)
                }
            }
        }
        FnUnitRule::Unknown => {
            (Unit::Unknown(UnknownReason::Propagated { of: "this function".to_string() }), Vec::new())
        }
    }
}

/// `Unit::pow` — the `PowN` combinator's leaf operation, mirroring
/// [`UnitExpr::pow`]/[`UnitExpr::sqrt`] but over the three-state lattice.
fn pow_unit(u: Unit, k: Ratio) -> Unit {
    match u {
        Unit::Scalar => Unit::Scalar,
        Unit::Known(expr) => Unit::Known(expr.pow(k)),
        Unit::Unknown(reason) => Unit::Unknown(reason),
    }
}

/// `AllMatch(idxs)`'s rule: every listed argument must share one `Known`
/// unit (Scalars adopt it; all-`Scalar` stays `Scalar`); a disagreement
/// among two or more `Known` units is the §3.3.1 diagnostic, output
/// `Unknown(Mismatch)`. Any `Unknown` operand propagates.
fn all_match(idxs: &[usize], arg_units: &[Unit]) -> (Unit, Option<UnitNote>) {
    let units: Vec<Unit> = idxs
        .iter()
        .map(|&i| {
            arg_units
                .get(i)
                .cloned()
                .unwrap_or_else(|| Unit::Unknown(UnknownReason::Propagated { of: "a missing argument".to_string() }))
        })
        .collect();

    if let Some(Unit::Unknown(reason)) = units.iter().find(|u| matches!(u, Unit::Unknown(_))) {
        return (Unit::Unknown(reason.clone()), None);
    }

    let known: Vec<&UnitExpr> = units.iter().filter_map(|u| if let Unit::Known(e) = u { Some(e) } else { None }).collect();
    let Some(first) = known.first() else {
        return (Unit::Scalar, None);
    };
    match known.iter().find(|e| **e != *first) {
        None => (Unit::Known((*first).clone()), None),
        Some(other) => {
            let note = UnitNote {
                message: format!("arguments must share one unit; found `{first}` and `{other}`"),
            };
            (
                Unit::Unknown(UnknownReason::Mismatch {
                    left: (*first).clone(),
                    right: (*other).clone(),
                    op: "call",
                }),
                Some(note),
            )
        }
    }
}

/// Applies §3.3.1's non-fatal checks, appending a [`UnitNote`] for each
/// violation. Never changes an already-computed output unit.
fn apply_fn_checks(checks: &[FnUnitCheck], arg_units: &[Unit], notes: &mut Vec<UnitNote>) {
    for check in checks {
        match check {
            FnUnitCheck::Expect(n, atom) => {
                if let Some(Unit::Known(u)) = arg_units.get(*n) {
                    let expected = UnitExpr::atom(atom);
                    if u != &expected {
                        notes.push(UnitNote { message: format!("argument {n}: expected `{atom}`, found `{u}`") });
                    }
                }
            }
            FnUnitCheck::ExpectDimensionless(n) => {
                if let Some(Unit::Known(u)) = arg_units.get(*n) {
                    if !u.is_dimensionless() {
                        notes.push(UnitNote {
                            message: format!("argument {n}: expected dimensionless, found `{u}`"),
                        });
                    }
                }
            }
            FnUnitCheck::SameUnit(a, b) => {
                if let (Some(Unit::Known(ua)), Some(Unit::Known(ub))) = (arg_units.get(*a), arg_units.get(*b)) {
                    if ua != ub {
                        notes.push(UnitNote {
                            message: format!("arguments {a} and {b}: units differ (`{ua}` and `{ub}`)"),
                        });
                    }
                }
            }
        }
    }
}

/// Display text for a comparison [`BinOp`], used only inside a `Mismatch`
/// reason's `op` field.
fn comparison_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
        BinOp::EqEq => "==",
        BinOp::BangEq => "!=",
        _ => unreachable!("comparison_symbol called with a non-comparison BinOp"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_atom_round_trips() {
        // Arrange
        let s = "mm";

        // Act
        let u = UnitExpr::parse(s).unwrap();

        // Assert
        assert_eq!(u, UnitExpr::atom("mm"));
        assert_eq!(u.to_string(), "mm");
    }

    #[test]
    fn parse_c1_unit_strings_round_trip() {
        // Arrange — every unit string seen recorded in core/ (task 1 note)
        let units = ["g", "dps", "pulse", "bar", "bpm", "deg", "m", "s", "mm", "km/h", "count"];

        for s in units {
            // Act
            let parsed = UnitExpr::parse(s).unwrap();
            let rendered = parsed.to_string();

            // Assert
            assert_eq!(rendered, s, "round-trip failed for '{s}'");
        }
    }

    #[test]
    fn parse_empty_string_is_an_error_not_dimensionless() {
        // Arrange
        let s = "";

        // Act
        let result = UnitExpr::parse(s);

        // Assert — C1's "no unit recorded" is a distinct case the caller
        // must handle explicitly, never silently treated as dimensionless
        assert_eq!(result, Err(UnitParseError::Empty));
    }

    #[test]
    fn mul_adds_exponents_on_matching_atoms() {
        // Arrange
        let mm_per_s = UnitExpr::parse("mm/s").unwrap();
        let s = UnitExpr::atom("s");

        // Act
        let product = mm_per_s.mul(&s);

        // Assert — the s and 1/s cancel, leaving mm
        assert_eq!(product, UnitExpr::atom("mm"));
    }

    #[test]
    fn mul_of_two_different_atoms_keeps_both() {
        // Arrange
        let mm = UnitExpr::atom("mm");
        let m_per_s = UnitExpr::parse("m/s").unwrap();

        // Act
        let product = mm.mul(&m_per_s);

        // Assert
        assert_eq!(product.to_string(), "m·mm/s");
    }

    #[test]
    fn div_of_atom_by_itself_is_dimensionless() {
        // Arrange
        let a = UnitExpr::atom("count");

        // Act
        let ratio = a.div(&a);

        // Assert
        assert!(ratio.is_dimensionless());
        assert_eq!(ratio.to_string(), "");
    }

    #[test]
    fn pow_with_integer_exponent_scales_every_atom() {
        // Arrange
        let g = UnitExpr::atom("g");

        // Act
        let g_squared = g.pow(Ratio::from_int(2));

        // Assert
        assert_eq!(g_squared.to_string(), "g^2");
    }

    #[test]
    fn pow_with_negative_exponent_moves_atom_to_denominator() {
        // Arrange
        let hz = UnitExpr::atom("Hz");

        // Act
        let inverse = hz.pow(Ratio::from_int(-1));

        // Assert
        assert_eq!(inverse.to_string(), "1/Hz");
    }

    #[test]
    fn sqrt_halves_the_exponent() {
        // Arrange
        let g_squared_per_hz = UnitExpr::parse("g^2/Hz").unwrap();

        // Act
        let g_per_sqrt_hz = g_squared_per_hz.sqrt();

        // Assert
        assert_eq!(g_per_sqrt_hz.to_string(), "g/Hz^(1/2)");
    }

    #[test]
    fn sqrt_is_total_over_an_already_negative_exponent() {
        // Arrange — g/Hz has Hz at exponent -1; halving must not panic
        let g_per_hz = UnitExpr::parse("g/Hz").unwrap();

        // Act
        let result = g_per_hz.sqrt();

        // Assert
        assert_eq!(result.to_string(), "g^(1/2)/Hz^(1/2)");
    }

    #[test]
    fn equality_distinguishes_different_atoms_of_the_same_kind() {
        // Arrange — the whole reason to prefer atoms over a dimension
        // vector: mm and m must never compare equal
        let mm = UnitExpr::atom("mm");
        let m = UnitExpr::atom("m");

        // Act / Assert
        assert_ne!(mm, m);
    }

    #[test]
    fn equality_is_order_independent() {
        // Arrange — built via two different multiplication orders
        let a = UnitExpr::atom("mm").mul(&UnitExpr::atom("s"));
        let b = UnitExpr::atom("s").mul(&UnitExpr::atom("mm"));

        // Act / Assert
        assert_eq!(a, b);
    }

    #[test]
    fn parse_rejects_malformed_fragment() {
        // Arrange
        let s = "/h";

        // Act
        let result = UnitExpr::parse(s);

        // Assert
        assert!(matches!(result, Err(UnitParseError::MissingAtom(_))));
    }

    #[test]
    fn parse_rejects_bad_exponent() {
        // Arrange
        let s = "Hz^abc";

        // Act
        let result = UnitExpr::parse(s);

        // Assert
        assert!(matches!(result, Err(UnitParseError::BadExponent(_))));
    }

    #[test]
    fn ratio_reduces_to_lowest_terms() {
        // Arrange / Act
        let r = Ratio::new(2, 4);

        // Assert
        assert_eq!(r, Ratio::new(1, 2));
    }

    // --- task 2: the `Unit` lattice and `infer` ---

    use crate::math::eval::{ChannelLookup, LookupChannel};
    use std::collections::HashMap;

    /// A test double supplying only `unit_of`, from a `name -> C1 unit
    /// string` map. `""` is stored the same way C1 does for "no unit
    /// recorded"; the trait's own filtering (mirroring `SessionHandle`) is
    /// reproduced here rather than exercised through it, since this module
    /// must not depend on `session`.
    struct UnitOnlyLookup(HashMap<&'static str, &'static str>);
    impl ChannelLookup for UnitOnlyLookup {
        fn lookup(&self, _name: &str) -> Option<LookupChannel> {
            None
        }
        fn unit_of(&self, name: &str) -> Option<String> {
            self.0.get(name).filter(|u| !u.is_empty()).map(|u| u.to_string())
        }
    }

    fn units(pairs: &[(&'static str, &'static str)]) -> UnitOnlyLookup {
        UnitOnlyLookup(pairs.iter().cloned().collect())
    }

    fn parse(src: &str) -> Ast {
        crate::math::parse::parse(src).unwrap()
    }

    #[test]
    fn infer_number_literal_is_scalar() {
        // Arrange
        let ast = parse("10");
        let lk = units(&[]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Scalar);
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_g_carries_m_per_s2() {
        // Arrange — R162: g must not be a dimensionless Scalar, or
        // [body_accel] / g would infer body_accel's own unit
        let ast = parse("g");
        let lk = units(&[]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("m/s^2").unwrap()));
    }

    #[test]
    fn infer_pi_is_dimensionless_not_scalar() {
        // Arrange
        let ast = parse("pi");
        let lk = units(&[]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
    }

    #[test]
    fn infer_accel_divided_by_g_does_not_collapse_to_the_accel_channel_unit() {
        // Arrange — the exact confidently-wrong-label case R162 names:
        // treating g as Scalar would make [BodyAccel] / g report "g" (the
        // accelerometer's own unit), the wrong answer.
        let ast = parse("[BodyAccel] / g");
        let lk = units(&[("BodyAccel", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert — the atom `g` (channel unit) and the constant's `m/s^2`
        // are never equated (R154: never canonicalise, never convert), so
        // the division is g/(m/s^2), not a bare "g".
        assert_ne!(unit, Unit::Known(UnitExpr::atom("g")));
    }

    #[test]
    fn infer_channel_ref_with_recorded_unit_is_known() {
        // Arrange
        let ast = parse("[Travel]");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
    }

    #[test]
    fn infer_channel_ref_with_no_recorded_unit_is_unknown() {
        // Arrange — C1's "" case, and a channel the lookup has never heard of
        let ast = parse("[Raw]");
        let lk = units(&[("Raw", "")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Unknown(UnknownReason::NoSourceUnit));
    }

    #[test]
    fn infer_cell_ref_is_unknown_no_source_unit() {
        // Arrange
        let ast = parse("{cell}");
        let lk = units(&[]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Unknown(UnknownReason::NoSourceUnit));
    }

    #[test]
    fn infer_unary_neg_keeps_the_operand_unit() {
        // Arrange
        let ast = parse("-[Travel]");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
    }

    #[test]
    fn infer_not_is_dimensionless() {
        // Arrange
        let ast = parse("not [Travel]");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
    }

    #[test]
    fn infer_literal_plus_channel_adopts_the_channel_unit() {
        // Arrange — the Scalar lattice: [travel] + 10 stays mm
        let ast = parse("[Travel] + 10");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_mismatched_addition_is_unknown_with_a_visible_note() {
        // Arrange — R154 §2.1: a diagnostic, not a hard error
        let ast = parse("[HR] + [Speed]");
        let lk = units(&[("HR", "bpm"), ("Speed", "km/h")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert!(matches!(unit, Unit::Unknown(UnknownReason::Mismatch { .. })));
        assert_eq!(notes.len(), 1);
        assert!(notes[0].message.contains("bpm"));
        assert!(notes[0].message.contains("km/h"));
    }

    #[test]
    fn infer_matching_addition_keeps_the_unit() {
        // Arrange
        let ast = parse("[A] + [B]");
        let lk = units(&[("A", "mm"), ("B", "mm")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_multiplication_of_two_knowns_combines_units() {
        // Arrange
        let ast = parse("[Travel] * [Speed]");
        let lk = units(&[("Travel", "mm"), ("Speed", "m/s")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("m·mm/s").unwrap()));
    }

    #[test]
    fn infer_division_by_scalar_keeps_the_unit() {
        // Arrange
        let ast = parse("[Travel] / 2");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
    }

    #[test]
    fn infer_comparison_is_always_dimensionless_but_still_checks_operands() {
        // Arrange — R154 §2: the result is unambiguous (Known(dimensionless)
        // never Unknown) even though the operands mismatch
        let ast = parse("[HR] < [Speed]");
        let lk = units(&[("HR", "bpm"), ("Speed", "km/h")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn infer_and_or_is_dimensionless_and_does_not_check_operands() {
        // Arrange — mismatched operands, but `and` does not compare them to
        // each other, only their own nested unit rules run
        let ast = parse("[HR] and [Speed]");
        let lk = units(&[("HR", "bpm"), ("Speed", "km/h")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_propagates_unknown_through_arithmetic() {
        // Arrange
        let ast = parse("[Raw] * 2");
        let lk = units(&[("Raw", "")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert!(matches!(unit, Unit::Unknown(UnknownReason::Propagated { .. })));
    }

    #[test]
    fn infer_call_propagates_an_unknown_operand_through_same_as_arg() {
        // Arrange — task 3's real rule table still propagates Unknown
        // through a SameAsArg function when the operand itself is unknown
        let ast = parse("abs([Raw])");
        let lk = units(&[("Raw", "")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert!(matches!(unit, Unit::Unknown(UnknownReason::NoSourceUnit)));
    }

    #[test]
    fn infer_call_still_surfaces_a_mismatch_nested_in_its_arguments() {
        // Arrange — the call itself yields Propagated, but a mismatch
        // buried inside an argument must not be swallowed
        let ast = parse("abs([HR] + [Speed])");
        let lk = units(&[("HR", "bpm"), ("Speed", "km/h")]);

        // Act
        let (_, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn infer_string_literal_is_not_numeric() {
        // Arrange — only reachable as a function argument (e.g. a
        // spectrogram scaling literal), never as a definition body itself
        let ast = Ast::Str("density".to_string());
        let lk = units(&[]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Unknown(UnknownReason::NotNumeric));
    }

    // --- UnitLabel (task 4's wire shape) ---

    #[test]
    fn unit_label_known_renders_the_unit_string() {
        // Arrange
        let unit = Unit::Known(UnitExpr::atom("mm"));

        // Act
        let label = UnitLabel::from(&unit);

        // Assert
        assert_eq!(label, UnitLabel::Known { text: "mm".to_string() });
    }

    #[test]
    fn unit_label_scalar_folds_to_dimensionless() {
        // Arrange — Scalar never crosses the wire (R154 §5)
        let unit = Unit::Scalar;

        // Act
        let label = UnitLabel::from(&unit);

        // Assert
        assert_eq!(label, UnitLabel::Dimensionless);
    }

    #[test]
    fn unit_label_known_dimensionless_folds_to_dimensionless() {
        // Arrange
        let unit = Unit::Known(UnitExpr::dimensionless());

        // Act
        let label = UnitLabel::from(&unit);

        // Assert
        assert_eq!(label, UnitLabel::Dimensionless);
    }

    #[test]
    fn unit_label_unknown_carries_a_display_ready_reason() {
        // Arrange
        let unit = Unit::Unknown(UnknownReason::NoSourceUnit);

        // Act
        let label = UnitLabel::from(&unit);

        // Assert
        assert_eq!(
            label,
            UnitLabel::Unknown { reason: "no unit recorded for this channel".to_string() }
        );
    }

    // --- task 3: the function unit-rule table (C2 §3.3/§3.3.1) ---

    #[test]
    fn function_unit_rule_catalog_completeness_every_builtin_has_a_real_rule() {
        // Arrange — task 3's own required test: a function added to
        // call_function with no entry here must fail loudly, not silently
        // report Unknown by omission.
        for builtin in crate::math::catalog::math_builtin_catalog() {
            // Act
            let (rule, _) = function_unit_rule(builtin.name, builtin.arity[0] as usize);

            // Assert — "pow" is the one deliberate FnUnitRule::Unknown
            // placeholder (special-cased directly in `infer`, see
            // `pow_rule`'s doc comment); every other name must have a real
            // rule.
            if builtin.name != "pow" {
                assert!(
                    !matches!(rule, FnUnitRule::Unknown),
                    "'{}' has no unit rule in function_unit_rule",
                    builtin.name
                );
            }
        }
    }

    #[test]
    fn infer_cumulative_trapezoid_multiplies_by_the_atom_s() {
        // Arrange
        let ast = parse("cumulative_trapezoid([Travel])");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("mm*s").unwrap()));
    }

    #[test]
    fn infer_differentiate_divides_by_the_atom_s() {
        // Arrange
        let ast = parse("differentiate([Travel])");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("mm/s").unwrap()));
    }

    #[test]
    fn infer_sqrt_call_halves_the_exponent() {
        // Arrange
        let ast = parse("sqrt([PsdG])");
        let lk = units(&[("PsdG", "g^2/Hz")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("g/Hz^(1/2)").unwrap()));
    }

    #[test]
    fn infer_body_accel_is_fixed_g_regardless_of_argument() {
        // Arrange
        let ast = parse("body_accel(\"long\")");
        let lk = units(&[]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("g")));
    }

    #[test]
    fn infer_count_is_dimensionless_not_the_atom_count() {
        // Arrange — a channel whose C1 unit happens to be the string
        // "count" is Known(count); count(ch) is Dimensionless regardless.
        let ast = parse("count([Wheel])");
        let lk = units(&[("Wheel", "count")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
    }

    #[test]
    fn infer_min_one_arg_is_same_as_arg_min_two_args_is_all_match() {
        // Arrange
        let one_arg = parse("min([Travel])");
        let two_args_ok = parse("min([A], [B])");
        let two_args_mismatch = parse("min([A], [C])");
        let lk = units(&[("Travel", "mm"), ("A", "mm"), ("B", "mm"), ("C", "km/h")]);

        // Act
        let (u1, _) = infer(&one_arg, &lk);
        let (u2, n2) = infer(&two_args_ok, &lk);
        let (u3, n3) = infer(&two_args_mismatch, &lk);

        // Assert
        assert_eq!(u1, Unit::Known(UnitExpr::atom("mm")));
        assert_eq!(u2, Unit::Known(UnitExpr::atom("mm")));
        assert!(n2.is_empty());
        assert!(matches!(u3, Unit::Unknown(UnknownReason::Mismatch { .. })));
        assert_eq!(n3.len(), 1);
    }

    #[test]
    fn infer_where_takes_the_matching_branch_unit_cond_is_unconstrained() {
        // Arrange — cond ([HR], bpm) is unrelated to t/f's shared mm
        let ast = parse("where([HR] > 0, [A], [B])");
        let lk = units(&[("HR", "bpm"), ("A", "mm"), ("B", "mm")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("mm")));
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_sin_of_a_degree_argument_is_a_diagnostic_not_a_conversion() {
        // Arrange — §3.3.1's own headline example
        let ast = parse("sin([AngleDeg])");
        let lk = units(&[("AngleDeg", "deg")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert — result is still Dimensionless, never blocked or converted
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
        assert_eq!(notes.len(), 1);
        assert!(notes[0].message.contains("rad"));
    }

    #[test]
    fn infer_sin_of_a_radian_argument_is_clean() {
        // Arrange
        let ast = parse("sin([AngleRad])");
        let lk = units(&[("AngleRad", "rad")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
        assert!(notes.is_empty());
    }

    #[test]
    fn infer_atan2_mismatched_operands_is_a_diagnostic_result_still_fixed_rad() {
        // Arrange
        let ast = parse("atan2([Y], [X])");
        let lk = units(&[("Y", "mm"), ("X", "m")]);

        // Act
        let (unit, notes) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("rad")));
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn infer_periodogram_density_scaling_is_ch_squared_per_hz() {
        // Arrange
        let ast = parse("periodogram([AccelG], scaling=\"density\")");
        let lk = units(&[("AccelG", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("g^2/Hz").unwrap()));
    }

    #[test]
    fn infer_periodogram_spectrum_scaling_is_ch_squared() {
        // Arrange
        let ast = parse("periodogram([AccelG], scaling=\"spectrum\")");
        let lk = units(&[("AccelG", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("g^2").unwrap()));
    }

    #[test]
    fn infer_periodogram_raw_magnitude_scaling_is_same_as_arg() {
        // Arrange
        let ast = parse("periodogram([AccelG], scaling=\"raw_magnitude\")");
        let lk = units(&[("AccelG", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("g")));
    }

    #[test]
    fn infer_periodogram_omitted_scaling_defaults_to_density() {
        // Arrange
        let ast = parse("periodogram([AccelG])");
        let lk = units(&[("AccelG", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("g^2/Hz").unwrap()));
    }

    #[test]
    fn spectral_output_unit_density_scaling_is_ch_squared_per_hz() {
        // Arrange / Act
        let label = spectral_output_unit("g", "density");

        // Assert
        assert_eq!(label, UnitLabel::Known { text: "g^2/Hz".to_string() });
    }

    #[test]
    fn spectral_output_unit_spectrum_scaling_is_ch_squared() {
        // Arrange / Act
        let label = spectral_output_unit("g", "spectrum");

        // Assert
        assert_eq!(label, UnitLabel::Known { text: "g^2".to_string() });
    }

    #[test]
    fn spectral_output_unit_raw_magnitude_scaling_is_same_as_source() {
        // Arrange / Act
        let label = spectral_output_unit("g", "raw_magnitude");

        // Assert
        assert_eq!(label, UnitLabel::Known { text: "g".to_string() });
    }

    #[test]
    fn spectral_output_unit_matches_infers_periodogram_call_for_every_scaling() {
        // Arrange
        let lk = units(&[("AccelG", "g")]);

        // Act / Assert — the direct helper and infer()'s Ast::Call path must
        // never disagree, since both read spectral_rule().
        for scaling in ["density", "spectrum", "raw_magnitude"] {
            let ast = parse(&format!("periodogram([AccelG], scaling=\"{scaling}\")"));
            let (via_infer, _) = infer(&ast, &lk);
            let via_helper = spectral_output_unit("g", scaling);
            assert_eq!(UnitLabel::from(&via_infer), via_helper);
        }
    }

    #[test]
    fn spectral_output_unit_empty_source_unit_is_unknown_no_source_unit() {
        // Arrange / Act
        let label = spectral_output_unit("", "raw_magnitude");

        // Assert
        assert_eq!(label, UnitLabel::Unknown { reason: UnknownReason::NoSourceUnit.describe() });
    }

    #[test]
    fn infer_welch_shares_periodograms_scaling_rule() {
        // Arrange
        let ast = parse("welch([AccelG], scaling=\"spectrum\")");
        let lk = units(&[("AccelG", "g")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("g^2").unwrap()));
    }

    #[test]
    fn infer_pow_with_integer_literal_exponent() {
        // Arrange
        let ast = parse("pow([Travel], 2)");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::parse("mm^2").unwrap()));
    }

    #[test]
    fn infer_pow_with_half_literal_exponent() {
        // Arrange
        let ast = parse("pow([PsdG], 0.5)");
        let lk = units(&[("PsdG", "g^2")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::atom("g")));
    }

    #[test]
    fn infer_pow_with_non_literal_exponent_on_a_dimensioned_base_is_non_literal_exponent() {
        // Arrange
        let ast = parse("pow([Travel], [N])");
        let lk = units(&[("Travel", "mm"), ("N", "count")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Unknown(UnknownReason::NonLiteralExponent));
    }

    #[test]
    fn infer_pow_with_non_literal_exponent_on_a_dimensionless_base_is_dimensionless() {
        // Arrange — 0 * anything = 0: a dimensionless base stays
        // dimensionless under any exponent, literal or not. `count([Wheel])`
        // is genuinely `Known(dimensionless)` (the *function*, not a
        // channel whose atom happens to be the string "count" — those are
        // different claims, §3.3.1's own "count and the atom count" note).
        let ast = parse("pow(count([Wheel]), [N])");
        let lk = units(&[("Wheel", "pulse"), ("N", "count")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert_eq!(unit, Unit::Known(UnitExpr::dimensionless()));
    }

    #[test]
    fn ratio_from_f64_recovers_integers_and_halves_not_arbitrary_fractions() {
        // Arrange / Act / Assert
        assert_eq!(ratio_from_f64(2.0), Some(Ratio::from_int(2)));
        assert_eq!(ratio_from_f64(-1.0), Some(Ratio::from_int(-1)));
        assert_eq!(ratio_from_f64(0.5), Some(Ratio::new(1, 2)));
        assert_eq!(ratio_from_f64(-1.5), Some(Ratio::new(-3, 2)));
        assert_eq!(ratio_from_f64(0.3), None);
    }
}
