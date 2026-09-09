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
            // Task 2 stop-gap (design §7, task 2 row): every call is
            // Unknown(Propagated) regardless of function, but nested
            // mismatches inside its arguments are still surfaced.
            let mut notes = Vec::new();
            for arg in args {
                let (_, arg_notes) = infer(arg, lookup);
                notes.extend(arg_notes);
            }
            for (_, expr) in kwargs {
                let (_, kw_notes) = infer(expr, lookup);
                notes.extend(kw_notes);
            }
            (Unit::Unknown(UnknownReason::Propagated { of: format!("{name}(…)") }), notes)
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
    fn infer_call_is_unknown_propagated_regardless_of_function() {
        // Arrange — task 2 stop-gap; task 3 installs the real per-function
        // rule table
        let ast = parse("sqrt([Travel])");
        let lk = units(&[("Travel", "mm")]);

        // Act
        let (unit, _) = infer(&ast, &lk);

        // Assert
        assert!(matches!(unit, Unit::Unknown(UnknownReason::Propagated { .. })));
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
}
