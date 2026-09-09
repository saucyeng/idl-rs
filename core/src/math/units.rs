//! Unit algebra (ruling R154 / `runs/2026-09-08/unit-model.md`). A small
//! symbolic model over the unit strings C1 §4.1 already records: opaque
//! atoms (`mm`, `g`, `km`, `h`, `Hz`, …) taken verbatim, mapped to rational
//! exponents, never canonicalised and never converted. `mm` and `m` stay two
//! different atoms — that is what lets `[travel_mm] + [altitude_m]` be
//! caught, which a dimension-vector model cannot do (R154 §1(b)).
//!
//! This module is task 1 of the design's 8-task plan: the algebra only.
//! Inference over the `Ast` (the `Unit`/`UnitLabel` lattice, `infer`, the
//! function rule table) is later tasks in this same module.

use std::collections::BTreeMap;
use std::fmt;

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
}
