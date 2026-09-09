//! Recursive-descent parser producing an [`Ast`].
//!
//! Precedence ladder mirrors the Dart evaluator's parse functions
//! (`math_channel_evaluator.dart` `_parseOr`..`_parsePrimary`):
//! `or → and → comparison → additive → multiplicative → unary → primary`.
//! Operator semantics (elementwise application, division-by-zero, truthiness)
//! are NOT here — they live in `eval.rs`. This module only shapes the tree.

use std::collections::HashMap;

use crate::math::token::{tokenize, Token, TokenKind};
use crate::math::{MathEvalError, MathEvalErrorKind};

/// Binary operators, in the grammar's recognised set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Lt,
    Gt,
    LtEq,
    GtEq,
    EqEq,
    BangEq,
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

/// Parsed expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Number(f64),
    /// One of the four universal named constants (`pi`, `tau`, `e`, `g`,
    /// [`constant_value`]) — resolved to its numeric `value` at parse time,
    /// same as a workbook constant, but keeping `name` so a later pass can
    /// tell it apart from an arbitrary literal. `eval` treats this exactly
    /// like `Number(value)`; the unit-inference pass (R154/R162) is the
    /// only reader of `name` — `pi`/`tau`/`e` are dimensionless, `g`
    /// carries `m/s²` (C2 §3.2). Declared workbook constants (C2 §3.1)
    /// stay plain `Number`s (R154 open question 4: no unit this revision).
    Constant { name: &'static str, value: f64 },
    Str(String),
    ChannelRef(String),
    /// `{ … }` cell reference — resolved against the table's cell namespace.
    CellRef(String),
    Unary { op: UnOp, expr: Box<Ast> },
    Binary { op: BinOp, left: Box<Ast>, right: Box<Ast> },
    Call {
        name: String,
        args: Vec<Ast>,
        /// `name=expr` keyword arguments (C2 §3.2, R143 item 1), in
        /// call-site order — additive to `args`, never a replacement for
        /// it (positional stays valid). Parsing enforces the grammar's two
        /// purely syntactic rules: no keyword argument may precede a
        /// positional one in the same call, and no name may be bound twice
        /// by keyword. Binding a name both positionally *and* by keyword,
        /// and an unknown keyword name, both require knowing the callee's
        /// parameter names — `eval::call_function`'s job, not the
        /// parser's, so neither is rejected here.
        kwargs: Vec<(String, Ast)>,
    },
}

fn parse_err(msg: impl Into<String>) -> MathEvalError {
    MathEvalError::new(MathEvalErrorKind::Parse, msg)
}

/// Universal scalar constants usable as bare identifiers in any expression
/// (e.g. `[IMU1_AccelZ] * g`, `2 * pi * [Freq]`). They are not stored anywhere
/// — they resolve to a literal at parse time and are always available, so they
/// travel with a portable `.idl0wb`. Channel references are always bracketed
/// (`[g]`), so a bare `g` is unambiguously the constant.
///
/// `g` is standard gravity in m/s²; `pi` / `tau` / `e` are the math
/// constants. Returns the canonical `&'static str` name alongside the value
/// so a caller can build an [`Ast::Constant`] without re-matching (R162).
fn constant_value(name: &str) -> Option<(&'static str, f64)> {
    match name {
        "pi" => Some(("pi", std::f64::consts::PI)),
        "tau" => Some(("tau", std::f64::consts::TAU)),
        "e" => Some(("e", std::f64::consts::E)),
        "g" => Some(("g", 9.806_65)),
        _ => None,
    }
}

/// Tokenizes and parses `src` into an [`Ast`]. Rejects trailing tokens.
/// Equivalent to `parse_with_constants(src, &HashMap::new())` — no
/// workbook-level constants table in scope, matching every existing v2
/// caller's behaviour unchanged.
pub fn parse(src: &str) -> Result<Ast, MathEvalError> {
    parse_with_constants(src, &HashMap::new())
}

/// Tokenizes and parses `src` into an [`Ast`], additionally resolving a bare
/// identifier against `constants` (`name → f64`, C2 §3.1's flat workbook
/// constants namespace — front matter + `const` lines, already merged by
/// `workbook::v3::constants::merge_constants`) when it is neither a call nor
/// one of the four universal constants (`pi`/`tau`/`e`/`g`, [`constant_value`]
/// — checked first, so they always take precedence and can never be
/// shadowed by `constants`).
///
/// **Precondition, not enforced here:** this function has no opinion on C2
/// §3.5.A's `ReservedName` rule — it trusts its caller (`merge_constants`)
/// already excluded the universal four (and every other reserved name) from
/// `constants` before calling. A `constants` entry that does share a
/// universal-four name is simply shadowed by the built-in at parse time
/// (`constant_value` wins), never treated as an error by this function.
pub fn parse_with_constants(src: &str, constants: &HashMap<String, f64>) -> Result<Ast, MathEvalError> {
    let tokens = tokenize(src)?;
    let mut p = Parser { tokens, pos: 0, constants };
    let ast = p.parse_or()?;
    if p.cur().kind != TokenKind::Eof {
        return Err(parse_err(format!(
            "Unexpected token \"{}\" after expression end",
            p.cur().str_val
        )));
    }
    Ok(ast)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    constants: &'a HashMap<String, f64>,
}

impl<'a> Parser<'a> {
    fn cur(&self) -> &Token {
        &self.tokens[self.pos]
    }
    fn check(&self, kind: TokenKind) -> bool {
        self.cur().kind == kind
    }
    fn check_ident(&self, name: &str) -> bool {
        self.cur().kind == TokenKind::Ident && self.cur().str_val == name
    }
    fn match_kind(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, kind: TokenKind) -> Result<(), MathEvalError> {
        if self.cur().kind != kind {
            return Err(parse_err(format!(
                "Expected {:?} but got {:?} (\"{}\")",
                kind,
                self.cur().kind,
                self.cur().str_val
            )));
        }
        self.pos += 1;
        Ok(())
    }

    // or → and (('or') and)*
    fn parse_or(&mut self) -> Result<Ast, MathEvalError> {
        let mut left = self.parse_and()?;
        while self.check_ident("or") {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Ast::Binary { op: BinOp::Or, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    // and → comparison (('and') comparison)*
    fn parse_and(&mut self) -> Result<Ast, MathEvalError> {
        let mut left = self.parse_comparison()?;
        while self.check_ident("and") {
            self.pos += 1;
            let right = self.parse_comparison()?;
            left = Ast::Binary { op: BinOp::And, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    // comparison → additive (('<'|'>'|'<='|'>='|'=='|'!=') additive)*
    fn parse_comparison(&mut self) -> Result<Ast, MathEvalError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = if self.match_kind(TokenKind::Lt) {
                BinOp::Lt
            } else if self.match_kind(TokenKind::Gt) {
                BinOp::Gt
            } else if self.match_kind(TokenKind::LtEq) {
                BinOp::LtEq
            } else if self.match_kind(TokenKind::GtEq) {
                BinOp::GtEq
            } else if self.match_kind(TokenKind::EqEq) {
                BinOp::EqEq
            } else if self.match_kind(TokenKind::BangEq) {
                BinOp::BangEq
            } else {
                break;
            };
            let right = self.parse_additive()?;
            left = Ast::Binary { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    // additive → multiplicative (('+' | '-') multiplicative)*
    fn parse_additive(&mut self) -> Result<Ast, MathEvalError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = if self.match_kind(TokenKind::Plus) {
                BinOp::Add
            } else if self.match_kind(TokenKind::Minus) {
                BinOp::Sub
            } else {
                break;
            };
            let right = self.parse_multiplicative()?;
            left = Ast::Binary { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    // multiplicative → unary (('*' | '/') unary)*
    fn parse_multiplicative(&mut self) -> Result<Ast, MathEvalError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = if self.match_kind(TokenKind::Star) {
                BinOp::Mul
            } else if self.match_kind(TokenKind::Slash) {
                BinOp::Div
            } else {
                break;
            };
            let right = self.parse_unary()?;
            left = Ast::Binary { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    // unary → '-' unary | 'not' unary | primary
    fn parse_unary(&mut self) -> Result<Ast, MathEvalError> {
        if self.match_kind(TokenKind::Minus) {
            return Ok(Ast::Unary { op: UnOp::Neg, expr: Box::new(self.parse_unary()?) });
        }
        if self.check_ident("not") {
            self.pos += 1;
            return Ok(Ast::Unary { op: UnOp::Not, expr: Box::new(self.parse_unary()?) });
        }
        self.parse_primary()
    }

    // primary → number | string | '[' ident ']' | ident '(' args ')' | '(' expr ')'
    fn parse_primary(&mut self) -> Result<Ast, MathEvalError> {
        if self.check(TokenKind::Number) {
            let v = self.cur().num_val;
            self.pos += 1;
            return Ok(Ast::Number(v));
        }
        if self.check(TokenKind::Str) {
            let s = self.cur().str_val.clone();
            self.pos += 1;
            return Ok(Ast::Str(s));
        }
        if self.check(TokenKind::CellRef) {
            let name = self.cur().str_val.clone();
            self.pos += 1;
            return Ok(Ast::CellRef(name));
        }
        if self.match_kind(TokenKind::LBracket) {
            if !self.check(TokenKind::Ident) {
                return Err(parse_err(format!(
                    "Expected channel name inside [...], got {:?}",
                    self.cur().kind
                )));
            }
            let name = self.cur().str_val.clone();
            self.pos += 1;
            self.expect(TokenKind::RBracket)?;
            return Ok(Ast::ChannelRef(name));
        }
        if self.check(TokenKind::Ident) {
            let name = self.cur().str_val.clone();
            self.pos += 1;
            if self.match_kind(TokenKind::LParen) {
                let (args, kwargs) = self.parse_args()?;
                return Ok(Ast::Call { name, args, kwargs });
            }
            // A bare identifier that is a universal constant (pi / tau / e / g)
            // resolves to a literal; failing that, a threaded workbook
            // constants-table lookup (C2 §3.1); anything else is a
            // missing-bracket error.
            if let Some((canonical_name, value)) = constant_value(&name) {
                return Ok(Ast::Constant { name: canonical_name, value });
            }
            if let Some(value) = self.constants.get(&name) {
                return Ok(Ast::Number(*value));
            }
            return Err(parse_err(format!(
                "Unexpected identifier \"{name}\" — did you mean [{name}] for a channel reference?"
            )));
        }
        if self.match_kind(TokenKind::LParen) {
            let inner = self.parse_or()?;
            self.expect(TokenKind::RParen)?;
            return Ok(inner);
        }
        // A bare `=` reaching here is not a keyword argument's separator
        // (that form is consumed by `parse_one_arg`'s own lookahead before
        // ever calling into an expression) — it's a genuinely misplaced
        // `=`, most often a typo'd comparison (mirrors the tokenizer's
        // former message, moved here per C2 §3.2 since only the parser
        // knows an `=` wasn't a keyword name's separator).
        if self.check(TokenKind::Equals) {
            return Err(parse_err("Unexpected \"=\" in expression — did you mean \"==\"?"));
        }
        Err(parse_err(format!(
            "Unexpected token {:?} (\"{}\") in expression",
            self.cur().kind,
            self.cur().str_val
        )))
    }

    // Parses a comma-separated argument list, consuming the closing ')'.
    // Each argument is either a bare expression (positional) or
    // `ident '=' expression` (keyword, C2 §3.2) — see `parse_one_arg`.
    fn parse_args(&mut self) -> Result<(Vec<Ast>, Vec<(String, Ast)>), MathEvalError> {
        let mut args = Vec::new();
        let mut kwargs: Vec<(String, Ast)> = Vec::new();
        if !self.check(TokenKind::RParen) {
            self.parse_one_arg(&mut args, &mut kwargs)?;
            while self.match_kind(TokenKind::Comma) {
                self.parse_one_arg(&mut args, &mut kwargs)?;
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok((args, kwargs))
    }

    // One call argument: `ident '=' expr` (keyword) when the next two
    // tokens are exactly that — a two-token lookahead so a bare identifier
    // *expression* (a workbook constant, `mean([X], k)`'s `k`) is never
    // mistaken for a keyword name — otherwise a positional expression.
    // Enforces the grammar's two syntactic keyword-argument rules (see
    // `Ast::Call::kwargs`'s doc comment); the two rules that need the
    // callee's own parameter names are the evaluator's job.
    fn parse_one_arg(&mut self, args: &mut Vec<Ast>, kwargs: &mut Vec<(String, Ast)>) -> Result<(), MathEvalError> {
        if self.check(TokenKind::Ident) && self.peek_kind(1) == Some(TokenKind::Equals) {
            let kw_name = self.cur().str_val.clone();
            self.pos += 2; // the identifier, then '='
            if kwargs.iter().any(|(bound, _)| bound == &kw_name) {
                return Err(parse_err(format!(
                    "keyword argument \"{kw_name}\" is bound more than once in this call"
                )));
            }
            let value = self.parse_or()?;
            kwargs.push((kw_name, value));
            return Ok(());
        }
        if !kwargs.is_empty() {
            return Err(parse_err(
                "a positional argument can't follow a keyword argument in the same call",
            ));
        }
        args.push(self.parse_or()?);
        Ok(())
    }

    fn peek_kind(&self, ahead: usize) -> Option<TokenKind> {
        self.tokens.get(self.pos + ahead).map(|t| t.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ast(src: &str) -> Ast {
        parse(src).unwrap()
    }

    #[test]
    fn parse_precedence_multiplication_binds_tighter_than_addition() {
        // Arrange / Act — 1 + 2 * 3 must group as 1 + (2 * 3).
        let a = ast("1 + 2 * 3");

        // Assert — root is Binary(Add, 1, Binary(Mul, 2, 3)).
        match a {
            Ast::Binary { op: BinOp::Add, left, right } => {
                assert!(matches!(*left, Ast::Number(n) if n == 1.0));
                assert!(matches!(*right, Ast::Binary { op: BinOp::Mul, .. }));
            }
            other => panic!("unexpected root: {other:?}"),
        }
    }

    #[test]
    fn parse_cell_ref_yields_cellref_node() {
        assert_eq!(parse("{fork_max}").unwrap(), Ast::CellRef("fork_max".into()));
    }

    #[test]
    fn parse_cell_ref_in_expression() {
        // {a} - min({fork_max[]}) → Binary(Sub, CellRef, Call(min,[CellRef]))
        let ast = parse("{a} - min({fork_max[]})").unwrap();
        match ast {
            Ast::Binary { op: BinOp::Sub, left, right } => {
                assert_eq!(*left, Ast::CellRef("a".into()));
                assert!(matches!(*right, Ast::Call { .. }));
            }
            _ => panic!("expected subtraction"),
        }
    }

    #[test]
    fn parse_channel_reference() {
        // Arrange / Act
        let a = ast("[GPS_SpeedKmh]");

        // Assert
        assert!(matches!(a, Ast::ChannelRef(name) if name == "GPS_SpeedKmh"));
    }

    #[test]
    fn parse_function_call_with_mixed_args() {
        // Arrange / Act
        let a = ast("butter(2, 0.3, \"high\", [IMU1_AccelZ])");

        // Assert — Call("butter", [Number, Number, Str, ChannelRef]).
        match a {
            Ast::Call { name, args, kwargs } => {
                assert_eq!(name, "butter");
                assert_eq!(args.len(), 4);
                assert!(matches!(args[2], Ast::Str(ref s) if s == "high"));
                assert!(matches!(args[3], Ast::ChannelRef(ref n) if n == "IMU1_AccelZ"));
                assert!(kwargs.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_function_call_with_a_keyword_argument() {
        // Arrange / Act
        let a = ast("mean([X], window=5)");

        // Assert — one positional arg, one keyword arg.
        match a {
            Ast::Call { name, args, kwargs } => {
                assert_eq!(name, "mean");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Ast::ChannelRef(ref n) if n == "X"));
                assert_eq!(kwargs.len(), 1);
                assert_eq!(kwargs[0].0, "window");
                assert!(matches!(kwargs[0].1, Ast::Number(v) if v == 5.0));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_function_call_with_only_keyword_arguments() {
        // Arrange / Act
        let a = ast("periodogram(ch=[X], window=\"hann\")");

        // Assert
        match a {
            Ast::Call { name, args, kwargs } => {
                assert_eq!(name, "periodogram");
                assert!(args.is_empty());
                assert_eq!(kwargs.len(), 2);
                assert_eq!(kwargs[0].0, "ch");
                assert_eq!(kwargs[1].0, "window");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_a_bare_identifier_expression_is_never_mistaken_for_a_keyword_name() {
        // Arrange — a workbook constant `k` used as a positional argument
        // must not be consumed as a keyword name just because it's an
        // identifier; only `ident '=' ...` is a keyword argument.
        let mut constants = HashMap::new();
        constants.insert("k".to_string(), 3.0);

        // Act
        let a = parse_with_constants("mean([X], k)", &constants).unwrap();

        // Assert
        match a {
            Ast::Call { args, kwargs, .. } => {
                assert_eq!(args.len(), 2);
                assert!(matches!(args[1], Ast::Number(v) if v == 3.0));
                assert!(kwargs.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_a_positional_argument_after_a_keyword_argument_is_a_parse_error() {
        // Act
        let err = parse("mean(window=5, [X])").unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("positional argument can't follow"));
    }

    #[test]
    fn parse_bare_equals_where_an_expression_is_expected_says_did_you_mean_eqeq() {
        // Arrange — C2 §3.2: the tokenizer no longer hard-errors on a bare
        // `=` (it's a keyword argument's separator); a `=` that reaches
        // `parse_primary` is genuinely misplaced, most often a typo'd `==`.
        // Act — `=` appears where an operand was expected (right after
        // `+`), so parsing reaches `parse_primary` with `=` as `cur()`.
        let err = parse("1 + = 2").unwrap_err();

        // Assert — mirrors the message the tokenizer used to raise.
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("=="), "{}", err.message);
    }

    #[test]
    fn parse_the_same_keyword_name_bound_twice_is_a_parse_error() {
        // Act
        let err = parse("mean([X], window=5, window=6)").unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("bound more than once"));
    }

    #[test]
    fn parse_unary_minus_and_not() {
        // Arrange / Act / Assert
        assert!(matches!(ast("-[x]"), Ast::Unary { op: UnOp::Neg, .. }));
        assert!(matches!(ast("not [x]"), Ast::Unary { op: UnOp::Not, .. }));
    }

    #[test]
    fn parse_parenthesized_overrides_precedence() {
        // Arrange / Act — (1 + 2) * 3 groups the addition first.
        let a = ast("(1 + 2) * 3");

        // Assert
        match a {
            Ast::Binary { op: BinOp::Mul, left, .. } => {
                assert!(matches!(*left, Ast::Binary { op: BinOp::Add, .. }));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_trailing_token_after_expression_is_error() {
        // Arrange / Act / Assert — mirrors Dart "Unexpected token after expression end".
        assert_eq!(parse("1 2").unwrap_err().kind, MathEvalErrorKind::Parse);
    }

    #[test]
    fn parse_bare_identifier_without_call_is_error() {
        // Arrange / Act / Assert — Dart hints "did you mean [name]?".
        let err = parse("Speed").unwrap_err();
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("Speed"));
    }

    #[test]
    fn parse_universal_constants_resolve_to_named_constants() {
        // Arrange / Act / Assert — bare pi / tau / e / g are literal
        // constants, but keep their name (R162) so the unit-inference pass
        // can tell `g` apart from an arbitrary literal.
        assert!(matches!(ast("pi"), Ast::Constant { name: "pi", value } if (value - std::f64::consts::PI).abs() < 1e-12));
        assert!(matches!(ast("tau"), Ast::Constant { name: "tau", value } if (value - std::f64::consts::TAU).abs() < 1e-12));
        assert!(matches!(ast("e"), Ast::Constant { name: "e", value } if (value - std::f64::consts::E).abs() < 1e-12));
        assert!(matches!(ast("g"), Ast::Constant { name: "g", value } if (value - 9.806_65).abs() < 1e-12));
    }

    #[test]
    fn parse_constant_in_expression() {
        // Arrange / Act — `[X] * g` multiplies the channel by gravity.
        let a = ast("[IMU1_AccelZ] * g");

        // Assert — Binary(Mul, ChannelRef, Constant("g", 9.80665)).
        match a {
            Ast::Binary { op: BinOp::Mul, left, right } => {
                assert!(matches!(*left, Ast::ChannelRef(ref n) if n == "IMU1_AccelZ"));
                assert!(matches!(*right, Ast::Constant { name: "g", value } if (value - 9.806_65).abs() < 1e-12));
            }
            other => panic!("unexpected root: {other:?}"),
        }
    }

    #[test]
    fn parse_with_constants_bare_identifier_k_evaluates_as_literal_9_81() {
        // Arrange
        let constants = HashMap::from([("k".to_string(), 9.81)]);

        // Act
        let a = parse_with_constants("k * 2", &constants).unwrap();

        // Assert — Binary(Mul, Number(9.81), Number(2)).
        match a {
            Ast::Binary { op: BinOp::Mul, left, right } => {
                assert!(matches!(*left, Ast::Number(n) if (n - 9.81).abs() < 1e-12));
                assert!(matches!(*right, Ast::Number(n) if n == 2.0));
            }
            other => panic!("unexpected root: {other:?}"),
        }
    }

    #[test]
    fn parse_with_constants_pi_in_table_still_resolves_to_the_builtin_pi() {
        // Arrange — parse_with_constants trusts its caller (merge_constants)
        // already excluded the universal four from the table; this call
        // succeeds using the parser's own built-in `pi`, not the table's
        // `1.0` entry, since constant_value is checked first.
        let constants = HashMap::from([("pi".to_string(), 1.0)]);

        // Act
        let a = parse_with_constants("pi * 2", &constants).unwrap();

        // Assert
        match a {
            Ast::Binary { op: BinOp::Mul, left, right } => {
                assert!(matches!(*left, Ast::Constant { name: "pi", value } if (value - std::f64::consts::PI).abs() < 1e-12));
                assert!(matches!(*right, Ast::Number(n) if n == 2.0));
            }
            other => panic!("unexpected root: {other:?}"),
        }
    }

    #[test]
    fn parse_with_constants_unknown_identifier_is_parse_error() {
        // Act
        let err = parse_with_constants("nope * 2", &HashMap::new()).unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("Unexpected identifier \"nope\""));
    }

    #[test]
    fn parse_with_empty_constants_table_equals_parse() {
        // Act / Assert — parse() is a zero-constants call into the same
        // underlying function; HashMap::new() does not allocate.
        assert_eq!(parse("1 + [X] * g"), parse_with_constants("1 + [X] * g", &HashMap::new()));
    }
}
