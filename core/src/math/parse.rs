//! Recursive-descent parser producing an [`Ast`].
//!
//! Precedence ladder mirrors the Dart evaluator's parse functions
//! (`math_channel_evaluator.dart` `_parseOr`..`_parsePrimary`):
//! `or → and → comparison → additive → multiplicative → unary → primary`.
//! Operator semantics (elementwise application, division-by-zero, truthiness)
//! are NOT here — they live in `eval.rs`. This module only shapes the tree.

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
    Str(String),
    ChannelRef(String),
    /// `{ … }` cell reference — resolved against the table's cell namespace.
    CellRef(String),
    Unary { op: UnOp, expr: Box<Ast> },
    Binary { op: BinOp, left: Box<Ast>, right: Box<Ast> },
    Call { name: String, args: Vec<Ast> },
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
/// `g` is standard gravity in m/s²; `pi` / `tau` / `e` are the math constants.
fn constant_value(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "tau" => Some(std::f64::consts::TAU),
        "e" => Some(std::f64::consts::E),
        "g" => Some(9.806_65),
        _ => None,
    }
}

/// Tokenizes and parses `src` into an [`Ast`]. Rejects trailing tokens.
pub fn parse(src: &str) -> Result<Ast, MathEvalError> {
    let tokens = tokenize(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let ast = p.parse_or()?;
    if p.cur().kind != TokenKind::Eof {
        return Err(parse_err(format!(
            "Unexpected token \"{}\" after expression end",
            p.cur().str_val
        )));
    }
    Ok(ast)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
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
                let args = self.parse_args()?;
                return Ok(Ast::Call { name, args });
            }
            // A bare identifier that is a universal constant (pi / tau / e / g)
            // resolves to a literal; anything else is a missing-bracket error.
            if let Some(value) = constant_value(&name) {
                return Ok(Ast::Number(value));
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
        Err(parse_err(format!(
            "Unexpected token {:?} (\"{}\") in expression",
            self.cur().kind,
            self.cur().str_val
        )))
    }

    // Parses a comma-separated argument list, consuming the closing ')'.
    fn parse_args(&mut self) -> Result<Vec<Ast>, MathEvalError> {
        let mut args = Vec::new();
        if !self.check(TokenKind::RParen) {
            args.push(self.parse_or()?);
            while self.match_kind(TokenKind::Comma) {
                args.push(self.parse_or()?);
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(args)
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
            Ast::Call { name, args } => {
                assert_eq!(name, "butter");
                assert_eq!(args.len(), 4);
                assert!(matches!(args[2], Ast::Str(ref s) if s == "high"));
                assert!(matches!(args[3], Ast::ChannelRef(ref n) if n == "IMU1_AccelZ"));
            }
            other => panic!("unexpected: {other:?}"),
        }
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
    fn parse_universal_constants_resolve_to_numbers() {
        // Arrange / Act / Assert — bare pi / tau / e / g are literal constants.
        assert!(matches!(ast("pi"), Ast::Number(n) if (n - std::f64::consts::PI).abs() < 1e-12));
        assert!(matches!(ast("tau"), Ast::Number(n) if (n - std::f64::consts::TAU).abs() < 1e-12));
        assert!(matches!(ast("e"), Ast::Number(n) if (n - std::f64::consts::E).abs() < 1e-12));
        assert!(matches!(ast("g"), Ast::Number(n) if (n - 9.806_65).abs() < 1e-12));
    }

    #[test]
    fn parse_constant_in_expression() {
        // Arrange / Act — `[X] * g` multiplies the channel by gravity.
        let a = ast("[IMU1_AccelZ] * g");

        // Assert — Binary(Mul, ChannelRef, Number(9.80665)).
        match a {
            Ast::Binary { op: BinOp::Mul, left, right } => {
                assert!(matches!(*left, Ast::ChannelRef(ref n) if n == "IMU1_AccelZ"));
                assert!(matches!(*right, Ast::Number(n) if (n - 9.806_65).abs() < 1e-12));
            }
            other => panic!("unexpected root: {other:?}"),
        }
    }
}
