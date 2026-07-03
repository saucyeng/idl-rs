//! Tokenizer for the math-channel expression grammar.
//!
//! Ported verbatim from the Dart `_tokenize` (`math_channel_evaluator.dart`).
//! The `[ChannelName]` form is captured as three tokens — `LBracket`, a single
//! `Ident` holding the *entire* bracketed name verbatim (names may contain
//! spaces and digit-leading segments), and `RBracket` — so the parser's
//! `'[' ident ']'` rule never tokenizes a channel name into sub-tokens.

use crate::math::{MathEvalError, MathEvalErrorKind};

/// Token kinds. Mirrors the Dart `_TT` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Number,
    Str,
    Ident,
    LBracket,
    RBracket,
    /// Whole `{ … }` cell reference, inner text captured verbatim.
    CellRef,
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Gt,
    LtEq,
    GtEq,
    EqEq,
    BangEq,
    Eof,
}

/// A single lexical token. `num_val` is meaningful for `Number`; `str_val` for
/// `Str` and `Ident`. Both are default/empty otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub num_val: f64,
    pub str_val: String,
}

impl Token {
    fn simple(kind: TokenKind) -> Self {
        Token { kind, num_val: 0.0, str_val: String::new() }
    }
    fn number(v: f64) -> Self {
        Token { kind: TokenKind::Number, num_val: v, str_val: String::new() }
    }
    fn with_str(kind: TokenKind, s: String) -> Self {
        Token { kind, num_val: 0.0, str_val: s }
    }
}

fn parse_err(msg: impl Into<String>) -> MathEvalError {
    MathEvalError::new(MathEvalErrorKind::Parse, msg)
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn is_alpha(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Tokenizes `src` into a `Vec<Token>` ending in `Eof`, or a `Parse` error.
pub fn tokenize(src: &str) -> Result<Vec<Token>, MathEvalError> {
    // Operate on chars by index. The grammar is ASCII; channel names may
    // contain arbitrary chars but are captured verbatim by char slice.
    let chars: Vec<char> = src.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;
    let n = chars.len();

    while i < n {
        let c = chars[i];

        // Whitespace
        if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
            i += 1;
            continue;
        }

        // Numeric literal (integer or float, optional exponent)
        if is_digit(c) || (c == '.' && i + 1 < n && is_digit(chars[i + 1])) {
            let start = i;
            let mut j = i;
            while j < n && (is_digit(chars[j]) || chars[j] == '.') {
                j += 1;
            }
            if j < n && (chars[j] == 'e' || chars[j] == 'E') {
                j += 1;
                if j < n && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                while j < n && is_digit(chars[j]) {
                    j += 1;
                }
            }
            let lit: String = chars[start..j].iter().collect();
            let v: f64 = lit
                .parse()
                .map_err(|_| parse_err(format!("Invalid number literal \"{lit}\"")))?;
            tokens.push(Token::number(v));
            i = j;
            continue;
        }

        // String literal — single or double quoted
        if c == '"' || c == '\'' {
            let quote = c;
            let mut j = i + 1;
            while j < n && chars[j] != quote {
                j += 1;
            }
            if j >= n {
                return Err(parse_err(format!("Unterminated string starting at position {i}")));
            }
            let s: String = chars[i + 1..j].iter().collect();
            tokens.push(Token::with_str(TokenKind::Str, s));
            i = j + 1;
            continue;
        }

        // Identifier or keyword
        if is_alpha(c) || c == '_' {
            let start = i;
            let mut j = i;
            while j < n && (is_alpha(chars[j]) || is_digit(chars[j]) || chars[j] == '_') {
                j += 1;
            }
            let s: String = chars[start..j].iter().collect();
            tokens.push(Token::with_str(TokenKind::Ident, s));
            i = j;
            continue;
        }

        // Channel reference — capture the whole bracketed name verbatim.
        if c == '[' {
            let mut j = i + 1;
            while j < n && chars[j] != ']' {
                j += 1;
            }
            if j >= n {
                return Err(parse_err(format!(
                    "Unclosed \"[\" at position {i} — expected a closing \"]\""
                )));
            }
            let name: String = chars[i + 1..j].iter().collect();
            tokens.push(Token::simple(TokenKind::LBracket));
            tokens.push(Token::with_str(TokenKind::Ident, name));
            tokens.push(Token::simple(TokenKind::RBracket));
            i = j + 1;
            continue;
        }

        // Cell reference — capture the whole braced body verbatim ({A1},
        // {colname}, {colname[]}). A distinct sigil from [Channel] so the two
        // namespaces never collide.
        if c == '{' {
            let mut j = i + 1;
            while j < n && chars[j] != '}' {
                j += 1;
            }
            if j >= n {
                return Err(parse_err(format!(
                    "Unclosed \"{{\" at position {i} — expected a closing \"}}\""
                )));
            }
            let name: String = chars[i + 1..j].iter().collect();
            tokens.push(Token::with_str(TokenKind::CellRef, name));
            i = j + 1;
            continue;
        }

        // Single- and two-character symbols.
        match c {
            ']' => { tokens.push(Token::simple(TokenKind::RBracket)); i += 1; }
            '(' => { tokens.push(Token::simple(TokenKind::LParen)); i += 1; }
            ')' => { tokens.push(Token::simple(TokenKind::RParen)); i += 1; }
            ',' => { tokens.push(Token::simple(TokenKind::Comma)); i += 1; }
            '+' => { tokens.push(Token::simple(TokenKind::Plus)); i += 1; }
            '-' => { tokens.push(Token::simple(TokenKind::Minus)); i += 1; }
            '*' => { tokens.push(Token::simple(TokenKind::Star)); i += 1; }
            '/' => { tokens.push(Token::simple(TokenKind::Slash)); i += 1; }
            '<' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::simple(TokenKind::LtEq)); i += 2;
                } else {
                    tokens.push(Token::simple(TokenKind::Lt)); i += 1;
                }
            }
            '>' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::simple(TokenKind::GtEq)); i += 2;
                } else {
                    tokens.push(Token::simple(TokenKind::Gt)); i += 1;
                }
            }
            '=' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::simple(TokenKind::EqEq)); i += 2;
                } else {
                    return Err(parse_err(format!(
                        "Unexpected \"=\" at position {i} — did you mean \"==\"?"
                    )));
                }
            }
            '!' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::simple(TokenKind::BangEq)); i += 2;
                } else {
                    return Err(parse_err(format!(
                        "Unexpected \"!\" at position {i} — did you mean \"!=\"?"
                    )));
                }
            }
            other => {
                return Err(parse_err(format!("Unexpected character \"{other}\" at position {i}")));
            }
        }
    }

    tokens.push(Token::simple(TokenKind::Eof));
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn tokenize_channel_ref_captures_whole_bracketed_name() {
        // Arrange — names may contain spaces / digit-leading segments.
        let toks = tokenize("[Declipped 1_AccelX]").unwrap();

        // Assert — exactly LBracket, Ident("Declipped 1_AccelX"), RBracket, Eof.
        assert_eq!(toks.len(), 4);
        assert_eq!(toks[0].kind, TokenKind::LBracket);
        assert_eq!(toks[1].kind, TokenKind::Ident);
        assert_eq!(toks[1].str_val, "Declipped 1_AccelX");
        assert_eq!(toks[2].kind, TokenKind::RBracket);
        assert_eq!(toks[3].kind, TokenKind::Eof);
    }

    #[test]
    fn tokenize_cell_ref_captures_whole_braced_name() {
        // Arrange — cell refs may be A1, named, or a column form `name[]`.
        let toks = tokenize("{fork_max[]} - min({B2})").unwrap();

        // Assert — first token is a CellRef holding the inner text verbatim.
        assert_eq!(toks[0].kind, TokenKind::CellRef);
        assert_eq!(toks[0].str_val, "fork_max[]");
        // …and a later CellRef for {B2}.
        assert!(toks.iter().any(|t| t.kind == TokenKind::CellRef && t.str_val == "B2"));
    }

    #[test]
    fn tokenize_unclosed_brace_is_error() {
        let err = tokenize("{oops").unwrap_err();
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains('{'));
    }

    #[test]
    fn tokenize_number_with_exponent() {
        // Arrange / Act
        let toks = tokenize("1.5e-3").unwrap();

        // Assert
        assert_eq!(toks[0].kind, TokenKind::Number);
        assert!((toks[0].num_val - 1.5e-3).abs() < 1e-18);
    }

    #[test]
    fn tokenize_two_char_operators() {
        // Arrange / Act / Assert
        assert_eq!(kinds("<="), vec![TokenKind::LtEq, TokenKind::Eof]);
        assert_eq!(kinds(">="), vec![TokenKind::GtEq, TokenKind::Eof]);
        assert_eq!(kinds("=="), vec![TokenKind::EqEq, TokenKind::Eof]);
        assert_eq!(kinds("!="), vec![TokenKind::BangEq, TokenKind::Eof]);
    }

    #[test]
    fn tokenize_string_single_or_double_quoted() {
        // Arrange / Act
        let a = tokenize("\"hann\"").unwrap();
        let b = tokenize("'low'").unwrap();

        // Assert
        assert_eq!(a[0].kind, TokenKind::Str);
        assert_eq!(a[0].str_val, "hann");
        assert_eq!(b[0].str_val, "low");
    }

    #[test]
    fn tokenize_bare_equals_is_error() {
        // Arrange / Act
        let err = tokenize("a = b").unwrap_err();

        // Assert — mirrors Dart: "did you mean ==?"
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains("=="));
    }

    #[test]
    fn tokenize_unclosed_bracket_is_error() {
        // Arrange / Act
        let err = tokenize("[Speed").unwrap_err();

        // Assert
        assert_eq!(err.kind, MathEvalErrorKind::Parse);
        assert!(err.message.contains('['));
    }

    #[test]
    fn tokenize_unterminated_string_is_error() {
        // Arrange / Act / Assert
        assert!(tokenize("\"oops").is_err());
    }

    #[test]
    fn tokenize_unexpected_char_is_error() {
        // Arrange / Act / Assert
        assert!(tokenize("@").is_err());
    }
}
