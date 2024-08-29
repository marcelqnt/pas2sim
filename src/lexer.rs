use anyhow::Context;
use ordered_float::OrderedFloat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    UnsignedInteger(u64),
    UnsignedFloat(OrderedFloat<f64>),
    Identifier(String),
    ConstantString(String),
    CommentSlash(String),
    CommentBracket(String),

    Point,
    DoublePoint,
    Colon,
    Comma,
    Semicolon,

    LeftParen,
    RightParen,
    LeftSquareBracket,
    RightSquareBracket,

    Plus,
    Minus,
    Asterisk,
    Slash,

    Equal,
    NotEqual,
    Greater,
    Smaller,
    GreateOrEqual,
    SmallerOrEqual,

    Caret,

    Assignment,

    Not,
    And,
    Or,
    In,
    Div,
    Mod,

    Nil,

    Array,
    Of,
    File,
    Set,
    Record,

    Type,
    Var,
    Const,

    Function,
    Procedure,
    Begin,
    End,

    Program,
    Uses,
    Unit,

    If,
    Then,
    Else,

    While,
    Do,
    For,
    To,
    Downto,
    Repeat,
    Until,

    With,

    Goto,
    Label,

    Exit,
    Continue,
    Break,
    Eof,
}

impl Token {
    pub fn unsigned_integer(n: u64) -> Self {
        Self::UnsignedInteger(n)
    }

    pub fn unsigned_float(n: f64) -> Self {
        Self::UnsignedFloat(n.into())
    }

    pub fn identifier(s: impl Into<String>) -> Self {
        Self::Identifier(s.into())
    }

    pub fn constant_string(s: impl Into<String>) -> Self {
        Self::ConstantString(s.into())
    }

    pub fn comment_slash(s: impl Into<String>) -> Self {
        Self::CommentSlash(s.into())
    }

    pub fn comment_bracket(s: impl Into<String>) -> Self {
        Self::CommentBracket(s.into())
    }
}

struct Parser<'a> {
    source: &'a [char],
    position: usize,
    tokens: Vec<Token>,
    curr_token_str: String,
    curr_token: Option<Token>,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a [char]) -> Self {
        Self {
            source,
            tokens: Default::default(),
            curr_token_str: Default::default(),
            curr_token: Default::default(),
            position: 0,
        }
    }

    fn finish_token(&mut self, token: Token) {
        self.curr_token = None;
        self.tokens.push(token);
    }

    fn finish_token_with_position_reset(&mut self, token: Token) {
        self.finish_token(token);
        self.position -= 1;
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.source.get(self.position).copied();
        self.position += 1;
        c
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.position).copied()
    }
}

pub fn tokenize(code: &str) -> anyhow::Result<Vec<Token>> {
    let source = code
        .to_lowercase()
        .chars()
        .chain(['\n'])
        .collect::<Vec<_>>();
    let mut parser = Parser::new(&source);

    while let Some(c) = parser.advance() {
        match parser.curr_token {
            None => {
                parser.curr_token_str = c.to_string();
                match c {
                    '0'..='9' => parser.curr_token = Some(Token::UnsignedInteger(0)),
                    'a'..='z' | '_' | '$' => parser.curr_token = Some(Token::Identifier("".into())),

                    '{' => {
                        parser.curr_token = Some(Token::CommentBracket("".into()));
                        parser.curr_token_str = "".into();
                    }

                    '\'' => {
                        parser.curr_token = Some(Token::ConstantString("".into()));
                        parser.curr_token_str = "".into();
                    }

                    '.' => parser.curr_token = Some(Token::Point),
                    ':' => parser.curr_token = Some(Token::Colon),
                    ',' => parser.finish_token(Token::Comma),
                    ';' => parser.finish_token(Token::Semicolon),

                    '(' => parser.finish_token(Token::LeftParen),
                    ')' => parser.finish_token(Token::RightParen),
                    '[' => parser.finish_token(Token::LeftSquareBracket),
                    ']' => parser.finish_token(Token::RightSquareBracket),

                    '+' => parser.finish_token(Token::Plus),
                    '-' => parser.finish_token(Token::Minus),
                    '*' => parser.finish_token(Token::Asterisk),
                    '/' => parser.curr_token = Some(Token::Slash),

                    '=' => parser.finish_token(Token::Equal),
                    '<' => parser.curr_token = Some(Token::Smaller),
                    '>' => parser.curr_token = Some(Token::Greater),

                    '^' => parser.finish_token(Token::Caret),

                    ' ' | '\r' | '\n' | '\t' => {}
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Unexpected character for next token: {}",
                            c
                        ))
                    }
                }
            }

            Some(Token::CommentSlash(_)) => match c {
                '\n' => parser.finish_token_with_position_reset(Token::CommentSlash(
                    parser.curr_token_str.clone(),
                )),
                _ => parser.curr_token_str.push(c),
            },
            Some(Token::CommentBracket(_)) => match c {
                '}' => {
                    parser.curr_token = None;
                    parser
                        .tokens
                        .push(Token::CommentBracket(parser.curr_token_str.clone()));
                }
                _ => parser.curr_token_str.push(c),
            },

            Some(Token::UnsignedInteger(_)) => match c {
                '0'..='9' => parser.curr_token_str.push(c),
                '.' => {
                    parser.curr_token_str.push(c);
                    parser.curr_token = Some(Token::UnsignedFloat(0.0.into()));
                }
                _ => parser.finish_token_with_position_reset(Token::UnsignedInteger(
                    parser
                        .curr_token_str
                        .parse()
                        .context("failed to pass unsigned integer token")?,
                )),
            },
            Some(Token::UnsignedFloat(_)) => match c {
                '0'..='9' => parser.curr_token_str.push(c),
                _ => parser.finish_token_with_position_reset(Token::UnsignedFloat(
                    parser
                        .curr_token_str
                        .parse()
                        .context("failed to pass unsigned float token")?,
                )),
            },
            Some(Token::Identifier(_)) => match c {
                '0'..='9' | 'a'..='z' | '_' => parser.curr_token_str.push(c),
                _ => parser.finish_token_with_position_reset(
                    match parser.curr_token_str.clone().as_str() {
                        "not" => Token::Not,
                        "and" => Token::And,
                        "or" => Token::Or,
                        "in" => Token::In,
                        "div" => Token::Div,
                        "mod" => Token::Mod,

                        "nil" => Token::Nil,

                        "array" => Token::Array,
                        "of" => Token::Of,
                        "file" => Token::File,
                        "set" => Token::Set,
                        "record" => Token::Record,

                        "type" => Token::Type,
                        "var" => Token::Var,
                        "const" => Token::Const,

                        "function" => Token::Function,
                        "procedure" => Token::Procedure,
                        "begin" => Token::Begin,
                        "end" => Token::End,

                        "program" => Token::Program,
                        "uses" => Token::Uses,
                        "unit" => Token::Unit,

                        "if" => Token::If,
                        "then" => Token::Then,
                        "else" => Token::Else,

                        "while" => Token::While,
                        "do" => Token::Do,
                        "for" => Token::For,
                        "to" => Token::To,
                        "downto" => Token::Downto,
                        "repeat" => Token::Repeat,
                        "until" => Token::Until,

                        "with" => Token::With,
                        "goto" => Token::Goto,
                        "label" => Token::Label,

                        "exit" => Token::Exit,
                        "continue" => Token::Continue,
                        "break" => Token::Break,
                        _ => Token::Identifier(parser.curr_token_str.clone()),
                    },
                ),
            },
            Some(Token::ConstantString(_)) => match c {
                '\'' => {
                    if let Some(next) = parser.peek() {
                        if next == '\'' {
                            parser.curr_token_str.push('\'');
                            parser.advance();
                        } else {
                            parser
                                .finish_token(Token::ConstantString(parser.curr_token_str.clone()));
                        }
                    } else {
                        return Err(anyhow::anyhow!("Unclosed string const"));
                    }
                }
                _ => parser.curr_token_str.push(c),
            },

            Some(Token::Point) => match c {
                '.' => parser.finish_token(Token::DoublePoint),
                _ => parser.finish_token_with_position_reset(Token::Point),
            },
            Some(Token::Colon) => match c {
                '=' => parser.finish_token(Token::Assignment),
                _ => parser.finish_token_with_position_reset(Token::Colon),
            },

            Some(Token::Slash) => match c {
                '/' => {
                    parser.curr_token = Some(Token::CommentSlash("".into()));
                    parser.curr_token_str = "".into();
                }
                _ => parser.finish_token_with_position_reset(Token::Slash),
            },

            Some(Token::Smaller) => match c {
                '=' => parser.finish_token(Token::SmallerOrEqual),
                '>' => parser.finish_token(Token::NotEqual),
                _ => parser.finish_token_with_position_reset(Token::Smaller),
            },
            Some(Token::Greater) => match c {
                '=' => parser.finish_token(Token::GreateOrEqual),
                _ => parser.finish_token_with_position_reset(Token::Greater),
            },

            _ => {
                return Err(anyhow::anyhow!(
                    "Unhandled token kind: {:?}",
                    parser.curr_token
                ))
            }
        }
    }

    Ok(parser.tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_token {
        ($name:ident, $token:expr, $input:expr) => {
            #[test]
            fn $name() {
                assert_eq!(vec![$token], tokenize($input).unwrap());
            }
        };
    }

    macro_rules! test_tokens {
        ($name:ident, [$($token:expr),*], $input:expr) => {
            #[test]
            fn $name() {
                assert_eq!(vec![$($token),*], tokenize($input).unwrap());
            }
        };
    }

    //test_single_token!(int, Token::SignedInteger(124), "124");

    #[test]
    fn complete_failure() {
        assert!(tokenize("§").is_err());
    }

    test_token!(uint, Token::UnsignedInteger(124), "124");

    test_token!(identifier, Token::Identifier("qwer2t".into()), "qwer2T");

    test_token!(floating, Token::UnsignedFloat(12.052.into()), "012.0520");

    test_tokens!(
        number_slash_letters,
        [
            Token::UnsignedInteger(124),
            Token::Slash,
            Token::Identifier("abs".into())
        ],
        "124/abs"
    );

    test_tokens!(
        slash_comment,
        [
            Token::UnsignedInteger(123),
            Token::CommentSlash("vier".into()),
            Token::Identifier("abc".into())
        ],
        "123 //vier\nabc"
    );

    test_token!(
        constant_string,
        Token::ConstantString("einszwei'vier".into()),
        "'einszwei''vier'"
    );

    test_tokens!(
        bracket_comment,
        [
            Token::UnsignedInteger(445),
            Token::CommentBracket("testumgebung".into()),
            Token::Identifier("aw_r".into())
        ],
        "445    {Testumgebung}   aw_r"
    );

    test_tokens!(
        dotdotdot,
        [
            Token::Point,
            Token::Identifier("abcsd".into()),
            Token::DoublePoint
        ],
        ".abcsd.."
    );

    test_tokens!(
        semi_colon,
        [
            Token::Identifier("a".into()),
            Token::Assignment,
            Token::Identifier("b".into()),
            Token::Colon,
            Token::Identifier("c".into()),
            Token::Semicolon
        ],
        "A := b : c;"
    );

    test_tokens!(
        punctuation,
        [
            Token::Point,
            Token::Comma,
            Token::Minus,
            Token::Plus,
            Token::Semicolon,
            Token::Colon,
            Token::Assignment,
            Token::Equal,
            Token::Smaller,
            Token::SmallerOrEqual,
            Token::Greater,
            Token::GreateOrEqual,
            Token::NotEqual,
            Token::Caret,
            Token::Asterisk
        ],
        ".,-+;::==<<=>>=<>^*"
    );

    test_tokens!(
        keywords,
        [
            Token::Identifier("andor".into()),
            Token::And,
            Token::Or,
            Token::In,
            Token::Div,
            Token::Mod,
            Token::Identifier("divmod".into()),
            Token::Nil,
            Token::Array,
            Token::Of,
            Token::File,
            Token::Record,
            Token::Set,
            Token::Type,
            Token::Var,
            Token::Const,
            Token::Function,
            Token::Procedure,
            Token::Begin,
            Token::End,
            Token::Program,
            Token::Uses,
            Token::Unit,
            Token::If,
            Token::Then,
            Token::Else,
            Token::For,
            Token::While,
            Token::To,
            Token::Do,
            Token::Downto,
            Token::Repeat,
            Token::Until,
            Token::With,
            Token::Goto,
            Token::Label,
            Token::Exit,
            Token::Continue,
            Token::Break
        ],
        "andor and or in div mod divmod nil array of file record set type var const function procedure begin end program uses unit if then else for while to do downto repeat until with goto label exit continue break"
    );
}
