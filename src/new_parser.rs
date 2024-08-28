use crate::{lexer::Token, parser::ParseError};

pub struct TokenParser<'a> {
    input: &'a [Token],
    position: usize,
}

impl<'a> TokenParser<'a> {
    pub fn new(input: &'a [Token]) -> Self {
        Self { input, position: 0 }
    }

    pub fn try_parse<F, E, O>(&mut self, f: F) -> Result<O, E>
    where
        F: Fn(&mut Self) -> Result<O, E>,
    {
        let mut parser = self.sliced();
        let result = f(&mut parser);
        if result.is_ok() {
            self.position += parser.position;
        }
        result
    }

    pub fn sliced(&self) -> Self {
        Self::new(&self.input[self.position..])
    }

    pub fn advance(&mut self) -> &'a Token {
        match self.input.get(self.position) {
            Some(i) => {
                self.position += 1;
                i
            }
            None => &Token::Eof,
        }
    }

    pub fn one_of(&mut self, expected: &[Token]) -> Result<&'a Token, ParseError> {
        let token = &self.peek();
        if expected.contains(token) {
            self.advance();
            Ok(token)
        } else {
            todo!()
        }
    }

    pub fn expect(&mut self, expected: &Token) -> Result<&'a Token, ParseError> {
        let token = self.peek();
        if token == expected {
            self.advance();
            Ok(token)
        } else {
            todo!()
        }
    }

    pub fn advance_by(&mut self, n: usize) -> &'a [Token] {
        self.position += n;

        if self.position > self.input.len() {
            self.position = self.input.len();
            return &[];
        }

        &self.input[self.position - n..self.position]
    }

    pub fn peek(&self) -> &'a Token {
        self.input.get(self.position).unwrap_or_else(|| &Token::Eof)
    }

    pub fn input(&self) -> &'a [Token] {
        self.input
    }

    pub fn parse<T: Parseable>(&mut self) -> Result<T, ParseError> {
        self.try_parse(T::parse)
    }

    pub fn unexpected_token<O>(&self, expected: impl Into<String>) -> Result<O, ParseError> {
        Err(ParseError::UnexpectedToken(
            Some(self.input[self.position - 1].clone()),
            expected.into(),
        ))
    }

    pub fn expect_identifier(&mut self) -> Result<&'a str, ParseError> {
        match self.advance() {
            Token::Identifier(s) => Ok(s),
            _ => self.unexpected_token("identifier"),
        }
    }
}

pub trait Parseable {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError>
    where
        Self: Sized;
}
