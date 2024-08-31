use anyhow::Context;
use ordered_float::OrderedFloat;

use crate::lexer::Token;
use crate::new_parser::{Parseable, TokenParser};

#[derive(Debug, Clone, PartialEq)]
pub enum UnsignedConstant {
    Identifier(String),
    UnsignedInteger(u64),
    UnsignedFloat(OrderedFloat<f64>),
    Nil,
    String(String),
}

pub fn parse_unsigned_constant(input: &[Token]) -> anyhow::Result<(UnsignedConstant, &[Token])> {
    use UnsignedConstant::*;
    let res = match input {
        [Token::Identifier(s)] => Identifier(s.clone()),
        [Token::UnsignedInteger(u)] => UnsignedInteger(*u),
        [Token::UnsignedFloat(f)] => UnsignedFloat(*f),
        [Token::Nil] => Nil,
        [Token::ConstantString(s)] => String(s.clone()),
        _ => {
            return Err(anyhow::anyhow!(
                "Unexpected next token for unsigned constant: {:?}",
                input.first()
            ))
        }
    };

    Ok((res, &input[1..]))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Identifier(ConstantIdentifier),
    Literal(String),
    Number(Number),
    Nil,
}

impl Parseable for Constant {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let sign = parser.parse::<Option<Sign>>()?;

        let res = match parser.advance() {
            Token::Nil => Constant::Nil,
            Token::Identifier(ident) => Constant::Identifier(ConstantIdentifier {
                name: ident.clone(),
                sign,
            }),
            Token::UnsignedInteger(u) if sign.map(Sign::is_plus).unwrap_or(true) => {
                Constant::Number(Number::Unsigned(*u))
            }
            Token::UnsignedInteger(u) if sign.map(Sign::is_minus).unwrap_or(false) => {
                let i: i64 = (*u)
                    .try_into()
                    .context("failed to convert number to signed integer")?;
                Constant::Number(Number::Signed(-i))
            }
            Token::UnsignedFloat(f) if sign.map(Sign::is_plus).unwrap_or(true) => {
                Constant::Number(Number::Real(*f))
            }
            Token::UnsignedFloat(f) if sign.map(Sign::is_minus).unwrap_or(false) => {
                Constant::Number(Number::Real(-f))
            }
            Token::ConstantString(s) => Constant::Literal(s.clone()),
            _ => parser.unexpected_token("expected constant")?,
        };

        Ok(res)
    }
}

impl Constant {
    pub fn identifier(name: impl Into<String>, sign: impl Into<Option<Sign>>) -> Self {
        Constant::Identifier(ConstantIdentifier {
            name: name.into(),
            sign: sign.into(),
        })
    }

    pub fn literal(s: impl Into<String>) -> Self {
        Constant::Literal(s.into())
    }

    pub fn number(n: impl Into<Number>) -> Self {
        Constant::Number(n.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Number {
    Signed(i64),
    Unsigned(u64),
    Real(OrderedFloat<f64>),
}

impl From<u32> for Number {
    fn from(u: u32) -> Self {
        Number::Unsigned(u as u64)
    }
}

impl From<u64> for Number {
    fn from(u: u64) -> Self {
        Number::Unsigned(u)
    }
}

impl From<i32> for Number {
    fn from(i: i32) -> Self {
        Number::Signed(i as i64)
    }
}

impl From<i64> for Number {
    fn from(i: i64) -> Self {
        Number::Signed(i)
    }
}

impl From<f32> for Number {
    fn from(f: f32) -> Self {
        (f as f64).into()
    }
}

impl From<f64> for Number {
    fn from(f: f64) -> Self {
        Number::Real(f.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sign {
    Plus,
    Minus,
}

impl Parseable for Sign {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        const SIGNS: &[Token] = &[Token::Plus, Token::Minus];
        let sign = match parser.one_of(SIGNS) {
            Ok(s) => s,
            Err(_) => return parser.unexpected_token("expected sign"),
        };

        if parser.one_of(SIGNS).is_ok() {
            return parser.unexpected_token("no double sign is allowed");
        }

        let sign = match sign {
            Token::Plus => Sign::Plus,
            Token::Minus => Sign::Minus,
            _ => unreachable!(),
        };

        Ok(sign)
    }
}

impl Parseable for Option<Sign> {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        if Sign::SIGNS.contains(parser.peek()) {
            Ok(Some(parser.parse()?))
        } else {
            Ok(None)
        }
    }
}

impl Sign {
    const SIGNS: &'static [Token] = &[Token::Plus, Token::Minus];

    fn is_plus(self) -> bool {
        match self {
            Sign::Plus => true,
            Sign::Minus => false,
        }
    }

    fn is_minus(self) -> bool {
        !self.is_plus()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantIdentifier {
    name: String,
    sign: Option<Sign>,
}

impl<T: AsRef<str>> From<T> for ConstantIdentifier {
    fn from(name: T) -> Self {
        Self {
            name: name.as_ref().into(),
            sign: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected token '{0:?}': {1:?}")]
    UnexpectedToken(Option<Token>, String),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimpleType {
    SoloType(String),
    Ordinal(Vec<String>),
    StaticArray(Constant, Constant),
}

impl Parseable for SimpleType {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl SimpleType {
    pub fn parse_old(input: &[Token]) -> anyhow::Result<(Self, &[Token])> {
        todo!();
        // let mut i = 0;
        // let mut ordinaries = Vec::new();
        // let mut mode = None;

        // let res = loop {
        //     if i == 0 {
        //         if let Ok((c, [Token::DoublePoint, rest @ ..])) = Constant::parse(input) {
        //             match Constant::parse(rest) {
        //                 Ok((c2, rest)) => return Ok((SimpleType::StaticArray(c, c2), rest)),
        //                 Err(e) => return Err(e.into()),
        //             }
        //         }
        //     }

        //     match &input[i..] {
        //         [Token::LeftParen, ..] => {
        //             i += 1;
        //             mode = Some(SimpleType::Ordinal(Vec::new()));
        //         }
        //         [Token::Identifier(ident), ..] => {
        //             i += 1;
        //             match mode {
        //                 Some(SimpleType::Ordinal(_)) => ordinaries.push(ident.clone()),
        //                 None => break SimpleType::SoloType(ident.clone()),
        //                 _ => {
        //                     return Err(anyhow::anyhow!(
        //                         "Unexpected next token for simple type (solo/ordinary): {:?}",
        //                         input.first()
        //                     ))
        //                 }
        //             }
        //         }
        //         [Token::RightParen, ..] if matches!(mode, Some(SimpleType::Ordinal(_))) => {
        //             i += 1;
        //             break SimpleType::Ordinal(std::mem::take(&mut ordinaries));
        //         }
        //         [Token::Comma, ..] if matches!(mode, Some(SimpleType::Ordinal(_))) => {
        //             i += 1;
        //         }
        //         _ => {
        //             return Err(anyhow::anyhow!(
        //                 "Unexpected next token for simple type: {:?}",
        //                 input.first()
        //             ))
        //         }
        //     }
        // };

        // Ok((res, &input[(i + 1).min(input.len())..]))
    }

    pub fn solo_type(name: impl Into<String>) -> Self {
        Self::SoloType(name.into())
    }

    pub fn ordinal(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Ordinal(names.into_iter().map(Into::into).collect())
    }

    pub fn static_array(lower: impl Into<Constant>, upper: impl Into<Constant>) -> Self {
        Self::StaticArray(lower.into(), upper.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    SimpleType(SimpleType),
    PointerOf(String),
    StaticArray(Vec<SimpleType>, Box<Type>),
    File(Box<Type>),
    Set(SimpleType),
    Record(Vec<Field>),
}

impl Parseable for Type {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        match parser.advance() {
            Token::Caret => {
                if let Token::Identifier(ident) = parser.advance() {
                    Ok(Type::PointerOf(ident.clone()))
                } else {
                    parser.unexpected_token("expected identifier after pointer")
                }
            }
            Token::Array => {
                parser.expect(&Token::LeftSquareBracket)?;

                let mut types = Vec::new();

                loop {
                    let simple_type = parser.parse()?;
                    types.push(simple_type);

                    if let Token::RightSquareBracket =
                        parser.one_of(&[Token::Comma, Token::RightSquareBracket])?
                    {
                        break;
                    }
                }

                parser.expect(&Token::Of)?;
                let t = parser.parse()?;

                Ok(Type::StaticArray(types, Box::new(t)))
            }
            Token::File => {
                parser.expect(&Token::Of)?;
                let t = parser.parse()?;
                Ok(Type::File(Box::new(t)))
            }
            Token::Set => {
                parser.expect(&Token::Of)?;
                let t = parser.parse()?;
                Ok(Type::Set(t))
            }
            Token::Record => {
                let mut fields = Vec::new();
                loop {
                    let name = parser.expect_identifier()?;
                    parser.expect(&Token::Colon)?;
                    let typ = parser.parse()?;
                    fields.push(Field {
                        name: name.to_owned(),
                        typ,
                    });

                    if let Token::Semicolon = parser.advance() {
                        continue;
                    } else {
                        break;
                    }
                }

                Ok(Type::Record(fields))
            }
            _ => {
                let t = parser.parse()?;
                Ok(Type::SimpleType(t))

                // parser.unexpected_token("expected ")}
            }
        }
    }

    // pub fn parse(input: &[Token]) -> anyhow::Result<(Self, &[Token])> {
    //     enum Expected {
    //         None,
    //         StaticArray,
    //     }

    //     let mut expected = Expected::None;
    //     let mut i = 0;

    //     match input {
    //         [Token::Caret, Token::Identifier(ident), ..] => {
    //             Ok((Type::PointerOf(ident.clone()), &input[2..]))
    //         }
    //         [Token::Array, Token::LeftSquareBracket, ..] => {
    //             i = 2;
    //             expected = Expected::StaticArray;
    //             let types = Vec::new();
    //             loop {
    //                 match input[i] {
    //                     Token::RightSquareBracket => {
    //                         i += 1;
    //                         break;}
    //                     Token::Comma => {
    //                         i += 1;
    //                     _ => types.push( )
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub typ: Type,
}

impl Field {
    pub fn parse(parser: &mut TokenParser) -> Result<Vec<Self>, ParseError> {
        let mut names = Vec::new();

        loop {
            let name = parser.expect_identifier()?;
            names.push(name);

            match parser.advance() {
                Token::Colon => {
                    break;
                }
                Token::Comma => {}
                _ => return parser.unexpected_token("expected comma or colon"),
            }
        }

        let typ = parser.parse::<Type>()?;

        Ok(names
            .into_iter()
            .map(|name| Field {
                name: name.to_owned(),
                typ: typ.clone(),
            })
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariableItem {
    Field(String),
    // foo.bar
    //    ^^^^
    SubField(String),
    Array(Vec<Box<Expression>>),
    Dereferenzation,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Variable {
    start_item: String,
    further_items: Vec<VariableItem>,
}

impl Parseable for Variable {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        match parser.advance() {
            Token::Identifier(var_or_field) => {
                let mut final_variable = Variable {
                    start_item: var_or_field.clone(),
                    further_items: Vec::new(),
                };

                loop {
                    match parser.advance() {
                        Token::LeftSquareBracket => {
                            let mut expressions = Vec::new();
                            loop {
                                let new_exp = parser.parse()?;
                                expressions.push(Box::new(new_exp));

                                if let Token::RightSquareBracket =
                                    parser.one_of(&[Token::Comma, Token::RightSquareBracket])?
                                {
                                    break;
                                }
                            }
                            final_variable
                                .further_items
                                .push(VariableItem::Array(expressions))
                        }
                        Token::Point => {
                            let ident = parser.expect_identifier()?;
                            final_variable
                                .further_items
                                .push(VariableItem::SubField(ident.to_string()));
                        }
                        Token::Caret => {
                            final_variable
                                .further_items
                                .push(VariableItem::Dereferenzation);
                        }
                        _ => {
                            parser.step_back(1);
                            break;
                        }
                    }
                }
                Ok(final_variable)
            }
            _ => parser.unexpected_token("expected variable or field identifier"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SingleExpressionArray {
    Single(Box<Expression>),
    Range(Box<Expression>, Box<Expression>),
}

impl Parseable for SingleExpressionArray {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let e = parser.parse()?;
        match parser.peek() {
            Token::DoublePoint => {
                parser.advance();
                let e_last = parser.parse()?;
                Ok(SingleExpressionArray::Range(Box::new(e), Box::new(e_last)))
            }
            _ => Ok(SingleExpressionArray::Single(Box::new(e))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub parameters: Vec<Box<Expression>>,
}

impl Parseable for FunctionCall {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        match parser.advance() {
            Token::Identifier(function_ident) => {
                let mut new_function = FunctionCall {
                    name: function_ident.clone(),
                    parameters: Vec::new(),
                };
                if let Token::LeftParen = parser.peek() {
                    loop {
                        parser.advance();
                        let new_param = parser.parse()?;
                        new_function.parameters.push(Box::new(new_param));

                        if let Token::RightParen =
                            parser.one_of(&[Token::Comma, Token::RightParen])?
                        {
                            break;
                        }
                    }
                }
                Ok(new_function)
            }

            _ => parser.unexpected_token("function call"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Factor {
    Constant(Constant),
    Variable(Variable),
    FunctionCall(FunctionCall),
    Expression(Box<Expression>),
    LogicalInversion(Box<Factor>),
    ExpressionArrays(Vec<SingleExpressionArray>),
}

impl Parseable for Factor {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        match parser.advance() {
            Token::Not => {
                let factor = parser.parse()?;
                Ok(Factor::LogicalInversion(Box::new(factor)))
            }
            Token::LeftParen => {
                let e = parser.parse()?;
                match parser.advance() {
                    Token::RightParen => Ok(Factor::Expression(Box::new(e))),
                    _ => parser.unexpected_token("expected ')'"),
                }
            }
            Token::LeftSquareBracket => {
                let mut arrs = Vec::new();

                loop {
                    if parser.expect(&Token::RightSquareBracket).is_ok() {
                        return Ok(Factor::ExpressionArrays(arrs));
                    }

                    let arr = parser.parse()?;
                    arrs.push(arr);

                    parser.expect(&Token::Comma).ok();
                }
            }
            _ => {
                parser.step_back(1);
                if let Ok(c) = parser.parse() {
                    return Ok(Factor::Constant(c));
                }

                if let Ok(v) = parser.parse() {
                    return Ok(Factor::Variable(v));
                }

                if let Ok(f) = parser.parse() {
                    return Ok(Factor::FunctionCall(f));
                }

                parser.unexpected_token("expected factor")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorPrimary {
    Multiply,
    Divide,
    IntegerDivide,
    Modulo,
    And,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub first_factor: Factor,
    pub following_factors: Vec<(OperatorPrimary, Factor)>,
}

impl Parseable for Term {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let first_factor = parser.parse()?;

        let mut res = Term {
            first_factor,
            following_factors: Vec::new(),
        };
        while let Ok(operator) = parser.one_of(&[
            Token::Asterisk,
            Token::Slash,
            Token::Div,
            Token::Mod,
            Token::And,
        ]) {
            let factor = parser.parse()?;
            let following_factor = match operator {
                Token::Slash => (OperatorPrimary::Divide, factor),
                Token::Div => (OperatorPrimary::IntegerDivide, factor),
                Token::Mod => (OperatorPrimary::Modulo, factor),
                Token::And => (OperatorPrimary::And, factor),
                _ => (OperatorPrimary::Multiply, factor),
            };
            res.following_factors.push(following_factor);
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSecondary {
    Plus,
    Minus,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleExpression {
    pub pre_sign: Option<Sign>,
    pub first_summand: Term,
    pub following_summands: Vec<(OperatorSecondary, Term)>,
}

impl Parseable for SimpleExpression {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let mut pre_sign = None;
        if let Ok(s) = parser.one_of(&[Token::Plus, Token::Minus]) {
            match s {
                Token::Plus => pre_sign = Some(Sign::Plus),
                _ => pre_sign = Some(Sign::Minus),
            }
        }

        let first_summand = parser.parse()?;

        let mut res = SimpleExpression {
            pre_sign,
            first_summand,
            following_summands: Vec::new(),
        };
        while let Ok(operator) = parser.one_of(&[Token::Plus, Token::Minus, Token::Or]) {
            let summand = parser.parse()?;
            let follow_summand = match operator {
                Token::Minus => (OperatorSecondary::Minus, summand),
                Token::Or => (OperatorSecondary::Or, summand),
                _ => (OperatorSecondary::Plus, summand),
            };
            res.following_summands.push(follow_summand);
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorTertiary {
    Equal,
    NotEqual,
    Smaller,
    SmallerOrEqual,
    Greater,
    GreaterOrEqual,
    In,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expression {
    pub first_operand: SimpleExpression,
    pub following_operands: Vec<(OperatorTertiary, SimpleExpression)>,
}

impl Parseable for Expression {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let first_operand = parser.parse()?;

        let mut res = Expression {
            first_operand,
            following_operands: Vec::new(),
        };
        while let Ok(operator) = parser.one_of(&[
            Token::Equal,
            Token::Smaller,
            Token::Greater,
            Token::NotEqual,
            Token::SmallerOrEqual,
            Token::GreateOrEqual,
            Token::In,
        ]) {
            let operand = parser.parse()?;
            let following_operands = match operator {
                Token::Smaller => (OperatorTertiary::Smaller, operand),
                Token::Greater => (OperatorTertiary::Greater, operand),
                Token::NotEqual => (OperatorTertiary::NotEqual, operand),
                Token::SmallerOrEqual => (OperatorTertiary::SmallerOrEqual, operand),
                Token::GreateOrEqual => (OperatorTertiary::GreaterOrEqual, operand),
                Token::In => (OperatorTertiary::In, operand),
                _ => (OperatorTertiary::Equal, operand),
            };
            res.following_operands.push(following_operands);
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterAs {
    Value,
    Pointer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParameter {
    pub as_pointer: ParameterAs,
    pub parameter: String,
    pub typ: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctionParameterList(Vec<FunctionParameter>);

impl Parseable for FunctionParameterList {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let mut fp = Vec::new();

        if parser.expect(&Token::LeftParen).is_ok() {
            loop {
                let mut as_pointer = ParameterAs::Value;
                if parser.expect(&Token::Var).is_ok() {
                    as_pointer = ParameterAs::Pointer;
                }

                let mut parameters = Vec::new();
                loop {
                    if let Ok(ident) = parser.expect_identifier() {
                        parameters.push(ident);
                    } else {
                        parser.unexpected_token("expected function parameter identifier")?;
                    }

                    match parser.advance() {
                        Token::Colon => break,
                        Token::Comma => {}
                        _ => parser.unexpected_token("expected function parameter seperator")?,
                    }
                }

                let typ = parser.expect_identifier()?;

                for parameter in parameters {
                    fp.push(FunctionParameter {
                        as_pointer,
                        parameter: parameter.into(),
                        typ: typ.into(),
                    })
                }

                match parser.advance() {
                    Token::RightParen => break,
                    Token::Semicolon => {}
                    _ => parser.unexpected_token("expected function parameter end")?,
                }
            }
        }

        Ok(FunctionParameterList(fp))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctionDeclaration {
    pub function_identifier: String,
    pub parameters: FunctionParameterList,
    pub return_type: Option<String>,
    pub body: Box<Block>,
}

impl Parseable for FunctionDeclaration {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let t = parser.advance();
        match t {
            Token::Function | Token::Procedure => {
                let ident = parser.expect_identifier()?;
                let mut res = FunctionDeclaration {
                    function_identifier: ident.into(),
                    ..Default::default()
                };
                res.parameters = parser.parse()?;
                if let Token::Function = t {
                    parser.expect(&Token::Colon)?;
                    res.return_type = Some(parser.expect_identifier()?.into());
                }
                parser.expect(&Token::Begin)?;
                let block = parser.parse()?;
                res.body = Box::new(block);
                parser.expect(&Token::End)?;
                Ok(res)
            }
            _ => parser.unexpected_token("expected function identifier"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Block {
    pub consts: Vec<(String, Constant)>,
    pub types: Vec<(String, Type)>,
    pub variables: Vec<(String, Type)>,
    pub function_declarations: Vec<FunctionDeclaration>,
    pub body: Vec<Statement>,
}

impl Parseable for Block {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let mut res = Block::default();

        loop {
            match parser.advance() {
                Token::Const => {
                    while let Ok(ident) = parser.expect_identifier() {
                        parser.expect(&Token::Equal)?;
                        let constant = parser.parse()?;
                        res.consts.push((ident.into(), constant));

                        parser.expect(&Token::Semicolon)?;
                    }
                }
                Token::Type => {
                    while let Ok(ident) = parser.expect_identifier() {
                        parser.expect(&Token::Equal)?;
                        let typ = parser.parse()?;
                        res.types.push((ident.into(), typ));

                        parser.expect(&Token::Semicolon)?;
                    }
                }
                Token::Var => {
                    while let Ok(pre_ident) = parser.expect_identifier() {
                        let mut idents = Vec::new();
                        idents.push(pre_ident.to_string());
                        loop {
                            match parser.advance() {
                                Token::Comma => {}
                                Token::Colon => break,
                                _ => parser.unexpected_token("expected separator in var block")?,
                            }
                            idents.push(parser.expect_identifier()?.into());

                            parser.expect(&Token::Semicolon)?;
                        }
                        let vartype: Type = parser.parse()?;
                        for ident in idents {
                            res.variables.push((ident, vartype.clone()));
                        }
                    }
                }
                Token::Procedure | Token::Function => {
                    parser.step_back(1);
                    res.function_declarations.push(parser.parse()?);
                }

                Token::Begin => break,
                _ => parser.unexpected_token("expected block part")?,
            }
        }

        loop {
            res.body.push(parser.parse()?);
            match parser.advance() {
                Token::End => break,
                &Token::Comma => {}
                _ => parser.unexpected_token("expected expression seperator")?,
            }
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfCondition {
    condition: Expression,
    statement: Box<Statement>,
    else_statements: Option<Box<Statement>>,
}

impl Parseable for IfCondition {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let condition = parser.parse()?;
        parser.expect(&Token::Then)?;
        let mut res: IfCondition = IfCondition {
            condition,
            statement: Box::new(parser.parse()?),
            else_statements: None,
        };
        if parser.expect(&Token::Else).is_ok() {
            res.else_statements = Some(Box::new(parser.parse()?));
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseItems {
    constants: Vec<Constant>,
    statement: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    expression: Expression,
    items: Vec<CaseItems>,
    else_statement: Option<Box<Statement>>,
}

impl Parseable for Case {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let expression = parser.parse()?;
        let mut res = Case {
            expression,
            items: Vec::new(),
            else_statement: None,
        };
        parser.expect(&Token::Of)?;
        loop {
            let mut constants = Vec::new();
            loop {
                constants.push(parser.parse()?);
                match parser.advance() {
                    Token::Comma => {}
                    Token::Colon => break,
                    _ => parser.unexpected_token("expected case constant separator")?,
                }
            }
            res.items.push(CaseItems {
                constants,
                statement: Box::new(parser.parse()?),
            });
            parser.expect(&Token::Semicolon)?;

            if let Ok(s) = parser.one_of(&[Token::Else, Token::End]) {
                if s.clone() == Token::Else {
                    res.else_statement = Some(Box::new(parser.parse()?));
                    parser.expect(&Token::Semicolon)?;
                    parser.expect(&Token::End)?;
                    break;
                } else {
                    break;
                }
            }
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileLoop {
    check: Expression,
    statement: Box<Statement>,
}

impl Parseable for WhileLoop {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let check = parser.parse()?;
        parser.expect(&Token::Do)?;
        let statement = Box::new(parser.parse()?);
        Ok(WhileLoop { check, statement })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatLoop {
    statements: Vec<Statement>,
    check_until: Expression,
}

impl Parseable for RepeatLoop {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let mut statements = Vec::new();
        loop {
            statements.push(parser.parse()?);

            match parser.advance() {
                Token::Repeat => break,
                Token::Semicolon => {}
                _ => parser.unexpected_token("expected repeat statement separator")?,
            }
        }
        Ok(RepeatLoop {
            statements,
            check_until: parser.parse()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForLoopDirection {
    To,
    Downto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForLoop {
    var_name: String,
    first: Expression,
    last: Expression,
    direction: ForLoopDirection,
    statement: Box<Statement>,
}

impl Parseable for ForLoop {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        let var_name = parser.expect_identifier()?;

        parser.expect(&Token::Assignment)?;

        let first = parser.parse()?;

        let predirection = parser.one_of(&[Token::To, Token::Downto])?;
        let direction = if *predirection == Token::Downto {
            ForLoopDirection::Downto
        } else {
            ForLoopDirection::To
        };

        let last = parser.parse()?;

        parser.expect(&Token::Do)?;

        Ok(ForLoop {
            var_name: var_name.into(),
            first,
            direction,
            last,
            statement: Box::new(parser.parse()?),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assignment(String, Expression),
    FunctionCall(FunctionCall),
    SubStatements(Vec<Box<Statement>>),
    IfCondition(IfCondition),
    Case(Case),
    WhileLoop(WhileLoop),
    RepeatLoop(RepeatLoop),
    ForLoop(ForLoop),
}

impl Parseable for Statement {
    fn parse(parser: &mut TokenParser) -> Result<Self, ParseError> {
        match parser.advance() {
            Token::Identifier(id) => {
                if parser.expect(&Token::Assignment).is_ok() {
                    parser.advance();

                    Ok(Self::Assignment(id.into(), parser.parse()?))
                } else {
                    parser.step_back(1);

                    Ok(Self::FunctionCall(parser.parse()?))
                }
            }
            Token::Begin => {
                let mut statements = Vec::new();
                loop {
                    statements.push(Box::new(parser.parse()?));

                    match parser.advance() {
                        Token::Semicolon => {}
                        Token::End => break,
                        _ => parser.unexpected_token("missing end for substatements")?,
                    }
                }
                Ok(Self::SubStatements(statements))
            }
            Token::If => Ok(Self::IfCondition(parser.parse()?)),
            Token::Case => Ok(Self::Case(parser.parse()?)),
            Token::While => Ok(Self::WhileLoop(parser.parse()?)),
            Token::Repeat => Ok(Self::RepeatLoop(parser.parse()?)),
            Token::For => Ok(Self::ForLoop(parser.parse()?)),
            _ => parser.unexpected_token("expected token for statement")?,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::lexer;
    use crate::lexer::tokenize;
    use crate::new_parser;
    use crate::parser;

    use super::*;

    fn print_from_code(code: String, expected: Variable) {
        let tokens = tokenize(&code);
        if let Ok(input) = tokens {
            // println!("Tokens: {:?}", input);
            let mut parser = TokenParser::new(&input);
            let v = Variable::parse(&mut parser);
            if let Ok(v) = v {
                // println!("Parser Entities: {:?}", v);
                assert_eq!(v, expected);
            }
        } else {
            assert!(tokens.is_err());
        }
    }

    #[test]
    fn test_variable() {
        print_from_code(
            "test_a".into(),
            Variable {
                start_item: "test_a".into(),
                further_items: Vec::new(),
            },
        );
        print_from_code(
            "test_b.a".into(),
            Variable {
                start_item: "test_b".into(),
                further_items: vec![VariableItem::SubField("a".into())],
            },
        );
    }

    // #[test]
    // fn test_unsigned_consts() {
    //     fn check(expected: UnsignedConstant, input: &[Token]) {
    //         let parser = TokenParser::new(&input);
    //         assert_eq!(parsed, expected);
    //     }
    //     s
    // check(
    //     UnsignedConstant::Identifier("foo".into()),
    //     Token::Identifier("foo".into()),
    // );
    // check(
    //     UnsignedConstant::UnsignedInteger(42),
    //     Token::UnsignedInteger(42),
    // );
    // check(
    //     UnsignedConstant::UnsignedFloat(420.69.into()),
    //     Token::UnsignedFloat(420.69.into()),
    // );
    // check(
    //     UnsignedConstant::String("bar".into()),
    //     Token::ConstantString("bar".into()),
    // );
    // }

    // #[test]
    // fn test_signed_consts() {
    //     fn check(expected: Constant, input: impl AsRef<[Token]>) {
    //         let parsed = Constant::parse(parser{input).unwrap();
    //         assert_eq!(parsed, expected);
    //     }

    //     check(
    //         Constant::identifier("restaurant", None),
    //         [Token::identifier("restaurant")],
    //     );

    //     check(
    //         Constant::identifier("abc", Sign::Minus),
    //         [Token::Minus, Token::Identifier("abc".into())],
    //     );

    //     check(Constant::number(123u64), [Token::UnsignedInteger(123)]);

    //     check(
    //         Constant::number(-123),
    //         [Token::Minus, Token::UnsignedInteger(123)],
    //     );

    //     check(Constant::number(123.045), [Token::unsigned_float(123.045)]);

    //     check(
    //         Constant::number(-123.045),
    //         [Token::Minus, Token::unsigned_float(123.045)],
    //     );

    //     check(Constant::literal("olap"), [Token::constant_string("olap")]);
    // }

    // #[test]
    // fn test_simple_types() {
    //     fn check(name: &str, expected: SimpleType, input: impl AsRef<[Token]>) {
    //         let (parsed, _) = SimpleType::parse(input.as_ref()).unwrap();
    //         assert_eq!(parsed, expected, "{name} failed");
    //     }

    //     check(
    //         "SoloType",
    //         SimpleType::solo_type("whisper"),
    //         [Token::identifier("whisper")],
    //     );

    //     check(
    //         "Ordinal",
    //         SimpleType::ordinal(["abc", "def"]),
    //         [
    //             Token::LeftParen,
    //             Token::identifier("abc"),
    //             Token::Comma,
    //             Token::identifier("def"),
    //             Token::RightParen,
    //         ],
    //     );

    //     check(
    //         "Array",
    //         SimpleType::static_array(
    //             Constant::identifier("bla", Sign::Minus),
    //             Constant::number(-23),
    //         ),
    //         [
    //             Token::Minus,
    //             Token::identifier("bla"),
    //             Token::DoublePoint,
    //             Token::Minus,
    //             Token::UnsignedInteger(23),
    //         ],
    //     );
    // }
}
