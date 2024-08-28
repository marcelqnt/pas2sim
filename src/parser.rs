use anyhow::Context;
use ordered_float::OrderedFloat;

use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub enum UnsignedConstant {
    Identifier(String),
    UnsignedInteger(u64),
    UnsignedFloat(OrderedFloat<f64>),
    Nil,
    String(String),
}

pub fn parse_unsigned_constant(
    next_tokens: &[Token],
) -> anyhow::Result<(UnsignedConstant, &[Token])> {
    use UnsignedConstant::*;
    let res = match next_tokens {
        [Token::Identifier(s)] => Identifier(s.clone()),
        [Token::UnsignedInteger(u)] => UnsignedInteger(*u),
        [Token::UnsignedFloat(f)] => UnsignedFloat(*f),
        [Token::Nil] => Nil,
        [Token::ConstantString(s)] => String(s.clone()),
        _ => {
            return Err(anyhow::anyhow!(
                "Unexpected next token for unsigned constant: {:?}",
                next_tokens.first()
            ))
        }
    };

    Ok((res, &next_tokens[1..]))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Identifier(ConstantIdentifier),
    Literal(String),
    Number(Number),
}

impl Constant {
    pub fn parse(next_tokens: &[Token]) -> Result<(Constant, &[Token]), ParseError> {
        let mut i = 0;
        let mut sign = None;
        let res = loop {
            match &next_tokens[i..] {
                [Token::Identifier(s), ..] => {
                    break Constant::Identifier(ConstantIdentifier {
                        name: s.clone(),
                        sign,
                    })
                }
                [Token::UnsignedInteger(u), ..] if sign.map(Sign::is_plus).unwrap_or(true) => {
                    break Constant::Number(Number::Unsigned(*u))
                }
                [Token::UnsignedInteger(u), ..] if sign.map(Sign::is_minus).unwrap_or(false) => {
                    let i: i64 = (*u)
                        .try_into()
                        .context("failed to convert number to signed integer")?;
                    break Constant::Number(Number::Signed(-i));
                }
                [Token::UnsignedFloat(f), ..] if sign.map(Sign::is_plus).unwrap_or(true) => {
                    break Constant::Number(Number::Real(*f))
                }
                [Token::UnsignedFloat(f), ..] if sign.map(Sign::is_minus).unwrap_or(false) => {
                    break Constant::Number(Number::Real(-f))
                }
                [Token::ConstantString(s), ..] => break Constant::Literal(s.clone()),
                [Token::Plus, Token::Plus, ..]
                | [Token::Minus, Token::Minus, ..]
                | [Token::Plus, Token::Minus, ..]
                | [Token::Minus, Token::Plus, ..] => {
                    return Err(ParseError::UnexpectedToken(
                        next_tokens.first().cloned(),
                        "no double sign is allowed".into(),
                    ));
                }
                [Token::Plus, ..] => {
                    i += 1;
                    sign = Some(Sign::Plus)
                }
                [Token::Minus, ..] => {
                    i += 1;
                    sign = Some(Sign::Minus)
                }
                _ => {
                    return Err(ParseError::UnexpectedToken(
                        next_tokens.first().cloned(),
                        "unexpected next token for signed constant".into(),
                    ));
                }
            };
        };

        Ok((res, &next_tokens[i + 1..]))
    }

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

impl Sign {
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

impl SimpleType {
    pub fn parse(next_tokens: &[Token]) -> anyhow::Result<(Self, &[Token])> {
        let mut i = 0;
        let mut ordinaries = Vec::new();
        let mut mode = None;

        let res = loop {
            if i == 0 {
                if let Ok((c, [Token::DoublePoint, rest @ ..])) = Constant::parse(next_tokens) {
                    match Constant::parse(rest) {
                        Ok((c2, rest)) => return Ok((SimpleType::StaticArray(c, c2), rest)),
                        Err(e) => return Err(e.into()),
                    }
                }
            }

            match &next_tokens[i..] {
                [Token::LeftParen, ..] => {
                    i += 1;
                    mode = Some(SimpleType::Ordinal(Vec::new()));
                }
                [Token::Identifier(ident), ..] => {
                    i += 1;
                    match mode {
                        Some(SimpleType::Ordinal(_)) => ordinaries.push(ident.clone()),
                        None => break SimpleType::SoloType(ident.clone()),
                        _ => {
                            return Err(anyhow::anyhow!(
                                "Unexpected next token for simple type (solo/ordinary): {:?}",
                                next_tokens.first()
                            ))
                        }
                    }
                }
                [Token::RightParen, ..] if matches!(mode, Some(SimpleType::Ordinal(_))) => {
                    i += 1;
                    break SimpleType::Ordinal(std::mem::take(&mut ordinaries));
                }
                [Token::Comma, ..] if matches!(mode, Some(SimpleType::Ordinal(_))) => {
                    i += 1;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unexpected next token for simple type: {:?}",
                        next_tokens.first()
                    ))
                }
            }
        };

        Ok((res, &next_tokens[(i + 1).min(next_tokens.len())..]))
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

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub typ: Box<Type>,
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

pub type Variable = Vec<VariableItem>;

#[derive(Debug, Clone, PartialEq)]
pub enum SingleExpressionArray {
    Single(Box<Expression>),
    Range(Box<Expression>, Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub parameters: Vec<Box<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Factor {
    UnsignedConstant(UnsignedConstant),
    Variable(Variable),
    FunctionCall(FunctionCall),
    Expression(Box<Expression>),
    LogicalInversion(Box<Factor>),
    ExpressionArrays(Vec<SingleExpressionArray>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorPrimary {
    Multiply,
    Divide,
    IntegerMultiply,
    Modulo,
    And,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub first_factor: Factor,
    pub following_factors: Vec<(OperatorPrimary, Factor)>,
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

pub type FunctionParameterList = Vec<FunctionParameter>;

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub function_identifier: String,
    pub parameters: FunctionParameterList,
    pub return_type: Option<String>,
    pub body: Box<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockPreamble {
    Label(Vec<Number>),
    Const(Vec<(String, Constant)>),
    Type(Vec<(String, Type)>),
    Var(Vec<(String, Type)>),
    FunctionCall(FunctionCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub pre_parts: Vec<BlockPreamble>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfCondition {
    condition: Expression,
    statements: Box<Statement>,
    else_statements: Option<Box<Statement>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseItems {
    constants: Vec<Constant>,
    statement: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    expression: Expression,
    items: CaseItems,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileLoop {
    check: Expression,
    statement: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatLoop {
    statements: Vec<Statement>,
    check: Expression,
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
    statement: Statement,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assignment(Variable, Expression),
    FunctionCall(FunctionCall),
    SubStatements(Vec<Statement>),
    IfCondition(IfCondition),
    Case(Case),
    WhileLoop(WhileLoop),
    RepeatLoop(RepeatLoop),
}

// pub fn parse_main(next_tokens: &[Token]) -> anyhow::Result<Block> {
//     let (block, _) = parse_block(next_tokens)?;
//     Ok(block)
// }

// fn next_token_starts_block(next_token: &Token) -> bool {
//     match next_token {
//         Token::Label
//         | Token::Const
//         | Token::Type
//         | Token::Var
//         | Token::Procedure
//         | Token::Function
//         | Token::Begin => true,
//         _ => false,
//     }
// }

// fn parse_block(next_tokens: &[Token]) -> anyhow::Result<(Block, &[Token])> {
//     // match next_token {
//     //     [Token::If, _, Token::Then, ..] => {
//     //         parse_block(&next_token[3..])?;
//     //     }
//     //     [a, b, ..] => {}
//     //     _ => {}
//     // }

//     match next_tokens {
//         [Token::Const] => {
//             let consts = Vec::new();
//             let i = 1;
//             match next_tokens[i] {
//                 Token::Identifier(str) =>
//                 _ => {
//                     return Err(anyhow::anyhow!(
//                         "Unexpected next token for const block: {:?}",
//                         next_tokens.get(i)
//                     ))
//                 }
//             }
//         }
//         _ => {
//             return Err(anyhow::anyhow!(
//                 "Unexpected next token for block: {:?}",
//                 next_tokens.get(0)
//             ))
//         }
//     }
// }

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_unsigned_consts() {
        fn check(expected: UnsignedConstant, input: Token) {
            let (parsed, _) = parse_unsigned_constant(&[input]).unwrap();
            assert_eq!(parsed, expected);
        }

        check(
            UnsignedConstant::Identifier("foo".into()),
            Token::Identifier("foo".into()),
        );
        check(
            UnsignedConstant::UnsignedInteger(42),
            Token::UnsignedInteger(42),
        );
        check(
            UnsignedConstant::UnsignedFloat(420.69.into()),
            Token::UnsignedFloat(420.69.into()),
        );
        check(
            UnsignedConstant::String("bar".into()),
            Token::ConstantString("bar".into()),
        );
    }

    #[test]
    fn test_signed_consts() {
        fn check(expected: Constant, input: impl AsRef<[Token]>) {
            let (parsed, _) = Constant::parse(input.as_ref()).unwrap();
            assert_eq!(parsed, expected);
        }

        check(
            Constant::identifier("restaurant", None),
            [Token::identifier("restaurant")],
        );

        check(
            Constant::identifier("abc", Sign::Minus),
            [Token::Minus, Token::Identifier("abc".into())],
        );

        check(Constant::number(123u64), [Token::UnsignedInteger(123)]);

        check(
            Constant::number(-123),
            [Token::Minus, Token::UnsignedInteger(123)],
        );

        check(Constant::number(123.045), [Token::unsigned_float(123.045)]);

        check(
            Constant::number(-123.045),
            [Token::Minus, Token::unsigned_float(123.045)],
        );

        check(Constant::literal("olap"), [Token::constant_string("olap")]);
    }

    #[test]
    fn test_simple_types() {
        fn check(name: &str, expected: SimpleType, input: impl AsRef<[Token]>) {
            let (parsed, _) = SimpleType::parse(input.as_ref()).unwrap();
            assert_eq!(parsed, expected, "{name} failed");
        }

        check(
            "SoloType",
            SimpleType::solo_type("whisper"),
            [Token::identifier("whisper")],
        );

        check(
            "Ordinal",
            SimpleType::ordinal(["abc", "def"]),
            [
                Token::LeftParen,
                Token::identifier("abc"),
                Token::Comma,
                Token::identifier("def"),
                Token::RightParen,
            ],
        );

        check(
            "Array",
            SimpleType::static_array(
                Constant::identifier("bla", Sign::Minus),
                Constant::number(-23),
            ),
            [
                Token::Minus,
                Token::identifier("bla"),
                Token::DoublePoint,
                Token::Minus,
                Token::UnsignedInteger(23),
            ],
        );
    }
}
