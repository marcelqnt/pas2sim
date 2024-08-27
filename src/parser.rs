// pub enum Block {
//     Label,
//     Const(Vec<Const>),
//     Type,
//     Var,
//     Procedure,
//     Function,
//     Statement,
// }

// pub struct Const {
//     identifier: String,
//     constant: Constant,
// }

// pub enum Constant {

// }

use anyhow::Context;
use ordered_float::OrderedFloat;

use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq, Eq)]
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
                next_tokens.get(0)
            ))
        }
    };

    Ok((res, &next_tokens[1..]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constant {
    Identifier(Identifier),
    Literal(String),
    Number(Number),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number {
    kind: NumberKind,
    sign: Option<Sign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberKind {
    Signed(i64),
    Unsigned(u64),
    Real(OrderedFloat<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Plus,
    Minus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier {
    name: String,
    sign: Option<Sign>,
}

pub fn parse_constant(next_tokens: &[Token]) -> anyhow::Result<(Constant, &[Token])> {
    let mut i = 1;
    let mut sign = None;
    let res = loop {
        match next_tokens {
            [Token::Identifier(s)] => {
                break Constant::Identifier(Identifier {
                    name: s.clone(),
                    sign,
                })
            }
            [Token::UnsignedInteger(u)] => {
                break Constant::Number(Number {
                    kind: NumberKind::Unsigned(*u),
                    sign,
                })
            }
            [Token::UnsignedFloat(f)] => {
                break Constant::Number(Number {
                    kind: NumberKind::Real(*f),
                    sign,
                })
            }
            [Token::ConstantString(s)] => break Constant::Literal(s.clone()),
            [Token::Plus, Token::Plus]
            | [Token::Minus, Token::Minus]
            | [Token::Plus, Token::Minus]
            | [Token::Minus, Token::Plus] => {
                return Err(anyhow::anyhow!(
                    "no double sign is allowed: {:?}",
                    &next_tokens[..2]
                ))
            }
            [Token::Plus] => {
                i += 1;
                sign = Some(Sign::Plus)
            }
            [Token::Minus] => {
                i += 1;
                sign = Some(Sign::Minus)
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unexpected next token for signed constant: {:?}",
                    next_tokens.get(0)
                ))
            }
        };
    };

    Ok((res, &next_tokens[i..]))
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
}
