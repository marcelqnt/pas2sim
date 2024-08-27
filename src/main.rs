use lexer::tokenize;

pub mod lexer;
pub mod parser;

fn main() {
    let result_string = std::fs::read_to_string("src/testdata.pas");

    if let Ok(s) = result_string {
        println!("Text:\n\r{}", s);

        println!("Tokens {:#?}", tokenize(&s).unwrap());
    }
}
