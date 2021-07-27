use wordcut_engine::load_dict;
use wordcut_engine::Wordcut;
use std::path::Path;

fn main() {
    let dict_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/dict.txt"
    ));
    let dict = load_dict(dict_path).unwrap();
    let wordcut = Wordcut::new(dict);
    println!("{}", wordcut.put_delimiters("หมากินไก่", "|"));
}
