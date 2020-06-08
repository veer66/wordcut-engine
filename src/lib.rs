#[macro_use]
extern crate lazy_static;
extern crate prefixtree;
#[macro_use]
extern crate serde_derive;

use self::prefixtree::{prefix_tree_from_str, PrefixTree};
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;

pub type Dict = PrefixTree<char, bool>;

pub fn create_prefix_tree(words: &[&str]) -> PrefixTree<char, bool> {
    let words_payloads: Vec<(&str, bool)> = words.iter().map(|&word| (word, true)).collect();
    prefix_tree_from_str(&words_payloads[..])
}

#[derive(Clone, PartialEq, Eq, Copy, Debug, Serialize, Deserialize)]
pub enum EdgeType {
    Init,
    Dict,
    Unk,
    Punc,
    Latin,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub w: usize,
    pub unk: usize,
    pub p: usize,
    pub etype: EdgeType,
}

impl Edge {
    pub fn is_unk(&self) -> bool {
        self.etype == EdgeType::Unk
    }

    pub fn better_than(&self, o: &Edge) -> bool {
        if self.unk < o.unk {
            return true;
        }

        if self.unk > o.unk {
            return false;
        }

        if self.w < o.w {
            return true;
        }

        if self.w > o.w {
            return false;
        }

        if o.is_unk() && !self.is_unk() {
            return true;
        }

        return false;
    }

    pub fn better(a: &Option<Edge>, b: &Option<Edge>) -> bool {
        if a.is_none() {
            return false;
        }

        if b.is_none() {
            return true;
        }

        return a.unwrap().better_than(&b.unwrap());
    }
}

#[derive(Debug, PartialEq)]
enum State {
    Init,
    Pat,
    PatFinal,
    NonPat,
    NonPatFinal,
}

type CharPredicate = Fn(char) -> bool;

pub struct PatEdgeBuilder {
    i: usize,
    pub start: usize,
    state: State,
    is_pat_char: Box<CharPredicate>,
    etype: EdgeType,
}

impl PatEdgeBuilder {
    pub fn new(is_pat_char: Box<CharPredicate>, etype: EdgeType) -> PatEdgeBuilder {
        PatEdgeBuilder {
            start: 0,
            i: 0,
            state: State::Init,
            is_pat_char: is_pat_char,
            etype: etype,
        }
    }

    fn to_text_state(&mut self, nch: Option<char>) -> State {
        match nch {
            Some(_nch) => {
                if (self.is_pat_char)(_nch) {
                    State::NonPatFinal
                } else {
                    State::NonPat
                }
            }
            None => State::NonPatFinal,
        }
    }

    fn to_space_state(&mut self, nch: Option<char>) -> State {
        match nch {
            Some(_nch) => {
                if (self.is_pat_char)(_nch) {
                    State::Pat
                } else {
                    State::PatFinal
                }
            }
            None => State::PatFinal,
        }
    }

    fn to_another_state(&mut self, ch: char, nch: Option<char>) -> State {
        if (self.is_pat_char)(ch) {
            self.to_space_state(nch)
        } else {
            self.to_text_state(nch)
        }
    }

    pub fn transit(&mut self, ch: char, nch: Option<char>) {
        match self.state {
            State::Init => {
                self.start = self.i;
                self.state = self.to_another_state(ch, nch)
            }
            State::NonPat => {
                self.state = self.to_another_state(ch, nch);
            }
            State::NonPatFinal => {
                self.start = self.i;
                self.state = self.to_space_state(nch);
            }
            State::PatFinal => {
                self.start = self.i;
                self.state = self.to_text_state(nch)
            }
            State::Pat => {
                self.state = self.to_another_state(ch, nch);
            }
        };
        self.i += 1;
    }

    pub fn is_pat_final(&self) -> bool {
        self.state == State::PatFinal
    }
}

pub trait EdgeBuilder {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge>;
}

pub struct EdgeBuildingContext<'a> {
    pub text: &'a Vec<char>,
    pub i: usize,
    pub ch: char,
    pub left_boundary: usize,
    pub best_edge: Option<Edge>,
}

impl EdgeBuilder for PatEdgeBuilder {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge> {
        let next_char = if context.i + 1 == context.text.len() {
            None
        } else {
            Some(context.text[context.i + 1])
        };
        self.transit(context.ch, next_char);
        if self.is_pat_final() {
            let source = path[self.start];
            Some(Edge {
                p: self.start,
                etype: self.etype,
                w: source.w + 1,
                unk: source.unk,
            })
        } else {
            None
        }
    }
}

fn is_latin(ch: char) -> bool {
    (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
}

pub fn create_latin_edge_builder() -> PatEdgeBuilder {
    PatEdgeBuilder::new(Box::new(is_latin), EdgeType::Latin)
}

lazy_static! {
    static ref PUNC_SET: HashSet<char> = {
        let mut m = HashSet::new();
        m.insert(' ');
        m.insert('\n');
        m.insert('\t');
        m.insert('\r');
        m.insert('"');
        m.insert('(');
        m.insert(')');
        m.insert('“');
        m.insert('”');
        m
    };
}

fn is_punc(ch: char) -> bool {
    PUNC_SET.contains(&ch)
}

pub fn create_punc_edge_builder() -> PatEdgeBuilder {
    PatEdgeBuilder::new(Box::new(is_punc), EdgeType::Punc)
}

pub struct UnkEdgeBuilder {}

impl UnkEdgeBuilder {
    pub fn new() -> UnkEdgeBuilder {
        UnkEdgeBuilder {}
    }
}

impl EdgeBuilder for UnkEdgeBuilder {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge> {
        if context.best_edge.is_some() {
            return None;
        }

        let source = path[context.left_boundary];
        Some(Edge {
            p: context.left_boundary,
            etype: EdgeType::Unk,
            unk: source.unk + 1,
            w: source.w + 1,
        })
    }
}

#[derive(Clone)]
struct Pointer {
    node_id: usize,
    s: usize,
    offset: usize,
    is_final: bool,
}

impl Pointer {
    fn update(&mut self, dict: &Dict, ch: char) -> bool {
        match dict.seek(&(self.node_id as u32, self.offset as u32, ch)) {
            None => false,
            Some(&(child_id, is_final, _)) => {
                self.node_id = child_id as usize;
                self.is_final = is_final;
                self.offset += 1;
                true
            }
        }
    }

    fn gen_edge(&self, path: &[Edge]) -> Edge {
        let source = path[self.s];
        Edge {
            etype: EdgeType::Dict,
            p: self.s,
            w: source.w + 1,
            unk: source.unk,
        }
    }
}

pub struct DictEdgeBuilder<'a> {
    dict: &'a Dict,
    pointers: Vec<Pointer>,
}

impl<'a> DictEdgeBuilder<'a> {
    pub fn new(dict: &Dict) -> DictEdgeBuilder {
        const MAX_SIZE: usize = 0xFF;
        DictEdgeBuilder {
            dict: dict,
            pointers: Vec::with_capacity(MAX_SIZE),
        }
    }

    fn add_pointer(&mut self, context: &EdgeBuildingContext) {
        self.pointers.push(Pointer {
            node_id: 0,
            offset: 0,
            is_final: false,
            s: context.i,
        });
    }

    fn update_pointers(&mut self, context: &EdgeBuildingContext) {
        let mut j = 0;
        for i in 0..self.pointers.len() {
            let valid = self.pointers[i].update(self.dict, context.ch);
            if valid {
                if j < i {
                    self.pointers[j] = self.pointers[i].clone()
                }
                j += 1
            }
        }
        self.pointers.truncate(j);
    }

    fn gen_edge(&self, pointers: &[Pointer], path: &[Edge]) -> Option<Edge> {
        let mut best_edge: Option<Edge> = None;
        for pointer in pointers {
            if pointer.is_final {
                let edge = pointer.gen_edge(path);
                if best_edge.is_none() {
                    best_edge = Some(edge)
                } else if edge.better_than(&best_edge.unwrap()) {
                    best_edge = Some(edge)
                }
            }
        }
        return best_edge;
    }
}

impl<'a> EdgeBuilder for DictEdgeBuilder<'a> {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge> {
        self.add_pointer(context);
        self.update_pointers(context);
        self.gen_edge(&self.pointers, path)
    }
}

pub fn build_path(dict: &Dict, text: &Vec<char>) -> Vec<Edge> {
    let mut builders: Vec<Box<EdgeBuilder>> = vec![
        Box::new(DictEdgeBuilder::new(dict)),
        Box::new(create_latin_edge_builder()),
        Box::new(create_punc_edge_builder()),
        Box::new(UnkEdgeBuilder::new()),
    ];

    let mut path = vec![];
    path.push(Edge {
        w: 0,
        unk: 0,
        p: 0,
        etype: EdgeType::Init,
    });

    let mut context = EdgeBuildingContext {
        text: &text,
        i: 0,
        ch: '\0',
        left_boundary: 0,
        best_edge: None,
    };

    for i in 0..text.len() {
        context.ch = text[i];
        context.i = i;
        context.best_edge = None;

        for builder in &mut builders {
            let edge = builder.build(&context, &path);
            if Edge::better(&edge, &context.best_edge) {
                context.best_edge = edge
            }
        }

        if context.best_edge.is_none() {
            panic!("Best edge cannot be None")
        }

        path.push(context.best_edge.unwrap());

        if !context.best_edge.unwrap().is_unk() {
            context.left_boundary = i + 1
        }
    }

    return path;
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagEdge {
    pub s: usize,
    pub e: usize,
    pub etype: EdgeType,
}

pub type Dag = Vec<Vec<DagEdge>>;

pub trait DagEdgeBuilder {
    fn build_dag_edges(&mut self, context: &EdgeBuildingContext) -> Vec<DagEdge>;
}

impl<'a> DagEdgeBuilder for DictEdgeBuilder<'a> {
    fn build_dag_edges(&mut self, context: &EdgeBuildingContext) -> Vec<DagEdge> {
        self.add_pointer(context);
        self.update_pointers(context);
        //self.gen_edge(&self.pointers, path)
        self.pointers
            .iter()
            .filter(|p| p.is_final)
            .map(|p| DagEdge {
                s: p.s,
                e: context.i + 1,
                etype: EdgeType::Dict,
            })
            .collect()
    }
}

impl DagEdgeBuilder for PatEdgeBuilder {
    fn build_dag_edges(&mut self, context: &EdgeBuildingContext) -> Vec<DagEdge> {
        let next_char = if context.i + 1 == context.text.len() {
            None
        } else {
            Some(context.text[context.i + 1])
        };
        self.transit(context.ch, next_char);
        if self.is_pat_final() {
            vec![DagEdge {
                s: self.start,
                e: context.i + 1,
                etype: self.etype,
            }]
        } else {
            vec![]
        }
    }
}

pub fn build_dag(dict: &Dict, text: &Vec<char>) -> Dag {
    let mut builders: Vec<Box<DagEdgeBuilder>> = vec![
        Box::new(DictEdgeBuilder::new(dict)),
        Box::new(create_latin_edge_builder()),
        Box::new(create_punc_edge_builder()),
    ];

    let mut dag = Vec::with_capacity(text.len() + 1);

    for _ in 0..text.len() + 1 {
        dag.push(vec![]);
    }
    dag[0].push(DagEdge {
        s: 0,
        e: 0,
        etype: EdgeType::Init,
    });
    let mut context = EdgeBuildingContext {
        text: &text,
        i: 0,
        ch: '\0',
        left_boundary: 0,
        best_edge: None,
    };

    for i in 0..text.len() {
        context.ch = text[i];
        context.i = i;
        context.best_edge = None;

        for builder in &mut builders {
            for edge in builder.build_dag_edges(&context) {
                dag[edge.e].push(edge)
            }
        }
    }

    let mut left_boundary = 0;
    for i in 1..text.len() + 1 {
        if dag[i].len() == 0 {
            dag[i].push(DagEdge {
                s: left_boundary,
                e: i,
                etype: EdgeType::Unk,
            });
        } else {
            left_boundary = i;
        }
    }

    return dag;
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TextRange {
    pub s: usize,
    pub e: usize,
}

pub fn path_to_ranges(path: &[Edge]) -> Vec<TextRange> {
    let len = path.len();

    if len == 0 {
        return vec![];
    }

    let mut ranges: Vec<TextRange> = Vec::with_capacity(len);
    let mut e = len - 1;
    while e > 0 {
        let edge = &path[e];
        let s = edge.p;
        ranges.push(TextRange { s: s, e: e });
        e = s;
    }
    ranges.reverse();
    return ranges;
}

pub fn path_to_byte_ranges(path: &[Edge], text: &[char]) -> Vec<TextRange> {
    let char_ranges = path_to_ranges(path);
    let mut ranges: Vec<TextRange> = Vec::with_capacity(char_ranges.len());
    let mut global_byte_offset = 0;
    for r in char_ranges {
        let mut word_byte_offset = 0;
        for i in r.s..r.e {
            word_byte_offset += text[i].len_utf8();
        }
        ranges.push(TextRange {
            s: global_byte_offset,
            e: global_byte_offset + word_byte_offset,
        });
        global_byte_offset += word_byte_offset;
    }
    return ranges;
}

pub fn path_to_str_vec(path: &[Edge], text: &[char]) -> Vec<String> {
    let ranges = path_to_ranges(path);
    let mut str_vec: Vec<String> = Vec::with_capacity(ranges.len());
    for r in ranges {
        let mut buf = String::with_capacity(3 * (r.e - r.s + 1));
        for i in r.s..r.e {
            buf.push(text[i]);
        }
        str_vec.push(buf)
    }
    return str_vec;
}

#[derive(Clone)]
pub struct Wordcut {
    dict: Dict,
}

impl Wordcut {
    pub fn new(dict: Dict) -> Wordcut {
        Wordcut { dict: dict }
    }

    #[allow(dead_code)]
    pub fn segment(&self, text: &str) -> Vec<TextRange> {
        let text: Vec<char> = text.chars().collect();
        let path = build_path(&self.dict, &text);
        return path_to_ranges(&path);
    }

    pub fn segment_into_byte_ranges(&self, text: &str) -> Vec<TextRange> {
        let text: Vec<char> = text.chars().collect();
        let path = build_path(&self.dict, &text);
        return path_to_byte_ranges(&path, &text);
    }

    pub fn segment_into_strings(&self, text: &str) -> Vec<String> {
        let text: Vec<char> = text.chars().collect();
        let path = build_path(&self.dict, &text);
        return path_to_str_vec(&path, &text);
    }

    pub fn put_delimiters(&self, text: &str, delim: &str) -> String {
        self.segment_into_strings(text).join(delim)
    }

    #[allow(dead_code)]
    pub fn build_dag(&self, text: &str) -> Dag {
        build_dag(&self.dict, &text.chars().collect())
    }
}

pub fn load_wordlist(path: &Path) -> io::Result<Vec<String>> {
    let f = File::open(path)?;
    let f = io::BufReader::new(f);
    Ok(f.lines().map(|line| line.unwrap()).collect())
}

pub fn load_dict(path: &Path) -> io::Result<Dict> {
    let wordlist = load_wordlist(path).unwrap();
    let wordlist: Vec<_> = wordlist.iter().map(|w| &w[..]).collect();
    return Ok(create_prefix_tree(&wordlist));
}

#[cfg(test)]
mod tests {
    extern crate serde_json;
    use DagEdge;
    use EdgeType;
    use TextRange;
    use Wordcut;

    #[test]
    fn test_prefix_tree() {
        let prefix_tree = super::create_prefix_tree(&["A"]);
        assert_eq!(
            prefix_tree.seek(&(0, 0, 'A')),
            Some(&(0 as u32, true, Some(true)))
        );
        assert_eq!(prefix_tree.seek(&(0, 0, 'B')), None);
    }

    #[test]
    fn test_segment() {
        let dict = super::create_prefix_tree(&["กา", "กาก"]);
        let wordcut = Wordcut::new(dict);
        let ranges = wordcut.segment("กากกา");
        let expected = vec![TextRange { s: 0, e: 3 }, TextRange { s: 3, e: 5 }];
        assert_eq!(ranges, expected)
    }

    #[test]
    fn test_segment_into_byte_ranges() {
        let dict = super::create_prefix_tree(&["กา", "กาก"]);
        let wordcut = Wordcut::new(dict);
        let ranges = wordcut.segment_into_byte_ranges("กากกา");
        let expected = vec![TextRange { s: 0, e: 9 }, TextRange { s: 9, e: 15 }];
        assert_eq!(ranges, expected)
    }

    #[test]
    fn test_segment_to_strings() {
        let dict = super::create_prefix_tree(&["กา", "กาก"]);
        let wordcut = Wordcut::new(dict);
        let toks = wordcut.segment_into_strings("กากกา");
        let expected = vec![String::from("กาก"), String::from("กา")];
        assert_eq!(toks, expected)
    }

    #[test]
    fn test_put_delimiters() {
        let dict = super::create_prefix_tree(&["กา", "กาก"]);
        let wordcut = Wordcut::new(dict);
        assert_eq!(wordcut.put_delimiters("กากกา", "|"), String::from("กาก|กา"))
    }

    #[test]
    fn test_load_wordlist() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let v = super::load_wordlist(path);
        assert_eq!(v.unwrap(), vec![String::from("กา"), String::from("กาก")])
    }

    #[test]
    fn test_wordcut() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new(dict.unwrap());
        assert_eq!(wordcut.put_delimiters("กากกา", "|"), String::from("กาก|กา"))
    }

    #[test]
    fn test_dag() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path).unwrap();
        let wordcut = Wordcut::new(dict);
        let dag = wordcut.build_dag("กากกา");
        let expected = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
            vec![DagEdge {
                s: 0,
                e: 1,
                etype: EdgeType::Unk,
            }], // 1
            vec![DagEdge {
                s: 0,
                e: 2,
                etype: EdgeType::Dict,
            }], // 2
            vec![DagEdge {
                s: 0,
                e: 3,
                etype: EdgeType::Dict,
            }], // 3
            vec![DagEdge {
                s: 3,
                e: 4,
                etype: EdgeType::Unk,
            }], // 4
            vec![DagEdge {
                s: 3,
                e: 5,
                etype: EdgeType::Dict,
            }], // 5
        ];
        assert_eq!(dag, expected);
    }

    #[test]
    fn test_dag_in_object() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let dag = super::build_dag(&dict.unwrap(), &"กากกา".chars().collect());
        let expected = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
            vec![DagEdge {
                s: 0,
                e: 1,
                etype: EdgeType::Unk,
            }], // 1
            vec![DagEdge {
                s: 0,
                e: 2,
                etype: EdgeType::Dict,
            }], // 2
            vec![DagEdge {
                s: 0,
                e: 3,
                etype: EdgeType::Dict,
            }], // 3
            vec![DagEdge {
                s: 3,
                e: 4,
                etype: EdgeType::Unk,
            }], // 4
            vec![DagEdge {
                s: 3,
                e: 5,
                etype: EdgeType::Dict,
            }], // 5
        ];
        assert_eq!(dag, expected);
    }

    #[test]
    fn test_dag_punc() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let dag = super::build_dag(&dict.unwrap(), &"กา กา".chars().collect());
        let expected = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
            vec![DagEdge {
                s: 0,
                e: 1,
                etype: EdgeType::Unk,
            }], // 1
            vec![DagEdge {
                s: 0,
                e: 2,
                etype: EdgeType::Dict,
            }], // 2
            vec![DagEdge {
                s: 2,
                e: 3,
                etype: EdgeType::Punc,
            }], // 3
            vec![DagEdge {
                s: 3,
                e: 4,
                etype: EdgeType::Unk,
            }], // 4
            vec![DagEdge {
                s: 3,
                e: 5,
                etype: EdgeType::Dict,
            }], // 5
        ];
        assert_eq!(dag, expected);
    }

    #[test]
    fn test_dag_latin() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let dag = super::build_dag(&dict.unwrap(), &"กาAB".chars().collect());
        let expected = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
            vec![DagEdge {
                s: 0,
                e: 1,
                etype: EdgeType::Unk,
            }], // 1
            vec![DagEdge {
                s: 0,
                e: 2,
                etype: EdgeType::Dict,
            }], // 2
            vec![DagEdge {
                s: 2,
                e: 3,
                etype: EdgeType::Unk,
            }], // 3
            vec![DagEdge {
                s: 2,
                e: 4,
                etype: EdgeType::Latin,
            }], // 4
        ];
        assert_eq!(dag, expected);
    }

    #[test]
    fn test_dag_empty() {
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let dag = super::build_dag(&dict.unwrap(), &"".chars().collect());
        let expected = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
        ];
        assert_eq!(dag, expected);
    }

    #[test]
    fn test_dag_to_json() {
        let dag = vec![
            vec![DagEdge {
                s: 0,
                e: 0,
                etype: EdgeType::Init,
            }], // 0
        ];
        let s = serde_json::to_string(&dag).unwrap();
        assert_eq!(s, "[[{\"s\":0,\"e\":0,\"etype\":\"Init\"}]]");
    }
}
