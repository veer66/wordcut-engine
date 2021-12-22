#[macro_use]
extern crate lazy_static;
extern crate prefixtree;
#[macro_use]
extern crate serde_derive;

use self::prefixtree::{prefix_tree_from_str, PrefixTree};
use regex_automata::dense::DenseDFA;
use regex_automata::Regex;
use regex_automata::DFA;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::iter::Peekable;
use std::path::Path;
use thiserror::Error;

pub type Dict = PrefixTree<char, bool>;

type ClusterRulesMatcher = DenseDFA<Vec<usize>, usize>;
type SplitRulesMatcher = Regex;

lazy_static! {
    static ref DEFAULT_THAI_SPLIT_RE: Regex =
        Regex::new("[\r\t\n ]+|[A-Za-z]+|[0-9]+|[๐-๙]+|“").unwrap();
}

#[derive(Error, Debug)]
pub enum WordcutError {
    #[error("Cannot open cluster rules at `{0}`")]
    CannotOpenClusterRulesAt(String),
    #[error("Cannot read a cluster rule")]
    CannotReadClusterRule,
    #[error("Cannot compile cluster rules `{0}`")]
    CannotCompileClusterRules(String),
    #[error("Cannot open split rules at `{0}`")]
    CannotOpenSplitRulesAt(String),
    #[error("Cannot compile split rules `{0}`")]
    CannotCompileSplitRules(String),
}

pub fn create_prefix_tree(words: &[&str]) -> PrefixTree<char, bool> {
    let words_payloads: Vec<(&str, bool)> = words.iter().map(|&word| (word, true)).collect();
    prefix_tree_from_str(&words_payloads[..])
}

#[derive(Clone, PartialEq, Eq, Copy, Debug, Serialize, Deserialize)]
pub enum EdgeType {
    Init,
    Dict,
    Unk,
    Pat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
	if self.etype == EdgeType::Pat && o.etype == EdgeType::Unk {
	    return true;
	}
	
	if self.etype == EdgeType::Unk && o.etype == EdgeType::Pat {
	    return false;
	}
	
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

pub trait EdgeBuilder {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge>;
}

#[derive(Debug)]
pub struct EdgeBuildingContext<'a> {
    pub text: &'a [char],
    pub i: usize,
    pub ch: char,
    pub left_boundary: usize,
    pub best_edge: Option<Edge>,
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

pub struct RuleBasedEdgeBuilder {
    range_peekable: Peekable<std::vec::IntoIter<TextRange>>,
}

impl RuleBasedEdgeBuilder {
    pub fn new(byte_to_char_idx_map: &[usize], text: &str, re: &Regex) -> Self {
        let mut ranges = vec![];
        for m in re.find_iter(text.as_bytes()) {
            let (ms, me) = m;
            let s = byte_to_char_idx_map[ms];
            let e = byte_to_char_idx_map[me];
            ranges.push(TextRange { s, e });
        }
        RuleBasedEdgeBuilder {
            range_peekable: ranges.into_iter().peekable(),
        }
    }
}

impl EdgeBuilder for RuleBasedEdgeBuilder {
    fn build(&mut self, context: &EdgeBuildingContext, path: &[Edge]) -> Option<Edge> {
        loop {
            if let Some(r) = self.range_peekable.peek() {
                if context.i >= r.e {
                    self.range_peekable.next();
                } else {
                    break;
                }
            } else {
                return None;
            }
        }
        if let Some(r) = self.range_peekable.peek() {
            if r.e != context.i + 1 {
                return None;
            }
            let source = path[r.s];
            return Some(Edge {
                etype: EdgeType::Pat,
                p: r.s,
                w: source.w + 1,
                unk: source.unk,
            });
        } else {
            return None;
        }
    }
}

#[inline]
fn does_not_break_cluster(s: usize, e: usize, text_len: usize, clusters: &[usize]) -> bool {
    (s == 0 || clusters[s] == 0 || clusters[s] != clusters[s - 1])
        && (e == text_len || clusters[e - 1] == 0 || clusters[e] != clusters[e - 1])
}

#[inline]
fn should_skip_edge(edge: &Option<Edge>, i: usize, text_len: usize, clusters: &[usize]) -> bool {
    let mut skip_edge = false;
    if let Some(edge) = edge {
        let s = edge.p;
        let e = i + 1;
        skip_edge = !edge.is_unk() && !does_not_break_cluster(s, e, text_len, clusters);
    }
    return skip_edge;
}

fn build_path_with_clusters(
    mut builders: Vec<&mut dyn EdgeBuilder>,
    clusters: &[usize],
    text: &[char],
) -> Vec<Edge> {
    let mut path = vec![];
    path.push(Edge {
        w: 0,
        unk: 0,
        p: 0,
        etype: EdgeType::Init,
    });

    let mut context = EdgeBuildingContext {
        text,
        i: 0,
        ch: '\0',
        left_boundary: 0,
        best_edge: None,
    };

    let text_len = text.len();
    for i in 0..text_len {
        context.ch = text[i];
        context.i = i;
        context.best_edge = None;
        for builder in &mut builders {
            let edge = builder.build(&context, &path);
            if !should_skip_edge(&edge, i, text_len, clusters)
                && Edge::better(&edge, &context.best_edge)
            {
                context.best_edge = edge
            }
        }
        path.push(context.best_edge.unwrap());
        if !context.best_edge.unwrap().is_unk() {
            context.left_boundary = i + 1;
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

pub fn build_dag(dict: &Dict, text: &Vec<char>) -> Dag {
    let mut builders: Vec<Box<dyn DagEdgeBuilder>> = vec![Box::new(DictEdgeBuilder::new(dict))];

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
        ranges.push(TextRange { s, e });
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

pub struct Wordcut {
    dict: Dict,
    cluster_re: Option<ClusterRulesMatcher>,
    split_re: SplitRulesMatcher,
}

impl Wordcut {
    pub fn new(dict: Dict) -> Wordcut {
        Wordcut {
            dict,
            cluster_re: None,
            split_re: DEFAULT_THAI_SPLIT_RE.clone(),
        }
    }

    pub fn new_with_cluster_re(dict: Dict, cluster_re: ClusterRulesMatcher) -> Wordcut {
        Wordcut {
            dict,
            cluster_re: Some(cluster_re),
            split_re: DEFAULT_THAI_SPLIT_RE.clone(),
        }
    }

    pub fn new_with_cluster_re_and_split_re(
        dict: Dict,
        cluster_re: ClusterRulesMatcher,
        split_re: SplitRulesMatcher,
    ) -> Wordcut {
        Wordcut {
            dict,
            cluster_re: Some(cluster_re),
            split_re,
        }
    }

    #[inline]
    pub fn build_path(&self, text: &str, text_chars: &[char]) -> Vec<Edge> {
        let byte_to_char_idx_map = create_byte_to_char_idx_map(text);
        let mut dict_edge_builder = DictEdgeBuilder::new(&self.dict);
        let mut unk_edge_builder = UnkEdgeBuilder::new();
        let mut rule_based_edge_builder =
            RuleBasedEdgeBuilder::new(&byte_to_char_idx_map, text, &self.split_re);
        let builders: Vec<&mut dyn EdgeBuilder> = vec![
            &mut dict_edge_builder,
            &mut unk_edge_builder,
            &mut rule_based_edge_builder,
        ];

        let clusters = if let Some(cluster_re) = &self.cluster_re {
            find_clusters(text, &byte_to_char_idx_map, cluster_re, text_chars.len())
        } else {
            let mut clusters = vec![];
            clusters.resize(text_chars.len() + 1, 0);
            clusters
        };
        build_path_with_clusters(builders, &clusters, text_chars)
    }

    #[allow(dead_code)]
    pub fn segment(&self, text: &str) -> Vec<TextRange> {
        let text_chars: Vec<char> = text.chars().collect();
        let path = self.build_path(text, &text_chars);
        path_to_ranges(&path)
    }

    pub fn segment_into_byte_ranges(&self, text: &str) -> Vec<TextRange> {
        let text_chars: Vec<char> = text.chars().collect();
        let path = self.build_path(text, &text_chars);
        return path_to_byte_ranges(&path, &text_chars);
    }

    pub fn segment_into_strings(&self, text: &str) -> Vec<String> {
        let text_chars: Vec<char> = text.chars().collect();
        let path = self.build_path(text, &text_chars);
        return path_to_str_vec(&path, &text_chars);
    }

    pub fn put_delimiters(&self, text: &str, delim: &str) -> String {
        self.segment_into_strings(text).join(delim)
    }

    #[allow(dead_code)]
    pub fn build_dag(&self, text: &str) -> Dag {
        build_dag(&self.dict, &text.chars().collect())
    }
}

pub fn create_byte_to_char_idx_map(text: &str) -> Vec<usize> {
    let mut byte_to_char_map = vec![];
    let mut i = 0;
    for b in text.as_bytes() {
        if (*b as i8) >= -0x40 {
            byte_to_char_map.push(i);
            i += 1;
        } else {
            byte_to_char_map.push(0);
        }
    }
    byte_to_char_map.push(i);
    return byte_to_char_map;
}

#[derive(Debug)]
pub struct ClusterPointer {
    state_id: usize,
    p: usize,
}

#[derive(Debug)]
pub struct ClusterEdge {
    acc_pat_len: usize,
    unk_cnt: usize,
    p: usize,
    is_unk: bool,
}

pub fn find_cluster_path(dfa: &ClusterRulesMatcher, text: &str) -> Vec<ClusterEdge> {
    let mut pointers = vec![];
    let mut ch_i = 0;
    let mut path = vec![];
    let mut left_boundary = 0;
    path.push(ClusterEdge {
        p: 0,
        acc_pat_len: 0,
        unk_cnt: 0,
        is_unk: false,
    });
    for ch in text.as_bytes() {
        let mut best_edge: Option<ClusterEdge> = None;
        pointers.push(ClusterPointer {
            state_id: dfa.start_state(),
            p: ch_i,
        });
        let mut j = 0;
        for i in 0..pointers.len() {
            let next_id = dfa.next_state(pointers[i].state_id, *ch);
            if !dfa.is_dead_state(next_id) {
                pointers[j] = ClusterPointer {
                    state_id: next_id,
                    p: pointers[i].p,
                };
                j += 1;
                if dfa.is_match_state(next_id) {
                    let source = &path[pointers[i].p];
                    let edge = ClusterEdge {
                        p: pointers[i].p,
                        acc_pat_len: source.acc_pat_len + (ch_i - pointers[i].p + 1),
                        unk_cnt: source.unk_cnt,
                        is_unk: false,
                    };
                    if match &best_edge {
                        Some(b_edge) => {
                            b_edge.unk_cnt > edge.unk_cnt
                                || (b_edge.unk_cnt == edge.unk_cnt
                                    && b_edge.acc_pat_len < edge.acc_pat_len)
                        }
                        None => true,
                    } {
                        best_edge = Some(edge);
                    }
                }
            }
        }
        pointers.truncate(j);
        if best_edge.is_none() {
            let source = &path[left_boundary];
            best_edge = Some(ClusterEdge {
                p: left_boundary,
                acc_pat_len: source.acc_pat_len,
                unk_cnt: source.unk_cnt + (ch_i - left_boundary + 1),
                is_unk: true,
            });
        }
        let best_edge = best_edge.unwrap();
        if !best_edge.is_unk {
            left_boundary = ch_i + 1;
        }
        path.push(best_edge);
        ch_i += 1;
    }
    path
}

pub fn find_clusters(
    text: &str,
    byte_to_char_idx_map: &[usize],
    dfa: &ClusterRulesMatcher,
    len: usize,
) -> Vec<usize> {
    let mut clusters = vec![];
    clusters.resize(len, 0);
    let mut id = 1;
    let path = find_cluster_path(dfa, text);
    let mut me = path.len() - 1;
    while me > 0 {
        let edge = &path[me];
        let ms = edge.p;
        let s = byte_to_char_idx_map[ms];
        let e = byte_to_char_idx_map[me];
        if !edge.is_unk {
            for i in s..e {
                clusters[i] = id;
            }
            id += 1;
        }
        me = ms;
    }
    clusters
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

pub fn load_cluster_rules(path: &Path) -> Result<ClusterRulesMatcher, WordcutError> {
    let f = File::open(path)
        .map_err(|_| WordcutError::CannotOpenClusterRulesAt(path.to_string_lossy().to_string()))?;
    let f = io::BufReader::new(f);
    let mut rules = vec![];
    for line in f.lines() {
        let line = line.map_err(|_| WordcutError::CannotReadClusterRule)?;
        rules.push(format!("({})", line.trim()));
    }
    let rules = rules.join("|");
    let mut builder = regex_automata::dense::Builder::new();
    builder.anchored(true);
    builder.unicode(true);
    Ok(builder
        .build(&rules)
        .map_err(|_| WordcutError::CannotCompileClusterRules(rules))?)
}

pub fn load_split_rules(path: &Path) -> Result<SplitRulesMatcher, WordcutError> {
    let f = File::open(path)
        .map_err(|_| WordcutError::CannotOpenSplitRulesAt(path.to_string_lossy().to_string()))?;
    let f = io::BufReader::new(f);
    let mut rules = vec![];
    for line in f.lines() {
        let line = line.map_err(|_| WordcutError::CannotReadClusterRule)?;
        rules.push(format!("({})", line.trim()));
    }
    let rules = rules.join("|");
    Ok(Regex::new(&rules).map_err(|_| WordcutError::CannotCompileSplitRules(rules))?)
}

#[cfg(test)]
mod tests {
    extern crate serde_json;
    use super::*;

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
    fn test_wordcut_with_latin() {
	let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new(dict.unwrap());
	assert_eq!(wordcut.put_delimiters("ฑฑACญญ", "|"), String::from("ฑฑ|AC|ญญ"))
    }

    #[test]
    fn test_wordcut_with_two_spaces() {
	let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new(dict.unwrap());
	assert_eq!(wordcut.put_delimiters("กา  มา", "|"), String::from("กา|  |มา"))
    }

    #[test]
    fn test_wordcut_with_two_spaces_unk() {
	let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new(dict.unwrap());
	assert_eq!(wordcut.put_delimiters("แแ  ยย", "|"), String::from("แแ|  |ยย"))
    }

    #[test]
    fn test_wordcut_with_unicode_quote() {
	let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/thai2words.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new(dict.unwrap());
	assert_eq!(wordcut.put_delimiters("“ฆกากา”", "|"), String::from("“|ฆ|กา|กา|”"))
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

    #[test]
    fn test_find_clusters() {
        let text = "กาแกกก์A";
        let path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_cluster_rules.txt"
        ));
        let cluster_re = super::load_cluster_rules(&path).unwrap();
        let byte_to_char_idx_map = create_byte_to_char_idx_map(text);
        let clusters = find_clusters(
            text,
            &byte_to_char_idx_map,
            &cluster_re,
            text.chars().count(),
        );
        assert_eq!(clusters, vec![2, 2, 1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn test_wordcut_with_clusters() {
        let text = "แมวแฐแกกก์มา";
        let cluster_path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_cluster_rules.txt"
        ));
        let cluster_re = super::load_cluster_rules(&cluster_path).unwrap();
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/words_th.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new_with_cluster_re(dict.unwrap(), cluster_re);
        assert_eq!(
            wordcut.put_delimiters(text, "|||"),
            String::from("แมว|||แฐแกกก์|||มา")
        );
    }

    #[test]
    fn test_wordcut_with_clusters2() {
        let text = "มีรีเควสต์อะไร";
        let cluster_path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_cluster_rules.txt"
        ));
        let cluster_re = super::load_cluster_rules(&cluster_path).unwrap();
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/words_th.txt"));
        let dict = super::load_dict(&path);
        let wordcut = Wordcut::new_with_cluster_re(dict.unwrap(), cluster_re);
        assert_eq!(
            wordcut.put_delimiters(text, "|||"),
            String::from("มี|||รี|||เค|||วสต์|||อะไร")
        );
    }

    #[test]
    fn test_rule_based_edge_builder() {
        let text = "  ABก";
        let text_chars: Vec<char> = text.chars().collect();
        let byte_to_char_idx_map = create_byte_to_char_idx_map(text);
        let mut builder =
            RuleBasedEdgeBuilder::new(&byte_to_char_idx_map, text, &DEFAULT_THAI_SPLIT_RE);
        let mut path = vec![];
        path.push(Edge {
            w: 10,
            unk: 20,
            p: 0,
            etype: EdgeType::Init,
        });
        let edge = builder.build(
            &EdgeBuildingContext {
                text: &text_chars,
                i: 0,
                ch: '\0',
                left_boundary: 0,
                best_edge: None,
            },
            &path,
        );
        assert!(edge.is_none());
        path.push(Edge {
            w: 20,
            unk: 30,
            p: 0,
            etype: EdgeType::Unk,
        });

        let edge = builder.build(
            &EdgeBuildingContext {
                text: &text_chars,
                i: 1,
                ch: '\0',
                left_boundary: 0,
                best_edge: None,
            },
            &path,
        );
        assert!(edge.is_some());
        path.push(Edge {
            w: 30,
            unk: 40,
            p: 0,
            etype: EdgeType::Pat,
        });

        let edge = builder.build(
            &EdgeBuildingContext {
                text: &text_chars,
                i: 2,
                ch: '\0',
                left_boundary: 0,
                best_edge: None,
            },
            &path,
        );
        assert!(edge.is_none());
        path.push(Edge {
            w: 50,
            unk: 60,
            p: 0,
            etype: EdgeType::Unk,
        });

        let edge = builder.build(
            &EdgeBuildingContext {
                text: &text_chars,
                i: 3,
                ch: '\0',
                left_boundary: 0,
                best_edge: None,
            },
            &path,
        );
        assert!(edge.is_some());
        let edge = edge.unwrap();
        assert_eq!(
            edge,
            Edge {
                w: 31,
                unk: 40,
                p: 2,
                etype: EdgeType::Pat
            }
        );
    }

    #[test]
    fn test_wordcut_with_split_rules() {
        let text = "AB   X(A)/12";
        let cluster_path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_cluster_rules.txt"
        ));
        let split_path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_split_rules.txt"
        ));

        let cluster_re = super::load_cluster_rules(&cluster_path).unwrap();
        let path = super::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data/words_th.txt"));
        let dict = super::load_dict(&path);
        let split_re = load_split_rules(&split_path).unwrap();
        let wordcut =
            Wordcut::new_with_cluster_re_and_split_re(dict.unwrap(), cluster_re, split_re);
        assert_eq!(
            wordcut.put_delimiters(text, "|||"),
            String::from("AB|||   |||X|||(|||A|||)|||/|||12")
        );
    }

    #[test]
    fn test_find_clusters_path() {
        let path = super::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/thai_cluster_rules.txt"
        ));
        let cluster_re = super::load_cluster_rules(&path).unwrap();
        let cluster_path = find_cluster_path(&cluster_re, "เกียำ");
        assert_eq!(cluster_path.len(), 16);
        assert_eq!(cluster_path[15].p, 9);
    }
}
