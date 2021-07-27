# wordcut-engine

Word segmentation library in Rust

## Example

```Rust
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
```

## Algorithm

wordcut-engine has three steps:

1. Identifying clusters, which are substrings that must not be split
2. Identifying edges of split directed acyclic graph (split-DAG); The program does not add edges that break any cluster to the graph.
3. Tokenizing a string by finding the shortest path in the split-DAG

## Identifying clusters

Identifying clusters identify which substrings that must _not_ be split.

1. Wrapping regular expressions with parentheses

For example,

```
[ก-ฮ]็
[ก-ฮ][่-๋]
[ก-ฮ][่-๋][ะาำ]
```

The above rules are wrapped with parentheses as shown below:

```
([ก-ฮ]็)
([ก-ฮ][่-๋])
([ก-ฮ][่-๋][ะาำ])
```

2. Joining regular expressions with vertical bars (|) 

for example, 

```
([ก-ฮ]็)|([ก-ฮ][่-๋])|([ก-ฮ][่-๋][ะาำ])
```

3. Building a DFA from the joined regular expression using [regex-automata](https://github.com/BurntSushi/regex-automata)

4. Creating a directed acyclic graph (DAG) by adding edges using the DFA

5. Identifying clusters following a shortest path of a DAG from step above

Note: wordcut-engine does not allow a context sensitive rule, since it hurts the performance too much. Moreover, instead of longest matching, we use a DAG, and its shortest path to contraint cluster boundary by another cluster, therefore [newmm](https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/tokenize/newmm.py)-style context sensitive rules are not required.


## Identifying split-DAG edges

In contrary to identifying clusters, identifying split-DAG edges identify what must be split. Split-DAG edge makers, wordcut-engine has three types of split-DAG edge maker, that are:

1. Dictionary-based maker
2. Rule-based maker
3. Default maker (Unk edge builder)

The dictionary-based maker traverses a prefix tree, which is particularly a trie in wordcut-engine and create an edge that matched word in the prefix tree. Rule-based maker uses [regex-automata](https://github.com/BurntSushi/regex-automata)'s Regex matcher built from split rules to find longest matched substrings, and add corresponding edges to the graph. wordcut-engine removes ddges that break clusters. The example of split rules are shown below:

```
[\r\t\n ]+
[A-Za-z]+
[0-9]+
[๐-๙]+
[\(\)"'`\[\]{}\\/]
```

If there is no edge for each of character indice yet, a default maker create a edge that connected a last known boundary.
