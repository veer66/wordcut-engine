# wordcut-engine

Word segmentation library in Rust

## Algorithm

wordcut-engine has three steps:

1. Identifying clusters, which are substrings that must not be split
2. Identifying edges of split directed acyclic graph (split-DAG); The program does not add edges that break any cluster to the graph.
3. Tokenizing a string by finding the shortest path in the split-DAG

## Identifying clusters

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
