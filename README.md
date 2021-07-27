# wordcut-engine

Word segmentation library in Rust

## Algorithm

wordcut-engine has three steps:

1. Identifying clusters, which are substrings that must not be split
2. Identifying edges of split directed acyclic graph (split-DAG); The program does not add edges that break any cluster to the graph.
3. Tokenizing a string by finding the shortest path in the split-DAG
