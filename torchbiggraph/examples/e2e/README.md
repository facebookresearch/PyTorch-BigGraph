This is intended as a simple end-to-end example of how to get your data into
the format that PyTorch BigGraph expects using SQL. It's implemented in SQLite
for portability, but similar techniques scale to billions of edges using cloud
databases such as BigQuery. This pipeline can be split into three different
components:

1. Data preparation
2. Data verification/checking
3. Training

To run the pipeline, you'll first need to download the edges.csv file,
available HERE (TODO: INSERT LINK). This graph was constructed by
taking the [ogbl-citation2](https://github.com/snap-stanford/ogb) graph, and
adding edges for both paper-citations and years-published. While this graph
might not make a huge amount of sense, it's intended to largely fulfill a
pedagogical purpose.

In the data preparation stage, we first load the graph
into a SQLite database and then we transform and partition it. The transformation
can be understood as first generating a mapping between the graph-ids and
ordinal ids per-type that PBG will expect.