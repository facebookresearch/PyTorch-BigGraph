This is intended as a simple end-to-end example of how to get your data into
the format that PyTorch BigGraph expects using SQL. It's implemented in SQLite
for portability, but similar techniques scale to billions of edges using cloud
databases such as BigQuery or SnowFlake. This pipeline can be split into three
different components:

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
can be understood as first partitioning the entities, then generating a mapping
between the graph-ids and ordinal ids per-type that PBG will expect, and finally
writing out all the files required to train, including the config file. By
keeping track of the vertex types, we're able to specifically verify our mappings
in a fully self consistent fashion.

Once the data has been prepared and generated, we're ready to embed the graph. We
do this by passing the generated config to `torchbiggraph_train` in the following
way:

```
torchbiggraph_train \
  path/to/generated/config.py
```

The `data_prep.py` script will also compute the approximate amount of shared memory
that will be needed for training. If the training demands are more than the
available shared memory, you'll need to regenerate your data with more partitions
than what you currently have. If you're seeing either a bus error or a OOM kill
message in the kernel ring buffer but your machine has enough ram, you'll want to
verify that `/dev/shm` is large enough to accomodate your embedding table.