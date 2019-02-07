# PyTorch-BigGraph

PyTorch-BigGraph (PBG) is a system for producing graph embeddings for large, multi-relation graphs, particularly large web interaction graphs with up to billions of entities and trillions of edges.

PBG is described in [PyTorch-BigGraph: A Large-Scale GraphEmbedding Framework](http://arxiv.org/FIXME) (SysML '19).

PBG takes as input a list of graph edges `(source, destination[, relation])` and produces a feature vector (embedding) for each entity. The entity embeddings are constructed so that entities that are connected by an edge are nearby in the vector space, while unconnected entities are farther apart. As a consequence, 'similar' entities (entities that have a similar distribution of neighbors) will have similar embeddings.

TODO something about multi-relation

The graph embedding models provided by Pytorch-BigGraph are similar to those used for knowledge graph embeddings, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf) anbd [DistMult](https://arxiv.org/abs/1412.6575). PBG provides similar models, loss functions, and negative sampling strategies as used in these works.

PBG is designed to scale to huge graphs via 
- Graph partitioning, which allows PBG to train models that are too large to fit in memory
- Optional distributed execution across multiple machines
- Multi-threaded execution on each machine
- Batched negative sampling, allowing for processing >1 million edges/sec/machine with 100 negatives per edge

PBG is *not* for model exploration with exotic models on small graphs, e.g. graph convnets, deep networks, etc.

## Requirements

- Python 3.6+
- Pytorch 1.0+
- No GPU required
- We recommend a machine with a high number of cores for fast training on large graphs

**Distributed mode only:** 
- Multiple machines with a shared filesystem (for checkpointing).
- at least 10Gbps bandwidth between machines recommended
- PBG uses [torch.distributed](https://pytorch.org/docs/stable/distributed.html), which uses the Gloo package which runs on top of TCP or MPI.

## Getting Started

1. Make sure you have Python 3.6+ installed. We recommend [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Install [Pytorch](https://pytorch.org/get-started/locally/).
3. `git clone https://github.com/facebookresearch/pytorch-biggraph`
4. `cd pytorch-biggraph`
5. `pip install -r requirements.txt`
6. `python setup.py install`

### Example Datasets

PBG contains scripts to train embeddings for some example datasets from the [PBG paper](http://arxiv.org/FIXME) end-to-end. That includes downloading the datasets, converting them to the partitioned PBG format, training the embedding, and evaluating the quality of the embeddings.

```
# train an embedding for the fb15k knowledge graph dataset
# TODO info about dynrel mode
python examples/fb15k.py

# train an embedding for livejournal's interaction graph
python examples/livejournal.py
```

### Your Own Dataset

0. Get your dataset into a standard input format. This consists of a text file where each line contains space-separated fields for a src-entity identifier and dest-entity identifier, and (optionally) a field for the relation identifier. You can look at the [freebase dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52312) for an example.

NOTE: If your edgelist is too large or not amenable to a text input format, you can consider modifying the edgelist converter code to support your input format.

1. Construct a config file for your dataset. You can start by looking at one of the [example configs](FIXME) or looking at the [configuration documentation](FIXME).

2. Convert your data into PBG's input format.
```
$ torchbiggraph_data_processor my_config.py ... # FIXME
```

3. Train a model
```
$ torchbiggraph_train my_config.py [-p param=override_value]
```

4. *Optional*: Evaluate the model on a held-out test set. Note that during training, PBG will intermittently evaluate your model on a held-out fraction of the data, so this is not necessary unless you have an explicit test set you are interested in evaluating on.
```
$ torchbiggraph_eval my_config.py
```

5. *Optional*: Convert the learned embeddings to csv format:
```
$ torchbiggraph_embeddings_to_csv my_config.py --out my_embeddings_tsv
```

## License

torchbiggraph is BSD licensed, as found in the LICENSE file.
