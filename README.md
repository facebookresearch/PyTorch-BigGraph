# PyTorch-BigGraph

[![CircleCI Status](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph.svg?style=svg)](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph) [![Documentation Status](https://readthedocs.org/projects/torchbiggraph/badge/?version=latest)](https://torchbiggraph.readthedocs.io/en/latest/?badge=latest)

PyTorch-BigGraph (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.

PBG was introduced in the [PyTorch-BigGraph: A Large-Scale Graph Embedding Framework](http://arxiv.org/FIXME) paper, presented at the [SysML conference](https://www.sysml.cc/) in 2019.

PBG trains on an input graph by ingesting its list of edges, each identified by its source and target entities and, possibly, a relation type. It outputs a feature vector (embedding) for each entity, trying to place adjacent entities close to each other in the vector space, while pushing unconnected entities apart. Therefore, entities that have a similar distribution of neighbors will end up being nearby.

It's possible to configure each relation type to calculate this "proximity score" in a different way, with the parameters (if any) learned during training. This allows the same underlying entity embeddings to be shared among multiple relation types.

The generality and extensibility of its model allows PBG to train a number of models from the knowledge graph embedding literature, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf), [DistMult](https://arxiv.org/abs/1412.6575) and [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

PBG is designed with scale in mind, and achieves it through:
- *graph partitioning*, so that the model doesn't have to be fully loaded into memory
- *multi-threaded computation* on each machine
- *distributed execution* across multiple machines (optional), all simultaneously operating on disjoint parts of the graph
- *batched negative sampling*, allowing for processing >1 million edges/sec/machine with 100 negatives per edge

PBG is *not* for model exploration with exotic models on small graphs, e.g. graph convnets, deep networks, etc.

## Requirements

PBG is written in Python (version 3.6 or later) and relies on [PyTorch](https://pytorch.org/) (at least version 1.0), [NumPy](http://www.numpy.org/), [HDF5](https://www.hdfgroup.org/solutions/hdf5/) / [h5py](https://www.h5py.org/) and a few other libraries.

All computations are performed on the CPU, therefore a large number of cores is advisable. No GPU is necessary.

When running on multiple machines, they need to be able to communicate to each other at high bandwidth (10 Gbps or higher recommended) and have access to a shared filesystem (for checkpointing). PBG uses [torch.distributed](https://pytorch.org/docs/stable/distributed.html), which uses the Gloo package which runs on top of TCP or MPI.

## Installation

To install the latest version of PBG run:
```bash
pip install torchbiggraph
```

As an alternative, one can instead install the *development* version from the repository. This may have newer features but could be more unstable. To do so, clone the repository (or download it as an archive) and, from inside the top-level directory, run:
```bash
pip install -r requirements.txt
./setup.py install
```

## Getting started

One can easily reproduce the results of [the paper](http://arxiv.org/FIXME) by running the sample scripts located in the [examples](examples). By running:
```bash
examples/fb15k.py
```
the script will download the Freebase 15k knowledge base dataset, put it into the right format, train on it using the ComplEx model and finally perform an evaluation of the learned embeddings that calculates the MRR and other metrics that should match the paper.

Another script, `examples/livejournal.py`, does the same for the LiveJournal interaction graph dataset.

## How to train on one's own data

The main entry point to PBG is the `torchbiggraph_train` command. It takes one command line argument: a path to a configuration file, which is a Python module implementing a function that returns the configuration as a dictionary. See [the documentation](https://torchbiggraph.readthedocs.io/en/latest/configuration_file.html) on how to prepare such a file, or [some examples](examples/configs).

PBG's input and output is file-based. The configuration specifies the directories in which PBG looks for information about the graph to train on and to which it writes back the embeddings it produces. These files are in various formats (text, JSON, HDF5). The complete specification can be found in [the documentation](https://torchbiggraph.readthedocs.io/en/latest/input_output.html).

Some converters are provided that can prepare the input data for PBG based on an edge list in TSV format (tab-separated values), and export the embeddings back to TSV. This is the format many public datasets are in. These tools are mainly provided for demonstration, as they don't scale beyond moderately-sized graphs.

For larger graphs, one will have to write their own custom partitioning and conversion utilities.

### Walkthrough of the FB15k demo

Let's trace, step by step, what the `fb15k.py` script does.

First, it [retrieves the dataset](https://dl.fbaipublicfiles.com/starspace/fb15k.tgz) and unpacks it, to obtain a directory with three edge sets as TSV files, for training, validation and testing. Each line of these files contains information about one edge. Using tabs as separators, the lines are divided into several columns which contain the identifiers of the source entities, the target entities and of the relation types.

Then, the script goes over all the edges of all these files. For each of them, it looks up the relation type in the configuration and this way finds out the types of the left- and right-hand side entities. It does so in order to record, for each entity type, all the identifiers of all the entities of that type. It then proceeds to randomly split the entities of each type into the appropriate number of partitions and, inside each partition, to randomly order them so as to assign an offset to each of them.

Having now processed the entities, the script goes over the edge list files once more to process the edges. For each of them it's now able to determine its "bucket" (i.e., the pair given by the partitions of its two entities). Each bucket, with each edge expressed as the offsets of its two entities within their partitions plus its relation type, is stored to disk as a separate HDF5 file.

Note: this conversion process can also be performed by a standalone command, `torchbiggraph_import_from_tsv`.

The data is now ready to be consumed by the trainer, so the script invokes the training command passing it the [sample configuration](examples/configs/fb15k_config.py), customized to include the training edge list. This step can also be manually replicated using the `torchbiggraph_train` command.

Once the specified number of training iterations have been completed, the script finishes by evaluating the obtained embeddings on the testing edge list (which was held out during training) in order to obtain some standard metrics (e.g., mean reciprocal rank). For this to be done correctly it first updates the config to use all other entities as negatives when scoring an edge, rather than only sampling a few as was done during training. This step can be performed by hand with the `torchbiggraph_eval ` command, although with a difference: the FB15k script uses a custom *filtered* MRR evaluation, whereas the standard command uses an unfiltered metric (for performance reasons).

One final step that the script does not do automatically is to convert the embeddings to a human-readable TSV format. This can be done by invoking
```bash
torchbiggraph_export_to_tsv --dict=data/FB15k/dictionary.json --checkpoint=model/fb15k/ --out=output.tsv
```
which will write all entity embeddings and relation parameters to the `output.tsv` file.

## License

PyTorch-BigGraph is BSD licensed, as found in the LICENSE file.
