# PyTorch-BigGraph

[![CircleCI Status](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph.svg?style=svg)](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph) [![Documentation Status](https://readthedocs.org/projects/torchbiggraph/badge/?version=latest)](https://torchbiggraph.readthedocs.io/en/latest/?badge=latest)

PyTorch-BigGraph (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.

PBG was introduced in the [PyTorch-BigGraph: A Large-scale Graph Embedding Framework](https://www.sysml.cc/doc/2019/71.pdf) paper, presented at the [SysML conference](https://www.sysml.cc/) in 2019.

PBG trains on an input graph by ingesting its list of edges, each identified by its source and target entities and, possibly, a relation type. It outputs a feature vector (embedding) for each entity, trying to place adjacent entities close to each other in the vector space, while pushing unconnected entities apart. Therefore, entities that have a similar distribution of neighbors will end up being nearby.

It is possible to configure each relation type to calculate this "proximity score" in a different way, with the parameters (if any) learned during training. This allows the same underlying entity embeddings to be shared among multiple relation types.

The generality and extensibility of its model allows PBG to train a number of models from the knowledge graph embedding literature, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf), [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf), [DistMult](https://arxiv.org/abs/1412.6575) and [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

PBG is designed with scale in mind, and achieves it through:
- *graph partitioning*, so that the model does not have to be fully loaded into memory
- *multi-threaded computation* on each machine
- *distributed execution* across multiple machines (optional), all simultaneously operating on disjoint parts of the graph
- *batched negative sampling*, allowing for processing >1 million edges/sec/machine with 100 negatives per edge

PBG is *not* for model exploration with exotic models on small graphs, e.g. graph convnets, deep networks, etc.

## Requirements

PBG is written in Python (version 3.6 or later) and relies on [PyTorch](https://pytorch.org/) (at least version 1.0) and a few other libraries.

All computations are performed on the CPU, therefore a large number of cores is advisable. No GPU is necessary.

When running on multiple machines, they need to be able to communicate to each other at high bandwidth (10 Gbps or higher recommended) and have access to a shared filesystem (for checkpointing). PBG uses [torch.distributed](https://pytorch.org/docs/stable/distributed.html), which uses the Gloo package which runs on top of TCP or MPI.

## Installation

To install the latest version of PBG run:
```bash
pip install torchbiggraph
```

As an alternative, one can instead install the *development* version from the repository. This may have newer features but could be more unstable. To do so, clone the repository (or download it as an archive) and, inside the top-level directory, run:
```bash
pip install -r requirements.txt
./setup.py install
```

## Getting started

The results of [the paper](https://www.sysml.cc/doc/2019/71.pdf) can easily be reproduced by running the sample scripts located in the [examples](examples) directory, for instance:
```bash
examples/fb15k.py
```
This will download the Freebase 15k knowledge base dataset, put it into the right format, train on it using the ComplEx model and finally perform an evaluation of the learned embeddings that calculates the MRR and other metrics that should match the paper. Another script, `examples/livejournal.py`, does the same for the LiveJournal interaction graph dataset.

To learn how to use PBG, let us walk through what the FB15k script does.

### Downloading the data

First, it [retrieves the dataset](https://dl.fbaipublicfiles.com/starspace/fb15k.tgz) and unpacks it, obtaining a directory with three edge sets as TSV files, for training, validation and testing.
```bash
wget https://dl.fbaipublicfiles.com/starspace/fb15k.tgz -P data
tar xf data/fb15k.tgz -C data
```

Each line of these files contains information about one edge. Using tabs as separators, the lines are divided into columns which contain the identifiers of the source entities, the relation types and the target entities. For example:
```
/m/027rn	/location/country/form_of_government	/m/06cx9
/m/017dcd	/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor	/m/06v8s0
/m/07s9rl0	/media_common/netflix_genre/titles	/m/0170z3
/m/01sl1q	/award/award_winner/awards_won./award/award_honor/award_winner	/m/044mz_
/m/0cnk2q	/soccer/football_team/current_roster./sports/sports_team_roster/position	/m/02nzb8
```

### Preparing the data

Then, the script converts the edge lists to PBG's input format. This amounts to assigning a numerical identifier to all entities and relation types, shuffling and partitioning the entities and edges and writing all down in the right format.

Luckily, there is a command that does all of this:
```bash
torchbiggraph_import_from_tsv --lhs-col=0 --rel-col=1 --rhs-col=2 examples/configs/fb15k_config.py data/FB15k/freebase_mtr100_mte100-*.txt
```
The outputs will be stored next to the inputs in the `data/FB15k` directory.

This simple utility is only suitable for small graphs that fit entirely in memory. To handle larger data one will have to implement their own custom preprocessor.

### Training

The `torchbiggraph_train` command is used to launch training. The training parameters are tucked away in a configuration file, whose path is given to the command. They can however be overridden from the command line with the `--param` flag. The sample config is used for both training and evaluation, so we will have to use the override to specify the edge set to use.
```bash
torchbiggraph_train examples/configs/fb15k_config.py -p edge_paths=data/FB15k/freebase_mtr100_mte100-train_partitioned
```

This will read data from the `entity_path` directory specified in the configuration and the `edge_paths` directory given on the command line. It will write checkpoints (which also double as the output data) to the `checkpoint_path` directory also defined in the configuration, which in this case is `model/fb15k`.

Training will proceed for 50 epochs in total, with the progress and some statistics logged to the console, for example:
```
Starting epoch 1 / 50 edge path 1 / 1 edge chunk 1 / 1
edge_path= data/FB15k/freebase_mtr100_mte100-train_partitioned
Swapping partitioned embeddings None ( 0 , 0 )
Loading entities
( 0 , 0 ): bucket 1 / 1 : Processed 483142 edges in 20.58 s ( 0.023 M/sec ); io: 0.02 s ( 296.64 MB/sec )
( 0 , 0 ): loss:  6663.96 , violators_lhs:  0 , violators_rhs:  0 , count:  483142
Swapping partitioned embeddings ( 0 , 0 ) None
Writing partitioned embeddings
Finished epoch 1 path 1 pass 1; checkpointing global state.
My rank: 0
Writing metadata...
Writing the checkpoint...
Switching to new checkpoint version...
```

### Evaluation

Once training is complete, the entity embeddings it produced can be evaluated against a held-out edge set, as follows:
```bash
torchbiggraph_eval examples/configs/fb15k_config.py -p edge_paths=data/FB15k/freebase_mtr100_mte100-test_partitioned relations.0.all_negs=true
```

This computes a set of metrics on the quality on the embeddings and prints them out. The last line should look like:
```
Stats: pos_rank:  65.4821 , mrr:  0.789921 , r1:  0.738501 , r10:  0.876894 , r50:  0.92647 , auc:  0.989868 , count:  59071
```
The values of `mrr` (Mean Reciprocal Rank, MRR) and `r10` (Hits@10) should match the ones reported in [the paper](https://www.sysml.cc/doc/2019/71.pdf).

The evaluation performed by the `examples/fb15k.py` script differs from the above `torchbiggraph_eval` command, in order to match the literature. It calculates the ranks of the edges in the evaluation set by comparing them against all other edges *except* the ones that are true positives in any of the training, validation or test set. This setup, called *filtered* MRR, is only used to evaluate small graphs because it scales very poorly.

### Converting the output

During preprocessing, the entities and relation types had their identifiers converted from strings to ordinals. In order to map the output embeddings back onto the original names, one can do:
```bash
torchbiggraph_export_to_tsv --dict data/FB15k/dictionary.json --checkpoint model/fb15k --out joined_embeddings.tsv
```
This will create the `joined_embeddings.tsv` file, which is a text file where each line contains the identifier of an entity or the name of a relation type followed respectively by its embedding or its parameters, each in a different column, all separated by tabs. For example, with each line shortened for brevity:
```
/m/0fphf3v	-0.524391472	-0.016430536	-0.461346656	-0.394277513	0.125605106	...
/m/01bns_	-0.122734159	-0.091636233	0.506501377	-0.503864646	0.215775326	...
/m/02ryvsw	-0.107151665	0.002058491	-0.094485454	-0.129078045	-0.123694092	...
/m/04y6_qr	-0.577532947	-0.215747222	-0.022358289	-0.352154016	-0.051905245	...
/m/02wrhj	-0.593656778	-0.557167351	0.042525314	-0.104738958	-0.265990764	...
```

## Documentation

More information can be found in [the full documentation](https://torchbiggraph.readthedocs.io/).

## Citation

To cite this work please use:
```tex
@inproceedings{pbg,
  title={{PyTorch-BigGraph: A Large-scale Graph Embedding System}},
  author={Lerer, Adam and Wu, Ledell and Shen, Jiajun and Lacroix, Timothee and Wehrstedt, Luca and Bose, Abhijit and Peysakhovich, Alex},
  booktitle={Proceedings of the 2nd SysML Conference},
  year={2019},
  address={Palo Alto, CA, USA}
}
```

## License

PyTorch-BigGraph is BSD licensed, as found in the [LICENSE](LICENSE) file.
