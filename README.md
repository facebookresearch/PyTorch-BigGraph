# ![PyTorch-BigGraph](docs/source/_static/logo_color.svg)

[![CircleCI Status](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph.svg?style=svg)](https://circleci.com/gh/facebookresearch/PyTorch-BigGraph) [![Documentation Status](https://readthedocs.org/projects/torchbiggraph/badge/?version=latest)](https://torchbiggraph.readthedocs.io/en/latest/?badge=latest)

PyTorch-BigGraph (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.

**Update:** *PBG now supports GPU training. Check out the [GPU Training](#gpu-training) section below!*

PBG was introduced in the [PyTorch-BigGraph: A Large-scale Graph Embedding Framework](https://mlsys.org/Conferences/2019/doc/2019/71.pdf) paper, presented at the [SysML conference](https://mlsys.org/) in 2019.

PBG trains on an input graph by ingesting its list of edges, each identified by its source and target entities and, possibly, a relation type. It outputs a feature vector (embedding) for each entity, trying to place adjacent entities close to each other in the vector space, while pushing unconnected entities apart. Therefore, entities that have a similar distribution of neighbors will end up being nearby.

It is possible to configure each relation type to calculate this "proximity score" in a different way, with the parameters (if any) learned during training. This allows the same underlying entity embeddings to be shared among multiple relation types.

The generality and extensibility of its model allows PBG to train a number of models from the knowledge graph embedding literature, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf), [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf), [DistMult](https://arxiv.org/abs/1412.6575) and [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

PBG is designed with scale in mind, and achieves it through:
- *graph partitioning*, so that the model does not have to be fully loaded into memory
- *multi-threaded computation* on each machine
- *distributed execution* across multiple machines (optional), all simultaneously operating on disjoint parts of the graph
- *batched negative sampling*, allowing for processing >1 million edges/sec/machine with 100 negatives per edge

PBG is not optimized for small graphs. If your graph has fewer than 100,000 nodes, consider using [KBC](https://github.com/facebookresearch/kbc) with the ComplEx model and N3 regularizer. KBC produces state-of-the-art embeddings for graphs that can fit on a single GPU. Compared to KBC, PyTorch-BigGraph enables learning on very large graphs whose embeddings wouldn't fit in a single GPU or a single machine, but may not produce high-quality embeddings for small graphs without careful tuning.


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
pip install .
```

Note: PyTorch-BigGraph includes some C++ kernels that are only used for the experimental GPU mode. If you are seeing C++ compilation errors during installation, you can turn off C++ compilation by running

```bash
PBG_INSTALL_CPP=0 pip install .
```

Everything will work identically except that you won't be able to run GPU training (`torchbiggraph_train_gpu`).


## Getting started

The results of [the paper](https://mlsys.org/Conferences/2019/doc/2019/71.pdf) can easily be reproduced by running the following command (which executes [this script](torchbiggraph/examples/fb15k.py)):
```bash
torchbiggraph_example_fb15k
```
This will download the Freebase 15k knowledge base dataset, put it into the right format, train on it using the ComplEx model and finally perform an evaluation of the learned embeddings that calculates the MRR and other metrics that should match the paper. Another command, `torchbiggraph_example_livejournal`, does the same for the LiveJournal interaction graph dataset.

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
torchbiggraph_import_from_tsv \
  --lhs-col=0 --rel-col=1 --rhs-col=2 \
  torchbiggraph/examples/configs/fb15k_config_cpu.py \
  data/FB15k/freebase_mtr100_mte100-train.txt \
  data/FB15k/freebase_mtr100_mte100-valid.txt \
  data/FB15k/freebase_mtr100_mte100-test.txt
```
The outputs will be stored next to the inputs in the `data/FB15k` directory.

This simple utility is only suitable for small graphs that fit entirely in memory. To handle larger data one will have to implement their own custom preprocessor.

### Training

The `torchbiggraph_train` command is used to launch training. The training parameters are tucked away in a configuration file, whose path is given to the command. They can however be overridden from the command line with the `--param` flag. The sample config is used for both training and evaluation, so we will have to use the override to specify the edge set to use.
```bash
torchbiggraph_train \
  torchbiggraph/examples/configs/fb15k_config.py \
  -p edge_paths=data/FB15k/freebase_mtr100_mte100-train_partitioned
```

This will read data from the `entity_path` directory specified in the configuration and the `edge_paths` directory given on the command line. It will write checkpoints (which also double as the output data) to the `checkpoint_path` directory also defined in the configuration, which in this case is `model/fb15k`.

Training will proceed for 50 epochs in total, with the progress and some statistics logged to the console, for example:
```
Starting epoch 1 / 50, edge path 1 / 1, edge chunk 1 / 1
Edge path: data/FB15k/freebase_mtr100_mte100-train_partitioned
still in queue: 0
Swapping partitioned embeddings None ( 0 , 0 )
( 0 , 0 ): Loading entities
( 0 , 0 ): bucket 1 / 1 : Processed 483142 edges in 17.36 s ( 0.028 M/sec ); io: 0.02 s ( 542.52 MB/sec )
( 0 , 0 ): loss:  309.695 , violators_lhs:  171.846 , violators_rhs:  165.525 , count:  483142
Swapping partitioned embeddings ( 0 , 0 ) None
Writing partitioned embeddings
Finished epoch 1 / 50, edge path 1 / 1, edge chunk 1 / 1
Writing the metadata
Writing the checkpoint
Switching to the new checkpoint version
```

### GPU Training

*Warning: GPU Training is still experimental; expect sharp corners and lack of documentation.*

`torchbiggraph_example_fb15k` will automatically detect if a GPU is available and run with the GPU training config. For your own training runs, you will need to change a few parameters to enable GPU training. Lets see how the two FB15k configs differ:

```
$ diff torchbiggraph/examples/configs/fb15k_config_cpu.py torchbiggraph/examples/configs/fb15k_config_gpu.py
37a38
>         batch_size=10000,
42a44,45
>         # GPU
>         num_gpus=1,

```
The most important difference is of course `num_gpus=1`, which says to run on 1 GPU. If `num_gpus=N>1`, PBG will recursively shard the embeddings within each partition into `N` subpartitions to run on multiple GPUs. The subpartitions need to fit in GPU memory, so if you get CUDA out-of-memory errors, you'll need to increase `num_partitions` or `num_gpus`.

The next most important difference for GPU training is that `batch_size` must be much larger. Since training is being performed on a single GPU instead of 40 cores, the batch size can be increased by about that factor as well. We suggest batch size of around 100,000 in order to achieve good speeds for GPU training.

Since evaluation still occurs on CPU, we suggest turning down `eval_fraction` to at most `0.01` so that evaluation does not become a bottleneck (not relevant for FB15k which doesn't do eval during training).

Finally, to take advantage of GPU speed, we suggest turning up `num_uniform_negatives` and/or `num_batch_negatives` to about `1000` rather than their default values of `50` (FB15k already uses 1000 uniform negatives).

### Evaluation

Once training is complete, the entity embeddings it produced can be evaluated against a held-out edge set. The `torchbiggraph_example_fb15k` command performs a *filtered* evaluation, which calculates the ranks of the edges in the evaluation set by comparing them against all other edges *except* the ones that are true positives in any of the training, validation or test set. Filtered evaluation is used in the literature for FB15k, but does not scale beyond small graphs.

The final results should match the values of `mrr` (Mean Reciprocal Rank, MRR) and `r10` (Hits@10) reported in [the paper](https://mlsys.org/Conferences/2019/doc/2019/71.pdf):
```
Stats: pos_rank:  65.4821 , mrr:  0.789921 , r1:  0.738501 , r10:  0.876894 , r50:  0.92647 , auc:  0.989868 , count:  59071
```

Evaluation can also be run directly from the command line as follows:
```bash
torchbiggraph_eval \
  torchbiggraph/examples/configs/fb15k_config_cpu.py \
  -p edge_paths=data/FB15k/freebase_mtr100_mte100-test_partitioned \
  -p relations.0.all_negs=true \
  -p num_uniform_negs=0
```

However, *filtered* evaluation *cannot* be performed on the command line, so the reported results will not match the paper. They will be something like:
```
Stats: pos_rank:  234.136 , mrr:  0.239957 , r1:  0.131757 , r10:  0.485382 , r50:  0.712693 , auc:  0.989648 , count:  59071
```

### Converting the output

During preprocessing, the entities and relation types had their identifiers converted from strings to ordinals. In order to map the output embeddings back onto the original names, one can do:
```bash
torchbiggraph_export_to_tsv \
  torchbiggraph/examples/configs/fb15k_config.py \
  --entities-output entity_embeddings.tsv \
  --relation-types-output relation_types_parameters.tsv
```
This will create the `entity_embeddings.tsv` file, which is a text file where each line contains the identifier of an entity followed respectively by the components of its embedding, each in a different column, all separated by tabs. For example, with each line shortened for brevity:
```
/m/0fphf3v	-0.524391472	-0.016430536	-0.461346656	-0.394277513	0.125605106	...
/m/01bns_	-0.122734159	-0.091636233	0.506501377	-0.503864646	0.215775326	...
/m/02ryvsw	-0.107151665	0.002058491	-0.094485454	-0.129078045	-0.123694092	...
/m/04y6_qr	-0.577532947	-0.215747222	-0.022358289	-0.352154016	-0.051905245	...
/m/02wrhj	-0.593656778	-0.557167351	0.042525314	-0.104738958	-0.265990764	...
```
It will also create a `relation_types_parameters.tsv` file which contains the parameters of the operators for the relation types. The format is similar to the above, but each line starts with more key columns containing, respectively, the name of a relation type, a side (`lhs` or `rhs`), the name of the operator which is used by that relation type on that side, the name of a parameter of that operator and the shape of the parameter (integers separated by `x`). These columns are followed by the values of the flattened parameter. For example, for two relation types, `foo` and `bar`, respectively using operators `linear` and `complex_diagonal`, with an embedding dimension of 200 and dynamic relations enabled, this file could look like:
```
foo	lhs	linear	linear_transformation	200x200	-0.683401227	0.209822774	-0.047136042	...
foo	rhs	linear	linear_transformation	200x200	-0.695254087	0.502532542	-0.131654695	...
bar	lhs	complex_diagonal	real	200	0.263731539	1.350529909	1.217602968	...
bar	lhs	complex_diagonal	imag	200	-0.089371338	-0.092713356	0.025076168	...
bar	rhs	complex_diagonal	real	200	-2.350617170	0.529571176	0.521403074	...
bar	rhs	complex_diagonal	imag	200	0.692483306	0.446569800	0.235914066	...
```

## Documentation

More information can be found in [the full documentation](https://torchbiggraph.readthedocs.io/).

## Pre-trained embeddings

We trained a PBG model on the full [Wikidata](https://www.wikidata.org/) graph, using a [translation operator](https://torchbiggraph.readthedocs.io/en/latest/scoring.html#operators) to represent relations. It can be downloaded [here](https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz) (36GiB, gzip-compressed). We used the truthy version of data from [here](https://dumps.wikimedia.org/wikidatawiki/entities/) to train our model. The model file is in TSV format as described in the above section. Note that the first line of the file contains the number of entities, the number of relations and the dimension of the embeddings, separated by tabs. The model contains 78 million entities, 4,131 relations and the dimension of the embeddings is 200.

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

PyTorch-BigGraph is BSD licensed, as found in the [LICENSE.txt](LICENSE.txt) file.
