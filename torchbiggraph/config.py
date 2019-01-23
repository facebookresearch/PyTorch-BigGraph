#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
from enum import Enum
from itertools import chain
from typing import ClassVar, Dict, List, Optional

import attr

from .schema import Schema, schema, unpack_optional


class Operator(Enum):
    DIAGONAL = 'diagonal'
    TRANSLATION = 'translation'
    AFFINE_RHS = 'affine_rhs'
    NONE = 'none'


class Metric(Enum):
    COS = 'cos'
    DOT = 'dot'


class LossFn(Enum):
    RANKING = 'ranking'
    LOGISTIC = 'logistic'
    LOGIT = 'logistic'  # Old deprecated alias.
    SOFTMAX = 'softmax'


class PartitionOrder(Enum):
    CHAINED_SYMMETRIC_PAIRS = 'chained_symmetric_pairs'
    INSIDE_OUT = 'inside_out'
    OUTSIDE_IN = 'outside_in'
    RANDOM = 'random'


@schema
class EntitySchema(Schema):

    NAME: ClassVar[str] = "entity"

    numPartitions: int = attr.ib(
        metadata={'help': "Number of partitions for this entity type. "
                          "Typically, should be either 1 or the global number "
                          "of partitions"},
    )
    # TODO Turn into bool.
    featurized: Optional[int] = attr.ib(
        default=0,
        metadata={'help': "Whether we use featurized representation for the "
                          "entity."},
    )


@schema
class RelationSchema(Schema):

    NAME: ClassVar[str] = "relation"

    name: str = attr.ib()
    lhs: str = attr.ib()
    rhs: str = attr.ib()
    weight: Optional[float] = attr.ib(
        default=1.0,
    )
    operator: Optional[Operator] = attr.ib(
        default=Operator.NONE,
        metadata={'help': "Relation operator."},
    )
    all_rhs_negs: Optional[int] = attr.ib(
        default=0,
        metadata={'help': "Use all RHS entities as negatives (StarSpace style) "
                          "rather than negative sampling."},
    )


@schema
class ConfigSchema(Schema):

    NAME: ClassVar[str] = "config"

    # model config

    model: Optional[str] = attr.ib(
        default='f2_model',
        metadata={'help': "Model name."},
    )
    dimension: int = attr.ib(
        metadata={'help': "Embedding dimension."},
    )
    relations: Optional[List[RelationSchema]] = attr.ib(
        metadata={'help': "List of relation configs, matching the relation "
                          "type ids in the edge list."},
    )
    entities: Optional[Dict[str, EntitySchema]] = attr.ib(
        metadata={'help': "Dictionary {name: config} for each entity type."},
    )
    metric: Optional[Metric] = attr.ib(
        default=Metric.COS,
        metadata={'help': "Distance metric"},
    )
    lossFn: Optional[LossFn] = attr.ib(
        default=LossFn.RANKING,
        metadata={'help': "Loss function"},
    )
    # TODO Turn into bool.
    bias: Optional[int] = attr.ib(
        default=0,
        metadata={'help': "Add a bias term to the embeddings. Should be "
                          "enabled for logit/softmax embeddings"},
    )
    # TODO Turn into bool.
    globalEmb: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "Use a learned global embedding feature for each "
                          "entity type"},
    )
    maxNorm: Optional[float] = attr.ib(
        default=None,
        metadata={'help': "max norm for the embeddings"},
    )
    margin: Optional[float] = attr.ib(
        default=0.1,
        metadata={'help': "margin for ranking loss"},
    )
    initScale: Optional[float] = attr.ib(
        default=1e-3,
        metadata={'help': "scale for randomly initialized entity embeddings"},
    )

    # data config

    entityPath: Optional[str] = attr.ib(
        metadata={'help': "Path to directory containing entity metadata (from "
                          "download_entities.py)."},
    )
    edgePaths: List[str] = attr.ib(
        metadata={'help': "List of paths to directories containing "
                          "(partitioned) edgelists (from download_edges.py). "
                          "Each path corresponds to one epoch."},
    )
    outdir: Optional[str] = attr.ib(
        metadata={'help': "Directory to write embeddings (will continue from "
                          "checkpoint if it exists)."},
    )
    loadPath: Optional[str] = attr.ib(
        default=None,
        metadata={'help': "Optional: Initialize embeddings from a previous "
                          "outdir."},
    )

    # training config

    batchSize: Optional[int] = attr.ib(
        default=1000,
        metadata={'help': "Number of edges per batch."},
    )
    workers: Optional[int] = attr.ib(
        default=40,
        metadata={'help': "Number of worker threads for HOGWILD training."},
    )
    lr: Optional[float] = attr.ib(
        default=1e-2,
        metadata={'help': "Learning rate for the optimizer."},
    )
    numEpochs: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "Number of times to iterate through the edges for "
                          "training (i.e. through all the edgePaths)"},
    )
    numEdgeChunks: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "Number of times to iterate through all the "
                          "partitions during each epoch/edgePath, training on "
                          "a subset (chunk) of the edges at each pass. A "
                          "higher value allows better mixing for "
                          "multi-partition models, at the cost of more time "
                          "spent on I/O."},
    )
    numUniformNegs: Optional[int] = attr.ib(
        default=50,
        metadata={'help': "The number of uniformly-sampled negatives per "
                          "positive."},
    )
    numBatchNegs: Optional[int] = attr.ib(
        default=50,
        metadata={'help': "The number of negatives sampled from the batch, per "
                          "positive."},
    )
    partitionOrder: PartitionOrder = attr.ib(
        default=PartitionOrder.INSIDE_OUT,
        metadata={'help': "The order in which to iterate over left-hand-side "
                          "and right-hand-side partitions."},
    )
    evalFraction: Optional[float] = attr.ib(
        default=0.05,
        metadata={'help': "The fraction of edges to withhold from training and "
                          "use to track evaluation metrics during training."},
    )
    evalNumUniformNegs: Optional[int] = attr.ib(
        default=1000,
        metadata={'help': "The value that overrides the number of "
                          "uniformly-sampled negatives per positive during the "
                          "evaluation steps that occur before and after each "
                          "training step."},
    )
    evalNumBatchNegs: Optional[int] = attr.ib(
        default=1000,
        metadata={'help': "The value that overrides the number of negatives "
                          "per positive sampled from the batch during the "
                          "evaluation steps that occur before and after each "
                          "training step."},
    )

    # expert options

    # TODO Turn into bool.
    background_io: Optional[int] = attr.ib(
        default=0,
        metadata={'help': "Do load/save in a background process"},
    )
    verbose: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "Verbosity level"},
    )
    hogwild_delay: float = attr.ib(
        default=2,
        metadata={'help': "Delay all threads but one; delay=2 seems to help."},
    )
    numDynamicRels: int = attr.ib(
        default=0,
        metadata={'help': "Run with dynamic relation batches. This is set "
                          "automatically in the code based on the edgelist. If "
                          "set (non-zero), there should only be a single entry "
                          "in the 'relations' list (whose config applies to "
                          "all relations)."},
    )

    # distributed training config options

    numMachines: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "Number of machines for distributed training."},
    )
    numPartitionServers: Optional[int] = attr.ib(
        default=-1,
        metadata={'help': "If -1, use machines as partition servers. If 0, no "
                          "partition servers (swap partitions through disk). "
                          "If >1, then expects N independent servers "
                          "(partitionserver.py) to be set up."},
    )
    distributedInitMethod: Optional[str] = attr.ib(
        default=None,
        metadata={'help': "A URI defining how to synchronize all the workers "
                          "of a distributed run. Must start with a scheme "
                          "(e.g., file:// or tcp://) supported by PyTorch."}
    )
    # TODO Turn into bool.
    distributedTreeInitOrder: Optional[int] = attr.ib(
        default=1,
        metadata={'help': "If enabled, then only one partition starts as "
                          "'initialized', and a pair can only be trained if "
                          ">=1 of its LHS or RHS partition is initialized. "
                          "Then the other side becomes initialized."},
    )


def parse_config_base(config, overrides=None):
    spec = importlib.util.spec_from_file_location("config_module", config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user_config = module.getConf()
    assert user_config is not None, "your config file didn't return anything"
    if overrides is not None:
        for override in overrides:
            try:
                key, value = override.split('=')
                param_type = attr.fields_dict(ConfigSchema)[key].type or str
                # this is a bit of a hack; we should do something better
                # but this is convenient for specifying lists of strings
                # e.g. edgePaths
                try:
                    param_type = unpack_optional(param_type)
                except TypeError:
                    pass
                if isinstance(param_type, type) and issubclass(param_type, list):
                    value = value.split(",")
                if isinstance(param_type, type) and not issubclass(param_type, Enum):
                    value = param_type(value)
                user_config[key] = value
            except Exception as e:
                assert False, "Can't parse override: {} . {}".format(override, e)
    return user_config


def parse_config(config, overrides=None):
    user_config = parse_config_base(config, overrides)
    full_config = ConfigSchema.from_dict(user_config)
    # Late import to avoid circular dependency.
    from . import util
    util._verbosity_level = full_config.verbose
    return full_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('query', help="Name of param to retrieve")
    parser.add_argument('-p', '--param', action='append', nargs='*')
    opt = parser.parse_args()

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
    else:
        overrides = None

    config = parse_config(opt.config, overrides)
    print(config[opt.query])


if __name__ == '__main__':
    main()
