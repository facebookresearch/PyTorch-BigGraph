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
    # No-op.
    NONE = 'none'
    # Add a vector of same dimension.
    TRANSLATION = 'translation'
    # Multiply by a diagonal matrix.
    DIAGONAL = 'diagonal'
    # Multiply by a full square matrix.
    LINEAR = 'linear'
    # Multiply by a full square matrix, then translate.
    AFFINE = 'affine'
    # Treat the D-dimensional embedding as D/2 complex numbers (the half of the
    # vector is their real parts, the second half their imaginary parts) and
    # multiply them pointwise by another vector of D/2 complex numbers.
    COMPLEX_DIAGONAL = 'complex_diagonal'


class Comparator(Enum):
    # Dot product.
    DOT = 'dot'
    # Cosine distance.
    COS = 'cos'


class LossFunction(Enum):
    # No loss if the positive score is greater than the negative score by at
    # least the given margin, otherwise the loss is the amount by which that
    # inequality is violated. This is the hinge loss on the difference between
    # positive and negative scores.
    RANKING = 'ranking'
    # The loss is the cross entropy between the probabilities of the edge
    # existing in the model (its score, passed through the logistic function)
    # and in the training set (1 for positives, 0 for negatives). The loss of
    # the negatives is renormalized so it compares with the positive loss.
    LOGISTIC = 'logistic'
    # For each positive and its corresponding negatives, their probabilities are
    # computed as a softmax, and the loss is the cross entropy between them and
    # the "target" distribution (1 for the positive, 0 for the negatives).
    SOFTMAX = 'softmax'


class BucketOrder(Enum):
    # Random permutation/shuffle.
    RANDOM = 'random'
    # Each bucket will have as many partitions as possible in common with the
    # preceding bucket (ideally both, otherwise only one, else none). If
    # multiple candidate buckets exist, one is picked randomly.
    AFFINITY = 'affinity'
    # Enforce that (L1, R1) comes before (L2, R2) iff min(L1, R1) > min(L2, R2)
    # (subject to that condition, shuffle randomly).
    INSIDE_OUT = 'inside_out'
    # The "per-layer" reverse of outside-in: (L1, R1) comes before (L2, R2) iff
    # min(L1, R1) > min(L2, R2).
    OUTSIDE_IN = 'outside_in'


@schema
class EntitySchema(Schema):

    NAME: ClassVar[str] = "entity"

    num_partitions: int = attr.ib(
        metadata={'help': "Number of partitions for this entity type. Set to 1 "
                          "if unpartitioned. All other entity types must have "
                          "the same number of partitions."},
    )
    featurized: bool = attr.ib(
        default=False,
        metadata={'help': "Whether the entities of this type are represented "
                          "as sets of features."},
    )


@schema
class RelationSchema(Schema):

    NAME: ClassVar[str] = "relation"

    name: str = attr.ib(
        metadata={'help': "A human-readable identifier for the relation type. "
                          "Not needed for training, only used for logging."},
    )
    lhs: str = attr.ib(
        metadata={'help': "The type of entities on the left-hand side of this "
                          "relation, i.e., its key in the entities dict."},
    )
    rhs: str = attr.ib(
        metadata={'help': "The type of entities on the right-hand side of this "
                          "relation, i.e., its key in the entities dict."},
    )
    weight: float = attr.ib(
        default=1.0,
        metadata={'help': "The weight by which the loss induced by edges of "
                          "this relation type will be multiplied."},
    )
    operator: Operator = attr.ib(
        default=Operator.NONE,
        metadata={'help': "The transformation to apply to the embedding of one "
                          "of the sides of the edge (typically the right-hand "
                          "one) before comparing it with the other one."},
    )
    all_negs: bool = attr.ib(
        default=False,
        metadata={'help': "If enabled, the negatives for (s, r, d) will "
                          "consist of (s, r, d') for all entities d' of the "
                          "same type and in the same partitition as d, and, "
                          "symmetrically, of (s', r, d) for all entities s' of "
                          "the same type and in the same partition as s."},
    )


@schema
class ConfigSchema(Schema):

    NAME: ClassVar[str] = "config"

    # model config

    entities: Dict[str, EntitySchema] = attr.ib(
        metadata={'help': "The entity types. The ID with which they are "
                          "referenced by the relation types is the key they "
                          "have in this dict."},
    )
    relations: List[RelationSchema] = attr.ib(
        metadata={'help': "The relation types. The ID with which they will be "
                          "referenced in the edge lists is their index in this "
                          "list plus one (as we use Lua-style indexing)."},
    )
    dimension: int = attr.ib(
        metadata={'help': "The dimension of the real space the embedding live "
                          "in."},
    )
    init_scale: float = attr.ib(
        default=1e-3,
        metadata={'help': "If no initial embeddings are provided, they are "
                          "generated by sampling each dimension from a "
                          "centered normal distribution having this standard "
                          "deviation. (For performance reasons, sampling isn't "
                          "fully independent.)"},
    )
    max_norm: Optional[float] = attr.ib(
        default=None,
        metadata={'help': "If set, rescale the embeddings if their norm "
                          "exceeds this value."},
    )
    global_emb: bool = attr.ib(
        default=True,
        metadata={'help': "If enabled, add to each embedding a vector that is "
                          "common to all the entities of a certain type. This "
                          "vector is learned during training."},
    )
    comparator: Comparator = attr.ib(
        default=Comparator.COS,
        metadata={'help': "How the embeddings of the two sides of an edge "
                          "(after having already undergone some processing) "
                          "are compared to each other to produce a score."},
    )
    bias: bool = attr.ib(
        default=False,
        metadata={'help': "If enabled, withhold the first dimension of the "
                          "embeddings from the comparator and instead use it "
                          "as a bias, adding back to the score. Makes sense "
                          "for logistic and softmax loss functions."},
    )
    loss_fn: LossFunction = attr.ib(
        default=LossFunction.RANKING,
        metadata={'help': "How the scores of positive edges and their "
                          "corresponding negatives are evaluated."},
    )
    margin: Optional[float] = attr.ib(
        default=0.1,
        metadata={'help': "When using ranking loss, this value controls the "
                          "minimum distance there must be between positive and "
                          "negative scores in order to incur in no loss."},
    )

    # data config

    entity_path: str = attr.ib(
        metadata={'help': "The path of the directory containing entity count "
                          "files."},
    )
    edge_paths: List[str] = attr.ib(
        metadata={'help': "A list of paths to directories containing "
                          "(partitioned) edgelists."},
    )
    checkpoint_path: str = attr.ib(
        metadata={'help': "The path to the directory where checkpoints (and "
                          "thus the output) will be written to. If checkpoints "
                          "are found in it, training will resume from them."},
    )
    init_path: Optional[str] = attr.ib(
        default=None,
        metadata={'help': "If set, it must be a path to a directory that "
                          "contains initial values for the embeddings of all "
                          "the entities."},
    )

    # training config

    num_epochs: int = attr.ib(
        default=1,
        metadata={'help': "The number of times the training loop iterates over "
                          "all the edges."},
    )
    num_edge_chunks: int = attr.ib(
        default=1,
        metadata={'help': "The number of equally-sized parts each bucket will "
                          "be split into. Training will first proceed over all "
                          "the first chunks of all buckets, then over all the "
                          "second chunks, and so on. A higher value allows "
                          "better mixing of partitions, at the cost of more "
                          "time spent on I/O."},
    )
    bucket_order: BucketOrder = attr.ib(
        default=BucketOrder.INSIDE_OUT,
        metadata={'help': "The order in which to iterate over the buckets."},
    )
    workers: int = attr.ib(
        default=40,
        metadata={'help': "The number of worker processes for \"Hogwild!\" "
                          "training."},
    )
    batch_size: int = attr.ib(
        default=1000,
        metadata={'help': "The number of edges per batch."},
    )
    num_batch_negs: int = attr.ib(
        default=50,
        metadata={'help': "The number of negatives sampled from the batch, per "
                          "positive."},
    )
    num_uniform_negs: int = attr.ib(
        default=50,
        metadata={'help': "The number of uniformly-sampled negatives per "
                          "positive."},
    )
    lr: float = attr.ib(
        default=1e-2,
        metadata={'help': "The learning rate for the optimizer."},
    )
    eval_fraction: float = attr.ib(
        default=0.05,
        metadata={'help': "The fraction of edges withheld from training and "
                          "used to track evaluation metrics during training."},
    )
    eval_num_batch_negs: int = attr.ib(
        default=1000,
        metadata={'help': "The value that overrides the number of negatives "
                          "per positive sampled from the batch during the "
                          "evaluation steps that occur before and after each "
                          "training step."},
    )
    eval_num_uniform_negs: int = attr.ib(
        default=1000,
        metadata={'help': "The value that overrides the number of "
                          "uniformly-sampled negatives per positive during the "
                          "evaluation steps that occur before and after each "
                          "training step."},
    )

    # expert options

    background_io: bool = attr.ib(
        default=False,
        metadata={'help': "Whether to do load/save in a background process."},
    )
    verbose: int = attr.ib(
        default=1,
        metadata={'help': "The verbosity level of logging, currently 0 or 1."},
    )
    hogwild_delay: float = attr.ib(
        default=2,
        metadata={'help': "The number of seconds by which to delay the start "
                          "of all \"Hogwild!\" processes except the first "
                          "one."},
    )
    num_dynamic_rels: int = attr.ib(
        default=0,
        metadata={'help': "When set to a non-zero value, activates the dynamic "
                          "relation mode, in which case, there must be a "
                          "single relation type in the config (whose "
                          "parameters will apply to all dynamic relations "
                          "types). This flag will get set automatically when "
                          "the input files are in the right format, so it "
                          "should not be activated in the config. In this mode "
                          "batches will contain edges of multiple relation "
                          "types and negatives will be samples differently."},
    )

    # distributed training config options

    num_machines: int = attr.ib(
        default=1,
        metadata={'help': "The number of machines for distributed training."},
    )
    num_partition_servers: int = attr.ib(
        default=-1,
        metadata={'help': "If -1, use trainer as partition servers. If 0, "
                          "don't use partition servers (instead, swap "
                          "partitions through disk). If >1, then that number "
                          "of partition servers must be started manually."},
    )
    distributed_init_method: Optional[str] = attr.ib(
        default=None,
        metadata={'help': "A URI defining how to synchronize all the workers "
                          "of a distributed run. Must start with a scheme "
                          "(e.g., file:// or tcp://) supported by PyTorch."}
    )
    distributed_tree_init_order: bool = attr.ib(
        default=True,
        metadata={'help': "If enabled, then distributed training can occur on "
                          "a bucket only if at least one of its partitions was "
                          "already trained on before in the same round (or if "
                          "one of its partitions is 0, for bootstrapping)."},
    )


def parse_config_base(config, overrides=None):
    spec = importlib.util.spec_from_file_location("config_module", config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user_config = module.get_torchbiggraph_config()
    assert user_config is not None, "your config file didn't return anything"
    if overrides is not None:
        for override in overrides:
            try:
                key, value = override.split('=')
                param_type = attr.fields_dict(ConfigSchema)[key].type or str
                # this is a bit of a hack; we should do something better
                # but this is convenient for specifying lists of strings
                # e.g. edge_paths
                try:
                    param_type = unpack_optional(param_type)
                except TypeError:
                    pass
                if isinstance(param_type, type) and issubclass(param_type, list):
                    value = value.split(",")
                # Convert numbers (caution: ignore bools, which are ints)
                if isinstance(param_type, type) \
                        and issubclass(param_type, (int, float)) \
                        and not issubclass(param_type, bool):
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
