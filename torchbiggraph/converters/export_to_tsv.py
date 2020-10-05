#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
from typing import Iterable, TextIO

from torchbiggraph.checkpoint_manager import CheckpointManager
from torchbiggraph.config import ConfigFileLoader, ConfigSchema
from torchbiggraph.graph_storages import (
    ENTITY_STORAGES,
    RELATION_TYPE_STORAGES,
    AbstractEntityStorage,
    AbstractRelationTypeStorage,
)
from torchbiggraph.model import MultiRelationEmbedder, make_model


def write(outf: TextIO, key: Iterable[str], value: Iterable[float]) -> None:
    outf.write("%s\t%s\n" % ("\t".join(key), "\t".join("%.9f" % x for x in value)))


def make_tsv(
    config: ConfigSchema, entities_tf: TextIO, relation_types_tf: TextIO
) -> None:
    print("Loading relation types and entities...")
    entity_storage = ENTITY_STORAGES.make_instance(config.entity_path)
    relation_type_storage = RELATION_TYPE_STORAGES.make_instance(config.entity_path)

    print("Initializing model...")
    model = make_model(config)

    print("Loading model check point...")
    checkpoint_manager = CheckpointManager(config.checkpoint_path)
    state_dict, _ = checkpoint_manager.read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    make_tsv_for_entities(model, checkpoint_manager, entity_storage, entities_tf)
    make_tsv_for_relation_types(model, relation_type_storage, relation_types_tf)


def make_tsv_for_entities(
    model: MultiRelationEmbedder,
    checkpoint_manager: CheckpointManager,
    entity_storage: AbstractEntityStorage,
    entities_tf: TextIO,
) -> None:
    print("Writing entity embeddings...")
    for ent_t_name, ent_t_config in model.entities.items():
        for partition in range(ent_t_config.num_partitions):
            print(
                f"Reading embeddings for entity type {ent_t_name} partition "
                f"{partition} from checkpoint..."
            )
            entities = entity_storage.load_names(ent_t_name, partition)
            embeddings, _ = checkpoint_manager.read(ent_t_name, partition)

            if model.global_embs is not None:
                embeddings += model.global_embs[model.EMB_PREFIX + ent_t_name]

            print(
                f"Writing embeddings for entity type {ent_t_name} partition "
                f"{partition} to output file..."
            )
            for ix in range(len(embeddings)):
                write(entities_tf, (entities[ix],), embeddings[ix])
                if (ix + 1) % 5000 == 0:
                    print(f"- Processed {ix+1}/{len(embeddings)} entities so far...")
            print(f"- Processed all {len(embeddings)} entities")

    entities_output_filename = getattr(entities_tf, "name", "the output file")
    print(f"Done exporting entity data to {entities_output_filename}")


def make_tsv_for_relation_types(
    model: MultiRelationEmbedder,
    relation_type_storage: AbstractRelationTypeStorage,
    relation_types_tf: TextIO,
) -> None:
    print("Writing relation type parameters...")
    relation_types = relation_type_storage.load_names()
    if model.num_dynamic_rels > 0:
        (rel_t_config,) = model.relations
        op_name = rel_t_config.operator
        (lhs_operator,) = model.lhs_operators
        (rhs_operator,) = model.rhs_operators
        for side, operator in [("lhs", lhs_operator), ("rhs", rhs_operator)]:
            for param_name, all_params in operator.named_parameters():
                for rel_t_name, param in zip(relation_types, all_params):
                    shape = "x".join(f"{d}" for d in param.shape)
                    write(
                        relation_types_tf,
                        (rel_t_name, side, op_name, param_name, shape),
                        param,
                    )
    else:
        for rel_t_name, rel_t_config, operator in zip(
            relation_types, model.relations, model.rhs_operators
        ):
            if rel_t_name != rel_t_config.name:
                raise ValueError(
                    f"Mismatch in relations names: got {rel_t_name} in the "
                    f"dictionary and {rel_t_config.name} in the config."
                )
            op_name = rel_t_config.operator
            for param_name, param in operator.named_parameters():
                shape = "x".join(f"{d}" for d in param.shape)
                write(
                    relation_types_tf,
                    (rel_t_name, "rhs", op_name, param_name, shape),
                    param,
                )

    relation_types_output_filename = getattr(
        relation_types_tf, "name", "the output file"
    )
    print(f"Done exporting relation type data to {relation_types_output_filename}")


def main():
    config_help = "\n\nConfig parameters:\n\n" + "\n".join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument("--entities-output", required=True)
    parser.add_argument("--relation-types-output", required=True)
    opt = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, opt.param)

    with open(opt.entities_output, "xt") as entities_tf, open(
        opt.relation_types_output, "xt"
    ) as relation_types_tf:
        make_tsv(config, entities_tf, relation_types_tf)


if __name__ == "__main__":
    main()
