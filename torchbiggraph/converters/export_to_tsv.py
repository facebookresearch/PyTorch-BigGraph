#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import json
from typing import Dict, IO, Iterable, List, TextIO

import torch

from torchbiggraph.fileio import CheckpointManager
from torchbiggraph.model import make_model


def write(outf: TextIO, word: str, emb: Iterable[float]) -> None:
    outf.write("%s\t%s\n" % (word, "\t".join("%.9f" % x for x in emb)))


def make_tsv(
    checkpoint: str,
    relation_types: List[str],
    entities_by_type: Dict[str, List[str]],
    ent_out_file: IO[str],
    rel_out_file: IO[str],
) -> None:
    print("Loading model check point...")
    checkpoint_manager = CheckpointManager(checkpoint)
    config = checkpoint_manager.read_config()
    state_dict, _ = checkpoint_manager.read_model()
    model = make_model(config)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    for entity_name, entity in config.entities.items():
        entities = entities_by_type[entity_name]
        part_begin = 0
        for part in range(entity.num_partitions):
            print("Writing embeddings for entity type %s partition %d..."
                  % (entity_name, part))
            embs, _ = checkpoint_manager.read(entity_name, part)

            if model.global_embs is not None:
                embs += model.global_embs[model.EMB_PREFIX + entity_name]

            for ix in range(len(embs)):
                write(ent_out_file, entities[part_begin + ix], embs[ix])
                if (ix + 1) % 5000 == 0:
                    print("- Processed %d entities so far..." % (ix + 1))
            print("- Processed %d entities in total" % len(embs))

            part_begin = part_begin + len(embs)

    print("Done exporting entities data to %s" % getattr(ent_out_file, "name", "the output file"))

    # TODO Provide a better output format for relation parameters.
    print("Writing relation parameters...")
    if model.num_dynamic_rels > 0:
        lhs_parameters = torch.cat([
            parameter.view(model.num_dynamic_rels, -1)
            for parameter in model.lhs_operators[0].parameters()
        ], dim=1)
        for rel_idx, rel_name in enumerate(relation_types):
            write(rel_out_file, rel_name, lhs_parameters[rel_idx])

        rhs_parameters = torch.cat([
            parameter.view(model.num_dynamic_rels, -1)
            for parameter in model.rhs_operators[0].parameters()
        ], dim=1)
        for rel_idx, rel_name in enumerate(relation_types):
            write(rel_out_file, rel_name + "_reverse_relation", rhs_parameters[rel_idx])
    else:
        for rel_name, operator in zip(relation_types, model.rhs_operators):
            write(rel_out_file, rel_name, torch.cat([
                parameter.flatten() for parameter in operator.parameters()
            ], dim=0))

    print("Done exporting relations data to %s" % getattr(rel_out_file, "name", "the output file"))


def main():
    parser = argparse.ArgumentParser(description='Convert Data for PBG')
    parser.add_argument('--checkpoint')
    parser.add_argument('--dict', required=True)
    parser.add_argument('--out-ent', required=True)
    parser.add_argument('--out-rel', required=True)

    args = parser.parse_args()

    print("Loading relation types and entities...")
    with open(args.dict, "rt") as tf:
        dump = json.load(tf)

    with open(args.out_ent, "wt") as ent_out_tf, open(args.out_rel, "wt") as ent_out_tf:
        make_tsv(args.checkpoint, dump["relations"], dump["entities"], ent_out_tf, ent_out_tf)


if __name__ == "__main__":
    main()
