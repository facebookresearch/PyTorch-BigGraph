#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import os.path
import random
from typing import Counter, DefaultDict, Dict, List, Optional, Tuple

import h5py
import numpy as np

from torchbiggraph.config import \
    ConfigSchema, EntitySchema, RelationSchema, get_config_dict_from_module
from torchbiggraph.converters.dictionary import Dictionary


def collect_relation_types(
    relation_configs: List[RelationSchema],
    edge_paths: List[str],
    dynamic_relations: bool,
    rel_col: Optional[int],
    relation_type_min_count: int,
) -> Dictionary:

    if dynamic_relations:
        if rel_col is None:
            raise RuntimeError("Need to specify rel_col in dynamic mode.")
        print("Looking up relation types in the edge files...")
        counter: Counter[str] = Counter()
        for edgepath in edge_paths:
            with open(edgepath, "rt") as tf:
                for line in tf:
                    counter[line.split()[rel_col]] += 1
        print("- Found %d relation types" % len(counter))
        if relation_type_min_count > 0:
            print("- Removing the ones with fewer than %d occurrences..."
                  % relation_type_min_count)
            counter = Counter({k: c for k, c in counter.items()
                               if c >= relation_type_min_count})
            print("- Left with %d relation types" % len(counter))
        print("- Shuffling them...")
        names = list(counter.keys())
        random.shuffle(names)

    else:
        names = [rconfig.name for rconfig in relation_configs]
        print("Using the %d relation types given in the config" % len(names))

    return Dictionary(names)


def collect_entities_by_type(
    relation_types: Dictionary,
    entity_configs: Dict[str, EntitySchema],
    relation_configs: List[RelationSchema],
    edge_paths: List[str],
    dynamic_relations: bool,
    lhs_col: int,
    rhs_col: int,
    rel_col: Optional[int],
    entity_min_count: int,
) -> Dict[str, Dictionary]:

    counters: Dict[str, Counter[str]] = {}
    for entity_name in entity_configs.keys():
        counters[entity_name] = Counter()

    print("Searching for the entities in the edge files...")
    for edgepath in edge_paths:
        with open(edgepath, "rt") as tf:
            for line in tf:
                words = line.split()

                if dynamic_relations or rel_col is None:
                    rel_id = 0
                else:
                    rel = words[rel_col]
                    try:
                        rel_id = relation_types.get_id(rel)
                    except KeyError:
                        raise RuntimeError("Could not find relation type in config")

                counters[relation_configs[rel_id].lhs][words[lhs_col]] += 1
                counters[relation_configs[rel_id].rhs][words[rhs_col]] += 1

    entities_by_type: Dict[str, Dictionary] = {}
    for entity_name, counter in counters.items():
        print("Entity type %s:" % entity_name)
        print("- Found %d entities" % len(counter))
        if entity_min_count > 0:
            print("- Removing the ones with fewer than %d occurrences..."
                  % entity_min_count)
            counter = Counter({k: c for k, c in counter.items()
                               if c >= entity_min_count})
            print("- Left with %d entities" % len(counter))
        print("- Shuffling them...")
        names = list(counter.keys())
        random.shuffle(names)
        entities_by_type[entity_name] = Dictionary(
            names, num_parts=entity_configs[entity_name].num_partitions)

    return entities_by_type


def generate_entity_path_files(
    entity_path: str,
    entities_by_type: Dict[str, Dictionary],
    relation_types: Dictionary,
    dynamic_relations: bool,
) -> None:

    print("Preparing entity path %s:" % entity_path)
    for entity_name, entities in entities_by_type.items():
        for part in range(entities.num_parts):
            print("- Writing count of entity type %s and partition %d"
                  % (entity_name, part))
            with open(os.path.join(
                entity_path, "entity_count_%s_%d.txt" % (entity_name, part)
            ), "wt") as tf:
                tf.write("%d" % entities.part_size(part))

    if dynamic_relations:
        print("- Writing count of dynamic relations")
        with open(os.path.join(entity_path, "dynamic_rel_count.txt"), "wt") as tf:
            tf.write("%d" % relation_types.size())


def generate_edge_path_files(
    edge_file_in: str,
    entities_by_type: Dict[str, Dictionary],
    relation_types: Dictionary,
    relation_configs: List[RelationSchema],
    dynamic_relations: bool,
    lhs_col: int,
    rhs_col: int,
    rel_col: Optional[int],
) -> None:

    basename, _ = os.path.splitext(edge_file_in)
    edge_path_out = basename + '_partitioned'

    print("Preparing edge path %s, out of the edges found in %s"
          % (edge_path_out, edge_file_in))
    os.makedirs(edge_path_out, exist_ok=True)

    num_lhs_parts = max(entities_by_type[rconfig.lhs].num_parts
                        for rconfig in relation_configs)
    num_rhs_parts = max(entities_by_type[rconfig.rhs].num_parts
                        for rconfig in relation_configs)

    print("- Edges will be partitioned in %d x %d buckets."
          % (num_lhs_parts, num_rhs_parts))

    buckets: DefaultDict[Tuple[int, int], List[Tuple[int, int, int]]] = \
        DefaultDict(list)
    processed = 0
    skipped = 0

    with open(edge_file_in, "rt") as tf:
        for line in tf:
            words = line.split()
            if rel_col is None:
                rel_id = 0
            else:
                try:
                    rel_id = relation_types.get_id(words[rel_col])
                except KeyError:
                    # Ignore edges whose relation type is not known.
                    skipped += 1
                    continue

            if dynamic_relations:
                lhs_type = relation_configs[0].lhs
                rhs_type = relation_configs[0].rhs
            else:
                lhs_type = relation_configs[rel_id].lhs
                rhs_type = relation_configs[rel_id].rhs

            try:
                lhs_part, lhs_offset = \
                    entities_by_type[lhs_type].get_partition(words[lhs_col])
                rhs_part, rhs_offset = \
                    entities_by_type[rhs_type].get_partition(words[rhs_col])
            except KeyError:
                # Ignore edges whose entities are not known.
                skipped += 1
                continue

            buckets[lhs_part, rhs_part].append((lhs_offset, rhs_offset, rel_id))

            processed = processed + 1
            if processed % 100000 == 0:
                print("- Processed %d edges so far..." % processed)

    print("- Processed %d edges in total" % processed)
    if skipped > 0:
        print("- Skipped %d edges because their relation type or entities were "
              "unknown (either not given in the config or filtered out as too "
              "rare)." % skipped)

    for i in range(num_lhs_parts):
        for j in range(num_rhs_parts):
            print("- Writing bucket (%d, %d), containing %d edges..."
                  % (i, j, len(buckets[i, j])))
            edges = np.asarray(buckets[i, j])
            with h5py.File(os.path.join(
                edge_path_out, "edges_%d_%d.h5" % (i, j)
            ), "w") as hf:
                hf.attrs["format_version"] = 1
                hf.create_dataset("lhs", data=edges[:, 0])
                hf.create_dataset("rhs", data=edges[:, 1])
                hf.create_dataset("rel", data=edges[:, 2])


def convert_input_data(
    config: str,
    edge_paths: List[str],
    lhs_col: int,
    rhs_col: int,
    rel_col: Optional[int] = None,
    entity_min_count: int = 1,
    relation_type_min_count: int = 1,
):

    entity_configs, relation_configs, entity_path, dynamic_relations = \
        validate_config(config)

    os.makedirs(entity_path, exist_ok=True)

    relation_types = collect_relation_types(
        relation_configs,
        edge_paths,
        dynamic_relations,
        rel_col,
        relation_type_min_count,
    )

    entities_by_type = collect_entities_by_type(
        relation_types,
        entity_configs,
        relation_configs,
        edge_paths,
        dynamic_relations,
        lhs_col,
        rhs_col,
        rel_col,
        entity_min_count,
    )

    dump = {
        "relations": relation_types.get_list(),
        "entities": {k: v.get_list() for k, v in entities_by_type.items()},
    }
    with open(os.path.join(entity_path, "dictionary.json"), "wt") as tf:
        json.dump(dump, tf, indent=4)

    generate_entity_path_files(
        entity_path,
        entities_by_type,
        relation_types,
        dynamic_relations,
    )

    for edge_path in edge_paths:
        generate_edge_path_files(
            edge_path,
            entities_by_type,
            relation_types,
            relation_configs,
            dynamic_relations,
            lhs_col,
            rhs_col,
            rel_col,
        )


def validate_config(
    config: str,
) -> Tuple[Dict[str, EntitySchema], List[RelationSchema], str, bool]:
    user_config = get_config_dict_from_module(config)

    # validate entites and relations config
    entities_config = user_config.get("entities")
    relations_config = user_config.get("relations")
    entity_path = user_config.get("entity_path")
    dynamic_relations = user_config.get("dynamic_relations", False)
    if not isinstance(entities_config, dict):
        raise TypeError("Config entities is not of type dict")
    if not isinstance(relations_config, list):
        raise TypeError("Config relations is not of type list")
    if not isinstance(entity_path, str):
        raise TypeError("Config entity_path is not of type str")
    if not isinstance(dynamic_relations, bool):
        raise TypeError("Config dynamic_relations is not of type bool")

    entities = {}
    relations = []
    for entity, entity_config in entities_config.items():
        entities[entity] = EntitySchema.from_dict(entity_config)
    for relation in relations_config:
        relations.append(RelationSchema.from_dict(relation))

    return entities, relations, entity_path, dynamic_relations


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('edge_paths', nargs='*', help='Input file paths')
    parser.add_argument('-l', '--lhs-col', type=int, required=True,
                        help='Column index for source entity')
    parser.add_argument('-r', '--rhs-col', type=int, required=True,
                        help='Column index for target entity')
    parser.add_argument('--rel-col', type=int,
                        help='Column index for relation entity')
    parser.add_argument('--relation-type-min-count', type=int, default=1,
                        help='Min count for relation types')
    parser.add_argument('--entity-min-count', type=int, default=1,
                        help='Min count for entities')

    opt = parser.parse_args()

    convert_input_data(
        opt.config,
        opt.edge_paths,
        opt.lhs_col,
        opt.rhs_col,
        opt.rel_col,
        opt.entity_min_count,
        opt.relation_type_min_count,
    )


if __name__ == "__main__":
    main()
