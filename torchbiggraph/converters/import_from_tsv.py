#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import os
import os.path
import random
from itertools import chain
from typing import Any, Counter, DefaultDict, Dict, List, Optional, Tuple

import h5py
import numpy as np

from torchbiggraph.config import (
    ConfigFileLoader,
    ConfigSchema,
    EntitySchema,
    RelationSchema,
    override_config_dict,
)
from torchbiggraph.converters.dictionary import Dictionary
from torchbiggraph.graph_storages import (
    AbstractEntityStorage,
    AbstractRelationTypeStorage,
    ENTITY_STORAGES,
    RELATION_TYPE_STORAGES,
)


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
                for line_num, line in enumerate(tf, start=1):
                    words = line.split()
                    try:
                        rel_word = words[rel_col]
                    except IndexError:
                        raise RuntimeError(
                            "Line %d of %s has only %d words"
                            % (line_num, edgepath, len(words))) from None
                    counter[rel_word] += 1
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
            for line_num, line in enumerate(tf, start=1):
                words = line.split()
                try:
                    lhs_word = words[lhs_col]
                    rhs_word = words[rhs_col]
                    rel_word = words[rel_col] if rel_col is not None else None
                except IndexError:
                    raise RuntimeError(
                        "Line %d of %s has only %d words"
                        % (line_num, edgepath, len(words))) from None

                if dynamic_relations or rel_col is None:
                    rel_id = 0
                else:
                    try:
                        rel_id = relation_types.get_id(rel_word)
                    except KeyError:
                        raise RuntimeError("Could not find relation type in config")

                counters[relation_configs[rel_id].lhs][lhs_word] += 1
                counters[relation_configs[rel_id].rhs][rhs_word] += 1

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
    entity_storage: AbstractEntityStorage,
    entities_by_type: Dict[str, Dictionary],
    relation_type_storage: AbstractRelationTypeStorage,
    relation_types: Dictionary,
    dynamic_relations: bool,
) -> None:
    print(f"Preparing counts and dictionaries for entities and relation types:")
    entity_storage.prepare()
    relation_type_storage.prepare()

    for entity_name, entities in entities_by_type.items():
        for part in range(entities.num_parts):
            print(f"- Writing count of entity type {entity_name} "
                  f"and partition {part}")
            entity_storage.save_count(entity_name, part, entities.part_size(part))
            entity_storage.save_names(entity_name, part, entities.get_part_list(part))

    if dynamic_relations:
        print("- Writing count of dynamic relations")
        relation_type_storage.save_count(relation_types.size())
        relation_type_storage.save_names(relation_types.get_list())


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
        for line_num, line in enumerate(tf, start=1):
            words = line.split()
            try:
                lhs_word = words[lhs_col]
                rhs_word = words[rhs_col]
                rel_word = words[rel_col] if rel_col is not None else None
            except IndexError:
                raise RuntimeError(
                    "Line %d of %s has only %d words"
                    % (line_num, edge_file_in, len(words))) from None

            if rel_col is None:
                rel_id = 0
            else:
                try:
                    rel_id = relation_types.get_id(rel_word)
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
                    entities_by_type[lhs_type].get_partition(lhs_word)
                rhs_part, rhs_offset = \
                    entities_by_type[rhs_type].get_partition(rhs_word)
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
            edges = np.array(buckets[i, j], dtype=np.int64).reshape((-1, 3))
            with h5py.File(os.path.join(
                edge_path_out, "edges_%d_%d.h5" % (i, j)
            ), "w") as hf:
                hf.attrs["format_version"] = 1
                hf.create_dataset("lhs", data=edges[:, 0])
                hf.create_dataset("rhs", data=edges[:, 1])
                hf.create_dataset("rel", data=edges[:, 2])


def convert_input_data(
    entity_configs: Dict[str, EntitySchema],
    relation_configs: List[RelationSchema],
    entity_path: str,
    edge_paths: List[str],
    lhs_col: int,
    rhs_col: int,
    rel_col: Optional[int] = None,
    entity_min_count: int = 1,
    relation_type_min_count: int = 1,
    dynamic_relations: bool = False,
) -> None:
    entity_storage = ENTITY_STORAGES.make_instance(entity_path)
    relation_type_storage = RELATION_TYPE_STORAGES.make_instance(entity_path)

    some_files_exists = []
    some_files_exists.extend(
        entity_storage.has_count(entity_name, partition)
        for entity_name, entity_config in entity_configs.items()
        for partition in range(entity_config.num_partitions))
    some_files_exists.extend(
        entity_storage.has_names(entity_name, partition)
        for entity_name, entity_config in entity_configs.items()
        for partition in range(entity_config.num_partitions))
    if dynamic_relations:
        some_files_exists.append(relation_type_storage.has_count())
        some_files_exists.append(relation_type_storage.has_names())
    some_files_exists.extend(
        os.path.exists(os.path.join(os.path.splitext(edge_file)[0] + "_partitioned", "edges_0_0.h5"))
        for edge_file in edge_paths)

    if all(some_files_exists):
        print("Found some files that indicate that the input data "
              "has already been preprocessed, not doing it again.")
        print(f"These files are in {entity_path} and {edge_paths}")
        return

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

    generate_entity_path_files(
        entity_storage,
        entities_by_type,
        relation_type_storage,
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


def parse_config_partial(
    config_dict: Any,
) -> Tuple[Dict[str, EntitySchema], List[RelationSchema], str, bool]:
    entities_config = config_dict.get("entities")
    relations_config = config_dict.get("relations")
    entity_path = config_dict.get("entity_path")
    dynamic_relations = config_dict.get("dynamic_relations", False)
    if not isinstance(entities_config, dict):
        raise TypeError("Config entities is not of type dict")
    if any(not isinstance(k, str) for k in entities_config.keys()):
        raise TypeError("Config entities has some keys that are not of type str")
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
    parser.add_argument('-p', '--param', action='append', nargs='*')
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

    loader = ConfigFileLoader()
    config_dict = loader.load_raw_config(opt.config)

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
        config_dict = override_config_dict(config_dict, overrides)

    entity_configs, relation_configs, entity_path, dynamic_relations = \
        parse_config_partial(config_dict)

    convert_input_data(
        entity_configs,
        relation_configs,
        entity_path,
        opt.edge_paths,
        opt.lhs_col,
        opt.rhs_col,
        opt.rel_col,
        opt.entity_min_count,
        opt.relation_type_min_count,
        dynamic_relations,
    )


if __name__ == "__main__":
    main()
