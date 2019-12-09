#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import random
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Tuple, Union

import torch

from torchbiggraph.config import (
    ConfigFileLoader,
    ConfigSchema,
    EntitySchema,
    RelationSchema,
    override_config_dict,
)
from torchbiggraph.converters.dictionary import Dictionary
from torchbiggraph.converters.utils import (
    EdgelistReader,
    TSVEdgelistReader,
    ParquetEdgelistReader
)
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.graph_storages import (
    AbstractEdgeAppender,
    AbstractEdgeStorage,
    AbstractEntityStorage,
    AbstractRelationTypeStorage,
    EDGE_STORAGES,
    ENTITY_STORAGES,
    RELATION_TYPE_STORAGES,
)


def collect_relation_types(
    relation_configs: List[RelationSchema],
    edge_paths: List[Path],
    dynamic_relations: bool,
    edgelist_reader: EdgelistReader,
    relation_type_min_count: int,
) -> Dictionary:

    if dynamic_relations:
        if edgelist_reader.rel_col is None:
            raise RuntimeError("Need to specify rel_col in dynamic mode.")
        print("Looking up relation types in the edge files...")
        counter: Counter[str] = Counter()
        for edgepath in edge_paths:
            for _lhs_word, _rhs_word, rel_word in edgelist_reader.read(edgepath):
                counter[rel_word] += 1
        print(f"- Found {len(counter)} relation types")
        if relation_type_min_count > 0:
            print(f"- Removing the ones with fewer than {relation_type_min_count} occurrences...")
            counter = Counter({k: c for k, c in counter.items()
                               if c >= relation_type_min_count})
            print(f"- Left with {len(counter)} relation types")
        print("- Shuffling them...")
        names = list(counter.keys())
        random.shuffle(names)

    else:
        names = [rconfig.name for rconfig in relation_configs]
        print(f"Using the {len(names)} relation types given in the config")

    return Dictionary(names)


def collect_entities_by_type(
    relation_types: Dictionary,
    entity_configs: Dict[str, EntitySchema],
    relation_configs: List[RelationSchema],
    edge_paths: List[Path],
    dynamic_relations: bool,
    edgelist_reader: EdgelistReader,
    entity_min_count: int,
) -> Dict[str, Dictionary]:

    counters: Dict[str, Counter[str]] = {}
    for entity_name in entity_configs.keys():
        counters[entity_name] = Counter()

    print("Searching for the entities in the edge files...")
    for edgepath in edge_paths:
        for lhs_word, rhs_word, rel_word in edgelist_reader.read(edgepath):
            if dynamic_relations or rel_word is None:
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
        print(f"Entity type {entity_name}:")
        print(f"- Found {len(counter)} entities")
        if entity_min_count > 0:
            print(f"- Removing the ones with fewer than {entity_min_count} occurrences...")
            counter = Counter({k: c for k, c in counter.items()
                               if c >= entity_min_count})
            print(f"- Left with {len(counter)} entities")
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
    edge_file_in: Path,
    edge_path_out: Path,
    edge_storage: AbstractEdgeStorage,
    entities_by_type: Dict[str, Dictionary],
    relation_types: Dictionary,
    relation_configs: List[RelationSchema],
    dynamic_relations: bool,
    edgelist_reader: EdgelistReader,
) -> None:
    print(f"Preparing edge path {edge_path_out}, "
          f"out of the edges found in {edge_file_in}")
    edge_storage.prepare()

    num_lhs_parts = max(entities_by_type[rconfig.lhs].num_parts
                        for rconfig in relation_configs)
    num_rhs_parts = max(entities_by_type[rconfig.rhs].num_parts
                        for rconfig in relation_configs)

    print(f"- Edges will be partitioned in {num_lhs_parts} x {num_rhs_parts} buckets.")

    processed = 0
    skipped = 0

    # We use an ExitStack in order to close the dynamically-created edge appenders.
    with ExitStack() as appender_stack:
        appenders: Dict[Tuple[int, int], AbstractEdgeAppender] = {}
        for lhs_word, rhs_word, rel_word in edgelist_reader.read(edge_file_in):
            if rel_word is None:
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

            if (lhs_part, rhs_part) not in appenders:
                appenders[lhs_part, rhs_part] = appender_stack.enter_context(
                    edge_storage.save_edges_by_appending(lhs_part, rhs_part))
            appenders[lhs_part, rhs_part].append_edges(EdgeList(
                EntityList.from_tensor(torch.tensor([lhs_offset], dtype=torch.long)),
                EntityList.from_tensor(torch.tensor([rhs_offset], dtype=torch.long)),
                torch.tensor([rel_id], dtype=torch.long),
            ))

            processed = processed + 1
            if processed % 100000 == 0:
                print(f"- Processed {processed} edges so far...")

    print(f"- Processed {processed} edges in total")
    if skipped > 0:
        print(f"- Skipped {skipped} edges because their relation type or "
              f"entities were unknown (either not given in the config or "
              f"filtered out as too rare).")


def convert_input_data(
    entity_configs: Dict[str, EntitySchema],
    relation_configs: List[RelationSchema],
    entity_path: str,
    edge_paths_out: List[str],
    edge_paths_in: List[Path],
    edgelist_reader: EdgelistReader,
    edgelist_format: str = "tsv",
    entity_min_count: int = 1,
    relation_type_min_count: int = 1,
    dynamic_relations: bool = False,
) -> None:
    if len(edge_paths_in) != len(edge_paths_out):
        raise ValueError(
            f"The edge paths passed as inputs ({edge_paths_in}) don't match "
            f"the ones specified as outputs ({edge_paths_out})")

    entity_storage = ENTITY_STORAGES.make_instance(entity_path)
    relation_type_storage = RELATION_TYPE_STORAGES.make_instance(entity_path)
    edge_storages = [EDGE_STORAGES.make_instance(ep) for ep in edge_paths_out]

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
        edge_storage.has_edges(0, 0) for edge_storage in edge_storages)

    if all(some_files_exists):
        print("Found some files that indicate that the input data "
              "has already been preprocessed, not doing it again.")
        all_paths = ", ".join(str(p) for p in [entity_path] + edge_paths_out)
        print(f"These files are in: {all_paths}")
        return

    relation_types = collect_relation_types(
        relation_configs,
        edge_paths_in,
        dynamic_relations,
        edgelist_reader,
        relation_type_min_count,
    )

    entities_by_type = collect_entities_by_type(
        relation_types,
        entity_configs,
        relation_configs,
        edge_paths_in,
        dynamic_relations,
        edgelist_reader,
        entity_min_count,
    )

    generate_entity_path_files(
        entity_storage,
        entities_by_type,
        relation_type_storage,
        relation_types,
        dynamic_relations,
    )

    for edge_path_in, edge_path_out, edge_storage \
            in zip(edge_paths_in, edge_paths_out, edge_storages):
        generate_edge_path_files(
            edge_path_in,
            edge_path_out,
            edge_storage,
            entities_by_type,
            relation_types,
            relation_configs,
            dynamic_relations,
            edgelist_reader,
        )


def parse_config_partial(
    config_dict: Any,
) -> Tuple[Dict[str, EntitySchema], List[RelationSchema], str, bool]:
    entities_config = config_dict.get("entities")
    relations_config = config_dict.get("relations")
    entity_path = config_dict.get("entity_path")
    edge_paths = config_dict.get("edge_paths")
    dynamic_relations = config_dict.get("dynamic_relations", False)
    if not isinstance(entities_config, dict):
        raise TypeError("Config entities is not of type dict")
    if any(not isinstance(k, str) for k in entities_config.keys()):
        raise TypeError("Config entities has some keys that are not of type str")
    if not isinstance(relations_config, list):
        raise TypeError("Config relations is not of type list")
    if not isinstance(entity_path, str):
        raise TypeError("Config entity_path is not of type str")
    if not isinstance(edge_paths, list):
        raise TypeError("Config edge_paths is not of type list")
    if any(not isinstance(p, str) for p in edge_paths):
        raise TypeError("Config edge_paths has some items that are not of type str")
    if not isinstance(dynamic_relations, bool):
        raise TypeError("Config dynamic_relations is not of type bool")

    entities = {}
    relations = []
    for entity, entity_config in entities_config.items():
        entities[entity] = EntitySchema.from_dict(entity_config)
    for relation in relations_config:
        relations.append(RelationSchema.from_dict(relation))

    return entities, relations, entity_path, edge_paths, dynamic_relations


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('edge_paths', type=Path, nargs='*', help='Input file paths')
    parser.add_argument('-l', '--lhs-col', type=int, required=True,
                        help='Column index for source entity')
    parser.add_argument('-r', '--rhs-col', type=int, required=True,
                        help='Column index for target entity')
    parser.add_argument('--rel-col', type=int,
                        help='Column index for relation entity')
    parser.add_argument('--edgelist-format', type=str, default='tsv',
                        help='Edgelist format [tsv|parquet]')
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

    entity_configs, relation_configs, entity_path, edge_paths, dynamic_relations = \
        parse_config_partial(config_dict)

    if len(opt.edge_paths) != len(edge_paths):
        print(f"The edge paths provided on the command line ({opt.edge_paths}) "
              f"don't match the ones found in the config file ({edge_paths})")

    if opt.edgelist_format == "tsv":
        edgelist_reader = TSVEdgelistReader(opt.lhs_col, opt.rhs_col, opt.rel_col)
    elif opt.edgelist_format == "parquet":
        edgelist_reader = ParquetEdgelistReader(opt.lhs_col, opt.rhs_col, opt.rel_col)
    else:
        raise RuntimeError(f"Unknown edgelist format: {opt.edgelist_format}")

    convert_input_data(
        entity_configs,
        relation_configs,
        entity_path,
        edge_paths,
        opt.edge_paths,
        edgelist_reader,
        opt.entity_min_count,
        opt.relation_type_min_count,
        dynamic_relations,
    )


if __name__ == "__main__":
    main()
