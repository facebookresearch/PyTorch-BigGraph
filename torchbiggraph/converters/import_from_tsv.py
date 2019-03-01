#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import os
import os.path

import h5py
import numpy as np
import torch

from torchbiggraph.config import \
    ConfigSchema, EntitySchema, RelationSchema, parse_config_base
from torchbiggraph.converters.dictionary import Dictionary


def build_entity_dict(
    rdict,
    entities,
    relations,
    edge_paths,
    isDynamic,
    srcCol,
    destCol,
    relationCol,
    entityMinCount,
):

    print('Building dict for entites.')
    entity_dict = {}
    # initialize entity dict
    for entity, entity_config in entities.items():
        print(entity, entity_config)
        freq = entityMinCount
        npart = entity_config.num_partitions
        entity_dict[entity] = Dictionary(freq, 'ENTITY', entity, npart)

    # read entities in files to build entity dict
    for edgepath in edge_paths:
        print(edgepath)
        if isDynamic:
            cols = [srcCol, destCol]
            entity_list = list(entity_dict.keys())
            assert len(entity_list) == 1
            entity = entity_list[0]
            entity_dict[entity].add_from_file(edgepath, cols)
        else:
            with open(edgepath, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    src = parts[srcCol]
                    dest = parts[destCol]
                    if relationCol is not None:
                        rel = parts[relationCol]
                        rel_id = rdict.getId(rel)
                        assert rel_id > -1, "Could not find relation in config"
                    else:
                        rel_id = 0

                    src_type = relations[rel_id].lhs
                    dest_type = relations[rel_id].rhs

                    entity_dict[src_type].add(src)
                    entity_dict[dest_type].add(dest)

    # filter and shuffle entity dict
    for entity, _ in entities.items():
        entity_dict[entity].filter_and_shuffle()
        print('%s dict size : %d' % (entity, entity_dict[entity].size()))

    return entity_dict


def build_relation_dict(
    relations,
    edge_paths,
    isDynamic,
    relationCol,
    relationMinCount,
):

    print('Building dict for relation.')
    rdict = Dictionary(relationMinCount, 'RELATION', 'RELATION', 1)

    if isDynamic:
        assert relationCol is not None, \
            "Need to set relationCol in dynamic mode."
        for edgepath in edge_paths:
            rdict.add_from_file(edgepath, [relationCol])
        rdict.filter_and_shuffle()

    else:
        # construct the relation dict from relation config
        rel_names = [relation.name for relation in relations]
        rdict.build_from_list(rel_names)

    print('relation dict size : %d' % rdict.size())
    return rdict


def generate_entity_size_file(entities, entity_path, entity_dict):
    # save entity file
    for entity, entity_config in entities.items():
        npart = entity_config.num_partitions
        part_size = entity_dict[entity].part_size()

        for i in range(npart):
            with open(os.path.join(
                entity_path, "entity_count_%s_%d.txt" % (entity, i)
            ), "wt") as tf:
                tf.write("%d" % part_size)

    return


def convert_and_partition_data(
    edict,
    rdict,
    fname,
    relations,
    isDynamic,
    srcCol,
    destCol,
    relationCol
):

    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'

    print('Reading edges from %s , writing processed edges to %s' % (fname, out_dir))
    os.makedirs(out_dir, exist_ok=True)

    lhs_part = max(edict[rel.lhs].npart for rel in relations)
    rhs_part = max(edict[rel.rhs].npart for rel in relations)

    print('Partitioning edges into (%d, %d) parts.' % (lhs_part, rhs_part))

    # initialize buckets, which contains lhs, rhs, rel edges
    # in different partitions
    buckets = {}
    for i in range(lhs_part):
        for j in range(rhs_part):
            buckets[i, j] = [[], [], []]

    cnt = 0
    skipped = 0

    with io.open(fname, 'r') as f:
        for line in f:
            words = line.strip().split()
            if relationCol is None:
                # only has one relation
                e_rel = 0
            else:
                rel_name = words[relationCol]
                e_rel = rdict.getId(rel_name)

            # filter examples which contains relation that's not in dictionary
            if e_rel < 0:
                skipped += 1
                continue

            # detect lhs and rhs entitiy type from relation config
            if isDynamic:
                entity_list = list(edict.keys())
                assert len(entity_list) == 1
                lhs_ent = entity_list[0]
                rhs_ent = lhs_ent
            else:
                lhs_ent = relations[e_rel].lhs
                rhs_ent = relations[e_rel].rhs

            e_lhs = edict[lhs_ent].getId(words[srcCol])
            e_rhs = edict[rhs_ent].getId(words[destCol])

            # filter examples which contains entity that's not in dictionary
            if e_lhs < 0 or e_rhs < 0:
                skipped += 1
                continue

            # map the example to corresponding partitions

            l_part, l_offset = edict[lhs_ent].get_partition(e_lhs)
            r_part, r_offset = edict[rhs_ent].get_partition(e_rhs)

            buckets[l_part, r_part][0].append(l_offset)
            buckets[l_part, r_part][1].append(r_offset)
            buckets[l_part, r_part][2].append(e_rel)

            cnt = cnt + 1
            if cnt % 100000 == 0:
                print('Load ', cnt, ' examples.')

    print('Total number of examples : %d\tskipped: %d.' % (cnt, skipped))

    for i in range(lhs_part):
        for j in range(rhs_part):
            p_lhs = buckets[i, j][0]
            p_rhs = buckets[i, j][1]
            p_rel = buckets[i, j][2]

            print('Partition (%d, %d) contains %d edges.' % (i, j, len(p_lhs)))
            out_f = os.path.join(out_dir, "edges_%d_%d.h5" % (i, j))
            print("Saving edges to %s" % out_f)
            with h5py.File(out_f, "w") as hf:
                hf.attrs["format_version"] = 1
                hf.create_dataset("lhs", data=np.asarray(p_lhs))
                hf.create_dataset("rhs", data=np.asarray(p_rhs))
                hf.create_dataset("rel", data=np.asarray(p_rel))


def convert_input_data(
    config,
    edge_paths,
    isDynamic=0,
    srcCol=0,
    destCol=1,
    relationCol=None,
    entityMinCount=1,
    relationMinCount=1,
):

    entities, relations, entity_path = validate_config(config)

    os.makedirs(entity_path, exist_ok=True)

    rdict = build_relation_dict(
        relations,
        edge_paths,
        isDynamic,
        relationCol,
        relationMinCount
    )

    edict = build_entity_dict(
        rdict,
        entities,
        relations,
        edge_paths,
        isDynamic,
        srcCol,
        destCol,
        relationCol,
        entityMinCount
    )

    if isDynamic:
        with open(os.path.join(entity_path, "dynamic_rel_count.txt"), "wt") as tf:
            tf.write("%d" % rdict.size())

    edict_params = {k : v.get_params() for k, v in edict.items()}
    rdict_params = rdict.get_params()
    torch.save(
        (edict_params, rdict_params),
        os.path.join(entity_path, 'dict.pt')
    )

    generate_entity_size_file(entities, entity_path, edict)

    for edgepath in edge_paths:
        convert_and_partition_data(
            edict,
            rdict,
            edgepath,
            relations,
            isDynamic,
            srcCol,
            destCol,
            relationCol
        )


def validate_config(config):
    user_config = parse_config_base(config)

    # validate entites and relations config
    entities_config = user_config.get("entities")
    relations_config = user_config.get("relations")
    entity_path = user_config.get("entity_path")
    if not isinstance(entities_config, dict):
        raise TypeError("Config entities is not of type dict")
    if not isinstance(relations_config, list):
        raise TypeError("Config relations is not of type list")
    if not isinstance(entity_path, str):
        raise TypeError("Config entity_path is not of type str")

    entities = {}
    relations = []
    for entity, entity_config in entities_config.items():
        entities[entity] = EntitySchema.from_dict(entity_config)
    for relation in relations_config:
        relations.append(RelationSchema.from_dict(relation))

    return entities, relations, entity_path


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help='Path to config file')

    parser.add_argument('edge_paths', nargs='*', help='Input file paths')
    parser.add_argument('-r', '--relationCol', type=int,
                        help='Column index for relation entity')
    parser.add_argument('-s', '--srcCol', type=int, required=True,
                        help='Column index for source entity')
    parser.add_argument('-d', '--destCol', type=int, required=True,
                        help='Column index for target entity')
    parser.add_argument('--isDynamic', default=0,
                        help='whether to use dynamic mode')
    parser.add_argument('-rc', '--relationMinCount', default=1,
                        help='Min count for relation')
    parser.add_argument('-ec', '--entityMinCount', default=1,
                        help='Min count for entity')

    opt = parser.parse_args()

    convert_input_data(
        opt.config,
        opt.edge_paths,
        opt.isDynamic,
        opt.srcCol,
        opt.destCol,
        opt.relationCol,
        opt.entityMinCount,
        opt.relationMinCount
    )


if __name__ == "__main__":
    main()
