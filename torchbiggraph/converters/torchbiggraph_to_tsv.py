#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch

from torchbiggraph.config import ConfigSchema
from torchbiggraph.converters.dictionary import Dictionary
from torchbiggraph.fileio import CheckpointManager
from torchbiggraph.model import make_model


def write(outf, word, emb):
    outf.write("%s\t%s\n" % (word, "\t".join(str.format("%.9f") % x for x in emb)))


def make_dict(params):
    if params is None:
        return None

    return Dictionary(
        params["freq"],
        params["dict_type"],
        params["entity_type"],
        params["npart"],
        params["ix_to_word"]
    )


def make_tsv(checkpoint, dictfile, outfile):
    print('Load entity and relation dictionary.')
    edict_params, rdict_params = torch.load(dictfile)
    edict = {}
    for entity, params in edict_params.items():
        edict[entity] = make_dict(params)

    rdict = make_dict(rdict_params)

    print('Load model check point.')
    checkpoint_manager = CheckpointManager(checkpoint)
    config, state_dict, _ = checkpoint_manager.read_metadata()

    model = make_model(config)
    if state_dict is not None:
        print('Init model from state dict.')
        model.load_state_dict(state_dict, strict=False)

    print('Done init model.')
    data = model.get_relation_parameters()

    entities = config.entities.keys()
    embeddings = {}
    for entity in entities:
        parts = range(config.entities[entity].num_partitions)
        size = edict[entity].size()
        embeddings[entity] = torch.FloatTensor(size, config.dimension)
        idx = 0
        for part in parts:
            print('Load embeddings for entity %s, part %d.' % (entity, part))
            embs, _ = checkpoint_manager.read(entity, part)
            # remove dummy embeddings due to partition
            sz = min(len(embs), size - idx)
            embs = embs[:sz]

            if config.global_emb:
                print("Applying the global embedding...")
                # FIXME: it's pretty hacky to hardcode this ...
                global_emb = state_dict['global_embs.emb_%s' % entity]
                embs += global_emb

            embeddings[entity][idx:idx + sz] = embs
            idx = idx + sz

    print('Writing to output file..')
    with open(outfile, 'w') as outf:
        for entity in entities:
            embs = embeddings[entity]
            dt = edict[entity]
            assert dt.size() == len(embs)
            for ix, word in dt.ix_to_word.items():
                write(outf, word, embs[ix])

        if model.num_dynamic_rels > 0:
            rels_lhs, rels_rhs = data
            for ix, rel in rdict.ix_to_word.items():
                write(outf, rel, rels_lhs[ix])

            for ix, rel in rdict.ix_to_word.items():
                write(outf, rel + "_reverse_relation", rels_rhs[ix])
        else:
            assert len(model.relations) == len(data)
            for i in range(len(data)):
                write(outf, model.relations[i].name, data[i])

    print('Done converting to file %s.' % outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Data for Filament2')
    parser.add_argument('--checkpoint')
    parser.add_argument('--dict', required=True)
    parser.add_argument('--out', required=True)

    args = parser.parse_args()

    make_tsv(args.checkpoint, args.dict, args.out)
