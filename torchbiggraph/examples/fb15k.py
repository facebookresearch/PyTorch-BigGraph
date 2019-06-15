#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import os
from itertools import chain

import attr
import pkg_resources

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.filtered_eval import FilteredRankingEvaluator
from torchbiggraph.train import train

FB15K_URL = 'https://dl.fbaipublicfiles.com/starspace/fb15k.tgz'
FILENAMES = {
    'train': 'FB15k/freebase_mtr100_mte100-train.txt',
    'valid': 'FB15k/freebase_mtr100_mte100-valid.txt',
    'test': 'FB15k/freebase_mtr100_mte100-test.txt',
}

# Figure out the path where the sample config was installed by the package manager.
# This can be overridden with --config.
DEFAULT_CONFIG = pkg_resources.resource_filename("torchbiggraph.examples",
                                                 "configs/fb15k_config.py")


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Example on FB15k')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='Path to config file')
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--data_dir', default='data',
                        help='where to save processed data')
    parser.add_argument('--no-filtered', dest='filtered', action='store_false',
                        help='Run unfiltered eval')
    args = parser.parse_args()

    if args.param is not None:
        overrides = chain.from_iterable(args.param)  # flatten
    else:
        overrides = None

    # download data
    data_dir = args.data_dir
    fpath = utils.download_url(FB15K_URL, data_dir)
    utils.extract_tar(fpath)
    print('Downloaded and extracted file.')

    edge_paths = [os.path.join(data_dir, name) for name in FILENAMES.values()]
    convert_input_data(
        args.config,
        edge_paths,
        lhs_col=0,
        rhs_col=2,
        rel_col=1,
    )

    config = parse_config(args.config, overrides)

    train_path = [convert_path(os.path.join(data_dir, FILENAMES['train']))]
    train_config = attr.evolve(config, edge_paths=train_path)

    train(train_config)

    eval_path = [convert_path(os.path.join(data_dir, FILENAMES['test']))]
    relations = [attr.evolve(r, all_negs=True) for r in config.relations]
    eval_config = attr.evolve(config, edge_paths=eval_path, relations=relations, num_uniform_negs=0)
    if args.filtered:
        filter_paths = [
            convert_path(os.path.join(data_dir, FILENAMES['test'])),
            convert_path(os.path.join(data_dir, FILENAMES['valid'])),
            convert_path(os.path.join(data_dir, FILENAMES['train'])),
        ]
        do_eval(eval_config,
                evaluator=FilteredRankingEvaluator(eval_config, filter_paths))
    else:
        do_eval(eval_config)


if __name__ == "__main__":
    main()
