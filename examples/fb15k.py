#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from itertools import chain

import attr

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.converters.data_processor import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train


FB15K_URL = 'https://dl.fbaipublicfiles.com/starspace/fb15k.tgz'
FILENAMES = {
    'train': 'FB15k/freebase_mtr100_mte100-train.txt',
    'valid': 'FB15k/freebase_mtr100_mte100-valid.txt',
    'test': 'FB15k/freebase_mtr100_mte100-test.txt',
}


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Example on FB15k')
    parser.add_argument('--config', default='examples/configs/fb15k_config.py',
                        help='Path to config file')
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--data_dir', default='data',
                        help='where to save processed data')

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

    edgePaths = [os.path.join(data_dir, name) for name in FILENAMES.values()]
    convert_input_data(
        args.config,
        edgePaths,
        isDynamic=1,
        srcCol=0,
        destCol=2,
        relationCol=1,
    )

    config = parse_config(args.config, overrides)

    trainPath = [convert_path(os.path.join(data_dir, FILENAMES['train']))]
    train_config = attr.evolve(config, edgePaths=trainPath)

    train(train_config)

    evalPath = [convert_path(os.path.join(data_dir, FILENAMES['test']))]
    relations = [attr.evolve(r, all_rhs_negs=True) for r in config.relations]
    eval_config = attr.evolve(config, edgePaths=evalPath, relations=relations)

    do_eval(eval_config)


if __name__ == "__main__":
    main()
