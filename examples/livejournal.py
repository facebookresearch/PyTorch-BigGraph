#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
from itertools import chain

import attr

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.contrib.data_processor import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train


URL = 'https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz'
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}
TRAIN_FRACTION = 0.75


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


def random_split_file(fpath):
    print('Shuffling and spliting train/test file. This may take a while.')
    root = os.path.dirname(fpath)
    train_file = os.path.join(root, FILENAMES['train'])
    test_file = os.path.join(root, FILENAMES['test'])

    print('Reading data from file: ', fpath)
    with open(fpath, 'r') as f:
        lines = f.readlines()

    # The first few lines are comments
    lines = lines[4:]
    print('Shuffling data')
    random.shuffle(lines)
    split_len = int(len(lines) * TRAIN_FRACTION)

    print('Spliting to train and test file.')
    with open(train_file, 'w') as fo_train:
        fo_train.write(''.join(lines[:split_len]))

    with open(test_file, 'w') as fo_test:
        fo_test.write(''.join(lines[split_len:]))


def main():
    parser = argparse.ArgumentParser(description='Example on Livejournal')
    parser.add_argument('--config',
                        default='examples/configs/livejournal_config.py',
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
    os.makedirs(data_dir, exist_ok=True)
    fpath = utils.download_url(URL, data_dir)
    fpath = utils.extract_gzip(fpath)
    print('Downloaded and extracted file.')

    # random split file for train and test
    random_split_file(fpath)

    edge_paths = [os.path.join(data_dir, name) for name in FILENAMES.values()]
    convert_input_data(
        args.config,
        edge_paths,
        isDynamic=0,
        srcCol=0,
        destCol=1,
        relationCol=None,
    )
    config = parse_config(args.config, overrides)

    trainPath = [convert_path(os.path.join(data_dir, FILENAMES['train']))]
    train_config = attr.evolve(config, edge_paths=trainPath)

    train(train_config)

    evalPath = [convert_path(os.path.join(data_dir, FILENAMES['test']))]
    eval_config = attr.evolve(config, edge_paths=evalPath)

    do_eval(eval_config)


if __name__ == "__main__":
    main()
