#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
from pathlib import Path

import attr
import pkg_resources
import torch
from torchbiggraph.config import add_to_sys_path, ConfigFileLoader
from torchbiggraph.converters.importers import convert_input_data, TSVEdgelistReader
from torchbiggraph.converters.utils import download_url, extract_tar
from torchbiggraph.eval import do_eval
from torchbiggraph.filtered_eval import FilteredRankingEvaluator
from torchbiggraph.train import train
from torchbiggraph.util import (
    set_logging_verbosity,
    setup_logging,
    SubprocessInitializer,
)


FB15K_URL = "https://dl.fbaipublicfiles.com/starspace/fb15k.tgz"
FILENAMES = [
    "FB15k/freebase_mtr100_mte100-train.txt",
    "FB15k/freebase_mtr100_mte100-valid.txt",
    "FB15k/freebase_mtr100_mte100-test.txt",
]

# Figure out the path where the sample config was installed by the package manager.
# This can be overridden with --config.
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEFAULT_CONFIG = pkg_resources.resource_filename(
        "torchbiggraph.examples", "configs/fb15k_config_gpu.py"
    )
else:
    DEFAULT_CONFIG = pkg_resources.resource_filename(
        "torchbiggraph.examples", "configs/fb15k_config_cpu.py"
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Example on FB15k")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument(
        "--data_dir", type=Path, default="data", help="where to save processed data"
    )
    parser.add_argument(
        "--no-filtered",
        dest="filtered",
        action="store_false",
        help="Run unfiltered eval",
    )
    args = parser.parse_args()

    # download data
    data_dir = args.data_dir
    fpath = download_url(FB15K_URL, data_dir)
    extract_tar(fpath)
    print("Downloaded and extracted file.")

    loader = ConfigFileLoader()
    config = loader.load_config(args.config, args.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    input_edge_paths = [data_dir / name for name in FILENAMES]
    output_train_path, output_valid_path, output_test_path = config.edge_paths

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    train_config = attr.evolve(config, edge_paths=[output_train_path])
    train(train_config, subprocess_init=subprocess_init)

    relations = [attr.evolve(r, all_negs=True) for r in config.relations]
    eval_config = attr.evolve(
        config, edge_paths=[output_test_path], relations=relations, num_uniform_negs=0
    )
    if args.filtered:
        filter_paths = [output_test_path, output_valid_path, output_train_path]
        do_eval(
            eval_config,
            evaluator=FilteredRankingEvaluator(eval_config, filter_paths),
            subprocess_init=subprocess_init,
        )
    else:
        do_eval(eval_config, subprocess_init=subprocess_init)


if __name__ == "__main__":
    main()
