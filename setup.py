#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

with open("README.md", "rt") as f:
    long_description = f.read()

with open("requirements.txt", "rt") as f:
    requirements = f.readlines()

setup(
    name="torchbiggraph",
    version="1.dev",
    description="A distributed system to learn embeddings of large graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/PyTorch-BigGraph",
    author="Facebook AI Research",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning knowledge-base graph-embedding link-prediction",
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={
        "torchbiggraph.examples": [
            "configs/*.py",
        ],
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "torchbiggraph_config=torchbiggraph.config:main",
            "torchbiggraph_eval=torchbiggraph.eval:main",
            "torchbiggraph_example_fb15k=torchbiggraph.examples.fb15k:main",
            "torchbiggraph_example_livejournal=torchbiggraph.examples.livejournal:main",
            "torchbiggraph_export_to_tsv=torchbiggraph.converters.export_to_tsv:main",
            "torchbiggraph_import_from_tsv=torchbiggraph.converters.import_from_tsv:main",
            "torchbiggraph_partitionserver=torchbiggraph.partitionserver:main",
            "torchbiggraph_train=torchbiggraph.train:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/facebookresearch/PyTorch-BigGraph/issues",
        "Source": "https://github.com/facebookresearch/PyTorch-BigGraph",
    },
)
