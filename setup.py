#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setuptools.setup(
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning knowledge-base graph-embedding link-prediction",
    packages=[
        "torchbiggraph",
        "torch_extensions.rpc",
        "torch_extensions.tensorlist",
    ],
    package_dir={
        "torchbiggraph": "torchbiggraph",
        "torch_extensions": "torch_extensions",
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "torchbiggraph_config=torchbiggraph.config:main",
            "torchbiggraph_eval=torchbiggraph.eval:main",
            "torchbiggraph_partitionserver=torchbiggraph.partitionserver:main",
            "torchbiggraph_train=torchbiggraph.train:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/facebookresearch/PyTorch-BigGraph/issues",
        "Source": "https://github.com/facebookresearch/PyTorch-BigGraph",
    },
)
