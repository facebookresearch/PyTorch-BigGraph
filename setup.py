#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import os

from setuptools import setup
from torch.utils import cpp_extension


if __name__ == "__main__":
    if int(os.getenv("PBG_INSTALL_CPP", 1)) == 0:
        setup()
    else:
        setup(
            ext_modules=[
                cpp_extension.CppExtension(
                    "torchbiggraph._C", ["torchbiggraph/util.cpp"]
                )
            ],
            cmdclass={"build_ext": cpp_extension.BuildExtension},
        )
