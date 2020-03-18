#!/usr/bin/env python3

import pkg_resources


__version__ = (
    pkg_resources.resource_string("torchbiggraph", "VERSION.txt")
    .decode("utf-8")
    .strip()
)
