#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from torchbiggraph.plugin import URLPluginRegistry
from torchbiggraph.types import EntityName, Partition


logger = logging.getLogger("torchbiggraph")


class AbstractEntityCountReader(ABC):

    @abstractmethod
    def __init__(self, url: str) -> None:
        pass

    @abstractmethod
    def read_entity_count(
        self,
        entity_name: EntityName,
        partition: Partition,
    ) -> int:
        pass


ENTITY_COUNT_READERS = URLPluginRegistry[AbstractEntityCountReader]()


# Names and values of metadata attributes for the HDF5 files.
FORMAT_VERSION_ATTR = "format_version"
FORMAT_VERSION = 1


@ENTITY_COUNT_READERS.register_as("")  # No scheme
@ENTITY_COUNT_READERS.register_as("file")
class FileEntityCountReader(AbstractEntityCountReader):

    def __init__(self, path: str) -> None:
        if path.startswith("file://"):
            path = path[len("file://"):]
        self.path = Path(path).resolve(strict=False)
        if not self.path.is_dir():
            raise RuntimeError(f"Invalid entity count dir: {self.path}")

    def read_entity_count(
        self,
        entity_name: EntityName,
        partition: Partition,
    ) -> int:
        file_path = self.path / f"entity_count_{entity_name}_{partition}.txt"
        with file_path.open("rt") as tf:
            return int(tf.read().strip())
