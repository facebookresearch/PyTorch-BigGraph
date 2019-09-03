#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Type
from urllib.parse import urlparse

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


ENTITY_COUNT_READERS: Dict[str, Type[AbstractEntityCountReader]] = {}


def register_entity_count_reader_for_scheme(
    scheme: str,
) -> Callable[[Type[AbstractEntityCountReader]], Type[AbstractEntityCountReader]]:
    def decorator(
        class_: Type[AbstractEntityCountReader],
    ) -> Type[AbstractEntityCountReader]:
        reg_class = ENTITY_COUNT_READERS.setdefault(scheme, class_)
        if reg_class is not class_:
            raise RuntimeError(
                f"Attempting to re-register an entity count reader for scheme "
                f"{scheme} which was already set to {reg_class!r}")
        return class_
    return decorator


def get_entity_count_reader_for_url(url: str) -> AbstractEntityCountReader:
    scheme = urlparse(url).scheme
    try:
        class_: Type[AbstractEntityCountReader] = ENTITY_COUNT_READERS[scheme]
    except LookupError:
        raise RuntimeError(f"Couldn't find any edgelist reader "
                           f"for scheme {scheme} used by {url}")
    reader = class_(url)
    return reader


# Names and values of metadata attributes for the HDF5 files.
FORMAT_VERSION_ATTR = "format_version"
FORMAT_VERSION = 1


@register_entity_count_reader_for_scheme("")  # No scheme
@register_entity_count_reader_for_scheme("file")
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
