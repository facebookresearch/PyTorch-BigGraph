#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from torchbiggraph.plugin import URLPluginRegistry
from torchbiggraph.util import CouldNotLoadData


logger = logging.getLogger("torchbiggraph")


class AbstractEntityStorage(ABC):

    @abstractmethod
    def __init__(self, url: str) -> None:
        pass

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def has_count(self, entity_name: str, partition: int) -> bool:
        pass

    @abstractmethod
    def save_count(self, entity_name: str, partition: int, count: int) -> None:
        pass

    @abstractmethod
    def load_count(self, entity_name: str, partition: int) -> int:
        pass

    @abstractmethod
    def has_names(self, entity_name: str, partition: int) -> bool:
        pass

    @abstractmethod
    def save_names(self, entity_name: str, partition: int, names: List[str]) -> None:
        pass

    @abstractmethod
    def load_names(self, entity_name: str, partition: int) -> List[str]:
        pass


class AbstractRelationTypeStorage(ABC):

    @abstractmethod
    def __init__(self, url: str) -> None:
        pass

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def has_count(self) -> None:
        pass

    @abstractmethod
    def save_count(self, count: int) -> None:
        pass

    @abstractmethod
    def load_count(self) -> int:
        pass

    @abstractmethod
    def has_names(self) -> bool:
        pass

    @abstractmethod
    def save_names(self, names: List[str]) -> None:
        pass

    @abstractmethod
    def load_names(self) -> List[str]:
        pass


ENTITY_STORAGES = URLPluginRegistry[AbstractEntityStorage]()
RELATION_TYPE_STORAGES = URLPluginRegistry[AbstractRelationTypeStorage]()


def save_count(path: Path, count: int) -> None:
    with path.open("wt") as tf:
        tf.write(f"{count}\n")


def load_count(path: Path) -> int:
    try:
        with path.open("rt") as tf:
            return int(tf.read().strip())
    except FileNotFoundError as err:
        raise CouldNotLoadData() from err


def save_names(path: Path, names: List[str]) -> None:
    with path.open("wt") as tf:
        json.dump(names, tf, indent=4)


def load_names(path: Path) -> List[str]:
    try:
        with path.open("rt") as tf:
            return json.load(tf)
    except FileNotFoundError as err:
        raise CouldNotLoadData() from err


@ENTITY_STORAGES.register_as("")  # No scheme
@ENTITY_STORAGES.register_as("file")
class FileEntityStorage(AbstractEntityStorage):

    def __init__(self, path: str) -> None:
        if path.startswith("file://"):
            path = path[len("file://"):]
        self.path = Path(path).resolve(strict=False)

    def get_count_file(self, entity_name: str, partition: int) -> Path:
        return self.path / f"entity_count_{entity_name}_{partition}.txt"

    def get_names_file(self, entity_name: str, partition: int) -> Path:
        return self.path / f"entity_names_{entity_name}_{partition}.json"

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def has_count(self, entity_name: str, partition: int) -> bool:
        return self.get_count_file(entity_name, partition).is_file()

    def save_count(self, entity_name: str, partition: int, count: int) -> None:
        save_count(self.get_count_file(entity_name, partition), count)

    def load_count(self, entity_name: str, partition: int) -> int:
        return load_count(self.get_count_file(entity_name, partition))

    def has_names(self, entity_name: str, partition: int) -> bool:
        return self.get_names_file(entity_name, partition).is_file()

    def save_names(self, entity_name: str, partition: int, names: List[str]) -> None:
        save_names(self.get_names_file(entity_name, partition), names)

    def load_names(self, entity_name: str, partition: int) -> List[str]:
        return load_names(self.get_names_file(entity_name, partition))


@RELATION_TYPE_STORAGES.register_as("")  # No scheme
@RELATION_TYPE_STORAGES.register_as("file")
class FileRelationTypeStorage(AbstractRelationTypeStorage):

    def __init__(self, path: str) -> None:
        if path.startswith("file://"):
            path = path[len("file://"):]
        self.path = Path(path).resolve(strict=False)

    def get_count_file(self) -> Path:
        return self.path / "dynamic_rel_count.txt"

    def get_names_file(self) -> Path:
        return self.path / f"dynamic_rel_names.json"

    def prepare(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def has_count(self) -> None:
        return self.get_count_file().is_file()

    def save_count(self, count: int) -> None:
        save_count(self.get_count_file(), count)

    def load_count(self) -> int:
        return load_count(self.get_count_file())

    def has_names(self) -> bool:
        return self.get_names_file().is_file()

    def save_names(self, names: List[str]) -> None:
        save_names(self.get_names_file(), names)

    def load_names(self) -> List[str]:
        return load_names(self.get_names_file())
