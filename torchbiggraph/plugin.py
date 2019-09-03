#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from typing import Callable, Dict, Generic, Type, TypeVar
from urllib.parse import urlparse


T = TypeVar("T")


class PluginRegistry(Generic[T]):

    def __init__(self) -> None:
        self.registry: Dict[str, Type[T]] = {}

    def register(self, name: str, class_: Type[T]) -> None:
        reg_class = self.registry.setdefault(name, class_)
        if reg_class is not class_:
            raise RuntimeError(f"Attempting to re-register {name} "
                               f"which was already set to {reg_class!r}")

    def register_as(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(class_: Type[T]) -> Type[T]:
            self.register(name, class_)
            return class_
        return decorator

    def get_class(self, name: str) -> Type[T]:
        try:
            return self.registry[name]
        except KeyError:
            all_names = ", ".join(sorted(self.registry.keys()))
            raise NotImplementedError(
                f"Unknown name {name} (known names: {all_names})")


class URLPluginRegistry(PluginRegistry[T]):

    def make_instance(self, url: str) -> T:
        scheme = urlparse(url).scheme
        try:
            class_: Type[T] = self.get_class(scheme)
        except NotImplementedError as err:
            raise NotImplementedError(f"Unsupported URL {url}: {err}") from None
        return class_(url)
