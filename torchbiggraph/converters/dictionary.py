#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import math
from typing import Dict, List, Tuple


class Dictionary:
    def __init__(self, ix_to_word: List[str], *, num_parts: int = 1) -> None:
        self.ix_to_word: List[str] = ix_to_word
        self.word_to_ix: Dict[str, int] = {w: i for i, w in enumerate(ix_to_word)}
        self.num_parts = num_parts

    def get_id(self, word: str) -> int:
        return self.word_to_ix[word]

    def size(self) -> int:
        return len(self.ix_to_word)

    def get_list(self) -> List[str]:
        return self.ix_to_word

    def part_start(self, part: int) -> int:
        return math.ceil(part / self.num_parts * self.size())

    def part_end(self, part: int) -> int:
        return self.part_start(part + 1)

    def part_size(self, part: int) -> int:
        if not 0 <= part < self.num_parts:
            raise ValueError(f"{part} not in [0, {self.num_parts})")
        return self.part_end(part) - self.part_start(part)

    def get_partition(self, word: str) -> Tuple[int, int]:
        idx = self.get_id(word)
        part = math.floor(idx / self.size() * self.num_parts)
        assert self.part_start(part) <= idx < self.part_end(part)
        return part, idx - self.part_start(part)

    def get_part_list(self, part: int) -> List[str]:
        if not 0 <= part < self.num_parts:
            raise ValueError(f"{part} not in [0, {self.num_parts})")
        return self.ix_to_word[self.part_start(part) : self.part_end(part)]
