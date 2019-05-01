#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from typing import Dict, List, Tuple


class Dictionary:

    def __init__(
        self,
        ix_to_word: List[str],
        *,
        num_parts: int = 1,
    ) -> None:
        self.ix_to_word: List[str] = ix_to_word
        self.word_to_ix: Dict[str, int] = {w: i for i, w in enumerate(ix_to_word)}
        self.num_parts = num_parts

    def get_id(self, word: str) -> int:
        return self.word_to_ix[word]

    def size(self) -> int:
        return len(self.ix_to_word)

    def part_size(self, part: int) -> int:
        if not 0 <= part < self.num_parts:
            raise ValueError("%d not in [0, %d)" % (part, self.num_parts))
        part_begin = (part * self.size() - 1) // self.num_parts + 1
        part_end = ((part + 1) * self.size() - 1) // self.num_parts
        return part_end - part_begin + 1

    def get_partition(self, word: str) -> Tuple[int, int]:
        idx = self.get_id(word)
        part = idx * self.num_parts // self.size()
        part_begin = (part * self.size() - 1) // self.num_parts + 1
        return part, idx - part_begin

    def get_list(self) -> List[str]:
        return self.ix_to_word
