#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import torch


class Dictionary:
    def __init__(self, freq, dict_type, entity_name, npart, ix_to_word=None):
        self.s = Counter()
        self.freq = freq
        self.dict_type = dict_type
        self.entity_type = entity_name
        self.npart = npart
        self.ix_to_word = ix_to_word

    def add(self, word):
        self.s[word] += 1

    def add_from_file(self, fname, cols):
        print('Adding entities in file %s.' % fname)
        with open(fname, 'r') as f:
            for line in f.readlines():
                for i, col in enumerate(cols):
                    col = cols[i]
                    words = line.split()
                    word = words[col]
                    self.add(word)

    def build_from_list(self, lst):
        self.word_to_ix = {}
        self.ix_to_word = {}
        for i in range(len(lst)):
            self.word_to_ix[lst[i]] = i
            self.ix_to_word[i] = lst[i]

    def filter_and_shuffle(self):
        vocab = [k for k, v in self.s.items() if v >= self.freq]
        perm = torch.arange(len(vocab))

        # make a random permutation for data partition
        if self.npart > 1:
            perm = torch.randperm(len(vocab))

        self.word_to_ix = {w: perm[i].item() for i, w in enumerate(vocab)}
        self.ix_to_word = {perm[i].item(): w for i, w in enumerate(vocab)}
        self.s = None

    def getId(self, word):
        return self.word_to_ix.get(word, -1)

    def size(self):
        return len(self.ix_to_word)

    def part_size(self):
        return (self.size() - 1) // self.npart + 1

    # map id to partiion (part, offset)
    def get_partition(self, idx):
        assert 0 <= idx < self.size()
        part_size = self.part_size()
        return idx // part_size, idx % part_size

    def get_params(self):
        return {
            "freq": self.freq,
            "dict_type": self.dict_type,
            "entity_type" : self.entity_type,
            "npart" : self.npart,
            "ix_to_word": self.ix_to_word,
        }
