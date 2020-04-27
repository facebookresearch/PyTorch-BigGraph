#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, Optional, Type, Union, cast

from torchbiggraph.types import FloatTensorType


def average_of_sums(*tensors: FloatTensorType) -> float:
    return mean(t.sum().item() for t in tensors)


SerializedStats = Dict[str, Union[int, Dict[str, float]]]


class Stats:
    """A class collecting a set of metrics.

    When defining the stats produced by a certain operation (say, training or
    evaluation), subclass this class, decorate it with @stats and define the
    metrics you want to collect as class attributes with type annotations whose
    values are attr.ib() instances. A metric named count is automatically added.
    Doing this automatically gives you space-optimized classes (using slots)
    equipped with the most common magic methods (__init__, __eq__, ...) plus
    some convenience methods to aggregate, convert and format stats (see below).

    """

    def __init__(self, *, count: int, **metrics: float) -> None:
        self.count = count
        self.metrics = metrics

    @classmethod
    def sum(cls: Type["Stats"], stats: Iterable["Stats"]) -> "Stats":
        """Return a stats whose metrics are the sums of the given stats.

        """
        total_metrics = defaultdict(lambda: 0)
        for s in stats:
            for k, v in s.metrics.items():
                total_metrics[k] += v
        return cls(count=sum(s.count for s in stats), **total_metrics)

    def average(self) -> "Stats":
        """Return these stats with all metrics, except count, averaged.

        """
        if self.count == 0:
            return self
        return type(self)(
            count=self.count, **{k: v / self.count for k, v in self.metrics.items()}
        )

    @classmethod
    def average_list(cls: Type["Stats"], stats: Iterable["Stats"]) -> "Stats":
        """Return a stats whose metrics are the average of all stats.
        """

        return cls.sum([s * s.count for s in stats]).average()

    def __str__(self) -> str:
        return "%s , count:  %d" % (
            " , ".join("%s:  %.6g" % (k, v) for k, v in self.metrics.items()),
            self.count,
        )

    def __eq__(self, other: "Stats") -> bool:
        return (
            isinstance(other, Stats)
            and self.count == other.count
            and self.metrics == other.metrics
        )

    def __mul__(self, c: float) -> "Stats":
        return type(self)(
            count=self.count, **{k: v * c for k, v in self.metrics.items()}
        )

    def to_dict(self) -> SerializedStats:
        return {"count": self.count, "metrics": self.metrics}

    @classmethod
    def from_dict(cls, d: SerializedStats) -> "Stats":
        if set(d.keys()) != {"count", "metrics"}:
            raise ValueError(
                f"Expect keys ['count', 'metrics'] from input but get {list(d.keys())}."
            )
        return Stats(
            count=cast(int, d["count"]), **cast(Dict[str, float], d["metrics"])
        )


class StatsHandler:
    def on_stats(
        self,
        index: int,
        eval_stats_before: Optional[Stats] = None,
        train_stats: Optional[Stats] = None,
        eval_stats_after: Optional[Stats] = None,
        eval_stats_chunk_avg: Optional[Stats] = None,
    ) -> None:
        pass
