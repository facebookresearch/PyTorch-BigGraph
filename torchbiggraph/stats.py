#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Type, TypeVar

import attr


# This decorator must be applied to all classes that are intended to be used as
# stats. It parses the class-level attributes defined as attr.ibs and produces
# __init__, __eq__, __hash__, __str__ and other magic methods for them.
# TODO Remove noqa when flake8 will understand kw_only added in attrs-18.2.0.
stats = attr.s(kw_only=True, slots=True, frozen=True)  # noqa


StatsType = TypeVar('StatsType', bound='Stats')


@stats
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

    count: int = attr.ib()  # The number of data points this stats aggregates.

    @classmethod
    def sum(cls: Type[StatsType], stats: Iterable[StatsType]) -> StatsType:
        """Return a stats whose metrics are the sums of the given stats.

        """
        # TODO Remove noqa when flake8 will understand kw_only added in attrs-18.2.0.
        return cls(  # noqa
            **{
                k: sum(getattr(s, k) for s in stats)
                for k in attr.fields_dict(cls)
            }
        )

    def average(self: StatsType) -> StatsType:
        """Return these stats with all metrics, except count, averaged.

        """
        if self.count == 0:
            return self
        # TODO Remove noqa when flake8 will understand kw_only added in attrs-18.2.0.
        return type(self)(  # noqa
            count=self.count,
            **{
                k: getattr(self, k) / self.count
                for k in attr.fields_dict(type(self))
                if k != 'count'
            },
        )

    def __str__(self) -> str:
        fields = attr.fields(type(self))
        # Count is first but should be printed last.
        assert fields[0].name == "count"
        return " , ".join(
            "%s:  %.6g" % (f.name, getattr(self, f.name))
            for f in fields[1:] + (fields[0],)
        )
