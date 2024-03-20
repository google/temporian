# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, Callable, Tuple, Iterable, Optional, Iterator
from abc import ABC, abstractmethod

import apache_beam as beam

from temporian.core.operators.base import Operator
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)


class BeamOperatorImplementation(ABC):
    def __init__(self, operator: Operator):
        assert operator is not None
        self._operator = operator

    @property
    def operator(self):
        return self._operator

    @abstractmethod
    def call(self, **inputs: BeamEventSet) -> Dict[str, BeamEventSet]:
        pass

    def __call__(self, **inputs: BeamEventSet) -> Dict[str, BeamEventSet]:
        outputs = self.call(**inputs)
        return outputs


def beam_eventset_map(
    src: BeamEventSet, name: str, fn: Callable[[FeatureItem, int], FeatureItem]
) -> BeamEventSet:
    """Applies a function on each feature of a Beam eventset."""

    def apply(idx, item):
        return item | f"Map on feature #{idx} {name}" >> beam.Map(fn, idx)

    return tuple([apply(idx, item) for idx, item in enumerate(src)])


def beam_eventset_flatmap(
    src: BeamEventSet, name: str, fn: Callable[[FeatureItem, int], FeatureItem]
) -> BeamEventSet:
    """Applies a function on each feature of a Beam eventset."""

    def apply(idx, item):
        return item | f"Map on feature #{idx} {name}" >> beam.FlatMap(fn, idx)

    return tuple([apply(idx, item) for idx, item in enumerate(src)])


def _extract_from_iterable(
    src: Iterable[FeatureItemValue],
) -> Optional[FeatureItemValue]:
    for x in src:
        return x
    return None


def beam_eventset_map_with_sampling(
    input: BeamEventSet,
    sampling: BeamEventSet,
    name: str,
    fn: Callable[
        [BeamIndexKey, Optional[FeatureItemValue], FeatureItemValue, int],
        FeatureItem,
    ],
) -> BeamEventSet:
    """Applies a function on each feature of a Beam eventset."""

    assert len(sampling) >= 1

    def fn_on_cogroup(
        item: Tuple[
            BeamIndexKey,
            Tuple[Iterable[FeatureItemValue], Iterable[FeatureItemValue]],
        ],
        idx: int,
    ) -> Iterator[FeatureItem]:
        index, (it_feature, it_sampling) = item
        feature = _extract_from_iterable(it_feature)
        sampling = _extract_from_iterable(it_sampling)
        if sampling is not None:
            yield fn(index, feature, sampling, idx)

    def apply(idx, item):
        return (
            (item, sampling[0])
            | f"Join feature and sampling on feature #{idx} {name}"
            >> beam.CoGroupByKey()
            | f"Map on feature #{idx} {name}"
            >> beam.FlatMap(fn_on_cogroup, idx)
        )

    return tuple([apply(idx, item) for idx, item in enumerate(input)])
