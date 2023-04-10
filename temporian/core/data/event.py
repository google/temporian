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

"""An event is a collection (possibly empty) of timesampled feature values."""

from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.data.sampling import IndexDtypes
from temporian.utils import string

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


class Event(object):
    def __init__(
        self,
        features: List[Feature],
        sampling: Sampling,
        name: Optional[str] = None,
        creator: Optional[Operator] = None,
    ):
        self._features = features
        self._sampling = sampling
        self._creator = creator
        self._name = name

    def __getitem__(self, feature_names: List[str]) -> Event:
        # import select operator
        from temporian.core.operators.select import select

        # return select output
        return select(self, feature_names)

    def __repr__(self) -> str:
        features_print = "\n".join(
            [string.indent(repr(feature)) for feature in self._features]
        )
        return (
            "features:\n"
            f"{features_print}\n"
            f"sampling: {self._sampling},\n"
            f"name: {self._name},\n"
            f"creator: {self._creator},\n"
            f"id:{id(self)}\n"
        )

    def __add__(self, other):
        from temporian.core.operators.arithmetic import sum

        return sum(event_1=self, event_2=other)

    def __sub__(self, other):
        from temporian.core.operators.arithmetic import substract

        return substract(event_1=self, event_2=other)

    def __mul__(self, other):
        from temporian.core.operators.arithmetic import multiply

        return multiply(event_1=self, event_2=other)

    def __truediv__(self, other):
        from temporian.core.operators.arithmetic import divide

        return divide(numerator=self, denominator=other)

    @property
    def sampling(self) -> Sampling:
        return self._sampling

    @property
    def features(self) -> List[Feature]:
        return self._features

    @property
    def name(self) -> str:
        return self._name

    @property
    def creator(self) -> Optional[Operator]:
        return self._creator

    @name.setter
    def name(self, name: str):
        self._name = name

    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator


def input_event(
    features: List[Feature],
    index: List[Tuple[str, IndexDtypes]] = [],
    name: Optional[str] = None,
    sampling: Optional[Sampling] = None,
) -> Event:
    if sampling is None:
        sampling = Sampling(index=index, creator=None)

    for feature in features:
        if feature.sampling is not None:
            raise ValueError(
                "Cannot call input_event on already linked features."
            )
        feature.sampling = sampling

    return Event(
        features=features,
        sampling=sampling,
        name=name,
        creator=None,
    )
