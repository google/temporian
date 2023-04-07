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

"""Event class definition."""

from typing import Any, List, Optional

from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.utils import string


class Event(object):
    """Collection of feature values for a certain sampling.

    An event represents the structure, or schema, of a collection of indexed
    multivariate time series. An event does not contain any actual data, but is
    instead used as a reference to describe the format of the input,
    intermediate results, or output of a Processor (i.e., a computation graph).

    Informally, an event defines the name and data types of each time series, as
    well as the key and data type of the index (if any).

    There are several ways to create an event:
    - Through the `.schema()` method in a NumpyEvent.
    - Through applying operators to other events.
    - Manually using the `tp.input_event(...)` method to specify the name and
        data types of each time series and the key and data type of the index.
    - (Not recommended) By instantiating the Event class directly.
    """

    def __init__(
        self,
        features: List[Feature],
        sampling: Sampling,
        name: Optional[str] = None,
        # TODO: make Operator the creator's type. I don't know how to circumvent
        # the cyclical import error
        creator: Optional[Any] = None,
    ):
        self._features = features
        self._sampling = sampling
        self._creator = creator
        self._name = name

    def __getitem__(self, feature_names: List[str]) -> "Event":
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

    def sampling(self):
        return self._sampling

    def features(self):
        return self._features

    def name(self) -> str:
        return self._name

    def creator(self):
        return self._creator

    def set_name(self, name) -> None:
        self._name = name

    def set_creator(self, creator):
        self._creator = creator


def input_event(
    features: List[Feature],
    index: List[str] = [],
    name: Optional[str] = None,
    sampling: Optional[Sampling] = None,
) -> Event:
    """Creates an event with the specified attributes."""
    if sampling is None:
        sampling = Sampling(index=index, creator=None)

    for feature in features:
        if feature.sampling() is not None:
            raise ValueError(
                "Cannot call input_event on already linked features."
            )
        feature.set_sampling(sampling)

    return Event(
        features=features,
        sampling=sampling,
        name=name,
        creator=None,
    )
