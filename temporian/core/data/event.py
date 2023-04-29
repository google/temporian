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

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.data.sampling import IndexDType
from temporian.utils import string

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


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
            [
                string.indent(feature.to_string(include_sampling=False))
                for feature in self._features
            ]
        )
        return (
            "features:\n"
            f"{features_print}\n"
            f"sampling: {self._sampling},\n"
            f"name: {self._name}\n"
            f"creator: {self._creator}\n"
            f"id:{id(self)}\n"
        )

    def __add__(self, other: Any) -> Event:
        if isinstance(other, Event):
            from temporian.core.operators.arithmetic import add

            return add(event_1=self, event_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import add_scalar

            return add_scalar(event=self, value=other)

        raise ValueError(
            f"Cannot add {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __radd__(self, other: Any) -> Event:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Event:
        if isinstance(other, Event):
            from temporian.core.operators.arithmetic import subtract

            return subtract(event_1=self, event_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=self, subtrahend=other)

        raise ValueError(
            f"Cannot subtract {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rsub__(self, other: Any) -> Event:
        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=other, subtrahend=self)

        raise ValueError(
            f"Cannot subtract {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __mul__(self, other: Any) -> Event:
        if isinstance(other, Event):
            from temporian.core.operators.arithmetic import multiply

            return multiply(event_1=self, event_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                multiply_scalar,
            )

            return multiply_scalar(event=self, value=other)

        raise ValueError(
            f"Cannot multiply {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rmul__(self, other: Any) -> Event:
        return self.__mul__(other)

    def __neg__(self):
        from temporian.core.operators.arithmetic_scalar import negate

        return negate(self)

    def __truediv__(self, other: Any) -> Event:
        if isinstance(other, Event):
            from temporian.core.operators.arithmetic import divide

            return divide(numerator=self, denominator=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import divide_scalar

            return divide_scalar(numerator=self, denominator=other)

        raise ValueError(
            f"Cannot divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rtruediv__(self, other: Any) -> Event:
        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import divide_scalar

            return divide_scalar(numerator=other, denominator=self)

        raise ValueError(
            f"Cannot divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __floordiv__(self, other: Any) -> Event:
        if isinstance(other, Event):
            from temporian.core.operators.arithmetic import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=self, denominator=other)

        raise ValueError(
            f"Cannot floor divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rfloordiv__(self, other: Any) -> Event:
        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=other, denominator=self)

        raise ValueError(
            f"Cannot floor divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    @property
    def sampling(self) -> Sampling:
        return self._sampling

    @property
    def features(self) -> List[Feature]:
        return self._features

    @property
    def feature_names(self) -> List[str]:
        return [feature.name for feature in self._features]

    @property
    def index_names(self) -> List[str]:
        return self.sampling.index.names

    @property
    def dtypes(self) -> Dict[str, DType]:
        return {feature.name: feature.dtype for feature in self._features}

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
    index_levels: List[Tuple[str, IndexDType]] = [],
    name: Optional[str] = None,
    sampling: Optional[Sampling] = None,
) -> Event:
    """Creates an event with the specified attributes."""
    if sampling is None:
        sampling = Sampling(index_levels=index_levels, creator=None)

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
