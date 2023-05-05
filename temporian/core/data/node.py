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

"""Node class definition."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.data.sampling import IndexDType
from temporian.utils import string

if TYPE_CHECKING:
    from temporian.core.operators.base import Operator


class Node(object):
    """Schema definition of an event set in the preprocessing graph.

    A node represents the structure, or schema, of a collection of indexed
    multivariate time series, or EventSets. A node does not contain any actual
    data, but is instead used as a reference to describe the format of the
    input, intermediate results, or output of a Processor.

    Informally, a node defines the name and data types of each time series, as
    well as the key and data type of the index (if any).

    There are several ways to create a node:
    - Through the `.node()` method in an EventSet.
    - Through applying operators to other nodes.
    - Manually using the `tp.input_node(...)` method to specify the name and
        data types of each time series and the key and data type of the index.
    - (Not recommended) By instantiating the Node class directly.
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

    def __getitem__(self, feature_names: List[str]) -> Node:
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

    def __add__(self, other: Any) -> Node:
        # TODO: In this and other operants, factor code and add support for
        # swapping operators (e.g. a+1, a+b, 1+a).

        if isinstance(other, Node):
            from temporian.core.operators.binary import add

            return add(input_1=self, input_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import add_scalar

            return add_scalar(input=self, value=other)

        raise ValueError(
            f"Cannot add {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __radd__(self, other: Any) -> Node:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import subtract

            return subtract(input_1=self, input_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=self, subtrahend=other)

        raise ValueError(
            f"Cannot subtract {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rsub__(self, other: Any) -> Node:
        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=other, subtrahend=self)

        raise ValueError(
            f"Cannot subtract {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __mul__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import multiply

            return multiply(input_1=self, input_2=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import (
                multiply_scalar,
            )

            return multiply_scalar(input=self, value=other)

        raise ValueError(
            f"Cannot multiply {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rmul__(self, other: Any) -> Node:
        return self.__mul__(other)

    def __neg__(self):
        from temporian.core.operators.scalar import multiply_scalar

        return multiply_scalar(input=self, value=-1)

    def __truediv__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import divide

            return divide(numerator=self, denominator=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=self, denominator=other)

        raise ValueError(
            f"Cannot divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rtruediv__(self, other: Any) -> Node:
        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=other, denominator=self)

        raise ValueError(
            f"Cannot divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __floordiv__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=self, denominator=other)

        raise ValueError(
            f"Cannot floor divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __rfloordiv__(self, other: Any) -> Node:
        if isinstance(other, (int, float)):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=other, denominator=self)

        raise ValueError(
            f"Cannot floor divide {type(self)} and {type(other)} objects. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __gt__(self, other: Any):
        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                greater_scalar,
            )

            return greater_scalar(input=self, value=other)

        raise ValueError(
            f"Cannot compute {type(self)} > {type(other)}. "
            "Only Event and scalar values of type int or float are supported."
        )

    def __lt__(self, other: Any):
        if isinstance(other, (int, float)):
            from temporian.core.operators.arithmetic_scalar import (
                less_scalar,
            )

            return less_scalar(input=self, value=other)

        raise ValueError(
            f"Cannot compute {type(self)} < {type(other)}. "
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


def input_node(
    features: List[Feature],
    index_levels: List[Tuple[str, IndexDType]] = [],
    name: Optional[str] = None,
    sampling: Optional[Sampling] = None,
) -> Node:
    """Creates a node with the specified attributes."""
    if sampling is None:
        sampling = Sampling(
            index_levels=index_levels, is_unix_timestamp=False, creator=None
        )

    for feature in features:
        if feature.sampling is not None:
            raise ValueError(
                "Cannot call input_node on already linked features."
            )
        feature.sampling = sampling

    return Node(
        features=features,
        sampling=sampling,
        name=name,
        creator=None,
    )
