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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any, Union

from temporian.core.data.dtype import DType
from temporian.core.data.feature import Feature, FeatureTuple
from temporian.core.data.sampling import Sampling
from temporian.core.data.sampling import IndexDType
from temporian.utils import string

if TYPE_CHECKING:
    from temporian.core.evaluation import EvaluationInput
    from temporian.core.operators.base import Operator
    from temporian.implementation.numpy.data.event_set import EventSet

T_SCALAR = (int, float)


class Node(object):
    """Schema definition of an event set in the preprocessing graph.

    A node represents the structure, or schema, of a collection of indexed
    multivariate time series, or EventSets. A node does not contain any actual
    data, but is instead used as a reference to describe the format of the
    input, intermediate results, or output of a Graph.

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
        features: List[Feature, FeatureTuple],
        sampling: Sampling,
        name: Optional[str] = None,
        creator: Optional[Operator] = None,
    ):
        for idx, feature in enumerate(features):
            # Convert tuples to feature
            if isinstance(feature, tuple):
                features[idx] = Feature.from_tuple(feature)
                features[idx].sampling = sampling
            elif not isinstance(feature, Feature):
                raise ValueError(f"Unrecognized feature format: {feature}")

        self._features = features
        self._sampling = sampling
        self._creator = creator
        self._name = name

    def evaluate(
        self,
        input: EvaluationInput,
        verbose: int = 1,
        check_execution: bool = True,
    ) -> EventSet:
        """Evaluates the node on the specified input.

        See `tp.evaluate` for details.
        """
        from temporian.core.evaluation import evaluate

        return evaluate(
            query=self,
            input=input,
            verbose=verbose,
            check_execution=check_execution,
        )

    def __getitem__(self, feature_names: Union[str, List[str]]) -> Node:
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

    def __bool__(self) -> bool:
        # Called on "if node" conditions
        # TODO: modify to similar numpy msg if we implement .any() or .all()
        raise ValueError(
            "The truth value of a node is ambiguous. Check condition"
            " element-wise or use cast() operator to convert to boolean."
        )

    def _nope(
        self, op_name: str, other: Any, allowed_types: Tuple[type]
    ) -> None:
        raise ValueError(
            f"Cannot {op_name} Node and {type(other)} objects. "
            f"Only Node or values of type ({allowed_types}) are supported."
        )

    def __ne__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import not_equal

            return not_equal(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR + (bool, str)):
            from temporian.core.operators.scalar import not_equal_scalar

            return not_equal_scalar(input=self, value=other)

        self._nope("compare", other, "(int,float,bool,str)")

    def __add__(self, other: Any) -> Node:
        # TODO: In this and other operants, factor code and add support for
        # swapping operators (e.g. a+1, a+b, 1+a).

        if isinstance(other, Node):
            from temporian.core.operators.binary import add

            return add(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import add_scalar

            return add_scalar(input=self, value=other)

        self._nope("add", other, "(int,float)")

    def __radd__(self, other: Any) -> Node:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import subtract

            return subtract(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=self, subtrahend=other)

        self._nope("subtract", other, "(int,float)")

    def __rsub__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=other, subtrahend=self)

        self._nope("subtract", other, "(int,float)")

    def __mul__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import multiply

            return multiply(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                multiply_scalar,
            )

            return multiply_scalar(input=self, value=other)

        self._nope("multiply", other, "(int,float)")

    def __rmul__(self, other: Any) -> Node:
        return self.__mul__(other)

    def __neg__(self):
        from temporian.core.operators.scalar import multiply_scalar

        return multiply_scalar(input=self, value=-1)

    def __invert__(self):
        from temporian.core.operators.unary import invert

        return invert(input=self)

    def __abs__(self):
        from temporian.core.operators.unary import abs

        return abs(input=self)

    def __truediv__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import divide

            return divide(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=self, denominator=other)

        self._nope("divide", other, "(int,float)")

    def __rtruediv__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=other, denominator=self)

        self._nope("divide", other, "(int,float)")

    def __floordiv__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=self, denominator=other)

        self._nope("floor_divide", other, "(int,float)")

    def __rfloordiv__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=other, denominator=self)

        self._nope("floor_divide", other, "(int,float)")

    def __pow__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import power

            return power(base=self, exponent=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=self, exponent=other)

        self._nope("exponentiate", other, "(int,float)")

    def __rpow__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=other, exponent=self)

        self._nope("exponentiate", other, "(int,float)")

    def __mod__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import modulo

            return modulo(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=self, denominator=other)

        self._nope("compute modulo (%)", other, "(int,float)")

    def __rmod__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=other, denominator=self)

        self._nope("compute modulo (%)", other, "(int,float)")

    def __gt__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import greater

            return greater(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_scalar,
            )

            return greater_scalar(input=self, value=other)

        self._nope("compare", other, "(int,float)")

    def __ge__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import greater_equal

            return greater_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_equal_scalar,
            )

            return greater_equal_scalar(input=self, value=other)

        self._nope("compare", other, "(int,float)")

    def __lt__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import less

            return less(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_scalar,
            )

            return less_scalar(input=self, value=other)

        self._nope("compare", other, "(int,float)")

    def __le__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import less_equal

            return less_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_equal_scalar,
            )

            return less_equal_scalar(input=self, value=other)

        self._nope("compare", other, "(int,float)")

    def _nope_only_boolean(self, boolean_op: str, other: Any) -> None:
        raise ValueError(
            f"Cannot compute 'Node {boolean_op} {type(other)}'. "
            "Only Nodes with boolean features are supported."
        )

    def __and__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_and

            return logical_and(input_1=self, input_2=other)
        self._nope_only_boolean("&", other)

    def __or__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_or

            return logical_or(input_1=self, input_2=other)
        self._nope_only_boolean("|", other)

    def __xor__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_xor

            return logical_xor(input_1=self, input_2=other)
        self._nope_only_boolean("^", other)

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
    features: List[Union[Feature, FeatureTuple]],
    index_levels: Optional[List[Tuple[str, IndexDType]]] = None,
    name: Optional[str] = None,
    sampling: Optional[Sampling] = None,
) -> Node:
    """Creates a node with the specified attributes."""
    if index_levels is None:
        index_levels = []

    if sampling is None:
        sampling = Sampling(
            index_levels=index_levels, is_unix_timestamp=False, creator=None
        )

    for feature in features:
        if not isinstance(feature, Feature):
            # These cases are handled in Node
            continue
        if feature.sampling is None:
            feature.sampling = sampling
        elif feature.sampling is not sampling:
            raise ValueError(
                f"Cannot add feature {feature.name} to node since it has a"
                " different sampling."
            )

    return Node(
        features=features,
        sampling=sampling,
        name=name,
        creator=None,
    )
