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

"""Node and related classes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING, Any, Union

from temporian.core.data.dtype import DType, IndexDType
from temporian.core.data.schema import Schema, FeatureSchema, IndexSchema
from temporian.utils import string

if TYPE_CHECKING:
    from temporian.core.evaluation import EvaluationInput, EvaluationResult
    from temporian.core.operators.base import Operator

T_SCALAR = (int, float)


class Node:
    """A Node is a reference to the input/output of ops in a compute graph.

    Use [`tp.input_node()`][temporian.input_node] to create a Node manually, or
    use [`event_set.node()`][temporian.EventSet.node] to create a Node
    compatible with a given [`EventSet`][temporian.EventSet].

    A Node does not contain any data. Use
    [`node.run()`][temporian.Node.run] to get the
    [`EventSet`][temporian.EventSet] resulting from a [`Nodes`][temporian.Node].
    """

    def __init__(
        self,
        schema: Schema,
        features: List[Feature],
        sampling: Sampling,
        name: Optional[str] = None,
        creator: Optional[Operator] = None,
    ):
        self._schema = schema
        self._features = features
        self._sampling = sampling
        self._creator = creator
        self._name = name

    @property
    def schema(self) -> Schema:
        """Schema of the Node.

        The schema defines the name and dtype of the features and the index.

        Returns:
            Schema of the Node.
        """
        return self._schema

    @property
    def sampling_node(self) -> Sampling:
        """Sampling node.

        Equality between sampling nodes is used to check that two nodes are
        sampled similarly. Use
        [`node.check_same_sampling()`][temporian.Node.check_same_sampling]
        instead of `sampling_node`.
        """
        return self._sampling

    @property
    def feature_nodes(self) -> List[Feature]:
        """Feature nodes.

        Equality between feature nodes is used to check that two nodes use
        the same feature data.
        """
        return self._features

    @property
    def name(self) -> Optional[str]:
        """Name of a Node.

        The name of a Node is used to facilitate debugging and to specify the
        input / output signature of a graph during graph import / export."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def creator(self) -> Optional[Operator]:
        """Creator.

        The creator is the operator that outputs this Node. Manually created
        Nodes have a `None` creator.
        """
        return self._creator

    @creator.setter
    def creator(self, creator: Optional[Operator]):
        self._creator = creator

    @property
    def features(self) -> List[FeatureSchema]:
        """Feature schema.

        Alias for `node.schema.features`.
        """
        return self.schema.features

    @property
    def indexes(self) -> List[IndexSchema]:
        """Index schema.

        Alias for `node.schema.indexes`.
        """
        return self.schema.indexes

    def check_same_sampling(self, other: Node):
        """Checks if two Nodes have the same sampling."""

        self.schema.check_compatible_index(other.schema)
        if self.sampling_node is not other.sampling_node:
            raise ValueError(
                "Arguments should have the same sampling. "
                f"{self.sampling_node} is different from "
                f"{other.sampling_node}. To create input Nodes with the same "
                "sampling, use the argument `same_sampling_as` of "
                "`tp.input_node` or `tp.event_set`. To align the sampling of "
                "two Nodes with same indexes but different sampling, use the "
                "operator `tp.resample`."
            )

    def run(
        self,
        input: EvaluationInput,
        verbose: int = 1,
        check_execution: bool = True,
    ) -> EvaluationResult:
        """Evaluates the Node on the specified input.

        See [`tp.run()`][temporian.run] for details.
        """
        from temporian.core.evaluation import run

        return run(
            query=self,
            input=input,
            verbose=verbose,
            check_execution=check_execution,
        )

    def __getitem__(self, feature_names: Union[str, List[str]]) -> Node:
        """Creates a Node with a subset of the features."""

        from temporian.core.operators.select import select

        return select(self, feature_names)

    def __setitem__(self, feature_names: Any, value: Any) -> None:
        """Fails, features cannot be assigned"""

        raise TypeError(
            "Cannot assign features to an existing node. "
            "New nodes should be created instead."
        )

    def __repr__(self) -> str:
        """Human readable representation of a Node."""

        schema_print = string.indent(repr(self._schema))
        return (
            f"schema:\n{schema_print}\n"
            f"features: {self._features}\n"
            f"sampling: {self._sampling},\n"
            f"name: {self._name}\n"
            f"creator: {self._creator}\n"
            f"id:{id(self)}\n"
        )

    def __bool__(self) -> bool:
        """Catches bool evaluation with an error message."""

        # Called on "if node" conditions
        # TODO: modify to similar numpy msg if we implement .any() or .all()
        raise ValueError(
            "The truth value of a Node is ambiguous. Check condition"
            " element-wise or use cast() operator to convert to boolean."
        )

    def _raise_error(
        self, op_name: str, other: Any, allowed_types: str
    ) -> None:
        """Raises an error message.

        This utility method is used in operator implementations, e.g., +, - *.
        """

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

        self._raise_error("ne", other, "int,float,bool,str")
        assert False

    def __add__(self, other: Any) -> Node:
        # TODO: In this and other operants, factor code and add support for
        # swapping operators (e.g. a+1, a+b, 1+a).

        if isinstance(other, Node):
            from temporian.core.operators.binary import add

            return add(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import add_scalar

            return add_scalar(input=self, value=other)

        self._raise_error("add", other, "int,float")
        assert False

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

        self._raise_error("subtract", other, "int,float")
        assert False

    def __rsub__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                subtract_scalar,
            )

            return subtract_scalar(minuend=other, subtrahend=self)

        self._raise_error("subtract", other, "int,float")
        assert False

    def __mul__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import multiply

            return multiply(input_1=self, input_2=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                multiply_scalar,
            )

            return multiply_scalar(input=self, value=other)

        self._raise_error("multiply", other, "int,float")
        assert False

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

        self._raise_error("divide", other, "(int,float)")
        assert False

    def __rtruediv__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import divide_scalar

            return divide_scalar(numerator=other, denominator=self)

        self._raise_error("divide", other, "(int,float)")
        assert False

    def __floordiv__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import floordiv

            return floordiv(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=self, denominator=other)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __rfloordiv__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                floordiv_scalar,
            )

            return floordiv_scalar(numerator=other, denominator=self)

        self._raise_error("floor_divide", other, "(int,float)")
        assert False

    def __pow__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import power

            return power(base=self, exponent=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=self, exponent=other)

        self._raise_error("exponentiate", other, "(int,float)")
        assert False

    def __rpow__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import power_scalar

            return power_scalar(base=other, exponent=self)

        self._raise_error("exponentiate", other, "(int,float)")
        assert False

    def __mod__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import modulo

            return modulo(numerator=self, denominator=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=self, denominator=other)

        self._raise_error("compute modulo (%)", other, "(int,float)")
        assert False

    def __rmod__(self, other: Any) -> Node:
        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import modulo_scalar

            return modulo_scalar(numerator=other, denominator=self)

        self._raise_error("compute modulo (%)", other, "(int,float)")
        assert False

    def __gt__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import greater

            return greater(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_scalar,
            )

            return greater_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __ge__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import greater_equal

            return greater_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                greater_equal_scalar,
            )

            return greater_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __lt__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import less

            return less(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_scalar,
            )

            return less_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def __le__(self, other: Any):
        if isinstance(other, Node):
            from temporian.core.operators.binary import less_equal

            return less_equal(input_left=self, input_right=other)

        if isinstance(other, T_SCALAR):
            from temporian.core.operators.scalar import (
                less_equal_scalar,
            )

            return less_equal_scalar(input=self, value=other)

        self._raise_error("compare", other, "(int,float)")
        assert False

    def _error_only_boolean(self, boolean_op: str, other: Any) -> None:
        raise ValueError(
            f"Cannot compute 'Node {boolean_op} {type(other)}'. "
            "Only Nodes with boolean features are supported."
        )

    def __and__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_and

            return logical_and(input_1=self, input_2=other)

        self._error_only_boolean("&", other)
        assert False

    def __or__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_or

            return logical_or(input_1=self, input_2=other)

        self._error_only_boolean("|", other)
        assert False

    def __xor__(self, other: Any) -> Node:
        if isinstance(other, Node):
            from temporian.core.operators.binary import logical_xor

            return logical_xor(input_1=self, input_2=other)

        self._error_only_boolean("^", other)
        assert False


def input_node(
    features: List[Tuple[str, DType]],
    indexes: Optional[List[Tuple[str, IndexDType]]] = None,
    is_unix_timestamp: bool = False,
    same_sampling_as: Optional[Node] = None,
    name: Optional[str] = None,
) -> Node:
    """Creates an input [`Node`][temporian.Node].

    An input Node can be used to feed data into a graph.

    Usage example:

        ```python
        >>> # Without index
        >>> a = tp.input_node(features=[("f1", tp.float64), ("f2", tp.str_)])

        >>> # With an index
        >>> a = tp.input_node(
        ...     features=[("f1", tp.float64), ("f2", tp.str_)],
        ...     indexes=["f2"],
        ... )

        >>> # Two nodes with the same sampling
        >>> a = tp.input_node(features=[("f1", tp.float64)])
        >>> b = tp.input_node(features=[("f2", tp.float64)], same_sampling_as=a)

        ```

    Args:
        features: List of names and dtypes of the features.
        indexes: List of names and dtypes of the index. If empty, the data is
            assumed not indexed.
        is_unix_timestamp: If true, the timestamps are interpreted as unix
            timestamps in seconds.
        same_sampling_as: If set, the created Node is guaranteed to have the
            same sampling as same_sampling_as`. In this case, `indexes` and
            `is_unix_timestamp` should not be provided. Some operators require
            for input Nodes to have the same sampling.

    Returns:
        Node with the given specifications.
    """

    if same_sampling_as is not None:
        if indexes is not None:
            raise ValueError(
                "indexes cannot be provided with same_sampling_as=True"
            )
        return create_node_new_features_existing_sampling(
            features=features,
            sampling_node=same_sampling_as,
            name=name,
            creator=None,
        )

    else:
        if indexes is None:
            indexes = []

        return create_node_new_features_new_sampling(
            features=features,
            indexes=indexes,
            is_unix_timestamp=is_unix_timestamp,
            name=name,
            creator=None,
        )


@dataclass
class Sampling:
    """A sampling is a reference to the way data is sampled."""

    creator: Optional[Operator] = None

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Sampling(id={id(self)}, creator={self.creator})"


@dataclass
class Feature:
    """A feature is a reference to sampled data."""

    creator: Optional[Operator] = None

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Feature(id={id(self)}, creator={self.creator})"


def create_node_new_features_existing_sampling(
    features: Union[List[FeatureSchema], List[Tuple[str, DType]]],
    sampling_node: Node,
    creator: Optional[Operator],
    name: Optional[str] = None,
) -> Node:
    """Creates a Node with an existing sampling and new features.

    When possible, this is the Node creation function to use.
    """

    # TODO: Use better way
    assert sampling_node is not None
    assert features is not None
    assert isinstance(sampling_node, Node)
    assert isinstance(features, List)
    assert (
        len(features) == 0
        or isinstance(features[0], FeatureSchema)
        or isinstance(features[0], tuple)
    )

    return Node(
        schema=Schema(
            features=features,
            # The indexes and is_unix_timestamp are defined by the sampling.
            indexes=sampling_node.schema.indexes,
            is_unix_timestamp=sampling_node.schema.is_unix_timestamp,
        ),
        # Making use to use the same sampling reference.
        sampling=sampling_node.sampling_node,
        # New features.
        features=[Feature(creator=creator) for _ in features],
        name=name,
        creator=creator,
    )


def create_node_new_features_new_sampling(
    features: Union[List[FeatureSchema], List[Tuple[str, DType]]],
    indexes: Union[List[IndexSchema], List[Tuple[str, IndexDType]]],
    is_unix_timestamp: bool,
    creator: Optional[Operator],
    name: Optional[str] = None,
) -> Node:
    """Creates a Node with a new sampling and new features."""

    # TODO: Use better way
    assert isinstance(features, List)
    assert (
        len(features) == 0
        or isinstance(features[0], FeatureSchema)
        or isinstance(features[0], tuple)
    )

    return Node(
        schema=Schema(
            features=features,
            indexes=indexes,
            is_unix_timestamp=is_unix_timestamp,
        ),
        # New sampling
        sampling=Sampling(creator=creator),
        # New features.
        features=[Feature(creator=creator) for _ in features],
        name=name,
        creator=creator,
    )


def create_node_with_new_reference(
    schema: Schema,
    sampling: Optional[Sampling] = None,
    features: Optional[List[Feature]] = None,
    name: Optional[str] = None,
    creator: Optional[Operator] = None,
) -> Node:
    """Creates a Node with NEW features and NEW sampling.

    If sampling is not specified, a new sampling is created.
    Similarly, if features is not specifies, new features are created.
    """

    if sampling is None:
        sampling = Sampling(creator=creator)

    if features is None:
        features = [Feature(creator=creator) for _ in schema.features]
    assert len(features) == len(schema.features)

    return Node(
        schema=schema,
        sampling=sampling,
        features=features,
        name=name,
        creator=creator,
    )
