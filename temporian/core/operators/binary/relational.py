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

"""Binary relational operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.base import EventSetOrNode
from temporian.core.operators.binary.base import BaseBinaryOperator
from temporian.core.operators.scalar.relational_scalar import equal_scalar


class BaseRelationalOperator(BaseBinaryOperator):
    DEF_KEY = ""
    PREFIX = ""

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.DEF_KEY

    @property
    def prefix(self) -> str:
        return self.PREFIX

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> DType:
        return DType.BOOLEAN


class EqualOperator(BaseRelationalOperator):
    DEF_KEY = "EQUAL"
    PREFIX = "eq"


class NotEqualOperator(BaseRelationalOperator):
    DEF_KEY = "NOT_EQUAL"
    PREFIX = "ne"


class GreaterOperator(BaseRelationalOperator):
    DEF_KEY = "GREATER"
    PREFIX = "gt"


class GreaterEqualOperator(BaseRelationalOperator):
    DEF_KEY = "GREATER_EQUAL"
    PREFIX = "ge"


class LessOperator(BaseRelationalOperator):
    DEF_KEY = "LESS"
    PREFIX = "lt"


class LessEqualOperator(BaseRelationalOperator):
    DEF_KEY = "LESS_EQUAL"
    PREFIX = "le"


@compile
def equal(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) for equality between two nodes.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.
    Note that it will always return False on NaN elements.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # WARN: Don't use this for element-wise comparison
        >>> a == b
        False

        >>> # Element-wise comparison
        >>> c = tp.equal(a, b)
        >>> c.run({source: evset})
        indexes: []
        features: [('eq_f1_f2', bool_), ('eq_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'eq_f1_f2': [False True False]
                'eq_f2_f1': [False True False]
        ...

        ```

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the comparison.
    """

    if not isinstance(input_2, Node):
        return equal_scalar(input=input_1, value=input_2)

    return EqualOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def not_equal(
    input_1: EventSetOrNode,
    input_2: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) for differences between two nodes.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.
    Note that it will always return True on NaNs (even if both are).

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.not_equal(a, b)
        >>> c = a != b
        >>> c.run({source: evset})
        indexes: []
        features: [('ne_f1_f2', bool_), ('ne_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'ne_f1_f2': [ True False True]
                'ne_f2_f1': [ True False True]
        ...

        ```

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the comparison.
    """
    return NotEqualOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


@compile
def greater(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) if input_left > input_right.

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same
    number of features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.greater(a, b)
        >>> c = a > b
        >>> c.run({source: evset})
        indexes: []
        features: [('gt_f1_f2', bool_), ('gt_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'gt_f1_f2': [ True False True]
                'gt_f2_f1': [False False False]
        ...

        ```

    Args:
        input_left: node to the left of the operator
        input_right: node to the right of the operator

    Returns:
        Node with the result of the comparison.
    """
    return GreaterOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def greater_equal(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) if input_left >= input_right.

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same
    number of features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.greater_equal(a, b)
        >>> c = a >= b
        >>> c.run({source: evset})
        indexes: []
        features: [('ge_f1_f2', bool_), ('ge_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'ge_f1_f2': [ True True True]
                'ge_f2_f1': [False True False]
        ...

        ```

    Args:
        input_left: node to the left of the operator
        input_right: node to the right of the operator

    Returns:
        Node with the result of the comparison.
    """
    return GreaterEqualOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def less(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) if input_left < input_right.

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same
    number of features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.less(a, b)
        >>> c = a < b
        >>> c.run({source: evset})
        indexes: []
        features: [('lt_f1_f2', bool_), ('lt_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'lt_f1_f2': [False False False]
                'lt_f2_f1': [ True False True]
        ...

        ```

    Args:
        input_left: node to the left of the operator
        input_right: node to the right of the operator

    Returns:
        Node with the result of the comparison.
    """
    return LessOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


@compile
def less_equal(
    input_left: EventSetOrNode,
    input_right: EventSetOrNode,
) -> EventSetOrNode:
    """Checks (element-wise) if input_left <= input_right.

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same
    number of features.

    Basic example:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 100, 200], "f2": [-10, 100, 5]}
        ... )
        >>> source = evset.node()
        >>> a = source[["f1", "f2"]]
        >>> b = source[["f2", "f1"]]  # Reverse order

        >>> # Equivalent
        >>> c = tp.less_equal(a, b)
        >>> c = a <= b
        >>> c.run({source: evset})
        indexes: []
        features: [('le_f1_f2', bool_), ('le_f2_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'le_f1_f2': [False True False]
                'le_f2_f1': [ True True True]
        ...

        ```

    Args:
        input_left: node to the left of the operator
        input_right: node to the right of the operator

    Returns:
        Node with the result of the comparison.
    """
    return LessEqualOperator(
        input_1=input_left,
        input_2=input_right,
    ).outputs["output"]


operator_lib.register_operator(EqualOperator)
operator_lib.register_operator(NotEqualOperator)
operator_lib.register_operator(GreaterOperator)
operator_lib.register_operator(GreaterEqualOperator)
operator_lib.register_operator(LessOperator)
operator_lib.register_operator(LessEqualOperator)
