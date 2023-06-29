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

"""Binary logic operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.schema import FeatureSchema
from temporian.core.operators.binary.base import BaseBinaryOperator


class BaseLogicalOperator(BaseBinaryOperator):
    OP_NAME = ""

    def __init__(self, input_1: Node, input_2: Node):
        super().__init__(input_1, input_2)

        # Check that all features are boolean
        # Note: Assuming that input_1 and input_2 features have the same dtype
        for feature in input_1.schema.features:
            if feature.dtype != DType.BOOLEAN:
                raise ValueError(
                    "Logic operators only support BOOLEAN types, but feature"
                    f" {feature.name} has dtype {feature.dtype}"
                )

    def output_feature_dtype(
        self, feature_1: FeatureSchema, feature_2: FeatureSchema
    ) -> DType:
        return DType.BOOLEAN

    @classmethod
    def operator_def_key(cls) -> str:
        return cls.OP_NAME.upper()

    @property
    def prefix(self) -> str:
        return self.OP_NAME.lower()


class LogicalAndOperator(BaseLogicalOperator):
    OP_NAME = "and"


class LogicalOrOperator(BaseLogicalOperator):
    OP_NAME = "or"


class LogicalXorOperator(BaseLogicalOperator):
    OP_NAME = "xor"


def logical_and(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Gets the logical AND (`&`) between boolean features, element-wise.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling, the same number of
    features, and all feature types must be `bool` (see cast example below).

    Usage example:
        ```python
        >>> evset = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})
        >>> source = evset.node()
        >>> a = source["f1"]

        >>> # Sample boolean features
        >>> cond_1 = a > 100
        >>> cond_2 = a < 200

        >>> # Equivalent
        >>> c = tp.logical_and(cond_1, cond_2)
        >>> c = cond_1 & cond_2

        >>> c.run({source: evset})
        indexes: []
        features: [('and_f1_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'and_f1_f1': [False True False]
        ...

        ```

    Cast integer to boolean:
        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 2, 3],
        ...     features={"f1": [0, 1, 1], "f2": [1, 1, 0]}
        ... )
        >>> source = evset.node()

        >>> c = tp.cast(source["f1"], bool) & tp.cast(source["f2"], bool)

        >>> c.run({source: evset})
        indexes: []
        features: [('and_f1_f2', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'and_f1_f2': [False True False]
        ...

        ```

    Args:
        input_1: First node, with only boolean features.
        input_2: Second node, with only boolean features.

    Returns:
        Node with boolean features.
    """
    return LogicalAndOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def logical_or(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Gets the logical OR (`|`) between boolean features, element-wise.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling, the same number of
    features, and all feature types must be `bool`
    (see cast example in [`tp.logical_and()`](../logical_and)).

    Usage example:
        ```python
        >>> evset = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})
        >>> source = evset.node()
        >>> a = source["f1"]

        >>> # Sample boolean features
        >>> cond_1 = a <= 100
        >>> cond_2 = a >= 200

        >>> # Equivalent
        >>> c = tp.logical_or(cond_1, cond_2)
        >>> c = cond_1 | cond_2

        >>> c.run({source: evset})
        indexes: []
        features: [('or_f1_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'or_f1_f1': [ True False True]
        ...

        ```

    Args:
        input_1: First node, with only boolean features.
        input_2: Second node, with only boolean features.

    Returns:
        Node with boolean features.
    """
    return LogicalOrOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def logical_xor(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Gets the logical XOR (`^`) between boolean features, element-wise.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling, the same number of
    features, and all feature types must be `bool`
    (see cast example in [`tp.logical_and()`](../logical_and)).

    Usage example:
        ```python
        >>> evset = tp.event_set(timestamps=[1, 2, 3], features={"f1": [100, 150, 200]})
        >>> source = evset.node()
        >>> a = source["f1"]

        >>> # Sample boolean features
        >>> cond_1 = a > 100
        >>> cond_2 = a < 200

        >>> # Equivalent
        >>> c = tp.logical_xor(cond_1, cond_2)
        >>> c = cond_1 ^ cond_2

        >>> c.run({source: evset})
        indexes: []
        features: [('xor_f1_f1', bool_)]
        events:
            (3 events):
                timestamps: [1. 2. 3.]
                'xor_f1_f1': [ True False True]
        ...

        ```

    Args:
        input_1: First node, with only boolean features.
        input_2: Second node, with only boolean features.

    Returns:
        Node with boolean features.
    """
    return LogicalXorOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


operator_lib.register_operator(LogicalAndOperator)
operator_lib.register_operator(LogicalOrOperator)
operator_lib.register_operator(LogicalXorOperator)
