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
from temporian.core.data.feature import Feature
from temporian.core.operators.binary.base import BaseBinaryOperator


class BaseLogicalOperator(BaseBinaryOperator):
    OP_NAME = ""

    def __init__(self, input_1: Node, input_2: Node):
        super().__init__(input_1, input_2)

        # Check that all features are boolean
        # Note: Assuming that input_1 and input_2 features have the same dtype
        for feature in input_1.features:
            if feature.dtype != DType.BOOLEAN:
                raise ValueError(
                    "Logic operators only support BOOLEAN types, but feature"
                    f" {feature.name} has dtype {feature.dtype}"
                )

    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN

    @classmethod
    @property
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
    """Element-wise boolean AND operator.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the operator.
    """
    return LogicalAndOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def logical_or(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Element-wise boolean OR operator.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the operator.
    """
    return LogicalOrOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def logical_xor(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Element-wise boolean XOR operator.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the operator.
    """
    return LogicalXorOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


operator_lib.register_operator(LogicalAndOperator)
operator_lib.register_operator(LogicalOrOperator)
operator_lib.register_operator(LogicalXorOperator)
