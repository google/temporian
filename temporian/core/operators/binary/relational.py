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
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.operators.binary.base import BaseBinaryOperator


class EqualOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "EQUAL"

    @property
    def prefix(self) -> str:
        return "eq"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


class NotEqualOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "NOT_EQUAL"

    @property
    def prefix(self) -> str:
        return "neq"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


class GreaterOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "GREATER"

    @property
    def prefix(self) -> str:
        return "gt"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


class GreaterEqualOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "GREATER_EQUAL"

    @property
    def prefix(self) -> str:
        return "ge"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


class LessOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "LESS"

    @property
    def prefix(self) -> str:
        return "lt"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


class LessEqualOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "LESS_EQUAL"

    @property
    def prefix(self) -> str:
        return "le"

    # override parent dtype method
    def output_feature_dtype(
        self, feature_1: Feature, feature_2: Feature
    ) -> DType:
        return DType.BOOLEAN


def equal(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Checks (element-wise) for equality between two nodes.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.
    Note that it will always return False on NaN elements.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Node containing the result of the comparison.
    """
    return EqualOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def not_equal(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Checks (element-wise) for differences between two nodes.

    Each feature in `input_1` is compared element-wise to the feature in
    `input_2` in the same position.
    Note that it will always return True on NaNs (even if both are).

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

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


def greater(
    input_left: Node,
    input_right: Node,
) -> Node:
    """Checks (element-wise) if input_left > input_right

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same number of
    features.

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


def greater_equal(
    input_left: Node,
    input_right: Node,
) -> Node:
    """Checks (element-wise) if input_left >= input_right

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same number of
    features.

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


def less(
    input_left: Node,
    input_right: Node,
) -> Node:
    """Checks (element-wise) if input_left < input_right

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same number of
    features.

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


def less_equal(
    input_left: Node,
    input_right: Node,
) -> Node:
    """Checks (element-wise) if input_left <= input_right

    Each feature in `input_left` is compared element-wise to the feature in
    `input_right` in the same position.
    Note that it will always return False on NaN elements.

    `input_left` and `input_right` must have the same sampling and the same number of
    features.

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
