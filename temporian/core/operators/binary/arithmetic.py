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

"""Binary arithmetic operators classes and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.data.dtype import DType
from temporian.core.operators.binary.base import BaseBinaryOperator


class AddOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "ADDITION"

    @property
    def prefix(self) -> str:
        return "add"


class SubtractOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "SUBTRACTION"

    @property
    def prefix(self) -> str:
        return "sub"


class MultiplyOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "MULTIPLICATION"

    @property
    def prefix(self) -> str:
        return "mult"


class DivideOperator(BaseBinaryOperator):
    def __init__(
        self,
        input_1: Node,
        input_2: Node,
    ):
        super().__init__(input_1, input_2)

        # Assuming previous dtype check of input_1 and input_2 features
        for feat in input_1.features:
            if feat.dtype in [DType.INT32, DType.INT64]:
                raise ValueError(
                    "Cannot use the divide operator on feature "
                    f"{feat.name} of type {feat.dtype}. Cast to "
                    "a floating point type or use "
                    "floordiv operator (//) instead, on these integer types."
                )

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "DIVISION"

    @property
    def prefix(self) -> str:
        return "div"


class FloorDivOperator(BaseBinaryOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "FLOORDIV"

    @property
    def prefix(self) -> str:
        return "floordiv"


operator_lib.register_operator(AddOperator)
operator_lib.register_operator(SubtractOperator)
operator_lib.register_operator(DivideOperator)
operator_lib.register_operator(MultiplyOperator)
operator_lib.register_operator(FloorDivOperator)


def add(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Adds two nodes.

    Each feature in `input_1` is added to the feature in `input_2` in the same
    position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Sum of `input_1`'s and `input_2`'s features.
    """
    return AddOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def subtract(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Subtracts two nodes.

    Each feature in `input_2` is subtracted from the feature in `input_1` in the
    same position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Subtraction of `input_2`'s features from `input_1`'s.
    """
    return SubtractOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def multiply(
    input_1: Node,
    input_2: Node,
) -> Node:
    """Multiplies two nodes.

    Each feature in `input_1` is multiplied by the feature in `input_2` in the
    same position.

    `input_1` and `input_2` must have the same sampling and the same number of
    features.

    Args:
        input_1: First node.
        input_2: Second node.

    Returns:
        Multiplication of `input_1`'s and `input_2`'s features.
    """
    return MultiplyOperator(
        input_1=input_1,
        input_2=input_2,
    ).outputs["output"]


def divide(
    numerator: Node,
    denominator: Node,
) -> Node:
    """Divides two nodes.

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    `numerator` and `denominator` must have the same sampling and the same
    number of features.

    Args:
        numerator: Numerator node.
        denominator: Denominator node.

    Returns:
        Division of `numerator`'s features by `denominator`'s features.
    """
    return DivideOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]


def floordiv(
    numerator: Node,
    denominator: Node,
) -> Node:
    """Divides two nodes and takes the floor of the result.

    I.e. computes numerator//denominator.

    Each feature in `numerator` is divided by the feature in `denominator` in
    the same position.

    `numerator` and `denominator` must have the same sampling and the same
    number of features.

    Args:
        numerator: Numerator node.
        denominator: Denominator node.

    Returns:
        Integer division of `numerator`'s features by `denominator`'s features.
    """
    return FloorDivOperator(
        input_1=numerator,
        input_2=denominator,
    ).outputs["output"]
