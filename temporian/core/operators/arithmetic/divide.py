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

"""Divide operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class DivideOperator(BaseArithmeticOperator):
    def __init__(
        self,
        node_1: Node,
        node_2: Node,
    ):
        super().__init__(node_1, node_2)

        # Assuming previous dtype check of node_1 and node_2 features
        for feat in node_1.features:
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


operator_lib.register_operator(DivideOperator)


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
        node_1=numerator,
        node_2=denominator,
    ).outputs["node"]
