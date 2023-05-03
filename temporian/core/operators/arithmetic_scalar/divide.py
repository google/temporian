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

"""Divide scalar operator class and public API function definition."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class DivideScalarOperator(BaseArithmeticScalarOperator):
    def __init__(
        self,
        input: Node,
        value: Union[float, int],
        is_value_first: bool = False,
    ):
        super().__init__(input, value, is_value_first)

        for feat in input.features:
            if feat.dtype in [DType.INT32, DType.INT64]:
                raise ValueError(
                    "Cannot use the divide operator on feature "
                    f"{feat.name} of type {feat.dtype}. Cast to a "
                    "floating point type or use floordiv operator (//)."
                )

    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "DIVISION_SCALAR"

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(DivideScalarOperator)

SCALAR = Union[float, int]


def divide_scalar(
    numerator: Union[Node, SCALAR],
    denominator: Union[Node, SCALAR],
) -> Node:
    """Divides a node and a scalar value.

    Each item in each feature in the node is divided with the scalar value.

    Either `numerator` or `denominator` should be a scalar value, but not both.
    If looking to divide two nodes, use the `divide` operator instead.

    Args:
        numerator: Numerator node or value.
        denominator: Denominator node or value.

    Returns:
        Division of `numerator` and `denominator`.
    """
    scalars_types = (float, int)

    if isinstance(numerator, Node) and isinstance(denominator, scalars_types):
        return DivideScalarOperator(
            input=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(numerator, scalars_types) and isinstance(denominator, Node):
        return DivideScalarOperator(
            input=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for divide_scalar. "
        "Expected (Node, SCALAR) or (SCALAR, Node), "
        f"got ({type(numerator)}, {type(denominator)})."
    )
