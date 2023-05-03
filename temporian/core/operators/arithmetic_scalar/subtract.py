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

"""Subtract scalar operator class and public API function definition."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class SubtractScalarOperator(BaseArithmeticScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "SUBTRACTION_SCALAR"

    @property
    def prefix(self) -> str:
        return "sub"

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(SubtractScalarOperator)


SCALAR = Union[float, int]


def subtract_scalar(
    minuend: Union[Node, SCALAR],
    subtrahend: Union[Node, SCALAR],
) -> Node:
    """Subtracts a node and a scalar value.

    Each item in each feature in the node is subtracted with the scalar value.

    Either `minuend` or `subtrahend` should be a scalar value, but not both. If
    looking to subtract two nodes, use the `subtract` operator instead.

    Args:
        minuend: Node or scalar value being subtracted from.
        subtrahend: Node or scalar number being subtracted.

    Returns:
        Node with the difference between the minuend and subtrahend.
    """
    scalars_types = (float, int)

    if isinstance(minuend, Node) and isinstance(subtrahend, scalars_types):
        return SubtractScalarOperator(
            input=minuend,
            value=subtrahend,
            is_value_first=False,
        ).outputs["output"]

    if isinstance(minuend, scalars_types) and isinstance(subtrahend, Node):
        return SubtractScalarOperator(
            input=subtrahend,
            value=minuend,
            is_value_first=True,
        ).outputs["output"]

    raise ValueError(
        "Invalid input types for subtract_scalar. "
        "Expected (Node, SCALAR) or (SCALAR, Node), "
        f"got ({type(minuend)}, {type(subtrahend)})."
    )
