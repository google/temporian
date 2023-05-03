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

"""Floor division scalar operator class and public API function definition."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class FloorDivScalarOperator(BaseArithmeticScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "FLOORDIV_SCALAR"

    @property
    def prefix(self) -> str:
        return "floordiv"

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(FloorDivScalarOperator)


SCALAR = Union[float, int]


def floordiv_scalar(
    numerator: Union[Node, SCALAR],
    denominator: Union[Node, SCALAR],
) -> Node:
    """Divides a node and a scalar and takes the result's floor.

    Each item in each feature in the node is divided with the scalar value.

    Either `numerator` or `denominator` should be a scalar value, but not both.
    If looking to floordiv two nodes, use the `floordiv` operator instead.

    Args:
        numerator: Numerator node or value.
        denominator: Denominator node or value.

    Returns:
        Integer division of `numerator` and `denominator`.
    """
    scalars_types = (float, int)

    if isinstance(numerator, Node) and isinstance(denominator, scalars_types):
        return FloorDivScalarOperator(
            node=numerator,
            value=denominator,
            is_value_first=False,
        ).outputs["node"]

    if isinstance(numerator, scalars_types) and isinstance(denominator, Node):
        return FloorDivScalarOperator(
            node=denominator,
            value=numerator,
            is_value_first=True,
        ).outputs["node"]

    raise ValueError(
        "Invalid input types for floordiv_scalar. "
        "Expected (Node, SCALAR) or (SCALAR, Node), "
        f"got ({type(numerator)}, {type(denominator)})."
    )
