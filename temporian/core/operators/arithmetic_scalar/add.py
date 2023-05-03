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

"""Add scalar operator class and public API function definition."""

from typing import Union, List

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic_scalar.base import (
    BaseArithmeticScalarOperator,
)


class AddScalarOperator(BaseArithmeticScalarOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "ADDITION_SCALAR"

    @property
    def prefix(self) -> str:
        return "add"

    @property
    def supported_value_dtypes(self) -> List[DType]:
        return [
            DType.FLOAT32,
            DType.FLOAT64,
            DType.INT32,
            DType.INT64,
        ]


operator_lib.register_operator(AddScalarOperator)


def add_scalar(
    node: Node,
    value: Union[float, int],
) -> Node:
    """Adds a scalar value to a node.

    `value` is added to each item in each feature in `node`.

    Args:
        node: Node to add a scalar to.
        value: Scalar value to add to the node.

    Returns:
        Addition of `node` and `value`.
    """
    return AddScalarOperator(
        node=node,
        value=value,
    ).outputs["node"]
