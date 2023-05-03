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

"""Floor operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class FloorDivOperator(BaseArithmeticOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "FLOORDIV"

    @property
    def prefix(self) -> str:
        return "floordiv"


operator_lib.register_operator(FloorDivOperator)


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
        node_1=numerator,
        node_2=denominator,
    ).outputs["node"]
