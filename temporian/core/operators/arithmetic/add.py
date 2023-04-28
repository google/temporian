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

"""Add operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.operators.arithmetic.base import BaseArithmeticOperator


class AddOperator(BaseArithmeticOperator):
    @classmethod
    @property
    def operator_def_key(cls) -> str:
        return "ADDITION"

    @property
    def prefix(self) -> str:
        return "add"


operator_lib.register_operator(AddOperator)


def add(
    node_1: Node,
    node_2: Node,
) -> Node:
    """Adds two nodes.

    Each feature in `node_1` is added to the feature in `node_2` in the same
    position.

    `node_1` and `node_2` must have the same sampling and the same number of
    features.

    Args:
        node_1: First node.
        node_2: Second node.

    Returns:
        Sum of `node_1`'s and `node_2`'s features.
    """
    return AddOperator(
        node_1=node_1,
        node_2=node_2,
    ).outputs["node"]
