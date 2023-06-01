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

"""Filter operator class and public API function definition."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.data.dtype import DType
from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class FilterOperator(Operator):
    def __init__(self, input: Node, condition: Node):
        super().__init__()

        # check that condition is a single feature
        if len(condition.schema.features) != 1:
            raise ValueError(
                "Condition must be a single feature. Got"
                f" {condition.schema} instead."
            )

        # check that condition is a boolean feature
        if condition.schema.features[0].dtype != DType.BOOLEAN:
            raise ValueError(
                "Condition must be a boolean feature. Got"
                f" {condition.schema} instead."
            )

        # check both nodes have same sampling
        input.check_same_sampling(condition)

        # inputs
        self.add_input("input", input)
        self.add_input("condition", condition)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=input.schema.features,
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="FILTER",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="condition"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(FilterOperator)


# pylint: disable=redefined-builtin
def filter(
    input: Node,
    condition: Optional[Node] = None,
) -> Node:
    """Filters out timestamps in a node for which a condition is false.

    Each timestamp in `input` is only kept if the corresponding value for that
    timestamp in `condition` is `True`.

    `input` and `condition` must have the same sampling.

    filter(x) is equivalent to filter(x,x). filter(x) can be used to convert
    a boolean mask into a timestamps. For example:
        Input:
            timestamps: 1 2 3
            value: True False True
        Output:
            timestamps: 1 3
            value: True True

    Args:
        input: Node to filter.
        condition: Node with a single boolean feature condition.

    Returns:
        Filtered input.
    """

    if condition is None:
        condition = input

    return FilterOperator(input, condition).outputs["output"]
