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

from datetime import datetime
from typing import Optional, Union

from temporian.core import operator_lib
from temporian.core.compilation import (
    compile,
)  # pylint: disable=redefined-builtin
from temporian.core.data.dtype import DType
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.core.data import duration_utils


class FilterOperator(Operator):
    def __init__(self, input: EventSetNode, condition: EventSetNode):
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


@compile
def filter(
    input: EventSetOrNode,
    condition: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    if condition is None:
        condition = input

    assert isinstance(input, EventSetNode)
    assert isinstance(condition, EventSetNode)

    return FilterOperator(input, condition).outputs["output"]


@compile
def before(
    input: EventSetOrNode, timestamp: Union[float, datetime]
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    if isinstance(timestamp, datetime):
        timestamp = duration_utils.normalize_timestamp(timestamp)
    return filter(input, input.timestamps() < timestamp)


@compile
def after(
    input: EventSetOrNode, timestamp: Union[float, datetime]
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    if isinstance(timestamp, datetime):
        timestamp = duration_utils.normalize_timestamp(timestamp)
    return filter(input, input.timestamps() > timestamp)
