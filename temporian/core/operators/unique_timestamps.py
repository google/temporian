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


"""Unique timestamps operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class UniqueTimestamps(Operator):
    def __init__(self, input: Node):
        super().__init__()

        self.add_input("input", input)

        self.add_output(
            "output",
            Node(
                features=[],
                sampling=Sampling(
                    index_levels=input.sampling.index,
                    creator=self,
                    is_unix_timestamp=input.sampling.is_unix_timestamp,
                ),
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="UNIQUE_TIMESTAMPS",
            attributes=[],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(UniqueTimestamps)


def unique_timestamps(input: Node) -> Node:
    """Removes duplicated timestamps.

    Returns a feature-less node where each timestamps from `input` only appears
    once. If the input is indexed, the unique operation is applied independently
    for each index.

    Example:

        Inputs:
            input:
                feature_1: ['a', 'b', 'c', 'd']
                timestamps: [1, 2, 2, 4]

        Output:
            timestamps: [1, 2, 4]

    Args:
        input: Node, possibly with features, to process.

    Returns:
        Node without features with unique timestamps in `input`.
    """

    return UniqueTimestamps(input=input).outputs["output"]
