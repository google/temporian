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
from temporian.core.data.node import Node, create_node_new_features_new_sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class UniqueTimestamps(Operator):
    def __init__(self, input: Node):
        super().__init__()

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
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
    """Removes events with duplicated timestamps from a Node.

    Returns a feature-less node where each timestamps from `input` only appears
    once. If the input is indexed, the unique operation is applied independently
    for each index.

    Usage example:
        ```python
        >>> a_evset = tp.event_set(timestamps=[5, 9, 9, 16], features={'f': [1,2,3,4]})
        >>> a = a_evset.node()

        >>> result = tp.unique_timestamps(a)
        >>> result.run({a: a_evset})
        indexes: []
        features: []
        events:
             (3 events):
                timestamps: [ 5. 9. 16.]
        ...

    Args:
        input: Node, possibly with features, to process.

    Returns:
        Node without features with unique timestamps in `input`.
    """

    return UniqueTimestamps(input=input).outputs["output"]
