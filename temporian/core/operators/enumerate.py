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


"""Enumerate operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.node import (
    Node,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data import dtype


class Enumerate(Operator):
    def __init__(self, input: Node, name: str):
        super().__init__()

        self.add_input("input", input)
        self.add_attribute("name", name)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[(name, dtype.int64)],
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ENUMERATE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="name",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Enumerate)


def enumerate(input: Node, name: str) -> Node:
    """Create an `int64` feature with the ordinal position of each event.

    Each index is enumerated independently.

    Usage:
        ```python
        >>> evset = tp.event_set(
        ...    timestamps=[-1, 2, 3, 5, 0],
        ...    features={"a": ["A", "A", "A", "A", "B"]},
        ...    indexes=["a"],
        ...    name='empty_features'
        ... )
        >>> tp.enumerate(evset.node(), name="enumerate_result").run(evset)
        indexes: [('a', str_)]
        features: [('enumerate_result', int64)]
        events:
            (5 events):
                timestamps: [-1. 2. 3. 5. 0.]
                'enumerate_result': [0. 1. 2. 3. 0.]
        ...

        ```

    Args:
        input: Node to enumerate.
        name: Name for the feature with the enumeration result.

    Returns:
        Single feature with each event's ordinal position in index.
    """

    return Enumerate(input=input, name=name).outputs["output"]
