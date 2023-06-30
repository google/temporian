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


"""Timestamps operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.node import (
    Node,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data import dtype


class Timestamps(Operator):
    def __init__(self, input: Node):
        super().__init__()

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[("timestamps", dtype.float64)],
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TIMESTAMPS",
            attributes=[],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Timestamps)


def timestamps(input: Node) -> Node:
    """Create a `float64` feature from the timestamps of an event.

    Features in the input node are ignored, only the timestamps are used.
    Datetime timestamps are converted to unix timestamps.

    Integer timestamps example:
        ```python
        >>> from datetime import datetime
        >>> evset = tp.event_set(
        ...    timestamps=[1, 2, 3, 5],
        ...    name='simple_timestamps'
        ... )
        >>> tp.timestamps(evset.node()).run(evset)
        indexes: []
        features: [('timestamps', float64)]
        events:
            (4 events):
                timestamps: [1. 2. 3. 5.]
                'timestamps': [1. 2. 3. 5.]
        ...

        ```

    Unix timestamps example:
        ```python
        >>> from datetime import datetime
        >>> evset = tp.event_set(
        ...    timestamps=[datetime(1970,1,1,0,0,30), datetime(1970,1,1,1,0,0)],
        ...    name='old_times'
        ... )
        >>> tp.timestamps(evset.node()).run(evset)
        indexes: []
        features: [('timestamps', float64)]
        events:
            (2 events):
                timestamps: [ 30. 3600.]
                'timestamps': [ 30. 3600.]
        ...

        ```

    Args:
        input: Node to get the timestamps from.

    Returns:
        Single feature `timestamps` with each event's timestamp value.
    """

    return Timestamps(input=input).outputs["output"]
