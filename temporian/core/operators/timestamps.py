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
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.compilation import compile
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.core.data import dtype


class Timestamps(Operator):
    def __init__(self, input: EventSetNode):
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


@compile
def timestamps(input: EventSetOrNode) -> EventSetOrNode:
    """Converts an [`EventSet`][temporian.EventSet]'s timestamps into a
    `float64` feature.

    Features in the input EventSet are ignored, only the timestamps are used.

    Datetime timestamps are converted to unix timestamps.

    Integer timestamps example:
        ```python
        >>> from datetime import datetime
        >>> a = tp.event_set(timestamps=[1, 2, 3, 5])
        >>> b = tp.timestamps(a)
        >>> b
        indexes: []
        features: [('timestamps', float64)]
        events:
            (4 events):
                timestamps: [1. 2. 3. 5.]
                'timestamps': [1. 2. 3. 5.]
        ...

        ```

    Unix timestamps and filter example:
        ```python
        >>> from datetime import datetime
        >>> a = tp.event_set(
        ...    timestamps=[datetime(1970,1,1,0,0,30), datetime(2023,1,1,1,0,0)],
        ... )
        >>> b = tp.timestamps(a)

        >>> # Filter using the timestamps
        >>> max_date = datetime(2020, 1, 1).timestamp()
        >>> c = b.filter(b < max_date)

        >>> # Operate like any other feature
        >>> d = c * 5
        >>> e = tp.glue(c.rename('filtered'), d.rename('multiplied'))
        >>> e
        indexes: []
        features: [('filtered', float64), ('multiplied', float64)]
        events:
            (1 events):
                timestamps: [30.]
                'filtered': [30.]
                'multiplied': [150.]
        ...

        ```

    Args:
        input: EventSet to get the timestamps from.

    Returns:
        EventSet with a single feature named `timestamps` with each event's
        timestamp.
    """
    assert isinstance(input, EventSetNode)

    return Timestamps(input=input).outputs["output"]
