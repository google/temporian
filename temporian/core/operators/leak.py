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

"""Lag operator class and public API function definitions."""


from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.duration_utils import (
    Duration,
    NormalizedDuration,
    normalize_duration,
)
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class LeakOperator(Operator):
    def __init__(
        self,
        input: EventSetNode,
        duration: NormalizedDuration,
    ):
        super().__init__()

        self._duration = duration

        self.add_input("input", input)
        self.add_attribute("duration", duration)

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

    @property
    def duration(self) -> Duration:
        return self._duration

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="LEAK",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="duration",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(LeakOperator)


@compile
def leak(input: EventSetNode, duration: Duration) -> EventSetNode:
    """Subtracts a duration from an EventSetNode's timestamps.

    In other words, shifts the timestamp values backward in time.

    Note that this operator moves future data into the past, and should be used
    with caution to prevent unwanted future leakage. For instance, this op
    should generally not be used to compute the input features of a model.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[0, 1, 5, 6],
        ...     features={"value": [0, 1, 5, 6]},
        ... )

        >>> b = tp.leak(a, tp.duration.seconds(2))
        >>> b
        indexes: ...
            (4 events):
                timestamps: [-2. -1. 3. 4.]
                'value': [0 1 5 6]
        ...

        ```

    Args:
        input: EventSetNode to leak.
        duration: Duration to leak by.

    Returns:
        Leaked node.
    """

    normalized_duration = normalize_duration(duration)

    return LeakOperator(
        input=input,
        duration=normalized_duration,
    ).outputs["output"]
