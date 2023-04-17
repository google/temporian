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


"""UniqueTimestamps operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data.sampling import Sampling


class UniqueTimestamps(Operator):
    def __init__(self, event: Event):
        super().__init__()

        self.add_input("event", event)

        self.add_output(
            "event",
            Event(
                features=[],
                sampling=Sampling(
                    index=event.sampling.index,
                    creator=self,
                    is_unix_timestamp=event.sampling.is_unix_timestamp,
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
            inputs=[pb.OperatorDef.Input(key="event")],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(UniqueTimestamps)


def unique_timestamps(event: Event) -> Event:
    """Removes duplicated timestamps.

    unique_timestamps returns a feature-less event where each timestamps from
    "event" only appears once.

    Example:

        Inputs:
            event:
                feature_1: ['a', 'b', 'c', 'd']
                timestamps: [1, 2, 2, 4]

        Output:
            timestamps: [1, 2, 4]

    Args:
        event: An event, possibly with features, to process.

    Returns:
        An event without features with unique timestamps.
    """

    return UniqueTimestamps(event=event).outputs["event"]
