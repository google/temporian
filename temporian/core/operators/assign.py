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

"""Assign operator."""

from typing import Dict, Optional

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class AssignOperator(Operator):
    """Assign operator."""

    def __init__(
        self,
        event_1: Event,
        event_2: Event,
        event_3: Optional[Event] = None,
        event_4: Optional[Event] = None,
    ):
        super().__init__()

        events = [event_1, event_2]
        if event_3 is not None:
            events.append(event_3)
        if event_4 is not None:
            events.append(event_4)

        # inputs
        output_features = []
        feature_names = set()
        for idx, event in enumerate(events):
            self.add_input(f"event_{idx+1}", event)
            output_features.extend(event.features())

            for f in event.features():
                if f.name() in feature_names:
                    raise ValueError(
                        f"Feature {f.name()} is defined in multiple "
                        "input events."
                    )
                feature_names.add(f.name())

            if event.sampling() is not events[0].sampling():
                raise ValueError(
                    "All the events do not have the same sampling."
                )

        # outputs
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=events[0].sampling(),
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ASSIGN",
            inputs=[
                pb.OperatorDef.Input(key="event_1"),
                pb.OperatorDef.Input(key="event_2"),
                pb.OperatorDef.Input(key="event_3", is_optional=True),
                pb.OperatorDef.Input(key="event_4", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(AssignOperator)


def assign(
    event_1: Event,
    event_2: Event,
    event_3: Optional[Event] = None,
    event_4: Optional[Event] = None,
) -> Event:
    """Concatenates together events with the same sampling.

    Example:
        event_1 = ... # Feature A & B
        event_2 = ... # Feature C & D
        event_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F
        output = np.assign(event_1, event_2, event_3)

    All the events should have the same sampling. To concatenate events with a
    different sampling, use the operator 'tp.sample(...)' before.

    Example:

        # Assume event_1, event_2 and event_3 dont have the same sampling
        event_1 = ... # Feature A & B
        event_2 = ... # Feature C & D
        event_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F, and the same sampling as
        # event_1
        output = np.assign(event_1,
            tp.sample(event_2, sampling=event_1),
            tp.sample(event_3, sampling=event_1))
    """

    return AssignOperator(event_1, event_2, event_3, event_4).outputs()["event"]
