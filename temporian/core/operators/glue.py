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

"""Glue operator."""

from typing import Dict, Optional, List

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb

# Maximum number of arguments taken by the glue operator
MAX_NUM_ARGUMENTS = 10


class GlueOperator(Operator):
    """Glue operator."""

    def __init__(
        self,
        **dict_events: Dict[str, Event],
    ):
        super().__init__()

        # Note: Support for dictionaries of events is required for
        # serialization.

        if len(dict_events) < 2:
            raise ValueError("At least two arguments should be provided")

        if len(dict_events) >= MAX_NUM_ARGUMENTS:
            raise ValueError(
                f"Too much (>{MAX_NUM_ARGUMENTS}) arguments provided"
            )

        # inputs
        output_features = []
        feature_names = set()
        first_sampling = None
        for key, event in dict_events.items():
            self.add_input(key, event)
            output_features.extend(event.features())

            for f in event.features():
                if f.name() in feature_names:
                    raise ValueError(
                        f"Feature {f.name()} is defined in multiple "
                        "input events."
                    )
                feature_names.add(f.name())

            if first_sampling is None:
                first_sampling = event.sampling()
            elif event.sampling() is not first_sampling:
                raise ValueError(
                    "All the events do not have the same sampling."
                )

        # outputs
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=first_sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="GLUE",
            # TODO: Add support to array of events arguments.
            inputs=[
                pb.OperatorDef.Input(key=f"event_{idx}", is_optional=idx >= 2)
                for idx in range(MAX_NUM_ARGUMENTS)
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(GlueOperator)


def glue(
    *events: List[Event],
) -> Event:
    """Concatenates together events with the same sampling.

    Example:

        ```
        event_1 = ... # Feature A & B
        event_2 = ... # Feature C & D
        event_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F
        output = np.glue(event_1, event_2, event_3)
        ```

    All the events should have the same sampling. To concatenate events with a
    different sampling, use the operator 'tp.sample(...)' before.

    Example:

        ```
        # Assume event_1, event_2 and event_3 dont have the same sampling
        event_1 = ... # Feature A & B
        event_2 = ... # Feature C & D
        event_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F, and the same sampling as
        # event_1
        output = np.glue(event_1,
            tp.sample(event_2, sampling=event_1),
            tp.sample(event_3, sampling=event_1))
        ```
    """

    # Note: The event should be called "event_{idx}" with idx in [0, MAX_NUM_ARGUMENTS).
    dict_events = {f"event_{idx}": event for idx, event in enumerate(events)}
    return GlueOperator(**dict_events).outputs()["event"]
