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

"""Lag operator."""
from typing import List
from typing import Union

from temporian.core import operator_lib
from temporian.core.data.duration import Duration
from temporian.core.data.duration import duration_abbreviation
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class LagOperator(Operator):
    """Lag operator."""

    def __init__(
        self,
        event: Event,
        duration: Duration,
    ):
        super().__init__()

        # inputs
        self.add_input("event", event)

        self.add_attribute("duration", duration)

        output_sampling = Sampling(index=event.sampling().index(), creator=self)

        prefix = "lag" if duration > 0 else "leak"
        duration_str = duration_abbreviation(duration)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{prefix}[{duration_str}]_{f.name()}",
                dtype=f.dtype(),
                sampling=output_sampling,
                creator=self,
            )
            for f in event.features()
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="LAG",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="duration",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="event")],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(LagOperator)


def _implementation(
    event: Event,
    duration: Union[Duration, List[Duration]],
    should_leak: bool = False,
) -> Event:
    """Lag & Leak Implementation."""

    if not isinstance(duration, list):
        duration = [duration]

    if not all(isinstance(d, (int, float)) and d > 0 for d in duration):
        raise ValueError(
            "duration must be a list of positive numbers. Got"
            f" {duration}, type {type(duration)}"
        )

    # Ensure that all durations are of type Duration. This converts ints to
    # float64 for consistent behavior.
    duration = [
        Duration(d) if not isinstance(d, Duration) else d for d in duration
    ]

    used_duration = duration if not should_leak else [-d for d in duration]

    if len(used_duration) == 1:
        return LagOperator(
            event=event,
            duration=used_duration[0],
        ).outputs()["event"]

    return [
        LagOperator(event=event, duration=d).outputs()["event"]
        for d in used_duration
    ]


def lag(
    event: Event, duration: Union[Duration, List[Duration]]
) -> Union[Event, List[Event]]:
    """Lag operator. Shifts the event sampling backwards in time by a specified
    duration.

    Args:
        event: Event to lag.
        duration: Duration to lag by. Can be a list of Durations.

    Returns:
        Lagged event. If a list of Durations is provided, a list of lagged
        events is returned.
    """
    return _implementation(event=event, duration=duration)


def leak(
    event: Event, duration: Union[Duration, List[Duration]]
) -> Union[Event, List[Event]]:
    """Leak operator. Shifts the event sampling forward in time by a specified
    duration.

    Args:
        event: Event to leak.
        duration: Duration to shift the sampling. Can be a list of Durations.

    Returns:
        Leaked event. If a list of Durations is provided, a list of leaked
        events is returned.
    """
    return _implementation(event=event, duration=duration, should_leak=True)
