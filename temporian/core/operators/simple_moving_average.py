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

"""Simple moving average operator."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data import duration
from temporian.core.data.duration import Duration


class SimpleMovingAverage(Operator):
    """Simple moving average operator."""

    def __init__(
        self,
        event: Event,
        window_length: Duration,
        sampling: Optional[Event],
    ):
        super().__init__()

        self._window_length = window_length
        self.add_attribute("window_length", window_length)

        if sampling is not None:
            self.add_input("sampling", sampling)
        else:
            sampling = event.sampling()

        self.add_input("event", event)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"sma_{f.name()}",
                dtype=f.dtype(),
                sampling=sampling,
                creator=self,
            )
            for f in event.features()
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=sampling,
                creator=self,
            ),
        )

        self.check()

    def window_length(self):
        return self._window_length

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SIMPLE_MOVING_AVERAGE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="window_length",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(SimpleMovingAverage)


def sma(
    event: Event,
    window_length: Duration,
    sampling: Optional[Event] = None,
) -> Event:
    """Simple Moving average

    For each sampling, and for each feature independently, returns at time "t"
    the average value of the feature in the time windows [t-window, t].

    If "sampling" is provided, applies the operator for each timestamps of
    "sampling". If "sampling" is not provided, apply the operator for each
    timestamps of "event".

    Args:
        event: The features to average.
        window_length: The window length for averaging.
        sampling: If provided, define when the operator is applied. If not
          provided, the operator is applied for each timestamp of "event".

    Returns:
        An event containing the moving average of each feature in "event".
    """
    return SimpleMovingAverage(
        event=event,
        window_length=window_length,
        sampling=sampling,
    ).outputs()["event"]
