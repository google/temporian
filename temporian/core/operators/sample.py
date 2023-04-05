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

"""Sample operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Sample(Operator):
    def __init__(
        self,
        event: Event,
        sampling: Event,
    ):
        super().__init__()

        self.add_input("event", event)
        self.add_input("sampling", sampling)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name(),
                dtype=f.dtype(),
                sampling=sampling.sampling(),
                creator=self,
            )
            for f in event.features()
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=sampling.sampling(),
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SAMPLE",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="sampling"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(Sample)


def sample(
    event: Event,
    sampling: Event,
) -> Event:
    """Samples an event at each timestamp of a sampling.

    If a timestamp in 'sampling' does not have a corresponding timestamp in
    'event', the last timestamp in 'event' is used instead. If this timestamp
    is anterior to an value in 'event', the value is replaced by
    dtype.MissingValue(...).

    Example:

        Inputs:
            event:
                timestamps: 1, 5, 8, 9
                feature_1:  1.0, 2.0, 3.0, 4.0
            sampling:
                timestamps: -1, 1, 6, 10

        Output:
            timestamps: -1, 1, 6, 10
            feature_1: nan, 1.0, 2.0, 4.0

    Args:
        event: The event to sample.
        sampling: The event to use the sampling of.

    Returns:
        A sampled event, with same sampling as `sampling`.
    """

    return Sample(event=event, sampling=sampling).outputs()["event"]
