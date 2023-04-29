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

from typing import Optional

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data.dtype import DType


class SinceLast(Operator):
    def __init__(
        self,
        event: Event,
        sampling: Optional[Event] = None,
    ):
        super().__init__()

        self.add_input("event", event)

        if sampling is not None:
            self.add_input("sampling", sampling)
            self._has_sampling = True
            effective_sampling = sampling.sampling

            if event.sampling.index != sampling.sampling.index:
                raise ValueError(
                    "Event and sampling do not have the same index."
                    f" {event.sampling.index} != {sampling.sampling.index}"
                )

        else:
            effective_sampling = event.sampling
            self._has_sampling = False

        output_features = [
            Feature(
                name="since_last",
                dtype=DType.FLOAT64,
                sampling=effective_sampling,
                creator=self,
            )
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=effective_sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SINCE_LAST",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="event"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )

    @property
    def has_sampling(self) -> bool:
        return self._has_sampling


operator_lib.register_operator(SinceLast)


def since_last(
    event: Event,
    sampling: Optional[Event] = None,
) -> Event:
    """Amount of time since the last distinct timestamp.

    Example 1:
        ```
        Inputs:
            event:
                timestamps: 1, 5, 8, 8, 9

        Output:
            since_last: NaN, 4, 3, 0, 1
            timestamps: 1, 5, 8, 8, 9
        ```

    Example 2:
        ```
        Inputs:
            event:
                timestamps: 1, 5, 8, 9
            sampling:
                timestamps: -1, 1, 6, 10

        Output:
            since_last: NaN, 0, 1, 1
            timestamps: -1, 1, 5, 6, 10
        ```

    Args:
        event: Event to sample.
        sampling: Event to use the sampling of.

    Returns:
        Sampled event, with same sampling as `sampling`.
    """

    return SinceLast(event=event, sampling=sampling).outputs["event"]
