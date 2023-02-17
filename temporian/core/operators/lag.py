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

from temporian.core import operator_lib
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
        duration: float,
    ):
        super().__init__()

        # inputs
        self.add_input("event", event)

        self.add_attribute("duration", duration)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"lag_{f.name()}",
                dtype=f.dtype(),
                sampling=Sampling(index=event.sampling().index, creator=self),
                creator=self,
            )
            for f in event.features()
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=Sampling(index=event.sampling().index, creator=self),
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
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(LagOperator)


def lag(event: Event, duration: float) -> Event:
    return LagOperator(
        event=event,
        duration=duration,
    ).outputs()["event"]
