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

"""Sum operator."""

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


from enum import Enum


class Resolution(str, Enum):
    PER_FEATURE_IDX = "PER_FEATURE_IDX"
    PER_FEATURE_NAME = "PER_FEATURE_NAME"


class SumOperator(Operator):
    """Sum operator."""

    def __init__(
        self,
        event_1: Event,
        event_2: Event,
        resolution: Resolution = Resolution.PER_FEATURE_IDX,
    ):
        super().__init__()

        # inputs
        self.add_input("event_1", event_1)
        self.add_input("event_2", event_2)

        self.add_attribute("resolution", resolution)

        if event_1.sampling() != event_2.sampling():
            raise ValueError("event_1 and event_2 must have same sampling.")

        if len(event_1.features()) != len(event_2.features()):
            raise ValueError(
                "event_1 and event_2 must have same number of features."
            )

        sampling = event_1.sampling()

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"sum_{event_1_f.name()}_{event_2_f.name()}",
                dtype=event_1_f.dtype(),
                sampling=sampling,
                creator=self,
            )
            for event_1_f, event_2_f in zip(
                event_1.features(), event_2.features()
            )
        ]

        self.add_output(
            "output",
            Event(
                features=output_features,
                sampling=sampling,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SUM",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="resolution",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event_1"),
                pb.OperatorDef.Input(key="event_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SumOperator)


def sum(
    event_1: Event,
    event_2: Event,
    resolution: Resolution = Resolution.PER_FEATURE_IDX,
) -> Event:
    return SumOperator(
        event_1=event_1, event_2=event_2, resolution=resolution
    ).outputs()["output"]
