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

from typing import Optional

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class SumOperator(Operator):
    """Sum operator."""

    def __init__(
        self,
        data_1: Event,
        data_2: Event,
        resolution: Optional[str] = None,
    ):
        super().__init__()

        # inputs
        self.add_input("data_1_event", data_1)
        self.add_input("data_2_event", data_2)

        if resolution is not None:
            self.add_attribute("resolution", resolution)

        sampling = data_1.sampling()

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"sum_{data_1.features()[f].name()}_{data_2.features()[f].name()}",
                dtype=f.dtype(),
                sampling=sampling,
                creator=self,
            )
            for f in range(
                data_1.shape[1]
            )  # assuming data_1 and data_2 have the same shape
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
                    is_optional=True,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="data_1_event"),
                pb.OperatorDef.Input(key="data_2_event"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SumOperator)


def sum(
    data_1: Event,
    data_2: Event,
    resolution: Optional[str] = None,
) -> Event:
    return SumOperator(
        data_1=data_1, data_2=data_2, resolution=resolution
    ).outputs()["output"]
