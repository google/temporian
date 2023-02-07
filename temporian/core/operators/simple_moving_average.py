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


class SimpleMovingAverage(Operator):
    """Simple moving average operator."""

    def __init__(
        self,
        data: Event,
        window_length: str,
        sampling: Optional[Event] = None,
    ):
        super().__init__()

        self.add_attribute("window_length", window_length)

        if sampling is not None:
            self.add_input("sampling", sampling)
        else:
            sampling = data.sampling()

        self.add_input("data", data)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"sma_{f.name()}",
                dtype=f.dtype(),
                sampling=sampling,
                creator=self,
            )
            for f in data.features()
        ]

        self.add_output(
            "output",
            Event(
                features=output_features,
                sampling=sampling,
                creator=self,
            ),
        )

        self.check()

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
                pb.OperatorDef.Input(key="data"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SimpleMovingAverage)


def sma(
    data: Event,
    window_length: str,
    sampling: Optional[Event] = None,
) -> Event:
    return SimpleMovingAverage(
        data=data,
        window_length=window_length,
        sampling=sampling,
    ).outputs()["output"]
