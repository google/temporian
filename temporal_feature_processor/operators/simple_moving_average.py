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

from temporal_feature_processor import core_pb2 as pb
from temporal_feature_processor import operator_lib
from temporal_feature_processor.data.event import Event
from temporal_feature_processor.data.feature import Feature
from temporal_feature_processor.implementation.pandas.operators.base import (
    PandasOperator,
)
from temporal_feature_processor.implementation.pandas.operators.window.simple_moving_average import (
    PandasSimpleMovingAverageOperator,
)
from temporal_feature_processor.operators.base import Operator


class SimpleMovingAverage(Operator):
    """Simple moving average operator."""

    def __init__(
        self,
        data: Event,
        window_length: int,
        sampling: Optional[Event] = None,
    ):
        super().__init__()

        self.window_length = window_length

        if sampling is not None:
            self.add_input("sampling", sampling)
        else:
            sampling = data.sampling()

        self.add_input("data", data)

        features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name(),
                dtype=f.dtype(),
                sampling=sampling,
            )
            for f in data.features()
        ]

        self.add_output(
            "output",
            Event(
                features=features,
                sampling=sampling,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SIMPLE_MOVING_AVERAGE",
            inputs=[
                pb.OperatorDef.Input(key="data"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def _get_pandas_implementation(self) -> PandasOperator:
        return PandasSimpleMovingAverageOperator(window_length=self.window_length)


operator_lib.register_operator(SimpleMovingAverage)


def sma(
    data: Event,
    window_length: int,
    sampling: Optional[Event] = None,
) -> Event:
    return SimpleMovingAverage(
        data=data,
        window_length=window_length,
        sampling=sampling,
    ).outputs()["output"]
