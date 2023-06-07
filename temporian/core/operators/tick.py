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


"""Tick operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.data.node import Node, create_node_new_features_new_sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data.duration import (
    Duration,
    NormalizedDuration,
    normalize_duration,
)


class Tick(Operator):
    def __init__(
        self, input: Node, interval: NormalizedDuration, rounding: bool
    ):
        super().__init__()

        self._interval = interval
        self._rounding = rounding

        self.add_input("input", input)
        self.add_attribute("interval", interval)
        self.add_attribute("rounding", rounding)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @property
    def interval(self) -> NormalizedDuration:
        return self._interval

    @property
    def rounding(self) -> bool:
        return self._rounding

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="TICK",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="interval",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                ),
                pb.OperatorDef.Attribute(
                    key="rounding",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Tick)


def tick(input: Node, interval: Duration, rounding: bool = True) -> Node:
    """Generates timestamps at regular intervaled in the range of a guide.

    Args:
        input: Guide node. The start and end time boundaries to generate the new
            timestamps are defined by the range of timestamps in `input`.
        interval: Tick interval.
        rounding: If false, the first tick is generated at the first timestamp
            (similar to "tp.begin"). If true (default), ticks are generated on
            timestamps that are multiple of "interval".

    Example #1:
        Input
            events: [1, 5.5, 5.6, 8.6]
        Argument
            interval: 4
            rounding: false
        Output
            timestamp: 1, 5

    Example #2:
        Input
            events: [1, 5.5, 5.6, 8.6]
        Argument
            interval: 4
            rounding: true
        Output
            timestamp: 4, 8

    Returns:
        A feature-less node with regular timestamps.
    """

    return Tick(
        input=input, interval=normalize_duration(interval), rounding=rounding
    ).outputs["output"]
