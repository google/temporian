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

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class AssignOperator(Operator):
    """Simple moving average operator."""

    def __init__(
        self,
        left_event: Event,
        right_event: Event,
    ):
        super().__init__()

        # inputs
        self.add_input("left_event", left_event)
        self.add_input("right_event", right_event)

        # outputs
        output_features = left_event.features() + [
            Feature(name=feature.name(), dtype=feature.dtype(), creator=self)
            for feature in right_event.features()
        ]
        output_sampling = left_event.sampling()
        self.add_output(
            "event",
            Event(
                features=output_features, sampling=output_sampling, creator=self
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ASSIGN",
            inputs=[
                pb.OperatorDef.Input(key="left_event"),
                pb.OperatorDef.Input(key="right_event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(AssignOperator)


def assign(
    left_event: Event,
    right_event: Event,
) -> Event:
    return AssignOperator(left_event, right_event).outputs()["event"]
