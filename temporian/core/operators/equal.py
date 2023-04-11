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

"""Equal operator."""
from temporian.core import operator_lib
from temporian.core.data.dtype import BOOLEAN
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class EqualOperator(Operator):
    """Equal operator."""

    def __init__(self, event: Event, value: any):
        super().__init__()

        self.add_attribute("value", value)

        # inputs
        self.add_input("event", event)

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f"{f.name}_equal_{value}",
                dtype=BOOLEAN,
                sampling=f.sampling,
                creator=self,
            )
            for f in event.features
        ]

        output_sampling = event.sampling
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="EQUAL",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="value",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(EqualOperator)


def equal(event: Event, value: any) -> Event:
    """Equal operator."""
    return EqualOperator(event, value).outputs["event"]
