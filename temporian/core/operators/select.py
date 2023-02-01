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

"""Select operator."""
from typing import List, Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class SelectOperator(Operator):
    """Select operator."""

    def __init__(
        self, input_event: Event, feature_names: Union[str, List[str]]
    ):
        super().__init__()

        # inputs
        self.add_input("input_event", input_event)

        # outputs
        output_features = [
            Feature(
                name=feature.name(),
                dtype=feature.dtype(),
                sampling=feature.sampling(),
                creator=self,
            )
            for feature in input_event.features()
            if feature.name() in feature_names
        ]
        output_sampling = input_event.sampling()
        self.add_output(
            "output_event",
            Event(features=output_features, sampling=output_sampling),
        )
        # feature names
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        self.add_attribute("feature_names", feature_names)

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SELECT",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="feature_names",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input_event"),
            ],
            outputs=[pb.OperatorDef.Output(key="output_event")],
        )


operator_lib.register_operator(SelectOperator)


def select(
    input_event: Event,
    feature_names: List[str],
) -> Event:
    return SelectOperator(input_event, feature_names).outputs()["output_event"]
