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

"""Prefix operator."""

from typing import List, Union, Any, Dict

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Prefix(Operator):
    """Prefix operator."""

    def __init__(
        self,
        prefix: str,
        event: Event,
    ):
        super().__init__()

        self.add_attribute("prefix", prefix)
        self.add_input("event", event)

        # TODO: When supported, re-use existing feature instead of creating a
        # new one.
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=prefix + f.name(),
                dtype=f.dtype(),
                sampling=event.sampling(),
                creator=self,
            )
            for f in event.features()
        ]

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=event.sampling(),
                creator=self,
            ),
        )

        self.check()

    def prefix(self):
        return self.attributes()["prefix"]

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="PREFIX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="prefix",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                )
            ],
            inputs=[pb.OperatorDef.Input(key="event")],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(Prefix)


def prefix(
    prefix: str,
    event: Event,
) -> Event:
    """Add a prefix to the feature names.

    Example:

        Inputs:
            prefix: "hello_"
            event:
                feature_1: ...
                feature_2: ...

        Output:
            hello_feature_1: ...
            hello_feature_2: ...

    Args:
        prefix: Prefix to add in front of the feature names.
        event: Event to prefix.

    Returns:
        Prefixed event.
    """

    return Prefix(prefix=prefix, event=event).outputs()["event"]
