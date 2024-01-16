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

"""Since last operator class and public API function definition."""

from typing import Optional

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.core.data.dtype import DType


class SinceLast(Operator):
    def __init__(
        self,
        input: EventSetNode,
        steps: int,
        sampling: Optional[EventSetNode] = None,
    ):
        super().__init__()

        if steps <= 0:
            raise ValueError(
                f"Number of steps must be greater than 0. Got {steps=}."
            )
        self.add_attribute("steps", steps)
        self.add_input("input", input)

        if sampling is not None:
            self.add_input("sampling", sampling)
            self._has_sampling = True
            effective_sampling_node = sampling
            input.schema.check_compatible_index(sampling.schema)

        else:
            effective_sampling_node = input
            self._has_sampling = False

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=[("since_last", DType.FLOAT64)],
                sampling_node=effective_sampling_node,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SINCE_LAST",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="steps",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                )
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling", is_optional=True),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def has_sampling(self) -> bool:
        return self._has_sampling

    @property
    def steps(self) -> int:
        steps = self.attributes["steps"]
        assert type(steps) is int  # linter typecheck
        return steps


operator_lib.register_operator(SinceLast)


@compile
def since_last(
    input: EventSetOrNode,
    steps: int,
    sampling: Optional[EventSetOrNode] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)
    if sampling is not None:
        assert isinstance(sampling, EventSetNode)

    return SinceLast(input=input, sampling=sampling, steps=steps).outputs[
        "output"
    ]
