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


"""Map operator class and public API function definitions."""

from inspect import signature
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode, MapFunction
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class Map(Operator):
    def __init__(self, input: EventSetNode, func: MapFunction):
        super().__init__()

        if len(signature(func).parameters) > 2:
            raise ValueError("`func` must receive at most 2 arguments.")

        self.add_attribute("func", func)
        self._func = func

        self.add_input("input", input)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=input.schema.features,
                sampling_node=input,
                creator=self,
            ),
        )

        self.check()

    @property
    def func(self) -> MapFunction:
        return self._func

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="MAP",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="func",
                    type=pb.OperatorDef.Attribute.Type.CALLABLE,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Map)


@typecheck
@compile
def map(input: EventSetOrNode, func: MapFunction) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return Map(input=input, func=func).outputs["output"]
