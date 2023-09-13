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


"""FilterMaxMovingCount operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck
from temporian.core.data.duration_utils import (
    Duration,
    normalize_duration,
    NormalizedDuration,
)


class FilterMaxMovingCount(Operator):
    def __init__(
        self,
        input: EventSetNode,
        window_length: NormalizedDuration,
    ):
        super().__init__()

        self.add_input("input", input)
        self.add_attribute("window_length", window_length)

        self._window_length = window_length

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
    def window_length(self) -> NormalizedDuration:
        return self._window_length

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="FILTER_MAX_MOVING_COUNT",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="window_length",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(FilterMaxMovingCount)


@typecheck
@compile
def filter_moving_count(
    input: EventSetOrNode, window_length: Duration
) -> EventSetOrNode:
    return FilterMaxMovingCount(
        input=input,
        window_length=normalize_duration(window_length),
    ).outputs["output"]
