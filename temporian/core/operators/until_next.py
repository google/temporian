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


"""UntilNext operator class and public API function definitions."""

import math

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
from temporian.core.data.dtype import DType
from temporian.core.data.duration_utils import (
    Duration,
    NormalizedDuration,
    normalize_duration,
)


class UntilNext(Operator):
    def __init__(
        self,
        input: EventSetNode,
        sampling: EventSetNode,
        timeout: NormalizedDuration,
    ):
        super().__init__()

        self.add_input("input", input)
        self.add_input("sampling", sampling)

        self.add_attribute("timeout", timeout)
        self._timeout = timeout

        if not math.isfinite(timeout):
            raise ValueError(
                f"Timeout should be finite. Instead, got {timeout}"
            )

        if timeout <= 0:
            raise ValueError(
                f"Timeout should be strictly positive. Instead, got {timeout}"
            )

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[("until_next", DType.FLOAT64)],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @property
    def timeout(self) -> float:
        return self._timeout

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="UNTIL_NEXT",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="timeout",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(UntilNext)


@typecheck
@compile
def until_next(
    input: EventSetOrNode,
    sampling: EventSetNode,
    timeout: Duration,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return UntilNext(
        input=input,
        sampling=sampling,
        timeout=normalize_duration(timeout),
    ).outputs["output"]
