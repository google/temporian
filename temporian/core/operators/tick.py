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
from temporian.core.compilation import compile
from temporian.core.data.duration_utils import (
    Duration,
    NormalizedDuration,
    normalize_duration,
)
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb


class Tick(Operator):
    def __init__(
        self,
        input: EventSetNode,
        interval: NormalizedDuration,
        align: bool,
        after_last: bool = True,
        before_first: bool = False,
    ):
        super().__init__()

        self._interval = interval
        self._align = align
        self._after_last = after_last
        self._before_first = before_first

        self.add_input("input", input)
        self.add_attribute("interval", interval)
        self.add_attribute("align", align)
        self.add_attribute("after_last", after_last)
        self.add_attribute("before_first", before_first)

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
    def align(self) -> bool:
        return self._align

    @property
    def after_last(self) -> bool:
        return self._after_last

    @property
    def before_first(self) -> bool:
        return self._before_first

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
                    key="align",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
                pb.OperatorDef.Attribute(
                    key="after_last",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
                pb.OperatorDef.Attribute(
                    key="before_first",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Tick)


# TODO: Add support for begin/end arguments.
@compile
def tick(
    input: EventSetOrNode,
    interval: Duration,
    align: bool = True,
    after_last: bool = True,
    before_first: bool = False,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return Tick(
        input=input,
        interval=normalize_duration(interval),
        align=align,
        after_last=after_last,
        before_first=before_first,
    ).outputs["output"]
