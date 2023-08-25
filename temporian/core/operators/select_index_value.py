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


"""SelectIndexValue operator class and public API function definitions."""

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import EventSetNode, create_node_new_features_new_sampling
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class SelectIndexValue(Operator):
    def __init__(self, input: EventSetNode, param: float):
        super().__init__()

        self.add_input("input", input)
        self.add_attribute("param", param)

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

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SELECT_INDEX_VALUE",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="param",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SelectIndexValue)


@typecheck
@compile
def select_index_value(input: EventSetOrNode, param: float) -> EventSetOrNode:
    """<Text>

    Args:
        input: <Text>
        param: <Text>

    Example:

        ```python
        >>> a = tp.event_set(timestamps=[0, 1, 2], features={"A": [0, 10, 20]})
        >>> b = tp.select_index_value(a)
        >>> b
        indexes: []
        features: [('A', int64)]
        events:
            (3 events):
                timestamps: [0. 1. 2.]
                'A': [ 0 10 20]
        ...

        ```

    Returns:
        <Text>
    """

    return SelectIndexValue(input=input, param=param).outputs["output"]  # type: ignore

