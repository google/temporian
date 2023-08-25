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


"""SelectIndexValues operator class and public API function definitions."""

from typing import List, Optional, Union
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.implementation.numpy.data.event_set import IndexKey
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class SelectIndexValues(Operator):
    def __init__(self, input: EventSetNode, keys: Optional[List[IndexKey]]):
        super().__init__()

        self.add_input("input", input)

        self._keys = keys
        if keys:
            self.add_attribute("keys", keys)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=input.schema.features,
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SELECT_INDEX_VALUES",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="keys",
                    type=pb.OperatorDef.Attribute.Type.LIST_INDEX_VALUES,
                    is_optional=True,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def keys(self) -> Optional[List[IndexValue]]:
        return self._keys


operator_lib.register_operator(SelectIndexValues)


@typecheck
@compile
def select_index_values(
    input: EventSetOrNode,
    keys: Optional[Union[IndexKey, List[IndexKey]]] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    if isinstance(keys, list) and all(isinstance(k, tuple) for k in keys):
        pass
    elif isinstance(keys, tuple):
        keys = [keys]
    else:
        raise TypeError(
            "Unexpected type for keys. Expect a tuple or list of"
            f" tuples. Got '{keys}' instead."
        )

    return SelectIndexValues(input=input, keys=keys).outputs["output"]
