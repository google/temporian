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

from typing import List, Optional
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import (
    EventSetOrNode,
    IndexKeyList,
    NormalizedIndexKey,
)
from temporian.implementation.numpy.data.dtype_normalization import (
    normalize_index_key_list,
)
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class SelectIndexValues(Operator):
    def __init__(
        self,
        input: EventSetNode,
        keys: Optional[IndexKeyList],
        number: Optional[int],
        fraction: Optional[float],
    ):
        super().__init__()

        self.add_input("input", input)

        if (keys is None) == (number is None) == (fraction is None):
            raise ValueError(
                "Exactly one of the parameters keys, number or fraction must be"
                " provided."
            )

        if keys is not None:
            normalized_keys = normalize_index_key_list(keys)
            self._keys = normalized_keys
            self.add_attribute("keys", normalized_keys)
        else:
            self._keys = None

        self._number = number
        if number is not None:
            self.add_attribute("number", number)

        self._fraction = fraction
        if fraction is not None:
            self.add_attribute("fraction", fraction)

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
                    type=pb.OperatorDef.Attribute.Type.LIST_INDEX_KEYS,
                    is_optional=True,
                ),
                pb.OperatorDef.Attribute(
                    key="number",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                    is_optional=True,
                ),
                pb.OperatorDef.Attribute(
                    key="fraction",
                    type=pb.OperatorDef.Attribute.Type.FLOAT_64,
                    is_optional=True,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def keys(self) -> Optional[List[NormalizedIndexKey]]:
        return self._keys

    @property
    def number(self) -> Optional[int]:
        return self._number

    @property
    def fraction(self) -> Optional[float]:
        return self._fraction


operator_lib.register_operator(SelectIndexValues)


@typecheck
@compile
def select_index_values(
    input: EventSetOrNode,
    keys: Optional[IndexKeyList],
    number: Optional[int],
    fraction: Optional[float],
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    return SelectIndexValues(
        input=input, keys=keys, number=number, fraction=fraction
    ).outputs["output"]
