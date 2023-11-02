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

"""Add and set index operators classes and public API function definitions."""

from typing import List, Union

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.data.schema import FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.core.operators.drop_index import drop_index


class AddIndexOperator(Operator):
    def __init__(
        self,
        input: EventSetNode,
        indexes: List[str],
    ) -> None:
        super().__init__()

        output_feature_schemas = self._get_output_feature_schemas(
            input, indexes
        )
        output_indexes = self._get_output_indexes(input, indexes)
        self._indexes = indexes

        self.add_input("input", input)

        self.add_attribute("indexes", indexes)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=output_feature_schemas,
                indexes=output_indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )
        self.check()

    def _get_output_feature_schemas(
        self, input: EventSetNode, indexes: List[str]
    ) -> List[FeatureSchema]:
        return [
            feature
            for feature in input.schema.features
            if feature.name not in indexes
        ]

    def _get_output_indexes(
        self, input: EventSetNode, indexes: List[str]
    ) -> List[IndexSchema]:
        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_indexes: List[IndexSchema] = []
        for index in indexes:
            if index in index_dict:
                raise ValueError(f"{index} is already an index in input.")

            if index not in feature_dict:
                raise ValueError(f"{index} is not a feature in input.")

            new_indexes.append(
                IndexSchema(name=index, dtype=feature_dict[index])
            )
        # Note: The new indexes are added after the existing ones.
        return input.schema.indexes + new_indexes

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ADD_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="indexes",
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def indexes(self) -> List[str]:
        return self._indexes


operator_lib.register_operator(AddIndexOperator)


def _normalize_indexes(
    indexes: Union[str, List[str]],
) -> List[str]:
    if isinstance(indexes, str):
        return [indexes]

    if len(indexes) == 0:
        raise ValueError("Cannot specify empty list as `indexes` argument.")

    return indexes


def _normalize_indexes_to_set(
    indexes: Union[str, List[str]],
) -> List[str]:
    if isinstance(indexes, str):
        return [indexes]

    if len(indexes) == 0:
        raise ValueError("Cannot specify empty list as `indexes` argument.")

    return indexes


@compile
def add_index(
    input: EventSetOrNode, indexes: Union[str, List[str]]
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    indexes = _normalize_indexes(indexes)
    return AddIndexOperator(input, indexes).outputs["output"]


@compile
def set_index(
    input: EventSetOrNode, indexes: Union[str, List[str]]
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    indexes = _normalize_indexes_to_set(indexes)

    # Note
    # The set_index is implemented as a drop_index + add_index.
    # The implementation could be improved (simpler, faster) with a new
    # operator to re-order the indexes.

    if len(input.schema.indexes) != 0:
        input = drop_index(input)

    if len(indexes) != 0:
        input = add_index(input, indexes)

    return input
