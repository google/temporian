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

"""Set index operator class and public API function definition."""

from typing import List, Union

from temporian.core import operator_lib
from temporian.core.data.node import Node, create_node_new_features_new_sampling
from temporian.core.data.schema import FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.operators.drop_index import drop_index


class AddIndexOperator(Operator):
    def __init__(
        self,
        input: Node,
        index_to_add: List[str],
    ) -> None:
        super().__init__()

        self._output_feature_schemas = self._get_output_feature_schemas(
            input, index_to_add
        )
        self._output_indexes = self._get_output_indexes(input, index_to_add)
        self._index_to_add = index_to_add

        self.add_input("input", input)

        self.add_attribute("index_to_add", index_to_add)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=self._output_feature_schemas,
                indexes=self._output_indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )
        self.check()

    def _get_output_feature_schemas(
        self, input: Node, index_to_add: List[str]
    ) -> List[FeatureSchema]:
        return [
            feature
            for feature in input.schema.features
            if feature.name not in index_to_add
        ]

    def _get_output_indexes(
        self, input: Node, index_to_add: List[str]
    ) -> List[IndexSchema]:
        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_indexes: List[IndexSchema] = []
        for index_name in index_to_add:
            if index_name not in feature_dict:
                raise ValueError(
                    f"Add index {index_name} is not part of the features."
                )

            if index_name in index_dict:
                raise ValueError(
                    f"Add index {index_name} is already part of the index."
                )

            new_indexes.append(
                IndexSchema(name=index_name, dtype=feature_dict[index_name])
            )
        # Note: The new index is added after the existing index items.
        return input.schema.indexes + new_indexes

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ADD_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_to_add",
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def output_feature_schemas(self) -> List[FeatureSchema]:
        return self._output_feature_schemas

    @property
    def output_indexes(self) -> List[IndexSchema]:
        return self._output_indexes

    @property
    def index_to_add(self) -> List[str]:
        return self._index_to_add


operator_lib.register_operator(AddIndexOperator)


def _normalize_indexes_to_add(
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


def add_index(input: Node, indexes: Union[str, List[str]]) -> Node:
    """Adds one or more features as indexes in a node.

    Examples:

        Given an input `Node` with indexes ['A', 'B', 'C'] and features
        ['X', 'Y', 'Z']:

        1. `add_index(input, indexes=['X', 'Y'])`
           Output `Node` will have indexes ['A', 'B', 'C', 'X', 'Y'] and
           features ['Z'].

    Args:
        input: Input `Node` object for which the indexes are to be set or
            updated.
        indexes: List of feature names (strings) that should be added to the
            indexes. These feature names should already exist in `input`.

    Returns:
        New `Node` with the updated index.

    Raises:
        KeyError: If any of the specified `indexes` are not found in `input`.
    """

    indexes = _normalize_indexes_to_add(indexes)
    return AddIndexOperator(input, indexes).outputs["output"]


def set_index(input: Node, indexes: Union[str, List[str]]) -> Node:
    """Replaces the indexes in a node.

    `set_index` is implemented as `drop_index` + `add_index`.

    Examples:

        Given an input `Node` with indexes ['A', 'B', 'C'] and features
        ['X', 'Y', 'Z']:

        1. `set_index(input, indexes=['X'])`
           Output `Node` will have indexes ['X'] and features
           ['A', 'B', 'C', 'Y', 'Z'].

        2. `set_index(input, indexes=['X', 'Y'])`
           Output `Node` will have indexes ['X', 'Y'] and features
           ['A', 'B', 'C', 'Z'].

    Args:
        input: Input `Node` object for which the indexes are to be set.
        indexes: List of index / feature names (strings) used as
            the new indexes. These feature names should be either indexes or
            features in `input`.

    Returns:
        New `Node` with the updated indexes.

    Raises:
        KeyError: If any of the specified `indexes` are not found in `input`.
    """

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
