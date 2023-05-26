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

from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.data.schema import Schema, FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


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
            Node.create_new_features_new_sampling(
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
        # Note: The new index is added before the existing index items.
        return new_indexes + input.schema.indexes

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ADD_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_to_add",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
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


def _normalize_index_to_add(
    index_names: str | List[str],
) -> List[str]:
    if isinstance(index_names, str):
        return [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    return index_names


# Replace with Add index
def add_index(input: Node, index_to_add: str | List[str]) -> Node:
    """Adds one or more features as index in a node.

    Optionally, the new index columns can be appended to the existing index.

    The input `input` object remains unchanged. The function returns a new
    node with the specified index changes.

    Examples:
        Given an input `Node` with index names ['A', 'B', 'C'] and features
        names ['X', 'Y', 'Z']:

        1. `add_index(input, feature_names=['X'], append=False)`
           Output `Node` will have index names ['X'] and features names
           ['Y', 'Z'].

        2. `add_index(input, feature_names=['X', 'Y'], append=False)`
           Output `Node` will have index names ['X', 'Y'] and
           features names ['Z'].

        3. `add_index(input, feature_names=['X', 'Y'], append=True)`
           Output `Node` will have index names ['A', 'B', 'C', 'X', 'Y'] and
           features names ['Z'].

    Args:
        input: Input `Node` object for which the index is to be set or
            updated.
        index_to_add: List of feature names (strings) that should be used as
            the new index. These feature names should already exist in `input`.

    Returns:
        New `Node` with the updated index, where the specified feature names
        have been set or appended as index columns, based on the `append` flag.

    Raises:
        KeyError: If any of the specified `index_to_add` are not found in
            `input`.
    """

    index_to_add = _normalize_index_to_add(index_to_add)
    return AddIndexOperator(input, index_to_add).outputs["output"]
