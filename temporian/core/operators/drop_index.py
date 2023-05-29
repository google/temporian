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

"""Drop index operator class and public API function definition."""

from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.data.schema import Schema, FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class DropIndexOperator(Operator):
    def __init__(
        self,
        input: Node,
        index_to_drop: List[str],
        keep: bool,
    ) -> None:
        super().__init__()

        # "index_to_drop" is the list of indexes in `input`` to drop. If `keep`
        # is true, those indexes will be converted into features.
        self._index_to_drop = index_to_drop
        self._keep = keep

        self.add_input("input", input)
        self.add_attribute("index_to_drop", index_to_drop)
        self.add_attribute("keep", keep)

        self._output_feature_schemas = self._get_output_feature_schemas(
            input, index_to_drop, keep
        )

        self._output_indexes = [
            index_level
            for index_level in input.schema.indexes
            if index_level.name not in index_to_drop
        ]

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
        self, input: Node, index_to_drop: List[str], keep: bool
    ) -> List[FeatureSchema]:
        if not keep:
            return input.schema.features

        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_features: List[FeatureSchema] = []
        for index_name in index_to_drop:
            if index_name not in index_dict:
                raise ValueError(
                    f"Drop index {index_name} is not part of the index."
                )

            if index_name in feature_dict:
                raise ValueError(
                    f"Feature name {index_name} coming from index already"
                    " exists in input."
                )

            new_features.append(
                FeatureSchema(name=index_name, dtype=index_dict[index_name])
            )

        # Note: The new features are added after the existing features.
        return input.schema.features + new_features

    @property
    def output_feature_schemas(self) -> List[FeatureSchema]:
        return self._output_feature_schemas

    @property
    def output_indexes(self) -> List[IndexSchema]:
        return self._output_indexes

    @property
    def index_to_drop(self) -> List[str]:
        return self._index_to_drop

    @property
    def keep(self) -> bool:
        return self._keep

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="DROP_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_to_drop",
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="keep",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(DropIndexOperator)


def _normalize_index_to_drop(
    input: Node,
    index_names: Optional[Union[List[str], str]],
) -> List[str]:
    if index_names is None:
        # Drop all the indexes
        return input.schema.index_names()

    if isinstance(index_names, str):
        index_names = [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    # Check that requested indexes exist
    index_dict = input.schema.index_name_to_dtype()
    for index_item in index_names:
        if index_item not in index_dict:
            raise KeyError(
                f"Dropped index {index_item} is not part of the index"
                f" {input.schema.indexes}."
            )

    return index_names


def drop_index(
    input: Node,
    index_to_drop: Optional[Union[str, List[str]]] = None,
    keep: bool = True,
) -> Node:
    """Removes one or more index columns from a node.

    Examples:
        Given an input `Node` with index names ['A', 'B', 'C'] and features
        names ['X', 'Y', 'Z']:

        1. `drop_index(input, index_names='A', keep=True)`
           Output `Node` will have index names ['B', 'C'] and features names
           ['X', 'Y', 'Z', 'A'].

        2. `drop_index(input, index_names=['A', 'B'], keep=False)`
           Output `Node` will have index names ['C'] and features names
           ['X', 'Y', 'Z'].

        3. `drop_index(input, index_names=None, keep=True)`
           Output `Node` will have index names [] (empty index) and features
           names ['X', 'Y', 'Z', 'A', 'B', 'C'].

    Args:
        input: Node from which the specified index columns should be removed.
        index_to_drop: Index column(s) to be removed from `input`. This can be a
            single column name (`str`) or a list of column names (`List[str]`).
            If not specified or set to `None`, all index columns in `input` will
            be removed. Defaults to `None`.
        keep: Flag indicating whether the removed index columns should be kept
            as features in the output `Node`. Defaults to `True`.

    Returns:
        A new `Node` object with the updated index, where the specified index
        column(s) have been removed. If `keep` is set to `True`, the removed
        index columns will be included as features in the output `Node`.

    Raises:
        ValueError: If an empty list is provided as the `index_names` argument.
        KeyError: If any of the specified `index_names` are missing from
            `input`'s index.
        ValueError: If a feature name coming from the index already exists in
            `input`, and the `keep` flag is set to `True`.
    """
    index_to_drop = _normalize_index_to_drop(input, index_to_drop)
    return DropIndexOperator(input, index_to_drop, keep).outputs["output"]
