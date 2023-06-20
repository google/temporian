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

"""Rename operator."""
from typing import Dict, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.node import (
    Node,
    create_node_new_features_new_sampling,
    create_node_new_features_existing_sampling,
)
from temporian.core.data.schema import Schema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class RenameOperator(Operator):
    """Rename operator."""

    def __init__(
        self,
        input: Node,
        features: Dict[str, str],
        index: Dict[str, str],
    ):
        super().__init__()

        self.add_attribute("features", features)
        self.add_attribute("index", index)

        self._features = features
        self._index = index

        self.add_input("input", input)

        new_indexes = [
            (index.get(i.name, i.name), i.dtype) for i in input.schema.indexes
        ]
        new_feature_schemas = [
            (features.get(f.name, f.name), f.dtype)
            for f in input.schema.features
        ]

        if index:
            self.add_output(
                "output",
                create_node_new_features_new_sampling(
                    features=new_feature_schemas,
                    indexes=new_indexes,
                    is_unix_timestamp=input.schema.is_unix_timestamp,
                    creator=self,
                ),
            )

        else:
            self.add_output(
                "output",
                create_node_new_features_existing_sampling(
                    features=new_feature_schemas,
                    sampling_node=input,
                    creator=self,
                ),
            )
        self.check()

    @property
    def features(self) -> Dict[str, str]:
        return self._features

    @property
    def index(self) -> Dict[str, str]:
        return self._index

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="RENAME",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="features",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                ),
                pb.OperatorDef.Attribute(
                    key="index",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(RenameOperator)


def _normalize_rename_features(
    schema: Schema,
    features: Optional[Union[str, Dict[str, str]]],
) -> Dict[str, str]:
    if features is None:
        return {}

    if isinstance(features, str):
        if len(schema.features) != 1:
            raise ValueError(
                "Cannot apply rename operator with a single rename string when "
                "the EventSet contains multiple features. Pass a dictionary "
                "of rename strings instead."
            )
        features = {schema.features[0].name: features}

    feature_dict = schema.feature_name_to_dtype()

    if len(features) != len(set(features.values())):
        raise ValueError("Multiple features renamed to the same name")

    for src, dst in features.items():
        if dst == "":
            raise ValueError("Cannot rename to an empty string")
        if src not in feature_dict:
            raise KeyError(f"The feature {src!r} does not exist.")
    return features


def _normalize_rename_indexes(
    schema: Schema,
    indexes: Optional[Union[str, Dict[str, str]]],
) -> Dict[str, str]:
    if indexes is None:
        return {}

    if isinstance(indexes, str):
        if len(schema.indexes) != 1:
            raise ValueError(
                "Cannot apply rename operator with a single rename string when "
                "the EventSet contains multiple indexes. Pass a dictionary "
                "of rename strings instead."
            )
        indexes = {schema.indexes[0].name: indexes}

    indexes_dict = schema.index_name_to_dtype()

    if len(indexes) != len(set(indexes.values())):
        raise ValueError("Multiple indexes renamed to the same name")

    for src, dst in indexes.items():
        if dst == "":
            raise ValueError("Cannot rename to an empty string")
        if src not in indexes_dict:
            raise KeyError(f"The index {src!r} does not exist.")
    return indexes


def rename(
    input: Node,
    features: Optional[Union[str, Dict[str, str]]] = None,
    index: Optional[Union[str, Dict[str, str]]] = None,
) -> Node:
    """Renames a Node's features and index.

    If the input has a single feature, then the `features` can be a
    single string with the new name.

    If the input has multiple features, then `features` must be a mapping
    with the old names as keys and the new names as values.

    The index renaming follows the same criteria, accepting a single string or
    a mapping for multiple index levels.

    Args:
        input: Node to rename.

        features: New feature name or mapping from old names to new names.
        index: New index name or mapping from old names to new names.

    Returns:
        Node with renamed features and index.
    """

    features = _normalize_rename_features(input.schema, features)
    index = _normalize_rename_indexes(input.schema, index)

    return RenameOperator(input, features, index).outputs["output"]
