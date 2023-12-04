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
from typing import Dict, Optional, Union, List

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
    create_node_new_features_existing_sampling,
)
from temporian.core.data.schema import Schema
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck


class RenameOperator(Operator):
    """Rename operator."""

    def __init__(
        self,
        input: EventSetNode,
        features: Dict[str, str],
        indexes: Dict[str, str],
    ):
        super().__init__()

        self.add_attribute("features", features)
        self.add_attribute("indexes", indexes)

        self._features = features
        self._indexes = indexes

        self.add_input("input", input)

        new_indexes = [
            (indexes.get(i.name, i.name), i.dtype) for i in input.schema.indexes
        ]
        new_feature_schemas = [
            (features.get(f.name, f.name), f.dtype)
            for f in input.schema.features
        ]

        if indexes:
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
        return self._indexes

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
                    key="indexes",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(RenameOperator)


def _normalize_rename_features(
    schema: Schema,
    features: Optional[Union[str, Dict[str, str], List[str]]],
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
    elif isinstance(features, list):
        if len(features) != len(schema.features):
            raise ValueError(
                f"The provided {len(features)} new names ({features!r}) don't"
                f" match the number of features ({len(schema.features)}) in the"
                " event-set."
            )
        features = {
            old_feature.name: new_name
            for old_feature, new_name in zip(schema.features, features)
        }
    elif not isinstance(features, dict):
        raise ValueError(
            "The rename argument should be string, a map of string to string,"
            f" or a list of string. Got {features} instead."
        )

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
    indexes: Optional[Union[str, Dict[str, str], List[str]]],
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
    elif isinstance(indexes, list):
        if len(indexes) != len(schema.indexes):
            raise ValueError(
                f"The provided {len(indexes)} new names ({indexes!r}) don't"
                f" match the number of indexes ({len(schema.indexes)}) in the"
                " event-set."
            )
        indexes = {
            old_index.name: new_name
            for old_index, new_name in zip(schema.indexes, indexes)
        }
    elif not isinstance(indexes, dict):
        raise ValueError(
            "The rename argument should be string, a map of string to string,"
            f" or a list of string. Got {indexes} instead."
        )

    indexes_dict = schema.index_name_to_dtype()

    if len(indexes) != len(set(indexes.values())):
        raise ValueError("Multiple indexes renamed to the same name")

    for src, dst in indexes.items():
        if dst == "":
            raise ValueError("Cannot rename to an empty string")
        if src not in indexes_dict:
            raise KeyError(f"The index {src!r} does not exist.")
    return indexes


@typecheck
@compile
def rename(
    input: EventSetOrNode,
    features: Optional[Union[str, Dict[str, str], List[str]]] = None,
    indexes: Optional[Union[str, Dict[str, str], List[str]]] = None,
) -> EventSetOrNode:
    assert isinstance(input, EventSetNode)

    features = _normalize_rename_features(input.schema, features)
    indexes = _normalize_rename_indexes(input.schema, indexes)

    return RenameOperator(input, features, indexes).outputs["output"]
