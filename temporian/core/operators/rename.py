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
from typing import List, Union, Dict

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.data.sampling import IndexLevel
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class RenameOperator(Operator):
    """Rename operator."""

    def __init__(
        self,
        input: Node,
        features: Union[str, Dict[str, str]] = None,
        index: Union[str, Dict[str, str]] = None,
    ):
        def check_rename_dict(
            rename_dict: Dict[str, str],
            node_names: List[str],
            dict_name: str = "features",
        ) -> Dict[str, str]:
            # check that there are no duplicate values in rename_dict values
            if len(set(rename_dict.values())) != len(rename_dict.values()):
                raise ValueError(
                    f"Duplicate values in {dict_name}. Values must be unique."
                )

            # check that every value is a string
            if not all(
                isinstance(value, str) for value in rename_dict.values()
            ):
                raise ValueError(
                    f"Expected all values in {dict_name} to be strings."
                )

            # check that every dict value is non empty
            if not all(value for value in rename_dict.values()):
                raise ValueError(
                    f"Expected all values in {dict_name} to be non-empty."
                )

            # check that every key is in event_names
            for key in rename_dict.keys():
                if key not in node_names:
                    raise KeyError(
                        f"Key '{key}' not found in node. Possible names:"
                        f" {node_names}."
                    )

            return rename_dict

        super().__init__()

        self.features = {}

        if isinstance(features, str):
            # check that node only has one feature
            if len(input.features) != 1:
                raise ValueError(
                    "Expected input to have only one feature when passed a"
                    f" single string. Got {len(input.features)} features"
                    " instead."
                )
            if not features:
                raise ValueError("Expected feature to be a non-empty string.")
            only_feature_name = input.features[0].name
            self.features = {only_feature_name: features}

        elif isinstance(features, dict):
            # check that every key is a feature name in node
            feature_names = [feature.name for feature in input.features]
            self.features = check_rename_dict(features, feature_names)

        node_index_names = input.index_names

        self.index = {}

        if isinstance(index, str):
            # check that node only has one index
            if len(node_index_names) != 1:
                raise ValueError(
                    "Expected node to have only one index when passed a"
                    " single string in index. Got"
                    f" {len(node_index_names)} indexes instead."
                )
            # check index is non empty
            if not index:
                raise ValueError("Expected index to be a non-empty string.")
            only_index_name = node_index_names[0]
            self.index = {only_index_name: index}

        elif isinstance(index, dict):
            # check that every key is an index name in node
            self.index = check_rename_dict(index, node_index_names, "index")

        self.add_attribute("features", self.features)

        self.add_attribute("index", self.index)

        # inputs
        self.add_input("input", input)

        output_sampling = input.sampling

        if index:
            output_sampling = self.new_sampling(input.sampling)

        # outputs
        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=self.features.get(f.name, f.name),
                dtype=f.dtype,
                sampling=output_sampling,
                creator=self,
            )
            for f in input.features
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )

        self.check()

    def new_sampling(self, old_sampling: Sampling) -> Sampling:
        new_index_levels = []
        for level in old_sampling.index.levels:
            new_name = level.name
            if level.name in self.index:
                new_name = self.index[level.name]

            new_index_levels.append(
                IndexLevel(name=new_name, dtype=level.dtype)
            )

        return Sampling(
            index_levels=new_index_levels,
            creator=self,
            is_unix_timestamp=old_sampling.is_unix_timestamp,
        )

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="RENAME",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="features",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                    is_optional=False,
                ),
                pb.OperatorDef.Attribute(
                    key="index",
                    type=pb.OperatorDef.Attribute.Type.MAP_STR_STR,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(RenameOperator)


def rename(
    input: Node,
    features: Union[str, Dict[str, str]] = None,
    index: Union[str, Dict[str, str]] = None,
) -> Node:
    """Renames a node's features and index.

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
        Event with renamed features and index.
    """
    return RenameOperator(input, features, index).outputs["output"]
