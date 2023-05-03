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
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class SetIndexOperator(Operator):
    def __init__(
        self,
        input: Node,
        feature_names: Optional[Union[List[str], str]] = None,
        append: bool = False,
    ) -> None:
        super().__init__()

        # process feature_names
        feature_names = self._process_feature_names(input, feature_names)

        # input node
        self.add_input("input", input)

        # attributes
        self.add_attribute("feature_names", feature_names)
        self.add_attribute("append", append)

        # output features
        output_features = self._generate_output_features(input, feature_names)

        # output sampling
        output_sampling = Sampling(
            index_levels=[
                (index_name, index_dtype)
                for index_name, index_dtype in input.sampling.index
            ]
            + [
                (index_name, input.dtypes[index_name])
                for index_name in feature_names
            ]
            if append
            else [
                (index_name, input.dtypes[index_name])
                for index_name in feature_names
            ],
            is_unix_timestamp=node.sampling.is_unix_timestamp,
        )
        # output node
        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    def _process_feature_names(
        self,
        input: Node,
        feature_names: Optional[Union[List[str], str]],
    ) -> List[str]:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        missing_feature_names = [
            label
            for label in feature_names
            if label not in [feature.name for feature in input.features]
        ]
        if missing_feature_names:
            raise KeyError(missing_feature_names)

        return feature_names

    def _generate_output_features(
        self, input: Node, feature_names: List[str]
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in input.features
            if feature.name not in feature_names
        ]
        return output_features

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SET_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="feature_names",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="append",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SetIndexOperator)


def set_index(
    input: Node, feature_names: List[str], append: bool = False
) -> Node:
    """Sets one or more features as index in a node.

    Optionally, the new index columns can be appended to the existing index.

    The input `input` object remains unchanged. The function returns a new
    node with the specified index changes.

    Examples:
        Given an input `Node` with index names ['A', 'B', 'C'] and features
        names ['X', 'Y', 'Z']:

        1. `set_index(input, feature_names=['X'], append=False)`
           Output `Node` will have index names ['X'] and features names
           ['Y', 'Z'].

        2. `set_index(input, feature_names=['X', 'Y'], append=False)`
           Output `Node` will have index names ['X', 'Y'] and
           features names ['Z'].

        3. `set_index(input, feature_names=['X', 'Y'], append=True)`
           Output `Node` will have index names ['A', 'B', 'C', 'X', 'Y'] and
           features names ['Z'].

    Args:
        input: Input `Node` object for which the index is to be set or
            updated.
        feature_names: List of feature names (strings) that should be used as
            the new index. These feature names should already exist in `input`.
        append: Flag indicating whether the new index should be appended to the
            existing index (True) or replace it (False). Defaults to `False`.

    Returns:
        New `Node` with the updated index, where the specified feature names
        have been set or appended as index columns, based on the `append` flag.

    Raises:
        KeyError: If any of the specified `feature_names` are not found in
            `input`.
    """
    return SetIndexOperator(input, feature_names, append).outputs["output"]
