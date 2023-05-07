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

"""Select operator class and public API function definition."""

from typing import List, Union

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class SelectOperator(Operator):
    def __init__(self, input: Node, feature_names: Union[str, List[str]]):
        super().__init__()

        # store selected feature names
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        if not isinstance(feature_names, list):
            raise ValueError(
                "Unexpected type for feature_names. Expect str or list of"
                f" str. Got '{feature_names}' instead."
            )

        self._feature_names = feature_names
        self.add_attribute("feature_names", feature_names)

        # verify all selected features exist in the input node
        selected_features_set = set(feature_names)
        node_features_set = set([feature.name for feature in input.features])
        if not set(selected_features_set).issubset(node_features_set):
            raise KeyError(selected_features_set.difference(node_features_set))

        # inputs
        self.add_input("input", input)

        # outputs
        output_features = []
        for feature_name in feature_names:
            for feature in input.features:
                # TODO: maybe implement features attributes of Node as dict
                # so we can index by name?
                if feature.name == feature_name:
                    output_features.append(feature)

        output_sampling = input.sampling
        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )

        self.check()

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SELECT",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="feature_names",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SelectOperator)


def select(
    input: Node,
    feature_names: List[str],
) -> Node:
    """Selects a subset of features from a node.

    Args:
        input: Node to select features from.
        feature_names: Names of the features to select from the input.

    Returns:
        Node containing only the selected features.
    """
    if isinstance(feature_names, list):
        pass
    elif isinstance(feature_names, str):
        feature_names = [feature_names]
    else:
        raise ValueError(
            "Unexpected type for feature_names. Expect str or list of"
            f" str. Got '{feature_names}' instead."
        )

    return SelectOperator(input, feature_names).outputs["output"]
