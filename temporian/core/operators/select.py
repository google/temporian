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
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_with_new_reference,
)
from temporian.core.operators.base import Operator
from temporian.core.data.schema import Schema
from temporian.proto import core_pb2 as pb


class SelectOperator(Operator):
    def __init__(self, input: EventSetNode, feature_names: List[str]):
        super().__init__()

        self._feature_names = feature_names
        self.add_attribute("feature_names", feature_names)
        self.add_input("input", input)

        # outputs
        output_features = []
        output_feature_schemas = []
        input_feature_names = input.schema.feature_names()

        for feature_name in feature_names:
            if feature_name not in input_feature_names:
                raise IndexError(
                    f"Selected features {feature_name!r} is not part of the"
                    f" available features {input_feature_names!r}."
                )
            feature_idx = input_feature_names.index(feature_name)
            output_features.append(input.feature_nodes[feature_idx])
            output_feature_schemas.append(input.schema.features[feature_idx])

        self.add_output(
            "output",
            create_node_with_new_reference(
                schema=Schema(
                    features=output_feature_schemas,
                    indexes=input.schema.indexes,
                    is_unix_timestamp=input.schema.is_unix_timestamp,
                ),
                sampling=input.sampling_node,
                features=output_features,
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
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                    is_optional=False,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(SelectOperator)


@compile
def select(
    input: EventSetNode,
    feature_names: Union[str, List[str]],
) -> EventSetNode:
    """Selects a subset of features from an EventSetNode.

    Args:
        input: EventSetNode to select features from.
        feature_names: Name or list of names of the features to select from the
            input.

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2],
        ...     features={"A": [1, 2], "B": ['s', 'm'], "C": [5.0, 5.5]},
        ... )

        >>> # Select single feature
        >>> b = tp.select(a, 'B')
        >>> # Equivalent
        >>> b = a['B']
        >>> b
        indexes: []
        features: [('B', str_)]
        events:
            (2 events):
                timestamps: [1. 2.]
                'B': [b's' b'm']
        ...

        >>> # Select multiple features
        >>> bc = tp.select(a, ['B', 'C'])
        >>> # Equivalent
        >>> bc = a[['B', 'C']]
        >>> bc
        indexes: []
        features: [('B', str_), ('C', float64)]
        events:
            (2 events):
                timestamps: [1. 2.]
                'B': [b's' b'm']
                'C': [5.  5.5]
        ...

        ```


    Returns:
        EventSetNode containing only the selected features.
    """
    if isinstance(feature_names, list) and all(
        isinstance(f, str) for f in feature_names
    ):
        pass
    elif isinstance(feature_names, str):
        feature_names = [feature_names]
    else:
        raise TypeError(
            "Unexpected type for feature_names. Expect str or list of"
            f" str. Got '{feature_names}' instead."
        )

    return SelectOperator(input, feature_names).outputs["output"]
