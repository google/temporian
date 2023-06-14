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

"""Glue operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.node import Node, create_node_with_new_reference
from temporian.core.data.schema import Schema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb

# Maximum number of arguments taken by the glue operator
MAX_NUM_ARGUMENTS = 30


class GlueOperator(Operator):
    def __init__(
        self,
        **inputs: Node,
    ):
        super().__init__()

        # Note: Support for dictionaries of nodes is required for
        # serialization.

        if len(inputs) < 2:
            raise ValueError("At least two arguments should be provided")

        if len(inputs) >= MAX_NUM_ARGUMENTS:
            raise ValueError(
                f"Too many (>{MAX_NUM_ARGUMENTS}) arguments provided"
            )

        # inputs
        output_features = []
        output_feature_schemas = []
        feature_names = set()
        first_sampling_node = None

        for key, input in inputs.items():
            self.add_input(key, input)

            output_features.extend(input.feature_nodes)
            output_feature_schemas.extend(input.schema.features)

            for f in input.schema.features:
                if f.name in feature_names:
                    raise ValueError(
                        f'Feature "{f.name}" is defined in multiple input'
                        " nodes to glue. Consider using prefix() or rename()."
                    )
                feature_names.add(f.name)

            if first_sampling_node is None:
                first_sampling_node = input
            else:
                input.check_same_sampling(first_sampling_node)

        assert first_sampling_node is not None

        self.add_output(
            "output",
            create_node_with_new_reference(
                schema=Schema(
                    features=output_feature_schemas,
                    indexes=first_sampling_node.schema.indexes,
                    is_unix_timestamp=first_sampling_node.schema.is_unix_timestamp,
                ),
                sampling=first_sampling_node.sampling_node,
                features=output_features,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="GLUE",
            # TODO: Add support to array of nodes arguments.
            inputs=[
                pb.OperatorDef.Input(key=f"input_{idx}", is_optional=idx >= 2)
                for idx in range(MAX_NUM_ARGUMENTS)
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(GlueOperator)


def glue(
    *inputs: Node,
) -> Node:
    """Concatenates together nodes with the same sampling. Feature names cannot be duplicated across nodes.

    Example:

        ```python
        >>> a = tp.input_node(features=[("A1", tp.float64), ("A2", tp.float64)])
        >>> b = a["A1"] + a["A2"]
        >>> c = a["A1"] - a["A2"]

        # Glue all features from a,b,c
        >>> d = tp.glue(a, b, c)
        >>> d.features
        [('A1', float64), ('A2', float64), ('add_A1_A2', float64), ('sub_A1_A2', float64)]

        ```

    To glue nodes with duplicated feature names, add a prefix or rename before:

    Example:

    ```python
        >>> a = tp.input_node(features=[("f1", tp.float64), ("f2", tp.float64)])

        # Same feature names as a
        >>> b = tp.simple_moving_average(a, 5)

        # Add a prefix before glue
        >>> output = tp.glue(a, tp.prefix("sma_", b))
        >>> output.features
        [('f1', float64), ('f2', float64), ('sma_f1', float64), ('sma_f2', float64)]

        # Or rename before glue
        >>> output = tp.glue(a["f1"], tp.rename(b["f1"], "new_feature"))
        >>> output.features
        [('f1', float64), ('new_feature', float64)]

        ```

    To concatenate nodes with different samplings, use
    [`tp.resample()`][temporian.resample] first.

    Example:

        ```python
        >>> a = tp.input_node(features=[("A", tp.str_)])
        >>> b = tp.input_node(features=[("B", tp.float64)])
        >>> c = tp.input_node(features=[("C", tp.float64)])
        >>> output = tp.glue(a,
        ...                  tp.resample(b, sampling=a),
        ...                  tp.resample(c, sampling=a)
        ...          )
        >>> output.features
        [('A', str_), ('B', float64), ('C', float64)]

        ```

    Args:
        *inputs: Nodes to concatenate.

    Returns:
        Concatenated nodes.
    """
    if len(inputs) == 1:
        return inputs[0]

    # Note: The node should be called "input_{idx}" with idx in [0, MAX_NUM_ARGUMENTS).
    inputs_dict = {f"input_{idx}": input for idx, input in enumerate(inputs)}
    return GlueOperator(**inputs_dict).outputs["output"]
