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

from typing import Dict, List

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb

# Maximum number of arguments taken by the glue operator
MAX_NUM_ARGUMENTS = 30


class GlueOperator(Operator):
    def __init__(
        self,
        **nodes_dict: Dict[str, Node],
    ):
        super().__init__()

        # Note: Support for dictionaries of nodes is required for
        # serialization.

        if len(nodes_dict) < 2:
            raise ValueError("At least two arguments should be provided")

        if len(nodes_dict) >= MAX_NUM_ARGUMENTS:
            raise ValueError(
                f"Too many (>{MAX_NUM_ARGUMENTS}) arguments provided"
            )

        # inputs
        output_features = []
        feature_names = set()
        first_sampling = None
        for key, node in nodes_dict.items():
            self.add_input(key, node)
            output_features.extend(node.features)

            for f in node.features:
                if f.name in feature_names:
                    raise ValueError(
                        f'Feature "{f.name}" is defined in multiple input'
                        " nodes."
                    )
                feature_names.add(f.name)

            if first_sampling is None:
                first_sampling = node.sampling
            elif node.sampling is not first_sampling:
                raise ValueError(
                    "All glue arguments should have the same sampling."
                    f" {first_sampling} is different from {node.sampling}."
                )

        # outputs
        self.add_output(
            "node",
            Node(
                features=output_features,
                sampling=first_sampling,
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
                pb.OperatorDef.Input(key=f"node_{idx}", is_optional=idx >= 2)
                for idx in range(MAX_NUM_ARGUMENTS)
            ],
            outputs=[pb.OperatorDef.Output(key="node")],
        )


operator_lib.register_operator(GlueOperator)


def glue(
    *nodes: List[Node],
) -> Node:
    """Concatenates together nodes with the same sampling.

    Example:

        ```
        node_1 = ... # Feature A & B
        node_2 = ... # Feature C & D
        node_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F
        output = np.glue(node_1, node_2, node_3)
        ```

    To concatenate nodes with a different sampling, use the operator
    'tp.sample(...)' first.

    Example:

        ```
        # Assume node_1, node_2 and node_3 dont have the same sampling
        node_1 = ... # Feature A & B
        node_2 = ... # Feature C & D
        node_3 = ... # Feature E & F

        # Output has features A, B, C, D, E & F, and the same sampling as
        # node_1
        output = tp.glue(node_1,
            tp.sample(node_2, sampling=node_1),
            tp.sample(node_3, sampling=node_1))
        ```

    Args:
        *nodes: Nodes to concatenate.

    Returns:
        The concatenated nodes.
    """
    if len(nodes) == 1:
        return nodes[0]

    # Note: The node should be called "node_{idx}" with idx in [0, MAX_NUM_ARGUMENTS).
    nodes_dict = {f"node_{idx}": node for idx, node in enumerate(nodes)}
    return GlueOperator(**nodes_dict).outputs["node"]
