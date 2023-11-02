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
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_with_new_reference,
)
from temporian.core.data.schema import Schema
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.typecheck import typecheck

# Maximum number of arguments taken by the glue operator
MAX_NUM_ARGUMENTS = 100


class GlueOperator(Operator):
    def __init__(
        self,
        **inputs: EventSetNode,
    ):
        super().__init__()

        # Note: Support for dictionaries of nodes is required for
        # serialization.

        if len(inputs) < 2:
            raise ValueError("At least two arguments should be provided.")

        if len(inputs) >= MAX_NUM_ARGUMENTS:
            raise ValueError(
                f"Too many (>{MAX_NUM_ARGUMENTS}) arguments provided."
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
                        " EventSetNodes to glue. Consider using prefix() or"
                        " rename()."
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


@typecheck
@compile
def glue(
    *inputs: EventSetOrNode,
) -> EventSetOrNode:
    """Concatenates features from [`EventSets`][temporian.EventSet] with the
    same sampling.

    Feature names cannot be duplicated across EventSets.

    See the examples below for workarounds on gluing EventSets with duplicated
    feature names or different samplings.

    Example:

        ```python
        >>> a = tp.event_set(
        ...     timestamps=[0, 1, 5],
        ...     features={"M": [0, 10, 50], "N": [50, 100, 500]},
        ... )
        >>> b = a["M"] + a["N"]
        >>> c = a["M"] - a["N"]

        # Glue all features from a,b,c
        >>> d = tp.glue(a, b, c)
        >>> d
        indexes: []
        features: [('M', int64), ('N', int64), ('add_M_N', int64), ('sub_M_N', int64)]
        events:
            (3 events):
                timestamps: [0. 1. 5.]
                'M': [ 0 10 50]
                'N': [ 50 100 500]
                'add_M_N': [ 50 110 550]
                'sub_M_N': [ -50  -90 -450]
        ...

        ```

    To glue EventSets with duplicated feature names, add a prefix or rename them
    before.

    Example with duplicated names:

        ```python
        >>> a = tp.event_set(
        ...     timestamps=[0, 1, 5],
        ...     features={"M": [0, 10, 50], "N": [50, 100, 500]},
        ... )

        # Same feature names as a
        >>> b = 3 * a

        # Add a prefix before glue
        >>> output = tp.glue(a, b.prefix("3x"))
        >>> output.schema.features
        [('M', int64), ('N', int64), ('3xM', int64), ('3xN', int64)]

        # Or rename before glue
        >>> output = tp.glue(a["M"], b["M"].rename("M_new"))
        >>> output.schema.features
        [('M', int64), ('M_new', int64)]

        ```

    To concatenate EventSets with different samplings, use
    [`EventSet.resample()`][temporian.EventSet.resample] first.

    Example with different samplings:

        ```python
        >>> a = tp.event_set(timestamps=[0, 2], features={"A": [0, 20]})
        >>> b = tp.event_set(timestamps=[0, 2], features={"B": [1, 21]})
        >>> c = tp.event_set(timestamps=[1, 4], features={"C": [10, 40]})
        >>> output = tp.glue(a, b.resample(a), c.resample(a))
        >>> output
        indexes: []
        features: [('A', int64), ('B', int64), ('C', int64)]
        events:
            (2 events):
                timestamps: [0. 2.]
                'A': [ 0 20]
                'B': [ 1 21]
                'C': [ 0 10]
        ...

        ```

    Args:
        *inputs: EventSets to concatenate the features of.

    Returns:
        EventSet with concatenated features.
    """
    if len(inputs) == 1 and isinstance(inputs[0], EventSetNode):
        return inputs[0]

    # Note: The node should be called "input_{idx}" with idx in [0, MAX_NUM_ARGUMENTS).
    inputs_dict = {f"input_{idx}": input for idx, input in enumerate(inputs)}

    return GlueOperator(**inputs_dict).outputs["output"]  # type: ignore
