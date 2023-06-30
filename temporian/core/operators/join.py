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


"""Join operator class and public API function definitions."""

from typing import Optional

from temporian.core.data.dtype import DType
from temporian.core import operator_lib
from temporian.core.data.node import (
    Node,
    create_node_with_new_reference,
    Feature,
)
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.data.schema import Schema

JOIN_LEFT = "left"


class Join(Operator):
    def __init__(
        self,
        input_1: Node,
        input_2: Node,
        how: str = "left",
        on: Optional[str] = None,
    ):
        super().__init__()

        self.add_input("input_1", input_1)
        self.add_input("input_2", input_2)
        self.add_attribute("how", how)
        if on is not None:
            self.add_attribute("on", on)

        input_1.schema.check_compatible_index(
            input_2.schema, "input_1 and input_2"
        )

        if how not in [JOIN_LEFT]:
            raise ValueError(
                f"Non supported join type {how}. Supported join type(s) are:"
                f" {JOIN_LEFT}"
            )

        if on is not None:
            for node, node_name in [(input_1, "input_1"), (input_2, "input_2")]:
                feature_names = node.schema.feature_names()
                if on not in feature_names:
                    raise ValueError(
                        f'Feature "{on}" does not exist in {node_name}'
                    )
                on_dtype = node.schema.features[feature_names.index(on)].dtype
                if on_dtype != DType.INT64:
                    raise ValueError(
                        '"on" feature should be of type int64. Got'
                        f" {on_dtype} instead for {node_name}."
                    )

        output_features = []
        output_feature_schemas = []
        output_features.extend(input_1.feature_nodes)
        output_feature_schemas.extend(input_1.schema.features)

        input_1_feature_names = input_1.schema.feature_names()
        for i2_feature in input_2.schema.features:
            if on is not None and i2_feature.name == on:
                continue
            output_features.append(Feature(creator=self))
            output_feature_schemas.append(i2_feature)
            if i2_feature.name in input_1_feature_names:
                raise ValueError(
                    f'Feature "{i2_feature.name}" is defined in both inputs'
                )

        self.add_output(
            "output",
            create_node_with_new_reference(
                schema=Schema(
                    features=output_feature_schemas,
                    indexes=input_1.schema.indexes,
                    is_unix_timestamp=input_1.schema.is_unix_timestamp,
                ),
                sampling=input_1.sampling_node,
                features=output_features,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="JOIN",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="how",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="on",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=True,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="input_1"),
                pb.OperatorDef.Input(key="input_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Join)


def join(
    input_1: Node,
    input_2: Node,
    how: str = "left",
    on: Optional[str] = None,
) -> Node:
    """Join [`Nodes`][temporian.Node] with different samplings.

    Join features from two nodes based on timestamps. Optionally, join on
    timestamps and an extra in64 feature. Joined nodes should have the the same
    index and non overlapping feature names.

    To concatenates nodes with the same sampling, use
    [`tp.glue`][temporian.glue] instead. [`tp.glue`][temporian.glue] is almost
    free while [`tp.join`][temporian.join] can be expensive.
    To resample a node according to another nodes's sampling,
    [`tp.resample`][temporian.resample] instead.

    Example:

        ```python
        >>> a = tp.input_node(features=[("f1", tp.float64)])
        >>> b = tp.input_node(features=[("f2", tp.float64)])
        >>> c = tp.join(a, b)
        >>> c.features
        [('f1', float64), ('f2', float64)]

        ```

    Args:
        input_1: Left node to join.
        input_2: Right node to join.
        how: Should this be a "left", "inner", or "outer" join. Currently, only
            "left" join is supported.
    """

    if input_1.sampling_node is input_2.sampling_node:
        raise ValueError(
            "Both inputs have the same sampling. Use tp.glue instead of"
            " tp.join."
        )

    return Join(
        input_1=input_1,
        input_2=input_2,
        how=how,
        on=on,
    ).outputs["output"]
