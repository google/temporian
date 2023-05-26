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

"""Sample operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.node import Node
from temporian.core.data.schema import Schema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Sample(Operator):
    def __init__(
        self,
        input: Node,
        sampling: Node,
    ):
        super().__init__()

        self.add_input("input", input)
        self.add_input("sampling", sampling)

        if input.sampling.index != sampling.sampling.index:
            raise ValueError(
                "Node and sampling do not have the same index."
                f" {input.sampling.index} != {sampling.sampling.index}"
            )

        output_features = [  # pylint: disable=g-complex-comprehension
            Feature(
                name=f.name,
                dtype=f.dtype,
                sampling=sampling.sampling,
                creator=self,
            )
            for f in input.features
        ]

        self.add_output(
            "output",
            Node(
                features=output_features,
                sampling=sampling.sampling,
                creator=self,
            ),
        )

        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SAMPLE",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Sample)


def sample(
    input: Node,
    sampling: Node,
) -> Node:
    """Samples a node at each timestamp of a sampling.

    If a timestamp in `sampling` does not have a corresponding timestamp in
    `input`, the last timestamp in `input` is used instead. If this timestamp
    is anterior to an value in `input`, the value is replaced by
    `dtype.MissingValue(...)`.

    Example:
        ```
        Inputs:
            input:
                timestamps: 1, 5, 8, 9
                feature_1:  1.0, 2.0, 3.0, 4.0
            sampling:
                timestamps: -1, 1, 6, 10

        Output:
            timestamps: -1, 1, 6, 10
            feature_1: nan, 1.0, 2.0, 4.0
        ```

    Args:
        input: Node to sample.
        sampling: Node to use the sampling of.

    Returns:
        Sampled node, with same sampling as `sampling`.
    """

    return Sample(input=input, sampling=sampling).outputs["output"]
