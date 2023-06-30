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

"""Resample operator class and public API function definition."""

from temporian.core import operator_lib
from temporian.core.data.node import (
    Node,
    create_node_new_features_existing_sampling,
)
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class Resample(Operator):
    def __init__(
        self,
        input: Node,
        sampling: Node,
    ):
        super().__init__()

        self.add_input("input", input)
        self.add_input("sampling", sampling)

        input.schema.check_compatible_index(sampling.schema)

        self.add_output(
            "output",
            create_node_new_features_existing_sampling(
                features=input.schema.features,
                sampling_node=sampling,
                creator=self,
            ),
        )
        self.check()

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="RESAMPLE",
            attributes=[],
            inputs=[
                pb.OperatorDef.Input(key="input"),
                pb.OperatorDef.Input(key="sampling"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(Resample)


def resample(
    input: Node,
    sampling: Node,
) -> Node:
    """Resamples a Node at each timestamp of another Node.

    If a timestamp in `sampling` does not have a corresponding timestamp in
    `input`, the last timestamp in `input` is used instead. If this timestamp
    is anterior to an value in `input`, the value is replaced by
    `dtype.MissingValue(...)`.

    Example:

        ```python
        >>> evset = tp.event_set(
        ...     timestamps=[1, 5, 8, 9],
        ...     features={"f1": [1.0, 2.0, 3.0, 4.0]}
        ... )
        >>> sampling = tp.event_set(timestamps=[-1, 1, 6, 10])
        >>> input_node = evset.node()
        >>> sampling_node = sampling.node()
        >>> out_node = tp.resample(input_node, sampling=sampling_node)
        >>> out_node.run({input_node: evset, sampling_node: sampling})
        indexes: ...
                timestamps: [-1.  1.  6. 10.]
                'f1': [nan  1.  2.  4.]
        ...

        ```

    Args:
        input: Node to sample.
        sampling: Node to use the sampling of.

    Returns:
        Resampled node, with same sampling as `sampling`.
    """

    return Resample(input=input, sampling=sampling).outputs["output"]
